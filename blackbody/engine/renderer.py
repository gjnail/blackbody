"""Volume renderer and compositor.

Per frame: a half-resolution light volume (fire light, key-light shadows, sky occlusion,
occupancy), a ray march into beauty / emission / aux buffers, embers on top, a bloom pyramid,
and a composite over the footage with heat haze and fire light cast onto the scene.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import camera as cam
from .gpu import GPU, Uniforms, TU
from .lut import BB_MAX_K, BB_MIN_K, BB_SIZE, blackbody_lut, detail_noise_volume

BB_DEFINES = {'BB_MIN': float(BB_MIN_K), 'BB_MAX': float(BB_MAX_K)}

LIGHT_BLOCK = 8  # light-volume cells per side of each point light standing in for the fire
COLOURANT_RADIANCE = 25.0  # radiance per metre of one unit of flame colourant at gain 1

VIEW_MODES = {'composite': 0, 'fire': 1, 'alpha': 2, 'emission': 3, 'heat': 4, 'depth': 5, 'temperature': 6}
VIEW_TRANSFORMS = {'standard': 0, 'agx': 1, 'aces': 2, 'raw': 3}
INPUT_TRANSFORMS = {'srgb': 0, 'rec709': 1, 'linear': 2, 'acescg': 3}


@dataclass
class LookParams:
    ambient_k: float = 300.0
    flame_k: float = 1650.0
    max_k: float = 2250.0
    dynamic_range: float = 0.75
    intensity: float = 1.0
    exposure: float = 0.0          # EV
    flame_density: float = 1.0
    flame_sharpness: float = 1.6
    flame_threshold: float = 0.02
    flame_absorption: float = 4.0  # 1/m per unit flame: how quickly flame becomes optically thick
    flame_occlusion: float = 0.35  # how much flame hides the background (alpha)
    soot_glow: float = 0.25
    blue: float = 0.0
    blue_color: tuple = (0.12, 0.30, 1.0)
    ignition: float = 0.25
    smoke_albedo: tuple = (0.22, 0.21, 0.20)
    smoke_density: float = 6.0
    ambient: tuple = (0.10, 0.11, 0.13)
    anisotropy: float = 0.35
    sun_color: tuple = (1.0, 0.95, 0.88)
    sun_intensity: float = 0.0
    sun_azimuth: float = 35.0
    sun_elevation: float = 40.0
    fire_scatter: float = 1.0
    light_spread: float = 0.35     # metres
    shadow: float = 1.0
    detail: float = 0.35
    detail_freq: float = 3.0       # 1/m
    detail_disp: float = 1.2       # cells
    detail_rise: float = 1.0       # m/s
    step: float = 0.7              # cells
    edge_fade: float = 10.0        # cells
    colour_gain: float = 1.0       # flame colourant brightness
    humidity: float = 50.0         # % relative humidity of the surrounding air
    steam_density: float = 0.4     # 1/m per g/m^3 of condensed water
    steam_albedo: tuple = (0.92, 0.93, 0.95)
    multiple_scattering: float = 1.0  # share of the light that bounces more than once inside smoke
    vapour: bool = False           # the volume carries water vapour
    colourant: bool = False        # the volume carries flame colourant


@dataclass
class CompParams:
    mode: str = 'composite'
    view: str = 'standard'
    knee: float = 0.8
    plate_gain: float = 1.0
    plate_transform: str = 'srgb'
    fire_gain: float = 1.0
    smoke_opacity: float = 1.0
    bloom: float = 0.15
    bloom_radius: float = 1.0
    light_cast: float = 0.6
    haze: float = 1.0              # strength, in pixels at 1080p lines
    haze_freq: float = 1.0         # relative frequency
    haze_speed: float = 1.0
    saturation: float = 1.0
    grain: float = 0.0
    bg: tuple = (0.0, 0.0, 0.0)
    bg_checker: bool = False
    tint: tuple = (1.0, 1.0, 1.0)
    depth_range: float = 12.0
    surface_light: float = 1.0     # fire light on the ground and on colliders in the shot
    scorch: float = 0.8            # darkening of burnt ground and objects


def pack_look(u: Uniforms, L: LookParams, log_y_ref: float, time: float):
    gain = L.intensity * (2.0 ** L.exposure)
    u.v4(L.ambient_k, L.flame_k, max(L.max_k, L.flame_k + 1.0), L.dynamic_range)
    u.v4(gain, L.flame_absorption, L.soot_glow, L.flame_occlusion)
    u.v4(*(np.asarray(L.blue_color) * L.blue * gain), L.ignition)
    u.v4(L.flame_sharpness, L.flame_threshold, L.flame_density)
    u.v4(*L.smoke_albedo, L.smoke_density)
    u.v4(*L.ambient, L.anisotropy)
    sun = np.asarray(L.sun_color) * L.sun_intensity
    u.v4(*sun, L.fire_scatter)
    u.v3(cam.sun_direction(L.sun_azimuth, L.sun_elevation), L.shadow)
    u.v4(L.detail, L.detail_freq, L.detail_disp, L.detail_rise)
    u.v4(log_y_ref, L.edge_fade, max(L.step, 0.2), time)
    u.v4(*L.steam_albedo, L.steam_density)
    q_air = max(0.0, min(L.humidity, 100.0)) / 100.0 * vapour_saturation(L.ambient_k)
    u.v4(q_air, L.colour_gain * gain * COLOURANT_RADIANCE, 1.0 if L.vapour else 0.0, 1.0 if L.colourant else 0.0)
    u.v4(max(0.0, min(L.multiple_scattering, 1.0)))
    return u


@dataclass
class SurfaceInputs:
    """What the ray march needs about the surfaces around the fire: the colliders in the shot (they
    hide the fire behind them), the ground, and the burn state for scorch."""
    colliders: list = field(default_factory=list)   # ColliderGPU, with burn slots
    meshes: Any = None                               # the solver's MeshLibrary (mesh colliders)
    burn: Any = None                                 # burnable floor (simulation grid, or just its bottom layer)
    burn_obj: Any = None                             # object-burn atlas
    slots: Any = None                                # object-burn regions (buffer)
    grid: tuple = (1, 1, 1)                          # simulation grid dims (for the floor)
    cell: float = 1.0                                # simulation cell size (m)
    ground: bool = True                              # the ground plane is in the shot
    lit: bool = True                                 # gather fire light on surfaces
    scorch: bool = False                             # look up scorch


def vapour_saturation(tk):
    """Saturation vapour density of water (g/m^3) at tk Kelvin (Magnus formula; matches shade.wgsl)."""
    tc = tk - 273.15
    es = 610.94 * math.exp(min(17.625 * tc / max(tc + 243.04, 1.0), 60.0))
    return es / (461.5 * max(tk, 1.0)) * 1000.0


class Renderer:
    BLOOM_LEVELS = 7

    def __init__(self, gpu: GPU):
        self.gpu = gpu
        g = gpu
        lut_fmt = 'rgba32float' if gpu.float32_filterable else 'rgba16float'
        self.bb = g.texture2d(BB_SIZE, 1, lut_fmt, 'blackbody-lut')
        self._lut = blackbody_lut()
        g.upload(self.bb, self._lut.reshape(1, BB_SIZE, 4))
        self.noise = g.texture3d((64, 64, 64), 'rgba8unorm', 'detail-noise',
                                 usage=TU.TEXTURE_BINDING | TU.COPY_DST)
        g.upload(self.noise, detail_noise_volume(64))
        self.k_emit = g.kernel('light_emit.wgsl', ['tex3d', 'tex3d', 'tex3d', 'tex2d', 'smp', 'st3d:rgba16float:w'], defines=BB_DEFINES)
        self.k_blur = g.kernel('blur3.wgsl', ['utex3d', 'st3d:rgba16float:w'])
        self.k_shadow = g.kernel('light_shadow.wgsl', ['tex3d', 'utex3d', 'smp', 'st3d:rgba16float:w', 'st3d:rgba16float:w'])
        self.k_march = g.kernel('raymarch.wgsl', ['tex3d', 'tex3d', 'tex3d', 'tex3d', 'tex3d', 'tex2d', 'smp', 'smp', 'tex3d', 'tex3d',
                                                  'st2d:rgba16float:w', 'st2d:rgba16float:w', 'st2d:rgba16float:w',
                                                  'st2d:rgba16float:w', 'st2d:rgba16float:w', 'rbuf', 'rbuf',
                                                  'utex3d', 'utex3d', 'rbuf', 'utex3d', 'utex2d'],
                                defines=BB_DEFINES, workgroup=(8, 8, 1))
        self.k_lights = g.kernel('lights.wgsl', ['utex3d', 'buf', 'buf'], workgroup=(4, 4, 4))
        self.light_count = g.buffer(16, 'light-count')
        self.lights = None
        self._no_slots = g.buffer(32, 'no-slots')
        self._empty_r32 = g.texture3d((1, 1, 1), 'r32float', 'empty-r32')
        g.upload(self._empty_r32, np.full((1, 1, 1, 1), 1.0e6, np.float32))
        self.k_down = g.kernel('bloom_down.wgsl', ['tex2d', 'smp', 'st2d:rgba16float:w'], workgroup=(8, 8, 1))
        self.k_up = g.kernel('bloom_up.wgsl', ['tex2d', 'utex2d', 'smp', 'st2d:rgba16float:w'], workgroup=(8, 8, 1))
        self.k_comp = g.kernel('composite.wgsl', ['tex2d', 'tex2d', 'tex2d', 'tex2d', 'tex2d', 'tex2d', 'smp',
                                                  'st2d:rgba8unorm:w', 'st2d:rgba16float:w', 'tex2d', 'tex2d'], workgroup=(8, 8, 1))
        self.light_dims = None
        self.fire_size = None
        self.out_size = None
        self._black = g.texture2d(4, 4, 'rgba16float', 'black')
        g.upload(self._black, np.zeros((4, 4, 4), np.float16))
        self._empty = g.texture3d((1, 1, 1), 'rgba16float', 'empty-volume')  # for volumes without vapour or colourant
        g.upload(self._empty, np.zeros((1, 1, 1, 4), np.float16))
        self.plate = None
        self.plate_size = None

    # -- resources -------------------------------------------------------------------------------

    def _ensure_light(self, sim_dims):
        ld = tuple(max(4, (d + 1) // 2) for d in sim_dims)
        if ld == self.light_dims:
            return
        for name in ('E', 'T1', 'T2', 'EB', 'L0', 'L1'):
            t = getattr(self, name, None)
            if t is not None:
                t.destroy()
        g = self.gpu
        self.E, self.T1, self.T2, self.EB, self.L0, self.L1 = (g.texture3d(ld, 'rgba16float', n) for n in
                                                                ('E', 'T1', 'T2', 'EB', 'L0', 'L1'))
        self.light_dims = ld
        blocks = int(np.prod([-(-d // LIGHT_BLOCK) for d in ld]))
        if self.lights is not None:
            self.lights.destroy()
        self.lights = g.buffer(max(32, blocks * 32), 'fire-lights')

    def _ensure_fire(self, w, h):
        if (w, h) == self.fire_size:
            return
        g = self.gpu
        for name in ('beauty', 'emit', 'aux', 'surf', 'mask'):
            t = getattr(self, name, None)
            if t is not None:
                t.destroy()
        usage = TU.TEXTURE_BINDING | TU.STORAGE_BINDING | TU.COPY_SRC | TU.RENDER_ATTACHMENT
        self.beauty = g.texture2d(w, h, 'rgba16float', 'beauty', usage)
        self.emit = g.texture2d(w, h, 'rgba16float', 'emission', usage)
        self.aux = g.texture2d(w, h, 'rgba16float', 'aux', usage)
        # surf: fire light on the surface seen (rgb), holdout coverage (a); mask: scorch, holdout depth * coverage, coverage
        self.surf = g.texture2d(w, h, 'rgba16float', 'surface-light', usage)
        self.mask = g.texture2d(w, h, 'rgba16float', 'surface-mask', usage)
        for t in getattr(self, 'down', []) + getattr(self, 'up', []):
            t.destroy()
        self.down, self.up = [], []
        bw, bh = w, h
        for i in range(self.BLOOM_LEVELS):
            bw, bh = max(1, (bw + 1) // 2), max(1, (bh + 1) // 2)
            self.down.append(g.texture2d(bw, bh, 'rgba16float', f'bloom-d{i}'))
            self.up.append(g.texture2d(bw, bh, 'rgba16float', f'bloom-u{i}'))
            if bw <= 2 and bh <= 2:
                break
        self.fire_size = (w, h)

    def _ensure_out(self, w, h):
        if (w, h) == self.out_size:
            return
        g = self.gpu
        for name in ('disp', 'lin'):
            t = getattr(self, name, None)
            if t is not None:
                t.destroy()
        self.disp = g.texture2d(w, h, 'rgba8unorm', 'display')
        self.lin = g.texture2d(w, h, 'rgba16float', 'composite-linear')
        self.out_size = (w, h)

    def set_plate(self, image, fmt='rgba8unorm'):
        """Upload a footage frame: (h, w, 4) uint8 (display-encoded) or float16/float32 (any)."""
        if image is None:
            self.plate = None
            return
        h, w = image.shape[:2]
        if image.dtype != np.uint8:
            fmt = 'rgba16float'
            image = image.astype(np.float16)
        if self.plate is None or self.plate_size != (w, h) or self.plate.format != fmt:
            if self.plate is not None:
                self.plate.destroy()
            self.plate = self.gpu.texture2d(w, h, fmt, 'plate', TU.TEXTURE_BINDING | TU.COPY_DST)
            self.plate_size = (w, h)
        self.gpu.upload(self.plate, image)

    def log_y_ref(self, flame_k):
        u = (flame_k - BB_MIN_K) / (BB_MAX_K - BB_MIN_K) * (BB_SIZE - 1)
        i = int(np.clip(math.floor(u), 0, BB_SIZE - 2))
        f = float(np.clip(u - i, 0.0, 1.0))
        return float(self._lut[i, 3] * (1 - f) + self._lut[i + 1, 3] * f)

    def _aux_of(self, vol):
        a = getattr(vol, 'aux', None)
        return (a[0] if isinstance(a, (list, tuple)) else a) or self._empty

    def _chem_of(self, vol):
        c = getattr(vol, 'chem', None)
        return (c[0] if isinstance(c, (list, tuple)) else c) or self._empty

    # -- passes ----------------------------------------------------------------------------------

    def light(self, b, solver, look: LookParams, fire: cam.FireXform, time=0.0):
        self._ensure_light(solver.dims)
        ld = self.light_dims
        u = Uniforms().v4(*solver.dims, 0).v4(*ld, 0)
        pack_look(u, look, self.log_y_ref(look.flame_k), time)
        b.run(self.k_emit, [solver.scal[0], self._aux_of(solver), self._chem_of(solver), self.bb, self.gpu.linear, self.E], u, ld)
        hl = solver.h * solver.dims[1] / ld[1]
        # a short list of point lights standing in for the fire, for lighting surfaces
        self.gpu.write_buffer(self.light_count, np.zeros(4, np.uint32))
        blocks = tuple(-(-d // LIGHT_BLOCK) for d in ld)
        b.run(self.k_lights, [self.E, self.lights, self.light_count],
              Uniforms().v4(*ld, hl).v4(*solver.origin, LIGHT_BLOCK), blocks)
        sigma = max(0.75, look.light_spread / hl)
        r = int(min(24, math.ceil(sigma * 2.5)))
        src = self.E
        for ax, dst in (((1, 0, 0), self.T1), ((0, 1, 0), self.T2), ((0, 0, 1), self.EB)):
            b.run(self.k_blur, [src, dst], Uniforms().v4(*ld, 0).v4(*ax, r).v4(sigma), ld)
            src = dst
        # key-light direction in light-grid space (undo the fire's yaw)
        sd = cam.sun_direction(look.sun_azimuth, look.sun_elevation)
        ry = cam.rot_y(math.radians(fire.yaw))
        sg = ry.T @ sd
        sg = sg / (np.abs(sg).max() + 1e-9)  # at most one cell per step on the major axis
        steps = int(sum(ld))
        fire_scale = look.light_spread * 4.0
        u = Uniforms().v4(*ld, hl).v3(sg, steps).v4(look.shadow, 0.5 * look.shadow, fire_scale)
        b.run(self.k_shadow, [self.E, self.EB, self.gpu.linear, self.L0, self.L1], u, ld)

    def march(self, b, solver, camstate: cam.CameraState, fire: cam.FireXform, look: LookParams, size,
              jitter=(0.0, 0.0), seed=0.0, shutter=0.0, ground=True, time=0.0, max_steps=None, surfaces=None,
              limit=None):
        """limit: a texture with a liquid's depth (y, m) and coverage (w) per pixel; the march stops there."""
        from .solver import pack_colliders
        w, h = size
        self._ensure_fire(w, h)
        w2g = cam.world_to_grid(fire, solver.origin, solver.h)
        nx, ny, nz = solver.dims
        steps = max_steps or int(math.sqrt(nx * nx + ny * ny + nz * nz) / max(look.step, 0.2) * 1.2) + 8
        sf = surfaces or SurfaceInputs(ground=False, lit=False)
        holdouts = any(c.holdout for c in sf.colliders)
        u = (Uniforms().m4(camstate.inv_view_proj).m4(w2g).m4(camstate.view_proj)
             .v4(nx, ny, nz, solver.h).v4(*self.light_dims, 1.0 if ground else 0.0)
             .v4(w, h, *jitter).v4(seed, shutter, 1.0 if shutter > 0 else 0.0, steps))
        pack_look(u, look, self.log_y_ref(look.flame_k), time)
        u.v4(*solver.origin, getattr(solver, 'vel_k', 1))
        u.v4(1.0 if holdouts else 0.0, 1.0 if sf.ground else 0.0, 1.0 if sf.lit else 0.0,
             1.0 if (sf.scorch and (sf.burn is not None or sf.burn_obj is not None)) else 0.0)
        u.v4(*sf.grid, sf.cell)
        pack_colliders(u, sf.colliders, sf.meshes)
        u.v4(1.0 if limit is not None else 0.0)
        b.run(self.k_march, [solver.scal[0], solver.vel[0], self.L0, self.L1, self.noise, self.bb,
                             self.gpu.linear, self.gpu.repeat, self._aux_of(solver), self._chem_of(solver),
                             self.beauty, self.emit, self.aux, self.surf, self.mask, self.lights, self.light_count,
                             sf.burn or self._empty, sf.burn_obj or self._empty, sf.slots or self._no_slots,
                             sf.meshes.atlas if sf.meshes is not None else self._empty_r32,
                             limit if limit is not None else self._black], u, (w, h, 1))

    def bloom(self, b, radius=1.0):
        """Downsample the emission pass into a pyramid and fold it back up. Leaves the full bloom in
        up[0] and a wide glow (for light cast onto footage) in up[k]."""
        g = self.gpu
        src = self.emit
        sw, sh = self.fire_size
        for i, d in enumerate(self.down):
            dw, dh = d.size[:2]
            b.run(self.k_down, [src, g.linear, d], Uniforms().v4(sw, sh, 1 / sw, 1 / sh).v4(dw, dh, 1 if i == 0 else 0),
                  (dw, dh, 1))
            src, sw, sh = d, dw, dh
        n = len(self.down)
        last = self.down[-1]
        low = last
        for i in range(n - 2, -1, -1):
            d = self.down[i]
            dw, dh = d.size[:2]
            lw, lh = low.size[:2]
            weight = 1.0 if radius >= 1.0 else max(0.0, radius) ** (n - 1 - i)
            b.run(self.k_up, [low, d, g.linear, self.up[i]],
                  Uniforms().v4(lw, lh, 1 / lw, 1 / lh).v4(dw, dh, weight * (1.0 / max(radius, 1.0)) ** (i * 0.25), 1.0),
                  (dw, dh, 1))
            low = self.up[i]
        self.bloom_tex = self.up[0] if n > 1 else self.down[0]
        self.glow_tex = self.up[min(3, n - 2)] if n > 2 else self.bloom_tex

    def composite(self, b, out_size, comp: CompParams, time=0.0, frame=0, plate_fit=(1.0, 1.0), liquid=False):
        w, h = out_size
        self._ensure_out(w, h)
        has_plate = self.plate is not None
        haze_px = comp.haze * h / 1080.0 * 4.0
        u = (Uniforms()
             .v4(w, h, 1.0 if has_plate else 0.0, VIEW_MODES.get(comp.mode, 0))
             .v4(comp.plate_gain, INPUT_TRANSFORMS.get(comp.plate_transform, 0), *plate_fit)
             .v4(comp.fire_gain, comp.smoke_opacity, comp.bloom, comp.light_cast)
             .v4(haze_px, 0.018 * comp.haze_freq * 1080.0 / h, 40.0 * comp.haze_speed, time)
             .v4(comp.saturation, comp.grain, frame, comp.knee)
             .v4(VIEW_TRANSFORMS.get(comp.view, 0), 1.0 if comp.bg_checker else 0.0, max(8, h / 60), comp.depth_range)
             .v4(*comp.bg, 0)
             .v4(*comp.tint, 1.0)
             .v4(2.0 if liquid == 'both' else (1.0 if liquid else 0.0))
             .v4(comp.surface_light, comp.scorch))
        plate = self.plate if has_plate else self._black
        surf, mask = (self._black, self._black) if liquid is True else (self.surf, self.mask)
        b.run(self.k_comp, [plate, self.beauty, self.emit, self.aux, self.bloom_tex, self.glow_tex, self.gpu.linear,
                            self.disp, self.lin, surf, mask], u, (w, h, 1))

    def read_display(self):
        return self.gpu.read(self.disp)

    def read_linear(self):
        return self.gpu.read(self.lin)
