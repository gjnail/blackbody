"""Volume renderer and compositor.

Per frame: a half-resolution light volume (fire light, key-light shadows, sky occlusion,
occupancy), a ray march into beauty / emission / aux buffers, embers on top, a bloom pyramid,
and a composite over the footage with heat haze and fire light cast onto the scene.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import camera as cam
from .gpu import GPU, Uniforms, TU
from .lut import BB_MAX_K, BB_MIN_K, BB_SIZE, blackbody_lut, detail_noise_volume
from ..io.platestats import NOISE_BINS, NOISE_LO, measure_plate

BB_DEFINES = {'BB_MIN': float(BB_MIN_K), 'BB_MAX': float(BB_MAX_K)}

LIGHT_BLOCK = 8  # light-volume cells per side of each point light standing in for the fire
COLOURANT_RADIANCE = 25.0  # radiance per metre of one unit of flame colourant at gain 1
COAL_FREQ = 4.0            # 1/m: the coal bed's lumps
HAZE_EDDY_FREQ = 3.0       # 1/m: the small eddies that make heat shimmer (the noise tile's base frequency)
HAZE_EDDIES = 1.0          # their strength, relative to how much hotter than the air they are

VIEW_MODES = {'composite': 0, 'fire': 1, 'alpha': 2, 'emission': 3, 'heat': 4, 'depth': 5, 'temperature': 6}
VIEW_TRANSFORMS = {'standard': 0, 'agx': 1, 'aces': 2, 'raw': 3, 'ocio': 4}
INPUT_TRANSFORMS = {'srgb': 0, 'rec709': 1, 'linear': 2, 'acescg': 3, 'ocio': 4}
DEPTH_KINDS = {'z': 0, 'distance': 1, 'inverse': 2}
DEEP_SPLIT = 0.12  # opacity (or light) gathered before the march starts another deep sample
EMBER_DEEP_SCALE = 65536.0  # fixed-point scale of the embers' deep light (embers_draw.wgsl)
EMBER_DEEP_MIN_T = 0.02      # an ember behind deep samples letting less than this through goes in front of them
MAX_LAMPS = 8               # lights in the set the smoke is lit by
LAMP_SCALE = 3.0 / 50000.0  # lux at 1 m -> key-light units (Key intensity 3 is about 50 000 lux of daylight)
LAMP_KINDS = {'point': 0, 'spot': 1, 'area': 2}


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
    coal_bed: float = 0.0          # brightness of glowing coals on the ground under the fire (0 = none)
    coal_k: float = 1250.0         # their temperature (K)
    coal_height: float = 0.08      # how high they are heaped (m)
    lamps: list = field(default_factory=list)  # lights in the set (Scene.lamps): they light the smoke
    colour_response: str = 'camera'  # blackbody colours as a camera records them, or as the eye sees them


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
    surface_shadows: float = 1.0   # colliders and smoke shade the fire light on surfaces
    soot: float = 0.8              # darkening where smoke has left soot
    wet: float = 0.5               # darkening of soaked surfaces
    depth_kind: str = 'z'          # the footage depth pass: z, distance or inverse (1/Z)
    depth_scale: float = 1.0       # metres per unit of the depth pass
    highlight_white: float = 0.25  # Standard view: channel crosstalk that takes over-bright highlights to white
    visibility: float = 0.0        # m: how far one can see through the air in the shot (0 = clear air)
    atmos_colour: tuple = (0.55, 0.6, 0.7)  # colour of the haze (linear), unless taken from the footage
    atmos_from_footage: bool = True
    grain_match: bool = True       # give the fire the footage's own measured noise
    f_stop: float = 0.0            # aperture for depth of field (0 = everything sharp)
    focus_distance: float = 0.0    # m (0 = focused on the fire)
    softness: float = 0.0          # px at 1080p: the footage's own softness, given to the element
    focal_mm: float = 35.0         # the camera's lens, for depth of field
    sensor_mm: float = 36.0
    lens_k1: float = 0.0           # radial distortion of the footage's lens (barrel < 0 < pincushion), of the half-diagonal
    fringing: float = 0.0          # lateral chromatic aberration at the frame corners (px at 1080p)
    halation: float = 0.0          # red halo around the brightest light, as film shows


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
    stain: Any = None                                # soot left on surfaces (simulation grid, r32float)
    stain_obj: Any = None                            # soot on colliders, each in its own frame (r32float atlas)
    stain_slots: Any = None                          # their regions of it (one per collider, in order)
    stain_regions: int = 0
    shadows: float = 1.0                             # how much colliders and smoke shade the fire light
    wet: bool = False                                # look up wetness (burnable surfaces)


def vapour_saturation(tk):
    """Saturation vapour density of water (g/m^3) at tk Kelvin (Magnus formula; matches shade.wgsl)."""
    tc = tk - 273.15
    es = 610.94 * math.exp(min(17.625 * tc / max(tc + 243.04, 1.0), 60.0))
    return es / (461.5 * max(tk, 1.0)) * 1000.0


class Renderer:
    BLOOM_LEVELS = 7
    _shaper_lo = -12.0   # io/ocio.py SHAPER_LO / SHAPER_HI: the log shaper the OCIO view LUT is baked over
    _shaper_hi = 8.0

    def __init__(self, gpu: GPU):
        self.gpu = gpu
        g = gpu
        lut_fmt = 'rgba32float' if gpu.float32_filterable else 'rgba16float'
        self.bb = g.texture2d(BB_SIZE, 1, lut_fmt, 'blackbody-lut')
        self._lut_response = 'eye'
        self._lut = blackbody_lut()
        g.upload(self.bb, self._lut.reshape(1, BB_SIZE, 4))
        self.noise = g.texture3d((64, 64, 64), 'rgba8unorm', 'detail-noise',
                                 usage=TU.TEXTURE_BINDING | TU.COPY_DST)
        g.upload(self.noise, detail_noise_volume(64))
        self.k_emit = g.kernel('light_emit.wgsl', ['tex3d', 'tex3d', 'tex3d', 'tex2d', 'smp', 'st3d:rgba16float:w'], defines=BB_DEFINES)
        self.k_blur = g.kernel('blur3.wgsl', ['utex3d', 'st3d:rgba16float:w'])
        self.k_shadow = g.kernel('light_shadow.wgsl', ['tex3d', 'utex3d', 'smp', 'st3d:rgba16float:w', 'st3d:rgba16float:w'])
        # fabric in the light volume (cloth.py occlusion): its extinction, and the objects' (L1.z, L1.w)
        self.k_occlude = g.kernel('cloth_occlude.wgsl', ['tex3d', 'rbuf', 'utex3d', 'st3d:rgba16float:w', 'st3d:rgba16float:w'],
                                  workgroup=(4, 4, 4))
        self.k_occ_l1 = g.kernel('cloth_l1.wgsl', ['tex3d', 'tex3d', 'st3d:rgba16float:w'], workgroup=(4, 4, 4))
        self.CO = None
        self.occluded = False       # this frame's light volume has cloth in it
        self.k_march = g.kernel('raymarch.wgsl', ['tex3d', 'tex3d', 'tex3d', 'tex3d', 'tex3d', 'tex2d', 'smp', 'smp', 'tex3d', 'tex3d',
                                                  'st2d:rgba16float:w', 'st2d:rgba16float:w', 'st2d:rgba16float:w',
                                                  'st2d:rgba16float:w', 'st2d:rgba16float:w', 'rbuf', 'rbuf',
                                                  'utex3d', 'utex3d', 'rbuf', 'utex3d', 'utex2d', 'utex3d', 'tex2d', 'buf',
                                                  'utex3d', 'rbuf', 'tex3d', 'rbuf', 'st2d:rgba16float:w'],
                                defines=BB_DEFINES, workgroup=(8, 8, 1))
        self.k_lights = g.kernel('lights.wgsl', ['utex3d', 'buf', 'buf'], workgroup=(4, 4, 4))
        self.k_lamps = g.kernel('lamps.wgsl', ['tex3d', 'smp', 'rbuf', 'st3d:rgba16float:w'])
        self._lamp_buf = g.buffer(MAX_LAMPS * 64, 'lamps')
        self._no_soot_slots = g.buffer(64, 'no-soot-slots')
        self._lamps_on = 0          # lights in the set this frame (the march reads their buffer)
        self.LT = None              # their transmittance through the smoke: light volume x (1 or 2) deep
        self._lt_dims = None
        self.light_count = g.buffer(16, 'light-count')
        self.lights = None
        self._no_slots = g.buffer(32, 'no-slots')
        self._empty_r32 = g.texture3d((1, 1, 1), 'r32float', 'empty-r32')
        g.upload(self._empty_r32, np.full((1, 1, 1, 1), 1.0e6, np.float32))
        self.k_down = g.kernel('bloom_down.wgsl', ['tex2d', 'smp', 'st2d:rgba16float:w'], workgroup=(8, 8, 1))
        self.k_up = g.kernel('bloom_up.wgsl', ['tex2d', 'utex2d', 'smp', 'st2d:rgba16float:w'], workgroup=(8, 8, 1))
        self.k_comp = g.kernel('composite.wgsl', ['tex2d', 'tex2d', 'tex2d', 'tex2d', 'tex2d', 'tex2d', 'smp',
                                                  'st2d:rgba8unorm:w', 'st2d:rgba16float:w', 'tex2d', 'tex2d', 'tex3d', 'tex3d',
                                                  'tex2d', 'tex2d'],
                               workgroup=(8, 8, 1),
                               defines={'NOISE_BINS': NOISE_BINS, 'NOISE_LO': NOISE_LO})
        self.k_dof = g.kernel('defocus.wgsl', ['tex2d', 'tex2d', 'tex2d', 'smp', 'st2d:rgba16float:w', 'st2d:rgba16float:w',
                                               'st2d:rgba16float:w'], workgroup=(8, 8, 1))
        self.k_copy3 = g.kernel('copy2d.wgsl', ['tex2d', 'tex2d', 'tex2d', 'st2d:rgba16float:w', 'st2d:rgba16float:w',
                                                'st2d:rgba16float:w'], workgroup=(8, 8, 1))
        self.k_opl = g.kernel('haze_opl.wgsl', ['tex3d', 'tex3d', 'tex3d', 'smp', 'smp', 'st2d:rgba16float:w'], workgroup=(8, 8, 1))
        self._haze_src = None   # what the last march rendered, for the heat haze pass
        self.light_dims = None
        self.fire_size = None
        self.out_size = None
        self._black = g.texture2d(4, 4, 'rgba16float', 'black')
        g.upload(self._black, np.zeros((4, 4, 4), np.float16))
        self._empty = g.texture3d((1, 1, 1), 'rgba16float', 'empty-volume')  # for volumes without vapour or colourant
        g.upload(self._empty, np.zeros((1, 1, 1, 4), np.float16))
        self.plate = None
        self.plate_size = None
        # holdouts from the footage (x = matte, y = depth pass), deep samples, OCIO LUTs
        self.hold = None
        self.hold_on = (False, False)
        self.deep_n = 0
        self.deep = None
        self._no_deep = g.buffer(32, 'no-deep')
        self._lut_none = g.texture3d((2, 2, 2), 'rgba16float', 'no-lut')
        g.upload(self._lut_none, np.zeros((2, 2, 2, 4), np.float16))
        self.lut_view = None
        self.lut_plate = None
        self.lut_plate_log = False
        self.lut_size = 2
        self._plate_img = None      # the footage frame as given, for measuring its noise and haze
        self._stats_key = None
        self._stats = {}

    # -- resources -------------------------------------------------------------------------------

    @staticmethod
    def light_dims_for(sim_dims):
        """The light volume's dims for a simulation grid (half resolution)."""
        return tuple(max(4, (d + 1) // 2) for d in sim_dims)

    def _ensure_light(self, sim_dims):
        ld = self.light_dims_for(sim_dims)
        if ld == self.light_dims:
            return
        for name in ('E', 'T1', 'T2', 'EB', 'L0', 'L1', 'CO'):
            t = getattr(self, name, None)
            if t is not None:
                t.destroy()
        g = self.gpu
        self.E, self.T1, self.T2, self.EB, self.L0, self.L1, self.CO = (g.texture3d(ld, 'rgba16float', n) for n in
                                                                         ('E', 'T1', 'T2', 'EB', 'L0', 'L1', 'CO'))
        self.light_dims = ld
        blocks = int(np.prod([-(-d // LIGHT_BLOCK) for d in ld]))
        if self.lights is not None:
            self.lights.destroy()
        self.lights = g.buffer(max(32, blocks * 32), 'fire-lights')

    def _ensure_fire(self, w, h):
        if (w, h) == self.fire_size:
            return
        g = self.gpu
        for name in ('beauty', 'emit', 'aux', 'surf', 'mask', 'lamp_surf', 'dof_b', 'dof_e', 'dof_x', 'opl'):
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
        # lamp_surf: how the lights in the set change the light on the surface seen (the footage times 1 + rgb)
        self.lamp_surf = g.texture2d(w, h, 'rgba16float', 'lamp-surface', usage)
        # depth of field works into these, then copies back; heat haze is traced at half resolution
        self.dof_b, self.dof_e, self.dof_x = (g.texture2d(w, h, 'rgba16float', n, usage) for n in ('dof-beauty', 'dof-emit', 'dof-aux'))
        self.opl = g.texture2d(max(4, w // 2), max(4, h // 2), 'rgba16float', 'haze-path', usage)
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
        self._plate_img = image
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

    def set_holdout(self, matte=None, depth=None):
        """Holdouts from the footage: matte (h, w) 0..1 of what is in front of the fire, and/or a depth
        pass (h, w) in its own units (CompParams.depth_kind / depth_scale). None clears."""
        if matte is None and depth is None:
            self.hold_on = (False, False)
            return
        ref = matte if matte is not None else depth
        h, w = ref.shape[:2]
        img = np.zeros((h, w, 4), np.float32)
        if matte is not None:
            img[..., 0] = matte
        if depth is not None:
            d = np.nan_to_num(np.asarray(depth, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            img[..., 1] = np.clip(d, 0.0, 65000.0)
        if self.hold is None or self.hold.size[:2] != (w, h):
            if self.hold is not None:
                self.hold.destroy()
            self.hold = self.gpu.texture2d(w, h, 'rgba16float', 'footage-holdout', TU.TEXTURE_BINDING | TU.COPY_DST)
        self.gpu.upload(self.hold, img.astype(np.float16))
        self.hold_on = (matte is not None, depth is not None)

    def set_ocio(self, view_lut=None, plate_lut=None, plate_log=False):
        """OCIO 3D LUTs (size^3, 4) from io/ocio.py for the view and the footage, or None."""
        for name, lut in (('lut_view', view_lut), ('lut_plate', plate_lut)):
            old = getattr(self, name)
            if lut is None:
                if old is not None:
                    old.destroy()
                setattr(self, name, None)
                continue
            n = lut.shape[0]
            if old is None or old.size != (n, n, n):
                if old is not None:
                    old.destroy()
                old = self.gpu.texture3d((n, n, n), 'rgba16float', name, TU.TEXTURE_BINDING | TU.COPY_DST)
                setattr(self, name, old)
            self.gpu.upload(old, lut.astype(np.float16))
            self.lut_size = n
        self.lut_plate_log = bool(plate_log)

    def set_deep(self, samples, passes=1):
        """Gather up to `samples` deep samples per pixel (0 turns it off), over `passes` anti-aliasing
        passes of march (pass index deep_pass = 0 .. passes - 1), plus the embers drawn afterwards."""
        self.deep_n = max(0, int(samples))
        self.deep_passes = max(1, int(passes))
        self._ember_deep_used = False

    def _ensure_deep(self, w, h):
        need = w * h * self.deep_n * 32
        if self.deep is None or self.deep.size < need:
            if self.deep is not None:
                self.deep.destroy()
            self.deep = self.gpu.buffer(max(32, need), 'deep-samples')

    def ember_deep_buffer(self, w, h, clear=True):
        """The embers' deep light per pixel: fixed-point rgb (EMBER_DEEP_SCALE) and the nearest depth (as
        float bits), gathered with atomics by embers_draw.wgsl."""
        need = w * h * 16
        if getattr(self, 'ember_deep', None) is None or self.ember_deep.size < need:
            if getattr(self, 'ember_deep', None) is not None:
                self.ember_deep.destroy()
            self.ember_deep = self.gpu.buffer(max(32, need), 'ember-deep')
        if clear:
            init = np.zeros((w * h, 4), np.uint32)
            init[:, 3] = 0x7F7FFFFF   # the largest float: no ember yet
            self.gpu.write_buffer(self.ember_deep, init)
            self._ember_deep_used = False
        return self.ember_deep

    def read_deep(self):
        """(h, w, samples + 1, 8) float32 deep samples, front to back: premultiplied rgba, then the front
        and back depth (m, along the view axis). They composite back (over, front to back) to the beauty
        averaged over all the anti-aliasing passes; the last slot holds the embers' light (no alpha)."""
        w, h = self.fire_size
        n = self.deep_n
        raw = np.frombuffer(self.gpu.read_buffer(self.deep), np.float32)[:w * h * n * 8].reshape(h, w, n, 8).copy()
        # bins opened by later passes come after the first pass's: put every pixel's bins front to back
        used = (raw[..., 3] > 0.0) | (raw[..., :3].max(-1) > 0.0)
        order = np.argsort(np.where(used, raw[..., 4], np.inf), axis=-1, kind='stable')
        raw = np.take_along_axis(raw, order[..., None], axis=2)
        k = float(max(1, getattr(self, 'deep_passes', 1)))
        dc = raw[..., :3] / k
        da = np.clip(raw[..., 3] / k, 0.0, 1.0)
        # each bin holds what it adds to the pixel; as a deep sample it is that over the light left in front of it
        t_front = np.clip(1.0 - (np.cumsum(da, axis=-1) - da), 1e-6, 1.0)
        out = np.zeros((h, w, n + 1, 8), np.float32)
        out[..., :n, :3] = dc / t_front[..., None]
        out[..., :n, 3] = np.clip(da / t_front, 0.0, 1.0)
        out[..., :n, 4:6] = raw[..., 4:6]
        if getattr(self, '_ember_deep_used', False):
            e = np.frombuffer(self.gpu.read_buffer(self.ember_deep), np.uint32)[:w * h * 4].reshape(h, w, 4)
            light = e[..., :3].astype(np.float32) / EMBER_DEEP_SCALE
            z = e[..., 3].copy().view(np.float32)
            has = light.max(-1) > 0.0
            # the embers were added over the flat image at full strength: undo the dimming the deep samples
            # in front will apply when it is composited. The flat image shows embers even behind thick
            # flame, so an ember the samples in front would all but hide goes at the front of the sample
            # that hides it, where it still shows as it does in the image.
            front = (out[..., :n, 4] < z[..., None]) & (out[..., :n, 3] > 0.0)
            keep = np.where(front, 1.0 - out[..., :n, 3], 1.0)          # samples are front to back
            before = np.cumprod(np.concatenate([np.ones((h, w, 1), np.float32), keep[..., :-1]], axis=-1), axis=-1)
            hides = front & (before * keep < EMBER_DEEP_MIN_T)
            first = np.argmax(hides, axis=-1)
            hidden = hides.any(-1)
            t_at = np.where(hidden, np.take_along_axis(before, first[..., None], -1)[..., 0], np.prod(keep, axis=-1))
            z_hide = np.take_along_axis(out[..., :n, 4], first[..., None], -1)[..., 0]
            z = np.where(hidden, np.nextafter(z_hide, np.float32(-np.inf)), z)
            out[..., n, :3] = np.where(has[..., None], light / np.maximum(t_at, 1e-6)[..., None], 0.0)
            out[..., n, 4] = np.where(has, z, 0.0)
            out[..., n, 5] = np.where(has, z, 0.0)
        return out

    def plate_stats(self, comp: CompParams):
        """(noise model, haze colour) of the current footage frame, each None when not needed or not
        measurable. Measured once per frame and kept while only the look changes."""
        img = self._plate_img
        want_noise = comp.grain_match
        want_haze = comp.visibility > 0.0 and comp.atmos_from_footage
        if img is None or not (want_noise or want_haze):
            return None, None
        key = (img.shape, img.dtype.str, comp.plate_transform, comp.plate_gain,
               zlib.crc32(np.ascontiguousarray(img[::29, ::31]).tobytes()))
        if key != self._stats_key:
            self._stats_key, self._stats = key, {}
        noise = want_noise and 'noise' not in self._stats
        haze = want_haze and 'haze' not in self._stats
        if noise or haze:
            model, colour = measure_plate(img, comp.plate_transform, comp.plate_gain, noise, haze)
            if noise:
                self._stats['noise'] = model
            if haze:
                self._stats['haze'] = colour
        return self._stats.get('noise') if want_noise else None, self._stats.get('haze') if want_haze else None

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

    def _use_response(self, response):
        """Load the blackbody colours for this response (camera or eye) if they are not loaded already."""
        if response != self._lut_response:
            self._lut = blackbody_lut(response=response)
            self.gpu.upload(self.bb, self._lut.reshape(1, BB_SIZE, 4))
            self._lut_response = response

    def light(self, b, solver, look: LookParams, fire: cam.FireXform, time=0.0, emit=None, occluder=None,
              colliders=None, meshes=None):
        """emit: a caller's own emission pass in place of light_emit.wgsl, emit(b, uniforms, textures, dims),
        given light_emit's uniforms (to add to) and bindings (fire, water and lava: both_engine.py).
        occluder: cloth in the light, occluder(b, light dims, light-grid corner, light cell (m, per axis)) ->
        a buffer of how much light it stops per light cell (cloth.py occlusion), or None; with it the
        colliders (and meshes) are put in the light volume too, for the cloth's shadows (L1.w)."""
        from .solver import pack_colliders
        self._use_response(look.colour_response)
        self._ensure_light(solver.dims)
        ld = self.light_dims
        u = Uniforms().v4(*solver.dims, 0).v4(*ld, 0)
        pack_look(u, look, self.log_y_ref(look.flame_k), time)
        res = [solver.scal[0], self._aux_of(solver), self._chem_of(solver), self.bb, self.gpu.linear, self.E]
        if emit is not None:
            emit(b, u, res, ld)
        else:
            b.run(self.k_emit, res, u, ld)
        hl = solver.h * solver.dims[1] / ld[1]
        lcell = np.asarray(solver.dims, float) * solver.h / np.asarray(ld, float)
        occ = occluder(b, ld, tuple(solver.origin), tuple(lcell)) if occluder is not None else None
        self.occluded = occ is not None
        if occ is not None:
            # cloth stops light like smoke (it goes into the extinction every shadow is marched through)
            u = Uniforms().v4(*ld, 0).v4(*solver.origin, 0).v4(*lcell, 0)
            pack_colliders(u, colliders or [], meshes)
            atlas = meshes.atlas if meshes is not None else self._empty_r32
            b.run(self.k_occlude, [self.E, occ, atlas, self.T1, self.CO], u, ld)
            self.E, self.T1 = self.T1, self.E
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
        if occ is not None:
            b.run(self.k_occ_l1, [self.L1, self.CO, self.T2], Uniforms().v4(*ld, 0), ld)
            self.L1, self.T2 = self.T2, self.L1
        # lights in the set: their buffer for the march, and the smoke's shadow on each (light volume)
        lamps = list(getattr(look, 'lamps', None) or [])[:MAX_LAMPS]
        self._lamps_on = len(lamps)
        if lamps:
            cell = np.asarray(solver.dims, float) * solver.h / np.asarray(ld, float)   # metres per light cell, per axis
            self.pack_lamps(lamps)
            blocks = 1 if len(lamps) <= 4 else 2
            lt = (ld[0], ld[1], ld[2] * blocks)
            if self._lt_dims != lt:
                if self.LT is not None:
                    self.LT.destroy()
                self.LT = self.gpu.texture3d(lt, 'rgba16float', 'lamp-transmittance')
                self._lt_dims = lt
            u = Uniforms().v4(*ld, blocks).v4(*cell, look.shadow).v4(*solver.origin, len(lamps)).v4(int(sum(ld)))
            b.run(self.k_lamps, [self.E, self.gpu.linear, self._lamp_buf, self.LT], u, lt)

    def pack_lamps(self, lamps):
        """Write the lights in the set (Scene.lamps) into their buffer, for the march and the fabric; returns
        how many there are."""
        lamps = list(lamps or [])[:MAX_LAMPS]
        if lamps:
            data = np.zeros((MAX_LAMPS, 16), np.float32)
            for i, L in enumerate(lamps):
                aim = np.asarray(L['direction'], float)
                aim = aim / max(float(np.linalg.norm(aim)), 1e-12)
                data[i] = (*L['position'], max(float(L['radius']), 1e-3),
                           *(np.asarray(L['power'], float) * LAMP_SCALE), LAMP_KINDS.get(L['kind'], 0),
                           *aim, L['cos_outer'], L['cos_inner'], 1.0 if L.get('shadows', True) else 0.0,
                           1.0 if L.get('in_footage', True) else 0.0, 0.0)
            self.gpu.write_buffer(self._lamp_buf, data)
        return len(lamps)

    def march(self, b, solver, camstate: cam.CameraState, fire: cam.FireXform, look: LookParams, size,
              jitter=(0.0, 0.0), seed=0.0, shutter=0.0, ground=True, time=0.0, max_steps=None, surfaces=None,
              limit=None, comp: CompParams | None = None, plate_fit=(1.0, 1.0), deep_pass=0):
        """limit: a texture with a liquid's depth (y, m) and coverage (w) per pixel; the march stops there.
        comp: the footage holdout settings (depth pass units), with set_holdout."""
        from .solver import pack_colliders
        self._use_response(look.colour_response)
        w, h = size
        self._ensure_fire(w, h)
        w2g = cam.world_to_grid(fire, solver.origin, solver.h)
        nx, ny, nz = solver.dims
        steps = max_steps or int(math.sqrt(nx * nx + ny * ny + nz * nz) / max(look.step, 0.2) * 1.2) + 8
        sf = surfaces or SurfaceInputs(ground=False, lit=False)
        holdouts = any(c.holdout for c in sf.colliders)
        if self.deep_n:
            self._ensure_deep(w, h)
        u = (Uniforms().m4(camstate.inv_view_proj).m4(w2g).m4(camstate.view_proj)
             .v4(nx, ny, nz, solver.h).v4(*self.light_dims, 1.0 if ground else 0.0)
             .v4(w, h, *jitter).v4(seed, shutter, 1.0 if shutter > 0 else 0.0, steps))
        pack_look(u, look, self.log_y_ref(look.flame_k), time)
        u.v4(*solver.origin, getattr(solver, 'vel_k', 1))
        u.v4(1.0 if holdouts else 0.0, 1.0 if sf.ground else 0.0, 1.0 if sf.lit else 0.0,
             1.0 if (sf.scorch and (sf.burn is not None or sf.burn_obj is not None)) else 0.0)
        u.v4(*sf.grid, sf.cell)
        stain = getattr(sf, 'stain', None)
        u.v4(max(0.0, min(float(getattr(sf, 'shadows', 1.0)), 1.0)), 1.0 if stain is not None else 0.0,
             1.0 if getattr(sf, 'wet', False) else 0.0, 1.0)
        c = comp or CompParams()
        matte_on, depth_on = self.hold_on if self.hold is not None else (False, False)
        u.v4(1.0 if matte_on else 0.0, 1.0 if depth_on else 0.0, DEPTH_KINDS.get(c.depth_kind, 0), c.depth_scale)
        soot_obj = getattr(sf, 'stain_obj', None) if stain is not None else None
        u.v4(*plate_fit, getattr(sf, 'stain_regions', 0) if soot_obj is not None else 0, self._lamps_on)
        view = np.asarray(camstate.view)
        fwd = -view[2, :3] / max(np.linalg.norm(view[2, :3]), 1e-12)
        p0 = camstate.inv_view_proj @ np.array([0.0, 0.0, 0.0, 1.0])
        near = float(np.dot(p0[:3] / p0[3] - np.asarray(camstate.eye), fwd))
        u.v3(fwd, near)
        u.v4(self.deep_n, w, DEEP_SPLIT, deep_pass)
        pack_colliders(u, sf.colliders, sf.meshes)
        u.v4(1.0 if limit is not None else 0.0, 1.0 if self.occluded else 0.0)
        u.v4(look.coal_bed, look.coal_k, COAL_FREQ, max(look.coal_height, 0.005))
        self._haze_src = (solver, camstate, fire, look, time)
        b.run(self.k_march, [solver.scal[0], solver.vel[0], self.L0, self.L1, self.noise, self.bb,
                             self.gpu.linear, self.gpu.repeat, self._aux_of(solver), self._chem_of(solver),
                             self.beauty, self.emit, self.aux, self.surf, self.mask, self.lights, self.light_count,
                             sf.burn or self._empty, sf.burn_obj or self._empty, sf.slots or self._no_slots,
                             sf.meshes.atlas if sf.meshes is not None else self._empty_r32,
                             limit if limit is not None else self._black,
                             stain if stain is not None else self._empty_r32,
                             self.hold if (self.hold is not None and any(self.hold_on)) else self._black,
                             self.deep if self.deep_n else self._no_deep,
                             soot_obj if soot_obj is not None else self._empty_r32,
                             sf.stain_slots if soot_obj is not None else self._no_soot_slots,
                             self.LT if self._lamps_on else self._empty, self._lamp_buf, self.lamp_surf], u, (w, h, 1))

    def defocus(self, b, comp: CompParams, camstate: cam.CameraState, fire: cam.FireXform):
        """Depth of field and the footage's softness on the rendered element (beauty, emission, aux), after
        the march and before bloom. Nothing to do with the aperture at 0 and no softness."""
        if comp.f_stop <= 0.0 and comp.softness <= 0.0:
            return
        w, h = self.fire_size
        dist = float(np.linalg.norm(np.asarray(camstate.eye, float) - np.asarray(fire.position, float)))
        focus = comp.focus_distance if comp.focus_distance > 0.0 else dist
        a = 0.0
        if comp.f_stop > 0.0:
            f = comp.focal_mm * 1e-3
            # circle of confusion on the sensor, f^2 / (N (s - f)) |d - s| / d, in element pixels
            a = f * f / (comp.f_stop * max(focus - f, 1e-3)) / max(comp.sensor_mm * 1e-3, 1e-6) * w
        u = Uniforms().v4(w, h).v4(a, focus, comp.softness * h / 1080.0, dist)
        b.run(self.k_dof, [self.beauty, self.emit, self.aux, self.gpu.linear, self.dof_b, self.dof_e, self.dof_x], u, (w, h, 1))
        b.run(self.k_copy3, [self.dof_b, self.dof_e, self.dof_x, self.beauty, self.emit, self.aux], Uniforms().v4(w, h), (w, h, 1))

    def _heat_haze(self, b, comp: CompParams):
        """Trace the extra optical path through the fire's hot air (haze_opl.wgsl) for the composite."""
        vol, camstate, fire, look, t = self._haze_src
        ow, oh = self.opl.size[:2]
        nx, ny, nz = vol.dims
        u = (Uniforms().m4(camstate.inv_view_proj).m4(cam.world_to_grid(fire, vol.origin, vol.h))
             .v4(nx, ny, nz, vol.h).v4(ow, oh)
             .v4(look.ambient_k, look.flame_k, max(look.max_k, look.flame_k + 1.0))
             .v4(HAZE_EDDIES, HAZE_EDDY_FREQ * comp.haze_freq, comp.haze_speed, t)
             .v4(*vol.origin, getattr(vol, 'vel_k', 1)))
        b.run(self.k_opl, [vol.scal[0], vol.vel[0], self.noise, self.gpu.linear, self.gpu.repeat, self.opl], u, (ow, oh, 1))

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

    def composite(self, b, out_size, comp: CompParams, time=0.0, frame=0, plate_fit=(1.0, 1.0), liquid=False,
                  liquid_haze=False, hfov=None):
        """liquid_haze: the liquid traced its own heat haze into self.opl (a molten liquid's hot air:
        LiquidRenderer.lava_haze), which bends the liquid element as well as the footage behind it.
        hfov: the camera's (for the haze's bend), when no fire was marched."""
        w, h = out_size
        self._ensure_out(w, h)
        has_plate = self.plate is not None
        haze_on = comp.haze > 0.0 and ((liquid is not True and self._haze_src is not None) or liquid_haze)
        if haze_on and not liquid_haze:
            self._heat_haze(b, comp)
        u = (Uniforms()
             .v4(w, h, 1.0 if has_plate else 0.0, VIEW_MODES.get(comp.mode, 0))
             .v4(comp.plate_gain, INPUT_TRANSFORMS.get(comp.plate_transform, 0), *plate_fit)
             .v4(comp.fire_gain, comp.smoke_opacity, comp.bloom, comp.light_cast)
             .v4(comp.haze, 0.0, 0.0, time)
             .v4(comp.saturation, comp.grain, frame, comp.knee)
             .v4(VIEW_TRANSFORMS.get(comp.view, 0), 1.0 if comp.bg_checker else 0.0, max(8, h / 60), comp.depth_range)
             .v4(*comp.bg, 0)
             .v4(*comp.tint, 1.0)
             .v4(2.0 if liquid == 'both' else (1.0 if liquid else 0.0))
             .v4(comp.surface_light, comp.scorch)
             .v4(comp.soot, comp.wet)
             .v4(self._shaper_lo, self._shaper_hi, 1.0 if self.lut_plate_log else 0.0, self.lut_size))
        noise, haze = self.plate_stats(comp) if has_plate else (None, None)
        u.v4(min(max(comp.highlight_white, 0.0), 0.9))
        # Koschmieder: at the visibility distance the air leaves 2% of the contrast
        u.v4(*(haze if haze is not None else comp.atmos_colour), 3.912 / comp.visibility if comp.visibility > 0 else 0.0)
        if noise is not None:
            # grain size in output pixels (measured in footage pixels)
            u.v4(1.0, noise.size * w / max(self.plate_size[0] * plate_fit[0], 1e-6))
            for c in range(3):
                u.v4(*noise.chol[:, c])
            for i in range(NOISE_BINS):
                u.v4(*noise.sigma[i])
        else:
            u.v4()
            for _ in range(3 + NOISE_BINS):
                u.v4()
        # the footage's lens: distortion, colour fringing (scale at the corners), halation
        half_diag = 0.5 * math.hypot(w, h)
        u.v4(comp.lens_k1, comp.fringing * h / 1080.0 / max(half_diag, 1.0), comp.halation)
        if hfov is None:
            hfov = self._haze_src[1].hfov if self._haze_src is not None else math.radians(54.0)
        u.v4(0.5 * w / math.tan(0.5 * hfov), 1.0 if haze_on else 0.0, 1.0 if liquid_haze else 0.0)
        plate = self.plate if has_plate else self._black
        surf, mask, lamp_surf = (self._black,) * 3 if liquid is True else (self.surf, self.mask, self.lamp_surf)
        b.run(self.k_comp, [plate, self.beauty, self.emit, self.aux, self.bloom_tex, self.glow_tex, self.gpu.linear,
                            self.disp, self.lin, surf, mask, self.lut_view or self._lut_none,
                            self.lut_plate or self._lut_none, self.opl if haze_on else self._black, lamp_surf], u, (w, h, 1))

    def read_display(self):
        return self.gpu.read(self.disp)

    def read_linear(self):
        return self.gpu.read(self.lin)
