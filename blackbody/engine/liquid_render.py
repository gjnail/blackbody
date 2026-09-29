"""Liquid rendering: a smooth surface from the particles, then ray tracing it over the footage.

The surface is a signed distance on a grid finer than the simulation (Surface detail, 1-3x),
built from the packed particles in z slabs so its accumulators stay small, then blurred. The ray
tracer (liq_march.wgsl) writes the same beauty / emission / aux buffers as the fire renderer, so
sampling, bloom, compositing and every output work unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import camera as cam
from .gpu import BU, GPU, SS, Uniforms, ceil_div, groups_1d, load_wgsl
from .liquid import PACKED_BYTES, WW_PACKED_BYTES
from .solver import pack_colliders

SLAB_NODES = 8 << 20   # surface-grid nodes per accumulation slab
ACC_BYTES = 48         # accumulator bytes per node (12 slots)


@dataclass
class WaterLook:
    ior: float = 1.333
    color: tuple = (0.70, 0.90, 0.93)   # what white light keeps after travelling `clarity` metres
    clarity: float = 3.0                # metres
    murk: float = 0.0                   # scattering extinction (1/m)
    murk_color: tuple = (0.55, 0.65, 0.60)
    roughness: float = 0.06
    reflection: float = 1.0
    reflect_footage: float = 0.75
    ripple: float = 0.25
    ripple_freq: float = 25.0           # 1/m
    ripple_speed: float = 2.0           # m/s for full ripples
    radius: float = 1.0                 # particle radius, in particle spacings
    influence: float = 2.0              # kernel radius, in particle radii
    smoothing: int = 1
    calm: int = 12                      # passes of smoothing along the surface, in dense liquid
    surface_res: float = 2.0
    wet_darken: float = 0.5
    wet_gloss: float = 0.5
    shadow: float = 1.0
    backdrop: float = 15.0              # metres behind the liquid that refracted rays see
    sky: tuple = (0.55, 0.62, 0.72)
    sun: tuple = (0.0, 0.0, 0.0)        # colour x intensity
    sun_azimuth: float = 35.0
    sun_elevation: float = 40.0
    sun_size: float = 0.02              # apparent radius (rad), softens glints
    exposure: float = 0.0
    whitewater: bool = True
    foam: float = 1.0                   # coverage per unit foam density
    spray: float = 0.35                 # optical depth per unit spray density per cell
    bubbles: float = 0.6                # optical depth per unit bubble density per cell
    foam_color: tuple = (0.92, 0.94, 0.95)
    foam_scale: float = 0.012           # m, cells of the foam lace
    foam_lace: float = 0.6
    droplets: float = 1.0
    droplet_size: float = 0.003         # m
    sheets: float = 0.6                 # anisotropic stretch of the particle kernels
    caustics: float = 1.0
    colliders_look: str = 'holdout'     # 'holdout' (in the footage) or 'shaded' (grey stand-ins)
    environment: str = ''               # lat-long HDRI file
    env_rotation: float = 0.0           # degrees
    env_strength: float = 1.0
    env_sun: bool = False
    wind: tuple = (0.0, 0.0, 0.0)       # m/s, fire-local
    step_out: float = 0.35              # surface cells
    step_in: float = 0.6
    max_steps: int = 640
    events: int = 6


@dataclass
class LiquidView:
    """What the renderer needs from a liquid frame (live or cached)."""
    packed: object      # Buffer of packed particles
    count: int
    wet: object         # Texture2D (nx, nz) r32float
    dims: tuple
    h: float
    origin: tuple
    ppc: int
    ww: object = None   # Buffer of packed whitewater particles
    ww_count: int = 0
    open: tuple = (True, True, False)   # sides, top, bottom open (False: solid)
    colliders: tuple = ()               # ColliderGPU list at this frame
    meshes: object = None               # MeshLibrary holding the mesh atlas
    level: float = 0.0                  # open water level (m), 0 = off
    level_blend: float = 8.0            # cells over which the box's surface blends into the open water


class LiquidRenderer:
    def __init__(self, gpu: GPU, renderer):
        self.gpu = gpu
        self.renderer = renderer
        g = gpu
        self.k_count = g.kernel('liq_surf_count.wgsl', ['rbuf', 'buf'], workgroup=(64, 1, 1))
        self.k_deep = g.kernel('liq_surf_deep.wgsl', ['rbuf', 'buf'])
        self.k_splat = g.kernel('liq_surf_splat.wgsl', ['rbuf', 'buf', 'rbuf'], workgroup=(64, 1, 1))
        self.k_resolve = g.kernel('liq_surf_resolve.wgsl', ['rbuf', 'st3d:rgba16float:w', 'st3d:rgba16float:w', 'rbuf'])
        self.k_ww = g.kernel('liq_surf_ww.wgsl', ['rbuf', 'buf'], workgroup=(64, 1, 1))
        self.k_blur = g.kernel('liq_surf_blur.wgsl', ['utex3d', 'st3d:rgba16float:w'])
        self.k_flatten = g.kernel('liq_surf_flatten.wgsl', ['tex3d', 'utex3d', 'smp', 'st3d:rgba16float:w'])
        self.k_march = g.kernel('liq_march.wgsl', ['tex3d', 'tex2d', 'utex2d', 'smp', 'st2d:rgba16float:w',
                                                   'st2d:rgba16float:w', 'st2d:rgba16float:w', 'tex3d', 'utex3d',
                                                   'tex2d', 'tex2d', 'smp'], workgroup=(8, 8, 1))
        self.k_caus = g.kernel('liq_caustics.wgsl', ['tex3d', 'smp', 'buf'], workgroup=(8, 8, 1))
        self.k_caus_res = g.kernel('liq_caustics_resolve.wgsl', ['rbuf', 'st2d:rgba16float:w'], workgroup=(8, 8, 1))
        self._one = g.texture2d(1, 1, 'rgba16float', 'liq-one')
        g.upload(self._one, np.ones((1, 1, 4), np.float16))
        self._no_env = g.texture2d(1, 1, 'rgba16float', 'liq-no-env')
        g.upload(self._no_env, np.zeros((1, 1, 4), np.float16))
        self._no_atlas = g.texture3d((1, 1, 1), 'r32float', 'liq-no-atlas')
        g.upload(self._no_atlas, np.full((1, 1, 1, 1), 1.0e6, np.float32))
        self.caus = None
        self.caus_tex = self._one
        self._caus_acc = None
        self._caus_dims = None
        self.env_tex = None
        self._env_key = None
        self.env_info = None
        self.surf = None
        self._surf = []
        self.nf = None
        self.acc = None
        self._acc_nodes = 0
        self.occ = None
        self.deep = None
        self._cells = 0
        self.cache_buf = None
        self._cache_cap = 0
        self.ww_buf = None
        self._ww_cap = 0
        self.ww_tex = None
        self._black = g.texture2d(4, 4, 'rgba8unorm', 'liq-black')
        g.upload(self._black, np.zeros((4, 4, 4), np.uint8))
        self.sig = None
        self._drops_pipeline()

    def _drops_pipeline(self):
        """Spray droplets: streaks drawn over the traced image (premultiplied over the beauty, their
        glints added to the emission that feeds the bloom)."""
        dev = self.gpu.device
        vis = SS.VERTEX | SS.FRAGMENT
        self._drop_layout = dev.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': vis, 'buffer': {'type': 'read-only-storage'}},
            {'binding': 1, 'visibility': vis, 'texture': {'sample_type': 'float', 'view_dimension': '2d'}},
        ])
        module = dev.create_shader_module(code=load_wgsl('liq_drops.wgsl'), label='liq_drops')
        over = {'color': {'src_factor': 'one', 'dst_factor': 'one-minus-src-alpha', 'operation': 'add'},
                'alpha': {'src_factor': 'one', 'dst_factor': 'one-minus-src-alpha', 'operation': 'add'}}
        add = {'color': {'src_factor': 'one', 'dst_factor': 'one', 'operation': 'add'},
               'alpha': {'src_factor': 'zero', 'dst_factor': 'one', 'operation': 'add'}}
        self._drop_pipe = dev.create_render_pipeline(
            layout=dev.create_pipeline_layout(bind_group_layouts=[self._drop_layout, self.gpu.arena.layout]),
            vertex={'module': module, 'entry_point': 'vs', 'buffers': []},
            fragment={'module': module, 'entry_point': 'fs', 'targets': [
                {'format': 'rgba16float', 'blend': over}, {'format': 'rgba16float', 'blend': add}]},
            primitive={'topology': 'triangle-strip'}, label='liq-drops')
        self._drop_bg = None
        self._drop_key = None

    # -- resources -------------------------------------------------------------------------------

    def upload_particles(self, packed: np.ndarray):
        """A buffer holding cached packed particles (for re-rendering cached frames)."""
        n = max(1, len(packed))
        if self.cache_buf is None or self._cache_cap < n:
            if self.cache_buf is not None:
                self.cache_buf.destroy()
            self._cache_cap = int(n * 1.25) + 1024
            self.cache_buf = self.gpu.buffer(self._cache_cap * PACKED_BYTES, 'liq-cached-particles')
        if len(packed):
            self.gpu.write_buffer(self.cache_buf, np.ascontiguousarray(packed, np.uint32))
        return self.cache_buf

    def upload_whitewater(self, packed: np.ndarray):
        n = max(1, len(packed))
        if self.ww_buf is None or self._ww_cap < n:
            if self.ww_buf is not None:
                self.ww_buf.destroy()
            self._ww_cap = int(n * 1.25) + 1024
            self.ww_buf = self.gpu.buffer(self._ww_cap * WW_PACKED_BYTES, 'liq-cached-whitewater')
        if len(packed):
            self.gpu.write_buffer(self.ww_buf, np.ascontiguousarray(packed, np.uint32))
        return self.ww_buf

    def _ensure_surface(self, nf):
        if nf == self.nf:
            return
        for t in self._surf:
            t.destroy()
        self._surf = [self.gpu.texture3d(nf, 'rgba16float', f'liq-surface{i}') for i in range(2)]
        self.ww_tex = self.gpu.texture3d(nf, 'rgba16float', 'liq-whitewater')
        self._surf.append(self.ww_tex)
        self.nf = nf

    def _ensure_acc(self, nodes):
        if nodes <= self._acc_nodes:
            return
        if self.acc is not None:
            self.acc.destroy()
        self.acc = self.gpu.buffer(nodes * ACC_BYTES, 'liq-surface-acc')
        self._acc_nodes = nodes

    @staticmethod
    def surface_dims(dims, f):
        return tuple(max(8, int(round(d * f))) for d in dims)

    def memory_bytes(self, dims, f):
        nf = self.surface_dims(dims, f)
        nodes = nf[0] * nf[1] * nf[2]
        return nodes * 24 + min(nodes, SLAB_NODES) * ACC_BYTES

    # -- surface ----------------------------------------------------------------------------------

    def build(self, b, view: LiquidView, look: WaterLook):
        """Signed distance and velocity of the liquid on the surface grid, from the particles."""
        n = view.dims
        f = max(0.5, float(look.surface_res))
        nf = self.surface_dims(n, f)
        self._ensure_surface(nf)
        fx = nf[0] / n[0]
        spacing = 1.0 / max(view.ppc, 1) ** (1.0 / 3.0)
        r_cells = max(look.radius, 0.3) * spacing
        R_cells = look.influence * r_cells
        bulk = view.ppc * (4.0 / 3.0 * math.pi * R_cells ** 3) * 0.152
        slab = max(1, min(nf[2], SLAB_NODES // (nf[0] * nf[1])))
        self._ensure_acc(nf[0] * nf[1] * slab)
        count = int(view.count)
        # the deep inside of the liquid cannot affect the surface: find it and skip its particles
        cells = int(np.prod(n))
        if cells != self._cells:
            for bf in (self.occ, self.deep):
                if bf is not None:
                    bf.destroy()
            self.occ = self.gpu.buffer(cells * 4, 'liq-surface-occupancy')
            self.deep = self.gpu.buffer(cells * 4, 'liq-surface-deep')
            self._cells = cells
        reach = max(1, math.ceil(R_cells))
        b.clear_buffer(self.occ)
        if count:
            b.run(self.k_count, [view.packed, self.occ], Uniforms().v4(*n, count), groups=groups_1d(count))
        b.run(self.k_deep, [self.occ, self.deep], Uniforms().v4(*n, reach)
              .v4(math.ceil(0.75 * view.ppc), *(0.0 if o else 1.0 for o in view.open)), n)
        dst = self._surf[0]
        for z0 in range(0, nf[2], slab):
            z1 = min(nf[2], z0 + slab)
            u = (Uniforms().v4(*n, count).v4(*nf, r_cells * fx).v4(z0, z1, R_cells * fx, bulk).v4(reach))
            b.clear_buffer(self.acc, 0, nf[0] * nf[1] * (z1 - z0) * ACC_BYTES)
            if count:
                b.run(self.k_splat, [view.packed, self.acc, self.deep], u, groups=groups_1d(count))
            if view.ww_count and look.whitewater:
                uw = Uniforms().v4(*n, view.ww_count).v4(*nf, 0).v4(z0, z1, max(1.0, 0.5 * fx))
                b.run(self.k_ww, [view.ww, self.acc], uw, groups=groups_1d(view.ww_count))
            b.run(self.k_resolve, [self.acc, dst, self.ww_tex, self.deep], u, (nf[0], nf[1], z1 - z0))
        src = dst
        k = 0
        for _ in range(max(0, int(look.smoothing))):
            k += 1
            out = self._surf[k % 2]
            b.run(self.k_blur, [src, out], Uniforms().v4(*nf, 0), nf)
            src = out
        for _ in range(max(0, int(look.calm))):
            k += 1
            out = self._surf[k % 2]
            b.run(self.k_flatten, [src, self.ww_tex, self.gpu.linear, out], Uniforms().v4(*nf, 1.5).v4(1.0), nf)
            src = out
        self.surf = src
        self.fscale = fx

    # -- environment and caustics -------------------------------------------------------------------

    def environment(self, look: WaterLook):
        """Load (or reuse) the HDRI. Returns None, or (sky ambient colour, sun azimuth, elevation,
        colour) measured from it, rotation included."""
        path = look.environment
        if not path:
            self.env_info = None
            return None
        import os
        try:
            key = (path, os.path.getmtime(path))
        except OSError:
            self.env_info = None
            return None
        if key != self._env_key:
            from ..io.hdri import brightest, load_hdri, sky_average
            img = load_hdri(path)
            if self.env_tex is not None:
                self.env_tex.destroy()
            h, w = img.shape[:2]
            self.env_tex = self.gpu.texture2d(w, h, 'rgba16float', 'liq-hdri')
            rgba = np.concatenate([img, np.ones((h, w, 1), np.float32)], -1)
            self.gpu.upload(self.env_tex, np.minimum(rgba, 6.0e4).astype(np.float16))
            az, el, colour = brightest(img)
            self.env_info = (sky_average(img), az, el, colour)
            self._env_key = key
        sky, az, el, colour = self.env_info
        return tuple(x * look.env_strength for x in sky), az + look.env_rotation, el, colour

    def caustics(self, b, view: LiquidView, look: WaterLook, fire: cam.FireXform):
        """Trace key-light photons through the liquid onto the ground (the map the renderer reads
        for caustics and shadows). Uses the surface build() left."""
        sd = cam.sun_direction(look.sun_azimuth, look.sun_elevation)
        if look.caustics <= 0.0 or max(look.sun) <= 0.0 or sd[1] < 0.03 or view.count == 0:
            self.caus_tex = self._one
            return
        nf = self.nf
        dims = (nf[0], nf[2])
        if dims != self._caus_dims:
            for x in (self.caus, self._caus_acc):
                if x is not None:
                    x.destroy()
            self.caus = self.gpu.texture2d(dims[0], dims[1], 'rgba16float', 'liq-caustics')
            self._caus_acc = self.gpu.buffer(dims[0] * dims[1] * 4, 'liq-caustics-acc')
            self._caus_dims = dims
        ry = cam.rot_y(math.radians(fire.yaw))
        sg = ry.T @ sd
        sg = sg / np.linalg.norm(sg)
        absorb = -np.log(np.clip(np.asarray(look.color, float), 1e-4, 1.0)) / max(look.clarity, 1e-3)
        k = 2
        n = view.dims
        lvl_on = view.level > 0.0 and view.open[0]
        u = (Uniforms().v4(*n, view.h).v4(*nf, self.fscale).v4(dims[0], dims[1], k)
             .v3(sg, look.ior).v4(*absorb, look.murk)
             .v4(view.level / view.h if lvl_on else -1.0, 1.0 if lvl_on else 0.0, 0.0, view.level_blend))
        b.clear_buffer(self._caus_acc)
        b.run(self.k_caus, [self.surf, self.gpu.linear, self._caus_acc], u, (dims[0] * k, dims[1] * k, 1))
        b.run(self.k_caus_res, [self._caus_acc, self.caus], Uniforms().v4(dims[0], dims[1], k * k), (dims[0], dims[1], 1))
        self.caus_tex = self.caus

    # -- tracing ----------------------------------------------------------------------------------

    def drops(self, b, view: LiquidView, camstate: cam.CameraState, fire: cam.FireXform, look: WaterLook, size,
              jitter=(0.0, 0.0), shutter=0.0):
        """Draw the spray particles as droplets over what march() left (call right after it)."""
        if look.droplets <= 0.0 or not look.whitewater or not view.ww_count or view.ww is None:
            return
        r = self.renderer
        key = (id(view.ww), view.ww.size, id(r.aux))
        if self._drop_bg is None or self._drop_key != key:
            self._drop_bg = self.gpu.device.create_bind_group(layout=self._drop_layout, entries=[
                {'binding': 0, 'resource': {'buffer': view.ww.buf, 'offset': 0, 'size': view.ww.size}},
                {'binding': 1, 'resource': r.aux.view},
            ])
            self._drop_key = key
        w, h = size
        P = camstate.proj
        focal_px = 0.5 * h * math.sqrt(abs(P[0, 0] * P[1, 1] - P[0, 1] * P[1, 0]) * camstate.aspect)
        gain = 2.0 ** look.exposure
        sd = cam.sun_direction(look.sun_azimuth, look.sun_elevation)
        u = (Uniforms().m4(camstate.view_proj).m4(fire.local_to_world())
             .v4(*view.dims, view.h).v4(*view.origin, look.droplets)
             .v4(w, h, shutter, focal_px)
             .v4(look.droplet_size, 0.8, look.ior, 0.0)
             .v4(*camstate.eye, 0.0)
             .v4(*(np.asarray(look.sky) * gain), 0.0)
             .v4(*(np.asarray(look.sun) * gain), 0.0)
             .v3(sd, 0.0)
             .v4(*jitter))
        off = b.uniform_offset(u)
        rp = b.render_pass(color_attachments=[
            {'view': r.beauty.view, 'load_op': 'load', 'store_op': 'store'},
            {'view': r.emit.view, 'load_op': 'load', 'store_op': 'store'}])
        rp.set_pipeline(self._drop_pipe)
        rp.set_bind_group(0, self._drop_bg)
        rp.set_bind_group(1, self.gpu.arena.group, [off])
        rp.draw(4, view.ww_count)
        rp.end()

    def march(self, b, view: LiquidView, camstate: cam.CameraState, fire: cam.FireXform, look: WaterLook, comp, size,
              plate=None, plate_fit=(1.0, 1.0), plate_transform=0, plate_gain=1.0, jitter=(0.0, 0.0), seed=0.0,
              shutter=0.0, ground=True, time=0.0):
        w, h = size
        r = self.renderer
        r._ensure_fire(w, h)
        w2g = cam.world_to_grid(fire, view.origin, view.h)
        g2w = np.linalg.inv(w2g)
        fwd = -camstate.view[2, :3]
        fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
        gain = 2.0 ** look.exposure
        absorb = -np.log(np.clip(np.asarray(look.color, float), 1e-4, 1.0)) / max(look.clarity, 1e-3)
        sd = cam.sun_direction(look.sun_azimuth, look.sun_elevation)
        n = view.dims
        u = (Uniforms().m4(camstate.inv_view_proj).m4(camstate.view_proj).m4(w2g).m4(g2w)
             .v4(*n, view.h).v4(*self.nf, self.fscale)
             .v4(w, h, *jitter).v4(seed, shutter, 1.0 if shutter != 0.0 else 0.0, time)
             .v4(1.0 if plate is not None else 0.0, plate_transform, *plate_fit)
             .v4(*comp.bg, 1.0 if comp.bg_checker else 0.0)
             .v4(*camstate.eye, look.backdrop)
             .v4(*fwd, 1.0 if ground else 0.0)
             .v4(look.ior, look.roughness, look.reflection, look.reflect_footage)
             .v4(*absorb, look.murk)
             .v4(*look.murk_color, plate_gain)
             .v4(*(np.asarray(look.sky) * gain), max(8, h / 60))
             .v4(*(np.asarray(look.sun) * gain), look.sun_size)
             .v3(sd, look.shadow)
             .v4(look.ripple, look.ripple_freq, look.ripple_speed)
             .v4(look.wet_darken, look.wet_gloss)
             .v4(look.max_steps, look.step_out, look.step_in, look.events)
             .v4(look.spray * (1.0 - 0.7 * min(1.0, look.droplets)), look.foam, look.bubbles,
                 1.0 if (look.whitewater and view.ww_count) else 0.0)
             .v4(*look.foam_color, look.foam_lace))
        lvl_on = view.level > 0.0 and view.open[0]
        env_on = self.env_tex is not None and bool(look.environment)
        u.v4(*view.origin, 1.0 if look.colliders_look == 'shaded' else 0.0)
        u.v4(view.level / view.h if lvl_on else -1.0, 1.0 if lvl_on else 0.0,
             look.caustics if self.caus_tex is not self._one else 0.0, view.level_blend)
        u.v4(1.0 if env_on else 0.0, math.radians(look.env_rotation), look.env_strength * gain, look.foam_scale)
        pack_colliders(u, view.colliders, view.meshes)
        atlas = view.meshes.atlas if view.meshes is not None else self._no_atlas
        b.run(self.k_march, [self.surf, plate if plate is not None else self._black, view.wet, self.gpu.linear,
                             r.beauty, r.emit, r.aux, self.ww_tex, atlas, self.caus_tex,
                             self.env_tex if env_on else self._no_env, self.gpu.repeat], u, (w, h, 1))
