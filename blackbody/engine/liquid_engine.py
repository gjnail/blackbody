"""The engine's liquid path: preparing, stepping, caching, rendering and reporting a liquid scene.

Engine dispatches here when the scene's kind is 'liquid'. Frames are cached as packed particles
(16 bytes each) plus the ground wetness, and the surface is rebuilt from them at render time, so
changing the look re-renders any cached frame without re-simulating.
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import camera as cam
from .gpu import Uniforms
from .liquid import LiquidSolver
from .liquid_render import LiquidRenderer, LiquidView
from .renderer import INPUT_TRANSFORMS


def _halton(i, b):
    f, r = 1.0, 0.0
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r


class LiquidEngine:
    """Mixin for Engine (which owns gpu, solver, renderer, cache and the accumulation kernels)."""

    liquid = None
    liquid_r = None
    kind = 'fire'
    _rwet = None

    def _liq(self):
        if self.liquid is None:
            self.liquid = LiquidSolver(self.gpu, self.solver.meshes)
            self.liquid_r = LiquidRenderer(self.gpu, self.renderer)
        return self.liquid

    def _prepare_liquid(self, scene, final=False, soft=False):
        sig = scene.sim_signature(final)
        dims, h, origin = scene.sim_layout(final)
        L = self._liq()
        meshes = [scene.mesh_path(d['mesh']) for d in scene.emitters + scene.colliders
                  if d['enabled'] and d['shape'] == 'mesh' and d['mesh']]
        if self.solver.set_meshes(meshes, scene.data['domain']['mesh_resolution']):
            L.colliders = None  # the atlas changed: rewrite the solid distance field
        changed = L.configure(dims, h, origin, scene.liquid_capacity(final), scene.whitewater_capacity())
        L._prm = scene.liquid_params(scene.start)
        L.set_colliders(scene.colliders_gpu(scene.start))
        if self.kind != 'liquid':
            self.kind = 'liquid'
            changed = True
        if changed or final != self.final or self.sig is None:
            self.sig = sig
            self.final = final
            self.cache.clear()
            self.reset()
            return 'reset'
        if sig != self.sig:
            self.sig = sig
            self.cache.clear()
            if soft and self.sim_frame is not None:
                return 'soft'
            self.reset()
            return 'reset'
        return None

    def _reset_liquid(self):
        self._filled = set()
        if self.liquid is not None and self.liquid.dims is not None:
            self.liquid.reset()

    def _step_liquid(self, scene, frame):
        t0 = time.perf_counter()
        fdt = scene.v('domain', 'time_scale', frame) / scene.fps
        d = scene.data['domain']
        prm = scene.liquid_params(frame)
        if frame < scene.start and scene.data['liquid']['settle']:
            prm.damping = 6.0   # still liquid starts still: calm the filling transient before the shot
        L = self.liquid
        L._prm = prm
        # surface tension can demand more substeps than the domain allows: stability comes first
        hi = min(40, max(d['substeps_max'], L.capillary_substeps(fdt)))
        n = L.substeps_for(fdt, cfl=d['cfl'], lo=d['substeps_min'], hi=hi)
        moving = scene.colliders_animated()
        filled = getattr(self, '_filled', None)
        if filled is None:
            filled = self._filled = set()
        with self.gpu.batch() as b:
            for i in range(n):
                fs = frame - 1 + (i + 0.5) / n
                srcs = scene.sources_gpu(fs, filled)
                cols = scene.colliders_gpu(fs) if moving else None
                L.step(b, fdt / n, prm, srcs, cols)
            L.pack(b)
        L.measure()
        self.sim_frame = frame
        self.last_substeps = n
        self.last_step_ms = (time.perf_counter() - t0) * 1000.0

    def _snapshot_liquid(self):
        return {'liq': self.liquid.read_packed(), 'wet': self.liquid.read_wet(), 'ww': self.liquid.read_ww_packed()}

    def _liquid_view(self, scene, frame):
        L = self.liquid
        ppc = int(scene.data['liquid']['ppc'])
        d = scene.data['domain']
        q = scene.data['liquid']
        bounds = (bool(d['open_sides']), bool(d['open_top']), not d['ground'])
        extra = dict(colliders=tuple(scene.colliders_gpu(frame)), meshes=self.solver.meshes,
                     level=q['water_level'] if d['open_sides'] else 0.0, level_blend=q['level_absorb'])
        if self.sim_frame == frame:
            return LiquidView(L.packed, L.packed_count, L.WET, L.dims, L.h, L.origin, ppc, L.wpacked, L.ww_count, bounds,
                              **extra), True
        entry = self.cache.get(frame)
        if entry is None or 'liq' not in entry:
            return None, False
        buf = self.liquid_r.upload_particles(entry['liq'])
        nx, nz = L.dims[0], L.dims[2]
        if self._rwet is None or self._rwet.size[:2] != (nx, nz):
            if self._rwet is not None:
                self._rwet.destroy()
            self._rwet = self.gpu.texture2d(nx, nz, 'r32float', 'cached-wet')
        self.gpu.upload(self._rwet, entry['wet'].astype(np.float32)[..., None])
        ww = entry.get('ww')
        wbuf = self.liquid_r.upload_whitewater(ww) if ww is not None and len(ww) else None
        return LiquidView(buf, len(entry['liq']), self._rwet, L.dims, L.h, L.origin, ppc, wbuf,
                          len(ww) if wbuf is not None else 0, bounds, **extra), False

    def _render_liquid(self, scene, frame, out_size, mode='composite', final=False, samples=1, motion_blur=False,
                       fire_scale=1.0, plate=None, plate_fit=(1.0, 1.0), seed=None):
        t0 = time.perf_counter()
        vol, live = self._liquid_view(scene, frame)
        if vol is None:
            raise RuntimeError(f'frame {frame} is neither simulated nor cached')
        W, H = out_size
        fw, fh = max(8, int(round(W * fire_scale))), max(8, int(round(H * fire_scale)))
        spec, fire = scene.camera(frame)
        cs = cam.compute(spec, W / H, fire)
        look = scene.water_look(frame, final)
        if plate is not None and scene.data['lighting'].get('ambient_from_footage', True):
            look.sky = self.footage_ambient(plate, scene, frame)
        env = self.liquid_r.environment(look)
        if env is not None:
            # the HDRI is the set's own light: its sky for the ambient, its sun for the key light
            look.sky = env[0]
            if look.env_sun:
                strength = max(look.sun) or 3.0
                look.sun_azimuth, look.sun_elevation = env[1], env[2]
                look.sun = tuple(c * strength for c in env[3])
        comp = scene.comp(frame, mode)
        shutter = 0.0
        if motion_blur:
            shutter = scene.data['render']['shutter_angle'] / 360.0 / scene.fps * scene.v('domain', 'time_scale', frame)
        r = self.renderer
        if plate is not None:
            r.set_plate(plate)
        elif mode == 'composite':
            r.set_plate(None)
        ptex = r.plate if plate is not None else None
        base_seed = frame * 64 if seed is None else seed
        samples = max(1, int(samples))
        t = scene.seconds(frame)
        ground = scene.data['domain']['ground']
        LR = self.liquid_r

        drop_shutter = scene.data['render']['shutter_angle'] / 360.0 / scene.fps if motion_blur else 0.0

        def march(b, jit, s):
            LR.march(b, vol, cs, fire, look, comp, (fw, fh), plate=ptex, plate_fit=plate_fit,
                     plate_transform=INPUT_TRANSFORMS.get(comp.plate_transform, 0), plate_gain=comp.plate_gain,
                     jitter=jit, seed=s, shutter=shutter, ground=ground, time=t)
            LR.drops(b, vol, cs, fire, look, (fw, fh), jitter=jit, shutter=drop_shutter)

        with self.gpu.batch() as b:
            LR.build(b, vol, look)
            LR.caustics(b, vol, look, fire)
            if samples == 1:
                march(b, (0.0, 0.0), base_seed)
            else:
                self._ensure_acc(fw, fh)
                set0, set1 = self._acc[0:3], self._acc[3:6]
                for i in range(samples):
                    march(b, (_halton(i + 1, 2) - 0.5, _halton(i + 1, 3) - 0.5), base_seed + i)
                    src_in, dst = (set0, set1) if i % 2 == 0 else (set1, set0)
                    b.run(self.k_accum, [r.beauty, r.emit, r.aux, *src_in, *dst], Uniforms().v4(fw, fh, 0, i), (fw, fh, 1))
                last, other = (set1, set0) if (samples - 1) % 2 == 0 else (set0, set1)
                b.run(self.k_accum, [r.beauty, r.emit, r.aux, *last, *other], Uniforms().v4(fw, fh, 1, samples), (fw, fh, 1))
                b.run(self.k_copy, [*other, r.beauty, r.emit, r.aux], Uniforms().v4(fw, fh), (fw, fh, 1))
            r.bloom(b, comp.bloom_radius)
            r.composite(b, (W, H), comp, time=t, frame=frame, plate_fit=plate_fit, liquid=True)
        self.last_render_ms = (time.perf_counter() - t0) * 1000.0
        return cs

    def _stats_liquid(self):
        L = self.liquid
        f = self.liquid_r.surface_dims(L.dims, 2.0) if L.dims else None
        return {
            'gpu': self.gpu.name, 'backend': self.gpu.backend, 'kind': 'liquid', 'dims': L.dims,
            'voxels': int(np.prod(L.dims)) if L.dims else 0, 'cell_mm': L.h * 1000 if L.h else 0,
            'max_speed': L.max_speed, 'substeps': self.last_substeps, 'sim_ms': self.last_step_ms,
            'render_ms': self.last_render_ms, 'cached': len(self.cache.items), 'cache_mb': self.cache.bytes / 1e6,
            'particles': L.count, 'whitewater': L.ww_count, 'embers': 0, 'burning': 0,
            'particle_limit': L.capacity and L.count >= 0.995 * L.capacity,
            'memory_mb': (L.memory_bytes() + (self.liquid_r.memory_bytes(L.dims, 2.0) if f else 0)) / 1e6 if L.dims else 0,
            'mesh_errors': dict(self.solver.meshes.errors),
        }
