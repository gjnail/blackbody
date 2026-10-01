"""The engine's side of cloud scenes (kind 'cloud', engine/cloud.py): set-up, the frame step, the cache and
the render. Mixed into engine.Engine like the liquid's LiquidEngine."""
from __future__ import annotations

import time

import numpy as np

from . import camera as cam
from .cloud import CloudSolver
from .cloud_render import CloudRenderer


class CloudEngine:
    cloud = None
    cloud_r = None
    _cloud_prm = None
    _cloud_tex = None

    def _prepare_cloud(self, scene, final=False, soft=False):
        sig = scene.sim_signature(final)
        dims, h, origin = scene.sim_layout(final)
        if self.cloud is None:
            self.cloud = CloudSolver(self.gpu, self.solver.meshes)
            self.cloud_r = CloudRenderer(self.gpu, self.renderer)
        meshes = [scene.mesh_path(d['mesh']) for d in scene.colliders if d['enabled'] and d['shape'] == 'mesh' and d['mesh']]
        self.solver.set_meshes(meshes, scene.data['domain']['mesh_resolution'])
        scale = float(scene.data['atmosphere']['scale'])
        changed = self.cloud.configure(dims, h, origin, scale)
        self.cloud.set_colliders(scene.colliders_gpu(scene.start))
        self._cloud_prm = scene.cloud_params(scene.start)
        if self.kind != 'cloud':
            self.kind = 'cloud'
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

    def _reset_cloud(self):
        if self.cloud is not None and self.cloud.dims is not None and self._cloud_prm is not None:
            self.cloud.reset(self._cloud_prm)

    def _step_cloud(self, scene, frame):
        t0 = time.perf_counter()
        prm = scene.cloud_params(frame)
        self._cloud_prm = prm
        C = self.cloud
        fdt = float(scene.v('atmosphere', 'time_lapse', frame)) * scene.v('domain', 'time_scale', frame) / scene.fps
        d = scene.data['domain']
        n = C.substeps_for(fdt, cfl=float(d.get('cfl', 1.0)), hi=max(int(d['substeps_max']), 1))
        with self.gpu.batch() as b:
            for _ in range(n):
                C.step(b, fdt / n, prm)
        C.measure()
        self.sim_frame = frame
        self.last_substeps = n
        self.last_step_ms = (time.perf_counter() - t0) * 1000.0

    def _snapshot_cloud(self):
        a, b = self.cloud.read_fields()
        pr = self.gpu.read(self.cloud.PRECIP)
        pr = pr[0] if pr.ndim == 4 else pr
        # what the renderer needs: the condensate (cloud water, rain; ice, snow, graupel) and the rain at the ground
        return {'cloud_a': a[..., 2:4].astype(np.float16), 'cloud_b': b[..., 0:4].astype(np.float16),
                'cloud_precip': pr.astype(np.float16), 'cloud_time': self.cloud.time}

    def _cloud_textures(self, entry):
        """Upload a cached frame's fields for the renderer."""
        C = self.cloud
        g = self.gpu
        dims = C.dims
        if self._cloud_tex is None or self._cloud_tex[0].size != tuple(dims):
            for t in self._cloud_tex or ():
                t.destroy()
            nx, ny, nz = dims
            self._cloud_tex = (g.texture3d(dims, 'rgba32float', 'cloud-cached-a'), g.texture3d(dims, 'rgba32float', 'cloud-cached-b'),
                               g.texture2d(nx, nz, 'rgba32float', 'cloud-cached-precip'))
        ta, tb, tp = self._cloud_tex
        a = entry['cloud_a'].astype(np.float32)
        b = entry['cloud_b'].astype(np.float32)
        A = np.zeros(a.shape[:3] + (4,), np.float32)
        A[..., 2:4] = a
        B = np.zeros(b.shape[:3] + (4,), np.float32)
        B[..., 0:b.shape[-1]] = b
        g.upload(ta, A)
        g.upload(tb, B)
        g.upload(tp, entry['cloud_precip'].astype(np.float32))
        return ta, tb, tp

    def _render_cloud(self, scene, frame, out_size, mode='composite', final=False, samples=1, motion_blur=False,
                      fire_scale=1.0, plate=None, plate_fit=(1.0, 1.0), seed=None):
        t0 = time.perf_counter()
        C = self.cloud
        if self.sim_frame == frame:
            A, B, P = C.A[0], C.B[0], C.PRECIP
        else:
            entry = self.cache.get(frame)
            if entry is None or 'cloud_a' not in entry:
                raise RuntimeError(f'frame {frame} is neither simulated nor cached')
            A, B, P = self._cloud_textures(entry)
        W, H = out_size
        fw, fh = max(8, int(round(W * fire_scale))), max(8, int(round(H * fire_scale)))
        spec, fire = scene.camera(frame)
        cs = cam.compute(spec, W / H, fire)
        look = scene.sky_look(frame, final)
        if plate is not None:
            look.draw_sky = False if not scene.data['sky']['draw_sky'] else look.draw_sky
        comp = scene.comp(frame, mode)
        r = self.renderer
        if plate is not None:
            r.set_plate(plate)
        elif mode == 'composite':
            r.set_plate(None)
        r._ensure_fire(fw, fh)
        base_seed = frame * 64 if seed is None else seed
        w = C.base_info['wind'] if hasattr(C, 'base_info') else ((0.0,), (0.0,))
        mid = len(w[0]) // 3
        wind = (float(w[0][mid]), 0.0, float(w[1][mid]))
        samples = max(1, int(samples))
        step = 0.5 if final else 0.75
        with self.gpu.batch() as b:
            if samples == 1:
                self.cloud_r.render(b, C, A, B, P, cs, fire, look, (fw, fh), seed=base_seed, step=step, wind=wind)
            else:
                # jittered rays averaged (the accumulators the fire and liquid renders use)
                from .gpu import Uniforms
                self._ensure_acc(fw, fh)
                set0, set1 = self._acc[0:3], self._acc[3:6]
                for i in range(samples):
                    self.cloud_r.render(b, C, A, B, P, cs, fire, look, (fw, fh), seed=base_seed + i * 7919, step=step,
                                        wind=wind, light=(i == 0))
                    src_in, dst = (set0, set1) if i % 2 == 0 else (set1, set0)
                    b.run(self.k_accum, [r.beauty, r.emit, r.aux, *src_in, *dst], Uniforms().v4(fw, fh, 0, i), (fw, fh, 1))
                last, other = (set1, set0) if (samples - 1) % 2 == 0 else (set0, set1)
                b.run(self.k_accum, [r.beauty, r.emit, r.aux, *last, *other], Uniforms().v4(fw, fh, 1, samples), (fw, fh, 1))
                b.run(self.k_copy, [*other, r.beauty, r.emit, r.aux], Uniforms().v4(fw, fh), (fw, fh, 1))
            r.defocus(b, comp, cs, fire)
            r.bloom(b, comp.bloom_radius)
            r.composite(b, (W, H), comp, time=scene.seconds(frame), frame=frame, plate_fit=plate_fit, liquid=True,
                        hfov=cs.hfov)
        self.last_render_ms = (time.perf_counter() - t0) * 1000.0
        return cs

    def _stats_cloud(self):
        C = self.cloud
        out = {'gpu': self.gpu.name, 'backend': self.gpu.backend, 'dims': C.dims, 'voxels': int(np.prod(C.dims)) if C.dims else 0,
               'cell_mm': C.h * 1000.0 if C.h else 0.0, 'cell_m': C.h, 'max_speed': C.stats.get('max_speed', 0.0),
               'substeps': self.last_substeps, 'sim_ms': self.last_step_ms, 'render_ms': self.last_render_ms,
               'cached': len(self.cache.items), 'cache_mb': self.cache.bytes / 1e6, 'sky_minutes': C.time / 60.0}
        out.update({'cloud_' + k: v for k, v in C.stats.items()})
        return out
