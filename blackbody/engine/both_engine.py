"""The engine's path for fire and liquid in one box (domain kind 'both'): a hose on a fire, burning
fuel on water, a fire by a fountain.

Both solvers run on the same grid. Each substep the liquid moves first, its share of every cell is
written into the fire solver's water field (where the fire is put out and steam is made), liquid in
gas hotter than boiling slowly boils away, then the fire steps.

Rendering, per sample: the fire over the footage becomes the backdrop the liquid refracts and
reflects (flames behind a sheet of water show through it), the liquid is traced, then the fire is
marched again only up to the liquid's surface and laid over it. The composite then treats the
result as one element, with the liquid's wet ground, shadows and caustics on the footage.
"""
from __future__ import annotations

import time

import numpy as np

from . import camera as cam
from .gpu import TU, Uniforms
from .liquid_float import Floats
from .renderer import INPUT_TRANSFORMS


def _halton(i, b):
    f, r = 1.0, 0.0
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r


class BothEngine:
    """Mixin for Engine, alongside LiquidEngine."""

    _both = None
    _both_size = None
    EVAPORATION = 0.6   # 1/s: share of the liquid in a cell at flame heat that boils away per second

    # -- simulation ------------------------------------------------------------------------------

    def _prepare_both(self, scene, final=False, soft=False):
        sig = scene.sim_signature(final)
        dims, h, origin = scene.sim_layout(final)
        L = self._liq()
        meshes = [scene.mesh_path(d['mesh']) for d in scene.emitters + scene.colliders
                  if d['enabled'] and d['shape'] == 'mesh' and d['mesh']]
        if self.solver.set_meshes(meshes, scene.data['domain']['mesh_resolution']):
            L.colliders = None
        changed = self.solver.configure(dims, h, origin, scene.features(), scene.data['render']['upres'] if final else 1)
        self.solver.set_colliders(scene.colliders_gpu(scene.start))
        changed = L.configure(dims, h, origin, scene.liquid_capacity(final), scene.whitewater_capacity()) or changed
        L._prm = scene.liquid_params(scene.start)
        L.set_colliders(scene.colliders_gpu(scene.start))
        if self.kind != 'both':
            self.kind = 'both'
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

    def _reset_both(self):
        self.solver.reset()
        self.embers.reset()
        self._reset_liquid()

    def _step_both(self, scene, frame):
        t0 = time.perf_counter()
        fdt = scene.v('domain', 'time_scale', frame) / scene.fps
        d = scene.data['domain']
        prm = scene.solver_params(frame)
        ep = scene.ember_params(frame)
        look = scene.look(frame)
        lprm = scene.liquid_params(frame)
        if frame < scene.start and scene.data['liquid']['settle']:
            lprm.damping = 6.0
        L = self.liquid
        L._prm = lprm
        n_fire = self.solver.substeps_for(fdt, cfl=d['cfl'], lo=d['substeps_min'], hi=d['substeps_max'])
        hi = min(40, max(d['substeps_max'], L.capillary_substeps(fdt)))
        n = max(n_fire, L.substeps_for(fdt, cfl=d['cfl'], lo=d['substeps_min'], hi=hi))
        cam_g = self._cam_grid(scene, frame) if ep.enabled else (0.0, 0.0, 0.0)
        if not self._floats_ready:
            self._floats = Floats.from_scene(scene, frame - 1, self.solver.meshes)
            self._floats_ready = True
        floats = self._floats
        moving = scene.colliders_animated() or bool(floats)
        filled = getattr(self, '_filled', None)
        if filled is None:
            filled = self._filled = set()
        boil = scene.boil_temp()
        with self.gpu.batch() as b:
            if floats:
                L.clear_float(b)
            for i in range(n):
                fs = frame - 1 + (i + 0.5) / n
                dt = fdt / n
                ov = floats.overrides(fdt * (i + 0.5) / n) if floats else None
                cols = scene.colliders_gpu(fs, ov) if moving else None
                L.step(b, dt, lprm, scene.sources_gpu(fs, filled), cols)
                if floats:
                    L.float_forces(b, floats.regions(scene), i, dt)
                if self.solver.water is not None:
                    L.water_into(b, self.solver)
                L.evaporate(b, self.solver, boil, self.EVAPORATION, dt)
                self.solver.step(b, dt, prm, scene.emitters_gpu(fs), cols)
                if ep.enabled:
                    ember_ems = scene.emitters_gpu(fs, embers_only=True)
                    if ember_ems or self.embers.count:
                        self.embers.step(b, self.solver, ep, ember_ems, dt, cam_g, look.smoke_density, look.ambient_k)
            L.pack(b)
        self.solver.measure()
        L.measure()
        if floats:
            floats.step(L.read_float(len(floats.bodies), n), fdt, n, L.h, lprm.rho, L.origin, L.dims,
                        ground=bool(d['ground']), walls=not d['open_sides'])
        self.sim_frame = frame
        self.last_substeps = n
        self.last_step_ms = (time.perf_counter() - t0) * 1000.0

    def _stats_both(self, stats):
        L = self.liquid
        stats.update(kind='both', particles=L.count, whitewater=L.ww_count,
                     particle_limit=bool(L.capacity and L.count >= 0.995 * L.capacity),
                     memory_mb=stats.get('memory_mb', 0.0) + (L.memory_bytes() / 1e6 if L.dims else 0.0))
        return stats

    # -- rendering -------------------------------------------------------------------------------

    def _ensure_both(self, w, h):
        g = self.gpu
        if self._both is None:
            self.k_bfp = g.kernel('liq_both_fp.wgsl', ['tex2d', 'tex2d', 'smp', 'st2d:rgba16float:w'], workgroup=(8, 8, 1))
            self.k_bmerge = g.kernel('liq_both_merge.wgsl', ['utex2d'] * 6 + ['st2d:rgba16float:w'] * 3, workgroup=(8, 8, 1))
        if self._both_size == (w, h):
            return
        for t in (self._both or {}).values():
            t.destroy()
        usage = TU.TEXTURE_BINDING | TU.STORAGE_BINDING | TU.COPY_DST | TU.COPY_SRC
        self._both = {k: g.texture2d(w, h, 'rgba16float', f'both-{k}', usage) for k in ('fp', 'lb', 'le', 'la', 'fb', 'fe', 'fa')}
        self._both_size = (w, h)

    def _render_both(self, scene, frame, out_size, mode='composite', final=False, samples=1, motion_blur=False,
                     fire_scale=1.0, plate=None, plate_fit=(1.0, 1.0), seed=None):
        t0 = time.perf_counter()
        vol, live = self.volume_for(scene, frame)
        lv, _ = self._liquid_view(scene, frame)
        if vol is None or lv is None:
            raise RuntimeError(f'frame {frame} is neither simulated nor cached')
        W, H = out_size
        fw, fh = max(8, int(round(W * fire_scale))), max(8, int(round(H * fire_scale)))
        spec, fire = scene.camera(frame)
        cs = cam.compute(spec, W / H, fire)
        look = scene.look(frame, final)
        look.vapour = bool(vol.aux) and self.solver.features.get('vapour', False)
        look.colourant = bool(vol.chem)
        wlook = scene.water_look(frame, final)
        if plate is not None and scene.data['lighting'].get('ambient_from_footage', True):
            look.ambient = self.footage_ambient(plate, scene, frame)
            wlook.sky = look.ambient
        env = self.liquid_r.environment(wlook)
        if env is not None:
            wlook.sky = env[0]
            if wlook.env_sun:
                strength = max(wlook.sun) or 3.0
                wlook.sun_azimuth, wlook.sun_elevation = env[1], env[2]
                wlook.sun = tuple(c * strength for c in env[3])
        comp = scene.comp(frame, mode)
        ep = scene.ember_params(frame)
        t = vol.time
        shutter = lshutter = drop_shutter = 0.0
        if motion_blur:
            drop_shutter = scene.data['render']['shutter_angle'] / 360.0 / scene.fps
            lshutter = drop_shutter * scene.v('domain', 'time_scale', frame)
            if live:
                shutter = lshutter
        r = self.renderer
        if plate is not None:
            r.set_plate(plate)
        elif mode == 'composite':
            r.set_plate(None)
        ptex = r.plate if plate is not None else None
        base_seed = frame * 64 if seed is None else seed
        samples = max(1, int(samples))
        surfaces = self.surfaces_for(scene, frame, vol, comp)
        ground = scene.data['domain']['ground']
        LR = self.liquid_r
        r._ensure_fire(fw, fh)
        self._ensure_both(fw, fh)
        T = self._both
        fp_u = (Uniforms().v4(fw, fh).v4(1.0 if ptex is not None else 0.0, INPUT_TRANSFORMS.get(comp.plate_transform, 0), *plate_fit)
                .v4(comp.plate_gain, comp.fire_gain, comp.smoke_opacity).v4(*comp.bg))
        size = (fw, fh, 1)

        def one(b, jit, s):
            # the fire, whole: behind the liquid it is seen through it
            r.march(b, vol, cs, fire, look, (fw, fh), jitter=jit, seed=s, shutter=shutter, ground=ground, time=t,
                    surfaces=surfaces)
            b.run(self.k_bfp, [r.beauty, ptex if ptex is not None else r._black, self.gpu.linear, T['fp']], fp_u, size)
            LR.march(b, lv, cs, fire, wlook, comp, (fw, fh), plate=T['fp'], plate_fit=(1.0, 1.0),
                     plate_transform=INPUT_TRANSFORMS['linear'], plate_gain=1.0, jitter=jit, seed=s, shutter=lshutter,
                     ground=ground, time=t)
            LR.drops(b, lv, cs, fire, wlook, (fw, fh), jitter=jit, shutter=drop_shutter)
            for src, key in ((r.beauty, 'lb'), (r.emit, 'le'), (r.aux, 'la')):
                b.copy_texture(src, T[key], size)
            # the fire in front of the liquid, over it
            r.march(b, vol, cs, fire, look, (fw, fh), jitter=jit, seed=s, shutter=shutter, ground=ground, time=t,
                    surfaces=surfaces, limit=T['la'])
            for src, key in ((r.beauty, 'fb'), (r.emit, 'fe'), (r.aux, 'fa')):
                b.copy_texture(src, T[key], size)
            b.run(self.k_bmerge, [T['fb'], T['fe'], T['fa'], T['lb'], T['le'], T['la'], r.beauty, r.emit, r.aux],
                  Uniforms().v4(fw, fh), size)

        with self.gpu.batch() as b:
            r.light(b, vol, look, fire, t)
            LR.build(b, lv, wlook)
            LR.caustics(b, lv, wlook, fire)
            if samples == 1:
                one(b, (0.0, 0.0), base_seed)
            else:
                self._ensure_acc(fw, fh)
                self._ensure_acc2(fw, fh)
                set0, set1 = self._acc[0:3], self._acc[3:6]
                sur0, sur1 = self._acc2[0:3], self._acc2[3:6]
                srcs2 = [r.surf, r.mask, r.mask]
                for i in range(samples):
                    one(b, (_halton(i + 1, 2) - 0.5, _halton(i + 1, 3) - 0.5), base_seed + i)
                    src_in, dst = (set0, set1) if i % 2 == 0 else (set1, set0)
                    b.run(self.k_accum, [r.beauty, r.emit, r.aux, *src_in, *dst], Uniforms().v4(fw, fh, 0, i), size)
                    s_in, s_dst = (sur0, sur1) if i % 2 == 0 else (sur1, sur0)
                    b.run(self.k_accum, [*srcs2, *s_in, *s_dst], Uniforms().v4(fw, fh, 0, i), size)
                last, other = (set1, set0) if (samples - 1) % 2 == 0 else (set0, set1)
                b.run(self.k_accum, [r.beauty, r.emit, r.aux, *last, *other], Uniforms().v4(fw, fh, 1, samples), size)
                b.run(self.k_copy, [*other, r.beauty, r.emit, r.aux], Uniforms().v4(fw, fh), size)
                slast, sother = (sur1, sur0) if (samples - 1) % 2 == 0 else (sur0, sur1)
                b.run(self.k_accum, [*srcs2, *slast, *sother], Uniforms().v4(fw, fh, 1, samples), size)
                b.run(self.k_copy, [*sother, r.surf, r.mask, self._acc2_spare], Uniforms().v4(fw, fh), size)
            if ep.enabled and self.embers.count:
                self.embers.draw(b, r, cs, fire, ep, look, (fw, fh), scene.fps)
            r.bloom(b, comp.bloom_radius)
            r.composite(b, (W, H), comp, time=t, frame=frame, plate_fit=plate_fit, liquid='both')
        self.last_render_ms = (time.perf_counter() - t0) * 1000.0
        return cs
