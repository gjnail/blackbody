"""The engine: runs a scene's simulation frame by frame and renders any frame of it.

Simulation is stateful and only moves forward. Every simulated frame's scalar fields (and ember
state) go into a RAM cache, so scrubbing back and changing the look, lighting, camera or
composite re-renders from the cache without re-simulating. Anything that changes the simulation
itself invalidates the cache.
"""
from __future__ import annotations

import logging
import math
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from . import camera as cam
from .embers import Embers
from .gpu import GPU, TU, Uniforms
from .both_engine import BothEngine
from .liquid_engine import LiquidEngine
from .renderer import Renderer, SurfaceInputs
from .solver import Solver

log = logging.getLogger('blackbody.engine')


class FrameCache:
    """Per-frame simulation snapshots (scalars as float16, plus ember state) under a memory budget."""

    def __init__(self, budget_bytes=4 << 30):
        self.budget = int(budget_bytes)
        self.items = OrderedDict()
        self.bytes = 0

    def clear(self):
        self.items.clear()
        self.bytes = 0

    def put(self, frame, entry):
        size = sum(a.nbytes for a in entry.values() if isinstance(a, np.ndarray))
        if frame in self.items:
            self.bytes -= self.items.pop(frame)['_size']
        entry['_size'] = size
        self.items[frame] = entry
        self.bytes += size
        while self.bytes > self.budget and len(self.items) > 1:
            _, old = self.items.popitem(last=False)
            self.bytes -= old['_size']

    def get(self, frame):
        e = self.items.get(frame)
        if e is not None:
            self.items.move_to_end(frame)
        return e

    def frames(self):
        return sorted(self.items)

    def __contains__(self, frame):
        return frame in self.items


@dataclass
class VolumeView:
    """What the renderer needs from a volume: textures plus layout (duck-types the Solver)."""
    scal: list
    vel: list
    dims: tuple
    h: float
    origin: tuple
    aux: list | None = None     # oxygen and water vapour, when the scene carries them
    chem: list | None = None    # flame colourant
    time: float = 0.0           # simulation time of the frame (animates render-time detail, haze and grain)
    vel_k: int = 1              # cells of this volume per velocity cell (the upres factor)
    burn: object = None         # burnable floor (for scorch)
    burn_obj: object = None     # object-burn atlas (for scorch on burnable colliders)


def halton(i, b):
    f, r = 1.0, 0.0
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r


class Engine(LiquidEngine, BothEngine):
    def __init__(self, gpu: GPU | None = None, cache_bytes=4 << 30):
        self.gpu = gpu or GPU()
        self.solver = Solver(self.gpu)
        self.renderer = Renderer(self.gpu)
        self.embers = Embers(self.gpu)
        self.cache = FrameCache(cache_bytes)
        self.sim_frame = None
        self.sig = None
        self.final = False
        self.last_step_ms = 0.0
        self.last_render_ms = 0.0
        self.last_substeps = 0
        self._rscal = None
        self._raux = None
        self._rchem = None
        self._rburn = None
        self._rburn_obj = None
        self._acc2 = None
        self._zero_vel = self.gpu.texture3d((4, 4, 4), self.gpu.vel_format, 'zero-vel')
        self.gpu.upload(self._zero_vel, np.zeros((4, 4, 4, 4), np.float32 if self.gpu.vel_format == 'rgba32float' else np.float16))
        g = self.gpu
        self.k_accum = g.kernel('accum.wgsl', ['utex2d'] * 6 + ['st2d:rgba32float:w'] * 3, workgroup=(8, 8, 1))
        self.k_copy = g.kernel('copy2d.wgsl', ['utex2d'] * 3 + ['st2d:rgba16float:w'] * 3, workgroup=(8, 8, 1))
        self._acc = None
        self._acc_size = None

    # -- simulation ------------------------------------------------------------------------------

    def prepare(self, scene, final=False, soft=False):
        """Match the solver to the scene. Returns None (nothing changed), 'soft' or 'reset'.

        A new grid layout always restarts the simulation. Other simulation changes restart it too,
        unless `soft` is set: then the running simulation carries on with the new settings (live
        tweaking), and only the cache, now stale, is dropped."""
        if scene.kind == 'liquid':
            return self._prepare_liquid(scene, final, soft)
        if scene.kind == 'both':
            return self._prepare_both(scene, final, soft)
        if self.kind != 'fire':
            self.kind = 'fire'
            self.sig = None
        sig = scene.sim_signature(final)
        dims, h, origin = scene.sim_layout(final)
        meshes = [scene.mesh_path(d['mesh']) for d in scene.emitters + scene.colliders
                  if d['enabled'] and d['shape'] == 'mesh' and d['mesh']]
        self.solver.set_meshes(meshes, scene.data['domain']['mesh_resolution'])
        changed = self.solver.configure(dims, h, origin, scene.features(), scene.data['render']['upres'] if final else 1)
        self.solver.set_colliders(scene.colliders_gpu(scene.start))
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

    def reset(self):
        if self.kind == 'liquid':
            self._reset_liquid()
            self.sim_frame = None
            return
        if self.kind == 'both':
            self._reset_both()
            self.sim_frame = None
            return
        self.solver.reset()
        self.embers.reset()
        self.sim_frame = None

    def invalidate(self):
        self.sig = None

    def preroll_frames(self, scene):
        return int(round(scene.data['domain']['preroll'] * scene.fps))

    def _cam_grid(self, scene, frame):
        spec, fire = scene.camera(frame)
        w, h = scene.output_size()
        cs = cam.compute(spec, w / h, fire)
        w2g = cam.world_to_grid(fire, self.solver.origin, self.solver.h)
        return (w2g @ np.append(cs.eye, 1.0))[:3]

    def step_frame(self, scene, frame):
        """Advance the live simulation to `frame` (exactly one frame after the current one)."""
        if self.kind == 'liquid':
            return self._step_liquid(scene, frame)
        if self.kind == 'both':
            return self._step_both(scene, frame)
        t0 = time.perf_counter()
        fps = scene.fps
        fdt = scene.v('domain', 'time_scale', frame) / fps
        d = scene.data['domain']
        prm = scene.solver_params(frame)
        ep = scene.ember_params(frame)
        look = scene.look(frame)
        n = self.solver.substeps_for(fdt, cfl=d['cfl'], lo=d['substeps_min'], hi=d['substeps_max'])
        cam_g = self._cam_grid(scene, frame) if ep.enabled else (0.0, 0.0, 0.0)
        moving = scene.colliders_animated()
        with self.gpu.batch() as b:
            for i in range(n):
                # emitters and colliders move within the frame, so fast ones leave a continuous trail
                fs = frame - 1 + (i + 0.5) / n
                ems = scene.emitters_gpu(fs)
                cols = scene.colliders_gpu(fs) if moving else None
                self.solver.step(b, fdt / n, prm, ems, cols)
                if ep.enabled:
                    ember_ems = scene.emitters_gpu(fs, embers_only=True)
                    if ember_ems or self.embers.count:
                        self.embers.step(b, self.solver, ep, ember_ems, fdt / n, cam_g, look.smoke_density, look.ambient_k)
        self.solver.measure()
        self.sim_frame = frame
        self.last_substeps = n
        self.last_step_ms = (time.perf_counter() - t0) * 1000.0

    def snapshot(self):
        if self.kind == 'liquid':
            return self._snapshot_liquid()
        if self.kind == 'both':
            entry = self._snapshot_fire()
            entry.update(self._snapshot_liquid())
            return entry
        return self._snapshot_fire()

    def _snapshot_fire(self):
        s = self.solver
        entry = {'scal': s.read_scalars_fine(), 'time': s.time, 'upres': s.upres}
        if s.burn:
            entry['burn0'] = np.ascontiguousarray(s.read_burn()[:, 0:1])  # the floor: only its bottom layer
        if s.burn_obj:
            entry['burn_obj'] = s.read_burn_obj()
        if self.solver.aux:
            entry['aux'] = self.solver.read_aux()
        if self.solver.chem:
            entry['chem'] = self.solver.read_chem()
        if self.embers.count:
            entry['embers'] = np.stack([
                np.frombuffer(self.gpu.read_buffer(buf), np.float32).reshape(-1, 4) for buf in self.embers.cached_buffers()])
        return entry

    def simulate_to(self, scene, frame, progress=None, cancelled=None, cache=True):
        """Bring the live simulation to `frame`, restarting (with pre-roll) if it is ahead of it."""
        start = scene.start
        if self.sim_frame is None or frame < self.sim_frame:
            self.reset()
            self.sim_frame = start - self.preroll_frames(scene) - 1
        first = self.sim_frame
        total = max(1, frame - first)
        while self.sim_frame < frame:
            if cancelled is not None and cancelled():
                return False
            self.step_frame(scene, self.sim_frame + 1)
            if cache and self.sim_frame >= start:
                self.cache.put(self.sim_frame, self.snapshot())
            if progress is not None:
                progress((self.sim_frame - first) / total, self.sim_frame)
        return True

    # -- rendering -------------------------------------------------------------------------------

    def volume_for(self, scene, frame):
        """The volume to render for `frame`: the live simulation if it is there, else the cache."""
        if self.kind == 'liquid':
            return self._liquid_view(scene, frame)
        s = self.solver
        k = s.upres
        dims_v = s.dims_fine if k > 1 else s.dims
        if self.sim_frame == frame:
            scal = s.scal_fine[0] if k > 1 else s.scal[0]
            return VolumeView([scal], [s.vel[0]], dims_v, s.h / k, s.origin,
                              [s.aux[0]] if s.aux else None, [s.chem[0]] if s.chem else None, s.time, k,
                              s.burn[0] if s.burn else None, s.burn_obj[0] if s.burn_obj else None), True
        entry = self.cache.get(frame)
        if entry is None:
            return None, False
        k = entry.get('upres', 1)
        dims_v = tuple(d * k for d in s.dims)
        self._rscal = self._cached_tex(self._rscal, dims_v, 'cached-scal')
        self.gpu.upload(self._rscal, entry['scal'])
        burn = burn_obj = None
        if 'burn0' in entry:
            self._rburn = self._cached_tex(self._rburn, (s.dims[0], 1, s.dims[2]), 'cached-burn')
            self.gpu.upload(self._rburn, entry['burn0'])
            burn = self._rburn
        if 'burn_obj' in entry:
            bo = entry['burn_obj']
            self._rburn_obj = self._cached_tex(self._rburn_obj, (bo.shape[2], bo.shape[1], bo.shape[0]), 'cached-burn-obj')
            self.gpu.upload(self._rburn_obj, bo)
            burn_obj = self._rburn_obj
        aux = chem = None
        if 'aux' in entry:
            self._raux = self._cached_tex(self._raux, s.dims, 'cached-aux')
            self.gpu.upload(self._raux, entry['aux'])
            aux = [self._raux]
        if 'chem' in entry:
            self._rchem = self._cached_tex(self._rchem, s.dims, 'cached-chem')
            self.gpu.upload(self._rchem, entry['chem'])
            chem = [self._rchem]
        if 'embers' in entry and self.embers.count == entry['embers'].shape[1]:
            for buf, data in zip(self.embers.cached_buffers(), entry['embers']):
                self.gpu.write_buffer(buf, data)
        return VolumeView([self._rscal], [self._zero_vel], dims_v, s.h / k, s.origin, aux, chem, entry.get('time', s.time), k,
                          burn, burn_obj), False

    def _cached_tex(self, tex, dims, label):
        if tex is None or tex.size != tuple(dims):
            if tex is not None:
                tex.destroy()
            tex = self.gpu.texture3d(dims, 'rgba16float', label)
        return tex

    @staticmethod
    def footage_ambient(plate, scene, frame):
        """Average scene-linear colour of the footage, used as the ambient light on the smoke."""
        a = np.asarray(plate)[::16, ::16, :3].astype(np.float32)
        x = a / 255.0 if plate.dtype == np.uint8 else a
        kind = scene.data['composite']['plate_transform']
        if kind == 'srgb':
            lin = np.where(x <= 0.04045, x / 12.92, ((np.maximum(x, 0) + 0.055) / 1.055) ** 2.4)
        elif kind == 'rec709':
            lin = np.maximum(x, 0) ** 2.4
        elif kind == 'acescg':
            m = np.array([[1.70505, -0.62179, -0.08326], [-0.13026, 1.14080, -0.01055], [-0.02400, -0.12897, 1.15297]])
            lin = x @ m.T
        else:
            lin = x
        mean = lin.reshape(-1, 3).mean(axis=0) * (2.0 ** scene.data['composite']['plate_exposure'])
        k = scene.get(('lighting', 'ambient_intensity'), frame)
        return tuple(float(c) for c in mean * k)

    def surfaces_for(self, scene, frame, vol, comp):
        """The surfaces around the fire at `frame`: colliders in the shot (from the scene, so animated ones
        are where they are at this frame), the ground, and the burn state saved with the frame."""
        s = self.solver
        cols = s._slotted(scene.colliders_gpu(frame))
        spread = bool(scene.data['spread']['enabled'])
        return SurfaceInputs(
            colliders=cols, meshes=s.meshes, burn=vol.burn, burn_obj=vol.burn_obj,
            slots=s.burn_slots if vol.burn_obj is not None else None, grid=s.dims, cell=s.h,
            ground=bool(scene.data['domain']['ground']), lit=comp.surface_light > 0.0, scorch=spread and comp.scorch > 0.0)

    def _ensure_acc2(self, w, h):
        if self._acc2 is not None and self._acc2[0].size[:2] == (w, h):
            return
        for t in self._acc2 or []:
            t.destroy()
        if getattr(self, '_acc2_spare', None) is not None:
            self._acc2_spare.destroy()
        self._acc2 = [self.gpu.texture2d(w, h, 'rgba32float', f'acc-surf{i}') for i in range(6)]
        self._acc2_spare = self.gpu.texture2d(w, h, 'rgba16float', 'acc-surf-spare')

    def _ensure_acc(self, w, h):
        if self._acc_size == (w, h):
            return
        if self._acc:
            for t in self._acc:
                t.destroy()
        self._acc = [self.gpu.texture2d(w, h, 'rgba32float', f'acc{i}') for i in range(6)]
        self._acc_size = (w, h)

    def render(self, scene, frame, out_size, mode='composite', final=False, samples=1, motion_blur=False,
               fire_scale=1.0, plate=None, plate_fit=(1.0, 1.0), seed=None):
        """Render `frame` into the renderer's buffers. The frame must be live or cached."""
        if self.kind == 'liquid':
            return self._render_liquid(scene, frame, out_size, mode, final, samples, motion_blur, fire_scale, plate,
                                       plate_fit, seed)
        if self.kind == 'both':
            return self._render_both(scene, frame, out_size, mode, final, samples, motion_blur, fire_scale, plate,
                                     plate_fit, seed)
        t0 = time.perf_counter()
        vol, live = self.volume_for(scene, frame)
        if vol is None:
            raise RuntimeError(f'frame {frame} is neither simulated nor cached')
        W, H = out_size
        fw, fh = max(8, int(round(W * fire_scale))), max(8, int(round(H * fire_scale)))
        spec, fire = scene.camera(frame)
        cs = cam.compute(spec, W / H, fire)
        look = scene.look(frame, final)
        look.vapour = bool(vol.aux) and self.solver.features.get('vapour', False)
        look.colourant = bool(vol.chem)
        if plate is not None and scene.data['lighting'].get('ambient_from_footage', True):
            look.ambient = self.footage_ambient(plate, scene, frame)
        comp = scene.comp(frame, mode)
        ep = scene.ember_params(frame)
        t = vol.time  # the frame's own time, so a cached frame renders exactly as it did live
        shutter = 0.0
        if motion_blur and live:
            shutter = scene.data['render']['shutter_angle'] / 360.0 / scene.fps * scene.v('domain', 'time_scale', frame)
        r = self.renderer
        if plate is not None:
            r.set_plate(plate)
        elif mode == 'composite':
            r.set_plate(None)
        base_seed = frame * 64 if seed is None else seed
        samples = max(1, int(samples))
        surfaces = self.surfaces_for(scene, frame, vol, comp)
        ground = scene.data['domain']['ground']
        with self.gpu.batch() as b:
            r.light(b, vol, look, fire, t)
            if samples == 1:
                r.march(b, vol, cs, fire, look, (fw, fh), seed=base_seed, shutter=shutter, ground=ground, time=t,
                        surfaces=surfaces)
            else:
                self._ensure_acc(fw, fh)
                self._ensure_acc2(fw, fh)
                r._ensure_fire(fw, fh)
                set0, set1 = self._acc[0:3], self._acc[3:6]
                sur0, sur1 = self._acc2[0:3], self._acc2[3:6]
                srcs2 = [r.surf, r.mask, r.mask]
                for i in range(samples):
                    jit = (halton(i + 1, 2) - 0.5, halton(i + 1, 3) - 0.5)
                    r.march(b, vol, cs, fire, look, (fw, fh), jitter=jit, seed=base_seed + i, shutter=shutter,
                            ground=ground, time=t, surfaces=surfaces)
                    src_in, dst = (set0, set1) if i % 2 == 0 else (set1, set0)
                    b.run(self.k_accum, [r.beauty, r.emit, r.aux, *src_in, *dst], Uniforms().v4(fw, fh, 0, i), (fw, fh, 1))
                    s_in, s_dst = (sur0, sur1) if i % 2 == 0 else (sur1, sur0)
                    b.run(self.k_accum, [*srcs2, *s_in, *s_dst], Uniforms().v4(fw, fh, 0, i), (fw, fh, 1))
                last, other = (set1, set0) if (samples - 1) % 2 == 0 else (set0, set1)
                b.run(self.k_accum, [r.beauty, r.emit, r.aux, *last, *other], Uniforms().v4(fw, fh, 1, samples), (fw, fh, 1))
                b.run(self.k_copy, [*other, r.beauty, r.emit, r.aux], Uniforms().v4(fw, fh), (fw, fh, 1))
                slast, sother = (sur1, sur0) if (samples - 1) % 2 == 0 else (sur0, sur1)
                b.run(self.k_accum, [*srcs2, *slast, *sother], Uniforms().v4(fw, fh, 1, samples), (fw, fh, 1))
                b.run(self.k_copy, [*sother, r.surf, r.mask, self._acc2_spare], Uniforms().v4(fw, fh), (fw, fh, 1))
            if ep.enabled and self.embers.count:
                self.embers.draw(b, r, cs, fire, ep, look, (fw, fh), scene.fps)
            r.bloom(b, comp.bloom_radius)
            r.composite(b, (W, H), comp, time=t, frame=frame, plate_fit=plate_fit)
        self.last_render_ms = (time.perf_counter() - t0) * 1000.0
        return cs

    def display_image(self):
        return self.renderer.read_display()

    def aovs(self):
        """Fire element passes at render resolution, scene-linear float16:
        beauty (RGBA premultiplied), emission (RGB + flame coverage), aux (heat, depth, kelvin/1000, alpha)."""
        r = self.renderer
        out = {'beauty': self.gpu.read(r.beauty), 'emission': self.gpu.read(r.emit), 'aux': self.gpu.read(r.aux)}
        if self.kind != 'liquid' and getattr(r, 'surf', None) is not None:
            # fire light on surfaces (rgb) + holdout coverage (a); scorch, holdout depth * coverage, coverage
            out['surface'] = self.gpu.read(r.surf)
            out['mask'] = self.gpu.read(r.mask)
        return out

    def linear_comp(self):
        return self.renderer.read_linear()

    # -- info ------------------------------------------------------------------------------------

    def stats(self):
        if self.kind == 'liquid' and self.liquid is not None:
            return self._stats_liquid()
        s = self.solver
        out = {
            'gpu': self.gpu.name, 'backend': self.gpu.backend, 'dims': s.dims, 'voxels': int(np.prod(s.dims)) if s.dims else 0,
            'cell_mm': s.h * 1000 if s.h else 0, 'max_speed': s.max_speed, 'substeps': self.last_substeps,
            'sim_ms': self.last_step_ms, 'render_ms': self.last_render_ms, 'cached': len(self.cache.items),
            'cache_mb': self.cache.bytes / 1e6, 'embers': self.embers.count, 'burning': s.burning,
            'memory_mb': s.memory_bytes() / 1e6 if s.dims else 0,
            'mesh_errors': dict(s.meshes.errors),
        }
        if self.kind == 'both' and self.liquid is not None:
            return self._stats_both(out)
        return out
