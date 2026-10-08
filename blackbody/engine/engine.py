"""The engine: runs a scene's simulation frame by frame and renders any frame of it.

Simulation is stateful and only moves forward. Every simulated frame's scalar fields (and ember
state) go into a RAM cache, so scrubbing back and changing the look, lighting, camera or
composite re-renders from the cache without re-simulating. Anything that changes the simulation
itself invalidates the cache. With Domain › Disk cache the frames also go to disk (io/simcache.py),
with a checkpoint of the whole simulation every few frames, so a simulation survives closing the
app, resumes from its last checkpoint, and can be rendered on other machines.
"""
from __future__ import annotations

import logging
import math
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from . import camera as cam
from .cloth import STEPS_PER_SECOND, Cloth
from .embers import Embers
from .gpu import GPU, TU, Uniforms, groups_1d
from .ballistics_engine import BallisticsEngine
from .both_engine import BothEngine
from .cloud_engine import CloudEngine
from .liquid_engine import LiquidEngine
from .matter_engine import MatterEngine
from .objheat_engine import HeatEngine
from .strands_engine import StrandsEngine
from .renderer import Renderer, SurfaceInputs
from . import stage as stage_mod
from .bodyfield import BodyField
from .solids import Solids, attached
from .solver import Solver

log = logging.getLogger('blackbody.engine')


class FrameCache:
    """Per-frame simulation snapshots (scalars as float16, plus ember state) under a memory budget,
    optionally backed by a disk cache (SimCache) that also holds checkpoints."""

    def __init__(self, budget_bytes=4 << 30):
        self.budget = int(budget_bytes)
        self.items = OrderedDict()
        self.bytes = 0
        self.disk = None

    def attach(self, disk):
        """Back the cache with a SimCache (or None)."""
        if self.disk is not None and self.disk is not disk:
            self.disk.flush()
        self.disk = disk

    def clear(self):
        self.items.clear()
        self.bytes = 0

    def put(self, frame, entry):
        if self.disk is not None:
            self.disk.put(frame, {k: v for k, v in entry.items() if not k.startswith('_')})
            if 'state' in entry:
                self.disk.mark_checkpoint(frame)
        # checkpoints (the whole solver state) stay on disk only
        entry = {k: v for k, v in entry.items() if k not in ('state', 'ember_state')}
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
        if self.disk is not None and frame in self.disk:
            e = self.disk.get(frame)
            if e is not None:
                disk, self.disk = self.disk, None   # already on disk: only keep it in RAM
                self.put(frame, e)
                self.disk = disk
                return self.items.get(frame)
        return None

    def frames(self):
        if self.disk is not None:
            return sorted(set(self.items) | set(self.disk.frames()))
        return sorted(self.items)

    def __contains__(self, frame):
        return frame in self.items or (self.disk is not None and frame in self.disk)


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
    stain: object = None        # soot left on surfaces (simulation grid)
    grid: tuple = None          # the simulation grid's dims (the volume may be finer, with upres)
    cell: float = 0.0           # the simulation grid's cell size (m)
    stain_obj: object = None    # soot on colliders, each in its own frame
    cloth: object = None        # fabric: 'live', or a cached frame's arrays (Cloth.snapshot)


def halton(i, b):
    f, r = 1.0, 0.0
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r


class Engine(LiquidEngine, BothEngine, CloudEngine, MatterEngine, StrandsEngine, BallisticsEngine, HeatEngine):
    def __init__(self, gpu: GPU | None = None, cache_bytes=4 << 30):
        self.gpu = gpu or GPU()
        self.solver = Solver(self.gpu)
        self.renderer = Renderer(self.gpu)
        self.embers = Embers(self.gpu)
        self.cloth = Cloth(self.gpu)
        self.solids = Solids()     # rigid bodies: objects that fall, tumble and float (MuJoCo)
        self._air_bufs = None
        self._stage = None         # the set drawn in CG (stage.py), made when first needed
        self._body_fields = {}     # broken pieces in the simulation grids (bodyfield.py), one per grid, made when needed
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
        self._rvel = None
        self._rstain = None
        self._acc2 = None
        self._disk_key = None
        self.base_dims = None
        self._ocio_key = None
        self._ocio_error = None
        self.cache_readonly = False   # render farm machines read a shared disk cache without writing to it
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
        if scene.kind != 'fire':
            self.solver.cloth_hook = None   # fabric is simulated in fire scenes only
        if scene.kind == 'liquid':
            out = self._prepare_liquid(scene, final, soft)
        elif scene.kind == 'both':
            out = self._prepare_both(scene, final, soft)
        elif scene.kind == 'cloud':
            self.solids.clear()   # a sky kilometres across: no falling objects in it
            out = self._prepare_cloud(scene, final, soft)
        else:
            out = self._prepare_fire(scene, final, soft)
        if getattr(self, '_lvol', None) is not None:
            self._lvol.reset()   # (Lume's light in the smoke: none carried over from another scene or run)
        self._attach_disk(scene, final)
        self._scene_notices(scene)
        return out

    def _scene_notices(self, scene):
        """What the caps leave out of this scene (scene/caps.py) and whether its disk cache opened, for notices(); and its
        objects' names, for what is said about them as it runs."""
        from ..scene import caps
        try:
            self._cap_notes = caps.notices(scene)
        except Exception as ex:   # (a check that fails must not stop the simulation)
            log.warning('Could not check the scene against the caps: %s', ex)
            self._cap_notes = []
        if getattr(self, '_stage', None) is not None:
            self._stage.clear_notes()     # (what the last scene's glowing matter left out of its lights)
        if scene.data['domain'].get('disk_cache') and self.cache.disk is None:
            self._cap_notes.append('The disk cache could not be opened (Domain › Disk cache): the frames are kept in memory '
                                   'only.')
        self._col_names = [c.get('name', '') for c in scene.colliders]

    def _attach_disk(self, scene, final):
        """Back the frame cache with the scene's disk cache (Domain › Disk cache), or detach it."""
        d = scene.data['domain']
        if not d.get('disk_cache'):
            self.cache.attach(None)
            self._disk_key = None
            return
        from pathlib import Path
        from ..io.simcache import SimCache, default_root
        folder = (Path(d['cache_dir']) if d.get('cache_dir') else default_root(scene.path)) / ('final' if final else 'preview')
        sig = scene.sim_signature(final)
        if self.solids.active:
            # (and the bodies as built: cut into other pieces since, or round another hit, the frames are another
            # simulation's, and drawn with these pieces they would be wrong)
            sig = f'{sig}-{self.solids.fingerprint()}'
        key = (str(folder), sig, self.cache_readonly)
        if key != self._disk_key or self.cache.disk is None:
            try:
                self.cache.attach(SimCache(folder, sig, readonly=self.cache_readonly))
                self._disk_key = key
            except OSError as ex:
                log.warning('Disk cache unavailable (%s): %s', folder, ex)
                self.cache.attach(None)
                self._disk_key = None

    def _prepare_fire(self, scene, final=False, soft=False):
        if self.kind != 'fire':
            self.kind = 'fire'
            self.sig = None
        sig = scene.sim_signature(final)
        dims, h, origin = scene.sim_layout(final)
        self.solver.set_meshes(scene.mesh_items(scene.start), scene.data['domain']['mesh_resolution'])
        changed = self.solver.configure(dims, h, origin, scene.features(), scene.upres_for(final))
        self.base_dims = dims
        self.solver.set_colliders(scene.colliders_gpu(scene.start))
        if self.cloth.configure(scene.fabric_specs()):
            changed = True
        if self.solids.configure(scene, (dims, h, origin)):
            changed = True
        if self._prepare_matter(scene, (dims, h, origin)):
            changed = True
        if self._prepare_strands(scene):
            changed = True
        if self._prepare_heat(scene):
            changed = True
        self.solver.cloth_hook = self._solver_hook()
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
        self.solids.reset()
        self._reset_heat()
        if self._matter is not None:
            self._matter.reset()
        if self._strands is not None:
            self._strands.reset()
        if self.kind == 'liquid':
            self._reset_liquid()
            self.sim_frame = None
            return
        if self.kind == 'both':
            self._reset_both()
            self.sim_frame = None
            return
        if self.kind == 'cloud':
            self._reset_cloud()
            self.sim_frame = None
            return
        self.solver.restore_base()
        self.solver.reset()
        self.embers.reset()
        self.cloth.reset()
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
        if self.kind == 'cloud':
            return self._step_cloud(scene, frame)
        t0 = time.perf_counter()
        fps = scene.fps
        fdt = scene.v('domain', 'time_scale', frame) / fps
        d = scene.data['domain']
        prm = scene.solver_params(frame)
        ep = scene.ember_params(frame)
        look = scene.look(frame)
        n = self.solver.substeps_for(fdt, cfl=d['cfl'], lo=d['substeps_min'], hi=scene.substep_cap(frame, self.solver.h))
        cam_g = self._cam_grid(scene, frame) if ep.enabled else (0.0, 0.0, 0.0)
        # rigid bodies move first through the frame (with the air's drag from the frame before); the gas then
        # sees them where they are at each substep
        poses = None
        self._cloth_meets_matter(fdt)   # (fabric and sand, snow, mud: each the other's surface this frame)
        if self.solids.active:
            self._air_for_solids()
            self._objects_meet_cloth(fdt)
            self._shots_ahead(scene, fdt)         # (bullets: where their ways cross the sand, snow, mud and jelly)
            poses = self.solids.advance(scene, frame, fdt, n, couple=self._matter_couple(scene, frame, fdt, self.solver.meshes))
            self._shots_kick(fdt)                    # (and what they did to it)
            self._cloth_takes_objects(fdt)
        self._step_heat(scene, frame, fdt, gas=self.solver)   # (the objects warm and cool: objheat_engine.py)
        moving = scene.colliders_animated() or poses is not None
        # deforming meshes: the frames either side of this step in the atlas
        self.solver.set_meshes(scene.mesh_items(frame - 1), d['mesh_resolution'])
        cloth = self.cloth.active
        if cloth and not self.cloth.placed:
            self.cloth.place(scene.fabrics_at(frame - 1), look.ambient_k)
        self.cloth.prepare_frame(self.solver)   # (also clears the solver's steam-off-cloth flag without cloth)
        cloth_steps = max(1, int(math.ceil(fdt * STEPS_PER_SECOND / n)))
        grass = self.strands_on
        if grass:
            self._strands.prepare_frame(self.solver)
            wind = scene.wind(frame, scene.v('camera', 'fire_yaw', frame))
            gust = scene.v('motion', 'gust', frame)
            ground_on = bool(d['ground'])
        burning = poses is not None and self._burn_pieces(scene, fdt, frame)   # (things that break and burn)
        pieces = poses is not None and self._pieces_for(scene, self.solver)
        dust = self.solids.dust(scene, fdt, n) if (poses is not None and self.solids.sets) else None
        shot = self._shot_puffs(scene, fdt, n) if self.shooting else None   # (bullets' dust and gun smoke)
        if shot:
            dust = [a + b for a, b in zip(dust or [[]] * n, shot)]
        with self.gpu.batch() as b:
            self._build_radiant(b, scene, frame, gas=self.solver)   # (what radiates heat this frame: radiant.py)
            if burning:
                self.piece_fire.splat(b, self.solver)
            self._splat_matter_fire(b, scene, self.solver)   # (dry leaves, sawdust, coal: their flames)
            msolid = self._matter_solid(b, self.solver)   # (sand, snow and mud: solid to the gas)
            for i in range(n):
                # emitters and colliders move within the frame, so fast ones leave a continuous trail
                fs = frame - 1 + (i + 0.5) / n
                self.solver.pieces_step = self._solids_step(self.body_field if pieces else None, msolid, i)
                carried = poses[i] if poses else None   # (things attached to falling objects go with them)
                ems = scene.emitters_gpu(fs, substeps=n, moved=attached(scene, 'emitter', carried))
                if dust:
                    ems = ems + dust[i]    # (dust where things broke; the scene's own sources come first)
                cols = scene.colliders_gpu(fs, carried) if moving else None
                self.solver.step(b, fdt / n, prm, ems, cols)
                if cloth:
                    # fabric moves in the air just stepped, then spreads onto the gas for the next substep
                    self.cloth.step(b, self.solver, fdt / n, scene.fabrics_at(fs + 0.5 / n, moved=attached(scene, 'fabric', carried)),
                                    prm, look, self.solver.colliders, self.solver.meshes, steps=cloth_steps)
                    self.cloth.splat(b, self.solver, look)
                if grass:
                    # the grass in the air just stepped, then its fire onto the gas for the next substep
                    self._step_strands(b, scene, fs, fdt / n, self.solver, self.solver.colliders, self.solver.meshes, look,
                                       ground_on, self.solver.origin[1], wind, gust)
                if ep.enabled:
                    ember_ems = scene.emitters_gpu(fs, embers_only=True, moved=attached(scene, 'emitter', carried))
                    if ember_ems or self.embers.count:
                        self.embers.step(b, self.solver, ep, ember_ems, fdt / n, cam_g, look.smoke_density, look.ambient_k)
        self.solver.pieces_step = None
        self.solver.measure()
        self._step_matter(scene, frame, fdt, poses, n, self.solver.meshes)
        self._melt_matter(scene, frame, fdt, self.solver)
        self._heat_matter(scene, frame, fdt, self.solver)
        self._heat_back()
        if d.get('grow'):
            self._grow(scene, prm)
        self.sim_frame = frame
        self.last_substeps = n
        self.last_step_ms = (time.perf_counter() - t0) * 1000.0

    def _grow(self, scene, prm):
        """Domain › Grow to fit: enlarge the box on each open side the smoke comes near."""
        s = self.solver
        base = np.asarray(s.base[0] if s.base else s.dims)
        limit = np.maximum(base, np.floor(base * max(1.0, scene.data['domain']['grow_limit']) / 8) * 8).astype(int)
        margin = int(prm.sponge) + 4
        step = max(8, int(round(max(s.dims) * 0.125 / 8)) * 8)
        lo, hi = s.growth_needed(margin, step, limit, prm)
        if lo.any() or hi.any():
            old = s.dims
            s.grow(lo, hi)
            log.info('Domain grew from %s to %s cells', old, s.dims)

    def _air_for_solids(self):
        """The gas velocity at every rigid body, for the air's drag on it (a blast blows light things away)."""
        pts = self.solids.sample_points()
        n = len(pts)
        if n == 0 or self.kind not in ('fire', 'both') or not self.solver.dims:
            return
        s, g = self.solver, self.gpu
        if self._air_bufs is None or self._air_bufs[0] < n:
            cap = max(16, n)
            self._air_bufs = (cap, g.buffer(cap * 16, 'solid-points'), g.buffer(cap * 16, 'solid-air'))
        cap, pb, vb = self._air_bufs
        p4 = np.zeros((cap, 4), np.float32)
        p4[:n, :3] = pts
        g.write_buffer(pb, p4)
        k = g.kernel('solids_air.wgsl', ['tex3d', 'tex3d', 'smp', 'rbuf', 'buf'], workgroup=(64, 1, 1))
        with g.batch() as b:
            b.run(k, [s.vel[0], s.scal[0], g.linear, pb, vb], Uniforms().v4(*s.origin, s.h).v4(*s.dims, n), groups=groups_1d(n))
        out = np.frombuffer(g.read_buffer(vb, n * 16), np.float32).reshape(n, 4)
        self.solids.set_air(np.nan_to_num(out[:, :3]))

    _pfire = None

    @property
    def piece_fire(self):
        if self._pfire is None:
            from .piece_fire import PieceFire
            self._pfire = PieceFire(self.gpu)
        return self._pfire

    def _burn_pieces(self, scene, fdt, frame=None):
        """Things that break and burn (solids.py): the gas's heat round each piece, their fire through the frame, and the
        fuel the burning ones give the gas this frame. True when some burn. (Before the frame's batch: it reads back.)"""
        if not (self.solids.active and self.solids.burning and self.solver.dims):
            if self._pfire is not None:
                self._pfire.n = 0
            return False
        pf = self.piece_fire
        pts, own = self.solids.fire_points()
        T = pf.temperatures(self.solver, pts)
        sp = dict(scene.data['spread'])
        sp['burn_speed'] = float(scene.get(('spread', 'burn_speed'), frame))   # (it can be keyed)
        self.solids.burn(fdt, T, own, sp)
        fp, fv = self.solids.fuel_points(sp)
        pf.prepare_frame(self.solver, fp, fv)
        return pf.n > 0

    def snapshot(self, scene=None):
        if self.kind == 'liquid':
            entry = self._snapshot_liquid()
        elif self.kind == 'both':
            entry = self._snapshot_both(scene)
        elif self.kind == 'cloud':
            return self._snapshot_cloud()
        else:
            entry = self._snapshot_fire(scene)
        self._snapshot_matter(entry)
        self._snapshot_strands(entry)
        self._snapshot_heat(entry)
        return entry

    def _snapshot_fire(self, scene=None):
        s = self.solver
        entry = {'scal': s.read_scalars_fine(), 'time': s.time, 'upres': s.upres,
                 'dims': list(s.dims), 'origin': list(s.origin), 'h': s.h}
        if scene is None or scene.data['render'].get('motion_blur', True):
            # face velocities at half precision, so frames shown from the cache get motion blur too
            entry['vel'] = s.read_velocity()
        if s.stain:
            entry['stain'] = s.read_stain().astype(np.float16)
        if s.stain_obj is not None:
            entry['stain_obj'] = s.read_stain_obj().astype(np.float16)
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
        if self.cloth.active:
            c = self.cloth.snapshot()
            if c is not None:
                for k, v in c.items():
                    entry['cloth_' + k] = v
        if self.solids.active:
            entry['solids'] = self.solids.state()
        return entry

    def simulate_to(self, scene, frame, progress=None, cancelled=None, cache=True):
        """Bring the live simulation to `frame`, restarting (with pre-roll) if it is ahead of it."""
        start = scene.start
        if self.sim_frame is None or frame < self.sim_frame:
            self.reset()
            self.sim_frame = start - self.preroll_frames(scene) - 1
            if cache and self.kind == 'fire':
                self._resume(scene, frame)
        first = self.sim_frame
        total = max(1, frame - first)
        while self.sim_frame < frame:
            if cancelled is not None and cancelled():
                return False
            self.step_frame(scene, self.sim_frame + 1)
            if cache and self.sim_frame >= start:
                entry = self.snapshot(scene)
                if self._checkpoint_due(scene, self.sim_frame):
                    entry['state'] = self.solver.save_state()
                    entry['ember_state'] = self.embers.save_state()
                    cst = self.cloth.save_state()
                    if cst is not None:
                        entry['cloth_state'] = cst
                    self._checkpoint_strands(entry)
                self.cache.put(self.sim_frame, entry)
            if progress is not None:
                progress((self.sim_frame - first) / total, self.sim_frame)
        return True

    def _checkpoint_due(self, scene, frame):
        disk = self.cache.disk
        if disk is None or self.kind != 'fire':
            return False
        every = max(1, int(scene.data['domain'].get('checkpoint_every', 10)))
        return (frame - scene.start) % every == 0 or frame == scene.end

    def _resume(self, scene, frame):
        """Carry on from the latest checkpoint in the disk cache at or before `frame`, if there is one."""
        disk = self.cache.disk
        if disk is None:
            return False
        touched = False
        for c in reversed(disk.checkpoints()):
            if c > frame or c < scene.start:
                continue
            entry = disk.get(c)
            if not entry or 'state' not in entry:
                continue
            # (the objects changed since: this checkpoint cannot carry on. Asked before the solver is set up for it)
            if self.solids.active and not self.solids.fits(entry.get('solids')):
                continue
            touched = True
            st = entry['state']
            s = self.solver
            s._prm = scene.solver_params(c)
            base = s.base
            if tuple(st['dims']) != tuple(s.dims):
                # the domain had grown by then: take its layout, then the state
                s.base = None
                s.configure(tuple(st['dims']), st['h'], tuple(st['origin']), dict(s.features), s.upres)
                s.base = base
            s.set_meshes(scene.mesh_items(c), scene.data['domain']['mesh_resolution'])
            s.colliders = []
            if self.solids.active and not self.solids.load_state(entry.get('solids')):
                continue
            s.set_colliders(scene.colliders_gpu(c, self.solids.overrides() if self.solids.active else None))
            s.load_state(st)
            if 'ember_state' in entry:
                self.embers.load_state(entry['ember_state'])
            if self.cloth.active and not self.cloth.load_state(entry.get('cloth_state')):
                continue   # the fabric changed since: this checkpoint cannot carry on
            if not self._resume_strands(entry):
                continue   # (and the grass)
            self.sim_frame = c
            log.info('Resumed the simulation from the checkpoint at frame %d', c)
            return True
        if touched:
            # (none carried on, but one was partly loaded, and the solver set up for it: back to the start, as if none
            # had been tried)
            first = self.sim_frame
            self.reset()
            self.sim_frame = first
        return False

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
                              s.burn[0] if s.burn else None, s.burn_obj[0] if s.burn_obj else None,
                              s.stain, tuple(s.dims), s.h, s.stain_obj, 'live' if self.cloth.active else None), True
        entry = self.cache.get(frame)
        if entry is None:
            return None, False
        k = entry.get('upres', 1)
        # the frame's own layout: a growing domain may have been smaller then
        gd = tuple(int(x) for x in entry.get('dims', s.dims))
        h = float(entry.get('h', s.h))
        origin = tuple(float(x) for x in entry.get('origin', s.origin))
        dims_v = tuple(d * k for d in gd)
        self._rscal = self._cached_tex(self._rscal, dims_v, 'cached-scal')
        self.gpu.upload(self._rscal, entry['scal'])
        vel = self._zero_vel
        if 'vel' in entry:
            vd = tuple(d + 1 for d in gd)
            if self._rvel is None or self._rvel.size != vd:
                if self._rvel is not None:
                    self._rvel.destroy()
                self._rvel = self.gpu.texture3d(vd, self.gpu.vel_format, 'cached-vel')
            self.gpu.upload(self._rvel, entry['vel'])
            vel = self._rvel
        stain = stain_obj = None
        if 'stain' in entry:
            if self._rstain is None or self._rstain.size != gd:
                if self._rstain is not None:
                    self._rstain.destroy()
                self._rstain = self.gpu.texture3d(gd, 'r32float', 'cached-stain')
            self.gpu.upload(self._rstain, entry['stain'].astype(np.float32))
            stain = self._rstain
        if 'stain_obj' in entry:
            so = entry['stain_obj']
            size = (so.shape[2], so.shape[1], so.shape[0])
            if getattr(self, '_rstain_obj', None) is None or self._rstain_obj.size != size:
                if getattr(self, '_rstain_obj', None) is not None:
                    self._rstain_obj.destroy()
                self._rstain_obj = self.gpu.texture3d(size, 'r32float', 'cached-stain-obj')
            self.gpu.upload(self._rstain_obj, so.astype(np.float32))
            stain_obj = self._rstain_obj
        burn = burn_obj = None
        if 'burn0' in entry:
            self._rburn = self._cached_tex(self._rburn, (gd[0], 1, gd[2]), 'cached-burn')
            self.gpu.upload(self._rburn, entry['burn0'])
            burn = self._rburn
        if 'burn_obj' in entry:
            bo = entry['burn_obj']
            self._rburn_obj = self._cached_tex(self._rburn_obj, (bo.shape[2], bo.shape[1], bo.shape[0]), 'cached-burn-obj')
            self.gpu.upload(self._rburn_obj, bo)
            burn_obj = self._rburn_obj
        aux = chem = None
        if 'aux' in entry:
            self._raux = self._cached_tex(self._raux, gd, 'cached-aux')
            self.gpu.upload(self._raux, entry['aux'])
            aux = [self._raux]
        if 'chem' in entry:
            self._rchem = self._cached_tex(self._rchem, gd, 'cached-chem')
            self.gpu.upload(self._rchem, entry['chem'])
            chem = [self._rchem]
        if 'embers' in entry:
            self.embers.ensure(entry['embers'].shape[1])
            if self.embers.count == entry['embers'].shape[1]:
                for buf, data in zip(self.embers.cached_buffers(), entry['embers']):
                    self.gpu.write_buffer(buf, data)
        cl = {k[6:]: v for k, v in entry.items() if k.startswith('cloth_') and k != 'cloth_state'} or None
        return VolumeView([self._rscal], [vel], dims_v, h / k, origin, aux, chem, entry.get('time', s.time), k,
                          burn, burn_obj, stain, gd, h, stain_obj, cl), False

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
        cols = s._slotted(scene.colliders_gpu(frame, self.floating_overrides(frame)))
        clear = stage_mod.see_through(scene)
        if clear:
            import dataclasses
            cols = [dataclasses.replace(c, holdout=False) if i in clear else c for i, c in enumerate(cols)]
        spread = bool(scene.data['spread']['enabled'])
        return SurfaceInputs(
            colliders=cols, meshes=s.meshes, burn=vol.burn, burn_obj=vol.burn_obj,
            slots=s.burn_slots if vol.burn_obj is not None else None, grid=vol.grid or s.dims, cell=vol.cell or s.h,
            ground=bool(scene.data['domain']['ground']), lit=comp.surface_light > 0.0, scorch=spread and comp.scorch > 0.0,
            stain=vol.stain if comp.soot > 0.0 else None, shadows=comp.surface_shadows,
            stain_obj=vol.stain_obj if (comp.soot > 0.0 and s.stain_slots is not None) else None,
            stain_slots=s.stain_slots, stain_regions=s.stain_regions,
            wet=spread and comp.wet > 0.0 and (vol.burn is not None or vol.burn_obj is not None))

    def _set_ocio(self, scene, comp):
        """Bake (once per change) the OCIO LUTs the composite needs; on a bad config, fall back to the
        Standard view and sRGB footage, with a warning."""
        c = scene.data['composite']
        want_view = comp.view == 'ocio'
        want_plate = comp.plate_transform == 'ocio' and bool(c.get('ocio_plate'))
        key = (want_view, want_plate, c.get('ocio_config', ''), c.get('ocio_working', ''), c.get('ocio_display', ''),
               c.get('ocio_view', ''), c.get('ocio_look', ''), c.get('ocio_plate', '') if want_plate else '')
        if key == getattr(self, '_ocio_key', None):
            if self._ocio_error:
                comp.view = 'standard' if want_view else comp.view
                comp.plate_transform = 'srgb' if comp.plate_transform == 'ocio' else comp.plate_transform
            return
        self._ocio_key = key
        self._ocio_error = None
        if not (want_view or want_plate):
            self.renderer.set_ocio(None, None)
            if comp.plate_transform == 'ocio':
                comp.plate_transform = 'srgb'
            return
        from ..io import ocio
        try:
            pipe = ocio.pipeline(c)
            view_lut = pipe.view_lut() if want_view else None
            plate_lut, plate_log = pipe.plate_lut(c['ocio_plate']) if want_plate else (None, False)
            self.renderer.set_ocio(view_lut, plate_lut, plate_log)
        except Exception as ex:
            self._ocio_error = str(ex)
            log.warning('OCIO: %s', ex)
            self.renderer.set_ocio(None, None)
            comp.view = 'standard' if want_view else comp.view
            comp.plate_transform = 'srgb' if comp.plate_transform == 'ocio' else comp.plate_transform

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
               fire_scale=1.0, plate=None, plate_fit=(1.0, 1.0), seed=None, holdout=None, deep=0):
        """holdout: (matte, depth) arrays from the footage (either may be None; see Renderer.set_holdout).
        deep: gather up to this many deep samples per pixel (Renderer.read_deep)."""
        """Render `frame` into the renderer's buffers. The frame must be live or cached."""
        if self.kind == 'liquid':
            return self._render_liquid(scene, frame, out_size, mode, final, samples, motion_blur, fire_scale, plate,
                                       plate_fit, seed)
        if self.kind == 'both':
            return self._render_both(scene, frame, out_size, mode, final, samples, motion_blur, fire_scale, plate,
                                     plate_fit, seed)
        if self.kind == 'cloud':
            return self._render_cloud(scene, frame, out_size, mode, final, samples, motion_blur, fire_scale, plate,
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
        self._carried_lamps(scene, frame, look)
        if plate is not None and scene.data['lighting'].get('ambient_from_footage', True):
            look.ambient = self.footage_ambient(plate, scene, frame)
        self._env_look(scene, look, frame, plate=plate, cs=cs, plate_fit=plate_fit)
        comp = scene.comp(frame, mode)
        self._set_ocio(scene, comp)
        ep = scene.ember_params(frame)
        t = vol.time  # the frame's own time, so a cached frame renders exactly as it did live
        shutter = 0.0
        if motion_blur and (live or vol.vel[0] is not self._zero_vel):
            shutter = scene.data['render']['shutter_angle'] / 360.0 / scene.fps * scene.v('domain', 'time_scale', frame)
        if not live:
            # deforming meshes as they were at this frame (and the simulation's frames stay in the atlas)
            items = scene.mesh_items(frame)
            if self.sim_frame is not None:
                items += scene.mesh_items(self.sim_frame - 1)
            self.solver.set_meshes(items, scene.data['domain']['mesh_resolution'])
        r = self.renderer
        if plate is not None:
            r.set_plate(plate)
        elif mode == 'composite':
            r.set_plate(None)
        base_seed = frame * 64 if seed is None else seed
        samples = max(1, int(samples))
        surfaces = self.surfaces_for(scene, frame, vol, comp)
        r.set_holdout(*(holdout or (None, None)))
        r.set_deep(deep, samples)
        ground = scene.data['domain']['ground']
        cloth = self.cloth.active and vol.cloth is not None
        grass = self.strands_for(frame)
        self._cloth_drawn = cloth or grass
        if cloth:
            self.cloth.use_view(None if isinstance(vol.cloth, str) else vol.cloth)
        look.time = t

        def cloth_pass(b, i, jit):
            # the cloth and the grass, drawn before the march (it stops at them)
            off = shutter * ((i + 0.5) / samples - 0.5) if samples > 1 else 0.0
            return self._draw_raster(b, r, cs, fire, look, (fw, fh), jit, off, vol, cloth, grass)

        if cloth:
            self.cloth.prepare_light(r.light_dims_for(vol.dims))

        footage = plate is not None
        pieces = self.piece_poses(frame)   # (broken things: drawn, and holding out what is behind them, in every view)
        ropes = self.rope_poses(frame)
        matter = self.matter_for(frame) if mode == 'composite' else None
        bolts = scene.bolts(frame) if mode == 'composite' and scene.lights else None   # (lightning)
        shots = self.shot_view(frame) if mode == 'composite' else None    # (bullets: debris, sparks, holes)
        stage_on = (stage_mod.wanted(scene, footage, mode) or bool(pieces) or bool(ropes) or matter is not None or bool(bolts)
                    or bool(shots))
        r.hold_stage = None

        with self.gpu.batch() as b:
            r.light(b, vol, look, fire, t, occluder=self.cloth.occlusion if cloth else None,
                    colliders=surfaces.colliders, meshes=surfaces.meshes)
            march_look = self._lume_volume(b, r, scene, vol, look, fire, surfaces, footage, final, t)
            stage = None
            if stage_on:
                light = stage_mod.fire_light(scene, frame, look, comp)
                light.lamps = r._lamps_on
                self._stage_env(scene, light, frame, plate=plate, cs=cs, plate_fit=plate_fit)
                size = r.plate_size if footage else (W, H)
                self.stage.heat = self._stage_heat(frame, surfaces.colliders)   # (hot objects glow: objheat_engine.py)
                stage = self.stage.draw(b, r, scene, cs, fire, surfaces.colliders, surfaces.meshes, light, comp, size,
                                        plate_fit=plate_fit, samples=samples, shutter=shutter, footage=footage, vol=vol,
                                        ground_y=vol.origin[1], frame=frame, pieces=pieces, ropes=ropes, matter=matter, bolts=bolts,
                                        shots=shots,
                                        grass=self._strands.ground_map(b) if grass else None, burns=surfaces, final=final)
                if self.stage.has_pieces or self.stage.has_matter:   # the march stops at the pieces and the matter too
                    r.hold_stage = self.stage.hold
                    r.hold_stage_matte = bool(footage and r.hold is not None and r.hold_on[0])
            if samples == 1:
                lim = cloth_pass(b, 0, (0.0, 0.0))
                r.march(b, vol, cs, fire, march_look, (fw, fh), seed=base_seed, shutter=shutter, ground=ground, time=t,
                        surfaces=surfaces, comp=comp, plate_fit=plate_fit, deep_pass=0, limit=lim)
                if cloth or grass:
                    self.raster.merge(b, r, 0, 1, 0.35 * (vol.cell or vol.h))
            else:
                self._ensure_acc(fw, fh)
                self._ensure_acc2(fw, fh)
                r._ensure_fire(fw, fh)
                set0, set1 = self._acc[0:3], self._acc[3:6]
                sur0, sur1 = self._acc2[0:3], self._acc2[3:6]
                srcs2 = [r.surf, r.mask, r.lamp_surf]
                for i in range(samples):
                    jit = (halton(i + 1, 2) - 0.5, halton(i + 1, 3) - 0.5)
                    lim = cloth_pass(b, i, jit)
                    r.march(b, vol, cs, fire, march_look, (fw, fh), jitter=jit, seed=base_seed + i, shutter=shutter,
                            ground=ground, time=t, surfaces=surfaces, comp=comp, plate_fit=plate_fit, deep_pass=i, limit=lim,
                            vec_pass=i)
                    if cloth or grass:
                        self.raster.merge(b, r, i, samples, 0.35 * (vol.cell or vol.h))
                    src_in, dst = (set0, set1) if i % 2 == 0 else (set1, set0)
                    b.run(self.k_accum, [r.beauty, r.emit, r.aux, *src_in, *dst], Uniforms().v4(fw, fh, 0, i), (fw, fh, 1))
                    s_in, s_dst = (sur0, sur1) if i % 2 == 0 else (sur1, sur0)
                    b.run(self.k_accum, [*srcs2, *s_in, *s_dst], Uniforms().v4(fw, fh, 0, i), (fw, fh, 1))
                last, other = (set1, set0) if (samples - 1) % 2 == 0 else (set0, set1)
                b.run(self.k_accum, [r.beauty, r.emit, r.aux, *last, *other], Uniforms().v4(fw, fh, 1, samples), (fw, fh, 1))
                b.run(self.k_copy, [*other, r.beauty, r.emit, r.aux], Uniforms().v4(fw, fh), (fw, fh, 1))
                slast, sother = (sur1, sur0) if (samples - 1) % 2 == 0 else (sur0, sur1)
                b.run(self.k_accum, [*srcs2, *slast, *sother], Uniforms().v4(fw, fh, 1, samples), (fw, fh, 1))
                b.run(self.k_copy, [*sother, r.surf, r.mask, r.lamp_surf], Uniforms().v4(fw, fh), (fw, fh, 1))
            if ep.enabled and self.embers.count:
                self.embers.draw(b, r, cs, fire, ep, look, (fw, fh), scene.fps)
            r.defocus(b, comp, cs, fire)
            r.bloom(b, comp.bloom_radius)
            r.composite(b, (W, H), comp, time=t, frame=frame, plate_fit=plate_fit, stage=stage)
        r.hold_stage = None
        r.L0_march = r.L1_march = r.LV_lume = r.LVD_lume = None
        self.last_render_ms = (time.perf_counter() - t0) * 1000.0
        return cs

    def _lume_volume(self, b, r, scene, vol, look, fire, surfaces, footage, final, t):
        """With Lume on: the light scattered in the smoke traced by Lume (lume_volume.py) for the ray march, and the
        look the march then uses. Else the look as it is."""
        from . import lume as LU
        from .lume_volume import LumeVolume, march_look
        from ..scene.materials import FLOORS
        if not LU.settings(scene).on or vol.scal is None:
            return look
        if getattr(self, '_lvol', None) is None:
            self._lvol = LumeVolume(self.gpu)
        rows = stage_mod.looks(scene, footage)
        albedos = []
        for row in rows[:len(surfaces.colliders)]:
            drawn, colour, _rough, metal, clear = row[:5]
            albedos.append((tuple(float(x) * (1.0 - 0.7 * float(metal)) for x in colour), drawn != stage_mod.NOT_DRAWN and clear < 0.5))
        cd = scene.data['composite']
        fl = FLOORS.get(cd.get('floor', 'concrete'), FLOORS['concrete'])
        floor = (0.25, 0.25, 0.25) if footage else tuple(float(c) * float(k) for c, k in zip(fl.colour, cd.get('floor_tint', (1.0, 1.0, 1.0))))
        ground = bool(scene.data['domain']['ground'])
        key = (round(float(t), 6), repr(look), floor, ground,
               tuple((tuple(c.pos), tuple(c.size), float(c.rot_y), tuple(c.quat)) for c in surfaces.colliders))
        self._lvol.compute(b, r, vol, look, fire, surfaces.colliders, surfaces.meshes, albedos, floor, ground,
                           float(vol.origin[1]), final, t, key=key)
        return march_look(look)

    def _carried_lamps(self, scene, frame, look):
        """Lights attached to falling or floating objects, where those are at `frame`."""
        if any(l['child'][0] == 'light' for l in getattr(scene, 'links', None) or []):
            carried = attached(scene, 'light', self.floating_overrides(frame))
            if carried:
                look.lamps = scene.lamps(frame, moved=carried)

    def body_field_for(self, grid='gas'):
        f = self._body_fields.get(grid)
        if f is None:
            f = self._body_fields[grid] = BodyField(self.gpu)
        return f

    @property
    def body_field(self):
        return self.body_field_for('gas')

    def _pieces_for(self, scene, solver, grid='gas'):
        """Upload this frame's broken pieces for a solver's grid; True if there are any."""
        sub = getattr(self.solids, 'substep_pieces', None)
        if not (self.solids.sets or self.solids.asms) or not sub or not any(sub):
            return False
        return self.body_field_for(grid).prepare(scene, sub, solver.dims, solver.h, solver.origin)

    @property
    def stage(self):
        if self._stage is None:
            self._stage = stage_mod.Stage(self.gpu)
        return self._stage

    def _environment(self, scene, frame=None, plate=None, cs=None, plate_fit=(1.0, 1.0)):
        """The environment round the set this render, or None: (what, key, HDRI image, rotation in degrees, strength), what
        being 'footage' (Environment from the footage: footage_env.py), 'sky' (the physical sky: sky.py) or 'file' (the
        Environment (HDRI), its image kept while the file stays the same)."""
        import os
        from . import footage_env as FE
        fe = FE.of_scene(scene, frame, plate, cs.view_proj if cs is not None else None, plate_fit)
        if fe is not None:
            return 'footage', fe[0], fe[1], 0.0, 1.0
        from . import sky as sky_mod
        phys = sky_mod.of_scene(scene, frame if frame is not None else scene.start)
        if phys is not None:
            return 'sky', phys[0], phys[3], 0.0, phys[4]
        lt = scene.data['lighting']
        path = scene.mesh_path(lt.get('environment', ''))   # (relative to the project, as the liquid engine has it)
        try:
            key = (path, os.path.getmtime(path)) if path else None
        except OSError:
            key = None
        if key is None:
            return None
        hdri = getattr(self, '_hdri', None)
        if hdri is None or hdri[0] != key:
            from ..io.hdri import load_hdri
            try:
                img = load_hdri(path)
            except Exception as ex:   # (a file it cannot read: lit as with none)
                log.warning('Environment %s unavailable: %s', path, ex)
                img = None
            hdri = self._hdri = (key, img)
        if hdri[1] is None:
            return None
        return 'file', key, hdri[1], float(lt.get('env_rotation', 0.0)), float(lt.get('env_strength', 1.0))

    def _env_light(self, env):
        """(its average light on an upward surface, (azimuth with its rotation, elevation, colour) of its brightest spot: the
        sun) of an environment from _environment, kept while its key is the same."""
        got = getattr(self, '_env_stats', None)
        if got is None or got[0] != env[1]:
            from ..io.hdri import brightest, sky_average
            got = self._env_stats = (env[1], sky_average(env[2]), brightest(env[2]))
        az, el, colour = got[2]
        return got[1], (az + env[3], el, colour)

    def _env_look(self, scene, look, frame, plate=None, cs=None, plate_fit=(1.0, 1.0)):
        """The environment's light on the smoke, as the liquid engine has it on the liquid: an HDRI file's average is the
        ambient (it is the set's sky too: _stage_env); with Key light from environment, the key light comes from its
        brightest spot (the sun), in its colour. (The physical sky's light is the look's already: Scene.look; the footage's
        leaves the ambient to Match ambient to footage.)"""
        lt = scene.data['lighting']
        if not (lt.get('environment') or lt.get('env_sun')):
            return      # (nothing in it for the smoke: the footage's HDRI is then built only for the set)
        env = self._environment(scene, frame, plate, cs, plate_fit)
        if env is None or env[0] == 'sky':
            return
        sky, (az, el, colour) = self._env_light(env)
        if env[0] == 'file':
            look.ambient = tuple(float(c) * env[4] for c in sky)
        if lt.get('env_sun'):
            strength = float(look.sun_intensity) * max(look.sun_color) or 3.0
            look.sun_azimuth, look.sun_elevation = az, el
            look.sun_color, look.sun_intensity = tuple(colour), strength

    def _stage_env(self, scene, light, frame=None, plate=None, cs=None, plate_fit=(1.0, 1.0)):
        """The environment (_environment) behind the stage, and its light as the sky's: its average from _env_light (kept
        by its key here; the stage keeps only its texture through a render without one)."""
        env = self._environment(scene, frame, plate, cs, plate_fit)
        if env is None:
            self.stage.env_sky = None
            return
        _what, key, img, rot, k = env
        sky, _sun = self._env_light(env)
        light.env, light.env_rotation, light.env_strength = self.stage.environment_image(key, img), rot, k
        light.env_image = (key, img)   # (Lume's too: a file as found, not as typed)
        self.stage.env_sky = tuple(float(c) for c in sky)   # (for Lume's pick of it among the lights)
        light.sky = tuple(c * k for c in self.stage.env_sky)

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
            out['lamps'] = self.gpu.read(r.lamp_surf)
            if getattr(self, '_cloth_drawn', False):
                out['fabric'] = self.gpu.read(self.raster.fabric)
        return out

    def linear_comp(self):
        return self.renderer.read_linear()

    # -- info ------------------------------------------------------------------------------------

    _cap_notes = ()
    _col_names = ()

    def notices(self):
        """What the person making the shot should know was left out or cut short, in words: what the scene's caps leave
        out (scene/caps.py, checked in prepare, with a disk cache that would not open), and what the simulation's parts
        said as they set up and ran: the matter, the grass, the rigid bodies, meshes that could not be used, weather that
        found no room to fall, the liquid's push on things held back, glowing matter past its lights and ice past its
        pieces (_list_notices), an OCIO set-up that failed, and what a render's compositing passes cut short. The viewer
        shows them and the command line prints them."""
        from pathlib import Path
        from .solids import MAX_ACCEL
        out = list(self._cap_notes)
        if self.kind != 'cloud':
            if self._matter is not None and self._matter.specs:
                out += self._matter.warnings
            if self._strands is not None:
                out += self._strands.warnings
        out += self.solids.warnings
        for i, n in sorted(getattr(self.solids, 'capped', {}).items()):
            name = self._col_names[i] if i < len(self._col_names) else f'Object {i + 1}'
            out.append(f'{name}: the liquid pushed it harder than {MAX_ACCEL:g} m/s² ({n} step{"" if n == 1 else "s"}), '
                       'so its push was held to that: it may move less than it should.')
        W = self.weather
        if self.kind in ('liquid', 'both') and self._wx_on and W is not None and (W.short or W.fill_short):
            out.append(f'The weather is past its particle limit ({W.capacity / 1e6:g} M falling pieces): '
                       f'{W.short + W.fill_short:,} were not made, so it falls thinner than its rate. Raise Weather › '
                       'Particle limit or make its Area smaller.')
        out += self._list_notices()
        for src, msg in sorted(self.solver.meshes.errors.items()):
            out.append(msg if str(src) in msg else f'The mesh {Path(str(src)).name} could not be used: {msg}')
        if getattr(self, '_ocio_error', None):
            out.append(f'OCIO could not be used ({self._ocio_error}): the Standard view and sRGB footage are used instead.')
        out += getattr(getattr(self, '_stage', None), 'notes', None) or []   # (a render job's compositing passes)
        return list(dict.fromkeys(out))

    def _list_notices(self):
        """What the GPU's fixed-size lists left out (the brightest and the first are kept, the same every run): glowing
        matter past its lights, and pieces of ice past the rigid bodies."""
        from .liquid_thermal import MAX_BODIES
        from .stage import MATTER_LIGHTS
        out = []
        stage = getattr(self, '_stage', None)
        cut = stage.matter_lights_cut() if stage is not None else 0
        if cut:
            out.append(f'Glowing matter lights what is round it from at most {MATTER_LIGHTS} patches of its surface (fewer '
                       f'with hot objects): up to {cut:,} more glowed, and only the brightest light the set.')
        th = getattr(getattr(self, 'liquid', None), 'thermal', None) if self.kind in ('liquid', 'both') else None
        most = th.most_pieces() if th is not None else 0
        if most > MAX_BODIES:
            out.append(f'The liquid froze into {most:,} separate pieces of ice at once: only {MAX_BODIES:,} of them move as '
                       'solid pieces, the rest move with the water.')
        return out

    def stats(self):
        out = self._stats()
        out['notices'] = self.notices()
        return out

    def _stats(self):
        if self.kind == 'liquid' and self.liquid is not None:
            return self._stats_liquid()
        if self.kind == 'cloud' and self.cloud is not None:
            return self._stats_cloud()
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
