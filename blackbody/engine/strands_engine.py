"""Grass (strands.py) in the engines: laid out with the scene, grown onto the ground and the objects at the first step,
stepped with the gas at each of its substeps (dragged by it, and feeding it where it burns), cached as bytes a blade,
and drawn into the raster layer (raster.py) with the cloth, before the march."""
from __future__ import annotations

from .raster import Raster
from .strands import Strands


class StrandsEngine:
    """Mixed into Engine (engine.py): the grass of every kind of scene but the sky."""

    _strands = None
    _raster_layer = None

    @property
    def strands(self) -> Strands:
        if self._strands is None:
            self._strands = Strands(self.gpu)
        return self._strands

    @property
    def raster(self) -> Raster:
        """The raster layer the cloth and the grass draw into (one, so each hides what is behind it of the other)."""
        if self._raster_layer is None:
            self._raster_layer = Raster(self.gpu)
        self.cloth.raster = self._raster_layer
        return self._raster_layer

    @property
    def strands_on(self):
        return self._strands is not None and self._strands.active

    def _prepare_strands(self, scene):
        """Match the grass to the scene. True when it grows again."""
        specs = scene.strand_specs() if scene.kind != 'cloud' and hasattr(scene, 'strand_specs') else []
        if not specs and self._strands is None:
            return False
        return self.strands.configure(specs)

    def _solver_hook(self):
        """Solver.cloth_hook: the burning cloth and the burning grass feed the gas (the cloth also holds the air)."""
        hooks = []
        if self.cloth.active:
            hooks.append(self.cloth.hook)
        if self.strands_on and self._strands.burns:
            hooks.append(self._strands.hook)
        if self.solids.active and self.solids.burning:   # (things that break and burn: engine._burn_pieces)
            hooks.append(self.piece_fire.hook)
        if getattr(self, '_matter_burns', None) is not None and self._matter_burns():   # (matter_engine)
            hooks.append(self.matter_fire.hook)
        if not hooks:
            return None
        if len(hooks) == 1:
            return hooks[0]

        def both(b, solver, dt, stage):
            for h in hooks:
                h(b, solver, dt, stage)
        return both

    @staticmethod
    def _fixed_mask(scene):
        """A bit for each enabled collider (in Scene.colliders_gpu's order) that stays put: grass grows on those."""
        from .solids import falls
        mask, j = 0, 0
        for i, c in enumerate(scene.colliders):
            if not c['enabled']:
                continue
            moves = falls(c) or c.get('floating') or any(scene.curve(('collider', i, k)) is not None
                                                         for k in ('position', 'yaw', 'pitch', 'roll', 'size'))
            if not moves and j < 16:
                mask |= 1 << j
            j += 1
        return mask

    def _step_strands(self, b, scene, frame, dt, solver, colliders, meshes, look, ground, ground_y, wind, gust):
        """The grass through dt seconds (a solver substep): grown first if it has not been, then dragged by the air,
        pushed by the objects; where it burns, its fuel onto the coupling grid for the gas's next substep."""
        s = self._strands
        if s is None or not s.active:
            return
        if not s.placed:
            s.place(b, colliders, meshes, ground, ground_y, self._fixed_mask(scene))
        s.step(b, dt, solver, colliders, meshes, wind, gust, look, ground, ground_y)
        if solver is not None:
            s.splat(b, solver, look)

    def _snapshot_strands(self, entry):
        s = self._strands
        if s is not None and s.active:
            snap = s.snapshot()
            if snap is not None:
                entry['strands'] = snap

    def _checkpoint_strands(self, entry):
        s = self._strands
        if s is not None and s.active:
            st = s.save_state()
            if st is not None:
                entry['strands_state'] = st

    def _resume_strands(self, entry):
        """Back to a checkpoint's grass: False when the grass changed since (the checkpoint cannot carry on)."""
        s = self._strands
        if s is None or not s.active:
            return True
        return s.load_state(entry.get('strands_state'))

    def strands_for(self, frame):
        """Show the grass of `frame` (the live blades, or the cache's). False when there is none to draw."""
        s = self._strands
        if s is None or not s.active or not s.placed:
            return False
        if self.sim_frame == frame:
            s.use_view(None)
            return True
        entry = self.cache.get(frame) if self.cache is not None else None
        if entry is None or entry.get('strands') is None:
            return False
        s.use_view(entry['strands'])
        return s._view_on

    def _draw_raster(self, b, r, cs, fire, look, size, jit, shutter, solver, cloth, grass, light_gain=1.0, fire_lights=True,
                     lamp_count=None):
        """Draw the cloth and the grass into the raster layer for one anti-aliasing pass. Its aux (the march's limit),
        or None when neither is there."""
        if not (cloth or grass):
            return None
        w, h = size
        rl = self.raster
        rp = rl.begin(b, w, h)
        if cloth:
            self.cloth.draw(b, r, cs, fire, look, size, jitter=jit, shutter=shutter, solver=solver, light_gain=light_gain,
                            fire_lights=fire_lights, lamp_count=lamp_count, rp=rp)
        if grass:
            self._strands.draw(b, rp, r, cs, fire, look, size, jitter=jit, solver=solver, light_gain=light_gain,
                               fire_lights=fire_lights, lamp_count=lamp_count)
        rp.end()
        return rl.aux
