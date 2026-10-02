"""Matter (sand, snow, mud, jelly, clay: matter.py) in the engines: set up with the box, stepped once a frame after the
rigid bodies (it meets the objects where they are at each of the frame's substeps, and pushes the falling ones back),
cached as its particles' 8-byte form, and drawn on the stage (stage.py)."""
from __future__ import annotations

import numpy as np

from .matter import Matter
from .matter_field import Fields, MatterField
from .solver import pack_colliders


class MatterEngine:
    """Mixed into Engine (engine.py): the matter of every kind of scene but the sky."""

    _matter = None

    @property
    def matter(self) -> Matter:
        if self._matter is None:
            self._matter = Matter(self.gpu)
        return self._matter

    def _prepare_matter(self, scene, layout):
        """Match the matter to the scene and its box (layout: the simulation grid's dims, cell, corner). True when it
        starts again."""
        specs = scene.matter_specs() if scene.kind != 'cloud' else []
        if not specs and self._matter is None:
            return False
        dims, h, origin = layout
        d = scene.data['domain']
        size = np.asarray(dims, float) * float(h)
        gravity = float(scene.data['liquid']['gravity']) if scene.kind in ('liquid', 'both') else 9.81
        duration = (scene.end - scene.start + 1) / scene.fps + float(d['preroll'])
        return self.matter.configure(specs, origin, size, resolution=int(d.get('matter_detail', 128)),
                                     max_particles=int(d.get('matter_particles', 2_000_000)), gravity=gravity,
                                     ground=bool(d['ground']), closed=not bool(d['open_sides']), fps=scene.fps,
                                     duration=duration)

    _coupled = False

    def _matter_couple(self, scene, frame, fdt, meshes):
        """When there are things that fall: (longest step, fn) for the rigid bodies to step in lockstep with the matter
        through frame `frame` (Solids.advance), so the sand holds them up and they dent it. None otherwise."""
        m = self._matter
        if m is None or not m.active or not self.solids.active or not self.solids.bodies:
            return None
        self._blast_matter(scene, frame)
        dt, _n = m.begin(fdt)
        enabled = [i for i, c in enumerate(scene.colliders) if c['enabled']]
        atlas = meshes.atlas if meshes is not None else None
        self._coupled = True

        def fn(h, poses, f):
            cols = scene.colliders_gpu(frame - 1 + f, poses)
            pushed = m.step(h, cols, atlas, lambda u, c: pack_colliders(u, c, meshes))
            return {enabled[k]: v for k, v in pushed.items() if k < len(enabled)}
        return dt, fn

    def _step_matter(self, scene, frame, fdt, poses, n, meshes):
        """Move the matter through frame `frame` (fdt seconds), meeting the objects as they are at each of the frame's n
        substeps (poses: the falling ones', from the rigid bodies). (With things that fall it has already moved, in
        lockstep with them: _matter_couple.)"""
        m = self._matter
        if m is None or not m.active:
            return
        if self._coupled:
            self._coupled = False
            m.end()
            return
        self._blast_matter(scene, frame)
        cache = {}

        def cols_at(f):
            i = min(int(f * n), n - 1)
            if i not in cache:
                cache[i] = scene.colliders_gpu(frame - 1 + (i + 0.5) / n, poses[i] if poses else None)
            return cache[i]

        m.advance(fdt, cols_at, meshes.atlas if meshes is not None else None, lambda u, c: pack_colliders(u, c, meshes))

    _mfields = None

    def _matter_solid(self, b, solver, grid='gas'):
        """Count the matter into a solver's cells for this frame (in batch b), so the gas or the liquid goes round it: its
        MatterField, or None when there is no matter."""
        m = self._matter
        if m is None or not m.active or not m.count or solver is None or solver.dims is None:
            return None
        if self._mfields is None:
            self._mfields = {}
        f = self._mfields.get(grid)
        if f is None:
            f = self._mfields[grid] = MatterField(self.gpu)
        return f if f.prepare(b, solver, m) else None

    @staticmethod
    def _solids_step(pieces, matter, i):
        """A solver's pieces_step for substep i: the broken pieces (a BodyField, or None) and the matter (a MatterField,
        or None) folded into its solids."""
        if pieces is None and matter is None:
            return None
        return (Fields(pieces, matter), i)

    def _melt_matter(self, scene, frame, fdt, gas=None, liquid=None):
        """Snow melting through frame `frame` (fdt seconds) in the gas's heat, its water joining the liquid."""
        m = self._matter
        if m is None or not m.melts():
            return
        look = scene.look(frame)
        with self.gpu.batch() as b:
            m.melt(b, fdt, gas, liquid, look.ambient_k, look.flame_k)

    def _blast_matter(self, scene, frame):
        """The blasts that go off in frame `frame` throw the matter (at the frame's start)."""
        for fb, where, kg in scene.blasts():
            if frame - 1 <= fb < frame:
                self._matter.blast(where, kg)

    def _snapshot_matter(self, entry):
        m = self._matter
        if m is not None and m.active:
            snap = m.snapshot()
            if snap is not None:
                entry['matter'] = snap

    def matter_for(self, frame):
        """The matter to draw at `frame` (the live particles, or the cache's), or None. Its surface is built here, before
        the render's batch (a batch of its own inside another would overwrite the other's parameters)."""
        m = self._matter
        if m is None or not m.active:
            return None
        if self.sim_frame == frame:
            m.show_live()
            if not m.count:
                return None
        else:
            entry = self.cache.get(frame) if self.cache is not None else None
            if entry is None or entry.get('matter') is None:
                return None
            m.show(entry['matter'])
        m.surface()
        return m
