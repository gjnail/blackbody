"""Sand, snow, mud, jelly and clay (matter.py) as solid to a grid solver, the gas or the liquid: its particles counted into
the solver's cells once a frame, then folded into the solver's distance to solids and its solid velocity each substep
(matter_field.wgsl), alongside the broken pieces (bodyfield.py: the solver's pieces_step)."""
from __future__ import annotations

import numpy as np

from .gpu import GPU, Uniforms, groups_1d


class MatterField:
    def __init__(self, gpu: GPU):
        self.gpu = gpu
        res = ['rbuf', 'buf', 'st3d:r32float:rw', 'st3d:rgba16float:w']
        self.k_clear = gpu.kernel('matter_field.wgsl', res, entry='clear', workgroup=(64, 1, 1))
        self.k_splat = gpu.kernel('matter_field.wgsl', res, entry='splat', workgroup=(64, 1, 1))
        self.k_bake = gpu.kernel('matter_field.wgsl', res, entry='bake', workgroup=(8, 8, 4))
        self.A = None
        self._cells = 0
        self.ready = False
        self._u = None

    def prepare(self, b, solver, matter):
        """Count the matter's particles into the solver's cells (once a frame, in batch b). False when there is none."""
        self.ready = False
        if matter is None or not matter.active or not matter.count or solver.dims is None:
            return False
        cells = int(np.prod(solver.dims))
        if self.A is None or self._cells != cells:
            if self.A is not None:
                self.A.destroy()
            self.A = self.gpu.buffer(cells * 16, 'matter-field')
            self._cells = cells
        share = (matter.dx / 2.0) ** 3 / solver.h ** 3        # (eight particles to a matter cell)
        u = solver._grid(0.0).v4(*matter.origin, matter.dx).v4(matter.count, share)
        self._u = u
        self._matter = matter
        res = [matter._buf['P'], self.A, solver.SDF if hasattr(solver, 'SDF') else solver.sdf, solver.solid_vel()]
        b.run(self.k_clear, res, u, groups=groups_1d(4 * cells))
        b.run(self.k_splat, res, u, groups=groups_1d(matter.count))
        self.ready = True
        return True

    def bake(self, b, solver, i, after=False):
        """Fold the matter into the solver's distance to solids and solid velocity for substep i. after: broken pieces
        were baked first this substep (their velocity stays where there is no matter)."""
        if not self.ready:
            return False
        u = Uniforms().raw(solver._grid(0.0).data)
        u.v4(*self._matter.origin, self._matter.dx).v4(self._matter.count, (self._matter.dx / 2.0) ** 3 / solver.h ** 3,
                                                       1.0 if after else 0.0)
        res = [self._matter._buf['P'], self.A, solver.SDF if hasattr(solver, 'SDF') else solver.sdf, solver.solid_vel()]
        b.run(self.k_bake, res, u, solver.dims)
        return True


class Fields:
    """Several things folded into a solver's solids each substep (the solver's pieces_step): broken pieces first, then
    the matter."""

    def __init__(self, pieces=None, matter=None):
        self.pieces, self.matter = pieces, matter

    def bake(self, b, solver, i):
        on = bool(self.pieces.bake(b, solver, i)) if self.pieces is not None else False
        if self.matter is not None:
            on = self.matter.bake(b, solver, i, after=on) or on
        return on
