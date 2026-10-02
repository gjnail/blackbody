"""Matter that burns (dry leaves, sawdust, coal: matter.py, mpm_heat.wgsl) feeding the fire: once a frame its burning
particles give their fuel, heat and smoke to a coupling grid in the cloth's layout (mpm_fire.wgsl), which goes into the
gas every substep (cloth_feed.wgsl), as burning pieces' do (piece_fire.py)."""
from __future__ import annotations

import math

import numpy as np

from .gpu import GPU, Uniforms, groups_1d
from .piece_fire import COUPLE_CHANNELS, COUPLE_REF_FUEL
from .strands import FUEL_PER_KG


class MatterFire:
    def __init__(self, gpu: GPU):
        self.gpu = gpu
        g = gpu
        res = ['rbuf', 'buf']
        self.k_clear = g.kernel('mpm_fire.wgsl', res, entry='clear', workgroup=(64, 1, 1))
        self.k_splat = g.kernel('mpm_fire.wgsl', res, entry='splat', workgroup=(64, 1, 1))
        self.k_feed = g.kernel('cloth_feed.wgsl', ['utex3d', 'rbuf', 'st3d:rgba16float:w'], workgroup=(4, 4, 4))
        self.G = None
        self.couple_dims = None
        self._splatted = False

    def splat(self, b, solver, matter, sp, ambient_k=293.0, flame_k=1650.0):
        """The matter's burning particles onto the coupling grid (once a frame, inside the frame's batch): each gives the
        fuel of what of it burns away a second (FUEL_PER_KG, as burning grass), at its burning temperature, with Spreading
        fire's Smoke (sp: the scene's spread settings)."""
        self._splatted = False
        if matter is None or not matter.burns() or solver is None or solver.dims is None:
            return
        cd = tuple(int(math.ceil(d / 2)) for d in solver.dims)
        if cd != self.couple_dims or self.G is None:
            if self.G is not None:
                self.G.destroy()
            self.G = self.gpu.buffer(int(np.prod(cd)) * COUPLE_CHANNELS * 4, 'matter-fire-couple')
            self.couple_dims = cd
        cells = int(np.prod(cd))
        u = (Uniforms().v4(*matter.origin, matter.dx).v4(matter.count, cells).v4(*cd, 2.0 * solver.h).v4(*solver.origin)
             .raw(matter.fire_table(FUEL_PER_KG, float(sp['smoke']), ambient_k, flame_k)))
        res = [matter._buf['P'], self.G]
        b.run(self.k_clear, res, u, groups=groups_1d(cells * COUPLE_CHANNELS))
        b.run(self.k_splat, res, u, groups=groups_1d(matter.count))
        self._splatted = True

    def hook(self, b, solver, dt, stage):
        """Solver.cloth_hook: the burning matter feeds the gas."""
        if not self._splatted or self.couple_dims != tuple(int(math.ceil(d / 2)) for d in solver.dims):
            return
        cd = self.couple_dims
        hc = 2.0 * solver.h
        if stage == 'sources':
            u = Uniforms().v4(*solver.dims, solver.h).v4(*cd, hc).v4(dt, COUPLE_REF_FUEL)
            b.run(self.k_feed, [solver.scal[0], self.G, solver.scal[1]], u, solver.dims)
            solver.scal.reverse()
        elif stage == 'sources_fine' and solver.scal_fine is not None:
            u = Uniforms().v4(*solver.dims_fine, solver.h / solver.upres).v4(*cd, hc).v4(dt, COUPLE_REF_FUEL)
            b.run(self.k_feed, [solver.scal_fine[0], self.G, solver.scal_fine[1]], u, solver.dims_fine)
            solver.scal_fine.reverse()
