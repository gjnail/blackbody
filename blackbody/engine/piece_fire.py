"""Things that break and burn, on the GPU side (solids.py keeps each piece's fire): the gas's temperature round the
pieces, taken once a frame, and the burning pieces' fuel, heat and smoke given to the gas at every substep through a
coupling grid in the cloth's layout (points_splat.wgsl, then cloth_feed.wgsl)."""
from __future__ import annotations

import math

import numpy as np

from .gpu import GPU, Uniforms, groups_1d

COUPLE_CHANNELS = 13      # cloth_splat.wgsl's layout
COUPLE_REF_FUEL = 2.0     # fuel density (F/s) at which the heat is fully in the gas (as cloth's)


class PieceFire:
    def __init__(self, gpu: GPU):
        self.gpu = gpu
        g = gpu
        self.k_sample = g.kernel('solids_air.wgsl', ['tex3d', 'tex3d', 'smp', 'rbuf', 'buf'], workgroup=(64, 1, 1))
        self.k_clear = g.kernel('points_splat.wgsl', ['rbuf', 'buf'], entry='clear', workgroup=(64, 1, 1))
        self.k_splat = g.kernel('points_splat.wgsl', ['rbuf', 'buf'], entry='splat', workgroup=(64, 1, 1))
        self.k_feed = g.kernel('cloth_feed.wgsl', ['utex3d', 'rbuf', 'st3d:rgba16float:w'], workgroup=(4, 4, 4))
        self._pts = None          # (capacity, points buffer, values buffer)
        self._fuel = None         # (capacity, buffer)
        self.G = None
        self.couple_dims = None
        self.n = 0                # fuel points this frame

    def temperatures(self, solver, points):
        """The gas's temperature (field units: 0 the air, 1 the flame) at fire-local `points` (m, 3); 0 outside the box."""
        n = len(points)
        if n == 0 or solver is None or not solver.dims:
            return np.zeros(n)
        g = self.gpu
        if self._pts is None or self._pts[0] < n:
            cap = max(64, n)
            self._pts = (cap, g.buffer(cap * 16, 'piece-points'), g.buffer(cap * 16, 'piece-gas'))
        cap, pb, vb = self._pts
        p4 = np.zeros((cap, 4), np.float32)
        p4[:n, :3] = points
        g.write_buffer(pb, p4)
        with g.batch() as b:
            b.run(self.k_sample, [solver.vel[0], solver.scal[0], g.linear, pb, vb], Uniforms().v4(*solver.origin, solver.h).v4(*solver.dims, n),
                  groups=groups_1d(n))
        out = np.frombuffer(g.read_buffer(vb, n * 16), np.float32).reshape(n, 4)
        return np.nan_to_num(out[:, 3].astype(float))

    def prepare_frame(self, solver, points, values):
        """This frame's burning: the fuel points (fire-local m) and their (fuel F m^3/s, heat, smoke /s). Sizes the coupling
        grid for the solver's grid (before the frame's commands are recorded)."""
        self.n = len(points)
        if self.n == 0 or solver is None or solver.dims is None:
            self.n = 0
            return
        g = self.gpu
        cd = tuple(int(math.ceil(d / 2)) for d in solver.dims)
        if cd != self.couple_dims or self.G is None:
            if self.G is not None:
                self.G.destroy()
            self.G = g.buffer(int(np.prod(cd)) * COUPLE_CHANNELS * 4, 'piece-fire-couple')
            self.couple_dims = cd
        if self._fuel is None or self._fuel[0] < self.n:
            cap = max(64, self.n)
            self._fuel = (cap, g.buffer(cap * 32, 'piece-fuel'))
        data = np.zeros((self._fuel[0], 2, 4), np.float32)
        data[:self.n, 0, :3] = points
        data[:self.n, 0, 3] = values[:, 0]
        data[:self.n, 1, 0] = values[:, 1]
        data[:self.n, 1, 1] = values[:, 2]
        g.write_buffer(self._fuel[1], data.reshape(-1, 4))
        self._splatted = False

    def splat(self, b, solver):
        """The fuel points onto the coupling grid (once a frame, inside the frame's batch)."""
        if self.n == 0 or self.G is None:
            return
        cd = self.couple_dims
        cells = int(np.prod(cd))
        u = Uniforms().v4(self.n, cells).v4(*cd, 2.0 * solver.h).v4(*solver.origin)
        b.run(self.k_clear, [self._fuel[1], self.G], u, groups=groups_1d(cells * COUPLE_CHANNELS))
        b.run(self.k_splat, [self._fuel[1], self.G], u, (self.n, 1, 1))
        self._splatted = True

    def hook(self, b, solver, dt, stage):
        """Solver.cloth_hook: the burning pieces feed the gas."""
        if self.n == 0 or not getattr(self, '_splatted', False):
            return
        if self.couple_dims != tuple(int(math.ceil(d / 2)) for d in solver.dims):
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
