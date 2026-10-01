"""The open water around the simulation box: a square layer of cells over the water, several times the
box's size, that carries on where the box stops.

Each simulated frame it takes from the box, per column: how far its surface stands from the sea
(a wake, the rings of a splash), how its surface water moves and how much foam floats on it. Then:

  waves    spread out from the box across the open water: Tessendorf's eWave, exact for linear
           waves on water of the sea's depth (wakes fan out into their V, splash rings keep
           spreading), carried by the current; islands and piers hold the water still; a sponge
           round the edge takes them out
  current  the box's own surface flow carries on outside it, round islands and posts, into the
           open water's current (a two-dimensional incompressible flow)
  foam     rides that current and the wind's pull, fades; the box's foam trails out past its
           edges; surf where waves break over the shallows; churned where the current meets a solid
  depth    the sea bed under it (colliders, terrain, the ground), for shoaling and surf

The renderer adds the layer's waves to the sea's, its foam to the sea's foam, and lets the sea's
waves feel the depth. Cached frames keep the layer as the renderer reads it.
"""
from __future__ import annotations

import math

import numpy as np

from .gpu import Uniforms, groups_1d
from .solver import pack_colliders

WAVE_SUBSTEPS = 2
JACOBI = 40


class OceanLayer:
    """The layer while simulating (the renderer reads it through texture() and uniforms())."""

    def __init__(self, gpu, meshes):
        self.gpu = gpu
        self.meshes = meshes
        g = gpu
        W = (8, 8, 1)
        self.k_count = g.kernel('ocl_count.wgsl', ['rbuf', 'rbuf', 'buf'], workgroup=(64, 1, 1))
        self.k_cols = g.kernel('ocl_cols.wgsl', ['utex3d', 'utex3d', 'utex2d', 'rbuf', 'st2d:rgba32float:w'], workgroup=W)
        self.k_depth = g.kernel('ocl_depth.wgsl', ['utex3d', 'st2d:rgba16float:w'], workgroup=W)
        wsig = ['utex3d', 'utex2d', 'utex2d', 'st3d:rgba32float:w']
        self.k_pre = g.kernel('ocl_wave.wgsl', wsig, 'pre', workgroup=W)
        self.k_prop = g.kernel('ocl_wave.wgsl', wsig, 'prop', workgroup=W)
        self.k_adv = g.kernel('ocl_adv.wgsl', ['utex2d', 'utex2d', 'utex2d', 'st2d:rg32float:w'], workgroup=W)
        self.k_div = g.kernel('ocl_div.wgsl', ['utex2d', 'utex2d', 'st2d:r32float:w'], workgroup=W)
        self.k_jac = g.kernel('ocl_jac.wgsl', ['utex2d', 'utex2d', 'utex2d', 'st2d:r32float:w'], workgroup=W)
        self.k_proj = g.kernel('ocl_proj.wgsl', ['utex2d', 'utex2d', 'utex2d', 'st2d:rg32float:w'], workgroup=W)
        self.k_foam = g.kernel('ocl_foam.wgsl', ['utex2d', 'utex2d', 'utex2d', 'utex2d', 'st2d:r32float:w'], workgroup=W)
        self.k_out = g.kernel('ocl_out.wgsl', ['utex3d', 'utex2d', 'utex2d', 'utex2d', 'st2d:rgba16float:w'], workgroup=W)
        self.n = 0
        self.on = False
        self.corner = (0.0, 0.0)
        self.size = 1.0
        self.fft = None
        self._tex = []
        self._cols_dims = None
        self._depth_key = None
        self.hs = 0.0
        self.kp = 1.0
        self.offset = 0.0

    # -- resources --------------------------------------------------------------------------------

    def _alloc(self, n):
        if n == self.n:
            return
        self.release()
        g = self.gpu
        from .fft import FFT2D
        self.n = n
        self.fft = FFT2D(g, n, 1, 'ocean-layer-waves')
        t2 = lambda fmt, label: self._keep(g.texture2d(n, n, fmt, label))
        self.TMP = self._keep(g.texture3d((n, n, 1), 'rgba32float', 'ocean-layer-tmp'))
        self.V = [t2('rg32float', f'ocean-layer-current{i}') for i in range(2)]
        self.P = [t2('r32float', f'ocean-layer-pressure{i}') for i in range(2)]
        self.DIV = t2('r32float', 'ocean-layer-divergence')
        self.F = [t2('r32float', f'ocean-layer-foam{i}') for i in range(2)]
        self.DEP = t2('rgba16float', 'ocean-layer-depth')
        self.OUT = t2('rgba16float', 'ocean-layer')
        self._depth_key = None
        self.reset()

    def _keep(self, t):
        self._tex.append(t)
        return t

    def release(self):
        for t in self._tex:
            t.destroy()
        self._tex = []
        if self.fft is not None:
            self.fft.destroy()
            self.fft = None
        self.n = 0

    def reset(self):
        """Still water: no waves, no current of its own, no foam."""
        if not self.n:
            return
        n = self.n
        g = self.gpu
        g.upload(self.fft.a, np.zeros((1, n, n, 4), np.float32))
        for t in self.V:
            g.upload(t, np.zeros((n, n, 2), np.float32))
        for t in self.P + self.F + [self.DIV]:
            g.upload(t, np.zeros((n, n, 1), np.float32))
        g.upload(self.OUT, np.zeros((n, n, 4), np.float16))
        self._vi = self._pi = self._fi = 0
        self._current = None
        self._cols_ready = False
        self.offset = 0.0

    def _ensure_cols(self, dims):
        key = (dims[0], dims[2])
        if key == self._cols_dims:
            return
        if self._cols_dims is not None:
            self.COLS.destroy()
            self.FCOUNT.destroy()
        self.COLS = self.gpu.texture2d(dims[0], dims[2], 'rgba32float', 'ocean-layer-columns')
        self.FCOUNT = self.gpu.buffer(dims[0] * dims[2] * 4, 'ocean-layer-foam-count')
        self._cols_dims = key

    # -- placement ----------------------------------------------------------------------------------

    def place(self, L, area, n):
        """Centre the layer on the box (area: its size as a multiple of the box's larger side). When
        the box has moved (Box follows) the layer moves by whole cells, keeping what it carries."""
        self._alloc(int(n))
        nx, ny, nz = L.dims
        size = max(nx, nz) * L.h * float(area)
        cx, cz = L.origin[0] + 0.5 * nx * L.h, L.origin[2] + 0.5 * nz * L.h
        dl = size / self.n
        if abs(size - self.size) > 1e-9 or not self.on:
            self.size = size
            self.corner = (cx - 0.5 * size, cz - 0.5 * size)
            self._depth_key = None
            self.reset()
        else:
            # re-centre by whole cells once the box has drifted a few cells off
            ex, ez = cx - (self.corner[0] + 0.5 * size), cz - (self.corner[1] + 0.5 * size)
            sx = int(round(ex / dl)) if abs(ex) > 3.0 * dl else 0
            sz = int(round(ez / dl)) if abs(ez) > 3.0 * dl else 0
            if sx or sz:
                self._shift(sx, sz)
                self.corner = (self.corner[0] + sx * dl, self.corner[1] + sz * dl)
                self._depth_key = None
        self.on = True

    def _shift(self, sx, sz):
        """Move the layer's contents by whole cells (what falls off is dropped; what comes in is still)."""
        g = self.gpu
        g.upload(self.fft.a, _shifted(g.read(self.fft.a)[0], sx, sz)[None])
        for lst in (self.V, self.F, self.P):
            for t in lst:
                a = g.read(t)
                g.upload(t, _shifted(a, sx, sz))

    def _layer_u(self, L, margin, dt):
        dl = self.size / self.n
        nx, ny, nz = L.dims
        return (Uniforms().v4(self.corner[0], self.corner[1], dl, self.n)
                .v4(L.origin[0], L.origin[2], L.h, margin)
                .v4(nx, nz, 3.0, dt))

    # -- one frame ------------------------------------------------------------------------------------

    def step(self, b, L, prm, dt, spec=None, wind=(0.0, 0.0), foam_life=12.0, surf=1.0, ground_is_bed=True, sea_depth=None):
        """One frame of dt seconds after the box's steps (L: the LiquidSolver). spec: the sea (OceanSpec
        or None); wind: fire-local (x, z) m/s."""
        if not self.on or L.dims is None:
            return
        n = self.n
        nx, ny, nz = L.dims
        level = prm.water_level
        depth = float(sea_depth if sea_depth else (spec.depth if spec is not None else level))
        margin = max(1.0, float(prm.level_absorb))
        cur = (float(prm.current[0]), float(prm.current[2]))
        self._ensure_cols(L.dims)
        self.offset = self._steady_offset(L, margin)
        # the sea bed, when the colliders or the level change
        dkey = (tuple(L.colliders), round(level, 4), ground_is_bed, round(depth, 3), self.corner, self.size)
        if dkey != self._depth_key:
            u = Uniforms().raw(self._layer_u(L, margin, dt).data).v4(level, 1.0 if ground_is_bed else 0.0,
                                                                     depth if not ground_is_bed else max(level, 0.05))
            pack_colliders(u, L.colliders, self.meshes)
            b.run(self.k_depth, [self.meshes.atlas, self.DEP], u, (n, n, 1))
            self._depth_key = dkey
        # what the box hands over
        b.clear_buffer(self.FCOUNT)
        gb = L._grid(0.0).tobytes()
        if L.ww_capacity > 64:
            b.run(self.k_count, [L.WA, L.WB, self.FCOUNT], gb + Uniforms().v4(L.ww_capacity).tobytes(),
                  groups=groups_1d(L.ww_capacity))
        vel = L.vel_tex if L.vel_tex is not None else L.VA
        b.run(self.k_cols, [L.DENS, vel, L.OCN, self.FCOUNT, self.COLS],
              gb + Uniforms().v4(float(prm.ppc), 3.0, L.ww_capacity, 0.0).tobytes(), (nx, nz, 1))
        # waves
        sub = WAVE_SUBSTEPS
        lu = self._layer_u(L, margin, dt / sub)
        dl = self.size / n
        # (the shortest waves the layer holds fade in about a second: grid noise does not build up)
        wu = Uniforms().raw(lu.data).v4(depth, prm.gravity, *cur).v4(1.5, 0.5 * dl * dl / math.pi ** 2, 1.0, self.offset)
        for _ in range(sub):
            b.run(self.k_pre, [self.fft.a, self.COLS, self.DEP, self.TMP], wu, (n, n, 1))
            b.copy_texture(self.TMP, self.fft.a)
            self.fft.run(b, inverse=False)
            b.run(self.k_prop, [self.fft.a, self.COLS, self.DEP, self.TMP], wu, (n, n, 1))
            b.copy_texture(self.TMP, self.fft.a)
            self.fft.run(b, inverse=True)
        # current
        lu = self._layer_u(L, margin, dt)
        V0, V1 = self.V[self._vi], self.V[1 - self._vi]
        b.run(self.k_adv, [V0, self.COLS, self.DEP, V1], Uniforms().raw(lu.data).v4(*cur), (n, n, 1))
        b.run(self.k_div, [V1, self.DEP, self.DIV], lu, (n, n, 1))
        for _ in range(JACOBI):
            P0, P1 = self.P[self._pi], self.P[1 - self._pi]
            b.run(self.k_jac, [P0, self.DIV, self.DEP, P1], lu, (n, n, 1))
            self._pi = 1 - self._pi
        b.run(self.k_proj, [V1, self.P[self._pi], self.DEP, V0], lu, (n, n, 1))
        # foam
        self.hs = float(math.hypot(spec.height, spec.swell)) if spec is not None and spec.waves_on else 0.0
        self.kp = 2.0 * math.pi / max(spec.peak, 0.05) if spec is not None and spec.waves_on else 1.0
        F0, F1 = self.F[self._fi], self.F[1 - self._fi]
        fu = Uniforms().raw(lu.data).v4(foam_life, 0.03 * wind[0], 0.03 * wind[1], self.hs).v4(self.kp, 0.25 * surf, 0.15, 0.0)
        b.run(self.k_foam, [F0, V0, self.COLS, self.DEP, F1], fu, (n, n, 1))
        self._fi = 1 - self._fi
        b.run(self.k_out, [self.fft.a, self.F[self._fi], self.DEP, V0, self.OUT], lu, (n, n, 1))

    def _steady_offset(self, L, margin):
        """How far the box's surface sits off the sea's on the whole (from the columns of the frame
        before): its surface as found from the particles sits a little low, and a hull displaces water.
        Only what moves about that is a wave to hand on."""
        if not getattr(self, '_cols_ready', False):
            self._cols_ready = True
            return self.offset
        a = self.gpu.read(self.COLS)[..., 0]
        m = int(margin) + 1
        inner = a[m:-m, m:-m] if min(a.shape) > 2 * m + 2 else a
        v = inner[inner > -1.0e3]
        return float(np.median(v)) if v.size else self.offset

    # -- what the renderer reads -------------------------------------------------------------------

    def texture(self):
        return self.OUT

    def uniforms(self, u):
        """ocx[8..9]: the layer's corner (x, z), 1 / its size (twice); on, texels across, the sea's
        significant height (m), its peak wavenumber (1/m)."""
        return layer_uniforms(u, self.corner, self.size, self.n, self.hs, self.kp)

    def read(self):
        """The layer as the renderer reads it, for the frame cache."""
        return {'out': self.gpu.read(self.OUT).copy(), 'corner': tuple(self.corner), 'size': self.size,
                'hs': self.hs, 'kp': self.kp}


def layer_uniforms(u, corner, size, n, hs, kp):
    u.v4(corner[0], corner[1], 1.0 / size, 1.0 / size)
    return u.v4(1.0, n, hs, kp)


def _shifted(a, sx, sz):
    """a (z, x, ...) moved so that what was at cell (x, z) is at (x - sx, z - sz); the rest zero."""
    out = np.zeros_like(a)
    n = a.shape[0]
    xs = slice(max(sx, 0), n + min(sx, 0))
    xd = slice(max(-sx, 0), n + min(-sx, 0))
    zs = slice(max(sz, 0), n + min(sz, 0))
    zd = slice(max(-sz, 0), n + min(-sz, 0))
    out[zd, xd] = a[zs, xs]
    return out


class LayerView:
    """A cached frame's layer, for the renderer."""

    def __init__(self, gpu):
        self.gpu = gpu
        self.tex = None
        self.on = False

    def load(self, entry):
        d = entry.get('sea_layer') if entry else None
        if not d:
            self.on = False
            return self
        a = np.asarray(d['out'], np.float16)
        n = a.shape[0]
        if self.tex is None or self.tex.size[:2] != (n, n):
            if self.tex is not None:
                self.tex.destroy()
            self.tex = self.gpu.texture2d(n, n, 'rgba16float', 'ocean-layer-cached')
        self.gpu.upload(self.tex, a)
        self.corner, self.size, self.n = tuple(d['corner']), float(d['size']), n
        self.hs, self.kp = float(d.get('hs', 0.0)), float(d.get('kp', 1.0))
        self.on = True
        return self

    def texture(self):
        return self.tex

    def uniforms(self, u):
        return layer_uniforms(u, self.corner, self.size, self.n, self.hs, self.kp)
