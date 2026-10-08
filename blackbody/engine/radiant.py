"""The heat things radiate, shared: once a frame, one list of radiant sources that everything warmed by radiation reads.

What radiates writes to it:
  - the fire's hot gas, an optically thin flame: 4 kappa sigma (T^4 - Ta^4) a cubic metre, from the gas's real
    temperature (flames 1100-1300 K: a campfire with 1.2 m flames radiates about 40 kW, a fifth to a third of the heat it
    releases, as real fires do), not the look's flame colour temperature, which is often far hotter;
  - glowing matter (molten iron, coal) and lava, from their surfaces open to the air: emissivity sigma (T^4 - Ta^4) a
    square metre, facing out of them;
  - hot objects (objheat.py), a source to each face of them.
The box (the gas's, or the liquid's) is cut into at most CELLS coarse cells; each cell with anything radiating in it is
one source at the power-weighted centre of what radiates there. Matter and lava add to the cells through a fixed-point
accumulator (RA) as they step; the gas is gathered as the list is built. Objects' faces come after the cells.

A source is three vec4 (rad_common.wgsl): (centre, softening m^2), (power of its volume W, of its surfaces W, owner,
_), (its surfaces' power-weighted outward normal, _). What reads it (cloth_rad.wgsl, mpm_heat.wgsl, obj_heat.wgsl) sums
the irradiance from each, by the inverse square of the distance: a volume's power spread every way, a surface's thrown
forward as a flat one throws it (cosine), toward a point facing it. Nothing is shadowed. The owner keeps a thing from
warming itself by its own heat (-1 the gas and lava, -2 matter, 0..15 the objects).
"""
from __future__ import annotations

import math

import numpy as np

from .gpu import Uniforms

KAPPA = 0.2            # 1/m: absorption coefficient of the flames (and so how strongly they radiate)
FLAME_K = 1250.0       # K: the flames' real temperature (wood and hydrocarbon flames average 1100-1300 K) at gas heat 1
MAX_K = 1450.0         # K: the hottest gas's real temperature
CELLS = 4096           # coarse cells, at most
OBJECT_SOURCES = 128   # objects' faces, at most (16 objects, 6 faces each)
SOURCES = CELLS + OBJECT_SOURCES
RA_WORDS = 7           # per coarse cell: power, power x (centre share x, y, z), power x (normal x, y, z)
FX_P = 100.0           # the accumulator's fixed point: a hundredth of a watt
OWNER_GAS = -1.0
OWNER_MATTER = -2.0


def gas_kelvin(t, ambient_k):
    """The gas's real temperature (K) for its heat t (0 the ambient air, 1 flame), as rad_build.wgsl."""
    t = np.asarray(t, float)
    return ambient_k + (FLAME_K - ambient_k) * np.minimum(t, 1.0) + (MAX_K - FLAME_K) * (1.0 - np.exp(-np.maximum(t - 1.0, 0.0)))


class Radiant:
    """The frame's radiant sources (RL, count in RLC), and the accumulator matter and lava add theirs to (RA)."""

    def __init__(self, gpu):
        self.gpu = gpu
        self.RL = gpu.buffer(SOURCES * 48, 'radiant-sources')
        self.RLC = gpu.buffer(16, 'radiant-count')
        self.RA = gpu.buffer(CELLS * RA_WORDS * 4, 'radiant-accumulator')
        self.OB = gpu.buffer(OBJECT_SOURCES * 48, 'radiant-objects')
        self.RT = gpu.buffer(CELLS * 48, 'radiant-cells')   # each coarse cell's source, before the list is packed
        self._k = None
        self.grid = None            # (corner (m), coarse cell size (m), coarse dims, gas cells to a coarse cell or 0)
        self.built = False
        self.n_objects = 0
        self._cleared = False

    def _kernels(self):
        if self._k is None:
            g = self.gpu
            res = ['utex3d', 'buf', 'buf', 'buf', 'buf']
            self._k = {
                'build': g.kernel('rad_build.wgsl', res, 'build', workgroup=(64, 1, 1)),
                'compact': g.kernel('rad_build.wgsl', res, 'compact', workgroup=(256, 1, 1)),
                'objects': g.kernel('rad_build.wgsl', res, 'objects', workgroup=(64, 1, 1)),
            }
        return self._k

    def layout(self, origin, dims, h, gas=True):
        """Cut the box (its corner, cells and cell size) into coarse cells: whole blocks of the gas's cells when the gas
        radiates (gas True), else a cut of the box into cubes. The accumulator starts over when the cut changes."""
        dims = tuple(int(d) for d in dims)
        bs = 1
        while int(np.prod([math.ceil(d / bs) for d in dims])) > CELLS:
            bs += 1
        cd = tuple(int(math.ceil(d / bs)) for d in dims)
        grid = (tuple(float(x) for x in origin), float(bs * h), cd, bs if gas else 0, float(h))
        if grid != self.grid:
            self.grid = grid
            self._cleared = False
        return grid

    def uniform_block(self, u):
        """The coarse grid for a shader that adds to the accumulator (rad_common.wgsl RadGrid): (corner, coarse cell
        size), (coarse dims, on)."""
        if self.grid is None:
            return u.v4(0.0, 0.0, 0.0, 1.0).v4(1.0, 1.0, 1.0, 0.0)
        o, cs, cd, _, _ = self.grid
        return u.v4(*o, cs).v4(*cd, 1.0)

    def set_objects(self, sources):
        """The objects' faces this frame: a list of (centre (3), softening m^2, power W, normal (3), owner)."""
        n = min(len(sources), OBJECT_SOURCES)
        a = np.zeros((max(n, 1), 12), np.float32)
        for i, (c, soft, p, nrm, owner) in enumerate(sources[:n]):
            a[i, 0:3] = c
            a[i, 3] = soft
            a[i, 5] = p
            a[i, 6] = owner
            a[i, 8:11] = np.asarray(nrm, float) * p
        if n:
            self.gpu.write_buffer(self.OB, a)
        self.n_objects = n

    def build(self, b, scal=None, gas_dims=None, ambient_k=293.0):
        """Record the frame's list, in batch b: the coarse cells (the gas as `scal` holds it now, its grid `gas_dims`,
        when the layout is the gas's, plus what matter and lava added since the last build), then the objects' faces. The accumulator is
        cleared for the next frame's additions."""
        if self.grid is None:
            return False
        k = self._kernels()
        o, cs, cd, bs, h = self.grid
        cells = int(np.prod(cd))
        if not self._cleared:
            b.clear_buffer(self.RA)
            self._cleared = True
        b.clear_buffer(self.RLC)
        gas = scal is not None and bs > 0 and gas_dims is not None
        u = (Uniforms().v4(*o, cs).v4(*cd, cells).v4(ambient_k, FLAME_K, MAX_K, bs if gas else 0)
             .v4(KAPPA, h, self.n_objects, 1.0 / FX_P))
        u.v4(*(tuple(gas_dims) if gas else (1, 1, 1)), 0.0)
        tex = scal if gas else self._dummy()
        b.run(k['build'], [tex, self.RA, self.RL, self.RLC, self.RT], u, groups=(-(-cells // 64), 1, 1))
        # (packed in the order of the cells, then the objects in theirs: the same list every run)
        b.run(k['compact'], [tex, self.RA, self.RL, self.RLC, self.RT], u, groups=(1, 1, 1))
        b.clear_buffer(self.RA)
        if self.n_objects:
            b.run(k['objects'], [tex, self.OB, self.RL, self.RLC, self.RT], u, groups=(-(-self.n_objects // 64), 1, 1))
        self.built = True
        return True

    def _dummy(self):
        t = getattr(self, '_tex0', None)
        if t is None:
            t = self._tex0 = self.gpu.texture3d((1, 1, 1), 'rgba16float', 'radiant-no-gas')
            self.gpu.upload(t, np.zeros((1, 1, 1, 4), np.float16))
        return t
