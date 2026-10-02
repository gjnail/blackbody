"""The pieces of broken objects in the simulation (bodies_sdf.wgsl): however many there are, the gas (and the liquid)
see them as moving solids. Each substep the pieces' distance is folded into the solver's collider distance field and
their velocity written into the cells inside them, which the boundary pass gives the faces round them, so falling
rubble shoves the smoke aside and the smoke flows round it where it lies.

The 16 colliders stay as they are (colliders.wgsl); this is only for the pieces, which can number in the hundreds.
A frame's substeps are uploaded together before the frame is recorded (one batch runs them all).
"""
from __future__ import annotations

import numpy as np

from .gpu import Uniforms

TILE = 8   # cells per side of a tile of the grid, each listing the pieces near it


def piece_records(scene, pieces, rows=None, owners=None):
    """The pieces as the shaders take them: (P (n, 5, 4) float32, PL (planes, 4) float32, centres (n, 3),
    radii (n,)). pieces: {collider index: Solids.piece_poses entry}; rows: collider index -> material row (the
    stage's; 0 when None); owners: a list that gets (collider index, piece index) for each, in order.

    P per piece: centre (fire-local m), planes (count); orientation (x y z w); velocity (m/s), first plane; spin
    (rad/s, world), material row; its centre in the object's frame before it broke, bounding radius. PL: each plane's
    normal and offset in the piece's frame (n . x <= d); a cut face's normal is stored twice as long."""
    from .solids import fractured
    P, PL, centres, radii = [], [], [], []
    n_planes = 0
    for ci, pose in (pieces or {}).items():
        if ci >= len(scene.colliders) or (rows is not None and ci not in rows):
            continue
        frac = fractured(scene.colliders[ci], pose['size'], float(pose.get('hollow', 0.0)))
        for k in range(min(len(frac.pieces), len(pose['pos']))):
            if owners is not None:
                owners.append((ci, k))
            pc = frac.pieces[k]
            pl = pc.planes.copy()
            pl[:, 3] -= pl[:, :3] @ pc.centroid
            pl[pc.inner, :3] *= 2.0
            rad = float(np.linalg.norm(pc.verts - pc.centroid, axis=1).max())
            pos = np.asarray(pose['pos'][k], float)
            P.append([[*pos, len(pl)], [*np.asarray(pose['quat'][k], float)], [*np.asarray(pose['vel'][k], float), n_planes],
                      [*np.asarray(pose['omega'][k], float), 0 if rows is None else rows[ci]], [*pc.centroid, rad]])
            PL.append(pl)
            n_planes += len(pl)
            centres.append(pos)
            radii.append(rad)
    if not P:
        return None
    return (np.asarray(P, np.float32), np.concatenate(PL).astype(np.float32), np.asarray(centres, float),
            np.asarray(radii, float))


class BodyField:
    def __init__(self, gpu):
        self.gpu = gpu
        self.k = gpu.kernel('bodies_sdf.wgsl', ['rbuf', 'rbuf', 'rbuf', 'rbuf', 'st3d:r32float:rw', 'st3d:rgba16float:w'])
        self._bufs = {}
        self.steps = []          # per substep: (first piece, first tile, pieces)
        self.owners = []         # per substep: (collider index, piece index) of each of its pieces
        self.tiles = (1, 1, 1)

    def _buffer(self, name, data):
        data = np.ascontiguousarray(data)
        size = max(int(data.nbytes), 16)
        b = self._bufs.get(name)
        if b is None or b.size < size:
            if b is not None:
                b.destroy()
            b = self._bufs[name] = self.gpu.buffer(max(size, 1024), f'bodies-{name}')
        if data.nbytes:
            self.gpu.write_buffer(b, data)
        return b

    def prepare(self, scene, substeps, dims, h, origin, region=None):
        """Upload a frame's substeps (a list of {collider: Solids.piece_poses entry}, one per substep) for a grid
        (dims, cell h, corner origin); region: (lo, hi) fire-local m, only the pieces that reach into it. False if
        there are no pieces."""
        self.steps = []
        self.owners = []
        dims = np.asarray(dims, int)
        nt = np.maximum((dims + TILE - 1) // TILE, 1)
        self.tiles = tuple(int(x) for x in nt)
        P_all, TC_all, TL_all = [], [], []
        PL = None
        n_p = n_t = n_l = 0
        o = np.asarray(origin, float)
        for poses in substeps:
            owners = []
            rec = piece_records(scene, poses, owners=owners)
            self.owners.append(owners)
            if rec is None:
                self.steps.append(None)
                continue
            P, planes, c, r = rec
            PL = planes if PL is None else PL
            lists = [[] for _ in range(int(np.prod(nt)))]
            reach = r + 1.5 * h
            a = np.clip(np.floor((c - reach[:, None] - o) / (h * TILE)).astype(int), 0, nt - 1)
            b = np.clip(np.floor((c + reach[:, None] - o) / (h * TILE)).astype(int), 0, nt - 1)
            inside = np.all((c + reach[:, None] >= o) & (c - reach[:, None] <= o + dims * h), axis=1)
            if region is not None:
                inside &= np.all((c + reach[:, None] >= region[0]) & (c - reach[:, None] <= region[1]), axis=1)
            for k in np.nonzero(inside)[0]:
                for z in range(a[k, 2], b[k, 2] + 1):
                    for y in range(a[k, 1], b[k, 1] + 1):
                        for x in range(a[k, 0], b[k, 0] + 1):
                            lists[x + nt[0] * (y + nt[1] * z)].append(k)
            counts = np.array([len(L) for L in lists], np.uint32)
            starts = (np.concatenate([[0], np.cumsum(counts)[:-1]]) + n_l).astype(np.uint32)
            flat = np.array([k for L in lists for k in L], np.uint32)
            self.steps.append((n_p, n_t, len(P)))
            P_all.append(P)
            TC_all.append(np.stack([starts, counts], 1))
            TL_all.append(flat)
            n_p += len(P)
            n_t += len(lists)
            n_l += len(flat)
        if not P_all:
            return False
        self.TC_counts = [int(len(x)) for x in TL_all]
        self.P = self._buffer('pieces', np.concatenate(P_all))
        self.PL = self._buffer('planes', PL)
        self.TC = self._buffer('tiles', np.concatenate(TC_all))
        self.TL = self._buffer('list', np.concatenate(TL_all) if n_l else np.zeros(1, np.uint32))
        return True

    def bake(self, b, solver, i):
        """Record substep i's pieces into the solver's distance field and solid velocity (batch b). False if that
        substep has none."""
        st = self.steps[i] if i < len(self.steps) else None
        if st is None:
            return False
        u = solver._grid(0.0).v4(TILE, *self.tiles).v4(*st, 0)
        sdf = solver.SDF if hasattr(solver, 'SDF') else solver.sdf
        b.run(self.k, [self.P, self.PL, self.TC, self.TL, sdf, solver.solid_vel()], u, solver.dims)
        return True
