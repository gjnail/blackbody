"""Breaking things: an object cut beforehand into convex pieces, and the bonds that glue them (engine/solids.py
holds the pieces together with welds that break when a bond is overloaded).

A piece is a convex polyhedron in the object's own frame (metres): the planes that bound it (n . x <= d, n a unit
normal), its corners, its centroid and its volume. Each plane is either one of the object's own faces or a cut
(an inner face, drawn in the material's inside colour). Two pieces that share a cut are bonded across it: the
shared face's area, its normal (from the first piece toward the second) and its centre.

Patterns (`impact`: where it is hit, its own frame: solids.py finds it by a run of the scene with it whole):
- bends: metal and plastic, cut where they crease: bars into segments, sheets into cells through them (their joints
  yield in solids.py);
- voronoi: cells round seeds spread through the object, crowded round the impact: small chips where it is hit,
  larger chunks away from it, as a stone or a block of concrete breaks;
- bricks: a box as bricks in running bond (a wall); each brick is a piece and the mortar joints are the bonds;
- shards: glass: a spider's web round the impact (web): cracks radiating from it to the frame, crossed by rings that
  grow apart outward, so a pane breaks into small splinters where it is hit and long daggers toward its edges;
- splinters: wood: cells stretched along the grain (the object's long axis).

A hollow object (a vase, a tank, a crate) breaks as a shell: a cylinder into curved shards of its wall, irregular and
crowded round the impact as pottery breaks (cells on its unrolled wall), and pieces of its base; a box wall by wall,
each wall into chunks or splinters. The shell's inside is its own surface, not a cut.

Spheres and cylinders are cut as the polyhedra that hold them (48 and 20 faces): a few percent more volume.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

BRICK = (0.215, 0.065, 0.1025)   # m: a standard brick (length, height, depth)
JOINT = 0.01                     # m: a mortar joint
TOL = 1e-7


@dataclass
class Piece:
    planes: np.ndarray       # (k, 4): unit normal, offset (n . x <= d), the object's frame
    inner: np.ndarray        # (k,) bool: a cut face
    verts: np.ndarray        # (m, 3) corners
    centroid: np.ndarray     # (3,)
    volume: float
    face_area: np.ndarray = None     # (k,) the area of its face on each plane (m^2)
    face_centre: np.ndarray = None   # (k, 3) and that face's centre
    face_poly: list = None           # (k) its corners on each plane, in order round the face


@dataclass
class Bond:
    i: int
    j: int
    area: float              # m^2
    normal: np.ndarray       # from piece i toward piece j (object frame)
    centre: np.ndarray       # the shared face's centre (object frame)


@dataclass
class Fracture:
    pieces: list = field(default_factory=list)
    bonds: list = field(default_factory=list)

    @property
    def volume(self):
        return float(sum(p.volume for p in self.pieces))


# ---- shapes as planes ----------------------------------------------------------------------------------------

def _fibonacci(n):
    k = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / n)
    theta = math.pi * (1.0 + 5.0 ** 0.5) * k
    return np.stack([np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)], -1)


def shape_planes(shape, size):
    """The planes bounding a shape (its own frame): a box's six faces; a cylinder (axis y) as a 20-sided prism; a
    sphere as 48 faces round it."""
    s = np.abs(np.asarray(size, float))
    if shape == 'sphere':
        n = _fibonacci(48)
        return np.concatenate([n, np.full((48, 1), s[0])], 1)
    if shape == 'cylinder':
        a = np.arange(20) * (2.0 * math.pi / 20)
        side = np.stack([np.cos(a), np.zeros(20), np.sin(a), np.full(20, s[0])], 1)
        caps = np.array([[0.0, 1.0, 0.0, s[1]], [0.0, -1.0, 0.0, s[1]]])
        return np.concatenate([side, caps])
    return np.array([[1.0, 0, 0, s[0]], [-1.0, 0, 0, s[0]], [0, 1.0, 0, s[1]], [0, -1.0, 0, s[1]],
                     [0, 0, 1.0, s[2]], [0, 0, -1.0, s[2]]])


def _inside(planes, x, eps=1e-9):
    return np.all(x @ planes[:, :3].T <= planes[:, 3] + eps, axis=-1)


# ---- a convex polyhedron from planes ---------------------------------------------------------------------------

def polyhedron(planes):
    """Corners of the convex polyhedron n . x <= d (planes (k, 4)), or an empty array: every point where three
    planes meet that is inside all the others."""
    k = len(planes)
    if k < 4:
        return np.zeros((0, 3))
    tri = np.array(list(combinations(range(k), 3)), np.int64)
    A = planes[tri, :3]
    b = planes[tri, 3]
    det = np.linalg.det(A)
    ok = np.abs(det) > 1e-12
    if not ok.any():
        return np.zeros((0, 3))
    x = np.linalg.solve(A[ok], b[ok][..., None])[..., 0]
    scale = max(1.0, float(np.abs(planes[:, 3]).max()))
    x = x[_inside(planes, x, 1e-7 * scale)]
    if len(x) == 0:
        return x
    # the same corner from several triples: keep one
    q = np.round(x / (1e-7 * scale)).astype(np.int64)
    _, first = np.unique(q, axis=0, return_index=True)
    return x[np.sort(first)]


def _face(verts, plane, scale):
    """The corners on a plane, in order round it: (polygon (m, 3) or None, area, centre)."""
    n, d = plane[:3], plane[3]
    on = np.abs(verts @ n - d) <= 1e-6 * scale
    if on.sum() < 3:
        return None, 0.0, None
    p = verts[on]
    c = p.mean(0)
    u = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    ang = np.arctan2((p - c) @ v, (p - c) @ u)
    p = p[np.argsort(ang)]
    # area and area centroid (a fan round the mean)
    a = np.cross(p - c, np.roll(p, -1, 0) - c) @ n * 0.5
    area = float(a.sum())
    if area <= 1e-14:
        return None, 0.0, None
    cen = ((c + p + np.roll(p, -1, 0)) / 3.0 * a[:, None]).sum(0) / area
    return p, area, cen


def make_piece(planes, inner):
    """A Piece from its planes (dropping the ones that do not touch it), or None if it is empty."""
    v = polyhedron(planes)
    if len(v) < 4:
        return None, []
    scale = max(1.0, float(np.abs(v).max()))
    keep, faces = [], []
    for k, pl in enumerate(planes):
        poly, area, cen = _face(v, pl, scale)
        if poly is not None:
            keep.append(k)
            faces.append((k, poly, area, cen))
    if len(keep) < 4:
        return None, []
    # volume and centroid: pyramids from a point inside to every face
    a = v.mean(0)
    vol = 0.0
    cen = np.zeros(3)
    for k, poly, area, fc in faces:
        h = planes[k, 3] - planes[k, :3] @ a
        pv = area * h / 3.0
        vol += pv
        cen += pv * (a + 0.75 * (fc - a))
    if vol <= 1e-15:
        return None, []
    piece = Piece(planes=planes[keep].copy(), inner=np.asarray(inner, bool)[keep].copy(), verts=v, centroid=cen / vol,
                  volume=float(vol), face_area=np.array([f[2] for f in faces]), face_centre=np.array([f[3] for f in faces]),
                  face_poly=[f[1] for f in faces])
    return piece, [(keep.index(k), area, fc) for k, _poly, area, fc in faces]


# ---- seeds ------------------------------------------------------------------------------------------------------

def _seeds(planes, n, rng, impact=None, crowd=0.0, lo=None, hi=None):
    """n points inside the shape, kept apart (dart throwing); with an impact point and crowd > 0, denser round it."""
    lo = planes_bounds(planes)[0] if lo is None else lo
    hi = planes_bounds(planes)[1] if hi is None else hi
    vol = float(np.prod(hi - lo))
    r_min = 0.55 * (vol / max(n, 1)) ** (1.0 / 3.0)
    out = []
    tries = 0
    while len(out) < n and tries < 200 * n:
        tries += 1
        if impact is not None and crowd > 0.0 and rng.random() < crowd:
            # crowded round the impact: distance grows as u^2 (dense near it)
            d = rng.normal(size=3)
            d /= np.linalg.norm(d) + 1e-12
            p = np.asarray(impact, float) + d * float(np.linalg.norm(hi - lo)) * 0.5 * rng.random() ** 2
        else:
            p = lo + rng.random(3) * (hi - lo)
        if not _inside(planes, p[None])[0]:
            continue
        if impact is not None and crowd > 0.0:
            rr = r_min * min(1.0, 0.25 + float(np.linalg.norm(p - impact)) / (0.5 * float(np.linalg.norm(hi - lo))))
        else:
            rr = r_min
        if out and np.min(np.linalg.norm(np.asarray(out) - p, axis=1)) < rr:
            if tries < 100 * n:
                continue
        out.append(p)
    return np.asarray(out)


def planes_bounds(planes):
    v = polyhedron(planes)
    return v.min(0), v.max(0)


# ---- patterns ----------------------------------------------------------------------------------------------------

def voronoi(planes, seeds, stretch=None, neighbours=26):
    """Voronoi cells of the seeds inside the shape (its planes), each clipped by the shape. stretch: scale per
    axis applied before measuring distances (a grain: cells longer along the axes it shrinks)."""
    S = np.ones(3) if stretch is None else np.asarray(stretch, float)
    sp = seeds * S        # seeds in the stretched space
    n = len(seeds)
    frac = Fracture()
    face_of = {}          # (i, j) -> (area, centre) of the cut between cells i and j, seen from i
    pieces = []
    for i in range(n):
        order = np.argsort(np.linalg.norm(sp - sp[i], axis=1))[1:]
        k = min(neighbours, n - 1)
        while True:
            near = order[:k]
            # the bisector of seeds i and j in the stretched space, back in the object's frame
            nn = sp[near] - sp[i]
            d_ = (nn * (sp[near] + sp[i]) * 0.5).sum(1)
            nrm = nn * S                            # (n' . S x <= d')
            ln = np.linalg.norm(nrm, axis=1, keepdims=True)
            cut = np.concatenate([nrm / ln, (d_ / ln[:, 0])[:, None]], 1)
            pl = np.concatenate([planes, cut])
            inner = np.concatenate([np.zeros(len(planes), bool), np.ones(len(cut), bool)])
            piece, faces = make_piece(pl, inner)
            if piece is None or k >= n - 1:
                break
            # every corner must be no nearer another seed than its own: else widen the neighbours
            vs = piece.verts * S
            dist_own = np.linalg.norm(vs - sp[i], axis=1)
            others = np.linalg.norm(vs[:, None, :] - sp[None, :, :], axis=2)
            if np.all(others.min(1) >= dist_own - 1e-9 * max(1.0, float(np.abs(vs).max()))):
                break
            k = min(n - 1, k * 2)
        pieces.append(piece)
        if piece is None:
            continue
        # the cuts this cell keeps, by the neighbour they face
        cut_rows = {tuple(np.round(p, 9)) for p in piece.planes[piece.inner]}
        for jj, j in enumerate(near):
            row = tuple(np.round(cut[jj], 9))
            if row in cut_rows:
                for kk, area, fc in faces:
                    if np.allclose(piece.planes[kk], cut[jj], atol=1e-9) and area > 0.0:
                        face_of[(i, int(j))] = (area, fc, cut[jj, :3].copy())
                        break
    index = {}
    for i, p in enumerate(pieces):
        if p is not None:
            index[i] = len(frac.pieces)
            frac.pieces.append(p)
    big = max((a for a, _c, _n in face_of.values()), default=0.0)
    for (i, j), (area, fc, nrm) in face_of.items():
        if i < j and i in index and j in index:
            a2 = face_of.get((j, i), (area,))[0]
            if min(area, a2) > 1e-3 * big:     # (a corner barely touching holds nothing)
                frac.bonds.append(Bond(index[i], index[j], float(min(area, a2)), nrm, fc))
    return frac


def inset(piece, gap):
    """The piece's corners with its cut faces moved in by `gap` (its own frame): welded neighbours then do not
    touch, so they do not collide while they are glued."""
    pl = piece.planes.copy()
    pl[piece.inner, 3] -= gap
    v = polyhedron(pl)
    return v if len(v) >= 4 else piece.verts


def bricks(size, brick=BRICK, joint=JOINT):
    """A box (half extents) as bricks in running bond: courses along y, bricks along its longer horizontal axis, a
    brick as deep as the wall (several where the wall is thicker than two bricks). Bonds are the mortar joints."""
    s = np.abs(np.asarray(size, float))
    along = 0 if s[0] >= s[2] else 2                 # bricks run along the wall
    deep = 2 - along
    L, H, D = 2 * s[along], 2 * s[1], 2 * s[deep]
    bl, bh = brick[0] + joint, brick[1] + joint
    ny = max(1, int(round(H / bh)))
    nx = max(1, int(round(L / bl)))
    nd = max(1, int(round(D / (brick[2] + joint)))) if D > 1.5 * (brick[2] + joint) else 1
    ch, cl, cd = H / ny, L / nx, D / nd               # course height, brick length, depth (joints included)
    frac = Fracture()
    rows = []
    for j in range(ny):
        off = 0.5 * cl if j % 2 else 0.0
        cuts = [-L / 2] + [-L / 2 + off + k * cl for k in range(0 if off else 1, nx + 1) if -L / 2 < -L / 2 + off + k * cl < L / 2] + [L / 2]
        cuts = sorted(set(round(c, 9) for c in cuts))
        row = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a < 1e-6:
                continue
            for kd in range(nd):
                lo = np.zeros(3)
                hi = np.zeros(3)
                lo[along], hi[along] = a, b
                lo[1], hi[1] = -H / 2 + j * ch, -H / 2 + (j + 1) * ch
                lo[deep], hi[deep] = -D / 2 + kd * cd, -D / 2 + (kd + 1) * cd
                pl = np.array([[1.0, 0, 0, hi[0]], [-1.0, 0, 0, -lo[0]], [0, 1.0, 0, hi[1]], [0, -1.0, 0, -lo[1]],
                               [0, 0, 1.0, hi[2]], [0, 0, -1.0, -lo[2]]])
                inner = np.array([abs(hi[0]) < s[0] - 1e-9, abs(lo[0]) < s[0] - 1e-9, abs(hi[1]) < s[1] - 1e-9,
                                  abs(lo[1]) < s[1] - 1e-9, abs(hi[2]) < s[2] - 1e-9, abs(lo[2]) < s[2] - 1e-9])
                p, _ = make_piece(pl, inner)
                if p is not None:
                    row.append((len(frac.pieces), lo, hi))
                    frac.pieces.append(p)
        rows.append(row)
    # joints: touching faces (head joints within a course, bed joints between courses, between brick layers)
    boxes = [(i, lo, hi) for row in rows for i, lo, hi in row]
    for (i, lo1, hi1), (j, lo2, hi2) in combinations(boxes, 2):
        for ax in range(3):
            o = [a for a in range(3) if a != ax]
            touch = abs(hi1[ax] - lo2[ax]) < 1e-9 or abs(hi2[ax] - lo1[ax]) < 1e-9
            if not touch:
                continue
            ov = [min(hi1[a], hi2[a]) - max(lo1[a], lo2[a]) for a in o]
            if min(ov) <= 1e-6:
                continue
            nrm = np.zeros(3)
            nrm[ax] = 1.0 if abs(hi1[ax] - lo2[ax]) < 1e-9 else -1.0
            cen = np.zeros(3)
            cen[ax] = hi1[ax] if nrm[ax] > 0 else lo1[ax]
            for a in o:
                cen[a] = 0.5 * (max(lo1[a], lo2[a]) + min(hi1[a], hi2[a]))
            frac.bonds.append(Bond(i, j, float(ov[0] * ov[1]), nrm, cen))
    return frac


def fracture(shape, size, pieces=24, pattern='voronoi', seed=0, impact=None, hollow=0.0, rim=None, mesh=None):
    """Cut an object (shape, size as a collider has them) into about `pieces` convex pieces (a shell if it is
    hollow: wall thickness `hollow`; a hollow cylinder open at the top down to height `rim`, closed if None; a mesh's
    own vertices and triangles in its frame, `mesh`)."""
    if shape == 'mesh':
        return mesh_pieces(mesh[0], mesh[1], max(2, int(pieces)), np.random.default_rng(int(seed) + 7919), impact)
    if hollow > 0.0 and shape in ('cylinder', 'box') and pattern != 'bricks':
        return shell(shape, size, hollow, pieces, seed, pattern if pattern in ('splinters', 'bends') else 'voronoi',
                     impact=impact, rim=rim)
    rng = np.random.default_rng(int(seed) + 7919)
    if pattern == 'bricks' and shape == 'box':
        return bricks(size)
    planes = shape_planes(shape, size)
    lo, hi = planes_bounds(planes)
    n = max(2, int(pieces))
    s = np.abs(np.asarray(size, float))
    if pattern == 'shards':
        if shape == 'box':
            return web(s, n, rng, impact)
        # (a glass ball or rod: crowded round the impact, slivers radiating from it)
        thin = int(np.argmin(hi - lo))
        p0 = np.zeros(3) if impact is None else np.asarray(impact, float)
        seeds = _seeds(planes, n, rng, impact=p0, crowd=0.8, lo=lo, hi=hi)
        seeds[:, thin] = 0.0        # through the whole thickness
        return voronoi(planes, _spread(seeds, thin, rng, hi - lo))
    if pattern == 'bends':
        return bends(planes, n, rng, lo, hi)
    if pattern == 'splinters':
        long_ax = int(np.argmax(hi - lo))
        stretch = np.ones(3)
        stretch[long_ax] = 0.3      # distances along the grain count a third: long cells
        seeds = _seeds(planes, n, rng, lo=lo, hi=hi)
        return voronoi(planes, seeds, stretch=stretch)
    seeds = _seeds(planes, n, rng, impact=impact, crowd=0.6 if impact is not None else 0.0, lo=lo, hi=hi)
    del s
    return voronoi(planes, seeds)


# the directions a broken mesh's pieces are bounded along (besides its cuts and its surface's own facing): the faces,
# edges and corners of a cube
_DOP = np.array([d for d in ((x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)) if any(d)], float)
_DOP /= np.linalg.norm(_DOP, axis=1, keepdims=True)
_AROUND = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)]


def mesh_pieces(v, t, n, rng, impact=None, near=16):
    """A closed mesh (vertices v, triangles t, its own frame) cut into about n pieces: Voronoi cells of seeds spread
    through its inside (crowded round the impact), each piece the cell's cuts (shared exactly with its neighbours, so
    they glue) and, round the part of the mesh in it, planes along 26 fixed directions and the few its surface there
    faces most (a convex piece: a hollow or a dent in it is filled in)."""
    from .solids import mesh_occupancy
    v = np.asarray(v, float)
    t = np.asarray(t, np.int64)
    lo, hi = v.min(0), v.max(0)
    ext = np.maximum(hi - lo, 1e-6)
    cell = float(ext.max()) / 48.0
    occ = mesh_occupancy(v, t, lo, hi, cell)
    if occ is None or not occ.any():          # (open: its convex hull)
        ii = np.stack(np.meshgrid(*[np.arange(int(np.ceil(e / cell))) for e in ext], indexing='ij'), -1).reshape(-1, 3)
    else:
        ii = np.argwhere(occ)
    inside = lo + (ii + 0.5) * cell
    # the surface: its vertices and points spread over its triangles by area, each with its triangle's facing
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    nrm = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    nrm = nrm / np.maximum(2.0 * area[:, None], 1e-18)
    m = 40000
    pick = rng.choice(len(t), m, p=area / max(area.sum(), 1e-18))
    r1, r2 = rng.random(m), rng.random(m)
    s1 = np.sqrt(r1)
    surf = a[pick] * (1 - s1)[:, None] + b[pick] * (s1 * (1 - r2))[:, None] + c[pick] * (s1 * r2)[:, None]
    surf_n = nrm[pick]
    surf_a = np.full(m, area.sum() / m)
    # seeds: inside points, kept apart, crowded round a hit
    k = min(n, len(inside))
    r_min = 0.6 * (len(inside) * cell ** 3 / max(k, 1)) ** (1.0 / 3.0)
    reach = 0.5 * float(np.linalg.norm(ext))
    order = rng.permutation(len(inside))
    if impact is not None:
        dd = np.linalg.norm(inside - np.asarray(impact, float), axis=1)
        order = order[np.argsort(dd[order] * rng.uniform(0.3, 1.0, len(order)), kind='stable')]
    seeds = []
    for i in order:
        p = inside[i]
        rr = r_min if impact is None else r_min * min(1.0, 0.3 + float(np.linalg.norm(p - impact)) / reach)
        if not seeds or np.min(np.linalg.norm(np.asarray(seeds) - p, axis=1)) >= rr:
            seeds.append(p)
        if len(seeds) >= k:
            break
    seeds = np.asarray(seeds, float)
    if len(seeds) < 2:
        seeds = np.array([lo + 0.35 * ext, lo + 0.65 * ext])
    # each point's cell: its nearest seed
    def owner(pts):
        out = np.empty(len(pts), np.int64)
        for s in range(0, len(pts), 4096):
            out[s:s + 4096] = np.argmin(((pts[s:s + 4096, None, :] - seeds[None]) ** 2).sum(-1), 1)
        return out
    own_s, own_i = owner(surf), owner(inside)
    own_v = owner(v)
    frac = Fracture()
    for i in range(len(seeds)):
        o = np.argsort(np.linalg.norm(seeds - seeds[i], axis=1))[1:near + 1]
        nn = seeds[o] - seeds[i]
        d_ = (nn * (seeds[o] + seeds[i]) * 0.5).sum(1)
        ln = np.linalg.norm(nn, axis=1)
        cuts = np.concatenate([nn / ln[:, None], (d_ / ln)[:, None]], 1)
        # the mesh in this cell, in its separate parts (two logs, a chair's legs): a piece each, not one bridging them
        vox = np.nonzero(own_i == i)[0]
        if not len(vox):
            continue
        part = _parts(ii[vox])
        si_ = np.nonzero(own_s == i)[0]
        vi_ = np.nonzero(own_v == i)[0]
        near_s = np.argmin(((surf[si_][:, None, :] - inside[vox][None]) ** 2).sum(-1), 1) if len(si_) else np.zeros(0, np.int64)
        near_v = np.argmin(((v[vi_][:, None, :] - inside[vox][None]) ** 2).sum(-1), 1) if len(vi_) else np.zeros(0, np.int64)
        groups = [dict(vox=inside[vox[part == g]], s=si_[part[near_s] == g], v=vi_[part[near_v] == g], cuts=list(cuts))
                  for g in range(int(part.max()) + 1)]
        groups = _separate(groups, surf, v, cell)
        for gr in groups:
            frac.pieces.extend(_mesh_split(gr['cuts'], gr['vox'], surf[gr['s']], surf_n[gr['s']], surf_a[gr['s']], v[gr['v']], cell))
    frac.bonds = touching_bonds(frac.pieces)
    return frac


def _mesh_split(cuts, vox, sp, sn, sa, vp, cell, depth=1):
    """The pieces of one part of a broken mesh (its inside cells vox, surface points sp with their facing sn and area
    sa, its vertices vp) within cuts: one convex piece, or, where that would fill in much more than the part (an L of a
    chair's seat and arm), the part split in two across the line between its two halves (a plane both halves share,
    so they glue)."""
    piece = _mesh_piece(np.asarray(cuts, float), np.concatenate([sp, vp, vox]), sn, sa, cell)
    if piece is None:
        return []
    if depth <= 0 or len(vox) < 16 or piece.volume <= 1.5 * len(vox) * cell ** 3:
        return [piece]
    # two halves: from the two inside cells farthest apart, a few rounds of 2-means
    c0 = vox[np.argmax(np.linalg.norm(vox - vox.mean(0), axis=1))]
    c1 = vox[np.argmax(np.linalg.norm(vox - c0, axis=1))]
    for _ in range(6):
        side = np.linalg.norm(vox - c0, axis=1) > np.linalg.norm(vox - c1, axis=1)
        if side.all() or not side.any():
            return [piece]
        c0, c1 = vox[~side].mean(0), vox[side].mean(0)
    nrm = c1 - c0
    ln = float(np.linalg.norm(nrm))
    if ln < 1e-9:
        return [piece]
    nrm /= ln
    d = float(nrm @ (0.5 * (c0 + c1)))
    out = []
    for sign in (1.0, -1.0):     # (the half on the c0 side: n.x <= d; the other: -n.x <= -d)
        keep = lambda x: sign * (x @ nrm) <= sign * d
        out += _mesh_split(cuts + [np.append(sign * nrm, sign * d)], vox[keep(vox)], sp[keep(sp)], sn[keep(sp)], sa[keep(sp)],
                           vp[keep(vp)], cell, depth - 1)
    return out


def _separate(groups, surf, v, cell):
    """Separate parts of a mesh in one Voronoi cell, kept from overlapping as convex pieces: two that interleave (no
    direction parts them by more than a couple of cells' overlap) merged into one; between each two others, a plane
    both share, across the direction (of the cube's 26 and the line between them) they lie furthest apart along."""
    pts = lambda g: np.concatenate([g['vox'], surf[g['s']], v[g['v']]])

    def apart(A, B):          # the direction A and B lie furthest apart along, how far, and the plane between them
        line = B.mean(0) - A.mean(0)
        ln = float(np.linalg.norm(line))
        dirs = _DOP if ln < 1e-9 else np.concatenate([_DOP, (line / ln)[None]])
        pa, pb = A @ dirs.T, B @ dirs.T
        gap = pb.min(0) - pa.max(0)
        k = int(np.argmax(gap))
        return dirs[k], float(gap[k]), 0.5 * float(pa[:, k].max() + pb[:, k].min())

    merging = True
    while merging and len(groups) > 1:
        merging = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if apart(pts(groups[a]), pts(groups[b]))[1] < -2.0 * cell:
                    ga, gb = groups[a], groups.pop(b)
                    for k in ('vox', 's', 'v'):
                        ga[k] = np.concatenate([ga[k], gb[k]])
                    merging = True
                    break
            if merging:
                break
    for a in range(len(groups)):
        for b in range(a + 1, len(groups)):
            n, _gap, d = apart(pts(groups[a]), pts(groups[b]))
            groups[a]['cuts'].append(np.append(n, d))
            groups[b]['cuts'].append(np.append(-n, -d))
    return groups


def _parts(ijk):
    """The connected parts (touching at a face, an edge or a corner) of a set of grid cells: each cell's part number."""
    at = {tuple(c): n for n, c in enumerate(ijk.tolist())}
    part = np.full(len(ijk), -1, np.int64)
    g = 0
    for n in range(len(ijk)):
        if part[n] >= 0:
            continue
        part[n] = g
        todo = [n]
        while todo:
            x, y, z = ijk[todo.pop()]
            for dx, dy, dz in _AROUND:
                m = at.get((x + dx, y + dy, z + dz))
                if m is not None and part[m] < 0:
                    part[m] = g
                    todo.append(m)
        g += 1
    return part


def _mesh_piece(cuts, pts, sn, sa, cell):
    """One piece of a broken mesh: its cell's cuts, and round its points planes along the cube's 26 directions and the
    few its surface faces most."""
    if len(pts) < 4:
        return None
    # the directions it is bounded along: the cube's, and the few its surface here faces most
    dirs = [_DOP]
    for _ in range(4):
        if not len(sn):
            break
        mean = (sn * sa[:, None]).sum(0)
        if np.linalg.norm(mean) < 1e-9:
            break
        mean /= np.linalg.norm(mean)
        dirs.append(mean[None])
        keep = sn @ mean < 0.8
        sn, sa = sn[keep], sa[keep]
    dirs = np.concatenate(dirs)
    h = (pts @ dirs.T).max(0) + 0.05 * cell
    bound = np.concatenate([dirs, h[:, None]], 1)
    pl = np.concatenate([cuts, bound])
    inner = np.concatenate([np.ones(len(cuts), bool), np.zeros(len(bound), bool)])
    piece, _ = make_piece(pl, inner)
    return piece


def bends(planes, n, rng, lo, hi):
    """Metal or plastic, cut where it creases as it bends: a bar or a post (one side 3x the others) into segments across
    its length, a little uneven; a sheet (its thinnest side a fifth of the next) into cells right through it; anything
    else into chunks. Its joints yield (solids.py), so it bends there and stays bent."""
    ext = hi - lo
    order = np.argsort(ext)
    if ext[order[2]] >= 3.0 * ext[order[1]]:
        ax = int(order[2])
        k = max(2, int(n))
        cuts = lo[ax] + ext[ax] * (np.arange(1, k) + rng.uniform(-0.2, 0.2, k - 1)) / k
        bounds = np.concatenate([[lo[ax] - 1.0], np.sort(cuts), [hi[ax] + 1.0]])
        frac = Fracture()
        for j in range(k):
            up, dn = np.zeros(4), np.zeros(4)
            up[ax], up[3] = 1.0, bounds[j + 1]          # (below the next cut)
            dn[ax], dn[3] = -1.0, -bounds[j]            # (above the last)
            pl = np.concatenate([planes, [up, dn]])
            inner = np.concatenate([np.zeros(len(planes), bool), [j < k - 1, j > 0]])
            piece, _ = make_piece(pl, inner)
            if piece is not None:
                frac.pieces.append(piece)
        frac.bonds = touching_bonds(frac.pieces)
        return frac
    if ext[order[0]] * 5.0 <= ext[order[1]]:
        thin = int(order[0])
        seeds = _seeds(planes, n, rng, lo=lo, hi=hi)
        seeds[:, thin] = 0.5 * (lo[thin] + hi[thin])   # (right through it)
        return voronoi(planes, _spread(seeds, thin, rng, ext))
    return voronoi(planes, _seeds(planes, n, rng, lo=lo, hi=hi))


def web(size, n, rng, impact=None):
    """A pane (a box, its thin axis through it) broken as glass breaks where it is hit: a spider's web round the impact
    (the pane's middle without one): cracks radiating from it out to the frame, crossed in each wedge between two of
    them by rings that grow further apart outward (each a straight chord, a little uneven from wedge to wedge), so the
    pieces are small splinters at the hit and long daggers between the outer cracks. About n pieces."""
    s = np.abs(np.asarray(size, float))
    thin = int(np.argmin(s))
    a, b = [k for k in range(3) if k != thin]
    p0 = np.zeros(3) if impact is None else np.asarray(impact, float).copy()
    p0[a] = float(np.clip(p0[a], -0.98 * s[a], 0.98 * s[a]))
    p0[b] = float(np.clip(p0[b], -0.98 * s[b], 0.98 * s[b]))
    p0[thin] = 0.0
    box = shape_planes('box', s)
    reach = max(math.hypot(ca * s[a] - p0[a], cb * s[b] - p0[b]) for ca in (-1, 1) for cb in (-1, 1))
    K = int(np.clip(round(n / 4.0), 6, 18))                         # cracks radiating from the hit
    M = max(1, int(round(n / K)) - 1)                                # rings
    turn = rng.uniform(0.0, 2.0 * math.pi)
    ang = np.sort(turn + (np.arange(K) + rng.uniform(-0.3, 0.3, K)) * (2.0 * math.pi / K))
    r1 = max(0.04 * reach, 0.6 * float(s[thin]) * 4.0)              # the first ring: a few centimetres on a window
    r_out = 0.6 * reach                                              # the last: the outer cracks run on alone
    radii = r1 * (r_out / r1) ** (np.arange(M) / max(M - 1, 1)) if M > 1 else np.array([r1])
    frac = Fracture()

    def plane(u, d):     # a cut through the pane: normal u (in the pane, along a and b), offset d
        nrm = np.zeros(3)
        nrm[a], nrm[b] = u
        return np.append(nrm, d)

    for k in range(K):
        t0, t1 = ang[k], ang[(k + 1) % K] + (2.0 * math.pi if k == K - 1 else 0.0)
        mid, half = 0.5 * (t0 + t1), 0.5 * (t1 - t0)
        cm = np.array([math.cos(mid), math.sin(mid)])
        q0 = np.array([p0[a], p0[b]])
        cuts = [plane((math.sin(t0), -math.cos(t0)), math.sin(t0) * q0[0] - math.cos(t0) * q0[1]),
                plane((-math.sin(t1), math.cos(t1)), -math.sin(t1) * q0[0] + math.cos(t1) * q0[1])]
        rr = radii * (1.0 + rng.uniform(-0.15, 0.15, M))             # (each wedge's rings its own)
        for j in range(M + 1):
            pl = list(box) + cuts
            inner = [False] * 6 + [True, True]
            if j < M:      # inside ring j
                pl.append(plane(cm, float(cm @ q0) + rr[j] * math.cos(half)))
                inner.append(True)
            if j > 0:      # outside ring j - 1
                pl.append(plane(-cm, -(float(cm @ q0) + rr[j - 1] * math.cos(half))))
                inner.append(True)
            piece, _ = make_piece(np.asarray(pl, float), inner)
            if piece is not None:
                frac.pieces.append(piece)
    frac.bonds = touching_bonds(frac.pieces)
    return frac


def _cells_2d(seeds, box):
    """Voronoi cells (convex polygons, counter-clockwise) of 2-d seeds inside a box (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    pts_all = pts = np.asarray(seeds, float)
    out = []
    for i, c in enumerate(pts):
        poly = np.asarray([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], float)
        order = np.argsort(np.linalg.norm(pts_all - c, axis=1))
        for j in order[1:]:
            o = pts_all[j]
            if np.linalg.norm(o - c) < 1e-12:
                continue
            # half-plane nearer c than o: (o - c) . x <= (o - c) . (o + c) / 2
            nrm = o - c
            d = float(nrm @ (0.5 * (o + c)))
            # past twice the farthest corner it cuts nothing more
            if 2.0 * np.max(np.linalg.norm(poly - c, axis=1)) < np.linalg.norm(o - c):
                break
            poly = _clip_half(poly, nrm, d)
            if len(poly) < 3:
                break
        if len(poly) >= 3:
            out.append(poly)
    return out


def _clip_half(poly, nrm, d):
    """A convex polygon clipped to n . x <= d."""
    out = []
    m = len(poly)
    for k in range(m):
        p, q = poly[k], poly[(k + 1) % m]
        dp, dq = float(nrm @ p) - d, float(nrm @ q) - d
        if dp <= 0.0:
            out.append(p)
        if (dp < 0.0) != (dq < 0.0) and abs(dp - dq) > 1e-18:
            out.append(p + (q - p) * (dp / (dp - dq)))
    return np.asarray(out, float) if out else np.zeros((0, 2))


def _scatter(box, spacing, rng, near=None, spread=0.0, inside=None, wrap=None, most=600):
    """2-d points in a box (x0, y0, x1, y1) kept apart by spacing(points) (an array of distances, one each: two
    points no nearer than the mean of theirs), thrown until there is no room left: as many as the spacing makes.
    near, spread: half the throws land round this point (a hit, where the spacing is small). inside: a test the
    points must pass. wrap: x repeats with this period."""
    x0, y0, x1, y1 = box
    lo, ext = np.array([x0, y0]), np.array([x1 - x0, y1 - y0])
    pts, rad = np.zeros((0, 2)), np.zeros(0)
    misses = 0
    while misses < 400 and len(pts) < most:
        m = 64     # (a batch of throws, each tested against the points so far)
        c = lo + rng.random((m, 2)) * ext
        if near is not None:
            k = rng.random(m) < 0.5
            c[k] = np.asarray(near, float) + rng.normal(size=(int(k.sum()), 2)) * spread
            if wrap:
                c[:, 0] = x0 + (c[:, 0] - x0) % wrap
        ok = (c[:, 0] >= x0) & (c[:, 0] <= x1) & (c[:, 1] >= y0) & (c[:, 1] <= y1)
        if inside is not None:
            ok &= inside(c)
        c = c[ok]
        r = spacing(c) if len(c) else np.zeros(0)
        took = False
        for q, rq in zip(c, r):
            if len(pts):
                dd = pts - q
                if wrap:
                    dd[:, 0] = (dd[:, 0] + 0.5 * wrap) % wrap - 0.5 * wrap
                if np.any(np.hypot(dd[:, 0], dd[:, 1]) < 0.5 * (rad + rq)):
                    continue
            pts = np.vstack([pts, q])
            rad = np.append(rad, rq)
            took = True
        misses = 0 if took else misses + 1
    return pts


def _spread(seeds, thin, rng, ext):
    """Seeds flattened onto a pane's middle plane, nudged apart where two landed on top of each other."""
    out = seeds.copy()
    for i in range(len(out)):
        for j in range(i):
            if np.linalg.norm(out[i] - out[j]) < 1e-4 * float(np.max(ext)):
                d = rng.normal(size=3)
                d[thin] = 0.0
                out[i] += d / (np.linalg.norm(d) + 1e-12) * 1e-3 * float(np.max(ext))
    return out


# ---- glue between any pieces whose faces touch ---------------------------------------------------------------

def _clip_polygon(subject, clip):
    """The part of convex polygon `subject` inside convex polygon `clip` (both (m, 2), counter-clockwise)."""
    out = list(subject)
    m = len(clip)
    for k in range(m):
        a, b = clip[k], clip[(k + 1) % m]
        edge = b - a
        inside = lambda p: edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) >= -1e-12
        src, out = out, []
        if not src:
            break
        for i in range(len(src)):
            p, q = src[i], src[(i + 1) % len(src)]
            ip, iq = inside(p), inside(q)
            if ip:
                out.append(p)
            if ip != iq:
                d = q - p
                den = edge[0] * d[1] - edge[1] * d[0]
                if abs(den) > 1e-18:
                    t = (edge[1] * (p[0] - a[0]) - edge[0] * (p[1] - a[1])) / den
                    out.append(p + t * d)
    return np.asarray(out) if len(out) >= 3 else None


def _area2(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def touching_bonds(pieces, tol=1e-6):
    """Bonds between pieces whose faces lie on the same plane facing each other and overlap: the overlap's area,
    the first face's normal and the overlap's centre."""
    out = []
    cen = np.array([p.centroid for p in pieces])
    rad = np.array([float(np.linalg.norm(p.verts - p.centroid, axis=1).max()) for p in pieces])
    for i in range(len(pieces)):
        near = np.nonzero(np.linalg.norm(cen[i + 1:] - cen[i], axis=1) <= rad[i + 1:] + rad[i] + 1e-6)[0] + i + 1
        pa = pieces[i]
        for j in near:
            pb = pieces[j]
            for ka in range(len(pa.planes)):
                na, da = pa.planes[ka, :3], pa.planes[ka, 3]
                hit = np.nonzero((pb.planes[:, :3] @ na < -0.9999) & (np.abs(pb.planes[:, 3] + da) < tol * max(1.0, abs(da))))[0]
                for kb in hit:
                    # both faces in the plane's own 2D frame
                    u = np.cross(na, [1.0, 0.0, 0.0] if abs(na[0]) < 0.9 else [0.0, 1.0, 0.0])
                    u /= np.linalg.norm(u)
                    v = np.cross(na, u)
                    A = np.stack([pa.face_poly[ka] @ u, pa.face_poly[ka] @ v], 1)
                    B = np.stack([pb.face_poly[kb] @ u, pb.face_poly[kb] @ v], 1)
                    if _area2(A) < 0:
                        A = A[::-1]
                    if _area2(B) < 0:
                        B = B[::-1]
                    ov = _clip_polygon(A, B)
                    if ov is None:
                        continue
                    area = _area2(ov)
                    if area <= 1e-9:
                        continue
                    c2 = ov.mean(0)
                    out.append(Bond(i, int(j), float(area), na.copy(), na * da + u * c2[0] + v * c2[1]))
    return out


# ---- shells -----------------------------------------------------------------------------------------------------

def _wall_cells(seeds, S, R, t, y0, y1, lid, span_max, rng, near=12, tries=3):
    """Shards of a cylinder's wall (radius R, thickness t, from the base's top y0 to y1: its rim, or under its lid if
    lid): the Voronoi cells
    of seeds on its middle surface (distances measured after scaling by S), so the crack between two shards is the
    one plane both share; each cut straight through the wall, its outside a few facets, its inside a chord. A cell
    wider round than span_max (radians: its flat inside would stand off the curved wall) gets two seeds side by side
    in place of its one (cracks still where cells meet); still too wide after a few tries, it is split across its
    middle, the cut tilted at random."""
    rim = [np.array([0.0, -1.0, 0.0, -y0]), np.array([0.0, 1.0, 0.0, y1])]     # (on the base; the rim or the lid)
    ring = [np.array([math.cos(a), 0.0, math.sin(a), R]) for a in np.arange(16) * (2.0 * math.pi / 16)]
    rm = float(np.hypot(seeds[0, 0], seeds[0, 2]))

    def cuts(sp, i, k):
        order = np.argsort(np.linalg.norm(sp - sp[i], axis=1))[1:k + 1]
        nn = sp[order] - sp[i]
        d_ = (nn * (sp[order] + sp[i]) * 0.5).sum(1)
        nrm = nn * S
        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
        return list(np.concatenate([nrm / ln, (d_ / ln[:, 0])[:, None]], 1))

    def reach(v, th_i):         # how far round from th_i its corners in the wall go
        v = v[np.hypot(v[:, 0], v[:, 2]) > R - 2.0 * t]
        if not len(v):
            return th_i, th_i
        da = (np.arctan2(v[:, 2], v[:, 0]) - th_i + math.pi) % (2.0 * math.pi) - math.pi
        return th_i + float(da.min()), th_i + float(da.max())

    def rough(th_i, cut):       # its corners with the wall's outside a ring of facets, its inside deep
        own = np.array([math.cos(th_i), 0.0, math.sin(th_i)])
        return polyhedron(np.asarray(cut + rim + ring + [np.append(-own, -0.4 * (R - t))]))

    def build(th_i, cut, lo, hi):
        mid, half = 0.5 * (lo + hi), max(0.5 * (hi - lo), 1e-3)
        nm = np.array([math.cos(mid), 0.0, math.sin(mid)])
        skin = [np.array([math.cos(a), 0.0, math.sin(a), R]) for a in np.linspace(lo, hi, 5)]
        # the inside, a chord: as far in as keeps its cross-section the curved wall's (thinner at its edges,
        # thicker in its middle, as much clay)
        skin.append(np.append(-nm, -(R - t) * math.sqrt(half / math.tan(half))))
        piece, _ = make_piece(np.asarray(cut + rim + skin, float), [True] * len(cut) + [True, lid] + [False] * len(skin))
        return piece

    # cells too wide: two seeds in place of one, from their rough shapes
    for _ in range(tries):
        sp = seeds * S
        n = len(seeds)
        keep, add = [], []
        for i in range(n):
            th_i = math.atan2(seeds[i, 2], seeds[i, 0])
            v = rough(th_i, cuts(sp, i, min(near, n - 1)))
            lo, hi = reach(v, th_i) if len(v) >= 4 else (th_i, th_i)
            if hi - lo > span_max:
                for a in (0.75 * lo + 0.25 * hi, 0.25 * lo + 0.75 * hi):
                    add.append(np.array([rm * math.cos(a), seeds[i, 1], rm * math.sin(a)]))
            else:
                keep.append(i)
        if not add:
            break
        seeds = np.concatenate([seeds[keep], np.asarray(add)])
    sp = seeds * S
    n = len(seeds)
    out = []
    for i in range(n):
        th_i = math.atan2(seeds[i, 2], seeds[i, 0])
        k = min(near, n - 1)
        while True:
            cut = cuts(sp, i, k)
            v = rough(th_i, cut)
            piece = build(th_i, cut, *reach(v, th_i)) if len(v) >= 4 else None
            if piece is None or k >= n - 1:
                break
            # every corner no nearer another seed than its own: else more neighbours
            vs = piece.verts * S
            own = np.linalg.norm(vs - sp[i], axis=1)
            other = np.linalg.norm(vs[:, None, :] - sp[None, :, :], axis=2).min(1)
            if np.all(other >= own - 1e-9):
                break
            k = min(n - 1, 2 * k)
        if piece is None:
            continue
        piece = build(th_i, cut, *reach(piece.verts, th_i))     # (the chord again, for how far round it truly goes)
        todo = [(cut, piece)]
        while todo:
            cut, piece = todo.pop()
            if piece is None:
                continue
            lo, hi = reach(piece.verts, th_i)
            if hi - lo > span_max and len(cut) < near + 8:
                c_th = math.atan2(piece.centroid[2], piece.centroid[0])
                tilt = rng.uniform(-0.6, 0.6)
                tg = np.array([-math.sin(c_th), 0.0, math.cos(c_th)])
                nrm = math.cos(tilt) * tg + math.sin(tilt) * np.array([0.0, 1.0, 0.0])
                d = float(nrm @ piece.centroid)
                for half_cut in ([np.append(nrm, d)], [np.append(-nrm, -d)]):
                    c2 = cut + half_cut
                    v = rough(th_i, c2)
                    todo.append((c2, build(th_i, c2, *reach(v, th_i)) if len(v) >= 4 else None))
                continue
            out.append(piece)
    return out


def _disc_cells(R, lo, hi, below, spacing, SX, ip, reach, rng):
    """Pieces of a cylinder's base (or lid), the disc of radius R from height lo to hi: cells of it, kept apart by
    spacing (as the wall's, / sqrt(SX)), crowded round a hit on or near it. below: the base (the wall stands on its
    top), else the lid (the wall holds up its underside)."""
    ym = 0.5 * (lo + hi)
    near = None
    if ip is not None and abs(ip[1] - ym) < 0.3 * R + 0.5 * (hi - lo):     # (hit on or near it: throws round it there too)
        near = np.array([ip[0], ip[2]])
    seeds = _scatter((-R, -R, R, R), lambda c: spacing(np.stack([c[:, 0], np.full(len(c), ym), c[:, 1]], 1)) / math.sqrt(SX),
                     rng, near=near, spread=0.5 * reach, inside=lambda c: np.hypot(c[:, 0], c[:, 1]) < R)
    if len(seeds) < 2:
        seeds = np.array([[0.3 * R, 0.0], [-0.3 * R, 0.0]])
    disc = [np.append([math.cos(a), 0.0, math.sin(a)], R) for a in np.arange(20) * (2.0 * math.pi / 20)]
    out = []
    for poly in _cells_2d(seeds, (-R, -R, R, R)):
        pl = list(disc)
        inner = [False] * 20
        m = len(poly)
        cx, cz = poly[:, 0].mean(), poly[:, 1].mean()
        for k in range(m):
            p_, q_ = poly[k], poly[(k + 1) % m]
            e = q_ - p_
            nrm = np.array([e[1], 0.0, -e[0]])
            nl = float(np.linalg.norm(nrm))
            if nl < 1e-12:
                continue
            nrm /= nl
            d = float(nrm[0] * p_[0] + nrm[2] * p_[1])
            if nrm[0] * cx + nrm[2] * cz > d:
                nrm, d = -nrm, -d
            on_box = abs(abs(p_[0]) - R) < 1e-9 and abs(abs(q_[0]) - R) < 1e-9 or abs(abs(p_[1]) - R) < 1e-9 and abs(abs(q_[1]) - R) < 1e-9
            pl.append(np.append(nrm, d))
            inner.append(not on_box)
        pl += [[0.0, 1.0, 0.0, hi], [0.0, -1.0, 0.0, -lo]]
        inner += [below, not below]           # (the face the wall meets is glued to it)
        piece, _ = make_piece(np.asarray(pl, float), inner)
        if piece is not None:
            out.append(piece)
    return out


def shell(shape, size, wall, pieces=24, seed=0, pattern='voronoi', impact=None, rim=None):
    """A hollow object (wall thickness `wall`) cut as a shell: a cylinder into irregular curved shards of its wall,
    crowded round the impact, and pieces of its base, and of its top unless it is open there (rim: how high its wall
    then goes, a pot's); a box wall by wall."""
    rng = np.random.default_rng(int(seed) + 104729)
    s = np.abs(np.asarray(size, float))
    t = float(min(max(wall, 1e-3), 0.45 * s.min()))
    frac = Fracture()
    if shape == 'cylinder':
        R, H = s[0], s[1]
        # the wall's top: the lid's underside, or its rim
        top = H - t if rim is None else float(np.clip(rim, -H + 3.0 * t, H))
        wall_area = 2.0 * math.pi * R * (top + H - t)
        base_area = math.pi * R * R * (1.0 if rim is not None else 2.0)
        rm = R - 0.5 * t
        L = 2.0 * math.pi * R
        # the widest round a shard can go: a convex piece's flat inside stands off the curved wall at its edges by
        # about (R - t) x (half its angle)^2 / 3; at most 0.4 of the wall
        span_max = float(np.clip(2.0 * math.sqrt(1.2 * t / max(R - t, 1e-6)), 0.35, 1.2))
        SX = 2.2                  # (distances round it count this much more: the shards come narrower round, taller, as
                                  # cracks run up a pot's wall from where it lands)
        # the spacing of the cracks (on the wall with distances round x SX; on the base / sqrt(SX), as large shards):
        # as many pieces as asked for, no wider than span_max; down to a few wall thicknesses round the hit, opening
        # out over about the object's size
        far = min(math.sqrt(0.7 * SX * (wall_area + base_area) / max(int(pieces), 1)), SX * span_max * R / 1.5)
        small = max(0.15 * far, min(far, 1.2 * t * math.sqrt(SX)))
        ip = None if impact is None else np.asarray(impact, float)
        reach = 0.5 * (H + R)

        def spacing(p3):
            if ip is None:
                return np.full(len(p3), far)
            d = np.linalg.norm(p3 - ip, axis=1) / reach
            return small + (far - small) * np.clip(d, 0.0, 1.0)

        def on_wall(c):           # unrolled, stretched (arc x SX, height) -> on the wall's middle
            a = c[:, 0] / (SX * R)
            return np.stack([rm * np.cos(a), c[:, 1], rm * np.sin(a)], 1)

        near_w = spread_w = None
        if ip is not None:
            th = math.atan2(ip[2], ip[0]) % (2.0 * math.pi)
            near_w = np.array([R * th * SX, float(np.clip(ip[1], -H + t, top))])
            spread_w = 0.5 * reach
        sd = _scatter((0.0, -H + t, L * SX, top), lambda c: spacing(on_wall(c)), rng, near=near_w, spread=spread_w,
                      wrap=L * SX)
        wall = _wall_cells(on_wall(sd), np.array([1.0, 1.0 / SX, 1.0]), R, t, -H + t, top, rim is None, span_max, rng)
        frac.pieces.extend(wall)
        # the base, and a lid: cells of the disc, the same spacing
        for lo, hi, below in ((-H, -H + t, True),) + (((top, H, False),) if rim is None else ()):
            frac.pieces.extend(_disc_cells(R, lo, hi, below, spacing, SX, ip, reach, rng))
    else:
        # a box: its six walls, each cut into chunks in proportion to its area
        walls = []
        for ax in range(3):
            for sign in (-1.0, 1.0):
                lo, hi = -s.copy(), s.copy()
                if sign < 0:
                    hi[ax] = -s[ax] + t
                else:
                    lo[ax] = s[ax] - t
                # (the side walls stand between the top and bottom ones; front and back between the sides)
                for other in range(ax):
                    lo[other], hi[other] = -s[other] + t, s[other] - t
                walls.append((lo, hi))
        areas = np.array([np.prod(np.sort(hi - lo)[1:]) for lo, hi in walls])
        for (lo, hi), a in zip(walls, areas):
            n = max(1, int(round(pieces * a / areas.sum())))
            half, mid = 0.5 * (hi - lo), 0.5 * (hi + lo)
            if n == 1:
                pl = shape_planes('box', half)
                pl[:, 3] += pl[:, :3] @ mid
                p, _ = make_piece(pl, np.zeros(6, bool))
                sub = [p] if p is not None else []
            else:
                sub = fracture('box', half, n, pattern, seed=int(rng.integers(1 << 30))).pieces
                for p in sub:
                    p.planes[:, 3] += p.planes[:, :3] @ mid
                    p.verts = p.verts + mid
                    p.centroid = p.centroid + mid
                    p.face_centre = p.face_centre + mid
                    p.face_poly = [q + mid for q in p.face_poly]
            for p in sub:
                # its faces on the box's own outside or inside surfaces are surfaces, not cuts
                on_box = np.abs(np.abs(p.planes[:, 3]) - s @ np.abs(p.planes[:, :3].T)) < 1e-9
                on_inside = np.abs(np.abs(p.planes[:, 3]) - (s - t) @ np.abs(p.planes[:, :3].T)) < 1e-9
                p.inner = p.inner & ~on_box & ~on_inside
            frac.pieces.extend(sub)
    bonds = touching_bonds(frac.pieces)
    if bonds:
        # (slivers where two faces barely overlap hold nothing: small against the smaller piece's own size, so the
        # small shards round a hit keep their bonds)
        size = [max(float(p.volume), 1e-15) ** (2.0 / 3.0) for p in frac.pieces]
        bonds = [b for b in bonds if b.area > 0.05 * min(size[b.i], size[b.j])]
    frac.bonds = bonds
    return frac
