"""Breaking things: an object cut beforehand into convex pieces, and the bonds that glue them (engine/solids.py
holds the pieces together with welds that break when a bond is overloaded).

A piece is a convex polyhedron in the object's own frame (metres): the planes that bound it (n . x <= d, n a unit
normal), its corners, its centroid and its volume. Each plane is either one of the object's own faces or a cut
(an inner face, drawn in the material's inside colour). Two pieces that share a cut are bonded across it: the
shared face's area, its normal (from the first piece toward the second) and its centre.

Patterns:
- voronoi: cells round seeds spread evenly through the object (or crowded round an impact point);
- bricks: a box as bricks in running bond (a wall); each brick is a piece and the mortar joints are the bonds;
- shards: glass: seeds crowded round a point of the largest face, so a pane breaks into slivers radiating from it;
- splinters: wood: cells stretched along the grain (the object's long axis).

A hollow object (a vase, a tank, a crate) breaks as a shell: a cylinder into curved strips of its wall and wedges of its
base, a box wall by wall, each wall into chunks or splinters. The shell's inside is its own surface, not a cut.

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


def fracture(shape, size, pieces=24, pattern='voronoi', seed=0, impact=None, hollow=0.0):
    """Cut an object (shape, size as a collider has them) into about `pieces` convex pieces (a shell if it is
    hollow: wall thickness `hollow`)."""
    if hollow > 0.0 and shape in ('cylinder', 'box') and pattern != 'bricks':
        return shell(shape, size, hollow, pieces, seed, 'splinters' if pattern == 'splinters' else 'voronoi')
    rng = np.random.default_rng(int(seed) + 7919)
    if pattern == 'bricks' and shape == 'box':
        return bricks(size)
    planes = shape_planes(shape, size)
    lo, hi = planes_bounds(planes)
    n = max(2, int(pieces))
    s = np.abs(np.asarray(size, float))
    if pattern == 'shards':
        # glass: crowded round a point on the middle of its largest face, slivers radiating from it
        thin = int(np.argmin(hi - lo))
        p0 = np.zeros(3) if impact is None else np.asarray(impact, float)
        seeds = _seeds(planes, n, rng, impact=p0, crowd=0.8, lo=lo, hi=hi)
        seeds[:, thin] = 0.0        # through the whole thickness
        return voronoi(planes, _spread(seeds, thin, rng, hi - lo))
    if pattern == 'splinters':
        long_ax = int(np.argmax(hi - lo))
        stretch = np.ones(3)
        stretch[long_ax] = 0.3      # distances along the grain count a third: long cells
        seeds = _seeds(planes, n, rng, lo=lo, hi=hi)
        return voronoi(planes, seeds, stretch=stretch)
    seeds = _seeds(planes, n, rng, impact=impact, crowd=0.6 if impact is not None else 0.0, lo=lo, hi=hi)
    del s
    return voronoi(planes, seeds)


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

def shell(shape, size, wall, pieces=24, seed=0, pattern='voronoi'):
    """A hollow object (wall thickness `wall`) cut as a shell: a cylinder into strips of its wall in bands (the
    cuts a little ragged) and wedges of its base; a box wall by wall."""
    rng = np.random.default_rng(int(seed) + 104729)
    s = np.abs(np.asarray(size, float))
    t = float(min(max(wall, 1e-3), 0.45 * s.min()))
    frac = Fracture()
    if shape == 'cylinder':
        R, H = s[0], s[1]
        bands = max(1, int(round(max(pieces, 2) / 14.0)))
        around = max(14, int(round(max(pieces, 2) / (bands + 0.5))))   # (narrow strips: their flat insides stay close to the wall)
        edges_y = np.linspace(-H + t, H, bands + 1)
        edges_y[1:-1] += rng.uniform(-0.25, 0.25, bands - 1) * (2 * H / bands)
        for b in range(bands):
            y0, y1 = edges_y[b], edges_y[b + 1]
            cuts = np.sort((np.arange(around) + rng.uniform(-0.3, 0.3, around)) * (2 * math.pi / around) + rng.uniform(0, 1))
            for k in range(around):
                a0, a1 = cuts[k], cuts[(k + 1) % around] + (2 * math.pi if k == around - 1 else 0.0)
                mid = 0.5 * (a0 + a1)
                n_out = [np.array([math.cos(a), 0.0, math.sin(a)]) for a in np.linspace(a0, a1, 4)]
                pl = [np.append(n, R) for n in n_out]                     # the outside, as a few facets
                nm = np.array([math.cos(mid), 0.0, math.sin(mid)])
                pl.append(np.append(-nm, -(R - t) * math.cos(0.5 * (a1 - a0)) * 0.999))   # the inside (a chord)
                pl.append(np.append([math.sin(a0), 0.0, -math.cos(a0)], 0.0))  # the cut at a0 (its normal away from the strip)
                pl.append(np.append([-math.sin(a1), 0.0, math.cos(a1)], 0.0))  # the cut at a1
                pl.append([0.0, 1.0, 0.0, y1])
                pl.append([0.0, -1.0, 0.0, -y0])
                inner = [False] * 4 + [False, True, True, b < bands - 1, True]
                p, _ = make_piece(np.asarray(pl, float), inner)
                if p is not None:
                    frac.pieces.append(p)
        # the base: wedges
        wedges = max(2, around // 2)
        cuts = np.sort(np.arange(wedges) * (2 * math.pi / wedges) + rng.uniform(-0.2, 0.2, wedges))
        for k in range(wedges):
            a0, a1 = cuts[k], cuts[(k + 1) % wedges] + (2 * math.pi if k == wedges - 1 else 0.0)
            pl = [np.append([math.cos(a), 0.0, math.sin(a)], R) for a in np.linspace(a0, a1, 4)]
            pl.append(np.append([math.sin(a0), 0.0, -math.cos(a0)], 0.0))
            pl.append(np.append([-math.sin(a1), 0.0, math.cos(a1)], 0.0))
            pl.append([0.0, 1.0, 0.0, -H + t])
            pl.append([0.0, -1.0, 0.0, H])
            p, _ = make_piece(np.asarray(pl, float), [False] * 4 + [True, True, False, False])
            if p is not None:
                frac.pieces.append(p)
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
        # (slivers where two faces barely overlap hold nothing)
        typical = float(np.median([b.area for b in bonds]))
        bonds = [b for b in bonds if b.area > 0.15 * typical]
    frac.bonds = bonds
    return frac
