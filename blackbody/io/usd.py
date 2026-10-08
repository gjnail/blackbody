"""Scene import from USD (Universal Scene Description): cameras, objects and lights.

Blender, Houdini, Maya, Cinema 4D, Unreal and most 3D trackers export USD. Alembic (.abc) has no
reader for Python on Windows, so convert Alembic to USD first (in Blender: File > Import > Alembic,
then File > Export > Universal Scene Description; Houdini and Maya export USD directly).

Units and axes are converted: the stage's metersPerUnit becomes metres, and a Z-up stage is turned
y-up. USD time codes are read as frames at the stage's frame rate, so frame 1001 in the USD scene is
frame 1001 in Blackbody, unless an offset is given: with offset 1000, USD frame 1001 is frame 1 (the
keys move, and objects that change read the USD 1000 frames on, through their Mesh frame offset).
import_usd can also take the stage's frame range and rate for the scene (match_range).

What is read:
  cameras        the free camera, keyed on every frame (position, rotation, focal length, sensor)
  meshes         polygon meshes, including skinned characters (UsdSkel: the skinning is baked into
                 the points, in memory) and meshes inside instances (instanceable references)
  shapes         Cube, Sphere, Cylinder, Capsule and Cone prims: an exact box, sphere or cylinder
                 collider when the shape and its placement allow, otherwise the shape as a mesh
  point instancers  every instance of every prototype (instancers inside prototypes included), as
                 one object
  curves         BasisCurves (linear, or cubic Bezier, B-spline and Catmull-Rom), NURBS and Hermite
                 curves, as tubes as thick as their widths: cables, ropes, branches, fuses
  points         Points prims: a particle system (points that move or carry velocities) becomes a
                 source, its points splatted into a field like a volume's (io/volume.py), moving with
                 their velocities; still points without velocities (debris, gravel) become solid
                 balls as wide as their widths
  volumes        Volume prims with an OpenVDB field (density, or the first field): the region where
                 the field is above a quarter of its high values (inside, for a level set), as a solid;
                 a VDB sequence (animated file path) deforms frame by frame
  lights         the first distant light becomes the key light (direction, colour, intensity), the
                 first dome light the ambient light (and its texture the environment image); sphere,
                 disk, rect and cylinder lights become lights in the set (spots, with a cone; a light
                 with an IES profile shines through it, a panel's as a point light), which light the
                 smoke, with their shadow switch. A portal becomes a panel of the dome's light shining
                 in through it; a geometry light (GeometryLight, MeshLightAPI) a point light at its
                 geometry giving off what its surface does. Further distant and dome lights, and light
                 textures, are listed as not imported.
Invisible prims and guides are skipped.

Objects become colliders (or emitters) in one of three ways:
  static   an object that does not move: its shape in its own frame, placed by position, rotation
           about the vertical and size
  rigid    an object that moves without changing shape and only turns about the vertical: the same,
           with position and rotation (and size, if it scales) keyframed on every frame
  world    anything else (a deforming character, a tumbling object, a point instancer): its shape in
           world space on every frame, baked as a deforming mesh (source 'file.usd#/prim?world')

A mesh source 'file.usd#/World/Car' (see engine/mesh.py) reads the prim's geometry in its own frame;
'...?world' reads it in world space at the frame asked for.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger('blackbody.usd')

WORLD = '?world'
SHAPES = ('Cube', 'Sphere', 'Cylinder', 'Capsule', 'Cone')
CURVES = ('BasisCurves', 'NurbsCurves', 'HermiteCurves')
GEOM = ('Mesh',) + SHAPES + CURVES + ('Points', 'Volume')
LOCAL_LIGHTS = ('SphereLight', 'DiskLight', 'RectLight', 'CylinderLight')
DEFAULT_WIDTH = 0.02        # m: curves and points without widths
VOLUME_MAX_CELLS = 160      # a volume's surface is built on at most this many cells a side
MAX_TRIANGLES = 2_000_000   # a point instancer beyond this is cut down (a warning says so)
LUX_REFERENCE = 50000.0     # a distant light of this many lux (USD's default, bright daylight) is Key intensity 3
_stages = {}


class USDError(ValueError):
    pass


def _pxr():
    try:
        from pxr import Usd, UsdGeom  # noqa: F401
    except ImportError as ex:  # pragma: no cover
        raise USDError('USD support needs the usd-core package (pip install usd-core).') from ex
    from pxr import Usd, UsdGeom
    return Usd, UsdGeom


def open_stage(path):
    """The stage for a USD file (.usd, .usda, .usdc, .usdz), cached until the file changes. Skinned
    meshes are baked (in memory: the file is never written) so their points are animated."""
    Usd, _ = _pxr()
    p = str(Path(path).resolve())
    if not Path(p).exists():
        raise USDError(f'USD file not found: {path}')
    key = (p, os.path.getmtime(p))
    st = _stages.get(key)
    if st is None:
        st = Usd.Stage.Open(p)
        if st is None:
            raise USDError(f'Cannot open {path} as USD')
        _bake_skinning(st)
        for k in [k for k in _stages if k[0] == p]:
            del _stages[k]
        _stages[key] = st
    return st


def _bake_skinning(stage):
    from pxr import UsdSkel
    if not any(p.IsA(UsdSkel.Root) for p in stage.Traverse()):
        return
    stage.SetEditTarget(stage.GetSessionLayer())   # the file itself is never changed
    try:
        UsdSkel.BakeSkinning(stage.Traverse())
    except Exception as ex:  # an odd rig: fall back to the rest pose
        log.warning('Could not bake the skinning in %s: %s', stage.GetRootLayer().identifier, ex)


def axis_matrix(stage):
    """4x4 turning the stage's axes y-up and its units into metres."""
    _, UsdGeom = _pxr()
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    m = np.eye(4) * mpu
    m[3, 3] = 1.0
    if str(UsdGeom.GetStageUpAxis(stage)).upper() == 'Z':
        c = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        m = c @ m
    return m


def _turn(ax):
    return ax[:3, :3] / abs(np.linalg.det(ax[:3, :3])) ** (1.0 / 3.0)


def frame_rate(stage):
    fps = stage.GetFramesPerSecond() or stage.GetTimeCodesPerSecond() or 24.0
    return float(fps)


def time_code(stage, frame):
    """The USD time code of a Blackbody frame (frames at the stage's frame rate)."""
    Usd, _ = _pxr()
    if frame is None:
        return Usd.TimeCode.EarliestTime()
    tcps = stage.GetTimeCodesPerSecond() or 24.0
    return Usd.TimeCode(float(frame) * tcps / frame_rate(stage))


def frame_range(stage):
    tcps = stage.GetTimeCodesPerSecond() or 24.0
    k = frame_rate(stage) / tcps
    if not stage.HasAuthoredTimeCodeRange():
        return None
    return int(round(stage.GetStartTimeCode() * k)), int(round(stage.GetEndTimeCode() * k))


def _world(prim, tc):
    """Prim-to-world 4x4 (column vectors) at a time code, in the stage's own units and axes."""
    _, UsdGeom = _pxr()
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(tc)
    return np.array(m, dtype=np.float64).T   # Gf matrices act on row vectors


def _prim(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise USDError(f'No prim {prim_path} in the USD file')
    return prim


def _split(prim_path):
    if prim_path.endswith(WORLD):
        return prim_path[:-len(WORLD)], True
    return prim_path, False


def _range(prim):
    """The prim and everything under it, through instances (instance proxies)."""
    Usd, _ = _pxr()
    return Usd.PrimRange(prim, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))


def _shown(prim, tc=None):
    """Not invisible, and not a guide."""
    Usd, UsdGeom = _pxr()
    img = UsdGeom.Imageable(prim)
    if not img:
        return True
    if img.ComputeVisibility(tc if tc is not None else Usd.TimeCode.EarliestTime()) == UsdGeom.Tokens.invisible:
        return False
    return img.ComputePurpose() != UsdGeom.Tokens.guide


def _geometry(prim):
    """The geometric prims a prim stands for: itself, or what is under it (point instancers count as
    one; their prototypes are only drawn through the instancer)."""
    t = prim.GetTypeName()
    if t in GEOM or t == 'PointInstancer':
        return [prim]
    out = []
    it = iter(_range(prim))
    for p in it:
        if p == prim:
            continue
        tp = p.GetTypeName()
        if tp == 'PointInstancer':
            out.append(p)
            it.PruneChildren()
        elif tp in GEOM and _shown(p):
            out.append(p)
    return out


# -- geometry ------------------------------------------------------------------------------------------

def _triangles(mesh, tc):
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(tc) or [], np.int64)
    idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(tc) or [], np.int64)
    if not len(counts):
        return np.zeros((0, 3), np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    tris = []
    for k in range(3, int(counts.max()) + 1):  # fans, one polygon size at a time
        sel = np.where(counts == k)[0]
        if not len(sel):
            continue
        s0 = starts[sel]
        for j in range(1, k - 1):
            tris.append(np.stack([idx[s0], idx[s0 + j], idx[s0 + j + 1]], 1))
    return np.concatenate(tris) if tris else np.zeros((0, 3), np.int64)


def _revolve(profile, segments=24):
    """A closed surface of revolution about +z from (radius, z) points, bottom to top."""
    prof = np.asarray(profile, np.float64)
    ang = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    v = np.stack([np.outer(prof[:, 0], np.cos(ang)), np.outer(prof[:, 0], np.sin(ang)),
                  np.repeat(prof[:, 1:2], segments, 1)], -1).reshape(-1, 3)
    n = len(prof)
    i, j = np.meshgrid(np.arange(n - 1), np.arange(segments), indexing='ij')
    a = i * segments + j
    b = i * segments + (j + 1) % segments
    c = (i + 1) * segments + (j + 1) % segments
    d = (i + 1) * segments + j
    tris = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    return v, tris


def _outward(v, tris):
    """Turn every triangle of a convex shape centred on the origin to face outward."""
    p = v[tris]
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    flip = np.einsum('ij,ij->i', nrm, p.mean(1)) < 0
    tris = tris.copy()
    tris[flip] = tris[flip][:, ::-1]
    return tris


def _shape_mesh(prim, tc):
    """A Cube, Sphere, Cylinder, Capsule or Cone prim as (points, triangles) in its own frame."""
    _, UsdGeom = _pxr()
    t = prim.GetTypeName()
    if t == 'Cube':
        h = 0.5 * float(UsdGeom.Cube(prim).GetSizeAttr().Get(tc) or 2.0)
        v = np.array([(x, y, z) for x in (-h, h) for y in (-h, h) for z in (-h, h)], np.float64)
        quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        tris = np.array([(q[0], q[1], q[2]) for q in quads] + [(q[0], q[2], q[3]) for q in quads], np.int64)
        return v, _outward(v, tris)
    if t == 'Sphere':
        r = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get(tc) or 1.0)
        th = np.linspace(-0.5 * math.pi, 0.5 * math.pi, 13)
        v, tris = _revolve(np.stack([r * np.cos(th), r * np.sin(th)], -1))
        return v, _outward(v, tris)
    api = {'Cylinder': UsdGeom.Cylinder, 'Capsule': UsdGeom.Capsule, 'Cone': UsdGeom.Cone}[t](prim)
    r = float(api.GetRadiusAttr().Get(tc) or 1.0)
    hh = 0.5 * float(api.GetHeightAttr().Get(tc) or 2.0)
    if t == 'Cylinder':
        prof = [(0.0, -hh), (r, -hh), (r, hh), (0.0, hh)]
    elif t == 'Cone':
        prof = [(0.0, -hh), (r, -hh), (0.0, hh)]
    else:
        th = np.linspace(-0.5 * math.pi, 0.0, 7)
        lower = [(r * math.cos(a), -hh + r * math.sin(a)) for a in th]
        prof = lower + [(x, -z) for x, z in reversed(lower)]
    v, tris = _revolve(prof)
    axis = str(api.GetAxisAttr().Get() or 'Z').upper()
    if axis == 'X':
        v = v[:, [2, 0, 1]]
    elif axis == 'Y':
        v = v[:, [1, 2, 0]]
    return v, _outward(v, tris)


def _bezier(p, n=6):
    """Cubic Bezier segments (every 3 points after the first), sampled."""
    out = []
    t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    for i in range(0, len(p) - 3, 3):
        a, b, c, d = p[i:i + 4]
        out.append((1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t ** 2 * c + t ** 3 * d)
    out.append(p[-1:] if len(p) >= 4 else p)
    return np.concatenate(out)


def _bspline(p, n=6, catmull=False):
    """Uniform cubic B-spline (or Catmull-Rom) through a control polygon, sampled."""
    if len(p) < 4:
        return p
    t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    out = []
    for i in range(len(p) - 3):
        a, b, c, d = p[i:i + 4]
        if catmull:
            out.append(0.5 * ((2 * b) + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t ** 2 + (-a + 3 * b - 3 * c + d) * t ** 3))
        else:
            out.append(((1 - t) ** 3 * a + (3 * t ** 3 - 6 * t ** 2 + 4) * b + (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) * c
                        + t ** 3 * d) / 6.0)
    last = p[-2] if catmull else (p[-3] + 4 * p[-2] + p[-1]) / 6.0
    out.append(last[None])
    return np.concatenate(out)


def _tube(q, radius, sides=8):
    """A closed tube of `radius` along the polyline q (m, 3), with flat caps."""
    q = np.asarray(q, np.float64)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(q, axis=0), axis=1) > 1e-9])
    q = q[keep]
    if len(q) < 2:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    tan = np.gradient(q, axis=0)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-12)
    ref = np.array([0.0, 1.0, 0.0]) if abs(tan[0, 1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(tan[0], ref)
    u /= np.linalg.norm(u)
    us = []
    for k in range(len(q)):   # parallel transport, so the tube does not twist
        u = u - tan[k] * np.dot(u, tan[k])
        u /= max(np.linalg.norm(u), 1e-12)
        us.append(u)
    us = np.array(us)
    vs = np.cross(tan, us)
    ang = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    r = np.broadcast_to(np.asarray(radius, np.float64), (len(q),))[:, None, None]
    ring = q[:, None, :] + r * (np.cos(ang)[None, :, None] * us[:, None, :] + np.sin(ang)[None, :, None] * vs[:, None, :])
    v = np.concatenate([ring.reshape(-1, 3), q[[0, -1]]])
    n = len(q)
    i, j = np.meshgrid(np.arange(n - 1), np.arange(sides), indexing='ij')
    a, b = i * sides + j, i * sides + (j + 1) % sides
    c, d = (i + 1) * sides + (j + 1) % sides, (i + 1) * sides + j
    tris = [np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)]
    start, end = n * sides, n * sides + 1
    jj = np.arange(sides)
    tris.append(np.stack([np.full(sides, start), (jj + 1) % sides, jj], 1))
    tris.append(np.stack([np.full(sides, end), (n - 1) * sides + jj, (n - 1) * sides + (jj + 1) % sides], 1))
    t = np.concatenate(tris).astype(np.int64)
    # outward: away from the centre line (sides) or along the tube's ends (caps)
    centre = np.concatenate([np.repeat(q, sides, 0), q[[0, -1]]])
    p = v[t]
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    cap = (t == start).any(1) | (t == end).any(1)
    out = np.where(cap[:, None], np.where((t == start).any(1)[:, None], -tan[0], tan[-1]), p.mean(1) - centre[t[:, 0]])
    flip = np.einsum('ij,ij->i', nrm, out) < 0
    t[flip] = t[flip][:, ::-1]
    return v, t


def _curves_mesh(prim, tc):
    """A curves prim as tubes (stage units)."""
    _, UsdGeom = _pxr()
    cv = UsdGeom.Curves(prim)
    pts = np.asarray(cv.GetPointsAttr().Get(tc) or [], np.float64).reshape(-1, 3)
    counts = list(cv.GetCurveVertexCountsAttr().Get(tc) or [len(pts)])
    widths = np.asarray(cv.GetWidthsAttr().Get(tc) or [], np.float64)
    mpu = float(UsdGeom.GetStageMetersPerUnit(prim.GetStage()) or 1.0)
    kind, basis, wrap = 'linear', 'bezier', 'nonperiodic'
    if prim.GetTypeName() == 'BasisCurves':
        bc = UsdGeom.BasisCurves(prim)
        kind = str(bc.GetTypeAttr().Get() or 'cubic')
        basis = str(bc.GetBasisAttr().Get() or 'bezier')
        wrap = str(bc.GetWrapAttr().Get() or 'nonperiodic')
    vs, ts, n, start = [], [], 0, 0
    for ci, k in enumerate(counts):
        p = pts[start:start + k]
        w = widths[start:start + k] if len(widths) == len(pts) else (widths[ci:ci + 1] if len(widths) == len(counts)
                                                                     else widths[:1])
        start += k
        if len(p) < 2:
            continue
        if wrap == 'periodic':
            p = np.concatenate([p, p[:3 if kind == 'cubic' else 1]])
        if kind == 'cubic' and basis == 'bezier':
            q = _bezier(p)
        elif kind == 'cubic' and basis in ('bspline', 'catmullRom'):
            q = _bspline(p, catmull=basis == 'catmullRom')
        else:
            q = p
        radius = 0.5 * (float(w.mean()) if len(w) else DEFAULT_WIDTH / mpu)
        v, tri = _tube(q, radius)
        if len(tri):
            vs.append(v)
            ts.append(tri + n)
            n += len(v)
    return _join(vs, ts)


def ball(faces=80):
    """A closed ball around the origin: an icosphere of 80 or 20 triangles, or an octahedron (8), as big as it must be
    to hold a unit ball's volume (so a coarse one is not smaller than the ball it stands for)."""
    if faces == 8:
        v = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], np.float64)
        t = np.array([(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)], np.int64)
    else:
        g = (1.0 + 5.0 ** 0.5) / 2.0
        v = np.array([(-1, g, 0), (1, g, 0), (-1, -g, 0), (1, -g, 0), (0, -1, g), (0, 1, g), (0, -1, -g), (0, 1, -g),
                      (g, 0, -1), (g, 0, 1), (-g, 0, -1), (-g, 0, 1)], np.float64)
        t = np.array([(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
                      (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11),
                      (6, 2, 10), (8, 6, 7), (9, 8, 1)], np.int64)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        while len(t) < faces:   # split every triangle in four, its new corners pushed out onto the sphere
            mids = {}

            def mid(a, b):
                k = (min(a, b), max(a, b))
                if k not in mids:
                    mids[k] = len(v) + len(mids)
                return mids[k]
            new = []
            for a, b, c in t:
                ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
                new += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
            extra = np.zeros((len(mids), 3))
            for (a, b), i in mids.items():
                m = v[a] + v[b]
                extra[i - len(v)] = m / np.linalg.norm(m)
            v, t = np.concatenate([v, extra]), np.array(new, np.int64)
    t = _outward(v, t)
    p = v[t]
    vol = abs(np.einsum('ij,ij->i', p[:, 0], np.cross(p[:, 1], p[:, 2])).sum()) / 6.0
    return v * (4.0 / 3.0 * math.pi / vol) ** (1.0 / 3.0), t


def _point_radii(prim, tc, n):
    """Half the widths of a Points prim's n points (stage units)."""
    _, UsdGeom = _pxr()
    widths = np.asarray(UsdGeom.Points(prim).GetWidthsAttr().Get(tc) or [], np.float64)
    mpu = float(UsdGeom.GetStageMetersPerUnit(prim.GetStage()) or 1.0)
    return 0.5 * (widths if len(widths) == n else np.full(n, widths[0] if len(widths) else DEFAULT_WIDTH / mpu))


def _ball_faces(n):
    """Triangles a ball when n of them stand for a Points prim: as many as MAX_TRIANGLES allows (80, 20 or 8)."""
    return next((f for f in (80, 20) if n * f <= MAX_TRIANGLES), 8)


def _points_mesh(prim, tc):
    """A Points prim as solid balls, one at every point as wide as its width (stage units): instanced spheres (see
    ball), as round as MAX_TRIANGLES allows for their number."""
    _, UsdGeom = _pxr()
    pt = UsdGeom.Points(prim)
    pts = np.asarray(pt.GetPointsAttr().Get(tc) or [], np.float64).reshape(-1, 3)
    if not len(pts):
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    r = _point_radii(prim, tc, len(pts))
    cap = max(1, MAX_TRIANGLES // 8)
    if len(pts) > cap:
        log.warning('%s: more than %d points; the rest are left out', prim.GetPath(), cap)
        pts, r = pts[:cap], r[:cap]
    bv, bt = ball(_ball_faces(len(pts)))
    v = (pts[:, None, :] + r[:, None, None] * bv[None]).reshape(-1, 3)
    t = (bt[None] + len(bv) * np.arange(len(pts))[:, None, None]).reshape(-1, 3)
    return v, t


def points_are_particles(prim):
    """Whether a Points prim is a particle system (its points move, or carry velocities): it comes in as a source,
    splatted into a field (io/volume.py), rather than as solid balls (gravel, debris)."""
    Usd, UsdGeom = _pxr()
    pt = UsdGeom.Points(prim)
    if pt.GetPointsAttr().GetNumTimeSamples() > 1:
        return True
    a = pt.GetVelocitiesAttr()
    if a.GetNumTimeSamples() > 1:
        return True
    vel = a.Get(Usd.TimeCode.EarliestTime())   # (a one-frame cache's velocities are a time sample, not a default)
    return vel is not None and len(vel) > 0 and float(np.abs(np.asarray(vel, np.float64)).max()) > 0.0


def _volume_field(prim, tc):
    """(VDB file, grid name) of a Volume prim's density field (or its first field)."""
    from pxr import UsdVol
    vol = UsdVol.Volume(prim)
    fields = dict(vol.GetFieldPaths())
    if not fields:
        raise USDError(f'{prim.GetPath()} has no fields')
    name = 'density' if 'density' in fields else sorted(fields)[0]
    fp = prim.GetStage().GetPrimAtPath(fields[name])
    asset = UsdVol.OpenVDBAsset(fp)
    ap = asset.GetFilePathAttr().Get(tc)
    path = (ap.resolvedPath or ap.path) if ap is not None else ''
    if path and not Path(path).is_absolute():
        path = str(Path(prim.GetStage().GetRootLayer().realPath).parent / path)
    grid = asset.GetFieldNameAttr().Get(tc) or name
    return path, str(grid)


def _volume_mesh(prim, tc):
    """A Volume prim's region above its iso-level, as a solid (in the prim's own frame)."""
    from .vdbread import read_float_grid, voxel_surface
    path, grid = _volume_field(prim, tc)
    if not path or not Path(path).exists():
        raise USDError(f'{prim.GetPath()}: the VDB file {path or "(none)"} is missing')
    dense, lo, xf, cls = read_float_grid(path, grid, with_class=True)
    if cls == 'level set':
        occ = dense < 0.0
    else:
        pos = dense[dense > 0]
        occ = dense > (0.25 * float(np.percentile(pos, 99)) if len(pos) else np.inf)
    step = max(1, int(math.ceil(max(occ.shape) / VOLUME_MAX_CELLS)))
    if step > 1:   # coarser cells: a cell is solid if anything in it is
        pad = [(0, (-s) % step) for s in occ.shape]
        o = np.pad(occ, pad)
        occ = o.reshape(o.shape[0] // step, step, o.shape[1] // step, step, o.shape[2] // step, step).any(axis=(1, 3, 5))
    v, t = voxel_surface(occ)
    if not len(t):
        return v, t
    idx = np.asarray(lo, np.float64) + v * step - 0.5   # VDB voxel centres sit at integer index coordinates
    w = (xf[:3, :3] @ idx.T).T + xf[:3, 3]
    if np.linalg.det(xf[:3, :3]) < 0:
        t = t[:, ::-1]
    return w, t


def _local_geometry(prim, tc):
    """(points, triangles) of a Mesh, shape, curves, points or volume prim in its own frame (stage
    units), faces outward."""
    _, UsdGeom = _pxr()
    tp = prim.GetTypeName()
    if tp in CURVES:
        return _curves_mesh(prim, tc)
    if tp == 'Points':
        return _points_mesh(prim, tc)
    if tp == 'Volume':
        return _volume_mesh(prim, tc)
    if tp != 'Mesh':
        return _shape_mesh(prim, tc)
    mesh = UsdGeom.Mesh(prim)
    pts = np.asarray(mesh.GetPointsAttr().Get(tc) or [], np.float64).reshape(-1, 3)
    tri = _triangles(mesh, tc)
    if str(mesh.GetOrientationAttr().Get() or 'rightHanded') == 'leftHanded':
        tri = tri[:, ::-1]
    return pts, tri


def _instances(prim, tc, depth=0):
    """(points, triangles) of every instance of a point instancer, in the instancer's frame
    (instancers inside its prototypes expanded too, a few levels deep)."""
    _, UsdGeom = _pxr()
    inst = UsdGeom.PointInstancer(prim)
    stage = prim.GetStage()
    xf = [np.array(m, np.float64).T for m in inst.ComputeInstanceTransformsAtTime(tc, tc)]
    idx = list(inst.GetProtoIndicesAttr().Get(tc) or [])
    protos = []
    for path in inst.GetPrototypesRel().GetTargets():
        root = stage.GetPrimAtPath(path)
        root_inv = np.linalg.inv(_world(root, tc))
        parts = []
        for g in _geometry(root):
            if g.GetTypeName() == 'PointInstancer':
                if depth >= 4:
                    continue
                pts, tri = _instances(g, tc, depth + 1)
            else:
                pts, tri = _local_geometry(g, tc)
            if not len(tri):
                continue
            m = root_inv @ _world(g, tc)
            if np.linalg.det(m[:3, :3]) < 0:
                tri = tri[:, ::-1]
            parts.append(((m[:3, :3] @ pts.T).T + m[:3, 3], tri))
        protos.append(parts)
    vs, ts, n = [], [], 0
    total = 0
    for i, (m, pi) in enumerate(zip(xf, idx)):
        if pi < 0 or pi >= len(protos):
            continue
        for pts, tri in protos[pi]:
            if total + len(tri) > MAX_TRIANGLES:
                log.warning('%s: more than %d triangles; the remaining instances are left out', prim.GetPath(), MAX_TRIANGLES)
                return _join(vs, ts)
            vs.append((m[:3, :3] @ pts.T).T + m[:3, 3])
            ts.append(tri[:, ::-1] + n if np.linalg.det(m[:3, :3]) < 0 else tri + n)
            n += len(pts)
            total += len(tri)
    return _join(vs, ts)


def _join(vs, ts):
    if not vs:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    return np.concatenate(vs), np.concatenate(ts)


def load_usd_mesh(path, prim_path, frame=None):
    """(vertices (n, 3) metres y-up, triangles (m, 3)) of a USD prim's geometry (meshes, shapes and
    point instancers under it), in its own frame or, for '...?world' sources, in world space at `frame`."""
    stage = open_stage(path)
    pp, world = _split(prim_path)
    prim = _prim(stage, pp)
    tc = time_code(stage, frame)
    parts = _geometry(prim)
    if not parts:
        raise USDError(f'{pp} has no mesh or shape')
    ax = axis_matrix(stage)
    root_inv = np.linalg.inv(_world(prim, tc))
    vs, ts, n = [], [], 0
    for g in parts:
        if not _shown(g, tc):
            continue
        pts, tri = _instances(g, tc) if g.GetTypeName() == 'PointInstancer' else _local_geometry(g, tc)
        if not len(pts) or not len(tri):
            continue
        m = _world(g, tc)
        if not world:
            m = root_inv @ m   # into the chosen prim's own frame
        # turned y-up and scaled to metres: for an object frame this is the object's shape in y-up
        # metres, which mesh_placement's transforms (see _yup) place in the world
        m = ax @ m
        v = (m[:3, :3] @ pts.T).T + m[:3, 3]
        if np.linalg.det(m[:3, :3]) < 0:
            tri = tri[:, ::-1]
        vs.append(v)
        ts.append(tri + n)
        n += len(v)
    if not vs:
        raise USDError(f'{pp} has no points')
    return np.concatenate(vs), np.concatenate(ts)


def usd_mesh_bounds(path, prim_path):
    """Bounding box over all frames of a prim's geometry (metres, y-up; world or own frame as the source says)."""
    stage = open_stage(path)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for f in _sample_frames(stage, prim_path):
        v, _ = load_usd_mesh(path, prim_path, f)
        lo, hi = np.minimum(lo, v.min(0)), np.maximum(hi, v.max(0))
    return lo, hi


def _sample_frames(stage, prim_path, limit=400):
    """Frames worth sampling for a prim: its range if it is animated, else just one."""
    if not usd_prim_animated_stage(stage, prim_path):
        return [None]
    r = frame_range(stage)
    if r is None:
        return [None]
    a, b = r
    step = max(1, (b - a) // limit)
    return list(range(a, b + 1, step)) + ([b] if (b - a) % step else [])


_SHAPE_ATTRS = ('points', 'size', 'radius', 'height', 'positions', 'orientations', 'scales', 'protoIndices', 'widths',
                'curveVertexCounts')


def _changes_shape(prim):
    """Whether a geometric prim's own geometry is animated (points, sizes, instance positions...)."""
    for name in _SHAPE_ATTRS:
        a = prim.GetAttribute(name)
        if a and a.GetNumTimeSamples() > 1:
            return True
    if prim.GetTypeName() == 'Volume':
        from pxr import UsdVol
        for path in dict(UsdVol.Volume(prim).GetFieldPaths()).values():
            f = prim.GetStage().GetPrimAtPath(path)
            for name in ('filePath', 'fieldName'):
                a = f.GetAttribute(name)
                if a and a.GetNumTimeSamples() > 1:
                    return True   # a VDB sequence
    if prim.GetTypeName() == 'PointInstancer':
        _, UsdGeom = _pxr()
        stage = prim.GetStage()
        for path in UsdGeom.PointInstancer(prim).GetPrototypesRel().GetTargets():
            for g in _geometry(stage.GetPrimAtPath(path)):
                if _changes_shape(g) or _xform_varies_below(g, stage.GetPrimAtPath(path)):
                    return True
    return False


def _xform_varies_below(prim, root):
    """Whether any transform from prim up to (not including) root is animated."""
    _, UsdGeom = _pxr()
    p = prim
    while p and p.IsValid() and p != root and not p.IsPseudoRoot():
        x = UsdGeom.Xformable(p)
        if x and x.TransformMightBeTimeVarying():
            return True
        p = p.GetParent()
    return False


def usd_prim_animated_stage(stage, prim_path):
    pp, world = _split(prim_path)
    prim = _prim(stage, pp)
    for g in _geometry(prim):
        if _changes_shape(g):
            return True
        if world and _xform_varies(g):
            return True
        if not world and g != prim and _xform_varies_below(g, prim):
            return True   # parts moving relative to each other deform the whole
    return False


def usd_prim_animated(path, prim_path):
    """Whether a mesh source changes over time: its geometry is animated, or (world sources) it moves."""
    return usd_prim_animated_stage(open_stage(path), prim_path)


# -- scanning and importing ----------------------------------------------------------------------------

@dataclass
class USDItem:
    path: str                 # prim path
    kind: str                 # 'camera', 'mesh' or 'light'
    name: str
    motion: str = 'static'    # mesh: static, rigid or world (see the module notes); camera: static or animated
    triangles: int = 0
    prim_type: str = ''       # Mesh, Cube, ..., PointInstancer; DistantLight, DomeLight, ..., MeshLight (MeshLightAPI)
    particles: bool = False   # Points: a particle system (points_are_particles), not still balls


@dataclass
class USDScene:
    file: str
    fps: float
    frames: tuple | None
    meters_per_unit: float
    up: str
    items: list = field(default_factory=list)

    def cameras(self):
        return [i for i in self.items if i.kind == 'camera']

    def meshes(self):
        return [i for i in self.items if i.kind == 'mesh']

    def lights(self):
        return [i for i in self.items if i.kind == 'light']


def _xform_varies(prim):
    _, UsdGeom = _pxr()
    p = prim
    while p and p.IsValid() and not p.IsPseudoRoot():
        x = UsdGeom.Xformable(p)
        if x and x.TransformMightBeTimeVarying():
            return True
        p = p.GetParent()
    return False


def decompose(m):
    """A 4x4 (y-up metres) as translation, rotation about y (degrees) and per-axis scale, or None if it
    turns about other axes too (or shears)."""
    a = m[:3, :3]
    scale = np.linalg.norm(a, axis=0)
    if np.any(scale < 1e-12):
        return None
    r = a / scale
    if np.linalg.det(r) < 0:
        return None
    if abs(r[1, 1] - 1.0) > 1e-4 or np.abs(r[1, [0, 2]]).max() > 1e-4 or np.abs(r[[0, 2], 1]).max() > 1e-4:
        return None
    yaw = math.degrees(math.atan2(r[0, 2], r[0, 0]))
    return m[:3, 3].copy(), yaw, scale


def _triangle_count(prim):
    _, UsdGeom = _pxr()
    t = prim.GetTypeName()
    if t == 'Mesh':
        counts = UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get() or []
        return int(sum(max(0, c - 2) for c in counts))
    if t in SHAPES:
        return {'Cube': 12}.get(t, 24 * 2 * 12)
    if t in CURVES:
        counts = UsdGeom.Curves(prim).GetCurveVertexCountsAttr().Get() or []
        return int(sum(max(0, c - 1) * 6 * 16 + 16 for c in counts))
    if t == 'Points':   # (a particle cache's points are time samples only)
        n = len(UsdGeom.Points(prim).GetPointsAttr().Get(_pxr()[0].TimeCode.EarliestTime()) or [])
        return n * _ball_faces(n)
    if t == 'Volume':
        return 10000   # not known until its VDB is read
    if t == 'PointInstancer':
        inst = UsdGeom.PointInstancer(prim)
        n = len(inst.GetProtoIndicesAttr().Get() or [])
        protos = inst.GetPrototypesRel().GetTargets()
        per = [sum(_triangle_count(g) for g in _geometry(prim.GetStage().GetPrimAtPath(p))) for p in protos]
        return int(n * (sum(per) / max(len(per), 1)))
    return 0


def scan(path):
    """What a USD file holds that Blackbody can use (cameras, objects, lights), and how each moves."""
    Usd, UsdGeom = _pxr()
    stage = open_stage(path)
    out = USDScene(file=str(path), fps=frame_rate(stage), frames=frame_range(stage),
                   meters_per_unit=float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0),
                   up=str(UsdGeom.GetStageUpAxis(stage)))
    ax = axis_matrix(stage)
    frames = _sample_frames_all(stage)
    it = iter(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)))
    for prim in it:
        t = prim.GetTypeName()
        if prim.IsA(UsdGeom.Imageable) and not _shown(prim):
            it.PruneChildren()
            continue
        if prim.IsA(UsdGeom.Camera):
            out.items.append(USDItem(str(prim.GetPath()), 'camera', prim.GetName(),
                                     'animated' if _xform_varies(prim) or _camera_varies(prim) else 'static', 0, t))
        elif t in GEOM or t == 'PointInstancer':
            if t == 'PointInstancer':
                it.PruneChildren()   # its prototypes are drawn only through it
                motion = 'world'
            elif _changes_shape(prim):
                motion = 'world'
            elif _xform_varies(prim):
                ok = all(decompose(_yup(ax, _world(prim, time_code(stage, f)))) is not None
                         for f in frames[::max(1, len(frames) // 24)])
                motion = 'rigid' if ok else 'world'
            else:
                motion = 'static' if decompose(_yup(ax, _world(prim, time_code(stage, None)))) is not None else 'world'
            out.items.append(USDItem(str(prim.GetPath()), 'mesh', prim.GetName(), motion, _triangle_count(prim), t,
                                     particles=t == 'Points' and points_are_particles(prim)))
            if t == 'Mesh' and _has_mesh_light(prim):   # (a mesh that gives off light: a light too)
                out.items.append(USDItem(str(prim.GetPath()), 'light', prim.GetName(),
                                         'animated' if _xform_varies(prim) else 'static', 0, 'MeshLight'))
        elif t.endswith('Light'):
            out.items.append(USDItem(str(prim.GetPath()), 'light', prim.GetName(),
                                     'animated' if _xform_varies(prim) else 'static', 0, t))
    return out


def _camera_varies(prim):
    _, UsdGeom = _pxr()
    cam = UsdGeom.Camera(prim)
    return cam.GetFocalLengthAttr().GetNumTimeSamples() > 1


def _has_mesh_light(prim):
    try:
        from pxr import UsdLux
        return prim.HasAPI(UsdLux.MeshLightAPI)
    except (ImportError, AttributeError):   # (a USD before MeshLightAPI)
        return False


def _sample_frames_all(stage):
    r = frame_range(stage)
    if r is None:
        return [None]
    return list(range(r[0], r[1] + 1))


def _yup(ax, m):
    """A stage-space transform expressed in y-up metres: ax @ m @ ax^-1, with the unit scale kept on
    the translation only (the object frame is in metres too)."""
    c = ax[:3, :3]
    turn = _turn(ax)
    out = np.eye(4)
    out[:3, :3] = turn @ m[:3, :3] @ turn.T
    out[:3, 3] = c @ m[:3, 3]
    return out


def camera_keys(path, prim_path, frames=None):
    """Keys for Blackbody's free camera from a USD camera: position (m), rotation (degrees, XYZ order),
    focal length (mm) per frame, and the sensor width (mm)."""
    _, UsdGeom = _pxr()
    from .chan import _to_xyz_euler
    stage = open_stage(path)
    prim = _prim(stage, prim_path)
    cam = UsdGeom.Camera(prim)
    ax = axis_matrix(stage)
    frames = frames or _sample_frames_all(stage)
    pos, rot, focal = [], [], []
    prev = None
    turn = _turn(ax)
    for f in frames:
        tc = time_code(stage, f)
        w = _world(prim, tc)
        # the camera looks down its own -z with +y up whatever the stage's up axis: turn only the world
        r = turn @ w[:3, :3]
        r = r / np.linalg.norm(r, axis=0)
        e = np.array(_to_xyz_euler(r))
        if prev is not None:  # continuous Euler angles, so interpolation does not spin the long way
            e = prev + ((e - prev + 180.0) % 360.0 - 180.0)
        prev = e
        key = 1 if f is None else f
        pos.append([key, tuple(float(x) for x in ax[:3, :3] @ w[:3, 3]), 'linear'])
        rot.append([key, tuple(float(x) for x in e), 'linear'])
        focal.append([key, float(cam.GetFocalLengthAttr().Get(tc) or 50.0), 'linear'])
    tc0 = time_code(stage, frames[0])
    sensor = float(cam.GetHorizontalApertureAttr().Get(tc0) or 36.0)
    clip = cam.GetClippingRangeAttr().Get(tc0)
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    near_far = (float(clip[0]) * mpu, float(clip[1]) * mpu) if clip else None
    return pos, rot, focal, sensor, near_far


def mesh_placement(path, prim_path, frames=None):
    """For a rigid or static object: per frame (translation (m), yaw (degrees), scale) of its frame in
    the world, or None if it cannot be expressed that way."""
    stage = open_stage(path)
    prim = _prim(stage, prim_path)
    ax = axis_matrix(stage)
    frames = frames or _sample_frames_all(stage)
    out = []
    prev = None
    for f in frames:
        d = decompose(_yup(ax, _world(prim, time_code(stage, f))))
        if d is None:
            return None
        t, yaw, s = d
        if prev is not None:
            yaw = prev + ((yaw - prev + 180.0) % 360.0 - 180.0)
        prev = yaw
        out.append((1 if f is None else f, t, yaw, s))
    return out


def analytic_shape(path, prim_path, as_='collider'):
    """(shape, size in its own y-up metre frame) for a Cube, Sphere or Cylinder prim Blackbody can
    represent exactly (before the placement's scale), or None."""
    _, UsdGeom = _pxr()
    stage = open_stage(path)
    prim = _prim(stage, prim_path)
    t = prim.GetTypeName()
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    tc = time_code(stage, None)
    if t == 'Cube':
        h = 0.5 * float(UsdGeom.Cube(prim).GetSizeAttr().Get(tc) or 2.0) * mpu
        return 'box', np.array([h, h, h])
    if t == 'Sphere':
        r = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get(tc) or 1.0) * mpu
        return 'sphere', np.array([r, r, r])
    if t == 'Cylinder':
        cyl = UsdGeom.Cylinder(prim)
        axis = {'X': (1.0, 0.0, 0.0), 'Y': (0.0, 1.0, 0.0), 'Z': (0.0, 0.0, 1.0)}[str(cyl.GetAxisAttr().Get() or 'Z').upper()]
        up = _turn(axis_matrix(stage)) @ np.array(axis)
        if abs(abs(up[1]) - 1.0) > 1e-6:
            return None   # lying on its side: Blackbody's cylinders stand upright
        r = float(cyl.GetRadiusAttr().Get(tc) or 1.0) * mpu
        hh = 0.5 * float(cyl.GetHeightAttr().Get(tc) or 2.0) * mpu
        return 'cylinder', np.array([r, hh, r])
    return None


def _analytic_size(shape, base, scale, as_):
    """The collider or emitter size for an analytic shape under a per-axis scale, or None."""
    s = np.asarray(scale, float)
    if shape == 'box':
        return base * s
    if shape == 'sphere':
        if as_ == 'emitter':
            return base * s              # emitter spheres are ellipsoids
        if np.ptp(s) > 1e-3 * s.max():
            return None                  # a squashed sphere collider: keep it a mesh
        return base * s
    if shape == 'cylinder':
        if abs(s[0] - s[2]) > 1e-3 * max(s[0], s[2]):
            return None
        return base * s
    return None


def _light_colour(api, tc):
    c = np.asarray(api.GetColorAttr().Get(tc) or (1.0, 1.0, 1.0), float)
    try:
        if api.GetEnableColorTemperatureAttr().Get(tc):
            from ..engine.lut import blackbody_lut
            k = float(api.GetColorTemperatureAttr().Get(tc) or 6500.0)
            lut = blackbody_lut(8, max(1000.0, k - 1.0), k + 1.0)
            bb = lut[4, :3] / max(lut[4, :3].max(), 1e-9)
            c = c * bb
    except Exception:
        pass
    return c


def light_keys(path, prim_path, frames):
    """For a distant light: per frame (azimuth, elevation) in degrees toward the light, and its colour
    and intensity at the first frame."""
    from pxr import UsdLux
    stage = open_stage(path)
    prim = _prim(stage, prim_path)
    ax = axis_matrix(stage)
    turn = _turn(ax)
    keys = []
    for f in frames:
        w = _world(prim, time_code(stage, f))
        # a distant light shines down its own -z: the light is toward +z
        d = turn @ w[:3, :3] @ np.array([0.0, 0.0, 1.0])
        d = d / max(np.linalg.norm(d), 1e-12)
        keys.append((1 if f is None else f, math.degrees(math.atan2(d[0], d[2])),
                     math.degrees(math.asin(max(-1.0, min(1.0, d[1]))))))
    tc = time_code(stage, frames[0])
    api = UsdLux.DistantLight(prim)
    intensity = float(api.GetIntensityAttr().Get(tc) or 1.0) * 2.0 ** float(api.GetExposureAttr().Get(tc) or 0.0)
    return keys, _light_colour(api, tc), intensity


def _shifted(keys, offset):
    """Keys [frame, value, interpolation] moved `offset` frames earlier (USD frames to Blackbody frames)."""
    return [[k[0] - offset] + list(k[1:]) for k in keys] if offset else keys


def adopt_range(scene, info, offset=0, keep_length=False):
    """Take the stage's frame range (USD frames, less `offset`) and frame rate for the scene; with keep_length (the
    footage sets how long the shot is and its rate), only where it starts. Returns a report line, or None when the
    stage has no range."""
    if not info.frames:
        return None
    a, b = int(info.frames[0]) - int(offset), int(info.frames[1]) - int(offset)
    r = scene.data['render']
    if keep_length:
        n = int(r['end']) - int(r['start'])
        r['start'], r['end'] = a, a + n
        return f'frames from {a} (USD {a + offset}), as long as the footage'
    r['start'], r['end'], r['fps'] = a, b, float(info.fps)
    return f'frames {a}-{b} at {info.fps:g} fps' + (f' (USD {a + offset}-{b + offset})' if offset else '')


def import_usd(scene, path, cameras=None, meshes=None, as_='collider', holdout=True, burnable=False, frames=None,
               lights=True, volumes='smoke', offset=0, match_range=False):
    """Add a USD file's camera, objects and lights to a scene. `cameras` / `meshes`: prim paths to
    import (None: the first camera and every object). Objects become colliders (or emitters with
    as_='emitter'), placed in the fire's frame: the fire's position and rotation are taken into
    account, so they land where they are in the USD scene. Volume prims, and Points prims that are
    particle systems, become smoke (Volume emitters that fill the box, or keep it topped up when they
    change; in a liquid scene, liquid sources that fill once) with volumes='smoke', or solid objects
    like the meshes with volumes='solid'. `offset`: USD frame F becomes frame F - offset. `match_range`:
    take the stage's frame range and rate for the scene (adopt_range; 'start' keeps the scene's length
    and rate). Returns a short report."""
    from ..engine import camera as cam_mod
    from ..scene.anim import Curve
    from ..scene.params import FRAME_OFFSET_MAX, collider_defaults, emitter_defaults
    offset = int(offset or 0)
    if abs(offset) > FRAME_OFFSET_MAX:   # (what changes is read through its Mesh frame offset, which goes no further)
        raise USDError(f'Frame offset {offset} is out of range (at most {FRAME_OFFSET_MAX:.0f} frames either way)')
    info = scan(path)
    rel = _relative(scene, path)
    report = []
    all_frames = list(range(info.frames[0], info.frames[1] + 1)) if info.frames else None
    if frames is not None:
        all_frames = list(frames)
    if match_range:
        line = adopt_range(scene, info, offset, keep_length=match_range == 'start')
        if line:
            report.append(line)
    cams = [c.path for c in info.cameras()] if cameras is None else list(cameras)
    if cameras is None:
        cams = cams[:1]
    for cp in cams:
        pos, rot, focal, sensor, near_far = camera_keys(path, cp, all_frames)
        pos, rot, focal = (_shifted(k, offset) for k in (pos, rot, focal))
        c = scene.data['camera']
        c['mode'] = 'free'
        c['use_anchor'] = False
        c['position'] = Curve(pos) if len(pos) > 1 else pos[0][1]
        c['rotation'] = Curve(rot) if len(rot) > 1 else rot[0][1]
        fvals = {v for _, v, _ in focal}
        c['focal_mm'] = Curve(focal) if len(fvals) > 1 else focal[0][1]
        c['sensor_mm'] = sensor
        if near_far:
            c['near'], c['far'] = max(near_far[0], 1e-3), max(near_far[1], near_far[0] * 10)
        report.append(f'camera {cp}')
    if lights:
        report += _import_lights(scene, path, info, all_frames, offset)
    # the fire's frame (fire-local = rot_y(fire_yaw)^T (world - fire_position)), at the first frame
    f0 = all_frames[0] if all_frames else scene.start + offset
    fy = scene.v('camera', 'fire_yaw', f0 - offset)
    fp = np.asarray(scene.v('camera', 'fire_position', f0 - offset), float)
    R = cam_mod.rot_y(math.radians(fy))
    todo = [m for m in info.meshes() if (meshes is None or m.path in meshes) and m.triangles > 0]
    limit = 16
    for m in todo:
        particles = m.prim_type == 'Points' and m.particles
        smoke = (m.prim_type == 'Volume' or particles) and volumes == 'smoke'
        target = scene.emitters if (smoke or as_ != 'collider') else scene.colliders
        if len(target) >= limit:
            report.append(f'skipped {m.path}: at most {limit}')
            continue
        if smoke:
            d = emitter_defaults()
            animated_file = _changes_shape(_prim(open_stage(path), m.path))
            d.update(name=m.name, shape='volume', enabled=True, fuel=0.0, temperature=0.0, smoke=4.0, noise=0.0,
                     start=0.0, embers=False, volume_mode='hold' if animated_file else 'fill')
            if scene.kind == 'liquid':   # (water where it is dense, poured once: a liquid sim's particles carried on)
                d.update(liquid_mode='fill')
        else:
            d = collider_defaults() if as_ == 'collider' else emitter_defaults()
            d.update(name=m.name, shape='mesh', enabled=True)
            if as_ == 'collider':
                d.update(holdout=bool(holdout), burnable=bool(burnable))
        if offset:
            d['mesh_offset'] = float(-offset)   # (what changes is read at the USD's own frame)
        key = 'volume' if smoke else 'mesh'
        what_smoke = ('particles as smoke' if particles else 'smoke') if scene.kind != 'liquid' else \
            ('particles as water' if particles else 'water')
        place = mesh_placement(path, m.path, all_frames if m.motion == 'rigid' else [f0]) if m.motion != 'world' else None
        if place is not None:
            place = [(f - offset, t, yaw, s) for f, t, yaw, s in place]
        exact = analytic_shape(path, m.path, as_) if (place is not None and m.prim_type in SHAPES) else None
        if exact is not None and any(_analytic_size(exact[0], exact[1], s, as_) is None for _, _, _, s in place):
            exact = None
        if place is None:
            # world space, baked frame by frame: the collider's frame is the world's
            d[key] = f'{rel}#{m.path}{WORLD}'
            d['position'] = tuple(float(x) for x in -(R.T @ fp))
            d['yaw'] = -fy
            d['size'] = (1.0, 1.0, 1.0)
            what = 'instances' if m.prim_type == 'PointInstancer' else 'deforming'
            report.append(f'{m.path}: {what_smoke} ({d["volume_mode"]})' if smoke else
                          f'{m.path}: {what} (baked per frame)' if m.motion == 'world' else f'{m.path}: baked')
        else:
            if exact is not None:
                d['shape'] = exact[0]
                d['mesh'] = ''
                sizes = [_analytic_size(exact[0], exact[1], s, as_) for _, _, _, s in place]
            else:
                d[key] = f'{rel}#{m.path}'
                sizes = [np.asarray(s, float) for _, _, _, s in place]
            keys_p, keys_y, keys_s = [], [], []
            for (f, t, yaw, _), size in zip(place, sizes):
                keys_p.append([f, tuple(float(x) for x in R.T @ (t - fp)), 'linear'])
                keys_y.append([f, float(yaw - fy), 'linear'])
                keys_s.append([f, tuple(float(x) for x in size), 'linear'])
            moves = len(place) > 1 and m.motion == 'rigid'
            d['position'] = Curve(keys_p) if moves else keys_p[0][1]
            d['yaw'] = Curve(keys_y) if moves else keys_y[0][1]
            scales = moves and np.ptp(np.array([k[1] for k in keys_s]), axis=0).max() > 1e-6
            d['size'] = Curve(keys_s) if scales else keys_s[0][1]
            report.append(f'{m.path}: {what_smoke} ({d["volume_mode"]}, {m.motion})' if smoke else
                          f'{m.path}: {m.motion}' + (f' {exact[0]}' if exact is not None else '')
                          + (' balls' if m.prim_type == 'Points' else ''))
        target.append(d)
    if info.frames and frames is None and not match_range:
        report.append(f'frames {info.frames[0]}-{info.frames[1]} at {info.fps:g} fps'
                      + (f' (frames {info.frames[0] - offset}-{info.frames[1] - offset} here)' if offset else ''))
    return report


def _import_lights(scene, path, info, frames, offset=0):
    """The first distant light as the key light, the first dome light as the ambient light; local, portal and
    geometry lights as lights in the set."""
    from pxr import UsdLux
    from ..scene.anim import Curve
    report = []
    lights = info.lights()
    L = scene.data['lighting']
    distant = [l for l in lights if l.prim_type == 'DistantLight']
    dome = [l for l in lights if l.prim_type == 'DomeLight']
    if distant:
        l = distant[0]
        keys, colour, intensity = light_keys(path, l.path, frames if (frames and l.motion == 'animated') else [frames[0] if frames else None])
        keys = [(f - offset, az, el) for f, az, el in keys] if len(keys) > 1 else keys
        # lux-like intensities (USD's default is 50 000) against Blackbody's relative key intensity
        k = 3.0 * intensity / LUX_REFERENCE if intensity > 100.0 else intensity
        L['sun_on'] = True
        L['sun_intensity'] = float(min(max(k, 0.0), 20.0))
        L['sun_color'] = tuple(float(x) for x in colour / max(colour.max(), 1e-9))
        if len(keys) > 1:
            L['sun_azimuth'] = Curve([[f, az, 'linear'] for f, az, _ in keys])
            L['sun_elevation'] = Curve([[f, el, 'linear'] for f, _, el in keys])
        else:
            L['sun_azimuth'], L['sun_elevation'] = float(keys[0][1]), float(keys[0][2])
        report.append(f'key light {l.path}')
    if dome:
        l = dome[0]
        stage = open_stage(path)
        prim = _prim(stage, l.path)
        api = UsdLux.DomeLight(prim)
        tc = time_code(stage, frames[0] if frames else None)
        intensity = float(api.GetIntensityAttr().Get(tc) or 1.0) * 2.0 ** float(api.GetExposureAttr().Get(tc) or 0.0)
        colour = _light_colour(api, tc)
        L['ambient'] = tuple(float(x) for x in colour / max(colour.max(), 1e-9) * 0.18)
        L['ambient_intensity'] = float(min(max(intensity, 0.0), 10.0))
        tex = api.GetTextureFileAttr().Get(tc)
        p = ''
        if tex is not None and (tex.resolvedPath or tex.path):
            p = tex.resolvedPath or str(Path(stage.GetRootLayer().realPath).parent / tex.path)
        if p:
            if Path(p).exists():
                L['environment'] = p
                L['env_strength'] = float(min(max(intensity, 0.01), 10.0))
                yaw = decompose(_yup(axis_matrix(stage), _world(prim, tc)))
                if yaw is not None:
                    L['env_rotation'] = float(((yaw[1] + 180.0) % 360.0) - 180.0)
        report.append(f'ambient light {l.path}')
    local = [l for l in lights if l.prim_type in LOCAL_LIGHTS + ('PortalLight', 'GeometryLight', 'MeshLight')]
    if local:
        report += _import_local_lights(scene, path, local, frames, offset, dome[0].path if dome else None)
    others = [l for l in lights if l not in local and l.prim_type not in ('DistantLight', 'DomeLight')] + distant[1:] + dome[1:]
    if others:
        names = ', '.join(sorted({l.prim_type for l in others}))
        report.append(f'{len(others)} light(s) not imported ({names}): Blackbody has one key light and one ambient light')
    return report


def _radiance(api, tc):
    """A UsdLux light's radiance (intensity times 2^exposure) and colour."""
    return (float(api.GetIntensityAttr().Get(tc) or 1.0) * 2.0 ** float(api.GetExposureAttr().Get(tc) or 0.0),
            _light_colour(api, tc))


def _glowing(path, prim, frame):
    """What a geometry light gives off from: (area (m^2), its centre and its half extent (m, y-up world)) of the
    geometry it names (GeometryLight) or is (a mesh with MeshLightAPI), at a frame."""
    from pxr import UsdLux
    targets = [prim.GetPath()] if prim.GetTypeName() != 'GeometryLight' else \
        list(UsdLux.GeometryLight(prim).GetGeometryRel().GetTargets())
    area, centre, lo, hi = 0.0, np.zeros(3), np.full(3, np.inf), np.full(3, -np.inf)
    for t in targets:
        try:
            v, tri = load_usd_mesh(path, str(t) + WORLD, frame)
        except USDError:
            continue
        p = v[tri]
        a = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
        if a.sum() <= 0:
            continue
        centre = centre + (a[:, None] * p.mean(1)).sum(0)
        area += float(a.sum())
        lo, hi = np.minimum(lo, v.min(0)), np.maximum(hi, v.max(0))
    if area <= 0:
        return None
    return area, centre / area, 0.5 * (hi - lo)


def _import_local_lights(scene, path, lights, frames, offset=0, dome=None):
    """Sphere, disk, rect and cylinder lights as lights in the set, placed in the fire's frame. Their
    intensity (radiance, nits) times their area gives the illuminance one metre in front (lux at 1 m).
    A light with an IES profile shines through it (a disc or rect light as a point light with it, as the
    profile is the whole fixture's spread), and one whose shadows are off (ShadowAPI) casts no smoke
    shadows. A portal is a panel giving off the dome light's radiance (`dome`: its prim) into the room;
    a geometry light a point light at its geometry's centre, as bright as its surface (a quarter of its
    area, as a ball's outline is), as big as it is."""
    from pxr import UsdLux
    from ..engine import camera as cam_mod
    from ..scene.anim import Curve
    stage = open_stage(path)
    ax = axis_matrix(stage)
    turn = _turn(ax)
    mpu = float(__import__('pxr').UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    f0 = frames[0] if frames else scene.start + offset
    fy = scene.v('camera', 'fire_yaw', f0 - offset)
    fp = np.asarray(scene.v('camera', 'fire_position', f0 - offset), float)
    R = cam_mod.rot_y(math.radians(fy))
    report = []
    for l in lights:
        if len(scene.lights) >= 8:
            report.append(f'skipped {l.path}: at most 8 lights')
            continue
        prim = _prim(stage, l.path)
        tc = time_code(stage, f0)
        t = l.prim_type
        api = UsdLux.LightAPI(prim) if t == 'MeshLight' else getattr(UsdLux, t)(prim)
        radiance, colour = _radiance(api, tc)
        kind = 'point'
        pw = ph = None
        notes = []
        glow = None
        if t == 'SphereLight':
            r = float(api.GetRadiusAttr().Get(tc) or 0.5) * mpu
            area, size = math.pi * r * r, r
        elif t == 'DiskLight':
            r = float(api.GetRadiusAttr().Get(tc) or 0.5) * mpu
            area, size, kind = math.pi * r * r, r, 'area'
            pw = ph = r * math.sqrt(math.pi)
        elif t in ('RectLight', 'PortalLight'):
            w = float(api.GetWidthAttr().Get(tc) or 1.0) * mpu
            h = float(api.GetHeightAttr().Get(tc) or 1.0) * mpu
            area, size, kind = w * h, 0.5 * math.hypot(w, h), 'area'
            pw, ph = w, h
            if t == 'PortalLight':
                if dome is not None:   # (the dome's light, let in through the portal: times the portal's own)
                    rd, cd = _radiance(UsdLux.DomeLight(_prim(stage, dome)), tc)
                    radiance, colour = radiance * rd, colour * cd
                    notes.append('the dome light through it')
            elif t == 'RectLight' and api.GetTextureFileAttr().Get(tc):
                notes.append('its texture is not used')
        elif t in ('GeometryLight', 'MeshLight'):
            glow = _glowing(path, prim, f0)
            if glow is None:
                report.append(f'light {l.path} not imported: it names no geometry')
                continue
            area, size = 0.25 * glow[0], float(np.max(glow[2]))
            notes.append(f'from its {glow[0]:.3g} m² of surface')
        else:
            r = float(api.GetRadiusAttr().Get(tc) or 0.5) * mpu
            ln = float(api.GetLengthAttr().Get(tc) or 1.0) * mpu
            area, size = 2.0 * r * ln, 0.5 * ln
        cone, softness, ies_file = 90.0, 0.0, ''
        if prim.HasAPI(UsdLux.ShapingAPI):
            sh = UsdLux.ShapingAPI(prim)
            cone = float(sh.GetShapingConeAngleAttr().Get(tc) or 90.0)
            softness = float(sh.GetShapingConeSoftnessAttr().Get(tc) or 0.0)
            if cone < 89.0 and t not in ('RectLight', 'PortalLight'):   # (a disc or bulb with a cone: a spot)
                kind = 'spot'
            f = sh.GetShapingIesFileAttr().Get(tc)
            if f:
                p = getattr(f, 'resolvedPath', '') or getattr(f, 'path', '') or str(f)
                ies_file = str((Path(path).parent / p).resolve()) if p and not Path(p).is_absolute() else str(p)
        if ies_file and kind == 'area':
            kind = 'point'   # (the profile is how the whole fixture spreads its light: Blackbody's panels take none)
            notes.append('a point light through its IES profile')
        keys_p, keys_d = [], []
        spin = 0.0
        for f in (frames if (frames and l.motion == 'animated') else [f0]):
            m = _world(prim, time_code(stage, f))
            pos = ax[:3, :3] @ m[:3, 3]
            if glow is not None:
                g = glow if f == f0 else (_glowing(path, prim, f) or glow)
                pos = g[1]
            aim = turn @ m[:3, :3] @ np.array([0.0, 0.0, -1.0])   # lights shine down their own -z
            if f in (f0, None) and pw is not None:
                # the panel's width along the light's own x: the Turn from the axes Blackbody takes round its aim
                from .ies import frame as lamp_frame
                xw = R.T @ (turn @ m[:3, :3] @ np.array([1.0, 0.0, 0.0]))
                t0, t1 = lamp_frame(R.T @ (aim / max(np.linalg.norm(aim), 1e-12)))
                spin = math.degrees(math.atan2(float(xw @ t1), float(xw @ t0)))
            aim = aim / max(np.linalg.norm(aim), 1e-12)
            keys_p.append([1 if f is None else f - offset, tuple(float(x) for x in R.T @ (pos - fp)), 'linear'])
            keys_d.append([1 if f is None else f - offset, tuple(float(x) for x in R.T @ aim), 'linear'])
        scene.add_light(name=l.name, kind=kind, colour=tuple(float(x) for x in colour / max(colour.max(), 1e-9)),
                        intensity=radiance * area * float(colour.max()), radius=min(max(size, 0.01), 5.0),
                        cone=min(max(cone, 1.0), 90.0), softness=min(max(softness, 0.0), 1.0))
        d = scene.lights[-1]
        if pw is not None:
            d['width'], d['height'], d['spin'] = max(pw, 0.01), max(ph, 0.01), round(spin, 2)
        if ies_file:
            # (its profile as the shape only: the light's own intensity stays, as USD scales the file's by it)
            d['profile'], d['profile_brightness'] = _relative(scene, ies_file), False
        if prim.HasAPI(UsdLux.ShadowAPI) and UsdLux.ShadowAPI(prim).GetShadowEnableAttr().Get(tc) is False:
            d['shadows'] = False
            notes.append('no shadows')
        d['position'] = Curve(keys_p) if len(keys_p) > 1 else keys_p[0][1]
        d['direction'] = Curve(keys_d) if len(keys_d) > 1 else keys_d[0][1]
        report.append(f'light {l.path} ({kind}' + ''.join(f'; {n}' for n in notes) + ')')
    return report


def _relative(scene, path):
    """The USD file's path relative to the project (absolute for unsaved scenes)."""
    p = Path(path).resolve()
    if scene.path:
        try:
            return os.path.relpath(p, Path(scene.path).resolve().parent).replace('\\', '/')
        except ValueError:
            pass
    return str(p)
