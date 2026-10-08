"""The shot as a USD scene, for lighting and rendering it in Houdini (Solaris), Blender, Maya, Nuke, Unreal or Omniverse
next to the elements Blackbody renders: its camera, every object as it falls, floats, is keyed and breaks, the ropes,
the sand, snow, mud, jelly and clay, the grass and the embers, frame by frame.

Layout (the scene's metres, y up; a time code is a frame of the shot, at its frame rate):
  /World/Camera               the shot's camera (io/camera_out.py: its picture lines up with Blackbody's exactly)
  /World/Objects/<name>       every enabled object: an Xform moved frame by frame (hidden once it has broken), its
                              shape under it as Shape: a Cube, Sphere or Cylinder, or the mesh, scaled to its size
  /World/Pieces/<name>/Piece_0001 ...
                              a broken object's pieces (and a person's or a car's parts), each a mesh in its own frame,
                              moved frame by frame; hidden until it breaks. Their cut faces are the GeomSubset 'cut'
  /World/Ropes/<name>         ropes, cables, chains and springs as curves
  /World/Matter               sand, snow, mud, jelly and clay as points: ids, velocities (m/s, for motion blur), widths
                              (a sphere of the matter each stands for), displayColor, and primvars material (its slot in
                              blackbody:materials) and temperature (K, when it heats)
  /World/Grass/<name>         every blade of a patch as a curve of 5 points: widths, displayColor, primvars burn (0
                              unburnt, 0-1 burning, 1-2 the stubble cooling) and burning (1/0)
  /World/Embers               embers and sparks as points: velocities, widths, displayColor (their glow's colour) and
                              primvars temperature (K)
  /Render/Settings            the camera and the picture's size
A shot with layers puts every other layer's under /World/<layer> (and its camera there, if it is not the shot's).

Objects, pieces and the camera move by a translate and an orient (a quaternion), not a matrix: USD interpolates a
quaternion by slerp, so a piece spinning fast keeps its shape at the subframes motion blur samples (a matrix's entries
are interpolated one by one, which shrinks and shears it). A prim that hides (an object broken, a piece burnt to ash)
stays where it was last seen: USD would otherwise slide it, still shown, toward the 'gone' pose far below through the
subframes before it hides.

The points and curves are big (a million grains a frame), so they are written a frame at a time into their own files
(<name>_points/<name>.####.usdc), which the scene reads as value clips: any program reading USD sees them as animated
points, and the writer holds one frame in memory, not the shot. The rest is in the one file.
"""
from __future__ import annotations

import logging
import math
import re
from pathlib import Path

import numpy as np

from ..engine import camera as cam

log = logging.getLogger('blackbody.usd')

MATTER_WIDTH = (6.0 / math.pi) ** (1.0 / 3.0) * 0.5   # node spacings: a sphere of the matter one particle stands for
                                                       # (a cube of half a node spacing a side: matter.PER_AXIS)
GRASS_TAPER = 2.6       # a blade is full width most of the way, to a point at its tip (strand_draw.wgsl)
CHAR = (0.025, 0.021, 0.018)   # burnt grass's colour (strand_draw.wgsl)


def prim_name(name, taken, fallback='Item'):
    """A valid USD prim name from `name`, unique among `taken` (which it is added to)."""
    s = re.sub(r'[^A-Za-z0-9_]+', '_', str(name or '')).strip('_') or fallback
    if not (s[0].isalpha() or s[0] == '_'):
        s = '_' + s
    out, k = s, 2
    while out in taken:
        out = f'{s}_{k}'
        k += 1
    taken.add(out)
    return out


def quat_matrices(q):
    """(n, 3, 3) rotations of quaternions (n, 4) x y z w."""
    q = np.asarray(q, float).reshape(-1, 4)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    x, y, z, w = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def matrix_quats(R):
    """Quaternions (n, 4) x y z w of rotations (n, 3, 3): quat_matrices' inverse. (Shepperd's: 4 q q^T is read off
    the matrix, and q from its row of the largest of w, x, y, z, so it is exact at a half turn too.)"""
    R = np.asarray(R, float).reshape(-1, 3, 3)
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    ww, xx, yy, zz = 1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22
    wx, wy, wz = R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]
    xy, xz, yz = R[:, 0, 1] + R[:, 1, 0], R[:, 0, 2] + R[:, 2, 0], R[:, 1, 2] + R[:, 2, 1]
    P = np.stack([np.stack(r, -1) for r in ((ww, wx, wy, wz), (wx, xx, xy, xz), (wy, xy, yy, yz), (wz, xz, yz, zz))], 1)
    i = np.arange(len(R))
    k = np.stack([ww, xx, yy, zz], 1).argmax(1)
    q = P[i, k] / (2.0 * np.sqrt(np.maximum(P[i, k, k], 1e-30)))[:, None]
    return q[:, [1, 2, 3, 0]]


def rigid_poses(M, prev=None):
    """Rigid 4x4s (n, 4, 4: column vectors) as translations (n, 3) and quaternions (n, 4) x y z w, each on the same
    side as `prev`'s (n, 4) where given (q and -q are the same turn: this way the slerp between frames takes the short
    way round)."""
    M = np.asarray(M, float).reshape(-1, 4, 4)
    R = M[:, :3, :3] / np.maximum(np.linalg.norm(M[:, :3, :3], axis=1, keepdims=True), 1e-12)
    q = matrix_quats(R)
    if prev is not None:
        q[np.einsum('ij,ij->i', q, np.asarray(prev, float).reshape(-1, 4)) < 0.0] *= -1.0
    return M[:, :3, 3].copy(), q


def fire_matrix(scene, frame):
    """The simulation's frame (fire-local metres) to the world's at `frame`, 4x4."""
    return scene.camera(frame)[1].local_to_world()


def _to_world(F, p):
    """Fire-local points to the world's, as float32 (USD's points are)."""
    F = np.asarray(F, np.float32)
    return np.asarray(p, np.float32).reshape(-1, 3) @ F[:3, :3].T + F[:3, 3]


# -- what a frame holds (from an engine: live or from its cache) ----------------------------------------------------

def objects_at(eng, scene, frame):
    """Every enabled object at `frame`: [dict(index, name, shape, matrix (its own frame -> world, 4x4, rigid), scale
    (3,) of a unit shape (a Cube of size 2, a Sphere or a Cylinder of radius 1 and height 2 along y, or the mesh as
    modelled), mesh (its file), mesh_frame (a deforming mesh's frame, else None), shown, guide (it hides nothing: a
    helper), colour (linear))]. Falling, floating and broken ones are where the simulation has them (a broken one is not
    shown: its pieces are). A hollow one, or one with an opening, is its outer shape (job.output_notes says so)."""
    from ..engine.solver import MAX_COLLIDERS
    from ..scene.materials import MATERIALS
    from ..scene.model import quat_matrix
    over = eng.floating_overrides(frame) if hasattr(eng, 'floating_overrides') else None
    cols = scene.colliders_gpu(frame, over)
    idx = [i for i, c in enumerate(scene.colliders) if c['enabled']][:MAX_COLLIDERS]
    F = fire_matrix(scene, frame)
    out = []
    for i, cg in zip(idx, cols):
        c = scene.colliders[i]
        M = np.eye(4)
        M[:3, :3] = quat_matrix(cg.quat) @ cam.rot_y(float(cg.rot_y))
        M[:3, 3] = cg.pos
        s = np.abs(np.asarray(cg.size, float))
        scale = {'sphere': (s[0], s[0], s[0]), 'cylinder': (s[0], s[1], s[0])}.get(cg.shape, tuple(s))
        mat = MATERIALS.get(c.get('material'))
        colour = tuple(c['colour']) if c.get('own_colour') else (mat.colour if mat is not None else (0.6, 0.6, 0.6))
        out.append(dict(index=i, name=c.get('name') or f'Object {i + 1}', shape=cg.shape, matrix=F @ M,
                        scale=np.asarray(scale, float), mesh=cg.mesh, mesh_frame=cg.mesh_frame, shown=cg.pos[1] > -1.0e3,
                        guide=not c.get('holdout', True), colour=tuple(float(x) for x in colour)))
    return out


def pieces_at(eng, scene, frame):
    """The broken objects' pieces at `frame`: {collider index: dict(name, frac (fracture.Fracture: their shapes), n,
    matrices (n, 4, 4): each piece's own frame (centred on its centroid) -> world, shown (n,): not burnt to ash, colour,
    inside: the cut faces' colour)}."""
    from ..engine.solids import fractured
    from ..scene.materials import MATERIALS
    poses = eng.piece_poses(frame) if hasattr(eng, 'piece_poses') else None
    F = fire_matrix(scene, frame)
    out = {}
    for ci, pose in (poses or {}).items():
        ci = int(ci)
        if ci >= len(scene.colliders):
            continue
        c = scene.colliders[ci]
        hollow = float(np.asarray(pose.get('hollow', 0.0), float).ravel()[0])
        frac = fractured(c, pose['size'], hollow, pose.get('impact'), scene)
        pos = np.asarray(pose['pos'], float).reshape(-1, 3)
        n = min(len(frac.pieces), len(pos))
        M = np.tile(np.eye(4), (n, 1, 1))
        M[:, :3, :3] = quat_matrices(np.asarray(pose['quat'], float).reshape(-1, 4)[:n])
        M[:, :3, 3] = pos[:n]
        mat = MATERIALS.get(c.get('material'))
        colour = tuple(c['colour']) if c.get('own_colour') else (mat.colour if mat is not None else (0.6, 0.6, 0.6))
        inside = (mat.inside if mat is not None and mat.inside is not None else colour)
        out[ci] = dict(name=c.get('name') or f'Object {ci + 1}', frac=frac, n=n, matrices=F @ M,
                       shown=pos[:n, 1] > -1.0e3, colour=tuple(float(x) for x in colour),
                       inside=tuple(float(x) for x in inside))
    return out


def piece_mesh(pc):
    """A piece's surface in its own frame (from its centroid): (points (m, 3), face vertex counts, indices, the cut
    faces' indices). Each face is its own polygon (flat, its corners not shared)."""
    polys = [np.asarray(p, float) for p in (pc.face_poly or [])]
    if not polys:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int32)
    pts = (np.concatenate(polys) - np.asarray(pc.centroid, float)).astype(np.float32)
    counts = np.array([len(p) for p in polys], np.int32)
    inner = np.asarray(pc.inner, bool)
    cut = np.array([k for k in range(len(polys)) if k < len(inner) and inner[k]], np.int32)
    return pts, counts, np.arange(len(pts), dtype=np.int32), cut


def ropes_at(eng, scene, frame):
    """The ropes, cables, chains and springs at `frame`: {collider index: dict(name, points (m, 3) world, width (m))}."""
    from ..engine.ropes import num, rope_points
    ropes = eng.rope_poses(frame) if hasattr(eng, 'rope_poses') else None
    F = fire_matrix(scene, frame)
    ground = 0.0 if scene.data['domain'].get('ground', True) else None
    out = {}
    for ci, rope in (ropes or {}).items():
        ci = int(ci)
        pts, _vel = rope_points(rope, ground=ground)
        name = scene.colliders[ci].get('name') if ci < len(scene.colliders) else None
        out[ci] = dict(name=name or f'Rope {ci + 1}', points=_to_world(F, pts), width=2.0 * num(rope['radius']))
    return out


def matter_points(snap, temps, origin, dx, dims, mats, colours, F, vel=None):
    """Matter's particles (the frame cache's 8-byte form, matter.Matter.snapshot) as world points: dict(points, ids,
    width, colours, slots, temperature or None, velocities (from `vel`, the live particles' m/s, else None))."""
    s = np.asarray(snap, np.uint16).reshape(-1, 4)
    slot = (s[:, 3] >> 12).astype(np.int32)
    live = slot < 15
    ids = np.nonzero(live)[0].astype(np.int64)
    q = s[live, :3].astype(np.float32) * (np.asarray(dims, np.float32) * np.float32(dx / 65535.0))
    pts = _to_world(F, q + np.asarray(origin, np.float32))
    rnd = (s[live, 3] & 4095).astype(np.float32) / 4095.0
    slot = np.clip(slot[live], 0, max(len(mats) - 1, 0))
    base = np.array([c[0] if c[0] is not None else m.colour for m, c in zip(mats, colours)] or [(0.5, 0.5, 0.5)], np.float32)
    var = np.array([m.variation for m in mats] or [0.0], np.float32)
    col = base[slot] * (1.0 + var[slot] * (rnd - 0.5))[:, None]     # (mpm_surf_p2g.wgsl: each grain a little different)
    t = None
    if temps is not None and len(temps) == len(s):
        t = np.asarray(temps, np.float32)[live]
    v = None
    if vel is not None and len(vel) >= len(s):
        v = (np.asarray(vel, float)[:len(s)][live] @ F[:3, :3].T).astype(np.float32)
    return dict(points=pts, ids=ids, width=float(MATTER_WIDTH * dx), colours=np.clip(col, 0.0, None).astype(np.float32),
                slots=slot, temperature=t, velocities=v)


def matter_names(mats, colours):
    """Each material slot's name, for blackbody:materials."""
    return [m.label + ('' if c[0] is None else ' (own colour)') for m, c in zip(mats, colours)]


def matter_at(eng, scene, frame):
    """The sand, snow, mud, jelly and clay at `frame` (matter_points, and their slots' names), or None."""
    m = getattr(eng, '_matter', None)
    if m is None or not m.active or scene.kind == 'cloud':
        return None
    vel = None
    if eng.sim_frame == frame:
        snap, temps = m.snapshot(), m.snapshot_temperatures()
        if snap is not None and len(snap):
            P = m.read_particles()          # (the live particles' velocities, for motion blur)
            vel = P[:, 4:7] * float(scene.v('domain', 'time_scale', frame))
    else:
        entry = eng.cache.get(frame) if getattr(eng, 'cache', None) is not None else None
        snap = entry.get('matter') if entry else None
        temps = entry.get('matter_t') if entry else None
    if snap is None or not len(snap):
        return None
    out = matter_points(snap, temps, m.origin, m.dx, m.dims, m._mats, m._colours, fire_matrix(scene, frame), vel)
    out['names'] = matter_names(m._mats, m._colours)
    return out


def grass_blades(x, st, grows, BL, patches, names, F):
    """Every growing blade as a curve, per patch: [dict(patch (its index), name, points (n, 5, 3) world, widths (n, 5),
    colours (n, 3), burn (n,), burning (n,))]. x: (N, 5, 3) fire-local points; st: (N, 4) its fire (strands.py ST); BL: the blades;
    patches: strands.patch_data."""
    out = []
    x = np.asarray(x, float)
    st = np.asarray(st, np.float32)
    BL = np.asarray(BL, np.float32)
    patch = np.round(BL[:, 0, 2]).astype(int)
    P = x.shape[1]
    u = np.arange(P) / max(P - 1, 1)
    taper = np.minimum(1.0, GRASS_TAPER * (1.0 - u))
    for p in sorted(set(patch[grows].tolist())):
        k = grows & (patch == p)
        rnd = BL[k, 0, 3]
        fresh, dry, dryness = patches[p, 0, :3], patches[p, 1, :3], float(patches[p, 1, 3])
        d = np.clip(dryness + 0.2 * (rnd - 0.5), 0.0, 1.0)[:, None]
        col = (fresh * (1.0 - d) + dry * d) * (0.8 + 0.4 * np.modf(rnd * 7.31)[0])[:, None]
        s = st[k]
        burning, done = s[:, 2] > 0.5, s[:, 1] >= 1.0
        brown = 0.85 * np.clip(s[:, 0], 0.0, 1.0) * ~(burning | done)
        col = col * (1.0 - brown[:, None]) + col * np.array([0.5, 0.33, 0.15], np.float32) * brown[:, None]
        ch = np.where(done, 1.0, np.where(burning, np.clip(3.0 * s[:, 1], 0.0, 1.0), 0.0))[:, None]
        col = col * (1.0 - ch) + np.array(CHAR, np.float32) * (0.8 + 0.4 * rnd)[:, None] * ch
        pts = _to_world(F, x[k].reshape(-1, 3)).reshape(-1, P, 3)
        out.append(dict(patch=int(p), name=names[p] if p < len(names) else f'Grass {p + 1}', points=pts,
                        widths=(BL[k, 1, 1][:, None] * taper[None, :]).astype(np.float32), colours=col.astype(np.float32),
                        burn=np.clip(s[:, 1], 0.0, 2.0).astype(np.float32), burning=burning.astype(np.float32)))
    return out


def grass_at(eng, scene, frame):
    """The grass at `frame` (grass_blades), or None."""
    from ..engine.strands import POINTS
    s = getattr(eng, '_strands', None)
    if s is None or not s.active or not s.placed or scene.kind == 'cloud':
        return None
    RT = s._read('RT')
    if eng.sim_frame == frame:
        x = s._read('X', POINTS).reshape(s.n, POINTS, 4)[:, :, :3]
        st = s._read('ST')
    else:
        entry = eng.cache.get(frame) if getattr(eng, 'cache', None) is not None else None
        arrays = entry.get('strands') if entry else None
        if arrays is None or len(arrays['off']) != s.n:
            return None
        h = np.maximum(s.BL[:, 1, 0], 1e-6)[:, None, None]
        x = np.empty((s.n, POINTS, 3), np.float32)
        x[:, 0] = RT[:, :3]
        x[:, 1:] = RT[:, None, :3] + np.asarray(arrays['off'], np.float32) * (h / 127.0)
        fire = np.asarray(arrays['fire'])
        st = np.stack([fire[:, 0] / 255.0, fire[:, 1] / 127.5, (fire[:, 2] > 127).astype(np.float32),
                       np.zeros(len(fire))], 1)
    names = [d['name'] for d in getattr(scene, 'strands', None) or [] if d['enabled']]
    return grass_blades(x, st, RT[:, 3] > 0.5, s.BL, s.patches, names, fire_matrix(scene, frame)) or None


_BB = None


def glow_colours(kelvin):
    """The colour of a glow at `kelvin` (K), its brightest channel 1 (engine/lut.py's blackbody)."""
    global _BB
    from ..engine.lut import blackbody_lut
    if _BB is None:
        lut = blackbody_lut(256, 500.0, 6000.0)[:, :3]
        _BB = lut / np.maximum(lut.max(1, keepdims=True), 1e-9)
    k = np.clip((np.asarray(kelvin, np.float32) - 500.0) / 5500.0 * 255.0, 0.0, 255.0)
    i = np.minimum(k.astype(int), 254)
    f = (k - i)[:, None]
    return (_BB[i] * (1.0 - f) + _BB[i + 1] * f).astype(np.float32)


def ember_points(A, B, D, E, F, time_scale=1.0):
    """The embers alive (particle buffers A, B, D, E of embers.wgsl) as world points: dict(points, velocities (m/s of
    the shot), widths, temperature, colours: their glow's colour, tinted by the colourant they carry)."""
    A, B, D, E = (np.asarray(a, np.float32).reshape(-1, 4) for a in (A, B, D, E))
    live = A[:, 3] > 0.0
    t = B[live, 3]
    col = glow_colours(t)
    tint = np.clip(E[live, 3:4], 0.0, 1.0)
    col = col * (1.0 - tint) + E[live, :3] * tint
    return dict(points=_to_world(F, A[live, :3]), velocities=(B[live, :3] @ F[:3, :3].T * time_scale).astype(np.float32),
                widths=np.maximum(D[live, 1], 1e-4).astype(np.float32), temperature=t.astype(np.float32),
                colours=col.astype(np.float32))


def embers_at(eng, scene, frame):
    """The embers and sparks at `frame` (ember_points), or None."""
    em = getattr(eng, 'embers', None)
    if em is None or not em.count or scene.kind not in ('fire', 'both'):
        return None
    if eng.sim_frame == frame:
        bufs = [np.frombuffer(eng.gpu.read_buffer(b), np.float32).reshape(-1, 4) for b in em.cached_buffers()]
    else:
        entry = eng.cache.get(frame) if getattr(eng, 'cache', None) is not None else None
        if not entry or 'embers' not in entry:
            return None
        bufs = list(entry['embers'])
    out = ember_points(*bufs, fire_matrix(scene, frame), float(scene.v('domain', 'time_scale', frame)))
    return out if len(out['points']) else None


def frame_data(eng, scene, frame):
    """Everything a frame of `scene` (a layer of the shot) puts in the USD scene, from its engine."""
    return dict(objects=objects_at(eng, scene, frame), pieces=pieces_at(eng, scene, frame), ropes=ropes_at(eng, scene, frame),
                matter=matter_at(eng, scene, frame), grass=grass_at(eng, scene, frame), embers=embers_at(eng, scene, frame))


# -- the writer --------------------------------------------------------------------------------------------------------

def _vt(kind, a):
    from pxr import Vt
    a = np.ascontiguousarray(a)
    return {'v3': Vt.Vec3fArray, 'f': Vt.FloatArray, 'i': Vt.IntArray, 'i64': Vt.Int64Array}[kind].FromNumpy(a)


def _extent(pts, pad=0.0):
    from pxr import Gf, Vt
    if not len(pts):
        return Vt.Vec3fArray([Gf.Vec3f(0.0), Gf.Vec3f(0.0)])
    lo, hi = np.asarray(pts).reshape(-1, 3).min(0) - pad, np.asarray(pts).reshape(-1, 3).max(0) + pad
    return Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])


def _rigid_ops(xf):
    """A rigid motion's ops on an Xformable: a translate and an orient (the module's notes), in doubles (a float
    quaternion moves the far edge of the camera's picture by a few thousandths of a pixel)."""
    from pxr import UsdGeom
    return xf.AddTranslateOp(), xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)


def _set_ops(ops, pos, quat, t):
    from pxr import Gf
    x, y, z, w = (float(v) for v in quat)
    ops[0].Set(Gf.Vec3d(*(float(v) for v in pos)), t)
    ops[1].Set(Gf.Quatd(w, x, y, z), t)


def _pose(M, prev=None):
    """A rigid 4x4 (column vectors) as (translation, quaternion x y z w), on the side of `prev`'s (rigid_poses)."""
    pos, quat = rigid_poses(M, None if prev is None else prev[1])
    return pos[0], quat[0]


# the attributes the points and curves files hold (value clips' manifest): name -> USD type
POINT_ATTRS = {'points': 'Point3fArray', 'velocities': 'Vector3fArray', 'widths': 'FloatArray', 'ids': 'Int64Array',
               'extent': 'Float3Array', 'primvars:displayColor': 'Color3fArray'}
MATTER_ATTRS = dict(POINT_ATTRS, **{'primvars:material': 'IntArray', 'primvars:temperature': 'FloatArray'})
EMBER_ATTRS = dict(POINT_ATTRS, **{'primvars:temperature': 'FloatArray'})
CURVE_ATTRS = {'points': 'Point3fArray', 'curveVertexCounts': 'IntArray', 'widths': 'FloatArray', 'extent': 'Float3Array',
               'primvars:displayColor': 'Color3fArray', 'primvars:burn': 'FloatArray', 'primvars:burning': 'FloatArray'}


class SceneWriter:
    """A whole shot as one USD scene (the module's notes), written frame by frame: add() each layer's frame_data
    (and the camera), close() at the end."""

    def __init__(self, path, fps):
        from pxr import Usd, UsdGeom
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.stage = Usd.Stage.CreateNew(str(self.path))
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self.stage.SetTimeCodesPerSecond(self.fps)
        self.stage.SetFramesPerSecond(self.fps)
        world = UsdGeom.Xform.Define(self.stage, '/World')
        self.stage.SetDefaultPrim(world.GetPrim())
        self.frames = []           # every frame added, in order
        self.notes = []            # what could not be written as it is (said once each)
        self.extra = []            # the points files written (beside the scene)
        self._taken = {}           # parent path -> prim names used under it
        self._objs, self._pcs, self._ropes, self._cams, self._grass_paths = {}, {}, {}, {}, {}
        self._heavy = {}           # prim path -> (its attributes' types, its type name): read from the points files
        self._clip_dir = self.path.parent / f'{self.path.stem}_points'
        self._clips = {}           # frame -> its points file's name
        self._clip = None          # (frame, stage) of the points file being filled
        self._prev = {}            # prim path -> (frame, ids, points): for velocities between cached frames
        self._groups = {'/World'}
        self._first = None

    # -- helpers ---------------------------------------------------------------------------------------------------

    def _name(self, parent, name, fallback='Item'):
        return prim_name(name, self._taken.setdefault(parent, set()), fallback)

    def _group(self, path):
        from pxr import UsdGeom
        if path not in self._groups:
            UsdGeom.Xform.Define(self.stage, path)
            self._groups.add(path)
        return path

    def _note(self, text):
        if text not in self.notes:
            self.notes.append(text)
            log.info('%s', text)

    def _hidden_before(self, img, t):
        """A prim that first appears at time t: hidden at every frame before it."""
        from pxr import UsdGeom
        if self._first is not None and t > self._first:
            img.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible, self._first)

    # -- frames ----------------------------------------------------------------------------------------------------

    def add(self, frame, data=None, root='/World', camera=None):
        """One frame of one layer: its frame_data (None: only the camera) under `root`, and the camera (CameraFrame)
        at root/Camera if given."""
        from pxr import Usd
        frame = float(frame)
        if not self.frames or self.frames[-1] != frame:
            if self._clip is not None and self._clip[0] != frame:
                self._save_clip()
            self.frames.append(frame)
        if self._first is None:
            self._first = frame
        t = Usd.TimeCode(frame)
        self._group(root)
        if camera is not None:
            self._camera(f'{root}/Camera', t, camera)
        if not data:
            return
        for ob in data.get('objects') or []:
            self._object(root, t, ob)
        for ci, ps in (data.get('pieces') or {}).items():
            self._pieces(root, t, ci, ps)
        for ci, rope in (data.get('ropes') or {}).items():
            self._rope(root, t, ci, rope)
        if data.get('matter') is not None:
            self._matter(root, frame, data['matter'])
        for g in data.get('grass') or []:
            self._grass(root, frame, g)
        if data.get('embers') is not None:
            self._embers(root, frame, data['embers'])

    def _camera(self, path, t, cf):
        from pxr import Gf, Sdf, UsdGeom, UsdRender
        rec = self._cams.get(path)
        if rec is None:
            c = UsdGeom.Camera.Define(self.stage, path)
            c.CreateProjectionAttr(UsdGeom.Tokens.perspective)
            rec = self._cams[path] = dict(cam=c, ops=_rigid_ops(c), pose=None)
            if path == '/World/Camera':
                rs = UsdRender.Settings.Define(self.stage, '/Render/Settings')
                rs.CreateResolutionAttr(Gf.Vec2i(*(int(x) for x in cf.size)))
                rs.CreateCameraRel().SetTargets([c.GetPath()])
            if cf.lens_k1:
                # (the footage's lens distortion the element is bent by, which no camera holds)
                a = c.GetPrim().CreateAttribute('blackbody:lens_k1', Sdf.ValueTypeNames.Float, custom=True)
                a.Set(float(cf.lens_k1))
                a.SetDocumentation('Radial distortion of the picture: x_d = x_u (1 + k1 r_u^2), r from the frame\'s '
                                   'centre in half-diagonals (Blackbody\'s Composite > Lens distortion).')
        c = rec['cam']
        rec['pose'] = _pose(cf.to_world, rec['pose'])
        _set_ops(rec['ops'], *rec['pose'], t)
        c.GetFocalLengthAttr().Set(cf.focal_mm, t)
        c.GetHorizontalApertureAttr().Set(cf.h_aperture, t)
        c.GetVerticalApertureAttr().Set(cf.v_aperture, t)
        c.GetHorizontalApertureOffsetAttr().Set(cf.offset_mm[0], t)
        c.GetVerticalApertureOffsetAttr().Set(cf.offset_mm[1], t)
        c.GetClippingRangeAttr().Set(Gf.Vec2f(cf.near, cf.far), t)

    def _object(self, root, t, ob):
        from pxr import Gf, UsdGeom, Vt
        key = (root, ob['index'])
        rec = self._objs.get(key)
        if rec is None:
            parent = self._group(f'{root}/Objects')
            path = f'{parent}/{self._name(parent, ob["name"], "Object")}'
            x = UsdGeom.Xform.Define(self.stage, path)
            shape, mesh = ob['shape'], None
            if shape == 'mesh':
                mesh = self._mesh_of(ob)
                if mesh is None:
                    shape = 'box'
            if shape == 'sphere':
                g = UsdGeom.Sphere.Define(self.stage, path + '/Shape')
                g.CreateRadiusAttr(1.0)
            elif shape == 'cylinder':
                g = UsdGeom.Cylinder.Define(self.stage, path + '/Shape')
                g.CreateRadiusAttr(1.0)
                g.CreateHeightAttr(2.0)
                g.CreateAxisAttr(UsdGeom.Tokens.y)
            elif shape == 'mesh':
                g = UsdGeom.Mesh.Define(self.stage, path + '/Shape')
                g.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                self._set_mesh(g, mesh, None)
            else:
                g = UsdGeom.Cube.Define(self.stage, path + '/Shape')
                g.CreateSizeAttr(2.0)
            g.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*ob['colour'])]))
            if ob['guide']:
                g.CreatePurposeAttr(UsdGeom.Tokens.guide)     # (a helper that hides nothing: not rendered)
            deforms = shape == 'mesh' and ob['mesh_frame'] is not None
            rec = self._objs[key] = dict(ops=_rigid_ops(x), pose=None, scale=g.AddScaleOp(), img=UsdGeom.Imageable(x),
                                         mesh=g if deforms else None, ob=ob)
            self._hidden_before(rec['img'], t)
        # (hidden: held where it was last seen, not slid toward the 'gone' pose far below: the module's notes)
        if ob['shown'] or rec['pose'] is None:
            rec['pose'] = _pose(ob['matrix'], rec['pose'])
        _set_ops(rec['ops'], *rec['pose'], t)
        rec['scale'].Set(Gf.Vec3f(*(float(v) for v in ob['scale'])), t)
        rec['img'].GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if ob['shown'] else UsdGeom.Tokens.invisible, t)
        if rec['mesh'] is not None:
            m = self._mesh_of(ob)
            if m is not None:
                self._set_mesh(rec['mesh'], m, t)

    def _mesh_of(self, ob):
        from ..engine.mesh import load_mesh
        try:
            v, tris = load_mesh(ob['mesh'], ob['mesh_frame'])
            return np.asarray(v, np.float32), np.asarray(tris, np.int32)
        except Exception as ex:
            self._note(f'{ob["name"]}: its mesh could not be read ({ex}), so the USD scene holds a box for it.')
            return None

    @staticmethod
    def _set_mesh(g, mesh, t):
        from pxr import Usd
        v, tris = mesh
        tc = Usd.TimeCode.Default() if t is None else t
        g.GetPointsAttr().Set(_vt('v3', v), tc)
        g.GetFaceVertexCountsAttr().Set(_vt('i', np.full(len(tris), 3, np.int32)), tc)
        g.GetFaceVertexIndicesAttr().Set(_vt('i', tris.ravel()), tc)
        g.GetExtentAttr().Set(_extent(v), tc)

    def _pieces(self, root, t, ci, ps):
        from pxr import UsdGeom
        key = (root, int(ci))
        rec = self._pcs.get(key)
        if rec is None:
            parent = self._group(f'{root}/Pieces')
            base = self._group(f'{parent}/{self._name(parent, ps["name"], "Object")}')
            prims = []
            for k in range(ps['n']):
                pts, counts, idx, cut = piece_mesh(ps['frac'].pieces[k])
                g = UsdGeom.Mesh.Define(self.stage, f'{base}/Piece_{k + 1:04d}')
                g.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                g.CreatePointsAttr(_vt('v3', pts))
                g.CreateFaceVertexCountsAttr(_vt('i', counts))
                g.CreateFaceVertexIndicesAttr(_vt('i', idx))
                g.CreateExtentAttr(_extent(pts))
                # its colour face by face: the cut faces in the inside's (and the cut faces as a subset, for a
                # material of their own)
                col = np.tile(np.asarray(ps['colour'], np.float32), (len(counts), 1))
                if len(cut):
                    col[cut] = ps['inside']
                    UsdGeom.Subset.CreateGeomSubset(g, 'cut', UsdGeom.Tokens.face, _vt('i', cut), 'materialBind')
                g.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(_vt('v3', col))
                img = UsdGeom.Imageable(g)
                prims.append((_rigid_ops(g), img))
                self._hidden_before(img, t)
            rec = self._pcs[key] = dict(prims=prims, pos=np.zeros((ps['n'], 3)), seen=np.zeros(ps['n'], bool),
                                        quat=np.tile([0.0, 0.0, 0.0, 1.0], (ps['n'], 1)))
        n = min(ps['n'], len(rec['prims']))
        # (one burnt to ash is held where it was last seen, as a hidden object is)
        pos, quat = rigid_poses(ps['matrices'][:n], rec['quat'][:n])
        now = np.asarray(ps['shown'][:n], bool) | ~rec['seen'][:n]
        rec['pos'][:n][now], rec['quat'][:n][now] = pos[now], quat[now]
        rec['seen'][:n] |= np.asarray(ps['shown'][:n], bool)
        for k, (ops, img) in enumerate(rec['prims'][:n]):
            _set_ops(ops, rec['pos'][k], rec['quat'][k], t)
            img.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if ps['shown'][k] else UsdGeom.Tokens.invisible, t)

    def _rope(self, root, t, ci, rope):
        from pxr import UsdGeom
        key = (root, int(ci))
        rec = self._ropes.get(key)
        if rec is None:
            parent = self._group(f'{root}/Ropes')
            g = UsdGeom.BasisCurves.Define(self.stage, f'{parent}/{self._name(parent, rope["name"], "Rope")}')
            g.CreateTypeAttr(UsdGeom.Tokens.linear)
            g.CreateWidthsAttr(_vt('f', np.array([rope['width']], np.float32)))
            g.SetWidthsInterpolation(UsdGeom.Tokens.constant)
            rec = self._ropes[key] = g
            self._hidden_before(UsdGeom.Imageable(g), t)
        p = rope['points']
        rec.GetPointsAttr().Set(_vt('v3', p), t)
        rec.GetCurveVertexCountsAttr().Set(_vt('i', np.array([len(p)] if len(p) >= 2 else [], np.int32)), t)
        rec.GetExtentAttr().Set(_extent(p, 0.5 * rope['width']), t)

    # -- the points and curves (value clips) ---------------------------------------------------------------------------

    def _clip_stage(self, frame):
        """The points file of `frame` (made when first needed), a stage of its own."""
        from pxr import Usd
        if self._clip is not None and self._clip[0] == frame:
            return self._clip[1]
        if self._clip is not None:
            self._save_clip()
        self._clip_dir.mkdir(parents=True, exist_ok=True)
        name = f'{self.path.stem}.{int(round(frame)):04d}.usdc'
        st = Usd.Stage.CreateNew(str(self._clip_dir / name))
        self._clip = (frame, st)
        self._clips[frame] = name
        return st

    def _save_clip(self):
        """Save the points file being filled and let it go: one frame of points in memory. (Saving it on a thread of its
        own, beside the next frame's simulation, measured no faster: the two compete for the CPU.)"""
        frame, st = self._clip
        self._clip = None
        st.GetRootLayer().Save()
        self.extra.append(st.GetRootLayer().realPath)

    def _heavy_prim(self, path, typename, attrs, setup):
        """Declare a prim whose values come from the points files, in the scene (once): its type, and `setup` for what
        does not change over time (the primvars' interpolation, a curve's basis)."""
        if path in self._heavy:
            return
        prim = self.stage.DefinePrim(path, typename)
        setup(prim)
        self._heavy[path] = (dict(attrs), typename)

    def _velocities(self, path, frame, ids, pts):
        """Velocities (m/s) of points from where the same ids were a frame before (cached matter keeps no velocities)."""
        prev = self._prev.get(path)
        self._prev[path] = (frame, ids, pts)
        v = np.zeros_like(pts)
        if prev is None or prev[0] >= frame or not len(ids):
            return v
        pf, pids, ppts = prev
        lookup = np.full(int(max(ids.max(), pids.max() if len(pids) else 0)) + 1, -1, np.int64)
        lookup[pids] = np.arange(len(pids))
        j = lookup[ids]
        ok = j >= 0
        v[ok] = (pts[ok] - ppts[j[ok]]) * (self.fps / (frame - pf))
        return v

    def _matter(self, root, frame, m):
        from pxr import UsdGeom, Vt
        path = f'{root}/Matter'
        heats = m.get('temperature') is not None

        def setup(prim):
            g = UsdGeom.Points(prim)
            g.CreateWidthsAttr()
            g.SetWidthsInterpolation(UsdGeom.Tokens.constant)
            pv = UsdGeom.PrimvarsAPI(prim)
            pv.CreatePrimvar('displayColor', _sdf('Color3fArray'), UsdGeom.Tokens.vertex)
            pv.CreatePrimvar('material', _sdf('IntArray'), UsdGeom.Tokens.vertex)
            if heats:
                pv.CreatePrimvar('temperature', _sdf('FloatArray'), UsdGeom.Tokens.vertex)
            prim.CreateAttribute('blackbody:materials', _sdf('StringArray'), custom=True)

        attrs = MATTER_ATTRS if heats else {k: v for k, v in MATTER_ATTRS.items() if k != 'primvars:temperature'}
        self._heavy_prim(path, 'Points', attrs, setup)
        names = self.stage.GetPrimAtPath(path).GetAttribute('blackbody:materials')
        if list(names.Get() or []) != list(m['names']):
            names.Set(Vt.StringArray(list(m['names'])))
        pts = m['points']
        vel = m['velocities'] if m.get('velocities') is not None else self._velocities(path, frame, m['ids'], pts)
        if m.get('velocities') is not None:
            self._prev[path] = (frame, m['ids'], pts)
        g = UsdGeom.Points.Define(self._clip_stage(frame), path)
        t = frame
        g.GetPointsAttr().Set(_vt('v3', pts), t)
        g.GetVelocitiesAttr().Set(_vt('v3', vel), t)
        g.GetIdsAttr().Set(_vt('i64', m['ids']), t)
        g.GetWidthsAttr().Set(_vt('f', np.array([m['width']], np.float32)), t)
        g.GetExtentAttr().Set(_extent(pts, 0.5 * m['width']), t)
        pv = UsdGeom.PrimvarsAPI(g)
        pv.CreatePrimvar('displayColor', _sdf('Color3fArray')).Set(_vt('v3', m['colours']), t)
        pv.CreatePrimvar('material', _sdf('IntArray')).Set(_vt('i', m['slots'].astype(np.int32)), t)
        if heats and 'primvars:temperature' in self._heavy[path][0]:
            pv.CreatePrimvar('temperature', _sdf('FloatArray')).Set(_vt('f', m['temperature']), t)

    def _grass(self, root, frame, gr):
        from pxr import UsdGeom
        parent = f'{root}/Grass'
        key = (root, gr.get('patch', gr['name']))     # (two patches may share a name)
        path = self._grass_paths.get(key)
        if path is None:
            self._group(parent)
            path = self._grass_paths[key] = f'{parent}/{self._name(parent, gr["name"], "Grass")}'

        def setup(prim):
            g = UsdGeom.BasisCurves(prim)
            g.CreateTypeAttr(UsdGeom.Tokens.linear)
            g.CreateWidthsAttr()
            g.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
            pv = UsdGeom.PrimvarsAPI(prim)
            pv.CreatePrimvar('displayColor', _sdf('Color3fArray'), UsdGeom.Tokens.uniform)
            pv.CreatePrimvar('burn', _sdf('FloatArray'), UsdGeom.Tokens.uniform)
            pv.CreatePrimvar('burning', _sdf('FloatArray'), UsdGeom.Tokens.uniform)

        self._heavy_prim(path, 'BasisCurves', CURVE_ATTRS, setup)
        g = UsdGeom.BasisCurves.Define(self._clip_stage(frame), path)
        n, P = gr['points'].shape[:2]
        t = frame
        g.GetPointsAttr().Set(_vt('v3', gr['points'].reshape(-1, 3)), t)
        g.GetCurveVertexCountsAttr().Set(_vt('i', np.full(n, P, np.int32)), t)
        g.GetWidthsAttr().Set(_vt('f', gr['widths'].reshape(-1)), t)
        g.GetExtentAttr().Set(_extent(gr['points'], float(gr['widths'].max(initial=0.0))), t)
        pv = UsdGeom.PrimvarsAPI(g)
        pv.CreatePrimvar('displayColor', _sdf('Color3fArray')).Set(_vt('v3', gr['colours']), t)
        pv.CreatePrimvar('burn', _sdf('FloatArray')).Set(_vt('f', gr['burn']), t)
        pv.CreatePrimvar('burning', _sdf('FloatArray')).Set(_vt('f', gr['burning']), t)

    def _embers(self, root, frame, em):
        from pxr import UsdGeom
        path = f'{root}/Embers'

        def setup(prim):
            g = UsdGeom.Points(prim)
            g.CreateWidthsAttr()
            g.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
            pv = UsdGeom.PrimvarsAPI(prim)
            pv.CreatePrimvar('displayColor', _sdf('Color3fArray'), UsdGeom.Tokens.vertex)
            pv.CreatePrimvar('temperature', _sdf('FloatArray'), UsdGeom.Tokens.vertex)

        self._heavy_prim(path, 'Points', EMBER_ATTRS, setup)
        g = UsdGeom.Points.Define(self._clip_stage(frame), path)
        t = frame
        g.GetPointsAttr().Set(_vt('v3', em['points']), t)
        g.GetVelocitiesAttr().Set(_vt('v3', em['velocities']), t)
        g.GetWidthsAttr().Set(_vt('f', em['widths']), t)
        g.GetExtentAttr().Set(_extent(em['points'], float(em['widths'].max(initial=0.0))), t)
        pv = UsdGeom.PrimvarsAPI(g)
        pv.CreatePrimvar('displayColor', _sdf('Color3fArray')).Set(_vt('v3', em['colours']), t)
        pv.CreatePrimvar('temperature', _sdf('FloatArray')).Set(_vt('f', em['temperature']), t)

    def _write_clips(self):
        """Point every points-and-curves prim at the points files, frame by frame (value clips), with a manifest of
        what they hold (empty where a frame has none of it: matter not poured yet, embers all out)."""
        from pxr import Sdf, Usd
        if self._clip is not None:
            self._save_clip()
        if not self._heavy:
            return
        # frames without a points file of their own (before the first): an empty one
        empty = None
        if any(f not in self._clips for f in self.frames):
            empty = f'{self.path.stem}.empty.usdc'
            Sdf.Layer.CreateNew(str(self._clip_dir / empty)).Save()
            self.extra.append(str(self._clip_dir / empty))
        files = sorted(set(self._clips.values())) + ([empty] if empty else [])
        index = {name: i for i, name in enumerate(files)}
        manifest = self._clip_dir / f'{self.path.stem}.manifest.usda'
        man = Sdf.Layer.CreateNew(str(manifest))
        for path, (attrs, _typename) in self._heavy.items():
            ps = Sdf.CreatePrimInLayer(man, path)
            ps.specifier = Sdf.SpecifierOver
            for name, tn in attrs.items():
                a = Sdf.AttributeSpec(ps, name, getattr(Sdf.ValueTypeNames, tn), Sdf.VariabilityVarying)
                a.default = getattr(Sdf.ValueTypeNames, tn).defaultValue
        man.Save()
        self.extra.append(str(manifest))
        rel = lambda name: Sdf.AssetPath(f'./{self._clip_dir.name}/{name}')
        active = [(f, float(index[self._clips.get(f, empty)])) for f in self.frames]
        times = [(f, f) for f in self.frames]
        for path in self._heavy:
            c = Usd.ClipsAPI(self.stage.GetPrimAtPath(path))
            c.SetClipAssetPaths([rel(n) for n in files])
            c.SetClipPrimPath(path)
            c.SetClipActive(active)
            c.SetClipTimes(times)
            c.SetClipManifestAssetPath(rel(manifest.name))

    def close(self):
        self._write_clips()
        if self.frames:
            self.stage.SetStartTimeCode(self.frames[0])
            self.stage.SetEndTimeCode(self.frames[-1])
        self.stage.GetRootLayer().Save()
        return self.path


def _sdf(name):
    from pxr import Sdf
    return getattr(Sdf.ValueTypeNames, name)
