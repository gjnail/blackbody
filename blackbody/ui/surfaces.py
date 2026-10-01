"""Surfaces: the real things in the footage that an effect meets, lined up like the ground. A wall, something flat and
raised (a table, a step, a platform), a ramp, stairs. Each becomes a solid (a mesh collider) where it stands in the
world, the same in every layer, and stays put when the effect is moved: water splashes off it and runs down it,
fire spreads over it if it is made burnable, smoke flows round it, and it hides what is behind it.

A surface is kept in the shot's ground match (scene.ground['surfaces']) in world metres; its collider in each layer is
placed from it relative to where that layer's effect stands (sync)."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import QStandardPaths

from ..engine import camera as cam
from ..scene import groundmatch as GM

KINDS = [('wall', 'A wall, or anything upright on the ground'),
         ('level', 'Something flat and raised: a table, a step, a platform'),
         ('ramp', 'A ramp or slope rising from the ground'),
         ('stairs', 'Stairs: put the grid on the lowest step')]
NAMES = {'wall': 'Wall', 'level': 'Platform', 'ramp': 'Ramp', 'stairs': 'Stairs', 'slope': 'Sloping ground'}


def folder():
    return Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)) / 'surfaces'


def camera_of(scene, frame):
    """(R camera-to-world, eye, focal px, (W, H)) of the shot's camera at a frame, in world metres."""
    spec, fire = scene.camera(frame)
    W, H = scene.output_size()
    st = cam.compute(spec, W / H, fire)
    R = st.view[:3, :3].T
    f = spec.focal_mm * W / spec.sensor_mm
    return R, np.asarray(st.eye, float), float(f), (W, H)


def ground_plane(scene):
    """(normal, point) of the lined-up ground in the world."""
    g = scene.ground or {}
    return np.asarray(g.get('plane', (0.0, 1.0, 0.0)), float), np.zeros(3)


def _hit(eye, d, n, p0):
    den = float(n @ d)
    if abs(den) < 1e-9:
        return None
    t = float(n @ (p0 - eye)) / den
    return None if t <= 0 else eye + d * t


def solve(kind, corners_px, R, eye, f, size, ground=((0.0, 1.0, 0.0), (0.0, 0.0, 0.0)), height=0.75, rise=0.17, steps=4,
          solid=True):
    """The surface the grid (4 pixel corners) marks, as {'kind', 'top': 4 world corners, 'verts', 'tris'} (the solid,
    in world metres). Raises ValueError."""
    rays = [R @ GM.ray(c, size, f) for c in corners_px]
    gn, gp = np.asarray(ground[0], float), np.asarray(ground[1], float)
    up = np.array([0.0, 1.0, 0.0])
    order = sorted(range(4), key=lambda k: -corners_px[k][1])   # lowest in the picture first: nearest the ground
    if kind in ('wall', 'ramp'):
        a, b = sorted(order[:2])
        if (b - a) % 4 not in (1, 3):
            raise ValueError('Put the grid’s bottom edge (two neighbouring corners) where it meets the ground.')
        B1, B2 = _hit(eye, rays[a], gn, gp), _hit(eye, rays[b], gn, gp)
        if B1 is None or B2 is None:
            raise ValueError('The bottom edge is above the horizon: put it where the thing meets the ground.')
        edge = B2 - B1
        if kind == 'wall':
            n = np.cross(edge, up)
        else:   # the ramp's sides run up it, toward a vanishing point
            others = [k for k in range(4) if k not in (a, b)]
            # the side from each bottom corner to its neighbour on the far edge
            sa = next(k for k in others if (k - a) % 4 in (1, 3))
            sb = next(k for k in others if k != sa)
            v = np.cross(np.cross(GM._h(corners_px[a]), GM._h(corners_px[sa])), np.cross(GM._h(corners_px[b]), GM._h(corners_px[sb])))
            g = R @ GM._dir(v, size, f)
            if np.linalg.norm(g) < 1e-12:
                raise ValueError('The ramp’s sides cannot be followed: spread the grid out along it.')
            n = np.cross(edge, g)
        if np.linalg.norm(n) < 1e-9:
            raise ValueError('The bottom edge has no length: spread its corners apart.')
        n /= np.linalg.norm(n)
        if n @ (eye - B1) < 0:
            n = -n
        if kind == 'ramp' and n @ up < 0:
            n = -n
        top = []
        for k in range(4):
            if k in (a, b):
                top.append(B1 if k == a else B2)
            else:
                p = _hit(eye, rays[k], n, B1)
                if p is None:
                    raise ValueError('Part of the grid is beyond the surface’s horizon: keep its corners on the thing.')
                top.append(p)
        top = np.array(top)
        thick = 0.15 if kind == 'wall' else 0.08
        verts, tris = _slab(top, n, thick)
    elif kind in ('level', 'stairs'):
        y = float(rise if kind == 'stairs' else height)
        top = []
        for k in range(4):
            p = _hit(eye, rays[k], up, np.array([0.0, y, 0.0]))
            if p is None:
                raise ValueError('Part of the grid is above the horizon at that height: keep its corners on the top.')
            top.append(p)
        top = np.array(top)
        if kind == 'level':
            verts, tris = (_box_down(top, y) if solid else _slab(top, up, 0.05))
        else:
            front = order[:2]
            back = [k for k in range(4) if k not in front]
            f0, f1 = top[front[0]], top[front[1]]
            mid_f = (f0 + f1) / 2
            mid_b = (top[back[0]] + top[back[1]]) / 2
            go = mid_b - mid_f
            go[1] = 0.0
            depth = float(np.linalg.norm(go))
            if depth < 0.05:
                raise ValueError('The step has no depth: put the grid on its top, front edge to back edge.')
            go /= depth
            verts, tris = _stairs(f0, f1, go, depth, float(rise), int(steps))
            across = (f1 - f0) / 2
            across[1] = 0.0
            mid = np.array([mid_f[0], 0.0, mid_f[2]])
            treads = []
            for k in range(max(1, min(int(steps), 60))):   # each step's top, for putting things on them
                y = (k + 1) * float(rise)
                a0, a1 = mid + go * k * depth, mid + go * (k + 1) * depth
                treads.append([tuple(float(v) for v in q + np.array([0, y, 0])) for q in (a0 - across, a0 + across, a1 + across, a1 - across)])
            return {'kind': kind, 'top': [tuple(float(v) for v in p) for p in top], 'treads': treads, 'verts': verts, 'tris': tris}
    else:
        raise ValueError(f'Unknown surface {kind}')
    return {'kind': kind, 'top': [tuple(float(v) for v in p) for p in top], 'verts': verts, 'tris': tris}


def _prism(poly, e1, e2, n, origin, c0, c1):
    """A closed solid: a planar polygon (2D, in the e1 e2 basis about origin) swept along n from c0 to c1."""
    from . import textmesh as T
    v, t = T.extrude(T.faces([np.asarray(poly, float)]), c1 - c0)
    w = origin + np.outer(v[:, 0], e1) + np.outer(v[:, 1], e2) + np.outer(v[:, 2] + (c0 + c1) / 2, n)
    return w, t


def _basis(n):
    n = n / np.linalg.norm(n)
    a = np.cross(n, [0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.cross(n, [1.0, 0.0, 0.0])
    a /= np.linalg.norm(a)
    return a, np.cross(n, a), n


def _slab(top, n, thick):
    """The quad top, thick behind its face (away from n)."""
    e1, e2, n = _basis(np.asarray(n, float))
    c = top.mean(0)
    poly = [((p - c) @ e1, (p - c) @ e2) for p in top]
    return _prism(poly, e1, e2, n, c, -thick, 0.0)


def _box_down(top, y):
    """The quad top (level, at height y) as a block down to the ground."""
    c = top.mean(0)
    poly = [(p[0] - c[0], p[2] - c[2]) for p in top]
    origin = np.array([c[0], 0.0, c[2]])
    return _prism(poly, np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), np.array([0, -1.0, 0]), origin, -y, 0.0)


def _stairs(f0, f1, go, depth, rise, steps):
    """Steps going up away from the front edge f0 f1 (each `rise` high and `depth` deep), as one solid: the side
    profile swept across the width."""
    steps = max(1, min(int(steps), 60))
    across = f1 - f0
    across[1] = 0.0
    width = float(np.linalg.norm(across))
    across /= max(width, 1e-9)
    base = np.array([(f0[0] + f1[0]) / 2, 0.0, (f0[2] + f1[2]) / 2])
    prof = [(0.0, 0.0)]
    for k in range(steps):
        prof += [(k * depth, (k + 1) * rise), ((k + 1) * depth, (k + 1) * rise)]
    prof += [(steps * depth, 0.0)]
    up = np.array([0.0, 1.0, 0.0])
    return _prism(prof, go, up, across, base, -width / 2, width / 2)


def slope_surface(normal, reach):
    """The lined-up sloping ground about the origin, `reach` metres each way, as a solid 0.3 m deep."""
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)
    e1, e2, n = _basis(n)
    top = np.array([e1 * a + e2 * b for a, b in ((-reach, -reach), (reach, -reach), (reach, reach), (-reach, reach))])
    verts, tris = _slab(top, n, 0.3)
    return {'kind': 'slope', 'top': [tuple(float(v) for v in p) for p in top], 'verts': verts, 'tris': tris}


def write(surface):
    """The surface's solid as an OBJ (vertices about its middle), made once. Returns (path, middle in the world)."""
    verts, tris = np.asarray(surface['verts'], float), np.asarray(surface['tris'])
    centre = verts.mean(0)
    rel = verts - centre
    text = ''.join(f'v {x:.5f} {y:.5f} {z:.5f}\n' for x, y, z in rel) + ''.join(f'f {a + 1} {b + 1} {c + 1}\n' for a, b, c in tris)
    key = hashlib.sha1(text.encode()).hexdigest()[:12]
    path = folder() / f'{surface["kind"]}_{key}.obj'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'# Blackbody surface: {surface["kind"]}\n' + text, encoding='utf-8')
    return str(path), tuple(float(x) for x in centre)


def local(scene, p, frame=None):
    """A world point in a layer's simulation (fire-local) coordinates."""
    fp = np.asarray(scene.get(('camera', 'fire_position'), scene.start if frame is None else frame), float)
    yaw = math.radians(float(scene.get(('camera', 'fire_yaw'), scene.start if frame is None else frame)))
    d = np.asarray(p, float) - fp
    c, s = math.cos(-yaw), math.sin(-yaw)
    return np.array([c * d[0] + s * d[2], d[1], -s * d[0] + c * d[2]])


def to_world(scene, p, frame=None):
    fp = np.asarray(scene.get(('camera', 'fire_position'), scene.start if frame is None else frame), float)
    yaw = math.radians(float(scene.get(('camera', 'fire_yaw'), scene.start if frame is None else frame)))
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.asarray(p, float)
    return fp + np.array([c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2]])


def sync(scene):
    """Put each surface's collider where the surface is, relative to where this layer's effect stands now."""
    g = scene.ground or {}
    yaw = float(scene.get(('camera', 'fire_yaw'), scene.start))
    for srf in g.get('surfaces') or []:
        for c in scene.colliders:
            if c.get('shape') == 'mesh' and c.get('mesh') == srf['mesh']:
                c['position'] = tuple(float(x) for x in local(scene, srf['centre']))
                c['yaw'] = -yaw
                c['size'] = (1.0, 1.0, 1.0)


def collider_of(scene, srf):
    return dict(name=srf['name'], shape='mesh', mesh=srf['mesh'], position=tuple(float(x) for x in local(scene, srf['centre'])),
                size=(1.0, 1.0, 1.0), yaw=-float(scene.get(('camera', 'fire_yaw'), scene.start)), holdout=True)


def ground_hit(scene, eye, d, upward_only=True):
    """The nearest world point the ray meets on the ground or on a surface facing up (a table top, a step, a ramp),
    or None."""
    best, bt = None, np.inf
    gn, gp = ground_plane(scene)
    p = _hit(eye, d, gn, gp)
    if p is not None:
        best, bt = p, float(np.linalg.norm(p - eye))
    quads = []
    for srf in (scene.ground or {}).get('surfaces') or []:
        quads += srf.get('treads') or [srf['top']]
    for quad in quads:
        top = np.asarray(quad, float)
        n = np.cross(top[1] - top[0], top[2] - top[1])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if n[1] < 0:
            n = -n
        if upward_only and n[1] < 0.5:
            continue
        q = _hit(eye, d, n, top[0])
        if q is None:
            continue
        a1, a2, _ = _basis(n)
        poly = np.array([((c - top[0]) @ a1, (c - top[0]) @ a2) for c in top])
        from . import textmesh as T
        if T._inside(((q - top[0]) @ a1, (q - top[0]) @ a2), poly):
            dist = float(np.linalg.norm(q - eye))
            if dist < bt:
                best, bt = q, dist
    return best
