"""Working on several objects at once: move or turn them together, copy them, and repeat them in a row, a ring or
scattered over the ground. Things attached to them (scene.links) come along, and copies keep their attachments
among themselves. Animated places and turns are moved key by key, so a moving thing keeps its path."""
from __future__ import annotations

import copy
import math
import random
import re

import numpy as np

from .anim import Curve

LISTS = {'emitter': 'emitters', 'collider': 'colliders', 'light': 'lights', 'fabric': 'fabrics'}


def items(scene, kind):
    return getattr(scene, LISTS[kind])


def _turn_xz(v, theta):
    """A vector turned about the vertical by theta degrees (positive turns +x toward -z, as the objects' Rotation)."""
    c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    return (v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c)


def _place(v, pivot, theta, move):
    p = np.asarray(v, float) - pivot
    q = np.asarray(_turn_xz(p, theta), float) + pivot + move
    return tuple(float(x) for x in q)


def _map(v, fn):
    if isinstance(v, Curve):
        return Curve([[f, fn(x), it] for f, x, it in v.keys])
    return fn(v)


def _wrap(a):
    return (float(a) + 180.0) % 360.0 - 180.0


def transform(kind, d, pivot=(0.0, 0.0, 0.0), theta=0.0, move=(0.0, 0.0, 0.0)):
    """Turn an object about the vertical through `pivot` by theta degrees, then move it (in place, keys and all)."""
    pivot = np.asarray(pivot, float)
    move = np.asarray(move, float)
    for k in ('position', 'end'):
        if k in d:
            d[k] = _map(d[k], lambda x: _place(x, pivot, theta, move))
    if theta:
        if 'yaw' in d:   # keys are turned without wrapping, so the object does not spin the long way between them
            y = d['yaw']
            d['yaw'] = Curve([[f, float(a) + theta, it] for f, a, it in y.keys]) if isinstance(y, Curve) else _wrap(y + theta)
        for k in (('direction',) if kind == 'light' else ('velocity',) if kind == 'emitter' else ()):
            if k in d:
                d[k] = _map(d[k], lambda x: tuple(float(y) for y in _turn_xz(x, theta)))


def footprint(scene, sel, frame):
    """(min, max) corners of what the objects take up at a frame (lights by their place)."""
    from .components import _extent
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for kind, i in sel:
        d = dict(items(scene, kind)[i])
        for k in ('position', 'end', 'size'):
            if k in d and isinstance(d[k], Curve):
                d[k] = tuple(scene.get((kind, i, k), frame))
        if kind == 'light':
            a = b = np.asarray(d['position'], float)
        else:
            a, b = _extent(kind, d)
        lo, hi = np.minimum(lo, a), np.maximum(hi, b)
    return lo, hi


def copy_objects(scene, sel, pivot=(0.0, 0.0, 0.0), theta=0.0, move=(0.0, 0.0, 0.0), seed=1):
    """Copies of objects sel = [(kind, i)] (attached things included by the caller), turned and moved, added to the
    scene with names of their own and their attachments among themselves. Returns the copies' [(kind, i)]."""
    renamed = {}
    out = []
    for kind, i in sel:
        d = copy.deepcopy(items(scene, kind)[i])
        transform(kind, d, pivot, theta, move)
        if kind == 'emitter':   # each copy flickers in its own way
            d['seed'] = (int(d.get('seed', 0)) + 7 * seed) % 10000
        names = {o['name'] for o in items(scene, kind)}
        base = d['name']
        m = re.match(r'^(.*?)(?: (\d+))?$', base)
        stem, n = (m.group(1), int(m.group(2) or 1) + 1) if m else (base, 2)
        name = base
        while name in names:
            name = f'{stem} {n}'
            n += 1
        d['name'] = name
        renamed[(kind, base)] = name
        items(scene, kind).append(d)
        out.append((kind, len(items(scene, kind)) - 1))
    for l in list(getattr(scene, 'links', None) or []):
        c, p = tuple(l['child']), tuple(l['parent'])
        if c in renamed and p in renamed:
            scene.links.append(dict(l, child=[c[0], renamed[c]], parent=[p[0], renamed[p]]))
    return out


def layout(pattern, count, spacing=1.0, angle=0.0, radius=1.0, area=(2.0, 2.0), face_out=True, random_turn=True, seed=0):
    """Where the copies go, as [(dx, dz, turn)] about the original's middle: 'row' (count more, `spacing` apart along a
    direction `angle` degrees from +x), 'ring' (count in all, the original among them, on a circle of `radius` about
    where the original was) or 'scatter' (count more, at random over an area, `area` = (width, depth))."""
    out = []
    if pattern == 'row':
        a = math.radians(angle)
        for k in range(1, int(count) + 1):
            out.append((spacing * k * math.cos(a), -spacing * k * math.sin(a), 0.0))
    elif pattern == 'ring':
        n = max(2, int(count))
        for k in range(n):
            phi = 360.0 * k / n
            x, _, z = _turn_xz((radius, 0.0, 0.0), phi)
            out.append((x, z, phi if face_out else 0.0))
    elif pattern == 'scatter':
        rnd = random.Random(seed)
        w, dpt = area
        for k in range(int(count)):
            out.append((rnd.uniform(-w / 2, w / 2), rnd.uniform(-dpt / 2, dpt / 2), rnd.uniform(-180.0, 180.0) if random_turn else 0.0))
    return out
