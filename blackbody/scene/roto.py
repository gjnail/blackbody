"""Roto: shapes drawn over the footage that hide the effect behind them (a person walking in front of the
fire, a lamp post the smoke passes behind), as a holdout matte made in Blackbody.

A shape is a closed polygon in frame coordinates (0..1 across and down the picture), keyed over time:
its points move linearly between keys and hold before the first and after the last. Each shape has a
feather (a soft edge, in pixels of the output frame) and can be inverted (hide the effect everywhere but
inside it). The shapes are rasterised into the same holdout matte a matte sequence gives (io/holdout.py),
so they work in the viewer, in renders and on the command line.
"""
from __future__ import annotations

import bisect

import numpy as np


def new_shape(name='Roto 1'):
    return {'name': name, 'enabled': True, 'invert': False, 'feather': 2.0, 'keys': {}}


def keys_sorted(shape):
    return sorted((float(f), pts) for f, pts in shape.get('keys', {}).items())


def points_at(shape, frame):
    """The shape's points at a frame (list of [x, y]), or None if it has no keys."""
    ks = keys_sorted(shape)
    if not ks:
        return None
    frames = [k[0] for k in ks]
    if frame <= frames[0]:
        return [list(p) for p in ks[0][1]]
    if frame >= frames[-1]:
        return [list(p) for p in ks[-1][1]]
    i = bisect.bisect_right(frames, frame) - 1
    (f0, a), (f1, b) = ks[i], ks[i + 1]
    if len(a) != len(b):
        return [list(p) for p in a]
    t = (frame - f0) / max(f1 - f0, 1e-9)
    return [[pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t] for pa, pb in zip(a, b)]


def set_points(shape, frame, pts):
    """Key the shape's points at a frame (they keep the same count at every key: a point added to a shape is
    added at every key, where it sits between its neighbours)."""
    keys = shape.setdefault('keys', {})
    n = len(pts)
    for f, old in list(keys.items()):
        if len(old) != n and float(f) != float(frame):
            keys[f] = _resample(old, n)
    keys[_fkey(frame)] = [[float(x), float(y)] for x, y in pts]


def insert_point(shape, index, frame, xy):
    """Add a point after `index` at every key (between its neighbours there), placed at xy on this frame."""
    keys = shape.setdefault('keys', {})
    for f, pts in list(keys.items()):
        a, b = pts[index % len(pts)], pts[(index + 1) % len(pts)]
        pts.insert(index + 1, [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2])
    cur = points_at(shape, frame)
    cur[index + 1] = [float(xy[0]), float(xy[1])]
    keys[_fkey(frame)] = cur


def delete_point(shape, index):
    keys = shape.setdefault('keys', {})
    for f, pts in list(keys.items()):
        if len(pts) > 3:
            pts.pop(index)


def has_key(shape, frame):
    return _fkey(frame) in shape.get('keys', {})


def remove_key(shape, frame):
    shape.get('keys', {}).pop(_fkey(frame), None)


def _fkey(frame):
    return str(int(round(frame)))


def _resample(pts, n):
    """A closed polygon resampled to n points along its outline (when point counts differ between keys)."""
    p = np.asarray(pts, float)
    if len(p) == 0:
        return [[0.0, 0.0]] * n
    q = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, s[-1], n, endpoint=False) if s[-1] > 0 else np.zeros(n)
    x = np.interp(t, s, q[:, 0])
    y = np.interp(t, s, q[:, 1])
    return [[float(a), float(b)] for a, b in zip(x, y)]


def rasterise(shapes, frame, size, out_size=None):
    """The matte (h, w) 0..1 of the enabled shapes at a frame, at size (w, h). The feather is in pixels of
    out_size (the output frame), scaled to this size. None if no shape has points."""
    from PIL import Image, ImageDraw, ImageFilter
    w, h = int(size[0]), int(size[1])
    ow = (out_size or size)[0]
    total = None
    for sh in shapes or []:
        if not sh.get('enabled', True):
            continue
        pts = points_at(sh, frame)
        if not pts or len(pts) < 3:
            continue
        img = Image.new('L', (w, h), 0)
        ImageDraw.Draw(img).polygon([(x * w, y * h) for x, y in pts], fill=255)
        r = float(sh.get('feather', 0.0)) * w / max(ow, 1)
        if r > 0.3:
            img = img.filter(ImageFilter.GaussianBlur(r))
        m = np.asarray(img, np.float32) / 255.0
        if sh.get('invert'):
            m = 1.0 - m
        total = m if total is None else np.maximum(total, m)
    return total
