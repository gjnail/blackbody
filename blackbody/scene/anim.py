"""Keyframe curves for animated parameters."""
from __future__ import annotations

import bisect
import math

INTERPS = ('smooth', 'linear', 'step')


def _lerp(a, b, t):
    if isinstance(a, (list, tuple)):
        return type(a)(x + (y - x) * t for x, y in zip(a, b))
    return a + (b - a) * t


def _hermite(p0, p1, m0, m1, t):
    t2, t3 = t * t, t * t * t
    return (2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * p1 + (t3 - t2) * m1


class Curve:
    """Keys are (frame, value, interpolation-to-next). Values may be floats or tuples."""

    __slots__ = ('keys',)

    def __init__(self, keys=None):
        self.keys = sorted([list(k) for k in (keys or [])], key=lambda k: k[0])
        for k in self.keys:
            if len(k) < 3:
                k.append('smooth')

    def to_json(self):
        return {'keys': [[float(f), list(v) if isinstance(v, (list, tuple)) else v, i] for f, v, i in self.keys]}

    @classmethod
    def from_json(cls, d):
        keys = []
        for f, v, *rest in d.get('keys', []):
            keys.append([f, tuple(v) if isinstance(v, list) else v, rest[0] if rest else 'smooth'])
        return cls(keys)

    def frames(self):
        return [k[0] for k in self.keys]

    def set(self, frame, value, interp=None):
        frame = float(frame)
        for k in self.keys:
            if abs(k[0] - frame) < 1e-6:
                k[1] = value
                if interp:
                    k[2] = interp
                return
        i = bisect.bisect_left(self.frames(), frame)
        self.keys.insert(i, [frame, value, interp or 'smooth'])

    def remove(self, frame):
        self.keys = [k for k in self.keys if abs(k[0] - frame) > 1e-6]

    def has_key(self, frame):
        return any(abs(k[0] - frame) < 1e-6 for k in self.keys)

    def eval(self, frame):
        ks = self.keys
        if not ks:
            return None
        if frame <= ks[0][0]:
            return ks[0][1]
        if frame >= ks[-1][0]:
            return ks[-1][1]
        i = bisect.bisect_right([k[0] for k in ks], frame) - 1
        f0, v0, mode = ks[i]
        f1, v1, _ = ks[i + 1]
        t = (frame - f0) / max(f1 - f0, 1e-9)
        if mode == 'step':
            return v0
        if mode == 'linear' or isinstance(v0, (list, tuple, bool, str)):
            return v0 if isinstance(v0, (bool, str)) else _lerp(v0, v1, t)
        # smooth: cubic Hermite with Catmull-Rom tangents, flattened at extremes to avoid overshoot
        fp, vp = (ks[i - 1][0], ks[i - 1][1]) if i > 0 else (f0, v0)
        fn, vn = (ks[i + 2][0], ks[i + 2][1]) if i + 2 < len(ks) else (f1, v1)
        span = f1 - f0

        def tangent(pa, fa, pb, fb, here):
            if fb == fa:
                return 0.0
            m = (pb - pa) / (fb - fa) * span
            return m

        m0 = tangent(vp, fp, v1, f1, v0) if i > 0 else (v1 - v0)
        m1 = tangent(v0, f0, vn, fn, v1) if i + 2 < len(ks) else (v1 - v0)
        if i > 0 and (v0 - vp) * (v1 - v0) <= 0:
            m0 = 0.0
        if i + 2 < len(ks) and (v1 - v0) * (vn - v1) <= 0:
            m1 = 0.0
        return _hermite(v0, v1, m0, m1, t)


def is_curve(v):
    return isinstance(v, Curve)
