"""The transform gizmo on the selected object: arrows to move it along X, Y and Z, squares to stretch it
along its own axes, a ring to turn it, and its middle to slide it along the ground. It keeps the same size
on screen however near or far the object is. Ctrl snaps to a grid sized for the scene.

Everything is in the effect's own (fire-local) metres, like the scene."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF

from ..scene import kinds as K


AXES = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
AXIS_NAMES = ('x', 'y', 'z')
AXIS_COLOURS = ('#ef5b5b', '#6fd06a', '#5b9cff')
SCREEN_LEN = 78.0   # pixels


def _rot_y(v, yaw_deg):
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    x, y, z = v
    return np.array([c * x + s * z, y, -s * x + c * z])


def nice_step(x):
    """A round grid step near x (1, 2 or 5 times a power of ten)."""
    if x <= 0:
        return 0.1
    e = 10 ** math.floor(math.log10(x))
    for m in (1, 2, 5, 10):
        if m * e >= x:
            return m * e
    return 10 * e


class Gizmo:
    """The gizmo of one object (kind, index) as the viewport sees it now."""

    def __init__(self, vp, kind, i):
        self.vp = vp
        self.kind = kind
        self.i = i
        sc = vp.doc.scene
        self.item = K.items(sc, kind)[i]
        g = lambda k: sc.get((kind, i, k), vp.doc.frame)
        self.g = g
        self.pos = np.asarray(g('position'), float)
        self.yaw = float(g('yaw')) if kind in ('emitter', 'collider', 'fabric', 'matter') else 0.0
        self.cs, self.fire, _ = vp.camstate()
        self.W, self.H = vp.out_size()
        self.capsule = kind == 'emitter' and self.item.get('shape') == 'capsule'
        self.L = self._length()

    # -- geometry -------------------------------------------------------------------------------------------

    def screen(self, pts):
        px, ok = self.vp._project_local(self.cs, self.fire, pts)
        return [self.vp.to_widget(q) for q in px], ok

    def _length(self):
        """Metres that make SCREEN_LEN pixels at the object (along the axis that shows longest)."""
        pts = [self.pos] + [self.pos + a * 0.1 for a in AXES]
        s, ok = self.screen(pts)
        if not ok.all():
            return 0.3
        d = max(math.hypot((q - s[0]).x(), (q - s[0]).y()) for q in s[1:]) / 0.1
        return SCREEN_LEN / max(d, 1e-6)

    def handles(self):
        """Screen positions: 'move_x/y/z' (shaft from, tip), 'scale_x/y/z' (square), 'rot' (knob), 'centre', and the
        ring (local polyline)."""
        out = {}
        c = self.pos
        tips = [c + a * self.L for a in AXES]
        s, ok = self.screen([c] + tips)
        out['centre'] = s[0] if ok[0] else None
        for k, name in enumerate(AXIS_NAMES):
            if ok[0] and ok[k + 1]:
                out['move_' + name] = (s[0], s[k + 1])
        for name, local in self.scale_axes():
            q, okq = self.screen([c + local * self.L * 0.62])
            if okq[0]:
                out['scale_' + name] = q[0]
        turns = not (self.kind == 'matter' and (self.item.get('pours') or self.item.get('shape') != 'box'))
        if self.kind in ('emitter', 'collider', 'fabric', 'matter') and not self.capsule and turns:
            r = self.L * 0.85
            knob = c + _rot_y(np.array([r, 0.0, 0.0]), self.yaw + 45.0)   # between the arrows, clear of them
            q, okq = self.screen([knob])
            if okq[0]:
                out['rot'] = q[0]
            ang = np.linspace(0, 2 * math.pi, 65)
            out['ring'] = np.stack([c + np.array([r * math.cos(a), 0.0, r * math.sin(a)]) for a in ang])
        return out

    def scale_axes(self):
        """(name, local unit axis) of the stretchable axes of this object."""
        k = self.kind
        if k in ('emitter', 'collider') and not self.capsule:
            shape = self.item.get('shape')
            if shape == 'sphere' and k == 'collider':
                return [('x', _rot_y(AXES[0], self.yaw))]
            if shape in ('cylinder', 'cone', 'ring'):
                return [('x', _rot_y(AXES[0], self.yaw)), ('y', AXES[1])]
            return [('x', _rot_y(AXES[0], self.yaw)), ('y', AXES[1]), ('z', _rot_y(AXES[2], self.yaw))]
        if self.capsule:
            return [('x', _rot_y(AXES[0], 0.0))]
        if k == 'matter':
            shape = self.item.get('shape')
            if self.item.get('pours') or shape == 'sphere':   # a nozzle's or a ball's radius
                return [('x', AXES[0])]
            if shape in ('cylinder', 'pile'):
                return [('x', AXES[0]), ('y', AXES[1])]
            return [('x', _rot_y(AXES[0], self.yaw)), ('y', AXES[1]), ('z', _rot_y(AXES[2], self.yaw))]
        if k == 'fabric':
            if self.item.get('shape') == 'mesh':
                return [('x', _rot_y(AXES[0], self.yaw)), ('y', AXES[1]), ('z', _rot_y(AXES[2], self.yaw))]
            if self.item.get('orientation') == 'lying':
                return [('x', _rot_y(AXES[0], self.yaw)), ('z', _rot_y(AXES[2], self.yaw))]
            return [('x', _rot_y(AXES[0], self.yaw)), ('y', AXES[1])]
        return []

    def hit(self, pos):
        hs = self.handles()
        best, bd = None, 9.0
        for key, v in hs.items():
            if key.startswith('scale_') or key == 'rot':
                d = math.hypot((v - pos).x(), (v - pos).y())
                if d < bd:
                    best, bd = key, d
        if best is not None:
            return best
        for key, v in hs.items():
            if key.startswith('move_'):
                a, b = v
                if _seg_dist(pos, a, b) < 7.0 and math.hypot((pos - a).x(), (pos - a).y()) > 10:
                    return key
        c = hs.get('centre')
        if c is not None and math.hypot((c - pos).x(), (c - pos).y()) < 9:
            return 'centre'
        return None

    # -- dragging ---------------------------------------------------------------------------------------------

    def axis_param(self, pos, origin, axis):
        return self.vp._axis_param(pos, origin, axis)

    def snap(self):
        sc = self.vp.doc.scene
        return nice_step(max(sc.domain_size()) / 40.0)


def _seg_dist(p, a, b):
    ab = b - a
    L2 = ab.x() ** 2 + ab.y() ** 2
    if L2 < 1e-9:
        return math.hypot((p - a).x(), (p - a).y())
    t = max(0.0, min(1.0, ((p - a).x() * ab.x() + (p - a).y() * ab.y()) / L2))
    q = a + ab * t
    return math.hypot((p - q).x(), (p - q).y())


def arrow_head(a: QPointF, b: QPointF, size=8.0):
    """Points of a filled arrow head at b, pointing away from a."""
    dx, dy = b.x() - a.x(), b.y() - a.y()
    n = math.hypot(dx, dy) or 1.0
    ux, uy = dx / n, dy / n
    return [b + QPointF(ux * 3, uy * 3), b + QPointF(-ux * size + -uy * size * 0.5, -uy * size + ux * size * 0.5),
            b + QPointF(-ux * size - -uy * size * 0.5, -uy * size - ux * size * 0.5)]
