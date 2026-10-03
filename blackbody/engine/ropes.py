"""Ropes and springs as they are drawn: the curve a rope hangs in, a spring's coil, and the thin straight segments
the stage traces them as (each a convex prism, like a broken object's piece).

The physics of a rope is only a limit on the distance between its ends (solids.py); its shape in between is
worked out here from that distance and its length: straight when it is taut, a catenary when it is slack.
"""
from __future__ import annotations

import math

import numpy as np

ROPE, CABLE, SPRING, CHAIN = 0, 1, 2, 3          # looks (Solids.rope_poses)
SIDES = 8                              # a segment's cross-section: a regular octagon round the rope's radius
UP = np.array([0.0, 1.0, 0.0])


def num(x):
    """A number from a rope entry's value (a float, or an array of one: the disk cache keeps them as arrays)."""
    return float(np.asarray(x, float).ravel()[0])


def _sinhc_inverse(r):
    """u > 0 with sinh(u) / u = r (r > 1)."""
    u = math.sqrt(6.0 * (r - 1.0)) if r < 3.0 else math.log(2.0 * r) + math.log(math.log(2.0 * r) + 1.0)
    for _ in range(40):
        su = math.sinh(u)
        f = su / u - r
        df = (math.cosh(u) * u - su) / (u * u)
        step = f / df
        u = max(u - step, 0.5 * u)
        if abs(step) < 1e-12 * u:
            break
    return u


def rope_curve(a, b, length, n):
    """n + 1 points along a rope of `length` between a and b, evenly spaced along it: a catenary in the vertical
    plane through a and b when it is slack, a straight line when it is taut."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = np.linspace(0.0, 1.0, n + 1)
    d = b - a
    chord = float(np.linalg.norm(d))
    if length <= chord * 1.0005 or length <= 1e-6:
        return a + t[:, None] * d
    h = np.array([d[0], 0.0, d[2]])
    span = float(np.linalg.norm(h))
    across = h / span if span > 1e-9 else np.array([1.0, 0.0, 0.0])
    # (straight up and down it would fold onto itself: it is worked out a little apart, as a narrow loop)
    span = max(span, 0.01 * length)
    v = float(d[1])
    s = math.sqrt(max(length * length - v * v, 1e-18))
    if s <= span * (1.0 + 1e-9):
        return a + t[:, None] * d
    u = _sinhc_inverse(s / span)
    c = span / (2.0 * u)
    x0 = 0.5 * span - c * math.atanh(max(min(v / length, 1.0 - 1e-12), -1.0 + 1e-12))
    # equal steps along its length from a: arc length from x = 0 is c (sinh((x - x0) / c) + sinh(x0 / c))
    s0 = math.sinh(-x0 / c)
    x = x0 + c * np.arcsinh(t * length / c + s0)
    y = c * (np.cosh((x - x0) / c) - math.cosh(x0 / c))
    pts = a + np.outer(x, across) + np.outer(y, UP)
    return pts + np.outer(t, b - pts[-1])     # (the loop opened up ends at b again)


def coil(a, b, rest, wire, n_per_turn=12):
    """Points along a coil spring from a to b: about one turn per two coil radii of its length at rest, the coil's
    radius six times the wire's (at least 2 cm)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    ln = float(np.linalg.norm(d))
    ax = d / ln if ln > 1e-9 else UP.copy()
    R = max(6.0 * wire, 0.02)
    turns = max(3, int(round(rest / (2.0 * R))))
    side = np.cross(ax, UP if abs(ax[1]) < 0.9 else np.array([1.0, 0.0, 0.0]))
    side /= np.linalg.norm(side)
    up = np.cross(side, ax)
    # a straight lead from each end to the coil, a tenth of the length each
    lead = 0.1
    m = turns * n_per_turn
    th = np.linspace(0.0, 2.0 * math.pi * turns, m + 1)
    f = lead + (1.0 - 2.0 * lead) * np.linspace(0.0, 1.0, m + 1)
    ring = R * (np.outer(np.cos(th), side) + np.outer(np.sin(th), up))
    body = a + np.outer(f * ln, ax) + ring
    return np.concatenate([a[None], body, b[None]])


def chain_links(pts, wire):
    """A chain's links along its polyline (one link a segment), each a ring of four bars (its two sides and two ends),
    each turned a quarter round from the last, as links interlock: [(centre, orientation (x, y, z, w) turning y onto
    the bar, half length, bar radius)]. wire: the thickness of a link's wire."""
    out = []
    r = 0.5 * wire
    for k in range(len(pts) - 1):
        p0, p1 = np.asarray(pts[k], float), np.asarray(pts[k + 1], float)
        d = p1 - p0
        ln = float(np.linalg.norm(d))
        if ln < 1e-7:
            continue
        y = d / ln
        side = np.cross(y, [0.0, 1.0, 0.0] if abs(y[1]) < 0.9 else [1.0, 0.0, 0.0])
        side /= np.linalg.norm(side)
        if k % 2:
            side = np.cross(y, side)
        c = 0.5 * (p0 + p1)
        half_w = 1.6 * wire
        # its two sides, along it (a little longer than a segment: links overlap where they hook)
        for s in (-1.0, 1.0):
            out.append((c + s * half_w * side, _quat_onto_y(y), 0.5 * ln + 1.2 * wire, r))
        # its two ends, across it
        for e in (p0 - 0.6 * wire * y, p1 + 0.6 * wire * y):
            out.append((e, _quat_onto_y(side), half_w + r, r))
    return out


def rope_points(rope, ground=None, n=None):
    """The polyline a rope (Solids.rope_poses entry) is drawn along and the velocity of each of its points:
    (points (m, 3), velocities (m, 3)). A snapped rope hangs from its anchor end, half its length (down to the
    ground, `ground` the ground's height or None)."""
    a, b = np.asarray(rope['a'], float), np.asarray(rope['b'], float)
    va, vb = np.asarray(rope['va'], float), np.asarray(rope['vb'], float)
    length = num(rope['length'])
    look = int(round(num(rope['look'])))
    path = rope.get('path')
    if path is not None and len(path) >= 2 and num(rope.get('broken', 0.0)) < 0.5:
        # (wrapped round a post: along its path, taut)
        path = np.asarray(path, float)
        t = np.clip(np.linspace(0.0, 1.0, len(path)), 0.0, 1.0)
        return path, va + t[:, None] * (vb - va)
    if look == SPRING:
        if num(rope.get('broken', 0.0)) > 0.5:
            return np.zeros((0, 3)), np.zeros((0, 3))
        pts = coil(a, b, length, num(rope['radius']))
        t = np.clip(np.linalg.norm(pts - a, axis=1) / max(float(np.linalg.norm(b - a)), 1e-9), 0.0, 1.0)
        return pts, va + t[:, None] * (vb - va)
    if num(rope.get('broken', 0.0)) > 0.5:
        drop = 0.5 * length
        if ground is not None:
            drop = min(drop, max(b[1] - float(ground) - num(rope['radius']), 0.0))
        if drop <= 1e-3:
            return np.zeros((0, 3)), np.zeros((0, 3))
        k = max(2, int(math.ceil(drop / 0.25)))
        pts = b + np.outer(np.linspace(0.0, -drop, k + 1), UP)
        return pts, np.tile(vb, (k + 1, 1))
    if n is None:
        n = int(np.clip(math.ceil(length / 0.15), 8, 48))
    pts = rope_curve(a, b, length, n)
    t = np.linspace(0.0, 1.0, n + 1)
    return pts, va + t[:, None] * (vb - va)


def segments(pts, vel, radius):
    """The straight pieces a polyline is drawn as: [(centre, orientation (x, y, z, w) turning y onto the segment,
    velocity, half length, its middle's distance along the line)]. Each reaches a little past its ends, so the bends
    leave no gaps."""
    out = []
    along = 0.0
    for k in range(len(pts) - 1):
        p0, p1 = pts[k], pts[k + 1]
        d = p1 - p0
        ln = float(np.linalg.norm(d))
        if ln < 1e-7:
            continue
        y = d / ln
        out.append((0.5 * (p0 + p1), _quat_onto_y(y), 0.5 * (vel[k] + vel[k + 1]), 0.5 * ln + 0.5 * radius, along + 0.5 * ln))
        along += ln
    return out


def prism_planes(radius, half):
    """A segment's planes in its own frame (n . x <= d): SIDES round the y axis at `radius`, and its two ends."""
    th = 2.0 * math.pi * (np.arange(SIDES) + 0.5) / SIDES
    side = np.stack([np.cos(th), np.zeros(SIDES), np.sin(th), np.full(SIDES, radius)], 1)
    ends = np.array([[0.0, 1.0, 0.0, half], [0.0, -1.0, 0.0, half]])
    return np.concatenate([side, ends])


def _quat_onto_y(y):
    """The rotation (x, y, z, w) that turns the y axis onto unit vector y."""
    c = float(np.dot(UP, y))
    if c > 1.0 - 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if c < -1.0 + 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    ax = np.cross(UP, y)
    s = math.sqrt(2.0 * (1.0 + c))
    return np.array([ax[0] / s, ax[1] / s, ax[2] / s, 0.5 * s])
