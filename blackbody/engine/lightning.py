"""Lightning: a bolt (or an arc) from a light's Position to the point it strikes. It is drawn as a branching channel on
the stage (stage.py, as glowing segments) and lights the set as lamps along it (Scene.lamps), through its flash.

The channel is jagged at every scale, as lightning is: the line from the top to the strike point, its midpoints moved
aside again and again (midpoint displacement), each time by a share of the piece's length. Branches leave it in its
upper part, angled down and away, jagged the same way, with branches of their own; they only light up in the first
stroke (the stepped leader's channels), the later strokes going down the main channel alone.

The flash is a few return strokes, each a sharp peak that fades in tens of milliseconds, with a dim glow between them
(the continuing current). A frame shows the flash averaged over its exposure.
"""
from __future__ import annotations

import math

import numpy as np

JAG = 0.13          # how far a midpoint moves aside, as a share of its piece's length (tortuosity)
DECAY = 0.018       # s: a stroke fades by e in this
GLOW = 0.04         # the continuing current's glow between strokes, as a share of a stroke's peak


def _jagged(a, b, levels, rng, jag=JAG):
    """Points from a to b with their midpoints moved aside `levels` times."""
    pts = np.array([a, b], float)
    for _ in range(levels):
        d = pts[1:] - pts[:-1]
        ln = np.linalg.norm(d, axis=1, keepdims=True)
        u = d / np.maximum(ln, 1e-12)
        # a direction across each piece, at random round it
        ref = np.where(np.abs(u[:, 1:2]) < 0.9, np.array([[0.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]))
        s1 = np.cross(u, ref)
        s1 /= np.maximum(np.linalg.norm(s1, axis=1, keepdims=True), 1e-12)
        s2 = np.cross(u, s1)
        ang = rng.random(len(d)) * 2.0 * math.pi
        side = s1 * np.cos(ang)[:, None] + s2 * np.sin(ang)[:, None]
        mid = 0.5 * (pts[1:] + pts[:-1]) + side * (ln * jag * rng.normal(0.0, 1.0, (len(d), 1)))
        out = np.empty((2 * len(pts) - 1, 3))
        out[0::2] = pts
        out[1::2] = mid
        pts = out
    return pts


def bolt(top, bottom, seed=0, branching=0.5, levels=7):
    """The channel from `top` to `bottom` (fire-local m): [(points (n, 3), thickness (share of the main channel's),
    brightness (share of it), first stroke only)]."""
    rng = np.random.default_rng(int(seed) * 7919 + 17)
    top, bottom = np.asarray(top, float), np.asarray(bottom, float)
    main = _jagged(top, bottom, levels, rng)
    out = [(main, 1.0, 1.0, False)]
    length = float(np.linalg.norm(bottom - top))
    if length <= 1e-6 or branching <= 0.0:
        return out
    down = (bottom - top) / length

    def branches(path, n, scale, thick, bright, depth):
        if depth > 2 or n <= 0:
            return
        k = len(path)
        for _ in range(n):
            # leaving the upper two thirds of its parent, down and away from it
            i = int(rng.uniform(0.05, 0.7) * (k - 2))
            p = path[i]
            here = path[min(i + 2, k - 1)] - path[max(i - 2, 0)]
            here = here / max(float(np.linalg.norm(here)), 1e-12)
            ref = np.array([0.0, 1.0, 0.0]) if abs(here[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
            s1 = np.cross(here, ref)
            s1 /= max(float(np.linalg.norm(s1)), 1e-12)
            s2 = np.cross(here, s1)
            az = rng.random() * 2.0 * math.pi
            tilt = math.radians(rng.uniform(20.0, 55.0))
            d = here * math.cos(tilt) + (s1 * math.cos(az) + s2 * math.sin(az)) * math.sin(tilt)
            d = d + 0.35 * down
            d /= max(float(np.linalg.norm(d)), 1e-12)
            ln = length * scale * rng.uniform(0.35, 1.0) * (1.0 - i / k)
            q = _jagged(p, p + d * ln, max(levels - 2 - depth, 3), rng, JAG * 1.2)
            out.append((q, thick, bright, True))
            branches(q, int(round(n * 0.5)), scale * 0.45, thick * 0.6, bright * 0.6, depth + 1)

    branches(main, int(round(1 + 6 * branching)), 0.35, 0.55, 0.45, 0)
    return out


def strokes(strike_at, count, seed=0):
    """When each return stroke is (s) and how bright its peak is (the first brightest)."""
    rng = np.random.default_rng(int(seed) * 104729 + 3)
    t = float(strike_at)
    out = []
    for k in range(max(int(count), 1)):
        out.append((t, 1.0 if k == 0 else float(rng.uniform(0.45, 0.9))))
        t += float(rng.uniform(0.035, 0.11))
    return out


def brightness(t, strike_at, count, seed=0, exposure=0.0, samples=16):
    """How bright the flash is at time t (s) (a stroke's peak: 1), averaged over the `exposure` (s) round t, and
    whether the branches show (the first stroke's)."""
    ss = strokes(strike_at, count, seed)
    last = ss[-1][0]
    ts = np.linspace(t - 0.5 * exposure, t + 0.5 * exposure, samples) if exposure > 0.0 else np.array([t])
    b = np.zeros(len(ts))
    first = np.zeros(len(ts))
    for k, (tk, peak) in enumerate(ss):
        after = ts >= tk
        e = np.where(after, peak * np.exp(-np.clip(ts - tk, 0.0, None) / DECAY), 0.0)
        b += e
        if k == 0:
            first += e
    # the continuing current: a dim glow from the first stroke to a little after the last
    glow = (ts >= ss[0][0]) & (ts <= last + 3.0 * DECAY)
    b += GLOW * glow
    b, first = float(b.mean()), float(first.mean())
    return (b if b > 1e-6 else 0.0), (first if first > 1e-6 else 0.0)     # (its tail, long after: dark)


_BOLTS = {}


def cached_bolt(top, bottom, seed=0, branching=0.5):
    """bolt(), kept (it is drawn and lights the set every frame of its flash)."""
    key = (tuple(round(float(x), 5) for x in top), tuple(round(float(x), 5) for x in bottom), int(seed), round(float(branching), 4))
    got = _BOLTS.get(key)
    if got is None:
        if len(_BOLTS) > 32:
            _BOLTS.clear()
        got = _BOLTS[key] = bolt(top, bottom, seed, branching)
    return got
