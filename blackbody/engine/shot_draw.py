"""Drawing what bullets do (ballistics.py Ballistics.view) on the stage (stage.py, stage.wgsl): the debris as small
convex pieces among the broken objects' pieces, the bullets in flight as short copper prisms (the stage blurs them
along their way over the shutter), and sparks, tracers and muzzle flashes as segments that glow, as lightning's do.
The marks (holes, dents, craters, cracks) go to marks.wgsl as a buffer of their own.
"""
from __future__ import annotations

import math

import numpy as np

from .ropes import _quat_onto_y, prism_planes

# material rows past the objects', ropes' and lightning's (stage.py SHOT_ROW on): what is not an object's
BULLET_LOOK = ((0.78, 0.47, 0.2), 0.28, 1.0)     # gilding metal (a copper jacket): colour, roughness, metal
LEAD_LOOK = ((0.33, 0.33, 0.35), 0.55, 1.0)      # lead, splashed
SPARK_CORE = 0.00025     # m: a spark's glowing core's radius
TRACER_CORE = 0.0025     # m: a tracer's burning compound, seen end on (its glare makes it look wider)
TRACER_GLOW = (60.0, 6.0, 1.5)     # its radiance (linear rgb): a red-orange flare
FLASH_GLOW = (40.0, 22.0, 6.0)     # a muzzle flash's (a brief, hot yellow-orange)
MARKS_MOST = 256
BORES_MOST = 128


def debris_rows(view, row_of, ground_row, lead_row, shutter=0.0):
    """The debris's solid bits as the stage's pieces: [(P entry (6, 4), planes (k, 4), centre, bound radius)]. A bit of
    an object takes its material's row (its corner cuts drawn in the inside colour), the ground's the ground row, lead
    the lead row. Sparks are left for glow_rows."""
    out = []
    db = view.get('debris') if view else None
    if not db:
        return out
    from .debris import SPARK
    pos, quat, vel, omega = db['pos'], db['quat'], db['vel'], db['omega']
    for k in range(len(pos)):
        kind = int(db['kind'][k])
        if kind == SPARK:
            continue
        r = int(db['row'][k])
        row = row_of.get(r) if r >= 0 else (ground_row if r == -1 else lead_row)
        if row is None:
            continue
        pl = np.asarray(db['planes'][k], np.float32).copy()
        inner = np.arange(len(pl)) >= 6               # (the corners cut off it: broken faces)
        pl[inner, :3] *= 2.0
        rad = float(np.max(np.abs(pl[:6, 3]))) * 1.8
        v = np.asarray(vel[k], float)
        seed = (k * 0.618034) % 1.0
        out.append((np.array([[*pos[k], len(pl)], [*quat[k]], [*v, 0.0], [*omega[k], row],
                              [seed * 0.37, seed * 0.71, seed * 0.13, rad], [0.0, 0.0, 0.0, 0.0]], np.float32),
                    pl, np.asarray(pos[k], float), rad + float(np.linalg.norm(v)) * 0.5 * shutter))
    return out


def bullet_rows(view, bullet_row, shutter=0.0):
    """The bullets in flight: a prism as long and as wide as each, along its way, moving at its speed."""
    out = []
    B = view.get('bullets') if view else None
    if B is None:
        return out
    for b in np.asarray(B, float):
        pos, cal, vel, length = b[0:3], b[3], b[4:7], b[7]
        sp = float(np.linalg.norm(vel))
        if sp < 1e-9:
            continue
        q = _quat_onto_y(vel / sp)
        if sp < 0.01:
            vel = np.zeros(3)       # (lodged: still, pointing the way it went)
        pl = prism_planes(0.5 * cal, 0.5 * length).astype(np.float32)
        bound = math.hypot(0.5 * length, 0.5 * cal)
        out.append((np.array([[*pos, len(pl)], [*q], [*vel, 0.0], [0.0, 0.0, 0.0, bullet_row], [0.0, 0.0, 0.0, bound],
                              [0.0] * 4], np.float32), pl, pos, bound + sp * 0.5 * shutter))
    return out


def glow_rows(view, glow_row, shutter=0.0, frame_dt=1.0 / 24.0):
    """Segments that glow: sparks (along their way, blurred over the shutter), tracers (their burning trail over the
    last few milliseconds), muzzle flashes (a short star of flame out of the muzzle, as bright as it is over the
    frame: a flash lasts about a millisecond)."""
    out = []
    if not view:
        return out
    db = view.get('debris')
    if db:
        from .debris import SPARK
        glow = db['glow']
        for k in np.nonzero(np.asarray(db['kind']) == SPARK)[0]:
            g = np.asarray(glow[k], float)
            if g.max() <= 1e-3:
                continue
            v = np.asarray(db['vel'][k], float)
            sp = float(np.linalg.norm(v))
            d = v / sp if sp > 1e-6 else np.array([0.0, 1.0, 0.0])
            # (drawn as the streak it leaves over the shutter, whole in every sample: as long as it flies in that time,
            # each point of it lit for that share of the exposure)
            run = sp * max(shutter, 0.0015)
            half = max(SPARK_CORE * 2.0, 0.5 * run)
            g = g * min(1.0, 0.02 / max(run, 1e-6))
            pl = prism_planes(SPARK_CORE, half).astype(np.float32)
            c = np.asarray(db['pos'][k], float) - 0.5 * v * shutter
            out.append((np.array([[*c, len(pl)], [*_quat_onto_y(d)], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, glow_row],
                                  [*g, math.hypot(half, SPARK_CORE)], [0.0] * 4], np.float32), pl, c, half))
    for pts in view.get('tracers') or []:
        pts = np.asarray(pts, float)
        for a, b in zip(pts[:-1], pts[1:]):
            d = b - a
            ln = float(np.linalg.norm(d))
            if ln < 1e-6:
                continue
            half = 0.5 * ln
            pl = prism_planes(TRACER_CORE, half).astype(np.float32)
            c = 0.5 * (a + b)
            out.append((np.array([[*c, len(pl)], [*_quat_onto_y(d / ln)], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, glow_row],
                                  [*TRACER_GLOW, math.hypot(half, TRACER_CORE)], [0.0] * 4], np.float32), pl, c, half))
    F = view.get('flashes')
    if F is not None:
        rng = np.random.default_rng(7)
        for f in np.asarray(F, float):
            pos, strength, d, muzzle = f[0:3], f[3], f[4:7], f[7] > 0.5
            # (a flash lasts about a millisecond: over a frame's shutter it is that much of the time as bright)
            share = min(1.0, 0.0015 / max(shutter, frame_dt * 0.5))
            g = np.asarray(FLASH_GLOW if muzzle else (30.0, 30.0, 26.0), float) * strength * share * 6.0
            n_petal = 6 if muzzle else 4
            dn = d / max(float(np.linalg.norm(d)), 1e-9)
            t1 = np.cross(dn, [0.0, 1.0, 0.0] if abs(dn[1]) < 0.9 else [1.0, 0.0, 0.0])
            t1 /= max(float(np.linalg.norm(t1)), 1e-9)
            t2 = np.cross(dn, t1)
            for p in range(n_petal + 1):
                if p == 0:
                    ax, ln, core = dn, (0.09 if muzzle else 0.02) * strength ** 0.5, 0.006
                else:
                    ang = 2.0 * math.pi * p / n_petal + rng.uniform(-0.3, 0.3)
                    ax = dn * 0.55 + (t1 * math.cos(ang) + t2 * math.sin(ang)) * 0.85
                    ax /= np.linalg.norm(ax)
                    ln, core = (0.045 if muzzle else 0.015) * strength ** 0.5 * rng.uniform(0.6, 1.2), 0.003
                c = pos + ax * 0.5 * ln
                pl = prism_planes(core, 0.5 * ln).astype(np.float32)
                out.append((np.array([[*c, len(pl)], [*_quat_onto_y(ax)], [0.0] * 4, [0.0, 0.0, 0.0, glow_row],
                                      [*g, math.hypot(0.5 * ln, core)], [0.0] * 4], np.float32), pl, c, 0.5 * ln))
    return out


def marks_buffer(view):
    """The marks, the bores and the splinters for marks.wgsl: a header (how many marks, how many bores, where the bores
    start, in vec4, how many wood exits have splinters), then four vec4 a mark (the newest MARKS_MOST), four a bore,
    one each exit (splinters), then three each splinter; float32."""
    M = view.get('marks') if view else None
    B = view.get('bores') if view else None
    M = np.zeros((0, 4, 4), np.float32) if M is None else np.asarray(M, np.float32)[-MARKS_MOST:]
    B = np.zeros((0, 4, 4), np.float32) if B is None else np.asarray(B, np.float32)[-BORES_MOST:]
    if not len(M) and not len(B):
        return np.zeros(4, np.float32)
    first = 1 + 4 * len(M)
    scoops, spl = splinters(B, first)
    head = np.array([len(M), len(B), first, len(scoops)], np.float32)
    s0 = first + 4 * len(B) + len(scoops)
    S = np.asarray(scoops, np.float32).reshape(-1, 4)
    if len(S):
        S[:, 1] += s0
    return np.concatenate([head, M.reshape(-1), B.reshape(-1), S.reshape(-1),
                           np.asarray(spl, np.float32).reshape(-1)])


SPLINTERS = 22           # splinters standing round a wood exit (some left out)
BORE_LIP = 0.0015        # m: a bore starts and ends this far outside the surface it goes through (ballistics.py _carve)


def splinters(B, first=0):
    """The splinters standing out of the back of a board where bullets came out of it, for each of the bores that is
    such a scoop (wood, along its grain, in an object: marks.wgsl is_scoop): strips of the face along the grain, held at
    their far end, their free end at the split lifted out toward where the bullet went, tapering, curling. Returns
    (scoops: [(the bore's record in the buffer, its first splinter (counted from the first), how many, the radius round
    it they reach)], splinters: [((hinge, width), (tip, thickness), (the way across, curl))])."""
    scoops, out = [], []
    for j, b in enumerate(np.asarray(B, float)):
        code = int(round(float(b[3, 2])))
        if code & 15 != 5 or b[3, 0] <= 0.0 or (code >> 4) - 1 < 0:
            continue
        A, Bp = b[0, :3], b[1, :3]
        u = Bp - A
        u /= max(float(np.linalg.norm(u)), 1e-12)
        g = b[2, :3] - u * float(b[2, :3] @ u)
        g /= max(float(np.linalg.norm(g)), 1e-12)
        hh = np.cross(u, g)
        seed, r, elong = float(b[2, 3]), float(b[1, 3]), float(b[3, 0])
        rng = np.random.default_rng(int(seed * 1.0e6) + 7)
        asym = 0.65 + 0.7 * ((seed * 7.7) % 1.0)    # (as bore_one tears it: longer one way along the grain)
        face = Bp - u * BORE_LIP
        k0 = len(out)
        for i in range(SPLINTERS):
            hx, hy, hz, hw = rng.random(4)
            if hw <= 0.25:
                continue
            phi = (i + 0.5 + 0.7 * (hx - 0.5)) / SPLINTERS * 2.0 * math.pi
            cp = math.cos(phi)
            side = asym if cp >= 0.0 else 2.0 - asym
            xr, yr = r * (1.0 + elong * side) * cp * 0.85, r * math.sin(phi) * 0.85
            sgn = (1.0 if hy > 0.5 else -1.0) if abs(cp) < 0.35 else (1.0 if cp >= 0.0 else -1.0)
            ln = r * (0.7 + 2.0 * hy * hy) * (0.45 + 0.55 * abs(cp))
            lift = 0.2 + 0.9 * hz * hz
            w = r * (0.05 + 0.16 * hx * hw)
            h = face + g * (xr + sgn * ln) + hh * yr
            t = h - g * (sgn * ln * math.cos(lift)) + u * (ln * math.sin(lift))
            out.append([[*h, w], [*t, w * (0.3 + 0.25 * hy)], [*hh, 0.25 * (hz - 0.3) * (1.0 + hw)]])
        if len(out) > k0:
            scoops.append((first + 4 * j, 3 * k0, len(out) - k0, r * (4.2 + 1.4 * elong)))
    return scoops, out


def flash_lamps(view, shutter=0.0, frame_dt=1.0 / 24.0):
    """The flashes as point lights for the frame: [(position (fire-local m), colour (linear rgb, W/sr-ish))]."""
    out = []
    F = view.get('flashes') if view else None
    if F is None:
        return out
    for f in np.asarray(F, float):
        share = min(1.0, 0.0015 / max(shutter, frame_dt * 0.5))
        col = np.asarray((1.0, 0.62, 0.25) if f[7] > 0.5 else (1.0, 1.0, 0.9), float) * 400.0 * f[3] * share
        out.append((f[0:3] + f[4:7] * 0.03, col))
    return out
