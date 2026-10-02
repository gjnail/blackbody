"""The camera through a shot, from spots tracked in the footage (io/tracker.MultiTracker), starting from the
camera lined up on one frame (scene/groundmatch.py).

Two ways, and the one that fits better wins:
- A camera that only turns (a tripod, someone standing still): on every frame, the turn that best takes the
  directions to the spots to the lined-up ones (Kabsch).
- A camera that travels (a dolly, a walk, a car, a drone): the spots of the lined-up frame are placed where their
  rays meet the lined-up ground, which pins the scale, so the move is in real metres; on every frame the camera that
  sees the placed spots where they are is found, and every spot is placed again where the rays of all the frames that
  saw it meet, once they were seen from far enough apart, so spots on walls, posts and parked cars hold the move
  too instead of pulling it off.
Spots that disagree with the rest (people, cars, flags, things above the ground) are dropped. The fit is reported
in pixels."""
from __future__ import annotations

import math

import numpy as np

from .groundmatch import euler_xyz


def rays(points, size, f):
    """Unit camera-space directions through normalised points (x, y across and down the frame)."""
    W, H = size
    p = np.asarray(points, float)
    d = np.column_stack([(p[:, 0] * W - W / 2) / f, -(p[:, 1] * H - H / 2) / f, -np.ones(len(p))])
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def kabsch(a, b, w=None):
    """The rotation R that best takes directions a to b (R a ~ b)."""
    w = np.ones(len(a)) if w is None else w
    Hm = (b * w[:, None]).T @ a
    U, _, Vt = np.linalg.svd(Hm)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt)) or 1.0])
    return U @ D @ Vt


def _as_dicts(tracks):
    """{frame: {spot: (x, y)}} from either that or {frame: (n, 2) array with NaN where lost}."""
    out = {}
    for fr, pts in tracks.items():
        if isinstance(pts, dict):
            out[fr] = pts
        else:
            a = np.asarray(pts, float)
            out[fr] = {k: (float(p[0]), float(p[1])) for k, p in enumerate(a) if not np.isnan(p).any()}
    return out


def solve(tracks, ref, R0, size, f, max_err_px=1.5):
    """Camera-to-world rotations through the shot. tracks: {frame: (n, 2) normalised points, NaN where lost} with the
    reference frame `ref` among them, where the camera's rotation is R0. Returns ({frame: R}, {frame: error px},
    message). Frames with too few spots that agree are left out."""
    tracks = _as_dicts(tracks)
    world = {}   # spot: its direction in the world, from the solved frame nearest where it was first seen
    out, err = {}, {}
    R_of = {ref: np.asarray(R0, float)}
    for k, xy in tracks.get(ref, {}).items():
        world[k] = rays([xy], size, f)[0] @ R_of[ref].T
    for fr in sorted(tracks, key=lambda q: (abs(q - ref), q)):
        pts = tracks[fr]
        if fr != ref:
            near = min(R_of, key=lambda q: abs(q - fr))
            for k, xy in tracks.get(near, {}).items():
                if k not in world:
                    world[k] = rays([xy], size, f)[0] @ R_of[near].T
        ids = [k for k in pts if k in world]
        if len(ids) < 3:
            continue
        a = rays([pts[k] for k in ids], size, f)
        b = np.array([world[k] for k in ids])
        keep = np.ones(len(a), bool)
        R = None
        for _ in range(4):   # drop spots that do not turn with the rest
            if keep.sum() < 3:
                break
            R = kabsch(a[keep], b[keep])
            ang = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', a @ R.T, b), -1.0, 1.0)))
            px = np.radians(ang) * f
            lim = max(max_err_px, 2.5 * float(np.median(px[keep])))
            new = px <= lim
            if (new == keep).all():
                break
            keep = new
        if R is None or keep.sum() < 3:
            continue
        out[fr] = R
        R_of[fr] = R
        err[fr] = float(np.sqrt(np.mean(px[keep] ** 2)))
    if not out:
        return out, err, 'Too few spots could be followed to work out the camera move.'
    worst = max(err.values())
    med = float(np.median(list(err.values())))
    note = (f'The camera move is worked out for {len(out)} frames, within {med:.1f} px (worst {worst:.1f} px).'
            + (' Some frames fit loosely: the camera may also move along, not only turn: import a camera solve '
               '(File › Import camera track) for that.' if worst > 3 * max_err_px else ''))
    return out, err, note


def rotation_keys(rots):
    """[(frame, (rx, ry, rz))] Euler keys for rotations, each turned the short way from the last (no spins)."""
    keys = []
    prev = None
    for fr in sorted(rots):
        e = list(euler_xyz(rots[fr]))
        if prev is not None:
            e = [v + 360.0 * round((p - v) / 360.0) for v, p in zip(e, prev)]
        keys.append((fr, tuple(e)))
        prev = e
    return keys


def angle_between(Ra, Rb):
    """Degrees between two rotations."""
    c = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


# -- a camera that travels ------------------------------------------------------------------------------------

def project(X, R, eye, f, size):
    """World points to pixels, and their depth in front of the camera."""
    W, H = size
    pc = (np.asarray(X, float) - np.asarray(eye, float)) @ np.asarray(R, float)
    z = -pc[:, 2]
    zz = np.where(z > 1e-9, z, 1e-9)
    return np.column_stack([W / 2 + f * pc[:, 0] / zz, H / 2 - f * pc[:, 1] / zz]), z


def _skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def _exp(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3) + _skew(w)
    K = _skew(np.asarray(w) / th)
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def refine(R, eye, X, uv, f, size, iters=10):
    """The camera (R, eye) that best sees world points X at pixels uv (damped Gauss-Newton on the reprojection)."""
    R, eye = np.asarray(R, float), np.asarray(eye, float)
    X, uv = np.asarray(X, float), np.asarray(uv, float)
    lam = 1e-3

    def resid(R_, e_):
        px, z = project(X, R_, e_, f, size)
        r = (px - uv).ravel()
        return np.where(np.repeat(z, 2) > 1e-6, r, 1e3)

    r = resid(R, eye)
    cost = float(r @ r)
    for _ in range(iters):
        J = np.zeros((len(r), 6))
        for k in range(6):
            d = np.zeros(6)
            d[k] = 1e-6
            J[:, k] = (resid(_exp(d[:3]) @ R, eye + d[3:]) - r) / 1e-6
        A = J.T @ J
        g = J.T @ r
        moved = False
        while lam < 1e8:
            try:
                step = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
            except np.linalg.LinAlgError:
                break
            R2, e2 = _exp(step[:3]) @ R, eye + step[3:]
            r2 = resid(R2, e2)
            c2 = float(r2 @ r2)
            if c2 < cost:
                R, eye, r, cost = R2, e2, r2, c2
                lam = max(lam / 3, 1e-9)
                moved = True
                break
            lam *= 4
        if not moved or float(np.abs(step).max()) < 1e-10:
            break
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt, eye


def solve_travel(tracks, ref, R0, eye0, size, f, ground_y=0.0, thresh=3.0, min_angle=1.0):
    """Camera rotations and positions through the shot, for a camera that may travel. Returns ({frame: R},
    {frame: eye}, {frame: error px}, message).

    Outward from the lined-up frame, a frame at a time: the camera is the one that sees the spots placed so far where
    they were followed to (refined from the last frame's camera, and from a few random sixes in case some spots
    misled it); then every spot it sees is placed again where the rays of all the frames that saw it meet, once they
    were seen from far enough apart to tell (min_angle degrees). Until then a spot below the horizon is taken to be on
    the lined-up ground where it was first seen, which the ground of the lined-up frame makes true at the start.
    Footage rarely shows only ground: a wall, a post or a parked car is seen too, and spots on them kept on the ground
    pull the whole move off; placed where their rays meet, they hold it instead. Spots that disagree three times
    (something moving, an edge they slid along) are dropped."""
    tracks = _as_dicts(tracks)
    W, H = size
    R_of, eye_of = {ref: np.asarray(R0, float)}, {ref: np.asarray(eye0, float)}
    height = max(abs(float(eye0[1]) - ground_y), 0.05)
    X, met, strikes = {}, set(), {}   # spot: point; spots placed where their rays meet; spot: times it disagreed

    def on_ground(fr, k, xy):
        d = np.asarray(rays([xy], size, f)[0]) @ R_of[fr].T
        if d[1] > -0.02:   # at or above the horizon
            return
        t = (ground_y - eye_of[fr][1]) / d[1]
        if 0 < t < 60 * height:
            X[k] = eye_of[fr] + d * t

    def place(fr):
        for k, xy in tracks[fr].items():
            if strikes.get(k, 0) >= 3:
                continue
            obs = [(g, tracks[g][k]) for g in R_of if k in tracks[g]]
            tr = _triangulate(obs, R_of, eye_of, size, f) if len(obs) >= 3 else None
            if tr is not None and tr[1] >= min_angle and tr[2] < 1.5 and np.linalg.norm(tr[0] - eye_of[fr]) < 80 * height:
                if k not in met:
                    strikes.pop(k, None)   # its disagreeing was being taken to be on the ground
                    met.add(k)
                X[k] = tr[0]
            elif k not in X:
                on_ground(fr, k, xy)

    for k, xy in tracks.get(ref, {}).items():
        on_ground(ref, k, xy)
    err = {ref: 0.0}
    rng = np.random.default_rng(1)
    for direction in (1, -1):
        prev = ref
        for fr in sorted((fr for fr in tracks if (fr - ref) * direction > 0), key=lambda fr: abs(fr - ref)):
            pts = tracks[fr]
            ids = [k for k in pts if k in X and strikes.get(k, 0) < 3]
            pose = None
            if len(ids) >= 6:
                Xa = np.array([X[k] for k in ids])
                uv = np.array([pts[k] for k in ids]) * np.array([W, H])
                pose, inl = _pose_robust(Xa, uv, f, size, (R_of[prev], eye_of[prev]), thresh, rng=rng)
                if pose is not None:
                    for k, good in zip(ids, inl):
                        if not good:
                            strikes[k] = strikes.get(k, 0) + 1
                    px, _ = project(Xa[inl], pose[0], pose[1], f, size)
                    err[fr] = float(np.sqrt(np.mean(np.sum((px - uv[inl]) ** 2, axis=1))))
            if pose is None:   # too few spots placed: it is taken to have only turned since the last frame
                common = [k for k in pts if k in tracks.get(prev, {})]
                if len(common) < 3:
                    continue
                a = rays([pts[k] for k in common], size, f)
                b = rays([tracks[prev][k] for k in common], size, f) @ R_of[prev].T
                pose = (kabsch(a, b), eye_of[prev])
                err[fr] = float('nan')
            R_of[fr], eye_of[fr] = pose
            place(fr)
            prev = fr
    if len(R_of) < 2:
        return R_of, eye_of, err, 'Too few spots could be followed to work out the camera move.'
    good = [e for e in err.values() if e == e]
    travel = max(float(np.linalg.norm(e - eye_of[ref])) for e in eye_of.values())
    med = float(np.median(good)) if good else float('nan')
    note = f'The camera move is worked out for {len(R_of)} frames: it travels {travel:.2f} m, within {med:.1f} px.'
    return R_of, eye_of, err, note


def _triangulate(obs, R_of, eye_of, size, f):
    """The point where the rays of obs [(frame, xy)] come closest together, the widest angle between them (degrees)
    and its reprojection error (rms px), or None."""
    o = np.array([eye_of[fr] for fr, _ in obs])
    d = np.array([rays([xy], size, f)[0] @ R_of[fr].T for fr, xy in obs])
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for oi, di in zip(o, d):
        M = np.eye(3) - np.outer(di, di)
        A += M
        b += M @ oi
    try:
        X = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    W, H = size
    errs = []
    for (fr, xy) in obs:
        px, z = project(X[None], R_of[fr], eye_of[fr], f, size)
        if z[0] <= 0:
            return None
        errs.append(np.sum((px[0] - np.array([xy[0] * W, xy[1] * H])) ** 2))
    cosmin = float(np.min(d @ d[np.argmax(np.linalg.norm(o - o[0], axis=1))]))
    ang = math.degrees(math.acos(float(np.clip(cosmin, -1.0, 1.0))))
    return X, ang, float(np.sqrt(np.mean(errs)))


def _pose_robust(X, uv, f, size, guess, thresh=2.5, rng=None, rounds=12):
    """The camera that sees most of the 3D spots X at uv within thresh px: refined from `guess` (the camera of the
    frame next to it) on all of them, trimmed, and from a few random sixes; and which spots agree."""
    rng = rng or np.random.default_rng(3)
    n = len(X)
    if n < 6:
        return None, np.zeros(n, bool)

    def fit(sel, start):
        R, eye = refine(start[0], start[1], X[sel], uv[sel], f, size, iters=8)
        px, z = project(X, R, eye, f, size)
        e = np.linalg.norm(px - uv, axis=1)
        inl = (e < thresh) & (z > 0)
        return (R, eye), inl, float(np.median(e[inl])) if inl.any() else float('inf')

    best = None
    sel = np.ones(n, bool)
    pose = guess
    for _ in range(3):   # trimmed refinement from the neighbour's camera
        pose, inl, med = fit(sel, pose)
        if inl.sum() < 6:
            break
        sel = inl
    if inl.sum() >= 6:
        best = (pose, inl)
    for _ in range(rounds):   # and from small samples, in case outliers pulled it
        idx = rng.choice(n, 6, replace=False)
        sel = np.zeros(n, bool)
        sel[idx] = True
        p2, inl2, _ = fit(sel, guess)
        if inl2.sum() >= 6:
            p2, inl2, _ = fit(inl2, p2)
        if best is None or inl2.sum() > best[1].sum():
            best = (p2, inl2)
    if best is None or best[1].sum() < 6:
        return None, np.zeros(n, bool)
    return best


def solve_shot(tracks, ref, R0, eye0, size, f, ground_y=0.0):
    """Both ways, keeping whichever fits better; a camera that only turns is kept in place when that fits about as
    well. Returns ({frame: R}, {frame: eye} or None, {frame: error px}, message, 'turn' | 'travel')."""
    rots, err_t, note_t = solve(tracks, ref, R0, size, f)
    try:
        R_m, eye_m, err_m, note_m = solve_travel(tracks, ref, R0, eye0, size, f, ground_y)
    except (np.linalg.LinAlgError, ValueError):
        R_m, eye_m, err_m, note_m = {}, {}, {}, ''
    et = float(np.median(list(err_t.values()))) if err_t else float('inf')
    em_vals = [e for e in err_m.values() if e == e]
    em = float(np.median(em_vals)) if em_vals else float('inf')
    height = max(abs(float(eye0[1]) - ground_y), 0.05)
    travel = max((float(np.linalg.norm(np.asarray(e) - np.asarray(eye0))) for e in eye_m.values()), default=0.0) if eye_m else 0.0
    if len(R_m) >= 2 and (not rots or (em < 0.85 * et - 0.1 and travel > 0.03 * height)):
        return R_m, eye_m, err_m, note_m, 'travel'
    return rots, None, err_t, note_t, 'turn'
