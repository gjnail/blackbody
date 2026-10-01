"""The camera through a shot, from spots tracked in the footage (io/tracker.MultiTracker), starting from the
camera lined up on one frame (scene/groundmatch.py).

Two ways, and the one that fits better wins:
- A camera that only turns (a tripod, someone standing still): on every frame, the turn that best takes the
  directions to the spots to the lined-up ones (Kabsch).
- A camera that travels (a dolly, a walk, a car, a drone): spots on the ground are placed in the world where their
  rays meet the lined-up ground; on every frame the camera that sees them where they are (a plane homography,
  RANSAC against spots that are not on the ground, then refined on the reprojection) is found, and spots that come
  into view are placed from it. The ground pins the scale, so the move is in real metres.
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


def _homography(src, dst):
    """3 x 3 H with dst ~ H src (normalised DLT), from 4 or more points."""
    def norm(q):
        c = q.mean(0)
        sc = math.sqrt(2) / max(float(np.mean(np.linalg.norm(q - c, axis=1))), 1e-12)
        T = np.array([[sc, 0, -sc * c[0]], [0, sc, -sc * c[1]], [0, 0, 1]])
        return np.column_stack([q, np.ones(len(q))]) @ T.T, T
    a, Ta = norm(np.asarray(src, float))
    b, Tb = norm(np.asarray(dst, float))
    rows = []
    for (x, y, _), (u, v, _) in zip(a, b):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.array(rows))
    return np.linalg.inv(Tb) @ Vt[-1].reshape(3, 3) @ Ta


def _pose_from_ground(xz, uv, f, size, y0=0.0):
    """The camera from points on the level ground y = y0 (their x, z) seen at pixels uv, or None."""
    W, H = size
    ab = np.column_stack([(uv[:, 0] - W / 2) / f, -(uv[:, 1] - H / 2) / f])
    try:
        Hm = _homography(xz, ab)
    except np.linalg.LinAlgError:
        return None
    S = np.diag([1.0, 1.0, -1.0])
    lam = 1.0 / max(float(np.linalg.norm(Hm[:, 0])), 1e-12)
    if (Hm @ np.array([xz[0, 0], xz[0, 1], 1.0]))[2] * lam < 0:
        lam = -lam
    r1 = S @ (lam * Hm[:, 0])
    r3 = S @ (lam * Hm[:, 1])
    t = S @ (lam * Hm[:, 2])
    r2 = np.cross(r3, r1)
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, r3]))
    Rt = U @ Vt   # world to camera
    if np.linalg.det(Rt) < 0:
        return None
    R = Rt.T
    return R, -R @ t + np.array([0.0, y0, 0.0])


def _pose_ransac(X, uv, f, size, guess, thresh=3.0, rounds=80, rng=None):
    """The camera that sees most of the ground points X at uv within thresh px, refined, and which they are."""
    rng = rng or np.random.default_rng(0)
    xz = X[:, [0, 2]]
    best, best_in = None, np.zeros(len(X), bool)
    cands = [guess] if guess is not None else []
    for _ in range(rounds):
        idx = rng.choice(len(X), 4, replace=False)
        pose = _pose_from_ground(xz[idx], uv[idx], f, size, float(X[0, 1]))
        if pose is not None:
            cands.append(pose)
    for pose in cands:
        px, z = project(X, pose[0], pose[1], f, size)
        inl = (np.linalg.norm(px - uv, axis=1) < thresh) & (z > 0)
        if inl.sum() > best_in.sum():
            best, best_in = pose, inl
    if best is None or best_in.sum() < 4:
        return None, best_in
    for _ in range(2):
        R, eye = refine(best[0], best[1], X[best_in], uv[best_in], f, size)
        px, z = project(X, R, eye, f, size)
        inl = (np.linalg.norm(px - uv, axis=1) < thresh) & (z > 0)
        best = (R, eye)
        if inl.sum() >= 4:
            best_in = inl
    return best, best_in


def solve_travel(tracks, ref, R0, eye0, size, f, ground_y=0.0, thresh=3.0):
    """Camera rotations and positions through the shot, for a camera that may travel. Returns ({frame: R},
    {frame: eye}, {frame: error px}, message)."""
    tracks = _as_dicts(tracks)
    W, H = size
    R_of, eye_of = {ref: np.asarray(R0, float)}, {ref: np.asarray(eye0, float)}
    X = {}
    strikes = {}
    height = max(abs(float(eye0[1]) - ground_y), 0.05)

    def place(fr, k, xy):
        d = np.asarray(rays([xy], size, f)[0]) @ R_of[fr].T
        if d[1] > -0.02:   # at or above the horizon
            return
        t = (ground_y - eye_of[fr][1]) / d[1]
        if 0 < t < 60 * height:
            X[k] = eye_of[fr] + d * t

    for k, xy in tracks.get(ref, {}).items():
        place(ref, k, xy)
    err = {ref: 0.0}
    rng = np.random.default_rng(1)
    for direction in (1, -1):
        prev = ref
        frames = sorted((fr for fr in tracks if (fr - ref) * direction > 0), key=lambda fr: abs(fr - ref))
        for fr in frames:
            pts = tracks[fr]
            ids = [k for k in pts if k in X and strikes.get(k, 0) < 3]
            pose = None
            if len(ids) >= 6:
                Xa = np.array([X[k] for k in ids])
                uv = np.array([pts[k] for k in ids]) * np.array([W, H])
                pose, inl = _pose_ransac(Xa, uv, f, size, (R_of[prev], eye_of[prev]), thresh, rng=rng)
                if pose is not None:
                    for k, good in zip(ids, inl):
                        if not good:
                            strikes[k] = strikes.get(k, 0) + 1
                    px, _ = project(Xa[inl], pose[0], pose[1], f, size)
                    err[fr] = float(np.sqrt(np.mean(np.sum((px - uv[inl]) ** 2, axis=1)))) if inl.any() else 0.0
            if pose is None:   # too few spots on the ground: it is taken to have only turned since the last frame
                common = [k for k in pts if k in tracks.get(prev, {})]
                if len(common) < 3:
                    continue
                a = rays([pts[k] for k in common], size, f)
                b = rays([tracks[prev][k] for k in common], size, f) @ R_of[prev].T
                pose = (kabsch(a, b), eye_of[prev])
                err[fr] = float('nan')
            R_of[fr], eye_of[fr] = pose
            for k, xy in pts.items():
                if k not in X:
                    place(fr, k, xy)
            prev = fr
    if len(R_of) < 2:
        return R_of, eye_of, err, 'Too few spots on the ground could be followed to work out the camera move.'
    # refine: each spot first seen after the lined-up frame is placed from every frame that sees it (not only the one
    # it came into view on), then every camera is solved again on the spots; twice. Errors then no longer pile up
    # along the move. The spots of the lined-up frame stay where it put them: they hold the world in place.
    anchored = {k for k in tracks.get(ref, {}) if k in X}
    for _ in range(5):
        hits = {}
        for fr, pts in tracks.items():
            if fr not in R_of or fr == ref:
                continue
            for k, xy in pts.items():
                if k in anchored or strikes.get(k, 0) >= 3:
                    continue
                d = np.asarray(rays([xy], size, f)[0]) @ R_of[fr].T
                if d[1] > -0.02:
                    continue
                t = (ground_y - eye_of[fr][1]) / d[1]
                if 0 < t < 60 * height:
                    hits.setdefault(k, []).append(eye_of[fr] + d * t)
        for k, hs in hits.items():
            if len(hs) >= 2:
                X[k] = np.median(np.array(hs), axis=0)
        for fr, pts in tracks.items():
            if fr == ref or fr not in R_of:
                continue
            ids = [k for k in pts if k in X and strikes.get(k, 0) < 3]
            if len(ids) < 6:
                continue
            Xa = np.array([X[k] for k in ids])
            uv = np.array([pts[k] for k in ids]) * np.array([W, H])
            px, z = project(Xa, R_of[fr], eye_of[fr], f, size)
            inl = (np.linalg.norm(px - uv, axis=1) < thresh * 1.5) & (z > 0)
            if inl.sum() < 6:
                continue
            R_of[fr], eye_of[fr] = refine(R_of[fr], eye_of[fr], Xa[inl], uv[inl], f, size, iters=6)
            px, _ = project(Xa[inl], R_of[fr], eye_of[fr], f, size)
            err[fr] = float(np.sqrt(np.mean(np.sum((px - uv[inl]) ** 2, axis=1))))
    # then everything together (a bundle adjustment), with the spots that disagree with the rest left out
    ok = {k for k in X if strikes.get(k, 0) < 3}
    Xok = {k: X[k] for k in ok}
    ba = bundle_adjust(R_of, eye_of, Xok, tracks, ref, anchored & ok, size, f)
    X.update(Xok)
    err.update({fr: e for fr, e in ba.items() if fr != ref})
    good = [e for e in err.values() if e == e]
    travel = max(float(np.linalg.norm(e - eye_of[ref])) for e in eye_of.values())
    med = float(np.median(good)) if good else float('nan')
    note = f'The camera move is worked out for {len(R_of)} frames: it travels {travel:.2f} m, within {med:.1f} px.'
    return R_of, eye_of, err, note


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


def bundle_adjust(R_of, eye_of, X, tracks, ref, anchored, size, f, iters=10, huber=2.0, max_err=6.0):
    """Every camera and every spot on the ground, together: the cameras (but the lined-up one) and the ground spots
    (but those placed from the lined-up frame, which hold the world in place) that best explain every tracked
    position (Levenberg-Marquardt with the Schur complement, Huber-weighted). Changes R_of, eye_of and X in place;
    returns the per-frame error in px."""
    W, H = size
    frames = [fr for fr in sorted(R_of) if fr != ref]
    fidx = {fr: i for i, fr in enumerate(frames)}
    free = sorted(k for k in X if k not in anchored)
    pidx = {k: i for i, k in enumerate(free)}
    obs_f, obs_p, obs_k, obs_fr, uv = [], [], [], [], []
    for fr, pts in tracks.items():
        if fr not in R_of:
            continue
        for k, xy in pts.items():
            if k in X:
                obs_f.append(fidx.get(fr, -1))
                obs_p.append(pidx.get(k, -1))
                obs_k.append(k)
                obs_fr.append(fr)
                uv.append((xy[0] * W, xy[1] * H))
    if not obs_f:
        return {}
    obs_f, obs_p, uv = np.array(obs_f), np.array(obs_p), np.array(uv, float)
    F, P = len(frames), len(free)
    Rs = np.array([R_of[fr] for fr in obs_fr])
    eyes = np.array([eye_of[fr] for fr in obs_fr])
    Xs = np.array([X[k] for k in obs_k])

    def residuals(Rs, eyes, Xs):
        a = Xs - eyes
        pc = np.einsum('nji,nj->ni', Rs, a)    # R^T a
        z = -pc[:, 2]
        zz = np.where(z > 1e-6, z, 1e-6)
        px = np.column_stack([W / 2 + f * pc[:, 0] / zz, H / 2 - f * pc[:, 1] / zz])
        return px - uv, pc, zz, a

    r, pc, zz, a = residuals(Rs, eyes, Xs)
    keep = (np.linalg.norm(r, axis=1) < max_err) & (pc[:, 2] < 0)   # outliers as they stand now are left out

    def cost(r):
        e = np.linalg.norm(r, axis=1)
        return float(np.sum(np.where(e < huber, 0.5 * e * e, huber * (e - 0.5 * huber))[keep]))

    lam = 1e-3
    c0 = cost(r)
    for _ in range(iters):
        e = np.linalg.norm(r, axis=1)
        w = np.where(e < huber, 1.0, huber / np.maximum(e, 1e-9)) * keep
        # d px / d pc
        Jpc = np.zeros((len(r), 2, 3))
        Jpc[:, 0, 0] = f / zz
        Jpc[:, 0, 2] = f * pc[:, 0] / zz ** 2
        Jpc[:, 1, 1] = -f / zz
        Jpc[:, 1, 2] = -f * pc[:, 1] / zz ** 2
        RT = np.transpose(Rs, (0, 2, 1))
        ax = np.zeros((len(r), 3, 3))   # [a]x
        ax[:, 0, 1], ax[:, 0, 2], ax[:, 1, 0] = -a[:, 2], a[:, 1], a[:, 2]
        ax[:, 1, 2], ax[:, 2, 0], ax[:, 2, 1] = -a[:, 0], -a[:, 1], a[:, 0]
        Jw = Jpc @ RT @ ax                      # rotation (world-frame turn)
        Je = -(Jpc @ RT)                         # camera position
        Jc = np.concatenate([Jw, Je], axis=2)    # (n, 2, 6)
        Jp = (Jpc @ RT)[:, :, [0, 2]]            # ground spot (x, z)
        sw = np.sqrt(w)[:, None]
        Jc_w, Jp_w, r_w = Jc * sw[:, :, None], Jp * sw[:, :, None], r * sw
        U = np.zeros((F, 6, 6))
        V = np.zeros((P, 2, 2))
        gc = np.zeros((F, 6))
        gp = np.zeros((P, 2))
        fc = obs_f >= 0
        pf = obs_p >= 0
        np.add.at(U, obs_f[fc], np.einsum('nki,nkj->nij', Jc_w[fc], Jc_w[fc]))
        np.add.at(gc, obs_f[fc], np.einsum('nki,nk->ni', Jc_w[fc], r_w[fc]))
        np.add.at(V, obs_p[pf], np.einsum('nki,nkj->nij', Jp_w[pf], Jp_w[pf]))
        np.add.at(gp, obs_p[pf], np.einsum('nki,nk->ni', Jp_w[pf], r_w[pf]))
        both = fc & pf
        Wb = np.einsum('nki,nkj->nij', Jc_w[both], Jp_w[both])   # (m, 6, 2) per observation of a free spot by a free camera
        bf, bp = obs_f[both], obs_p[both]
        while lam < 1e8:
            Ud = U + lam * (np.einsum('nii->ni', U)[:, :, None] * np.eye(6) + 1e-9 * np.eye(6))
            Vd = V + lam * (np.einsum('nii->ni', V)[:, :, None] * np.eye(2) + 1e-9 * np.eye(2))
            Vinv = np.linalg.inv(Vd)
            S = np.zeros((F * 6, F * 6))
            for i in range(F):
                S[i * 6:i * 6 + 6, i * 6:i * 6 + 6] = Ud[i]
            rhs = -gc.copy()
            WV = np.einsum('nij,njk->nik', Wb, Vinv[bp])            # W V^-1 per observation
            np.add.at(rhs, bf, np.einsum('nij,nj->ni', WV, gp[bp]))
            order = np.argsort(bp, kind='stable')
            bps, starts = np.unique(bp[order], return_index=True)
            ends = list(starts[1:]) + [len(order)]
            for s0, s1 in zip(starts, ends):   # pairs of cameras that see the same spot
                ids = order[s0:s1]
                blocks = np.einsum('aij,bkj->abik', WV[ids], Wb[ids])
                for ai, fa in enumerate(bf[ids]):
                    for bi, fb in enumerate(bf[ids]):
                        S[fa * 6:fa * 6 + 6, fb * 6:fb * 6 + 6] -= blocks[ai, bi]
            try:
                dc = np.linalg.solve(S, rhs.ravel()).reshape(F, 6)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            gp2 = gp.copy()
            np.add.at(gp2, bp, np.einsum('nij,ni->nj', Wb, dc[bf]))
            dp = -np.einsum('nij,nj->ni', Vinv, gp2)
            # try the step
            newR = {fr: _exp(dc[i, :3]) @ R_of[fr] for fr, i in fidx.items()}
            newE = {fr: eye_of[fr] + dc[i, 3:] for fr, i in fidx.items()}
            newX = {k: X[k] + np.array([dp[i, 0], 0.0, dp[i, 1]]) for k, i in pidx.items()}
            Rs2 = np.array([newR.get(fr, R_of[fr]) for fr in obs_fr])
            eyes2 = np.array([newE.get(fr, eye_of[fr]) for fr in obs_fr])
            Xs2 = np.array([newX.get(k, X[k]) for k in obs_k])
            r2, pc2, zz2, a2 = residuals(Rs2, eyes2, Xs2)
            c2 = cost(r2)
            if c2 < c0:
                for fr in newR:
                    U_, _, Vt_ = np.linalg.svd(newR[fr])
                    R_of[fr] = U_ @ Vt_
                    eye_of[fr] = newE[fr]
                X.update(newX)
                Rs, eyes, Xs, r, pc, zz, a, c0 = Rs2, eyes2, Xs2, r2, pc2, zz2, a2, c2
                lam = max(lam / 3, 1e-9)
                break
            lam *= 5
        else:
            break
    e = np.linalg.norm(r, axis=1)
    out = {}
    for fr in R_of:
        m = (np.array(obs_fr) == fr) & keep
        if m.any():
            out[fr] = float(np.sqrt(np.mean(e[m] ** 2)))
    return out
