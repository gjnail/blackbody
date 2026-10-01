"""Procedural backplates for the promo shots, seen through the shot camera (so the renderer's
refraction, fire light and holdouts line up with them): a night ground, a daylight pebble bed, and a
night yard with a wall and an oil drum (with its depth pass, for holdouts and fire light)."""
from __future__ import annotations

import numpy as np


# ---- noise ----------------------------------------------------------------------------------------

def _hash2(ix, iy, seed=0):
    h = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263 + seed * 2147483647) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFFF).astype(np.float32) / float(0xFFFFFF)


def value_noise(x, y, seed=0):
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy
    ux, uy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a = _hash2(ix, iy, seed)
    b = _hash2(ix + 1, iy, seed)
    c = _hash2(ix, iy + 1, seed)
    d = _hash2(ix + 1, iy + 1, seed)
    return (a + (b - a) * ux) + ((c + (d - c) * ux) - (a + (b - a) * ux)) * uy


def fbm(x, y, octaves=5, seed=0, gain=0.5, lac=2.03):
    s = np.zeros_like(x, dtype=np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        s += amp * value_noise(x, y, seed + o * 17)
        tot += amp
        amp *= gain
        x, y = x * lac + 5.3, y * lac + 1.7
    return s / tot


def worley(x, y, seed=0):
    """Distance to the nearest and second-nearest feature point (cells of size 1)."""
    ix, iy = np.floor(x), np.floor(y)
    d1 = np.full(x.shape, 9.0, np.float32)
    d2 = np.full(x.shape, 9.0, np.float32)
    idn = np.zeros(x.shape, np.float32)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            cx, cy = ix + ox, iy + oy
            px = cx + _hash2(cx, cy, seed)
            py = cy + _hash2(cx, cy, seed + 7)
            d = np.sqrt((x - px) ** 2 + (y - py) ** 2)
            closer = d < d1
            d2 = np.where(closer, d1, np.minimum(d2, d))
            idn = np.where(closer, _hash2(cx, cy, seed + 13), idn)
            d1 = np.minimum(d1, d)
    return d1, d2, idn


# ---- camera rays ---------------------------------------------------------------------------------

def rays(cs, W, H):
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64) + 0.5
    ndc = np.stack([xs / W * 2 - 1, 1 - ys / H * 2], -1)
    ones = np.ones((H, W, 1))
    a = np.concatenate([ndc, 0 * ones, ones], -1) @ cs.inv_view_proj.T
    b = np.concatenate([ndc, ones, ones], -1) @ cs.inv_view_proj.T
    a, b = a[..., :3] / a[..., 3:], b[..., :3] / b[..., 3:]
    d = b - a
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    eye = np.asarray(cs.eye, np.float64)
    return eye, d


def to_srgb(lin):
    lin = np.clip(lin, 0.0, None)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


def _pack(rgb_lin, depth=None):
    H, W = rgb_lin.shape[:2]
    out = np.full((H, W, 4), 255, np.uint8)
    out[..., :3] = (np.clip(to_srgb(rgb_lin), 0, 1) * 255 + 0.5).astype(np.uint8)
    return out if depth is None else (out, depth.astype(np.float32))


# ---- grounds ---------------------------------------------------------------------------------------

def _ground_tex(u, v, kind):
    """Linear albedo (H, W, 3) of the ground at world (u, v) metres."""
    if kind == 'dirt':
        n = fbm(u * 0.6, v * 0.6, 5, 3)
        m = fbm(u * 5.0, v * 5.0, 4, 11)
        d1, d2, idn = worley(u * 3.0, v * 3.0, 21)
        stones = np.clip((0.35 - d1) * 6.0, 0, 1) * (idn > 0.72)
        base = 0.10 + 0.07 * n + 0.05 * m
        col = np.stack([base * 1.05, base * 0.93, base * 0.80], -1)
        col = col * (1 - stones[..., None]) + stones[..., None] * np.stack([0.16 + 0.06 * idn] * 3, -1) * np.array([1.0, 0.97, 0.92])
        return col
    if kind == 'concrete':
        n = fbm(u * 0.8, v * 0.8, 5, 5)
        m = fbm(u * 9.0, v * 9.0, 3, 9)
        crack_n = fbm(u * 1.4, v * 1.4, 4, 31)
        crack = np.clip(1.0 - np.abs(crack_n - 0.5) * 140.0, 0, 1) * 0.3
        joints = (np.minimum(np.abs(((u / 2.0) % 1.0) - 0.5), np.abs(((v / 2.0) % 1.0) - 0.5)) > 0.492) * 0.5
        base = (0.20 + 0.08 * n + 0.04 * m) * (1 - crack) * (1 - joints)
        return np.stack([base, base * 0.98, base * 0.95], -1)
    if kind in ('pebbles', 'stone'):
        # loose river pebbles: a dome on each Voronoi point, of its own size and colour, on sand
        scale = 2.2 if kind == 'stone' else 5.5
        pal = np.array([[0.42, 0.40, 0.37], [0.30, 0.28, 0.26], [0.52, 0.49, 0.44], [0.25, 0.24, 0.23],
                        [0.46, 0.37, 0.29], [0.60, 0.58, 0.55], [0.36, 0.33, 0.31], [0.40, 0.31, 0.25]], np.float32)

        def layer(uu, vv, sc, seed):
            d1, d2, idn = worley(uu * sc, vv * sc, seed)
            r = 0.24 + 0.14 * _hash2(np.floor(idn * 4099), np.zeros_like(idn), seed + 3)
            return np.sqrt(np.clip(1.0 - (d1 / r) ** 2, 0, 1)) * r / sc, idn, d1 / r

        def height(uu, vv):
            h1, i1, r1 = layer(uu, vv, scale, 41)
            h2, i2, r2 = layer(uu, vv, scale * 2.3, 53)
            use2 = h2 * 0.85 > h1
            return np.where(use2, h2 * 0.85, h1), np.where(use2, i2, i1), np.minimum(r1, r2 * 1.0)

        eps = 0.06 / scale
        h, idn, rr = height(u, v)
        hu, _, _ = height(u + eps, v)
        hv, _, _ = height(u, v + eps)
        k = 1.0
        nx, nz = -(hu - h) / eps * k, -(hv - h) / eps * k
        nl = np.sqrt(nx * nx + 1 + nz * nz)
        lam = np.clip((nx * -0.5 + 0.8 + nz * -0.35) / nl, 0, 1)
        stone = h > 1e-5
        col = pal[(idn * 7.999).astype(int)] * (0.85 + 0.3 * fbm(u * 30.0, v * 30.0, 3, 47))[..., None]
        col = col * (0.45 + 0.8 * lam)[..., None]
        sand = fbm(u * 8.0, v * 8.0, 4, 7)
        sandc = np.stack([0.36 + 0.06 * sand, 0.32 + 0.05 * sand, 0.25 + 0.04 * sand], -1)
        ao = np.clip(rr - 0.9, 0, 1) * 0.0 + np.clip((rr - 1.0) * 1.5 + 0.55, 0.55, 1.0)
        out = np.where(stone[..., None], col, sandc * ao[..., None])
        if kind == 'stone':
            out = out * 0.55
        return out
    raise ValueError(kind)


def ground_plate(cs, W, H, kind='dirt', night=True, horizon=None, fog=40.0, sky=None, seed=0, treeline=True):
    """A ground plane to the horizon under a sky, through camera cs. Night: dark, cool, moonlit;
    day: bright sky and a warm sun. Returns (rgba uint8, depth (m, 0 = sky))."""
    eye, d = rays(cs, W, H)
    t = -eye[1] / np.where(np.abs(d[..., 1]) < 1e-9, -1e-9, d[..., 1])
    hit = (d[..., 1] < 0) & (t > 0)
    t = np.where(hit, t, 0.0)
    q = eye + d * t[..., None]
    alb = _ground_tex(q[..., 0].astype(np.float32), q[..., 2].astype(np.float32), kind)
    if night:
        light = np.array([0.050, 0.060, 0.085])     # moon and sky glow
        zen = np.array([0.004, 0.006, 0.013])
        hor = np.array([0.020, 0.024, 0.034]) if horizon is None else np.asarray(horizon)
    else:
        light = np.array([1.05, 1.0, 0.92])
        zen = np.array([0.18, 0.30, 0.55])
        hor = np.array([0.62, 0.70, 0.80]) if horizon is None else np.asarray(horizon)
    elev = np.clip(d[..., 1], 0, 1)[..., None]
    skyc = hor + (zen - hor) * np.power(elev, 0.45)
    if sky is not None:
        skyc = skyc * np.asarray(sky)
    fogf = 1.0 - np.exp(-t / fog)[..., None]
    grd = alb * light
    col = np.where(hit[..., None], grd * (1 - fogf) + hor * fogf, skyc)
    if treeline and night:
        # a far treeline against the sky
        az = np.arctan2(d[..., 0], d[..., 2])
        h = 0.012 + 0.02 * fbm(az * 12.0 + seed, np.zeros_like(az) + 3.0, 4, 61) + 0.015 * fbm(az * 60.0, np.zeros_like(az), 3, 71)
        trees = (d[..., 1] > 0) & (d[..., 1] < h)
        col = np.where(trees[..., None], hor * 0.35, col)
    depth = np.where(hit, t, 0.0)
    return _pack(col, depth)


# ---- the night yard ------------------------------------------------------------------------------

def yard_plate(cs, W, H, drum=(-0.9, 0.0, 1.4), drum_r=0.30, drum_h=0.88, wall_z=-3.2, fire_xform=None):
    """A night yard through camera cs (world coordinates = fire-local, no yaw): concrete ground, a
    breeze-block wall behind the fire, and an oil drum in front of it. Returns (rgba uint8, depth m)."""
    eye, d = rays(cs, W, H)
    INF = 1e9
    # ground
    tg = np.where(d[..., 1] < -1e-9, -eye[1] / np.where(d[..., 1] < -1e-9, d[..., 1], -1.0), INF)
    # wall (plane z = wall_z, facing +z), 3.2 m tall
    tw = np.where(d[..., 2] < -1e-9, (wall_z - eye[2]) / np.where(d[..., 2] < -1e-9, d[..., 2], -1.0), INF)
    pw = eye + d * np.where(tw < INF, tw, 0.0)[..., None]
    tw = np.where((tw > 0) & (pw[..., 1] > 0) & (pw[..., 1] < 3.2), tw, INF)
    # drum: vertical cylinder
    cx, cz = drum[0], drum[2]
    ox, oz = eye[0] - cx, eye[2] - cz
    a = d[..., 0] ** 2 + d[..., 2] ** 2
    b = 2 * (ox * d[..., 0] + oz * d[..., 2])
    c = ox * ox + oz * oz - drum_r * drum_r
    disc = b * b - 4 * a * c
    sq = np.sqrt(np.maximum(disc, 0))
    tc = (-b - sq) / (2 * np.maximum(a, 1e-12))
    pc = eye + d * np.where(disc > 0, tc, 0.0)[..., None]
    tc = np.where((disc > 0) & (tc > 0) & (pc[..., 1] > 0) & (pc[..., 1] < drum_h), tc, INF)
    # drum lid
    tl = np.where(np.abs(d[..., 1]) > 1e-9, (drum_h - eye[1]) / np.where(np.abs(d[..., 1]) > 1e-9, d[..., 1], 1.0), INF)
    pl = eye + d * np.where(tl < INF, tl, 0.0)[..., None]
    tl = np.where((tl > 0) & ((pl[..., 0] - cx) ** 2 + (pl[..., 2] - cz) ** 2 < drum_r ** 2), tl, INF)
    t = np.minimum(np.minimum(tg, tw), np.minimum(tc, tl))
    p = eye + d * np.where(t < INF, t, 0.0)[..., None]
    moon = np.array([0.045, 0.055, 0.080])
    amb = np.array([0.018, 0.021, 0.030])
    col = np.zeros((H, W, 3))
    # ground: concrete
    g = (t == tg) & (tg < INF)
    alb = _ground_tex(p[..., 0].astype(np.float32), p[..., 2].astype(np.float32), 'concrete')
    col = np.where(g[..., None], alb * (moon + amb), col)
    # wall: breeze blocks
    w = (t == tw) & (tw < INF)
    bu, bv = p[..., 0] / 0.44, p[..., 1] / 0.22
    row = np.floor(bv)
    bu = bu + 0.5 * (row % 2)
    mortar = (np.minimum(np.abs((bu % 1.0) - 0.5), 1.0) > 0.47) | (np.abs((bv % 1.0) - 0.5) > 0.44)
    blk = 0.16 + 0.05 * _hash2(np.floor(bu), row, 3) + 0.05 * fbm(p[..., 0] * 6.0, p[..., 1] * 6.0, 3, 19)
    walb = np.where(mortar, 0.10, blk)
    wcol = np.stack([walb, walb * 0.98, walb * 0.96], -1) * (moon * 0.8 + amb)
    col = np.where(w[..., None], wcol, col)
    # drum: dark blue paint, rust, a highlight rim
    dr = (t == tc) & (tc < INF)
    ang = np.arctan2(p[..., 0] - cx, p[..., 2] - cz)
    rust = np.clip(fbm(ang * 3.0, p[..., 1] * 4.0, 4, 29) * 1.8 - 0.75, 0, 1)
    ribs = (np.abs((p[..., 1] / drum_h * 3.0) % 1.0 - 0.5) > 0.47) * 0.35
    paint = np.array([0.05, 0.09, 0.16])
    rustc = np.array([0.16, 0.07, 0.03])
    dalb = paint * (1 - rust[..., None]) + rustc * rust[..., None]
    nrm = np.stack([p[..., 0] - cx, np.zeros_like(ang), p[..., 2] - cz], -1) / drum_r
    ndl = np.clip(nrm @ np.array([-0.4, 0.3, 0.87]), 0, 1)[..., None]
    col = np.where(dr[..., None], dalb * (1 - ribs[..., None]) * (amb + moon * ndl), col)
    lid = (t == tl) & (tl < INF)
    col = np.where(lid[..., None], np.array([0.06, 0.08, 0.12]) * (moon + amb) * 3.0, col)
    # sky
    sky = (t >= INF)
    elev = np.clip(d[..., 1], 0, 1)[..., None]
    skyc = np.array([0.020, 0.024, 0.034]) + (np.array([0.004, 0.006, 0.013]) - np.array([0.020, 0.024, 0.034])) * np.power(elev, 0.45)
    col = np.where(sky[..., None], skyc, col)
    fogf = 1.0 - np.exp(-np.where(sky, 0.0, t) / 30.0)[..., None]
    col = col * (1 - fogf) + np.array([0.020, 0.024, 0.034]) * fogf
    depth = np.where(sky, 0.0, t)
    return _pack(col, depth)


# ---- the burning logo -------------------------------------------------------------------------------

def logo_plate(cs, W, H, logo_dir, fire_xform=None, ground='concrete'):
    """The BLACKBODY letters (dark steel, front faces) standing on a night ground, through camera cs.
    Returns (rgba uint8, depth m)."""
    import json
    import os
    L = json.load(open(os.path.join(logo_dir, 'layout.json')))
    mask = np.load(os.path.join(logo_dir, 'mask.npy'))
    mpp = L['mask_mpp']
    Hm, Wm = mask.shape
    zf = L['depth'] / 2.0
    eye, d = rays(cs, W, H)
    # ground first (the plate's own ground_plate), then the letters' front faces over it
    img, depth = ground_plate(cs, W, H, kind=ground, night=True, treeline=False)
    lin = np.power(img[..., :3].astype(np.float32) / 255.0, 2.2)
    tz = np.where(d[..., 2] < -1e-9, (zf - eye[2]) / np.where(d[..., 2] < -1e-9, d[..., 2], -1.0), -1.0)
    p = eye + d * np.maximum(tz, 0.0)[..., None]
    c = (p[..., 0] + L['width'] / 2) / mpp + 10
    r = (Hm - 10) - p[..., 1] / mpp
    inside = (tz > 0) & (c >= 0) & (c < Wm - 1) & (r >= 0) & (r < Hm - 1)
    ci = np.clip(c, 0, Wm - 1.001)
    ri = np.clip(r, 0, Hm - 1.001)
    c0, r0 = np.floor(ci).astype(int), np.floor(ri).astype(int)
    fc, fr = ci - c0, ri - r0
    m = (mask[r0, c0] * (1 - fc) * (1 - fr) + mask[r0, c0 + 1] * fc * (1 - fr) + mask[r0 + 1, c0] * (1 - fc) * fr
         + mask[r0 + 1, c0 + 1] * fc * fr)
    cover = np.where(inside, m, 0.0)
    # dark blued steel with brushed streaks and a little soot
    brush = fbm(p[..., 0].astype(np.float32) * 40.0, p[..., 1].astype(np.float32) * 2.0, 3, 91)
    soot = fbm(p[..., 0].astype(np.float32) * 3.0, p[..., 1].astype(np.float32) * 3.0, 4, 93)
    alb = (0.055 + 0.03 * brush) * (1.0 - 0.5 * np.clip(soot * 1.6 - 0.5, 0, 1))
    steel = np.stack([alb * 0.95, alb * 0.98, alb * 1.05], -1) * np.array([0.050, 0.060, 0.085]) * 6.0
    lin = lin * (1 - cover[..., None]) + steel * cover[..., None]
    depth = np.where(cover > 0.5, tz, depth)
    return _pack(lin, depth)
