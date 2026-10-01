"""Fast versions of the plates: textures are baked once per shot into world-space maps (cached on disk,
with mip levels) and sampled per frame, instead of evaluating the noise at every pixel of every frame."""
from __future__ import annotations

import json
import os

import numpy as np

import plates as P

# baked ground textures (hundreds of MB): kept out of the repo, under out/
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'out', 'docs_media', 'texcache')
_MEM = {}


def _gen(fn, x0, y0, sx, sy, rx, ry, chunk=192):
    gx = (x0 + (np.arange(rx, dtype=np.float64) + 0.5) * (sx / rx)).astype(np.float32)
    gy = (y0 + (np.arange(ry, dtype=np.float64) + 0.5) * (sy / ry)).astype(np.float32)
    out = np.empty((ry, rx, 3), np.float16)
    for r in range(0, ry, chunk):
        vv, uu = np.meshgrid(gy[r:r + chunk], gx, indexing='ij')
        out[r:r + chunk] = np.asarray(fn(uu, vv), np.float32).astype(np.float16)
    return out


class Tex2D:
    """A texture over the world rectangle [x0, x0+sx] x [y0, y0+sy] (u along x, v along y), with mips."""

    def __init__(self, name, fn, x0, y0, sx, sy, rx, ry):
        key = f'{name}_{x0:.2f}_{y0:.2f}_{sx:.2f}_{sy:.2f}_{rx}x{ry}'
        mips = _MEM.get(key)
        if mips is None:
            os.makedirs(CACHE, exist_ok=True)
            path = os.path.join(CACHE, key + '.npy')
            if os.path.exists(path):
                tex = np.load(path)
            else:
                tex = _gen(fn, x0, y0, sx, sy, rx, ry)
                np.save(path, tex)
            mips = [tex.astype(np.float32)]
            while min(mips[-1].shape[:2]) >= 16 and len(mips) < 9:
                t = mips[-1]
                h, w = t.shape[0] // 2 * 2, t.shape[1] // 2 * 2
                mips.append(t[:h, :w].reshape(h // 2, 2, w // 2, 2, 3).mean((1, 3)))
            _MEM[key] = mips
        self.mips = mips
        self.x0, self.y0, self.sx, self.sy = x0, y0, sx, sy
        self.texel = sx / rx

    def sample(self, u, v, fp):
        """Bilinear from the mip whose texels match the footprint fp (m) of each point."""
        lvl = np.clip(np.floor(np.log2(np.maximum(fp / self.texel, 1.0))), 0, len(self.mips) - 1).astype(np.int32)
        out = np.zeros(u.shape + (3,), np.float32)
        for L in np.unique(lvl):
            m = lvl == L
            tex = self.mips[L]
            ny, nx = tex.shape[:2]
            x = np.clip((u[m] - self.x0) / self.sx * nx - 0.5, 0, nx - 1.001)
            y = np.clip((v[m] - self.y0) / self.sy * ny - 0.5, 0, ny - 1.001)
            xi, yi = x.astype(np.int32), y.astype(np.int32)
            fx, fy = (x - xi)[:, None], (y - yi)[:, None]
            out[m] = (tex[yi, xi] * (1 - fx) * (1 - fy) + tex[yi, xi + 1] * fx * (1 - fy)
                      + tex[yi + 1, xi] * (1 - fx) * fy + tex[yi + 1, xi + 1] * fx * fy)
        return out


class GroundTex:
    """The ground's albedo: a fine map near the action and a coarse one out to the horizon."""

    def __init__(self, kind, near_half=12.0, far_half=260.0, near_res=3072, far_res=2048):
        fn = lambda u, v: P._ground_tex(u, v, kind)
        self.nh = near_half
        self.near = Tex2D(kind, fn, -near_half, -near_half, 2 * near_half, 2 * near_half, near_res, near_res)
        self.far = Tex2D(kind, fn, -far_half, -far_half, 2 * far_half, 2 * far_half, far_res, far_res)

    def sample(self, u, v, fp):
        out = self.far.sample(u, v, fp)
        r = np.maximum(np.abs(u), np.abs(v)) / self.nh
        near = r < 1.0
        if near.any():
            w = np.clip((1.0 - r[near]) / 0.12, 0, 1)[:, None]
            out[near] = self.near.sample(u[near], v[near], fp[near]) * w + out[near] * (1 - w)
        return out


def _footprint(cs, W, t, dy):
    pa = 2.0 * np.tan(cs.hfov / 2.0) / W
    return t * pa / np.maximum(np.abs(dy), 0.06)


def ground_plate(cs, W, H, kind='dirt', night=True, horizon=None, fog=40.0, sky=None, seed=0, treeline=False, near_half=12.0,
                 gain=1.0, zenith=None):
    eye, d = P.rays(cs, W, H)
    t = -eye[1] / np.where(np.abs(d[..., 1]) < 1e-9, -1e-9, d[..., 1])
    hit = (d[..., 1] < 0) & (t > 0)
    t = np.where(hit, t, 0.0)
    if night:
        light = np.array([0.050, 0.060, 0.085])
        zen = np.array([0.004, 0.006, 0.013]) if zenith is None else np.asarray(zenith)
        hor = np.array([0.020, 0.024, 0.034]) if horizon is None else np.asarray(horizon)
    else:
        light = np.array([1.05, 1.0, 0.92])
        zen = np.array([0.18, 0.30, 0.55]) if zenith is None else np.asarray(zenith)
        hor = np.array([0.62, 0.70, 0.80]) if horizon is None else np.asarray(horizon)
    elev = np.clip(d[..., 1], 0, 1)[..., None]
    col = hor + (zen - hor) * np.power(elev, 0.45)
    if sky is not None:
        col = col * np.asarray(sky)
    col = col.astype(np.float32)
    if hit.any():
        q = eye + d[hit] * t[hit][:, None]
        if kind == 'sky':
            alb = np.full((len(q), 3), 0.18, np.float32)
        else:
            g = GroundTex(kind, near_half)
            alb = g.sample(q[:, 0].astype(np.float32), q[:, 2].astype(np.float32), _footprint(cs, W, t[hit], d[hit][:, 1]))
        fogf = (1.0 - np.exp(-t[hit] / fog))[:, None]
        col[hit] = alb * light * gain * (1 - fogf) + hor * fogf
    return P._pack(col, np.where(hit, t, 0.0))


def _wall_albedo(u, v):
    bu, bv = u / 0.44, v / 0.22
    row = np.floor(bv)
    bu = bu + 0.5 * (row % 2)
    ci = np.floor(bu)
    fu, fv = bu % 1.0, bv % 1.0
    jit = 0.012 * P.fbm(u * 40.0, v * 40.0, 2, 51)
    mortar = (np.abs(fu - 0.5) > 0.47 - jit) | (np.abs(fv - 0.5) > 0.44 - jit)
    h1, h2 = P._hash2(ci, row, 3), P._hash2(ci, row, 7)
    blk = 0.15 + 0.07 * h1 + 0.05 * P.fbm(u * 6.0, v * 6.0, 3, 19)
    blk = blk * (0.82 + 0.36 * P.fbm(u * 30.0, v * 30.0, 2, 23))
    blk = blk * (1 - 0.35 * np.clip(P.fbm(u * 90.0, v * 90.0, 2, 29) * 3.0 - 2.1, 0, 1))   # pitted block face
    # relief: each block's top edge catches the light from above, its underside is shaded
    blk = blk * (1 + 0.18 * np.clip((fv - 0.87) / 0.07, 0, 1) - 0.22 * np.clip((0.13 - fv) / 0.07, 0, 1))
    walb = np.where(mortar, 0.085 + 0.02 * P.fbm(u * 50.0, v * 50.0, 2, 57), blk)
    # grime: streaks run down from the top, damp and dirt rise at the foot
    streaks = np.clip(P.fbm(u * 4.0, v * 0.35, 4, 61) * 1.6 - 0.55, 0, 1)
    damp = np.exp(-v / 0.45) * (0.6 + 0.4 * P.fbm(u * 2.0, v * 3.0, 3, 67))
    walb = walb * (1 - 0.35 * streaks - 0.45 * damp)
    return np.stack([walb * (1 + 0.06 * (h2 - 0.5)), walb * 0.98, walb * (0.96 - 0.04 * (h2 - 0.5))], -1)


def yard_plate(cs, W, H, drum=(-0.9, 0.0, 1.4), drum_r=0.30, drum_h=0.88, wall_z=-3.2, near_half=8.0):
    eye, d = P.rays(cs, W, H)
    INF = 1e9
    tg = np.where(d[..., 1] < -1e-9, -eye[1] / np.where(d[..., 1] < -1e-9, d[..., 1], -1.0), INF)
    tw = np.where(d[..., 2] < -1e-9, (wall_z - eye[2]) / np.where(d[..., 2] < -1e-9, d[..., 2], -1.0), INF)
    pw = eye + d * np.where(tw < INF, tw, 0.0)[..., None]
    tw = np.where((tw > 0) & (pw[..., 1] > 0) & (pw[..., 1] < 3.2), tw, INF)
    cx, cz = drum[0], drum[2]
    ox, oz = eye[0] - cx, eye[2] - cz
    a = d[..., 0] ** 2 + d[..., 2] ** 2
    b = 2 * (ox * d[..., 0] + oz * d[..., 2])
    c = ox * ox + oz * oz - drum_r * drum_r
    disc = b * b - 4 * a * c
    tc = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * np.maximum(a, 1e-12))
    pc = eye + d * np.where(disc > 0, tc, 0.0)[..., None]
    tc = np.where((disc > 0) & (tc > 0) & (pc[..., 1] > 0) & (pc[..., 1] < drum_h), tc, INF)
    tl = np.where(np.abs(d[..., 1]) > 1e-9, (drum_h - eye[1]) / np.where(np.abs(d[..., 1]) > 1e-9, d[..., 1], 1.0), INF)
    pl = eye + d * np.where(tl < INF, tl, 0.0)[..., None]
    tl = np.where((tl > 0) & ((pl[..., 0] - cx) ** 2 + (pl[..., 2] - cz) ** 2 < drum_r ** 2), tl, INF)
    t = np.minimum(np.minimum(tg, tw), np.minimum(tc, tl))
    p = eye + d * np.where(t < INF, t, 0.0)[..., None]
    moon = np.array([0.072, 0.088, 0.125], np.float32)
    amb = np.array([0.028, 0.033, 0.046], np.float32)
    Lm = np.array([-0.4, 0.3, 0.87])
    Lm = Lm / np.linalg.norm(Lm)
    paint = np.array([0.06, 0.12, 0.24])
    rustc = np.array([0.20, 0.085, 0.035])
    col = np.zeros((H, W, 3), np.float32)
    g = (t == tg) & (tg < INF)
    if g.any():
        gt = GroundTex('concrete', near_half)
        q = p[g]
        alb = gt.sample(q[:, 0].astype(np.float32), q[:, 2].astype(np.float32), _footprint(cs, W, tg[g], d[g][:, 1]))
        # the drum's long moon shadow, softening with distance; contact darkening at its foot and along the wall
        hl = np.linalg.norm(Lm[[0, 2]])
        lh = Lm[[0, 2]] / hl
        rel = np.stack([cx - q[:, 0], cz - q[:, 2]], -1)
        s = rel @ lh
        dist = np.linalg.norm(rel - s[:, None] * lh, axis=-1)
        pen = 0.015 + 0.03 * np.maximum(s, 0.0)
        sh = (np.clip((drum_r + pen - dist) / (2 * pen), 0, 1) * np.clip((drum_h - s * Lm[1] / hl) / 0.06 + 0.5, 0, 1)
              * (s > 0))
        rd = np.maximum(np.hypot(q[:, 0] - cx, q[:, 2] - cz) - drum_r, 0.0)
        ao = (1 - 0.55 * np.exp(-rd / 0.06)) * (1 - 0.35 * np.exp(-np.maximum(q[:, 2] - wall_z, 0.0) / 0.25))
        col[g] = alb * (moon * (1 - 0.85 * sh[:, None]) + amb) * ao[:, None]
    w = (t == tw) & (tw < INF)
    if w.any():
        wt = Tex2D('wall2', _wall_albedo, -10.0, 0.0, 20.0, 3.2, 4096, 656)
        ao = 1 - 0.35 * np.exp(-p[w][:, 1] / 0.3)
        col[w] = (wt.sample(p[w][:, 0].astype(np.float32), p[w][:, 1].astype(np.float32), _footprint(cs, W, tw[w], d[w][:, 2]))
                  * (moon * 0.8 + amb) * ao[:, None])
    dr = (t == tc) & (tc < INF)
    if dr.any():
        q = p[dr]
        ang = np.arctan2(q[:, 0] - cx, q[:, 2] - cz).astype(np.float32)
        yy = q[:, 1].astype(np.float32)
        rust = np.clip(P.fbm(ang * 3.0, yy * 4.0, 4, 29) * 2.2 - 0.95, 0, 1)
        foot = np.clip((0.14 - yy) / 0.14, 0, 1) * np.clip(P.fbm(ang * 9.0, yy * 20.0, 3, 31) * 2.0 - 0.3, 0, 1)
        runs = np.clip(P.fbm(ang * 40.0, yy * 1.5, 3, 37) * 1.6 - 0.9, 0, 1) * 0.5
        rust = np.clip(np.maximum(rust, 0.8 * foot) + runs, 0, 1)
        dalb = paint * (1 - rust[:, None]) + rustc * rust[:, None]
        dalb = dalb * (0.85 + 0.3 * P.fbm(ang * 20.0, yy * 20.0, 2, 41))[:, None]
        nrm = np.stack([q[:, 0] - cx, np.zeros_like(ang), q[:, 2] - cz], -1) / drum_r
        # the two rolling hoops and the rolled top rim: raised bands, lit on top and shaded underneath
        ny = np.zeros_like(yy)
        for yh, hw in ((drum_h / 3, 0.014), (2 * drum_h / 3, 0.014), (drum_h - 0.012, 0.012)):
            sb = (yy - yh) / hw
            ny += np.where(np.abs(sb) < 1, np.sin(sb * np.pi) * 0.9, 0.0)
        n2 = nrm + ny[:, None] * np.array([0.0, 1.0, 0.0])
        n2 = n2 / np.linalg.norm(n2, axis=-1, keepdims=True)
        ndl = np.clip(n2 @ Lm, 0, 1)[:, None]
        v = -d[dr] / np.linalg.norm(d[dr], axis=-1, keepdims=True)
        hv = v + Lm
        hv = hv / np.linalg.norm(hv, axis=-1, keepdims=True)
        spec = np.clip(np.sum(n2 * hv, -1), 0, 1) ** 40 * (1 - rust) * 0.5
        col[dr] = dalb * (amb + moon * ndl) + moon * spec[:, None]
    lid = (t == tl) & (tl < INF)
    if lid.any():
        q = p[lid]
        rx, rz = q[:, 0] - cx, q[:, 2] - cz
        rr = np.hypot(rx, rz) / drum_r
        rust = np.clip(P.fbm((rx * 8).astype(np.float32), (rz * 8).astype(np.float32), 3, 43) * 2.0 - 0.9, 0, 1)
        lalb = paint * (1 - rust[:, None]) + rustc * rust[:, None]
        shade = np.where(rr > 0.93, 1.4, np.where((rr > 0.85) & (rr < 0.89), 0.55, 1.0))
        shade = shade * np.where(np.hypot(rx - 0.14, rz + 0.08) < 0.035, 0.45, 1.0)
        col[lid] = lalb * shade[:, None] * (amb + moon * Lm[1] * 1.6)
    sky = t >= INF
    elev = np.clip(d[..., 1], 0, 1)[..., None]
    skyc = np.array([0.020, 0.024, 0.034]) + (np.array([0.004, 0.006, 0.013]) - np.array([0.020, 0.024, 0.034])) * np.power(elev, 0.45)
    col = np.where(sky[..., None], skyc, col)
    fogf = 1.0 - np.exp(-np.where(sky, 0.0, t) / 30.0)[..., None]
    col = col * (1 - fogf) + np.array([0.020, 0.024, 0.034]) * fogf
    return P._pack(col, np.where(sky, 0.0, t))


def logo_plate(cs, W, H, logo_dir, ground='concrete', near_half=12.0):
    L = json.load(open(os.path.join(logo_dir, 'layout.json')))
    mask = _MEM.get('logo_mask')
    if mask is None:
        mask = _MEM['logo_mask'] = np.load(os.path.join(logo_dir, 'mask.npy'))
    mpp = L['mask_mpp']
    Hm, Wm = mask.shape
    zf = L['depth'] / 2.0
    eye, d = P.rays(cs, W, H)
    img, depth = ground_plate(cs, W, H, kind=ground, night=True, near_half=near_half)
    lin = np.power(img[..., :3].astype(np.float32) / 255.0, 2.2)
    tz = np.where(d[..., 2] < -1e-9, (zf - eye[2]) / np.where(d[..., 2] < -1e-9, d[..., 2], -1.0), -1.0)
    p = eye + d * np.maximum(tz, 0.0)[..., None]
    c = (p[..., 0] + L['width'] / 2) / mpp + 10
    r = (Hm - 10) - p[..., 1] / mpp
    inside = (tz > 0) & (c >= 0) & (c < Wm - 1) & (r >= 0) & (r < Hm - 1)
    ci = np.clip(c[inside], 0, Wm - 1.001)
    ri = np.clip(r[inside], 0, Hm - 1.001)
    c0, r0 = ci.astype(int), ri.astype(int)
    fc, fr = ci - c0, ri - r0
    m = (mask[r0, c0] * (1 - fc) * (1 - fr) + mask[r0, c0 + 1] * fc * (1 - fr) + mask[r0 + 1, c0] * (1 - fc) * fr
         + mask[r0 + 1, c0 + 1] * fc * fr)
    cover = np.zeros((H, W), np.float32)
    cover[inside] = m
    on = cover > 0
    if on.any():
        q = p[on]
        brush = P.fbm((q[:, 0] * 40.0).astype(np.float32), (q[:, 1] * 2.0).astype(np.float32), 3, 91)
        soot = P.fbm((q[:, 0] * 3.0).astype(np.float32), (q[:, 1] * 3.0).astype(np.float32), 4, 93)
        alb = (0.055 + 0.03 * brush) * (1.0 - 0.5 * np.clip(soot * 1.6 - 0.5, 0, 1))
        steel = np.stack([alb * 0.95, alb * 0.98, alb * 1.05], -1) * np.array([0.050, 0.060, 0.085]) * 6.0
        cv = cover[on][:, None]
        lin[on] = lin[on] * (1 - cv) + steel * cv
    depth = np.where(cover > 0.5, tz, depth)
    return P._pack(lin, depth)
