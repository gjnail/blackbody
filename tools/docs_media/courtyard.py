"""A dusk courtyard plate with a handheld camera move, for the line-up and tracking demos: concrete paving in
1.2 m slabs (rectangles to line the ground up on), a brick wall, an oil drum. Procedural, rendered from a known
camera, with sensor noise, so the app's line-up and tracker work on it as on real footage.

    python courtyard.py OUT_DIR [--frames 144] [--size 1920x1080] [--move dolly|pan|still] [--test]

Writes OUT_DIR/court.####.png and OUT_DIR/truth.json (the camera of every frame, and the pixel corners of the slab
to line up on at the first frame). record_ui.py's lineup and track recordings use the dolly as out/court_dolly.mp4,
with its truth.json beside it as out/court_dolly.json; place and views use the locked-off one as out/court_still.mp4:

    python courtyard.py out/court_dolly --move dolly
    ffmpeg -framerate 24 -i out/court_dolly/court.%04d.png -c:v libx264 -crf 14 -pix_fmt yuv420p out/court_dolly.mp4
    (copy out/court_dolly/truth.json to out/court_dolly.json)
    python courtyard.py out/court_still --move still --frames 72      (and encode it the same way)"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)

from blackbody.engine import camera as cam   # noqa: E402
import plates as P                           # noqa: E402
import plates_fast as PF                     # noqa: E402

SLAB = 1.2
WALL_Z, WALL_H = -4.0, 2.6
DRUM = (1.5, 0.0, -1.6)
DRUM_R, DRUM_H = 0.30, 0.88
FOCAL = 26.0


def pose(move, s, rng_phase):
    """Camera position and rotation (degrees XYZ) at s in [0, 1]."""
    bob = np.array([0.012 * math.sin(2 * math.pi * (1.7 * s + rng_phase[0])) + 0.006 * math.sin(2 * math.pi * 5.3 * s),
                    0.010 * math.sin(2 * math.pi * (2.3 * s + rng_phase[1])),
                    0.0])
    if move == 'dolly':
        pos = np.array([-1.3 + 1.6 * s, 1.55, 5.2 - 2.0 * s]) + bob
        yaw = -12.0 + 9.0 * s
    elif move == 'pan':
        pos = np.array([-0.6, 1.55, 4.6]) + bob * 0.4
        yaw = -14.0 + 22.0 * s
    else:
        pos = np.array([-0.4, 1.55, 4.8])
        yaw = -6.0
    pitch = -15.0 + 0.6 * math.sin(2 * math.pi * (1.1 * s + rng_phase[2]))
    roll = 0.5 * math.sin(2 * math.pi * (0.9 * s + rng_phase[0]))
    return pos, (pitch, yaw, roll)


def cam_state(pos, rot, W, H):
    spec = cam.CameraSpec(mode='free', position=tuple(pos), rotation=tuple(rot), focal_mm=FOCAL, use_anchor=False)
    return cam.compute(spec, W / H, cam.FireXform())


def project(cs, pts, W, H):
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ np.asarray(cs.view_proj, float).T
    ndc = p[:, :2] / p[:, 3:4]
    return np.c_[(ndc[:, 0] + 1) / 2 * W, (1 - ndc[:, 1]) / 2 * H]


def joints(x, z, w=0.012):
    """0 on the slab joints, 1 elsewhere (soft-edged), with a little wander."""
    jx = x + 0.01 * np.sin(z * 3.1)
    jz = z + 0.01 * np.sin(x * 2.7)
    dx = np.abs(((jx + SLAB / 2) % SLAB) - SLAB / 2)
    dz = np.abs(((jz + SLAB / 2) % SLAB) - SLAB / 2)
    d = np.minimum(np.abs(dx - SLAB / 2), np.abs(dz - SLAB / 2))
    return np.clip((d - w * 0.5) / w, 0, 1)


def frame(cs, W, H, gt, wt, seed):
    eye, d = P.rays(cs, W, H)
    INF = 1e9
    tg = np.where(d[..., 1] < -1e-9, -eye[1] / np.where(d[..., 1] < -1e-9, d[..., 1], -1.0), INF)
    tw = np.where(d[..., 2] < -1e-9, (WALL_Z - eye[2]) / np.where(d[..., 2] < -1e-9, d[..., 2], -1.0), INF)
    pw = eye + d * np.where(tw < INF, tw, 0.0)[..., None]
    tw = np.where((tw > 0) & (pw[..., 1] > 0) & (pw[..., 1] < WALL_H), tw, INF)
    cx, cz = DRUM[0], DRUM[2]
    ox, oz = eye[0] - cx, eye[2] - cz
    a = d[..., 0] ** 2 + d[..., 2] ** 2
    b = 2 * (ox * d[..., 0] + oz * d[..., 2])
    c = ox * ox + oz * oz - DRUM_R ** 2
    disc = b * b - 4 * a * c
    tc = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * np.maximum(a, 1e-12))
    pc = eye + d * np.where(disc > 0, tc, 0.0)[..., None]
    tc = np.where((disc > 0) & (tc > 0) & (pc[..., 1] > 0) & (pc[..., 1] < DRUM_H), tc, INF)
    tl = np.where(np.abs(d[..., 1]) > 1e-9, (DRUM_H - eye[1]) / np.where(np.abs(d[..., 1]) > 1e-9, d[..., 1], 1.0), INF)
    pl = eye + d * np.where(tl < INF, tl, 0.0)[..., None]
    tl = np.where((tl > 0) & ((pl[..., 0] - cx) ** 2 + (pl[..., 2] - cz) ** 2 < DRUM_R ** 2), tl, INF)
    t = np.minimum(np.minimum(tg, tw), np.minimum(tc, tl))
    p = eye + d * np.where(t < INF, t, 0.0)[..., None]
    sky_l = np.array([0.105, 0.125, 0.175], np.float32)  # dusk sky light from above
    key = np.array([0.085, 0.058, 0.038], np.float32)     # the last of the sunset, low and warm
    L = np.array([-0.55, 0.22, -0.8])
    L = L / np.linalg.norm(L)
    col = np.zeros((H, W, 3), np.float32)
    g = (t == tg) & (tg < INF)
    if g.any():
        q = p[g]
        alb = gt.sample(q[:, 0].astype(np.float32), q[:, 2].astype(np.float32), PF._footprint(cs, W, tg[g], d[g][:, 1]))
        alb = alb * (0.55 + 0.45 * joints(q[:, 0], q[:, 2], 0.009))[:, None]
        # the drum's shadow and contact darkening, and darkening along the foot of the wall
        rel = np.stack([cx - q[:, 0], cz - q[:, 2]], -1)
        lh = L[[0, 2]] / np.linalg.norm(L[[0, 2]])
        s = rel @ (-lh)
        dist = np.linalg.norm(rel - s[:, None] * (-lh), axis=-1)
        sh = np.clip((DRUM_R + 0.04 - dist) / 0.08, 0, 1) * (s < 0) * np.clip(1 - (-s) / 3.0, 0, 1)
        rd = np.maximum(np.hypot(q[:, 0] - cx, q[:, 2] - cz) - DRUM_R, 0.0)
        ao = (1 - 0.5 * np.exp(-rd / 0.07)) * (1 - 0.4 * np.exp(-np.maximum(q[:, 2] - WALL_Z, 0.0) / 0.3))
        col[g] = alb * (sky_l * ao[:, None] + key * L[1] * 2.0 * (1 - 0.9 * sh[:, None]))
    w = (t == tw) & (tw < INF)
    if w.any():
        ao = 1 - 0.35 * np.exp(-p[w][:, 1] / 0.3)
        col[w] = (wt.sample(p[w][:, 0].astype(np.float32), p[w][:, 1].astype(np.float32), PF._footprint(cs, W, tw[w], d[w][:, 2]))
                  * (sky_l * 0.75 + key * 0.9) * ao[:, None])
    dr = (t == tc) & (tc < INF)
    if dr.any():
        q = p[dr]
        ang = np.arctan2(q[:, 0] - cx, q[:, 2] - cz).astype(np.float32)
        yy = q[:, 1].astype(np.float32)
        rust = np.clip(P.fbm(ang * 3.0, yy * 4.0, 4, 29) * 2.2 - 0.95, 0, 1)
        dalb = np.array([0.06, 0.12, 0.24]) * (1 - rust[:, None]) + np.array([0.20, 0.085, 0.035]) * rust[:, None]
        nrm = np.stack([q[:, 0] - cx, np.zeros_like(ang), q[:, 2] - cz], -1) / DRUM_R
        ndl = np.clip(nrm @ L, 0, 1)[:, None]
        col[dr] = dalb * (sky_l * 0.8 + key * 2.0 * ndl)
    lid = (t == tl) & (tl < INF)
    if lid.any():
        q = p[lid]
        rx, rz = q[:, 0] - cx, q[:, 2] - cz
        rr = np.hypot(rx, rz) / DRUM_R
        shade = np.where(rr > 0.93, 1.35, np.where((rr > 0.85) & (rr < 0.89), 0.55, 1.0))
        col[lid] = np.array([0.06, 0.12, 0.24]) * shade[:, None] * (sky_l * 1.1 + key)
    sky = t >= INF
    elev = np.clip(d[..., 1], 0, 1)[..., None]
    hor, zen = np.array([0.12, 0.085, 0.075]), np.array([0.012, 0.018, 0.04])
    skyc = hor + (zen - hor) * np.power(elev, 0.4)
    col = np.where(sky[..., None], skyc, col)
    fogf = 1.0 - np.exp(-np.where(sky, 0.0, t) / 60.0)[..., None]
    col = col * (1 - fogf) + hor * 0.6 * fogf
    # the lens: a gentle vignette, then sensor noise (more in the shadows)
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
    col = col * (1 - 0.18 * r2)[..., None]
    rng = np.random.default_rng(seed)
    srgb = np.clip(P.to_srgb(np.clip(col, 0, None)), 0, 1)
    srgb = srgb + rng.normal(0, 1, srgb.shape) * (0.006 + 0.012 * (1 - srgb))
    return (np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8), np.where(sky, 0.0, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--frames', type=int, default=144)
    ap.add_argument('--size', default='1920x1080')
    ap.add_argument('--move', default='dolly')
    ap.add_argument('--test', action='store_true')
    args = ap.parse_args()
    W, H = (int(v) for v in args.size.split('x'))
    os.makedirs(args.out, exist_ok=True)
    gt = PF.GroundTex('concrete', 8.0)
    wt = PF.Tex2D('wall2', PF._wall_albedo, -10.0, 0.0, 20.0, 3.2, 4096, 656)
    rng = np.random.default_rng(5)
    phase = rng.random(3)
    n = args.frames
    idx = [0, n // 2, n - 1] if args.test else range(n)
    truth = {'focal_mm': FOCAL, 'size': [W, H], 'frames': {}}
    for i in idx:
        s = i / max(n - 1, 1)
        pos, rot = pose(args.move, s, phase)
        cs = cam_state(pos, rot, W, H)
        img, _ = frame(cs, W, H, gt, wt, seed=i)
        Image.fromarray(img).save(os.path.join(args.out, f'court.{i + 1:04d}.png'))
        truth['frames'][i + 1] = {'position': [float(v) for v in pos], 'rotation': list(rot)}
        if i == 0:
            # the slab in front of the drum to line up on, and the one under the middle of the frame
            for name, (x0, z0) in {'slab': (0.0, 0.0), 'slab2': (-1.2, 0.0)}.items():
                corners = [(x0, 0, z0), (x0 + SLAB, 0, z0), (x0 + SLAB, 0, z0 + SLAB), (x0, 0, z0 + SLAB)]
                truth[name] = project(cs, corners, W, H).tolist()
        print('frame', i + 1, flush=True)
    with open(os.path.join(args.out, 'truth.json'), 'w') as f:
        json.dump(truth, f, indent=1)


if __name__ == '__main__':
    main()
