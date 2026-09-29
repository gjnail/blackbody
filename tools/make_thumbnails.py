"""Render the preset library thumbnails (fire over black; liquids over a neutral ground; final quality)."""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from blackbody.engine.engine import Engine
from blackbody.scene import presets

FRAMES = {'campfire': 60, 'bonfire': 72, 'torch': 48, 'candle': 48, 'gas_ring': 36, 'pool_fire': 72, 'fire_line': 60,
          'fireball': 22, 'flamethrower': 48, 'vehicle_fire': 72, 'smoke_plume': 96,
          'fire_whirl': 144, 'waved_torch': 42, 'hose_douse': 72, 'grass_fire': 168, 'curtain_fire': 144, 'armchair_fire': 312,
          'room_fire': 170, 'coloured_flames': 48, 'road_flare': 48, 'grinder_sparks': 24, 'fireworks': 30, 'car_through_smoke': 72,
          'kettle_steam': 60, 'steam_vent': 96,
          'water_pour': 45, 'rock_splash': 19, 'bucket_throw': 12, 'fountain': 48, 'hose': 36, 'wave': 30, 'waterfall': 36,
          'spill': 9}


def ground_plate(cs, W, H):
    """A neutral backplate seen through the shot camera (paving slabs to the horizon, a pale sky), so
    clear liquid, which shows only what it bends and reflects, has something to show."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64) + 0.5
    ndc = np.stack([xs / W * 2 - 1, 1 - ys / H * 2], -1)
    ones = np.ones((H, W, 1))
    a = np.concatenate([ndc, 0 * ones, ones], -1) @ cs.inv_view_proj.T
    b = np.concatenate([ndc, ones, ones], -1) @ cs.inv_view_proj.T
    a, b = a[..., :3] / a[..., 3:], b[..., :3] / b[..., 3:]
    d = b - a
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    t = -a[..., 1] / np.where(np.abs(d[..., 1]) < 1e-9, -1e-9, d[..., 1])
    hit = (d[..., 1] < 0) & (t > 0)
    q = a + d * t[..., None]
    u, v = q[..., 0] / 0.4, q[..., 2] / 0.4
    fu, fv = u - np.floor(u), v - np.floor(v)
    joint = np.minimum(np.minimum(fu, 1 - fu), np.minimum(fv, 1 - fv)) < 0.02
    rnd = (np.sin(np.floor(u) * 12.9898 + np.floor(v) * 78.233) * 43758.5453) % 1.0
    slab = np.stack([0.34 + 0.06 * rnd, 0.32 + 0.05 * rnd, 0.29 + 0.04 * rnd], -1)
    ground = np.where(joint[..., None], 0.16, slab)
    fog = np.exp(-np.maximum(t, 0) / 60.0)[..., None]
    sky = np.stack([0.52 + 0.2 * d[..., 1], 0.6 + 0.2 * d[..., 1], 0.72 + 0.18 * d[..., 1]], -1)
    img = np.where(hit[..., None], ground * fog + sky * 0.7 * (1 - fog), sky)
    out = np.full((H, W, 4), 255, np.uint8)
    out[..., :3] = (np.clip(img, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('names', nargs='*')
    ap.add_argument('--out', default=str(ROOT / 'blackbody' / 'assets' / 'presets'))
    ap.add_argument('--size', default='640x360')
    ap.add_argument('--samples', type=int, default=4)
    ap.add_argument('--mode', default='fire')
    ap.add_argument('--draft', action='store_true')
    ap.add_argument('--frame', type=int, default=None)
    args = ap.parse_args()
    W, H = (int(x) for x in args.size.split('x'))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eng = Engine()
    for name in args.names or presets.ORDER:
        t0 = time.perf_counter()
        sc = presets.make(name)
        sc.data['render']['width'], sc.data['render']['height'] = W, H
        f = sc.start + (args.frame or FRAMES.get(name, 60)) - 1
        eng.prepare(sc, final=not args.draft)
        eng.simulate_to(sc, f, cache=False)
        if sc.kind == 'liquid':
            from blackbody.engine import camera as cam
            spec, fire = sc.camera(f)
            eng.render(sc, f, (W, H), mode='composite', final=not args.draft, samples=args.samples, motion_blur=True,
                       plate=ground_plate(cam.compute(spec, W / H, fire), W, H))
        else:
            eng.render(sc, f, (W, H), mode=args.mode, final=not args.draft, samples=args.samples, motion_blur=True)
        img = eng.display_image()
        Image.fromarray(np.ascontiguousarray(img[..., :3])).save(out / f'{name}.png')
        st = eng.stats()
        print(f'{name:14s} frame {f} dims {st["dims"]} {time.perf_counter() - t0:.1f}s')


if __name__ == '__main__':
    main()
