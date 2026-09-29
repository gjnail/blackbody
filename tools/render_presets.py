"""Render every preset (or the named ones) through the Engine and save a frame of each."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image

from blackbody.engine.engine import Engine
from blackbody.scene import presets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('names', nargs='*')
    ap.add_argument('--frame', type=int, default=48, help='frame to render (from the first frame)')
    ap.add_argument('--size', default='960x540')
    ap.add_argument('--final', action='store_true')
    ap.add_argument('--samples', type=int, default=1)
    ap.add_argument('--mode', default='composite')
    ap.add_argument('--out', default='out/presets')
    ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--frames', default='', help='comma list of frames to save (default: just --frame)')
    args = ap.parse_args()
    W, H = (int(x) for x in args.size.split('x'))
    eng = Engine()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = args.names or presets.ORDER
    for name in names:
        sc = presets.make(name)
        sc.data['render']['width'], sc.data['render']['height'] = W, H
        for kv in args.set:
            k, v = kv.split('=')
            sec, key = k.split('.')
            sc.set((sec, key), v)
        frames = [int(x) for x in args.frames.split(',')] if args.frames else [args.frame]
        t0 = time.perf_counter()
        eng.prepare(sc, final=args.final)
        for fr in frames:
            target = sc.start + fr - 1
            eng.simulate_to(sc, target, cache=False)
            t1 = time.perf_counter()
            eng.render(sc, target, (W, H), mode=args.mode, final=args.final, samples=args.samples,
                       motion_blur=args.final)
            img = eng.display_image()
            Image.fromarray(np.ascontiguousarray(img[..., :3])).save(out / f'{name}_{fr:03d}.png')
            st = eng.stats()
            print(f'{name:14s} f{fr:3d} dims {st["dims"]} cell {st["cell_mm"]:.1f}mm vmax {st["max_speed"]:.1f} '
                  f'substeps {st["substeps"]} sim {st["sim_ms"]:.0f}ms render {1000*(time.perf_counter()-t1):.0f}ms '
                  f'total {time.perf_counter()-t0:.1f}s')


if __name__ == '__main__':
    main()
