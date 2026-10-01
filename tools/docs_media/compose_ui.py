"""Turn a recording (out/docs_media/ui/NAME: grabbed frames + meta.json) into a steady 24 fps sequence with the
cursor, clicks and captions drawn in: out/docs_media/ui_out/NAME/####.png.

    python compose_ui.py NAME [--trim START END] [--scale 1.0]"""
import argparse
import bisect
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WORK = Path(__file__).resolve().parents[2] / 'out' / 'docs_media'
FPS = 24
ARROW = [(0, 0), (0, 17), (4.2, 13.2), (7.2, 20), (10, 18.8), (7.1, 12.2), (12.5, 12.2)]


def font(size):
    for name in ('segoeuib.ttf', 'arialbd.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def cursor_at(cur, ts, t):
    i = bisect.bisect_right(ts, t) - 1
    if i < 0:
        return cur[0][1], cur[0][2], cur[0][3]
    if i >= len(cur) - 1:
        return cur[-1][1], cur[-1][2], cur[-1][3]
    (t0, x0, y0, d0), (t1, x1, y1, _) = cur[i], cur[i + 1]
    a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
    return x0 + (x1 - x0) * a, y0 + (y1 - y0) * a, d0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--trim', nargs=2, type=float, default=None, help='seconds to keep, from START to END')
    args = ap.parse_args()
    src = WORK / 'ui' / args.name
    meta = json.loads((src / 'meta.json').read_text())
    out = WORK / 'ui_out' / args.name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    frames = meta['frames']
    ft = [f[0] for f in frames]
    cur = meta['cursor']
    ct = [c[0] for c in cur]
    t_start = frames[0][0]
    t_end = meta['end']
    if args.trim:
        t_start, t_end = max(t_start, args.trim[0]), min(t_end, args.trim[1])
    n = int((t_end - t_start) * FPS)
    cache = {}
    fcap = font(26)
    for k in range(n):
        t = t_start + k / FPS
        i = max(0, bisect.bisect_right(ft, t) - 1)
        name = frames[i][1]
        if name not in cache:
            cache.clear()
            cache[name] = Image.open(src / name).convert('RGB')
        img = cache[name].copy()
        d = ImageDraw.Draw(img, 'RGBA')
        for (t0, x, y) in meta['ripples']:
            a = (t - t0) / 0.45
            if 0 <= a <= 1:
                r = 8 + 26 * a
                d.ellipse((x - r, y - r, x + r, y + r), outline=(255, 150, 50, int(255 * (1 - a))), width=3)
        for (t0, text, dur) in meta['captions']:
            if t0 <= t < t0 + dur:
                fade = min(1.0, (t - t0) / 0.15, (t0 + dur - t) / 0.15)
                tw = d.textlength(text, font=fcap)
                W, H = img.size
                x0, y0 = (W - tw) / 2 - 22, H - 170
                d.rounded_rectangle((x0, y0, x0 + tw + 44, y0 + 52), 12, fill=(15, 15, 18, int(220 * fade)),
                                    outline=(255, 150, 50, int(255 * fade)), width=2)
                d.text((x0 + 22, y0 + 26), text, font=fcap, fill=(245, 245, 245, int(255 * fade)), anchor='lm')
        x, y, down = cursor_at(cur, ct, t)
        s = 1.3
        pts = [(x + px * s, y + py * s) for px, py in ARROW]
        if down:
            d.ellipse((x - 12, y - 12, x + 12, y + 12), fill=(255, 150, 50, 95))
        d.polygon([(px + 1.5, py + 2.0) for px, py in pts], fill=(0, 0, 0, 100))
        d.polygon(pts, fill=(255, 255, 255, 255), outline=(10, 10, 10, 255))
        img.save(out / f'{k:04d}.png', compress_level=1)
    print(args.name, n, 'frames', round(n / FPS, 1), 's; captured', len(frames), 'at',
          round(len(frames) / max(1e-6, frames[-1][0] - frames[0][0]), 1), 'fps')


if __name__ == '__main__':
    main()
