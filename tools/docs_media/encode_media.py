"""Encode rendered clips, promo shots and UI recordings into the repo's docs/media:
video/NAME.mp4 (+ NAME.jpg poster) for the site, gif/NAME.gif for the README and the guides on GitHub.

    python encode_media.py clip NAME [NAME ...]          # out/docs_media/clips/NAME/####.png (1280x720)
    python encode_media.py promo NAME[:OUTNAME] ...      # the promo's rendered shots (1920x1080; set BLACKBODY_PROMO_SHOTS)
    python encode_media.py ui NAME [NAME ...]            # out/docs_media/ui_out/NAME/####.png (1600x900)
Options: --gif-seconds S (GIF length, from --gif-start, both in seconds of the source), --gif-speed X (play the GIF X times
faster: a long screen recording in a short GIF), --gif-width W, --gif-max MB, --no-gif, --poster F (frame index).
BLACKBODY_MEDIA_WORK points at another out/docs_media (a frozen copy of the repo the media was made in)."""
import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get('BLACKBODY_MEDIA_WORK', str(ROOT / 'out' / 'docs_media')))
DOCS = ROOT / 'docs' / 'media'
PROMO = Path(os.environ.get('BLACKBODY_PROMO_SHOTS', str(WORK / 'promo_shots')))
FFMPEG = shutil.which('ffmpeg') or 'ffmpeg'


def run(*args):
    subprocess.run([FFMPEG, '-v', 'error', '-y', *args], check=True)


def frames_of(folder):
    fs = sorted(p for p in folder.glob('[0-9][0-9][0-9][0-9].png'))
    if not fs:
        raise SystemExit(f'no frames in {folder}')
    return fs


def encode(name, folder, kind, a):
    fs = frames_of(folder)
    n = len(fs)
    (DOCS / 'video').mkdir(parents=True, exist_ok=True)
    (DOCS / 'gif').mkdir(parents=True, exist_ok=True)
    pattern = str(folder / '%04d.png')
    first = int(fs[0].stem)
    mp4 = DOCS / 'video' / f'{name}.mp4'
    if kind == 'ui':
        vf, crf, tune = 'scale=1600:900:flags=lanczos', '22', 'animation'
    else:
        vf, crf, tune = 'scale=1280:720:flags=lanczos', '23', 'film'
    run('-framerate', '24', '-start_number', str(first), '-i', pattern, '-vf', vf + ',format=yuv420p', '-c:v', 'libx264', '-preset', 'slow',
        '-crf', crf, '-tune', tune, '-profile:v', 'high', '-movflags', '+faststart', '-an', str(mp4))
    pi = a.poster if a.poster is not None else (n * 2) // 3
    im = Image.open(fs[min(pi, n - 1)]).convert('RGB')
    im = im.resize((1280, 720) if kind != 'ui' else (1600, 900), Image.LANCZOS)
    im.save(DOCS / 'video' / f'{name}.jpg', quality=84, optimize=True, progressive=True)
    out = [f'{mp4.name} {mp4.stat().st_size / 1e6:.2f} MB']
    if not a.no_gif:
        width = a.gif_width or (960 if kind == 'ui' else 480)
        fps = 10 if kind == 'ui' else 12
        secs = a.gif_seconds or (12.0 if kind == 'ui' else 4.0)
        start = a.gif_start if a.gif_start is not None else max(0.0, (n / 24 - secs) / 2)
        speed = a.gif_speed or 1.0
        gif = DOCS / 'gif' / f'{name}.gif'
        for attempt in range(4):
            filt = (f'trim=start={start:.3f}:duration={secs:.3f},setpts=(PTS-STARTPTS)/{speed:.3f},fps={fps},'
                    f'scale={width}:-1:flags=lanczos,split[a][b];'
                    f'[a]palettegen=stats_mode=full:max_colors=256[p];[b][p]paletteuse=dither=sierra2_4a')
            run('-framerate', '24', '-start_number', str(first), '-i', pattern, '-filter_complex', filt, '-loop', '0', str(gif))
            mb = gif.stat().st_size / 1e6
            if mb <= a.gif_max:
                break
            width = int(width * 0.85) // 2 * 2
            fps = max(8, fps - 2)
        out.append(f'{gif.name} {gif.stat().st_size / 1e6:.2f} MB ({width}px, {fps} fps)')
    print(name, '·', ' · '.join(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['clip', 'promo', 'ui'])
    ap.add_argument('names', nargs='+')
    ap.add_argument('--gif-seconds', type=float, default=None)
    ap.add_argument('--gif-start', type=float, default=None)
    ap.add_argument('--gif-width', type=int, default=None)
    ap.add_argument('--gif-speed', type=float, default=None)
    ap.add_argument('--gif-max', type=float, default=4.0, help='MB; shrink the GIF until it fits')
    ap.add_argument('--no-gif', action='store_true')
    ap.add_argument('--poster', type=int, default=None)
    a = ap.parse_args()
    for spec in a.names:
        src, _, dst = spec.partition(':')
        dst = dst or src
        if a.kind == 'clip':
            folder = WORK / 'clips' / src
        elif a.kind == 'promo':
            folder = PROMO / src
        else:
            folder = WORK / 'ui_out' / src
            dst = dst if dst.startswith('ui-') else f'ui-{dst}'
        encode(dst, folder, a.kind, a)


if __name__ == '__main__':
    main()
