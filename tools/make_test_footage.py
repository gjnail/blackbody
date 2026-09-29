"""Generate a synthetic test plate: dusk landscape, slow pan, with a stereo AAC audio track."""
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


_TEX = {}


def ground_texture(w, h):
    """Pebbly ground detail (fixed to the scene, so it slides with the pan)."""
    key = (w, h)
    if key not in _TEX:
        rng = np.random.default_rng(7)
        base = rng.random((h // 3 + 2, (w * 2) // 3 + 2)).astype(np.float32)
        t = np.kron(base, np.ones((3, 3), np.float32))[:h, :w * 2]
        k = np.array([1, 2, 3, 2, 1], np.float32) / 9
        t = np.apply_along_axis(lambda r: np.convolve(r, k, 'same'), 1, t)
        t = np.apply_along_axis(lambda c: np.convolve(c, k, 'same'), 0, t)
        _TEX[key] = (t - t.mean()) * 0.35
    return _TEX[key]


def frame_image(i, w, h, fps):
    t = i / fps
    y = np.linspace(0, 1, h)[:, None]
    x = np.linspace(0, 1, w)[None, :] + t * 0.03
    tex = ground_texture(w, h)
    cols = np.clip((x[0] * w).astype(int), 0, tex.shape[1] - 1)
    pebbles = tex[:, cols]
    sky = np.stack([0.30 - 0.18 * y, 0.26 - 0.16 * y, 0.38 - 0.22 * y], -1)
    horizon = 0.62 + 0.04 * np.sin(x * 7.0) + 0.02 * np.sin(x * 23.0)
    ground = (y > horizon)
    g = np.stack([0.10 + 0.04 * np.sin(x * 40) + 0 * y, 0.09 + 0.03 * np.sin(x * 37) + 0 * y, 0.07 + 0.0 * x + 0 * y], -1) * (0.7 + 0.3 * (1 - y))[..., None]
    g = g * (1.0 + pebbles[..., None])
    img = np.where(ground[..., None], g, sky)
    rng = np.random.default_rng(i)
    img = img + rng.normal(0, 0.008, img.shape)
    return (np.clip(img, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def main(path='out/plate.mp4', w=1280, h=720, fps=30, seconds=3.0):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    c = av.open(path, 'w')
    vs = c.add_stream('libx264', rate=fps)
    vs.width, vs.height, vs.pix_fmt = w, h, 'yuv420p'
    vs.options = {'crf': '18', 'preset': 'veryfast'}
    au = c.add_stream('aac', rate=48000)
    au.layout = 'stereo'
    n = int(seconds * fps)
    sr = 48000
    samples_per = 1024
    total_samples = int(seconds * sr)
    s_done = 0
    for i in range(n):
        fr = av.VideoFrame.from_ndarray(frame_image(i, w, h, fps), format='rgb24')
        for pk in vs.encode(fr):
            c.mux(pk)
        while s_done < (i + 1) / fps * sr and s_done < total_samples:
            tt = (np.arange(samples_per) + s_done) / sr
            tone = (0.2 * np.sin(2 * np.pi * 220 * tt)).astype(np.float32)
            af = av.AudioFrame.from_ndarray(np.stack([tone, tone]), format='fltp', layout='stereo')
            af.sample_rate = sr
            af.pts = s_done
            for pk in au.encode(af):
                c.mux(pk)
            s_done += samples_per
    for pk in vs.encode(None):
        c.mux(pk)
    for pk in au.encode(None):
        c.mux(pk)
    c.close()
    print('wrote', path)


if __name__ == '__main__':
    main(*sys.argv[1:2])
