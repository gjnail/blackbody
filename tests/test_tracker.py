import math

import numpy as np

from blackbody.io.tracker import PointTracker


class FakeFootage:
    """A random texture sliding along a known path, with a little noise per frame."""

    def __init__(self, n=40, w=640, h=360):
        rng = np.random.default_rng(1)
        base = rng.random((h // 4 + 8, w // 4 + 8)).astype(np.float32)
        # smooth, detailed texture
        self.tex = np.kron(base, np.ones((4, 4), np.float32))
        k = np.ones(5, np.float32) / 5
        self.tex = np.apply_along_axis(lambda r: np.convolve(r, k, 'same'), 1, self.tex)
        self.tex = np.apply_along_axis(lambda c: np.convolve(c, k, 'same'), 0, self.tex)
        self.n, self.w, self.h = n, w, h
        self.path = [(1.7 * i + 3 * math.sin(i * 0.3), 0.9 * i) for i in range(n)]
        self.frames = n
        self.width, self.height = w, h

    def read(self, i):
        dx, dy = self.path[i]
        ys = np.arange(self.h)[:, None] + 8 + dy * 0.0
        xs = np.arange(self.w)[None, :] + 8
        # sample the texture shifted by (dx, dy) with bilinear interpolation
        X = xs - dx + 0 * ys
        Y = ys - dy + 0 * xs
        x0 = np.floor(X).astype(int)
        y0 = np.floor(Y).astype(int)
        fx, fy = X - x0, Y - y0
        t = self.tex
        H, W = t.shape
        x0c, x1c = np.clip(x0, 0, W - 1), np.clip(x0 + 1, 0, W - 1)
        y0c, y1c = np.clip(y0, 0, H - 1), np.clip(y0 + 1, 0, H - 1)
        v = ((1 - fy) * ((1 - fx) * t[y0c, x0c] + fx * t[y0c, x1c]) + fy * ((1 - fx) * t[y1c, x0c] + fx * t[y1c, x1c]))
        v = v + np.random.default_rng(i).normal(0, 0.01, v.shape)
        img = (np.clip(v, 0, 1) * 255).astype(np.uint8)
        return np.stack([img, img, img, np.full_like(img, 255)], -1)


def test_tracks_known_motion():
    fake = FakeFootage()
    tr = PointTracker(fake, radius=12, search=16)
    start = 10
    p0 = (0.5, 0.5)
    pts, msg = tr.track(start, p0, 0, fake.n - 1)
    assert len(pts) == fake.n, msg
    sx, sy = fake.path[start]
    errs = []
    for i, (x, y) in pts.items():
        ex = p0[0] * fake.w + (fake.path[i][0] - sx)
        ey = p0[1] * fake.h + (fake.path[i][1] - sy)
        errs.append(math.hypot(x * fake.w - ex, y * fake.h - ey))
    assert max(errs) < 0.6, f'max error {max(errs):.2f}px'
