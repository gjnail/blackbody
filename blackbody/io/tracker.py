"""2D point tracker for footage: normalised cross-correlation with sub-pixel peaks.

Tracks one feature (where the fire should sit) through the shot so the fire follows the footage
without a solved 3D camera. Handheld shots, drifting cameras and moving objects all work as long
as the feature stays visible.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def to_gray(frame, max_width=960):
    """RGBA uint8/float16 frame -> float32 luminance, downscaled so tracking is fast."""
    a = np.asarray(frame)
    step = max(1, int(math.ceil(a.shape[1] / max_width)))
    a = a[::step, ::step, :3].astype(np.float32)
    if frame.dtype == np.uint8:
        a /= 255.0
    g = a @ np.array([0.299, 0.587, 0.114], np.float32)
    return g, step


def _bilinear_patch(img, cx, cy, r):
    """(2r+1)^2 patch centred at a sub-pixel position."""
    x0, y0 = int(math.floor(cx)), int(math.floor(cy))
    fx, fy = cx - x0, cy - y0
    h, w = img.shape
    ys = np.clip(np.arange(y0 - r, y0 + r + 2), 0, h - 1)
    xs = np.clip(np.arange(x0 - r, x0 + r + 2), 0, w - 1)
    blk = img[np.ix_(ys, xs)]
    return ((1 - fy) * ((1 - fx) * blk[:-1, :-1] + fx * blk[:-1, 1:]) + fy * ((1 - fx) * blk[1:, :-1] + fx * blk[1:, 1:]))


def ncc_search(img, tmpl, cx, cy, search):
    """Best match of `tmpl` in `img` within +-search px of (cx, cy). Returns (x, y, score)."""
    r = tmpl.shape[0] // 2
    h, w = img.shape
    cx = min(max(cx, 0.0), w - 1.0)   # keep the window on the image
    cy = min(max(cy, 0.0), h - 1.0)
    x0 = int(round(cx)) - search - r
    y0 = int(round(cy)) - search - r
    x1 = int(round(cx)) + search + r + 1
    y1 = int(round(cy)) + search + r + 1
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - w), max(0, y1 - h)
    win = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if pad_l or pad_t or pad_r or pad_b:
        win = np.pad(win, ((pad_t, pad_b), (pad_l, pad_r)), mode='edge')
    views = sliding_window_view(win, tmpl.shape)          # (2S+1, 2S+1, 2r+1, 2r+1)
    t = tmpl - tmpl.mean()
    tn = math.sqrt(float((t * t).sum())) + 1e-6
    vm = views.mean(axis=(2, 3), keepdims=True)
    vc = views - vm
    num = (vc * t).sum(axis=(2, 3))
    den = np.sqrt((vc * vc).sum(axis=(2, 3))) * tn + 1e-6
    score = num / den
    iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
    best = float(score[iy, ix])

    def sub(a, b, c):
        d = a - 2 * b + c
        return 0.0 if abs(d) < 1e-9 else 0.5 * (a - c) / d

    dx = sub(score[iy, ix - 1], score[iy, ix], score[iy, ix + 1]) if 0 < ix < score.shape[1] - 1 else 0.0
    dy = sub(score[iy - 1, ix], score[iy, ix], score[iy + 1, ix]) if 0 < iy < score.shape[0] - 1 else 0.0
    return (x0 + r + ix + max(-0.5, min(0.5, dx)), y0 + r + iy + max(-0.5, min(0.5, dy)), best)


class PointTracker:
    def __init__(self, footage, radius=14, search=28, min_score=0.55, max_width=960):
        self.footage = footage
        self.radius = radius
        self.search = search
        self.min_score = min_score
        self.max_width = max_width

    def track(self, start_index, point, first_index, last_index, progress=None, cancelled=None):
        """Track `point` (normalised x, y) from footage frame `start_index` forwards to last_index and
        backwards to first_index. Returns ({index: (x, y) normalised}, message)."""
        g0, step = to_gray(self.footage.read(start_index), self.max_width)
        h, w = g0.shape
        px, py = point[0] * w, point[1] * h
        tmpl0 = _bilinear_patch(g0, px, py, self.radius)
        out = {start_index: (point[0], point[1])}
        if float(tmpl0.std()) < 0.012:
            return out, ('There is too little detail at this spot to track (plain sky, flat wall, blur). '
                         'Move the fire base onto a spot with texture or a corner and track again.')
        total = max(1, last_index - first_index)
        done = 0
        msg = ''
        for direction, end in ((1, last_index), (-1, first_index)):
            cx, cy = px, py
            vx, vy = 0.0, 0.0
            tmpl = tmpl0.copy()
            i = start_index
            while (i < end) if direction > 0 else (i > end):
                if cancelled is not None and cancelled():
                    return out, 'Tracking stopped.'
                i += direction
                g, _ = to_gray(self.footage.read(i), self.max_width)
                nx, ny, score = ncc_search(g, tmpl, cx + vx, cy + vy, self.search)
                if score < self.min_score:
                    # retry around the last position without the velocity prediction
                    nx2, ny2, s2 = ncc_search(g, tmpl, cx, cy, self.search * 2)
                    if s2 > score:
                        nx, ny, score = nx2, ny2, s2
                if score < self.min_score:
                    msg = f'Lost the point at footage frame {i} (match {score:.2f}). The track stops there.'
                    break
                if not (0.0 <= nx < w and 0.0 <= ny < h):
                    msg = f'The point left the frame at footage frame {i}. The track stops there.'
                    break
                vx, vy = 0.7 * (nx - cx) + 0.3 * vx, 0.7 * (ny - cy) + 0.3 * vy
                cx, cy = nx, ny
                out[i] = (cx / w, cy / h)
                if score > 0.85:  # follow slow appearance changes, anchored to the original look
                    new = _bilinear_patch(g, cx, cy, self.radius)
                    tmpl = 0.85 * tmpl + 0.1 * new + 0.05 * tmpl0
                done += 1
                if progress is not None:
                    progress(done / total, i)
        return out, msg or f'Tracked {len(out)} frames.'
