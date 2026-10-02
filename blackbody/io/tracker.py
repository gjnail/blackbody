"""2D point tracker for footage: normalised cross-correlation with sub-pixel peaks.

Tracks one feature (where the fire should sit) through the shot so the fire follows the footage
without a solved 3D camera. Handheld shots, drifting cameras and moving objects all work as long
as the feature stays visible.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def to_gray(frame, max_width=960, smooth=True):
    """RGBA uint8/float16 frame -> float32 luminance, downscaled so tracking is fast: each pixel the average of the
    step x step block it covers (so a pixel's middle is the block's, and the sensor's noise averages out rather than
    being sampled), then lightly blurred (a 3x3 binomial): noisy footage (dusk, high ISO, compression) otherwise
    matches from frame to frame too poorly to follow."""
    a = np.asarray(frame)
    step = max(1, int(math.ceil(a.shape[1] / max_width)))
    h, w = a.shape[0] // step * step, a.shape[1] // step * step
    a = a[:h, :w, :3].astype(np.float32)
    if frame.dtype == np.uint8:
        a /= 255.0
    g = a @ np.array([0.299, 0.587, 0.114], np.float32)
    if step > 1:
        g = g.reshape(h // step, step, w // step, step).mean(axis=(1, 3))
    if smooth and g.shape[0] > 2 and g.shape[1] > 2:
        p = np.pad(g, 1, mode='edge')
        g = (p[:-2] + 2 * p[1:-1] + p[2:]) / 4
        g = (g[:, :-2] + 2 * g[:, 1:-1] + g[:, 2:]) / 4
    return np.ascontiguousarray(g, np.float32), step


def _bilinear_patch(img, cx, cy, r):
    """(2r+1)^2 patch centred at a sub-pixel position."""
    x0, y0 = int(math.floor(cx)), int(math.floor(cy))
    fx, fy = cx - x0, cy - y0
    h, w = img.shape
    ys = np.clip(np.arange(y0 - r, y0 + r + 2), 0, h - 1)
    xs = np.clip(np.arange(x0 - r, x0 + r + 2), 0, w - 1)
    blk = img[np.ix_(ys, xs)]
    return ((1 - fy) * ((1 - fx) * blk[:-1, :-1] + fx * blk[:-1, 1:]) + fy * ((1 - fx) * blk[1:, :-1] + fx * blk[1:, 1:]))


def ncc_map(img, tmpl, cx, cy, search):
    """Normalised cross-correlation of `tmpl` over img within +-search px of (cx, cy): (scores, x0, y0), where
    scores[j, i] is the match with the template centred at (x0 + r + i, y0 + r + j)."""
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
    return num / den, x0, y0


def ambiguous(img, x, y, radius, search, limit=0.95):
    """Whether the spot at (x, y) looks like others near it (tiles, bricks, a fence): a tracker could slip onto them."""
    tmpl = _bilinear_patch(img, x, y, radius)
    score, x0, y0 = ncc_map(img, tmpl, x, y, search)
    S = score.shape[0] // 2
    yy, xx = np.mgrid[0:score.shape[0], 0:score.shape[1]]
    away = (np.abs(yy - S) > 3) | (np.abs(xx - S) > 3)
    return bool(away.any() and float(score[away].max()) > limit)


def ncc_search(img, tmpl, cx, cy, search):
    """Best match of `tmpl` in `img` within +-search px of (cx, cy). Returns (x, y, score)."""
    r = tmpl.shape[0] // 2
    score, x0, y0 = ncc_map(img, tmpl, cx, cy, search)
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


def pick_features(gray, n=24, radius=14, margin=48, cols=6, rows=4):
    """Up to n well-textured spots spread over the frame (the strongest corner in each cell of a grid), as (x, y) px:
    what a camera track follows."""
    g = np.asarray(gray, np.float32)
    gy, gx = np.gradient(g)

    def box(a, r):
        c = np.cumsum(np.cumsum(np.pad(a, ((r + 1, r), (r + 1, r))), 0), 1)
        return c[2 * r + 1:, 2 * r + 1:] - c[:-2 * r - 1, 2 * r + 1:] - c[2 * r + 1:, :-2 * r - 1] + c[:-2 * r - 1, :-2 * r - 1]
    ixx, iyy, ixy = box(gx * gx, radius // 2), box(gy * gy, radius // 2), box(gx * gy, radius // 2)
    score = (ixx + iyy) / 2 - np.sqrt(((ixx - iyy) / 2) ** 2 + ixy ** 2)   # the weaker direction: corners, not edges
    h, w = g.shape
    out = []
    floor = float(np.percentile(score, 60)) if score.size else 0.0
    for r_ in range(rows):
        for c_ in range(cols):
            y0, y1 = margin + (h - 2 * margin) * r_ // rows, margin + (h - 2 * margin) * (r_ + 1) // rows
            x0, x1 = margin + (w - 2 * margin) * c_ // cols, margin + (w - 2 * margin) * (c_ + 1) // cols
            if y1 <= y0 or x1 <= x0:
                continue
            cell = score[y0:y1, x0:x1]
            iy, ix = np.unravel_index(int(np.argmax(cell)), cell.shape)
            if cell[iy, ix] > max(floor, 1e-5) and float(_bilinear_patch(g, x0 + ix, y0 + iy, radius).std()) > 0.015:
                out.append((float(x0 + ix), float(y0 + iy), float(cell[iy, ix])))
    out.sort(key=lambda t: -t[2])
    return [(x, y) for x, y, _ in out[:n]]


def _sample(img, xs, ys):
    """Bilinear samples of img at (xs, ys) (any shape), edges held."""
    h, w = img.shape
    xs = np.clip(xs, 0.0, w - 1.001)
    ys = np.clip(ys, 0.0, h - 1.001)
    x0, y0 = np.floor(xs).astype(np.int64), np.floor(ys).astype(np.int64)
    fx, fy = xs - x0, ys - y0
    a = img[y0, x0]
    b = img[y0, x0 + 1]
    c = img[y0 + 1, x0]
    d = img[y0 + 1, x0 + 1]
    return (1 - fy) * ((1 - fx) * a + fx * b) + fy * ((1 - fx) * c + fx * d)


class AffineKLT:
    """Sub-pixel refinement of many spots at once: each spot's reference look is warped (moved, scaled, sheared and
    turned) to match the frame (inverse-compositional Lucas-Kanade, with brightness and contrast normalised), so a
    patch of ground that grows and leans as the camera moves on it is followed exactly, without drift."""

    def __init__(self, radius):
        self.r = radius
        g = np.arange(-radius, radius + 1, dtype=np.float64)
        self.v, self.u = np.meshgrid(g, g, indexing='ij')   # template rows (y) and columns (x)

    def prepare(self, refs):
        """refs: (n, P, P) reference patches. Precomputes the normalised templates and the steepest-descent images."""
        T = np.asarray(refs, np.float64)
        T = (T - T.mean(axis=(1, 2), keepdims=True)) / (T.std(axis=(1, 2), keepdims=True) + 1e-6)
        gy, gx = np.gradient(T, axis=(1, 2))
        u, v = self.u[None], self.v[None]
        SD = np.stack([gx * u, gx * v, gy * u, gy * v, gx, gy], axis=-1).reshape(len(T), -1, 6)
        Hm = np.einsum('npi,npj->nij', SD, SD) + 1e-6 * np.eye(6)
        return {'T': T.reshape(len(T), -1), 'SD': SD, 'Hinv': np.linalg.inv(Hm)}

    def refine(self, img, prep, A, iters=12):
        """Warps A (n, 2, 3): template (u, v, 1) to image (x, y). Returns refined warps and the normalised residual."""
        A = np.array(A, np.float64)
        n = len(A)
        uu, vv = self.u.ravel(), self.v.ravel()
        res = np.full(n, np.inf)
        for _ in range(iters):
            xs = A[:, 0, 0, None] * uu + A[:, 0, 1, None] * vv + A[:, 0, 2, None]
            ys = A[:, 1, 0, None] * uu + A[:, 1, 1, None] * vv + A[:, 1, 2, None]
            I = _sample(img, xs, ys)
            I = (I - I.mean(axis=1, keepdims=True)) / (I.std(axis=1, keepdims=True) + 1e-6)
            e = I - prep['T']
            res = np.sqrt(np.mean(e * e, axis=1))
            dp = np.einsum('nij,nj->ni', prep['Hinv'], np.einsum('npi,np->ni', prep['SD'], e))
            M = np.zeros((n, 3, 3))
            M[:, 0, 0], M[:, 0, 1], M[:, 0, 2] = 1 + dp[:, 0], dp[:, 1], dp[:, 4]
            M[:, 1, 0], M[:, 1, 1], M[:, 1, 2] = dp[:, 2], 1 + dp[:, 3], dp[:, 5]
            M[:, 2, 2] = 1.0
            A3 = np.concatenate([A, np.tile([[[0.0, 0.0, 1.0]]], (n, 1, 1))], axis=1)
            try:
                A = (A3 @ np.linalg.inv(M))[:, :2]
            except np.linalg.LinAlgError:
                break
            if float(np.abs(dp[:, 4:]).max()) < 1e-3:
                break
        return A, res


class MultiTracker:
    """Many spots tracked at once, each frame read once: the input of a camera solve. Spots that are lost (out of
    the frame, hidden, changed) are replaced by new ones where the frame has none, so a camera that travels keeps
    enough to follow."""

    def __init__(self, footage, radius=12, search=24, min_score=0.7, max_width=960, points=60, reseed_every=3):
        self.footage = footage
        self.radius = radius
        self.search = search
        self.min_score = min_score
        self.max_width = max_width
        self.points = points
        self.reseed_every = reseed_every
        self.klt = AffineKLT(radius)

    def _seed(self, g, alive_pos, next_id, want):
        """New spots in the cells of a 8 x 5 grid that have none, up to `want`."""
        h, w = g.shape
        cols, rows = 10, 6
        taken = set()
        for x, y in alive_pos:
            taken.add((min(cols - 1, int(x / w * cols)), min(rows - 1, int(y / h * rows))))
        out = {}
        if want <= 0:
            return out, next_id
        margin = self.radius + self.search // 2 + 4
        for x, y in pick_features(g, cols * rows, self.radius, margin=margin, cols=cols, rows=rows):
            cell = (min(cols - 1, int(x / w * cols)), min(rows - 1, int(y / h * rows)))
            if cell in taken or ambiguous(g, x, y, self.radius, self.search):
                continue
            taken.add(cell)
            out[next_id] = (x, y)
            next_id += 1
            if len(out) >= want:
                break
        return out, next_id

    def _spot(self, g, xy):
        x, y = float(xy[0]), float(xy[1])
        ref = _bilinear_patch(g, x, y, self.radius)
        prep = self.klt.prepare(ref[None])
        return {'pos': np.array([x, y], np.float64), 'vel': np.zeros(2), 'tm': ref.copy(),
                'A': np.array([[1.0, 0.0, x], [0.0, 1.0, y]]), 'prep': {k: v[0] for k, v in prep.items()}}

    def track(self, start_index, first_index, last_index, progress=None, cancelled=None):
        """Track spots from frame start_index forwards and backwards. Returns ({index: {spot id: (x, y) normalised}},
        message). Spot ids are unique over the shot."""
        g0, _ = to_gray(self.footage.read(start_index), self.max_width)
        h, w = g0.shape
        seeds, next_id = self._seed(g0, [], 0, self.points)
        if len(seeds) < 4:
            return {}, 'There is too little detail in this footage to track the camera (sky, fog, a plain wall).'
        # positions are pixel indices; +0.5 makes them the pixel's middle, as the camera's projection has it
        out = {start_index: {k: ((x + 0.5) / w, (y + 0.5) / h) for k, (x, y) in seeds.items()}}
        total = max(1, last_index - first_index)
        done = 0
        for direction, end in ((1, last_index), (-1, first_index)):
            spots = {k: self._spot(g0, xy) for k, xy in seeds.items()}
            i = start_index
            n = 0
            prev_g = g0
            while (i < end) if direction > 0 else (i > end):
                if cancelled is not None and cancelled():
                    return out, 'Tracking stopped.'
                i += direction
                n += 1
                g, _ = to_gray(self.footage.read(i), self.max_width)
                for k in list(spots):
                    sp = spots[k]
                    nx, ny, sc = ncc_search(g, sp['tm'], sp['pos'][0] + sp['vel'][0], sp['pos'][1] + sp['vel'][1], self.search)
                    if sc < self.min_score or not (self.radius <= nx < w - self.radius and self.radius <= ny < h - self.radius):
                        del spots[k]
                        continue
                    # back again: the patch found here must lead back to where the spot was, or it slipped
                    here = _bilinear_patch(g, nx, ny, self.radius)
                    bx, by, bs = ncc_search(prev_g, here, sp['pos'][0], sp['pos'][1], 4)
                    if bs < self.min_score or math.hypot(bx - sp['pos'][0], by - sp['pos'][1]) > 1.5:
                        del spots[k]
                        continue
                    sp['A'][:, 2] = (nx, ny)   # the coarse match; the warp's shape carries over from the last frame
                    sp['tm'] = here
                # exact places: each spot's reference look warped onto the frame (no drift, perspective followed)
                ks = list(spots)
                if ks:
                    prep = {key: np.stack([spots[k]['prep'][key] for k in ks]) for key in ('T', 'SD', 'Hinv')}
                    A, res = self.klt.refine(g, prep, np.stack([spots[k]['A'] for k in ks]))
                    for k, a_, r_ in zip(ks, A, res):
                        sp = spots[k]
                        scale = math.sqrt(abs(float(np.linalg.det(a_[:, :2]))))
                        coarse = sp['A'][:, 2].copy()
                        near = float(np.hypot(*(a_[:, 2] - coarse))) < 2.0
                        if near and r_ < 0.75 and 0.6 < scale < 1.7:
                            sp['A'] = a_
                        else:   # the reference no longer fits (it changed too much): it is taken again from here
                            fresh = self._spot(g, a_[:, 2] if near else coarse)
                            sp['A'], sp['prep'] = fresh['A'], fresh['prep']
                        new = sp['A'][:, 2].copy()
                        sp['vel'] = 0.7 * (new - sp['pos']) + 0.3 * sp['vel']
                        sp['pos'] = new
                prev_g = g
                if n % self.reseed_every == 0 and len(spots) < self.points:
                    more, next_id = self._seed(g, [sp['pos'] for sp in spots.values()], next_id, self.points - len(spots))
                    for k, xy in more.items():
                        spots[k] = self._spot(g, xy)
                if len(spots) < 3:
                    break
                out[i] = {k: (float((sp['pos'][0] + 0.5) / w), float((sp['pos'][1] + 0.5) / h)) for k, sp in spots.items()}
                done += 1
                if progress is not None:
                    progress(done / total, i)
        lost = total + 1 - len(out)
        return out, (f'Tracked {next_id} spots over {len(out)} frames.' + (f' {lost} frames at the ends had too few spots left.' if lost > 0 else ''))
