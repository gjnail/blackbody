"""Measurements of the footage that let the fire sit in it: its noise and the colour of its haze.

Both are measured from the frame being composited, so a frame rendered on its own comes out exactly as
it does in a sequence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

NOISE_BINS = 10          # brightness bins, one stop apart
NOISE_LO = -9.0          # log2 of the linear value at the centre of the first bin
MIN_SAMPLES = 200        # pixel pairs a bin needs before its measurement counts
_CHI2_MEDIAN = 0.45493642  # median of chi-squared with one degree of freedom
_ACESCG_TO_709 = np.array([[1.70505, -0.62179, -0.08326], [-0.13026, 1.14080, -0.01055],
                           [-0.02400, -0.12897, 1.15297]], np.float32)


@dataclass
class NoiseModel:
    """The footage's noise, as seen by a compositor: its standard deviation in scene-linear light at each
    brightness (per channel), how coarse it is, and how much the channels move together."""
    sigma: np.ndarray    # (NOISE_BINS, 3) standard deviation at 2^(NOISE_LO + i), per channel
    size: float          # grain size: standard deviation (px) of the Gaussian the noise looks blurred by
    chol: np.ndarray     # (3, 3) lower Cholesky factor of the channel correlation

    def sigma_at(self, x):
        """Noise standard deviation at linear values x (..., 3), as the composite shader interpolates it."""
        x = np.asarray(x, np.float32)
        f = np.clip(np.log2(np.maximum(x, 1e-12)) - NOISE_LO, 0.0, NOISE_BINS - 1.0)
        i = np.minimum(np.floor(f).astype(int), NOISE_BINS - 2)
        t = f - i
        ch = np.arange(3)
        return self.sigma[i, ch] * (1.0 - t) + self.sigma[i + 1, ch] * t


def plate_linear(img, kind='srgb', gain=1.0):
    """A footage frame (uint8 display-encoded, or float) -> scene-linear Rec.709, float32 (h, w, 3).
    Matches the composite shader's input transform."""
    x = np.asarray(img)[..., :3]
    if x.dtype == np.uint8 and kind != 'acescg':
        return _LUT8[kind if kind in _LUT8 else 'linear'][x] * np.float32(gain)
    x = x.astype(np.float32) / 255.0 if x.dtype == np.uint8 else x.astype(np.float32)
    if kind == 'srgb':
        lin = np.where(x <= 0.04045, x / 12.92, ((np.maximum(x, 0.0) + 0.055) / 1.055) ** 2.4)
    elif kind == 'rec709':
        lin = np.maximum(x, 0.0) ** 2.4
    elif kind == 'acescg':
        lin = x @ _ACESCG_TO_709.T
    else:
        lin = x
    return (lin * np.float32(gain)).astype(np.float32)


_CODES = np.arange(256, dtype=np.float32) / 255.0
_LUT8 = {'srgb': np.where(_CODES <= 0.04045, _CODES / 12.92, ((_CODES + 0.055) / 1.055) ** 2.4).astype(np.float32),
         'rec709': (_CODES ** 2.4).astype(np.float32), 'linear': _CODES}


def _noise_var(d):
    """Variance of the zero-mean noise in pixel differences d, not counting edges and texture: from the median
    of d^2 or, where the footage is quantized coarser than its noise (most differences exactly zero, so the
    median is zero too), from the mean of d^2 with outliers beyond 3 sigma left out."""
    d2 = d * d
    if np.count_nonzero(d) > 0.8 * d.size:
        return float(np.median(d2)) / _CHI2_MEDIAN
    v = float(d2.mean())
    for _ in range(4):
        if v <= 0.0:
            return 0.0
        v = float(d2[d2 < 9.0 * v].mean()) / 0.9733  # E[z^2 | |z| < 3] for a normal z
    return v


# Noise blurred by a Gaussian correlates as u^(k^2) at k pixels. The second differences
# a = x[2] - (x[1] + x[3]) / 2 and b = x[2] - (x[0] + x[4]) / 2, which ignore smooth gradients in the picture,
# then have variances sigma^2 * _va(u) and sigma^2 * _vb(u).
def _va(u):
    return 1.5 + 0.5 * u ** 4 - 2.0 * u


def _vb(u):
    return 1.5 + 0.5 * u ** 16 - 2.0 * u ** 4


def _grain_u(ratio):
    """Correlation of neighbouring pixels, u, from Var(b) / Var(a)."""
    lo, hi = 0.0, 0.97
    if ratio >= _vb(hi) / _va(hi):
        return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _vb(mid) / _va(mid) < ratio:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _stride(h, w, samples):
    """Every how many rows (and columns) to look along for about `samples` pixels."""
    return max(1, int(round(2.0 * h * w / samples)))


def measure_noise(lin, samples=100_000):
    """Measure the noise of a scene-linear frame (h, w, 3). None when the frame is too small to tell."""
    lin = np.asarray(lin, np.float32)
    h, w = lin.shape[:2]
    if h < 16 or w < 16:
        return None
    s = _stride(h, w, samples)
    return _noise_along(lin[::s], lin[:, ::s].transpose(1, 0, 2))


def measure_plate(img, kind='srgb', gain=1.0, noise=True, haze=True, samples=100_000):
    """(noise model, haze colour) of a footage frame as given (see plate_linear), each None when not asked
    for or not measurable. Linearises only the pixels the measurements look at."""
    h, w = img.shape[:2]
    model = colour = None
    if noise and h >= 16 and w >= 16:
        s = _stride(h, w, samples)
        model = _noise_along(plate_linear(img[::s], kind, gain), plate_linear(img[:, ::s], kind, gain).transpose(1, 0, 2))
    if haze:
        s = max(1, max(h, w) // 480)
        colour = haze_colour(plate_linear(img[::s, ::s], kind, gain))
    return model, colour


def _noise_along(*lines):
    """The noise model from lines of pixels (each (n, length, 3), scene-linear).

    Uses second differences over pixels one and two apart along each line: their variances give both the
    noise's strength and its coarseness, and smooth gradients (skies, vignetting) cancel out. Robust
    statistics keep edges and texture from counting as noise."""
    d1s, d2s, ms = [], [], []
    for a in lines:
        x0, x1, x2, x3, x4 = a[:, :-4], a[:, 1:-3], a[:, 2:-2], a[:, 3:-1], a[:, 4:]
        d1s.append((x2 - 0.5 * (x1 + x3)).reshape(-1, 3))
        d2s.append((x2 - 0.5 * (x0 + x4)).reshape(-1, 3))
        ms.append(((x1 + x2 + x3) * (1.0 / 3.0)).reshape(-1, 3))
    d1, d2, m = np.concatenate(d1s), np.concatenate(d2s), np.concatenate(ms)
    b = np.clip(np.round(np.log2(np.maximum(m, 1e-12)) - NOISE_LO), 0, NOISE_BINS - 1).astype(np.int32)

    v1 = np.full((NOISE_BINS, 3), np.nan)
    num = den = 0.0
    for c in range(3):
        order = np.argsort(b[:, c], kind='stable')
        ends = np.cumsum(np.bincount(b[:, c], minlength=NOISE_BINS))
        e1, e2 = d1[order, c], d2[order, c]
        for i in range(NOISE_BINS):
            lo, hi = (ends[i - 1] if i else 0), ends[i]
            n = int(hi - lo)
            if n < MIN_SAMPLES:
                continue
            a1, a2 = _noise_var(e1[lo:hi]), _noise_var(e2[lo:hi])
            v1[i, c] = a1
            num += n * a2
            den += n * a1
    if den <= 0.0 or not np.isfinite(v1).any():
        return None
    u = _grain_u(num / den)
    size = math.sqrt(-1.0 / (4.0 * math.log(u))) if u > 1e-3 else 0.0
    var = v1 / _va(u)

    # bins the frame has no pixels in take their neighbours' values
    idx = np.arange(NOISE_BINS)
    sigma = np.zeros((NOISE_BINS, 3), np.float32)
    for c in range(3):
        ok = np.isfinite(var[:, c])
        if ok.any():
            sigma[:, c] = np.sqrt(np.maximum(np.interp(idx, idx[ok], var[ok, c]), 0.0))

    # how the channels move together, from the one-pixel second differences scaled to unit noise, leaving
    # out edges and texture (beyond 3 sigma)
    sd = math.sqrt(_va(u)) * sigma[b, np.arange(3)]
    z = np.where(sd > 0, d1 / np.maximum(sd, 1e-12), 0.0)
    keep = np.all(np.abs(z) < 3.0, axis=1) & np.any(z != 0.0, axis=1)
    corr = np.eye(3)
    if keep.sum() >= MIN_SAMPLES:
        with np.errstate(invalid='ignore', divide='ignore'):
            corr = np.nan_to_num(np.corrcoef(z[keep].T), nan=0.0)
        np.fill_diagonal(corr, 1.0)
    ev, evec = np.linalg.eigh(corr)
    corr = (evec * np.maximum(ev, 1e-3)) @ evec.T
    dg = np.sqrt(np.diag(corr))
    corr = corr / np.outer(dg, dg)
    chol = np.linalg.cholesky(corr).astype(np.float32)
    return NoiseModel(sigma=sigma, size=float(size), chol=chol)


def _min_filter(a, r):
    """Minimum over a (2r+1)^2 neighbourhood, edges clamped."""
    p = np.pad(a, r, mode='edge')
    h, w = a.shape
    rows = p[:, :w].copy()
    for k in range(1, 2 * r + 1):
        np.minimum(rows, p[:, k:k + w], out=rows)
    out = rows[:h].copy()
    for k in range(1, 2 * r + 1):
        np.minimum(out, rows[k:k + h], out=out)
    return out


def haze_colour(lin):
    """Colour of the haze in the footage (scene-linear rgb): the average of its haziest bright spots, where
    even the darkest channel of the darkest nearby pixel is bright (the dark channel prior, He et al. 2009).
    That is usually the sky near the horizon."""
    lin = np.asarray(lin, np.float32)
    h, w = lin.shape[:2]
    s = max(1, int(max(h, w) // 480))
    small = np.maximum(lin[::s, ::s], 0.0)
    dark = _min_filter(small.min(axis=-1), 3)
    k = max(1, int(dark.size * 0.005))
    idx = np.argpartition(dark.ravel(), -k)[-k:]
    return tuple(float(v) for v in small.reshape(-1, 3)[idx].mean(axis=0))
