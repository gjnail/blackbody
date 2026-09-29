"""Physical lookup tables: blackbody colour and brightness, and the tileable detail-noise volume."""
from __future__ import annotations

import numpy as np

BB_MIN_K = 400.0
BB_MAX_K = 6500.0
BB_SIZE = 1024

H = 6.62607015e-34
C = 2.99792458e8
KB = 1.380649e-23

# linear sRGB / Rec.709 primaries, D65 white
XYZ_TO_REC709 = np.array([[3.2404542, -1.5371385, -0.4985314],
                          [-0.9692660, 1.8760108, 0.0415560],
                          [0.0556434, -0.2040259, 1.0572252]])


def _g(x, mu, s1, s2):
    s = np.where(x < mu, s1, s2)
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


def cie_cmf(lam_nm):
    """CIE 1931 2-degree colour matching functions (Wyman, Sloan & Shirley 2013 multi-lobe fit)."""
    x = (1.056 * _g(lam_nm, 599.8, 37.9, 31.0) + 0.362 * _g(lam_nm, 442.0, 16.0, 26.7)
         - 0.065 * _g(lam_nm, 501.1, 20.4, 26.2))
    y = 0.821 * _g(lam_nm, 568.8, 46.9, 40.5) + 0.286 * _g(lam_nm, 530.9, 16.3, 31.1)
    z = 1.217 * _g(lam_nm, 437.0, 11.8, 36.0) + 0.681 * _g(lam_nm, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], axis=-1)


def planck(lam_m, T):
    """Spectral radiance B(lambda, T) in W / (sr m^3)."""
    with np.errstate(over='ignore'):
        return (2.0 * H * C * C / lam_m ** 5) / np.expm1(H * C / (lam_m * KB * T))


def blackbody_xyz(T):
    lam = np.arange(380.0, 781.0, 2.0)
    cmf = cie_cmf(lam)
    T = np.atleast_1d(np.asarray(T, np.float64))
    B = planck(lam[None, :] * 1e-9, T[:, None])  # (n, L)
    return (B[..., None] * cmf[None]).sum(axis=1) * 2e-9  # (n, 3)


def blackbody_lut(size=BB_SIZE, t_min=BB_MIN_K, t_max=BB_MAX_K):
    """(size, 4) float32: rgb = linear Rec.709 chromaticity scaled to luminance 1, a = log10 luminance.
    Out-of-gamut reds are desaturated toward white at constant luminance rather than clipped."""
    T = np.linspace(t_min, t_max, size)
    xyz = blackbody_xyz(T)
    Y = np.maximum(xyz[:, 1], 1e-300)
    rgb = xyz @ XYZ_TO_REC709.T
    rgb = rgb / Y[:, None]
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    lo = rgb.min(axis=1)
    k = np.where(lo < 0, lum / np.maximum(lum - lo, 1e-9), 1.0)
    rgb = lum[:, None] + (rgb - lum[:, None]) * k[:, None]
    out = np.zeros((size, 4), np.float32)
    out[:, :3] = np.maximum(rgb, 0.0)
    out[:, 3] = np.log10(Y)
    return out


def kelvin_to_display_rgb(T, exposure=1.0):
    """Rough sRGB 0..255 swatch for UI: blackbody chromaticity at fixed brightness."""
    lut = blackbody_lut(64, max(BB_MIN_K, T - 1), T + 1)
    rgb = lut[32, :3] * exposure
    rgb = rgb / max(rgb.max(), 1e-6)
    srgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(np.maximum(rgb, 0), 1 / 2.4) - 0.055)
    return tuple(int(round(v * 255)) for v in np.clip(srgb, 0, 1))


def detail_noise_volume(size=64, seed=7):
    """Tileable (size^3, 4) uint8 volume of four decorrelated fBm channels for render-time detail."""
    rng = np.random.default_rng(seed)
    out = np.zeros((size, size, size, 4), np.float32)
    coords = np.arange(size)
    for ch in range(4):
        acc = np.zeros((size, size, size), np.float32)
        amp, total = 1.0, 0.0
        for octave, cells in enumerate((4, 8, 16, 32)):
            g = rng.standard_normal((cells, cells, cells)).astype(np.float32)
            # periodic cubic (Catmull-Rom) upsampling of lattice values -> smooth tileable value noise
            f = coords * cells / size
            i0 = np.floor(f).astype(int)
            t = f - i0
            idx = [(i0 + o) % cells for o in (-1, 0, 1, 2)]
            w = [(-t ** 3 + 2 * t ** 2 - t) / 2, (3 * t ** 3 - 5 * t ** 2 + 2) / 2,
                 (-3 * t ** 3 + 4 * t ** 2 + t) / 2, (t ** 3 - t ** 2) / 2]
            layer = np.zeros((size, cells, cells), np.float32)
            for a in range(4):
                layer += w[a][:, None, None] * g[idx[a]]
            layer2 = np.zeros((size, size, cells), np.float32)
            for a in range(4):
                layer2 += w[a][None, :, None] * layer[:, idx[a], :]
            layer3 = np.zeros((size, size, size), np.float32)
            for a in range(4):
                layer3 += w[a][None, None, :] * layer2[:, :, idx[a]]
            acc += amp * layer3
            total += amp
            amp *= 0.55
        acc /= total
        acc = (acc - acc.mean()) / (acc.std() + 1e-6)
        out[..., ch] = acc
    out = np.clip(out * 0.18 + 0.5, 0, 1)
    return (out * 255 + 0.5).astype(np.uint8)
