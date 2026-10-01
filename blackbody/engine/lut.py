"""Physical lookup tables: blackbody colour and brightness, and the tileable detail-noise volume."""
from __future__ import annotations

import math

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


def luminance(T):
    """Luminance (cd/m^2) of a blackbody at T Kelvin."""
    return 683.0 * float(blackbody_xyz(T)[0, 1])


def flame_ev(T):
    """What a light meter reads (EV at ISO 100) off a blackbody at T Kelvin: EV = log2(L S / K), K = 12.5."""
    return math.log2(max(luminance(T), 1e-12) * 100.0 / 12.5)


def camera_sensitivity(lam_nm):
    """Spectral sensitivities (r, g, b) of a typical Bayer CMOS sensor behind its infrared-cut filter: smooth
    fits to the general shape of published camera curves, not any one camera."""
    def lobe(mu, s1, s2):
        return np.exp(-0.5 * ((lam_nm - mu) / np.where(lam_nm < mu, s1, s2)) ** 2)
    ir_cut = 1.0 / (1.0 + np.exp((lam_nm - 655.0) / 9.0))
    return np.stack([lobe(598, 38, 30) + 0.06 * lobe(450, 25, 25), lobe(532, 42, 48), lobe(462, 30, 40)], -1) * ir_cut[:, None]


def camera_to_xyz():
    """3x3 matrix taking the sensor's white-balanced (daylight) rgb to CIE XYZ, fitted by least squares
    over smooth reflectances under daylight and over blackbodies, as camera makers fit theirs."""
    lam = np.arange(380.0, 781.0, 2.0)
    cmf, cam = cie_cmf(lam), camera_sensitivity(lam)
    rng = np.random.default_rng(0)
    day = planck(lam * 1e-9, 6500.0)
    spectra = []
    for _ in range(400):
        r = sum(rng.uniform(0, 1) * np.exp(-0.5 * ((lam - rng.uniform(400, 700)) / rng.uniform(20, 120)) ** 2) for _ in range(3))
        spectra.append(np.clip(r, 0, 1) * day)
    spectra += [planck(lam * 1e-9, t) for t in np.linspace(1500.0, 10000.0, 30)]
    S = np.array(spectra)
    rgb, xyz = S @ cam, S @ cmf
    norm = rgb.sum(axis=1, keepdims=True)
    w_rgb, w_xyz = day @ cam, day @ cmf
    a = np.vstack([rgb / norm, 50.0 * w_rgb / w_rgb.sum()])     # daylight white must stay white
    b = np.vstack([xyz / norm, 50.0 * w_xyz / w_rgb.sum()])
    return np.linalg.lstsq(a, b, rcond=None)[0].T


def blackbody_lut(size=BB_SIZE, t_min=BB_MIN_K, t_max=BB_MAX_K, response='eye'):
    """(size, 4) float32: rgb = linear Rec.709 chromaticity scaled to luminance 1, a = log10 luminance.
    response 'camera' takes the colour as a typical camera sensor records it (camera_sensitivity), the
    brightness still as the eye sees it. Out-of-gamut reds are desaturated toward white at constant
    luminance rather than clipped."""
    T = np.linspace(t_min, t_max, size)
    xyz = blackbody_xyz(T)
    Y = np.maximum(xyz[:, 1], 1e-300)
    if response == 'camera':
        lam = np.arange(380.0, 781.0, 2.0)
        B = planck(lam[None, :] * 1e-9, T[:, None])
        c_xyz = (B @ camera_sensitivity(lam)) @ camera_to_xyz().T
        rgb = c_xyz @ XYZ_TO_REC709.T
        rgb = rgb / np.maximum(rgb @ np.array([0.2126, 0.7152, 0.0722]), 1e-300)[:, None]
    else:
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
