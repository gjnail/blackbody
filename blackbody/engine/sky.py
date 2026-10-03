"""A physical sky (Lighting › Sky: Physical): the sky's light in every direction and the sun's at the ground, worked out
from the air itself, as Hillaire (2020, "A Scalable and Production Ready Sky and Atmosphere Rendering Technique") does it:
the air's molecules scatter (Rayleigh: blue far more than red), its haze scatters forward (Mie, as much as the haze
there is) and absorbs a little, its ozone absorbs, all thinning with height, over a round earth. Light is followed from
the sun through the air once exactly, and the light scattered more than once by Hillaire's isotropic sum (each further
bounce a share f_ms of the last, summed as 1 / (1 - f_ms)), which is what keeps a low sun's sky bright and its colours
right. Three wavelengths (680, 550, 440 nm: red, green, blue).

What it gives: an HDRI of the sky (latitude-longitude, radiance; the ground below the horizon lit by the sun and the
sky), the sun's irradiance at the ground (its colour reddening as it sets), the sky's light on an upward surface (the
classic engine's ambient), all in units where the sun outside the air has irradiance 1: times the Key light's intensity.
The sun's disc is not in the HDRI (the key light is the sun: Lume samples it as a cone, the classic engine as a light).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

R_GROUND = 6360e3        # m
R_TOP = 6460e3
RAYLEIGH = np.array([5.802, 13.558, 33.1]) * 1e-6    # scattering at sea level (1/m), 680 / 550 / 440 nm
H_RAYLEIGH = 8000.0
MIE_SCATTER = 3.996e-6                                # haze at sea level, per unit of its density (Hillaire's)
MIE_ABSORB = 0.444e-6                                 # (its extinction 4.44e-6: haze mostly scatters)
H_MIE = 1200.0
MIE_G = 0.8
OZONE = np.array([0.650, 1.881, 0.085]) * 1e-6        # absorption at its peak (1/m)
OZONE_PEAK, OZONE_WIDTH = 25e3, 15e3                  # a tent: peak height, half width
LUMA = np.array([0.2126, 0.7152, 0.0722])


@dataclass(frozen=True)
class Air:
    haze: float = 0.1        # the haze's optical depth at 550 nm, straight up (aerosol optical depth): 0.02 the
                             # clearest mountain air, 0.1 a clear day, 0.3 hazy, 0.6 a thick summer haze
    ozone: float = 1.0       # the ozone over the usual
    ground: float = 0.3      # the ground's albedo (the light it sends back up into the sky)
    altitude: float = 0.0    # m: the camera's height above the sea


def _densities(h, air: Air):
    h = np.maximum(h, 0.0)
    r = np.exp(-h / H_RAYLEIGH)
    m = np.exp(-h / H_MIE) * (air.haze / ((MIE_SCATTER + MIE_ABSORB) * H_MIE))
    o = np.maximum(1.0 - np.abs(h - OZONE_PEAK) / OZONE_WIDTH, 0.0) * air.ozone
    return r, m, o


def _extinction(h, air: Air):
    """Extinction (1/m, rgb) at heights h (..., ) -> (..., 3)."""
    r, m, o = _densities(h, air)
    return (r[..., None] * RAYLEIGH + m[..., None] * (MIE_SCATTER + MIE_ABSORB) + o[..., None] * OZONE)


def _ray_top(r, mu):
    """Distance from radius r along a ray at cosine mu (to the vertical) to the top of the air."""
    return -r * mu + np.sqrt(np.maximum(r * r * (mu * mu - 1.0) + R_TOP * R_TOP, 0.0))


def _hits_ground(r, mu):
    return (mu < 0.0) & (r * r * (mu * mu - 1.0) + R_GROUND * R_GROUND >= 0.0)


def _transmittance_direct(r, mu, air: Air, steps=40):
    """Transmittance from radius r along cosine mu to the top of the air (0 where the ray meets the ground)."""
    r = np.asarray(r, float)
    mu = np.asarray(mu, float)
    L = _ray_top(r, mu)
    t = (np.arange(steps) + 0.5) / steps
    s = L[..., None] * t                                       # (..., steps)
    rr = np.sqrt(r[..., None] ** 2 + s * s + 2.0 * r[..., None] * mu[..., None] * s)
    ext = _extinction(rr - R_GROUND, air)                      # (..., steps, 3)
    od = ext.sum(-2) * (L / steps)[..., None]
    T = np.exp(-od)
    return np.where(_hits_ground(r, mu)[..., None], 0.0, T)


class _TransmittanceLUT:
    """Transmittance to the top of the air by height and the cosine of the angle to the vertical (a table, looked up)."""

    NH, NMU = 64, 128

    def __init__(self, air: Air):
        h = np.linspace(0.0, R_TOP - R_GROUND, self.NH)
        mu = np.linspace(-1.0, 1.0, self.NMU)
        H, M = np.meshgrid(h, mu, indexing='ij')
        self.T = _transmittance_direct(R_GROUND + H, M, air)    # (NH, NMU, 3)
        self.hmax = R_TOP - R_GROUND

    def __call__(self, r, mu):
        r = np.asarray(r, float)
        mu = np.asarray(mu, float)
        x = np.clip((r - R_GROUND) / self.hmax, 0.0, 1.0) * (self.NH - 1)
        y = np.clip((mu + 1.0) * 0.5, 0.0, 1.0) * (self.NMU - 1)
        x0 = np.minimum(np.floor(x).astype(int), self.NH - 2)
        y0 = np.minimum(np.floor(y).astype(int), self.NMU - 2)
        fx, fy = (x - x0)[..., None], (y - y0)[..., None]
        T = self.T
        v = (T[x0, y0] * (1 - fx) * (1 - fy) + T[x0 + 1, y0] * fx * (1 - fy) + T[x0, y0 + 1] * (1 - fx) * fy
             + T[x0 + 1, y0 + 1] * fx * fy)
        return np.where(_hits_ground(r, mu)[..., None], 0.0, v)


def _phase_rayleigh(c):
    return 3.0 / (16.0 * math.pi) * (1.0 + c * c)


def _phase_mie(c, g=MIE_G):
    # Cornette-Shanks
    k = 3.0 / (8.0 * math.pi) * (1.0 - g * g) / (2.0 + g * g)
    return k * (1.0 + c * c) / np.power(np.maximum(1.0 + g * g - 2.0 * g * c, 1e-6), 1.5)


def _sphere_dirs(n):
    k = np.arange(n) + 0.5
    z = 1.0 - 2.0 * k / n
    a = math.pi * (3.0 - math.sqrt(5.0)) * k
    s = np.sqrt(1.0 - z * z)
    return np.stack([s * np.cos(a), z, s * np.sin(a)], -1)     # y up


class _MultiLUT:
    """Hillaire's multiple scattering: per height and sun angle, the light a point gets from scattering more than once,
    for an isotropic phase (Psi_ms: L_2nd / (1 - f_ms))."""

    NH, NMU = 24, 24

    def __init__(self, air: Air, trans: _TransmittanceLUT, dirs=64, steps=20):
        h = np.linspace(0.0, R_TOP - R_GROUND, self.NH)
        mus = np.linspace(-1.0, 1.0, self.NMU)
        D = _sphere_dirs(dirs)                                  # (dirs, 3)
        self.psi = np.zeros((self.NH, self.NMU, 3))
        for i, hh in enumerate(h):
            r0 = R_GROUND + hh
            for j, ms in enumerate(mus):
                sun = np.array([math.sqrt(max(1.0 - ms * ms, 0.0)), ms, 0.0])
                mu = D[:, 1]
                L = np.where(_hits_ground(r0, mu), -r0 * mu - np.sqrt(np.maximum(r0 * r0 * (mu * mu - 1.0) + R_GROUND ** 2, 0.0)),
                             _ray_top(r0, mu))
                t = (np.arange(steps) + 0.5) / steps
                s = L[:, None] * t                               # (dirs, steps)
                ds = (L / steps)[:, None]
                pos = np.array([0.0, r0, 0.0]) + D[:, None, :] * s[..., None]   # (dirs, steps, 3)
                rr = np.linalg.norm(pos, axis=-1)
                up = pos / rr[..., None]
                ext = _extinction(rr - R_GROUND, air)            # (dirs, steps, 3)
                rd, md, _ = _densities(rr - R_GROUND, air)
                scat = rd[..., None] * RAYLEIGH + md[..., None] * MIE_SCATTER
                # transmittance from the point back along the ray
                od = np.cumsum(ext * ds[..., None], axis=1) - 0.5 * ext * ds[..., None]
                Tv = np.exp(-od)
                mu_s = (up * sun).sum(-1)
                Ts = trans(rr, mu_s)                              # (dirs, steps, 3)
                iso = 1.0 / (4.0 * math.pi)
                L2 = (Tv * scat * Ts * iso * ds[..., None]).sum(1)          # (dirs, 3): sun light scattered once toward r0
                fms = (Tv * scat * iso * ds[..., None]).sum(1)              # the share of an even light scattered once back
                # the ground at the end of the rays that meet it: the sun on it, sent back up
                hit = _hits_ground(r0, mu)
                if hit.any():
                    pg = np.array([0.0, r0, 0.0]) + D * L[:, None]
                    ng = pg / np.linalg.norm(pg, axis=-1, keepdims=True)
                    Tg = np.exp(-(ext * ds[..., None]).sum(1))
                    sg = trans(np.full(dirs, R_GROUND + 1.0), (ng * sun).sum(-1))
                    L2 += np.where(hit[:, None], Tg * sg * np.maximum((ng * sun).sum(-1), 0.0)[:, None] * air.ground / math.pi, 0.0)
                L2m = L2.mean(0) * 4.0 * math.pi * iso      # (the mean over the sphere times the isotropic phase)
                Fm = fms.mean(0) * 4.0 * math.pi
                self.psi[i, j] = L2m / np.maximum(1.0 - Fm, 1e-4)
        self.hmax = R_TOP - R_GROUND

    def __call__(self, r, mu_s):
        x = np.clip((np.asarray(r, float) - R_GROUND) / self.hmax, 0.0, 1.0) * (self.NH - 1)
        y = np.clip((np.asarray(mu_s, float) + 1.0) * 0.5, 0.0, 1.0) * (self.NMU - 1)
        x0 = np.minimum(np.floor(x).astype(int), self.NH - 2)
        y0 = np.minimum(np.floor(y).astype(int), self.NMU - 2)
        fx, fy = (x - x0)[..., None], (y - y0)[..., None]
        P = self.psi
        return (P[x0, y0] * (1 - fx) * (1 - fy) + P[x0 + 1, y0] * fx * (1 - fy) + P[x0, y0 + 1] * (1 - fx) * fy
                + P[x0 + 1, y0 + 1] * fx * fy)


@lru_cache(maxsize=8)
def _luts(air: Air):
    trans = _TransmittanceLUT(air)
    return trans, _MultiLUT(air, trans)


def sun_direction(azimuth_deg, elevation_deg):
    """Toward the sun (y up), as camera.sun_direction."""
    from .camera import sun_direction as sd
    return np.asarray(sd(azimuth_deg, elevation_deg), float)


def radiance(dirs, sun, air: Air = Air(), steps=32):
    """The sky's radiance along unit directions dirs (..., 3; y up) with the sun toward unit `sun`, per unit of the sun's
    irradiance outside the air: (..., 3). Below the horizon: the ground (its albedo, lit by the sun and the sky) seen
    through the air between."""
    trans, multi = _luts(air)
    dirs = np.asarray(dirs, float)
    shp = dirs.shape[:-1]
    D = dirs.reshape(-1, 3)
    r0 = R_GROUND + max(air.altitude, 1.0)
    mu = D[:, 1]
    ground = _hits_ground(r0, mu)
    L = np.where(ground, -r0 * mu - np.sqrt(np.maximum(r0 * r0 * (mu * mu - 1.0) + R_GROUND ** 2, 0.0)), _ray_top(r0, mu))
    L = np.minimum(L, 1.0e6)
    # steps closer together near the start (the air is densest there)
    u = (np.arange(steps) + 0.5) / steps
    t = L[:, None] * u * u
    tn = L[:, None] * (np.arange(1, steps + 1) / steps) ** 2
    tp = L[:, None] * (np.arange(0, steps) / steps) ** 2
    ds = tn - tp                                               # (n, steps)
    pos = np.array([0.0, r0, 0.0]) + D[:, None, :] * t[..., None]
    rr = np.linalg.norm(pos, axis=-1)
    up = pos / rr[..., None]
    ext = _extinction(rr - R_GROUND, air)
    rd, md, _ = _densities(rr - R_GROUND, air)
    od = np.cumsum(ext * ds[..., None], axis=1) - 0.5 * ext * ds[..., None]
    Tv = np.exp(-od)
    c = D @ np.asarray(sun, float)
    mu_s = (up * np.asarray(sun, float)).sum(-1)
    Ts = trans(rr, mu_s)
    psi = multi(rr, mu_s)
    sr = rd[..., None] * RAYLEIGH
    sm = md[..., None] * MIE_SCATTER
    ins = sr * (_phase_rayleigh(c)[:, None, None] * Ts + psi) + sm * (_phase_mie(c)[:, None, None] * Ts + psi)
    out = (Tv * ins * ds[..., None]).sum(1)
    # the ground below the horizon: lit by the sun and the sky (roughly: an even sky of the zenith's), seen through the air
    if ground.any():
        pg = np.array([0.0, r0, 0.0]) + D * L[:, None]
        ng = pg / np.linalg.norm(pg, axis=-1, keepdims=True)
        Tg = np.exp(-(ext * ds[..., None]).sum(1))
        ns = (ng * np.asarray(sun, float)).sum(-1)
        sun_g = trans(np.full(len(D), R_GROUND + 1.0), ns) * np.maximum(ns, 0.0)[:, None]
        sky_g = math.pi * _zenith(air, tuple(np.round(np.asarray(sun, float), 4)))
        out = np.where(ground[:, None], out + Tg * (sun_g + sky_g) * air.ground / math.pi, out)
    return out.reshape(*shp, 3)


@lru_cache(maxsize=32)
def _zenith(air: Air, sun: tuple):
    return radiance(np.array([[0.0, 1.0, 0.0]]), np.asarray(sun), air)[0] * 0.8


def sun_irradiance(sun, air: Air = Air()):
    """The sun's irradiance at the camera (normal to it, rgb), per unit outside the air: its colour as it sets. 0 below
    the horizon."""
    trans, _ = _luts(air)
    r0 = R_GROUND + max(air.altitude, 1.0)
    return trans(np.array(r0), np.array(float(np.asarray(sun)[1])))


def image(sun, air: Air = Air(), width=256):
    """The sky as a latitude-longitude HDRI (h x w x 3, radiance per unit of the sun outside the air), in the convention
    Blackbody's HDRIs take (io/hdri.py, lume.py env_table: row 0 the zenith; the middle column -z)."""
    h = width // 2
    v = (np.arange(h) + 0.5) / h
    u = (np.arange(width) + 0.5) / width
    T, P = np.meshgrid(v * math.pi, (u - 0.5) * 2.0 * math.pi, indexing='ij')
    d = np.stack([np.sin(T) * np.sin(P), np.cos(T), -np.sin(T) * np.cos(P)], -1)
    return radiance(d, sun, air).astype(np.float32)


def ambient(sun, air: Air = Air(), n=512):
    """The sky's light on an upward surface over pi: the even sky radiance that would light it as much (the classic
    engine's ambient)."""
    D = _sphere_dirs(2 * n)
    D = D[D[:, 1] > 0.0]
    L = radiance(D, sun, air)
    # E = int L cos dw = mean(L cos) * 2 pi over the upper hemisphere's uniform directions
    E = (L * D[:, 1:2]).mean(0) * 2.0 * math.pi
    return E / math.pi


@lru_cache(maxsize=24)
def state(azimuth, elevation, haze=0.1, ground=0.3, altitude=0.0, width=256):
    """The physical sky for a sun at (azimuth, elevation) degrees: (key, ambient rgb, the sun's irradiance at the ground rgb,
    the HDRI), per unit of the sun outside the air. (Cached: the look asks every frame.)"""
    air = Air(haze=float(haze), ground=float(ground), altitude=float(altitude))
    s = sun_direction(azimuth, elevation)
    key = ('physical-sky', round(float(azimuth), 3), round(float(elevation), 3), air, int(width))
    return key, ambient(s, air), sun_irradiance(s, air), image(s, air, width)


def of_scene(scene, frame):
    """The physical sky of a scene at a frame (Lighting › Sky: Physical, no HDRI file), or None: (key, ambient, sun,
    image, the sun's strength outside the air: Key intensity)."""
    l = scene.data['lighting']
    if l.get('sky', 'colour') != 'physical' or l.get('environment'):
        return None
    az = round(float(scene.v('lighting', 'sun_azimuth', frame)) * 4.0) / 4.0     # (quarter degrees: an animated sun reuses)
    el = round(float(scene.v('lighting', 'sun_elevation', frame)) * 4.0) / 4.0
    key, amb, sun, img = state(az, el, round(float(l.get('haze', 0.1)), 4), round(float(l.get('ground_albedo', 0.3)), 3),
                               round(float(l.get('altitude', 0.0)), 1))
    return key, amb, sun, img, float(scene.v('lighting', 'sun_intensity', frame))
