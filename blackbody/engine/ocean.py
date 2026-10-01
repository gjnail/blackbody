"""The sea on open water: a Tessendorf FFT ocean in cascades, its foam, and the boundary it sets
for the liquid simulation.

Spectrum. The wind sea is JONSWAP-shaped around the peak wavelength, with the TMA correction for
the sea's depth and Mitsuyasu spreading (short waves come from wider angles than the peak). An
optional swell adds a second, narrow, long-crested system from its own direction. Dispersion is
omega^2 = (g k + sigma/rho k^3) tanh(k d): gravity waves, capillary ripples and finite depth. The
wind feeds the short waves (and without wind there are no capillary ripples: the water is glassy).
Each system is scaled to its significant wave height (4 x the standard deviation of the surface).
Short waves beyond the wind sea's peak are the wind's own: their level makes the surface's mean
square slope Cox and Munk's for that wind.

Cascades. The spectrum is split over four square tiles of decreasing size, each an n x n FFT, so
one sea spans swell hundreds of metres long down to ripples of a centimetre or two. A tile repeats
seamlessly; the tile ratios are irrational-ish so the cascades never line up.

Per frame the renderer sums each cascade at time t into two texture layers, a = (height,
displacement x, displacement z, d(disp x)/dz) and b = (dh/dx, dh/dz, d(disp x)/dx, d(disp z)/dz),
metres and metres per metre. A current carries the whole pattern (Doppler). The simulation only
sees the waves its grid can carry (at least six cells long): for the renderer those also exist as
separate 'low' layers, so inside the box the render adds exactly the waves the simulation lacks.
A surge (a solitary long wave, a tsunami, or a bore front) rides on top of it all.

Foam. Where crests fold over (the Jacobian of the two larger cascades' choppy displacement falls
below a threshold set so that whitecaps cover what they do on a real sea in that wind, after Monahan)
fresh whitecap foam is born on the largest tile; it fades into thinner aged foam which lasts much longer,
spreads and is stretched along the wind into streaks. It drifts with the wind and the current. The
foam is state carried from frame to frame on a fixed time lattice; a jump in time re-runs a pre-roll
long enough for the old foam to have faded, so a frame looks the same however it was reached.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, replace

import numpy as np

from .gpu import Uniforms

G = 9.81
SIGMA_RHO = 0.0728 / 1000.0      # water's surface tension over its density (m^3/s^2)
CASCADES = 4
RATIOS = (5.83, 6.17, 6.61)      # tile size ratios between neighbouring cascades
BAND = 8.0                       # lattice index at which each finer cascade takes over
LOW_CASCADES = 2                 # cascades whose longest waves the simulation can carry
SIM_CELLS = 6.0                  # shortest wave the simulation carries, in cells
RENDER_CELLS = 12.0              # shortest the rendered box surface keeps: shorter ones are damped in the
                                 # simulation and smoothed in its surface, so the sea's own are laid on
CAPILLARY_CUT = 2.0 * math.pi / 0.006   # 1/m: nothing shorter than about 6 mm
FOAM_DT = 1.0 / 24.0             # s, the foam's time lattice
FOLD_CASCADES = 2                # cascades whose crests break into whitecaps (ripples folding over make none)
FOAM_RES = 2                     # foam texels per spectrum texel (on the largest cascade's tile)
SURGE_KINDS = {'wave': 0.0, 'bore': 1.0, 'tsunami': 2.0}
DRAWDOWN = 0.6                   # a tsunami's trough ahead of its front, as a share of its height
DETAIL_SIZES = (128, 256, 512)

# spectrum layers: cascades, then the low parts of the first LOW_CASCADES cascades
LAYERS = CASCADES + LOW_CASCADES

OCEAN_KEYS = ('ocean_height', 'ocean_length', 'ocean_dir', 'ocean_spread', 'ocean_chop', 'ocean_depth',
              'swell_height', 'swell_length', 'swell_dir', 'ocean_detail',
              'surge_height', 'surge_length', 'surge_dir', 'surge_time', 'surge_kind')


@dataclass(frozen=True)
class OceanSpec:
    height: float = 0.0          # m, significant wave height of the wind sea (0 = none)
    length: float = 6.0          # m, peak wavelength of the wind sea
    direction: float = 90.0      # degrees, the way the wind sea travels (fire-local)
    spread: float = 0.3          # 0 long-crested .. 1 confused
    chop: float = 0.6            # crest sharpness (horizontal displacement)
    wind: float = 0.0            # m/s: feeds the short waves and ripples
    depth: float = 1.0           # m, depth of the sea (dispersion, shoaling)
    seed: int = 0
    swell: float = 0.0           # m, significant height of a swell from afar (0 = none)
    swell_length: float = 60.0   # m
    swell_dir: float = 90.0      # degrees
    detail: int = 256            # FFT size per cascade
    cap_depth: float = 0.0       # m, water depth the waves are held within (the box's water level; 0 = no cap)
    surge: float = 0.0           # m, height of a single long wave (a tsunami, a solitary wave) or bore front (0 = none)
    surge_length: float = 40.0   # m, how long its hump (or front) is
    surge_dir: float = 90.0      # degrees, the way it travels
    surge_time: float = 2.0      # s, when its crest (or front) passes the middle of the effect
    surge_kind: str = 'wave'     # 'wave' a hump that passes, 'bore' a front behind which the water stays raised,
                                 # 'tsunami' a bore that the sea draws back from first (a trough ahead of it)

    @property
    def on(self):
        return self.height > 1e-4 or self.swell > 1e-4 or self.surge > 1e-4

    @property
    def waves_on(self):
        """The random sea (wind sea or swell) is on (the surge aside)."""
        return self.height > 1e-4 or self.swell > 1e-4

    @property
    def n(self):
        return int(self.detail) if int(self.detail) in DETAIL_SIZES else 256

    @property
    def peak(self):
        """Longest peak wavelength of the two systems (m)."""
        lw = self.length if self.height > 1e-4 else 0.0
        ls = self.swell_length if self.swell > 1e-4 else 0.0
        return max(lw, ls, 0.05)

    @property
    def tiles(self):
        """Tile size of each cascade (m): the largest sixteen times the wind sea's peak wavelength
        (its peak sampled finely), and at least six swell wavelengths (a swell is narrow: a few
        components carry it)."""
        lw = self.length if self.height > 1e-4 else 0.0
        ls = self.swell_length if self.swell > 1e-4 else 0.0
        l0 = max(16.0 * lw, 6.0 * ls, 2.0)
        out = [l0]
        for r in RATIOS[:CASCADES - 1]:
            out.append(out[-1] / r)
        return tuple(out)

    def shape_key(self):
        """What the spectrum's shape depends on (not the choppiness, which acts at synthesis, nor the
        surge, which is added on top)."""
        return replace(self, chop=0.0, surge=0.0, surge_length=40.0, surge_dir=90.0, surge_time=2.0, surge_kind='wave')

    # -- the surge: a single long wave on top of the sea -----------------------------------------------

    @property
    def surge_speed(self):
        """How fast the surge travels (m/s): a long wave on water of the sea's depth, the faster the
        taller (sqrt(g (d + H)))."""
        return math.sqrt(G * (max(self.depth, 0.05) + max(self.surge, 0.0)))

    def surge_front(self, t):
        """Where the surge's crest (or front) is at time t (m along its direction from the origin)."""
        return self.surge_speed * (float(t) - self.surge_time)

    def surge_uniforms(self, u, t, origin=(0.0, 0.0)):
        """(height, 2 / length, direction x, z), (crest position, kind (1 bore), speed, depth); origin: a
        point the positions are measured from (fire-local x, z)."""
        if self.surge <= 1e-4:
            return u.v4().v4()
        a = math.radians(self.surge_dir)
        d = (math.sin(a), math.cos(a))
        u.v4(self.surge, 2.0 / max(self.surge_length, 0.05), d[0], d[1])
        return u.v4(self.surge_front(t) - (origin[0] * d[0] + origin[1] * d[1]), SURGE_KINDS.get(self.surge_kind, 0.0),
                    self.surge_speed, max(self.depth, 0.05))

    def surge_at(self, x, z, t):
        """Reference (CPU): the surge's height (m) at points (x, z) at time t."""
        if self.surge <= 1e-4:
            return np.zeros(np.shape(x))
        a = math.radians(self.surge_dir)
        xi = np.asarray(x) * math.sin(a) + np.asarray(z) * math.cos(a) - self.surge_front(t)
        q = 2.0 * xi / max(self.surge_length, 0.05)
        if self.surge_kind == 'bore':
            return 0.5 * self.surge * (1.0 - np.tanh(q))
        if self.surge_kind == 'tsunami':
            return 0.5 * self.surge * (1.0 - np.tanh(q)) - DRAWDOWN * self.surge / np.cosh(np.clip((q - 6.0) / 3.0, -40.0, 40.0)) ** 2
        return self.surge / np.cosh(np.clip(q, -40.0, 40.0)) ** 2


def dispersion(k, depth):
    """Angular frequency (rad/s) of waves of wavenumber k (1/m) on water of this depth (m)."""
    k = np.asarray(k, np.float64)
    return np.sqrt((G * k + SIGMA_RHO * k ** 3) * np.tanh(k * max(float(depth), 0.02)))


def _dispersion_dk(k, depth):
    d = max(float(depth), 0.02)
    k = np.maximum(np.asarray(k, np.float64), 1e-9)
    th = np.tanh(k * d)
    w = np.sqrt((G * k + SIGMA_RHO * k ** 3) * th)
    return ((G + 3.0 * SIGMA_RHO * k * k) * th + (G * k + SIGMA_RHO * k ** 3) * d * (1.0 - th * th)) / (2.0 * np.maximum(w, 1e-9))


_S_TABLE = np.linspace(0.0, 90.0, 1801)
_LOG_NORM = np.array([2.0 * math.lgamma(x + 1.0) + (2.0 * x - 1.0) * math.log(2.0) - math.log(math.pi) - math.lgamma(2.0 * x + 1.0)
                      for x in _S_TABLE])


def _log_norm_cos2s(s):
    """log of the normalisation of cos^(2s)(theta/2) over a full turn."""
    return np.interp(np.asarray(s, np.float64), _S_TABLE, _LOG_NORM)


def _jonswap_omni(k, kp, depth, gamma):
    """Omnidirectional wavenumber spectrum F(k) (unscaled; energy per unit k), JONSWAP in frequency
    with the TMA depth factor, and the angular frequency at each k."""
    w = np.maximum(dispersion(k, depth), 1e-6)
    wp = float(dispersion(kp, depth))
    sig = np.where(w <= wp, 0.07, 0.09)
    r = np.exp(-((w - wp) ** 2) / (2.0 * sig ** 2 * wp ** 2))
    S = G * G * w ** -5.0 * np.exp(-1.25 * (wp / w) ** 4) * gamma ** r
    # TMA: the depth limits the energy of long waves in shallow water
    wh = w * math.sqrt(max(depth, 0.02) / G)
    phi = np.where(wh <= 1.0, 0.5 * wh * wh, np.where(wh < 2.0, 1.0 - 0.5 * (2.0 - wh) ** 2, 1.0))
    return S * phi * _dispersion_dk(k, depth), w / wp


def _spreading(theta, s):
    """cos^(2s)(theta/2), normalised over a full turn."""
    return np.exp(_log_norm_cos2s(s) + 2.0 * s * np.log(np.maximum(np.abs(np.cos(0.5 * theta)), 1e-12)))


def _lattice(n, L):
    idx = np.fft.fftfreq(n, 1.0 / n)          # 0, 1, .. n/2-1, -n/2, .. -1
    kx, kz = np.meshgrid(idx * 2.0 * math.pi / L, idx * 2.0 * math.pi / L, indexing='xy')
    ix, iz = np.meshgrid(idx, idx, indexing='xy')
    return kx, kz, ix, iz


@functools.lru_cache(maxsize=8)
def _noise(seed, n, layer):
    rng = np.random.default_rng([int(seed) & 0x7FFFFFFF, n, layer, 7919])
    return (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / math.sqrt(2.0)


def _masks(spec):
    """Per cascade: (kx, kz, k, in-band mask) on its lattice. Arrays are (n, n), row = z index."""
    n = spec.n
    tiles = spec.tiles
    out = []
    for c, L in enumerate(tiles):
        kx, kz, ix, iz = _lattice(n, L)
        k = np.hypot(kx, kz)
        m = (k > 0) & (np.hypot(ix, iz) < n / 2 - 1)
        if c + 1 < len(tiles):
            m &= k < BAND * 2.0 * math.pi / tiles[c + 1]
        if c > 0:
            m &= k >= BAND * 2.0 * math.pi / L
        m &= k < CAPILLARY_CUT
        out.append((kx, kz, k, m))
    return out


def _spread_s(spread):
    t = min(max(float(spread), 0.0), 1.0)
    return 30.0 ** (1.0 - t) * 2.0 ** t


def _mitsuyasu(sp):
    def s_of(x):
        x = np.maximum(x, 1e-3)
        return np.clip(np.where(x < 1.0, sp * x ** 5, sp * x ** -2.5), 0.6, 80.0)
    return s_of


def _wind_tail(spec):
    """(wavenumber where the wind's own short waves take over (1/m), their saturation level B).
    Past the wind sea's peak the height spectrum approaches B/2 k^-3, whose level is set by the wind
    alone: its slopes then add up to the Cox & Munk mean square slope, 0.00512 per m/s of wind.
    No wind, no short waves: the water is glassy."""
    wind = max(float(spec.wind), 0.0)
    kp = 2.0 * math.pi / max(spec.length if spec.height > 1e-4 else spec.swell_length, 0.05)
    kj = 3.0 * kp
    B = 2.0 * 0.00512 * wind / max(math.log(CAPILLARY_CUT / kj), 1.0)
    return kj, B


def _heights(spec):
    """The significant heights asked for, held within what the water's depth can carry (deeper
    waves would break on the bottom)."""
    hw, hs = max(spec.height, 0.0), max(spec.swell, 0.0)
    if spec.cap_depth > 0.0:
        cap = 0.45 * spec.cap_depth
        tot = math.hypot(hw, hs)
        if tot > cap:
            hw, hs = hw * cap / tot, hs * cap / tot
    return hw, hs


def spectrum(spec: OceanSpec):
    """The sea's initial amplitudes (cached per shape: the choppiness does not change them)."""
    return _spectrum(spec.shape_key())


@functools.lru_cache(maxsize=12)
def _spectrum(spec: OceanSpec):
    """Returns a dict:
      h0: (LAYERS, n, n) complex64: h0(k) per layer (cascades, then the low parts)
      h0m: (LAYERS, n, n) complex64: conj(h0(-k))
      tiles: tile size per layer (m)
      var: mean square slope per cascade (both axes together)
      tail: mean square slope of the ripples finer than the last cascade
      crest: a bound on how far the surface strays from the level (m)
    The low layers start at zero; see low_parts()."""
    n = spec.n
    masks = _masks(spec)
    hw, hs = _heights(spec)
    depth = spec.depth
    kj, B = _wind_tail(spec)
    sp_w = _mitsuyasu(_spread_s(spec.spread))
    th_w = math.radians(spec.direction)
    th_s = math.radians(spec.swell_dir)
    # each system's omni spectrum and spreading on the lattice points, and its energy for the scaling
    rows = []
    energy = {'wind': 0.0, 'swell': 0.0}
    for c, (kx, kz, k, m) in enumerate(masks):
        dk2 = (2.0 * math.pi / spec.tiles[c]) ** 2
        ang = np.arctan2(kx, kz)
        row = {'k': k, 'm': m, 'dk2': dk2}
        kk = np.maximum(k, 1e-9)
        if spec.height > 1e-4:
            F, x = _jonswap_omni(k, 2.0 * math.pi / max(spec.length, 0.05), depth, 3.3)
            D = _spreading(ang - th_w, sp_w(x))
            row['wind'] = (F, D)
            energy['wind'] += float((np.where(m, F / kk * D, 0.0) * dk2).sum())
        if spec.swell > 1e-4:
            F, x = _jonswap_omni(k, 2.0 * math.pi / max(spec.swell_length, 0.05), depth, 7.0)
            D = _spreading(ang - th_s, np.full_like(x, 60.0))
            row['swell'] = (F, D)
            energy['swell'] += float((np.where(m, F / kk * D, 0.0) * dk2).sum())
        # the wind's short waves spread like the wind sea's (from the wind's direction)
        x = dispersion(k, depth) / max(float(dispersion(kj / 3.0, depth)), 1e-6)
        row['tail'] = (0.5 * B * kk ** -3.0, _spreading(ang - th_w, sp_w(x)))
        rows.append(row)
    scale = {'wind': (hw / 4.0) ** 2 / max(energy['wind'], 1e-30), 'swell': (hs / 4.0) ** 2 / max(energy['swell'], 1e-30)}
    h0 = np.zeros((LAYERS, n, n), np.complex128)
    var = []
    expected = 0.0
    for c, row in enumerate(rows):
        k, m, dk2 = row['k'], row['m'], row['dk2']
        kk = np.maximum(k, 1e-9)
        # the wind sea gives way to the wind's own short waves past kj
        t = _smooth(kj, 2.0 * kj, k)
        Ft, Dt = row['tail']
        psi = Ft * t / kk * Dt
        if 'wind' in row:
            F, D = row['wind']
            psi = psi + scale['wind'] * F * (1.0 - t) / kk * D
        if 'swell' in row:
            F, D = row['swell']
            psi = psi + scale['swell'] * F / kk * D
        amp2 = np.where(m, psi, 0.0) * dk2
        h0[c] = _noise(spec.seed, n, c) * np.sqrt(amp2 / 2.0)
        expected += float(amp2.sum())
    # a swell is carried by a handful of components: their random draw alone could make it a fifth
    # taller or shorter than asked. Keep the sea exactly as tall as the spectrum says.
    realized = float((2.0 * np.abs(h0) ** 2).sum())
    if realized > 0.0 and expected > 0.0:
        h0 *= math.sqrt(expected / realized)
    for c, row in enumerate(rows):
        var.append(float((2.0 * np.abs(h0[c]) ** 2 * row['k'] ** 2).sum()))
    h0m = np.conj(_flip(h0))
    # ripples past the finest cascade (for the roughness of the surface): the spectrum's k^-3 tail
    # extrapolated from the finest cascade's top octave
    # (a k^-3 height spectrum puts equal slope variance in every octave)
    kx, kz, k, m = masks[-1]
    tail = 0.0
    if m.any():
        k_top = float(k[m].max())
        top = m & (k > 0.5 * k_top)
        s_top = float((2.0 * np.abs(h0[CASCADES - 1][top]) ** 2 * k[top] ** 2).sum())
        tail = s_top * math.log(max(CAPILLARY_CUT / k_top, 1.0)) / math.log(2.0)
    amp_sum = float(np.abs(h0).sum()) * 2.0
    crest = min(amp_sum, 2.2 * math.hypot(hw, hs) + 0.02)
    return {'h0': h0.astype(np.complex64), 'h0m': h0m.astype(np.complex64), 'tiles': tuple(spec.tiles) + tuple(spec.tiles[:LOW_CASCADES]),
            'var': tuple(var), 'tail': tail, 'crest': crest, 'kp': 2.0 * math.pi / spec.peak}


def _smooth(a, b, x):
    t = np.clip((np.asarray(x, np.float64) - a) / max(b - a, 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _flip(a):
    """a(-k) on the FFT lattice (index i -> -i mod n on both axes)."""
    return np.roll(a[..., ::-1, ::-1], 1, axis=(-2, -1))


def sim_cut(h, cells=SIM_CELLS):
    """Wavenumber (1/m) above which the simulation's grid (cell size h) cannot carry waves."""
    return 2.0 * math.pi / (cells * max(float(h), 1e-6))


def low_weight(k, h, cells=SIM_CELLS):
    """How much of a wave of wavenumber k the simulation carries (1 long .. 0 too short)."""
    kc = sim_cut(h, cells)
    return 1.0 - _smooth(0.7 * kc, kc, k)


def low_parts(spec: OceanSpec, h, cells=SIM_CELLS):
    """h0 and conj(h0(-k)) of the waves the simulation carries (at least `cells` cells of size h
    long), per low cascade (LOW_CASCADES, n, n)."""
    return _low_parts(spec.shape_key(), round(float(h), 9), float(cells))


@functools.lru_cache(maxsize=12)
def _low_parts(spec: OceanSpec, h, cells):
    sp = _spectrum(spec)
    out0, outm = [], []
    for c in range(LOW_CASCADES):
        kx, kz, ix, iz = _lattice(spec.n, spec.tiles[c])
        w = low_weight(np.hypot(kx, kz), h, cells)
        out0.append(sp['h0'][c] * w)
        outm.append(sp['h0m'][c] * w)
    return np.stack(out0).astype(np.complex64), np.stack(outm).astype(np.complex64)


def surface_at(spec: OceanSpec, x, z, t, chop=None, current=(0.0, 0.0), layers=None):
    """Reference (CPU, direct summation): height, displacement x and z at points (x, z) (m) at time t.
    layers: which spectrum layers to sum (default: the cascades)."""
    sp = spectrum(spec)
    chop = spec.chop if chop is None else chop
    x = np.atleast_1d(np.asarray(x, np.float64))
    z = np.atleast_1d(np.asarray(z, np.float64))
    out = np.zeros((3, x.size))
    for li in (range(CASCADES) if layers is None else layers):
        L = sp['tiles'][li]
        kx, kz, ix, iz = _lattice(spec.n, L)
        h0, h0m = sp['h0'][li].astype(np.complex128), sp['h0m'][li].astype(np.complex128)
        nz = np.nonzero(np.abs(h0) + np.abs(h0m))
        kxs, kzs = kx[nz], kz[nz]
        k = np.hypot(kxs, kzs)
        w = dispersion(k, spec.depth)
        ht = (h0[nz] * np.exp(-1j * w * t) + h0m[nz] * np.exp(1j * w * t)) * np.exp(-1j * (kxs * current[0] + kzs * current[1]) * t)
        e = np.exp(1j * (np.outer(x, kxs) + np.outer(z, kzs)))
        out[0] += np.real(e @ ht)
        out[1] += np.real(e @ (1j * chop * kxs / k * ht))
        out[2] += np.real(e @ (1j * chop * kzs / k * ht))
    return out


def spec_from(q, level, wind=(0.0, 0.0, 0.0), seed=0, yaw=0.0):
    """An OceanSpec from the liquid settings (q: scene.data['liquid'] values at the frame), or None
    for flat water. Directions turn into the effect's own frame (yaw: its rotation, degrees)."""
    hw = float(q.get('ocean_height', 0.0))
    hs = float(q.get('swell_height', 0.0))
    hg = float(q.get('surge_height', 0.0))
    if level <= 0.0 or (hw <= 0.0 and hs <= 0.0 and hg <= 0.0):
        return None
    depth = float(q.get('ocean_depth', 0.0)) or float(level)
    return OceanSpec(height=hw, length=float(q.get('ocean_length', 6.0)),
                     direction=float(q.get('ocean_dir', 90.0)) - float(yaw), spread=float(q.get('ocean_spread', 0.3)),
                     chop=float(q.get('ocean_chop', 0.6)), wind=float(math.sqrt(sum(x * x for x in wind))),
                     depth=depth, seed=int(seed), swell=hs, swell_length=float(q.get('swell_length', 60.0)),
                     swell_dir=float(q.get('swell_dir', 90.0)) - float(yaw), detail=int(q.get('ocean_detail', 256)),
                     cap_depth=float(level), surge=hg, surge_length=float(q.get('surge_length', 40.0)),
                     surge_dir=float(q.get('surge_dir', 90.0)) - float(yaw), surge_time=float(q.get('surge_time', 2.0)),
                     surge_kind=str(q.get('surge_kind', 'wave')))


def crest_bound(spec):
    """How far above the level the sea's crests can reach (m)."""
    if spec is None or not spec.on:
        return 0.0
    crest = float(spectrum(spec)['crest']) if spec.waves_on else 0.0
    return crest + 1.05 * max(spec.surge, 0.0) + 0.05


# ---------------------------------------------------------------------------------------------
# GPU: the sea for the renderer
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SeaFoam:
    """How the sea's foam forms and lasts (the look's settings)."""
    whitecaps: float = 1.0       # 0 none .. 4: how easily folding crests break into foam
    fresh_life: float = 0.7      # s, whitecap foam
    life: float = 12.0           # s, the thinner foam it leaves
    bubbles_life: float = 0.9    # s
    streaks: float = 1.0         # how far aged foam is drawn out along the wind


def whitecap_cover(spec, wind, amount=1.0):
    """Share of the sea's surface under whitecaps: Monahan & O'Muircheartaigh (1980), 3.84e-6 U^3.41
    for a wind of U m/s (1 % at 10 m/s, 3 % at 14, 10 % at 20), times amount. A sea taller than its
    wind could raise counts as that wind's (a sea fully raised by wind U is 0.21 U^2 / g tall)."""
    u_sea = math.sqrt(max(spec.height, 0.0) * G / 0.21)
    u = max(float(wind), u_sea)
    return min(max(float(amount), 0.0) * 3.84e-6 * u ** 3.41, 0.3)


def fold_threshold(spec, wind, amount=1.0):
    """(Jacobian under which crests break into whitecaps, the spread of the Jacobian). The Jacobian of
    the two larger cascades is close to normal with standard deviation chop x their rms slope; the
    threshold is where the whitecap cover falls (a little inside, for its shorter tail)."""
    from statistics import NormalDist
    sp = spectrum(spec)
    sig = max(spec.chop * math.sqrt(sp['var'][0] + sp['var'][1]), 1e-4)
    # (the foam a breaking crest leaves as it moves on makes up about half of the whitecap cover)
    cov = 0.5 * whitecap_cover(spec, wind, amount)
    if cov <= 1e-6:
        return -100.0, sig
    return 1.0 + 0.92 * NormalDist().inv_cdf(cov) * sig, sig


def _pow2_at_least(x):
    return 1 << max(0, int(math.ceil(math.log2(max(x, 1)))))


def _tile_uniforms(u, tiles, count=8):
    inv = [1.0 / t for t in tiles] + [1.0] * (count - len(tiles))
    for i in range(0, count, 4):
        u.v4(*inv[i:i + 4])
    return u


class Sea:
    """The sea's layers and foam at a frame's time, as the renderer's textures:
      W: (n, n, 2 * LAYERS) rgba16float, layer 2l = a, 2l + 1 = b of spectrum layer l
      F: (2n, 2n, 1) rgba16float foam (fresh, aged, bubbles, height) on the largest cascade's tile,
         in the foam's drifting frame
    Layers: the cascades, then the low parts of the first LOW_CASCADES (the waves the simulation
    carries, for a grid of cell size h)."""

    def __init__(self, gpu):
        self.gpu = gpu
        sig = ['utex3d', 'st3d:rgba32float:w']
        self.k_spec = gpu.kernel('ocn_spectrum.wgsl', sig, 'main', workgroup=(8, 8, 1))
        self.k_jac = gpu.kernel('ocn_spectrum.wgsl', sig, 'jac', workgroup=(8, 8, 1))
        self.k_pack = gpu.kernel('ocn_pack.wgsl', ['utex3d', 'st3d:rgba16float:w'], workgroup=(8, 8, 1))
        self.k_foam = gpu.kernel('ocn_foam.wgsl', ['utex3d', 'utex3d', 'st3d:rgba16float:w'], workgroup=(8, 8, 1))
        self.n = 0
        self.H0 = self.fft = self.jfft = self.W = None
        self.F = []
        self._h0_key = None
        self.spec = None
        self.on = False
        self.flat_w = gpu.texture3d((1, 1, 2 * LAYERS), 'rgba16float', 'sea-flat')
        gpu.upload(self.flat_w, np.zeros((2 * LAYERS, 1, 1, 4), np.float16))
        self.flat_f = gpu.texture3d((1, 1, 1), 'rgba16float', 'sea-flat-foam')
        gpu.upload(self.flat_f, np.zeros((1, 1, 1, 4), np.float16))
        self.time = None
        self._built = None
        # foam state
        self.foam_k = None
        self.foam_key = None
        self.foam_off = np.zeros(2)
        self._fi = 0
        self.preroll_steps = 0
        self.info = {}

    # -- resources ------------------------------------------------------------------------------

    def _alloc(self, n):
        if n == self.n:
            return
        self.release()
        g = self.gpu
        self.n = n
        self.H0 = g.texture3d((n, n, LAYERS), 'rgba32float', 'sea-h0')
        from .fft import FFT2D
        self.fft = FFT2D(g, n, 2 * LAYERS, 'sea-fft')
        self.jfft = FFT2D(g, n, FOLD_CASCADES, 'sea-foam-fft')
        self.W = g.texture3d((n, n, 2 * LAYERS), 'rgba16float', 'sea-layers')
        nf = min(FOAM_RES * n, 1024)
        self.nfoam = nf
        self.F = [g.texture3d((nf, nf, 1), 'rgba16float', f'sea-foam{i}') for i in range(2)]
        self._h0_key = None
        self._built = None
        self.foam_k = None

    def release(self):
        for t in [self.H0, self.W] + list(self.F):
            if t is not None:
                t.destroy()
        for f in (self.fft, self.jfft):
            if f is not None:
                f.destroy()
        self.H0 = self.fft = self.jfft = self.W = None
        self.F = []
        self.n = 0

    def _upload(self, spec, h):
        key = (spec.shape_key(), round(float(h), 9))
        if key == self._h0_key:
            return
        sp = spectrum(spec)
        h0 = np.array(sp['h0'])
        h0m = np.array(sp['h0m'])
        lo0, lom = low_parts(spec, h, RENDER_CELLS)
        h0[CASCADES:] = lo0
        h0m[CASCADES:] = lom
        arr = np.stack([h0.real, h0.imag, h0m.real, h0m.imag], -1).astype(np.float32)   # (layers, z, x, 4)
        self.gpu.upload(self.H0, arr)
        self._h0_key = key
        self._built = None
        self.info = {'tiles': sp['tiles'], 'var': sp['var'], 'tail': sp['tail'], 'crest': sp['crest'], 'kp': sp['kp'],
                     'low_cut': sim_cut(h, RENDER_CELLS)}

    # -- per frame ----------------------------------------------------------------------------------

    def _spec_u(self, spec, t, current, chop=None, tiles=None):
        u = Uniforms().v4(self.n, t, spec.depth, spec.chop if chop is None else chop)
        u.v4(current[0], current[1], SIGMA_RHO, 0.0)
        return _tile_uniforms(u, tiles if tiles is not None else self.info['tiles'])

    def update(self, b, spec, t, h, current=(0.0, 0.0), wind=(0.0, 0.0), foam=None):
        """The sea at time t (s) for a simulation grid of cell size h (m). current, wind: fire-local
        (x, z) m/s. foam: SeaFoam."""
        if spec is None or not spec.on:
            self.on = False
            self.spec = spec
            return
        self.on = True
        self._alloc(spec.n)
        self._upload(spec, h)
        cur = (float(current[0]), float(current[1]))
        key = (spec, cur, round(float(h), 9), float(t))
        if key != self._built:
            b.run(self.k_spec, [self.H0, self.fft.a], self._spec_u(spec, t, cur), (self.n, self.n, LAYERS))
            self.fft.run(b)
            b.run(self.k_pack, [self.fft.a, self.W], None, (self.n, self.n, 2 * LAYERS))
            self._built = key
        self.spec = spec
        self.time = float(t)
        self._foam(b, spec, float(t), cur, (float(wind[0]), float(wind[1])), foam or SeaFoam())

    # -- foam ---------------------------------------------------------------------------------------

    def _foam(self, b, spec, t, cur, wind, fp):
        """Bring the foam state to the lattice step at t: step on from the last state, or run a
        pre-roll after a jump (or when the settings changed while the time stood still)."""
        k = int(math.floor(t / FOAM_DT + 1e-6))
        key = (spec.shape_key(), spec.chop, fp, self.n)
        drift = np.array([cur[0] + 0.03 * wind[0], cur[1] + 0.03 * wind[1]])
        restart = (self.foam_k is None or k < self.foam_k or k - self.foam_k > int(1.0 / FOAM_DT)
                   or (key != self.foam_key and k == self.foam_k))
        self.preroll_steps = 0
        if restart:
            pre = min(3.0 * fp.life + 3.0 * fp.fresh_life, 45.0) if fp.whitecaps > 0.0 else 0.0
            k0 = k - int(math.ceil(pre / FOAM_DT))
            self.foam_off = drift * (k0 * FOAM_DT)
            self._foam_step(b, spec, k0, cur, wind, drift, fp, start=True)
            self.preroll_steps = k - k0
            for i in range(k0 + 1, k + 1):
                self._foam_step(b, spec, i, cur, wind, drift, fp)
        else:
            for i in range(self.foam_k + 1, k + 1):
                self._foam_step(b, spec, i, cur, wind, drift, fp)
        self.foam_k = k
        self.foam_key = key

    def _foam_step(self, b, spec, i, cur, wind, drift, fp, start=False):
        t = i * FOAM_DT
        if not start:
            self.foam_off = self.foam_off + drift * FOAM_DT
        tiles = self.info['tiles'][:CASCADES]
        b.run(self.k_jac, [self.H0, self.jfft.a], self._spec_u(spec, t, cur, tiles=tiles), (self.n, self.n, FOLD_CASCADES))
        self.jfft.run(b)
        wl = math.hypot(*wind)
        wd = (wind[0] / wl, wind[1] / wl) if wl > 1e-3 else (0.0, 1.0)
        thr, sig = fold_threshold(spec, wl, fp.whitecaps)
        nf = self.nfoam
        # aged foam spreads about a tenth of a metre per second along the wind at 5 m/s
        spread = 0.004 * fp.streaks * (1.0 + wl / 5.0) * nf / tiles[0] * FOAM_DT * 24.0
        u = (Uniforms().v4(nf, FOAM_DT, fp.fresh_life, fp.life)
             .v4(thr, fp.bubbles_life, spread, 1.0 if start else 0.0)
             .v4(wd[0], wd[1], 0.25, self.n)
             .v4(self.foam_off[0], self.foam_off[1], tiles[0], sig)
             .v4(*(1.0 / x for x in tiles[:4])))
        src, dst = self.F[self._fi], self.F[1 - self._fi]
        b.run(self.k_foam, [self.jfft.a, src, dst], u, (nf, nf, 1))
        self._fi = 1 - self._fi

    def foam_offsets(self):
        """Where the foam's frame has drifted to, in units of the largest cascade's tile."""
        if not self.on or self.W is None:
            return (0.0, 0.0)
        return tuple(self.foam_off / self.info['tiles'][0])

    # -- the sea's height over the box ----------------------------------------------------------------

    def box_map(self, b, origin, dims, h, res, level_m=0.0, layer=None, sides=(1.0, 1.0, 1.0, 1.0)):
        """Map the sea's height over the simulation box's footprint (res: texels x, z), with the
        open-water layer's waves (layer: OceanLayer or LayerView, or None)."""
        if not self.on or self.W is None:
            return
        res = (max(8, int(res[0])), max(8, int(res[1])))
        if self.BOX is None or self.BOX.size[:2] != res:
            if self.BOX is not None:
                self.BOX.destroy()
            self.BOX = self.gpu.texture2d(res[0], res[1], 'rgba16float', 'sea-box-height')
            self._box_key = None
        on = layer is not None and layer.on
        key = (self._built, tuple(origin), tuple(dims), h, res, id(layer.texture()) if on else None, self._layer_stamp)
        if key == self._box_key:
            return
        if self.k_box is None:
            self.k_box = self.gpu.kernel('ocn_boxmap.wgsl', ['tex3d', 'tex3d', 'smp', 'st2d:rgba16float:w', 'tex2d'],
                                         workgroup=(8, 8, 1))
        u = self.uniforms(Uniforms(), level_m, time=self.time or 0.0, layer=layer, sides=sides)
        u.v4(res[0], res[1], origin[0], origin[2]).v4(dims[0] * h, dims[2] * h)
        lt = layer.texture() if on else self._flat2()
        b.run(self.k_box, [self.W, self.F[self._fi], self.gpu.repeat, self.BOX, lt], u, (res[0], res[1], 1))
        self._box_key = key

    _layer_stamp = 0

    def layer_changed(self):
        """The open-water layer has stepped (the box's height map must be redone)."""
        self._layer_stamp += 1

    def _flat2(self):
        if self.flat2 is None:
            self.flat2 = self.gpu.texture2d(1, 1, 'rgba16float', 'sea-flat-layer')
            self.gpu.upload(self.flat2, np.zeros((1, 1, 4), np.float16))
        return self.flat2

    BOX = None
    k_box = None
    _box_key = None

    # -- what the renderer binds ----------------------------------------------------------------------

    def textures(self, layer=None):
        """The four textures the renderer binds: the sea's layers, its foam, the open-water layer
        and the sea's height over the box (flat stand-ins without them)."""
        lt = layer.texture() if layer is not None and layer.on else self._flat2()
        if self.on and self.W is not None:
            return [self.W, self.F[self._fi], lt, self.BOX if self.BOX is not None else self.flat2]
        return [self.flat_w, self.flat_f, lt, self._flat2()]

    flat2 = None

    def uniforms(self, u: Uniforms, level_m, whitecaps=0.0, rainbow=0.0, look=None, pix=0.0, current=(0.0, 0.0),
                 time=0.0, layer=None, sides=(1.0, 1.0, 1.0, 1.0)):
        """ocn, ocn2 and ocx (see ocn_sample.wgsl). look: the WaterLook (foam, crest glow, gusts, wind)."""
        on = self.on and self.W is not None
        info = self.info if on else {}
        tiles = info.get('tiles', (1.0,) * LAYERS)
        if on:
            u.v4(1.0, 1.0 / tiles[0], 1.0 / tiles[1], level_m)
            u.v4(0.0, whitecaps, crest_bound(self.spec), rainbow)
        else:
            u.v4(0.0, 1.0, 1.0, level_m)
            u.v4(0.0, 0.0, 0.0, rainbow)
        u.v4(*(1.0 / t for t in tiles[:4]))
        u.v4(1.0 / tiles[4], 1.0 / tiles[5], max(self.n, 1), 1.0 / (2 * LAYERS))
        var = info.get('var', (0.0,) * CASCADES)
        u.v4(*var[:4])
        offs = self.foam_offsets()
        u.v4(offs[0], offs[1], info.get('tail', 0.0), pix)
        g = lambda k, d: float(getattr(look, k, d)) if look is not None else d
        wind = getattr(look, 'wind', (0.0, 0.0, 0.0)) if look is not None else (0.0, 0.0, 0.0)
        wl = math.hypot(wind[0], wind[2])
        thr, sig = fold_threshold(self.spec, wl, whitecaps) if on else (-100.0, 1.0)
        u.v4(g('sea_foam', 1.0), g('foam_streaks', 0.7), sig, 1.0)
        wd = (wind[0] / wl, wind[2] / wl) if wl > 1e-3 else (0.0, 1.0)
        u.v4(wd[0], wd[1], wl, g('gusts', 0.0))
        gc = tuple(getattr(look, 'crest_glow_color', (0.1, 0.55, 0.45))) if look is not None else (0.0, 0.0, 0.0)
        u.v4(gc[0], gc[1], gc[2], g('crest_glow', 1.0) if look is not None else 0.0)
        u.v4(current[0], current[1], time, thr)
        if layer is not None and layer.on:
            layer.uniforms(u)
        else:
            u.v4().v4()
        if on and self.spec.surge > 1e-4:
            self.spec.surge_uniforms(u, time)
        else:
            u.v4().v4()
        sc = tuple(getattr(look, 'standin_color', (0.3, 0.3, 0.3))) if look is not None else (0.3, 0.3, 0.3)
        u.v4(sc[0], sc[1], sc[2], 1.0 if look is not None else 0.0)
        return u.v4(*sides)


# ---------------------------------------------------------------------------------------------
# GPU: the sea at the simulation's edges
# ---------------------------------------------------------------------------------------------

class SeaBoundary:
    """Per substep: the open water's surface height and orbital velocity per column of the
    simulation grid (the OCN texture the liquid kernels read), from the waves the grid carries."""

    def __init__(self, gpu):
        self.gpu = gpu
        self.k_spec = gpu.kernel('ocn_spectrum.wgsl', ['utex3d', 'st3d:rgba32float:w'], 'sim', workgroup=(8, 8, 1))
        self.k_bc = gpu.kernel('ocn_bc.wgsl', ['utex3d', 'st2d:rgba32float:w'], workgroup=(8, 8, 1))
        self._key = None
        self.fft = None
        self.H0 = None
        self.ns = 0
        self.layers = 0
        self.tiles = ()
        self.kp = 1.0
        self.depth = 1.0
        self._dummy = gpu.texture3d((1, 1, 1), 'rgba32float', 'sea-bc-none')
        gpu.upload(self._dummy, np.zeros((1, 1, 1, 4), np.float32))

    def release(self):
        for x in (self.fft, self.H0):
            if x is not None:
                x.destroy()
        self.fft = self.H0 = None
        self.ns = 0
        self._key = None

    def prepare(self, spec, h):
        key = (spec.shape_key(), round(float(h), 9)) if (spec is not None and spec.on) else None
        if key == self._key:
            return
        if key is None:
            self._key = key
            self.layers = 0
            return
        lo0, lom = low_parts(spec, h)
        n = spec.n
        idx = np.fft.fftfreq(n, 1.0 / n).astype(int)
        ix, iz = np.meshgrid(idx, idx, indexing='xy')
        live = ((np.abs(lo0) + np.abs(lom)) > 0).any(0)
        reach = int(max(np.abs(ix)[live].max(initial=0), np.abs(iz)[live].max(initial=0)))
        # at least about six texels along the shortest wave (sampled bicubic), within the tile's own
        # resolution
        ns = min(max(n, 512), max(16, _pow2_at_least(6 * reach + 4)))
        # the same wave vectors on an ns x ns lattice (cropped, or padded with zeros)
        sidx = np.fft.fftfreq(ns, 1.0 / ns).astype(int)
        keep = np.abs(sidx) < min(n, ns) // 2
        pick = sidx[keep] % n
        small0 = np.zeros((LOW_CASCADES, ns, ns), np.complex64)
        smallm = np.zeros((LOW_CASCADES, ns, ns), np.complex64)
        small0[:, np.ix_(keep, keep)[0], np.ix_(keep, keep)[1]] = lo0[:, pick[:, None], pick[None, :]]
        smallm[:, np.ix_(keep, keep)[0], np.ix_(keep, keep)[1]] = lom[:, pick[:, None], pick[None, :]]
        if ns != self.ns or self.fft is None:
            self.release()
            from .fft import FFT2D
            self.ns = ns
            self.fft = FFT2D(self.gpu, ns, LOW_CASCADES, 'sea-bc-fft')
            self.H0 = self.gpu.texture3d((ns, ns, LOW_CASCADES), 'rgba32float', 'sea-bc-h0')
        self._key = key
        arr = np.stack([small0.real, small0.imag, smallm.real, smallm.imag], -1).astype(np.float32)
        self.gpu.upload(self.H0, arr)
        self.layers = LOW_CASCADES if live.any() else 0
        self.tiles = spec.tiles[:LOW_CASCADES]
        self.kp = spectrum(spec)['kp']
        self.depth = spec.depth

    def write(self, b, ocn, grid_bytes, dims, level_cells, spec, t, current=(0.0, 0.0), h=None, origin=None):
        """Record the columns' sea at time t into ocn and return the wavenumber the orbital motion
        fades with under the surface (the sea's peak; a passing surge moves the whole depth).
        grid_bytes: the solver's Grid uniforms; origin: the grid's corner (fire-local m)."""
        if spec is not None and spec.on and h is not None:
            self.prepare(spec, h)
        else:
            self.layers = 0
        if self.layers:
            u = Uniforms().v4(self.ns, t, self.depth, 0.0).v4(current[0], current[1], SIGMA_RHO, 0.0)
            _tile_uniforms(u, self.tiles)
            b.run(self.k_spec, [self.H0, self.fft.a], u, (self.ns, self.ns, self.layers))
            self.fft.run(b)
            src = self.fft.a
        else:
            src = self._dummy
        u = _tile_uniforms(Uniforms().v4(level_cells, self.layers, max(self.ns, 1), 0.0), self.tiles)
        kp = self.kp if self.layers else 1.0
        if spec is not None and spec.surge > 1e-4:
            spec.surge_uniforms(u, t)
            # a long wave moves the water all the way down to the bottom alike: while it passes, the
            # motion at the surface is the motion at every depth
            if origin is not None and h is not None:
                a = math.radians(spec.surge_dir)
                c = (origin[0] + 0.5 * dims[0] * h, origin[2] + 0.5 * dims[2] * h)
                gap = spec.surge_front(t) - (c[0] * math.sin(a) + c[1] * math.cos(a))
                reach = 3.0 * spec.surge_length + max(dims[0], dims[2]) * h
                if spec.surge_kind == 'tsunami':
                    reach += 4.5 * spec.surge_length   # (its trough runs 3 lengths ahead of the front)
                if abs(gap) < reach or (spec.surge_kind in ('bore', 'tsunami') and gap > -reach):
                    kp = min(kp, 1e-3)
        else:
            u.v4().v4()
        b.run(self.k_bc, [src, ocn], grid_bytes + u.tobytes(), (dims[0], dims[2], 1))
        return kp
