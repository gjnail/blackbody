"""The atmosphere and what falls through it: moist thermodynamics, soundings, and the physics of
raindrops, snowflakes, graupel, hailstones and ice pellets (sleet) falling through the air.

Shared by the precipitation particles (engine/weather.py, wgsl/wx_*.wgsl, whose WGSL twin is
wx_common.wgsl) and the cloud solver. SI units unless a name says otherwise; temperatures in C.

What arrives at the ground is decided the way the sky decides it: each size of particle is followed
down through the temperatures and humidity of the air column (fall_column). Snow that falls through a
layer above freezing melts (slowly in dry air, which it cools by evaporating); partly melted snow
refreezes into ice pellets in a deep enough cold layer below; drops that melted completely stay liquid
below freezing (they need a nucleus to freeze, Bigg 1953) and freeze only on what they hit: freezing
rain. Hail melts on the way down from the freezing level, so small hail arrives as big raindrops.

Sources: Pruppacher and Klett (1997) for the growth, melting and ventilation of hydrometeors;
Locatelli and Hobbs (1974) for snow and graupel masses and fall speeds; Atlas et al. (1973) for
raindrops; Marshall and Palmer (1948), Gunn and Marshall (1958) for size distributions; Mitra et al.
(1990) for melting snow; Bolton (1980) and Buck (1981) for saturation vapour pressure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# -- constants ------------------------------------------------------------------------------------
G = 9.80665
RD = 287.05            # J/(kg K), dry air
RV = 461.5             # J/(kg K), water vapour
CP = 1004.6            # J/(kg K), dry air at constant pressure
EPS = RD / RV          # 0.622
KELVIN = 273.15
LF = 3.336e5           # J/kg, melting
CW = 4186.0            # J/(kg K), water
CI = 2106.0            # J/(kg K), ice
RHO_W = 1000.0
RHO_I = 917.0
P0 = 101325.0


def lv(t_c):
    """Latent heat of vaporisation (J/kg) at t_c."""
    return 2.501e6 - 2370.0 * t_c


def ls(t_c):
    """Latent heat of sublimation (J/kg)."""
    return 2.834e6 - 290.0 * np.minimum(t_c, 0.0)


def es_water(t_c):
    """Saturation vapour pressure over liquid water (Pa), Bolton (1980); valid supercooled too."""
    t = np.asarray(t_c, dtype=np.float64)
    return 611.2 * np.exp(17.67 * t / (t + 243.5))


def es_ice(t_c):
    """Saturation vapour pressure over ice (Pa), Buck (1981)."""
    t = np.asarray(t_c, dtype=np.float64)
    return 611.15 * np.exp(22.452 * t / (t + 272.55))


def rho_vs(t_c, over_ice=False):
    """Saturation vapour density (kg/m^3) at t_c over water or ice."""
    e = es_ice(t_c) if over_ice else es_water(t_c)
    return e / (RV * (np.asarray(t_c) + KELVIN))


def air_density(t_c, p, rh=0.0):
    """Density of moist air (kg/m^3)."""
    e = rh * es_water(t_c)
    return (p - e) / (RD * (t_c + KELVIN)) + e / (RV * (t_c + KELVIN))


def conductivity(t_c):
    """Thermal conductivity of air (W/(m K))."""
    return 2.40e-2 + 7.73e-5 * t_c


def diffusivity(t_c, p):
    """Diffusivity of water vapour in air (m^2/s)."""
    return 2.11e-5 * ((t_c + KELVIN) / KELVIN) ** 1.94 * (P0 / p)


def viscosity(t_c):
    """Dynamic viscosity of air (Pa s), Sutherland."""
    tk = t_c + KELVIN
    return 1.458e-6 * tk ** 1.5 / (tk + 110.4)


def mixing_ratio(e, p):
    return EPS * e / (p - e)


def moist_lapse(t_c, p):
    """Saturated (pseudo)adiabatic lapse rate (K/m) at t_c and p."""
    tk = t_c + KELVIN
    rs = mixing_ratio(es_water(t_c), p)
    L = lv(t_c)
    return G * (1.0 + L * rs / (RD * tk)) / (CP + L * L * rs * EPS / (RD * tk * tk))


DRY_LAPSE = G / CP     # 9.76 K/km


# -- soundings -------------------------------------------------------------------------------------

@dataclass
class Sounding:
    """The air column: temperature (C) and relative humidity (0..1) at heights z (m above the ground),
    linear between; pressure from hydrostatic balance. Above the top level the temperature keeps the top
    lapse rate. The humidity is over ice below freezing (where clouds and falling snow keep the air near
    ice saturation) and over water above."""
    z: list = field(default_factory=lambda: [0.0, 10000.0])
    t: list = field(default_factory=lambda: [15.0, -50.0])
    rh: list = field(default_factory=lambda: [0.8, 0.5])
    p0: float = P0

    def __post_init__(self):
        self._z = np.asarray(self.z, np.float64)
        self._t = np.asarray(self.t, np.float64)
        self._rh = np.asarray(self.rh, np.float64)
        # pressure on a fine grid by integrating dp/dz = -g p / (Rd Tv)
        zz = np.linspace(0.0, max(self._z[-1], 16000.0), 1601)
        tt = self.temp(zz)
        tv = (tt + KELVIN) * (1.0 + 0.61 * 0.005)
        lnp = np.log(self.p0) - np.concatenate([[0.0], np.cumsum(G / (RD * 0.5 * (tv[1:] + tv[:-1])) * np.diff(zz))])
        self._pz, self._lnp = zz, lnp

    def temp(self, z):
        z = np.asarray(z, np.float64)
        t = np.interp(z, self._z, self._t)
        if len(self._z) > 1:
            lapse = (self._t[-1] - self._t[-2]) / max(self._z[-1] - self._z[-2], 1e-6)
            t = np.where(z > self._z[-1], self._t[-1] + lapse * (z - self._z[-1]), t)
        return t

    def humidity(self, z):
        return np.interp(np.asarray(z, np.float64), self._z, self._rh)

    def pressure(self, z):
        return np.exp(np.interp(np.asarray(z, np.float64), self._pz, self._lnp))

    def vapour(self, z):
        """Water vapour density (kg/m^3) at z."""
        t = self.temp(z)
        return self.humidity(z) * np.where(t < 0.0, rho_vs(t, True), rho_vs(t, False))

    def freezing_level(self):
        """Lowest height (m) where the air is at or below freezing going up, 0 if the ground is."""
        zz = np.linspace(0.0, 12000.0, 2401)
        tt = self.temp(zz)
        if tt[0] <= 0.0:
            return 0.0
        i = int(np.argmax(tt <= 0.0))
        return float(zz[i]) if tt[i] <= 0.0 else float('inf')

    def level_of(self, t_c, lo=0.0, hi=12000.0):
        """Lowest height between lo and hi where the temperature is at or below t_c (hi if never)."""
        zz = np.linspace(lo, hi, 2401)
        tt = self.temp(zz)
        below = np.nonzero(tt <= t_c)[0]
        return float(zz[below[0]]) if len(below) else hi


def winter_sounding(ground_t, warm_t=None, warm_z=1500.0, rh=0.97, lapse=6.5e-3):
    """A precipitating winter column: ground_t at the ground, then (optionally) a layer of warmer air
    peaking at warm_t at height warm_z (an inversion: warm air riding over cold), and the standard lapse
    rate above it. Saturated-ish (rh) as precipitating air is."""
    if warm_t is None or warm_t <= ground_t - lapse * warm_z:
        return Sounding(z=[0.0, 10000.0], t=[ground_t, ground_t - lapse * 10000.0], rh=[rh, rh])
    top = warm_z + 8000.0
    return Sounding(z=[0.0, warm_z, top], t=[ground_t, warm_t, warm_t - lapse * 8000.0], rh=[rh, rh, rh])


def convective_sounding(ground_t, rh=0.7, lapse=6.5e-3):
    """A summer afternoon column (thunderstorms, hail): the standard lapse rate from a warm ground."""
    return Sounding(z=[0.0, 12000.0], t=[ground_t, ground_t - lapse * 12000.0], rh=[rh, 0.5])


# -- hydrometeors ----------------------------------------------------------------------------------
# Kinds of particle, as the GPU code numbers them too (wx_common.wgsl)
RAIN, SNOW, GRAUPEL, HAIL, PELLET = 0, 1, 2, 3, 4
KIND_NAMES = {RAIN: 'rain', SNOW: 'snow', GRAUPEL: 'graupel', HAIL: 'hail', PELLET: 'ice pellets'}


def rain_speed(d_mm, rho_air=1.2):
    """Terminal speed (m/s) of a raindrop d_mm across (Atlas et al. 1973), corrected for air density."""
    v = np.maximum(9.65 - 10.3 * np.exp(-0.6 * np.asarray(d_mm, np.float64)), 0.0)
    # small drops (Stokes-like) never reach zero
    v = np.where(np.asarray(d_mm) < 0.3, 4.0 * np.asarray(d_mm, np.float64) ** 1.2 * 3.0, v)
    return v * (1.2 / rho_air) ** 0.4


def snow_mass(d_mm):
    """Mass (kg) of a dry snowflake aggregate d_mm across: 0.0069 D^2 g with D in cm (Heymsfield's
    aggregates; a 5 mm flake is 1.7 mg, about 26 kg/m^3 of its sphere). (Locatelli and Hobbs' fluffiest
    dendrite aggregates are a third of that, and melted three times too fast for the warm layers that
    turn snow to sleet.)"""
    return 6.9e-8 * np.asarray(d_mm, np.float64) ** 2


def snow_speed(d_mm, rho_air=1.2):
    """Terminal speed (m/s) of a dry snowflake aggregate (Locatelli and Hobbs 1974: 0.8 D^0.16)."""
    return 0.8 * np.asarray(d_mm, np.float64) ** 0.16 * (1.2 / rho_air) ** 0.4


RHO_GRAUPEL = 300.0    # kg/m^3, rimed snow pellets


def graupel_mass(d_mm):
    """Mass (kg) of a graupel pellet d_mm across (a sphere of rime, about 300 kg/m^3)."""
    return math.pi / 6.0 * (np.asarray(d_mm, np.float64) * 1e-3) ** 3 * RHO_GRAUPEL


def graupel_speed(d_mm, rho_air=1.2):
    """Terminal speed (m/s) of lump graupel (Locatelli and Hobbs 1974: 1.3 D^0.66)."""
    return 1.3 * np.asarray(d_mm, np.float64) ** 0.66 * (1.2 / rho_air) ** 0.4


def sphere_speed(d_mm, rho_p, rho_air=1.2, cd=0.6):
    """Terminal speed (m/s) of a dense sphere (hail, an ice pellet) with drag coefficient cd."""
    d = np.asarray(d_mm, np.float64) * 1e-3
    return np.sqrt(4.0 * rho_p * G * d / (3.0 * cd * rho_air))


def sphere_mass(d_mm, rho_p):
    return math.pi / 6.0 * (np.asarray(d_mm, np.float64) * 1e-3) ** 3 * rho_p


def drop_diameter(mass):
    """Diameter (mm) of the drop a mass of water (kg) makes."""
    return (6.0 * np.asarray(mass, np.float64) / (math.pi * RHO_W)) ** (1.0 / 3.0) * 1e3


class Particles:
    """Hydrometeors' states as arrays (one entry per particle): what each is (kind: the shape it keeps
    while it holds ice), its ice and water (kg), temperature (C), size when dry (d0, mm: snow's aggregate
    size), whether its water can freeze (it has met a nucleus, or holds ice), and the most of it that has
    ever been melted (fmax: a flake that melts collapses, and stays collapsed if it refreezes)."""

    def __init__(self, kind, ice, water, t, d0, nucleated, fmax=None):
        self.kind = np.array(kind, np.int32, ndmin=1)
        self.ice = np.array(ice, np.float64, ndmin=1)
        self.water = np.array(water, np.float64, ndmin=1)
        self.t = np.array(t, np.float64, ndmin=1)
        self.d0 = np.array(d0, np.float64, ndmin=1)
        self.nucleated = np.array(nucleated, bool, ndmin=1)
        self.fmax = np.zeros(len(self.ice)) if fmax is None else np.array(fmax, np.float64, ndmin=1)

    def __len__(self):
        return len(self.ice)

    @property
    def mass(self):
        return self.ice + self.water

    @property
    def melted(self):
        """Share of each one's mass that is liquid."""
        m = self.mass
        return np.where(m > 0, self.water / np.maximum(m, 1e-30), 0.0)

    def arriving_kind(self):
        """What each is when it lands: rain (all liquid, above freezing or supercooled), snow (partly
        melted snow counts as wet snow until most of it is water), graupel, hail, or ice pellets (a drop
        that froze on the way down)."""
        f = self.melted
        k = self.kind.copy()
        k = np.where(self.kind == RAIN, PELLET, k)
        k = np.where((self.kind == SNOW) & (f >= 0.7), RAIN, k)
        k = np.where((f > 0.98) | (self.ice <= 0.0), RAIN, k)
        return k

    def diameter(self):
        """Size now (mm). Melting snow collapses toward the drop its water would make (Mitra et al. 1990);
        graupel is a ball of rime (its meltwater soaks in first); a stone or a frozen drop is a sphere of
        its ice with its meltwater as a film over it."""
        m = self.mass
        dw = drop_diameter(m)
        f = self.melted
        has_ice = self.ice > 0.0
        d_snow = dw + (np.maximum(self.d0, dw) - dw) * (1.0 - np.maximum(f, self.fmax)) ** 0.7
        d_gr = np.maximum((6.0 * self.ice / (math.pi * RHO_GRAUPEL)) ** (1.0 / 3.0) * 1e3, dw)
        d_sph = (6.0 * (self.ice / RHO_I + self.water / RHO_W) / math.pi) ** (1.0 / 3.0) * 1e3
        d = np.where(self.kind == SNOW, d_snow, np.where(self.kind == GRAUPEL, d_gr, d_sph))
        return np.where(has_ice, d, dw)

    def speed(self, rho_air):
        """Terminal fall speeds (m/s) in air of density rho_air."""
        d = self.diameter()
        f = self.melted
        dw = drop_diameter(self.mass)
        v_rain = rain_speed(dw, rho_air)
        # melting flakes speed up toward the speed of their drop as they collapse (Mitra et al. 1990)
        v_snow = snow_speed(np.maximum(self.d0, d), rho_air)
        v_snow = v_snow + (v_rain - v_snow) * np.maximum(f, self.fmax) ** 2
        v_gr = graupel_speed(d, rho_air)
        v_gr = v_gr + (v_rain - v_gr) * f ** 2
        v_sph = sphere_speed(d, np.where(f < 0.5, RHO_I, RHO_W), rho_air)
        v = np.where(self.kind == SNOW, v_snow, np.where(self.kind == GRAUPEL, v_gr, v_sph))
        return np.where(self.ice > 0.0, v, v_rain)

    def capacitance(self):
        """Electrostatic capacitance analogue (m) that sets each one's exchange of heat and vapour:
        aggregates of thin branches have less than a sphere of the same size."""
        d = self.diameter() * 1e-3
        fluffy = (self.kind == SNOW) & (self.ice > 0.0) & (np.maximum(self.melted, self.fmax) < 0.5)
        return np.where(fluffy, 0.3 * d, 0.5 * d)


def ventilation(d_mm, v, t_c, p):
    """Ventilation factor of falling particles (Pruppacher and Klett): 0.78 + 0.308 Sc^1/3 Re^1/2."""
    rho = air_density(t_c, p)
    nu = viscosity(t_c) / rho
    re = np.maximum(v * d_mm * 1e-3 / nu, 0.0)
    sc = nu / diffusivity(t_c, p)
    return 0.78 + 0.308 * sc ** (1.0 / 3.0) * np.sqrt(re)


# Bigg (1953) immersion freezing of supercooled water: rate per volume B (exp(-a T) - 1)
BIGG_B = 100.0          # 1/(m^3 s)
BIGG_A = 0.66           # 1/K


def exchange(p: Particles, ta, rv_air, pa, dt, rng=None):
    """Advance particles' heat and water by dt in air at ta (C) with vapour density rv_air (kg/m^3) and
    pressure pa (scalars or arrays): conduction and the latent heat of evaporation or sublimation, both
    ventilated, melting, freezing (supercooled water freezes once it holds ice or a nucleus finds it,
    Bigg 1953), and the mass gained or lost to the vapour. The temperature of a particle with one phase is
    stepped implicitly about its heat balance (the balance linearised: stable at any step). In place."""
    m = p.mass
    live = m > 0.0
    rho = air_density(ta, pa)
    v = p.speed(rho)
    d = p.diameter()
    f = ventilation(d, v, ta, pa)
    C = p.capacitance()
    k = conductivity(ta)
    dv = diffusivity(ta, pa)
    has_ice = p.ice > 0.0
    wet = p.water > 0.0
    mixed = has_ice & wet
    # the surface: at 0 while ice and water coexist, else the particle's own temperature
    ts = np.where(mixed, 0.0, p.t)
    over_ice = has_ice & ~wet
    rvs = np.where(over_ice, rho_vs(ts, True), rho_vs(ts, False))
    L = np.where(over_ice, ls(ts), lv(ts))
    A = 4.0 * math.pi * C * k * f                      # W/K, conduction
    B = 4.0 * math.pi * C * dv * f                     # m^3/s, vapour
    # d(rho_vs)/dT by Clausius-Clapeyron, for the linearised balance
    tk = ts + KELVIN
    s = rvs * (L / (RV * tk * tk) - 1.0 / tk)
    heat0 = A * (ta - ts) + L * B * (rv_air - rvs)     # W at the surface temperature ts
    cm = p.ice * CI + p.water * CW
    # one phase: implicit step of its temperature toward the balance
    K = A + L * B * s
    t1 = ts + (heat0 * dt) / np.maximum(cm + K * dt, 1e-30)
    t1 = np.where(mixed, 0.0, t1)
    # its vapour exchange at the new temperature
    rvs1 = np.where(mixed, rvs, rvs + s * (t1 - ts))
    dm = B * (rv_air - rvs1) * dt
    p.water = np.where(wet, np.maximum(p.water + dm, 0.0), p.water)
    p.ice = np.where(~wet, np.maximum(p.ice + dm, 0.0), p.ice)
    # ice and water together: melting or freezing at 0 by the heat it takes in or gives off
    q = np.where(mixed, heat0 * dt, 0.0)
    melt = np.where(q > 0.0, np.minimum(q / LF, p.ice), 0.0)
    frz = np.where(q < 0.0, np.minimum(-q / LF, p.water), 0.0)
    p.ice = p.ice - melt + frz
    p.water = p.water + melt - frz
    # ice warmed past freezing: the excess heat melts it
    over = has_ice & ~wet & (t1 > 0.0)
    melt2 = np.where(over, np.minimum(t1 * p.ice * CI / LF, p.ice), 0.0)
    p.ice = p.ice - melt2
    p.water = p.water + melt2
    t1 = np.where(over, 0.0, t1)
    p.t = np.where(live, t1, p.t)
    # supercooled water freezes once it holds ice or meets a nucleus (Bigg)
    sc = (p.water > 0.0) & (p.t < 0.0)
    if np.any(sc & (p.ice <= 0.0) & ~p.nucleated):
        vol = p.water / RHO_W
        rate = BIGG_B * vol * (np.exp(-BIGG_A * np.minimum(p.t, 0.0)) - 1.0)
        u = rng.random(len(p)) if rng is not None else np.full(len(p), 0.5)
        hit = sc & (p.ice <= 0.0) & ~p.nucleated & (u < 1.0 - np.exp(-rate * dt))
        p.nucleated = p.nucleated | hit
    can = sc & ((p.ice > 0.0) | p.nucleated)
    if np.any(can):
        # what freezes at once warms it back to 0 with its latent heat (the rest freezes as it loses heat)
        cm2 = p.ice * CI + p.water * CW
        q2 = np.minimum(p.water, cm2 * (-p.t) / LF)
        p.water = np.where(can, p.water - q2, p.water)
        p.ice = np.where(can, p.ice + q2, p.ice)
        p.t = np.where(can & (p.water > 0.0), 0.0, p.t)
    p.fmax = np.maximum(p.fmax, p.melted)
    # a flake that melted well into a drop and froze again is a pellet of ice: sleet
    refrozen = (p.kind == SNOW) & (p.fmax > 0.4) & (p.melted < 0.05) & (p.ice > 0.0)
    p.kind = np.where(refrozen, PELLET, p.kind)
    # melted right through, it is a drop: it has no ice left to freeze onto, so below freezing again it
    # supercools until a nucleus finds it (freezing rain), where a partly melted one refreezes (sleet)
    gone = (p.ice <= 0.0) & (p.water > 0.0) & (p.kind != RAIN)
    p.nucleated = np.where(gone, False, p.nucleated)
    p.kind = np.where(gone, RAIN, p.kind)
    return p


def fall_column(snd: Sounding, p: Particles, z_top, z_bottom=0.0, rng=None, dt=0.5):
    """Follow particles down the column from z_top (m) to z_bottom, all at once in steps of dt seconds
    (each stops as it reaches the bottom). Returns (particles, seconds each took)."""
    n = len(p)
    z = np.full(n, float(z_top))
    secs = np.zeros(n)
    going = np.ones(n, bool)
    for _ in range(200000):
        if not going.any():
            break
        ta = snd.temp(z)
        pa = snd.pressure(z)
        rv = snd.vapour(z)
        v = np.maximum(p.speed(air_density(ta, pa)), 0.05)
        # the last step lands exactly on the bottom
        h = np.where(going, np.minimum(dt, (z - z_bottom) / v), 0.0)
        before = (p.ice.copy(), p.water.copy(), p.t.copy(), p.kind.copy(), p.nucleated.copy(), p.fmax.copy())
        exchange(p, ta, rv, pa, np.maximum(h, 1e-6), rng)
        # those that have landed keep their state
        p.ice = np.where(going, p.ice, before[0])
        p.water = np.where(going, p.water, before[1])
        p.t = np.where(going, p.t, before[2])
        p.kind = np.where(going, p.kind, before[3])
        p.nucleated = np.where(going, p.nucleated, before[4])
        p.fmax = np.where(going, p.fmax, before[5])
        z = z - v * h
        secs += h
        going = going & (z > z_bottom + 1e-6) & (p.mass > 1e-15)
    return p, secs


# -- what forms aloft ------------------------------------------------------------------------------

def size_distribution(kind, rate_mm_h, size_mm=None):
    """(diameters in mm, number density per m^3 per mm) of what forms aloft at a precipitation rate
    (mm/h of water): Marshall-Palmer drops, Gunn-Marshall snow, exponential graupel and hail. size_mm,
    if given, sets the typical size (the median by mass)."""
    r = max(rate_mm_h, 0.01)
    if kind == RAIN:
        lam = 4.1 * r ** -0.21                      # 1/mm
        if size_mm:
            lam = 3.67 / size_mm                    # median volume diameter = 3.67 / lambda
        d = np.linspace(0.5, min(8.0, 12.0 / lam), 96)     # (smaller drops hold little water and are not seen)
        return d, 8000.0 * np.exp(-lam * d)
    if kind == SNOW:
        # Gunn and Marshall measured melted diameters; aggregates are about three times the drop they melt
        # into, so their slope is a third
        lam = 2.55 * r ** -0.48 / 3.0
        if size_mm:
            lam = 3.67 / size_mm
        d = np.linspace(1.0, min(40.0, 12.0 / lam), 96)
        return d, 3800.0 * r ** -0.87 * np.exp(-lam * d)
    if kind == GRAUPEL:
        lam = 3.67 / (size_mm or 3.0)
        d = np.linspace(0.5, 12.0 / lam, 64)
        return d, 4000.0 * np.exp(-lam * d)
    if kind == HAIL:
        lam = 3.67 / (size_mm or 15.0)
        d = np.linspace(5.0, max(12.0 / lam, 10.0), 64)
        return d, 1.0 * np.exp(-lam * d)
    raise ValueError(kind)


def form(kind, d_mm, t_c):
    """Particles of one kind as they form aloft, d_mm across (an array): dry snow, graupel or hail at the
    air's temperature (below freezing), or drops (which have met no nucleus yet)."""
    d = np.array(d_mm, np.float64, ndmin=1)
    n = len(d)
    t = np.full(n, min(float(t_c), 0.0) if kind != RAIN else float(t_c))
    if kind == RAIN:
        return Particles(np.full(n, RAIN), np.zeros(n), sphere_mass(d, RHO_W), t, d, np.zeros(n, bool))
    mass = {SNOW: snow_mass, GRAUPEL: graupel_mass, HAIL: lambda x: sphere_mass(x, RHO_I)}[kind](d)
    return Particles(np.full(n, kind), mass, np.zeros(n), t, d, np.ones(n, bool))


def source_height(snd: Sounding, kind):
    """Height (m) the precipitation starts falling from: snow and graupel from where it is -12 C (the
    dendrite growth zone), hail from -20 C (high in the storm), rain from the cloud base (1.5 km, or 1 km
    over the freezing level when that is higher)."""
    if kind == SNOW or kind == GRAUPEL:
        return snd.level_of(-12.0, 300.0, 9000.0)
    if kind == HAIL:
        return snd.level_of(-20.0, 1000.0, 12000.0)
    fz = snd.freezing_level()
    if not math.isfinite(fz):
        return 1500.0
    return max(1500.0, fz + 1000.0) if fz > 0.0 else 1500.0


@dataclass
class Arrivals:
    """What a spread of sizes formed aloft becomes at the bottom of the column: the sizes aloft (mm), the
    particles as they arrive, the seconds each took, and a summary (share of the arriving mass of each
    kind, the share lost to evaporation or sublimation on the way, the share arriving as supercooled
    drops: freezing rain)."""
    d_aloft: np.ndarray
    particles: Particles
    seconds: np.ndarray
    summary: dict


def arrivals(snd: Sounding, kind, rate_mm_h, size_mm=None, z_bottom=0.0, seed=1, samples=64, dt=0.5):
    """Follow `samples` sizes of what forms aloft (spread evenly through the number falling per second, so
    each stands for an equal share of the particles) down to z_bottom."""
    rng = np.random.default_rng(seed)
    d, n = size_distribution(kind, rate_mm_h, size_mm)
    z0 = source_height(snd, kind)
    t0 = float(snd.temp(z0))
    rho0 = air_density(t0, float(snd.pressure(z0)))
    flux = n * form(kind, d, t0).speed(rho0)
    cdf = np.cumsum(flux)
    cdf /= cdf[-1]
    sizes = np.interp((np.arange(samples) + 0.5) / samples, cdf, d)
    p = form(kind, sizes, t0)
    m_in = p.mass.sum()
    p, secs = fall_column(snd, p, z0, z_bottom, rng=rng, dt=dt)
    k = p.arriving_kind()
    m = p.mass
    total = m.sum()
    summary = {name: (float(m[k == kk].sum() / total) if total > 0 else 0.0) for kk, name in KIND_NAMES.items()}
    summary['lost'] = float(1.0 - total / m_in) if m_in > 0 else 0.0
    summary['freezing_rain'] = float(m[(k == RAIN) & (p.t < 0.0)].sum() / total) if total > 0 else 0.0
    summary['wet_snow'] = float(m[(k == SNOW) & (p.melted > 0.1)].sum() / total) if total > 0 else 0.0
    summary['source_height'] = z0
    return Arrivals(sizes, p, secs, summary)
