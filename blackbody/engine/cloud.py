"""Clouds: moist convection in a box of sky kilometres across (scene kind 'cloud').

The air is a perturbation on a background column at rest (a sounding: the temperature, humidity and wind
at every height, in hydrostatic balance). Thermals rise from the sun-warmed ground (or a warm bubble starts
a storm); rising air cools as it expands, and where it passes its condensation level its vapour condenses
into cloud, whose latent heat keeps it rising: cumulus, which in an unstable afternoon air mass towers into
cumulonimbus, spreads its anvil under the tropopause, and rains, snows and hails (cloud_micro.wgsl).

  cloud_aux.wgsl     thermals from the ground (source), spin (curl), the start (init, init_vel)
  cloud_adv.wgsl     the fields ride the wind (semi-Lagrangian with a MacCormack correction)
  cloud_advv.wgsl    the wind rides itself
  cloud_micro.wgsl   condensation, rain, ice, snow, graupel and hail, melting (bulk microphysics)
  cloud_fall.wgsl    precipitation falls through the air, and reaches the ground
  cloud_force.wgsl   buoyancy (with the weight of the water), vorticity confinement, the sponge, the sides
  divergence, mg_*, project (the fire solver's): the air stays incompressible

Lengths in the scene are scaled by `scale` (metres of sky per metre of scene: 1000 makes the scene's metres
kilometres); everything here is in the sky's own metres and seconds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import atmos as A
from .gpu import Uniforms
from .solver import pack_colliders

FIELD_FMT = 'rgba32float'
VDEF = {'VELFMT': 'rgba32float'}


@dataclass
class CloudParams:
    # the air column (sounding)
    surface_t: float = 28.0         # C, the air at the ground
    surface_rh: float = 0.65        # relative humidity at the ground
    mixed_layer: float = 1200.0     # m: the afternoon's well-mixed layer (dry adiabatic)
    lapse: float = 6.8              # K/km above it
    tropopause: float = 11000.0     # m: above it the air is stable (the anvil spreads under it)
    free_rh: float = 0.55           # humidity above the mixed layer
    inversion_z: float = 0.0        # m: a lid of warmer air (0: none): caps cumulus at its height
    inversion_dt: float = 0.0       # K warmer across it
    # wind
    wind: float = 4.0               # m/s at the ground
    wind_dir: float = 0.0           # degrees (toward +x at 0)
    shear: float = 2.0              # m/s more per km of height (up to the tropopause)
    veer: float = 0.0               # degrees the wind turns from the ground to the tropopause
    follow: bool = False            # the box moves with the storm (the mean wind of the lowest 6 km taken off)
    # what drives it
    heat_flux: float = 300.0        # W/m^2 of sensible heat from the sunlit ground
    evaporation: float = 150.0      # W/m^2 of latent heat (moisture) from the ground
    thermal_size: float = 1500.0    # m: the patches of warmer ground
    patchy: float = 0.7             # 0..1: how uneven the ground's heating is
    bubble: float = 0.0             # K: a warm bubble at the start (0: none)
    bubble_r: float = 5000.0        # m, its radius
    # microphysics
    rain_threshold: float = 1.0e-3  # kg/kg of cloud water before it starts to rain (maritime 0.5e-3)
    rain_rate: float = 1.0e-3       # 1/s, autoconversion
    hail: float = 1.0               # fall speed factor of graupel and hail (2-4 for big hail)
    glaciation: float = 900.0       # s for mixed cloud to turn to ice
    condensation: bool = True
    # numerics
    vorticity: float = 0.0           # vorticity confinement (strong values make the air unstable)
    sponge: float = 0.25            # share of the box's height under the lid that damps
    side_relax: float = 1.0 / 120.0 # 1/s, toward the background wind at the open sides
    seed: int = 0


def base_state(prm: CloudParams, ny: int, h: float):
    """The background column per level (cell centres): (theta, Exner, density, vapour) and (wind x, wind z,
    temperature K, pressure Pa), as an (ny * 2, 4) float32 array, plus the arrays themselves."""
    z = (np.arange(ny) + 0.5) * h
    zz = np.linspace(0.0, max(z[-1] + h, 20000.0), 4001)
    t = np.empty_like(zz)
    ml = prm.mixed_layer
    tml = prm.surface_t - A.DRY_LAPSE * ml
    for i, x in enumerate(zz):
        if x <= ml:
            t[i] = prm.surface_t - A.DRY_LAPSE * x
        elif x <= prm.tropopause:
            t[i] = tml - prm.lapse * 1e-3 * (x - ml)
        else:
            t[i] = tml - prm.lapse * 1e-3 * (prm.tropopause - ml) + 1.0e-3 * (x - prm.tropopause)
    if prm.inversion_z > 0.0 and prm.inversion_dt != 0.0:
        t = t + prm.inversion_dt * np.clip((zz - prm.inversion_z) / 150.0, 0.0, 1.0)
    rh = np.where(zz <= ml, prm.surface_rh - 0.05 * zz / max(ml, 1.0), prm.free_rh)
    rh = np.where(zz > prm.tropopause, 0.05, rh)
    if prm.inversion_z > 0.0:
        rh = np.where(zz > prm.inversion_z + 100.0, np.minimum(rh, prm.free_rh * 0.6), rh)
    snd = A.Sounding(z=list(zz), t=list(t), rh=list(np.clip(rh, 0.01, 1.0)))
    tc = snd.temp(z)
    p = snd.pressure(z)
    tk = tc + A.KELVIN
    exner = (p / 1.0e5) ** (A.RD / A.CP)
    theta = tk / exner
    e = np.clip(snd.humidity(z), 0.0, 1.0) * A.es_water(tc)
    qv = A.EPS * e / (p - e)
    # the mixed layer is well mixed in its vapour as in its heat: the ground's mixing ratio all the way up
    # (its humidity rising toward its top), kept just under saturation. (Humidity held instead would leave
    # moist air under dry, lighter under heavier: a column that overturns by itself.)
    e0 = min(max(prm.surface_rh, 0.0), 1.0) * float(A.es_water(prm.surface_t))
    p0 = float(snd.pressure(0.0))
    q_sfc = A.EPS * e0 / (p0 - e0)
    qsat = A.EPS * A.es_water(tc) / (p - A.es_water(tc))
    qv = np.where(z <= ml, np.minimum(q_sfc, 0.97 * qsat), qv)
    rho = p / (A.RD * tk * (1.0 + 0.61 * qv))
    # the background wind: speed shear and veering up to the tropopause, constant above
    frac = np.clip(z / max(prm.tropopause, 1.0), 0.0, 1.0)
    speed = prm.wind + prm.shear * np.minimum(z, prm.tropopause) * 1e-3
    ang = math.radians(prm.wind_dir) + math.radians(prm.veer) * frac
    ux, uz = speed * np.cos(ang), speed * np.sin(ang)
    if prm.follow:
        # the box moves with the storm, as research models' do: storms move about with the mean wind of the
        # lowest 6 km, so that is taken off the wind at every height (the ground slides under the box instead)
        sel = z <= 6000.0
        wt = rho[sel] if sel.any() else rho[:1]
        ux = ux - float(np.average(ux[sel] if sel.any() else ux[:1], weights=wt))
        uz = uz - float(np.average(uz[sel] if sel.any() else uz[:1], weights=wt))
    out = np.zeros((ny * 2, 4), np.float32)
    out[0::2] = np.stack([theta, exner, rho, qv], 1)
    out[1::2] = np.stack([ux, uz, tk, p], 1)
    return out, dict(z=z, t=tc, p=p, theta=theta, rho=rho, qv=qv, rh=snd.humidity(z), wind=(ux, uz), sounding=snd)


def parcel(prm: CloudParams, h=50.0, top=16000.0):
    """Lift a surface parcel (a diagnostic): its condensation level, level of free convection, the top it
    could reach, and its CAPE (J/kg): what kind of cloud the sounding allows."""
    ny = int(top / h)
    st, d = base_state(prm, ny, h)
    z, tenv, p = d['z'], d['t'], d['p']
    q = float(d['qv'][0])
    t = float(tenv[0])
    lcl = lfc = el = None
    cape = 0.0
    saturated = False
    for j in range(1, ny):
        dz = z[j] - z[j - 1]
        if not saturated:
            t -= A.DRY_LAPSE * dz
            e = A.es_water(t)
            if q >= A.EPS * e / (p[j] - e):
                saturated = True
                lcl = float(z[j])
        else:
            t -= A.moist_lapse(t, p[j]) * dz
        tv = (t + A.KELVIN) * (1 + 0.61 * (min(q, A.EPS * A.es_water(t) / (p[j] - A.es_water(t)))))
        tve = (tenv[j] + A.KELVIN) * (1 + 0.61 * d['qv'][j])
        b = A.G * (tv - tve) / tve
        if saturated and b > 0.0:
            cape += b * dz
            if lfc is None:
                lfc = float(z[j])
            el = float(z[j])
    return dict(lcl=lcl, lfc=lfc, top=el, cape=cape)


class CloudSolver:
    """The cloud simulation on the GPU."""

    def __init__(self, gpu, meshes=None):
        self.gpu = gpu
        self.meshes = meshes
        self.dims = None
        self.h = 0.0
        self.origin = (0.0, 0.0, 0.0)
        self.scale = 1000.0
        self.time = 0.0
        self.steps = 0
        self.colliders = []
        self._k = {}
        self._base_key = None
        self.stats = {}

    # -- set-up ------------------------------------------------------------------------------------

    def _kernels(self):
        if self._k:
            return
        g = self.gpu
        k = self._k
        adv = ['utex3d', 'utex3d', 'utex3d', 'st3d:rgba32float:w', 'rbuf']
        for e in ('sl_a', 'sl_b', 'mc_a', 'mc_b'):
            k['adv_' + e] = g.kernel('cloud_adv.wgsl', adv, e)
        k['advv_sl'] = g.kernel('cloud_advv.wgsl', adv, 'sl', VDEF)
        k['advv_mc'] = g.kernel('cloud_advv.wgsl', adv, 'mc', VDEF)
        k['micro'] = g.kernel('cloud_micro.wgsl', ['utex3d', 'utex3d', 'st3d:rgba32float:w', 'st3d:rgba32float:w', 'rbuf'])
        k['fall'] = g.kernel('cloud_fall.wgsl', ['utex3d', 'utex3d', 'st3d:rgba32float:w', 'st3d:rgba32float:w', 'rbuf',
                                                 'st2d:rgba32float:w'])
        k['force'] = g.kernel('cloud_force.wgsl', ['utex3d'] * 5 + ['st3d:rgba32float:w', 'rbuf'], 'main', VDEF)
        aux = ['utex3d', 'utex3d', 'st3d:rgba32float:w', 'rbuf']
        for e in ('curl', 'source', 'init', 'init_vel'):
            k[e] = g.kernel('cloud_aux.wgsl', aux, e)
        k['stats'] = g.kernel('cloud_stats.wgsl', ['utex3d', 'utex3d', 'utex3d', 'buf', 'rbuf'])
        # the fire solver's incompressibility (kernels shared)
        k['div'] = g.kernel('divergence.wgsl', ['utex3d', 'utex3d', 'utex3d', 'st3d:r32float:w'])
        k['smooth'] = g.kernel('mg_smooth.wgsl', ['st3d:r32float:rw', 'utex3d', 'utex3d'])
        k['residual'] = g.kernel('mg_residual.wgsl', ['utex3d', 'utex3d', 'utex3d', 'st3d:r32float:w'])
        k['restrict'] = g.kernel('mg_restrict.wgsl', ['utex3d', 'st3d:r32float:w', 'st3d:r32float:w'])
        k['prolong'] = g.kernel('mg_prolong.wgsl', ['st3d:r32float:rw', 'utex3d', 'utex3d'])
        k['project'] = g.kernel('project.wgsl', ['utex3d', 'utex3d', 'utex3d', 'st3d:rgba32float:w'], 'main', VDEF)
        k['sdf'] = g.kernel('sdf.wgsl', ['utex3d', 'st3d:r32float:w'])
        k['fill_r'] = g.kernel('fill.wgsl', ['st3d:r32float:w'], 'main', {'FMT': 'r32float'})
        k['fill_4'] = g.kernel('fill.wgsl', ['st3d:rgba32float:w'], 'main', {'FMT': 'rgba32float'})

    def _t3(self, size, fmt, label):
        return self.gpu.texture3d(size, fmt, 'cloud-' + label)

    def configure(self, dims, h_scene, origin_scene, scale):
        """Allocate for dims cells of h_scene (scene metres) at origin_scene, scale metres of sky per scene
        metre. True if anything changed (then the sky is reset)."""
        self._kernels()
        dims = tuple(int(x) for x in dims)
        self.scale = float(scale)
        changed = dims != self.dims
        self.h = h_scene * self.scale
        self.origin = tuple(float(o) * self.scale for o in origin_scene)
        self.h_scene = h_scene
        self.origin_scene = tuple(origin_scene)
        if changed:
            self._release()
            self.dims = dims
            nx, ny, nz = dims
            vd = (nx + 1, ny + 1, nz + 1)
            self.V = [self._t3(vd, FIELD_FMT, f'v{i}') for i in range(3)]
            self.A = [self._t3(dims, FIELD_FMT, f'a{i}') for i in range(3)]
            self.B = [self._t3(dims, FIELD_FMT, f'b{i}') for i in range(3)]
            self.CURL = self._t3(dims, FIELD_FMT, 'curl')
            self.SDF = self._t3(dims, 'r32float', 'sdf')
            self.EXPO = self._t3(dims, 'r32float', 'expansion')
            self.PRECIP = self.gpu.texture2d(nx, nz, 'rgba32float', 'cloud-precip')
            self.levels = [dims]
            while True:
                d = self.levels[-1]
                if max(d) <= 8 or min(d) <= 3:
                    break
                self.levels.append(tuple((x + 1) // 2 for x in d))
            self.P = [self._t3(d, 'r32float', f'p{i}') for i, d in enumerate(self.levels)]
            self.RHS = [self._t3(d, 'r32float', f'rhs{i}') for i, d in enumerate(self.levels)]
            self.RES = [self._t3(d, 'r32float', f'res{i}') for i, d in enumerate(self.levels[:-1])]
            self.base = self.gpu.buffer(ny * 2 * 16, 'cloud-base')
            self.stats_buf = self.gpu.buffer(64, 'cloud-stats')
            self._base_key = None
        return changed

    def _release(self):
        for name in ('V', 'A', 'B', 'P', 'RHS', 'RES'):
            for t in getattr(self, name, None) or []:
                t.destroy()
        for name in ('CURL', 'SDF', 'EXPO', 'PRECIP', 'base', 'stats_buf'):
            t = getattr(self, name, None)
            if t is not None:
                t.destroy()

    def set_colliders(self, colliders):
        self.colliders = list(colliders)

    def _grid(self, dt, scene_frame=False):
        nx, ny, nz = self.dims
        h = self.h_scene if scene_frame else self.h
        o = self.origin_scene if scene_frame else self.origin
        # open sides, lid on top, the ground
        return Uniforms().v4(nx, ny, nz, h).v4(*o, self.time).v4(1.0, 0.0, 0.0, dt)

    def _u(self, dt, prm: CloudParams, marker=0.0):
        u = self._grid(dt)
        u.v4(self.time, self.steps * 7 + prm.seed * 131, prm.sponge * self.dims[1], prm.side_relax)
        u.v4(prm.rain_threshold, prm.rain_rate, prm.hail, prm.glaciation)
        heat = prm.heat_flux
        evap = prm.evaporation / A.lv(prm.surface_t)
        if marker < 0.0:
            u.v4(heat, evap, prm.bubble_r, prm.bubble)
        else:
            u.v4(heat, evap, prm.thermal_size, prm.patchy)
        u.v4(prm.vorticity, 0.0, marker, 1.0 if prm.condensation else 0.0)
        return u

    def upload_base(self, prm: CloudParams):
        key = (tuple(sorted(vars(prm).items())), self.dims, round(self.h, 3))
        if key == self._base_key:
            return
        arr, info = base_state(prm, self.dims[1], self.h)
        self.gpu.write_buffer(self.base, arr)
        self.base_info = info
        self._base_key = key

    def reset(self, prm: CloudParams):
        """The sky at rest in its background state (with a warm bubble if asked)."""
        self.upload_base(prm)
        self.time = 0.0
        self.steps = 0
        g = self.gpu
        k = self._k
        with g.batch() as b:
            u = self._u(0.0, prm, marker=-1.0 if prm.bubble > 0.0 else 0.0)
            b.run(k['init'], [self.V[0], self.A[0], self.A[1], self.base], u, self.dims)
            b.run(k['init_vel'], [self.V[0], self.A[0], self.V[2], self.base], u, tuple(x + 1 for x in self.dims))
            b.run(k['init_vel'], [self.V[2], self.A[0], self.V[0], self.base], u, tuple(x + 1 for x in self.dims))
            for t in (self.B[0], self.B[1], self.CURL):
                b.run(k['fill_4'], [t], Uniforms().v4(*t.size, 0).v4(0.0, 0.0, 0.0, 0.0), t.size)
            for t in self.P + [self.EXPO]:
                b.run(k['fill_r'], [t], Uniforms().v4(*t.size, 0).v4(0.0, 0.0, 0.0, 0.0), t.size)
            self._write_sdf(b)
        # (init wrote A[1]: make it the current)
        self.A[0], self.A[1] = self.A[1], self.A[0]

    def _write_sdf(self, b):
        u = pack_colliders(self._grid(0.0, scene_frame=True), self.colliders, self.meshes)
        b.run(self._k['sdf'], [self.meshes.atlas if self.meshes is not None else self.SDF, self.SDF], u, self.dims)

    # -- the step ----------------------------------------------------------------------------------

    def _mg(self, lvl, parity=0):
        d = self.levels[lvl]
        nc = self.levels[lvl + 1] if lvl + 1 < len(self.levels) else d
        return (Uniforms().v4(*d, parity).v4(1.0, 0.0, 0.0, 1.0 if (lvl == 0 and self.colliders) else 0.0)
                .v4(1.0).v4(*nc, 0))

    def _smooth(self, b, lvl, parity):
        d = self.levels[lvl]
        b.run(self._k['smooth'], [self.P[lvl], self.RHS[lvl], self.SDF], self._mg(lvl, parity), ((d[0] + 1) // 2, d[1], d[2]))

    def _vcycle(self, b, lvl):
        last = len(self.levels) - 1
        if lvl == last:
            for _ in range(24):
                self._smooth(b, lvl, 0)
                self._smooth(b, lvl, 1)
            return
        for _ in range(2):
            self._smooth(b, lvl, 0)
            self._smooth(b, lvl, 1)
        k = self._k
        b.run(k['residual'], [self.P[lvl], self.RHS[lvl], self.SDF, self.RES[lvl]], self._mg(lvl), self.levels[lvl])
        b.run(k['restrict'], [self.RES[lvl], self.RHS[lvl + 1], self.P[lvl + 1]], self._mg(lvl), self.levels[lvl + 1])
        self._vcycle(b, lvl + 1)
        b.run(k['prolong'], [self.P[lvl], self.P[lvl + 1], self.SDF], self._mg(lvl), self.levels[lvl])
        for _ in range(2):
            self._smooth(b, lvl, 1)
            self._smooth(b, lvl, 0)

    def step(self, b, dt, prm: CloudParams):
        """One substep of dt seconds of sky time."""
        self.upload_base(prm)
        k = self._k
        n = self.dims
        vd = tuple(x + 1 for x in n)
        u = self._u(dt, prm)
        A, B, V = self.A, self.B, self.V
        base = self.base
        # the ground's heat and moisture
        b.run(k['source'], [V[0], A[0], A[1], base], u, n)
        A[0], A[1] = A[1], A[0]
        # everything rides the wind
        b.run(k['adv_sl_a'], [V[0], A[0], A[0], A[1], base], u, n)
        b.run(k['adv_mc_a'], [V[0], A[0], A[1], A[2], base], u, n)
        b.run(k['adv_sl_b'], [V[0], B[0], B[0], B[1], base], u, n)
        b.run(k['adv_mc_b'], [V[0], B[0], B[1], B[2], base], u, n)
        b.run(k['advv_sl'], [V[0], V[0], V[0], V[1], base], u, vd)
        b.run(k['advv_mc'], [V[0], V[0], V[1], V[2], base], u, vd)
        A[0], A[2] = A[2], A[0]
        B[0], B[2] = B[2], B[0]
        V[0], V[2] = V[2], V[0]
        # the water changes phase, and falls
        b.run(k['micro'], [A[0], B[0], A[1], B[1], base], u, n)
        b.run(k['fall'], [A[1], B[1], A[0], B[0], base, self.PRECIP], u, n)
        # forces
        b.run(k['curl'], [V[0], A[0], self.CURL, base], u, n)
        b.run(k['force'], [V[0], A[0], B[0], self.CURL, self.SDF, V[1], base], u, vd)
        V[0], V[1] = V[1], V[0]
        # the air stays incompressible
        g = self._grid(dt)
        b.run(k['div'], [V[0], self.EXPO, self.SDF, self.RHS[0]], g, n)
        for _ in range(2):
            self._vcycle(b, 0)
        b.run(k['project'], [V[0], self.P[0], self.SDF, V[1]], g, vd)
        V[0], V[1] = V[1], V[0]
        self.time += dt
        self.steps += 1

    def substeps_for(self, frame_dt, cfl=1.0, hi=8):
        vmax = max(float(self.stats.get('max_speed', 20.0)), 5.0)
        return int(min(hi, max(1, math.ceil(vmax * frame_dt / (cfl * self.h)))))

    def measure(self):
        """Read back: fastest wind (m/s), strongest updraft, cloud top (m), cloud cover (share of columns),
        most cloud water and ice, and precipitation at the ground (mm/h, its peak and mean)."""
        g = self.gpu
        g.write_buffer(self.stats_buf, np.zeros(16, np.uint32))
        with g.batch() as b:
            b.run(self._k['stats'], [self.V[0], self.A[0], self.B[0], self.stats_buf, self.base],
                  self._grid(0.0), (self.dims[0], 1, self.dims[2]))
        raw = np.frombuffer(g.read_buffer(self.stats_buf), np.uint32)
        f = np.frombuffer(raw.tobytes(), np.float32)
        nx, ny, nz = self.dims
        pr = g.read(self.PRECIP)
        pr = pr[0] if pr.ndim == 4 else pr
        rate = pr[..., :3].sum(axis=-1) * 3600.0     # mm/h of water
        self.stats = dict(max_speed=float(f[0]), max_updraft=float(f[1]), cloud_top=float(f[2]),
                          cover=float(raw[3]) / (nx * nz), max_cloud_water=float(f[4]) * 1e3, max_ice=float(f[5]) * 1e3,
                          max_hail=float(f[6]) * 1e3, rain_peak=float(rate.max()), rain_mean=float(rate.mean()))
        return self.stats

    def read_fields(self):
        """(A, B) as (nz, ny, nx, 4) float32 arrays."""
        a = self.gpu.read(self.A[0])
        b = self.gpu.read(self.B[0])
        return a, b
