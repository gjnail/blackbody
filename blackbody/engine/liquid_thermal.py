"""Heat and phase changes of a liquid (water): it warms and cools, freezes into ice that moves as solid
pieces, melts, boils in bubbles and evaporates, with the real heat capacities and latent heats.

Every particle carries its enthalpy (liq_therm_common.wgsl). Each substep, alongside the liquid solver
(LiquidSolver.step calls these in order):

  heat()        after the particles are splatted: their heat per cell (liq_therm_p2g / _norm), then what
                each cell gains or loses from the air, the ground, colliders at their own temperature, fire
                gas and lava in the box, and its neighbours (liq_therm_heat), and the buoyancy that heat,
                ice and steam bubbles give the liquid (liq_therm_buoy), which the forces apply.
  ice_rigid()   after the pressure solve: connected frozen cells become rigid pieces, and the velocity in
                the free ones is made rigid (liq_ice).
  hide()        before the particles move: frozen particles in a piece are set aside (liq_ice_parts).
  after_move()  after they move: frozen ones move with their piece (liq_ice_parts), then every particle
                takes its heat and changes phase: seeding and freezing, melting, boiling into steam
                bubbles, evaporating (liq_therm_g2p).
  steam()       after the whitewater moves: steam bubbles rise, condense or burst (liq_steam), and what
                the liquid gave the gas this substep is written for the fire solver (gas_flux).

Temperatures are in C. With fire in the box (LiquidSolver.gas set to the fire Solver) the air the
liquid meets is the gas (its temperature and water vapour); otherwise the scene's air.
"""
from __future__ import annotations

import math

import numpy as np

from .gpu import Uniforms, groups_1d
from .solver import MAX_COLLIDERS, MAX_EMITTERS, pack_colliders, pack_emitters

MAX_BODIES = 8192          # matches liq_therm_common.wgsl
BODY_WORDS = 36
STATS = 12
LF = 333.55                # kJ/kg
CW = 4.18
CI = 2.05


def enthalpy(t_c, freeze=0.0, frozen=False):
    """Specific enthalpy (kJ/kg) of water at t_c C, measured from ice at the freezing point."""
    if frozen and t_c < freeze:
        return CI * (t_c - freeze)
    return LF + CW * (t_c - freeze)


def temperature(h, freeze=0.0, seeded=True):
    """Temperature (C) of water of enthalpy h (kJ/kg)."""
    if not seeded:
        return freeze + (h - LF) / CW
    if h < 0.0:
        return freeze + h / CI
    if h <= LF:
        return freeze
    return freeze + (h - LF) / CW


def vapour_sat(t_c, over_ice=False):
    """Water vapour in saturated air at t_c C (g/m^3), as liq_therm_common.wgsl."""
    t = min(max(float(t_c), -80.0), 374.0)
    if over_ice and t < 0.0:
        e = 6.112 * math.exp(22.46 * t / (t + 272.62))
    else:
        e = 6.112 * math.exp(17.67 * t / (t + 243.5))
    return e * 216.7 / (t + 273.15)


def water_density(t_c):
    """Density of liquid water (kg/m^3), as liq_therm_common.wgsl (largest at 4 C)."""
    t = min(max(float(t_c), -30.0), 150.0)
    a = t - 3.983035
    return 999.97495 * (1.0 - a * a * (t + 301.797) / (522528.9 * (t + 69.34881)))


def evaporation_flux(t_surface, t_air, vapour_air, wind=0.0, ice=False):
    """Evaporation (kg per m^2 per second) of a water surface at t_surface into air at t_air holding
    vapour_air g/m^3, with the liquid's speed relative to the air `wind` (m/s), as liq_therm_heat.wgsl."""
    hc = max(1.52 * abs(t_surface - t_air) ** (1.0 / 3.0), 5.7 + 3.8 * wind)
    return hc / 1206.0 * (vapour_sat(t_surface, ice) - vapour_air) * 1e-3


def boil_flux(superheat):
    """Heat flux (W/m^2) from a wall `superheat` K over boiling into water, as liq_therm_heat.wgsl
    (without the film regime's radiation)."""
    d = float(superheat)
    if d <= 3.0:
        return 0.0
    if d <= 20.0:
        return min(1.37e5 * (d / 10.0) ** 3, 1.1e6)
    if d < 110.0:
        s = (d - 20.0) / 90.0
        return math.exp((1 - s) * math.log(1.1e6) + s * math.log(2.0e4))
    return 250.0 * d


class Thermal:
    """The heat of one LiquidSolver (created by it when a scene turns heat on)."""

    def __init__(self, liquid):
        self.L = liquid
        self.gpu = liquid.gpu
        self.therm = None
        self._k = {}
        self._dims = None
        self._cap = 0
        self._gas_dims = None
        self.gas_flux = None
        self.stats = {}
        self.ice_seen = False
        self.bodies_on = False
        self._cur = 0
        self._stats_clear = True

    # -- resources -----------------------------------------------------------------------------

    def _kernels(self):
        if self._k:
            return self._k
        g = self.gpu
        P = (64, 1, 1)
        k = self._k
        k['p2g'] = g.kernel('liq_therm_p2g.wgsl', ['rbuf', 'buf', 'buf', 'utex3d', 'utex3d'], workgroup=P)
        k['born'] = g.kernel('liq_therm_p2g.wgsl', ['rbuf', 'buf', 'buf', 'utex3d', 'utex3d'], 'born', workgroup=P)
        k['norm'] = g.kernel('liq_therm_norm.wgsl', ['rbuf', 'utex3d', 'utex3d', 'st3d:rgba32float:w'])
        k['heat'] = g.kernel('liq_therm_heat.wgsl', ['utex3d'] * 8 + ['st3d:rgba32float:w', 'buf'])
        k['buoy'] = g.kernel('liq_therm_buoy.wgsl', ['utex3d', 'utex3d', 'rbuf', 'st3d:r32float:w'])
        ice = ['utex3d', 'utex3d', 'utex3d', 'utex3d', 'st3d:rgba32float:w', 'buf', 'buf', 'buf', 'buf']
        for e in ('label_init', 'label_union', 'label_flatten', 'label_roots', 'label_assign', 'body_sums', 'body_faces'):
            k[e] = g.kernel('liq_ice.wgsl', ice, e)
        k['body_solve'] = g.kernel('liq_ice.wgsl', ice, 'body_solve', workgroup=P)
        parts = ['buf', 'rbuf', 'rbuf', 'rbuf', 'utex3d', 'buf', 'buf', 'buf']
        k['hide'] = g.kernel('liq_ice_parts.wgsl', parts, 'hide', workgroup=P)
        k['move'] = g.kernel('liq_ice_parts.wgsl', parts, 'move_ice', workgroup=P)
        k['g2p'] = g.kernel('liq_therm_g2p.wgsl', ['buf'] * 8 + ['utex3d'] * 4 + ['buf'], workgroup=P)
        k['steam'] = g.kernel('liq_steam.wgsl', ['buf'] * 4 + ['utex3d'] * 4, workgroup=P)
        k['gas'] = g.kernel('liq_therm_gas.wgsl', ['rbuf', 'st3d:rgba32float:w'])
        return k

    def _ensure(self, b):
        L = self.L
        g = self.gpu
        n = tuple(L.dims)
        if self._dims != n or self._cap != L.capacity:
            self.release()
            cells = int(np.prod(n))
            self.therm = L._b(L.capacity * 8, 'liq-therm')
            self.pice = L._b(L.capacity * 4, 'liq-packed-ice')
            self.T = [L._t3(n, 'rgba32float', f'liq-therm-cells{i}') for i in range(2)]
            self.DH = L._t3(n, 'rgba32float', 'liq-therm-change')
            self.TB = L._t3(n, 'r32float', 'liq-therm-buoyancy')
            self.tacc = L._b(cells * 3 * 4, 'liq-therm-p2g')
            self.lab = L._b(cells * 4, 'liq-ice-labels')
            self.bodycell = L._b(cells * 4, 'liq-ice-bodies-per-cell')
            self.sums = L._b((MAX_BODIES * BODY_WORDS + 4) * 4, 'liq-ice-sums')
            self.bodies = L._b(MAX_BODIES * 3 * 16, 'liq-ice-bodies')
            self._zero = L._t3((1, 1, 1), 'rgba32float', 'liq-therm-zero')
            self._dims = n
            self._cap = L.capacity
            self._gas_dims = None
            for t in self.T:
                L.fill(b, t)
            L.fill(b, self.TB)
            L.fill(b, self._zero)
            b.clear_buffer(self.therm)
            b.clear_buffer(self.bodycell)
            self._cur = 0
            self._stats_clear = True
        gdims = tuple(L.gas.dims) if L.gas is not None and L.gas.dims else n
        if gdims != self._gas_dims:
            nf, nl = int(np.prod(gdims)), int(np.prod(n))
            if getattr(self, 'gacc', None) is not None:
                self.gacc.destroy()
                self.gas_flux.destroy()
            self.gacc = g.buffer((3 * nf + 2 * nl + STATS) * 4, 'liq-therm-gas-acc')
            self.gas_flux = g.texture3d(gdims, 'rgba32float', 'liq-therm-gas-flux')
            b.clear_buffer(self.gacc)
            L.fill(b, self.gas_flux)
            self._gas_dims = gdims
            self._nf, self._nl = nf, nl
            self._stats_clear = True

    def release(self):
        for name in ('gacc', 'gas_flux'):
            x = getattr(self, name, None)
            if x is not None:
                x.destroy()
            setattr(self, name, None)
        self.therm = None
        self._dims = None
        self._gas_dims = None

    def reset(self, b):
        """Called from LiquidSolver.reset (the liquid's own buffers are released and cleared there)."""
        if self._dims is None:
            return
        for t in self.T:
            self.L.fill(b, t)
        self.L.fill(b, self.TB)
        b.clear_buffer(self.therm)
        b.clear_buffer(self.bodycell)
        if self.gacc is not None:
            b.clear_buffer(self.gacc)
            self.L.fill(b, self.gas_flux)
        self._cur = 0
        self.stats = {}
        self.ice_seen = False
        self._stats_clear = True

    # -- uniforms ------------------------------------------------------------------------------

    def _therm_u(self, u, prm):
        gas = self.L.gas
        span_air, span_flame = prm.gas_air_k, prm.gas_flame_k
        air_t, rh = prm.air_temp, prm.humidity
        air_vapour = max(0.0, min(rh, 100.0)) / 100.0 * vapour_sat(air_t, air_t < 0.0)
        ref = self.stats.get('mean_temp', prm.temp)
        u.v4(prm.freeze_point, prm.boil_point, max(prm.supercool, 0.0), max(prm.heat_speed, 1e-3))
        u.v4(air_t, air_vapour, prm.ground_temp, prm.temp)
        u.v4(float(prm.ppc), prm.rho, 1.0 if gas is not None else 0.0, 1.0 if self.L.lava_heat is not None else 0.0)
        u.v4(span_air, span_flame, ref, prm.bubble_size)
        return u

    def _col_u(self, u, prm):
        L = self.L
        pack_colliders(u, L.colliders, L.meshes)
        temps = list(prm.collider_temps)[:MAX_COLLIDERS]
        temps += [prm.ground_temp] * (MAX_COLLIDERS - len(temps))
        for i in range(0, MAX_COLLIDERS, 4):
            u.v4(*temps[i:i + 4])
        return u

    def _bounds(self, prm):
        """Enthalpy (kJ/kg) of the coldest and the hottest thing the liquid can meet: the air, the ground,
        the colliders, its sources, fire gas and lava in the box."""
        temps = [prm.air_temp, prm.ground_temp, prm.temp, prm.min_source_temp] + list(prm.collider_temps)
        if self.L.gas is not None:
            temps.append(prm.gas_flame_k * 1.6 - 273.15)
        if self.L.lava_heat is not None:
            temps.append(1500.0)
        lo, hi = min(temps), max(temps)
        return enthalpy(lo, prm.freeze_point, frozen=True), enthalpy(max(hi, prm.boil_point + 1.0), prm.freeze_point)

    def _cold(self, prm):
        """Anything in the scene at or below freezing (so ice can form), or ice already in it."""
        tf = prm.freeze_point
        temps = [prm.air_temp, prm.ground_temp, prm.temp] + list(prm.collider_temps)
        return self.ice_seen or min(temps) <= tf + 0.5 or prm.min_source_temp <= tf + 0.5

    # -- the substep ----------------------------------------------------------------------------

    @property
    def cells(self):
        return self.T[self._cur]

    def heat(self, b, prm, dt, srcs, attr_buoy=None):
        """Heat per cell and this step's change; returns the buoyancy texture for the forces."""
        L = self.L
        k = self._kernels()
        self._ensure(b)
        n = L.dims
        gb = L._grid(dt, prm).tobytes()
        # the statistics are the last step's, but the vapour given off adds up until it is read
        b.clear_buffer(self.gacc, (3 * self._nf + 2 * self._nl) * 4, 7 * 4)
        if self._stats_clear:
            b.clear_buffer(self.gacc, (3 * self._nf + 2 * self._nl + 7) * 4, (STATS - 7) * 4)
            self._stats_clear = False
        # gas regions and the bubbles' volume are per step (the collapse heat is read here, cleared after)
        b.clear_buffer(self.gacc, 0, 3 * self._nf * 4)
        prev, cur = self.T[self._cur], self.T[1 - self._cur]
        b.clear_buffer(self.tacc)
        u = L._grid(dt, prm).v4(L.capacity)
        self._therm_u(u, prm)
        pack_emitters(u, srcs[:MAX_EMITTERS], meshes=L.meshes)
        b.run_indirect(k['p2g'], [L.parts, self.therm, self.tacc, prev, L.meshes.atlas], u, L.args, 0)
        u = L._grid(dt, prm)
        self._therm_u(u, prm)
        b.run(k['norm'], [self.tacc, L.DENS, prev, cur], u, n)
        self._cur = 1 - self._cur
        gas = L.gas
        zero = self._zero
        if gas is not None:
            gas_t = gas.scal[0]
            gas_aux = gas.aux[0] if gas.aux is not None else zero
            gh = gas.h
        else:
            gas_t = gas_aux = zero
            gh = L.h
        lava = L.lava_heat if L.lava_heat is not None else zero
        level = prm.water_level / L.h if L.level_on(prm) else -1.0
        u = L._grid(dt, prm)
        self._therm_u(u, prm)
        u.v4(*self._gas_dims, level).v4(*prm.wind, gh).v4(self._nf, self._nl)
        self._col_u(u, prm)
        b.run(k['heat'], [cur, L.DENS, L.SDF, L.VOLD, L.meshes.atlas, gas_t, gas_aux, lava, self.DH, self.gacc], u, n)
        b.clear_buffer(self.gacc, 3 * self._nf * 4, self._nl * 4)      # the collapse heat, now taken
        u = L._grid(dt, prm)
        self._therm_u(u, prm)
        u.v4(self._nf, self._nl, 1.0 if attr_buoy is not None else 0.0)
        b.run(k['buoy'], [cur, attr_buoy if attr_buoy is not None else L._zero3, self.gacc, self.TB], u, n)
        b.clear_buffer(self.gacc, (3 * self._nf + self._nl) * 4, self._nl * 4)   # bubbles' volume: this step's, next
        return self.TB

    def born(self, b, prm, dt, srcs):
        """Give the particles born this step (sources, reseeds, the open water's strips) their heat."""
        L = self.L
        k = self._kernels()
        self._ensure(b)
        u = L._grid(dt, prm).v4(L.capacity)
        self._therm_u(u, prm)
        pack_emitters(u, list(srcs)[:MAX_EMITTERS], meshes=L.meshes)
        b.run_indirect(k['born'], [L.parts, self.therm, self.tacc, self.cells, L.meshes.atlas], u, L.args, 0)

    def _ice_res(self, prm, vel, spare):
        L = self.L
        u = L._grid(0.0, prm)
        self._therm_u(u, prm)
        u.v4(0.5)
        self._col_u(u, prm)
        return [self.cells, L.SDF, L.meshes.atlas, vel, spare, self.lab, self.bodycell, self.sums, self.bodies], u

    def ice_rigid(self, b, prm, vel, spare):
        """After the pressure solve: the pieces of ice (connected frozen cells), and the velocity in each
        free one made rigid. Returns the velocity texture to carry on with (vel, or spare holding the
        rigid version)."""
        self.bodies_on = self._cold(prm)
        if not self.bodies_on:
            return vel
        k = self._k
        n = self.L.dims
        res, u = self._ice_res(prm, vel, spare)
        b.clear_buffer(self.sums)
        for e in ('label_init', 'label_union', 'label_flatten', 'label_roots', 'label_assign', 'body_sums'):
            b.run(k[e], res, u, n)
        b.run(k['body_solve'], res, u, groups=(MAX_BODIES // 64, 1, 1))
        b.run(k['body_faces'], res, u, tuple(x + 1 for x in n))
        return spare

    def hide(self, b, prm):
        if not self.bodies_on:
            return
        L = self.L
        u = L._grid(0.0, prm).v4(L.capacity, prm.collision_radius)
        b.run_indirect(self._k['hide'], [L.parts, self.therm, self.bodycell, self.bodies, L.SDF, L.ctr, L.freelist,
                                         L.cellcount], u, L.args, 0)

    def after_move(self, b, prm, dt, ww):
        L = self.L
        k = self._k
        if self.bodies_on:
            u = L._grid(dt, prm).v4(L.capacity, prm.collision_radius)
            b.run_indirect(k['move'], [L.parts, self.therm, self.bodycell, self.bodies, L.SDF, L.ctr, L.freelist,
                                       L.cellcount], u, L.args, 0)
        mp = prm.rho * L.h ** 3 / max(prm.ppc, 1)
        u = L._grid(dt, prm).v4(L.capacity, L.ww_capacity if ww else 0.0, L.steps * 13 + 5 + prm.seed * 131.0, mp)
        self._therm_u(u, prm)
        u.v4(self._nf, self._nl, max(0.6 * L.h * 1000.0, 0.5 * prm.bubble_size), prm.wall_drag).v4(*self._gas_dims)
        u.v4(*self._bounds(prm))
        b.run_indirect(k['g2p'], [L.parts, self.therm, L.ctr, L.freelist, L.WA, L.WB, L.wctr, self.gacc,
                                  self.DH, self.cells, L.DENS, L.SDF, L.cellcount], u, L.args, 0)

    def steam(self, b, prm, dt, vel, ww):
        L = self.L
        if ww:
            u = L._grid(dt, prm).v4(L.ww_capacity, prm.ppc, L.steps * 7 + 1 + prm.seed * 131.0, prm.gravity)
            self._therm_u(u, prm)
            u.v4(self._nf, self._nl).v4(*self._gas_dims)
            b.run(self._k['steam'], [L.WA, L.WB, L.wctr, self.gacc, vel, L.DENS, L.SDF, self.cells], u,
                  groups=groups_1d(L.ww_capacity))
        gas = L.gas
        gh = gas.h if gas is not None else L.h
        span = (prm.gas_flame_k - prm.gas_air_k) if gas is not None else 1.0
        u = Uniforms().v4(*self._gas_dims, gh).v4(dt, span, self._nf)
        b.run(self._k['gas'], [self.gacc, self.gas_flux], u, self._gas_dims)

    # -- output --------------------------------------------------------------------------------

    def read_stats(self):
        """Per frame: liquid and frozen cells, mean temperature, the hottest and coldest, boiling cells,
        and the vapour given off (g) since the last read. Clears them for the next frame."""
        if self.gacc is None:
            return {}
        off = (3 * self._nf + 2 * self._nl) * 4
        raw = np.frombuffer(self.gpu.read_buffer(self.gacc, STATS * 4, off), np.int32)
        cells = int(raw[0])
        st = {'liquid_cells': cells, 'ice_cells': int(raw[1]), 'boiling_cells': int(raw[4]),
              'vapour_g': float(raw[7]) / 4194304.0, 'boiled': int(raw[8]), 'evaporated': int(raw[9]),
              'boil_heat_j': float(raw[10]) / 256.0}
        if cells:
            st['mean_temp'] = float(raw[2]) / 2.0 / cells
            st['max_temp'] = float(raw[5]) / 100.0 - 1000.0
            st['min_temp'] = 1000.0 - float(raw[6]) / 100.0
        self.stats = st
        self.ice_seen = st['ice_cells'] > 0
        self._stats_clear = True
        return st
