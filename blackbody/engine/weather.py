"""Weather: precipitation simulated as particles over the scene (snow, sleet, freezing rain, graupel, hail,
and rain where it is made by melting), and what it leaves on the ground and in the water.

What falls into the shot is decided by the sky above it (engine/atmos.py): each size of what forms in the
cloud (snow, graupel, hail, or drops) is followed down through the column of air, whose temperatures and
humidity melt it, refreeze it or evaporate it on the way, and the states that reach the top of the weather
area are the table new particles are drawn from. Inside the area each particle carries on doing the same
with the air round it (wx_common.wgsl), falls through the wind, gusts and eddies, and lands:

  wx_step.wgsl    spawns, moves and lands the particles (snow settles, hail bounces and rests, drops wet
                  or glaze), and hands what reaches the liquid to it
  wx_surf.wgsl    the surface the cover lies on, seen from above (the ground, the colliders' tops)
  wx_cover.wgsl   the snowpack, granular ice and glaze on it: falls, settles, melts, drains
  wx_liquid.wgsl  precipitation that reached the liquid becomes liquid, carrying its heat
  wx_pack.wgsl    the live particles packed for the renderer and the cache

A scene falls the same way every run: new pieces take the free slots in order (each block of 64 slots' free ones,
counted as a step ends, turned into where each block's first goes: block_scan.wgsl), each drawn from its number among
the step's new pieces, and the pack follows the slots. (Slots raced for with an atomic counter went to whichever threads
ran first, and seeded by their slot, which hailstones fell changed from run to run.) Hail's splashes are the exception:
they go into the liquid's spray ring in the order the threads reach it (wx_step.wgsl splash).

The weather area is a box over the scene (the liquid's box and a margin round it, from the ground up to
`top`); it wraps round at its sides. Coordinates are the simulation's own (metres, y up, ground at 0).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import atmos as A
from .gpu import BU, Uniforms, groups_1d
from .solver import MAX_COLLIDERS, pack_colliders

KINDS = ('none', 'snow', 'rain', 'sleet', 'freezing_rain', 'graupel', 'hail', 'sky')
PART_BYTES = 64
PACK_BYTES = 48
LUT_ENTRIES = 256
FX_UG = 1.0e9
SPAWN_LOG = 256       # steps a frame whose spawned count is read back (how many found a free slot)


@dataclass
class WeatherParams:
    kind: str = 'none'            # what falls (KINDS); 'sky' takes it from the temperatures aloft
    rate: float = 2.0             # mm/h of water (snow's melted equivalent)
    size: float = 0.0             # mm, typical size of what forms (0: what the rate gives); hail's stones
    ground_t: float = -2.0        # C, the air at the ground
    humidity: float = 0.9         # relative humidity of the air (over ice below freezing)
    warm_t: float = 2.0           # C, warmest air aloft ('sky': a layer of it melts the snow on the way)
    warm_z: float = 1500.0        # m, its height
    wind: tuple = (0.0, 0.0, 0.0)  # m/s
    gust: float = 0.3             # 0..1, how much the wind comes and goes
    turbulence: float = 0.4       # m/s of eddies
    eddy: float = 1.5             # m, their size
    area: tuple = (-3.0, -3.0, 3.0, 3.0)   # x0, z0, x1, z1 (m)
    top: float = 6.0              # m above the ground particles start
    ground: bool = True
    ground_temp: float = 0.0      # C, of the ground (and collider tops without their own)
    collider_temps: tuple = ()
    cover: bool = True            # keep what lands (snow, glaze) on the surfaces
    cover_cell: float = 0.02      # m, the cover map's cell
    start: float = 0.0            # s, when it begins
    lying: float = 0.0            # m of snow already on the ground at the start
    buildup: float = 1.0          # how many times faster what lands builds up (a time-lapse of the cover)
    heat_speed: float = 1.0       # the liquid's heat speed-up (the cover melts with it)
    freeze_point: float = 0.0
    bounce_hail: float = 0.45
    bounce_grain: float = 0.2
    capacity: int = 2_000_000
    seed: int = 0


def kind_column(prm: WeatherParams):
    """(what forms aloft, the air column, the sizes' typical size) for a weather kind. Sleet and freezing
    rain are snow falling through a layer of warmer air aloft: a weak one melts it partly and the cold air
    below refreezes it into ice pellets; a strong one melts it into drops that stay liquid below freezing
    until they hit something. Whether that is what arrives depends on the air at the ground: above
    freezing it is rain."""
    g = prm.ground_t
    rh = min(max(prm.humidity, 0.05), 1.0)
    k = prm.kind
    if k == 'snow':
        return A.SNOW, A.winter_sounding(g, rh=rh), prm.size or None
    if k == 'sleet':
        return A.SNOW, A.winter_sounding(g, 1.0, 1200.0, rh=rh), prm.size or None
    if k == 'freezing_rain':
        return A.SNOW, A.winter_sounding(g, 4.0, 1000.0, rh=rh), prm.size or None
    if k == 'graupel':
        return A.GRAUPEL, A.winter_sounding(g, rh=rh), prm.size or 3.0
    if k == 'hail':
        return A.HAIL, A.convective_sounding(g, rh=rh), prm.size or 15.0
    if k == 'rain':
        return A.RAIN, A.convective_sounding(g, rh=rh), prm.size or None
    # from the sky's temperatures: snow aloft, through the warm layer given
    return A.SNOW, A.winter_sounding(g, prm.warm_t, prm.warm_z, rh=rh), prm.size or None


@dataclass
class Column:
    """What reaches the top of the weather area: a table of particle states (LUT_ENTRIES x 8: kind, ice,
    water, temperature, dry size, nucleated, most melted, fall speed), the mean mass of one (kg), the
    number in each cubic metre of falling air per (kg/m^2/s) of rate, and the share of the mass arriving
    as each kind."""
    lut: np.ndarray
    mean_mass: float
    density_per_flux: float
    summary: dict


_COLUMNS = {}


def column(prm: WeatherParams, z_bottom: float) -> Column | None:
    """The column's arrivals for these settings (cached)."""
    if prm.kind == 'none':
        return None
    kind, snd, size = kind_column(prm)
    key = (prm.kind, round(prm.rate, 3), size, round(prm.ground_t, 2), round(prm.humidity, 3), round(prm.warm_t, 2),
           round(prm.warm_z), round(z_bottom, 1), prm.seed)
    hit = _COLUMNS.get(key)
    if hit is not None:
        return hit
    a = A.arrivals(snd, kind, max(prm.rate, 0.01), size, z_bottom=z_bottom, seed=prm.seed + 1, samples=LUT_ENTRIES,
                   dt=1.0)
    p = a.particles
    m = p.mass
    keep = m > 1e-13
    if not keep.any():
        col = Column(np.zeros((0, 8), np.float32), 0.0, 0.0, a.summary)
        _COLUMNS[key] = col
        return col
    t_bot = float(snd.temp(z_bottom))
    rho = A.air_density(t_bot, float(snd.pressure(z_bottom)))
    v = p.speed(rho)
    lut = np.stack([p.kind, p.ice, p.water, p.t, p.d0, p.nucleated.astype(np.float64), p.fmax, v], axis=1)[keep]
    mean_mass = float(m[keep].mean())
    # number per m^3 in steady fall: each arriving particle spends 1/v seconds per metre of height
    density_per_flux = float(np.mean(1.0 / np.maximum(v[keep], 0.05))) / mean_mass
    col = Column(lut.astype(np.float32), mean_mass, density_per_flux, a.summary)
    if len(_COLUMNS) > 64:
        _COLUMNS.clear()
    _COLUMNS[key] = col
    return col


class Weather:
    """The precipitation particles, the cover map and the hand-over to a liquid, on the GPU."""

    def __init__(self, gpu, meshes=None):
        self.gpu = gpu
        self.meshes = meshes
        self.capacity = 0
        self.parts = None
        self.packed = None
        # slots handed out in order (the same every run): each block of 64 slots' free ones (counted as each step ends),
        # and where each block's first new (or packed) piece goes
        self.cnt = None
        self.off = None
        self.packed_count = 0
        self.map_dims = None
        self._k = {}
        self._spawn_acc = 0.0
        self._filled = False
        self.time = 0.0
        self.steps = 0
        self.stats = {}
        self._lut_key = None
        self.col = None
        self._liq_dims = None
        self.liq_dep = None
        self._dummy3 = None
        self._dummy_buf = None
        self._asked = []      # this frame's steps: (pieces asked for, the fill at the start)
        self._fresh = 0.0     # pieces the rate asked for this frame (what a shortfall carried on may come to)
        self.short = 0        # pieces that found no free slot since the start: the particle limit was full
        self.fill_short = 0   # and those the air should hold at the start past what the limit leaves room for

    # -- set-up ------------------------------------------------------------------------------------

    @staticmethod
    def _blocks(cap):
        return -(-int(cap) // 64)

    def _starts(self, b, live):
        """Where each block of 64 slots' first new piece goes (its free slots: the step's), or its first packed one (its
        live ones: the pack's), from the free slots the last step counted (block_scan.wgsl)."""
        u = Uniforms().v4(self._blocks(self.capacity), 64 if live else 0, self.capacity)
        b.run(self._k['scan'], [self.cnt, self.off], u, groups=(1, 1, 1))

    def _kernels(self):
        if self._k:
            return
        g = self.gpu
        P = (64, 1, 1)
        self._k['step'] = g.kernel('wx_step.wgsl', ['buf', 'rbuf', 'buf', 'buf', 'buf', 'utex3d', 'utex3d', 'utex2d',
                                                    'buf', 'buf', 'buf', 'rbuf', 'buf'], workgroup=P)
        self._k['surf'] = g.kernel('wx_surf.wgsl', ['utex3d', 'utex3d', 'st2d:rgba32float:w'], workgroup=(8, 8, 1))
        self._k['cover'] = g.kernel('wx_cover.wgsl', ['buf', 'buf', 'buf', 'utex2d', 'st2d:rgba32float:w'],
                                    workgroup=(8, 8, 1))
        self._k['liquid'] = g.kernel('wx_liquid.wgsl', ['buf'] * 6)
        self._k['pack'] = g.kernel('wx_pack.wgsl', ['rbuf', 'buf', 'buf', 'buf'], workgroup=P)
        self._k['count'] = g.kernel('wx_pack.wgsl', ['rbuf', 'buf', 'buf', 'buf'], 'count', workgroup=P)
        self._k['scan'] = g.kernel('block_scan.wgsl', ['rbuf', 'buf'], workgroup=(256, 1, 1))

    def configure(self, prm: WeatherParams):
        """Allocate for these settings (particle capacity, cover map size). True if anything changed."""
        self._kernels()
        g = self.gpu
        x0, z0, x1, z1 = prm.area
        nx = max(8, min(2048, int(math.ceil((x1 - x0) / max(prm.cover_cell, 1e-3)))))
        nz = max(8, min(2048, int(math.ceil((z1 - z0) / max(prm.cover_cell, 1e-3)))))
        changed = False
        cap = int(max(1024, prm.capacity))
        if cap != self.capacity:
            for b in (self.parts, self.packed, self.cnt, self.off):
                if b is not None:
                    b.destroy()
            self.parts = g.buffer(cap * PART_BYTES, 'wx-particles')
            self.packed = g.buffer(cap * PACK_BYTES, 'wx-packed')
            nb = self._blocks(cap)
            self.cnt = g.buffer(-(-nb // 4) * 16, 'wx-free-counts')     # (read four blocks at a time)
            self.off = g.buffer((nb + 1) * 4, 'wx-block-starts')
            self.capacity = cap
            changed = True
        if (nx, nz) != self.map_dims:
            for name in ('cover_dep', 'cover', 'wet'):
                b = getattr(self, name, None)
                if b is not None:
                    b.destroy()
            for name in ('SURF', 'COVER'):
                t = getattr(self, name, None)
                if t is not None:
                    t.destroy()
            self.cover_dep = g.buffer(nx * nz * 16, 'wx-cover-deposits')
            self.cover = g.buffer(nx * nz * 16, 'wx-cover')
            self.wet = g.buffer(nx * nz * 4, 'wx-wet')
            self.SURF = g.texture2d(nx, nz, 'rgba32float', 'wx-surface')
            self.COVER = g.texture2d(nx, nz, 'rgba32float', 'wx-cover-tex')
            self.map_dims = (nx, nz)
            changed = True
        if getattr(self, 'lut', None) is None:
            self.lut = g.buffer(LUT_ENTRIES * 32, 'wx-lut')
            self.ctr = g.buffer(16 * 4, 'wx-counters')
            self.spawn_log = g.buffer(SPAWN_LOG * 4, 'wx-spawn-log')
            self._dummy_buf = g.buffer(64, 'wx-dummy')
            self._dummy_buf2 = g.buffer(64, 'wx-dummy2')
            self._dummy_buf3 = g.buffer(64, 'wx-dummy3')
            self._dummy_buf4 = g.buffer(64, 'wx-dummy4')
            self._dummy3 = g.texture3d((1, 1, 1), 'r32float', 'wx-dummy3')
        self.area = tuple(prm.area)
        if changed:
            self.reset()
        return changed

    def reset(self):
        """Empty the sky and the ground."""
        if self.parts is None:
            return
        g = self.gpu
        init = np.zeros((self.capacity, 16), np.float32)
        init[:, 3] = -1.0
        g.write_buffer(self.parts, init)
        with g.batch() as b:
            for buf in (self.cover_dep, self.cover, self.wet, self.ctr):
                b.clear_buffer(buf)
            if self.liq_dep is not None:
                b.clear_buffer(self.liq_dep)
            # (the free slots counted afresh: each step counts them as it ends)
            b.run(self._k['count'], [self.parts, self.packed, self.ctr, self.cnt], Uniforms().v4(self.capacity),
                  groups=groups_1d(self.capacity))
        self.packed_count = 0
        self._spawn_acc = 0.0
        self._asked = []
        self._fresh = 0.0
        self.short = 0
        self.fill_short = 0
        self._filled = False
        self._lying_done = False
        self.time = 0.0
        self.steps = 0
        self._surf_key = None
        self.stats = {}

    def _upload_column(self, prm: WeatherParams):
        col = column(prm, prm.top)
        self.col = col
        if col is None or not len(col.lut):
            return
        key = id(col)
        if key == self._lut_key:
            return
        lut = np.zeros((LUT_ENTRIES, 8), np.float32)
        n = len(col.lut)
        lut[:n] = col.lut
        self._lut_n = n
        self.gpu.write_buffer(self.lut, lut)
        self._lut_key = key

    def _liquid(self, b, L):
        """Deposit buffer for the liquid's cells (cleared in batch b: never open a batch inside another, it
        reuses the outer one's uniforms)."""
        if L is None or L.dims is None:
            return None
        n = tuple(L.dims)
        if n != self._liq_dims:
            if self.liq_dep is not None:
                self.liq_dep.destroy()
            self.liq_dep = self.gpu.buffer(int(np.prod(n)) * 16, 'wx-liquid-deposits')
            b.clear_buffer(self.liq_dep)
            self._liq_dims = n
        return self.liq_dep

    def _grid(self, L, dt):
        if L is not None and L.dims is not None:
            u = L._grid(dt)
            return u
        return Uniforms().v4(1, 1, 1, 1.0).v4(0.0, 0.0, 0.0, self.time).v4(0.0, 1.0, 0.0, dt)

    def surface(self, b, prm: WeatherParams, L=None):
        """Find the surface the cover lies on (after the colliders change)."""
        nx, nz = self.map_dims
        u = self._grid(L, 0.0)
        u.v4(*prm.area).v4(nx, nz, 1.0 if prm.ground else 0.0, prm.ground_temp)
        on = L is not None and L.dims is not None
        u.v4(prm.top + 1.0, 0.0, 0.0, 0.0)
        pack_colliders(u, L.colliders if on else [], self.meshes)
        temps = list(prm.collider_temps)[:MAX_COLLIDERS]
        temps += [prm.ground_temp] * (MAX_COLLIDERS - len(temps))
        for i in range(0, MAX_COLLIDERS, 4):
            u.v4(*temps[i:i + 4])
        sdf = L.SDF if on else self._dummy3
        atlas = self.meshes.atlas if self.meshes is not None else self._dummy3
        b.run(self._k['surf'], [sdf, atlas, self.SURF], u, (nx, nz, 1))

    # -- the step ----------------------------------------------------------------------------------

    def step(self, b, dt, prm: WeatherParams, L=None):
        """Record one substep: particles in, moved, landed; the cover; what reached the liquid."""
        if prm.kind == 'none' or self.parts is None:
            return
        self._upload_column(prm)
        col = self.col
        nx, nz = self.map_dims
        x0, z0, x1, z1 = prm.area
        area_m2 = (x1 - x0) * (z1 - z0)
        rate = max(prm.rate, 0.0) / 3600.0          # kg/m^2/s
        active = self.time >= prm.start and col is not None and len(col.lut) > 0
        spawn = 0
        fill = 0.0
        if active:
            per_s = rate * area_m2 / max(col.mean_mass, 1e-15)
            self._spawn_acc += per_s * dt
            if not self._filled and prm.start <= 0.0:
                # falling all along: the air already full of it at the start (a steady fall), not a front
                want = col.density_per_flux * rate * area_m2 * prm.top
                spawn = int(min(want, self.capacity * 0.9))
                self.fill_short = max(int(want) - spawn, 0)     # (more than the limit holds: said, Engine.notices)
                self._filled = True
                fill = 1.0
                self._spawn_acc = 0.0
            else:
                self._fresh += per_s * dt
                spawn = int(self._spawn_acc)
                self._spawn_acc -= spawn
        on = L is not None and L.dims is not None
        dep = self._liquid(b, L) if on else None
        wind = prm.wind
        surf_dens = 0.0
        if on:
            surf_dens = L._prm.surface_density * L._prm.ppc
        u = self._grid(L, dt)
        u.v4(*prm.area).v4(prm.top, 1.0 if prm.ground else 0.0, nx, nz)
        u.v4(prm.ground_t, 6.5e-3, min(max(prm.humidity, 0.0), 1.0), A.P0)
        u.v4(*wind, prm.gust).v4(prm.turbulence, prm.eddy, self.time, self.steps * 31 + prm.seed * 7919)
        u.v4(spawn, getattr(self, '_lut_n', 1), surf_dens, 1.0 if on else 0.0)
        u.v4(prm.bounce_hail, prm.bounce_grain, 60.0, fill)
        mp = (L._prm.rho * L.h ** 3 / max(L._prm.ppc, 1)) if on else 0.0
        vmin = float(col.lut[:, 7].min()) if (col is not None and len(col.lut)) else 1.0
        ww = on and getattr(L, 'ww_capacity', 0) > 64 and L._prm.whitewater
        u.v4(prm.freeze_point, mp, vmin, L.ww_capacity if ww else 0.0)
        u.v4(L._prm.foam_life * 0.15 if ww else 0.0, 0.0, 0.0, 0.0)
        b.clear_buffer(self.ctr, 0, 2 * 4)        # spawned and alive: this step's (the landings add up)
        sdf = L.SDF if on else self._dummy3
        dens = L.DENS if on else self._dummy3
        wa, wb, wc = (L.WA, L.WB, L.wctr) if ww else (self._dummy_buf, self._dummy_buf2, self._dummy_buf3)
        if spawn > 0:
            self._starts(b, live=False)
        b.run(self._k['step'], [self.parts, self.lut, self.ctr, self.cover_dep, dep if dep is not None else self._dummy_buf4,
                                 sdf, dens, self.SURF, wa, wb, wc, self.off, self.cnt], u, groups=groups_1d(self.capacity))
        if spawn > 0 and len(self._asked) < SPAWN_LOG:
            # how many of them found a free slot, read back with the frame's counters (measure)
            b.copy_buffer(self.ctr, self.spawn_log, 0, 4 * len(self._asked), 4)
            self._asked.append((spawn, fill > 0.0))
        # the cover
        lying = prm.lying if not self._lying_done else 0.0
        self._lying_done = True
        cu = (Uniforms().v4(nx, nz, (x1 - x0) * (z1 - z0) / (nx * nz), dt)
              .v4(prm.ground_t, math.hypot(wind[0], wind[2]), prm.heat_speed * max(prm.buildup, 1.0), 0.0)
              .v4(0.0, prm.ground_t, max(prm.buildup, 1.0), lying))
        if prm.cover:
            b.run(self._k['cover'], [self.cover_dep, self.cover, self.wet, self.SURF, self.COVER], cu, (nx, nz, 1))
        else:
            b.clear_buffer(self.cover_dep)
        # what reached the liquid becomes liquid
        if on:
            th = L.thermal if (L.thermal is not None and L._prm.thermal and L.thermal.therm is not None) else None
            carry = getattr(L, 'attr', None) is not None and getattr(L, '_dye_on', False)
            lu = L._grid(dt).v4(mp * FX_UG, L.capacity, 1.0 if th is not None else 0.0, self.steps * 13 + 1)
            lu.v4(1.0 if carry else 0.0, 0.0, 0.0, 0.0)
            b.run(self._k['liquid'], [self.liq_dep, L.parts, L.ctr, L.freelist,
                                      th.therm if th is not None else self._dummy_buf2,
                                      L.attr if carry else self._dummy_buf3], lu, tuple(L.dims))
        self.time += dt
        self.steps += 1

    def pack(self, b):
        """Pack the live particles (for the renderer and the cache), in the order of their slots."""
        if self.parts is None:
            return
        b.clear_buffer(self.ctr, 15 * 4, 4)
        self._starts(b, live=True)
        b.run(self._k['pack'], [self.parts, self.packed, self.ctr, self.off], Uniforms().v4(self.capacity),
              groups=groups_1d(self.capacity))

    def measure(self):
        """Read this frame's counters: particles alive, packed, landed by kind, fell into the liquid."""
        if self.parts is None:
            return {}
        raw = np.frombuffer(self.gpu.read_buffer(self.ctr), np.uint32).copy()
        with self.gpu.batch() as b:
            b.clear_buffer(self.ctr, 2 * 4, 13 * 4)
        self.packed_count = int(min(raw[15], self.capacity))
        landed = {A.KIND_NAMES[k]: int(raw[2 + k]) for k in range(5)}
        missed = self._missed()
        self.stats = {'alive': int(raw[1]), 'packed': self.packed_count, 'landed': landed, 'into_liquid': int(raw[8]),
                      'missed': missed, 'short': self.short}
        if self.col is not None:
            self.stats['arriving'] = {k: round(v, 3) for k, v in self.col.summary.items()
                                      if k in ('rain', 'snow', 'graupel', 'hail', 'ice pellets', 'freezing_rain', 'wet_snow')
                                      and v > 0.001}
        return self.stats

    def _missed(self):
        """The pieces this frame's steps asked for that found no free slot (the particle limit full). They are carried on
        into the next frame's (at most a frame's worth of the rate, so a limit that stays full builds no backlog); what is
        not is lost, and counted in `short`."""
        asked, self._asked = self._asked, []
        fresh, self._fresh = self._fresh, 0.0
        if not asked:
            return 0
        got = np.frombuffer(self.gpu.read_buffer(self.spawn_log, 4 * len(asked)), np.uint32)
        missed = carry = 0
        for (want, fill), n in zip(asked, got):
            m = max(want - min(int(n), want), 0)
            missed += m
            if not fill:
                carry += m
        kept = int(min(carry, fresh))
        self._spawn_acc += kept
        self.short += missed - kept
        return missed

    # -- reading back ----------------------------------------------------------------------------

    def read_packed(self):
        """(n, 12) float32: position, size; velocity, phase; kind, melted, resting, temperature."""
        if not self.packed_count:
            return np.zeros((0, 12), np.float32)
        data = self.gpu.read_buffer(self.packed, self.packed_count * PACK_BYTES)
        return np.frombuffer(data, np.float32).reshape(-1, 12).copy()

    def haze(self, prm: WeatherParams):
        """The fall seen past the area, as a haze: (snow weight, rain weight, extinction (1/m), top (m)). The
        extinction is the pieces' number per cubic metre times twice their cross-section (big particles take
        twice the light their shadow does)."""
        col = self.col
        if col is None or not len(col.lut) or prm.kind == 'none':
            return (0.0, 0.0, 0.0, 0.0)
        lut = col.lut
        rate = max(prm.rate, 0.0) / 3600.0
        n = col.density_per_flux * rate                        # pieces per m^3
        # their sizes now, as the pack kernel gives them (snow: its aggregate size)
        p = A.Particles(lut[:, 0].astype(np.int32), lut[:, 1], lut[:, 2], lut[:, 3], lut[:, 4], lut[:, 5] > 0.5, lut[:, 6])
        d = p.diameter() * 1e-3
        w = 1.0 / np.maximum(lut[:, 7], 0.05)                  # as many of each as hang in the air
        # a snowflake aggregate is open and branched: it shadows about 40 % of its circle
        fill = np.where((p.kind == A.SNOW) & (p.ice > 0.0), 0.4, 1.0)
        area = float(np.sum(w * fill * np.pi * d * d / 4.0) / np.sum(w))
        sigma = 2.0 * n * area
        frozen = float(np.mean(lut[:, 1] > lut[:, 2]))
        return (frozen, 1.0 - frozen, sigma, 1500.0)

    def read_cover(self):
        """(nz, nx, 4) float32: snow depth (m), water equivalent (kg/m^2), wetness, glaze (m)."""
        if self.COVER is None:
            return None
        a = self.gpu.read(self.COVER)
        return a[0] if a.ndim == 4 else a

    def read_surface(self):
        a = self.gpu.read(self.SURF)
        return a[0] if a.ndim == 4 else a
