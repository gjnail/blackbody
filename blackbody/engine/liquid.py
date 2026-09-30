"""Liquid solver: FLIP / APIC particles on a MAC grid, entirely on the GPU.

Per substep:
  1. particles splat their momentum (with the APIC affine part) onto the grid faces
  2. gravity, source nozzles and solid boundaries (moving colliders push the liquid)
  3. velocity extrapolated a couple of layers into the air
  4. free-surface pressure: ghost-fluid boundary at the surface (found from the particle density),
     a volume-preserving correction where particles bunch up, solved by conjugate gradients
     preconditioned with a multigrid V-cycle
  5. the new velocity extrapolated further into the air
  6. particles take the new velocity (PIC/FLIP blend), move through it (midpoint rule), bounce off
     solids and walls, and leave through open sides
  7. sources top up the cells they cover with new particles
  8. whitewater (spray, foam, bubbles) born in step 6 where the liquid is fast and turbulent or
     breaking, then carried along as secondary particles that do not affect the liquid

Viscosity is an implicit diffusion of the grid velocity between steps 3 and 4. With an open water
level the sides below it meet still water (its hydrostatic pressure) instead of air, a layer along
them calms waves, and sources along them keep the water topped up.

Particle slots are recycled through a free stack; the live count stays on the GPU and particle
kernels are dispatched indirectly, so nothing is read back while stepping.

Sources ride the fire emitter layout (EmitterGPU / pack_emitters) so every emitter shape, meshes
included, can pour liquid: fuel = strength (0..1), temperature = mode (0 stream, 1 fill now),
smoke = velocity jitter (fraction of the speed), vel_blend = how firmly the source sets the velocity
of the liquid inside it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .gpu import BU, GPU, Batch, Program, Uniforms, ceil_div, groups_1d
from .solver import EmitterGPU, MAX_COLLIDERS, MAX_EMITTERS, Solver, pack_colliders, pack_emitters

PARTICLE_BYTES = 80
PACKED_BYTES = 16
WW_BYTES = 32
WW_PACKED_BYTES = 16
FLOAT_SUBSTEPS = 16     # substep slots of the floating-object force sums
FLOAT_SLOT = 12         # i32 per body and substep (see liq_float.wgsl)


@dataclass
class LiquidParams:
    gravity: float = 9.81
    flip: float = 0.9               # 0 = PIC (calm, viscous-looking), 1 = FLIP (lively, noisy)
    apic: bool = True
    ppc: int = 8                    # particles per cell
    surface_density: float = 0.4    # fraction of the rest density where the surface lies
    volume_correction: float = 0.3  # per step, fraction of excess density pushed apart
    compression: float = 1.15       # density (x rest) above which the volume correction acts
    theta_min: float = 0.1
    pressure_iters: int = 12
    mg_sweeps: int = 1
    coarse_sweeps: int = 4
    extrapolate: int = 4
    speed_limit: float = 80.0       # m/s
    collision_radius: float = 0.25  # cells
    wall_drag: float = 2.0         # m/s^2 deceleration of liquid touching solids and the ground
    surface_tension: float = 0.0728  # N/m (water); 0 turns it off
    rho: float = 1000.0             # kg/m^3
    drying: float = 0.02            # 1/s, ground wetness
    damping: float = 0.0            # 1/s on every particle (settling still liquid during pre-roll)
    seed: float = 0.0
    whitewater: bool = True
    ww_rate: float = 25.0           # whitewater particles per second per liquid particle, at full potential
    ww_min_speed: float = 1.2       # m/s below which nothing is released
    ww_turbulence: float = 1.0
    ww_crests: float = 1.0
    foam_life: float = 2.5          # s
    bubble_rise: float = 6.0        # m/s^2
    spray_drag: float = 0.3         # 1/s
    bubble_drag: float = 12.0       # 1/s
    viscosity: float = 0.0          # Pa s
    contact_angle: float = 90.0     # degrees
    water_level: float = 0.0        # m, open water past the sides (0 = off)
    level_absorb: float = 8.0       # cells
    wind: tuple = (0.0, 0.0, 0.0)   # m/s, fire-local
    wind_surface: float = 1.0
    narrow_band: bool = False
    band_width: int = 4             # cells
    open_sides: bool = True
    open_top: bool = True
    ground: bool = True


def source(shape='sphere', pos=(0.0, 0.5, 0.0), size=(0.1, 0.1, 0.1), p1=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0),
           radial=0.0, strength=1.0, fill=False, jitter=0.02, vel_blend=1.0, yaw=0.0, mesh='', soft=0.0):
    """A liquid source in the emitter layout the kernels share (see the module docstring)."""
    return EmitterGPU(shape=shape, pos=tuple(pos), size=tuple(size), p1=tuple(p1), soft=soft, fuel=float(strength),
                      temp=1.0 if fill else 0.0, smoke=float(jitter), vel=tuple(vel), radial=float(radial),
                      vel_blend=float(vel_blend), noise=0.0, yaw=float(yaw), mesh=mesh)


class _Recorder:
    """Stands in for a Batch to capture a fixed dispatch sequence into a Program."""

    def __init__(self):
        self.ops = []

    def run(self, kernel, resources, uniforms, size=None, groups=None):
        self.ops.append((kernel, list(resources), uniforms, size, groups))


class LiquidSolver:
    kind = 'liquid'

    def __init__(self, gpu: GPU, meshes=None):
        self.gpu = gpu
        if meshes is None:
            from .mesh import MeshLibrary
            meshes = MeshLibrary(gpu)
        self.meshes = meshes
        self.dims = None
        self.h = 0.0
        self.origin = (0.0, 0.0, 0.0)
        self.capacity = 0
        self.ww_capacity = 0
        self.ww_count = 0
        self.time = 0.0
        self.steps = 0
        self.max_speed = 0.0
        self.bbox = None
        self.count = 0
        self.packed_count = 0
        self.colliders = []
        self.levels = []
        self._tex = []
        self._buf = []
        self._k = {}
        self._prm = LiquidParams()
        self.vel_tex = None
        self._press = None
        self._press_key = None
        self.VVISC = None
        self._level_filled = False

    # -- allocation ---------------------------------------------------------------------------

    dims_for = staticmethod(Solver.dims_for)

    @staticmethod
    def capacity_for(dims, ppc, max_particles):
        cells = int(np.prod(dims))
        return int(ceil_div(min(int(max_particles), cells * int(ppc)), 64) * 64)

    def memory_bytes(self, dims=None, capacity=None, ww_capacity=None):
        nx, ny, nz = dims or self.dims
        cells = nx * ny * nz
        faces = (nx + 1) * (ny + 1) * (nz + 1)
        cap = self.capacity if capacity is None else capacity
        wcap = self.ww_capacity if ww_capacity is None else ww_capacity
        return int(faces * (16 * 3 + 32) + cells * (4 * 11 + 8 * 2 + 4 * 1.3) + cap * (PARTICLE_BYTES + PACKED_BYTES + 4)
                   + wcap * (WW_BYTES + WW_PACKED_BYTES))

    def configure(self, dims, h, origin, capacity, ww_capacity=0):
        dims = tuple(int(x) for x in dims)
        capacity = int(capacity)
        ww_capacity = int(ceil_div(max(int(ww_capacity), 64), 64) * 64)
        if (dims == self.dims and capacity == self.capacity and ww_capacity == self.ww_capacity
                and abs(h - self.h) < 1e-12 and tuple(origin) == tuple(self.origin)):
            return False
        realloc = dims != self.dims or capacity != self.capacity or ww_capacity != self.ww_capacity
        self.h = float(h)
        self.origin = tuple(float(x) for x in origin)
        if realloc:
            self._release()
            self.dims = dims
            self.capacity = capacity
            self.ww_capacity = ww_capacity
            self._allocate()
            self._compile()
        self.reset()
        return True

    def _release(self):
        self.VVISC = None
        if self._press is not None:
            self._press.destroy()
            self._press = None
        for t in self._tex:
            t.destroy()
        for b in self._buf:
            b.destroy()
        self._tex, self._buf = [], []

    def _t3(self, size, fmt, label):
        t = self.gpu.texture3d(size, fmt, label)
        self._tex.append(t)
        return t

    def _b(self, size, label, usage=BU.STORAGE | BU.COPY_DST | BU.COPY_SRC):
        b = self.gpu.buffer(max(16, int(size)), label, usage)
        self._buf.append(b)
        return b

    def _allocate(self):
        n = self.dims
        m = tuple(x + 1 for x in n)
        g = self.gpu
        self.VOLD = self._t3(m, 'rgba32float', 'liq-vold')
        self.VA = self._t3(m, 'rgba32float', 'liq-va')
        self.VB = self._t3(m, 'rgba32float', 'liq-vb')
        self.DENS = self._t3(n, 'r32float', 'liq-density')
        self.SDF = self._t3(n, 'r32float', 'liq-solid')
        self.X = self._t3(n, 'r32float', 'liq-pressure')
        self.R = self._t3(n, 'r32float', 'liq-r')
        self.Z = self._t3(n, 'r32float', 'liq-z')
        self.PP = self._t3(n, 'r32float', 'liq-p')
        self.AP = self._t3(n, 'r32float', 'liq-ap')
        self.NRM = self._t3(n, 'rgba16float', 'liq-normal')
        self.KAPPA = self._t3(n, 'r32float', 'liq-curvature')
        self.levels = [n]
        while True:
            d = self.levels[-1]
            if max(d) <= 8 or min(d) <= 3:
                break
            self.levels.append(tuple((x + 1) // 2 for x in d))
        L = self.levels
        self.TYPE = [self._t3(d, 'r32float', f'liq-type{i}') for i, d in enumerate(L)]
        self.CO = [self._t3(d, 'rg32float', f'liq-co{i}') for i, d in enumerate(L)]
        self.MZ = [self.Z] + [self._t3(d, 'r32float', f'liq-mz{i}') for i, d in enumerate(L) if i > 0]
        self.MR = [self.R] + [self._t3(d, 'r32float', f'liq-mr{i}') for i, d in enumerate(L) if i > 0]
        self.RES = [self._t3(d, 'r32float', f'liq-res{i}') for i, d in enumerate(L[:-1])]
        self.WET = g.texture2d(n[0], n[2], 'r32float', 'liq-wet')
        self._tex.append(self.WET)
        cap = self.capacity
        self.parts = self._b(cap * PARTICLE_BYTES, 'liq-particles')
        self.freelist = self._b(cap * 4, 'liq-free')
        self.packed = self._b(cap * PACKED_BYTES, 'liq-packed')
        self.ctr = self._b(16, 'liq-counters')
        self.args = self._b(32, 'liq-dispatch', BU.STORAGE | BU.INDIRECT | BU.COPY_DST | BU.COPY_SRC)
        self.acc = self._b(int(np.prod(m)) * 8 * 4, 'liq-p2g')
        self.cellcount = self._b(int(np.prod(n)) * 4, 'liq-cellcount')
        self.nwg = ceil_div(n[0], 8) * ceil_div(n[1], 8) * ceil_div(n[2], 4)
        self.partials = self._b(self.nwg * 4, 'liq-partials')
        self.S = self._b(32, 'liq-cg-scalars')
        self.stats_buf = self._b(32, 'liq-stats')
        wc = self.ww_capacity
        self.WA = self._b(wc * 16, 'liq-ww-a')
        self.WB = self._b(wc * 16, 'liq-ww-b')
        self.wctr = self._b(16, 'liq-ww-counters')
        self.wpacked = self._b(wc * WW_PACKED_BYTES, 'liq-ww-packed')
        self.fbuf = self._b(FLOAT_SUBSTEPS * MAX_COLLIDERS * FLOAT_SLOT * 4, 'liq-floating')

    def _compile(self):
        g = self.gpu
        k = self._k
        P = (64, 1, 1)
        k['p2g'] = g.kernel('liq_p2g.wgsl', ['rbuf', 'buf'], workgroup=P)
        k['norm'] = g.kernel('liq_norm.wgsl', ['rbuf', 'st3d:rgba32float:w', 'st3d:r32float:w'])
        k['forces'] = g.kernel('liq_forces.wgsl', ['utex3d', 'utex3d', 'utex3d', 'st3d:rgba32float:w', 'utex3d'])
        k['visc'] = g.kernel('liq_visc.wgsl', ['utex3d', 'utex3d', 'utex3d', 'st3d:rgba32float:w'])
        k['extrap'] = g.kernel('liq_extrap.wgsl', ['utex3d', 'st3d:rgba32float:w'])
        k['setup'] = g.kernel('liq_setup.wgsl', ['utex3d'] * 4 + ['st3d:r32float:w', 'st3d:rg32float:w', 'st3d:r32float:w', 'utex3d'])
        k['normal'] = g.kernel('liq_normal.wgsl', ['utex3d', 'utex3d', 'st3d:rgba16float:w'])
        k['curv'] = g.kernel('liq_curv.wgsl', ['utex3d', 'st3d:r32float:w'])
        k['dot'] = g.kernel('liq_cg_dot.wgsl', ['utex3d', 'utex3d', 'buf'])
        k['matvec'] = g.kernel('liq_cg_matvec.wgsl', ['utex3d', 'utex3d', 'st3d:r32float:w', 'buf'])
        k['final'] = g.kernel('liq_cg_final.wgsl', ['buf', 'buf'], workgroup=(256, 1, 1))
        k['upd_p'] = g.kernel('liq_cg_update_p.wgsl', ['utex3d', 'st3d:r32float:rw', 'rbuf'])
        k['upd_xr'] = g.kernel('liq_cg_update_xr.wgsl', ['utex3d', 'utex3d', 'st3d:r32float:rw', 'st3d:r32float:rw', 'rbuf'])
        k['smooth'] = g.kernel('liq_mg_smooth.wgsl', ['st3d:r32float:rw', 'utex3d', 'utex3d'])
        k['residual'] = g.kernel('liq_mg_residual.wgsl', ['utex3d', 'utex3d', 'utex3d', 'st3d:r32float:w'])
        k['restrict'] = g.kernel('liq_mg_restrict.wgsl', ['utex3d', 'utex3d', 'st3d:r32float:w', 'st3d:r32float:w'])
        k['prolong'] = g.kernel('liq_mg_prolong.wgsl', ['st3d:r32float:rw', 'utex3d', 'utex3d'])
        k['coarsen'] = g.kernel('liq_mg_coarsen.wgsl', ['utex3d', 'st3d:r32float:w'])
        k['mgco'] = g.kernel('liq_mg_co.wgsl', ['utex3d', 'st3d:rg32float:w'])
        k['project'] = g.kernel('liq_project.wgsl', ['utex3d'] * 5 + ['st3d:rgba32float:w', 'utex3d'])
        k['g2p'] = g.kernel('liq_g2p.wgsl', ['buf', 'utex3d', 'utex3d', 'utex3d', 'buf', 'buf', 'buf', 'utex3d',
                                              'buf', 'buf', 'buf', 'utex3d', 'utex3d'], workgroup=P)
        k['ww'] = g.kernel('liq_ww.wgsl', ['buf', 'buf', 'utex3d', 'utex3d', 'utex3d'], workgroup=P)
        k['ww_pack'] = g.kernel('liq_ww_pack.wgsl', ['rbuf', 'rbuf', 'buf', 'buf'], workgroup=P)
        k['spawn'] = g.kernel('liq_spawn.wgsl', ['buf', 'buf', 'rbuf', 'rbuf', 'utex3d', 'utex3d'])
        k['indirect'] = g.kernel('liq_indirect.wgsl', ['rbuf', 'buf'], workgroup=(1, 1, 1))
        k['stats'] = g.kernel('liq_stats.wgsl', ['rbuf', 'buf'], workgroup=P)
        k['pack'] = g.kernel('liq_pack.wgsl', ['rbuf', 'buf', 'buf'], workgroup=P)
        k['wet'] = g.kernel('liq_wet.wgsl', ['utex3d', 'st2d:r32float:rw'], workgroup=(8, 8, 1))
        k['sdf'] = g.kernel('sdf.wgsl', ['utex3d', 'st3d:r32float:w'])
        k['float'] = g.kernel('liq_float.wgsl', ['utex3d', 'utex3d', 'utex3d', 'utex3d', 'buf'])
        k['water'] = g.kernel('liq_water.wgsl', ['utex3d', 'st3d:r32float:w'])
        k['evap'] = g.kernel('liq_evap.wgsl', ['buf', 'buf', 'buf', 'utex3d'], workgroup=P)
        for fmt in ('rgba32float', 'r32float', 'rg32float', 'rgba16float'):
            k['fill_' + fmt] = g.kernel('fill.wgsl', [f'st3d:{fmt}:w'], 'main', {'FMT': fmt})
        # static uniforms
        self._mg_u = {}
        self._cg_u = {}

    # -- state ---------------------------------------------------------------------------------

    def fill(self, b: Batch, tex, value=(0.0, 0.0, 0.0, 0.0)):
        b.run(self._k['fill_' + tex.format], [tex], Uniforms().v4(*tex.size, 0).v4(*value), tex.size)

    def reset(self):
        g = self.gpu
        with g.batch() as b:
            for t in [self.VOLD, self.VA, self.VB, self.DENS, self.X, self.R, self.Z, self.PP, self.AP, self.NRM, self.KAPPA]:
                self.fill(b, t)
            for lst in (self.TYPE, self.CO, self.MZ[1:], self.MR[1:], self.RES):
                for t in lst:
                    self.fill(b, t)
            for buf in (self.parts, self.freelist, self.ctr, self.args, self.S, self.cellcount, self.packed,
                        self.WA, self.WB, self.wctr, self.wpacked):
                b.clear_buffer(buf)
            self._write_sdf(b)
        g.upload(self.WET, np.zeros((self.dims[2], self.dims[0], 1), np.float32))
        g.write_buffer(self.args, np.array([0, 1, 1, 0, 1, 1, 0, 0], np.uint32))
        self.time = 0.0
        self.steps = 0
        self.max_speed = 0.0
        self.bbox = None
        self.count = 0
        self.packed_count = 0
        self.ww_count = 0
        self.vel_tex = self.VA
        self._filled = set()
        self._level_filled = False

    def set_colliders(self, colliders):
        colliders = list(colliders)[:MAX_COLLIDERS]
        if colliders == self.colliders:
            return
        self.colliders = colliders
        with self.gpu.batch() as b:
            self._write_sdf(b)

    def update_colliders(self, b: Batch, colliders):
        colliders = list(colliders)[:MAX_COLLIDERS]
        if colliders == self.colliders:
            return
        self.colliders = colliders
        self._write_sdf(b)

    def _write_sdf(self, b):
        u = pack_colliders(self._grid(0.0), self.colliders, self.meshes)
        b.run(self._k['sdf'], [self.meshes.atlas, self.SDF], u, self.dims)

    def _grid(self, dt, prm: LiquidParams | None = None):
        p = prm or self._prm
        nx, ny, nz = self.dims
        return (Uniforms()
                .v4(nx, ny, nz, self.h)
                .v4(*self.origin, self.time)
                .v4(1.0 if p.open_sides else 0.0, 1.0 if p.open_top else 0.0, 0.0 if p.ground else 1.0, dt))

    def _lmg(self, lvl, parity=0):
        key = (lvl, parity)
        u = self._mg_u.get(key)
        if u is None:
            p = self._prm
            d = self.levels[lvl]
            nc = self.levels[lvl + 1] if lvl + 1 < len(self.levels) else d
            u = (Uniforms().v4(*d, parity).v4(*nc, 0)
                 .v4(1.0 if p.open_sides else 0.0, 1.0 if p.open_top else 0.0, 0.0 if p.ground else 1.0).tobytes())
            self._mg_u[key] = u
        return u

    def _cgu(self, mode=0.0, first=0.0):
        key = (mode, first)
        u = self._cg_u.get(key)
        if u is None:
            u = Uniforms().v4(*self.dims, self.nwg).v4(mode, first).tobytes()
            self._cg_u[key] = u
        return u

    # -- stepping ------------------------------------------------------------------------------

    def level_on(self, prm: LiquidParams | None = None):
        p = prm or self._prm
        return p.water_level > 0.0 and p.open_sides and self.dims is not None

    def _level_sources(self, prm: LiquidParams):
        """Open water: the box filled to the level at the start, then kept topped up along the open
        sides (a cell thick, a cell short of the level, so a passing trough is not filled in)."""
        if not self.level_on(prm):
            return []
        nx, ny, nz = self.dims
        h = self.h
        ox, oy, oz = self.origin
        top = min(prm.water_level, ny * h)
        cx, cz = ox + 0.5 * nx * h, oz + 0.5 * nz * h
        out = []
        if not self._level_filled:
            self._level_filled = True
            out.append(source('box', (cx, oy + 0.5 * top, cz), (0.5 * nx * h, 0.5 * top, 0.5 * nz * h), fill=True,
                              jitter=0.0, vel_blend=0.0))
        hi = top - h
        if hi > h:
            ym = oy + 0.5 * hi
            for pos, size in (((ox + 0.5 * h, ym, cz), (0.5 * h, 0.5 * hi, 0.5 * nz * h)),
                              ((ox + (nx - 0.5) * h, ym, cz), (0.5 * h, 0.5 * hi, 0.5 * nz * h)),
                              ((cx, ym, oz + 0.5 * h), (0.5 * nx * h, 0.5 * hi, 0.5 * h)),
                              ((cx, ym, oz + (nz - 0.5) * h), (0.5 * nx * h, 0.5 * hi, 0.5 * h))):
                out.append(source('box', pos, size, jitter=0.0, vel_blend=0.0))
        return out

    def _viscosity(self, b, dt, prm, rho0):
        """Implicit viscosity on VA (Jacobi, an even number of sweeps); returns the texture holding
        the result."""
        nu = prm.viscosity / max(prm.rho, 1.0)
        a = nu * dt / (self.h * self.h)
        if a <= 1e-6:
            return self.VA
        if self.VVISC is None:
            self.VVISC = self._t3(tuple(x + 1 for x in self.dims), 'rgba32float', 'liq-viscous')
        iters = int(min(64, max(8, 8 + 6 * math.sqrt(a))))
        iters += iters % 2
        u = (self._grid(dt, prm).v4(a, prm.surface_density * rho0)).tobytes()
        m = tuple(x + 1 for x in self.dims)
        cur = self.VA
        for i in range(iters):
            dst = self.VB if i % 2 == 0 else self.VVISC
            b.run(self._k['visc'], [self.VA, cur, self.DENS, dst], u, m)
            cur = dst
        return cur

    def step(self, b: Batch, dt, prm: LiquidParams, sources, colliders=None):
        """Record one substep of length dt (seconds) into batch b."""
        if (prm.open_sides, prm.open_top, prm.ground) != (self._prm.open_sides, self._prm.open_top, self._prm.ground):
            self._mg_u = {}
            self._press_key = None
        self._prm = prm
        k = self._k
        n = self.dims
        m = tuple(x + 1 for x in n)
        cap = self.capacity
        atlas = self.meshes.atlas
        if colliders is not None:
            self.update_colliders(b, colliders)
        srcs = (self._level_sources(prm) + list(sources))[:MAX_EMITTERS]
        G = self._grid(dt, prm)
        gb = G.tobytes()
        apic = 1.0 if prm.apic else 0.0

        # 1. particles -> grid
        b.clear_buffer(self.acc)
        b.run_indirect(k['p2g'], [self.parts, self.acc], gb + Uniforms().v4(cap, apic, prm.speed_limit).tobytes(), self.args, 0)
        b.run(k['norm'], [self.acc, self.VOLD, self.DENS], gb, m)

        # 2. forces and boundaries
        rho0 = float(prm.ppc)
        wind = tuple(float(x) for x in prm.wind)
        wspeed = math.sqrt(sum(x * x for x in wind))
        level = prm.water_level / self.h if self.level_on(prm) else -1.0
        wcells = max(1.0, float(prm.level_absorb))
        calm = 6.0 * math.sqrt(prm.gravity / (wcells * self.h)) if prm.gravity > 0 else 0.0
        u = (self._grid(dt, prm).v4(prm.gravity)
             .v4(*wind, 0.0006 * prm.wind_surface * wspeed / self.h)
             .v4(level, wcells, calm, prm.surface_density * rho0))
        pack_emitters(u, srcs, meshes=self.meshes)
        pack_colliders(u, self.colliders, self.meshes)
        b.run(k['forces'], [self.VOLD, self.SDF, atlas, self.VA, self.DENS], u, m)

        # 3. a little extrapolation so every face of a liquid cell has a value
        b.run(k['extrap'], [self.VA, self.VB], gb, m)
        b.run(k['extrap'], [self.VB, self.VA], gb, m)

        # viscosity
        vin = self._viscosity(b, dt, prm, rho0) if prm.viscosity > 0 else self.VA

        # 4. pressure
        ku = gb + (Uniforms().v4(prm.surface_density * rho0, rho0, prm.volume_correction, prm.theta_min)
                   .v4(prm.compression, prm.surface_tension / prm.rho * dt)
                   .v4(level, prm.gravity * dt).tobytes())
        ww = prm.whitewater and self.ww_capacity > 64
        if prm.surface_tension > 0 or ww:
            ca = math.radians(prm.contact_angle)
            on = 1.0 if abs(prm.contact_angle - 90.0) > 0.5 else 0.0
            b.run(k['normal'], [self.DENS, self.SDF, self.NRM], gb + Uniforms().v4(rho0, math.cos(ca), math.sin(ca), on).tobytes(), n)
            b.run(k['curv'], [self.NRM, self.KAPPA], gb + Uniforms().v4(0.05).tobytes(), n)
        b.run(k['setup'], [vin, self.DENS, self.SDF, self.X, self.TYPE[0], self.CO[0], self.R, self.KAPPA], ku, n)
        self._pressure(b, prm)
        pout = self.VA if vin is self.VB else self.VB
        b.run(k['project'], [vin, self.VOLD, self.X, self.TYPE[0], self.DENS, pout, self.KAPPA], ku, m)

        # 5. extrapolate the divergence-free velocity into the air
        src, dst = pout, (self.VA if pout is self.VB else self.VB)
        for _ in range(max(1, int(prm.extrapolate))):
            b.run(k['extrap'], [src, dst], gb, m)
            src, dst = dst, src
        self.vel_tex = src

        # 6. grid -> particles, move
        b.clear_buffer(self.cellcount)
        u = gb + (Uniforms().v4(cap, prm.flip, apic, prm.speed_limit)
                  .v4(prm.collision_radius, prm.wall_drag, 0.15 * rho0, 0.5 * rho0)
                  .v4(prm.ww_rate, prm.ww_min_speed, prm.ww_turbulence, prm.ww_crests)
                  .v4(self.ww_capacity if ww else 0.0, prm.foam_life, self.steps + prm.seed * 131.0, rho0)
                  .v4(prm.damping).tobytes())
        b.run_indirect(k['g2p'], [self.parts, src, self.VOLD, self.SDF, self.ctr, self.freelist, self.cellcount,
                                  self.DENS, self.WA, self.WB, self.wctr, self.NRM, self.KAPPA], u, self.args, 0)
        if ww:
            u = gb + (Uniforms().v4(self.ww_capacity, rho0, prm.gravity, prm.bubble_rise)
                      .v4(prm.spray_drag, prm.bubble_drag, 0.3, 0.85)
                      .v4(*wind, 0.03 * prm.wind_surface).tobytes())
            b.run(k['ww'], [self.WA, self.WB, src, self.DENS, self.SDF], u, groups=groups_1d(self.ww_capacity))
        b.run(k['wet'], [self.DENS, self.WET], gb + Uniforms().v4(0.25 * rho0, prm.drying).tobytes(), (n[0], n[2], 1))

        # 7. sources
        if srcs:
            m = round(prm.ppc ** (1.0 / 3.0))
            u = self._grid(dt, prm).v4(prm.ppc, cap, self.steps + prm.seed * 7919.0, m if m ** 3 == prm.ppc else 1)
            pack_emitters(u, srcs, meshes=self.meshes)
            b.run(k['spawn'], [self.parts, self.ctr, self.freelist, self.cellcount, self.SDF, atlas], u, n)
        b.run(k['indirect'], [self.ctr, self.args], Uniforms().v4(0, cap, 0), groups=(1, 1, 1))

        self.time += dt
        self.steps += 1

    def _pressure(self, b, prm):
        """Coarse levels and the preconditioned CG, replayed from a pre-bound Program."""
        key = (int(prm.pressure_iters), int(prm.mg_sweeps), int(prm.coarse_sweeps), prm.open_sides, prm.open_top, prm.ground)
        if self._press is None or self._press_key != key:
            if self._press is not None:
                self._press.destroy()
            rec = _Recorder()
            self._coarsen(rec)
            self._pcg(rec, prm)
            self._press = Program(self.gpu, rec.ops)
            self._press_key = key
        b.run_program(self._press)

    def _coarsen(self, b):
        k = self._k
        for lvl in range(len(self.levels) - 1):
            b.run(k['coarsen'], [self.TYPE[lvl], self.TYPE[lvl + 1]], self._lmg(lvl), self.levels[lvl + 1])
            b.run(k['mgco'], [self.TYPE[lvl + 1], self.CO[lvl + 1]], self._lmg(lvl + 1), self.levels[lvl + 1])

    def _pcg(self, b, prm):
        k = self._k
        n = self.dims
        for it in range(max(1, int(prm.pressure_iters))):
            self.fill(b, self.Z)
            self._vcycle(b, 0, prm)
            b.run(k['dot'], [self.R, self.Z, self.partials], self._cgu(), n)
            b.run(k['final'], [self.partials, self.S], self._cgu(0.0, 1.0 if it == 0 else 0.0), groups=(1, 1, 1))
            b.run(k['upd_p'], [self.Z, self.PP, self.S], self._cgu(), n)
            b.run(k['matvec'], [self.PP, self.CO[0], self.AP, self.partials], self._cgu(), n)
            b.run(k['final'], [self.partials, self.S], self._cgu(1.0), groups=(1, 1, 1))
            b.run(k['upd_xr'], [self.PP, self.AP, self.X, self.R, self.S], self._cgu(), n)

    def _smooth(self, b, lvl, parity):
        d = self.levels[lvl]
        b.run(self._k['smooth'], [self.MZ[lvl], self.MR[lvl], self.CO[lvl]], self._lmg(lvl, parity),
              ((d[0] + 1) // 2, d[1], d[2]))

    def _vcycle(self, b, lvl, prm):
        k = self._k
        last = len(self.levels) - 1
        if lvl == last:
            for _ in range(prm.coarse_sweeps):
                self._smooth(b, lvl, 0)
                self._smooth(b, lvl, 1)
            for _ in range(prm.coarse_sweeps):
                self._smooth(b, lvl, 1)
                self._smooth(b, lvl, 0)
            return
        for _ in range(prm.mg_sweeps):
            self._smooth(b, lvl, 0)
            self._smooth(b, lvl, 1)
        b.run(k['residual'], [self.MZ[lvl], self.MR[lvl], self.CO[lvl], self.RES[lvl]], self._lmg(lvl), self.levels[lvl])
        b.run(k['restrict'], [self.RES[lvl], self.CO[lvl + 1], self.MR[lvl + 1], self.MZ[lvl + 1]], self._lmg(lvl),
              self.levels[lvl + 1])
        self._vcycle(b, lvl + 1, prm)
        b.run(k['prolong'], [self.MZ[lvl], self.MZ[lvl + 1], self.CO[lvl]], self._lmg(lvl), self.levels[lvl])
        for _ in range(prm.mg_sweeps):
            self._smooth(b, lvl, 1)
            self._smooth(b, lvl, 0)

    # -- fire in the same box --------------------------------------------------------------------

    def water_into(self, b: Batch, fire):
        """Write the liquid's share of each cell into the fire solver's water field."""
        b.run(self._k['water'], [self.DENS, fire.water], Uniforms().v4(*fire.dims).v4(*self.dims, float(self._prm.ppc)),
              fire.dims)

    def evaporate(self, b: Batch, fire, boil, rate, dt):
        """Liquid in fire gas hotter than boiling (fire temperature scale) boils away at up to `rate`
        (share per second)."""
        if rate <= 0.0:
            return
        u = self._grid(dt).v4(self.capacity, boil, rate, self.steps + 17).v4(*fire.dims)
        b.run_indirect(self._k['evap'], [self.parts, self.ctr, self.freelist, fire.scal[0]], u, self.args, 0)

    # -- floating objects ------------------------------------------------------------------------

    def clear_float(self, b: Batch):
        b.clear_buffer(self.fbuf)

    def float_forces(self, b: Batch, bodies, substep, dt):
        """Gather the liquid's push on floating colliders after a step. bodies: (collider index in
        the current collider list, bounding radius in metres), at most MAX_COLLIDERS."""
        h = self.h
        n = self.dims
        o = np.asarray(self.origin)
        sub = substep % FLOAT_SUBSTEPS
        for k, (ci, radius) in enumerate(list(bodies)[:MAX_COLLIDERS]):
            if ci >= len(self.colliders):
                continue
            c = (np.asarray(self.colliders[ci].pos, float) - o) / h
            r = radius / h + 2.0
            lo = np.maximum(np.floor(c - r), 0).astype(int)
            hi = np.minimum(np.ceil(c + r), n).astype(int)
            size = hi - lo
            if np.any(size <= 0):
                continue
            u = self._grid(dt).v4(*lo, ci).v4(*size, sub * MAX_COLLIDERS + k).v4(0.5 * self._prm.gravity * dt * h)
            pack_colliders(u, self.colliders, self.meshes)
            b.run(self._k['float'], [self.X, self.TYPE[0], self.vel_tex, self.meshes.atlas, self.fbuf], u, tuple(int(x) for x in size))

    def read_float(self, count, substeps):
        """Per body, averaged over the substeps: pressure force sum (x units, cells^2 folded in by
        the caller), vertical moment, liquid velocity next to it (m/s), solid cells."""
        raw = np.frombuffer(self.gpu.read_buffer(self.fbuf), np.int32).reshape(FLOAT_SUBSTEPS, MAX_COLLIDERS, FLOAT_SLOT)
        used = raw.astype(np.float64)  # slots wrap past FLOAT_SUBSTEPS: sum them all
        out = []
        for k in range(min(count, MAX_COLLIDERS)):
            s = used[:, k, :]
            nsub = max(1, substeps)
            f = s[:, 0:3].sum(0) / 65536.0 / nsub
            tq = s[:, 3].sum() / 4096.0 / nsub
            nv = s[:, 7].sum()
            v = s[:, 4:7].sum(0) / 1024.0 / nv if nv > 0 else np.zeros(3)
            cells = s[:, 11].sum() / nsub
            if s[:, 10].sum() > 0:
                # seepage under a body resting on the ground (see liq_float.wgsl)
                f[1] += s[:, 8].sum() / nsub * (s[:, 9].sum() / 65536.0) / s[:, 10].sum()
            out.append({'force': f, 'moment': tq, 'liquid_vel': v, 'wet': nv > 0, 'cells': cells})
        return out

    # -- output ----------------------------------------------------------------------------------

    def pack(self, b: Batch):
        """Pack the live particles (16 bytes each) for the surface builder and the frame cache."""
        b.clear_buffer(self.ctr, 8, 4)
        b.run_indirect(self._k['pack'], [self.parts, self.packed, self.ctr], self._grid(0.0).v4(self.capacity), self.args, 0)
        b.clear_buffer(self.wctr, 4, 4)
        if self.ww_capacity > 64:
            b.run(self._k['ww_pack'], [self.WA, self.WB, self.wpacked, self.wctr],
                  self._grid(0.0).v4(self.ww_capacity, self._prm.foam_life), groups=groups_1d(self.ww_capacity))

    def measure(self):
        """Live particle count, top speed and liquid bounding box (cells), read back from the GPU.
        Also reads the packed count if pack() ran."""
        init = np.array([0, 0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0, 0], np.uint32)
        self.gpu.write_buffer(self.stats_buf, init)
        with self.gpu.batch() as b:
            b.run_indirect(self._k['stats'], [self.parts, self.stats_buf], self._grid(0.0).v4(self.capacity), self.args, 0)
        raw = np.frombuffer(self.gpu.read_buffer(self.stats_buf), np.uint32)
        ctr = np.frombuffer(self.gpu.read_buffer(self.ctr), np.int32)
        self.count = int(raw[0])
        self.max_speed = float(np.frombuffer(raw[1:2].tobytes(), np.float32)[0]) if raw[0] else 0.0
        self.bbox = None if raw[0] == 0 else (tuple(int(x) for x in raw[2:5]), tuple(int(x) + 1 for x in raw[5:8]))
        self.packed_count = int(max(0, ctr[2]))
        wctr = np.frombuffer(self.gpu.read_buffer(self.wctr), np.uint32)
        self.ww_count = int(min(wctr[1], self.ww_capacity))
        return self.max_speed, self.bbox, self.count

    def capillary_substeps(self, frame_dt):
        """Substeps per frame the free surface needs to stay stable: explicit surface tension below
        the capillary time step of a cell, and gravity waves below 0.3 sqrt(h / g) (longer steps let
        cell-sized ripples grow out of the particle noise)."""
        n = 1
        if not self.h:
            return n
        st = self._prm.surface_tension
        if st > 0:
            dt_cap = 0.8 * math.sqrt(self._prm.rho * self.h ** 3 / (2.0 * math.pi * st))
            n = max(n, math.ceil(frame_dt / dt_cap))
        if self._prm.gravity > 0:
            n = max(n, math.ceil(frame_dt / (0.3 * math.sqrt(self.h / self._prm.gravity))))
        return n

    def substeps_for(self, frame_dt, cfl=2.0, lo=1, hi=8):
        n = math.ceil(self.max_speed * frame_dt / (cfl * self.h)) if self.max_speed > 0 else lo
        n = max(n, self.capillary_substeps(frame_dt))
        return int(min(hi, max(lo, n)))

    def read_packed(self):
        """(n, 4) uint32 packed particles of the last pack()."""
        if not self.packed_count:
            return np.zeros((0, 4), np.uint32)
        data = self.gpu.read_buffer(self.packed, self.packed_count * PACKED_BYTES)
        return np.frombuffer(data, np.uint32).reshape(-1, 4).copy()

    def read_ww_packed(self):
        """(n, 4) uint32 packed whitewater particles of the last pack()."""
        if not self.ww_count:
            return np.zeros((0, 4), np.uint32)
        data = self.gpu.read_buffer(self.wpacked, self.ww_count * WW_PACKED_BYTES)
        return np.frombuffer(data, np.uint32).reshape(-1, 4).copy()

    def read_wet(self):
        return self.gpu.read(self.WET)[..., 0].astype(np.float16)

    def read_particles(self):
        """Live particle positions (fire-local metres) and velocities (m/s), float32 (n, 3) each."""
        pk = self.read_packed()
        return unpack_particles(pk, self.dims, self.h, self.origin)

    def world_bounds(self):
        nx, ny, nz = self.dims
        o = np.array(self.origin)
        return o, o + np.array([nx, ny, nz]) * self.h


def unpack_particles(pk, dims, h, origin):
    """Packed particles -> positions (fire-local metres) and velocities (m/s)."""
    if len(pk) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    xy = pk[:, 0]
    zz = pk[:, 1]
    q = np.stack([(xy & 0xFFFF), (xy >> 16), (zz & 0xFFFF)], -1).astype(np.float32) / 65535.0
    pos = q * np.asarray(dims, np.float32) * h + np.asarray(origin, np.float32)
    v = np.frombuffer(pk[:, 2:4].astype('<u4').tobytes(), np.float16).reshape(-1, 4)[:, :3].astype(np.float32)
    return pos, v
