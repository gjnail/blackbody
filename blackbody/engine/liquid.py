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
level the sides below it hold the liquid in (the water past them is still), a layer along them
calms waves so they do not reflect, liquid heaped above the level at the edge runs off over it, and
sources along the sides top the water up if it ever drops.

Narrow band: particles are kept only within a band under the surface. Deeper cells are 'deep'
liquid carried by the grid alone: after the splat they count as full and keep the grid velocity of
the last step; after the move, particles that sank into the deep liquid are freed and cells the
surface came back down to are reseeded; a filling source marks its deep inside directly.

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

PARTICLE_BYTES = 48
PACKED_BYTES = 16
WW_BYTES = 32
WW_PACKED_BYTES = 16
FLOAT_SUBSTEPS = 16     # substep slots of the floating-object force sums
FLOAT_SLOT = 16         # i32 per body and substep (see liq_float.wgsl)
HEAT_MIXING = 1.5       # 1/s: how fast a molten liquid's particles even out their heat with their cell's
CRUST_LATCH = 0.02      # s out in the air a molten liquid's melt needs by one of the moments to form crust


def crust_restarts(t):
    """The last of the moments a molten liquid's new skin takes its crust coordinate (liq_crust_adv.wgsl) at or
    before time t (s), and the next: irregular, 0.3 to 0.8 s apart, the same every run."""
    a, k = 0.0, 0
    while True:
        b = a + 0.3 + 0.5 * ((k * 0.6180339887 + 0.137) % 1.0)
        if b > t:
            return a, b
        a, k = b, k + 1


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
    spray_drag: float = 1.5         # 1/s: millimetre drops, falling at about 6 m/s at most, carried by the wind
    bubble_drag: float = 12.0       # 1/s
    viscosity: float = 0.0          # Pa s
    cooling: float = 0.0            # 1/s: how fast a molten liquid's surface loses its heat (0: never)
    solidify: float = 1000.0        # how much stiffer it is cooled than molten (with cooling)
    contact_angle: float = 90.0     # degrees
    water_level: float = 0.0        # m, open water past the sides (0 = off)
    level_absorb: float = 8.0       # cells
    wind: tuple = (0.0, 0.0, 0.0)   # m/s, fire-local
    wind_surface: float = 1.0
    narrow_band: bool = False
    band_width: int = 4             # cells
    ocean: object = None            # OceanSpec: waves on the open water (None or height 0: flat)
    clock: float = 0.0              # s, the shot's time at this step (the sea's waves run on it)
    current: tuple = (0.0, 0.0, 0.0)  # m/s, fire-local: the open water flows past (a river)
    sea_sides: tuple = (1.0, 1.0, 1.0, 1.0)  # open sides (-x, +x, -z, +z) the open water flows through;
                                             # the others are walls under the level (a wave flume)
    dye_mixing: float = 0.25        # 1/s: how fast a dye evens out with the liquid around it
    rain: float = 0.0               # mm/h of rain landing on the liquid (splashes of spray)
    rain_drop: float = 2.5          # mm, raindrop diameter
    # heat and phase changes (liquid_thermal.py): temperatures in C
    thermal: bool = False           # the liquid has a temperature: it freezes, melts, boils and evaporates
    temp: float = 20.0              # the liquid as sources pour it (unless they set their own) and the open water
    air_temp: float = 20.0          # the air round the liquid (with fire in the box: the gas's instead)
    humidity: float = 50.0          # % relative humidity of that air
    ground_temp: float = 20.0       # the ground, the footage's surfaces and closed walls
    collider_temps: tuple = ()      # each collider's surface temperature, in the collider list's order
    min_source_temp: float = 20.0   # the coldest source (ice can form from the start)
    freeze_point: float = 0.0
    boil_point: float = 100.0
    supercool: float = 1.0          # K below freezing still water holds before it freezes by itself
    heat_speed: float = 1.0         # heat flows this many times faster than real (a time-lapse of freezing)
    bubble_size: float = 2.5        # mm, diameter steam bubbles leave a hot surface at
    gas_air_k: float = 300.0        # with fire in the box, its temperature scale: the air ...
    gas_flame_k: float = 1500.0     # ... and the flame (K)
    open_sides: bool = True
    open_top: bool = True
    ground: bool = True


def rain_speed(drop_mm):
    """Terminal speed (m/s) of a raindrop of diameter drop_mm (Atlas et al. 1973)."""
    return max(0.3, 9.65 - 10.3 * math.exp(-0.6 * float(drop_mm)))


def rain_drops_per_m2s(rate_mm_h, drop_mm):
    """Raindrops landing per square metre per second in rain of rate_mm_h of drops drop_mm across."""
    d = max(float(drop_mm), 0.1) * 1e-3
    return max(float(rate_mm_h), 0.0) / 3.6e6 / (math.pi / 6.0 * d ** 3)


def source(shape='sphere', pos=(0.0, 0.5, 0.0), size=(0.1, 0.1, 0.1), p1=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0),
           radial=0.0, strength=1.0, fill=False, jitter=0.02, vel_blend=1.0, yaw=0.0, mesh='', soft=0.0, top_up=0.0,
           dye=(1.0, 1.0, 1.0), dye_amount=0.0, dye_cloud=0.5, density=1.0, temp=None):
    """A liquid source in the emitter layout the kernels share (see the module docstring). top_up > 0
    makes a stream source refill only cells that have lost more than that share of their particles.
    A dye (colour, strength per metre, cloudiness) and a density relative to the liquid ride in the
    emitter's colourant slots: colour = absorption per metre (rgb), vapour = scattering per metre,
    noise frequency = density ratio. A temperature (C) of its own rides in the swirl slot as 1000 + temp
    (liq_therm_p2g.wgsl); without one the liquid's temperature applies."""
    amt = max(float(dye_amount), 0.0)
    absorb = tuple(-math.log(min(max(c, 1e-3), 1.0)) * amt * (1.0 - 0.7 * dye_cloud) for c in dye)
    return EmitterGPU(shape=shape, pos=tuple(pos), size=tuple(size), p1=tuple(p1), soft=soft, fuel=float(strength),
                      temp=1.0 if fill else 0.0, smoke=float(jitter), vel=tuple(vel), radial=float(radial),
                      vel_blend=float(vel_blend), noise=float(top_up), yaw=float(yaw), mesh=mesh,
                      color=absorb, vapour=amt * float(dye_cloud), noise_freq=float(density),
                      swirl=1000.0 + float(temp) if temp is not None else 0.0)


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
        self.VMRG = None
        self.band = None
        self.bank = None        # the narrow band's balance (liq_band_bank.wgsl)
        self._band_i = 0
        self._level_filled = False
        self.attr = None
        self._dye_on = False
        self._rho_on = False
        self.CX = None
        self.CA = None
        self.CV = None
        self._crust_on = False
        self._footage = None
        self._origin0 = None
        self._strips = []
        self.WET2 = None
        self.thermal = None         # liquid_thermal.Thermal, once a scene turns heat on
        self.gas = None             # the fire Solver sharing the box ('both'): the air the liquid meets
        self.lava_heat = None       # lava sharing the box: gas-grid texture, x = temperature (K), y = share

    # -- allocation ---------------------------------------------------------------------------

    dims_for = staticmethod(Solver.dims_for)

    @staticmethod
    def capacity_for(dims, ppc, max_particles, band=None):
        """Particle slots: a full box of particles, or with a narrow band (width in cells) about two
        layers of the band across the box's footprint."""
        cells = int(np.prod(dims))
        if band:
            cells = min(cells, int(dims[0] * dims[2] * (int(band) + 6) * 2))
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
        self._origin0 = self.origin
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
        if self.thermal is not None:
            self.thermal.release()
        self.VVISC = None
        self.VMRG = None
        self.band = None
        self.bank = None
        self.attr = None
        self.CX = None
        self.CA = None
        self.CV = None
        self.WET2 = None
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
        self.HEAT = self._t3(n, 'r32float', 'liq-heat')
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
        self._noband = self._b(16, 'liq-no-band')
        self._noattr = self._b(16, 'liq-no-attributes')
        self._noice = self._b(16, 'liq-no-heat')
        self._noice2 = self._b(16, 'liq-no-ice')
        self.OCN = g.texture2d(n[0], n[2], 'rgba32float', 'liq-open-water')
        self._tex.append(self.OCN)
        self._zero3 = self._t3((1, 1, 1), 'r32float', 'liq-zero')
        self._sea_kp = 1.0
        self._seabc = None

    def _compile(self):
        g = self.gpu
        k = self._k
        P = (64, 1, 1)
        k['p2g'] = g.kernel('liq_p2g.wgsl', ['rbuf', 'buf'], workgroup=P)
        k['norm'] = g.kernel('liq_norm.wgsl', ['rbuf', 'st3d:rgba32float:w', 'st3d:r32float:w', 'st3d:r32float:w'])
        k['forces'] = g.kernel('liq_forces.wgsl', ['utex3d', 'utex3d', 'utex3d', 'st3d:rgba32float:w', 'utex3d', 'utex2d',
                                                   'utex3d'])
        k['attr_p2g'] = g.kernel('liq_attr_p2g.wgsl', ['rbuf', 'rbuf', 'buf'], workgroup=P)
        k['attr_norm'] = g.kernel('liq_attr_norm.wgsl', ['rbuf', 'utex3d', 'st3d:rgba16float:w', 'st3d:rgba16float:w'])
        k['attr_buoy'] = g.kernel('liq_attr_buoy.wgsl', ['utex3d', 'st3d:r32float:w'])
        k['attr_g2p'] = g.kernel('liq_attr_g2p.wgsl', ['rbuf', 'buf', 'utex3d'], workgroup=P)
        k['visc'] = g.kernel('liq_visc.wgsl', ['utex3d', 'utex3d', 'utex3d', 'st3d:rgba32float:w', 'utex3d'])
        k['extrap'] = g.kernel('liq_extrap.wgsl', ['utex3d', 'st3d:rgba32float:w'])
        k['setup'] = g.kernel('liq_setup.wgsl', ['utex3d'] * 4 + ['st3d:r32float:w', 'st3d:rg32float:w', 'st3d:r32float:w', 'utex3d',
                                                              'utex2d'])
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
        k['project'] = g.kernel('liq_project.wgsl', ['utex3d'] * 5 + ['st3d:rgba32float:w', 'utex3d', 'utex2d'])
        k['g2p'] = g.kernel('liq_g2p.wgsl', ['buf', 'utex3d', 'utex3d', 'utex3d', 'buf', 'buf', 'buf', 'utex3d',
                                              'buf', 'buf', 'buf', 'utex3d', 'utex3d', 'utex2d'], workgroup=P)
        k['ww'] = g.kernel('liq_ww.wgsl', ['buf', 'buf', 'utex3d', 'utex3d', 'utex3d'], workgroup=P)
        k['rain'] = g.kernel('liq_rain.wgsl', ['utex3d', 'utex3d', 'buf', 'buf', 'buf'], workgroup=(8, 8, 1))
        k['ww_pack'] = g.kernel('liq_ww_pack.wgsl', ['rbuf', 'rbuf', 'buf', 'buf'], workgroup=P)
        k['spawn'] = g.kernel('liq_spawn.wgsl', ['buf', 'buf', 'rbuf', 'rbuf', 'utex3d', 'utex3d', 'buf', 'utex2d', 'buf'])
        k['band_merge'] = g.kernel('liq_band_merge.wgsl', ['utex3d', 'utex3d', 'rbuf', 'st3d:r32float:rw', 'st3d:rgba32float:w'])
        k['band_dist'] = g.kernel('liq_band_dist.wgsl', ['utex3d', 'rbuf', 'utex3d', 'utex3d', 'st3d:r32float:w'])
        k['band_update'] = g.kernel('liq_band_update.wgsl', ['utex3d', 'rbuf', 'utex3d', 'utex3d', 'buf', 'buf'])
        k['band_cull'] = g.kernel('liq_band_cull.wgsl', ['buf', 'buf', 'buf', 'rbuf', 'rbuf', 'buf', 'utex3d', 'buf'],
                                  workgroup=P)
        k['band_reseed'] = g.kernel('liq_band_reseed.wgsl', ['buf', 'buf', 'rbuf', 'rbuf', 'buf', 'utex3d', 'utex3d', 'buf',
                                                             'rbuf', 'buf'])
        k['band_emit'] = g.kernel('liq_band_emit.wgsl', ['buf', 'buf', 'rbuf', 'rbuf', 'utex3d', 'utex3d', 'buf', 'buf', 'buf'])
        k['band_bank'] = g.kernel('liq_band_bank.wgsl', ['buf'], workgroup=(1, 1, 1))
        k['indirect'] = g.kernel('liq_indirect.wgsl', ['rbuf', 'buf'], workgroup=(1, 1, 1))
        k['stats'] = g.kernel('liq_stats.wgsl', ['rbuf', 'buf'], workgroup=P)
        k['pack'] = g.kernel('liq_pack.wgsl', ['rbuf', 'buf', 'buf', 'rbuf', 'buf', 'rbuf', 'buf'], workgroup=P)
        k['crust_adv'] = g.kernel('liq_crust_adv.wgsl', ['utex3d'] * 5 + ['st3d:rgba32float:w', 'st3d:r32float:w',
                                                                        'st3d:rgba32float:w'])
        k['heat_mix'] = g.kernel('liq_heat_mix.wgsl', ['buf', 'utex3d', 'utex3d'], workgroup=P)
        k['kill'] = g.kernel('liq_kill.wgsl', ['buf'], workgroup=P)
        k['wet'] = g.kernel('liq_wet.wgsl', ['utex3d', 'st2d:r32float:rw'], workgroup=(8, 8, 1))
        k['sdf'] = g.kernel('sdf.wgsl', ['utex3d', 'st3d:r32float:w'])
        k['footage_sdf'] = g.kernel('liq_footage_sdf.wgsl', ['utex2d', 'st3d:r32float:rw'])
        k['shift'] = g.kernel('liq_shift.wgsl', ['buf', 'buf', 'buf', 'buf'], workgroup=P)
        k['shift_ww'] = g.kernel('liq_shift_fields.wgsl', ['buf'], 'ww', workgroup=P)
        k['shift_band'] = g.kernel('liq_shift_band.wgsl', ['rbuf', 'buf', 'utex2d', 'st2d:r32float:w'])
        k['shift_vel'] = g.kernel('liq_shift_vel.wgsl', ['utex3d', 'st3d:rgba32float:w'])
        k['shift_scalar'] = g.kernel('liq_shift_scalar.wgsl', ['utex3d', 'st3d:r32float:w'])
        k['float'] = g.kernel('liq_float.wgsl', ['utex3d', 'utex3d', 'utex3d', 'utex3d', 'buf'])
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
        if self._origin0 is not None and self.origin != self._origin0:
            self.origin = self._origin0     # a box that followed the liquid starts where it was set
        self._strips = []
        with g.batch() as b:
            for t in [self.VOLD, self.VA, self.VB, self.DENS, self.X, self.R, self.Z, self.PP, self.AP, self.NRM, self.KAPPA]:
                self.fill(b, t)
            for lst in (self.TYPE, self.CO, self.MZ[1:], self.MR[1:], self.RES):
                for t in lst:
                    self.fill(b, t)
            for buf in (self.parts, self.freelist, self.ctr, self.args, self.S, self.cellcount, self.packed,
                        self.WA, self.WB, self.wctr, self.wpacked):
                b.clear_buffer(buf)
            for buf in self.band or ():
                b.clear_buffer(buf)
            if self.bank is not None:
                b.clear_buffer(self.bank)
            if self.attr is not None:
                b.clear_buffer(self.attr)
            if self.CX is not None:
                for t in self.CX + self.CA + [self.CV]:
                    self.fill(b, t)
            # every slot free: cleared, the slots past those in use read as live particles at the grid's
            # corner, and the particle kernels run over whole rows of workgroups past them (liq_kill.wgsl)
            b.run(self._k['kill'], [self.parts], Uniforms().v4(self.capacity).tobytes(), groups=groups_1d(self.capacity))
            if self.thermal is not None:
                self.thermal.reset(b)
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
        self._band_i = 0
        self._dye_on = False
        self._rho_on = False
        self._crust_on = False

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
        if self._footage is not None:
            tex, fu = self._footage
            b.run(self._k['footage_sdf'], [tex, self.SDF], self._grid(0.0).tobytes() + fu, self.dims)

    def set_footage(self, b: Batch, depth_tex=None, camstate=None, fire=None, depth_kind='z', depth_scale=1.0,
                    fit=(1.0, 1.0), thickness=0.3):
        """Make the footage's surfaces solid for the liquid: a slab `thickness` metres deep behind what its
        depth pass (texture, .y in the pass's units) shows through camera `camstate` (None: off)."""
        if depth_tex is None:
            if self._footage is not None:
                self._footage = None
                self._write_sdf(b)
            return
        from .renderer import DEPTH_KINDS
        fwd = -camstate.view[2, :3]
        fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
        fu = (Uniforms().m4(camstate.view_proj).m4(fire.local_to_world())
              .v4(*camstate.eye, max(float(thickness), 3.0 * self.h)).v4(*fwd, 0.0)
              .v4(DEPTH_KINDS.get(depth_kind, 0), depth_scale, *fit)).tobytes()
        self._footage = (depth_tex, fu)
        self._write_sdf(b)

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

    def _ensure_band(self, b: Batch):
        """Allocate the narrow band's buffers (cleared through the caller's batch: a batch opened
        inside another would recycle the uniforms the outer one has already recorded)."""
        if self.band is None:
            cells = int(np.prod(self.dims))
            self.band = [self._b(cells * 4, f'liq-band{i}') for i in range(2)]
            self.bank = self._b(32, 'liq-band-bank')
            self.VMRG = self._t3(tuple(x + 1 for x in self.dims), 'rgba32float', 'liq-vold-merged')
            for buf in self.band + [self.bank]:
                b.clear_buffer(buf)

    def _ensure_attrs(self, b: Batch):
        """Allocate what the particles carry besides their motion (dye, density; see liq_common.wgsl),
        cleared through the caller's batch."""
        if self.attr is None:
            n = self.dims
            cells = int(np.prod(n))
            self.attr = self._b(self.capacity * 16, 'liq-attributes')
            self.aacc = self._b(cells * 5 * 4, 'liq-attr-p2g')
            self.pattr = self._b(self.capacity * 8, 'liq-packed-dye')
            self.ATTR = self._t3(n, 'rgba16float', 'liq-dye')
            self.RHO = self._t3(n, 'rgba16float', 'liq-rho')
            self.BUOY = self._t3(n, 'r32float', 'liq-buoyancy')
            b.clear_buffer(self.attr)

    def _ensure_crust(self, b: Batch):
        """Allocate a molten liquid's crust field (liq_crust_adv.wgsl): the crust coordinate and whether crust
        has formed, the skin's age (each twice, read and written in turn), and both for the renderer; cleared
        through the caller's batch."""
        if self.CX is None:
            n = self.dims
            self.CX = [self._t3(n, 'rgba32float', f'liq-crust{i}') for i in range(2)]
            self.CA = [self._t3(n, 'r32float', f'liq-skin-age{i}') for i in range(2)]
            self.CV = self._t3(n, 'rgba32float', 'liq-crust-view')
            for t in self.CX + self.CA + [self.CV]:
                self.fill(b, t)

    def _carry(self, b: Batch, srcs):
        """Turn on carrying dye / density once a source pours either (it stays on: the particles
        keep it)."""
        for s in srcs:
            if not self._dye_on and (s.vapour > 0.0 or max(s.color) > 0.0):
                self._dye_on = True
            if not self._rho_on and abs(s.noise_freq - 1.0) > 1e-4:
                self._rho_on = True
        if self._dye_on or self._rho_on:
            self._ensure_attrs(b)
        return self._dye_on or self._rho_on

    @property
    def dye_on(self):
        return self._dye_on

    @property
    def band_buffer(self):
        """The deep-liquid flags (u32 per cell) of a narrow-band simulation, or None."""
        return self.band[self._band_i] if (self.band is not None and self._prm.narrow_band) else None

    def read_band(self):
        """The deep-liquid flags as packed bits (np.packbits over the cells, x fastest), or None."""
        buf = self.band_buffer
        if buf is None:
            return None
        return np.packbits(np.frombuffer(self.gpu.read_buffer(buf), np.uint32) != 0)

    def _band_update(self, b, prm, vel, rho0, gb):
        """After the particles moved: what flowed out of the deep liquid back as particles, the new deep
        liquid, then free the particles in it and reseed what is no longer deep. Every pass books what
        it creates and frees in the bank, and the emission pays the balance back, so the exchange with
        the deep liquid keeps the volume (liq_band_bank.wgsl)."""
        k = self._k
        n = self.dims
        cur, new = self.band[self._band_i], self.band[1 - self._band_i]
        thr = prm.surface_density * rho0
        w = max(1, int(prm.band_width))
        carry = self.attr is not None and (self._dye_on or self._rho_on)
        b.run(k['band_bank'], [self.bank], None, groups=(1, 1, 1))
        # what flowed out of the deep liquid during the move comes back as particles, matching what the
        # cull takes in; they count toward the cells joining or leaving the deep liquid below
        u = gb + Uniforms().v4(prm.ppc, self.capacity, self.steps * 5 + 3 + prm.seed * 131.0, 1.0 if carry else 0.0).tobytes()
        b.run(k['band_emit'], [self.parts, self.ctr, self.freelist, cur, self.SDF, vel,
                               self.attr if carry else self._noattr, self.cellcount, self.bank], u, n)
        b.run(k['band_dist'], [self.DENS, cur, self.SDF, self.Z, self.PP], gb + Uniforms().v4(thr, 0.0).tobytes(), n)
        src, dst = self.PP, self.Z
        for _ in range(w + 1):
            b.run(k['band_dist'], [self.DENS, cur, self.SDF, src, dst], gb + Uniforms().v4(thr, 1.0).tobytes(), n)
            src, dst = dst, src
        b.run(k['band_update'], [self.DENS, cur, src, self.SDF, new, self.bank],
              gb + Uniforms().v4(thr, w, prm.ppc).tobytes(), n)
        b.run_indirect(k['band_cull'], [self.parts, self.ctr, self.freelist, new, cur, self.cellcount, self.SDF, self.bank],
                       gb + Uniforms().v4(self.capacity, prm.ppc).tobytes(), self.args, 0)
        m = round(prm.ppc ** (1.0 / 3.0))
        u = gb + (Uniforms().v4(prm.ppc, self.capacity, self.steps * 3 + 1 + prm.seed * 131.0, m if m ** 3 == prm.ppc else 1)
                  .v4(1.0 if carry else 0.0).tobytes())
        b.run(k['band_reseed'], [self.parts, self.ctr, self.freelist, cur, new, self.SDF, vel,
                                 self.attr if carry else self._noattr, self.cellcount, self.bank], u, n)
        self._band_i = 1 - self._band_i

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
        # the sources reach over the tallest crest; the kernel keeps to the surface
        crest = min(ny * h - top, self._sea_crest(prm)) if prm.ocean is not None else 0.0
        top_s = top + max(crest, 0.0)
        if not self._level_filled:
            self._level_filled = True
            out.append(source('box', (cx, oy + 0.5 * top_s, cz), (0.5 * nx * h, 0.5 * top_s, 0.5 * nz * h), fill=True,
                              jitter=0.0, vel_blend=0.0, top_up=-1.0))
        top = top_s
        hi = top - h
        if hi > h:
            ym = oy + 0.5 * hi
            strips = (((ox + 0.5 * h, ym, cz), (0.5 * h, 0.5 * hi, 0.5 * nz * h)),
                      ((ox + (nx - 0.5) * h, ym, cz), (0.5 * h, 0.5 * hi, 0.5 * nz * h)),
                      ((cx, ym, oz + 0.5 * h), (0.5 * nx * h, 0.5 * hi, 0.5 * h)),
                      ((cx, ym, oz + (nz - 0.5) * h), (0.5 * nx * h, 0.5 * hi, 0.5 * h)))
            # the sea: its sides are topped up right to the sea's own surface, wherever a cell is a
            # quarter short (the flow in and out through them evens the box's level with the sea's:
            # liq_forces.wgsl, absorb_flow; filled to the brim every step they overfilled it)
            top_up = 0.15 if self._sea_on(prm) else 0.5
            for (pos, size), side in zip(strips, prm.sea_sides):
                if side > 0.5:   # only where the open water flows in (not along a flume's walls)
                    out.append(source('box', pos, size, jitter=0.0, vel_blend=0.0, top_up=top_up))
        return out

    def shift(self, b: Batch, dx: int, dz: int, prm: LiquidParams | None = None):
        """Move the box by whole cells along x and z (it follows the liquid, or a boat): the liquid, its
        whitewater, the deep liquid of the narrow band and the ground's wetness keep their place in the
        world; what the box leaves behind is dropped, and the open water fills what it moves over at
        once (left empty for a step, the liquid would slump into it and the box slowly drain)."""
        dx, dz = int(dx), int(dz)
        if not (dx or dz):
            return
        prm = prm or self._prm
        k = self._k
        n = self.dims
        gb = self._grid(0.0).tobytes()
        d = Uniforms().v4(dx, 0.0, dz, self.capacity).tobytes()
        b.clear_buffer(self.cellcount)
        b.run_indirect(k['shift'], [self.parts, self.ctr, self.freelist, self.cellcount], gb + d, self.args, 0)
        if self.ww_capacity > 64:
            b.run(k['shift_ww'], [self.WA], gb + Uniforms().v4(dx, 0.0, dz, self.ww_capacity).tobytes(),
                  groups=groups_1d(self.ww_capacity))
        if self.WET2 is None:
            self.WET2 = self.gpu.texture2d(n[0], n[2], 'r32float', 'liq-wet-shift')
            self._tex.append(self.WET2)
        band = self.band is not None
        src = self.band[self._band_i] if band else self._noband
        dst = self.band[1 - self._band_i] if band else self._noattr
        b.run(k['shift_band'], [src, dst, self.WET, self.WET2], gb + Uniforms().v4(dx, 0.0, dz, 1.0 if band else 0.0).tobytes(), n)
        b.copy_texture(self.WET2, self.WET, (n[0], n[2], 1))
        if band:
            self._band_i = 1 - self._band_i
        su = gb + Uniforms().v4(dx, 0.0, dz).tobytes()
        if self.vel_tex is not None:
            # the velocity the deep liquid carries into the next step
            vdst = self.VB if self.vel_tex is self.VA else self.VA
            b.run(k['shift_vel'], [self.vel_tex, vdst], su, tuple(x + 1 for x in n))
            self.vel_tex = vdst
        # the pressure the next solve starts from
        b.run(k['shift_scalar'], [self.X, self.Z], su, n)
        b.copy_texture(self.Z, self.X, n)
        h = self.h
        ox, oy, oz = self.origin
        self.origin = (ox + dx * h, oy, oz + dz * h)
        self._strips.append((dx, dz))
        self._write_sdf(b)
        # the open water over the strip it moved onto, now
        srcs = self._strip_sources(prm)
        if srcs:
            gb = self._grid(0.0, prm).tobytes()
            self._sea(b, prm, gb)
            band = bool(prm.narrow_band) and self.band is not None
            carry = self.attr is not None and (self._dye_on or self._rho_on)
            cur = tuple(float(x) for x in prm.current) if self.level_on(prm) else (0.0, 0.0, 0.0)
            m = round(prm.ppc ** (1.0 / 3.0))
            u = (self._grid(0.0, prm).v4(prm.ppc, self.capacity, self.steps * 5 + 3 + prm.seed * 7919.0, m if m ** 3 == prm.ppc else 1)
                 .v4(1.0 if band else 0.0, max(1, int(prm.band_width)), self._sea_kp, 1.0 if self._sea_on(prm) else 0.0)
                 .v4(*cur, 1.0 if carry else 0.0))
            pack_emitters(u, srcs[:MAX_EMITTERS], meshes=self.meshes)
            b.run(k['spawn'], [self.parts, self.ctr, self.freelist, self.cellcount, self.SDF, self.meshes.atlas,
                               self.band[self._band_i] if band else self._noband, self.OCN,
                               self.attr if carry else self._noattr], u, n)
            b.run(k['indirect'], [self.ctr, self.args], Uniforms().v4(0, self.capacity, 0), groups=(1, 1, 1))
            therm = self._thermal(prm)
            if therm is not None:
                therm.born(b, prm, 0.0, srcs)

    def _strip_sources(self, prm: LiquidParams):
        """After the box moved: the open water's fill over the strips it moved over (once)."""
        if not self._strips:
            return []
        strips, self._strips = self._strips, []
        if not self.level_on(prm):
            return []
        nx, ny, nz = self.dims
        h = self.h
        ox, oy, oz = self.origin
        top = min(prm.water_level, ny * h)
        crest = min(ny * h - top, self._sea_crest(prm)) if prm.ocean is not None else 0.0
        top_s = top + max(crest, 0.0)
        out = []
        sides = prm.sea_sides
        for dx, dz in strips:
            # (only across a side the open water flows through: past a flume's wall there is none)
            if dx and sides[1 if dx > 0 else 0] < 0.5:
                dx = 0
            if dz and sides[3 if dz > 0 else 2] < 0.5:
                dz = 0
            if dx:
                w = min(abs(dx), nx)
                x0 = nx - w if dx > 0 else 0
                out.append(source('box', (ox + (x0 + 0.5 * w) * h, oy + 0.5 * top_s, oz + 0.5 * nz * h),
                                  (0.5 * w * h, 0.5 * top_s, 0.5 * nz * h), fill=True, jitter=0.0, vel_blend=0.0, top_up=-1.0))
            if dz:
                w = min(abs(dz), nz)
                z0 = nz - w if dz > 0 else 0
                out.append(source('box', (ox + 0.5 * nx * h, oy + 0.5 * top_s, oz + (z0 + 0.5 * w) * h),
                                  (0.5 * nx * h, 0.5 * top_s, 0.5 * w * h), fill=True, jitter=0.0, vel_blend=0.0, top_up=-1.0))
        return out

    def _leave_mask(self, prm, cur):
        """The open sides (bits -x, +x, -z, +z) the liquid may leave through under the level: those the
        sea flows through, and the one a current runs out of; bits 4-7, the sides that are walls all the
        way up (a wave flume's, _flume_walls)."""
        m = 0
        if self._sea_on(prm):
            m |= sum(1 << i for i, s in enumerate(prm.sea_sides) if s > 0.5)
        if max(abs(cur[0]), abs(cur[2])) > 1e-4:
            m |= sum(1 << i for i, s in enumerate(prm.sea_sides) if s > 0.5)
            m |= (2 if cur[0] > 1e-4 else 1 if cur[0] < -1e-4 else 0) | (8 if cur[2] > 1e-4 else 4 if cur[2] < -1e-4 else 0)
        return float(m | (self._flume_walls(prm) << 4))

    def _flume_walls(self, prm):
        """Bits (-x, +x, -z, +z) of the sides that are walls all the way up: with the sea's waves let in
        only where they come from, those along the waves are a wave flume's walls (sea_sides -1). Open
        above the level, a crest or a run-up along them would spill out of the box, and it would drain a
        little with every wave."""
        if not (self.level_on(prm) and self._sea_on(prm)):
            return 0
        return sum(1 << i for i, s in enumerate(prm.sea_sides) if s < -0.5)

    def _sea_on(self, prm):
        """The sea's waves are on: its water flows in and out through the open sides."""
        return self.level_on(prm) and prm.ocean is not None and prm.ocean.on

    def _sea_crest(self, prm):
        from .ocean import crest_bound
        return crest_bound(prm.ocean)

    def _sea(self, b, prm, gb):
        """The open water's surface and velocity per column this step (OCN), from the waves the grid
        can carry (ocean.py)."""
        if self._seabc is None:
            from .ocean import SeaBoundary
            self._seabc = SeaBoundary(self.gpu)
        spec = prm.ocean if (prm.ocean is not None and prm.ocean.on) else None
        cur = tuple(float(x) for x in prm.current) if self.level_on(prm) else (0.0, 0.0, 0.0)
        self._sea_kp = self._seabc.write(b, self.OCN, gb, self.dims, prm.water_level / self.h, spec, prm.clock,
                                         (cur[0], cur[2]), self.h, self.origin)

    def _viscosity(self, b, dt, prm, rho0):
        """Implicit viscosity on VA (Jacobi, an even number of sweeps); returns the texture holding
        the result."""
        nu = prm.viscosity / max(prm.rho, 1.0)
        a = nu * dt / (self.h * self.h)
        cool = prm.cooling > 0.0 and prm.solidify > 1.0
        if a <= 1e-6 and not cool:
            return self.VA
        a = max(a, 1e-4) if cool else a
        if self.VVISC is None:
            self.VVISC = self._t3(tuple(x + 1 for x in self.dims), 'rgba32float', 'liq-viscous')
        iters = int(min(64, max(8, 8 + 6 * math.sqrt(a * (prm.solidify if cool else 1.0) ** 0.25))))
        iters += iters % 2
        u = (self._grid(dt, prm).v4(a, prm.surface_density * rho0, math.log(max(prm.solidify, 1.0)) if cool else 0.0,
                                    1.0 if cool else 0.0)).tobytes()
        m = tuple(x + 1 for x in self.dims)
        cur = self.VA
        for i in range(iters):
            dst = self.VB if i % 2 == 0 else self.VVISC
            b.run(self._k['visc'], [self.VA, cur, self.DENS, dst, self.HEAT], u, m)
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
        rho0 = float(prm.ppc)
        band = bool(prm.narrow_band)
        if band:
            self._ensure_band(b)
        carry = self._carry(b, srcs)
        cur = tuple(float(x) for x in prm.current) if self.level_on(prm) else (0.0, 0.0, 0.0)

        # 1. particles -> grid
        b.clear_buffer(self.acc)
        b.run_indirect(k['p2g'], [self.parts, self.acc], gb + Uniforms().v4(cap, apic, prm.speed_limit).tobytes(), self.args, 0)
        b.run(k['norm'], [self.acc, self.VOLD, self.DENS, self.HEAT], gb, m)
        if band:
            # the deep liquid: full, and moving as the grid did at the end of the last step
            b.run(k['band_merge'], [self.VOLD, self.vel_tex, self.band[self._band_i], self.DENS, self.VMRG],
                  gb + Uniforms().v4(rho0).tobytes(), m)
            self.VOLD, self.VMRG = self.VMRG, self.VOLD
        if carry:
            # what the particles carry, per cell: dye, and the buoyancy between liquids
            b.clear_buffer(self.aacc)
            b.run_indirect(k['attr_p2g'], [self.parts, self.attr, self.aacc], gb + Uniforms().v4(cap).tobytes(), self.args, 0)
            b.run(k['attr_norm'], [self.aacc, self.DENS, self.ATTR, self.RHO], gb + Uniforms().v4(rho0).tobytes(), n)
            if self._rho_on:
                b.run(k['attr_buoy'], [self.RHO, self.BUOY], gb + Uniforms().v4(2.0).tobytes(), n)
        buoy = self.BUOY if (carry and self._rho_on) else None
        therm = self._thermal(prm)
        if therm is not None:
            # heat: each cell's change this step, and the buoyancy of heat, ice and steam bubbles
            buoy = therm.heat(b, prm, dt, srcs, buoy)

        # 2. forces and boundaries
        wind = tuple(float(x) for x in prm.wind)
        wspeed = math.sqrt(sum(x * x for x in wind))
        level = prm.water_level / self.h if self.level_on(prm) else -1.0
        wcells = max(1.0, float(prm.level_absorb))
        calm = 6.0 * math.sqrt(prm.gravity / (wcells * self.h)) if prm.gravity > 0 else 0.0
        if level >= 0.0:
            self._sea(b, prm, gb)
        u = (self._grid(dt, prm).v4(prm.gravity)
             .v4(*wind, 0.0006 * prm.wind_surface * wspeed / self.h)
             .v4(level, wcells, calm, prm.surface_density * rho0)
             .v4(self._sea_kp, cur[0], cur[2], 1.0 if buoy is not None else 0.0)
             .v4(*prm.sea_sides))
        pack_emitters(u, srcs, meshes=self.meshes)
        pack_colliders(u, self.colliders, self.meshes)
        b.run(k['forces'], [self.VOLD, self.SDF, atlas, self.VA, self.DENS, self.OCN,
                            buoy if buoy is not None else self._zero3], u, m)

        # 3. a little extrapolation so every face of a liquid cell has a value
        b.run(k['extrap'], [self.VA, self.VB], gb, m)
        b.run(k['extrap'], [self.VB, self.VA], gb, m)

        # viscosity
        vin = self._viscosity(b, dt, prm, rho0) if (prm.viscosity > 0 or prm.cooling > 0) else self.VA

        # 4. pressure
        ku = gb + (Uniforms().v4(prm.surface_density * rho0, rho0, prm.volume_correction, prm.theta_min)
                   .v4(prm.compression, prm.surface_tension / prm.rho * dt)
                   .v4(level, float(self._flume_walls(prm))).tobytes())
        ww = prm.whitewater and self.ww_capacity > 64
        if prm.surface_tension > 0 or ww:
            ca = math.radians(prm.contact_angle)
            on = 1.0 if abs(prm.contact_angle - 90.0) > 0.5 else 0.0
            b.run(k['normal'], [self.DENS, self.SDF, self.NRM], gb + Uniforms().v4(rho0, math.cos(ca), math.sin(ca), on).tobytes(), n)
            b.run(k['curv'], [self.NRM, self.KAPPA], gb + Uniforms().v4(0.05).tobytes(), n)
        b.run(k['setup'], [vin, self.DENS, self.SDF, self.X, self.TYPE[0], self.CO[0], self.R, self.KAPPA, self.OCN], ku, n)
        self._pressure(b, prm)
        pout = self.VA if vin is self.VB else self.VB
        b.run(k['project'], [vin, self.VOLD, self.X, self.TYPE[0], self.DENS, pout, self.KAPPA, self.OCN], ku, m)
        if therm is not None:
            # ice moves as rigid pieces
            pout = therm.ice_rigid(b, prm, pout, self.VA if pout is self.VB else self.VB)

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
                  .v4(prm.damping, level, prm.cooling, self._leave_mask(prm, cur)).tobytes())
        if therm is not None:
            therm.hide(b, prm)
        b.run_indirect(k['g2p'], [self.parts, src, self.VOLD, self.SDF, self.ctr, self.freelist, self.cellcount,
                                  self.DENS, self.WA, self.WB, self.wctr, self.NRM, self.KAPPA, self.OCN], u, self.args, 0)
        if therm is not None:
            # frozen particles move with their ice; every particle takes its heat and changes phase
            therm.after_move(b, prm, dt, ww)
        if carry and self._dye_on and prm.dye_mixing > 0.0:
            mix = 1.0 - math.exp(-prm.dye_mixing * dt)
            b.run_indirect(k['attr_g2p'], [self.parts, self.attr, self.ATTR], gb + Uniforms().v4(cap, mix).tobytes(), self.args, 0)
        if ww and prm.rain > 0.0 and prm.open_top:
            # raindrops landing on the liquid throw up crowns of spray
            per_cell = rain_drops_per_m2s(prm.rain, prm.rain_drop) * self.h * self.h * dt
            u = gb + (Uniforms().v4(per_cell, rain_speed(prm.rain_drop), self.ww_capacity, prm.surface_density * rho0)
                      .v4(self.steps * 7 + 3 + prm.seed * 131.0, 0.25).tobytes())
            b.run(k['rain'], [self.DENS, self.SDF, self.WA, self.WB, self.wctr], u, (n[0], n[2], 1))
        if ww:
            u = gb + (Uniforms().v4(self.ww_capacity, rho0, prm.gravity, prm.bubble_rise)
                      .v4(prm.spray_drag, prm.bubble_drag, 0.3, 0.85)
                      .v4(*wind, 0.03 * prm.wind_surface).tobytes())
            b.run(k['ww'], [self.WA, self.WB, src, self.DENS, self.SDF], u, groups=groups_1d(self.ww_capacity))
        if therm is not None:
            # steam bubbles rise, condense or burst; what the liquid gave the gas, for the fire solver
            therm.steam(b, prm, dt, src, ww)
        b.run(k['wet'], [self.DENS, self.WET], gb + Uniforms().v4(0.25 * rho0, prm.drying).tobytes(), (n[0], n[2], 1))
        if band:
            self._band_update(b, prm, src, rho0, gb)

        # 7. sources
        if srcs:
            m = round(prm.ppc ** (1.0 / 3.0))
            u = (self._grid(dt, prm).v4(prm.ppc, cap, self.steps + prm.seed * 7919.0, m if m ** 3 == prm.ppc else 1)
                 .v4(1.0 if band else 0.0, max(1, int(prm.band_width)), self._sea_kp, 1.0 if self._sea_on(prm) else 0.0)
                 .v4(*cur, 1.0 if carry else 0.0))
            pack_emitters(u, srcs, meshes=self.meshes)
            b.run(k['spawn'], [self.parts, self.ctr, self.freelist, self.cellcount, self.SDF, atlas,
                               self.band[self._band_i] if band else self._noband, self.OCN,
                               self.attr if carry else self._noattr], u, n)
        b.run(k['indirect'], [self.ctr, self.args], Uniforms().v4(0, cap, 0), groups=(1, 1, 1))
        if therm is not None:
            therm.born(b, prm, dt, srcs)

        self.time += dt
        self.steps += 1
        if prm.cooling > 0.0:
            # a molten liquid: its crust field carried along by the flow, and its particles' heat evened out
            # (liq_crust_adv.wgsl, liq_heat_mix.wgsl)
            self._ensure_crust(b)
            self._crust_on = True
            last, _ = crust_restarts(self.time)
            moment = last > self.time - dt
            b.run(k['crust_adv'], [self.vel_tex, self.DENS, self.HEAT, self.CX[0], self.CA[0], self.CX[1], self.CA[1],
                                   self.CV], self._grid(dt, prm).v4(1.0 if moment else 0.0, CRUST_LATCH, rho0), n)
            self.CX.reverse()
            self.CA.reverse()
            hmix = 1.0 - math.exp(-HEAT_MIXING * dt)
            b.run_indirect(k['heat_mix'], [self.parts, self.HEAT, self.DENS], self._grid(dt, prm).v4(cap, hmix),
                           self.args, 0)

    def _thermal(self, prm):
        """The heat model (liquid_thermal.Thermal) when the scene has it on, else None."""
        if not prm.thermal:
            return None
        if self.thermal is None:
            from .liquid_thermal import Thermal
            self.thermal = Thermal(self)
        return self.thermal

    @property
    def gas_flux(self):
        """What the liquid gives the gas each substep (liquid_thermal.py, liq_therm_gas.wgsl), on the gas
        grid, or None without heat."""
        t = self.thermal
        return t.gas_flux if (t is not None and self._prm.thermal and t.therm is not None) else None

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
            tq = s[:, 3:6].sum(0) / 4096.0 / nsub
            nv = s[:, 9].sum()
            v = s[:, 6:9].sum(0) / 1024.0 / nv if nv > 0 else np.zeros(3)
            cells = s[:, 13].sum() / nsub
            if s[:, 12].sum() > 0:
                # seepage under a body resting on the ground (see liq_float.wgsl)
                f[1] += s[:, 10].sum() / nsub * (s[:, 11].sum() / 65536.0) / s[:, 12].sum()
            out.append({'force': f, 'moment': tq, 'liquid_vel': v, 'wet': nv > 0, 'cells': cells})
        return out

    # -- output ----------------------------------------------------------------------------------

    def pack(self, b: Batch):
        """Pack the live particles (16 bytes each, and 8 of dye) for the surface builder and the frame cache."""
        b.clear_buffer(self.ctr, 8, 4)
        dye = self._dye_on and self.attr is not None
        ice = self.ice_buffer is not None
        b.run_indirect(self._k['pack'], [self.parts, self.packed, self.ctr, self.attr if dye else self._noattr,
                                         self.pattr if dye else self._noband,
                                         self.thermal.therm if ice else self._noice, self.thermal.pice if ice else self._noice2],
                       self._grid(0.0).v4(self.capacity, 1.0 if dye else 0.0, 0.0, 1.0 if ice else 0.0),
                       self.args, 0)
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
        if self.thermal is not None and self._prm.thermal:
            self.thermal.read_stats()
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

    @property
    def dye_buffer(self):
        """The packed particles' dye (8 bytes each, same order), or None without dye."""
        return self.pattr if (self._dye_on and self.attr is not None) else None

    def read_dye(self):
        """(n, 2) uint32 dye of the packed particles of the last pack(), or None without dye."""
        if self.dye_buffer is None:
            return None
        if not self.packed_count:
            return np.zeros((0, 2), np.uint32)
        data = self.gpu.read_buffer(self.pattr, self.packed_count * 8)
        return np.frombuffer(data, np.uint32).reshape(-1, 2).copy()

    @property
    def crust_field(self):
        """A molten liquid's crust field for the renderer (rgba32float on the grid: the crust coordinate less the
        place, in cells, and the skin's age in seconds; liq_crust_adv.wgsl), or None."""
        return self.CV if (self._crust_on and self.CX is not None) else None

    def read_crust(self):
        """The crust field as (nz, ny, nx, 4) float32, or None."""
        if self.crust_field is None:
            return None
        return np.ascontiguousarray(self.gpu.read(self.CV), np.float32)

    @property
    def ice_buffer(self):
        """The packed particles' frozen share and ice cloudiness (u32 each, two 16-bit fractions, same order),
        or None without heat."""
        t = self.thermal
        return t.pice if (t is not None and self._prm.thermal and t.therm is not None) else None

    def read_ice(self):
        """(n,) uint32 ice of the packed particles of the last pack(), or None without heat."""
        if self.ice_buffer is None:
            return None
        if not self.packed_count:
            return np.zeros((0,), np.uint32)
        return np.frombuffer(self.gpu.read_buffer(self.thermal.pice, self.packed_count * 4), np.uint32).copy()

    @property
    def thermal_stats(self):
        """The heat's statistics of the last measured frame (liquid_thermal.Thermal.read_stats), or {}."""
        return dict(self.thermal.stats) if (self.thermal is not None and self._prm.thermal) else {}

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
