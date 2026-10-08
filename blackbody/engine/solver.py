"""Pyro solver: an incompressible, buoyant, reacting gas on a MAC grid, entirely on the GPU.

Per substep: MacCormack advection of scalars and velocity, burnable surfaces, sources and
combustion, vorticity confinement / turbulence / disturbance, buoyancy, wind and swirl, then a
multigrid pressure projection that includes the gas expansion released by burning fuel. Colliders
may move: their distance field is rebuilt every substep and their surfaces push the gas.

Scalar fields (one rgba16float texture): x = temperature, y = fuel, z = soot, w = flame.
Optional fields, allocated only when a scene uses them:
  aux  (rgba16float): x = share of the air's oxygen used up, y = water vapour (g/m^3)
  chem (rgba16float): rgb = flame colourant
  burn (rgba16float): burnable surfaces (see burn.wgsl)
  water (rgba16float): liquid water from a liquid simulation sharing the box (solver.water, written
    each step by both_wet.wgsl): x = share of the cell filled with it (it puts fire out like a hose),
    y = how soaked the fuel is, z = steam off a hot wet fuel bed (g/m^3/s), w = smoke from a doused bed
  burn_obj (rgba16float atlas): burnable colliders' surfaces, one region per collider in its own
    frame, so a burning object can move
With upres, fire and smoke are also carried on a grid `upres` times finer (scal_fine), moved by the
simulated air plus sub-grid swirls, and burned there too; renders use it for the finer detail.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

import numpy as np

from .gpu import GPU, BU, Uniforms, Batch, ceil_div
from .mesh import NO_ANIM, NO_MESH, MeshLibrary
from .renderer import vapour_saturation

SHAPES = {'sphere': 0, 'box': 1, 'cylinder': 2, 'capsule': 3, 'ring': 4, 'cone': 5, 'mesh': 6, 'volume': 7}
VOLUME_MODES = {'source': 1, 'fill': 2, 'hold': 3}   # a Volume emitter releases at its rates, fills the box once, or keeps it topped up
COLLIDER_SHAPES = {'sphere': 0, 'box': 1, 'cylinder': 2, 'mesh': 3}
MAX_EMITTERS = 16   # matches MAX_EMITTERS in emitters.wgsl
MAX_COLLIDERS = 16  # matches MAX_COLLIDERS in colliders.wgsl
EMITTER_VEC4 = 14   # vec4s per Emitter struct (emitters.wgsl)
COLLIDER_VEC4 = 11  # vec4s per Collider struct (colliders.wgsl)
# Latent heat of condensing water: 2.26 MJ/kg into air of about 1.2 kg/m^3 and 1005 J/(kg K)
LATENT_K_PER_G = 2.26e6 / (1.2 * 1005.0) / 1000.0  # Kelvin per g/m^3 condensed
# Water vapour is lighter than air (18 against 29 g/mol): lift per g/m^3 of vapour, taking pure
# steam at 100 C (about 590 g/m^3) as the reference.
VAPOUR_LIFT = 9.81 * (1.0 - 18.0 / 29.0) / 590.0
NO_FEATURES = {'oxygen': False, 'vapour': False, 'chem': False, 'burn': False, 'aux': False, 'water': False,
               'stain': False}


@dataclass
class SpreadParams:
    enabled: bool = False
    ground: bool = True
    area: tuple = (0.0, 0.0)       # half width, half depth of the burnable floor (m); 0 = all of it
    coverage: float = 0.85
    patch_freq: float = 1.5
    burn_time: float = 3.0
    fuel: float = 8.0
    heat: float = 0.45
    smoke: float = 0.5
    catch_temp: float = 0.35
    catch_time: float = 0.4
    creep: float = 0.05
    smoulder: float = 4.0
    smoulder_smoke: float = 1.0
    dry_time: float = 20.0         # s for a soaked surface to dry out before it can catch again (0 = never soaked)
    spotting: float = 0.0          # chance that a hot ember landing on the burnable floor starts a spot fire
    spot_temp: float = 900.0       # K: embers cooler than this land without starting anything


@dataclass
class SolverParams:
    # combustion
    ignition: float = 0.25
    burn_rate: float = 6.0
    heat: float = 0.6
    soot: float = 0.3
    flame_gain: float = 1.0
    flame_life: float = 0.08
    expansion: float = 0.5
    radiative: float = 0.0
    cooling: float = 2.0
    smoke_dissipation: float = 0.4
    fuel_dissipation: float = 0.5
    temp_cap: float = 3.0
    rich: float = 4.0              # fuel level at which lack of mixing halves the burn rate
    flame_speed: float = 0.0       # m/s a premixed flame front runs through fuel mixed with air (0 = off)
    fuel_weight: float = 0.0       # m/s^2 per unit fuel: heavier-than-air vapour sinks and spreads along the ground
    soot_stain: float = 0.0        # 1/s: how fast smoke leaves soot on walls, ceilings and the floor (0 = off)
    expansion_cap: float = 40.0    # 1/s, upper limit on gas expansion per cell
    sponge: float = 8.0            # cells of absorbing layer at open boundaries
    sponge_strength: float = 6.0   # 1/s at the boundary
    # air and water
    oxygen: bool = False           # track the air's oxygen (closed rooms, backdraft)
    air_use: float = 1.0
    air_mixing: float = 0.03       # m^2/s, turbulent diffusivity of used-up air (sub-grid mixing)
    vapour: bool = False           # carry water vapour (steam)
    steam_yield: float = 150.0     # g/m^3 of vapour per unit of temperature removed by dousing
    water_yield: float = 0.0       # g/m^3 of vapour per unit of fuel burned
    vapour_dissipation: float = 1.5
    water_douse: float = 40.0      # 1/s: how fast a cell full of liquid water puts fire out
    latent_heat: float = 1.0       # share of the physical heat released by condensing steam
    thermal_expansion: float = 0.0  # share of the physical swelling of gas as it heats (and shrinking as it cools)
    ambient_k: float = 300.0       # the air's temperature (K) and the flame temperature at T = 1 (K):
    flame_k: float = 1650.0        #   the temperature field's scale, for expansion and condensation
    humidity: float = 50.0         # % relative humidity of the air
    upres_turbulence: float = 1.0  # strength of the sub-grid swirls on the upres grid
    boil_temp: float = 0.055       # boiling point of water on the temperature field's scale
    steam_expansion: float = 0.0   # with liquid in the box: share of the physical swelling of water flashing to steam
    steam_inert: tuple = (0.25, 0.5)  # with liquid in the box: steam's share of the gas where flames start to starve, and go out
    # forces
    buoyancy: float = 5.0
    soot_weight: float = 0.02
    damping: float = 0.05
    vorticity: float = 1.5
    vort_outside: float = 0.3
    turbulence: float = 3.0
    turb_freq: float = 2.5
    turb_rise: float = 1.5
    turb_evolve: float = 0.8
    turb_octave2: float = 0.5
    disturbance: float = 2.0
    disturb_block: float = 0.05
    disturb_rate: float = 24.0
    mask_temp: float = 1.0
    mask_flame: float = 1.0
    mask_smoke: float = 0.0
    wind: tuple = (0.0, 0.0, 0.0)
    wind_relax: float = 0.4
    # numerics
    maccormack: float = 1.0        # scalars
    maccormack_vel: float = 0.5    # velocity (full strength can go unstable in expanding flow)
    mg_cycles: int = 2
    mg_pre: int = 2
    mg_post: int = 2
    sor: float = 1.0
    seed: float = 0.0
    # boundaries
    open_sides: bool = True
    open_top: bool = True
    ground: bool = True
    spread: SpreadParams = field(default_factory=SpreadParams)


@dataclass
class EmitterGPU:
    shape: str = 'sphere'
    pos: tuple = (0.0, 0.2, 0.0)
    size: tuple = (0.2, 0.2, 0.2)
    p1: tuple = (0.0, 0.0, 0.0)
    soft: float = 0.0
    fuel: float = 10.0
    temp: float = 1.0
    smoke: float = 0.0
    vel: tuple = (0.0, 0.0, 0.0)
    radial: float = 0.0
    vel_blend: float = 0.0
    noise: float = 0.6
    noise_freq: float = 4.0
    noise_rise: float = 1.0
    contrast: float = 1.0
    seed: float = 0.0
    yaw: float = 0.0               # radians
    swirl: float = 0.0             # m/s at the core edge
    swirl_width: float = 4.0       # reach, in core radii
    douse: float = 0.0             # 1/s
    vapour: float = 0.0            # g/m^3 in the released gas
    color: tuple = (0.0, 0.0, 0.0)  # colourant per second (colour * amount)
    thickness: float = 0.0         # mesh surface depth (m)
    mesh: str = ''                 # resolved mesh path
    mesh_frame: float | None = None  # frame of a deforming mesh (numbered sequence or animated USD prim)
    mesh_fps: float = 24.0         # its frames per second of simulation time
    volume_mode: int = 0           # Volume emitters: 1 releases at its rates where the volume is dense, 2 fills (VOLUME_MODES)
    volume_vel: float = 0.0        # Volume emitters: how strongly the air takes up the volume's own velocity (0..1)
    quat: tuple = (0.0, 0.0, 0.0, 1.0)   # orientation (x, y, z, w) after the yaw: carried by a tipped or tumbling object


@dataclass
class ColliderGPU:
    shape: str = 'box'
    pos: tuple = (0.0, 0.0, 0.0)
    size: tuple = (0.5, 0.5, 0.5)
    rot_y: float = 0.0
    vel: tuple = (0.0, 0.0, 0.0)   # m/s
    spin: float = 0.0              # radians/s about y
    burnable: bool = False
    mesh: str = ''
    hollow: float = 0.0                  # wall thickness (m); 0 = solid
    opening: tuple = (0.0, 0.0, 0.0)     # half size of a box cut out of it (m); 0 = none
    opening_at: tuple = (0.0, 0.0, 0.0)  # centre of the cut-out, in the collider's own frame (m)
    holdout: bool = True                 # hides the fire behind it in renders
    burn_slot: int = -1                  # its region in the solver's surface-burn atlas (set by the solver)
    mesh_frame: float | None = None      # frame of a deforming mesh (numbered sequence or animated USD prim)
    mesh_fps: float = 24.0               # its frames per second of simulation time
    quat: tuple = (0.0, 0.0, 0.0, 1.0)   # orientation (x, y, z, w) after the yaw: a tumbling object
    omega: tuple = (0.0, 0.0, 0.0, 0.0)  # angular velocity (rad/s, world axes; w unused)

    @property
    def moving(self):
        return (any(abs(x) > 1e-6 for x in self.vel) or abs(self.spin) > 1e-6 or self.mesh_frame is not None
                or any(abs(x) > 1e-6 for x in self.omega[:3]))


def _mesh_ref(meshes, path, frame=None, fps=24.0):
    """(m0, m1, m2, anim) of a mesh in the atlas (m0.w its atlas code: meshsdf.wgsl atlas_org; negative: none), anim =
    (next frame's atlas code or -1, blend, fps, 0)."""
    ref = meshes.ref_at(path, frame) if (meshes is not None and path) else None
    if ref is None:
        return NO_MESH + (NO_ANIM,)
    m0, m1, m2, an = ref
    return m0, m1, m2, (an[0], an[1], fps if an[0] >= 0 else 0.0, 0.0)


def _quat_turn(q, v):
    """Vector v turned by the unit quaternion q (x, y, z, w), as quat_rotate in colliders.wgsl."""
    u, w = np.asarray(q[:3], float), float(q[3])
    v = np.asarray(v, float)
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def emitter_extent(e: EmitterGPU, meshes=None):
    """Swirl axis (x, z), core radius and base height of an emitter, in world metres."""
    s = np.asarray(e.size, float)
    p = np.asarray(e.pos, float)
    if e.shape == 'capsule':
        q = np.asarray(e.p1, float)
        mid = 0.5 * (p + q)
        return mid[0], mid[2], max(s[0], 0.5 * float(np.linalg.norm((q - p)[[0, 2]]))), min(p[1], q[1]) - s[0]
    if e.shape in ('mesh', 'volume'):
        b = meshes.bounds(e.mesh) if meshes is not None else None
        if b is not None:
            lo, hi = np.asarray(b[0]) * s, np.asarray(b[1]) * s
            c = 0.5 * (lo + hi)
            cs, sn = math.cos(e.yaw), math.sin(e.yaw)
            c = _quat_turn(e.quat, (cs * c[0] + sn * c[2], c[1], -sn * c[0] + cs * c[2]))
            return p[0] + c[0], p[2] + c[2], max(0.5 * max(hi[0] - lo[0], hi[2] - lo[2]), 1e-3), p[1] + lo[1]
        return p[0], p[2], 0.5, p[1]
    if e.shape == 'ring':
        return p[0], p[2], s[0] + s[1], p[1] - s[1]
    return p[0], p[2], max(s[0], s[2]), p[1] - s[1]


def collider_scale(c: ColliderGPU):
    """What a collider's shape is scaled by along its own axes (m); matches col_scale in colliders.wgsl."""
    s = np.maximum(np.asarray(c.size, float), 1e-4)
    if c.shape == 'sphere':
        return np.full(3, s[0])
    if c.shape == 'cylinder':
        return np.array([s[0], s[1], s[0]])
    return s


def collider_extent(c: ColliderGPU, meshes=None):
    """The collider's bounding box in its own frame (m), before hollowing: (lo, hi)."""
    s = np.asarray(c.size, float)
    if c.shape == 'box':
        return -s, s
    if c.shape == 'sphere':
        return -np.full(3, s[0]), np.full(3, s[0])
    if c.shape == 'cylinder':
        r = np.array([s[0], s[1], s[0]])
        return -r, r
    b = meshes.bounds(c.mesh) if (meshes is not None and c.mesh) else None
    if b is not None:
        return np.asarray(b[0]) * s, np.asarray(b[1]) * s
    return -np.full(3, 0.5), np.full(3, 0.5)


def pack_emitters(u: Uniforms, emitters, extra1=0.0, extra2=0.0, meshes=None):
    em = list(emitters)[:MAX_EMITTERS]
    fuels = [max(e.fuel, 0.0) for e in em]
    mean_fuel = (sum(fuels) / max(1, sum(1 for f in fuels if f > 0))) if any(fuels) else 1.0
    u.v4(len(em), extra1, extra2)
    for e in em:
        cx, cz, core, base = emitter_extent(e, meshes)
        m0, m1, m2, an = _mesh_ref(meshes, e.mesh, e.mesh_frame, e.mesh_fps) if e.shape in ('mesh', 'volume') else NO_MESH + (NO_ANIM,)
        # a mesh's m1.w is its surface depth; a volume's says what layers it has (a temperature, a velocity)
        depth = m1[3] if e.shape == 'volume' else e.thickness
        if e.shape == 'volume':   # (its next frame and the blend toward it, its velocity's strength, its mode)
            an = (an[0], an[1], float(e.volume_vel), float(e.volume_mode))
        u.v4(*e.pos, SHAPES.get(e.shape, 0))
        u.v4(*e.size, e.soft)
        u.v4(*e.p1, e.noise)
        u.v4(e.fuel, e.temp, e.smoke, e.vel_blend)
        u.v4(*e.vel, e.radial)
        u.v4(e.noise_freq, e.noise_rise, e.seed, e.contrast)
        u.v4(e.swirl, cx, cz, core)
        u.v4(core * max(e.swirl_width, 1.0), base, e.yaw, e.douse)
        u.v4(*e.color, e.vapour)
        u.v4(*m0[:3], m0[3])
        u.v4(*m1[:3], depth)
        # embers come mostly from where the fuel is; fuel-less emitters (sparks) still get a share
        u.v4(*m2[:3], max(e.fuel, 0.0) if e.fuel > 0 else mean_fuel)
        u.v4(*an)
        u.v4(*e.quat)
    for _ in range(MAX_EMITTERS - len(em)):
        for _ in range(EMITTER_VEC4 - 2):
            u.v4()
        u.v4(*NO_ANIM)
        u.v4(0.0, 0.0, 0.0, 1.0)
    return u


def pack_colliders(u: Uniforms, colliders, meshes=None, pieces=False):
    """pieces: the solid velocity texture has broken pieces in it this substep (apply.wgsl)."""
    cols = list(colliders or [])[:MAX_COLLIDERS]
    u.v4(len(cols), 1.0 if any(c.moving for c in cols) else 0.0, 1.0 if pieces else 0.0)
    for c in cols:
        m0, m1, m2, an = _mesh_ref(meshes, c.mesh, c.mesh_frame, c.mesh_fps) if c.shape == 'mesh' else NO_MESH + (NO_ANIM,)
        u.v4(*c.pos, COLLIDER_SHAPES.get(c.shape, 1))
        u.v4(*c.size, c.rot_y)
        u.v4(*c.vel, c.spin)
        u.v4(*m0[:3], m0[3])
        u.v4(*m1[:3], 1.0 if c.burnable else 0.0)
        u.v4(*m2[:3], c.burn_slot)
        u.v4(*c.opening_at, max(c.hollow, 0.0))
        u.v4(*c.opening, 1.0 if c.holdout else 0.0)
        u.v4(*an)
        u.v4(*c.quat)
        u.v4(*c.omega)
    for _ in range(MAX_COLLIDERS - len(cols)):
        for _ in range(COLLIDER_VEC4 - 3):
            u.v4()
        u.v4(*NO_ANIM)
        u.v4(0.0, 0.0, 0.0, 1.0)
        u.v4()
    return u


class Solver:
    def __init__(self, gpu: GPU):
        self.gpu = gpu
        self.dims = None
        self.h = 0.0
        self.origin = (0.0, 0.0, 0.0)
        self.time = 0.0
        self.steps = 0
        self.max_speed = 0.0
        self.bbox = None
        self.burning = 0
        self.levels = []
        self._tex = []
        self._opt = []
        self.colliders = []
        self.features = dict(NO_FEATURES)
        self._burn_dirty = True
        # the pieces of broken objects (bodyfield.py): (BodyField, substep) for the next step, or None
        self.pieces_step = None
        self._pieces_on = False
        self._had_pieces = False
        self.svel = None
        self._k = {}
        self.meshes = MeshLibrary(gpu)
        # stand-ins bound where an optional field is not in use (separate ones for reading and writing)
        self._dummy = {n: gpu.texture3d((1, 1, 1), 'rgba16float', f'unused-{n}') for n in ('aux_r', 'aux_w', 'chem_r', 'chem_w', 'burn_r')}
        for t in self._dummy.values():
            gpu.upload(t, np.zeros((1, 1, 1, 4), np.float16))
        for n in ('water', 'expo', 'stain'):
            self._dummy[n] = gpu.texture3d((1, 1, 1), 'r32float', f'unused-{n}')
            gpu.upload(self._dummy[n], np.zeros((1, 1, 1, 1), np.float32))
        self._no_slots = gpu.buffer(32, 'no-burn-slots')
        self._no_spots = gpu.buffer(32, 'no-spots')
        self.stain = None
        self.stain_obj = None      # soot on colliders, each in its own frame (stain_obj.wgsl)
        self.stain_slots = None
        self.stain_regions = 0
        self._stain_dirty = True
        self.spots = None
        self.spots_obj = None
        self.base = None         # the layout asked for: (dims, h, origin); a growing domain may be larger
        self.grow_count = 0
        self.water = None
        self.burn_obj = None
        self.burn_slots = None
        self.upres = 1
        self.scal_fine = None
        self._fine = []

    # -- allocation ---------------------------------------------------------------------------

    @staticmethod
    def dims_for(size_m, res, multiple=8):
        """Cells per axis for a domain of size (w, h, d) metres with about `res` cells on its longest
        side. Counts are rounded to multiples of 8 so the multigrid levels nest exactly."""
        w, h, d = size_m
        cell = max(w, h, d) / float(res)
        return tuple(max(multiple, int(round(x / cell / multiple)) * multiple) for x in (w, h, d)), cell

    def memory_bytes(self, dims=None, features=None):
        nx, ny, nz = dims or self.dims
        f = features or self.features
        cells = nx * ny * nz
        vb = 16 if self.gpu.vel_format == 'rgba32float' else 8
        faces = (nx + 1) * (ny + 1) * (nz + 1)
        extra = 8 * 2 * (int(f.get('aux', False)) + int(f.get('chem', False)) + int(f.get('burn', False))) + 4 * int(f.get('stain', False))
        fine = 28 * self.upres ** 3 if self.upres > 1 else 0
        return int(faces * vb * 3 + cells * (8 * 5 + 4 * 2 + 4 * 3 * 8 / 7 + extra + fine))

    def configure(self, dims, h, origin, features=None, upres=1):
        """Lay out the grid. Returns False when nothing changed; a domain that has grown (see grow)
        keeps its larger size as long as the layout asked for stays the same."""
        dims = tuple(int(x) for x in dims)
        feats = dict(NO_FEATURES, **(features or {}))
        upres = max(1, int(upres))
        base = (dims, float(h), tuple(float(x) for x in origin))
        if base == self.base and feats == self.features and upres == self.upres and self.dims is not None:
            return False
        self.base = base
        self.grow_count = 0
        same_shape = dims == self.dims
        self.h = float(h)
        self.origin = tuple(float(x) for x in origin)
        if not same_shape:
            self._release()
            self.dims = dims
            self._allocate()
        if not same_shape or feats != self.features:
            self.features = feats
            self._allocate_optional()
        if not same_shape or upres != self.upres:
            self.upres = upres
            self._allocate_fine()
        self._compile()
        self.reset()
        return True

    @property
    def dims_fine(self):
        return tuple(d * self.upres for d in self.dims)

    def _release(self):
        for t in self._tex + self._opt + self._fine:
            t.destroy()
        self.svel = None            # (made in _tex: gone with it)
        self._tex = []
        self._opt = []
        self._fine = []
        self.scal_fine = None

    def _t3(self, size, fmt, label, group=None):
        t = self.gpu.texture3d(size, fmt, label)
        (self._tex if group is None else group).append(t)
        return t

    def _allocate(self):
        nx, ny, nz = self.dims
        vd = (nx + 1, ny + 1, nz + 1)
        vf = self.gpu.vel_format
        self.vel = [self._t3(vd, vf, 'vel0'), self._t3(vd, vf, 'vel1')]
        self.vtmp = self._t3(vd, vf, 'vtmp')
        self.scal = [self._t3(self.dims, 'rgba16float', 'scal0'), self._t3(self.dims, 'rgba16float', 'scal1')]
        self.stmp = self._t3(self.dims, 'rgba16float', 'stmp')
        self.curl = self._t3(self.dims, 'rgba16float', 'curl')
        self.force = self._t3(self.dims, 'rgba16float', 'force')
        self.expo = self._t3(self.dims, 'r32float', 'expansion')
        self.sdf = self._t3(self.dims, 'r32float', 'sdf')
        self.levels = [self.dims]
        while True:
            d = self.levels[-1]
            if max(d) <= 8 or min(d) <= 3:
                break
            self.levels.append(tuple((x + 1) // 2 for x in d))
        self.P = [self._t3(d, 'r32float', f'p{i}') for i, d in enumerate(self.levels)]
        self.RHS = [self._t3(d, 'r32float', f'rhs{i}') for i, d in enumerate(self.levels)]
        self.RES = [self._t3(d, 'r32float', f'res{i}') for i, d in enumerate(self.levels[:-1])]
        self.stats_buf = self.gpu.buffer(32, 'stats', BU.STORAGE | BU.COPY_SRC | BU.COPY_DST)
        groups = ceil_div(nx, 8) * ceil_div(ny, 8) * ceil_div(nz, 4)
        self.rhs_part = self.gpu.buffer((groups + 1) * 8, 'rhs-partial-sums')
        self.fuel_part = self.gpu.buffer(max(32, groups * 4), 'fuel-partial-sums')
        self.fuel_tot = self.gpu.buffer(32, 'fuel-totals')

    def _allocate_fine(self):
        for t in self._fine:
            t.destroy()
        self._fine = []
        self.scal_fine = None
        if self.upres <= 1:
            return
        d = self.dims_fine
        mk = lambda label, fmt='rgba16float': self._t3(d, fmt, label, self._fine)
        self.scal_fine = [mk('scal-fine0'), mk('scal-fine1')]
        self.stmp_fine = mk('stmp-fine')
        self.sdf_fine = mk('sdf-fine', 'r32float')

    def _allocate_optional(self):
        for t in self._opt:
            t.destroy()
        self._opt = []
        f = self.features
        mk = lambda name: [self._t3(self.dims, 'rgba16float', f'{name}0', self._opt), self._t3(self.dims, 'rgba16float', f'{name}1', self._opt)]
        self.aux = mk('aux') if f['aux'] else None
        self.chem = mk('chem') if f['chem'] else None
        self.burn = mk('burn') if f['burn'] else None
        # soot left on surfaces; hot embers landed per floor column since the last step
        self.stain = self._t3(self.dims, 'r32float', 'stain', self._opt) if f.get('stain') else None
        if self.spots is not None:
            self.spots.destroy()
        self.spots = self.gpu.buffer(max(32, self.dims[0] * self.dims[2] * 4), 'ember-spots') if f['burn'] else None
        # written from outside (a liquid simulation in the same box) before each step
        self.water = self._t3(self.dims, 'rgba16float', 'water', self._opt) if f.get('water') else None

    # the textures kernels bind for the optional fields: (read, write)
    def _aux_rw(self):
        return (self.aux[0], self.aux[1]) if self.aux else (self._dummy['aux_r'], self._dummy['aux_w'])

    def _chem_rw(self):
        return (self.chem[0], self.chem[1]) if self.chem else (self._dummy['chem_r'], self._dummy['chem_w'])

    def _compile(self):
        g = self.gpu
        vf = g.vel_format
        V = {'VELFMT': vf}
        k = self._k
        k['adv_s_sl'] = g.kernel('adv_scalar.wgsl', ['tex3d', 'tex3d', 'tex3d', 'st3d:rgba16float:w', 'smp'], 'sl')
        k['adv_s_mc'] = g.kernel('adv_scalar.wgsl', ['tex3d', 'tex3d', 'tex3d', 'st3d:rgba16float:w', 'smp'], 'mc')
        k['adv_v_sl'] = g.kernel('adv_vel.wgsl', ['tex3d', 'tex3d', f'st3d:{vf}:w', 'smp'], 'sl', V)
        k['adv_v_mc'] = g.kernel('adv_vel.wgsl', ['tex3d', 'tex3d', f'st3d:{vf}:w', 'smp'], 'mc', V)
        k['react'] = g.kernel('react.wgsl', ['utex3d'] * 6 + ['st3d:rgba16float:w', 'st3d:r32float:w',
                                                             'st3d:rgba16float:w', 'st3d:rgba16float:w', 'utex3d',
                                                             'utex3d', 'rbuf', 'st3d:r32float:rw', 'rbuf'])
        k['stain_obj'] = g.kernel('stain_obj.wgsl', ['tex3d', 'utex3d', 'utex3d', 'st3d:r32float:rw', 'rbuf', 'smp'])
        k['fuel_partial'] = g.kernel('fuel_sum.wgsl', ['utex3d', 'buf', 'buf'], 'partial')
        k['fuel_total'] = g.kernel('fuel_sum.wgsl', ['utex3d', 'buf', 'buf'], 'total', workgroup=(256, 1, 1))
        k['burn'] = g.kernel('burn.wgsl', ['utex3d', 'utex3d', 'utex3d', 'st3d:rgba16float:w', 'utex3d', 'buf'])
        k['burn_init'] = g.kernel('burn_init.wgsl', ['utex3d', 'st3d:rgba16float:w'])
        burn_obj = ['tex3d', 'utex3d', 'utex3d', 'st3d:rgba16float:w', 'utex3d', 'rbuf', 'smp', 'buf']
        k['burn_obj'] = g.kernel('burn_obj.wgsl', burn_obj, 'main')
        k['burn_obj_init'] = g.kernel('burn_obj.wgsl', burn_obj, 'init')
        k['up_sl'] = g.kernel('upres.wgsl', ['tex3d', 'tex3d', 'tex3d', 'tex3d', 'st3d:rgba16float:w', 'smp'], 'sl')
        k['up_mc'] = g.kernel('upres.wgsl', ['tex3d', 'tex3d', 'tex3d', 'tex3d', 'st3d:rgba16float:w', 'smp'], 'mc')
        k['up_vel'] = g.kernel('upres.wgsl', ['tex3d', 'tex3d', 'tex3d', 'tex3d', 'st3d:rgba16float:w', 'smp'], 'velocity')
        k['rhs_partial'] = g.kernel('rhs_mean.wgsl', ['st3d:r32float:rw', 'utex3d', 'buf'], 'partial')
        k['rhs_total'] = g.kernel('rhs_mean.wgsl', ['st3d:r32float:rw', 'utex3d', 'buf'], 'total', workgroup=(256, 1, 1))
        k['rhs_subtract'] = g.kernel('rhs_mean.wgsl', ['st3d:r32float:rw', 'utex3d', 'buf'], 'subtract')
        k['curl'] = g.kernel('curl.wgsl', ['utex3d', 'st3d:rgba16float:w'])
        k['force'] = g.kernel('force.wgsl', ['utex3d', 'utex3d', 'st3d:rgba16float:w'])
        k['apply'] = g.kernel('apply.wgsl', ['utex3d'] * 6 + [f'st3d:{vf}:w', 'utex3d'], 'main', V)
        k['div'] = g.kernel('divergence.wgsl', ['utex3d', 'utex3d', 'utex3d', 'st3d:r32float:w'])
        k['smooth'] = g.kernel('mg_smooth.wgsl', ['st3d:r32float:rw', 'utex3d', 'utex3d'])
        k['residual'] = g.kernel('mg_residual.wgsl', ['utex3d', 'utex3d', 'utex3d', 'st3d:r32float:w'])
        k['restrict'] = g.kernel('mg_restrict.wgsl', ['utex3d', 'st3d:r32float:w', 'st3d:r32float:w'])
        k['prolong'] = g.kernel('mg_prolong.wgsl', ['st3d:r32float:rw', 'utex3d', 'utex3d'])
        k['project'] = g.kernel('project.wgsl', ['utex3d', 'utex3d', 'utex3d', f'st3d:{vf}:w'], 'main', V)
        k['sdf'] = g.kernel('sdf.wgsl', ['utex3d', 'st3d:r32float:w'])
        k['stats'] = g.kernel('stats.wgsl', ['utex3d', 'utex3d', 'buf'])
        for fmt in {vf, 'rgba16float', 'r32float'}:
            k['fill_' + fmt] = g.kernel('fill.wgsl', [f'st3d:{fmt}:w'], 'main', {'FMT': fmt})

    # -- state ---------------------------------------------------------------------------------

    def fill(self, b: Batch, tex, value=(0.0, 0.0, 0.0, 0.0)):
        u = Uniforms().v4(*tex.size, 0).v4(*value)
        b.run(self._k['fill_' + tex.format], [tex], u, tex.size)

    def reset(self):
        with self.gpu.batch() as b:
            for t in self.vel + [self.vtmp]:
                self.fill(b, t)
            for t in self.scal + [self.stmp, self.curl, self.force] + self._opt + self._fine:
                self.fill(b, t)
            for t in [self.expo] + self.P + self.RHS + self.RES:
                self.fill(b, t)
            self._write_sdf(b)
        if self.spots is not None:
            self.gpu.write_buffer(self.spots, np.zeros(self.spots.size // 4, np.uint32))
        self.time = 0.0
        self.steps = 0
        self.max_speed = 0.0
        self.bbox = None
        self.burning = 0
        self._burn_dirty = True
        self._stain_dirty = True

    def set_meshes(self, paths, resolution=96, cell=None):
        """Bake and load the meshes the scene uses, and its volumes' fields as fine as a simulation cell of `cell` (m;
        this solver's own when not given) can use. True if the atlas changed."""
        self.meshes.field_cell = cell or self.h or None
        changed = self.meshes.require(paths, resolution)
        if changed and self.dims is not None:
            with self.gpu.batch() as b:
                self._write_sdf(b)
        return changed

    def _slotted(self, colliders):
        """The colliders with their regions of the object-burn atlas (burnable ones, in order)."""
        colliders = list(colliders)[:MAX_COLLIDERS]
        k = 0
        for c in colliders:
            c.burn_slot = k if (c.burnable and self.features.get('burn')) else -1
            k += c.burn_slot >= 0
        return colliders

    def set_colliders(self, colliders):
        colliders = self._slotted(colliders)
        if colliders == self.colliders:
            return
        if len(colliders) != len(self.colliders):
            self._stain_dirty = True
        self.colliders = colliders
        self._burn_dirty = True
        with self.gpu.batch() as b:
            self._write_sdf(b)

    def update_colliders(self, b: Batch, colliders):
        """Record a collider change (a moving collider, every substep) into batch b."""
        colliders = self._slotted(colliders)
        if colliders == self.colliders:
            return
        if len(colliders) != len(self.colliders):
            self._stain_dirty = True
        self.colliders = colliders
        self._write_sdf(b)

    def solid_vel(self):
        """The broken pieces' velocity in the cells inside them (w = 1), made when first needed."""
        if self.svel is None or self.svel.size != tuple(self.dims):
            if self.svel is not None:
                self.svel.destroy()
            self.svel = self._t3(self.dims, 'rgba16float', 'solid-velocity')
        return self.svel

    def _write_sdf(self, b):
        u = pack_colliders(self._grid(0.0), self.colliders, self.meshes)
        b.run(self._k['sdf'], [self.meshes.atlas, self.sdf], u, self.dims)
        if self.scal_fine is not None:
            u = pack_colliders(self._grid_fine(0.0), self.colliders, self.meshes)
            b.run(self._k['sdf'], [self.meshes.atlas, self.sdf_fine], u, self.dims_fine)

    def _grid(self, dt, prm: SolverParams | None = None):
        p = prm or self._prm
        nx, ny, nz = self.dims
        return (Uniforms()
                .v4(nx, ny, nz, self.h)
                .v4(*self.origin, self.time)
                .v4(1.0 if p.open_sides else 0.0, 1.0 if p.open_top else 0.0, 0.0 if p.ground else 1.0, dt))

    def _grid_fine(self, dt, prm: SolverParams | None = None):
        p = prm or self._prm
        nx, ny, nz = self.dims_fine
        return (Uniforms()
                .v4(nx, ny, nz, self.h / self.upres)
                .v4(*self.origin, self.time)
                .v4(1.0 if p.open_sides else 0.0, 1.0 if p.open_top else 0.0, 0.0 if p.ground else 1.0, dt))

    _prm = SolverParams()
    cloth_hook = None   # engine.Cloth.hook: fabric feeding the fire and holding the air (set by the engine)

    # -- stepping ------------------------------------------------------------------------------

    def _advect(self, b, pair, dims, ga, mc):
        k = self._k
        lin = self.gpu.linear
        if mc > 0:
            b.run(k['adv_s_sl'], [self.vel[0], pair[0], pair[0], self.stmp, lin], ga().v4(mc), dims)
            b.run(k['adv_s_mc'], [self.vel[0], pair[0], self.stmp, pair[1], lin], ga().v4(mc), dims)
        else:
            b.run(k['adv_s_sl'], [self.vel[0], pair[0], pair[0], pair[1], lin], ga().v4(0), dims)
        pair.reverse()

    def _vapour_on(self, prm):
        """Water vapour is carried when the scene asks for it, and always when liquid water shares the box."""
        return bool(self.aux) and (prm.vapour or self.water is not None)

    def _init_burn(self, b, prm: SolverParams):
        sp = prm.spread
        u = (self._grid(0.0, prm)
             .v4(1.0 if sp.ground else 0.0, sp.area[0] * 0.5, sp.area[1] * 0.5, sp.coverage)
             .v4(sp.patch_freq, prm.seed * 1.37 + 0.5))
        pack_colliders(u, self.colliders, self.meshes)
        for t in self.burn:
            b.run(self._k['burn_init'], [self.meshes.atlas, t], u, self.dims)
        self._layout_burn_obj()
        if self.burn_obj:
            u = self._burn_obj_uniforms(0.0, prm, [])
            t = self.burn_obj
            for src, dst in ((t[1], t[0]), (t[0], t[1])):
                b.run(self._k['burn_obj_init'], self._burn_obj_res(src, dst), u, dst.size)
        self._burn_dirty = False

    def _layout_burn_obj(self):
        """Give every burnable collider a region of the object-burn atlas: a grid of the simulation's
        cell size around it, in its own frame, stacked along z."""
        for t in self.burn_obj or []:
            t.destroy()
        self.burn_obj = None
        if getattr(self, 'spots_obj', None) is not None:
            self.spots_obj.destroy()
        self.spots_obj = None
        meta = []
        z = 0
        for c in self.colliders:
            if c.burn_slot < 0:
                continue
            lo, hi = collider_extent(c, self.meshes)
            lo = lo - 2 * self.h
            dims = np.minimum(np.ceil((hi + 2 * self.h - lo) / self.h).astype(int), 384)
            meta.append((lo, dims, z))
            z += int(dims[2])
        if not meta:
            return
        w = max(int(m[1][0]) for m in meta)
        hgt = max(int(m[1][1]) for m in meta)
        self.burn_obj = [self.gpu.texture3d((w, hgt, z), 'rgba16float', f'burn-obj{i}') for i in range(2)]
        # hot embers landed per atlas cell (embers.wgsl counts them, burn_obj.wgsl may start spot fires)
        if getattr(self, 'spots_obj', None) is not None:
            self.spots_obj.destroy()
        self.spots_obj = self.gpu.buffer(max(32, w * hgt * z * 4), 'ember-spots-obj')
        self.gpu.write_buffer(self.spots_obj, np.zeros(max(8, w * hgt * z), np.uint32))
        data = np.zeros((len(meta), 8), np.float32)
        for i, (lo, dims, z0) in enumerate(meta):
            data[i] = (*lo, z0, *dims, self.h)
        if self.burn_slots is not None:
            self.burn_slots.destroy()
        self.burn_slots = self.gpu.buffer(max(32, data.nbytes), 'burn-slots')
        self.gpu.write_buffer(self.burn_slots, data)

    def _layout_stain_obj(self):
        """Give every collider a region of the soot atlas: a grid around it in its own frame and in its
        own size (coordinates divided by its scale, so the soot grows and shrinks with it), cells about a
        simulation cell across at its size now (coarser for big colliders), stacked along z."""
        if self.stain_obj is not None:
            self.stain_obj.destroy()
        self.stain_obj = None
        self.stain_regions = 0
        self._stain_dirty = False
        if not self.features.get('stain') or not self.colliders:
            return
        scales = [collider_scale(c) for c in self.colliders]
        ext = [(lo / s, hi / s) for (lo, hi), s in zip((collider_extent(c, self.meshes) for c in self.colliders), scales)]
        grow = 1.0
        for _ in range(12):   # coarser cells until the atlas fits a sensible budget
            cells = [self.h * grow / s for s in scales]               # size-relative cell per axis
            dims = [np.maximum(np.ceil((hi - lo) / c + 4).astype(int), 1) for (lo, hi), c in zip(ext, cells)]
            if max(int(d.max()) for d in dims) <= 256 and sum(int(np.prod(d)) for d in dims) <= 24_000_000:
                break
            grow *= 1.4
        data = np.zeros((len(ext), 12), np.float32)
        z = 0
        for i, ((lo, hi), d, c) in enumerate(zip(ext, dims, cells)):
            data[i] = (*(lo - 2 * c), z, *d, 0.0, *c, 0.0)
            z += int(d[2])
        w = max(int(d[0]) for d in dims)
        hgt = max(int(d[1]) for d in dims)
        self.stain_obj = self.gpu.texture3d((w, hgt, z), 'r32float', 'stain-obj')
        if self.stain_slots is not None:
            self.stain_slots.destroy()
        self.stain_slots = self.gpu.buffer(max(32, data.nbytes), 'stain-slots')
        self.gpu.write_buffer(self.stain_slots, data)
        self.stain_regions = len(ext)

    def read_stain_obj(self):
        """(depth, height, width, 1) float32 soot on colliders (their own frames), or None."""
        return self.gpu.read(self.stain_obj) if self.stain_obj is not None else None

    def _burn_obj_uniforms(self, dt, prm, douse):
        sp = prm.spread
        u = (self._grid(dt, prm)
             .v4(sp.catch_temp, 1.0 / max(sp.catch_time, 1e-3), sp.creep, 1.0 / max(sp.burn_time, 1e-3))
             .v4(1.0 / max(sp.smoulder, 1e-3), 1.0 if self.water else 0.0, prm.water_douse)
             .v4(sp.coverage, sp.patch_freq, prm.seed * 1.37 + 0.5)
             .v4(1.0 / sp.dry_time if sp.dry_time > 0 else 0.0, sp.spotting, self.steps % 100003))
        pack_emitters(u, douse, meshes=self.meshes)
        return pack_colliders(u, self.colliders, self.meshes)

    def _burn_obj_res(self, src, dst):
        return [self.scal[0], src, self.meshes.atlas, dst, self.water or self._dummy['water'], self.burn_slots,
                self.gpu.linear, self.spots_obj or self._no_spots]

    def _react(self, b, dt, prm, ems, fine=False):
        """Record the reaction pass on the simulation grid, or (fine) on the upres grid."""
        sp = prm.spread
        span = max(prm.flame_k - prm.ambient_k, 1.0)
        q_air = min(max(prm.humidity, 0.0), 100.0) / 100.0 * vapour_saturation(prm.ambient_k)
        latent = prm.latent_heat * LATENT_K_PER_G / span
        u = ((self._grid_fine(dt, prm) if fine else self._grid(dt, prm))
             .v4(prm.ignition, prm.burn_rate, prm.heat, prm.soot)
             .v4(prm.flame_gain, prm.flame_life, prm.expansion, prm.radiative)
             .v4(prm.cooling, prm.smoke_dissipation, prm.fuel_dissipation, prm.temp_cap)
             .v4(prm.rich, prm.expansion_cap, prm.air_use, 1.0 if (prm.oxygen and self.aux) else 0.0)
             .v4(1.0 if self._vapour_on(prm) else 0.0, prm.steam_yield, prm.water_yield, prm.vapour_dissipation)
             .v4(sp.fuel, sp.heat, sp.smoke, sp.smoulder_smoke)
             .v4(1.0 if self.chem else 0.0, 1.0 if (self.burn or self.burn_obj) else 0.0, prm.air_mixing, prm.boil_temp)
             .v4(1.0 if self.water else 0.0, prm.water_douse, 1.0 if (self._conserve_fuel(prm) and not fine) else 0.0)
             .v4(prm.thermal_expansion, prm.ambient_k, span, latent)
             .v4(self.upres if fine else 1, q_air, prm.flame_speed, prm.soot_stain if self.stain else 0.0)
             .v4(prm.steam_inert[0], prm.steam_inert[1], prm.steam_expansion,
                 1.0 if getattr(self, 'steam_smothers', False) else 0.0))
        pack_emitters(u, ems, prm.sponge, prm.sponge_strength, self.meshes)
        pack_colliders(u, self.colliders, self.meshes)
        aux_r, aux_w = self._aux_rw()
        chem_r, chem_w = self._chem_rw()
        burn_r = self.burn[0] if self.burn else self._dummy['burn_r']
        common = [self.water or self._dummy['water'], self.burn_obj[0] if self.burn_obj else self._dummy['burn_r'],
                  self.burn_slots if self.burn_obj else self._no_slots]
        if fine:
            d = self._dummy
            b.run(self._k['react'], [self.scal_fine[0], self.sdf_fine, aux_r, chem_r, burn_r, self.meshes.atlas,
                                     self.scal_fine[1], d['expo'], d['aux_w'], d['chem_w'], *common, d['stain'], self.fuel_tot],
                  u, self.dims_fine)
            self.scal_fine.reverse()
            return
        b.run(self._k['react'], [self.scal[0], self.sdf, aux_r, chem_r, burn_r, self.meshes.atlas,
                                 self.scal[1], self.expo, aux_w, chem_w, *common, self.stain or self._dummy['stain'],
                                 self.fuel_tot],
              u, self.dims)
        self.scal.reverse()
        if self.aux:
            self.aux.reverse()
        if self.chem:
            self.chem.reverse()

    @staticmethod
    def _conserve_fuel(prm):
        """Heavy vapour settling on the floor, and premixed clouds, need the fuel's total kept."""
        return prm.fuel_weight > 0.0 or prm.flame_speed > 0.0

    def _fuel_total(self, b, ga, slot):
        res = [self.scal[0], self.fuel_part, self.fuel_tot]
        b.run(self._k['fuel_partial'], res, ga().v4(slot), self.dims)
        b.run(self._k['fuel_total'], res, ga().v4(slot), groups=(1, 1, 1))

    def _step_fine(self, b, dt, prm, ems):
        """Carry and burn the fire on the upres grid, moved by the air just projected."""
        k = self._k
        mc = float(prm.maccormack)
        u = (self._grid_fine(dt, prm).v4(self.upres, prm.upres_turbulence, prm.seed * 3.71, mc)
             .v4(*self.dims, self.h))
        lin = self.gpu.linear
        f = self.scal_fine
        if mc > 0:
            b.run(k['up_sl'], [self.vel[0], self.curl, f[0], f[0], self.stmp_fine, lin], u, self.dims_fine)
            b.run(k['up_mc'], [self.vel[0], self.curl, f[0], self.stmp_fine, f[1], lin], u, self.dims_fine)
        else:
            b.run(k['up_sl'], [self.vel[0], self.curl, f[0], f[0], f[1], lin], u, self.dims_fine)
        f.reverse()
        self._react(b, dt, prm, ems, fine=True)
        if self.cloth_hook is not None:
            self.cloth_hook(b, self, dt, 'sources_fine')

    def _closed(self, prm):
        return not prm.open_sides and not prm.open_top and prm.ground

    def step(self, b: Batch, dt, prm: SolverParams, emitters, colliders=None):
        """Record one substep of length dt (seconds) into batch b. `colliders`, if given, are the
        colliders at this substep (for animated ones)."""
        self._prm = prm
        k = self._k
        dims = self.dims
        vdims = (dims[0] + 1, dims[1] + 1, dims[2] + 1)
        lin = self.gpu.linear
        mc = float(prm.maccormack)
        ga = lambda: self._grid(dt, prm)
        f = self.features
        atlas = self.meshes.atlas
        ems = list(emitters)[:MAX_EMITTERS]
        if colliders is not None:
            self.update_colliders(b, colliders)
        # broken pieces: folded into the colliders' distance (rewritten first, so last substep's are gone)
        self._pieces_on = False
        if self.pieces_step is not None:
            field, i = self.pieces_step
            self._write_sdf(b)
            self._pieces_on = field.bake(b, self, i)
        elif self._had_pieces:
            self._write_sdf(b)
        self._had_pieces = self._pieces_on
        if f['burn'] and self._burn_dirty:
            self._init_burn(b, prm)
        if f.get('stain') and self._stain_dirty:
            self._layout_stain_obj()

        # 1. scalars ride the current velocity (with the fuel's total kept, for heavy vapour and flame fronts)
        conserve = self._conserve_fuel(prm)
        if conserve:
            self._fuel_total(b, ga, 0)
        self._advect(b, self.scal, dims, ga, mc)
        if conserve:
            self._fuel_total(b, ga, 1)
        if self.aux:
            self._advect(b, self.aux, dims, ga, mc)
        if self.chem:
            self._advect(b, self.chem, dims, ga, mc)

        # 2. velocity rides itself; air comes in through the open sides and top no faster than the wind blows in
        mcv = float(prm.maccormack_vel)
        if mcv > 0:
            b.run(k['adv_v_sl'], [self.vel[0], self.vel[0], self.vtmp, lin], ga().v4(mcv).v4(*prm.wind), vdims)
            b.run(k['adv_v_mc'], [self.vel[0], self.vtmp, self.vel[1], lin], ga().v4(mcv).v4(*prm.wind), vdims)
        else:
            b.run(k['adv_v_sl'], [self.vel[0], self.vel[0], self.vel[1], lin], ga().v4(0).v4(*prm.wind), vdims)
        self.vel.reverse()

        # 3. burnable surfaces catch, burn and burn out: the floor, then burnable colliders
        sp = prm.spread
        douse = [e for e in ems if e.douse > 0]
        if self.burn:
            u = (ga().v4(sp.catch_temp, 1.0 / max(sp.catch_time, 1e-3), sp.creep, 1.0 / max(sp.burn_time, 1e-3))
                 .v4(1.0 / max(sp.smoulder, 1e-3), 1.0 if self.water else 0.0, prm.water_douse)
                 .v4(1.0 / sp.dry_time if sp.dry_time > 0 else 0.0, sp.spotting, self.steps % 100003))
            pack_emitters(u, douse, meshes=self.meshes)
            b.run(k['burn'], [self.scal[0], self.burn[0], atlas, self.burn[1], self.water or self._dummy['water'],
                              self.spots], u, dims)
            self.burn.reverse()
        if self.burn_obj:
            t = self.burn_obj
            b.run(k['burn_obj'], self._burn_obj_res(t[0], t[1]), self._burn_obj_uniforms(dt, prm, douse), t[0].size)
            t.reverse()

        # 4. sources, combustion, dissipation (and burning fabric); soot settles on the colliders
        self._react(b, dt, prm, ems)
        if self.cloth_hook is not None:
            self.cloth_hook(b, self, dt, 'sources')
        if self.stain_obj is not None and prm.soot_stain > 0:
            u = self._grid(dt, prm).v4(prm.soot_stain, self.stain_regions)
            pack_colliders(u, self.colliders, self.meshes)
            b.run(k['stain_obj'], [self.scal[0], self.sdf, atlas, self.stain_obj, self.stain_slots, lin], u,
                  self.stain_obj.size)

        # 5. vorticity and body forces
        b.run(k['curl'], [self.vel[0], self.curl], ga(), dims)
        u = (ga().v4(prm.vorticity, prm.turbulence, prm.disturbance, prm.vort_outside)
             .v4(prm.turb_freq, prm.turb_rise, prm.turb_evolve, prm.disturb_block)
             .v4(prm.mask_temp, prm.mask_flame, prm.mask_smoke, prm.disturb_rate)
             .v4(prm.turb_octave2, prm.seed * 7.31))
        b.run(k['force'], [self.curl, self.scal[0], self.force], u, dims)

        # 6. buoyancy, forces, wind, swirl, emitter velocities, walls and moving colliders
        wet = 1.0 if self._vapour_on(prm) else 0.0
        u = (ga().v4(prm.buoyancy, prm.soot_weight, prm.damping, VAPOUR_LIFT * wet).v4(prm.fuel_weight)
             .v4(*prm.wind, prm.wind_relax))
        pack_colliders(u, self.colliders, self.meshes, pieces=self._pieces_on)
        pack_emitters(u, ems, 1.0 if any(e.swirl != 0 for e in ems) else 0.0, meshes=self.meshes)
        b.run(k['apply'], [self.vel[0], self.scal[0], self.force, self.sdf, atlas, self._aux_rw()[0], self.vel[1],
                           self.svel if self._pieces_on else self._dummy['aux_r']], u, vdims)
        self.vel.reverse()
        if self.cloth_hook is not None:
            self.cloth_hook(b, self, dt, 'velocity')   # fabric holds the air back

        # 7-9. projection
        b.run(k['div'], [self.vel[0], self.expo, self.sdf, self.RHS[0]], ga(), dims)
        if self._closed(prm):
            res = [self.RHS[0], self.sdf, self.rhs_part]
            b.run(k['rhs_partial'], res, ga(), dims)
            b.run(k['rhs_total'], res, ga(), groups=(1, 1, 1))
            b.run(k['rhs_subtract'], res, ga(), dims)
        for _ in range(max(1, int(prm.mg_cycles))):
            self._vcycle(b, 0, prm)
        b.run(k['project'], [self.vel[0], self.P[0], self.sdf, self.vel[1]], ga(), vdims)
        self.vel.reverse()

        # 10. upres: the finer fire and smoke ride the new air
        if self.scal_fine is not None:
            self._step_fine(b, dt, prm, ems)

        self.time += dt
        self.steps += 1

    def _mg(self, lvl, prm, parity=0):
        d = self.levels[lvl]
        nc = self.levels[lvl + 1] if lvl + 1 < len(self.levels) else d
        return (Uniforms().v4(*d, parity)
                .v4(1.0 if prm.open_sides else 0.0, 1.0 if prm.open_top else 0.0, 0.0 if prm.ground else 1.0,
                    1.0 if (lvl == 0 and self.colliders) else 0.0)
                .v4(prm.sor).v4(*nc, 0))

    def _smooth(self, b, lvl, prm, parity):
        d = self.levels[lvl]
        b.run(self._k['smooth'], [self.P[lvl], self.RHS[lvl], self.sdf], self._mg(lvl, prm, parity),
              ((d[0] + 1) // 2, d[1], d[2]))

    def _vcycle(self, b, lvl, prm):
        last = len(self.levels) - 1
        if lvl == last:
            for _ in range(24):
                self._smooth(b, lvl, prm, 0)
                self._smooth(b, lvl, prm, 1)
            return
        for _ in range(int(prm.mg_pre)):
            self._smooth(b, lvl, prm, 0)
            self._smooth(b, lvl, prm, 1)
        k = self._k
        b.run(k['residual'], [self.P[lvl], self.RHS[lvl], self.sdf, self.RES[lvl]], self._mg(lvl, prm), self.levels[lvl])
        b.run(k['restrict'], [self.RES[lvl], self.RHS[lvl + 1], self.P[lvl + 1]], self._mg(lvl, prm), self.levels[lvl + 1])
        self._vcycle(b, lvl + 1, prm)
        b.run(k['prolong'], [self.P[lvl], self.P[lvl + 1], self.sdf], self._mg(lvl, prm), self.levels[lvl])
        for _ in range(int(prm.mg_post)):
            self._smooth(b, lvl, prm, 1)
            self._smooth(b, lvl, prm, 0)

    # -- diagnostics -----------------------------------------------------------------------------

    def measure(self, thresholds=(0.02, 0.02, 0.05)):
        """Max speed, visible bounding box (cells) and burning cell count, read back from the GPU."""
        init = np.array([0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0, 0, 0], np.uint32)
        self.gpu.write_buffer(self.stats_buf, init)
        with self.gpu.batch() as b:
            b.run(self._k['stats'], [self.vel[0], self.scal[0], self.stats_buf],
                  self._grid(0.0).v4(*thresholds), self.dims)
        raw = np.frombuffer(self.gpu.read_buffer(self.stats_buf), np.uint32)
        self.max_speed = float(np.frombuffer(raw[:1].tobytes(), np.float32)[0])
        lo, hi = raw[1:4], raw[4:7]
        self.bbox = None if lo[0] == 0xFFFFFFFF else (tuple(int(x) for x in lo), tuple(int(x) + 1 for x in hi))
        self.burning = int(raw[7])
        return self.max_speed, self.bbox, self.burning

    def substeps_for(self, frame_dt, cfl=2.0, lo=1, hi=8):
        if self.max_speed <= 0:
            return lo
        n = math.ceil(self.max_speed * frame_dt / (cfl * self.h))
        return int(min(hi, max(lo, n)))

    def read_scalars(self):
        """(nz, ny, nx, 4) float16: temperature, fuel, soot, flame."""
        return self.gpu.read(self.scal[0])

    def read_scalars_fine(self):
        """The upres grid's fields (as read_scalars), or the simulation grid's without upres."""
        return self.gpu.read(self.scal_fine[0]) if self.scal_fine is not None else self.read_scalars()

    def read_burn_obj(self):
        """(depth, height, width, 4) float16 object-burn atlas, or None."""
        return self.gpu.read(self.burn_obj[0]) if self.burn_obj else None

    def read_aux(self):
        """(nz, ny, nx, 4) float16: oxygen used up, water vapour (g/m^3), or None."""
        return self.gpu.read(self.aux[0]) if self.aux else None

    def read_chem(self):
        """(nz, ny, nx, 4) float16: flame colourant (rgb), or None."""
        return self.gpu.read(self.chem[0]) if self.chem else None

    def read_burn(self):
        """(nz, ny, nx, 4) float16: burnable surfaces (fuel left, catching, burnable, smoulder), or None."""
        return self.gpu.read(self.burn[0]) if self.burn else None

    def read_velocity(self):
        """(nz+1, ny+1, nx+1, 4) float16 face velocities (m/s): x, y, z faces of each cell."""
        return self.gpu.read(self.vel[0]).astype(np.float16)

    def read_stain(self):
        """(nz, ny, nx, 1) float32 soot left on surfaces next to each cell, or None."""
        return self.gpu.read(self.stain) if self.stain else None

    def read_velocity_fine(self):
        """(nz, ny, nx, 3) float32 velocity (m/s) at the upres grid's cell centres: the simulated air
        plus the small swirls the fine grid moves with. Without upres, the cell-centred velocity."""
        if self.scal_fine is None:
            return self.read_velocity_centres()
        u = (self._grid_fine(0.0).v4(self.upres, self._prm.upres_turbulence, self._prm.seed * 3.71, 0.0)
             .v4(*self.dims, self.h))
        f = self.scal_fine
        with self.gpu.batch() as b:
            b.run(self._k['up_vel'], [self.vel[0], self.curl, f[0], f[0], self.stmp_fine, self.gpu.linear], u,
                  self.dims_fine)
        return self.gpu.read(self.stmp_fine)[..., :3].astype(np.float32)

    # -- checkpoints and a growing domain ------------------------------------------------------------

    def save_state(self):
        """Everything the simulation needs to carry on from here, as numpy arrays (a checkpoint)."""
        st = {'dims': list(self.dims), 'h': self.h, 'origin': list(self.origin), 'time': self.time,
              'steps': self.steps, 'max_speed': self.max_speed, 'upres': self.upres,
              'vel': self.gpu.read(self.vel[0]), 'scal': self.gpu.read(self.scal[0]), 'p': self.gpu.read(self.P[0])}
        for name in ('aux', 'chem', 'burn'):
            t = getattr(self, name)
            if t:
                st[name] = self.gpu.read(t[0])
        if self.burn_obj:
            st['burn_obj'] = self.gpu.read(self.burn_obj[0])
        if self.stain:
            st['stain'] = self.gpu.read(self.stain)
        if self.stain_obj is not None:
            st['stain_obj'] = self.gpu.read(self.stain_obj)
        if self.scal_fine is not None:
            st['scal_fine'] = self.gpu.read(self.scal_fine[0])
        return st

    def load_state(self, st, offset=(0, 0, 0)):
        """Carry on from a checkpoint (save_state). The grid may be larger than the saved one: the saved
        fields land `offset` cells in from its low corner (a domain that has grown)."""
        off = tuple(int(x) for x in offset)
        prm = self._prm

        def paste(tex, arr, k=1):
            a = np.asarray(arr)
            if tuple(a.shape[2::-1]) == tuple(tex.size) and not any(off):
                self.gpu.upload(tex, a)
                return
            out = np.zeros(tex.size[::-1] + (a.shape[3],), a.dtype)
            z, y, x = (o * k for o in off[::-1])
            d, hh, w = (min(a.shape[i], out.shape[i] - (z, y, x)[i]) for i in range(3))
            out[z:z + d, y:y + hh, x:x + w] = a[:d, :hh, :w]
            self.gpu.upload(tex, out)

        paste(self.vel[0], st['vel'])
        paste(self.scal[0], st['scal'])
        if 'p' in st:
            paste(self.P[0], st['p'])
        for name in ('aux', 'chem'):
            t = getattr(self, name)
            if t and name in st:
                paste(t[0], st[name])
        if self.stain and 'stain' in st:
            paste(self.stain, st['stain'])
        if 'stain_obj' in st and self.features.get('stain'):
            self._layout_stain_obj()   # the same colliders give the same layout
            if self.stain_obj is not None and tuple(st['stain_obj'].shape[2::-1]) == tuple(self.stain_obj.size):
                self.gpu.upload(self.stain_obj, st['stain_obj'])
        if self.scal_fine is not None and 'scal_fine' in st and int(st.get('upres', 1)) == self.upres:
            paste(self.scal_fine[0], st['scal_fine'], self.upres)
        if self.burn and 'burn' in st:
            # lay out the new floor, then put the saved burn state back over its part of it
            with self.gpu.batch() as b:
                self._init_burn(b, prm)
            fresh = self.gpu.read(self.burn[0])
            a = st['burn']
            z, y, x = off[::-1]
            d, hh, w = (min(a.shape[i], fresh.shape[i] - (z, y, x)[i]) for i in range(3))
            fresh[z:z + d, y:y + hh, x:x + w] = a[:d, :hh, :w]
            for t in self.burn:
                self.gpu.upload(t, fresh)
            if self.burn_obj and 'burn_obj' in st and st['burn_obj'].shape[2::-1] == tuple(self.burn_obj[0].size):
                for t in self.burn_obj:
                    self.gpu.upload(t, st['burn_obj'])
            self._burn_dirty = False
        self.time = float(st.get('time', 0.0))
        self.steps = int(st.get('steps', 0))
        self.max_speed = float(st.get('max_speed', 0.0))

    def grow(self, lo_cells, hi_cells):
        """Enlarge the domain by lo_cells / hi_cells (x, y, z) on its low and high sides, keeping the
        simulation. The origin moves out by lo_cells, so everything stays where it is in the world."""
        lo = np.asarray(lo_cells, int)
        hi = np.asarray(hi_cells, int)
        if not (lo.any() or hi.any()):
            return False
        st = self.save_state()
        base, feats, upres, colliders = self.base, dict(self.features), self.upres, list(self.colliders)
        count = self.grow_count
        dims = tuple(int(x) for x in np.asarray(self.dims) + lo + hi)
        origin = tuple(float(x) for x in np.asarray(self.origin) - lo * self.h)
        self.base = None
        self.configure(dims, self.h, origin, feats, upres)
        self.base = base
        self.colliders = []
        self.set_colliders(colliders)
        self.load_state(st, tuple(lo))
        self.grow_count = count + 1
        return True

    def grown(self):
        return self.base is not None and self.dims is not None and tuple(self.dims) != tuple(self.base[0])

    def restore_base(self):
        """Go back to the layout asked for (after a grown domain), for a fresh start."""
        if self.grown():
            base, feats, upres = self.base, dict(self.features), self.upres
            self.base = None
            self.configure(*base, feats, upres)

    def growth_needed(self, margin, step, limit_dims, prm: SolverParams):
        """Cells to add on each side (lo, hi) when the smoke's bounding box comes within `margin` cells
        of an open side, `step` cells at a time, up to `limit_dims` cells per axis."""
        lo, hi = np.zeros(3, int), np.zeros(3, int)
        if self.bbox is None:
            return lo, hi
        b0, b1 = np.asarray(self.bbox[0]), np.asarray(self.bbox[1])
        d = np.asarray(self.dims)
        step = max(8, int(step) // 8 * 8)             # whole blocks of 8 cells, for the multigrid
        room = (np.asarray(limit_dims) - d) // 8 * 8
        open_lo = [prm.open_sides, False, prm.open_sides]   # never below the ground
        open_hi = [prm.open_sides, prm.open_top, prm.open_sides]
        for a in range(3):
            if open_lo[a] and b0[a] < margin and room[a] > 0:
                lo[a] = min(step, room[a])
                room[a] -= lo[a]
            if open_hi[a] and b1[a] > d[a] - margin and room[a] > 0:
                hi[a] = min(step, room[a])
        return lo, hi

    def read_velocity_centres(self):
        """(nz, ny, nx, 3) float32 cell-centred velocity (m/s)."""
        v = self.gpu.read(self.vel[0]).astype(np.float32)
        u = 0.5 * (v[:-1, :-1, :-1, 0] + v[:-1, :-1, 1:, 0])
        w_ = 0.5 * (v[:-1, :-1, :-1, 1] + v[:-1, 1:, :-1, 1])
        z = 0.5 * (v[:-1, :-1, :-1, 2] + v[1:, :-1, :-1, 2])
        return np.stack([u, w_, z], axis=-1)

    def world_bounds(self):
        nx, ny, nz = self.dims
        o = np.array(self.origin)
        return o, o + np.array([nx, ny, nz]) * self.h
