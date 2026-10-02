"""Matter: sand, snow, mud, jelly and clay, simulated as MPM particles on the GPU (MLS-MPM, Hu et al. 2018), in every
kind of scene.

Each particle carries its position, velocity, the affine part of the velocity round it (APIC) and its elastic
deformation gradient F. Every step (mpm_*.wgsl):
- p2g: the particles add their mass, momentum and the push of their stress to the grid's nodes (fixed-point integer
  atomics: the same sums on any GPU, in any order);
- grid: the nodes' velocities, gravity, and the ground, the box's walls and the objects in the scene (falling and
  keyframed ones moving), with friction; the momentum each object takes from the matter is kept for the rigid
  bodies (solids.py), which it pushes back;
- g2p: the particles pick up the new velocity, deform with it, and give way where their material yields
  (plastic): sand slides past its friction angle (Drucker-Prager, Klar et al. 2016), snow packs and breaks up
  (Stomakhin et al. 2013), mud and clay flow past their yield stress (von Mises; mud relaxes toward it, clay is
  perfectly plastic), and jelly springs back (neo-Hookean).

The step is as long as the stiffest material's sound speed and the fastest particle allow. For drawing, the
particles make a distance field round them on the grid (Zhu and Bridson 2005) with their look at each node, which the
stage traces like any other object (stage.wgsl).

Sources (Scene.matter_specs): a body of matter at the start (a pile of sand, a snowball, a block of jelly), held
where it is until it is let go, or a stream poured from a nozzle.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from .gpu import GPU, Batch, Uniforms, groups_1d

log = logging.getLogger(__name__)

PARTICLE_BYTES = 128
PER_AXIS = 2              # particles per node spacing along each axis (8 to a cell)
MAX_MATS = 16             # (slot 15 marks a particle that is gone in the 8-byte form: 15 materials at most)
CFL_SOUND = 0.4           # a step lets sound cross this share of a cell
CFL_MOVE = 0.3            # and the fastest particle move this share
FX_R = 64.0               # mpm_grid.wgsl react's fixed point
SURF_R = 2.0              # the surface's kernel radius (cells)
SURF_PARTICLE = 0.55      # a particle's radius in it (cells)
MARGIN = 3                # nodes of grid past the box all round
MAX_PIECES = 4096         # broken objects' pieces the matter can tell apart (what each takes from it)
HEAT_SPEED = 4.0          # matter takes on and gives off heat this many times faster than for real, unless the scene
                          # says (Domain > Heat speed): melting in seconds
HEAT_CONVECTION = 50.0    # W/m^2/K: the air (or the fire's gas) flowing past its surface
HEAT_BLOCK = 4            # the gas's cells to a heat source (a block of them), for the fire's radiant heat
HEAT_LIGHTS = 512         # the most heat sources
FLAME_ABSORPTION = 1.0    # 1/m: how strongly the fire's hot gas absorbs, and so radiates (a sooty flame)
SURF_SMOOTH = 2           # passes smoothing the surface

# The heap sand poured onto the ground makes (its angle of repose: atan of its height over its radius) for the friction
# angle of the Drucker-Prager model, measured: the grid and the transfers smooth the heap down a few degrees.
_REPOSE = (21.4, 28.7, 33.8, 38.7)
_PHI = (25.0, 33.0, 40.0, 48.0)


def friction_angle(repose):
    """The model's friction angle (degrees) that gives sand an angle of repose `repose` (degrees)."""
    r = float(repose)
    if r > _REPOSE[-1]:
        return _PHI[-1] + (r - _REPOSE[-1]) * (_PHI[-1] - _PHI[-2]) / (_REPOSE[-1] - _REPOSE[-2])
    return float(np.interp(r, _REPOSE, _PHI)) if r >= _REPOSE[0] else r * _PHI[0] / _REPOSE[0]


@dataclass(frozen=True)
class MatterMaterial:
    key: str
    label: str
    model: str                  # jelly (neo-Hookean), sand (Drucker-Prager), snow (Stomakhin), mud, clay (von Mises)
    E: float                    # Young's modulus (Pa): soft for a simulation, as MPM work uses them
    nu: float                   # Poisson's ratio
    density: float              # kg/m^3
    friction: float = 0.5       # against the ground and objects (Coulomb)
    angle: float = 0.0          # sand: its angle of repose (degrees)
    cohesion: float = 0.0       # sand: the stretch it holds (wet sand)
    theta_c: float = 0.0        # snow: the squeeze and the stretch it holds before it packs or breaks
    theta_s: float = 0.0
    xi: float = 0.0             # snow: hardening (how much stiffer packed snow is)
    h_max: float = 1.0          # snow: the most it hardens
    yield_stress: float = 0.0   # mud, clay: Pa
    relax: float = 0.0          # mud: the share of the stress past its yield it lets go per second
    tension: bool = False       # clay holds together when pulled; mud tears
    sticks: float = 0.0         # Pa: how hard it sticks to the objects it touches (packing snow on a wall; dry sand: 0)
    colour: tuple = (0.5, 0.5, 0.5)   # linear albedo (jelly: its tint)
    roughness: float = 0.8
    clear: float = 0.0          # light through it (jelly)
    sparkle: float = 0.0        # glints from its grains or crystals
    wrap: float = 0.0           # light into it (snow)
    variation: float = 0.2      # how much its particles' colours differ
    metal: float = 0.0          # how metallic its look is (it reflects in its own colour)
    melts_at: float = 0.0       # K: past it, it melts (into `melt`), and below it, it sets (into `freeze`); 0: never
    melt: str = ''              # what it becomes past its melting point
    freeze: str = ''            # what it becomes below it
    heat_capacity: float = 0.0  # J/kg/K (0: its temperature stays as it starts)
    absorbs: float = 0.0        # the share of radiant heat it takes in (shiny aluminium little, chocolate most)
    diffusivity: float = 0.0    # m^2/s: how fast heat spreads through it
    water_cools: float = 20.0   # times faster the water cools its surface than the air does
    burns_at: float = 0.0       # K: past it, it catches and burns (0: it does not)
    burn_rate: float = 0.0      # the share of it that burns away a second while it is lit (where the air reaches it)
    burn_temp: float = 0.0      # K: how hot it burns
    burns_to: str = ''          # what is left (nothing: it is gone)
    flames: float = 0.0         # how much fuel it gives the fire as it burns (1: as grass does)
    ash_share: float = 0.0      # the share of it left as ash once it has burnt (the rest is gone: a heap burns down)

    @property
    def mu(self):
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lam(self):
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    @property
    def porous(self):
        """Water gets in between its grains (sand, snow, mud): the water's pressure is in it as well as round it."""
        return self.model in ('sand', 'snow', 'mud')

    def sound(self, hardening=1.0):
        """Its fastest wave (m/s): pressure waves through it."""
        return math.sqrt((self.lam + 2.0 * self.mu) * hardening / self.density)


MATTERS = {m.key: m for m in (
    MatterMaterial('sand', 'Sand', 'sand', 3.5e5, 0.3, 1600.0, friction=0.5, angle=34.0, colour=(0.55, 0.42, 0.25),
                   roughness=0.95, sparkle=0.6, variation=0.3),
    MatterMaterial('wet_sand', 'Wet sand', 'sand', 3.5e5, 0.3, 1900.0, friction=0.7, angle=38.0, cohesion=0.004,
                   colour=(0.23, 0.17, 0.1), roughness=0.55, sparkle=0.2, variation=0.25, sticks=300.0),
    # (sand the water has soaked through: its grains let go of each other. What wet sand under the water becomes)
    MatterMaterial('soaked_sand', 'Soaked sand', 'sand', 3.5e5, 0.3, 2000.0, friction=0.45, angle=30.0,
                   colour=(0.2, 0.145, 0.085), roughness=0.3, sparkle=0.1, variation=0.25),
    MatterMaterial('snow', 'Snow', 'snow', 1.4e5, 0.2, 400.0, friction=0.3, theta_c=0.025, theta_s=0.0075, xi=10.0, h_max=3.0,
                   colour=(0.85, 0.88, 0.92), roughness=0.8, sparkle=1.0, wrap=0.6, variation=0.04, sticks=200.0),
    MatterMaterial('packing_snow', 'Packing snow', 'snow', 2.5e5, 0.2, 600.0, friction=0.4, theta_c=0.019, theta_s=0.0075,
                   xi=10.0, h_max=3.0, colour=(0.8, 0.83, 0.87), roughness=0.7, sparkle=0.6, wrap=0.5, variation=0.04,
                   sticks=700.0),
    MatterMaterial('mud', 'Mud', 'mud', 1.5e5, 0.4, 1700.0, friction=0.9, yield_stress=400.0, relax=800.0,
                   colour=(0.11, 0.075, 0.045), roughness=0.3, variation=0.15, sticks=1500.0),
    MatterMaterial('jelly', 'Jelly', 'jelly', 3.0e4, 0.42, 1050.0, friction=0.6, colour=(0.92, 0.22, 0.26), roughness=0.08,
                   clear=0.85, variation=0.0),
    MatterMaterial('clay', 'Clay', 'clay', 4.0e5, 0.35, 1800.0, friction=0.8, yield_stress=2.0e4, tension=True,
                   colour=(0.48, 0.22, 0.12), roughness=0.7, variation=0.08, sticks=6000.0),
    # things that melt: a solid (clay's model, firm) and its melt (mud's: runny, or thick as chocolate is), each turning
    # into the other at the melting point. A melt's small yield stress stands in for its surface tension, which keeps
    # a puddle as deep as it is for real (its yield stress / (density g): molten iron's some 7 mm) instead of spreading
    # thinner than the particles
    # things that burn: catching where the fire's heat takes them past their ignition point, flaming (dry leaves),
    # smouldering (sawdust) or glowing (coal), and burning down to ash
    MatterMaterial('leaves', 'Dry leaves', 'sand', 2.0e4, 0.3, 80.0, friction=0.8, angle=45.0, cohesion=0.002,
                   colour=(0.42, 0.24, 0.08), roughness=0.85, variation=0.45, heat_capacity=1500.0, absorbs=0.8,
                   diffusivity=1.0e-7, burns_at=530.0, burn_rate=0.6, burn_temp=1050.0, burns_to='ash', flames=0.6,
                   ash_share=0.1),
    MatterMaterial('sawdust', 'Sawdust', 'sand', 1.0e5, 0.3, 250.0, friction=0.6, angle=40.0, colour=(0.72, 0.56, 0.34),
                   roughness=0.9, variation=0.15, heat_capacity=1700.0, absorbs=0.8, diffusivity=1.0e-7, burns_at=560.0,
                   burn_rate=0.15, burn_temp=950.0, burns_to='ash', flames=0.5, ash_share=0.2),
    MatterMaterial('coal', 'Coal', 'sand', 3.0e5, 0.3, 800.0, friction=0.6, angle=38.0, colour=(0.035, 0.033, 0.035),
                   roughness=0.55, sparkle=0.35, variation=0.15, heat_capacity=1300.0, absorbs=0.95, diffusivity=2.0e-7,
                   burns_at=720.0, burn_rate=0.02, burn_temp=1300.0, burns_to='ash', flames=0.15, ash_share=0.3),
    MatterMaterial('ash', 'Ash', 'sand', 2.0e4, 0.3, 120.0, friction=0.6, angle=42.0, colour=(0.42, 0.41, 0.39),
                   roughness=0.95, variation=0.2, heat_capacity=800.0, absorbs=0.9, diffusivity=1.0e-7),
    MatterMaterial('wax', 'Wax', 'clay', 4.0e5, 0.35, 900.0, friction=0.5, yield_stress=1.5e4, tension=True,
                   colour=(0.78, 0.74, 0.63), roughness=0.45, wrap=0.4, variation=0.02, melts_at=333.0,
                   melt='molten_wax', heat_capacity=2900.0, absorbs=0.85, diffusivity=1.4e-7),
    MatterMaterial('molten_wax', 'Molten wax', 'mud', 3.0e5, 0.42, 850.0, friction=0.2, yield_stress=30.0, relax=1.0e5,
                   colour=(0.8, 0.73, 0.55), roughness=0.08, wrap=0.55, variation=0.0, melts_at=333.0, freeze='wax',
                   heat_capacity=2900.0, absorbs=0.85, diffusivity=1.4e-7),
    MatterMaterial('chocolate', 'Chocolate', 'clay', 4.0e5, 0.35, 1300.0, friction=0.6, yield_stress=2.0e4, tension=True,
                   colour=(0.075, 0.035, 0.016), roughness=0.3, variation=0.02, melts_at=307.0, melt='molten_chocolate',
                   heat_capacity=1600.0, absorbs=0.9, diffusivity=1.2e-7),
    MatterMaterial('molten_chocolate', 'Melted chocolate', 'mud', 3.0e5, 0.42, 1250.0, friction=0.6, yield_stress=25.0,
                   relax=2000.0, colour=(0.07, 0.032, 0.015), roughness=0.08, variation=0.0, melts_at=307.0,
                   freeze='chocolate', heat_capacity=1600.0, absorbs=0.9, diffusivity=1.2e-7),
    MatterMaterial('aluminium', 'Aluminium', 'clay', 1.5e6, 0.33, 2700.0, friction=0.6, yield_stress=4.0e5, tension=True,
                   colour=(0.75, 0.75, 0.77), roughness=0.35, variation=0.02, metal=1.0, melts_at=933.0,
                   melt='molten_aluminium', heat_capacity=900.0, absorbs=0.2, diffusivity=9.7e-5),
    MatterMaterial('molten_aluminium', 'Molten aluminium', 'mud', 6.0e5, 0.42, 2400.0, friction=0.3, yield_stress=250.0,
                   relax=1.0e5, colour=(0.75, 0.75, 0.77), roughness=0.05, variation=0.0, metal=1.0, melts_at=933.0,
                   freeze='aluminium', heat_capacity=1100.0, absorbs=0.2, diffusivity=4.0e-5),
    MatterMaterial('iron', 'Iron', 'clay', 1.5e6, 0.3, 7200.0, friction=0.6, yield_stress=6.0e5, tension=True,
                   colour=(0.36, 0.34, 0.33), roughness=0.55, variation=0.03, metal=1.0, melts_at=1423.0,
                   melt='molten_iron', heat_capacity=450.0, absorbs=0.75, diffusivity=2.0e-5),
    MatterMaterial('molten_iron', 'Molten iron', 'mud', 6.0e5, 0.42, 7000.0, friction=0.3, yield_stress=500.0, relax=1.0e5,
                   colour=(0.36, 0.34, 0.33), roughness=0.08, variation=0.0, metal=1.0, melts_at=1423.0, freeze='iron',
                   heat_capacity=820.0, absorbs=0.4, diffusivity=7.0e-6),
)}
MODELS = {'jelly': 0, 'sand': 1, 'snow': 2, 'mud': 3, 'clay': 4}


def material(key):
    return MATTERS.get(key, MATTERS['sand'])


def _stamp(source):
    """A mesh file's modification time (so editing it refills the matter), or 0."""
    if not source:
        return 0.0
    try:
        from .mesh import MeshLibrary
        return MeshLibrary._stamp(source)
    except (OSError, ValueError):
        return 0.0


@dataclass
class MatterSpec:
    """One source of matter (Scene.matter_specs): a body of it at the start, or a stream poured from a nozzle."""
    material: str = 'sand'
    shape: str = 'box'          # box, sphere, cylinder, pile (a cone: radius size x, height size y), mesh (it fills it)
    pos: tuple = (0.0, 0.5, 0.0)          # fire-local m: its middle, whatever its shape (a pour: the nozzle)
    size: tuple = (0.25, 0.25, 0.25)      # half extents, or radius and half height (a mesh: its scale on each axis)
    yaw: float = 0.0                      # radians
    velocity: tuple = (0.0, 0.0, 0.0)     # m/s as it starts (a thrown snowball) or as it pours
    release: float = 0.0                  # s of simulation: held still until then
    pour: bool = False                    # a stream instead of a body
    rate: float = 0.0                     # a stream's flow (m^3/s)
    start: float = 0.0                    # when it pours (s of simulation)
    stop: float = 1.0e9
    colour: tuple = None                  # its own colour (linear), or None: the material's
    stiffness: float = 1.0                # times the material's
    seed: int = 0
    temperature: float = 293.15           # K, as it starts
    mesh: str = ''                        # a mesh it fills (shape mesh): its source (mesh.load_mesh)


def fill_points(shape, size, spacing, rng, sdf=None):
    """Points filling a shape (its own frame, m), on a lattice `spacing` apart, each jittered within its cell. A mesh
    (sdf: its mesh.MeshSDF, in its own units) is scaled by size on each axis, its own origin at the frame's."""
    if shape == 'mesh':
        return _fill_mesh(sdf, size, spacing, rng)
    s = np.abs(np.asarray(size, float))
    ext = np.array([s[0], s[1], s[2]]) if shape in ('box', 'cylinder', 'pile') else np.array([s[0]] * 3)
    if shape == 'cylinder':
        ext = np.array([s[0], s[1], s[0]])
    if shape == 'pile':
        ext = np.array([s[0], s[1], s[0]])
    n = np.maximum(np.ceil(2.0 * ext / spacing).astype(int), 1)
    axes = [(np.arange(k) + 0.5) * spacing - e for k, e in zip(n, ext)]
    g = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    g = g + (rng.random(g.shape) - 0.5) * spacing
    if shape == 'sphere':
        keep = np.linalg.norm(g, axis=1) <= s[0]
    elif shape == 'cylinder':
        keep = (np.hypot(g[:, 0], g[:, 2]) <= s[0]) & (np.abs(g[:, 1]) <= s[1])
    elif shape == 'pile':
        # a cone standing on its base: radius s0 at the bottom (y = -s1), its tip at y = +s1
        h = (g[:, 1] + s[1]) / max(2.0 * s[1], 1e-9)
        keep = (h >= 0.0) & (h <= 1.0) & (np.hypot(g[:, 0], g[:, 2]) <= s[0] * (1.0 - h))
    else:
        keep = np.all(np.abs(g) <= s, axis=1)
    return g[keep]


def _fill_mesh(sdf, size, spacing, rng):
    if sdf is None:
        return np.zeros((0, 3))
    scale = np.maximum(np.abs(np.asarray(size, float)), 1e-6)
    lo = np.asarray(sdf.mesh_min, float) * scale
    hi = np.asarray(sdf.mesh_max, float) * scale
    n = np.maximum(np.ceil((hi - lo) / spacing).astype(int), 1)
    if int(np.prod(n)) > 60_000_000:
        raise ValueError('mesh too big for the matter detail')
    axes = [lo[k] + (np.arange(n[k]) + 0.5) * spacing for k in range(3)]
    g = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    g = g + (rng.random(g.shape) - 0.5) * spacing
    return g[sample_sdf(sdf, g / scale) < 0.0]


def sample_sdf(sdf, q):
    """A mesh's baked distance (mesh.MeshSDF, negative inside) at points q (n, 3) in its own units, trilinear; outside
    its grid, its edge's."""
    data = np.asarray(sdf.data, np.float32)          # (nz, ny, nx)
    dims = np.asarray(sdf.dims, int)                  # (nx, ny, nz)
    cell = (np.asarray(sdf.bmax, float) - np.asarray(sdf.bmin, float)) / dims
    f = (q - np.asarray(sdf.bmin, float)) / cell - 0.5
    f = np.clip(f, 0.0, dims - 1.000001)
    i = np.floor(f).astype(int)
    t = f - i
    i1 = np.minimum(i + 1, dims - 1)
    out = np.zeros(len(q), np.float64)
    for cx in (0, 1):
        for cy in (0, 1):
            for cz in (0, 1):
                ix = i1[:, 0] if cx else i[:, 0]
                iy = i1[:, 1] if cy else i[:, 1]
                iz = i1[:, 2] if cz else i[:, 2]
                w = ((t[:, 0] if cx else 1.0 - t[:, 0]) * (t[:, 1] if cy else 1.0 - t[:, 1])
                     * (t[:, 2] if cz else 1.0 - t[:, 2]))
                out += w * data[iz, iy, ix]
    return out


def _yaw(v, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    v = np.asarray(v, float)
    return np.stack([c * v[..., 0] + s * v[..., 2], v[..., 1], -s * v[..., 0] + c * v[..., 2]], -1)


def particles(xs, mat_index, vel, release, rng, temperature=293.15):
    """New particles (n, 32) float32 at grid positions xs (n, 3), `temperature` K."""
    n = len(xs)
    P = np.zeros((n, 32), np.float32)
    P[:, 0:3] = xs
    P[:, 3] = mat_index
    P[:, 4:7] = vel
    P[:, 7] = 1.0                       # snow's Jp
    P[:, 11] = rng.random(n)            # its own random number (look)
    P[:, 15] = release                  # held until
    P[:, 20] = P[:, 25] = P[:, 30] = 1.0  # F = I
    P[:, 23] = temperature              # (f0.w: mpm_heat.wgsl)
    return P


class Matter:
    """The matter in a simulation: its particles, grid and surface on the GPU."""

    def __init__(self, gpu: GPU, meshes=None):
        self.gpu = gpu
        self.meshes = meshes
        self.specs: list[MatterSpec] = []
        self.key = None
        self.dims = None
        self.dx = 0.0
        self.origin = np.zeros(3)
        self.capacity = 0
        self.count = 0               # particle slots in use (some may be gone)
        self.time = 0.0
        self.max_speed = 0.0
        self.min_jp = 1.0
        self.bounds = None           # (lo, hi) grid units of the matter last frame
        self.forces = {}             # collider index (in the solver's list) -> (force, torque) from the last frame
        self.warnings = []
        self._buf = {}
        self._tex = {}
        self._k = {}
        self._mats = []              # MatterMaterial per material slot
        self._colours = []           # per slot: colour (linear) or None
        self._mat_bytes = b''
        self._poured = []            # per source: particles poured so far (and the fraction owed)
        self._rng = None
        self._init = None            # the particles as they start (numpy)
        self.surface_ready = False
        self._push_on = False        # the liquid pushing it through its next steps (liquid_push)
        self._cloth = None           # the fabric it meets through this frame (cloth_field)

    # -- set up --------------------------------------------------------------------------------

    @property
    def active(self):
        return self.dims is not None and bool(self.specs)

    def configure(self, specs, box_origin, box_size, resolution=128, max_particles=1_000_000, gravity=9.81, ground=True,
                  closed=False, fps=24.0, duration=10.0, wets=False, heat_speed=HEAT_SPEED):
        """Match the matter to the scene's sources and box. Returns True when it has to start again."""
        specs = list(specs or [])
        key = (tuple((s.material, s.shape, tuple(map(float, s.pos)), tuple(map(float, s.size)), float(s.yaw),
                      tuple(map(float, s.velocity)), float(s.release), bool(s.pour), float(s.rate), float(s.start),
                      float(s.stop), None if s.colour is None else tuple(map(float, s.colour)), float(s.stiffness), int(s.seed),
                      float(s.temperature), str(s.mesh), _stamp(s.mesh))
                     for s in specs),
               tuple(map(float, box_origin)), tuple(map(float, box_size)), int(resolution), int(max_particles), float(gravity),
               bool(ground), bool(closed), float(fps), float(duration), bool(wets), float(heat_speed))
        if not specs:
            changed = self.dims is not None
            self.release()
            self.specs = []
            self.key = None
            return changed
        if key == self.key:
            return False
        self.key = key
        self.specs = specs
        self.gravity = float(gravity)
        self.ground = bool(ground)
        self.closed = bool(closed)
        size = np.asarray(box_size, float)
        self.dx = float(size.max()) / max(int(resolution), 8)
        # (the grid reaches MARGIN nodes past the box all round, so matter resting on the ground or against a closed
        # side is well inside it)
        dims = tuple(int(x) for x in np.ceil(size / self.dx).astype(int) + 1 + 2 * MARGIN)
        self.box = (np.asarray(box_origin, float), np.asarray(box_origin, float) + size)
        self.origin = np.asarray(box_origin, float) - MARGIN * self.dx
        self.warnings = []
        # materials: one slot per material and colour
        slots, self._mats, self._colours = {}, [], []
        self._slot = []
        for s in specs:
            k = (s.material, None if s.colour is None else tuple(s.colour), float(s.stiffness))
            if k not in slots and len(slots) < MAX_MATS - 1:
                slots[k] = len(slots)
                self._mats.append(material(s.material))
                self._colours.append((k[1], k[2]))
            self._slot.append(slots.get(k, 0))
        # (with a liquid in the box: sand and water. Dry sand the liquid wets becomes damp, damp sand it seeps into
        # soaked, and soaked sand away from it drains back to damp: per slot (its damp slot, its soaked slot, what it
        # is: 0 dry, 1 damp, 2 soaked, -1 none of these), mpm_wet.wgsl)
        self._wet = [[-1.0, -1.0, -1.0, 0.0] for _ in range(MAX_MATS)]
        if wets:
            def slot_of(k):
                if k not in slots:
                    if len(slots) >= MAX_MATS - 1:
                        return None
                    slots[k] = len(slots)
                    self._mats.append(material(k[0]))
                    self._colours.append((k[1], k[2]))
                return slots[k]

            for k in [k for k in slots if k[0] in ('sand', 'wet_sand')]:
                c, st = k[1], k[2]
                damp = ('wet_sand', None if c is None else tuple(0.42 * x for x in c), st) if k[0] == 'sand' else k
                d = slot_of(damp)
                if d is None:
                    continue
                s = slot_of(('soaked_sand', None if damp[1] is None else tuple(0.85 * x for x in damp[1]), st))
                if k[0] == 'sand':
                    self._wet[slots[k]] = [float(d), -1.0, 0.0, 0.0]
                self._wet[d] = [float(d), float(-1 if s is None else s), -1.0 if s is None else 1.0, 0.0]
                if s is not None:
                    self._wet[s] = [float(d), float(s), 2.0, 0.0]
        # things that melt: each one's melt (or what a melt sets into) gets a slot too, in the same colour; per slot
        # (melts at, its melt's slot, what it sets into, how fast it takes on the air's temperature) and (how fast its
        # heat evens out, how much faster the water cools it), mpm_heat.wgsl
        for k in list(slots):
            m = material(k[0])
            for other in (m.melt, m.freeze, m.burns_to):
                if other and (other, k[1], k[2]) not in slots and len(slots) < MAX_MATS - 1:
                    slots[(other, k[1], k[2])] = len(slots)
                    self._mats.append(material(other))
                    self._colours.append((k[1], k[2]))
        # (its surface is a particle deep, half a node spacing: the rates follow from that and the material's constants,
        # heat_speed times as fast as for real)
        self._heat = [[0.0, -1.0, -1.0, 0.0] for _ in range(MAX_MATS)]
        self._cond = [[0.0, 1.0, 0.0, 0.0] for _ in range(MAX_MATS)]
        self._burn = [[0.0, 0.0, 0.0, -1.0] for _ in range(MAX_MATS)]
        self._ash = [0.0] * MAX_MATS
        self._flames = [0.0] * MAX_MATS
        skin = self.dx / PER_AXIS
        for k, slot in slots.items():
            m = material(k[0])
            if m.heat_capacity <= 0.0:
                continue
            melt, freeze = slots.get((m.melt, k[1], k[2]), -1), slots.get((m.freeze, k[1], k[2]), -1)
            per_k = heat_speed / (m.density * m.heat_capacity * skin)        # K/s for a W/m^2 into its surface
            self._heat[slot] = [m.melts_at, float(melt), float(freeze), HEAT_CONVECTION * per_k]
            self._cond[slot] = [heat_speed * m.diffusivity / skin ** 2, m.water_cools, m.absorbs * per_k,
                                m.density * m.heat_capacity * math.sqrt(m.diffusivity)]      # (effusivity)
            if m.burns_at > 0.0:
                self._burn[slot] = [m.burns_at, m.burn_rate, m.burn_temp, float(slots.get((m.burns_to, k[1], k[2]), -1))]
                self._flames[slot] = m.flames
                self._ash[slot] = m.ash_share
        self.thermal = any(m.heat_capacity > 0.0 for m in self._mats)
        self._mat_bytes = self._pack_materials()
        # the particles each source makes: its body at the start, or its stream over the shot
        self._rng = np.random.default_rng(12345)
        bodies = []
        need = 0
        vp = (self.dx / PER_AXIS) ** 3
        for n, s in enumerate(specs):
            if s.pour:
                t = max(min(s.stop, duration) - max(s.start, 0.0), 0.0)
                need += int(s.rate * t / vp) + 64
                bodies.append(None)
            else:
                pts = self._body(s, n, dims)
                bodies.append(pts)
                need += len(pts)
        cap = min(need, int(max_particles))
        if need > max_particles:
            self.warnings.append(f'The matter needs {need:,} particles; only {int(max_particles):,} are made (Matter › Most '
                                 'particles). Lower Matter › Detail for fewer, bigger particles.')
        cap = max(int(math.ceil(cap / 64.0) * 64), 64)
        init = np.concatenate([b for b in bodies if b is not None] or [np.zeros((0, 32), np.float32)])[:cap]
        if dims != self.dims or cap != self.capacity:
            self.release()
            self.dims, self.capacity = dims, cap
            self._allocate()
            self._compile()
        self._init = init
        self.reset()
        log.info('Matter: %d sources, grid %s at %.1f mm, %d particles to start (%d slots)', len(specs), dims, self.dx * 1e3,
                 len(init), cap)
        return True

    def _body(self, s: MatterSpec, n, dims):
        rng = np.random.default_rng(1000 + 7 * n + int(s.seed))
        spacing = self.dx / PER_AXIS
        sdf = None
        if s.shape == 'mesh':
            sdf = self._mesh_sdf(s, spacing)
            if sdf is None:
                return particles(np.zeros((0, 3)), self._slot[n], np.zeros(3), 0.0, rng)
        try:
            local = fill_points(s.shape, s.size, spacing, rng, sdf)
        except ValueError:
            self.warnings.append(f'The mesh {s.mesh} is too big to fill at this Matter detail: it is left out.')
            local = np.zeros((0, 3))
        if not len(local):
            local = np.zeros((1, 3))
        world = _yaw(local, s.yaw) + np.asarray(s.pos, float)
        xs = (world - self.origin) / self.dx
        inside = np.all((xs >= 2.0) & (xs < np.asarray(dims) - 3.0), axis=1)
        if not inside.all():
            self.warnings.append(f'Some of the {material(s.material).label.lower()} is outside the box: it is left out.')
        xs = xs[inside]
        return particles(xs, self._slot[n], np.asarray(s.velocity, float), float(s.release), rng, float(s.temperature))

    def _mesh_sdf(self, s, spacing):
        """The baked distance of the mesh a body fills (mesh.load_or_bake: cached on disk), fine enough for its particles,
        or None (with a warning) when it cannot be read."""
        from .mesh import MeshError, load_or_bake, mesh_bounds
        if not s.mesh:
            self.warnings.append('A body of matter shaped as a mesh has no mesh file: it is left out.')
            return None
        try:
            lo, hi = mesh_bounds(s.mesh)
            extent = float(np.max((np.asarray(hi, float) - np.asarray(lo, float)) * np.abs(np.asarray(s.size, float))))
            res = int(np.clip(extent / spacing * 0.75, 32, 256))
            return load_or_bake(self.gpu, s.mesh, resolution=res)
        except (OSError, MeshError, ValueError) as ex:
            self.warnings.append(f'The mesh {s.mesh} could not be read ({ex}): that matter is left out.')
            return None

    def _pack_materials(self):
        u = Uniforms()
        for k in range(MAX_MATS):
            if k < len(self._mats):
                m = self._mats[k]
                colour, stiff = self._colours[k]
                mu, la = m.mu * stiff, m.lam * stiff
                model = MODELS[m.model]
                if m.model == 'sand':
                    sp = math.sin(math.radians(min(friction_angle(m.angle), 75.0)))
                    b = (math.sqrt(2.0 / 3.0) * 2.0 * sp / (3.0 - sp), m.cohesion, 0.0, 0.0)
                elif m.model == 'snow':
                    b = (m.theta_c, m.theta_s, m.xi, 0.0)
                else:
                    b = (m.yield_stress, m.relax, 1.0 if m.tension else 0.0, 0.0)
                u.v4(model, mu, la, m.density).v4(*b).v4(m.friction, m.h_max, m.metal, m.sticks) \
                 .v4(*(colour or m.colour), m.roughness) \
                 .v4(m.clear, m.sparkle, m.wrap, m.variation)
            else:
                u.v4(0.0, 1.0, 1.0, 1000.0).v4().v4().v4().v4()
        return u.data

    def _allocate(self):
        g = self.gpu
        nx, ny, nz = self.dims
        nodes = nx * ny * nz
        self._buf['P'] = g.buffer(self.capacity * PARTICLE_BYTES, 'matter-particles')
        self._buf['C'] = g.buffer(self.capacity * 8, 'matter-compact')
        self._buf['CV'] = g.buffer(self.capacity * 8, 'matter-compact-view')
        self._buf['CT'] = g.buffer(self.capacity * 4, 'matter-temperatures')
        self._buf['CTV'] = g.buffer(self.capacity * 4, 'matter-temperatures-view')
        self._buf['HG'] = g.buffer(nodes * 2 * 4, 'matter-heat-grid')
        self._buf['HL'] = g.buffer(HEAT_LIGHTS * 32, 'matter-heat-sources')
        self._buf['HLC'] = g.buffer(16, 'matter-heat-source-count')
        self._buf['G'] = g.buffer(nodes * 5 * 4, 'matter-grid')    # (mpm_common.wgsl NODE)
        self._buf['S'] = g.buffer(nodes * 13 * 4, 'matter-surface-sums')
        self._buf['react_i'] = g.buffer((16 + MAX_PIECES) * 6 * 4, 'matter-react-step')   # (objects', then pieces')
        self._buf['react'] = g.buffer(16 * 6 * 4, 'matter-react')
        self._buf['stats'] = g.buffer(8 * 4, 'matter-stats')
        self._tex['vel'] = g.texture3d(self.dims, 'rgba32float', 'matter-vel')
        self._tex['phi'] = g.texture3d(self.dims, 'r32float', 'matter-phi')
        self._tex['phi2'] = g.texture3d(self.dims, 'r32float', 'matter-phi2')
        self._tex['seed'] = g.texture3d(self.dims, 'rgba16float', 'matter-seeds')
        self._tex['seed2'] = g.texture3d(self.dims, 'rgba16float', 'matter-seeds2')
        self._tex['look0'] = g.texture3d(self.dims, 'rgba16float', 'matter-look0')
        self._tex['look1'] = g.texture3d(self.dims, 'rgba16float', 'matter-look1')
        self._tex['look2'] = g.texture3d(self.dims, 'rgba16float', 'matter-look2')
        self._tex['surf'] = g.texture3d(self.dims, 'rgba16float', 'matter-surface')

    def release(self):
        for b in self._buf.values():
            b.destroy()
        for t in self._tex.values():
            t.destroy()
        self._buf, self._tex = {}, {}
        self.dims = None
        self.capacity = 0
        self.count = 0
        self.surface_ready = False

    def _compile(self):
        g = self.gpu
        P = (64, 1, 1)
        self._k['p2g'] = g.kernel('mpm_p2g.wgsl', ['rbuf', 'buf', 'rbuf', 'buf'], workgroup=P)
        self._k['grid'] = g.kernel('mpm_grid.wgsl', ['utex3d', 'rbuf', 'st3d:rgba32float:w', 'buf', 'utex3d', 'utex3d',
                                                     'utex3d', 'utex3d', 'rbuf', 'buf'])
        cm = ['rbuf', 'rbuf', 'rbuf', 'rbuf', 'buf', 'buf', 'buf', 'rbuf']
        self._k['cm_splat'] = g.kernel('cloth_matter.wgsl', cm, 'splat', workgroup=P)
        self._k['cm_gather'] = g.kernel('cloth_matter.wgsl', cm, 'gather', workgroup=P)
        self._k['pieces_clear'] = g.kernel('mpm_pieces.wgsl', ['st3d:r32float:w'], 'clear')
        self._k['push'] = g.kernel('mpm_liquid.wgsl', ['utex3d', 'utex3d', 'utex3d', 'st3d:rgba32float:w',
                                                       'st3d:rgba32float:w'])
        self._k['g2p'] = g.kernel('mpm_g2p.wgsl', ['buf', 'utex3d', 'buf', 'rbuf', 'buf'], workgroup=P)
        self._k['react'] = g.kernel('mpm_react.wgsl', ['buf', 'buf'], workgroup=P)
        self._k['compact'] = g.kernel('mpm_compact.wgsl', ['rbuf', 'buf', 'buf'], workgroup=P)
        self._k['blast'] = g.kernel('mpm_blast.wgsl', ['buf', 'utex3d'], workgroup=P)
        self._k['melt'] = g.kernel('mpm_melt.wgsl', ['buf', 'tex3d', 'smp', 'buf', 'buf', 'rbuf', 'utex3d'], workgroup=P)
        self._k['wet'] = g.kernel('mpm_wet.wgsl', ['buf', 'utex3d', 'utex3d'], workgroup=P)
        heat = ['buf', 'buf', 'tex3d', 'smp', 'utex3d', 'utex3d', 'buf', 'buf', 'utex3d']
        self._k['heat_lights'] = g.kernel('mpm_heat.wgsl', heat, 'lights', workgroup=(4, 4, 4))
        self._k['heat_splat'] = g.kernel('mpm_heat.wgsl', heat, 'splat', workgroup=P)
        self._k['heat'] = g.kernel('mpm_heat.wgsl', heat, 'main', workgroup=P)
        self._k['wd_init'] = g.kernel('mpm_wetdist.wgsl', ['utex3d', 'utex3d', 'st3d:r32float:w'], 'init')
        self._k['wd_step'] = g.kernel('mpm_wetdist.wgsl', ['utex3d', 'utex3d', 'st3d:r32float:w'], 'step')
        self._k['surf_p2g'] = g.kernel('mpm_surf_p2g.wgsl', ['rbuf', 'buf', 'rbuf'], workgroup=P)
        self._k['surf_norm'] = g.kernel('mpm_surf_norm.wgsl', ['rbuf', 'st3d:r32float:w', 'st3d:rgba16float:w',
                                                               'st3d:rgba16float:w', 'st3d:rgba16float:w'])
        self._k['surf_smooth'] = g.kernel('mpm_surf_smooth.wgsl', ['utex3d', 'st3d:r32float:w'])
        self._k['jfa_init'] = g.kernel('mpm_jfa.wgsl', ['utex3d', 'st3d:rgba16float:w'], 'init')
        self._k['jfa_step'] = g.kernel('mpm_jfa.wgsl', ['utex3d', 'st3d:rgba16float:w'], 'step')
        self._k['surf_pack'] = g.kernel('mpm_surf_pack.wgsl', ['utex3d', 'utex3d', 'utex3d', 'st3d:rgba16float:w'])

    def reset(self):
        """Back to the start: the bodies of matter where they begin, nothing poured yet."""
        if self.dims is None:
            return
        self.time = 0.0
        self.max_speed = max([float(np.linalg.norm(s.velocity)) for s in self.specs] + [1.0])
        self.min_jp = 1.0
        self.forces = {}
        self.bounds = None
        self._poured = [0.0 for _ in self.specs]
        self._pour_rng = np.random.default_rng(777)
        self.count = len(self._init)
        if self.count:
            self.gpu.write_buffer(self._buf['P'], self._init)
        self.gpu.write_buffer(self._buf['react'], np.zeros(96, np.float32))
        self.gpu.write_buffer(self._buf['react_i'], np.zeros(96, np.int32))
        self.surface_ready = False
        self._view = None            # what the surface is drawn from: None the live particles, else a cached frame's
        self._compact(None)

    # -- stepping -------------------------------------------------------------------------------

    def step_for(self, frame_dt):
        """The step its stiffest material and fastest particle allow (s), and how many fit in a frame."""
        hard = max([math.exp(min(m.xi * (1.0 - self.min_jp), math.log(max(m.h_max, 1.0)))) if m.model == 'snow' else 1.0
                    for m in self._mats] + [1.0])
        sound = max([m.sound(hard) * math.sqrt(st) for m, (_, st) in zip(self._mats, self._colours)] + [1.0])
        dt = min(CFL_SOUND * self.dx / sound, CFL_MOVE * self.dx / max(self.max_speed, 1e-3), frame_dt)
        n = max(1, int(math.ceil(frame_dt / dt - 1e-9)))
        return frame_dt / n, n

    def _pour(self, fdt):
        """Particles streaming from the nozzles over the next fdt seconds, spread along their flow so the stream is even."""
        vp = (self.dx / PER_AXIS) ** 3
        new = []
        t0, t1 = self.time, self.time + fdt
        for n, s in enumerate(self.specs):
            if not s.pour:
                continue
            a, b = max(t0, s.start), min(t1, s.stop)
            if b <= a:
                continue
            self._poured[n] += s.rate * (b - a) / vp
            k = int(self._poured[n])
            self._poured[n] -= k
            if k <= 0:
                continue
            rng = self._pour_rng
            r = float(abs(s.size[0]))
            ang = rng.random(k) * 2.0 * math.pi
            rad = r * np.sqrt(rng.random(k))
            v = np.asarray(s.velocity, float)
            sp = float(np.linalg.norm(v))
            d = v / sp if sp > 1e-6 else np.array([0.0, -1.0, 0.0])
            side = np.cross(d, [0.0, 1.0, 0.0] if abs(d[1]) < 0.9 else [1.0, 0.0, 0.0])
            side /= np.linalg.norm(side)
            up = np.cross(side, d)
            along = rng.random(k) * max(sp, 0.5) * (b - a)
            pts = (np.asarray(s.pos, float) + np.outer(rad * np.cos(ang), side) + np.outer(rad * np.sin(ang), up)
                   + np.outer(along, d))
            xs = (pts - self.origin) / self.dx
            ok = np.all((xs >= 2.0) & (xs < np.asarray(self.dims) - 3.0), axis=1)
            if ok.any():
                new.append(particles(xs[ok], self._slot[n], v, 0.0, rng, float(s.temperature)))
        if not new:
            return
        P = np.concatenate(new)
        room = self.capacity - self.count
        if room <= 0:
            return
        P = P[:room]
        self.gpu.write_buffer(self._buf['P'], P, offset=self.count * PARTICLE_BYTES)
        self.count += len(P)

    def _particle_u(self, dt, t_end):
        return Uniforms().v4(*self.dims, self.count).v4(dt, self.dx, self._fabric(), t_end).raw(self._mat_bytes)

    def _fabric(self):
        """The kernels' "fabric": 0 for none, else 1 + how long since its sheet was laid (s; mpm_common.wgsl SHEET_N)."""
        return 1.0 + max(self.time - self._cloth_t0, 0.0) if self._cloth is not None else 0.0

    def advance(self, frame_dt, colliders_at=None, meshes_atlas=None, pack=None):
        """Simulate frame_dt seconds in one batch. colliders_at(f): the colliders (ColliderGPU list, the solver's order)
        at fraction f of the frame; pack(u, colliders): adds them to a uniform block (solver.pack_colliders). The push
        on each object over the frame ends up in self.forces."""
        if not self.active:
            return
        dt, n = self.begin(frame_dt)
        last, gu = None, None
        with self.gpu.batch() as b:
            for i in range(n):
                cols = colliders_at((i + 0.5) / n) if colliders_at is not None else []
                if cols is not last or gu is None:
                    gu = self._grid_u(dt, cols, pack)
                    last = cols
                self._substep(b, dt, gu, meshes_atlas, fold=True)
        react = np.frombuffer(self.gpu.read_buffer(self._buf['react']), np.float32).reshape(16, 6)
        self.forces = self._react_forces(react, frame_dt)
        self.end()

    def begin(self, frame_dt):
        """Start a frame: pour what the nozzles pour in it, and clear its measures. Returns its step and step count."""
        self._pour(frame_dt)
        dt, n = self.step_for(frame_dt)
        self._frame_dt = frame_dt
        g = self.gpu
        g.write_buffer(self._buf['react'], np.zeros(96, np.float32))
        g.write_buffer(self._buf['react_i'], np.zeros(96, np.int32))
        g.write_buffer(self._buf['stats'], np.array([0, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0, 0, 0,
                                                     np.float32(1.0).view(np.uint32)], np.uint32))
        self.forces = {}
        return dt, n

    MELT_RATE = 7.0e-4     # per s per K above freezing: in flames (some 1400 K above) snow melts in about a second

    def melts(self):
        """Whether any of the matter melts (snow)."""
        return self.active and bool(self.count) and any(m.model == 'snow' for m in self._mats)

    def melt(self, b, dt, gas=None, liquid=None, ambient_k=293.0, flame_k=1650.0, colliders=(), touch=(), meshes=None):
        """Snow melting through dt seconds (mpm_melt.wgsl), in batch b: in the gas of `gas` (a Solver; None: the ambient
        air) and on the objects it touches warmer than freezing (colliders, touch: as heat()), its water joining
        `liquid` (a Liquid; None: it is gone)."""
        if not self.melts():
            return
        g = self.gpu
        live = liquid is not None and getattr(liquid, 'dims', None) is not None and liquid.capacity > 0
        lp = liquid.h ** 3 / max(int(liquid._prm.ppc), 1) if live else 1.0
        vp = (self.dx / PER_AXIS) ** 3
        w = np.zeros(16)
        for k, m in enumerate(self._mats[:15]):
            if m.model == 'snow':
                w[k] = vp * m.density / 1000.0 / lp if live else 1.0      # (the water it holds, in liquid particles)
        gas_on = gas is not None and gas.dims is not None
        u = Uniforms().v4(*self.origin, self.dx)
        u.raw((gas._grid(0.0) if gas_on else Uniforms().v4(1, 1, 1, 1).v4().v4()).data)
        u.v4(*(liquid.origin if live else (0.0, 0.0, 0.0)), liquid.h if live else 1.0)
        self._melt_seed = getattr(self, '_melt_seed', 0) + 1
        u.v4(self.count, dt, liquid.capacity if live else 0, self._melt_seed % 100000)
        u.v4(ambient_k, flame_k, self.MELT_RATE, 1.0 if gas_on else 0.0)
        u.raw(w)
        cols = self._pack_touch(u, colliders, touch, meshes)
        if getattr(self, '_melt_dummy', None) is None:
            t = g.texture3d((1, 1, 1), 'rgba16float', 'matter-melt-no-gas')
            g.upload(t, np.zeros((1, 1, 1, 4), np.float16))
            self._melt_dummy = (t, g.buffer(64, 'matter-melt-no-liquid'), g.buffer(16, 'matter-melt-no-counters'),
                                g.buffer(16, 'matter-melt-no-free'))
        dt_, dp, dc, df = self._melt_dummy
        res = [self._buf['P'], gas.scal[0] if gas_on else dt_, g.linear, liquid.parts if live else dp,
               liquid.ctr if live else dc, liquid.freelist if live else df,
               meshes.atlas if (meshes is not None and cols) else self._empty_atlas()]
        b.run(self._k['melt'], res, u, groups=groups_1d(self.count))
        self._compact(b)            # (drawn as it is now, melted snow gone)
        self.surface_ready = False

    def burns(self):
        """Whether any of the matter burns (dry leaves, sawdust, coal)."""
        return self.active and bool(self.count) and any(r[0] > 0.0 for r in getattr(self, '_burn', []))

    def fire_table(self, fuel_per_kg, smoke, ambient_k, flame_k):
        """Per material slot, what a particle of it burning in the air gives the fire (mpm_fire.wgsl): (fuel, F m^3/s:
        the fuel of what of it burns away a second, times its flames; its heat, field temperature; its smoke, /s; _)."""
        vol = (self.dx / PER_AXIS) ** 3
        out = []
        for k in range(MAX_MATS):
            if k < len(self._mats) and self._burn[k][0] > 0.0:
                m = self._mats[k]
                fuel = m.density * vol * m.burn_rate * fuel_per_kg * self._flames[k]
                heat = (m.burn_temp - ambient_k) / max(flame_k - ambient_k, 1.0)
                out += [fuel, min(max(heat, 0.0), 1.5), fuel * smoke, 0.0]
            else:
                out += [0.0, 0.0, 0.0, 0.0]
        return out

    @staticmethod
    def _pack_touch(u, colliders, touch, meshes):
        """The objects matter can touch, for heat() and melt(): pack_colliders' block, then each one's (temperature K,
        effusivity). Returns the colliders packed."""
        from .solver import MAX_COLLIDERS, pack_colliders
        cols = list(colliders or [])[:MAX_COLLIDERS]
        pack_colliders(u, cols, meshes)
        touch = list(touch or [])[:len(cols)]
        for i in range(MAX_COLLIDERS):
            u.v4(*(touch[i] if i < len(touch) else (0.0, 0.0)))
        return cols

    def heats(self):
        """Whether any of the matter takes on heat, gives it off or melts (wax, chocolate, metal)."""
        return self.active and bool(self.count) and getattr(self, 'thermal', False)

    def heat(self, b, dt, gas=None, liquid=None, ambient_k=293.0, flame_k=1650.0, colliders=(), touch=(), meshes=None):
        """Heat through dt seconds (mpm_heat.wgsl), in batch b: its heat evening out through it, its surface taking on
        the temperature of the gas of `gas` (a Solver; None: the ambient air), the water of `liquid` (a Liquid, or
        None) and the objects it touches (colliders: ColliderGPU, the solver's order; touch: each one's (temperature K,
        effusivity), Scene.collider_heat), and melting and setting at the melting point."""
        if not self.heats():
            return
        g = self.gpu
        gas_on = gas is not None and gas.dims is not None
        live = liquid is not None and getattr(liquid, 'dims', None) is not None
        u = Uniforms().v4(*self.origin, self.dx).v4(*self.dims, self.count)
        u.raw((gas._grid(0.0) if gas_on else Uniforms().v4(1, 1, 1, 1).v4().v4()).data)
        u.v4(*(liquid.origin if live else (0.0, 0.0, 0.0)), liquid.h if live else 1.0)
        u.v4(*(liquid.dims if live else (1, 1, 1)), 1.0 if live else 0.0)
        u.v4(dt, ambient_k, flame_k, 1.0 if gas_on else 0.0)
        u.v4(HEAT_BLOCK, 0.2, FLAME_ABSORPTION, HEAT_LIGHTS)
        u.raw([x for h in self._heat for x in h]).raw([x for c in self._cond for x in c]).raw([x for r in self._burn for x in r])
        u.raw(self._ash)
        cols = self._pack_touch(u, colliders, touch, meshes)
        if getattr(self, '_heat_dummy', None) is None:
            t = g.texture3d((1, 1, 1), 'rgba16float', 'matter-heat-no-gas')
            g.upload(t, np.zeros((1, 1, 1, 4), np.float16))
            self._heat_dummy = (t, g.texture3d((1, 1, 1), 'r32float', 'matter-heat-none'))
        tg, tn = self._heat_dummy
        res = [self._buf['P'], self._buf['HG'], gas.scal[0] if gas_on else tg, g.linear,
               gas.sdf if gas_on else tn, liquid.TYPE[0] if live else tn, self._buf['HL'], self._buf['HLC'],
               meshes.atlas if (meshes is not None and cols) else self._empty_atlas()]
        b.clear_buffer(self._buf['HLC'])
        if gas_on:
            # (the fire as heat sources, for its radiant heat)
            b.run(self._k['heat_lights'], res, u, groups=tuple(-(-int(d) // (4 * HEAT_BLOCK)) for d in gas.dims))
        if any(c[0] > 0.0 for c in self._cond):
            b.clear_buffer(self._buf['HG'])
            b.run(self._k['heat_splat'], res, u, groups=groups_1d(self.count))
        b.run(self._k['heat'], res, u, groups=groups_1d(self.count))
        self._compact(b)            # (drawn as it is now: its glow, what has melted)
        self.surface_ready = False

    WET_TIME = 0.5         # s: how long a grain of dry sand next to liquid takes to get damp
    SOAK_TIME = 1.0        # s: how long the water takes to soak damp sand a cell in from it (n cells in: n + 1 times)
    SOAK_DEPTH = 8         # the liquid's cells: the furthest into the sand the water soaks it
    DRAIN_TIME = 4.0       # s: how long soaked sand away from the water takes to drain back to damp
    DRAIN_FROM = 3         # the liquid's cells: how far from the water soaked sand drains
    BED_DRAG = 0.05        # the drag coefficient of the liquid running over it (the stress rho C |du| du): more than
                           # a real bed's, as its grains are far bigger than sand's

    def liquid_push(self, liquid, rho=1000.0):
        """What `liquid` (a Liquid) does to the matter through the frame's steps (until end()), from the liquid's last
        pressure solve (mpm_liquid.wgsl): its pressure where it meets the matter, the buoyancy of grains in it, and its
        drag. None: nothing."""
        self._push_on = False
        dt = float(getattr(liquid, 'step_dt', 0.0) or 0.0) if liquid is not None else 0.0
        if not self.active or not self.count or dt <= 0.0 or getattr(liquid, 'dims', None) is None:
            return
        g = self.gpu
        if 'push' not in self._tex:
            self._tex['push'] = g.texture3d(self.dims, 'rgba32float', 'matter-liquid-push')
            self._tex['flow'] = g.texture3d(self.dims, 'rgba32float', 'matter-liquid-flow')
        u = (Uniforms().raw(liquid._grid(dt).data).v4(*self.origin, self.dx).v4(*self.dims)
             .v4(rho / dt, 0.5 * liquid._prm.gravity * dt * liquid.h, rho * self.BED_DRAG,
                 1.0 if all(m.porous for m in self._mats) else 0.0))
        with g.batch() as b:
            b.run(self._k['push'], [liquid.X, liquid.TYPE[0], liquid.vel_tex, self._tex['push'], self._tex['flow']], u,
                  self.dims)
        self._push_on = True

    def wets(self):
        """Whether any of the matter gets wet (sand, with a liquid in the box)."""
        return self.active and bool(self.count) and any(w[2] >= 0.0 for w in getattr(self, '_wet', []))

    def wet(self, b, dt, liquid):
        """Sand and the water of `liquid` (a Liquid) through dt seconds (mpm_wet.wgsl), in batch b: dry sand next to it
        damp after WET_TIME; the water seeping on into damp sand, soaking it SOAK_TIME a cell in from it; soaked sand away
        from it damp again after DRAIN_TIME."""
        if not self.wets() or liquid is None or getattr(liquid, 'dims', None) is None:
            return
        # how far through the sand each of the liquid's cells is from the water (mpm_wetdist.wgsl)
        dims = tuple(int(x) for x in liquid.dims)
        wd = self._tex.get('wd0')
        if wd is None or tuple(wd.size) != dims:
            for k in ('wd0', 'wd1'):
                if k in self._tex:
                    self._tex[k].destroy()
                self._tex[k] = self.gpu.texture3d(dims, 'r32float', f'matter-{k}')
        w0, w1 = self._tex['wd0'], self._tex['wd1']
        du = Uniforms().v4(*dims)
        b.run(self._k['wd_init'], [liquid.TYPE[0], w1, w0], du, dims)
        for _ in range(self.SOAK_DEPTH // 2):
            b.run(self._k['wd_step'], [liquid.TYPE[0], w0, w1], du, dims)
            b.run(self._k['wd_step'], [liquid.TYPE[0], w1, w0], du, dims)
        u = (Uniforms().v4(*self.origin, self.dx).v4(*liquid.origin, liquid.h).v4(*liquid.dims)
             .v4(self.count, dt, self.WET_TIME, self.SOAK_TIME).v4(self.DRAIN_TIME, self.SOAK_DEPTH, self.DRAIN_FROM)
             .raw([x for w in self._wet for x in w]))
        b.run(self._k['wet'], [self._buf['P'], liquid.TYPE[0], w0], u, groups=groups_1d(self.count))
        self._compact(b)            # (drawn as it is now, wet sand wet)
        self.surface_ready = False

    def blast(self, where, kg, depth=0.1):
        """A blast of `kg` of TNT at `where` (fire-local m) throws the matter away from it (mpm_blast.wgsl): the impulse on
        its surface goes into the matter within `depth` metres of it."""
        if not self.active or not self.count or kg <= 0.0:
            return
        self.surface()          # (how deep each particle is)
        u = Uniforms().v4(*self.dims, self.count).v4(*self.origin, self.dx).v4(*where, kg)             .v4(depth, max(kg, 1e-9) ** (1.0 / 3.0) / 3.0).raw(self._mat_bytes)
        with self.gpu.batch() as b:
            b.run(self._k['blast'], [self._buf['P'], self._tex['surf']], u, groups=groups_1d(self.count))
        # (the fastest it can be thrown, for the next step's length: the nearest of it, the lightest of it)
        from .solids import blast_impulse
        bounds = self.world_bounds()
        near = 0.0 if bounds is None else float(np.linalg.norm(np.asarray(where) - np.clip(where, bounds[0], bounds[1])))
        self.max_speed = max(self.max_speed, blast_impulse(kg, near) / (min(m.density for m in self._mats) * depth))

    def step(self, dt, colliders, meshes_atlas=None, pack=None, pieces=None, substeps=1):
        """One step of dt seconds on its own, the objects where `colliders` has them, and broken objects' pieces where
        `pieces` has them ((scene, {collider index: Solids.piece_poses entry}), or None); returns the push on each
        object over it ({index in the solver's collider list: (force, torque about its position)}, and for the pieces
        {('piece', collider index, piece index): (force, torque about the piece's position)}, N and N m). For moving in
        lockstep with the rigid bodies (solids.py), which take the push into their next step. substeps: the step taken
        as that many, the push over all of them."""
        owners, rebake = self._pieces(pieces)
        with self.gpu.batch() as b:
            if rebake:
                b.run(self._k['pieces_clear'], [self._tex['psdf']], Uniforms().v4(*self.dims), self.dims)
                self._bf.bake(b, self._pgrid, 0)
            n_sub = max(1, int(substeps))
            gu = self._grid_u(dt / n_sub, colliders, pack, bool(owners))
            for _ in range(n_sub):
                self._substep(b, dt / n_sub, gu, meshes_atlas, fold=False, pieces_on=bool(owners))
        out = {}
        n = min(len(owners), MAX_PIECES)
        raw = np.frombuffer(self.gpu.read_buffer(self._buf['react_i'], size=(16 + n) * 24), np.int32).reshape(16 + n, 6)
        react, pr = raw[:16], raw[16:]
        if raw.any():
            self.gpu.write_buffer(self._buf['react_i'], np.zeros((16 + n) * 6, np.int32))
        if react.any():
            out = self._react_forces(react.astype(np.float64) / FX_R, dt)
        if owners:
            if pr.any():
                unit = 1000.0 * (self.dx / PER_AXIS) ** 3 / dt
                poses = pieces[1]
                for j in np.nonzero(pr.any(axis=1))[0]:
                    ci, k = owners[j]
                    f = pr[j, :3].astype(np.float64) / FX_R * unit
                    t_origin = pr[j, 3:].astype(np.float64) / FX_R * unit
                    c = np.asarray(poses[ci]['pos'][k], float)
                    out[('piece', ci, k)] = (f, t_origin - np.cross(c, f))
        return out

    def _pieces(self, pieces):
        """Broken objects' pieces onto the matter's grid for the next step (bodyfield.py: only the ones near the matter):
        (each piece's (collider index, piece index), in the order the grid knows them by, or [] when none are near;
        whether to bake them again: only once they have moved a fifth of a node or turned)."""
        if not pieces or not pieces[1] or self.bounds is None:
            self._baked = None
            return [], False
        lo, hi = self.world_bounds()
        pos = np.concatenate([np.asarray(p['pos'], float).reshape(-1, 3) for p in pieces[1].values()])
        quat = np.concatenate([np.asarray(p['quat'], float).reshape(-1, 4) for p in pieces[1].values()])
        if not len(pos) or not np.any(np.all((pos >= lo - 0.5) & (pos <= hi + 0.5), axis=1)):
            self._baked = None
            return [], False
        last = getattr(self, '_baked', None)
        if (last is not None and last[0].shape == pos.shape and np.abs(pos - last[0]).max() < 0.2 * self.dx
                and np.abs(quat - last[1]).max() < 0.01):
            return last[2], False
        owners = self._bake_pieces(pieces, lo, hi)
        self._baked = (pos, quat, owners) if owners else None
        return owners, bool(owners)

    def _bake_pieces(self, pieces, lo, hi):
        from .bodyfield import BodyField
        if getattr(self, '_bf', None) is None:
            self._bf = BodyField(self.gpu)
        if 'psdf' not in self._tex:
            self._tex['psdf'] = self.gpu.texture3d(self.dims, 'r32float', 'matter-pieces-distance')
            self._tex['psvel'] = self.gpu.texture3d(self.dims, 'rgba16float', 'matter-pieces-velocity')
            self._pgrid = _PieceGrid(self)
        margin = 3.0 * self.dx
        if not self._bf.prepare(pieces[0], [pieces[1]], self.dims, self.dx, self.origin - 0.5 * self.dx,
                                region=(lo - margin, hi + margin)):
            return []
        st = self._bf.steps[0]
        if st is None or not any(self._bf.TC_counts):
            return []
        return self._bf.owners[0]

    def end(self):
        """Finish a frame: the particles' 8-byte form for drawing, and what the frame's measures say."""
        g = self.gpu
        self._compact(None)
        st = np.frombuffer(g.read_buffer(self._buf['stats']), np.uint32)
        self.max_speed = max(float(st[0:1].view(np.float32)[0]), 0.05)
        lo, hi = st[1:4].astype(np.int64) - (1 << 20), st[4:7].astype(np.int64) - (1 << 20)
        self.bounds = (lo, hi) if np.all(hi >= lo) else None
        self.min_jp = float(st[7:8].view(np.float32)[0])
        self.surface_ready = False
        self._view = None
        self._push_on = False        # (the liquid's push is set again for each frame: liquid_push)
        if self._cloth is not None:
            self._cloth_push()           # (what it gave the fabric through the frame, for the fabric's next)
            self._cloth = None

    def _react_forces(self, react, seconds):
        """The momentum the objects took (in particles of water times m/s, (16, 6)) as forces over `seconds` (N, N m)."""
        unit = 1000.0 * (self.dx / PER_AXIS) ** 3 / seconds
        return {i: (np.asarray(react[i, :3], float) * unit, np.asarray(react[i, 3:], float) * unit)
                for i in range(16) if np.any(react[i] != 0.0)}

    def _grid_u(self, dt, cols, pack, pieces=False):
        ground_y = self.box[0][1] if self.ground else -1.0e9
        u = Uniforms().v4(*self.dims, 1.0 if self._cloth is not None else 0.0).v4(dt, self.dx, self.gravity, self._wall_friction()) \
            .v4(*self.origin, ground_y).v4(1.0 if self.closed else 0.0, MARGIN, 1.0 if self._push_on else 0.0,
                                           min(m.density for m in self._mats))
        if pack is not None:
            pack(u, cols, pieces) if pieces else pack(u, cols)
        else:
            _pack_none(u)
        return u

    def _substep(self, b, dt, gu, meshes_atlas, fold, pieces_on=False):
        k = self._k
        groups = groups_1d(max(self.count, 1))
        atlas = meshes_atlas if meshes_atlas is not None else self._empty_atlas()
        t_end = self.time + dt
        cf, ct = (self._buf['CF'], self._buf['CTK']) if self._cloth is not None else self._no_cloth()
        b.clear_buffer(self._buf['G'])
        b.run(k['p2g'], [self._buf['P'], self._buf['G'], cf, ct], self._particle_u(dt, self.time), groups=groups)
        push, flow = (self._tex['push'], self._tex['flow']) if self._push_on else (self._no_push(),) * 2
        psdf, psvel = (self._tex['psdf'], self._tex['psvel']) if pieces_on else (self._no_push(),) * 2
        gu.data[3] = self._fabric()          # (n.w: the sheet moves on through the frame)
        b.run(k['grid'], [atlas, self._buf['G'], self._tex['vel'], self._buf['react_i'], push, flow, psdf, psvel, cf, ct],
              gu, self.dims)
        if fold:
            b.run(k['react'], [self._buf['react_i'], self._buf['react']], Uniforms().v4(96, 1.0 / FX_R), groups=(2, 1, 1))
        # (held or let go as at the start of the step in both passes: p2g's momentum is what g2p picks up)
        b.run(k['g2p'], [self._buf['P'], self._tex['vel'], self._buf['stats'], cf, ct], self._particle_u(dt, self.time),
              groups=groups)
        self.time = t_end

    def _compact(self, b):
        """The particles' 8-byte form (mpm_compact.wgsl), for drawing and the frame cache."""
        if not self.count:
            return
        if b is None:
            with self.gpu.batch() as bb:
                self._compact(bb)
            return
        b.run(self._k['compact'], [self._buf['P'], self._buf['C'], self._buf['CT']], Uniforms().v4(*self.dims, self.count),
              groups=groups_1d(self.count))

    def _wall_friction(self):
        return max([m.friction for m in self._mats] + [0.0])

    CLOTH_REACH = 1.5      # node spacings: how far round a vertex of fabric its sheet reaches on the matter's grid

    def cloth_link(self):
        """What the fabric needs to drape over the matter (cloth.py matter_link): (its surface's distance texture, its
        grid's velocity texture, node 0, node spacing, nodes). Build the surface first (surface())."""
        return (self._tex['surf'], self._tex['vel'], tuple(float(x) for x in self.origin), float(self.dx),
                tuple(int(x) for x in self.dims))

    def cloth_field(self, b, cloth):
        """The fabric `cloth` (cloth.Cloth) as a thin sheet on the grid through this frame's steps, which the matter
        cannot pass through (cloth_matter.wgsl splat; in batch b). Its push on the fabric is gathered as the frame
        ends (end())."""
        self._cloth = None
        if not self.active or not self.count or cloth is None or not cloth.active or not cloth.placed:
            return
        nodes = int(np.prod(self.dims))
        if 'CF' not in self._buf:
            self._buf['CF'] = self.gpu.buffer(nodes * 9 * 4, 'matter-cloth-field')
            self._buf['CTK'] = self.gpu.buffer(nodes * 3 * 4, 'matter-cloth-taken')
        b.clear_buffer(self._buf['CF'])
        b.clear_buffer(self._buf['CTK'])
        k = cloth.bufs
        n = cloth.built.n
        u = Uniforms().v4(*self.origin, self.dx).v4(*self.dims, n).v4(self._cloth_reach(cloth), 0.0)
        b.run(self._k['cm_splat'], [k['X'], k['V'], k['N'], k['S'], self._buf['CF'], self._buf['CTK'], k['MP'], k['R']], u,
              (n, 1, 1))
        self._cloth = cloth
        self._cloth_t0 = self.time

    def _cloth_push(self):
        """Each vertex of the fabric's share of what the grid's sheet took from the matter through the frame, as a
        force through the fabric's next (cloth_matter.wgsl gather)."""
        cloth = self._cloth
        if not cloth.active or not cloth.placed or 'CF' not in self._buf:
            return
        k = cloth.bufs
        n = cloth.built.n
        unit = 1000.0 * (self.dx / PER_AXIS) ** 3 / max(getattr(self, '_frame_dt', 1.0 / 24.0), 1e-6)
        u = Uniforms().v4(*self.origin, self.dx).v4(*self.dims, n).v4(self._cloth_reach(cloth), unit, self.gravity)
        with self.gpu.batch() as b:
            b.run(self._k['cm_gather'], [k['X'], k['V'], k['N'], k['S'], self._buf['CF'], self._buf['CTK'], k['MP'],
                                         k['R']], u,
                  (n, 1, 1))

    def _cloth_reach(self, cloth):
        """How far round a vertex its sheet reaches (m): CLOTH_REACH nodes, or past the fabric's own spacing, so a coarse
        cloth leaves no holes between its vertices."""
        return max(self.CLOTH_REACH * self.dx, 1.25 * float(cloth.built.mean_edge))

    def _no_cloth(self):
        if 'no_cloth' not in self._buf:
            self._buf['no_cloth'] = self.gpu.buffer(64, 'matter-no-cloth')
            self._buf['no_cloth_t'] = self.gpu.buffer(64, 'matter-no-cloth-taken')
        return self._buf['no_cloth'], self._buf['no_cloth_t']

    def _no_push(self):
        t = self._tex.get('no_push')
        if t is None:
            t = self._tex['no_push'] = self.gpu.texture3d((1, 1, 1), 'rgba32float', 'matter-no-liquid')
        return t

    def _empty_atlas(self):
        t = self._tex.get('atlas1')
        if t is None:
            t = self._tex['atlas1'] = self.gpu.texture3d((1, 1, 1), 'r32float', 'matter-no-mesh')
        return t

    # -- drawing --------------------------------------------------------------------------------

    def surface(self):
        """Build the matter's surface for drawing (if the particles moved since): (surface, look0, look1) textures on the
        grid, rgba16float: surface = the distance to the surface (m, negative inside), how clear (jelly), sparkly and
        how much light it lets in (snow); look0 = its albedo (linear) and roughness; look2 = its temperature (K, for its
        glow) and how metallic it is."""
        if not self.active:
            return None
        if not self.surface_ready:
            g = self.gpu
            k = self._k
            count, src, src_t = ((self.count, self._buf['C'], self._buf['CT']) if self._view is None
                                 else (self._view[0], self._buf['CV'], self._buf['CTV']))
            temps = self.thermal and (self._view is None or self._view[2])
            mats = Uniforms().v4(*self.dims, count).v4(SURF_R, 1.0 if temps else 0.0).raw(self._mat_bytes)
            band = (SURF_R - SURF_PARTICLE) * self.dx
            with g.batch() as b:
                b.clear_buffer(self._buf['S'])
                b.run(k['surf_p2g'], [src, self._buf['S'], src_t], mats, groups=groups_1d(max(count, 1)))
                b.run(k['surf_norm'], [self._buf['S'], self._tex['phi'], self._tex['look0'], self._tex['look1'],
                                       self._tex['look2']],
                      Uniforms().v4(*self.dims, 0).v4(SURF_R, SURF_PARTICLE, self.dx), self.dims)
                # smoothed a little (it shows the lie of the matter, not its particles)
                phi = self._tex['phi']
                for _ in range(SURF_SMOOTH):
                    other = self._tex['phi2'] if phi is self._tex['phi'] else self._tex['phi']
                    b.run(k['surf_smooth'], [phi, other], Uniforms().v4(*self.dims).v4(band, 0.6), self.dims)
                    phi = other
                # the distance to it from everywhere in the grid (jump flooding), for rays, shadows and occlusion
                b.run(k['jfa_init'], [phi, self._tex['seed']], Uniforms().v4(*self.dims, self.dx).v4(band), self.dims)
                src, dst = self._tex['seed'], self._tex['seed2']
                step = 1 << int(math.ceil(math.log2(max(self.dims)))) - 1
                while step >= 1:
                    b.run(k['jfa_step'], [src, dst], Uniforms().v4(*self.dims, self.dx).v4(band, step), self.dims)
                    src, dst = dst, src
                    step //= 2
                b.run(k['surf_pack'], [phi, self._tex['look1'], src, self._tex['surf']],
                      Uniforms().v4(*self.dims, self.dx).v4(band), self.dims)
            self.surface_ready = True
        return self._tex['surf'], self._tex['look0'], self._tex['look1'], self._tex['look2']

    def world_bounds(self):
        """(lo, hi) of the matter (fire-local m), or None."""
        bounds = self.bounds if self._view is None else self._view[1]
        if bounds is None:
            return None
        lo, hi = bounds
        return self.origin + (lo - 2) * self.dx, self.origin + (hi + 2) * self.dx

    # -- state ---------------------------------------------------------------------------------

    def snapshot(self):
        """The particles as drawn this frame, for the frame cache: (n, 4) uint16 (mpm_compact.wgsl's form), or None."""
        if not self.active or not self.count:
            return None
        return np.frombuffer(self.gpu.read_buffer(self._buf['C'], size=self.count * 8), np.uint16).reshape(-1, 4).copy()

    def snapshot_temperatures(self):
        """The particles' temperatures as drawn this frame (float16 K, in snapshot()'s order), when the matter has heat."""
        if not self.active or not self.count or not self.thermal:
            return None
        return np.frombuffer(self.gpu.read_buffer(self._buf['CT'], size=self.count * 4), np.float32).astype(np.float16)

    def show(self, snap, temps=None):
        """Draw a cached frame's particles (snapshot(), and snapshot_temperatures()'s temps) instead of the live ones
        (until the next step or show_live())."""
        if not self.active:
            return False
        snap = np.ascontiguousarray(snap, np.uint16).reshape(-1, 4)[:self.capacity]
        if len(snap):
            self.gpu.write_buffer(self._buf['CV'], snap)
        if temps is not None and len(temps):
            self.gpu.write_buffer(self._buf['CTV'], np.asarray(temps, np.float32)[:len(snap)])
        live = (snap[:, 3] >> 12) < 15
        bounds = None
        if live.any():
            q = snap[live, :3].astype(np.float64) / 65535.0 * np.asarray(self.dims, float)
            bounds = (np.floor(q.min(0)).astype(np.int64), np.ceil(q.max(0)).astype(np.int64))
        self._view = (len(snap), bounds, temps is not None and len(temps) > 0)
        self.surface_ready = False
        return True

    def show_live(self):
        if self._view is not None:
            self._view = None
            self.surface_ready = False

    def read_particles(self):
        """The live particles (n, 32) float32 (for tests and the cache)."""
        if not self.active or not self.count:
            return np.zeros((0, 32), np.float32)
        P = np.frombuffer(self.gpu.read_buffer(self._buf['P'], size=self.count * PARTICLE_BYTES), np.float32).reshape(-1, 32)
        return P

    def positions(self):
        """Where the live particles are (fire-local m) and their material slots."""
        P = self.read_particles()
        live = P[:, 3] >= 0.0
        return self.origin + P[live, :3].astype(float) * self.dx, P[live, 3].astype(int)

    def state(self):
        """Everything needed to carry on from now."""
        if not self.active:
            return None
        return dict(particles=self.read_particles().copy(), time=float(self.time), poured=list(self._poured),
                    max_speed=float(self.max_speed), min_jp=float(self.min_jp))

    def load_state(self, st):
        if not self.active or st is None:
            return False
        P = np.asarray(st['particles'], np.float32).reshape(-1, 32)
        if len(P) > self.capacity:
            return False
        self.count = len(P)
        if self.count:
            self.gpu.write_buffer(self._buf['P'], P)
        self.time = float(st.get('time', 0.0))
        self._poured = list(st.get('poured', self._poured))
        self.max_speed = float(st.get('max_speed', 1.0))
        self.min_jp = float(st.get('min_jp', 1.0))
        self.surface_ready = False
        return True


class _PieceGrid:
    """The matter's grid as bodyfield.py bakes broken objects' pieces into a solver's: cells centred on its nodes."""

    def __init__(self, m):
        self.dims = tuple(int(x) for x in m.dims)
        self.h = float(m.dx)
        self.origin = np.asarray(m.origin, float) - 0.5 * m.dx
        self.sdf = m._tex['psdf']
        self._svel = m._tex['psvel']

    def _grid(self, dt):
        return Uniforms().v4(*self.dims, self.h).v4(*self.origin, 0.0).v4(0.0, 0.0, 0.0, dt)

    def solid_vel(self):
        return self._svel


def _pack_none(u):
    """No colliders (as solver.pack_colliders with an empty list)."""
    from .solver import pack_colliders
    pack_colliders(u, [])
