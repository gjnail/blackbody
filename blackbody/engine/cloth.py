"""Fabric: cloth that hangs, drapes, blows in the fire's air and burns (Scene.fabrics).

Simulation (GPU, every substep of the fire's step split further so the cloth takes about a thousand
steps a second):
- XPBD in small steps (Macklin et al. 2019: one constraint pass per step, many steps): stretch and
  shear along the threads (stiff when pulled, soft when pushed together, so the cloth wrinkles), and
  isometric bending (Bergou et al. 2006) with its compliance from the fabric's measured bending
  rigidity. Constraints are graph-coloured so each colour runs in parallel as Gauss-Seidel.
- The air: pressure drag across the sheet and skin friction along it, from the simulated gas (hot gas
  is thinner) or the wind outside the box; the cloth in turn holds the air back (cloth_air.wgsl).
- Collisions with the ground, the colliders (their exact shapes, with friction against their
  motion) and itself (a hashed grid of its vertices).
- Heat: the cloth heats toward the gas around it with its own heat capacity, spreads heat along
  itself, catches at its ignition temperature, burns (feeding fuel, heat and smoke to the fire),
  chars and burns through. Synthetics shrink as they soften and melt away.

Materials are real fabrics (areal density, tensile, shear and bending stiffness in SI units from
Kawabata-type measurements, ignition and burning behaviour). A fabric is a panel (curtain, flag,
sheet, banner) or a mesh (OBJ, STL or a USD prim), pinned along an edge, at corners or not at all.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from .gpu import GPU, Program, Uniforms, groups_1d, load_wgsl, SS

log = logging.getLogger('blackbody.cloth')

MAX_FABRICS = 16              # matches cloth_common.wgsl
FLAKES = 3                    # flakes of ash drawn per burnt-off vertex (matches cloth_draw.wgsl)
STEPS_PER_SECOND = 600.0      # cloth substeps per second of simulated time (25 a frame at 24 fps)
ITERATIONS = 4                # constraint passes per substep (multipliers kept, so they converge)
FUEL_PER_KG = 25.0            # fuel (grid units x m^3) a kilogram of pyrolysed cloth gives off
BEND_K = 1.2                  # isometric bending: stiffness = BEND_K * rigidity / (area of the two triangles); matches Peirce's cantilever
COMPRESSION_SOFT = 200.0      # compressed threads are this much more compliant than stretched ones
OVER_RELAX = 1.0              # Gauss-Seidel over-relaxation of the constraint passes
MULTIGRID = True              # panels' bending solved by multigrid (cloth_mg.py) instead of constraint passes
TETHER_SLACK = 0.002          # a tether lets a point drift this share past its rest distance from its pin
HASH_MIN = 1 << 12
COUPLE_REF_FUEL = 2.0         # fuel density (F/s) at which burning cloth's heat is fully in the gas
COUPLE_CHANNELS = 13          # values per coupling cell (cloth_splat.wgsl)
DROPS = 16384                 # drops of water dripping off wet cloth in flight at once (cloth_drops.wgsl)
RADIANT = True                # the fire's radiation heats the cloth (cloth_rad_src, cloth_rad, cloth_radiant)
RAD_KAPPA = 0.2               # (radiant.KAPPA, FLAME_K, MAX_K now) 1/m: absorption coefficient of the flames (a campfire with 1.2 m flames, some 190 kW,
                              # radiates about 40 kW: a fifth to a third, as real fires do)
RAD_FLAME_K = 1250.0          # K: the flames' real temperature, for their radiation (wood and hydrocarbon flames
RAD_MAX_K = 1450.0            # average 1100-1300 K; the look's flame temperature is a colour, often well above it)
RAD_CELLS = 4096              # the coarse grid the gas's radiation is gathered on, at most
C_P = 1300.0                  # J/(kg K), dry textile fibres
H_CONV = 30.0                 # W/(m^2 K) per face, air moving past the cloth


@dataclass(frozen=True)
class Material:
    label: str
    density: float        # kg/m^2
    stretch: float        # N/m, along the threads
    shear: float          # N/m
    bend: float           # N m (bending rigidity per width)
    damping: float        # 1/s
    friction: float
    thickness: float      # m (collision)
    ignition: float       # K
    burn_time: float      # s a spot of it burns for
    burn_temp: float      # K, while it burns (a smouldering fibre like wool stays cooler than its ignition)
    fuel: float           # share of its mass given off as fuel
    smoke: float          # smoke per fuel
    colour: tuple
    sheen: float
    sheen_rough: float
    spec: float = 0.0     # highlight along the threads (silk, satin, nylon)
    spec_rough: tuple = (0.25, 0.6)
    translucency: float = 0.2
    pitch: float = 0.0004  # m between threads
    weave: str = 'plain'   # plain, twill, pile, ripstop, none
    weave_depth: float = 0.5
    soften: float = 0.0    # K (synthetics)
    melt: float = 0.0      # K; 0 = does not melt
    shrink: float = 0.0    # as it softens
    char_shrink: float = 0.2
    spread: float = 4.0e-5  # m^2/s heat along the cloth
    cn: float = 1.2        # pressure drag across the sheet
    ct: float = 0.02       # skin friction along it
    fibre: float = 1500.0  # kg/m^3, the fibre itself (in water: what it weighs once the air is out of it)
    absorb: float = 1.2    # kg of water a kg of it holds, soaked (its pores; dry, they hold air and it floats)
    soak: float = 1.5      # s for it to soak through under water (synthetics and wool shed water at first)
    wick: float = 1.0e-5   # m^2/s: how fast water spreads along it by capillarity (h^2 = wick * t, Lucas-Washburn)
    retain: float = 0.5    # share of a soaking (absorb) it keeps once it has drained: the rest runs down and drips
    tears_at: float = 0.6  # how far past its length a thread stretches before it breaks (with Tears on: cloth_tear.wgsl;
                           # the cloth's soft stretch's, not the fibre's: a sling laden with sand stretches some 20 to 35%)


MATERIALS = {
    'cotton': Material('Cotton (shirting, sheets)', 0.15, 3000.0, 60.0, 1.0e-5, 0.4, 0.5, 0.0008, 620.0, 4.0, 1100.0, 0.9, 0.08,
                       (0.78, 0.74, 0.66), 0.35, 0.55, translucency=0.25, pitch=0.0003,
                       fibre=1540.0, absorb=2.0, wick=1.20e-05, retain=0.55, soak=1.0, tears_at=0.6),
    'linen': Material('Linen', 0.18, 5000.0, 90.0, 2.0e-5, 0.4, 0.5, 0.0009, 620.0, 4.0, 1100.0, 0.9, 0.08,
                      (0.72, 0.66, 0.55), 0.25, 0.6, translucency=0.2, pitch=0.0005,
                      fibre=1500.0, absorb=1.8, wick=1.50e-05, retain=0.5, soak=1.0, tears_at=0.45),
    'silk': Material('Silk', 0.06, 2000.0, 20.0, 1.0e-6, 0.3, 0.3, 0.0004, 840.0, 6.0, 900.0, 0.6, 0.12,
                     (0.62, 0.08, 0.10), 0.25, 0.35, spec=0.6, spec_rough=(0.12, 0.45), translucency=0.45, pitch=0.0001,
                     fibre=1340.0, absorb=1.2, wick=8.00e-06, retain=0.5, soak=2.0, tears_at=0.7),
    'chiffon': Material('Chiffon', 0.035, 800.0, 8.0, 3.0e-7, 0.3, 0.3, 0.0003, 700.0, 2.0, 1050.0, 0.9, 0.1,
                        (0.9, 0.86, 0.84), 0.3, 0.4, spec=0.15, translucency=0.7, pitch=0.00015,
                        fibre=1380.0, absorb=0.6, wick=3.00e-06, retain=0.3, soak=3.0, tears_at=0.5),
    'wool': Material('Wool', 0.28, 1500.0, 40.0, 3.0e-5, 0.6, 0.6, 0.0015, 870.0, 8.0, 700.0, 0.35, 0.2,
                     (0.35, 0.3, 0.26), 0.5, 0.8, translucency=0.08, pitch=0.001, weave='twill',
                     fibre=1310.0, absorb=1.8, wick=2.50e-06, retain=0.6, soak=4.0, tears_at=1.0),
    'denim': Material('Denim', 0.4, 8000.0, 150.0, 1.0e-4, 0.5, 0.6, 0.0012, 640.0, 6.0, 1100.0, 0.9, 0.1,
                      (0.12, 0.2, 0.36), 0.15, 0.6, translucency=0.02, pitch=0.0006, weave='twill', weave_depth=0.7,
                      fibre=1540.0, absorb=1.6, wick=8.00e-06, retain=0.6, soak=2.0, tears_at=0.8),
    'canvas': Material('Canvas (tents, sails)', 0.4, 10000.0, 300.0, 3.0e-4, 0.6, 0.6, 0.0012, 640.0, 7.0, 1100.0, 0.9, 0.1,
                       (0.66, 0.6, 0.47), 0.1, 0.7, translucency=0.03, pitch=0.0008, weave_depth=0.7,
                       fibre=1540.0, absorb=1.4, wick=8.00e-06, retain=0.6, soak=3.0, tears_at=1.0),
    'velvet': Material('Velvet', 0.3, 3000.0, 60.0, 4.0e-5, 0.7, 0.7, 0.002, 620.0, 5.0, 1100.0, 0.9, 0.12,
                       (0.35, 0.03, 0.06), 1.0, 0.3, translucency=0.02, pitch=0.0003, weave='pile',
                       fibre=1450.0, absorb=2.4, wick=1.00e-05, retain=0.6, soak=1.5, tears_at=0.7),
    'polyester': Material('Polyester', 0.12, 4000.0, 50.0, 1.5e-5, 0.3, 0.35, 0.0007, 760.0, 3.0, 1050.0, 1.0, 0.35,
                          (0.8, 0.8, 0.82), 0.3, 0.45, spec=0.2, translucency=0.2, pitch=0.0003,
                          soften=470.0, melt=530.0, shrink=0.3, fibre=1380.0, absorb=0.6, wick=1.50e-06, retain=0.25, soak=5.0,
                          tears_at=1.0),
    'nylon': Material('Nylon (flags)', 0.07, 3000.0, 30.0, 2.0e-6, 0.2, 0.3, 0.0004, 700.0, 2.5, 1050.0, 1.0, 0.3,
                      (0.75, 0.08, 0.06), 0.25, 0.4, spec=0.3, spec_rough=(0.15, 0.5), translucency=0.35, pitch=0.0004,
                      weave='ripstop', soften=450.0, melt=490.0, shrink=0.35, fibre=1140.0, absorb=0.6, wick=1.50e-06, retain=0.25, soak=4.0,
                      tears_at=1.5),
}

WEAVES = {'none': 0.0, 'plain': 1.0, 'twill': 2.0, 'pile': 3.0, 'ripstop': 4.0}


@dataclass
class FabricSpec:
    """What a fabric is (anything here changing rebuilds it)."""
    shape: str = 'panel'            # 'panel' or 'mesh'
    mesh: str = ''                  # mesh source (resolved), for shape 'mesh'
    width: float = 1.0              # panel (m)
    height: float = 1.0
    fullness: float = 1.0           # panel, hanging: how gathered it is (cloth width over the width it hangs across)
    detail: int = 48                # cells across the panel's longer side
    orientation: str = 'hanging'    # panel: 'hanging' (upright, facing z) or 'lying' (flat)
    pins: str = 'top'               # 'top', 'side', 'top_corners', 'corners', 'edges', 'none'
    pin_rows: int = 1
    material: str = 'cotton'
    stiffness: float = 1.0          # multipliers on the material
    bend: float = 1.0
    weight: float = 1.0
    burnable: bool = True
    wetness: float = 0.0            # how soaked it starts (0 dry .. 1 dripping wet)
    flammability: float = 1.0
    colour: tuple | None = None
    self_collide: bool = True
    tears: bool = False             # it tears where a thread is pulled past its breaking stretch (cloth_tear.wgsl)
    tear_strength: float = 1.0      # times the material's breaking stretch

    def key(self):
        return (self.shape, self.mesh, round(self.width, 6), round(self.height, 6), round(self.fullness, 4),
                int(self.detail), self.orientation,
                self.pins, int(self.pin_rows), self.material, round(self.stiffness, 6), round(self.bend, 6),
                round(self.weight, 6))


@dataclass
class FabricPlace:
    """Where a fabric's pins are at an instant."""
    pos: tuple = (0.0, 0.0, 0.0)
    yaw: float = 0.0                # radians
    scale: tuple = (1.0, 1.0, 1.0)
    released: bool = False


# -- building the cloth ---------------------------------------------------------------------------------

@dataclass
class FabricMesh:
    rest: np.ndarray                # (n, 3) fabric frame (m)
    uv: np.ndarray                  # (n, 2) weave coordinates (m)
    tris: np.ndarray                # (t, 3)
    pinned: np.ndarray              # (n,) bool
    stretch: np.ndarray             # (e, 2)
    stretch_k: np.ndarray           # (e,) N/m
    panel: bool = True


def pleats(width, fullness, cloth_width, n_seg, n_pleats=None):
    """A gathered curtain's shape across: a wave of pleats across `width` (m) whose length along the wave
    is `cloth_width`, with the cloth's n_seg + 1 points evenly along it. Returns their (x, z) (m), x
    from 0 to width, and the number of pleats. Pleats are about eight points long, 8 to 25 cm apart,
    and uneven, as hand-hung cloth is (their spacing and depth wander by a fifth or so)."""
    spacing = cloth_width / n_seg
    n = n_pleats or max(1, int(round(width / float(np.clip(8.0 * spacing / max(fullness, 1.0), 0.08, 0.25)))))
    k = 2.0 * math.pi * n / width
    xs = np.linspace(0.0, width, 96 * n + 1)
    q = xs / width                              # (the unevenness follows the curtain, not the scale)
    phase = k * xs + 0.55 * np.sin(2.0 * math.pi * 1.37 * q + 1.7) + 0.35 * np.sin(2.0 * math.pi * 2.71 * q + 4.1)
    depth = 1.0 + 0.22 * np.sin(2.0 * math.pi * 1.9 * q + 0.4) + 0.12 * np.sin(2.0 * math.pi * 4.3 * q + 2.2)
    shape = depth * np.sin(phase)

    def length(a):
        z = a * shape
        seg = np.hypot(np.diff(xs), np.diff(z))
        return np.concatenate([[0.0], np.cumsum(seg)])

    lo, hi = 0.0, width / n
    for _ in range(60):   # the depth whose wave is the cloth's length
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if length(mid)[-1] < cloth_width else (lo, mid)
    a = 0.5 * (lo + hi)
    s = length(a)
    x = np.interp(np.arange(n_seg + 1) * (s[-1] / n_seg), s, xs)
    return x, a * np.interp(x, xs, shape), n


def panel_mesh(spec: FabricSpec) -> FabricMesh:
    w, h = max(spec.width, 1e-3), max(spec.height, 1e-3)
    full = max(1.0, float(spec.fullness)) if spec.orientation != 'lying' else 1.0
    cw = w * full                     # the cloth's own width (gathered into w)
    d = max(2, int(spec.detail))
    nu = max(2, int(round(d * cw / max(cw, h))))
    nv = max(2, int(round(d * h / max(cw, h))))
    i, j = np.meshgrid(np.arange(nu + 1), np.arange(nv + 1), indexing='xy')
    i, j = i.reshape(-1), j.reshape(-1)
    u = i * (cw / nu)
    v = j * (h / nv)
    if spec.orientation == 'lying':
        rest = np.stack([u - w / 2, np.zeros_like(u), v - h / 2], -1)
    elif full > 1.001:
        # gathered: hung in pleats, as curtain tape holds them (the pins keep the pleats at the top;
        # lower down the folds fall open under their weight)
        # (gathered tightest at the top; the folds open out toward the hem: each row hangs across a
        # little more, its pleats lined up with those above)
        _, _, npl = pleats(w, full, cw, nu)
        rest = np.zeros((len(i), 3))
        for b in range(nv + 1):
            f_row = 1.0 + (full - 1.0) * (0.6 + 0.4 * b / nv)
            w_row = cw / f_row
            px, pz, _ = pleats(w_row, f_row, cw, nu, npl)
            row = j == b
            rest[row] = np.stack([px[i[row]] - w_row / 2, v[row] - h / 2, pz[i[row]]], -1)
    else:
        rest = np.stack([u - w / 2, v - h / 2, np.zeros_like(u)], -1)
    idx = lambda a, b: b * (nu + 1) + a
    tris = []
    for b in range(nv):
        for a in range(nu):
            v00, v10, v01, v11 = idx(a, b), idx(a + 1, b), idx(a, b + 1), idx(a + 1, b + 1)
            if (a + b) % 2 == 0:
                tris += [(v00, v10, v11), (v00, v11, v01)]
            else:
                tris += [(v00, v10, v01), (v10, v11, v01)]
    tris = np.array(tris, np.int64)
    m = MATERIALS.get(spec.material, MATERIALS['cotton'])
    ks, kg = m.stretch * spec.stiffness, m.shear * spec.stiffness
    edges, kk = [], []
    for b in range(nv + 1):
        for a in range(nu):
            edges.append((idx(a, b), idx(a + 1, b)))
            kk.append(ks)
    for b in range(nv):
        for a in range(nu + 1):
            edges.append((idx(a, b), idx(a, b + 1)))
            kk.append(ks)
    for b in range(nv):
        for a in range(nu):
            edges += [(idx(a, b), idx(a + 1, b + 1)), (idx(a + 1, b), idx(a, b + 1))]
            kk += [kg, kg]
    rows = max(1, int(spec.pin_rows))
    top = j >= nv - (rows - 1)
    side = i <= rows - 1
    corner = lambda a, b: (np.abs(i - a) <= rows - 1) & (np.abs(j - b) <= rows - 1)
    border = (i <= rows - 1) | (i >= nu - (rows - 1)) | (j <= rows - 1) | (j >= nv - (rows - 1))
    pinned = {'top': top, 'side': side, 'top_corners': corner(0, nv) | corner(nu, nv),
              'corners': corner(0, 0) | corner(nu, 0) | corner(0, nv) | corner(nu, nv),
              'edges': border}.get(spec.pins, np.zeros(len(i), bool))
    return FabricMesh(rest.astype(np.float64), np.stack([u, v], -1), tris, pinned.astype(bool), np.array(edges, np.int64),
                      np.array(kk, np.float64), True)


def mesh_fabric(spec: FabricSpec) -> FabricMesh:
    from .mesh import load_mesh
    v, t = load_mesh(spec.mesh)
    v = np.asarray(v, np.float64)
    t = np.asarray(t, np.int64)
    # weld duplicate vertices (OBJ exports split them at seams)
    key = np.round(v / max(1e-6, float(np.ptp(v, 0).max()) * 1e-6)).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    v = v[first]
    t = inv.reshape(-1)[t]
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    e = np.unique(e, axis=0)
    m = MATERIALS.get(spec.material, MATERIALS['cotton'])
    # weave coordinates: the mesh's two longest extents
    ext = np.ptp(v, 0)
    ax = np.argsort(-ext)[:2]
    uv = v[:, ax] - v[:, ax].min(0)
    lo, hi = v.min(0), v.max(0)
    el = float(np.mean(np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1))) if len(e) else 0.01
    rows = max(1, int(spec.pin_rows))
    tol = el * (rows - 0.5)
    if spec.pins == 'top':
        pinned = v[:, 1] >= hi[1] - tol
    elif spec.pins == 'side':
        pinned = v[:, 0] <= lo[0] + tol
    elif spec.pins in ('top_corners', 'corners'):
        pts = [(lo[0], hi[1], lo[2]), (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2]), (hi[0], hi[1], lo[2])]
        pinned = np.zeros(len(v), bool)
        for p in pts[:2] if spec.pins == 'top_corners' else pts:
            pinned |= np.linalg.norm(v - np.asarray(p), axis=1) <= np.linalg.norm(v - np.asarray(p), axis=1).min() + tol
    elif spec.pins == 'edges':
        # its open edges (those of one triangle only), and rows - 1 rings in from them
        sides = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
        sides, uses = np.unique(sides, axis=0, return_counts=True)
        pinned = np.zeros(len(v), bool)
        pinned[sides[uses == 1].ravel()] = True
        for _ in range(rows - 1):
            pinned[e[pinned[e[:, 0]] | pinned[e[:, 1]]].ravel()] = True
    else:
        pinned = np.zeros(len(v), bool)
    return FabricMesh(v, uv, t, pinned, e, np.full(len(e), m.stretch * spec.stiffness), False)


def _cot(a, b):
    c = np.linalg.norm(np.cross(a, b), axis=-1)
    return np.einsum('ij,ij->i', a, b) / np.maximum(c, 1e-12)


def hinges(rest, tris):
    """Inner edges with their two opposite vertices: (h, 4) vertex indices (edge first) and the
    isometric bending weights K (h, 4) (Bergou et al. 2006), and the two triangles' areas (h,)."""
    t = np.asarray(tris)
    e = np.concatenate([t[:, [0, 1, 2]], t[:, [1, 2, 0]], t[:, [2, 0, 1]]])   # edge (a, b), opposite c
    key = np.sort(e[:, :2], axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    key, e = key[order], e[order]
    same = np.all(key[1:] == key[:-1], axis=1)
    pairs = np.nonzero(same)[0]
    # an edge shared by more than two triangles (non-manifold) keeps only its first pair
    pairs = pairs[np.concatenate([[True], pairs[1:] != pairs[:-1] + 1])] if len(pairs) else pairs
    a = e[pairs]
    b = e[pairs + 1]
    idx = np.stack([a[:, 0], a[:, 1], a[:, 2], b[:, 2]], -1)
    x0, x1, x2, x3 = (rest[idx[:, k]] for k in range(4))
    e0, e1, e2, e3, e4 = x1 - x0, x2 - x0, x3 - x0, x2 - x1, x3 - x1
    c01, c02, c03, c04 = _cot(e0, e1), _cot(e0, e2), _cot(-e0, e3), _cot(-e0, e4)
    K = np.stack([c03 + c04, c01 + c02, -c01 - c03, -c02 - c04], -1)
    area = 0.5 * (np.linalg.norm(np.cross(e0, e1), axis=1) + np.linalg.norm(np.cross(e0, e2), axis=1))
    return idx, K, area


def colour(cons, n):
    """Greedy graph colouring: constraints sharing a vertex get different colours. (m,) colours."""
    used = [0] * n
    out = np.empty(len(cons), np.int32)
    for ci, vs in enumerate(cons.tolist()):
        mask = 0
        for q in vs:
            mask |= used[q]
        c = (~mask & (mask + 1)).bit_length() - 1
        out[ci] = c
        bit = 1 << c
        for q in vs:
            used[q] |= bit
    return out


def _csr(n, pairs):
    """Per-vertex lists as one u32 array: (first, count) per vertex, then the lists."""
    src, dst = pairs[:, 0], pairs[:, 1]
    order = np.argsort(src, kind='stable')
    src, dst = src[order], dst[order]
    counts = np.bincount(src, minlength=n)
    first = 2 * n + np.concatenate([[0], np.cumsum(counts)[:-1]])
    head = np.stack([first, counts], -1).reshape(-1)
    return np.concatenate([head, dst]).astype(np.uint32)


@dataclass
class Built:
    """All fabrics' cloth, joined, ready for the GPU."""
    n: int
    rest: np.ndarray
    uv: np.ndarray                  # (n, 4): u, v, fabric, _
    inv_mass: np.ndarray            # (n,) negative where pinned
    fabric: np.ndarray              # (n,)
    tris: np.ndarray                # (t, 4): 3 vertices, fabric
    stretch: np.ndarray             # (e, 4) u32
    stretch_ranges: list
    hinge_v: np.ndarray
    hinge_kc: np.ndarray            # (h, 8) f32: K, compliance, rest length, _, _
    hinge_ranges: list
    vt: np.ndarray                  # CSR vertex -> triangles
    nb: np.ndarray                  # CSR vertex -> neighbours
    ranges: list                    # per fabric: (first vertex, count)
    radius: float                   # self-collision radius (m), the largest of the fabrics'
    mean_edge: float
    tether: np.ndarray = None       # (n, 2): nearest pin (global index, -1 none), rest distance along the cloth (m)
    radii: list = None              # self-collision radius per fabric (m): under half its own cell


def tethers(rest, edges, pinned):
    """(nearest pin, distance along the cloth to it) for every vertex: Dijkstra from all the pins at once
    over the cloth's edges (the shortest way along the cloth, never shorter than it can stretch to)."""
    import heapq
    n = len(rest)
    pin = np.full(n, -1, np.int64)
    dist = np.full(n, np.inf)
    if not pinned.any():
        return pin, np.zeros(n)
    L = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    order = np.argsort(np.concatenate([edges[:, 0], edges[:, 1]]), kind='stable')
    src = np.concatenate([edges[:, 0], edges[:, 1]])[order]
    dst = np.concatenate([edges[:, 1], edges[:, 0]])[order]
    ln = np.concatenate([L, L])[order]
    start = np.searchsorted(src, np.arange(n + 1))
    heap = []
    for v in np.nonzero(pinned)[0]:
        dist[v] = 0.0
        pin[v] = v
        heap.append((0.0, int(v)))
    heapq.heapify(heap)
    dst_l, ln_l, start_l = dst.tolist(), ln.tolist(), start.tolist()
    dist_l, pin_l = dist.tolist(), pin.tolist()
    while heap:
        d, v = heapq.heappop(heap)
        if d > dist_l[v]:
            continue
        for k in range(start_l[v], start_l[v + 1]):
            u = dst_l[k]
            nd = d + ln_l[k]
            if nd < dist_l[u]:
                dist_l[u] = nd
                pin_l[u] = pin_l[v]
                heapq.heappush(heap, (nd, u))
    dist = np.asarray(dist_l)
    pin = np.asarray(pin_l, np.int64)
    pin[pinned] = -1                     # pins need no tether
    pin[~np.isfinite(dist)] = -1         # cloth not connected to a pin
    return pin, np.where(np.isfinite(dist), dist, 0.0)


def build(specs):
    rest, uv, inv, fab, tris, st, hv, hk, pr = [], [], [], [], [], [], [], [], []
    teth = []
    ranges = []
    base = 0
    edge_len = []
    for f, spec in enumerate(specs):
        m = MATERIALS.get(spec.material, MATERIALS['cotton'])
        fm = panel_mesh(spec) if spec.shape != 'mesh' else mesh_fabric(spec)
        n = len(fm.rest)
        t = fm.tris
        area_t = 0.5 * np.linalg.norm(np.cross(fm.rest[t[:, 1]] - fm.rest[t[:, 0]], fm.rest[t[:, 2]] - fm.rest[t[:, 0]]), axis=1)
        va = np.bincount(t.reshape(-1), weights=np.repeat(area_t / 3.0, 3), minlength=n)
        mass = np.maximum(va * m.density * spec.weight, 1e-9)
        w = 1.0 / mass
        w[fm.pinned] *= -1.0
        rest.append(fm.rest)
        uv.append(np.concatenate([fm.uv, np.full((n, 1), f), np.zeros((n, 1))], 1))
        inv.append(w)
        fab.append(np.full(n, f))
        tris.append(np.concatenate([t + base, np.full((len(t), 1), f)], 1))
        L = np.linalg.norm(fm.rest[fm.stretch[:, 0]] - fm.rest[fm.stretch[:, 1]], axis=1)
        edge_len.append(L[:max(1, len(L) // 2)])
        st.append((fm.stretch + base, L, 1.0 / np.maximum(fm.stretch_k, 1e-9)))
        hi, K, ha = hinges(fm.rest, t)
        B = m.bend * spec.bend
        comp = ha / max(BEND_K * B, 1e-12)
        v0 = np.linalg.norm(np.einsum('hk,hkd->hd', K, fm.rest[hi]), axis=1)
        v0 = np.where(v0 < 1e-7 * np.maximum(np.sqrt(ha), 1e-9), 0.0, v0) if not fm.panel else np.zeros(len(hi))
        hv.append(hi + base)
        # c.z: a panel's hinge (its bending is solved by multigrid while it is whole: cloth_mg.py)
        hk.append(np.concatenate([K, comp[:, None], v0[:, None], np.full((len(hi), 1), 1.0 if fm.panel else 0.0),
                                  np.zeros((len(hi), 1))], 1))
        # (none for cloth held all round its edges: it does not hang, and across its middle, where the nearest edge
        # changes, tethers to opposite edges would pull it apart as it sags)
        tp, td = tethers(fm.rest, fm.stretch, fm.pinned) if spec.pins not in ('none', 'edges') else (np.full(n, -1), np.zeros(n))
        teth.append(np.stack([np.where(tp >= 0, tp + base, -1), td], -1))
        ranges.append((base, n))
        base += n
    n = base
    rest = np.concatenate(rest)
    tris = np.concatenate(tris).astype(np.uint32)
    se = np.concatenate([s[0] for s in st])
    sl = np.concatenate([s[1] for s in st]).astype(np.float32)
    sc = np.concatenate([s[2] for s in st]).astype(np.float32)
    col = colour(se, n)
    order = np.argsort(col, kind='stable')
    stretch = np.stack([se[order, 0].astype(np.uint32), se[order, 1].astype(np.uint32),
                        sl[order].view(np.uint32), sc[order].view(np.uint32)], -1)
    s_ranges = _ranges(col[order])
    hv = np.concatenate(hv)
    hk = np.concatenate(hk).astype(np.float32)
    hcol = colour(hv, n) if len(hv) else np.zeros(0, np.int32)
    order = np.argsort(hcol, kind='stable')
    hv, hk = hv[order].astype(np.uint32), hk[order]
    h_ranges = _ranges(hcol[order])
    t3 = tris[:, :3].astype(np.int64)
    vt = _csr(n, np.stack([t3.reshape(-1), np.repeat(np.arange(len(t3)), 3)], -1))
    e2 = np.concatenate([se, se[:, ::-1]])
    nb = _csr(n, e2)
    mean_edge = float(np.mean(np.concatenate(edge_len))) if edge_len else 0.02
    return Built(n, rest, np.concatenate(uv).astype(np.float32), np.concatenate(inv), np.concatenate(fab), tris, stretch,
                 s_ranges, hv, hk, h_ranges, vt, nb, ranges, 0.45 * max(float(np.mean(e)) for e in edge_len), mean_edge,
                 np.concatenate(teth), [0.45 * float(np.mean(e)) for e in edge_len])


def _ranges(sorted_cols):
    out = []
    if not len(sorted_cols):
        return out
    edges = np.nonzero(np.diff(sorted_cols))[0] + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(sorted_cols)]])
    return [(int(a), int(b - a)) for a, b in zip(starts, ends)]


def place(rest, p: FabricPlace):
    s = np.asarray(rest) * np.asarray(p.scale)
    c, sn = math.cos(p.yaw), math.sin(p.yaw)
    return np.stack([c * s[:, 0] + sn * s[:, 2], s[:, 1], -sn * s[:, 0] + c * s[:, 2]], -1) + np.asarray(p.pos)


# -- the GPU side ---------------------------------------------------------------------------------------

class Cloth:
    def __init__(self, gpu: GPU):
        self.gpu = gpu
        self.built = None
        self.specs = []
        self._key = None
        self.placed = False
        self.steps = 0
        g = gpu
        vf = g.vel_format
        self.k_predict = g.kernel('cloth_predict.wgsl', ['buf', 'buf', 'buf', 'rbuf', 'rbuf', 'rbuf', 'rbuf', 'tex3d', 'tex3d', 'smp',
                                                         'buf', 'utex3d', 'utex3d', 'rbuf', 'rbuf', 'buf'], workgroup=(64, 1, 1))
        lres = ['rbuf', 'rbuf', 'buf', 'utex3d', 'utex3d', 'st3d:rgba32float:w']
        self.k_lsplat = g.kernel('cloth_liquid.wgsl', lres, 'splat', workgroup=(64, 1, 1))
        self.k_lapply = g.kernel('cloth_liquid.wgsl', lres, 'apply')
        self._lm = None             # the liquid's cells: the momentum the cloth gives back to them (cloth_liquid.wgsl)
        self._liquid_took = False
        self.k_stretch = g.kernel('cloth_stretch.wgsl', ['buf', 'rbuf', 'rbuf', 'rbuf', 'rbuf', 'buf', 'rbuf'], workgroup=(64, 1, 1))
        self.k_bend = g.kernel('cloth_bend.wgsl', ['buf', 'rbuf', 'rbuf', 'buf', 'rbuf', 'rbuf', 'rbuf'], workgroup=(64, 1, 1))
        # water in the cloth: wicking and draining (cloth_wick), dripping and steam (cloth_drip), the drops
        # falling (cloth_drops), the steam into the gas (cloth_steam)
        self.k_wick = g.kernel('cloth_wick.wgsl', ['rbuf'] * 7 + ['buf'], workgroup=(64, 1, 1))
        self.k_drip = g.kernel('cloth_drip.wgsl', ['buf', 'buf', 'buf', 'rbuf', 'rbuf', 'rbuf', 'rbuf', 'buf', 'utex3d'],
                               workgroup=(64, 1, 1))
        self.k_drops = g.kernel('cloth_drops.wgsl', ['buf', 'utex3d', 'utex3d'], workgroup=(64, 1, 1))
        self.k_steam = g.kernel('cloth_steam.wgsl', ['utex3d', 'utex3d', 'rbuf', 'st3d:rgba16float:w', 'st3d:rgba16float:w'],
                                workgroup=(4, 4, 4))
        self._steam_time = 0.0   # s of cloth substeps since the steam was last gathered (splat)
        # the fire's radiation: gathered on a coarse grid, falling on each vertex, heating the cloth
        self.k_rad = g.kernel('cloth_rad.wgsl', ['rbuf'] * 5 + ['buf'], workgroup=(64, 1, 1))
        self.k_radiant = g.kernel('cloth_radiant.wgsl', ['buf', 'buf', 'rbuf', 'rbuf', 'rbuf', 'tex3d', 'smp'], workgroup=(64, 1, 1))
        self.k_tether = g.kernel('cloth_tether.wgsl', ['buf', 'rbuf', 'rbuf', 'rbuf', 'buf'], workgroup=(64, 1, 1))
        self.k_hclear = g.kernel('cloth_hash.wgsl', ['rbuf', 'rbuf', 'buf'], entry='clear', workgroup=(64, 1, 1))
        self.k_hinsert = g.kernel('cloth_hash.wgsl', ['rbuf', 'rbuf', 'buf'], entry='insert', workgroup=(64, 1, 1))
        self.k_self = g.kernel('cloth_self.wgsl', ['rbuf'] * 5 + ['buf', 'rbuf', 'rbuf'], workgroup=(64, 1, 1))
        self.k_collide = g.kernel('cloth_collide.wgsl', ['buf', 'rbuf', 'rbuf', 'rbuf', 'rbuf', 'utex3d', 'tex3d', 'utex3d', 'smp'],
                                  workgroup=(64, 1, 1))
        self.matter_link = None       # the matter this frame (matter.py cloth_link): it drapes over it and is pushed by it
        tear = ['rbuf', 'rbuf', 'buf', 'rbuf', 'rbuf', 'rbuf', 'buf']
        self.k_tear_load = g.kernel('cloth_tear.wgsl', tear, entry='load', workgroup=(64, 1, 1))
        self.k_tear_mark = g.kernel('cloth_tear.wgsl', tear, entry='mark', workgroup=(64, 1, 1))
        self.k_tear = g.kernel('cloth_tear.wgsl', tear, entry='tear', workgroup=(64, 1, 1))
        self.k_finish = g.kernel('cloth_finish.wgsl', ['rbuf', 'buf', 'buf', 'rbuf', 'rbuf', 'buf', 'rbuf', 'rbuf', 'tex3d', 'smp'],
                                 workgroup=(64, 1, 1))
        self.k_normals = g.kernel('cloth_normals.wgsl', ['rbuf', 'rbuf', 'rbuf', 'rbuf', 'buf'], workgroup=(64, 1, 1))
        self.k_sclear = g.kernel('cloth_splat.wgsl', ['rbuf', 'rbuf', 'rbuf', 'rbuf', 'rbuf', 'buf', 'buf'], entry='clear',
                                 workgroup=(64, 1, 1))
        self.k_splat = g.kernel('cloth_splat.wgsl', ['rbuf', 'rbuf', 'rbuf', 'rbuf', 'rbuf', 'buf', 'buf'], entry='splat',
                                workgroup=(64, 1, 1))
        self.k_air = g.kernel('cloth_air.wgsl', ['utex3d', 'rbuf', f'st3d:{vf}:w'], defines={'VELFMT': vf}, workgroup=(4, 4, 4))
        self.k_feed = g.kernel('cloth_feed.wgsl', ['utex3d', 'rbuf', 'st3d:rgba16float:w'], workgroup=(4, 4, 4))
        self.raster = None    # the raster layer it draws into (raster.py; the engine's, shared with the grass)
        self.k_mgwet = g.kernel('cloth_mg_wet.wgsl', ['buf'] + ['rbuf'] * 4, workgroup=(64, 1, 1))
        self._may_be_wet = False      # the cloth may hold water (it has been in a liquid, or starts wet)
        self.k_oclear = g.kernel('cloth_occ.wgsl', ['rbuf'] * 4 + ['buf'], entry='clear', workgroup=(64, 1, 1))
        self.k_osplat = g.kernel('cloth_occ.wgsl', ['rbuf'] * 4 + ['buf'], entry='splat', workgroup=(64, 1, 1))
        self.OCC = None
        self._occ_cells = 0
        self._pipe = None
        self.couple_dims = None
        self.G = None
        self._view = None      # arrays of a cached frame being drawn (else the live cloth)
        self.bufs = {}

    # -- building -------------------------------------------------------------------------------------

    @property
    def active(self):
        return self.built is not None and self.built.n > 0

    def configure(self, specs):
        """Match the cloth to the scene's fabrics; True if it was rebuilt (and must be placed again)."""
        specs = list(specs)[:MAX_FABRICS]
        key = tuple(s.key() for s in specs)
        self.specs = specs
        if key == self._key:
            self._write_materials()
            return False
        self._key = key
        self._progs = {}
        for b in self.bufs.values():
            b.destroy()
        self.bufs = {}
        self.built = None
        self.placed = False
        if not specs:
            return True
        try:
            self.built = build(specs)
        except Exception as ex:
            log.warning('Cannot build the fabric: %s', ex)
            self.built = None
            return True
        B = self.built
        g = self.gpu
        n = B.n
        mk = lambda name, arr: self._upload(name, np.ascontiguousarray(arr))
        for name in ('X', 'P', 'V', 'S', 'S2', 'N', 'D', 'IMP', 'MP', 'LIMP'):
            self.bufs[name] = g.buffer(max(16, n * 16), f'cloth-{name}')
        self.bufs['TR'] = g.buffer(max(16, n * 16), 'cloth-tear')
        self.bufs['OP'] = g.buffer(max(16, n * 16), 'cloth-objects-push')   # (cloth_predict.wgsl: object_push)
        mk('R', np.concatenate([B.rest, B.inv_mass[:, None]], 1).astype(np.float32))
        mk('UV', B.uv)
        mk('T', B.tris)
        mk('C', B.stretch)
        mk('H', np.concatenate([B.hinge_v.view(np.float32).reshape(-1, 4), B.hinge_kc], 1).astype(np.float32)
           if len(B.hinge_v) else np.zeros((1, 12), np.float32))
        mk('VT', B.vt)
        mk('NB', B.nb)
        self.hash_size = 1 << max(12, int(math.ceil(math.log2(max(2 * n, HASH_MIN)))))
        te = np.zeros((n, 4), np.float32)
        te[:, 0] = B.tether[:, 0].astype(np.int64).astype(np.uint32).view(np.float32)
        te[:, 1] = B.tether[:, 1]
        mk('TE', te)
        self.bufs['HOLED'] = g.buffer(MAX_FABRICS * 4, 'cloth-holed')
        self.bufs['XB'] = g.buffer(max(16, n * 16), 'cloth-XB')
        self.bufs['W'] = g.buffer(max(32, n * 32), 'cloth-water')          # cloth_wick.wgsl
        self.bufs['DR'] = g.buffer(16 + DROPS * 32, 'cloth-drops')         # cloth_drip.wgsl
        self.bufs['QR'] = g.buffer(max(16, n * 16), 'cloth-radiation')     # cloth_rad.wgsl
        self._build_mg(specs)
        self.bufs['LS'] = g.buffer(max(16, len(B.stretch) * 4), 'cloth-lambda-stretch')
        self.bufs['LB'] = g.buffer(max(16, len(B.hinge_v) * 16), 'cloth-lambda-bend')
        self.bufs['HN'] = g.buffer((self.hash_size + n) * 4, 'cloth-hash')
        self.bufs['M'] = g.buffer(MAX_FABRICS * 112, 'cloth-materials')
        self.bufs['DM'] = g.buffer(MAX_FABRICS * 64, 'cloth-draw-materials')
        self._write_materials()
        log.info('Fabric: %d vertices, %d triangles, %d stretch colours, %d bending colours', n, len(B.tris),
                 len(B.stretch_ranges), len(B.hinge_ranges))
        return True

    def _build_mg(self, specs):
        """The multigrid hierarchy for the panels (cloth_mg.py), or None."""
        from . import cloth_mg
        if getattr(self, 'mg', None) is not None:
            self.mg.destroy()
        self.mg = None
        self.mg_mask = 0
        B = self.built
        panels = [f for f, s in enumerate(specs) if s.shape != 'mesh']
        if not MULTIGRID or not panels or not len(B.hinge_v):
            return
        hv = B.hinge_v.astype(np.int64)
        kc = B.hinge_kc
        inputs = []
        for f in panels:
            a, n = B.ranges[f]
            ids = np.arange(a, a + n)
            sel = (hv[:, 0] >= a) & (hv[:, 0] < a + n)
            inputs.append((ids, B.uv[ids, :2].astype(np.float64), 1.0 / np.maximum(np.abs(B.inv_mass[ids]), 1e-30),
                           B.inv_mass[ids] < 0, hv[sel], kc[sel, :4].astype(np.float64), 1.0 / np.maximum(kc[sel, 4], 1e-30)))
        try:
            hier = cloth_mg.build(inputs, B.n)
            if hier is None or len(hier.levels) < 2:
                return
            self.mg = cloth_mg.Runner(self.gpu, hier, hv, sum(1 << f for f in panels))
            self.mg_mask = self.mg.mask
        except Exception as ex:
            log.warning('Fabric multigrid unavailable (%s): bending by constraint passes', ex)
            self.mg = None
            self.mg_mask = 0

    def _upload(self, name, arr):
        buf = self.gpu.buffer(max(16, arr.nbytes), f'cloth-{name}')
        self.gpu.write_buffer(buf, arr)
        self.bufs[name] = buf
        return buf

    def _write_materials(self):
        if not self.active:
            return
        mat = np.zeros((MAX_FABRICS, 28), np.float32)
        dm = np.zeros((MAX_FABRICS, 16), np.float32)
        for f, s in enumerate(self.specs):
            m = MATERIALS.get(s.material, MATERIALS['cotton'])
            rho = m.density * s.weight
            tau = rho * C_P / (2.0 * H_CONV)
            mat[f] = (rho, m.damping, m.friction, m.thickness,
                      m.cn, m.ct, m.ignition, m.burn_time,
                      m.burn_temp, m.fuel, m.smoke, tau,
                      m.soften, m.melt, m.shrink, 1.0 if s.burnable else 0.0,
                      m.spread, m.char_shrink, max(s.flammability, 0.0),
                      m.tears_at * max(s.tear_strength, 0.01) if s.tears else 0.0,
                      m.fibre, m.absorb, m.soak, 0.0,
                      m.wick, m.retain, 0.0, 0.0)
            col = s.colour if s.colour is not None else m.colour
            dm[f] = (*col, m.sheen, m.sheen_rough, m.spec, m.spec_rough[0], m.spec_rough[1],
                     m.translucency, m.pitch, WEAVES.get(m.weave, 1.0), m.weave_depth,
                     m.burn_temp, 1.0 if m.melt > 0 else 0.0, m.ignition / max(s.flammability, 0.05), m.retain)
        self.gpu.write_buffer(self.bufs['M'], mat)
        self.gpu.write_buffer(self.bufs['DM'], dm)

    def place(self, places, ambient_k=293.0):
        """Put every fabric at its placement, at rest, cool and unburnt."""
        if not self.active:
            return
        B = self.built
        x = np.zeros((B.n, 4), np.float32)
        for (a, n), p in zip(B.ranges, places):
            x[a:a + n, :3] = place(B.rest[a:a + n], p)
        x[:, 3] = np.abs(B.inv_mass)
        g = self.gpu
        g.write_buffer(self.bufs['X'], x)
        p = x.copy()
        # dry, or as wet as each fabric starts
        p[:, 3] = np.array([float(np.clip(getattr(sp, 'wetness', 0.0), 0.0, 1.0)) for sp in self.specs] + [0.0],
                           np.float32)[np.minimum(B.fabric.astype(np.int64), len(self.specs))]
        g.write_buffer(self.bufs['P'], p)
        self._may_be_wet = False
        self._reset_water(p[:, 3])
        v = np.zeros((B.n, 4), np.float32)
        v[:, 3] = B.fabric
        g.write_buffer(self.bufs['V'], v)
        s = np.zeros((B.n, 4), np.float32)
        s[:, 0] = ambient_k
        g.write_buffer(self.bufs['S'], s)
        g.write_buffer(self.bufs['S2'], s)
        nrm = np.zeros((B.n, 4), np.float32)
        nrm[:, 2] = 1.0
        g.write_buffer(self.bufs['N'], nrm)
        g.write_buffer(self.bufs['D'], np.zeros((B.n, 4), np.float32))
        g.write_buffer(self.bufs['IMP'], np.zeros((B.n, 4), np.float32))
        g.write_buffer(self.bufs['LIMP'], np.zeros((B.n, 4), np.float32))
        g.write_buffer(self.bufs['MP'], np.zeros((B.n, 4), np.float32))
        g.write_buffer(self.bufs['OP'], np.zeros((B.n, 4), np.float32))
        self._op_on = False
        g.write_buffer(self.bufs['HOLED'], np.zeros(MAX_FABRICS, np.uint32))
        with g.batch() as b:
            self._normals(b)
        # (the coupling grid still holds the cloth of a run before, perhaps another scene's: the gas takes nothing from
        # it until splat() has filled it again)
        self._couple_filled = False
        self.placed = True
        self.steps = 0

    def reset(self):
        self.placed = False

    # -- stepping -------------------------------------------------------------------------------------

    def _fab_uniforms(self, u, places):
        for f in range(MAX_FABRICS):
            if f < len(places):
                p = places[f]
                u.v4(*p.pos, p.yaw).v4(*p.scale, 1.0 if p.released else 0.0)
            else:
                u.v4().v4(1.0, 1.0, 1.0, 0.0)
        return u

    def _normals(self, b):
        B = self.built
        k = self.bufs
        b.run(self.k_normals, [k['X'], k['S'], k['T'], k['VT'], k['N']], Uniforms().v4(B.n), (B.n, 1, 1))

    def step(self, b, solver, dt, places, prm, look, colliders, meshes, steps=1, liquid=None):
        """Record `steps` cloth substeps covering dt seconds, in the air of `solver` (a Solver or None) and the
        liquid of `liquid` (a LiquidSolver or None: its drag and buoyancy, and it soaks the cloth)."""
        if not self.active or not self.placed:
            return
        from .solver import pack_colliders
        B = self.built
        k = self.bufs
        n = B.n
        h = dt / max(1, steps)
        gas = solver is not None and solver.dims is not None
        vel = solver.vel[0] if gas else None
        scal = solver.scal[0] if gas else None
        if not gas:
            vel = scal = self._dummy()
        grid = solver._grid(h, prm) if gas else Uniforms().v4(1, 1, 1, 1).v4().v4()
        lin = self.gpu.linear
        wet = liquid is not None and liquid.dims is not None and liquid.vel_tex is not None
        if wet:
            lvel, ltype = liquid.vel_tex, liquid.TYPE[0]
            lq = Uniforms().v4(*liquid.dims, liquid.h).v4(*liquid.origin, 1.0).v4(float(getattr(liquid._prm, 'rho', 1000.0)))
        else:
            lvel = ltype = self._dummy_u()
            lq = Uniforms().v4(1, 1, 1, 1).v4().v4(1000.0)
        self._may_be_wet = self._may_be_wet or wet or any(getattr(s, 'wetness', 0.0) > 0.0 for s in self.specs)
        wind = tuple(prm.wind) if prm is not None else (0.0, 0.0, 0.0)
        amb = float(look.ambient_k)
        flame = float(look.flame_k)
        maxk = float(max(look.max_k, look.flame_k + 1.0))
        self_on = any(s.self_collide for s in self.specs)
        # a triangle within a radius of a vertex has its corners within a radius and an edge of it
        cell = 3.5 * B.radius
        rad = np.zeros(16, np.float32)
        rad[:len(B.radii)] = B.radii
        tears = any(s.tears for s in self.specs)
        radiant = RADIANT and gas and self._radiation(b, solver, amb, flame, maxk)
        for _ in range(max(1, steps)):
            u = Uniforms().v4(h, n, 9.81, 1.0 if gas else 0.0)
            u.raw(grid.data)
            u.v4(*wind, amb).v4(flame, maxk)
            self._fab_uniforms(u, places)
            u.raw(lq.data)
            self._liquid_took = self._liquid_took or liquid is not None
            b.run(self.k_predict, [k['X'], k['P'], k['V'], k['R'], k['S'], k['N'], k['M'], vel, scal, lin, k['IMP'],
                                   lvel, ltype, k['MP'], k['OP'], k['LIMP']], u, (n, 1, 1))
            mg_on = self.mg is not None
            if mg_on:
                # the panels' bending, solved (cloth_mg.py; its buffer swaps come back round within a cycle)
                b.run_program(self._program(('mg', h, id(k['S'])), lambda rec: self.mg.cycle(rec, h, dict(k))))
                if self._may_be_wet:        # (wet cloth is heavier: its bending moves it less)
                    b.run(self.k_mgwet, [k['X'], self.mg.lv[0]['XT'], k['P'], k['V'], k['M']],
                          Uniforms().v4(n).v4(self.mg_mask), (n, 1, 1))
            b.clear_buffer(k['LS'])
            b.clear_buffer(k['LB'])
            # with multigrid for every fabric, the bending passes only see charring cloth: one is enough
            bend_passes = 1 if (mg_on and all(s.shape != 'mesh' for s in self.specs)) else ITERATIONS
            mask = self.mg_mask if mg_on else 0

            def passes(rec):
                for it in range(ITERATIONS):
                    for off, cnt in B.stretch_ranges:
                        rec.run(self.k_stretch, [k['X'], k['C'], k['S'], k['V'], k['M'], k['LS'], k['P']],
                                Uniforms().v4(off, cnt, h, COMPRESSION_SOFT).v4(OVER_RELAX), (cnt, 1, 1))
                    for off, cnt in (B.hinge_ranges if it < bend_passes else ()):
                        rec.run(self.k_bend, [k['X'], k['H'], k['S'], k['LB'], k['V'], k['P'], k['M']],
                                Uniforms().v4(off, cnt, h, OVER_RELAX).v4(mask), (cnt, 1, 1))
            b.run_program(self._program(('passes', h, id(k['S']), id(k['X']), bend_passes, mask), passes))
            # hanging cloth never stretches far from its pins (long-range attachments)
            tu = Uniforms().v4(n, TETHER_SLACK)
            self._fab_uniforms(tu, places)
            b.run(self.k_tether, [k['X'], k['TE'], k['S'], k['V'], k['HOLED']], tu, (n, 1, 1))
            do_self = self_on and self.steps % 2 == 0
            if do_self:
                hu = Uniforms().v4(n, cell, self.hash_size, B.radius)
                b.run(self.k_hclear, [k['X'], k['S'], k['HN']], hu, (self.hash_size, 1, 1))
                b.run(self.k_hinsert, [k['X'], k['S'], k['HN']], hu, (n, 1, 1))
                su = Uniforms().v4(n, cell, self.hash_size, B.radius).raw(rad.tolist())
                b.run(self.k_self, [k['X'], k['P'], k['S'], k['UV'], k['HN'], k['D'], k['T'], k['VT']], su, (n, 1, 1))
            cu = Uniforms().v4(h, n, 1.0 if (prm is None or prm.ground) else 0.0, 1.0 if do_self else 0.0)
            pack_colliders(cu, colliders, meshes)
            atlas = meshes.atlas if meshes is not None else self._dummy()
            ml = self.matter_link
            if ml is not None:
                cu.v4(*ml[2], ml[3]).v4(*ml[4], 1.0)
                msurf, mvel = ml[0], ml[1]
            else:
                cu.v4().v4(1.0, 1.0, 1.0, 0.0)
                msurf, mvel = self._dummy(), self._dummy_u()
            b.run(self.k_collide, [k['X'], k['P'], k['D'], k['V'], k['M'], atlas, msurf, mvel, lin], cu, (n, 1, 1))
            if tears:
                # threads pulled past their breaking stretch tear (cloth_tear.wgsl)
                b.clear_buffer(k['TR'])
                ne = len(B.stretch)
                tu = Uniforms().v4(ne, n)
                res = [k['X'], k['C'], k['S'], k['V'], k['M'], k['UV'], k['TR']]
                b.run(self.k_tear_load, res, tu, (ne, 1, 1))
                b.run(self.k_tear_mark, res, tu, (ne, 1, 1))
                b.run(self.k_tear, res, tu, (n, 1, 1))
            fu = Uniforms().v4(h, n, 1.0 if gas else 0.0)
            fu.raw(grid.data)
            fu.v4(amb, flame, maxk)
            b.run(self.k_finish, [k['X'], k['P'], k['V'], k['R'], k['S'], k['S2'], k['M'], k['NB'], scal, lin], fu, (n, 1, 1))
            k['S'], k['S2'] = k['S2'], k['S']
            if self._may_be_wet:
                self._water(b, h, prm, colliders, meshes, liquid if wet else None, ltype)
            if radiant:
                ru = Uniforms().v4(h, n, 1.0)
                ru.raw(grid.data)
                ru.v4(amb, flame, maxk)
                b.run(self.k_radiant, [k['S'], k['P'], k['QR'], k['M'], k['V'], scal, lin], ru, (n, 1, 1))
            if self.steps % 2 == 0:
                self._normals(b)
            self.steps += 1

    # -- water in the cloth ------------------------------------------------------------------------------

    def _reset_water(self, wet):
        """The water buffers to match the cloth's wetness `wet` (n,): nothing waiting to drip, no drops."""
        n = self.built.n
        w = np.zeros((2 * n, 4), np.float32)
        w[0::2, 0] = wet[:n]
        w[0::2, 3] = wet[:n]
        self.gpu.write_buffer(self.bufs['W'], w)
        self.gpu.write_buffer(self.bufs['DR'], self._drops_array(None))
        self._steam_time = 0.0

    @staticmethod
    def _drops_array(live):
        """The drops buffer's contents (header, then 2 vec4 per drop) holding the drops `live` ((m, 8) or None)."""
        a = np.zeros(4 + DROPS * 8, np.float32)
        if live is not None and len(live):
            live = np.asarray(live, np.float32)[:DROPS]
            a[4:4 + 8 * len(live)] = live.reshape(-1)
            a[:4] = np.array([len(live), 0, 0, 0], np.uint32).view(np.float32)
        return a

    def _live_drops(self):
        """(m, 8) float32: the drops in flight now (position, life left; velocity, radius)."""
        d = np.frombuffer(self.gpu.read_buffer(self.bufs['DR']), np.float32)[4:4 + DROPS * 8].reshape(-1, 8)
        return d[d[:, 3] > 0.0].copy()

    def _water(self, b, h, prm, colliders, meshes, liquid, ltype):
        """Record one substep of the water in the cloth: wicking and draining, dripping, the steam boiled off
        it kept for the gas, and the drops falling."""
        from .solver import pack_colliders
        k = self.bufs
        n = self.built.n
        ground = 1.0 if (prm is None or prm.ground) else 0.0
        b.run(self.k_wick, [k['P'], k['X'], k['R'], k['NB'], k['M'], k['V'], k['S'], k['W']],
              Uniforms().v4(h, n, ground, 0.0), (n, 1, 1))
        if liquid is not None:
            lq = Uniforms().v4(*liquid.dims, liquid.h).v4(*liquid.origin, 1.0)
        else:
            lq = Uniforms().v4(1, 1, 1, 1).v4(0.0, 0.0, 0.0, 0.0)
        u = Uniforms().v4(h, n, DROPS, self.steps % 1000003).raw(lq.data)
        b.run(self.k_drip, [k['P'], k['W'], k['IMP'], k['N'], k['M'], k['V'], k['X'], k['DR'], ltype], u, (n, 1, 1))
        wind = tuple(prm.wind) if prm is not None else (0.0, 0.0, 0.0)
        du = Uniforms().v4(h, DROPS, ground, 9.81).v4(*wind).raw(lq.data)
        pack_colliders(du, colliders, meshes)
        atlas = meshes.atlas if meshes is not None else self._dummy()
        b.run(self.k_drops, [k['DR'], atlas, ltype], du, (DROPS, 1, 1))
        self._steam_time += h

    radiant = None      # the engine's shared radiant sources (radiant.Radiant, built each frame), set by it

    def liquid_hook(self, b, liquid, dt):
        """LiquidSolver.force_hook, after its forces each substep: the momentum the water's drag gave the cloth since the
        last one, the water loses (cloth_liquid.wgsl), so cloth moving through water pushes it and cloth held in a flow
        slows it."""
        if not self.active or self.built is None or not self._liquid_took or liquid.dims is None:
            return
        g = self.gpu
        cells = int(np.prod(liquid.dims))
        if self._lm is None or self._lm[0] != cells:
            if self._lm is not None:
                self._lm[1].destroy()
            self._lm = (cells, g.buffer(cells * 3 * 4, 'cloth-liquid-momentum'))
        lm = self._lm[1]
        k = self.bufs
        prm = liquid._prm
        u = liquid._grid(dt, prm).v4(self.built.n, float(prm.rho), float(prm.ppc))
        res = [k['X'], k['LIMP'], lm, liquid.VA, liquid.DENS, liquid.VB]
        b.clear_buffer(lm)
        b.run(self.k_lsplat, res, u, (self.built.n, 1, 1))
        b.run(self.k_lapply, res, u, tuple(int(d) + 1 for d in liquid.dims))
        b.copy_texture(liquid.VB, liquid.VA)
        b.clear_buffer(k['LIMP'])
        self._liquid_took = False

    def _radiation(self, b, solver, amb, flame, maxk):
        """Record the radiation falling on the cloth (W/m^2 per vertex, into QR) from the frame's shared radiant sources
        (radiant.py: the fire's gas, glowing matter, lava, hot objects); without the engine's, from the gas as it is now.
        True if recorded."""
        r = self.radiant
        if r is None or not r.built:
            from .radiant import Radiant
            if getattr(self, '_own_radiant', None) is None:
                self._own_radiant = Radiant(self.gpu)
            r = self._own_radiant
            r.layout(solver.origin, solver.dims, solver.h, gas=True)
            r.build(b, solver.scal[0], solver.dims, amb)
        k = self.bufs
        b.run(self.k_rad, [k['X'], k['N'], k['S'], r.RL, r.RLC, k['QR']], Uniforms().v4(self.built.n), (self.built.n, 1, 1))
        return True

    def _steam(self, b, solver, dt, fine):
        """Record the steam boiled off wet cloth going into the gas (cloth_steam.wgsl): on the simulation
        grid (the temperature, and the vapour when the gas carries it) or the finer upres grid."""
        from .solver import LATENT_K_PER_G
        from .renderer import vapour_saturation
        prm = getattr(solver, '_prm', None)
        if prm is None:
            return
        cd = self.couple_dims
        span = max(prm.flame_k - prm.ambient_k, 1.0)
        q_air = min(max(prm.humidity, 0.0), 100.0) / 100.0 * vapour_saturation(prm.ambient_k)
        if fine:
            dims, hh, src = solver.dims_fine, solver.h / solver.upres, solver.scal_fine
        else:
            dims, hh, src = solver.dims, solver.h, solver.scal
        water = not fine and solver.aux is not None
        u = (Uniforms().v4(*dims, hh).v4(*cd, 2.0 * solver.h).v4(dt, prm.boil_temp, q_air, LATENT_K_PER_G / span)
             .v4(1.0 if water else 0.0))
        if water:
            b.run(self.k_steam, [src[0], solver.aux[0], self.G, src[1], solver.aux[1]], u, dims)
            solver.aux.reverse()
        else:
            b.run(self.k_steam, [src[0], src[0], self.G, src[1], self._dummy()], u, dims)
        src.reverse()

    class _Rec:
        """Stands in for a Batch to capture a fixed dispatch sequence into a Program."""

        def __init__(self):
            self.ops = []

        def run(self, kernel, resources, uniforms, size=None, groups=None):
            op = (kernel, list(resources), uniforms, size)
            self.ops.append(op if groups is None else op + (groups,))

    def _program(self, key, record):
        """The dispatches `record(recorder)` makes, bound once and replayed (gpu.Program): the parts of a
        substep that are the same every substep, keyed by what they depend on. (Dropped programs are
        released, not destroyed: one may be in a batch not yet submitted.)"""
        progs = self.__dict__.setdefault('_progs', {})
        p = progs.get(key)
        if p is None:
            if len(progs) >= 12:
                progs.clear()
            rec = Cloth._Rec()
            record(rec)
            p = progs[key] = Program(self.gpu, rec.ops)
        return p

    def _dummy_u(self):
        t = getattr(self, '_dummy_utex', None)
        if t is None:
            t = self._dummy_utex = self.gpu.texture3d((1, 1, 1), 'r32float', 'cloth-dummy-r32')
            self.gpu.upload(t, np.zeros((1, 1, 1, 1), np.float32))
        return t

    def _dummy(self):
        t = getattr(self, '_dummy_tex', None)
        if t is None:
            t = self._dummy_tex = self.gpu.texture3d((1, 1, 1), 'rgba16float', 'cloth-dummy')
            self.gpu.upload(t, np.zeros((1, 1, 1, 4), np.float16))
        return t

    # -- coupling with the gas --------------------------------------------------------------------------

    def prepare_frame(self, solver):
        """Size the coupling grid for the solver's grid (which may have grown). Called before a frame's
        commands are recorded: a buffer replaced while they are being recorded would be gone at submit."""
        if solver is not None:
            # steam off wet cloth smothers flames (react.wgsl), while there may be any
            solver.steam_smothers = bool(self.active and self.placed and self._may_be_wet)
        if not self.active or solver is None or solver.dims is None:
            return
        cd = tuple(int(math.ceil(d / 2)) for d in solver.dims)
        if cd != self.couple_dims or self.G is None:
            if self.G is not None:
                self.G.destroy()
            self.G = self.gpu.buffer(int(np.prod(cd)) * COUPLE_CHANNELS * 4, 'cloth-couple')
            self.couple_dims = cd
            self._couple_filled = False

    def _couple_ok(self, solver):
        return (self.G is not None and solver is not None and solver.dims is not None
                and self.couple_dims == tuple(int(math.ceil(d / 2)) for d in solver.dims))

    def splat(self, b, solver, look):
        """Spread the cloth (and what it burns) onto the coupling grid, for the next solver substep."""
        if not self.active or not self.placed or not self._couple_ok(solver):
            return
        cd = self.couple_dims
        k = self.bufs
        cells = int(np.prod(cd))
        u = (Uniforms().v4(self.built.n, cells, FUEL_PER_KG, 1.0 / max(self._steam_time, 1e-6)).v4(*cd, 2.0 * solver.h)
             .v4(*solver.origin, look.ambient_k).v4(look.flame_k))
        self._steam_time = 0.0
        res = [k['X'], k['V'], k['N'], k['S'], k['M'], self.G, k['IMP']]
        b.run(self.k_sclear, res, u, groups=groups_1d(cells * COUPLE_CHANNELS))
        b.run(self.k_splat, res, u, (self.built.n, 1, 1))
        self._couple_filled = True

    def prepare_light(self, light_dims):
        """Size the cloth's light-grid occupancy for the renderer's light volume (before recording)."""
        cells = int(np.prod(light_dims))
        if self.OCC is None or cells > self._occ_cells:
            if self.OCC is not None:
                self.OCC.destroy()
            self.OCC = self.gpu.buffer(cells * 4, 'cloth-occlusion')
            self._occ_cells = cells

    def occlusion(self, b, light_dims, corner, cell):
        """Renderer.light occluder: how much light the cloth (the live cloth, or the cached frame's) stops
        in each light cell, its area there times -ln(1 - opacity) (thin cloth lets some through)."""
        if not self.active or self.OCC is None or int(np.prod(light_dims)) > self._occ_cells:
            return None
        src = self._src()
        kap = np.zeros(MAX_FABRICS, np.float32)
        for f, s in enumerate(self.specs[:MAX_FABRICS]):
            m = MATERIALS.get(s.material, MATERIALS['cotton'])
            kap[f] = -math.log(1.0 - min(max(1.0 - 0.8 * m.translucency, 0.2), 0.98))
        cells = int(np.prod(light_dims))
        u = Uniforms().v4(*light_dims, self.built.n).v4(*corner, cells).v4(*cell, 0.0)
        for q in kap.reshape(4, 4):
            u.v4(*q)
        res = [src['X'], src['N'], src['S'], self.bufs['UV'], self.OCC]
        b.run(self.k_oclear, res, u, groups=groups_1d(cells))
        b.run(self.k_osplat, res, u, groups=groups_1d(self.built.n))
        return self.OCC

    def hook(self, b, solver, dt, stage):
        """Solver.cloth_hook: the burning cloth feeds the gas, and the cloth holds the air back."""
        if not self.active or not self.placed or not getattr(self, '_couple_filled', False) or not self._couple_ok(solver):
            return
        cd = self.couple_dims
        hc = 2.0 * solver.h
        if stage == 'sources':
            u = Uniforms().v4(*solver.dims, solver.h).v4(*cd, hc).v4(dt, COUPLE_REF_FUEL)
            b.run(self.k_feed, [solver.scal[0], self.G, solver.scal[1]], u, solver.dims)
            solver.scal.reverse()
            if self._may_be_wet:
                self._steam(b, solver, dt, fine=False)
        elif stage == 'sources_fine' and solver.scal_fine is not None:
            hf = solver.h / solver.upres
            u = Uniforms().v4(*solver.dims_fine, hf).v4(*cd, hc).v4(dt, COUPLE_REF_FUEL)
            b.run(self.k_feed, [solver.scal_fine[0], self.G, solver.scal_fine[1]], u, solver.dims_fine)
            solver.scal_fine.reverse()
            if self._may_be_wet:
                self._steam(b, solver, dt, fine=True)
        elif stage == 'velocity':
            vd = tuple(d + 1 for d in solver.dims)
            u = Uniforms().v4(*solver.dims, solver.h).v4(*cd, hc)
            b.run(self.k_air, [solver.vel[0], self.G, solver.vel[1]], u, vd)
            solver.vel.reverse()

    # -- state ---------------------------------------------------------------------------------------

    def snapshot(self):
        """What a cached frame needs to draw the cloth again: positions, velocities, normals and state."""
        if not self.active or not self.placed:
            return None
        rd = lambda name: np.frombuffer(self.gpu.read_buffer(self.bufs[name]), np.float32).reshape(-1, 4)[:self.built.n]
        out = {'X': rd('X').copy(), 'V': rd('V').astype(np.float16), 'N': rd('N').astype(np.float16),
               'S': rd('S').astype(np.float32), 'W': rd('P')[:, 3].astype(np.float16)}
        if self._may_be_wet:
            out['DR'] = self._live_drops()
        return out

    def save_state(self):
        if not self.active or not self.placed:
            return None
        rd = lambda name: np.frombuffer(self.gpu.read_buffer(self.bufs[name]), np.float32).reshape(-1, 4)[:self.built.n].copy()
        w = np.frombuffer(self.gpu.read_buffer(self.bufs['W']), np.float32).reshape(-1, 4)[:2 * self.built.n].copy()
        return {'key': repr(self._key), 'X': rd('X'), 'P': rd('P'), 'V': rd('V'), 'S': rd('S'), 'N': rd('N'), 'W': w,
                'MP': rd('MP'), 'steps': self.steps}

    def load_state(self, st):
        if not st or not self.active or st.get('key') != repr(self._key):
            return False
        for name in ('X', 'P', 'V', 'S', 'N'):
            self.gpu.write_buffer(self.bufs[name], np.ascontiguousarray(st[name], np.float32))
        self.gpu.write_buffer(self.bufs['S2'], np.ascontiguousarray(st['S'], np.float32))
        self.steps = int(st.get('steps', 0))
        self._may_be_wet = bool((np.asarray(st['P'])[:, 3] > 0.0).any())
        self._reset_water(np.asarray(st['P'], np.float32)[:, 3])
        if st.get('W') is not None and len(st['W']) == 2 * self.built.n:
            self.gpu.write_buffer(self.bufs['W'], np.ascontiguousarray(st['W'], np.float32))
        mp = st.get('MP')
        self.gpu.write_buffer(self.bufs['MP'], np.ascontiguousarray(mp if mp is not None else np.zeros((self.built.n, 4)),
                                                                    np.float32))
        self.placed = True
        return True

    def velocities(self):
        """(n, 3) velocity of every vertex now (m/s)."""
        return np.frombuffer(self.gpu.read_buffer(self.bufs['V']), np.float32).reshape(-1, 4)[:self.built.n, :3].copy()

    def object_push(self, force):
        """The push of the things that fall on each vertex through this frame ((n, 3) N, from solids.py; None: none), and
        its weight's worth of mass, riding on it (cloth_predict.wgsl OP)."""
        if not self.active or not self.placed:
            return
        if force is None:
            if self._op_on:
                self.gpu.write_buffer(self.bufs['OP'], np.zeros((self.built.n, 4), np.float32))
                self._op_on = False
            return
        f = np.asarray(force, np.float32).reshape(-1, 3)[:self.built.n]
        op = np.zeros((self.built.n, 4), np.float32)
        op[:len(f), :3] = f
        op[:len(f), 3] = np.linalg.norm(f, axis=1) / 9.81
        self.gpu.write_buffer(self.bufs['OP'], op)
        self._op_on = True

    def forget_matter(self):
        """No more push from sand, snow or mud (cloth_matter.wgsl gather): the matter it lay under has gone."""
        if self.active and self.placed and 'MP' in self.bufs:
            self.gpu.write_buffer(self.bufs['MP'], np.zeros((self.built.n, 4), np.float32))

    def positions(self):
        """(n, 3) positions and (n,) burnt-away flags of every vertex now (fire-local m)."""
        x = np.frombuffer(self.gpu.read_buffer(self.bufs['X']), np.float32).reshape(-1, 4)[:self.built.n]
        s = np.frombuffer(self.gpu.read_buffer(self.bufs['S']), np.float32).reshape(-1, 4)[:self.built.n]
        return x[:, :3].copy(), s[:, 1] >= 1.0

    def state(self):
        return np.frombuffer(self.gpu.read_buffer(self.bufs['S']), np.float32).reshape(-1, 4)[:self.built.n].copy()

    # -- drawing -------------------------------------------------------------------------------------

    def use_view(self, arrays):
        """Draw a cached frame's cloth (arrays from snapshot()), or the live cloth (None)."""
        if arrays is None or not self.active:
            self._view = None
            return
        g = self.gpu
        if self._view is None:
            self._view = {name: g.buffer(max(16, self.built.n * 16), f'cloth-view-{name}') for name in ('X', 'V', 'N', 'S', 'P')}
        for name in ('X', 'V', 'N', 'S'):
            a = np.ascontiguousarray(np.asarray(arrays[name], np.float32))
            if len(a) != self.built.n:
                self._view = None
                return
            g.write_buffer(self._view[name], a)
        w = np.zeros((self.built.n, 4), np.float32)   # (frames cached before cloth got wet have none)
        if arrays.get('W') is not None and len(arrays['W']) == self.built.n:
            w[:, 3] = np.asarray(arrays['W'], np.float32)
        g.write_buffer(self._view['P'], w)
        if 'DR' not in self._view:
            self._view['DR'] = g.buffer(16 + DROPS * 32, 'cloth-view-drops')
        g.write_buffer(self._view['DR'], self._drops_array(arrays.get('DR')))
        self._view_on = True

    def _holes(self):
        """The holes bullets made through the cloth (ballistics_engine.py sets holes_buf: [0].x how many, then u, v,
        radius, fabric each), or none."""
        hb = getattr(self, 'holes_buf', None)
        if hb is None:
            if getattr(self, '_no_holes_buf', None) is None:
                self._no_holes_buf = self.gpu.buffer(16, 'cloth-no-holes')
                self.gpu.write_buffer(self._no_holes_buf, np.zeros(4, np.float32))
            hb = self._no_holes_buf
        return hb

    def _src(self):
        if self._view is not None and getattr(self, '_view_on', False):
            return self._view
        return self.bufs

    def _raster(self):
        if self.raster is None:
            from .raster import Raster
            self.raster = Raster(self.gpu)
        return self.raster

    # (the raster layer's targets, as they were the cloth's own)
    col = property(lambda self: self._raster().col)
    aux = property(lambda self: self._raster().aux)
    glow = property(lambda self: self._raster().glow)
    fabric = property(lambda self: self._raster().fabric)
    lq = property(lambda self: self._raster().lq)

    def _ensure_draw(self, w, h):
        g = self.gpu
        if self._pipe is None:
            dev = g.device
            ent = []
            for i in range(7):   # the cloth's own buffers are read by the vertex stage; its materials by both
                ent.append({'binding': i, 'visibility': (SS.FRAGMENT | SS.VERTEX) if i == 6 else SS.VERTEX,
                            'buffer': {'type': 'read-only-storage'}})
            for i in (7, 8):
                ent.append({'binding': i, 'visibility': SS.FRAGMENT, 'texture': {'sample_type': 'float', 'view_dimension': '3d'}})
            ent.append({'binding': 9, 'visibility': SS.FRAGMENT, 'texture': {'sample_type': 'float', 'view_dimension': '2d'}})
            ent.append({'binding': 10, 'visibility': SS.FRAGMENT, 'sampler': {'type': 'filtering'}})
            for i in (11, 12, 13):
                ent.append({'binding': i, 'visibility': SS.FRAGMENT, 'buffer': {'type': 'read-only-storage'}})
            for i in (14, 15):
                ent.append({'binding': i, 'visibility': SS.FRAGMENT, 'texture': {'sample_type': 'float', 'view_dimension': '3d'}})
            ent.append({'binding': 16, 'visibility': SS.VERTEX, 'buffer': {'type': 'read-only-storage'}})   # wetness
            for i in (17, 18):   # Lume's light grids (lume_light.wgsl)
                ent.append({'binding': i, 'visibility': SS.FRAGMENT, 'texture': {'sample_type': 'float', 'view_dimension': '3d'}})
            ent.append({'binding': 19, 'visibility': SS.FRAGMENT, 'buffer': {'type': 'read-only-storage'}})   # bullet holes
            self.layout0 = dev.create_bind_group_layout(entries=ent)
            from .renderer import BB_DEFINES
            module = dev.create_shader_module(code=load_wgsl('cloth_draw.wgsl', BB_DEFINES), label='cloth_draw')
            self._pipe = dev.create_render_pipeline(
                layout=dev.create_pipeline_layout(bind_group_layouts=[self.layout0, g.arena.layout]),
                vertex={'module': module, 'entry_point': 'vs', 'buffers': []},
                fragment={'module': module, 'entry_point': 'fs', 'targets': [
                    {'format': 'rgba16float'}, {'format': 'rgba16float'}, {'format': 'rgba16float'}]},
                primitive={'topology': 'triangle-list', 'cull_mode': 'none'},
                depth_stencil={'format': 'depth32float', 'depth_write_enabled': True, 'depth_compare': 'less'},
                label='cloth')
            # burnt-off cloth: flakes of ash and embers (the same bindings)
            self._pipe_flake = dev.create_render_pipeline(
                layout=dev.create_pipeline_layout(bind_group_layouts=[self.layout0, g.arena.layout]),
                vertex={'module': module, 'entry_point': 'vs_flake', 'buffers': []},
                fragment={'module': module, 'entry_point': 'fs_flake', 'targets': [
                    {'format': 'rgba16float'}, {'format': 'rgba16float'}, {'format': 'rgba16float'}]},
                primitive={'topology': 'triangle-list', 'cull_mode': 'none'},
                depth_stencil={'format': 'depth32float', 'depth_write_enabled': True, 'depth_compare': 'less'},
                label='cloth-flakes')
            # drops of water dripping off wet cloth: the lights' bindings and the drops (binding 17)
            ent_d = [e for e in ent if 7 <= e['binding'] <= 15]
            ent_d.append({'binding': 17, 'visibility': SS.VERTEX, 'buffer': {'type': 'read-only-storage'}})
            self.layout_drop = dev.create_bind_group_layout(entries=ent_d)
            self._pipe_drop = dev.create_render_pipeline(
                layout=dev.create_pipeline_layout(bind_group_layouts=[self.layout_drop, g.arena.layout]),
                vertex={'module': module, 'entry_point': 'vs_drop', 'buffers': []},
                fragment={'module': module, 'entry_point': 'fs_drop', 'targets': [
                    {'format': 'rgba16float'}, {'format': 'rgba16float'}, {'format': 'rgba16float'}]},
                primitive={'topology': 'triangle-list', 'cull_mode': 'none'},
                depth_stencil={'format': 'depth32float', 'depth_write_enabled': True, 'depth_compare': 'less'},
                label='cloth-drops')
        self._raster().ensure(w, h)

    def draw(self, b, renderer, camstate, fire, look, size, jitter=(0.0, 0.0), shutter=0.0, solver=None,
             light_gain=1.0, fire_lights=True, lamp_count=None, rp=None):
        """Draw the cloth for one anti-aliasing pass (before the march; its aux is the march's limit), into the raster
        layer's render pass `rp` (raster.py), or a pass of its own.
        solver: the fire's volume (its light volume lights and shadows the cloth), or None (a liquid scene:
        the key light and the sky only); light_gain: its reflected light times this (the liquid render's
        exposure); fire_lights: lit by the fire's point lights (Renderer.light's); lamp_count: the lights in the
        set in the renderer's lamp buffer (Renderer.pack_lamps), when Renderer.light has not packed them."""
        from .. engine import camera as cam
        from .renderer import LAMP_SCALE  # noqa: F401  (lamps are packed by the renderer)
        w, h = size
        self._ensure_draw(w, h)
        g = self.gpu
        src = self._src()
        r = renderer
        lt = r.LT if (getattr(r, '_lamps_on', 0) and r.LT is not None) else r._empty
        vol_on = solver is not None and getattr(r, 'L0', None) is not None
        L0, L1, E = (r.L0, r.L1, r.E) if vol_on else (r._empty, r._empty, r._empty)
        lume = vol_on and getattr(r, 'LV_lume', None) is not None
        if lume:
            L0 = r.L0_march   # (Lume's: the key light's shadows through the smoke as deep as they are; only .a is read)
        lights = r.lights if (fire_lights and r.lights is not None) else self._no_lights()
        count = r.light_count if (fire_lights and r.lights is not None) else self._no_count
        entries = [
            {'binding': 0, 'resource': {'buffer': src['X'].buf, 'offset': 0, 'size': src['X'].size}},
            {'binding': 1, 'resource': {'buffer': src['N'].buf, 'offset': 0, 'size': src['N'].size}},
            {'binding': 2, 'resource': {'buffer': src['S'].buf, 'offset': 0, 'size': src['S'].size}},
            {'binding': 3, 'resource': {'buffer': self.bufs['T'].buf, 'offset': 0, 'size': self.bufs['T'].size}},
            {'binding': 4, 'resource': {'buffer': self.bufs['UV'].buf, 'offset': 0, 'size': self.bufs['UV'].size}},
            {'binding': 5, 'resource': {'buffer': src['V'].buf, 'offset': 0, 'size': src['V'].size}},
            {'binding': 6, 'resource': {'buffer': self.bufs['DM'].buf, 'offset': 0, 'size': self.bufs['DM'].size}},
            {'binding': 7, 'resource': L0.view},
            {'binding': 8, 'resource': L1.view},
            {'binding': 9, 'resource': r.bb.view},
            {'binding': 10, 'resource': g.linear},
            {'binding': 11, 'resource': {'buffer': lights.buf, 'offset': 0, 'size': lights.size}},
            {'binding': 12, 'resource': {'buffer': count.buf, 'offset': 0, 'size': count.size}},
            {'binding': 13, 'resource': {'buffer': r._lamp_buf.buf, 'offset': 0, 'size': r._lamp_buf.size}},
            {'binding': 14, 'resource': lt.view},
            {'binding': 15, 'resource': E.view},
            {'binding': 16, 'resource': {'buffer': src['P'].buf, 'offset': 0, 'size': src['P'].size}},
            {'binding': 17, 'resource': (r.LV_lume if lume else r._empty).view},
            {'binding': 18, 'resource': (r.LVD_lume if lume else r._empty).view},
            {'binding': 19, 'resource': {'buffer': self._holes().buf, 'offset': 0, 'size': self._holes().size}},
        ]
        bg = g.device.create_bind_group(layout=self.layout0, entries=entries)
        l2w = np.asarray(fire.local_to_world(), float)
        w2l = np.linalg.inv(l2w)
        eye = (w2l @ np.append(np.asarray(camstate.eye, float), 1.0))[:3]
        view = np.asarray(camstate.view)
        fwd_w = -view[2, :3] / max(np.linalg.norm(view[2, :3]), 1e-12)
        fwd = w2l[:3, :3] @ fwd_w
        fwd = fwd / max(np.linalg.norm(fwd), 1e-12)
        p0 = camstate.inv_view_proj @ np.array([0.0, 0.0, 0.0, 1.0])
        near = float(np.dot(p0[:3] / p0[3] - np.asarray(camstate.eye), fwd_w))
        sd = cam.sun_direction(look.sun_azimuth, look.sun_elevation)
        sd_l = w2l[:3, :3] @ np.asarray(sd, float)
        sd_l = sd_l / max(np.linalg.norm(sd_l), 1e-12)
        # the fire's point lights are absolute (emission x volume / r^2 is irradiance in the render's units);
        # scaled like the fire light on the smoke
        e_ref = float(getattr(look, 'fire_scatter', 1.0))
        dims = solver.dims if solver is not None else (1, 1, 1)
        hcell = solver.h if solver is not None else 1.0
        org = solver.origin if solver is not None else (0.0, 0.0, 0.0)
        ld = r.light_dims or (1, 1, 1)
        from .renderer import pack_look
        u = (Uniforms().m4(camstate.view_proj).m4(l2w).v4(w, h, *jitter).v4(*eye, near).v4(*fwd, shutter)
             .v4(*sd_l, e_ref).v4(*org, hcell).v4(*ld, lamp_count if lamp_count is not None else (getattr(r, '_lamps_on', 0) if vol_on else 0))
             .v4(*(np.asarray(ld, float) / np.asarray(dims, float)),
                 (2.0 if getattr(r, 'occluded', False) else 1.0) if vol_on else 0.0))
        pack_look(u, look, r.log_y_ref(look.flame_k), getattr(look, 'time', 0.0))
        u.v4(light_gain, 1.0 if lume else 0.0)
        off = b.uniform_offset(u)
        own = rp is None
        if own:
            rp = self._raster().begin(b, w, h)
        rp.set_pipeline(self._pipe)
        rp.set_bind_group(0, bg)
        rp.set_bind_group(1, g.arena.group, [off])
        rp.draw(3 * len(self.built.tris), 1)
        rp.set_pipeline(self._pipe_flake)
        rp.draw(6 * FLAKES * self.built.n, 1)
        drops = src.get('DR') if src is not self.bufs else (self.bufs['DR'] if self._may_be_wet else None)
        if drops is not None:
            ent_d = [e for e in entries if 7 <= e['binding'] <= 15]
            ent_d.append({'binding': 17, 'resource': {'buffer': drops.buf, 'offset': 0, 'size': drops.size}})
            rp.set_pipeline(self._pipe_drop)
            rp.set_bind_group(0, g.device.create_bind_group(layout=self.layout_drop, entries=ent_d))
            rp.set_bind_group(1, g.arena.group, [off])
            rp.draw(6 * DROPS, 1)
        if own:
            rp.end()

    def layer(self, b):
        """The drawn raster layer as one texture for the liquid's march (raster.py)."""
        return self._raster().layer(b)

    def _no_lights(self):
        if getattr(self, '_nolights', None) is None:
            self._nolights = self.gpu.buffer(32, 'cloth-no-lights')
            self._no_count = self.gpu.buffer(16, 'cloth-no-light-count')
            self.gpu.write_buffer(self._no_count, np.zeros(4, np.uint32))
        return self._nolights

    def merge(self, b, renderer, pass_index, passes, hold_tolerance=0.0):
        """Put the drawn raster layer under the marched fire (raster.py)."""
        self._raster().merge(b, renderer, pass_index, passes, hold_tolerance)
