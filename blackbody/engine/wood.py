"""Wood: species, and how wood breaks.

Wood is a bundle of long hollow fibres (tracheids, vessels) running along the trunk, laid down a ring a year, crossed
by thin radial plates (rays). That makes it one of the most anisotropic things there is: along the grain it is as
strong in tension as mild steel for its weight (pine about 90 MPa), across it a twentieth of that (2-6 MPa), and it
splits along the grain, between the fibres, at a fraction of the energy it takes to break across them. So a board
bent until it breaks does not snap clean: the fibres on its tension side give way one bundle after another, each at
its own weak point, and the bundles peel apart along the grain, leaving the long jagged splinters of a greenstick
break; a log struck along its grain splits along the rays into wedges; a bullet tears splinters out of the far side of
a board, along the grain.

The pattern here (fracture 'splinters'): the object cut along its grain (its longest side; a cylinder's axis) into
bundles of fibres, each bundle into lengths, the cuts between lengths staggered from bundle to bundle and slanted, so
each piece ends in a point and the break between them steps along the grain. A cylinder (a log, a post, a dowel)
splits radially, into wedges round its pith (and rings of them, for a thick one). Each bond gets a strength for the
way it is loaded: one between two lengths of a bundle (end grain to end grain) holds what the fibres do along the
grain, one between two bundles side by side holds what the wood does across it.

The species (SPECIES) carry what their wood is and looks like: density, stiffness and strengths from the USDA Wood
Handbook (FPL-GTR-282, at 12% moisture), and the look the stage draws (wood.wgsl): earlywood and latewood colours,
ring width, how much of each ring is latewood, rays, pores, knots, heartwood.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .fracture import Fracture, make_piece, shape_planes, touching_bonds


@dataclass(frozen=True)
class Species:
    key: str
    label: str
    density: float          # kg/m^3 (12% moisture)
    E: float                # Pa: stiffness along the grain
    mor: float              # Pa: bending strength (modulus of rupture)
    across: float           # Pa: tension across the grain (it splits)
    shear: float            # Pa: shear along the grain
    early: tuple            # earlywood colour (linear rgb)
    late: tuple             # latewood colour
    heart: tuple            # heartwood's tint (a multiplier on both, toward the pith)
    ring: float             # m: a ring's width (a year's growth)
    late_share: float       # how much of each ring is latewood (0..1)
    sharp: float            # how sharply earlywood turns to latewood (1: softwoods, abrupt; 0: gradual)
    rays: float             # how strongly its rays show (oak's flecks)
    pores: float            # pores on its faces: ring-porous species' earlywood vessels (oak, ash)
    knots: float            # knots per metre along the grain
    figure: float           # waviness of its grain (figured maple, walnut)
    roughness: float        # sawn, unfinished
    sheen: float = 0.3      # its silky sheen along the fibres
    pattern: int = 1        # the stage's pattern (materials.PATTERNS)


# Colours are of sawn, unfinished wood, linear (the sRGB of reference photos, decoded).
SPECIES = {s.key: s for s in (
    Species('pine', 'Pine', 510.0, 10.1e9, 87e6, 3.0e6, 9.5e6, (0.62, 0.43, 0.24), (0.43, 0.24, 0.1), (0.95, 0.85, 0.72),
            0.0045, 0.3, 0.9, 0.05, 0.0, 2.5, 0.05, 0.62, pattern=1),
    Species('spruce', 'Spruce', 450.0, 10.5e9, 70e6, 2.5e6, 7.0e6, (0.74, 0.6, 0.42), (0.58, 0.42, 0.25), (1.0, 1.0, 1.0),
            0.0035, 0.22, 0.85, 0.05, 0.0, 1.5, 0.03, 0.6, pattern=11),
    Species('fir', 'Douglas fir', 530.0, 13.4e9, 85e6, 2.3e6, 7.8e6, (0.62, 0.38, 0.2), (0.38, 0.17, 0.07), (0.9, 0.78, 0.68),
            0.004, 0.4, 0.95, 0.05, 0.0, 1.0, 0.04, 0.6, pattern=12),
    Species('oak', 'Oak', 700.0, 12.5e9, 99e6, 5.5e6, 12.3e6, (0.5, 0.36, 0.2), (0.38, 0.25, 0.12), (0.92, 0.85, 0.78),
            0.003, 0.55, 0.3, 0.9, 1.0, 0.6, 0.05, 0.6, sheen=0.45, pattern=13),
    Species('ash', 'Ash', 670.0, 12.0e9, 103e6, 6.5e6, 13.2e6, (0.66, 0.55, 0.38), (0.5, 0.39, 0.24), (0.95, 0.92, 0.86),
            0.004, 0.45, 0.3, 0.3, 0.9, 0.4, 0.05, 0.55, sheen=0.45, pattern=14),
    Species('maple', 'Maple', 705.0, 12.6e9, 109e6, 7.0e6, 16.2e6, (0.72, 0.58, 0.39), (0.64, 0.49, 0.31), (0.95, 0.9, 0.85),
            0.0035, 0.15, 0.1, 0.2, 0.1, 0.3, 0.25, 0.45, sheen=0.55, pattern=15),
    Species('birch', 'Birch', 690.0, 13.9e9, 114e6, 6.3e6, 13.0e6, (0.73, 0.57, 0.38), (0.66, 0.5, 0.31), (0.92, 0.88, 0.82),
            0.003, 0.15, 0.1, 0.1, 0.15, 0.4, 0.15, 0.5, sheen=0.5, pattern=16),
    Species('walnut', 'Walnut', 610.0, 11.6e9, 101e6, 4.8e6, 9.4e6, (0.26, 0.15, 0.08), (0.17, 0.09, 0.045), (0.85, 0.82, 0.8),
            0.0035, 0.3, 0.2, 0.1, 0.4, 0.5, 0.3, 0.5, sheen=0.55, pattern=17),
    Species('cherry', 'Cherry', 560.0, 10.3e9, 85e6, 3.9e6, 11.7e6, (0.55, 0.28, 0.13), (0.45, 0.2, 0.08), (0.88, 0.82, 0.78),
            0.003, 0.2, 0.15, 0.1, 0.1, 0.4, 0.15, 0.45, sheen=0.55, pattern=18),
    Species('mahogany', 'Mahogany', 590.0, 10.1e9, 80e6, 3.5e6, 8.5e6, (0.38, 0.16, 0.08), (0.3, 0.12, 0.055), (0.9, 0.88, 0.86),
            0.005, 0.15, 0.1, 0.15, 0.35, 0.1, 0.35, 0.5, sheen=0.65, pattern=19),
    Species('teak', 'Teak', 650.0, 12.3e9, 98e6, 4.0e6, 12.0e6, (0.5, 0.3, 0.12), (0.4, 0.22, 0.08), (0.9, 0.85, 0.8),
            0.004, 0.3, 0.3, 0.1, 0.4, 0.3, 0.1, 0.5, sheen=0.4, pattern=20),
    Species('cedar', 'Cedar', 370.0, 7.7e9, 52e6, 1.5e6, 6.8e6, (0.55, 0.27, 0.13), (0.42, 0.17, 0.07), (0.85, 0.75, 0.7),
            0.003, 0.3, 0.8, 0.05, 0.0, 1.0, 0.05, 0.6, pattern=21),
    Species('balsa', 'Balsa', 160.0, 3.7e9, 20e6, 0.8e6, 2.0e6, (0.8, 0.7, 0.53), (0.75, 0.64, 0.47), (1.0, 1.0, 1.0),
            0.008, 0.1, 0.1, 0.05, 0.3, 0.0, 0.05, 0.7, sheen=0.4, pattern=22),
    # engineered: cross-laid veneers (its faces one species' grain, its edges the plies), and a felt of fibres
    Species('plywood', 'Plywood', 600.0, 8.0e9, 50e6, 3.5e6, 7.0e6, (0.68, 0.52, 0.32), (0.55, 0.38, 0.2), (1.0, 1.0, 1.0),
            0.004, 0.25, 0.7, 0.05, 0.1, 0.3, 0.08, 0.6, pattern=23),
    Species('mdf', 'MDF', 750.0, 3.5e9, 35e6, 0.7e6, 2.0e6, (0.48, 0.33, 0.19), (0.48, 0.33, 0.19), (1.0, 1.0, 1.0),
            0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, sheen=0.1, pattern=24),
)}

WOOD_KEYS = ('wood',) + tuple(k for k in SPECIES if k != 'pine')
SPECIES_OF = {'wood': 'pine'}     # (the materials' key for pine is 'wood')

# A bond's strength over the material's (materials.py: what an irregular chunk of it holds). Between two lengths of a
# bundle, end grain to end grain, the fibres hold it, as strong in bending as a sawn board is (about half the clear
# wood's modulus of rupture, as common lumber with its knots and slope of grain is rated: pine's 0.48 x 87 MPa over the
# material's 2 MPa): ALONG times; the joint gives as the wood does first (bend_welds), so a long plank
# sags under a load before it breaks. Between two bundles side by side the wood holds it across its grain: SIDE x
# ACROSS x its strength across over its strength in bending, less on a wide face: a split
# along the grain runs as a crack, the load on its tip, so a face bigger than SPLIT_FACE holds (SPLIT_FACE / its
# area)^SPLIT_POWER of that. (Not the handbook's ratio: the welds judge a side bond's bending on a square section, which
# undersells a long thin face; these are what make a board behave.) Calibrated (tests/test_wood.py): a 19 mm pine board on a
# 30 cm span takes a 1 kg weight dropped 30 cm on it whole, and a 5 kg one dropped 50 cm snaps it in two, its fibres
# broken at staggered places along it.
ALONG = 21.0
SIDE = 5.0
ACROSS = 40.0
SPLIT_POWER = 1.0
SPLIT_FACE = 2.0e-4          # m^2
FLEX_STIFF = float(__import__('os').environ.get('BB_FLEX', '2.4'))   # (calibrated: tests/test_wood.py; the joints and the
                            # side bonds together give this much more than E I / L alone says)
BEND_IMPEDANCE = 0.8        # the softest impedance a bending joint is given: below about 0.7 a chain of them rings up
                            # and flies apart (stiff springs, (0.6 / dt)^2 at a small step, applied that weakly). Past
                            # it, the wood's give comes from a softer spring instead (bend_welds)
SLIVER = 0.12               # a bond with less than this of the median's area is dropped
SLANT = (0.35, 0.8)         # radians: how far a length's end is slanted off square to the grain (a splinter's point)


def species_of(key):
    """The species a material key is (pine for 'wood'), or None for a material that is not wood."""
    return SPECIES.get(SPECIES_OF.get(key, key))


def is_wood(key):
    return species_of(key) is not None


def grain_axis(shape, size):
    """The axis the grain runs along, in an object's own frame: a cylinder's axis; a box's longest side (the stage draws
    its rings round the same one: of equal sides, y before z before x)."""
    if shape in ('cylinder', 'capsule'):
        return 1
    s = np.abs(np.asarray(size, float))
    if s[1] >= s[0] and s[1] >= s[2]:
        return 1
    if s[2] >= s[0] and s[2] >= s[1]:
        return 2
    return 0


def bond_strengths(species: Species):
    """(along, across): a bond's strength as a multiple of the material's, end grain to end grain and side by side."""
    along = ALONG
    across = SIDE * max(species.across, species.shear * 0.5) / max(species.mor, 1.0) * ACROSS
    return along, across


def fibres(shape, size, pieces=24, seed=0, impact=None, species=None):
    """Wood cut along its grain (see the module's notes): a box into bundles of lengths, staggered and slanted; a
    cylinder into wedges round its pith, in lengths. Each bond's `k`: how much stronger than the material's strength it
    is (bond_strengths). None for a shape it does not cut (the caller falls back on its own)."""
    sp = species or SPECIES['pine']
    rng = np.random.default_rng(int(seed) + 4241)
    s = np.abs(np.asarray(size, float))
    if shape == 'box':
        frac = _boards(s, int(pieces), rng, impact)
    elif shape == 'cylinder':
        frac = _logs(s, int(pieces), rng, impact)
    else:
        return None
    along, across = bond_strengths(sp)
    ax = grain_axis(shape, s)
    # (slivers of contact where two slanted ends nearly miss each other carry nothing, and a weld that small breaks at
    # rest: dropped)
    if frac.bonds:
        med = float(np.median([b.area for b in frac.bonds]))
        frac.bonds = [b for b in frac.bonds if b.area >= SLIVER * med]
    for b in frac.bonds:
        end = abs(float(np.asarray(b.normal, float)[ax])) > 0.6
        # (a split along the grain runs as a crack, its tip carrying the load: the wider the face, the less of its
        # strength it holds, as fracture mechanics has it, by the square root of its size)
        b.k = along if end else across * min(1.0, (SPLIT_FACE / max(b.area, 1e-12)) ** SPLIT_POWER)
        if end:
            b.flex = _flex(frac.pieces[b.i], frac.pieces[b.j], b, ax)
    return frac


def bend_welds(solids):
    """Wood bends before it breaks: each weld between two lengths of a bundle (_flex) made as stiff in bending as the
    bundle of wood is, E I / L, by its impedance. MuJoCo's soft constraint at rest gives way by r = R f / (k d), its
    R = (1 - d) / d x A (A: the two bodies' mean inverse inertias, body_invweight0), k its stiffness (solref): so d is
    set where that is the wood's own give, (1 - d) / d^2 = k / (K A). Never stiffer than any weld (solids WELD_SOLIMP),
    and never softer than BEND_IMPEDANCE: there k is softened instead, k = K A (1 - d) / d^2, the same give. Run once
    the model is compiled."""
    import mujoco
    from .solids import WELD_SOLIMP
    m = solids.model
    for ps in solids.sets:
        cols = getattr(solids, '_scene', None).colliders
        c = cols[ps.index] if ps.index < len(cols) else {}
        sp = species_of(c.get('material', ''))
        if sp is None:
            continue
        soft = {}       # piece -> the softest of its end joints: (its give, its impedance, its stiffness)
        sides = []
        for n, bond in enumerate(ps.frac.bonds):
            e = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f'bond{ps.index}_{n}')
            if e < 0:
                continue
            flex = getattr(bond, 'flex', None)
            if flex is None:
                sides.append((e, bond.i, bond.j))
                continue
            K = FLEX_STIFF * sp.E * float(flex[0])
            A = float(m.body_invweight0[m.eq_obj1id[e], 1] + m.body_invweight0[m.eq_obj2id[e], 1])
            k = -float(m.eq_solref[e, 0])
            if K <= 0.0 or A <= 0.0 or k <= 0.0:
                continue
            x = k / (K * A)
            d = (math.sqrt(1.0 + 4.0 * x) - 1.0) / (2.0 * x)
            d = min(d, WELD_SOLIMP[1])
            if d < BEND_IMPEDANCE:
                d = BEND_IMPEDANCE
                k = K * A * (1.0 - d) / (d * d)
                m.eq_solref[e, 0] = -k
                m.eq_solref[e, 1] = -2.0 * math.sqrt(k)
            m.eq_solimp[e, 0] = d
            m.eq_solimp[e, 1] = d
            # (a short stiff length rings faster than the step can follow: its joint is held as stiff as the step
            # allows, softer than the wood. Struck, a softer thing feels a smaller peak force, as the square root of its
            # stiffness for the same energy: so its strength is held down by as much, that a blow breaks it as a blow
            # breaks the wood; a long plank, as stiff as its wood, keeps all of its strength)
            K_sim = k * d * d / max((1.0 - d) * A, 1e-30)
            if K_sim < K and n < len(ps.welds):
                w = list(ps.welds[n])
                w[4] *= math.sqrt(K_sim / K)
                ps.welds[n] = tuple(w)
            give = (1.0 - d) / max(k * d * d, 1e-30)
            for p in (bond.i, bond.j):
                if p not in soft or give > soft[p][0]:
                    soft[p] = (give, d, k)
        # (bundles side by side bend together: where their joints are staggered, a side bond as stiff as the bundles
        # would hold the bend off at each joint and take all of it itself, twisting apart. It gives as they do)
        for e, i, j in sides:
            if i in soft and j in soft:
                _g, d, k = max(soft[i], soft[j])
                m.eq_solimp[e, 0] = d
                m.eq_solimp[e, 1] = d
                m.eq_solref[e, 0] = -k
                m.eq_solref[e, 1] = -2.0 * math.sqrt(k)


def _flex(pi, pj, bond, ax):
    """How a joint between two lengths of a bundle bends (wood bends before it breaks): (I / L (m^3): times the wood's
    stiffness along its grain E, the joint's stiffness in bending, as the bundle's own over its length; the two
    lengths' moment of inertia about it, per unit density (m^5)). bend_welds makes the weld that soft."""
    perp = [k for k in range(3) if k != ax]
    ei, ej = np.ptp(pi.verts, axis=0), np.ptp(pj.verts, axis=0)
    h = max(min(float(ei[perp].min()), float(ej[perp].min())), 1e-4)       # (it bends through its thinnest side)
    A = float(bond.area) * abs(float(np.asarray(bond.normal, float)[ax]))  # (the bundle's section, square to the grain)
    L = max(abs(float(pj.centroid[ax] - pi.centroid[ax])), 1e-4)
    I_i = pi.volume * (ei[ax] ** 2 + h ** 2) / 12.0
    I_j = pj.volume * (ej[ax] ** 2 + h ** 2) / 12.0
    return (A * h * h / 12.0 / L, float(I_i * I_j / max(I_i + I_j, 1e-30)))


def _slants(cuts, lo, hi, reach, rng):
    """A slant (radians, either way) for each cut along the grain, as far off square as SLANT allows, but never so far
    that it crosses the cut beside it (or the end) within `reach` of the bundle's middle: the lengths still fill it."""
    out = []
    for j, c in enumerate(cuts):
        before = c - (cuts[j - 1] if j > 0 else lo)
        after = (cuts[j + 1] if j + 1 < len(cuts) else hi) - c
        most = math.atan(0.45 * min(before, after) / max(reach, 1e-9))
        out.append(min(rng.uniform(*SLANT), most) * rng.choice((-1.0, 1.0)))
    return out


def _staggered(lo, hi, n, rng, near=None, crowd=0.0):
    """n - 1 cut positions between lo and hi, about evenly spaced, each moved by up to 0.45 of a spacing; closer
    together round `near` (a hit) by `crowd`."""
    if n <= 1:
        return np.zeros(0)
    t = (np.arange(1, n) + rng.uniform(-0.45, 0.45, n - 1)) / n
    x = lo + (hi - lo) * t
    if near is not None and crowd > 0.0:
        x = x + (near - x) * crowd * np.exp(-((x - near) / max(0.25 * (hi - lo), 1e-6)) ** 2)
    return np.sort(x)


def _boards(s, pieces, rng, impact):
    """A box along its grain (its longest side): bundles across its width and through its thickness, each in lengths
    whose ends are staggered from bundle to bundle and slanted."""
    L = grain_axis('box', s)
    a, b = [k for k in range(3) if k != L]
    if s[a] < s[b]:
        a, b = b, a                      # (a: across its width, b: through its thickness)
    nb = 1 if s[b] < 0.02 else (2 if s[b] < 0.05 else 3)        # (a board one bundle thick: s[b] is half its thickness)
    wb = 2.0 * s[b] / nb
    na = int(np.clip(round(2.0 * s[a] / max(1.6 * wb, 0.01)), 2, 8))
    nl = max(2, int(round(pieces / (na * nb))))
    box = shape_planes('box', s)
    imp = None if impact is None else np.asarray(impact, float)
    frac = Fracture()
    a_cuts = _staggered(-s[a], s[a], na, rng)
    b_cuts = _staggered(-s[b], s[b], nb, rng)
    a_edges = np.concatenate([[-s[a] - 1.0], a_cuts, [s[a] + 1.0]])
    b_edges = np.concatenate([[-s[b] - 1.0], b_cuts, [s[b] + 1.0]])
    e = np.eye(3)
    for ia in range(na):
        for ib in range(nb):
            # this bundle's cuts along the grain: its own stagger, slanted toward a direction of its own
            cuts = _staggered(-s[L], s[L], nl, rng, near=None if imp is None else imp[L], crowd=0.5)
            ends = []
            wa = 0.5 * (min(a_edges[ia + 1], s[a]) - max(a_edges[ia], -s[a]))
            wb_ = 0.5 * (min(b_edges[ib + 1], s[b]) - max(b_edges[ib], -s[b]))
            sides = [(e[a], wa) if rng.random() < 0.6 else (e[b], wb_) for _ in cuts]
            th_all = _slants(cuts, -s[L], s[L], max(w for _, w in sides) if sides else 1.0, rng)
            for c, th, (side, _w) in zip(cuts, th_all, sides):
                nrm = e[L] * math.cos(th) + side * math.sin(th)
                # (through the bundle's middle at c along the grain)
                mid = np.zeros(3)
                mid[L] = c
                mid[a] = 0.5 * (max(a_edges[ia], -s[a]) + min(a_edges[ia + 1], s[a]))
                mid[b] = 0.5 * (max(b_edges[ib], -s[b]) + min(b_edges[ib + 1], s[b]))
                ends.append(np.append(nrm, float(nrm @ mid)))
            for j in range(len(ends) + 1):
                pl = list(box)
                inner = [False] * len(box)
                for axis, edges, k in ((a, a_edges, ia), (b, b_edges, ib)):
                    if k > 0:
                        p = np.zeros(4)
                        p[axis], p[3] = -1.0, -edges[k]
                        pl.append(p)
                        inner.append(True)
                    if k < len(edges) - 2:
                        p = np.zeros(4)
                        p[axis], p[3] = 1.0, edges[k + 1]
                        pl.append(p)
                        inner.append(True)
                if j > 0:
                    pl.append(-ends[j - 1])
                    inner.append(True)
                if j < len(ends):
                    pl.append(ends[j])
                    inner.append(True)
                piece, _ = make_piece(np.asarray(pl, float), np.asarray(inner, bool))
                if piece is not None and piece.volume > 1e-12:
                    frac.pieces.append(piece)
    frac.bonds = touching_bonds(frac.pieces)
    return frac


def _logs(s, pieces, rng, impact):
    """A cylinder along its grain (its axis, y): wedges round its pith (and, for a thick one, an inner ring of them
    inside an outer), each wedge in lengths, staggered and slanted as a board's bundles are."""
    R, H = float(s[0]), float(s[1])
    cyl = shape_planes('cylinder', s)
    nw = int(np.clip(round(math.sqrt(max(pieces, 4) * 1.2)), 5, 12))
    rings = 2 if R > 0.12 else 1
    per = max(2, int(round(pieces / (nw * rings))))
    turn = rng.uniform(0.0, 2.0 * math.pi)
    ang = turn + (np.arange(nw) + rng.uniform(-0.25, 0.25, nw)) * (2.0 * math.pi / nw)
    ang = np.sort(ang)
    r_in = R * rng.uniform(0.4, 0.55)
    frac = Fracture()
    imp = None if impact is None else np.asarray(impact, float)
    for k in range(nw):
        t0, t1 = ang[k], ang[(k + 1) % nw] + (2.0 * math.pi if k == nw - 1 else 0.0)
        mid = 0.5 * (t0 + t1)
        # the wedge's two radial planes through the axis
        side0 = np.array([math.sin(t0), 0.0, -math.cos(t0), 0.0])        # (keeps it on the side of t0 toward t1)
        side1 = np.array([-math.sin(t1), 0.0, math.cos(t1), 0.0])
        chord = np.array([math.cos(mid), 0.0, math.sin(mid)])
        half = 0.5 * (t1 - t0)
        for ring in range(rings):
            cuts = _staggered(-H, H, per, rng, near=None if imp is None else imp[1], crowd=0.5)
            ends = []
            for c, th in zip(cuts, _slants(cuts, -H, H, R, rng)):
                nrm = np.array([0.0, math.cos(th), 0.0]) + chord * math.sin(th)
                rmid = 0.5 * R if rings == 1 else (0.5 * r_in if ring == 0 else 0.5 * (r_in + R))
                p = chord * rmid
                p[1] = c
                ends.append(np.append(nrm, float(nrm @ p)))
            for j in range(len(ends) + 1):
                pl = list(cyl) + [side0, side1]
                inner = [False] * len(cyl) + [True, True]
                if rings == 2:
                    # the inner ring inside a chord at r_in, the outer outside it (a chord keeps each piece convex)
                    d = r_in * math.cos(half)
                    pl.append(np.append(chord, d) if ring == 0 else np.append(-chord, -d))
                    inner.append(True)
                if j > 0:
                    pl.append(-ends[j - 1])
                    inner.append(True)
                if j < len(ends):
                    pl.append(ends[j])
                    inner.append(True)
                piece, _ = make_piece(np.asarray(pl, float), np.asarray(inner, bool))
                if piece is not None and piece.volume > 1e-12:
                    frac.pieces.append(piece)
    frac.bonds = touching_bonds(frac.pieces)
    return frac


# ---- cleaving: an edge driven into end grain -------------------------------------------------------------------------

CLEAVE_G = 800.0        # J/m^2: what splitting wood along its grain takes, with the wedge's friction in the cleft (its
                        # fracture energy across the grain is 150-350 J/m^2; an axe needs about 100 J for a 25 cm round
                        # of pine, half a metre long)
CLEAVE_FACE = 0.9       # cos: an edge or a corner meets the wood (no face of the striker's box within ~25 deg of square
                        # to the contact)
CLEAVE_END = 0.75       # cos: the contact is on end grain (its normal within ~40 deg of the grain)
CLEAVE_OPEN = 0.15      # of the edge's speed: how fast the halves are pushed apart by the wedge's faces as it goes in
CLEAVE_REST = 0.08      # s: a striker that has cleaved a piece of wood does not cleave it again for this long


class Cleaver:
    """An edge driven into the end grain of wood splits it along the grain, in the plane of the edge, as an axe or a
    wedge does: a rigid striker cannot cut its way in, so its blow opens the split instead. Solids calls step() after
    each of its steps. Its blow's energy along the contact's normal (half its mass times its speed into the wood
    squared) runs the crack as far down the grain as it pays for at CLEAVE_G over the wood's width; the welds across the
    split there go, and the halves are pushed apart, the edge's faces then holding them open. A striker is any free body
    that is not wood, meeting it on an edge or a corner of a box (a wedge, an axe's head, a falling crate's corner)."""

    def __init__(self, solids):
        import mujoco
        m = solids.model
        self.piece = {}          # body -> (set index, piece index)
        self.axis = {}           # set index -> the grain's axis in its pieces' frame (0, 1, 2)
        self.width = {}          # set index -> its width across the grain (m)
        self.length = {}         # set index -> its length along the grain (m)
        cols = solids._scene.colliders
        for si, ps in enumerate(solids.sets):
            c = cols[ps.index] if ps.index < len(cols) else {}
            mat = c.get('material', '')
            if species_of(mat) is None or mat == 'mdf' or c.get('shape') not in ('box', 'cylinder'):
                continue
            ax = grain_axis(c['shape'], ps.size)
            self.axis[si] = ax
            s = np.abs(np.asarray(ps.size, float))
            self.width[si] = 2.0 * float(max(s[k] for k in range(3) if k != ax))
            self.length[si] = 2.0 * float(s[ax])
            for k, b in enumerate(ps.bodies):
                self.piece[int(b)] = (si, k)
        self.box = int(mujoco.mjtGeom.mjGEOM_BOX)
        self.v = {}              # body -> its velocity at the end of the last step (world m/s)
        self.rest = {}           # (body, set) -> the time it last cleaved it
        self.free = [b for b in range(1, m.nbody) if b not in self.piece and m.body_jntnum[b] > 0
                     and m.jnt_type[m.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE]

    @staticmethod
    def of(solids):
        """A Cleaver for this world, or None (no wood that splits)."""
        if solids.rehearsal or not solids.sets:
            return None
        c = Cleaver(solids)
        return c if c.axis and c.free else None

    def step(self, solids):
        m, d = solids.model, solids.data
        W = solids._w
        if W is not None and d.ncon:
            g1 = d.contact.geom1[:d.ncon]
            g2 = d.contact.geom2[:d.ncon]
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            for i in range(d.ncon):
                for wb, sb, sg, sgn in ((int(b1[i]), int(b2[i]), int(g2[i]), 1.0),
                                        (int(b2[i]), int(b1[i]), int(g1[i]), -1.0)):
                    if wb in self.piece and sb in self.v and m.geom_type[sg] == self.box:
                        self._blow(solids, i, wb, sb, sg, sgn)
        for b in self.free:
            a = m.jnt_dofadr[m.body_jntadr[b]]
            self.v[b] = np.array(d.qvel[a:a + 3], float)

    def _blow(self, solids, i, wb, sb, sg, sgn):
        m, d = solids.model, solids.data
        si, _k = self.piece[wb]
        if si not in self.axis:
            return
        last = self.rest.get((sb, si))
        if last is not None and 0.0 <= solids.time - last < CLEAVE_REST:
            return
        # (the contact's normal points from geom1 to geom2: `into` from the striker into the wood)
        n = np.array(d.contact.frame[i][:3], float)
        into = -n * sgn
        g = d.xmat[wb].reshape(3, 3)[:, self.axis[si]]
        if abs(float(g @ into)) < CLEAVE_END:
            return
        R = d.geom_xmat[sg].reshape(3, 3)
        c = np.abs(R.T @ into)
        if float(c.max()) > CLEAVE_FACE:
            return       # (a face of it, flat on the end grain: it bounces, as a mallet would)
        vn = float(self.v[sb] @ into)
        if vn <= 0.5:
            return
        E = 0.5 * float(m.body_subtreemass[sb]) * vn * vn
        edge = R[:, int(np.argmin(c))]
        pn = np.cross(g, edge)
        if float(np.linalg.norm(pn)) < 0.3:
            return       # (an edge along the grain cleaves nothing)
        pn /= np.linalg.norm(pn)
        self.rest[(sb, si)] = solids.time
        reach = min(E / (CLEAVE_G * self.width[si]), self.length[si] * 1.05)
        if reach < 0.005:
            return
        at = np.array(d.contact.pos[i], float)
        ps = solids.sets[si]
        W = solids._w
        rows = np.nonzero((W['set'] == si) & (W['over'] >= 0) & (W['other'] >= 0))[0]
        if not len(rows):
            return
        pf = d.xpos[ps.bodies[W['first'][rows]]]
        po = d.xpos[ps.bodies[W['other'][rows]]]
        across = ((pf - at) @ pn) * ((po - at) @ pn) < 0.0
        down = ((0.5 * (pf + po) - at) @ into) < reach
        cut = rows[across & down]
        if not len(cut):
            return
        d.eq_active[W['eq'][cut]] = 0
        W['over'][cut] = -(1 << 40)
        for r in cut:
            solids.breaks.append((solids.time, d.xpos[W['body'][r]].copy(), float(W['area'][r]), int(W['collider'][r])))
        # the wedge's faces push the halves apart as it goes in
        side = (d.xpos[ps.bodies] - at) @ pn
        near = ((d.xpos[ps.bodies] - at) @ into) < reach + 0.25 * self.length[si]
        for k in np.nonzero(near)[0]:
            a = m.jnt_dofadr[m.body_jntadr[ps.bodies[k]]]
            d.qvel[a:a + 3] += pn * (CLEAVE_OPEN * vn * (1.0 if side[k] > 0.0 else -1.0))
