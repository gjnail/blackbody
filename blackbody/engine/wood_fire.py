"""Wood burning as wood burns: what makes a piece of it catch, how fast a flame spreads over it, how it chars, how
much heat it gives off while it does, and when it burns through (solids.py Solids.burn, for things that break and
burn).

- It catches when the heat reaching its surface has warmed it to its piloted ignition temperature (about 300-350 °C):
  under a steady flux q a thermally thick solid gets there after t = (pi/4) k rho c (T_ig - T_0)^2 / q^2, so the dose
  of q^2 it has taken is what counts (and below about 12 kW/m^2 it never gets there: the heat leaks away as fast as
  it comes). The heat is the flame's (or hot smoke's) next to it: convection, and the radiation of the flame's soot.
- A flame spreads over it fast upward (the flame above a burning patch heats the wood above it: that comes from the
  gas) and slowly sideways and down: a millimetre or so a second over pine, slower over denser wood.
- Burning, it chars at its charring rate (0.65 mm/min for softwoods, 0.5 for hardwoods, faster in more heat) from
  each face the flames reach, and gives off heat at a rate that peaks as it catches and falls as its char thickens
  (a wood fire's heat release, as a cone calorimeter measures it).
- Its flames go out where too little heat reaches it to keep it burning (a lone board), leaving glowing char that can
  catch again; once charred through it is spent, and its char glows and crumbles.
Burn speed-up (Spreading fire) runs all of this that many times faster than in reality, the flames and the smoke at
their own speed (a time-lapse of a building burning down)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIGMA = 5.67e-8          # W/m^2/K^4
T_AMBIENT = 293.0        # K
FLAME_K = 1250.0         # K: the gas at field temperature 1 (the flames' real temperature, as cloth.RAD_FLAME_K)
HOT_K = 1450.0           # K: the hottest the gas gets
H_CONV = 15.0            # W/m^2/K: buoyant flames and hot smoke along a surface
FLAME_EMISSIVITY = 0.25  # of the flame's soot next to a surface (a flame a few tens of cm thick): with H_CONV, about
                         # 40-50 kW/m^2 in the flames, as fire tests measure on walls
Q_CRIT = 12.0e3          # W/m^2: piloted ignition of wood needs at least this
Q_SUSTAIN = 8.0e3        # W/m^2: the flames on wood go out where less than this reaches it for a while
SUSTAIN_TIME = 20.0      # s (real) under Q_SUSTAIN before its flames go out
SUSTAIN_GAS = 1.5        # s (of the shot) at the least: its flames' heat comes from the gas, which a time-lapse does
                         # not hurry (a flame takes a second or so to build up round wood that has just caught)
COOL_TIME = 60.0         # s (real): a surface that was warming toward catching cools back
HRR_TIME = 90.0          # s (real): its heat release falls from its peak to its plateau as its char thickens
HRR_PLATEAU = 0.4        # of its peak
Q_REF = 40.0e3           # W/m^2: the heat its own flames give it (char rate and heat release are measured there)
GLOW_TIME = 240.0        # s (real): spent char glows, dimming, for about this long before it falls to ash
EMBER_TIME = 90.0        # s (real): char that is put out but not spent glows for about this long
COVER_TIME = 30.0        # s (real): a burning piece's flames cross it, its char spreading over it (drawn: y past 1)


@dataclass(frozen=True)
class WoodFire:
    """How a wood burns (real units)."""
    ignition: float      # K, piloted
    inertia: float       # k rho c, J^2/(m^4 K^2 s)
    char_rate: float     # m/s in its own flames
    hrr: float           # W/m^2 at its peak (what it gives the fire, in proportion)
    creep: float         # m/s: a flame's spread over it sideways (down a third of it, up three times it)

    @property
    def dose(self):
        """The (q^2 s) it takes to catch."""
        return 0.25 * np.pi * self.inertia * (self.ignition - T_AMBIENT) ** 2


def props(material_key, density):
    """The burning of a wood, from its kind and density (Eurocode 5 charring rates; ignition and heat release from cone
    calorimeter data: denser woods are slower to catch and slower to char)."""
    k = 0.04 + 1.4e-4 * density          # W/m/K (MacLean)
    c = 2500.0                            # J/kg/K, warm and drying
    softwood = material_key in ('wood', 'pine', 'spruce', 'fir', 'cedar')
    if material_key == 'balsa':
        beta = 1.5 / 60000.0
    elif material_key == 'plywood':
        beta = 1.0 / 60000.0
    elif material_key == 'mdf':
        beta = 0.8 / 60000.0
    else:
        beta = (0.65 if softwood else 0.5) / 60000.0 * (450.0 / max(density, 150.0)) ** 0.25
    ignition = 273.15 + (300.0 if density < 300.0 else 320.0 if softwood else 350.0)
    hrr = 180.0e3 * (density / 500.0) ** 0.3
    creep = 1.0e-3 * (0.04 + 1.4e-4 * 500.0) * 500.0 / max(k * density, 5.0)
    return WoodFire(ignition, k * density * c, beta, hrr, creep)


def is_wood(material_key):
    from ..scene.materials import material
    m = material(material_key)
    return m.pattern == 'wood' or m.pattern.startswith('wood_')


def gas_kelvin(T):
    """The gas's real temperature (K) at field temperature T (0: the air, 1: the flames)."""
    T = np.asarray(T, float)
    return (T_AMBIENT + (FLAME_K - T_AMBIENT) * np.clip(T, 0.0, 1.0)
            + (HOT_K - FLAME_K) * (1.0 - np.exp(-np.maximum(T - 1.0, 0.0))))


def flux(T):
    """The heat (W/m^2) reaching a surface from gas at field temperature T next to it: convection and the soot's
    radiation."""
    tk = gas_kelvin(T)
    return H_CONV * (tk - T_AMBIENT) + FLAME_EMISSIVITY * SIGMA * (tk ** 4 - T_AMBIENT ** 4)


# A piece's surface, as the points its heat is felt and its burning is kept at: a 3 x 3 grid on each of its six faces
# (in its own frame, in half extents; face order -x, +x, -y, +y, -z, +z, and on each face its other two axes in order,
# u then v, at -1, 0, 1, its edges and its middle: stage.wgsl point_burn reads them so). A face's edge spots are the
# next face's too, and the next piece's across a glued joint: those burn as one (sync_spots).
GRID = (-1.0, 0.0, 1.0)
FACES = [(ax, sgn) for ax in range(3) for sgn in (-1.0, 1.0)]


def template():
    """The 54 points of a piece's surface, in half extents (54, 3)."""
    pts = []
    for ax, sgn in FACES:
        others = [a for a in range(3) if a != ax]
        for u in GRID:
            for v in GRID:
                q = np.zeros(3)
                q[ax] = sgn
                q[others[0]], q[others[1]] = u, v
                pts.append(q)
    return np.array(pts)


def template_edges():
    """Which of a piece's 54 points a flame creeps between: its grid neighbours on each face ((m, 2) index pairs)."""
    t = template()
    d = np.linalg.norm(t[:, None, :] - t[None, :, :], axis=2)
    i, j = np.nonzero(np.triu((d > 1e-6) & (d <= 1.05)))
    return np.stack([i, j], 1)


def template_opposite():
    """Each of a piece's 54 points' twin on the opposite face, straight through the piece from it."""
    j = np.arange(len(FACES) * 9)
    return ((j // 9) ^ 1) * 9 + j % 9


def template_sync():
    """Which of a piece's 54 points are the same point of it (on the edge two faces share, at a corner three do)."""
    t = template()
    d = np.linalg.norm(t[:, None, :] - t[None, :, :], axis=2)
    i, j = np.nonzero(np.triu(d <= 1e-6, k=1))
    return np.stack([i, j], 1)


def sync_spots(X, st, pairs):
    """Spots that are the same place on the wood burn as one: each takes the furthest on of them (caught, charred,
    burning, glowing)."""
    if not len(pairs):
        return
    a, b = pairs[:, 0], pairs[:, 1]
    for _ in range(2):      # (a corner is three faces' spot: twice round settles it)
        for col, fn in ((0, np.minimum), (1, np.maximum), (2, np.maximum), (3, np.maximum)):
            v = fn(X[a, col], X[b, col])
            X[a, col] = v
            X[b, col] = v
        for k in ('char', 'lit'):
            v = np.maximum(st[k][a], st[k][b])
            st[k][a] = v
            st[k][b] = v


def step_spots(wf, X, st, T, pos, edges, lens, thick, dts, speed=1.0, opp=None):
    """One step of a wood's spots (dts: real seconds, times Burn speed-up `speed`).
    X (N, 4), as burn_common.wgsl's spots: x fuel left (what is left of the wood through it), y catching (1: alight;
    past 1, how far its char has spread round it), z 1 (2 once charred through), w how fiercely it burns, or how its
    char glows. st: 'char' (m), 'lit' (s alight, real), 'low' (s of the shot under Q_SUSTAIN). T: the gas's field
    temperature at each; pos (N, 3): where each is now; edges (E, 2) and lens (E,): the spots a flame creeps between
    and how far apart; thick (N,): the wood's thickness straight through each, to its twin on the opposite face (opp,
    (N,)), whose char eats into it from the other side; with no opp, how deep its own char goes before it is through."""
    q = flux(T)
    fuel = X[:, 0] > 0.0
    spent = X[:, 2] >= 1.5
    alight = (X[:, 1] >= 1.0) & fuel & ~spent
    warming = ~alight & fuel & ~spent
    # catching: from the heat on it, or from a burning neighbour (faster upward, slower down)
    dose = np.maximum(q * q - Q_CRIT * Q_CRIT, 0.0) / wf.dose
    if len(edges):
        for a, b in ((edges[:, 0], edges[:, 1]), (edges[:, 1], edges[:, 0])):
            on = alight[a] & warming[b]
            if not on.any():
                continue
            a, b, L = a[on], b[on], lens[on]
            d = pos[b] - pos[a]
            up = d[:, 1] / np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            v = wf.creep * np.where(up > 0.0, 1.0 + 2.0 * up, 1.0 + 0.67 * up)
            np.maximum.at(dose, b, v / np.maximum(L, 1e-3))
    heat = warming & (dose > 0.0)
    X[heat, 1] = np.minimum(X[heat, 1] + dose[heat] * dts, 1.0)
    cool = warming & (dose <= 0.0)
    X[cool, 1] = np.maximum(X[cool, 1] - X[cool, 1] / COOL_TIME * dts, 0.0)
    X[warming, 3] = np.maximum(X[warming, 3] - dts / EMBER_TIME, 0.0)      # (put-out char glows on, dimming)
    new = warming & (X[:, 1] >= 1.0)
    st['lit'][new & (st['char'] <= 0.0)] = 0.0     # (wood that caught again, its char on it, does not flare as fresh)
    st['low'][new] = 0.0
    alight |= new
    # burning: charring, its heat release falling as its char thickens; out where too little heat reaches it
    if alight.any():
        a = alight
        st['lit'][a] += dts
        X[a, 1] = 1.0 + 0.5 * (1.0 - np.exp(-st['lit'][a] / COVER_TIME))
        boost = np.sqrt(np.clip(q[a] / Q_REF, 0.5, 2.0))
        st['char'][a] += wf.char_rate * boost * dts
        X[a, 3] = (HRR_PLATEAU + (1.0 - HRR_PLATEAU) * np.exp(-st['lit'][a] / HRR_TIME)) * np.clip(boost, 0.7, 1.3)
        low = q[a] < Q_SUSTAIN
        st['low'][a] = np.where(low, st['low'][a] + dts / max(speed, 1e-6), 0.0)
    # what is left of the wood straight through each spot: its char, and its twin's from the other face
    tot = st['char'] + (st['char'][opp] if opp is not None else 0.0)
    X[:, 0] = np.minimum(X[:, 0], np.maximum(1.0 - tot / np.maximum(thick, 1e-4), 0.0))
    if alight.any():
        starved = max(SUSTAIN_TIME / max(speed, 1e-6), SUSTAIN_GAS)
        idx = np.nonzero(alight)[0]
        out = idx[(st['low'][idx] > starved) & (X[idx, 0] > 0.0)]
        X[out, 1] = 0.9          # (its flames go out; glowing, it can catch again)
        X[out, 3] = 1.0
    # charred through (from either face): spent, its char showing on both and glowing
    through = (X[:, 0] <= 0.0) & (X[:, 2] < 1.5) & (tot > 0.0)
    X[through, 1] = np.maximum(X[through, 1], 1.0)
    X[through, 2] = 2.0
    X[through, 3] = 1.0
    # spent: its char glows, dimming
    s = (X[:, 2] >= 1.5) & (X[:, 3] > 0.0)
    X[s, 3] = np.maximum(X[s, 3] - dts / GLOW_TIME, 0.0)


def pieces_from_spots(X, n, open_=None):
    """A set's pieces' fire (n, 4) from their spots (n x 54), over their open faces' (a face glued to the next piece
    is inside the wood): fuel left, the mean of its spots'; alight while any of them burns; spent once most of it is
    charred through; how fiercely it burns, or glows, the mean of its spots'."""
    S = X.reshape(n, -1, 4)
    w = np.ones(S.shape[:2]) if open_ is None else np.asarray(open_, float).reshape(n, -1)
    w = np.where(w.sum(1, keepdims=True) > 0.0, w, 1.0)
    m = w.sum(1)
    F = np.zeros((n, 4))
    F[:, 0] = (S[:, :, 0] * w).sum(1) / m
    F[:, 1] = np.where(w > 0.0, S[:, :, 1], 0.0).max(1)
    F[:, 2] = np.where(((S[:, :, 2] >= 1.5) * w).sum(1) / m >= 0.6, 2.0, 1.0)
    F[:, 3] = (S[:, :, 3] * w).sum(1) / m
    return F
