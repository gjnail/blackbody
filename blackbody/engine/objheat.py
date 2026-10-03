"""Objects' heat: every object (collider) warms and cools as real things do, by what it is made of and how big it is.

Each object has a skin (as deep as heat soaks into its material in SKIN_TIME seconds: a steel bar's is most of it, a
log's a couple of millimetres) and a core, each at its own temperature, the heat between them flowing by the material's
conductivity. Everything else meets the skin. Once a frame, from points spread over the surface:
  - the air: convection (natural, or forced by the gas or wind moving past) with the air a cell off the surface (the
    fire's gas at its real temperature in a fire box, the ambient air otherwise), as the liquid's heat does
    (liq_therm_heat.wgsl);
  - radiation: what falls on it from the shared radiant sources (radiant.py: the fire's hot gas, glowing matter, lava,
    the other objects), absorbed by its emissivity, and its own radiation to the cool surroundings;
  - the ground, and other objects it rests on or against: two bodies in contact meet at the temperature their
    effusivities weigh them to, the heat flowing as e1 e2 / (e1 + e2) / sqrt(pi t) a square metre and kelvin for the
    time t they have touched (semi-infinite bodies: a pan on a hot plate, a hot brick on the floor);
  - water: with Heat and phase changes on, what the liquid's heat took from it or gave it (liq_therm_heat.wgsl adds it
    up per object, through the boiling curve); without, the same boiling curve and convection against water at the
    liquid's temperature;
  - lava: conduction through the chilled skin of melt against it;
  - matter: what the wax, chocolate, metal, sand and snow touching it took from it or gave it (mpm_heat.wgsl adds it
    up per object).
It warms its surroundings in turn: it radiates (a source for each of its faces in the shared list), the gas next to it
is heated (obj_air.wgsl), and everything that touches it reads its temperature. 'Keeps its temperature' objects (a hot
plate on its element) give and take heat but stay where they are set. Domain > Heat speed runs it faster than real.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .gpu import Uniforms, groups_1d
from .radiant import FLAME_K, MAX_K, OBJECT_SOURCES
from .solver import MAX_COLLIDERS

SIGMA = 5.670e-8
SKIN_TIME = 20.0          # s: an object's skin is as deep as heat soaks into its material in this long
CONTACT_GAP = 0.006       # m: surfaces this close touch (resting contacts are slightly soft)
CONTACT_MIN_S = 0.5       # s: the shortest contact time the conductance is taken at (it falls as 1/sqrt(t))
LAVA_H = 500.0            # W/m^2/K: lava against an object, through the chilled skin of melt between them
POINTS = 160              # points on an object's surface, about
GONE_Y = -1000.0          # an object put out of the way (a broken one's whole, a crumbled piece) is below this
MOST_K = 3000.0
STEP_SHARE = 0.3          # each substep changes a skin by at most about this share of the way to what it meets


def _yaw_matrix(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _quat_matrix(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotation(cg):
    """A ColliderGPU's orientation (own frame -> fire-local), as colliders.wgsl turns it: its yaw, then its quaternion."""
    return _quat_matrix(cg.quat) @ _yaw_matrix(cg.rot_y)


def surface_points(shape, size, n=POINTS, box=None):
    """Points spread over a shape's surface in its own frame: (positions (m, 3), outward normals (m, 3), the area each
    stands for (m,)). A mesh by its bounding box (`box`: its corner and far corner)."""
    s = np.abs(np.asarray(size, float))
    if shape == 'sphere':
        r = s[0]
        k = np.arange(n) + 0.5
        phi = np.arccos(1.0 - 2.0 * k / n)
        th = math.pi * (1.0 + 5.0 ** 0.5) * k
        nrm = np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1)
        return nrm * r, nrm, np.full(n, 4.0 * math.pi * r * r / n)
    if shape == 'cylinder':
        r, hh = s[0], s[1]
        a_side, a_cap = 2.0 * math.pi * r * 2.0 * hh, math.pi * r * r
        total = a_side + 2.0 * a_cap
        ns = max(8, int(round(n * a_side / total)))
        rows = max(1, int(round(math.sqrt(ns * 2.0 * hh / (2.0 * math.pi * r)))))
        cols = max(4, ns // rows)
        th = (np.arange(cols) + 0.5) / cols * 2.0 * math.pi
        y = ((np.arange(rows) + 0.5) / rows * 2.0 - 1.0) * hh
        T, Y = np.meshgrid(th, y)
        nrm = np.stack([np.cos(T).ravel(), np.zeros(T.size), np.sin(T).ravel()], 1)
        pos = nrm * r + np.stack([np.zeros(T.size), Y.ravel(), np.zeros(T.size)], 1)
        area = np.full(T.size, a_side / T.size)
        nc = max(4, int(round(n * a_cap / total)))
        k = np.arange(nc) + 0.5
        rr = r * np.sqrt(k / nc)
        tc = math.pi * (1.0 + 5.0 ** 0.5) * k
        caps_p, caps_n, caps_a = [pos], [nrm], [area]
        for sgn in (1.0, -1.0):
            caps_p.append(np.stack([rr * np.cos(tc), np.full(nc, sgn * hh), rr * np.sin(tc)], 1))
            caps_n.append(np.tile([0.0, sgn, 0.0], (nc, 1)))
            caps_a.append(np.full(nc, a_cap / nc))
        return np.concatenate(caps_p), np.concatenate(caps_n), np.concatenate(caps_a)
    centre = np.zeros(3)
    half = s[:3]
    if box is not None:
        lo, hi = np.asarray(box[0], float), np.asarray(box[1], float)
        centre, half = 0.5 * (lo + hi), np.maximum(0.5 * (hi - lo), 1e-4)
    faces = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    areas = [4.0 * half[b] * half[c] for a, b, c in faces]
    total = 2.0 * sum(areas)
    ps, ns, as_ = [], [], []
    for (a, b, c), fa in zip(faces, areas):
        m = max(1, int(round(n * fa / total)))
        nb = max(1, int(round(math.sqrt(m * half[b] / max(half[c], 1e-9)))))
        nc = max(1, int(round(m / nb)))
        u = ((np.arange(nb) + 0.5) / nb * 2.0 - 1.0) * half[b]
        v = ((np.arange(nc) + 0.5) / nc * 2.0 - 1.0) * half[c]
        U, V = np.meshgrid(u, v)
        for sgn in (1.0, -1.0):
            p = np.zeros((U.size, 3))
            p[:, a] = sgn * half[a]
            p[:, b] = U.ravel()
            p[:, c] = V.ravel()
            q = np.zeros((U.size, 3))
            q[:, a] = sgn
            ps.append(p + centre)
            ns.append(q)
            as_.append(np.full(U.size, fa / U.size))
    return np.concatenate(ps), np.concatenate(ns), np.concatenate(as_)


def contact_conductance(e1, e2, t):
    """W/m^2/K between two bodies (effusivities e1, e2) touching for t seconds."""
    return e1 * e2 / max(e1 + e2, 1e-9) / math.sqrt(math.pi * max(t, CONTACT_MIN_S))


def boil_or_convect(t_wall_c, t_water_c, boil_c=100.0):
    """Heat flux (W/m^2, out of the wall) from a wall at t_wall_c into water at t_water_c: the boiling curve over the
    boiling point (liquid_thermal.boil_flux, with film boiling's radiation), natural convection below it."""
    from .liquid_thermal import boil_flux
    d = np.asarray(t_wall_c, float) - t_water_c
    hn = 220.0 * np.abs(d) ** (1.0 / 3.0)
    conv = hn * d
    sup = np.asarray(t_wall_c, float) - boil_c
    bf = np.array([boil_flux(x) for x in np.atleast_1d(sup)]).reshape(np.shape(sup))
    film = sup >= 110.0
    tw, tb = np.asarray(t_wall_c, float) + 273.15, boil_c + 273.15
    bf = bf + np.where(film, 0.9 * SIGMA * (tw ** 4 - tb ** 4), 0.0)
    return np.where(bf > conv, bf, conv)


@dataclass
class Thing:
    """One object's heat."""
    index: int                 # its place in the scene's collider list
    shape: str
    size: tuple
    hollow: float
    keeps: bool
    start_k: float
    eps: float
    effusivity: float
    c_skin: float              # J/K
    c_core: float
    g_core: float              # W/K between skin and core
    area: float
    lp: np.ndarray = None      # its surface points (own frame), normals, areas
    ln: np.ndarray = None
    la: np.ndarray = None
    face: np.ndarray = None    # which of its six faces each point is on (by its normal)
    box: tuple = None          # a mesh's bounds
    skin: float = 293.0
    core: float = 293.0
    ground_s: float = 0.0      # how long it has touched the ground (heat seconds)
    touch_s: dict = field(default_factory=dict)


class ObjectHeat:
    """The heat of a scene's objects (colliders, in Scene.colliders_gpu's order)."""

    def __init__(self, gpu):
        self.gpu = gpu
        self.things: list[Thing] = []
        self._key = None
        self._k = None
        self._bufs = None
        self.speed = 4.0
        self.ambient_k = 293.0
        self.ground_k = 293.0
        self.ground_e = 1500.0
        self.ground_on = True
        self.ground_y = 0.0
        self.liquid_temp_c = 20.0
        self.liquid_lava_k = 0.0   # the scene's liquid is lava this hot (K; 0: it is water)
        self.boil_c = 100.0
        self.wind = (0.0, 0.0, 0.0)
        self.matter_j = None       # J each object gave the matter over the last frame (mpm_heat.wgsl)
        self.liquid_j = None       # and the water (liq_therm_heat.wgsl)
        self.last = {}             # the last step's sums, per object (tests, the log)
        self.world = None          # the last step's points in the world: (positions, normals, areas, owners)

    # -- set-up ---------------------------------------------------------------------------------

    @property
    def active(self):
        return bool(self.things)

    @staticmethod
    def wanted(scene):
        """Whether a scene's objects need their heat: there are objects, and something in the scene is hot or cold (a
        fire, an object not at the air's temperature, matter or a liquid that takes heat, lava)."""
        cols = [c for c in scene.colliders if c['enabled']]
        if not cols or scene.kind == 'cloud':
            return False
        if scene.kind in ('fire', 'both'):
            return True
        if any(float(c.get('temperature', 20.0)) != 20.0 for c in cols):
            return True
        q = scene.data.get('liquid', {})
        return bool(q.get('thermal')) or any(m.get('enabled', True) for m in getattr(scene, 'matter', []))

    def configure(self, scene):
        """The objects' make-up from the scene; True if it changed (and the heat starts over)."""
        from ..scene.materials import MATERIALS, FLOORS
        from .solids import mesh_of, shape_area, shape_volume
        if not self.wanted(scene):
            changed = bool(self.things)
            self.things = []
            self._key = None
            return changed
        d = scene.data['domain']
        self.speed = max(float(d.get('matter_heat_speed', 4.0)), 1e-3)
        comp = scene.data.get('composite', {})
        floor = FLOORS.get(comp.get('floor', 'studio')) if isinstance(comp, dict) else None
        self.ground_e = float(getattr(floor, 'effusivity', 1500.0)) if floor is not None else 1500.0
        self.ground_on = bool(d.get('ground', True))
        q = scene.data.get('liquid', {})
        both = scene.kind == 'both'
        amb = float(scene.data['shading']['ambient_k'])
        self.ambient_k = amb if (both or scene.kind == 'fire') else float(q.get('air_temp', 20.0)) + 273.15
        self.ground_k = (float(q.get('ground_temp', 20.0)) + 273.15) if scene.kind in ('liquid', 'both') else self.ambient_k
        self.liquid_temp_c = float(q.get('liquid_temp', 20.0))
        self.boil_c = float(q.get('boil_point', 100.0))
        key = []
        things = []
        j = 0
        for i, c in enumerate(scene.colliders):
            if not c['enabled']:
                continue
            if j >= MAX_COLLIDERS:
                break
            j += 1
            m = MATERIALS.get(c.get('material'), MATERIALS['wood'])
            size = tuple(float(x) for x in scene.get(('collider', i, 'size'), scene.start))
            shape = c['shape']
            hollow = float(scene.get(('collider', i, 'hollow'), scene.start) or 0.0)
            box = None
            if shape == 'mesh':
                got, _ = mesh_of(scene, c, size)
                if got is not None and len(got[0]):
                    v = got[0]
                    box = (v.min(axis=0), v.max(axis=0))
                ext = (0.5 * (box[1] - box[0])) if box is not None else np.abs(size)
                vol = shape_volume('box', ext) * 0.5
                area = shape_area('box', ext) * 0.8
            else:
                vol = shape_volume(shape, size)
                area = shape_area(shape, size)
                if hollow > 0.0:
                    inner = np.maximum(np.abs(np.asarray(size, float)) - hollow, 0.0)
                    vol = max(vol - shape_volume(shape, inner), 0.05 * vol)
            rho = float(c.get('density', 0.0) or 0.0) or m.density
            cp = m.heat_capacity
            k_cond = m.effusivity ** 2 / max(m.density * cp, 1e-9)
            alpha = k_cond / max(rho * cp, 1e-9)
            thick = vol / max(area, 1e-9)                 # (a slab of the same volume and area: half its thickness)
            skin = min(math.sqrt(alpha * SKIN_TIME), 0.5 * thick)
            total = rho * cp * vol
            c_skin = max(rho * cp * area * skin, 1e-3 * total)
            c_core = max(total - c_skin, 0.05 * total)
            g = k_cond * area / max(0.5 * (thick - skin), 0.5 * skin, 1e-5)
            t0 = float(c.get('temperature', 20.0)) + 273.15
            lp, ln, la = surface_points(shape, size, box=box)
            face = np.argmax(np.abs(ln), axis=1) * 2 + (np.take_along_axis(ln, np.argmax(np.abs(ln), axis=1)[:, None], 1)[:, 0] < 0)
            th = Thing(i, shape, size, hollow, bool(c.get('keeps_temperature', False)), t0, m.emissivity, m.effusivity,
                       c_skin, c_core, g, area, lp, ln, la, face.astype(int), box, t0, t0)
            things.append(th)
            key.append((i, shape, size, hollow, th.keeps, t0, m.key, rho, None if box is None else tuple(np.round(box[0], 5))))
        key = (tuple(key), self.speed, self.ambient_k, self.ground_k, self.ground_e, self.ground_on)
        changed = key != self._key
        if changed:
            self.things = things
            self._key = key
            self._bufs = None
        return changed

    def reset(self):
        for t in self.things:
            t.skin = t.core = t.start_k
            t.ground_s = 0.0
            t.touch_s = {}
        self.matter_j = self.liquid_j = None
        self.last = {}

    # -- what the rest reads --------------------------------------------------------------------

    def temps_k(self):
        """Each object's surface temperature (K), in colliders_gpu's order."""
        return [t.skin for t in self.things]

    def temps_c(self):
        return tuple(t.skin - 273.15 for t in self.things)

    def collider_heat(self):
        """(surface K, effusivity) per object, as Scene.collider_heat (mpm_heat.wgsl, mpm_melt.wgsl)."""
        return [(t.skin, t.effusivity) for t in self.things]

    def state(self):
        return np.array([[t.skin, t.core, t.ground_s] for t in self.things], np.float64)

    def load_state(self, st):
        if st is None:
            return
        st = np.asarray(st, float)
        for t, row in zip(self.things, st):
            t.skin, t.core, t.ground_s = float(row[0]), float(row[1]), float(row[2])
            t.touch_s = {}

    def glowing(self, k=700.0):
        return any(t.skin > k for t in self.things)

    # -- the frame -------------------------------------------------------------------------------

    def _kernel(self):
        if self._k is None:
            self._k = self.gpu.kernel('obj_heat.wgsl', ['tex3d', 'tex3d', 'smp', 'rbuf', 'rbuf', 'rbuf', 'utex3d',
                                                        'utex3d', 'buf'], workgroup=(64, 1, 1))
        return self._k

    def _place(self, cols):
        """The points in the world for the objects as `cols` (ColliderGPU) has them: (positions, normals, areas, owners,
        rotations, alive per object)."""
        P, N, A, O, R, alive = [], [], [], [], [], []
        for k, t in enumerate(self.things):
            cg = cols[k] if k < len(cols) else None
            ok = cg is not None and cg.pos[1] > GONE_Y
            rot = rotation(cg) if cg is not None else np.eye(3)
            R.append(rot)
            alive.append(ok)
            if not ok:
                continue
            P.append(np.asarray(cg.pos, float) + t.lp @ rot.T)
            N.append(t.ln @ rot.T)
            A.append(t.la)
            O.append(np.full(len(t.la), k))
        if not P:
            return None
        return np.concatenate(P), np.concatenate(N), np.concatenate(A), np.concatenate(O), R, alive

    def sample(self, cols, gas=None, liquid=None, lava=None, radiant=None):
        """What the points meet (obj_heat.wgsl): per point (air K, air speed m/s, radiation W/m^2, wet 1/0, lava K, lava
        share); reads back. None if no object is in the scene's space."""
        placed = self._place(cols)
        if placed is None:
            return None, None
        P, N, A, O, R, alive = placed
        n = len(A)
        g = self.gpu
        if self._bufs is None or self._bufs[0] < n:
            cap = max(64, n)
            self._bufs = (cap, g.buffer(cap * 32, 'objheat-points'), g.buffer(cap * 32, 'objheat-met'))
        cap, pb, ob = self._bufs
        pt = np.zeros((cap, 8), np.float32)
        pt[:n, 0:3] = P
        pt[:n, 3] = O
        pt[:n, 4:7] = N
        pt[:n, 7] = A
        g.write_buffer(pb, pt)
        gas_on = gas is not None and getattr(gas, 'dims', None) is not None
        live = liquid is not None and getattr(liquid, 'dims', None) is not None
        u = Uniforms()
        u.v4(*(gas.origin if gas_on else (0.0, 0.0, 0.0)), gas.h if gas_on else 1.0)
        u.v4(*(gas.dims if gas_on else (1, 1, 1)), 1.0 if gas_on else 0.0)
        u.v4(*(liquid.origin if live else (0.0, 0.0, 0.0)), liquid.h if live else 1.0)
        u.v4(*(liquid.dims if live else (1, 1, 1)), 1.0 if live else 0.0)
        u.v4(1.0 if (lava is not None and gas_on) else 0.0)
        u.v4(self.ambient_k, FLAME_K, MAX_K, n)
        u.v4(*self.wind, 0.0)
        d3 = self._dummies()
        res = [gas.vel[0] if gas_on else d3[0], gas.scal[0] if gas_on else d3[0], g.linear,
               pb, radiant.RL if radiant is not None else d3[1], radiant.RLC if radiant is not None else d3[2],
               liquid.TYPE[0] if live else d3[3], lava if (lava is not None and gas_on) else d3[3], ob]
        with g.batch() as b:
            if radiant is None:
                b.clear_buffer(d3[2])
            b.run(self._kernel(), res, u, groups=groups_1d(n))
        out = np.frombuffer(g.read_buffer(ob, n * 32), np.float32).reshape(n, 8).astype(float)
        self.world = (P, N, A, O)
        return placed, np.nan_to_num(out)

    def _dummies(self):
        d = getattr(self, '_d3', None)
        if d is None:
            g = self.gpu
            t = g.texture3d((1, 1, 1), 'rgba16float', 'objheat-none')
            g.upload(t, np.zeros((1, 1, 1, 4), np.float16))
            t2 = g.texture3d((1, 1, 1), 'rgba32float', 'objheat-none-u')
            g.upload(t2, np.zeros((1, 1, 1, 4), np.float32))
            d = self._d3 = (t, g.buffer(48, 'objheat-no-sources'), g.buffer(16, 'objheat-no-count'), t2)
        return d

    def step(self, cols, fdt, gas=None, liquid=None, lava=None, radiant=None, water_thermal=False):
        """Warm and cool the objects through a frame of fdt seconds (before the frame's batch: it reads back), the objects
        where `cols` (ColliderGPU, colliders_gpu's order) has them."""
        if not self.things:
            return
        placed, met = self.sample(cols, gas, liquid, lava, radiant)
        nt = len(self.things)
        fixed = np.zeros(nt)                     # J this frame that does not depend on its temperature
        if self.matter_j is not None:
            fixed[:len(self.matter_j)] -= np.asarray(self.matter_j[:nt])
        if self.liquid_j is not None and water_thermal:
            fixed[:len(self.liquid_j)] -= np.asarray(self.liquid_j[:nt])
        self.matter_j = None
        self.liquid_j = None
        if placed is None:
            self._integrate(fixed, None, None, fdt)
            return
        P, N, A, O, R, alive = placed
        wet = met[:, 3] > 0.5
        in_lava = met[:, 5] > 0.5
        # the ground, and what touches what
        ground = np.zeros(len(A), bool)
        if self.ground_on:
            ground = (P[:, 1] - self.ground_y < CONTACT_GAP) & (N[:, 1] < -0.5)
        pairs = self._contacts(P, A, O, cols, R, alive)
        # (water against it: with the liquid's heat on, what it took is counted there, liq_therm_heat.wgsl)
        lava_k = met[:, 4]
        if self.liquid_lava_k > 0.0:
            # (the liquid itself is lava: it touches as lava)
            lava_k = np.where(wet, self.liquid_lava_k, lava_k)
            in_lava = in_lava | wet
            wet = wet & False
        terms = dict(air_k=met[:, 0], speed=met[:, 1], rad=met[:, 2], wet=wet & (not water_thermal), lava_k=lava_k,
                     in_lava=in_lava, open=~(wet | in_lava | ground), ground=ground, pairs=pairs, owner=O, area=A)
        self._integrate(fixed, terms, alive, fdt)

    def _contacts(self, P, A, O, cols, R, alive):
        """{(i, j): contact area m^2} for objects whose surfaces touch."""
        from .solids import shape_distance
        out = {}
        nt = len(self.things)
        if nt < 2:
            return out
        for j, tj in enumerate(self.things):
            if not alive[j]:
                continue
            cg = cols[j]
            sel = O != j
            if not sel.any():
                continue
            q = (P[sel] - np.asarray(cg.pos, float)) @ R[j]
            hull = None
            if tj.box is not None:
                hull = np.array([tj.box[0], tj.box[1]])
            d, _ = shape_distance(tj.shape, tj.size, q, hull)
            if tj.hollow > 0.0:
                d = np.maximum(d, -(d + tj.hollow))
            hit = d < CONTACT_GAP
            if not hit.any():
                continue
            own = O[sel][hit]
            ar = A[sel][hit]
            for i in np.unique(own):
                out[(int(i), j)] = float(ar[own == i].sum())
        # (each pair once: the smaller of the two sides' areas, where both see it)
        sym = {}
        for (i, j), a in out.items():
            key = (min(i, j), max(i, j))
            b = out.get((j, i), 0.0)
            sym[key] = min(a, b) if b > 0.0 else a
        return sym

    def _integrate(self, fixed, terms, alive, fdt):
        nt = len(self.things)
        T_s = np.array([t.skin for t in self.things])
        T_c = np.array([t.core for t in self.things])
        cs = np.array([t.c_skin for t in self.things])
        cc = np.array([t.c_core for t in self.things])
        gc = np.array([t.g_core for t in self.things])
        eps = np.array([t.eps for t in self.things])
        eff = np.array([t.effusivity for t in self.things])
        keep = np.array([t.keeps for t in self.things])
        heat_t = fdt * self.speed                   # the heat's seconds this frame
        amb = self.ambient_k
        if terms is not None:
            O, A = terms['owner'], terms['area']
            on = terms['open']
            ground = terms['ground']
            gs = np.array([t.ground_s for t in self.things])
            touching = np.zeros(nt, bool)
            np.logical_or.at(touching, O[ground], True)
            gs = np.where(touching, gs + heat_t, 0.0)
            g_ground = np.array([contact_conductance(eff[i], self.ground_e, gs[i]) for i in range(nt)])
            pair_g = {}
            for (i, j), a in terms['pairs'].items():
                t0 = self.things[i].touch_s.get(j, 0.0) + heat_t
                self.things[i].touch_s[j] = t0
                pair_g[(i, j)] = contact_conductance(eff[i], eff[j], t0) * a
            for i, t in enumerate(self.things):
                for j in [j for j in t.touch_s if (i, j) not in pair_g]:
                    del t.touch_s[j]       # (parted: a later touch starts over)
            # radiation absorbed is fixed over the frame; the rest depends on the skin's temperature
            q_rad = np.bincount(O, weights=eps[O] * terms['rad'] * A * on, minlength=nt)
        else:
            q_rad = np.zeros(nt)
        # substeps short enough for the stiffest exchange
        g_lin = gc / cs
        if terms is not None:
            h_air = np.maximum(1.52 * np.abs(terms['air_k'] - T_s[O]) ** (1.0 / 3.0), 5.7 + 3.8 * terms['speed'])
            lin = np.bincount(O, weights=A * (h_air * on + 4.0 * eps[O] * SIGMA * T_s[O] ** 3 * on
                                             + 5.0e4 * terms['wet'] + LAVA_H * terms['in_lava']
                                             + g_ground[O] * ground), minlength=nt)
            for (i, j), g in pair_g.items():
                lin[i] += g
                lin[j] += g
            g_lin = g_lin + lin / cs
        n_sub = int(min(400, max(1, math.ceil(heat_t * float(np.max(g_lin)) / STEP_SHARE))))
        dt = heat_t / n_sub
        sums = np.zeros(nt)
        for _ in range(n_sub):
            q = q_rad + fixed / heat_t
            if terms is not None:
                Ts = T_s[O]
                h_air = np.maximum(1.52 * np.abs(terms['air_k'] - Ts) ** (1.0 / 3.0), 5.7 + 3.8 * terms['speed'])
                q_p = (h_air * (terms['air_k'] - Ts) - eps[O] * SIGMA * (Ts ** 4 - amb ** 4)) * on
                q_p = q_p + g_ground[O] * (self.ground_k - Ts) * ground
                q_p = q_p + LAVA_H * (terms['lava_k'] - Ts) * terms['in_lava']
                if terms['wet'].any():
                    w = terms['wet']
                    q_p[w] = q_p[w] - boil_or_convect(Ts[w] - 273.15, self.liquid_temp_c, self.boil_c)
                q = q + np.bincount(O, weights=q_p * A, minlength=nt)
                for (i, j), g in pair_g.items():
                    f = g * (T_s[j] - T_s[i])
                    q[i] += f
                    q[j] -= f
            core_flow = gc * (T_c - T_s)
            sums += q * dt
            T_s = T_s + (q + core_flow) * dt / cs
            T_c = T_c - core_flow * dt / cc
            T_s = np.where(keep, [t.start_k for t in self.things], np.clip(T_s, 1.0, MOST_K))
            T_c = np.where(keep, [t.start_k for t in self.things], np.clip(T_c, 1.0, MOST_K))
        for k, t in enumerate(self.things):
            t.skin, t.core = float(T_s[k]), float(T_c[k])
            if terms is not None:
                t.ground_s = float(gs[k])
        self.last = {'substeps': n_sub, 'heat_j': sums, 'radiation_w': q_rad}

    def faces_at(self, cols, temps=None, hot=700.0):
        """The faces of the objects hotter than `hot` K where `cols` (ColliderGPU) has them, for drawing their glow and
        its light: (centre, outward normal, area m^2, size m, temperature K, emissivity) each; temps: their surface
        temperatures (K) to draw (a cached frame's), else their own."""
        out = []
        for k, t in enumerate(self.things):
            T = float(temps[k]) if temps is not None and k < len(temps) else t.skin
            cg = cols[k] if k < len(cols) else None
            if T <= hot or cg is None or cg.pos[1] <= GONE_Y:
                continue
            rot = rotation(cg)
            P = np.asarray(cg.pos, float) + t.lp @ rot.T
            N = t.ln @ rot.T
            for f in range(6):
                m = t.face == f
                if not m.any():
                    continue
                a = float(t.la[m].sum())
                c = (P[m] * t.la[m, None]).sum(0) / a
                nrm = (N[m] * t.la[m, None]).sum(0)
                nrm = nrm / max(float(np.linalg.norm(nrm)), 1e-9)
                out.append((c, nrm, a, float(np.linalg.norm(P[m].max(0) - P[m].min(0))), T, t.eps))
        return out

    # -- what it gives back -----------------------------------------------------------------------

    def radiant_sources(self):
        """Each hot object's faces as radiant sources (radiant.Radiant.set_objects): (centre, softening m^2, power W,
        outward normal, owner), from where the last step had them."""
        if self.world is None:
            return []
        P, N, A, O = self.world
        out = []
        for k, t in enumerate(self.things):
            sel = O == k
            if not sel.any():
                continue
            e = t.eps * SIGMA * (t.skin ** 4 - self.ambient_k ** 4)
            if e * t.area < 1.0:
                continue
            for f in range(6):
                m = t.face == f
                if len(m) != sel.sum():
                    continue
                m = np.flatnonzero(sel)[m]
                if not len(m):
                    continue
                a = A[m].sum()
                c = (P[m] * A[m, None]).sum(0) / a
                nrm = (N[m] * A[m, None]).sum(0)
                nrm = nrm / max(np.linalg.norm(nrm), 1e-9)
                ext = P[m].max(0) - P[m].min(0)
                soft = (0.5 * float(np.linalg.norm(ext))) ** 2 + 1e-4
                out.append((c, soft, e * a, nrm, float(k)))
        return out[:OBJECT_SOURCES]
