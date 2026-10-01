"""Rigid bodies: objects that fall, tumble, slide, bounce, stack and knock each other over, in every kind of
scene (fire, liquid, both).

MuJoCo (https://mujoco.org, Apache-2.0) integrates them. This module:
- builds MuJoCo's model from the scene:
  - the ground and the box's closed walls;
  - colliders that stay put, as fixed shapes;
  - keyframed colliders, as mocap bodies that move as keyed and push the others;
  - colliders marked to fall (or to float), as free bodies;
- steps it through each frame, ahead of the gas or the liquid, and hands the poses at every substep to the
  simulation as moving colliders (Scene.colliders_gpu's overrides). So falling things push the smoke, the
  water and the cloth, and a burnable one carries its fire with it.

What the gas and the liquid do back to the bodies, added as forces:
- Drag in the air. The gas velocity is sampled at each body once a frame, so a blast blows light things
  away and a strong wind slides a cardboard box.
- In a liquid: buoyancy, the push of the water and drag toward its flow. These come from the forces the
  liquid solver measures on each body every frame (liq_float.wgsl). The still-water part is worked out again
  at every step from the body's pose and the waterline the measured lift implies, as liquid_float.Floats
  did. The water's added mass slows the body's motion through it.

Contacts are MuJoCo's soft constraints, with the materials' friction and restitution
(scene/materials.py). Restitution comes from the contact's damping: a spring of stiffness k per unit mass,
damped by the ratio that keeps the material's share of the speed through a bounce. k is as stiff as the
time step allows.

Orientation in this module is a quaternion (x, y, z, w), as in ColliderGPU. MuJoCo's own is (w, x, y, z).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from ..scene.materials import resolved
from .liquid_float import FloatBody, inertia, q_mul, q_rot, q_yaw

log = logging.getLogger(__name__)

G = 9.81
AIR_DENSITY = 1.2          # kg/m^3
DRAG_COEFF = 1.0           # bluff bodies
MAX_ACCEL = 60.0           # m/s^2: a cap on the liquid's push (a frame of bad pressure)
Z_TO_Y = (0.7071067811865476, -0.7071067811865476, 0.0, 0.0)   # MuJoCo (w,x,y,z): its z axis turned onto y
MIN_DT, MAX_DT = 2.5e-4, 2.0e-3


# The share of the speed MuJoCo's contact gives back through a bounce, for the damping ratio of its spring, at
# the solids' time steps (measured: a ball dropped from 0.3 to 3 m, averaged). Resolved over a few steps it gives
# back more than an ideal spring and damper would, and past a ratio of about 0.85 the contact gains energy, so
# the least bounce there is is about 0.2.
_RATIOS = (0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02)
_BOUNCES = (0.202, 0.229, 0.253, 0.311, 0.377, 0.458, 0.595, 0.750, 0.863, 0.948)


def damping_ratio(e):
    """The damping ratio of a contact spring that keeps a share e of the speed through a bounce."""
    return float(np.interp(float(e), _BOUNCES, _RATIOS))


def wxyz(q):
    x, y, z, w = q
    return [float(w), float(x), float(y), float(z)]


def xyzw(q):
    w, x, y, z = q
    return np.array([x, y, z, w], float)


def _yaw_wxyz(yaw):
    return [math.cos(0.5 * yaw), 0.0, math.sin(0.5 * yaw), 0.0]


def _quat_mul_wxyz(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw]


def shape_volume(shape, size):
    s = np.abs(np.asarray(size, float))
    if shape == 'sphere':
        return 4.0 / 3.0 * math.pi * s[0] ** 3
    if shape == 'cylinder':
        return 2.0 * math.pi * s[0] ** 2 * s[1]
    return 8.0 * float(np.prod(s))


def shape_area(shape, size):
    """Surface area (m^2); a convex body's average projected area is a quarter of it."""
    s = np.abs(np.asarray(size, float))
    if shape == 'sphere':
        return 4.0 * math.pi * s[0] ** 2
    if shape == 'cylinder':
        return 2.0 * math.pi * s[0] * (s[0] + 2.0 * s[1])
    a, b, c = 2.0 * s
    return 2.0 * (a * b + b * c + a * c)


@dataclass
class Body:
    index: int                  # the collider's index in the scene
    shape: str                  # box, sphere, cylinder (meshes are boxes or hulls)
    size: np.ndarray            # the collider's size (half extents; radius; radius, half height; a mesh's scale)
    density: float
    friction: float
    bounce: float
    volume: float
    area: float                 # surface area (m^2), for drag in the air
    hull: np.ndarray = None     # a mesh's vertices in its own frame (m), for its convex hull
    body_id: int = -1           # MuJoCo body id
    qadr: int = -1              # its free joint's qpos address
    vadr: int = -1              # and qvel address
    fb: FloatBody = None        # buoyancy in a liquid (its sample points, waterline)
    hydro: tuple = None         # the liquid's measured push for the next frame: (f_dyn, t_dyn, sub, liquid vel, waterline)
    air: np.ndarray = field(default_factory=lambda: np.zeros(3))   # the gas velocity round it (m/s)
    release: float = None       # scene frame it is let go at (held where its keys put it until then); None: free
    held: bool = False          # held by its keys last step
    throw: np.ndarray = field(default_factory=lambda: np.zeros(3))   # Thrown at (m/s), given when it is let go
    spin: np.ndarray = field(default_factory=lambda: np.zeros(3))    # Spinning at (rad/s, world axes)


MORTAR = 0.3e6          # Pa: mortar in tension, the glue between the bricks of a wall
JOINT_FRICTION = 0.6    # shear a joint holds beyond its cohesion, per newton of compression across it
GAP = 0.001             # m: cut faces are moved in by this for the physics, so glued neighbours do not touch
BREAK_STEPS = 1         # steps in a row a weld must be overloaded to break (an impact lasts only a few)


@dataclass
class PieceSet:
    """A breakable object: its pieces (MuJoCo bodies, each at its centroid, turned as the object is) and the welds
    that glue them to each other (bonds) and, if it stands where it is, to the world (anchors)."""
    index: int                  # the collider
    frac: object                # fracture.Fracture
    names: list                 # its pieces' body names
    size: tuple = ()            # the size it was cut at (fractured's key, for drawing a cached frame)
    hollow: float = 0.0         # and its wall thickness then
    dynamic: bool = False       # it falls (else it stands where it is until it breaks)
    release: float = None       # (falling ones) the scene frame it is let go at
    bodies: np.ndarray = None   # (n,) MuJoCo body ids
    qadr: np.ndarray = None     # (n,) free-joint qpos addresses
    vadr: np.ndarray = None     # (n,) qvel addresses
    welds: list = field(default_factory=list)   # (name, first piece, normal in its frame, area, strength)
    held: bool = False          # held by its keys last step
    throw: np.ndarray = field(default_factory=lambda: np.zeros(3))   # Thrown at (m/s) and Spinning at (rad/s, world)
    spin: np.ndarray = field(default_factory=lambda: np.zeros(3))


_FRACTURES = {}


def breaks(c):
    """Whether collider c is breakable (a mesh cannot be cut yet)."""
    return bool(c.get('breakable')) and c.get('shape') != 'mesh'


def fractured(c, size, hollow=0.0):
    """The pieces a breakable collider is cut into (kept: cutting takes a fraction of a second)."""
    from .fracture import fracture
    key = (c['shape'], tuple(round(float(x), 6) for x in np.abs(np.asarray(size, float))), int(c.get('pieces', 24)),
           c.get('fracture', 'voronoi'), int(c.get('fracture_seed', 0)), round(float(hollow or 0.0), 6))
    f = _FRACTURES.get(key)
    if f is None:
        if len(_FRACTURES) > 32:
            _FRACTURES.clear()
        f = _FRACTURES[key] = fracture(key[0], key[1], key[2], key[3], key[4], hollow=key[5])
    return f


def _box_of(piece):
    """A piece's half extents and centre (its own frame) if it is a box along its axes, else None."""
    n = piece.planes[:, :3]
    if len(n) != 6 or not np.allclose(np.abs(n).max(1), 1.0, atol=1e-9):
        return None
    lo, hi = piece.verts.min(0), piece.verts.max(0)
    return 0.5 * (hi - lo), 0.5 * (hi + lo)


class Solids:
    """The rigid bodies of a simulation, stepped by MuJoCo."""

    def __init__(self):
        self.model = None
        self.data = None
        self.bodies: list[Body] = []
        self.mocap: list[tuple[int, int]] = []   # (collider index, mocap id) of keyframed colliders
        self.key = None            # what the model was built from
        self.time = 0.0            # simulated seconds since the start
        self.started = False
        self.liquid = False        # the scene has a liquid: buoyancy and the water's push
        self.rho = 1000.0
        self.gravity = G
        self.warnings: list[str] = []
        self._last = {}            # overrides at the end of the last frame
        self.sets: list[PieceSet] = []   # breakable objects
        self.substep_pieces = []   # their pieces at each substep of the last frame (advance)
        self.breaks = []           # (time, world point, area) of every bond that broke, for dust and debris
        self._w = None             # the welds' data, flattened (_index_welds)

    # -- set up --------------------------------------------------------------------------------

    @property
    def active(self):
        return bool(self.bodies or self.sets)

    @staticmethod
    def wanted(scene):
        """Indices of the colliders that move by themselves: Falls, or Floats in a liquid."""
        out = []
        for i, c in enumerate(scene.colliders):
            if not c['enabled']:
                continue
            if c.get('dynamic') or breaks(c) or (c.get('floating') and scene.kind in ('liquid', 'both')):
                out.append(i)
        return out

    def configure(self, scene, layout):
        """Match the bodies to the scene. Returns True when the model has to be built again (and the
        simulation restarted). layout: (dims, h, origin) of the simulation grid, for the box's walls."""
        idx = self.wanted(scene)
        if not idx:
            return self.clear()
        key = self._key(scene, idx, layout)
        if key == self.key and self.model is not None:
            return False
        self._build(scene, idx, layout)
        self.key = key
        return True

    def clear(self):
        """No rigid bodies (a scene without any, or a sky). True if there were some."""
        changed = self.model is not None
        self.model = self.data = None
        self.bodies, self.mocap, self.key = [], [], None
        self.sets, self.breaks, self._w = [], [], None
        self.started = False
        self._last = {}
        return changed

    @staticmethod
    def _key(scene, idx, layout):
        import json
        cols = [{k: (list(v) if isinstance(v, tuple) else (str(v) if not isinstance(v, (int, float, bool, str)) else v))
                 for k, v in c.items() if k != 'name'} for c in scene.colliders]
        d = scene.data['domain']
        blob = dict(idx=idx, cols=cols, ground=bool(d['ground']), sides=bool(d['open_sides']), kind=scene.kind,
                    layout=[list(map(float, x)) if hasattr(x, '__len__') else float(x) for x in layout],
                    g=float(scene.data['liquid']['gravity']) if scene.kind in ('liquid', 'both') else G)
        return json.dumps(blob, sort_keys=True, default=str)

    def _mesh_points(self, scene, c):
        """A mesh collider's vertices in its own frame (m): the file's, scaled by its Size."""
        from .mesh import MeshError, load_mesh
        try:
            v, _t = load_mesh(scene.mesh_path(c['mesh']))
        except (MeshError, OSError, ValueError) as ex:
            self.warnings.append(f'{c["name"]}: {ex}')
            return None
        v = np.asarray(v, float) * np.abs(np.asarray(c['size'], float))
        if len(v) > 4000:   # the hull only needs its outline
            v = v[np.random.default_rng(0).choice(len(v), 4000, replace=False)]
        return v

    def _build_pieces(self, scene, spec, w, idx, by_index, contact, fixed_geom, k):
        """Breakable colliders as their pieces (free bodies) glued by welds: one per bond, and, for one that stands
        where it is, anchors to the world along its base or its edges. Returns the PieceSets."""
        import mujoco
        from .fracture import inset
        sets = []
        for i in idx:
            c = scene.colliders[i]
            cg = by_index.get(i)
            if cg is None or not breaks(c):
                continue
            frac = fractured(c, cg.size, cg.hollow)
            if not frac.pieces:
                continue
            r = resolved(c)
            strength = (MORTAR if c.get('fracture') == 'bricks' else r['strength']) * float(c.get('strength', 1.0))
            q0 = _quat_mul_wxyz(wxyz(cg.quat), _yaw_wxyz(float(cg.rot_y)))
            R0 = q_rot(xyzw(q0))
            dynamic = bool(c.get('dynamic'))
            ps = PieceSet(index=i, frac=frac, names=[], size=tuple(float(x) for x in cg.size), hollow=float(cg.hollow),
                          throw=np.asarray(c.get('start_velocity', (0.0, 0.0, 0.0)), float) if dynamic else np.zeros(3),
                          spin=np.radians(np.asarray(c.get('start_spin', (0.0, 0.0, 0.0)), float)) if dynamic else np.zeros(3),
                          dynamic=dynamic,
                          release=scene.start + float(c.get('release', 0.0)) * scene.fps if dynamic else None)
            for n, p in enumerate(frac.pieces):
                b = w.add_body()
                b.name = f'piece{i}_{n}'
                b.pos = list(map(float, np.asarray(cg.pos, float) + R0 @ p.centroid))
                b.quat = list(q0)
                b.add_freejoint()
                v = inset(p, GAP) - p.centroid
                box = _box_of(p)
                if box is not None:
                    lo, hi = v.min(0), v.max(0)
                    g = fixed_geom(b, 'box', 0.5 * (hi - lo), None)
                    g.pos = list(map(float, 0.5 * (lo + hi)))
                else:
                    ma = spec.add_mesh()
                    ma.name = f'piece{i}_{n}'
                    ma.uservert = v.ravel().tolist()
                    g = fixed_geom(b, 'mesh', None, ma.name)
                g.density = float(r['density'])
                g.priority = 1
                contact(g, r['friction'], r['bounce'])
                ps.names.append(b.name)
            # the bonds: welds between pieces that share a cut
            stiff = [-k, -2.0 * math.sqrt(k)]
            for n, bond in enumerate(frac.bonds):
                e = spec.add_equality()
                e.type = mujoco.mjtEq.mjEQ_WELD
                e.objtype = mujoco.mjtObj.mjOBJ_BODY
                e.name = f'bond{i}_{n}'
                e.name1, e.name2 = ps.names[bond.i], ps.names[bond.j]
                e.solref = stiff
                # anchored on the shared face (in the second piece's frame), so the weld's torque is the moment there
                data = np.array(e.data, float)
                data[0:3] = np.asarray(bond.centre, float) - frac.pieces[bond.j].centroid
                e.data = data
                ps.welds.append((e.name, bond.i, np.asarray(bond.normal, float), float(bond.area), strength))
            # standing where it is: glued to the world along its base (or its base and sides)
            held = c.get('held', 'base')
            if not dynamic and held != 'free':
                # (its edges: the faces round its rim, not the broad faces of a pane or a wall, and not its top)
                sz = np.abs(np.asarray(cg.size, float))
                thin = np.eye(3)[int(np.argmin(sz))] if c['shape'] == 'box' else np.zeros(3)
                for n, p in enumerate(frac.pieces):
                    outer = ~p.inner
                    ny = p.planes[:, 1]
                    if held == 'base' or c['shape'] != 'box':
                        face = outer & (ny < -0.9)
                    else:
                        face = outer & (ny < 0.1) & (np.abs(p.planes[:, :3] @ thin) < 0.5)
                    area = float(p.face_area[face].sum()) if face.any() else 0.0
                    if area <= 0.0:
                        continue
                    nrm = (p.planes[face, :3] * p.face_area[face, None]).sum(0)
                    nrm /= max(float(np.linalg.norm(nrm)), 1e-12)
                    fc = (p.face_centre[face] * p.face_area[face, None]).sum(0) / area
                    e = spec.add_equality()
                    e.type = mujoco.mjtEq.mjEQ_WELD
                    e.objtype = mujoco.mjtObj.mjOBJ_BODY
                    e.name = f'anchor{i}_{n}'
                    e.name1 = ps.names[n]
                    e.solref = stiff
                    data = np.array(e.data, float)
                    data[0:3] = np.asarray(cg.pos, float) + R0 @ fc     # (the world's frame: where its glued face is)
                    e.data = data
                    ps.welds.append((e.name, n, nrm, area, strength))
            sets.append(ps)
            if len(frac.pieces) > 150:
                self.warnings.append(f'{c["name"]}: {len(frac.pieces)} pieces take a while to simulate')
        return sets

    def _index_welds(self):
        """The welds of every breakable, flattened into arrays for the per-step check (_break)."""
        import mujoco
        m = self.model
        rows = []
        for ps in self.sets:
            for name, first, nrm, area, strength in ps.welds:
                rows.append((mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, name), int(ps.bodies[first]), nrm, area, strength,
                             ps.index))
        if not rows:
            self._w = None
            return
        eq = np.array([r[0] for r in rows], np.int64)
        lookup = np.full(max(int(m.neq), 1), -1, np.int64)
        lookup[eq] = np.arange(len(rows))
        self._w = dict(eq=eq, body=np.array([r[1] for r in rows], np.int64), normal=np.array([r[2] for r in rows], float),
                       area=np.array([r[3] for r in rows], float), strength=np.array([r[4] for r in rows], float),
                       collider=np.array([r[5] for r in rows], np.int64), lookup=lookup, over=np.zeros(len(rows), np.int64))

    def _break(self):
        """Break the welds whose joint is overloaded (BREAK_STEPS steps running): pulled apart harder than its strength over
        its area, sheared harder than that plus the friction of what presses it together, or bent past what its
        section holds. Records where each broke (self.breaks)."""
        import mujoco
        W = self._w
        if W is None:
            return
        d = self.data
        n = d.nefc
        if n == 0:
            W['over'][:] = 0
            return
        eqm = d.efc_type[:n] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        if not eqm.any():
            return
        F = d.efc_force[:n][eqm].reshape(-1, 6)
        E = d.efc_id[:n][eqm].reshape(-1, 6)[:, 0]
        rows = W['lookup'][E]
        ok = rows >= 0
        rows, F = rows[ok], F[ok]
        if len(rows) == 0:
            return
        R = d.xmat[W['body'][rows]].reshape(-1, 3, 3)
        nw = np.einsum('bij,bj->bi', R, W['normal'][rows])
        f = F[:, :3]
        fn = (f * nw).sum(1)                                   # + pulls it apart (on the first piece, toward the other)
        fs = np.linalg.norm(f - fn[:, None] * nw, axis=1)
        mt = np.linalg.norm(F[:, 3:], axis=1)
        A = W['area'][rows]
        sig = W['strength'][rows]
        over = (fn > sig * A) | (fs > sig * A + JOINT_FRICTION * np.maximum(-fn, 0.0)) | (mt > sig * A * np.sqrt(A) / 6.0)
        hit = np.zeros(len(W['over']), bool)
        hit[rows[over]] = True
        W['over'][hit] += 1
        W['over'][~hit] = 0
        broke = np.nonzero(W['over'] >= BREAK_STEPS)[0]
        if len(broke):
            d.eq_active[W['eq'][broke]] = 0
            W['over'][broke] = -(1 << 40)       # never again
            for b in broke:
                self.breaks.append((self.time, d.xpos[W['body'][b]].copy(), float(W['area'][b]), int(W['collider'][b])))

    def dust(self, scene, fdt, substeps, last=0.2, most=6):
        """Puffs of dust where things broke in the frame just stepped (fdt seconds, in `substeps` substeps): for each
        substep, sources of smoke (EmitterGPU) where bonds broke within the last `last` seconds, gathered into
        30 cm clusters, the `most` biggest. Each throws up dust for as long, as much as the broken faces' area and
        the material's dustiness give."""
        from ..scene.materials import material
        from .solver import EmitterGPU
        if not self.breaks:
            return None
        t1 = self.time
        t0 = t1 - fdt
        recent = [b for b in self.breaks if b[0] > t0 - last]
        if not recent:
            return None
        out = []
        for i in range(substeps):
            ti = t0 + (i + 0.5) * fdt / substeps
            cells = {}
            for tb, pos, area, ci in recent:
                if not (tb <= ti < tb + last) or ci >= len(scene.colliders):
                    continue
                dust = material(scene.colliders[ci].get('material', 'wood')).dust
                if dust <= 0.0:
                    continue
                key = tuple(np.floor(np.asarray(pos) / 0.3).astype(int))
                a, w, p = cells.get(key, (0.0, 0.0, np.zeros(3)))
                cells[key] = (a + area * dust, w + area, p + area * np.asarray(pos))
            puffs = sorted(cells.values(), key=lambda c: -c[0])[:most]
            ems = []
            for k, (amount, w, p) in enumerate(puffs):
                c = p / max(w, 1e-12)
                r = float(np.clip(0.5 * np.sqrt(w) + 0.06, 0.06, 0.5))
                ems.append(EmitterGPU(shape='sphere', pos=tuple(float(x) for x in c), size=(r, r, r), fuel=0.0, temp=0.0,
                                      smoke=float(np.clip(40.0 * amount, 0.5, 8.0)), vel=(0.0, 0.4, 0.0), radial=1.2,
                                      vel_blend=0.2, noise=0.8, noise_freq=6.0, seed=float(k * 17 + i)))
            out.append(ems)
        return out if any(out) else None

    def piece_poses(self):
        """Every breakable's pieces as they are now: {collider index: dict(pos (n, 3), quat (n, 4) x y z w,
        vel (n, 3), omega (n, 3) world, size: the size it was cut at)}."""
        d = self.data
        out = {}
        for ps in self.sets:
            q = d.xquat[ps.bodies]
            rot = d.xmat[ps.bodies].reshape(-1, 3, 3)
            w_local = np.stack([d.qvel[a + 3:a + 6] for a in ps.vadr]) if len(ps.vadr) else np.zeros((0, 3))
            out[ps.index] = dict(pos=d.xpos[ps.bodies].copy(), quat=np.concatenate([q[:, 1:], q[:, :1]], 1),
                                 vel=np.stack([d.qvel[a:a + 3] for a in ps.vadr]) if len(ps.vadr) else np.zeros((0, 3)),
                                 omega=np.einsum('bij,bj->bi', rot, w_local), size=np.asarray(ps.size, float),
                                 hollow=np.float32(ps.hollow))
        return out

    def _mesh_boxes(self, scene, c, most=600):
        """A fixed mesh collider as boxes filling its inside, in its own frame (m): [(half extents, centre)], or
        None (an open mesh, or one too intricate for `most` boxes)."""
        from .mesh import MeshError, load_mesh
        try:
            v, t = load_mesh(scene.mesh_path(c['mesh']))
        except (MeshError, OSError, ValueError):
            return None
        v = np.asarray(v, float) * np.abs(np.asarray(c['size'], float))
        t = np.asarray(t, np.int64).reshape(-1, 3)
        if len(t) == 0:
            return None
        lo, hi = v.min(0), v.max(0)
        for n in (48, 32, 24, 16):
            cell = float(np.max(hi - lo)) / n
            if cell <= 0.0:
                return None
            occ = mesh_occupancy(v, t, lo, hi, cell)
            if occ is None or not occ.any():
                return None
            boxes = greedy_boxes(occ)
            if len(boxes) <= most:
                return [(0.5 * cell * (np.asarray(b[3:]) - np.asarray(b[:3]) + 1.0),
                         lo + cell * 0.5 * (np.asarray(b[:3]) + np.asarray(b[3:]) + 1.0)) for b in boxes]
        return None

    def _build(self, scene, idx, layout):
        import mujoco
        self.warnings = []
        dims, h, origin = layout
        o = np.asarray(origin, float)
        top = o + np.asarray(dims, float) * float(h)
        self.box = (o, top)
        self.open_sides = bool(scene.data['domain']['open_sides'])
        self.liquid = scene.kind in ('liquid', 'both')
        q = scene.data['liquid']
        self.gravity = float(q['gravity']) if self.liquid else G
        self.rho = 1000.0
        d = scene.data['domain']
        spec = mujoco.MjSpec()
        spec.option.gravity = [0.0, -self.gravity, 0.0]
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.noslip_iterations = 2
        spec.option.iterations = 30
        w = spec.worldbody
        start = scene.start - int(round(d['preroll'] * scene.fps)) - 1
        cols = scene.colliders_gpu(start)
        enabled = [i for i, c in enumerate(scene.colliders) if c['enabled']]
        by_index = {i: cg for i, cg in zip(enabled, cols)}

        # the bodies, and the time step their size allows
        bodies = []
        smallest = 1.0
        for i in idx:
            c = scene.colliders[i]
            cg = by_index.get(i)
            if cg is None or breaks(c):
                continue
            r = resolved(c)
            if c.get('floating') and not c.get('dynamic') and float(c.get('density', 0.0) or 0.0) > 0.0:
                r['density'] = float(c['density'])
            shape = c['shape']
            size = np.abs(np.asarray(c['size'], float))
            hull = None
            if shape == 'mesh':
                hull = self._mesh_points(scene, c)
                if hull is None or len(hull) < 4:
                    continue
                lo, hi = hull.min(0), hull.max(0)
                vol = float(np.prod(hi - lo)) * 0.6
                area = shape_area('box', 0.5 * (hi - lo))
                smallest = min(smallest, float(np.min(hi - lo)) * 0.5)
            else:
                vol = shape_volume(shape, size)
                area = shape_area(shape, size)
                smallest = min(smallest, float(size[0]) if shape in ('sphere', 'cylinder') else float(size.min()))
            # objects that fall are held until their release (by default the shot's first frame, so the pre-roll
            # that gets a fire going does not drop them); floating objects of older scenes are free from the start
            release = scene.start + float(c.get('release', 0.0)) * scene.fps if c.get('dynamic') else None
            bodies.append(Body(index=i, shape=shape, size=size, density=r['density'], friction=r['friction'],
                               bounce=r['bounce'], volume=max(vol, 1e-9), area=area, hull=hull, release=release))
        for i in idx:
            c = scene.colliders[i]
            if breaks(c) and i in by_index:
                fr = fractured(c, by_index[i].size, by_index[i].hollow)
                for pc in fr.pieces:
                    # (a thin piece, a shard of a pane, can take a longer step than a small solid of that size)
                    ext = pc.verts.max(0) - pc.verts.min(0)
                    smallest = min(smallest, 0.5 * float(np.sort(ext)[1]), 2.0 * float(ext.min()))
        dt = min(max(0.1 * smallest, MIN_DT), MAX_DT)
        spec.option.timestep = dt
        k = (0.6 / dt) ** 2   # as stiff as the step allows (about 0.6 radian of the contact's spring per step)

        def contact(g, friction, bounce, roll=False):
            g.friction = [float(friction), 0.005, 0.0002 if roll else 0.0001]
            g.solref = [-k, -2.0 * damping_ratio(bounce) * math.sqrt(k)]
            g.condim = 6 if roll else 3

        def fixed_geom(parent, shape, size, mesh=None):
            g = parent.add_geom()
            if shape == 'sphere':
                g.type = mujoco.mjtGeom.mjGEOM_SPHERE
                g.size = [float(size[0]), 0.0, 0.0]
            elif shape == 'cylinder':
                g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
                g.size = [float(size[0]), float(size[1]), 0.0]
                g.quat = list(Z_TO_Y)
            elif mesh is not None:
                g.type = mujoco.mjtGeom.mjGEOM_MESH
                g.meshname = mesh
            else:
                g.type = mujoco.mjtGeom.mjGEOM_BOX
                g.size = [float(max(size[0], 1e-4)), float(max(size[1], 1e-4)), float(max(size[2], 1e-4))]
            return g

        # The ground and the box's closed walls. Fixed and keyframed things have priority 0 and free bodies 1,
        # so in a contact between them the body's material decides the friction and the bounce. Between two
        # bodies, MuJoCo takes the larger friction and averages the bounce.
        planes = []
        if d['ground']:
            planes.append(((0.0, float(o[1]), 0.0), (0.0, 1.0, 0.0)))
        if not d['open_sides']:
            planes += [((float(o[0]), 0.0, 0.0), (1.0, 0.0, 0.0)), ((float(top[0]), 0.0, 0.0), (-1.0, 0.0, 0.0)),
                       ((0.0, 0.0, float(o[2])), (0.0, 0.0, 1.0)), ((0.0, 0.0, float(top[2])), (0.0, 0.0, -1.0))]
        for pos, nrm in planes:
            g = w.add_geom()
            g.type = mujoco.mjtGeom.mjGEOM_PLANE
            g.size = [0.0, 0.0, 1.0]
            g.pos = list(pos)
            g.quat = self._z_onto(nrm)
            contact(g, 0.6, 0.2)

        # the other colliders: fixed shapes, or mocap bodies that follow their keys
        moving = set()
        for i, c in enumerate(scene.colliders):
            if i in idx or not c['enabled']:
                continue
            if any(scene.curve(('collider', i, kk)) is not None for kk in ('position', 'yaw', 'size')):
                moving.add(i)
        mocap = []
        n_mesh = 0
        for i in enabled:
            if i in idx:
                continue
            c, cg = scene.colliders[i], by_index[i]
            size = np.abs(np.asarray(cg.size, float))
            r = resolved(c)
            yaw = float(cg.rot_y)
            # its shapes in its own frame: (shape, size, centre, mesh asset)
            parts = []
            if cg.hollow > 0.0 and cg.shape == 'box':
                parts = [('box', hs, p, None) for p, hs in self._hollow_slabs(size, float(cg.hollow), np.asarray(cg.opening, float),
                                                                            np.asarray(cg.opening_at, float))]
            elif cg.hollow > 0.0:
                continue   # a hollow sphere or cylinder: only its walls for the fluids matter
            elif cg.shape == 'mesh':
                src = scene.mesh_path(c['mesh'])
                if src.split('#')[0].lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.exr', '.bmp')) and i not in moving:
                    self._add_hfield(spec, w, src, size, np.asarray(cg.pos, float), yaw, contact, r)
                    continue
                # boxes filling its inside, so things rest in its hollows (between logs, on stairs)
                boxes = self._mesh_boxes(scene, c)
                if boxes:
                    parts = [('box', hs, p, None) for hs, p in boxes]
                else:
                    pts = self._mesh_points(scene, c)
                    if pts is None or len(pts) < 4:
                        continue
                    name = f'fixed{n_mesh}'
                    n_mesh += 1
                    ma = spec.add_mesh()
                    ma.name = name
                    ma.uservert = pts.ravel().tolist()
                    parts = [('mesh', size, np.zeros(3), name)]
                    self.warnings.append(f'{c["name"]}: things collide with its convex hull (its outline with the hollows filled in)')
            else:
                parts = [(cg.shape, size, np.zeros(3), None)]
            if i in moving:
                parent = w.add_body()
                parent.mocap = True
                parent.pos = list(map(float, cg.pos))
                parent.quat = _yaw_wxyz(yaw)
                mocap.append(i)
            for shape, hs, centre, mesh in parts:
                g = fixed_geom(w if i not in moving else parent, shape, hs, mesh)
                if i in moving:
                    g.pos = list(map(float, centre))
                else:
                    g.pos = list(map(float, self._yaw_vec(yaw, centre) + np.asarray(cg.pos, float)))
                    g.quat = _quat_mul_wxyz(_yaw_wxyz(yaw), list(g.quat))
                contact(g, r['friction'], r['bounce'])

        # the free bodies
        for n, bd in enumerate(bodies):
            cg = by_index[bd.index]
            b = w.add_body()
            b.name = f'body{n}'
            b.pos = list(map(float, cg.pos))
            b.quat = _quat_mul_wxyz(wxyz(cg.quat), _yaw_wxyz(float(cg.rot_y)))
            b.add_freejoint()
            mesh = None
            if bd.shape == 'mesh':
                mesh = f'body{n}'
                ma = spec.add_mesh()
                ma.name = mesh
                ma.uservert = bd.hull.ravel().tolist()
            g = fixed_geom(b, bd.shape, bd.size, mesh)
            g.density = float(bd.density)
            g.priority = 1
            contact(g, bd.friction, bd.bounce, roll=bd.shape in ('sphere', 'cylinder'))
            bd.fb = self._float_body(bd, cg)
        sets = self._build_pieces(scene, spec, w, idx, by_index, contact, fixed_geom, k)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        for n, bd in enumerate(bodies):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'body{n}')
            bd.body_id = bid
            j = m.body_jntadr[bid]
            bd.qadr, bd.vadr = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
            bd.volume = max(float(m.body_mass[bid]) / max(bd.density, 1e-6), 1e-9)
            if bd.fb is not None:
                bd.fb.volume = bd.volume
        # mocap ids follow the order the mocap bodies were added
        self.mocap = [(i, n) for n, i in enumerate(mocap)]
        self.bodies = bodies
        for ps in sets:
            ps.bodies = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm) for nm in ps.names], np.int64)
            jnt = m.body_jntadr[ps.bodies]
            ps.qadr = np.asarray(m.jnt_qposadr[jnt], np.int64)
            ps.vadr = np.asarray(m.jnt_dofadr[jnt], np.int64)
            if ps.dynamic:
                self._spin_set(ps, self.data, np.asarray(by_index[ps.index].vel, float) + ps.throw, ps.spin)
        self.sets = sets
        self.breaks = []
        self._index_welds()
        # starting motion
        d = self.data
        for bd in bodies:
            c = scene.colliders[bd.index]
            v0 = np.asarray(c.get('start_velocity', (0.0, 0.0, 0.0)), float)
            w0 = np.radians(np.asarray(c.get('start_spin', (0.0, 0.0, 0.0)), float))
            bd.throw, bd.spin = v0.copy(), w0.copy()
            cg = by_index[bd.index]
            v0 = v0 + np.asarray(cg.vel, float)
            d.qvel[bd.vadr:bd.vadr + 3] = v0
            # free joint angular velocity is in the body's own frame
            R = q_rot(xyzw(d.qpos[bd.qadr + 3:bd.qadr + 7]))
            d.qvel[bd.vadr + 3:bd.vadr + 6] = R.T @ w0
        mujoco.mj_forward(m, d)
        # things that start inside each other are thrown apart when they are let go
        owner = {int(m.geom_bodyid[g]): bd for bd in bodies for g in range(m.ngeom) if int(m.geom_bodyid[g]) == bd.body_id}
        seen = set()
        for k in range(d.ncon):
            con = d.contact[k]
            a, b = owner.get(int(m.geom_bodyid[con.geom1])), owner.get(int(m.geom_bodyid[con.geom2]))
            if a is None or b is None or a is b or con.dist > -0.02 * float(min(a.size.min(), b.size.min())) - 1e-3:
                continue
            pair = tuple(sorted((a.index, b.index)))
            if pair not in seen:
                seen.add(pair)
                names = [scene.colliders[i]['name'] for i in pair]
                self.warnings.append(f'{names[0]} and {names[1]} start inside each other: they will fly apart when let go.')
        for w_ in self.warnings:
            log.warning('Rigid bodies: %s', w_)
        self._init = (d.qpos.copy(), d.qvel.copy())
        self.time = 0.0
        self.started = False
        self._last = self._poses()
        log.info('Rigid bodies: %d free, %d keyframed, step %.2f ms', len(bodies), len(self.mocap), dt * 1e3)

    @staticmethod
    def _z_onto(n):
        """A MuJoCo quaternion (w,x,y,z) turning +z onto the unit vector n."""
        n = np.asarray(n, float)
        z = np.array([0.0, 0.0, 1.0])
        c = float(z @ n)
        if c > 1 - 1e-9:
            return [1.0, 0.0, 0.0, 0.0]
        if c < -1 + 1e-9:
            return [0.0, 1.0, 0.0, 0.0]
        ax = np.cross(z, n)
        ax /= np.linalg.norm(ax)
        a = math.acos(c)
        return [math.cos(0.5 * a), *(ax * math.sin(0.5 * a))]

    @staticmethod
    def _yaw_vec(yaw, v):
        R = q_rot(q_yaw(yaw))
        return R @ np.asarray(v, float)

    @staticmethod
    def _hollow_slabs(size, t, opening, at):
        """The six walls of a hollow box (half sizes `size`, walls t thick) as boxes (centre, half size) in its
        own frame, the wall the opening is in split around the opening."""
        sx, sy, sz = size
        t = min(t, sx, sy, sz)
        walls = [((sx - t / 2, 0, 0), (t / 2, sy, sz), 0), ((-sx + t / 2, 0, 0), (t / 2, sy, sz), 0),
                 ((0, sy - t / 2, 0), (sx, t / 2, sz), 1), ((0, -sy + t / 2, 0), (sx, t / 2, sz), 1),
                 ((0, 0, sz - t / 2), (sx, sy, t / 2), 2), ((0, 0, -sz + t / 2), (sx, sy, t / 2), 2)]
        out = []
        has = np.all(np.asarray(opening) > 1e-6)
        for c, hs, ax in walls:
            c, hs = np.array(c, float), np.array(hs, float)
            if not has:
                out.append((c, hs))
                continue
            olo, ohi = np.asarray(at) - opening, np.asarray(at) + opening
            wlo, whi = c - hs, c + hs
            if np.any(ohi <= wlo) or np.any(olo >= whi):
                out.append((c, hs))
                continue
            # split along the two axes in the wall's plane: the parts left, right, below and above the hole
            u, v = [a for a in range(3) if a != ax]
            pieces = []
            lo, hi = wlo.copy(), whi.copy()
            if olo[u] > wlo[u]:
                a, b = wlo.copy(), whi.copy()
                b[u] = olo[u]
                pieces.append((a, b))
                lo[u] = olo[u]
            if ohi[u] < whi[u]:
                a, b = wlo.copy(), whi.copy()
                a[u] = ohi[u]
                pieces.append((a, b))
                hi[u] = ohi[u]
            if olo[v] > wlo[v]:
                a, b = lo.copy(), hi.copy()
                b[v] = olo[v]
                pieces.append((a, b))
            if ohi[v] < whi[v]:
                a, b = lo.copy(), hi.copy()
                a[v] = ohi[v]
                pieces.append((a, b))
            for a, b in pieces:
                if np.all(b - a > 1e-5):
                    out.append(((a + b) / 2, (b - a) / 2))
        return out

    def _add_hfield(self, spec, w, src, size, pos, yaw, contact, r):
        """A heightfield image as MuJoCo terrain: the image spans x and z of the collider's Size, from its top
        edge (back, -z) to its bottom; black is 0.02 of its height above the base and white 1.02 (mesh.py)."""
        import mujoco
        from .mesh import _read_grey
        img = _read_grey(src)
        step = max(1, int(math.ceil(max(img.shape) / 256)))
        if step > 1:
            hh, ww = (img.shape[0] // step) * step, (img.shape[1] // step) * step
            img = img[:hh, :ww].reshape(hh // step, step, ww // step, step).mean(axis=(1, 3))
        hts = (0.02 + img) / 1.02
        nrow, ncol = hts.shape
        hf = spec.add_hfield()
        hf.name = f'terrain{len(spec.hfields)}'
        hf.nrow, hf.ncol = nrow, ncol
        hf.size = [0.5 * float(size[0]), 0.5 * float(size[2]), 1.02 * float(size[1]), max(0.05, 0.1 * float(size[1]))]
        # the field's rows run along its own y, which the turn below puts along -z: so the image's last row
        # (the front, +z) comes first
        hf.userdata = hts[::-1, :].astype(float).ravel().tolist()
        b = w.add_body()
        b.pos = list(map(float, pos))
        b.quat = _quat_mul_wxyz(_yaw_wxyz(yaw), list(Z_TO_Y))
        g = b.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_HFIELD
        g.hfieldname = hf.name
        contact(g, r['friction'], r['bounce'])

    def _float_body(self, bd, cg):
        """The buoyancy helper for a body in a liquid (its own sample points through its volume)."""
        if bd.shape == 'mesh':
            lo, hi = bd.hull.min(0), bd.hull.max(0)
            size, shape = 0.5 * (hi - lo), 'box'
        elif bd.shape == 'sphere':
            size, shape = np.array([bd.size[0]] * 3), 'sphere'
        elif bd.shape == 'cylinder':
            size, shape = np.array([bd.size[0], bd.size[1], bd.size[0]]), 'cylinder'
        else:
            size, shape = bd.size, 'box'
        return FloatBody(index=bd.index, density=bd.density, shape=shape, size=np.asarray(size, float),
                         pos=np.asarray(cg.pos, float), vel=np.zeros(3), quat=np.array([0.0, 0.0, 0.0, 1.0]),
                         omega=np.zeros(3), volume=bd.volume, area=max(4.0 * size[0] * size[2], 1e-9),
                         radius=float(np.linalg.norm(size)))

    def reset(self):
        """Back to where everything starts (as built from the scene)."""
        import mujoco
        self.started = False
        self.time = 0.0
        if self.data is None:
            return
        q, v = self._init
        self.data.qpos[:] = q
        self.data.qvel[:] = v
        self.data.xfrc_applied[:] = 0.0
        self.data.eq_active[:] = self.model.eq_active0
        self.breaks = []
        if self._w is not None:
            self._w['over'][:] = 0
        for ps in self.sets:
            ps.held = False
        mujoco.mj_forward(self.model, self.data)
        for bd in self.bodies:
            bd.hydro = None
            bd.air = np.zeros(3)
            bd.held = False
        self._last = self._poses()

    # -- stepping -------------------------------------------------------------------------------

    def _keyed(self, scene, frame):
        """Moves this step from the keys: keyframed colliders (mocap bodies) and bodies not yet let go, which
        are held where their keys put them, moving at the keys' speed."""
        held = [bd for bd in self.bodies if bd.release is not None and frame < bd.release]
        d = self.data
        self._hold_sets(scene, frame)
        for bd in self.bodies:
            if bd.held and bd not in held:
                # let go: it carries on at its keys' speed, plus the throw and spin it was given
                d.qvel[bd.vadr:bd.vadr + 3] += bd.throw
                R = q_rot(xyzw(d.qpos[bd.qadr + 3:bd.qadr + 7]))
                d.qvel[bd.vadr + 3:bd.vadr + 6] += R.T @ bd.spin
            bd.held = bd in held
        if not self.mocap and not held:
            return
        cols = scene.colliders_gpu(frame)
        enabled = [i for i, c in enumerate(scene.colliders) if c['enabled']]
        at = {i: cg for i, cg in zip(enabled, cols)}
        for i, mid in self.mocap:
            cg = at.get(i)
            if cg is None:
                continue
            d.mocap_pos[mid] = cg.pos
            d.mocap_quat[mid] = _yaw_wxyz(float(cg.rot_y))
        for bd in held:
            cg = at.get(bd.index)
            if cg is None:
                continue
            q = _quat_mul_wxyz(wxyz(cg.quat), _yaw_wxyz(float(cg.rot_y)))
            d.qpos[bd.qadr:bd.qadr + 3] = cg.pos
            d.qpos[bd.qadr + 3:bd.qadr + 7] = q
            d.qvel[bd.vadr:bd.vadr + 3] = cg.vel
            R = q_rot(xyzw(q))
            d.qvel[bd.vadr + 3:bd.vadr + 6] = R.T @ np.array([0.0, float(cg.spin), 0.0])

    def _spin_set(self, ps, d, vel, spin):
        """Give a breakable's pieces the motion of one rigid thing: moving at vel, spinning at spin (world) round its
        middle."""
        pos = np.stack([d.qpos[a:a + 3] for a in ps.qadr])
        mid = pos.mean(0)
        for n, (a, v) in enumerate(zip(ps.qadr, ps.vadr)):
            d.qvel[v:v + 3] = vel + np.cross(spin, pos[n] - mid)
            R = q_rot(xyzw(d.qpos[a + 3:a + 7]))
            d.qvel[v + 3:v + 6] = R.T @ spin

    def _hold_sets(self, scene, frame):
        """Breakables that fall are held whole where their keys put them until they are let go (then thrown)."""
        hold = [ps for ps in self.sets if ps.release is not None and frame < ps.release]
        for ps in self.sets:
            if ps.held and ps not in hold:
                v0 = np.mean(np.stack([self.data.qvel[v:v + 3] for v in ps.vadr]), axis=0)
                self._spin_set(ps, self.data, v0 + ps.throw, ps.spin)
            ps.held = ps in hold
        if not hold:
            return
        cols = scene.colliders_gpu(frame)
        enabled = [i for i, c in enumerate(scene.colliders) if c['enabled']]
        at = {i: cg for i, cg in zip(enabled, cols)}
        d = self.data
        for ps in hold:
            cg = at.get(ps.index)
            if cg is None:
                continue
            q = _quat_mul_wxyz(wxyz(cg.quat), _yaw_wxyz(float(cg.rot_y)))
            R = q_rot(xyzw(q))
            for n, p in enumerate(ps.frac.pieces):
                a, v = int(ps.qadr[n]), int(ps.vadr[n])
                d.qpos[a:a + 3] = np.asarray(cg.pos, float) + R @ p.centroid
                d.qpos[a + 3:a + 7] = q
                d.qvel[v:v + 3] = cg.vel
                d.qvel[v + 3:v + 6] = 0.0

    def _keep_in_box(self):
        """In a liquid box with open sides, keep each body's centre over the box: it only feels the liquid
        inside (as liquid_float.Floats did)."""
        if not (self.liquid and self.open_sides):
            return
        o, top = self.box
        d = self.data
        for bd in self.bodies:
            r = float(np.linalg.norm(bd.fb.size)) if bd.fb is not None else float(np.linalg.norm(bd.size))
            for ax in (0, 2):
                lo, hi = o[ax] + r, top[ax] - r
                x = d.qpos[bd.qadr + ax]
                if lo < hi and not (lo <= x <= hi):
                    d.qpos[bd.qadr + ax] = min(max(x, lo), hi)
                    d.qvel[bd.vadr + ax] *= 0.5

    def _forces(self):
        """Air drag and the liquid's push on every body, as MuJoCo's applied forces (world, at the centre of mass)."""
        import mujoco
        m, d = self.model, self.data
        d.xfrc_applied[:] = 0.0
        vel6 = np.zeros(6)
        for bd in self.bodies:
            bid = bd.body_id
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, vel6, 0)
            omega, v = vel6[:3].copy(), vel6[3:].copy()
            mass = float(m.body_mass[bid])
            f = np.zeros(3)
            t = np.zeros(3)
            # the air's drag (quadratic), toward the gas velocity
            rel = bd.air - v
            sp = float(np.linalg.norm(rel))
            if sp > 1e-3:
                f += 0.5 * AIR_DENSITY * DRAG_COEFF * (0.25 * bd.area) * sp * rel
            if self.liquid and bd.hydro is not None and bd.fb is not None:
                f_dyn, t_dyn, sub, vl, wl = bd.hydro
                fb = bd.fb
                fb.pos = d.xipos[bid].copy()
                fb.quat = xyzw(d.xquat[bid])
                f_hyd, t_hyd = fb.buoyancy(wl, self.rho)
                m_tot = mass + 0.5 * self.rho * bd.volume * sub
                kk = self.rho * self.gravity * fb.area if 0.0 < sub < 1.0 else 0.0
                c = 2.0 * 0.35 * math.sqrt(kk * m_tot) if kk > 0 else 0.0
                drag = 1.5 * sub + 0.05
                grav = np.array([0.0, -mass * self.gravity, 0.0])
                F = f_hyd + f_dyn + grav
                F[1] -= c * v[1]
                a = F / m_tot + drag * (vl - v)
                an = float(np.linalg.norm(a))
                if an > MAX_ACCEL:
                    a *= MAX_ACCEL / an
                f += mass * a - grav          # MuJoCo adds the weight itself
                # turning: the measured moment, damped by the water it has to push round
                R = q_rot(fb.quat)
                I_w = R @ np.diag(inertia(fb.shape, fb.size, mass)) @ R.T
                t += (mass / m_tot) * (t_hyd + 0.5 * t_dyn) - (0.3 + 6.0 * sub) * (I_w @ omega)
            d.xfrc_applied[bid, :3] = f
            d.xfrc_applied[bid, 3:] = t

    def advance(self, scene, frame, fdt, substeps):
        """Move every body through frame `frame` (fdt seconds of simulation). Returns, for each of the
        frame's `substeps` substeps, the colliders' overrides at the middle of it."""
        import mujoco
        if self.model is None:
            return [None] * substeps
        m, d = self.model, self.data
        dt = m.opt.timestep
        steps = max(1, int(math.ceil(fdt / dt - 1e-9)))
        h = fdt / steps
        m.opt.timestep = h
        marks = [(i + 0.5) / substeps * fdt for i in range(substeps)]
        out = []
        self.substep_pieces = []   # the broken pieces at the middle of each substep (bodyfield.py)
        mi = 0
        t = 0.0
        for k in range(steps):
            while mi < len(marks) and marks[mi] <= t + 0.5 * h:
                out.append(self._poses())
                self.substep_pieces.append(self.piece_poses() if self.sets else None)
                mi += 1
            self._keyed(scene, frame - 1 + (t + 0.5 * h) / fdt)
            self._forces()
            mujoco.mj_step(m, d)
            self._keep_in_box()
            self._break()
            t += h
            self.time += h
        while mi < len(marks):
            out.append(self._poses())
            self.substep_pieces.append(self.piece_poses() if self.sets else None)
            mi += 1
        m.opt.timestep = dt
        self.started = True
        self._last = self._poses()
        return out

    def _poses(self):
        """Overrides (Scene.colliders_gpu) for every body as it is now: its frame's origin, velocity there,
        orientation and spin."""
        import mujoco
        m, d = self.model, self.data
        out = {}
        vel6 = np.zeros(6)
        for bd in self.bodies:
            bid = bd.body_id
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, vel6, 0)
            out[bd.index] = dict(pos=tuple(float(x) for x in d.xpos[bid]), rot_y=0.0, spin=0.0,
                                 vel=tuple(float(x) for x in vel6[3:]), quat=tuple(float(x) for x in xyzw(d.xquat[bid])),
                                 omega=(*(float(x) for x in vel6[:3]), 0.0))
        for ps in self.sets:
            out[ps.index] = self.gone()
        return out

    def overrides(self):
        return dict(self._last)

    @staticmethod
    def gone():
        """The override that takes a broken object's whole shape out of the simulation and the render (its pieces
        stand for it): far below, hiding nothing."""
        return dict(pos=(0.0, -1.0e4, 0.0), vel=(0.0, 0.0, 0.0), rot_y=0.0, spin=0.0, quat=(0.0, 0.0, 0.0, 1.0),
                    omega=(0.0, 0.0, 0.0, 0.0), holdout=False)

    # -- coupling ----------------------------------------------------------------------------------

    SIDES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)

    def sample_points(self):
        """Where to sample the gas velocity: six points just outside each body, on either side of it along each
        axis (inside a solid the gas does not move). Fire-local m, six rows per body."""
        if self.data is None or not self.bodies:
            return np.zeros((0, 3))
        out = []
        for bd in self.bodies:
            c = self.data.xipos[bd.body_id]
            r = 1.15 * (float(np.linalg.norm(bd.fb.size)) if bd.fb is not None else float(np.linalg.norm(bd.size))) + 0.02
            out.append(c + self.SIDES * r)
        return np.concatenate(out)

    def set_air(self, vels):
        """The gas velocity at sample_points(): each body feels their average."""
        v = np.asarray(vels, float).reshape(-1, 6, 3)
        for bd, vv in zip(self.bodies, v):
            bd.air = vv.mean(axis=0)

    def regions(self, scene):
        """(index in the solver's collider list, bounding radius) of every body, for the liquid's force sums."""
        enabled = [i for i, c in enumerate(scene.colliders) if c['enabled']]
        out = []
        for bd in self.bodies:
            if bd.index in enabled:
                r = float(np.linalg.norm(bd.fb.size)) if bd.fb is not None else float(np.linalg.norm(bd.size))
                out.append((enabled.index(bd.index), r))
        return out

    def liquid_measures(self, measures, frame_dt, substeps, h, rho):
        """Take in the liquid's push on every body over the last frame (LiquidSolver.read_float), for the next."""
        self.rho = float(rho)
        dt_sub = frame_dt / max(1, substeps)
        for bd, m in zip(self.bodies, measures):
            if bd.fb is None:
                continue
            if not m['wet']:
                bd.hydro = None
                continue
            if m['cells'] > 0.5:
                bd.fb.volume = bd.volume   # (the measured volume can be off for thin shapes: keep the true one)
            fb = bd.fb
            fb.pos = self.data.xipos[bd.body_id].copy()
            fb.quat = xyzw(self.data.xquat[bd.body_id])
            force = rho * h * h / dt_sub * np.asarray(m['force'], float)
            moment = rho * h ** 3 / dt_sub * np.asarray(m['moment'], float)
            sub = min(1.0, max(0.0, force[1] / (rho * self.gravity * bd.volume)))
            vl = np.asarray(m['liquid_vel'], float)
            wl = fb.waterline(max(force[1], 0.0), rho)
            f_hyd, t_hyd = fb.buoyancy(wl, rho)
            bd.hydro = (force - f_hyd, moment - t_hyd, sub, vl, wl)

    # -- state ---------------------------------------------------------------------------------------

    def state(self):
        """Everything needed to carry on from now, and the bodies' poses (for drawing a cached frame)."""
        if self.data is None:
            return None
        d = self.data
        return dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), time=float(self.time),
                    poses={int(k): (v['pos'], v['vel'], v['quat'], v['omega'][:3]) for k, v in self._last.items()},
                    hydro=[None if bd.hydro is None else tuple(np.asarray(x, float).tolist() if hasattr(x, '__len__') else float(x)
                                                                for x in bd.hydro) for bd in self.bodies],
                    held={str(bd.index): bool(bd.held) for bd in self.bodies},
                    eq_active=d.eq_active.copy(), over=None if self._w is None else self._w['over'].copy(),
                    pieces={int(k): {kk: np.asarray(vv, np.float32) for kk, vv in v.items()} for k, v in self.piece_poses().items()})

    def load_state(self, st):
        """Carry on from a saved state (state()). False if it does not fit the current model."""
        import mujoco
        if st is None or self.data is None:
            return False
        q, v = np.asarray(st['qpos'], float), np.asarray(st['qvel'], float)
        if q.shape != self.data.qpos.shape or v.shape != self.data.qvel.shape:
            return False
        self.data.qpos[:] = q
        self.data.qvel[:] = v
        mujoco.mj_forward(self.model, self.data)
        self.time = float(st.get('time', 0.0))
        for bd, hy in zip(self.bodies, st.get('hydro') or []):
            bd.hydro = None if hy is None else (np.asarray(hy[0]), np.asarray(hy[1]), float(hy[2]), np.asarray(hy[3]), float(hy[4]))
        for bd in self.bodies:
            bd.held = bool(st.get('held', {}).get(str(bd.index), False))
        ea = st.get('eq_active')
        if ea is not None and np.shape(ea) == self.data.eq_active.shape:
            self.data.eq_active[:] = ea
        if self._w is not None and st.get('over') is not None and np.shape(st['over']) == self._w['over'].shape:
            self._w['over'][:] = st['over']
        self.started = True
        self._last = self._poses()
        return True

    @staticmethod
    def overrides_from(state):
        """Overrides from a saved state's poses (or from an older cache's floating objects)."""
        poses = state.get('poses', {}) if isinstance(state, dict) and 'qpos' in state else None
        if poses is None:
            from .liquid_float import Floats
            return Floats.overrides_from(state)
        out = {int(i): dict(pos=tuple(p[0]), vel=tuple(p[1]), rot_y=0.0, spin=0.0, quat=tuple(p[2]),
                            omega=(*p[3][:3], 0.0)) for i, p in poses.items() if not (isinstance(p, dict) or p[0][1] < -1.0e3)}
        for i in (state.get('pieces') or {}):
            out[int(i)] = Solids.gone()
        return out


def _rotation(q):
    """3x3 rotation of a quaternion (x, y, z, w)."""
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def attached(scene, kind, poses):
    """Where the things of a kind ('emitter', 'light', 'fabric') attached to a falling or floating object are,
    from the object's simulated pose (poses: collider index -> pose, as Solids.overrides gives them):
    {index: {position, end, yaw, direction, velocity}} for Scene.emitters_gpu / lamps / fabrics_at.

    A link (Scene.links) keeps its child at an offset from its parent, set while the parent was where its keys
    put it at the start; the offset turns with the parent from there."""
    out = {}
    if not poses:
        return out
    for link in getattr(scene, 'links', None) or []:
        if link['child'][0] != kind or link['parent'][0] != 'collider':
            continue
        pi, parent = scene.find_object(*link['parent'])
        ci, child = scene.find_object(*link['child'])
        if pi is None or ci is None or pi not in poses:
            continue
        pose = poses[pi]
        yaw0 = math.radians(float(scene.get(('collider', pi, 'yaw'), scene.start)))
        c0, s0 = math.cos(yaw0), math.sin(yaw0)
        r0t = np.array([[c0, 0.0, -s0], [0.0, 1.0, 0.0], [s0, 0.0, c0]])     # the parent's start yaw, undone
        turned = _rotation(pose.get('quat', (0.0, 0.0, 0.0, 1.0)))           # (ColliderGPU: its quaternion after its yaw)
        if pose.get('rot_y'):
            ry = float(pose['rot_y'])
            turned = turned @ np.array([[math.cos(ry), 0.0, math.sin(ry)], [0.0, 1.0, 0.0], [-math.sin(ry), 0.0, math.cos(ry)]])
        rot = turned @ r0t                                                     # how far it has turned since the start
        centre = np.asarray(pose['pos'], float)
        omega = np.asarray(pose.get('omega', (0.0, 0.0, 0.0))[:3], float)
        vel = np.asarray(pose.get('vel', (0.0, 0.0, 0.0)), float)
        arm = rot @ np.asarray(link.get('offset', (0.0, 0.0, 0.0)), float)
        p = centre + arm
        entry = dict(position=tuple(float(x) for x in p), velocity=tuple(float(x) for x in vel + np.cross(omega, arm)))
        # its turn about the vertical, for things that only turn that way (an emitter's shape, a fabric's pins)
        dyaw = math.degrees(math.atan2(rot[0, 2], rot[2, 2]))
        entry['yaw'] = float(scene.get((kind, ci, 'yaw'), scene.start)) + dyaw if 'yaw' in child else dyaw
        if kind == 'emitter' and child.get('shape') == 'capsule' and 'end_offset' in link:
            entry['end'] = tuple(float(x) for x in centre + rot @ np.asarray(link['end_offset'], float))
        if kind == 'light':
            aim = np.asarray(scene.get(('light', ci, 'direction'), scene.start), float)
            entry['direction'] = tuple(float(x) for x in rot @ aim)
        out[ci] = entry
    return out


def mesh_occupancy(v, t, lo, hi, cell):
    """Which cells of a grid over [lo, hi] (cells of `cell` m) are inside a closed triangle mesh: (nx, ny, nz) bools,
    from how many times a ray along z through each column crosses it below each cell's centre. None if the mesh
    is open (a ray crosses it an odd number of times)."""
    dims = np.maximum(np.ceil((hi - lo) / cell - 1e-9).astype(int), 1)
    nx, ny, nz = (int(x) for x in dims)
    # (the rays pass a hair off the columns' centres, so none goes exactly through an edge or a corner)
    xs = lo[0] + (np.arange(nx) + 0.5 + 1.37e-4) * cell
    ys = lo[1] + (np.arange(ny) + 0.5 + 2.71e-4) * cell
    zs = lo[2] + (np.arange(nz) + 0.5) * cell
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    d = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    keep = np.abs(d) > 1e-18
    a, b, c, d = a[keep], b[keep], c[keep], d[keep]
    x0, x1 = np.minimum(np.minimum(a[:, 0], b[:, 0]), c[:, 0]), np.maximum(np.maximum(a[:, 0], b[:, 0]), c[:, 0])
    y0, y1 = np.minimum(np.minimum(a[:, 1], b[:, 1]), c[:, 1]), np.maximum(np.maximum(a[:, 1], b[:, 1]), c[:, 1])
    occ = np.zeros((nx, ny, nz), bool)
    odd = 0
    for ix, x in enumerate(xs):
        sx = (x0 <= x) & (x1 >= x)
        if not sx.any():
            continue
        A, B, C, D, Y0, Y1 = a[sx], b[sx], c[sx], d[sx], y0[sx], y1[sx]
        for iy, y in enumerate(ys):
            sy = (Y0 <= y) & (Y1 >= y)
            if not sy.any():
                continue
            Ai, Bi, Ci, Di = A[sy], B[sy], C[sy], D[sy]
            l1 = ((Bi[:, 1] - Ci[:, 1]) * (x - Ci[:, 0]) + (Ci[:, 0] - Bi[:, 0]) * (y - Ci[:, 1])) / Di
            l2 = ((Ci[:, 1] - Ai[:, 1]) * (x - Ci[:, 0]) + (Ai[:, 0] - Ci[:, 0]) * (y - Ci[:, 1])) / Di
            l3 = 1.0 - l1 - l2
            hit = (l1 >= 0.0) & (l2 >= 0.0) & (l3 >= 0.0)
            if not hit.any():
                continue
            z = np.sort(l1[hit] * Ai[hit, 2] + l2[hit] * Bi[hit, 2] + l3[hit] * Ci[hit, 2])
            odd += len(z) % 2
            occ[ix, iy] = (np.searchsorted(z, zs) % 2) == 1
    if odd > 0.02 * nx * ny + 2:
        return None
    return occ


def greedy_boxes(occ):
    """Filled cells merged into few boxes: [(x0, y0, z0, x1, y1, z1)] (inclusive cell ranges), each grown as far as
    it goes along x, then z, then y."""
    occ = occ.copy()
    nx, ny, nz = occ.shape
    out = []
    for y in range(ny):
        for z in range(nz):
            row = occ[:, y, z]
            if not row.any():
                continue
            for x in np.nonzero(row)[0]:
                if not occ[x, y, z]:
                    continue
                x1 = x
                while x1 + 1 < nx and occ[x1 + 1, y, z]:
                    x1 += 1
                z1 = z
                while z1 + 1 < nz and occ[x:x1 + 1, y, z1 + 1].all():
                    z1 += 1
                y1 = y
                while y1 + 1 < ny and occ[x:x1 + 1, y1 + 1, z:z1 + 1].all():
                    y1 += 1
                occ[x:x1 + 1, y:y1 + 1, z:z1 + 1] = False
                out.append((int(x), y, z, int(x1), y1, z1))
    return out
