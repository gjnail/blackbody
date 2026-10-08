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
from .liquid_float import FloatBody, inertia, q_rot

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
    pivot: np.ndarray = None    # the fixed point a hinge or a ball joint turns it about (world), if any: it spins about that
    rho_l: float = 0.0          # the density of the liquid it was last in (kg/m^3; 0: the scene's), water or lava
    liquid_drag: float = 1.0    # times the drag of water the liquid it is in has on it (lava: LAVA_DRAG)


LAVA_DRAG = 12.0        # lava's drag on what sits in it, times water's (a thousand times as viscous: things in it move
                        # with it, and sink or rise slowly)
CLOTH_HZ = 12.0         # a thing lying on cloth springs on it at about this rate (contact stiffness from its mass):
                        # it sinks g / (2 pi f)^2 = 1.7 mm into its margin
CLOTH_DAMPING = 0.7     # and damped this much of critically, for a thing that does not bounce (by its own bounce, less)
CLOTH_HOLD = 100.0      # N: the most a vertex of cloth pushes back (fabric's strength, at a couple of centimetres
                        # between vertices: a sheet cannot stop a wrecking ball, whatever the contact's stiffness)
CLOTH_MARGIN = 0.006    # m: how close to a thing's surface a vertex of cloth holds it off (resting, it stays further off
                        # than the cloth's own thickness, or the cloth, pushed out of it each step, would let it sink)
MORTAR = 0.6e6          # Pa: mortar in tension, the glue between the bricks of a wall (real: 0.2-0.8 MPa)
JOINT_FRICTION = 0.6    # shear a joint holds beyond its cohesion, per newton of compression across it
GAP = 0.001             # m: cut faces are moved in by this for the physics, so glued neighbours do not touch
BREAK_STEPS = 1         # steps in a row a weld must be overloaded to break (an impact lasts only a few)
CHAIN_KG_M2 = 20000.0   # kg/m per square metre of its wire: a chain's weight (1 cm wire: 2 kg a metre)
WELD_SOLIMP = (0.99, 0.999, 0.001)   # how hard a weld holds its pose (MuJoCo's impedance): MuJoCo's own 0.9-0.95 gives
                        # as the inertia of a small piece, so a post of steel segments whipped a metre either way like a
                        # fishing rod and a wall swayed; 0.999-0.9999 goes unstable
IMPACT_FLOOR = 0.5      # m/s: a piece stopped by less than this never breaks by impact (contacts at rest chatter)
IMPACT_KEEP = 0.5       # of a piece's stop so far, what carries to the next step (a hit lasts a few steps)
REHEARSE_S = 20.0       # s: the longest a rehearsal (where things are hit, _rehearse) runs the shot for
REHEARSE_WALL = 15.0    # s: and the longest it may take


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
    burnable: bool = False      # its pieces burn (burnable, with Spreading fire on)
    joints_show: bool = False   # its pieces' joints show even whole (a brick wall's mortar)
    fire: np.ndarray = None     # (n, 4) each piece as a burnable spot is (burn_common.wgsl): fuel left, catching (1:
                                # alight), 1 (2 once burnt out), smoulder left
    gone: np.ndarray = None     # (n,) crumbled to ash: no longer there
    thick: np.ndarray = None    # (n,) its thinnest dimension (m)
    area: np.ndarray = None     # (n,) its surface (m^2)
    half: np.ndarray = None     # (n, 3) its half extents in its frame (m)
    pairs: np.ndarray = None    # (b, 2) the pieces glued to each other
    crumble: np.ndarray = None  # (n,) how far its smoulder dies down before it falls to ash (0..1)
    wood: object = None         # a wood's burning (wood_fire.WoodFire) when it is wood: it burns as wood does, spot by spot
    spots: np.ndarray = None    # (n x 54, 4) its pieces' surfaces as wood_fire's spots (a 3 x 3 grid on each face)
    wstate: dict = None         # wood_fire.step_spots' state per spot: char depth (m), seconds alight, seconds starved of heat
    spot_local: np.ndarray = None   # (n, 54, 3) where each spot is on its piece (its own frame, from its middle)
    spot_area: np.ndarray = None    # (n, 54) the surface each stands for (m^2)
    spot_edges: np.ndarray = None   # (E, 2) the spots a flame creeps between (on a piece and across to the next)
    spot_lens: np.ndarray = None    # (E,) how far apart (m)
    spot_open: np.ndarray = None    # (n x 54,) which are on its open faces (not on one glued to the next piece: inside the wood)
    spot_sync: np.ndarray = None    # (S, 2) spots that are the same place on the wood (a shared edge, a corner, a seam)
    spot_thick: np.ndarray = None   # (n x 54,) the wood's thickness straight through each, to its twin on the other face
    spot_opp: np.ndarray = None     # (n x 54,) that twin
    spot_blend: np.ndarray = None   # (B, 2) spots beside each other on the same face of the wood, across a seam (drawn
                                    # blended: _spots_drawn)
    spot_blend_w: np.ndarray = None     # (B,) how much each takes of the other
    spot_blend_rest: np.ndarray = None  # (B,) how far apart they are, the wood whole (m)
    v_break: float = 0.0        # m/s: a piece of it stopped this fast by a hit breaks away (materials impact x Strength)
    impact: np.ndarray = None   # where it is hit (its own frame, _rehearse), what its cracks crowd round; None: nowhere
    brittle: float = 0.0        # how far a crack runs on from a hit (materials)
    yields: float = 0.0         # Pa: its joints bend and stay bent past this (materials x Strength); 0: they never bend
    ductility: float = 0.0      # radians: how far a joint bends in all before it tears
    bent: bool = False          # a joint of it has bent (it is drawn as its pieces from then on)


ROPE_BOUNCE = 0.3       # the share of its speed a falling thing keeps when its rope snaps taut

from . import wood_fire as WF   # noqa: E402  (wood burning as wood does)
SPOTS = 54     # a wood piece's spots: a 3 x 3 grid on each of its faces (wood_fire.template)


def blast_impulse(kg, r):
    """The impulse per area (Pa s) a blast of `kg` of TNT gives a surface facing it `r` metres off: twice the side-on
    impulse of the shock (it reflects off the surface), which falls off about as 1 / r at blasting distances
    (Hopkinson-Cranz scaling: about 200 Pa s for 1 kg at 1 m side-on). Inside the fireball (r < W^1/3 / 3) it is
    taken as there."""
    w = max(float(kg), 0.0)
    if w <= 0.0:
        return 0.0
    r = max(float(r), w ** (1.0 / 3.0) / 3.0)
    return 2.0 * 200.0 * w ** (2.0 / 3.0) / r

CHAR_DEPTH = 0.05       # m: a piece this thick burns for Spreading fire's Burn time; thicker ones longer, in proportion
FIRE_AREA_DEPTH = 0.025 # m: a burning piece gives off fuel as a burning surface does over this depth of cells
SPRING_DAMPING = 0.05   # a spring's damping ratio (with the object on its end)
HINGE_SPAN = 0.05       # m: a hinge's two pins are at least this far apart
MOTOR_STEPS = 4         # a motor closes the gap to its speed over about this many steps (fewer would overshoot)
RPM = 2.0 * math.pi / 60.0


@dataclass
class Joint:
    """A rope, spring, hinge or ball joint holding an object that falls (Properties › Joint): a MuJoCo tendon between
    two sites (a rope: a limit on its length; a spring: its stiffness), or connect constraints (a ball joint: one at
    its pivot; a hinge: two, at its pins along its axis)."""
    index: int                  # the collider it holds
    kind: str                   # rope, spring, hinge, ball
    body: str                   # the MuJoCo body it is on (the object, or the piece of a breakable it is tied to)
    other: str = ''             # and the one at its other end ('': the world)
    sites: tuple = ()           # its sites: (on it, on the other) for a rope, spring or ball joint, two such pairs for a hinge
    eqs: tuple = ()             # its connect constraints (hinge, ball)
    tendon: str = ''            # its tendon (rope, spring)
    length: float = 0.0         # the rope's length, or the spring's at rest (m)
    strength: float = 0.0       # the force that breaks it (N); 0: it never breaks
    axis: np.ndarray = None     # a hinge's axis in its body's frame
    friction: float = 0.0       # how fast a hinge's or ball joint's turning dies away (1/s)
    motor: float = 0.0          # a hinge's motor: the speed it turns it at (rad/s, about its axis), 0: none
    torque: float = 0.0         # and the most torque it has (N m)
    keyed: bool = False         # the motor's speed is keyframed
    radius: float = 0.0125      # the rope's (or the spring's wire's) radius, for drawing (m)
    look: int = 0               # ropes.ROPE, CABLE or SPRING
    broken: bool = False
    post: tuple = None          # a rope: the post or ball it may wrap round (geom name, centre, axis, radius)
    links: list = None          # a chain: its links' body names, from the far end to the object
    link_ids: np.ndarray = None
    link_half: np.ndarray = None   # each link's start (toward the far end) from its middle, in its own frame
    bid: int = -1               # MuJoCo ids, once compiled: its body, the other's (0: the world), sites, constraints, tendon
    oid: int = 0
    other_free: bool = False    # the other end is on a body that moves freely (it takes the joint's friction back)
    sids: tuple = ()
    eids: tuple = ()
    tid: int = -1


_FRACTURES = {}
_HITS = {}          # rehearsals (_rehearse) kept: scene -> {collider index: where it is hit}


def breaks(c):
    """Whether collider c is breakable (not a person or a car: those are their parts)."""
    return bool(c.get('breakable')) and kind_of(c) is None


def kind_of(c):
    """The kind of assembly (engine/assemblies.py: 'figure', 'car') collider c is built as, or None."""
    from .assemblies import kind_of as k
    return k(c)


def joined(c):
    """Collider c's joint kind ('rope', 'spring', 'hinge', 'ball'), or None."""
    k = c.get('joint', 'none')
    return k if k in ('rope', 'spring', 'hinge', 'ball') else None


def falls(c):
    """Whether collider c moves as a rigid body that falls: Falls is ticked, or it hangs on a joint."""
    return bool(c.get('dynamic')) or joined(c) is not None


def shape_distance(shape, size, q, hull=None):
    """Signed distance (m, negative inside) from points q (n, 3: the thing's own frame) to its shape, and the outward
    normal there (n, 3): a sphere, a box, a cylinder standing along y, or a mesh by its hull's box (colliders.wgsl's
    shapes)."""
    q = np.asarray(q, float)
    if shape == 'sphere':
        L = np.linalg.norm(q, axis=1)
        n = np.where(L[:, None] > 1e-12, q / np.maximum(L, 1e-12)[:, None], np.array([0.0, 1.0, 0.0]))
        return L - float(size[0]), n
    if shape == 'cylinder':
        r, hh = float(size[0]), float(size[1])
        rho = np.hypot(q[:, 0], q[:, 2])
        radial = np.stack([q[:, 0], np.zeros(len(q)), q[:, 2]], 1) / np.maximum(rho, 1e-12)[:, None]
        axial = np.stack([np.zeros(len(q)), np.sign(q[:, 1]) + (q[:, 1] == 0), np.zeros(len(q))], 1)
        a = np.stack([rho - r, np.abs(q[:, 1]) - hh], 1)
        out = np.maximum(a, 0.0)
        lo = np.linalg.norm(out, axis=1)
        d = lo + np.minimum(a.max(axis=1), 0.0)
        n_out = (radial * out[:, :1] + axial * out[:, 1:]) / np.maximum(lo, 1e-12)[:, None]
        n_in = np.where((a[:, 0] >= a[:, 1])[:, None], radial, axial)
        return d, np.where((lo > 0.0)[:, None], n_out, n_in)
    centre = np.zeros(3)
    half = np.asarray(size, float)[:3]
    if shape == 'mesh' and hull is not None and len(hull):
        hv = np.asarray(hull, float)
        centre = 0.5 * (hv.min(axis=0) + hv.max(axis=0))
        half = 0.5 * (hv.max(axis=0) - hv.min(axis=0))
    p = q - centre
    a = np.abs(p) - half
    out = np.maximum(a, 0.0)
    lo = np.linalg.norm(out, axis=1)
    d = lo + np.minimum(a.max(axis=1), 0.0)
    s = np.where(p >= 0.0, 1.0, -1.0)
    n_out = s * out / np.maximum(lo, 1e-12)[:, None]
    n_in = np.zeros_like(p)
    k = a.argmax(axis=1)
    n_in[np.arange(len(p)), k] = s[np.arange(len(p)), k]
    return d, np.where((lo > 0.0)[:, None], n_out, n_in)


def surface_toward(shape, size, u, hull=None):
    """The point of a shape's surface (its own frame) straight out from its middle along unit vector u."""
    s = np.abs(np.asarray(size, float))
    u = np.asarray(u, float)
    if shape == 'sphere':
        return u * s[0]
    if shape == 'mesh' and hull is not None and len(hull):
        return u * max(float((np.asarray(hull, float) @ u).max()), 0.0)
    if shape == 'cylinder':
        r = math.hypot(u[0], u[2])
        t = min(s[1] / abs(u[1]) if abs(u[1]) > 1e-9 else np.inf, s[0] / r if r > 1e-9 else np.inf)
    else:
        with np.errstate(divide='ignore'):
            t = float(np.min(np.where(np.abs(u) > 1e-9, s / np.maximum(np.abs(u), 1e-12), np.inf)))
    return u * (t if np.isfinite(t) else 0.0)


def turn_wxyz(cg):
    """A ColliderGPU's whole orientation as a MuJoCo quaternion (w, x, y, z): its yaw, then its quaternion."""
    return _quat_mul_wxyz(wxyz(cg.quat), _yaw_wxyz(float(cg.rot_y)))


def joint_ends(c, cg, other=None, hull=None):
    """Where a joint's two ends start (world points): (on the object, at the other end). c: the collider; cg: its
    ColliderGPU at the start; other: the ColliderGPU of the object it is joined to (None: Anchor, a fixed point).
    A hinge or a ball joint has both ends at its pivot (Joint on it); a rope or a spring is tied at Joint on it, or
    where the object's surface faces the other end, to Anchor or to Joint on the other."""
    pos = np.asarray(cg.pos, float)
    R = q_rot(xyzw(_quat_mul_wxyz(wxyz(cg.quat), _yaw_wxyz(float(cg.rot_y)))))
    at = np.asarray(c.get('joint_at', (0.0, 0.0, 0.0)), float)
    if joined(c) not in ('rope', 'spring'):
        P = pos + R @ at
        return P, P.copy()
    if other is not None:
        Ro = q_rot(xyzw(_quat_mul_wxyz(wxyz(other.quat), _yaw_wxyz(float(other.rot_y)))))
        B = np.asarray(other.pos, float) + Ro @ np.asarray(c.get('joint_to_at', (0.0, 0.0, 0.0)), float)
    else:
        B = np.asarray(c.get('joint_anchor', (0.0, 2.0, 0.0)), float)
    if np.allclose(at, 0.0):
        u = R.T @ (B - pos)
        nu = float(np.linalg.norm(u))
        if nu > 1e-9:
            at = surface_toward(c['shape'], cg.size, u / nu, hull)
    return pos + R @ at, B


def extent_along(shape, size, axis, hull=None):
    """How far a shape reaches from its middle along unit vector `axis` (its own frame)."""
    s = np.abs(np.asarray(size, float))
    a = np.abs(np.asarray(axis, float))
    if shape == 'sphere':
        return float(s[0])
    if shape == 'mesh' and hull is not None and len(hull):
        return float(np.abs(np.asarray(hull, float) @ np.asarray(axis, float)).max())
    if shape == 'cylinder':
        return float(a[1] * s[1] + math.hypot(a[0], a[2]) * s[0])
    return float(a @ s)


def fractured(c, size, hollow=0.0, impact=None, scene=None):
    """The pieces a breakable collider is cut into (kept: cutting takes a fraction of a second). impact: where it is
    hit (its own frame: PieceSet.impact), the cracks crowd round it. scene: for a mesh, where its file is."""
    from .fracture import fracture
    if kind_of(c):           # (a person's or a car's parts)
        from .assemblies import assembly
        return assembly(kind_of(c), size).fracture()
    mesh = mesh_key = None
    if c['shape'] == 'mesh':
        mesh, mesh_key = mesh_of(scene, c, size)
        if mesh is None:
            from .fracture import Fracture
            return Fracture()
    if impact is not None and uses_impact(c):
        impact = tuple(round(float(x), 4) for x in impact)
    else:
        impact = None
    rim = open_top(c, size)
    key = (c['shape'], tuple(round(float(x), 6) for x in np.abs(np.asarray(size, float))), int(c.get('pieces', 24)),
           c.get('fracture', 'voronoi'), int(c.get('fracture_seed', 0)), round(float(hollow or 0.0), 6), impact,
           None if rim is None else round(rim, 6), mesh_key)
    f = _FRACTURES.get(key)
    if f is None:
        if len(_FRACTURES) > 32:
            _FRACTURES.clear()
        f = _FRACTURES[key] = fracture(key[0], key[1], key[2], key[3], key[4], hollow=key[5],
                                       impact=None if impact is None else np.asarray(impact, float), rim=rim, mesh=mesh)
    return f


_MESHES = {}


def mesh_of(scene, c, size):
    """A mesh collider's vertices (scaled by its size) and triangles in its own frame, and a key for them (its file
    and when it changed); (None, None) if it cannot be read."""
    import os
    from .mesh import MeshError, load_mesh
    if scene is None or not c.get('mesh'):
        return None, None
    path = scene.mesh_path(c['mesh'])
    try:
        stamp = os.path.getmtime(path.split('#')[0])
    except OSError:
        stamp = 0.0
    key = (path, stamp, tuple(round(float(x), 6) for x in np.abs(np.asarray(size, float))))
    got = _MESHES.get(key)
    if got is None:
        try:
            v, t = load_mesh(path)
        except (MeshError, OSError, ValueError):
            return None, None
        if len(_MESHES) > 8:
            _MESHES.clear()
        got = _MESHES[key] = (np.asarray(v, float) * np.abs(np.asarray(size, float)), np.asarray(t, np.int64))
    return got, key


def open_top(c, size):
    """How high a hollow cylinder's wall goes when its opening cuts its whole top off (a pot: the opening a box
    wider than it, across its top), else None (closed)."""
    if c.get('shape') != 'cylinder' or float(c.get('hollow', 0.0) or 0.0) <= 0.0:
        return None
    s = np.abs(np.asarray(size, float))
    o = np.abs(np.asarray(c.get('opening', (0.0, 0.0, 0.0)), float))
    at = np.asarray(c.get('opening_at', (0.0, 0.0, 0.0)), float)
    if not (o > 0.0).all() or o[0] - abs(at[0]) < s[0] or o[2] - abs(at[2]) < s[0] or at[1] + o[1] < s[1]:
        return None
    return float(at[1] - o[1])


def uses_impact(c):
    """Whether where a breakable is hit shapes its cracks (not a brick wall's: its joints are where they are)."""
    return c.get('fracture', 'voronoi') != 'bricks'


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
        self._imp = None           # the impact check's (_index_impact)
        self._set_bonds = []       # each breakable's welds between its pieces (rows of _w)
        self.asms = []             # people and cars (_build_assemblies): their parts' bodies, joints and motors
        self._mocap_vel = {}       # mocap id -> its keys' velocity now (how fast a keyed thing hits)
        self._btab = None          # per body, for how fast contacts close (_bodies_table)
        self.rehearsal = False     # a rehearsal (_rehearse): nothing breaks; where each breakable is hit is noted (hits)
        self.hits = {}             # collider index -> [its own frame point, how fast it closed, (unused)]
        self._reh_geom = {}        # a rehearsal's geoms of breakables -> collider index
        self._reh_frame = {}       # collider index -> (position, rotation) of a standing one's frame; (None, None) falling
        self._hit_at = {}          # collider index -> where the rehearsal found it hit (the pieces cut round it)
        self.joints: list[Joint] = []   # ropes, springs, hinges and ball joints
        self.snaps = []            # (time, collider index) of every joint that broke
        self._tendon0 = None       # the tendons' limits, stiffness and damping as built (a broken one has them taken away)
        self._pushed = {}          # collider index -> (force, torque) the matter (matter.py) put on it over the last frame
        self._cloth = None         # the fabric through this frame (meet_cloth): its vertices, the ones near each thing
        self._cloth_took = None    # what each vertex of it took from the things it held off through the frame (N s)
        self.shots = None          # the scene's bullets (ballistics.py), flown with the bodies
        self.capped = {}           # collider index -> steps the liquid's push on it was held to MAX_ACCEL (since reset)

    # -- set up --------------------------------------------------------------------------------

    @property
    def active(self):
        return bool(self.bodies or self.sets or self.asms or self.shots)

    @staticmethod
    def wanted(scene):
        """Indices of the colliders that move by themselves: Falls (or hangs on a joint), breakable, or Floats in a
        liquid."""
        out = []
        for i, c in enumerate(scene.colliders):
            if not c['enabled']:
                continue
            if falls(c) or breaks(c) or kind_of(c) or (c.get('floating') and scene.kind in ('liquid', 'both')):
                out.append(i)
        return out

    def configure(self, scene, layout):
        """Match the bodies to the scene. Returns True when the model has to be built again (and the
        simulation restarted). layout: (dims, h, origin) of the simulation grid, for the box's walls."""
        from .ballistics import Ballistics
        idx = self.wanted(scene)
        if not idx and not Ballistics.wanted(scene):
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
        self.sets, self.breaks, self._w, self._imp = [], [], None, None
        self.asms = []
        self.joints, self.snaps, self._tendon0 = [], [], None
        self.shots = None
        self.started = False
        self._last = {}
        self.warnings = []
        self.capped = {}
        return changed

    @staticmethod
    def _key(scene, idx, layout):
        import json
        cols = [{k: (list(v) if isinstance(v, tuple) else (str(v) if not isinstance(v, (int, float, bool, str)) else v))
                 for k, v in c.items() if k != 'name'} for c in scene.colliders]
        d = scene.data['domain']
        blob = dict(idx=idx, cols=cols, ground=bool(d['ground']), sides=bool(d['open_sides']), kind=scene.kind,
                    layout=[list(map(float, x)) if hasattr(x, '__len__') else float(x) for x in layout],
                    g=float(scene.data['liquid']['gravity']) if scene.kind in ('liquid', 'both') else G,
                    shots=[{k: (list(v) if isinstance(v, tuple) else str(v)) for k, v in sh.items() if k != 'name'}
                           for sh in getattr(scene, 'shots', None) or []])
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
            frac = fractured(c, cg.size, cg.hollow, self._hit_at.get(i), scene)
            if not frac.pieces:
                continue
            r = resolved(c)
            strength = (MORTAR if c.get('fracture') == 'bricks' else r['strength']) * float(c.get('strength', 1.0))
            q0 = _quat_mul_wxyz(wxyz(cg.quat), _yaw_wxyz(float(cg.rot_y)))
            R0 = q_rot(xyzw(q0))
            dynamic = falls(c)
            ps = PieceSet(index=i, frac=frac, names=[], size=tuple(float(x) for x in cg.size), hollow=float(cg.hollow),
                          v_break=float(r.get('impact', 0.0)) * float(c.get('strength', 1.0)),
                          impact=self._hit_at.get(i) if uses_impact(c) else None, brittle=float(r.get('brittle', 0.0)),
                          yields=float(r.get('yields', 0.0)) * float(c.get('strength', 1.0)),
                          ductility=float(r.get('ductility', 0.0)),
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
            # (what burning needs: each piece's size and surface, and which are glued together)
            ext = np.array([np.ptp(p.verts, axis=0) for p in frac.pieces], float)
            ps.half = 0.5 * ext
            ps.thick = np.maximum(ext.min(1), 1e-3)
            ps.area = 2.0 * (ext[:, 0] * ext[:, 1] + ext[:, 1] * ext[:, 2] + ext[:, 2] * ext[:, 0])
            ps.pairs = np.array([(bd.i, bd.j) for bd in frac.bonds], np.int64).reshape(-1, 2)
            ps.burnable = bool(c.get('burnable')) and bool(scene.data['spread']['enabled'])
            ps.wood = WF.props(str(c.get('material', 'wood')), float(r['density'])) if (ps.burnable and WF.is_wood(str(c.get('material', 'wood')))) else None
            if ps.wood is not None:
                self._wood_spots(ps, frac.pieces)
            ps.joints_show = c.get('fracture') == 'bricks'
            rng = np.random.default_rng(31 * i + 7)
            # (most burnt-out pieces fall to ash as their embers die; some stay, black charcoal)
            ps.crumble = np.where(rng.random(len(frac.pieces)) < 0.6, rng.random(len(frac.pieces)) * 0.8, -1.0)
            self._reset_fire(ps)
            # the bonds: welds between pieces that share a cut
            stiff = [-k, -2.0 * math.sqrt(k)]
            for n, bond in enumerate(frac.bonds):
                e = spec.add_equality()
                e.type = mujoco.mjtEq.mjEQ_WELD
                e.objtype = mujoco.mjtObj.mjOBJ_BODY
                e.name = f'bond{i}_{n}'
                e.name1, e.name2 = ps.names[bond.i], ps.names[bond.j]
                e.solref = stiff
                e.solimp = list(WELD_SOLIMP) + list(e.solimp)[3:]
                # anchored on the shared face (in the second piece's frame), so the weld's torque is the moment there
                data = np.array(e.data, float)
                data[0:3] = np.asarray(bond.centre, float) - frac.pieces[bond.j].centroid
                e.data = data
                # (a bond can be stronger or weaker than the material's: wood's along its grain and across it, wood.py)
                ps.welds.append((e.name, bond.i, np.asarray(bond.normal, float), float(bond.area),
                                 strength * float(getattr(bond, 'k', 1.0)), bond.j))
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
                    e.solimp = list(WELD_SOLIMP) + list(e.solimp)[3:]
                    data = np.array(e.data, float)
                    data[0:3] = np.asarray(cg.pos, float) + R0 @ fc     # (the world's frame: where its glued face is)
                    e.data = data
                    ps.welds.append((e.name, n, nrm, area, strength, -1))
            sets.append(ps)
            if len(frac.pieces) > 150:
                self.warnings.append(f'{c["name"]}: {len(frac.pieces)} pieces take a while to simulate')
        return sets

    def _index_welds(self):
        """The welds of every breakable, flattened into arrays for the per-step check (_break)."""
        import mujoco
        m = self.model
        rows = []
        for si, ps in enumerate(self.sets):
            for name, first, nrm, area, strength, other in ps.welds:
                rows.append((mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, name), int(ps.bodies[first]), nrm, area, strength,
                             ps.index, si, first, other))
        if not rows:
            self._w = None
            self._imp = None
            return
        eq = np.array([r[0] for r in rows], np.int64)
        lookup = np.full(max(int(m.neq), 1), -1, np.int64)
        lookup[eq] = np.arange(len(rows))
        self._w = dict(eq=eq, body=np.array([r[1] for r in rows], np.int64), normal=np.array([r[2] for r in rows], float),
                       area=np.array([r[3] for r in rows], float), strength=np.array([r[4] for r in rows], float),
                       collider=np.array([r[5] for r in rows], np.int64), lookup=lookup, over=np.zeros(len(rows), np.int64),
                       set=np.array([r[6] for r in rows], np.int64), first=np.array([r[7] for r in rows], np.int64),
                       other=np.array([r[8] for r in rows], np.int64))
        self._w['strength0'] = self._w['strength'].copy()
        sets = self.sets
        self._w['yield'] = np.array([sets[si].yields for si in self._w['set']], float)
        self._w['duct'] = np.array([sets[si].ductility for si in self._w['set']], float)
        self._w['plastic'] = np.zeros(len(rows))          # how far each has bent in all (radians, and stretch / its size)
        self._w['rel0'] = m.eq_data[eq, 3:10].copy()      # their rest poses as built (a bent one takes a new one)
        self._index_impact()

    def _index_impact(self):
        """What the impact check (_impact) needs: each body's piece (or -1), each piece's free-joint velocity address,
        the speed that breaks it, and the welds that hold it."""
        W = self._w
        m = self.model
        # (the welds between each one's pieces, for whether it is still whole)
        self._set_bonds = [np.nonzero((W['set'] == si) & (W['other'] >= 0))[0] if W is not None else np.zeros(0, np.int64)
                           for si in range(len(self.sets))]
        if W is None or not self.sets:
            self._imp = None
            return
        body_piece = np.full(int(m.nbody), -1, np.int64)
        lin, vb, start = [], [], []
        g = 0
        for si, ps in enumerate(self.sets):
            start.append(g)
            for n, b in enumerate(ps.bodies):
                body_piece[int(b)] = g + n
                lin.extend(range(int(ps.vadr[n]), int(ps.vadr[n]) + 3))
                vb.append(ps.v_break)
            g += len(ps.bodies)
        start = np.asarray(start, np.int64)
        # each weld's pieces, as global piece numbers (an anchor holds only its first)
        a = start[W['set']] + W['first']
        b = np.where(W['other'] >= 0, start[W['set']] + W['other'], -1)
        holds = [[] for _ in range(g)]
        nb = [[] for _ in range(g)]
        for r in range(len(a)):
            holds[a[r]].append(r)
            if b[r] >= 0:
                holds[b[r]].append(r)
                nb[a[r]].append((int(b[r]), r))
                nb[b[r]].append((int(a[r]), r))
        # (for cracks running on: each piece's centre in its object's frame, how brittle it is, and how far apart its
        # set's glued pieces typically are)
        cen = np.concatenate([np.array([p.centroid for p in ps.frac.pieces[:len(ps.bodies)]], float).reshape(-1, 3)
                              for ps in self.sets])
        brittle = np.concatenate([np.full(len(ps.bodies), ps.brittle) for ps in self.sets])
        ell = np.zeros(g)
        for si, ps in enumerate(self.sets):
            sel = (W['set'] == si) & (W['other'] >= 0)
            gap = np.linalg.norm(cen[a[sel]] - cen[b[sel]], axis=1)
            ell[start[si]:start[si] + len(ps.bodies)] = float(np.median(gap)) if len(gap) else 1.0
        self._imp = dict(piece=body_piece, lin=np.asarray(lin, np.int64), v_break=np.asarray(vb, float),
                         acc=np.zeros(g), holds=[np.asarray(h, np.int64) for h in holds], on=np.ones(g, bool),
                         nb=nb, cen=cen, brittle=brittle, ell=np.maximum(ell, 1e-4), ends=(a, b))

    def _note_hits(self, before):
        """A rehearsal's note of where each breakable is hit hardest: of the contacts on it this step, the one that
        closed fastest (the two things' speeds toward each other there, just before), in its own frame."""
        d, m = self.data, self.model
        nc = int(d.ncon)
        if nc == 0 or not self._reh_geom:
            return
        ks = [k for k in range(nc) if int(d.contact.geom1[k]) in self._reh_geom or int(d.contact.geom2[k]) in self._reh_geom]
        if not ks:
            return
        ks = np.asarray(ks, np.int64)
        speeds = self._closing(before, np.asarray(m.geom_bodyid[d.contact.geom1[ks]], np.int64),
                               np.asarray(m.geom_bodyid[d.contact.geom2[ks]], np.int64), np.asarray(d.contact.pos[ks], float),
                               np.asarray(d.contact.frame[ks, :3], float))
        for k, closing in zip(ks, speeds):
            g1, g2 = int(d.contact.geom1[k]), int(d.contact.geom2[k])
            i = self._reh_geom.get(g1, self._reh_geom.get(g2))
            p = np.asarray(d.contact.pos[k], float)
            closing = float(closing)
            got = self.hits.get(i)
            if closing <= 0.5 * IMPACT_FLOOR or (got is not None and closing <= got[1]):
                continue
            pos, R = self._reh_frame[i]
            if pos is None:            # (it falls: its body's frame is its own)
                bid = int(m.geom_bodyid[g1] if g1 in self._reh_geom else m.geom_bodyid[g2])
                pos, R = d.xpos[bid], d.xmat[bid].reshape(3, 3)
            self.hits[i] = [np.round(R.T @ (p - pos), 4), closing, False]

    def _build_assemblies(self, scene, spec, w, idx, by_index, contact):
        """People and cars: each its parts as MuJoCo bodies in a tree (the root on a free joint), on ball joints,
        hinges and sliding springs within their ranges; a car's wheels driven and steered by motors. Returns a record
        each for _index_assemblies."""
        import mujoco
        from .assemblies import STAND, LIMP, CAR_TORQUE, assembly
        out = []
        for i in idx:
            c = scene.colliders[i]
            cg = by_index.get(i)
            kind = kind_of(c)
            if cg is None or kind is None:
                continue
            A = assembly(kind, cg.size)
            r = resolved(c)
            q0 = turn_wxyz(cg)
            R0 = q_rot(xyzw(q0))
            limp = kind == 'figure' and c.get('stance', 'stands') == 'limp'
            made, joints, names = [], [], []
            for k, B in enumerate(A.bodies):
                parent = w if B.parent < 0 else made[B.parent]
                b = parent.add_body()
                b.name = f'asm{i}_{k}'
                if B.parent < 0:
                    b.pos = list(map(float, np.asarray(cg.pos, float) + R0 @ B.at))
                    b.quat = list(q0)
                    b.add_freejoint()
                else:
                    b.pos = list(map(float, B.at - A.bodies[B.parent].at))
                    j = b.add_joint()
                    j.name = f'asm{i}_{k}_j'
                    j.pos = list(map(float, B.anchor - B.at))
                    j.armature = 0.01
                    # (ranges in degrees: MuJoCo's compiler takes angles in degrees unless told otherwise)
                    if B.joint == 'ball':
                        j.type = mujoco.mjtJoint.mjJNT_BALL
                        j.range = [0.0, float(B.range[1])]
                    else:
                        j.type = mujoco.mjtJoint.mjJNT_HINGE if B.joint == 'hinge' else mujoco.mjtJoint.mjJNT_SLIDE
                        j.axis = list(map(float, B.axis))
                        if B.range is not None:
                            j.range = [float(x) for x in B.range]
                    j.limited = mujoco.mjtLimited.mjLIMITED_TRUE if B.range is not None else mujoco.mjtLimited.mjLIMITED_FALSE
                    if B.tone:
                        stiff, damp = (LIMP if limp else STAND)[B.tone]
                        j.stiffness, j.damping = [stiff, 0.0, 0.0], [damp, 0.0, 0.0]   # (3-vectors in MuJoCo 3.14's spec)
                    if B.spring is not None:
                        j.stiffness, j.damping = [B.spring[0], 0.0, 0.0], [B.spring[1], 0.0, 0.0]
                        j.springref = B.spring[2]
                    joints.append((k, j.name, B.tone))
                mine = [n for n, kb in enumerate(A.piece_body) if kb == k]
                for n in mine:
                    pc = A.pieces[n]
                    ma = spec.add_mesh()
                    ma.name = f'asm{i}_{k}_{n}'
                    ma.uservert = (pc.verts - B.at).ravel().tolist()
                    g = b.add_geom()
                    g.type = mujoco.mjtGeom.mjGEOM_MESH
                    g.meshname = ma.name
                    g.density = float(A.density[n])
                    g.priority = 1
                    look = A.looks[n]
                    contact(g, 0.95 if look == 'rubber' else r['friction'], 0.3 if look == 'rubber' else r['bounce'])
                if not mine:
                    b.mass = float(B.mass)
                    b.inertia = [1e-3 * B.mass] * 3
                    b.explicitinertial = True
                made.append(b)
                names.append(b.name)
            if kind == 'car':
                # (its wheels turn in its wheel arches, and its hubs and steering knuckles inside it: none of them meet it)
                for k, B in enumerate(A.bodies):
                    if k > 0:
                        ex = spec.add_exclude()
                        ex.bodyname1, ex.bodyname2 = names[0], names[k]
            motors = []
            if kind == 'car' and c.get('drive', 'rear') != 'off':
                driven = A.drive if c.get('drive', 'rear') == 'all' else [d for d in A.drive if A.bodies[d[0]].at[0] < 0.0]
                for kb, radius in driven:
                    a = spec.add_actuator()
                    a.name = f'asm{i}_{kb}_drive'
                    a.trntype = mujoco.mjtTrn.mjTRN_JOINT
                    a.target = f'asm{i}_{kb}_j'
                    a.set_to_velocity(kv=CAR_TORQUE / 2.0)
                    a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
                    a.forcerange = [-CAR_TORQUE, CAR_TORQUE]
                    a.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
                    motors.append(('drive', a.name, radius))
            if kind == 'car':
                for kb in A.steer:
                    a = spec.add_actuator()
                    a.name = f'asm{i}_{kb}_steer'
                    a.trntype = mujoco.mjtTrn.mjTRN_JOINT
                    a.target = f'asm{i}_{kb}_j'
                    a.set_to_position(kp=4000.0, kv=200.0)
                    a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
                    a.forcerange = [-2000.0, 2000.0]
                    a.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
                    motors.append(('steer', a.name, 0.0))
            out.append(dict(index=i, kind=kind, A=A, names=names, joints=joints, motors=motors, limp=limp, size=tuple(cg.size),
                            throw=np.asarray(c.get('start_velocity', (0.0, 0.0, 0.0)), float),
                            keyed=any(scene.curve(('collider', i, key)) is not None for key in ('drive_speed', 'steer'))))
        return out

    def _index_assemblies(self, asms):
        """Each person's and car's body, joint, dof and motor ids, once the model is built; a car keyed to a speed starts
        rolling at it."""
        import mujoco
        m, d = self.model, self.data
        from .assemblies import kind_of as _k
        del _k
        for a in asms:
            a['bodies'] = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in a['names']], np.int64)
            a['jids'] = [(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n), tone) for _k2, n, tone in a['joints']]
            a['acts'] = [(kind, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n), radius) for kind, n, radius in a['motors']]
            a['part_body'] = a['bodies'][np.asarray(a['A'].piece_body, np.int64)]
            a['part_off'] = np.array([pc.centroid - a['A'].bodies[kb].at for pc, kb in zip(a['A'].pieces, a['A'].piece_body)])
            root = int(a['bodies'][0])
            adr = int(m.jnt_dofadr[m.body_jntadr[root]])
            d.qvel[adr:adr + 3] += a['throw']
            if a['kind'] == 'figure' and a['limp']:
                # (limp, no one balances: a little sway, about a level axis, to fall by)
                rng = np.random.default_rng(a['index'] + 11)
                th = rng.uniform(0.0, 2.0 * math.pi)
                d.qvel[adr + 3:adr + 6] += 0.4 * np.array([math.cos(th), 0.0, math.sin(th)])
        self.asms = asms
        self._asm_body = {int(b): n for n, a in enumerate(asms) for b in a['bodies']}

    def _assemblies_step(self, scene, frame):
        """People and cars at this step: a car's motors set to its keyed speed and steering (0 km/h brakes it)."""
        if not self.asms:
            return
        d = self.data
        for a in self.asms:
            if a['kind'] != 'car' or not a['acts']:
                continue
            i = a['index']
            v = float(scene.get(('collider', i, 'drive_speed'), frame)) / 3.6
            steer = math.radians(float(scene.get(('collider', i, 'steer'), frame)))
            for kind, aid, radius in a['acts']:
                d.ctrl[aid] = v / max(radius, 1e-3) if kind == 'drive' else steer

    def _assemblies_hit(self, q0):
        """A standing person hit hard (a contact closing on one of its parts faster than LIMP_SPEED, from anything
        not of it) goes limp: its joints lose their bracing and it falls as a body does."""
        from .assemblies import LIMP, LIMP_SPEED
        if not self.asms or q0 is None or not any(a['kind'] == 'figure' and not a['limp'] for a in self.asms):
            return
        m, d = self.model, self.data
        nc = int(d.ncon)
        if nc == 0:
            return
        b1 = np.asarray(m.geom_bodyid[d.contact.geom1[:nc]], np.int64)
        b2 = np.asarray(m.geom_bodyid[d.contact.geom2[:nc]], np.int64)
        own1 = np.array([self._asm_body.get(int(b), -1) for b in b1])
        own2 = np.array([self._asm_body.get(int(b), -1) for b in b2])
        sel = np.nonzero(((own1 >= 0) | (own2 >= 0)) & (own1 != own2))[0]
        if not len(sel):
            return
        speed = self._closing(q0, b1[sel], b2[sel], np.asarray(d.contact.pos[sel], float),
                              np.asarray(d.contact.frame[sel, :3], float))
        for k, v in zip(sel, speed):
            if v < LIMP_SPEED:
                continue
            for n in (own1[k], own2[k]):
                a = self.asms[n] if n >= 0 else None
                if a is not None and a['kind'] == 'figure' and not a['limp']:
                    self._go_limp(a, LIMP)

    def _go_limp(self, a, tones):
        m = self.model
        for jid, tone in a['jids']:
            if tone:
                stiff, damp = tones[tone]
                m.jnt_stiffness[jid] = stiff
                adr, n = int(m.jnt_dofadr[jid]), 3 if m.jnt_type[jid] == 1 else 1
                m.dof_damping[adr:adr + n] = damp
        a['limp'] = True

    def assembly_poses(self):
        """People's and cars' parts as they are now, in piece_poses' form."""
        import mujoco
        m, d = self.model, self.data
        out = {}
        v6 = np.zeros(6)
        for a in self.asms:
            bs = a['part_body']
            R = d.xmat[bs].reshape(-1, 3, 3)
            pos = d.xpos[bs] + np.einsum('bij,bj->bi', R, a['part_off'])
            q = d.xquat[bs]
            vel, om = np.zeros((len(bs), 3)), np.zeros((len(bs), 3))
            for n, b in enumerate(bs):
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, int(b), v6, 0)
                om[n] = v6[:3]
                vel[n] = v6[3:] + np.cross(v6[:3], pos[n] - d.xpos[b])
            out[a['index']] = dict(pos=pos, quat=np.concatenate([q[:, 1:], q[:, :1]], 1), vel=vel, omega=om,
                                   size=np.asarray(a['size'], float), hollow=np.float32(0.0))
        return out

    @staticmethod
    def _post_for(scene, by_index, free, i, j, P, B, L):
        """A post or a ball a rope from P to B (L long) may wrap round: the cylinder or sphere (not either end's
        object) whose middle passes nearest its line, within its radius and the rope's slack: (geom name, centre, axis,
        radius) or None. (MuJoCo's tendon goes round it only while it is in the way.)"""
        d = B - P
        ln = float(np.linalg.norm(d))
        if ln < 1e-6:
            return None
        u = d / ln
        slack = 0.5 * max(L - ln, 0.0) + 0.05
        best = None
        up = np.array([0.0, 1.0, 0.0])
        for n, c in enumerate(scene.colliders):
            if n in (i, j) or not c['enabled'] or n not in by_index or c['shape'] not in ('cylinder', 'sphere'):
                continue
            if float(c.get('hollow', 0.0) or 0.0) > 0.0 or breaks(c) or kind_of(c):
                continue
            cg = by_index[n]
            ctr = np.asarray(cg.pos, float)
            rad = float(abs(cg.size[0]))
            ax = q_rot(xyzw(turn_wxyz(cg))) @ np.array([0.0, 1.0, 0.0])
            t = float((ctr - P) @ u)
            dh = np.array([d[0], 0.0, d[2]])
            th = float((ctr - P) @ dh) / max(float(dh @ dh), 1e-12) if float(np.linalg.norm(dh)) > 0.2 * ln else -1.0
            if not (0.05 * ln < t < 0.95 * ln or 0.05 < th < 0.95):     # (between its ends, along it or across the ground)
                continue
            t = float(np.clip(t, 0.0, ln))
            off = P + u * t - ctr
            if c['shape'] == 'cylinder':
                off = off - ax * float(off @ ax)
                if abs(float((P + u * t - ctr) @ ax)) > abs(float(cg.size[1])):
                    continue
            gap = float(np.linalg.norm(off)) - rad
            if gap < slack and (best is None or gap < best[0]):
                name = f'bodygeom{n}' if n in free else f'fixed{n}_0'
                # the side it goes round: away from the line (a line through it: over its top, or any way round)
                away = -off
                if c['shape'] == 'cylinder':
                    away = away - ax * float(away @ ax)
                if np.linalg.norm(away) < 1e-3 * rad:
                    away = up - (ax * float(up @ ax) if c['shape'] == 'cylinder' else 0.0)
                    if np.linalg.norm(away) < 1e-3:
                        away = np.cross(ax, u)
                away = away / max(float(np.linalg.norm(away)), 1e-12)
                R = q_rot(xyzw(turn_wxyz(cg)))
                best = (gap, (name, ctr, ax if c['shape'] == 'cylinder' else None, rad, free[n][0] if n in free else None,
                              away, R))
        return None if best is None else best[1]

    def _rehearsal_parts(self, spec, i, c, cg):
        """A breakable's pieces as one thing's shapes (a rehearsal): (shape, half size, centre, mesh) in its own frame."""
        from .fracture import inset
        out = []
        for n, p in enumerate(fractured(c, cg.size, cg.hollow, None, self._scene).pieces):
            v = inset(p, GAP)
            box = _box_of(p)
            if box is not None:
                lo, hi = v.min(0), v.max(0)
                out.append(('box', 0.5 * (hi - lo), 0.5 * (lo + hi), None))
            else:
                ma = spec.add_mesh()
                ma.name = f'rehearse{i}_{n}'
                ma.uservert = (v - p.centroid).ravel().tolist()
                out.append(('mesh', None, p.centroid, ma.name))
        return out

    def _rehearse(self, scene, idx, layout):
        """Where each breakable thing is hit hardest, found by running the shot once with nothing breaking (each
        breakable one rigid body of all its pieces, or fixed where it stands): its own frame, {collider index: point}.
        Its cracks then crowd round where it is really hit: a pot dropped on its rim shatters at the rim, a pane where
        the ball strikes it. Kept, for the same scene. (Only the bodies: what a liquid, sand or a fire does to them is
        not rehearsed.)"""
        want = [i for i in idx if breaks(scene.colliders[i]) and uses_impact(scene.colliders[i])]
        if not want:
            return {}
        import time as _time
        dd = scene.data['domain']
        key = (self._key(scene, idx, layout), scene.start, scene.end, float(scene.fps), float(dd['preroll']),
               str(dd.get('time_scale')))
        got = _HITS.get(key)
        if got is not None:
            return dict(got)
        r = Solids()
        r.rehearsal = True
        out = {}
        try:
            r._build(scene, idx, layout)
            fps = scene.fps
            first = scene.start - int(round(float(dd['preroll']) * fps))
            t_wall, t_sim = _time.perf_counter(), 0.0
            for frame in range(first, scene.end + 1):
                fdt = float(scene.v('domain', 'time_scale', frame)) / fps
                r.advance(scene, frame, fdt, 1)
                t_sim += fdt
                if t_sim > REHEARSE_S or _time.perf_counter() - t_wall > REHEARSE_WALL:
                    break
            out = {i: np.asarray(v[0], float) for i, v in r.hits.items()}
        except Exception as ex:     # (a rehearsal that fails only loses where things are hit)
            self.warnings.append(f'where things are hit: {ex}')
        if len(_HITS) > 16:
            _HITS.clear()
        _HITS[key] = out
        return dict(out)

    def _crack_on(self, hit):
        """The crack runs on from the pieces a hit breaks away: to each piece still glued to one, the hit carries
        brittle^(distance / the set's piece spacing) of its strength (materials brittle: glass and pottery shatter right
        through, wood splits off where it is hit), and breaks it away too while that is past what holds it. The
        pieces that go."""
        I, W = self._imp, self._w
        out = set(int(p) for p in hit)
        todo = list(out)
        while todo:
            p = todo.pop()
            k = float(I['brittle'][p])
            if k <= 0.0:
                continue
            for q, r in I['nb'][p]:
                if q in out or not I['on'][q] or W['over'][r] < 0 or I['v_break'][q] <= 0.0:
                    continue
                carried = I['acc'][p] * k ** (float(np.linalg.norm(I['cen'][p] - I['cen'][q])) / I['ell'][p])
                if carried > I['acc'][q]:
                    I['acc'][q] = carried
                if I['acc'][q] > max(I['v_break'][q], IMPACT_FLOOR):
                    out.add(q)
                    todo.append(q)
        return sorted(out)

    def _impact_sync(self):
        """The impact check's state after a reset or a load: no hit under way; a piece still held if any weld holds it."""
        I, W = self._imp, self._w
        if I is None or W is None:
            return
        I['acc'][:] = 0.0
        I['on'][:] = [bool((W['over'][r] >= 0).any()) if len(r) else False for r in I['holds']]

    def _piece_v(self):
        """The velocities now (the impact check's before): every degree of freedom's."""
        return None if self._imp is None and not self.rehearsal and not self.asms else self.data.qvel.copy()

    def _bodies_table(self):
        """Per body, for how fast contacts close (_closing): its free joint's first dof (-1: none), its mocap id (-1:
        none), whether it moves on other joints, and its mass as what it hits feels it (the world, fixed and keyed
        things endless)."""
        if self._btab is None:
            import mujoco
            m = self.model
            nb = int(m.nbody)
            free = np.full(nb, -1, np.int64)
            other = np.zeros(nb, bool)
            heavy = np.full(nb, np.inf)
            for b in range(1, nb):
                if m.body_mocapid[b] >= 0 or m.body_jntnum[b] == 0:
                    continue
                heavy[b] = float(m.body_mass[b])
                j = int(m.body_jntadr[b])
                if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    free[b] = int(m.jnt_dofadr[j])
                else:
                    other[b] = True
            self._btab = (free, np.asarray(m.body_mocapid, np.int64).copy(), other, heavy)
        return self._btab

    def _closing(self, q0, b1, b2, pos, nrm):
        """How fast each contact closed just before this step (q0: qvel then): bodies b1, b2 (arrays) at points pos,
        normal nrm from the first toward the second. A free body from q0; a keyed one at its keys' speed; one on a
        hinge or a ball joint as it moves now; the world and fixed things not at all."""
        import mujoco
        m, d = self.model, self.data
        free, mocap, other, _heavy = self._bodies_table()
        u = []
        for b in (b1, b2):
            v = np.zeros((len(b), 3))
            f = free[b]
            k = f >= 0
            if k.any():
                a = f[k][:, None]
                lin = q0[a + np.arange(3)]
                w = np.einsum('bij,bj->bi', d.xmat[b[k]].reshape(-1, 3, 3), q0[a + 3 + np.arange(3)])
                v[k] = lin + np.cross(w, pos[k] - d.xpos[b[k]])
            for n in np.nonzero(mocap[b] >= 0)[0]:
                v[n] = self._mocap_vel.get(int(mocap[b[n]]), (0.0, 0.0, 0.0))
            for n in np.nonzero(other[b])[0]:
                v6 = np.zeros(6)
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, int(b[n]), v6, 0)
                v[n] = v6[3:] + np.cross(v6[:3], pos[n] - d.xpos[b[n]])
            u.append(v)
        return np.einsum('ij,ij->i', u[0] - u[1], nrm)

    def _impact(self, v0, h):
        """Break away a piece hit hard: how fast a contact on it closed (the speed it and what it met came together
        at, along the contact's normal, just before; scaled down by how much lighter the other is than it, so a grain
        of sand does not crack a pane), or the change in its own velocity over this step along the normal (what the
        contact did to it: gravity's part taken out; resting, carrying a load or pushed steadily it hardly changes),
        added up over the few steps a hit lasts, past its material's impact speed (materials: a hit's stress wave,
        density x sound speed x that speed, reaching what holds it). All its welds go: a dropped pot's rim, the glass
        a stone goes through, a wall's bricks where it lands. (The welds' forces alone miss a thing stopped all at
        once, a slab landing flat, every piece stopped together, and a pane's glass the stone strikes, held fast by
        the rest of the pane.)"""
        I = self._imp
        if I is None or v0 is None:
            return
        import mujoco
        m, d = self.model, self.data
        I['acc'] *= IMPACT_KEEP
        nc = int(d.ncon)
        if nc == 0:
            return
        g1 = np.asarray(d.contact.geom1[:nc], np.int64)
        g2 = np.asarray(d.contact.geom2[:nc], np.int64)
        nrm = np.asarray(d.contact.frame[:nc, :3], float)       # from geom1 toward geom2
        p1 = I['piece'][m.geom_bodyid[g1]]
        p2 = I['piece'][m.geom_bodyid[g2]]
        if not ((p1 >= 0).any() or (p2 >= 0).any()):
            return
        dv = (d.qvel[I['lin']] - v0[I['lin']]).reshape(-1, 3) - np.asarray(m.opt.gravity, float) * h
        step = np.zeros(len(I['acc']))
        for pp, sign in ((p1, -1.0), (p2, 1.0)):      # (pushed back from what it met: geom1 along -n, geom2 along +n)
            sel = pp >= 0
            if sel.any():
                dvn = sign * np.einsum('ij,ij->i', dv[pp[sel]], nrm[sel])
                np.maximum.at(step, pp[sel], dvn)
        # how fast each contact on a piece closed, against how much lighter what it met is
        on = np.nonzero((p1 >= 0) | (p2 >= 0))[0]
        if len(on):
            b1 = np.asarray(m.geom_bodyid[g1[on]], np.int64)
            b2 = np.asarray(m.geom_bodyid[g2[on]], np.int64)
            closing = self._closing(v0, b1, b2, np.asarray(d.contact.pos[on], float), nrm[on])
            heavy = self._bodies_table()[3]
            with np.errstate(invalid='ignore', divide='ignore'):
                for pp, me, other in ((p1[on], b1, b2), (p2[on], b2, b1)):
                    sel = (pp >= 0) & (closing > IMPACT_FLOOR)
                    if sel.any():
                        light = np.nan_to_num(np.minimum(1.0, heavy[other[sel]] / np.maximum(heavy[me[sel]], 1e-9)), nan=1.0)
                        np.maximum.at(step, pp[sel], closing[sel] * light)
        I['acc'] += step
        hit = np.nonzero(I['on'] & (I['v_break'] > 0.0) & (I['acc'] > np.maximum(I['v_break'], IMPACT_FLOOR)))[0]
        if not len(hit):
            return
        W = self._w
        struck = set(int(p) for p in hit)
        hit = self._crack_on(hit)
        cracked = set(hit)
        for p in hit:
            I['on'][p] = False
            rows = I['holds'][p]
            rows = rows[W['over'][rows] >= 0]          # (not broken already)
            if p not in struck:
                # one the crack ran to comes away from the others it ran through, and stays glued to what it did not
                # reach and held where it is held (a pane's frame): the shattered zone's edge holds
                a, b = I['ends']
                rows = np.asarray([r for r in rows if b[r] >= 0 and int(a[r] + b[r] - p) in cracked], np.int64)
            if not len(rows):
                continue
            d.eq_active[W['eq'][rows]] = 0
            W['over'][rows] = -(1 << 40)
            for r in rows:
                self.breaks.append((self.time, d.xpos[W['body'][r]].copy(), float(W['area'][r]), int(W['collider'][r])))

    def _build_joints(self, scene, spec, by_index, bodies, sets, mocap, k, dt):
        """The objects' ropes, springs, hinges and ball joints (Properties › Joint): sites on the two bodies, and a
        tendon or connect constraints between them. Returns the Joints (their MuJoCo ids are found once compiled)."""
        import mujoco
        from .ropes import CABLE, CHAIN, ROPE, SPRING, rope_curve
        out = []
        free = {bd.index: (f'body{n}', bd) for n, bd in enumerate(bodies)}
        pieces = {ps.index: ps for ps in sets}
        by_name = {}
        for j, c in enumerate(scene.colliders):
            by_name.setdefault(c['name'], j)
        for j, c in enumerate(scene.colliders):
            by_name.setdefault(c['name'].strip().lower(), j)
        world = ('', False, np.zeros(3), np.eye(3))

        def turn(cg):
            return q_rot(xyzw(_quat_mul_wxyz(wxyz(cg.quat), _yaw_wxyz(float(cg.rot_y)))))

        def mass(j):
            c = scene.colliders[j]
            if j in free:
                return free[j][1].density * free[j][1].volume
            return resolved(c)['density'] * shape_volume(c['shape'], by_index[j].size)

        def carrier(j, near):
            """The MuJoCo body that carries collider j at world point `near`, and where it starts: (name ('': the
            world), whether it moves freely, its position, its rotation). A breakable's is the piece nearest."""
            cg = by_index[j]
            pos, R = np.asarray(cg.pos, float), turn(cg)
            if j in free:
                return free[j][0], True, pos, R
            if j in pieces:
                ps = pieces[j]
                cents = pos + np.array([p.centroid for p in ps.frac.pieces]) @ R.T
                n = int(np.argmin(np.linalg.norm(cents - near, axis=1)))
                return ps.names[n], True, cents[n], R
            if j in mocap:
                return f'mocap{j}', False, pos, R
            return world

        def site(on, P, name):
            body, _f, bpos, bR = on
            s = (spec.body(body) if body else spec.worldbody).add_site()
            s.name = name
            s.pos = list(map(float, bR.T @ (np.asarray(P, float) - bpos)))
            return name

        for i in sorted(set(free) | set(pieces)):
            c = scene.colliders[i]
            kind = joined(c)
            if kind is None or i not in by_index:
                continue
            cg = by_index[i]
            R = turn(cg)
            to = str(c.get('joint_to', '') or '').strip()
            j = None
            if to:
                j = by_name.get(to, by_name.get(to.lower()))
                if j is None or j == i or not scene.colliders[j]['enabled'] or j not in by_index:
                    why = 'itself' if j == i else (f'{to}, which is turned off' if j is not None else f'{to}: there is no such object')
                    self.warnings.append(f'{c["name"]} is joined to {why}; it is held by a fixed point instead.')
                    j = None
            hull = free[i][1].hull if i in free else None
            P, B = joint_ends(c, cg, by_index[j] if j is not None else None, hull)
            me = carrier(i, P)
            ot = carrier(j, B) if j is not None else world
            if kind in ('hinge', 'ball') and i in free and not ot[1]:
                free[i][1].pivot = P.copy()
            tag = f'joint{i}'
            jt = Joint(index=i, kind=kind, body=me[0], other=ot[0], other_free=ot[1],
                       strength=float(c.get('joint_break', 0.0) or 0.0), friction=float(c.get('joint_friction', 0.2)),
                       motor=float(scene.get(('collider', i, 'motor_speed'), scene.start)) * RPM if kind == 'hinge' else 0.0,
                       torque=float(c.get('motor_torque', 0.0) or 0.0),
                       keyed=kind == 'hinge' and scene.curve(('collider', i, 'motor_speed')) is not None,
                       radius=0.5 * float(c.get('rope_thickness', 0.025)),
                       look=SPRING if kind == 'spring' else {'cable': CABLE, 'chain': CHAIN}.get(c.get('rope_look'), ROPE))
            if kind == 'rope' and jt.look == CHAIN:
                # a chain: steel links, each a body on a ball joint to the last, from the far end (held there) to it
                dist = float(np.linalg.norm(B - P))
                L = float(c.get('rope_length', 0.0) or 0.0) or dist
                wire = 2.0 * jt.radius
                n_links = int(np.clip(round(L / (5.0 * wire)), 4, 60))
                pts = rope_curve(B, P, L, n_links)
                per_m = CHAIN_KG_M2 * wire * wire
                load = mass(i) + (mass(j) if j is not None and ot[1] else 0.0)
                prev, names, halves = None, [], []
                for kl in range(n_links):
                    a_, b_ = pts[kl], pts[kl + 1]
                    seg = float(np.linalg.norm(b_ - a_))
                    mid = 0.5 * (a_ + b_)
                    parent = spec.worldbody if prev is None else prev
                    lb = parent.add_body()
                    lb.name = f'{tag}_link{kl}'
                    lb.pos = list(map(float, mid if prev is None else mid - prev_mid))
                    j_ = lb.add_joint() if (prev is not None or not ot[1] and not ot[0]) else None
                    if prev is None and j_ is None:
                        lb.add_freejoint()
                    if j_ is not None:
                        j_.type = mujoco.mjtJoint.mjJNT_BALL
                        j_.pos = list(map(float, a_ - mid))
                        j_.damping = [0.002 * per_m * seg, 0.0, 0.0]
                        # (a light link between heavy ends blows up in MuJoCo: rotational inertia in proportion to
                        # what it carries keeps it in hand; a 200 kg weight on 12 mm chain was unstable at once)
                        j_.armature = max(1e-4, 0.6 * load * seg * seg)
                    g_ = lb.add_geom()
                    g_.type = mujoco.mjtGeom.mjGEOM_CAPSULE
                    g_.fromto = [*map(float, a_ - mid), *map(float, b_ - mid)]
                    g_.size = [1.6 * wire, 0.0, 0.0]
                    g_.mass = max(per_m * seg, 1e-3)
                    g_.priority = 1
                    g_.friction = [0.5, 0.005, 0.0001]                  # (steel links: as contact() sets a body's)
                    g_.solref = [-k, -2.0 * damping_ratio(0.1) * math.sqrt(k)]
                    prev, prev_mid = lb, mid
                    names.append(lb.name)
                    halves.append(a_ - mid)
                # held at the far end (on another thing: glued to it there) and at this end to the object
                e1 = spec.add_equality()
                e1.type = mujoco.mjtEq.mjEQ_CONNECT
                e1.objtype = mujoco.mjtObj.mjOBJ_BODY
                e1.name1 = names[-1]
                e1.name2 = me[0]
                d1 = np.array(e1.data, float)
                d1[0:3] = pts[-1] - 0.5 * (pts[-2] + pts[-1])
                e1.data = d1
                e1.name = f'{tag}_end'
                if me[0]:
                    ex = spec.add_exclude()
                    ex.bodyname1, ex.bodyname2 = names[-1], me[0]
                if ot[0]:
                    ex = spec.add_exclude()
                    ex.bodyname1, ex.bodyname2 = names[0], ot[0]
                    e2 = spec.add_equality()
                    e2.type = mujoco.mjtEq.mjEQ_CONNECT
                    e2.objtype = mujoco.mjtObj.mjOBJ_BODY
                    e2.name1 = names[0]
                    e2.name2 = ot[0]
                    d2 = np.array(e2.data, float)
                    d2[0:3] = pts[0] - 0.5 * (pts[0] + pts[1])
                    e2.data = d2
                    e2.name = f'{tag}_start'

                jt.links, jt.length, jt.link_half = names, L, np.asarray(halves, float)
                out.append(jt)
                continue
            if kind in ('rope', 'spring'):
                sa, sb = site(me, P, f'{tag}a'), site(ot, B, f'{tag}b')
                dist = float(np.linalg.norm(B - P))
                L = float(c.get('rope_length', 0.0) or 0.0)
                L = L if L > 0.0 else dist
                t = spec.add_tendon()
                t.name = tag
                t.wrap_site(sa)
                reach = float(c.get('rope_length', 0.0) or 0.0) or float(np.linalg.norm(B - P))
                post = self._post_for(scene, by_index, free, i, j, P, B, max(reach, 1e-3))
                if kind == 'rope' and post is not None:
                    # round a post or a ball in its way, on the side away from the straight line between its ends
                    # (over the top of a post it is draped over): a side site there, on the post's body
                    side = spec.worldbody if post[4] is None else spec.body(post[4])
                    ss = side.add_site()
                    ss.name = f'{tag}side'
                    at = post[1] + post[5] * (1.5 * post[3])
                    ss.pos = list(map(float, at if post[4] is None else post[6].T @ (at - post[1])))
                    t.wrap_geom(post[0], ss.name)
                    jt.post = post[:4]
                t.wrap_site(sb)
                if kind == 'rope':
                    t.limited = mujoco.mjtLimited.mjLIMITED_TRUE
                    t.range = [0.0, max(L, 1e-4)]
                    t.solref_limit = [-k, -2.0 * damping_ratio(ROPE_BOUNCE) * math.sqrt(k)]
                    if L < 0.98 * dist:
                        self.warnings.append(f'{c["name"]}: its rope ({L:.2f} m) is shorter than the {dist:.2f} m to where it is '
                                             f'tied: it is yanked in at the start.')
                else:
                    m1 = mass(i)
                    mu = m1 * mass(j) / (m1 + mass(j)) if (j is not None and ot[1]) else m1
                    ks = float(c.get('spring_k', 500.0))
                    most = mu * (0.5 / dt) ** 2       # (stiffer than the step can follow)
                    if ks > most:
                        self.warnings.append(f'{c["name"]}: its spring is too stiff for something so light; it is '
                                             f'{most:.0f} N/m instead.')
                        ks = most
                    t.stiffness = [ks, 0.0, 0.0]
                    t.springlength = [L, L]
                    t.damping = [2.0 * SPRING_DAMPING * math.sqrt(ks * mu), 0.0, 0.0]
                jt.sites, jt.tendon, jt.length = (sa, sb), tag, L
            else:
                pins = [P]
                if kind == 'hinge':
                    ax = np.asarray(c.get('joint_axis', (0.0, 1.0, 0.0)), float)
                    na = float(np.linalg.norm(ax))
                    ax = ax / na if na > 1e-9 else np.array([0.0, 1.0, 0.0])
                    half = max(extent_along(c['shape'], cg.size, ax, hull), HINGE_SPAN)
                    axw = R @ ax
                    pins = [P + half * axw, P - half * axw]
                    jt.axis = me[3].T @ axw
                sites, eqs = [], []
                for n, Q in enumerate(pins):
                    s1, s2 = site(me, Q, f'{tag}p{n}'), site(ot, Q, f'{tag}q{n}')
                    e = spec.add_equality()
                    e.type = mujoco.mjtEq.mjEQ_CONNECT
                    e.objtype = mujoco.mjtObj.mjOBJ_SITE
                    e.name = f'{tag}e{n}'
                    e.name1, e.name2 = s1, s2
                    e.solref = [-k, -2.0 * math.sqrt(k)]
                    sites += [s1, s2]
                    eqs.append(e.name)
                jt.sites, jt.eqs = tuple(sites), tuple(eqs)
            out.append(jt)
        return out

    def _break_joints(self):
        """Snap the ropes and springs pulled, and tear out the hinges and ball joints loaded, past their Breaks at."""
        import mujoco
        live = [jt for jt in self.joints if jt.strength > 0.0 and not jt.broken and not jt.links]   # (a chain does not snap)
        if not live:
            return
        m, d = self.model, self.data
        n = d.nefc
        typ, ids, F = d.efc_type[:n], d.efc_id[:n], d.efc_force[:n]
        for jt in live:
            if jt.kind == 'rope':
                f = float(F[(typ == mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON) & (ids == jt.tid)].sum())
            elif jt.kind == 'spring':
                f = abs(float(m.tendon_stiffness[jt.tid]) * (float(d.ten_length[jt.tid]) - jt.length))
            else:
                eq = typ == mujoco.mjtConstraint.mjCNSTR_EQUALITY
                f = sum(float(np.linalg.norm(F[eq & (ids == e)])) for e in jt.eids)
            if f > jt.strength:
                self._snap(jt)
                self.snaps.append((self.time, jt.index))

    def _snap(self, jt):
        """Joint jt is broken from now on."""
        m, d = self.model, self.data
        if jt.kind == 'rope':
            m.tendon_limited[jt.tid] = 0
        elif jt.kind == 'spring':
            m.tendon_stiffness[jt.tid] = 0.0
            m.tendon_damping[jt.tid] = 0.0
        else:
            d.eq_active[list(jt.eids)] = 0
        jt.broken = True

    def _unsnap_all(self):
        """Every joint whole again (as built)."""
        if self._tendon0 is not None and self.model is not None:
            m = self.model
            m.tendon_limited[:], m.tendon_stiffness[:], m.tendon_damping[:] = self._tendon0
        for jt in self.joints:
            jt.broken = False

    def rope_poses(self):
        """The ropes and springs as they are now, for drawing (ropes.rope_points): {collider index: dict(a, b (3,): its
        ends, on the object and at the other end, va, vb (3,): their velocities, length (m; a spring's at rest),
        radius (m), look (ropes.ROPE, CABLE, SPRING), broken (0 or 1))}."""
        import mujoco
        out = {}
        if self.data is None:
            return out
        m, d = self.model, self.data
        v6 = np.zeros(6)
        for jt in self.joints:
            if jt.kind not in ('rope', 'spring'):
                continue
            if jt.links:
                # (its links end to end, from the far end to the object: where each starts, and where the last ends)
                ids = jt.link_ids
                R = d.xmat[ids].reshape(-1, 3, 3)
                h = np.einsum('bij,bj->bi', R, jt.link_half)
                path = np.concatenate([d.xpos[ids] + h, (d.xpos[ids[-1]] - h[-1])[None]])
                still = np.zeros(3)
                out[jt.index] = dict(a=path[-1].copy(), va=still, b=path[0].copy(), vb=still, length=np.float32(jt.length),
                                     radius=np.float32(jt.radius), look=np.float32(jt.look), broken=np.float32(jt.broken),
                                     path=path[::-1].copy())
                continue
            ends = []
            for s in jt.sids[:2]:
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, s, v6, 0)
                ends += [d.site_xpos[s].copy(), v6[3:].copy()]
            out[jt.index] = dict(a=ends[0], va=ends[1], b=ends[2], vb=ends[3], length=np.float32(jt.length),
                                 radius=np.float32(jt.radius), look=np.float32(jt.look), broken=np.float32(jt.broken))
            if getattr(jt, 'post', None) is not None and jt.tid >= 0 and not jt.broken:
                path = self._wrapped(jt)
                if path is not None:
                    out[jt.index]['path'] = path
        return out

    def _wrapped(self, jt):
        """A rope's path round the post it wraps (MuJoCo's tendon path: its ends and where it meets and leaves the
        post), the arc between those drawn round the post's side: points (m, 3), or None while it runs straight."""
        m, d = self.model, self.data
        adr, num = int(d.ten_wrapadr[jt.tid]), int(d.ten_wrapnum[jt.tid])
        pts = np.asarray(d.wrap_xpos[adr:adr + num], float).reshape(-1, 3)
        if num < 4:
            return None
        # (the tendon's points: an end, the post's two tangent points, the other end; bodies move the post)
        _name, ctr0, ax0, rad = jt.post
        gid = int(m.geom(_name).id)
        ctr = d.geom_xpos[gid].copy()
        ax = d.geom_xmat[gid].reshape(3, 3)[:, 2] if ax0 is not None else None
        p1, p2 = pts[1], pts[2]
        r = rad + jt.radius
        a1, a2 = p1 - ctr, p2 - ctr
        if ax is not None:
            h1, h2 = float(a1 @ ax), float(a2 @ ax)
            a1, a2 = a1 - ax * h1, a2 - ax * h2
        n1, n2 = a1 / max(np.linalg.norm(a1), 1e-9), a2 / max(np.linalg.norm(a2), 1e-9)
        ang = math.acos(float(np.clip(n1 @ n2, -1.0, 1.0)))
        k = max(2, int(math.ceil(ang / 0.15)))
        arc = []
        for s in np.linspace(0.0, 1.0, k + 1):
            # (slerp round the post, the height along a cylinder's axis going linearly)
            w = math.sin((1 - s) * ang) / max(math.sin(ang), 1e-9), math.sin(s * ang) / max(math.sin(ang), 1e-9)
            dirn = n1 * w[0] + n2 * w[1] if ang > 1e-4 else n1
            q = ctr + dirn / max(np.linalg.norm(dirn), 1e-9) * r
            if ax is not None:
                q = q + ax * ((1 - s) * h1 + s * h2)
            arc.append(q)
        return np.concatenate([pts[:1], np.asarray(arc), pts[3:4]])

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
            W['over'][W['over'] > 0] = 0        # (a broken one stays broken)
            return
        eqm = d.efc_type[:n] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        if not eqm.any():
            return
        ids = d.efc_id[:n][eqm]
        weld = W['lookup'][ids] >= 0      # (a hinge's or a ball joint's connect constraints have 3 rows: _break_joints)
        if not weld.any():
            return
        F = d.efc_force[:n][eqm][weld].reshape(-1, 6)
        rows = W['lookup'][ids[weld].reshape(-1, 6)[:, 0]]
        R = d.xmat[W['body'][rows]].reshape(-1, 3, 3)
        nw = np.einsum('bij,bj->bi', R, W['normal'][rows])
        f = F[:, :3]
        fn = (f * nw).sum(1)                                   # + pulls it apart (on the first piece, toward the other)
        fs = np.linalg.norm(f - fn[:, None] * nw, axis=1)
        mt = np.linalg.norm(F[:, 3:], axis=1)
        A = W['area'][rows]
        sig = W['strength'][rows]
        sy = W['yield'][rows]
        if (sy > 0.0).any():
            # metal and plastic: loaded past its yield (and short of its strength) a joint bends and stays bent
            give = (sy > 0.0) & ((np.abs(fn) > sy * A) | (fs > sy * A) | (mt > sy * A * np.sqrt(A) / 6.0))
            if give.any():
                self._yield(rows[give])
        over = (fn > sig * A) | (fs > sig * A + JOINT_FRICTION * np.maximum(-fn, 0.0)) | (mt > sig * A * np.sqrt(A) / 6.0)
        hit = np.zeros(len(W['over']), bool)
        hit[rows[over]] = True
        W['over'][hit] += 1
        W['over'][~hit & (W['over'] > 0)] = 0   # (a broken one stays broken)
        broke = np.nonzero(W['over'] >= BREAK_STEPS)[0]
        if len(broke):
            d.eq_active[W['eq'][broke]] = 0
            W['over'][broke] = -(1 << 40)       # never again
            for b in broke:
                self.breaks.append((self.time, d.xpos[W['body'][b]].copy(), float(W['area'][b]), int(W['collider'][b])))

    def _yield(self, rows):
        """Welds forced past their yield: each takes the pose its two pieces are in now as its own (it bends, as metal
        does, and stays bent), and tears once it has bent past its ductility in all (with how far it has stretched over
        its size)."""
        import mujoco
        W, m, d = self._w, self.model, self.data
        torn = []
        for r in rows:
            e = int(W['eq'][r])
            b1, b2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
            R1, R2 = d.xmat[b1].reshape(3, 3), d.xmat[b2].reshape(3, 3)
            rel = R1.T @ (d.xpos[b2] + R2 @ m.eq_data[e, 0:3] - d.xpos[b1])
            q1 = np.zeros(4)
            mujoco.mju_negQuat(q1, d.xquat[b1])
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, q1, d.xquat[b2])
            old = m.eq_data[e, 3:10].copy()
            turn = 2.0 * math.acos(min(1.0, abs(float(np.dot(q, old[3:7])))))
            stretch = float(np.linalg.norm(rel - old[0:3])) / max(math.sqrt(float(W['area'][r])), 1e-4)
            m.eq_data[e, 3:6] = rel
            m.eq_data[e, 6:10] = q
            W['plastic'][r] += turn + stretch
            self.sets[int(W['set'][r])].bent = True
            if W['plastic'][r] > W['duct'][r]:
                torn.append(r)
        if torn:
            torn = np.asarray(torn, np.int64)
            d.eq_active[W['eq'][torn]] = 0
            W['over'][torn] = -(1 << 40)
            for r in torn:
                self.breaks.append((self.time, d.xpos[W['body'][r]].copy(), float(W['area'][r]), int(W['collider'][r])))

    def _restance(self, limp=None):
        """Every person back to its stance as built (or, limp: which have gone limp, from a saved state)."""
        from .assemblies import LIMP, STAND
        for n, a in enumerate(self.asms):
            if a['kind'] != 'figure':
                continue
            want = bool(limp[n]) if limp is not None and n < len(limp) else a.get('limp0', a['limp'])
            a.setdefault('limp0', a['limp'])
            a['limp'] = False
            self._go_limp(a, LIMP if want else STAND)
            a['limp'] = want

    def break_off(self, si, pieces):
        """Every weld holding these pieces of breakable si goes, as if each were hit too hard to hold (a bullet's hole:
        ballistics.py). Returns the welds broken."""
        W = self._w
        if W is None or not len(pieces):
            return np.zeros(0, np.int64)
        pieces = np.asarray(pieces, np.int64)
        rows = np.nonzero((W['set'] == si) & (W['over'] >= 0)
                          & (np.isin(W['first'], pieces) | ((W['other'] >= 0) & np.isin(W['other'], pieces))))[0]
        if len(rows):
            self.data.eq_active[W['eq'][rows]] = 0
            W['over'][rows] = -(1 << 40)
            for r in rows:
                self.breaks.append((self.time, self.data.xpos[W['body'][r]].copy(), float(W['area'][r]), int(W['collider'][r])))
            I = self._imp
            if I is not None:      # (a piece no weld holds is no longer checked for being knocked off)
                I['on'][:] = [bool((W['over'][h] >= 0).any()) if len(h) else False for h in I['holds']]
        return rows

    def _unbend(self):
        """Every weld back to its rest pose as built, nothing bent."""
        W = self._w
        if W is None or 'rel0' not in W:
            return
        self.model.eq_data[W['eq'], 3:10] = W['rel0']
        W['plastic'][:] = 0.0
        for ps in self.sets:
            ps.bent = False

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

    def whole(self, si):
        """Whether breakable si is still in one piece: none of the welds between its pieces broken. It is then drawn,
        and met by the fluids, as itself (no seams), not as its pieces; not a brick wall, whose joints show, nor one
        that burns (its pieces each char on their own)."""
        ps = self.sets[si]
        if ps.burnable or ps.joints_show or ps.bent:
            return False
        if self._w is None or si >= len(self._set_bonds):
            return True
        return bool(self.data.eq_active[self._w['eq'][self._set_bonds[si]]].all())

    def _whole_pose(self, ps):
        """A whole breakable's override (as a body's, _poses): where its pieces put it."""
        d = self.data
        b = int(ps.bodies[0])
        R = d.xmat[b].reshape(3, 3)
        pos = d.xpos[b] - R @ ps.frac.pieces[0].centroid
        a = int(ps.vadr[0])
        w = R @ d.qvel[a + 3:a + 6]
        vel = d.qvel[a:a + 3] + np.cross(w, pos - d.xpos[b])
        return dict(pos=tuple(float(x) for x in pos), rot_y=0.0, spin=0.0, vel=tuple(float(x) for x in vel),
                    quat=tuple(float(x) for x in xyzw(d.xquat[b])), omega=(*(float(x) for x in w), 0.0))

    def piece_poses(self, whole=False):
        """Every broken breakable's pieces as they are now: {collider index: dict(pos (n, 3), quat (n, 4) x y z w,
        vel (n, 3), omega (n, 3) world, size: the size it was cut at, impact: where it was hit (if anywhere))}. (One
        still whole is not here, unless whole: it is itself, _poses.)"""
        d = self.data
        out = self.assembly_poses() if self.asms else {}
        for si, ps in enumerate(self.sets):
            if not whole and self.whole(si):
                continue
            q = d.xquat[ps.bodies]
            rot = d.xmat[ps.bodies].reshape(-1, 3, 3)
            w_local = np.stack([d.qvel[a + 3:a + 6] for a in ps.vadr]) if len(ps.vadr) else np.zeros((0, 3))
            out[ps.index] = dict(pos=d.xpos[ps.bodies].copy(), quat=np.concatenate([q[:, 1:], q[:, :1]], 1),
                                 vel=np.stack([d.qvel[a:a + 3] for a in ps.vadr]) if len(ps.vadr) else np.zeros((0, 3)),
                                 omega=np.einsum('bij,bj->bi', rot, w_local), size=np.asarray(ps.size, float),
                                 hollow=np.float32(ps.hollow))
            if ps.impact is not None:
                out[ps.index]['impact'] = np.asarray(ps.impact, float)
            if ps.burnable:
                e = out[ps.index]
                e['burn'] = ps.fire.astype(np.float32).copy()
                if ps.wood is not None:
                    e['burn_spots'] = self._spots_drawn(ps).astype(np.float32).reshape(-1, SPOTS, 4)
                    e['burn_half'] = np.asarray(ps.half, np.float32).copy()
                if ps.gone.any():       # (crumbled to ash: far below, where nothing draws or meets it)
                    e['pos'][ps.gone] = (0.0, -1.0e4, 0.0)
                    e['vel'][ps.gone] = 0.0
                    e['omega'][ps.gone] = 0.0
        return out

    # -- burning pieces ------------------------------------------------------------------------------

    @property
    def burning(self):
        """Whether any breakable here burns (its pieces each burn on their own)."""
        return any(ps.burnable for ps in self.sets)

    @staticmethod
    def _reset_fire(ps):
        n = len(ps.names)
        ps.fire = np.zeros((n, 4))
        ps.fire[:, 0] = 1.0
        ps.fire[:, 2] = 1.0 if ps.burnable else 0.0
        ps.gone = np.zeros(n, bool)
        if ps.wood is not None:
            ps.spots = np.zeros((n * SPOTS, 4))
            ps.spots[:, 0] = 1.0
            ps.spots[:, 2] = 1.0
            ps.wstate = {k: np.zeros(n * SPOTS) for k in ('char', 'lit', 'low')}
        else:
            ps.spots = ps.wstate = None

    @staticmethod
    def _wood_spots(ps, pieces):
        """A wood's spots: a 3 x 3 grid on each face of each piece (wood_fire.template), the surface each stands for,
        which are on its open faces (a face glued to the next piece is inside the wood: its spots never feel the fire) and
        which a flame creeps between: neighbours on a piece, and across each glued joint the open spots that meet there."""
        t = WF.template()
        n = len(ps.half)
        centroids = np.array([p.centroid for p in pieces], float)
        ps.spot_local = t[None] * ps.half[:, None, :]
        area = np.zeros((n, SPOTS))
        for f, (ax, _sgn) in enumerate(WF.FACES):
            o = [a for a in range(3) if a != ax]
            area[:, f * 9:(f + 1) * 9] = (4.0 * ps.half[:, o[0]] * ps.half[:, o[1]] / 9.0)[:, None]
        ps.spot_area = area
        # (straight through each, to its twin on the opposite face: how thick the wood is there)
        axis = np.repeat(np.array([ax for ax, _sgn in WF.FACES]), 9)
        ps.spot_thick = (2.0 * ps.half[:, axis]).reshape(-1)
        ps.spot_opp = (np.arange(n)[:, None] * SPOTS + WF.template_opposite()[None, :]).reshape(-1)
        # (a spot a little off its face, inside the piece glued there: it is on a glued face)
        rest = centroids[:, None, :] + ps.spot_local
        off = rest + (np.abs(t) > 0.99)[None] * np.sign(t)[None] * 0.005
        shut = np.zeros((n, SPOTS), bool)
        for a, b in ps.pairs:
            for x, y in ((a, b), (b, a)):
                pl = pieces[y].planes
                shut[x] |= (off[x] @ pl[:, :3].T - pl[:, 3]).max(1) <= 0.0
        ps.spot_open = ~shut.reshape(-1)
        te = WF.template_edges()
        ts = WF.template_sync()
        E, L, Y = [], [], []
        for k in range(n):
            E.append(te + k * SPOTS)
            L.append(np.linalg.norm((t[te[:, 0]] - t[te[:, 1]]) * ps.half[k], axis=1))
            Y.append(ts + k * SPOTS)
        # (across a glued joint: the open spots that meet there, at rest, within about a spot's spacing of each other)
        space = 0.7 * np.sort(ps.half, axis=1)[:, 1]          # (a piece's finer spacing over its broad faces)
        op = ps.spot_open.reshape(n, SPOTS)
        for a, b in ps.pairs:
            d = np.linalg.norm(rest[a][:, None, :] - rest[b][None, :, :], axis=2)
            d[~op[a]] = np.inf
            d[:, ~op[b]] = np.inf
            if not np.isfinite(d).any():
                continue
            i, j = np.nonzero(d <= 0.004)          # (the same place: either side of the seam, where the joint meets the surface)
            Y.append(np.stack([a * SPOTS + i, b * SPOTS + j], 1))
            near = d <= space[a] + space[b] + 1e-4
            near &= d <= 1.2 * min(space[a], space[b]) + 1e-4
            if not near.any():
                near = d <= d.min() + 1e-6
            i, j = np.nonzero(near)
            E.append(np.stack([a * SPOTS + i, b * SPOTS + j], 1))
            L.append(np.maximum(d[i, j], 0.25 * min(space[a], space[b])))
        E = np.concatenate(E).astype(np.int64) if E else np.zeros((0, 2), np.int64)
        L = np.concatenate(L) if L else np.zeros(0)
        keep = ps.spot_open[E[:, 0]] & ps.spot_open[E[:, 1]] if len(E) else np.zeros(0, bool)
        ps.spot_edges, ps.spot_lens = E[keep], L[keep]
        Y = np.concatenate(Y).astype(np.int64) if Y else np.zeros((0, 2), np.int64)
        ps.spot_sync = Y[ps.spot_open[Y[:, 0]] & ps.spot_open[Y[:, 1]]] if len(Y) else Y
        # (drawn: across each glued joint, the open spots on the same face of the wood within about a spot's spacing
        # of each other, nearer ones taking more of each other)
        B, Wt, D = [], [], []
        for a, b in ps.pairs:
            for f, (ax, _sgn) in enumerate(WF.FACES):
                o = [k for k in range(3) if k != ax]
                ia = np.nonzero(op[a, f * 9:(f + 1) * 9])[0] + f * 9
                ib = np.nonzero(op[b, f * 9:(f + 1) * 9])[0] + f * 9
                if not len(ia) or not len(ib):
                    continue
                dd = rest[b][None, ib, :] - rest[a][ia, None, :]
                flat = np.abs(dd[..., ax]) <= 0.003
                hu = max(ps.half[a, o[0]], ps.half[b, o[0]], 1e-4)
                hv = max(ps.half[a, o[1]], ps.half[b, o[1]], 1e-4)
                dn = np.sqrt((dd[..., o[0]] / hu) ** 2 + (dd[..., o[1]] / hv) ** 2)
                i, j = np.nonzero(flat & (dn <= 1.2))
                if not len(i):
                    continue
                B.append(np.stack([a * SPOTS + ia[i], b * SPOTS + ib[j]], 1))
                Wt.append(np.exp(-dn[i, j] ** 2))
                D.append(np.linalg.norm(dd[i, j], axis=1))
        ps.spot_blend = np.concatenate(B).astype(np.int64) if B else np.zeros((0, 2), np.int64)
        ps.spot_blend_w = np.concatenate(Wt) if Wt else np.zeros(0)
        ps.spot_blend_rest = np.concatenate(D) if D else np.zeros(0)

    def _spots_drawn(self, ps):
        """A wood's spots as the stage draws them: each blended with the spots beside it on the same face of the wood
        across the seams to its neighbours, while they are still in place, so its char runs on across the boards rather
        than stepping at each one (where it burns is still each spot's own)."""
        X = ps.spots
        P = ps.spot_blend
        if P is None or not len(P):
            return X.copy()
        d = self.data
        R = d.xmat[ps.bodies].reshape(-1, 3, 3)
        pos = (d.xipos[ps.bodies][:, None, :] + np.einsum('bij,bkj->bki', R, ps.spot_local)).reshape(-1, 3)
        a, b = P[:, 0], P[:, 1]
        gone = np.repeat(ps.gone, SPOTS)
        w = ps.spot_blend_w * ((np.linalg.norm(pos[a] - pos[b], axis=1) <= ps.spot_blend_rest + 0.01)
                               & ~gone[a] & ~gone[b])
        cols = [0, 1, 3]
        S = X[:, cols]
        acc = S.copy()
        wsum = np.ones(len(X))
        np.add.at(acc, a, w[:, None] * S[b])
        np.add.at(acc, b, w[:, None] * S[a])
        np.add.at(wsum, a, w)
        np.add.at(wsum, b, w)
        out = X.copy()
        out[:, cols] = acc / wsum[:, None]
        return out

    # where on a piece the gas's temperature is taken: a 3 x 3 grid just off each of its six faces (its own frame, in half
    # extents; the 1.0 across a face is moved 2 cm further out)
    _FACE = np.array([[sgn * (ax == 0), sgn * (ax == 1), sgn * (ax == 2)] for ax in range(3) for sgn in (-1.0, 1.0)], float)
    _GRID = np.array([(u, v) for u in (-0.7, 0.0, 0.7) for v in (-0.7, 0.0, 0.7)], float)

    def _probe(self):
        if getattr(self, '_probe_pts', None) is None:
            pts = []
            for f in self._FACE:
                ax = int(np.argmax(np.abs(f)))
                others = [a for a in range(3) if a != ax]
                for u, v in self._GRID:
                    q = f.copy()
                    q[others[0]], q[others[1]] = u, v
                    pts.append(q)
            self._probe_pts = np.array(pts)          # (54, 3)
        return self._probe_pts

    def fire_points(self):
        """Where to take the gas's temperature round the pieces that can still catch: a grid of points just off each of
        their faces, so a flame licking any part of one is felt (and, wood, its spots: each burns as the heat on it says).
        (points (m, 3) fire-local, owners (m, 3): set, piece, which of its 54 points)."""
        pts, own = [], []
        if self.data is None:
            return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
        d = self.data
        probe = self._probe()
        out = np.abs(probe) > 0.99                    # (the face's own axis: 2 cm further out)
        for si, ps in enumerate(self.sets):
            if not ps.burnable:
                continue
            can = ~ps.gone & (ps.fire[:, 2] < 1.5) & (ps.fire[:, 0] > 0.0)
            # (wood that burns is felt too: the heat on it keeps its flames going, or they go out)
            live = np.nonzero(can if ps.wood is not None else can & (ps.fire[:, 1] < 1.0))[0]
            probe = self._probe() if ps.wood is None else WF.template()
            out = np.abs(probe) > 0.99
            if not len(live):
                continue
            c = d.xipos[ps.bodies[live]]
            R = d.xmat[ps.bodies[live]].reshape(-1, 3, 3)
            local = probe[None] * ps.half[live][:, None, :] + out[None] * np.sign(probe)[None] * 0.02     # (m, 54, 3)
            p = c[:, None, :] + np.einsum('bij,bkj->bki', R, local)
            o = np.stack([np.full(len(live) * len(probe), si), np.repeat(live, len(probe)),
                          np.tile(np.arange(len(probe)), len(live))], 1)
            p = p.reshape(-1, 3)
            if ps.wood is not None:     # (wood: its open spots, each its own)
                keep = ps.spot_open[o[:, 1] * SPOTS + o[:, 2]]
                p, o = p[keep], o[keep]
            pts.append(p)
            own.append(o)
        if not pts:
            return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
        return np.concatenate(pts), np.concatenate(own).astype(np.int64)

    def burn(self, dt, temps, owners, sp):
        """Burn the pieces through dt seconds. temps: the gas temperature (field units) at fire_points(), owners its
        owners; sp: the scene's Spreading fire (catch_temp, catch_time, creep, burn_time, smoulder). A piece heats in gas
        hotter than catching temperature, or from a piece glued to it that burns (the fire creeping across it), and
        catches; it burns through its fuel over Burn time (thicker pieces for longer), then smoulders, and when its
        smoulder has died down far enough it crumbles to ash. Returns the pieces that crumbled: [(where, size)]."""
        crumbled = []
        if self.data is None:
            return crumbled
        catch_t = max(float(sp['catch_temp']), 1e-3)
        inv_catch = 1.0 / max(float(sp['catch_time']), 1e-3)
        creep = float(sp['creep'])
        inv_smoulder = 1.0 / max(float(sp['smoulder']), 1e-3)
        hot = {}
        owners = np.asarray(owners, np.int64).reshape(-1, 3) if len(temps) else np.zeros((0, 3), np.int64)
        temps = np.asarray(temps, float)
        if len(temps):
            for (si, k, _j), T in zip(owners, temps):
                key = (int(si), int(k))
                hot[key] = max(hot.get(key, 0.0), float(T))
        speed = max(float(sp.get('burn_speed', 1.0)), 0.0)
        for si, ps in enumerate(self.sets):
            if not ps.burnable:
                continue
            F = ps.fire
            n = len(F)
            T = np.array([hot.get((si, k), 0.0) for k in range(n)])
            if ps.wood is not None:
                # wood: as wood burns (wood_fire.py), spot by spot over its pieces, Burn speed-up times faster than real
                mine = owners[:, 0] == si
                Ts = np.zeros(n * SPOTS)
                np.maximum.at(Ts, owners[mine, 1] * SPOTS + owners[mine, 2], temps[mine])
                d = self.data
                R = d.xmat[ps.bodies].reshape(-1, 3, 3)
                pos = (d.xipos[ps.bodies][:, None, :] + np.einsum('bij,bkj->bki', R, ps.spot_local)).reshape(-1, 3)
                WF.step_spots(ps.wood, ps.spots, ps.wstate, Ts, pos, ps.spot_edges, ps.spot_lens, ps.spot_thick,
                              dt * speed, speed, ps.spot_opp)
                WF.sync_spots(ps.spots, ps.wstate, ps.spot_sync)
                F[:] = WF.pieces_from_spots(ps.spots, n, ps.spot_open)
                for k in np.nonzero((F[:, 2] >= 1.5) & (F[:, 3] <= ps.crumble) & ~ps.gone)[0]:
                    crumbled.append((self.data.xipos[ps.bodies[k]].copy(), float(np.linalg.norm(ps.half[k]))))
                    self._crumble(si, ps, int(k))
                if ps.gone.any():   # (ash: its spots spent and cold)
                    g = np.repeat(ps.gone, SPOTS)
                    ps.spots[g, 2] = 2.0
                    ps.spots[g, 3] = 0.0
                continue
            alight = (F[:, 2] < 1.5) & (F[:, 1] >= 1.0) & (F[:, 0] > 0.0) & ~ps.gone
            nb = np.zeros(n, bool)
            if len(ps.pairs):
                a, b = ps.pairs[:, 0], ps.pairs[:, 1]
                nb[b[alight[a]]] = True
                nb[a[alight[b]]] = True
            warming = ~alight & (F[:, 0] > 0.0) & (F[:, 2] < 1.5) & ~ps.gone
            x = np.clip((T - catch_t) / (0.5 * catch_t + 0.02), 0.0, 1.0)
            heat = x * x * (3.0 - 2.0 * x) * inv_catch + creep / ps.thick * nb
            up = warming & (heat > 0.0)
            F[up, 1] = np.minimum(F[up, 1] + heat[up] * dt, 1.0)
            cool = warming & (heat <= 0.0)
            F[cool, 1] = np.maximum(F[cool, 1] - 0.25 * inv_catch * dt, 0.0)
            # burning: through its fuel, thicker pieces for longer
            life = float(sp['burn_time']) * (1.0 + ps.thick / CHAR_DEPTH)
            F[alight, 0] = np.maximum(F[alight, 0] - dt / life[alight], 0.0)
            out = alight & (F[:, 0] <= 0.0)
            F[out, 2] = 2.0
            F[out, 3] = 1.0
            done = (F[:, 2] >= 1.5) & ~ps.gone
            F[done & ~out, 3] = np.maximum(F[done & ~out, 3] - inv_smoulder * dt, 0.0)
            # to ash once its smoulder has died down far enough
            for k in np.nonzero(done & (F[:, 3] <= ps.crumble))[0]:
                crumbled.append((self.data.xipos[ps.bodies[k]].copy(), float(np.linalg.norm(ps.half[k]))))
                self._crumble(si, ps, int(k))
        self._weaken()
        return crumbled

    def _crumble(self, si, ps, k):
        """Piece k of set ps falls to ash: unglued, meeting nothing, and drawn nowhere (piece_poses)."""
        m, d = self.model, self.data
        ps.gone[k] = True
        bid = int(ps.bodies[k])
        self.breaks.append((self.time, d.xipos[bid].copy(), 0.25 * float(ps.area[k]), ps.index))   # (a puff of ash)
        g0, ng = int(m.body_geomadr[bid]), int(m.body_geomnum[bid])
        m.geom_contype[g0:g0 + ng] = 0
        m.geom_conaffinity[g0:g0 + ng] = 0
        d.qvel[ps.vadr[k]:ps.vadr[k] + 6] = 0.0
        W = self._w
        if W is not None:
            mine = (W['set'] == si) & ((W['first'] == k) | (W['other'] == k))
            d.eq_active[W['eq'][mine]] = 0
            W['over'][mine] = -(1 << 40)

    def _wood_fuel(self, ps, sp, pts, val):
        """A wood's flames: fuel, heat and smoke off each spot that burns, just off its face, as fierce as it burns; a
        little smoke and warmth off its glowing char."""
        X = ps.spots
        n = len(ps.half)
        live = ~np.repeat(ps.gone, SPOTS) & ps.spot_open
        burning = (X[:, 1] >= 1.0) & (X[:, 0] > 0.0) & (X[:, 2] < 1.5)
        on = np.nonzero(live & burning)[0]
        glow = np.nonzero(live & ~burning & (X[:, 3] > 0.0))[0]
        if not len(on) and not len(glow):
            return
        d = self.data
        probe = WF.template()
        out = (np.abs(probe) > 0.99) * np.sign(probe) * 0.02
        local = (ps.spot_local + out[None]).reshape(-1, 3)
        k = np.arange(n * SPOTS) // SPOTS
        for idx, flame in ((on, True), (glow, False)):
            if not len(idx):
                continue
            R = d.xmat[ps.bodies[k[idx]]].reshape(-1, 3, 3)
            p = d.xipos[ps.bodies[k[idx]]] + np.einsum('bij,bj->bi', R, local[idx])
            rate = float(sp['fuel']) * FIRE_AREA_DEPTH * ps.spot_area.reshape(-1)[idx]
            if flame:
                rate = rate * X[idx, 3] * (ps.wood.hrr / 180.0e3)
                v = np.stack([rate, np.full(len(idx), float(sp['heat'])), rate * float(sp['smoke'])], 1)
            else:
                w = X[idx, 3]
                v = np.stack([np.zeros(len(idx)), 0.45 * float(sp['heat']) * w, rate * float(sp['smoulder_smoke']) * w], 1)
            pts.append(p)
            val.append(v)

    def _weaken(self):
        """The glue between burning pieces: as strong as the less charred of the two lets it be, (1 - char)^2."""
        W = self._w
        if W is None or not self.burning:
            return
        char = np.zeros(len(W['eq']))
        for si, ps in enumerate(self.sets):
            if not ps.burnable:
                continue
            rows = W['set'] == si
            c = 1.0 - ps.fire[:, 0]
            a = c[W['first'][rows]]
            o = W['other'][rows]
            b = np.where(o >= 0, c[np.maximum(o, 0)], 0.0)
            char[rows] = np.maximum(a, b)
        W['strength'] = W['strength0'] * (1.0 - np.clip(char, 0.0, 1.0)) ** 2

    def fuel_points(self, sp):
        """What the burning pieces give the gas: (points (m, 3) fire-local, per point (fuel F m^3/s, heat (field
        temperature), smoke /s)): each burning piece's middle and its faces, as a burning surface its size would."""
        if self.data is None or not self.burning:
            return np.zeros((0, 3)), np.zeros((0, 3))
        d = self.data
        pts, val = [], []
        for ps in self.sets:
            if not ps.burnable:
                continue
            if ps.wood is not None:
                self._wood_fuel(ps, sp, pts, val)
                continue
            on = np.nonzero((ps.fire[:, 2] < 1.5) & (ps.fire[:, 1] >= 1.0) & (ps.fire[:, 0] > 0.0) & ~ps.gone)[0]
            sm = np.nonzero((ps.fire[:, 2] >= 1.5) & (ps.fire[:, 3] > 0.0) & ~ps.gone)[0]
            for idx, flame in ((on, True), (sm, False)):
                if not len(idx):
                    continue
                c = d.xipos[ps.bodies[idx]]
                R = d.xmat[ps.bodies[idx]].reshape(-1, 3, 3)
                off = np.einsum('bij,kbj->kbi', R, (self.SIDES[:, None, :] * (0.5 * ps.half[idx])[None]))
                p = np.concatenate([c[None], c[None] + off], 0).reshape(-1, 3)
                rate = float(sp['fuel']) * ps.area[idx] * FIRE_AREA_DEPTH / 7.0
                if flame and ps.wood is not None:
                    rate = rate * ps.fire[idx, 3] * (ps.wood.hrr / 180.0e3)    # (its heat release now, of its peak)
                if flame:
                    v = np.stack([rate, np.full(len(idx), float(sp['heat'])), rate * float(sp['smoke'])], 1)
                else:   # smouldering: a little smoke and warmth, no fuel
                    w = ps.fire[idx, 3]
                    v = np.stack([np.zeros(len(idx)), 0.45 * float(sp['heat']) * w, rate * float(sp['smoulder_smoke']) * w], 1)
                pts.append(p)
                val.append(np.tile(v, (7, 1)))
        if not pts:
            return np.zeros((0, 3)), np.zeros((0, 3))
        return np.concatenate(pts), np.concatenate(val)

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
        self._scene = scene
        self._hit_at = {} if self.rehearsal else self._rehearse(scene, idx, layout)
        self._mocap_vel = {}
        self._btab = None

        # the bodies, and the time step their size allows
        bodies = []
        smallest = 1.0
        for i in idx:
            c = scene.colliders[i]
            cg = by_index.get(i)
            if cg is None or kind_of(c) or (breaks(c) and not (self.rehearsal and falls(c))):
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
            release = scene.start + float(c.get('release', 0.0)) * scene.fps if falls(c) else None
            bodies.append(Body(index=i, shape=shape, size=size, density=r['density'], friction=r['friction'],
                               bounce=r['bounce'], volume=max(vol, 1e-9), area=area, hull=hull, release=release))
        for i in idx:
            c = scene.colliders[i]
            if breaks(c) and i in by_index:
                fr = fractured(c, by_index[i].size, by_index[i].hollow, self._hit_at.get(i), scene)
                for pc in fr.pieces:
                    # (a thin piece, a shard of a pane, can take a longer step than a small solid of that size)
                    ext = pc.verts.max(0) - pc.verts.min(0)
                    smallest = min(smallest, 0.5 * float(np.sort(ext)[1]), 2.0 * float(ext.min()))
        dt = min(max(0.1 * smallest, MIN_DT), MAX_DT)
        # a spring's swing takes a dozen steps at least (its stiffness is integrated explicitly)
        for bd in bodies:
            c = scene.colliders[bd.index]
            if joined(c) == 'spring':
                w_n = math.sqrt(max(float(c.get('spring_k', 500.0)), 1e-6) / max(bd.density * bd.volume, 1e-6))
                dt = max(min(dt, 0.5 / w_n), MIN_DT)
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
            if any(scene.curve(('collider', i, kk)) is not None for kk in ('position', 'yaw', 'pitch', 'roll', 'size')):
                moving.add(i)
        mocap = []
        n_mesh = 0
        for i in enabled:
            if i in idx and not (self.rehearsal and breaks(scene.colliders[i]) and not falls(scene.colliders[i])):
                continue
            c, cg = scene.colliders[i], by_index[i]
            size = np.abs(np.asarray(cg.size, float))
            r = resolved(c)
            q = turn_wxyz(cg)
            R = q_rot(xyzw(q))
            # its shapes in its own frame: (shape, size, centre, mesh asset)
            parts = []
            if i in idx:         # (a rehearsal's standing breakable: its pieces, fixed)
                parts = self._rehearsal_parts(spec, i, c, cg)
                self._reh_frame[i] = (np.asarray(cg.pos, float), R)
            elif cg.hollow > 0.0 and cg.shape == 'box':
                parts = [('box', hs, p, None) for p, hs in self._hollow_slabs(size, float(cg.hollow), np.asarray(cg.opening, float),
                                                                            np.asarray(cg.opening_at, float))]
            elif cg.hollow > 0.0:
                continue   # a hollow sphere or cylinder: only its walls for the fluids matter
            elif cg.shape == 'mesh':
                src = scene.mesh_path(c['mesh'])
                if src.split('#')[0].lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.exr', '.bmp')) and i not in moving:
                    self._add_hfield(spec, w, src, size, np.asarray(cg.pos, float), q, contact, r)
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
                parent.name = f'mocap{i}'
                parent.mocap = True
                parent.pos = list(map(float, cg.pos))
                parent.quat = list(q)
                mocap.append(i)
            for n_part, (shape, hs, centre, mesh) in enumerate(parts):
                g = fixed_geom(w if i not in moving else parent, shape, hs, mesh)
                g.name = f'fixed{i}_{n_part}'
                if i in idx:
                    self._reh_geom[g.name] = i
                if i in moving:
                    g.pos = list(map(float, centre))
                else:
                    g.pos = list(map(float, R @ np.asarray(centre, float) + np.asarray(cg.pos, float)))
                    g.quat = _quat_mul_wxyz(q, list(g.quat))
                contact(g, r['friction'], r['bounce'])

        # the free bodies
        for n, bd in enumerate(bodies):
            cg = by_index[bd.index]
            b = w.add_body()
            b.name = f'body{n}'
            b.pos = list(map(float, cg.pos))
            b.quat = _quat_mul_wxyz(wxyz(cg.quat), _yaw_wxyz(float(cg.rot_y)))
            b.add_freejoint()
            if breaks(scene.colliders[bd.index]):      # (a rehearsal's falling breakable: all its pieces, one body)
                parts = self._rehearsal_parts(spec, bd.index, scene.colliders[bd.index], cg)
                for n_part, (shape, hs, centre, mesh) in enumerate(parts):
                    g = fixed_geom(b, shape, hs, mesh)
                    g.name = f'body{n}_{n_part}'
                    g.pos = list(map(float, centre))
                    g.density = float(bd.density)
                    g.priority = 1
                    contact(g, bd.friction, bd.bounce)
                    self._reh_geom[g.name] = bd.index
                self._reh_frame[bd.index] = (None, None)
                continue
            mesh = None
            if bd.shape == 'mesh':
                mesh = f'body{n}'
                ma = spec.add_mesh()
                ma.name = mesh
                ma.uservert = bd.hull.ravel().tolist()
            g = fixed_geom(b, bd.shape, bd.size, mesh)
            g.name = f'bodygeom{bd.index}'
            g.density = float(bd.density)
            g.priority = 1
            contact(g, bd.friction, bd.bounce, roll=bd.shape in ('sphere', 'cylinder'))
            bd.fb = self._float_body(bd, cg)
        asms = self._build_assemblies(scene, spec, w, idx, by_index, contact)
        sets = [] if self.rehearsal else self._build_pieces(scene, spec, w, idx, by_index, contact, fixed_geom, k)
        joints = self._build_joints(scene, spec, by_index, bodies, sets, mocap, k, dt)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        if self._reh_geom:     # (geom names -> ids)
            self._reh_geom = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, nm): i for nm, i in self._reh_geom.items()}
        for jt in joints:
            jt.bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, jt.body)
            jt.oid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, jt.other) if jt.other else 0
            jt.sids = tuple(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in jt.sites)
            jt.eids = tuple(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, e) for e in jt.eqs)
            jt.tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON, jt.tendon) if jt.tendon else -1
            if jt.links:
                jt.link_ids = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm) for nm in jt.links], np.int64)
        self.joints = joints
        self.snaps = []
        self._tendon0 = (m.tendon_limited.copy(), m.tendon_stiffness.copy(), m.tendon_damping.copy())
        self._contype0, self._conaff0 = m.geom_contype.copy(), m.geom_conaffinity.copy()   # (a piece burnt to ash meets nothing)
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
        from .wood import Cleaver, bend_welds
        bend_welds(self)      # (wood bends before it breaks: its welds as stiff as the wood is)
        self._index_welds()
        self._cleaver = Cleaver.of(self)    # (an edge driven into wood's end grain splits it: wood.py)
        from .ballistics import Ballistics
        self.shots = Ballistics(scene, self) if Ballistics.wanted(scene) else None
        self._index_assemblies(asms)
        # starting motion
        d = self.data
        for bd in bodies:
            c = scene.colliders[bd.index]
            v0 = np.asarray(c.get('start_velocity', (0.0, 0.0, 0.0)), float)
            w0 = np.radians(np.asarray(c.get('start_spin', (0.0, 0.0, 0.0)), float))
            bd.throw, bd.spin = v0.copy(), w0.copy()
            cg = by_index[bd.index]
            v0 = v0 + np.asarray(cg.vel, float)
            if bd.pivot is not None:   # (on a hinge or a ball joint: it turns about that)
                v0 = v0 + np.cross(w0, np.asarray(cg.pos, float) - bd.pivot)
            d.qvel[bd.vadr:bd.vadr + 3] = v0
            # free joint angular velocity is in the body's own frame
            R = q_rot(xyzw(d.qpos[bd.qadr + 3:bd.qadr + 7]))
            d.qvel[bd.vadr + 3:bd.vadr + 6] = R.T @ w0
        mujoco.mj_forward(m, d)
        # things that start inside each other are thrown apart when they are let go
        # (and things that start inside something fixed are pushed out)
        owner = {int(m.geom_bodyid[g]): bd for bd in bodies for g in range(m.ngeom) if int(m.geom_bodyid[g]) == bd.body_id}
        fixed = {}
        for g in range(m.ngeom):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
            if nm.startswith('fixed'):
                fixed[g] = int(nm[5:].split('_')[0])
        seen = set()
        for k in range(d.ncon):
            con = d.contact[k]
            a, b = owner.get(int(m.geom_bodyid[con.geom1])), owner.get(int(m.geom_bodyid[con.geom2]))
            if a is None and b is not None:
                a, b, g_other = b, None, int(con.geom1)
            else:
                g_other = int(con.geom2)
            if a is None or a is b:
                continue
            if b is None:
                j = fixed.get(g_other)
                if j is None or con.dist > -0.02 * float(a.size.min()) - 1e-3 or ('in', a.index, j) in seen:
                    continue
                seen.add(('in', a.index, j))
                self.warnings.append(f'{scene.colliders[a.index]["name"]} starts inside {scene.colliders[j]["name"]}: it is pushed '
                                     f'out when it is let go.')
                continue
            if con.dist > -0.02 * float(min(a.size.min(), b.size.min())) - 1e-3:
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
        log.info('Rigid bodies: %d free, %d keyframed, %d joints, step %.2f ms', len(bodies), len(self.mocap), len(joints),
                 dt * 1e3)

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

    def _add_hfield(self, spec, w, src, size, pos, q, contact, r):
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
        b.quat = _quat_mul_wxyz(list(q), list(Z_TO_Y))
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
        self.snaps = []
        self._pushed = {}
        self.capped = {}
        self._unsnap_all()
        self._unash()
        if self._w is not None:
            self._w['over'][:] = 0
        self._unbend()
        self._impact_sync()
        self._restance()
        for ps in self.sets:
            ps.held = False
        if self.shots is not None:
            self.shots.reset()
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
                if bd.pivot is not None:   # (on a hinge or a ball joint: it turns about that)
                    d.qvel[bd.vadr:bd.vadr + 3] += np.cross(bd.spin, d.qpos[bd.qadr:bd.qadr + 3] - bd.pivot)
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
            d.mocap_quat[mid] = turn_wxyz(cg)
            self._mocap_vel[mid] = cg.vel
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
                rho = bd.rho_l or self.rho
                f_hyd, t_hyd = fb.buoyancy(wl, rho)
                m_tot = mass + 0.5 * rho * bd.volume * sub
                kk = rho * self.gravity * fb.area if 0.0 < sub < 1.0 else 0.0
                c = 2.0 * 0.35 * math.sqrt(kk * m_tot) if kk > 0 else 0.0
                drag = (1.5 * sub + 0.05) * bd.liquid_drag
                grav = np.array([0.0, -mass * self.gravity, 0.0])
                F = f_hyd + f_dyn + grav
                F[1] -= c * v[1]
                a = F / m_tot + drag * (vl - v)
                an = float(np.linalg.norm(a))
                if an > MAX_ACCEL:
                    a *= MAX_ACCEL / an
                    self.capped[bd.index] = self.capped.get(bd.index, 0) + 1   # (said: Engine.notices)
                f += mass * a - grav          # MuJoCo adds the weight itself
                # turning: the measured moment, damped by the water it has to push round
                R = q_rot(fb.quat)
                I_w = R @ np.diag(inertia(fb.shape, fb.size, mass)) @ R.T
                t += (mass / m_tot) * (t_hyd + 0.5 * t_dyn) - (0.3 + 6.0 * sub) * (I_w @ omega)
            if bd.index in self._pushed:
                # sand, snow, mud piled on it or shoving it (matter.py): their push over the last frame
                pf, pt = self._pushed[bd.index]
                f = f + pf
                t = t + pt - np.cross(d.xipos[bid] - d.xpos[bid], pf)
            if self._cloth is not None and not bd.held:
                # cloth it lands on or hits (cloth.py): held off its vertices as they were at the frame's start
                cf, ct = self._cloth_contact(bd, bid, mass, v, omega)
                f = f + cf
                t = t + ct
            d.xfrc_applied[bid, :3] = f
            d.xfrc_applied[bid, 3:] = t
        # broken objects' pieces the matter pushed (matter.py: ('piece', collider index, piece index) keys)
        sets = None
        for key, (pf, pt) in self._pushed.items():
            if not isinstance(key, tuple):
                continue
            if sets is None:
                sets = {ps.index: ps for ps in self.sets}
            ps = sets.get(key[1])
            if ps is None or key[2] >= len(ps.bodies):
                continue
            bid = ps.bodies[key[2]]
            d.xfrc_applied[bid, :3] += pf
            d.xfrc_applied[bid, 3:] += pt - np.cross(d.xipos[bid] - d.xpos[bid], pf)
        self._joint_friction()
        self._motors()

    def meet_cloth(self, x, v, alive, fdt):
        """The fabric (cloth.py) for the frame about to be stepped: its vertices (n, 3, fire-local m), their velocities,
        and which are still there. The things that fall meet it as it is now (_cloth_contact); what it takes from them
        is cloth_took(), after the frame. None: no fabric."""
        if x is None or not self.bodies:
            self._cloth = None
            return
        import mujoco
        m, d = self.model, self.data
        x = np.asarray(x, float)
        near = {}
        vel6 = np.zeros(6)
        for bd in self.bodies:
            bid = bd.body_id
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, vel6, 0)
            reach = self._radius(bd) + float(np.linalg.norm(vel6[3:])) * fdt + 0.05
            idx = np.nonzero(alive & (np.linalg.norm(x - d.xpos[bid], axis=1) < reach))[0]
            if len(idx):
                near[bd.index] = idx
        if not near:
            self._cloth = None
            return
        self._cloth = dict(x=x, v=np.asarray(v, float), near=near)
        self._cloth_took = np.zeros_like(x)

    def cloth_took(self):
        """What each vertex of the fabric took from the things it held off through the frame just stepped (N s, (n, 3)),
        or None."""
        took, self._cloth_took = (self._cloth_took if self._cloth is not None else None), None
        self._cloth = None
        return took

    def _radius(self, bd):
        s = np.asarray(bd.size, float)
        if bd.shape == 'sphere':
            return float(s[0])
        if bd.shape == 'cylinder':
            return float(np.hypot(s[0], s[1]))
        if bd.shape == 'mesh' and bd.hull is not None and len(bd.hull):
            return float(np.linalg.norm(np.asarray(bd.hull, float), axis=1).max())
        return float(np.linalg.norm(s[:3]))

    def _cloth_contact(self, bd, bid, mass, v, omega):
        """The push (force, torque about its centre of mass) on body bd from the fabric's vertices within CLOTH_MARGIN of
        its surface, held where they were at the frame's start: a spring of CLOTH_HZ for its mass, damped, with friction;
        what each vertex gave, it takes (self._cloth_took)."""
        zero = (np.zeros(3), np.zeros(3))
        idx = self._cloth['near'].get(bd.index)
        if idx is None:
            return zero
        d = self.data
        R = d.xmat[bid].reshape(3, 3)
        x = self._cloth['x'][idx]
        dist, nl = shape_distance(bd.shape, bd.size, (x - d.xpos[bid]) @ R, bd.hull)
        pen = CLOTH_MARGIN - dist
        on = pen > 0.0
        if not on.any():
            return zero
        nw = nl[on] @ R.T                                   # (outward, toward the vertex)
        at = x[on]
        arm = at - d.xipos[bid]
        rel = v + np.cross(omega, arm) - self._cloth['v'][idx][on]
        vn = np.einsum('ij,ij->i', rel, nw)
        n = int(on.sum())
        w = 2.0 * math.pi * CLOTH_HZ
        # damped to bounce back as it does (restitution e: a damping ratio of -ln e / sqrt(pi^2 + ln^2 e)): a rubber
        # ball bounces on a sheet, a crate settles
        e = math.log(min(max(float(bd.bounce), 0.01), 0.99))
        zeta = min(CLOTH_DAMPING, -e / math.sqrt(math.pi * math.pi + e * e))
        push = np.minimum(np.maximum(mass * w * w * pen[on] + 2.0 * zeta * mass * w * vn, 0.0) / n, CLOTH_HOLD)
        F = -nw * push[:, None]
        vt = rel - vn[:, None] * nw
        lt = np.linalg.norm(vt, axis=1)
        F -= vt / np.maximum(lt, 0.05)[:, None] * (float(bd.friction) * push)[:, None]
        if self._cloth_took is not None:
            self._cloth_took[idx[on]] -= F * float(self.model.opt.timestep)
        return F.sum(axis=0), np.cross(arm, F).sum(axis=0)

    def matter_push(self, forces, fdt=None):
        """The matter's push on the objects over the frame just stepped ({collider index: (force, torque about its
        position)}, N and N m), applied to the falling ones through the next frame."""
        self._pushed = {int(i): (np.asarray(f, float), np.asarray(t, float)) for i, (f, t) in (forces or {}).items()}

    def _blast(self, where, kg):
        """A blast of `kg` of TNT at `where` (fire-local m): every body and every piece of a broken or breakable object is
        thrown away from it by the impulse its face toward it takes (blast_impulse over a quarter of its surface: the
        area a convex thing shows any one way, on average). Glued pieces thrown apart break their welds."""
        m, d = self.model, self.data
        hits = []
        for bd in self.bodies:
            if bd.held:
                continue
            hits.append((bd.vadr, bd.body_id, 0.25 * bd.area))
        for ps in self.sets:
            if ps.held:
                continue
            for n, p in enumerate(ps.frac.pieces):
                hits.append((int(ps.vadr[n]), int(ps.bodies[n]), 0.25 * float(p.face_area.sum())))
        for vadr, bid, area in hits:
            r = d.xipos[bid] - where
            dist = float(np.linalg.norm(r))
            away = r / dist if dist > 1e-9 else np.array([0.0, 1.0, 0.0])
            d.qvel[vadr:vadr + 3] += away * (blast_impulse(kg, dist) * area / max(float(m.body_mass[bid]), 1e-6))

    def _about(self, b, site, ax):
        """Body b's moment of inertia about the line through `site` (a MuJoCo site id) along unit vector ax (world)."""
        m, d = self.model, self.data
        Ri = d.ximat[b].reshape(3, 3)
        r = d.xipos[b] - d.site_xpos[site]
        rp = r - ax * float(r @ ax)
        return float(ax @ (Ri @ np.diag(m.body_inertia[b]) @ Ri.T) @ ax) + float(m.body_mass[b]) * float(rp @ rp)

    def _drive(self, scene, frame):
        """Motors whose speed is keyframed: their speed at this step."""
        for jt in self.joints:
            if jt.keyed:
                jt.motor = float(scene.get(('collider', jt.index, 'motor_speed'), frame)) * RPM

    def _motors(self):
        """Hinges with a motor turn at its speed, one side against the other: a torque about the hinge's axis that
        closes the gap to the speed over a few steps (the two sides' inertia about the axis taken together), at most the
        motor's strength, pushing the other side back as much (a wheel drives its cart; a fan on a post just turns)."""
        import mujoco
        m, d = self.model, self.data
        h = float(m.opt.timestep)
        v6 = np.zeros(6)
        for jt in self.joints:
            if jt.broken or jt.kind != 'hinge' or jt.motor == 0.0 and not jt.keyed or jt.torque <= 0.0:
                continue
            b = jt.bid
            ax = d.xmat[b].reshape(3, 3) @ jt.axis
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, v6, 0)
            w = float(v6[:3] @ ax)
            inertia = self._about(b, jt.sids[0], ax)
            if jt.other_free:
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, jt.oid, v6, 0)
                w -= float(v6[:3] @ ax)
                other = self._about(jt.oid, jt.sids[1], ax)
                inertia = inertia * other / (inertia + other)
            tau = float(np.clip(inertia * (jt.motor - w) / (MOTOR_STEPS * h), -jt.torque, jt.torque)) * ax
            d.xfrc_applied[b, 3:] += tau
            if jt.other_free:
                d.xfrc_applied[jt.oid, 3:] -= tau

    def _joint_friction(self):
        """Hinges and ball joints resist turning: a torque against the turning of one side against the other, that
        slows it down at the rate their Joint friction gives (per second)."""
        import mujoco
        m, d = self.model, self.data
        v6 = np.zeros(6)
        for jt in self.joints:
            if jt.broken or jt.kind not in ('hinge', 'ball') or jt.friction <= 0.0:
                continue
            b = jt.bid
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, v6, 0)
            w = v6[:3].copy()
            if jt.other_free:
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, jt.oid, v6, 0)
                w -= v6[:3]
            Ri = d.ximat[b].reshape(3, 3)
            Iw = Ri @ np.diag(m.body_inertia[b]) @ Ri.T
            r = d.xipos[b] - d.site_xpos[jt.sids[0]]          # from the pivot (a hinge's pin) to its centre of mass
            if jt.kind == 'hinge':
                ax = d.xmat[b].reshape(3, 3) @ jt.axis
                rp = r - ax * float(r @ ax)
                inertia = float(ax @ Iw @ ax) + float(m.body_mass[b]) * float(rp @ rp)   # about the hinge's line
                tau = -jt.friction * inertia * ax * float(w @ ax)
            else:
                tau = -jt.friction * ((Iw + float(m.body_mass[b]) * (float(r @ r) * np.eye(3) - np.outer(r, r))) @ w)
            d.xfrc_applied[b, 3:] += tau
            if jt.other_free:
                d.xfrc_applied[jt.oid, 3:] -= tau

    def advance(self, scene, frame, fdt, substeps, couple=None):
        """Move every body through frame `frame` (fdt seconds of simulation). Returns, for each of the
        frame's `substeps` substeps, the colliders' overrides at the middle of it.

        couple: (longest step (s), fn) to move something in lockstep with the bodies (the matter: matter_engine.py).
        After each step, fn(h, overrides, f) moves it on h seconds (to fraction f of the frame) with the bodies where they
        now are, and returns its push on them ({collider index: (force, torque about its position)}), which the next
        step takes. (Its push a whole frame late would be far too late for anything as stiff as sand.)"""
        import mujoco
        if self.model is None:
            return [None] * substeps
        m, d = self.model, self.data
        dt = m.opt.timestep
        steps = max(1, int(math.ceil(fdt / min(dt, couple[0] if couple else dt) - 1e-9)))
        blasts = [b for b in scene.blasts() if frame - 1 <= b[0] < frame] if hasattr(scene, 'blasts') else []
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
                self.substep_pieces.append(self.piece_poses() if (self.sets or self.asms) else None)
                mi += 1
            self._keyed(scene, frame - 1 + (t + 0.5 * h) / fdt)
            self._drive(scene, frame - 1 + (t + 0.5 * h) / fdt)
            self._assemblies_step(scene, frame - 1 + (t + 0.5 * h) / fdt)
            for fb, where, kg in blasts:
                if frame - 1 + t / fdt <= fb < frame - 1 + (t + h) / fdt:
                    self._blast(np.asarray(where, float), kg)
            if self.shots is not None:     # (bullets fired and flown through this step: what they hit takes it now)
                self.shots.step(self, frame - 1 + t / fdt, frame - 1 + (t + h) / fdt, self.time, h)
            self._forces()
            v0 = self._piece_v()
            mujoco.mj_step(m, d)
            self._keep_in_box()
            if self.rehearsal:
                self._note_hits(v0)
            self._impact(v0, h)
            self._assemblies_hit(v0)
            self._break()
            self._break_joints()
            if getattr(self, '_cleaver', None) is not None:
                self._cleaver.step(self)
            t += h
            self.time += h
            if couple is not None:
                self._pushed = couple[1](h, self._poses(), t / fdt) or {}
        while mi < len(marks):
            out.append(self._poses())
            self.substep_pieces.append(self.piece_poses() if (self.sets or self.asms) else None)
            mi += 1
        if self.shots is not None:
            self.shots.frame_end(self, fdt)       # (the debris they threw up flies on)
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
        for si, ps in enumerate(self.sets):
            out[ps.index] = self._whole_pose(ps) if self.whole(si) and len(ps.bodies) else self.gone()
        for a in self.asms:
            out[a['index']] = self.gone()
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

    def liquid_measures(self, measures, frame_dt, substeps, h, rho, lava=None):
        """Take in the liquid's push on every body over the last frame (LiquidSolver.read_float), for the next. lava:
        the lava's measures and density as well (a fire-and-liquid box with lava): a body takes the push of the one it
        sits deeper in, so a log floats high on lava, a rock sinks slowly through it and a steel weight goes down."""
        self.rho = float(rho)
        dt_sub = frame_dt / max(1, substeps)
        lava_m, lava_rho = lava if lava is not None else ((), rho)
        for k, (bd, m) in enumerate(zip(self.bodies, measures)):
            if bd.fb is None:
                continue
            bd.rho_l, bd.liquid_drag = self.rho, 1.0
            lm = lava_m[k] if k < len(lava_m) else None
            if lm is not None and lm['wet'] and (not m['wet'] or lm['cells'] >= m['cells']):
                m = lm
                bd.rho_l, bd.liquid_drag = float(lava_rho), LAVA_DRAG
            rho = bd.rho_l
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
                    bend=None if self._w is None else (self.model.eq_data[self._w['eq'], 3:10].copy(), self._w['plastic'].copy(),
                                                       [bool(ps.bent) for ps in self.sets]),
                    pieces={int(k): {kk: np.asarray(vv, np.float32) for kk, vv in v.items()} for k, v in self.piece_poses().items()},
                    joints=[bool(jt.broken) for jt in self.joints],
                    ropes={int(k): {kk: np.asarray(vv, np.float32) for kk, vv in v.items()} for k, v in self.rope_poses().items()},
                    fire=[None if not ps.burnable else (ps.fire.copy(), ps.gone.copy(),
                                                        None if ps.wstate is None else dict({k: v.copy() for k, v in ps.wstate.items()},
                                                                                           spots=ps.spots.copy()))
                          for ps in self.sets],
                    shots=None if self.shots is None else self.shots.state(),
                    shots_view=None if self.shots is None else self.shots.view(self),
                    limp=[bool(a['limp']) for a in self.asms])

    def load_state(self, st):
        """Carry on from a saved state (state()). False if it does not fit the current model."""
        import mujoco
        if st is None or self.data is None:
            return False
        q, v = np.asarray(st['qpos'], float), np.asarray(st['qvel'], float)
        if q.shape != self.data.qpos.shape or v.shape != self.data.qvel.shape:
            return False
        # (nor one whose welds are not the model's: cut differently since, every weld it broke would hold again among
        # pieces already flung apart)
        ea = st.get('eq_active')
        if ea is not None and np.shape(ea) != self.data.eq_active.shape:
            return False
        if self._w is not None and st.get('over') is not None and np.shape(st['over']) != self._w['over'].shape:
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
        self._unsnap_all()
        for jt, broken in zip(self.joints, st.get('joints') or []):
            if broken:
                self._snap(jt)
        self._unash()
        for si, (ps, f) in enumerate(zip(self.sets, st.get('fire') or [])):
            if f is not None and ps.burnable and np.shape(f[0]) == ps.fire.shape:
                ps.fire[:] = f[0]
                if ps.wstate is not None and len(f) > 2 and f[2] is not None:
                    for k, v in f[2].items():
                        if k in ps.wstate and np.shape(v) == ps.wstate[k].shape:
                            ps.wstate[k][:] = v
                        elif k == 'spots' and np.shape(v) == ps.spots.shape:
                            ps.spots[:] = v
                for k in np.nonzero(np.asarray(f[1], bool))[0]:
                    self._crumble(si, ps, int(k))
        if self._w is not None and st.get('over') is not None and np.shape(st['over']) == self._w['over'].shape:
            self._w['over'][:] = st['over']
        self._unbend()
        bend = st.get('bend')
        if self._w is not None and bend is not None and np.shape(bend[0]) == self._w['rel0'].shape:
            self.model.eq_data[self._w['eq'], 3:10] = bend[0]
            self._w['plastic'][:] = bend[1]
            for ps, b in zip(self.sets, bend[2]):
                ps.bent = bool(b)
        self._restance(st.get('limp'))
        self._impact_sync()
        self._weaken()
        if self.shots is not None:
            self.shots.load_state(st.get('shots'))
        self.snaps = []
        self.started = True
        self._last = self._poses()
        return True

    def _unash(self):
        """Every burnt piece whole again (its contacts as built), and every piece unburnt."""
        if self.model is None:
            return
        m = self.model
        for ps in self.sets:
            if ps.gone is not None and ps.gone.any():
                for k in np.nonzero(ps.gone)[0]:
                    bid = int(ps.bodies[k])
                    g0, ng = int(m.body_geomadr[bid]), int(m.body_geomnum[bid])
                    m.geom_contype[g0:g0 + ng] = self._contype0[g0:g0 + ng]
                    m.geom_conaffinity[g0:g0 + ng] = self._conaff0[g0:g0 + ng]
            if ps.fire is not None:
                self._reset_fire(ps)
        if self._w is not None:
            self._w['strength'] = self._w['strength0'].copy()

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
        r0t = scene.turn(pi, scene.start).T                                    # the parent's start turn, undone
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
