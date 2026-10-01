"""Floating objects: colliders the liquid moves, as free rigid bodies (they bob, drift, tilt, roll and
knock into each other).

Each substep the solver sums the liquid's pressure on every face of a floating collider (buoyancy,
the push of waves and impacts) with its moment about the body's centre, and averages the liquid's
velocity next to it (liq_float.wgsl). Once per frame those sums are read back and the body moves
as a rigid body: the pressure force and moment, gravity, added mass, and a drag that carries it
along with the liquid. The pressure is measured a frame behind, which would let a body rock and bob
ever harder, so its still-water part is worked out again at every step of the frame from the
body's pose: points through its volume below the waterline the measured buoyancy implies push it up
at their own place. What the pressure adds beyond still water (waves, impacts, flow) is added as
measured. Contacts with the
ground, the box's closed walls, keyframed colliders and the other floating bodies are resolved with
impulses (restitution and friction) between surface points of each body and the other's shape.
Within the next frame the collider moves at its new velocity and spin, and pushes the liquid in turn.

Orientation is a quaternion (x, y, z, w); the collider's yaw is folded into it at the start.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

G = 9.81
MAX_ACCEL = 60.0         # m/s^2: a cap against a frame of bad pressure (a body dropped in from high up)
MAX_ALPHA = 25.0         # rad/s^2, likewise for turning
RESTITUTION = 0.2
FRICTION = 0.5


# -- quaternions -----------------------------------------------------------------------------------

def q_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def q_rot(q):
    """3x3 rotation matrix of unit quaternion q."""
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def q_yaw(yaw):
    return np.array([0.0, math.sin(0.5 * yaw), 0.0, math.cos(0.5 * yaw)])


# -- shapes ----------------------------------------------------------------------------------------

def shape_sdf(shape, size, q):
    """Signed distance (m) of a primitive in its own frame at local points q (n, 3)."""
    s = np.abs(np.asarray(size, float))
    if shape == 'sphere':
        return np.linalg.norm(q, axis=-1) - s[0]
    if shape == 'cylinder':
        d = np.stack([np.hypot(q[:, 0], q[:, 2]) - s[0], np.abs(q[:, 1]) - s[1]], -1)
        return np.minimum(d.max(-1), 0.0) + np.linalg.norm(np.maximum(d, 0.0), axis=-1)
    d = np.abs(q) - s   # boxes, and meshes by their bounding box
    return np.minimum(d.max(-1), 0.0) + np.linalg.norm(np.maximum(d, 0.0), axis=-1)


def shape_points(shape, size):
    """Points on a primitive's surface (its own frame) that can touch other things."""
    s = np.abs(np.asarray(size, float))
    if shape == 'sphere':
        d = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)], float)
        return d / np.linalg.norm(d, axis=-1, keepdims=True) * s[0]
    if shape == 'cylinder':
        a = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
        ring = np.stack([np.cos(a) * s[0], np.zeros_like(a), np.sin(a) * s[0]], -1)
        return np.concatenate([ring + [0.0, s[1], 0.0], ring - [0.0, s[1], 0.0], [[0.0, s[1], 0.0], [0.0, -s[1], 0.0]]])
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * s
    edges = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
                      if abs(x) + abs(y) + abs(z) == 2], float) * s
    faces = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float) * s
    return np.concatenate([corners, edges, faces])


def volume_points(shape, size, n=6):
    """Points filling a primitive (its own frame), each standing for an equal share of its volume."""
    s = np.abs(np.asarray(size, float))
    g = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    q = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3) * s
    if shape in ('sphere', 'cylinder'):
        q = q[shape_sdf(shape, size if shape != 'sphere' else (s[0], s[0], s[0]), q) <= 0.0]
    return q


def inertia(shape, size, mass):
    """Body-frame inertia tensor (diagonal) of a solid primitive."""
    s = np.abs(np.asarray(size, float))
    if shape == 'sphere':
        return np.full(3, 0.4 * mass * s[0] ** 2)
    if shape == 'cylinder':
        r, hy = s[0], s[1]
        side = mass * (3 * r * r + (2 * hy) ** 2) / 12.0
        return np.array([side, 0.5 * mass * r * r, side])
    a, b, c = s
    return mass / 3.0 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])


@dataclass
class FloatBody:
    index: int                  # the collider's index in the scene
    density: float              # kg/m^3
    shape: str
    size: np.ndarray            # the shape's size (half extents, radius; a mesh's by its bounding box)
    pos: np.ndarray             # centre (fire-local m)
    vel: np.ndarray             # m/s
    quat: np.ndarray            # orientation (x, y, z, w)
    omega: np.ndarray           # angular velocity (rad/s, world)
    volume: float               # m^3 of solid
    area: float                 # m^2, horizontal cross-section (its waterline when half under)
    radius: float               # m, bounding sphere
    measured: bool = field(default=False)
    samples: np.ndarray = None  # points through its volume (own frame), for its still-water buoyancy

    def __post_init__(self):
        if self.samples is None:
            self.samples = volume_points(self.shape, self.size)

    def buoyancy(self, water_y, rho):
        """Still-water buoyancy (force, moment about the centre; world) for the water surface at
        height water_y, at the body's current pose."""
        r = self.samples @ q_rot(self.quat).T
        under = (self.pos[1] + r[:, 1]) < water_y
        if not under.any():
            return np.zeros(3), np.zeros(3)
        dv = self.volume / len(self.samples)
        up = np.array([0.0, rho * G * dv, 0.0])
        f = up * under.sum()
        tq = np.cross(r[under], up).sum(0)
        return f, tq

    def waterline(self, lift, rho):
        """The water height at which still water would lift the body by `lift` (N) at its pose."""
        y = self.pos[1] + (self.samples @ q_rot(self.quat).T)[:, 1]
        k = int(round(lift / max(rho * G * self.volume / len(self.samples), 1e-12)))
        if k <= 0:
            return -1.0e9
        if k >= len(y):
            return 1.0e9
        ys = np.sort(y)
        return 0.5 * (ys[k - 1] + ys[k])

    @property
    def mass(self):
        return self.density * self.volume

    def state(self):
        return (tuple(float(x) for x in self.pos), tuple(float(x) for x in self.vel),
                tuple(float(x) for x in self.quat), tuple(float(x) for x in self.omega))

    def override(self, dt=0.0):
        """ColliderGPU fields for this body dt seconds on (at its current velocity and spin)."""
        q = self.quat
        w = float(np.linalg.norm(self.omega))
        if w * dt > 1e-9:
            ax = self.omega / w
            h = 0.5 * w * dt
            q = q_mul(np.array([*(ax * math.sin(h)), math.cos(h)]), q)
            q = q / np.linalg.norm(q)
        return dict(pos=tuple(float(x) for x in self.pos + self.vel * dt), rot_y=0.0, spin=0.0,
                    vel=tuple(float(x) for x in self.vel), quat=tuple(float(x) for x in q),
                    omega=(*(float(x) for x in self.omega), 0.0))

    def local(self, pw):
        return (pw - self.pos) @ q_rot(self.quat)

    def world(self, pl):
        return pl @ q_rot(self.quat).T + self.pos


def body_for(cg, index, density, meshes=None):
    """A floating body from its collider (ColliderGPU) at the start."""
    s = np.abs(np.asarray(cg.size, float))
    shape = cg.shape
    if shape == 'sphere':
        r = s[0]
        size = np.array([r, r, r])
        vol, area, radius = 4.0 / 3.0 * math.pi * r ** 3, math.pi * r * r, r
    elif shape == 'cylinder':
        r, hy = s[0], s[1]
        size = np.array([r, hy, r])
        vol, area, radius = 2.0 * math.pi * r * r * hy, math.pi * r * r, math.hypot(r, hy)
    elif shape == 'mesh' and meshes is not None and meshes.ref(cg.mesh) is not None:
        m0, m1, _ = meshes.ref(cg.mesh)
        lo, hi = np.asarray(m0[:3]) * s, np.asarray(m1[:3]) * s
        size = 0.5 * (hi - lo)
        vol = 8.0 * float(np.prod(size)) * 0.5
        area = 4.0 * size[0] * size[2] * 0.7
        radius = float(np.linalg.norm(np.maximum(np.abs(lo), np.abs(hi))))
        shape = 'box'
    else:
        size = s
        vol, area, radius = 8.0 * float(np.prod(s)), 4.0 * s[0] * s[2], float(np.linalg.norm(s))
    q = q_mul(np.asarray(cg.quat, float), q_yaw(float(cg.rot_y)))
    omega = np.array([0.0, float(cg.spin), 0.0]) + np.asarray(cg.omega[:3], float)
    return FloatBody(index=index, density=float(density), shape=shape, size=np.asarray(size, float),
                     pos=np.asarray(cg.pos, float).copy(), vel=np.asarray(cg.vel, float).copy(),
                     quat=q / np.linalg.norm(q), omega=omega, volume=max(float(vol), 1e-9),
                     area=max(float(area), 1e-9), radius=float(radius))


class Floats:
    """The floating bodies of a liquid simulation, by collider index."""

    def __init__(self, bodies):
        self.bodies = list(bodies)
        self.statics = []      # keyframed colliders they can knock against: (shape, size, pos, R)

    @classmethod
    def from_scene(cls, scene, frame, meshes=None):
        pairs = scene.floating()
        if not pairs:
            return None
        cols = scene.colliders_gpu(frame)
        enabled = [i for i, c in enumerate(scene.colliders) if c['enabled']]
        bodies = []
        for i, dens in pairs:
            if i in enabled and enabled.index(i) < len(cols):
                bodies.append(body_for(cols[enabled.index(i)], i, dens, meshes))
        return cls(bodies) if bodies else None

    def __bool__(self):
        return bool(self.bodies)

    def overrides(self, dt=0.0):
        return {b.index: b.override(dt) for b in self.bodies}

    def regions(self, scene):
        """(index in the solver's collider list, bounding radius) per body."""
        enabled = [i for i, c in enumerate(scene.colliders) if c['enabled']]
        return [(enabled.index(b.index), b.radius) for b in self.bodies if b.index in enabled]

    def state(self):
        return {b.index: b.state() for b in self.bodies}

    @staticmethod
    def overrides_from(state):
        out = {}
        for i, s in state.items():
            if len(s[2]) == 4 if isinstance(s[2], (tuple, list)) else False:
                out[int(i)] = dict(pos=tuple(s[0]), vel=tuple(s[1]), rot_y=0.0, spin=0.0, quat=tuple(s[2]),
                                   omega=(*s[3][:3], 0.0))
            else:   # older caches: yaw and spin
                out[int(i)] = dict(pos=tuple(s[0]), vel=tuple(s[1]), rot_y=float(s[2]), spin=float(s[3]))
        return out

    def set_statics(self, scene, frame):
        """The keyframed (not floating) colliders at `frame`, for contacts."""
        floating = {b.index for b in self.bodies}
        enabled = [i for i, c in enumerate(scene.colliders) if c['enabled']]
        cols = scene.colliders_gpu(frame)
        self.statics = []
        for i, cg in zip(enabled, cols):
            if i in floating or cg.hollow > 0.0 or cg.shape == 'mesh':
                continue
            R = q_rot(q_mul(np.asarray(cg.quat, float), q_yaw(float(cg.rot_y))))
            size = np.abs(np.asarray(cg.size, float))
            if cg.shape == 'sphere':
                size = np.array([size[0]] * 3)
            elif cg.shape == 'cylinder':
                size = np.array([size[0], size[1], size[0]])
            self.statics.append((cg.shape, size, np.asarray(cg.pos, float), R, np.asarray(cg.vel, float)))

    # -- stepping ------------------------------------------------------------------------------

    def step(self, measures, frame_dt, substeps, h, rho, origin, dims, ground=True, walls=True):
        """Move every body through one frame from the frame's measurements (LiquidSolver.read_float)."""
        dt_sub = frame_dt / max(1, substeps)
        o = np.asarray(origin, float)
        top = o + np.asarray(dims, float) * h
        forces = []
        for b, m in zip(self.bodies, measures):
            if m['cells'] > 0.5 and not b.measured:
                # the solid it is made of, as the grid sees it (hollow shapes and meshes included)
                b.volume = max(m['cells'] * h ** 3, 1e-9)
                b.measured = True
            force = rho * h * h / dt_sub * np.asarray(m['force'], float)
            moment = rho * h ** 3 / dt_sub * np.asarray(m['moment'], float)
            sub = min(1.0, max(0.0, force[1] / (rho * G * b.volume))) if m['wet'] else 0.0
            vl = np.asarray(m['liquid_vel'], float) if m['wet'] else np.zeros(3)
            # the waterline the measured lift implies, and what the pressure did beyond still water
            wl = b.waterline(max(force[1], 0.0), rho) if m['wet'] else -1.0e9
            f_hyd, t_hyd = b.buoyancy(wl, rho)
            forces.append((force - f_hyd, moment - t_hyd, sub, vl, wl))
        n = 6
        dt = frame_dt / n
        for _ in range(n):
            for b, (f_dyn, t_dyn, sub, vl, wl) in zip(self.bodies, forces):
                mass = b.mass
                m_tot = mass + 0.5 * rho * b.volume * sub
                f_hyd, t_hyd = b.buoyancy(wl, rho)
                k = rho * G * b.area if 0.0 < sub < 1.0 else 0.0
                c = 2.0 * 0.35 * math.sqrt(k * m_tot) if k > 0 else 0.0
                drag = 1.5 * sub + 0.05
                f = f_hyd + f_dyn
                f[1] += -mass * G - c * b.vel[1]
                moment = t_hyd + 0.5 * t_dyn
                a = f / m_tot + drag * (vl - b.vel)
                an = float(np.linalg.norm(a))
                if an > MAX_ACCEL:
                    a *= MAX_ACCEL / an
                b.vel = b.vel + a * dt
                # turning: the moment through the world inertia (with some of the water's added inertia)
                R = q_rot(b.quat)
                I_body = inertia(b.shape, b.size, m_tot)
                I_inv = R @ np.diag(1.0 / I_body) @ R.T
                alpha = I_inv @ (moment - np.cross(b.omega, R @ (I_body * (R.T @ b.omega))))
                al = float(np.linalg.norm(alpha))
                if al > MAX_ALPHA:
                    alpha *= MAX_ALPHA / al
                # turning through water is heavily damped (it has to push the water around it)
                b.omega = (b.omega + alpha * dt) * math.exp(-(0.3 + 6.0 * sub) * dt)
            self._contacts(o, top, ground, walls)
            for b in self.bodies:
                b.pos = b.pos + b.vel * dt
                w = float(np.linalg.norm(b.omega))
                if w > 1e-9:
                    ax = b.omega / w
                    hh = 0.5 * w * dt
                    b.quat = q_mul(np.array([*(ax * math.sin(hh)), math.cos(hh)]), b.quat)
                    b.quat /= np.linalg.norm(b.quat)
                # keep it over the box (it only feels the liquid inside)
                for ax_i in (0, 2):
                    lo, hi = o[ax_i] + b.radius, top[ax_i] - b.radius
                    if lo < hi and not (lo <= b.pos[ax_i] <= hi):
                        b.pos[ax_i] = min(max(b.pos[ax_i], lo), hi)
                        b.vel[ax_i] = 0.0 if walls else b.vel[ax_i] * 0.5
                if b.pos[1] - b.radius > top[1] + b.radius:
                    b.vel[1] = min(b.vel[1], 0.0)

    # -- contacts --------------------------------------------------------------------------------

    def _impulse(self, b, r, nrm, depth, other=None, r2=None, v_other=np.zeros(3)):
        """Resolve one contact point of body b (lever arm r, world) against a surface with normal nrm
        (pointing out of the other thing, into b), moving at v_other, or against body `other`."""
        R = q_rot(b.quat)
        Iinv = R @ np.diag(1.0 / inertia(b.shape, b.size, b.mass)) @ R.T
        v = b.vel + np.cross(b.omega, r)
        if other is not None:
            R2 = q_rot(other.quat)
            Iinv2 = R2 @ np.diag(1.0 / inertia(other.shape, other.size, other.mass)) @ R2.T
            v_other = other.vel + np.cross(other.omega, r2)
        vr = v - v_other
        vn = float(vr @ nrm)
        if vn >= 0.0:
            return
        k = 1.0 / b.mass + float(nrm @ np.cross(Iinv @ np.cross(r, nrm), r))
        if other is not None:
            k += 1.0 / other.mass + float(nrm @ np.cross(Iinv2 @ np.cross(r2, nrm), r2))
        j = -(1.0 + RESTITUTION) * vn / k
        imp = j * nrm
        # friction, up to the Coulomb limit
        vt = vr - vn * nrm
        st = float(np.linalg.norm(vt))
        if st > 1e-6:
            tdir = vt / st
            kt = 1.0 / b.mass + float(tdir @ np.cross(Iinv @ np.cross(r, tdir), r))
            if other is not None:
                kt += 1.0 / other.mass + float(tdir @ np.cross(Iinv2 @ np.cross(r2, tdir), r2))
            imp -= tdir * min(st / kt, FRICTION * j)
        b.vel = b.vel + imp / b.mass
        b.omega = b.omega + Iinv @ np.cross(r, imp)
        if other is not None:
            other.vel = other.vel - imp / other.mass
            other.omega = other.omega - Iinv2 @ np.cross(r2, imp)

    def _contacts(self, o, top, ground, walls):
        for b in self.bodies:
            pts = b.world(shape_points(b.shape, b.size))
            # the ground and the box's closed walls
            planes = []
            if ground:
                planes.append((np.array([0.0, 1.0, 0.0]), o[1]))
            if walls:
                for ax in (0, 2):
                    e = np.zeros(3)
                    e[ax] = 1.0
                    planes.append((e, o[ax]))
                    planes.append((-e, -top[ax]))
            for nrm, off in planes:
                d = pts @ nrm - off
                worst = int(np.argmin(d))
                if d[worst] < 0.0:
                    b.pos = b.pos - nrm * d[worst]          # push out
                    pts = pts - nrm * d[worst]
                    for i in np.nonzero(d < 0.0)[0]:
                        self._impulse(b, pts[i] - b.pos, nrm, -d[i])
            # keyframed colliders
            for shape, size, pos, R, vel in self.statics:
                ql = (pts - pos) @ R
                d = shape_sdf(shape, size, ql)
                hit = np.nonzero(d < 0.0)[0]
                for i in hit:
                    nrm = self._sdf_normal(shape, size, ql[i]) @ R.T
                    b.pos = b.pos + nrm * (-d[i]) / max(len(hit), 1)
                    self._impulse(b, pts[i] - b.pos, nrm, -d[i], v_other=vel)
        # the bodies against each other
        for i, a in enumerate(self.bodies):
            for b2 in self.bodies[i + 1:]:
                if np.linalg.norm(a.pos - b2.pos) > a.radius + b2.radius:
                    continue
                for p_body, q_body in ((a, b2), (b2, a)):
                    pts = p_body.world(shape_points(p_body.shape, p_body.size))
                    ql = q_body.local(pts)
                    d = shape_sdf(q_body.shape, q_body.size, ql)
                    for k in np.nonzero(d < 0.0)[0]:
                        nrm = self._sdf_normal(q_body.shape, q_body.size, ql[k]) @ q_rot(q_body.quat).T
                        sep = nrm * (-d[k]) * 0.5
                        p_body.pos = p_body.pos + sep
                        q_body.pos = q_body.pos - sep
                        self._impulse(p_body, pts[k] - p_body.pos, nrm, -d[k], other=q_body, r2=pts[k] - q_body.pos)

    @staticmethod
    def _sdf_normal(shape, size, ql):
        e = 1e-3
        g = np.array([shape_sdf(shape, size, (ql + d)[None])[0] - shape_sdf(shape, size, (ql - d)[None])[0]
                      for d in np.eye(3) * e])
        n = float(np.linalg.norm(g))
        return g / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
