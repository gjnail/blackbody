"""Floating objects: colliders the liquid moves.

Each substep the solver sums the liquid's pressure on every face of a floating collider (buoyancy,
the push of waves and impacts) and averages the liquid's velocity next to it (liq_float.wgsl).
Once per frame those sums are read back and the body is moved as a rigid body that translates
and turns about the vertical: the pressure force and its moment, gravity, added mass, a drag
that carries it along with the liquid, and a spring for its waterline (the pressure is measured a
frame behind, so bobbing uses the measured force plus how much the waterline has moved since).
Within the next frame the collider then moves at its new velocity, and pushes the liquid in turn.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

G = 9.81
MAX_ACCEL = 60.0   # m/s^2: a cap against a frame of bad pressure (a body dropped in from high up)


@dataclass
class FloatBody:
    index: int                  # the collider's index in the scene
    density: float              # kg/m^3
    pos: np.ndarray             # centre (fire-local m)
    vel: np.ndarray             # m/s
    yaw: float                  # radians
    spin: float                 # radians/s about y
    volume: float               # m^3 of solid
    area: float                 # m^2, horizontal cross-section (its waterline when half under)
    radius: float               # m, bounding sphere
    half_height: float          # m, from the centre to its lowest point
    r2: float                   # m^2, moment of inertia about y per unit mass
    measured: bool = field(default=False)

    def state(self):
        return (tuple(float(x) for x in self.pos), tuple(float(x) for x in self.vel), float(self.yaw), float(self.spin))

    def override(self, dt=0.0):
        """ColliderGPU fields for this body dt seconds on (at its current velocity)."""
        return dict(pos=tuple(float(x) for x in self.pos + self.vel * dt), rot_y=self.yaw + self.spin * dt,
                    vel=tuple(float(x) for x in self.vel), spin=self.spin)


def body_for(cg, index, density, meshes=None):
    """Geometry of a floating collider from its ColliderGPU at the start."""
    s = np.abs(np.asarray(cg.size, float))
    shape = cg.shape
    if shape == 'sphere':
        r = s[0]
        vol, area, radius, hh, r2 = 4.0 / 3.0 * math.pi * r ** 3, math.pi * r * r, r, r, 0.4 * r * r
    elif shape == 'cylinder':
        r, hy = s[0], s[1]
        vol, area, radius, hh, r2 = 2.0 * math.pi * r * r * hy, math.pi * r * r, math.hypot(r, hy), hy, 0.5 * r * r
    elif shape == 'mesh' and meshes is not None and meshes.ref(cg.mesh) is not None:
        m0, m1, _ = meshes.ref(cg.mesh)
        lo, hi = np.asarray(m0[:3]) * s, np.asarray(m1[:3]) * s
        ext = 0.5 * (hi - lo)
        vol = 8.0 * float(np.prod(ext)) * 0.5
        area = 4.0 * ext[0] * ext[2] * 0.7
        radius = float(np.linalg.norm(np.maximum(np.abs(lo), np.abs(hi))))
        hh, r2 = float(-lo[1]), float(ext[0] ** 2 + ext[2] ** 2) / 3.0
    else:
        vol, area = 8.0 * float(np.prod(s)), 4.0 * s[0] * s[2]
        radius, hh, r2 = float(np.linalg.norm(s)), s[1], float(s[0] ** 2 + s[2] ** 2) / 3.0
    return FloatBody(index=index, density=float(density), pos=np.asarray(cg.pos, float).copy(),
                     vel=np.asarray(cg.vel, float).copy(), yaw=float(cg.rot_y), spin=float(cg.spin),
                     volume=max(float(vol), 1e-9), area=max(float(area), 1e-9), radius=float(radius),
                     half_height=float(hh), r2=max(float(r2), 1e-9))


class Floats:
    """The floating bodies of a liquid simulation, by collider index."""

    def __init__(self, bodies):
        self.bodies = list(bodies)

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
        return {int(i): dict(pos=s[0], vel=s[1], rot_y=s[2], spin=s[3]) for i, s in state.items()}

    def step(self, measures, frame_dt, substeps, h, rho, origin, dims, ground=True, walls=True):
        """Move every body through one frame from the frame's measurements (LiquidSolver.read_float)."""
        dt_sub = frame_dt / max(1, substeps)
        o = np.asarray(origin, float)
        top = o + np.asarray(dims, float) * h
        for b, m in zip(self.bodies, measures):
            if m['cells'] > 0.5 and not b.measured:
                # the solid it is made of, as the grid sees it (hollow shapes and meshes included)
                b.volume = max(m['cells'] * h ** 3, 1e-9)
                b.measured = True
            mass = b.density * b.volume
            force = rho * h * h / dt_sub * np.asarray(m['force'], float)
            moment = rho * h ** 3 / dt_sub * float(m['moment'])
            sub = min(1.0, max(0.0, force[1] / (rho * G * b.volume))) if m['wet'] else 0.0
            m_tot = mass + 0.5 * rho * b.volume * sub
            # the measurement is from the middle of the frame; the waterline spring accounts for
            # the bobbing since then
            y_meas = b.pos[1] + 0.5 * b.vel[1] * frame_dt
            k = rho * G * b.area if 0.0 < sub < 1.0 else 0.0
            c = 2.0 * 0.35 * math.sqrt(k * m_tot) if k > 0 else 0.0
            drag = 1.5 * sub + 0.05
            vl = np.asarray(m['liquid_vel'], float) if m['wet'] else np.zeros(3)
            inertia = m_tot * b.r2
            n = 4
            dt = frame_dt / n
            for _ in range(n):
                f = force.copy()
                f[1] += -k * (b.pos[1] - y_meas) - mass * G - c * b.vel[1]
                a = f / m_tot + drag * (vl - b.vel)
                an = float(np.linalg.norm(a))
                if an > MAX_ACCEL:
                    a *= MAX_ACCEL / an
                b.vel = b.vel + a * dt
                b.pos = b.pos + b.vel * dt
                b.spin += moment / inertia * dt
                b.spin *= math.exp(-(0.3 + 2.0 * sub) * dt)
                b.yaw += b.spin * dt
                if ground and b.pos[1] - b.half_height < o[1]:
                    b.pos[1] = o[1] + b.half_height
                    b.vel[1] = max(b.vel[1], 0.0)
                    b.vel[0::2] *= math.exp(-4.0 * dt)
                if b.pos[1] + b.half_height > top[1] + 2.0 * b.radius:
                    b.vel[1] = min(b.vel[1], 0.0)
                # keep it over the box (it only feels the liquid inside)
                for ax in (0, 2):
                    lo, hi = o[ax] + b.radius, top[ax] - b.radius
                    if lo < hi and not (lo <= b.pos[ax] <= hi):
                        b.pos[ax] = min(max(b.pos[ax], lo), hi)
                        b.vel[ax] = 0.0 if walls else b.vel[ax] * 0.5
