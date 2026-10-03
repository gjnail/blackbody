"""Things built of several parts on joints, from one object (a collider with `build` set): a person (a crash-test
figure) and a car. Each is one object in the scene (one of the 16) but many MuJoCo bodies: its parts' shapes are convex
pieces like a broken thing's (fracture.Piece), drawn and met by the fluids the same way (Solids.piece_poses).

A person: 13 parts (pelvis, chest, head, upper arms, forearms with the hands, thighs, shins, feet) on ball joints and
hinges within a body's range (a knee only bends back, an elbow only forward). Standing (Stance), its joints are held
stiff, as someone braced, and it stands on its feet until something hits it hard; then it goes limp and falls as a
body does. Limp, it falls from the start. About 70 kg at 1.8 m (its density is a body's, 1010 kg/m^3).

A car: its body and cabin (one part, the cabin tinted glass), and four wheels on springs (6 cm of sag under its own
weight, damped), the front two turning to steer. Driven (Drive), its wheels' motors hold the speed it is keyed to
(Speed, km/h; 0 brakes it), as far as their grip and a car's torque allow; Steer turns the front wheels. About 1300 kg at
4.4 m.

Everything is in the object's own frame, scaled by its Size (half extents: a person 0.25 x 0.9 x 0.15, a car 2.2 x 0.75 x
0.9 as built)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .fracture import Fracture, make_piece

KINDS = ('figure', 'car')
BASE_SIZE = {'figure': (0.25, 0.9, 0.15), 'car': (2.2, 0.75, 0.9)}

# a person's joints: (stiffness N m/rad, damping N m s/rad) standing braced, and limp (a body's own give)
STAND = {'big': (1500.0, 40.0), 'small': (300.0, 8.0)}
LIMP = {'big': (2.0, 6.0), 'small': (0.5, 1.5)}
LIMP_SPEED = 1.5        # m/s: hit this hard (a contact closing on one of its parts), a standing person goes limp
BODY_DENSITY = 1010.0   # kg/m^3

CAR_MASS = 1300.0       # kg, as built (a car's size scales it by its volume)
CAR_SAG = 0.06          # m its springs sink under its weight
CAR_TORQUE = 1500.0     # N m: the most a driven wheel's motor (or its brake) turns it with
WHEEL_R = 0.33          # m, as built
STEER_MAX = 35.0        # degrees

_DOP = np.array([d for d in ((x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)) if any(d)], float)
_DOP /= np.linalg.norm(_DOP, axis=1, keepdims=True)


@dataclass
class Body:
    """One MuJoCo body of an assembly: where its frame is (object frame, the centre of its first piece, or the joint
    for a body with none), its parent (-1: the root, on a free joint), and the joint to it."""
    name: str
    at: np.ndarray
    parent: int = -1
    joint: str = 'free'          # 'free', 'ball', 'hinge', 'slide'
    anchor: np.ndarray = None    # the joint's position (object frame)
    axis: np.ndarray = None
    range: tuple = None          # hinge: degrees (lo, hi); ball: (0, most); slide: metres
    tone: str = ''               # a person's joint: 'big' or 'small' (STAND, LIMP)
    spring: tuple = None         # (stiffness, damping, rest) of a car's suspension
    mass: float = 0.0            # for a body with no piece of its own (a wheel's hub): its mass


@dataclass
class Assembly:
    kind: str
    bodies: list = field(default_factory=list)
    pieces: list = field(default_factory=list)       # fracture.Piece, object frame
    piece_body: list = field(default_factory=list)   # which body each piece is part of
    looks: list = field(default_factory=list)        # None (the object's own), 'rubber', 'glass'
    density: list = field(default_factory=list)      # kg/m^3 of each piece
    drive: list = field(default_factory=list)        # (body, radius): wheels a motor turns
    steer: list = field(default_factory=list)        # bodies that turn to steer
    wheels: list = field(default_factory=list)       # every wheel body (rolls)

    def fracture(self):
        """Its pieces as a Fracture (no bonds), for drawing them."""
        f = Fracture()
        f.pieces = list(self.pieces)
        f.looks = list(self.looks)
        return f


# ---- shapes ---------------------------------------------------------------------------------------------------

def _piece(planes):
    piece, _ = make_piece(np.asarray(planes, float), np.zeros(len(planes), bool))
    if piece is None:
        raise ValueError('an assembly part came out empty')
    return piece


def limb(a, b, r, sides=8):
    """A rounded rod from a to b, radius r: an n-sided prism, its ends bevelled round (about a capsule)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    u = b - a
    u /= np.linalg.norm(u)
    e1 = np.cross(u, [1.0, 0.0, 0.0] if abs(u[0]) < 0.9 else [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    c = 0.5 * (a + b)
    pl = []
    for k in range(sides):
        th = 2.0 * math.pi * (k + 0.5) / sides
        n = math.cos(th) * e1 + math.sin(th) * e2
        pl.append([*n, float(n @ c) + r])
        for end, s in ((b, 1.0), (a, -1.0)):
            m = s * u + n
            m /= np.linalg.norm(m)
            pl.append([*m, float(m @ end) + r])
    pl.append([*u, float(u @ b) + 0.85 * r])
    pl.append([*(-u), float(-u @ a) + 0.85 * r])
    return _piece(pl)


def rounded_box(c, half, r):
    """A box (centre c, half extents) with its edges and corners rounded by about r: planes along 26 directions."""
    c, half = np.asarray(c, float), np.asarray(half, float)
    inner = np.maximum(half - r, 1e-4)
    h = np.abs(_DOP) @ inner + r
    return _piece(np.concatenate([_DOP, (h + _DOP @ c)[:, None]], 1))


def ellipsoid(c, radii):
    """An ellipsoid (centre c, radii) bounded by planes along 26 directions (and its own axes)."""
    c, radii = np.asarray(c, float), np.asarray(radii, float)
    h = np.sqrt(((_DOP * radii) ** 2).sum(1))
    return _piece(np.concatenate([_DOP, (h + _DOP @ c)[:, None]], 1))


def wheel(c, r, w, sides=14):
    """A tyre: a prism round the z axis (radius r, width w), its shoulders bevelled."""
    c = np.asarray(c, float)
    pl = []
    for k in range(sides):
        th = 2.0 * math.pi * (k + 0.5) / sides
        n = np.array([math.cos(th), math.sin(th), 0.0])
        pl.append([*n, float(n @ c) + r])
        for s in (1.0, -1.0):
            m = n + s * np.array([0.0, 0.0, 1.0])
            m /= np.linalg.norm(m)
            pl.append([*m, float(m @ c) + (r + 0.5 * w - 0.06 * r) / math.sqrt(2.0)])
    pl.append([0.0, 0.0, 1.0, c[2] + 0.5 * w])
    pl.append([0.0, 0.0, -1.0, -c[2] + 0.5 * w])
    return _piece(pl)


def cabin(x0, x1, xt0, xt1, y0, y1, zb, zt):
    """A car's cabin: from y0 to y1, its floor from x0 to x1 and roof from xt0 to xt1 (the windscreens slope), half
    width zb at its floor and zt at its roof."""
    pl = [[0.0, -1.0, 0.0, -y0], [0.0, 1.0, 0.0, y1]]
    for (xa, xb, s) in ((x1, xt1, 1.0), (x0, xt0, -1.0)):      # front and back: through (xa, y0) and (xb, y1)
        t = np.array([xb - xa, y1 - y0])
        n = s * np.array([t[1], -t[0]])
        if n[0] * s < 0:
            n = -n
        n /= np.linalg.norm(n)
        pl.append([n[0], n[1], 0.0, n[0] * xa + n[1] * y0])
    for s in (1.0, -1.0):                                       # its sides, leaning in
        t = np.array([zt - zb, y1 - y0])
        n = np.array([t[1], -t[0]])
        n /= np.linalg.norm(n)
        pl.append([0.0, n[1], s * n[0], n[0] * zb + n[1] * y0])
    return _piece(pl)


# ---- the templates ----------------------------------------------------------------------------------------------

def figure(size):
    """A person, 1.8 m at the base size (feet at the bottom of its box)."""
    s = np.abs(np.asarray(size, float)) / np.asarray(BASE_SIZE['figure'])
    sy = s[1]
    k = np.array([s[0], s[1], s[2]])
    P = lambda x, y, z: np.array([x, y - 0.9, z]) * k                  # (as built: feet at y 0, head at 1.8)
    R = lambda r: r * float(np.cbrt(s.prod()))
    A = Assembly('figure')

    def add(name, piece, parent, joint, anchor=None, axis=None, rng=None, tone=''):
        A.bodies.append(Body(name, piece.centroid.copy(), parent, joint, anchor, None if axis is None else np.asarray(axis, float),
                             rng, tone))
        A.pieces.append(piece)
        A.piece_body.append(len(A.bodies) - 1)
        A.looks.append(None)
        A.density.append(BODY_DENSITY)
        return len(A.bodies) - 1

    pelvis = add('pelvis', rounded_box(P(0, 0.97, 0), np.array([0.16, 0.09, 0.10]) * k, R(0.04)), -1, 'free')
    chest = add('chest', rounded_box(P(0, 1.29, 0), np.array([0.17, 0.20, 0.10]) * k, R(0.06)), pelvis, 'ball', P(0, 1.07, 0),
                rng=(0.0, 35.0), tone='big')
    add('head', ellipsoid(P(0, 1.66, 0.01), np.array([0.085, 0.115, 0.10]) * k), chest, 'ball', P(0, 1.52, 0), rng=(0.0, 50.0),
        tone='small')
    for side in (-1.0, 1.0):
        up = add('upper_arm', limb(P(0.215 * side, 1.44, 0), P(0.25 * side, 1.16, 0), R(0.047)), chest, 'ball',
                 P(0.215 * side, 1.44, 0), rng=(0.0, 130.0), tone='small')
        add('forearm', limb(P(0.25 * side, 1.16, 0), P(0.265 * side, 0.80, 0.03), R(0.040)), up, 'hinge', P(0.25 * side, 1.16, 0),
            axis=(-1.0, 0.0, 0.0), rng=(0.0, 145.0), tone='small')
        th = add('thigh', limb(P(0.10 * side, 0.93, 0), P(0.10 * side, 0.50, 0), R(0.068)), pelvis, 'ball',
                 P(0.10 * side, 0.93, 0), rng=(0.0, 100.0), tone='big')
        sh = add('shin', limb(P(0.10 * side, 0.50, 0), P(0.10 * side, 0.095, 0), R(0.055)), th, 'hinge', P(0.10 * side, 0.50, 0),
                 axis=(1.0, 0.0, 0.0), rng=(0.0, 150.0), tone='big')
        add('foot', rounded_box(P(0.10 * side, 0.045, 0.05), np.array([0.05, 0.045, 0.12]) * k, R(0.02)), sh, 'hinge',
            P(0.10 * side, 0.09, 0), axis=(1.0, 0.0, 0.0), rng=(-35.0, 35.0), tone='big')
    del sy
    return A


def car(size):
    """A car, 4.4 m long at the base size (x forward, its wheels' bottoms at the bottom of its box)."""
    s = np.abs(np.asarray(size, float)) / np.asarray(BASE_SIZE['car'])
    P = lambda x, y, z: np.array([x, y - 0.75, z]) * s
    A = Assembly('car')
    rw = WHEEL_R * float(min(s[0], s[1]))
    w = 0.22 * float(s[2])
    shell = rounded_box(P(0.0, 0.62, 0.0), np.array([2.2, 0.30, 0.88]) * s, 0.10 * float(s.min()))
    glass = cabin(*(np.array([-1.35, 1.05]) * s[0]), *(np.array([-0.95, 0.35]) * s[0]),
                  *(np.array([0.90, 1.45]) * s[1] - 0.75 * s[1]), 0.82 * s[2], 0.66 * s[2])
    A.bodies.append(Body('body', shell.centroid.copy(), -1, 'free'))
    for pc, look in ((shell, None), (glass, 'glass')):
        A.pieces.append(pc)
        A.piece_body.append(0)
        A.looks.append(look)
    vol = shell.volume + glass.volume
    wheel_mass = 20.0 * float(s.prod())
    body_density = max(CAR_MASS * float(s.prod()) - 4.0 * wheel_mass, 1.0) / vol
    A.density += [body_density, body_density]
    k_spring = (CAR_MASS * float(s.prod()) / 4.0) * 9.81 / CAR_SAG
    c_spring = 2.0 * 0.6 * math.sqrt(k_spring * CAR_MASS * float(s.prod()) / 4.0)
    for xs, front in ((1.35, True), (-1.35, False)):
        for zs in (0.77, -0.77):
            c = P(xs, WHEEL_R, zs)
            c[1] = P(0, 0, 0)[1] + rw
            hub = len(A.bodies)
            A.bodies.append(Body('hub', c.copy(), 0, 'slide', c.copy(), np.array([0.0, 1.0, 0.0]), (-0.12, 0.10), '',
                                 (k_spring, c_spring, -CAR_SAG), mass=8.0 * float(s.prod())))
            parent = hub
            if front:
                A.bodies.append(Body('steer', c.copy(), hub, 'hinge', c.copy(), np.array([0.0, 1.0, 0.0]), (-STEER_MAX, STEER_MAX),
                                     mass=4.0 * float(s.prod())))
                parent = len(A.bodies) - 1
                A.steer.append(parent)
            tyre = wheel(c, rw, w)
            A.bodies.append(Body('wheel', tyre.centroid.copy(), parent, 'hinge', c.copy(), np.array([0.0, 0.0, -1.0])))
            A.pieces.append(tyre)
            A.piece_body.append(len(A.bodies) - 1)
            A.looks.append('rubber')
            A.density.append(wheel_mass / tyre.volume)
            A.wheels.append(len(A.bodies) - 1)
            A.drive.append((len(A.bodies) - 1, rw))
    del glass
    return A


_CACHE = {}


def assembly(kind, size):
    """The assembly of a kind at a size (kept)."""
    key = (kind, tuple(round(float(x), 6) for x in np.abs(np.asarray(size, float))))
    got = _CACHE.get(key)
    if got is None:
        if len(_CACHE) > 16:
            _CACHE.clear()
        got = _CACHE[key] = (figure if kind == 'figure' else car)(size)
    return got


def kind_of(c):
    """The kind of assembly collider c is built as, or None."""
    k = c.get('build', 'none')
    return k if k in KINDS else None
