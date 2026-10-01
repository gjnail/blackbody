"""Match the camera to the footage from the ground: four points on a rectangle lying on the ground in the
picture (floor tiles, a road, a table top, a rug, a parking bay) and how high the camera was, or how long one side
of the rectangle is.

The rectangle's sides meet at two vanishing points. Their directions from the camera lie in the ground, so
together they give the ground's tilt and roll; being square to each other, they also give the lens (unless it is
known, or a pair of sides runs parallel in the picture). The camera's height then sets the scale. The result is a
free camera in metres, looking at the world origin on the ground in the middle of the rectangle, where the effect
goes. Pixels: x right, y down, from the top left of the output frame; the lens is centred.

Two more ways in: with no rectangle in view (a field, a beach, the sea), the horizon and the lens give the ground's
tilt and roll, and the camera's height the scale. And on a slope, two upright edges (poles, walls, door frames, tree
trunks) give which way is really up: the world stays level, with gravity straight down, and the lined-up ground is a
slope in it."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

HEIGHTS = (('Eye level, hand-held', 1.6), ('On a tripod', 1.4), ('Low, near the ground', 0.4), ('Sitting down', 1.1),
           ('From a balcony', 6.0), ('From a drone', 30.0))
PERSON = 1.75   # m: the figure that shows the scale


@dataclass
class Match:
    position: tuple      # camera in world metres
    rotation: tuple      # XYZ Euler degrees, as the camera settings take it
    focal_px: float
    focal_solved: bool   # the lens came from the picture (else it was the one given)
    height: float        # camera above the ground (m)
    sides: tuple         # the rectangle's sides on the ground (m): first (corner 0 to 1), second (1 to 2)
    R: np.ndarray        # camera-to-world rotation (columns: right, up, back)
    tilt: float          # degrees the camera looks down
    roll: float          # degrees the horizon leans
    slope: float = 0.0   # degrees the lined-up ground slopes (0: level)
    plane: tuple = (0.0, 1.0, 0.0)   # the lined-up ground's normal in the world (through the origin)


def _h(p):
    return np.array([float(p[0]), float(p[1]), 1.0])


def ray(px, size, f):
    """Camera-space direction through a pixel (camera looks along -z, y up)."""
    W, H = size
    return np.array([(px[0] - W / 2) / f, -(px[1] - H / 2) / f, -1.0])


def _dir(v, size, f):
    """Camera-space direction of a homogeneous image point (a vanishing point, possibly at infinity)."""
    W, H = size
    x, y, w = v
    return np.array([(x - W / 2 * w) / f, -(y - H / 2 * w) / f, -w])


def vanishing(corners):
    """The two vanishing points (homogeneous) of a quad's opposite sides: 0-1 with 3-2, and 1-2 with 0-3."""
    q = [_h(c) for c in corners]
    v1 = np.cross(np.cross(q[0], q[1]), np.cross(q[3], q[2]))
    v2 = np.cross(np.cross(q[1], q[2]), np.cross(q[0], q[3]))
    return v1, v2


def solve_focal(corners, size):
    """The lens (focal length in pixels) that makes the quad a rectangle, or None if the picture cannot tell (a pair
    of sides parallel, or the quad cannot be a rectangle seen through any lens)."""
    W, H = size
    v1, v2 = vanishing(corners)
    w1, w2 = v1[2], v2[2]
    scale = max(np.abs(v1[:2]).max(), np.abs(v2[:2]).max(), 1.0)
    if abs(w1) < 1e-9 * scale or abs(w2) < 1e-9 * scale:
        return None
    a = (v1[0] / w1 - W / 2) * (v2[0] / w2 - W / 2) + (v1[1] / w1 - H / 2) * (v2[1] / w2 - H / 2)
    if a >= 0:
        return None
    f = math.sqrt(-a)
    return f if 0.15 * W < f < 20 * W else None   # outside about 5 mm to 700 mm (full frame): not a believable lens


def euler_xyz(R):
    """XYZ Euler angles (degrees) of a rotation R = Rz(c) Ry(b) Rx(a), with c = 0 when looking straight along an axis."""
    if R[2, 0] < -0.999999:
        b, c = math.pi / 2, 0.0
        a = math.atan2(R[0, 1], R[1, 1])
    elif R[2, 0] > 0.999999:
        b, c = -math.pi / 2, 0.0
        a = math.atan2(-R[0, 1], R[1, 1])
    else:
        b = math.asin(-R[2, 0])
        a = math.atan2(R[2, 1], R[2, 2])
        c = math.atan2(R[1, 0], R[0, 0])
    return (math.degrees(a), math.degrees(b), math.degrees(c))


def _segment_vp(segs):
    """The vanishing point (homogeneous) of two image segments [((x, y), (x, y)), ...]."""
    (a, b), (c, d) = segs[:2]
    return np.cross(np.cross(_h(a), _h(b)), np.cross(_h(c), _h(d)))


def _world(n, up, G, h, f, size, f_solved, sides, d2=None):
    """The Match for a ground normal n and true up (camera coordinates), the origin G on the ground (camera
    coordinates)."""
    fwd = np.array([0.0, 0.0, -1.0])
    zf = -(fwd - (fwd @ up) * up)
    if np.linalg.norm(zf) < 1e-6:   # looking straight down: the grid's second side runs toward the camera
        zf = (d2 if d2 is not None else np.array([0.0, 1.0, 0.0]))
        zf = zf - (zf @ up) * up
    zw = zf / np.linalg.norm(zf)
    xw = np.cross(up, zw)
    M = np.column_stack([xw, up, zw])   # world axes, in camera coordinates
    R = M.T                             # camera axes, in world coordinates
    eye = -(M.T @ G)
    look = R @ fwd
    tilt = math.degrees(math.asin(max(-1.0, min(1.0, -look[1]))))
    roll = math.degrees(math.atan2(up[0], up[1]))   # the picture turns by the roll about the view axis
    plane = M.T @ n
    slope = math.degrees(math.acos(max(-1.0, min(1.0, float(n @ up)))))
    return Match(tuple(float(x) for x in eye), euler_xyz(R), float(f), f_solved, h, sides, R, tilt, roll, slope,
                 tuple(float(x) for x in plane))


def _true_up(verticals, size, f, n):
    """Which way is up (camera coordinates) from two upright edges, pointing the same side as the ground's normal."""
    v3 = _segment_vp(verticals)
    u = _dir(v3, size, f)
    if np.linalg.norm(u) < 1e-12:
        raise ValueError('The upright lines are one line: put them on two different upright edges.')
    u = u / np.linalg.norm(u)
    if u @ n < 0:
        u = -u
    if u @ n < 0.5:
        raise ValueError('The upright lines do not stand up from this ground: drag them along things that are truly '
                         'vertical (poles, door frames, wall corners).')
    return u


def solve_horizon(horizon, base, size, focal_px, height=1.6, verticals=None):
    """The camera from the horizon (two pixel points on it), the lens, and the camera's height; the effect goes on
    the ground at pixel `base`. Raises ValueError."""
    size = (float(size[0]), float(size[1]))
    f = float(focal_px)
    a, b = ray(horizon[0], size, f), ray(horizon[1], size, f)
    n = np.cross(a, b)
    if np.linalg.norm(n) < 1e-12:
        raise ValueError('Drag the two ends of the horizon apart.')
    n /= np.linalg.norm(n)
    dc = ray(base, size, f)
    if n @ dc > 0:
        n = -n
    if n @ dc > -1e-6:
        raise ValueError('The spot where the effect goes is on or above the horizon: put it on the ground, below.')
    up = _true_up(verticals, size, f, n) if verticals else n
    h = float(height)
    G = dc * (-h / (n @ dc))
    return _world(n, up, G, h, f, size, False, (0.0, 0.0))


def solve(corners, size, focal_px=None, height=None, side=None, use_solved_lens=True, verticals=None):
    """The camera that sees the quad `corners` (4 pixel points around a rectangle on the ground) as it is drawn.
    Scale from `height` (camera above the ground, m) or `side` (length of side 0-1, m); 1.6 m eye level if neither.
    The lens is solved from the picture when it can be (and use_solved_lens), else focal_px is used.
    Returns a Match, or raises ValueError when the quad cannot be a rectangle on the ground."""
    size = (float(size[0]), float(size[1]))
    corners = [tuple(map(float, c)) for c in corners]
    def turn(a, b, c):
        return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
    turns = [turn(corners[k], corners[(k + 1) % 4], corners[(k + 2) % 4]) for k in range(4)]
    if not (all(t > 1e-6 for t in turns) or all(t < -1e-6 for t in turns)):
        raise ValueError('The grid is folded or bent inward: drag its corners round the rectangle in order, so it '
                         'stays a four-sided shape.')
    f_solved = solve_focal(corners, size) if use_solved_lens or focal_px is None else None
    f = f_solved or focal_px
    if not f:
        raise ValueError('The lens cannot be told from these lines: give the focal length, or make the sides of the grid '
                         'meet further away (follow lines on the ground that run into the distance).')
    v1, v2 = vanishing(corners)
    d1, d2 = _dir(v1, size, f), _dir(v2, size, f)
    n = np.cross(d1, d2)
    if np.linalg.norm(n) < 1e-9:
        raise ValueError('The grid has collapsed to a line: spread its corners out over the ground.')
    n /= np.linalg.norm(n)
    q = [_h(c) for c in corners]
    centre = np.cross(np.cross(q[0], q[2]), np.cross(q[1], q[3]))   # where the diagonals cross: the true middle
    if abs(centre[2]) < 1e-12:
        raise ValueError('The grid is folded over itself: drag its corners around the rectangle in order.')
    cpx = centre[:2] / centre[2]
    dc = ray(cpx, size, f)
    if n @ dc > 0:
        n = -n   # the ground's normal points up, toward the camera
    rays = [ray(c, size, f) for c in corners]
    if any(n @ r >= -1e-9 for r in rays + [dc]):
        raise ValueError('Part of the grid is above the horizon: keep all four corners on the ground.')
    # the ground at a camera height of 1, then scaled
    P1 = [r * (-1.0 / (n @ r)) for r in rays]
    s1 = float(np.linalg.norm(P1[1] - P1[0]))
    if side:
        h = float(side) / max(s1, 1e-9)
    else:
        h = float(height) if height else 1.6
    P = [p * h for p in P1]
    G = dc * (-h / (n @ dc))
    # world axes in camera space: up is the ground's normal; the camera looks along -z (seen from above)
    sides = (float(np.linalg.norm(P[1] - P[0])), float(np.linalg.norm(P[2] - P[1])))
    up = _true_up(verticals, size, f, n) if verticals else n
    return _world(n, up, G, h, f, size, f_solved is not None, sides, d2)


def project(points, R, eye, f, size):
    """World points to pixels (and whether each is in front of the camera)."""
    W, H = size
    pc = (np.asarray(points, float) - np.asarray(eye, float)) @ R   # camera coordinates
    z = -pc[:, 2]
    ok = z > 1e-6
    zz = np.where(ok, z, 1.0)
    return np.column_stack([W / 2 + f * pc[:, 0] / zz, H / 2 - f * pc[:, 1] / zz]), ok


def ground_lines(R, eye, f, size, step=1.0, reach=30.0):
    """Grid lines on the ground (y = 0) around the origin, as pixel polylines (each cut at the camera's near side)."""
    lines = []
    n = int(min(60, reach / step))
    for k in range(-n, n + 1):
        for a, b in ((( k * step, 0, -reach), (k * step, 0, reach)), ((-reach, 0, k * step), (reach, 0, k * step))):
            pts = np.linspace(a, b, 48)
            px, ok = project(pts, R, eye, f, size)
            seg = []
            for p, o in zip(px, ok):
                if o:
                    seg.append(p)
                elif seg:
                    break
            if len(seg) >= 2:
                lines.append((np.array(seg), k == 0))
    return lines


def horizon(R, f, size):
    """Two pixel points on the horizon (the ground's vanishing line), or None if it is out of the picture's plane."""
    W, H = size
    up_c = R.T @ np.array([0.0, 1.0, 0.0])        # world up in camera coordinates
    # image line: points d = ((x - W/2)/f, -(y - H/2)/f, -1) with up_c . d = 0
    a, b, c = up_c
    if abs(b) < 1e-9 and abs(a) < 1e-9:
        return None
    pts = []
    for x in (-W, 2 * W):
        if abs(b) > 1e-9:
            y = H / 2 + f * ((a * (x - W / 2) / f) - c) / b
            pts.append((x, y))
    return pts if len(pts) == 2 else None


def person(at=(0.0, 0.0), height=PERSON):
    """A stick figure standing on the ground at (x, z): [(polyline)] in world metres."""
    x, z = at
    h = float(height)
    head = [(x + 0.11 * math.cos(t), h - 0.11 + 0.11 * math.sin(t), z) for t in np.linspace(0, 2 * math.pi, 16)]
    return [head,
            [(x, h - 0.22, z), (x, h * 0.52, z)],                                       # body
            [(x - 0.22, h * 0.6, z), (x, h - 0.32, z), (x + 0.22, h * 0.6, z)],        # arms
            [(x - 0.14, 0.0, z), (x, h * 0.52, z), (x + 0.14, 0.0, z)]]                # legs
