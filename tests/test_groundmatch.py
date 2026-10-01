"""Matching the camera from a rectangle on the ground: solved cameras see the rectangle where it was drawn."""
import math

import numpy as np
import pytest

from blackbody.engine import camera as cam
from blackbody.scene import groundmatch as G

SIZE = (1920, 1080)


def _camera(eye, target, focal_mm=28.0, sensor=36.0, roll=0.0):
    m = cam.look_at(np.asarray(eye, float), np.asarray(target, float))
    R = m[:3, :3].T
    if roll:
        c, s = math.cos(math.radians(roll)), math.sin(math.radians(roll))
        R = R @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    f = SIZE[0] * focal_mm / sensor
    return R, np.asarray(eye, float), f


def _quad(R, eye, f, w=2.0, d=3.0, at=(0.0, 0.0), turn=0.0):
    c, s = math.cos(math.radians(turn)), math.sin(math.radians(turn))
    pts = []
    for x, z in ((-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)):
        pts.append((at[0] + x * c + z * s, 0.0, at[1] - x * s + z * c))
    px, ok = G.project(pts, R, eye, f, SIZE)
    assert ok.all()
    return [tuple(p) for p in px]


@pytest.mark.parametrize('eye,target,focal,roll,turn', [
    ((0.0, 1.6, 6.0), (0.0, 0.0, 0.0), 28.0, 0.0, 25.0),
    ((2.0, 1.2, 5.0), (0.3, 0.2, 0.0), 50.0, 3.0, 40.0),
    ((-1.0, 6.0, 8.0), (0.0, 0.0, 0.0), 24.0, -2.0, 15.0),
])
def test_solves_lens_tilt_roll_and_height(eye, target, focal, roll, turn):
    R, e, f = _camera(eye, target, focal, roll=roll)
    corners = _quad(R, e, f, turn=turn)
    m = G.solve(corners, SIZE, height=eye[1])
    assert m.focal_solved
    assert m.focal_px == pytest.approx(f, rel=1e-6)
    assert m.height == pytest.approx(eye[1], rel=1e-6)
    assert sorted(m.sides) == pytest.approx(sorted((2.0, 3.0)), rel=1e-6)
    # the solved camera sees the drawn corners where they were drawn (the world it solves is the ground about the
    # rectangle's middle, turned to face the camera)
    back = G.project([(0, 0, 0)], m.R, m.position, m.focal_px, SIZE)[0][0]
    q = [G._h(c) for c in corners]
    mid = np.cross(np.cross(q[0], q[2]), np.cross(q[1], q[3]))
    assert back == pytest.approx(mid[:2] / mid[2], abs=1e-6)
    assert m.roll == pytest.approx(roll, abs=1e-6)
    # what the engine makes of the Euler angles is the same camera
    R2 = cam.euler_xyz(*m.rotation)
    assert np.abs(R2 - m.R).max() < 1e-9


def test_known_lens_and_a_side_length():
    R, e, f = _camera((0.5, 1.7, 7.0), (0.0, 0.0, 0.0), 35.0)
    # a rectangle seen square-on (sides parallel in the picture): the lens cannot be told, so it is given
    corners = [tuple(p) for p in G.project([(-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1)], *_camera((0.0, 1.7, 7.0), (0, 0, 0), 35.0), SIZE)[0]]
    assert G.solve_focal(corners, SIZE) is None
    m = G.solve(corners, SIZE, focal_px=f, side=2.0)
    assert not m.focal_solved
    assert m.height == pytest.approx(1.7, rel=1e-6)
    with pytest.raises(ValueError):
        G.solve(corners, SIZE)   # no lens, and none in the picture


def test_bad_grids_are_refused():
    with pytest.raises(ValueError):
        G.solve([(100, 100), (200, 100), (300, 100), (400, 100)], SIZE, focal_px=1500)   # a line
    R, e, f = _camera((0.0, 1.6, 6.0), (0.0, 0.0, 0.0))
    q = _quad(R, e, f)
    crossed = [q[0], q[2], q[1], q[3]]   # corners out of order: a bow tie
    with pytest.raises(ValueError):
        G.solve(crossed, SIZE, focal_px=f)


def test_overlay_helpers():
    R, e, f = _camera((0.0, 1.6, 6.0), (0.0, 0.0, 0.0))
    m = G.solve(_quad(R, e, f, turn=20), SIZE, height=1.6)
    lines = G.ground_lines(m.R, m.position, m.focal_px, SIZE)
    assert len(lines) > 20 and any(main for _, main in lines)
    hz = G.horizon(m.R, m.focal_px, SIZE)
    assert hz is not None and hz[0][1] < SIZE[1] / 2   # looking down: the horizon is above the middle
    fig = G.person((0.5, 0.0))
    px, ok = G.project(np.concatenate([np.asarray(l, float) for l in fig]), m.R, m.position, m.focal_px, SIZE)
    assert ok.all()


def test_camera_turn_from_tracked_spots():
    from blackbody.scene import camsolve as CS
    rng = np.random.default_rng(3)
    R0, e, f = _camera((0.0, 1.6, 6.0), (0.0, 0.5, 0.0), 28.0)
    world = rng.uniform([-8, 0, -15], [8, 4, -2], (20, 3))   # spots on the set, in front of the camera
    tracks, truth = {}, {}
    for fr in range(1, 31):
        yaw, pitch = math.radians(0.6 * (fr - 1)), math.radians(-0.2 * (fr - 1))
        Ry = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
        Rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
        R = Ry @ R0 @ Rx
        px, ok = G.project(world, R, e, f, SIZE)
        pts = px / np.array(SIZE)
        pts[~ok] = np.nan
        pts[5] += 0.02 * fr / 30   # a spot on something that moves by itself
        tracks[fr], truth[fr] = pts, R
    rots, err, note = CS.solve(tracks, 1, R0, SIZE, f)
    assert len(rots) == 30 and max(err.values()) < 1.5   # the moving spot is dropped once it strays past a few px
    assert max(CS.angle_between(rots[fr], truth[fr]) for fr in rots) < 0.05   # degrees
    keys = CS.rotation_keys(rots)
    assert all(abs(a - b) < 5 for (_, ea), (_, eb) in zip(keys, keys[1:]) for a, b in zip(ea, eb))


def test_camera_travel_from_ground_spots():
    from blackbody.scene import camsolve as CS
    rng = np.random.default_rng(5)
    ground = np.column_stack([rng.uniform(-6, 6, 300), np.zeros(300), rng.uniform(-14, 4, 300)])
    above = np.column_stack([rng.uniform(-5, 5, 30), rng.uniform(1, 3, 30), rng.uniform(-10, -2, 30)])   # walls, poles
    pts = np.concatenate([ground, above])
    f = SIZE[0] * 28.0 / 36.0
    truth, tracks = {}, {}
    for fr in range(1, 41):
        s = (fr - 1) / 39
        eye = np.array([0.0 + 1.5 * s, 1.6 + 0.2 * s, 6.0 - 3.0 * s])   # dolly in, rising a little, sliding right
        target = np.array([0.4 * s, 0.0, -1.0 * s])
        R = cam.look_at(eye, target)[:3, :3].T
        px, ok = G.project(pts, R, eye, f, SIZE)
        inside = ok & (px[:, 0] > 20) & (px[:, 0] < SIZE[0] - 20) & (px[:, 1] > 20) & (px[:, 1] < SIZE[1] - 20)
        tracks[fr] = {k: (px[k, 0] / SIZE[0], px[k, 1] / SIZE[1]) for k in np.nonzero(inside)[0]}
        truth[fr] = (R, eye)
    R0, e0 = truth[1]
    rots, eyes, err, note, how = CS.solve_shot(tracks, 1, R0, e0, SIZE, f)
    assert how == 'travel', note
    assert len(eyes) == 40
    assert max(float(np.linalg.norm(eyes[fr] - truth[fr][1])) for fr in eyes) < 0.01   # metres
    assert max(CS.angle_between(rots[fr], truth[fr][0]) for fr in rots) < 0.05        # degrees
    # a camera that only turns stays put
    tr2 = {}
    for fr in range(1, 21):
        yaw = math.radians(0.5 * (fr - 1))
        R = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]]) @ R0
        px, ok = G.project(pts, R, e0, f, SIZE)
        tr2[fr] = {k: (px[k, 0] / SIZE[0], px[k, 1] / SIZE[1]) for k in np.nonzero(ok)[0]}
    rots2, eyes2, err2, note2, how2 = CS.solve_shot(tr2, 1, R0, e0, SIZE, f)
    assert how2 == 'turn' and eyes2 is None


def test_horizon_and_slope():
    R, e, f = _camera((0.5, 1.7, 6.0), (0.0, 0.3, 0.0), 35.0, roll=2.0)
    # the horizon: where far-off ground points meet the sky
    far = G.project([(-500.0, 0, -2000.0), (500.0, 0, -2000.0)], R, e, f, SIZE)[0]
    base = G.project([(0.0, 0, 0.0)], R, e, f, SIZE)[0][0]
    m = G.solve_horizon(far, base, SIZE, f, height=1.7)
    assert m.roll == pytest.approx(2.0, abs=0.05) and m.height == pytest.approx(1.7)
    assert G.project([(0, 0, 0)], m.R, m.position, m.focal_px, SIZE)[0][0] == pytest.approx(base, abs=1e-6)
    # a 12 degree slope (rising away from the camera), lined up with a rectangle on it and two upright poles
    t = math.radians(12)
    up_s = np.array([0.0, math.cos(t), math.sin(t)])          # the slope's normal
    a1, a2 = np.array([1.0, 0, 0]), np.cross(up_s, [1.0, 0, 0])   # along the slope
    rect = [c * 1.0 for c in (-a1 - a2, a1 - a2, a1 + a2, -a1 + a2)]
    corners = [tuple(p) for p in G.project(rect, R, e, f, SIZE)[0]]
    poles = [G.project([(x, 0, z), (x, 3, z)], R, e, f, SIZE)[0] for x, z in ((-2.5, -3.0), (2.0, -4.0))]
    m2 = G.solve(corners, SIZE, focal_px=f, side=2.0, use_solved_lens=False, verticals=[tuple(map(tuple, q)) for q in poles])
    assert m2.slope == pytest.approx(12.0, abs=0.05)
    assert m2.R[:, 1] @ np.array([0, 1.0, 0]) > 0   # the world is level: up is up
    assert m2.roll == pytest.approx(2.0, abs=0.05)
