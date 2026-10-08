"""Rigid bodies (engine/solids.py): things that fall, bounce, slide, stack, push and get pushed, checked against
textbook physics. MuJoCo runs on its own here (no GPU); test_stage.py draws them."""
import math

import numpy as np
import pytest

pytest.importorskip('mujoco')

from blackbody.engine.solids import Solids, _rotation, attached
from blackbody.scene import components
from blackbody.scene.anim import Curve
from blackbody.scene.model import Scene

G = 9.81
SIZE = 12.0   # m: the simulation box; its sides are open, so only the ground holds things


def scene_of(*cols, fps=50, ground=True):
    s = Scene()
    s.data['domain'].update(ground=ground, open_sides=True, preroll=0.0)
    s.data['render']['fps'] = float(fps)
    for c in cols:
        s.add_collider(**c)
    return s


def run(scene, seconds, gravity=None):
    """Step the bodies `seconds` from the first frame: (the solids, their poses at the end of every frame)."""
    S = Solids()
    n = 96
    S.configure(scene, ((n, n, n), SIZE / n, (-SIZE / 2, 0.0, -SIZE / 2)))
    if gravity is not None:
        S.model.opt.gravity[:] = gravity
    S.reset()
    out = []
    for f in range(scene.start + 1, scene.start + 1 + int(round(seconds * scene.fps))):
        S.advance(scene, f, 1.0 / scene.fps, 1)
        out.append(S.overrides())
    return S, out


def track(poses, i, key='pos'):
    return np.array([p[i][key] for p in poses], float)


def test_free_fall_matches_the_fall_through_air():
    from blackbody.engine.solids import AIR_DENSITY, DRAG_COEFF
    r = 0.05
    s = scene_of(dict(name='Ball', shape='sphere', position=(0.0, 10.0, 0.0), size=(r,) * 3, dynamic=True, material='steel'))
    S, poses = run(s, 1.0)
    y, v = track(poses, 0)[:, 1], track(poses, 0, 'vel')[:, 1]
    t = np.arange(1, len(y) + 1) / s.fps
    dt = S.model.opt.timestep
    # with quadratic drag: y = y0 - vt^2/g ln cosh(g t / vt), vt the terminal speed (92 m/s here: close to 1/2 g t^2)
    m = 7850.0 * 4.0 / 3.0 * math.pi * r ** 3
    vt = math.sqrt(2.0 * m * G / (AIR_DENSITY * DRAG_COEFF * math.pi * r * r))
    fall = vt * vt / G * np.log(np.cosh(G * t / vt))
    # (semi-implicit Euler lags the exact fall by about g dt t / 2)
    assert np.all(np.abs(y - (10.0 - fall)) <= 0.5 * G * dt * t + 1e-3)
    assert abs(v[-1] + vt * math.tanh(G / vt)) < 0.01 * G


@pytest.mark.parametrize('material', ['rubber', 'wood'])
def test_bounce_height_follows_the_restitution(material):
    from blackbody.scene.materials import material as mat
    r, h = 0.05, 1.0
    s = scene_of(dict(name='Ball', shape='sphere', position=(0.0, h + r, 0.0), size=(r,) * 3, dynamic=True, material=material),
                 fps=200)
    _, poses = run(s, 1.6)
    y = track(poses, 0)[:, 1] - r
    vy = track(poses, 0, 'vel')[:, 1]
    first = int(np.argmax(vy > 0.0))                 # just after the first bounce
    apex = float(y[first:first + int(1.2 * s.fps)].max())
    e = mat(material).bounce
    # (MuJoCo's soft contact bounces within about 0.05 of the restitution asked for)
    assert abs(math.sqrt(apex / h) - e) < 0.06, (math.sqrt(apex / h), e)


def test_a_box_slides_down_a_slope_steeper_than_its_friction_angle():
    from blackbody.scene.materials import material as mat
    mu = mat('wood').friction
    critical = math.degrees(math.atan(mu))
    moved = {}
    for tilt in (critical - 6.0, critical + 6.0):
        s = scene_of(dict(name='Box', shape='box', position=(0.0, 0.1, 0.0), size=(0.1, 0.1, 0.1), dynamic=True, material='wood'))
        a = math.radians(tilt)
        _, poses = run(s, 1.5, gravity=[G * math.sin(a), -G * math.cos(a), 0.0])
        moved[tilt] = float(np.linalg.norm(track(poses, 0)[-1] - track(poses, 0)[0]))
    hold, slide = moved[critical - 6.0], moved[critical + 6.0]
    assert hold < 0.01, moved
    assert slide > 0.3, moved


def test_a_tower_of_ten_blocks_stands():
    blocks = [dict(name=f'Block {k}', shape='box', position=(0.0, 0.1 + 0.2 * k, 0.0), size=(0.1, 0.1, 0.1), dynamic=True,
                   material='wood') for k in range(10)]
    _, poses = run(scene_of(*blocks), 5.0)
    top = track(poses, 9)
    assert np.linalg.norm(top[-1, [0, 2]] - [0.0, 0.0]) < 0.02
    assert abs(top[-1, 1] - 1.9) < 0.03


def test_a_collision_keeps_the_momentum():
    a = dict(name='A', shape='sphere', position=(-1.0, 3.0, 0.0), size=(0.1,) * 3, dynamic=True, material='steel',
             start_velocity=(2.0, 0.0, 0.0))
    b = dict(name='B', shape='sphere', position=(0.0, 3.0, 0.0), size=(0.1,) * 3, dynamic=True, material='steel')
    S, poses = run(scene_of(a, b), 1.0, gravity=[0.0, 0.0, 0.0])
    va, vb = track(poses, 0, 'vel')[-1], track(poses, 1, 'vel')[-1]
    assert vb[0] > 0.5                                # it was hit
    assert abs(va[0] + vb[0] - 2.0) < 0.06            # equal masses: the sum of the speeds is the momentum


def test_the_bodies_are_deterministic():
    s = components.new_scene('auto', 'person')
    components.add(s, 'dominoes')
    s.data['render']['fps'] = 25.0
    _, p1 = run(s, 2.0)
    _, p2 = run(s, 2.0)
    assert all(np.array_equal(track(p1, i), track(p2, i)) for i in p1[-1])


def test_dominoes_fall_in_turn():
    s = components.new_scene('auto', 'person')
    components.add(s, 'dominoes')
    s.data['render']['fps'] = 25.0
    _, poses = run(s, 4.0)
    # every one has fallen over (its centre has dropped from 12 cm), and in order along the run
    ys = [track(poses, i)[:, 1] for i in range(10)]
    assert all(y[-1] < 0.08 for y in ys), [round(y[-1], 3) for y in ys]
    when = [int(np.argmax(y < 0.1)) for y in ys]
    assert when == sorted(when), when


def test_a_keyframed_object_pushes_a_falling_one():
    s = scene_of(dict(name='Box', shape='box', position=(0.0, 0.1, 0.0), size=(0.1, 0.1, 0.1), dynamic=True, material='wood'),
                 fps=25)
    i = s.add_collider(name='Pusher', shape='box', position=(-1.0, 0.15, 0.0), size=(0.1, 0.15, 0.3))
    s.colliders[i]['position'] = Curve([[s.start, (-1.0, 0.15, 0.0), 'linear'], [s.start + 50, (1.0, 0.15, 0.0), 'linear']])
    _, poses = run(s, 2.0)
    assert track(poses, 0)[-1, 0] > 0.5


def test_held_until_let_go_then_thrown():
    s = scene_of(dict(name='Ball', shape='sphere', position=(0.0, 2.0, 0.0), size=(0.05,) * 3, dynamic=True, material='steel',
                      release=0.5, start_velocity=(3.0, 0.0, 0.0)), fps=50)
    _, poses = run(s, 1.0)
    p, v = track(poses, 0), track(poses, 0, 'vel')
    assert np.allclose(p[:20], [0.0, 2.0, 0.0], atol=1e-6)       # held for half a second
    assert abs(v[-1, 0] - 3.0) < 0.1 and v[-1, 1] < -2.0          # then thrown, and falling


def test_a_hollow_box_holds_a_ball():
    tub = dict(name='Tub', shape='box', position=(0.0, 0.3, 0.0), size=(0.3, 0.3, 0.3), hollow=0.03,
               opening=(0.27, 0.1, 0.27), opening_at=(0.0, 0.3, 0.0))
    ball = dict(name='Ball', shape='sphere', position=(0.0, 1.2, 0.0), size=(0.08,) * 3, dynamic=True, material='rubber')
    _, poses = run(scene_of(tub, ball), 3.0)
    p = track(poses, 1)[-1]
    assert abs(p[0]) < 0.27 and abs(p[2]) < 0.27 and 0.03 < p[1] < 0.6, p


def test_attached_things_ride_along():
    s = scene_of(dict(name='Crate', shape='box', position=(0.0, 1.0, 0.0), size=(0.2, 0.2, 0.2), dynamic=True, material='wood'))
    i = s.add_emitter(name='Fire', shape='sphere', position=(0.0, 1.3, 0.0), size=(0.1, 0.1, 0.1))
    s.links = [{'child': ['emitter', 'Fire'], 'parent': ['collider', 'Crate'], 'offset': [0.0, 0.3, 0.0]}]
    # turned a quarter turn about z (x, y, z, w), the offset turns with it
    q = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    out = attached(s, 'emitter', {0: dict(pos=(1.0, 0.2, 0.0), vel=(0.0, -1.0, 0.0), quat=q, omega=(0.0, 0.0, 0.0, 0.0))})
    assert np.allclose(out[i]['position'], (0.7, 0.2, 0.0), atol=1e-9)
    assert np.allclose(out[i]['velocity'], (0.0, -1.0, 0.0))
    ems = s.emitters_gpu(s.start, moved=out)
    assert any(np.allclose(e.pos, (0.7, 0.2, 0.0)) for e in ems)


def test_every_block_that_falls_starts_clear_of_everything():
    from blackbody.engine.solids import falls
    for comp in components.COMPONENTS:
        if comp.pick:
            continue
        s = components.new_scene('auto', 'person')
        components.add(s, comp.key)
        if any(falls(c) for c in s.colliders):
            S = Solids()
            S.configure(s, ((64, 64, 64), SIZE / 64, (-SIZE / 2, 0.0, -SIZE / 2)))
            assert not [w for w in S.warnings if 'inside' in w or 'joined to' in w], (comp.key, S.warnings)

# ---- tipped over (Tilt, Roll) ----------------------------------------------------------------------------------------

def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_a_tilted_plank_is_a_ramp_that_things_slide_down_past_their_friction_angle():
    from blackbody.scene.materials import material as mat
    mu = mat('wood').friction
    critical = math.degrees(math.atan(mu))
    moved = {}
    for tilt in (critical - 6.0, critical + 6.0):
        a = math.radians(tilt)
        up, n = np.array([math.cos(a), math.sin(a), 0.0]), np.array([-math.sin(a), math.cos(a), 0.0])   # (Roll leans it to -x)
        plank = dict(name='Plank', shape='box', position=(0.0, 1.0, 0.0), size=(1.5, 0.05, 0.5), roll=tilt, material='wood')
        box = dict(name='Box', shape='box', position=tuple(np.array([0.0, 1.0, 0.0]) + 0.5 * up + 0.152 * n), size=(0.1, 0.1, 0.1),
                   roll=tilt, dynamic=True, material='wood')
        S, poses = run(scene_of(plank, box), 1.0)
        assert not S.warnings
        moved[tilt] = track(poses, 1)[-1] - track(poses, 1)[0]
        assert abs(float(moved[tilt] @ n)) < 0.01                      # (on the plank all the way)
    hold, slide = moved[critical - 6.0], moved[critical + 6.0]
    assert np.linalg.norm(hold) < 0.01, moved
    # down the slope at g (sin a - mu cos a): 1/2 a t^2 in a second, give or take its few millimetres' drop onto the plank
    a = math.radians(critical + 6.0)
    ideal = 0.5 * G * (math.sin(a) - mu * math.cos(a))
    assert float(slide @ np.array([-math.cos(a), -math.sin(a), 0.0])) == pytest.approx(ideal, rel=0.1)


def test_a_tipped_over_object_is_turned_the_same_way_everywhere():
    import mujoco
    s = scene_of(dict(name='Slab', shape='box', position=(0.5, 1.0, -0.3), size=(0.4, 0.1, 0.2), yaw=30.0, pitch=20.0, roll=-35.0,
                      material='stone'),
                 dict(name='Post', shape='box', position=(0.0, 0.5, 2.0), size=(0.2, 0.5, 0.2), yaw=40.0),
                 dict(name='Ball', shape='sphere', position=(-2.0, 0.1, 0.0), size=(0.1, 0.1, 0.1), dynamic=True))
    slab, post, _ = s.colliders_gpu()
    # its Rotation about the vertical, then its Tilt about its own x, then its Roll about its own z
    R = _ry(math.radians(30.0)) @ _rx(math.radians(20.0)) @ _rz(math.radians(-35.0))
    assert np.allclose(s.turn(0), R)
    # the shaders (colliders.wgsl: world = position + quat (yaw (local)))
    assert np.allclose(_rotation(slab.quat) @ _ry(slab.rot_y), R)
    assert post.quat == (0.0, 0.0, 0.0, 1.0) and post.rot_y == pytest.approx(math.radians(40.0))   # (upright: just turned)
    assert not slab.moving
    # MuJoCo: the slab where the shaders put it
    S = Solids()
    S.configure(s, ((96, 96, 96), SIZE / 96, (-SIZE / 2, 0.0, -SIZE / 2)))
    S.reset()
    mujoco.mj_forward(S.model, S.data)
    g = mujoco.mj_name2id(S.model, mujoco.mjtObj.mjOBJ_GEOM, 'fixed0_0')
    assert np.allclose(S.data.geom_xmat[g].reshape(3, 3), R, atol=1e-9)
    assert np.allclose(S.data.geom_xpos[g], (0.5, 1.0, -0.3))
    # the box round it, for placing things
    lo, hi = components._extent('collider', s.colliders[0])
    corners = np.array([[x, y, z] for x in (-0.4, 0.4) for y in (-0.1, 0.1) for z in (-0.2, 0.2)]) @ R.T + (0.5, 1.0, -0.3)
    assert np.allclose(lo, corners.min(0)) and np.allclose(hi, corners.max(0))


def test_a_keyed_tilt_turns_it_at_the_speed_of_its_keys():
    s = scene_of(dict(name='Flap', shape='box', position=(0.0, 1.0, 0.0), size=(0.5, 0.02, 0.3), yaw=30.0, material='wood'))
    s.set_key(('collider', 0, 'pitch'), s.start, 0.0)
    s.set_key(('collider', 0, 'pitch'), s.start + 50, 90.0)
    f = s.start + 20
    cg = s.colliders_gpu(f)[0]
    assert cg.moving and s.colliders_animated()
    # turning about its own sideways axis (turned by its yaw) at the keys' rate
    axis = _ry(math.radians(30.0)) @ (1.0, 0.0, 0.0)
    assert np.allclose(cg.omega[:3], math.radians(s.rate(('collider', 0, 'pitch'), f)) * axis, rtol=1e-3, atol=1e-9)


def test_things_attached_to_a_tipped_over_object_keep_their_place_on_it():
    s = scene_of(dict(name='Crate', shape='box', position=(0.0, 1.0, 0.0), size=(0.2, 0.2, 0.2), pitch=40.0, dynamic=True,
                      material='wood'))
    i = s.add_emitter(name='Fire', shape='sphere', position=(0.0, 1.3, 0.0), size=(0.1, 0.1, 0.1))
    s.links = [{'child': ['emitter', 'Fire'], 'parent': ['collider', 'Crate'], 'offset': [0.0, 0.3, 0.0]}]
    # where it starts, the offset is as it was set (the crate's start turn is undone first)
    cg = s.colliders_gpu()[0]
    out = attached(s, 'emitter', {0: dict(pos=cg.pos, vel=(0.0, 0.0, 0.0), rot_y=cg.rot_y, quat=cg.quat, omega=(0.0, 0.0, 0.0, 0.0))})
    assert np.allclose(out[i]['position'], (0.0, 1.3, 0.0), atol=1e-9)
    # tipped back upright, the offset turns back with it
    out = attached(s, 'emitter', {0: dict(pos=cg.pos, vel=(0.0, 0.0, 0.0), quat=(0.0, 0.0, 0.0, 1.0), omega=(0.0, 0.0, 0.0, 0.0))})
    assert np.allclose(out[i]['position'], np.array([0.0, 1.0, 0.0]) + _rx(math.radians(-40.0)) @ (0.0, 0.3, 0.0), atol=1e-9)


def _emitter_turn(e):
    """An EmitterGPU's own frame -> fire-local, as emitters.wgsl turns it: its yaw, then its quaternion."""
    return _rotation(e.quat) @ _ry(e.yaw)


def test_an_emitter_on_a_tumbling_object_tips_over_with_it_shape_and_jet():
    s = scene_of(dict(name='Rocket', shape='box', position=(0.0, 1.0, 0.0), size=(0.1, 0.4, 0.1), dynamic=True, material='wood'))
    i = s.add_emitter(name='Nozzle', shape='box', position=(0.0, 0.55, 0.0), size=(0.05, 0.1, 0.02), yaw=30.0, fuel=20.0,
                      velocity=(0.0, -6.0, 0.0), vel_blend=1.0, inherit=1.0)
    s.links = [{'child': ['emitter', 'Nozzle'], 'parent': ['collider', 'Rocket'], 'offset': [0.0, -0.45, 0.0]}]
    # fallen onto its side (a quarter turn about z) and sliding along at 2 m/s
    q = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    out = attached(s, 'emitter', {0: dict(pos=(0.0, 0.1, 0.0), vel=(2.0, 0.0, 0.0), quat=q, omega=(0.0, 0.0, 0.0, 0.0))})
    assert np.allclose(out[i]['turn'], q) and 'yaw' not in out[i]
    e = s.emitters_gpu(s.start, moved=out)[i]
    assert np.allclose(e.pos, (0.45, 0.1, 0.0), atol=1e-9)
    # its shape: its own Rotation, then the whole turn of the rocket (not just the part of it about the vertical)
    assert np.allclose(_emitter_turn(e), _rz(math.pi / 2) @ _ry(math.radians(30.0)), atol=1e-9)
    # its jet fires along the rocket, out of its tail, plus the rocket's own motion (once)
    assert np.allclose(e.vel, (6.0 + 2.0, 0.0, 0.0), atol=1e-9)
    # upright and still, it is as it was set
    out = attached(s, 'emitter', {0: dict(pos=(0.0, 1.0, 0.0), vel=(0.0, 0.0, 0.0), quat=(0.0, 0.0, 0.0, 1.0),
                                          omega=(0.0, 0.0, 0.0, 0.0))})
    e = s.emitters_gpu(s.start, moved=out)[i]
    assert np.allclose(e.quat, (0.0, 0.0, 0.0, 1.0)) and e.yaw == pytest.approx(math.radians(30.0))
    assert np.allclose(e.vel, (0.0, -6.0, 0.0))


def test_fire_shaped_like_tipped_over_letters_tips_with_them():
    s = scene_of(dict(name='Sign', shape='box', position=(0.3, 0.5, 0.0), size=(0.4, 0.2, 0.05), yaw=25.0, pitch=-70.0, roll=10.0))
    i = s.add_emitter(name='Sign fire', shape='box', position=(0.3, 0.5, 0.0), size=(0.4, 0.2, 0.05), yaw=25.0)
    s.links = [{'child': ['emitter', 'Sign fire'], 'parent': ['collider', 'Sign'], 'offset': [0.0, 0.0, 0.0], 'shape': True}]
    e, cg = s.emitters_gpu(s.start)[i], s.colliders_gpu()[0]
    # turned exactly as the object it burns on (colliders.wgsl turns that by its yaw, then its quaternion)
    assert np.allclose(_emitter_turn(e), s.turn(0), atol=1e-9)
    assert np.allclose(_emitter_turn(e), _rotation(cg.quat) @ _ry(cg.rot_y), atol=1e-9)
    # an emitter merely attached to it (not shaped like it) keeps its own turn
    s.links[0]['shape'] = False
    e = s.emitters_gpu(s.start)[i]
    assert np.allclose(_emitter_turn(e), _ry(math.radians(25.0)), atol=1e-9)
    # made to fall, the fire turns from the sign's own tilt as the sign tumbles
    from blackbody.scene.model import matrix_quat
    s.links[0]['shape'] = True
    s.colliders[0]['dynamic'] = True
    turned = _rx(0.6) @ s.turn(0)
    out = attached(s, 'emitter', {0: dict(pos=(0.3, 0.5, 0.0), vel=(0.0, 0.0, 0.0), quat=matrix_quat(turned),
                                          omega=(0.0, 0.0, 0.0, 0.0))})
    e = s.emitters_gpu(s.start, moved=out)[i]
    assert np.allclose(_emitter_turn(e), turned, atol=1e-9)


def test_the_viewer_draws_fire_on_tipped_over_letters_tipped_as_it_burns():
    pytest.importorskip('PySide6.QtGui')
    from blackbody.ui.viewport import emitter_lines
    s = scene_of(dict(name='Sign', shape='box', position=(0.3, 0.5, 0.0), size=(0.4, 0.2, 0.05), yaw=25.0, pitch=-70.0, roll=10.0))
    i = s.add_emitter(name='Sign fire', shape='box', position=(0.3, 0.5, 0.0), size=(0.4, 0.2, 0.05), yaw=25.0)
    def drawn_as(R):   # the outline's corners are the box's turned by R, and nothing else
        want = np.array([[0.3, 0.5, 0.0] + R @ (np.array([x, y, z]) * (0.4, 0.2, 0.05))
                         for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        got = np.array([q for line in emitter_lines(s, i, s.start) for q in line])
        far = np.linalg.norm(got[:, None] - want[None], axis=-1)
        return far.min(axis=1).max() < 1e-9 and far.min(axis=0).max() < 1e-9
    assert drawn_as(_ry(math.radians(25.0)))               # (not attached: its own Rotation)
    s.links = [{'child': ['emitter', 'Sign fire'], 'parent': ['collider', 'Sign'], 'offset': [0.0, 0.0, 0.0], 'shape': True}]
    assert drawn_as(s.turn(0)) and drawn_as(_emitter_turn(s.emitters_gpu(s.start)[i]))


def test_a_turn_survives_the_round_trip_through_a_quaternion():
    from blackbody.scene.model import matrix_quat, quat_matrix
    rng = np.random.default_rng(3)
    turns = [np.eye(3), _rx(math.pi), _ry(math.pi), _rz(math.pi), _rx(math.pi) @ _ry(0.5 * math.pi), _rz(math.pi - 1e-9)]
    for q in rng.normal(size=(200, 4)):
        turns.append(_rotation(q / np.linalg.norm(q)))
    for m in turns:
        q = matrix_quat(m)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-12 and q[3] >= 0.0
        assert np.allclose(quat_matrix(q), m, atol=1e-9)


def test_an_emitter_carries_its_turn_to_the_gpu():
    import re
    from pathlib import Path
    from blackbody.engine.gpu import Uniforms
    from blackbody.engine.solver import EMITTER_VEC4, MAX_EMITTERS, EmitterGPU, pack_emitters
    src = (Path(__file__).resolve().parents[1] / 'blackbody/engine/wgsl/emitters.wgsl').read_text(encoding='utf-8')
    body = re.search(r'struct Emitter \{(.*?)\};', src, re.S).group(1)
    assert len(re.findall(r'^\s*\w+: vec4<f32>,', body, re.M)) == EMITTER_VEC4
    q = (0.1, 0.2, 0.3, math.sqrt(1.0 - 0.14))
    d = np.asarray(pack_emitters(Uniforms(), [EmitterGPU(quat=q)]).data, float).reshape(-1, 4)
    assert len(d) == 1 + MAX_EMITTERS * EMITTER_VEC4
    rows = d[1:].reshape(MAX_EMITTERS, EMITTER_VEC4, 4)
    assert np.allclose(rows[0, -1], q) and np.allclose(rows[1:, -1], (0.0, 0.0, 0.0, 1.0))   # (the empty slots: no turn)


# ---- blasts ---------------------------------------------------------------------------------------------------------

def test_a_blast_falls_off_with_distance_and_grows_with_its_charge():
    from blackbody.engine.solids import blast_impulse
    assert blast_impulse(0.0, 1.0) == 0.0
    assert abs(blast_impulse(1.0, 2.0) / blast_impulse(1.0, 4.0) - 2.0) < 1e-9          # 1 / r
    assert abs(blast_impulse(8.0, 3.0) / blast_impulse(1.0, 3.0) - 4.0) < 1e-9          # W^(2/3)
    assert blast_impulse(1.0, 0.0) == blast_impulse(1.0, 1.0 / 3.0)                     # (inside its fireball: as at its edge)


def test_a_blast_throws_what_falls_away_from_it():
    from blackbody.engine.solids import blast_impulse
    near = dict(name='Near', shape='box', position=(1.0, 0.2, 0.0), size=(0.2, 0.2, 0.2), dynamic=True, material='wood', density=300.0)
    far = dict(near, name='Far', position=(-3.0, 0.2, 0.0))
    s = scene_of(near, far)
    s.emitters = []
    s.add_emitter(name='Charge', shape='sphere', position=(0.0, 0.2, 0.0), size=(0.1, 0.1, 0.1), start=0.1, blast=2.0)
    S = Solids()
    S.configure(s, ((96, 96, 96), 12.0 / 96, (-6.0, 0.0, -6.0)))
    S.reset()
    for f in range(s.start + 1, s.start + 7):      # (it goes off at 0.1 s: frame 5 at 50 frames a second)
        S.advance(s, f, 1.0 / s.fps, 1)
    v = {i: np.asarray(S.overrides()[i]['vel']) for i in (0, 1)}
    mass = 0.4 ** 3 * 300.0
    want = blast_impulse(2.0, 1.0) * 0.25 * 6 * 0.4 ** 2 / mass
    assert v[0][0] > 0.6 * want and v[0][0] < 1.05 * want          # thrown away, at about the impulse's speed (less friction)
    assert v[1][0] < 0.0 and abs(v[1][0]) < 0.4 * v[0][0]           # the far one, the other way and slower
