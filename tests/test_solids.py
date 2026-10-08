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


# ---- concave falling meshes, hollow things that stay put --------------------------------------------------------------

def _lathe(profile, n=48):
    """A closed surface of revolution about y: profile [(r, y)] from the axis round to the axis."""
    v, t, ring = [], [], []
    for r, y in profile:
        if r <= 1e-12:
            ring.append([len(v)] * n)
            v.append((0.0, y, 0.0))
        else:
            ring.append(list(range(len(v), len(v) + n)))
            v += [(r * math.cos(2 * math.pi * j / n), y, r * math.sin(2 * math.pi * j / n)) for j in range(n)]
    for a, b in zip(ring[:-1], ring[1:]):
        for j in range(n):
            j1 = (j + 1) % n
            t += [tri for tri in ((a[j], a[j1], b[j1]), (a[j], b[j1], b[j])) if len(set(tri)) == 3]
    return np.array(v), np.array(t)


R_BOWL, WALL, FOOT = 0.15, 0.012, 0.03


def _write_obj(path, v, t):
    with open(path, 'w') as f:
        f.writelines(f'v {x:.6f} {y:.6f} {z:.6f}\n' for x, y, z in v)
        f.writelines(f'f {a + 1} {b + 1} {c + 1}\n' for a, b, c in t)
    return path


def _bowl(path, rows=24, round_=48):
    """A 30 cm pottery bowl, its walls 12 mm thick, its rim at y 0, standing on a flat foot (y -0.12) with a flat
    inside floor 12 mm above it (rows and round_: how finely its triangles are cut); written to `path` as OBJ unless
    None. Returns (vertices, triangles)."""
    yb = -R_BOWL + FOOT
    ri, yf = R_BOWL - WALL, yb + WALL
    prof = [(0.0, yb)] + [(R_BOWL * math.cos(p), -R_BOWL * math.sin(p)) for p in np.linspace(math.asin(-yb / R_BOWL), 0.0, rows)]
    prof += [(ri * math.cos(p), -ri * math.sin(p)) for p in np.linspace(0.0, math.asin(-yf / ri), rows)] + [(0.0, yf)]
    v, t = _lathe(prof, round_)
    if path:
        _write_obj(path, v, t)
    return v, t


def _into_bowl(parts, n=3000):
    """How far the convex parts reach into the bowl's hollow at most (m), from points through each part's hull."""
    rng = np.random.default_rng(0)
    ri, yf = R_BOWL - WALL, -R_BOWL + FOOT + WALL
    worst = 0.0
    for p in parts:
        q = rng.dirichlet(np.full(len(p), 0.2), n) @ p
        r = np.linalg.norm(q, axis=1)
        worst = max(worst, float(np.minimum(ri - r, q[:, 1] - yf).max()))
        assert np.all(r <= R_BOWL + 1e-6) and np.all(q[:, 1] >= -R_BOWL + FOOT - 1e-6)   # never outside the bowl
    return worst


def test_a_concave_mesh_is_cut_into_convex_parts_that_keep_its_hollow(tmp_path):
    from blackbody.engine.solids import DECOMP_CELLS, convex_parts
    v, t = _bowl(str(tmp_path / 'bowl.obj'))
    parts, filled, volume = convex_parts(v, t)
    assert not filled and 20 < len(parts) <= 128
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    assert volume == pytest.approx(abs(np.einsum('ij,ij->', a, np.cross(b, c))) / 6.0, rel=1e-6)
    assert _into_bowl(parts) < 0.5 * 2.0 * R_BOWL / DECOMP_CELLS     # its hollow kept, to within half a cell (5 mm)


def test_cutting_a_detailed_mesh_keeps_its_hollow_and_takes_little_memory():
    # (every point over its surface went through 154 directions in doubles: a 50,000-triangle bowl took 670 MB, and was
    # cut into a third as many parts as a plain one, its hollow partly filled)
    import tracemalloc
    from blackbody.engine.solids import DECOMP_CELLS, convex_parts
    v, t = _bowl(None, rows=64, round_=192)
    assert len(t) > 45000
    tracemalloc.start()
    try:
        parts, filled, _volume = convex_parts(v, t)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 250e6                                             # (about 80 MB)
    assert not filled and 80 < len(parts) <= 128                    # as many as the plain bowl's (about 110)
    assert _into_bowl(parts, 1000) < 0.5 * 2.0 * R_BOWL / DECOMP_CELLS


def test_a_ball_rests_inside_a_falling_bowl(tmp_path):
    path = str(tmp_path / 'bowl.obj')
    v, t = _bowl(path)
    yf = -R_BOWL + FOOT + WALL
    s = scene_of(dict(name='Bowl', shape='mesh', mesh=path, position=(0.0, 0.4, 0.0), size=(1.0, 1.0, 1.0), material='ceramic',
                      dynamic=True),
                 dict(name='Ball', shape='sphere', position=(0.0, 0.4 + yf + 0.045, 0.0), size=(0.04,) * 3, material='rubber',
                      dynamic=True))
    S, poses = run(s, 3.0)
    assert not S.warnings                                       # (as its hull, the ball started inside it and flew out)
    bowl, ball = track(poses, 0)[-1], track(poses, 1)[-1]
    assert abs(bowl[1] - (R_BOWL - FOOT)) < 0.003                # the bowl landed on its foot
    rel = ball - bowl
    assert abs(rel[1] - (yf + 0.04)) < 0.003                     # and the ball lies on its floor, inside it
    assert math.hypot(rel[0], rel[2]) < 0.09
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    volume = abs(np.einsum('ij,ij->', a, np.cross(b, c))) / 6.0
    assert S.model.body_mass[S.bodies[0].body_id] == pytest.approx(2300.0 * volume, rel=1e-3)   # it weighs what the bowl does
    # cloth meets it by its parts too (solids.meet_cloth): its hollow is open, its wall and foot solid
    from blackbody.engine.solids import parts_distance, shape_distance
    q = np.array([(0.0, yf + 0.02, 0.0), (0.0, yf - 0.5 * WALL, 0.0), (0.0, -R_BOWL + FOOT - 0.01, 0.0), (R_BOWL - 0.5 * WALL, -0.03, 0.0)])
    dist, nrm = parts_distance(S.bodies[0].planes, q)
    assert abs(dist[0] - 0.02) < 0.004 and nrm[0][1] > 0.9      # 2 cm above its floor, in its hollow (facing up)
    assert dist[1] < 0.0 and dist[3] < 0.0                      # in its floor and in its wall
    assert abs(dist[2] - 0.01) < 0.004                           # 1 cm under its foot
    assert shape_distance('mesh', (1.0, 1.0, 1.0), q[:1], v)[0][0] < 0.0     # (its hull's box had the hollow solid)


def test_things_rest_inside_a_barrel_that_stays_put():
    barrel = dict(name='Barrel', shape='cylinder', position=(0.0, 0.45, 0.0), size=(0.3, 0.45, 0.3), hollow=0.02,
                  opening=(0.35, 0.05, 0.35), opening_at=(0.0, 0.45, 0.0), material='wood')    # (its top open)
    balls = [dict(name=f'Ball {k}', shape='sphere', position=(x, 1.2 + 0.3 * k, z), size=(0.06,) * 3, material='wood',
                  dynamic=True) for k, (x, z) in enumerate(((0.0, 0.0), (0.1, 0.05), (-0.08, -0.1)))]
    S, poses = run(scene_of(barrel, *balls), 3.0)
    assert not S.warnings
    for k in (1, 2, 3):
        p = track(poses, k)[-1]
        assert math.hypot(p[0], p[2]) < 0.3 - 0.02 - 0.06 + 0.002 and abs(p[1] - (0.02 + 0.06)) < 0.003, (k, p)  # on its floor
    # a ball dropped into a hollow tank through the hole in its top rests at its bottom, inside
    tank = dict(name='Tank', shape='sphere', position=(0.0, 0.6, 0.0), size=(0.6, 0.6, 0.6), hollow=0.03,
                opening=(0.2, 0.2, 0.2), opening_at=(0.0, 0.6, 0.0), material='steel')
    ball = dict(name='Ball', shape='sphere', position=(0.0, 1.5, 0.0), size=(0.1,) * 3, material='wood', dynamic=True)
    S, poses = run(scene_of(tank, ball), 4.0)
    p = track(poses, 1)[-1]
    assert np.linalg.norm(p - (0.0, 0.6, 0.0)) < 0.57 - 0.1 + 0.01 and p[1] < 0.2


def test_a_hollow_cylinders_walls_leave_its_opening_open():
    from blackbody.engine.solids import Solids
    R, H, t = 0.3, 0.5, 0.03
    closed = Solids._hollow_tiles('cylinder', (R, H, R), t, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    door = ((0.1, 0.25, 0.2), (R, -0.2, 0.0))                     # (a door in its side, at +x)
    opened = Solids._hollow_tiles('cylinder', (R, H, R), t, *door)
    for tiles in (closed, opened):
        P = np.concatenate(tiles)
        r = np.hypot(P[:, 0], P[:, 2])
        assert r.max() <= R + 1e-9 and np.abs(P[:, 1]).max() <= H + 1e-9          # within its outline
        for q in tiles:                                                             # each a slab of its wall, t thick
            rq = np.hypot(q[:, 0], q[:, 2])
            assert rq.min() >= R - t - 1e-9 or np.abs(q[:, 1]).min() >= H - t - 1e-9
    mids = np.array([q.mean(0) for q in opened])
    assert not np.any(np.all(np.abs(mids - door[1]) < door[0], axis=1))           # none in the door
    assert len(opened) < len(closed) + 40                                           # (cut where the door is, not finely all over)
    # every way round its side is walled, but for the door
    ang = np.degrees(np.arctan2(mids[:, 2], mids[:, 0]))
    side = np.hypot(mids[:, 0], mids[:, 2]) > R - t
    low = side & (mids[:, 1] < -0.45)                                              # (below the door)
    assert np.histogram(ang[low], bins=12, range=(-180, 180))[0].min() > 0


def test_cloth_meets_a_bowl_by_the_parts_near_it_only(tmp_path, monkeypatch):
    # (every step measured each vertex near it against every plane of every part, 4,600 for this bowl: a sheet over it
    # took the solids from 8 ms a frame to nearly a second)
    from blackbody.engine import solids as SO
    path = str(tmp_path / 'bowl.obj')
    _bowl(path)
    s = scene_of(dict(name='Bowl', shape='mesh', mesh=path, position=(0.0, 0.4, 0.0), size=(1.0, 1.0, 1.0), material='ceramic',
                      dynamic=True))
    S = Solids()
    S.configure(s, ((96, 96, 96), SIZE / 96, (-SIZE / 2, 0.0, -SIZE / 2)))
    S.reset()
    planes = S.bodies[0].planes
    # a sheet draped over it 3 mm off: in its hollow, over its rim, on the floor round it
    g = np.linspace(-0.25, 0.25, 64)
    X, Z = np.meshgrid(g, g, indexing='ij')
    r = np.hypot(X, Z)
    ri, yf = R_BOWL - WALL, -R_BOWL + FOOT + WALL
    y = np.where(r < ri, np.maximum(-np.sqrt(np.maximum(ri * ri - r * r, 0.0)), yf), np.where(r < R_BOWL + 0.01, 0.0, -R_BOWL + FOOT))
    q = np.stack([X.ravel(), y.ravel() + 0.003, Z.ravel()], 1)
    pi, _pp = SO.near_parts(planes, q, SO.CLOTH_MARGIN)
    assert len(pi) < 0.02 * len(q) * len(planes[2])                # a few parts for a vertex near it, none for the rest
    d, n = SO.parts_distance(planes, q, SO.CLOTH_MARGIN)
    d_all, n_all = SO.parts_distance(planes, q)
    near = d_all < SO.CLOTH_MARGIN
    assert near.sum() > 500 and np.array_equal(d[near], d_all[near]) and np.array_equal(n[near], n_all[near])
    assert np.all(d[~near] >= SO.CLOTH_MARGIN)                     # (the rest beyond reach: not touching)
    # and the cloth's contact asks for those only
    reach = []
    whole = SO.parts_distance
    monkeypatch.setattr(SO, 'parts_distance', lambda pl, x, rr=np.inf: reach.append(rr) or whole(pl, x, rr))
    x = q + S.data.xpos[S.bodies[0].body_id]
    S.meet_cloth(x, np.zeros_like(x), np.ones(len(x), bool), 1.0 / s.fps)
    S.advance(s, s.start + 1, 1.0 / s.fps, 1)
    assert reach and set(reach) == {SO.CLOTH_MARGIN}


def _boxes(path, *boxes):
    """Boxes [(low corner, high corner, wound inside out)] in one OBJ, each a shell of its own."""
    v, t = [], []
    tri = np.array([(0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6), (0, 1, 5), (0, 5, 4), (2, 6, 7), (2, 7, 3), (0, 4, 6), (0, 6, 2),
                    (1, 3, 7), (1, 7, 5)])
    for lo, hi, flip in boxes:
        t.append((tri[:, ::-1] if flip else tri) + 8 * len(v))
        v.append([[(hi if (k >> a) & 1 else lo)[a] for a in range(3)] for k in range(8)])
    return _write_obj(path, np.concatenate(v), np.concatenate(t))


def test_a_mesh_of_several_shells_weighs_what_they_take_up(tmp_path):
    # (its volume was its triangles' signed volumes summed: a box wound inside out cancelled the other one, so it weighed
    # nothing and MuJoCo blew up; boxes that overlap were counted twice where they do)
    flipped = _boxes(str(tmp_path / 'flipped.obj'), ((0.0, 0.0, 0.0), (0.2, 0.2, 0.2), False), ((0.3, 0.0, 0.0), (0.5, 0.2, 0.2), True))
    ell = _boxes(str(tmp_path / 'ell.obj'), ((0.0, 0.0, 0.0), (0.4, 0.4, 0.4), False), ((0.2, 0.0, 0.0), (0.8, 0.2, 0.4), False))
    for path, volume in ((flipped, 2 * 0.008), (ell, 0.064 + 0.048 - 0.016)):     # (the L's boxes share 0.016 m^3)
        c = dict(name='Shells', shape='mesh', mesh=path, position=(0.0, 0.3, 0.0), size=(1.0, 1.0, 1.0), material='wood',
                 density=500.0, dynamic=True)
        S, poses = run(scene_of(c), 1.0)
        assert S.model.body_mass[S.bodies[0].body_id] == pytest.approx(500.0 * volume, rel=0.03), path
        p = track(poses, 0)
        assert np.all(np.isfinite(p)) and abs(p[-1][1]) < 0.01            # landed, flat on the ground


def test_a_convex_mesh_is_not_cut_and_cut_parts_are_kept_on_disk(tmp_path, monkeypatch):
    from blackbody.engine import solids as SO
    s = Scene()

    def never(*_a, **_k):
        raise AssertionError('cut again')
    # a ball of 18,000 triangles: convex, so it falls as its hull, uncut (cutting a detailed one into its one part took
    # seconds and a gigabyte)
    r = 0.15
    prof = [(0.0, -r)] + [(r * math.cos(p), r * math.sin(p)) for p in np.linspace(-math.pi / 2, math.pi / 2, 72)[1:-1]] + [(0.0, r)]
    ball = dict(name='Ball', shape='mesh', mesh=_write_obj(str(tmp_path / 'ball.obj'), *_lathe(prof, 128)), size=(1.0, 1.0, 1.0))
    monkeypatch.setattr(SO, 'convex_parts', never)
    parts, how, volume = SO.mesh_parts(s, ball, (1.0, 1.0, 1.0), Solids()._mesh_points(s, ball))
    assert parts is None and how == 'convex' and volume == pytest.approx(4.0 / 3.0 * math.pi * r ** 3, rel=0.01)
    monkeypatch.undo()
    # a bowl is cut once, and its parts read back from the disk in the next session
    bowl = dict(name='Bowl', shape='mesh', mesh=str(tmp_path / 'bowl.obj'), size=(1.0, 1.0, 1.0))
    _bowl(bowl['mesh'])
    hull = Solids()._mesh_points(s, bowl)
    parts, how, volume = SO.mesh_parts(s, bowl, (1.0, 1.0, 1.0), hull)
    assert how == 'parts' and len(parts) > 20
    SO._PARTS.clear()
    monkeypatch.setattr(SO, 'convex_parts', never)
    again = SO.mesh_parts(s, bowl, (1.0, 1.0, 1.0), hull)
    assert again[1:] == (how, volume) and len(again[0]) == len(parts)
    assert all(np.array_equal(a, b) for a, b in zip(again[0], parts))


def test_a_hatch_smaller_than_a_tile_still_lets_things_in():
    # (a hollow's opening dropped only the tiles whose middle was in it: a hatch smaller than one, a tank's 16 cm hatch or
    # a hole in a drum's lid, stayed shut, and a ball dropped through it lay on top)
    tank = dict(name='Tank', shape='sphere', position=(0.0, 0.5, 0.0), size=(0.5,) * 3, hollow=0.03, opening=(0.08,) * 3,
                opening_at=(0.0, 0.5, 0.0), material='steel')
    drum = dict(name='Drum', shape='cylinder', position=(0.0, 0.5, 0.0), size=(1.0, 0.5, 1.0), hollow=0.02,
                opening=(0.08, 0.05, 0.08), opening_at=(0.3, 0.5, 0.2), material='steel')
    for hollow, (x, z), floor in ((tank, (0.0, 0.0), 0.03), (drum, (0.3, 0.2), 0.02)):
        ball = dict(name='Ball', shape='sphere', position=(x, 1.3, z), size=(0.025,) * 3, material='wood', dynamic=True)
        S, poses = run(scene_of(hollow, ball), 2.0)
        p = track(poses, 1)[-1]
        assert abs(p[1] - (floor + 0.025)) < 0.01, (hollow['name'], p)   # through it, resting on the floor inside
