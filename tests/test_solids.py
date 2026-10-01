"""Rigid bodies (engine/solids.py): things that fall, bounce, slide, stack, push and get pushed, checked against
textbook physics. MuJoCo runs on its own here (no GPU); test_stage.py draws them."""
import math

import numpy as np
import pytest

pytest.importorskip('mujoco')

from blackbody.engine.solids import Solids, attached
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
