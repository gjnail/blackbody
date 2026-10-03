"""People and cars (engine/assemblies.py in engine/solids.py): a person stands braced until hit and falls limp; a car
rides on its springs, drives at its keyed speed and steers. MuJoCo on its own (no GPU)."""
import math

import numpy as np
import pytest

pytest.importorskip('mujoco')

from blackbody.engine.assemblies import assembly
from blackbody.engine.solids import Solids
from blackbody.scene.anim import Curve
from blackbody.scene.model import Scene

SIZE = 40.0   # m: the simulation box (open sides: only the ground holds things)


def scene_of(*cols, fps=50):
    s = Scene()
    s.data['domain'].update(ground=True, open_sides=True, preroll=0.0)
    s.data['render']['fps'] = float(fps)
    s.emitters = []
    for c in cols:
        s.add_collider(**c)
    return s


def run(scene, seconds):
    S = Solids()
    n = 96
    S.configure(scene, ((n, n, n), SIZE / n, (-SIZE / 2, 0.0, -SIZE / 2)))
    S.reset()
    for f in range(scene.start + 1, scene.start + 1 + int(round(seconds * scene.fps))):
        S.advance(scene, f, 1.0 / scene.fps, 1)
    return S


PERSON = dict(name='Person', shape='box', position=(0.0, 0.9, 0.0), size=(0.25, 0.9, 0.15), build='figure', material='person')
CAR = dict(name='Car', shape='box', position=(0.0, 0.75, 0.0), size=(2.2, 0.75, 0.9), build='car', material='painted')


def head(S):
    return S.piece_poses()[0]['pos'][2]


def test_a_person_is_about_a_person():
    A = assembly('figure', (0.25, 0.9, 0.15))
    assert len(A.bodies) == 13
    mass = sum(p.volume * d for p, d in zip(A.pieces, A.density))
    assert 65.0 < mass < 95.0
    top = max(float(p.verts[:, 1].max()) for p in A.pieces)
    assert 0.8 < top <= 0.9


def test_a_person_stands_braced_and_falls_limp():
    S = run(scene_of(PERSON), 3.0)
    assert head(S)[1] > 1.55                                          # braced, on its feet
    S = run(scene_of(dict(PERSON, stance='limp')), 3.0)
    assert head(S)[1] < 0.5                                           # limp, in a heap


def test_a_person_hit_hard_goes_limp_and_falls():
    ball = dict(name='Ball', shape='sphere', position=(-1.2, 1.3, 0.0), size=(0.12, 0.12, 0.12), dynamic=True, material='steel',
                start_velocity=(6.0, 0.0, 0.0))
    S = run(scene_of(PERSON, ball), 3.0)
    assert S.asms[0]['limp']
    assert head(S)[1] < 0.6
    # nudged (softly), it stays up
    S = run(scene_of(PERSON, dict(ball, start_velocity=(0.6, 0.0, 0.0))), 3.0)
    assert not S.asms[0]['limp'] and head(S)[1] > 1.5


def test_a_car_rides_on_its_springs_and_drives_and_steers():
    s = scene_of(dict(CAR, drive='rear'))
    s.colliders[0]['drive_speed'] = 30.0
    s.colliders[0]['steer'] = Curve([[s.start, 0.0, 'linear'], [s.start + 200, 0.0, 'linear'], [s.start + 210, 20.0, 'linear']])
    S = run(s, 4.0)
    body = S.piece_poses()[0]['pos'][0]
    assert abs(body[1] - (-0.13 + 0.75)) < 0.03                       # on its springs, at its height
    v = np.linalg.norm(S.piece_poses()[0]['vel'][0]) * 3.6
    assert 25.0 < v < 31.0                                            # at its speed
    S2 = run(s, 6.0)
    assert S2.piece_poses()[0]['pos'][0][2] < -0.5                    # steered left (-z)
    # braked (0 km/h), it stays put
    S3 = run(scene_of(dict(CAR, drive='rear', drive_speed=0.0)), 2.0)
    assert np.linalg.norm(S3.piece_poses()[0]['pos'][0][[0, 2]]) < 0.05
