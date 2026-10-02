"""Ropes, springs, hinges and ball joints (engine/solids.py Joint, engine/ropes.py), checked against textbook
values. No GPU: MuJoCo runs on its own."""
import math

import numpy as np
import pytest

from blackbody.engine.ropes import coil, rope_curve

G = 9.81


def arc_length(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


@pytest.mark.parametrize('a,b,length', [((0, 3, 0), (2, 1, 0), 3.5), ((0, 3, 0), (2, 3, 0), 2.5), ((0, 3, 0), (0, 1.5, 0), 2.0),
                                        ((0, 0, 0), (1, 0, 0), 10.0)])
def test_a_slack_rope_hangs_in_a_curve_of_its_length(a, b, length):
    p = rope_curve(a, b, length, 96)
    assert np.allclose(p[0], a) and np.allclose(p[-1], b)
    assert abs(arc_length(p) - length) < 2e-3 * length
    assert p[:, 1].min() < min(a[1], b[1])          # it sags below its ends


def test_a_taut_rope_is_straight():
    p = rope_curve((0, 3, 0), (1, 0, 0), 2.0, 8)       # (shorter than the distance: stretched straight)
    assert np.allclose(np.cross(p[1:] - p[0], np.array([1.0, -3.0, 0.0])), 0.0, atol=1e-12)


def test_a_spring_coils_from_end_to_end():
    c = coil((0, 2, 0), (0, 1, 0), 1.0, 0.004)
    assert np.allclose(c[0], (0, 2, 0)) and np.allclose(c[-1], (0, 1, 0))
    assert c[:, 1].max() <= 2.0 + 1e-9 and c[:, 1].min() >= 1.0 - 1e-9


# ---- the physics ---------------------------------------------------------------------------------------------------

pytest.importorskip('mujoco')

from blackbody.engine.solids import Solids, _rotation  # noqa: E402
from blackbody.scene.model import Scene  # noqa: E402


def scene_of(*cols, fps=24):
    s = Scene()
    s.data['domain'].update(ground=True, open_sides=True, preroll=0.0)
    s.data['render']['fps'] = float(fps)
    s.emitters = []
    for c in cols:
        s.add_collider(**c)
    return s


def built(scene):
    S = Solids()
    S.configure(scene, ((96, 96, 96), 12.0 / 96, (-6.0, 0.0, -6.0)))
    S.reset()
    return S


def run(scene, seconds, S=None, first=None):
    """Step through `seconds`: the solids, and every frame's (overrides, rope poses)."""
    S = S or built(scene)
    f0 = scene.start + 1 if first is None else first
    out = []
    for f in range(f0, f0 + int(round(seconds * scene.fps))):
        S.advance(scene, f, 1.0 / scene.fps, 1)
        out.append((S.overrides(), S.rope_poses()))
    return S, out


def crossings(x, fps):
    """Times x passes down through its mean."""
    x = np.asarray(x) - np.mean(x)
    k = np.nonzero((x[:-1] > 0.0) & (x[1:] <= 0.0))[0]
    return (k + 1 + x[k] / (x[k] - x[k + 1])) / fps


def test_a_pendulum_swings_at_its_period():
    L, r, th = 2.0, 0.03, math.radians(10.0)
    d = L + r
    ball = dict(name='Bob', shape='sphere', position=(d * math.sin(th), 4.0 - d * math.cos(th), 0.0), size=(r, r, r), material='steel',
                joint='rope', joint_anchor=(0.0, 4.0, 0.0))
    S, out = run(scene_of(ball, fps=48), 6.0)
    assert S.joints[0].kind == 'rope' and abs(S.joints[0].length - L) < 1e-6
    T = float(np.diff(crossings([o[0]['pos'][0] for o, _ in out], 48)).mean())
    ideal = 2.0 * math.pi * math.sqrt(d / G) * (1.0 + th * th / 16.0)
    assert abs(T - ideal) < 0.004 * ideal


def test_a_rope_goes_slack_but_never_stretches():
    ball = dict(name='Ball', shape='sphere', position=(0.3, 2.0, 0.0), size=(0.1, 0.1, 0.1), material='rubber', joint='rope',
                joint_anchor=(0.0, 3.0, 0.0), rope_length=1.2, start_velocity=(0.0, 4.0, 0.0))
    S, out = run(scene_of(ball), 3.0)
    dist = np.array([float(np.linalg.norm(r[0]['a'] - r[0]['b'])) for _, r in out])
    assert dist.max() < 1.2 * 1.01
    assert dist.min() < 0.9            # it was thrown up toward its anchor: slack
    assert abs(dist[-1] - 1.2) < 0.012  # and hangs from it in the end


def test_a_door_turns_on_its_hinges_and_settles():
    door = dict(name='Door', shape='box', position=(0.0, 1.01, 0.0), size=(0.45, 1.0, 0.02), material='wood', joint='hinge',
                joint_at=(-0.45, 0.0, 0.0), joint_axis=(0.0, 1.0, 0.0), start_spin=(0.0, 90.0, 0.0), joint_friction=0.3)
    S, out = run(scene_of(door), 4.0)
    yaw, edge, tilt = [], [], []
    for o, _ in out:
        R = _rotation(o[0]['quat'])
        edge.append(np.asarray(o[0]['pos']) + R @ np.array([-0.45, 0.0, 0.0]))
        yaw.append(math.atan2(R[0, 2], R[2, 2]))
        tilt.append(math.degrees(math.acos(min(1.0, R[1, 1]))))
    assert np.linalg.norm(np.array(edge) - (-0.45, 1.01, 0.0), axis=1).max() < 0.002
    assert max(tilt) < 0.5
    # pushed at 90 degrees a second about its hinge, slowing at 0.3 a second: 90 / 0.3 (1 - e^-0.3t) degrees
    turned = math.degrees(abs(np.unwrap(yaw)[-1]))
    assert abs(turned - 300.0 * (1.0 - math.exp(-1.2))) < 6.0


def test_a_ball_joint_keeps_its_pivot():
    box = dict(name='Bob', shape='box', position=(0.0, 2.0, 0.0), size=(0.1, 0.1, 0.1), material='steel', joint='ball',
               joint_at=(0.0, 0.5, 0.0), start_velocity=(1.0, 0.0, 0.5), joint_friction=0.0)
    S, out = run(scene_of(box), 3.0)
    pivot = np.array([np.asarray(o[0]['pos']) + _rotation(o[0]['quat']) @ (0.0, 0.5, 0.0) for o, _ in out])
    assert np.linalg.norm(pivot - (0.0, 2.5, 0.0), axis=1).max() < 0.002
    swing = np.ptp([o[0]['pos'] for o, _ in out], axis=0)
    assert swing[0] > 0.2 and swing[2] > 0.1          # it swings both ways about it


def test_a_spring_bounces_at_its_period_and_sags_under_its_weight():
    m = 7850.0 * 0.2 * 0.08 * 0.2
    k = 2000.0
    weight = dict(name='Weight', shape='box', position=(0.0, 1.9, 0.0), size=(0.1, 0.04, 0.1), material='steel', joint='spring',
                  joint_anchor=(0.0, 3.0, 0.0), rope_length=1.0, spring_k=k)
    S, out = run(scene_of(weight, fps=60), 8.0)
    y = np.array([o[0]['pos'][1] for o, _ in out])
    T = float(np.diff(crossings(y, 60)).mean())
    ideal = 2.0 * math.pi * math.sqrt(m / k) / math.sqrt(1.0 - 0.05 ** 2)
    assert abs(T - ideal) < 0.01 * ideal
    assert abs(y[-60:].mean() - (3.0 - 1.0 - m * G / k - 0.04)) < 0.003
    assert np.ptp(y[-60:]) < 0.1 * np.ptp(y[:60])     # and its bounce dies away


def test_a_rope_snaps_when_jerked_but_holds_what_hangs_on_it():
    m = 4.0 / 3.0 * math.pi * 0.085 ** 3 * 7850.0
    ball = dict(name='Ball', shape='sphere', size=(0.085, 0.085, 0.085), material='steel', joint='rope', joint_anchor=(0.0, 3.0, 0.0),
                rope_length=1.0, joint_break=3.0 * m * G)
    S1, out1 = run(scene_of(dict(ball, position=(0.0, 2.0 - 0.085, 0.0))), 2.0)
    S2, out2 = run(scene_of(dict(ball, position=(0.9, 3.0, 0.0))), 2.0)
    assert not S1.snaps and abs(out1[-1][0][0]['pos'][1] - (2.0 - 0.085)) < 0.002
    assert len(S2.snaps) == 1 and out2[-1][1][0]['broken'] == 1.0
    assert out2[-1][0][0]['pos'][1] < 0.1                  # it fell to the ground


def test_a_hinge_tears_out_of_a_door_too_heavy_for_it():
    door = dict(name='Door', shape='box', position=(0.0, 1.5, 0.0), size=(0.45, 1.0, 0.02), material='wood', joint='hinge',
                joint_at=(-0.45, 0.0, 0.0))
    weight = 0.9 * 2.0 * 0.04 * 600.0 * G                         # (a 43 kg door, hung half a metre off the ground)
    S1, out1 = run(scene_of(dict(door, joint_break=2.0 * weight)), 1.0)
    S2, out2 = run(scene_of(dict(door, joint_break=0.5 * weight)), 1.0)
    assert not S1.snaps and abs(out1[-1][0][0]['pos'][1] - 1.5) < 0.002
    assert len(S2.snaps) == 1 and out2[-1][0][0]['pos'][1] < 1.1


def test_something_hung_from_a_moving_object_goes_with_it():
    sc = scene_of(dict(name='Arm', shape='box', position=(0.0, 4.0, 0.0), size=(0.5, 0.1, 0.1), material='steel'),
                  dict(name='Ball', shape='sphere', position=(0.0, 2.5, 0.0), size=(0.2, 0.2, 0.2), material='steel', joint='rope',
                       joint_to='Arm'))
    sc.set_key(('collider', 0, 'position'), sc.start, (0.0, 4.0, 0.0))
    sc.set_key(('collider', 0, 'position'), sc.start + 48, (2.0, 4.0, 0.0))
    S, out = run(sc, 5.0)
    assert np.allclose(out[-1][1][1]['b'], (2.0, 4.0, 0.0), atol=1e-3)     # the rope's top went with the arm
    assert abs(out[-1][0][1]['pos'][0] - 2.0) < 0.5


def test_two_objects_tied_together():
    a = dict(name='A', shape='box', position=(0.0, 0.25, 0.0), size=(0.25, 0.25, 0.25), material='wood', dynamic=True,
             start_velocity=(5.0, 0.0, 0.0))
    b = dict(name='B', shape='box', position=(-1.0, 0.25, 0.0), size=(0.25, 0.25, 0.25), material='wood', joint='rope', joint_to='A',
             rope_length=1.0)
    S, out = run(scene_of(a, b), 2.0)
    assert S.joints[0].other == 'body0'
    assert out[-1][0][1]['pos'][0] > -0.5                       # A dragged B along


def test_a_snapped_rope_is_kept_with_the_state_and_mended_by_a_reset():
    ball = dict(name='Ball', shape='sphere', position=(0.9, 3.0, 0.0), size=(0.085, 0.085, 0.085), material='steel', joint='rope',
                joint_anchor=(0.0, 3.0, 0.0), rope_length=1.0, joint_break=600.0)
    sc = scene_of(ball)
    S, _ = run(sc, 0.25)
    early = S.state()
    S, _ = run(sc, 0.75, S, first=sc.start + 7)
    later = S.state()
    assert S.joints[0].broken
    S2 = built(sc)
    assert S2.load_state(later) and S2.joints[0].broken and not S2.model.tendon_limited[0]
    S3 = built(sc)
    S3.load_state(early)
    S3, _ = run(sc, 0.75, S3, first=sc.start + 7)
    assert np.allclose(S3.overrides()[0]['pos'], S.overrides()[0]['pos'])
    S.reset()
    assert not S.joints[0].broken and S.model.tendon_limited[0]


def test_joints_are_deterministic():
    door = dict(name='Door', shape='box', position=(0.0, 1.01, 0.0), size=(0.45, 1.0, 0.02), material='wood', joint='hinge',
                joint_at=(-0.45, 0.0, 0.0), start_spin=(0.0, 60.0, 0.0))
    ball = dict(name='Ball', shape='sphere', position=(1.5, 1.2, 0.0), size=(0.1, 0.1, 0.1), material='steel', joint='rope',
                joint_anchor=(2.0, 2.5, 0.0), start_velocity=(0.0, 0.0, 1.0))
    _, a = run(scene_of(door, ball), 1.0)
    _, b = run(scene_of(door, ball), 1.0)
    assert all(np.array_equal(x[0][i]['pos'], y[0][i]['pos']) for x, y in zip(a, b) for i in (0, 1))


def test_a_rope_to_nothing_by_that_name_hangs_from_a_fixed_point():
    ball = dict(name='Ball', shape='sphere', position=(0.0, 1.0, 0.0), size=(0.1, 0.1, 0.1), material='steel', joint='rope',
                joint_to='Nowhere', joint_anchor=(0.0, 2.0, 0.0))
    S, out = run(scene_of(ball), 1.0)
    assert any('Nowhere' in w for w in S.warnings)
    assert S.joints[0].other == '' and abs(out[-1][0][0]['pos'][1] - 1.0) < 0.01


def test_a_thing_started_inside_something_fixed_is_warned_about():
    post = dict(name='Post', shape='box', position=(0.0, 1.0, 0.0), size=(0.1, 1.0, 0.1), material='wood')
    ball = dict(name='Ball', shape='sphere', position=(0.15, 1.0, 0.0), size=(0.1, 0.1, 0.1), material='steel', dynamic=True)
    S = built(scene_of(post, ball))
    assert any('Ball starts inside Post' in w for w in S.warnings)


# ---- motors ----------------------------------------------------------------------------------------------------------

def wheel(**kw):
    """A 60 cm steel disc on its side (its axle along -x), on a fixed axle 1.5 m up, driven at 60 rpm with 50 N m."""
    return dict(dict(name='Wheel', shape='cylinder', position=(0.0, 1.5, 0.0), size=(0.3, 0.05, 0.3), roll=90.0, material='steel',
                     joint='hinge', joint_axis=(0.0, 1.0, 0.0), motor_speed=60.0, motor_torque=50.0, joint_friction=0.0), **kw)


def test_a_motor_spins_a_wheel_up_with_its_strength_to_its_speed():
    S, out = run(scene_of(wheel(), fps=50), 2.5)
    I = 0.5 * (7850.0 * math.pi * 0.3 ** 2 * 0.1) * 0.3 ** 2            # (a 222 kg disc: 10 kg m^2 about its axle)
    w = np.array([o[0]['omega'][:3] for o, _ in out])
    spin = w @ (-1.0, 0.0, 0.0)
    t = np.arange(1, len(spin) + 1) / 50.0
    early = t <= 1.0
    assert np.polyfit(t[early], spin[early], 1)[0] == pytest.approx(50.0 / I, rel=0.01)   # spun up at its torque over its inertia
    assert abs(spin[-1] - 2.0 * math.pi) < 0.01                         # then held at 60 turns a minute
    assert np.abs(w - np.outer(spin, (-1.0, 0.0, 0.0))).max() < 1e-6     # about its axle only
    assert np.abs(np.array([o[0]['pos'] for o, _ in out]) - (0.0, 1.5, 0.0)).max() < 1e-3


def test_a_motor_too_weak_for_its_load_stalls():
    # a 79 kg steel arm hinged at one end, held out level: its weight turns it down with m g L / 2 = 385 N m
    arm = dict(name='Arm', shape='box', position=(0.5, 1.5, 0.0), size=(0.5, 0.05, 0.05), material='steel', joint='hinge',
               joint_at=(-0.5, 0.0, 0.0), joint_axis=(0.0, 0.0, 1.0), motor_speed=10.0, joint_friction=0.0)
    tip = {}
    for torque in (800.0, 200.0):
        S, out = run(scene_of(dict(arm, motor_torque=torque), fps=50), 1.0)
        tip[torque] = [float((np.asarray(o[0]['pos']) + _rotation(o[0]['quat']) @ (0.5, 0.0, 0.0))[1]) for o, _ in out]
    assert tip[800.0][-1] > 1.5 + math.sin(math.radians(50.0))            # lifted at 10 rpm (60 degrees a second)
    assert max(tip[200.0]) < 1.5 + 0.01 and min(tip[200.0]) < 1.0         # too weak: it swings down


def test_a_motor_speed_keyed_down_to_nothing_brakes_it():
    sc = scene_of(wheel(), fps=50)
    sc.set_key(('collider', 0, 'motor_speed'), sc.start + 75, 60.0)
    sc.set_key(('collider', 0, 'motor_speed'), sc.start + 100, 0.0)
    S, out = run(sc, 3.0)
    spin = np.array([o[0]['omega'][:3] for o, _ in out]) @ (-1.0, 0.0, 0.0)
    assert abs(spin[70] - 2.0 * math.pi) < 0.05 and abs(spin[-1]) < 0.01


def test_a_cart_on_motored_wheels_drives_at_their_speed():
    chassis = dict(name='Chassis', shape='box', position=(0.0, 0.25, 0.0), size=(0.3, 0.06, 0.5), material='wood', dynamic=True)
    wheels = [dict(name=f'Wheel {k}', shape='cylinder', position=(sx * 0.36, 0.15, sz * 0.35), size=(0.15, 0.04, 0.15), roll=90.0,
                   material='rubber', joint='hinge', joint_to='Chassis', joint_axis=(0.0, 1.0, 0.0), motor_speed=60.0,
                   motor_torque=20.0, joint_friction=0.0)
              for k, (sx, sz) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1)))]
    S, out = run(scene_of(chassis, *wheels, fps=50), 3.0)
    assert not S.warnings
    p = np.array([o[0]['pos'] for o, _ in out])
    v = np.array([o[0]['vel'] for o, _ in out])
    # rolling: the wheels' rim speed (60 rpm on a 15 cm radius), forward (-z: the axles point -x), straight and level
    assert v[-1][2] == pytest.approx(-2.0 * math.pi * 0.15, rel=0.03)
    assert np.abs(p[:, 0]).max() < 0.01 and np.abs(p[:, 1] - 0.25).max() < 0.02
    assert _rotation(out[-1][0][0]['quat'])[1, 1] > 0.999


def test_the_cart_preset_jumps_its_ramp_bowls_the_tower_over_and_brakes():
    from blackbody.scene import presets
    sc = presets.make('cart_jump')
    S, out = run(sc, 5.0)
    assert not S.warnings
    names = [c['name'] for c in sc.colliders]
    cart = np.array([o[0]['pos'] for o, _ in out])
    assert cart[:, 1].max() > 0.6                                         # up the ramp and off its end
    assert out[-1][0][names.index('Block 6')]['pos'][1] < 0.5              # the tower is down
    assert np.linalg.norm(out[-1][0][0]['vel']) < 0.05 and 1.5 < cart[-1, 0] < 3.2   # stopped short of its barrier
    # its fire rides on it (a link), and applying the preset to a shot brings its links instead of the shot's
    assert sc.links == [{'child': ['emitter', 'Cart fire'], 'parent': ['collider', 'Cart'], 'offset': [0.0, 0.22, 0.0]}]
    shot = presets.make('campfire')
    shot.links = [{'child': ['emitter', 'Log A'], 'parent': ['collider', 'Cart'], 'offset': [0.0, 0.0, 0.0]}]
    presets.apply_to(shot, 'cart_jump')
    assert shot.links == sc.links
