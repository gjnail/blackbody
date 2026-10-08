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


# ---- a rope over a post, a chain ------------------------------------------------------------------------------------

def test_a_rope_goes_over_a_post_and_holds_what_hangs_from_it():
    post = dict(name='Post', shape='cylinder', position=(0.0, 2.0, 0.0), size=(0.08, 0.6, 0.08), pitch=90.0, material='wood')
    crate = dict(name='Crate', shape='box', position=(-0.6, 1.2, 0.0), size=(0.15, 0.15, 0.15), dynamic=True, material='wood',
                 joint='rope', joint_anchor=(0.6, 0.3, 0.0), rope_length=3.2)
    S, poses = run(scene_of(post, crate, fps=50), 4.0)
    xs = np.array([o[1]['pos'][0] for o, _r in poses])
    ys = np.array([o[1]['pos'][1] for o, _r in poses])
    assert ys.min() > 0.4                                    # held up by the rope over the post (not dropped)
    assert xs.min() < -0.3 and xs.max() > 0.2                # swinging under it
    for _o, r in poses:                                      # and the rope is drawn over it every frame, never through it
        path = r[1]['path']
        q = np.concatenate([np.linspace(a, b, 20) for a, b in zip(path[:-1], path[1:])])
        assert path[:, 1].max() > 2.02 and np.hypot(q[:, 0], q[:, 1] - 2.0).min() > 0.08


def test_a_chain_hangs_with_its_weight_and_holds_its_load():
    box = dict(name='Box', shape='box', position=(0.8, 1.5, 0.0), size=(0.15, 0.15, 0.15), dynamic=True, material='wood',
               density=740.0, joint='rope', rope_look='chain', rope_thickness=0.012, joint_anchor=(0.0, 2.5, 0.0),
               rope_length=1.2)
    S, poses = run(scene_of(box, fps=50), 3.0)
    path = S.rope_poses()[0]['path']
    assert np.allclose(path[-1], (0.0, 2.5, 0.0), atol=0.01)            # held at its far end
    links = np.linalg.norm(np.diff(path, axis=0), axis=1)
    assert np.allclose(links, links.mean(), rtol=0.05) and abs(links.sum() - 1.2) < 0.06   # links that are links
    ys = np.array([o[0]['pos'][1] for o, _r in poses])
    assert ys.min() > 0.9                                     # it holds the box up, swinging below it
    m = S.model
    chain_mass = sum(float(m.body_mass[b]) for b in S.joints[0].link_ids)
    assert 2.5 < chain_mass < 4.5                             # 12 mm chain: about 2.9 kg a metre


# ---- several joints on one object, ropes over several posts, ropes with weight ---------------------------------------

def _tilt(quat):
    return math.degrees(math.acos(min(1.0, float(_rotation(quat)[1, 1]))))


def test_a_plank_hangs_level_from_two_ropes():
    from blackbody.engine.solids import JOINT_SLOT
    plank = dict(name='Plank', shape='box', position=(0.0, 2.5, 0.0), size=(0.9, 0.03, 0.15), material='wood', joint='rope',
                 joint_at=(-0.8, 0.03, 0.0), joint_anchor=(-0.8, 4.5, 0.0),
                 joints=[dict(joint='rope', joint_at=(0.8, 0.03, 0.0), joint_anchor=(0.8, 4.5, 0.0))])
    S, out = run(scene_of(plank, fps=50), 3.0)
    assert not S.warnings and [(jt.kind, jt.slot) for jt in S.joints] == [('rope', 0), ('rope', 1)]
    assert max(_tilt(o[0]['quat']) for o, _ in out) < 0.5                      # level, end to end
    assert max(abs(o[0]['pos'][1] - 2.5) for o, _ in out) < 0.005             # held where it hangs
    ropes = out[-1][1]
    assert sorted(ropes) == [0, JOINT_SLOT]                                     # both drawn
    for r in ropes.values():
        assert abs(float(np.linalg.norm(r['a'] - r['b'])) - 1.97) < 0.01         # each taut at its length
    # (on its first rope alone it swings down from that end toward hanging on end)
    one = dict(plank, joints=[])
    S1, out1 = run(scene_of(one, fps=50), 3.0)
    assert max(_tilt(o[0]['quat']) for o, _ in out1) > 60.0


def test_a_rope_goes_over_two_posts():
    posts = [dict(name=f'Post {k}', shape='cylinder', position=(x, 2.0, 0.0), size=(0.08, 0.6, 0.08), pitch=90.0, material='wood')
             for k, x in ((1, -0.5), (2, 0.5))]
    crate = dict(name='Crate', shape='box', position=(-0.5, 1.2, 0.0), size=(0.15, 0.15, 0.15), material='wood', joint='rope',
                 joint_anchor=(0.6, 0.3, 0.0))
    # named in Goes over, its length as it is round them at the start: it hangs where it is, the rope over both
    S, out = run(scene_of(*posts, dict(crate, rope_over='Post 1, Post 2'), fps=50), 3.0)
    jt = S.joints[0]
    assert not S.warnings and jt.posts == ('fixed0_0', 'fixed1_0')
    assert 3.55 < jt.length < 3.65                         # up to Post 1, across to Post 2 and down, round their tops
    ys = np.array([o[2]['pos'][1] for o, _ in out])
    assert np.abs(ys - 1.2).max() < 0.02
    path = out[-1][1][2]['path']
    for x in (-0.5, 0.5):
        assert path[np.abs(path[:, 0] - x) < 0.05, 1].max() > 2.07                # over the top of each
    # found by itself: a rope that long reaches round both, so it goes over both
    S2, out2 = run(scene_of(*posts, dict(crate, rope_length=jt.length + 0.01), fps=50), 3.0)
    assert S2.joints[0].posts == ('fixed0_0', 'fixed1_0')
    assert min(o[2]['pos'][1] for o, _ in out2) > 1.15


def test_a_rope_with_weight_drapes_over_posts_and_weighs_what_it_does():
    from blackbody.engine.solids import ROPE_KG_M3
    posts = [dict(name=f'Post {k}', shape='cylinder', position=(x, 2.0, 0.0), size=(0.08, 0.6, 0.08), pitch=90.0, material='wood')
             for k, x in ((1, -0.5), (2, 0.5))]
    crate = dict(name='Crate', shape='box', position=(-0.5, 1.2, 0.0), size=(0.15, 0.15, 0.15), material='wood', joint='rope',
                 joint_anchor=(0.6, 0.3, 0.0), rope_over='Post 1, Post 2', rope_heavy=True, rope_thickness=0.025)
    S, out = run(scene_of(*posts, crate, fps=50), 3.0)
    jt = S.joints[0]
    assert not S.warnings and jt.links and jt.look == 0                         # links of rope, drawn as rope
    mass = sum(float(S.model.body_mass[b]) for b in jt.link_ids)
    ideal = ROPE_KG_M3['rope'] * math.pi * 0.0125 ** 2 * jt.length              # (about 0.4 kg a metre)
    assert abs(mass - ideal) < 0.1 * ideal
    ys = np.array([o[2]['pos'][1] for o, _ in out])
    assert ys.min() > 1.05 and abs(ys[-1] - ys[-25]) < 0.01                      # held up by the rope over the posts, settled
    path = out[-1][1][2]['path']
    assert path[:, 1].max() > 2.08 and path[:, 1].max() < 2.2                    # lying on them
    # it gives way where it is tied when it is pulled harder than its Breaks at (the crate weighs 25 N)
    S2, out2 = run(scene_of(*posts, dict(crate, joint_break=10.0), fps=50), 2.0)
    assert len(S2.snaps) == 1 and S2.joints[0].broken
    assert out2[-1][0][2]['pos'][1] < 0.2                                         # the crate fell
    assert out2[-1][1][2]['path'][:, 1].max() > 2.0                              # and the rope still hangs over the posts


def test_a_chain_snaps_where_it_is_tied():
    box = dict(name='Box', shape='box', position=(0.8, 1.5, 0.0), size=(0.15, 0.15, 0.15), dynamic=True, material='wood',
               joint='rope', rope_look='chain', rope_thickness=0.012, joint_anchor=(0.0, 2.5, 0.0), rope_length=1.2,
               joint_break=100.0)
    S, out = run(scene_of(box, fps=50), 2.0)
    assert S.snaps and out[-1][0][0]['pos'][1] < 0.2                  # jerked taut past 100 N as it swung: dropped
    assert np.allclose(out[-1][1][0]['path'][-1], (0.0, 2.5, 0.0), atol=0.01)   # the chain still hangs from its anchor


def test_more_joints_are_added_saved_and_follow_renames():
    from blackbody.engine.solids import joint_specs
    from blackbody.scene import components as C
    sc = scene_of(dict(name='Beam', shape='box', position=(0.0, 3.0, 0.0), size=(1.0, 0.05, 0.05), material='wood'),
                  dict(name='Seat', shape='box', position=(0.0, 1.0, 0.0), size=(0.5, 0.02, 0.15), material='wood'))
    C.add_joint(sc, 1, 'rope', to=0)
    notes = C.add_joint(sc, 1, 'rope', to=0, more=True)
    c = sc.colliders[1]
    assert notes and c['dynamic'] and len(joint_specs(c)) == 2
    # its first rope moved from its middle to one end, the second at the other, each tied straight above on the beam
    (k1, a), (k2, b) = joint_specs(c)
    assert np.allclose(a['joint_at'], (-0.5, 0.02, 0.0)) and np.allclose(b['joint_at'], (0.5, 0.02, 0.0))
    assert np.allclose(a['joint_to_at'], (-0.5, -0.05, 0.0)) and np.allclose(b['joint_to_at'], (0.5, -0.05, 0.0))
    S, out = run(sc, 2.0)
    assert len(S.joints) == 2 and max(_tilt(o[1]['quat']) for o, _ in out) < 0.5
    # saved and opened again
    back = Scene.from_dict(sc.to_dict())
    assert joint_specs(back.colliders[1])[1][1]['joint_to'] == 'Beam' and back.sim_signature() == sc.sim_signature()
    # renamed (the viewer's Rename): every joint follows
    C.rename_in_joints(c, {'Beam': 'Bar'})
    assert [s['joint_to'] for _k, s in joint_specs(c)] == ['Bar', 'Bar']
    # and taken off: all of them
    C.add_joint(sc, 1, on=False)
    assert not joint_specs(sc.colliders[1])


def test_a_rope_told_to_go_over_what_is_not_a_post_says_so():
    crate = dict(name='Crate', shape='box', position=(0.0, 1.0, 0.0), size=(0.15, 0.15, 0.15), material='wood', joint='rope',
                 joint_anchor=(0.0, 2.5, 0.0), rope_over='Nowhere, Shelf')
    shelf = dict(name='Shelf', shape='box', position=(1.0, 2.0, 0.0), size=(0.3, 0.02, 0.2), material='wood')
    S = built(scene_of(crate, shelf))
    assert any('Nowhere (there is no such object)' in w for w in S.warnings)
    assert any('Shelf (only a fixed or falling cylinder' in w for w in S.warnings)
    assert S.joints[0].posts == ()


def test_a_keyed_motor_drives_only_the_first_of_its_joints():
    # (a further joint took the object's keyed Motor speed, a curve, as a number: building the whole scene failed)
    sc = scene_of(wheel(motor_speed=0.0, joints=[dict(joint='ball')]), fps=50)
    sc.set_key(('collider', 0, 'motor_speed'), sc.start + 25, 0.0)
    sc.set_key(('collider', 0, 'motor_speed'), sc.start + 50, 60.0)
    S, out = run(sc, 2.0)
    assert not S.warnings and [(jt.kind, jt.keyed) for jt in S.joints] == [('hinge', True), ('ball', False)]
    spin = np.array([o[0]['omega'][:3] for o, _ in out]) @ (-1.0, 0.0, 0.0)
    assert abs(spin[20]) < 0.05 and abs(spin[-1] - 2.0 * math.pi) < 0.05       # held, then spun up as keyed
    # a further hinge turns at the speed it gives, if any (not the object's: two motors on one axle)
    S2 = built(scene_of(wheel(joints=[dict(joint='hinge', motor_speed=30.0), dict(joint='hinge')]), fps=50))
    assert [jt.motor for jt in S2.joints] == pytest.approx([2.0 * math.pi, math.pi, 0.0])


def _beam_and_crate(z, **kw):
    """A 2.24 m beam along z, 2.46 m up (chain_swing's), and a crate under one side of it, z along it, on a rope from
    the ground on its other side."""
    beam = dict(name='Beam', shape='cylinder', position=(-0.1, 2.46, 0.0), size=(0.07, 1.12, 0.07), pitch=90.0, material='wood')
    crate = dict(name='Crate', shape='box', position=(-0.75, 1.5, z), size=(0.18, 0.18, 0.18), dynamic=True, material='wood',
                 joint='rope', joint_anchor=(0.55, 0.0, z))
    return scene_of(beam, dict(crate, **kw))


def test_a_rope_finds_a_beam_wherever_along_it_it_goes_over():
    # (the search measured the way round a beam from its middle, and from where the crate's side faced the far end:
    # off its middle a rope 0.7 m longer than the way round it went straight through it, and one told to go over it was
    # tied skewed toward its middle)
    taut = built(_beam_and_crate(0.0, rope_over='Beam')).joints[0].length
    for z in (0.0, 0.5, 0.9):
        S = built(_beam_and_crate(z, rope_over='Beam'))
        assert abs(S.joints[0].length - taut) < 0.002                       # the same way round wherever it goes over
        a = S.model.site('joint1a').pos
        assert a[1] == pytest.approx(0.18) and abs(a[2]) < 1e-6              # tied on its top, square under the beam
        assert built(_beam_and_crate(z, rope_length=taut + 0.01)).joints[0].posts == ('fixed0_0',)   # found
        assert built(_beam_and_crate(z, rope_length=taut - 0.05)).joints[0].posts == ()              # too short to


def test_a_rope_finds_only_a_fixed_level_bar_by_itself():
    """A falling log above a slack rope, or a bar sloping 30 degrees, is not gone round unless it is named in Goes over:
    MuJoCo takes a cylinder as endless and turns the side it goes round with a falling one, and either flung the load
    (most such ropes gained energy, up to millions of J/kg)."""
    taut = built(_beam_and_crate(0.0, rope_over='Beam')).joints[0].length
    assert built(_beam_and_crate(0.0, rope_length=taut + 0.01)).joints[0].posts == ('fixed0_0',)   # level and fixed

    def with_beam(**beam):
        sc = _beam_and_crate(0.0, rope_length=taut + 0.3)
        sc.colliders[0].update(beam)
        return built(sc).joints[0].posts

    assert with_beam(pitch=60.0) == ()                         # sloping 30 degrees
    assert with_beam(dynamic=True) == ()                       # falling
    sc = _beam_and_crate(0.0, rope_length=taut + 0.3, rope_over='Beam')
    sc.colliders[0].update(dynamic=True)
    assert built(sc).joints[0].posts == ('bodygeom0',)         # named: gone round all the same


def test_a_rope_tied_against_a_post_does_not_go_round_it():
    # (chain_swing's chain hangs from just under its beam: it was laid round the beam, and the weight swung short of the
    # crates it is there to knock over)
    import mujoco
    from blackbody.scene import presets
    sc = presets.make('chain_swing')
    names = [c['name'] for c in sc.colliders]
    S = built(sc)
    chain = next(jt for jt in S.joints if jt.index == names.index('Weight'))
    mujoco.mj_kinematics(S.model, S.data)
    assert S.data.xpos[chain.link_ids][:, 1].max() < 2.46 - 0.07           # its links laid under the beam, not over it
    S, out = run(sc, 3.0, S=S)
    assert max(o[names.index('Weight')]['pos'][0] for o, _ in out) > 0.6    # into the crates
    assert out[-1][0][names.index('Crate 4')]['pos'][1] < 0.5               # and the stack is down


def test_a_slack_rope_swings_past_posts_beside_and_under_it():
    # (a slack rope went round any post it could reach round taut, on the side away from its line: round an upright pole
    # beside it or under a bar below it. Swung across the post, MuJoCo's rope jumped round to that side and flung the
    # crate away)
    def energy(*cols, seconds=3.0):
        """The crate's (the last object's) posts, and its energy per kg at the start and at most (J/kg)."""
        S, out = run(scene_of(*cols, fps=50), seconds)
        k = len(cols) - 1
        e = [G * o[k]['pos'][1] + 0.5 * float(np.dot(o[k]['vel'], o[k]['vel'])) for o, _ in out]
        return S.joints[0].posts, e[0], max(e)

    crate = dict(name='Crate', shape='box', position=(1.6, 2.0, 0.0), size=(0.1, 0.1, 0.1), dynamic=True, material='wood',
                 joint='rope', joint_anchor=(0.0, 3.0, 0.0), rope_length=2.0, start_velocity=(0.0, 0.0, 1.0))
    poles = [dict(name=f'Pole {k}', shape='cylinder', position=(0.8, 1.5, z), size=(0.05, 1.5, 0.05), material='wood')
             for k, z in enumerate((0.3, -0.3))]
    posts, e0, most = energy(*poles, crate)
    assert posts == () and most < e0 + 0.1                  # upright poles either side of it, pushed sideways between them
    under = dict(name='Bar', shape='cylinder', position=(0.55, 2.4, 0.0), size=(0.04, 0.4, 0.04), pitch=90.0, material='steel')
    posts, e0, most = energy(under, dict(crate, rope_length=2.3, start_velocity=(0.0, 0.0, 0.0)))
    assert posts == () and most < e0 + 0.1                  # a bar 20 cm under its line
    # a bar it hangs over is still gone over (and keeps its energy too), unless it is told to go over none
    over = dict(under, position=(0.55, 2.9, 0.0))
    posts, e0, most = energy(over, dict(crate, rope_length=2.3, start_velocity=(0.0, 0.0, 0.0)))
    assert posts == ('fixed0_0',) and most < e0 + 0.1
    S = built(scene_of(over, dict(crate, rope_length=2.3, rope_over='None')))
    assert S.joints[0].posts == () and not S.warnings       # (none: not an object it cannot find)


def test_a_short_rope_just_reaching_round_its_pulley_is_not_warned_about():
    # (a pulley is gone round when the rope is up to 2 cm short of reaching round it, but a rope 2% short of where it is
    # tied was warned about: a 60 cm one 1.5 cm short was both)
    pulley = dict(name='Pulley', shape='cylinder', position=(0.0, 1.0, 0.0), size=(0.04, 0.05, 0.04), pitch=90.0, material='steel')
    crate = dict(name='Crate', shape='box', position=(-0.1, 0.7, 0.0), size=(0.05, 0.05, 0.05), dynamic=True, material='wood',
                 joint='rope', joint_at=(0.0, 0.05, 0.0), joint_anchor=(0.1, 0.7, 0.0))
    taut = built(scene_of(pulley, dict(crate, rope_over='Pulley'))).joints[0].length
    assert taut < 0.75
    S = built(scene_of(pulley, dict(crate, rope_length=taut - 0.015)))
    assert S.joints[0].posts == ('fixed0_0',) and not S.warnings
    S = built(scene_of(pulley, dict(crate, rope_over='Pulley', rope_length=taut - 0.05)))
    assert any('yanked in' in w for w in S.warnings)                     # (told to go over it, far too short: said)


def test_more_ropes_go_to_free_corners_and_tie_to_a_mesh_by_its_shape(tmp_path):
    from itertools import combinations
    from blackbody.engine.solids import joint_specs
    from blackbody.scene import components as C
    # a branch modelled as a mesh, 40 x 10 x 10 cm, at its own size (Size scales it: 1)
    v = np.array([[(0.2 if k & 1 else -0.2), (0.05 if k & 2 else -0.05), (0.05 if k & 4 else -0.05)] for k in range(8)])
    t = np.array([(0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6), (0, 1, 5), (0, 5, 4), (2, 6, 7), (2, 7, 3), (0, 4, 6), (0, 6, 2),
                  (1, 3, 7), (1, 7, 5)])
    path = tmp_path / 'branch.obj'
    path.write_text(''.join(f'v {x} {y} {z}\n' for x, y, z in v) + ''.join(f'f {a + 1} {b + 1} {c + 1}\n' for a, b, c in t))
    sc = scene_of(dict(name='Branch', shape='mesh', mesh=str(path), position=(0.0, 3.0, 0.0), size=(1.0, 1.0, 1.0), material='wood'),
                  dict(name='Seat', shape='box', position=(0.0, 1.0, 0.0), size=(0.5, 0.02, 0.15), material='wood'))
    C.add_joint(sc, 1, 'rope', to=0)
    assert np.allclose(sc.colliders[1]['joint_to_at'], (0.0, -0.05, 0.0))    # (on its underside, not 1 m below it)
    C.add_joint(sc, 1, 'rope', to=0, more=True)
    C.add_joint(sc, 1, 'rope', to=0, more=True)
    specs = joint_specs(sc.colliders[1])
    ats = [np.asarray(cj['joint_at'], float) for _k, cj in specs]
    assert len(ats) == 3 and min(float(np.linalg.norm(a - b)) for a, b in combinations(ats, 2)) > 0.3   # (the third not on the first)
    assert np.allclose(ats[2], (0.0, 0.02, 0.15)) or np.allclose(ats[2], (0.0, 0.02, -0.15))            # a free corner
    for _k, cj in specs[1:]:                                                  # each tied straight above, on its underside
        to = np.asarray(cj['joint_to_at'], float)
        assert to[1] == pytest.approx(-0.05) and abs(to[0] - min(max(cj['joint_at'][0], -0.2), 0.2)) < 1e-6
    S, out = run(sc, 1.0)
    assert len(S.joints) == 3 and not S.warnings


def test_renames_follow_joints_whatever_their_case():
    from blackbody.engine.solids import joint_specs
    from blackbody.scene import components as C
    c = dict(name='Seat', joint='rope', joint_to='beam', rope_over='POST, Post 2',
             joints=(dict(joint='rope', joint_to='Beam'), dict(joint='rope', joint_to='Bar')))
    C.rename_in_joints(c, {'Beam': 'Gallows', 'Post': 'Pole'}, ['Gallows', 'Pole', 'Post 2', 'Bar', 'Seat'])
    assert [cj['joint_to'] for _k, cj in joint_specs(c)] == ['Gallows', 'Gallows', 'Bar'] and c['rope_over'] == 'Pole, Post 2'
    # (but one that is exactly another object's name stays its)
    c = dict(name='Seat', joint='rope', joint_to='beam')
    C.rename_in_joints(c, {'Beam': 'Gallows'}, ['Gallows', 'beam', 'Seat'])
    assert c['joint_to'] == 'beam'


def test_joined_by_nothing_takes_its_more_joints_off_too():
    from blackbody.engine.solids import falls, joint_specs, joined
    plank = dict(name='Plank', shape='box', position=(0.0, 2.5, 0.0), size=(0.9, 0.03, 0.15), material='wood', joint='none',
                 joints=[dict(joint='rope', joint_at=(0.8, 0.03, 0.0), joint_anchor=(0.8, 4.5, 0.0))])
    sc = scene_of(plank)
    c = sc.colliders[0]
    # (they were left hidden and working: the object still hung, the menus and the viewer had it as unjoined)
    assert not joint_specs(c) and joined(c) is None and not falls(c)
    assert not built(sc).joints
    # Properties: set to Nothing, they go with it, in the same undo step
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    QtWidgets = pytest.importorskip('PySide6.QtWidgets')
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from blackbody.ui.document import Document
    doc = Document()
    doc.scene = scene_of(dict(plank, joint='rope', joint_anchor=(-0.8, 4.5, 0.0), joint_at=(-0.8, 0.03, 0.0)))
    told = []
    doc.message.connect(told.append)
    doc.set(('collider', 0, 'joint'), 'none')
    assert doc.scene.colliders[0]['joints'] == () and any('More joints' in t for t in told)
    doc.undo.undo()
    assert doc.scene.colliders[0]['joint'] == 'rope' and len(doc.scene.colliders[0]['joints']) == 1
    del app


def test_more_joints_from_the_command_line(capsys):
    import argparse
    from blackbody.cli import _apply_setting, cmd_settings
    sc = scene_of(dict(name='Plank', shape='box', position=(0.0, 2.5, 0.0), size=(0.9, 0.03, 0.15), joint='rope'))
    assert _apply_setting(sc, 'collider.0.joints=[{"joint": "rope", "joint_at": [0.8, 0.03, 0]}]') is None
    assert sc.colliders[0]['joints'] == ({'joint': 'rope', 'joint_at': [0.8, 0.03, 0]},)
    assert 'JSON list' in _apply_setting(sc, 'collider.0.joints=[{"joint": "rope"')       # (it was taken as none at all)
    assert 'JSON list' in _apply_setting(sc, 'collider.0.joints={"joint": "rope"}')
    assert len(sc.colliders[0]['joints']) == 1
    cmd_settings(argparse.Namespace(section='collider'))
    line = next(x for x in capsys.readouterr().out.splitlines() if x.startswith('collider.N.joints'))
    assert 'JSON list of joint settings' in line and '"x y z"' not in line
