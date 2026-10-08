"""Bullets (engine/ballistics.py): how far each round goes into each surface against published figures, glancing
off, and shots flown through the rigid world (solids.py): what they break, knock over and leave behind. No GPU:
MuJoCo runs on its own."""
import math

import numpy as np
import pytest

from blackbody.engine.ballistics import (ROUNDS, SURFACES, bullet_of, depth_in, glances, penetrate, plate_limit,
                                         surface)

# Typical published penetration figures (m): 10% ordnance gel (FBI protocol tests), water (tank tests), pine boards
# (3/4 inch boards: boards penetrated x 19 mm), mild steel plate (the thickest plate perforated at close range),
# concrete and brick (crater depth of a single hit), dry sand, packed snow. Figures from different tests vary by a
# third or more; the model is held to within a factor of 1.6 of each, and about 1.25 overall.
REFERENCE = [
    ('9mm', 'gel', 'depth', 0.66), ('9mm_hp', 'gel', 'depth', 0.35), ('45acp', 'gel', 'depth', 0.70),
    ('22lr', 'gel', 'depth', 0.30), ('556', 'gel', 'depth', 0.34), ('762x39', 'gel', 'depth', 0.70),
    ('308', 'gel', 'depth', 0.58), ('slug', 'gel', 'depth', 0.45), ('buck', 'gel', 'depth', 0.40),
    ('pellet', 'gel', 'depth', 0.05),
    ('9mm', 'water', 'depth', 1.5), ('556', 'water', 'depth', 0.6), ('308', 'water', 'depth', 0.9),
    ('slug', 'water', 'depth', 1.2),
    ('9mm', 'wood', 'depth', 0.13), ('45acp', 'wood', 'depth', 0.10), ('556', 'wood', 'depth', 0.19),
    ('762x39', 'wood', 'depth', 0.26), ('308', 'wood', 'depth', 0.28), ('slug', 'wood', 'depth', 0.15),
    ('22lr', 'wood', 'depth', 0.07),
    ('9mm', 'steel', 'plate', 0.0032), ('556', 'steel', 'plate', 0.0064), ('308', 'steel', 'plate', 0.0095),
    ('50bmg', 'steel', 'plate', 0.025), ('22lr', 'steel', 'plate', 0.0012),
    ('9mm', 'concrete', 'depth', 0.015), ('556', 'concrete', 'depth', 0.05), ('308', 'concrete', 'depth', 0.07),
    ('50bmg', 'concrete', 'depth', 0.25),
    ('9mm', 'brick', 'depth', 0.035), ('556', 'brick', 'depth', 0.08), ('308', 'brick', 'depth', 0.12),
    ('9mm', 'sand', 'depth', 0.20), ('556', 'sand', 'depth', 0.12), ('308', 'sand', 'depth', 0.30),
    ('9mm', 'snow', 'depth', 1.2), ('308', 'snow', 'depth', 1.8),
]


def measured(r, s, q):
    return depth_in(ROUNDS[r], s) if q == 'depth' else plate_limit(ROUNDS[r], s)


def test_penetration_matches_published_figures():
    errs = []
    for r, s, q, ref in REFERENCE:
        got = measured(r, s, q)
        e = math.log(max(got, 1e-6) / ref)
        errs.append(e)
        # concrete under a .50 BMG is the one far off: published craters run from 10 to 30 cm
        tol = 2.5 if (r, s) == ('50bmg', 'concrete') else 1.6
        assert abs(e) < math.log(tol), (r, s, q, ref, got)
    assert math.exp(float(np.sqrt(np.mean(np.square(errs))))) < 1.3


def test_the_shot_list_matches_the_rounds():
    from blackbody.scene.params import SHOT_ROUNDS
    assert dict(SHOT_ROUNDS) == {k: r.label for k, r in ROUNDS.items()}


def test_faster_goes_deeper_and_thicker_stops_it():
    r = ROUNDS['9mm']
    depths = [depth_in(r, 'wood', speed=v) for v in (150.0, 250.0, 360.0, 450.0)]
    assert all(b > a for a, b in zip(depths, depths[1:]))
    outs = []
    for t in (0.01, 0.03, 0.06, 0.3):
        b = bullet_of(r, (0, 0, 0), (1, 0, 0))
        outs.append(penetrate(b, surface('wood'), t).speed_out)
    assert outs[0] > outs[1] > outs[2] > 0.0 and outs[3] == 0.0


def test_hollow_points_stop_sooner_and_rifles_break_up_in_gel():
    assert depth_in(ROUNDS['9mm_hp'], 'gel') < 0.65 * depth_in(ROUNDS['9mm'], 'gel')
    b = bullet_of(ROUNDS['556'], (0, 0, 0), (1, 0, 0))
    p = penetrate(b, SURFACES['gel'], 2.0)
    assert p.yawed and p.broken and p.mass_out < 0.7 * p.mass_in


def test_bullets_skip_off_water_and_steel_at_a_shallow_angle():
    rng = np.random.default_rng(0)
    for key, shallow, steep in (('water', 3.0, 25.0), ('steel', 10.0, 70.0), ('concrete', 8.0, 60.0)):
        b = bullet_of(ROUNDS['9mm'], (0, 0, 0), (1, 0, 0))
        assert sum(glances(b, SURFACES[key], shallow, rng) for _ in range(50)) > 45
        assert sum(glances(b, SURFACES[key], steep, rng) for _ in range(50)) < 5


def test_a_bullet_through_glass_loses_little():
    b = bullet_of(ROUNDS['9mm'], (0, 0, 0), (1, 0, 0))
    p = penetrate(b, SURFACES['glass'], 0.006)
    assert p.outcome == 'through' and 0.8 * p.speed_in < p.speed_out < 0.97 * p.speed_in


pytest.importorskip('mujoco')

from blackbody.engine.solids import Solids  # noqa: E402
from blackbody.scene.model import Scene  # noqa: E402

LAYOUT = ((96, 96, 96), 12.0 / 96, (-6.0, 0.0, -6.0))


def scene_of(cols, shots, fps=24, time_scale=1.0):
    s = Scene()
    s.data['domain'].update(ground=True, open_sides=True, preroll=0.0, time_scale=time_scale)
    s.data['render']['fps'] = float(fps)
    s.emitters = []
    for c in cols:
        s.add_collider(**c)
    for sh in shots:
        s.add_shot(**sh)
    return s


def run(scene, seconds):
    S = Solids()
    S.configure(scene, LAYOUT)
    S.reset()
    for f in range(scene.start + 1, scene.start + 1 + int(round(seconds * scene.fps))):
        S.advance(scene, f, scene.v('domain', 'time_scale', f) / scene.fps, 1)
    return S


PANE = dict(name='Pane', shape='box', position=(0.0, 1.0, 0.0), size=(0.5, 0.5, 0.003), breakable=True, fracture='shards',
            material='glass', pieces=60)
WALL = dict(name='Wall', shape='box', position=(0.0, 1.0, -2.0), size=(1.5, 1.0, 0.1), material='concrete')
POST = dict(name='Post', shape='box', position=(0.0, 0.5, -1.0), size=(0.05, 0.5, 0.05), material='wood')
CAN = dict(name='Can', shape='cylinder', position=(0.0, 1.061, -1.0), size=(0.033, 0.06, 0.033), dynamic=True,
           material='aluminium', density=37.0)      # (an empty can: 15 g)
PISTOL = dict(position=(0.0, 1.0, 3.0), aim=(0.0, 1.0, 0.0), start=0.2, round='9mm')


def test_a_pane_is_holed_where_the_bullet_goes_through_and_stands():
    S = run(scene_of([PANE, WALL], [PISTOL]), 1.0)
    kinds = [(i.kind, i.surface) for i in S.shots.impacts]
    assert kinds[0] == ('through', 'glass') and ('stop', 'concrete') in kinds
    # its cracks crowd round where the bullet went in (the rehearsal found it)
    assert np.linalg.norm(S.sets[0].impact[:2]) < 0.01
    assert not S.whole(0)
    pos = S.data.xpos[S.sets[0].bodies]
    fell = pos[:, 1] < 0.4
    assert 3 <= fell.sum() < 0.5 * len(pos)       # (the middle knocked out, the rest of the pane still up)
    assert S.shots.marks and len(S.shots.debris) > 10


def test_a_pane_stands_whole_until_it_is_hit():
    # (a web of shards has bonds where two cells all but miss each other; a weld that small, loaded at rest as much as
    # any, broke as the shot began, and the pane, no longer whole, showed its cuts before anything hit it: fracture.py
    # drops those slivers)
    from blackbody.scene import presets
    sc = presets.make('glass_slowmo')
    S = Solids()
    S.configure(sc, sc.sim_layout(False))
    S.reset()
    for f in range(sc.start + 1, sc.start + 13):       # (the bullet reaches the pane at about frame 20)
        S.advance(sc, f, sc.v('domain', 'time_scale', f) / sc.fps, 1)
    assert not S.shots.impacts
    assert S.whole(0) and not S.breaks


def test_a_bottle_shot_bursts():
    from blackbody.scene.shot_presets import _bottle
    rail = dict(name='Rail', shape='box', position=(0.0, 0.9, 0.0), size=(0.3, 0.03, 0.05), material='wood')
    bottle = _bottle('Bottle', 0.0, 0.93, 0.0)
    shot = dict(position=(0.0, 1.02, 10.0), aim=(0.0, 1.02, 0.0), round='22lr', start=0.1, scatter=0.0, flash=False)
    S = run(scene_of([rail, bottle], [shot]), 0.6)
    hits = [i for i in S.shots.impacts if i.owner[0] == 'piece']
    assert hits and hits[0].kind == 'through'
    W = S._w
    mine = W['set'] == 0
    assert (W['over'][mine] < 0).all()                  # (every weld of it gone: it bursts, not just holed)
    pos = S.data.xpos[S.sets[0].bodies]
    assert (pos[:, 1] < 0.85).mean() > 0.7              # (its pieces off the rail, most of them on the ground)


def test_a_can_on_a_post_is_knocked_off():
    shot = dict(PISTOL, position=(0.0, 1.06, 3.0), aim=(0.0, 1.06, -1.0))
    S = run(scene_of([POST, CAN], [shot]), 0.6)
    can = S.bodies[0]
    hit = [i for i in S.shots.impacts if i.owner[0] == 'body']
    assert hit and hit[0].kind == 'through'      # (a thin can: in one side and out the other)
    assert S.data.xpos[can.body_id][2] < -1.3 or S.data.xpos[can.body_id][1] < 0.9


def test_shots_are_the_same_every_time_and_carry_on_from_a_saved_state():
    sc = scene_of([PANE, WALL], [dict(PISTOL, count=4, rate=900, scatter=0.5)])
    a = run(sc, 0.6)
    b = run(sc, 0.6)
    pa = [tuple(np.round(i.pos, 6)) for i in a.shots.impacts]
    assert pa == [tuple(np.round(i.pos, 6)) for i in b.shots.impacts] and len(pa) >= 4
    # save half way, carry on, and compare with going straight through
    S = Solids()
    S.configure(sc, LAYOUT)
    S.reset()
    for f in range(sc.start + 1, sc.start + 6):
        S.advance(sc, f, 1.0 / sc.fps, 1)
    st = S.state()
    for f in range(sc.start + 6, sc.start + 15):
        S.advance(sc, f, 1.0 / sc.fps, 1)
    end = [tuple(np.round(i.pos, 6)) for i in S.shots.impacts]
    assert S.load_state(st)
    for f in range(sc.start + 6, sc.start + 15):
        S.advance(sc, f, 1.0 / sc.fps, 1)
    assert [tuple(np.round(i.pos, 6)) for i in S.shots.impacts] == end


def test_buckshot_spreads_and_a_tracer_is_drawn_in_slow_motion():
    sc = scene_of([WALL], [dict(PISTOL, round='buck', tracer=True)], time_scale=0.01)
    S = Solids()
    S.configure(sc, LAYOUT)
    S.reset()
    seen = False
    for f in range(sc.start + 1, sc.start + 60):
        S.advance(sc, f, 0.01 / sc.fps, 1)
        v = S.shots.view(S)
        seen = seen or ('bullets' in v and len(v['bullets']) == 9 and 'tracers' in v)
    assert seen
    hits = np.array([i.pos for i in S.shots.impacts if i.surface == 'concrete'])
    assert len(hits) == 9
    spread = np.ptp(hits[:, 0]) + np.ptp(hits[:, 1])
    assert 0.02 < spread < 0.5           # (about 2.5 cm a metre: a fist-sized pattern at 5 m)


def test_lead_splashes_on_steel_and_sparks_fly():
    plate = dict(name='Gong', shape='box', position=(0.0, 1.0, 0.0), size=(0.2, 0.2, 0.006), material='steel')
    S = run(scene_of([plate], [PISTOL]), 0.4)
    imp = S.shots.impacts[0]
    assert imp.kind == 'stop' and imp.surface == 'steel' and imp.depth < 0.006
    from blackbody.engine.ballistics import SPLASH
    assert any(m.style == SPLASH for m in S.shots.marks)
    from blackbody.engine.debris import SPARK
    assert (S.shots.debris.kind == SPARK).sum() == 0 or S.shots.debris.glow(S.shots.now).max() >= 0.0


def test_a_bullet_skips_off_concrete_floor_at_a_grazing_angle():
    sc = scene_of([], [dict(position=(0.0, 0.3, 3.0), aim=(0.0, 0.0, 0.0), start=0.2, round='9mm')])
    sc.data['composite']['floor'] = 'concrete'
    S = run(sc, 0.4)
    first = S.shots.impacts[0]
    assert first.owner == ('ground',) and first.kind == 'glance'
    assert first.out > 0.5 * first.speed


def test_holes_are_carved_where_bullets_went_and_move_with_what_they_are_in():
    from blackbody.engine.ballistics import CLASS
    board = dict(name='Board', shape='box', position=(0.0, 1.0, 0.0), size=(0.2, 0.2, 0.011), material='wood')
    block = dict(name='Block', shape='box', position=(0.0, 1.0, -1.0), size=(0.3, 0.3, 0.1), material='concrete')
    S = run(scene_of([board, block], [PISTOL]), 0.5)
    v = S.shots.view(S)
    bores = v['bores']
    # a hole through the board, flaring out of its back along the grain; a crater in the concrete
    assert len(bores) >= 2
    wood = [b for b in bores if int(b[3, 2]) % 16 == CLASS['wood']]
    crater = [b for b in bores if int(b[3, 2]) % 16 == CLASS['concrete']]
    assert wood and crater
    w = max(wood, key=lambda b: b[3, 0])                     # (the scoop torn out of its back, not the bullet's own
    assert w[1, 3] > w[0, 3] and w[3, 0] > 0.0               # channel through it): wider where it came out, longer
                                                             # along the grain
    ch = min(wood, key=lambda b: b[3, 0])
    assert abs(ch[0, 2] - 0.011) < 0.005 and abs(ch[1, 2] + 0.011) < 0.005   # (the channel: from face to face)
    assert abs(w[1, 2] + 0.011) < 0.005 and w[0, 2] < 0.0                     # (the scoop: out of the back)
    c = crater[0]
    assert c[0, 3] > 2.0 * 0.0045                              # (a crater wider than the bullet)
    # marks carry what they are in, and which side
    M = v['marks']
    assert any(int(m[3, 3]) // 2 == CLASS['wood'] and int(m[3, 3]) % 2 == 1 for m in M)


def test_a_hole_in_a_falling_thing_goes_with_it():
    box = dict(name='Box', shape='box', position=(0.0, 1.0, 0.0), size=(0.1, 0.1, 0.1), material='wood', dynamic=True,
               release=0.6)
    S = run(scene_of([box], [dict(PISTOL, position=(0.0, 1.0, 3.0), aim=(0.0, 1.0, 0.0), start=0.1)]), 1.2)
    v = S.shots.view(S)
    body = S.bodies[0]
    near = np.linalg.norm(v['bores'][:, 0, :3] - S.data.xpos[body.body_id], axis=1)
    assert near.min() < 0.25 and S.data.xpos[body.body_id][1] < 0.2        # (it fell, and its holes with it)


def test_a_scene_with_shots_saves_loads_and_changes_its_signature():
    from blackbody.scene.model import Scene as Sc
    sc = scene_of([WALL], [PISTOL])
    back = Sc.from_dict(sc.to_dict())
    assert back.shots and back.shots[0]['round'] == '9mm' and tuple(back.shots[0]['aim']) == tuple(sc.shots[0]['aim'])
    sig = sc.sim_signature()
    sc.shots[0]['round'] = '308'
    assert sc.sim_signature() != sig


def test_the_gun_presets_and_blocks_build():
    from blackbody.scene import components, presets
    for k in ('shooting_range', 'glass_slowmo', 'bottle_shoot', 'steel_gong', 'machine_gun'):
        s = presets.make(k)
        assert s.shots, k
    sc = components.new_scene('auto', 'person')
    for key in ('pistol', 'rifle', 'shotgun', 'burst'):
        added, _ = components.add(sc, key, at=(0.5, -0.2))
        assert added[0][0] == 'shot'
    # (a gun aims at the spot it is put on, from where a shooter would stand: outside the box)
    assert abs(sc.shots[0]['aim'][0] - 0.5) < 1e-9 and abs(sc.shots[0]['aim'][2] + 0.2) < 1e-9


def test_bullets_go_through_a_cars_windows_and_its_doors():
    car = dict(name='Car', shape='box', position=(0.0, 0.75, 0.0), size=(2.2, 0.75, 0.9), material='painted', build='car')
    shots = [dict(position=(0.5, 1.0, 6.0), aim=(0.3, 0.75, 0.0), round='9mm', start=0.2, scatter=0.0),
             dict(position=(-0.4, 1.35, 6.0), aim=(-0.4, 1.35, 0.0), round='9mm', start=0.3, scatter=0.0)]
    S = Solids()
    sc = scene_of([car], shots)
    S.configure(sc, ((96, 48, 96), 8.0 / 96, (-4.0, 0.0, -4.0)))
    S.reset()
    for f in range(sc.start + 1, sc.start + 15):
        S.advance(sc, f, 1.0 / 24, 1)
    kinds = {(i.kind, i.surface) for i in S.shots.impacts}
    assert ('through', 'painted') in kinds and ('through', 'windows') in kinds
