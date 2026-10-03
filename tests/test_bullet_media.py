"""Bullets into what lives on the GPU (engine/bullet_media.py): sand and gel (the matter), water (the liquid), fabric.
Needs a GPU (the engine fixture): not in CI's list."""
import numpy as np

from blackbody.scene import components


def shot_scene(kind='auto'):
    sc = components.new_scene(kind, 'person')
    sc.emitters = []
    sc.data['domain'].update(size_x=1.0, size_y=1.0, size_z=1.0, resolution=32, preroll=0.0, disk_cache=False)
    return sc


def matter_ke(eng):
    m = eng._matter
    P = np.frombuffer(eng.gpu.read_buffer(m._buf['P'], size=m.count * 128), np.float32).reshape(-1, 32)
    live = P[:, 3] >= 0.0
    return float((P[live, 4:7] ** 2).sum()), P[live]


def test_a_bullet_into_gel_goes_through_it_slowing_and_sets_it_moving(engine):
    sc = shot_scene()
    sc.data['domain'].update(time_scale=0.02, matter_detail=96)
    sc.add_collider(name='Table', shape='box', position=(0.0, 0.35, 0.0), size=(0.3, 0.05, 0.3), material='wood')
    sc.add_matter(material='gel', shape='box', position=(0.0, 0.48, 0.0), size=(0.08, 0.08, 0.2))
    sc.add_shot(position=(0.0, 0.48, 3.0), aim=(0.0, 0.48, 0.0), round='9mm', start=0.0005, scatter=0.0, flash=False)
    engine.prepare(sc, final=False)
    for f in range(sc.start, sc.start + 13):
        engine.simulate_to(sc, f, cache=False)
    imps = [i for i in engine.solids.shots.impacts if i.surface == 'gel']
    assert imps and imps[0].kind == 'through' and imps[0].out < 0.8 * imps[0].speed      # (40 cm of gel: through, slower)
    assert abs(imps[0].depth - 0.4) < 0.03
    ke, _ = matter_ke(engine)
    assert ke > 50.0           # (the gel round its track thrown outward: its temporary cavity)


def test_a_bullet_into_sand_stops_in_it_and_throws_it_up(engine):
    sc = shot_scene()
    sc.data['domain'].update(time_scale=0.05, matter_detail=96)
    sc.add_matter(material='sand', shape='box', position=(0.0, 0.1, 0.0), size=(0.3, 0.1, 0.3))
    sc.add_shot(position=(0.0, 1.2, 2.0), aim=(0.0, 0.2, 0.0), round='9mm', start=0.001, scatter=0.0, flash=False)
    engine.prepare(sc, final=False)
    for f in range(sc.start, sc.start + 16):
        engine.simulate_to(sc, f, cache=False)
    imps = [i for i in engine.solids.shots.impacts if i.surface == 'sand']
    assert imps and imps[0].kind == 'stop' and 0.05 < imps[0].depth < 0.4
    _ke, P = matter_ke(engine)
    m = engine._matter
    y = m.origin[1] + P[:, 1] * m.dx
    assert (y > 0.22).sum() > 50     # (grains thrown up out of the hole: in 25 ms, a couple of centimetres up and rising)


def test_a_bullet_into_water_splashes_and_slows(engine):
    sc = shot_scene('liquid')
    sc.data['domain'].update(time_scale=0.05, preroll=0.3)
    sc.data['liquid'].update(water_level=0.3, settle=True)
    sc.add_shot(position=(0.0, 1.3, 1.7), aim=(0.0, 0.3, 0.0), round='9mm', start=0.002, scatter=0.0, flash=False)
    engine.prepare(sc, final=False)
    for f in range(sc.start, sc.start + 10):
        engine.simulate_to(sc, f, cache=False)
    imps = [i for i in engine.solids.shots.impacts if i.surface == 'water']
    assert imps and imps[0].out < imps[0].speed
    engine.liquid.measure()
    assert engine.liquid.max_speed > 2.0      # (the splash)


def test_a_bullet_holes_a_curtain(engine):
    sc = shot_scene()
    sc.data['domain'].update(size_x=2.0, size_y=2.0, size_z=2.0)
    sc.add_fabric(name='Curtain', width=1.2, height=1.6, position=(0.0, 1.0, 0.0), pins='top', material='cotton')
    sc.add_shot(position=(0.0, 1.0, 4.0), aim=(0.0, 1.0, 0.0), round='buck', start=0.2)
    engine.prepare(sc, final=False)
    for f in range(sc.start, sc.start + 9):
        engine.simulate_to(sc, f, cache=False)
    _x, gone = engine.cloth.positions()
    assert gone.sum() >= 1
