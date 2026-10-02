"""Things that burn, on the GPU: breakable, burnable things catch from the real gas, feed it as they burn and fall in;
and on the stage what fire has done shows (char, embers). The GPU-free part of breaking and burning is in
test_fracture.py."""
import numpy as np


def post_scene(burnable=True, burner_stops=None):
    from blackbody.scene import components
    sc = components.new_scene('fire', 'person')
    sc.emitters = []
    sc.add_emitter(name='Fire', shape='cylinder', position=(0.15, 0.05, 0.0), size=(0.25, 0.05, 0.25), fuel=10, temperature=0.5,
                   **({'stop': burner_stops} if burner_stops else {}))
    sc.add_collider(name='Post', shape='box', position=(0.0, 0.8, 0.0), size=(0.07, 0.8, 0.07), material='wood', breakable=True,
                    burnable=burnable, pieces=14, held='base')
    sc.data['spread'].update(enabled=True, ground=False, burn_time=2.0, catch_temp=0.3, catch_time=0.4, creep=0.06, smoulder=4.0)
    sc.data['domain'].update(size_x=2.4, size_y=2.6, size_z=2.0, resolution=64, preroll=0.0)
    return sc


def run(engine, sc, secs):
    engine.prepare(sc, final=True)
    gas = []
    for f in range(sc.start, sc.start + int(secs * sc.fps) + 1):
        engine.simulate_to(sc, f, cache=False)
        gas.append(engine.stats().get('burning', 0))
    return gas


def test_a_post_by_a_fire_catches_from_the_gas_falls_in_and_feeds_the_fire(engine):
    sc = post_scene(burner_stops=2.0)
    gas = run(engine, sc, 9.0)
    ps = engine.solids.sets[0]
    assert ps.burnable and (ps.fire[:, 1] >= 1.0).all()                       # all of it caught
    assert engine.solids.data.xipos[ps.bodies[~ps.gone]][:, 1].max() < 1.0    # and it fell
    cold = run(engine, post_scene(burnable=False, burner_stops=2.0), 9.0)
    # after the burner has gone out (2 s), the burning post keeps the fire going
    assert sum(gas[24 * 3:]) > 5 * max(sum(cold[24 * 3:]), 1)


def test_the_stage_shows_burnt_wood_darker_than_fresh(engine):
    from blackbody.scene import components
    shots = []
    for burnt in (False, True):
        sc = components.new_scene('fire', 'person')
        sc.emitters = []
        sc.add_collider(name='Crate', shape='box', position=(0.0, 0.3, 0.0), size=(0.3, 0.3, 0.3), material='wood', burnable=True,
                        own_colour=True, colour=(0.6, 0.45, 0.3))
        sc.data['spread'].update(enabled=True, ground=False)
        sc.data['composite'].update(backdrop='stage', floor='concrete')
        sc.data['camera'].update(distance=2.4, target_y=0.3, pitch=10.0, yaw=0.0, use_anchor=False)
        engine.prepare(sc, final=False)
        engine.simulate_to(sc, sc.start, cache=False)
        if burnt:   # (as a burnt-out surface leaves it: no fuel left, burnt out)
            s = engine.solver
            t = s.burn_obj[0]
            n = t.size
            engine.gpu.upload(t, np.tile(np.array([0.0, 1.0, 2.0, 0.0], np.float16), (n[2], n[1], n[0], 1)))
        engine.render(sc, sc.start, (160, 120), samples=1)
        shots.append(engine.display_image()[..., :3].astype(float))
    fresh, black = shots
    crate = np.abs(fresh - black).sum(axis=2) > 20.0
    assert crate.mean() > 0.05 and black[crate].mean() < 0.6 * fresh[crate].mean()
