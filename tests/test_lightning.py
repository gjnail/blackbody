"""Lightning (engine/lightning.py and the Lightning kind of light): its channel, its flash, the light it gives and the
fire it starts. No GPU."""
import numpy as np

from blackbody.engine.lightning import bolt, brightness, strokes
from blackbody.scene.model import Scene


def test_a_bolt_runs_from_its_top_to_where_it_strikes_jagged_like_lightning():
    for seed in range(5):
        channels = bolt((0.0, 6.0, 0.0), (0.5, 0.0, 0.3), seed=seed, branching=0.6)
        main = channels[0][0]
        assert np.allclose(main[0], (0.0, 6.0, 0.0)) and np.allclose(main[-1], (0.5, 0.0, 0.3))
        winding = np.linalg.norm(np.diff(main, axis=0), axis=1).sum() / np.linalg.norm((0.5, -6.0, 0.3))
        assert 1.1 < winding < 1.7           # (lightning's path is 1.3 to 1.5 times its straight length)
        assert len(channels) > 3 and all(first_only for _p, _t, _b, first_only in channels[1:])
        assert all(b < 1.0 and t < 1.0 for _p, t, b, _f in channels[1:])          # branches: thinner and dimmer
    assert np.array_equal(bolt((0, 6, 0), (0, 0, 0), seed=2)[0][0], bolt((0, 6, 0), (0, 0, 0), seed=2)[0][0])


def test_it_flashes_a_few_times_and_goes_dark():
    ss = strokes(0.5, 3, seed=1)
    assert len(ss) == 3 and ss[0] == (0.5, 1.0)
    assert all(0.03 < b[0] - a[0] < 0.12 for a, b in zip(ss, ss[1:]))
    assert brightness(0.49, 0.5, 3, 1) == (0.0, 0.0)
    peak = brightness(0.5, 0.5, 3, 1)[0]
    assert peak > 0.9
    assert brightness(0.52, 0.5, 3, 1)[0] < 0.5 * peak          # each flash fades in tens of milliseconds
    assert brightness(ss[1][0], 0.5, 3, 1)[1] < 0.2              # the branches only in the first
    assert brightness(1.5, 0.5, 3, 1) == (0.0, 0.0)


def lightning_scene(kind='fire', **kw):
    s = Scene()
    s.data['domain']['kind'] = kind
    s.emitters = []
    s.add_light(**dict(dict(kind='lightning', position=(0.0, 5.0, 0.0), end=(0.3, 0.0, 0.0), intensity=5e5, strike_at=0.5), **kw))
    return s


def test_its_flash_lights_the_set_only_while_it_lasts():
    s = lightning_scene()
    at = s.start + 0.5 * s.fps
    lamps = s.lamps(at)
    assert len(lamps) == Scene.LIGHTNING_LAMPS and all(l['kind'] == 'point' and l['power'][0] > 0.0 for l in lamps)
    ys = sorted(l['position'][1] for l in lamps)
    assert ys[0] < 2.0 < ys[-1]                                  # spread down its channel
    assert s.lamps(at - 3) == [] and s.lamps(at + 24) == []
    assert s.bolts(at) and not s.bolts(at - 3)


def test_it_sets_fire_where_it_strikes():
    s = lightning_scene()
    at = s.start + 0.5 * s.fps
    fires = [e for e in s.emitters_gpu(at + 1) if e.pos == (0.3, 0.0, 0.0)]
    assert len(fires) == 1 and fires[0].fuel > 0.0
    assert not [e for e in s.emitters_gpu(at + 12) if e.pos == (0.3, 0.0, 0.0)]     # (a quarter of a second)
    assert not lightning_scene(ignites=False).lightning_fires(at + 1)
    assert not lightning_scene('liquid').lightning_fires(at + 1)                    # (no fire in a liquid scene)
    # where it strikes and when matter to the simulation, so they are in its signature
    a, b = lightning_scene(), lightning_scene(end=(0.5, 0.0, 0.0))
    assert a.sim_signature() != b.sim_signature()
    assert lightning_scene(ignites=False).sim_signature() == lightning_scene(ignites=False, end=(0.5, 0.0, 0.0)).sim_signature()
