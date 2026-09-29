"""GPU tests: solver stability and incompressibility, determinism, rendering sanity."""
import numpy as np
import pytest

from blackbody.scene import presets


def _divergence(solver):
    v = solver.gpu.read(solver.vel[0]).astype(np.float64)
    return ((v[:-1, :-1, 1:, 0] - v[:-1, :-1, :-1, 0]) + (v[:-1, 1:, :-1, 1] - v[:-1, :-1, :-1, 1])
            + (v[1:, :-1, :-1, 2] - v[:-1, :-1, :-1, 2])) / solver.h


@pytest.mark.parametrize('name', ['campfire', 'fireball', 'flamethrower'])
def test_presets_stay_stable(engine, name):
    sc = presets.make(name)
    sc.data['domain']['resolution'] = 64
    engine.prepare(sc, final=False)
    for f in range(sc.start, sc.start + 60):
        engine.simulate_to(sc, f, cache=False)
        assert np.isfinite(engine.solver.max_speed)
        assert engine.solver.max_speed < 150.0, f'{name} frame {f}: {engine.solver.max_speed:.1f} m/s'
    sc_arr = engine.solver.read_scalars().astype(np.float32)
    assert np.isfinite(sc_arr).all()
    assert sc_arr.min() >= 0.0


def test_projection_removes_divergence(engine):
    sc = presets.make('campfire')
    sc.data['domain']['resolution'] = 64
    sc.data['combustion']['expansion'] = 0.0    # with no expansion the flow must be divergence free
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 20, cache=False)
    d = _divergence(engine.solver)
    # interior cells (boundary faces are free at open boundaries)
    inner = d[2:-2, 2:-2, 2:-2]
    v = engine.solver.read_velocity_centres()
    typical = np.abs(v).max() / engine.solver.h
    assert np.sqrt((inner ** 2).mean()) < 0.05 * typical


def test_simulation_is_deterministic(engine):
    sc = presets.make('torch')
    sc.data['domain']['resolution'] = 48
    outs = []
    for _ in range(2):
        engine.invalidate()
        engine.prepare(sc, final=False)
        engine.simulate_to(sc, sc.start + 15, cache=False)
        outs.append(engine.solver.read_scalars().astype(np.float32))
    assert np.array_equal(outs[0], outs[1])


def test_render_and_cache(engine):
    sc = presets.make('campfire')
    sc.data['domain']['resolution'] = 64
    engine.invalidate()
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 12)
    assert (sc.start + 12) in engine.cache
    engine.render(sc, sc.start + 12, (320, 180), mode='fire', samples=2)
    aov = engine.aovs()
    b = aov['beauty'].astype(np.float32)
    assert np.isfinite(b).all()
    assert 0.0 <= b[..., 3].min() and b[..., 3].max() <= 1.0
    assert b[..., :3].max() > 0.05, 'the fire should be visible'
    img = engine.display_image()
    assert img.shape == (180, 320, 4)
    # re-render an earlier cached frame without re-simulating
    live = engine.sim_frame
    engine.render(sc, sc.start + 5, (320, 180), mode='composite')
    assert engine.sim_frame == live


def test_soft_changes_keep_the_simulation(engine):
    sc = presets.make('campfire')
    sc.data['domain']['resolution'] = 48
    engine.invalidate()
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 5, cache=False)
    sc.set(('motion', 'turbulence'), 6.0)
    assert engine.prepare(sc, final=False, soft=True) == 'soft'
    assert engine.sim_frame == sc.start + 5
    sc.set(('domain', 'size_x'), 3.0)            # a new grid always restarts
    assert engine.prepare(sc, final=False, soft=True) == 'reset'
