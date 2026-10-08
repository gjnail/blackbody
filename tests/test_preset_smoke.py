"""Every preset simulates and renders: a few frames of each on a coarse grid, drawn small, with everything it simulates
finite, something alive in it, something drawn that was not there at its first frame, and nothing logged as wrong. Slow
(every preset, several minutes), so it runs only when asked for: pytest -m slow tests/test_preset_smoke.py (-k NAME for
one)."""
import logging

import numpy as np
import pytest

from blackbody.scene import presets

pytestmark = pytest.mark.slow

RES = 96          # cells along the box's longest side (before the preview scale), or the preset's own if fewer (much
                  # coarser and a lava vent is thinner than a cell: at 32 a few dozen drops come out of it)
FRAMES = 5        # frames simulated before the first look (a preset shorter than this is a still)
STEP = 3          # frames between looks after that, until something is alive and drawn, up to the preset's end (a gas
                  # cloud lights at 2 s, a flash fire at 3 s, fireworks go up at half a second)
SIZE = (160, 90)
SEED = 1          # (every look is drawn with the same noise, so two looks differ only where the scene does)
ADVICE = ('pieces take a while to simulate',)   # notices that are advice to the user, not faults


def _finite(what, a):
    a = np.asarray(a, np.float32)
    bad = int((~np.isfinite(a)).sum())
    assert bad == 0, f'{what}: {bad} of {a.size} values are not finite'
    return a


def _alive(engine):
    """What the scene simulates now (each part's state checked finite on the way)."""
    out = []
    if engine.kind in ('fire', 'both'):
        s = engine.solver
        if _finite('gas', s.read_scalars()).max() > 0.0:
            out.append('gas')
        _finite('gas velocity', s.read_velocity_centres())
        if engine.embers.count and engine.embers.alive():   # (count is the pool's size, not how many fly)
            out.append('embers')
    if engine.kind in ('liquid', 'both'):
        for what, L in (('liquid', engine.liquid), ('lava', engine.lava if engine.kind == 'both' and engine._lava_on else None)):
            if L is None:
                continue
            x, v = L.read_particles()
            _finite(what, x)
            _finite(f'{what} velocity', v)
            if len(x):
                out.append(what)
    if engine.kind == 'cloud':   # (a sky has nothing else in it; what the other parts hold is from an earlier scene)
        a, b = engine.cloud.read_fields()
        if np.abs(_finite('cloud', a)).max() > 0.0:
            out.append('air')
        _finite('cloud', b)
        return out
    if engine.solids.active:
        _finite('objects', engine.solids.data.qpos)
        out.append('objects')
    if engine.cloth.active:
        _finite('fabric', engine.cloth.positions()[0])
        out.append('fabric')
    if engine._matter is not None and engine._matter.active:
        _finite('matter', engine._matter.read_particles())
        out.append('matter')
    if engine._strands is not None and engine._strands.active:
        _finite('grass', engine._strands.positions()[0])
        out.append('grass')
    return out


def _look(engine, sc, frame):
    """The frame drawn small (its beauty pass checked finite on the way): the picture shown, as integers."""
    engine.render(sc, frame, SIZE, mode='composite', seed=SEED)
    beauty = _finite('beauty', engine.aovs()['beauty'])
    assert beauty.shape[:2] == (SIZE[1], SIZE[0])
    return engine.display_image()[..., :3].astype(np.int16)


def _notices(engine, caplog):
    """What was logged as a warning or worse while it ran, and the notices of the parts that keep their own."""
    out = [f'{r.name}: {r.getMessage()}' for r in caplog.records if r.name.startswith('blackbody')]
    if engine.kind != 'cloud':
        for what, part in (('matter', engine._matter), ('grass', engine._strands)):
            if part is not None and part.active:
                out += [f'{what}: {w}' for w in part.warnings]
    return out


def _scene(name):
    """The preset on the coarse grid, drawn small (its preroll kept: a sky's clouds form in it)."""
    sc = presets.make(name)
    d = sc.data['domain']
    d['resolution'] = min(d['resolution'], RES)
    sc.data['render']['width'], sc.data['render']['height'] = SIZE
    return sc


@pytest.mark.parametrize('name', presets.ORDER)
def test_every_preset_simulates_and_renders(engine, name, caplog):
    caplog.set_level(logging.WARNING)
    sc = _scene(name)
    still = sc.end - sc.start + 1 < FRAMES
    engine.invalidate()
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start, cache=False)
    first = _look(engine, sc, sc.start)
    frame, alive, drawn = sc.start, _alive(engine), False
    while frame < sc.end and not (alive and drawn):
        frame = min(max(frame + STEP, sc.start + FRAMES - 1), sc.end)
        engine.simulate_to(sc, frame, cache=False)
        alive = _alive(engine)
        drawn = bool((np.abs(_look(engine, sc, frame) - first) > 2).any())
    st = engine.stats()
    assert np.isfinite(st['max_speed']), st['max_speed']
    assert not st.get('mesh_errors'), st['mesh_errors']
    if still:   # (nothing moves in a still: what it shows is there from its first frame)
        assert first.max() > 0, 'nothing is drawn'
    else:
        assert alive, f'nothing simulates in it by frame {frame}'
        assert drawn, f'nothing it simulates shows by frame {frame}: every look is drawn as its first frame was'
    wrong = [n for n in _notices(engine, caplog) if not any(a in n for a in ADVICE)]
    assert not wrong, wrong
