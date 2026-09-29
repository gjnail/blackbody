"""GPU tests for liquids: stability, volume, incompressibility, determinism, sources, whitewater,
rendering, the frame cache and VDB output."""
import math

import numpy as np
import pytest

from blackbody.engine.liquid import LiquidParams, LiquidSolver, source
from blackbody.scene import presets
from blackbody.scene.model import Scene
from blackbody.scene.params import applies


@pytest.fixture(scope='module')
def liquid(engine):
    return LiquidSolver(engine.gpu)


def _setup(s, size, res, ppc=8, ww=0):
    dims, h = LiquidSolver.dims_for(size, res)
    origin = (-dims[0] * h / 2, 0.0, -dims[2] * h / 2)
    s.configure(dims, h, origin, LiquidSolver.capacity_for(dims, ppc, 4_000_000), ww)
    return dims, h, origin


def _run(s, prm, sources_first, frames, fps=30, sources_every=None):
    s._prm = prm
    for f in range(frames):
        n = s.substeps_for(1 / fps, cfl=1.5, lo=1, hi=12)
        with s.gpu.batch() as b:
            for _ in range(n):
                srcs = sources_first if s.steps == 0 else (sources_every or [])
                s.step(b, 1 / fps / n, prm, srcs)
            s.pack(b)
        s.measure()


def _dam(s, res=40, **kw):
    dims, h, origin = _setup(s, (1.0, 0.6, 0.5), res)
    prm = LiquidParams(open_sides=False, open_top=False, whitewater=False, **kw)
    s.reset()
    fill = source('box', pos=(origin[0] + 0.15, 0.25, 0.0), size=(0.15, 0.25, 0.25), fill=True)
    return prm, fill


def test_dam_break_is_stable_and_keeps_its_particles(liquid):
    prm, fill = _dam(liquid)
    _run(liquid, prm, [fill], 1)
    n0 = liquid.count
    assert n0 > 1000
    for _ in range(4):
        _run(liquid, prm, [], 15)
        assert liquid.count == n0, 'a closed box must keep every particle'
        assert np.isfinite(liquid.max_speed) and liquid.max_speed < 8.0, f'{liquid.max_speed:.1f} m/s'
    pos, vel = liquid.read_particles()
    assert np.isfinite(pos).all() and np.isfinite(vel).all()
    lo, hi = liquid.world_bounds()
    assert (pos >= lo - 1e-3).all() and (pos <= hi + 1e-3).all()


def test_projection_makes_the_liquid_divergence_free(liquid):
    prm, fill = _dam(liquid, volume_correction=0.0, pressure_iters=30)
    _run(liquid, prm, [fill], 8)
    v = liquid.gpu.read(liquid.VB).astype(np.float64)      # right after the projection
    t = liquid.gpu.read(liquid.TYPE[0])[..., 0]
    d = ((v[:-1, :-1, 1:, 0] - v[:-1, :-1, :-1, 0]) + (v[:-1, 1:, :-1, 1] - v[:-1, :-1, :-1, 1])
         + (v[1:, :-1, :-1, 2] - v[:-1, :-1, :-1, 2]))
    liq = (t > 0.5) & (t < 1.5)
    assert liq.sum() > 500
    typical = np.abs(v[..., :3]).max()
    assert np.sqrt((d[liq] ** 2).mean()) < 1e-3 * typical


def test_still_water_stays_still_and_keeps_its_volume(liquid):
    dims, h, origin = _setup(liquid, (0.8, 0.4, 0.8), 64)
    prm = LiquidParams(open_sides=False, whitewater=False)
    liquid.reset()
    fill = source('box', pos=(0, 0.06, 0), size=(0.4, 0.06, 0.4), fill=True)
    _run(liquid, prm, [fill], 30)
    pos, vel = liquid.read_particles()
    speed = np.linalg.norm(vel, axis=1)
    assert np.percentile(speed, 99) < 0.03, f'p99 {np.percentile(speed, 99):.3f} m/s'
    top = np.percentile(pos[:, 1], 99.5)
    assert abs(top - 0.12) < 0.012, f'surface at {top:.4f} m, filled to 0.12 m'


def test_liquid_is_deterministic(liquid):
    outs = []
    for _ in range(2):
        prm, fill = _dam(liquid, res=32)
        _run(liquid, prm, [fill], 12)
        pk = liquid.read_packed()
        outs.append(np.sort(pk.view(np.dtype((np.void, 16))).ravel()))
    assert len(outs[0]) == len(outs[1])
    assert np.array_equal(outs[0], outs[1])


def test_stream_source_pours_at_speed_times_area(liquid):
    dims, h, origin = _setup(liquid, (1.0, 1.0, 1.0), 64)
    prm = LiquidParams(whitewater=False, surface_tension=0.0)
    liquid.reset()
    r, v = 0.05, 1.0
    tap = source('cylinder', pos=(0.0, 0.7, 0.0), size=(r, 0.02, r), vel=(0.0, -v, 0.0))
    _run(liquid, prm, [tap], 1, sources_every=[tap])
    n1 = liquid.count
    _run(liquid, prm, [tap], 15, sources_every=[tap])
    rate = (liquid.count - n1) / 0.5 / prm.ppc * h ** 3           # m^3/s poured
    expect = math.pi * r * r * v
    assert 0.6 * expect < rate < 1.5 * expect, f'{rate * 1000:.2f} l/s, expected about {expect * 1000:.2f}'


def test_plunging_into_water_makes_whitewater(liquid):
    dims, h, origin = _setup(liquid, (1.0, 0.8, 1.0), 64, ww=200_000)
    prm = LiquidParams(whitewater=True, open_sides=False)
    liquid.reset()
    pool = source('box', pos=(0.0, 0.08, 0.0), size=(0.5, 0.08, 0.5), fill=True)
    ball = source('sphere', pos=(0.0, 0.45, 0.0), size=(0.1, 0.1, 0.1), fill=True, vel=(0.0, -3.5, 0.0))
    _run(liquid, prm, [pool, ball], 2)
    assert liquid.ww_count < 20, 'a blob flying through the air sheds (almost) nothing'
    _run(liquid, prm, [], 8)
    assert liquid.ww_count > 100


def _liquid_scene(name='rock_splash', res=64):
    sc = presets.make(name)
    sc.data['domain']['resolution'] = res
    sc.data['domain']['preroll'] = 0.2
    sc.data['render']['width'], sc.data['render']['height'] = 320, 180
    return sc


def test_liquid_render_and_cache(engine):
    sc = _liquid_scene()
    assert sc.kind == 'liquid'
    engine.invalidate()
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 20)
    assert engine.kind == 'liquid'
    assert (sc.start + 20) in engine.cache
    plate = np.full((180, 320, 4), 128, np.uint8)
    engine.render(sc, sc.start + 20, (320, 180), mode='composite', samples=2, plate=plate)
    b = engine.aovs()['beauty'].astype(np.float32)
    assert np.isfinite(b).all()
    assert b[..., 3].max() > 0.99, 'the pond should cover part of the frame'
    assert 0.0 <= b[..., 3].min()
    img = engine.display_image()
    assert img.shape == (180, 320, 4)
    live = engine.sim_frame
    engine.render(sc, sc.start + 5, (320, 180), mode='composite')   # from the cache
    assert engine.sim_frame == live
    st = engine.stats()
    assert st['kind'] == 'liquid' and st['particles'] > 0


def test_switching_between_fire_and_liquid(engine):
    fire = presets.make('campfire')
    fire.data['domain']['resolution'] = 48
    engine.invalidate()
    engine.prepare(fire, final=False)
    engine.simulate_to(fire, fire.start + 3, cache=False)
    assert engine.kind == 'fire'
    liq = _liquid_scene('water_pour', 48)
    assert engine.prepare(liq, final=False) == 'reset'
    engine.simulate_to(liq, liq.start + 3, cache=False)
    assert engine.kind == 'liquid' and engine.liquid.count > 0
    assert engine.prepare(fire, final=False) == 'reset'
    engine.simulate_to(fire, fire.start + 2, cache=False)
    assert engine.kind == 'fire'


def test_liquid_vdb_round_trip(engine, tmp_path):
    from blackbody.io.vdb import read_vdb, write_liquid_vdb_frame
    sc = _liquid_scene(res=48)
    engine.invalidate()
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 2, cache=False)
    p = tmp_path / 'liquid.vdb'
    write_liquid_vdb_frame(p, engine, sc, sc.start + 2)
    grids = read_vdb(p)['grids']
    assert {'density', 'vel'} <= set(grids)
    vals = np.concatenate([v.ravel() for v in grids['density']['leaves'].values()])
    assert vals.max() > 0.99 and vals.min() >= 0.0


def test_settings_follow_the_simulation_kind():
    assert not applies('combustion', None, 'liquid')
    assert not applies('water', None, 'fire')
    assert applies('domain', 'resolution', 'liquid')
    assert not applies('emitter', 'fuel', 'liquid')
    assert applies('emitter', 'liquid_mode', 'liquid') and not applies('emitter', 'liquid_mode', 'fire')


def test_liquid_scene_round_trip(tmp_path):
    sc = presets.make('hose')
    p = tmp_path / 'hose.bbfire'
    sc.save(p)
    back = Scene.load(p)
    assert back.kind == 'liquid'
    assert back.data['water']['ior'] == pytest.approx(1.333)
    assert back.emitters[0]['liquid_mode'] == 'stream'
    assert back.sim_signature() == sc.sim_signature()
