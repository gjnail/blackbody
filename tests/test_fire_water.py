"""GPU tests for fire, water and lava in one box (engine/both_engine.py): water soaking a fire's fuel
bed and boiling off its embers as steam, the liquid dragging the air, lava as a second liquid that the
water flows round, boils where they meet and chills, lava heating the air, and the scene settings
behind them."""
import numpy as np
import pytest

from blackbody.scene import presets
from blackbody.scene.params import applies


def _scene(name, res, **sets):
    sc = presets.make(name)
    sc.data['domain']['resolution'] = res
    for k, v in sets.items():
        sec, key = k.split('__')
        sc.set((sec, key), v)
    return sc


def _run(engine, sc, frames):
    engine.invalidate()
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + frames - 1, cache=False)


def _flame(engine):
    return float(engine.solver.read_scalars().astype(np.float32)[..., 3].sum())


def test_lava_settings_come_from_the_lava_preset():
    sc = presets.make('lava_sea')
    assert sc.kind == 'both' and sc.has_lava()
    lava = presets.PRESETS['lava']
    prm = sc.lava_params(sc.start)
    assert prm.viscosity == pytest.approx(lava['liquid']['viscosity'])
    assert prm.cooling == pytest.approx(lava['liquid']['cooling'])
    assert not prm.whitewater and prm.water_level == 0.0
    look = sc.lava_look(sc.start)
    assert look.glow == pytest.approx(lava['water']['glow'])
    assert look.colliders_look == sc.data['water']['colliders_look'], 'the scene draws its own colliders'
    # a setting changed in the scene's Lava section wins
    sc.set(('lava', 'viscosity'), 2000.0)
    assert sc.lava_params(sc.start).viscosity == pytest.approx(2000.0)
    assert sc.liquid_params(sc.start).viscosity != pytest.approx(2000.0), 'the water keeps its own'


def test_lava_emitters_are_neither_fire_nor_water():
    sc = presets.make('lava_quench')
    t = sc.start + 40
    names = [e['name'] for e in sc.emitters]
    assert len(sc.sources_gpu(t, set(), emits='lava')) >= 1
    water = sc.sources_gpu(t, set())
    assert len(water) == 1, 'only the hose pours water'
    assert sc.emitters_gpu(t) == [], 'no lava or water source feeds the fire'
    assert 'Hose' in names
    assert applies('lava', None, 'both') and not applies('lava', None, 'fire') and not applies('lava', None, 'liquid')
    assert not applies('combustion', 'soak', 'fire') and applies('combustion', 'soak', 'both')


def test_lava_look_settings_do_not_resimulate():
    sc = presets.make('lava_sea')
    sig = sc.sim_signature()
    sc.set(('lava', 'glow'), 3.0)
    assert sc.sim_signature() == sig
    sc.set(('lava', 'quench'), 9.0)
    assert sc.sim_signature() != sig


def test_hose_knocks_the_fire_down_and_makes_steam(engine):
    frames = 84
    dry = _scene('hose_on_fire', 72)
    dry.emitters = [e for e in dry.emitters if e['name'] != 'Hose']
    _run(engine, dry, frames)
    flame_dry = _flame(engine)
    wet = _scene('hose_on_fire', 72)
    _run(engine, wet, frames)
    flame_wet = _flame(engine)
    assert flame_wet < 0.5 * flame_dry, 'the water puts most of the fire out'
    bed = engine.gpu.read(engine._bed[0]).astype(np.float32)
    assert bed[..., 0].max() > 0.8, 'the fuel bed soaks'
    ax = engine.solver.read_aux().astype(np.float32)
    assert ax[..., 1].max() > 50.0, 'the embers boil the water off as steam'
    assert ax[..., 2].sum() > 0.0, 'and it condenses into a visible cloud'


def test_wet_fuel_dries_and_catches_again(engine):
    # a short burst of water: the bed soaks, then dries out in the heat and burns again
    sc = _scene('hose_on_fire', 64)
    hose = next(e for e in sc.emitters if e['name'] == 'Hose')
    hose['stop'] = 1.6
    sc.set(('combustion', 'rekindle'), 1.5)
    sc.set(('combustion', 'soak'), 8.0)
    _run(engine, sc, 52)
    soak_mid = float(engine.gpu.read(engine._bed[0]).astype(np.float32)[..., 0].max())
    engine.simulate_to(sc, sc.start + 150, cache=False)
    soak_late = engine.gpu.read(engine._bed[0]).astype(np.float32)[..., 0]
    assert soak_mid > 0.5
    assert float(np.percentile(soak_late[soak_late > 0.0], 50)) < soak_mid if (soak_late > 0).any() else True
    assert _flame(engine) > 0.0, 'the fire has caught again'


def test_liquid_drags_the_air(engine):
    sc = _scene('hose_on_fire', 64)
    _run(engine, sc, 56)
    vel = engine.gpu.read(engine.solver.vel[0]).astype(np.float32)
    dens = engine.gpu.read(engine.liquid.DENS)[..., 0]
    wet = dens > 0.6 * engine.liquid._prm.ppc
    lvel = engine.gpu.read(engine.liquid.vel_tex).astype(np.float32)
    n = dens.shape
    # face velocities in cells full of water follow the water
    gas = vel[:n[0], :n[1], :n[2], 0][wet]
    liq = lvel[:n[0], :n[1], :n[2], 0][wet]
    assert wet.any()
    assert np.abs(gas - liq).mean() < 0.35 * (np.abs(liq).mean() + 0.1)


def test_lava_meets_the_sea(engine):
    sc = _scene('lava_sea', 64)
    _run(engine, sc, 48)
    V = engine.lava
    assert engine._lava_on and V is not None and V.count > 0
    field = engine.gpu.read(engine._lava_field).astype(np.float32)
    assert field[..., 2].sum() > 0.0, 'water touching the lava boils'
    ax = engine.solver.read_aux().astype(np.float32)
    assert ax[..., 1].max() > 20.0, 'the steam goes into the air'
    # the lava is solid ground to the water
    sdf = engine.gpu.read(engine.liquid.SDF)[..., 0]
    lava = engine.gpu.read(V.DENS)[..., 0] > 0.9 * V._prm.ppc
    assert (sdf[lava] < 0.0).mean() > 0.9
    # the air over the lava is hot
    T = engine.solver.read_scalars().astype(np.float32)[..., 0]
    assert T[field[..., 1] > 0.5].max() > 0.15
    # lava in the water is cooler than lava in the air
    heat = engine.gpu.read(V.HEAT)[..., 0]
    wet = field[..., 3] > 0.3
    dry = (field[..., 3] <= 0.0) & (field[..., 1] > 0.5)
    both = lava & wet
    if both.any() and (lava & dry).any():
        assert heat[both].mean() < heat[lava & dry].mean()


def test_lava_sets_the_grass_alight(engine):
    sc = _scene('lava_grass', 64)
    _run(engine, sc, 36)
    burn = engine.gpu.read(engine.solver.burn[0]).astype(np.float32)
    assert (burn[..., 0] > 0.05).any(), 'the grass by the lava has caught'


def test_both_renders_with_lava_and_caches(engine):
    sc = _scene('lava_quench', 56)
    engine.invalidate()
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 30, cache=True)
    plate = np.full((180, 320, 4), 50, np.uint8)
    for f in (sc.start + 30, sc.start + 29):   # live, then from the cache
        engine.render(sc, f, (320, 180), plate=plate)
        aov = engine.aovs()
        b = aov['beauty'].astype(np.float32)
        e = aov['emission'].astype(np.float32)
        assert np.isfinite(b).all() and np.isfinite(e).all()
        assert e[..., 0].max() > 0.02 and e[..., 0].max() >= e[..., 2].max(), 'the lava glows red-orange'
