"""Heat, wind and push across things (on the GPU): objects that warm and cool (objheat.py) and the shared radiant
sources (radiant.py), the wind on sand and snow and the grass slowing it, lava meeting objects and matter, and cloth
pushing the water. (The parts that need no GPU are in test_objheat_model.py.)"""
import math

import numpy as np
import pytest

from blackbody.scene import components, presets


def _temps(engine):
    return [t.skin - 273.15 for t in engine.objheat.things]


def test_a_steel_bar_over_a_fire_heats_and_one_kept_cold_does_not(engine):
    sc = presets.make('campfire')
    sc.add_collider(name='Bar', shape='box', position=(0.0, 0.75, -0.1), size=(0.3, 0.012, 0.012), material='steel')
    sc.add_collider(name='Held', shape='box', position=(0.0, 0.75, 0.1), size=(0.3, 0.012, 0.012), material='steel',
                    keeps_temperature=True)
    sc.data['domain'].update(preroll=0.0, resolution=48, matter_heat_speed=30.0)
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 72, cache=False)
    bar, held = _temps(engine)
    assert bar > 60.0                          # (some 0.7 K/s real in the plume: minutes to glow, seconds at 30x)
    assert held == pytest.approx(20.0, abs=1e-3)
    assert engine.radiant.built and engine.objheat.last['radiation_w'][0] > 10.0   # (the flames' radiation on it)


def test_red_hot_steel_dropped_in_water_boils_it_and_cools(engine):
    sc = presets.make('quench')
    sc.data['domain'].update(resolution=80, preroll=0.25)
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 14, cache=False)        # (falling, let go at 0.6 s)
    hot = _temps(engine)[1]
    boiled = 0
    for f in range(sc.start + 15, sc.start + 40):
        engine.simulate_to(sc, f, cache=False)
        boiled += engine.liquid.thermal_stats.get('boiling_cells', 0)
    T = _temps(engine)[1]
    assert hot > 900.0 and T < hot - 150.0      # (the water takes its heat through the boiling curve)
    assert boiled > 20


def test_a_glowing_object_lights_the_shared_sources_and_scorches_cloth_beside_it(engine):
    sc = components.new_scene('fire', 'person')
    sc.emitters = []
    sc.add_collider(name='Ingot', shape='box', position=(0.0, 0.1, 0.0), size=(0.1, 0.1, 0.1), material='steel',
                    temperature=1100.0, keeps_temperature=True)
    sc.add_fabric(name='Cloth', width=0.4, height=0.4, position=(0.0, 0.25, 0.16), pins='top', burnable=False)
    sc.data['domain'].update(size_x=1.0, size_y=0.8, size_z=0.8, resolution=40, preroll=0.0)
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 24, cache=False)
    assert engine.radiant.n_objects >= 4            # (its faces, each a source)
    c = engine.cloth
    n = c.built.n
    S = np.frombuffer(engine.gpu.read_buffer(c.bufs['S'], n * 16), np.float32).reshape(n, 4)
    assert S[:, 0].max() > 330.0                    # (some 30 kW/m^2 at a few cm from 1100 C steel)


def test_wind_carries_dry_sand_downwind_at_the_saltation_rate_and_loses_none(engine):
    sc = components.new_scene('fire', 'person')
    sc.emitters = []
    sc.add_matter(name='Sand', material='sand', shape='box', position=(-0.25, 0.04, 0.0), size=(0.1, 0.04, 0.12))
    sc.data['domain'].update(size_x=1.6, size_y=0.6, size_z=0.6, resolution=48, preroll=0.0, matter_detail=96,
                             open_sides=False)
    sc.data['motion'].update(wind_speed=20.0, wind_dir=90.0, gust=0.0)
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 12, cache=False)
    m = engine.matter

    def live():
        P = m.read_particles()
        return m.origin + P[P[:, 3] >= 0, :3].astype(float) * m.dx

    x0 = live()
    edge = np.percentile(x0[:, 0], 99.5)
    engine.simulate_to(sc, sc.start + 12 + 96, cache=False)
    x = live()
    assert len(x) == len(x0)
    moved = (x[:, 0] > edge + 0.01).sum() * 1600.0 * (m.dx / 2) ** 3
    u = 0.4 * 20.0 / math.log(max(1.5 * engine.solver.h, 0.02) / (2.5e-4 / 30.0))
    r = 0.23 / u
    q = 2.78 * 1.2 / 9.81 * u ** 3 * (1 - r) * (1 + r) ** 2 * 0.24 * 4.0    # kg over the heap's width in 4 s
    assert q / 3.0 < moved < 3.0 * q


def test_a_meadow_slows_the_wind_through_it_but_not_over_it(engine):
    def run(grass):
        sc = components.new_scene('fire', 'person')
        sc.emitters = []
        if grass:
            sc.add_strands(name='Meadow', kind='meadow', shape='box', position=(0.0, 0.0, 0.0), size=(0.9, 0.45, 0.6),
                           burns=False)
        sc.data['domain'].update(size_x=2.4, size_y=1.6, size_z=1.6, resolution=48, preroll=0.0)
        sc.data['motion'].update(wind_speed=5.0, wind_dir=90.0, gust=0.0)
        engine.prepare(sc, final=False)
        engine.simulate_to(sc, sc.start + 36, cache=False)
        s = engine.solver
        v = engine.gpu.read(s.vel[0])[..., 0]
        o, h = np.asarray(s.origin), s.h
        ix = slice(int((0.2 - o[0]) / h), int((0.8 - o[0]) / h))
        iz = slice(int((-0.3 - o[2]) / h), int((0.3 - o[2]) / h))
        return float(v[iz, int((0.15 - o[1]) / h), ix].mean()), float(v[iz, int((0.9 - o[1]) / h), ix].mean())
    bare_in, bare_over = run(False)
    in_, over = run(True)
    assert in_ < 0.85 * bare_in
    assert abs(over - bare_over) < 0.1 * bare_over


def test_lava_floats_wood_carries_it_off_heats_it_and_radiates(engine):
    sc = presets.make('lava_grass')
    sc.data['spread']['enabled'] = False
    sc.data['domain'].update(size_x=3.0, size_z=1.6, resolution=80, preroll=0.0, matter_heat_speed=8.0)
    sc.emitters[0]['position'] = (-1.3, 0.12, 0.0)
    sc.add_collider(name='Log', shape='cylinder', position=(-0.6, 0.08, 0.0), size=(0.06, 0.25, 0.06), roll=90.0,
                    material='wood', dynamic=True)
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 72, cache=False)
    log = sc.colliders_gpu(sc.start + 72, engine.solids.overrides())[0]
    assert log.pos[0] > -0.3                         # carried downstream by the flow,
    assert _temps(engine)[0] > 60.0                   # warmed where it touches the lava and by its glow,
    cnt = int(np.frombuffer(engine.gpu.read_buffer(engine.radiant.RLC, 4), np.uint32)[0])
    L = np.frombuffer(engine.gpu.read_buffer(engine.radiant.RL, cnt * 48), np.float32).reshape(-1, 12)
    assert L[:, 5].sum() > 1.0e4                      # and the lava's surface radiates (W)


def test_cloth_swept_through_water_pushes_it(engine):
    def run(couple):
        sc = presets.make('towel_dip')
        sc.data['domain'].update(preroll=0.5, resolution=80)
        fab = sc.fabrics[0]
        fab.update(width=0.5, height=0.4, wetness=1.0, pins='edges')
        for f, z in ((sc.start, -0.4), (sc.start + 12, -0.4), (sc.start + 36, 0.4)):
            sc.set_key(('fabric', 0, 'position'), f, (0.0, 0.26, z))
        engine.invalidate()                      # (the same scene twice: simulate it afresh)
        engine.prepare(sc, final=False)
        if not couple:
            engine.cloth.liquid_hook = lambda *a, **k: None
        engine.simulate_to(sc, sc.start + 32, cache=False)
        engine.cloth.__dict__.pop('liquid_hook', None)
        p, v = engine.liquid.read_particles()
        sel = (np.abs(p[:, 0]) < 0.4) & (p[:, 1] < 0.48) & (np.abs(p[:, 2]) < 0.6)
        return float((np.linalg.norm(v[sel], axis=1) ** 2).sum())
    assert run(True) > 5.0 * run(False)
