"""Swirl, moving emitters and colliders, dousing, sparks, colourants, spreading fire, tracked air,
meshes and steam: scene handling on the CPU, and behaviour on the GPU."""
import math

import numpy as np
import pytest

from blackbody.engine.mesh import grid_for, load_obj
from blackbody.scene import Scene, presets
from blackbody.scene.anim import Curve
from blackbody.scene.params import coerce, param
from blackbody.scene.presets import _em

NEW_PRESETS = ['fire_whirl', 'waved_torch', 'hose_douse', 'grass_fire', 'curtain_fire', 'armchair_fire', 'room_fire',
               'coloured_flames', 'road_flare', 'grinder_sparks', 'fireworks', 'car_through_smoke', 'kettle_steam', 'steam_vent']

CUBE_OBJ = """v -0.5 -0.5 -0.5
v 0.5 -0.5 -0.5
v 0.5 0.5 -0.5
v -0.5 0.5 -0.5
v -0.5 -0.5 0.5
v 0.5 -0.5 0.5
v 0.5 0.5 0.5
v -0.5 0.5 0.5
f 1 4 3 2
f 5 6 7 8
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
"""


def _cube(tmp_path):
    p = tmp_path / 'cube.obj'
    p.write_text(CUBE_OBJ)
    return str(p)


def _quiet(sc, res=48):
    """A scene with no emitters, turbulence or wind: a clean slate for one feature."""
    sc.data['domain'].update(resolution=res, preroll=0.0)
    sc.data['motion'].update(turbulence=0.0, disturbance=0.0, vorticity=0.0, wind_speed=0.0)
    sc.data['embers']['enabled'] = False
    sc.emitters = []
    sc.colliders = []
    return sc


def _cell_coords(solver):
    o, _ = solver.world_bounds()
    nx, ny, nz = solver.dims
    h = solver.h
    return (o[0] + (np.arange(nx) + 0.5) * h, (np.arange(ny) + 0.5) * h, o[2] + (np.arange(nz) + 0.5) * h)


# -- scene and presets (no GPU) ---------------------------------------------------------------------

def test_new_presets_are_listed_and_build():
    for name in NEW_PRESETS:
        assert name in presets.ORDER
        s = presets.make(name)
        assert s.emitters


def test_preset_keyframes_follow_the_shot_timing():
    a = presets.make('waved_torch')
    b = presets.make('waved_torch', fps=30, start=1001)
    ka, kb = a.curve(('emitter', 0, 'position')).keys, b.curve(('emitter', 0, 'position')).keys
    assert len(ka) == len(kb) > 2
    # the same moments in seconds, whatever the frame rate and first frame
    for x, y in zip(ka, kb):
        assert math.isclose((x[0] - a.start) / a.fps, (y[0] - b.start) / b.fps, abs_tol=1e-6)
    shot = presets.make('campfire', fps=30, start=1001)
    presets.apply_to(shot, 'car_through_smoke')
    first = shot.curve(('collider', 0, 'position')).keys[0][0]
    assert first == 1001


def test_features_follow_the_settings():
    s = presets.make('campfire')
    assert not any(s.features().values())
    s.data['combustion']['air'] = 'tracked'
    assert s.features()['oxygen'] and s.features()['aux']
    s = presets.make('campfire')
    s.emitters[0]['color_amount'] = Curve([[1, 0.0], [10, 2.0]])
    assert s.features()['chem']
    s.emitters[0]['vapour'] = 100.0
    assert s.features()['vapour']
    s.data['spread']['enabled'] = True
    assert s.features()['burn']


def test_new_settings_round_trip(tmp_path):
    s = presets.make('room_fire')
    s.set(('collider', 0, 'burnable'), True)
    s.set(('emitter', 0, 'swirl'), 2.5)
    s.set(('embers', 'direction'), (1.0, 0.0, 0.0))
    s.set(('spread', 'enabled'), True)
    p = tmp_path / 'room.bbfire'
    s.save(p)
    t = Scene.load(p)
    assert t.colliders[0]['burnable'] is True
    assert t.emitters[0]['swirl'] == 2.5
    assert tuple(t.data['embers']['direction']) == (1.0, 0.0, 0.0)
    assert isinstance(t.colliders[-1]['position'], Curve)
    assert t.sim_signature() == s.sim_signature()


def test_rates_of_animated_settings():
    s = presets.make('car_through_smoke')
    f = s.start + int(2.0 * s.fps)
    v = s.rate(('collider', 0, 'position'), f)
    assert math.isclose(v[0], 21.0 / 2.6, rel_tol=0.02)     # 21 m in 2.6 s
    cols = s.colliders_gpu(f)
    assert cols[0].moving and s.colliders_animated()
    assert s.rate(('emitter', 0, 'fuel'), f) == 0.0


def test_moving_emitter_passes_on_its_motion():
    s = presets.make('waved_torch')
    f = s.start + int(0.35 * s.fps)   # mid-swing
    e = s.emitters_gpu(f)[0]
    motion = np.asarray(s.rate(('emitter', 0, 'position'), f))
    assert np.linalg.norm(motion) > 1.0
    assert np.allclose(e.vel, motion, atol=1e-6)
    assert e.vel_blend > 0.5
    s.emitters[0]['inherit'] = 0.0
    assert np.allclose(s.emitters_gpu(f)[0].vel, 0.0)


def test_mesh_paths_and_obj_loading(tmp_path):
    s = Scene()
    assert s.mesh_path('builtin:armchair.obj').endswith('armchair.obj')
    v, t = load_obj(_cube(tmp_path))
    assert v.shape == (8, 3) and t.shape == (12, 3)       # quads split into two triangles each
    lo, hi, dims, cell = grid_for(v.min(0), v.max(0), 32)
    assert all(d >= 4 for d in dims) and (np.asarray(hi) - np.asarray(lo) > 1.0).all()


def test_cli_accepts_mesh_paths(tmp_path):
    from blackbody.cli import _apply_setting
    s = presets.make('campfire')
    cube = _cube(tmp_path)
    assert _apply_setting(s, f'emitter.0.mesh={cube}') is None
    assert s.emitters[0]['mesh'] == cube
    assert 'no file' in _apply_setting(s, 'emitter.0.mesh=nowhere.obj')
    assert _apply_setting(s, 'spread.enabled=true') is None and s.data['spread']['enabled']


# -- GPU ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('name', NEW_PRESETS)
def test_new_presets_stay_stable(engine, name):
    sc = presets.make(name)
    sc.data['domain']['resolution'] = 48
    sc.data['domain']['preroll'] = min(sc.data['domain']['preroll'], 1.0)
    engine.invalidate()
    engine.prepare(sc, final=False)
    for f in range(sc.start, sc.start + 48, 4):
        engine.simulate_to(sc, f, cache=False)
        assert np.isfinite(engine.solver.max_speed)
        assert engine.solver.max_speed < 100.0, f'{name} frame {f}: {engine.solver.max_speed:.1f} m/s'
    s = engine.solver.read_scalars().astype(np.float32)
    assert np.isfinite(s).all() and s.min() >= 0.0
    for extra in (engine.solver.read_aux(), engine.solver.read_chem(), engine.solver.read_burn()):
        if extra is not None:
            assert np.isfinite(extra.astype(np.float32)).all()


def test_swirl_spins_the_air(engine):
    sc = _quiet(presets.make('campfire'), 48)
    sc.emitters = [_em(sc, shape='cylinder', position=(0, 0.05, 0), size=(0.3, 0.05, 0.3), fuel=0.0, temperature=0.3, swirl=2.0,
                       swirl_width=3.0)]
    engine.invalidate()
    engine.prepare(sc)
    engine.simulate_to(sc, sc.start + 36, cache=False)
    v = engine.solver.read_velocity_centres()
    xs, ys, zs = _cell_coords(engine.solver)
    X, Z = np.meshgrid(xs, zs)                    # (z, x)
    r = np.hypot(X, Z)
    band = (r > 0.2) & (r < 0.6)
    j = int(0.5 / engine.solver.h)                # half a metre up
    # tangential velocity of an anticlockwise swirl seen from above: (z, -x) / r
    vt = (v[:, j, :, 0] * Z - v[:, j, :, 2] * X) / np.maximum(r, 1e-6)
    assert vt[band].mean() > 0.3
    assert engine.solver.max_speed < 10.0


def test_moving_collider_pushes_the_air(engine):
    sc = _quiet(presets.make('campfire'), 48)
    sc.data['domain'].update(size_x=4.0, size_y=2.0, size_z=2.0)
    from blackbody.scene.presets import K, _col
    sc.colliders = [_col(sc, 0, shape='box', size=(0.3, 0.3, 0.3),
                         position=K((0.0, (-1.2, 0.6, 0.0)), (1.0, (1.2, 0.6, 0.0)), interp='linear'))]
    engine.invalidate()
    engine.prepare(sc)
    engine.simulate_to(sc, sc.start + 12, cache=False)
    s = engine.solver
    col = sc.colliders_gpu(sc.start + 12)[0]
    v = s.read_velocity_centres()
    xs, ys, zs = _cell_coords(s)
    ahead = (np.abs(xs - (col.pos[0] + 0.45)) < 0.1)
    mid = (np.abs(ys - 0.6) < 0.2)
    u = v[:, mid][:, :, ahead, 0]
    assert u.max() > 0.5, 'air in front of the moving box is pushed along'
    sdf = s.gpu.read(s.sdf)[..., 0]
    i = int((col.pos[0] - s.origin[0]) / s.h)
    assert sdf[s.dims[2] // 2, int(0.6 / s.h), i] < 0, 'the distance field follows the collider'


def test_douse_puts_the_fire_out_and_makes_steam(engine):
    sc = presets.make('campfire')
    sc.data['domain'].update(resolution=48, preroll=1.5)
    sc.data['embers']['enabled'] = False
    engine.invalidate()
    engine.prepare(sc)
    engine.simulate_to(sc, sc.start + 6, cache=False)
    flame_before = engine.solver.read_scalars()[..., 3].astype(np.float32).sum()
    sc.emitters.append(_em(sc, shape='box', position=(0, 0.6, 0), size=(1.0, 0.6, 1.0), fuel=0.0, temperature=0.0, noise=0.0,
                           douse=40.0, start=0.3, fade_in=0.05, embers=False))
    engine.invalidate()
    engine.prepare(sc)
    engine.simulate_to(sc, sc.start + 30, cache=False)
    s = engine.solver
    flame_after = s.read_scalars()[..., 3].astype(np.float32).sum()
    assert flame_after < 0.1 * flame_before
    assert s.read_aux()[..., 1].astype(np.float32).max() > 10.0, 'water turned to steam'


def test_tracked_air_starves_a_closed_fire(engine):
    def run(closed):
        sc = presets.make('campfire')
        sc.data['domain'].update(resolution=40, preroll=0.0, open_sides=not closed, open_top=not closed,
                                 size_x=1.2, size_y=1.2, size_z=1.2)
        sc.data['combustion'].update(air='tracked', air_use=3.0, cooling=0.3, expansion=0.0)
        sc.data['embers']['enabled'] = False
        sc.emitters = [_em(sc, shape='cylinder', position=(0, 0.05, 0), size=(0.25, 0.05, 0.25), fuel=30, temperature=0.6)]
        engine.invalidate()
        engine.prepare(sc)
        engine.simulate_to(sc, sc.start + 120, cache=False)
        return engine.solver.burning, engine.solver.read_aux()[..., 0].astype(np.float32).mean()
    open_burning, open_used = run(False)
    closed_burning, closed_used = run(True)
    assert closed_used > 0.4 and open_used < 0.25, 'a sealed box runs out of air; an open one does not'
    assert closed_burning < 0.75 * open_burning


def test_sparks_bounce_off_colliders(engine):
    sc = _quiet(presets.make('grinder_sparks'), 32)
    sc.data['domain'].update(size_x=2.0, size_y=2.0, size_z=2.0)
    sc.data['embers'].update(enabled=True, direction=(0.0, -1.0, 0.0), cone=10.0, launch=6.0, gravity=9.81, lifetime=1.0, rate=800)
    sc.emitters = [_em(sc, shape='sphere', position=(0.0, 1.6, 0.0), size=(0.02, 0.02, 0.02), fuel=0.0, temperature=0.0, noise=0.0)]
    sc.colliders = [presets._col(sc, 0, shape='box', position=(0.0, 0.5, 0.0), size=(0.6, 0.1, 0.6))]
    engine.invalidate()
    engine.prepare(sc)
    engine.simulate_to(sc, sc.start + 18, cache=False)
    a = np.frombuffer(engine.gpu.read_buffer(engine.embers.A), np.float32).reshape(-1, 4)
    alive = a[a[:, 3] > 0]
    assert len(alive) > 50
    over = (np.abs(alive[:, 0]) < 0.5) & (np.abs(alive[:, 2]) < 0.5)
    inside = over & (alive[:, 1] < 0.55) & (alive[:, 1] > 0.45)
    below = over & (alive[:, 1] < 0.35)
    assert inside.sum() <= 0.02 * len(alive) and below.sum() <= 0.02 * len(alive), 'sparks land on the box, not through it'


def test_ground_fire_spreads(engine):
    sc = presets.make('grass_fire')
    sc.data['domain']['resolution'] = 64
    sc.data['embers']['enabled'] = False
    engine.invalidate()
    engine.prepare(sc)
    counts = []
    for t in (1.5, 4.0, 6.5):
        engine.simulate_to(sc, sc.start + int(t * sc.fps), cache=False)
        b = engine.solver.read_burn().astype(np.float32)
        counts.append(int(((b[..., 2] > 0.5) & (b[..., 1] >= 1.0)).sum()))
    assert counts[0] > 0 and counts[1] > counts[0] and counts[2] > counts[1], counts


def test_mesh_collider_and_emitter(engine, tmp_path):
    cube = _cube(tmp_path)
    sc = _quiet(presets.make('campfire'), 48)
    sc.data['domain'].update(size_x=2.0, size_y=2.0, size_z=2.0)
    sc.colliders = [presets._col(sc, 0, shape='mesh', mesh=cube, position=(0.0, 0.8, 0.0), size=(0.8, 0.8, 0.8), yaw=30.0)]
    sc.emitters = [_em(sc, shape='mesh', mesh=cube, position=(0.0, 0.8, 0.0), size=(0.8, 0.8, 0.8), yaw=30.0, thickness=0.06,
                       fuel=10.0, temperature=0.0, noise=0.0)]
    engine.invalidate()
    engine.prepare(sc)
    s = engine.solver
    assert not s.meshes.errors
    engine.simulate_to(sc, sc.start + 2, cache=False)
    sdf = s.gpu.read(s.sdf)[..., 0] * s.h        # metres
    xs, ys, zs = _cell_coords(s)
    j = int(0.8 / s.h)
    c = s.dims[2] // 2, j, s.dims[0] // 2
    assert sdf[c] == pytest.approx(-0.32, abs=0.06)          # half of 0.8 m * 0.8 scale
    out = int((0.8 - s.origin[0]) / s.h)
    assert sdf[s.dims[2] // 2, j, out] > 0.25
    fuel = s.read_scalars()[..., 1].astype(np.float32)
    assert fuel.sum() > 0 and fuel[sdf < -0.1].sum() == 0, 'fuel comes off the surface, never from inside the solid'


def test_steam_condenses_away_from_the_spout(engine):
    sc = presets.make('kettle_steam')
    sc.data['domain']['resolution'] = 64
    engine.invalidate()
    engine.prepare(sc)
    f = sc.start + int(1.5 * sc.fps)
    engine.simulate_to(sc, f, cache=True)
    engine.render(sc, f, (256, 256), mode='fire')
    alpha = engine.aovs()['beauty'][..., 3].astype(np.float32)
    assert alpha.max() > 0.1, 'the steam is visible'
    # cached frames keep the vapour, so they render the same steam
    engine.simulate_to(sc, f + 3, cache=True)
    engine.render(sc, f, (256, 256), mode='fire')
    alpha2 = engine.aovs()['beauty'][..., 3].astype(np.float32)
    assert np.abs(alpha2 - alpha).max() < 0.01, 'a cached frame renders as it did live'


def test_colourants_colour_the_flame(engine):
    sc = presets.make('coloured_flames')
    sc.data['domain']['resolution'] = 64
    engine.invalidate()
    engine.prepare(sc)
    f = sc.start + int(1.0 * sc.fps)
    engine.simulate_to(sc, f, cache=False)
    engine.render(sc, f, (384, 216), mode='fire')
    e = engine.aovs()['emission'][..., :3].astype(np.float32)
    left, mid = e[:, : 384 // 3].sum((0, 1)), e[:, 384 // 3: 2 * 384 // 3].sum((0, 1))
    assert left[1] > left[0], 'copper burns green'
    assert mid[0] > 2 * mid[1], 'strontium burns red'


def test_vdb_has_the_extra_grids(engine, tmp_path):
    from blackbody.io.vdb import read_vdb, write_vdb_frame
    sc = presets.make('hose_douse')
    sc.data['domain'].update(resolution=40, preroll=0.5)
    sc.data['combustion']['air'] = 'tracked'
    sc.emitters[0].update(color_amount=1.0)
    engine.invalidate()
    engine.prepare(sc)
    engine.simulate_to(sc, sc.start + int(2.5 * sc.fps), cache=False)
    p = tmp_path / 'f.vdb'
    write_vdb_frame(p, engine.solver, sc)
    names = set(read_vdb(p)['grids'])
    assert {'density', 'temperature', 'flame', 'fuel', 'vel', 'vapour', 'steam', 'oxygen_used', 'color'} <= names


def test_liquid_water_puts_fire_out(engine):
    # a liquid simulation sharing the box writes the share of each cell that is water into solver.water
    sc = presets.make('campfire')
    sc.data['domain'].update(resolution=40, preroll=0.0)
    sc.data['embers']['enabled'] = False
    engine.invalidate()
    engine.prepare(sc)
    engine.simulate_to(sc, sc.start + 24, cache=False)
    flame_dry = engine.solver.read_scalars()[..., 3].astype(np.float32).sum()
    s = engine.solver
    s.configure(s.dims, s.h, s.origin, dict(sc.features(), water=True, vapour=True, aux=True))
    engine.sim_frame = None
    # (x = the share of the cell that is water; y, z, w: soaked fuel, steam off it, char smoke: both_wet.wgsl)
    assert s.water is not None and s.water.format == 'rgba16float' and s.water.size == tuple(s.dims)
    nx, ny, nz = s.dims
    wet = np.zeros((nz, ny, nx, 4), np.float16)
    wet[:, : ny // 3, :, 0] = 1.0                # the lower third of the box flooded
    engine.simulate_to(sc, sc.start, cache=False)
    s.gpu.upload(s.water, wet)
    for f in range(sc.start + 1, sc.start + 25):
        engine.step_frame(sc, f)
        engine.sim_frame = f
    assert s.read_scalars()[..., 3].astype(np.float32).sum() < 0.1 * flame_dry
    assert s.read_aux()[..., 1].astype(np.float32).max() > 5.0, 'the water boiled to steam'


# -- round two: bigger limits, hollow colliders, holdouts, surface light, scorch, upres, heat --------------

def _sphere_mesh(n=60, r=0.5):
    th = np.linspace(0, np.pi, n + 1)
    ph = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing='ij')
    v = (np.stack([np.sin(T) * np.cos(P), np.cos(T), np.sin(T) * np.sin(P)], -1).reshape(-1, 3) * r).astype(np.float32)
    m = 2 * n
    tris = [(i * m + j, (i + 1) * m + j, i * m + (j + 1) % m) for i in range(n) for j in range(m)]
    tris += [(i * m + (j + 1) % m, (i + 1) * m + j, (i + 1) * m + (j + 1) % m) for i in range(n) for j in range(m)]
    return v, np.array(tris)


def test_banded_mesh_bake_matches_the_exact_one(engine):
    from blackbody.engine.mesh import bake
    v, t = _sphere_mesh(30)
    exact = bake(engine.gpu, v, t, 48, method='exact')
    band = bake(engine.gpu, v, t, 48, method='band')
    cell = (np.array(exact.bmax) - np.array(exact.bmin))[0] / exact.dims[0]
    near = np.abs(exact.data) < 3 * cell
    assert np.abs(band.data - exact.data)[near].max() < 1e-3
    assert (np.sign(band.data) == np.sign(exact.data))[np.abs(exact.data) > 0.02].all()


def test_baked_meshes_are_cached_on_disk(engine, tmp_path, monkeypatch):
    from blackbody.engine import mesh as meshmod
    monkeypatch.setenv('BLACKBODY_CACHE', str(tmp_path / 'cache'))
    cube = _cube(tmp_path)
    first = meshmod.load_or_bake(engine.gpu, cube, 32)
    assert list((tmp_path / 'cache' / 'meshes').glob('*.npz'))

    def no_bake(*a, **k):
        raise AssertionError('baked again')
    monkeypatch.setattr(meshmod, 'bake', no_bake)
    second = meshmod.load_or_bake(engine.gpu, cube, 32)
    assert np.array_equal(first.data, second.data) and second.dims == first.dims


def test_sixteen_emitters_and_hollow_colliders(engine):
    sc = _quiet(presets.make('campfire'), 48)
    sc.data['domain'].update(size_x=4.0, size_y=2.0, size_z=4.0)
    sc.emitters = [_em(sc, shape='sphere', position=(-1.6 + 0.2 * i, 0.2, 0.0), size=(0.06, 0.06, 0.06), fuel=0.0,
                       temperature=0.0, smoke=20.0, noise=0.0, embers=False) for i in range(16)]
    sc.colliders = [presets._col(sc, 0, shape='box', position=(0.0, 1.0, 1.2), size=(0.6, 0.5, 0.6), hollow=0.1,
                                 opening=(0.2, 0.2, 0.3), opening_at=(0.0, 0.0, -0.6))]
    engine.invalidate()
    engine.prepare(sc)
    engine.simulate_to(sc, sc.start + 3, cache=False)
    s = engine.solver
    smoke = s.read_scalars()[..., 2].astype(np.float32)
    j = int(0.2 / s.h)
    k = int((0.0 - s.origin[2]) / s.h)
    for i in range(16):  # every one of the 16 emitters released smoke
        assert smoke[k, j, int((-1.6 + 0.2 * i - s.origin[0]) / s.h)] > 0.01, i
    sdf = s.gpu.read(s.sdf)[..., 0] * s.h
    cell = lambda x, y, z: (int((z - s.origin[2]) / s.h), int(y / s.h), int((x - s.origin[0]) / s.h))
    assert sdf[cell(0.0, 1.0, 1.2)] > 0.2, 'hollow: air inside'
    assert sdf[cell(0.3, 1.0, 1.75)] < 0.0, 'the wall is solid'
    assert sdf[cell(0.0, 1.0, 0.63)] > 0.0, 'the opening goes through the wall'


def test_holdout_hides_the_fire_behind_it(engine):
    sc = presets.make('campfire')
    sc.data['domain']['resolution'] = 48
    sc.data['render']['width'], sc.data['render']['height'] = 256, 256
    sc.data['embers']['enabled'] = False
    # a wall between the camera (at +z) and the fire, covering its lower part
    sc.colliders = [presets._col(sc, 0, shape='box', position=(0.0, 0.3, 0.9), size=(1.2, 0.3, 0.05), holdout=False)]
    f = sc.start + 20
    alphas = {}
    cover = None
    for hold in (False, True):
        sc.colliders[0]['holdout'] = hold
        engine.invalidate()
        engine.prepare(sc)
        engine.simulate_to(sc, f, cache=False)
        engine.render(sc, f, (256, 256), mode='fire')
        aov = engine.aovs()
        alphas[hold] = aov['beauty'][..., 3].astype(np.float32)
        if hold:
            cover = aov['mask'][..., 2].astype(np.float32)
    behind = cover > 0.5
    assert behind.sum() > 100
    assert alphas[False][behind].max() > 0.3, 'without the holdout the fire shows there'
    assert alphas[True][behind].max() < 0.05, 'the holdout hides the fire behind it'


def test_fire_lights_the_ground_and_scorches_it(engine):
    sc = presets.make('grass_fire')
    sc.data['domain']['resolution'] = 64
    sc.data['embers']['enabled'] = False
    engine.invalidate()
    engine.prepare(sc)
    f = sc.start + int(6.0 * sc.fps)
    engine.simulate_to(sc, f, cache=True)
    engine.render(sc, f, (320, 180), mode='composite')
    aov = engine.aovs()
    light = aov['surface'][..., :3].astype(np.float32).mean(-1)
    scorch = aov['mask'][..., 0].astype(np.float32)
    assert light.max() > 0.1, 'the fire lights the ground'
    assert light[: 180 // 5].max() == 0.0, 'nothing is lit above the horizon'
    assert scorch.max() > 0.5, 'burnt ground shows as scorch'
    # a cached frame keeps its own burn state
    engine.simulate_to(sc, f + 12, cache=True)
    engine.render(sc, f, (320, 180), mode='composite')
    assert np.abs(engine.aovs()['mask'][..., 0].astype(np.float32) - scorch).max() < 0.02


def test_multiple_scattering_brightens_pale_smoke(engine):
    sc = presets.make('steam_vent')
    sc.data['domain']['resolution'] = 48
    engine.invalidate()
    engine.prepare(sc)
    f = sc.start + int(2.0 * sc.fps)
    engine.simulate_to(sc, f, cache=False)
    out = {}
    for ms in (0.0, 1.0):
        sc.data['shading']['multiple_scattering'] = ms
        engine.render(sc, f, (192, 192), mode='fire')
        out[ms] = engine.aovs()['beauty'][..., :3].astype(np.float32).sum()
    assert out[1.0] > 1.3 * out[0.0]


def test_upres_carries_finer_fire(engine, tmp_path):
    from blackbody.io.vdb import read_vdb, write_vdb_frame
    sc = presets.make('torch')
    sc.data['domain'].update(resolution=32, preroll=0.5)
    sc.data['render'].update(upres=2, final_scale=1.0)
    engine.invalidate()
    engine.prepare(sc, final=True)
    s = engine.solver
    assert s.upres == 2 and s.dims_fine == tuple(2 * d for d in s.dims)
    f = sc.start + 12
    engine.simulate_to(sc, f, cache=True)
    fine = s.read_scalars_fine().astype(np.float32)
    assert fine.shape[:3] == tuple(reversed(s.dims_fine)) and np.isfinite(fine).all()
    assert fine[..., 3].max() > 0.05, 'flames burn on the finer grid'
    engine.render(sc, f, (160, 160), mode='fire', final=True)
    assert engine.aovs()['beauty'][..., 3].astype(np.float32).max() > 0.1
    p = tmp_path / 'up.vdb'
    write_vdb_frame(p, s, sc)
    assert 'density' in read_vdb(p)['grids']
    engine.invalidate()


def test_heat_expansion_in_a_closed_box(engine):
    sc = presets.make('campfire')
    sc.data['domain'].update(resolution=40, preroll=0.0, open_sides=False, open_top=False, size_y=2.2)
    sc.data['combustion'].update(thermal_expansion=1.0, expansion=0.0)
    sc.data['embers']['enabled'] = False
    engine.invalidate()
    engine.prepare(sc)
    for f in range(sc.start, sc.start + 48, 8):
        engine.simulate_to(sc, f, cache=False)
        assert np.isfinite(engine.solver.max_speed) and engine.solver.max_speed < 30.0
    assert np.isfinite(engine.solver.read_scalars().astype(np.float32)).all()


def test_latent_heat_tracks_condensed_water(engine):
    peaks = {}
    for lat in (0.0, 1.0):
        sc = presets.make('kettle_steam')
        sc.data['domain']['resolution'] = 48
        sc.data['combustion']['latent_heat'] = lat
        engine.invalidate()
        engine.prepare(sc)
        engine.simulate_to(sc, sc.start + 24, cache=False)
        peaks[lat] = float(engine.solver.read_aux()[..., 2].astype(np.float32).max())
    assert peaks[1.0] > 1.0, 'condensed water is tracked'
    assert peaks[0.0] == 0.0


def test_a_burning_object_carries_its_fire(engine):
    from blackbody.scene.presets import K, _col
    sc = _quiet(presets.make('campfire'), 48)
    sc.data['domain'].update(size_x=4.0, size_y=2.0, size_z=2.0)
    sc.data['spread'].update(enabled=True, ground=False, coverage=1.0, burn_time=20.0, catch_time=0.1, creep=0.3)
    sc.colliders = [_col(sc, 0, shape='box', size=(0.3, 0.15, 0.3), burnable=True,
                         position=K((0.0, (-1.0, 0.3, 0.0)), (1.0, (-1.0, 0.3, 0.0)), (2.0, (1.0, 0.3, 0.0)), interp='linear'))]
    sc.emitters = [_em(sc, shape='sphere', position=(-1.0, 0.55, 0.0), size=(0.12, 0.08, 0.12), fuel=10.0, temperature=0.8,
                       stop=0.8, fade_out=0.1, embers=False)]
    engine.invalidate()
    engine.prepare(sc)
    engine.simulate_to(sc, sc.start + int(2.2 * sc.fps), cache=False)
    s = engine.solver
    atlas = s.read_burn_obj().astype(np.float32)
    assert ((atlas[..., 2] > 0.5) & (atlas[..., 1] >= 1.0)).sum() > 10, 'the object caught fire'
    fuel = s.read_scalars()[..., 1].astype(np.float32)
    xs, ys, zs = _cell_coords(s)
    near_now = fuel[:, :, np.abs(xs - 1.0) < 0.5].sum()
    near_start = fuel[:, :, np.abs(xs + 1.0) < 0.3].sum()
    assert near_now > 5 * max(near_start, 1e-6), 'its fuel comes off where the object is now'


def test_exr_has_light_holdout_and_scorch(engine, tmp_path):
    from blackbody.render.job import Output, RenderJob
    import OpenEXR
    sc = presets.make('armchair_fire')
    sc.data['domain'].update(resolution=40)
    sc.data['render'].update(width=160, height=90)
    p = tmp_path / 'e.####.exr'
    f = sc.start + 30
    engine.invalidate()
    RenderJob(sc, [Output('exr', str(p), 'element')], engine, frames=(f, f), final=False).run()
    with OpenEXR.File(str(tmp_path / f'e.{f:04d}.exr')) as fh:
        names = set(fh.channels())
    assert any(n.startswith('light') for n in names), names   # grouped as one RGB layer by OpenEXR
    assert {'holdout.Y', 'scorch.Y'} <= names
