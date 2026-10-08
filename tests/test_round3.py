"""Flame fronts, heavy vapour, spot fires, wet surfaces, soot, terrain, deforming meshes, a growing box,
the disk cache, USD import, footage holdouts, surface shadows, deep EXR, OCIO, motion blur from the
cache and upres velocity."""
import math

import numpy as np
import pytest

from blackbody.scene import presets
from blackbody.scene.presets import K, _col, _em

ROUND3_PRESETS = ['flash_fire', 'gas_cloud', 'backdraft', 'hillside_fire', 'spot_fires']


def _plain(sc):
    """Real-time and without puffing, whatever the look presets use."""
    sc.data['domain']['time_scale'] = 1.0
    if 'puffing' in sc.data['motion']:
        sc.data['motion']['puffing'] = 0.0
    return sc


def _quiet(sc, res=48):
    _plain(sc)
    sc.data['domain'].update(resolution=res, preroll=0.0)
    sc.data['motion'].update(turbulence=0.0, disturbance=0.0, vorticity=0.0, wind_speed=0.0)
    sc.data['embers']['enabled'] = False
    sc.emitters = []
    sc.colliders = []
    return sc


def _box_obj(path, half):
    """A closed box mesh of the given half size, written as OBJ."""
    hx, hy, hz = half
    v = [(sx * hx, sy * hy, sz * hz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    faces = ((1, 3, 4, 2), (5, 6, 8, 7), (1, 2, 6, 5), (3, 7, 8, 4), (1, 5, 7, 3), (2, 4, 8, 6))
    lines = [f'v {x} {y} {z}' for x, y, z in v] + ['f ' + ' '.join(map(str, f)) for f in faces]
    path.write_text('\n'.join(lines) + '\n')


# -- scene (no GPU) ------------------------------------------------------------------------------------

def test_round3_presets_build():
    for name in ROUND3_PRESETS:
        assert name in presets.ORDER
        sc = presets.make(name)
        assert sc.emitters
    assert presets.make('flash_fire').data['combustion']['flame_speed'] > 0
    assert presets.make('backdraft').features()['stain']
    assert presets.make('spot_fires').solver_params(1).spread.spotting > 0


def test_mesh_sources(tmp_path):
    from blackbody.engine.mesh import frame_file, is_numbered, load_mesh, mesh_deforms, sequence_files, split_source
    assert split_source('a/shot.usd#/World/Car') == ('a/shot.usd', '/World/Car')
    assert split_source('a/fire.####.obj') == ('a/fire.####.obj', '')
    assert is_numbered('a/fire.####.obj') and not is_numbered('shot.usd#/World/Car')
    for f in (3, 4, 5):
        _box_obj(tmp_path / f'blob.{f:04d}.obj', (0.1 * f, 0.2, 0.2))
    pat = str(tmp_path / 'blob.####.obj')
    assert sorted(sequence_files(pat)) == [3, 4, 5]
    assert frame_file(pat, 12).endswith('blob.0012.obj')
    assert mesh_deforms(pat)
    v, _ = load_mesh(pat, 4)
    assert v[:, 0].max() == pytest.approx(0.4)
    v, _ = load_mesh(pat, 99)            # holds the last frame
    assert v[:, 0].max() == pytest.approx(0.5)
    # a heightfield image is a closed terrain solid from 0 up to 1.02
    from PIL import Image
    img = np.linspace(0, 255, 64 * 64).reshape(64, 64).astype(np.uint8)
    Image.fromarray(img).save(tmp_path / 'hill.png')
    v, t = load_mesh(str(tmp_path / 'hill.png'))
    assert v.min(0) == pytest.approx([-0.5, 0.0, -0.5]) and v[:, 1].max() == pytest.approx(1.02, abs=1e-3)


def test_new_settings_reach_the_solver():
    sc = presets.make('campfire')
    sc.data['combustion'].update(flame_speed=3.0, fuel_weight=0.5, soot_stain=0.2)
    sc.data['spread'].update(dry_time=12.0, spotting=0.1)
    p = sc.solver_params(1)
    assert (p.flame_speed, p.fuel_weight, p.soot_stain) == (3.0, 0.5, 0.2)
    assert (p.spread.dry_time, p.spread.spotting) == (12.0, 0.1)
    assert sc.features()['stain']
    a = sc.sim_signature()
    sc.data['domain'].update(disk_cache=True, cache_dir='x', checkpoint_every=3)
    assert sc.sim_signature() == a, 'where frames are kept does not change the simulation'
    sc.data['render']['upres'] = 2
    assert sc.upres_for(False) == 1 and sc.upres_for(True) == 2
    sc.data['render']['upres_preview'] = True
    assert sc.upres_for(False) == 2


def test_simcache_round_trip(tmp_path):
    from blackbody.io.simcache import CacheMismatch, SimCache, read_entry, write_entry
    e = {'scal': np.arange(24, dtype=np.float16).reshape(2, 3, 4), 'time': 1.5, 'dims': [4, 3, 2],
         'state': {'vel': np.ones((2, 2), np.float32), 'k': (1, 2)}, 'names': {3: 'x'}}
    write_entry(tmp_path / 'a.bbc', e)
    r = read_entry(tmp_path / 'a.bbc')
    assert np.array_equal(r['scal'], e['scal']) and r['time'] == 1.5 and r['state']['k'] == (1, 2) and r['names'] == {3: 'x'}
    c = SimCache(tmp_path / 'c', 'sig1')
    c.put(5, e)
    c.mark_checkpoint(5)
    c.flush()
    assert 5 in c and c.checkpoints() == [5]
    with pytest.raises(CacheMismatch):
        SimCache(tmp_path / 'c', 'sig2', readonly=True)
    assert 5 in SimCache(tmp_path / 'c', 'sig1', readonly=True)
    assert 5 not in SimCache(tmp_path / 'c', 'sig2'), 'other settings clear the cache'


def test_ocio_view_lut_matches_the_processor():
    ocio = pytest.importorskip('PyOpenColorIO')
    from blackbody.io import ocio as O
    pipe = O.pipeline({'view': 'ocio'})
    x = np.array([[0.02, 0.05, 0.1], [0.18, 0.18, 0.18], [3.0, 1.2, 0.3]], np.float32)
    exact = pipe.to_display(x)
    lut = pipe.view_lut(33)
    # trilinear lookup through the shaper
    s = O.shaper(x) * 32
    i0 = np.clip(np.floor(s).astype(int), 0, 31)
    f = s - i0
    out = np.zeros_like(exact)
    for c in range(8):
        o = np.array([c & 1, (c >> 1) & 1, c >> 2])
        w = np.prod(np.where(o, f, 1 - f), axis=1)
        idx = i0 + o
        out += w[:, None] * lut[idx[:, 2], idx[:, 1], idx[:, 0], :3]
    assert np.abs(out - exact).max() < 0.01
    assert pipe.to_space(np.array([[0.18, 0.18, 0.18]]), 'ACEScg') == pytest.approx(0.18, abs=1e-3)


def test_usd_import_places_camera_and_objects(tmp_path):
    pxr = pytest.importorskip('pxr')
    from pxr import Gf, Usd, UsdGeom, Vt
    from blackbody.engine import camera as cam
    from blackbody.io import usd as U
    path = tmp_path / 'shot.usda'
    st = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 0.01)
    st.SetFramesPerSecond(24)
    st.SetTimeCodesPerSecond(24)
    st.SetStartTimeCode(1)
    st.SetEndTimeCode(24)
    box = UsdGeom.Mesh.Define(st, '/World/Box')
    box.CreatePointsAttr([(-50, -50, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0),
                          (-50, -50, 100), (50, -50, 100), (50, 50, 100), (-50, 50, 100)])
    box.CreateFaceVertexCountsAttr([4] * 6)
    box.CreateFaceVertexIndicesAttr([0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7])
    x = UsdGeom.Xformable(box)
    t_op, r_op = x.AddTranslateOp(), x.AddRotateZOp()
    for f in (1, 24):
        t_op.Set(Gf.Vec3d(100 + 10 * f, 200, 0), f)
        r_op.Set(3.0 * f, f)
    blob = UsdGeom.Mesh.Define(st, '/World/Blob')
    blob.CreateFaceVertexCountsAttr([3] * 4)
    blob.CreateFaceVertexIndicesAttr([0, 2, 1, 0, 1, 3, 1, 2, 3, 0, 3, 2])
    for f in (1, 24):
        s = 20 + 2 * f
        blob.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0, 0, 0), Gf.Vec3f(s, 0, 0), Gf.Vec3f(0, s, 0), Gf.Vec3f(0, 0, s)]), f)
    c = UsdGeom.Camera.Define(st, '/World/Cam')
    cx = UsdGeom.Xformable(c)
    cx.AddTranslateOp().Set(Gf.Vec3d(0, -500, 150))
    cx.AddRotateXOp().Set(90.0)
    c.CreateFocalLengthAttr(35.0)
    c.CreateHorizontalApertureAttr(36.0)
    st.GetRootLayer().Save()

    info = U.scan(path)
    assert {(i.path, i.motion) for i in info.meshes()} == {('/World/Box', 'rigid'), ('/World/Blob', 'world')}
    sc = presets.make('campfire')
    sc.data['camera'].update(fire_yaw=30.0, fire_position=(0.5, 0.0, -0.2))
    U.import_usd(sc, path)
    i = [c['name'] for c in sc.colliders].index('Box')
    box_c = sc.colliders[i]
    v, _ = U.load_usd_mesh(path, '/World/Box', 12)
    R = cam.rot_y(math.radians(30.0))
    for f in (1, 12, 24):
        pos = np.asarray(sc.get(('collider', i, 'position'), f))
        yaw = sc.get(('collider', i, 'yaw'), f)
        p_f = pos + cam.rot_y(math.radians(yaw)) @ (np.asarray(box_c['size']) * v[6])
        m = np.array(UsdGeom.Xformable(st.GetPrimAtPath('/World/Box')).ComputeLocalToWorldTransform(Usd.TimeCode(f))).T
        w = m @ np.array([50, 50, 100.0, 1.0])
        w = np.array([w[0], w[2], -w[1]]) * 0.01
        assert p_f == pytest.approx(R.T @ (w - np.array([0.5, 0.0, -0.2])), abs=1e-6)
    assert sc.colliders[[c['name'] for c in sc.colliders].index('Blob')]['mesh'].endswith('#/World/Blob?world')
    spec, fire = sc.camera(1)
    cs = cam.compute(spec, 16 / 9, fire)
    assert cs.eye == pytest.approx([0.0, 1.5, 5.0], abs=1e-6)
    assert (-np.linalg.inv(cs.view)[:3, 2]) == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)


# -- GPU ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('name', ROUND3_PRESETS)
def test_round3_presets_stay_stable(engine, name):
    sc = presets.make(name)
    sc.data['domain']['resolution'] = 48
    engine.invalidate()
    engine.prepare(sc, final=False)
    for f in range(sc.start, sc.start + 60, 6):
        engine.simulate_to(sc, f, cache=False)
        assert np.isfinite(engine.solver.max_speed) and engine.solver.max_speed < 100.0
    assert np.isfinite(engine.solver.read_scalars().astype(np.float32)).all()


def _vapour_scene(speed, weight=1.0):
    sc = _plain(presets.make('campfire'))
    sc.data['domain'].update(resolution=48, preview_scale=1.0, preroll=0.0, size_x=3.0, size_z=1.0, size_y=1.5)
    sc.data['combustion'].update(flame_speed=speed, fuel_weight=weight, fuel_dissipation=0.02)
    sc.data['embers']['enabled'] = False
    sc.emitters = [
        _em(sc, shape='box', position=(-1.0, 0.02, 0.0), size=(0.3, 0.02, 0.3), fuel=4.0, temperature=0.0, noise=0.0, embers=False),
        _em(sc, shape='sphere', position=(-0.5, 0.05, 0.0), size=(0.06, 0.06, 0.06), fuel=0.0, temperature=1.5, start=3.0,
            embers=False)]
    return sc


def test_flame_front_runs_back_through_the_vapour(engine):
    reach = {}
    for speed in (0.0, 4.0):
        sc = _vapour_scene(speed)
        engine.invalidate()
        engine.prepare(sc)
        engine.simulate_to(sc, sc.start + 96, cache=False)
        T = engine.solver.read_scalars()[..., 0].astype(np.float32)
        hot = np.where((T > 0.3).any(axis=(0, 1)))[0]
        reach[speed] = (hot.min() * engine.solver.h - 1.5) if len(hot) else 0.0
    assert reach[4.0] < -1.1, f'the front reached the puddle ({reach[4.0]:.2f} m)'
    assert reach[0.0] > -0.8, f'without a front the fire stays at the spark ({reach[0.0]:.2f} m)'


def test_heavy_vapour_hugs_the_ground(engine):
    heights = {}
    for weight in (0.0, 1.0):
        sc = _vapour_scene(0.0, weight)
        # a puff of cold vapour released in mid-air: heavy vapour sinks, neutral vapour stays put
        sc.emitters = [_em(sc, shape='sphere', position=(0.0, 0.7, 0.0), size=(0.15, 0.15, 0.15), fuel=6.0, temperature=0.0,
                           noise=0.0, stop=0.5, fade_out=0.05, embers=False)]
        engine.invalidate()
        engine.prepare(sc)
        engine.simulate_to(sc, sc.start + 48, cache=False)
        F = engine.solver.read_scalars()[..., 1].astype(np.float32)
        heights[weight] = float((F.sum(axis=(0, 2)) * np.arange(F.shape[1])).sum() / F.sum())
    assert heights[1.0] < 0.6 * heights[0.0], heights


def test_soaked_ground_does_not_relight_until_dry(engine):
    def run(dry):
        sc = _quiet(presets.make('campfire'), 48)
        sc.data['domain'].update(size_x=2.0, size_z=2.0, size_y=1.5)
        sc.data['spread'].update(enabled=True, ground=True, coverage=1.0, burn_time=30.0, catch_time=0.2, creep=0.2,
                                 dry_time=dry)
        sc.emitters = [
            _em(sc, shape='box', position=(0, 0.5, 0), size=(1.0, 0.5, 1.0), fuel=0.0, temperature=0.0, noise=0.0, douse=30.0,
                stop=0.5, fade_out=0.05, embers=False),
            _em(sc, shape='sphere', position=(0.0, 0.1, 0.0), size=(0.2, 0.1, 0.2), fuel=10.0, temperature=0.8, start=0.7,
                fade_in=0.05, embers=False)]
        engine.invalidate()
        engine.prepare(sc)
        engine.simulate_to(sc, sc.start + int(1.6 * sc.fps), cache=False)
        b = engine.solver.read_burn().astype(np.float32)
        floor = b[:, 0]
        return int(((floor[..., 2] > 0.5) & (floor[..., 1] >= 1.0)).sum()), float((-floor[..., 1]).max())
    wet_lit, wetness = run(60.0)
    dry_lit, _ = run(0.0)
    assert wetness > 0.5, 'the hosed ground is soaked'
    assert dry_lit > 5 and wet_lit < 0.3 * dry_lit, (wet_lit, dry_lit)


def test_embers_start_spot_fires(engine):
    counts = {}
    for spot in (0.0, 0.2):
        sc = presets.make('spot_fires')
        sc.data['domain'].update(resolution=64)
        sc.data['spread']['spotting'] = spot
        engine.invalidate()
        engine.prepare(sc)
        engine.simulate_to(sc, sc.start + int(4.0 * sc.fps), cache=False)
        b = engine.solver.read_burn().astype(np.float32)[:, 0]
        burnt = (b[..., 2] > 1.5)   # has caught at some point
        xs = np.where(burnt.any(axis=0))[0]
        counts[spot] = (int(burnt.sum()), xs.max() if len(xs) else 0)
    assert counts[0.2][0] > counts[0.0][0], counts
    assert counts[0.2][1] >= counts[0.0][1], 'spot fires land ahead of the front'


def test_soot_stains_the_walls(engine):
    sc = presets.make('backdraft')
    sc.data['domain'].update(resolution=48)
    engine.invalidate()
    engine.prepare(sc)
    engine.simulate_to(sc, sc.start + 48, cache=True)
    walls = engine.solver.read_stain_obj()[..., 0]
    assert walls.max() > 0.05, 'soot on the room, in its own frame'
    st = engine.solver.read_stain()[..., 0]
    sdf = engine.gpu.read(engine.solver.sdf)[..., 0]
    assert st[sdf < 0].max() == 0.0, 'no soot inside solids'
    v, live = engine.volume_for(sc, sc.start + 30)
    assert v.stain is not None and v.stain_obj is not None and not live


def test_fire_runs_uphill_on_terrain(engine):
    sc = presets.make('hillside_fire')
    sc.data['domain'].update(resolution=64)
    sc.data['embers']['enabled'] = False
    engine.invalidate()
    engine.prepare(sc)
    s = engine.solver
    assert not s.meshes.errors
    engine.simulate_to(sc, sc.start + int(5.0 * sc.fps), cache=False)
    atlas = s.read_burn_obj().astype(np.float32)
    caught = (atlas[..., 2] > 1.5)
    assert caught.sum() > 50, 'the slope caught'
    # the burn region is laid out in the collider's frame: z runs from the back (top of the hill) to the front
    zs = np.where(caught.any(axis=(1, 2)))[0]
    lo = s.burn_slots and np.frombuffer(s.gpu.read_buffer(s.burn_slots), np.float32)[:8]
    zmin = lo[2] + zs.min() * lo[7]
    assert zmin < 3.0, f'the fire climbed from the foot of the hill (z 3.4) toward the top (reached {zmin:.2f})'


def test_deforming_mesh_sequence_pushes_the_air(engine, tmp_path):
    for f in range(1, 40):
        r = 0.1 + 0.02 * f
        _box_obj(tmp_path / f'grow.{f:04d}.obj', (r, r, r))
    pat = str(tmp_path / 'grow.####.obj')
    sc = _quiet(presets.make('campfire'), 48)
    sc.data['domain'].update(size_x=2.0, size_y=2.0, size_z=2.0)
    sc.colliders = [_col(sc, 0, shape='mesh', mesh=pat, position=(0.0, 1.0, 0.0), size=(1.0, 1.0, 1.0), holdout=False)]
    engine.invalidate()
    engine.prepare(sc)
    s = engine.solver
    engine.simulate_to(sc, sc.start + 20, cache=False)
    assert not s.meshes.errors
    sdf = s.gpu.read(s.sdf)[..., 0] * s.h
    j = int(1.0 / s.h)
    c = s.dims[0] // 2
    half = 0.1 + 0.02 * (sc.start + 19.5)
    row = sdf[s.dims[2] // 2, j]
    inside = (np.arange(s.dims[0]) + 0.5) * s.h + s.origin[0]
    edge = inside[np.where(row < 0)[0]]
    assert edge.max() == pytest.approx(half, abs=1.5 * s.h), 'the distance field follows the sequence'
    v = s.read_velocity_centres()
    out = v[s.dims[2] // 2, j, c + int((half + 2 * s.h) / s.h), 0]
    assert out > 0.1, f'the growing box pushes the air outward ({out:.2f} m/s)'


def test_growing_domain_keeps_the_simulation(engine):
    sc = _plain(presets.make('campfire'))
    sc.data['domain'].update(resolution=48, preroll=0.0, size_x=1.0, size_z=1.0, size_y=1.5, grow=True, grow_limit=2.0)
    sc.data['embers']['enabled'] = False
    engine.invalidate()
    engine.prepare(sc)
    d0 = engine.solver.dims
    engine.simulate_to(sc, sc.start + 30, cache=True)
    s = engine.solver
    assert s.grown() and all(a >= b for a, b in zip(s.dims, d0)) and s.dims != d0
    assert all(a <= 2 * b for a, b in zip(s.dims, d0))
    assert np.isfinite(s.read_scalars().astype(np.float32)).all()
    engine.render(sc, sc.start + 2, (160, 90))    # a frame from before it grew
    engine.render(sc, sc.start + 30, (160, 90))
    engine.reset()
    assert tuple(engine.solver.dims) == tuple(d0), 'a fresh start goes back to the box asked for'


def test_disk_cache_resumes_from_a_checkpoint(engine, tmp_path):
    from blackbody.engine.engine import Engine
    sc = _plain(presets.make('campfire'))
    sc.data['domain'].update(resolution=40, preroll=0.0, disk_cache=True, cache_dir=str(tmp_path), checkpoint_every=4)
    engine.invalidate()
    engine.prepare(sc)
    engine.simulate_to(sc, sc.start + 10)
    engine.cache.disk.flush()
    ref = engine.solver.read_scalars().astype(np.float32)
    assert engine.cache.disk.checkpoints() == [sc.start, sc.start + 4, sc.start + 8]
    other = Engine(engine.gpu)
    other.prepare(sc)
    other.simulate_to(sc, sc.start + 10)
    assert np.abs(other.solver.read_scalars().astype(np.float32) - ref).max() < 1e-3, 'resumed exactly'
    other.cache.clear()
    v, live = other.volume_for(sc, sc.start + 3)
    assert v is not None and not live, 'frames come back from disk'
    engine.cache.attach(None)
    other.cache.attach(None)


def test_the_disk_cache_keeps_to_the_code_and_the_objects_it_was_simulated_with(engine, tmp_path, monkeypatch):
    # (bottle_shoot's frames cached before its bottles burst still showed them standing: the signature had no version of
    # the code, nor of the objects as built; and a checkpoint the objects refused was refused after the solver was set up
    # for it)
    from blackbody.engine.engine import Engine
    from blackbody.engine.solids import Solids
    from blackbody.scene import model
    sc = _plain(presets.make('campfire'))
    sc.data['domain'].update(resolution=40, preroll=0.0, disk_cache=True, cache_dir=str(tmp_path), checkpoint_every=4)
    sc.add_collider(name='Box', shape='box', position=(0.5, 0.4, 0.0), size=(0.08, 0.08, 0.08), dynamic=True, material='wood')
    engine.invalidate()
    engine.prepare(sc)
    assert engine.solids.active
    engine.simulate_to(sc, sc.start + 8)
    disk = engine.cache.disk
    disk.flush()
    assert disk.checkpoints() == [sc.start, sc.start + 4, sc.start + 8]
    engine.cache.attach(None)
    for c in disk.checkpoints():            # (objects that no longer fit them: another number of bodies)
        e = disk.get(c)
        e['solids']['qpos'] = np.zeros(3)
        disk.put(c, e, wait=True)
    other = Engine(engine.gpu)
    other.prepare(sc)
    assert sc.start + 4 in other.cache.disk, 'the same scene and code: its frames are kept'
    seen = []
    monkeypatch.setattr(other.solver, 'set_meshes', lambda *a, **k: seen.append('meshes'))
    monkeypatch.setattr(other.solver, 'load_state', lambda *a, **k: seen.append('state'))
    other.sim_frame = sc.start - 1
    assert not other._resume(sc, sc.start + 8)
    assert not seen, 'nothing set up for a checkpoint the objects refuse'
    monkeypatch.undo()
    # frames simulated by older code, or with the objects as built otherwise, are simulated again
    monkeypatch.setattr(model, 'SIM_VERSION', model.SIM_VERSION + 1)
    other.prepare(sc)
    assert not other.cache.disk.frames()
    other.simulate_to(sc, sc.start + 2)
    other.cache.disk.flush()
    assert other.cache.disk.frames()
    monkeypatch.setattr(Solids, 'fingerprint', lambda self: 'cut otherwise')
    other.prepare(sc)
    assert not other.cache.disk.frames()
    other.cache.attach(None)


def test_surface_shadows_and_footage_holdouts(engine):
    from blackbody.engine import camera as cam
    sc = _plain(presets.make('campfire'))
    sc.data['domain'].update(resolution=48, preroll=1.0)
    sc.data['embers']['enabled'] = False
    sc.colliders = [_col(sc, 0, shape='box', position=(0.7, 0.35, -0.4), size=(0.15, 0.35, 0.15))]
    engine.invalidate()
    engine.prepare(sc)
    f = sc.start + 12
    engine.simulate_to(sc, f, cache=False)
    W, H = 240, 136
    light = {}
    for sh in (0.0, 1.0):
        sc.data['composite']['surface_shadows'] = sh
        engine.render(sc, f, (W, H))
        light[sh] = engine.aovs()['surface'][..., :3].astype(np.float32).sum()
    assert light[1.0] < light[0.0], 'colliders and smoke shade the fire light'
    engine.render(sc, f, (W, H))
    a0 = engine.aovs()['beauty'][..., 3].astype(np.float32)
    matte = np.zeros((H, W), np.float32)
    matte[:, W // 2:] = 1.0
    engine.render(sc, f, (W, H), holdout=(matte, None))
    aov = engine.aovs()
    a1 = aov['beauty'][..., 3].astype(np.float32)
    assert a1[:, W // 2 + 1:].max() == 0.0 and a1[:, :W // 2 - 1].sum() == pytest.approx(a0[:, :W // 2 - 1].sum(), rel=1e-3)
    assert aov['mask'][..., 2].astype(np.float32)[:, W // 2 + 1:].min() > 0.99, 'embers are held out too'
    engine.render(sc, f, (W, H), holdout=(None, np.full((H, W), 2.0, np.float32)))
    assert engine.aovs()['beauty'][..., 3].astype(np.float32).max() == 0.0, 'a surface 2 m away hides the fire'
    engine.render(sc, f, (W, H), holdout=(None, np.full((H, W), 50.0, np.float32)))
    assert engine.aovs()['beauty'][..., 3].astype(np.float32).sum() == pytest.approx(a0.sum(), rel=0.02)


def test_deep_samples_flatten_to_the_beauty(engine, tmp_path):
    from blackbody.io.images import write_deep_exr
    import OpenEXR
    sc = _plain(presets.make('campfire'))
    sc.data['domain'].update(resolution=48, preroll=1.0)
    sc.data['embers']['enabled'] = False
    engine.invalidate()
    engine.prepare(sc)
    f = sc.start + 6
    engine.simulate_to(sc, f, cache=False)
    for samples in (1, 4):
        # anti-aliasing passes share the deep bins, so the samples composite back to the averaged beauty
        engine.render(sc, f, (200, 120), deep=8, samples=samples)
        d = engine.renderer.read_deep()
        b = engine.aovs()['beauty'].astype(np.float32)
        order = np.argsort(np.where(d[..., 3] + d[..., :3].max(-1) > 0, d[..., 4], np.inf), axis=-1)
        d = np.take_along_axis(d, order[..., None], axis=2)
        acc = np.zeros(d.shape[:2] + (4,), np.float32)
        tr = np.ones(d.shape[:2], np.float32)
        for k in range(d.shape[2]):
            acc += d[:, :, k, :4] * tr[..., None]
            tr *= 1.0 - d[:, :, k, 3]
        assert np.abs(acc - b).max() < 0.02 * max(1.0, float(b.max())), samples
    assert ((d[..., 3] > 1e-4).sum(-1) > 1).any(), 'pixels hold several samples'
    z = d[..., 4][d[..., 3] > 1e-3]
    assert z.min() > 1.0 and (d[..., 5] >= d[..., 4] - 1e-4).all()
    p = tmp_path / 'd.exr'
    write_deep_exr(p, d)
    with OpenEXR.File(str(p)) as fh:
        assert fh.header()['type'] == OpenEXR.deepscanline and 'Z' in fh.channels()


def test_ocio_view_in_the_composite(engine):
    pytest.importorskip('PyOpenColorIO')
    from blackbody.io import ocio as O
    sc = _plain(presets.make('campfire'))
    sc.data['domain'].update(resolution=40, preroll=0.5)
    engine.invalidate()
    engine.prepare(sc)
    f = sc.start + 3
    engine.simulate_to(sc, f, cache=False)
    sc.data['composite']['view'] = 'ocio'
    engine.render(sc, f, (160, 90))
    disp = engine.display_image()[..., :3].astype(np.float32)
    exact = O.pipeline(sc.data['composite']).to_display(engine.linear_comp()[..., :3].astype(np.float32)) * 255
    assert np.abs(disp - exact).max() <= 3.0
    sc.data['composite']['ocio_config'] = 'Z:/no/such/config.ocio'
    engine.render(sc, f, (160, 90))    # a bad config falls back to the Standard view instead of failing


def test_cached_frames_get_motion_blur(engine):
    sc = _plain(presets.make('torch'))
    sc.data['domain'].update(resolution=40, preroll=0.5)
    engine.invalidate()
    engine.prepare(sc)
    engine.simulate_to(sc, sc.start + 8)
    v, live = engine.volume_for(sc, sc.start + 5)
    assert not live and v.vel[0] is not engine._zero_vel
    out = []
    for mb in (False, True):
        engine.render(sc, sc.start + 5, (160, 160), mode='fire', motion_blur=mb)
        out.append(engine.aovs()['beauty'].astype(np.float32))
    assert np.abs(out[0] - out[1]).mean() > 2e-3 * out[0].mean(), 'the cached velocity blurs the frame'


def test_upres_in_the_viewer_and_its_velocity(engine):
    sc = _plain(presets.make('torch'))
    sc.data['domain'].update(resolution=32, preroll=0.3)
    sc.data['render'].update(upres=2, upres_preview=True)
    engine.invalidate()
    engine.prepare(sc, final=False)
    s = engine.solver
    assert s.upres == 2
    engine.simulate_to(sc, sc.start + 6, cache=False)
    vf = s.read_velocity_fine()
    assert vf.shape[:3] == tuple(reversed(s.dims_fine))
    coarse = s.read_velocity_centres()
    rep = np.repeat(np.repeat(np.repeat(coarse, 2, 0), 2, 1), 2, 2)
    assert np.abs(vf - rep).mean() > 1e-3, 'the fine velocity is not just the coarse one repeated'
    assert np.isfinite(vf).all()


def test_embers_start_spot_fires_on_burnable_objects(engine):
    """Hot sparks raining onto a burnable deck (a collider, not the floor) start fires on it."""
    burnt = {}
    for spot in (0.0, 0.3):
        sc = _quiet(presets.make('campfire'), 48)
        sc.data['domain'].update(size_x=2.0, size_y=2.0, size_z=2.0)
        sc.data['spread'].update(enabled=True, ground=False, coverage=1.0, burn_time=20.0, catch_time=5.0, creep=0.0,
                                 spotting=spot, spot_temp=600.0)
        sc.data['embers'].update(enabled=True, rate=600, direction=(0.0, -1.0, 0.0), cone=25.0, launch=2.0, gravity=9.81,
                                 lifetime=1.5, cooling=0.1)
        sc.emitters = [_em(sc, shape='sphere', position=(0.0, 1.6, 0.0), size=(0.05, 0.05, 0.05), fuel=0.0, temperature=0.0,
                           noise=0.0)]
        sc.colliders = [_col(sc, 0, shape='box', position=(0.0, 0.3, 0.0), size=(0.7, 0.05, 0.7), burnable=True)]
        engine.invalidate()
        engine.prepare(sc)
        engine.simulate_to(sc, sc.start + 36, cache=False)
        atlas = engine.solver.read_burn_obj().astype(np.float32)
        burnt[spot] = int((atlas[..., 2] > 1.5).sum())
    assert burnt[0.0] == 0, 'no flame touches the deck'
    assert burnt[0.3] > 5, burnt



def test_deep_output_includes_the_embers(engine):
    sc = _plain(presets.make('campfire'))
    sc.data['domain'].update(resolution=40, preroll=1.0)
    sc.data['embers'].update(enabled=True, rate=400)
    engine.invalidate()
    engine.prepare(sc)
    f = sc.start + 8
    engine.simulate_to(sc, f, cache=False)
    engine.render(sc, f, (200, 120), deep=8, samples=2)
    d = engine.renderer.read_deep()
    b = engine.aovs()['beauty'].astype(np.float32)
    ember = d[:, :, -1]
    assert (ember[..., :3].max(-1) > 0).sum() > 5, 'embers have their own deep samples'
    assert (ember[..., 3] == 0).all(), 'ember light has no alpha'
    order = np.argsort(np.where(d[..., 3] + d[..., :3].max(-1) > 0, d[..., 4], np.inf), axis=-1)
    d = np.take_along_axis(d, order[..., None], axis=2)
    acc = np.zeros(d.shape[:2] + (3,), np.float32)
    tr = np.ones(d.shape[:2], np.float32)
    for k in range(d.shape[2]):
        acc += d[:, :, k, :3] * tr[..., None]
        tr *= 1.0 - d[:, :, k, 3]
    lit = ember[..., :3].max(-1) > 0
    assert np.abs(acc - b[..., :3])[lit].max() < 0.05 * max(1.0, float(b[..., :3][lit].max()))


def test_soot_moves_with_a_moving_collider(engine):
    """Soot laid on a block while it sits over a smoky fire goes with the block when it moves away."""
    sc = _quiet(presets.make('campfire'), 48)
    sc.data['domain'].update(size_x=3.0, size_y=2.0, size_z=2.0)
    sc.data['combustion'].update(soot_stain=1.0, soot=2.0)
    sc.emitters = [_em(sc, shape='cylinder', position=(0.0, 0.05, 0.0), size=(0.2, 0.05, 0.2), fuel=10.0, temperature=0.6,
                       smoke=10.0, embers=False)]
    sc.colliders = [_col(sc, 0, shape='box', size=(0.3, 0.1, 0.3),
                         position=K((0.0, (0.0, 0.6, 0.0)), (2.0, (0.0, 0.6, 0.0)), (2.5, (1.0, 0.6, 0.0)), interp='linear'))]
    engine.invalidate()
    engine.prepare(sc)
    engine.simulate_to(sc, sc.start + int(2.0 * sc.fps), cache=True)
    s = engine.solver
    before = float(s.read_stain_obj().sum())
    assert before > 0.1, 'soot builds up on the block over the fire'
    engine.simulate_to(sc, sc.start + int(2.6 * sc.fps), cache=True)
    after = float(s.read_stain_obj().sum())
    assert after >= 0.9 * before, 'the soot stays on the block as it moves'
    # the render finds it where the block is now, and a cached frame where the block was then
    for f in (sc.start + int(2.6 * sc.fps), sc.start + int(2.0 * sc.fps)):
        engine.render(sc, f, (240, 136), mode='composite')
        assert engine.aovs()['mask'][..., 3].astype(np.float32).max() > 0.05, f


def _build_usd_scene(path):
    """Shapes, instances, a skinned mesh and lights in one Z-up stage."""
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdSkel
    st = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    st.SetFramesPerSecond(24)
    st.SetTimeCodesPerSecond(24)
    st.SetStartTimeCode(1)
    st.SetEndTimeCode(24)
    UsdGeom.Xform.Define(st, '/World')
    cube = UsdGeom.Cube.Define(st, '/World/Crate')
    cube.CreateSizeAttr(1.0)
    x = UsdGeom.Xformable(cube)
    x.AddTranslateOp().Set(Gf.Vec3d(2, 0, 0.5))
    x.AddScaleOp().Set(Gf.Vec3f(2, 1, 1))
    sph = UsdGeom.Sphere.Define(st, '/World/Ball')
    sph.CreateRadiusAttr(0.3)
    sx = UsdGeom.Xformable(sph)
    t = sx.AddTranslateOp()
    t.Set(Gf.Vec3d(-2, 0, 0.3), 1)
    t.Set(Gf.Vec3d(-2, 2, 0.3), 24)
    cyl = UsdGeom.Cylinder.Define(st, '/World/Post')
    cyl.CreateRadiusAttr(0.1)
    cyl.CreateHeightAttr(2.0)
    cyl.CreateAxisAttr('Z')
    UsdGeom.Xformable(cyl).AddTranslateOp().Set(Gf.Vec3d(0, -2, 1))
    cap = UsdGeom.Capsule.Define(st, '/World/Pipe')
    cap.CreateRadiusAttr(0.1)
    cap.CreateHeightAttr(1.0)
    cap.CreateAxisAttr('X')
    UsdGeom.Xformable(cap).AddTranslateOp().Set(Gf.Vec3d(0, 2, 0.1))
    hid = UsdGeom.Cube.Define(st, '/World/Hidden')
    hid.CreateVisibilityAttr('invisible')
    # a point instancer: three small cubes
    pi = UsdGeom.PointInstancer.Define(st, '/World/Rocks')
    proto = UsdGeom.Cube.Define(st, '/World/Rocks/Protos/Rock')
    proto.CreateSizeAttr(0.2)
    pi.CreatePrototypesRel().SetTargets([proto.GetPath()])
    pi.CreateProtoIndicesAttr([0, 0, 0])
    pi.CreatePositionsAttr([Gf.Vec3f(1, 1, 0.1), Gf.Vec3f(1.5, 1, 0.1), Gf.Vec3f(2, 1, 0.1)])
    # an instanced reference: a prototype-able prim marked instanceable
    UsdGeom.Xform.Define(st, '/Library/Chair')
    seat = UsdGeom.Cube.Define(st, '/Library/Chair/Seat')
    seat.CreateSizeAttr(0.5)
    ref = UsdGeom.Xform.Define(st, '/World/ChairA')
    ref.GetPrim().GetReferences().AddInternalReference('/Library/Chair')
    ref.GetPrim().SetInstanceable(True)
    UsdGeom.Xformable(ref).AddTranslateOp().Set(Gf.Vec3d(-1, -1, 0.25))
    st.GetPrimAtPath('/Library').SetActive(False)   # the library itself is not in the scene
    # a skinned mesh: two joints, the top one bends
    root = UsdSkel.Root.Define(st, '/World/Guy')
    skel = UsdSkel.Skeleton.Define(st, '/World/Guy/Skel')
    joints = ['hip', 'hip/spine']
    skel.CreateJointsAttr(joints)
    skel.CreateBindTransformsAttr([Gf.Matrix4d(1), Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, 1))])
    skel.CreateRestTransformsAttr([Gf.Matrix4d(1), Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, 1))])
    anim = UsdSkel.Animation.Define(st, '/World/Guy/Skel/Anim')
    anim.CreateJointsAttr(joints)
    anim.CreateTranslationsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(0, 0, 1)])
    anim.CreateScalesAttr([Gf.Vec3h(1, 1, 1), Gf.Vec3h(1, 1, 1)])
    rot = anim.CreateRotationsAttr()
    rot.Set([Gf.Quatf(1, 0, 0, 0), Gf.Quatf(1, 0, 0, 0)], 1)
    q = Gf.Rotation(Gf.Vec3d(1, 0, 0), 60).GetQuat()
    rot.Set([Gf.Quatf(1, 0, 0, 0), Gf.Quatf(q.GetReal(), *q.GetImaginary())], 24)
    UsdSkel.BindingAPI.Apply(skel.GetPrim()).CreateAnimationSourceRel().SetTargets([anim.GetPath()])
    body = UsdGeom.Mesh.Define(st, '/World/Guy/Body')
    pts = [(-0.1, -0.1, 0), (0.1, -0.1, 0), (0.1, 0.1, 0), (-0.1, 0.1, 0), (-0.1, -0.1, 2), (0.1, -0.1, 2), (0.1, 0.1, 2), (-0.1, 0.1, 2)]
    body.CreatePointsAttr(pts)
    body.CreateFaceVertexCountsAttr([4] * 6)
    body.CreateFaceVertexIndicesAttr([0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7])
    b = UsdSkel.BindingAPI.Apply(body.GetPrim())
    b.CreateSkeletonRel().SetTargets([skel.GetPath()])
    b.CreateJointIndicesPrimvar(False, 1).Set([0, 0, 0, 0, 1, 1, 1, 1])
    b.CreateJointWeightsPrimvar(False, 1).Set([1.0] * 8)
    b.CreateGeomBindTransformAttr(Gf.Matrix4d(1))
    # lights: a low sun in the west, warm, and a sky dome
    sun = UsdLux.DistantLight.Define(st, '/World/Sun')
    sun.CreateIntensityAttr(50000.0)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.8, 0.6))
    # tilt the light's -z (its direction of travel) to come down from 30 degrees above the +x horizon
    UsdGeom.Xformable(sun).AddRotateYOp().Set(90.0 - 30.0)
    dome = UsdLux.DomeLight.Define(st, '/World/Sky')
    dome.CreateIntensityAttr(1.5)
    dome.CreateColorAttr(Gf.Vec3f(0.5, 0.6, 1.0))
    UsdLux.SphereLight.Define(st, '/World/Lamp')
    st.GetRootLayer().Save()



def test_usd_import_shapes_instances_skinning_and_lights(tmp_path):
    pytest.importorskip('pxr')
    from blackbody.io import usd as U
    path = tmp_path / 'shot.usda'
    _build_usd_scene(path)
    info = U.scan(path)
    kinds = {i.path: (i.kind, i.motion) for i in info.items}
    assert '/World/Hidden' not in kinds, 'invisible prims are skipped'
    assert kinds['/World/Rocks'] == ('mesh', 'world') and kinds['/World/Guy/Body'] == ('mesh', 'world')
    assert kinds['/World/ChairA/Seat'][0] == 'mesh', 'meshes inside instances are found'
    assert kinds['/World/Sun'][0] == 'light' and kinds['/World/Sky'][0] == 'light'
    sc = presets.make('campfire')
    report = U.import_usd(sc, path)
    col = {c['name']: c for c in sc.colliders}
    assert col['Crate']['shape'] == 'box' and col['Crate']['size'] == pytest.approx((1.0, 0.5, 0.5))
    assert col['Post']['shape'] == 'cylinder' and col['Post']['size'] == pytest.approx((0.1, 1.0, 0.1))
    assert col['Ball']['shape'] == 'sphere' and col['Pipe']['shape'] == 'mesh'
    assert col['Rocks']['mesh'].endswith('#/World/Rocks?world')
    v, t = U.load_usd_mesh(path, '/World/Rocks?world')
    assert len(t) == 36 and v[:, 0].min() == pytest.approx(0.9) and v[:, 0].max() == pytest.approx(2.1)
    top1 = U.load_usd_mesh(path, '/World/Guy/Body?world', 1)[0][4:].mean(0)
    top24 = U.load_usd_mesh(path, '/World/Guy/Body?world', 24)[0][4:].mean(0)
    assert top1 == pytest.approx([0.0, 2.0, 0.0], abs=1e-4)
    assert top24 == pytest.approx([0.0, 1.5, 0.866], abs=1e-3), 'the skinning is baked into the points'
    L = sc.data['lighting']
    assert L['sun_on'] and L['sun_azimuth'] == pytest.approx(90.0) and L['sun_elevation'] == pytest.approx(30.0)
    assert L['sun_intensity'] == pytest.approx(3.0) and L['ambient_intensity'] == pytest.approx(1.5)
    assert [l['name'] for l in sc.lights] == ['Lamp'], 'the sphere light becomes a light in the set'


def test_soot_grows_with_a_collider_whose_size_is_animated(engine):
    sc = _quiet(presets.make('campfire'), 48)
    sc.data['domain'].update(size_x=3.0, size_y=2.0, size_z=2.0)
    sc.data['combustion'].update(soot_stain=1.0, soot=2.0)
    sc.emitters = [_em(sc, shape='cylinder', position=(0.0, 0.05, 0.0), size=(0.2, 0.05, 0.2), fuel=10.0, temperature=0.6,
                       smoke=10.0, embers=False)]
    sc.colliders = [_col(sc, 0, shape='box', position=(0.0, 0.6, 0.0),
                         size=K((0.0, (0.3, 0.1, 0.3)), (2.0, (0.3, 0.1, 0.3)), (2.5, (0.6, 0.2, 0.6)), interp='linear'))]
    engine.invalidate()
    engine.prepare(sc)
    s = engine.solver
    engine.simulate_to(sc, sc.start + int(2.0 * sc.fps), cache=True)
    before = s.read_stain_obj()
    engine.simulate_to(sc, sc.start + int(2.6 * sc.fps), cache=True)
    after = s.read_stain_obj()
    assert before.sum() > 0.1 and after.shape == before.shape
    assert after.sum() >= 0.9 * before.sum(), 'the soot stays when the box grows'
    # the region is in the box's own size: its soot cells still line the grown box's surface
    slot = np.frombuffer(s.gpu.read_buffer(s.stain_slots), np.float32)[:12]
    lo, dims, cell = slot[0:3], slot[4:7], slot[8:11]
    assert np.all(lo < -1.0) and np.all(lo + dims * cell > 1.0), 'the region covers the whole box at any size'
    engine.render(sc, sc.start + int(2.6 * sc.fps), (240, 136), mode='composite')
    assert engine.aovs()['mask'][..., 3].astype(np.float32).max() > 0.05


def test_deep_bins_for_a_wisp_only_a_later_pass_sees(engine):
    """Pass 0 sees only a far puff; pass 1 also sees a near wisp: the wisp gets its own deep sample."""
    from blackbody.engine import camera as cam
    from blackbody.engine.engine import VolumeView
    sc = _plain(presets.make('campfire'))
    sc.data['domain'].update(resolution=48, preroll=0.0)
    sc.data['embers']['enabled'] = False
    engine.invalidate()
    engine.prepare(sc)
    s = engine.solver
    spec, fire = sc.camera(sc.start)
    W, H = 96, 64
    cs = cam.compute(spec, W / H, fire)
    w2g = cam.world_to_grid(fire, s.origin, s.h)
    fwd = -np.linalg.inv(cs.view)[:3, 2]
    target = np.array(fire.position) + np.array([0.0, 0.8, 0.0])
    dist = float(np.dot(target - cs.eye, fwd))
    nx, ny, nz = s.dims
    g = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij'), -1) + 0.5   # (x, y, z) cells

    def blob(centre_world, r):
        cg = (w2g @ np.append(centre_world, 1.0))[:3]
        return (np.linalg.norm(g - cg, axis=-1) < r / s.h).transpose(2, 1, 0)   # (z, y, x)

    far = blob(cs.eye + fwd * (dist + 0.5), 0.35)
    near = blob(cs.eye + fwd * (dist - 0.5), 0.25)
    vols = []
    for mask in (far, far | near):
        a = np.zeros((nz, ny, nx, 4), np.float16)
        a[..., 2] = mask * 3.0
        t = engine.gpu.texture3d(s.dims, 'rgba16float', 'test-scal')
        engine.gpu.upload(t, a)
        vols.append((t, VolumeView([t], [engine._zero_vel], s.dims, s.h, s.origin)))
    r = engine.renderer
    look = sc.look(sc.start)
    r.set_deep(8, 2)
    with engine.gpu.batch() as b:
        r.light(b, vols[1][1], look, fire, 0.0)
        for i, (_, v) in enumerate(vols):
            r.march(b, v, cs, fire, look, (W, H), deep_pass=i)
    d = r.read_deep()
    has = d[..., :8, 3] > 1e-3
    z = np.where(has, d[..., :8, 4], np.inf)
    zmin = z.min(-1)
    seen = np.isfinite(zmin)
    assert seen.sum() > 20
    near_depth = dist - 0.5 - 0.25
    far_front = dist + 0.5 - 0.35
    zf, zb = d[..., :8, 4], d[..., :8, 5]
    own = has & (np.abs(zf - near_depth) < 0.3) & (zb < far_front - 0.1)
    assert own.any(-1).sum() > 5, 'the wisp has deep samples of its own, in front of the far puff'
    behind = has & (zf > far_front - 0.1)
    assert (own.any(-1) & behind.any(-1)).sum() > 5, 'and the far puff keeps its own behind it'
    for t, _ in vols:
        t.destroy()


def test_lamps_light_the_smoke_and_the_smoke_shadows_them(engine):
    sc = _plain(presets.make('smoke_plume'))
    sc.data['domain'].update(resolution=48, preroll=1.0)
    sc.data['lighting'].update(sun_on=False, ambient_intensity=0.0, ambient_from_footage=False)
    engine.invalidate()
    engine.prepare(sc)
    f = sc.start + 30
    engine.simulate_to(sc, f, cache=False)
    out = {}
    for case in ('none', 'shadowed', 'unshadowed'):
        sc.lights = []
        if case != 'none':
            sc.add_light(kind='point', position=(1.5, 0.4, 0.0), intensity=20000.0, shadows=case == 'shadowed')
        engine.render(sc, f, (160, 160), mode='fire')
        out[case] = engine.aovs()['beauty'][..., :3].astype(np.float32).sum()
    assert out['shadowed'] > 1.5 * out['none'], out
    assert out['unshadowed'] > 1.1 * out['shadowed'], 'the smoke shades the light behind it'


# -- VDB files as other programs write them ----------------------------------------------------------------

def _lz4_literals(data):
    """A valid LZ4 block holding `data` as literals only."""
    n = len(data)
    head = bytearray([0xF0 if n >= 15 else n << 4])
    if n >= 15:
        rest = n - 15
        while rest >= 255:
            head.append(255)
            rest -= 255
        head.append(rest)
    return bytes(head) + bytes(data)


def _blosc(data, typesize):
    """A Blosc 1 buffer (byte-shuffled, LZ4, one block) of `data`; copied as is when small, as Blosc does."""
    import struct
    n = len(data)
    if n < 128:
        return struct.pack('<BBBBIII', 2, 1, 0x01 | 0x02 | (1 << 5), typesize, n, n, 16 + n) + bytes(data)
    a = np.frombuffer(data, np.uint8)
    k = n // typesize
    shuffled = a[:k * typesize].reshape(k, typesize).T.reshape(-1).tobytes() + a[k * typesize:].tobytes()
    nsplits = typesize if (k >= 128 and typesize <= 16) else 1
    split = n // nsplits
    body = b''
    for i in range(nsplits):
        z = _lz4_literals(shuffled[i * split:(i + 1) * split])
        body += struct.pack('<i', len(z)) + z
    head = struct.pack('<BBBBIII', 2, 1, 0x01 | (1 << 5), typesize, n, n, 16 + 4 + len(body))
    return head + struct.pack('<I', 20) + body


def _test_vdb(path, leaves, tiles=(), comp='zip', half=False, active_only=False, bg=0.0, cls='fog volume'):
    """A VDB with one float grid: `leaves` {origin: 512 values}, `tiles` [(origin, value)] of 8^3 voxels
    in the lowest internal nodes (they must share one with a leaf)."""
    import io
    import struct
    import uuid
    import zlib

    def s_(b, t):
        e = t.encode()
        b.write(struct.pack('<I', len(e)))
        b.write(e)

    def mask(bits, n):
        m = np.zeros(n, bool)
        m[list(bits)] = True
        return np.packbits(m, bitorder='little').tobytes()

    flags = {'none': 0, 'zip': 1, 'blosc': 4}[comp] | (2 if active_only else 0)

    def values(vals, active):
        vals = np.asarray(vals, np.float32)
        meta = 0 if (active_only and not active.all()) else 6
        out = struct.pack('<b', meta)
        if meta == 0:
            vals = vals[active]
        raw = vals.astype('<f2' if half else '<f4').tobytes()
        if comp == 'zip':
            z = zlib.compress(raw)
            return out + struct.pack('<q', len(z)) + z
        if comp == 'blosc':
            z = _blosc(raw, 2 if half else 4)
            return out + struct.pack('<q', len(z)) + z
        return out + raw

    body = io.BytesIO()
    body.write(struct.pack('<I', flags))
    metas = [('class', 'string', cls.encode())] + ([('is_saved_as_half_float', 'bool', b'\x01')] if half else [])
    body.write(struct.pack('<I', len(metas)))
    for name, kind, raw in metas:
        s_(body, name)
        s_(body, kind)
        body.write(struct.pack('<I', len(raw)))
        body.write(raw)
    s_(body, 'UniformScaleTranslateMap')
    body.write(struct.pack('<3d', 0.0, 0.0, 0.0))
    for x in (0.1, 0.1, 10.0, 100.0, 5.0):
        body.write(struct.pack('<3d', x, x, x))
    body.write(struct.pack('<i', 1))
    body.write(struct.pack('<f', bg))
    body.write(struct.pack('<II', 0, 1))
    body.write(struct.pack('<3i', 0, 0, 0))
    origins = sorted(leaves)

    def up_slot(o):
        return ((o[0] >> 7) << 10) | ((o[1] >> 7) << 5) | (o[2] >> 7)

    def low_slot(o):
        return (((o[0] & 127) >> 3) << 8) | (((o[1] & 127) >> 3) << 4) | ((o[2] & 127) >> 3)

    lows = sorted({up_slot(o) for o in origins})
    body.write(mask(lows, 32768))
    body.write(mask([], 32768))
    body.write(values(np.full(32768, bg), np.zeros(32768, bool)))
    order = []
    for up in lows:
        slots = sorted((low_slot(o), o) for o in origins if up_slot(o) == up)
        body.write(mask([sl for sl, _ in slots], 4096))
        tv = np.full(4096, bg, np.float32)
        ta = np.zeros(4096, bool)
        for o, v in tiles:
            if up_slot(o) == up:
                tv[low_slot(o)] = v
                ta[low_slot(o)] = True
        body.write(mask(np.nonzero(ta)[0], 4096))
        body.write(values(tv, ta))
        for _, o in slots:
            body.write(mask(np.nonzero(np.asarray(leaves[o]) != bg)[0], 512))
            order.append(o)
    for o in order:
        v = np.asarray(leaves[o], np.float32)
        act = v != bg
        body.write(mask(np.nonzero(act)[0], 512))
        body.write(values(v, act))
    blob = body.getvalue()
    with open(path, 'wb') as f:
        f.write(struct.pack('<q', 0x56444220))
        f.write(struct.pack('<III', 224, 10, 0))
        f.write(struct.pack('<b', 1))
        f.write(str(uuid.uuid4()).encode())
        f.write(struct.pack('<I', 0))
        f.write(struct.pack('<i', 1))
        head = io.BytesIO()
        s_(head, 'density')
        s_(head, 'Tree_float_5_4_3')
        s_(head, '')
        pos = f.tell() + len(head.getvalue()) + 24
        head.write(struct.pack('<3q', pos, pos, pos + len(blob)))
        f.write(head.getvalue())
        f.write(blob)


@pytest.mark.parametrize('comp,half,active_only', [('none', False, False), ('zip', False, True), ('blosc', False, False),
                                                   ('blosc', True, True), ('zip', True, False)])
def test_vdb_reader_reads_other_writers_files(tmp_path, comp, half, active_only):
    from blackbody.io.vdbread import read_float_grid
    rng = np.random.default_rng(3)
    a = np.where(rng.random(512) < 0.4, rng.random(512).astype(np.float32) + 0.5, 0.0).astype(np.float32)
    b = np.full(512, 2.0, np.float32)
    p = tmp_path / 'v.vdb'
    _test_vdb(p, {(8, 0, 0): a, (16, 8, 0): b}, tiles=[((0, 0, 8), 0.75)], comp=comp, half=half, active_only=active_only)
    dense, lo, xf = read_float_grid(p)
    assert tuple(lo) == (0, 0, 0) and dense.shape == (24, 16, 16) and xf[0, 0] == pytest.approx(0.1)
    tol = 2e-3 if half else 0.0
    # leaf values are stored x-major: index = x * 64 + y * 8 + z
    assert np.abs(dense[8:16, 0:8, 0:8] - a.reshape(8, 8, 8)).max() <= tol
    assert np.abs(dense[16:24, 8:16, 0:8] - 2.0).max() <= tol
    assert np.abs(dense[0:8, 0:8, 8:16] - 0.75).max() <= tol, 'a tile fills its 8^3 voxels'
    assert dense[0:8, 8:16, 0:8].max() == 0.0, 'the background elsewhere'


def test_usd_curves_points_volumes_nested_instancers_and_lights(tmp_path, engine):
    pytest.importorskip('pxr')
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdVol
    from blackbody.engine.mesh import bake
    from blackbody.io import usd as U
    vdb = tmp_path / 'puff.vdb'
    leaf = np.zeros((8, 8, 8), np.float32)
    leaf[2:6, 2:6, 2:6] = 1.0
    _test_vdb(vdb, {(0, 0, 0): leaf.reshape(-1)}, comp='zip')
    path = tmp_path / 'set.usda'
    st = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    cable = UsdGeom.BasisCurves.Define(st, '/Set/Cable')
    cable.CreateTypeAttr('linear')
    cable.CreatePointsAttr([(0, 1, 0), (1, 1.2, 0), (2, 1, 0)])
    cable.CreateCurveVertexCountsAttr([3])
    cable.CreateWidthsAttr([0.1])
    rope = UsdGeom.BasisCurves.Define(st, '/Set/Rope')
    rope.CreateTypeAttr('cubic')
    rope.CreateBasisAttr('catmullRom')
    rope.CreatePointsAttr([(0, 0, 1), (0, 0.5, 1), (0, 1, 1), (0, 1.5, 1)])
    rope.CreateCurveVertexCountsAttr([4])
    gravel = UsdGeom.Points.Define(st, '/Set/Gravel')
    gravel.CreatePointsAttr([(1, 0.05, 1), (1.2, 0.05, 1), (1.4, 0.05, 1)])
    gravel.CreateWidthsAttr([0.1, 0.1, 0.1])
    vol = UsdVol.Volume.Define(st, '/Set/Puff')
    field = UsdVol.OpenVDBAsset.Define(st, '/Set/Puff/density')
    field.CreateFilePathAttr(Sdf.AssetPath(str(vdb)))
    field.CreateFieldNameAttr('density')
    vol.CreateFieldRelationship('density', field.GetPath())
    UsdGeom.Xformable(vol).AddTranslateOp().Set(Gf.Vec3d(-2, 0, 0))
    # an instancer of instancers: 2 rows of 2 bricks
    inner = UsdGeom.PointInstancer.Define(st, '/Set/Rows/Protos/Row')
    cube = UsdGeom.Cube.Define(st, '/Set/Rows/Protos/Row/Protos/Brick')
    cube.CreateSizeAttr(0.2)
    inner.CreatePrototypesRel().SetTargets([cube.GetPath()])
    inner.CreateProtoIndicesAttr([0, 0])
    inner.CreatePositionsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5, 0, 0)])
    outer = UsdGeom.PointInstancer.Define(st, '/Set/Rows')
    outer.CreatePrototypesRel().SetTargets([inner.GetPath()])
    outer.CreateProtoIndicesAttr([0, 0])
    outer.CreatePositionsAttr([Gf.Vec3f(0, 0, -2), Gf.Vec3f(0, 0, -3)])
    lamp = UsdLux.RectLight.Define(st, '/Set/Window')
    lamp.CreateWidthAttr(2.0)
    lamp.CreateHeightAttr(1.0)
    lamp.CreateIntensityAttr(500.0)
    UsdGeom.Xformable(lamp).AddTranslateOp().Set(Gf.Vec3d(0, 2, 3))
    spot = UsdLux.DiskLight.Define(st, '/Set/Spot')
    spot.CreateRadiusAttr(0.1)
    spot.CreateIntensityAttr(100000.0)
    UsdLux.ShapingAPI.Apply(spot.GetPrim()).CreateShapingConeAngleAttr(20.0)
    UsdGeom.Xformable(spot).AddTranslateOp().Set(Gf.Vec3d(0, 4, 0))
    UsdGeom.Xformable(spot).AddRotateXOp().Set(-90.0)   # pointing straight down
    UsdLux.PortalLight.Define(st, '/Set/Portal')
    st.GetRootLayer().Save()

    info = U.scan(path)
    kinds = {i.path: i.prim_type for i in info.items}
    for p in ('/Set/Cable', '/Set/Rope', '/Set/Gravel', '/Set/Puff', '/Set/Rows'):
        assert p in kinds, p
    assert '/Set/Rows/Protos/Row/Protos/Brick' not in kinds, 'prototypes are drawn only through their instancer'
    v, t = U.load_usd_mesh(path, '/Set/Cable?world')
    assert v[:, 0].min() == pytest.approx(0.0, abs=0.06) and v[:, 0].max() == pytest.approx(2.0, abs=0.06)
    assert v[:, 1].max() == pytest.approx(1.25, abs=0.02), 'a tube as thick as the width'
    v, t = U.load_usd_mesh(path, '/Set/Rope?world')
    assert v[:, 1].min() > 0.4 and v[:, 1].max() < 1.1, 'a Catmull-Rom curve runs between its inner points'
    v, t = U.load_usd_mesh(path, '/Set/Gravel?world')
    assert len(t) == 24
    v, t = U.load_usd_mesh(path, '/Set/Rows?world')
    assert len(t) == 4 * 12, 'every brick of every row'
    assert sorted({round(float(z), 2) for z in v[:, 2]}) == [-3.1, -2.9, -2.1, -1.9]
    v, t = U.load_usd_mesh(path, '/Set/Puff?world')
    assert v.min(0) == pytest.approx([-2.0 + 0.15, 0.15, 0.15], abs=1e-6), 'the VDB voxels, placed by the prim'
    assert v.max(0) == pytest.approx([-2.0 + 0.55, 0.55, 0.55], abs=1e-6)
    sdf = bake(engine.gpu, v - v.mean(0), t, 24)
    d = sdf.data
    assert float(d[d.shape[0] // 2, d.shape[1] // 2, d.shape[2] // 2]) < 0, 'a closed solid'
    sc = presets.make('campfire')
    report = U.import_usd(sc, path)
    lights = {l['name']: l for l in sc.lights}
    assert lights['Window']['kind'] == 'area' and lights['Window']['intensity'] == pytest.approx(500.0 * 2.0)
    assert (lights['Window']['width'], lights['Window']['height']) == pytest.approx((2.0, 1.0))   # its panel
    assert lights['Spot']['kind'] == 'spot' and lights['Spot']['cone'] == pytest.approx(20.0)
    assert lights['Spot']['direction'] == pytest.approx((0.0, -1.0, 0.0), abs=1e-6)
    assert any('PortalLight' in r for r in report)


def _lamp_scene(engine, res=48):
    sc = _plain(presets.make('smoke_plume'))
    sc.data['domain'].update(resolution=res, preroll=1.0)
    sc.data['lighting'].update(sun_on=False, ambient_intensity=0.0, ambient_from_footage=False)
    engine.invalidate()
    engine.prepare(sc)
    f = sc.start + 30
    engine.simulate_to(sc, f, cache=False)
    return sc, f


def test_a_narrow_spot_beam_is_as_sharp_as_its_cone(engine):
    """The beam is worked out at every step of the march, so a spot narrower than a cell of the light
    volume still has its own width: a 4 degree beam is about twice as wide as a 2 degree one."""
    sc, f = _lamp_scene(engine)
    sc.data['embers']['enabled'] = False
    sc.lights = []
    engine.render(sc, f, (320, 320), mode='fire')
    dark = engine.aovs()['beauty'][..., :3].astype(np.float32).sum(-1)
    widths = {}
    for cone in (2.0, 4.0):
        sc.lights = []
        sc.add_light(kind='spot', position=(0.0, 3.5, 0.0), direction=(0.0, -1.0, 0.0), intensity=200000.0,
                     cone=cone, softness=0.0, shadows=False)
        engine.render(sc, f, (320, 320), mode='fire')
        lit = engine.aovs()['beauty'][..., :3].astype(np.float32).sum(-1) - dark
        prof = np.clip(lit, 0.0, None).sum(0)          # across the image: the beam is a vertical stripe
        x = np.arange(len(prof), dtype=np.float64)
        m = (prof * x).sum() / prof.sum()
        widths[cone] = float(np.sqrt((prof * (x - m) ** 2).sum() / prof.sum()))
        assert prof.sum() > 0.0
    ratio = widths[4.0] / widths[2.0]
    assert 1.6 < ratio < 2.5, widths


def test_lights_in_the_set_light_the_ground_and_smoke_shadows_real_ones(engine):
    sc, f = _lamp_scene(engine)
    sc.data['embers']['enabled'] = False
    sc.data['lighting'].update(ambient_intensity=0.3)
    plate = np.full((160, 160, 4), 150, np.uint8)
    probe = {}

    def comp(**kw):
        sc.lights = []
        if kw:
            sc.add_light(kind='spot', direction=(0.0, -1.0, 0.0), intensity=3000.0, cone=40.0, **kw)
        engine.render(sc, f, (160, 160), mode='composite', plate=plate)
        lp = engine.aovs()['lamps'][..., :3].astype(np.float32)
        return engine.linear_comp()[..., :3].astype(np.float32), lp

    base, lp0 = comp()
    assert np.abs(lp0).max() == 0.0
    # a light added in CG, off to the side of the smoke: it brightens the ground under it
    cg, lp = comp(position=(0.9, 1.5, 0.3), in_footage=False)
    assert lp.max() > 0.3 and lp.min() >= -1e-4
    pool = lp.max(-1) > 0.2
    assert (cg[pool] / base[pool]).mean() > 1.15, 'the footage is brighter in the pool of light'
    # a real lamp shining down through the plume: where the smoke shadows it, the footage darkens
    real, lp = comp(position=(0.0, 3.0, 0.0), in_footage=True, shadows=True)
    assert lp.min() < -0.05 and lp.max() <= 1e-4, 'only light taken away'
    shade = lp.min(-1) < -0.05
    assert (real[shade] / base[shade]).mean() < 0.97
    # and without smoke shadows a real lamp changes nothing
    same, lp = comp(position=(0.0, 3.0, 0.0), in_footage=True, shadows=False)
    assert np.abs(lp).max() < 1e-4
    clear = engine.aovs()['beauty'][..., 3].astype(np.float32) < 1e-3   # the footage, away from the lit smoke
    assert np.allclose(same[clear], base[clear], atol=1e-3)


# -- volumes as smoke --------------------------------------------------------------------------------------

def _ball_vdb(path, centre=(0.0, 1.4, 0.0), radius=0.45, voxel=0.05, heat=None, zup=False):
    """A VDB (our own writer) with a ball of smoke, and a temperature grid if `heat`."""
    from blackbody.io.vdb import write_vdb
    n = int(2 * radius / voxel) + 4
    c = np.asarray(centre, float)
    if zup:
        c = np.array([c[0], -c[2], c[1]])   # the same place, in a Z-up file
    org = c - (n - 1) / 2 * voxel
    idx = np.indices((n, n, n)).transpose(1, 2, 3, 0)
    r = np.linalg.norm(idx * voxel + org - c, axis=-1)
    d = np.clip(1.0 - r / radius, 0.0, None).astype(np.float32) * 3.0
    grids = {'density': d}
    if heat == 'lower':   # hot only below the ball's centre
        y = idx[..., 2 if zup else 1] * voxel + org[2 if zup else 1]
        grids['temperature'] = np.where(y < c[2 if zup else 1], d / 3.0 * 2.0, 0.0).astype(np.float32)
    elif heat:
        grids['temperature'] = (d / 3.0 * heat).astype(np.float32)
    write_vdb(path, grids, voxel, tuple(float(x) for x in org))
    return path


def _volume_scene(engine, vdb, res=40, **kw):
    sc = _quiet(presets.make('smoke_plume'), res)
    for e in sc.emitters:
        e['enabled'] = False
    args = dict(shape='volume', volume=str(vdb), position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), smoke=4.0,
                temperature=0.0, fuel=0.0, noise=0.0, start=0.25, volume_mode='fill', embers=False)
    args.update(kw)
    sc.add_emitter(**args)
    engine.invalidate()
    engine.prepare(sc)
    return sc


def _smoke_centre(engine):
    s = engine.solver
    soot = s.read_scalars()[..., 2].astype(np.float64)   # (z, y, x)
    total = soot.sum()
    if total <= 0:
        return total, None
    z, y, x = np.indices(soot.shape)
    ijk = np.array([(soot * x).sum(), (soot * y).sum(), (soot * z).sum()]) / total
    return total, np.asarray(s.origin) + (ijk + 0.5) * s.h


def test_a_volume_fills_the_box_with_its_smoke_where_it_is(engine, tmp_path):
    vdb = _ball_vdb(tmp_path / 'ball.vdb')
    sc = _volume_scene(engine, vdb)
    engine.simulate_to(sc, sc.start + 3, cache=False)
    before, _ = _smoke_centre(engine)
    assert before == 0.0, 'nothing before its start time'
    engine.simulate_to(sc, sc.start + 7, cache=False)
    total, c = _smoke_centre(engine)
    assert total > 0.0
    assert c == pytest.approx((0.0, 1.4, 0.0), abs=1.5 * engine.solver.h), 'the smoke is where the ball is'
    peak = float(engine.solver.read_scalars()[..., 2].max())
    assert 2.0 < peak < 4.5, 'about Smoke where the volume is densest'
    # filled once: without a source it only thins out from here
    engine.simulate_to(sc, sc.start + 20, cache=False)
    later, _ = _smoke_centre(engine)
    assert later <= total * 1.02


def test_volume_modes_hold_and_release(engine, tmp_path):
    vdb = _ball_vdb(tmp_path / 'ball.vdb')
    sums = {}
    for mode in ('fill', 'hold', 'source'):
        sc = _volume_scene(engine, vdb, volume_mode=mode, smoke=2.0, start=0.0)
        sc.data['motion']['wind_speed'] = 1.5       # wind carries the smoke off the ball
        engine.invalidate()
        engine.prepare(sc)
        engine.simulate_to(sc, sc.start + 30, cache=False)
        soot = engine.solver.read_scalars()[..., 2].astype(np.float64)
        sums[mode] = (soot.sum(), float(soot.max()))
    assert sums['hold'][1] > 1.2 * sums['fill'][1], ('topped up: the ball stays as thick as the volume', sums)
    assert sums['hold'][0] > sums['fill'][0]
    assert sums['source'][0] > sums['fill'][0], ('a source keeps releasing', sums)


def test_a_volumes_heat_follows_its_temperature_grid(engine, tmp_path):
    """With a temperature grid, the heat is where the file says; without one, the density shapes it."""
    plain = _ball_vdb(tmp_path / 'plain.vdb')
    lower = _ball_vdb(tmp_path / 'lower.vdb', heat='lower')
    gap = {}
    for name, vdb in (('plain', plain), ('lower', lower)):
        sc = _volume_scene(engine, vdb, temperature=1.0, start=0.0)
        engine.simulate_to(sc, sc.start, cache=False)
        s = engine.solver
        f = s.read_scalars().astype(np.float64)
        y = (np.indices(f.shape[:3])[1] + 0.5) * s.h + s.origin[1]
        heat, soot = np.maximum(f[..., 0], 0.0), f[..., 2]
        gap[name] = (heat * y).sum() / heat.sum() - (soot * y).sum() / soot.sum()
    assert abs(gap['plain']) < 0.05, gap
    assert gap['lower'] < -0.1, ('hot only in the lower half', gap)


def test_a_z_up_volume_is_turned_y_up(engine, tmp_path):
    from blackbody.io.volume import field_source, load_field
    a, _ = load_field(field_source(str(_ball_vdb(tmp_path / 'y.vdb'))), None, 64)
    b, _ = load_field(field_source(str(_ball_vdb(tmp_path / 'z.vdb', zup=True)), zup=True), None, 64)
    ca = (np.asarray(a.mesh_min) + np.asarray(a.mesh_max)) / 2
    cb = (np.asarray(b.mesh_min) + np.asarray(b.mesh_max)) / 2
    assert ca == pytest.approx((0.0, 1.4, 0.0), abs=0.06)
    assert cb == pytest.approx(ca, abs=0.06)


def test_usd_volumes_come_in_as_smoke(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdVol
    from blackbody.io import usd as U
    vdb = _ball_vdb(tmp_path / 'puff.vdb', centre=(0.0, 0.5, 0.0))
    path = tmp_path / 'set.usda'
    st = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    vol = UsdVol.Volume.Define(st, '/Set/Puff')
    field = UsdVol.OpenVDBAsset.Define(st, '/Set/Puff/density')
    field.CreateFilePathAttr(Sdf.AssetPath(str(vdb)))
    field.CreateFieldNameAttr('density')
    vol.CreateFieldRelationship('density', field.GetPath())
    UsdGeom.Xformable(vol).AddTranslateOp().Set(Gf.Vec3d(1.0, 0.0, 0.0))
    st.GetRootLayer().Save()
    sc = presets.make('campfire')
    n0 = len(sc.emitters)
    report = U.import_usd(sc, path)
    assert len(sc.emitters) == n0 + 1 and any('smoke' in r for r in report), report
    e = sc.emitters[-1]
    assert e['shape'] == 'volume' and e['volume_mode'] == 'fill' and e['volume'].endswith('#/Set/Puff')
    assert tuple(e['position']) == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)
    from blackbody.io.volume import load_field
    g, _ = load_field(sc.item_source(e), None, 64)
    c = (np.asarray(g.mesh_min) + np.asarray(g.mesh_max)) / 2
    assert c == pytest.approx((0.0, 0.5, 0.0), abs=0.06), 'in the prim\'s own frame; the emitter places it'
    # as solid objects instead
    sc2 = presets.make('campfire')
    U.import_usd(sc2, path, volumes='solid')
    assert sc2.colliders[-1]['shape'] == 'mesh' and len(sc2.emitters) == n0


# -- open sides: inflow can't feed itself ---------------------------------------------------------------

def _mean_u(solver):
    """Mean x velocity of the x faces (m/s): the whole box, its x = 0 side, its middle."""
    u = solver.gpu.read(solver.vel[0]).astype(np.float32)[:-1, :-1, :, 0]
    return u.mean(), u[..., 0].mean(), u[..., solver.dims[0] // 2].mean()


def test_air_only_blows_in_through_open_sides_as_hard_as_the_air_outside(engine):
    """A through-flow once coasted forever: air traced back out of an open side brought the edge's own
    inflow in again. Now still air outside lets it die down, a wind holds it to the wind, and a wind
    spinning the box up from rest does so as evenly as before."""
    def through_flow(wind, u0):
        sc = _quiet(presets.make('campfire'), res=48)
        sc.data['motion'].update(damping=0.0, wind_relax=0.0, wind_speed=wind, wind_dir=90.0, gust=0.0)
        engine.invalidate()
        engine.prepare(sc, final=False)
        engine.simulate_to(sc, sc.start, cache=False)
        s = engine.solver
        nx, ny, nz = s.dims
        v = np.zeros((nz + 1, ny + 1, nx + 1, 4), np.float32 if s.gpu.vel_format == 'rgba32float' else np.float16)
        v[:nz, :ny, :, 0] = u0
        s.gpu.upload(s.vel[0], v)
        engine.simulate_to(sc, sc.start + 48, cache=False)
        return _mean_u(s)

    box, inflow, _ = through_flow(0.0, 3.0)
    assert box < 1.2 and inflow < 0.6, 'still air outside: nothing keeps it blowing'
    box, inflow, _ = through_flow(1.0, 3.0)
    assert 1.0 < box < 1.8 and inflow > 0.9, 'a 1 m/s wind still blows in'

    sc = _quiet(presets.make('campfire'), res=48)
    sc.data['motion'].update(wind_speed=2.0, wind_dir=90.0, gust=0.0)
    engine.invalidate()
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 24, cache=False)
    box, inflow, mid = _mean_u(engine.solver)
    assert 0.5 < box < 0.9 and inflow == pytest.approx(mid, abs=1e-3), 'the whole box spins up together'


def test_a_small_fuel_rich_fire_on_a_fine_grid_stays_stable(engine):
    """The reported runaway: campfire logs shrunk to 0.6x in a 1.4 x 2.4 x 1.4 m box with twice the fuel,
    80 x 144 x 80 cells (1.7 cm). Cold air coming in through the open top and sides fed itself into a
    gale through the whole box (45 m/s by frame 140, 100 by 160); a campfire's gas moves at about 6."""
    sc = _plain(presets.make('campfire', fps=30))
    sc.data['domain'].update(size_x=1.4, size_y=2.4, size_z=1.4, resolution=144, preview_scale=1.0, preroll=0.0)
    sc.data['combustion']['fuel_scale'] = 2.0
    sc.data['embers']['enabled'] = False
    a, b, c = sc.emitters
    a.update(position=(-0.25, 0.05, -0.11), end=(0.24, 0.05, 0.12), size=(0.06, 0.06, 0.06))
    b.update(position=(-0.21, 0.06, 0.15), end=(0.23, 0.06, -0.13), size=(0.06, 0.06, 0.06))
    c.update(size=(0.2, 0.03, 0.2))
    engine.invalidate()
    engine.prepare(sc, final=False)
    assert engine.solver.dims == (80, 144, 80)
    for f in range(20, 161, 20):
        engine.simulate_to(sc, sc.start + f, cache=False)
        assert np.isfinite(engine.solver.max_speed) and engine.solver.max_speed < 25.0, (f, engine.solver.max_speed)
    assert engine.solver.burning > 0
