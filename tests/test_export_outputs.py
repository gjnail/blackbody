"""Render outputs that used to be written wrong or not at all: VDBs from the disk cache (the cached frame, not
the solver's start), VDBs of a sky (its water, not the fire solver's fields), deep EXRs in shots with layers
(every fire and liquid layer's samples) and in scenes without any (refused), fabric meshes from the Render
window, and outputs a scene has nothing for (refused before anything renders)."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from blackbody.io.vdb import dense_from_leaves, read_vdb, write_vdb_frame
from blackbody.render import job as J
from blackbody.render.job import Output, RenderJob
from blackbody.scene import presets


def _x(a):
    """(z, y, x, ...) to the VDB writer's x-major order."""
    return np.transpose(a, (2, 1, 0) + tuple(range(3, a.ndim)))


def _dense(grid, shape):
    vec = 'vec3' in grid['type']
    return dense_from_leaves(grid['leaves'], shape, vec)


# -- planning (GPU-free) -------------------------------------------------------------------------------------------

def test_outputs_a_scene_has_nothing_for_are_refused():
    deep, mesh = Output('deep', 'd.deep.####.exr'), Output('mesh', 'm.usdc')
    fab, liq = Output('mesh', 'f.usdc', 'fabric'), Output('mesh', 'l.usdc', 'liquid')
    fire, flag, pool = presets.make('campfire'), presets.make('flag_wind'), presets.make('towel_dip')
    both, sky = presets.make('hose_on_fire'), presets.make('cumulus_day')
    assert J.check_outputs(fire, [deep]) == [] and J.check_outputs(pool, [deep, mesh, fab, liq]) == []
    assert J.check_outputs(flag, [fab, mesh]) == []          # a fire scene's mesh is its fabric
    for sc in (both, sky):
        msg, = J.check_outputs(sc, [deep])
        assert 'fire scenes and liquid scenes' in msg and J.KIND_NAMES[sc.kind] in msg
    assert len(J.check_outputs(fire, [mesh, fab, liq])) == 3
    assert J.check_outputs(both, [liq]) == [] and 'no fabric' in J.check_outputs(both, [fab])[0]
    assert 'samples per pixel' in J.check_outputs(fire, [Output('deep', 'd.exr', deep_samples=32)])[0]


def test_deep_in_a_shot_with_layers():
    shot = presets.make('campfire')
    water, sky = presets.make('towel_dip'), presets.make('cumulus_day')
    water.uid, sky.uid, sky.name = 'water', 'sky', 'Afternoon sky'
    shot.layers = [water, sky]
    assert [u for u, _ in J.deep_layers(shot)] == ['base', 'water']
    deep = Output('deep', 'd.deep.####.exr')
    assert J.check_outputs(shot, [deep]) == []
    note, = J.output_notes(shot, [deep])
    assert 'Afternoon sky is not in it' in note
    # a fire-and-liquid base with a fire layer: the layer's samples
    both = presets.make('hose_on_fire')
    layer = presets.make('torch')
    layer.uid = 'torch'
    both.layers = [layer]
    assert [u for u, _ in J.deep_layers(both)] == ['torch'] and J.check_outputs(both, [deep]) == []
    # a sky with a fire-and-liquid layer: nothing to make one of
    sky2 = presets.make('cumulus_day')
    sky2.layers = [presets.make('hose_on_fire')]
    sky2.layers[0].uid = 'hose'
    msg, = J.check_outputs(sky2, [deep])
    assert 'layers' in msg


def test_output_labels_and_cli_refusals(tmp_path, capsys):
    from blackbody.cli import main
    assert Output('mesh', 'f.usdc', 'fabric').label() == 'Fabric · USD'
    assert Output('mesh', 'l.####.obj', 'liquid').label() == 'Liquid surface · OBJ sequence'
    assert '16 samples' in Output('deep', 'd.exr', deep_samples=16).label()
    # refused before the GPU starts, with exit code 2 (bad arguments)
    assert main(['render', '--preset', 'hose_on_fire', '-o', str(tmp_path / 'x.deep.####.exr')]) == 2
    assert 'fire-and-liquid scene' in capsys.readouterr().err
    assert main(['render', '--preset', 'campfire', '-o', str(tmp_path / 'm.usdc')]) == 2
    assert 'no fabric' in capsys.readouterr().err
    assert main(['render', '--preset', 'cumulus_day', '-o', str(tmp_path / 'c.####.obj')]) == 2


# -- VDBs from the cache and of a sky (GPU-free, with stand-ins for the engine) ----------------------------------------

def _fire_entry(dims, upres=1, vel=True, seed=1):
    rng = np.random.default_rng(seed)
    nx, ny, nz = dims
    k = upres
    scal = rng.uniform(0.0, 2.0, (nz * k, ny * k, nx * k, 4)).astype(np.float16)
    scal[: nz * k // 2] = 0.0                      # empty space in front
    e = {'scal': scal, 'time': 1.5, 'upres': upres, 'dims': list(dims), 'origin': [-0.3, 0.0, -0.2], 'h': 0.05}
    if vel:
        e['vel'] = rng.uniform(-1.0, 1.0, (nz + 1, ny + 1, nx + 1, 4)).astype(np.float16)
    return e


def test_a_cached_fire_frame_writes_its_own_vdb(tmp_path):
    sc = presets.make('campfire')
    f = sc.start + 3
    # the solver as a render from the disk cache leaves it: prepared at its start, never stepped
    solver = SimpleNamespace(dims=(4, 4, 4), h=0.1, origin=(0.0, 0.0, 0.0), features={})
    entry = _fire_entry((16, 24, 12))
    eng = SimpleNamespace(sim_frame=None, solver=solver, cache={f: entry})
    job = RenderJob(sc, [Output('vdb', str(tmp_path / 'v.####.vdb'))], eng, frames=(f, f))
    src = job._fire_fields(f)
    assert isinstance(src, J.CachedFire) and src.dims == (16, 24, 12)
    p = tmp_path / 'c.vdb'
    write_vdb_frame(p, src, sc, f)
    g = read_vdb(p)['grids']
    shape = (16, 24, 12)
    dens = _dense(g['density'], shape)
    want = _x(entry['scal'][..., 2].astype(np.float32))
    assert np.allclose(dens, np.where(want > 1e-4, want, 0.0))
    assert np.allclose(_dense(g["temperature"], shape), _x(entry["scal"][..., 0].astype(np.float32)), atol=2e-4)
    v = entry['vel'].astype(np.float32)
    ux = 0.5 * (v[:-1, :-1, :-1, 0] + v[:-1, :-1, 1:, 0])
    assert np.allclose(_dense(g["vel"], shape)[..., 0], _x(ux), atol=2e-4)
    assert g['density']['xform']['voxel'] == pytest.approx(0.05)
    # the live solver when it holds the frame; a frame neither live nor cached is an error, not a stale file
    eng.sim_frame = f
    assert job._fire_fields(f) is solver
    with pytest.raises(RuntimeError, match='neither simulated nor cached'):
        job._fire_fields(f + 1)


def test_a_cached_upres_frame_without_velocities(tmp_path, caplog):
    sc = presets.make('campfire')
    f = sc.start
    solver = SimpleNamespace(dims=(4, 4, 4), h=0.1, origin=(0.0, 0.0, 0.0), features={})
    entry = _fire_entry((8, 12, 8), upres=2, vel=False)
    eng = SimpleNamespace(sim_frame=None, solver=solver, cache={f: entry})
    job = RenderJob(sc, [], eng, frames=(f, f))
    with caplog.at_level('WARNING', logger='blackbody.job'):
        src = job._fire_fields(f)
        job._fire_fields(f)
    assert sum('no velocities' in r.message for r in caplog.records) == 1     # said once a job
    p = tmp_path / 'u.vdb'
    write_vdb_frame(p, src, sc, f)
    g = read_vdb(p)['grids']
    assert g['density']['xform']['voxel'] == pytest.approx(0.025)             # at the finer grid's size
    assert np.allclose(_dense(g['density'], (16, 24, 16)), np.where(_x(entry['scal'][..., 2].astype(np.float32)) > 1e-4,
                                                                    _x(entry['scal'][..., 2].astype(np.float32)), 0.0))
    assert not g['vel']['leaves'], 'no velocity in the cache: an empty vel grid'


def _sky_fields(dims=(10, 6, 8)):
    nx, ny, nz = dims
    a = np.zeros((nz, ny, nx, 4), np.float32)
    b = np.zeros((nz, ny, nx, 4), np.float32)
    a[..., 0] = 0.5                         # warm air and vapour everywhere: not water
    a[..., 1] = 8e-3
    a[2:5, 3:5, 3:7, 2] = 1.0e-3            # a cloud: water, with ice at its top
    b[2:5, 5, 3:7, 0] = 0.5e-3
    a[3, 0:3, 4, 3] = 2.0e-3                # rain falling out of it
    vel = np.zeros((nz + 1, ny + 1, nx + 1, 4), np.float32)
    vel[..., 0] = 10.0                      # a 10 m/s wind along x
    return a, b, vel


def test_sky_vdb_grids():
    a, b, vel = _sky_fields()
    rho = np.linspace(1.2, 0.7, 6)
    g = J.cloud_vdb_grids(a, b, rho, vel, speed=0.5, rot=np.eye(3))
    assert set(g) == {'density', 'cloud_water', 'cloud_ice', 'rain', 'vel'}
    assert g['density'][4, 3, 2] == pytest.approx(1.0e-3 * rho[3] * 1000.0)              # g/m^3
    assert g['density'][4, 5, 2] == pytest.approx(0.5e-3 * rho[5] * 1000.0)
    assert g['density'][4, 1, 3] == 0.0 and g['rain'][4, 1, 3] > 0.0                    # rain is not cloud
    assert g['vel'][4, 3, 2, 0] == pytest.approx(5.0) and g['vel'][0, 0, 0, 0] == 0.0  # only where there is water
    turned = J.cloud_vdb_grids(a, b, rho, vel, rot=np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
    assert turned['vel'][4, 3, 2, 2] == pytest.approx(-10.0)


def test_sky_vdb_live_and_from_the_cache(tmp_path):
    sc = presets.make('cumulus_day')
    f = sc.start + 2
    a, b, vel = _sky_fields()
    scale = float(sc.data['atmosphere']['scale'])
    C = SimpleNamespace(dims=(10, 6, 8), h=0.25 * scale, h_scene=0.25, origin_scene=(-1.25, 0.0, -1.0), scale=scale,
                        V=[vel], read_fields=lambda: (a, b))
    entry = {'cloud_a': a[..., 2:4].astype(np.float16), 'cloud_b': b.astype(np.float16)}
    eng = SimpleNamespace(cloud=C, sim_frame=f, gpu=SimpleNamespace(read=lambda t: t), cache={f: entry})
    J.write_cloud_vdb_frame(tmp_path / 'live.vdb', eng, sc, f)
    eng.sim_frame = None
    J.write_cloud_vdb_frame(tmp_path / 'cached.vdb', eng, sc, f)
    live, cached = read_vdb(tmp_path / 'live.vdb'), read_vdb(tmp_path / 'cached.vdb')
    assert 'vel' in live['grids'] and 'vel' not in cached['grids']        # the cache keeps only the water
    shape = (10, 6, 8)
    d0, d1 = _dense(live['grids']['density'], shape), _dense(cached['grids']['density'], shape)
    assert d0.max() > 0.1 and np.allclose(d0, d1, rtol=2e-3, atol=1e-4)
    x = live['grids']['density']['xform']
    spec, fire = sc.camera(f)
    if 'voxel' in x:
        assert x['voxel'] == pytest.approx(0.25)                          # the scene's metres, not the sky's
        assert np.allclose(x['translation'], np.array([-1.125, 0.125, -0.875]) + np.asarray(fire.position))
    else:
        assert np.linalg.norm(x['matrix'][:3, 0]) == pytest.approx(0.25)
    lapse = float(sc.v('atmosphere', 'time_lapse', f)) * float(sc.v('domain', 'time_scale', f))
    v = _dense(live['grids']['vel'], shape)
    assert np.abs(v).max() == pytest.approx(10.0 * lapse / scale, rel=1e-4)
    assert float(np.frombuffer(live['meta']['blackbody_sky_metres_per_unit'], '<f4')[0]) == pytest.approx(scale)
    eng.cache = {}
    with pytest.raises(RuntimeError, match='neither simulated nor cached'):
        J.write_cloud_vdb_frame(tmp_path / 'none.vdb', eng, sc, f)


# -- the Render window (GPU-free) -----------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def qapp():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    QtWidgets = pytest.importorskip('PySide6.QtWidgets')
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _dialog(scene):
    from blackbody.ui.document import Document
    from blackbody.ui.export_dialog import ExportDialog
    d = Document()
    d.scene = scene
    return ExportDialog(d)


def test_render_window_offers_what_each_scene_has(qapp, tmp_path):
    dlg = _dialog(presets.make('flag_wind'))
    assert dlg.fabric.isEnabled() and not dlg.mesh.isEnabled()
    for w in (dlg.exr, dlg.png, dlg.mov, dlg.comp, dlg.comp_exr, dlg.vdb, dlg.deep):
        w.setChecked(False)
    dlg.folder.setText(str(tmp_path))
    dlg.fabric.setChecked(True)
    dlg.deep.setChecked(True)
    dlg.deep_n.setCurrentIndex(dlg.deep_n.findData(16))
    outs = {o.kind: o for o in dlg.outputs()}
    assert outs['mesh'].content == 'fabric' and outs['mesh'].path.endswith('_fabric.usdc')
    assert outs['deep'].deep_samples == 16
    assert 'embers are in them' in dlg.deep.toolTip() and 'not included' not in dlg.deep.toolTip()
    assert J.check_outputs(presets.make('flag_wind'), list(outs.values())) == []

    dlg = _dialog(presets.make('campfire'))
    assert not dlg.fabric.isEnabled() and dlg.deep.isEnabled()
    dlg.fabric.setChecked(True)
    assert all(o.kind != 'mesh' for o in dlg.outputs())

    dlg = _dialog(presets.make('hose_on_fire'))
    assert not dlg.deep.isEnabled() and 'not fire-and-liquid' in dlg.deep.toolTip()
    dlg.deep.setChecked(True)
    assert all(o.kind != 'deep' for o in dlg.outputs())

    dlg = _dialog(presets.make('cumulus_day'))
    assert 'cloud_water' in dlg.vdb.toolTip() and not dlg.deep.isEnabled()

    shot = presets.make('campfire')
    sky = presets.make('cumulus_day')
    sky.uid, sky.name = 'sky', 'Sky'
    shot.layers = [sky]
    dlg = _dialog(shot)
    assert dlg.deep.isEnabled() and 'layers' in dlg.deep.toolTip()
    dlg.deep.setChecked(True)
    assert 'Sky is not in it' in dlg.preview_paths.text()


# -- GPU ------------------------------------------------------------------------------------------------------------

def test_vdb_from_the_disk_cache_matches_the_live_one(engine, tmp_path):
    from blackbody.engine.engine import Engine
    sc = presets.make('campfire')
    sc.data['domain'].update(resolution=32, preroll=0.0, disk_cache=True, cache_dir=str(tmp_path / 'cache'))
    sc.data['render'].update(motion_blur=True, upres=1)
    f = sc.start + 4
    engine.invalidate()
    engine.prepare(sc)
    try:
        engine.simulate_to(sc, f)
        engine.cache.disk.flush()
        write_vdb_frame(tmp_path / 'live.vdb', engine.solver, sc, f)
        dims = tuple(engine.solver.dims)
        other = Engine(engine.gpu)
        try:
            out = RenderJob(sc, [Output('vdb', str(tmp_path / 'farm.####.vdb'))], other, frames=(f, f), final=False,
                            from_cache=True).run()
        finally:
            other.cache.attach(None)
    finally:
        engine.cache.attach(None)
        engine.invalidate()
    farm = read_vdb(J.frame_path(str(tmp_path / 'farm.####.vdb'), f))['grids']
    live = read_vdb(tmp_path / 'live.vdb')['grids']
    assert str(J.frame_path(str(tmp_path / 'farm.####.vdb'), f)) in out
    for name in ('density', 'temperature', 'flame', 'fuel'):
        a, b = _dense(live[name], dims), _dense(farm[name], dims)
        assert a.max() > 0.0 and np.array_equal(a, b), name
    va, vb = _dense(live['vel'], dims), _dense(farm['vel'], dims)
    assert np.abs(va).max() > 0.05 and np.abs(va - vb).max() < 0.01 * np.abs(va).max() + 1e-3


def test_sky_vdb_from_a_fresh_engine(engine, tmp_path):
    # a sky's VDB holds the sky, also in an engine whose fire solver was never set up
    from blackbody.engine.engine import Engine
    sc = presets.make('cumulus_day')
    sc.data['domain'].update(resolution=24, preroll=0.0)
    f = sc.start + 1
    fresh = Engine(engine.gpu)
    out = RenderJob(sc, [Output('vdb', str(tmp_path / 'sky.####.vdb'))], fresh, frames=(f, f), final=False).run()
    g = read_vdb(out[0])['grids']
    assert {'density', 'vel'} <= set(g) and 'temperature' not in g and 'flame' not in g
    x = g['density']['xform']
    h = fresh.cloud.h_scene
    assert (x['voxel'] if 'voxel' in x else np.linalg.norm(x['matrix'][:3, 0])) == pytest.approx(h)
    assert g['vel']['leaves'] or not g['density']['leaves']


def test_deep_exr_of_a_shot_with_layers(engine, tmp_path):
    # a shot with layers used to write no deep EXR at all: now it holds every fire layer's samples
    import OpenEXR
    shot = presets.make('campfire')
    shot.data['domain'].update(resolution=32, preroll=0.5)
    shot.data['embers']['enabled'] = False
    layer = presets.make('torch')
    layer.uid = 'torch'
    layer.data['domain'].update(resolution=32, preroll=0.5)
    f = shot.start + 3

    def samples(layers, name):
        shot.layers = layers
        shot.sync_layers()
        p = str(tmp_path / f'{name}.deep.####.exr')
        out = RenderJob(shot, [Output('deep', p, deep_samples=16)], engine, frames=(f, f), final=False, size=(96, 54),
                        samples=1).run()
        q = J.frame_path(p, f)
        assert q in out and os.path.exists(q), name
        with OpenEXR.File(q) as fh:
            assert fh.header()['type'] == OpenEXR.deepscanline
            return sum(0 if z is None else len(z) for z in fh.channels()['Z'].pixels.ravel())

    try:
        alone = samples([], 'alone')
        both = samples([layer], 'layers')
    finally:
        engine.invalidate()
    assert alone > 0 and both > alone
