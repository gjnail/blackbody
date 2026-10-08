"""Exports that used to be missing: the shot's camera (a USD camera, a Nuke .chan, the EXRs' headers), and the shot as a
USD scene (the objects as they move and break, the ropes, the matter, the grass, the embers), read back with pxr; and
shots with layers, whose VDBs and meshes cover every layer (each beside the base layer's) instead of the base alone."""
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

from blackbody.engine import camera as cam
from blackbody.io import camera_out as CO
from blackbody.io import scene_usd as SU
from blackbody.render import job as J
from blackbody.render.job import Output, RenderJob
from blackbody.scene import presets
from blackbody.scene.params import collider_defaults

pxr = pytest.importorskip('pxr')
from pxr import Usd, UsdGeom  # noqa: E402

SIZE = (640, 360)


def _usd_project(stage, path, t, pts, size=SIZE):
    """World points through a USD camera (pxr's own camera maths) to pixels."""
    gc = UsdGeom.Camera(stage.GetPrimAtPath(path)).GetCamera(Usd.TimeCode(t))
    fr = gc.frustum
    h = np.c_[pts, np.ones(len(pts))] @ np.array(fr.ComputeViewMatrix()) @ np.array(fr.ComputeProjectionMatrix())
    ndc = h[:, :2] / h[:, 3:4]
    return np.stack([(ndc[:, 0] + 1) * 0.5 * size[0], (1 - ndc[:, 1]) * 0.5 * size[1]], 1)


def _points_near(sc, f, n=60, seed=0):
    spec, fire = sc.camera(f)
    return np.asarray(fire.position) + np.random.default_rng(seed).uniform(-1.5, 1.5, (n, 3)) + [0.0, 1.2, 0.0]


def _world(prim, t):
    """A prim's own frame to the world at t (column vectors)."""
    return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode(t))).T


# -- the camera (GPU-free) ----------------------------------------------------------------------------------------------

def _placed(name, **cam_keys):
    sc = presets.make(name)
    sc.data['render']['width'], sc.data['render']['height'] = SIZE
    sc.data['camera'].update(cam_keys)
    return sc


CAMERAS = [('campfire', {}), ('campfire', {'roll': 14.0, 'scale': 1.4, 'anchor_x': 0.3}), ('towel_dip', {}),
           ('campfire', {'mode': 'free', 'use_anchor': False, 'position': (1.0, 1.8, 5.0), 'rotation': (-8.0, 12.0, 3.0),
                         'focal_mm': 50.0}),
           ('campfire', {'mode': 'free', 'use_anchor': True, 'position': (-2.0, 1.2, 4.0), 'rotation': (5.0, -20.0, 0.0),
                         'roll': -30.0, 'scale': 0.7, 'anchor_y': 0.6})]


@pytest.mark.parametrize('name,keys', CAMERAS)
def test_the_usd_camera_sees_the_picture_blackbody_renders(tmp_path, name, keys):
    sc = _placed(name, **keys)
    f = sc.start + 3
    cf = CO.camera_at(sc, f, SIZE)
    w = SU.SceneWriter(tmp_path / 'cam.usda', sc.fps)
    w.add(f, None, camera=cf)
    w.close()
    st = Usd.Stage.Open(str(tmp_path / 'cam.usda'))
    pts = _points_near(sc, f)
    spec, fire = sc.camera(f)
    want, ok = cam.project(cam.compute(spec, SIZE[0] / SIZE[1], fire), pts, *SIZE)
    got = _usd_project(st, '/World/Camera', f, pts)
    assert ok.sum() > 20 and np.abs(got - want)[ok].max() < 1e-3      # (pixels: the same picture, anchor and all)
    rs = st.GetPrimAtPath('/Render/Settings')
    assert tuple(rs.GetAttribute('resolution').Get()) == SIZE
    assert UsdGeom.GetStageUpAxis(st) == 'Y' and UsdGeom.GetStageMetersPerUnit(st) == 1.0


def test_the_usd_camera_comes_back_through_the_usd_import(tmp_path):
    # a camera without a placement (nothing a free camera cannot hold) imported again is the same camera
    from blackbody.io.usd import camera_keys
    sc = _placed('campfire', mode='free', use_anchor=False, position=(1.0, 1.8, 5.0), rotation=(-8.0, 12.0, 3.0),
                 focal_mm=40.0)
    w = SU.SceneWriter(tmp_path / 'cam.usda', sc.fps)
    for f in (1, 2):
        sc.data['camera']['focal_mm'] = 40.0 + f
        w.add(f, None, camera=CO.camera_at(sc, f, SIZE))
    w.close()
    pos, rot, focal, sensor, near_far = camera_keys(str(tmp_path / 'cam.usda'), '/World/Camera', [1, 2])
    assert np.allclose(pos[0][1], (1.0, 1.8, 5.0)) and np.allclose(rot[0][1], (-8.0, 12.0, 3.0), atol=1e-6)
    assert focal[1][1] == pytest.approx(42.0) and sensor == pytest.approx(sc.data['camera']['sensor_mm'])


def test_chan_round_trip_through_the_chan_import(tmp_path):
    from blackbody.io.chan import chan_keys, read_chan
    sc = _placed('campfire', mode='free', use_anchor=False, position=(1.0, 1.8, 5.0), rotation=(-8.0, 12.0, 3.0),
                 focal_mm=50.0)
    frames = []
    for f in range(1, 6):
        sc.data['camera']['rotation'] = (-8.0, 12.0 + 50.0 * f, 3.0)     # (turning right round: no jump at 180)
        frames.append((f, CO.camera_at(sc, f, SIZE)))
    p = CO.write_chan(tmp_path / 'cam.chan', frames)
    rows = read_chan(p)
    assert len(rows) == 5 and all(len(r) == 8 for r in rows)
    assert max(abs(b[5] - a[5]) for a, b in zip(rows, rows[1:])) < 90.0     # ry keeps on, no wrap
    pos, rot, focal = chan_keys(p, sensor_mm=sc.data['camera']['sensor_mm'], aspect=SIZE[0] / SIZE[1])
    for (f, cf), pk, rk, fk in zip(frames, pos, rot, focal):
        assert np.allclose(pk[1], cf.to_world[:3, 3], atol=1e-5)
        assert np.allclose(cam.euler_xyz(*rk[1]), cf.to_world[:3, :3], atol=1e-5)
        assert fk[1] == pytest.approx(50.0, rel=1e-5)


def test_chan_holds_the_placements_turn_and_zoom_and_says_what_it_cannot(tmp_path):
    sc = _placed('campfire', roll=20.0, scale=1.5)
    f = sc.start
    cf = CO.camera_at(sc, f, SIZE)
    # the turn is the camera's roll, the zoom its focal length: the picture differs from Blackbody's only by a slide
    spec, fire = sc.camera(f)
    pts = _points_near(sc, f)
    want, ok = cam.project(cam.compute(spec, SIZE[0] / SIZE[1], fire), pts, *SIZE)
    # a plain lens at the .chan's field of view, from the turned camera
    P = cam.perspective(math.radians(cf.vfov), SIZE[0] / SIZE[1], cf.near, cf.far)
    h = np.c_[pts, np.ones(len(pts))] @ (P @ cf.view).T
    got = np.stack([(h[:, 0] / h[:, 3] + 1) * 0.5 * SIZE[0], (1 - h[:, 1] / h[:, 3]) * 0.5 * SIZE[1]], 1)
    slide = np.array(cf.offset_px)
    assert np.abs(slide).max() > 1.0
    assert np.abs(got + slide - want)[ok].max() < 1e-3
    assert cf.focal_mm == pytest.approx(spec.focal_mm * 1.5)
    o = Output('camera', str(tmp_path / 'c.chan'), 'camera')
    note, = [n for n in J.output_notes(sc, [o]) if 'off the lens' in n]
    assert 'px' in note and 'window translate' in note
    # a camera without a placement: nothing to say
    sc2 = _placed('campfire', mode='free', use_anchor=False)
    assert not [n for n in J.output_notes(sc2, [o]) if 'off the lens' in n]


def test_exr_headers_carry_the_camera():
    sc = _placed('campfire', roll=10.0)
    f = sc.start + 2
    attrs = CO.exr_attrs(CO.camera_at(sc, f, SIZE))
    assert attrs['worldToNDC'].dtype == np.float32 and attrs['worldToNDC'].shape == (4, 4)
    pts = _points_near(sc, f)
    spec, fire = sc.camera(f)
    want, ok = cam.project(cam.compute(spec, SIZE[0] / SIZE[1], fire), pts, *SIZE)
    h = np.c_[pts, np.ones(len(pts))] @ attrs['worldToNDC'].astype(float)     # (Imath: row vectors)
    got = np.stack([(h[:, 0] / h[:, 3] + 1) * 0.5 * SIZE[0], (1 - h[:, 1] / h[:, 3]) * 0.5 * SIZE[1]], 1)
    assert np.abs(got - want)[ok].max() < 0.05
    # worldToCamera: the camera at the origin looking down -z
    eye = np.r_[CO.camera_at(sc, f, SIZE).to_world[:3, 3], 1.0] @ attrs['worldToCamera'].astype(float)
    assert np.allclose(eye[:3], 0.0, atol=1e-4)


def test_element_and_composite_exrs_are_written_with_the_camera(tmp_path):
    import OpenEXR
    sc = _placed('campfire')
    job = RenderJob(sc, [], SimpleNamespace(), frames=(1, 1), size=(8, 4))
    job._cam_attrs = CO.exr_attrs(CO.camera_at(sc, 1, (8, 4)))
    z = np.zeros((4, 8, 4), np.float16)
    aov = {'beauty': z, 'emission': z, 'aux': z}
    job._write_element(Output('exr', str(tmp_path / 'e.####.exr'), layers=()), 1, aov, None, None, {}, None)
    job._write_comp(Output('exr', str(tmp_path / 'c.####.exr'), 'composite'), 1, z.astype(np.float32), {}, None)
    for name in ('e.0001.exr', 'c.0001.exr'):
        with OpenEXR.File(str(tmp_path / name)) as fh:
            h = fh.header()
            assert np.allclose(h['worldToNDC'], job._cam_attrs['worldToNDC']) and 'worldToCamera' in h


def test_deep_exrs_are_written_with_the_camera(tmp_path):
    import OpenEXR
    sc = _placed('campfire', roll=10.0)
    job = RenderJob(sc, [], SimpleNamespace(), frames=(1, 1), size=(8, 4))
    job._cam_attrs = CO.exr_attrs(CO.camera_at(sc, 1, (8, 4)))
    s = np.zeros((4, 8, 2, 8), np.float32)
    s[..., 0, 3], s[..., 0, 4], s[..., 0, 5] = 0.5, 2.0, 2.5
    job._write_deep(Output('deep', str(tmp_path / 'd.deep.####.exr')), 1, s)
    with OpenEXR.File(str(tmp_path / 'd.deep.0001.exr')) as fh:
        h = fh.header()
        assert h['type'] == OpenEXR.deepscanline
        assert np.allclose(h['worldToNDC'], job._cam_attrs['worldToNDC'])
        assert np.allclose(h['worldToCamera'], job._cam_attrs['worldToCamera'])


def _no_render_engine():
    """An engine for jobs that need no picture: anything rendering fails the test."""
    def render(*a, **k):
        raise AssertionError('rendered a picture no output needs')
    return SimpleNamespace(cache=SimpleNamespace(disk=None), gpu=SimpleNamespace(name='none'), sim_frame=None,
                           prepare=lambda *a, **k: None, simulate_to=lambda *a, **k: None, render=render)


def test_scene_and_camera_outputs_made_without_a_content_render_no_picture(tmp_path):
    # Output's content defaults to 'element': a scene or camera output made so (from Python) rendered an element for
    # nothing, and a camera-only job (nothing prepared) then rendered on an engine that was never prepared
    sc = _placed('campfire')
    sc.colliders = []
    chan, usd = str(tmp_path / 'c.chan'), str(tmp_path / 's.usda')
    written = RenderJob(sc, [Output('camera', chan)], _no_render_engine(), frames=(1, 2), size=SIZE).run()
    assert chan in written
    written = RenderJob(sc, [Output('scene', usd)], _no_render_engine(), frames=(1, 2), size=SIZE).run()
    st = Usd.Stage.Open(usd)
    assert usd in written and st.GetPrimAtPath('/World/Camera').GetTypeName() == 'Camera'


def test_a_slide_between_the_frames_looked_at_is_said_after_the_render(tmp_path):
    # the note before the render looks at the first, middle and last frames; an Anchor keyed between them slides the
    # picture further, and the .chan's writer says so from every frame it wrote
    import re
    sc = _placed('campfire')
    for f, x in ((sc.start, 0.5), (sc.start + 5, 0.2), (sc.start + 10, 0.5)):
        sc.set_key(('camera', 'anchor_x'), f, x)
    o = Output('camera', str(tmp_path / 'c.chan'), 'camera')
    px = lambda n: float(re.search(r'up to (\d+) px', n).group(1))
    before, = [n for n in J.output_notes(sc, [o]) if 'off the lens' in n]
    job = RenderJob(sc, [o], _no_render_engine(), frames=(sc.start, sc.start + 10), size=SIZE)
    job.run()
    after, = [n for n in job.notes if 'off the lens' in n]
    assert px(after) > px(before) + 50 and 'window translate' in after
    # nothing more to say when the frames looked at had the largest slide
    sc2 = _placed('campfire')
    job = RenderJob(sc2, [o], _no_render_engine(), frames=(1, 4), size=SIZE)
    job.run()
    assert not job.notes


def test_the_footages_lens_distortion_is_said_and_kept_on_the_usd_camera(tmp_path):
    # the element is bent by Composite › Lens distortion, which no camera holds: said before the render, and kept on
    # the USD camera as blackbody:lens_k1
    sc = _placed('campfire')
    sc.data['composite']['lens_k1'] = -0.1
    usd, chan = str(tmp_path / 'c.usda'), str(tmp_path / 'c.chan')
    outs = [Output('scene', usd, 'camera'), Output('camera', chan, 'camera')]
    notes = [n for n in J.output_notes(sc, outs) if 'Lens distortion' in n]
    assert len(notes) == 2 and 'blackbody:lens_k1' in notes[0] and '-0.100' in notes[1]
    RenderJob(sc, outs, _no_render_engine(), frames=(1, 1), size=SIZE).run()
    st = Usd.Stage.Open(usd)
    assert st.GetPrimAtPath('/World/Camera').GetAttribute('blackbody:lens_k1').Get() == pytest.approx(-0.1)
    sc.data['composite']['lens_k1'] = 0.0
    assert not [n for n in J.output_notes(sc, outs) if 'Lens distortion' in n]


# -- planning (GPU-free) --------------------------------------------------------------------------------------------------

def test_outputs_from_paths():
    assert J.infer_output('cam/shot.chan').kind == 'camera'
    o = J.infer_output('usd/shot.scene.usdc')
    assert (o.kind, o.content) == ('scene', 'scene')
    o = J.infer_output('usd/shot.camera.usda')
    assert (o.kind, o.content) == ('scene', 'camera')
    assert J.infer_output('usd/x.usdc', 'scene').kind == 'scene' and J.infer_output('usd/x.usdc').kind == 'mesh'
    # --content meant for the USD scene does not turn an EXR on the same command line into something else
    assert J.infer_output('r/f.####.exr', 'scene').kind == 'exr' and J.infer_output('r/f.####.exr', 'scene').content == 'element'
    assert J.infer_output('r/f.####.exr', 'deep').kind == 'deep'
    assert Output('scene', 'a.usdc', 'camera').label() == 'Camera · USD' and 'chan' in Output('camera', 'a.chan').label()


def test_layer_files_and_prims_are_named_for_their_layers():
    assert J.tagged('v/fire.####.vdb', '.sky') == 'v/fire.sky.####.vdb'.replace('/', os.sep) or \
        J.tagged('v/fire.####.vdb', '.sky').replace('\\', '/') == 'v/fire.sky.####.vdb'
    t = lambda *a: J.tagged(*a).replace('\\', '/')
    assert t('m/pond.usdc', '.water') == 'm/pond.water.usdc'
    assert t('m/pond.%04d.obj', '.fabric') == 'm/pond.fabric.%04d.obj'
    assert t('fire.vdb', '.liquid') == 'fire.liquid.vdb'
    assert J.frame_path(t('v/fire.####.vdb', '.liquid'), 7) == 'v/fire.liquid.0007.vdb'
    assert t('a.####.vdb', '') == 'a.####.vdb'
    shot = presets.make('campfire')
    a, b, c = presets.make('torch'), presets.make('torch'), presets.make('cumulus_day')
    a.uid, b.uid, c.uid = 't1', 't2', 'sky'
    a.name = b.name = 'Torch'
    c.name = 'Afternoon sky!'
    shot.layers = [a, b, c]
    order = shot.layer_order()
    tags = J.layer_tags(order)
    assert tags['base'] == '' and tags['sky'] == '.afternoon_sky' and len({tags['t1'], tags['t2']}) == 2
    roots = J.layer_roots(order)
    assert roots['base'] == '/World' and roots['sky'] == '/World/Afternoon_sky' and len(set(roots.values())) == 4


def test_a_layer_named_liquid_or_fabric_keeps_off_the_base_layers_files():
    # a fire-and-liquid base writes its liquid to name.liquid.####.vdb, a liquid base mesh its fabric to name.fabric.*:
    # layers named so must not be given those tags, or one layer's files would overwrite the other's
    shot = presets.make('campfire')
    a, b, c = presets.make('towel_dip'), presets.make('towel_dip'), presets.make('torch')
    a.uid, a.name, b.uid, b.name, c.uid, c.name = 'l', 'Liquid', 'f', 'Fabric', 'd', 'Deep'
    shot.layers = [a, b, c]
    tags = J.layer_tags(shot.layer_order())
    assert not {tags['l'], tags['f'], tags['d']} & {'', '.liquid', '.fabric', '.deep'}
    assert len(set(tags.values())) == 4 and tags['l'].startswith('.liquid_')


def test_a_fire_and_liquid_vdb_and_the_fabric_beside_a_liquid_mesh_are_named_as_sequences(tmp_path, monkeypatch):
    # the frame number last: name.liquid.0001.vdb and name.fabric.0001.obj (they used to be name.0001.liquid.vdb and
    # name.0001.fabric.obj, which a sequence reader does not see as one sequence), and a layer's beside them
    from blackbody.io import fabric_mesh as FM
    from blackbody.io import liquid_mesh as LM
    from blackbody.io import vdb as V
    got = []
    monkeypatch.setattr(V, 'write_vdb_frame', lambda p, *a: got.append(os.path.basename(p)))
    monkeypatch.setattr(V, 'write_liquid_vdb_frame', lambda p, *a: got.append(os.path.basename(p)))
    monkeypatch.setattr(LM, 'liquid_surface_mesh', lambda *a: 'surface')
    monkeypatch.setattr(LM, 'write_obj', lambda p, m: got.append(os.path.basename(p)))
    monkeypatch.setattr(FM, 'fabric_meshes', lambda *a: 'cloth')
    monkeypatch.setattr(FM, 'write_obj', lambda p, m: got.append(os.path.basename(p)))
    both = presets.make('campfire')
    both.data['domain']['kind'] = 'both'
    job = RenderJob(both, [], SimpleNamespace(), frames=(1, 1))
    eng = SimpleNamespace(sim_frame=1, solver=None)
    vdb = Output('vdb', str(tmp_path / 'fire.####.vdb'))
    for tag in ('', '.pool'):
        job._write_vdb(vdb, 1, eng, both, tag)
    assert got == ['fire.0001.vdb', 'fire.liquid.0001.vdb', 'fire.pool.0001.vdb', 'fire.pool.liquid.0001.vdb']
    got.clear()
    pond = presets.make('towel_dip')
    assert pond.kind == 'liquid' and J.has_fabric(pond)
    mesh = Output('mesh', str(tmp_path / 'pond.####.obj'))
    for uid, tag in (('base', ''), ('pool', '.pool')):
        job._write_meshes([mesh], 1, eng, pond, uid, tag, {})
    assert got == ['pond.0001.obj', 'pond.fabric.0001.obj', 'pond.pool.0001.obj', 'pond.pool.fabric.0001.obj']


def test_meshes_and_vdbs_of_a_shot_with_layers_are_planned_for_every_layer():
    shot = presets.make('campfire')
    water = presets.make('towel_dip')
    water.uid, water.name = 'water', 'Pool'
    shot.layers = [water]
    liq = Output('mesh', 'm/x_liquid.usdc', 'liquid')
    vdb = Output('vdb', 'v/x.####.vdb')
    # the fire base has no liquid, its layer has: the mesh output is the layer's, beside the given path
    assert J.check_outputs(shot, [liq, vdb]) == []
    notes = J.output_notes(shot, [liq, vdb])
    assert any('Pool: x_liquid.pool.usdc' in n for n in notes)
    assert any('Pool: x.pool.####.vdb' in n and "base layer's" in n for n in notes)
    # a sky with neither: refused, pointing to the USD scene for what it has
    sky = presets.make('cumulus_day')
    msg, = J.check_outputs(sky, [Output('mesh', 'm.usdc')])
    assert 'scene.usdc' in msg


def test_scene_notes_say_what_the_usd_scene_leaves_out():
    sc = presets.make('campfire')
    d = collider_defaults()
    d.update(name='Room', hollow=0.1)
    sc.colliders.append(d)
    o = Output('scene', 's.usdc', 'scene')
    notes = J.output_notes(sc, [o])
    assert any('Room is hollow' in n for n in notes)
    assert not any('bullets' in n for n in notes)
    assert any('bullets' in n for n in J.output_notes(presets.make('machine_gun'), [o]))
    assert any('liquid and the fabric' in n for n in J.output_notes(presets.make('towel_dip'), [o]))
    assert J.output_notes(sc, [Output('scene', 'c.usda', 'camera')]) == []
    assert J.scene_contents(presets.make('jelly_ball'))[:2] == ['camera', 'objects'] and 'matter' in J.scene_contents(
        presets.make('jelly_ball'))
    assert 'grass' in J.scene_contents(presets.make('meadow_fire')) and 'embers' in J.scene_contents(presets.make('campfire'))
    assert 'ropes' in J.scene_contents(presets.make('chain_swing')) and 'pieces' in J.scene_contents(presets.make('vase_drop'))


def test_a_layer_with_its_own_placement_has_its_own_camera():
    shot = presets.make('campfire')
    same, moved = presets.make('torch'), presets.make('torch')
    same.uid, moved.uid, moved.name = 'same', 'moved', 'Left torch'
    shot.layers = [same, moved]
    shot.sync_layers()
    same.data['camera'] = dict(shot.data['camera'])
    moved.data['camera'] = dict(shot.data['camera'], anchor_x=0.2)
    assert [u for u, _ in J.own_cameras(shot)] == ['moved']
    note, = [n for n in J.output_notes(shot, [Output('scene', 's.usdc', 'scene')]) if 'camera' in n]
    assert 'Left torch' in note and '/World/<layer>/Camera' in note


# -- the scene (GPU-free, with stand-ins for the engine) -------------------------------------------------------------

def _collider(**kw):
    d = collider_defaults()
    d.update(kw)
    return d


def _objects_scene():
    sc = presets.make('campfire')
    sc.data['camera']['fire_position'] = (2.0, 0.0, -1.0)    # (the simulation's frame is not the world's)
    sc.data['camera']['fire_yaw'] = 30.0
    sc.colliders = [_collider(name='Table', shape='box', position=(0.0, 0.4, 0.0), size=(0.8, 0.4, 0.5), yaw=20.0),
                    _collider(name='Ball', shape='sphere', position=(1.0, 2.0, 0.0), size=(0.3, 0.3, 0.3), dynamic=True),
                    _collider(name='Vase', shape='box', position=(-1.0, 1.0, 0.0), size=(0.2, 0.3, 0.2), breakable=True,
                              pieces=6, holdout=False),
                    _collider(name='Log', shape='mesh', mesh='builtin:firewood.obj', position=(0.0, 0.1, 1.0),
                              size=(1.0, 2.0, 1.0)),
                    _collider(name='Pipe', shape='cylinder', position=(0.0, 1.0, -1.0), size=(0.2, 0.5, 0.2), pitch=90.0)]
    return sc


def _stand_in(sc):
    """An engine as the scene writer reads one: the ball falls and the vase breaks at frame 3."""
    from blackbody.engine.solids import Solids, fractured
    from blackbody.scene.model import turn_quat
    c = sc.colliders[2]
    frac = fractured(c, c['size'], 0.0, None, sc)
    vase_q = turn_quat(0.0, math.radians(40.0), 0.0)
    R = SU.quat_matrices([vase_q])[0]

    def over(f):
        o = {1: dict(pos=(1.0, 2.0 - 0.2 * f, 0.0), vel=(0.0, -4.8, 0.0), rot_y=0.0, spin=0.0,
                     quat=turn_quat(0.1 * f, 0.0, 0.0), omega=(0.0, 0.0, 0.0, 0.0))}
        o[2] = Solids.gone() if f >= 3 else dict(pos=(-1.0, 1.0, 0.0), vel=(0.0, 0.0, 0.0), rot_y=0.0, spin=0.0,
                                                 quat=vase_q, omega=(0.0, 0.0, 0.0, 0.0))
        return o

    def pieces(f):
        if f < 3:
            return None
        cen = np.array([p.centroid for p in frac.pieces])
        pos = (cen @ R.T) + np.array([-1.0, 1.0, 0.0])
        pos[0] += 0.5 * (f - 3)                       # (the first piece flies off)
        if f >= 4:
            pos[1, 1] = -1.0e4                        # (and the second has burnt to ash)
        return {2: dict(pos=pos, quat=np.tile(vase_q, (len(cen), 1)), vel=np.zeros_like(pos), omega=np.zeros_like(pos),
                        size=np.asarray(c['size'], float), hollow=np.float32(0.0))}

    ropes = lambda f: {4: dict(a=np.array([0.0, 2.0, 0.0]), b=np.array([0.0, 3.0, 0.0]), va=np.zeros(3), vb=np.zeros(3),
                               length=np.float32(1.0), radius=np.float32(0.01), look=np.float32(0.0), broken=np.float32(0.0))}
    return SimpleNamespace(floating_overrides=over, piece_poses=pieces, rope_poses=ropes, sim_frame=None, cache={}), frac, R


def test_objects_pieces_and_ropes_in_the_usd_scene(tmp_path):
    sc = _objects_scene()
    eng, frac, R = _stand_in(sc)
    w = SU.SceneWriter(tmp_path / 'scene.usdc', sc.fps)
    for f in (1, 2, 3, 4):
        w.add(f, SU.frame_data(eng, sc, f), camera=CO.camera_at(sc, f, SIZE))
    w.close()
    st = Usd.Stage.Open(str(tmp_path / 'scene.usdc'))
    F = SU.fire_matrix(sc, 1)
    # the table: keyed where it is, turned, sized; a unit cube scaled to its half sizes
    shape = st.GetPrimAtPath('/World/Objects/Table/Shape')
    assert shape.GetTypeName() == 'Cube' and UsdGeom.Cube(shape).GetSizeAttr().Get() == 2.0
    M = _world(shape, 1)
    corner = M @ np.array([1.0, 1.0, 1.0, 1.0])
    want = F @ np.r_[np.array([0.0, 0.4, 0.0]) + cam.rot_y(math.radians(20.0)) @ np.array([0.8, 0.4, 0.5]), 1.0]
    assert np.allclose(corner, want, atol=1e-5)
    # the ball falls: where the simulation has it, frame by frame
    for f in (1, 4):
        c = _world(st.GetPrimAtPath('/World/Objects/Ball/Shape'), f) @ np.array([0.0, 0.0, 0.0, 1.0])
        assert np.allclose(c[:3], (F @ np.array([1.0, 2.0 - 0.2 * f, 0.0, 1.0]))[:3], atol=1e-5)
    # a tipped cylinder lies along the turned axis
    tip = _world(st.GetPrimAtPath('/World/Objects/Pipe/Shape'), 1) @ np.array([0.0, 1.0, 0.0, 1.0])
    from blackbody.scene.model import turn_matrix
    assert np.allclose(tip[:3], (F @ np.r_[np.array([0.0, 1.0, -1.0]) + turn_matrix(0.0, 90.0, 0.0) @ [0.0, 0.5, 0.0], 1.0])[:3],
                       atol=1e-5)
    # the log's mesh as modelled, its Size as the scale
    log = UsdGeom.Mesh(st.GetPrimAtPath('/World/Objects/Log/Shape'))
    assert len(log.GetPointsAttr().Get()) > 8 and np.allclose(_world(log.GetPrim(), 1)[:3, :3] @ [0.0, 1.0, 0.0],
                                                               F[:3, :3] @ [0.0, 2.0, 0.0], atol=1e-5)
    # the vase: whole until frame 3, then hidden; a helper (it hides nothing) is a guide
    vase = UsdGeom.Imageable(st.GetPrimAtPath('/World/Objects/Vase'))
    assert [vase.ComputeVisibility(Usd.TimeCode(f)) for f in (1, 2, 3)] == ['inherited', 'inherited', 'invisible']
    assert UsdGeom.Imageable(st.GetPrimAtPath('/World/Objects/Vase/Shape')).ComputePurpose() == 'guide'
    # its pieces: hidden before, then where they are; at rest they make up the vase exactly
    p1 = st.GetPrimAtPath('/World/Pieces/Vase/Piece_0002')
    assert len(st.GetPrimAtPath('/World/Pieces/Vase').GetChildren()) == len(frac.pieces)
    img = UsdGeom.Imageable(p1)
    assert [img.ComputeVisibility(Usd.TimeCode(f)) for f in (1, 2, 3, 4)] == ['invisible', 'invisible', 'inherited', 'invisible']
    p0 = st.GetPrimAtPath('/World/Pieces/Vase/Piece_0001')
    assert [UsdGeom.Imageable(p0).ComputeVisibility(Usd.TimeCode(f)) for f in (2, 3)] == ['invisible', 'inherited']
    allpts = []
    for k, prim in enumerate(st.GetPrimAtPath('/World/Pieces/Vase').GetChildren()):
        pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
        world = (_world(prim, 3) @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
        allpts.append(world)
        if k == 2:
            cut = UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Mesh(prim))
            assert cut and cut[0].GetPrim().GetName() == 'cut'
    local = (np.linalg.inv(F) @ np.c_[np.concatenate(allpts), np.ones(sum(map(len, allpts)))].T).T[:, :3]
    inside = (local - [-1.0, 1.0, 0.0]) @ R          # (the vase's own frame)
    assert np.allclose(np.abs(inside).max(0), [0.2, 0.3, 0.2], atol=1e-4)
    # the rope, between its ends
    rope = UsdGeom.BasisCurves(st.GetPrimAtPath('/World/Ropes/Pipe'))
    pts = np.asarray(rope.GetPointsAttr().Get(Usd.TimeCode(2)))
    ends = [(F @ [0.0, y, 0.0, 1.0])[:3] for y in (2.0, 3.0)]
    assert len(pts) >= 2 and {tuple(np.round(pts[0], 4)), tuple(np.round(pts[-1], 4))} == {tuple(np.round(e, 4)) for e in ends}
    assert st.GetStartTimeCode() == 1 and st.GetEndTimeCode() == 4 and st.GetTimeCodesPerSecond() == sc.fps


def test_a_broken_object_and_a_burnt_piece_stay_where_they_were_last_seen(tmp_path):
    # the 'gone' pose is 10 km below: written as a sample, USD slid the vase there, still shown, through the subframes
    # before it broke (and a piece burnt to ash the same way), a smear down out of frame in motion blur
    sc = _objects_scene()
    eng, _frac, _R = _stand_in(sc)
    w = SU.SceneWriter(tmp_path / 'scene.usdc', sc.fps)
    for f in (1, 2, 3, 4):
        w.add(f, SU.frame_data(eng, sc, f))
    w.close()
    st = Usd.Stage.Open(str(tmp_path / 'scene.usdc'))
    vase = st.GetPrimAtPath('/World/Objects/Vase')
    for t in (2.25, 2.5, 2.75):
        assert UsdGeom.Imageable(vase).ComputeVisibility(Usd.TimeCode(t)) == 'inherited'
        assert np.allclose(_world(vase, t), _world(vase, 2), atol=1e-6)
    piece = st.GetPrimAtPath('/World/Pieces/Vase/Piece_0002')        # (burnt to ash at frame 4)
    assert UsdGeom.Imageable(piece).ComputeVisibility(Usd.TimeCode(3.5)) == 'inherited'
    assert np.allclose(_world(piece, 3.5), _world(piece, 3), atol=1e-6)
    # the one flying off still moves between frames
    first = st.GetPrimAtPath('/World/Pieces/Vase/Piece_0001')
    assert np.allclose(_world(first, 3.5), 0.5 * (_world(first, 3) + _world(first, 4)), atol=1e-6)


def test_spinning_pieces_keep_their_shape_between_frames(tmp_path):
    # a matrix's entries are interpolated one by one: a piece turning a third of a turn a frame shrank to half its size
    # half way between frames. A translate and an orient (slerped) keep it rigid, turning the short way round
    from blackbody.engine.solids import fractured
    from blackbody.scene.model import turn_quat
    sc = _objects_scene()
    c = sc.colliders[2]
    frac = fractured(c, c['size'], 0.0, None, sc)
    n = len(frac.pieces)

    def pieces(f):
        q = turn_quat(math.radians(120.0 * f), math.radians(30.0), 0.0)
        return {2: dict(pos=np.tile([-1.0, 1.0, 0.0], (n, 1)), quat=np.tile(q, (n, 1)), vel=np.zeros((n, 3)),
                        omega=np.zeros((n, 3)), size=np.asarray(c['size'], float), hollow=np.float32(0.0))}

    eng = SimpleNamespace(floating_overrides=lambda f: None, piece_poses=pieces, sim_frame=None, cache={})
    w = SU.SceneWriter(tmp_path / 'spin.usdc', sc.fps)
    for f in (1, 2, 3, 4):
        w.add(f, SU.frame_data(eng, sc, f))
    w.close()
    st = Usd.Stage.Open(str(tmp_path / 'spin.usdc'))
    prim = st.GetPrimAtPath('/World/Pieces/Vase/Piece_0001')
    F = SU.fire_matrix(sc, 1)
    for t in (1.5, 2.5, 3.5):
        R = _world(prim, t)[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)                 # (no shrinking, no shearing)
        want = F[:3, :3] @ SU.quat_matrices([turn_quat(math.radians(120.0 * t), math.radians(30.0), 0.0)])[0]
        assert np.allclose(R, want, atol=1e-6)                             # (half way round, the short way)


def test_a_hollow_object_is_said_once(tmp_path):
    # the note before the render says it; the writer said it again after the render (and logged it as a warning)
    sc = _objects_scene()
    sc.colliders[0]['hollow'] = 0.05
    notes = J.output_notes(sc, [Output('scene', 's.usdc', 'scene')])
    assert len([n for n in notes if 'hollow' in n]) == 1
    eng, _frac, _R = _stand_in(sc)
    w = SU.SceneWriter(tmp_path / 'h.usda', sc.fps)
    w.add(1, SU.frame_data(eng, sc, 1))
    w.close()
    assert w.notes == []


def test_each_layer_goes_under_its_own_prim(tmp_path):
    shot = _objects_scene()
    torch = presets.make('torch')
    torch.uid, torch.name = 'torch', 'Torch'
    torch.colliders = [_collider(name='Barrel', position=(0.5, 0.5, 0.0))]
    torch.data['camera']['fire_position'] = (-3.0, 0.0, 0.0)
    shot.layers = [torch]
    order = shot.layer_order()
    roots = J.layer_roots(order)
    eng, _frac, _R = _stand_in(shot)
    other = SimpleNamespace(sim_frame=None, cache={})       # (nothing falls in the torch's layer)
    w = SU.SceneWriter(tmp_path / 's.usda', shot.fps)
    for f in (1, 2):
        for uid, lay in order:
            w.add(f, SU.frame_data(eng if uid == 'base' else other, lay, f), roots[uid],
                  camera=CO.camera_at(lay, f, SIZE))
    w.close()
    st = Usd.Stage.Open(str(tmp_path / 's.usda'))
    assert st.GetPrimAtPath('/World/Torch').GetTypeName() == 'Xform'
    assert st.GetPrimAtPath('/World/Objects/Table') and not st.GetPrimAtPath('/World/Torch/Objects/Table')
    barrel = _world(st.GetPrimAtPath('/World/Torch/Objects/Barrel/Shape'), 1) @ np.array([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(barrel[:3], (SU.fire_matrix(torch, 1) @ [0.5, 0.5, 0.0, 1.0])[:3], atol=1e-5)
    assert st.GetPrimAtPath('/World/Torch/Camera').GetTypeName() == 'Camera'


def _matter_snap(pos, slots, dims, origin, dx, rnd=0.5):
    q = np.clip(np.round((np.asarray(pos) - origin) / dx / np.asarray(dims) * 65535.0), 0, 65535).astype(np.uint16)
    tag = (np.asarray(slots, np.uint32) * 4096 + int(rnd * 4095)).astype(np.uint16)
    return np.c_[q, tag].astype(np.uint16)


def test_matter_points_through_value_clips(tmp_path):
    from blackbody.engine.matter import material
    sc = presets.make('campfire')
    sc.data['camera']['fire_position'] = (0.0, 0.0, 0.0)
    sc.data['camera']['fire_yaw'] = 0.0
    origin, dx, dims = np.array([-1.0, 0.0, -1.0]), 0.05, (40, 30, 40)
    mats = [material('sand'), material('snow')]
    m = SimpleNamespace(active=True, origin=origin, dx=dx, dims=dims, _mats=mats, _colours=[(None, 1.0), ((0.9, 0.1, 0.1), 1.0)])
    start = np.array([[0.0, 0.5, 0.0], [0.2, 0.6, 0.1], [0.3, 0.2, -0.2], [0.1, 0.1, 0.1]])
    v = np.array([0.0, -1.2, 0.0])
    cache = {}
    for f in (2, 3, 4):
        pos = start + v * (f - 2) / sc.fps
        slots = [0, 1, 15, 0]                    # (the third has gone)
        cache[f] = {'matter': _matter_snap(pos, slots, dims, origin, dx)}
    cache[3]['matter'] = np.r_[cache[3]['matter'], _matter_snap([[0.5, 1.0, 0.5]], [1], dims, origin, dx)]   # (poured)
    cache[4]['matter'] = np.r_[cache[4]['matter'], _matter_snap([[0.5, 1.0, 0.5]], [1], dims, origin, dx)]
    eng = SimpleNamespace(_matter=m, sim_frame=None, cache=cache)
    w = SU.SceneWriter(tmp_path / 'm.usdc', sc.fps)
    for f in (1, 2, 3, 4):                       # (frame 1: no matter yet)
        w.add(f, SU.frame_data(eng, sc, f))
    w.close()
    assert all(os.path.exists(p) for p in w.extra) and len(w.extra) >= 4
    st = Usd.Stage.Open(str(tmp_path / 'm.usdc'))
    g = UsdGeom.Points(st.GetPrimAtPath('/World/Matter'))
    assert len(g.GetPointsAttr().Get(Usd.TimeCode(1))) == 0
    p3 = np.asarray(g.GetPointsAttr().Get(Usd.TimeCode(3)))
    assert len(p3) == 4 and np.allclose(p3[:2], start[:2] + v / sc.fps, atol=1e-3)
    assert list(g.GetIdsAttr().Get(Usd.TimeCode(3))) == [0, 1, 3, 4]
    vel = np.asarray(g.GetVelocitiesAttr().Get(Usd.TimeCode(4)))
    assert np.allclose(vel[:3], v, atol=0.1) and np.allclose(vel[3], 0.0, atol=0.1)
    w0 = g.GetWidthsAttr().Get(Usd.TimeCode(3))
    assert g.GetWidthsInterpolation() == 'constant' and w0[0] == pytest.approx(SU.MATTER_WIDTH * dx)
    col = np.asarray(UsdGeom.PrimvarsAPI(g).GetPrimvar('displayColor').Get(Usd.TimeCode(3)))
    assert np.allclose(col[0], np.array(mats[0].colour) * (1.0 + mats[0].variation * (int(0.5 * 4095) / 4095 - 0.5)), atol=1e-4)
    assert np.allclose(col[1] / col[1].max(), np.array((0.9, 0.1, 0.1)) / 0.9, atol=1e-3)
    assert list(st.GetPrimAtPath('/World/Matter').GetAttribute('blackbody:materials').Get()) == ['Sand', 'Snow (own colour)']
    assert not UsdGeom.PrimvarsAPI(g).HasPrimvar('temperature')
    ext = np.asarray(g.GetExtentAttr().Get(Usd.TimeCode(3)))
    assert (ext[0] <= p3.min(0)).all() and (ext[1] >= p3.max(0)).all()


def test_matter_decoding_and_live_velocities():
    from blackbody.engine.matter import material
    origin, dx, dims = np.array([-1.0, 0.0, -1.0]), 0.05, (40, 30, 40)
    pos = np.array([[0.0, 0.5, 0.0], [0.7, 1.2, -0.4]])
    F = np.eye(4)
    F[:3, :3] = cam.rot_y(math.radians(90.0))
    F[:3, 3] = (5.0, 0.0, 0.0)
    out = SU.matter_points(_matter_snap(pos, [0, 0], dims, origin, dx), np.array([300.0, 900.0], np.float16), origin, dx,
                           dims, [material('chocolate')], [(None, 1.0)], F, vel=np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))
    assert np.allclose(out['points'], pos @ F[:3, :3].T + F[:3, 3], atol=dx * 30 / 65535 * 2)
    assert np.allclose(out['velocities'][0], F[:3, :3] @ [1.0, 0.0, 0.0]) and list(out['temperature']) == [300.0, 900.0]


def test_grass_blades_as_curves(tmp_path):
    from blackbody.engine.strands import POINTS, StrandSpec, blades, patch_data
    specs = [StrandSpec(kind='meadow', size=(0.3, 0.45, 0.3), seed=1), StrandSpec(kind='wheat', pos=(1.0, 0.0, 0.0),
                                                                                    size=(0.2, 0.9, 0.2))]
    BL = np.concatenate([blades(s, p, 200) for p, s in enumerate(specs)])
    n = len(BL)
    x = np.zeros((n, POINTS, 3))
    x[:, :, 0], x[:, :, 2] = BL[:, 0, 0, None], BL[:, 0, 1, None]
    x[:, :, 1] = np.linspace(0.0, 1.0, POINTS)[None] * BL[:, 1, 0, None]
    st = np.zeros((n, 4), np.float32)
    st[:5, 1] = 1.5                                          # (burnt to stubble)
    grows = np.ones(n, bool)
    grows[-3:] = False
    out = SU.grass_blades(x, st, grows, BL, patch_data(specs), ['Long grass', 'Wheat'], np.eye(4))
    assert [g['name'] for g in out] == ['Long grass', 'Wheat'] and sum(len(g['points']) for g in out) == n - 3
    g0 = out[0]
    assert g0['widths'][:, -1].max() == 0.0 and np.allclose(g0['widths'][:, 0], BL[:len(g0['widths']), 1, 1])
    assert np.allclose(g0['colours'][:5], np.array(SU.CHAR)[None] * (0.8 + 0.4 * BL[:5, 0, 3, None]), atol=1e-5)
    w = SU.SceneWriter(tmp_path / 'g.usdc', 24.0)
    w.add(1, dict(grass=out))
    w.add(2, dict(grass=out))
    w.close()
    stg = Usd.Stage.Open(str(tmp_path / 'g.usdc'))
    c = UsdGeom.BasisCurves(stg.GetPrimAtPath('/World/Grass/Wheat'))
    assert c.GetTypeAttr().Get() == 'linear'
    counts = c.GetCurveVertexCountsAttr().Get(Usd.TimeCode(2))
    assert len(counts) == len(out[1]['points']) and set(counts) == {POINTS}
    assert np.allclose(np.asarray(c.GetPointsAttr().Get(Usd.TimeCode(2))), out[1]['points'].reshape(-1, 3))
    burn = UsdGeom.PrimvarsAPI(stg.GetPrimAtPath('/World/Grass/Long_grass')).GetPrimvar('burn')
    assert burn.GetInterpolation() == 'uniform' and list(burn.Get(Usd.TimeCode(1)))[:5] == [1.5] * 5


def test_grass_patches_with_the_same_name_are_kept_apart(tmp_path):
    # (a copied patch keeps its name: keyed by name, the second overwrote the first in every frame)
    from blackbody.engine.strands import POINTS, StrandSpec, blades, patch_data
    specs = [StrandSpec(kind='meadow', size=(0.3, 0.45, 0.3), seed=1), StrandSpec(kind='meadow', pos=(1.0, 0.0, 0.0),
                                                                                    size=(0.2, 0.45, 0.2), seed=2)]
    BL = np.concatenate([blades(s, p, 200) for p, s in enumerate(specs)])
    x = np.zeros((len(BL), POINTS, 3))
    x[:, :, 0], x[:, :, 2] = BL[:, 0, 0, None], BL[:, 0, 1, None]
    x[:, :, 1] = np.linspace(0.0, 1.0, POINTS)[None] * BL[:, 1, 0, None]
    out = SU.grass_blades(x, np.zeros((len(BL), 4), np.float32), np.ones(len(BL), bool), BL, patch_data(specs),
                          ['Grass', 'Grass'], np.eye(4))
    w = SU.SceneWriter(tmp_path / 'g.usdc', 24.0)
    w.add(1, dict(grass=out))
    w.close()
    st = Usd.Stage.Open(str(tmp_path / 'g.usdc'))
    for path, g in zip(('/World/Grass/Grass', '/World/Grass/Grass_2'), out):
        pts = np.asarray(UsdGeom.BasisCurves(st.GetPrimAtPath(path)).GetPointsAttr().Get(Usd.TimeCode(1)))
        assert np.allclose(pts, g['points'].reshape(-1, 3))


def test_embers_as_points(tmp_path):
    n = 6
    A = np.zeros((n, 4), np.float32)
    A[:, :3] = np.arange(n)[:, None] * [0.1, 0.2, 0.0]
    A[:4, 3] = 1.0                                           # (four alive)
    B = np.zeros((n, 4), np.float32)
    B[:, 1] = 2.0
    B[:, 3] = [900.0, 1500.0, 2500.0, 1200.0, 0.0, 0.0]
    D = np.zeros((n, 4), np.float32)
    D[:, 1] = 0.004
    E = np.zeros((n, 4), np.float32)
    F = np.eye(4)
    F[:3, 3] = (1.0, 0.0, 0.0)
    em = SU.ember_points(A, B, D, E, F, time_scale=0.5)
    assert len(em['points']) == 4 and np.allclose(em['points'][1], [1.1, 0.2, 0.0]) and np.allclose(em['velocities'][:, 1], 1.0)
    c = em['colours']
    assert np.allclose(c.max(1), 1.0) and c[0, 2] < c[2, 2]          # (cooler embers redder)
    w = SU.SceneWriter(tmp_path / 'e.usdc', 24.0)
    w.add(5, dict(embers=em))
    w.close()
    st = Usd.Stage.Open(str(tmp_path / 'e.usdc'))
    g = UsdGeom.Points(st.GetPrimAtPath('/World/Embers'))
    assert len(g.GetPointsAttr().Get(Usd.TimeCode(5))) == 4
    assert list(UsdGeom.PrimvarsAPI(g).GetPrimvar('temperature').Get(Usd.TimeCode(5))) == [900.0, 1500.0, 2500.0, 1200.0]


def test_a_shot_with_layers_writes_each_layers_vdb_beside_the_base_layers(tmp_path):
    # the layered branch used to write only the base layer's VDB: now every layer's, named for it (GPU-free stand-ins)
    from blackbody.io.vdb import read_vdb
    shot = presets.make('campfire')
    torch = presets.make('torch')
    torch.uid, torch.name = 'torch', 'Torch'
    shot.layers = [torch]
    f = shot.start
    rng = np.random.default_rng(3)

    def eng(seed):
        scal = np.zeros((8, 8, 8, 4), np.float16)
        scal[2:6, 2:6, 2:6, 2] = 1.0 + seed
        e = {'scal': scal, 'dims': [8, 8, 8], 'origin': [0.0, 0.0, 0.0], 'h': 0.1, 'vel': rng.uniform(-1, 1, (9, 9, 9, 4))}
        return SimpleNamespace(sim_frame=None, solver=SimpleNamespace(dims=(4, 4, 4), h=0.1, origin=(0, 0, 0), features={}),
                               cache={f: e})

    engines = {'base': eng(0), 'torch': eng(1)}
    job = RenderJob(shot, [], engines['base'], frames=(f, f))
    o = Output('vdb', str(tmp_path / 'v' / 'fire.####.vdb'))
    tags = J.layer_tags(shot.layer_order())
    for uid, lay in shot.layer_order():
        job._write_vdb(o, f, engines[uid], lay, tags[uid])
    base, layer = J.frame_path(o.path, f), J.frame_path(J.tagged(o.path, '.torch'), f)
    assert job.written == [base, layer] and os.path.basename(layer) == f'fire.torch.{f:04d}.vdb'
    from blackbody.io.vdb import dense_from_leaves
    d0 = dense_from_leaves(read_vdb(base)['grids']['density']['leaves'], (8, 8, 8), False).max()
    d1 = dense_from_leaves(read_vdb(layer)['grids']['density']['leaves'], (8, 8, 8), False).max()
    assert d0 == pytest.approx(1.0) and d1 == pytest.approx(2.0)


def test_render_window_offers_the_scene_and_the_camera(tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    QtWidgets = pytest.importorskip('PySide6.QtWidgets')
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from blackbody.ui.document import Document
    from blackbody.ui.export_dialog import ExportDialog
    shot = presets.make('campfire')
    water = presets.make('towel_dip')
    water.uid, water.name = 'water', 'Pool'
    shot.layers = [water]
    doc = Document()
    doc.scene = shot
    dlg = ExportDialog(doc)
    assert dlg.mesh.isEnabled() and dlg.fabric.isEnabled()             # (the pool layer has liquid and a towel)
    for wd in (dlg.exr, dlg.png, dlg.mov, dlg.comp, dlg.comp_exr, dlg.vdb, dlg.deep, dlg.mesh, dlg.fabric):
        wd.setChecked(False)
    dlg.folder.setText(str(tmp_path))
    dlg.scene_usd.setChecked(True)
    dlg.camera.setChecked(True)
    outs = dlg.outputs()
    kinds = sorted((o.kind, o.content, os.path.basename(o.path)) for o in outs)
    name = dlg.name.text()
    assert kinds == [('camera', 'camera', f'{name}_camera.chan'), ('scene', 'camera', f'{name}_camera.usda'),
                     ('scene', 'scene', f'{name}_scene.usdc')]
    assert dlg.scene_what.text().split(' · ') == ['camera', 'embers'] and J.check_outputs(shot, outs) == []
    assert 'liquid and the fabric' in dlg.preview_paths.text()


def test_the_camera_alone_needs_no_simulation(tmp_path):
    # a job of camera outputs only never touches the engine
    sc = _placed('campfire', roll=5.0)
    outs = [Output('camera', str(tmp_path / 'c.chan'), 'camera'), Output('scene', str(tmp_path / 'c.usda'), 'camera')]
    eng = SimpleNamespace(cache=SimpleNamespace(disk=None), gpu=SimpleNamespace(name='none'))
    written = RenderJob(sc, outs, eng, frames=(1, 3), size=SIZE).run()
    assert str(tmp_path / 'c.chan') in written and str(tmp_path / 'c.usda') in written
    st = Usd.Stage.Open(str(tmp_path / 'c.usda'))
    assert [p.GetName() for p in st.GetPrimAtPath('/World').GetChildren()] == ['Camera']
    assert st.GetPrimAtPath('/World/Camera').GetAttribute('focalLength').GetTimeSamples() == [1.0, 2.0, 3.0]


# -- GPU ------------------------------------------------------------------------------------------------------------

def test_scene_of_falling_matter_and_a_breaking_vase(engine, tmp_path):
    from blackbody.engine.solids import fractured
    try:
        # jelly dropped on a steel ball: its points are the live particles, where the simulation has them
        sc = presets.make('jelly_ball')
        sc.data['domain'].update(resolution=32, preroll=0.0)
        f0, f1 = sc.start, sc.start + 3
        out = str(tmp_path / 'jelly.scene.usdc')
        engine.invalidate()
        RenderJob(sc, [Output('scene', out, 'scene')], engine, frames=(f0, f1), final=False, size=(160, 90)).run()
        st = Usd.Stage.Open(out)
        g = UsdGeom.Points(st.GetPrimAtPath('/World/Matter'))
        pts = np.asarray(g.GetPointsAttr().Get(Usd.TimeCode(f1)))
        live, _slots = engine._matter.positions()
        F = SU.fire_matrix(sc, f1)
        want = live @ F[:3, :3].T + F[:3, 3]
        assert len(pts) == len(want) > 1000
        tol = float(np.max(np.asarray(engine._matter.dims) * engine._matter.dx)) / 65535.0 * 2.0
        assert np.abs(np.sort(pts, axis=0) - np.sort(want, axis=0)).max() < tol
        vel = np.asarray(g.GetVelocitiesAttr().Get(Usd.TimeCode(f1)))
        P = engine._matter.read_particles()
        v = P[P[:, 3] >= 0.0, 4:7] @ F[:3, :3].T * float(sc.v('domain', 'time_scale', f1))
        assert np.abs(v).max() > 0.05 and np.allclose(vel, v, atol=1e-5)    # (the particles' own: for motion blur)
        assert st.GetPrimAtPath('/World/Objects/Steel_ball/Shape').GetTypeName() == 'Sphere'

        # a vase dropped onto a table breaks: whole, then its pieces where the simulation has them
        sc = presets.make('vase_drop')
        sc.data['domain'].update(resolution=32, preroll=0.0)
        out = str(tmp_path / 'vase.scene.usdc')
        engine.invalidate()
        RenderJob(sc, [Output('scene', out, 'scene')], engine, frames=(sc.start, sc.end), final=False, size=(160, 90)).run()
        st = Usd.Stage.Open(out)
        vase = UsdGeom.Imageable(st.GetPrimAtPath('/World/Objects/Vase'))
        assert vase.ComputeVisibility(Usd.TimeCode(sc.start)) == 'inherited'
        assert vase.ComputeVisibility(Usd.TimeCode(sc.end)) == 'invisible'
        poses = engine.piece_poses(sc.end)
        (ci, pose), = poses.items()
        frac = fractured(sc.colliders[ci], pose['size'], float(pose['hollow']), pose.get('impact'), sc)
        F = SU.fire_matrix(sc, sc.end)
        prims = st.GetPrimAtPath('/World/Pieces/Vase').GetChildren()
        assert len(prims) == len(frac.pieces)
        assert UsdGeom.Imageable(prims[0]).ComputeVisibility(Usd.TimeCode(sc.start)) == 'invisible'
        for k in (0, len(prims) // 2, len(prims) - 1):
            M = _world(prims[k], sc.end)
            assert np.allclose(M[:3, 3], (F @ np.r_[pose['pos'][k], 1.0])[:3], atol=1e-4)
    finally:
        engine.invalidate()


def test_layered_shot_writes_every_layers_vdb_and_liquid_mesh(engine, tmp_path):
    # a fire with a pool as a layer: the fire's VDB, and the pool's VDB and surface beside it (the base layer has no
    # liquid: its mesh output used to be skipped, and the layer's VDB never written)
    from blackbody.io.vdb import read_vdb
    shot = presets.make('campfire')
    shot.data['domain'].update(resolution=32, preroll=0.0)
    pool = presets.make('towel_dip')
    pool.uid, pool.name = 'pool', 'Pool'
    pool.data['domain'].update(resolution=32, preroll=0.0)
    shot.layers = [pool]
    shot.sync_layers()
    f = shot.start + 1
    outs = [Output('vdb', str(tmp_path / 'v' / 'fx.####.vdb')), Output('mesh', str(tmp_path / 'm' / 'fx.usdc'), 'liquid')]
    try:
        written = RenderJob(shot, outs, engine, frames=(f, f), final=False, size=(96, 54)).run()
    finally:
        engine.invalidate()
    base, layer = J.frame_path(outs[0].path, f), J.frame_path(J.tagged(outs[0].path, '.pool'), f)
    surface = str(tmp_path / 'm' / 'fx.pool.usdc')
    assert {base, layer, surface} <= set(written) and not os.path.exists(outs[1].path)
    # (the fire's, and the pool's: a liquid's surface)
    assert 'blackbody_temperature_1_kelvin' in read_vdb(base)['meta']
    assert 'blackbody_liquid_surface_iso' in read_vdb(layer)['meta'] and read_vdb(layer)['grids']['density']['leaves']
    st = Usd.Stage.Open(surface)
    assert len(UsdGeom.Mesh(st.GetPrimAtPath('/World/Liquid/Surface')).GetPointsAttr().Get(Usd.TimeCode(f))) > 100
