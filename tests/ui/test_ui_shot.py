"""The shot's camera from the Build view, and a project packed with its files, through the window (conftest.py)."""
import dataclasses
from pathlib import Path

import numpy as np

from uikit import pump


def _view(spec, fire):
    from blackbody.engine import camera as cam
    return cam.compute(spec, 16 / 9, fire).view_proj


def test_the_shot_camera_takes_the_build_view(win):
    doc, vp = win.doc, win.viewport
    win.add_component('campfire')
    pump()
    doc.set_playing(False)
    assert win.workspace == 'build' and vp.camerabar is not None and vp.camerabar.isVisible()

    def shot(frame):
        return _view(*doc.scene.camera(frame))

    def build(frame):
        spec, fire = doc.camera(frame)
        sensor = doc.scene.camera(frame)[0].sensor_mm
        return _view(dataclasses.replace(spec, sensor_mm=sensor, focal_mm=spec.focal_mm * sensor / 36.0), fire)

    doc.scene.data['camera']['sensor_mm'] = 24.89   # (a Super 35 shot: the field of view still matches)
    wv = doc.work_view
    wv.yaw, wv.pitch, wv.distance, wv.target, wv.focal_mm = 60.0, 15.0, 4.0, (0.3, 0.8, -0.2), 30.0
    doc.move_work_view()
    pump()
    vp.camerabar.use_view()
    pump()
    assert np.abs(shot(doc.frame) - build(doc.frame)).max() < 1e-4
    # a camera move keyed from two views, turning the short way round
    doc.set_frame(1)
    wv.yaw = 170.0
    doc.move_work_view()
    vp.camerabar.key_view()
    doc.set_frame(60)
    wv.yaw, wv.distance = -170.0, 3.0
    doc.move_work_view()
    vp.camerabar.key_view()
    assert np.abs(shot(60) - build(60)).max() < 1e-4
    yaws = [k[1][1] for k in doc.scene.data['camera']['rotation'].keys]
    assert abs(yaws[-1] - yaws[0]) < 180.0, f'no spin the long way round: {yaws}'
    # looking through it at a frame between the keys
    doc.set_frame(30)
    vp.camerabar.look_through()
    pump()
    assert np.abs(shot(30) - build(30)).max() < 1e-4
    # with footage the camera is the footage's
    doc.scene.footage = {'path': 'x.mp4'}
    vp.camerabar.sync()
    assert not vp.camerabar.use.isEnabled()
    doc.scene.footage = None
    win.set_workspace('shot')
    pump()
    assert not vp.camerabar.isVisible()


def test_a_project_is_packed_with_its_files(win, monkeypatch, tmp_path):
    from blackbody.scene import pack
    from blackbody.ui import actions as A, packdialog, textdialog, textmesh
    doc = win.doc
    monkeypatch.setattr(textdialog, 'ask', lambda *a, **k: textmesh.spec_of('PACK', 'Arial Black', False, False, 0.3, 0.08))
    win.add_component('text_fire')
    win.add_component('logs')
    pump()
    doc.set_playing(False)
    doc.save(str(tmp_path / 'demo.bbfire'))
    refs = pack.gather(doc.shot)
    assert refs and all(Path(r.real).exists() for r in refs)
    d = packdialog.PackDialog(win, refs, doc.shot.path)
    d.show()
    pump()
    d.close()
    n = packdialog.pack_project(win, ask=False)
    assert n
    meshes = [c['mesh'] for c in doc.scene.colliders if c.get('mesh') and not c['mesh'].startswith('builtin:')]
    assert meshes, 'the text shape is a mesh file of its own (the logs are built in: they ship with the app)'
    for m in meshes:   # (beside the project, named relative to it: the folder moves as one)
        assert not Path(m).is_absolute() and (tmp_path / m).exists(), m
    assert A.text_spec(doc.scene.colliders[0], doc.scene) is not None, 'the packed text is still text'
