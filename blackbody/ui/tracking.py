"""Tracking (in its own thread, with its own decoder): the effect's base pinned to a spot of the footage, or, once
the ground is lined up, the camera's turn through the shot from many spots (scene/camsolve.py)."""
from __future__ import annotations

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import QMessageBox, QProgressDialog

from ..io.footage import Footage
from ..io.tracker import MultiTracker, PointTracker


class TrackThread(QThread):
    progress = Signal(float, int)
    finished_track = Signal(object, str)

    def __init__(self, path, start_index, point, first, last, parent=None):
        super().__init__(parent)
        self.args = (path, start_index, point, first, last)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        path, start, point, first, last = self.args
        try:
            fo = Footage(path)
            tr = PointTracker(fo)
            pts, msg = tr.track(start, point, first, last, progress=lambda f, i: self.progress.emit(f, i),
                                cancelled=lambda: self._stop)
            fo.close()
            self.finished_track.emit(pts, msg)
        except Exception as ex:
            self.finished_track.emit(None, f'Tracking failed: {ex}')


class CameraTrackThread(QThread):
    progress = Signal(float, int)
    finished_track = Signal(object, str)

    def __init__(self, path, start_index, first, last, parent=None):
        super().__init__(parent)
        self.args = (path, start_index, first, last)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        path, start, first, last = self.args
        try:
            fo = Footage(path)
            tracks, msg = MultiTracker(fo).track(start, first, last, progress=lambda f, i: self.progress.emit(f, i),
                                                 cancelled=lambda: self._stop)
            fo.close()
            self.finished_track.emit(tracks, msg)
        except Exception as ex:
            self.finished_track.emit(None, f'Tracking failed: {ex}')


def track_camera(window):
    """Work out the camera's pan, tilt and roll through the shot, from the frame the ground was lined up on."""
    import numpy as np
    from ..engine import camera as cam
    from ..scene import camsolve
    from ..scene.anim import Curve
    doc = window.doc
    sc = doc.scene
    g = sc.ground
    doc.set_playing(False)
    off = int(sc.footage.get('offset', 0))
    ref = int(g.get('frame', doc.frame))
    start_index = ref - sc.start + off
    first_index, last_index = off, sc.end - sc.start + off
    W, H = sc.output_size()
    spec, _ = sc.camera(ref)
    R0 = cam.euler_xyz(*spec.rotation)
    eye0 = np.asarray(spec.position, float)
    f = spec.focal_mm * W / spec.sensor_mm
    dlg = QProgressDialog('Tracking the camera move: following spots through the footage…', 'Stop', 0, 1000, window)
    dlg.setWindowTitle('Track the camera')
    dlg.setWindowModality(Qt.WindowModal)
    dlg.setMinimumDuration(0)
    th = CameraTrackThread(sc.footage['path'], start_index, first_index, last_index, window)
    th.progress.connect(lambda fr, i: dlg.setValue(int(fr * 1000)))
    dlg.canceled.connect(th.stop)

    def done(tracks, msg):
        dlg.canceled.disconnect()
        dlg.close()
        if not tracks or start_index not in tracks or len(tracks) < 2:
            window.msg.setText(msg)
            return
        frames = {int(i - off + sc.start): pts for i, pts in tracks.items()}
        rots, eyes, err, note, how = camsolve.solve_shot(frames, ref, R0, eye0, (W, H), f)
        if len(rots) < 2:
            window.msg.setText(note)
            return
        keys = camsolve.rotation_keys(rots)
        still = how == 'turn' and max(camsolve.angle_between(R0, r) for r in rots.values()) < 0.05
        good = [e for e in err.values() if e == e]

        def fn(s):
            for x in [doc.shot] + list(doc.shot.layers):
                c = x.data['camera']
                c['rotation'] = c['rotation'] if still else Curve([[float(fr), e, 'linear'] for fr, e in keys])
                if how == 'travel':
                    c['position'] = Curve([[float(fr), tuple(float(v) for v in eyes[fr]), 'linear'] for fr in sorted(eyes)])
                elif isinstance(c.get('position'), Curve):
                    x.clear_anim(('camera', 'position'), ref)
                if x.ground:
                    x.ground = dict(x.ground, tracked=len(rots), error=round(float(np.median(good)) if good else 0.0, 2),
                                    travel=how == 'travel')
        doc.edit('Track the camera', fn, structure=True)
        window.msg.setText(('The camera does not move: nothing to follow. ' if still else '') + note)

    th.finished_track.connect(done)
    th.finished.connect(th.deleteLater)
    window._track_thread = th
    th.start()


def track_fire_base(window):
    doc = window.doc
    sc = doc.scene
    if not sc.footage or not sc.footage.get('path'):
        QMessageBox.information(window, 'Track', 'Import footage first (File › Import footage).')
        return
    spec, fire = sc.camera(doc.frame)
    if not spec.use_anchor and sc.ground:   # the camera matches the footage: follow its move
        return track_camera(window)
    if not spec.use_anchor:
        QMessageBox.information(window, 'Track', 'Tracking moves the fire in the frame, so it needs "Place in frame" on (Camera settings).')
        return
    doc.set_playing(False)
    off = int(sc.footage.get('offset', 0))
    start_index = doc.frame - sc.start + off
    first_index = off
    last_index = sc.end - sc.start + off
    if sc.track and sc.track.get('points'):
        from ..scene.model import track_point
        tp = track_point(sc.track, doc.frame)
        o = sc.track.get('offset', (0.0, 0.0))
        point = (tp[0] + o[0], tp[1] + o[1])
    else:
        point = spec.anchor
    dlg = QProgressDialog('Tracking the fire base through the footage…', 'Stop', 0, 1000, window)
    dlg.setWindowTitle('Track')
    dlg.setWindowModality(Qt.WindowModal)
    dlg.setMinimumDuration(0)
    th = TrackThread(sc.footage['path'], start_index, point, first_index, last_index, window)
    th.progress.connect(lambda f, i: dlg.setValue(int(f * 1000)))
    dlg.canceled.connect(th.stop)

    def done(pts, msg):
        dlg.canceled.disconnect()
        dlg.close()
        if pts and len(pts) > 1:
            frames = {int(i - off + sc.start): (float(x), float(y)) for i, (x, y) in pts.items()}

            def fn(s):
                s.track = {'points': frames, 'offset': (0.0, 0.0)}
            doc.edit('Track fire base', fn, structure=True)
            doc.sceneReplaced.emit()
        window.msg.setText(msg)

    th.finished_track.connect(done)
    th.finished.connect(th.deleteLater)  # only once run() has returned
    window._track_thread = th
    th.start()


def clear_track(window):
    from ..scene.anim import Curve
    doc = window.doc

    def fn(s):
        s.track = None
        for x in [doc.shot] + list(doc.shot.layers):   # a tracked camera move goes back to the lined-up camera
            if x.ground:
                for k in ('rotation', 'position'):
                    if isinstance(x.data['camera'].get(k), Curve):
                        x.clear_anim(('camera', k), int(x.ground.get('frame', x.start)))
                x.ground = {k: v for k, v in x.ground.items() if k not in ('tracked', 'error', 'travel')}
    window.doc.edit('Clear track', fn, structure=True)
    window.doc.sceneReplaced.emit()
    window.msg.setText('Track cleared. The fire stays at the Base X / Base Y position.')
