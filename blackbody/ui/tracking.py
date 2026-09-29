"""Track the fire base through the footage (runs in its own thread with its own decoder)."""
from __future__ import annotations

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import QMessageBox, QProgressDialog

from ..io.footage import Footage
from ..io.tracker import PointTracker


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


def track_fire_base(window):
    doc = window.doc
    sc = doc.scene
    if not sc.footage or not sc.footage.get('path'):
        QMessageBox.information(window, 'Track', 'Import footage first (File › Import footage).')
        return
    spec, fire = sc.camera(doc.frame)
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
    def fn(s):
        s.track = None
    window.doc.edit('Clear track', fn, structure=True)
    window.doc.sceneReplaced.emit()
    window.msg.setText('Track cleared. The fire stays at the Base X / Base Y position.')
