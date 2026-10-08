"""What the window tests use (tests/ui/conftest.py has the fixtures): a stand-in for the engine's worker, and pumping
and dragging the window offscreen."""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
if sys.platform == 'win32' and Path('C:/Windows/Fonts').is_dir():
    os.environ.setdefault('QT_QPA_FONTDIR', 'C:/Windows/Fonts')   # (the offscreen platform finds no fonts of its own there)

ROOT = Path(__file__).resolve().parents[2]
try:
    from PySide6.QtCore import QObject, QPoint, QPointF, Qt, QTimer, Signal
    from PySide6.QtGui import QImage, QMouseEvent
    from PySide6.QtWidgets import QApplication
except ImportError:      # (no Qt here: conftest.py leaves the window tests out)
    QApplication = None
    QObject = object
    Signal = lambda *a: None   # noqa: E731

STATS = {'dims': (64, 112, 64), 'cell_mm': 33.3, 'substeps': 3, 'sim_ms': 14, 'render_ms': 4, 'max_speed': 5.1,
         'refined': True, 'voxels': 460000, 'memory_mb': 51, 'cached': 18, 'cache_mb': 80}


class FakeWorker(QObject):
    """EngineWorker's signals and calls (ui/worker.py), with no engine: what it is sent is kept in `posts`, and a scene or
    a frame comes back as the preset's thumbnail (stamped with the scene's load_seq, as the real one does)."""
    ready = Signal(dict)
    failed = Signal(str)
    frameReady = Signal(object, int, dict)
    simProgress = Signal(float, int)
    cacheChanged = Signal(object)
    footageInfo = Signal(object)
    message = Signal(str)
    jobProgress = Signal(float, str, object)
    jobDone = Signal(object, bool, str)
    stillDone = Signal(object, str)
    status = Signal(str, float)

    def __init__(self):
        super().__init__()
        self.scene = None
        self.frame = 1
        self.posts = []
        QTimer.singleShot(10, lambda: self.ready.emit({'name': 'Stand-in GPU', 'backend': 'none'}))

    def start(self):
        pass

    def isRunning(self):
        return False

    def stop(self):
        pass

    def cancel_job(self):
        pass

    def post(self, kind, value=None):
        self.posts.append(kind)
        if kind == 'scene':
            self.scene = value
            QTimer.singleShot(5, self._emit)
        elif kind == 'frame':
            self.frame = value
            QTimer.singleShot(5, self._emit)

    def _emit(self):
        sc = self.scene
        if sc is None:
            return
        w, h = sc.output_size()
        f = ROOT / 'blackbody' / 'assets' / 'presets' / f'{sc.preset}.png'
        img = QImage(str(f)) if f.exists() else QImage(w, h, QImage.Format_RGB32)
        img = img.scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.frameReady.emit(img, self.frame, dict(STATS, load_seq=getattr(sc, 'load_seq', 0), frame=self.frame))
        self.cacheChanged.emit(list(range(sc.start, sc.start + 30)))


def pump(seconds=0.05):
    """Let the window's timers and queued signals run for a while."""
    app = QApplication.instance()
    t = time.monotonic()
    while True:
        app.processEvents()
        if time.monotonic() - t >= seconds:
            break
        time.sleep(0.005)


def drag(widget, a, b, mods=Qt.NoModifier, steps=8):
    """Press the left button at a, move to b in steps, let go (points in the widget's coordinates)."""
    a, b = QPointF(a), QPointF(b)
    from PySide6.QtTest import QTest
    QTest.mousePress(widget, Qt.LeftButton, mods, QPoint(int(a.x()), int(a.y())))
    for k in range(1, steps + 1):
        q = a + (b - a) * (k / steps)
        QApplication.sendEvent(widget, QMouseEvent(QMouseEvent.MouseMove, q, widget.mapToGlobal(q), Qt.LeftButton,
                                                   Qt.LeftButton, mods))
    QTest.mouseRelease(widget, Qt.LeftButton, mods, QPoint(int(b.x()), int(b.y())))
    pump()
