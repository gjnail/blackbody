"""Screen recordings of the app for the docs: drive the real window offscreen with real mouse and key
events, grab it as it plays, and draw a cursor (and the keys pressed) over the frames.

    python record_ui.py NAME [NAME ...]      # into out/docs_media/ui/NAME/ (one folder per recording)

Then compose_ui.py NAME draws the cursor in at 24 fps, and encode_media.py ui NAME encodes it.

Uses throwaway settings, so the user's own Blackbody settings are left alone."""
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
if sys.platform == 'win32':
    os.environ.setdefault('QT_QPA_FONTDIR', 'C:/Windows/Fonts')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = ROOT / 'out' / 'docs_media'
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont                        # noqa: E402
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, QSettings, Qt   # noqa: E402
from PySide6.QtGui import QKeyEvent, QMouseEvent                   # noqa: E402
from PySide6.QtWidgets import QApplication                         # noqa: E402

PLATE = ROOT / 'out' / 'practice_plate.mp4'   # tools/make_test_footage.py makes it
SIZE = (1600, 900)
FPS = 20


def qimage_to_pil(img):
    img = img.convertToFormat(img.Format.Format_RGBA8888)
    return Image.frombuffer('RGBA', (img.width(), img.height()), bytes(img.constBits()), 'raw', 'RGBA',
                            img.bytesPerLine(), 1).convert('RGB')


def font(size, bold=True):
    for name in (('segoeuib.ttf', 'arialbd.ttf') if bold else ('segoeui.ttf', 'arial.ttf')):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


ARROW = [(0, 0), (0, 17), (4.2, 13.2), (7.2, 20), (10, 18.8), (7.1, 12.2), (12.5, 12.2)]


class Recorder:
    """Grabs the window whenever a frame is due (as fast as grabbing allows) and logs the cursor, clicks
    and captions against the clock; compose() later draws them over the frames at a steady frame rate."""

    def __init__(self, app, win, out):
        self.app, self.win, self.out = app, win, Path(out)
        if self.out.exists():
            shutil.rmtree(self.out)
        self.out.mkdir(parents=True)
        self.t0 = time.perf_counter()
        self.cut = 0.0           # seconds cut out of the recording (waits for the simulation)
        self.frames = []         # (t, file)
        self.cursor = []         # (t, x, y, down)
        self.ripples = []        # (t, x, y)
        self.captions = []       # (t, text, seconds)
        self.cur = QPointF(SIZE[0] * 0.55, SIZE[1] * 0.5)   # cursor, in window coordinates
        self.down = False
        self.next_t = 0.0
        self._log()

    def now(self):
        return time.perf_counter() - self.t0 - self.cut

    def wait_for_frames(self, got, n=3, timeout=60.0):
        """Cut the wait out of the recording: keep the app running, without grabbing, until it has drawn `n` new frames
        (after a change restarts the simulation and its pre-roll), then carry on as if no time had passed."""
        got.clear()
        t0 = time.perf_counter()
        while len(got) < n and time.perf_counter() - t0 < timeout:
            self.app.processEvents()
            time.sleep(0.005)
        self.cut += time.perf_counter() - t0

    def _log(self):
        self.cursor.append((self.now(), self.cur.x(), self.cur.y(), self.down))

    # -- frames ---------------------------------------------------------------------------------------
    def tick(self):
        self.app.processEvents()
        if self.now() >= self.next_t:
            self.snap()
            self.next_t = self.now() + 1.0 / FPS

    def pump(self, seconds):
        t_end = self.now() + seconds
        while self.now() < t_end:
            self.tick()
            time.sleep(0.002)

    def snap(self):
        t = self.now()
        name = f'{len(self.frames):04d}.bmp'
        QApplication.primaryScreen().grabWindow(self.win.winId()).toImage().save(str(self.out / name))
        self.frames.append((t, name))

    def finish(self):
        self.pump(0.05)
        self._log()
        meta = dict(size=[self.win.width(), self.win.height()], frames=self.frames, cursor=self.cursor,
                    ripples=self.ripples, captions=self.captions, end=self.now())
        (self.out / 'meta.json').write_text(json.dumps(meta))

    # -- input ----------------------------------------------------------------------------------------
    def _to(self, widget, p):
        """Window point -> widget-local point."""
        return QPointF(widget.mapFrom(self.win, QPoint(int(p.x()), int(p.y()))))

    def where(self, widget, local):
        """Widget-local point -> window point."""
        q = widget.mapTo(self.win, QPoint(int(local.x()), int(local.y())))
        return QPointF(q.x(), q.y())

    def _send(self, widget, kind, p, button=Qt.NoButton, buttons=Qt.NoButton, mods=Qt.NoModifier):
        lp = self._to(widget, p)
        gp = QPointF(widget.mapToGlobal(lp.toPoint()))
        QApplication.sendEvent(widget, QMouseEvent(kind, lp, gp, button, buttons, mods))

    def move(self, target, seconds=0.6, widget=None, mods=Qt.NoModifier):
        """Glide the cursor to `target` (window point) in `seconds` of real time; with the button down,
        drag `widget` along."""
        start = QPointF(self.cur)
        t0 = self.now()
        while True:
            a = min(1.0, (self.now() - t0) / max(seconds, 1e-3))
            e = a * a * (3 - 2 * a)
            self.cur = start + (target - start) * e
            self._log()
            if widget is not None:
                self._send(widget, QEvent.MouseMove, self.cur, Qt.NoButton,
                           Qt.LeftButton if self.down else Qt.NoButton, mods)
            self.tick()
            if a >= 1.0:
                return
            time.sleep(0.004)

    def press(self, widget, mods=Qt.NoModifier):
        self.down = True
        self._log()
        self.ripples.append((self.now(), self.cur.x(), self.cur.y()))
        self._send(widget, QEvent.MouseButtonPress, self.cur, Qt.LeftButton, Qt.LeftButton, mods)
        self.pump(0.08)

    def release(self, widget, mods=Qt.NoModifier):
        self.down = False
        self._log()
        self._send(widget, QEvent.MouseButtonRelease, self.cur, Qt.LeftButton, Qt.NoButton, mods)
        self.pump(0.05)

    def click(self, widget):
        self.press(widget)
        self.release(widget)

    def double_click(self, widget):
        self.press(widget)
        self.release(widget)
        self.ripples.append((self.now() + 0.05, self.cur.x(), self.cur.y()))
        self._send(widget, QEvent.MouseButtonDblClick, self.cur, Qt.LeftButton, Qt.LeftButton)
        self._send(widget, QEvent.MouseButtonRelease, self.cur, Qt.LeftButton, Qt.NoButton)
        self.pump(0.05)

    def key(self, widget, key, text='', mods=Qt.NoModifier, caption=None):
        if caption:
            self.say(caption, 1.1)
        QApplication.sendEvent(widget, QKeyEvent(QEvent.KeyPress, key, mods, text))
        QApplication.sendEvent(widget, QKeyEvent(QEvent.KeyRelease, key, mods, text))
        self.pump(0.05)

    def say(self, text, seconds=1.6):
        self.captions.append((self.now(), text, seconds))


# ---- the app -------------------------------------------------------------------------------------------

def start_app():
    QSettings.setDefaultFormat(QSettings.IniFormat)
    settings_dir = tempfile.mkdtemp(prefix='bb_record_')
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, settings_dir)
    QCoreApplication.setOrganizationName('BlackbodyRecord')
    QCoreApplication.setApplicationName('Blackbody')
    app = QApplication([sys.argv[0]])
    from blackbody.ui import theme
    theme.apply(app)
    from blackbody.ui.main_window import MainWindow
    from blackbody.ui.worker import EngineWorker
    QSettings().setValue('ui/welcome_done', True)
    QSettings().setValue('ui/quality_index', 1)
    w = EngineWorker()
    w.start()
    win = MainWindow(w)
    win._autosave.stop()
    win.resize(*SIZE)
    win.show()
    got = []
    w.frameReady.connect(lambda img, f, st: got.append((f, bool(st.get('refined')))))
    return app, win, w, got, settings_dir


def settle(app, got, seconds=90, frame=None):
    got.clear()
    t = time.time()
    while time.time() - t < seconds:
        app.processEvents()
        if any(r and (frame is None or f == frame) for f, r in got):
            break
        time.sleep(0.01)
    for _ in range(20):
        app.processEvents()
        time.sleep(0.01)


def lib_item_point(rec, win, key):
    lst = win.library.list
    for i in range(lst.count()):
        it = lst.item(i)
        if it.data(Qt.UserRole) == ('builtin', key):
            lst.scrollToItem(it)
            QApplication.processEvents()
            r = lst.visualItemRect(it)
            return it, rec.where(lst.viewport(), QPointF(r.center().x(), r.top() + r.height() * 0.38))
    raise KeyError(key)


def row_for(win, path):
    for r in win.props.panel.rows:
        if tuple(r.path) == tuple(path):
            return r
    raise KeyError(path)


# ---- the recordings --------------------------------------------------------------------------------------

def rec_library(app, win, got, out):
    """Pick fires and liquids from the Library: double-click, and it simulates live."""
    doc = win.doc
    doc.load_preset('campfire', keep_shot=False)
    settle(app, got)
    win.left.show_page('effects')
    doc.set_playing(True)
    rec = Recorder(app, win, out)
    rec.pump(1.2)
    lst = win.library.list
    for key, hold in (('fire_whirl', 3.0), ('coloured_flames', 2.6), ('floating', 3.4)):
        it, p = lib_item_point(rec, win, key)
        rec.move(p, 0.7)
        rec.pump(0.25)
        rec.double_click(lst.viewport())
        doc.set_playing(True)
        rec.pump(hold)
    doc.set_playing(False)
    return rec


def rec_place(app, win, got, out):
    """Import footage, then drag the ring onto the ground and the square to scale."""
    doc = win.doc
    doc.load_preset('campfire', keep_shot=False)
    doc.import_footage(str(PLATE))
    t = time.time()
    while doc.footage_info is None and time.time() - t < 30:
        app.processEvents()
        time.sleep(0.01)
    doc.set(('camera', 'anchor_x'), 0.62, merge=False)
    doc.set(('camera', 'anchor_y'), 0.70, merge=False)
    doc.set(('camera', 'scale'), 0.55, merge=False)
    doc.set_frame(doc.scene.start + 30)
    settle(app, got, frame=doc.scene.start + 30)
    rec = Recorder(app, win, out)
    vp = win.viewport
    rec.pump(0.8)
    base, top, ok, spec = vp._handles()
    rec.move(rec.where(vp, base), 0.8)
    rec.pump(0.2)
    rec.press(vp)
    target = rec.where(vp, QPointF(vp.frame_rect().x() + vp.frame_rect().width() * 0.30,
                                   vp.frame_rect().y() + vp.frame_rect().height() * 0.90))
    rec.move(target, 1.6, widget=vp)
    rec.release(vp)
    rec.pump(0.8)
    base, top, ok, spec = vp._handles()
    rec.move(rec.where(vp, top), 0.7)
    rec.pump(0.2)
    rec.press(vp)
    rec.move(rec.where(vp, top) + QPointF(0, -110), 1.4, widget=vp)
    rec.release(vp)
    rec.pump(0.6)
    doc.set_playing(True)
    rec.pump(3.0)
    doc.set_playing(False)
    return rec


def rec_scrub(app, win, got, out):
    """Drag a setting's name to scrub it: wind leans the flames while the simulation runs."""
    doc = win.doc
    doc.load_preset('campfire', keep_shot=False)
    settle(app, got)
    doc.select(('section', 'motion'))
    for _ in range(30):
        app.processEvents()
        time.sleep(0.01)
    row = row_for(win, ('motion', 'wind_speed'))
    win.props.scroll.ensureWidgetVisible(row, 0, 120)
    app.processEvents()
    doc.set_playing(True)
    rec = Recorder(app, win, out)
    rec.pump(1.5)
    lab = row.label
    p = rec.where(lab, QPointF(30, lab.height() / 2))
    rec.move(p, 0.8)
    rec.pump(0.2)
    rec.say('Drag a setting’s name to scrub it', 2.4)
    rec.press(lab)
    rec.move(p + QPointF(48, 0), 1.6, widget=lab)
    rec.release(lab)
    rec.pump(2.4)
    rec.press(lab)
    rec.move(p + QPointF(-22, 0), 1.2, widget=lab)
    rec.release(lab)
    rec.pump(1.6)
    rec.say('Double-click it to reset', 1.6)
    rec.double_click(lab)
    rec.pump(1.8)
    doc.set_playing(False)
    return rec


def rec_views(app, win, got, out):
    """The view modes: composite, fire, alpha, emission, heat, depth, temperature (keys 1 to 7)."""
    doc = win.doc
    doc.load_preset('campfire', keep_shot=False)
    doc.import_footage(str(PLATE))
    t = time.time()
    while doc.footage_info is None and time.time() - t < 30:
        app.processEvents()
        time.sleep(0.01)
    doc.set(('camera', 'anchor_x'), 0.32, merge=False)
    doc.set(('camera', 'anchor_y'), 0.9, merge=False)
    doc.set(('camera', 'scale'), 0.65, merge=False)
    doc.set_frame(doc.scene.start + 40)
    settle(app, got, frame=doc.scene.start + 40)
    rec = Recorder(app, win, out)
    rec.pump(0.8)
    names = ['Composite', 'Effect', 'Alpha', 'Emission', 'Heat', 'Depth', 'Temperature']
    for i, name in enumerate(names):
        b = win.mode_group.button(i)
        rec.move(rec.where(b, QPointF(b.width() / 2, b.height() / 2)), 0.35)
        rec.click(b)
        rec.say(f'{i + 1}   {name}', 1.2)
        rec.pump(1.25)
    rec.click(win.mode_group.button(0))
    rec.pump(0.6)
    return rec


def rec_track(app, win, got, out):
    """Track the fire base through a handheld pan (Ctrl+T) and play it back."""
    from blackbody.ui.tracking import track_fire_base
    doc = win.doc
    doc.load_preset('campfire', keep_shot=False)
    doc.import_footage(str(PLATE))
    t = time.time()
    while doc.footage_info is None and time.time() - t < 30:
        app.processEvents()
        time.sleep(0.01)
    doc.set(('camera', 'anchor_x'), 0.30, merge=False)
    doc.set(('camera', 'anchor_y'), 0.90, merge=False)
    doc.set(('camera', 'scale'), 0.65, merge=False)
    doc.set_frame(doc.scene.start + 30)
    settle(app, got, frame=doc.scene.start + 30)
    rec = Recorder(app, win, out)
    rec.pump(0.8)
    rec.say('Tracking › Track the fire base   (Ctrl+T)', 2.0)
    rec.pump(0.6)
    track_fire_base(win)
    t = time.time()
    while not doc.scene.track and time.time() - t < 120:
        rec.pump(0.1)
    rec.pump(0.5)
    doc.set_frame(doc.scene.start)
    doc.set_playing(True)
    rec.pump(6.0)
    doc.set_playing(False)
    return rec


def rec_liquid(app, win, got, out):
    """Water: a rock dropped into a pond, the camera orbited with Alt-drag."""
    doc = win.doc
    doc.load_preset('rock_splash', keep_shot=False)
    settle(app, got, seconds=120)
    rec = Recorder(app, win, out)
    vp = win.viewport
    doc.set_playing(True)
    rec.pump(3.0)
    r = vp.frame_rect()
    p = rec.where(vp, QPointF(r.x() + r.width() * 0.5, r.y() + r.height() * 0.5))
    rec.move(p, 0.6)
    rec.say('Alt-drag orbits the camera', 2.2)
    rec.press(vp, mods=Qt.AltModifier)
    rec.move(p + QPointF(-140, 0), 2.0, widget=vp, mods=Qt.AltModifier)
    rec.release(vp, mods=Qt.AltModifier)
    rec.pump(2.5)
    doc.set_playing(False)
    return rec


def rec_render(app, win, got, out):
    """The render dialog: pick outputs, then render."""
    from blackbody.ui.export_dialog import ExportDialog
    doc = win.doc
    doc.load_preset('campfire', keep_shot=False)
    settle(app, got)
    dlg = ExportDialog(doc, win)
    dlg.folder.setText('D:/Shots/campfire_shot')
    for box in (dlg.exr, dlg.png, dlg.mov, dlg.comp, dlg.vdb):
        box.setChecked(False)
    dlg.show()
    dlg.move(win.x() + (SIZE[0] - dlg.width()) // 2, win.y() + 90)
    rec = Recorder(app, win, out)
    rec.win = dlg
    rec.cur = QPointF(dlg.width() * 0.8, dlg.height() * 0.85)
    rec.pump(0.6)
    for box in (dlg.exr, dlg.comp, dlg.vdb):
        rec.move(rec.where(box, QPointF(10, box.height() / 2)), 0.5)
        rec.click(box)
        box.setChecked(True)
        rec.pump(0.5)
    rec.pump(1.2)
    rec.finish()
    dlg.close()


def rec_create(app, win, got, out):
    """Build a scene from scratch in the Create tab: an empty scene, then a campfire, a curtain and wind."""
    from blackbody.ui.create import KindButton, Tile
    from PySide6.QtWidgets import QScrollArea
    doc = win.doc
    doc.load_preset('campfire', keep_shot=False)
    settle(app, got)
    win.left.show_page('create')
    for _ in range(20):
        app.processEvents()
        time.sleep(0.01)
    rec = Recorder(app, win, out)
    rec.pump(0.8)
    kinds = {b.kind: b for b in win.create.findChildren(KindButton)}
    fire = kinds.get('auto') or kinds.get('fire')   # 'Empty scene' (what you add decides what it simulates)
    rec.move(rec.where(fire, QPointF(fire.width() / 2, fire.height() / 2)), 0.7)
    rec.say('Create › Start from scratch: an empty scene', 2.0)
    rec.click(fire)
    rec.wait_for_frames(got, n=1)
    rec.pump(1.2)
    scroll = win.create.findChild(QScrollArea)
    tiles = {t.comp.key: t for t in win.create.findChildren(Tile)}
    for key, caption, hold in (('campfire', 'Click a building block to add it', 3.0), ('curtain', None, 3.5),
                               ('wind', None, 3.0)):
        t = tiles[key]
        scroll.ensureWidgetVisible(t, 20, 40)
        rec.pump(0.15)
        rec.move(rec.where(t, QPointF(t.width() / 2, t.height() / 2)), 0.7)
        if caption:
            rec.say(caption, 2.0)
        rec.click(t)
        doc.set_playing(True)
        rec.wait_for_frames(got)
        rec.pump(hold)
    doc.set_playing(False)
    win.left.show_page('effects')
    return rec


RECORDINGS = {'library': rec_library, 'place': rec_place, 'scrub': rec_scrub, 'views': rec_views,
              'track': rec_track, 'liquid': rec_liquid, 'render': rec_render,
              'create': rec_create}


def main():
    names = sys.argv[1:] or list(RECORDINGS)
    app, win, w, got, settings_dir = start_app()
    settle(app, got)
    for n in names:
        t = time.time()
        win.set_mode('composite')   # each recording starts from the composite view
        rec = RECORDINGS[n](app, win, got, WORK / 'ui' / n)
        if rec is not None:
            rec.finish()
        print(n, 'done in', round(time.time() - t, 1), 's', flush=True)
    w.stop()
    shutil.rmtree(settings_dir, ignore_errors=True)
    from PySide6.QtCore import QStandardPaths
    data = Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation))
    if 'BlackbodyRecord' in data.parts:
        shutil.rmtree(data.parent if data.parent.name == 'BlackbodyRecord' else data, ignore_errors=True)
    os._exit(0)


if __name__ == '__main__':
    main()
