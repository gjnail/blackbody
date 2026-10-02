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

# a big virtual screen, so menus and dialogs open where they are asked to (the default offscreen screen is 800x800 and
# Qt moves popups to fit it); the config file is found relative to the working directory when Qt starts
OFFSCREEN = {'synchronizeWindowSystem': False, 'windowFrameMargins': False,
             'screens': [{'name': 'rec', 'x': 0, 'y': 0, 'width': 2560, 'height': 1600, 'logicalDpi': 96,
                          'logicalBaseDpi': 96, 'dpr': 1}]}
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen:configfile=bb_offscreen.json')
if sys.platform == 'win32':
    os.environ.setdefault('QT_QPA_FONTDIR', 'C:/Windows/Fonts')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = ROOT / 'out' / 'docs_media'
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont                        # noqa: E402
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, QSettings, Qt   # noqa: E402
from PySide6.QtGui import QKeyEvent, QMouseEvent, QPainter         # noqa: E402
from PySide6.QtWidgets import QApplication, QMenu                  # noqa: E402

PLATE = ROOT / 'out' / 'practice_plate.mp4'   # tools/make_test_footage.py makes it
# a dusk courtyard with a handheld dolly move and paving to line up on (promo/source/courtyard.py makes it)
COURT = Path(os.environ.get('BB_COURT', ROOT / 'out' / 'court_dolly.mp4'))
STILL = Path(os.environ.get('BB_STILL', ROOT / 'out' / 'court_still.mp4'))   # the courtyard, locked off
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
        img = QApplication.primaryScreen().grabWindow(self.win.winId()).toImage()
        # menus, dialogs and popups are windows of their own: paint them over the window where they are
        pops = [w for w in QApplication.topLevelWidgets() if w is not self.win and w.isVisible() and w.width() > 4
                and w.isWindow() and w.window() is not self.win and not w.inherits('QTipLabel')]
        if pops:
            p = QPainter(img)
            o = self.win.mapToGlobal(QPoint(0, 0))
            for w in pops:
                p.drawPixmap(w.mapToGlobal(QPoint(0, 0)) - o, w.grab())
            p.end()
        img.save(str(self.out / name))
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

    def type(self, widget, text, per_char=0.085):
        """Type text into the focused widget, a key at a time."""
        for ch in text:
            QApplication.sendEvent(widget, QKeyEvent(QEvent.KeyPress, 0, Qt.NoModifier, ch))
            QApplication.sendEvent(widget, QKeyEvent(QEvent.KeyRelease, 0, Qt.NoModifier, ch))
            self.pump(per_char)

    def at(self, widget, x=0.5, y=0.5):
        """The window point at a fraction of a widget."""
        return self.where(widget, QPointF(widget.width() * x, widget.height() * y))

    def press_on(self, widget, seconds=0.6, x=0.5, y=0.5):
        """Move to a widget and click it."""
        self.move(self.at(widget, x, y), seconds)
        self.pump(0.12)
        self.click(widget)

    def menu(self, menu, at, path, hold=0.35):
        """Pop a menu up at window point `at`, then glide down its items to the last label of `path` (going into
        submenus on the way) and trigger it. Labels match the start of an item's text."""
        o = self.win.mapToGlobal(QPoint(0, 0))
        menu.popup(QPoint(int(at.x()), int(at.y())) + o)
        self.pump(0.45)
        cur, opened = menu, [menu]
        for k, label in enumerate(path):
            act = next(a for a in cur.actions() if a.text().replace('&', '').startswith(label))
            r = cur.actionGeometry(act)
            g = cur.mapToGlobal(r.center()) - o
            self.move(QPointF(g.x() - r.width() * 0.25, g.y()), 0.45)
            cur.setActiveAction(act)
            self.pump(hold)
            if k < len(path) - 1:
                sub = act.menu()
                sub.popup(cur.mapToGlobal(r.topRight()) + QPoint(2, -4))
                self.pump(0.35)
                cur = sub
                opened.append(sub)
            else:
                self.ripples.append((self.now(), self.cur.x(), self.cur.y()))
                self.pump(0.12)
                for m in reversed(opened):
                    m.close()
                act.trigger()
                self.pump(0.1)


# ---- the app -------------------------------------------------------------------------------------------

def start_app():
    QSettings.setDefaultFormat(QSettings.IniFormat)
    settings_dir = tempfile.mkdtemp(prefix='bb_record_')
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, settings_dir)
    QCoreApplication.setOrganizationName('BlackbodyRecord')
    QCoreApplication.setApplicationName('Blackbody')
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / 'bb_offscreen.json').write_text(json.dumps(OFFSCREEN))
    here = os.getcwd()
    os.chdir(WORK)
    app = QApplication([sys.argv[0]])
    os.chdir(here)
    from blackbody.ui import theme
    theme.apply(app)
    from blackbody.ui.main_window import MainWindow
    from blackbody.ui.worker import EngineWorker
    QSettings().setValue('ui/welcome_done', True)
    QSettings().setValue('ui/quality_index', 1)
    QSettings().setValue('ui/shot_steps', False)   # the steps card would cover the viewer
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
    doc.import_footage(str(STILL))
    t = time.time()
    while doc.footage_info is None and time.time() - t < 30:
        app.processEvents()
        time.sleep(0.01)
    win.set_workspace('shot')
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
    doc.import_footage(str(STILL))
    t = time.time()
    while doc.footage_info is None and time.time() - t < 30:
        app.processEvents()
        time.sleep(0.01)
    win.set_workspace('shot')
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


def rec_track2d(app, win, got, out):
    """Track the fire base through a handheld pan (2D, no line-up) and play it back."""
    from blackbody.ui.tracking import track_fire_base
    doc = win.doc
    doc.load_preset('campfire', keep_shot=False)
    doc.import_footage(str(PLATE))
    t = time.time()
    while doc.footage_info is None and time.time() - t < 30:
        app.processEvents()
        time.sleep(0.01)
    win.set_workspace('shot')
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



# ---- building: Build, Create, the gizmo, the right-click menu ----------------------------------------------------------

def fresh(app, win, got, kind='auto', scale='person'):
    """An empty stage in Build (any dialog or menu a recording left open closed first)."""
    for w in QApplication.topLevelWidgets():
        if w is not win and w.isVisible() and w.isWindow():
            w.close()
    win.set_workspace('build')
    win.new_from_scratch(kind, scale)
    win.doc.set_playing(False)
    win.doc.set(('domain', 'preroll'), 0.0, merge=False)
    settle(app, got, seconds=40)


def view(doc, target, distance, yaw=None, pitch=None):
    """Point the Build camera."""
    wv = doc.work_view
    if wv is None:
        return
    wv.target, wv.distance = tuple(float(v) for v in target), float(distance)
    if yaw is not None:
        wv.yaw = float(yaw)
    if pitch is not None:
        wv.pitch = float(pitch)
    doc.move_work_view()


def tile(win, key):
    """The Create tile of a building block, scrolled into view."""
    from blackbody.ui.create import Tile
    from PySide6.QtWidgets import QScrollArea
    win.left.show_page('create')
    if hasattr(win.create, '_set_domain'):
        win.create._set_domain('all')
    QApplication.processEvents()
    t = {x.comp.key: x for x in win.create.findChildren(Tile)}[key]
    win.create.findChild(QScrollArea).ensureWidgetVisible(t, 20, 70)
    for _ in range(5):
        QApplication.processEvents()
    return t


def screen_of(vp, kind, i):
    """Widget point of an object's position."""
    sc = vp.doc.scene
    cs, fire, _ = vp.camstate()
    px, ok = vp._project_local(cs, fire, [sc.get((kind, i, 'position'), vp.doc.frame)])
    return vp.to_widget(px[0])


def gizmo_drag(rec, vp, handle, pixels, seconds=1.1):
    """Drag a gizmo arrow ('move_x') or square ('scale_y') `pixels` along its own screen direction."""
    gz = vp._gizmo()
    hs = gz.handles()
    if handle.startswith('move_'):
        a, b = hs[handle]
        start = a + (b - a) * 0.82
    else:
        a, start = hs['centre'], hs[handle]
    d = start - a
    L = max((d.x() ** 2 + d.y() ** 2) ** 0.5, 1e-6)
    u = QPointF(d.x() / L, d.y() / L)
    rec.move(rec.where(vp, start), 0.6)
    rec.pump(0.15)
    rec.press(vp)
    rec.move(rec.where(vp, start + u * pixels), seconds, widget=vp)
    rec.release(vp)
    rec.pump(0.3)


def object_menu(rec, win, sel, path):
    """Right-click an object in the viewer and pick `path` from its menu."""
    from blackbody.ui.actions import fill_menu
    vp = win.viewport
    q = screen_of(vp, *sel)
    rec.move(rec.where(vp, q), 0.6)
    rec.pump(0.15)
    win.doc.select(sel, force=True)
    rec.ripples.append((rec.now(), rec.cur.x(), rec.cur.y()))
    m = QMenu(vp)
    fill_menu(m, win, sel, path_mode=vp.start_path)
    rec.menu(m, rec.cur + QPointF(4, 4), path)
    return m


def save_project(doc, name):
    """Keep the recording's project (BB_SAVE_DIR/NAME.bbfire), to render it at full quality from the command line."""
    d = os.environ.get('BB_SAVE_DIR')
    if d:
        Path(d).mkdir(parents=True, exist_ok=True)
        doc.save(str(Path(d) / f'{name}.bbfire'))


def drag_tile(rec, win, key, ground, seconds=1.0):
    """Drag a building block from Create into the viewer and drop it on the ground at (x, z): the cursor makes the
    drag, then the block goes in where the app puts a dropped one (MainWindow.add_component with the ground point)."""
    vp = win.viewport
    t = tile(win, key)
    rec.move(rec.at(t), 0.6)
    rec.pump(0.12)
    rec.down = True
    rec.ripples.append((rec.now(), rec.cur.x(), rec.cur.y()))
    cs, fire, _ = vp.camstate()
    px, ok = vp._project_local(cs, fire, [(ground[0], 0.0, ground[1])])
    rec.move(rec.where(vp, vp.to_widget(px[0])), seconds)
    rec.down = False
    rec._log()
    win.add_component(key, at=tuple(ground))
    rec.pump(0.1)


def restart(rec, doc, got, n=2):
    doc.set_frame(doc.scene.start)
    doc.set_playing(True)
    rec.wait_for_frames(got, n=n)


def rec_build(app, win, got, out):
    """Build from nothing: a curtain from the card on the empty stage, moved and stretched with the gizmo, set on fire
    from its right-click menu, then wind from Create."""
    from blackbody.ui.startcard import BigTile
    doc = win.doc
    vp = win.viewport
    fresh(app, win, got)
    win.left.show_page('create')
    rec = Recorder(app, win, out)
    rec.pump(1.0)
    cloth = next(b for b in win.start_card.findChildren(BigTile) if b.label.lower().startswith('cloth'))
    rec.say('An empty stage: put something in', 1.8)
    rec.press_on(cloth, 0.8)
    doc.set_playing(False)
    doc.set_frame(doc.scene.start)
    rec.wait_for_frames(got, n=1)
    i = len(doc.scene.fabrics) - 1
    pos = doc.scene.get(('fabric', i, 'position'), doc.frame)
    view(doc, (float(pos[0]), float(pos[1]) * 0.85, float(pos[2])), 4.2, yaw=24.0, pitch=8.0)
    rec.wait_for_frames(got, n=1)
    rec.pump(0.6)
    doc.select(('fabric', i), force=True)
    rec.pump(0.3)
    rec.say('Arrows move it, squares stretch it', 2.2)
    gizmo_drag(rec, vp, 'move_x', -60)
    gizmo_drag(rec, vp, 'scale_x', 40)
    rec.wait_for_frames(got, n=1)
    rec.pump(0.4)
    rec.say('Right-click it for what it can do', 2.0)
    object_menu(rec, win, ('fabric', i), ['Set it on fire'])
    restart(rec, doc, got)
    rec.pump(6.0)
    rec.press_on(tile(win, 'wind'), 0.8)
    rec.say('Create › Forces › Wind', 1.6)
    restart(rec, doc, got)
    rec.pump(4.0)
    doc.set_playing(False)
    return rec


def rec_text(app, win, got, out):
    """Words on fire: Create › Burning text, type the words, and the letters burn."""
    from blackbody.ui import textdialog
    doc = win.doc
    fresh(app, win, got)
    real = textdialog.ask
    rec = Recorder(app, win, out)

    def ask(parent, spec=None, width=2.0, look='fire', edit=False):
        d = textdialog.TextDialog(parent, spec, width, look, 'Text')
        d.show()
        g = win.geometry()
        dg = d.geometry()
        d.move(g.x() + (g.width() - dg.width()) // 2, g.y() + (g.height() - dg.height()) // 2 - 40)
        rec.pump(0.7)
        rec.move(rec.at(d.text, 0.4, 0.5), 0.5)
        rec.click(d.text)
        d.text.selectAll()
        rec.type(d.text, os.environ.get('BB_TEXT', 'FIRE'), 0.16)
        rec.pump(1.0)
        rec.press_on(d.ok, 0.6)
        rec.pump(0.2)
        sp = d.spec()
        d.close()
        return sp
    textdialog.ask = ask
    try:
        rec.pump(0.8)
        rec.say('Create › Burning text', 1.8)
        rec.press_on(tile(win, 'text_fire'), 0.8)
    finally:
        textdialog.ask = real
    view(doc, (0.0, 0.3, 0.0), 2.3, yaw=0.0, pitch=6.0)
    restart(rec, doc, got)
    rec.pump(6.0)
    doc.set_playing(False)
    save_project(doc, 'text')
    return rec


def rec_physics(app, win, got, out):
    """Things that fall: a tower of blocks dragged in from Create, then a steel ball thrown at it."""
    doc = win.doc
    fresh(app, win, got)
    view(doc, (0.0, 0.55, 0.0), 3.4, yaw=24.0, pitch=9.0)
    settle(app, got)
    rec = Recorder(app, win, out)
    rec.pump(0.8)
    rec.say('Drag blocks in from Create › Things that fall', 2.2)
    drag_tile(rec, win, 'tower', (0.45, 0.0))
    doc.set_playing(False)
    doc.set_frame(doc.scene.start)
    rec.wait_for_frames(got, n=1)
    rec.pump(0.6)
    drag_tile(rec, win, 'cannonball', (-0.85, 0.0))
    doc.set_playing(False)
    doc.set_frame(doc.scene.start)
    rec.wait_for_frames(got, n=1)
    rec.say('A thrown steel ball', 1.4)
    rec.pump(0.8)
    restart(rec, doc, got, n=3)
    rec.pump(4.5)
    doc.set_playing(False)
    return rec


def rec_matter(app, win, got, out):
    """Sand, snow and mud: a sand pour builds a heap; right-click it and make it mud."""
    from blackbody.scene import kinds as K
    doc = win.doc
    fresh(app, win, got)
    rec = Recorder(app, win, out)
    rec.pump(0.8)
    rec.say('Create › Sand & mud', 1.8)
    rec.press_on(tile(win, 'sand_pour'), 0.8)
    doc.set(('lighting', 'sun_on'), True, merge=False)
    doc.set(('lighting', 'sun_elevation'), 30.0, merge=False)
    view(doc, (0.0, 0.35, 0.0), 2.0, yaw=26.0, pitch=10.0)
    restart(rec, doc, got, n=4)
    rec.pump(4.5)
    doc.set_playing(False)
    rec.say('Made of › Mud', 1.6)
    object_menu(rec, win, ('matter', len(K.items(doc.scene, 'matter')) - 1), ['Made of', 'Mud'])
    restart(rec, doc, got, n=4)
    rec.pump(4.5)
    doc.set_playing(False)
    return rec


# ---- into your footage: line up the ground, track the camera -------------------------------------------------------------

def court_truth():
    return json.loads(COURT.with_suffix('.json').read_text())


def load_court(app, win, got):
    doc = win.doc
    for w in QApplication.topLevelWidgets():
        if w is not win and w.isVisible() and w.isWindow():
            w.close()
    win.set_workspace('build')
    doc.load_preset('campfire', keep_shot=False)
    settle(app, got, seconds=60)
    doc.import_footage(str(COURT))
    t = time.time()
    while doc.footage_info is None and time.time() - t < 60:
        app.processEvents()
        time.sleep(0.01)
    win.set_workspace('shot')
    doc.set_frame(doc.scene.start)
    settle(app, got, seconds=40)


def rec_lineup(app, win, got, out):
    """Line up the ground: drag the grid's corners onto a paving slab, and the camera matches the footage; the campfire
    then stands on the real ground at its real size."""
    from PySide6.QtWidgets import QPushButton
    doc = win.doc
    vp = win.viewport
    load_court(app, win, got)
    truth = court_truth()
    TW, TH = truth['size']
    rec = Recorder(app, win, out)
    rec.pump(1.0)
    rec.say('Tracking › Line up the ground', 1.8)
    win.line_up_ground()
    rec.pump(1.0)
    W, H = vp.out_size()
    for k, (px, py) in enumerate(truth['slab']):
        corner = dict(vp._gm_points())[('corners', k)]
        rec.move(rec.where(vp, corner), 0.5)
        rec.pump(0.1)
        rec.press(vp)
        rec.move(rec.where(vp, vp.to_widget((px * W / TW, py * H / TH))), 0.8, widget=vp)
        rec.release(vp)
        rec.pump(0.25)
    rec.say('The lens, tilt and height come from the slab', 2.4)
    rec.pump(2.4)
    done = next(b for b in vp.groundbar.findChildren(QPushButton) if b.text() == 'Done')
    rec.press_on(done, 0.6)
    rec.wait_for_frames(got, n=1)
    rec.say('The fire stands on the real ground, at real size', 2.6)
    rec.pump(2.8)
    return rec


def rec_track(app, win, got, out):
    """Track the camera (Ctrl+T): spots followed through the footage, the move solved in metres, and the fire stays put
    on the ground as the camera travels."""
    doc = win.doc
    if not (doc.scene.ground and doc.scene.footage):   # line it up without recording, as rec_lineup does
        load_court(app, win, got)
        truth = court_truth()
        TW, TH = truth['size']
        g = {'corners': [[x / TW, y / TH] for x, y in truth['slab']], 'scale_by': 'side', 'side': 1.2, 'height': 1.55,
             'lens': 'picture', 'frame': doc.frame}
        doc.set_ground(g, merge=False)
        doc.finish_ground()
    doc.set_playing(False)
    doc.set_frame(doc.scene.start)
    settle(app, got, seconds=40)
    rec = Recorder(app, win, out)
    rec.pump(0.8)
    rec.say('Tracking › Track   (Ctrl+T)', 2.0)
    rec.pump(0.5)
    win.track()
    rec.pump(2.5)
    t0 = time.perf_counter()
    while not (doc.scene.ground or {}).get('tracked') and time.perf_counter() - t0 < 600:   # cut the rest of the wait
        app.processEvents()
        time.sleep(0.01)
    rec.cut += time.perf_counter() - t0
    rec.pump(0.4)
    rec.say(win.msg.text()[:70], 2.6)
    restart(rec, doc, got)
    rec.pump(6.0)
    doc.set_playing(False)
    save_project(doc, 'track')
    return rec


# ---- getting about: search, repeat, effects ---------------------------------------------------------------------------------

def rec_search(app, win, got, out):
    """Search everything (Ctrl+K): type what to do and press Enter."""
    from blackbody.ui import palette
    doc = win.doc
    fresh(app, win, got)
    win.add_component('box')
    doc.set_playing(False)
    doc.set(('collider', 0, 'position'), (0.0, 0.6, 0.0), merge=False)
    view(doc, (0.0, 0.4, 0.0), 3.0, yaw=24.0, pitch=10.0)
    doc.select(('collider', 0), force=True)
    settle(app, got)
    rec = Recorder(app, win, out)
    rec.pump(0.8)
    for words, caption in (('make it fall', 'Ctrl+K   search everything'), ('set it on fire', None)):
        if caption:
            rec.say(caption, 1.8)
        doc.select(('collider', 0), force=True)
        pal = palette.show(win)
        rec.pump(0.5)
        rec.type(pal.box, words, 0.08)
        rec.pump(0.7)
        rec.key(pal.box, Qt.Key_Return)
        rec.pump(0.2)
        restart(rec, doc, got)
        rec.pump(2.8)
    rec.pump(1.5)
    doc.set_playing(False)
    return rec


def rec_repeat(app, win, got, out):
    """Repeat: a torch, then right-click › Repeat… in a ring of eight."""
    from PySide6.QtWidgets import QPushButton
    from blackbody.ui import repeatdialog
    doc = win.doc
    fresh(app, win, got)
    win.add_component('torch')
    doc.set_playing(False)
    view(doc, (0.0, 0.6, 0.0), 4.4, yaw=24.0, pitch=26.0)
    settle(app, got)
    real = repeatdialog.ask
    rec = Recorder(app, win, out)

    def ask(parent, n_things, size_xz):
        d = repeatdialog.RepeatDialog(parent, n_things, size_xz)
        d.show()
        g = win.geometry()
        dg = d.geometry()
        d.move(g.x() + (g.width() - dg.width()) // 2, g.y() + (g.height() - dg.height()) // 2 - 40)
        rec.pump(0.6)
        ring = next(b for b in d.pattern.buttons() if b.property('key') == 'ring')
        rec.press_on(ring, 0.5)
        ring.setChecked(True)
        rec.pump(0.4)
        rec.move(rec.at(d.count, 0.5, 0.5), 0.5)
        rec.click(d.count)
        d.count.setValue(8)
        rec.pump(0.8)
        ok = next(b for b in d.findChildren(QPushButton) if b.text() == 'Repeat')
        rec.press_on(ok, 0.5)
        places = d.places()
        d.close()
        return places
    repeatdialog.ask = ask
    try:
        rec.pump(0.6)
        kind = 'collider' if doc.scene.colliders else 'emitter'
        object_menu(rec, win, (kind, 0), ['Repeat'])
    finally:
        repeatdialog.ask = real
    restart(rec, doc, got)
    rec.pump(4.0)
    doc.set_playing(False)
    return rec


def rec_effects(app, win, got, out):
    """The Effects tab: pick a chip, click an effect, and it simulates live."""
    from blackbody.ui.library import CATEGORIES
    doc = win.doc
    fresh(app, win, got)
    win.left.show_page('effects')
    lib = win.library
    settle(app, got)
    rec = Recorder(app, win, out)
    rec.pump(0.8)
    for chip, key, hold in (('Falling & breaking', 'wrecking_ball', 3.6), ('Sand, snow & mud', 'sand_hopper', 3.4),
                            ('Fire', 'campfire', 3.0)):
        b = lib.chips[next(k for k, lab in CATEGORIES if lab == chip)]
        rec.press_on(b, 0.6)
        rec.pump(0.4)
        it, p = lib_item_point(rec, win, key)
        rec.move(p, 0.7)
        rec.pump(0.2)
        rec.double_click(lib.list.viewport())
        restart(rec, doc, got)
        rec.pump(hold)
    doc.set_playing(False)
    return rec


RECORDINGS = {'library': rec_library, 'place': rec_place, 'scrub': rec_scrub, 'views': rec_views,
              'track2d': rec_track2d, 'liquid': rec_liquid, 'render': rec_render, 'create': rec_create,
              'build': rec_build, 'text': rec_text, 'physics': rec_physics, 'matter': rec_matter, 'lineup': rec_lineup,
              'track': rec_track, 'search': rec_search, 'repeat': rec_repeat, 'effects': rec_effects}


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
