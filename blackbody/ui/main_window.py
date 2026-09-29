"""Main window: viewer in the middle, library and outliner left, properties right, timeline below."""
from __future__ import annotations

import os
import time
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, QStandardPaths, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDockWidget, QFileDialog, QFrame, QHBoxLayout, QLabel,
                               QMainWindow, QMessageBox, QPushButton, QToolButton, QVBoxLayout, QWidget)

import blackbody

from ..io.footage import IMAGE_EXT, VIDEO_EXT
from ..scene import PROJECT_EXT, Scene, presets
from . import icons, theme
from .document import Document
from .export_dialog import ExportDialog, RenderProgress
from .library import Library
from .panels import Outliner, Properties
from .timeline import Timeline
from .viewport import MODE_NAMES, Viewport

VIEW_KEYS = ['composite', 'fire', 'alpha', 'emission', 'heat', 'depth', 'temperature']
FOOTAGE_FILTER = 'Footage (' + ' '.join(f'*{e}' for e in sorted(VIDEO_EXT | IMAGE_EXT)) + ');;All files (*)'
SCENE_FILTER = f'Blackbody scene (*{PROJECT_EXT})'


def app_data():
    p = Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation))
    p.mkdir(parents=True, exist_ok=True)
    return p


class Welcome(QFrame):
    def __init__(self, parent, on_close):
        super().__init__(parent)
        self.setStyleSheet(f'QFrame {{ background: rgba(28,28,31,235); border: 1px solid {theme.LINE}; border-radius: 6px; }}'
                           f'QLabel {{ background: transparent; border: 0; }}')
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 12, 16, 12)
        t = QLabel('Put fire or water in your shot')
        t.setStyleSheet('font-size: 12pt; font-weight: 600;')
        v.addWidget(t)
        steps = [
            ('1', 'Import your footage', 'File › Import footage (Ctrl+I). Size, frame rate and length follow it.'),
            ('2', 'Pick an effect', 'Double-click a fire or a liquid in the Library. Your shot settings stay.'),
            ('3', 'Place it', 'Drag the ring at its base onto the spot on the ground. Drag the square above it to scale. Alt-drag to orbit.'),
            ('4', 'Match it', 'Fire: exposure, haze, light on the footage. Liquid: colour, clarity, wet ground. Both: grain.'),
            ('5', 'Render', 'Render (Ctrl+M): EXR, PNG, ProRes with alpha, a finished composite, or VDB volumes.'),
        ]
        for n, head, body in steps:
            row = QHBoxLayout()
            num = QLabel(n)
            num.setFixedWidth(18)
            num.setStyleSheet(f'color: {theme.ACCENT}; font-weight: 700;')
            row.addWidget(num, 0, Qt.AlignTop)
            txt = QLabel(f'<b>{head}</b><br><span style="color:{theme.MUTED}">{body}</span>')
            txt.setWordWrap(True)
            txt.setFixedWidth(300)
            row.addWidget(txt)
            v.addLayout(row)
        b = QPushButton('Got it')
        b.clicked.connect(on_close)
        v.addWidget(b, 0, Qt.AlignRight)
        self.adjustSize()


class MainWindow(QMainWindow):
    def __init__(self, worker, open_path=None):
        super().__init__()
        self.worker = worker
        self.doc = Document(worker, self)
        self.setWindowIcon(icons.app_icon())
        self.resize(1600, 960)
        self.setDockOptions(QMainWindow.AnimatedDocks | QMainWindow.AllowTabbedDocks)

        # centre: view bar + viewport
        centre = QWidget()
        cv = QVBoxLayout(centre)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        cv.addWidget(self._view_bar())
        self.viewport = Viewport(self.doc)
        cv.addWidget(self.viewport, 1)
        self.setCentralWidget(centre)

        # docks
        self.library = Library(self.doc)
        self.library.presetChosen.connect(self._load_preset)
        self.library.userPresetChosen.connect(self._load_user_preset)
        self.outliner = Outliner(self.doc)
        self.props = Properties(self.doc)
        self.timeline = Timeline(self.doc)
        d_lib = self._dock('Library', self.library, Qt.LeftDockWidgetArea)
        d_out = self._dock('Scene', self.outliner, Qt.LeftDockWidgetArea)
        self.tabifyDockWidget(d_lib, d_out)
        d_lib.raise_()
        d_props = self._dock('Properties', self.props, Qt.RightDockWidgetArea)
        d_props.setMinimumWidth(400)
        d_tl = self._dock('Timeline', self.timeline, Qt.BottomDockWidgetArea)
        d_tl.setTitleBarWidget(QWidget())
        self.resizeDocks([d_lib, d_props], [390, 430], Qt.Horizontal)

        self._menus()
        self._status()
        self._wire()
        self._shortcuts()
        s = QSettings()
        geo = s.value('ui/geometry')
        state = s.value('ui/state')
        if geo is not None:
            self.restoreGeometry(geo)
        if state is not None:
            self.restoreState(state)

        self.welcome = None
        if not s.value('ui/welcome_done', False, type=bool):
            self.welcome = Welcome(self.viewport, self._close_welcome)
            QTimer.singleShot(0, self._place_welcome)

        self._autosave = QTimer(self)
        self._autosave.setInterval(120_000)
        self._autosave.timeout.connect(self._do_autosave)
        self._autosave.start()
        self._update_title()
        if open_path:
            QTimer.singleShot(0, lambda: self.open_path(open_path))
        else:
            self.doc.push_soon()
            worker.post('frame', self.doc.frame)
            QTimer.singleShot(200, self._offer_recovery)

    # -- construction ------------------------------------------------------------------------------

    def _dock(self, title, widget, area):
        d = QDockWidget(title, self)
        d.setObjectName('dock_' + title.lower())
        d.setWidget(widget)
        d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(area, d)
        return d

    def _view_bar(self):
        bar = QWidget()
        bar.setStyleSheet(f'background: {theme.PANEL}; border-bottom: 1px solid {theme.LINE};')
        h = QHBoxLayout(bar)
        h.setContentsMargins(8, 4, 8, 4)
        h.setSpacing(2)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        short = {'composite': 'Comp', 'fire': 'Fire', 'alpha': 'Alpha', 'emission': 'Emission', 'heat': 'Heat',
                 'depth': 'Depth', 'temperature': 'Temp'}
        for i, k in enumerate(VIEW_KEYS):
            b = QToolButton()
            b.setText(short[k])
            b.setCheckable(True)
            b.setToolTip(f'{MODE_NAMES[k]}  ({i + 1})')
            b.setChecked(k == 'composite')
            b.clicked.connect(lambda _=False, k=k: self.set_mode(k))
            self.mode_group.addButton(b, i)
            h.addWidget(b)
        h.addSpacing(16)
        ql = QLabel('Preview')
        ql.setObjectName('hint')
        h.addWidget(ql)
        self.quality = QComboBox()
        for label, q in (('Full', 1.0), ('Half', 0.5), ('Third', 0.333), ('Quarter', 0.25)):
            self.quality.addItem(label, q)
        self.quality.setCurrentIndex(QSettings().value('ui/quality_index', 1, type=int))
        self.quality.setToolTip('Resolution of the interactive preview. Idle frames refine to full quality.')
        self.quality.currentIndexChanged.connect(self._quality)
        h.addWidget(self.quality)
        self.guides_btn = QToolButton()
        self.guides_btn.setText('Guides')
        self.guides_btn.setCheckable(True)
        self.guides_btn.setChecked(True)
        self.guides_btn.setToolTip('Show the simulation box, emitters and placement handles (G)')
        self.guides_btn.toggled.connect(self._guides)
        h.addWidget(self.guides_btn)
        h.addStretch(1)
        self.footage_label = QLabel('No footage · File › Import footage')
        self.footage_label.setObjectName('hint')
        h.addWidget(self.footage_label)
        return bar

    def _menus(self):
        mb = self.menuBar()
        f = mb.addMenu('&File')
        self._act(f, 'New', self.new_scene, QKeySequence.New)
        self._act(f, 'Open…', self.open_dialog, QKeySequence.Open)
        self.recent_menu = f.addMenu('Open recent')
        self._fill_recent()
        f.addSeparator()
        self._act(f, 'Save', self.save, QKeySequence.Save)
        self._act(f, 'Save as…', self.save_as, QKeySequence.SaveAs)
        f.addSeparator()
        self._act(f, 'Import footage…', self.import_footage, 'Ctrl+I')
        self._act(f, 'Remove footage', self.doc.remove_footage)
        self._act(f, 'Import camera track (.chan)…', self.import_chan)
        f.addSeparator()
        self._act(f, 'Render…', self.render_dialog, 'Ctrl+M')
        self._act(f, 'Export this frame…', self.export_frame, 'Ctrl+Shift+E')
        f.addSeparator()
        self._act(f, 'Quit', self.close, QKeySequence.Quit)
        e = mb.addMenu('&Edit')
        u = self.doc.undo.createUndoAction(self, 'Undo')
        u.setShortcut(QKeySequence.Undo)
        r = self.doc.undo.createRedoAction(self, 'Redo')
        r.setShortcut(QKeySequence.Redo)
        e.addAction(u)
        e.addAction(r)
        s = mb.addMenu('&Simulation')
        self._act(s, 'Play / pause', lambda: self.doc.set_playing(not self.doc.playing))
        self._act(s, 'Restart simulation', lambda: self.worker.post('restart'), 'Ctrl+Backspace')
        self._act(s, 'Cache the frame range', lambda: self.worker.post('cache_range'), 'Ctrl+Shift+C')
        s.addSeparator()
        live = self._act(s, 'Live tweaking', self._live, checkable=True)
        live.setChecked(True)
        live.setToolTip('Apply changes to the running simulation instead of restarting it')
        from .panels import COLLIDER_SHAPES, add_collider, add_emitter
        add = s.addMenu('Add emitter')
        for shape, label in (('sphere', 'Sphere'), ('cylinder', 'Disc'), ('box', 'Box'), ('capsule', 'Line'), ('ring', 'Ring'), ('cone', 'Cone'),
                             ('mesh', 'Mesh…')):
            add.addAction(label, lambda sh=shape: add_emitter(self.doc, sh, self))
        addc = s.addMenu('Add collider')
        for shape, label in COLLIDER_SHAPES.items():
            addc.addAction(label + ('…' if shape == 'mesh' else ''), lambda sh=shape: add_collider(self.doc, sh, self))
        t = mb.addMenu('&Tracking')
        from .tracking import clear_track, track_fire_base
        self._act(t, 'Track the fire base', lambda: track_fire_base(self), 'Ctrl+T')
        self._act(t, 'Clear track', lambda: clear_track(self))
        v = mb.addMenu('&View')
        for i, k in enumerate(VIEW_KEYS):
            self._act(v, MODE_NAMES[k], lambda _=False, k=k: self.set_mode(k), str(i + 1))
        v.addSeparator()
        self._act(v, 'Fit', self.viewport.fit, 'F')
        self._act(v, 'Guides', lambda: self.guides_btn.toggle(), 'G')
        h = mb.addMenu('&Help')
        self._act(h, 'Getting started', self._show_welcome)
        self._act(h, 'Keyboard shortcuts', self._shortcuts_help)
        self._act(h, 'About Blackbody', self._about)

    def _act(self, menu, text, fn, shortcut=None, checkable=False):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.setCheckable(checkable)
        a.triggered.connect(fn)
        menu.addAction(a)
        self.addAction(a)
        return a

    def _status(self):
        sb = self.statusBar()
        self.msg = QLabel('Starting the GPU engine…')
        sb.addWidget(self.msg, 1)
        self.stats_label = QLabel()
        self.stats_label.setFont(theme.mono_font(8.5))
        sb.addPermanentWidget(self.stats_label)
        self.gpu_label = QLabel()
        sb.addPermanentWidget(self.gpu_label)

    def _wire(self):
        w = self.worker
        w.ready.connect(self._engine_ready)
        w.failed.connect(self._engine_failed)
        w.frameReady.connect(self._frame_ready)
        w.simProgress.connect(self._sim_progress)
        w.cacheChanged.connect(self.timeline.set_cached)
        w.footageInfo.connect(self._footage_info)
        w.message.connect(self.msg.setText)
        w.stillDone.connect(self._still_done)
        d = self.doc
        d.dirtyChanged.connect(lambda _: self._update_title())
        d.sceneReplaced.connect(self._update_title)

    def _shortcuts(self):
        for key, fn in ((Qt.Key_Space, lambda: self.doc.set_playing(not self.doc.playing)),
                        (Qt.Key_Left, lambda: self.doc.set_frame(self.doc.frame - 1)),
                        (Qt.Key_Right, lambda: self.doc.set_frame(self.doc.frame + 1)),
                        (Qt.Key_Home, lambda: self.doc.set_frame(self.doc.scene.start)),
                        (Qt.Key_End, lambda: self.doc.set_frame(self.doc.scene.end))):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(fn)

    # -- engine events ------------------------------------------------------------------------------------

    def _engine_ready(self, info):
        self.gpu_label.setText(f'  {info["name"]} · {info["backend"]}  ')
        self.msg.setText('Ready.')
        self.worker.post('quality', self.quality.currentData())

    def _engine_failed(self, text):
        QMessageBox.critical(self, 'GPU engine could not start',
                             'Blackbody needs a GPU with Vulkan, Direct3D 12 or Metal support.\n\n' + text[:1500])

    def _frame_ready(self, img, frame, stats):
        self.viewport.sim_msg = ''
        self.viewport.set_frame_image(img, frame, stats)
        if self.doc.playing:
            self.doc.frame_from_engine(frame)
        self.stats_label.setText(f'{stats.get("voxels", 0) / 1e6:.2f} M voxels · {stats.get("memory_mb", 0):.0f} MB · '
                                 f'cache {stats.get("cached", 0)} frames ({stats.get("cache_mb", 0):.0f} MB)  ')

    def _sim_progress(self, frac, frame):
        self.timeline.set_progress(frac, frame)
        self.viewport.sim_msg = '' if frac >= 1.0 else f'simulating… frame {frame}'
        self.viewport.update()

    def _footage_info(self, info):
        if info is None:
            self.footage_label.setText('No footage · File › Import footage')
        elif 'error' in info:
            self.footage_label.setText(f'Footage missing: {Path(info["path"]).name} · File › Import footage to relink')
            self.msg.setText(f'Could not open the footage {info["path"]}: {info["error"]}')
            box = QMessageBox(QMessageBox.Warning, 'Footage',
                              f'Could not open the footage:\n{info["path"]}\n\n{info["error"]}\n\n'
                              'Use File › Import footage to point to it again.', QMessageBox.Ok, self)
            box.setAttribute(Qt.WA_DeleteOnClose)
            box.setModal(False)
            box.show()
        else:
            self.footage_label.setText(f'{Path(info["path"]).name} · {info["describe"]}')
            self.msg.setText('Footage imported. The frame size, frame rate and range now follow it.')
        self.doc.footage_opened(info)

    # -- actions --------------------------------------------------------------------------------------------

    def set_mode(self, k):
        i = VIEW_KEYS.index(k)
        b = self.mode_group.button(i)
        if b and not b.isChecked():
            b.setChecked(True)
        self.viewport.mode = k
        self.worker.post('mode', k)

    def _quality(self, i):
        QSettings().setValue('ui/quality_index', i)
        self.worker.post('quality', self.quality.currentData())

    def _guides(self, on):
        self.viewport.guides = on
        self.viewport.update()

    def _live(self, on):
        self.worker.post('live', on)

    def _load_preset(self, name, keep):
        self.doc.load_preset(name, keep_shot=keep)
        self.msg.setText(f'Loaded {presets.PRESETS[name]["name"]}.')

    def _load_user_preset(self, path, keep):
        other = Scene.load(path)

        def fn(s):
            keep_sections = ('camera', 'render', 'composite') if keep else ()
            saved = {k: dict(s.data[k]) for k in keep_sections}
            s.data = other.data
            for k, v in saved.items():
                s.data[k] = v
            s.emitters = other.emitters
            s.colliders = other.colliders
        self.doc.edit(f'Preset: {Path(path).stem}', fn, structure=True)
        self.doc.sceneReplaced.emit()

    def _confirm_discard(self):
        if not self.isWindowModified():
            return True
        r = QMessageBox.question(self, 'Unsaved changes', 'Save changes to this scene first?',
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            return self.save()
        return r == QMessageBox.Discard

    def new_scene(self):
        if self._confirm_discard():
            self.doc.new()
            self.worker.post('frame', self.doc.frame)

    def open_dialog(self):
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Open scene', QSettings().value('ui/last_dir', ''), SCENE_FILTER)
        if path:
            self.open_path(path)

    def open_path(self, path):
        try:
            self.doc.open(path)
        except Exception as ex:
            QMessageBox.warning(self, 'Open', f'Could not open {path}:\n{ex}')
            return
        self._add_recent(path)
        self.worker.post('frame', self.doc.frame)
        self.msg.setText(f'Opened {Path(path).name}')

    def save(self):
        if not self.doc.scene.path:
            return self.save_as()
        try:
            self.doc.save()
            self._add_recent(self.doc.scene.path)
            self.msg.setText(f'Saved {Path(self.doc.scene.path).name}')
            return True
        except Exception as ex:
            QMessageBox.warning(self, 'Save', f'Could not save:\n{ex}')
            return False

    def save_as(self):
        start = self.doc.scene.path or str(Path(QSettings().value('ui/last_dir', str(Path.home()))) / f'{self.doc.scene.name}{PROJECT_EXT}')
        path, _ = QFileDialog.getSaveFileName(self, 'Save scene', start, SCENE_FILTER)
        if not path:
            return False
        try:
            p = self.doc.save(path)
        except Exception as ex:
            QMessageBox.warning(self, 'Save', f'Could not save:\n{ex}')
            return False
        QSettings().setValue('ui/last_dir', str(Path(p).parent))
        self._add_recent(p)
        self._update_title()
        return True

    def import_footage(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Import footage', QSettings().value('ui/footage_dir', ''), FOOTAGE_FILTER)
        if path:
            QSettings().setValue('ui/footage_dir', str(Path(path).parent))
            self.msg.setText('Opening footage…')
            self.doc.import_footage(path)

    def import_chan(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Import camera track', '', 'Nuke / Blender camera (*.chan);;All files (*)')
        if not path:
            return
        try:
            from ..io.chan import apply_chan
            n = apply_chan(self.doc, path)
            self.msg.setText(f'Camera track imported: {n} frames. The camera is now in Free mode; place the fire in world space.')
        except Exception as ex:
            QMessageBox.warning(self, 'Camera track', f'Could not read the camera track:\n{ex}')

    def render_dialog(self):
        self.doc.set_playing(False)
        dlg = ExportDialog(self.doc, self)
        if dlg.exec() != ExportDialog.Accepted:
            return
        spec = dlg.spec()
        prog = RenderProgress(self.worker, spec, self)
        self.worker.post('job', spec)
        prog.exec()

    def export_frame(self):
        self.doc.set_playing(False)
        path, flt = QFileDialog.getSaveFileName(self, 'Export this frame', str(Path(QSettings().value('ui/last_dir', str(Path.home()))) / f'{self.doc.scene.name}_{self.doc.frame:04d}.png'),
                                                'PNG image (*.png);;OpenEXR fire element (*.exr)')
        if not path:
            return
        self.msg.setText('Rendering the frame at final quality…')
        self.worker.post('still', {'scene': self.doc.scene.copy(), 'frame': self.doc.frame, 'path': path,
                                   'mode': 'fire' if path.lower().endswith('.exr') else self.viewport.mode})

    def _still_done(self, result, err):
        if err or result is None:
            QMessageBox.warning(self, 'Export frame', f'The frame could not be rendered:\n{err}')
            return
        path = result['path']
        try:
            if path.lower().endswith('.exr'):
                from ..io.images import write_exr
                b = result['aov']['beauty']
                write_exr(path, {'R': b[..., 0], 'G': b[..., 1], 'B': b[..., 2], 'A': b[..., 3]})
            else:
                from ..io.images import write_png
                write_png(path, np.ascontiguousarray(result['display'][..., :3]))
            self.msg.setText(f'Saved {Path(path).name}')
        except Exception as ex:
            QMessageBox.warning(self, 'Export frame', f'Could not write the file:\n{ex}')

    # -- recent / autosave / title ---------------------------------------------------------------------------

    def _add_recent(self, path):
        s = QSettings()
        rec = [p for p in s.value('ui/recent', [], type=list) if p != path]
        rec.insert(0, path)
        s.setValue('ui/recent', rec[:10])
        self._fill_recent()

    def _fill_recent(self):
        self.recent_menu.clear()
        rec = QSettings().value('ui/recent', [], type=list)
        for p in rec:
            if Path(p).exists():
                self.recent_menu.addAction(Path(p).name, lambda p=p: self._confirm_discard() and self.open_path(p))
        self.recent_menu.setEnabled(bool(self.recent_menu.actions()))

    def _do_autosave(self):
        if self.isWindowModified():
            try:
                self.doc.scene.copy().save(app_data() / f'autosave{PROJECT_EXT}')
            except Exception:
                pass

    def _offer_recovery(self):
        p = app_data() / f'autosave{PROJECT_EXT}'
        if p.exists() and not QSettings().value('ui/clean_exit', True, type=bool):
            if QMessageBox.question(self, 'Recover', 'Blackbody did not close cleanly last time. Open the autosaved scene?') == QMessageBox.Yes:
                self.open_path(str(p))
                self.doc.scene.path = None
        QSettings().setValue('ui/clean_exit', False)

    def _update_title(self):
        name = self.doc.scene.name or 'Untitled'
        self.setWindowTitle(f'{name}[*] — {blackbody.APP_NAME}')
        self.setWindowModified(not self.doc.undo.isClean())

    # -- help ---------------------------------------------------------------------------------------------------

    def _place_welcome(self):
        if self.welcome is not None:
            self.welcome.move(self.viewport.width() - self.welcome.width() - 16, 16)
            self.welcome.show()
            self.welcome.raise_()

    def _close_welcome(self):
        QSettings().setValue('ui/welcome_done', True)
        if self.welcome is not None:
            self.welcome.hide()

    def _show_welcome(self):
        if self.welcome is None:
            self.welcome = Welcome(self.viewport, self._close_welcome)
        self._place_welcome()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.welcome is not None and self.welcome.isVisible():
            self._place_welcome()

    def _shortcuts_help(self):
        rows = [('Space', 'Play / pause'), ('Left / Right', 'Previous / next frame'), ('Home / End', 'First / last frame'),
                ('1 – 7', 'View: composite, fire, alpha, emission, heat, depth, temperature'), ('F', 'Fit the view'),
                ('G', 'Guides on/off'), ('Mouse wheel', 'Zoom the view'), ('Middle drag', 'Pan the view'),
                ('Drag the ring', 'Move the fire in the frame'), ('Drag the square', 'Scale the fire in the frame'),
                ('Drag an emitter', 'Move it along the ground (Shift: up and down)'), ('Alt + drag', 'Orbit the camera'),
                ('Alt + right drag', 'Camera distance'), ('Drag a setting name', 'Scrub its value (Shift: fine)'),
                ('Double-click a setting name', 'Reset it'), ('Ctrl+I', 'Import footage'), ('Ctrl+M', 'Render'),
                ('Ctrl+Z / Ctrl+Y', 'Undo / redo')]
        html = '<table cellspacing="6">' + ''.join(f'<tr><td><b>{k}</b></td><td>{v}</td></tr>' for k, v in rows) + '</table>'
        QMessageBox.information(self, 'Keyboard shortcuts', html)

    def _about(self):
        gpu = self.gpu_label.text().strip()
        QMessageBox.about(self, 'About Blackbody',
                          f'<h3>{blackbody.APP_NAME} {blackbody.__version__}</h3>'
                          '<p>GPU fire, smoke and liquid simulation for compositing into live-action footage.</p>'
                          f'<p>Running on: {gpu}</p>'
                          '<p>Pyro solver on a MAC grid with multigrid pressure, physically based blackbody '
                          'rendering, a FLIP liquid solver with ray-traced water and whitewater, '
                          'OpenEXR, ProRes and OpenVDB output.</p>')

    def closeEvent(self, e):
        if not self._confirm_discard():
            e.ignore()
            return
        s = QSettings()
        s.setValue('ui/geometry', self.saveGeometry())
        s.setValue('ui/state', self.saveState())
        s.setValue('ui/clean_exit', True)
        self.worker.stop()
        super().closeEvent(e)
