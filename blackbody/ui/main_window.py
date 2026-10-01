"""Main window: a header with the main actions, the viewer in the middle, effects on the left,
properties on the right, the timeline below."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QSettings, QSize, QStandardPaths, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDockWidget, QFileDialog, QFrame, QHBoxLayout, QLabel,
                               QMainWindow, QMessageBox, QPushButton, QSizePolicy, QToolBar, QToolButton, QVBoxLayout,
                               QWidget)

import blackbody

from ..io.footage import IMAGE_EXT, VIDEO_EXT
from ..scene import PROJECT_EXT
from . import actions, icons, theme
from .document import Document
from .export_dialog import ExportDialog, RenderProgress
from .create import MIME as COMPONENT_MIME
from .create import CreatePanel, LeftPanel
from .library import Library
from .panels import Properties
from .params import guard_wheel
from .timeline import Timeline
from .usd_dialog import UsdImportDialog
from .viewport import MODE_NAMES, Viewport

VIEW_KEYS = ['composite', 'fire', 'alpha', 'emission', 'heat', 'depth', 'temperature']
FOOTAGE_FILTER = 'Footage (' + ' '.join(f'*{e}' for e in sorted(VIDEO_EXT | IMAGE_EXT)) + ');;All files (*)'
SCENE_FILTER = f'Blackbody scene (*{PROJECT_EXT})'
STATE_KEY = 'ui/state2'   # the dock layout (the panels changed: an older saved layout is not restored)


def app_data():
    p = Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation))
    p.mkdir(parents=True, exist_ok=True)
    return p


def mode_labels(kind):
    """What each view shows, in words for the kind of simulation (the views are the same passes)."""
    if kind == 'liquid':
        return {'composite': 'Composite', 'fire': 'Effect', 'alpha': 'Alpha', 'emission': 'Glow', 'heat': 'On footage',
                'depth': 'Depth', 'temperature': 'Speed'}
    if kind == 'cloud':
        return {'composite': 'Composite', 'fire': 'Effect', 'alpha': 'Alpha', 'emission': 'Emission', 'heat': 'Heat',
                'depth': 'Depth', 'temperature': 'Temp'}
    return {'composite': 'Composite', 'fire': 'Effect', 'alpha': 'Alpha', 'emission': 'Emission', 'heat': 'Heat',
            'depth': 'Depth', 'temperature': 'Temp'}


MODE_TIPS = {'composite': 'The effect in your footage', 'fire': 'The effect alone over black, with its alpha: what your editor gets',
             'alpha': 'How much the effect hides what is behind it', 'emission': 'The light the effect gives off by itself',
             'heat': 'Fire: the hot air that drives the heat haze. Liquid: what it does to the footage (wet ground, shadows, caustics)',
             'depth': 'Distance from the camera', 'temperature': 'Fire: gas temperature. Liquid: how fast it moves'}
HIDDEN_MODES = {'liquid': ('emission',), 'cloud': ('emission', 'heat', 'temperature')}


def _vsep():
    f = QFrame()
    f.setObjectName('vsep')
    f.setFixedHeight(28)
    return f


class ViewBar(QWidget):
    """The bar over the viewer. When the window is narrow the view buttons fold into a list and the
    toggles lose their words, rather than hold the window wide."""

    def __init__(self):
        super().__init__()
        self.modes = {}          # key: button
        self.hidden = ()         # views the scene has nothing for
        self.on_level = None     # called when the bar gets narrower or wider (the view names shorten)
        self.mode_combo = None
        self.texts = []          # (widget, text) shown only when there is room
        self.compact = None
        self.setMinimumWidth(400)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        w = self.width()
        level = 0 if w >= 1060 else (1 if w >= 760 else 2)   # 1: short words, toggles without; 2: views fold into a list
        if level != self.compact:
            self.compact = level
            if self.on_level is not None:
                self.on_level()
            for k, b in self.modes.items():
                b.setVisible(level < 2 and k not in self.hidden)
            self.mode_combo.setVisible(level == 2)
            compact = level >= 1
            for b, text in self.texts:
                if isinstance(b, QLabel):
                    b.setVisible(not compact)
                else:
                    b.setText('' if compact else text)
                    b.setToolButtonStyle(Qt.ToolButtonIconOnly if compact else Qt.ToolButtonTextBesideIcon)


class Welcome(QFrame):
    """First-run card: what to do, in order, with buttons for the steps that have one."""

    def __init__(self, parent, win, on_close):
        super().__init__(parent)
        self.setObjectName('welcome')
        self.setStyleSheet(f'QFrame#welcome {{ background: rgba(26,26,30,246); border: 1px solid {theme.LINE_HI}; border-radius: 12px; }}'
                           f'QLabel {{ background: transparent; border: 0; }}')
        self.setFixedWidth(410)
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 18, 20, 16)
        v.setSpacing(10)
        t = QLabel('Simulate fire, liquids, fabric and weather, and put them in your shot')
        t.setStyleSheet('font-size: 13pt; font-weight: 600;')
        t.setWordWrap(True)
        v.addWidget(t)
        steps = [
            ('Pick an effect, or make one', 'Click a ready-made one in Effects, on the left: it plays at once. Or open Create to '
             'start an empty scene and add fire, water, cloth, weather and objects to it.', ('Browse effects', win.focus_effects)),
            ('Bring in your shot', 'Import your footage, or drop a clip on the window. Frame size, frame rate and length follow it.',
             ('Import footage…', win.import_footage)),
            ('Place it', 'Drag the ring at its base onto the spot on the ground. Drag the square above it to scale. Alt-drag orbits.', None),
            ('Fine-tune', 'Essentials, on the right, has the settings that matter most. The search box finds any other.', None),
            ('Render', 'EXR, PNG, ProRes with alpha, a finished composite, or VDB volumes for 3D.', None),
        ]
        for n, (head, body, action) in enumerate(steps, 1):
            row = QHBoxLayout()
            row.setSpacing(10)
            num = QLabel(str(n))
            num.setFixedSize(22, 22)
            num.setAlignment(Qt.AlignCenter)
            num.setStyleSheet(f'background: {theme.ACCENT_DIM}; color: {theme.ACCENT_HI}; border-radius: 11px; font-weight: 700;')
            row.addWidget(num, 0, Qt.AlignTop)
            col = QVBoxLayout()
            col.setSpacing(4)
            txt = QLabel(f'<b>{head}</b><br><span style="color:{theme.MUTED}">{body}</span>')
            txt.setWordWrap(True)
            col.addWidget(txt)
            if action:
                b = QPushButton(action[0])
                b.clicked.connect(action[1])
                col.addWidget(b, 0, Qt.AlignLeft)
            row.addLayout(col, 1)
            v.addLayout(row)
        foot = QHBoxLayout()
        more = QLabel('Help › Getting started shows this again.')
        more.setObjectName('faint')
        foot.addWidget(more)
        foot.addStretch(1)
        b = QPushButton('Got it')
        b.setObjectName('primary')
        b.clicked.connect(on_close)
        foot.addWidget(b)
        v.addLayout(foot)
        self.adjustSize()


class MainWindow(QMainWindow):
    def __init__(self, worker, open_path=None):
        super().__init__()
        self.worker = worker
        self.doc = Document(worker, self)
        self.setWindowIcon(icons.app_icon())
        self.resize(1680, 980)
        self.setDockOptions(QMainWindow.AnimatedDocks)
        self.setAcceptDrops(True)

        # centre: view bar + viewport
        centre = QWidget()
        cv = QVBoxLayout(centre)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        self.viewport = Viewport(self.doc)
        cv.addWidget(self._view_bar())
        from .layerstrip import LayerStrip
        self.layer_strip = LayerStrip(self.doc)
        cv.addWidget(self.layer_strip)
        cv.addWidget(self.viewport, 1)
        self.setCentralWidget(centre)

        # panels
        self.library = Library(self.doc)
        self.library.presetChosen.connect(self._load_preset)
        self.library.userPresetChosen.connect(self._load_user_preset)
        self.create = CreatePanel(self.doc)
        self.create.newScene.connect(self.new_from_scratch)
        self.create.addComponent.connect(lambda k: self.add_component(k))
        self.left = LeftPanel(self.library, self.create)
        self.props = Properties(self.doc)
        self.outliner = self.props.objects
        self.timeline = Timeline(self.doc)
        self.timeline.keys.revealed.connect(lambda path: (self.d_props.show(), self.props.reveal(path)))
        self.d_lib = self._dock('Effects', self.left, Qt.LeftDockWidgetArea)
        self.d_lib.setMinimumWidth(250)
        self.d_props = self._dock('Properties', self.props, Qt.RightDockWidgetArea)
        self.d_props.setMinimumWidth(420)
        self.d_tl = self._dock('Timeline', self.timeline, Qt.BottomDockWidgetArea)
        self.resizeDocks([self.d_lib, self.d_props], [330, 530], Qt.Horizontal)

        self._header()
        self._menus()
        self._status()
        self._wire()
        self._shortcuts()
        s = QSettings()
        geo = s.value('ui/geometry')
        state = s.value(STATE_KEY)
        if geo is not None:
            self.restoreGeometry(geo)
        else:   # first run: most of the screen, however big it is
            scr = self.screen().availableGeometry() if self.screen() else None
            if scr is not None:
                self.resize(min(1680, int(scr.width() * 0.94)), min(1000, int(scr.height() * 0.92)))
        if state is not None:
            self.restoreState(state)
        else:
            QTimer.singleShot(0, self._size_docks)

        self.welcome = None

        QTimer.singleShot(0, self.viewport.setFocus)   # not the first combo box: keys go to the viewer
        QTimer.singleShot(0, self._sync_modes)
        self._autosave = QTimer(self)
        self._autosave.setInterval(120_000)
        self._autosave.timeout.connect(self._do_autosave)
        self._autosave.start()
        self._update_title()
        from .startcard import StartCard
        self.start_card = StartCard(self, self.viewport)
        self.start_card.hide()
        self.doc.sceneReplaced.connect(self._place_start)
        self.doc.structureChanged.connect(self._place_start)
        self.doc.paramChanged.connect(lambda *_: self._place_start())
        self.workspace = 'build'
        if not open_path:
            from ..scene import components
            self.doc.scene = components.new_scene('auto', QSettings().value('ui/new_scale', 'person'))
            self.doc.baseline = self.doc.scene.copy()
            self.doc.frame = self.doc.scene.start
            self.doc.selection = ('section', 'domain')
            self.doc.sceneReplaced.emit()
            self.left.show_page('create')
        QTimer.singleShot(0, lambda: self.set_workspace('build' if not open_path else 'shot'))
        if open_path:
            QTimer.singleShot(0, lambda: self.open_path(open_path))
        else:
            self.viewport.begin_load(self.doc.scene.name)
            self.doc.push_soon()
            worker.post('frame', self.doc.frame)
            self.doc.set_playing(True)
            QTimer.singleShot(200, self._offer_recovery)

    # -- construction ------------------------------------------------------------------------------

    def _dock(self, title, widget, area):
        d = QDockWidget(title, self)
        d.setObjectName('dock2_' + title.lower())
        d.setWidget(widget)
        d.setFeatures(QDockWidget.DockWidgetClosable)
        d.setTitleBarWidget(QWidget())   # each panel has its own heading
        self.addDockWidget(area, d)
        return d

    def _header(self):
        tb = QToolBar('Header')
        tb.setObjectName('header')
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.toggleViewAction().setVisible(False)
        tb.setStyleSheet(f'QToolBar#header {{ background: {theme.BG}; border-bottom: 1px solid {theme.LINE}; }}')
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(12, 6, 12, 6)
        h.setSpacing(6)
        logo = QLabel()
        logo.setPixmap(icons.app_icon().pixmap(24, 24))
        h.addWidget(logo)
        brand = QLabel('Blackbody')
        brand.setStyleSheet('font-size: 11pt; font-weight: 700; letter-spacing: 0.5px;')
        h.addWidget(brand)
        h.addSpacing(10)
        ws = QHBoxLayout()
        ws.setSpacing(1)
        self.ws_group = QButtonGroup(self)
        self.ws_buttons = {}
        for n, (key, label, tip) in enumerate((
                ('build', 'Build', 'Build the scene in 3D with a camera of your own: put in fire, water, cloth, weather and '
                                   'objects, move and shape them, make them interact (Tab switches)'),
                ('shot', 'Shot', 'Put the effect in your footage: the shot\u2019s camera, placement, roto, layers over the '
                                 'footage, and what you render (Tab switches)'))):
            b = QToolButton()
            b.setText(label)
            b.setObjectName('segFirst' if n == 0 else 'segLast')
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(tip)
            b.setMinimumWidth(64)
            b.setStyleSheet('QToolButton { font-weight: 600; padding: 5px 14px; }')
            b.clicked.connect(lambda _=False, k=key: self.set_workspace(k))
            self.ws_group.addButton(b)
            ws.addWidget(b)
            self.ws_buttons[key] = b
        h.addLayout(ws)
        h.addSpacing(10)
        self.scene_label = QLabel()
        self.scene_label.setObjectName('hint')
        self.scene_label.setMinimumWidth(80)
        h.addWidget(self.scene_label)
        self.kind_label = QLabel()
        self.kind_label.setObjectName('pill')
        self.kind_label.setToolTip('What this scene simulates. It follows what you add (Create), or set it in Domain › Simulation.')
        h.addWidget(self.kind_label)
        h.addSpacing(6)
        h.addWidget(_vsep())

        self._header_texts = []   # (button, text): words that go when the window is narrow (icons stay)

        def hb(glyph, text, fn, tip):
            b = QPushButton(text)
            b.setObjectName('ghost')
            b.setIcon(icons.glyph_icon(glyph, theme.MUTED, 18, active=theme.TEXT))
            b.setIconSize(QSize(18, 18))
            b.setToolTip(tip)
            b.clicked.connect(fn)
            h.addWidget(b)
            self._header_texts.append((b, text))
            return b
        hb('footage', ' Import footage', self.import_footage,
           'Bring in your clip, image sequence or still (Ctrl+I). You can also drop it on the window.')
        hb('open', ' Open', self.open_dialog, 'Open a scene (Ctrl+O)')
        hb('save', ' Save', self.save, 'Save the scene (Ctrl+S)')
        h.addWidget(_vsep())
        hb('layers', ' Layer', self.add_layer, 'Add a layer: another effect in the same shot, with its own simulation box, '
           'scale and settings, drawn in front of the others (a fire on the left and a waterfall on the right)')
        h.addWidget(_vsep())

        def ib(glyph, tip, fn):
            b = QToolButton()
            b.setIcon(icons.glyph_icon(glyph, theme.MUTED, 18, active=theme.TEXT))
            b.setIconSize(QSize(18, 18))
            b.setToolTip(tip)
            b.clicked.connect(fn)
            h.addWidget(b)
            return b
        self.undo_btn = ib('undo', 'Undo (Ctrl+Z)', self.doc.undo.undo)
        self.redo_btn = ib('redo', 'Redo (Ctrl+Y)', self.doc.undo.redo)
        self.undo_btn.setEnabled(False)
        self.redo_btn.setEnabled(False)
        self.doc.undo.canUndoChanged.connect(self.undo_btn.setEnabled)
        self.doc.undo.canRedoChanged.connect(self.redo_btn.setEnabled)
        self.doc.undo.undoTextChanged.connect(lambda t: self.undo_btn.setToolTip(f'Undo {t} (Ctrl+Z)' if t else 'Undo (Ctrl+Z)'))
        self.doc.undo.redoTextChanged.connect(lambda t: self.redo_btn.setToolTip(f'Redo {t} (Ctrl+Y)' if t else 'Redo (Ctrl+Y)'))
        h.addStretch(1)
        self.search_btn = hb('search', ' Search', self.search_everything,
                             'Search everything: what you can do to the selected thing, building blocks, commands, every setting, '
                             'ready-made effects and the things in the scene (Ctrl+K)')
        h.addSpacing(4)
        self.footage_label = QLabel('No footage')
        self.footage_label.setObjectName('pill')
        self.footage_label.setToolTip('The footage the effect is composited into')
        h.addWidget(self.footage_label)
        h.addSpacing(8)
        render = QPushButton(' Render')
        render.setObjectName('primary')
        render.setIcon(icons.glyph_icon('film', '#1b1107', 18))
        render.setIconSize(QSize(18, 18))
        render.setToolTip('Render the shot: EXR, PNG, ProRes with alpha, a composite, VDB volumes (Ctrl+M)')
        render.clicked.connect(self.render_dialog)
        h.addWidget(render)
        tb.addWidget(w)
        self.addToolBar(Qt.TopToolBarArea, tb)

    def _view_bar(self):
        bar = ViewBar()
        bar.setObjectName('viewbar')
        bar.setStyleSheet(f'QWidget#viewbar {{ background: {theme.PANEL}; border-bottom: 1px solid {theme.LINE}; }}')
        h = QHBoxLayout(bar)
        h.setContentsMargins(10, 6, 10, 6)
        h.setSpacing(6)
        seg = QHBoxLayout()
        seg.setSpacing(1)
        h.addLayout(seg)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        short = mode_labels('fire')
        for i, k in enumerate(VIEW_KEYS):
            b = QToolButton()
            b.setText(short[k])
            b.setObjectName('segFirst' if i == 0 else ('segLast' if i == len(VIEW_KEYS) - 1 else 'seg'))
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(f'{MODE_TIPS[k]}  ({i + 1})')
            b.setChecked(k == 'composite')
            b.clicked.connect(lambda _=False, k=k: self.set_mode(k))
            self.mode_group.addButton(b, i)
            seg.addWidget(b)
            bar.modes[k] = b
        self.mode_combo = QComboBox()
        for k in VIEW_KEYS:
            self.mode_combo.addItem(MODE_NAMES[k], k)
        self.mode_combo.setToolTip('What the viewer shows (1–7)')
        self.mode_combo.activated.connect(lambda i: self.set_mode(self.mode_combo.itemData(i)))
        guard_wheel(self.mode_combo)
        self.mode_combo.hide()
        bar.mode_combo = self.mode_combo
        bar.on_level = lambda: QTimer.singleShot(0, self._sync_modes)
        seg.addWidget(self.mode_combo)
        h.addStretch(1)
        ql = QLabel('Preview')
        ql.setObjectName('hint')
        h.addWidget(ql)
        bar.texts.append((ql, ''))
        self.quality = QComboBox()
        for label, q in (('Full', 1.0), ('Half', 0.5), ('Third', 0.333), ('Quarter', 0.25)):
            self.quality.addItem(label, q)
        self.quality.setCurrentIndex(QSettings().value('ui/quality_index', 1, type=int))
        self.quality.setToolTip('Resolution of the interactive preview. Frames refine to full quality when idle.')
        self.quality.currentIndexChanged.connect(self._quality)
        guard_wheel(self.quality)
        h.addWidget(self.quality)
        h.addSpacing(4)

        def toggle(glyph, text, tip, on):
            b = QToolButton()
            b.setObjectName('toggle')
            b.setText(text)
            b.setIcon(icons.glyph_icon(glyph, theme.MUTED, 16, checked=theme.TEXT))
            b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            b.setCheckable(True)
            b.setChecked(on)
            b.setToolTip(tip)
            h.addWidget(b)
            bar.texts.append((b, text))
            return b
        self.guides_btn = toggle('grid', 'Guides', 'Show the simulation box, emitters and placement handles (G)', True)
        self.guides_btn.toggled.connect(self._guides)
        self.stats_btn = toggle('stats', 'Stats', 'Show simulation statistics over the picture', QSettings().value('ui/stats', False, type=bool))
        self.stats_btn.toggled.connect(self._stats)
        self.work_btn = toggle('camera', 'Work view', 'Look around the scene with a camera of your own while you build it: drag to orbit, '
                               'middle-drag to pan, wheel to move closer, F to frame the box. The shot, its camera and the renders '
                               'do not change (W)', False)
        self.work_btn.toggled.connect(self._work_view)
        self.work_btn.hide()   # the Build / Shot switch in the header is the way to it
        self.roto_btn = toggle('roto', 'Roto', 'Draw shapes around what is in front of the effect in your footage: the effect goes '
                               'behind them. Key them over the shot as the object moves (R)', False)
        self.roto_btn.toggled.connect(self._roto)
        self.viewport.show_stats = self.stats_btn.isChecked()
        fit = QToolButton()
        fit.setIcon(icons.glyph_icon('fit', theme.MUTED, 16, active=theme.TEXT))
        fit.setToolTip('Fit the picture to the viewer (F)')
        fit.clicked.connect(lambda: self.viewport.fit())
        h.addWidget(fit)
        return bar

    def _menus(self):
        mb = self.menuBar()
        f = mb.addMenu('&File')
        new = f.addMenu('New')
        new.addAction('Empty scene', lambda: self.new_from_scratch('auto', self.create.scale.currentData()))
        new.addAction('Sky scene (clouds and storms)', lambda: self.new_from_scratch('cloud', None))
        new.addSeparator()
        new.addAction('From a ready-made effect…', self.focus_effects)
        self._act(f, 'Create (start from scratch, add building blocks)', self.focus_create, QKeySequence.New)
        self._act(f, 'Open…', self.open_dialog, QKeySequence.Open)
        self.recent_menu = f.addMenu('Open recent')
        self._fill_recent()
        f.addSeparator()
        self._act(f, 'Save', self.save, QKeySequence.Save)
        self._act(f, 'Save as…', self.save_as, QKeySequence.SaveAs)
        self._act(f, 'Save as preset…', lambda: self.library.save_current())
        self._act(f, 'Pack project (copy its files beside it)…', self.pack_project)
        f.addSeparator()
        self._act(f, 'Import footage…', self.import_footage, 'Ctrl+I')
        self._act(f, 'Remove footage', self.doc.remove_footage)
        self._act(f, 'Import camera track (.chan)…', self.import_chan)
        self._act(f, 'Import USD scene (cameras, objects)…', self.import_usd)
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
        e.addSeparator()
        self._act(e, 'Search everything…', self.search_everything, 'Ctrl+K')
        self._act(e, 'Search settings', self.focus_settings_search, 'Ctrl+F')
        self._act(e, 'Search effects', self.focus_effects, 'Ctrl+E')
        self._act(e, 'Create', self.focus_create)
        self._act(e, 'Add layer', self.add_layer, 'Ctrl+L')
        e.addSeparator()
        self._act(e, 'Select everything', self._select_all, 'Ctrl+A')
        self._act(e, 'Duplicate the selection', lambda: self._on_selection(lambda sel: self.doc.duplicate_objects(sel)), 'Ctrl+D')
        self._act(e, 'Repeat the selection…', lambda: self._on_selection(lambda sel: actions.repeat(self, sel)), 'Ctrl+R')
        self._act(e, 'Group the selection', lambda: self._on_selection(lambda sel: actions.group(self, sel)), 'Ctrl+G')
        self._act(e, 'Ungroup', lambda: self._on_selection(lambda sel: self.doc.ungroup(sel[0])), 'Ctrl+Shift+G')
        self._act(e, 'Delete the selection', lambda: self._on_selection(lambda sel: self.doc.delete_objects(sel)), 'Delete')
        e.addSeparator()
        self._act(e, 'Save the selection as a block…', self._save_block)
        self._act(e, 'Import a block…', lambda: self.create.import_block())
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
        self._act(t, 'Line up the ground (match the camera)…', self.line_up_ground, 'Ctrl+Shift+L')
        self._act(t, 'Track (the camera move, or the effect\u2019s base)', lambda: track_fire_base(self), 'Ctrl+T')
        self._act(t, 'Add a surface (a wall, a table, a ramp, stairs)…', self.add_surface)
        self._act(t, 'Clear track', lambda: clear_track(self))
        v = mb.addMenu('&View')
        for i, k in enumerate(VIEW_KEYS):
            self._act(v, MODE_NAMES[k], lambda _=False, k=k: self.set_mode(k), str(i + 1))
        v.addSeparator()
        self._act(v, 'Fit', self.viewport.fit, 'F')
        self._act(v, 'Guides', lambda: self.guides_btn.toggle(), 'G')
        self._act(v, 'Build / Shot', lambda: self.set_workspace('shot' if getattr(self, 'workspace', 'build') == 'build' else 'build'), 'W')
        cm = v.addMenu('Shot camera')
        self._act(cm, 'Use the Build view for the shot camera', lambda: self._shot_camera('use_view'))
        self._act(cm, 'Key the shot camera from the Build view', lambda: self._shot_camera('key_view'))
        self._act(cm, 'Look through the shot camera in Build', lambda: self._shot_camera('look_through'))
        self._act(v, 'Roto (shapes in front of the effect)', lambda: self.roto_btn.toggle(), 'R')
        steps = self._act(v, 'Shot steps (put it in your shot)', self._toggle_steps, checkable=True)
        steps.setChecked(QSettings().value('ui/shot_steps', True, type=bool))
        self._act(v, 'Statistics', lambda: self.stats_btn.toggle())
        v.addSeparator()
        for d in (self.d_lib, self.d_props, self.d_tl):
            a = d.toggleViewAction()
            a.setText(f'{d.windowTitle()} panel')
            v.addAction(a)
        self._act(v, 'Reset the layout', self.reset_layout)
        h = mb.addMenu('&Help')
        self._act(h, 'Getting started', self._show_welcome)
        self._act(h, 'Keyboard shortcuts', self._shortcuts_help)
        self._act(h, 'About Blackbody', self._about)

    def _toggle_steps(self, on):
        QSettings().setValue('ui/shot_steps', bool(on))
        if on and getattr(self, 'workspace', 'build') != 'shot':
            self.set_workspace('shot')
        self.viewport.sync_steps()

    def track(self):
        from .tracking import track_fire_base
        track_fire_base(self)

    def blend_settings(self):
        """Essentials, where 'Blend with the footage' is."""
        self.d_props.show()
        self.d_props.raise_()
        self.props.set_page('essentials')

    def add_surface(self, kind='wall'):
        """Line up a real surface in the footage (once the ground is lined up)."""
        if getattr(self, 'workspace', 'build') != 'shot':
            self.set_workspace('shot')
        if not self.viewport.start_surface(kind):
            self.msg.setText('Line up the ground first (Tracking › Line up the ground): surfaces are placed from it.')

    def line_up_ground(self):
        """Match the camera to the footage from the ground (in Shot)."""
        if getattr(self, 'workspace', 'build') != 'shot':
            self.set_workspace('shot')
        self.viewport.start_ground()

    def pack_project(self):
        from .packdialog import pack_project
        return pack_project(self)

    def search_everything(self):
        from . import palette
        self._palette = palette.show(self)

    def _on_selection(self, fn):
        """Do something to the selected objects (unless typing, drawing roto or a path)."""
        from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QLineEdit, QPlainTextEdit
        f = QApplication.focusWidget()
        if isinstance(f, (QLineEdit, QAbstractSpinBox, QPlainTextEdit)) or self.viewport.roto_mode or self.viewport.path_target:
            return
        sel = self.doc.selected_objects()
        if sel:
            fn(sel)

    def _select_all(self):
        from PySide6.QtWidgets import QApplication, QLineEdit
        if isinstance(QApplication.focusWidget(), QLineEdit):
            QApplication.focusWidget().selectAll()
            return
        sc = self.doc.scene
        every = [(k, i) for k, items in (('emitter', sc.emitters), ('collider', sc.colliders), ('fabric', sc.fabrics),
                                        ('light', sc.lights)) for i in range(len(items))]
        if every:
            self.doc.set_selected(every, self.doc.selection if self.doc.selection in every else every[0])
            self.msg.setText(f'All {len(every)} things selected.')

    def _save_block(self):
        from . import actions
        sel = self.doc.selected_objects() if hasattr(self.doc, 'selected_objects') else [self.doc.selection]
        actions.save_block(self, [s for s in sel if s and s[0] != 'section'])

    def _shot_camera(self, what):
        """The shot camera's Build bar, from the menu: in Shot, Build opens first."""
        if self.doc.work_view is None:
            self.set_workspace('build')
            if what != 'look_through':
                self.msg.setText('Frame the view you want in Build, then Use this view (or Key here) under the viewer.')
                return
        self.viewport.sync_camerabar()
        getattr(self.viewport.camerabar, what)()

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
        sb.setSizeGripEnabled(False)
        self.msg = QLabel('Starting the GPU engine…')
        self.msg.setContentsMargins(8, 0, 0, 0)
        self.msg.setMinimumWidth(100)
        self.msg.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        sb.addWidget(self.msg, 1)
        self.stats_label = QLabel()
        self.stats_label.setFont(theme.mono_font(8.5))
        self.stats_label.setObjectName('faint')
        sb.addPermanentWidget(self.stats_label)
        self.gpu_label = QLabel()
        self.gpu_label.setObjectName('faint')
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
        w.status.connect(self._engine_status)
        d = self.doc
        d.dirtyChanged.connect(lambda _: self._update_title())
        d.sceneReplaced.connect(self._update_title)
        d.sceneReplaced.connect(self._sync_modes)
        d.paramChanged.connect(lambda path: path == ('domain', 'kind') and (self._sync_modes(), self._update_title()))
        d.loadStarted.connect(self._load_started)

    def _shortcuts(self):
        for key, fn in ((Qt.Key_Space, lambda: self.doc.set_playing(not self.doc.playing)),
                        (Qt.Key_Left, lambda: self.doc.set_frame(self.doc.frame - 1)),
                        (Qt.Key_Right, lambda: self.doc.set_frame(self.doc.frame + 1)),
                        (Qt.Key_Home, lambda: self.doc.set_frame(self.doc.scene.start)),
                        (Qt.Key_End, lambda: self.doc.set_frame(self.doc.scene.end)),
                        (Qt.Key_Tab, lambda: self.set_workspace('shot' if getattr(self, 'workspace', 'build') == 'build' else 'build'))):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WindowShortcut)
            sc.activated.connect(lambda fn=fn: self._unless_typing(fn))

    def _unless_typing(self, fn):
        """Space and the arrow keys belong to a text field being typed in."""
        from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QLineEdit
        f = QApplication.focusWidget()
        if isinstance(f, (QLineEdit, QAbstractSpinBox)) or (isinstance(f, QComboBox) and f.isEditable()):
            return
        fn()

    # -- engine events ------------------------------------------------------------------------------------

    def _engine_ready(self, info):
        self.gpu_label.setText(f'  {info["name"]} · {info["backend"]}  ')
        self.msg.setText('Ready.')
        self.worker.post('quality', self.quality.currentData())

    def _engine_failed(self, text):
        QMessageBox.critical(self, 'GPU engine could not start',
                             'Blackbody needs a GPU with Vulkan, Direct3D 12 or Metal support.\n\n' + text[:1500])

    def _frame_ready(self, img, frame, stats):
        if stats.get('load_seq', 0) < self.doc.load_seq:
            return   # a frame of the scene this one replaced
        if self.viewport.loading:
            self.msg.setText(f'{self.viewport.loading} is ready. Space plays and pauses; drag on the timeline to scrub.')
        self.viewport.set_frame_image(img, frame, stats)
        if self.doc.playing:
            self.doc.frame_from_engine(frame)
        self.stats_label.setText(f'{stats.get("voxels", 0) / 1e6:.2f} M voxels · {stats.get("memory_mb", 0):.0f} MB · '
                                 f'cache {stats.get("cached", 0)} frames ({stats.get("cache_mb", 0):.0f} MB)  ')

    def _sim_progress(self, frac, frame):
        self.timeline.set_progress(frac, frame)
        sc = self.doc.scene
        if frac >= 1.0:
            self.viewport.set_busy('Drawing the frame', -1.0)
        elif frame < sc.start:
            pre = sc.data['domain']['preroll']
            self.viewport.set_busy(f'Pre-roll: running the simulation for {pre:g} s before the first frame', frac)
        else:
            self.viewport.set_busy(f'Simulating up to frame {self.doc.frame}  (now at {frame})', frac)

    def _engine_status(self, text, frac):
        self.viewport.set_busy(text, frac)
        if frac == -2.0:
            self.doc.set_playing(False)

    def _load_started(self, name):
        self.viewport.begin_load(name)
        self.timeline.set_cached([])
        self.msg.setText(f'Loading {name}…')

    def _footage_info(self, info):
        if info is None:
            self.footage_label.setText('No footage')
            self.footage_label.setToolTip('No footage yet: File › Import footage, or drop a clip on the window.')
        elif 'error' in info:
            self.footage_label.setText(f'Missing: {Path(info["path"]).name}')
            self.footage_label.setToolTip('Import footage again to point to it.')
            self.msg.setText(f'Could not open the footage {info["path"]}: {info["error"]}')
            box = QMessageBox(QMessageBox.Warning, 'Footage',
                              f'Could not open the footage:\n{info["path"]}\n\n{info["error"]}\n\n'
                              'Use File › Import footage to point to it again.', QMessageBox.Ok, self)
            box.setAttribute(Qt.WA_DeleteOnClose)
            box.setModal(False)
            box.show()
        else:
            self.footage_label.setText(f'{Path(info["path"]).name}')
            self.footage_label.setToolTip(f'{info["path"]}\n{info["describe"]}')
            self.msg.setText('Footage imported. The frame size, frame rate and range now follow it.')
        self.doc.footage_opened(info)
        self.viewport.update()

    # -- actions --------------------------------------------------------------------------------------------

    def _sync_modes(self):
        """The view buttons say what each view shows for this kind of simulation; views it has nothing for go."""
        kind = self.doc.scene.kind
        labels = mode_labels(kind)
        hidden = HIDDEN_MODES.get(kind, ())
        bar = self.centralWidget().layout().itemAt(0).widget()
        compact = bar.compact == 2
        short = {'Composite': 'Comp', 'Emission': 'Emis.', 'On footage': 'Ground'} if bar.compact == 1 else {}
        self.centralWidget().layout().itemAt(0).widget().hidden = hidden
        for i, k in enumerate(VIEW_KEYS):
            b = self.mode_group.button(i)
            b.setText(short.get(labels[k], labels[k]))
            b.setVisible(k not in hidden and not compact)
            self.mode_combo.setItemText(i, labels[k])
        self.viewport.mode_labels = labels
        if self.viewport.mode in hidden:
            self.set_mode('composite')
        self.viewport.update()

    def set_mode(self, k):
        i = VIEW_KEYS.index(k)
        b = self.mode_group.button(i)
        if b and not b.isChecked():
            b.setChecked(True)
        if self.mode_combo.currentIndex() != i:
            self.mode_combo.setCurrentIndex(i)
        self.viewport.mode = k
        self.worker.post('mode', k)

    def _quality(self, i):
        QSettings().setValue('ui/quality_index', i)
        self.worker.post('quality', self.quality.currentData())

    def _guides(self, on):
        self.viewport.guides = on
        self.viewport.update()

    def _stats(self, on):
        QSettings().setValue('ui/stats', bool(on))
        self.viewport.show_stats = on
        self.viewport.update()

    def set_workspace(self, ws):
        """Build: your own camera on the scene, to make things. Shot: the shot's camera over the footage."""
        self.workspace = ws
        self.ws_buttons[ws].setChecked(True)
        QSettings().setValue('ui/workspace', ws)
        if ws == 'build':
            if self.roto_btn.isChecked():
                self.roto_btn.setChecked(False)
            if not self.work_btn.isChecked():
                self.work_btn.setChecked(True)
        else:
            if self.work_btn.isChecked():
                self.work_btn.setChecked(False)
        self.roto_btn.setEnabled(ws == 'shot')
        self._place_start()
        self.viewport.sync_steps()

    def _work_view(self, on):
        if on == (self.doc.work_view is not None):
            return
        if on:
            if self.roto_btn.isChecked():
                self.roto_btn.setChecked(False)
            from .workview import WorkView
            wv = getattr(self, '_last_work_view', None)
            if wv is None or getattr(self, '_last_work_seq', None) != self.doc.load_seq:
                wv = (WorkView.behind_shot(self.doc.scene, self.doc.frame) if self.doc.scene.footage
                      else WorkView.framing(self.doc.scene))
            self.doc.set_work_view(wv)
            if getattr(self, 'workspace', 'build') != 'build':
                self.workspace = 'build'
                self.ws_buttons['build'].setChecked(True)
                self.roto_btn.setEnabled(False)
            self.msg.setText('Build: your own camera on the scene. Drag to look around, drag the arrows to move things. '
                             'Shot puts it in your footage.')
        else:
            self._last_work_view, self._last_work_seq = self.doc.work_view, self.doc.load_seq
            self.doc.set_work_view(None)
            if getattr(self, 'workspace', 'shot') != 'shot':
                self.workspace = 'shot'
                self.ws_buttons['shot'].setChecked(True)
                self.roto_btn.setEnabled(True)
            self.msg.setText('Shot: the effect through the shot\u2019s camera, over your footage.')
        self._place_start()

    def _roto(self, on):
        if on and self.work_btn.isChecked():
            self.work_btn.setChecked(False)   # roto is drawn over the footage, through the shot's camera
        self.viewport.set_roto_mode(on)
        if on:
            self.viewport.setFocus()
            self.msg.setText('Roto: click around what is in front of the effect, then click the first point to close the shape.')

    def _live(self, on):
        self.worker.post('live', on)

    def focus_effects(self):
        self.d_lib.show()
        self.d_lib.raise_()
        self.left.show_page('effects')
        self.library.focus_search()

    def add_layer(self):
        self.doc.add_layer(self.create.scale.currentData() or 'person')
        self.left.show_page('create')
        self.msg.setText('New layer, in front of the others. Load an effect into it from Effects, or build one in Create; '
                         'the strip over the viewer picks which layer you edit.')
        if not self.doc.playing:
            self.doc.set_playing(True)

    def focus_create(self):
        self.d_lib.show()
        self.d_lib.raise_()
        self.left.show_page('create')

    def new_from_scratch(self, kind, scale):
        self.doc.new_from_scratch(kind, scale or 'person')
        self.left.show_page('create')
        from ..scene.components import SCALES
        if kind == 'cloud':
            self.msg.setText('New sky scene. Ctrl+Z brings back your last scene.')
        else:
            self.msg.setText(f'New empty scene, {SCALES[scale or "person"][0].lower()}. Add fire, liquids, fabric, weather and objects '
                             'from Create; Ctrl+Z brings back your last scene.')
        self.doc.set_playing(True)

    def add_component(self, key, at=None):
        """Add a building block (from Create), at a ground point if it was dropped in the viewer."""
        from ..scene.components import get as get_block
        comp = get_block(key)
        mesh = None
        if comp.pick == 'mesh':
            from .params import pick_mesh_file
            mesh = pick_mesh_file(self)
            if not mesh:
                return
        elif comp.pick in ('text', 'shape'):
            from ..scene import components as C
            from . import shapedialog, textdialog, textmesh
            if C.target_kind(self.doc.scene, comp) is None:
                QMessageBox.information(self, comp.name, f'{comp.name} cannot go in a sky scene: skies are kilometres across. '
                                        'Start a fire or liquid scene for it.')
                return
            look = {'fire': 'fire', 'liquid': 'water'}.get(comp.need, 'solid')
            spec = (textdialog if comp.pick == 'text' else shapedialog).ask(self, None, width=C.scene_width(self.doc.scene), look=look)
            if not spec:
                return
            try:
                mesh = textmesh.make(spec)[0]
            except (ValueError, OSError) as ex:
                QMessageBox.information(self, comp.name, str(ex))
                return
        try:
            notes = self.doc.add_component(key, at=at, mesh=mesh)
        except ValueError as ex:
            QMessageBox.information(self, comp.name, str(ex))
            return
        if not comp.objects and comp.scene:
            sec = next((s for s in comp.scene if s in self.props.nav_keys()), None)
            if sec:
                self.doc.select(('section', sec), force=True)
        self.msg.setText(' '.join([f'Added {comp.name}.'] + list(notes) + (['Drag it in the viewer to move it.'] if comp.objects else [])))
        if not self.doc.playing:
            self.doc.set_playing(True)

    def focus_settings_search(self):
        self.d_props.show()
        self.d_props.raise_()
        self.props.focus_search()

    def _size_docks(self):
        """Panels sized to the window, so a laptop screen still leaves the viewer most of the room."""
        w = max(self.width(), 1000)
        self.resizeDocks([self.d_lib, self.d_props], [int(min(360, max(260, w * 0.2))), int(min(540, max(420, w * 0.3)))], Qt.Horizontal)

    def reset_layout(self):
        for d in (self.d_lib, self.d_props, self.d_tl):
            d.setFloating(False)
            d.show()
        self.addDockWidget(Qt.LeftDockWidgetArea, self.d_lib)
        self.addDockWidget(Qt.RightDockWidgetArea, self.d_props)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.d_tl)
        self._size_docks()

    def _load_preset(self, name, keep):
        self.doc.load_preset(name, keep_shot=keep)
        self.doc.set_playing(True)

    def _load_user_preset(self, path, keep):
        try:
            self.doc.load_user_preset(path, keep_shot=keep)
        except Exception as ex:
            QMessageBox.warning(self, 'Preset', f'Could not load the preset {Path(path).name}:\n{ex}')
            return
        self.doc.set_playing(True)

    def _confirm_discard(self):
        if not self.isWindowModified():
            return True
        r = QMessageBox.question(self, 'Unsaved changes', 'Save changes to this scene first?',
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            return self.save()
        return r == QMessageBox.Discard

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
        if self.doc.scene.footage:
            self.set_workspace('shot')

    def save(self):
        if not self.doc.shot.path:
            return self.save_as()
        try:
            self.doc.save()
            self._add_recent(self.doc.shot.path)
            self.msg.setText(f'Saved {Path(self.doc.shot.path).name}')
            return True
        except Exception as ex:
            QMessageBox.warning(self, 'Save', f'Could not save:\n{ex}')
            return False

    def save_as(self):
        start = self.doc.shot.path or str(Path(QSettings().value('ui/last_dir', str(Path.home()))) / f'{self.doc.shot.name}{PROJECT_EXT}')
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
            self.import_footage_path(path)

    def import_footage_path(self, path):
        QSettings().setValue('ui/footage_dir', str(Path(path).parent))
        self.msg.setText('Opening footage…')
        self.doc.import_footage(path)
        self.set_workspace('shot')

    def import_chan(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, 'Import camera track', '', 'Nuke / Blender camera (*.chan);;All files (*)')
        if not path:
            return
        try:
            from ..io.chan import apply_chan
            n = apply_chan(self.doc, path)
            self.msg.setText(f'Camera track imported: {n} frames. The camera is now in Free mode; place the fire in world space.')
        except Exception as ex:
            QMessageBox.warning(self, 'Camera track', f'Could not read the camera track:\n{ex}')

    def import_usd(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Import USD scene', QSettings().value('ui/usd_dir', ''),
                                              'USD (*.usd *.usda *.usdc *.usdz);;All files (*)')
        if not path:
            return
        QSettings().setValue('ui/usd_dir', str(Path(path).parent))
        try:
            from ..io.usd import scan
            info = scan(path)
        except Exception as ex:
            QMessageBox.warning(self, 'USD', f'Could not read {Path(path).name}:\n{ex}')
            return
        dlg = UsdImportDialog(info, self)
        if dlg.exec() != UsdImportDialog.Accepted:
            return
        cams, meshes, as_, holdout, burnable, lights = dlg.choice()
        from ..io.usd import import_usd
        report = []

        def fn(s):
            report.extend(import_usd(s, path, cameras=cams, meshes=meshes, as_=as_, holdout=holdout, burnable=burnable,
                                     lights=lights, volumes=dlg.volume_choice()))
            if info.frames and dlg.match_range.isChecked():
                s.data['render']['start'], s.data['render']['end'] = info.frames
                s.data['render']['fps'] = info.fps
        try:
            self.doc.edit('Import USD', fn, structure=True)
        except Exception as ex:
            QMessageBox.warning(self, 'USD', f'Import failed:\n{ex}')
            return
        self.doc.sceneReplaced.emit()
        self.msg.setText('Imported from USD: ' + '; '.join(report))

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
                                                'PNG image (*.png);;OpenEXR element with alpha (*.exr)')
        if not path:
            return
        self.msg.setText('Rendering the frame at final quality…')
        self.worker.post('still', {'scene': self.doc.shot.copy(), 'frame': self.doc.frame, 'path': path,
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

    # -- drag and drop --------------------------------------------------------------------------------------

    @staticmethod
    def _dropped(mime):
        """(what, path) for a file dropped on the window that Blackbody can take, else None. A building block
        dragged from Create is ('component', key)."""
        if mime.hasFormat(COMPONENT_MIME):
            return 'component', bytes(mime.data(COMPONENT_MIME)).decode()
        if not mime.hasUrls():
            return None
        for u in mime.urls():
            if not u.isLocalFile():
                continue
            p = u.toLocalFile()
            ext = Path(p).suffix.lower()
            if ext == PROJECT_EXT:
                return 'scene', p
            if ext == '.chan':
                return 'track', p
            if ext == '.bbblock':
                return 'block', p
            if ext in VIDEO_EXT or ext in IMAGE_EXT:
                return 'footage', p
        return None

    def dragEnterEvent(self, e):
        d = self._dropped(e.mimeData())
        if d is None:
            return
        e.acceptProposedAction()
        if d[0] == 'component':
            from ..scene.components import get as get_block
            self.viewport.set_drop_hint(f'Drop to add {get_block(d[1]).name} here', marker=True)
            self.viewport.set_drop_point(self._ground_at(e))
            return
        self.viewport.set_drop_hint({'scene': 'Drop to open this scene', 'track': 'Drop to import this camera track',
                                     'footage': 'Drop to use this as the footage',
                                     'block': 'Drop to add this block (it is kept under Yours in Create)'}[d[0]],
                                    marker=d[0] == 'block')
        if d[0] == 'block':
            self.viewport.set_drop_point(self._ground_at(e))

    def _ground_at(self, e):
        """The fire-local ground point (x, z) under a drop, or None if it is not over the viewer's picture. With the camera
        matched to the footage, (x, y, z): the ground, a table top, a step or a slope where it is dropped."""
        pos = self.viewport.mapFrom(self, e.position().toPoint())
        if not self.viewport.rect().contains(pos):
            return None
        sc = self.doc.scene
        if sc.ground and self.doc.work_view is None and not sc.data['camera']['use_anchor']:
            from . import surfaces
            from ..engine import camera as cam
            try:
                cs, fire, _ = self.viewport.camstate()
                W, H = self.viewport.out_size()
                fx, fy = self.viewport.to_frame(QPointF(pos))
                o, d = cam.pixel_ray(cs, fx, fy, W, H)
                w = surfaces.ground_hit(sc, o, d)
            except Exception:
                w = None
            if w is not None:
                q = surfaces.local(sc, w, self.doc.frame)
                return (float(q[0]), float(q[1]), float(q[2]))
        try:
            gp = self.viewport._ground_point(QPointF(pos), 0.0)
        except Exception:
            return None
        return None if gp is None else (float(gp[0]), float(gp[2]))

    def dragMoveEvent(self, e):
        d = self._dropped(e.mimeData())
        if d is not None:
            e.acceptProposedAction()
            if d[0] in ('component', 'block'):
                self.viewport.set_drop_point(self._ground_at(e))

    def dragLeaveEvent(self, e):
        self.viewport.set_drop_hint(None)

    def dropEvent(self, e):
        self.viewport.set_drop_hint(None)
        d = self._dropped(e.mimeData())
        if d is None:
            return
        e.acceptProposedAction()
        what, path = d
        if what == 'component':
            self.add_component(path, at=self._ground_at(e))
        elif what == 'scene':
            if self._confirm_discard():
                self.open_path(path)
        elif what == 'track':
            self.import_chan(path)
        elif what == 'block':
            key = self.create.import_block(path)
            if key:
                self.add_component(key, at=self._ground_at(e))
        else:
            self.import_footage_path(path)

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
                self.doc.shot.copy().save(app_data() / f'autosave{PROJECT_EXT}')
            except Exception:
                pass

    def _offer_recovery(self):
        p = app_data() / f'autosave{PROJECT_EXT}'
        if p.exists() and not QSettings().value('ui/clean_exit', True, type=bool):
            if QMessageBox.question(self, 'Recover', 'Blackbody did not close cleanly last time. Open the autosaved scene?') == QMessageBox.Yes:
                self.open_path(str(p))
                self.doc.shot.path = None
        QSettings().setValue('ui/clean_exit', False)

    def _update_title(self):
        name = self.doc.shot.name or 'Untitled'
        modified = not self.doc.undo.isClean()
        self.setWindowTitle(f'{name}[*] — {blackbody.APP_NAME}')
        self.setWindowModified(modified)
        saved = Path(self.doc.shot.path).name if self.doc.shot.path else 'not saved'
        from ..scene.components import KIND_BADGES
        self.scene_label.setText(f'{name}' + ('  •' if modified else ''))
        self.kind_label.setText(KIND_BADGES.get(self.doc.scene.kind, ''))
        self.scene_label.setToolTip(f'{name} · {saved}' + (' · changed since saved' if modified else ''))

    # -- help ---------------------------------------------------------------------------------------------------

    def _place_welcome(self):
        if self.welcome is not None:
            self.welcome.adjustSize()
            self.welcome.move(self.viewport.width() - self.welcome.width() - 18, 18)
            self.welcome.show()
            self.welcome.raise_()

    def _close_welcome(self):
        QSettings().setValue('ui/welcome_done', True)
        if self.welcome is not None:
            self.welcome.hide()

    def _show_welcome(self):
        if self.welcome is None:
            self.welcome = Welcome(self.viewport, self, self._close_welcome)
        self._place_welcome()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.welcome is not None and self.welcome.isVisible():
            self._place_welcome()
        self._place_start()
        narrow = self.width() < 1500   # above the full header's width, so the window can still shrink past it
        if narrow != getattr(self, '_header_narrow', None):
            self._header_narrow = narrow
            for b, text in getattr(self, '_header_texts', []):
                b.setText('' if narrow else text)

    def _place_start(self):
        """The start card, while the stage in Build is empty."""
        card = getattr(self, 'start_card', None)
        if card is None:
            return
        sc = self.doc.scene
        empty = (not (sc.emitters or sc.colliders or sc.fabrics or sc.lights) and sc.kind != 'cloud'
                 and not (sc.kind != 'fire' and (sc.data['liquid']['water_level'] > 0 or sc.data['liquid']['rain'] > 0
                                                 or sc.data['weather']['precip'] != 'none')))
        show = empty and getattr(self, 'workspace', 'build') == 'build'
        if show:
            card.adjustSize()
            vp = self.viewport
            card.move(max(8, (vp.width() - card.width()) // 2), max(8, (vp.height() - card.height()) // 2))
            card.raise_()
        card.setVisible(show)

    def _shortcuts_help(self):
        rows = [('Space', 'Play / pause'), ('Left / Right', 'Previous / next frame'), ('Home / End', 'First / last frame'),
                ('1 – 7', 'View: composite, fire, alpha, emission, heat, depth, temperature'), ('F', 'Fit the view'),
                ('G', 'Guides on/off'), ('W / Tab', 'Build / Shot'),
                ('R', 'Roto on/off: draw shapes the effect goes behind'),
                ('Ctrl+K', 'Search everything: actions, blocks, commands, settings, effects'),
                ('Ctrl+Shift+L', 'Line up the ground: match the camera to the footage'),
                ('Ctrl+T', 'Track the camera move (or the effect’s base, pinned in 2D)'),
                ('Ctrl+F', 'Search settings'), ('Ctrl+E', 'Search effects'),
                ('Ctrl+click / Shift+drag', 'Select several things'), ('Ctrl+A', 'Select everything'),
                ('Ctrl+D / Delete', 'Duplicate / delete the selection'), ('Ctrl+G / Ctrl+Shift+G', 'Group / ungroup'),
                ('Ctrl+R', 'Repeat the selection in a row, a ring or scattered'), ('Esc', 'Just the main selection'),
                ('Mouse wheel', 'Zoom the view'), ('Middle drag', 'Pan the view'),
                ('Drag the ring', 'Move the effect in the frame'), ('Drag the square', 'Scale the effect in the frame'),
                ('Drag a source, object, light or cloth', 'Move it along the ground (Shift: up and down)'), ('Alt + drag', 'Orbit the camera'),
                ('Alt + right drag', 'Camera distance'), ('Drag a number', 'Change it (Shift: fine); click it to type'),
                ('Drag a setting name', 'Scrub its value (Shift: fine)'),
                ('Double-click a setting name', 'Back to its default'), ('Ctrl+I', 'Import footage (or drop it on the window)'),
                ('Ctrl+M', 'Render'), ('Ctrl+Z / Ctrl+Y', 'Undo / redo')]
        html = '<table cellspacing="6">' + ''.join(f'<tr><td><b>{k}</b></td><td>{v}</td></tr>' for k, v in rows) + '</table>'
        QMessageBox.information(self, 'Keyboard shortcuts', html)

    def _about(self):
        gpu = self.gpu_label.text().strip()
        QMessageBox.about(self, 'About Blackbody',
                          f'<h3>{blackbody.APP_NAME} {blackbody.__version__}</h3>'
                          '<p>GPU simulation of fire, smoke, liquids, fabric and weather, for compositing into live-action footage.</p>'
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
        s.setValue(STATE_KEY, self.saveState())
        s.setValue('ui/clean_exit', True)
        self.worker.stop()
        super().closeEvent(e)
