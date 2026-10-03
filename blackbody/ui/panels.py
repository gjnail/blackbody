"""Properties: a side bar of pages (Essentials, the objects in the scene, every section of settings),
a search over all settings, and the settings of the page shown."""
from __future__ import annotations

import dataclasses
from pathlib import Path

from PySide6.QtCore import QRectF, QSettings, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import (QAbstractButton, QAbstractItemView, QFrame, QHBoxLayout, QLabel, QLineEdit, QMenu, QPushButton, QScrollArea,
                               QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..scene import kinds as K
from ..scene import presets
from ..scene.params import (COLLIDER_PARAMS, EMITTER_PARAMS, FABRIC_PARAMS, LIGHT_PARAMS, SECTION_TITLES, SECTIONS, applies,
                            param)
from ..scene.params import MATTER_MATERIALS, MATTER_PARAMS, SHOT_PARAMS, SHOT_ROUNDS, STRAND_KINDS, STRAND_PARAMS
from . import essentials, icons, theme
from .params import Group, ParamPanel, ParamRow, Switch, pick_mesh_file

PRESET_ASSETS = Path(__file__).resolve().parents[1] / 'assets' / 'presets'

SHAPE_NAMES = {'sphere': 'Sphere', 'box': 'Box', 'cylinder': 'Disc', 'capsule': 'Line', 'ring': 'Ring', 'cone': 'Cone', 'mesh': 'Mesh',
               'volume': 'Volume (VDB)'}
COLLIDER_SHAPES = {'box': 'Box', 'sphere': 'Sphere', 'cylinder': 'Cylinder', 'mesh': 'Mesh'}
SECTION_ORDER = ['combustion', 'motion', 'domain', 'shading', 'lighting', 'lume', 'embers', 'spread', 'camera', 'composite', 'render']
SECTION_HINTS = {
    'combustion': 'How fuel burns: flame height, heat and smoke.',
    'motion': 'Forces on the gas: buoyancy, swirl, turbulence and wind.',
    'domain': 'The simulation box: size, resolution, boundaries and time.',
    'shading': 'How flame and smoke look: colour temperature, brightness, smoke colour.',
    'lighting': 'Light on the smoke: ambient, a key light and the fire itself.',
    'embers': 'Sparks and embers thrown by the fire: launch, fall, bounce and glow.',
    'spread': 'Fire that spreads by itself over the ground and over colliders marked Burnable.',
    'camera': 'Lens and placement of the effect in the frame.',
    'composite': 'How the effect sits in your footage: haze, glow, light cast, grain.',
    'render': 'Frame size, frame range and final render quality.',
    'liquid': 'How the liquid behaves: gravity, splashiness, surface tension, grip, whitewater.',
    'water': 'How the liquid looks: colour, clarity, the surface, foam and spray, the wet ground.',
    'lava': 'Lava poured by Lava emitters: how thick it is and how it cools, what happens where it meets the water and the air, and how it glows.',
    'weather': 'Snow, hail, sleet, freezing rain and rain falling on the scene: what the sky makes of it on the way down, the wind, and what builds up on the ground.',
    'atmosphere': 'The sky simulated: the air column (temperature, humidity and wind with height), what warms the ground and sets off thermals, and how its clouds rain, snow and hail.',
    'sky': 'How the clouds and the sky look: their brightness and silver lining, the haze of distance, the sky and the ground.',
    'lume': 'Lume, the path-traced lighting engine: light that bounces from surface to surface, true soft shadows, glass '
            'that bends light, the fire lighting the set from every flame.',
}
LIQUID_SECTION_ORDER = ['liquid', 'domain', 'water', 'weather', 'lighting', 'lume', 'camera', 'composite', 'render']
LIQUID_HINTS = {
    'domain': 'The simulation box: size, resolution, boundaries and time. Liquids need finer grids than fire: keep the box tight around the action.',
    'lighting': 'Light on the liquid: the ambient (sky) colour it reflects and a key light (sun) for glints and shadows.',
    'camera': 'Lens and placement of the liquid in the frame.',
    'composite': 'How the liquid sits in your footage: footage colour space, glow on glints, grain.',
}


# Settings shared with fire whose wording is about fire: what they mean for a liquid.
LIQUID_WORDING = {
    ('emitter', 'start'): ('Starts at', 'Seconds from the first frame when the source starts pouring (a volume source fills '
                                        'then). -100 means from the start of the pre-roll.'),
    ('emitter', 'stop'): ('Stops at', 'Seconds from the first frame when the stream stops. -1 pours forever.'),
    ('emitter', 'velocity'): ('Velocity', 'Speed and direction the liquid leaves the source with. With a stream, the flow is '
                                          'this speed times the area of the source.'),
    ('emitter', 'vel_blend'): ('Nozzle strength', 'How firmly the source holds the liquid inside it at its velocity, as a '
                                                  'nozzle does. 1 for a jet or a tap.'),
    ('emitter', 'radial'): ('Burst speed', 'Outward speed from the source centre: a bursting water balloon, a thrown mass '
                                           'spreading as it flies.'),
    ('emitter', 'inherit'): ('Motion inheritance', 'How much a moving source (keyframed position) throws its liquid along '
                                                   'with it.'),
    ('collider', 'holdout'): ('In the shot', 'The object is in your footage: it hides the liquid behind it (or shows as a '
                                             'grey stand-in). Turn off for helper colliders that are not in the shot: they '
                                             'still hold the liquid but are never drawn.'),
    ('camera', 'fire_position'): ('Effect position', None),
    ('camera', 'fire_yaw'): ('Effect rotation', None),
}
# Words every kind of scene uses for the shared settings (they were written for fire)
ALL_WORDING = {('camera', 'fire_position'): ('Effect position', None), ('camera', 'fire_yaw'): ('Effect rotation', None)}


def worded(section, params, kind):
    """The parameters with liquid wording in a liquid scene (and kilometres in a cloud scene)."""
    if kind == 'cloud':
        out = []
        for p in params:
            w = CLOUD_WORDING.get((section, p.key))
            out.append(dataclasses.replace(p, label=w[0], tip=w[1] or p.tip, unit=w[2]) if w else p)
        return out
    if kind != 'liquid':
        out = []
        for p in params:
            w = ALL_WORDING.get((section, p.key))
            out.append(dataclasses.replace(p, label=w[0], tip=w[1] or p.tip) if w else p)
        return out
    out = []
    for p in params:
        w = LIQUID_WORDING.get((section, p.key))
        out.append(dataclasses.replace(p, label=w[0], tip=w[1] or p.tip) if w else p)
    return out


CLOUD_SECTION_ORDER = ['atmosphere', 'domain', 'sky', 'lighting', 'camera', 'composite', 'render']
# In a cloud scene the scene's metres are the sky's kilometres (Atmosphere › Scale 1000): (label, tip, unit)
CLOUD_WORDING = {
    ('domain', 'size_x'): ('Width', 'Width of the sky simulated (kilometres at Atmosphere › Scale 1000). Storms need 30 to 60.', 'km'),
    ('domain', 'size_y'): ('Height', 'Height of the sky simulated: up past the tropopause (12 to 16 for storms).', 'km'),
    ('domain', 'size_z'): ('Depth', 'Depth of the sky simulated.', 'km'),
    ('camera', 'distance'): ('Distance', 'Camera distance from the middle of the box.', 'km'),
    ('camera', 'target_y'): ('Look-at height', None, 'km'),
    ('camera', 'position'): ('Position', None, 'km'),
}
BOTH_SECTION_ORDER = ['combustion', 'motion', 'liquid', 'domain', 'shading', 'water', 'lava', 'weather', 'lighting', 'lume', 'embers', 'spread',
                      'camera', 'composite', 'render']


def section_order(kind):
    if kind == 'both':
        return BOTH_SECTION_ORDER
    if kind == 'cloud':
        return CLOUD_SECTION_ORDER
    return LIQUID_SECTION_ORDER if kind == 'liquid' else SECTION_ORDER


def emitter_kind(scene_kind, emitter):
    """What an emitter is for: in a fire-and-liquid scene, each one emits one or the other."""
    if scene_kind == 'both':
        return 'liquid' if emitter.get('emits') in ('liquid', 'lava') else 'fire'
    return scene_kind


def section_hint(sec, kind):
    return (LIQUID_HINTS.get(sec) if kind == 'liquid' else None) or SECTION_HINTS[sec]


def menu_label(shape, names):
    return names[shape] + ('…' if shape in ('mesh', 'volume') else '')


LIGHT_KINDS = {'point': 'Point light', 'spot': 'Spot light', 'area': 'Area light'}

# Ready-made fabrics: (label, settings). 'mesh' asks for its file.
FABRIC_KINDS = {
    'curtain': ('Curtain', dict(name='Curtain', width=1.2, height=2.0, position=(0.0, 1.2, -0.6), pins='top', material='cotton',
                                colour=(0.62, 0.12, 0.1))),
    'flag': ('Flag', dict(name='Flag', width=1.5, height=1.0, position=(0.75, 2.2, 0.0), pins='side', material='nylon',
                          colour=(0.75, 0.08, 0.06), detail=40)),
    'banner': ('Banner', dict(name='Banner', width=2.0, height=0.8, position=(0.0, 2.2, -0.8), pins='top_corners',
                              material='polyester', colour=(0.85, 0.82, 0.75))),
    'sheet': ('Sheet (falls)', dict(name='Sheet', width=1.6, height=1.6, orientation='lying', position=(0.0, 1.5, 0.0),
                                    pins='none', material='cotton', colour=(0.82, 0.8, 0.76))),
    'mesh': ('Mesh…', dict(name='Fabric', shape='mesh', position=(0.0, 0.0, 0.0), pins='none')),
}


def add_fabric(doc, kind, parent=None):
    """Add a ready-made fabric; a mesh fabric asks for its file first."""
    label, base = FABRIC_KINDS[kind]
    extra = dict(base)
    if kind == 'mesh':
        path = pick_mesh_file(parent)
        if not path:
            return
        from pathlib import Path
        extra.update(mesh=path, name=Path(path.split('#/')[0]).stem)
    doc.add_fabric(**extra)


def add_emitter(doc, shape, parent=None):
    """Add an emitter; a mesh or volume emitter asks for its file first."""
    if shape == 'mesh':
        path = pick_mesh_file(parent)
        if path:
            doc.add_emitter('mesh', mesh=path)
        return
    if shape == 'volume':
        from .params import pick_volume_file
        path = pick_volume_file(parent)
        if path:
            doc.add_emitter('volume', volume=path)
        return
    doc.add_emitter(shape)


def add_collider(doc, shape, parent=None):
    """Add a collider; a mesh collider asks for its file first."""
    if shape == 'mesh':
        path = pick_mesh_file(parent)
        if path:
            doc.add_collider('mesh', mesh=path)
        return
    doc.add_collider(shape)



OBJECT_KINDS = K.KINDS
NAV_GLYPHS = {'essentials': 'star', 'objects': 'cube', 'combustion': 'flame', 'motion': 'wind', 'shading': 'palette',
              'lighting': 'bulb', 'embers': 'sparks', 'spread': 'spread', 'domain': 'box', 'camera': 'camera',
              'composite': 'layers', 'render': 'film', 'liquid': 'drop', 'water': 'waves', 'lava': 'lava',
              'weather': 'weather', 'atmosphere': 'cloud', 'sky': 'sky'}
NAV_LABELS = {'essentials': 'Essentials', 'objects': 'Objects', 'spread': 'Spreading'}   # else the section's own title
PAGE_HINTS = {
    'objects': 'What is in the scene: sources that make the effect, objects it flows around, lights and fabric. '
               'Click one here or in the viewer to edit it.',
}


def nav_label(key):
    return NAV_LABELS.get(key) or SECTION_TITLES.get(key, key.title())


def page_title(key):
    if key == 'essentials':
        return 'Essentials'
    if key == 'objects':
        return 'Objects'
    return SECTION_TITLES.get(key, key.title())


class NavButton(QAbstractButton):
    """One entry of the Properties side bar: a glyph over a short name."""

    def __init__(self, key, parent=None):
        super().__init__(parent)
        self.key = key
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(QSize(74, 50))
        self.setToolTip(f'<b>{page_title(key)}</b><br>' + (
            'The settings that matter most, on one page.' if key == 'essentials' else
            PAGE_HINTS.get(key) or SECTION_HINTS.get(key, '')))

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(4, 2, -4, -2)
        on = self.isChecked()
        hot = self.underMouse()
        if on or hot:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(theme.FIELD if on else '#202024'))
            p.drawRoundedRect(r, 7, 7)
        if on:
            p.setBrush(QColor(theme.ACCENT))
            p.drawRoundedRect(QRectF(r.x(), r.y() + 10, 3, r.height() - 20), 1.5, 1.5)
        col = theme.ACCENT if on else (theme.TEXT if hot else theme.MUTED)
        icons.paint_glyph(p, NAV_GLYPHS.get(self.key, 'box'), QRectF(r.center().x() - 10, r.y() + 6, 20, 20), col, 1.5)
        f = QFont(self.font())
        f.setPointSizeF(7.8)
        if on:
            f.setWeight(QFont.DemiBold)
        p.setFont(f)
        p.setPen(QColor(theme.TEXT if on or hot else theme.MUTED))
        p.drawText(QRectF(r.x(), r.y() + 28, r.width(), 16), Qt.AlignHCenter | Qt.AlignTop, nav_label(self.key))


class NavBar(QWidget):
    picked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(4, 4, 2, 8)
        self.v.setSpacing(1)
        self.buttons = {}
        self.keys = []

    def set_items(self, keys):
        if keys == self.keys:
            return
        self.keys = list(keys)
        while self.v.count():
            it = self.v.takeAt(0)
            w = it.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        self.buttons = {}
        for k in keys:
            b = NavButton(k)
            b.clicked.connect(lambda _=False, k=k: self.picked.emit(k))
            self.v.addWidget(b)
            self.buttons[k] = b
            if k == 'objects':
                sep = QFrame()
                sep.setObjectName('sep')
                sep.setContentsMargins(10, 0, 10, 0)
                self.v.addSpacing(3)
                self.v.addWidget(sep)
                self.v.addSpacing(3)
        self.v.addStretch(1)

    def set_current(self, key):
        for k, b in self.buttons.items():
            b.setChecked(k == key)


def _object_lists(sc):
    return K.lists(sc)


class ObjectList(QWidget):
    """The objects in the scene, with an Add menu: click one to edit it, untick it to switch it off,
    double-click to rename, right-click for more."""

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 0, 20, 6)
        v.setSpacing(4)
        bar = QHBoxLayout()
        bar.setContentsMargins(2, 0, 0, 0)
        lab = QLabel('IN THE SCENE')
        lab.setObjectName('section')
        bar.addWidget(lab)
        bar.addStretch(1)
        self.add = QPushButton(' Add')
        self.add.setIcon(icons.glyph_icon('plus', theme.TEXT, 14))
        self.add.setToolTip('Add an emitter, collider, light or fabric to the scene')
        self.add_menu = QMenu(self.add)
        self.add_menu.aboutToShow.connect(self._fill_add_menu)
        self.add.setMenu(self.add_menu)
        bar.addWidget(self.add)
        v.addLayout(bar)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(10)
        self.tree.setRootIsDecorated(False)
        self.tree.setIconSize(QSize(16, 16))
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._menu)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)   # Ctrl and Shift select several
        self.tree.itemSelectionChanged.connect(self._picked)
        self.tree.itemChanged.connect(self._item_changed)
        self.tree.setStyleSheet(f'QTreeWidget {{ background: {theme.CARD}; border: 1px solid {theme.LINE}; border-radius: 8px; padding: 4px; }}')
        v.addWidget(self.tree)
        self.empty = QLabel('Nothing in the scene yet. Add fire, liquids, fabric and objects from <b>Create</b> (on the left), '
                            'or with <b>Add</b>.')
        self.empty.setObjectName('hint')
        self.empty.setWordWrap(True)
        v.addWidget(self.empty)
        self._building = False
        doc.sceneReplaced.connect(self.rebuild)
        doc.structureChanged.connect(self.rebuild)
        doc.selectionChanged.connect(self._sync_selection)
        doc.paramChanged.connect(lambda path: path == ('domain', 'kind') and QTimer.singleShot(0, self.rebuild))
        self.rebuild()

    def _fill_add_menu(self):
        m = self.add_menu
        m.clear()
        fill_add_menu(m, self.doc, self)

    def rebuild(self):
        self._building = True
        self.tree.clear()
        sc = self.doc.scene
        liquid = sc.kind == 'liquid'
        n = 0
        groups = tuple((k, K.TITLES[k], v) for k, v in K.lists(sc).items())
        for kind, title, items in groups:
            if not items:
                continue
            top = QTreeWidgetItem([f'{title.upper()}   {len(items)}'])
            top.setFlags(Qt.ItemIsEnabled)
            f = top.font(0)
            f.setPointSizeF(7.8)
            f.setWeight(QFont.DemiBold)
            top.setFont(0, f)
            top.setForeground(0, QColor(theme.FAINT))
            self.tree.addTopLevelItem(top)
            icon = icons.glyph_icon(kind, theme.OBJECT_COLOURS[kind], 16)
            for i, d in enumerate(items):
                it = QTreeWidgetItem([d['name']])
                ek = emitter_kind(sc.kind, d) if kind == 'emitter' else None
                if ek == 'liquid':   # a liquid or lava source
                    lava = d.get('emits') == 'lava'
                    it.setIcon(0, icons.glyph_icon('lava' if lava else 'drop', '#ff6a2a' if lava else '#6fb6ff', 16))
                else:
                    it.setIcon(0, icon)
                tip = {'emitter': f'{SHAPE_NAMES.get(d.get("shape"), d.get("shape"))} {"source" if liquid else "emitter"}',
                       'collider': f'{SHAPE_NAMES.get(d.get("shape"), d.get("shape"))} collider',
                       'light': LIGHT_KINDS.get(d.get('kind'), 'Light'),
                       'fabric': f'{str(d.get("material", "")).capitalize()} fabric',
                       'matter': (dict(MATTER_MATERIALS).get(d.get('material'), 'Matter')
                                  + (' poured' if d.get('pours') else '')),
                       'strands': dict(STRAND_KINDS).get(d.get('kind'), 'Grass'),
                       'shot': dict(SHOT_ROUNDS).get(d.get('round'), 'Gun')}[kind]
                it.setToolTip(0, tip + ' · double-click to rename, right-click for more')
                it.setData(0, Qt.UserRole, (kind, i))
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable | Qt.ItemIsUserCheckable)
                it.setCheckState(0, Qt.Checked if d['enabled'] else Qt.Unchecked)
                if not d['enabled']:
                    it.setForeground(0, QColor(theme.FAINT))
                top.addChild(it)
                n += 1
        self.tree.expandAll()
        rows = n + sum(1 for _, _, items in groups if items)
        self.tree.setFixedHeight(min(232, max(40, rows * 26 + 12)))
        self.tree.setVisible(n > 0)
        self.empty.setVisible(n == 0)
        self._building = False
        self._sync_selection(self.doc.selection)

    def _find(self, sel):
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            for j in range(top.childCount()):
                it = top.child(j)
                if it.data(0, Qt.UserRole) == sel:
                    return it
        return None

    def _sync_selection(self, sel):
        it = self._find(sel)
        self._building = True
        self.tree.clearSelection()
        if it is not None:
            self.tree.setCurrentItem(it)
            for other in self.doc.selected_objects()[1:]:
                o = self._find(other)
                if o is not None:
                    o.setSelected(True)
        self._building = False

    def _picked(self):
        if self._building:
            return
        sels = [it.data(0, Qt.UserRole) for it in self.tree.selectedItems()]
        sels = [tuple(s) for s in sels if s]
        if len(sels) > 1:
            cur = self.tree.currentItem()
            main = cur.data(0, Qt.UserRole) if cur is not None and cur.isSelected() else None
            self.doc.set_selected(sels, tuple(main) if main else sels[0])
        elif sels:
            self.doc.select(sels[0])

    def _item_changed(self, it, col):
        if self._building:
            return
        sel = it.data(0, Qt.UserRole)
        if not sel or sel[0] not in OBJECT_KINDS:
            return
        kind, i = sel
        d = _object_lists(self.doc.scene)[kind][i]
        name = it.text(0).strip()
        on = it.checkState(0) == Qt.Checked
        if name and name != d['name']:
            self.doc.rename(kind, i, name)
        elif on != d['enabled']:
            self.doc.set((kind, i, 'enabled'), on, merge=False)

    def _menu(self, pos):
        it = self.tree.itemAt(pos)
        sel = it.data(0, Qt.UserRole) if it else None
        m = QMenu(self)
        if sel and sel[0] in OBJECT_KINDS:
            from .actions import fill_menu
            win = self.window()
            fill_menu(m, win, sel, path_mode=getattr(getattr(win, 'viewport', None), 'start_path', None))
            m.addAction('Rename', lambda: self.tree.editItem(it, 0))
        else:
            fill_add_menu(m, self.doc, self)
        m.exec(self.tree.viewport().mapToGlobal(pos))


def fill_add_menu(m, doc, parent):
    win = parent.window() if parent is not None else None
    if win is not None and hasattr(win, 'focus_create'):
        a = m.addAction(icons.glyph_icon('star', theme.ACCENT, 16), 'Ready-made building blocks (Create)…', win.focus_create)
        a.setToolTip('Fire, water, objects, fabric and lights already set up, in the Create panel on the left')
        m.addSeparator()
    sub = m.addMenu(icons.glyph_icon('emitter', theme.OBJECT_COLOURS['emitter'], 16), 'Source (a plain shape)')
    for shape in SHAPE_NAMES:
        sub.addAction(menu_label(shape, SHAPE_NAMES), lambda s=shape: add_emitter(doc, s, parent))
    subc = m.addMenu(icons.glyph_icon('collider', theme.OBJECT_COLOURS['collider'], 16), 'Collider')
    for shape in COLLIDER_SHAPES:
        subc.addAction(menu_label(shape, COLLIDER_SHAPES), lambda s=shape: add_collider(doc, s, parent))
    subl = m.addMenu(icons.glyph_icon('light', theme.OBJECT_COLOURS['light'], 16), 'Light')
    for k, label in LIGHT_KINDS.items():
        subl.addAction(label, lambda k=k: doc.add_light(k))
    subf = m.addMenu(icons.glyph_icon('fabric', theme.OBJECT_COLOURS['fabric'], 16), 'Fabric')
    for k, (label, _) in FABRIC_KINDS.items():
        subf.addAction(label, lambda k=k: add_fabric(doc, k, parent))
    if hasattr(doc.scene, 'add_strands'):
        subg = m.addMenu(icons.glyph_icon('grass', theme.OBJECT_COLOURS['strands'], 16), 'Grass')
        for k, label in STRAND_KINDS:
            subg.addAction(label, lambda k=k: doc.add_strands(kind=k))
    if hasattr(doc.scene, 'add_shot'):
        subs = m.addMenu(icons.glyph_icon('shot', theme.OBJECT_COLOURS['shot'], 16), 'Gun')
        for k, label in SHOT_ROUNDS:
            subs.addAction(label, lambda k=k: doc.add_shot(round=k))
    if hasattr(doc.scene, 'add_matter'):
        subm = m.addMenu(icons.glyph_icon('matter', theme.OBJECT_COLOURS['matter'], 16), 'Sand, snow & mud')
        for k, label in MATTER_MATERIALS:
            subm.addAction(label, lambda k=k: doc.add_matter(material=k, shape='pile' if k in ('sand', 'wet_sand', 'mud') else 'box',
                                                             position=(0.0, 0.2, 0.0), size=(0.2, 0.2, 0.2)))


def object_menu(m, doc, sel, rename=None):
    kind, i = sel
    if rename is not None:
        m.addAction('Rename', rename)
    if kind == 'emitter':
        m.addAction(icons.glyph_icon('copy', theme.TEXT, 16), 'Duplicate', lambda: doc.duplicate_emitter(i))
        m.addAction(icons.glyph_icon('trash', theme.TEXT, 16), 'Delete', lambda: doc.remove_emitter(i))
    elif kind == 'fabric':
        m.addAction(icons.glyph_icon('copy', theme.TEXT, 16), 'Duplicate', lambda: doc.duplicate_fabric(i))
        m.addAction(icons.glyph_icon('trash', theme.TEXT, 16), 'Delete', lambda: doc.remove_fabric(i))
    elif kind == 'collider':
        m.addAction(icons.glyph_icon('trash', theme.TEXT, 16), 'Delete', lambda: doc.remove_collider(i))
    elif kind == 'light':
        m.addAction(icons.glyph_icon('trash', theme.TEXT, 16), 'Delete', lambda: doc.remove_light(i))
    elif kind == 'matter':
        m.addAction(icons.glyph_icon('trash', theme.TEXT, 16), 'Delete', lambda: doc.remove_matter(i))
    elif kind == 'strands':
        m.addAction(icons.glyph_icon('trash', theme.TEXT, 16), 'Delete', lambda: doc.remove_strands(i))
    elif kind == 'shot':
        m.addAction(icons.glyph_icon('trash', theme.TEXT, 16), 'Delete', lambda: doc.remove_shot(i))


class RowsPage(QWidget):
    """A page of setting cards (Essentials, search results) that refreshes like a ParamPanel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(10, 4, 10, 12)
        self.v.setSpacing(8)

    def add_group(self, title, items, doc, memo=None):
        """items: [(path, Param, context)]"""
        g = Group(title, memo=memo)
        for path, spec, ctx in items:
            row = ParamRow(doc, path, spec, context=ctx)
            g.lay.addWidget(row)
            self.rows.append(row)
        self.v.addWidget(g)
        return g

    def refresh(self, path=None):
        for r in self.rows:
            if path is None or r.path == path:
                r.refresh()

    def refresh_animated(self):
        for r in self.rows:
            if r.key is not None:
                r.refresh()


class Inspector(QWidget):
    """Properties: a side bar of pages (Essentials, the objects in the scene, then every section of
    settings), a search over all of them, and the page itself."""

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.page = 'essentials'
        self.panel = None
        self._rebuild_pending = False
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        top = QHBoxLayout()
        top.setContentsMargins(10, 10, 10, 8)
        top.setSpacing(6)
        self.search = QLineEdit()
        self.search.setObjectName('search')
        self.search.setPlaceholderText('Search all settings…   (Ctrl+F)')
        self.search.setClearButtonEnabled(True)
        self.search.addAction(icons.glyph_icon('search', theme.FAINT, 16), QLineEdit.LeadingPosition)
        self.search.textChanged.connect(self._search_changed)
        top.addWidget(self.search, 1)
        self.adv = QToolButton()
        self.adv.setObjectName('toggle')
        self.adv.setText('Advanced')
        self.adv.setCheckable(True)
        self.adv.setToolTip('Also show expert settings (solver accuracy, limits, fine detail)')
        self.adv.setChecked(QSettings().value('ui/advanced', False, type=bool))
        self.adv.toggled.connect(self._toggle_adv)
        top.addWidget(self.adv)
        v.addLayout(top)
        line = QFrame()
        line.setObjectName('sep')
        v.addWidget(line)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self.nav = NavBar()
        self.nav.picked.connect(self.set_page)
        nav_scroll = QScrollArea()
        nav_scroll.setWidget(self.nav)
        nav_scroll.setWidgetResizable(True)
        nav_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        nav_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        nav_scroll.setFixedWidth(82)
        nav_scroll.setStyleSheet(f'QScrollArea {{ background: {theme.BG}; border-right: 1px solid {theme.LINE}; }}'
                                 'QScrollBar:vertical { width: 4px; }')
        self.nav.setStyleSheet(f'background: {theme.BG};')
        body.addWidget(nav_scroll)

        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        head = QVBoxLayout()
        head.setContentsMargins(14, 10, 14, 6)
        head.setSpacing(2)
        self.title = QLabel()
        self.title.setObjectName('title')
        head.addWidget(self.title)
        self.hint = QLabel()
        self.hint.setObjectName('hint')
        self.hint.setWordWrap(True)
        head.addWidget(self.hint)
        col.addLayout(head)
        self.objects = ObjectList(doc)
        col.addWidget(self.objects)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        col.addWidget(self.scroll, 1)
        body.addLayout(col, 1)
        v.addLayout(body, 1)

        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(140)
        self._search_timer.timeout.connect(self.rebuild)

        doc.selectionChanged.connect(self._selection)
        doc.sceneReplaced.connect(self._scene_replaced)
        doc.structureChanged.connect(self._structure)
        doc.paramChanged.connect(self._param)
        doc.frameChanged.connect(lambda *_: self.panel is not None and self.panel.refresh_animated())
        self._update_nav()
        self.rebuild()

    # -- navigation ------------------------------------------------------------------------------------

    def nav_keys(self):
        return ['essentials', 'objects'] + [s for s in section_order(self.doc.scene.kind) if applies(s, None, self.doc.scene.kind)]

    def _update_nav(self):
        self.nav.set_items(self.nav_keys())
        self.nav.set_current(None if self.search.text().strip() else self.page)

    def set_page(self, page):
        """Show a page (from the side bar). Sections and objects also become the document's selection."""
        self._clear_search()
        if page == 'objects':
            if self.doc.selection[0] not in OBJECT_KINDS:
                for kind, items in _object_lists(self.doc.scene).items():
                    if items:
                        self.doc.select((kind, 0))
                        break
        elif page != 'essentials':
            self.doc.select(('section', page))
        self.page = page
        self._update_nav()
        self.schedule_rebuild()

    def show_section(self, sec):
        self.set_page(sec)

    def focus_search(self):
        self.search.setFocus()
        self.search.selectAll()

    def reveal(self, path):
        """Bring a setting up: its page, scrolled to it, its row lit for a moment."""
        self._clear_search()
        if path[0] in OBJECT_KINDS:
            self.doc.select((path[0], path[1]), force=True)
        elif path[0] in self.nav_keys():
            self.set_page(path[0])
        QTimer.singleShot(60, lambda: self._flash(path))

    def _flash(self, path):
        if self._rebuild_pending:
            QTimer.singleShot(40, lambda: self._flash(path))
            return
        for r in getattr(self.panel, 'rows', []):
            if tuple(r.path) == tuple(path):
                p = r.parentWidget()
                while p is not None and p is not self.panel:   # open the group it is in
                    if isinstance(p, Group) and not p.head.isChecked():
                        p.head.setChecked(True)
                    p = p.parentWidget()
                self.scroll.ensureWidgetVisible(r, 0, 80)
                r.setStyleSheet(f'ParamRow {{ background: {theme.ACCENT_DIM}; border-radius: 4px; }}')
                r.setAttribute(Qt.WA_StyledBackground, True)
                QTimer.singleShot(1200, lambda r=r: r.setStyleSheet(''))
                return

    def _selection(self, sel):
        if sel[0] in OBJECT_KINDS:
            page = 'objects'
        elif sel[0] == 'section' and sel[1] in self.nav_keys():
            page = sel[1]
        else:
            return
        self._clear_search()
        self.page = page
        self._update_nav()
        self.schedule_rebuild()

    def _scene_replaced(self):
        keys = self.nav_keys()
        if self.page not in keys:
            self.page = 'essentials'
        self._update_nav()
        self.schedule_rebuild()

    def _toggle_adv(self, on):
        QSettings().setValue('ui/advanced', bool(on))
        self.rebuild()

    def _structure(self):
        if self.page == 'objects':
            self.schedule_rebuild()

    def _param(self, path):
        if self.panel is None:
            return
        self.panel.refresh(path)
        if path == ('domain', 'kind'):
            QTimer.singleShot(0, self._scene_replaced)
        elif path and path[-1] in ('mode', 'shape', 'kind', 'liquid_mode', 'emits', 'volume_mode', 'burnable') and self.page == 'objects':
            self.schedule_rebuild()   # not inside the editor's own signal

    def _clear_search(self):
        if self.search.text():
            self._quiet = True
            self.search.clear()
            self._quiet = False

    def _search_changed(self, text):
        if getattr(self, '_quiet', False):
            return
        self.nav.set_current(None if text.strip() else self.page)
        self._search_timer.start()

    def schedule_rebuild(self):
        if not self._rebuild_pending:
            self._rebuild_pending = True
            QTimer.singleShot(0, self.rebuild)

    # -- pages -------------------------------------------------------------------------------------------

    def rebuild(self):
        self._rebuild_pending = False
        q = self.search.text().strip()
        pos = self.scroll.verticalScrollBar().value()
        same_page = getattr(self, '_built', None) == (self.page, q, self.doc.selection)
        self.objects.setVisible(not q and self.page == 'objects')
        if q:
            self._build_search(q)
        elif self.page == 'essentials':
            self._build_essentials()
        elif self.page == 'objects':
            self._build_object()
        else:
            self._build_section(self.page)
        self.scroll.setWidget(self.panel)
        if same_page:
            self.scroll.verticalScrollBar().setValue(pos)
        self._built = (self.page, q, self.doc.selection)

    def _set_head(self, title, hint):
        self.title.setText(title)
        self.hint.setText(hint)
        self.hint.setVisible(bool(hint))

    def _build_section(self, sec):
        kind = self.doc.scene.kind
        if not applies(sec, None, kind):
            sec = section_order(kind)[0]
        self._set_head(SECTION_TITLES[sec], section_hint(sec, kind))
        self.panel = ParamPanel(self.doc, lambda k, s=sec: (s, k),
                                worded(sec, [p for p in SECTIONS[sec] if applies(sec, p.key, kind)], kind), memo=sec)

    def _effect_card(self):
        """The effect that is loaded, with a way back to the effects."""
        sc = self.doc.scene
        info = presets.PRESETS.get(sc.preset) if sc.preset else None
        card = QFrame()
        card.setObjectName('card')
        h = QHBoxLayout(card)
        h.setContentsMargins(8, 8, 10, 8)
        h.setSpacing(10)
        thumb = QLabel()
        f = PRESET_ASSETS / f'{sc.preset}.png' if sc.preset else None
        pm = QPixmap(str(f)) if f is not None and f.exists() else QPixmap()
        if not pm.isNull():
            pm = pm.scaled(QSize(192, 108), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            pm.setDevicePixelRatio(2.0)
            thumb.setPixmap(pm)
            thumb.setFixedSize(96, 54)
            thumb.setStyleSheet('border-radius: 5px;')
            h.addWidget(thumb, 0, Qt.AlignTop)
        txt = QVBoxLayout()
        txt.setSpacing(1)
        name = QLabel(info['name'] if info else (sc.name or 'Untitled'))
        name.setStyleSheet('font-weight: 600; font-size: 10.5pt;')
        txt.addWidget(name)
        sub = QLabel((f'{info["category"]} · {info["size"]}' if info else 'Your scene'))
        sub.setObjectName('hint')
        txt.addWidget(sub)
        link = QLabel(f'<a href="fx" style="color:{theme.ACCENT_HI}; text-decoration:none;">Try another effect</a>')
        link.setTextInteractionFlags(Qt.LinksAccessibleByMouse)
        link.linkActivated.connect(lambda _: hasattr(self.window(), 'focus_effects') and self.window().focus_effects())
        txt.addWidget(link)
        txt.addStretch(1)
        h.addLayout(txt, 1)
        if info and info.get('blurb'):
            card.setToolTip(info['blurb'])
        return card

    def _build_essentials(self):
        sc = self.doc.scene
        self._set_head('Essentials', essentials.INTRO.get(sc.kind, '') + ' Every other one is in the sections on the left, or a search away.')
        page = RowsPage()
        page.v.addWidget(self._effect_card())
        for title, rows in essentials.groups(sc):
            page.add_group(title, [((sec, key), p, f'{SECTION_TITLES[sec]} › {param(sec, key).label}') for sec, key, p in rows],
                           self.doc, memo=f'essentials/{title}')
        page.v.addStretch(1)
        self.panel = page

    def _build_search(self, q):
        sc = self.doc.scene
        kind = sc.kind
        words = q.lower().split()
        found = []   # (score, order, group title, path, spec, context)
        n = 0
        for sec in section_order(kind):
            if not applies(sec, None, kind):
                continue
            for p in worded(sec, [p for p in SECTIONS[sec] if applies(sec, p.key, kind)], kind):
                n += 1
                s = _score(words, q.lower(), p, SECTION_TITLES[sec])
                if s is not None:
                    found.append((s, n, SECTION_TITLES[sec], (sec, p.key), p, f'{SECTION_TITLES[sec]} › {p.group}'))
        for kind_o, items in _object_lists(sc).items():
            for i, d in enumerate(items):
                for p in self._object_params(kind_o, i):
                    n += 1
                    s = _score(words, q.lower(), p, d['name'])
                    if s is not None:
                        found.append((s + 0.5, n, d['name'], (kind_o, i, p.key), p, d['name']))
        found.sort(key=lambda t: (t[0], t[1]))
        found = found[:60]
        page = RowsPage()
        if not found:
            self._set_head('Search', f'No setting matches “{q}”. Try another word: heat, smoke, haze, wind, foam, speed…')
        else:
            self._set_head('Search', f'{len(found)} setting{"s" if len(found) > 1 else ""} matching “{q}”. Edit them right here.')
            groups, order = {}, []
            for s, i, gtitle, path, p, ctx in found:
                if gtitle not in groups:
                    groups[gtitle] = []
                    order.append(gtitle)
                groups[gtitle].append((path, p, ctx))
            for gtitle in order:
                page.add_group(gtitle, groups[gtitle], self.doc)
        page.v.addStretch(1)
        self.panel = page

    def _object_params(self, kind_o, i):
        """The settings shown for one object, as in its page."""
        sc = self.doc.scene
        kind = sc.kind
        if kind_o == 'emitter':
            e = sc.emitters[i]
            ekind = emitter_kind(kind, e)
            params = [p for p in EMITTER_PARAMS if p.key not in ('name', 'enabled')
                      and (applies('emitter', p.key, ekind) or (kind == 'both' and p.key == 'emits'))]
            hide = {'end'} if e['shape'] != 'capsule' else {'yaw'}
            if e['shape'] != 'mesh':
                hide |= {'mesh', 'thickness'}
            if e['shape'] != 'volume':
                hide |= {'volume', 'volume_mode', 'volume_zup'}
            else:
                hide |= {'softness'}
                if e.get('volume_mode') in ('fill', 'hold'):
                    hide |= {'noise', 'noise_freq', 'noise_rise', 'contrast', 'seed', 'embers'}
                if e.get('volume_mode') == 'fill':
                    hide |= {'stop', 'fade_in', 'fade_out'}
            if e['shape'] not in ('mesh', 'volume'):
                hide |= {'mesh_offset'}
            if ekind == 'liquid' and e['liquid_mode'] == 'fill':
                hide |= {'flow', 'vel_blend', 'stop'}
            return worded('emitter', [p for p in params if p.key not in hide], ekind)
        if kind_o == 'collider':
            params = [p for p in COLLIDER_PARAMS if p.key not in ('name', 'enabled') and applies('collider', p.key, kind)]
            c = sc.colliders[i]
            if c['shape'] != 'mesh':
                params = [p for p in params if p.key != 'mesh']
            if c['shape'] == 'sphere':   # a ball: tipping it over changes nothing
                params = [p for p in params if p.key not in ('pitch', 'roll')]
            if c.get('joint', 'none') != 'hinge':   # a motor drives a hinge only
                params = [p for p in params if p.key not in ('motor_speed', 'motor_torque')]
            return worded('collider', params, kind)
        if kind_o == 'light':
            l = sc.lights[i]
            hide = {'name', 'enabled'}
            if l['kind'] == 'point':
                hide |= {'direction', 'cone', 'softness'}
            elif l['kind'] == 'area':
                hide |= {'cone', 'softness'}
            elif l['kind'] == 'lightning':   # a bolt: no aim or cone
                hide |= {'direction', 'cone', 'softness'}
            if l['kind'] != 'lightning':
                hide |= {p.key for p in LIGHT_PARAMS if p.group == 'Lightning'}
            return [p for p in LIGHT_PARAMS if p.key not in hide]
        if kind_o == 'shot':
            hide = {'name', 'enabled'}
            if int(sc.shots[i].get('count', 1)) <= 1:
                hide |= {'rate'}   # (one round: no rate of fire)
            return [p for p in SHOT_PARAMS if p.key not in hide]
        if kind_o == 'strands':
            gr = sc.strands[i]
            hide = {'name', 'enabled'}
            if gr.get('shape') == 'disc':
                hide |= {'yaw'}   # round: turning it changes nothing
            if not gr.get('own_colour'):
                hide |= {'colour'}
            return [p for p in STRAND_PARAMS if p.key not in hide]
        if kind_o == 'matter':
            m = sc.matter[i]
            hide = {'name', 'enabled'}
            if m.get('pours'):
                hide |= {'shape', 'yaw', 'release'}
            else:
                hide |= {'rate', 'pour_start', 'pour_stop'}
                if m.get('shape') != 'box':
                    hide |= {'yaw'}   # round: turning it changes nothing
            if not m.get('own_colour'):
                hide |= {'colour'}
            return [p for p in MATTER_PARAMS if p.key not in hide]
        f = sc.fabrics[i]
        hide = {'name', 'enabled'}
        if f['shape'] == 'mesh':
            hide |= {'width', 'height', 'orientation', 'detail'}
        else:
            hide |= {'mesh', 'scale'}
        if not f['burnable']:
            hide |= {'flammability'}
        return [p for p in FABRIC_PARAMS if p.key not in hide]

    def _build_object(self):
        sel = self.doc.selection
        sc = self.doc.scene
        kind = sc.kind
        lists = _object_lists(sc)
        if sel[0] not in OBJECT_KINDS or sel[1] >= len(lists[sel[0]]):
            self._set_head('Objects', PAGE_HINTS['objects'])
            self.panel = RowsPage()
            return
        okind, i = sel
        d = lists[okind][i]
        if okind == 'emitter':
            ekind = emitter_kind(kind, d)
            shape = SHAPE_NAMES.get(d['shape'], d['shape'])
            if ekind == 'liquid' and d.get('emits') == 'lava':
                hint = f'{shape} source · pours lava into the simulation.'
            elif ekind == 'liquid':
                hint = f'{shape} source · pours liquid into the simulation.'
            else:
                hint = f'{shape} source · releases fuel, heat, smoke or steam into the simulation. Drag it in the viewer to move it.'
        elif okind == 'collider':
            if kind == 'liquid':
                hint = 'Solid object the liquid flows around and splashes off (a wall, a rock, a step). Keyframe it to move it: it pushes the liquid out of its way.'
            elif kind == 'cloud':
                hint = 'Terrain the wind flows over (a hill, a mountain range: a heightfield mesh, sized in kilometres): air forced up its slopes cools and clouds over it.'
            else:
                hint = 'Solid object the gas flows around (a wall, a car, a log). Keyframe it to move it; mark it Burnable to let fire spread over it.'
        elif okind == 'light':
            hint = ('A light in the set: it lights the smoke and steam, and the smoke shadows it. A real light '
                    '(In the footage) is darkened on the ground where smoke blocks it; a CG light lights the ground too.')
        elif okind == 'strands':
            what = dict(STRAND_KINDS).get(d.get('kind'), 'Grass')
            hint = (f'{what}: blades that grow from the ground, bend in the wind and the fire\'s air, part round what moves '
                    'through them and, dry enough, catch and burn down to stubble. The outline is the patch and how tall it grows.')
        elif okind == 'shot':
            what = dict(SHOT_ROUNDS).get(d.get('round'), 'A gun')
            hint = (f'{what}, fired from the muzzle toward where it is aimed: each bullet flies on, holes, splinters, '
                    'chips or splashes on what it hits and pushes it. Drag the muzzle with its arrows, and the diamond '
                    'where it is aimed; the gun itself is not drawn.')
        elif okind == 'matter':
            what = dict(MATTER_MATERIALS).get(d.get('material'), 'Matter')
            hint = (f'{what}{" poured from a nozzle" if d.get("pours") else ""}: grains that pile up, slide, pack, slump '
                    'or wobble as the material does. It is pushed by and pushes on objects, water and the air. '
                    'The outline is where it starts; the render shows it as it goes.')
        else:
            hint = ('Cloth: it hangs from what holds it, drapes over objects, blows in the fire\'s air and the wind, '
                    'and if burnable catches, chars and burns through. In water it soaks; wet, it drips and steams.')
        self._set_head(d['name'], hint)
        self.panel = ParamPanel(self.doc, lambda k, okind=okind, i=i: (okind, i, k), self._object_params(okind, i),
                                header=self._object_header(okind, i, d), memo=okind)

    def _object_header(self, okind, i, d):
        """The object's card: on/off, duplicate, delete, and what it can do (actions.py) on a row that wraps."""
        w = QFrame()
        w.setObjectName('card')
        v = QVBoxLayout(w)
        v.setContentsMargins(10, 6, 6, 8)
        v.setSpacing(6)
        h = QHBoxLayout()
        h.setSpacing(6)
        v.addLayout(h)
        on = Switch()
        on.setChecked(d['enabled'])
        on.setToolTip('On or off: an object switched off takes no part in the simulation')
        on.toggled.connect(lambda val: self.doc.set((okind, i, 'enabled'), val, merge=False))
        h.addWidget(on)
        state = QLabel('On' if d['enabled'] else 'Off')
        state.setObjectName('hint')
        h.addWidget(state)
        h.addStretch(1)

        def tb(glyph, tip, fn):
            b = QToolButton()
            b.setIcon(icons.glyph_icon(glyph, theme.MUTED, 16, active=theme.TEXT))
            b.setToolTip(tip)
            b.clicked.connect(fn)
            h.addWidget(b)
        win = self.window()
        if okind in K.KINDS:
            from . import actions as A
            tb('copy', 'Duplicate', lambda: A.duplicate(win, okind, i))
        tb('trash', 'Delete', lambda: self.doc.remove_object(okind, i))
        if hasattr(win, 'doc'):
            from .actions import fill_menu, quick_actions, selected_actions
            from .library import FlowLayout
            many = selected_actions(win)
            if many:   # several things selected: what can be done to all of them, over this one's settings
                n = len(self.doc.selected_objects())
                lab = QLabel(f'<b>{n} selected</b> · the settings below are {d["name"]}’s')
                lab.setObjectName('hint')
                v.addWidget(lab)
                mrow = QWidget()
                mflow = FlowLayout(mrow, 4)
                mflow.setContentsMargins(0, 0, 0, 0)
                for label, glyph, fn in many:
                    b = QPushButton(' ' + label)
                    b.setObjectName('chip')
                    b.setIcon(icons.glyph_icon(glyph, theme.ACCENT, 14, active=theme.TEXT))
                    b.setCursor(Qt.PointingHandCursor)
                    b.clicked.connect(fn)
                    mflow.addWidget(b)
                v.addWidget(mrow)
            row = QWidget()
            flow = FlowLayout(row, 4)
            flow.setContentsMargins(0, 0, 0, 0)
            for label, glyph, fn in quick_actions(win, (okind, i)):
                b = QPushButton(' ' + label)
                b.setObjectName('chip')
                b.setIcon(icons.glyph_icon(glyph, theme.MUTED, 14, active=theme.TEXT))
                b.setCursor(Qt.PointingHandCursor)
                b.clicked.connect(fn)
                flow.addWidget(b)
            more = QPushButton('More…')
            more.setObjectName('chip')
            more.setToolTip('Everything this can do: turn into, attach to, move along a path, look at it…')
            mm = QMenu(more)
            mm.aboutToShow.connect(lambda: (mm.clear(), fill_menu(mm, win, (okind, i),
                                                                  path_mode=getattr(getattr(win, 'viewport', None), 'start_path', None))))
            more.setMenu(mm)
            flow.addWidget(more)
            v.addWidget(row)
        link = self.doc.scene.link_of(okind, d['name']) if hasattr(self.doc.scene, 'link_of') else None
        if link is not None:
            lab = QLabel(f'Attached to <b>{link["parent"][1]}</b>: it goes where that goes.')
            lab.setObjectName('hint')
            v.addWidget(lab)
        return w


def _score(words, q, p, where):
    """How well a setting matches a search (lower is better), or None."""
    label = p.label.lower()
    hay = ' '.join((label, p.group.lower(), where.lower(), p.key.replace('_', ' ')))
    if all(w in hay for w in words):
        if label.startswith(q):
            return 0
        if all(w in label for w in words):
            return 1
        return 2
    if len(q) >= 4 and all(w in (hay + ' ' + p.tip.lower()) for w in words):
        return 3
    return None


Properties = Inspector   # the old name
