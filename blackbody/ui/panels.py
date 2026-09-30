"""Outliner (what is in the scene) and Properties (the settings of the selected item)."""
from __future__ import annotations

import dataclasses

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QLineEdit, QMenu, QPushButton, QScrollArea, QToolButton,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..scene.params import COLLIDER_PARAMS, EMITTER_PARAMS, SECTION_TITLES, SECTIONS, applies
from . import theme
from .params import ParamPanel, pick_mesh_file

SHAPE_NAMES = {'sphere': 'Sphere', 'box': 'Box', 'cylinder': 'Disc', 'capsule': 'Line', 'ring': 'Ring', 'cone': 'Cone', 'mesh': 'Mesh'}
COLLIDER_SHAPES = {'box': 'Box', 'sphere': 'Sphere', 'cylinder': 'Cylinder', 'mesh': 'Mesh'}
SECTION_ORDER = ['combustion', 'motion', 'domain', 'shading', 'lighting', 'embers', 'spread', 'camera', 'composite', 'render']
SECTION_HINTS = {
    'combustion': 'How fuel burns: flame height, heat and smoke.',
    'motion': 'Forces on the gas: buoyancy, swirl, turbulence and wind.',
    'domain': 'The simulation box: size, resolution, boundaries and time.',
    'shading': 'How flame and smoke look: colour temperature, brightness, smoke colour.',
    'lighting': 'Light on the smoke: ambient, a key light and the fire itself.',
    'embers': 'Sparks and embers thrown by the fire: launch, fall, bounce and glow.',
    'spread': 'Fire that spreads by itself over the ground and over colliders marked Burnable.',
    'camera': 'Lens and placement of the fire in the frame.',
    'composite': 'How the fire sits in your footage: haze, glow, light cast, grain.',
    'render': 'Frame size, frame range and final render quality.',
    'liquid': 'How the liquid behaves: gravity, splashiness, surface tension, grip, whitewater.',
    'water': 'How the liquid looks: colour, clarity, the surface, foam and spray, the wet ground.',
}
LIQUID_SECTION_ORDER = ['liquid', 'domain', 'water', 'lighting', 'camera', 'composite', 'render']
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


def worded(section, params, kind):
    """The parameters with liquid wording in a liquid scene."""
    if kind != 'liquid':
        return params
    out = []
    for p in params:
        w = LIQUID_WORDING.get((section, p.key))
        out.append(dataclasses.replace(p, label=w[0], tip=w[1] or p.tip) if w else p)
    return out


BOTH_SECTION_ORDER = ['combustion', 'motion', 'liquid', 'domain', 'shading', 'water', 'lighting', 'embers', 'spread',
                      'camera', 'composite', 'render']


def section_order(kind):
    if kind == 'both':
        return BOTH_SECTION_ORDER
    return LIQUID_SECTION_ORDER if kind == 'liquid' else SECTION_ORDER


def emitter_kind(scene_kind, emitter):
    """What an emitter is for: in a fire-and-liquid scene, each one emits one or the other."""
    if scene_kind == 'both':
        return 'liquid' if emitter.get('emits') == 'liquid' else 'fire'
    return scene_kind


def section_hint(sec, kind):
    return (LIQUID_HINTS.get(sec) if kind == 'liquid' else None) or SECTION_HINTS[sec]


def menu_label(shape, names):
    return names[shape] + ('…' if shape == 'mesh' else '')


def add_emitter(doc, shape, parent=None):
    """Add an emitter; a mesh emitter asks for its file first."""
    if shape == 'mesh':
        path = pick_mesh_file(parent)
        if path:
            doc.add_emitter('mesh', mesh=path)
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


class Outliner(QWidget):
    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(14)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._menu)
        self.tree.itemSelectionChanged.connect(self._picked)
        self.tree.itemChanged.connect(self._item_changed)
        v.addWidget(self.tree, 1)
        bar = QHBoxLayout()
        bar.setContentsMargins(6, 6, 6, 6)
        add = QPushButton('+ Emitter')
        m = QMenu(add)
        for shape in SHAPE_NAMES:
            m.addAction(menu_label(shape, SHAPE_NAMES), lambda s=shape: add_emitter(doc, s, self))
        add.setMenu(m)
        bar.addWidget(add)
        addc = QPushButton('+ Collider')
        mc = QMenu(addc)
        for shape in COLLIDER_SHAPES:
            mc.addAction(menu_label(shape, COLLIDER_SHAPES), lambda s=shape: add_collider(doc, s, self))
        addc.setMenu(mc)
        bar.addWidget(addc)
        bar.addStretch(1)
        v.addLayout(bar)
        self._building = False
        doc.sceneReplaced.connect(self.rebuild)
        doc.structureChanged.connect(self.rebuild)
        doc.selectionChanged.connect(self._sync_selection)
        doc.paramChanged.connect(lambda path: path == ('domain', 'kind') and QTimer.singleShot(0, self.rebuild))
        self.rebuild()

    def rebuild(self):
        self._building = True
        self.tree.clear()
        sc = self.doc.scene
        liquid = sc.kind == 'liquid'
        fire = QTreeWidgetItem(['Sources' if liquid else 'Emitters'])
        fire.setFlags(Qt.ItemIsEnabled)
        self.tree.addTopLevelItem(fire)
        for i, e in enumerate(sc.emitters):
            it = QTreeWidgetItem([f'{e["name"]}'])
            it.setToolTip(0, f'{SHAPE_NAMES.get(e["shape"], e["shape"])} {"liquid source" if liquid else "emitter"}')
            it.setData(0, Qt.UserRole, ('emitter', i))
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable | Qt.ItemIsUserCheckable)
            it.setCheckState(0, Qt.Checked if e['enabled'] else Qt.Unchecked)
            fire.addChild(it)
        cols = QTreeWidgetItem(['Colliders'])
        cols.setFlags(Qt.ItemIsEnabled)
        self.tree.addTopLevelItem(cols)
        for i, c in enumerate(sc.colliders):
            it = QTreeWidgetItem([c['name']])
            it.setData(0, Qt.UserRole, ('collider', i))
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable | Qt.ItemIsUserCheckable)
            it.setCheckState(0, Qt.Checked if c['enabled'] else Qt.Unchecked)
            cols.addChild(it)
        sets = QTreeWidgetItem(['Settings'])
        sets.setFlags(Qt.ItemIsEnabled)
        self.tree.addTopLevelItem(sets)
        for sec in section_order(sc.kind):
            it = QTreeWidgetItem([SECTION_TITLES[sec]])
            it.setToolTip(0, section_hint(sec, sc.kind))
            it.setData(0, Qt.UserRole, ('section', sec))
            sets.addChild(it)
        self.tree.expandAll()
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
        if it is not None and not it.isSelected():
            self._building = True
            self.tree.setCurrentItem(it)
            self._building = False

    def _picked(self):
        if self._building:
            return
        items = self.tree.selectedItems()
        if items:
            sel = items[0].data(0, Qt.UserRole)
            if sel:
                self.doc.select(sel)

    def _item_changed(self, it, col):
        if self._building:
            return
        sel = it.data(0, Qt.UserRole)
        if not sel or sel[0] not in ('emitter', 'collider'):
            return
        kind, i = sel
        d = (self.doc.scene.emitters if kind == 'emitter' else self.doc.scene.colliders)[i]
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
        if sel and sel[0] == 'emitter':
            m.addAction('Rename', lambda: self.tree.editItem(it, 0))
            m.addAction('Duplicate', lambda: self.doc.duplicate_emitter(sel[1]))
            m.addAction('Delete', lambda: self.doc.remove_emitter(sel[1]))
        elif sel and sel[0] == 'collider':
            m.addAction('Rename', lambda: self.tree.editItem(it, 0))
            m.addAction('Delete', lambda: self.doc.remove_collider(sel[1]))
        else:
            sub = m.addMenu('Add emitter')
            for shape in SHAPE_NAMES:
                sub.addAction(menu_label(shape, SHAPE_NAMES), lambda s=shape: add_emitter(self.doc, s, self))
            subc = m.addMenu('Add collider')
            for shape in COLLIDER_SHAPES:
                subc.addAction(menu_label(shape, COLLIDER_SHAPES), lambda s=shape: add_collider(self.doc, s, self))
        m.exec(self.tree.viewport().mapToGlobal(pos))


class Properties(QWidget):
    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        head = QHBoxLayout()
        head.setContentsMargins(10, 8, 10, 4)
        self.title = QLabel()
        self.title.setStyleSheet(f'font-size: 12pt; font-weight: 600; color: {theme.TEXT};')
        head.addWidget(self.title, 1)
        self.adv = QCheckBox('Advanced')
        self.adv.setToolTip('Show expert settings')
        self.adv.setChecked(QSettings().value('ui/advanced', False, type=bool))
        self.adv.toggled.connect(self._toggle_adv)
        head.addWidget(self.adv)
        v.addLayout(head)
        self.hint = QLabel()
        self.hint.setObjectName('hint')
        self.hint.setWordWrap(True)
        self.hint.setContentsMargins(10, 0, 10, 4)
        v.addWidget(self.hint)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        v.addWidget(self.scroll, 1)
        self.panel = None
        doc.selectionChanged.connect(lambda *_: QTimer.singleShot(0, self.rebuild))
        doc.sceneReplaced.connect(lambda: QTimer.singleShot(0, self.rebuild))
        doc.structureChanged.connect(self._structure)
        doc.paramChanged.connect(self._param)
        doc.frameChanged.connect(lambda *_: self.panel and self.panel.refresh_animated())
        self.rebuild()

    def _toggle_adv(self, on):
        QSettings().setValue('ui/advanced', bool(on))
        self.rebuild()

    def _structure(self):
        sel = self.doc.selection
        if sel[0] in ('emitter', 'collider'):
            QTimer.singleShot(0, self.rebuild)

    def _param(self, path):
        if self.panel is None:
            return
        self.panel.refresh(path)
        if path and path[-1] in ('mode', 'shape', 'kind', 'liquid_mode', 'emits'):
            QTimer.singleShot(0, self.rebuild)  # not inside the editor's own signal

    def rebuild(self):
        sel = self.doc.selection
        sc = self.doc.scene
        kind = sc.kind
        pos = self.scroll.verticalScrollBar().value()
        header = None
        if sel[0] == 'emitter' and sel[1] < len(sc.emitters):
            i = sel[1]
            e = sc.emitters[i]
            self.title.setText(e['name'])
            ekind = emitter_kind(kind, e)
            if ekind == 'liquid':
                self.hint.setText(f'{SHAPE_NAMES.get(e["shape"], e["shape"])} source · pours liquid into the simulation.')
            else:
                self.hint.setText(f'{SHAPE_NAMES.get(e["shape"], e["shape"])} emitter · releases fuel, heat and smoke into the simulation.')
            header = self._emitter_header(i)
            params = [p for p in EMITTER_PARAMS if p.key not in ('name', 'enabled')
                      and (applies('emitter', p.key, ekind) or (kind == 'both' and p.key == 'emits'))]
            hide = {'end'} if e['shape'] != 'capsule' else {'yaw'}
            if e['shape'] != 'mesh':
                hide |= {'mesh', 'thickness'}
            if ekind == 'liquid' and e['liquid_mode'] == 'fill':
                hide |= {'flow', 'vel_blend', 'stop'}
            params = worded('emitter', [p for p in params if p.key not in hide], ekind)
            self.panel = ParamPanel(self.doc, lambda k, i=i: ('emitter', i, k), params, header=header)
        elif sel[0] == 'collider' and sel[1] < len(sc.colliders):
            i = sel[1]
            self.title.setText(sc.colliders[i]['name'])
            if kind == 'liquid':
                self.hint.setText('Solid object the liquid flows around and splashes off (a wall, a rock, a step). Keyframe it to move it: it pushes the liquid out of its way.')
            else:
                self.hint.setText('Solid object the gas flows around (a wall, a car, a log). Keyframe it to move it; mark it Burnable to let fire spread over it.')
            params = [p for p in COLLIDER_PARAMS if p.key not in ('name',) and applies('collider', p.key, kind)]
            if sc.colliders[i]['shape'] != 'mesh':
                params = [p for p in params if p.key != 'mesh']
            params = worded('collider', params, kind)
            self.panel = ParamPanel(self.doc, lambda k, i=i: ('collider', i, k), params)
        else:
            sec = sel[1] if sel[0] == 'section' else section_order(kind)[0]
            if not applies(sec, None, kind):
                sec = section_order(kind)[0]
            self.title.setText(SECTION_TITLES[sec])
            self.hint.setText(section_hint(sec, kind))
            self.panel = ParamPanel(self.doc, lambda k, s=sec: (s, k),
                                    worded(sec, [p for p in SECTIONS[sec] if applies(sec, p.key, kind)], kind))
        self.scroll.setWidget(self.panel)
        self.scroll.verticalScrollBar().setValue(pos)

    def _emitter_header(self, i):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 6)
        dup = QPushButton('Duplicate')
        dup.clicked.connect(lambda: self.doc.duplicate_emitter(i))
        dele = QPushButton('Delete')
        dele.clicked.connect(lambda: self.doc.remove_emitter(i))
        on = QCheckBox('Enabled')
        on.setChecked(self.doc.scene.emitters[i]['enabled'])
        on.toggled.connect(lambda v: self.doc.set(('emitter', i, 'enabled'), v, merge=False))
        h.addWidget(on)
        h.addStretch(1)
        h.addWidget(dup)
        h.addWidget(dele)
        return w
