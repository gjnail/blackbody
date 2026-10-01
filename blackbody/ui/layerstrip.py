"""The layer strip over the viewer: the shot's layers back to front, the one being edited lit. Click one to
edit it; right-click for rename, hide, move, duplicate and delete."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QInputDialog, QLabel, QMenu, QToolButton, QWidget

from . import icons, theme

KIND_GLYPH = {'fire': 'flame', 'liquid': 'drop', 'both': 'flame', 'cloud': 'cloud'}
KIND_COLOUR = {'fire': theme.ACCENT, 'liquid': '#6fb6ff', 'both': '#ffb27a', 'cloud': '#d0d6e0'}


class LayerStrip(QWidget):
    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.setObjectName('layerstrip')
        self.setStyleSheet(f'QWidget#layerstrip {{ background: {theme.PANEL}; border-bottom: 1px solid {theme.LINE}; }}')
        self.h = QHBoxLayout(self)
        self.h.setContentsMargins(10, 4, 10, 4)
        self.h.setSpacing(4)
        doc.layersChanged.connect(self.rebuild)
        doc.sceneReplaced.connect(self.rebuild)
        doc.structureChanged.connect(self.rebuild)
        doc.paramChanged.connect(lambda path: path == ('domain', 'kind') and self.rebuild())
        self.rebuild()

    def rebuild(self):
        while self.h.count():
            it = self.h.takeAt(0)
            w = it.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        layers = self.doc.layers()
        self.setVisible(len(layers) > 1)
        if len(layers) <= 1:
            return
        lab = QLabel('LAYERS')
        lab.setObjectName('section')
        lab.setToolTip('The shot’s effects, each its own simulation, drawn back to front: the last one is in front. '
                       'Effects and Create work on the one that is lit.')
        self.h.addWidget(lab)
        self.h.addSpacing(6)
        for n, (uid, sc) in enumerate(layers):
            b = QToolButton()
            b.setCheckable(True)
            b.setChecked(uid == self.doc.active)
            b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            hidden = not sc.enabled
            b.setIcon(icons.glyph_icon(KIND_GLYPH.get(sc.kind, 'flame'), theme.FAINT if hidden else KIND_COLOUR.get(sc.kind, theme.TEXT), 14))
            b.setText(f' {sc.name or "Untitled"}' + ('  (hidden)' if hidden else ''))
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(f'Layer {n + 1} of {len(layers)}{" (back)" if n == 0 else " (front)" if n == len(layers) - 1 else ""}: '
                         'click to edit it, double-click to rename, right-click for more')
            b.setStyleSheet(f'QToolButton {{ border: 1px solid {theme.LINE_HI}; border-radius: 11px; padding: 2px 10px; '
                            f'color: {theme.FAINT if hidden else theme.MUTED}; background: transparent; }}'
                            f'QToolButton:hover {{ color: {theme.TEXT}; border-color: {theme.FAINT}; }}'
                            f'QToolButton:checked {{ background: {theme.ACCENT_DIM}; border-color: {theme.ACCENT}; color: {theme.TEXT}; }}')
            b.clicked.connect(lambda _=False, u=uid: (self.doc.set_active(u), self.rebuild()))
            b.setContextMenuPolicy(Qt.CustomContextMenu)
            b.customContextMenuRequested.connect(lambda pos, u=uid, w=b: self._menu(u, w.mapToGlobal(pos)))
            b.mouseDoubleClickEvent = lambda e, u=uid: self._rename(u)
            self.h.addWidget(b)
            if n < len(layers) - 1:
                arrow = QLabel('›')
                arrow.setObjectName('faint')
                self.h.addWidget(arrow)
        add = QToolButton()
        add.setIcon(icons.glyph_icon('plus', theme.MUTED, 14, active=theme.TEXT))
        add.setToolTip('Add a layer in front: another effect in the same shot, with its own simulation')
        add.clicked.connect(lambda: self.window().add_layer() if hasattr(self.window(), 'add_layer') else self.doc.add_layer())
        self.h.addWidget(add)
        self.h.addStretch(1)
        hint = QLabel('back → front')
        hint.setObjectName('faint')
        self.h.addWidget(hint)

    def _rename(self, uid):
        sc = self.doc.shot.layer(uid)
        if sc is None:
            return
        name, ok = QInputDialog.getText(self, 'Rename layer', 'Name:', text=sc.name)
        if ok and name.strip():
            if uid == 'base':
                self.doc.edit('Rename layer', lambda s: setattr(self.doc.shot, 'name', name.strip()), structure=True)
                self.doc.layersChanged.emit()
            else:
                self.doc.layer_set(uid, 'name', name.strip(), 'Rename layer')

    def _menu(self, uid, gpos):
        sc = self.doc.shot.layer(uid)
        if sc is None:
            return
        order = [u for u, _ in self.doc.layers()]
        i = order.index(uid)
        m = QMenu(self)
        m.addAction('Edit this layer', lambda: self.doc.set_active(uid))
        m.addAction('Rename…', lambda: self._rename(uid))
        m.addAction('Show' if not sc.enabled else 'Hide', lambda: self.doc.layer_set(uid, 'enabled', not sc.enabled,
                                                                                   'Show layer' if not sc.enabled else 'Hide layer'))
        m.addSeparator()
        a = m.addAction('Move back', lambda: self.doc.move_layer(uid, -1))
        a.setEnabled(i > 0)
        a = m.addAction('Move forward', lambda: self.doc.move_layer(uid, +1))
        a.setEnabled(i < len(order) - 1)
        m.addSeparator()
        if uid != 'base':
            m.addAction('Duplicate', lambda: self.doc.duplicate_layer(uid))
            m.addAction(icons.glyph_icon('trash', theme.TEXT, 16), 'Delete layer', lambda: self.doc.remove_layer(uid))
        else:
            a = m.addAction('The base layer holds the shot (its footage and settings): it cannot be deleted')
            a.setEnabled(False)
        m.exec(gpos)
