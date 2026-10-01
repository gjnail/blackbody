"""The roto bar over the viewer while roto is on: the shapes, and the selected one's feather, invert and keys."""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QToolButton, QVBoxLayout

from ..scene import roto as R
from . import icons, theme
from .params import guard_wheel


class RotoBar(QFrame):
    def __init__(self, viewport):
        super().__init__(viewport)
        self.vp = viewport
        self.doc = viewport.doc
        self.setObjectName('rotobar')
        self.setStyleSheet(f'QFrame#rotobar {{ background: rgba(24,24,28,235); border: 1px solid {theme.LINE_HI}; border-radius: 9px; }}'
                           'QLabel { background: transparent; }')
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 6, 10, 6)
        v.setSpacing(4)
        h = QHBoxLayout()
        h.setSpacing(6)
        t = QLabel('Roto')
        t.setStyleSheet('font-weight: 600;')
        h.addWidget(t)
        self.shapes = QComboBox()
        self.shapes.setMinimumWidth(110)
        self.shapes.setToolTip('The shape you are editing')
        guard_wheel(self.shapes)
        self.shapes.activated.connect(self._pick)
        h.addWidget(self.shapes)
        self.on = QToolButton()
        self.on.setObjectName('toggle')
        self.on.setText('On')
        self.on.setCheckable(True)
        self.on.setToolTip('Use this shape (off: it is kept but hides nothing)')
        self.on.toggled.connect(lambda v: self._set('enabled', bool(v), 'Switch roto shape'))
        h.addWidget(self.on)
        self.inv = QToolButton()
        self.inv.setObjectName('toggle')
        self.inv.setText('Invert')
        self.inv.setCheckable(True)
        self.inv.setToolTip('Hide the effect everywhere except inside the shape')
        self.inv.toggled.connect(lambda v: self._set('invert', bool(v), 'Invert roto shape'))
        h.addWidget(self.inv)
        fl = QLabel('Feather')
        fl.setObjectName('hint')
        h.addWidget(fl)
        self.feather = QDoubleSpinBox()
        self.feather.setRange(0.0, 200.0)
        self.feather.setDecimals(1)
        self.feather.setSuffix(' px')
        self.feather.setKeyboardTracking(False)
        self.feather.setFixedWidth(76)
        self.feather.setToolTip('How soft the edge is, in pixels of the output frame')
        guard_wheel(self.feather)
        self.feather.valueChanged.connect(lambda v: self._set('feather', float(v), 'Feather roto shape'))
        h.addWidget(self.feather)
        self.key = QToolButton()
        self.key.setObjectName('toggle')
        self.key.setCheckable(True)
        self.key.setToolTip('A key of the shape on this frame. Moving a point or the shape keys it here by itself; '
                            'untick to remove this frame’s key.')
        self.key.clicked.connect(self._key)
        h.addWidget(self.key)
        dele = QToolButton()
        dele.setIcon(icons.glyph_icon('trash', theme.MUTED, 16, active=theme.TEXT))
        dele.setToolTip('Delete this shape (Delete)')
        dele.clicked.connect(self.vp.roto_delete)
        h.addWidget(dele)
        v.addLayout(h)
        self.hint = QLabel()
        self.hint.setObjectName('hint')
        self.hint.setWordWrap(True)
        v.addWidget(self.hint)
        self._busy = False
        self.doc.structureChanged.connect(self.sync)
        self.doc.sceneReplaced.connect(self.sync)
        self.doc.frameChanged.connect(lambda *_: self.sync())
        self.sync()

    def sync(self):
        if not self.isVisible() and not self.vp.roto_mode:
            return
        self._busy = True
        try:
            shapes = self.doc.scene.roto
            names = [s.get('name', f'Roto {i + 1}') for i, s in enumerate(shapes)]
            if [self.shapes.itemText(i) for i in range(self.shapes.count())] != names:
                self.shapes.clear()
                self.shapes.addItems(names)
            sel = self.vp.roto_sel
            have = sel is not None and sel < len(shapes)
            for w in (self.shapes, self.on, self.inv, self.feather, self.key):
                w.setEnabled(have)
            if have:
                sh = shapes[sel]
                self.shapes.setCurrentIndex(sel)
                self.on.setChecked(sh.get('enabled', True))
                self.inv.setChecked(sh.get('invert', False))
                self.feather.setValue(float(sh.get('feather', 0.0)))
                k = R.has_key(sh, self.doc.frame)
                self.key.setChecked(k)
                self.key.setText(f'Key at {self.doc.frame}' if k else f'No key at {self.doc.frame}')
            else:
                self.key.setText('Key')
            if self.doc.scene.kind == 'liquid':
                self.hint.setText('Roto hides fire and smoke; in a liquid scene, put a collider over the object and keep '
                                  '“In the shot” on instead.')
            elif self.vp._roto_draft is not None:
                self.hint.setText(f'{len(self.vp._roto_draft)} point(s). Click the first point (or press Enter) to close the shape, '
                                  'Esc to stop.')
            else:
                self.hint.setText('Click around what is in front of the effect to draw a shape, then click the first point to close it. '
                                  'Drag points or the whole shape on any frame (that keys it); Ctrl+click an edge adds a point, '
                                  'right-click a point removes it.')
        finally:
            self._busy = False
        self.adjustSize()
        self.vp.place_rotobar()

    def _pick(self, i):
        self.vp.roto_sel = i
        self.vp.update()
        self.sync()

    def _set(self, key, value, text):
        if self._busy or self.vp.roto_sel is None:
            return
        i = self.vp.roto_sel

        def fn(shapes):
            if i < len(shapes):
                shapes[i][key] = value
        self.doc.roto_edit(text, fn)

    def _key(self, on):
        i = self.vp.roto_sel
        if i is None or i >= len(self.doc.scene.roto):
            return
        f = self.doc.frame
        if on:
            pts = R.points_at(self.doc.scene.roto[i], f)
            if pts:
                self.doc.roto_edit('Key roto shape', lambda shapes: R.set_points(shapes[i], f, pts))
        else:
            self.doc.roto_edit('Remove roto key', lambda shapes: R.remove_key(shapes[i], f))
