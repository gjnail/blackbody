"""Parameter editors bound to scene paths, built from the parameter registry."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QSettings, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
                               QLineEdit, QMenu, QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox, QToolButton,
                               QVBoxLayout, QWidget)

from ..scene.params import Param
from . import theme


def _srgb(c):
    c = max(0.0, float(c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def _lin(c):
    c = float(c)
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


class KeyButton(QToolButton):
    """Keyframe diamond: outline = not animated, amber outline = animated, filled = key on this frame."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = 0
        self.setFixedSize(QSize(18, 18))
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip('Keyframe this setting at the current frame')

    def set_state(self, s):
        if s != self.state:
            self.state = s
            self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = QPointF(self.width() / 2, self.height() / 2)
        r = 4.5
        poly = QPolygonF([QPointF(c.x(), c.y() - r), QPointF(c.x() + r, c.y()), QPointF(c.x(), c.y() + r), QPointF(c.x() - r, c.y())])
        key = QColor(theme.KEY)
        if self.state == 2:
            p.setPen(QPen(key, 1.2))
            p.setBrush(key)
        elif self.state == 1:
            p.setPen(QPen(key, 1.2))
            p.setBrush(Qt.NoBrush)
        else:
            p.setPen(QPen(QColor(theme.MUTED if self.underMouse() else '#56565c'), 1.2))
            p.setBrush(Qt.NoBrush)
        p.drawPolygon(poly)


class ScrubLabel(QLabel):
    """A label you can drag sideways to change a number (Shift for fine), double-click to reset."""
    scrubbed = Signal(float)   # delta in slider units (0..1 of the soft range)
    released = Signal()
    reset = Signal()

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._x = None
        self.setCursor(Qt.SizeHorCursor)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._x = e.position().x()

    def mouseMoveEvent(self, e):
        if self._x is None:
            return
        dx = e.position().x() - self._x
        self._x = e.position().x()
        k = 0.0008 if e.modifiers() & Qt.ShiftModifier else 0.004
        self.scrubbed.emit(dx * k)

    def mouseReleaseEvent(self, e):
        if self._x is not None:
            self._x = None
            self.released.emit()

    def mouseDoubleClickEvent(self, e):
        self.reset.emit()


class ColorButton(QPushButton):
    changed = Signal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = (1.0, 1.0, 1.0)
        self.setFixedHeight(22)
        self.clicked.connect(self._pick)

    def set_value(self, rgb):
        self.value = tuple(rgb)
        s = [int(round(min(1.0, _srgb(c)) * 255)) for c in self.value]
        lum = 0.2126 * s[0] + 0.7152 * s[1] + 0.0722 * s[2]
        fg = '#111' if lum > 140 else '#eee'
        self.setText('  '.join(f'{c:.2f}' for c in self.value))
        self.setStyleSheet(f'QPushButton {{ background: rgb({s[0]},{s[1]},{s[2]}); color: {fg}; border: 1px solid {theme.LINE}; }}')

    def _pick(self):
        s = [min(1.0, _srgb(c)) for c in self.value]
        c = QColorDialog.getColor(QColor.fromRgbF(*s), self, 'Colour', QColorDialog.DontUseNativeDialog)
        if c.isValid():
            self.changed.emit((_lin(c.redF()), _lin(c.greenF()), _lin(c.blueF())))


def pick_mesh_file(parent, current=''):
    """Ask for an OBJ or STL mesh; returns its path or ''."""
    start = QSettings().value('ui/mesh_dir', '', type=str)
    if current and not current.startswith('builtin:'):
        start = current
    path, _ = QFileDialog.getOpenFileName(parent, 'Choose a mesh', start, 'Meshes (*.obj *.stl);;All files (*)')
    if path:
        from pathlib import Path
        QSettings().setValue('ui/mesh_dir', str(Path(path).parent))
    return path


class ParamRow(QWidget):
    """One setting: label, editor and (if it can animate) a keyframe button."""

    def __init__(self, doc, path, spec: Param, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.path = path
        self.spec = spec
        self._busy = False
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 1, 0, 1)
        lay.setSpacing(6)
        tip = spec.tip + (f'  ({spec.unit})' if spec.unit and spec.unit not in spec.tip else '')
        numeric = spec.kind in ('float', 'int')
        self.label = ScrubLabel(spec.label) if numeric else QLabel(spec.label)
        self.label.setToolTip(tip or spec.label)
        self.label.setFixedWidth(124)
        self.label.setWordWrap(False)
        lay.addWidget(self.label)
        self.editor = self._make_editor()
        self.editor.setToolTip(tip or spec.label)
        lay.addWidget(self.editor, 1)
        self.key = None
        if spec.anim:
            self.key = KeyButton()
            self.key.clicked.connect(lambda: doc.toggle_key(path))
            self.key.setContextMenuPolicy(Qt.CustomContextMenu)
            self.key.customContextMenuRequested.connect(self._key_menu)
            lay.addWidget(self.key)
        else:
            sp = QWidget()
            sp.setFixedWidth(18)
            lay.addWidget(sp)
        if numeric:
            self.label.scrubbed.connect(self._scrub)
            self.label.released.connect(doc.end_drag)
            self.label.reset.connect(lambda: doc.set(path, spec.default, merge=False))
        self.refresh()

    # -- editors ------------------------------------------------------------------------------------

    def _make_editor(self):
        s = self.spec
        if s.kind in ('float', 'int'):
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setRange(0, 1000)
            self.slider.setMinimumWidth(40)
            self.slider.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            self.slider.valueChanged.connect(self._slider_moved)
            self.slider.sliderReleased.connect(self.doc.end_drag)
            if s.kind == 'float':
                self.spin = QDoubleSpinBox()
                self.spin.setDecimals(s.decimals)
                lo = s.hard_lo if s.hard_lo is not None else (min(s.lo, 0.0) if s.lo >= 0 else s.lo * 10)
                hi = s.hard_hi if s.hard_hi is not None else max(s.hi * 10, s.hi + 1)
                if s.lo >= 0 and s.hard_lo is None:
                    lo = 0.0 if s.lo == 0 else min(s.lo, 0.0)
                self.spin.setRange(lo, hi)
                self.spin.setSingleStep(max(10 ** -s.decimals, (s.hi - s.lo) / 100))
            else:
                self.spin = QSpinBox()
                self.spin.setRange(int(s.hard_lo if s.hard_lo is not None else min(s.lo, 0) - 100000 * (s.lo < 0)),
                                   int(s.hard_hi if s.hard_hi is not None else max(s.hi * 4, s.hi + 10)))
            if s.unit:
                self.spin.setSuffix(f' {s.unit}')
            self.spin.setKeyboardTracking(False)
            self.spin.setFixedWidth(88)
            self.spin.setAlignment(Qt.AlignRight)
            self.spin.valueChanged.connect(self._spin_changed)
            self.spin.editingFinished.connect(self.doc.end_drag)
            h.addWidget(self.slider, 1)
            h.addWidget(self.spin)
            return w
        if s.kind == 'bool':
            self.check = QCheckBox()
            self.check.toggled.connect(lambda v: self._commit(bool(v), merge=False))
            return self.check
        if s.kind == 'enum':
            self.combo = QComboBox()
            self.combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            self.combo.setMinimumWidth(80)
            for val, text in s.options:
                self.combo.addItem(text, val)
            self.combo.currentIndexChanged.connect(lambda i: self._commit(self.combo.itemData(i), merge=False))
            return self.combo
        if s.kind == 'color':
            self.color = ColorButton()
            self.color.changed.connect(lambda v: self._commit(v, merge=False))
            return self.color
        if s.kind == 'vec3':
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(3)
            self.vec = []
            for i, (axis, col) in enumerate((('X', '#e06c6c'), ('Y', '#7cc47a'), ('Z', '#6f9fe8'))):
                sb = QDoubleSpinBox()
                sb.setDecimals(s.decimals)
                sb.setRange(-1e6, 1e6)
                sb.setSingleStep(10 ** -max(1, s.decimals - 1))
                sb.setKeyboardTracking(False)
                sb.setStyleSheet(f'QDoubleSpinBox {{ border-left: 2px solid {col}; }}')
                sb.setToolTip(f'{axis}')
                sb.setMinimumWidth(42)
                sb.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
                sb.valueChanged.connect(self._vec_changed)
                sb.editingFinished.connect(self.doc.end_drag)
                h.addWidget(sb, 1)
                self.vec.append(sb)
            return w
        if s.kind == 'file':
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(4)
            self.line = QLineEdit()
            self.line.setPlaceholderText('No file')
            self.line.editingFinished.connect(lambda: self._commit(self.line.text().strip(), merge=False))
            pick = QToolButton()
            pick.setText('…')
            pick.setToolTip('Choose a file')
            pick.clicked.connect(self._browse)
            h.addWidget(self.line, 1)
            h.addWidget(pick)
            return w
        self.line = QLineEdit()
        self.line.editingFinished.connect(lambda: self._commit(self.line.text(), merge=False))
        return self.line

    def _browse(self):
        path = pick_mesh_file(self, self.line.text())
        if path:
            self._commit(path, merge=False)

    # -- value mapping ----------------------------------------------------------------------------------

    def _to_t(self, v):
        s = self.spec
        if s.log and s.lo > 0:
            return (math.log(max(v, s.lo * 1e-6)) - math.log(s.lo)) / (math.log(s.hi) - math.log(s.lo))
        return (v - s.lo) / (s.hi - s.lo) if s.hi != s.lo else 0.0

    def _from_t(self, t):
        s = self.spec
        t = min(max(t, 0.0), 1.0)
        if s.log and s.lo > 0:
            v = math.exp(math.log(s.lo) + t * (math.log(s.hi) - math.log(s.lo)))
        else:
            v = s.lo + t * (s.hi - s.lo)
        if s.kind == 'int':
            return int(round(v))
        return round(v, s.decimals + 2)

    # -- events -------------------------------------------------------------------------------------------

    def _commit(self, value, merge=True):
        if self._busy:
            return
        self.doc.set(self.path, value, merge=merge)

    def _slider_moved(self, i):
        if self._busy:
            return
        self._commit(self._from_t(i / 1000.0))

    def _spin_changed(self, v):
        if self._busy:
            return
        self._commit(v)

    def _scrub(self, dt):
        cur = self.doc.value(self.path)
        t = self._to_t(cur) + dt
        self._commit(self._from_t(t))

    def _vec_changed(self, _):
        if self._busy:
            return
        self._commit(tuple(sb.value() for sb in self.vec))

    def _key_menu(self, pos):
        m = QMenu(self)
        m.addAction('Set key here' if not self.doc.has_key(self.path) else 'Remove key here', lambda: self.doc.toggle_key(self.path))
        a = m.addAction('Clear animation', lambda: self.doc.clear_animation(self.path))
        a.setEnabled(self.doc.is_animated(self.path))
        m.exec(self.key.mapToGlobal(pos))

    # -- refresh --------------------------------------------------------------------------------------------

    def refresh(self):
        try:
            v = self.doc.value(self.path)
        except (KeyError, IndexError):
            return
        self._busy = True
        try:
            s = self.spec
            if s.kind in ('float', 'int'):
                self.slider.setValue(int(round(min(max(self._to_t(v), 0.0), 1.0) * 1000)))
                self.spin.setValue(v)
            elif s.kind == 'bool':
                self.check.setChecked(bool(v))
            elif s.kind == 'enum':
                i = self.combo.findData(v)
                self.combo.setCurrentIndex(max(0, i))
            elif s.kind == 'color':
                self.color.set_value(v)
            elif s.kind == 'vec3':
                for sb, x in zip(self.vec, v):
                    sb.setValue(float(x))
            else:
                self.line.setText(str(v))
            if self.key is not None:
                self.key.set_state(2 if self.doc.has_key(self.path) else (1 if self.doc.is_animated(self.path) else 0))
        finally:
            self._busy = False


class Group(QWidget):
    """A collapsible group of rows."""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 6, 0, 2)
        v.setSpacing(2)
        self.head = QToolButton()
        self.head.setText(title.upper())
        self.head.setCheckable(True)
        self.head.setChecked(True)
        self.head.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.head.setArrowType(Qt.DownArrow)
        self.head.setStyleSheet(f'QToolButton {{ color: {theme.MUTED}; font-weight: 600; border: 0; padding: 2px 0; letter-spacing: 1px; }}'
                                f'QToolButton:checked {{ background: transparent; border: 0; }}')
        self.head.toggled.connect(self._toggle)
        v.addWidget(self.head)
        self.body = QWidget()
        self.lay = QVBoxLayout(self.body)
        self.lay.setContentsMargins(4, 0, 0, 0)
        self.lay.setSpacing(1)
        v.addWidget(self.body)

    def _toggle(self, on):
        self.body.setVisible(on)
        self.head.setArrowType(Qt.DownArrow if on else Qt.RightArrow)


class ParamPanel(QWidget):
    """Rows for a list of parameters, grouped, with advanced ones behind a toggle."""

    def __init__(self, doc, make_path, params, parent=None, header=None):
        super().__init__(parent)
        self.doc = doc
        self.rows = []
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 6, 10, 10)
        v.setSpacing(0)
        if header is not None:
            v.addWidget(header)
        show_adv = QSettings().value('ui/advanced', False, type=bool)
        groups = {}
        order = []
        for p in params:
            if p.advanced and not show_adv:
                continue
            g = p.group or 'Settings'
            if g not in groups:
                groups[g] = Group(g)
                order.append(g)
            row = ParamRow(doc, make_path(p.key), p)
            groups[g].lay.addWidget(row)
            self.rows.append(row)
        for g in order:
            v.addWidget(groups[g])
        v.addStretch(1)

    def refresh(self, path=None):
        for r in self.rows:
            if path is None or r.path == path:
                r.refresh()

    def refresh_animated(self):
        for r in self.rows:
            if r.key is not None:
                r.refresh()
