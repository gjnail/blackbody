"""Parameter editors bound to scene paths, built from the parameter registry."""
from __future__ import annotations

import math
import re

from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, QSettings, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QAbstractButton, QAbstractSpinBox, QColorDialog, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFrame, QHBoxLayout, QLabel, QLineEdit, QMenu, QPushButton, QSizePolicy, QToolButton,
                               QVBoxLayout, QWidget)

from ..scene.params import Param
from . import icons, theme


def _srgb(c):
    c = max(0.0, float(c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def _lin(c):
    c = float(c)
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


class WheelGuard(QObject):
    """Scrolling a panel must not change the combo box or number field that passes under the mouse:
    they take the wheel only once clicked into."""

    def eventFilter(self, obj, e):
        if e.type() == QEvent.Wheel and not obj.hasFocus():
            e.ignore()
            return True
        return False


WHEEL_GUARD = None


def guard_wheel(w):
    global WHEEL_GUARD
    if WHEEL_GUARD is None:
        WHEEL_GUARD = WheelGuard()
    w.setFocusPolicy(Qt.StrongFocus)
    w.installEventFilter(WHEEL_GUARD)


class KeyButton(QToolButton):
    """Keyframe diamond: outline = not animated, amber outline = animated, filled = key on this frame."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = 0
        self.setFixedSize(QSize(18, 18))
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip('Keyframe this setting at the current frame (right-click for more)')
        self.setStyleSheet('QToolButton { border: 0; padding: 0; background: transparent; }')

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
            p.setPen(QPen(QColor(theme.MUTED if self.underMouse() else '#46464d'), 1.2))
            p.setBrush(Qt.NoBrush)
        p.drawPolygon(poly)


class ResetButton(QToolButton):
    """Shown on a setting changed from its preset: puts the preset's value back."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.active = False
        self.setFixedSize(QSize(18, 18))
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet('QToolButton { border: 0; padding: 0; background: transparent; }')

    def set_active(self, on, tip=''):
        self.active = bool(on)
        self.setEnabled(self.active)
        self.setToolTip(tip)
        self.update()

    def paintEvent(self, e):
        if not self.active:
            return
        p = QPainter(self)
        col = theme.TEXT if self.underMouse() else theme.CHANGED
        icons.paint_glyph(p, 'reset', QRectF(3, 3, 12, 12), col, 1.4)


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
        self.setFixedHeight(24)
        self.setCursor(Qt.PointingHandCursor)
        self.clicked.connect(self._pick)

    def set_value(self, rgb):
        self.value = tuple(rgb)
        self.setToolTip('Colour (linear RGB): ' + '  '.join(f'{c:.3f}' for c in self.value) + '\nClick to choose another.')
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        s = [int(round(min(1.0, _srgb(c)) * 255)) for c in self.value]
        p.setPen(QPen(QColor(theme.LINE_HI if not self.underMouse() else theme.FAINT), 1))
        p.setBrush(QColor(*s))
        p.drawRoundedRect(r, 5, 5)
        lum = 0.2126 * s[0] + 0.7152 * s[1] + 0.0722 * s[2]
        p.setPen(QColor(17, 17, 17, 200) if lum > 140 else QColor(240, 240, 240, 200))
        f = QFont(self.font())
        f.setPointSizeF(8.5)
        p.setFont(f)
        p.drawText(r.adjusted(8, 0, -8, 0), Qt.AlignVCenter | Qt.AlignRight, '#%02X%02X%02X' % tuple(s))

    def _pick(self):
        s = [min(1.0, _srgb(c)) for c in self.value]
        c = QColorDialog.getColor(QColor.fromRgbF(*s), self, 'Colour', QColorDialog.DontUseNativeDialog)
        if c.isValid():
            self.changed.emit((_lin(c.redF()), _lin(c.greenF()), _lin(c.blueF())))


class Switch(QAbstractButton):
    """An on/off switch (for yes/no settings)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(QSize(34, 18))

    def sizeHint(self):
        return QSize(34, 18)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        on = self.isChecked()
        track = QColor(theme.ACCENT if on else (theme.LINE_HI if self.underMouse() else theme.FIELD_HI))
        knob = QColor('#ffffff' if on else theme.MUTED)
        if not self.isEnabled():
            track.setAlpha(80)
            knob.setAlpha(120)
        p.setPen(Qt.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(0, 1, 34, 16), 8, 8)
        p.setBrush(knob)
        x = 19.0 if on else 3.0
        p.drawEllipse(QRectF(x, 3, 12, 12))


def _safe_eval(text):
    """A number typed into a field: plain arithmetic is allowed (2*1.5, 1/24)."""
    t = text.strip().replace(',', '.')
    if not t:
        raise ValueError('empty')
    if not re.fullmatch(r'[0-9eE.+\-*/() ]+', t):
        raise ValueError(text)
    return float(eval(t, {'__builtins__': {}}, {}))


class ValueField(QWidget):
    """A number shown as a filled bar: drag it sideways to change the value (Shift for fine), click it to
    type one. The bar shows where the value sits in its usual range; typed values may go beyond it."""
    edited = Signal(float)      # live while dragging
    finished = Signal()         # drag released or value typed: the end of one undo step

    def __init__(self, spec: Param, to_t, from_t, lo, hi, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.to_t = to_t
        self.from_t = from_t
        self.lo, self.hi = lo, hi
        self.value = spec.default
        self._press = None
        self._dragging = False
        self.edit = None
        self.setFixedHeight(24)
        self.setMinimumWidth(70)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.SizeHorCursor)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setMouseTracking(True)

    def text(self):
        v = self.value
        if self.spec.kind == 'int':
            s = f'{int(round(v))}'
        else:
            s = f'{v:.{self.spec.decimals}f}'
        return f'{s} {self.spec.unit}' if self.spec.unit else s

    def set_value(self, v):
        self.value = v
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        hot = self.underMouse() or self._dragging
        p.setPen(QPen(QColor(theme.ACCENT if self.hasFocus() and self.edit is None and self._dragging else
                             (theme.FIELD_HI if hot else theme.FIELD)), 1))
        p.setBrush(QColor(theme.FIELD_HI if hot else theme.FIELD))
        p.drawRoundedRect(r, 5, 5)
        if self.edit is not None:
            return
        t = min(max(self.to_t(self.value), 0.0), 1.0)
        if t > 0:
            fill = QRectF(r.x(), r.y(), r.width() * t, r.height())
            p.save()
            p.setClipRect(fill)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(theme.ACCENT_FILL if not hot else '#47291a'))
            p.drawRoundedRect(r, 5, 5)
            p.restore()
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(theme.ACCENT if hot else theme.ACCENT_DIM))
            p.drawRoundedRect(QRectF(r.x() + 3, r.bottom() - 3.5, max(0.0, fill.width() - 6), 2), 1, 1)
        p.setPen(QColor(theme.TEXT))
        p.drawText(r.adjusted(8, 0, -8, 0), Qt.AlignVCenter | Qt.AlignRight, self.text())

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._press = (e.position().x(), self.to_t(self.value), self.value)
            self._dragging = False

    def mouseMoveEvent(self, e):
        if self._press is None:
            return
        x0, t0, v0 = self._press
        dx = e.position().x() - x0
        if not self._dragging and abs(dx) < 3:
            return
        self._dragging = True
        k = 0.15 if e.modifiers() & Qt.ShiftModifier else 1.0
        t = t0 + dx / max(40.0, self.width()) * k
        v = self.from_t(t)
        if v != self.value:
            self.value = v
            self.update()
            self.edited.emit(float(v))

    def mouseReleaseEvent(self, e):
        if self._press is None:
            return
        self._press = None
        if self._dragging:
            self._dragging = False
            self.update()
            self.finished.emit()
        else:
            self.start_edit()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_F2):
            self.start_edit()
        else:
            super().keyPressEvent(e)

    def start_edit(self):
        if self.edit is not None:
            return
        self.edit = QLineEdit(self)
        self.edit.setAlignment(Qt.AlignRight)
        self.edit.setStyleSheet(f'QLineEdit {{ background: {theme.FIELD}; border: 1px solid {theme.ACCENT}; border-radius: 5px; padding: 0 7px; }}')
        v = self.value
        self.edit.setText(f'{int(round(v))}' if self.spec.kind == 'int' else f'{v:.{self.spec.decimals}f}')
        self.edit.setGeometry(self.rect())
        self.edit.selectAll()
        self.edit.show()
        self.edit.setFocus()
        self.edit.editingFinished.connect(self._commit_edit)
        self.edit.installEventFilter(self)
        self.update()

    def eventFilter(self, obj, e):
        if obj is self.edit and e.type() == QEvent.KeyPress and e.key() == Qt.Key_Escape:
            self._close_edit()
            return True
        return False

    def _commit_edit(self):
        if self.edit is None:
            return
        text = self.edit.text()
        self._close_edit()
        try:
            v = _safe_eval(text)
        except Exception:
            return
        v = min(max(v, self.lo), self.hi)
        if self.spec.kind == 'int':
            v = int(round(v))
        self.value = v
        self.update()
        self.edited.emit(float(v))
        self.finished.emit()

    def _close_edit(self):
        e, self.edit = self.edit, None
        if e is not None:
            e.removeEventFilter(self)
            e.hide()
            e.deleteLater()
        self.update()

    def resizeEvent(self, e):
        if self.edit is not None:
            self.edit.setGeometry(self.rect())


def pick_image_file(parent, current=''):
    """Ask for an environment panorama (HDR, EXR or an ordinary image); returns its path or ''."""
    start = QSettings().value('ui/hdri_dir', '', type=str) or current
    path, _ = QFileDialog.getOpenFileName(parent, 'Choose an environment image', current or start,
                                          'Panoramas (*.hdr *.exr *.png *.jpg *.jpeg *.tif *.tiff);;All files (*)')
    if path:
        from pathlib import Path
        QSettings().setValue('ui/hdri_dir', str(Path(path).parent))
    return path


def pick_mesh_file(parent, current=''):
    """Ask for a mesh source: an OBJ or STL mesh (or a numbered sequence of them, for a deforming mesh),
    a greyscale heightfield image (terrain), or a mesh prim in a USD file. Returns the source or ''."""
    from pathlib import Path
    start = QSettings().value('ui/mesh_dir', '', type=str)
    if current and not current.startswith('builtin:'):
        start = current.split('#/')[0]
    path, _ = QFileDialog.getOpenFileName(
        parent, 'Choose a mesh', start,
        'Meshes, terrain and USD (*.obj *.stl *.png *.tif *.tiff *.exr *.usd *.usda *.usdc *.usdz);;'
        'Meshes (*.obj *.stl);;Heightfield images (*.png *.tif *.tiff *.exr);;USD (*.usd *.usda *.usdc *.usdz);;All files (*)')
    if not path:
        return ''
    QSettings().setValue('ui/mesh_dir', str(Path(path).parent))
    ext = Path(path).suffix.lower()
    if ext in ('.usd', '.usda', '.usdc', '.usdz'):
        from PySide6.QtWidgets import QInputDialog, QMessageBox
        try:
            from ..io.usd import scan
            meshes = scan(path).meshes()
        except Exception as ex:
            QMessageBox.warning(parent, 'USD', f'Cannot read {Path(path).name}:\n{ex}')
            return ''
        if not meshes:
            QMessageBox.information(parent, 'USD', f'{Path(path).name} has no meshes.')
            return ''
        labels = [f'{m.path}  ({m.motion}, {m.triangles} triangles)' for m in meshes]
        item, ok = QInputDialog.getItem(parent, 'USD mesh', 'Mesh to use:', labels, 0, False)
        if not ok:
            return ''
        m = meshes[labels.index(item)]
        # a mesh that deforms or tumbles is read in world space, frame by frame
        return f'{path}#{m.path}' + ('?world' if m.motion == 'world' else '')
    if ext in ('.obj', '.stl'):
        from ..io.footage import find_sequence
        seq = find_sequence(Path(path))
        if seq and len(seq[0]) > 1:
            from PySide6.QtWidgets import QMessageBox
            r = QMessageBox.question(parent, 'Mesh sequence',
                                     f'{Path(path).name} is part of a numbered sequence of {len(seq[0])} files. '
                                     'Use the whole sequence as a deforming mesh (one file per frame)?')
            if r == QMessageBox.Yes:
                import re
                m = re.search(r'(\d+)(?=\.[^.]+$)', Path(path).name)
                if m:
                    name = Path(path).name
                    return str(Path(path).with_name(name[:m.start()] + '#' * len(m.group(1)) + name[m.end():]))
    return path


def pick_ies_file(parent, current=''):
    """Ask for a light profile (an IES photometric file). Returns its path or ''."""
    from pathlib import Path
    start = current or QSettings().value('ui/ies_dir', '', type=str)
    path, _ = QFileDialog.getOpenFileName(parent, 'Choose a light profile', start, 'Light profiles (*.ies *.IES);;All files (*)')
    if path:
        QSettings().setValue('ui/ies_dir', str(Path(path).parent))
    return path or ''


def pick_volume_file(parent, current=''):
    """Ask for a volume: an OpenVDB file (or a numbered sequence of them) or a Volume prim in a USD file.
    Returns the source or ''."""
    from pathlib import Path
    start = QSettings().value('ui/volume_dir', '', type=str)
    if current:
        start = current.split('#/')[0]
    path, _ = QFileDialog.getOpenFileName(parent, 'Choose a volume', start,
                                          'Volumes (*.vdb *.usd *.usda *.usdc *.usdz);;OpenVDB (*.vdb);;USD (*.usd *.usda *.usdc *.usdz);;All files (*)')
    if not path:
        return ''
    QSettings().setValue('ui/volume_dir', str(Path(path).parent))
    ext = Path(path).suffix.lower()
    if ext in ('.usd', '.usda', '.usdc', '.usdz'):
        from PySide6.QtWidgets import QInputDialog, QMessageBox
        try:
            from ..io.usd import scan
            vols = [m for m in scan(path).meshes() if m.prim_type == 'Volume']
        except Exception as ex:
            QMessageBox.warning(parent, 'USD', f'Cannot read {Path(path).name}:\n{ex}')
            return ''
        if not vols:
            QMessageBox.information(parent, 'USD', f'{Path(path).name} has no volumes.')
            return ''
        labels = [f'{m.path}  ({m.motion})' for m in vols]
        item, ok = QInputDialog.getItem(parent, 'USD volume', 'Volume to use:', labels, 0, False)
        if not ok:
            return ''
        m = vols[labels.index(item)]
        return f'{path}#{m.path}' + ('?world' if m.motion == 'world' else '')
    if ext == '.vdb':
        from ..io.footage import find_sequence
        seq = find_sequence(Path(path))
        if seq and len(seq[0]) > 1:
            from PySide6.QtWidgets import QMessageBox
            r = QMessageBox.question(parent, 'Volume sequence',
                                     f'{Path(path).name} is part of a numbered sequence of {len(seq[0])} files. '
                                     'Use the whole sequence (one file per frame)?')
            if r == QMessageBox.Yes:
                import re
                m = re.search(r'(\d+)(?=\.[^.]+$)', Path(path).name)
                if m:
                    name = Path(path).name
                    return str(Path(path).with_name(name[:m.start()] + '#' * len(m.group(1)) + name[m.end():]))
    return path


def pick_footage_file(parent, current='', title='Choose an image sequence'):
    """Ask for an image sequence (pick any frame) or a video; returns its path or ''."""
    from pathlib import Path
    start = QSettings().value('ui/footage_dir', '', type=str) or current
    path, _ = QFileDialog.getOpenFileName(parent, title, current or start,
                                          'Images and video (*.png *.exr *.tif *.tiff *.jpg *.jpeg *.dpx *.mov *.mp4);;All files (*)')
    if path:
        QSettings().setValue('ui/footage_dir', str(Path(path).parent))
    return path


def pick_ocio_config(parent, current=''):
    path, _ = QFileDialog.getOpenFileName(parent, 'Choose an OCIO config', current, 'OCIO configs (*.ocio);;All files (*)')
    return path


# OCIO settings shown as a list of what the config offers (still editable, for names the config aliases)
OCIO_CHOICES = {'ocio_working': 'spaces', 'ocio_plate': 'spaces', 'exr_space': 'spaces', 'ocio_display': 'displays',
                'ocio_view': 'views', 'ocio_look': 'looks'}


def ocio_choices(doc, key):
    """Names for an OCIO setting from the scene's config, or [] if it cannot be read."""
    try:
        from ..io import ocio
        c = doc.scene.data['composite']
        cfg = ocio.load_config(c.get('ocio_config', ''))
        kind = OCIO_CHOICES[key]
        if kind == 'spaces':
            return [''] + ocio.colour_spaces(cfg)
        if kind == 'displays':
            return [''] + ocio.displays(cfg)
        if kind == 'views':
            return [''] + ocio.views(cfg, c.get('ocio_display', ''))
        return [''] + ocio.looks(cfg)
    except Exception:
        return []



def _same(a, b, decimals=2):
    """Two setting values equal for the eye (floats to a bit below the shown precision)."""
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y, decimals) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(float(a) - float(b)) <= 0.5 * 10 ** -(decimals + 1) + 1e-9 * max(abs(float(a)), abs(float(b)))
    return a == b


def _show(spec, v):
    """A value as the row would show it, for tooltips."""
    if spec.kind == 'bool':
        return 'on' if v else 'off'
    if spec.kind == 'enum':
        return next((t for val, t in spec.options if val == v), str(v))
    if spec.kind == 'float':
        return f'{v:.{spec.decimals}f}' + (f' {spec.unit}' if spec.unit else '')
    if spec.kind == 'int':
        return f'{int(v)}' + (f' {spec.unit}' if spec.unit else '')
    if spec.kind in ('vec3', 'color'):
        return '  '.join(f'{x:.2f}' for x in v)
    return str(v) or 'none'


LABEL_W = 140


class ParamRow(QWidget):
    """One setting: label, editor, a reset to the preset's value (when changed) and (if it can animate)
    a keyframe button."""

    def __init__(self, doc, path, spec: Param, parent=None, context=''):
        super().__init__(parent)
        self.doc = doc
        self.path = path
        self.spec = spec
        self._busy = False
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(6)
        tip = spec.tip + (f'  ({spec.unit})' if spec.unit and spec.unit not in spec.tip else '')
        if context:
            tip = f'<b>{context}</b><br>{tip}' if tip else context
        self.tip = tip or spec.label
        numeric = spec.kind in ('float', 'int')
        self.label = ScrubLabel(spec.label) if numeric else QLabel(spec.label)
        self.label.setToolTip(f'<b>{spec.label}</b><br>' + self.tip + ('<br><i>Drag the name to scrub, double-click it for the default.</i>' if numeric else ''))
        self.label.setFixedWidth(LABEL_W)
        self.label.setWordWrap(False)
        fm = self.label.fontMetrics()
        if fm.horizontalAdvance(spec.label) > LABEL_W - 4:
            self.label.setText(fm.elidedText(spec.label, Qt.ElideRight, LABEL_W - 4))
        lay.addWidget(self.label)
        self.editor = self._make_editor()
        self.editor.setToolTip(self.tip)
        lay.addWidget(self.editor, 1)
        self.reset_btn = ResetButton()
        self.reset_btn.clicked.connect(self._reset_to_preset)
        lay.addWidget(self.reset_btn)
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
        self.label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.label.customContextMenuRequested.connect(self._label_menu)
        self._changed = None
        self.refresh()

    # -- editors ------------------------------------------------------------------------------------

    def _make_editor(self):
        s = self.spec
        if s.kind in ('float', 'int'):
            if s.kind == 'float':
                lo = s.hard_lo if s.hard_lo is not None else (min(s.lo, 0.0) if s.lo >= 0 else s.lo * 10)
                hi = s.hard_hi if s.hard_hi is not None else max(s.hi * 10, s.hi + 1)
                if s.lo >= 0 and s.hard_lo is None:
                    lo = 0.0 if s.lo == 0 else min(s.lo, 0.0)
            else:
                lo = int(s.hard_lo if s.hard_lo is not None else min(s.lo, 0) - 100000 * (s.lo < 0))
                hi = int(s.hard_hi if s.hard_hi is not None else max(s.hi * 4, s.hi + 10))
            self.field = ValueField(s, self._to_t, self._from_t, lo, hi)
            self.field.edited.connect(self._field_edited)
            self.field.finished.connect(self.doc.end_drag)
            return self.field
        if s.kind == 'bool':
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            self.check = Switch()
            self.check.toggled.connect(lambda v: self._commit(bool(v), merge=False))
            h.addWidget(self.check)
            h.addStretch(1)
            return w
        if s.kind == 'enum':
            self.combo = QComboBox()
            self.combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            self.combo.setMinimumWidth(80)
            guard_wheel(self.combo)
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
                sb.setButtonSymbols(QAbstractSpinBox.NoButtons)
                sb.setStyleSheet(f'QDoubleSpinBox {{ border-left: 2px solid {col}; padding: 3px 3px 3px 4px; }}')
                sb.setToolTip(f'{axis}')
                sb.setMinimumWidth(42)
                sb.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
                guard_wheel(sb)
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
            pick.setText('Browse…')
            pick.setToolTip('Choose a file')
            pick.clicked.connect(self._browse)
            h.addWidget(self.line, 1)
            h.addWidget(pick)
            return w
        if s.key in OCIO_CHOICES:
            self.ocio = QComboBox()
            self.ocio.setEditable(True)
            self.ocio.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            self.ocio.setMinimumWidth(80)
            guard_wheel(self.ocio)
            self.ocio.lineEdit().setPlaceholderText('Default')
            self.ocio.activated.connect(lambda i: self._commit(self.ocio.itemText(i), merge=False))
            self.ocio.lineEdit().editingFinished.connect(lambda: self._commit(self.ocio.currentText().strip(), merge=False))
            return self.ocio
        self.line = QLineEdit()
        self.line.editingFinished.connect(lambda: self._commit(self.line.text(), merge=False))
        return self.line

    def _browse(self):
        key = self.spec.key
        if key == 'environment':
            path = pick_image_file(self, self.line.text())
        elif key == 'holdout_matte':
            path = pick_footage_file(self, self.line.text(), 'Choose the holdout matte (any frame)')
        elif key == 'holdout_depth':
            path = pick_footage_file(self, self.line.text(), 'Choose the depth pass (any frame)')
        elif key == 'ocio_config':
            path = pick_ocio_config(self, self.line.text())
        elif key == 'volume':
            path = pick_volume_file(self, self.line.text())
        elif key == 'profile':
            path = pick_ies_file(self, self.line.text())
        else:
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

    def _field_edited(self, v):
        if self._busy:
            return
        self._commit(int(round(v)) if self.spec.kind == 'int' else v)

    def _scrub(self, dt):
        cur = self.doc.value(self.path)
        t = self._to_t(cur) + dt
        self._commit(self._from_t(t))

    def _vec_changed(self, _):
        if self._busy:
            return
        self._commit(tuple(sb.value() for sb in self.vec))

    def _reset_to_preset(self):
        base = self.doc.baseline_value(self.path)
        if base is not None:
            self.doc.set(self.path, base, merge=False)

    def _label_menu(self, pos):
        s = self.spec
        m = QMenu(self)
        base = self.doc.baseline_value(self.path)
        a = m.addAction(f'Back to the preset’s value ({_show(s, base)})' if base is not None else 'Back to the preset’s value',
                        self._reset_to_preset)
        a.setEnabled(bool(self._changed))
        m.addAction(f'Back to the default ({_show(s, s.default)})', lambda: self.doc.set(self.path, s.default, merge=False))
        if self.key is not None:
            m.addSeparator()
            m.addAction('Set key here' if not self.doc.has_key(self.path) else 'Remove key here', lambda: self.doc.toggle_key(self.path))
            a = m.addAction('Clear animation', lambda: self.doc.clear_animation(self.path))
            a.setEnabled(self.doc.is_animated(self.path))
        if len(self.path) == 2:
            name = f'{self.path[0]}.{self.path[1]}'
            m.addSeparator()
            from PySide6.QtWidgets import QApplication
            m.addAction(f'Copy the command-line name ({name})', lambda: QApplication.clipboard().setText(name))
        m.exec(self.label.mapToGlobal(pos))

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
                self.field.set_value(v)
            elif s.kind == 'bool':
                self.check.setChecked(bool(v))
            elif s.kind == 'enum':
                i = self.combo.findData(v)
                self.combo.setCurrentIndex(max(0, i))
            elif s.kind == 'color':
                self.color.set_value(v)
            elif s.kind == 'vec3':
                for sb, x in zip(self.vec, v):
                    if not sb.hasFocus():
                        sb.setValue(float(x))
            elif getattr(self, 'ocio', None) is not None:
                names = ocio_choices(self.doc, s.key)
                if [self.ocio.itemText(i) for i in range(self.ocio.count())] != names:
                    self.ocio.clear()
                    self.ocio.addItems(names)
                self.ocio.setEditText(str(v))
            else:
                self.line.setText(str(v))
            animated = self.doc.is_animated(self.path)
            if self.key is not None:
                self.key.set_state(2 if self.doc.has_key(self.path) else (1 if animated else 0))
            base = None if animated else self.doc.baseline_value(self.path)
            changed = base is not None and not _same(v, base, s.decimals)
            if changed != self._changed:
                self._changed = changed
                self.label.setStyleSheet(f'color: {theme.CHANGED};' if changed else '')
            self.reset_btn.set_active(changed, f'Changed from the preset. Click to put back its value: {_show(s, base)}' if changed else '')
            if self.path == ('domain', 'resolution'):
                self._memory_tip()
        finally:
            self._busy = False

    def _memory_tip(self):
        """Voxels: what a final render needs of the GPU's memory at this setting, and what the GPU has room for."""
        from ..engine.gpu import GB, card
        try:
            _res, _up, need, cut = self.doc.scene.memory_plan(final=True)
        except Exception:   # (a scene being replaced)
            return
        plan = card()['plan']
        self.editor.setToolTip(self.tip + f'<br><br>A final render needs about {need / GB:.1f} GB of GPU memory'
                               + (f', of the {plan / GB:.1f} GB this GPU has room for' if plan else '') + '.'
                               + (f'<br>{cut}' if cut else ''))


# Groups of expert settings that start closed (a click opens them, and they then stay open)
CLOSED = {'composite/OCIO', 'composite/Holdouts from footage', 'composite/No footage', 'domain/Solver', 'domain/Cache',
          'motion/Force masks', 'shading/Quality', 'liquid/Quality', 'water/Quality', 'embers/Collisions', 'camera/Clipping',
          'atmosphere/Detail'}


class Group(QFrame):
    """A collapsible card of rows. Whether it is open is remembered (by `memo`) across sessions."""

    def __init__(self, title, parent=None, memo=None, hint=''):
        super().__init__(parent)
        self.setObjectName('card')
        self.memo = memo
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 4, 8, 6)
        v.setSpacing(0)
        self.head = QToolButton()
        self.head.setText(title)
        self.head.setCheckable(True)
        self.head.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.head.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.head.setCursor(Qt.PointingHandCursor)
        self.head.setStyleSheet(f'QToolButton {{ color: {theme.TEXT}; font-weight: 600; border: 0; padding: 5px 0; text-align: left; background: transparent; }}'
                                f'QToolButton:hover {{ color: #ffffff; background: transparent; }}'
                                f'QToolButton:checked {{ background: transparent; border: 0; }}')
        if hint:
            self.head.setToolTip(hint)
        v.addWidget(self.head)
        self.body = QWidget()
        self.lay = QVBoxLayout(self.body)
        self.lay.setContentsMargins(0, 2, 0, 2)
        self.lay.setSpacing(1)
        v.addWidget(self.body)
        is_open = True if memo is None else QSettings().value(f'ui/open/{memo}', memo not in CLOSED, type=bool)
        self.head.setChecked(is_open)
        self._toggle(is_open, save=False)
        self.head.toggled.connect(self._toggle)

    def _toggle(self, on, save=True):
        self.body.setVisible(on)
        self.head.setIcon(self._chevron(on))
        if save and self.memo is not None:
            QSettings().setValue(f'ui/open/{self.memo}', bool(on))

    @staticmethod
    def _chevron(down):
        from PySide6.QtGui import QIcon, QPixmap
        pm = QPixmap(24, 24)
        pm.setDevicePixelRatio(2.0)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(theme.MUTED), 1.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        if down:
            p.drawPolyline(QPolygonF([QPointF(3, 4.5), QPointF(6, 7.5), QPointF(9, 4.5)]))
        else:
            p.drawPolyline(QPolygonF([QPointF(4.5, 3), QPointF(7.5, 6), QPointF(4.5, 9)]))
        p.end()
        return QIcon(pm)


class ParamPanel(QWidget):
    """Rows for a list of parameters in cards by group, with advanced ones behind a toggle.
    `memo` names the panel so each group remembers whether it is open."""

    def __init__(self, doc, make_path, params, parent=None, header=None, memo=None, show_adv=None):
        super().__init__(parent)
        self.doc = doc
        self.rows = []
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 4, 10, 12)
        v.setSpacing(8)
        if header is not None:
            v.addWidget(header)
        if show_adv is None:
            show_adv = QSettings().value('ui/advanced', False, type=bool)
        groups = {}
        order = []
        self.hidden_advanced = 0
        for p in params:
            if p.advanced and not show_adv:
                self.hidden_advanced += 1
                continue
            g = p.group or 'Settings'
            if g not in groups:
                groups[g] = Group(g, memo=f'{memo}/{g}' if memo else None)
                order.append(g)
            row = ParamRow(doc, make_path(p.key), p)
            groups[g].lay.addWidget(row)
            self.rows.append(row)
        for g in order:
            v.addWidget(groups[g])
        if self.hidden_advanced:
            more = QLabel(f'{self.hidden_advanced} expert setting{"s" if self.hidden_advanced > 1 else ""} hidden · '
                          '<a href="adv" style="color:%s; text-decoration:none;">Show expert settings</a>' % theme.ACCENT_HI)
            more.setObjectName('faint')
            more.setTextInteractionFlags(Qt.LinksAccessibleByMouse)
            more.linkActivated.connect(lambda _: self._show_adv())
            more.setContentsMargins(4, 2, 0, 0)
            v.addWidget(more)
        v.addStretch(1)

    def _show_adv(self):
        w = self.window()
        insp = getattr(w, 'props', None)
        if insp is not None and hasattr(insp, 'adv'):
            insp.adv.setChecked(True)

    def refresh(self, path=None):
        for r in self.rows:
            if path is None or r.path == path:
                r.refresh()

    def refresh_animated(self):
        for r in self.rows:
            if r.key is not None:
                r.refresh()
