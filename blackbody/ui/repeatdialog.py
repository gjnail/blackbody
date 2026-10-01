"""Repeat: copies of what is selected in a row, in a ring, or scattered over the ground, with a sketch of where
they go."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSettings, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)

from ..scene import arrange
from . import theme
from .params import guard_wheel

PATTERNS = (('row', 'In a row', 'A line of them: torches along a path, lamps down a street'),
            ('ring', 'In a ring', 'Around a circle: flame jets round a stage, a ring of fire'),
            ('scatter', 'Scattered', 'At random over the ground: spot fires, debris, puddles'))


class Sketch(QWidget):
    """The ground from above: the original's footprint and where each copy lands."""

    def __init__(self, dialog):
        super().__init__(dialog)
        self.dlg = dialog
        self.setMinimumSize(300, 190)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(QColor(theme.LINE), 1))
        p.setBrush(QColor(theme.VIEWER))
        p.drawRoundedRect(r, 8, 8)
        places, ring = self.dlg.places()
        w, dpt = self.dlg.size_xz
        pts = [(0.0, 0.0, 0.0)] + list(places) if not ring else list(places)
        xs = [x for x, _, _ in pts] + [0.0]
        zs = [z for _, z, _ in pts] + [0.0]
        span = max(max(xs) - min(xs) + w, max(zs) - min(zs) + dpt, 1e-3)
        s = min(r.width(), r.height()) * 0.8 / span
        cx, cz = (max(xs) + min(xs)) / 2, (max(zs) + min(zs)) / 2

        def at(x, z):
            return QPointF(r.center().x() + (x - cx) * s, r.center().y() + (z - cz) * s)

        def box(x, z, turn, colour, fill):
            c = at(x, z)
            p.save()
            p.translate(c)
            p.rotate(-turn)   # Rotation turns +x toward -z: anticlockwise seen from above
            p.setPen(QPen(QColor(colour), 1.3))
            p.setBrush(QColor(fill))
            hw, hd = max(w * s / 2, 3), max(dpt * s / 2, 3)
            p.drawRect(QRectF(-hw, -hd, 2 * hw, 2 * hd))
            p.drawLine(QPointF(0, 0), QPointF(hw, 0))   # which way it faces
            p.restore()

        if ring:
            p.setPen(QPen(QColor(theme.LINE_HI), 1, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            c = at(0, 0)
            p.drawEllipse(c, self.dlg.radius.value() * s, self.dlg.radius.value() * s)
            p.drawText(QRectF(c.x() - 60, c.y() - 8, 120, 16), Qt.AlignCenter, '·')
        else:
            box(0, 0, 0, theme.ACCENT, QColor(255, 140, 60, 70))
        for k, (x, z, turn) in enumerate(places):
            if ring and k == 0:
                box(x, z, turn, theme.ACCENT, QColor(255, 140, 60, 70))
            else:
                box(x, z, turn, theme.MUTED, QColor(255, 255, 255, 18))
        p.setPen(QColor(theme.FAINT))
        p.drawText(r.adjusted(8, 4, -8, -4), Qt.AlignLeft | Qt.AlignTop, 'Seen from above · orange: the one you have')


class RepeatDialog(QDialog):
    def __init__(self, parent, n_things, size_xz):
        super().__init__(parent)
        self.setWindowTitle('Repeat')
        self.size_xz = (max(size_xz[0], 0.05), max(size_xz[1], 0.05))
        st = QSettings()
        v = QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 14)
        v.setSpacing(10)
        head = QLabel(f'Copies of what is selected ({n_things} thing{"s" if n_things != 1 else ""}, with what is attached to it).')
        head.setObjectName('faint')
        head.setWordWrap(True)
        v.addWidget(head)
        row = QHBoxLayout()
        row.setSpacing(0)
        self.pattern = QButtonGroup(self)
        for k, (key, label, tip) in enumerate(PATTERNS):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setObjectName('segFirst' if k == 0 else 'segLast' if k == len(PATTERNS) - 1 else 'seg')
            b.setProperty('key', key)
            self.pattern.addButton(b, k)
            row.addWidget(b)
        keys = [p[0] for p in PATTERNS]
        saved = st.value('ui/repeat_pattern', 'row', type=str)
        self.pattern.button(keys.index(saved) if saved in keys else 0).setChecked(True)
        row.addStretch(1)
        v.addLayout(row)
        body = QHBoxLayout()
        body.setSpacing(14)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        w = self.size_xz[0]
        self.count = QSpinBox()
        self.count.setRange(1, 200)
        self.count.setValue(4)
        self.spacing = self._spin(round(w * 1.5, 2), 0.01, 500.0, 'How far apart, middle to middle')
        self.angle = self._spin(0.0, -180.0, 180.0, 'Which way the row runs: 0 is to the right (+x), 90 away from the default camera', ' °', 0)
        self.radius = self._spin(round(max(w * 1.6, 0.2), 2), 0.01, 500.0, 'Radius of the circle, about where the original is now')
        self.face = QCheckBox('Turn each to face out')
        self.face.setChecked(True)
        self.area_w = self._spin(round(w * 6, 2), 0.01, 1000.0, 'Width of the area (x)')
        self.area_d = self._spin(round(w * 6, 2), 0.01, 1000.0, 'Depth of the area (z)')
        self.turn = QCheckBox('Turn each at random')
        self.turn.setChecked(True)
        self.seed = QSpinBox()
        self.seed.setRange(0, 9999)
        self.seed.setToolTip('Another number scatters them differently')
        for wdg in (self.count, self.seed):
            guard_wheel(wdg)
        self.rows = {}
        self.count_label = QLabel()
        form.addRow(self.count_label, self.count)
        for key, label, wdg in (('row', 'Spacing', self.spacing), ('row', 'Direction', self.angle), ('ring', 'Radius', self.radius),
                                ('ring', '', self.face), ('scatter', 'Area width', self.area_w), ('scatter', 'Area depth', self.area_d),
                                ('scatter', '', self.turn), ('scatter', 'Variation', self.seed)):
            lab = QLabel(label)
            form.addRow(lab, wdg)
            self.rows.setdefault(key, []).append((lab, wdg))
        fw = QWidget()
        fw.setLayout(form)
        body.addWidget(fw)
        self.sketch = Sketch(self)
        body.addWidget(self.sketch, 1)
        v.addLayout(body, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton('Cancel')
        cancel.clicked.connect(self.reject)
        ok = QPushButton('Repeat')
        ok.setObjectName('primary')
        ok.setDefault(True)
        ok.clicked.connect(self.accept)
        row.addWidget(cancel)
        row.addWidget(ok)
        v.addLayout(row)
        for wdg in (self.count, self.spacing, self.angle, self.radius, self.area_w, self.area_d, self.seed):
            wdg.valueChanged.connect(self._changed)
        for wdg in (self.face, self.turn):
            wdg.toggled.connect(self._changed)
        self.pattern.idToggled.connect(lambda *_: self._changed())
        self.resize(640, 360)
        self._changed()

    def _spin(self, value, lo, hi, tip, suffix=' m', decimals=2):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setSuffix(suffix)
        s.setValue(value)
        s.setToolTip(tip)
        s.setFixedWidth(110)
        guard_wheel(s)
        return s

    def key(self):
        return self.pattern.checkedButton().property('key')

    def _changed(self):
        key = self.key()
        for k, rows in self.rows.items():
            for lab, wdg in rows:
                lab.setVisible(k == key)
                wdg.setVisible(k == key)
        self.count_label.setText('How many in all' if key == 'ring' else 'How many more')
        if key == 'ring' and self.count.value() < 2:
            self.count.setValue(2)
        self.sketch.update()

    def places(self):
        key = self.key()
        places = arrange.layout(key, self.count.value(), spacing=self.spacing.value(), angle=self.angle.value(),
                                radius=self.radius.value(), area=(self.area_w.value(), self.area_d.value()),
                                face_out=self.face.isChecked(), random_turn=self.turn.isChecked(), seed=self.seed.value())
        return places, key == 'ring'

    def accept(self):
        QSettings().setValue('ui/repeat_pattern', self.key())
        super().accept()


def ask(parent, n_things, size_xz):
    """([(dx, dz, turn)], ring) for the copies, or None."""
    d = RepeatDialog(parent, n_things, size_xz)
    return d.places() if d.exec() == QDialog.Accepted else None
