"""The dialog for a logo or picture block: the picture, how big, and what of it is the shape, with a preview."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QDialog, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QSlider,
                               QVBoxLayout)

from . import textmesh, theme
from .params import guard_wheel
from .textdialog import LetterPreview


def pick_picture(parent):
    start = QSettings().value('ui/picture_dir', str(Path.home()), type=str)
    path, _ = QFileDialog.getOpenFileName(parent, 'Choose a logo or picture', start, textmesh.IMAGE_FILTER + ';;All files (*)')
    if path:
        QSettings().setValue('ui/picture_dir', str(Path(path).parent))
    return path


class ShapeDialog(QDialog):
    def __init__(self, parent, spec, width=2.0, look='fire', edit=False):
        super().__init__(parent)
        self.setWindowTitle('Edit shape' if edit else 'Logo or picture')
        self.setMinimumWidth(560)
        self.width_m = max(float(width), 0.05)
        self._auto = not edit
        self._quiet = False
        self._wide = 0.0
        self.image = spec['image']
        v = QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 14)
        v.setSpacing(10)
        head = QLabel('The shape is what stands out in the picture: its transparency if it has any, else what differs from '
                      'its background. It stands on the ground, as thick as Depth.')
        head.setWordWrap(True)
        head.setObjectName('faint')
        v.addWidget(head)
        row = QHBoxLayout()
        self.file = QLabel()
        self.file.setStyleSheet('font-weight: 600;')
        row.addWidget(self.file, 1)
        choose = QPushButton('Choose another…')
        choose.clicked.connect(self._choose)
        row.addWidget(choose)
        v.addLayout(row)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel('Height'))
        self.height = self._spin(spec['height'], 0.01, 100.0, 'How tall the shape is')
        row.addWidget(self.height)
        row.addSpacing(10)
        row.addWidget(QLabel('Depth'))
        self.depth = self._spin(spec['depth'], 0.002, 20.0, 'How thick it is, front to back')
        row.addWidget(self.depth)
        row.addStretch(1)
        self.size_label = QLabel()
        self.size_label.setObjectName('faint')
        row.addWidget(self.size_label)
        v.addLayout(row)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.invert = QCheckBox('Invert')
        self.invert.setChecked(bool(spec.get('invert')))
        self.invert.setToolTip('Use the other part of the picture: the background instead of the shape')
        row.addWidget(self.invert)
        row.addSpacing(10)
        row.addWidget(QLabel('Threshold'))
        self.threshold = QSlider(Qt.Horizontal)
        self.threshold.setRange(5, 95)
        self.threshold.setValue(int(round(float(spec.get('threshold', 0.5)) * 100)))
        self.threshold.setToolTip('How strongly a pixel must stand out to be part of the shape: lower takes in fainter parts')
        guard_wheel(self.threshold)
        row.addWidget(self.threshold, 1)
        v.addLayout(row)
        self.preview = LetterPreview()
        self.preview.look = look
        self.preview.setMinimumHeight(180)
        v.addWidget(self.preview, 1)
        self.error = QLabel()
        self.error.setStyleSheet(f'color: {theme.BAD};')
        self.error.setWordWrap(True)
        v.addWidget(self.error)
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton('Cancel')
        cancel.clicked.connect(self.reject)
        self.ok = QPushButton('Change' if edit else 'Add')
        self.ok.setObjectName('primary')
        self.ok.setDefault(True)
        self.ok.clicked.connect(self.accept)
        row.addWidget(cancel)
        row.addWidget(self.ok)
        v.addLayout(row)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._trace)
        self.invert.toggled.connect(lambda *_: self._timer.start())
        self.threshold.valueChanged.connect(lambda *_: self._timer.start())
        self.height.valueChanged.connect(lambda *_: self._sized())
        self.depth.valueChanged.connect(lambda *_: self._sized())
        self.resize(620, 480)
        self._trace()

    def _spin(self, value, lo, hi, tip):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(3)
        s.setSingleStep(0.01)
        s.setSuffix(' m')
        s.setValue(float(value))
        s.setFixedWidth(96)
        s.setToolTip(tip)
        guard_wheel(s)
        return s

    def _choose(self):
        p = pick_picture(self)
        if p:
            self.image = p
            self._auto = True
            self._trace()

    def _trace(self):
        self.file.setText(Path(self.image).name)
        try:   # a quick, coarser trace for the preview
            cs = textmesh.image_contours(self.image, 1.0, self.invert.isChecked(), self.threshold.value() / 100.0, max_px=320)
        except (ValueError, OSError) as ex:
            self.preview.set_path(None)
            self.error.setText(str(ex))
            self.ok.setEnabled(False)
            self._wide = 0.0
            self._label()
            return
        import numpy as np
        self.error.setText('')
        self.ok.setEnabled(True)
        allp = np.concatenate(cs)
        self._wide = float(np.ptp(allp[:, 0]) / max(np.ptp(allp[:, 1]), 1e-6))
        self.preview.set_path(textmesh.contour_path(cs))
        if self._auto:   # a third of the scene tall, or less so it takes no more than 85% of its width
            h = float(f'{min(0.35 * self.width_m, 0.85 * self.width_m / max(self._wide, 1e-6)):.2g}')
            self._quiet = True
            self.height.setValue(h)
            self.depth.setValue(float(f'{max(0.1 * h, 0.005):.2g}'))
            self._quiet = False
        self._label()

    def _sized(self):
        if not self._quiet:
            self._auto = False
        self._label()

    def _label(self):
        self.size_label.setText(f'{self._wide * self.height.value():.2f} m wide' if self._wide else '')

    def spec(self):
        return textmesh.image_spec(self.image, self.height.value(), self.depth.value(), self.invert.isChecked(),
                                   self.threshold.value() / 100.0)


def ask(parent, spec=None, width=2.0, look='fire', edit=False):
    """The shape spec the user asked for, or None. Without a spec, a picture is asked for first."""
    if spec is None:
        p = pick_picture(parent)
        if not p:
            return None
        spec = textmesh.image_spec(p)
    d = ShapeDialog(parent, spec, width, look, edit)
    return d.spec() if d.exec() == QDialog.Accepted else None
