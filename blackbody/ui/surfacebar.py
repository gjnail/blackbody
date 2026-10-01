"""The bar over the viewer while a surface (a wall, a table, a ramp, stairs) is lined up in the footage."""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QPushButton, QSpinBox,
                               QVBoxLayout)

from . import surfaces, theme
from .params import guard_wheel


class SurfaceBar(QFrame):
    def __init__(self, viewport):
        super().__init__(viewport)
        self.vp = viewport
        self.setObjectName('surfacebar')
        self.setStyleSheet(f'QFrame#surfacebar {{ background: rgba(24,24,28,238); border: 1px solid {theme.ACCENT}; border-radius: 9px; }}'
                           'QLabel { background: transparent; }')
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 8, 10, 8)
        v.setSpacing(6)
        self.head = QLabel()
        self.head.setWordWrap(True)
        v.addWidget(self.head)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.kind = QComboBox()
        for k, label in surfaces.KINDS:
            self.kind.addItem(label, k)
        self.kind.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        guard_wheel(self.kind)
        row.addWidget(self.kind)
        self.h_label = QLabel('Height')
        row.addWidget(self.h_label)
        self.h_spin = QDoubleSpinBox()
        self.h_spin.setRange(0.02, 50.0)
        self.h_spin.setDecimals(2)
        self.h_spin.setSuffix(' m')
        self.h_spin.setValue(0.75)
        self.h_spin.setFixedWidth(84)
        guard_wheel(self.h_spin)
        row.addWidget(self.h_spin)
        self.solid = QCheckBox('Solid to the ground')
        self.solid.setChecked(True)
        self.solid.setToolTip('A step or a platform is solid all the way down; a table top is not (untick it)')
        row.addWidget(self.solid)
        self.s_label = QLabel('Steps')
        row.addWidget(self.s_label)
        self.steps = QSpinBox()
        self.steps.setRange(1, 60)
        self.steps.setValue(4)
        guard_wheel(self.steps)
        row.addWidget(self.steps)
        row.addStretch(1)
        v.addLayout(row)
        self.info = QLabel()
        self.info.setObjectName('hint')
        self.info.setWordWrap(True)
        v.addWidget(self.info)
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton('Cancel')
        cancel.clicked.connect(self.vp.surface_cancel)
        row.addWidget(cancel)
        self.done = QPushButton('Add it')
        self.done.setObjectName('primary')
        self.done.clicked.connect(self.vp.surface_done)
        row.addWidget(self.done)
        v.addLayout(row)
        for w in (self.kind,):
            w.currentIndexChanged.connect(lambda *_: self._changed())
        for w in (self.h_spin, self.steps):
            w.valueChanged.connect(lambda *_: self._changed())
        self.solid.toggled.connect(lambda *_: self._changed())
        self._changed(quiet=True)

    def settings(self):
        k = self.kind.currentData()
        return {'kind': k, 'height': float(self.h_spin.value()), 'rise': float(self.h_spin.value()),
                'steps': int(self.steps.value()), 'solid': self.solid.isChecked()}

    def _changed(self, quiet=False):
        k = self.kind.currentData()
        self.h_label.setText('Step height' if k == 'stairs' else 'Height')
        if k == 'stairs' and self.h_spin.value() > 0.4:
            self.h_spin.blockSignals(True)
            self.h_spin.setValue(0.17)
            self.h_spin.blockSignals(False)
        elif k == 'level' and self.h_spin.value() < 0.2:
            self.h_spin.blockSignals(True)
            self.h_spin.setValue(0.75)
            self.h_spin.blockSignals(False)
        for w in (self.h_label, self.h_spin):
            w.setVisible(k in ('level', 'stairs'))
        self.solid.setVisible(k == 'level')
        for w in (self.s_label, self.steps):
            w.setVisible(k == 'stairs')
        tips = {'wall': 'drag the grid onto the wall (or a box, a car side, a fence): its <b>bottom edge</b> where it meets the ground.',
                'level': 'drag the grid onto the top of it (a table, a step, a platform), and give how high it is.',
                'ramp': 'drag the grid onto the ramp: its <b>bottom edge</b> where it meets the ground, the far edge up the slope.',
                'stairs': 'drag the grid onto the top of the <b>lowest step</b>: its front edge nearest you. Give the step height and how many.'}
        self.head.setText(f'<b>Add a surface</b> · {tips[k]}')
        if not quiet:
            self.vp.surface_changed()

    def show_result(self, srf, error=None):
        if error:
            self.info.setText(f'<span style="color:{theme.BAD}">{error}</span>')
            self.done.setEnabled(False)
            return
        import numpy as np
        top = np.asarray(srf['top'], float)
        a, b = float(np.linalg.norm(top[1] - top[0])), float(np.linalg.norm(top[2] - top[1]))
        extra = ''
        if srf['kind'] == 'ramp':
            n = np.cross(top[1] - top[0], top[2] - top[1])
            n /= max(np.linalg.norm(n), 1e-9)
            extra = f', rising {np.degrees(np.arccos(min(1.0, abs(n[1])))):.0f}°'
        self.info.setText(f'{a:.2f} × {b:.2f} m{extra}. It becomes a solid where it is in the footage: effects meet it, and '
                          'it hides them behind it.')
        self.done.setEnabled(True)
