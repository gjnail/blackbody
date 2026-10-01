"""The bar over the viewer while the ground is lined up: how the camera's scale is known (its height, or one side of
the rectangle), the lens, and what has been worked out so far."""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QMenu, QPushButton, QToolButton,
                               QVBoxLayout)

from ..scene import groundmatch as GM
from . import theme
from .params import guard_wheel


class GroundBar(QFrame):
    def __init__(self, viewport):
        super().__init__(viewport)
        self.vp = viewport
        self.setObjectName('groundbar')
        self.setStyleSheet(f'QFrame#groundbar {{ background: rgba(24,24,28,238); border: 1px solid {theme.ACCENT}; border-radius: 9px; }}'
                           'QLabel { background: transparent; }')
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 8, 10, 8)
        v.setSpacing(6)
        self.head = QLabel()
        self.head.setWordWrap(True)
        v.addWidget(self.head)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.mode = QComboBox()
        self.mode.addItem('From a rectangle on the ground', 'rect')
        self.mode.addItem('From the horizon (no rectangle in view)', 'horizon')
        self.mode.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.mode.setToolTip('No floor tiles, road markings or anything rectangular? Drag a line along the horizon (or where '
                             'it would be: the sea’s edge, the far end of a flat field), give the lens and the camera’s height.')
        guard_wheel(self.mode)
        row.addWidget(self.mode)
        self.slopes = QCheckBox('The ground slopes')
        self.slopes.setToolTip('Show two upright lines to drag along things that are truly vertical (poles, door frames, wall '
                               'corners, tree trunks): they say which way is up, so a hillside or a ramp-like street stays a '
                               'slope, with gravity straight down')
        row.addWidget(self.slopes)
        row.addStretch(1)
        v.addLayout(row)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.scale_by = QComboBox()
        self.scale_by.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.scale_by.addItem('Camera height', 'height')
        self.scale_by.addItem('Side length', 'side')
        self.scale_by.setToolTip('What you know: how high the camera was, or how long the side between the first two corners is')
        guard_wheel(self.scale_by)
        row.addWidget(self.scale_by)
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(0.05, 2000.0)
        self.size_spin.setDecimals(2)
        self.size_spin.setSuffix(' m')
        self.size_spin.setFixedWidth(88)
        guard_wheel(self.size_spin)
        row.addWidget(self.size_spin)
        self.presets = QToolButton()
        self.presets.setText('Typical…')
        self.presets.setToolTip('Typical camera heights')
        self.presets.setPopupMode(QToolButton.InstantPopup)
        m = QMenu(self.presets)
        for label, h in GM.HEIGHTS:
            m.addAction(f'{label} ({h:g} m)', lambda h=h: (self.scale_by.setCurrentIndex(0), self.size_spin.setValue(h)))
        self.presets.setMenu(m)
        row.addWidget(self.presets)
        row.addSpacing(10)
        self.lens_auto = QCheckBox('Lens from the picture')
        self.lens_auto.setToolTip('Work out the focal length from how the grid’s sides converge. Untick to give the '
                                  'lens it was shot with (needed when the sides run parallel in the picture).')
        row.addWidget(self.lens_auto)
        self.focal = QDoubleSpinBox()
        self.focal.setRange(4.0, 1200.0)
        self.focal.setDecimals(1)
        self.focal.setSuffix(' mm')
        self.focal.setFixedWidth(84)
        self.focal.setToolTip('The focal length the shot was filmed with, on the sensor in Camera › Sensor width')
        guard_wheel(self.focal)
        row.addWidget(self.focal)
        row.addStretch(1)
        v.addLayout(row)
        self.info = QLabel()
        self.info.setObjectName('hint')
        self.info.setWordWrap(True)
        v.addWidget(self.info)
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addStretch(1)
        self.flat = QPushButton('Pin in 2D instead')
        self.flat.setObjectName('ghost')
        self.flat.setToolTip('No matched camera: the effect is pinned to a point of the frame and scaled by hand (for '
                             'shots with no visible ground)')
        self.flat.clicked.connect(self.vp.ground_flat)
        row.addWidget(self.flat)
        cancel = QPushButton('Cancel')
        cancel.clicked.connect(self.vp.ground_cancel)
        row.addWidget(cancel)
        done = QPushButton('Done')
        done.setObjectName('primary')
        done.clicked.connect(self.vp.ground_done)
        row.addWidget(done)
        v.addLayout(row)
        self._quiet = False
        self.scale_by.currentIndexChanged.connect(lambda *_: self._changed())
        self.mode.currentIndexChanged.connect(lambda *_: self._changed())
        self.slopes.toggled.connect(lambda *_: self._changed())
        self.size_spin.valueChanged.connect(lambda *_: self._changed())
        self.lens_auto.toggled.connect(lambda *_: self._changed())
        self.focal.valueChanged.connect(lambda *_: self._changed())

    def load(self, g, focal):
        self._quiet = True
        self.mode.setCurrentIndex(1 if g.get('mode') == 'horizon' else 0)
        self.slopes.setChecked(bool(g.get('verticals')))
        self.scale_by.setCurrentIndex(1 if g.get('scale_by') == 'side' else 0)
        self.size_spin.setValue(float(g.get('side', 2.0)) if g.get('scale_by') == 'side' else float(g.get('height', 1.6)))
        self.lens_auto.setChecked(g.get('lens', 'picture') == 'picture')
        self.focal.setValue(float(g.get('focal_mm') or focal))
        self.focal.setEnabled(not self.lens_auto.isChecked())
        self._changed()   # the words and the controls for this way of lining up (quietly: nothing is solved again)
        self._quiet = False

    def settings(self):
        by = self.scale_by.currentData() if self.mode.currentData() == 'rect' else 'height'
        out = {'scale_by': by, 'lens': 'picture' if self.lens_auto.isChecked() else 'known', 'focal_mm': float(self.focal.value()),
               'mode': self.mode.currentData(), 'slopes': self.slopes.isChecked()}
        out['side' if by == 'side' else 'height'] = float(self.size_spin.value())
        return out

    def _changed(self):
        rect = self.mode.currentData() == 'rect'
        tip = ('drag the four corners onto something rectangular lying on the ground (floor tiles, a rug, a road, a parking '
               'bay, a table top). Drag inside to move the whole grid.' if rect else
               'drag the line along the horizon (the sea’s edge, the far end of a flat field, or where the ground '
               'would meet the sky) and the cross to where the effect goes; give the lens and the camera’s height.')
        if self.slopes.isChecked():
            tip += ' Drag the blue lines along two upright edges (poles, door frames, wall corners).'
        self.head.setText(f'<b>Line up the ground</b> · {tip}')
        self.scale_by.setVisible(rect)
        self.lens_auto.setVisible(rect)
        if not rect:
            self.scale_by.setCurrentIndex(0)
            self.lens_auto.setChecked(False)
        self.focal.setEnabled(not self.lens_auto.isChecked())
        self.presets.setEnabled(self.scale_by.currentData() == 'height')
        if not self._quiet:
            self.vp.ground_settings_changed()

    def show_match(self, m, error=None, focal_mm=None):
        if error:
            self.info.setText(f'<span style="color:{theme.BAD}">{error}</span>')
            return
        lens = (f'lens {focal_mm:.0f} mm' + (' (from the picture)' if m.focal_solved else '')) if focal_mm else ''
        rect = f' · the rectangle is {m.sides[0]:.2f} × {m.sides[1]:.2f} m' if m.sides[0] else ''
        slope = f' · the ground slopes {m.slope:.1f}°' if m.slope > 0.5 else ''
        self.info.setText(f'Camera {m.height:.2f} m up, looking down {m.tilt:.0f}°, {lens}, horizon leaning {m.roll:+.1f}°{rect}{slope}. '
                          f'The figure is {GM.PERSON:g} m tall: drag it next to a person in the shot to check.')
