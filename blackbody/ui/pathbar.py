"""The bar over the viewer while a path is being drawn for an object: how long it takes, and Done / Cancel."""
from __future__ import annotations

from PySide6.QtWidgets import QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QPushButton

from . import theme
from .params import guard_wheel


class PathBar(QFrame):
    def __init__(self, viewport):
        super().__init__(viewport)
        self.vp = viewport
        self.setObjectName('pathbar')
        self.setStyleSheet(f'QFrame#pathbar {{ background: rgba(24,24,28,235); border: 1px solid {theme.ACCENT}; border-radius: 9px; }}'
                           'QLabel { background: transparent; }')
        h = QHBoxLayout(self)
        h.setContentsMargins(12, 6, 8, 6)
        h.setSpacing(8)
        self.label = QLabel()
        self.label.setWordWrap(True)
        h.addWidget(self.label, 1)
        h.addWidget(QLabel('Takes'))
        self.secs = QDoubleSpinBox()
        self.secs.setRange(0.1, 600.0)
        self.secs.setValue(3.0)
        self.secs.setSuffix(' s')
        self.secs.setDecimals(1)
        self.secs.setFixedWidth(72)
        self.secs.setToolTip('How long it takes to go along the path, from this frame')
        guard_wheel(self.secs)
        h.addWidget(self.secs)
        done = QPushButton('Done')
        done.setObjectName('primary')
        done.clicked.connect(self.vp.path_finish)
        h.addWidget(done)
        cancel = QPushButton('Cancel')
        cancel.clicked.connect(self.vp.path_cancel)
        h.addWidget(cancel)
        self.setFixedWidth(560)

    def sync(self, name, n, frame):
        self.label.setText(f'<b>Path for {name}</b> · click points on the ground where it goes ({n} so far), from frame {frame}. '
                           'Enter or Done when it is there.')
        self.adjustSize()
