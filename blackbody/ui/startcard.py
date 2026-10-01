"""The card on an empty stage in Build: one click puts in a first thing to build from (fire, water, cloth,
smoke, snow or a solid object), or start from a ready-made effect, a project or your footage."""
from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QAbstractButton, QFrame, QGridLayout, QLabel, QPushButton, QVBoxLayout

from . import icons, theme

STARTERS = [('burner', 'flame', 'Fire', theme.ACCENT), ('pour', 'pour', 'Water', '#6fb6ff'), ('curtain', 'fabric', 'Cloth', '#d59cff'),
            ('smoke', 'cloud', 'Smoke', '#c9c9d1'), ('snow', 'snow', 'Snow', '#a8e0ff'), ('box', 'cube', 'An object', '#9fb0c4')]


class BigTile(QAbstractButton):
    def __init__(self, glyph, label, colour, parent=None):
        super().__init__(parent)
        self.glyph, self.label, self.colour = glyph, label, colour
        self.setFixedSize(QSize(104, 92))
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        hot = self.underMouse()
        p.setPen(QPen(QColor(self.colour if hot else theme.LINE_HI), 1.2))
        p.setBrush(QColor(theme.FIELD_HI if hot else theme.FIELD))
        p.drawRoundedRect(r, 10, 10)
        icons.paint_glyph(p, self.glyph, QRectF(r.center().x() - 17, r.y() + 14, 34, 34), self.colour, 1.8)
        f = QFont(self.font())
        f.setPointSizeF(9.5)
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        p.setPen(QColor(theme.TEXT))
        p.drawText(QRectF(r.x(), r.y() + 56, r.width(), 24), Qt.AlignCenter, self.label)


class StartCard(QFrame):
    def __init__(self, win, parent):
        super().__init__(parent)
        self.win = win
        self.setObjectName('startcard')
        self.setStyleSheet(f'QFrame#startcard {{ background: rgba(24,24,28,240); border: 1px solid {theme.LINE_HI}; border-radius: 14px; }}'
                           'QLabel { background: transparent; }')
        v = QVBoxLayout(self)
        v.setContentsMargins(24, 20, 24, 18)
        v.setSpacing(10)
        t = QLabel('Build something')
        t.setStyleSheet('font-size: 15pt; font-weight: 700;')
        v.addWidget(t)
        sub = QLabel('Start with one thing; drag it into place with the arrows, then add more from Create (left). Everything you '
                     'put in simulates together: fire burns the cloth, water puts the fire out, wind blows the smoke.')
        sub.setObjectName('hint')
        sub.setWordWrap(True)
        v.addWidget(sub)
        g = QGridLayout()
        g.setSpacing(10)
        for n, (key, glyph, label, col) in enumerate(STARTERS):
            b = BigTile(glyph, label, col)
            b.setToolTip(f'Put in {label.lower()} to start from')
            b.clicked.connect(lambda _=False, k=key: self.win.add_component(k))
            g.addWidget(b, n // 3, n % 3)
        v.addLayout(g)
        h = QVBoxLayout()
        h.setSpacing(2)
        for text, fn, tip in (('Ready-made effects', win.focus_effects, 'Start from one of the effects that come with Blackbody'),
                              ('Open a project…', win.open_dialog, 'Open a scene you saved'),
                              ('Put an effect in your footage…', win.import_footage, 'Import a clip; the Shot workspace places '
                               'effects in it')):
            b = QPushButton(text)
            b.setObjectName('ghost')
            b.setToolTip(tip)
            b.clicked.connect(fn)
            b.setStyleSheet(f'QPushButton {{ text-align: left; color: {theme.ACCENT_HI}; }}')
            h.addWidget(b, 0, Qt.AlignLeft)
        v.addLayout(h)
        self.setFixedWidth(400)
        self.adjustSize()
