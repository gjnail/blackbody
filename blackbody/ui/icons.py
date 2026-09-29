"""Small vector icons drawn with QPainter, so the app needs no image assets for its chrome."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

from . import theme


def _icon(draw, size=20, color=theme.TEXT):
    icon = QIcon()
    for mode, col in ((QIcon.Normal, color), (QIcon.Disabled, '#5d5d63'), (QIcon.Active, '#ffffff')):
        pm = QPixmap(size * 2, size * 2)
        pm.setDevicePixelRatio(2.0)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        c = QColor(col)
        p.setPen(QPen(c, 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.setBrush(c)
        draw(p, size)
        p.end()
        icon.addPixmap(pm, mode)
    return icon


def _tri(p, x, y, w, h, left=False):
    if left:
        pts = [QPointF(x + w, y), QPointF(x, y + h / 2), QPointF(x + w, y + h)]
    else:
        pts = [QPointF(x, y), QPointF(x + w, y + h / 2), QPointF(x, y + h)]
    p.drawPolygon(QPolygonF(pts))


def play():
    return _icon(lambda p, s: _tri(p, s * 0.32, s * 0.22, s * 0.46, s * 0.56))


def pause():
    def d(p, s):
        p.drawRect(QRectF(s * 0.28, s * 0.22, s * 0.14, s * 0.56))
        p.drawRect(QRectF(s * 0.58, s * 0.22, s * 0.14, s * 0.56))
    return _icon(d)


def step(forward=True):
    def d(p, s):
        if forward:
            _tri(p, s * 0.26, s * 0.26, s * 0.36, s * 0.48)
            p.drawRect(QRectF(s * 0.66, s * 0.26, s * 0.08, s * 0.48))
        else:
            _tri(p, s * 0.38, s * 0.26, s * 0.36, s * 0.48, left=True)
            p.drawRect(QRectF(s * 0.26, s * 0.26, s * 0.08, s * 0.48))
    return _icon(d)


def jump(forward=True):
    def d(p, s):
        if forward:
            _tri(p, s * 0.2, s * 0.28, s * 0.26, s * 0.44)
            _tri(p, s * 0.44, s * 0.28, s * 0.26, s * 0.44)
            p.drawRect(QRectF(s * 0.72, s * 0.28, s * 0.07, s * 0.44))
        else:
            _tri(p, s * 0.54, s * 0.28, s * 0.26, s * 0.44, left=True)
            _tri(p, s * 0.30, s * 0.28, s * 0.26, s * 0.44, left=True)
            p.drawRect(QRectF(s * 0.21, s * 0.28, s * 0.07, s * 0.44))
    return _icon(d)


def loop():
    def d(p, s):
        p.setBrush(Qt.NoBrush)
        path = QPainterPath()
        path.addRoundedRect(QRectF(s * 0.2, s * 0.3, s * 0.6, s * 0.4), s * 0.18, s * 0.18)
        p.drawPath(path)
        p.setBrush(p.pen().color())
        _tri(p, s * 0.52, s * 0.22, s * 0.12, s * 0.16)
    return _icon(d)


def restart():
    def d(p, s):
        p.setBrush(Qt.NoBrush)
        p.drawArc(QRectF(s * 0.22, s * 0.22, s * 0.56, s * 0.56), 60 * 16, 290 * 16)
        p.setBrush(p.pen().color())
        _tri(p, s * 0.60, s * 0.14, s * 0.16, s * 0.18)
    return _icon(d)


def cache():
    def d(p, s):
        for i in range(3):
            p.drawRect(QRectF(s * (0.2 + i * 0.22), s * 0.36, s * 0.16, s * 0.28))
    return _icon(d, color=theme.GOOD)


def flame():
    def d(p, s):
        path = QPainterPath()
        path.moveTo(s * 0.5, s * 0.12)
        path.cubicTo(s * 0.72, s * 0.38, s * 0.86, s * 0.56, s * 0.72, s * 0.78)
        path.cubicTo(s * 0.62, s * 0.92, s * 0.38, s * 0.92, s * 0.28, s * 0.78)
        path.cubicTo(s * 0.16, s * 0.6, s * 0.3, s * 0.42, s * 0.42, s * 0.34)
        path.cubicTo(s * 0.42, s * 0.46, s * 0.48, s * 0.52, s * 0.54, s * 0.5)
        path.cubicTo(s * 0.6, s * 0.36, s * 0.56, s * 0.24, s * 0.5, s * 0.12)
        p.setPen(Qt.NoPen)
        p.drawPath(path)
    return _icon(d, color=theme.ACCENT)


def app_icon():
    icon = QIcon()
    for size in (16, 32, 48, 64, 128, 256):
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor('#141416'))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(QRectF(0, 0, size, size), size * 0.2, size * 0.2)
        s = size
        path = QPainterPath()
        path.moveTo(s * 0.5, s * 0.12)
        path.cubicTo(s * 0.74, s * 0.38, s * 0.88, s * 0.58, s * 0.74, s * 0.8)
        path.cubicTo(s * 0.63, s * 0.94, s * 0.37, s * 0.94, s * 0.26, s * 0.8)
        path.cubicTo(s * 0.14, s * 0.6, s * 0.3, s * 0.42, s * 0.42, s * 0.32)
        path.cubicTo(s * 0.42, s * 0.46, s * 0.48, s * 0.54, s * 0.55, s * 0.52)
        path.cubicTo(s * 0.62, s * 0.36, s * 0.57, s * 0.22, s * 0.5, s * 0.12)
        from PySide6.QtGui import QLinearGradient
        g = QLinearGradient(0, s * 0.9, 0, s * 0.12)
        g.setColorAt(0.0, QColor('#fff1c9'))
        g.setColorAt(0.35, QColor('#ffb347'))
        g.setColorAt(0.75, QColor('#ff6a1a'))
        g.setColorAt(1.0, QColor('#b8320a'))
        p.setBrush(g)
        p.drawPath(path)
        p.end()
        icon.addPixmap(pm)
    return icon
