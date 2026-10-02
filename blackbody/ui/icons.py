"""Small vector icons drawn with QPainter, so the app needs no image assets for its chrome."""
from __future__ import annotations

import math

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


# -- line glyphs ---------------------------------------------------------------------------------------
# Each draws in an s×s box with the painter's pen; glyph_icon() and paint_glyph() set the colour.

def _flame_path(s, x0=0.0, y0=0.0, k=1.0):
    def P(x, y):
        return QPointF(x0 + x * s * k, y0 + y * s * k)
    path = QPainterPath()
    path.moveTo(P(0.5, 0.1))
    path.cubicTo(P(0.74, 0.36), P(0.86, 0.56), P(0.74, 0.76))
    path.cubicTo(P(0.64, 0.92), P(0.36, 0.92), P(0.26, 0.76))
    path.cubicTo(P(0.14, 0.58), P(0.3, 0.42), P(0.42, 0.32))
    path.cubicTo(P(0.42, 0.46), P(0.48, 0.52), P(0.55, 0.5))
    path.cubicTo(P(0.6, 0.36), P(0.56, 0.22), P(0.5, 0.1))
    return path


def _g_star(p, s):
    pts = []
    for i in range(10):
        r = s * (0.4 if i % 2 == 0 else 0.17)
        a = -math.pi / 2 + i * math.pi / 5
        pts.append(QPointF(s * 0.5 + r * math.cos(a), s * 0.54 + r * math.sin(a)))
    p.drawPolygon(QPolygonF(pts))


def _g_cube(p, s):
    top = [QPointF(s * 0.5, s * 0.14), QPointF(s * 0.84, s * 0.32), QPointF(s * 0.5, s * 0.5), QPointF(s * 0.16, s * 0.32)]
    p.drawPolygon(QPolygonF(top))
    p.drawLine(QPointF(s * 0.16, s * 0.32), QPointF(s * 0.16, s * 0.7))
    p.drawLine(QPointF(s * 0.84, s * 0.32), QPointF(s * 0.84, s * 0.7))
    p.drawLine(QPointF(s * 0.5, s * 0.5), QPointF(s * 0.5, s * 0.88))
    p.drawLine(QPointF(s * 0.16, s * 0.7), QPointF(s * 0.5, s * 0.88))
    p.drawLine(QPointF(s * 0.84, s * 0.7), QPointF(s * 0.5, s * 0.88))


def _g_flame(p, s):
    p.drawPath(_flame_path(s))


def _g_wind(p, s):
    for y, l, curl in ((0.3, 0.62, True), (0.5, 0.78, False), (0.7, 0.52, True)):
        path = QPainterPath()
        path.moveTo(s * 0.12, s * y)
        path.lineTo(s * l, s * y)
        if curl:
            path.cubicTo(s * (l + 0.16), s * y, s * (l + 0.16), s * (y - 0.16), s * (l + 0.02), s * (y - 0.14))
        p.drawPath(path)


def _g_palette(p, s):
    path = QPainterPath()
    path.moveTo(s * 0.5, s * 0.14)
    path.cubicTo(s * 0.86, s * 0.14, s * 0.92, s * 0.5, s * 0.78, s * 0.6)
    path.cubicTo(s * 0.68, s * 0.68, s * 0.6, s * 0.6, s * 0.56, s * 0.72)
    path.cubicTo(s * 0.52, s * 0.86, s * 0.44, s * 0.88, s * 0.36, s * 0.84)
    path.cubicTo(s * 0.14, s * 0.74, s * 0.1, s * 0.46, s * 0.22, s * 0.3)
    path.cubicTo(s * 0.3, s * 0.2, s * 0.4, s * 0.14, s * 0.5, s * 0.14)
    p.drawPath(path)
    for x, y in ((0.34, 0.36), (0.52, 0.28), (0.68, 0.38), (0.3, 0.56)):
        p.drawEllipse(QPointF(s * x, s * y), s * 0.035, s * 0.035)


def _g_bulb(p, s):
    path = QPainterPath()
    path.moveTo(s * 0.38, s * 0.68)
    path.cubicTo(s * 0.38, s * 0.56, s * 0.22, s * 0.5, s * 0.22, s * 0.36)
    path.cubicTo(s * 0.22, s * 0.2, s * 0.36, s * 0.1, s * 0.5, s * 0.1)
    path.cubicTo(s * 0.64, s * 0.1, s * 0.78, s * 0.2, s * 0.78, s * 0.36)
    path.cubicTo(s * 0.78, s * 0.5, s * 0.62, s * 0.56, s * 0.62, s * 0.68)
    path.closeSubpath()
    p.drawPath(path)
    p.drawLine(QPointF(s * 0.4, s * 0.79), QPointF(s * 0.6, s * 0.79))
    p.drawLine(QPointF(s * 0.44, s * 0.89), QPointF(s * 0.56, s * 0.89))


def _sparkle(p, c, r):
    path = QPainterPath()
    path.moveTo(c.x(), c.y() - r)
    path.quadTo(c.x(), c.y(), c.x() + r, c.y())
    path.quadTo(c.x(), c.y(), c.x(), c.y() + r)
    path.quadTo(c.x(), c.y(), c.x() - r, c.y())
    path.quadTo(c.x(), c.y(), c.x(), c.y() - r)
    p.drawPath(path)


def _g_sparks(p, s):
    _sparkle(p, QPointF(s * 0.42, s * 0.56), s * 0.3)
    _sparkle(p, QPointF(s * 0.76, s * 0.24), s * 0.13)
    p.drawEllipse(QPointF(s * 0.8, s * 0.72), s * 0.04, s * 0.04)


def _g_spread(p, s):
    p.drawPath(_flame_path(s, s * 0.02, s * 0.26, 0.66))
    p.drawPath(_flame_path(s, s * 0.4, s * 0.08, 0.6))
    p.drawLine(QPointF(s * 0.08, s * 0.9), QPointF(s * 0.92, s * 0.9))


def _g_box(p, s):
    a, b, o = 0.14, 0.66, 0.2
    p.drawRect(QRectF(s * a, s * (a + o), s * (b - a), s * (b - a)))
    pen = p.pen()
    dash = QPen(pen)
    dash.setStyle(Qt.DotLine)
    p.setPen(dash)
    p.drawRect(QRectF(s * (a + o), s * a, s * (b - a), s * (b - a)))
    p.setPen(pen)
    for x, y in ((a, a + o), (b, a + o), (b, b + o), (a, b + o)):
        p.drawLine(QPointF(s * x, s * y), QPointF(s * (x + o), s * (y - o)))


def _g_camera(p, s):
    p.drawRoundedRect(QRectF(s * 0.1, s * 0.3, s * 0.58, s * 0.44), s * 0.06, s * 0.06)
    p.drawPolygon(QPolygonF([QPointF(s * 0.68, s * 0.46), QPointF(s * 0.9, s * 0.34), QPointF(s * 0.9, s * 0.7),
                             QPointF(s * 0.68, s * 0.58)]))


def _g_layers(p, s):
    for dy in (0.0, 0.18, 0.36):
        p.drawPolygon(QPolygonF([QPointF(s * 0.5, s * (0.16 + dy)), QPointF(s * 0.88, s * (0.32 + dy)),
                                 QPointF(s * 0.5, s * (0.48 + dy)), QPointF(s * 0.12, s * (0.32 + dy))]))


def _g_film(p, s):
    p.drawRoundedRect(QRectF(s * 0.14, s * 0.18, s * 0.72, s * 0.64), s * 0.06, s * 0.06)
    p.drawLine(QPointF(s * 0.3, s * 0.18), QPointF(s * 0.3, s * 0.82))
    p.drawLine(QPointF(s * 0.7, s * 0.18), QPointF(s * 0.7, s * 0.82))
    for y in (0.32, 0.5, 0.68):
        p.drawLine(QPointF(s * 0.16, s * y), QPointF(s * 0.28, s * y))
        p.drawLine(QPointF(s * 0.72, s * y), QPointF(s * 0.84, s * y))


def _drop_path(s, cx=0.5, top=0.1, w=0.3, bottom=0.88):
    path = QPainterPath()
    path.moveTo(s * cx, s * top)
    path.cubicTo(s * (cx + w * 0.5), s * (top + 0.25), s * (cx + w), s * (bottom - 0.3), s * (cx + w), s * (bottom - 0.2))
    path.cubicTo(s * (cx + w), s * (bottom + 0.02), s * (cx - w), s * (bottom + 0.02), s * (cx - w), s * (bottom - 0.2))
    path.cubicTo(s * (cx - w), s * (bottom - 0.3), s * (cx - w * 0.5), s * (top + 0.25), s * cx, s * top)
    return path


def _g_drop(p, s):
    p.drawPath(_drop_path(s))


def _g_waves(p, s):
    for y in (0.38, 0.62):
        path = QPainterPath()
        path.moveTo(s * 0.1, s * y)
        for i in range(4):
            x = 0.1 + i * 0.2
            path.quadTo(s * (x + 0.05), s * (y - 0.1), s * (x + 0.1), s * y)
            path.quadTo(s * (x + 0.15), s * (y + 0.1), s * (x + 0.2), s * y)
        p.drawPath(path)


def _g_snow(p, s):
    c = QPointF(s * 0.5, s * 0.5)
    for k in range(3):
        a = math.pi / 2 + k * math.pi / 3
        d = QPointF(math.cos(a) * s * 0.38, math.sin(a) * s * 0.38)
        p.drawLine(c - d, c + d)
        for sign in (1, -1):
            e = c + d * 0.62 * sign
            for t in (a + 0.7, a - 0.7):
                q = QPointF(math.cos(t) * s * 0.12 * sign, math.sin(t) * s * 0.12 * sign)
                p.drawLine(e, e + q)


def _cloud_path(s, y=0.0):
    path = QPainterPath()
    path.moveTo(s * 0.26, s * (0.72 + y))
    path.cubicTo(s * 0.08, s * (0.72 + y), s * 0.08, s * (0.48 + y), s * 0.26, s * (0.48 + y))
    path.cubicTo(s * 0.28, s * (0.26 + y), s * 0.58, s * (0.22 + y), s * 0.64, s * (0.42 + y))
    path.cubicTo(s * 0.9, s * (0.38 + y), s * 0.94, s * (0.72 + y), s * 0.72, s * (0.72 + y))
    path.closeSubpath()
    return path


def _g_cloud(p, s):
    p.drawPath(_cloud_path(s))


def _g_weather(p, s):
    p.drawPath(_cloud_path(s, -0.14))
    for x in (0.34, 0.52, 0.7):
        p.drawLine(QPointF(s * x, s * 0.72), QPointF(s * (x - 0.06), s * 0.88))


def _g_sky(p, s):
    p.drawArc(QRectF(s * 0.28, s * 0.38, s * 0.44, s * 0.44), 0, 180 * 16)
    p.drawLine(QPointF(s * 0.08, s * 0.6), QPointF(s * 0.92, s * 0.6))
    for k in range(5):
        a = math.pi * (k + 0.5) / 5
        p.drawLine(QPointF(s * (0.5 + 0.3 * math.cos(a)), s * (0.6 - 0.3 * math.sin(a))),
                   QPointF(s * (0.5 + 0.4 * math.cos(a)), s * (0.6 - 0.4 * math.sin(a))))
    p.drawLine(QPointF(s * 0.22, s * 0.76), QPointF(s * 0.78, s * 0.76))


def _g_lava(p, s):
    p.drawPolygon(QPolygonF([QPointF(s * 0.08, s * 0.86), QPointF(s * 0.38, s * 0.36), QPointF(s * 0.62, s * 0.36),
                             QPointF(s * 0.92, s * 0.86)]))
    path = QPainterPath()
    path.moveTo(s * 0.44, s * 0.36)
    path.cubicTo(s * 0.44, s * 0.5, s * 0.34, s * 0.56, s * 0.36, s * 0.66)
    p.drawPath(path)
    p.drawEllipse(QPointF(s * 0.5, s * 0.18), s * 0.05, s * 0.05)
    p.drawEllipse(QPointF(s * 0.66, s * 0.1), s * 0.03, s * 0.03)


def _g_search(p, s):
    p.drawEllipse(QPointF(s * 0.43, s * 0.43), s * 0.25, s * 0.25)
    p.drawLine(QPointF(s * 0.62, s * 0.62), QPointF(s * 0.86, s * 0.86))


def _g_footage(p, s):
    p.drawRoundedRect(QRectF(s * 0.1, s * 0.22, s * 0.8, s * 0.56), s * 0.07, s * 0.07)
    p.drawPolyline(QPolygonF([QPointF(s * 0.16, s * 0.7), QPointF(s * 0.38, s * 0.46), QPointF(s * 0.54, s * 0.6),
                              QPointF(s * 0.64, s * 0.52), QPointF(s * 0.84, s * 0.7)]))
    p.drawEllipse(QPointF(s * 0.68, s * 0.36), s * 0.05, s * 0.05)


def _g_open(p, s):
    path = QPainterPath()
    path.moveTo(s * 0.1, s * 0.78)
    path.lineTo(s * 0.1, s * 0.24)
    path.lineTo(s * 0.38, s * 0.24)
    path.lineTo(s * 0.46, s * 0.33)
    path.lineTo(s * 0.82, s * 0.33)
    path.lineTo(s * 0.82, s * 0.44)
    p.drawPath(path)
    p.drawPolygon(QPolygonF([QPointF(s * 0.1, s * 0.78), QPointF(s * 0.22, s * 0.44), QPointF(s * 0.94, s * 0.44),
                             QPointF(s * 0.8, s * 0.78)]))


def _g_save(p, s):
    p.drawRoundedRect(QRectF(s * 0.16, s * 0.16, s * 0.68, s * 0.68), s * 0.06, s * 0.06)
    p.drawRect(QRectF(s * 0.3, s * 0.16, s * 0.36, s * 0.2))
    p.drawRect(QRectF(s * 0.28, s * 0.54, s * 0.44, s * 0.3))


def _g_undo(p, s, flip=False):
    path = QPainterPath()
    if not flip:
        path.moveTo(s * 0.24, s * 0.42)
        path.lineTo(s * 0.62, s * 0.42)
        path.cubicTo(s * 0.9, s * 0.42, s * 0.9, s * 0.8, s * 0.62, s * 0.8)
        path.lineTo(s * 0.42, s * 0.8)
        p.drawPath(path)
        p.drawPolyline(QPolygonF([QPointF(s * 0.38, s * 0.26), QPointF(s * 0.22, s * 0.42), QPointF(s * 0.38, s * 0.58)]))
    else:
        path.moveTo(s * 0.76, s * 0.42)
        path.lineTo(s * 0.38, s * 0.42)
        path.cubicTo(s * 0.1, s * 0.42, s * 0.1, s * 0.8, s * 0.38, s * 0.8)
        path.lineTo(s * 0.58, s * 0.8)
        p.drawPath(path)
        p.drawPolyline(QPolygonF([QPointF(s * 0.62, s * 0.26), QPointF(s * 0.78, s * 0.42), QPointF(s * 0.62, s * 0.58)]))


def _g_plus(p, s):
    p.drawLine(QPointF(s * 0.5, s * 0.2), QPointF(s * 0.5, s * 0.8))
    p.drawLine(QPointF(s * 0.2, s * 0.5), QPointF(s * 0.8, s * 0.5))


def _g_trash(p, s):
    p.drawLine(QPointF(s * 0.18, s * 0.28), QPointF(s * 0.82, s * 0.28))
    p.drawPolyline(QPolygonF([QPointF(s * 0.4, s * 0.28), QPointF(s * 0.42, s * 0.16), QPointF(s * 0.58, s * 0.16),
                              QPointF(s * 0.6, s * 0.28)]))
    p.drawPolyline(QPolygonF([QPointF(s * 0.26, s * 0.28), QPointF(s * 0.3, s * 0.86), QPointF(s * 0.7, s * 0.86),
                              QPointF(s * 0.74, s * 0.28)]))


def _g_copy(p, s):
    p.drawRoundedRect(QRectF(s * 0.32, s * 0.32, s * 0.5, s * 0.52), s * 0.06, s * 0.06)
    p.drawPolyline(QPolygonF([QPointF(s * 0.2, s * 0.68), QPointF(s * 0.18, s * 0.68), QPointF(s * 0.18, s * 0.16),
                              QPointF(s * 0.66, s * 0.16), QPointF(s * 0.66, s * 0.2)]))


def _g_reset(p, s):
    p.drawArc(QRectF(s * 0.2, s * 0.2, s * 0.6, s * 0.6), 100 * 16, 280 * 16)
    p.drawPolyline(QPolygonF([QPointF(s * 0.26, s * 0.1), QPointF(s * 0.42, s * 0.22), QPointF(s * 0.3, s * 0.36)]))


def _g_grid(p, s):
    p.drawRect(QRectF(s * 0.16, s * 0.16, s * 0.68, s * 0.68))
    for t in (0.39, 0.61):
        p.drawLine(QPointF(s * t, s * 0.16), QPointF(s * t, s * 0.84))
        p.drawLine(QPointF(s * 0.16, s * t), QPointF(s * 0.84, s * t))


def _g_stats(p, s):
    for x, h in ((0.24, 0.3), (0.44, 0.56), (0.64, 0.42)):
        p.drawRect(QRectF(s * x, s * (0.82 - h), s * 0.12, s * h))
    p.drawLine(QPointF(s * 0.14, s * 0.84), QPointF(s * 0.86, s * 0.84))


def _g_fit(p, s):
    for x, y, dx, dy in ((0.16, 0.16, 1, 1), (0.84, 0.16, -1, 1), (0.16, 0.84, 1, -1), (0.84, 0.84, -1, -1)):
        p.drawPolyline(QPolygonF([QPointF(s * x, s * (y + dy * 0.2)), QPointF(s * x, s * y), QPointF(s * (x + dx * 0.2), s * y)]))


def _g_emitter(p, s):
    p.drawPath(_flame_path(s, s * 0.08, s * 0.04, 0.84))


def _g_fabric(p, s):
    path = QPainterPath()
    path.moveTo(s * 0.16, s * 0.16)
    path.lineTo(s * 0.84, s * 0.16)
    path.cubicTo(s * 0.76, s * 0.4, s * 0.9, s * 0.6, s * 0.8, s * 0.86)
    path.cubicTo(s * 0.6, s * 0.78, s * 0.4, s * 0.92, s * 0.2, s * 0.84)
    path.cubicTo(s * 0.26, s * 0.6, s * 0.1, s * 0.4, s * 0.16, s * 0.16)
    p.drawPath(path)


def _g_logs(p, s):
    p.drawPath(_flame_path(s, s * 0.22, s * 0.0, 0.56))
    p.drawLine(QPointF(s * 0.12, s * 0.9), QPointF(s * 0.84, s * 0.66))
    p.drawLine(QPointF(s * 0.16, s * 0.66), QPointF(s * 0.88, s * 0.9))


def _g_pool(p, s):
    p.drawEllipse(QRectF(s * 0.08, s * 0.7, s * 0.84, s * 0.2))
    for x, k in ((0.14, 0.42), (0.36, 0.55), (0.58, 0.42)):
        p.drawPath(_flame_path(s, s * x, s * (0.82 - k), k * 0.9))


def _g_line(p, s):
    p.drawLine(QPointF(s * 0.06, s * 0.86), QPointF(s * 0.94, s * 0.86))
    for x in (0.06, 0.34, 0.62):
        p.drawPath(_flame_path(s, s * x, s * 0.42, 0.42))


def _g_torch(p, s):
    p.drawPath(_flame_path(s, s * 0.26, s * 0.0, 0.48))
    p.drawLine(QPointF(s * 0.5, s * 0.5), QPointF(s * 0.5, s * 0.92))
    p.drawLine(QPointF(s * 0.4, s * 0.5), QPointF(s * 0.6, s * 0.5))


def _g_ring(p, s):
    p.drawEllipse(QRectF(s * 0.12, s * 0.62, s * 0.76, s * 0.24))
    for x in (0.18, 0.42, 0.66):
        p.drawLine(QPointF(s * (x + 0.08), s * 0.62), QPointF(s * (x + 0.08), s * 0.36))
        p.drawEllipse(QPointF(s * (x + 0.08), s * 0.3), s * 0.04, s * 0.06)


def _g_burst(p, s):
    for k in range(10):
        a = k * math.pi / 5
        r0, r1 = (0.18, 0.44) if k % 2 == 0 else (0.22, 0.32)
        p.drawLine(QPointF(s * (0.5 + r0 * math.cos(a)), s * (0.5 + r0 * math.sin(a))),
                   QPointF(s * (0.5 + r1 * math.cos(a)), s * (0.5 + r1 * math.sin(a))))
    p.drawEllipse(QPointF(s * 0.5, s * 0.5), s * 0.1, s * 0.1)


def _g_jet(p, s):
    p.drawRect(QRectF(s * 0.06, s * 0.56, s * 0.2, s * 0.14))
    path = QPainterPath()
    path.moveTo(s * 0.28, s * 0.63)
    path.cubicTo(s * 0.55, s * 0.6, s * 0.7, s * 0.52, s * 0.82, s * 0.3)
    path.moveTo(s * 0.28, s * 0.63)
    path.cubicTo(s * 0.55, s * 0.72, s * 0.8, s * 0.62, s * 0.94, s * 0.42)
    p.drawPath(path)


def _g_spiral(p, s):
    path = QPainterPath()
    c = QPointF(s * 0.5, s * 0.5)
    for i in range(80):
        t = i / 79 * 4.2 * math.pi
        r = s * (0.04 + 0.36 * i / 79)
        q = QPointF(c.x() + r * math.cos(t), c.y() + r * math.sin(t) * 0.8)
        path.moveTo(q) if i == 0 else path.lineTo(q)
    p.drawPath(path)


def _g_steam(p, s):
    for x in (0.3, 0.5, 0.7):
        path = QPainterPath()
        path.moveTo(s * x, s * 0.88)
        path.cubicTo(s * (x - 0.12), s * 0.7, s * (x + 0.12), s * 0.5, s * x, s * 0.32)
        path.cubicTo(s * (x - 0.08), s * 0.22, s * (x + 0.04), s * 0.16, s * x, s * 0.1)
        p.drawPath(path)


def _g_pour(p, s):
    p.drawPolygon(QPolygonF([QPointF(s * 0.08, s * 0.12), QPointF(s * 0.4, s * 0.12), QPointF(s * 0.46, s * 0.26),
                             QPointF(s * 0.36, s * 0.44), QPointF(s * 0.12, s * 0.44)]))
    path = QPainterPath()
    path.moveTo(s * 0.46, s * 0.26)
    path.cubicTo(s * 0.62, s * 0.3, s * 0.66, s * 0.5, s * 0.66, s * 0.8)
    p.drawPath(path)
    p.drawEllipse(QRectF(s * 0.48, s * 0.8, s * 0.36, s * 0.1))


def _g_fountain(p, s):
    path = QPainterPath()
    path.moveTo(s * 0.5, s * 0.86)
    path.lineTo(s * 0.5, s * 0.16)
    path.moveTo(s * 0.5, s * 0.2)
    path.cubicTo(s * 0.3, s * 0.12, s * 0.2, s * 0.4, s * 0.18, s * 0.7)
    path.moveTo(s * 0.5, s * 0.2)
    path.cubicTo(s * 0.7, s * 0.12, s * 0.8, s * 0.4, s * 0.82, s * 0.7)
    p.drawPath(path)
    p.drawLine(QPointF(s * 0.1, s * 0.88), QPointF(s * 0.9, s * 0.88))


def _g_ball(p, s):
    p.drawEllipse(QPointF(s * 0.5, s * 0.5), s * 0.34, s * 0.34)
    p.drawArc(QRectF(s * 0.16, s * 0.38, s * 0.68, s * 0.24), 180 * 16, 180 * 16)


def _g_pillar(p, s):
    p.drawEllipse(QRectF(s * 0.3, s * 0.1, s * 0.4, s * 0.12))
    p.drawLine(QPointF(s * 0.3, s * 0.16), QPointF(s * 0.3, s * 0.84))
    p.drawLine(QPointF(s * 0.7, s * 0.16), QPointF(s * 0.7, s * 0.84))
    p.drawArc(QRectF(s * 0.3, s * 0.78, s * 0.4, s * 0.12), 180 * 16, 180 * 16)


def _g_wall(p, s):
    p.drawRect(QRectF(s * 0.1, s * 0.2, s * 0.8, s * 0.64))
    for y in (0.41, 0.62):
        p.drawLine(QPointF(s * 0.1, s * y), QPointF(s * 0.9, s * y))
    for x, y0, y1 in ((0.5, 0.2, 0.41), (0.3, 0.41, 0.62), (0.7, 0.41, 0.62), (0.5, 0.62, 0.84)):
        p.drawLine(QPointF(s * x, s * y0), QPointF(s * x, s * y1))


def _g_house(p, s):
    p.drawPolyline(QPolygonF([QPointF(s * 0.12, s * 0.46), QPointF(s * 0.5, s * 0.14), QPointF(s * 0.88, s * 0.46)]))
    p.drawPolyline(QPolygonF([QPointF(s * 0.2, s * 0.4), QPointF(s * 0.2, s * 0.86), QPointF(s * 0.8, s * 0.86),
                              QPointF(s * 0.8, s * 0.4)]))
    p.drawRect(QRectF(s * 0.42, s * 0.6, s * 0.16, s * 0.26))


def _g_car(p, s):
    path = QPainterPath()
    path.moveTo(s * 0.06, s * 0.7)
    path.lineTo(s * 0.06, s * 0.52)
    path.lineTo(s * 0.26, s * 0.48)
    path.lineTo(s * 0.38, s * 0.32)
    path.lineTo(s * 0.68, s * 0.32)
    path.lineTo(s * 0.8, s * 0.48)
    path.lineTo(s * 0.94, s * 0.52)
    path.lineTo(s * 0.94, s * 0.7)
    path.closeSubpath()
    p.drawPath(path)
    p.drawEllipse(QPointF(s * 0.26, s * 0.72), s * 0.08, s * 0.08)
    p.drawEllipse(QPointF(s * 0.74, s * 0.72), s * 0.08, s * 0.08)


def _g_hill(p, s):
    path = QPainterPath()
    path.moveTo(s * 0.04, s * 0.84)
    path.cubicTo(s * 0.3, s * 0.84, s * 0.32, s * 0.24, s * 0.56, s * 0.24)
    path.cubicTo(s * 0.76, s * 0.24, s * 0.8, s * 0.6, s * 0.96, s * 0.6)
    p.drawPath(path)
    p.drawLine(QPointF(s * 0.04, s * 0.86), QPointF(s * 0.96, s * 0.86))


def _grain(p, s, x, y, r=0.045):
    p.save()
    p.setBrush(p.pen().color())
    p.drawEllipse(QPointF(s * x, s * y), s * r, s * r)
    p.restore()


def _g_grains(p, s):
    """A heap of sand, grainy."""
    path = QPainterPath()
    path.moveTo(s * 0.04, s * 0.86)
    path.cubicTo(s * 0.26, s * 0.84, s * 0.36, s * 0.3, s * 0.5, s * 0.3)
    path.cubicTo(s * 0.64, s * 0.3, s * 0.74, s * 0.84, s * 0.96, s * 0.86)
    path.closeSubpath()
    p.drawPath(path)
    for x, y in ((0.36, 0.74), (0.5, 0.62), (0.64, 0.74), (0.5, 0.46)):
        _grain(p, s, x, y, 0.05)


def _g_mud(p, s):
    """A heap of mud slumping out flat, with a lump dropping onto it."""
    path = QPainterPath()
    path.moveTo(s * 0.04, s * 0.84)
    path.cubicTo(s * 0.16, s * 0.84, s * 0.2, s * 0.66, s * 0.34, s * 0.62)
    path.cubicTo(s * 0.4, s * 0.5, s * 0.62, s * 0.48, s * 0.66, s * 0.62)
    path.cubicTo(s * 0.8, s * 0.64, s * 0.84, s * 0.84, s * 0.96, s * 0.84)
    path.closeSubpath()
    p.drawPath(path)
    drop = QPainterPath()
    drop.moveTo(s * 0.5, s * 0.1)
    drop.cubicTo(s * 0.6, s * 0.24, s * 0.58, s * 0.34, s * 0.5, s * 0.34)
    drop.cubicTo(s * 0.42, s * 0.34, s * 0.4, s * 0.24, s * 0.5, s * 0.1)
    p.drawPath(drop)


def _g_jelly(p, s):
    """A wobbling block of jelly with a shine on it."""
    path = QPainterPath()
    path.moveTo(s * 0.16, s * 0.86)
    path.cubicTo(s * 0.1, s * 0.62, s * 0.22, s * 0.42, s * 0.18, s * 0.24)
    path.cubicTo(s * 0.4, s * 0.14, s * 0.6, s * 0.3, s * 0.82, s * 0.2)
    path.cubicTo(s * 0.78, s * 0.44, s * 0.9, s * 0.64, s * 0.84, s * 0.86)
    path.closeSubpath()
    p.drawPath(path)
    shine = QPainterPath()
    shine.moveTo(s * 0.32, s * 0.36)
    shine.cubicTo(s * 0.29, s * 0.46, s * 0.33, s * 0.54, s * 0.3, s * 0.64)
    p.drawPath(shine)


def _g_snowball(p, s):
    """A snowball flying, with lumps coming off it."""
    p.drawEllipse(QPointF(s * 0.56, s * 0.46), s * 0.26, s * 0.26)
    for a, b in (((0.06, 0.36), (0.22, 0.36)), ((0.1, 0.52), (0.24, 0.52)), ((0.04, 0.68), (0.2, 0.62))):
        p.drawLine(QPointF(s * a[0], s * a[1]), QPointF(s * b[0], s * b[1]))
    _grain(p, s, 0.86, 0.82, 0.05)
    _grain(p, s, 0.72, 0.88, 0.04)


def _g_clay(p, s):
    """A lump of clay squashed flat where it landed."""
    path = QPainterPath()
    path.moveTo(s * 0.1, s * 0.84)
    path.lineTo(s * 0.9, s * 0.84)
    path.cubicTo(s * 0.92, s * 0.6, s * 0.76, s * 0.4, s * 0.52, s * 0.42)
    path.cubicTo(s * 0.3, s * 0.38, s * 0.1, s * 0.56, s * 0.1, s * 0.84)
    p.drawPath(path)
    p.drawLine(QPointF(s * 0.34, s * 0.62), QPointF(s * 0.5, s * 0.58))


def _g_cart(p, s):
    """A cart on two wheels."""
    p.drawRoundedRect(QRectF(s * 0.1, s * 0.38, s * 0.8, s * 0.28), s * 0.04, s * 0.04)
    p.drawEllipse(QPointF(s * 0.28, s * 0.74), s * 0.12, s * 0.12)
    p.drawEllipse(QPointF(s * 0.72, s * 0.74), s * 0.12, s * 0.12)
    p.drawLine(QPointF(s * 0.18, s * 0.38), QPointF(s * 0.28, s * 0.22))


def _g_turntable(p, s):
    """A disc seen from the side, spinning, with a block riding on it."""
    p.drawEllipse(QRectF(s * 0.08, s * 0.52, s * 0.84, s * 0.26))
    p.drawLine(QPointF(s * 0.5, s * 0.78), QPointF(s * 0.5, s * 0.92))
    p.drawRect(QRectF(s * 0.56, s * 0.42, s * 0.16, s * 0.16))
    path = QPainterPath()
    path.moveTo(s * 0.16, s * 0.4)
    path.cubicTo(s * 0.22, s * 0.22, s * 0.42, s * 0.16, s * 0.56, s * 0.2)
    p.drawPath(path)
    p.drawLine(QPointF(s * 0.56, s * 0.2), QPointF(s * 0.47, s * 0.14))
    p.drawLine(QPointF(s * 0.56, s * 0.2), QPointF(s * 0.49, s * 0.27))


def _g_windmill(p, s):
    """A post with four sails crossed on its hub."""
    c = QPointF(s * 0.5, s * 0.38)
    p.drawLine(QPointF(s * 0.42, s * 0.92), c)
    p.drawLine(QPointF(s * 0.58, s * 0.92), c)
    for k in range(4):
        a = math.radians(20 + 90 * k)
        tip = c + QPointF(math.cos(a) * s * 0.34, -math.sin(a) * s * 0.34)
        side = QPointF(math.cos(a + 1.571) * s * 0.06, -math.sin(a + 1.571) * s * 0.06)
        p.drawPolygon(QPolygonF([c, tip, tip + side, c + side * 0.4]))


def _g_ramp(p, s):
    """A ramp with a ball at the top."""
    p.drawPolygon(QPolygonF([QPointF(s * 0.06, s * 0.86), QPointF(s * 0.94, s * 0.86), QPointF(s * 0.94, s * 0.46)]))
    p.drawEllipse(QPointF(s * 0.76, s * 0.4), s * 0.09, s * 0.09)


def _blade(p, s, x, h, lean):
    path = QPainterPath()
    path.moveTo(s * x, s * 0.9)
    path.quadTo(s * (x + lean * 0.2), s * (0.9 - h * 0.6), s * (x + lean), s * (0.9 - h))
    p.drawPath(path)


def _g_grass(p, s):
    """A tuft of grass blades bending in the wind."""
    for x, h, lean in ((0.2, 0.42, 0.06), (0.32, 0.62, 0.12), (0.45, 0.76, 0.16), (0.56, 0.58, 0.14), (0.68, 0.7, 0.18),
                       (0.8, 0.46, 0.1)):
        _blade(p, s, x, h, lean)
    p.drawLine(QPointF(s * 0.1, s * 0.9), QPointF(s * 0.9, s * 0.9))


def _g_wheat(p, s):
    """A stalk of wheat with its ear."""
    _blade(p, s, 0.48, 0.5, 0.06)
    for k in range(4):
        y = 0.34 - 0.08 * k
        p.drawEllipse(QPointF(s * 0.47, s * y), s * 0.05, s * 0.035)
        p.drawEllipse(QPointF(s * 0.61, s * (y + 0.03)), s * 0.05, s * 0.035)
    p.drawLine(QPointF(s * 0.54, s * 0.08), QPointF(s * 0.54, s * 0.38))
    _blade(p, s, 0.3, 0.3, -0.08)
    _blade(p, s, 0.7, 0.26, 0.08)


def _g_reeds(p, s):
    """Reeds, one with its cattail."""
    _blade(p, s, 0.3, 0.7, -0.06)
    _blade(p, s, 0.66, 0.66, 0.1)
    p.drawLine(QPointF(s * 0.48, s * 0.9), QPointF(s * 0.5, s * 0.1))
    p.drawRoundedRect(QRectF(s * 0.45, s * 0.2, s * 0.1, s * 0.26), s * 0.05, s * 0.05)


def _g_flag(p, s):
    p.drawLine(QPointF(s * 0.18, s * 0.08), QPointF(s * 0.18, s * 0.92))
    path = QPainterPath()
    path.moveTo(s * 0.18, s * 0.14)
    path.cubicTo(s * 0.4, s * 0.06, s * 0.6, s * 0.26, s * 0.86, s * 0.16)
    path.lineTo(s * 0.86, s * 0.48)
    path.cubicTo(s * 0.6, s * 0.58, s * 0.4, s * 0.38, s * 0.18, s * 0.46)
    p.drawPath(path)


def _g_spot(p, s):
    p.drawPolygon(QPolygonF([QPointF(s * 0.36, s * 0.1), QPointF(s * 0.64, s * 0.1), QPointF(s * 0.58, s * 0.3),
                             QPointF(s * 0.42, s * 0.3)]))
    pen = p.pen()
    dash = QPen(pen)
    dash.setStyle(Qt.DashLine)
    p.setPen(dash)
    p.drawLine(QPointF(s * 0.42, s * 0.32), QPointF(s * 0.16, s * 0.9))
    p.drawLine(QPointF(s * 0.58, s * 0.32), QPointF(s * 0.84, s * 0.9))
    p.setPen(pen)
    p.drawEllipse(QRectF(s * 0.16, s * 0.84, s * 0.68, s * 0.1))


def _g_window(p, s):
    p.drawRect(QRectF(s * 0.16, s * 0.12, s * 0.68, s * 0.68))
    p.drawLine(QPointF(s * 0.5, s * 0.12), QPointF(s * 0.5, s * 0.8))
    p.drawLine(QPointF(s * 0.16, s * 0.46), QPointF(s * 0.84, s * 0.46))
    p.drawLine(QPointF(s * 0.08, s * 0.9), QPointF(s * 0.92, s * 0.9))


def _g_mesh(p, s):
    pts = [QPointF(s * 0.5, s * 0.1), QPointF(s * 0.88, s * 0.34), QPointF(s * 0.78, s * 0.82), QPointF(s * 0.22, s * 0.82),
           QPointF(s * 0.12, s * 0.34)]
    p.drawPolygon(QPolygonF(pts))
    c = QPointF(s * 0.5, s * 0.5)
    for q in pts:
        p.drawLine(c, q)


def _g_crate(p, s):
    _g_cube(p, s)
    p.drawLine(QPointF(s * 0.16, s * 0.5), QPointF(s * 0.5, s * 0.68))
    p.drawLine(QPointF(s * 0.84, s * 0.5), QPointF(s * 0.5, s * 0.68))


def _g_roto(p, s):
    pts = [QPointF(s * 0.2, s * 0.3), QPointF(s * 0.62, s * 0.14), QPointF(s * 0.86, s * 0.56), QPointF(s * 0.44, s * 0.86),
           QPointF(s * 0.14, s * 0.66)]
    p.drawPolygon(QPolygonF(pts))
    for q in pts:
        p.drawRect(QRectF(q.x() - s * 0.05, q.y() - s * 0.05, s * 0.1, s * 0.1))


def _g_text(p, s):
    """A solid letter T, drawn as its outline."""
    p.drawPolygon(QPolygonF([QPointF(s * 0.16, s * 0.16), QPointF(s * 0.84, s * 0.16), QPointF(s * 0.84, s * 0.34),
                             QPointF(s * 0.6, s * 0.34), QPointF(s * 0.6, s * 0.86), QPointF(s * 0.4, s * 0.86),
                             QPointF(s * 0.4, s * 0.34), QPointF(s * 0.16, s * 0.34)]))


def _g_shape(p, s):
    """A star: a logo or picture made solid."""
    pts = []
    for k in range(10):
        r = s * (0.4 if k % 2 == 0 else 0.17)
        a = math.pi / 2 + k * math.pi / 5
        pts.append(QPointF(s * 0.5 + r * math.cos(a), s * 0.54 - r * math.sin(a)))
    p.drawPolygon(QPolygonF(pts))


def _g_book(p, s):
    p.drawRoundedRect(QRectF(s * 0.18, s * 0.14, s * 0.64, s * 0.72), s * 0.05, s * 0.05)
    p.drawLine(QPointF(s * 0.32, s * 0.14), QPointF(s * 0.32, s * 0.86))
    p.drawLine(QPointF(s * 0.44, s * 0.34), QPointF(s * 0.7, s * 0.34))
    p.drawLine(QPointF(s * 0.44, s * 0.48), QPointF(s * 0.64, s * 0.48))


GLYPHS = {
    'star': _g_star, 'cube': _g_cube, 'flame': _g_flame, 'wind': _g_wind, 'palette': _g_palette, 'bulb': _g_bulb,
    'sparks': _g_sparks, 'spread': _g_spread, 'box': _g_box, 'camera': _g_camera, 'layers': _g_layers, 'film': _g_film,
    'drop': _g_drop, 'waves': _g_waves, 'snow': _g_snow, 'cloud': _g_cloud, 'weather': _g_weather, 'sky': _g_sky,
    'lava': _g_lava, 'search': _g_search, 'footage': _g_footage, 'open': _g_open, 'save': _g_save,
    'undo': _g_undo, 'redo': lambda p, s: _g_undo(p, s, True), 'render': _g_film, 'plus': _g_plus, 'trash': _g_trash,
    'copy': _g_copy, 'reset': _g_reset, 'grid': _g_grid, 'stats': _g_stats, 'fit': _g_fit, 'emitter': _g_emitter,
    'collider': _g_cube, 'light': _g_bulb, 'fabric': _g_fabric, 'book': _g_book,
    'logs': _g_logs, 'pool': _g_pool, 'line': _g_line, 'torch': _g_torch, 'ring': _g_ring, 'burst': _g_burst, 'jet': _g_jet,
    'spiral': _g_spiral, 'steam': _g_steam, 'pour': _g_pour, 'fountain': _g_fountain, 'ball': _g_ball, 'pillar': _g_pillar,
    'wall': _g_wall, 'house': _g_house, 'car': _g_car, 'hill': _g_hill, 'flag': _g_flag, 'spot': _g_spot, 'window': _g_window,
    'mesh': _g_mesh, 'crate': _g_crate, 'roto': _g_roto, 'text': _g_text, 'shape': _g_shape,
    'cart': _g_cart, 'turntable': _g_turntable, 'windmill': _g_windmill, 'ramp': _g_ramp,
    'grass': _g_grass, 'strands': _g_grass, 'wheat': _g_wheat, 'reeds': _g_reeds,
    'matter': _g_grains, 'grains': _g_grains, 'mud': _g_mud, 'jelly': _g_jelly, 'snowball': _g_snowball, 'clay': _g_clay,
    'play': lambda p, s: p.drawPolygon(QPolygonF([QPointF(s * 0.3, s * 0.2), QPointF(s * 0.8, s * 0.5), QPointF(s * 0.3, s * 0.8)])),
    'stop': lambda p, s: p.drawRect(QRectF(s * 0.25, s * 0.25, s * 0.5, s * 0.5)),
}


def paint_glyph(p: QPainter, name, rect: QRectF, color, width=1.5):
    """Draw a line glyph into a square rect in a colour."""
    p.save()
    p.setRenderHint(QPainter.Antialiasing)
    p.translate(rect.x(), rect.y())
    p.setPen(QPen(QColor(color), width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.setBrush(Qt.NoBrush)
    GLYPHS[name](p, rect.width())
    p.restore()


def glyph_icon(name, color=theme.TEXT, size=18, active=None, checked=None):
    """A QIcon of a line glyph: `color` normally, `active` on hover, `checked` when on."""
    icon = QIcon()

    def pm(col):
        px = QPixmap(size * 2, size * 2)
        px.setDevicePixelRatio(2.0)
        px.fill(Qt.transparent)
        p = QPainter(px)
        paint_glyph(p, name, QRectF(0, 0, size, size), col, 1.5 if size >= 16 else 1.3)
        p.end()
        return px
    icon.addPixmap(pm(color), QIcon.Normal, QIcon.Off)
    icon.addPixmap(pm(active or color), QIcon.Active, QIcon.Off)
    icon.addPixmap(pm(theme.FAINT), QIcon.Disabled, QIcon.Off)
    if checked:
        icon.addPixmap(pm(checked), QIcon.Normal, QIcon.On)
        icon.addPixmap(pm(checked), QIcon.Active, QIcon.On)
    return icon
