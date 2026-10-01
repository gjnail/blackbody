"""The animation editor: every animated setting in the scene as a row, with its keys on a track over the
shot and a trace of its value. Drag keys to move them in time, double-click a track to add a key, Delete
removes the selected keys, right-click to change how a key eases into the next. Click a setting's name
to bring it up in Properties."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QMenu, QWidget

from ..scene.anim import Curve
from ..scene.params import SECTION_TITLES
from . import theme

LABEL_W = 230
ROW_H = 22
HEAD_H = 18
INTERP_NAMES = {'smooth': 'Smooth', 'linear': 'Linear', 'step': 'Hold (step)'}


def animated_paths(scene):
    """[(path, label, context)] for every animated setting, sections first, then objects."""
    out = []
    for sec, vals in scene.data.items():
        for k, v in vals.items():
            if isinstance(v, Curve):
                try:
                    label = scene.spec((sec, k)).label
                except KeyError:
                    label = k
                out.append(((sec, k), label, SECTION_TITLES.get(sec, sec)))
    for kind, items in (('emitter', scene.emitters), ('collider', scene.colliders), ('light', scene.lights), ('fabric', scene.fabrics)):
        for i, d in enumerate(items):
            for k, v in d.items():
                if isinstance(v, Curve):
                    try:
                        label = scene.spec((kind, i, k)).label
                    except KeyError:
                        label = k
                    out.append(((kind, i, k), label, d.get('name', kind)))
    return out


class KeyEditor(QWidget):
    revealed = Signal(object)   # a setting's path: bring it up in Properties

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.rows = []
        self.sel = set()          # {(path, frame)}
        self.hover = None
        self._drag = None
        self.scroll = 0
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setMinimumHeight(120)
        doc.sceneReplaced.connect(self.rebuild)
        doc.paramChanged.connect(lambda *_: self.rebuild())
        doc.structureChanged.connect(self.rebuild)
        doc.frameChanged.connect(lambda *_: self.update())
        self.rebuild()

    # -- data ------------------------------------------------------------------------------------------

    def rebuild(self):
        self.rows = animated_paths(self.doc.scene)
        paths = {r[0] for r in self.rows}
        self.sel = {(p, f) for p, f in self.sel if p in paths and self._curve(p) is not None and self._curve(p).has_key(f)}
        self.update()

    def _curve(self, path):
        try:
            return self.doc.scene.curve(path)
        except (KeyError, IndexError):
            return None

    # -- geometry -------------------------------------------------------------------------------------------

    def _x(self, f):
        sc = self.doc.scene
        a, b = sc.start, max(sc.end, sc.start + 1)
        return LABEL_W + 10 + (f - a) / (b - a) * (self.width() - LABEL_W - 22)

    def _f(self, x):
        sc = self.doc.scene
        a, b = sc.start, max(sc.end, sc.start + 1)
        return a + (x - LABEL_W - 10) / max(1.0, self.width() - LABEL_W - 22) * (b - a)

    def _row_at(self, y):
        i = int((y - HEAD_H) // ROW_H) + self.scroll
        return i if 0 <= i < len(self.rows) and y >= HEAD_H else None

    def _key_at(self, pos):
        i = self._row_at(pos.y())
        if i is None or pos.x() < LABEL_W:
            return None
        path = self.rows[i][0]
        c = self._curve(path)
        if c is None:
            return None
        best, bd = None, 7.0
        for f in c.frames():
            d = abs(self._x(f) - pos.x())
            if d < bd:
                best, bd = (path, f), d
        return best

    # -- painting -------------------------------------------------------------------------------------------

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.PANEL))
        sc = self.doc.scene
        a, b = sc.start, sc.end
        f = QFont(self.font())
        f.setPointSizeF(8.3)
        p.setFont(f)
        if not self.rows:
            p.setPen(QColor(theme.MUTED))
            p.drawText(QRectF(self.rect()).adjusted(16, 8, -16, -8), Qt.AlignCenter | Qt.TextWordWrap,
                       'Nothing is animated yet. Click the ◇ next to a setting in Properties to key it at this frame, '
                       'then change the frame and the value: it animates between the keys. Every animated setting shows up here.')
            return
        # frame ticks
        p.setPen(QColor(theme.FAINT))
        span = max(1, b - a)
        px = (self.width() - LABEL_W - 22) / span
        step = next((s for s in (1, 2, 5, 10, 12, 24, 25, 30, 48, 50, 60, 100, 120, 240, 500, 1000) if s * px >= 46), 1000)
        for fr in range(int(math.ceil(a / step) * step), b + 1, step):
            x = self._x(fr)
            p.drawText(QPointF(x + 2, 12), str(fr))
            p.setPen(QColor(theme.LINE))
            p.drawLine(QPointF(x, HEAD_H - 4), QPointF(x, self.height()))
            p.setPen(QColor(theme.FAINT))
        visible = max(1, (self.height() - HEAD_H) // ROW_H)
        self.scroll = max(0, min(self.scroll, len(self.rows) - visible))
        for r in range(self.scroll, min(len(self.rows), self.scroll + visible)):
            path, label, ctx = self.rows[r]
            y = HEAD_H + (r - self.scroll) * ROW_H
            row = QRectF(0, y, self.width(), ROW_H)
            if r % 2:
                p.fillRect(row, QColor(255, 255, 255, 6))
            selected = any(sp == path for sp, _ in self.sel)
            p.setPen(QColor(theme.TEXT if selected else theme.MUTED))
            text = p.fontMetrics().elidedText(f'{ctx} › {label}', Qt.ElideMiddle, LABEL_W - 70)
            p.drawText(QRectF(10, y, LABEL_W - 76, ROW_H), Qt.AlignVCenter | Qt.AlignLeft, text)
            c = self._curve(path)
            if c is None:
                continue
            v = c.eval(self.doc.frame)
            p.setPen(QColor(theme.FAINT))
            p.drawText(QRectF(LABEL_W - 66, y, 60, ROW_H), Qt.AlignVCenter | Qt.AlignRight, _fmt(v))
            self._trace(p, c, y, a, b)
            for fr in c.frames():
                if not (a - 0.5 <= fr <= b + 0.5):
                    continue
                x = self._x(fr + (self._drag['df'] if self._drag and (path, fr) in self.sel and self._drag.get('moved') else 0))
                on = (path, fr) in self.sel
                hot = self.hover == (path, fr)
                cy = y + ROW_H / 2
                rr = 5.5 if (on or hot) else 4.5
                poly = QPolygonF([QPointF(x, cy - rr), QPointF(x + rr, cy), QPointF(x, cy + rr), QPointF(x - rr, cy)])
                p.setPen(QPen(QColor('#ffffff' if on else theme.KEY), 1.2))
                p.setBrush(QColor(theme.KEY if not on else '#ffe7a3'))
                p.drawPolygon(poly)
        # the playhead
        x = self._x(self.doc.frame)
        p.setPen(QPen(QColor(theme.ACCENT), 1.5))
        p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        p.setPen(QPen(QColor(theme.LINE_HI), 1))
        p.drawLine(QPointF(LABEL_W, 0), QPointF(LABEL_W, self.height()))
        if len(self.rows) > visible:
            p.setPen(QColor(theme.FAINT))
            p.drawText(QRectF(10, self.height() - 16, LABEL_W - 16, 14), Qt.AlignLeft,
                       f'{self.scroll + 1}–{min(len(self.rows), self.scroll + visible)} of {len(self.rows)} · scroll for more')

    def _trace(self, p, c, y, a, b):
        """The value over the shot, faint, scaled to the row (each part of a colour or position on its own)."""
        n = 64
        frames = [a + (b - a) * i / (n - 1) for i in range(n)]
        vals = [c.eval(f) for f in frames]
        if not vals or isinstance(vals[0], (bool, str)) or vals[0] is None:
            return
        comps = [vals] if not isinstance(vals[0], (tuple, list)) else [[v[k] for v in vals] for k in range(len(vals[0]))]
        cols = ['#d9a24c'] if len(comps) == 1 else ['#e06c6c', '#7cc47a', '#6f9fe8']
        for comp, col in zip(comps, cols):
            lo, hi = min(comp), max(comp)
            if hi - lo < 1e-9:
                continue
            path = QPainterPath()
            for i, (f, v) in enumerate(zip(frames, comp)):
                q = QPointF(self._x(f), y + ROW_H - 4 - (v - lo) / (hi - lo) * (ROW_H - 8))
                path.moveTo(q) if i == 0 else path.lineTo(q)
            c2 = QColor(col)
            c2.setAlpha(120)
            p.setPen(QPen(c2, 1.0))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)

    # -- interaction ------------------------------------------------------------------------------------------

    def mousePressEvent(self, e):
        pos = e.position()
        i = self._row_at(pos.y())
        if e.button() == Qt.RightButton:
            k = self._key_at(pos)
            if k is not None and k not in self.sel:
                self.sel = {k}
                self.update()
            self._menu(e.globalPosition().toPoint(), k)
            return
        if e.button() != Qt.LeftButton:
            return
        if i is not None and pos.x() < LABEL_W:
            self.revealed.emit(self.rows[i][0])
            self.sel = {(self.rows[i][0], f) for f in (self._curve(self.rows[i][0]) or Curve()).frames()}
            self.update()
            return
        k = self._key_at(pos)
        if k is not None:
            if e.modifiers() & Qt.ShiftModifier:
                self.sel ^= {k}
            elif k not in self.sel:
                self.sel = {k}
            self._drag = {'x0': pos.x(), 'df': 0, 'moved': False}
        else:
            if not e.modifiers() & Qt.ShiftModifier:
                self.sel = set()
            if pos.x() > LABEL_W:   # a click on a track sets the frame
                if self.doc.playing:
                    self.doc.set_playing(False)
                self.doc.set_frame(int(round(self._f(pos.x()))))
                self._drag = {'scrub': True}
        self.update()

    def mouseMoveEvent(self, e):
        pos = e.position()
        d = self._drag
        if d is None:
            h = self._key_at(pos)
            if h != self.hover:
                self.hover = h
                self.setCursor(Qt.SizeHorCursor if h else Qt.ArrowCursor)
                self.update()
            return
        if d.get('scrub'):
            self.doc.set_frame(int(round(self._f(pos.x()))))
            return
        sc = self.doc.scene
        per_frame = (self.width() - LABEL_W - 22) / max(1, sc.end - sc.start)
        df = int(round((pos.x() - d['x0']) / max(per_frame, 1e-6)))
        if df != d['df'] or not d['moved']:
            d['df'] = df
            d['moved'] = d['moved'] or abs(pos.x() - d['x0']) > 3
            self.update()

    def mouseReleaseEvent(self, e):
        d, self._drag = self._drag, None
        if d and d.get('moved') and d['df']:
            moved = self.doc.move_keys(sorted(self.sel, key=lambda k: (str(k[0]), k[1])), d['df'])
            self.sel = set(moved)
        self.update()

    def mouseDoubleClickEvent(self, e):
        pos = e.position()
        i = self._row_at(pos.y())
        if i is None or pos.x() < LABEL_W or e.button() != Qt.LeftButton:
            return
        f = int(round(self._f(pos.x())))
        path = self.rows[i][0]
        self.doc.add_key(path, f)
        self.sel = {(path, float(f))}
        self.update()

    def wheelEvent(self, e):
        if self.rows:
            self.scroll = max(0, self.scroll - int(e.angleDelta().y() / 120))
            self.update()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.sel:
            self.doc.delete_keys(sorted(self.sel, key=lambda k: (str(k[0]), k[1])))
            self.sel = set()
            self.update()
        elif e.key() == Qt.Key_A and e.modifiers() & Qt.ControlModifier:
            self.sel = {(p, f) for p, _, _ in self.rows for f in (self._curve(p) or Curve()).frames()}
            self.update()
        else:
            super().keyPressEvent(e)

    def _menu(self, gpos, key):
        m = QMenu(self)
        if self.sel:
            n = len(self.sel)
            m.addAction(f'Delete {"this key" if n == 1 else f"{n} keys"}', lambda: (self.doc.delete_keys(sorted(self.sel, key=lambda k: (str(k[0]), k[1]))),
                                                                            self.sel.clear(), self.update()))
            sub = m.addMenu('Ease into the next key')
            for k, label in INTERP_NAMES.items():
                sub.addAction(label, lambda k=k: self.doc.set_key_interp(sorted(self.sel, key=lambda x: (str(x[0]), x[1])), k))
            if key is not None:
                m.addAction(f'Go to frame {int(key[1])}', lambda: self.doc.set_frame(int(key[1])))
        if key is None:
            m.addAction('Select all keys  (Ctrl+A)', lambda: (setattr(self, 'sel', {(p, f) for p, _, _ in self.rows
                                                                                  for f in (self._curve(p) or Curve()).frames()}), self.update()))
        m.exec(gpos)


def _fmt(v):
    if isinstance(v, bool):
        return 'on' if v else 'off'
    if isinstance(v, (int, float)):
        return f'{v:.3g}'
    if isinstance(v, (tuple, list)):
        return ' '.join(f'{x:.2g}' for x in v)
    return str(v)
