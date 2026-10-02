"""Timing: when each thing in the scene does what it does, as bars over the shot. A source's bar runs from when it
starts to when it stops (faded in and out at its ends); drag the bar to move it in time, drag an end to start or
stop it then (past the first frame: already going; past the last: going to the end). A cloth shows when it is let
go, a pour that fills once when it fills. Each thing's keys show on its row as diamonds: drag one to move all its
keys at that frame. Right-click a row for more."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QMenu, QWidget

from ..scene import kinds as K
from ..scene.anim import Curve
from . import icons, theme

LABEL_W = 230   # as the animation editor, so the two line up
ROW_H = 24
HEAD_H = 18
PRE = -100.0    # Ignite at: already burning before the first frame
EDGE = 6        # px of a bar's end that drags the end


def rows_of(scene):
    """[(kind, i, what)] for everything with a time: sources (a start and a stop, or a fill), cloths let go at a
    time, and anything with keys. what: 'span', 'fill', 'release' or 'keys'."""
    out = []
    for i, e in enumerate(scene.emitters):
        out.append(('emitter', i, 'fill' if e.get('liquid_mode') == 'fill' and _liquid(scene, e) else 'span'))
    for i, f in enumerate(scene.fabrics):
        out.append(('fabric', i, 'release'))
    for i, m in enumerate(K.items(scene, 'matter')):
        out.append(('matter', i, 'span' if m.get('pours') else 'release'))
    for kind, items in (('collider', scene.colliders), ('light', scene.lights)):
        for i, d in enumerate(items):
            if any(isinstance(v, Curve) for v in d.values()):
                out.append((kind, i, 'keys'))
    return out


def _liquid(scene, e):
    return scene.kind == 'liquid' or (scene.kind == 'both' and e.get('emits') in ('liquid', 'lava'))


def _items(scene, kind):
    return K.items(scene, kind)


POUR_KEYS = {'start': 'pour_start', 'stop': 'pour_stop'}


def _times(kind, d):
    """An object's times as the lanes read them: a pour's from and until as a source's start and stop."""
    if kind != 'matter' or not d.get('pours'):
        return d
    x = dict(d)
    x['start'], x['stop'] = float(d.get('pour_start', 0.0)), float(d.get('pour_stop', 5.0))
    return x


def key_frames(d):
    fr = set()
    for v in d.values():
        if isinstance(v, Curve):
            fr.update(float(k[0]) for k in v.keys)
    return sorted(fr)


class TimingEditor(QWidget):
    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.rows = []
        self.scroll = 0
        self._drag = None
        self.hover = None
        self.setMouseTracking(True)
        self.setMinimumHeight(150)
        doc.sceneReplaced.connect(self.rebuild)
        doc.structureChanged.connect(self.rebuild)
        doc.paramChanged.connect(lambda *_: self.rebuild())
        doc.frameChanged.connect(lambda *_: self.update())
        doc.selectionChanged.connect(lambda *_: self.update())
        self.rebuild()

    def rebuild(self):
        self.rows = rows_of(self.doc.scene)
        self.update()

    # -- geometry -----------------------------------------------------------------------------------------

    def _x(self, f):
        sc = self.doc.scene
        a, b = sc.start, max(sc.end, sc.start + 1)
        return LABEL_W + 10 + (f - a) / (b - a) * (self.width() - LABEL_W - 22)

    def _f(self, x):
        sc = self.doc.scene
        a, b = sc.start, max(sc.end, sc.start + 1)
        return a + (x - LABEL_W - 10) / max(1.0, self.width() - LABEL_W - 22) * (b - a)

    def _fr(self, t):
        """Frame of a time in seconds from the first frame."""
        sc = self.doc.scene
        return sc.start + t * sc.fps

    def _t(self, f):
        sc = self.doc.scene
        return round((f - sc.start) / sc.fps, 3)

    def _span(self, d):
        """(start x, stop x, starts before the shot, goes on past it) of a source's bar."""
        sc = self.doc.scene
        x0, x1 = self._x(sc.start), self._x(sc.end)
        st, sp = float(d.get('start', PRE)), float(d.get('stop', -1.0))
        pre = st <= -99.0 or self._fr(st) < sc.start
        post = sp < 0 or self._fr(sp) > sc.end
        a = x0 - 8 if pre else self._x(self._fr(st))
        b = x1 + 8 if post else self._x(self._fr(sp))
        return a, max(b, a + 2), pre, post

    def _row_at(self, y):
        i = int((y - HEAD_H) // ROW_H) + self.scroll
        return i if 0 <= i < len(self.rows) and y >= HEAD_H else None

    def _hit(self, pos):
        """What is under the mouse: (label, row), (key, row, frame), (start|stop|body|mark, row) or None."""
        r = self._row_at(pos.y())
        if r is None:
            return None
        if pos.x() < LABEL_W:
            return ('label', r)
        kind, i, what = self.rows[r]
        d = _times(kind, _items(self.doc.scene, kind)[i])
        for fr in key_frames(d):
            if abs(self._x(fr) - pos.x()) < 6 and abs(pos.y() - (HEAD_H + (r - self.scroll) * ROW_H + ROW_H - 6)) < 7:
                return ('key', r, fr)
        if what == 'span':
            a, b, pre, post = self._span(d)
            if not pre and abs(pos.x() - a) <= EDGE:
                return ('start', r)
            if not post and abs(pos.x() - b) <= EDGE:
                return ('stop', r)
            if a <= pos.x() <= b:
                return ('body', r)
        elif what in ('fill', 'release'):
            t = float(d.get('start' if what == 'fill' else 'release', -1.0))
            if t >= 0 and abs(self._x(self._fr(t)) - pos.x()) <= 8:
                return ('mark', r)
        return None

    # -- painting -----------------------------------------------------------------------------------------

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
                       'Nothing in the scene has a time yet. Add a source from Create: its bar shows here, from when it '
                       'starts to when it stops, to drag along the shot.')
            return
        span = max(1, b - a)
        px = (self.width() - LABEL_W - 22) / span
        step = next((s for s in (1, 2, 5, 10, 12, 24, 25, 30, 48, 50, 60, 100, 120, 240, 500, 1000) if s * px >= 46), 1000)
        for fr in range(int(math.ceil(a / step) * step), b + 1, step):
            x = self._x(fr)
            p.setPen(QColor(theme.FAINT))
            p.drawText(QPointF(x + 2, 12), f'{(fr - a) / sc.fps:.3g} s' if step >= sc.fps else str(fr))
            p.setPen(QColor(theme.LINE))
            p.drawLine(QPointF(x, HEAD_H - 4), QPointF(x, self.height()))
        visible = max(1, (self.height() - HEAD_H) // ROW_H)
        self.scroll = max(0, min(self.scroll, len(self.rows) - visible))
        chosen = set(self.doc.selected_objects())
        x0, x1 = self._x(a), self._x(b)
        for r in range(self.scroll, min(len(self.rows), self.scroll + visible)):
            kind, i, what = self.rows[r]
            d = _times(kind, _items(sc, kind)[i])
            y = HEAD_H + (r - self.scroll) * ROW_H
            row = QRectF(0, y, self.width(), ROW_H)
            if (kind, i) in chosen:
                p.fillRect(row, QColor(255, 140, 60, 22))
            elif r % 2:
                p.fillRect(row, QColor(255, 255, 255, 6))
            colour = QColor(theme.OBJECT_COLOURS[kind])
            if kind == 'emitter' and _liquid(sc, d):
                colour = QColor('#ff6a2a' if d.get('emits') == 'lava' else '#6fb6ff')
            icons.paint_glyph(p, kind, QRectF(10, y + 4, 16, 16), colour.name(), 1.3)
            p.setPen(QColor(theme.TEXT if (kind, i) in chosen else theme.MUTED if d.get('enabled', True) else theme.FAINT))
            p.drawText(QRectF(32, y, LABEL_W - 40, ROW_H), Qt.AlignVCenter | Qt.AlignLeft,
                       p.fontMetrics().elidedText(d['name'], Qt.ElideRight, LABEL_W - 44))
            cy = y + ROW_H / 2
            if what == 'span':
                bx0, bx1, pre, post = self._span(d)
                if self._drag and self._drag.get('row') == r and 'preview' in self._drag:
                    bx0, bx1, pre, post = self._drag['preview']
                bar = QRectF(max(bx0, x0 - 8), y + 5, min(bx1, x1 + 8) - max(bx0, x0 - 8), ROW_H - 12)
                g = QLinearGradient(bar.left(), 0, bar.right(), 0)
                fade_in = float(d.get('fade_in', 0.0)) * sc.fps * px / max(bar.width(), 1)
                fade_out = float(d.get('fade_out', 0.0)) * sc.fps * px / max(bar.width(), 1)
                c0 = QColor(colour)
                c0.setAlpha(60)
                c1 = QColor(colour)
                c1.setAlpha(170 if d.get('enabled', True) else 60)
                g.setColorAt(0.0, c1 if pre else c0)
                g.setColorAt(min(0.49, 0.0 if pre else fade_in), c1)
                g.setColorAt(max(0.51, 1.0 if post else 1.0 - fade_out), c1)
                g.setColorAt(1.0, c1 if post else c0)
                p.setPen(Qt.NoPen)
                p.setBrush(g)
                p.drawRoundedRect(bar, 4, 4)
                p.setPen(QColor('#16110c'))
                txt = ('already on' if pre else f'{d["start"]:.2f} s') + ' → ' + ('to the end' if post else f'{d["stop"]:.2f} s')
                if bar.width() > p.fontMetrics().horizontalAdvance(txt) + 12:
                    p.drawText(bar, Qt.AlignCenter, txt)
                p.setPen(QPen(QColor(theme.TEXT), 2))
                for x, open_end in ((bar.left(), pre), (bar.right(), post)):
                    if not open_end:
                        p.drawLine(QPointF(x, y + 6), QPointF(x, y + ROW_H - 6))
            elif what in ('fill', 'release'):
                t = float(d.get('start' if what == 'fill' else 'release', -1.0))
                label = ('fills' if what == 'fill' else 'lets go')
                if what == 'fill' and t <= -99:
                    t = 0.0
                if t >= 0:
                    x = self._x(self._fr(t)) if not (self._drag and self._drag.get('row') == r) else self._drag.get('x', self._x(self._fr(t)))
                    p.setPen(QPen(colour, 2))
                    p.drawLine(QPointF(x, y + 4), QPointF(x, y + ROW_H - 4))
                    p.setBrush(colour)
                    p.drawEllipse(QPointF(x, cy), 4, 4)
                    p.setPen(QColor(theme.MUTED))
                    p.drawText(QRectF(x + 8, y, 120, ROW_H), Qt.AlignVCenter | Qt.AlignLeft, f'{label} at {t:.2f} s')
                else:
                    p.setPen(QColor(theme.FAINT))
                    p.drawText(QRectF(LABEL_W + 12, y, 300, ROW_H), Qt.AlignVCenter | Qt.AlignLeft,
                               'held all through (right-click to let it go at a frame)')
            for fr in key_frames(d):   # its keys
                if not (a - 0.5 <= fr <= b + 0.5):
                    continue
                kx = self._x(fr + (self._drag['df'] if self._drag and self._drag.get('key') == (r, fr) else 0))
                ky = y + ROW_H - 6
                hot = self.hover == ('key', r, fr)
                rr = 5 if hot else 4
                p.setPen(QPen(QColor(theme.KEY), 1.1))
                p.setBrush(QColor('#ffe7a3' if hot else theme.KEY))
                p.drawPolygon(QPolygonF([QPointF(kx, ky - rr), QPointF(kx + rr, ky), QPointF(kx, ky + rr), QPointF(kx - rr, ky)]))
        x = self._x(self.doc.frame)
        p.setPen(QPen(QColor(theme.ACCENT), 1.5))
        p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        p.setPen(QPen(QColor(theme.LINE_HI), 1))
        p.drawLine(QPointF(LABEL_W, 0), QPointF(LABEL_W, self.height()))
        if len(self.rows) > visible:
            p.setPen(QColor(theme.FAINT))
            p.drawText(QRectF(10, self.height() - 16, LABEL_W - 16, 14), Qt.AlignLeft,
                       f'{self.scroll + 1}–{min(len(self.rows), self.scroll + visible)} of {len(self.rows)} · scroll for more')

    # -- editing ------------------------------------------------------------------------------------------

    def _set_times(self, kind, i, vals, label, merge=True):
        if kind == 'matter':   # a pour's from and until; 'to the end' and 'already going' within their ranges
            vals = {POUR_KEYS.get(k, k): (600.0 if k == 'stop' and float(v) < 0 else max(-10.0, float(v))) for k, v in vals.items()}

        def fn(s):
            d = _items(s, kind)[i]
            for k, v in vals.items():
                d[k] = float(v)
        self.doc.edit(label, fn, merge_key=('timing', kind, i, self.doc._gen) if merge else None, path=(kind, i, next(iter(vals))))

    def mousePressEvent(self, e):
        pos = e.position()
        h = self._hit(pos)
        if e.button() == Qt.RightButton:
            r = self._row_at(pos.y())
            if r is not None:
                self._menu(e.globalPosition().toPoint(), r, pos)
            return
        if e.button() != Qt.LeftButton or h is None:
            if e.button() == Qt.LeftButton and pos.x() >= LABEL_W:   # empty track: scrub there
                self.doc.set_frame(int(round(self._f(pos.x()))))
            return
        r = h[1]
        kind, i, what = self.rows[r]
        d = _times(kind, _items(self.doc.scene, kind)[i])
        if (kind, i) not in self.doc.selected_objects():
            self.doc.select((kind, i), force=True)
        if h[0] == 'label':
            return
        if h[0] == 'key':
            self._drag = {'row': r, 'key': (r, h[2]), 'x0': pos.x(), 'df': 0}
        else:
            self._drag = {'row': r, 'part': h[0], 'x0': pos.x(), 'start': float(d.get('start', PRE)),
                          'stop': float(d.get('stop', -1.0)), 'release': float(d.get('release', -1.0))}

    def mouseMoveEvent(self, e):
        pos = e.position()
        d = self._drag
        if d is None:
            h = self._hit(pos)
            if h != self.hover:
                self.hover = h
                cur = {'start': Qt.SizeHorCursor, 'stop': Qt.SizeHorCursor, 'body': Qt.OpenHandCursor, 'mark': Qt.SizeHorCursor,
                       'key': Qt.PointingHandCursor, 'label': Qt.PointingHandCursor}.get(h[0] if h else None, Qt.ArrowCursor)
                self.setCursor(cur)
                self.update()
            return
        sc = self.doc.scene
        kind, i, what = self.rows[d['row']]
        df = round((pos.x() - d['x0']) / max(1e-6, (self.width() - LABEL_W - 22) / max(1, sc.end - sc.start)))
        if 'key' in d:
            d['df'] = df
            self.update()
            return
        dt = df / sc.fps
        part = d['part']
        last = self._t(sc.end)
        if part == 'mark':
            k = 'start' if what == 'fill' else 'release'
            t0 = d[k] if d[k] >= 0 else 0.0
            self._set_times(kind, i, {k: max(0.0, round(t0 + dt, 3))}, 'Move in time')
        elif part == 'start':
            t = d['start'] + dt if d['start'] > -99 else dt
            stop = d['stop']
            if t < -0.5 / sc.fps:
                t = PRE   # dragged off the start: already going
            elif stop >= 0:
                t = min(t, stop - 1.0 / sc.fps)
            self._set_times(kind, i, {'start': round(t, 3)}, 'Change when it starts')
        elif part == 'stop':
            t = d['stop'] + dt
            if t > last + 0.5 / sc.fps:
                t = -1.0   # dragged off the end: going to the end
            else:
                t = max(t, (d['start'] if d['start'] > -99 else 0.0) + 1.0 / sc.fps)
            self._set_times(kind, i, {'stop': round(t, 3)}, 'Change when it stops')
        elif part == 'body':
            vals = {}
            if d['start'] > -99:
                vals['start'] = round(max(0.0, d['start'] + dt), 3)
            if d['stop'] >= 0:
                vals['stop'] = round(max(0.01, d['stop'] + dt), 3)
            if vals:
                self._set_times(kind, i, vals, 'Move in time')
        self.update()

    def mouseReleaseEvent(self, e):
        d = self._drag
        self._drag = None
        if d is None:
            return
        if 'key' in d and d['df']:
            r, fr = d['key']
            kind, i, _ = self.rows[r]
            obj = _items(self.doc.scene, kind)[i]
            keys = [((kind, i, k), fr) for k, v in obj.items() if isinstance(v, Curve) and v.has_key(fr)]
            self.doc.move_keys(keys, d['df'])
        self.doc.end_drag()
        self.update()

    def mouseDoubleClickEvent(self, e):
        h = self._hit(e.position())
        if h and h[0] == 'label':
            win = self.window()
            kind, i, _ = self.rows[h[1]]
            if hasattr(win, 'props'):
                d = _items(self.doc.scene, kind)[i]
                key = ('start' if kind == 'emitter' else 'release' if kind == 'fabric' else
                       ('pour_start' if d.get('pours') else 'release') if kind == 'matter' else 'position')
                win.props.reveal((kind, i, key))

    def wheelEvent(self, e):
        self.scroll = max(0, self.scroll - int(round(e.angleDelta().y() / 120)))
        self.update()

    def _menu(self, gpos, r, pos):
        kind, i, what = self.rows[r]
        d = _items(self.doc.scene, kind)[i]
        t = self._t(self.doc.frame)
        m = QMenu(self)
        if what == 'span':
            m.addAction(f'Start at this frame ({self.doc.frame})', lambda: self._set_times(kind, i, {'start': t}, 'Start here', False))
            m.addAction(f'Stop at this frame ({self.doc.frame})', lambda: self._set_times(kind, i, {'stop': t}, 'Stop here', False))
            m.addAction('Already going at the first frame', lambda: self._set_times(kind, i, {'start': PRE}, 'Already going', False))
            m.addAction('Going to the end', lambda: self._set_times(kind, i, {'stop': -1.0}, 'Going to the end', False))
            m.addAction('A short burst here', lambda: self._set_times(kind, i, {'start': t, 'stop': round(t + 0.2, 3)}, 'Short burst', False))
        elif what == 'fill':
            m.addAction(f'Fill at this frame ({self.doc.frame})', lambda: self._set_times(kind, i, {'start': t}, 'Fill here', False))
        elif what == 'release':
            m.addAction(f'Let go at this frame ({self.doc.frame})', lambda: self._set_times(kind, i, {'release': t}, 'Let go here', False))
            m.addAction('Never let go', lambda: self._set_times(kind, i, {'release': -1.0}, 'Hold it', False))
        m.addSeparator()
        win = self.window()
        if hasattr(win, 'doc'):
            from .actions import fill_menu
            sub = m.addMenu(d['name'])
            fill_menu(sub, win, (kind, i), path_mode=getattr(getattr(win, 'viewport', None), 'start_path', None))
        m.exec(gpos)
