"""Timeline: transport controls, frame ruler, cache bar, keyframes and emitter burn spans."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, QSettings, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QAbstractSpinBox, QHBoxLayout, QLabel, QSizePolicy, QSpinBox, QToolButton, QVBoxLayout, QWidget

from . import icons, theme


def timecode(frame, fps, start=0):
    f = max(0, frame - start)
    fr = int(round(fps))
    s, ff = divmod(f, max(fr, 1))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f'{h:02d}:{m:02d}:{s:02d}:{ff:02d}'


class Ruler(QWidget):
    scrubbed = Signal(int)
    released = Signal()

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.cached = set()
        self.progress_frame = None
        self.setMinimumHeight(54)
        self.setMouseTracking(True)
        self._drag = False

    def _x(self, f):
        sc = self.doc.scene
        a, b = sc.start, max(sc.end, sc.start + 1)
        return 12 + (f - a) / (b - a) * (self.width() - 24)

    def _f(self, x):
        sc = self.doc.scene
        a, b = sc.start, max(sc.end, sc.start + 1)
        return int(round(a + (x - 12) / max(1.0, self.width() - 24) * (b - a)))

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.PANEL))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.CARD))
        p.drawRoundedRect(QRectF(4, 0, self.width() - 8, self.height()), 6, 6)
        sc = self.doc.scene
        a, b = sc.start, sc.end
        w = self.width()
        f = QFont(theme.mono_font(8))
        p.setFont(f)
        # ticks
        span = max(1, b - a)
        px_per = (w - 24) / span
        steps = [1, 2, 5, 10, 12, 24, 25, 30, 48, 50, 60, 100, 120, 240, 250, 500, 1000]
        major = next((s for s in steps if s * px_per >= 60), steps[-1])
        minor = next((s for s in steps if s * px_per >= 8 and major % s == 0), major)
        p.setPen(QPen(QColor(theme.LINE), 1))
        first = int(math.ceil(a / minor) * minor)
        for fr in range(first, b + 1, minor):
            x = self._x(fr)
            p.drawLine(QPointF(x, 0), QPointF(x, 5 if fr % major else 9))
        p.setPen(QColor(theme.MUTED))
        first = int(math.ceil(a / major) * major)
        for fr in range(first, b + 1, major):
            p.drawText(QPointF(self._x(fr) + 3, 18), str(fr))
        # emitter burn spans
        y = 24
        for i, em in enumerate(sc.emitters):
            if not em['enabled']:
                continue
            s0 = sc.start + em['start'] * sc.fps if em['start'] > -99 else a
            s1 = sc.start + em['stop'] * sc.fps if em['stop'] >= 0 else b
            x0, x1 = max(self._x(s0), 12), min(self._x(s1), w - 12)
            if x1 > x0:
                col = QColor(theme.ACCENT)
                col.setAlpha(150 if self.doc.selection == ('emitter', i) else 70)
                p.fillRect(QRectF(x0, y, x1 - x0, 3), col)
            y += 4
            if y > 36:
                break
        # cache bar
        cy = self.height() - 12
        p.fillRect(QRectF(12, cy, w - 24, 4), QColor(theme.FIELD))
        if self.cached:
            col = QColor(theme.GOOD)
            col.setAlpha(200)
            run_start = None
            prev = None
            for fr in sorted(self.cached):
                if run_start is None:
                    run_start = prev = fr
                elif fr == prev + 1:
                    prev = fr
                else:
                    p.fillRect(QRectF(self._x(run_start - 0.5), cy, self._x(prev + 0.5) - self._x(run_start - 0.5), 4), col)
                    run_start = prev = fr
            if run_start is not None:
                p.fillRect(QRectF(self._x(run_start - 0.5), cy, self._x(prev + 0.5) - self._x(run_start - 0.5), 4), col)
        # keyframes
        ky = self.height() - 24
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.KEY))
        for fr in sc.key_frames():
            if a <= fr <= b:
                x = self._x(fr)
                p.drawPolygon(QPolygonF([QPointF(x, ky - 4), QPointF(x + 4, ky), QPointF(x, ky + 4), QPointF(x - 4, ky)]))
        # playhead
        x = self._x(self.doc.frame)
        p.setPen(QPen(QColor(theme.ACCENT), 1.5))
        p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        p.setBrush(QColor(theme.ACCENT))
        p.setPen(Qt.NoPen)
        p.drawPolygon(QPolygonF([QPointF(x - 5, 0), QPointF(x + 5, 0), QPointF(x, 6)]))
        if self.progress_frame is not None:
            xp = self._x(self.progress_frame)
            p.setPen(QPen(QColor(theme.GOOD), 1, Qt.DashLine))
            p.drawLine(QPointF(xp, 20), QPointF(xp, self.height()))

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = True
            self.scrubbed.emit(self._f(e.position().x()))

    def mouseMoveEvent(self, e):
        if self._drag:
            self.scrubbed.emit(self._f(e.position().x()))

    def mouseReleaseEvent(self, e):
        if self._drag:
            self._drag = False
            self.released.emit()


class ElidedLabel(QLabel):
    """A label that shortens its text with … when squeezed, instead of clipping it."""

    def minimumSizeHint(self):
        return QSize(10, super().minimumSizeHint().height())

    def paintEvent(self, e):
        p = QPainter(self)
        p.setPen(self.palette().color(self.foregroundRole()))
        text = self.fontMetrics().elidedText(self.text(), Qt.ElideRight, self.width())
        p.drawText(self.rect(), self.alignment(), text)


class Timeline(QWidget):
    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 6, 8, 8)
        v.setSpacing(6)
        bar = QHBoxLayout()
        bar.setSpacing(2)

        def tb(icon, tip, fn, checkable=False):
            b = QToolButton()
            b.setIcon(icon)
            b.setToolTip(tip)
            b.setCheckable(checkable)
            b.setAutoRaise(True)
            b.clicked.connect(fn)
            bar.addWidget(b)
            return b

        tb(icons.jump(False), 'First frame (Home)', lambda: doc.set_frame(doc.scene.start))
        tb(icons.step(False), 'Previous frame (Left)', lambda: doc.set_frame(doc.frame - 1))
        self.play_btn = tb(icons.play(), 'Play / pause (Space)', lambda: doc.set_playing(not doc.playing))
        self.play_btn.setFixedSize(34, 30)
        self.play_btn.setIconSize(QSize(20, 20))
        self.play_btn.setStyleSheet(f'QToolButton {{ background: {theme.FIELD}; border-radius: 15px; }}'
                                    f'QToolButton:hover {{ background: {theme.FIELD_HI}; }}')
        tb(icons.step(True), 'Next frame (Right)', lambda: doc.set_frame(doc.frame + 1))
        tb(icons.jump(True), 'Last frame (End)', lambda: doc.set_frame(doc.scene.end))
        bar.addSpacing(12)

        def spin(width, tip):
            sp = QSpinBox()
            sp.setRange(-100000, 100000)
            sp.setFixedWidth(width)
            sp.setAlignment(Qt.AlignRight)
            sp.setButtonSymbols(QAbstractSpinBox.NoButtons)
            sp.setKeyboardTracking(False)
            sp.setToolTip(tip)
            from .params import guard_wheel
            guard_wheel(sp)
            return sp
        self.frame_spin = spin(64, 'Current frame (type one to jump there)')
        self.frame_spin.valueChanged.connect(doc.set_frame)
        bar.addWidget(self.frame_spin)
        bar.addSpacing(6)
        self.tc = QLabel()
        self.tc.setFont(theme.mono_font(9))
        self.tc.setObjectName('hint')
        bar.addWidget(self.tc)
        self.info = ElidedLabel()
        self.info.setObjectName('faint')
        self.info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.info.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        bar.addWidget(self.info, 1)
        bar.addSpacing(14)
        self.start_spin = spin(58, 'First frame of the shot')
        self.end_spin = spin(58, 'Last frame of the shot')
        for sp, key, label in ((self.start_spin, 'start', 'Start'), (self.end_spin, 'end', 'End')):
            lab = QLabel(label)
            lab.setObjectName('hint')
            bar.addWidget(lab)
            bar.addSpacing(4)
            sp.valueChanged.connect(lambda v, k=key: self._range(k, v))
            bar.addWidget(sp)
            bar.addSpacing(8)
        self.cache_btn = tb(icons.cache(), 'Simulate and cache the whole frame range (the green bar shows what is cached)',
                            lambda: doc.worker and doc.worker.post('cache_range'))
        self.cache_btn.setText(' Cache all')
        self.cache_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.restart_btn = tb(icons.restart(), 'Restart the simulation from the first frame', lambda: doc.worker and doc.worker.post('restart'))
        self.restart_btn.setText(' Restart')
        self.restart_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        bar.addSpacing(6)
        self.timing_btn = QToolButton()
        self.timing_btn.setObjectName('toggle')
        self.timing_btn.setCheckable(True)
        self.timing_btn.setText('Timing')
        self.timing_btn.setToolTip('When each thing starts and stops, as bars over the shot: drag a bar to move it in time, '
                                   'drag its ends to start or stop it then')
        self.timing_btn.toggled.connect(self._show_timing)
        bar.addWidget(self.timing_btn)
        self.keys_btn = QToolButton()
        self.keys_btn.setObjectName('toggle')
        self.keys_btn.setCheckable(True)
        self.keys_btn.setToolTip('The animation editor: every animated setting, with its keys to move, add, delete and ease')
        self.keys_btn.toggled.connect(self._show_keys)
        bar.addWidget(self.keys_btn)
        v.addLayout(bar)
        self.ruler = Ruler(doc)
        self.ruler.scrubbed.connect(self._scrub)
        v.addWidget(self.ruler)
        from .keys import KeyEditor
        from .timing import TimingEditor
        self.timing = TimingEditor(doc)
        self.timing.hide()
        v.addWidget(self.timing, 1)
        self.keys = KeyEditor(doc)
        self.keys.hide()
        v.addWidget(self.keys, 1)
        self.keys_btn.setChecked(QSettings().value('ui/keys_open', False, type=bool))
        self.timing_btn.setChecked(QSettings().value('ui/timing_open', False, type=bool))
        doc.frameChanged.connect(self.sync)
        doc.sceneReplaced.connect(self.sync)
        doc.paramChanged.connect(lambda *_: self.sync())
        doc.structureChanged.connect(self.sync)
        doc.playingChanged.connect(self._playing)
        doc.selectionChanged.connect(lambda *_: self.ruler.update())
        self.sync()

    def _show_timing(self, on):
        QSettings().setValue('ui/timing_open', bool(on))
        self.timing.setVisible(on)
        if on:
            self.timing.rebuild()

    def _show_keys(self, on):
        QSettings().setValue('ui/keys_open', bool(on))
        self.keys.setVisible(on)
        if on:
            self.keys.rebuild()
            self.keys.setFocus()

    def _scrub(self, f):
        if self.doc.playing:
            self.doc.set_playing(False)
        self.doc.set_frame(f)

    def _range(self, key, v):
        if self.doc.scene.data['render'][key] != v:
            self.doc.set(('render', key), v, merge=False)

    def _playing(self, on):
        self.play_btn.setIcon(icons.pause() if on else icons.play())

    def set_cached(self, frames):
        self.ruler.cached = set(frames)
        self.ruler.update()

    def set_progress(self, frac, frame):
        self.ruler.progress_frame = None if frac >= 1.0 else frame
        self.ruler.update()

    def sync(self, *_):
        sc = self.doc.scene
        for sp, val in ((self.frame_spin, self.doc.frame), (self.start_spin, sc.start), (self.end_spin, sc.end)):
            sp.blockSignals(True)
            sp.setValue(val)
            sp.blockSignals(False)
        self.tc.setText(timecode(self.doc.frame, sc.fps, sc.start))
        n = sc.end - sc.start + 1
        self.info.setText(f'{n} frames · {sc.fps:g} fps · {n / sc.fps:.2f} s')
        from .keys import animated_paths
        na = len(animated_paths(sc))
        self.keys_btn.setText(f'Animation  {na}' if na else 'Animation')
        self.ruler.update()
