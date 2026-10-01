"""The dialog for a Text block: type the words, pick a font and a size, and see the letters before they go in."""
from __future__ import annotations

from PySide6.QtCore import QRectF, QSettings, Qt
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen, QTransform
from PySide6.QtWidgets import (QDialog, QDoubleSpinBox, QFontComboBox, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from . import textmesh, theme
from .params import guard_wheel


class LetterPreview(QWidget):
    """The letters as they will be made, standing on the ground line."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.path = None
        self.look = 'fire'
        self.setMinimumHeight(130)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_path(self, path):
        self.path = path
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(QColor(theme.LINE), 1))
        p.setBrush(QColor(theme.VIEWER))
        p.drawRoundedRect(r, 8, 8)
        if self.path is None or self.path.isEmpty():
            p.setPen(QColor(theme.FAINT))
            p.drawText(r, Qt.AlignCenter, 'Type some words')
            return
        b = self.path.boundingRect()
        box = r.adjusted(18, 16, -18, -22)
        s = min(box.width() / max(b.width(), 1e-6), box.height() / max(b.height(), 1e-6))
        t = QTransform()
        t.translate(box.center().x() - b.center().x() * s, box.bottom() - b.bottom() * s)
        t.scale(s, s)
        shape = t.map(self.path)
        ground = box.bottom() + 0.5
        p.setPen(QPen(QColor(theme.LINE_HI), 1, Qt.DashLine))
        p.drawLine(int(r.left() + 10), int(ground), int(r.right() - 10), int(ground))
        g = QLinearGradient(0, shape.boundingRect().bottom(), 0, shape.boundingRect().top())
        if self.look == 'fire':
            g.setColorAt(0.0, QColor('#fff1c4'))
            g.setColorAt(0.45, QColor('#ffad3b'))
            g.setColorAt(1.0, QColor('#d9481c'))
        elif self.look == 'water':
            g.setColorAt(0.0, QColor('#2f7fd0'))
            g.setColorAt(1.0, QColor('#a9dcff'))
        else:
            g.setColorAt(0.0, QColor('#9aa0a8'))
            g.setColorAt(1.0, QColor('#d8dce2'))
        p.setPen(Qt.NoPen)
        p.setBrush(g)
        p.drawPath(shape)


class TextDialog(QDialog):
    """Asks for the words and how they look. `width` is the scene's width (m): new text is sized to fit it."""

    def __init__(self, parent=None, spec=None, width=2.0, look='fire', title='Text'):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(560)
        self.width_m = max(float(width), 0.05)
        self._auto = spec is None   # sizes follow the text until they are set by hand
        self._quiet = False
        st = QSettings()
        spec = spec or textmesh.spec_of(st.value('ui/text_last', 'FIRE', type=str),
                                         st.value('ui/text_font', '', type=str) or textmesh.default_family(),
                                         st.value('ui/text_bold', False, type=bool), False, 0.4, 0.1)

        v = QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 14)
        v.setSpacing(10)
        head = QLabel('Type the words. Each line is centred over the one below; the letters stand on the ground.')
        head.setWordWrap(True)
        head.setObjectName('faint')
        v.addWidget(head)
        self.text = QPlainTextEdit(spec['text'])
        self.text.setFixedHeight(64)
        f = QFont(self.text.font())
        f.setPointSizeF(f.pointSizeF() * 1.25)
        self.text.setFont(f)
        v.addWidget(self.text)

        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel('Font'))
        self.font = QFontComboBox()
        self.font.setCurrentFont(QFont(spec['font']))
        guard_wheel(self.font)
        row.addWidget(self.font, 1)
        self.bold = QPushButton('B')
        self.italic = QPushButton('I')
        for b, on, tip in ((self.bold, spec['bold'], 'Bold'), (self.italic, spec['italic'], 'Italic')):
            b.setCheckable(True)
            b.setChecked(bool(on))
            b.setFixedWidth(34)
            b.setToolTip(tip)
            b.setObjectName('toggle')
            b.setStyleSheet(f'QPushButton {{ padding: 0px; }}'   # room for the letter in a narrow button
                            f'QPushButton:checked {{ background: {theme.ACCENT_FILL}; border: 1px solid {theme.ACCENT}; '
                            f'color: {theme.ACCENT_HI}; }}')
            row.addWidget(b)
        fb = QFont(self.bold.font())
        fb.setBold(True)
        self.bold.setFont(fb)
        fi = QFont(self.italic.font())
        fi.setItalic(True)
        self.italic.setFont(fi)
        v.addLayout(row)

        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel('Letter height'))
        self.height = self._spin(spec['height'], 0.01, 50.0, 'How tall a capital letter is')
        row.addWidget(self.height)
        row.addSpacing(10)
        row.addWidget(QLabel('Depth'))
        self.depth = self._spin(spec['depth'], 0.002, 20.0, 'How thick the letters are, front to back')
        row.addWidget(self.depth)
        row.addStretch(1)
        self.size_label = QLabel()
        self.size_label.setObjectName('faint')
        row.addWidget(self.size_label)
        v.addLayout(row)

        self.preview = LetterPreview()
        self.preview.look = look
        v.addWidget(self.preview, 1)

        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton('Cancel')
        cancel.clicked.connect(self.reject)
        self.ok = QPushButton('Change' if title.startswith('Edit') else 'Add')
        self.ok.setObjectName('primary')
        self.ok.setDefault(True)
        self.ok.clicked.connect(self.accept)
        row.addWidget(cancel)
        row.addWidget(self.ok)
        v.addLayout(row)

        self.text.textChanged.connect(self._changed)
        self.font.currentFontChanged.connect(lambda *_: self._changed())
        self.bold.toggled.connect(lambda *_: self._changed())
        self.italic.toggled.connect(lambda *_: self._changed())
        self.height.valueChanged.connect(lambda *_: self._sized())
        self.depth.valueChanged.connect(lambda *_: self._sized())
        self._changed()
        self.text.setFocus()
        self.text.selectAll()

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

    def words(self):
        return self.text.toPlainText().strip('\n')

    def _changed(self):
        text = self.words()
        path, cap = textmesh.outline(text, self.font.currentFont().family(), self.bold.isChecked(), self.italic.isChecked()) \
            if text.strip() else (None, 1.0)
        self.preview.set_path(path)
        self._cap = cap
        self._wide = (path.boundingRect().width() / cap) if path is not None and not path.isEmpty() else 0.0
        self.ok.setEnabled(bool(text.strip()))
        if self._auto and self._wide > 0:
            # as tall as a fifth of the scene, or less so the whole text takes no more than 85% of its width
            h = min(0.2 * self.width_m, 0.85 * self.width_m / self._wide)
            h = float(f'{h:.2g}')
            self._quiet = True
            self.height.setValue(h)
            self.depth.setValue(float(f'{max(0.25 * h, 0.005):.2g}'))
            self._quiet = False
        self._label()

    def _sized(self):
        if not self._quiet:
            self._auto = False
        self._label()

    def _label(self):
        if self._wide <= 0:
            self.size_label.setText('')
            return
        h = self.height.value()
        self.size_label.setText(f'{self._wide * h:.2f} m wide')

    def spec(self):
        return textmesh.spec_of(self.words(), self.font.currentFont().family(), self.bold.isChecked(), self.italic.isChecked(),
                                self.height.value(), self.depth.value())

    def accept(self):
        st = QSettings()
        st.setValue('ui/text_last', self.words())
        st.setValue('ui/text_font', self.font.currentFont().family())
        st.setValue('ui/text_bold', self.bold.isChecked())
        super().accept()


def ask(parent, spec=None, width=2.0, look='fire', edit=False):
    """The text spec the user asked for, or None. `look` colours the preview: 'fire', 'water' or 'solid'."""
    d = TextDialog(parent, spec, width, look, 'Edit text' if edit else 'Text')
    return d.spec() if d.exec() == QDialog.Accepted else None
