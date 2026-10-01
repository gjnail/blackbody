"""Save as a block: name what you built, say what it is, and see what goes in it."""
from __future__ import annotations

from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout

from ..scene import blocks
from . import icons, theme


class SaveBlockDialog(QDialog):
    def __init__(self, parent, scene, sel):
        super().__init__(parent)
        self.setWindowTitle('Save as a block')
        self.setMinimumWidth(460)
        self.sel = blocks.with_attached(scene, sel)
        v = QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 14)
        v.setSpacing(8)
        head = QLabel('It goes under <b>Yours</b> in Create, to add to any scene like the built-in blocks. '
                      'The block is one file you can share: meshes and all.')
        head.setWordWrap(True)
        head.setObjectName('faint')
        v.addWidget(head)
        v.addWidget(QLabel('Name'))
        first = blocks._items(scene, self.sel[0][0])[self.sel[0][1]]['name'] if self.sel else 'My block'
        self.name = QLineEdit(first.strip('“”'))
        v.addWidget(self.name)
        v.addWidget(QLabel('What it is (optional)'))
        self.tip = QLineEdit()
        self.tip.setPlaceholderText('A burning sign for the shop front')
        v.addWidget(self.tip)
        names = [blocks._items(scene, k)[i]['name'] for k, i in self.sel]
        inc = QLabel(f'<b>In it</b> ({len(names)}): ' + ', '.join(names[:12]) + ('…' if len(names) > 12 else '')
                     + ('<br><span style="color:%s">Things attached to what you picked come too.</span>' % theme.FAINT
                        if len(self.sel) > len(sel) else ''))
        inc.setWordWrap(True)
        v.addWidget(inc)
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton('Cancel')
        cancel.clicked.connect(self.reject)
        self.ok = QPushButton('Save block')
        self.ok.setIcon(icons.glyph_icon('save', '#ffffff', 14))
        self.ok.setObjectName('primary')
        self.ok.setDefault(True)
        self.ok.clicked.connect(self.accept)
        self.name.textChanged.connect(lambda t: self.ok.setEnabled(bool(t.strip())))
        row.addWidget(cancel)
        row.addWidget(self.ok)
        v.addLayout(row)
        self.name.selectAll()
        self.name.setFocus()


def ask(parent, scene, sel):
    """(name, what it is) for a new block, or None."""
    d = SaveBlockDialog(parent, scene, sel)
    if d.exec() != QDialog.Accepted:
        return None
    return d.name.text().strip(), d.tip.text().strip()
