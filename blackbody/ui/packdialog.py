"""Pack project: what files the project uses, how big they are, and copying them into a folder beside it."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtWidgets import QCheckBox, QDialog, QHBoxLayout, QLabel, QMessageBox, QProgressDialog, QPushButton, QVBoxLayout

from ..scene import pack
from . import theme


class PackDialog(QDialog):
    def __init__(self, parent, refs, project):
        super().__init__(parent)
        self.setWindowTitle('Pack project')
        self.setMinimumWidth(520)
        self.refs = refs
        v = QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 14)
        v.setSpacing(8)
        head = QLabel(f'Copies every file <b>{Path(project).name}</b> uses into <b>{pack.folder_for(project).name}</b>, beside it, '
                      'and points the project at the copies. Move or send the project and that folder together and it '
                      'opens anywhere.')
        head.setWordWrap(True)
        v.addWidget(head)
        groups = OrderedDict()
        for r in refs:
            if r.what != 'footage':
                groups.setdefault(pack.LABELS[r.what], []).append(r)
        for label, rs in groups.items():
            files = sum(len(r.files) for r in rs)
            names = ', '.join(Path(r.real.split('#')[0]).name for r in rs[:4]) + ('…' if len(rs) > 4 else '')
            lab = QLabel(f'<b>{label}</b>: {len(rs)} ({files} file{"s" if files != 1 else ""}, {pack.human(sum(r.size for r in rs))})'
                         f'<br><span style="color:{theme.FAINT}">{names}</span>')
            lab.setWordWrap(True)
            v.addWidget(lab)
        foot = [r for r in refs if r.what == 'footage']
        self.footage = QCheckBox()
        if foot:
            n = sum(len(r.files) for r in foot)
            self.footage.setText(f'Footage too: {Path(foot[0].real).name} ({n} file{"s" if n != 1 else ""}, '
                                 f'{pack.human(sum(r.size for r in foot))})')
            self.footage.setChecked(sum(r.size for r in foot) < 4 << 30)
            self.footage.setToolTip('Footage can be large. Without it the project still finds the footage where it is now.')
            v.addWidget(self.footage)
        else:
            self.footage.hide()
        if not groups and not foot:
            v.addWidget(QLabel('This project uses no files of its own (only built-in ones): there is nothing to pack.'))
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton('Cancel')
        cancel.clicked.connect(self.reject)
        ok = QPushButton('Pack')
        ok.setObjectName('primary')
        ok.setDefault(True)
        ok.setEnabled(bool(groups or foot))
        ok.clicked.connect(self.accept)
        row.addWidget(cancel)
        row.addWidget(ok)
        v.addLayout(row)

    def chosen(self):
        return [r for r in self.refs if r.what != 'footage' or self.footage.isChecked()]


def pack_project(win, ask=True):
    """File › Pack project. Saves first if the project has never been saved. Returns the number of files packed."""
    doc = win.doc
    if not doc.shot.path:
        QMessageBox.information(win, 'Pack project', 'Save the project first: its files go in a folder beside it.')
        if not win.save_as():
            return 0
    refs = pack.gather(doc.shot)
    if ask:
        d = PackDialog(win, refs, doc.shot.path)
        if d.exec() != QDialog.Accepted:
            return 0
        refs = d.chosen()
    if not refs:
        return 0
    bar = QProgressDialog('Copying files…', 'Stop', 0, 1000, win)
    bar.setWindowTitle('Pack project')
    bar.setWindowModality(Qt.WindowModal)
    bar.setMinimumDuration(300)

    def progress(frac, name):
        bar.setValue(int(frac * 1000))
        bar.setLabelText(f'Copying {name}…')
        QCoreApplication.processEvents()
        return not bar.wasCanceled()
    try:
        mapping = pack.copy(refs, doc.shot.path, progress)
    except InterruptedError:
        bar.close()
        win.msg.setText('Packing stopped: the project still points at the files where they were.')
        return 0
    except OSError as ex:
        bar.close()
        QMessageBox.warning(win, 'Pack project', f'Could not copy the files:\n{ex}')
        return 0
    bar.setValue(1000)
    bar.close()
    doc.edit('Pack project', lambda s: pack.apply(doc.shot, mapping), structure=True)   # every layer of the shot
    win.save()
    n = sum(len(r.files) for r in refs)
    win.msg.setText(f'Packed: {n} file{"s" if n != 1 else ""} in “{pack.folder_for(doc.shot.path).name}”, beside the project, '
                    'which now uses them there. Move the two together.')
    return n
