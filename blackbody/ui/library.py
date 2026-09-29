"""Preset library: built-in fires with rendered thumbnails, plus the user's own saved presets."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSize, QStandardPaths, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QHBoxLayout, QInputDialog, QLabel, QListWidget,
                               QListWidgetItem, QMenu, QMessageBox, QPushButton, QVBoxLayout, QWidget)

from ..scene import PROJECT_EXT, Scene, presets
from . import icons, theme

ASSETS = Path(__file__).resolve().parents[1] / 'assets' / 'presets'
THUMB = QSize(176, 99)


def user_preset_dir():
    base = Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)) / 'presets'
    base.mkdir(parents=True, exist_ok=True)
    return base


def _placeholder():
    pm = QPixmap(THUMB)
    pm.fill(QColor('#161618'))
    p = QPainter(pm)
    icons.flame().paint(p, pm.rect().adjusted(60, 20, -60, -20))
    p.end()
    return pm


class Library(QWidget):
    presetChosen = Signal(str, bool)       # built-in name, keep shot
    userPresetChosen = Signal(str, bool)   # file path, keep shot

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)
        hint = QLabel('Double-click a fire or a liquid to use it. Your footage, camera and output settings stay.')
        hint.setObjectName('hint')
        hint.setWordWrap(True)
        v.addWidget(hint)
        self.list = QListWidget()
        self.list.setViewMode(QListWidget.IconMode)
        self.list.setIconSize(THUMB)
        self.list.setGridSize(QSize(THUMB.width() + 12, THUMB.height() + 44))
        self.list.setResizeMode(QListWidget.Adjust)
        self.list.setMovement(QListWidget.Static)
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.setWordWrap(True)
        self.list.setSpacing(4)
        self.list.itemDoubleClicked.connect(self._use)
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._menu)
        v.addWidget(self.list, 1)
        row = QHBoxLayout()
        self.keep = QCheckBox('Keep my shot')
        self.keep.setChecked(True)
        self.keep.setToolTip('Keep footage, camera placement, frame range and output settings when loading a fire.')
        row.addWidget(self.keep)
        row.addStretch(1)
        save = QPushButton('Save as preset…')
        save.clicked.connect(self.save_current)
        row.addWidget(save)
        v.addLayout(row)
        self.populate()

    def populate(self):
        self.list.clear()
        ph = _placeholder()
        for key in presets.ORDER:
            p = presets.PRESETS[key]
            f = ASSETS / f'{key}.png'
            pm = QPixmap(str(f)).scaled(THUMB, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation) if f.exists() else ph
            it = QListWidgetItem(QIcon(pm), f'{p["name"]}\n{p["size"]}')
            it.setToolTip(f'<b>{p["name"]}</b> · {p["category"]} · {p["size"]}<br>{p["blurb"]}')
            it.setData(Qt.UserRole, ('builtin', key))
            it.setSizeHint(QSize(THUMB.width() + 8, THUMB.height() + 40))
            self.list.addItem(it)
        for f in sorted(user_preset_dir().glob('*' + PROJECT_EXT)):
            thumb = f.with_suffix('.png')
            pm = QPixmap(str(thumb)).scaled(THUMB, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation) if thumb.exists() else ph
            it = QListWidgetItem(QIcon(pm), f'{f.stem}\nmy preset')
            it.setToolTip(f'<b>{f.stem}</b> · saved preset')
            it.setData(Qt.UserRole, ('user', str(f)))
            self.list.addItem(it)

    def _use(self, it):
        kind, ref = it.data(Qt.UserRole)
        if kind == 'builtin':
            self.presetChosen.emit(ref, self.keep.isChecked())
        else:
            self.userPresetChosen.emit(ref, self.keep.isChecked())

    def _menu(self, pos):
        it = self.list.itemAt(pos)
        if it is None:
            return
        kind, ref = it.data(Qt.UserRole)
        m = QMenu(self)
        m.addAction('Use this fire', lambda: self._use(it))
        if kind == 'user':
            m.addAction('Delete preset', lambda: self._delete(ref))
        m.exec(self.list.viewport().mapToGlobal(pos))

    def _delete(self, path):
        p = Path(path)
        if QMessageBox.question(self, 'Delete preset', f'Delete the preset "{p.stem}"?') == QMessageBox.Yes:
            p.unlink(missing_ok=True)
            p.with_suffix('.png').unlink(missing_ok=True)
            self.populate()

    def save_current(self, thumbnail: QImage | None = None):
        name, ok = QInputDialog.getText(self, 'Save as preset', 'Preset name:', text=self.doc.scene.name)
        if not ok or not name.strip():
            return
        safe = ''.join(c for c in name.strip() if c not in '\\/:*?"<>|')
        path = user_preset_dir() / f'{safe}{PROJECT_EXT}'
        sc = self.doc.scene.copy()
        sc.footage = None
        sc.track = None
        sc.name = safe
        sc.save(path)
        img = self.window().viewport.image if hasattr(self.window(), 'viewport') else None
        if img is not None:
            img.scaled(THUMB * 2, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation).save(str(path.with_suffix('.png')))
        self.populate()
