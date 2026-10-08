"""File › Import USD scene: pick the camera and the objects to bring in from a USD file."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QListWidget,
                               QListWidgetItem, QSpinBox, QVBoxLayout)

from ..scene.params import FRAME_OFFSET_MAX

MOTION = {'static': 'still', 'rigid': 'moves', 'world': 'deforms or tumbles (baked per frame)'}
KINDS = {'Mesh': 'mesh', 'Cube': 'cube', 'Sphere': 'sphere', 'Cylinder': 'cylinder', 'Capsule': 'capsule', 'Cone': 'cone',
         'PointInstancer': 'instances', 'BasisCurves': 'curves', 'NurbsCurves': 'curves', 'HermiteCurves': 'curves',
         'Points': 'points', 'Volume': 'volume'}


class UsdImportDialog(QDialog):
    def __init__(self, info, parent=None):
        super().__init__(parent)
        self.info = info
        self.setWindowTitle('Import USD scene')
        self.setMinimumWidth(460)
        v = QVBoxLayout(self)
        rng = f'frames {info.frames[0]}–{info.frames[1]}' if info.frames else 'no frame range'
        head = QLabel(f'<b>{Path(info.file).name}</b><br>{rng} at {info.fps:g} fps · {info.meters_per_unit:g} m per unit · '
                      f'{info.up} up')
        head.setWordWrap(True)
        v.addWidget(head)
        form = QFormLayout()
        self.camera = QComboBox()
        self.camera.addItem('Keep the current camera', None)
        for c in info.cameras():
            self.camera.addItem(f'{c.path} ({c.motion})', c.path)
        if info.cameras():
            self.camera.setCurrentIndex(1)
        form.addRow('Camera', self.camera)
        self.as_ = QComboBox()
        self.as_.addItem('Colliders (objects in the shot)', 'collider')
        self.as_.addItem('Emitters (objects that burn)', 'emitter')
        form.addRow('Meshes become', self.as_)
        self.volumes = QComboBox()
        self.volumes.addItem('Smoke (the volume fills the box with its smoke)', 'smoke')
        self.volumes.addItem('Solid objects (like the meshes)', 'solid')
        self.volumes.setEnabled(any(m.prim_type == 'Volume' or m.particles for m in info.meshes()))
        self.volumes.setToolTip('Volumes, and points that are a particle system (they move, or carry velocities), come in '
                                'as smoke: in a liquid scene, as water poured once. Still points without velocities '
                                '(gravel, debris) are always solid balls.')
        form.addRow('Volumes and particles become', self.volumes)
        v.addLayout(form)
        self.meshes = QListWidget()
        for m in info.meshes():
            if m.particles:   # (a source, splatted: no triangles of its own unless made solid)
                it = QListWidgetItem(f'{m.path}  ·  particles  ·  {"move" if m.motion == "world" else "still"}')
            else:
                it = QListWidgetItem(f'{m.path}  ·  {KINDS.get(m.prim_type, "mesh")}  ·  {MOTION.get(m.motion, m.motion)}  ·  '
                                     f'{m.triangles:,} triangles')
            it.setData(Qt.UserRole, m.path)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)
            self.meshes.addItem(it)
        v.addWidget(QLabel('Objects (up to 16 per kind):'))
        v.addWidget(self.meshes, 1)
        lights = info.lights()
        names = [l.prim_type.replace('Light', '').lower() for l in lights]
        self.lights = QCheckBox('Lights: the first distant light becomes the key light, the first dome light the ambient'
                                + (f' ({", ".join(names)})' if names else ''))
        self.lights.setChecked(any(l.prim_type in ('DistantLight', 'DomeLight') for l in lights))
        self.lights.setEnabled(bool(lights))
        v.addWidget(self.lights)
        self.holdout = QCheckBox('Colliders hide the fire behind them (they are in the shot)')
        self.holdout.setChecked(True)
        self.burnable = QCheckBox('Colliders are burnable')
        self.match_range = QCheckBox('Use the USD file’s frame range and frame rate')
        self.match_range.setChecked(bool(info.frames))
        self.match_range.setEnabled(bool(info.frames))
        for w in (self.holdout, self.burnable, self.match_range):
            v.addWidget(w)
        self.offset = QSpinBox()
        self.offset.setRange(-int(FRAME_OFFSET_MAX), int(FRAME_OFFSET_MAX))
        self.offset.setToolTip('USD frame F becomes frame F minus this: 1000 brings a 1001–1100 shot in as frames 1–100. '
                               'The camera, objects and lights stay in step; objects that change read their own USD '
                               'frame through their Mesh frame offset (up to 100 000 frames either way).')
        off = QFormLayout()
        off.addRow('Frame offset', self.offset)
        v.addLayout(off)
        note = QLabel('Objects land where they are in the USD scene, relative to the fire’s position and rotation '
                      '(Camera › Fire position). Alembic files: convert them to USD first.')
        note.setWordWrap(True)
        note.setStyleSheet('color: gray')
        v.addWidget(note)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText('Import')
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def volume_choice(self):
        """'smoke' or 'solid': what the USD file's Volume prims (and particle systems) become."""
        return self.volumes.currentData()

    def timing(self):
        """(frame offset, take the USD's frame range and rate) for io.usd.import_usd."""
        return int(self.offset.value()), bool(self.info.frames and self.match_range.isChecked())

    def choice(self):
        cam = self.camera.currentData()
        meshes = [self.meshes.item(i).data(Qt.UserRole) for i in range(self.meshes.count())
                  if self.meshes.item(i).checkState() == Qt.Checked]
        return (([cam] if cam else []), meshes, self.as_.currentData(), self.holdout.isChecked(), self.burnable.isChecked(),
                self.lights.isChecked())
