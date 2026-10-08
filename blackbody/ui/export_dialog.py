"""Render / export dialog and the render progress window."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from PySide6.QtCore import QSettings, QStandardPaths, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)

from ..io.video import available_profiles
from ..render.job import DEEP_SAMPLES, Output, deep_layers, has_fabric, output_notes
from . import theme

COMP_PROFILES = ['prores422hq', 'h264', 'h265', 'dnxhr_hq', 'prores4444']
VDB_GRIDS = {'fire': 'density · temperature · flame · fuel · vel', 'both': 'density · temperature · flame · fuel · vel, '
             'and the liquid beside it', 'liquid': 'density (the surface) · vel · spray · foam · bubbles',
             'cloud': 'density · cloud_water · cloud_ice · rain · snow · hail · vel'}
VDB_TIPS = {'fire': 'density, temperature, flame, fuel and vel grids for Blender, Houdini, Maya, Unreal…',
            'both': 'The fire\'s density, temperature, flame, fuel and vel grids, and the liquid beside it as '
                    'name.liquid.####.vdb, for Blender, Houdini, Maya, Unreal…',
            'liquid': 'The liquid as density (its 0.5 level is the surface), vel, and the spray, foam and bubbles, for '
                      'Blender, Houdini, Maya, Unreal…',
            'cloud': 'The sky\'s water in g/m³ (density is the cloud itself, its water and ice; then cloud_water, '
                     'cloud_ice, rain, snow and hail) and vel, in the scene\'s metres, for Blender, Houdini, Maya, Unreal…'}


def default_folder():
    return QSettings().value('export/folder', str(Path(QStandardPaths.writableLocation(QStandardPaths.MoviesLocation) or Path.home()) / 'Blackbody'))


class ExportDialog(QDialog):
    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        sc = getattr(doc, 'shot', doc.scene)
        self.setWindowTitle('Render')
        self.setMinimumWidth(700)
        s = QSettings()
        v = QVBoxLayout(self)
        v.setSpacing(10)
        profiles = available_profiles()

        # outputs --------------------------------------------------------------------------------
        box = QGroupBox('Outputs')
        g = QGridLayout(box)
        g.setColumnStretch(1, 1)
        r = 0

        def row(check, *widgets):
            nonlocal r
            g.addWidget(check, r, 0)
            h = QHBoxLayout()
            h.setSpacing(6)
            for w in widgets:
                h.addWidget(w)
            h.addStretch(1)
            g.addLayout(h, r, 1)
            r += 1

        self.exr = QCheckBox('Fire element · OpenEXR sequence')
        self.exr.setToolTip('Scene-linear, premultiplied RGBA with emission, glow, heat and depth layers. For Nuke, After Effects, Fusion, Resolve.')
        self.exr_comp = QComboBox()
        for k, label in (('zip', 'ZIP (lossless)'), ('piz', 'PIZ (lossless)'), ('dwaa', 'DWAA (small)'), ('zips', 'ZIPS'), ('none', 'None')):
            self.exr_comp.addItem(label, k)
        self.exr_float = QCheckBox('32-bit float')
        self.exr_layers = QCheckBox('AOV layers')
        self.exr_layers.setChecked(True)
        self.exr_layers.setToolTip('emission, glow, heat (for distortion) and depth layers in the same file, plus light (fire light on the ground and colliders), holdout and scorch mattes')
        row(self.exr, self.exr_comp, self.exr_float, self.exr_layers)

        self.deep = QCheckBox('Fire element · deep EXR sequence')
        self.deep_n = QComboBox()
        self.deep_n.addItem('8 samples per pixel', 8)
        self.deep_n.addItem('16 samples per pixel', 16)
        self.deep_n.setToolTip('16 keeps thick, layered smoke in more slices, for objects merged deep inside it, at twice '
                               'the memory and file size.')
        row(self.deep, self.deep_n, QLabel('RGBA · Z · ZBack'))

        self.png = QCheckBox('Fire element · PNG sequence')
        self.png.setToolTip('RGBA with alpha, for editors and motion graphics.')
        self.png_bits = QComboBox()
        self.png_bits.addItem('16-bit', 16)
        self.png_bits.addItem('8-bit', 8)
        self.png_alpha = QComboBox()
        self.png_alpha.addItem('Premultiplied alpha', 'premultiplied')
        self.png_alpha.addItem('Straight alpha', 'straight')
        row(self.png, self.png_bits, self.png_alpha)

        self.mov = QCheckBox('Fire element · ProRes 4444 with alpha')
        self.mov.setToolTip('One file with alpha. Premiere, Final Cut, Resolve and After Effects read it directly.')
        self.mov_alpha = QComboBox()
        self.mov_alpha.addItem('Premultiplied alpha', 'premultiplied')
        self.mov_alpha.addItem('Straight alpha', 'straight')
        self.mov.setEnabled('prores4444' in profiles)
        row(self.mov, self.mov_alpha)

        self.comp = QCheckBox('Composite over the footage')
        self.comp.setToolTip('The finished shot: fire, haze, glow and light cast baked into your footage.')
        self.comp_profile = QComboBox()
        for k in COMP_PROFILES:
            if k in profiles:
                self.comp_profile.addItem(profiles[k].label, k)
        self.comp_audio = QCheckBox('Keep audio')
        self.comp_audio.setChecked(True)
        row(self.comp, self.comp_profile, self.comp_audio)

        self.comp_exr = QCheckBox('Composite · EXR sequence')
        self.comp_exr.setToolTip('The finished shot in scene-linear OpenEXR, for grading and finishing. With Lume\'s Light '
                                 'passes on, it also holds the set\'s light split by where it comes from (light_key, light_sky, '
                                 'light_fire, light_lamps), to turn each light up or down in the comp.')
        row(self.comp_exr, QLabel('RGB · Lume light passes'))

        self.vdb = QCheckBox('Volume · OpenVDB sequence')
        self.vdb.setToolTip(VDB_TIPS.get(sc.kind, VDB_TIPS['fire']))
        row(self.vdb, QLabel(VDB_GRIDS.get(sc.kind, VDB_GRIDS['fire'])))

        self.mesh = QCheckBox('Liquid surface · USD')
        self.mesh.setToolTip('The water surface as a polygon mesh for lighting and rendering in Blender, Houdini, Maya or '
                             'Omniverse: one USD file with the mesh (points, normals, velocities for motion blur) on '
                             'every frame, and the spray, foam and bubbles as point clouds.')
        row(self.mesh, QLabel('mesh · velocities · spray · foam · bubbles'))

        self.fabric = QCheckBox('Fabric · USD')
        self.fabric.setToolTip('The fabric as polygon meshes for Blender, Houdini, Maya or Omniverse: one USD file with each '
                               'fabric\'s mesh (points, normals, velocities for motion blur) on every frame, its burnt-through '
                               'holes left out, its weave as UVs and a burn value on every point for shading the char.')
        row(self.fabric, QLabel('mesh · velocities · UVs · burn'))
        v.addWidget(box)

        # where -----------------------------------------------------------------------------------------
        where = QGroupBox('Where')
        f = QFormLayout(where)
        folder_row = QHBoxLayout()
        self.folder = QLineEdit(default_folder())
        browse = QPushButton('Browse…')
        browse.clicked.connect(self._browse)
        folder_row.addWidget(self.folder, 1)
        folder_row.addWidget(browse)
        f.addRow('Folder', folder_row)
        self.name = QLineEdit((sc.name or 'fire').replace(' ', '_').lower())
        f.addRow('Name', self.name)
        self.preview_paths = QLabel()
        self.preview_paths.setObjectName('hint')
        self.preview_paths.setWordWrap(True)
        f.addRow('', self.preview_paths)
        v.addWidget(where)

        # frames & quality ----------------------------------------------------------------------------------
        q = QGroupBox('Frames and quality')
        f2 = QFormLayout(q)
        rng = QHBoxLayout()
        self.first = QSpinBox()
        self.last = QSpinBox()
        for sp in (self.first, self.last):
            sp.setRange(-100000, 100000)
        self.first.setValue(sc.start)
        self.last.setValue(sc.end)
        rng.addWidget(self.first)
        rng.addWidget(QLabel('to'))
        rng.addWidget(self.last)
        rng.addStretch(1)
        f2.addRow('Frames', rng)
        size = QHBoxLayout()
        self.w = QSpinBox()
        self.h = QSpinBox()
        for sp in (self.w, self.h):
            sp.setRange(64, 8192)
        W, H = sc.output_size()
        self.w.setValue(W)
        self.h.setValue(H)
        size.addWidget(self.w)
        size.addWidget(QLabel('×'))
        size.addWidget(self.h)
        size.addStretch(1)
        f2.addRow('Size', size)
        self.samples = QSpinBox()
        self.samples.setRange(1, 64)
        self.samples.setValue(int(sc.data['render']['aa_samples']))
        f2.addRow('Samples per pixel', self.samples)
        self.res_scale = QDoubleSpinBox()
        self.res_scale.setRange(0.25, 3.0)
        self.res_scale.setSingleStep(0.25)
        self.res_scale.setValue(float(sc.data['render']['final_scale']))
        self.res_scale.setSuffix(' ×')
        dims, hcell, _ = sc.sim_layout(final=True)
        self.res_label = QLabel()
        self.res_label.setObjectName('hint')
        rs = QHBoxLayout()
        rs.addWidget(self.res_scale)
        rs.addWidget(self.res_label)
        rs.addStretch(1)
        f2.addRow('Simulation resolution', rs)
        self.mblur = QCheckBox('Motion blur')
        self.mblur.setChecked(bool(sc.data['render']['motion_blur']))
        f2.addRow('', self.mblur)
        v.addWidget(q)
        self.res_scale.valueChanged.connect(self._update_labels)

        # buttons ----------------------------------------------------------------------------------------------
        b = QHBoxLayout()
        b.addStretch(1)
        cancel = QPushButton('Cancel')
        cancel.clicked.connect(self.reject)
        self.go = QPushButton('Render')
        self.go.setObjectName('primary')
        self.go.setDefault(True)
        self.go.clicked.connect(self._accept)
        b.addWidget(cancel)
        b.addWidget(self.go)
        v.addLayout(b)

        # restore choices
        self.exr.setChecked(s.value('export/exr', True, type=bool))
        self.png.setChecked(s.value('export/png', False, type=bool))
        self.mov.setChecked(s.value('export/mov', False, type=bool) and self.mov.isEnabled())
        self.comp.setChecked(s.value('export/comp', bool(sc.footage), type=bool))
        self.comp_exr.setChecked(s.value('export/comp_exr', False, type=bool))
        self.vdb.setChecked(s.value('export/vdb', False, type=bool))
        deep = deep_layers(sc)
        self.deep.setChecked(s.value('export/deep', False, type=bool) and bool(deep))
        self.deep.setEnabled(bool(deep))
        self.deep_n.setEnabled(bool(deep))
        self.deep_n.setCurrentIndex(max(0, self.deep_n.findData(s.value('export/deep_samples', DEEP_SAMPLES, type=int))))
        if deep and all(lay.kind == 'liquid' for _, lay in deep):
            self.deep.setText('Liquid element · deep EXR sequence')
        self.deep.setToolTip(self._deep_tip(sc, deep))
        self.mesh.setChecked(s.value('export/mesh', False, type=bool) and sc.kind in ('liquid', 'both'))
        self.mesh.setEnabled(sc.kind in ('liquid', 'both'))
        fabric = has_fabric(sc)
        self.fabric.setChecked(s.value('export/fabric', False, type=bool) and fabric)
        self.fabric.setEnabled(fabric)
        if not fabric:
            self.fabric.setToolTip(self.fabric.toolTip() + ' (There is no fabric in this scene.)')
        i = self.comp_profile.findData(s.value('export/comp_profile', 'prores422hq'))
        self.comp_profile.setCurrentIndex(max(0, i))
        for w in (self.exr, self.deep, self.png, self.mov, self.comp, self.comp_exr, self.vdb, self.mesh, self.fabric,
                  self.folder, self.name):
            (w.toggled if isinstance(w, QCheckBox) else w.textChanged).connect(self._update_labels)
        self.comp_profile.currentIndexChanged.connect(self._update_labels)
        self.deep_n.currentIndexChanged.connect(self._update_labels)
        self._update_labels()

    @staticmethod
    def _deep_tip(sc, deep):
        tip = ('Deep OpenEXR for deep compositing (Nuke DeepRead): 8 or 16 samples per pixel, each with its colour, alpha '
               'and depth, so the fire and smoke merge correctly with other deep renders. The embers are in them as '
               'light at their own depth.')
        if not deep:
            return tip + (' They are made for fire scenes and liquid scenes (and those layers of a shot), not fire-and-liquid '
                          'or sky scenes.')
        if any(u != 'base' for u, _ in (sc.layer_order() or [])):
            return tip + (' In a shot with layers, every fire and liquid layer\'s samples go into the one file, each at its '
                          'own depth.')
        return tip

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, 'Render folder', self.folder.text())
        if d:
            self.folder.setText(d)

    def _update_labels(self, *_):
        sc = getattr(self.doc, 'shot', self.doc.scene).copy()
        sc.data['render']['final_scale'] = self.res_scale.value()
        dims, h, _ = sc.sim_layout(final=True)
        mem = dims[0] * dims[1] * dims[2] * 88 / 1e9
        self.res_label.setText(f'{dims[0]}×{dims[1]}×{dims[2]} voxels · {h * 1000:.1f} mm · about {mem:.1f} GB of GPU memory')
        outs = self.outputs()
        paths = [o.path for o in outs]
        notes = output_notes(getattr(self.doc, 'shot', self.doc.scene), outs)
        self.preview_paths.setText('\n'.join(paths + notes) if paths else 'Choose at least one output.')
        self.go.setEnabled(bool(paths))

    def outputs(self):
        folder = Path(self.folder.text() or '.')
        name = self.name.text().strip() or 'fire'
        out = []
        if self.exr.isChecked():
            layers = ('emission', 'glow', 'heat', 'depth', 'surface') if self.exr_layers.isChecked() else ()
            out.append(Output('exr', str(folder / name / f'{name}.####.exr'), 'element', layers=layers,
                              compression=self.exr_comp.currentData(), half=not self.exr_float.isChecked()))
        if self.deep.isChecked() and self.deep.isEnabled():
            out.append(Output('deep', str(folder / f'{name}_deep' / f'{name}.deep.####.exr'), 'element',
                              deep_samples=self.deep_n.currentData()))
        if self.png.isChecked():
            out.append(Output('png', str(folder / f'{name}_png' / f'{name}.####.png'), 'element', bits=self.png_bits.currentData(),
                              alpha_mode=self.png_alpha.currentData()))
        if self.mov.isChecked():
            out.append(Output('video', str(folder / f'{name}_alpha.mov'), 'element', 'prores4444', alpha_mode=self.mov_alpha.currentData()))
        if self.comp.isChecked() and self.comp_profile.count():
            prof = self.comp_profile.currentData()
            ext = {'h264': '.mp4', 'h265': '.mp4'}.get(prof, '.mov')
            out.append(Output('video', str(folder / f'{name}_comp{ext}'), 'composite', prof, audio=self.comp_audio.isChecked()))
        if self.comp_exr.isChecked():
            out.append(Output('exr', str(folder / f'{name}_comp' / f'{name}_comp.####.exr'), 'composite',
                              compression=self.exr_comp.currentData(), half=not self.exr_float.isChecked()))
        if self.vdb.isChecked():
            out.append(Output('vdb', str(folder / f'{name}_vdb' / f'{name}.####.vdb')))
        if self.mesh.isChecked() and self.mesh.isEnabled():
            out.append(Output('mesh', str(folder / f'{name}_liquid.usdc'), 'liquid'))
        if self.fabric.isChecked() and self.fabric.isEnabled():
            out.append(Output('mesh', str(folder / f'{name}_fabric.usdc'), 'fabric'))
        return out

    def _accept(self):
        s = QSettings()
        s.setValue('export/folder', self.folder.text())
        for k, w in (('exr', self.exr), ('deep', self.deep), ('png', self.png), ('mov', self.mov), ('comp', self.comp),
                     ('comp_exr', self.comp_exr), ('vdb', self.vdb), ('mesh', self.mesh), ('fabric', self.fabric)):
            s.setValue(f'export/{k}', w.isChecked())
        s.setValue('export/comp_profile', self.comp_profile.currentData())
        s.setValue('export/deep_samples', self.deep_n.currentData())
        if self.last.value() < self.first.value():
            QMessageBox.warning(self, 'Render', 'The last frame comes before the first frame.')
            return
        self.accept()

    def spec(self):
        sc = getattr(self.doc, 'shot', self.doc.scene).copy()
        sc.data['render']['width'], sc.data['render']['height'] = self.w.value(), self.h.value()
        sc.data['render']['final_scale'] = self.res_scale.value()
        return {'scene': sc, 'outputs': self.outputs(), 'frames': (self.first.value(), self.last.value()),
                'samples': self.samples.value(), 'motion_blur': self.mblur.isChecked()}


class RenderProgress(QDialog):
    def __init__(self, worker, spec, parent=None):
        super().__init__(parent)
        self.worker = worker
        self.spec = spec
        self.setWindowTitle('Rendering')
        self.setMinimumWidth(560)
        self.setModal(True)
        v = QVBoxLayout(self)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(280)
        self.preview.setStyleSheet(f'background: {theme.VIEWER}; border: 1px solid {theme.LINE};')
        v.addWidget(self.preview)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        v.addWidget(self.bar)
        self.text = QLabel('Starting…')
        self.text.setObjectName('hint')
        v.addWidget(self.text)
        b = QHBoxLayout()
        b.addStretch(1)
        self.open_btn = QPushButton('Open folder')
        self.open_btn.setVisible(False)
        self.open_btn.clicked.connect(self._open_folder)
        self.cancel = QPushButton('Cancel')
        self.cancel.clicked.connect(self._cancel)
        b.addWidget(self.open_btn)
        b.addWidget(self.cancel)
        v.addLayout(b)
        self.t0 = time.perf_counter()
        self.done = False
        worker.jobProgress.connect(self._progress)
        worker.jobDone.connect(self._done)

    def _progress(self, frac, text, img):
        self.bar.setValue(int(frac * 1000))
        el = time.perf_counter() - self.t0
        eta = el / frac - el if frac > 0.02 else 0
        self.text.setText(f'{text} · {el:.0f} s elapsed' + (f' · about {eta:.0f} s left' if eta > 0 else ''))
        if img is not None:
            pm = QPixmap.fromImage(img).scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview.setPixmap(pm)

    def _done(self, written, cancelled, err):
        self.done = True
        self.written = written
        el = time.perf_counter() - self.t0
        if err:
            self.text.setText('Render failed: ' + err.splitlines()[0])
            self.text.setToolTip(err)
        elif cancelled:
            self.text.setText(f'Cancelled after {el:.0f} s. Files written so far are kept.')
        else:
            self.bar.setValue(1000)
            self.text.setText(f'Done in {el:.0f} s · {len(written)} files written.')
        self.cancel.setText('Close')
        self.open_btn.setVisible(bool(written))

    def _cancel(self):
        if self.done:
            self.accept()
        else:
            self.worker.cancel_job()
            self.cancel.setEnabled(False)
            self.text.setText('Stopping after the current frame…')

    def _open_folder(self):
        first = Path(self.written[0]) if self.written else None
        if first:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(first.parent)))

    def closeEvent(self, e):
        if not self.done:
            self.worker.cancel_job()
        super().closeEvent(e)
