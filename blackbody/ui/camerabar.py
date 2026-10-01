"""The bar under the Build view for the shot's camera: make it see what you see, key it for a camera move,
or start looking from where it is."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMessageBox, QToolButton

from ..scene.anim import Curve
from . import icons, theme


class CameraBar(QFrame):
    def __init__(self, viewport):
        super().__init__(viewport)
        self.vp = viewport
        self.doc = viewport.doc
        self.setObjectName('camerabar')
        self.setStyleSheet(f'QFrame#camerabar {{ background: rgba(24,24,28,225); border: 1px solid {theme.LINE}; border-radius: 9px; }}'
                           'QLabel { background: transparent; }')
        h = QHBoxLayout(self)
        h.setContentsMargins(10, 4, 6, 4)
        h.setSpacing(4)
        ic = QLabel()
        ic.setPixmap(icons.glyph_icon('camera', theme.MUTED, 16).pixmap(QSize(16, 16)))
        h.addWidget(ic)
        t = QLabel('Shot camera')
        t.setStyleSheet('font-weight: 600;')
        h.addWidget(t)
        self.state = QLabel()
        self.state.setObjectName('faint')
        h.addWidget(self.state)
        h.addSpacing(4)
        self.use = self._button('Use this view', 'camera', self.use_view,
                                'The shot’s camera sees what you see here (a still camera; Tab shows the shot)')
        self.key = self._button('Key here', 'plus', self.key_view,
                                'A key for the shot’s camera at this frame, from this view. Go to another frame, look '
                                'from somewhere else and key again: the camera moves between them')
        self.look = self._button('Look through it', 'fit', self.look_through, 'Start the Build view from where the shot’s camera is')
        for b in (self.use, self.key, self.look):
            h.addWidget(b)
        self.doc.frameChanged.connect(lambda *_: self.sync())
        self.doc.sceneReplaced.connect(self.sync)
        self.doc.paramChanged.connect(lambda path: path and path[0] == 'camera' and self.sync())
        self.doc.footageChanged.connect(lambda *_: self.sync())
        self.doc.layersChanged.connect(self.sync)
        self.doc.structureChanged.connect(self.sync)
        self.sync()

    def _button(self, text, glyph, fn, tip):
        b = QToolButton()
        b.setText(text)
        b.setIcon(icons.glyph_icon(glyph, theme.TEXT, 14))
        b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        b.setToolTip(tip)
        b.setAutoRaise(True)
        b.clicked.connect(fn)
        return b

    def sync(self):
        sc = self.doc.scene
        matched = bool(sc.footage or sc.track)
        for b in (self.use, self.key):
            b.setEnabled(not matched)
        if matched:
            self.state.setText('· matched to the footage')
            self.setToolTip('The shot’s camera matches your footage, so it is not moved from here. Place the effect in the '
                            'shot instead (Shot, then drag it).')
        else:
            c = sc.data['camera']
            keys = set()
            for k in ('position', 'rotation', 'focal_mm', 'yaw', 'pitch', 'distance', 'target_y'):
                if isinstance(c.get(k), Curve):
                    keys |= {round(f[0], 3) for f in c[k].keys}
            self.state.setText(f'· moves, {len(keys)} keys' if len(keys) > 1 else ('· 1 key' if keys else '· still'))
            self.setToolTip('')
        self.adjustSize()
        self.vp.place_rotobar()

    def _run(self, fn):
        try:
            text = fn()
        except ValueError as ex:
            QMessageBox.information(self.window(), 'Shot camera', str(ex))
            return
        win = self.window()
        if hasattr(win, 'msg') and text:
            win.msg.setText(text)
        self.sync()

    def use_view(self):
        self._run(lambda: self.doc.shot_from_view(key=False))

    def key_view(self):
        self._run(lambda: self.doc.shot_from_view(key=True))

    def look_through(self):
        from .workview import WorkView
        self.doc.set_work_view(WorkView.from_shot(self.doc.scene, self.doc.frame))
        win = self.window()
        if hasattr(win, 'msg'):
            win.msg.setText('Looking from the shot’s camera. Move around, then Use this view (or Key here) to change it.')
