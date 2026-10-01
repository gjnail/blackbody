"""Shot steps: the card in the Shot view that walks through putting an effect into footage, in order, each step
with its button and a tick once it is done. Folds to a small pill; advanced ways in are at the bottom."""
from __future__ import annotations

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QCheckBox, QFrame, QHBoxLayout, QLabel, QPushButton, QToolButton, QVBoxLayout, QWidget

from ..scene.anim import Curve
from . import theme


class Step(QWidget):
    """One step: a number (a tick when done) and its title; the step being done (or one clicked open) also shows
    what to do and its buttons."""

    def __init__(self, n, title, parent=None):
        super().__init__(parent)
        self.n = n
        self.opened = False   # clicked open, though it is not the step being done
        self.current = False
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 1, 0, 1)
        h.setSpacing(8)
        self.mark = QLabel(str(n))
        self.mark.setFixedSize(20, 20)
        self.mark.setAlignment(Qt.AlignCenter)
        h.addWidget(self.mark, 0, Qt.AlignTop)
        v = QVBoxLayout()
        v.setSpacing(2)
        self.title = QLabel(f'<b>{title}</b>')
        self.title.setCursor(Qt.PointingHandCursor)
        self.title.setToolTip('Click to open or close this step')
        self.title.mousePressEvent = lambda e: self.toggle()
        v.addWidget(self.title)
        self.body = QWidget()
        bv = QVBoxLayout(self.body)
        bv.setContentsMargins(0, 0, 0, 4)
        bv.setSpacing(4)
        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setObjectName('hint')
        bv.addWidget(self.text)
        self.row = QHBoxLayout()
        self.row.setSpacing(4)
        bv.addLayout(self.row)
        v.addWidget(self.body)
        h.addLayout(v, 1)

    def toggle(self):
        self.opened = not (self.opened or self.current) if not self.current else False
        self.body.setVisible(self.current or self.opened)
        card = self.parentWidget()
        while card is not None and not isinstance(card, QFrame):
            card = card.parentWidget()
        if card is not None:
            card.adjustSize()

    def button(self, text, fn, primary=False, tip=''):
        b = QPushButton(text)
        b.setObjectName('primary' if primary else 'chip')
        b.setCursor(Qt.PointingHandCursor)
        b.setToolTip(tip)
        b.clicked.connect(fn)
        self.row.addWidget(b)
        return b

    def set_state(self, done, current, text):
        self.text.setText(text)
        if done:
            self.mark.setText('✓')
            self.mark.setStyleSheet(f'background: {theme.GOOD}; color: #10140f; border-radius: 10px; font-weight: 700;')
        else:
            self.mark.setText(str(self.n))
            self.mark.setStyleSheet(f'background: {theme.ACCENT if current else theme.FIELD_HI}; color: '
                                    f'{"#1b1107" if current else theme.MUTED}; border-radius: 10px; font-weight: 700;')
        self.title.setStyleSheet(f'color: {theme.TEXT if (current or done) else theme.MUTED};')
        self.current = current
        self.body.setVisible(current or self.opened)


class ShotSteps(QFrame):
    def __init__(self, viewport):
        super().__init__(viewport)
        self.vp = viewport
        self.doc = viewport.doc
        self.setObjectName('shotsteps')
        self.setStyleSheet(f'QFrame#shotsteps {{ background: rgba(22,22,26,236); border: 1px solid {theme.LINE_HI}; border-radius: 10px; }}'
                           'QLabel { background: transparent; }')
        self.setFixedWidth(312)
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 8, 10, 10)
        v.setSpacing(4)
        head = QHBoxLayout()
        self.head = QLabel()
        head.addWidget(self.head, 1)
        self.fold = QToolButton()
        self.fold.setAutoRaise(True)
        self.fold.setToolTip('Fold the steps away (they stay in View › Shot steps)')
        self.fold.clicked.connect(self.toggle)
        head.addWidget(self.fold)
        v.addLayout(head)
        self.body = QWidget()
        b = QVBoxLayout(self.body)
        b.setContentsMargins(0, 2, 0, 0)
        b.setSpacing(4)
        win = viewport.window

        def w():
            return win()
        self.s_footage = Step(1, 'Bring in your footage')
        self.s_footage.button('Import footage…', lambda: w().import_footage(), tip='A clip, an image sequence or a still (Ctrl+I)')
        self.s_ground = Step(2, 'Line up the ground')
        self.b_ground = self.s_ground.button('Line up…', lambda: w().line_up_ground(), primary=True,
                                             tip='Drag a grid onto the ground in the footage: the camera, lens and scale follow')
        self.b_surface = self.s_ground.button('+ Surface', lambda: w().add_surface(),
                                              tip='A wall, a table, a ramp or stairs in the footage: effects meet it and go behind it')
        self.s_ground.button('No ground in view', lambda: (self.vp.ground_flat(), self._mark('flat', True)),
                             tip='Pin the effect to a point of the frame instead, and scale it by hand')
        self.s_place = Step(3, 'Put the effect where it goes')
        self.s_place.button('Ready-made effects', lambda: w().focus_effects(), tip='Load one (Keep my shot keeps your footage and camera)')
        self.s_place.button('Build one', lambda: w().focus_create(), tip='Create: drag blocks onto the footage')
        self.s_move = Step(4, 'Follow the camera move')
        self.b_track = self.s_move.button('Track the camera', lambda: w().track(), primary=True,
                                          tip='Follow spots through the footage and work out how the camera turns (Ctrl+T)')
        self.still = QCheckBox('It holds still')
        self.still.setToolTip('A locked-off shot: nothing to follow')
        self.still.toggled.connect(lambda on: self._mark('still', on))
        self.s_move.row.addWidget(self.still)
        self.s_front = Step(5, 'Anything in front of it?')
        self.s_front.button('Draw roto', lambda: w().roto_btn.setChecked(True),
                            tip='Draw round what passes in front of the effect: it goes behind it (R)')
        self.clear = QCheckBox('Nothing in front')
        self.clear.toggled.connect(lambda on: self._mark('front_clear', on))
        self.s_front.row.addWidget(self.clear)
        self.s_look = Step(6, 'Make it sit in the footage')
        self.s_look.button('Blend with the footage', lambda: w().blend_settings(),
                           tip='Heat haze, the effect’s light on the footage, grain and the lens: the Composite settings')
        self.s_render = Step(7, 'Render')
        self.s_render.button('Render…', lambda: w().render_dialog(), tip='Ctrl+M')
        self.steps = [self.s_footage, self.s_ground, self.s_place, self.s_move, self.s_front, self.s_look, self.s_render]
        for s in self.steps:
            b.addWidget(s)
        more = QLabel('<a href="chan" style="color:%s">Import a camera solve (.chan)</a> · <a href="usd" style="color:%s">'
                      'a USD scene</a> · <a href="cam" style="color:%s">exact camera settings</a>' % ((theme.MUTED,) * 3))
        more.setObjectName('hint')
        more.setWordWrap(True)
        more.linkActivated.connect(self._link)
        more.setToolTip('For shots tracked in another program (SynthEyes, PFTrack, Nuke, Blender, 3DEqualizer), or to type the lens, '
                        'sensor and camera position')
        b.addWidget(more)
        v.addWidget(self.body)
        self.open = QSettings().value('ui/shot_steps_open', True, type=bool)
        for sig in (self.doc.sceneReplaced, self.doc.structureChanged, self.doc.layersChanged):
            sig.connect(self.sync)
        self.doc.paramChanged.connect(lambda path: path and path[0] in ('camera', 'composite') and self.sync())
        self.doc.footageChanged.connect(lambda *_: self.sync())
        self.sync()

    def _link(self, what):
        win = self.vp.window()
        if what == 'chan':
            win.import_chan()
        elif what == 'usd':
            win.import_usd()
        elif hasattr(win, 'props'):
            win.props.reveal(('camera', 'focal_mm'))

    def _mark(self, key, on):
        sc = self.doc.scene
        if not sc.footage or bool(sc.footage.get(key)) == bool(on):
            return

        def fn(s):
            for x in [self.doc.shot] + list(self.doc.shot.layers):
                if x.footage:
                    x.footage = dict(x.footage, **{key: bool(on)})
        self.doc.edit('Shot steps', fn)
        self.sync()

    def toggle(self):
        self.open = not self.open
        QSettings().setValue('ui/shot_steps_open', self.open)
        self.sync()

    def state(self):
        """[(done, text)] for each step, from the shot as it is."""
        sc = self.doc.scene
        foot = sc.footage or {}
        info = self.doc.footage_info or {}
        c = sc.data['camera']
        g = sc.ground or {}
        flat = bool(c.get('use_anchor')) and not g
        has_fx = bool(sc.emitters or sc.colliders or sc.fabrics)
        tracked = bool(g.get('tracked')) or (isinstance(c.get('position'), Curve) and not c.get('use_anchor')) \
            or bool(sc.track and sc.track.get('points'))
        out = [
            (bool(foot.get('path')), f'{_name(foot.get("path"))}' if foot.get('path') else 'A clip, an image sequence or a still. '
             'Or drop it on the window.'),
            (bool(g) or (flat and bool(foot.get('path')) and foot.get('flat')),
             (f'The camera matches the footage: {float(c["focal_mm"]):.0f} mm lens, {g.get("height", 1.6) if g.get("scale_by") != "side" else "–"} m up.'
              if g else 'Drag a grid onto the ground in the footage (a floor, a road, a rug): the lens, the camera’s height '
                        'and angle follow, so effects stand on the real ground at their real size.')),
            (has_fx and bool(g), ('Drag the ring at its base over the ground; drop blocks from Create straight onto the footage.'
                                  if g else 'Drag the ring onto the spot and the square to size it.') if has_fx else
             'Load a ready-made effect, or build one: it stands in the middle of the grid.'),
            (tracked or bool(foot.get('still')), (f'Followed over {g["tracked"]} frames ({"the camera travels" if g.get("travel") else "the camera turns"}), within {g.get("error", 0):.1f} px.' if g.get('tracked')
                                                   else 'If the camera pans, tilts or shakes, follow it so the effect stays put.')),
            (bool(getattr(sc, 'roto', None)) or bool(foot.get('front_clear')),
             f'{len(sc.roto)} roto shape{"s" if len(sc.roto) != 1 else ""}.' if getattr(sc, 'roto', None) else
             'People or things passing in front of the effect: draw round them and the effect goes behind.'),
            (False, 'Heat haze, the light it throws on the footage, grain, the lens’s softness.'),
            (False, 'A finished composite, or the effect alone with alpha for your compositor.'),
        ]
        if info.get('focal_35') and not g:
            out[1] = (out[1][0], out[1][1] + f' The file says it was shot at {info["focal_35"]:g} mm (full-frame).')
        return out

    def sync(self):
        st = self.state()
        done = sum(1 for d, _ in st[:5] if d)
        current = next((k for k, (d, _) in enumerate(st) if not d), len(st) - 1)
        for k, (s, (d, text)) in enumerate(zip(self.steps, st)):
            s.set_state(d, k == current, text)
        self.b_ground.setText('Again…' if self.doc.scene.ground else 'Line up…')
        self.b_surface.setVisible(bool(self.doc.scene.ground))
        self.b_track.setEnabled(bool(self.doc.scene.footage))
        for box, key in ((self.still, 'still'), (self.clear, 'front_clear')):
            box.blockSignals(True)
            box.setChecked(bool((self.doc.scene.footage or {}).get(key)))
            box.blockSignals(False)
        self.head.setText(f'<b>Put it in your shot</b> <span style="color:{theme.FAINT}">· {done} of 5 done</span>')
        self.fold.setText('▸' if not self.open else '▾')
        self.body.setVisible(self.open)
        self.adjustSize()
        self.vp.place_rotobar()


def _name(path):
    from pathlib import Path
    return Path(str(path)).name
