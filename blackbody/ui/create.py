"""Create: make a scene from scratch. Start an empty fire, liquid, fire-and-liquid or sky scene at a
size, then add building blocks (scene/components.py): click one to add it, or drag it into the viewer
to put it down where you drop it."""
from __future__ import annotations

from PySide6.QtCore import QMimeData, QPoint, QRectF, QSettings, QSize, Qt, Signal
from PySide6.QtGui import QColor, QDrag, QFont, QPainter, QPen
from PySide6.QtWidgets import (QAbstractButton, QButtonGroup, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QSizePolicy,
                               QStackedWidget, QToolButton, QVBoxLayout, QWidget)

from ..scene import components
from . import icons, theme
from .library import FlowLayout
from .params import guard_wheel

MIME = 'application/x-blackbody-component'
GROUP_COLOURS = {'Fire': theme.ACCENT, 'Smoke, steam & sparks': '#c9c9d1', 'Liquids': '#6fb6ff', 'Fabric': '#d59cff',
                 'Weather': '#a8e0ff', 'Forces': '#9fe0c8', 'Objects': '#9fb0c4', 'Lights': '#ffdc78'}
GROUP_HINTS = {'Fire': 'Sources of flame, and things that burn.', 'Smoke, steam & sparks': 'Smoke, steam and sparks without flame.',
               'Liquids': 'Water, honey, ink and lava: sources, standing water. With fire in the scene too, the two meet.',
               'Fabric': 'Cloth that hangs, drapes, blows in the air, soaks up water and burns.',
               'Weather': 'Rain, snow, sleet, hail and wind over the whole scene.',
               'Objects': 'Solid things the effect flows around, bounces off or burns. In your shot they hide what is behind them.',
               'Forces': 'Air pushed around: wind, a fan, an updraft, suction, a vortex. It carries smoke, flame, embers and cloth.',
               'Lights': 'Lights in the set: they light the smoke and steam, and the smoke shadows them.'}
STARTERS = ['burner', 'pour', 'flag', 'snow', 'smoke', 'box']   # one of each kind, shown first under All
DOMAINS = [('all', 'All', None), ('fire', 'Fire', ('Fire', 'Smoke, steam & sparks')), ('liquid', 'Liquids', ('Liquids',)),
           ('fabric', 'Fabric', ('Fabric',)), ('weather', 'Weather', ('Weather',)), ('forces', 'Forces', ('Forces',)),
           ('objects', 'Objects', ('Objects', 'Lights'))]


class Tile(QAbstractButton):
    """A building block: a glyph over its name. Click to add it; drag it into the viewer to place it."""

    def __init__(self, comp, parent=None):
        super().__init__(parent)
        self.comp = comp
        self.colour = GROUP_COLOURS.get(comp.group, theme.TEXT)
        self.setFixedSize(QSize(94, 70))
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(f'<b>{comp.name}</b><br>{comp.tip}<br><i>Click to add it, or drag it into the viewer to place it.</i>')
        self._press = None

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        hot = self.underMouse() and self.isEnabled()
        p.setPen(QPen(QColor(theme.LINE_HI if hot else theme.LINE), 1))
        p.setBrush(QColor(theme.FIELD_HI if hot else theme.CARD))
        p.drawRoundedRect(r, 8, 8)
        col = QColor(self.colour if self.isEnabled() else theme.FAINT)
        icons.paint_glyph(p, self.comp.glyph, QRectF(r.center().x() - 12, r.y() + 9, 24, 24), col, 1.6)
        f = QFont(self.font())
        f.setPointSizeF(8.3)
        p.setFont(f)
        p.setPen(QColor(theme.TEXT if self.isEnabled() else theme.FAINT))
        text = p.fontMetrics().elidedText(self.comp.name, Qt.ElideRight, int(r.width() - 8))
        p.drawText(QRectF(r.x() + 4, r.y() + 40, r.width() - 8, 22), Qt.AlignHCenter | Qt.AlignVCenter, text)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._press = e.position().toPoint()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._press is not None and (e.position().toPoint() - self._press).manhattanLength() > 8 and self.isEnabled():
            self._press = None
            self.setDown(False)
            drag = QDrag(self)
            md = QMimeData()
            md.setData(MIME, self.comp.key.encode())
            drag.setMimeData(md)
            pm = self.grab()
            drag.setPixmap(pm)
            drag.setHotSpot(QPoint(pm.width() // 2, pm.height() // 2))
            drag.exec(Qt.CopyAction)
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._press is None and not self.isDown():
            return   # the end of a drag, not a click
        self._press = None
        super().mouseReleaseEvent(e)


class KindButton(QAbstractButton):
    """Start from scratch: one kind of scene."""

    def __init__(self, kind, glyph, label, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.glyph = glyph
        self.label = label
        self.setFixedHeight(62)
        self.setMinimumWidth(60)
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        hot = self.underMouse()
        p.setPen(QPen(QColor(theme.ACCENT if hot else theme.LINE_HI), 1))
        p.setBrush(QColor(theme.FIELD_HI if hot else theme.FIELD))
        p.drawRoundedRect(r, 7, 7)
        cx = r.center().x()
        glyphs = self.glyph if isinstance(self.glyph, tuple) else (self.glyph,)
        w = 20 * len(glyphs) + 2 * (len(glyphs) - 1)
        for i, g in enumerate(glyphs):
            col = {'flame': theme.ACCENT, 'drop': '#6fb6ff', 'cloud': '#d0d6e0', 'fabric': '#d59cff', 'weather': '#a8e0ff',
                   'snow': '#a8e0ff'}.get(g, theme.TEXT)
            icons.paint_glyph(p, g, QRectF(cx - w / 2 + i * 22, r.y() + 8, 20, 20), col, 1.5)
        f = QFont(self.font())
        f.setPointSizeF(8.3)
        p.setFont(f)
        p.setPen(QColor(theme.TEXT))
        p.drawText(QRectF(r.x(), r.y() + 32, r.width(), 20), Qt.AlignHCenter | Qt.AlignVCenter, self.label)


class CreatePanel(QWidget):
    newScene = Signal(str, str)        # kind, scale
    addComponent = Signal(str)         # component key, added at its own place

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        v = QVBoxLayout(body)
        v.setContentsMargins(0, 8, 6, 12)
        v.setSpacing(8)

        # start from scratch
        card = QFrame()
        card.setObjectName('card')
        cv = QVBoxLayout(card)
        cv.setContentsMargins(10, 8, 10, 10)
        cv.setSpacing(6)
        head = QHBoxLayout()
        t = QLabel('Start from scratch')
        t.setStyleSheet('font-weight: 600;')
        head.addWidget(t)
        head.addStretch(1)
        self.scale = QComboBox()
        for key, (label, w) in components.SCALES.items():
            self.scale.addItem(f'{label} ({w:g} m)', key)
        self.scale.setCurrentIndex(max(0, self.scale.findData(QSettings().value('ui/new_scale', 'person'))))
        self.scale.setToolTip('How big the effect is: the simulation box, the building blocks you add and the turbulence and '
                              'detail of the air follow it.')
        self.scale.currentIndexChanged.connect(lambda i: QSettings().setValue('ui/new_scale', self.scale.itemData(i)))
        guard_wheel(self.scale)
        head.addWidget(self.scale)
        cv.addLayout(head)
        sub = QLabel('An empty scene to build in: what you add decides what it simulates (fire, liquids, fabric, weather, '
                     'or several together). Ctrl+Z brings back the scene you had.')
        sub.setObjectName('hint')
        sub.setWordWrap(True)
        cv.addWidget(sub)
        kinds = QHBoxLayout()
        kinds.setSpacing(6)
        for kind, glyph, label in (('auto', ('flame', 'drop', 'fabric', 'weather'), 'Empty scene'), ('cloud', 'cloud', 'Sky')):
            b = KindButton(kind, glyph, label)
            b.setToolTip({'auto': 'An empty box at the size above. Add fire, liquids, fabric, weather and objects to it.',
                          'cloud': 'A sky kilometres across, where clouds build from the warmed ground into storms'}[kind])
            b.clicked.connect(lambda _=False, k=kind: self.newScene.emit(k, self.scale.currentData()))
            kinds.addWidget(b, 3 if kind == 'auto' else 1)
        cv.addLayout(kinds)
        v.addWidget(card)

        # building blocks
        head = QHBoxLayout()
        t = QLabel('Add to the scene')
        t.setObjectName('title')
        t.setStyleSheet('font-size: 11pt;')
        head.addWidget(t)
        head.addStretch(1)
        v.addLayout(head)
        hint = QLabel('Click a block to add it, or drag it into the viewer to put it where you drop it. '
                      'Then drag it in the viewer, and tune it in Properties.')
        hint.setObjectName('hint')
        hint.setWordWrap(True)
        v.addWidget(hint)
        self.search = QLineEdit()
        self.search.setObjectName('search')
        self.search.setPlaceholderText('Search building blocks…')
        self.search.setClearButtonEnabled(True)
        self.search.addAction(icons.glyph_icon('search', theme.FAINT, 16), QLineEdit.LeadingPosition)
        self.search.textChanged.connect(self._filter)
        v.addWidget(self.search)
        chips = QWidget()
        cf = FlowLayout(chips, 5)
        cf.setContentsMargins(0, 0, 0, 0)
        self.domain = 'all'
        self.domain_group = QButtonGroup(self)
        self.domain_buttons = {}
        for key, label, _ in DOMAINS:
            b = QPushButton(label)
            b.setObjectName('chip')
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, k=key: self._set_domain(k))
            self.domain_group.addButton(b)
            cf.addWidget(b)
            self.domain_buttons[key] = b
        self.domain_buttons['all'].setChecked(True)
        v.addWidget(chips)
        self.tiles = []
        self.sections = []
        lab = QLabel('START WITH')
        lab.setObjectName('section')
        lab.setToolTip('One of each kind of thing Blackbody simulates')
        lab.setContentsMargins(2, 6, 0, 0)
        v.addWidget(lab)
        w = QWidget()
        flow = FlowLayout(w, 6)
        flow.setContentsMargins(0, 0, 0, 0)
        starters = []
        for key in STARTERS:
            c = components.BY_KEY[key]
            t = Tile(c)
            t.clicked.connect(lambda _=False, k=c.key: self.addComponent.emit(k))
            flow.addWidget(t)
            starters.append(t)
        v.addWidget(w)
        self.starters = (lab, w, starters)
        self.tiles.extend(starters)
        for g in components.GROUPS:
            lab = QLabel(g.upper())
            lab.setObjectName('section')
            lab.setToolTip(GROUP_HINTS.get(g, ''))
            lab.setContentsMargins(2, 6, 0, 0)
            v.addWidget(lab)
            w = QWidget()
            flow = FlowLayout(w, 6)
            flow.setContentsMargins(0, 0, 0, 0)
            tiles = []
            for c in components.COMPONENTS:
                if c.group != g:
                    continue
                t = Tile(c)
                t.clicked.connect(lambda _=False, k=c.key: self.addComponent.emit(k))
                flow.addWidget(t)
                tiles.append(t)
            v.addWidget(w)
            self.tiles.extend(tiles)
            self.sections.append((lab, w, tiles))
        v.addStretch(1)
        doc.sceneReplaced.connect(self._sync)
        doc.paramChanged.connect(lambda path: path == ('domain', 'kind') and self._sync())
        self._sync()

    def _sync(self):
        """In a sky scene only lights and objects (terrain) can go in."""
        for t in self.tiles:
            ok = components.target_kind(self.doc.scene, t.comp) is not None
            t.setEnabled(ok)

    def _filter(self, text):
        words = text.lower().split()
        lab, w, starters = self.starters
        on = self.domain == 'all' and not words
        lab.setVisible(on)
        w.setVisible(on)
        for lab, w, tiles in self.sections:
            shown = 0
            for t in tiles:
                hay = f'{t.comp.name} {t.comp.group} {t.comp.tip} {t.comp.key}'.lower()
                on = all(x in hay for x in words)
                t.setVisible(on)
                shown += on
            dom = dict((k, g) for k, _, g in DOMAINS).get(self.domain)
            on_dom = dom is None or tiles and tiles[0].comp.group in dom
            for t in tiles:
                if not on_dom:
                    t.setVisible(False)
            lab.setVisible(shown > 0 and on_dom)
            w.setVisible(shown > 0 and on_dom)

    def _set_domain(self, key):
        self.domain = key
        self.domain_buttons[key].setChecked(True)
        self._filter(self.search.text())

    def focus_search(self):
        self.search.setFocus()
        self.search.selectAll()


class LeftPanel(QWidget):
    """Effects (ready-made presets) and Create (from scratch), one at a time."""

    def __init__(self, library, create, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 10, 6, 0)
        v.setSpacing(0)
        bar = QHBoxLayout()
        bar.setSpacing(1)
        self.group = QButtonGroup(self)
        self.buttons = {}
        for i, (key, label, tip) in enumerate((('effects', 'Effects', 'Ready-made effects: click one to load it (Ctrl+E)'),
                                              ('create', 'Create', 'Make your own: start an empty scene and add fire, liquids, '
                                                                   'fabric, weather, objects and lights (Ctrl+N)'))):
            b = QToolButton()
            b.setText(label)
            b.setObjectName('segFirst' if i == 0 else 'segLast')
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(tip)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setMinimumHeight(30)
            b.clicked.connect(lambda _=False, k=key: self.show_page(k))
            self.group.addButton(b)
            bar.addWidget(b)
            self.buttons[key] = b
        v.addLayout(bar)
        self.stack = QStackedWidget()
        self.pages = {'effects': library, 'create': create}
        self.stack.addWidget(library)
        self.stack.addWidget(create)
        v.addWidget(self.stack, 1)
        library.layout().setContentsMargins(0, 8, 0, 8)
        library.title.hide()
        self.show_page(QSettings().value('ui/left_page', 'effects'))

    def show_page(self, key):
        if key not in self.pages:
            key = 'effects'
        self.buttons[key].setChecked(True)
        self.stack.setCurrentWidget(self.pages[key])
        QSettings().setValue('ui/left_page', key)

    def page(self):
        return 'create' if self.stack.currentWidget() is self.pages['create'] else 'effects'
