"""Effects: the built-in presets with rendered thumbnails, plus the user's own saved presets, in
categories with a search. One click loads an effect (Undo brings the last one back)."""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QPoint, QRect, QRectF, QSize, QStandardPaths, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QHBoxLayout, QInputDialog, QLabel, QLayout,
                               QLineEdit, QListView, QListWidget, QListWidgetItem, QMenu, QMessageBox, QPushButton,
                               QStyle, QStyledItemDelegate, QVBoxLayout, QWidget)

from ..scene import PROJECT_EXT, presets
from . import icons, theme

ASSETS = Path(__file__).resolve().parents[1] / 'assets' / 'presets'
THUMB = QSize(176, 99)   # the size saved user-preset thumbnails are made at (twice this)

# The presets' own categories, gathered into fewer, plainer ones for the chips
CATEGORIES = [('all', 'All'), ('fire', 'Fire'), ('water', 'Liquids'), ('fabric', 'Fabric'), ('sky', 'Weather & sky'),
              ('sea', 'Sea'), ('smoke', 'Smoke & steam'), ('blast', 'Blasts & sparks'), ('ice', 'Ice & heat'),
              ('both', 'Mixed'), ('physics', 'Falling & breaking'), ('matter', 'Sand, snow & mud'), ('guns', 'Bullets'),
              ('wood', 'Wood'), ('mine', 'Mine')]
CATEGORY_TIPS = {'all': 'Every effect', 'fire': 'Fires and small flames', 'blast': 'Explosions, fireballs and sparks',
                 'smoke': 'Smoke and steam', 'water': 'Liquids: pours, splashes, honey, ink, lava', 'sea': 'Oceans, surf and rivers',
                 'fabric': 'Cloth: curtains, flags and towels that blow, burn and soak', 'ice': 'Ice, freezing, boiling and steam',
                 'sky': 'Rain, snow, hail, clouds and storms', 'both': 'Fire and liquid in one simulation',
                 'physics': 'Things that fall, tumble, break, swing on ropes and turn on hinges',
                 'matter': 'Sand that piles and pours, snowballs, mud, jelly and clay',
                 'guns': 'Bullets through glass, wood, steel, water and gel',
                 'wood': 'Wood that splits, splinters and snaps along its grain',
                 'mine': 'The presets you saved'}
CATEGORY_OF = {'Fires': 'fire', 'Small flames': 'fire', 'Explosions': 'blast', 'Sparks': 'blast', 'Smoke': 'smoke',
               'Steam': 'smoke', 'Liquids': 'water', 'Sea': 'sea', 'Ice and steam': 'ice', 'Weather': 'sky',
               'Sky and weather': 'sky', 'Fire and liquid': 'both', 'Things that fall': 'physics', 'Ropes and hinges': 'physics',
               'Breaking': 'physics', 'Sand, snow and mud': 'matter', 'Sand, snow & mud': 'matter', 'Bullets': 'guns',
               'Wood': 'wood'}
# presets whose subject is not the category they were filed under
CATEGORY_KEY = {'fabric_curtain': 'fabric', 'curtain_fire': 'fabric', 'wet_towels': 'fabric', 'flag_wind': 'fabric',
                'towel_dip': 'fabric', 'rain_pond': 'sky'}
ROUND_ROBIN = ['fire', 'water', 'fabric', 'sky', 'sea', 'smoke', 'blast', 'ice', 'both', 'physics', 'matter', 'guns', 'wood']


def mixed_order(keys):
    """The presets one category at a time in turn, so the first screen of All shows every kind of effect."""
    by = {c: [k for k in keys if category(k) == c] for c in ROUND_ROBIN}
    out = []
    while any(by.values()):
        for c in ROUND_ROBIN:
            if by[c]:
                out.append(by[c].pop(0))
    return out + [k for k in keys if k not in out]


def category(key):
    return CATEGORY_KEY.get(key) or CATEGORY_OF.get(presets.PRESETS[key]['category'], 'fire')
ROLE_REF = Qt.UserRole
ROLE_CAT = Qt.UserRole + 1
ROLE_SEARCH = Qt.UserRole + 2
ROLE_SIZE = Qt.UserRole + 3
ROLE_PIX = Qt.UserRole + 4


def user_preset_dir():
    base = Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)) / 'presets'
    base.mkdir(parents=True, exist_ok=True)
    return base


def _placeholder():
    pm = QPixmap(QSize(320, 180))
    pm.fill(QColor('#161618'))
    p = QPainter(pm)
    icons.flame().paint(p, pm.rect().adjusted(130, 50, -130, -50))
    p.end()
    return pm


class FlowLayout(QLayout):
    """Lays widgets out in rows that wrap, like words."""

    def __init__(self, parent=None, spacing=5):
        super().__init__(parent)
        self.items = []
        self.sp = spacing

    def addItem(self, item):
        self.items.append(item)

    def count(self):
        return len(self.items)

    def itemAt(self, i):
        return self.items[i] if 0 <= i < len(self.items) else None

    def takeAt(self, i):
        return self.items.pop(i) if 0 <= i < len(self.items) else None

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        return self._layout(QRect(0, 0, w, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        s = QSize()
        for it in self.items:
            s = s.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return s + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _layout(self, rect, test):
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line = r.x(), r.y(), 0
        for it in self.items:
            hint = it.sizeHint()
            if x + hint.width() > r.right() + 1 and line > 0:
                x = r.x()
                y += line + self.sp
                line = 0
            if not test:
                it.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self.sp
            line = max(line, hint.height())
        return y + line - rect.y() + m.bottom()


class CardDelegate(QStyledItemDelegate):
    """A preset as a card: rounded thumbnail, name, size. The loaded one has an ember outline."""

    def __init__(self, browser):
        super().__init__(browser)
        self.browser = browser
        self.cell = QSize(160, 130)
        self._cache = {}

    def sizeHint(self, option, index):
        return self.cell

    def paint(self, p, option, index):
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        r = QRectF(option.rect).adjusted(4, 4, -4, -4)
        tw = r.width()
        th = round(tw * 9 / 16)
        trect = QRectF(r.x(), r.y(), tw, th)
        hot = bool(option.state & QStyle.State_MouseOver)
        ref = index.data(ROLE_REF)
        current = ref == self.browser.current
        # thumbnail
        key = (index.row(), int(tw), int(th))
        pm = self._cache.get(key)
        if pm is None:
            src = index.data(ROLE_PIX)
            dpr = 2.0
            pm = src.scaled(QSize(int(tw * dpr), int(th * dpr)), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            pm = pm.copy(QRect((pm.width() - int(tw * dpr)) // 2, (pm.height() - int(th * dpr)) // 2, int(tw * dpr), int(th * dpr)))
            pm.setDevicePixelRatio(dpr)
            self._cache[key] = pm
        clip = QPainterPath()
        clip.addRoundedRect(trect, 7, 7)
        p.setClipPath(clip)
        p.drawPixmap(trect.topLeft(), pm)
        if hot and not current:
            p.fillRect(trect, QColor(255, 255, 255, 18))
        p.setClipping(False)
        if current or hot:
            p.setPen(QPen(QColor(theme.ACCENT if current else theme.FAINT), 2.0 if current else 1.2))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(trect.adjusted(0.5, 0.5, -0.5, -0.5), 7, 7)
        if current:
            badge = QRectF(trect.x() + 6, trect.y() + 6, 52, 17)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(theme.ACCENT))
            p.drawRoundedRect(badge, 8.5, 8.5)
            f = QFont(option.font)
            f.setPointSizeF(7.5)
            f.setWeight(QFont.DemiBold)
            p.setFont(f)
            p.setPen(QColor('#1b1107'))
            p.drawText(badge, Qt.AlignCenter, 'LOADED')
        # text
        f = QFont(option.font)
        f.setPointSizeF(9.0)
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        p.setPen(QColor(theme.TEXT))
        name_r = QRectF(r.x() + 2, trect.bottom() + 5, r.width() - 4, 17)
        p.drawText(name_r, Qt.AlignLeft | Qt.AlignVCenter, p.fontMetrics().elidedText(index.data(Qt.DisplayRole), Qt.ElideRight, int(name_r.width())))
        f.setPointSizeF(8.0)
        f.setWeight(QFont.Normal)
        p.setFont(f)
        p.setPen(QColor(theme.MUTED))
        size_r = QRectF(r.x() + 2, name_r.bottom(), r.width() - 4, 15)
        p.drawText(size_r, Qt.AlignLeft | Qt.AlignVCenter, p.fontMetrics().elidedText(index.data(ROLE_SIZE) or '', Qt.ElideRight, int(size_r.width())))
        p.restore()


class EffectList(QListWidget):
    """The card grid: as many columns as fit, cards stretched to fill the width."""

    def __init__(self, delegate, parent=None):
        super().__init__(parent)
        self.delegate = delegate
        self.setItemDelegate(delegate)
        self.setViewMode(QListView.ListMode)
        self.setFlow(QListView.LeftToRight)
        self.setWrapping(True)
        self.setResizeMode(QListView.Adjust)
        self.setUniformItemSizes(True)
        self.setMovement(QListView.Static)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.verticalScrollBar().setSingleStep(24)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMouseTracking(True)
        self.setSpacing(0)
        self.setStyleSheet('QListWidget::item, QListWidget::item:hover, QListWidget::item:selected { background: transparent; }')

    def resizeEvent(self, e):
        w = self.viewport().width() - 2
        cols = max(1, w // 128)
        cw = w // cols
        cell = QSize(cw, round((cw - 8) * 9 / 16) + 8 + 42)
        if cell != self.delegate.cell:
            self.delegate.cell = cell
            self.delegate._cache.clear()
            for i in range(self.count()):
                self.item(i).setSizeHint(cell)
        super().resizeEvent(e)


class Library(QWidget):
    presetChosen = Signal(str, bool)       # built-in name, keep shot
    userPresetChosen = Signal(str, bool)   # file path, keep shot

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.current = ('builtin', doc.scene.preset)
        self._last_use = (None, 0.0)
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 10, 6, 8)
        v.setSpacing(8)
        head = QHBoxLayout()
        self.title = QLabel('Effects')
        self.title.setObjectName('title')
        head.addWidget(self.title)
        head.addStretch(1)
        self.count_label = QLabel()
        self.count_label.setObjectName('faint')
        head.addWidget(self.count_label)
        v.addLayout(head)
        self.search = QLineEdit()
        self.search.setObjectName('search')
        self.search.setPlaceholderText('Search effects…   (Ctrl+E)')
        self.search.setClearButtonEnabled(True)
        self.search.addAction(icons.glyph_icon('search', theme.FAINT, 16), QLineEdit.LeadingPosition)
        self.search.textChanged.connect(self._filter)
        v.addWidget(self.search)
        chips = QWidget()
        self.flow = FlowLayout(chips, 5)
        self.flow.setContentsMargins(0, 0, 0, 0)
        self.chip_group = QButtonGroup(self)
        self.chip_group.setExclusive(True)
        self.chips = {}
        for key, label in CATEGORIES:
            b = QPushButton(label.replace('&', '&&'))
            b.setObjectName('chip')
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(CATEGORY_TIPS[key])
            b.clicked.connect(lambda _=False, k=key: self._set_category(k))
            self.chip_group.addButton(b)
            self.flow.addWidget(b)
            self.chips[key] = b
        self.category = 'all'
        self.chips['all'].setChecked(True)
        v.addWidget(chips)
        self.hint = QLabel()
        self.hint.setObjectName('hint')
        self.hint.setWordWrap(True)
        v.addWidget(self.hint)
        self.delegate = CardDelegate(self)
        self.list = EffectList(self.delegate)
        self.list.itemClicked.connect(self._use)
        self.list.itemActivated.connect(self._use)   # Enter on the selected one
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._menu)
        v.addWidget(self.list, 1)
        self.empty = QLabel('No effect matches. Try another word, or All.')
        self.empty.setObjectName('hint')
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.hide()
        v.addWidget(self.empty)
        row = QHBoxLayout()
        row.setContentsMargins(2, 0, 4, 0)
        self.keep = QCheckBox('Keep my shot')
        self.keep.setChecked(True)
        row.addWidget(self.keep)
        row.addStretch(1)
        save = QPushButton('Save as preset…')
        save.setObjectName('ghost')
        save.setIcon(icons.glyph_icon('save', theme.MUTED, 16, active=theme.TEXT))
        save.setToolTip('Keep the current effect and its settings as your own preset (under My presets)')
        save.clicked.connect(self.save_current)
        row.addWidget(save)
        v.addLayout(row)
        doc.sceneReplaced.connect(self._scene_replaced)
        doc.footageChanged.connect(lambda *_: self._sync_keep())
        self._sync_keep()
        self.populate()

    # -- state ------------------------------------------------------------------------------------------

    def _sync_keep(self):
        """Keep my shot only means something once there is a shot (footage, a track, a matched camera)."""
        shot = self.doc.has_shot()
        self.keep.setEnabled(shot)
        if shot:
            self.keep.setToolTip('Keep your footage, camera placement, frame range and output settings when loading an effect.')
            self.hint.setText('Click an effect to load it. <b>Keep my shot</b> keeps your footage, camera and frame range.')
        else:
            self.keep.setToolTip('Nothing to keep yet: until you import footage (File › Import footage), '
                                 'each effect loads with its own camera and frame range, as in its picture.')
            self.hint.setText('Click an effect to load it. Ctrl+Z brings back the last one.')

    def _scene_replaced(self):
        self._sync_keep()
        if self.current[0] != 'user':
            self.current = ('builtin', self.doc.scene.preset)
        self.list.viewport().update()

    def populate(self):
        self.list.clear()
        self.delegate._cache.clear()
        ph = _placeholder()
        counts = {k: 0 for k, _ in CATEGORIES}
        for key in mixed_order(presets.ORDER):
            p = presets.PRESETS[key]
            f = ASSETS / f'{key}.png'
            pm = QPixmap(str(f)) if f.exists() else ph
            cat = category(key)
            it = QListWidgetItem(p['name'])
            it.setToolTip(f'<b>{p["name"]}</b> · {p["size"]}<br>{p["blurb"]}')
            it.setData(ROLE_REF, ('builtin', key))
            it.setData(ROLE_CAT, cat)
            it.setData(ROLE_SIZE, p['size'])
            it.setData(ROLE_PIX, pm)
            it.setData(ROLE_SEARCH, ' '.join((p['name'], p['size'], p['category'], p.get('blurb', ''), key)).lower())
            it.setSizeHint(self.delegate.cell)
            self.list.addItem(it)
            counts[cat] += 1
        for f in sorted(user_preset_dir().glob('*' + PROJECT_EXT)):
            thumb = f.with_suffix('.png')
            pm = QPixmap(str(thumb)) if thumb.exists() else ph
            it = QListWidgetItem(f.stem)
            it.setToolTip(f'<b>{f.stem}</b> · your preset · right-click to delete')
            it.setData(ROLE_REF, ('user', str(f)))
            it.setData(ROLE_CAT, 'mine')
            it.setData(ROLE_SIZE, 'My preset')
            it.setData(ROLE_PIX, pm)
            it.setData(ROLE_SEARCH, f.stem.lower() + ' my preset')
            it.setSizeHint(self.delegate.cell)
            self.list.addItem(it)
            counts['mine'] += 1
        counts['all'] = self.list.count()
        for key, label in CATEGORIES:
            text = label.replace('&', '&&')
            self.chips[key].setText(f'{text}  {counts[key]}' if key != 'mine' or counts[key] else text)
        self._filter()

    def _set_category(self, key):
        self.category = key
        self.chips[key].setChecked(True)
        self._filter()

    def _filter(self, *_):
        words = self.search.text().lower().split()
        shown = 0
        for i in range(self.list.count()):
            it = self.list.item(i)
            ok = (self.category == 'all' or it.data(ROLE_CAT) == self.category) and all(w in it.data(ROLE_SEARCH) for w in words)
            it.setHidden(not ok)
            shown += ok
        self.count_label.setText(f'{shown} of {self.list.count()}')
        self.empty.setVisible(shown == 0)
        self.list.setVisible(shown > 0)

    def focus_search(self):
        self.search.setFocus()
        self.search.selectAll()

    # -- use ----------------------------------------------------------------------------------------------

    def _use(self, it):
        ref = it.data(ROLE_REF)
        last, t = self._last_use
        if last == ref and time.monotonic() - t < 0.8:   # the second click of a double-click
            return
        self._last_use = (ref, time.monotonic())
        kind, path = ref
        self.current = ref
        self.list.viewport().update()
        if kind == 'builtin':
            self.presetChosen.emit(path, self.keep.isChecked())
        else:
            self.userPresetChosen.emit(path, self.keep.isChecked())

    def _menu(self, pos):
        it = self.list.itemAt(pos)
        if it is None:
            return
        kind, ref = it.data(ROLE_REF)
        m = QMenu(self)
        m.addAction('Load this effect', lambda: self._use(it))
        if kind == 'user':
            m.addAction(icons.glyph_icon('trash', theme.TEXT, 16), 'Delete preset', lambda: self._delete(ref))
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
        self._set_category('mine')
