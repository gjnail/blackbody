"""Search everything (Ctrl+K): one box for what you can do to the selected thing, the building blocks, the menu
commands, every setting, the ready-made effects and the things in the scene. Type a few letters, Enter runs it."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from PySide6.QtCore import QEvent, QRectF, QSettings, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QKeySequence, QPainter, QPen
from PySide6.QtWidgets import (QDialog, QFrame, QLineEdit, QListWidget, QListWidgetItem, QMenu, QStyledItemDelegate, QStyle,
                               QVBoxLayout)

from . import icons, theme

KINDS = {'do': ('Do', theme.ACCENT, 0), 'add': ('Add', '#9fe0c8', 1), 'command': ('Command', '#c9c9d1', 2),
         'object': ('Select', '#9fb0c4', 3), 'set': ('Setting', '#a8c8ff', 4), 'effect': ('Effect', '#f0c674', 5)}
RECENT_KEY = 'ui/palette_recent'


@dataclass
class Entry:
    kind: str
    title: str
    hint: str
    run: Callable
    words: str = ''
    glyph: str = ''
    shortcut: str = ''
    key: str = field(default='')

    def __post_init__(self):
        self.key = self.key or f'{self.kind}:{self.hint}:{self.title}'
        self.hay = ' '.join((self.title, self.hint, self.words)).lower()


def _menu_entries(menu, trail, kind, out, skip=()):
    for a in menu.actions():
        if a.isSeparator():
            continue
        sub = a.menu()
        text = a.text().replace('&', '').strip()
        if sub is not None:
            _menu_entries(sub, trail + [text], kind, out, skip)
            continue
        if not text or not a.isEnabled() or text in skip:
            continue
        sc = a.shortcut().toString(QKeySequence.NativeText) if not a.shortcut().isEmpty() else ''
        title = ' › '.join(trail[1:] + [text]) if kind == 'do' else text
        if a.isCheckable():
            title += ' (on)' if a.isChecked() else ' (off)'
        out.append(Entry(kind, title, ' › '.join(trail[:1]) if kind == 'do' else ' › '.join(trail), a.trigger, a.toolTip(),
                         shortcut=sc))


def gather(win, keep=None):
    """Everything there is to search, now (the selection, the scene's kind and your blocks decide what is in it). Menus
    made for it are put in `keep`: their actions live as long as they do."""
    from ..scene import blocks, components, presets
    from ..scene.params import SECTION_TITLES, SECTIONS, applies
    from . import actions
    from .panels import section_order, worded
    doc = win.doc
    sc = doc.scene
    out = []
    # what can be done to the selected thing
    sel = doc.selection
    if sel and sel[0] in ('emitter', 'collider', 'light', 'fabric'):
        m = QMenu(win)
        actions.fill_menu(m, win, sel, path_mode=getattr(getattr(win, 'viewport', None), 'start_path', None))
        items = {'emitter': sc.emitters, 'collider': sc.colliders, 'light': sc.lights, 'fabric': sc.fabrics}[sel[0]]
        _menu_entries(m, [items[sel[1]]['name']], 'do', out)
        if keep is not None:
            keep.append(m)
    # building blocks
    for c in list(components.COMPONENTS) + blocks.all_blocks():
        if components.target_kind(sc, c) is None:
            continue
        out.append(Entry('add', c.name, 'Yours' if c.group == 'Mine' else c.group,
                         (lambda k=c.key: win.add_component(k)), c.tip, glyph=c.glyph, key=f'add:{c.key}'))
    # the menus
    _menu_entries(win.menuBar(), [], 'command', out, skip=('Search everything…',))
    # the things in the scene
    for kind, items in (('emitter', sc.emitters), ('collider', sc.colliders), ('light', sc.lights), ('fabric', sc.fabrics)):
        for i, d in enumerate(items):
            out.append(Entry('object', d['name'], {'emitter': 'Source', 'collider': 'Object', 'light': 'Light', 'fabric': 'Fabric'}[kind],
                             (lambda s=(kind, i): (doc.select(s, force=True), _show_props(win))), glyph=kind,
                             key=f'object:{kind}:{d["name"]}'))
    # every setting of the scene
    for sec in section_order(sc.kind):
        if not applies(sec, None, sc.kind):
            continue
        for p in worded(sec, [p for p in SECTIONS[sec] if applies(sec, p.key, sc.kind)], sc.kind):
            out.append(Entry('set', p.label, f'{SECTION_TITLES[sec]} › {p.group}',
                             (lambda path=(sec, p.key): _reveal(win, path)), f'{p.key.replace("_", " ")} {p.tip}',
                             key=f'set:{sec}:{p.key}'))
    # the ready-made effects
    for name in presets.ORDER:
        info = presets.PRESETS.get(name, {})
        out.append(Entry('effect', info.get('name', name), info.get('category', 'Effects'),
                         (lambda n=name: _load(win, n)), f'{info.get("size", "")} {info.get("blurb", "")}', key=f'effect:{name}'))
    return out


def _show_props(win):
    if hasattr(win, 'd_props'):
        win.d_props.show()
        win.d_props.raise_()


def _reveal(win, path):
    _show_props(win)
    win.props.reveal(path)


def _load(win, name):
    keep = win.library.keep.isChecked() and win.library.keep.isEnabled()
    if hasattr(win, 'focus_effects'):
        win.focus_effects()
    win._load_preset(name, keep)


def rank(entries, query, recent=()):
    """The entries matching a query, best first. Every word must be found; the title counts most."""
    q = query.lower().strip()
    words = q.split()
    rec = {k: n for n, k in enumerate(recent)}
    scored = []
    for e in entries:
        t = e.title.lower()
        if words:
            if not all(w in e.hay for w in words):
                continue
            starts = [x for x in t.replace('›', ' ').split()]
            if t == q or t.rstrip('…') == q:
                s = -1   # exactly what was typed
            elif t.startswith(q):
                s = 0
            elif all(any(x.startswith(w) for x in starts) for w in words):
                s = 0.5   # each word starts a word of the title
            elif all(w in t for w in words):
                s = 1
            else:
                s = 2 if all(w in (t + ' ' + e.hint.lower()) for w in words) else 3
            if s == 3 and len(q) < 3:
                continue
            if e.kind == 'do':
                s -= 0.6   # what can be done to the selected thing comes first
        else:
            if e.key not in rec and e.kind != 'do':
                continue
            s = 0
        boost = -0.6 + 0.01 * rec[e.key] if e.key in rec else 0.0
        scored.append((s + boost, KINDS[e.kind][2], len(e.title), e))
    scored.sort(key=lambda x: x[:3])
    return [e for *_, e in scored]


class EntryDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        return QSize(option.rect.width(), 38)

    def paint(self, p, option, index):
        e = index.data(Qt.UserRole)
        if e is None:
            return
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(option.rect).adjusted(4, 1, -4, -1)
        if option.state & QStyle.State_Selected:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(theme.FIELD_HI))
            p.drawRoundedRect(r, 6, 6)
        label, colour, _ = KINDS[e.kind]
        glyph = e.glyph if e.glyph in icons.GLYPHS else {'do': 'star', 'add': 'plus', 'command': 'grid', 'object': 'emitter',
                                                         'set': 'palette', 'effect': 'book'}[e.kind]
        icons.paint_glyph(p, glyph, QRectF(r.x() + 8, r.center().y() - 8, 16, 16), colour, 1.4)
        f = QFont(option.font)
        p.setFont(f)
        p.setPen(QColor(theme.TEXT))
        right = 150
        tw = r.width() - 40 - right
        title = p.fontMetrics().elidedText(e.title, Qt.ElideRight, int(tw * 0.62))
        p.drawText(QRectF(r.x() + 34, r.y(), tw, r.height()), Qt.AlignVCenter | Qt.AlignLeft, title)
        used = p.fontMetrics().horizontalAdvance(title)
        f2 = QFont(f)
        f2.setPointSizeF(f.pointSizeF() * 0.9)
        p.setFont(f2)
        p.setPen(QColor(theme.FAINT))
        hint = p.fontMetrics().elidedText(e.hint, Qt.ElideRight, int(max(0, tw - used - 12)))
        p.drawText(QRectF(r.x() + 34 + used + 10, r.y(), tw - used - 10, r.height()), Qt.AlignVCenter | Qt.AlignLeft, hint)
        if e.shortcut:
            p.setPen(QColor(theme.MUTED))
            p.drawText(QRectF(r.right() - right, r.y(), right - 70, r.height()), Qt.AlignVCenter | Qt.AlignRight, e.shortcut)
        badge = QRectF(r.right() - 62, r.center().y() - 9, 56, 18)
        p.setPen(QPen(QColor(colour), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(badge, 9, 9)
        p.setPen(QColor(colour))
        p.drawText(badge, Qt.AlignCenter, label)
        p.restore()


class Palette(QDialog):
    def __init__(self, win):
        super().__init__(win, Qt.FramelessWindowHint | Qt.Popup)
        self.win = win
        self.setObjectName('palette')
        self.setStyleSheet(f'QDialog#palette {{ background: {theme.PANEL}; border: 1px solid {theme.LINE_HI}; border-radius: 10px; }}')
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(6)
        self.box = QLineEdit()
        self.box.setObjectName('search')
        self.box.setPlaceholderText('Search everything: set on fire, add text, wind speed, campfire, save…')
        self.box.addAction(icons.glyph_icon('search', theme.FAINT, 16), QLineEdit.LeadingPosition)
        f = QFont(self.box.font())
        f.setPointSizeF(f.pointSizeF() * 1.15)
        self.box.setFont(f)
        self.box.setMinimumHeight(36)
        v.addWidget(self.box)
        self.list = QListWidget()
        self.list.setFrameShape(QFrame.NoFrame)
        self.list.setItemDelegate(EntryDelegate(self.list))
        self.list.setStyleSheet('QListWidget { background: transparent; }')
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        v.addWidget(self.list, 1)
        self._menus = []
        self.entries = gather(win, self._menus)
        self.recent = QSettings().value(RECENT_KEY, [], type=list) or []
        self.box.textChanged.connect(self._fill)
        self.box.installEventFilter(self)
        self.list.itemActivated.connect(self._run)
        self.list.itemClicked.connect(self._run)
        self._fill('')
        w = min(720, max(480, win.width() - 200))
        self.resize(w, 440)
        g = win.geometry()
        self.move(g.x() + (g.width() - w) // 2, g.y() + 70)

    def _fill(self, text):
        self.list.clear()
        found = rank(self.entries, text, self.recent)[:80]
        for e in found:
            it = QListWidgetItem()
            it.setData(Qt.UserRole, e)
            it.setToolTip(e.words[:300] if e.words else e.title)
            self.list.addItem(it)
        if not found:
            it = QListWidgetItem()
            it.setFlags(Qt.NoItemFlags)
            it.setText('   Nothing matches. Try another word.' if text.strip() else '   Type to search.')
            self.list.addItem(it)
        else:
            self.list.setCurrentRow(0)

    def eventFilter(self, obj, ev):
        if obj is self.box and ev.type() == QEvent.KeyPress:
            k = ev.key()
            if k in (Qt.Key_Down, Qt.Key_Up, Qt.Key_PageDown, Qt.Key_PageUp):
                n = self.list.count()
                if n:
                    step = {Qt.Key_Down: 1, Qt.Key_Up: -1, Qt.Key_PageDown: 8, Qt.Key_PageUp: -8}[k]
                    self.list.setCurrentRow(max(0, min(n - 1, self.list.currentRow() + step)))
                return True
            if k in (Qt.Key_Return, Qt.Key_Enter):
                it = self.list.currentItem()
                if it is not None:
                    self._run(it)
                return True
            if k == Qt.Key_Escape:
                self.reject()
                return True
        return super().eventFilter(obj, ev)

    def _run(self, it):
        e = it.data(Qt.UserRole)
        if e is None:
            return
        rec = [e.key] + [k for k in self.recent if k != e.key]
        QSettings().setValue(RECENT_KEY, rec[:12])
        self.accept()
        QTimer.singleShot(0, e.run)   # after the box has gone (some open dialogs of their own)


def show(win):
    p = Palette(win)
    p.show()
    p.box.setFocus()
    return p
