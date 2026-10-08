"""Searching everything (Ctrl+K) and drawing roto shapes, through the window (conftest.py)."""
from collections import Counter

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest

from uikit import drag, pump


def test_search_everything_finds_and_runs(win):
    from blackbody.ui import palette as P
    doc = win.doc
    win.add_component('box')
    pump()
    doc.set_playing(False)
    entries = P.gather(win, [])
    kinds = Counter(e.kind for e in entries)
    assert len(kinds) >= 3 and all(n > 0 for n in kinds.values()), kinds
    assert P.rank(entries, 'xyzzy') == []
    for q in ('campfire', 'wind', 'render', 'repeat'):
        assert P.rank(entries, q), q
    assert any('campfire' in (e.title + e.hint).lower() for e in P.rank(entries, 'campfire')[:3])
    # Ctrl+K, type, Enter: the first one runs
    win.search_everything()
    pump()
    pal = win._palette
    QTest.keyClicks(pal.box, 'make it fl')
    pump()
    first = pal.list.item(0).data(Qt.UserRole)
    assert 'float' in first.title.lower(), first.title
    QTest.keyClick(pal.box, Qt.Key_Return)
    pump(0.2)
    assert doc.scene.colliders[0].get('floating'), 'it ran: the box floats'
    # the arrow keys move through the list, Escape closes it
    win.search_everything()
    pump()
    pal = win._palette
    QTest.keyClicks(pal.box, 'campfire')
    pump()
    row = pal.list.currentRow()
    QTest.keyClick(pal.box, Qt.Key_Down)
    pump()
    assert pal.list.currentRow() == min(row + 1, pal.list.count() - 1)
    QTest.keyClick(pal.box, Qt.Key_Escape)
    pump()
    assert not pal.isVisible()
    # a setting: its page in Properties
    win.search_everything()
    pump()
    pal = win._palette
    QTest.keyClicks(pal.box, 'wind speed')
    pump()
    top = pal.list.currentItem().data(Qt.UserRole)
    assert top.kind == 'set', (top.kind, top.title)
    sec = top.key.split(':')[1]
    win.props.set_page('essentials')
    QTest.keyClick(pal.box, Qt.Key_Return)
    pump(0.5)
    assert win.props.page == sec


def test_roto_shapes_are_drawn_moved_keyed_and_deleted(win):
    doc, vp = win.doc, win.viewport
    doc.load_preset('campfire', keep_shot=False)
    pump()
    doc.set_playing(False)
    win.roto_btn.setChecked(True)
    pump()
    r = vp.frame_rect()
    pts = [(r.x() + r.width() * a, r.y() + r.height() * b) for a, b in ((0.2, 0.2), (0.6, 0.25), (0.5, 0.8))]
    for x, y in pts:
        QTest.mouseClick(vp, Qt.LeftButton, Qt.NoModifier, QPoint(int(x), int(y)))
    QTest.mouseClick(vp, Qt.LeftButton, Qt.NoModifier, QPoint(int(pts[0][0]), int(pts[0][1])))   # (closed on its first)
    pump()
    assert len(doc.scene.roto) == 1 and len(doc.scene.roto[0]['keys'][str(doc.frame)]) == 3
    # moving a point at another frame keys the shape there
    doc.set_frame(doc.scene.start + 10)
    pump()
    x, y = pts[1]
    drag(vp, QPointF(x, y), QPointF(x + 30, y + 10), steps=1)
    assert len(doc.scene.roto[0]['keys']) == 2
    vp.rotobar.inv.setChecked(True)
    vp.rotobar.feather.setValue(6.0)
    pump()
    assert doc.scene.roto[0]['invert'] and doc.scene.roto[0]['feather'] == 6.0
    vp.roto_delete()
    pump()
    assert doc.scene.roto == []
    doc.undo.undo()
    assert len(doc.scene.roto) == 1
    win.roto_btn.setChecked(False)
