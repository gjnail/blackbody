"""The main window without a GPU (conftest.py): every preset loads from the effects list with the load card and the
stale-frame guard, Properties' pages, search and Advanced, the effects list's categories and search, the work view, the
notices in the status bar and the compact header."""
import time

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from uikit import drag, pump


def _items(win):
    lib = win.library.list
    return [lib.item(i) for i in range(lib.count())]


def _use(win, key):
    it = next(i for i in _items(win) if i.data(Qt.UserRole) == ('builtin', key))
    win.library._last_use = (None, 0.0)
    win.library._use(it)
    pump()
    win.doc.set_playing(False)


def test_every_preset_loads_from_the_effects_list(win):
    from blackbody.scene import presets
    doc = win.doc
    keys = [it.data(Qt.UserRole)[1] for it in _items(win) if it.data(Qt.UserRole)[0] == 'builtin']
    assert set(keys) == set(presets.ORDER)
    for n, key in enumerate(keys):
        seq = doc.load_seq
        _use(win, key)
        assert doc.scene.preset == key and doc.load_seq > seq
        # (its first frame came back from the worker: the load card is gone)
        pump(0.03)
        assert win.viewport.loading is None, key
        pages = win.props.nav_keys()
        assert 'essentials' in pages and 'objects' in pages
        for pg in (pages if n % 10 == 0 else ['objects']):
            win.props.set_page(pg)
            assert win.props.page == pg, (key, pg)


def test_a_frame_of_the_scene_it_replaced_is_not_shown(win):
    """The engine tags each frame with the scene load it belongs to (load_seq): one that arrives after the next scene
    was asked for keeps the load card up."""
    from uikit import STATS
    from PySide6.QtGui import QImage
    _use(win, 'campfire')
    old = win.doc.load_seq
    _use(win, 'torch')
    win.viewport.begin_load('Torch')
    img = QImage(32, 18, QImage.Format_RGB32)
    win.worker.frameReady.emit(img, 1, dict(STATS, load_seq=old))       # (a direct connection: handled at once)
    assert win.viewport.loading == 'Torch' and win.viewport.image is None
    win.worker.frameReady.emit(img, 1, dict(STATS, load_seq=win.doc.load_seq))
    assert win.viewport.loading is None and win.viewport.image is img


def test_properties_search_and_advanced(win):
    _use(win, 'campfire')
    p = win.props
    counts = {}
    for q in ('haze', 'wind', 'smoke colour', 'temperature', 'xyzzy'):
        p.search.setText(q)
        pump(0.3)
        counts[q] = len(p.panel.rows)
    assert counts['xyzzy'] == 0
    assert counts['wind'] > 0 and counts['temperature'] > 0
    p.search.clear()
    pump(0.3)
    p.set_page('domain')
    pump()
    basic = len(p.panel.rows)
    p.adv.setChecked(True)
    pump()
    advanced = len(p.panel.rows)
    p.adv.setChecked(False)
    pump()
    assert advanced > basic > 0 and len(p.panel.rows) == basic


def test_the_effects_list_filters(win):
    lib = win.library
    total = lib.list.count()
    shown = lambda: sum(not lib.list.isRowHidden(i) for i in range(lib.list.count()))   # noqa: E731
    lib._set_category('sea')
    pump()
    sea = shown()
    assert 0 < sea < total
    lib._set_category('all')
    lib.search.setText('lava')
    pump(0.2)
    lava = [lib.list.item(i).data(Qt.UserRole)[1] for i in range(lib.list.count()) if not lib.list.isRowHidden(i)]
    assert {'lava', 'lava_sea', 'lava_quench'} <= set(lava) and len(lava) < 0.25 * total, lava
    lib.search.clear()
    pump(0.2)
    assert shown() == total


def test_a_number_field_drags_types_and_resets(win):
    _use(win, 'campfire')
    doc, p = win.doc, win.props
    p.set_page('combustion')
    pump()
    row = next(r for r in p.panel.rows if r.path == ('combustion', 'soot'))
    p.scroll.ensureWidgetVisible(row)
    pump()
    f = row.field
    v0 = doc.value(('combustion', 'soot'))
    y = f.height() // 2
    drag(f, QPoint(20, y), QPoint(80, y))
    assert doc.value(('combustion', 'soot')) != v0, 'dragging a number changes it'
    QTest.mouseClick(f, Qt.LeftButton, Qt.NoModifier, QPoint(30, y))
    pump()
    assert f.edit is not None, 'a click opens the editor'
    f.edit.selectAll()
    QTest.keyClicks(f.edit, '3/4')
    QTest.keyClick(f.edit, Qt.Key_Return)
    pump()
    assert abs(doc.value(('combustion', 'soot')) - 0.75) < 1e-9, 'typed arithmetic'
    row.reset_btn.click()
    pump()
    assert doc.value(('combustion', 'soot')) == doc.baseline.data['combustion']['soot']


def test_the_work_view_orbits_and_turns_off(win):
    _use(win, 'campfire')
    doc, vp = win.doc, win.viewport
    win.work_btn.setChecked(True)
    pump()
    assert doc.work_view is not None
    before = repr(doc.work_view)
    drag(vp, QPoint(40, 40), QPoint(200, 60))
    assert repr(doc.work_view) != before, 'a drag orbits it'
    for key in ('open_ocean', 'flag_wind', 'cumulus_day', 'campfire'):   # (scenes of every kind while it is on)
        doc.load_preset(key, keep_shot=False)
        pump()
    win.work_btn.setChecked(False)
    pump()
    assert doc.work_view is None


def test_notices_reach_the_status_bar(win):
    from uikit import STATS
    from PySide6.QtGui import QImage
    _use(win, 'campfire')
    note = 'Only the first 16 objects take part (simulated and drawn): Box 16 is left out.'
    img = QImage(32, 18, QImage.Format_RGB32)
    win.worker.frameReady.emit(img, 1, dict(STATS, load_seq=win.doc.load_seq, notices=[note]))
    b = win.notice_btn
    assert not b.isHidden() and b.text() == '1 notice' and note in b.toolTip()
    win.worker.frameReady.emit(img, 2, dict(STATS, load_seq=win.doc.load_seq))
    assert b.isHidden(), 'gone with the next frame that has none'


def test_the_header_shrinks_to_a_small_screen(win):
    """The window fits a 1366 px screen; the view buttons fold into a list there, and both say the same view."""
    win.resize(1366, 760)
    pump(0.2)
    assert win.minimumSizeHint().width() <= 1366
    win.set_mode('heat')
    pump()
    assert win.mode_combo.currentData() == 'heat' and win.viewport.mode == 'heat'
    assert win.mode_group.checkedButton() is win.mode_group.button(win.mode_combo.currentIndex())
    assert win.worker.posts[-1] == 'mode'


def test_the_window_opens_quickly(qapp, slot_errors, monkeypatch):
    """A window made and shown in well under a few seconds (stand-in worker: the window's own cost)."""
    from PySide6.QtCore import QSettings
    from blackbody.ui.main_window import MainWindow
    import uikit
    QSettings().setValue('ui/clean_exit', True)
    t = time.monotonic()
    w = MainWindow(uikit.FakeWorker())
    w._autosave.stop()
    w.show()
    pump(0.1)
    took = time.monotonic() - t
    w.doc.undo.setClean()
    w.close()
    w.deleteLater()
    pump()
    assert took < 8.0, f'{took:.1f} s'
