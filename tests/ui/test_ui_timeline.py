"""Time and layers through the window (conftest.py): the animation editor's keys, the Timing bars dragged, and layers
added, switched, renamed, hidden, moved, copied, saved, opened, removed and undone."""
from PySide6.QtCore import QPointF

from uikit import drag, pump


def test_keys_are_moved_added_deleted_and_undone(win):
    from blackbody.ui.keys import animated_paths
    doc = win.doc
    doc.load_preset('waved_torch', keep_shot=False)
    pump()
    doc.set_playing(False)
    rows = animated_paths(doc.scene)
    assert rows, 'the preset is animated'
    path = rows[0][0]
    before = doc.scene.curve(path).frames()
    moved = doc.move_keys([(path, before[-1])], 6)
    assert doc.scene.curve(path).frames()[-1] == before[-1] + 6 and moved
    doc.set_key_interp(moved, 'linear')
    doc.add_key(path, int(doc.scene.start + 3))
    assert float(doc.scene.start + 3) in doc.scene.curve(path).frames()
    doc.delete_keys([(path, float(doc.scene.start + 3))])
    assert float(doc.scene.start + 3) not in doc.scene.curve(path).frames()
    for _ in range(4):
        doc.undo.undo()
    assert doc.scene.curve(path).frames() == before
    win.timeline.keys_btn.setChecked(True)
    pump(0.1)
    ke = win.timeline.keys
    assert len(ke.rows) == len(rows)
    # (a row's name shows its setting in Properties: the object's page, or the section's)
    from blackbody.ui.panels import OBJECT_KINDS
    first = ke.rows[0][0]
    win.props.set_page('essentials')
    ke.revealed.emit(first)
    pump(0.3)
    if first[0] in OBJECT_KINDS:
        assert win.props.page == 'objects' and tuple(doc.selection) == (first[0], first[1])
    else:
        assert win.props.page == first[0]
    win.timeline.keys_btn.setChecked(False)


def test_timing_bars_drag_when_things_start_and_stop(win):
    from blackbody.ui import timing as T
    doc = win.doc
    win.add_component('campfire')
    win.add_component('curtain')
    win.add_component('box', at=(0.6, 0.6))
    pump()
    doc.set_playing(False)
    sc = doc.scene
    doc.set(('emitter', 0, 'start'), 1.0, merge=False)
    doc.set(('emitter', 0, 'stop'), 3.0, merge=False)
    doc.add_key(('collider', 0, 'position'), 1)
    doc.set_frame(48)
    doc.set(('collider', 0, 'position'), (0.6, 0.8, 0.6))
    doc.set_frame(1)
    doc.end_drag()
    win.timeline.timing_btn.setChecked(True)
    win.resize(1600, 1000)
    pump(0.2)
    te = win.timeline.timing
    kinds = [k for k, _, _ in te.rows]
    assert 'emitter' in kinds and 'fabric' in kinds and 'collider' in kinds

    def row_y(r):
        return T.HEAD_H + (r - te.scroll) * T.ROW_H + T.ROW_H / 2

    sc = doc.scene
    px_per_s = te._x(sc.start + sc.fps) - te._x(sc.start)
    x_a, x_b, _, _ = te._span(doc.scene.emitters[0])
    drag(te, QPointF((x_a + x_b) / 2, row_y(0)), QPointF((x_a + x_b) / 2 + px_per_s, row_y(0)))
    e0 = doc.scene.emitters[0]
    assert abs(e0['start'] - 2.0) < 0.1 and abs(e0['stop'] - 4.0) < 0.1, 'the bar moves, its length kept'
    x_a, x_b, _, _ = te._span(e0)
    drag(te, QPointF(x_b, row_y(0)), QPointF(x_b + px_per_s * 0.5, row_y(0)))
    assert abs(doc.scene.emitters[0]['stop'] - 4.5) < 0.1 and abs(doc.scene.emitters[0]['start'] - 2.0) < 0.1
    x_a, x_b, _, _ = te._span(doc.scene.emitters[0])
    drag(te, QPointF(x_a, row_y(0)), QPointF(te._x(sc.start) - 40, row_y(0)))
    assert doc.scene.emitters[0]['start'] <= -99, 'dragged off the left: already going when the shot starts'
    x_a, x_b, _, _ = te._span(doc.scene.emitters[0])
    drag(te, QPointF(x_b, row_y(0)), QPointF(te._x(sc.end) + 30, row_y(0)))
    assert doc.scene.emitters[0]['stop'] < 0, 'dragged off the right: on to the end'
    # the box's key at frame 48 moves to 72
    r_c = [n for n, (k, _, _) in enumerate(te.rows) if k == 'collider'][0]
    ky = T.HEAD_H + (r_c - te.scroll) * T.ROW_H + T.ROW_H - 6
    drag(te, QPointF(te._x(48), ky), QPointF(te._x(72), ky))
    assert [k[0] for k in doc.scene.colliders[0]['position'].keys][-1] == 72


def test_layers_are_added_switched_saved_and_undone(win, tmp_path):
    from PySide6.QtCore import Qt
    from blackbody.scene.model import Scene
    doc = win.doc
    win.add_layer()
    pump()
    assert len(doc.layers()) == 2 and doc.active != 'base' and not win.layer_strip.isHidden()
    it = next(win.library.list.item(i) for i in range(win.library.list.count())
              if win.library.list.item(i).data(Qt.UserRole) == ('builtin', 'waterfall'))
    win.library._last_use = (None, 0.0)
    win.library._use(it)
    pump()
    assert doc.scene.preset == 'waterfall' and doc.scene.kind == 'liquid' and doc.scene is not doc.shot, \
        'a preset chosen with a layer active goes into the layer'
    uid = doc.active
    doc.set(('water', 'clarity'), 9.0, merge=False)
    doc.set(('render', 'end'), 99, merge=False)
    assert [s.data['render']['end'] for _, s in doc.layers()] == [99, 99], 'the shot settings are shared'
    assert doc.scene.data['water']['clarity'] == 9.0
    doc.set_active('base')
    pump()
    assert doc.scene is doc.shot
    doc.layer_set(uid, 'name', 'Falls', 'Rename layer')
    doc.layer_set(uid, 'enabled', False, 'Hide layer')
    assert uid not in [u for u, _ in doc.shot.layer_order()]
    doc.layer_set(uid, 'enabled', True, 'Show layer')
    order = [u for u, _ in doc.layers()]
    doc.move_layer(uid, -1)
    assert [u for u, _ in doc.layers()] == list(reversed(order))
    doc.duplicate_layer(uid)
    assert len(doc.layers()) == 3
    p = tmp_path / 'layers.bbfire'
    doc.save(str(p))
    s2 = Scene.load(p)
    assert [s.name for _, s in s2.layer_order(enabled_only=False)] == [s.name for _, s in doc.layers()]
    win.open_path(str(p))
    pump()
    assert len(doc.layers()) == 3 and not win.layer_strip.isHidden()
    doc.remove_layer(doc.layers()[-1][0])
    assert len(doc.layers()) == 2
    doc.undo.undo()
    assert len(doc.layers()) == 3
    doc.undo.redo()
    assert len(doc.layers()) == 2
