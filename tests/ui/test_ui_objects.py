"""Things in the scene through the window (conftest.py): added, selected (one, or several with Ctrl and a box drag), renamed,
switched off, removed and the steps undone and redone; Repeat in a ring and a row, groups, duplicate and delete; text
shapes; your own blocks saved, found, used and shared."""
import shutil

import numpy as np
from PySide6.QtCore import QPointF, Qt

from uikit import drag, pump


def _counts(sc):
    return (len(sc.emitters), len(sc.colliders), len(sc.lights), len(sc.fabrics))


def test_objects_are_added_selected_renamed_removed_and_undone(win):
    from blackbody.ui.panels import add_fabric
    doc = win.doc
    doc.load_preset('campfire', keep_shot=False)
    pump()
    start = _counts(doc.scene)
    ne = start[0]
    doc.add_emitter('sphere')
    doc.add_collider('box')
    doc.add_light('spot')
    add_fabric(doc, 'flag')
    pump()
    assert _counts(doc.scene) == (start[0] + 1, start[1] + 1, start[2] + 1, start[3] + 1)
    assert win.props.page == 'objects' and doc.selection == ('fabric', 0)
    tree = win.props.objects.tree
    listed = sum(tree.topLevelItem(i).childCount() for i in range(tree.topLevelItemCount()))
    assert listed >= sum(_counts(doc.scene))
    doc.select(('emitter', ne))
    doc.rename('emitter', ne, 'Extra')
    doc.set(('emitter', ne, 'enabled'), False, merge=False)
    assert doc.scene.emitters[ne]['name'] == 'Extra' and not doc.scene.emitters[ne]['enabled']
    doc.remove_emitter(ne)
    pump()
    assert len(doc.scene.emitters) == start[0]
    done = _counts(doc.scene)
    for _ in range(7):
        doc.undo.undo()
    pump()
    assert _counts(doc.scene) == start
    for _ in range(7):
        doc.undo.redo()
    pump()
    assert _counts(doc.scene) == done
    # a click on a thing in the viewer (here: selecting it) shows its page
    win.props.set_page('essentials')
    doc.select(('emitter', 0), force=True)
    pump()
    assert win.props.page == 'objects'


def _screen(win, kind, i):
    vp = win.viewport
    cs, fire, _ = vp.camstate()
    px, ok = vp._project_local(cs, fire, [win.doc.scene.get((kind, i, 'position'), win.doc.frame)])
    return vp.to_widget(px[0])


def test_several_things_are_selected_moved_turned_and_repeated(win, monkeypatch):
    from blackbody.scene import arrange
    from blackbody.ui import actions as A, repeatdialog
    doc, vp = win.doc, win.viewport
    win.add_component('torch', at=(-0.5, 0.0))
    win.add_component('box', at=(0.5, 0.3))
    win.add_component('lamp')
    pump()
    doc.set_playing(False)
    sc = doc.scene
    assert len(sc.emitters) == 1 and len(sc.colliders) == 1 and len(sc.lights) == 1
    doc.select(('emitter', 0), force=True)
    pump()
    pb = _screen(win, 'collider', 0)
    drag(vp, pb, pb, Qt.ControlModifier, steps=1)
    assert set(doc.selected_objects()) == {('emitter', 0), ('collider', 0)} and doc.selection == ('collider', 0)
    drag(vp, pb, pb, Qt.ControlModifier, steps=1)
    assert doc.selected_objects() == [('emitter', 0)], 'Ctrl+click again takes it out'
    # a box drag with Shift round both
    pt = _screen(win, 'emitter', 0)
    lo = QPointF(min(pt.x(), pb.x()) - 30, min(pt.y(), pb.y()) - 30)
    hi = QPointF(max(pt.x(), pb.x()) + 30, max(pt.y(), pb.y()) + 30)
    drag(vp, lo, hi, Qt.ShiftModifier)
    sel = set(doc.selected_objects())
    assert {('emitter', 0), ('collider', 0)} <= sel
    # the gizmo's x arrow moves them all by the same amount, in one undo step
    p0 = {s: np.asarray(sc.get((s[0], s[1], 'position'), doc.frame), float) for s in sel}
    hs = vp._gizmo().handles()
    a, b = hs['move_x']
    tip = a + (b - a) * 0.8
    drag(vp, tip, tip + (b - a) * 0.6)
    moved = [np.asarray(doc.scene.get((s[0], s[1], 'position'), doc.frame), float) - p0[s] for s in sel]
    assert abs(moved[0][0]) > 1e-3 and all(np.allclose(m, moved[0], atol=1e-6) for m in moved)
    doc.undo.undo()
    back = [np.asarray(doc.scene.get((s[0], s[1], 'position'), doc.frame), float) - p0[s] for s in sel]
    assert all(np.allclose(m, 0.0, atol=1e-6) for m in back)
    # Repeat: a ring of 6
    monkeypatch.setattr(repeatdialog, 'ask', lambda parent, n, size, room=None: (arrange.layout('ring', 6, radius=1.0), True))
    doc.set_selected([('emitter', 0), ('collider', 0)], ('emitter', 0))
    A.repeat(win, doc.selected_objects())
    sc = doc.scene
    assert (len(sc.emitters), len(sc.colliders)) == (6, 6) and len(doc.selected_objects()) == 12
    assert len({e['seed'] for e in sc.emitters}) == 6, 'each copy burns its own way'
    doc.undo.undo()
    assert (len(doc.scene.emitters), len(doc.scene.colliders)) == (1, 1)
    # grouped with its light, a torch repeated in a row brings the light along
    doc.set_selected([('emitter', 0), ('light', 0)], ('emitter', 0))
    A.group(win)
    assert doc.scene.links
    doc.select(('emitter', 0), force=True)
    monkeypatch.setattr(repeatdialog, 'ask', lambda parent, n, size, room=None: (arrange.layout('row', 3, spacing=0.6), False))
    A.repeat(win)
    sc = doc.scene
    assert len(sc.emitters) == 4 and len(sc.lights) == 4 and len(sc.links) == 4
    xs = sorted(round(sc.get(('emitter', i, 'position'), doc.frame)[0], 2) for i in range(4))
    assert np.allclose(np.diff(xs), 0.6, atol=0.02)
    # Ctrl+A, duplicate, delete, ungroup
    win._select_all()
    n0 = sum(_counts(doc.scene))
    assert len(doc.selected_objects()) == n0
    win._on_selection(lambda s: doc.duplicate_objects(s))
    assert sum(_counts(doc.scene)) == 2 * n0
    win._on_selection(lambda s: doc.delete_objects(s))
    assert sum(_counts(doc.scene)) == n0
    doc.ungroup(('emitter', 0))
    assert len(doc.scene.links) == 3


def test_the_repeat_dialog_lays_things_out(win):
    from blackbody.ui import repeatdialog
    d = repeatdialog.RepeatDialog(win, 2, (0.4, 0.3))
    d.pattern.button(1).setChecked(True)
    d.count.setValue(8)
    pump()
    places, ring = d.places()
    assert len(places) == 8
    d.pattern.button(2).setChecked(True)
    pump()
    assert len(d.places()[0]) == 8
    d.close()


def test_your_blocks_are_saved_found_used_and_shared(win, monkeypatch, tmp_path):
    from blackbody.scene import blocks
    from blackbody.ui import actions as A, blockdialog, textdialog, textmesh
    doc = win.doc
    cr = win.create
    cr._set_domain('mine')
    pump()
    assert cr.mine_empty.isVisibleTo(cr) and not cr.mine[2], 'Yours starts empty, with a hint'
    monkeypatch.setattr(textdialog, 'ask', lambda *a, **k: textmesh.spec_of('OPEN', 'Arial Black', False, False, 0.3, 0.08))
    win.add_component('text_fire')
    win.add_component('lamp')
    pump()
    doc.set_playing(False)
    sc = doc.scene
    assert sc.colliders and A.text_spec(sc.colliders[0]) is not None, 'a text shape, editable as text'
    monkeypatch.setattr(blockdialog, 'ask', lambda parent, scene, sel: ('Open sign', 'A burning OPEN sign'))
    A.save_block(win, [('collider', 0), ('light', 0)])
    pump()
    tiles = cr.mine[2]
    assert [t.comp.name for t in tiles] == ['Open sign'] and tiles[0].comp.tip
    cr.search.setText('sign')
    pump()
    assert tiles[0].isVisibleTo(cr) and not cr.mine_empty.isVisibleTo(cr)
    cr.search.setText('')
    # used in a new scene, where you drop it
    win.new_from_scratch('auto', 'person')
    pump()
    win.add_component(cr.mine[2][0].comp.key, at=(0.5, -0.4))
    pump()
    doc.set_playing(False)
    sc = doc.scene
    assert len(sc.colliders) == 1 and len(sc.lights) == 1
    assert A.text_spec(sc.colliders[0]) is not None
    # shared as a file and imported on another machine
    share = tmp_path / 'Open_sign_copy.bbblock'
    shutil.copy2(cr.mine[2][0].comp.file, share)
    blocks.delete(cr.mine[2][0].comp.key)
    cr.reload_blocks()
    assert not cr.mine[2]
    key = cr.import_block(str(share))
    assert key and [t.comp.name for t in cr.mine[2]] == ['Open sign']
