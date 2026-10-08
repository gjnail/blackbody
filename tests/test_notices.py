"""The notices (scene/caps.py, Engine.notices): what the engine's caps leave out of a scene, checked from the scene alone,
and what its parts say as they run, gathered for the viewer, the status bar and the command line; Repeat's warning
before it makes copies past a cap. All without a GPU."""
import os
import types

import numpy as np
import pytest

from blackbody.engine.cloth import MAX_FABRICS
from blackbody.engine.matter import MAX_MATS, MatterSpec, material_slots
from blackbody.engine.renderer import MAX_LAMPS
from blackbody.engine.solver import MAX_COLLIDERS, MAX_EMITTERS
from blackbody.engine.strands import MAX_BLADES, MAX_PATCHES, StrandSpec, blades, blades_wanted, budgets
from blackbody.scene import caps, presets
from blackbody.scene.model import Scene


def scene(kind='fire'):
    s = Scene()
    s.data['domain']['kind'] = kind
    s.emitters.clear()        # (a new scene's fuel bed)
    return s


def test_objects_past_the_cap_are_named():
    s = scene()
    for i in range(MAX_COLLIDERS + 2):
        s.add_collider(name=f'Box {i}')
    s.add_collider(name='Off', enabled=False)       # (a disabled one takes no slot)
    got = caps.notices(s)
    assert len(got) == 1 and 'Box 16 and Box 17 are left out' in got[0]
    s.colliders[-2]['enabled'] = False
    s.colliders[-3]['enabled'] = False
    assert caps.notices(s) == []


def test_sources_are_counted_when_they_emit_at_once():
    s = scene()
    for i in range(20):        # one after another: never more than two at once
        s.add_emitter(name=f'Puff {i}', start=0.5 * i, stop=0.5 * i + 0.2, fade_out=0.1)
    assert caps.notices(s) == []
    for e in s.emitters:       # all at once
        e['start'], e['stop'] = -100.0, -1.0
    got = caps.notices(s)
    assert len(got) == 1 and f'At most {MAX_EMITTERS} sources emit at once' in got[0]
    assert 'Puff 16, Puff 17, Puff 18 and Puff 19' in got[0]


def test_lightning_that_sets_fire_is_cut_past_sixteen_sources():
    s = scene()
    for i in range(MAX_EMITTERS):
        s.add_emitter(name=f'Jet {i}')
    s.add_light(name='Bolt', kind='lightning', strike_at=1.0, ignites=True)
    assert any('the fire Bolt sets' in n for n in caps.notices(s))
    s.emitters[0]['enabled'] = False
    assert not any('sources' in n for n in caps.notices(s))


def test_the_water_level_keeps_source_slots_for_itself():
    s = scene('liquid')
    for i in range(12):
        s.add_emitter(name=f'Hose {i}', shape='sphere')
    assert caps.notices(s) == []
    s.data['liquid']['water_level'] = 0.3     # (the box filled to it at the first step, then topped up along 4 sides)
    got = caps.notices(s)
    assert len(got) == 1 and 'the water level keeps 4 of them' in got[0] and 'Hose 11' in got[0]
    assert 'Hose 10' not in got[0]            # (16 - 4 - 1 at the first step: only the 12th is cut)


def test_lights_past_eight_and_lightning_taking_four():
    s = scene()
    for i in range(MAX_LAMPS + 1):
        s.add_light(name=f'Lamp {i}')
    got = caps.notices(s)
    assert len(got) == 1 and 'Lamp 8 lights nothing' in got[0]
    s = scene()
    s.add_light(name='Bolt A', kind='lightning', strike_at=1.0, ignites=False)
    s.add_light(name='Bolt B', kind='lightning', strike_at=1.05, ignites=False)
    s.add_light(name='Street lamp')
    got = caps.notices(s)
    assert len(got) == 1 and 'while it flashes: then Street lamp lights nothing' in got[0]
    s.lights[1]['strike_at'] = 3.0            # (they flash apart: 4 + 1 at a time)
    assert caps.notices(s) == []
    s.lights[1]['strike_at'] = 1.05
    s.lights[0]['strike_at'] = s.lights[1]['strike_at'] = 60.0   # (after the shot)
    assert caps.notices(s) == []


def test_fabrics_past_the_cap():
    s = scene()
    for i in range(MAX_FABRICS + 1):
        s.add_fabric(name=f'Flag {i}')
    got = caps.notices(s)
    assert len(got) == 1 and f'Only the first {MAX_FABRICS} fabrics' in got[0] and 'Flag 16 is left out' in got[0]


def test_grass_patches_past_the_cap_and_the_blades_shared_out_evenly():
    s = scene()
    for i in range(MAX_PATCHES + 1):
        s.add_strands(name=f'Patch {i}', size=(0.2, 0.45, 0.2))
    got = caps.notices(s)
    assert len(got) == 1 and 'Patch 8 is left out' in got[0]
    # two lawns that want more than the budget between them: each thinned by the same share, not the first served and
    # the second left bare
    lawn = StrandSpec(kind='lawn', size=(8.0, 0.08, 8.0))
    most, share = budgets([lawn, lawn])
    assert share < 1.0 and most[0] == most[1] and sum(most) <= MAX_BLADES
    n = [len(blades(lawn, p, k)) for p, k in enumerate(most)]
    assert min(n) > 0.4 * MAX_BLADES
    small = StrandSpec(kind='meadow', size=(0.5, 0.45, 0.5))
    most, share = budgets([lawn, lawn, small])
    assert most[2] == int(blades_wanted(small) * share) > 0
    s = scene()
    s.add_strands(name='Big lawn', kind='lawn', size=(8.0, 0.08, 8.0))
    s.add_strands(name='Another', kind='lawn', size=(8.0, 0.08, 8.0))
    got = caps.notices(s)
    assert len(got) == 1 and f'thinned to {MAX_BLADES:,} blades' in got[0]


def test_matter_slots_that_run_out_are_said():
    # one sand in water: its damp and soaked forms take two more slots
    keys, slot, wet, short = material_slots([MatterSpec(material='sand')], wets=True)
    assert [k[0] for k in keys] == ['sand', 'wet_sand', 'soaked_sand'] and slot == [0] and not short
    # six sands, each its own colour, in water: the fifth gets damp but never soaked, the sixth never gets wet
    s = scene('liquid')
    for i in range(6):
        s.add_matter(name=f'Sand {i}', material='sand', own_colour=True, colour=(0.1 + 0.1 * i, 0.5, 0.3))
    got = caps.notices(s)
    assert len(got) == 1 and f'room for {MAX_MATS - 1} kinds' in got[0]
    assert 'Sand 4 gets damp but never soaked' in got[0] and 'Sand 5 never gets wet' in got[0]
    s.data['domain']['kind'] = 'fire'          # (dry: one slot each)
    assert caps.notices(s) == []
    # past the slots, a source is simulated and drawn as the first
    specs = [MatterSpec(material='sand', colour=(0.05 * i, 0.5, 0.5), name=f'Heap {i}') for i in range(MAX_MATS + 1)]
    keys, slot, _wet, short = material_slots(specs)
    assert slot[-1] == 0 and len(keys) == MAX_MATS - 1
    assert 'Heap 15 and Heap 16 are past them, so simulated and drawn as Heap 0' in short[0]


def test_the_presets_are_within_the_caps():
    # (the wood line-up has 20 objects: its last 4 logs are left out, which the check says)
    said = {k for k in presets.ORDER if caps.notices(presets.make(k))}
    assert said <= {'wood_lineup'}, said


def test_repeat_warns_before_copies_go_past_a_cap():
    s = scene()
    for i in range(10):
        s.add_emitter(name=f'Jet {i}')
    rows = caps.room(s, [('emitter', 0)])
    assert rows == [('sources at once', 10, 1, MAX_EMITTERS, 'left out')]
    assert caps.over(rows, 6) == []
    got = caps.over(rows, 14)
    assert len(got) == 1 and 'That makes 24 sources at once' in got[0] and '8 would be left out' in got[0]
    s.add_light(name='Bolt', kind='lightning', strike_at=0.5, ignites=False)
    s.add_light(name='Lamp')
    rows = caps.room(s, [('light', 0)])
    assert rows[0][1:] == (5, 4, MAX_LAMPS, 'left out')          # (a bolt takes 4 while it flashes)
    assert caps.over(rows, 1)
    s.add_strands(name='Lawn', kind='lawn', size=(4.0, 0.08, 4.0))
    rows = caps.room(s, [('strands', 0)])
    assert [r[0] for r in rows] == ['patches of grass', 'blades of grass']
    assert any('thinned' in t for t in caps.over(rows, 3))
    s.emitters[0]['enabled'] = False                               # (copies of a disabled thing are disabled)
    assert caps.room(s, [('emitter', 0)]) == []


def test_the_repeat_dialog_says_when_copies_go_past_a_cap():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    QtWidgets = pytest.importorskip('PySide6.QtWidgets')
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from blackbody.ui.repeatdialog import RepeatDialog
    d = RepeatDialog(None, 1, (0.3, 0.3), [('sources at once', 10, 1, MAX_EMITTERS, 'left out')])
    d.count.setValue(4)
    assert d.past.isHidden()
    d.count.setValue(20)
    assert not d.past.isHidden() and '14 would be left out' in d.past.text()
    d.pattern.button(1).setChecked(True)        # (a ring's count is all of them: 19 copies)
    assert '13 would be left out' in d.past.text()
    d.deleteLater()
    app.processEvents()


def _engine(**kw):
    """An Engine without a GPU, for what it gathers (notices only reads its parts' state)."""
    from blackbody.engine.engine import Engine
    E = Engine.__new__(Engine)
    E.kind = kw.get('kind', 'fire')
    E._cap_notes = kw.get('caps', [])
    E._col_names = kw.get('names', ['Ball', 'Boat'])
    E._matter = kw.get('matter')
    E._strands = kw.get('strands')
    E.solids = types.SimpleNamespace(warnings=kw.get('solids', []), capped=kw.get('capped', {}))
    E.solver = types.SimpleNamespace(meshes=types.SimpleNamespace(errors=kw.get('meshes', {})))
    E.weather = kw.get('weather')
    E._wx_on = E.weather is not None
    return E


def test_the_engine_gathers_what_its_parts_say_once_each():
    cap = 'Only the first 16 objects take part (simulated and drawn): Box 16 is left out.'
    E = _engine(kind='liquid', caps=[cap],
                matter=types.SimpleNamespace(specs=[1], warnings=['The matter needs 3,000,000 particles; only ...']),
                strands=types.SimpleNamespace(warnings=[cap]),     # (said twice: once)
                solids=['Ball starts inside Post: it is pushed out when it is let go.'], capped={1: 7},
                meshes={'C:/meshes/teapot.obj': 'Mesh not found: C:/meshes/teapot.obj', 'rock.obj': 'bad faces'},
                weather=types.SimpleNamespace(short=1200, fill_short=0, capacity=2_000_000))
    got = E.notices()
    assert got[0] == cap and got.count(cap) == 1
    assert any(n.startswith('The matter needs') for n in got)
    assert any(n.startswith('Ball starts inside') for n in got)
    assert any(n.startswith('Boat: the liquid pushed it harder than 60 m/s² (7 steps)') for n in got)
    assert any('past its particle limit (2 M falling pieces): 1,200 were not made' in n for n in got)
    assert 'Mesh not found: C:/meshes/teapot.obj' in got and 'The mesh rock.obj could not be used: bad faces' in got
    E._ocio_error = 'no such config'
    assert E.notices()[-1].startswith('OCIO could not be used (no such config)')
    # a sky has no matter or grass (left over from another scene, they say nothing)
    E = _engine(kind='cloud', matter=types.SimpleNamespace(specs=[1], warnings=['stale']))
    assert E.notices() == []
    # matter with no sources says nothing either
    E = _engine(matter=types.SimpleNamespace(specs=[], warnings=['stale']))
    assert E.notices() == []


def test_the_command_line_prints_each_notice_once(capsys):
    from blackbody.cli import _notices
    s = scene()
    for i in range(MAX_COLLIDERS + 1):
        s.add_collider(name=f'Box {i}')
    said = _notices([('base', s)])
    E = _engine(caps=caps.notices(s), capped={0: 2}, names=[c['name'] for c in s.colliders])
    _notices([('base', s)], types.SimpleNamespace(base=E, extra={}), said)
    out = capsys.readouterr().out.splitlines()
    assert out[0] == 'Note: Only the first 16 objects take part (simulated and drawn): Box 16 is left out.'
    assert len(out) == 2 and out[1].startswith('Note: Box 0: the liquid pushed it harder')


def test_solids_count_when_the_liquids_push_is_held_back():
    pytest.importorskip('mujoco')
    from blackbody.engine.solids import Solids
    s = scene('liquid')
    s.add_collider(name='Ball', shape='sphere', position=(0.0, 0.5, 0.0), size=(0.1, 0.1, 0.1), floating=True)
    S = Solids()
    S.configure(s, ((32, 32, 32), 2.0 / 32, (-1.0, 0.0, -1.0)))
    S.reset()
    bd = S.bodies[0]
    bd.hydro = (np.array([0.0, 1.0e6, 0.0]), np.zeros(3), 1.0, np.zeros(3), 2.0)   # (a frame of bad pressure)
    S._forces()
    assert S.capped == {0: 1}
    bd.hydro = (np.zeros(3), np.zeros(3), 1.0, np.zeros(3), 2.0)
    S._forces()
    assert S.capped == {0: 1}
    S.reset()
    assert S.capped == {}
