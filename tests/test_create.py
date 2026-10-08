"""Building a scene from scratch (scene/components.py): what a scene becomes as blocks go in. A fire gets the time
scale that makes its gas move at the speed of real fire, and hot things in a liquid make it a fire-and-liquid scene so
their steam shows. GPU-free."""
import pytest

from blackbody.scene import components as C, presets
from blackbody.scene.anim import Curve


def ts(sc):
    return sc.data['domain']['time_scale']


@pytest.mark.parametrize('scale', ['small', 'person', 'large'])
def test_a_fire_from_scratch_runs_at_the_speed_of_real_fire(scale):
    sc = C.new_scene('auto', scale)
    assert ts(sc) == 1.0                                   # (nothing in it yet: nothing to speed up)
    _, notes = C.add(sc, 'campfire')
    assert ts(sc) == C.FIRE_TIME_SCALE and any('Time scale' in n for n in notes)
    # as the fire presets are: between the gentlest and the fastest of their corrections
    pool_like = [presets.PRESETS[k]['domain']['time_scale'] for k in ('campfire', 'bonfire', 'torch', 'pool_fire', 'vehicle_fire')]
    assert min(pool_like) <= C.FIRE_TIME_SCALE <= max(pool_like)
    # a second fire leaves it as it is, and so does a fire in a fire-and-smoke scene made from scratch
    C.add(sc, 'burner', at=(0.6, 0.0))
    assert ts(sc) == C.FIRE_TIME_SCALE
    fire = C.new_scene('fire', scale)
    C.add(fire, 'pool')
    assert ts(fire) == C.FIRE_TIME_SCALE


def test_only_steady_fires_get_it_and_a_time_scale_of_your_own_stays():
    sc = C.new_scene('auto', 'person')
    for key in ('jet', 'fireball', 'fan', 'sparks'):   # (jets, bursts and forces: no puffing to match)
        C.add(sc, key)
    assert ts(sc) == 1.0
    sc = C.new_scene('auto', 'person')
    sc.data['domain']['time_scale'] = 0.5             # slow motion
    C.add(sc, 'burner')
    assert ts(sc) == 0.5
    sc = C.new_scene('auto', 'person')
    sc.data['domain']['time_scale'] = Curve([[1.0, 1.0, 'smooth'], [48.0, 0.3, 'smooth']])
    C.add(sc, 'burner')
    assert isinstance(ts(sc), Curve)
    # a preset keeps its own (the Bonfire's happens to be the same), with falling things too
    for name in ('campfire', 'bonfire'):
        sc = presets.make(name)
        was = ts(sc)
        C.add(sc, 'drop_box')
        assert ts(sc) == was, name


REAL_SPEED = ('drop_box', 'flag', 'sand_pile', 'lawn', 'pistol', 'wrecking_ball', 'pour', 'pond',
              'rain', 'snow', 'sleet', 'hail', 'freezing', 'crate', 'stone')   # (weather and floating things too)


@pytest.mark.parametrize('key', REAL_SPEED)
def test_things_that_move_at_their_real_speed_take_the_fire_time_scale_away(key):
    sc = C.new_scene('auto', 'person')
    C.add(sc, 'campfire')
    _, notes = C.add(sc, key)
    assert ts(sc) == 1.0 and any('Time scale is back to 1' in n for n in notes), (sc.kind, ts(sc))
    # with them there first, a fire does not get it (snow on a campfire: the snow falls at its real speed)
    sc = C.new_scene('auto', 'person')
    C.add(sc, key)
    C.add(sc, 'campfire')
    assert ts(sc) == 1.0, sc.kind


def test_water_and_fire_together_run_at_the_speed_of_the_water():
    sc = C.new_scene('liquid', 'person')
    C.add(sc, 'pour')
    C.add(sc, 'burner')
    assert sc.kind == 'both' and ts(sc) == 1.0
    # a fire turned into water takes its correction with it
    sc = C.new_scene('auto', 'person')
    C.add(sc, 'campfire')
    notes = C.turn_into(sc, 0, 'pour')
    assert sc.kind == 'both' and ts(sc) == 1.0 and any('Time scale is back to 1' in n for n in notes)
    # water keyed to rise (a tide) counts as water
    sc = C.new_scene('liquid', 'person')
    sc.data['liquid']['water_level'] = Curve([[1.0, 0.0, 'smooth'], [48.0, 0.3, 'smooth']])
    assert C.has_liquid(sc) and C.real_speed(sc)
    C.add(sc, 'campfire')
    assert sc.kind == 'both' and ts(sc) == 1.0


@pytest.mark.parametrize('verb', ['fall', 'rope', 'break'])
def test_the_viewers_verbs_that_make_something_fall_take_it_away(verb):
    sc = C.new_scene('auto', 'person')
    C.add(sc, 'campfire')
    added, _ = C.add(sc, 'box')
    assert ts(sc) == C.FIRE_TIME_SCALE                  # (a box that stays put)
    i = added[0][1]
    notes = {'fall': lambda: C.make_dynamic(sc, i), 'rope': lambda: C.add_joint(sc, i, 'rope'),
             'break': lambda: C.make_breakable(sc, i)}[verb]()
    assert ts(sc) == 1.0 and any('Time scale' in n for n in notes)


def test_a_time_scale_of_your_own_stays_where_no_fire_could_have_set_it():
    # water at 1.5 by hand, with things added that move at their real speed: it was not the fire's, so it stays
    sc = C.new_scene('liquid', 'person')
    C.add(sc, 'pour')
    sc.data['domain']['time_scale'] = 1.5
    for key in ('box', 'drop_box', 'stone'):
        _, notes = C.add(sc, key)
        assert ts(sc) == 1.5 and not any('Time scale' in n for n in notes), key
    C.make_dynamic(sc, len(sc.colliders) - 1)
    assert ts(sc) == 1.5
    # and with a fire added to it (the fire found it at 1.5)
    C.add(sc, 'burner')
    assert sc.kind == 'both' and ts(sc) == 1.5
    # smoke at 1.5 by hand (smoke does not puff: it never had the fire's correction)
    sc = C.new_scene('fire', 'person')
    C.add(sc, 'smoke')
    sc.data['domain']['time_scale'] = 1.5
    added, _ = C.add(sc, 'box')
    C.make_dynamic(sc, added[0][1])
    C.add(sc, 'flag')
    assert ts(sc) == 1.5


def test_turning_a_source_into_fire_paces_it_too():
    sc = C.new_scene('auto', 'person')
    C.add(sc, 'smoke')
    assert ts(sc) == 1.0
    C.turn_into(sc, 0, 'burner')
    assert ts(sc) == C.FIRE_TIME_SCALE


def test_a_hot_plate_makes_a_liquid_scene_a_fire_and_liquid_one():
    sc = C.new_scene('liquid', 'person')
    C.add(sc, 'pour')
    plate = C.BY_KEY['hot_plate']
    assert C.target_kind(sc, plate) == 'both'             # (what Create shows before it goes in)
    _, notes = C.add(sc, 'hot_plate')
    assert sc.kind == 'both' and sc.emitters[0]['emits'] == 'liquid'   # the water is still water
    assert any('steam' in n for n in notes)
    assert any('Heat and phase changes' in n for n in notes)            # (it is off: nothing boils yet)
    # with heat on there is nothing more to say
    sc = C.new_scene('liquid', 'person')
    sc.data['liquid']['thermal'] = True
    _, notes = C.add(sc, 'hot_plate')
    assert sc.kind == 'both' and not any('Heat and phase changes' in n for n in notes)
    # something cold leaves it a liquid scene
    sc = C.new_scene('liquid', 'person')
    C.add(sc, 'box')
    assert sc.kind == 'liquid'


def test_water_on_something_hot_makes_a_fire_and_liquid_scene():
    # the hot plate first, in an empty scene: no water yet, so nothing to boil
    sc = C.new_scene('auto', 'person')
    C.add(sc, 'hot_plate')
    assert sc.kind == 'fire'
    _, notes = C.add(sc, 'pour')
    assert sc.kind == 'both' and any('steam' in n for n in notes)
    # water poured into a liquid scene with an object made hot by hand
    sc = C.new_scene('liquid', 'person')
    added, _ = C.add(sc, 'box')
    sc.colliders[added[0][1]]['temperature'] = 250.0
    assert C.boils(sc) and not C.boils(C.new_scene('liquid', 'person'))
    C.add(sc, 'pour')
    assert sc.kind == 'both'
    # past the liquid's own boiling point only
    sc = C.new_scene('liquid', 'person')
    sc.data['liquid']['boil_point'] = 210.0
    C.add(sc, 'hot_plate')                                 # (200 C)
    assert sc.kind == 'liquid'


# ---- the same from the outliner's Add menu and Properties (ui/document.py; Qt, no GPU) --------------------------------

@pytest.fixture
def doc():
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    QtWidgets = pytest.importorskip('PySide6.QtWidgets')
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from blackbody.ui.document import Document
    d = Document()
    d.scene = C.new_scene('auto', 'person')
    d.told, d.changed = [], []
    d.message.connect(d.told.append)
    d.paramChanged.connect(d.changed.append)
    yield d
    del app


def test_a_fire_added_from_the_outliner_gets_it_and_says_so(doc):
    doc.add_emitter('sphere')
    assert ts(doc.scene) == C.FIRE_TIME_SCALE
    assert any('Time scale is 1.5' in t for t in doc.told) and ('domain', 'time_scale') in doc.changed
    doc.undo.undo()
    assert ts(doc.scene) == 1.0


@pytest.mark.parametrize('add', ['fabric', 'matter', 'strands', 'shot', 'collider'])
def test_things_added_from_the_outliner_take_it_away_and_say_so(doc, add):
    doc.add_component('campfire')
    assert ts(doc.scene) == C.FIRE_TIME_SCALE
    {'fabric': lambda: doc.add_fabric(), 'matter': lambda: doc.add_matter(material='sand', shape='pile'),
     'strands': lambda: doc.add_strands(), 'shot': lambda: doc.add_shot(),
     'collider': lambda: doc.add_collider('box', dynamic=True)}[add]()
    assert ts(doc.scene) == 1.0 and any('Time scale is back to 1' in t for t in doc.told)
    assert ('domain', 'time_scale') in doc.changed


def test_ticking_falls_in_properties_takes_it_away(doc):
    doc.add_component('campfire')
    doc.add_collider('box')
    assert ts(doc.scene) == C.FIRE_TIME_SCALE              # (a box that stays put)
    doc.set(('collider', len(doc.scene.colliders) - 1, 'dynamic'), True)
    assert ts(doc.scene) == 1.0 and any('Time scale is back to 1' in t for t in doc.told)
    doc.undo.undo()                                        # (one step, the tick and the Time scale together)
    assert ts(doc.scene) == C.FIRE_TIME_SCALE and not doc.scene.colliders[-1]['dynamic']
