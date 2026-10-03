"""Objects' heat (engine/objheat.py) and the shared radiant sources (engine/radiant.py), the parts that need no GPU:
the surface they meet the world through, the laws of contact, boiling and radiation, and the energy they keep."""
import math

import numpy as np
import pytest

from blackbody.engine import objheat as OH
from blackbody.engine.radiant import FLAME_K, MAX_K, gas_kelvin
from blackbody.engine.solids import shape_area
from blackbody.scene.materials import FLOORS, MATERIALS
from blackbody.scene.model import Scene


@pytest.mark.parametrize('shape,size', [('box', (0.3, 0.1, 0.05)), ('sphere', (0.07, 0.07, 0.07)), ('cylinder', (0.1, 0.25, 0.1))])
def test_surface_points_cover_the_whole_surface_once(shape, size):
    p, n, a = OH.surface_points(shape, size)
    assert abs(a.sum() - shape_area(shape, size)) < 1e-3 * shape_area(shape, size)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)
    assert np.all(np.einsum('ij,ij->i', p, n) > 0.0)       # (every normal points out)


def test_contact_falls_as_one_over_root_time_and_is_led_by_the_poorer_conductor():
    g1 = OH.contact_conductance(14000.0, 1900.0, 1.0)
    assert abs(OH.contact_conductance(14000.0, 1900.0, 4.0) - g1 / 2.0) < 1e-9
    # (steel on concrete: some 950 W/m2/K after a second; steel on wood far less, set by the wood)
    assert 900.0 < g1 < 1000.0
    assert OH.contact_conductance(14000.0, 400.0, 1.0) < 0.25 * g1
    assert OH.contact_conductance(14000.0, 1900.0, 0.0) == OH.contact_conductance(14000.0, 1900.0, OH.CONTACT_MIN_S)


def test_water_boils_off_a_hot_wall_through_the_boiling_curve():
    q = lambda t: float(OH.boil_or_convect(np.array([t]), 20.0)[0])
    assert q(60.0) < q(110.0)                    # convection below the boil, then nucleate boiling
    assert q(120.0) > 3.0e5                      # near the critical heat flux (~1 MW/m^2)
    assert q(300.0) < q(120.0)                   # film boiling past Leidenfrost: a vapour layer insulates it
    assert q(900.0) > q(300.0)                   # and radiates through it at red heat


def test_materials_carry_heat_and_floors_take_it():
    for k, m in MATERIALS.items():
        assert 100.0 < m.heat_capacity < 5000.0 and 0.0 < m.emissivity <= 1.0, k
    assert MATERIALS['steel'].heat_capacity < MATERIALS['wood'].heat_capacity
    assert MATERIALS['aluminium'].emissivity < 0.2 < MATERIALS['wood'].emissivity
    assert FLOORS['concrete'].effusivity > FLOORS['boards'].effusivity


def test_old_scenes_keep_their_heated_objects_held():
    s = Scene()
    s.add_collider(name='Plate', temperature=180.0)
    s.add_collider(name='Box')
    d = s.to_dict()
    for c in d['colliders']:
        c.pop('keeps_temperature', None)        # (as saved before objects warmed and cooled)
    t = Scene.from_dict(d)
    assert t.colliders[0]['keeps_temperature'] and not t.colliders[1]['keeps_temperature']
    s2 = Scene.from_dict(s.to_dict())
    assert not s2.colliders[0]['keeps_temperature']    # (saved now: as set)


def test_gas_temperature_is_the_flames_real_one():
    assert abs(gas_kelvin(0.0, 293.0) - 293.0) < 1e-9
    assert abs(gas_kelvin(1.0, 293.0) - FLAME_K) < 1e-9
    assert FLAME_K < gas_kelvin(3.0, 293.0) < MAX_K


def _thing(scene_temp=700.0, material='steel', shape='sphere', size=(0.05, 0.05, 0.05), keeps=False):
    s = Scene()
    s.data['domain']['kind'] = 'fire'
    s.add_collider(name='It', shape=shape, size=size, position=(0.0, 1.0, 0.0), material=material,
                   temperature=scene_temp, keeps_temperature=keeps)
    oh = OH.ObjectHeat(None)
    oh.configure(s)
    oh.speed = 1.0
    return oh


def _open_air(oh, rad=0.0):
    t = oh.things[0]
    n = len(t.la)
    return dict(air_k=np.full(n, oh.ambient_k), speed=np.zeros(n), rad=np.full(n, rad), wet=np.zeros(n, bool),
                lava_k=np.zeros(n), in_lava=np.zeros(n, bool), open=np.ones(n, bool), ground=np.zeros(n, bool), pairs={},
                owner=np.zeros(n, int), area=t.la)


def test_a_hot_steel_ball_in_still_air_loses_what_it_radiates_and_convects():
    oh = _thing(700.0)
    t = oh.things[0]
    T0, Ta = t.skin, oh.ambient_k
    e0 = t.c_skin * t.skin + t.c_core * t.core
    oh._integrate(np.zeros(1), _open_air(oh), [True], 0.25)
    e1 = t.c_skin * t.skin + t.c_core * t.core
    dT = T0 - Ta
    q = t.area * (t.eps * OH.SIGMA * (T0 ** 4 - Ta ** 4) + max(1.52 * dT ** (1.0 / 3.0), 5.7) * dT)
    assert abs((e0 - e1) / 0.25 - q) < 0.03 * q
    assert abs(oh.last['heat_j'][0] + (e0 - e1)) < 1e-6 * e0    # (the sums are the heat it lost)


def test_radiation_falling_on_it_warms_it_by_its_emissivity():
    oh = _thing(20.0)
    t = oh.things[0]
    e0 = t.c_skin * t.skin + t.c_core * t.core
    oh._integrate(np.zeros(1), _open_air(oh, rad=20000.0), [True], 0.1)
    gained = t.c_skin * t.skin + t.c_core * t.core - e0
    assert abs(gained / 0.1 - t.eps * 20000.0 * t.area) < 0.05 * t.eps * 20000.0 * t.area


def test_an_object_kept_at_its_temperature_never_changes():
    oh = _thing(250.0, keeps=True)
    oh._integrate(np.full(1, -5000.0), _open_air(oh), [True], 1.0)
    assert oh.things[0].skin == pytest.approx(250.0 + 273.15)


def test_a_wooden_skin_heats_far_faster_than_its_core_and_far_more_than_steels():
    rises = {}
    for mat in ('wood', 'steel'):
        oh = _thing(20.0, material=mat, shape='box', size=(0.05, 0.05, 0.05))
        t = oh.things[0]
        oh._integrate(np.zeros(1), _open_air(oh, rad=30000.0), [True], 2.0)
        rises[mat] = (t.skin - 293.15, max(t.core - 293.15, 1e-9))
    (ws, wc), (ss, sc) = rises['wood'], rises['steel']
    assert ws > 30.0 * ss                       # (in a fire's radiation a log's skin chars in seconds, steel barely warms)
    assert ws / wc > 10.0 * (ss / sc)           # (and its heart stays cold; steel's heat soaks in)


def test_heat_given_away_comes_back_as_the_matter_and_water_report_it():
    oh = _thing(500.0)
    t = oh.things[0]
    e0 = t.c_skin * t.skin + t.c_core * t.core
    oh._integrate(np.array([-2000.0]), None, [True], 0.04)   # (2 kJ to the water this frame)
    assert abs(e0 - (t.c_skin * t.skin + t.c_core * t.core) - 2000.0) < 1e-6 * e0
