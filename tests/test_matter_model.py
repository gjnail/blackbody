"""Matter (engine/matter.py) without a GPU: filling shapes, the angle of repose asked for, and matter in the scene
(saved, loaded, signed). The simulation's own tests are in test_matter.py (they need a GPU)."""
import math

import numpy as np

from blackbody.engine.matter import MATTERS, fill_points, friction_angle, material
from blackbody.scene.model import Scene


def test_a_shape_is_filled_with_eight_particles_to_a_cell():
    rng = np.random.default_rng(0)
    assert len(fill_points('box', (0.1, 0.1, 0.1), 0.01, rng)) == 8000
    assert abs(len(fill_points('sphere', (0.1, 0.1, 0.1), 0.01, rng)) / (4.0 / 3.0 * math.pi * 1000.0) - 1.0) < 0.04
    assert abs(len(fill_points('cylinder', (0.1, 0.1, 0.1), 0.01, rng)) / (math.pi * 100.0 * 20.0) - 1.0) < 0.04
    # a pile: a cone 40 cm across and 20 cm high
    assert abs(len(fill_points('pile', (0.2, 0.1, 0.2), 0.01, rng)) / (math.pi * 400.0 * 20.0 / 3.0) - 1.0) < 0.05


def test_the_angle_of_repose_asked_for_sets_the_friction_angle():
    assert abs(friction_angle(28.7) - 33.0) < 1e-9
    a = [friction_angle(x) for x in (15.0, 25.0, 30.0, 35.0, 40.0, 45.0)]
    assert all(b > c for b, c in zip(a[1:], a[:-1]))
    assert all(f >= r for f, r in zip(a, (15.0, 25.0, 30.0, 35.0, 40.0, 45.0)))


def test_matter_is_saved_with_the_scene():
    s = Scene()
    i = s.add_matter(material='snow', shape='sphere', position=(0.0, 1.0, 0.0), size=(0.1, 0.1, 0.1), velocity=(3.0, 1.0, 0.0))
    j = s.add_matter(material='sand', pours=True, rate=2.0, own_colour=True, colour=(1.0, 0.5, 0.0))
    assert s.matter[i]['name'] == 'Snow' and s.matter[j]['name'] == 'Sand'
    s2 = Scene.from_dict(s.to_dict())
    assert [m['material'] for m in s2.matter] == ['snow', 'sand']
    assert s.sim_signature() == s2.sim_signature()
    specs = s2.matter_specs()
    assert specs[0].shape == 'sphere' and specs[0].velocity == (3.0, 1.0, 0.0)
    assert specs[1].pour and abs(specs[1].rate - 0.002) < 1e-12             # (litres a second: m^3/s)
    assert np.allclose(specs[1].colour, (1.0, 0.214, 0.0), atol=1e-3)      # (sRGB as it is picked: linear)
    s2.matter[0]['material'] = 'packing_snow'
    assert s.sim_signature() != s2.sim_signature()


def test_what_melts_melts_into_its_own_melt_and_sets_back():
    melting = [m for m in MATTERS.values() if m.melt]
    assert {m.key for m in melting} == {'wax', 'chocolate', 'aluminium', 'iron'}
    for m in melting:
        melt = material(m.melt)
        assert melt.key == m.melt and melt.freeze == m.key and melt.melts_at == m.melts_at > 0.0
        assert m.heat_capacity > 0.0 and melt.heat_capacity > 0.0 and melt.model == 'mud'
    # (a melt's yield stress keeps a puddle about as deep as its surface tension does for real: a few millimetres)
    for key in ('molten_wax', 'molten_aluminium', 'molten_iron'):
        m = material(key)
        assert 0.002 < m.yield_stress / (m.density * 9.81) < 0.012


def test_a_melt_starts_hotter_than_its_melting_point():
    s = Scene()
    s.add_matter(material='molten_iron', temperature=20.0)
    s.add_matter(material='iron', temperature=1000.0)
    s.add_matter(material='chocolate')
    t = [spec.temperature for spec in s.matter_specs()]
    assert t[0] > material('iron').melts_at + 100.0      # poured hot, whatever its Temperature says
    assert abs(t[1] - 1273.15) < 1e-6 and abs(t[2] - 293.15) < 1e-6


def test_a_mesh_is_filled_with_eight_particles_to_a_cell():
    from blackbody.engine.matter import fill_points
    from blackbody.engine.mesh import MeshSDF
    # a ball 10 cm across as a baked distance grid, 2 mm cells
    n, cell = 70, 0.002
    c = (np.arange(n) + 0.5) * cell - n * cell / 2
    z, y, x = np.meshgrid(c, c, c, indexing='ij')
    sdf = MeshSDF(path='ball', bmin=(-n * cell / 2,) * 3, bmax=(n * cell / 2,) * 3, dims=(n, n, n),
                  data=(np.sqrt(x * x + y * y + z * z) - 0.05).astype(np.float32), triangles=0,
                  mesh_min=(-0.05,) * 3, mesh_max=(0.05,) * 3)
    rng = np.random.default_rng(0)
    pts = fill_points('mesh', (1.0, 1.0, 1.0), 0.005, rng, sdf)
    assert abs(len(pts) / (4.0 / 3.0 * math.pi * 0.05 ** 3 / 0.005 ** 3) - 1.0) < 0.05
    assert np.linalg.norm(pts, axis=1).max() < 0.052
    # scaled: twice as tall
    tall = fill_points('mesh', (1.0, 2.0, 1.0), 0.005, rng, sdf)
    assert abs(len(tall) / len(pts) - 2.0) < 0.1 and np.ptp(tall[:, 1]) > 1.9 * np.ptp(pts[:, 1])


def test_what_burns_burns_hotter_than_it_catches_and_leaves_ash():
    burning = [m for m in MATTERS.values() if m.burns_at > 0.0]
    assert {m.key for m in burning} == {'leaves', 'sawdust', 'coal'}
    for m in burning:
        assert m.burn_temp > m.burns_at + 200.0 and 0.0 < m.burn_rate < 1.0 and 0.0 < m.ash_share < 1.0
        assert material(m.burns_to).key == m.burns_to and material(m.burns_to).burns_at == 0.0
