"""Matter (engine/matter.py) without a GPU: filling shapes, the angle of repose asked for, and matter in the scene
(saved, loaded, signed). The simulation's own tests are in test_matter.py (they need a GPU)."""
import math

import numpy as np

from blackbody.engine.matter import fill_points, friction_angle
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
