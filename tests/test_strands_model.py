"""Grass and plants (engine/strands.py) without a GPU: how the blades are laid out, what a patch's settings come to,
and the scene's patches (saved, loaded, in the simulation's signature, from Create). The GPU tests are in
test_strands.py."""
import math

import numpy as np

from blackbody.engine.strands import KINDS, MAX_BLADES, StrandSpec, blades, patch_data
from blackbody.scene import components
from blackbody.scene.model import Scene


def test_a_patch_has_as_many_blades_as_its_kind_grows_on_its_area():
    for kind in ('lawn', 'meadow', 'wheat', 'reeds'):
        spec = StrandSpec(kind=kind, size=(1.0, KINDS[kind].height, 0.75))
        n = len(blades(spec, 0, MAX_BLADES))
        full = 1.5 * 2.0 * KINDS[kind].density
        assert 0.75 * full < n <= full, (kind, n, full)        # (fewer at its ragged edge)
    thick = len(blades(StrandSpec(size=(1.0, 0.45, 1.0), thickness=2.0), 0, MAX_BLADES))
    thin = len(blades(StrandSpec(size=(1.0, 0.45, 1.0), thickness=0.5), 0, MAX_BLADES))
    assert abs(thick / thin - 4.0) < 0.2


def test_blades_stay_inside_their_patch_turned_with_it():
    b = blades(StrandSpec(shape='disc', pos=(2.0, 0.0, -1.0), size=(0.8, 0.45, 0.3)), 0, MAX_BLADES)
    assert np.hypot(b[:, 0, 0] - 2.0, b[:, 0, 1] + 1.0).max() <= 0.8 + 1e-6
    r = blades(StrandSpec(pos=(0.0, 0.0, 0.0), size=(1.0, 0.45, 0.2), yaw=math.radians(90.0)), 0, MAX_BLADES)
    assert np.abs(r[:, 0, 0]).max() <= 0.2 + 1e-5 and np.abs(r[:, 0, 1]).max() <= 1.0 + 1e-5   # (a quarter turn: x and z swap)


def test_the_blades_are_spread_evenly_shorter_at_the_edge_and_the_same_every_time():
    spec = StrandSpec(size=(1.0, 0.45, 1.0), seed=3)
    b = blades(spec, 0, MAX_BLADES)
    counts, _, _ = np.histogram2d(b[:, 0, 0], b[:, 0, 1], bins=4, range=[[-0.6, 0.6], [-0.6, 0.6]])
    assert counts.std() < 0.15 * counts.mean()                      # (a jittered grid, not clumps and holes)
    edge = np.minimum(1.0 - np.abs(b[:, 0, 0]), 1.0 - np.abs(b[:, 0, 1]))
    assert b[edge < 0.05, 1, 0].mean() < 0.75 * b[edge > 0.4, 1, 0].mean()
    assert np.array_equal(b, blades(spec, 0, MAX_BLADES))
    assert not np.array_equal(b, blades(StrandSpec(size=(1.0, 0.45, 1.0), seed=4), 0, MAX_BLADES))


def test_the_blades_are_capped():
    assert 45_000 < len(blades(StrandSpec(kind='lawn', size=(10.0, 0.08, 10.0)), 0, 50_000)) <= 50_000   # (thinned over all of it)


def test_drier_grass_lights_more_easily_and_burns_faster():
    fresh, dry = patch_data([StrandSpec(dryness=0.0), StrandSpec(dryness=1.0)])[:2]
    assert dry[3, 1] < fresh[3, 1]          # catches in cooler gas
    assert dry[3, 2] < fresh[3, 2]          # and sooner
    assert dry[3, 3] < fresh[3, 3]          # and burns out faster
    assert fresh[5, 2] > dry[5, 2]          # fresh grass smokes more
    assert patch_data([StrandSpec(burns=False)])[0][3, 0] == 0.0
    stiff, soft = patch_data([StrandSpec(stiffness=4.0), StrandSpec(stiffness=0.5)])[:2]
    assert stiff[2, 0] > soft[2, 0]
    own = patch_data([StrandSpec(colour=(0.5, 0.1, 0.1))])[0]
    assert np.allclose(own[0, :3], (0.5, 0.1, 0.1))


def test_the_scene_keeps_its_grass():
    s = Scene()
    i = s.add_strands(kind='wheat')
    assert s.strands[i]['name'] == 'Wheat' and s.strands[i]['size'][1] == KINDS['wheat'].height
    j = s.add_strands(kind='wheat', size=(2.0, 1.0, 2.0))
    assert s.strands[j]['name'] == 'Wheat 2' and s.strands[j]['size'][1] == 1.0
    t = Scene.from_dict(s.to_dict())
    assert t.strands == s.strands and t.sim_signature() == s.sim_signature()
    specs = t.strand_specs()
    assert len(specs) == 2 and specs[0].kind == 'wheat'
    # what the grass does is in the signature; only its colour is not
    sig = s.sim_signature()
    s.strands[0]['own_colour'], s.strands[0]['colour'] = True, (0.3, 0.2, 0.1)
    assert s.sim_signature() == sig
    s.strands[0]['dryness'] = 0.9
    assert s.sim_signature() != sig
    s.strands[1]['enabled'] = False
    assert len(s.strand_specs()) == 1


def test_grass_blocks_go_in_any_scene_but_the_sky():
    keys = [c.key for c in components.COMPONENTS if c.group == 'Grass & plants']
    assert {'lawn', 'meadow', 'dry_grass', 'wheat', 'reeds', 'grassy_hill'} <= set(keys)
    for key in keys:
        for kind in ('fire', 'liquid', 'both'):
            s = components.new_scene(kind, 'person')
            added, _notes = components.add(s, key, at=(0.5, -0.3))
            assert ('strands', 0) in added and s.strands[0]['position'][0] == 0.5
    s = components.new_scene('fire', 'person')
    components.add(s, 'grassy_hill')
    assert s.strands[0]['grows_on'] == 'everything'
    lo, hi = components._extent('strands', s.strands[0])
    assert np.allclose(lo, (-3.9, 0.0, -3.9)) and np.allclose(hi, (3.9, 0.4, 3.9))
