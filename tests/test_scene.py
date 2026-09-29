import math
from pathlib import Path

import pytest

from blackbody.scene import Scene, presets
from blackbody.scene.anim import Curve
from blackbody.scene.params import SECTIONS, coerce, param


def test_every_preset_builds_and_round_trips(tmp_path):
    for name in presets.ORDER:
        s = presets.make(name)
        p = tmp_path / f'{name}.bbfire'
        s.save(p)
        t = Scene.load(p)
        assert t.to_dict()['sections'] == s.to_dict()['sections']
        assert len(t.emitters) == len(s.emitters)
        assert t.sim_signature() == s.sim_signature()


def test_curves_and_keys():
    s = Scene()
    path = ('motion', 'wind_speed')
    s.set_key(path, 1, 0.0)
    s.set_key(path, 11, 5.0)
    assert s.get(path, 1) == pytest.approx(0.0)
    assert s.get(path, 11) == pytest.approx(5.0)
    assert 0.0 < s.get(path, 6) < 5.0
    assert s.get(path, 100) == pytest.approx(5.0)       # holds after the last key
    s.set(path, 3.0, frame=6)                            # editing an animated value sets a key
    assert s.curve(path).has_key(6)
    d = s.to_dict()
    t = Scene.from_dict(d)
    assert t.get(path, 6) == pytest.approx(3.0)
    t.remove_key(path, 1)
    t.remove_key(path, 6)
    t.remove_key(path, 11)
    assert t.curve(path) is None


def test_vector_curve_linear():
    c = Curve([[0, (0.0, 0.0, 0.0), 'smooth'], [10, (10.0, 20.0, 30.0), 'smooth']])
    assert c.eval(5) == pytest.approx((5.0, 10.0, 15.0))


def test_coerce_respects_types_and_limits():
    assert coerce(param('domain', 'resolution'), '3000') == 768
    assert coerce(param('domain', 'ground'), 'false') is False
    assert coerce(param('shading', 'smoke_albedo'), '0.1 0.2 0.3') == (0.1, 0.2, 0.3)
    assert coerce(param('camera', 'mode'), 'nonsense') == 'orbit'


def test_signature_tracks_simulation_changes_only():
    s = presets.make('campfire')
    sig = s.sim_signature()
    s.set(('shading', 'exposure'), 2.0)          # look only
    s.set(('composite', 'haze'), 3.0)
    assert s.sim_signature() == sig
    s.set(('motion', 'buoyancy'), 9.0)            # simulation
    assert s.sim_signature() != sig


def test_envelope_and_master_fuel():
    s = presets.make('fireball')
    fps = s.fps
    e = s.emitters[0]
    assert s.envelope(e, s.start) < 0.01                       # t = 0: just igniting
    assert s.envelope(e, s.start + int(0.1 * fps)) > 0.9       # burning
    assert s.envelope(e, s.start + int(1.0 * fps)) == 0.0      # stopped
    s.set(('combustion', 'fuel_scale'), 0.5)
    f = s.start + int(0.1 * fps)
    assert s.emitters_gpu(f)[0].fuel == pytest.approx(e['fuel'] * s.envelope(e, f) * 0.5)


def test_layout_uses_multiples_of_eight():
    s = presets.make('bonfire')
    for final in (False, True):
        dims, h, origin = s.sim_layout(final)
        assert all(d % 8 == 0 for d in dims)
        assert origin[1] == 0.0
        assert math.isclose(origin[0], -dims[0] * h / 2)


def test_footage_relinks_when_project_and_footage_move(tmp_path):
    import os
    import shutil
    a = tmp_path / 'a'
    (a / 'media').mkdir(parents=True)
    (a / 'proj').mkdir()
    (a / 'media' / 'plate.mov').write_bytes(b'x')
    s = presets.make('torch')
    s.footage = {'path': str(a / 'media' / 'plate.mov'), 'offset': 0}
    s.save(a / 'proj' / 'shot.bbfire')
    b = tmp_path / 'b'
    shutil.move(str(a), str(b))
    t = Scene.load(b / 'proj' / 'shot.bbfire')
    assert os.path.exists(t.footage['path'])
    assert Path(t.footage['path']).resolve() == (b / 'media' / 'plate.mov').resolve()
