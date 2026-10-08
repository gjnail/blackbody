"""GPU-free checks that every preset starts clean: nothing that moves starts inside something else, and all its matter
starts inside its box (what is outside is left out). The slow smoke test (test_preset_smoke.py) sees these as the
notices the engine gives; these catch them without a GPU."""
import numpy as np

from blackbody.engine.matter import PER_AXIS, _yaw, fill_points
from blackbody.engine.solids import Solids, breaks
from blackbody.scene import presets


def test_no_preset_starts_a_thing_inside_another():
    # (presets with breakable things are left to the smoke test: working out their pieces takes up to half a minute)
    for name in presets.ORDER:
        sc = presets.make(name)
        if sc.kind == 'cloud' or any(breaks(c) for c in sc.colliders if c['enabled']):
            continue
        S = Solids()
        S.configure(sc, sc.sim_layout(False))
        assert not [w for w in S.warnings if 'inside' in w], (name, S.warnings)


def test_every_presets_matter_starts_inside_its_box():
    for name in presets.ORDER:
        sc = presets.make(name)
        if sc.kind == 'cloud' or not hasattr(sc, 'matter_specs'):
            continue
        for final in (False, True):
            dims, h, origin = sc.sim_layout(final)
            lo, hi = np.asarray(origin, float), np.asarray(origin, float) + np.asarray(dims, float) * h
            dx = float((hi - lo).max()) / max(int(sc.data['domain'].get('matter_detail', 128)), 8)   # (Matter.configure)
            for s in sc.matter_specs():
                if s.pour or s.shape == 'mesh':
                    continue
                pts = _yaw(fill_points(s.shape, s.size, dx / PER_AXIS, np.random.default_rng(0)), s.yaw) + np.asarray(s.pos)
                out = ~np.all((pts >= lo - dx) & (pts < hi + dx), axis=1)
                assert not out.any(), f'{name} ({"final" if final else "preview"}): {int(out.sum())} of {len(pts)} ' \
                                      f'particles of its {s.material} start outside the box {lo} - {hi}'
