"""Simulate a small scene of each solver family and print a fingerprint of its state (one JSON line), for
tests/test_determinism.py to compare between processes: python tests/determinism_run.py [family ...].

Each family is a preset on a coarse grid for a few frames, simulated afresh (no frame cache, no disk cache). The
fingerprint is a hash of what it holds then, read back exactly: the gas's fields; the liquid's and lava's particles (as a
set, sorted: which slot a particle sits in may differ, what it does may not); the fabric's points; the weather's packed
pieces (in the order of their slots) and its cover; the falling objects' positions and velocities (a run of their own:
nothing the solids memo kept from another run of the scene in this process)."""
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

# family: (preset, cells along the box's longest side, frames after the start, what to read, changes to the preset)
FAMILIES = {
    'fire': ('torch', 40, 10, ('gas',), {}),
    'liquid': ('water_pour', 40, 10, ('liquid',), {}),
    'cloth': ('flag_wind', 32, 10, ('gas', 'fabric'), {'self_collide': False}),
    'cloth_folds': ('flag_wind', 32, 10, ('gas', 'fabric'), {}),
    'weather': ('hail_pond', 32, 8, ('liquid', 'weather'), {}),
    'fire_water': ('hose_on_fire', 40, 8, ('gas', 'liquid'), {}),
    'lava': ('lava', 40, 10, ('liquid',), {}),
    'lava_sea': ('lava_sea', 40, 8, ('gas', 'liquid', 'lava'), {}),
    'objects': ('vase_drop', 32, 12, ('gas', 'objects'), {}),
}


def _hash(*arrays):
    h = hashlib.sha1()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def _set(x, v):
    """Particles as a set: their rows (position, velocity) sorted."""
    if not len(x):
        return np.zeros(0, np.float32)
    rows = np.concatenate([np.asarray(x, np.float32).reshape(len(x), -1), np.asarray(v, np.float32).reshape(len(v), -1)],
                          axis=1)
    rows = np.ascontiguousarray(rows)
    return np.sort(rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).ravel())


def fingerprint(engine, family):
    from blackbody.scene import presets
    name, res, frames, parts, change = FAMILIES[family]
    sc = presets.make(name)
    d = sc.data['domain']
    d['resolution'] = min(d['resolution'], res)
    d['disk_cache'] = False
    for e in sc.emitters:   # (water poured from the start: a hose that waits a second for the fire would add nothing)
        if e.get('emits') == 'liquid':
            e['start'] = min(e['start'], 0.0)
    if 'self_collide' in change:
        for f in sc.fabrics:
            f['self_collide'] = change['self_collide']
    engine.invalidate()
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + frames, cache=False)
    out = {}
    for p in parts:
        if p == 'gas':
            s = engine.solver
            out[p] = _hash(s.read_scalars(), s.read_velocity_centres())
        elif p in ('liquid', 'lava'):
            L = engine.liquid if p == 'liquid' else engine.lava
            x, v = L.read_particles()
            out[p] = _hash(_set(x, v)) + f':{len(x)}'
        elif p == 'fabric':
            out[p] = _hash(*engine.cloth.positions())
        elif p == 'weather':
            W = engine.weather
            out[p] = _hash(W.read_packed(), W.read_cover()) + f':{len(W.read_packed())}'
        elif p == 'objects':
            out[p] = _hash(np.asarray(engine.solids.data.qpos, np.float64), np.asarray(engine.solids.data.qvel, np.float64))
    return out


def main(families):
    os.environ.setdefault('BLACKBODY_CACHE', str(Path(os.environ.get('TEMP', '.')) / 'blackbody-determinism-cache'))
    from blackbody.engine.engine import Engine
    engine = Engine()
    out = {f: fingerprint(engine, f) for f in families}
    print(json.dumps(out, sort_keys=True))


if __name__ == '__main__':
    main(sys.argv[1:] or list(FAMILIES))
