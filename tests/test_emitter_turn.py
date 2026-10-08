"""An emitter turned by more than a yaw (fire shaped like tipped-over letters, a nozzle on a tumbling object): the gas
solver releases its fuel in the turned shape (emitters.wgsl). GPU; the GPU-free side is in test_solids.py."""
import numpy as np

from blackbody.scene.model import Scene


def _fuel_extent(engine, tipped):
    """How far the fuel a thin, flat slab of an emitter releases reaches up and across (m), with no air moving it."""
    s = Scene()
    s.data['domain'].update(size_x=1.0, size_y=1.0, size_z=1.0, resolution=48, preview_scale=1.0, preroll=0.0, ground=False)
    s.data['motion'].update(buoyancy=0.0, turbulence=0.0, vorticity=0.0, disturbance=0.0, puffing=0.0)
    s.data['embers']['enabled'] = False
    s.emitters = []
    # (cold: its fuel does not catch, so nothing burns or swells)
    s.add_emitter(name='Slab', shape='box', position=(0.0, 0.5, 0.0), size=(0.3, 0.03, 0.3), fuel=20.0, temperature=0.0,
                  noise=0.0, embers=False)
    if tipped:   # shaped like a small sign in a corner, tipped onto its back (Tilt 90)
        s.add_collider(name='Sign', shape='box', position=(0.4, 0.1, 0.4), size=(0.03, 0.03, 0.03), pitch=90.0)
        s.links = [{'child': ['emitter', 'Slab'], 'parent': ['collider', 'Sign'], 'offset': [0.0, 0.0, 0.0], 'shape': True}]
    engine.invalidate()
    engine.prepare(s, final=False)
    engine.simulate_to(s, s.start + 2, cache=False)
    fuel = engine.solver.gpu.read(engine.solver.scal[0]).astype(np.float32)[..., 1]   # (z, y, x)
    inside = fuel > 0.3 * fuel.max()
    h = engine.solver.h
    return inside.any(axis=(0, 2)).sum() * h, inside.any(axis=(1, 2)).sum() * h   # (up, across in z)


def test_a_tipped_emitter_releases_fuel_in_its_tipped_shape(engine):
    up, deep = _fuel_extent(engine, tipped=False)
    assert up < 0.15 and deep > 0.45, (up, deep)          # lying flat
    up, deep = _fuel_extent(engine, tipped=True)
    assert up > 0.45 and deep < 0.15, (up, deep)          # stood on its edge, as the sign it is shaped like
