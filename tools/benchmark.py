"""Measure simulation and preview-render time per frame at several resolutions, and what each
optional feature costs on top of the plain campfire.

python tools/benchmark.py              # resolutions, then features at 144 voxels
python tools/benchmark.py --features   # features only
python tools/benchmark.py --res 192    # features at another resolution
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blackbody.engine.engine import Engine
from blackbody.scene import presets
from blackbody.scene.presets import K, _col, _em


def measure(eng, sc, frames=24, final=False, size=(960, 540)):
    """Average simulate and render milliseconds per frame, after the fire has developed."""
    eng.invalidate()
    eng.prepare(sc, final=final)
    eng.simulate_to(sc, sc.start + 24, cache=False)   # warm up and let the fire develop
    sim_ms = rend_ms = 0.0
    for f in range(sc.start + 25, sc.start + 25 + frames):
        a = time.perf_counter()
        eng.simulate_to(sc, f, cache=False)
        eng.gpu.sync()
        b = time.perf_counter()
        eng.render(sc, f, size, mode='composite', samples=1, final=final)
        eng.display_image()
        c = time.perf_counter()
        sim_ms += (b - a) * 1000
        rend_ms += (c - b) * 1000
    return sim_ms / frames, rend_ms / frames


def campfire(res):
    sc = presets.make('campfire')
    sc.data['domain'].update(resolution=res, preview_scale=1.0, preroll=0.0)
    return sc


def feature_cases(res):
    """(label, scene) pairs: the campfire with one feature switched on at a time."""
    def with_(fn):
        sc = campfire(res)
        fn(sc)
        return sc
    yield 'plain campfire', campfire(res)
    yield 'swirl', with_(lambda s: s.emitters[2].update(swirl=2.0))
    yield 'moving emitter', with_(lambda s: s.emitters.__setitem__(2, _em(
        s, shape='cylinder', size=(0.2, 0.04, 0.2), fuel=8, temperature=0.4,
        position=K((0.0, (-0.5, 0.05, 0.0)), (2.0, (0.5, 0.05, 0.0)), interp='linear'))))
    yield 'tracked air', with_(lambda s: s.data['combustion'].update(air='tracked'))
    yield 'steam (dousing)', with_(lambda s: s.emitters.append(_em(
        s, shape='box', position=(0, 0.8, 0), size=(0.4, 0.3, 0.4), fuel=0.0, temperature=0.0, douse=5.0, embers=False)))
    yield 'colourant', with_(lambda s: s.emitters[2].update(color_amount=2.0))
    yield 'spreading fire', with_(lambda s: s.data['spread'].update(enabled=True))
    yield 'moving collider', with_(lambda s: s.colliders.append(_col(
        s, 0, shape='box', size=(0.2, 0.2, 0.2), position=K((0.0, (-0.8, 1.2, 0.0)), (2.0, (0.8, 1.2, 0.0)), interp='linear'))))
    yield 'mesh collider', with_(lambda s: s.colliders.append(_col(
        s, 0, shape='mesh', mesh='builtin:armchair.obj', position=(0.6, 0.0, -0.6), size=(0.8, 0.8, 0.8))))
    yield 'heat expansion', with_(lambda s: s.data['combustion'].update(thermal_expansion=1.0))
    yield 'multiple scattering off', with_(lambda s: s.data['shading'].update(multiple_scattering=0.0))
    yield 'surface light off', with_(lambda s: s.data['composite'].update(surface_light=0.0))
    yield 'surface shadows off', with_(lambda s: s.data['composite'].update(surface_shadows=0.0))
    yield 'flame fronts + heavy fuel', with_(lambda s: s.data['combustion'].update(flame_speed=4.0, fuel_weight=0.5))
    yield 'soot stains', with_(lambda s: s.data['combustion'].update(soot_stain=0.3))
    yield 'growing box', with_(lambda s: s.data['domain'].update(grow=True, grow_limit=2.0, size_x=1.0, size_z=1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', action='store_true', help='only the feature costs')
    ap.add_argument('--res', type=int, default=144, help='resolution for the feature costs')
    args = ap.parse_args()
    eng = Engine()
    print(f'GPU: {eng.gpu.name} ({eng.gpu.backend})')
    if not args.features:
        for res in (96, 144, 192, 256, 384):
            sc = campfire(res)
            t0 = time.perf_counter()
            sim, rend = measure(eng, sc)
            dims = eng.solver.dims
            mem = eng.solver.memory_bytes() / 1e9
            print(f'res {res:3d}: grid {dims[0]}x{dims[1]}x{dims[2]} ({dims[0]*dims[1]*dims[2]/1e6:.2f} M voxels, ~{mem:.2f} GB) '
                  f'sim {sim:6.1f} ms  render {rend:5.1f} ms  -> {1000.0 / (sim + rend):5.1f} fps '
                  f'(substeps {eng.last_substeps})')
    print(f'\nFeature costs, campfire at {args.res} voxels (simulate + 960x540 preview):')
    base = None
    for label, sc in feature_cases(args.res):
        sim, rend = measure(eng, sc)
        total = sim + rend
        base = base or total
        print(f'  {label:26s} sim {sim:6.1f} ms  render {rend:5.1f} ms  total {total:6.1f} ms  ({(total / base - 1) * 100:+5.0f}%)'
              f'  ~{eng.solver.memory_bytes() / 1e9:.2f} GB')
    sc = campfire(args.res // 2)
    sc.data['render'].update(upres=2, final_scale=1.0)
    sim, rend = measure(eng, sc, frames=8, final=True)
    print(f'  upres 2x from {args.res // 2:3d} voxels   sim {sim:6.1f} ms  render {rend:5.1f} ms  (final quality)')
    sc = campfire(args.res)
    sim, rend = measure(eng, sc, frames=8, final=True)
    print(f'  plain at {args.res:3d} voxels         sim {sim:6.1f} ms  render {rend:5.1f} ms  (final quality)')


if __name__ == '__main__':
    main()
