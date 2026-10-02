"""The Lume side of the benchmark: each scene of scenes.py built in Blackbody and rendered with Lume at a series of sample
counts, timed. Run with Blackbody's own environment:

    python tools/lume_bench/lume_side.py OUT_DIR [SCENE ...] [--spp 4,16,64,256,1024]

Writes OUT_DIR/lume_SCENE.npz: for every sample count, the stage's picture (scene-linear rgb, rows top to bottom) and
the seconds it took. Materials, the floor and the lamps are set exactly (not through Blackbody's material table), the
floor has no pattern and no horizon fade, and nothing else lights the set."""
import argparse
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenes import SCENES                                          # noqa: E402


def uniform_hdr(path):
    """A tiny uniform HDRI (radiance 1 everywhere), as a Radiance .hdr."""
    w, h = 16, 8
    rgbe = np.zeros((h, w, 4), np.uint8)
    rgbe[..., :3] = 128      # 0.5 * 2^(129-128) = 1.0
    rgbe[..., 3] = 129
    with open(path, 'wb') as fh:
        fh.write(b'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n' + f'-Y {h} +X {w}\n'.encode() + rgbe.tobytes())


def look_at_euler(eye, target):
    """The free camera's rotation (XYZ Euler degrees) looking from eye at target, y up."""
    from blackbody.scene.groundmatch import euler_xyz
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, (0.0, 1.0, 0.0))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.column_stack([r, u, -f])     # camera to world: right, up, back
    return euler_xyz(R)


def build(spec, sky_path):
    """A Blackbody scene for a benchmark scene, and the exact looks of its objects (stage.looks rows)."""
    from blackbody.engine.renderer import LAMP_SCALE
    from blackbody.engine.stage import CG
    from blackbody.scene.model import Scene
    sc = Scene()
    for k in ('emitters', 'colliders', 'lights', 'fabrics', 'matter', 'strands'):   # (a new scene has a fire in it)
        getattr(sc, k, []).clear()
    d = sc.data['domain']
    d.update(size_x=8.0, size_y=4.0, size_z=8.0, resolution=32, ground=spec['floor'] is not None)
    W, H = spec['size']
    sc.data['render'].update(width=W, height=H, motion_blur=False)
    cam = spec['camera']
    sc.data['camera'].update(mode='free', position=tuple(cam['eye']), rotation=tuple(look_at_euler(cam['eye'], cam['target'])),
                             focal_mm=18.0 / math.tan(math.radians(cam['hfov']) / 2.0), sensor_mm=36.0, use_anchor=False,
                             fire_position=(0.0, 0.0, 0.0), fire_yaw=0.0, scale=1.0, roll=0.0, near=0.02)
    sc.data['lighting'].update(environment=str(sky_path), env_strength=float(spec['sky']), env_rotation=0.0, sun_on=False)
    sc.data['composite'].update(backdrop='stage', floor='bench', floor_tint=(1.0, 1.0, 1.0), visibility=0.0)
    rows = []
    for o in spec['objects']:
        shape = o['shape']
        sc.add_collider(name=f'o{len(rows)}', shape=shape, position=tuple(o['pos']), size=tuple(o['size']), material='plaster',
                        look='cg')
        alb = tuple(float(x) for x in o['alb'])
        rows.append((CG, alb, float(o.get('rough', 0.5)), float(o.get('metal', 0.0)), float(o.get('clear', 0.0)), 0, alb,
                     float(o.get('ior', 1.5))))
    for L in spec['lamps']:
        p = np.asarray(L['power'], float)
        peak = float(p.max())
        sc.add_light(kind='point', position=tuple(L['pos']), radius=float(L['radius']), colour=tuple(p / peak),
                     intensity=peak / LAMP_SCALE, shadows=False)
    return sc, rows


def patch(spec, rows):
    """Exact materials and a plain, unfading floor for this scene (in this process only)."""
    from blackbody.engine import stage
    from blackbody.scene import materials
    fl = spec['floor'] or dict(alb=(0.5, 0.5, 0.5), rough=0.5)
    materials.FLOORS['bench'] = materials.Floor('bench', 'Bench', tuple(fl['alb']), float(fl['rough']), 0)
    stage.FLOORS['bench'] = materials.FLOORS['bench']
    stage.HORIZON_FADE = 1.0e9
    stage.looks = lambda scene, footage: list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('scenes', nargs='*')
    ap.add_argument('--spp', default='4,16,64,256,1024')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spps = [int(x) for x in args.spp.split(',')]
    from blackbody.engine.engine import Engine
    eng = Engine()
    sky = Path(tempfile.mkdtemp()) / 'uniform.hdr'
    uniform_hdr(sky)
    for name in args.scenes or list(SCENES):
        spec = SCENES[name]
        sc, rows = build(spec, sky)
        patch(spec, rows)
        f = sc.start
        eng.prepare(sc, final=True)
        eng.simulate_to(sc, f, cache=False)
        W, H = spec['size']
        imgs, secs = [], []
        for i, spp in enumerate([spps[0]] + spps):     # (the first render compiles the shaders: not timed)
            sc.data['lume'].update(engine='lume', samples=spp, bounces=int(spec['bounces']), denoise=False, clamp=0.0)
            t0 = time.perf_counter()
            eng.render(sc, f, (W, H), mode='composite', final=True, samples=4, motion_blur=False)
            img = eng.gpu.read(eng.stage.tex).astype(np.float32)[..., :3]
            dt = time.perf_counter() - t0
            if i > 0:
                imgs.append(img)
                secs.append(dt)
                print(f'lume {name} {spp} spp: {dt:.2f} s', flush=True)
        np.savez_compressed(out / f'lume_{name}.npz', spp=np.array(spps), secs=np.array(secs), imgs=np.stack(imgs))


if __name__ == '__main__':
    main()
