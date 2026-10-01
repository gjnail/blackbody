"""Render one showcase clip for the docs with Blackbody to out/docs_media/clips/NAME/####.png.

    python render_clip.py NAME [--size 1280x720] [--draft] [--test]

--test renders three frames (first, middle, last) at 640x360 in draft quality into out/docs_media/tests/NAME_*.png,
to check the framing before a final render. Then encode_media.py turns the frames into the docs' video and GIF."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
WORK = os.path.join(ROOT, 'out', 'docs_media')
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np                                    # noqa: E402
from PIL import Image                                 # noqa: E402

from blackbody.engine import camera as cam            # noqa: E402
from blackbody.engine.engine import Engine            # noqa: E402
from blackbody.scene import presets                   # noqa: E402
from blackbody.scene.presets import K                 # noqa: E402

import plates_fast as plates                           # noqa: E402
from clips import CLIPS                                # noqa: E402

FPS = 24


def build_scene(clip, W, H):
    spec = copy.deepcopy(presets.PRESETS[clip['preset']])
    for sec, vals in clip.get('set', {}).items():
        spec.setdefault(sec, {})
        spec[sec] = {**spec[sec], **vals}
    for k in ('emitters', 'colliders'):
        if k in clip:
            spec[k] = clip[k]
    at = float(clip.get('at', 0.0))
    spec['camera'] = {**spec.get('camera', {}), **clip.get('camera', {})}
    every = int(clip.get('every', 1))
    for k, keys in clip.get('camera_keys', {}).items():
        # keyframe times are in seconds of the clip as played back (so a time-lapse clip's camera moves at
        # playback speed): one played second covers `every` simulated seconds
        spec['camera'][k] = K(*[(at + t * every, v) for t, v in keys], interp=clip.get('interp', 'smooth'))
    key = '__clip__'
    presets.PRESETS[key] = spec
    sc = presets.make(key, fps=FPS)
    del presets.PRESETS[key]
    sc.data['render']['width'], sc.data['render']['height'] = W, H
    first = sc.start + int(round(at * FPS))
    return sc, first


def near_half_for(sc, frame):
    spec, fire = sc.camera(frame)
    want = 2.5 * float(spec.distance)
    for s in (3.0, 6.0, 12.0, 24.0, 48.0):
        if want <= s:
            return s
    return 96.0


def plate_for(clip, sc, frame, W, H, near_half=12.0):
    kind = clip.get('plate')
    if not kind:
        return None, None
    spec, fire = sc.camera(frame)
    cs = cam.compute(spec, W / H, fire)
    args = dict(clip.get('plate_args', {}))
    args.setdefault('near_half', near_half)
    args.pop('treeline', None)
    if kind == 'yard':
        return plates.yard_plate(cs, W, H, **args)
    return plates.ground_plate(cs, W, H, **args)


def memory_guard(limit_gb):
    """Stop the render if this process's private memory passes `limit_gb` (Windows): a big liquid or weather scene
    in final quality can otherwise eat the machine's memory before its first frame."""
    if sys.platform != 'win32' or not limit_gb:
        return
    import ctypes
    import threading

    class PMC(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] +                    [(n, ctypes.c_size_t) for n in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                                                   'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                                                   'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage',
                                                   'PrivateUsage')]

    def watch():
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        me = ctypes.windll.kernel32.GetCurrentProcess()
        while True:
            ctypes.windll.kernel32.K32GetProcessMemoryInfo(me, ctypes.byref(pmc), pmc.cb)
            if pmc.PrivateUsage > limit_gb * 1e9:
                print(f'memory guard: {pmc.PrivateUsage / 1e9:.1f} GB is over {limit_gb} GB; stopping '
                      '(try --draft for this clip)', flush=True)
                os._exit(3)
            time.sleep(1.0)
    threading.Thread(target=watch, daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--size', default='1280x720')
    ap.add_argument('--samples', type=int, default=None)
    ap.add_argument('--draft', action='store_true')
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--max-ram-gb', type=float, default=10.0, help='stop if the process passes this much memory')
    ap.add_argument('--out', default=os.path.join(WORK, 'clips'))
    args = ap.parse_args()
    clip = CLIPS[args.name]
    memory_guard(args.max_ram_gb)
    if args.test:
        args.size, args.draft = '640x360', True
    W, H = (int(x) for x in args.size.split('x'))
    sc, first = build_scene(clip, W, H)
    n = int(clip['frames'])
    every = int(clip.get('every', 1))
    idx = list(range(n))
    if args.test:
        idx = [0, n // 2, n - 1]
        out = os.path.join(WORK, 'tests')
    else:
        out = os.path.join(args.out, args.name)
    os.makedirs(out, exist_ok=True)
    last = first + (n - 1) * every
    sc.data['render']['end'] = max(int(sc.data['render']['end']), last)
    samples = args.samples or clip.get('samples', 3)
    final = not (args.draft or clip.get('draft'))   # 'draft': interactive quality (big seas, weather)
    eng = Engine()
    t0 = time.perf_counter()
    eng.prepare(sc, final=final)
    nh = near_half_for(sc, first)
    for j in idx:
        f = first + j * every
        dst = os.path.join(out, f'{args.name}_{j:04d}.png' if args.test else f'{j:04d}.png')
        ts = time.perf_counter()
        eng.simulate_to(sc, f, cache=False)   # the live frame is enough; caching every frame in RAM runs a big liquid out of memory
        tsim = time.perf_counter() - ts
        plate, depth = plate_for(clip, sc, f, W, H, nh)
        kw = {}
        if clip.get('holdout') and depth is not None:
            kw['holdout'] = (None, depth)
        tr = time.perf_counter()
        eng.render(sc, f, (W, H), mode=clip.get('mode', 'composite'), final=final, samples=samples,
                   motion_blur=clip.get('motion_blur', True), plate=plate, **kw)
        img = eng.display_image()[..., :3]
        Image.fromarray(np.ascontiguousarray(img)).save(dst)
        st = eng.stats()
        print(f'{args.name} {j + 1}/{n} frame {f} sim {tsim:.1f}s render {time.perf_counter() - tr:.1f}s '
              f'dims {st.get("dims")}', flush=True)
    if not args.test:
        with open(os.path.join(out, 'done.json'), 'w') as fh:
            json.dump({'frames': n, 'every': every, 'seconds': time.perf_counter() - t0, 'first': first,
                       'size': [W, H]}, fh)
    print(f'{args.name}: {len(idx)} frames in {time.perf_counter() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
