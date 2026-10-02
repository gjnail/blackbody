"""The Lume benchmark: render every scene with Lume and with Mitsuba 3 (the reference), and report how far Lume is from
the truth and how fast it gets there.

    python tools/lume_bench/run.py --mitsuba-python PATH\\TO\\mitsuba-env\\python.exe [--out out/lume_bench/DATE]
                                   [--ref 262144] [--spp 4,16,64,256,1024] [--samplers ...] [SCENE ...]

Lume's side runs in this interpreter (Blackbody's environment), the reference in the one given (any Python with
`pip install mitsuba`; it runs on the GPU with OptiX). Writes OUT/report.md, OUT/SCENE.png (Lume, the reference and
where they differ) and the raw pictures (npz).

Measures, per scene, over every pixel but the lamps themselves (Lume does not draw them):
- bias: Lume's mean brightness at its most samples over the reference's (1.000 is exact);
- error: relative mean squared error, mean((x - ref)^2 / (ref^2 + 0.01)), at each sample count, for both renderers;
- equal-time: each renderer's error after the same render time (log-interpolated), against each of Mitsuba's samplers
  (--samplers: plain random, stratified, multi-jittered, low discrepancy) and the one best for it on that scene, and how
  long each takes to get under 0.01."""
import argparse
import datetime
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from scenes import SCENES                                          # noqa: E402


def rel_mse(x, ref, mask):
    d = (x - ref) ** 2 / (ref ** 2 + 0.01)
    return float(d[mask].mean())


def measured(spec, ref):
    """The pixels the measures are taken over: all but the lamps (Lume does not draw them for the camera; two pixels round
    them too, their edges) and the horizon (the reference's ground is a square 4 km across, Lume's has no end, so the
    row the horizon crosses differs by a fraction of a pixel of sky: geometry, not light)."""
    lamp = ref.max(-1) >= 20.0
    near = lamp.copy()
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            near |= np.roll(np.roll(lamp, dy, 0), dx, 1)
    mask = ~near
    if spec.get('floor') is not None:
        W, H = spec['size']
        cam = spec['camera']
        eye, tgt = np.asarray(cam['eye'], float), np.asarray(cam['target'], float)
        f = (tgt - eye) / np.linalg.norm(tgt - eye)
        r = np.cross(f, [0.0, 1.0, 0.0])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        t = math.tan(math.radians(cam['hfov']) / 2)
        x = (2 * (np.arange(W) + 0.5) / W - 1) * t
        y = (1 - 2 * (np.arange(H) + 0.5) / H) * t * H / W
        d = f[None, None] + x[None, :, None] * r[None, None] + y[:, None, None] * u[None, None]
        el = np.arcsin(d[..., 1] / np.linalg.norm(d, axis=-1))
        mask &= np.abs(el) > 1.5 * 2 * t / W
    return mask


def time_to(err, secs, goal):
    """Seconds to get under goal error (log-log interpolation between the measured counts), or inf."""
    for i in range(len(err)):
        if err[i] <= goal:
            if i == 0:
                return float(secs[0]) * err[0] / goal
            t = (math.log(goal) - math.log(err[i - 1])) / (math.log(err[i]) - math.log(err[i - 1]))
            return math.exp(math.log(secs[i - 1]) + t * (math.log(secs[i]) - math.log(secs[i - 1])))
    return float('inf')


def err_at(err, secs, t):
    """Error after t seconds (log-log interpolation; extrapolated as 1/time beyond the ends)."""
    ls, le = np.log(np.asarray(secs)), np.log(np.asarray(err))
    lt = math.log(t)
    if lt <= ls[0]:
        return float(math.exp(le[0] + (ls[0] - lt)))
    if lt >= ls[-1]:
        return float(math.exp(le[-1] - (lt - ls[-1])))
    return float(math.exp(np.interp(lt, ls, le)))


def picture(lume, ref, path):
    from PIL import Image
    tm = lambda a: (np.clip(a / (1 + a), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    r = np.clip(lume.mean(-1) / np.maximum(ref.mean(-1), 1e-4), 0.0, 2.0)
    heat = np.stack([np.clip((1 - r) * 5, 0, 1), np.clip(1 - np.abs(1 - r) * 5, 0, 1), np.clip((r - 1) * 5, 0, 1)], -1)
    Image.fromarray(np.concatenate([tm(lume), tm(ref), (heat * 255).astype(np.uint8)], 1)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scenes', nargs='*')
    ap.add_argument('--mitsuba-python', required=True)
    ap.add_argument('--out', default=str(ROOT / 'out' / 'lume_bench' / datetime.date.today().isoformat()))
    ap.add_argument('--ref', type=int, default=262144)
    ap.add_argument('--spp', default='4,16,64,256,1024')
    ap.add_argument('--samplers', default='independent,stratified,multijitter,ldsampler',
                    help="Mitsuba's samplers to time (independent first: its renders go with the reference)")
    ap.add_argument('--skip-render', action='store_true', help='report on pictures already in OUT')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = args.scenes or list(SCENES)
    if not args.skip_render:
        subprocess.run([sys.executable, str(HERE / 'lume_side.py'), str(out), *names, '--spp', args.spp], check=True)
        samplers = args.samplers.split(',')
        subprocess.run([args.mitsuba_python, str(HERE / 'mitsuba_side.py'), str(out), *names, '--ref', str(args.ref),
                        '--spp', args.spp, '--sampler', samplers[0]], check=True)
        for s in samplers[1:]:
            subprocess.run([args.mitsuba_python, str(HERE / 'mitsuba_side.py'), str(out / f'mitsuba_{s}'), *names,
                            '--spp', args.spp, '--sampler', s, '--ref-from', str(out)], check=True)
    samplers = args.samplers.split(',')
    lines = [f'# Lume benchmark, {datetime.date.today().isoformat()}', '',
             f'Reference: Mitsuba 3 path tracer (OptiX, GPU) at {args.ref} samples per pixel (multi-jittered), with Lume\'s '
             'own material, so the difference is in how the light is carried. Lume: Blackbody\'s Lume engine, denoiser off. '
             f'Equal time: against Mitsuba with each of its samplers ({", ".join(samplers)}), and with the one that does best '
             'on the scene.', '',
             '| Scene | Bias (Lume / reference) | Lume error at most samples | Equal-time error, Lume / Mitsuba (random) | '
             'Lume / Mitsuba (its best sampler) | Time to error 0.01: Lume | Mitsuba (random) |',
             '|---|---|---|---|---|---|---|']
    for n in names:
        L = np.load(out / f'lume_{n}.npz')
        M = np.load(out / f'mitsuba_{n}.npz')
        ref = M['ref']
        mask = measured(SCENES[n], ref) & np.isfinite(L['imgs'][-1]).all(-1)
        lume_best = L['imgs'][-1]
        bias = float(lume_best[mask].mean() / ref[mask].mean())
        le = [rel_mse(x, ref, mask) for x in L['imgs']]
        me = [rel_mse(x, ref, mask) for x in M['imgs']]
        t_mid = float(np.sqrt(L['secs'][0] * L['secs'][-1]))
        eq = err_at(le, L['secs'], t_mid) / max(err_at(me, M['secs'], t_mid), 1e-12)
        tl, tmi = time_to(le, L['secs'], 0.01), time_to(me, M['secs'], 0.01)
        # (Mitsuba's best sampler here: the least error after the same time)
        best, best_s = eq, samplers[0]
        for s in samplers[1:]:
            p = out / f'mitsuba_{s}' / f'mitsuba_{n}.npz'
            if p.exists():
                Ms = np.load(p)
                e_s = err_at(le, L['secs'], t_mid) / max(err_at([rel_mse(x, ref, mask) for x in Ms['imgs']], Ms['secs'], t_mid), 1e-12)
                if e_s > best:
                    best, best_s = e_s, s
        lines.append(f'| {n} | {bias:.4f} | {le[-1]:.6f} | {eq:.2g}x | {best:.2g}x ({best_s}) | {tl:.3f} s | {tmi:.3f} s |')
        picture(lume_best, ref, out / f'{n}.png')
        detail = ', '.join(f'{int(s)}: {e:.2e} ({t:.3f} s)' for s, e, t in zip(L['spp'], le, L['secs']))
        mdetail = ', '.join(f'{int(s)}: {e:.2e} ({t:.3f} s)' for s, e, t in zip(M['spp'], me, M['secs']))
        lines.append(f'|  | Lume spp: error (time): {detail} | | | | | |')
        lines.append(f'|  | Mitsuba spp: error (time): {mdetail} | | | | | |')
    lines += ['', 'Pictures: SCENE.png is Lume (left), the reference (middle) and their ratio (green: equal, red: Lume '
              'darker, blue: Lume brighter, full colour at 20%).']
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
