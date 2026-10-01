"""Check how fires move against measurements of real fires.

python tools/fire_check.py                          # the pool-like fire presets
python tools/fire_check.py campfire bonfire         # some presets
python tools/fire_check.py out/campfire_shot.bbfire # a project
python tools/fire_check.py campfire --seconds 12 --res 96

Each fire is simulated for a few seconds and measured for:
  puffing  how often the flame's big eddies form, roll up and pinch off, from the flicker of the flame's
           volume. Pool fires of every fuel and size puff at about f = 1.5 / sqrt(D) Hz, with D the base
           diameter in metres (Cetegen & Ahmed 1993; Malalasekera, Versteeg & Gilchrist 1996). A fire
           that puffs too slowly looks big and syrupy; too fast looks small and hectic.
  height   the mean height of visible flame: where it is present half the time (intermittency 0.5), and
           the heat release rate that height implies by Heskestad's correlation L = 0.235 Q^0.4 - 1.02 D
           (L, D in m, Q in kW), as a sanity check of the fire's size.
  speed    how fast the hot gas rises in the flame (95th percentile), against McCaffrey's measurements of
           real flames of that heat release rate: about 1.9 Q^0.2 m/s in the flickering upper flame.
Puffing too slow with speeds about right means the flame's big eddies form too slowly (too little
buoyancy contrast at the flame's edge, or too coarse a grid); both too slow means the whole fire runs
slow, which Domain > Time scale fixes while keeping the fire's shape.

Jets, fireballs, sparks and smoke plumes have no puffing law; check pool-like fires only.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blackbody.engine.engine import Engine
from blackbody.engine.lut import BB_MAX_K, BB_MIN_K, blackbody_lut
from blackbody.scene import Scene, presets
from blackbody.scene.model import base_width

POOL_LIKE = ['campfire', 'bonfire', 'torch', 'pool_fire', 'vehicle_fire', 'fire_whirl']


def puff_law(d):
    """Puffing frequency (Hz) of a pool fire of base diameter d (m)."""
    return 1.5 / math.sqrt(max(d, 1e-3))


def mccaffrey_speed(q):
    """Gas speed (m/s) in the flickering upper part of a real flame of heat release rate q (kW)."""
    return 1.93 * max(q, 0.0) ** 0.2


def visible_flame(v, look, h, lut=blackbody_lut()):
    """Flame radiance seen from the front (x, y), relative to thick flame at the flame temperature, as the
    renderer shades it: flame absorbs and emits at its blackbody brightness (shade.wgsl)."""
    temp, flame = v[..., 0], np.maximum(v[..., 3], 0.0)
    sig = np.maximum(flame * look.flame_density - look.flame_threshold, 0.0) ** look.flame_sharpness * look.flame_absorption
    t = np.maximum(temp, 0.0)
    max_k = max(look.max_k, look.flame_k + 1.0)
    k = look.ambient_k + (look.flame_k - look.ambient_k) * np.minimum(t, 1.0) + (max_k - look.flame_k) * (1.0 - np.exp(-np.maximum(t - 1.0, 0.0)))
    u = np.clip((k - BB_MIN_K) / (BB_MAX_K - BB_MIN_K), 0.0, 1.0) * (len(lut) - 1)
    log_y = np.interp(u, np.arange(len(lut)), lut[:, 3])
    ref = np.interp((look.flame_k - BB_MIN_K) / (BB_MAX_K - BB_MIN_K) * (len(lut) - 1), np.arange(len(lut)), lut[:, 3])
    b = np.minimum(10.0 ** ((log_y - ref) * look.dynamic_range), 1e4)
    tau = sig.sum(axis=0) * h                                   # along z, toward the camera
    return (1.0 - np.exp(-tau)) * (sig * b).sum(axis=0) * h / np.maximum(tau, 1e-9)   # (y, x)


def footprint_width(sc, frame, cell=0.01):
    """Width (m) of the circle with the area of the ground the fire's fuel emitters cover together, found by
    drawing their footprints on a fine grid. None when a mesh emitter or no fuel emitter is involved."""
    shapes = []
    for i, e in enumerate(sc.emitters):
        g = lambda k: sc.get(('emitter', i, k), frame)
        if not e['enabled'] or g('fuel') <= 0.0:
            continue
        if base_width(e, g('position'), g('size'), g('end')) is None:
            return None
        shapes.append((e['shape'], np.asarray(g('position'), float), np.asarray(g('size'), float), np.asarray(g('end'), float)))
    if not shapes:
        return None
    reach = max(float(np.abs(p[[0, 2]]).max() + np.abs(sz).max() + np.abs(en[[0, 2]]).max()) for _, p, sz, en in shapes) + 0.1
    xs = np.arange(-reach, reach, cell)
    X, Z = np.meshgrid(xs, xs)
    cover = np.zeros_like(X, bool)
    for shape, p, sz, en in shapes:
        dx, dz = X - p[0], Z - p[2]
        if shape == 'box':
            cover |= (np.abs(dx) <= sz[0]) & (np.abs(dz) <= sz[2])
        elif shape == 'capsule':
            ax, az = en[0] - p[0], en[2] - p[2]
            t = np.clip((dx * ax + dz * az) / max(ax * ax + az * az, 1e-9), 0.0, 1.0)
            cover |= np.hypot(dx - t * ax, dz - t * az) <= sz[0]
        elif shape == 'ring':
            cover |= np.hypot(dx, dz) <= sz[0] + abs(sz[1])
        else:
            cover |= (dx / max(sz[0], 1e-6)) ** 2 + (dz / max(sz[2], 1e-6)) ** 2 <= 1.0
    return 2.0 * math.sqrt(cover.sum() * cell * cell / math.pi)


def heskestad_q(length, d):
    """Heat release rate (kW) that gives a mean flame height `length` (m) over a base `d` (m) wide."""
    return max((length + 1.02 * d) / 0.235, 0.0) ** 2.5


def spectrum_peak(signal, fps, lo=0.35, hi=12.0, seg_seconds=4.0):
    """Dominant frequency (Hz) of a signal sampled at fps: Welch-averaged spectrum, parabolic peak."""
    x = np.asarray(signal, np.float64)
    # drop the mean and slow drift (a fire still growing or dying down): subtract a 3 s moving average
    k = max(3, int(round(3.0 * fps)) | 1)
    pad = np.pad(x, k // 2, mode='reflect')
    x = x - np.convolve(pad, np.ones(k) / k, mode='valid')
    n = min(len(x), max(16, int(round(seg_seconds * fps))))
    step = max(1, n // 2)
    win = np.hanning(n)
    power = None
    for s in range(0, len(x) - n + 1, step):
        p = np.abs(np.fft.rfft(x[s:s + n] * win)) ** 2
        power = p if power is None else power + p
    freqs = np.fft.rfftfreq(n, 1.0 / fps)
    band = (freqs >= lo) & (freqs <= min(hi, fps / 2.0))
    if power is None or not band.any():
        return float('nan'), freqs, power
    i = int(np.flatnonzero(band)[np.argmax(power[band])])
    if 0 < i < len(power) - 1 and band[i - 1] and band[i + 1]:
        a, b, c = np.log(power[i - 1:i + 2] + 1e-30)
        off = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        return float(freqs[i] + np.clip(off, -0.5, 0.5) * (freqs[1] - freqs[0])), freqs, power
    return float(freqs[i]), freqs, power


def measure(eng, sc, seconds=8.0, settle=1.0, visible=0.1):
    """Simulate `sc` and measure its base diameter, mean flame height, puffing frequency and gas speed.
    Flame counts as visible where it shines at `visible` times the brightness of thick flame."""
    eng.invalidate()
    eng.prepare(sc)
    s = eng.solver
    h = s.h
    fps = sc.fps
    first = sc.start + int(round(settle * fps))
    frames = list(range(first, first + int(round(seconds * fps))))
    images, base_masks, speeds = [], [], []
    for i, f in enumerate(frames):
        eng.simulate_to(sc, f, cache=False)
        v = s.gpu.read(s.scal[0]).astype(np.float32)          # (z, y, x, [temperature, fuel, smoke, flame])
        images.append(visible_flame(v, sc.look(f), h))
        fuel = np.maximum(v[..., 1], 0.0)
        rows = fuel.sum(axis=(0, 2))
        if rows.max() > 0:
            y0 = int(np.argmax(rows > 0.05 * rows.max()))
            base_masks.append(fuel[:, y0:y0 + 3, :].max(axis=1))
        if i % 6 == 0:
            flame = v[..., 3] > 0.25
            if flame.any():
                speeds.append(float(np.percentile(s.read_velocity_centres()[..., 1][flame], 95)))
    img = np.array(images)                                     # (t, y, x)
    present = (img > visible).any(axis=2)                      # (t, y)
    inter = present.mean(axis=0)
    above = np.flatnonzero(inter >= 0.5)
    if not above.size or not base_masks:
        return None
    length = float((above.max() + 1) * h)
    bm = np.array(base_masks)
    area = float((bm > 0.1 * np.percentile(bm.max(axis=(1, 2)), 90)).mean(axis=0).sum()) * h * h
    d = footprint_width(sc, first) or 2.0 * math.sqrt(area / math.pi)
    # puffing: the flicker of the visible flame over its upper part, where the eddies pinch off
    signal = img[:, int(0.3 * length / h):, :].sum(axis=(1, 2))
    f_puff, _, _ = spectrum_peak(signal, fps)
    return {'D': d, 'L': length, 'puff': f_puff, 'speed': float(np.median(speeds)) if speeds else float('nan')}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('fires', nargs='*', help='preset names or .bbfire projects (default: the pool-like presets)')
    ap.add_argument('--seconds', type=float, default=8.0, help='seconds measured, after the fire settles')
    ap.add_argument('--res', type=int, default=0, help='voxels on the longest side (default: the fire\'s own)')
    ap.add_argument('--json', help='also write the results here')
    args = ap.parse_args()
    eng = Engine()
    results = []
    print(f'{"fire":<18}{"D m":>6}{"L m":>6}{"L/D":>5}{"Q kW":>7}{"puff Hz":>8}{"real":>6}{"ratio":>6}'
          f'{"speed":>7}{"real":>6}{"ratio":>6}')
    for name in args.fires or POOL_LIKE:
        sc = Scene.load(name) if name.endswith('.bbfire') else presets.make(name)
        if sc.kind != 'fire':
            print(f'{name:<22}  not a fire; skipped')
            continue
        sc.data['embers']['enabled'] = False
        if args.res:
            sc.data['domain']['resolution'] = args.res
            sc.data['domain']['preview_scale'] = 1.0
        m = measure(eng, sc, args.seconds)
        if m is None or m['D'] <= 0 or not math.isfinite(m['puff']):
            print(f'{name:<22}  no steady flame to measure')
            continue
        real = puff_law(m['D'])
        m['speed'] *= sc.data['domain']['time_scale']   # in the footage's seconds, as the puffing is
        q = heskestad_q(m['L'], m['D'])
        u_real = mccaffrey_speed(q)
        results.append(dict(fire=name, D=m['D'], L=m['L'], Q_kW=q, puff_hz=m['puff'], real_hz=real,
                            puff_ratio=m['puff'] / real, speed=m['speed'], real_speed=u_real,
                            speed_ratio=m['speed'] / u_real, time_scale=sc.data['domain']['time_scale']))
        print(f'{Path(name).stem:<18}{m["D"]:6.2f}{m["L"]:6.2f}{m["L"] / max(m["D"], 1e-6):5.1f}{q:7.0f}'
              f'{m["puff"]:8.2f}{real:6.2f}{m["puff"] / real:6.2f}{m["speed"]:7.2f}{u_real:6.2f}{m["speed"] / u_real:6.2f}')
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
