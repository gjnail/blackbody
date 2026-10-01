"""Write the built-in lava shoreline heightfield (blackbody/assets/meshes/lava_shore.png, 16-bit
greyscale, read as terrain): a rough basalt bench on the left, a shallow channel along the middle that
steers a flow toward the sea, a broken slope down into the water and a gently shelving sea bed.
The image spans 10 m (x, columns) by 8 m (z, rows), well past the simulation box (4 m by 2 m, at its
middle), so the coast carries on out of the box; white is 0.8 m. scene/both_presets.py's 'Lava into
the sea' sets its sea level at 0.35 m."""
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'blackbody' / 'assets' / 'meshes' / 'lava_shore.png'
WIDTH, DEPTH, TOP = 10.0, 8.0, 0.8   # metres the image spans, and the height of white


def smooth_noise(shape, scale, rng):
    """Value noise, smoothed: lumps about `scale` pixels across."""
    h, w = shape
    gh, gw = max(2, int(h / scale) + 2), max(2, int(w / scale) + 2)
    g = rng.random((gh, gw))
    y = np.linspace(0, gh - 1.001, h)
    x = np.linspace(0, gw - 1.001, w)
    y0, x0 = y.astype(int), x.astype(int)
    fy, fx = (y - y0)[:, None], (x - x0)[None, :]
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
    a = g[y0][:, x0]
    b = g[y0][:, x0 + 1]
    c = g[y0 + 1][:, x0]
    d = g[y0 + 1][:, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def main(w=192, h=192, seed=11):
    rng = np.random.default_rng(seed)
    x = (np.arange(w) + 0.5) / w * WIDTH - WIDTH / 2    # metres, -5 .. 5
    z = (np.arange(h) + 0.5) / h * DEPTH - DEPTH / 2    # metres, -4 (back, top row) .. 4
    X, Z = np.meshgrid(x, z)
    # the shore line wanders a little along z
    edge = -0.25 + 0.12 * np.sin(Z * 2.1 + 0.7) + 0.06 * np.sin(Z * 5.3)
    bench = 0.64 - 0.06 * (X + 2.0)                   # the bench, falling toward the sea
    sea_bed = 0.10 - 0.02 * X
    t = np.clip((X - edge) / 0.55, 0.0, 1.0)          # the slope down, over about half a metre
    t = t * t * (3 - 2 * t)
    hgt = bench * (1 - t) + sea_bed * t
    # rough basalt: pahoehoe lobes on the bench, boulders on the slope
    hgt += 0.05 * (smooth_noise((h, w), 12, rng) - 0.5) * (1 - t)
    hgt += 0.06 * (smooth_noise((h, w), 4, rng) - 0.5) * np.exp(-((X - edge - 0.2) / 0.3) ** 2)
    hgt += 0.015 * (smooth_noise((h, w), 2, rng) - 0.5)
    # a shallow channel along the middle of the bench, where the flow comes down
    hgt -= 0.035 * np.exp(-(Z / 0.28) ** 2) * (1 - t)
    # (stored mirrored, the bench on the right: the heightfield bake finds inside and outside along +x and
    # wants the image's low side on the left, as the other built-in terrains have it; the preset turns the
    # collider round 180 degrees)
    hgt[0, 0] = TOP   # (a corner at full height, so the image's white is TOP whatever the noise does)
    img = np.clip(hgt / TOP, 0.0, 1.0)[::-1, ::-1]
    Image.fromarray((img * 65535).astype(np.uint16)).save(OUT)
    print('wrote', OUT, img.shape, float(hgt.min()), float(hgt.max()))


if __name__ == '__main__':
    main()
