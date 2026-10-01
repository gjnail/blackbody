"""Write the built-in pond heightfield (blackbody/assets/meshes/pond_basin.png, 16-bit greyscale, read as
terrain): an oval hollow with a wandering shore, about 1.3 m by 0.9 m, its banks sloping up from a flat
bed to rough ground a few centimetres above the water. The image spans 3.2 m (x, columns) by 2.4 m (z,
rows), past the simulation box (1.6 m by 1.2 m, at its middle), so the ground carries on out of the box;
white is 0.4 m. scene/phase_presets.py's 'Pond freezing over' places it 3 cm down (position y -0.03),
so the bed is just under the box's floor: the pond's deep middle lies on the floor (the ground, at the
ground's temperature: unfrozen mud) and its slopes and banks are the collider (frozen earth, at the
collider's temperature). The water's surface is at 0.2 m."""
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'blackbody' / 'assets' / 'meshes' / 'pond_basin.png'
WIDTH, DEPTH, TOP = 3.2, 2.4, 0.4   # metres the image spans, and the height of white
DROP = 0.03                         # the preset's collider sits this far down
BED, GROUND = -0.03, 0.245          # the bed (under the floor) and the ground round the pond (m)


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


def main(w=192, h=144, seed=5):
    rng = np.random.default_rng(seed)
    x = (np.arange(w) + 0.5) / w * WIDTH - WIDTH / 2    # metres, -1.6 .. 1.6
    z = (np.arange(h) + 0.5) / h * DEPTH - DEPTH / 2    # metres, -1.2 (back, top row) .. 1.2
    X, Z = np.meshgrid(x, z)
    # an oval with a wandering shore: rho is 1 about where the water meets the bank
    th = np.arctan2(Z / 0.44, X / 0.62)
    wob = 1.0 + 0.07 * np.sin(3 * th + 1.0) + 0.045 * np.sin(5 * th + 2.3) + 0.02 * np.sin(9 * th + 0.4)
    rho = np.hypot(X / 0.62, Z / 0.44) / wob
    t = np.clip((rho - 0.5) / 0.62, 0.0, 1.0)           # the slope up, from the bed to the ground
    t = t * t * (3 - 2 * t)
    hgt = BED + (GROUND - BED) * t
    # the ground: low tussocks on the banks and a little rise away from the pond (lumps about 15 cm across:
    # finer ones are below what the colliders' distance field holds, and come out blocky); the bed is left
    # flat (it lies under the box's floor)
    bank = np.clip((rho - 0.75) / 0.3, 0.0, 1.0)
    hgt += 0.025 * (smooth_noise((h, w), 9, rng) - 0.5) * bank
    hgt += 0.02 * np.clip(rho - 1.3, 0.0, None)
    hgt[0, 0] = TOP - 0.02 * TOP - DROP   # (a corner at full height, so the image's white is TOP whatever the noise does)
    # world height = (0.02 + image) * TOP - DROP
    img = np.clip((hgt + DROP) / TOP - 0.02, 0.0, 1.0)
    Image.fromarray((img * 65535).astype(np.uint16)).save(OUT)
    print('wrote', OUT, img.shape, float(hgt.min()), float(hgt.max()))


if __name__ == '__main__':
    main()
