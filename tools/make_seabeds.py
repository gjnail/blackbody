"""Write the built-in sea beds (blackbody/assets/meshes/*.png, 16-bit greyscale heightfields read as
terrain): shapes the sea's waves shoal and break over. Each rises from deep water on the left (the
image's left edge, -x: where waves come from with a direction of 90 degrees) toward the shore or the
reef on the right. Use one as a mesh collider (builtin:NAME.png) sized in metres: width, height, depth.

  beach      a sandy beach: a gently concave slope (an equilibrium profile), a few ripples along it
  sandbar    a beach with an offshore bar and a trough, cut by a rip channel (waves break on the bar,
             reform, break again on the beach; the rip carries water back out)
  reef       an A-frame reef: a shallow platform pointing into the swell, with steep edges, so a wave
             breaks first at its tip and peels away both ways along its edges (a surfable barrel)
  point      a point break: a shore running at an angle to the swell, so waves peel along it
  slab       a slab reef, drawn in metres at 25 x 17 m (use it at that size, 3.64 m tall): deep water right
             up to a steep reef face at 35 degrees to the swell, a shallow top behind its lip, a lagoon
             beyond. Each wave jacks up over the lip and throws it over a hollow tube, peeling toward +z
  coast      a town's shore, drawn in metres at 98 x 66 m (use it at that size, 7.84 m tall, for a 6 m sea):
             the sea floor rising from 6 m of water to the beach, a sea wall and flat land 2 m above the sea
  coast_wide the same shore 300 x 300 m round it (centred 40 m inland: position (40, 0, 0)), flat where
             coast lies, for the open water and the land past the box
"""
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'blackbody' / 'assets' / 'meshes'
N = 192


def _grid():
    z, x = np.mgrid[0:N, 0:N] / (N - 1.0)     # x: 0 upwave (deep) .. 1 shore; z: 0 back .. 1 front
    return x, z


def _smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _undulate(x, z, seed, amount):
    rng = np.random.default_rng(seed)
    h = np.zeros_like(x)
    for _ in range(12):
        fz, fx, ph, a = rng.uniform(1.0, 4.0), rng.uniform(0.5, 2.0), rng.uniform(0, 6.3), rng.uniform(0.3, 1.0)
        h += a * np.sin(2 * np.pi * (fz * z + fx * x) + ph)
    return amount * h / 12.0


def beach():
    x, z = _grid()
    u = np.clip((x - 0.1) / 0.9, 0.0, 1.0)
    h = 1.0 - (1.0 - u) ** (2.0 / 3.0)           # depth grows as distance^(2/3) from the shore (Dean)
    return h + _undulate(x, z, 1, 0.012)


def sandbar():
    x, z = _grid()
    h = beach()
    rip = np.exp(-((z - 0.35) / 0.07) ** 2)       # a rip channel through the bar
    h += 0.16 * np.exp(-((x - 0.42) / 0.07) ** 2) * (1.0 - 0.9 * rip)
    h -= 0.05 * np.exp(-((x - 0.56) / 0.06) ** 2)
    return h


def reef():
    x, z = _grid()
    base = 0.08 + 0.1 * _smooth(0.0, 1.0, x)
    # a platform pointing into the swell: its tip at x 0.32, its edges opening at about 35 degrees
    across = np.abs(z - 0.5)
    edge = (x - 0.32) * np.tan(np.radians(35.0)) - across
    top = 0.72 + 0.06 * _smooth(0.4, 1.0, x)
    h = base + (top - base) * _smooth(-0.015, 0.035, edge)
    # a lagoon behind the reef, and the beach past it
    h -= 0.1 * _smooth(0.75, 0.85, x) * (1.0 - _smooth(0.9, 0.98, x))
    return h + _undulate(x, z, 2, 0.01)


def point():
    x, z = _grid()
    # the shore line runs from far out at the back to close in at the front, 30 degrees to the swell
    s = x + (z - 0.5) * np.tan(np.radians(30.0))
    u = np.clip((s - 0.2) / 0.75, 0.0, 1.0)
    h = 1.0 - (1.0 - u) ** 0.8
    return h + _undulate(x, z, 3, 0.012)


SLAB_SIZE = (25.0, 17.0)   # m (x, z): the slab reef is drawn in metres at this size


def slab():
    x, z = _grid()
    X = (x - 0.5) * SLAB_SIZE[0]
    Z = (z - 0.5) * SLAB_SIZE[1]
    xl = -1.0 + math.tan(math.radians(35.0)) * Z          # the lip of the ledge
    h = 0.2 + 3.2 * _smooth(xl - 2.5, xl, X)              # a steep face out of deep water (m)
    h = h + 0.25 * _smooth(xl, xl + 3.0, X)                # the reef top, shallowest just behind the lip
    h = h - 1.2 * _smooth(xl + 4.0, xl + 9.0, X)           # a lagoon behind it
    rng = np.random.default_rng(7)
    rough = np.zeros_like(X)
    for _ in range(10):                                   # coral heads on the reef
        kx, kz, ph = rng.uniform(0.8, 3.0), rng.uniform(0.8, 3.0), rng.uniform(0, 6.3)
        rough += np.sin(kx * X + ph) * np.sin(kz * Z + 2 * ph)
    h = h + 0.02 * rough * _smooth(xl - 1.0, xl + 1.0, X)
    # (in metres at its size: less the 2 % the loader stands every heightfield on, so it comes back exact)
    return h - 0.02 * float(h.max()) / 1.02


COAST_SIZE = (98.0, 66.0)          # m (x, z)
COAST_WIDE = (300.0, 300.0, 40.0)  # m (x, z, centre x)


def _coast_profile(X, Z):
    """A 6 m sea's shore (m): the floor rising from 6 m of water to the waterline at x = 0, the beach, a
    sea wall at x = 12.6 and land 2 m above the sea past it, a few sand waves on the floor."""
    h = np.interp(X, [-48.0, -44.0, 0.0, 12.0, 12.6, 400.0], [0.0, 0.0, 6.0, 7.0, 8.0, 8.0])
    rng = np.random.default_rng(3)
    ripple = np.zeros_like(X)
    for _ in range(6):
        kx, kz, ph = rng.uniform(0.15, 0.6), rng.uniform(0.1, 0.4), rng.uniform(0, 6.3)
        ripple += np.sin(kx * X + kz * Z + ph)
    return h + 0.08 * ripple / 2.5 * (X < 12.0)


def _exact(h, top=8.0):
    # (in metres at its size: less the 2 % the loader stands every heightfield on, so it comes back exact)
    return h - 0.02 * top / 1.02


def coast():
    x, z = _grid()
    return _exact(_coast_profile((x - 0.5) * COAST_SIZE[0], (z - 0.5) * COAST_SIZE[1]))


def coast_wide():
    x, z = _grid()
    X = (x - 0.5) * COAST_WIDE[0] + COAST_WIDE[2]
    Z = (z - 0.5) * COAST_WIDE[1]
    inside = (np.abs(X) < 0.5 * COAST_SIZE[0] - 0.5) & (np.abs(Z) < 0.5 * COAST_SIZE[1] - 0.5)
    h = _exact(_coast_profile(X, Z))
    h[inside] = 0.0   # (the land past the box keeps its top at 8 m: the same height scale as coast)
    return h


def main():
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (('beach', beach), ('sandbar', sandbar), ('reef', reef), ('point', point), ('slab', slab),
                     ('coast', coast), ('coast_wide', coast_wide)):
        h = fn()
        h = np.clip(h, 0.0, None)
        h = h / max(float(h.max()), 1e-6)
        Image.fromarray((h * 65535 + 0.5).astype(np.uint16)).save(OUT / f'{name}.png')
        print('wrote', OUT / f'{name}.png')


if __name__ == '__main__':
    main()
