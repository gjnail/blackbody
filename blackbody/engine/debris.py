"""Debris: the bits a hit throws off. Chips of stone, concrete and brick, splinters of wood, shards of glass and pottery,
flakes of paint and metal, clods of earth, the lead a bullet splashes against steel, and sparks.

Each bit is a small convex solid (a few planes in its own frame, so the stage traces it exactly, as it does a broken
object's pieces) or, for sparks, a glowing streak. They fly under gravity and the air's drag (quadratic, by their
size), tumble, and bounce off whatever they meet in the rigid world (solids.py's MuJoCo model, by ray casts), losing
most of their speed to it and sliding to a stop; sparks cool from white through orange to dull red and go out.

The bits are drawn as the material they came off (its row in the stage's materials: the inside colour on their
broken faces), the ground's bits in the floor's colour.
"""
from __future__ import annotations

import math

import numpy as np

AIR = 1.2
BOUNCE = 0.25         # of the speed into a surface, what a bit keeps off it
SLIDE = 0.55          # of the speed along it
REST = 0.05           # m/s: slower than this on the ground, a bit lies still
MOST = 1500           # the most bits at once (the oldest settled ones go first)

# kinds of bit: how they are shaped, how long they glow
CHIP, SPLINTER, SHARD, FLAKE, CLOD, LEAD, SPARK, FLUFF = range(8)
KIND_OF = {'chips': CHIP, 'splinters': SPLINTER, 'shards': SHARD, 'flakes': FLAKE, 'dirt': CLOD, 'fluff': FLUFF,
           'sparks': SPARK}


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def shape_planes(kind, size, rng, grain=None):
    """A bit's planes (n . x <= d, own frame, n unit) and its half extents along its own axes: a chip an irregular
    block, a splinter a long thin prism (along x, the grain), a shard or a flake a thin irregular plate (thin along y),
    a clod a lump, a drop of lead a flattened lump."""
    s = float(size)
    if kind == SPLINTER:
        half = np.array([s * rng.uniform(2.5, 6.0), s * rng.uniform(0.15, 0.35), s * rng.uniform(0.2, 0.5)])
    elif kind in (SHARD, FLAKE):
        half = np.array([s * rng.uniform(0.6, 1.4), s * (0.08 if kind == FLAKE else rng.uniform(0.12, 0.25)),
                         s * rng.uniform(0.5, 1.2)])
    elif kind == LEAD:
        half = np.array([s, s * 0.35, s * rng.uniform(0.6, 1.0)])
    else:
        half = s * rng.uniform(0.55, 1.0, 3)
    n_cut = 4 if kind in (SPLINTER, SHARD, FLAKE) else 6
    planes = [np.array([sx * (ax == 0), sx * (ax == 1), sx * (ax == 2), half[ax]], float)
              for ax in range(3) for sx in (1.0, -1.0)]
    # cut corners off at random (a chip is not a box): planes through points a little inside the corners
    for _ in range(n_cut):
        nrm = _unit(rng.normal(size=3) * (1.0 / np.maximum(half, 1e-6)))
        d = float(np.abs(nrm) @ half) * rng.uniform(0.55, 0.85)
        planes.append(np.append(nrm, d))
    if kind == SPLINTER:
        # its ends taper to a point: two planes leaning in at each end
        for sx in (1.0, -1.0):
            for side in (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])):
                k = rng.uniform(0.15, 0.35)
                nrm = _unit(np.array([sx * k, 0.0, 0.0]) + side * rng.choice((-1.0, 1.0)))
                planes.append(np.append(nrm, float(np.abs(nrm) @ half) * 0.8))
    return np.asarray(planes, float), half


class Debris:
    """The bits flying (and lying) round a scene's hits."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.clear()

    def clear(self):
        self.pos = np.zeros((0, 3))
        self.vel = np.zeros((0, 3))
        self.quat = np.zeros((0, 4))          # x y z w
        self.omega = np.zeros((0, 3))         # world rad/s
        self.half = np.zeros((0, 3))          # half extents (m)
        self.kind = np.zeros(0, np.int64)
        self.row = np.zeros(0, np.int64)      # the collider whose material it is (-1: the ground, -2: lead, -3: a spark)
        self.born = np.zeros(0)
        self.life = np.zeros(0)               # s it lasts (sparks), inf for solid bits
        self.still = np.zeros(0, bool)        # lying still
        self.planes = []                      # per bit, (k, 4)

    def __len__(self):
        return len(self.pos)

    def spawn(self, kind, at, normal, out_dir, count, speed, size, row, now, spread=0.6, life=np.inf, grain=None,
              fan=False):
        """`count` bits of a kind thrown off a surface at `at` (its unit normal `normal`): mostly along `out_dir`
        (unit), spread over a cone of `spread` radians round it, at about `speed` m/s; about `size` m across. fan: out
        over the surface instead, all round, a little off it (a bullet splashing flat on a plate)."""
        n = int(count)
        if n <= 0:
            return
        rng = self.rng
        at = np.asarray(at, float)
        out_dir = _unit(np.asarray(out_dir, float))
        normal = _unit(np.asarray(normal, float))
        # directions: a cone round out_dir, never into the surface
        t1 = _unit(np.cross(out_dir, [0.0, 1.0, 0.0] if abs(out_dir[1]) < 0.9 else [1.0, 0.0, 0.0]))
        t2 = np.cross(out_dir, t1)
        th = spread * np.sqrt(rng.random(n))
        ph = rng.uniform(0.0, 2.0 * math.pi, n)
        dirs = (out_dir[None] * np.cos(th)[:, None] + (t1[None] * np.cos(ph)[:, None] + t2[None] * np.sin(ph)[:, None])
                * np.sin(th)[:, None])
        below = dirs @ normal < 0.05
        dirs[below] -= normal[None] * ((dirs[below] @ normal) - 0.05)[:, None]
        if fan:
            ph = rng.uniform(0.0, 2.0 * math.pi, n)
            s1 = _unit(np.cross(normal, [0.0, 1.0, 0.0] if abs(normal[1]) < 0.9 else [1.0, 0.0, 0.0]))
            s2 = np.cross(normal, s1)
            dirs = (s1[None] * np.cos(ph)[:, None] + s2[None] * np.sin(ph)[:, None]
                    + normal[None] * rng.uniform(0.05, 0.35, n)[:, None])
        dirs = _unit(dirs)
        speeds = speed * rng.lognormal(0.0, 0.45, n)
        sizes = size * rng.lognormal(0.0, 0.5, n)
        quats = _unit(rng.normal(size=(n, 4)))
        if grain is not None and kind == SPLINTER:
            # splinters lie along the grain: their long axis (x) turned onto it
            g = _unit(np.asarray(grain, float))
            quats = np.repeat(_quat_from_to(np.array([1.0, 0.0, 0.0]), g)[None], n, 0)
            spin = _unit(rng.normal(size=(n, 4))) * 0.15
            quats = _unit(quats + spin)
        pl, half = [], []
        for k in range(n):
            p, h = shape_planes(kind, sizes[k], rng, grain)
            pl.append(p)
            half.append(h)
        self.planes += pl
        self.pos = np.concatenate([self.pos, at[None] + normal[None] * 0.002 + dirs * 0.003])
        self.vel = np.concatenate([self.vel, dirs * speeds[:, None]])
        self.quat = np.concatenate([self.quat, quats])
        self.omega = np.concatenate([self.omega, rng.normal(size=(n, 3)) * (speeds / np.maximum(sizes, 1e-3) * 0.15)[:, None]])
        self.half = np.concatenate([self.half, np.asarray(half)])
        self.kind = np.concatenate([self.kind, np.full(n, kind, np.int64)])
        self.row = np.concatenate([self.row, np.full(n, row, np.int64)])
        self.born = np.concatenate([self.born, np.full(n, now)])
        life_n = np.full(n, life) if np.isfinite(life) else np.full(n, np.inf)
        if np.isfinite(life):
            life_n = life * rng.uniform(0.5, 1.5, n)
        self.life = np.concatenate([self.life, life_n])
        self.still = np.concatenate([self.still, np.zeros(n, bool)])
        self._trim(now)

    def _trim(self, now):
        if len(self.pos) <= MOST:
            return
        # (the oldest still ones go first, then the oldest)
        order = np.lexsort((self.born, ~self.still))
        keep = np.sort(order[len(self.pos) - MOST:])
        self._keep(keep)

    def _keep(self, keep):
        for name in ('pos', 'vel', 'quat', 'omega', 'half', 'kind', 'row', 'born', 'life', 'still'):
            setattr(self, name, getattr(self, name)[keep])
        self.planes = [self.planes[k] for k in keep]

    def step(self, h, now, ray=None, gravity=9.81, ground=None):
        """Move the bits on h seconds: gravity, the air's drag on them, and their bounces off whatever `ray` finds
        (ray(points (n, 3), vectors (n, 3)) -> (fraction along each of where it meets something (>1: nothing),
        unit normals)); `ground` a height they never fall through (or None)."""
        n = len(self.pos)
        if n == 0:
            return
        gone = np.isfinite(self.life) & (now - self.born > self.life)
        if gone.any():
            self._keep(np.nonzero(~gone)[0])
            n = len(self.pos)
            if n == 0:
                return
        mov = ~self.still
        if not mov.any():
            return
        idx = np.nonzero(mov)[0]
        v = self.vel[idx]
        s = np.linalg.norm(v, axis=1)
        area = 4.0 * np.maximum(self.half[idx, 0] * self.half[idx, 2], 1e-8)
        mass = np.maximum(8.0 * self.half[idx].prod(1), 1e-10) * np.where(self.kind[idx] == LEAD, 11340.0,
                                                                           np.where(self.kind[idx] == SPARK, 7800.0, 2000.0))
        drag = 0.5 * AIR * 1.0 * area / mass
        v = v + h * (np.array([0.0, -gravity, 0.0])[None] - drag[:, None] * s[:, None] * v)
        step = v * h
        if ray is not None:
            frac, nrm = ray(self.pos[idx], step)
            hit = frac <= 1.0
            if hit.any():
                k = np.nonzero(hit)[0]
                vn = np.einsum('ij,ij->i', v[k], nrm[k])[:, None]
                vt = v[k] - vn * nrm[k]
                v[k] = vt * SLIDE - vn * nrm[k] * BOUNCE * (vn < 0.0)
                # (it goes to where it met the surface, a hair off it)
                step[k] = step[k] * np.maximum(frac[k] - 1e-3, 0.0)[:, None] + nrm[k] * 1e-4
                self.omega[idx[k]] *= 0.5
                rest = np.linalg.norm(v[k], axis=1) < max(REST, gravity * h * 3.0)
                if rest.any():
                    r = idx[k[rest]]
                    self.still[r] = self.kind[r] != SPARK
                    v[k[rest]] = 0.0
        self.pos[idx] += step
        if ground is not None:
            low = self.pos[idx, 1] < ground + self.half[idx].min(1)
            if low.any():
                k = idx[low]
                self.pos[k, 1] = ground + self.half[k].min(1)
                vk = v[low]
                falling = vk[:, 1] < 0.0
                vk[falling, 1] = -vk[falling, 1] * BOUNCE
                fi = np.nonzero(falling)[0]
                vk[fi[:, None], [0, 2]] *= SLIDE
                rest = np.linalg.norm(vk, axis=1) < max(REST, gravity * h * 3.0)
                vk[rest] = 0.0
                v[low] = vk
                self.still[k[rest]] = self.kind[k[rest]] != SPARK
        self.vel[idx] = v
        # tumbling: q += 1/2 (omega, 0) q h
        w = self.omega[idx]
        q = self.quat[idx]
        dq = 0.5 * h * _qmul(np.concatenate([w, np.zeros((len(idx), 1))], 1), q)
        self.quat[idx] = _unit(q + dq)
        self.omega[idx] *= math.exp(-0.5 * h)

    def glow(self, now):
        """Each spark's glow (linear rgb, W-ish): white-hot as it flies off, through orange to dull red as it cools,
        out at the end of its life; 0 for the rest."""
        out = np.zeros((len(self.pos), 3))
        sp = self.kind == SPARK
        if not sp.any():
            return out
        age = np.clip((now - self.born[sp]) / np.maximum(self.life[sp], 1e-6), 0.0, 1.0)
        T = 2400.0 - 1400.0 * age
        # a blackbody's colour, roughly (red stays, green and blue fall away as it cools)
        r = np.ones_like(T)
        g = np.clip((T - 900.0) / 1500.0, 0.0, 1.0) ** 1.4
        b = np.clip((T - 1500.0) / 1500.0, 0.0, 1.0) ** 2.0
        bright = 60.0 * (T / 2400.0) ** 6 * (1.0 - age) ** 0.5
        out[sp] = np.stack([r, g, b], 1) * bright[:, None]
        return out

    def state(self):
        return dict(pos=self.pos.copy(), vel=self.vel.copy(), quat=self.quat.copy(), omega=self.omega.copy(),
                    half=self.half.copy(), kind=self.kind.copy(), row=self.row.copy(), born=self.born.copy(),
                    life=self.life.copy(), still=self.still.copy(), planes=[p.copy() for p in self.planes],
                    rng=self.rng.bit_generator.state)

    def load_state(self, st):
        if not st:
            self.clear()
            return
        for k in ('pos', 'vel', 'quat', 'omega', 'half', 'kind', 'row', 'born', 'life', 'still'):
            setattr(self, k, np.asarray(st[k]).copy())
        self.planes = [np.asarray(p, float).copy() for p in st['planes']]
        self.rng.bit_generator.state = st['rng']


def _qmul(a, b):
    """Quaternion products (x y z w), row by row."""
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], 1)


def _quat_from_to(a, b):
    """The quaternion (x y z w) turning unit vector a onto unit vector b."""
    c = float(np.dot(a, b))
    if c < -0.999999:
        ax = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(a, [0.0, 1.0, 0.0])
        ax = ax / np.linalg.norm(ax)
        return np.array([ax[0], ax[1], ax[2], 0.0])
    v = np.cross(a, b)
    q = np.array([v[0], v[1], v[2], 1.0 + c])
    return q / np.linalg.norm(q)
