"""Bullets into sand, snow, mud, jelly, clay and water (ballistics.py): the matter (matter.py) and the liquid
(liquid.py) live on the GPU, so the rigid world's ray casts cannot see them. Each frame:

- before the bullets fly (look_ahead): each bullet's way through the frame is checked against them on the GPU (the
  matter's particles, bullet_mfind.wgsl; the liquid's cell types, bullet_lfind.wgsl), and the stretches of it inside
  them handed to the bullets (Ballistics.ahead), which meet them there as surfaces (sand, gel, water...) and slow, stop
  or go through as they do in any other;
- after (kick): what each bullet did along its track, as far as it got this frame, goes into them as a push
  (bullet_matter.wgsl, bullet_liquid.wgsl): the temporary cavity a bullet opens in jelly and water, the crater and
  spray it throws up out of sand and snow, the splash at the entry.
"""
from __future__ import annotations

import math

import numpy as np

from .ballistics import MATTER_SURFACES, SURFACES
from .gpu import Uniforms, groups_1d

NB = 256              # bins along a path (bullet_*find.wgsl)
MAX_PATHS = 8
TRACK = 32            # samples along a track (bullet_matter.wgsl, bullet_liquid.wgsl)
REACH = 200.0         # m: how far on a bullet's way is looked along
CORE = 1.5            # grid spacings: the core of the flow round a bullet's path (track_samples)

# How each takes a bullet's energy: the pressure its temporary cavity opens against (Pa: the cavity's radius is
# sqrt(dE/ds / (pi p))), the share of the energy that goes into moving it (the rest is heat, crushing, friction), how
# deep below the entry the throw turns back out of the hole (m), and how much of it does there.
MEDIA = {'gel': (5.0e5, 0.45, 0.02, 0.5), 'water': (2.0e5, 0.5, 0.03, 0.9), 'sand': (2.0e6, 0.15, 0.06, 0.85),
         'snow': (3.0e5, 0.2, 0.08, 0.9), 'mud': (5.0e5, 0.3, 0.05, 0.8), 'clay': (2.5e6, 0.3, 0.02, 0.4),
         'wax': (5.0e6, 0.25, 0.01, 0.3)}
FASTEST = {'matter': 60.0, 'liquid': 35.0}   # m/s: the most a kick moves any of it (the step it then needs)


def clip(A, u, L, lo, hi):
    """The part of the segment A + s u (0 <= s <= L) inside the box lo..hi: (s0, s1), or None."""
    t0, t1 = 0.0, L
    for k in range(3):
        if abs(u[k]) < 1e-12:
            if A[k] < lo[k] or A[k] > hi[k]:
                return None
            continue
        a = (lo[k] - A[k]) / u[k]
        b = (hi[k] - A[k]) / u[k]
        if a > b:
            a, b = b, a
        t0, t1 = max(t0, a), min(t1, b)
        if t0 > t1:
            return None
    return t0, t1


def stretches(bins, L, gap=2):
    """Runs of marked bins along a path of length L: [(s0, s1, bits)] with gaps of up to `gap` bins bridged."""
    out = []
    on = np.nonzero(bins)[0]
    if not len(on):
        return out
    start = prev = int(on[0])
    bits = int(bins[on[0]])
    for j in on[1:]:
        j = int(j)
        if j - prev > gap + 1:
            out.append((start * L / NB, (prev + 1) * L / NB, bits))
            start, bits = j, 0
        bits |= int(bins[j])
        prev = j
    out.append((start * L / NB, (prev + 1) * L / NB, bits))
    return out


def track_samples(med, core, n=TRACK):
    """A bullet's track through a medium (Ballistics.media entry) as the kernels take it: [(s, K, R, speed)] at n depths,
    K and R from the energy it left per metre there. The flow it drives is u = K / r out to R, and (as the grid can carry
    it: within `core` of the path the two sides' flows would cancel on its nodes) u = K r / core^2 inside `core`; K is
    what gives that flow the energy's share: KE per metre = pi rho K^2 (1/4 + ln(R / core))."""
    tr = med['track']
    if len(tr) < 2:
        return np.zeros((n, 4), np.float32)
    s = np.array([t[0] for t in tr], float)
    v = np.array([t[1] for t in tr], float)
    area = np.array([t[2] for t in tr], float)
    p_cav, eta, _depth, _share = MEDIA.get(med['surface'], MEDIA['gel'])
    rho = SURFACES[med['surface']].rho if med['surface'] in SURFACES else 1000.0
    m = med.get('mass', 0.008)
    ds = np.maximum(np.diff(s), 1e-6)
    dEds = np.maximum(0.5 * m * (v[:-1] ** 2 - v[1:] ** 2) / ds, 0.0)
    dEds = np.append(dEds, dEds[-1] if len(dEds) else 0.0)
    a = np.sqrt(area / math.pi)
    c = np.maximum(a, core)
    R = np.clip(np.sqrt(dEds / (math.pi * p_cav)), 2.0 * c, 0.25)
    K = np.sqrt(eta * dEds / (math.pi * rho * (0.25 + np.log(R / c))))
    pick = np.linspace(0, len(s) - 1, n).round().astype(int)
    return np.stack([s[pick], K[pick], R[pick], v[pick]], 1).astype(np.float32)


def reached(med, now):
    """How far along its track the bullet had got by simulation time `now` (m): by the time each stretch of it takes at
    the speed it went there."""
    tr = med['track']
    if not tr:
        return 0.0
    s = np.array([t[0] for t in tr], float)
    v = np.maximum(np.array([t[1] for t in tr], float), 1.0)
    t = np.concatenate([[0.0], np.cumsum(np.diff(s) / (0.5 * (v[:-1] + v[1:])))]) if len(s) > 1 else np.zeros(1)
    dt = now - med['time']
    if dt >= t[-1]:
        return float(s[-1])
    return float(np.interp(dt, t, s))


KEEP_S = 0.3           # s: how long a bullet's way through deep water is kept in the narrow band (its cavity's life)
HOLE_SHARE = 0.9       # a bullet's hole through cloth: its radius as a share of the bullet's (the weave closes a little)

class BulletMedia:
    """The kernels that let bullets meet the matter and the liquid, and push them."""

    def __init__(self, gpu):
        self.gpu = gpu
        self._k = {}
        self._occ = None

    def _kernel(self, name, bindings):
        k = self._k.get(name)
        if k is None:
            k = self._k[name] = self.gpu.kernel(name, bindings, workgroup=(64, 1, 1))
        return k

    def _occupancy(self):
        if self._occ is None:
            self._occ = self.gpu.buffer(MAX_PATHS * NB * 4, 'bullet-occupancy')
        return self._occ

    # -- looking ahead ----------------------------------------------------------------------------------------

    def look_ahead(self, shots, fdt, matter=None, liquid=None):
        """Where the bullets flying (or fired) this frame of fdt seconds cross the matter and the liquid:
        shots.ahead = {(shot, n): [(A, B, surface key, medium)]}."""
        shots.ahead = {}
        paths = self._paths(shots, fdt)
        if not paths:
            return
        if matter is not None and matter.active and matter.count:
            b = matter.world_bounds()
            if b is not None:
                self._find(shots, paths, b, 'matter', matter)
        if liquid is not None and getattr(liquid, 'dims', None) is not None and liquid.h:
            lo = np.asarray(liquid.origin, float)
            hi = lo + np.asarray(liquid.dims, float) * liquid.h
            self._find(shots, paths, (lo, hi), 'liquid', liquid)
        for v in shots.ahead.values():
            v.sort(key=lambda x: float(np.linalg.norm(x[0])))

    def reahead(self, shots, b, matter=None, liquid=None):
        """Look ahead again along the way bullet b has just turned onto (Ballistics._turned): what of the matter and the
        liquid lies on it joins shots.ahead."""
        sp = b.speed
        paths = [((b.shot, b.n), b.pos.copy(), b.vel / sp, REACH)]
        if matter is not None and matter.active and matter.count:
            wb = matter.world_bounds()
            if wb is not None:
                self._find(shots, paths, wb, 'matter', matter)
        if liquid is not None and getattr(liquid, 'dims', None) is not None and liquid.h:
            lo = np.asarray(liquid.origin, float)
            self._find(shots, paths, (lo, lo + np.asarray(liquid.dims, float) * liquid.h), 'liquid', liquid)

    @staticmethod
    def _paths(shots, fdt):
        """Each bullet's way on from now: [((shot, n), start, unit direction, length)]: those in flight from where they
        are, those fired this frame from the muzzle. As far as it goes (REACH): clipped to the matter's or the
        liquid's box, the whole of what lies on its way is found, however far into it the bullet gets this frame."""
        out = []
        for b in shots.bullets:
            if not b.alive:
                continue
            sp = b.speed
            if sp < 1.0:
                continue
            out.append(((b.shot, b.n), b.pos.copy(), b.vel / sp, REACH))
        # (rounds still to be fired this frame: from their muzzles toward their aim)
        sc = shots.scene
        for si, spec in enumerate(shots.specs):
            k0 = shots.fired[si]
            for k in range(k0, spec.count):
                t = (shots.t_first[si] + k * spec.interval) if shots.t_first[si] is not None else None
                f = spec.first if t is None else None
                if t is not None and t > shots.now + fdt:
                    break
                if t is None and k > 0:
                    break
                frame = f if f is not None else sc.start
                muzzle = np.asarray(sc.get(('shot', spec.index, 'position'), frame), float)
                aim = np.asarray(sc.get(('shot', spec.index, 'aim'), frame), float)
                d = aim - muzzle
                n = float(np.linalg.norm(d))
                if n < 1e-9:
                    continue
                for p in range(spec.round.pellets):
                    out.append(((si, k * spec.round.pellets + p), muzzle, d / n, REACH))
                if t is None:
                    break
        return out

    def _find(self, shots, paths, bounds, medium, obj):
        lo, hi = (np.asarray(x, float) for x in bounds)
        sel = []
        for key, A, u, L in paths:
            c = clip(A, u, L, lo, hi)
            if c is not None and c[1] - c[0] > 1e-4:
                sel.append((key, A + u * c[0], u, c[1] - c[0]))
        if not sel:
            return
        occ = self._occupancy()
        for i in range(0, len(sel), MAX_PATHS):
            chunk = sel[i:i + MAX_PATHS]
            self.gpu.write_buffer(occ, np.zeros(MAX_PATHS * NB, np.uint32))
            if medium == 'matter':
                m = obj
                r = 0.6 * m.dx
                u = Uniforms().v4(*m.dims, m.count).v4(*m.origin, m.dx).v4(len(chunk))
                for k in range(MAX_PATHS):
                    u.v4(*(tuple(chunk[k][1]) + (chunk[k][3],)) if k < len(chunk) else (0.0, 0.0, 0.0, 0.0))
                for k in range(MAX_PATHS):
                    u.v4(*(tuple(chunk[k][2]) + (r,)) if k < len(chunk) else (0.0, 0.0, 0.0, 0.0))
                with self.gpu.batch() as b:
                    b.run(self._kernel('bullet_mfind.wgsl', ['rbuf', 'buf']), [m._buf['P'], occ], u,
                          groups=groups_1d(m.count))
            else:
                L = obj
                u = L._grid(0.0).v4(len(chunk))
                for k in range(MAX_PATHS):
                    u.v4(*(tuple(chunk[k][1]) + (chunk[k][3],)) if k < len(chunk) else (0.0, 0.0, 0.0, 0.0))
                for k in range(MAX_PATHS):
                    u.v4(*(tuple(chunk[k][2]) + (0.0,)) if k < len(chunk) else (0.0, 0.0, 0.0, 0.0))
                with self.gpu.batch() as b:
                    b.run(self._kernel('bullet_lfind.wgsl', ['tex3d', 'buf']), [L.TYPE[0], occ], u,
                          groups=(-(-len(chunk) * NB // 64), 1, 1))
            bins = np.frombuffer(self.gpu.read_buffer(occ), np.uint32).reshape(MAX_PATHS, NB)
            # (gaps between the particles, or cells, along a path are bridged: up to one and a half of their spacing)
            spacing = 0.5 * obj.dx if medium == 'matter' else obj.h
            for k, (key, A, u_, L_) in enumerate(chunk):
                gap = max(2, int(math.ceil(1.5 * spacing * NB / max(L_, 1e-9))))
                for s0, s1, bits in stretches(bins[k], L_, gap):
                    if medium == 'matter':
                        slot = (bits & -bits).bit_length() - 1
                        mats = getattr(obj, '_mats', [])
                        mkey = mats[slot].key if 0 <= slot < len(mats) else 'sand'
                        surf = MATTER_SURFACES.get(mkey, 'sand')
                    else:
                        surf = 'water'
                        self._keep_band(shots, obj, A + u_ * s0, A + u_ * s1, key)
                    shots.ahead.setdefault(key, []).append((A + u_ * s0, A + u_ * s1, surf, medium))

    @staticmethod
    def _keep_band(shots, L, a, b, key):
        """Water with a narrow band (liquid.py) holds particles only near its surface: deeper, a bullet's cavity would have
        nothing to push aside. Its way through the water is kept in the band (liq_band_update.wgsl) while its cavity
        opens and closes (KEEP_S), as wide as the cavity gets, so particles are seeded along it first."""
        if not getattr(L, 'band', None):
            return
        cal = 0.009
        for bl in shots.bullets:
            if (bl.shot, bl.n) == key:
                cal = bl.calibre
        reach = max(14.0 * cal, 0.04) + 2.0 * L.h
        A = tuple(float(x) for x in a)
        kb = [k for k in (getattr(L, 'keep_band', None) or []) if k[3] > shots.now and not (k[4] == key and k[0] == A)]
        kb.append((A, tuple(float(x) for x in b), reach, shots.now + KEEP_S, key))
        L.keep_band = kb[-8:]

    # -- kicking ---------------------------------------------------------------------------------------------

    def kick(self, shots, matter=None, liquid=None):
        """Push the matter and the liquid along the bullets' tracks through them, as far as each bullet has got by now;
        tracks done are dropped."""
        keep = []
        for med in shots.media:
            obj = matter if med['medium'] == 'matter' else liquid
            if obj is None:
                continue
            depth = float(med['track'][-1][0]) if med['track'] else 0.0
            s0 = float(med.get('done', 0.0))
            s1 = reached(med, shots.now)
            if s1 > s0 or s0 == 0.0:
                self._kick(med, obj, s0, s1)
                med['done'] = max(s1, 1e-9)
            if med['done'] < depth - 1e-6:
                keep.append(med)
        shots.media = keep

    def _kick(self, med, obj, s0, s1):
        core = max(0.5 * float(med.get('calibre', 0.009)), CORE * (obj.dx if med['medium'] == 'matter' else obj.h))
        T = track_samples(med, core)
        depth = float(med['track'][-1][0]) if med['track'] else 0.0
        _p, _eta, ej_depth, ej_share = MEDIA.get(med['surface'], MEDIA['gel'])
        a = 0.5 * float(med.get('calibre', 0.009))
        exit_ = 1.0 if med.get('outcome') == 'through' else 0.0
        if med['medium'] == 'matter':
            m = obj
            if not m.count:
                return
            u = (Uniforms().v4(*m.dims, m.count).v4(*m.origin, m.dx).v4(*med['at'], s0).v4(*med['dir'], s1)
                 .v4(a, core, FASTEST['matter'], 0.0).v4(ej_depth, ej_share, depth, exit_))
            for row in T:
                u.v4(*row)
            u.raw(m._mat_bytes)
            with self.gpu.batch() as b:
                b.run(self._kernel('bullet_matter.wgsl', ['buf']), [m._buf['P']], u, groups=groups_1d(m.count))
            top = float(min(FASTEST['matter'], np.max(T[:, 1]) / core + 0.12 * np.max(T[:, 3])))
            m.max_speed = max(m.max_speed, top)
        else:
            L = obj
            u = (L._grid(0.0).v4(*med['at'], s0).v4(*med['dir'], s1).v4(a, core, FASTEST['liquid'], L.capacity)
                 .v4(ej_depth, ej_share, depth, exit_))
            for row in T:
                u.v4(*row)
            with self.gpu.batch() as b:
                b.run(self._kernel('bullet_liquid.wgsl', ['buf']), [L.parts], u, groups=groups_1d(L.capacity))
            L.max_speed = max(getattr(L, 'max_speed', 0.0), FASTEST['liquid'])

    # -- fabric ----------------------------------------------------------------------------------------------

    def _cloth_crossings(self, shots, cloth, segs):
        """Where the segments (a, b, bullet radius) cross the cloth's triangles now: holes (Ballistics.cloth_holes)."""
        B = cloth.built
        X, gone = cloth.positions()
        tri = np.asarray(B.tris[:, :3], np.int64)
        ok = ~gone[tri].any(axis=1)
        v0, v1, v2 = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
        e1, e2 = v1 - v0, v2 - v0
        UV = np.asarray(B.uv, float)
        for a, b, rb in segs:
            d = b - a
            L = float(np.linalg.norm(d))
            if L < 1e-9:
                continue
            dn = d / L
            pv = np.cross(dn, e2)
            det = np.einsum('ij,ij->i', e1, pv)
            good = ok & (np.abs(det) > 1e-14)
            inv = np.where(good, 1.0 / np.where(good, det, 1.0), 0.0)
            tv = a - v0
            u = np.einsum('ij,ij->i', tv, pv) * inv
            qv = np.cross(tv, e1)
            v = (qv @ dn) * inv
            t = np.einsum('ij,ij->i', e2, qv) * inv
            hit = good & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t >= 0.0) & (t <= L)
            for k in np.nonzero(hit)[0][:4]:
                w = np.array([1.0 - u[k] - v[k], u[k], v[k]])
                q = w @ UV[tri[k], :2]
                shots.cloth_holes.append((float(q[0]), float(q[1]), float(rb * HOLE_SHARE), float(UV[tri[k, 0], 2])))
        if len(shots.cloth_holes) > 256:
            del shots.cloth_holes[:-256]

    def cloth(self, shots, cloth, fdt):
        """The bullets' ways through this frame (fdt seconds) through the fabric: holes torn in it, and a push round them
        (bullet_cloth.wgsl)."""
        if cloth is None or not cloth.active or not getattr(cloth, 'placed', False) or cloth.built is None:
            return
        segs = []
        t0 = shots.now - fdt
        for b in shots.bullets:
            pts = [(t, p) for (t, p) in b.trail if t >= t0 - 1e-9]
            before = [(t, p) for (t, p) in b.trail if t < t0 - 1e-9]
            if before:
                pts = [before[-1]] + pts
            for (ta, pa), (tb, pb) in zip(pts[:-1], pts[1:]):
                if float(np.linalg.norm(np.asarray(pb) - np.asarray(pa))) > 1e-6:
                    segs.append((np.asarray(pa, float), np.asarray(pb, float), 0.5 * b.calibre))
        if not segs:
            return
        # where each crosses the cloth: its hole, in the weave's own coordinates (drawn as finely as the pixels,
        # however coarse the cloth: cloth_draw.wgsl), a little smaller than the bullet (the weave stretches round it
        # before its threads part)
        self._cloth_crossings(shots, cloth, segs)
        # (vertices are torn away only where the hole is bigger than the cloth's own grain: a slug, a load of shot
        # close up; otherwise the cloth round the hole is only pushed along its way)
        big = max(sg[2] for sg in segs) * HOLE_SHARE
        grain = 0.45 * float(getattr(cloth.built, 'mean_edge', 0.02))
        hole = big if big > grain else 0.0
        reach = 4.0 * max(big, grain)
        k = self._k.get('cloth')
        if k is None:
            k = self._k['cloth'] = self.gpu.kernel('bullet_cloth.wgsl', ['rbuf', 'buf', 'buf'], workgroup=(64, 1, 1))
        n = cloth.built.n
        for i in range(0, len(segs), 32):
            chunk = segs[i:i + 32]
            u = Uniforms().v4(n, len(chunk), hole, reach)
            for j in range(32):
                u.v4(*(tuple(chunk[j][0]) + (1.5,)) if j < len(chunk) else (0.0, 0.0, 0.0, 0.0))
            for j in range(32):
                u.v4(*(tuple(chunk[j][1]) + (0.0,)) if j < len(chunk) else (0.0, 0.0, 0.0, 0.0))
            with self.gpu.batch() as b:
                b.run(k, [cloth.bufs['X'], cloth.bufs['S'], cloth.bufs['V']], u, groups=(-(-n // 64), 1, 1))
