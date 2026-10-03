"""Grass and plants: patches of blades simulated on the GPU, one thread to a blade (wgsl/strand_*.wgsl).

A blade is a chain of POINTS points from its root, where it grows, to its tip. At rest it stands up from its root and
arches over a little (its kind's lean); the air drags each point across it (the gas in the box, the wind outside it,
in gusts that roll downwind across the field), it springs back toward its rest shape from the root up (follow the
leader: each segment's rest direction turned with its parent) and keeps its length, and the ground and the objects
push it aside as they move through it. In hot gas it heats, catches and burns down to stubble, giving its fuel, heat
and smoke to the gas through the coupling grid cloth.py uses (cloth_feed.wgsl puts it in), so fire runs through a
field by itself, faster downwind.

It is drawn as ribbons into the raster layer (raster.py), with the cloth: strand_draw.wgsl. Cached frames keep each
blade's points as bytes off its root (a 1/127th of its height), and its fire as three bytes.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from .gpu import GPU, SS, Uniforms, groups_1d, load_wgsl

POINTS = 5                   # points a blade (its root and four joints up to its tip); matches strand_common.wgsl
MAX_PATCHES = 8              # matches strand_common.wgsl
MAX_BLADES = 400_000         # in all the patches (a thick one is thinned to fit)
STEPS_PER_SECOND = 300.0     # blade substeps a second of simulated time
FUEL_PER_KG = 5.0            # fuel (grid units x m^3) a kilogram of burning grass gives off (a fifth of cloth's: a
                             # field's flames are drawn as a ground fire's, not as the metres-long flames of a real one)
BLADE_DENSITY = 700.0        # kg/m^3: a fresh blade, water and all
GRASS_FLAME_K = 1200.0       # K: burning grass's flames
STUBBLE = 0.15               # what is left of a blade once it has burnt (a share of its height); matches strand_common.wgsl
COUPLE_CHANNELS = 13         # values per coupling cell (cloth_splat.wgsl's layout)
COUPLE_REF_FUEL = 2.0        # fuel density (F/s) at which burning grass's heat is fully in the gas (as cloth's)
PATCH_VEC4 = 6               # vec4s per patch (strand_common.wgsl Patch)


@dataclass(frozen=True)
class Kind:
    label: str
    height: float        # m
    density: float       # blades per m^2
    width: float         # m: a blade's width
    thick: float         # m: its thickness (how much it weighs for its area: how the wind moves it, how much fuel it is)
    sway: float          # Hz: how fast it springs back
    lean: float          # degrees its tip arches over at rest
    fresh: tuple         # colour fresh (scene-linear)
    dry: tuple           # and dry
    translucency: float  # the share of the light from behind that comes through it
    burn: float          # s a blade burns (fresh: half as long again)
    ear: bool = False    # an ear of grain at its tip


KINDS = {
    'lawn': Kind('Lawn', 0.08, 4000.0, 0.003, 0.0003, 4.0, 15.0, (0.06, 0.18, 0.025), (0.40, 0.36, 0.17), 0.40, 0.8),
    'meadow': Kind('Long grass', 0.45, 900.0, 0.005, 0.0003, 1.6, 32.0, (0.09, 0.22, 0.035), (0.52, 0.45, 0.22), 0.45, 1.5),
    'wheat': Kind('Wheat', 0.9, 350.0, 0.004, 0.0008, 2.2, 6.0, (0.17, 0.30, 0.07), (0.60, 0.48, 0.22), 0.30, 2.0, ear=True),
    'reeds': Kind('Reeds', 1.5, 120.0, 0.012, 0.0008, 0.9, 5.0, (0.14, 0.24, 0.06), (0.50, 0.43, 0.24), 0.35, 3.0),
}


@dataclass
class StrandSpec:
    kind: str = 'meadow'
    shape: str = 'box'            # 'box' (a rectangle) or 'disc'
    pos: tuple = (0.0, 0.0, 0.0)  # its middle, at the blades' feet (fire-local m)
    size: tuple = (1.0, 0.45, 1.0)  # half its width, the blades' height, half its depth (a disc: its radius, first)
    yaw: float = 0.0              # radians
    thickness: float = 1.0        # times its kind's blades per m^2
    dryness: float = 0.3
    burns: bool = True
    on_objects: bool = False      # grows on the objects that stay put too (else only on the ground)
    stiffness: float = 1.0        # times its kind's
    width: float = 1.0            # times its kind's blade width
    colour: tuple | None = None   # its own colour (else its kind's)
    seed: int = 0


def kind_of(spec):
    return KINDS.get(spec.kind, KINDS['meadow'])


def blades(spec: StrandSpec, patch: int, budget: int):
    """The blades of a patch as (n, 2, 4) float32: (root x, z (fire-local m), patch, a random number), (height (m),
    width (m), the azimuth its face looks along, the azimuth it leans toward). Spread evenly (a jittered grid) and in
    tufts: neighbours a little taller or shorter together. At most `budget` of them."""
    k = kind_of(spec)
    sx, h, sz = (abs(float(v)) for v in spec.size)
    disc = spec.shape == 'disc'
    if disc:
        sz = sx
    area = math.pi * sx * sx if disc else 4.0 * sx * sz
    n = min(int(round(area * k.density * max(float(spec.thickness), 0.0))), max(int(budget), 0))
    if n <= 0 or area <= 0.0:
        return np.zeros((0, 2, 4), np.float32)
    rng = np.random.default_rng(1009 + 7919 * int(spec.seed) + 104729 * int(patch))
    box_n = int(math.ceil(n * 4.0 * sx * sz / area))
    cols = max(1, int(round(math.sqrt(box_n * sx / max(sz, 1e-9)))))
    rows = max(1, int(math.ceil(box_n / cols)))
    gx, gz = np.meshgrid((np.arange(cols) + 0.5) / cols, (np.arange(rows) + 0.5) / rows)
    u = gx.ravel() + (rng.random(gx.size) - 0.5) / cols
    v = gz.ravel() + (rng.random(gz.size) - 0.5) / rows
    lx, lz = (2.0 * u - 1.0) * sx, (2.0 * v - 1.0) * sz
    if disc:
        inside = lx * lx + lz * lz <= sx * sx
        lx, lz = lx[inside], lz[inside]
    if len(lx) > n:
        pick = np.sort(rng.choice(len(lx), n, replace=False))
        lx, lz = lx[pick], lz[pick]
    # its margin: thinner and shorter toward its edge, raggedly (no patch of grass ends in a straight line)
    edge = (sx - np.hypot(lx, lz)) if disc else np.minimum(sx - np.abs(lx), sz - np.abs(lz))
    band = max(min(0.3, 0.3 * min(sx, sz)), 1e-6)
    ragged = edge + band * 0.45 * (rng.random(len(lx)) - 0.5)
    fall = np.clip(ragged / band, 0.0, 1.0)
    fall = fall * fall * (3.0 - 2.0 * fall)
    keep = rng.random(len(lx)) < 0.25 + 0.75 * fall
    lx, lz, fall = lx[keep], lz[keep], fall[keep]
    n = len(lx)
    cy, sy = math.cos(spec.yaw), math.sin(spec.yaw)
    x = float(spec.pos[0]) + cy * lx + sy * lz
    z = float(spec.pos[2]) - sy * lx + cy * lz
    # tufts: a few long waves of taller and shorter across the patch
    tuft = np.zeros(n)
    for _ in range(4):
        a = rng.random() * 2.0 * math.pi
        wl = 0.35 + 0.9 * rng.random()
        tuft += np.cos((x * math.cos(a) + z * math.sin(a)) * (2.0 * math.pi / wl) + rng.random() * 6.283)
    tuft = 0.5 + 0.125 * tuft
    r = rng.random(n)
    out = np.zeros((n, 2, 4), np.float32)
    out[:, 0, 0], out[:, 0, 1], out[:, 0, 2], out[:, 0, 3] = x, z, patch, r
    out[:, 1, 0] = h * (0.72 + 0.28 * rng.random(n)) * (0.82 + 0.36 * tuft) * (0.45 + 0.55 * fall)
    out[:, 1, 1] = k.width * float(spec.width) * (0.8 + 0.4 * rng.random(n))
    out[:, 1, 2] = rng.random(n) * 2.0 * math.pi
    out[:, 1, 3] = rng.random(n) * 2.0 * math.pi
    return out


def patch_data(specs):
    """The patches' settings as the kernels' Patch array: (MAX_PATCHES, PATCH_VEC4, 4) float32."""
    out = np.zeros((MAX_PATCHES, PATCH_VEC4, 4), np.float32)
    dt = 1.0 / STEPS_PER_SECOND
    for p, s in enumerate(list(specs)[:MAX_PATCHES]):
        k = kind_of(s)
        dryness = min(max(float(s.dryness), 0.0), 1.0)
        fresh = tuple(s.colour) if s.colour is not None else k.fresh
        dry = tuple(0.45 * c + 0.55 * d for c, d in zip(fresh, k.dry)) if s.colour is not None else k.dry
        sway = k.sway * math.sqrt(max(float(s.stiffness), 1e-3))
        bend = min((2.0 * math.pi * sway * dt) ** 2 * 6.0, 0.5)    # its share of the way back to its rest shape a substep
        drag = 1.2 * 1.2 / (2.0 * BLADE_DENSITY * k.thick)         # rho_air C_d / (2 rho_blade thickness), 1/m
        ign = 0.32 + (0.16 - 0.32) * dryness                       # hotter gas to light it fresh
        catch = 0.8 + (0.15 - 0.8) * dryness
        burn = k.burn * (1.5 + (0.8 - 1.5) * dryness)
        canopy = 0.5 * k.density * float(s.thickness) * k.width * float(s.width)
        out[p, 0] = (*fresh, k.translucency)
        out[p, 1] = (*dry, dryness)
        out[p, 2] = (bend, math.radians(k.lean), drag, 0.0)
        out[p, 3] = (1.0 if s.burns else 0.0, ign, catch, burn)
        out[p, 4] = (k.thick, 1.0 if k.ear else 0.0, canopy, abs(float(s.size[1])))
        out[p, 5] = (float(s.pos[1]), 1.0 if s.on_objects else 0.0, 1.2 + (0.35 - 1.2) * dryness, 0.0)
    return out


class Strands:
    def __init__(self, gpu: GPU):
        self.gpu = gpu
        self.specs = []
        self.n = 0
        self.placed = False
        self._key = None
        self.bufs = {}
        self.patches = patch_data([])
        self.couple_dims = None
        self.G = None
        self._couple_filled = False
        self._view = None
        self._pipe = None
        self._dummy_tex = None
        self.time = 0.0
        self.warnings = []
        g = gpu
        self.k_root = g.kernel('strand_root.wgsl', ['rbuf', 'buf', 'buf', 'buf', 'buf', 'buf', 'utex3d'], workgroup=(64, 1, 1))
        self.k_step = g.kernel('strand_step.wgsl', ['rbuf', 'rbuf', 'rbuf', 'buf', 'buf', 'buf', 'tex3d', 'tex3d', 'smp',
                                                    'utex3d', 'utex3d', 'utex3d'], workgroup=(64, 1, 1))
        self.k_sclear = g.kernel('strand_splat.wgsl', ['rbuf', 'rbuf', 'rbuf', 'rbuf', 'buf'], entry='clear', workgroup=(64, 1, 1))
        self.k_splat = g.kernel('strand_splat.wgsl', ['rbuf', 'rbuf', 'rbuf', 'rbuf', 'buf'], entry='splat', workgroup=(64, 1, 1))
        self.k_feed = g.kernel('cloth_feed.wgsl', ['utex3d', 'rbuf', 'st3d:rgba16float:w'], workgroup=(4, 4, 4))
        cres = ['rbuf', 'rbuf', 'rbuf', 'buf']
        self.k_cclear = g.kernel('strand_canopy.wgsl', cres, entry='clear', workgroup=(64, 1, 1))
        self.k_canopy = g.kernel('strand_canopy.wgsl', cres, entry='splat', workgroup=(64, 1, 1))
        vf = g.vel_format
        self.k_drag = g.kernel('strand_drag.wgsl', ['utex3d', 'rbuf', f'st3d:{vf}:w'], defines={'VELFMT': vf}, workgroup=(4, 4, 4))
        self.CA = None              # the canopy: blade area per coupling cell (strand_canopy.wgsl), for the wind's drag
        self.canopy_dims = None
        self._canopy_filled = False
        mres = ['rbuf', 'rbuf', 'rbuf', 'buf', 'st2d:rgba16float:w']
        self.k_mclear = g.kernel('strand_map.wgsl', mres, entry='clear', workgroup=(64, 1, 1))
        self.k_msplat = g.kernel('strand_map.wgsl', mres, entry='splat', workgroup=(64, 1, 1))
        self.k_mresolve = g.kernel('strand_map.wgsl', mres, entry='resolve', workgroup=(8, 8, 1))
        self._gmap = None      # the ground map: (texture, buffer, cells x, cells z, cell (m), corner (x, z))

    # -- building -------------------------------------------------------------------------------------

    @property
    def active(self):
        return self.n > 0

    @property
    def burns(self):
        return any(s.burns for s in self.specs)

    def configure(self, specs) -> bool:
        """Lay out the scene's patches of grass (StrandSpecs). True when they changed (they grow again: place())."""
        specs = list(specs)
        self.warnings = []
        if len(specs) > MAX_PATCHES:
            self.warnings.append(f'Only the first {MAX_PATCHES} patches of grass are simulated.')
            specs = specs[:MAX_PATCHES]
        key = repr([dataclasses.astuple(s) for s in specs])
        if key == self._key:
            return False
        self._key = key
        self.specs = specs
        parts, budget = [], MAX_BLADES
        for p, s in enumerate(specs):
            bl = blades(s, p, budget)
            budget -= len(bl)
            parts.append(bl)
        if budget <= 0:
            self.warnings.append(f'The grass is thinned to {MAX_BLADES:,} blades in all.')
        BL = np.concatenate(parts) if parts else np.zeros((0, 2, 4), np.float32)
        self._free()
        self.n = len(BL)
        self.BL = BL
        self.patches = patch_data(specs)
        if self.n:
            g = self.gpu
            self.bufs['BL'] = g.buffer(max(16, BL.nbytes), 'strands-blades')
            g.write_buffer(self.bufs['BL'], np.ascontiguousarray(BL.reshape(-1, 4)))
            for name, per in (('RT', 1), ('RN', 1), ('ST', 1), ('X', POINTS), ('XP', POINTS)):
                self.bufs[name] = g.buffer(self.n * per * 16, f'strands-{name}')
        self.placed = False
        self._view = None
        return True

    def _free(self):
        for b in self.bufs.values():
            b.destroy()
        self.bufs = {}
        if self._gmap is not None:
            self._gmap[0].destroy()
            self._gmap[1].destroy()
            self._gmap = None
        if self._view is not None:
            for b in self._view.values():
                b.destroy()
            self._view = None

    def reset(self):
        """Grow again (place()) at the next step."""
        self._canopy_filled = False
        self.placed = False
        self.time = 0.0

    def _dummy(self):
        if self._dummy_tex is None:
            self._dummy_tex = self.gpu.texture3d((1, 1, 1), 'rgba16float', 'strands-dummy')
            self.gpu.upload(self._dummy_tex, np.zeros((1, 1, 1, 4), np.float16))
        return self._dummy_tex

    def place(self, b, colliders, meshes, ground, ground_y, fixed_mask):
        """Where each blade grows (on the ground, or on what is under it), standing at rest. fixed_mask: a bit for each
        of `colliders` that stays put (grass grows on those)."""
        if not self.active:
            return
        from .solver import pack_colliders
        u = Uniforms().v4(self.n, 1.0 if ground else 0.0, ground_y, float(fixed_mask))
        pack_colliders(u, colliders, meshes)
        u.raw(self.patches.ravel())
        atlas = meshes.atlas if meshes is not None else self._dummy()
        k = self.bufs
        b.run(self.k_root, [k['BL'], k['RT'], k['RN'], k['X'], k['XP'], k['ST'], atlas], u, (self.n, 1, 1))
        self.placed = True
        self.time = 0.0

    def step(self, b, dt, solver, colliders, meshes, wind, gust, look, ground, ground_y):
        """Move every blade through dt seconds (a solver substep's): in the gas of `solver` (None: the wind only), round
        `colliders` as they are now."""
        if not self.active or not self.placed:
            return
        from .solver import pack_colliders
        steps = max(1, int(math.ceil(dt * STEPS_PER_SECOND - 1e-9)))
        h = dt / steps
        gas = solver is not None and solver.dims is not None
        u = Uniforms().v4(h, steps, self.n, 1.0 if gas else 0.0)
        u.raw((solver._grid(h) if gas else Uniforms().v4(1, 1, 1, 1).v4().v4()).data)
        pieces = gas and bool(getattr(solver, '_pieces_on', False))     # (broken pieces, people's and cars' parts)
        u.v4(*wind, gust).v4(look.ambient_k, look.flame_k, self.time).v4(1.0 if ground else 0.0, ground_y, 1.0 if pieces else 0.0)
        pack_colliders(u, colliders, meshes)
        u.raw(self.patches.ravel())
        vel = solver.vel[0] if gas else self._dummy()
        scal = solver.scal[0] if gas else self._dummy()
        atlas = meshes.atlas if meshes is not None else self._dummy()
        k = self.bufs
        bsdf = solver.sdf if pieces else self._dummy()
        bvel = solver.solid_vel() if pieces else self._dummy()
        b.run(self.k_step, [k['BL'], k['RT'], k['RN'], k['X'], k['XP'], k['ST'], vel, scal, self.gpu.linear, atlas, bsdf, bvel], u,
              (self.n, 1, 1))
        self.time += dt

    # -- the fire it feeds ------------------------------------------------------------------------------

    def prepare_frame(self, solver):
        """Size the coupling grids for the solver's grid (before a frame's commands are recorded)."""
        if not self.active or solver is None or solver.dims is None:
            return
        cd = tuple(int(math.ceil(d / 2)) for d in solver.dims)
        if cd != self.canopy_dims or self.CA is None:
            if self.CA is not None:
                self.CA.destroy()
            self.CA = self.gpu.buffer(int(np.prod(cd)) * 4, 'strands-canopy')
            self.canopy_dims = cd
            self._canopy_filled = False
        if not self.burns:
            return
        if cd != self.couple_dims or self.G is None:
            if self.G is not None:
                self.G.destroy()
            self.G = self.gpu.buffer(int(np.prod(cd)) * COUPLE_CHANNELS * 4, 'strands-couple')
            self.couple_dims = cd
            self._couple_filled = False

    def _couple_ok(self, solver):
        return (self.G is not None and solver is not None and solver.dims is not None
                and self.couple_dims == tuple(int(math.ceil(d / 2)) for d in solver.dims))

    def splat(self, b, solver, look):
        """The burning blades' fuel, heat and smoke onto the coupling grid, for the solver's next substep."""
        if not self.active or not self.placed or not self.burns or not self._couple_ok(solver):
            return
        cd = self.couple_dims
        cells = int(np.prod(cd))
        u = (Uniforms().v4(self.n, cells, FUEL_PER_KG, BLADE_DENSITY).v4(*cd, 2.0 * solver.h)
             .v4(*solver.origin, look.ambient_k).v4(look.flame_k, GRASS_FLAME_K))
        u.raw(self.patches.ravel())
        k = self.bufs
        res = [k['BL'], k['RT'], k['X'], k['ST'], self.G]
        b.run(self.k_sclear, res, u, groups=groups_1d(cells * COUPLE_CHANNELS))
        b.run(self.k_splat, res, u, (self.n, 1, 1))
        self._couple_filled = True

    def canopy(self, b, solver):
        """The blades' area onto the canopy grid as they stand now (strand_canopy.wgsl), for the wind's drag on them."""
        if not self.active or not self.placed or self.CA is None or solver is None or solver.dims is None:
            return
        cd = self.canopy_dims
        if cd != tuple(int(math.ceil(d / 2)) for d in solver.dims):
            return
        cells = int(np.prod(cd))
        u = Uniforms().v4(self.n, cells).v4(*cd, 2.0 * solver.h).v4(*solver.origin)
        k = self.bufs
        res = [k['BL'], k['RT'], k['X'], self.CA]
        b.run(self.k_cclear, res, u, groups=groups_1d(cells))
        b.run(self.k_canopy, res, u, groups=groups_1d(self.n))
        self._canopy_filled = True

    def hook(self, b, solver, dt, stage):
        """Solver.cloth_hook: the burning grass feeds the gas; the canopy slows the wind through it."""
        if stage == 'velocity':
            if self.active and self._canopy_filled and self.canopy_dims == tuple(int(math.ceil(d / 2)) for d in solver.dims):
                vd = tuple(d + 1 for d in solver.dims)
                u = Uniforms().v4(*solver.dims, solver.h).v4(*self.canopy_dims, 2.0 * solver.h).v4(dt)
                b.run(self.k_drag, [solver.vel[0], self.CA, solver.vel[1]], u, vd)
                solver.vel.reverse()
            return
        if not self.active or not self._couple_filled or not self._couple_ok(solver):
            return
        cd = self.couple_dims
        hc = 2.0 * solver.h
        if stage == 'sources':
            u = Uniforms().v4(*solver.dims, solver.h).v4(*cd, hc).v4(dt, COUPLE_REF_FUEL)
            b.run(self.k_feed, [solver.scal[0], self.G, solver.scal[1]], u, solver.dims)
            solver.scal.reverse()
        elif stage == 'sources_fine' and solver.scal_fine is not None:
            u = Uniforms().v4(*solver.dims_fine, solver.h / solver.upres).v4(*cd, hc).v4(dt, COUPLE_REF_FUEL)
            b.run(self.k_feed, [solver.scal_fine[0], self.G, solver.scal_fine[1]], u, solver.dims_fine)
            solver.scal_fine.reverse()

    # -- state ---------------------------------------------------------------------------------------

    def _read(self, name, per=1):
        a = np.frombuffer(self.gpu.read_buffer(self.bufs[name]), np.float32).reshape(-1, 4)
        return a[:self.n * per]

    def snapshot(self):
        """What a cached frame needs to draw the grass again: each blade's points as bytes off its root (127 a blade's
        height), and its fire (heat, burnt share, burning) as bytes."""
        if not self.active or not self.placed:
            return None
        x = self._read('X', POINTS).reshape(self.n, POINTS, 4)[:, :, :3]
        h = np.maximum(self.BL[:, 1, 0], 1e-6)[:, None, None]
        off = np.clip(np.round((x[:, 1:] - x[:, :1]) / h * 127.0), -127, 127).astype(np.int8)
        st = self._read('ST')
        fire = np.stack([np.round(np.clip(st[:, 0], 0.0, 1.0) * 255.0), np.round(np.clip(st[:, 1], 0.0, 2.0) * 127.5),
                         (st[:, 2] > 0.5) * 255.0], axis=1).astype(np.uint8)
        return {'off': off, 'fire': fire}

    def use_view(self, arrays):
        """Draw a cached frame's grass (snapshot()'s arrays), or the live grass (None)."""
        if arrays is None or not self.active:
            self._view_on = False
            return
        off, fire = np.asarray(arrays['off']), np.asarray(arrays['fire'])
        if len(off) != self.n:
            self._view_on = False
            return
        g = self.gpu
        if self._view is None:
            self._view = {'X': g.buffer(self.n * POINTS * 16, 'strands-view-X'), 'ST': g.buffer(self.n * 16, 'strands-view-ST')}
        root = self._read('RT')[:, :3]
        h = np.maximum(self.BL[:, 1, 0], 1e-6)[:, None, None]
        x = np.zeros((self.n, POINTS, 4), np.float32)
        x[:, 0, :3] = root
        x[:, 1:, :3] = root[:, None, :] + off.astype(np.float32) * (h / 127.0)
        st = np.zeros((self.n, 4), np.float32)
        st[:, 0] = fire[:, 0] / 255.0
        st[:, 1] = fire[:, 1] / 127.5
        st[:, 2] = (fire[:, 2] > 127).astype(np.float32)
        g.write_buffer(self._view['X'], x.reshape(-1, 4))
        g.write_buffer(self._view['ST'], st)
        self._view_on = True

    _view_on = False

    def _src(self):
        if self._view is not None and self._view_on:
            return self._view
        return self.bufs

    def save_state(self):
        if not self.active or not self.placed:
            return None
        return {'key': self._key, 'X': self._read('X', POINTS).copy(), 'XP': self._read('XP', POINTS).copy(),
                'ST': self._read('ST').copy(), 'RT': self._read('RT').copy(), 'RN': self._read('RN').copy(),
                'time': self.time}

    def load_state(self, st):
        if not st or not self.active or st.get('key') != self._key:
            return False
        for name in ('X', 'XP', 'ST', 'RT', 'RN'):
            self.gpu.write_buffer(self.bufs[name], np.ascontiguousarray(st[name], np.float32))
        self.time = float(st.get('time', 0.0))
        self.placed = True
        return True

    def positions(self):
        """(n, POINTS, 3) points of every blade now (fire-local m), and whether each grows."""
        x = self._read('X', POINTS).reshape(self.n, POINTS, 4)[:, :, :3].copy()
        return x, self._read('RT')[:, 3] > 0.5

    def state(self):
        """(n, 4): each blade's heat toward catching, burnt share (1..2: the stubble cooling), burning (1/0), _."""
        return self._read('ST').copy()

    # -- drawing -------------------------------------------------------------------------------------

    def ground_map(self, b):
        """The grass on the ground for the stage's floor (stage.py): (a texture of how burnt (r) and how thick (g) the
        grass is at each spot, the corner (x, z) it starts at, its size (x, z) in m), or None."""
        if not self.active or not self.placed:
            return None
        if self._gmap is None:
            lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
            for s in self.specs:
                r = (abs(s.size[0]), abs(s.size[0])) if s.shape == 'disc' else (abs(s.size[0]), abs(s.size[2]))
                if s.shape != 'disc' and abs(s.yaw) > 1e-6:
                    r = (math.hypot(*r),) * 2
                c = np.array([s.pos[0], s.pos[2]], float)
                lo, hi = np.minimum(lo, c - r), np.maximum(hi, c + r)
            ext = np.maximum(hi - lo, 1e-3)
            # (cells about one and a half blades apart, in the sparsest patch: nearly every cell has a blade in it)
            spacing = max(1.0 / math.sqrt(max(kind_of(s).density * float(s.thickness), 1e-3)) for s in self.specs)
            cell = max(0.02, float(ext.max()) / 512.0, 1.5 * spacing)
            nx, nz = (int(math.ceil(e / cell)) + 2 for e in ext)
            lo = lo - cell
            from .gpu import TU
            tex = self.gpu.texture2d(nx, nz, 'rgba16float', 'strands-ground', TU.TEXTURE_BINDING | TU.STORAGE_BINDING | TU.COPY_SRC)
            buf = self.gpu.buffer(nx * nz * 2 * 4, 'strands-ground-sums')
            self._gmap = (tex, buf, nx, nz, cell, (float(lo[0]), float(lo[1])))
        tex, buf, nx, nz, cell, lo = self._gmap
        src = self._src()
        k = self.bufs
        u = Uniforms().v4(self.n, nx, nz, cell).v4(*lo)
        res = [k['BL'], k['RT'], src['ST'], buf, tex]
        b.run(self.k_mclear, res, u, groups=groups_1d(2 * nx * nz))
        b.run(self.k_msplat, res, u, (self.n, 1, 1))
        b.run(self.k_mresolve, res, u, (nx, nz, 1))
        return tex, lo, (nx * cell, nz * cell)

    def _ensure_pipe(self):
        if self._pipe is not None:
            return
        g = self.gpu
        dev = g.device
        ent = [{'binding': i, 'visibility': SS.VERTEX | SS.FRAGMENT, 'buffer': {'type': 'read-only-storage'}} for i in range(4)]
        for i in (4, 5):
            ent.append({'binding': i, 'visibility': SS.FRAGMENT, 'texture': {'sample_type': 'float', 'view_dimension': '3d'}})
        ent.append({'binding': 6, 'visibility': SS.FRAGMENT, 'texture': {'sample_type': 'float', 'view_dimension': '2d'}})
        ent.append({'binding': 7, 'visibility': SS.FRAGMENT, 'sampler': {'type': 'filtering'}})
        for i in (8, 9, 10):
            ent.append({'binding': i, 'visibility': SS.FRAGMENT, 'buffer': {'type': 'read-only-storage'}})
        for i in (11, 12, 13, 14):   # (13, 14: Lume's light grids, lume_light.wgsl)
            ent.append({'binding': i, 'visibility': SS.FRAGMENT, 'texture': {'sample_type': 'float', 'view_dimension': '3d'}})
        self.layout0 = dev.create_bind_group_layout(entries=ent)
        from .renderer import BB_DEFINES
        module = dev.create_shader_module(code=load_wgsl('strand_draw.wgsl', BB_DEFINES), label='strand_draw')
        self._pipe = dev.create_render_pipeline(
            layout=dev.create_pipeline_layout(bind_group_layouts=[self.layout0, g.arena.layout]),
            vertex={'module': module, 'entry_point': 'vs', 'buffers': []},
            fragment={'module': module, 'entry_point': 'fs', 'targets': [
                {'format': 'rgba16float'}, {'format': 'rgba16float'}, {'format': 'rgba16float'}]},
            primitive={'topology': 'triangle-list', 'cull_mode': 'none'},
            depth_stencil={'format': 'depth32float', 'depth_write_enabled': True, 'depth_compare': 'less'},
            label='strands')

    def _no_lights(self):
        if getattr(self, '_nolights', None) is None:
            self._nolights = self.gpu.buffer(32, 'strands-no-lights')
            self._no_count = self.gpu.buffer(16, 'strands-no-light-count')
            self.gpu.write_buffer(self._no_count, np.zeros(4, np.uint32))
        return self._nolights

    def draw(self, b, rp, renderer, camstate, fire, look, size, jitter=(0.0, 0.0), solver=None, light_gain=1.0,
             fire_lights=True, lamp_count=None):
        """Draw the grass into the raster layer's render pass `rp` (raster.py) for one anti-aliasing pass. solver: the
        fire's volume (its light volume lights and shadows the grass), or None (a liquid scene: the key light and the
        sky only)."""
        if not self.active or not self.placed:
            return
        self._ensure_pipe()
        from . import camera as cam
        from .renderer import pack_look
        w, h = size
        g = self.gpu
        src = self._src()
        r = renderer
        lt = r.LT if (getattr(r, '_lamps_on', 0) and r.LT is not None) else r._empty
        vol_on = solver is not None and getattr(r, 'L0', None) is not None
        L0, L1, E = (r.L0, r.L1, r.E) if vol_on else (r._empty, r._empty, r._empty)
        if vol_on and getattr(r, 'LV_lume', None) is not None:
            L0 = r.L0_march   # (Lume's: the key light's shadows through the smoke as deep as they are; only .a is read)
        lights = r.lights if (fire_lights and r.lights is not None) else self._no_lights()
        if not (fire_lights and r.lights is not None):
            self._no_lights()
        count = r.light_count if (fire_lights and r.lights is not None) else self._no_count
        lume = vol_on and getattr(r, 'LV_lume', None) is not None
        k = self.bufs
        entries = [
            {'binding': 0, 'resource': {'buffer': src['X'].buf, 'offset': 0, 'size': src['X'].size}},
            {'binding': 1, 'resource': {'buffer': k['BL'].buf, 'offset': 0, 'size': k['BL'].size}},
            {'binding': 2, 'resource': {'buffer': k['RT'].buf, 'offset': 0, 'size': k['RT'].size}},
            {'binding': 3, 'resource': {'buffer': src['ST'].buf, 'offset': 0, 'size': src['ST'].size}},
            {'binding': 4, 'resource': L0.view},
            {'binding': 5, 'resource': L1.view},
            {'binding': 6, 'resource': r.bb.view},
            {'binding': 7, 'resource': g.linear},
            {'binding': 8, 'resource': {'buffer': lights.buf, 'offset': 0, 'size': lights.size}},
            {'binding': 9, 'resource': {'buffer': count.buf, 'offset': 0, 'size': count.size}},
            {'binding': 10, 'resource': {'buffer': r._lamp_buf.buf, 'offset': 0, 'size': r._lamp_buf.size}},
            {'binding': 11, 'resource': lt.view},
            {'binding': 12, 'resource': E.view},
            {'binding': 13, 'resource': (r.LV_lume if lume else r._empty).view},
            {'binding': 14, 'resource': (r.LVD_lume if lume else r._empty).view},
        ]
        bg = g.device.create_bind_group(layout=self.layout0, entries=entries)
        l2w = np.asarray(fire.local_to_world(), float)
        w2l = np.linalg.inv(l2w)
        eye = (w2l @ np.append(np.asarray(camstate.eye, float), 1.0))[:3]
        view = np.asarray(camstate.view)
        fwd_w = -view[2, :3] / max(np.linalg.norm(view[2, :3]), 1e-12)
        fwd = w2l[:3, :3] @ fwd_w
        fwd = fwd / max(np.linalg.norm(fwd), 1e-12)
        p0 = camstate.inv_view_proj @ np.array([0.0, 0.0, 0.0, 1.0])
        near = float(np.dot(p0[:3] / p0[3] - np.asarray(camstate.eye), fwd_w))
        sd = cam.sun_direction(look.sun_azimuth, look.sun_elevation)
        sd_l = w2l[:3, :3] @ np.asarray(sd, float)
        sd_l = sd_l / max(np.linalg.norm(sd_l), 1e-12)
        e_ref = float(getattr(look, 'fire_scatter', 1.0))
        dims = solver.dims if solver is not None else (1, 1, 1)
        hcell = solver.h if solver is not None else 1.0
        org = solver.origin if solver is not None else (0.0, 0.0, 0.0)
        ld = r.light_dims or (1, 1, 1)
        u = (Uniforms().m4(camstate.view_proj).m4(l2w).v4(w, h, *jitter).v4(*eye, near).v4(*fwd, 0.0)
             .v4(*sd_l, e_ref).v4(*org, hcell).v4(*ld, lamp_count if lamp_count is not None else (getattr(r, '_lamps_on', 0) if vol_on else 0))
             .v4(*(np.asarray(ld, float) / np.asarray(dims, float)),
                 (2.0 if getattr(r, 'occluded', False) else 1.0) if vol_on else 0.0))
        pack_look(u, look, r.log_y_ref(look.flame_k), getattr(look, 'time', 0.0))
        u.v4(light_gain, getattr(look, 'time', 0.0), 1.0 if lume else 0.0)
        u.raw(self.patches.ravel())
        off = b.uniform_offset(u)
        rp.set_pipeline(self._pipe)
        rp.set_bind_group(0, bg)
        rp.set_bind_group(1, g.arena.group, [off])
        rp.draw(self.n * (POINTS - 1) * 6, 1)
