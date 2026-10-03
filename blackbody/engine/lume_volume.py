"""Lume for the smoke and steam (wgsl/lume_volume.wgsl): the light scattered in the volume, traced as Lume traces the
set, for the ray march to read in place of its own estimate of everything but the key light's (and the lamps') first
scattering, and for the cloth and the grass (wgsl/lume_light.wgsl). Each frame a pass or a few (a bounce each),
carried from frame to frame; the viewer's blended into the last frame's so it settles as the frame holds still."""
from __future__ import annotations

import math

import numpy as np

from .gpu import Uniforms
from .renderer import BB_DEFINES, pack_look
from .solver import MAX_COLLIDERS, pack_colliders

FINAL = (3, 24)       # a final render's passes (bounces) and rays a cell each pass
FRESH_FINAL = 24      # a final render's passes with no light carried from a frame before (thick steam needs them)
JUMP_S = 0.5          # s: a frame this far from the last starts afresh
VIEWER = (1, 10)      # the viewer's: one pass a frame, blended into the last
VIEWER_KEEP = 0.5     # of the viewer's new pass, the share kept (the rest the last frame's) ...
VIEWER_SETTLE = 24    # ... falling as the picture holds still, so its passes add up: refined until it has this many
LV_MAX = 56           # cells on the longest side of the traced grid (the light scattered in smoke is smooth: its cost
                      # goes as the cells times the steps along each ray, the fourth power of this)


class LumeVolume:
    def __init__(self, gpu):
        self.gpu = gpu
        self.k_trace = gpu.kernel('lume_volume.wgsl', ['tex3d', 'tex3d', 'tex3d', 'tex2d', 'smp', 'tex3d', 'utex3d', 'tex3d',
                                                       'tex3d', 'st3d:rgba16float:w', 'st3d:rgba16float:w', 'rbuf', 'rbuf',
                                                       'tex3d'], entry='trace', defines=BB_DEFINES, workgroup=(4, 4, 4))
        self.k_compose = gpu.kernel('lume_volume_compose.wgsl', ['tex3d', 'tex3d', 'tex3d', 'smp', 'st3d:rgba16float:w',
                                                                 'st3d:rgba16float:w'], workgroup=(4, 4, 4))
        self.dims = None
        self.ld = None
        self.lv = [None, None]
        self.lvd = [None, None]     # the way each cell's light comes from (lume_volume.wgsl)
        self.L0m = self.L1m = None
        self.seed = 0
        self.fresh = True      # (no light carried from an earlier frame)
        self.t_last = None
        self.key = None        # the picture the viewer's passes are of (Engine._lume_volume), and how many it has
        self.held = 0
        self.last_final = False

    @property
    def pending(self):
        """Whether the viewer should refine again: the picture held still has not had VIEWER_SETTLE passes yet."""
        return self.key is not None and not self.last_final and self.held < VIEWER_SETTLE

    def _ensure(self, ld):
        """LV at half the light volume's size, at most LV_MAX cells a side (the scattered light is smooth), and the
        march's light volume."""
        ld = tuple(int(x) for x in ld)
        k = min(0.5, LV_MAX / max(ld))
        lv = tuple(max(4, int(round(x * k))) for x in ld)
        if self.dims != lv:
            for t in self.lv + self.lvd:
                if t is not None:
                    t.destroy()
            self.lv = [self.gpu.texture3d(lv, 'rgba16float', f'lume-vol{i}') for i in range(2)]
            self.lvd = [self.gpu.texture3d(lv, 'rgba16float', f'lume-vold{i}') for i in range(2)]
            for t in self.lv + self.lvd:
                self.gpu.upload(t, np.zeros((lv[2], lv[1], lv[0], 4), np.float16))
            self.dims = lv
            self.fresh = True
        if self.ld != ld:
            for t in (self.L0m, self.L1m):
                if t is not None:
                    t.destroy()
            self.L0m = self.gpu.texture3d(ld, 'rgba16float', 'lume-L0')
            self.L1m = self.gpu.texture3d(ld, 'rgba16float', 'lume-L1')
            self.ld = ld

    def reset(self):
        """Forget the light carried from earlier frames (a cut, a jump in time)."""
        self.fresh = True

    def compute(self, b, r, solver, look, fire, colliders, meshes, albedos, floor_albedo, ground, ground_y, final, time=0.0,
                key=None):
        """Trace this frame's light into the volume (batch b) and set the renderer's light volume for the march
        (r.L0_march, r.L1_march). r: the Renderer, its light volume made this frame (Renderer.light); albedos: per
        collider (colliders' order) (rgb, whether rays stop at it); floor_albedo: the ground's (rgb); key: what the
        picture is of (the same key again: the viewer refining it, its passes averaged)."""
        from . import camera as cam
        self._ensure(r.light_dims)
        if self.t_last is not None and abs(time - self.t_last) > JUMP_S:
            self.fresh = True
        self.t_last = time
        if key != self.key:
            self.key, self.held = key, 0
        self.last_final = final
        passes, rays = FINAL if final else VIEWER
        keep = 1.0 if (final or self.fresh) else max(VIEWER_KEEP / (1.0 + self.held / 4.0), 1.0 / VIEWER_SETTLE)
        if self.fresh:
            passes = max(passes, FRESH_FINAL if final else 2)   # (from nothing: the bounces it needs)
        sd = np.asarray(cam.sun_direction(look.sun_azimuth, look.sun_elevation), float)
        sg = cam.rot_y(math.radians(fire.yaw)).T @ sd         # (the fire's frame: its yaw undone)
        step = float(np.linalg.norm(sg / (np.abs(sg).max() + 1e-9)))   # (Renderer.light's steps toward it, in cells)
        cols = list(colliders or [])[:MAX_COLLIDERS]
        alb = list(albedos or [])[:len(cols)] + [((0.0, 0.0, 0.0), 0.0)] * (len(cols) - len(albedos or []))
        atlas = meshes.atlas if meshes is not None else r._empty_r32
        for k in range(passes):
            self.seed = (self.seed + 1) % 1000003
            u = Uniforms().v4(*solver.dims, solver.h).v4(*solver.origin, ground_y)
            u.v4(*self.dims, rays).v4(*r.light_dims, 1.0 if ground else 0.0)
            u.v3(sg, keep if k == 0 else 1.0).v4(*floor_albedo, self.seed).v4(step)
            pack_look(u, look, r.log_y_ref(look.flame_k), time)
            pack_colliders(u, cols, meshes)
            for colour, on in alb:
                u.v4(*colour, 1.0 if on else 0.0)
            for _ in range(MAX_COLLIDERS - len(alb)):
                u.v4()
            (src, dst), (srcd, dstd) = self.lv, self.lvd
            b.run(self.k_trace, [solver.scal[0], r._aux_of(solver), r._chem_of(solver), r.bb, self.gpu.linear, r.L0, atlas,
                                 src, srcd, dst, dstd, r.lights, r.light_count, r.L1], u, self.dims)
            self.lv, self.lvd = [dst, src], [dstd, srcd]
        self.fresh = False
        if not final:
            self.held += 1
        cu = Uniforms().v4(*r.light_dims).v4(*self.dims, step)
        b.run(self.k_compose, [self.lv[0], r.L0, r.L1, self.gpu.linear, self.L0m, self.L1m], cu, r.light_dims)
        r.L0_march, r.L1_march = self.L0m, self.L1m
        r.LV_lume, r.LVD_lume = self.lv[0], self.lvd[0]      # (for the cloth and the grass: lume_light.wgsl)


def march_look(look):
    """The look the ray march uses with Lume's light volume: Fire light scatter folded into Lume's light already (1),
    and no estimate of its own of the light scattered more than once (Lume's holds it)."""
    from dataclasses import replace
    return replace(look, fire_scatter=1.0, multiple_scattering=0.0)
