"""Lume, Blackbody's path-traced lighting engine: the light on the set drawn in CG traced as it really travels
(wgsl/lume.wgsl, included by the stage), and its denoiser (wgsl/lume_denoise.wgsl).

Here: Lume's settings as the stage takes them (the Lume section), the HDRI's brightness map to pick light directions
from (so a small bright sun in it is found at once, not by chance), and the passes: each traces one path per pixel and
the stage adds them up, a few at a time in the viewer (it sharpens while you look) and all at once in a final render;
then the denoiser smooths away the grain left."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

from .gpu import Uniforms

ENV_WIDTH = 512           # the HDRI's brightness map: at most this many columns (and half as many rows)
VIEWER_PASSES = 2         # passes the viewer adds per refinement
FINAL_SUBMIT = 4          # a final render submits its passes this many at a time (each command list stays short)
FINAL_PER_PASS = 4        # paths per pixel each pass of a final render traces, at 1080p: more on fewer pixels
FINAL_PASS_MAX = 16       # ... at most (final_per_pass)
TARGETS = 8               # caustics: at most this many curved clear things the lights are traced through
CAUSTIC_SHARE = 16        # caustics: one light path per this many paths from the camera (per 4 pixels in the viewer)
MIRROR_ROUGH = 0.35       # caustics: bare metal smoother than this focuses light (lume.wgsl LU_MIRROR_ROUGH)
TAIL_FLOATS = 8 + 8 + 32 + 4 * TARGETS   # the stage's parameters after its materials: Lume's (lume, lume2, caustics)
ATROUS_STEPS = (1, 2, 4, 8, 16)


def final_per_pass(w, h, samples):
    """Paths per pixel each pass of a final render of w x h traces: FINAL_PER_PASS at 1080p, more on fewer pixels (up to
    FINAL_PASS_MAX: each pass's own cost, its launch and its sums, held up a small picture by a third), so a pass is never
    longer than one at 1080p (a dispatch running seconds would have the driver reset the GPU); fewer when there are not
    many samples."""
    n = min(FINAL_PASS_MAX, max(FINAL_PER_PASS, FINAL_PER_PASS * 1920 * 1080 // max(int(w) * int(h), 1)))
    return max(1, min(n, int(samples) // 4))


@dataclass
class LumeSettings:
    on: bool = False
    bounces: int = 4
    samples: int = 256
    viewer_samples: int = 64
    denoise: bool = True
    clamp: float = 0.0


def settings(scene) -> LumeSettings:
    """Lume's settings in a scene (an older project without them: Lume off)."""
    d = scene.data.get('lume') or {}
    return LumeSettings(on=d.get('engine', 'classic') == 'lume', bounces=int(d.get('bounces', 4)),
                        samples=max(1, int(d.get('samples', 256))), viewer_samples=max(1, int(d.get('viewer_samples', 64))),
                        denoise=bool(d.get('denoise', True)), clamp=float(d.get('clamp', 0.0)))


def env_table(img, width=ENV_WIDTH):
    """An HDRI (latitude-longitude, h x w x 3, scene-linear) as a brightness map to pick directions from: (the table as
    lume.wgsl reads it, its width, its height). The table: the rows' cumulative sums (h), then each row's (h x w), then
    each pixel's pdf over the picture (h x w: 1 everywhere for an even sky). A pixel's weight is its luminance times
    the sine of its latitude (the solid angle it covers), with a floor so no direction that has light is never picked."""
    img = np.asarray(img, np.float32)[..., :3]
    h0, w0 = img.shape[:2]
    w = int(min(width, w0))
    h = max(1, int(round(w * h0 / max(w0, 1))))
    lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    # box-average down to w x h (each output pixel the mean of the source pixels it covers)
    ys = (np.arange(h0) * h // h0).astype(int)
    xs = (np.arange(w0) * w // w0).astype(int)
    acc = np.zeros((h, w), np.float64)
    cnt = np.zeros((h, w), np.float64)
    np.add.at(acc, (ys[:, None], xs[None, :]), lum.astype(np.float64))
    np.add.at(cnt, (ys[:, None], xs[None, :]), 1.0)
    lum = acc / np.maximum(cnt, 1.0)
    st = np.sin((np.arange(h) + 0.5) / h * math.pi)
    f = np.maximum(lum, 0.0) * st[:, None]
    f = f + max(float(f.mean()), 1e-12) * 1e-3 * st[:, None]
    rows = f.sum(axis=1)
    total = float(rows.sum())
    marginal = np.cumsum(rows) / total
    cond = np.cumsum(f, axis=1) / rows[:, None]
    pdf = f / total * (w * h)
    marginal[-1] = 1.0
    cond[:, -1] = 1.0
    table = np.concatenate([marginal, cond.ravel(), pdf.ravel()]).astype(np.float32)
    return table, w, h


ENV_TRACE_SHARE = 0.25    # caustics: an HDRI with this much of its light in its brightest 0.5% (a sun) is traced


def env_peaked(table, w, h):
    """Whether an HDRI (its env_table) gathers its light in a small bright part (a sun): ENV_TRACE_SHARE of it or more in
    its brightest 0.5% of the sky. Its light is then traced from it for caustics (lume.wgsl caustics); an even sky's is
    found well by bouncing."""
    pdf = np.asarray(table[h + w * h:], np.float64).reshape(h, w)
    st = np.sin((np.arange(h) + 0.5) / h * math.pi)
    solid = np.broadcast_to(st[:, None], (h, w)).ravel()          # (each pixel's solid angle, relatively)
    power = pdf.ravel()                                            # (its share of the light, relatively)
    order = np.argsort(-(power / np.maximum(solid, 1e-9)))         # (brightest first)
    cum = np.cumsum(solid[order]) / solid.sum()
    top = order[: max(1, int(np.searchsorted(cum, 0.005)) + 1)]
    return float(power[top].sum() / max(power.sum(), 1e-30)) >= ENV_TRACE_SHARE


def caustic_targets(scene, cols, rows, meshes, matter=None):
    """What the lights are traced at for caustics (lume.wgsl caustics): the curved clear things in the set (glass, ice or
    jelly balls, cylinders, meshes, clear matter) and the mirrors (bare smooth metal, any shape), as spheres round them,
    fire-local (x, y, z, radius); and which they are (a bit for each object row, bit 16 the matter). Flat clear things
    (boxes, broken pieces) are left out: their straight-through shadow is exact."""
    from .solver import _mesh_ref
    out = []
    mask = 0
    for i, (c, row) in enumerate(zip(cols, rows)):
        if len(out) >= TARGETS:
            break
        glass = row[4] > 0.0 and c.shape in ('sphere', 'cylinder', 'mesh')
        mirror = row[4] <= 0.0 and row[3] >= 1.0 - 1e-6 and row[2] < MIRROR_ROUGH and c.shape in ('sphere', 'box', 'cylinder', 'mesh')
        if row[0] == 0 or not (glass or mirror):   # (0: not drawn)
            continue
        s = np.abs(np.asarray(c.size, float))
        if c.shape == 'sphere':
            r = s[0]
        elif c.shape == 'cylinder':
            r = math.hypot(s[0], s[1])
        elif c.shape == 'box':
            r = float(np.linalg.norm(s))
        else:
            m0, m1 = _mesh_ref(meshes, c.mesh, c.mesh_frame, c.mesh_fps)[:2]
            ext = np.maximum(np.abs(np.asarray(m0[:3], float)), np.abs(np.asarray(m1[:3], float)))
            r = float(np.linalg.norm(ext * s))
        out.append((float(c.pos[0]), float(c.pos[1]), float(c.pos[2]), float(r) * 1.02 + 1e-3))
        mask |= 1 << i
    if matter is not None and any(str(m.get('material', '')) == 'jelly' for m in (getattr(scene, 'matter', None) or [])):
        b = matter.world_bounds()
        if b is not None:
            lo, hi = np.asarray(b[0], float), np.asarray(b[1], float)
            if len(out) < TARGETS:
                out.append((*((lo + hi) / 2).tolist(), float(np.linalg.norm(hi - lo)) / 2))
                mask |= 1 << 16
    return out, mask


class Lume:
    """The stage's Lume state: the passes added up so far (ACC, AOV on the GPU), the HDRI's brightness map, and the
    denoiser."""

    def __init__(self, gpu):
        self.gpu = gpu
        self.acc = None
        self.aov = None
        self.cau = None
        self.size = None
        self.key = None            # what the passes added up so far were traced of
        self.passes = 0            # how many
        self.target = 0            # how many the viewer wants
        self.pending = False       # the viewer should refine again (it wants more passes of this picture)
        self._env = None
        self._env_key = None
        self.env_dims = (0, 0)
        self.env_traced = False   # the HDRI's light is traced for caustics (env_peaked)
        self._none = gpu.buffer(16, 'lume-no-env')
        self._tmp = [None, None]
        dn = ['rbuf', 'rbuf', 'tex2d', 'st2d:rgba16float:w']
        self.k_demod = gpu.kernel('lume_denoise.wgsl', dn, 'demod', workgroup=(8, 8, 1))
        self.k_atrous = gpu.kernel('lume_denoise.wgsl', dn, 'atrous', workgroup=(8, 8, 1))
        self.k_remod = gpu.kernel('lume_denoise.wgsl', dn, 'remod', workgroup=(8, 8, 1))

    def ensure(self, w, h):
        """The accumulation buffers for a w x h picture (afresh when the size changes)."""
        if self.size == (w, h):
            return
        for b in (self.acc, self.aov, self.cau, *self._tmp):
            if b is not None:
                b.destroy()
        n = w * h
        self.acc = self.gpu.buffer(n * 16, 'lume-acc')
        self.aov = self.gpu.buffer(n * 32, 'lume-aov')
        self.cau = self.gpu.buffer(n * 12, 'lume-caustics')
        self._tmp = [self.gpu.texture2d(w, h, 'rgba16float', f'lume-dn{i}') for i in range(2)]
        self.size = (w, h)
        self.key = None
        self.passes = 0

    def environment(self, path):
        """The brightness map of the HDRI at path (cached), or None."""
        if not path:
            self.env_dims = (0, 0)
            return None
        try:
            key = (path, os.path.getmtime(path))
        except OSError:
            self.env_dims = (0, 0)
            return None
        if key != self._env_key:
            from ..io.hdri import load_hdri
            table, w, h = env_table(load_hdri(path))
            self.env_traced = env_peaked(table, w, h)
            if self._env is not None:
                self._env.destroy()
            self._env = self.gpu.buffer(table.nbytes, 'lume-env')
            self.gpu.write_buffer(self._env, table)
            self.env_dims = (w, h)
            self._env_key = key
        return self._env

    def env_buffer(self):
        return self._env if (self._env is not None and self.env_dims[0] > 0) else self._none

    def plan(self, s: LumeSettings, key, final, samples):
        """Which passes to trace now: (first pass index, how many). A final render traces them all; the viewer's live
        picture (samples <= 1) one; a refinement carries on from where the last one stopped while it shows the same
        picture, a couple of passes at a time, until the viewer's samples are in."""
        if final:
            self.key, self.target, self.pending = None, s.samples, False
            return 0, s.samples
        if samples <= 1:
            self.key, self.target, self.pending = None, 1, False
            return 0, 1
        self.target = s.viewer_samples
        start = self.passes if (key == self.key and 0 < self.passes < self.target) else 0
        if key == self.key and self.passes >= self.target:
            start = self.target        # done: nothing more to trace (the picture is shown again as it is)
        self.key = key
        return start, max(0, min(VIEWER_PASSES, self.target - start))

    def finish(self, b, out_tex, passes, denoise):
        """After the passes: how many there are, and the denoiser over them into out_tex (the stage's picture)."""
        self.passes = passes
        self.pending = self.key is not None and passes < self.target
        if not denoise or passes <= 0:
            return
        w, h = self.size
        a, c = self._tmp
        b.run(self.k_demod, [self.acc, self.aov, out_tex, a], Uniforms().v4(w, h, passes, 1), (w, h, 1))
        src, dst = a, c
        for step in ATROUS_STEPS:
            b.run(self.k_atrous, [self.acc, self.aov, src, dst], Uniforms().v4(w, h, passes, step), (w, h, 1))
            src, dst = dst, src
        b.run(self.k_remod, [self.acc, self.aov, src, out_tex], Uniforms().v4(w, h, passes, 1), (w, h, 1))
