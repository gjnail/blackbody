"""Drawing clouds (engine/cloud.py): what each cell holds for the eye (cloud_sig.wgsl), the light reaching
each cell through the cloud as it is drawn (cloud_light.wgsl), then the eye rays through the cloud, the sky
behind and the ground under it (cloud_march.wgsl), into the renderer's targets, which its composite
finishes (exposure, bloom, the footage)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import camera as cam
from .gpu import Uniforms

# extinction per kg of condensate (m^2/kg): 3 / (2 rho r) for droplets of 10 um, ice crystals of 40 um,
# raindrops of 1 mm, snowflakes (fluffy: 100 kg/m^3, 2 mm) and graupel/hail (400 kg/m^3, 1.5 mm)
EXT = (150.0, 41.0, 1.5, 7.5, 2.5)
LUME_MAX = 64          # Lume's light grid: at most this many cells a side
LUME_RAYS = 16         # rays from each of its cells a pass
LUME_EVENTS = 512      # scatterings each ray follows (the droplets' phase, the cloud's own density) before it takes
                       # the last pass's light: long walks carry the light through a thick cloud in a few passes
FRESH_PASSES = 40      # passes from nothing (a cumulus then within 1% of Mitsuba path tracing it: docs/lume.md) ...
LUME_AVG = 12          # ... the last of them averaged (their grain)


@dataclass
class SkyLook:
    """How the sky looks: the sun (direction, colour and strength), the sky and haze colours, the ground,
    the clouds' brightness, their silver lining and multiple scattering, the haze's visibility (m)."""
    sun_azimuth: float = 40.0
    sun_elevation: float = 35.0
    sun: tuple = (3.0, 2.9, 2.7)
    sky: tuple = (0.25, 0.42, 0.85)
    horizon: tuple = (0.75, 0.82, 0.92)
    ground: tuple = (0.25, 0.27, 0.2)
    brightness: float = 1.0
    skylight: float = 1.0
    silver: float = 1.0
    multiple: float = 0.8
    shafts: float = 1.0
    visibility: float = 60000.0
    draw_sky: bool = True
    draw_ground: bool = True
    exposure: float = 0.0
    density: float = 1.0
    detail: float = 1.0


class CloudRenderer:
    def __init__(self, gpu, renderer):
        self.gpu = gpu
        self.renderer = renderer
        self.k_sig = gpu.kernel('cloud_sig.wgsl', ['utex3d', 'utex3d', 'st3d:rgba32float:w', 'st3d:rgba16float:w', 'rbuf'])
        self.k_light = gpu.kernel('cloud_light.wgsl', ['utex3d', 'st3d:rgba32float:w'])
        self.k_march = gpu.kernel('cloud_march.wgsl', ['utex3d', 'utex2d', 'st2d:rgba16float:w', 'st2d:rgba16float:w',
                                                       'st2d:rgba16float:w', 'st2d:rgba16float:w', 'utex3d', 'utex3d', 'utex3d'],
                                  workgroup=(8, 8, 1))
        self.k_lume = gpu.kernel('cloud_lume.wgsl', ['utex3d', 'utex3d', 'utex3d', 'utex3d', 'st3d:rgba16float:w',
                                                     'utex3d', 'st3d:rgba16float:w'], workgroup=(4, 4, 4))
        self.LIGHT = None
        self.MAT = None
        self.SIG = None
        self.LV = [None, None]      # Lume's light (cloud_lume.wgsl), ping-ponged between passes
        self.LD = [None, None]      # ... the way it flows (its first moment)
        self.lv_dims = None
        self.lv_key = None          # (what the light carried in LV[0] is of: the grid; a frame jump starts afresh)
        self._none3 = gpu.texture3d((1, 1, 1), 'rgba16float', 'cloud-lume-none')
        gpu.upload(self._none3, np.zeros((1, 1, 1, 4), np.float16))
        self.seed = 0

    def _ensure(self, dims):
        if self.LIGHT is None or self.LIGHT.size != tuple(dims):
            if self.LIGHT is not None:
                for t in (self.LIGHT, self.MAT, self.SIG):
                    t.destroy()
            self.LIGHT = self.gpu.texture3d(tuple(dims), 'rgba32float', 'cloud-light')
            self.MAT = self.gpu.texture3d(tuple(dims), 'rgba16float', 'cloud-ice')
            self.SIG = self.gpu.texture3d(tuple(dims), 'rgba32float', 'cloud-sigma')

    def _ensure_lume(self, dims):
        """Lume's grid: the cloud's, at most LUME_MAX cells a side (the light scattered many times is smooth)."""
        k = max(1, -(-max(dims) // LUME_MAX))
        lv = tuple(max(2, -(-int(d) // k)) for d in dims)
        if self.lv_dims != (lv, k):
            for t in self.LV + self.LD:
                if t is not None:
                    t.destroy()
            self.LV = [self.gpu.texture3d(lv, 'rgba16float', f'cloud-lume{i}') for i in range(2)]
            self.LD = [self.gpu.texture3d(lv, 'rgba16float', f'cloud-lume-dir{i}') for i in range(2)]
            for t in self.LV + self.LD:
                self.gpu.upload(t, np.zeros(lv[::-1] + (4,), np.float16))
            self.lv_dims = (lv, k)
            self.lv_key = None
        return lv, k

    def lume(self, b, C, look: SkyLook, passes, fresh, sd_l):
        """Lume's passes over the cloud's light (cloud_lume.wgsl): fresh, from nothing (FRESH_PASSES, the last LUME_AVG
        averaged); else `passes` more carried on from the last (a frame or a refinement later: the light changes slowly)."""
        lv, k = self._ensure_lume(C.dims)
        gain = 2.0 ** look.exposure
        sun_level = max(float(np.mean(look.sun)), 1e-3)
        sun_e = np.asarray(look.sun, float) * gain * math.pi
        sky = np.asarray(look.sky) * sun_level * 0.22 * gain
        hor = np.asarray(look.horizon) * sun_level * 0.3 * gain
        glow = np.asarray(look.sun, float) * gain   # (the clouds' Brightness: the march's, over all of it)
        total = FRESH_PASSES if fresh else max(1, int(passes))
        for i in range(total):
            self.seed += 1
            keep = 0.0 if (fresh and i < total - LUME_AVG) else (1.0 - 1.0 / (i - (total - LUME_AVG) + 2) if fresh else 0.5)
            u = (C._grid(0.0).v4(*lv, k).v4(*sd_l, float(self.seed * 9781 % 1000003)).v4(*sun_e, LUME_RAYS)
                 .v4(*sky, LUME_EVENTS).v4(*hor, keep).v4(*look.ground, 1.0 if look.draw_ground else 0.0).v4(*glow, look.silver))
            b.run(self.k_lume, [self.SIG, self.MAT, self.LIGHT, self.LV[0], self.LV[1], self.LD[0], self.LD[1]], u, lv)
            self.LV = [self.LV[1], self.LV[0]]
            self.LD = [self.LD[1], self.LD[0]]

    def render(self, b, C, A, B, precip, camstate: cam.CameraState, fire: cam.FireXform, look: SkyLook, size, seed=0,
               step=0.6, wind=(0.0, 0.0, 0.0), light=True, lume=None):
        """Draw the cloud solver C's fields A, B (textures: live, or uploaded from the cache) and its ground
        precipitation map into the renderer's beauty, emission, aux and mask. lume: (passes, fresh) for Lume to light
        the cloud (cloud_lume.wgsl), or None."""
        self._ensure(C.dims)
        r = self.renderer
        w, h = size
        sd = np.asarray(cam.sun_direction(look.sun_azimuth, look.sun_elevation), float)
        l2w = fire.local_to_world()
        w2l = np.linalg.inv(l2w)
        sd_l = w2l[:3, :3] @ sd
        sd_l = sd_l / (np.linalg.norm(sd_l) + 1e-12)
        u = (C._grid(0.0).v4(*sd_l, 0.0).v4(*(e * look.density for e in EXT[:4]))
             .v4(EXT[4] * look.density, 1.0, 1.0 if lume is not None else 0.0, 0.0))
        if light:
            b.run(self.k_sig, [A, B, self.SIG, self.MAT, C.base], u, C.dims)
            b.run(self.k_light, [self.SIG, self.LIGHT], u, C.dims)
            if lume is not None:
                self.lume(b, C, look, lume[0], lume[1], sd_l)
        eye_l = (w2l @ np.append(np.asarray(camstate.eye, float), 1.0))[:3]
        gain = 2.0 ** look.exposure
        sun_level = max(float(np.mean(look.sun)), 1e-3)
        sky = np.asarray(look.sky) * sun_level * 0.22
        horizon = np.asarray(look.horizon) * sun_level * 0.3
        nx, ny, nz = C.dims
        u = (Uniforms().m4(camstate.inv_view_proj).m4(w2l)
             .v4(*eye_l, C.scale).v4(w, h, step, seed % 65536).v4(nx, ny, nz, C.h).v4(*C.origin, look.visibility)
             .v4(*sd_l, 1.0).v4(*(np.asarray(look.sun) * gain), look.brightness)
             .v4(*(sky * gain), look.skylight).v4(*(horizon * gain), 1.0 if look.draw_sky else 0.0)
             .v4(*look.ground, 1.0 if look.draw_ground else 0.0).v4(look.silver, look.multiple, look.shafts, look.detail)
             .v4(*wind, C.time))
        if lume is not None and self.lv_dims is not None:
            lv, k = self.lv_dims
            u.v4(1.0, k, lv[1], lv[2]).v4(lv[0], 0.0, 0.0, 0.0)
            lt, ld = self.LV[0], self.LD[0]
        else:
            u.v4().v4()
            lt = ld = self._none3
        b.run(self.k_march, [self.LIGHT, precip, r.beauty, r.aux, r.emit, r.mask, self.MAT, lt, ld], u, (w, h, 1))
