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
                                                       'st2d:rgba16float:w', 'st2d:rgba16float:w', 'utex3d'], workgroup=(8, 8, 1))
        self.LIGHT = None
        self.MAT = None
        self.SIG = None

    def _ensure(self, dims):
        if self.LIGHT is None or self.LIGHT.size != tuple(dims):
            if self.LIGHT is not None:
                for t in (self.LIGHT, self.MAT, self.SIG):
                    t.destroy()
            self.LIGHT = self.gpu.texture3d(tuple(dims), 'rgba32float', 'cloud-light')
            self.MAT = self.gpu.texture3d(tuple(dims), 'rgba16float', 'cloud-ice')
            self.SIG = self.gpu.texture3d(tuple(dims), 'rgba32float', 'cloud-sigma')

    def render(self, b, C, A, B, precip, camstate: cam.CameraState, fire: cam.FireXform, look: SkyLook, size, seed=0,
               step=0.6, wind=(0.0, 0.0, 0.0), light=True):
        """Draw the cloud solver C's fields A, B (textures: live, or uploaded from the cache) and its ground
        precipitation map into the renderer's beauty, emission, aux and mask."""
        self._ensure(C.dims)
        r = self.renderer
        w, h = size
        sd = np.asarray(cam.sun_direction(look.sun_azimuth, look.sun_elevation), float)
        l2w = fire.local_to_world()
        w2l = np.linalg.inv(l2w)
        sd_l = w2l[:3, :3] @ sd
        sd_l = sd_l / (np.linalg.norm(sd_l) + 1e-12)
        u = (C._grid(0.0).v4(*sd_l, 0.0).v4(*(e * look.density for e in EXT[:4])).v4(EXT[4] * look.density, 1.0, 0.0, 0.0))
        if light:
            b.run(self.k_sig, [A, B, self.SIG, self.MAT, C.base], u, C.dims)
            b.run(self.k_light, [self.SIG, self.LIGHT], u, C.dims)
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
        b.run(self.k_march, [self.LIGHT, precip, r.beauty, r.aux, r.emit, r.mask, self.MAT], u, (w, h, 1))
