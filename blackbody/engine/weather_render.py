"""Drawing the weather (engine/weather.py) into a render: the falling and resting precipitation particles
(wx_draw.wgsl, over what the march drew, hidden behind it) and what lies on the ground (wx_cover_shade.wgsl:
the snow cover traced as a height field, glaze and wet ground as a clear reflecting coat)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import camera as cam
from .gpu import Uniforms, load_wgsl, SS
from .weather import PACK_BYTES


@dataclass
class WeatherView:
    """What to draw: the packed particles (a GPU buffer and count), the cover and surface maps (2D textures),
    the weather area (x0, z0, x1, z1), the range of heights the cover spans, and the look."""
    packed: object
    count: int
    cover: object
    surf: object
    area: tuple
    map_dims: tuple
    yr: tuple
    look: dict
    haze: tuple = (0.0, 0.0, 0.0, 0.0)   # the fall's haze past the area: colour weights (snow, rain), extinction (1/m), top (m)


class WeatherRenderer:
    def __init__(self, gpu, renderer):
        self.gpu = gpu
        self.renderer = renderer
        dev = gpu.device
        vis = SS.VERTEX | SS.FRAGMENT
        self._layout = dev.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': vis, 'buffer': {'type': 'read-only-storage'}},
            {'binding': 1, 'visibility': vis, 'texture': {'sample_type': 'float', 'view_dimension': '2d'}},
        ])
        module = dev.create_shader_module(code=load_wgsl('wx_draw.wgsl'), label='wx_draw')
        over = {'color': {'src_factor': 'one', 'dst_factor': 'one-minus-src-alpha', 'operation': 'add'},
                'alpha': {'src_factor': 'one', 'dst_factor': 'one-minus-src-alpha', 'operation': 'add'}}
        add = {'color': {'src_factor': 'one', 'dst_factor': 'one', 'operation': 'add'},
               'alpha': {'src_factor': 'zero', 'dst_factor': 'one', 'operation': 'add'}}
        matte = {'color': {'src_factor': 'zero', 'dst_factor': 'one', 'operation': 'add'},
                 'alpha': {'src_factor': 'one', 'dst_factor': 'one-minus-src-alpha', 'operation': 'add'}}
        self._pipe = dev.create_render_pipeline(
            layout=dev.create_pipeline_layout(bind_group_layouts=[self._layout, gpu.arena.layout]),
            vertex={'module': module, 'entry_point': 'vs', 'buffers': []},
            fragment={'module': module, 'entry_point': 'fs', 'targets': [
                {'format': 'rgba16float', 'blend': over}, {'format': 'rgba16float', 'blend': add},
                {'format': 'rgba16float', 'blend': matte}]},
            primitive={'topology': 'triangle-strip'}, label='wx-draw')
        self._bg = None
        self._bg_key = None
        self.k_cover = gpu.kernel('wx_cover_shade.wgsl', ['utex2d', 'utex2d', 'utex2d', 'utex2d', 'utex2d',
                                                          'st2d:rgba16float:w', 'st2d:rgba16float:w', 'st2d:rgba16float:w'],
                                  workgroup=(8, 8, 1))
        self._tmp = None
        self._cached = None      # (packed buffer, capacity) for cached frames
        self._cached_maps = None

    # -- cached frames -----------------------------------------------------------------------------

    def upload(self, packed: np.ndarray, cover: np.ndarray | None, surf: np.ndarray | None):
        """GPU copies of a cached frame's particles and maps; returns (buffer, count, cover tex, surf tex)."""
        g = self.gpu
        n = len(packed)
        need = max(n, 1) * PACK_BYTES
        if self._cached is None or self._cached[1] < need:
            if self._cached is not None:
                self._cached[0].destroy()
            self._cached = (g.buffer(max(need, 1 << 16), 'wx-cached-packed'), max(need, 1 << 16))
        if n:
            g.write_buffer(self._cached[0], np.ascontiguousarray(packed, np.float32))
        ctex = stex = None
        if cover is not None and surf is not None:
            nz, nx = cover.shape[:2]
            if self._cached_maps is None or self._cached_maps[0].size[:2] != (nx, nz):
                if self._cached_maps is not None:
                    for t in self._cached_maps:
                        t.destroy()
                self._cached_maps = (g.texture2d(nx, nz, 'rgba32float', 'wx-cached-cover'),
                                     g.texture2d(nx, nz, 'rgba32float', 'wx-cached-surf'))
            ctex, stex = self._cached_maps
            g.upload(ctex, np.ascontiguousarray(cover, np.float32))
            g.upload(stex, np.ascontiguousarray(surf, np.float32))
        return self._cached[0], n, ctex, stex

    # -- drawing -----------------------------------------------------------------------------------

    def cover(self, b, view: WeatherView, camstate: cam.CameraState, fire: cam.FireXform, look, size):
        """The snow cover, glaze and wet ground over what the march drew (and their distance into aux), and
        the haze of the fall past the area."""
        if view.cover is None or view.surf is None:
            return
        w, h = size
        r = self.renderer
        if self._tmp is None or self._tmp[0].size[:2] != (w, h):
            for t in self._tmp or ():
                t.destroy()
            self._tmp = tuple(self.gpu.texture2d(w, h, 'rgba16float', f'wx-{k}') for k in ('beauty', 'emit', 'aux'))
        tb, te, ta = self._tmp
        b.copy_texture(r.beauty, tb, (w, h, 1))
        b.copy_texture(r.emit, te, (w, h, 1))
        b.copy_texture(r.aux, ta, (w, h, 1))
        l2w = fire.local_to_world()
        w2l = np.linalg.inv(l2w)
        eye_l = (w2l @ np.append(np.asarray(camstate.eye, float), 1.0))[:3]
        sd = cam.sun_direction(look.sun_azimuth, look.sun_elevation)
        sd_l = w2l[:3, :3] @ np.asarray(sd, float)
        sd_l = sd_l / (np.linalg.norm(sd_l) + 1e-12)
        gain = 2.0 ** look.exposure
        nx, nz = view.map_dims
        x0, z0, x1, z1 = view.area
        cell = (x1 - x0) / max(nx, 1)
        u = (Uniforms().m4(camstate.inv_view_proj).m4(w2l)
             .v4(*eye_l, 0.0).v4(w, h, nx, nz).v4(*view.area)
             .v4(view.yr[0], view.yr[1], max(cell * 0.5, 0.005), max(0.1 * (x1 - x0), 0.2))
             .v4(*(np.asarray(look.sky) * gain), view.look.get('snow_bright', 1.0))
             .v4(*(np.asarray(look.sun) * gain), view.look.get('sparkle', 1.0))
             .v4(*sd_l, view.look.get('gloss', 1.0)))
        snow_w, rain_w, sigma, top = view.haze
        sky = np.asarray(look.sky) * gain
        sun = np.asarray(look.sun) * gain
        # snow scatters the sky and the sun round it: a bright haze; rain a dimmer, greyer one
        hcol = (sky * 1.0 + sun * 0.12) * snow_w + sky * 0.7 * rain_w
        u.v4(*hcol, sigma).v4(max(top, 1.0), 0.0, 0.0, 0.0)
        b.run(self.k_cover, [view.cover, view.surf, tb, te, ta, r.beauty, r.emit, r.aux], u, (w, h, 1))

    def draw(self, b, view: WeatherView, camstate: cam.CameraState, fire: cam.FireXform, look, size,
             jitter=(0.0, 0.0), shutter=0.0, time=0.0):
        """The particles over the render, as streaks over the shutter (or 1/96 s without motion blur)."""
        if view.count <= 0:
            return
        r = self.renderer
        key = (id(view.packed), id(r.aux))
        if self._bg is None or self._bg_key != key:
            self._bg = self.gpu.device.create_bind_group(layout=self._layout, entries=[
                {'binding': 0, 'resource': {'buffer': view.packed.buf, 'offset': 0, 'size': view.packed.size}},
                {'binding': 1, 'resource': r.aux.view}])
            self._bg_key = key
        w, h = size
        P = camstate.proj
        focal_px = 0.5 * h * math.sqrt(abs(P[0, 0] * P[1, 1] - P[0, 1] * P[1, 0]) * camstate.aspect)
        l2w = fire.local_to_world()
        gain = 2.0 ** look.exposure
        sd = cam.sun_direction(look.sun_azimuth, look.sun_elevation)
        u = (Uniforms().m4(camstate.view_proj).m4(l2w)
             .v4(*camstate.eye, time)
             .v4(w, h, max(shutter, 0.0), focal_px)
             .v4(*(np.asarray(look.sky) * gain), view.look.get('snow_bright', 1.0))
             .v4(*(np.asarray(look.sun) * gain), 0.0)
             .v3(sd, 0.0)
             .v4(*jitter, view.look.get('opacity', 1.0), 1.0))
        off = b.uniform_offset(u)
        rp = b.render_pass(color_attachments=[
            {'view': r.beauty.view, 'load_op': 'load', 'store_op': 'store'},
            {'view': r.emit.view, 'load_op': 'load', 'store_op': 'store'},
            {'view': r.mask.view, 'load_op': 'load', 'store_op': 'store'}])
        rp.set_pipeline(self._pipe)
        rp.set_bind_group(0, self._bg)
        rp.set_bind_group(1, self.gpu.arena.group, [off])
        rp.draw(4, view.count)
        rp.end()
