"""Embers and sparks: GPU particles carried by the simulated air, drawn as motion-blurred streaks."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import wgpu

from . import camera as cam
from .gpu import GPU, BU, SS, Uniforms, load_wgsl
from .solver import pack_emitters


@dataclass
class EmberParams:
    enabled: bool = True
    rate: float = 60.0            # new embers per second
    count: int = 16384            # pool size (oldest are recycled)
    lifetime: float = 2.2         # seconds
    life_jitter: float = 0.6
    temperature: float = 1850.0   # Kelvin at launch
    cooling: float = 0.8          # 1/s toward ambient
    drag: float = 3.0             # 1/s coupling to the air (for a 4 mm ember)
    gravity: float = 1.0          # m/s^2 (light, buoyant embers fall slowly)
    launch: float = 1.8           # m/s upward launch
    spread: float = 0.7           # m/s random launch
    turbulence: float = 3.0       # m/s^2
    turb_freq: float = 2.0        # 1/m
    size_min: float = 0.002       # m
    size_max: float = 0.010       # m
    brightness: float = 5.0
    min_width_px: float = 1.3
    fade_in: float = 0.06         # s
    shutter: float = 0.5          # fraction of a frame the streak covers
    direction: tuple = (0.0, 1.0, 0.0)  # launch direction (fire-local)
    cone: float = 0.0             # launch cone half-angle (degrees)
    collide: bool = True          # bounce off colliders
    bounce: float = 0.25          # share of the normal speed kept off a surface
    friction: float = 0.5         # share of the sliding speed lost on a hit


class Embers:
    def __init__(self, gpu: GPU):
        self.gpu = gpu
        self.count = 0
        self.cursor = 0
        self.accum = 0.0
        self.seed = 0
        self.k_update = gpu.kernel('embers.wgsl', ['tex3d', 'tex3d', 'smp', 'utex3d', 'utex3d', 'buf', 'buf', 'buf', 'buf', 'buf'],
                                   workgroup=(64, 1, 1))
        dev = gpu.device
        vis = SS.VERTEX | SS.FRAGMENT
        self.layout0 = dev.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': vis, 'buffer': {'type': 'read-only-storage'}},
            {'binding': 1, 'visibility': vis, 'buffer': {'type': 'read-only-storage'}},
            {'binding': 2, 'visibility': vis, 'buffer': {'type': 'read-only-storage'}},
            {'binding': 3, 'visibility': vis, 'texture': {'sample_type': 'float', 'view_dimension': '2d'}},
            {'binding': 4, 'visibility': vis, 'sampler': {'type': 'filtering'}},
            {'binding': 5, 'visibility': vis, 'buffer': {'type': 'read-only-storage'}},
            {'binding': 6, 'visibility': vis, 'texture': {'sample_type': 'float', 'view_dimension': '2d'}},
        ])
        module = dev.create_shader_module(code=load_wgsl('embers_draw.wgsl'), label='embers_draw')
        add = {'color': {'src_factor': 'one', 'dst_factor': 'one', 'operation': 'add'},
               'alpha': {'src_factor': 'zero', 'dst_factor': 'one', 'operation': 'add'}}
        self.pipe = dev.create_render_pipeline(
            layout=dev.create_pipeline_layout(bind_group_layouts=[self.layout0, gpu.arena.layout]),
            vertex={'module': module, 'entry_point': 'vs', 'buffers': []},
            fragment={'module': module, 'entry_point': 'fs', 'targets': [
                {'format': 'rgba16float', 'blend': add}, {'format': 'rgba16float', 'blend': add}]},
            primitive={'topology': 'triangle-strip'}, label='embers')
        self._bg = None
        self._bg_key = None

    def ensure(self, count):
        count = int(max(256, count))
        if count == self.count:
            return
        for name in ('A', 'B', 'C', 'D', 'E'):
            buf = getattr(self, name, None)
            if buf is not None:
                buf.destroy()
        g = self.gpu
        self.A, self.B, self.C, self.D, self.E = (g.buffer(count * 16, f'embers-{n}') for n in 'ABCDE')
        self.count = count
        self._bg = None
        self.reset()

    def reset(self):
        if not self.count:
            return
        zero = np.zeros((self.count, 4), np.float32)
        for buf in (self.A, self.B, self.C, self.D, self.E):
            self.gpu.write_buffer(buf, zero)
        self.cursor = 0
        self.accum = 0.0
        self.seed = 0

    def step(self, b, solver, prm: EmberParams, emitters, dt, cam_grid, smoke_ext, ambient_k=300.0):
        if not prm.enabled:
            return
        self.ensure(prm.count)
        # embers keep flying and cooling after every emitter has stopped; only spawning needs one
        self.accum += max(prm.rate, 0.0) * dt if emitters else 0.0
        spawn = int(self.accum)
        self.accum -= spawn
        spawn = min(spawn, self.count)
        u = solver._grid(dt)
        u.v4(self.cursor, spawn, self.count, self.seed)
        u.v4(prm.lifetime, prm.life_jitter, prm.temperature, prm.cooling)
        u.v4(prm.drag, prm.gravity, prm.launch, prm.spread)
        u.v4(prm.turbulence, prm.turb_freq, prm.size_min, prm.size_max)
        u.v3(cam_grid, smoke_ext)
        d = np.asarray(prm.direction, float)
        d = d / n if (n := float(np.linalg.norm(d))) > 1e-6 else np.array([0.0, 1.0, 0.0])
        u.v3(d, math.radians(max(0.0, min(180.0, prm.cone))))
        u.v4(prm.bounce, prm.friction, 1.0 if (prm.collide and solver.colliders) else 0.0)
        pack_emitters(u, emitters, ambient_k, meshes=solver.meshes)
        b.run(self.k_update, [solver.vel[0], solver.scal[0], self.gpu.linear, solver.sdf, solver.meshes.atlas,
                              self.A, self.B, self.C, self.D, self.E], u, (self.count, 1, 1))
        self.cursor = (self.cursor + spawn) % self.count
        self.seed += 1

    def draw(self, b, renderer, camstate: cam.CameraState, fire: cam.FireXform, prm: EmberParams, look,
             size, fps):
        if not prm.enabled or not self.count:
            return
        key = (id(renderer.bb), id(self.A), id(renderer.mask))
        if self._bg is None or self._bg_key != key:
            self._bg = self.gpu.device.create_bind_group(layout=self.layout0, entries=[
                {'binding': 0, 'resource': {'buffer': self.A.buf, 'offset': 0, 'size': self.A.size}},
                {'binding': 1, 'resource': {'buffer': self.B.buf, 'offset': 0, 'size': self.B.size}},
                {'binding': 2, 'resource': {'buffer': self.D.buf, 'offset': 0, 'size': self.D.size}},
                {'binding': 3, 'resource': renderer.bb.view},
                {'binding': 4, 'resource': self.gpu.linear},
                {'binding': 5, 'resource': {'buffer': self.E.buf, 'offset': 0, 'size': self.E.size}},
                {'binding': 6, 'resource': renderer.mask.view},
            ])
            self._bg_key = key
        w, h = size
        P = camstate.proj
        focal_px = 0.5 * h * math.sqrt(abs(P[0, 0] * P[1, 1] - P[0, 1] * P[1, 0]) * camstate.aspect)
        shutter = prm.shutter / max(fps, 1.0)
        gain = look.intensity * (2.0 ** look.exposure) * prm.brightness
        u = (Uniforms().m4(camstate.view_proj).m4(fire.local_to_world())
             .v4(w, h, shutter, gain)
             .v4(look.ambient_k, look.flame_k, look.dynamic_range, prm.min_width_px)
             .v4(renderer.log_y_ref(look.flame_k), prm.fade_in, focal_px))
        off = b.uniform_offset(u)
        rp = b.render_pass(color_attachments=[
            {'view': renderer.beauty.view, 'load_op': 'load', 'store_op': 'store'},
            {'view': renderer.emit.view, 'load_op': 'load', 'store_op': 'store'}])
        rp.set_pipeline(self.pipe)
        rp.set_bind_group(0, self._bg)
        rp.set_bind_group(1, self.gpu.arena.group, [off])
        rp.draw(4, self.count)
        rp.end()

    def cached_buffers(self):
        """The particle state a cached frame needs to draw its embers again."""
        return [self.A, self.B, self.D, self.E]

    def alive(self):
        a = np.frombuffer(self.gpu.read_buffer(self.A), np.float32).reshape(-1, 4)
        return int((a[:, 3] > 0).sum())
