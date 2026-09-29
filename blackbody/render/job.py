"""Render jobs: simulate at final quality and write any number of outputs in one pass.

Outputs:
  exr       multi-layer OpenEXR sequence of the fire element, scene-linear, premultiplied
            (RGBA beauty plus emission, glow, heat and depth layers)
  png       PNG sequence (8 or 16 bit) of the element with alpha, or of the composite
  video     ProRes / DNxHR / H.264 / H.265 / VP9 of the element or the composite (source audio kept)
  vdb       OpenVDB volume sequence of the simulation (density, temperature, flame, fuel, velocity; for
            a liquid: the surface as density, velocity, and the spray, foam and bubble densities)

A liquid element is rendered against the footage: the liquid refracts what is behind it, so its
pixels carry the footage as seen through the liquid (alpha 1 where there is liquid). Its 'heat'
layer is written as 'ground' (the multiplier the wet and shadowed ground puts on the footage), its
'temperature' layer as 'speed' (m/s).
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

import blackbody

from ..io.colour import view_transform
from ..io.images import element_to_display, float_to_uint, write_exr, write_png
from ..io.video import PROFILES, VideoWriter

log = logging.getLogger('blackbody.job')

CONTENT = ('element', 'composite')


@dataclass
class Output:
    kind: str = 'exr'                 # exr | png | video | vdb
    path: str = 'renders/fire.####.exr'
    content: str = 'element'          # element (fire with alpha) | composite (over the footage)
    profile: str = 'prores4444'       # video profile key
    layers: tuple = ('emission', 'glow', 'heat', 'depth', 'surface')
    compression: str = 'zip'
    half: bool = True
    bits: int = 16                    # PNG
    alpha_mode: str = 'premultiplied'  # element PNG / video: premultiplied or straight
    audio: bool = True
    quality: int | None = None

    def label(self):
        if self.kind == 'video':
            return f'{PROFILES[self.profile].label} · {self.content}'
        if self.kind == 'exr':
            return f'EXR sequence · {self.compression.upper()} · {"half" if self.half else "float"}'
        if self.kind == 'png':
            return f'PNG {self.bits}-bit sequence · {self.content}'
        return 'OpenVDB volume sequence'


def frame_path(pattern, frame):
    """Expand '####' (padded to the number of #) or '%04d' in a path pattern."""
    if '#' in pattern:
        return re.sub(r'#+', lambda m: f'{frame:0{len(m.group(0))}d}', pattern, count=1)
    if re.search(r'%0?\d*d', pattern):
        return pattern % frame
    p = Path(pattern)
    return str(p.with_name(f'{p.stem}.{frame:04d}{p.suffix}'))


def infer_output(path, content=None, profile=None):
    ext = Path(path).suffix.lower()
    if ext == '.exr':
        return Output('exr', path, content or 'element')
    if ext in ('.png',):
        return Output('png', path, content or 'element')
    if ext == '.vdb':
        return Output('vdb', path)
    if ext in ('.mov', '.mp4', '.webm', '.mxf'):
        c = content or ('element' if ext in ('.mov', '.webm') else 'composite')
        if profile is None:
            profile = {'.mp4': 'h264', '.webm': 'vp9_alpha'}.get(ext, 'prores4444' if c == 'element' else 'prores422hq')
        return Output('video', path, c, profile)
    raise ValueError(f'Cannot tell the output type from "{path}". Use .exr, .png, .vdb, .mov, .mp4 or .webm.')


class RenderJob:
    def __init__(self, scene, outputs, engine, frames=None, final=True, footage=None, size=None, samples=None,
                 motion_blur=None):
        self.scene = scene
        self.outputs = list(outputs)
        self.engine = engine
        r = scene.data['render']
        self.first, self.last = frames or (scene.start, scene.end)
        self.final = final
        self.footage = footage
        self.size = size or scene.output_size()
        self.samples = samples if samples is not None else (r['aa_samples'] if final else 1)
        self.motion_blur = r['motion_blur'] if motion_blur is None else motion_blur
        self.written = []
        self.cancelled = False

    def _plate(self, frame):
        if self.footage is None:
            return None
        off = int((self.scene.footage or {}).get('offset', 0))
        return self.footage.read(frame - self.scene.start + off)

    def _plate_fit(self):
        if self.footage is None:
            return (1.0, 1.0)
        W, H = self.size
        fa = self.footage.width / self.footage.height
        oa = W / H
        return (1.0, 1.0) if abs(fa - oa) < 1e-3 else ((oa / fa, 1.0) if fa > oa else (1.0, fa / oa))

    def run(self, progress=None, cancelled=None):
        sc, eng = self.scene, self.engine
        t_start = time.perf_counter()
        eng.prepare(sc, final=self.final)
        W, H = self.size
        writers = {}
        need_elem = any(o.content == 'element' and o.kind != 'vdb' for o in self.outputs)
        need_comp = any(o.content == 'composite' and o.kind != 'vdb' for o in self.outputs)
        need_vdb = [o for o in self.outputs if o.kind == 'vdb']
        audio_src = sc.footage['path'] if (sc.footage and self.footage is not None and self.footage.audio) else None
        total = self.last - self.first + 1
        try:
            for i, frame in enumerate(range(self.first, self.last + 1)):
                if (cancelled and cancelled()) or self.cancelled:
                    self.cancelled = True
                    break

                def sim_progress(frac, f, _i=i):
                    if progress:
                        progress((_i + 0.5 * frac) / total, f'Simulating frame {f}')

                eng.simulate_to(sc, frame, progress=sim_progress if i == 0 else None, cancelled=cancelled, cache=False)
                if need_elem:
                    liquid = sc.kind == 'liquid'
                    eng.render(sc, frame, (W, H), mode='fire', final=self.final, samples=self.samples,
                               motion_blur=self.motion_blur, plate=self._plate(frame) if liquid else None,
                               plate_fit=self._plate_fit())
                    aov = eng.aovs()
                    elem_lin = eng.linear_comp()
                    glow = self.engine.gpu.read(eng.renderer.bloom_tex) if any('glow' in o.layers for o in self.outputs if o.kind == 'exr') else None
                    for o in self.outputs:
                        if o.content == 'element' and o.kind != 'vdb':
                            self._write_element(o, frame, aov, elem_lin, glow, writers, audio_src)
                if need_comp:
                    eng.render(sc, frame, (W, H), mode='composite', final=self.final, samples=self.samples,
                               motion_blur=self.motion_blur, plate=self._plate(frame), plate_fit=self._plate_fit())
                    comp_lin = eng.linear_comp()
                    for o in self.outputs:
                        if o.content == 'composite' and o.kind != 'vdb':
                            self._write_comp(o, frame, comp_lin, writers, audio_src)
                for o in need_vdb:
                    from ..io.vdb import write_liquid_vdb_frame, write_vdb_frame
                    p = frame_path(o.path, frame)
                    if sc.kind == 'liquid':
                        write_liquid_vdb_frame(p, eng, sc, frame)
                    else:
                        write_vdb_frame(p, eng.solver, sc)
                    self.written.append(p)
                if progress:
                    progress((i + 1) / total, f'Frame {frame} of {self.last}')
        finally:
            for w in writers.values():
                try:
                    self.written.append(str(w.close()))
                except Exception as ex:
                    log.error('Closing %s failed: %s', w.path, ex)
        self._sidecar(time.perf_counter() - t_start)
        return self.written

    # -- writers ---------------------------------------------------------------------------------

    def _video(self, o, writers, audio_src):
        w = writers.get(id(o))
        if w is None:
            W, H = self.size
            start_s = ((self.scene.footage or {}).get('offset', 0) + self.first - self.scene.start) / self.scene.fps
            dur = (self.last - self.first + 1) / self.scene.fps
            w = VideoWriter(o.path, o.profile, W, H, self.scene.fps,
                            audio_source=audio_src if (o.audio and audio_src) else None,
                            audio_start=start_s, audio_duration=dur, quality=o.quality)
            writers[id(o)] = w
        return w

    def _write_element(self, o, frame, aov, elem_lin, glow, writers, audio_src):
        view = self.scene.data['composite']
        if o.kind == 'exr':
            b = aov['beauty'].astype(np.float32)
            e = aov['emission'].astype(np.float32)
            x = aov['aux'].astype(np.float32)
            dt = np.float16 if o.half else np.float32
            ch = {'R': b[..., 0], 'G': b[..., 1], 'B': b[..., 2], 'A': b[..., 3]}
            if 'emission' in o.layers:
                ch.update({'emission.R': e[..., 0], 'emission.G': e[..., 1], 'emission.B': e[..., 2], 'emission.A': e[..., 3]})
            if 'glow' in o.layers and glow is not None:
                g = _resize_to(glow.astype(np.float32), b.shape[:2])
                ch.update({'glow.R': g[..., 0], 'glow.G': g[..., 1], 'glow.B': g[..., 2]})
            liquid = self.scene.kind == 'liquid'
            if 'heat' in o.layers:
                ch['ground.Y' if liquid else 'heat.Y'] = x[..., 0]
            if 'depth' in o.layers:
                ch['depth.Z'] = x[..., 1]
            if 'surface' in o.layers and not liquid and 'surface' in aov:
                # fire light on the ground and on colliders in the shot (multiply with the plate and add),
                # the holdout matte of those colliders, and scorch where the fire has burnt
                sl = aov['surface'].astype(np.float32)
                mk = aov['mask'].astype(np.float32)
                ch.update({'light.R': sl[..., 0], 'light.G': sl[..., 1], 'light.B': sl[..., 2],
                           'holdout.Y': sl[..., 3], 'scorch.Y': mk[..., 0]})
            if 'temperature' in o.layers:
                if liquid:
                    ch['speed.Y'] = x[..., 2] * 10.0
                else:
                    ch['temperature.Y'] = x[..., 2] * 1000.0
            ch = {k: v.astype(dt) for k, v in ch.items()}
            p = frame_path(o.path, frame)
            write_exr(p, ch, o.compression, {'software': f'{blackbody.APP_NAME} {blackbody.__version__}'})
            self.written.append(p)
            return
        rgba = element_to_display(elem_lin[..., :3].astype(np.float32), elem_lin[..., 3].astype(np.float32),
                                  knee=view['knee'], mode=o.alpha_mode)
        if o.kind == 'png':
            p = frame_path(o.path, frame)
            write_png(p, float_to_uint(rgba, o.bits))
            self.written.append(p)
        elif o.kind == 'video':
            w = self._video(o, writers, audio_src)
            w.write(float_to_uint(rgba, 16 if PROFILES[o.profile].deep else 8))

    def _write_comp(self, o, frame, comp_lin, writers, audio_src):
        view = self.scene.data['composite']
        rgb = view_transform(comp_lin[..., :3].astype(np.float32), view['view'], view['knee'])
        rgba = np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,), np.float32)], -1)
        if o.kind == 'exr':
            c = comp_lin.astype(np.float16 if o.half else np.float32)
            p = frame_path(o.path, frame)
            write_exr(p, {'R': c[..., 0], 'G': c[..., 1], 'B': c[..., 2]}, o.compression)
            self.written.append(p)
        elif o.kind == 'png':
            p = frame_path(o.path, frame)
            write_png(p, float_to_uint(rgba[..., :3], o.bits))
            self.written.append(p)
        elif o.kind == 'video':
            w = self._video(o, writers, audio_src)
            w.write(float_to_uint(rgba, 16 if PROFILES[o.profile].deep else 8))

    def _sidecar(self, seconds):
        if not self.written:
            return
        first = Path(self.written[0])
        info = {
            'software': f'{blackbody.APP_NAME} {blackbody.__version__}', 'frames': [self.first, self.last],
            'size': list(self.size), 'samples': self.samples, 'motion_blur': self.motion_blur, 'seconds': round(seconds, 2),
            'outputs': [dict(asdict(o), layers=list(o.layers)) for o in self.outputs],
            'gpu': self.engine.gpu.name, 'scene': self.scene.to_dict(),
        }
        try:
            (first.parent / f'{self.scene.name or "render"}_render.json').write_text(json.dumps(info, indent=1), encoding='utf-8')
        except Exception:
            pass


def _resize_to(img, shape):
    """Nearest-neighbour + box resample of a small image (the bloom buffer) up to `shape`."""
    h, w = shape
    ih, iw = img.shape[:2]
    if (ih, iw) == (h, w):
        return img
    ys = np.clip(((np.arange(h) + 0.5) * ih / h).astype(int), 0, ih - 1)
    xs = np.clip(((np.arange(w) + 0.5) * iw / w).astype(int), 0, iw - 1)
    return img[ys][:, xs]
