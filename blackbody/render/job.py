"""Render jobs: simulate at final quality and write any number of outputs in one pass.

Outputs:
  exr       multi-layer OpenEXR sequence of the fire element, scene-linear, premultiplied
            (RGBA beauty plus emission, glow, heat and depth layers)
  deep      deep OpenEXR sequence of the fire element (up to DEEP_SAMPLES samples per pixel, each
            with its colour, alpha and front and back depth), for deep compositing in Nuke
  png       PNG sequence (8 or 16 bit) of the element with alpha, or of the composite
  video     ProRes / DNxHR / H.264 / H.265 / VP9 of the element or the composite (source audio kept)
  vdb       OpenVDB volume sequence of the simulation (density, temperature, flame, fuel, velocity; for
            a liquid: the surface as density, velocity, and the spray, foam and bubble densities)

A liquid element is rendered against the footage: the liquid refracts what is behind it, so its
pixels carry the footage as seen through the liquid (alpha 1 where there is liquid). Its 'heat'
layer is written as 'ground' (the multiplier the wet and shadowed ground puts on the footage), its
'temperature' layer as 'speed' (m/s), and its 'surface' layers as mattes: water (the liquid surface),
foam, spray (the fine mist), drops (spray droplets) and bubbles (seen inside the liquid).
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
DEEP_SAMPLES = 8


@dataclass
class Output:
    kind: str = 'exr'                 # exr | png | video | vdb | deep | mesh (liquid surface or fabric: .obj per frame, or one .usd)
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
        if self.kind == 'mesh':
            return 'Liquid surface · ' + ('USD' if Path(self.path).suffix.lower().startswith('.usd') else 'OBJ sequence')
        if self.kind == 'deep':
            return 'Deep EXR sequence · element'
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
    if ext == '.exr' and (content == 'deep' or '.deep.' in Path(path).name.lower()):
        return Output('deep', path, 'element')
    if ext == '.exr':
        return Output('exr', path, content or 'element')
    if ext in ('.png',):
        return Output('png', path, content or 'element')
    if ext == '.vdb':
        return Output('vdb', path)
    if ext in ('.obj', '.usd', '.usdc', '.usda'):
        return Output('mesh', path)
    if ext in ('.mov', '.mp4', '.webm', '.mxf'):
        c = content or ('element' if ext in ('.mov', '.webm') else 'composite')
        if profile is None:
            profile = {'.mp4': 'h264', '.webm': 'vp9_alpha'}.get(ext, 'prores4444' if c == 'element' else 'prores422hq')
        return Output('video', path, c, profile)
    raise ValueError(f'Cannot tell the output type from "{path}". Use .exr, .png, .vdb, .obj, .usd, .mov, .mp4 or .webm.')


class RenderJob:
    def __init__(self, scene, outputs, engine, frames=None, final=True, footage=None, size=None, samples=None,
                 motion_blur=None, from_cache=False):
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
        self.from_cache = from_cache   # render frames already in the disk cache instead of simulating them
        self.holdout = None

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
        eng.cache_readonly = bool(self.from_cache)
        eng.prepare(sc, final=self.final)
        W, H = self.size
        writers = {}
        need_elem = any(o.content == 'element' and o.kind not in ('vdb', 'mesh') for o in self.outputs)
        deep = DEEP_SAMPLES if any(o.kind == 'deep' for o in self.outputs) else 0
        from ..io.holdout import FootageHoldout
        self.holdout = FootageHoldout(sc) if sc.kind != 'liquid' else None
        hold = self.holdout if (self.holdout is not None and self.holdout.active) else None
        need_comp = any(o.content == 'composite' and o.kind != 'vdb' for o in self.outputs)
        need_vdb = [o for o in self.outputs if o.kind == 'vdb']
        need_mesh = [o for o in self.outputs if o.kind == 'mesh'] if sc.kind in ('liquid', 'both') else []
        need_fabric = [o for o in self.outputs if o.kind == 'mesh'] if sc.fabrics else []
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

                if self.from_cache and frame in eng.cache:
                    pass   # simulated already (another machine, or an earlier run): render it from the disk cache
                elif self.from_cache:
                    raise RuntimeError(f'Frame {frame} is not in the disk cache; simulate it first (blackbody simulate).')
                else:
                    eng.simulate_to(sc, frame, progress=sim_progress if i == 0 else None, cancelled=cancelled,
                                    cache=eng.cache.disk is not None)
                holdout = hold.read(frame) if hold is not None else None
                if need_elem:
                    liquid = sc.kind == 'liquid'
                    # a liquid refracts the footage, so its element is rendered against it
                    eng.render(sc, frame, (W, H), mode='fire', final=self.final, samples=self.samples,
                               motion_blur=self.motion_blur, plate=self._plate(frame) if sc.kind in ('liquid', 'both') else None,
                               plate_fit=self._plate_fit(), holdout=holdout, deep=deep if sc.kind == 'fire' else 0)
                    aov = eng.aovs()
                    if liquid:
                        aov['mattes'] = eng.gpu.read(eng.renderer.mask)
                    elem_lin = eng.linear_comp()
                    glow = self.engine.gpu.read(eng.renderer.bloom_tex) if any('glow' in o.layers for o in self.outputs if o.kind == 'exr') else None
                    for o in self.outputs:
                        if o.kind == 'deep' and sc.kind == 'fire':
                            self._write_deep(o, frame)
                        elif o.kind == 'deep' and sc.kind == 'liquid':
                            self._write_deep_liquid(o, frame, aov)
                        elif o.content == 'element' and o.kind not in ('vdb', 'deep', 'mesh'):
                            self._write_element(o, frame, aov, elem_lin, glow, writers, audio_src)
                if need_comp:
                    eng.render(sc, frame, (W, H), mode='composite', final=self.final, samples=self.samples,
                               motion_blur=self.motion_blur, plate=self._plate(frame), plate_fit=self._plate_fit(),
                               holdout=holdout)
                    comp_lin = eng.linear_comp()
                    for o in self.outputs:
                        if o.content == 'composite' and o.kind not in ('vdb', 'mesh'):
                            self._write_comp(o, frame, comp_lin, writers, audio_src)
                for o in need_vdb:
                    from ..io.vdb import write_liquid_vdb_frame, write_vdb_frame
                    p = frame_path(o.path, frame)
                    if sc.kind == 'liquid':
                        write_liquid_vdb_frame(p, eng, sc, frame)
                    else:
                        write_vdb_frame(p, eng.solver, sc)
                    self.written.append(p)
                    if sc.kind == 'both':
                        # the liquid alongside the fire: name.liquid.####.vdb
                        pl = str(Path(p).with_name(Path(p).stem + '.liquid' + Path(p).suffix))
                        write_liquid_vdb_frame(pl, eng, sc, frame)
                        self.written.append(pl)
                if need_mesh:
                    from ..io.liquid_mesh import UsdWriter, liquid_surface_mesh, write_obj
                    mesh = liquid_surface_mesh(eng, sc, frame)
                    for o in need_mesh:
                        if Path(o.path).suffix.lower().startswith('.usd'):
                            w = writers.get(id(o))
                            if w is None:
                                w = writers[id(o)] = UsdWriter(o.path, sc.fps, sc.data['water']['droplet_size'])
                            w.add(frame, mesh)
                        else:
                            p = frame_path(o.path, frame)
                            write_obj(p, mesh)
                            self.written.append(p)
                if need_fabric:
                    from ..io import fabric_mesh as FM
                    meshes = FM.fabric_meshes(eng, sc, frame)
                    for o in need_fabric:
                        # beside a liquid's surface, the fabric goes to name.fabric.####.obj (or name.fabric.usd)
                        path = o.path
                        if sc.kind in ('liquid', 'both'):
                            q = Path(o.path)
                            path = str(q.with_name(q.stem + '.fabric' + q.suffix))
                        if Path(path).suffix.lower().startswith('.usd'):
                            w = writers.get((id(o), 'fabric'))
                            if w is None:
                                w = writers[(id(o), 'fabric')] = FM.UsdWriter(path, sc.fps)
                            w.add(frame, meshes)
                        else:
                            p = frame_path(path, frame)
                            FM.write_obj(p, meshes)
                            self.written.append(p)
                if progress:
                    progress((i + 1) / total, f'Frame {frame} of {self.last}')
        finally:
            if eng.cache.disk is not None:
                eng.cache.disk.flush()
            if self.holdout is not None:
                self.holdout.close()
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
            if 'surface' in o.layers and liquid and 'mattes' in aov:
                m = aov['mattes'].astype(np.float32)
                ch.update({'water.Y': m[..., 0], 'foam.Y': e[..., 3], 'spray.Y': m[..., 1], 'drops.Y': m[..., 3],
                           'bubbles.Y': m[..., 2]})
            if 'surface' in o.layers and not liquid and 'surface' in aov:
                # fire light on the ground and on colliders in the shot (multiply with the plate and add),
                # the holdout matte of those colliders, and scorch where the fire has burnt
                sl = aov['surface'].astype(np.float32)
                mk = aov['mask'].astype(np.float32)
                ch.update({'light.R': sl[..., 0], 'light.G': sl[..., 1], 'light.B': sl[..., 2],
                           'holdout.Y': mk[..., 2], 'scorch.Y': mk[..., 0], 'soot.Y': mk[..., 3], 'wet.Y': sl[..., 3]})
                if 'lamps' in aov and self.scene.lights:
                    # the lights in the set on the footage: multiply the plate by 1 + lamps
                    lp = aov['lamps'].astype(np.float32)
                    ch.update({'lamps.R': lp[..., 0], 'lamps.G': lp[..., 1], 'lamps.B': lp[..., 2]})
            if 'temperature' in o.layers:
                if liquid:
                    ch['speed.Y'] = x[..., 2] * 10.0
                else:
                    ch['temperature.Y'] = x[..., 2] * 1000.0
            attrs = {'software': f'{blackbody.APP_NAME} {blackbody.__version__}'}
            space = self._exr_space()
            if space:
                ch = self._to_space(ch, space)
                attrs['colorspace'] = space
            ch = {k: v.astype(dt) for k, v in ch.items()}
            p = frame_path(o.path, frame)
            write_exr(p, ch, o.compression, attrs)
            self.written.append(p)
            return
        pipe = self._pipe() if view['view'] == 'ocio' else None
        if pipe is not None:
            # through the OCIO display and view, exactly (the viewer uses a baked LUT)
            rgb = elem_lin[..., :3].astype(np.float32)
            a = np.clip(elem_lin[..., 3].astype(np.float32), 0.0, 1.0)
            if o.alpha_mode == 'straight':
                lum = np.clip(rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32), 0.0, 1.0)
                a2 = np.maximum(a, np.clip(lum * 1.2, 0.0, 1.0))
                rgb = np.where(a2[..., None] > 1e-5, rgb / np.maximum(a2[..., None], 1e-5), 0.0)
                a = a2
            rgba = np.concatenate([pipe.to_display(rgb), a[..., None]], -1)
        else:
            rgba = element_to_display(elem_lin[..., :3].astype(np.float32), elem_lin[..., 3].astype(np.float32),
                                      knee=view['knee'], mode=o.alpha_mode, white=view['highlight_white'])
        if o.kind == 'png':
            p = frame_path(o.path, frame)
            write_png(p, float_to_uint(rgba, o.bits))
            self.written.append(p)
        elif o.kind == 'video':
            w = self._video(o, writers, audio_src)
            w.write(float_to_uint(rgba, 16 if PROFILES[o.profile].deep else 8))

    def _pipe(self):
        from ..io import ocio
        try:
            return ocio.pipeline(self.scene.data['composite'])
        except Exception as ex:
            log.warning('OCIO: %s', ex)
            return None

    def _exr_space(self):
        return (self.scene.data['composite'].get('exr_space') or '').strip()

    def _to_space(self, ch, space):
        """Colour layers from the working space to an OCIO colour space (for EXRs)."""
        pipe = self._pipe()
        if pipe is None:
            return ch
        out = dict(ch)
        for pre in ('', 'emission.', 'glow.', 'light.'):
            keys = [pre + c for c in 'RGB']
            if all(k in ch for k in keys):
                rgb = pipe.to_space(np.stack([ch[k].astype(np.float32) for k in keys], -1), space)
                for i, k in enumerate(keys):
                    out[k] = rgb[..., i]
        return out

    def _write_deep(self, o, frame):
        from ..io.images import write_deep_exr
        samples = self.engine.renderer.read_deep()
        space = self._exr_space()
        if space:
            pipe = self._pipe()
            if pipe is not None:
                samples = samples.copy()
                samples[..., :3] = pipe.to_space(samples[..., :3], space)
        p = frame_path(o.path, frame)
        write_deep_exr(p, samples, {'software': f'{blackbody.APP_NAME} {blackbody.__version__}'})
        self.written.append(p)

    def _write_deep_liquid(self, o, frame, aov):
        """A liquid element as deep samples: the water surface at its depth (opaque where there is
        water), and the spray and droplets in front of it as a partly transparent sample."""
        from ..engine import camera as cam
        from ..io.images import write_deep_exr
        sc = self.scene
        b = aov['beauty'].astype(np.float32)
        x = aov['aux'].astype(np.float32)
        m = aov.get('mattes')
        m = m.astype(np.float32) if m is not None else np.zeros_like(b)
        h, w = b.shape[:2]
        spec, fire = sc.camera(frame)
        cs = cam.compute(spec, w / h, fire)
        # metres along the ray to metres along the view axis
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64) + 0.5
        ndc = np.stack([xs / w * 2 - 1, 1 - ys / h * 2, np.ones_like(xs), np.ones_like(xs)], -1)
        fpt = ndc @ cs.inv_view_proj.T
        ndc[..., 2] = 0.0
        npt = ndc @ cs.inv_view_proj.T
        rd = fpt[..., :3] / fpt[..., 3:] - npt[..., :3] / npt[..., 3:]
        rd /= np.linalg.norm(rd, axis=-1, keepdims=True)
        fwd = -cs.view[2, :3] / (np.linalg.norm(cs.view[2, :3]) + 1e-12)
        cosv = (rd @ fwd).astype(np.float32)
        water = m[..., 0] > 0.5
        z_water = np.where(water, x[..., 1] * cosv, 0.0)
        # spray and droplets: in front of the water, or around the middle of the box without any
        centre = np.asarray(fire.position, float) + np.array([0.0, 0.5 * sc.domain_size()[1], 0.0])
        z_mid = float((centre - cs.eye) @ fwd)
        haze_a = np.clip(1.0 - (1.0 - m[..., 1]) * (1.0 - m[..., 3]), 0.0, 1.0)
        samples = np.zeros((h, w, 2, 8), np.float32)
        # the water sample: its colour is what is left after the haze in front of it
        wa = np.where(water, 1.0, 0.0)
        samples[..., 0, :3] = np.where(water[..., None], b[..., :3], 0.0)
        samples[..., 0, 3] = wa
        samples[..., 0, 4] = z_water
        samples[..., 0, 5] = z_water
        # the haze: alpha from the mattes, colour from the element where there is no water
        z_h = np.where(water, np.maximum(z_water * 0.97, 0.01), z_mid)
        samples[..., 1, :3] = np.where(water[..., None], 0.0, b[..., :3])
        samples[..., 1, 3] = np.where(water, 0.0, np.clip(b[..., 3], 0.0, 1.0)) * (haze_a > 0.0)
        samples[..., 1, 4] = z_h
        samples[..., 1, 5] = z_h
        order = samples[..., 4].argsort(axis=-1)
        samples = np.take_along_axis(samples, order[..., None], axis=2)
        space = self._exr_space()
        if space:
            pipe = self._pipe()
            if pipe is not None:
                samples[..., :3] = pipe.to_space(samples[..., :3], space)
        p = frame_path(o.path, frame)
        write_deep_exr(p, samples, {'software': f'{blackbody.APP_NAME} {blackbody.__version__}'})
        self.written.append(p)

    def _write_comp(self, o, frame, comp_lin, writers, audio_src):
        view = self.scene.data['composite']
        pipe = self._pipe() if view['view'] == 'ocio' else None
        if pipe is not None:
            rgb = pipe.to_display(comp_lin[..., :3].astype(np.float32))
        else:
            rgb = view_transform(comp_lin[..., :3].astype(np.float32), view['view'], view['knee'], view['highlight_white'])
        rgba = np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,), np.float32)], -1)
        if o.kind == 'exr':
            ch = {'R': comp_lin[..., 0], 'G': comp_lin[..., 1], 'B': comp_lin[..., 2]}
            attrs = None
            space = self._exr_space()
            if space:
                ch = self._to_space({k: v.astype(np.float32) for k, v in ch.items()}, space)
                attrs = {'colorspace': space}
            dt = np.float16 if o.half else np.float32
            p = frame_path(o.path, frame)
            write_exr(p, {k: v.astype(dt) for k, v in ch.items()}, o.compression, attrs)
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
