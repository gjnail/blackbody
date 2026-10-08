"""Render jobs: simulate at final quality and write any number of outputs in one pass.

Outputs:
  exr       multi-layer OpenEXR sequence of the fire element, scene-linear, premultiplied
            (RGBA beauty plus emission, glow, heat and depth layers), or of the composite; both with the compositing
            passes (render/passes.py): motion vectors, normals and positions, per-object mattes and Cryptomatte
  deep      deep OpenEXR sequence of the fire element (DEEP_SAMPLES, or up to DEEP_MAX, samples per
            pixel, each with its colour, alpha and front and back depth), for deep compositing in Nuke; in
            a shot with layers, every fire and liquid layer's samples in one image
  png       PNG sequence (8 or 16 bit) of the element with alpha, or of the composite
  video     ProRes / DNxHR / H.264 / H.265 / VP9 of the element or the composite (source audio kept)
  vdb       OpenVDB volume sequence of the simulation (density, temperature, flame, fuel, velocity; for
            a liquid: the surface as density, velocity, and the spray, foam and bubble densities; for a
            sky: its cloud water, ice, rain, snow and hail), live or from the disk cache
  mesh      the liquid's surface and the fabric as meshes (one .obj per frame, or one .usd)
  scene     the shot as a USD scene (io/scene_usd.py): its camera, the objects as they move and break, the
            ropes, the sand, snow and mud, the grass and the embers; content 'camera': the camera alone
  camera    the shot's camera as a Nuke .chan file (io/camera_out.py)

In a shot with layers, VDB, mesh and scene outputs hold every layer: the base layer's VDB and meshes go to
the path given and each other layer's beside it (name.<layer>.####.vdb), and a USD scene puts each other
layer under /World/<layer>. EXRs and deep EXRs carry the camera in their headers (worldToCamera, worldToNDC).

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
from . import passes as P
from .passes import PASSES

log = logging.getLogger('blackbody.job')

CONTENT = ('element', 'composite')
DEEP_SAMPLES = 8
DEEP_MAX = 16                     # the most samples per pixel the march gathers (raymarch.wgsl)
DEEP_KINDS = ('fire', 'liquid')   # the simulations deep samples are made for
FABRIC_KINDS = ('fire', 'liquid', 'both')
DATA_KINDS = ('vdb', 'mesh', 'scene', 'camera')   # outputs that are not pictures: no element or composite is rendered for them
PASS_KINDS = ('fire', 'liquid', 'both')   # the scenes with a set to write compositing passes of (a sky has none)
LOSSY_EXR = ('dwaa', 'dwab')   # EXR compressions that change the values (of every R, G, B and Y channel, float ones too)
KIND_NAMES = {'fire': 'a fire scene', 'liquid': 'a liquid scene', 'both': 'a fire-and-liquid scene', 'cloud': 'a sky scene'}


@dataclass
class Output:
    kind: str = 'exr'                 # exr | png | video | vdb | deep | mesh (liquid surface or fabric: .obj per frame, or one .usd)
                                      # | scene (a USD scene) | camera (a .chan file)
    path: str = 'renders/fire.####.exr'
    content: str = 'element'          # element (fire with alpha) | composite (over the footage); a mesh: liquid, fabric,
                                      # or element (the liquid's surface, with the fabric beside it as name.fabric.*);
                                      # a scene: scene (everything) or camera (the camera alone)
    profile: str = 'prores4444'       # video profile key
    layers: tuple = ('emission', 'glow', 'heat', 'depth', 'surface') + PASSES   # (EXR; a composite's: only PASSES)
    compression: str = 'zip'
    half: bool = True
    bits: int = 16                    # PNG
    alpha_mode: str = 'premultiplied'  # element PNG / video: premultiplied or straight
    audio: bool = True
    quality: int | None = None
    deep_samples: int = DEEP_SAMPLES  # deep: samples per pixel (up to DEEP_MAX)

    def label(self):
        if self.kind == 'scene':
            return 'Camera · USD' if self.content == 'camera' else 'Scene · USD (camera, objects, particles)'
        if self.kind == 'camera':
            return 'Camera · Nuke .chan'
        if self.kind == 'mesh':
            what = 'Fabric' if self.content == 'fabric' else 'Liquid surface'
            return what + ' · ' + ('USD' if Path(self.path).suffix.lower().startswith('.usd') else 'OBJ sequence')
        if self.kind == 'deep':
            return f'Deep EXR sequence · {self.deep_samples} samples · element'
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


USD_EXTS = ('.usd', '.usdc', '.usda')


def infer_output(path, content=None, profile=None):
    """The output a path asks for, by its extension (and `content`, or a name with .deep., .scene. or .camera. in it)."""
    ext = Path(path).suffix.lower()
    name = Path(path).name.lower()
    if content in ('scene', 'camera', 'deep') and ext not in ('.exr',) + USD_EXTS:
        content = None          # (meant for another output on the same command line)
    if ext == '.exr' and (content == 'deep' or '.deep.' in name):
        return Output('deep', path, 'element')
    if ext == '.exr':
        return Output('exr', path, content if content in CONTENT else 'element')
    if ext in ('.png',):
        return Output('png', path, content or 'element')
    if ext == '.vdb':
        return Output('vdb', path)
    if ext == '.chan':
        return Output('camera', path, 'camera')
    if ext in USD_EXTS and (content in ('scene', 'camera') or '.scene.' in name or '.camera.' in name):
        return Output('scene', path, 'camera' if (content == 'camera' or ('.camera.' in name and content != 'scene')) else 'scene')
    if ext in ('.obj',) + USD_EXTS:
        return Output('mesh', path)
    if ext in ('.mov', '.mp4', '.webm', '.mxf'):
        c = content or ('element' if ext in ('.mov', '.webm') else 'composite')
        if profile is None:
            profile = {'.mp4': 'h264', '.webm': 'vp9_alpha'}.get(ext, 'prores4444' if c == 'element' else 'prores422hq')
        return Output('video', path, c, profile)
    raise ValueError(f'Cannot tell the output type from "{path}". Use .exr, .png, .vdb, .obj, .usd, .chan, .mov, .mp4 or .webm.')


def _order(scene):
    """The shot's layers back to front, as a render takes them, and whether there are any but the base."""
    order = scene.layer_order() or [('base', scene)]
    return order, any(u != 'base' for u, _ in order)


def deep_layers(scene):
    """The shot's layers [(uid, scene)] a deep EXR holds: its fire and liquid ones."""
    return [(u, s) for u, s in _order(scene)[0] if s.kind in DEEP_KINDS]


def has_fabric(scene):
    """True if the scene simulates fabric, so a mesh output can carry it."""
    return scene.kind in FABRIC_KINDS and bool(scene.fabric_specs())


def liquid_layers(scene):
    """The shot's layers [(uid, scene)] with a liquid (whose surface a mesh output holds)."""
    return [(u, s) for u, s in _order(scene)[0] if s.kind in ('liquid', 'both')]


def fabric_layers(scene):
    """The shot's layers [(uid, scene)] with fabric."""
    return [(u, s) for u, s in _order(scene)[0] if has_fabric(s)]


def layer_tags(order):
    """{uid: what goes into a file's name for that layer}: '' for the base layer (its files are named as asked), and
    '.<its name>' for each other (name.<layer>.####.vdb), every one different, and never the tag of a file the base
    layer writes beside its own (name.liquid.####.vdb beside a fire-and-liquid VDB, name.fabric.* beside a liquid mesh)."""
    out, seen = {}, {'liquid', 'fabric', 'deep'}
    for uid, s in order:
        if uid == 'base':
            out[uid] = ''
            continue
        t = re.sub(r'[^A-Za-z0-9_-]+', '_', s.name or str(uid)).strip('_').lower() or str(uid)
        while t in seen:
            t += '_' + re.sub(r'[^A-Za-z0-9_-]+', '_', str(uid))
        seen.add(t)
        out[uid] = '.' + t
    return out


def layer_roots(order):
    """{uid: where a layer's prims go in a USD scene}: /World for the base layer, /World/<its name> for each other."""
    from ..io.scene_usd import prim_name
    taken = {'Camera', 'Objects', 'Pieces', 'Ropes', 'Matter', 'Grass', 'Embers'}
    return {uid: '/World' if uid == 'base' else '/World/' + prim_name(s.name or uid, taken, 'Layer') for uid, s in order}


def tagged(pattern, tag):
    """A path pattern with `tag` put in before its frame number: name.####.vdb -> name.<tag>.####.vdb, and
    name.usdc -> name.<tag>.usdc."""
    if not tag:
        return pattern
    p = Path(pattern)
    m = re.search(r'[._]?(#+|%0?\d*d)', p.name)
    if m is not None:
        return str(p.with_name(p.name[:m.start()] + tag + p.name[m.start():]))
    return str(p.with_name(p.stem + tag + p.suffix))


def same_camera(a, b, frames, size):
    """Whether two layers of a shot see it through the same camera (their placements can differ: each effect's own)."""
    from ..io.camera_out import camera_at
    return all(np.allclose(camera_at(a, f, size).view_proj, camera_at(b, f, size).view_proj, rtol=1e-6, atol=1e-9)
               for f in frames)


def _shot_frames(scene):
    return sorted({scene.start, (scene.start + scene.end) // 2, scene.end})


def own_cameras(scene, size=None):
    """The layers [(uid, scene)] whose camera is not the shot's (the base layer's)."""
    order, layered = _order(scene)
    if not layered:
        return []
    size = size or scene.output_size()
    return [(u, s) for u, s in order if u != 'base' and not same_camera(s, scene, _shot_frames(scene), size)]


def scene_contents(scene):
    """What a USD scene of this shot holds, in words (for the Render window)."""
    from ..engine.solids import breaks, joined, kind_of
    order = _order(scene)[0]
    words = ['camera']
    cols = [c for _, s in order for c in s.colliders if c['enabled']]
    if cols:
        words.append('objects')
    if any(breaks(c) or kind_of(c) for c in cols):
        words.append('pieces')
    if any(joined(c) in ('rope', 'spring') for c in cols):
        words.append('ropes')
    if any(s.kind != 'cloud' and s.matter_specs() for _, s in order):
        words.append('matter')
    if any(s.kind != 'cloud' and s.strand_specs() for _, s in order):
        words.append('grass')
    if any(s.kind in ('fire', 'both') and s.data['embers'].get('enabled') for _, s in order):
        words.append('embers')
    return words


def check_outputs(scene, outputs):
    """Why these outputs cannot be written for this scene (or shot): a message for each, none when they can."""
    out = []
    kind = KIND_NAMES.get(scene.kind, 'this kind of scene')
    layered = _order(scene)[1]
    for o in outputs:
        if o.kind == 'deep':
            if not deep_layers(scene):
                if layered:
                    out.append(f'{o.path}: deep EXRs are made of fire and liquid layers, and this shot has none.')
                else:
                    out.append(f'{o.path}: deep EXRs are made for fire scenes and liquid scenes, and this is {kind}.')
            elif not 1 <= int(o.deep_samples) <= DEEP_MAX:
                out.append(f'{o.path}: a deep EXR holds 1 to {DEEP_MAX} samples per pixel, not {o.deep_samples}.')
        elif o.kind == 'mesh':
            liquid, fabric = bool(liquid_layers(scene)), bool(fabric_layers(scene))
            what = 'this shot' if layered else 'this scene'
            if o.content == 'fabric' and not fabric:
                out.append(f'{o.path}: there is no fabric in {what} to write as a mesh.')
            elif o.content == 'liquid' and not liquid:
                out.append(f'{o.path}: there is no liquid to write as a mesh: ' +
                           ('no layer of the shot has one.' if layered else f'this is {kind}.'))
            elif not (liquid or fabric):
                out.append(f'{o.path}: a mesh holds a liquid\'s surface or fabric, and {what} has neither (for the '
                           'camera, the objects and the particles, write a USD scene: name.scene.usdc).')
        elif o.kind == 'scene' and Path(o.path).suffix.lower() not in USD_EXTS:
            out.append(f'{o.path}: a USD scene is a .usd, .usdc or .usda file.')
    return out


def output_notes(scene, outputs):
    """What these outputs leave out of this shot, or where they put each layer, to say before it renders."""
    notes = []
    order, layered = _order(scene)
    left = [s.name or 'a layer' for u, s in order if s.kind not in DEEP_KINDS]
    tags = layer_tags(order)
    others = [(u, s) for u, s in order if u != 'base']
    own = None
    for o in outputs:
        if o.kind == 'deep' and left and layered and deep_layers(scene):
            notes.append(f'{o.path} holds the fire and liquid layers; {", ".join(left)} '
                         f'{"is" if len(left) == 1 else "are"} not in it.')
        if o.kind in ('vdb', 'mesh') and layered:
            if o.kind == 'mesh':
                has = {u for u, _ in (liquid_layers(scene) if o.content != 'fabric' else [])}
                has |= {u for u, _ in (fabric_layers(scene) if o.content != 'liquid' else [])}
            else:
                has = {u for u, _ in order}
            beside = [f'{s.name or u}: {Path(tagged(o.path, tags[u])).name}' for u, s in others if u in has]
            if beside and 'base' in has:
                notes.append(f'{o.path} holds the base layer\'s; the other layers\' go beside it ({"; ".join(beside)}).')
            elif beside:
                notes.append(f'{o.path}: the base layer has nothing for it, so only the layers that have go beside it '
                             f'({"; ".join(beside)}).')
        if o.kind in ('scene', 'camera'):
            if own is None:
                own = own_cameras(scene)
            if own:
                names = ', '.join(s.name or u for u, s in own)
                notes.append(f'{o.path}: {names} {"has a camera" if len(own) == 1 else "have cameras"} of '
                             f'{"its" if len(own) == 1 else "their"} own (placed by {"its" if len(own) == 1 else "their"} '
                             f'Anchor): ' + ('written under /World/<layer>/Camera.' if o.kind == 'scene' else
                                             'the .chan holds the shot\'s (the base layer\'s); a USD scene holds them all.'))
        if o.kind == 'camera':
            from ..io.camera_out import slide_note, slide_px
            px = slide_px(scene, _shot_frames(scene))
            if px > 0.5:
                notes.append(slide_note(o.path, px))     # (a larger slide between these frames is said after the render)
        if o.kind in ('scene', 'camera'):
            from ..io.camera_out import lens_k1
            k1 = lens_k1(scene)
            if k1 != 0.0:
                notes.append(f'{o.path}: the element is bent by the footage\'s Lens distortion ({k1:+.3f}), which a camera '
                             'cannot hold: CG rendered through it lines up at the centre of the frame but less and less '
                             'toward the edges. Distort it the same way in the comp: x_d = x_u (1 + k1 r²), r from the '
                             'centre in half-diagonals' + (' (blackbody:lens_k1 on the USD camera).' if o.kind == 'scene'
                                                           else '.'))
        if o.kind == 'scene' and o.content != 'camera':
            notes += _scene_notes(o, order)
        if o.kind == 'exr' and layered and wants_passes(scene, o):
            notes.append(f'{o.path}: its motion vectors, normals, mattes and Cryptomatte are the base layer\'s (its set '
                         'and its fire); the other layers are not in them.')
        if o.kind == 'exr' and exr_compression(scene, o) != o.compression:
            c = o.compression.upper()
            notes.append(f'{o.path} is written with ZIP, not {c}: {c} is lossy, and would garble its Cryptomatte and '
                         f'positions (leave the compositing passes out to keep {c}).')
    return notes


def _scene_notes(o, order):
    """What a USD scene leaves out of these layers."""
    from ..engine.ballistics import Ballistics
    out = []
    hollow = [c.get('name') or 'an object' for _, s in order for i, c in enumerate(s.colliders) if c['enabled'] and (
        float(s.get(('collider', i, 'hollow'), s.start)) > 0.0
        or all(float(x) > 0.0 for x in s.get(('collider', i, 'opening'), s.start)))]
    if hollow:
        out.append(f'{o.path}: {", ".join(dict.fromkeys(hollow))} {"is" if len(hollow) == 1 else "are"} hollow or cut '
                   'open: the USD scene holds the outer shape.')
    if any(Ballistics.wanted(s) for _, s in order):
        out.append(f'{o.path}: what bullets throw up (debris, sparks, tracers) and the holes they leave are not in the '
                   'USD scene.')
    if any(s.kind in ('liquid', 'both') or has_fabric(s) for _, s in order):
        out.append(f'{o.path}: the liquid and the fabric are not in the USD scene: write them with their own outputs '
                   '(Liquid surface · USD, Fabric · USD).')
    return out


def wants_passes(scene, o):
    """Whether output o writes compositing passes (render/passes.py) for this scene."""
    return o.kind == 'exr' and scene.kind in PASS_KINDS and bool(set(o.layers) & set(PASSES))


def exr_compression(scene, o):
    """The compression EXR output o is written with: its own, or ZIP in place of a lossy one when it carries
    Cryptomatte or positions, which must come back exactly (DWA's lossy maths works on every channel named R, G, B or
    Y, 32-bit float ones too: a Cryptomatte id changed in its last bit names nothing)."""
    if o.compression in LOSSY_EXR and wants_passes(scene, o) and {'crypto', 'normals'} & set(o.layers):
        return 'zip'
    return o.compression


class RenderJob:
    def __init__(self, scene, outputs, engine, frames=None, final=True, footage=None, size=None, samples=None,
                 motion_blur=None, from_cache=False, engines=None):
        self.scene = scene
        self.engines = engines   # render/layers.LayerEngines for a shot with layers (made here if not given)
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
        self._no_cached_vel = False
        self._cam_attrs = None         # the frame's camera for the EXRs' headers (io/camera_out.py exr_attrs)
        self.notes = []                # what the writers could not write as it is (a USD scene's), said after the run
        self.names = None    # what the compositing passes name (render/passes.SetNames), when an EXR has them

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
        problems = check_outputs(sc, self.outputs)
        if problems:
            raise RuntimeError(' '.join(problems))
        W, H = self.size
        deep_out = [o for o in self.outputs if o.kind == 'deep']
        deep = max(int(o.deep_samples) for o in deep_out) if deep_out else 0
        if deep and any(s.kind == 'fire' for _, s in deep_layers(sc)):
            lim = [v for k, v in (getattr(eng.gpu, 'limits', None) or {}).items()
                   if k in ('max-storage-buffer-binding-size', 'max-buffer-size')]
            need = W * H * deep * 32   # (the march's deep bins: two vec4s each)
            if lim and need > min(lim):
                raise RuntimeError(f'{deep} deep samples per pixel at {W}x{H} need {need / 2 ** 30:.1f} GB in one GPU '
                                   f'buffer, more than this GPU allows ({min(lim) / 2 ** 30:.1f} GB): use fewer '
                                   'samples or a smaller size.')
        eng.cache_readonly = bool(self.from_cache)
        sc.from_cache = bool(self.from_cache)   # (the layout the cache was simulated at is read as it is: Scene.memory_plan)
        # (the camera alone needs nothing simulated)
        camera_only = all(o.kind == 'camera' or (o.kind == 'scene' and o.content == 'camera') for o in self.outputs)
        if not camera_only:
            eng.prepare(sc, final=self.final)
        writers = {}
        need_elem = any(o.content == 'element' and o.kind not in DATA_KINDS for o in self.outputs)
        from ..io.holdout import FootageHoldout
        self.holdout = FootageHoldout(sc) if sc.kind != 'liquid' else None
        hold = self.holdout if (self.holdout is not None and self.holdout.active) else None
        need_comp = any(o.content == 'composite' and o.kind not in DATA_KINDS for o in self.outputs)
        need_vdb = [o for o in self.outputs if o.kind == 'vdb']
        mesh_out = [o for o in self.outputs if o.kind == 'mesh']
        scene_out = [o for o in self.outputs if o.kind == 'scene']
        chan_out = [o for o in self.outputs if o.kind == 'camera']
        audio_src = sc.footage['path'] if (sc.footage and self.footage is not None and self.footage.audio) else None
        total = self.last - self.first + 1
        from .layers import LayerEngines, merge_elements, render as render_layers, simulate as simulate_layers
        order = sc.layer_order() or [('base', sc)]
        layered = [x for x in order if x[0] != 'base']   # the shot's other layers, each its own simulation
        LE = self.engines if (self.engines is not None and self.engines.base is eng) else LayerEngines(eng)
        # (VDB, mesh and scene outputs hold every layer: each other layer's files beside the base layer's, its prims
        # under /World/<layer>; a layer placed by an Anchor of its own has its own camera there)
        tags, roots = layer_tags(order), layer_roots(order)
        own_cam = {u for u, _ in own_cameras(sc, (W, H))} if (scene_out and layered) else set()
        cam_attrs = any(o.kind in ('exr', 'deep') for o in self.outputs)
        # the compositing passes: the set's, traced per frame; the fire's vectors from its march (Renderer.motion)
        pass_out = [o for o in self.outputs if wants_passes(sc, o)]
        elem_passes = any(o.content == 'element' for o in pass_out)
        comp_passes = any(o.content == 'composite' for o in pass_out)
        fire_vectors = sc.kind in ('fire', 'both') and any('motion' in o.layers for o in pass_out)
        if getattr(eng, '_stage', None) is not None:
            eng.stage.notes = []   # (what an earlier render's passes said)
        if pass_out:
            self.names = P.set_names(sc, footage=self.footage is not None)
        try:
            for i, frame in enumerate(range(self.first, self.last + 1)):
                if (cancelled and cancelled()) or self.cancelled:
                    self.cancelled = True
                    break
                if fire_vectors:
                    eng.renderer.motion = P.march_motion(sc, frame, (W, H))
                epass = None

                def sim_progress(frac, f, _i=i):
                    if progress:
                        progress((_i + 0.5 * frac) / total, f'Simulating frame {f}')

                if camera_only:
                    pass
                elif self.from_cache and frame in eng.cache:
                    pass   # simulated already (another machine, or an earlier run): render it from the disk cache
                elif self.from_cache:
                    raise RuntimeError(f'Frame {frame} is not in the disk cache; simulate it first (blackbody simulate).')
                else:
                    eng.simulate_to(sc, frame, progress=sim_progress if i == 0 else None, cancelled=cancelled,
                                    cache=eng.cache.disk is not None)
                if layered and not camera_only and not simulate_layers(LE, layered, frame, final=self.final,
                                                                       cancelled=cancelled):
                    self.cancelled = True
                    break
                if cam_attrs:
                    # the camera in the EXRs' headers (the shot's: the base layer's)
                    from ..io.camera_out import camera_at, exr_attrs
                    self._cam_attrs = exr_attrs(camera_at(sc, frame, (W, H)))
                holdout = hold.read(frame) if hold is not None else None
                if need_elem and layered:
                    # each layer's element, merged back to front; the extra passes are the base layer's, and the
                    # deep samples of every fire and liquid layer go into one deep image, each at its own depth
                    beauties, aov, dparts = [], None, []
                    for uid, lay in order:
                        e = LE.get(uid)
                        e.render(lay, frame, (W, H), mode='fire', final=self.final, samples=self.samples,
                                 motion_blur=self.motion_blur, plate=self._plate(frame) if lay.kind in ('liquid', 'both') else None,
                                 plate_fit=self._plate_fit(), holdout=holdout if lay.kind != 'liquid' else None,
                                 deep=deep if lay.kind == 'fire' else 0)
                        a = e.aovs()
                        if elem_passes and uid == 'base' and lay.kind == 'liquid':
                            a['mattes'] = e.gpu.read(e.renderer.mask)   # (where its liquid is, for the passes)
                        if deep and lay.kind in DEEP_KINDS:
                            dparts.append(self._deep_samples(e, lay, frame, a))
                        beauties.append(a['beauty'])
                        if uid == 'base' or aov is None:
                            aov = a
                    if elem_passes:
                        # (the base layer's set and fire: only its march gathers the fire's vectors)
                        epass = self._passes(eng, sc, frame, aov, 0.0)
                    aov = dict(aov)
                    aov['beauty'] = merge_elements(beauties).astype(np.float16)
                    aov['passes'] = epass
                    dparts = [d for d in dparts if d.shape[:2] == dparts[0].shape[:2]]
                    for o in self.outputs:
                        if o.kind == 'deep':
                            self._write_deep(o, frame, np.concatenate(dparts, axis=2))
                        elif o.content == 'element' and o.kind not in DATA_KINDS:
                            self._write_element(o, frame, aov, aov['beauty'], None, writers, audio_src)
                elif need_elem:
                    liquid = sc.kind == 'liquid'
                    # a liquid refracts the footage, so its element is rendered against it
                    eng.render(sc, frame, (W, H), mode='fire', final=self.final, samples=self.samples,
                               motion_blur=self.motion_blur, plate=self._plate(frame) if sc.kind in ('liquid', 'both') else None,
                               plate_fit=self._plate_fit(), holdout=holdout, deep=deep if sc.kind == 'fire' else 0)
                    aov = eng.aovs()
                    if liquid:
                        aov['mattes'] = eng.gpu.read(eng.renderer.mask)
                    if elem_passes:
                        aov['passes'] = epass = self._passes(eng, sc, frame, aov, 0.0)
                    elem_lin = eng.linear_comp()
                    glow = self.engine.gpu.read(eng.renderer.bloom_tex) if any('glow' in o.layers for o in self.outputs if o.kind == 'exr') else None
                    dsamples = self._deep_samples(eng, sc, frame, aov) if deep else None
                    for o in self.outputs:
                        if o.kind == 'deep':
                            self._write_deep(o, frame, dsamples)
                        elif o.content == 'element' and o.kind not in DATA_KINDS:
                            self._write_element(o, frame, aov, elem_lin, glow, writers, audio_src)
                if need_comp:
                    front = render_layers(LE, order, frame, (W, H), mode='composite', final=self.final, samples=self.samples,
                                          motion_blur=self.motion_blur, plate=self._plate(frame), plate_fit=self._plate_fit(),
                                          holdout=holdout)
                    comp_lin = front.linear_comp()
                    lpass = _light_passes(front, self.scene, comp_lin.shape[:2])
                    cpass = None
                    if comp_passes:
                        # through the footage's lens, as the composite shows it (the element's are a pinhole's)
                        k1 = float(sc.data['composite'].get('lens_k1', 0.0))
                        if epass is not None and k1 == 0.0:
                            cpass = epass
                        else:
                            a = eng.aovs()
                            if sc.kind == 'liquid':
                                a['mattes'] = eng.gpu.read(eng.renderer.mask)
                            cpass = self._passes(eng, sc, frame, a, k1)
                    for o in self.outputs:
                        if o.content == 'composite' and o.kind not in DATA_KINDS:
                            self._write_comp(o, frame, comp_lin, writers, audio_src, lpass, cpass)
                for uid, lay in order:
                    e = LE.get(uid)
                    for o in need_vdb:
                        self._write_vdb(o, frame, e, lay, tags[uid])
                    if mesh_out:
                        self._write_meshes(mesh_out, frame, e, lay, uid, tags[uid], writers)
                if scene_out:
                    from ..io.camera_out import camera_at
                    from ..io.scene_usd import SceneWriter, frame_data
                    data = {}
                    for o in scene_out:
                        w = writers.get(id(o))
                        if w is None:
                            w = writers[id(o)] = SceneWriter(o.path, sc.fps)
                        for uid, lay in order:
                            cf = camera_at(lay, frame, (W, H)) if (uid == 'base' or uid in own_cam) else None
                            if o.content == 'camera':
                                if cf is not None:
                                    w.add(frame, None, roots[uid], camera=cf)
                                continue
                            if uid not in data:   # (once a frame, whatever the number of scene outputs)
                                data[uid] = frame_data(LE.get(uid), lay, frame)
                            w.add(frame, data[uid], roots[uid], camera=cf)
                if chan_out:
                    from ..io.camera_out import ChanWriter, camera_at, slide_px
                    cf = camera_at(sc, frame, (W, H))
                    for o in chan_out:
                        w = writers.get(id(o))
                        if w is None:
                            # (output_notes said the slide of a few frames before the render: a larger one over every
                            # frame is said again at the end)
                            w = writers[id(o)] = ChanWriter(o.path, said=slide_px(sc, _shot_frames(sc), (W, H)))
                        w.add(frame, cf)
                if progress:
                    progress((i + 1) / total, f'Frame {frame} of {self.last}')
        finally:
            if fire_vectors:   # (a scene or camera output's job may have no renderer at all)
                eng.renderer.motion = None
            if eng.cache.disk is not None:
                eng.cache.disk.flush()
            if self.holdout is not None:
                self.holdout.close()
            for w in writers.values():
                try:
                    self.written.append(str(w.close()))
                    self.written += [str(p) for p in getattr(w, 'extra', [])]
                    for n in getattr(w, 'notes', []):
                        if n not in self.notes:
                            self.notes.append(n)
                except Exception as ex:
                    log.error('Closing %s failed: %s', w.path, ex)
        self._sidecar(time.perf_counter() - t_start)
        return self.written

    # -- writers ---------------------------------------------------------------------------------

    def _write_vdb(self, o, frame, eng, sc, tag):
        """A frame of one layer (`sc`, simulated by `eng`) as a VDB: the base layer's at the path given, another's
        beside it (tag: '.<layer>'); a fire-and-liquid one's liquid beside its fire (name.liquid.####.vdb)."""
        from ..io.vdb import write_liquid_vdb_frame, write_vdb_frame
        p = frame_path(tagged(o.path, tag), frame)
        if sc.kind == 'liquid':
            write_liquid_vdb_frame(p, eng, sc, frame)
        elif sc.kind == 'cloud':
            write_cloud_vdb_frame(p, eng, sc, frame)
        else:
            write_vdb_frame(p, self._fire_fields(frame, eng), sc, frame)
        self.written.append(p)
        if sc.kind == 'both':
            pl = frame_path(tagged(o.path, tag + '.liquid'), frame)
            write_liquid_vdb_frame(pl, eng, sc, frame)
            self.written.append(pl)

    def _write_meshes(self, mesh_out, frame, eng, sc, uid, tag, writers):
        """A frame of one layer's liquid surface and fabric for the mesh outputs: a mesh output holds the liquid's
        surface, the fabric, or (content 'element') both, the fabric beside it (name.fabric.####.obj or name.fabric.usd);
        another layer's beside the base layer's (tag: '.<layer>')."""
        liquid = [o for o in mesh_out if o.content != 'fabric'] if sc.kind in ('liquid', 'both') else []
        fabric = [o for o in mesh_out if o.content != 'liquid'] if has_fabric(sc) else []
        if liquid:
            from ..io.liquid_mesh import UsdWriter, liquid_surface_mesh, write_obj
            mesh = liquid_surface_mesh(eng, sc, frame)
            for o in liquid:
                path = tagged(o.path, tag)
                if Path(path).suffix.lower().startswith('.usd'):
                    w = writers.get((id(o), uid, 'liquid'))
                    if w is None:
                        w = writers[(id(o), uid, 'liquid')] = UsdWriter(path, sc.fps, sc.data['water']['droplet_size'])
                    w.add(frame, mesh)
                else:
                    p = frame_path(path, frame)
                    write_obj(p, mesh)
                    self.written.append(p)
        if fabric:
            from ..io import fabric_mesh as FM
            meshes = FM.fabric_meshes(eng, sc, frame)
            for o in fabric:
                path = tagged(o.path, tag + ('.fabric' if any(o is m for m in liquid) else ''))
                if Path(path).suffix.lower().startswith('.usd'):
                    w = writers.get((id(o), uid, 'fabric'))
                    if w is None:
                        w = writers[(id(o), uid, 'fabric')] = FM.UsdWriter(path, sc.fps)
                    w.add(frame, meshes)
                else:
                    p = frame_path(path, frame)
                    FM.write_obj(p, meshes)
                    self.written.append(p)

    def _passes(self, eng, sc, frame, aov, lens):
        """The compositing passes of the frame `eng` has just rendered (render/passes.frame_passes), through a lens of
        distortion `lens`; what they cut short goes to the engine's notices."""
        fire_vec = eng.renderer.read_vectors() if eng.renderer.motion is not None else None
        out = P.frame_passes(eng, sc, frame, self.size, self.names, aov, lens, footage=self.footage is not None,
                             motion_blur=self.motion_blur, fire_vec=fire_vec, plate_fit=self._plate_fit())
        notes = list(out['notes'])
        if fire_vec is not None and eng.sim_frame != frame:
            entry = eng.cache.get(frame)
            if entry is not None and 'vel' not in entry:
                notes.append('Motion vectors: the cache holds no velocities (it was simulated with motion blur off), so '
                             'the fire\'s vectors are only the camera\'s motion.')
        for n in notes:
            if n not in eng.stage.notes:
                eng.stage.notes.append(n)
        return out

    def _fire_fields(self, frame, eng=None):
        """What a fire VDB is written from: the solver when it holds `frame`, else the frame from the cache (a
        render from the disk cache never steps the solver, which holds only its start)."""
        eng = eng or self.engine
        if eng.sim_frame == frame:
            return eng.solver
        entry = eng.cache.get(frame)
        if entry is None or 'scal' not in entry:
            raise RuntimeError(f'Frame {frame} is neither simulated nor cached.')
        if 'vel' not in entry and not self._no_cached_vel:
            self._no_cached_vel = True
            log.warning('The cache holds no velocities (it was simulated with motion blur off), so the VDBs have an '
                        'empty vel grid.')
        return CachedFire(entry, eng.solver)

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
            attrs = {'software': f'{blackbody.APP_NAME} {blackbody.__version__}', **(self._cam_attrs or {})}
            keep32 = set()
            if aov.get('passes') is not None:
                pc, pa = P.channels(aov['passes'], self.names, o.layers, keep32)
                ch.update(pc)
                attrs.update(pa)
            space = self._exr_space()
            if space:
                ch = self._to_space(ch, space)
                attrs['colorspace'] = space
            ch = {k: v.astype(np.float32 if k in keep32 else dt) for k, v in ch.items()}
            p = frame_path(o.path, frame)
            write_exr(p, ch, exr_compression(self.scene, o), attrs)
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
        for pre in ('', 'emission.', 'glow.', 'light.', 'light_key.', 'light_sky.', 'light_fire.', 'light_lamps.'):
            keys = [pre + c for c in 'RGB']
            if all(k in ch for k in keys):
                rgb = pipe.to_space(np.stack([ch[k].astype(np.float32) for k in keys], -1), space)
                for i, k in enumerate(keys):
                    out[k] = rgb[..., i]
        return out

    def _deep_samples(self, eng, sc, frame, aov):
        """(h, w, k, 8) deep samples of the element `eng` has just rendered for scene (or layer) `sc`."""
        if sc.kind == 'liquid':
            if 'mattes' not in aov:
                aov = dict(aov, mattes=eng.gpu.read(eng.renderer.mask))
            return self._liquid_deep(sc, frame, aov)
        return eng.renderer.read_deep()

    def _write_deep(self, o, frame, samples):
        from ..io.images import write_deep_exr
        space = self._exr_space()
        if space:
            pipe = self._pipe()
            if pipe is not None:
                samples = samples.copy()
                samples[..., :3] = pipe.to_space(samples[..., :3], space)
        p = frame_path(o.path, frame)
        write_deep_exr(p, samples, {'software': f'{blackbody.APP_NAME} {blackbody.__version__}', **(self._cam_attrs or {})})
        self.written.append(p)

    def _liquid_deep(self, sc, frame, aov):
        """A liquid element as deep samples: the water surface at its depth (opaque where there is
        water), and the spray and droplets in front of it as a partly transparent sample."""
        from ..engine import camera as cam
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
        return np.take_along_axis(samples, order[..., None], axis=2)

    def _write_comp(self, o, frame, comp_lin, writers, audio_src, lpass=None, cpass=None):
        view = self.scene.data['composite']
        pipe = self._pipe() if view['view'] == 'ocio' else None
        if pipe is not None:
            rgb = pipe.to_display(comp_lin[..., :3].astype(np.float32))
        else:
            rgb = view_transform(comp_lin[..., :3].astype(np.float32), view['view'], view['knee'], view['highlight_white'])
        rgba = np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,), np.float32)], -1)
        if o.kind == 'exr':
            ch = {'R': comp_lin[..., 0], 'G': comp_lin[..., 1], 'B': comp_lin[..., 2]}
            ch.update(lpass or {})          # (Lume's per-light passes of the set)
            attrs = dict(self._cam_attrs or {})
            keep32 = set()
            if cpass is not None:
                pc, pa = P.channels(cpass, self.names, o.layers, keep32)
                ch.update(pc)
                attrs.update(pa)
            space = self._exr_space()
            if space:
                ch = self._to_space({k: v.astype(np.float32) for k, v in ch.items()}, space)
                attrs['colorspace'] = space
            dt = np.float16 if o.half else np.float32
            p = frame_path(o.path, frame)
            write_exr(p, {k: v.astype(np.float32 if k in keep32 else dt) for k, v in ch.items()},
                      exr_compression(self.scene, o), attrs or None)
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


def _centres(v):
    """Cell-centred velocity (nz, ny, nx, 3) from face velocities (nz+1, ny+1, nx+1, 4), as the solvers keep them."""
    v = np.asarray(v, np.float32)
    return np.stack([0.5 * (v[:-1, :-1, :-1, 0] + v[:-1, :-1, 1:, 0]), 0.5 * (v[:-1, :-1, :-1, 1] + v[:-1, 1:, :-1, 1]),
                     0.5 * (v[:-1, :-1, :-1, 2] + v[1:, :-1, :-1, 2])], -1)


class CachedFire:
    """A cached fire frame, read as io/vdb.write_vdb_frame reads the solver: its fields as they were simulated,
    at the frame's own layout (a growing box may have been smaller then). Its velocity is the simulated air's,
    without the finer grid's swirls (the cache does not keep them), and none if the cache has none."""

    def __init__(self, entry, solver):
        self.entry = entry
        self.upres = int(entry.get('upres', 1))
        self.dims = tuple(int(x) for x in entry.get('dims', solver.dims))
        self.h = float(entry.get('h', solver.h))
        self.origin = tuple(float(x) for x in entry.get('origin', solver.origin))
        self.features = dict(getattr(solver, 'features', None) or {})

    def read_scalars(self):
        return self.entry['scal']   # (at the finer grid's size with upres)

    read_scalars_fine = read_scalars

    def read_velocity_centres(self):
        if 'vel' not in self.entry:
            nx, ny, nz = self.dims
            return np.zeros((nz, ny, nx, 3), np.float32)
        return _centres(self.entry['vel'])

    def read_aux(self):
        return self.entry.get('aux')

    def read_chem(self):
        return self.entry.get('chem')

    def read_stain(self):
        return self.entry.get('stain')


CLOUD_GRIDS = (('cloud_water', 0, 2), ('cloud_ice', 1, 0), ('rain', 0, 3), ('snow', 1, 1), ('hail', 1, 2))


def cloud_vdb_grids(a, b, rho, vel=None, speed=1.0, rot=None):
    """A sky frame's VDB grids (x-major) from the cloud solver's fields, (nz, ny, nx, 4) arrays (engine/cloud.py):
    a = (potential temperature excess, vapour, cloud water, rain), b = (cloud ice, snow, graupel and hail, _), in
    kg/kg; rho is the air's density on each level (kg/m^3). The water comes out in g/m^3: 'density' is the cloud
    itself (its water and ice), then each kind on its own. vel: the face velocities (m/s of sky) or None; it is
    written at the cell centres where there is water, times `speed` and turned by `rot` into the scene's frame."""
    to_x = lambda x: np.ascontiguousarray(np.transpose(x, (2, 1, 0) + tuple(range(3, x.ndim))))
    g = 1000.0 * np.asarray(rho, np.float32)[None, :, None]
    src = (np.asarray(a, np.float32), np.asarray(b, np.float32))
    parts = {name: np.maximum(src[i][..., c], 0.0) * g for name, i, c in CLOUD_GRIDS}
    grids = {'density': to_x(parts['cloud_water'] + parts['cloud_ice'])}
    for name, x in parts.items():
        if x.max() > 1e-4:
            grids[name] = to_x(x)
    if vel is not None:
        c = _centres(vel) * float(speed)
        if rot is not None:
            c = c @ np.asarray(rot, np.float32).T
        wet = sum(parts.values()) > 1e-4
        grids['vel'] = to_x(np.where(wet[..., None], c, 0.0).astype(np.float32))
    return grids


def write_cloud_vdb_frame(path, engine, scene, frame):
    """Write a sky frame as a VDB (cloud_vdb_grids), placed and sized as the scene shows the sky: a voxel is a cell
    of the scene's box (Atmosphere › Scale metres of sky to its metre), and vel moves the cloud as it moves in the
    shot, in the scene's metres per second. From the disk cache there is no vel (the cache keeps only the water)."""
    from ..engine import camera as cam
    from ..engine.cloud import base_state
    from ..io.vdb import write_vdb
    C = engine.cloud
    if engine.sim_frame == frame:
        a, b = C.read_fields()
        vel = engine.gpu.read(C.V[0])
    else:
        entry = engine.cache.get(frame)
        if entry is None or 'cloud_a' not in entry:
            raise RuntimeError(f'Frame {frame} is neither simulated nor cached.')
        ca = entry['cloud_a'].astype(np.float32)
        a = np.zeros(ca.shape[:3] + (4,), np.float32)
        a[..., 2:4] = ca
        b = entry['cloud_b'].astype(np.float32)
        vel = None
    rho = base_state(scene.cloud_params(frame), C.dims[1], C.h)[1]['rho']
    spec, fire = scene.camera(frame)
    yaw = math.radians(fire.yaw)
    rot = cam.rot_y(yaw)
    # m/s of sky to the scene's metres per second of the shot
    speed = float(scene.v('atmosphere', 'time_lapse', frame)) * float(scene.v('domain', 'time_scale', frame)) / C.scale
    grids = cloud_vdb_grids(a, b, rho, vel, speed, rot)
    h = float(C.h_scene)
    translation = tuple(rot @ (np.asarray(C.origin_scene, float) + 0.5 * h) + np.asarray(fire.position, float))
    meta = [('blackbody_sky_metres_per_unit', 'float', float(C.scale)), ('blackbody_water_units', 'string', 'g/m^3')]
    write_vdb(path, grids, h, translation, yaw, file_meta=meta)


def _light_passes(eng, scene, shape):
    """Lume's per-light passes of the set (Lume › Light passes) as EXR layers at shape, or {}."""
    from ..engine import lume as LU
    if not LU.settings(scene).light_passes or not LU.settings(scene).on:
        return {}
    st = getattr(eng, '_stage', None)
    ch = {}
    for name, img in (st.light_passes() if st is not None else {}).items():
        img = _resize_to(img, shape)
        ch.update({f'{name}.R': img[..., 0], f'{name}.G': img[..., 1], f'{name}.B': img[..., 2]})
    return ch


def _resize_to(img, shape):
    """Nearest-neighbour + box resample of a small image (the bloom buffer) up to `shape`."""
    h, w = shape
    ih, iw = img.shape[:2]
    if (ih, iw) == (h, w):
        return img
    ys = np.clip(((np.arange(h) + 0.5) * ih / h).astype(int), 0, ih - 1)
    xs = np.clip(((np.arange(w) + 0.5) * iw / w).astype(int), 0, iw - 1)
    return img[ys][:, xs]
