"""Compositing passes for the EXRs (render/job.py): motion vectors, normals and positions, per-object mattes and
Cryptomatte.

What every pixel sees of the set (the objects in the shot, the pieces of broken ones, ropes and springs, the sand and
snow, debris, the floor) is traced by the stage (stage_ids.wgsl, with the trace that draws it) at the output's size,
ID_SAMPLES samples a pixel spread over it and over the shutter: how much of the pixel each thing covers, the normal and
the place of the surface seen, and how far that moves on the screen. The fire's own vectors come from its ray march
(raymarch.wgsl, with Renderer.motion set), each step weighted by what it adds to the beauty and each anti-aliasing pass
by its share of the pixel; a liquid's from its surface's velocity where the liquid is seen (liq_probe.wgsl). In front
of the set: the fire by its share of the pixel (its alpha and its light), the liquid by its coverage, and the fabric and
grass (one drawn layer, one name) by theirs in the mattes.

- Motion vectors are Nuke's layers: forward.u, forward.v (pixels to where it is at the next frame: u to the right, v
  up) and backward.u, backward.v (to the last frame), the camera's own motion in them.
- N (normals) and P (positions) are in world axes and metres, averaged over the samples that see something.
- Each thing gets a matte (matte_<name>.Y), and Cryptomatte layers CryptoObject and CryptoMaterial name them
  (io/cryptomatte.py) for any Cryptomatte node.
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass

import numpy as np

from ..engine import camera as cam
from ..io import cryptomatte as CM

PASSES = ('motion', 'normals', 'mattes', 'crypto')   # the EXR layers (Output.layers) these make
ID_SAMPLES = 16                 # samples per pixel of the set's trace (what its coverage is counted in)
CRYPTO_LAYERS = ('CryptoObject', 'CryptoMaterial')


@dataclass
class MarchMotion:
    """What the fire's march needs for its vectors (Renderer.motion): fire-local -> clip at the last, this and the next
    frame, and the simulation's seconds per frame."""
    prev: np.ndarray
    cur: np.ndarray
    next: np.ndarray
    dt: float


def frame_seconds(scene, frame):
    """Seconds of simulation from one frame to the next (what velocities are per)."""
    return float(scene.v('domain', 'time_scale', frame)) / float(scene.fps)


def clip_matrices(scene, frame, aspect):
    """Fire-local -> clip at the last, this and the next frame (the camera's and the fire's placing's own motion). At
    the shot's first and last frames the camera is taken to stand still before and after them."""
    out = []
    for f in (max(frame - 1, scene.start), frame, min(frame + 1, scene.end)):
        spec, fire = scene.camera(f)
        out.append(np.asarray(cam.compute(spec, aspect, fire).view_proj, float) @ fire.local_to_world())
    return out


def march_motion(scene, frame, size):
    """Renderer.motion for a frame rendered at size (w, h)."""
    prev, cur, nxt = clip_matrices(scene, frame, size[0] / size[1])
    return MarchMotion(prev, cur, nxt, frame_seconds(scene, frame))


def screen_motion(p, v, mats, dt, size):
    """How far fire-local points p (n, 3) moving at v (n, 3) m/s move on the screen through a pinhole (pixels, u
    right, v up), to the next frame and from the last: (n, 4). mats: clip_matrices; size: (w, h)."""
    w, h = size
    p = np.asarray(p, np.float64).reshape(-1, 3)
    v = np.asarray(v, np.float64).reshape(-1, 3)

    def px(m, q):
        c = np.concatenate([q, np.ones((len(q), 1))], 1) @ np.asarray(m, np.float64).T
        ok = c[:, 3] > 1e-6
        wc = np.where(ok, c[:, 3], 1.0)
        return np.stack([(0.5 * c[:, 0] / wc + 0.5) * w, (0.5 - 0.5 * c[:, 1] / wc) * h], 1), ok

    c, okc = px(mats[1], p)
    f, okf = px(mats[2], p + v * dt)
    b, okb = px(mats[0], p - v * dt)
    out = np.zeros((len(p), 4))
    fwd, bwd = okc & okf, okc & okb
    out[fwd, 0], out[fwd, 1] = f[fwd, 0] - c[fwd, 0], c[fwd, 1] - f[fwd, 1]
    out[bwd, 2], out[bwd, 3] = b[bwd, 0] - c[bwd, 0], c[bwd, 1] - b[bwd, 1]
    return out


# -- names ------------------------------------------------------------------------------------------------------------

@dataclass
class SetNames:
    """What the passes name in a scene, the same on every frame (so each frame's EXR has the same layers): names[code]
    and materials[code], '' for a code that names nothing; code 0 is nothing, an object's code its row + 1
    (stage_ids.wgsl), then the ropes, the floor, the matter and the rest (special)."""
    names: list
    materials: list
    special: dict    # 'floor', 'matter', 'splinters', 'debris', 'bullets', 'sparks', 'lightning', 'fabric', 'liquid' -> code
    rows: dict       # collider index -> code
    ropes: dict = dataclasses.field(default_factory=dict)   # collider index -> the code of the rope or spring it hangs on

    def code(self, owner):
        """The code of what a piece is a piece of (Stage.piece_arrays' owners)."""
        if isinstance(owner, tuple):
            return (self.ropes if owner[0] == 'rope' else self.rows).get(owner[1], 0)
        return self.special.get(owner, 0)

    def listed(self):
        """The codes that name something, in order."""
        return [c for c, n in enumerate(self.names) if n]


def _bottomless(scene):
    """A liquid over no floor (Water › Bottomless)."""
    return scene.kind in ('liquid', 'both') and bool(scene.data.get('water', {}).get('bottomless', False))


def floor_drawn(scene, footage):
    """Whether the stage draws the floor (stage.py draw's floor_on): a set without footage, over the ground."""
    return (not footage and scene.data['composite'].get('backdrop', 'stage') == 'stage'
            and bool(scene.data['domain']['ground']) and not _bottomless(scene))


def set_names(scene, footage=False):
    """The SetNames of a scene rendered over footage or not: every object in the shot (and a breakable one, whose pieces
    are), the rope or spring each hangs on (its own name: '<object> rope', 'cable', 'chain' or 'spring'), the floor
    when it is drawn, the matter, what bullets leave, lightning, the fabric and grass, the liquid; names made unique."""
    from ..engine import stage as S
    from ..engine.solids import joined
    from ..engine.solver import MAX_COLLIDERS
    from ..scene.materials import FLOORS, material
    names, mats, rows, special, ropes, taken = [''], [''], {}, {}, {}, set()

    def unique(n):
        n = str(n or '').strip() or 'Object'
        out, k = n, 2
        while out in taken:
            out, k = f'{n} {k}', k + 1
        taken.add(out)
        return out

    enabled = [i for i, c in enumerate(scene.colliders) if c['enabled']][:MAX_COLLIDERS]
    shown = S.looks(scene, footage)
    for r, ci in enumerate(enabled):
        c = scene.colliders[ci]
        rows[ci] = r + 1
        on = shown[r][0] != S.NOT_DRAWN or bool(c.get('breakable'))
        names.append(unique(c.get('name') or f'Object {r + 1}') if on else '')
        mats.append(material(c.get('material', 'wood')).label if on else '')
    names += [''] * (MAX_COLLIDERS + 1 - len(names))
    mats += [''] * (MAX_COLLIDERS + 1 - len(mats))

    def add(key, name, mat):
        special[key] = len(names)
        names.append(unique(name))
        mats.append(mat)

    for ci in enabled:
        c = scene.colliders[ci]
        kind = joined(c)
        if kind in ('rope', 'spring'):
            look = 'spring' if kind == 'spring' else (c.get('rope_look') if c.get('rope_look') in ('cable', 'chain') else 'rope')
            ropes[ci] = len(names)
            names.append(unique(f'{c.get("name") or "Object"} {look}'))
            mats.append('Rope' if look == 'rope' else material('steel').label)

    cd = scene.data['composite']
    if floor_drawn(scene, footage):
        add('floor', 'Floor', FLOORS.get(cd.get('floor', 'concrete'), FLOORS['concrete']).label)
    specs = scene.matter_specs() if scene.kind != 'cloud' else []
    if specs:
        from ..engine.matter import material as matter_material
        labels = sorted({matter_material(s.material).label for s in specs})
        one = len(specs) == 1
        add('matter', (specs[0].name or labels[0]) if one else 'Matter', labels[0] if len(labels) == 1 else 'Matter')
    if getattr(scene, 'shots', None):
        add('splinters', 'Splinters', 'Splinters')
        add('debris', 'Debris', 'Debris')
        add('bullets', 'Bullets', 'Bullets')
        add('sparks', 'Sparks', 'Sparks')
    if any(d['enabled'] and d['kind'] == 'lightning' for d in scene.lights):
        add('lightning', 'Lightning', 'Lightning')
    if scene.kind in ('fire', 'both'):
        # (fabric and grass are drawn as one layer, in front of the set: one name for what it has)
        cloth, grass = bool(scene.fabric_specs()), bool(scene.strand_specs())
        if cloth or grass:
            what = 'Fabric and grass' if (cloth and grass) else ('Fabric' if cloth else 'Grass')
            add('fabric', what, what)
    if scene.kind == 'liquid':
        add('liquid', 'Liquid', 'Liquid')
    return SetNames(names, mats, special, rows, ropes)


def safe_name(name):
    """A name as an EXR layer name compositors keep as it is: letters, digits and underscores."""
    s = re.sub(r'[^A-Za-z0-9_]+', '_', str(name)).strip('_') or 'object'
    return 'n' + s if s[0].isdigit() else s


def matte_layers(names):
    """{code: 'matte_<name>'} for every named thing, unique."""
    out, used = {}, set()
    for c in names.listed():
        base = 'matte_' + safe_name(names.names[c])
        n, k = base, 2
        while n in used:
            n, k = f'{base}_{k}', k + 1
        used.add(n)
        out[c] = n
    return out


# -- the set ----------------------------------------------------------------------------------------------------------

def _ground_y(eng, frame):
    """The ground's height (fire-local m) in the frame: the bottom of the simulation box, as the render has it."""
    src = eng.liquid if eng.kind == 'liquid' else eng.solver
    origin = src.origin
    if eng.sim_frame != frame and eng.cache is not None:
        entry = eng.cache.get(frame)
        if entry is not None and 'origin' in entry:
            origin = entry['origin']
    return float(origin[1])


def trace_set(eng, scene, frame, size, names, lens=0.0, footage=False, motion_blur=False, samples=ID_SAMPLES,
              plate_fit=(1.0, 1.0)):
    """What the set shows in each pixel at size (w, h), through a lens of distortion `lens` (stage_ids.wgsl):
    {'codes', 'cover': (h, w, 6) the things covering most of each pixel, 'N', 'P': (h, w, 3) world, 'seen': (h, w) how
    much of it shows something, 'motion': (h, w, 4) forward u, v, backward u, v (pixels), 'over': pixels that saw more
    things than they keep count of}. plate_fit: the footage's scale in the picture (stage.wgsl U.fit), where its
    holdouts (the renderer's, as the last render set them) are looked up."""
    from ..engine.stage import ID_HEAD, IdPass, StageLight
    W, H = int(size[0]), int(size[1])
    spec, fire = scene.camera(frame)
    cs = cam.compute(spec, W / H, fire)
    prev, cur, nxt = clip_matrices(scene, frame, W / H)
    head = np.zeros((ID_HEAD, 4), np.float32)
    head[0:4], head[4:8], head[8:12] = prev.T, cur.T, nxt.T
    head[12, 2:] = (samples, frame_seconds(scene, frame))
    sp = names.special
    head[13] = (sp.get('floor', 0), sp.get('matter', 0), sp.get('splinters', 0), 0.0 if footage else 1.0)
    head[14, 2:] = plate_fit
    req = IdPass(head, names.code)
    comp = dataclasses.replace(scene.comp(frame, 'composite'), lens_k1=float(lens))
    shutter = 0.0
    if motion_blur:
        shutter = scene.data['render']['shutter_angle'] / 360.0 / scene.fps * scene.v('domain', 'time_scale', frame)
    cols = scene.colliders_gpu(frame, eng.floating_overrides(frame))
    matter = eng.matter_for(frame) if 'matter' in sp else None   # (its surface is built before the batch)
    shots = eng.shot_view(frame) if 'debris' in sp else None
    bolts = scene.bolts(frame) if 'lightning' in sp else None
    with eng.gpu.batch() as b:
        eng.stage.draw(b, eng.renderer, scene, cs, fire, cols, eng.solver.meshes, StageLight(), comp, (W, H),
                       samples=1, shutter=shutter, footage=footage, ground_y=_ground_y(eng, frame), frame=frame,
                       floor=not _bottomless(scene), pieces=eng.piece_poses(frame),
                       ropes=eng.rope_poses(frame), matter=matter, bolts=bolts, shots=shots, ids=req)
    o = req.out
    r = o[:, :, 0:3, :].reshape(H, W, 6, 2)
    return {'codes': np.rint(r[..., 0]).astype(np.int32), 'cover': r[..., 1].copy(), 'N': o[:, :, 3, :3].copy(),
            'seen': o[:, :, 3, 3].copy(), 'P': o[:, :, 4, :3].copy(), 'motion': o[:, :, 5, :].copy(),
            'over': int(np.count_nonzero(o[:, :, 4, 3] > 0.5))}


# -- what is in front of it: the fire, a liquid -----------------------------------------------------------------------

def _view_rays(scene, frame, size, ys, xs):
    """For pixels (ys, xs) at size (w, h), their centres' rays (pinhole): where each starts on the near plane and its
    unit direction (world), (n, 3) each, and the fire's placing."""
    w, h = size
    spec, fire = scene.camera(frame)
    cs = cam.compute(spec, w / h, fire)
    x, y = (np.asarray(xs, np.float64) + 0.5) / w, (np.asarray(ys, np.float64) + 0.5) / h
    ndc = np.stack([x * 2 - 1, 1 - y * 2, np.zeros_like(x), np.ones_like(x)], -1)
    a = ndc @ cs.inv_view_proj.T
    ndc[:, 2] = 1.0
    f = ndc @ cs.inv_view_proj.T
    a = a[:, :3] / a[:, 3:]
    d = f[:, :3] / f[:, 3:] - a
    return a, d / np.linalg.norm(d, axis=-1, keepdims=True), fire


PROBE_CHUNK = 1 << 20   # points the surface is sampled at in one dispatch (liq_probe.wgsl: 48 MB of buffers)


def surface_at(gpu, surf, nf, q):
    """A liquid's surface grid (LiquidRenderer.surf, dims nf: x, y, z) sampled on the GPU at grid places q (n, 3: cells
    from its corner) as its march samples it (liq_probe.wgsl): (velocity (n, 3) m/s, the distance's gradient (n, 3)).
    Only the points go up and their samples come back: the grid itself can be hundreds of megabytes."""
    from ..engine.gpu import Uniforms, groups_1d
    q = np.asarray(q, np.float32).reshape(-1, 3)
    n = len(q)
    vel, grad = np.zeros((n, 3), np.float32), np.zeros((n, 3), np.float32)
    if not n:
        return vel, grad
    k = gpu.kernel('liq_probe.wgsl', ['tex3d', 'smp', 'rbuf', 'buf'], workgroup=(64, 1, 1))
    m = min(n, PROBE_CHUNK)
    pts, out = gpu.buffer(m * 16, 'liquid-probe-points'), gpu.buffer(m * 32, 'liquid-probe')
    try:
        for i0 in range(0, n, m):
            c = min(m, n - i0)
            gpu.write_buffer(pts, np.concatenate([q[i0:i0 + c], np.zeros((c, 1), np.float32)], 1))
            with gpu.batch() as b:
                b.run(k, [surf, gpu.linear, pts, out], Uniforms().v4(*nf, c), groups=groups_1d(c))
            r = np.frombuffer(gpu.read_buffer(out, c * 32), np.float32).reshape(c, 2, 4)
            vel[i0:i0 + c], grad[i0:i0 + c] = r[:, 0, :3], r[:, 1, :3]
    finally:
        pts.destroy()
        out.destroy()
    return vel, grad


def liquid_front(eng, scene, frame, size, aov):
    """The liquid seen in a liquid scene's element just rendered (its aux and mattes in aov): {'share' (h, w): its
    coverage, 'motion' (h, w, 4), 'N', 'P' (h, w, 3) world}, from its surface grid (LiquidRenderer.surf: the distance
    to the surface and the velocity, m/s) at the surface each pixel sees (surface_at); or None."""
    lr = getattr(eng, 'liquid_r', None)
    L = getattr(eng, 'liquid', None)
    if lr is None or L is None or lr.surf is None or lr.nf is None or 'mattes' not in aov:
        return None
    w, h = size
    water = np.clip(np.asarray(aov['mattes'], np.float32)[..., 0], 0.0, 1.0)
    depth = np.asarray(aov['aux'], np.float32)[..., 1]
    if water.shape != (h, w):
        return None
    on = (water > 1e-3) & (depth > 0.0)
    out = {'share': water, 'motion': np.zeros((h, w, 4), np.float32), 'N': np.zeros((h, w, 3), np.float32),
           'P': np.zeros((h, w, 3), np.float32)}
    if not on.any():
        return out
    ys, xs = np.nonzero(on)
    ro, rd, fire = _view_rays(scene, frame, size, ys, xs)
    pw = ro + rd * depth[ys, xs][:, None].astype(np.float64)              # world
    l2w = fire.local_to_world()
    pl = (pw - l2w[:3, 3]) @ l2w[:3, :3]                                   # fire-local
    origin = np.asarray(L.origin, float)
    if eng.sim_frame != frame and eng.cache is not None:
        entry = eng.cache.get(frame)
        if entry is not None and 'origin' in entry:
            origin = np.asarray(entry['origin'], float)
    vs = float(L.h) / float(lr.fscale)                                     # the surface grid's cell (m)
    vel, grad = surface_at(eng.gpu, lr.surf, lr.nf, (pl - origin) / vs)
    nl = grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-12)
    nw = nl @ l2w[:3, :3].T
    nw = np.where((np.sum(nw * rd, 1) > 0.0)[:, None], -nw, nw)            # (facing the camera)
    mats = clip_matrices(scene, frame, w / h)
    out['motion'][ys, xs] = screen_motion(pl, vel, mats, frame_seconds(scene, frame), size)
    out['N'][ys, xs] = nw
    out['P'][ys, xs] = pw
    return out


# -- through the lens -------------------------------------------------------------------------------------------------

def _lens(uv, k1, aspect, forward):
    """The footage's lens (composite.wgsl undistort; forward: pinhole -> lens) on uv (..., 2)."""
    asp = np.array([aspect, 1.0])
    hd = 0.5 * np.linalg.norm(asp)
    d = (uv - 0.5) * asp / hd
    if forward:
        u = d * (1.0 + k1 * np.sum(d * d, -1, keepdims=True))
    else:
        u = d.copy()
        for _ in range(5):
            u = d / (1.0 + k1 * np.sum(u * u, -1, keepdims=True))
    return u * hd / asp + 0.5


def through_lens(front, k1, size):
    """What is in front of the set (fire_front, liquid_front: element pixels, a pinhole's) as the composite shows it
    through the footage's lens (k1): each picture pixel takes the element pixel it shows, its vectors bent as the lens
    bends them."""
    if k1 == 0.0 or front is None:
        return front
    w, h = size
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64) + 0.5
    uv = np.stack([xs / w, ys / h], -1)
    euv = _lens(uv, k1, w / h, False)
    inside = np.all((euv >= 0.0) & (euv <= 1.0), -1)
    ex = np.clip((euv[..., 0] * w).astype(int), 0, w - 1)
    ey = np.clip((euv[..., 1] * h).astype(int), 0, h - 1)
    out = {}
    for k, v in front.items():
        out[k] = np.where(inside[..., None] if v.ndim == 3 else inside, v[ey, ex], 0.0)
    mv = out.get('motion')
    if mv is not None:
        for a, b in ((0, 1), (2, 3)):
            end = euv + np.stack([mv[..., a] / w, -mv[..., b] / h], -1)
            d = (_lens(end, k1, w / h, True) - _lens(euv, k1, w / h, True)) * np.array([w, h])
            mv[..., a], mv[..., b] = d[..., 0], -d[..., 1]
        out['motion'] = mv.astype(np.float32)
    return out


# -- the frame --------------------------------------------------------------------------------------------------------

def frame_passes(eng, scene, frame, size, names, aov, lens=0.0, footage=False, motion_blur=False, fire_vec=None,
                 plate_fit=(1.0, 1.0)):
    """The passes of the picture `eng` has just rendered at size (w, h) (aov: its element's, Engine.aovs, with a
    liquid's mattes): the set's (trace_set) through a lens of distortion `lens` (0: the element's pinhole; the
    composite's: Composite › Lens distortion), the fire (fire_vec: Renderer.read_vectors, its vectors and its share of
    each pixel; in a fire-and-liquid scene the fire in front of the liquid) or the liquid in front by their share.
    plate_fit: the footage's scale in the picture (RenderJob._plate_fit), where its holdouts are looked up. Returns
    trace_set's dict with the motion, N and P of what is in front folded in, the codes and cover with the fabric's
    (aov's 'fabric') and the liquid's, and 'notes': what was cut short."""
    out = trace_set(eng, scene, frame, size, names, lens, footage, motion_blur, plate_fit=plate_fit)
    front = None
    if fire_vec is not None and np.asarray(fire_vec[0]).shape[:2] == out['seen'].shape:
        front = {'share': np.asarray(fire_vec[1], np.float32), 'motion': np.asarray(fire_vec[0], np.float32)}
    elif scene.kind == 'liquid':
        front = liquid_front(eng, scene, frame, size, aov)
    front = through_lens(front, float(lens), size)
    fab = aov.get('fabric') if names.special.get('fabric') else None
    if fab is not None and np.asarray(fab).shape[:2] == out['seen'].shape:
        # the fabric and the grass (one layer, gathered over the anti-aliasing passes: its alpha is how much of the
        # pixel it covers, in front of the set): in the mattes and Cryptomatte, over what the set shows
        fa = through_lens({'share': np.clip(np.asarray(fab, np.float32)[..., 3], 0.0, 1.0)}, float(lens), size)['share']
        fa = fa[..., None]
        out['cover'] = np.concatenate([out['cover'] * (1.0 - fa), fa], -1)
        out['codes'] = np.concatenate([out['codes'], np.full(fa.shape, names.special['fabric'], np.int32)], -1)
    notes = []
    if out['over']:
        notes.append(f'Compositing passes: {out["over"]} pixel{"s" if out["over"] != 1 else ""} saw more than 8 things; '
                     'the mattes and Cryptomatte there leave out the ones that cover the least.')
    if front is not None:
        s = front['share'][..., None]
        out['motion'] = (front['motion'] * s + out['motion'] * (1.0 - s)).astype(np.float32)
        if 'N' in front:
            # a liquid: its surface in front, its own normal and place where it covers most of the pixel
            near = s[..., 0] >= 0.5
            out['N'] = np.where(near[..., None], front['N'], out['N'])
            out['P'] = np.where(near[..., None], front['P'], out['P'])
            lc = names.special.get('liquid', 0)
            if lc:
                out['cover'] = np.concatenate([out['cover'] * (1.0 - s), s], -1)
                out['codes'] = np.concatenate([out['codes'], np.full(s.shape, lc, np.int32)], -1)
    out['notes'] = notes
    return out


def channels(passes, names, layers, keep32):
    """The EXR channels of a frame's passes for these Output.layers: {name: (h, w) float32}, and the header attributes
    (Cryptomatte's). keep32 is given the names that must stay 32-bit float in a half EXR (Cryptomatte's ids and
    coverage). (P follows the EXR: in half, steps of 4 mm or less within 8 m of the world's origin, ample for comp, at
    a sixteenth of the size: averaged positions hardly compress in float.)"""
    ch, attrs = {}, {}
    if 'motion' in layers:
        mv = passes['motion']
        ch.update({'forward.u': mv[..., 0], 'forward.v': mv[..., 1], 'backward.u': mv[..., 2], 'backward.v': mv[..., 3]})
    if 'normals' in layers:
        n, p = passes['N'], passes['P']
        ch.update({'N.R': n[..., 0], 'N.G': n[..., 1], 'N.B': n[..., 2], 'P.R': p[..., 0], 'P.G': p[..., 1], 'P.B': p[..., 2]})
    codes, cover = passes['codes'], passes['cover']
    if 'mattes' in layers:
        # (the first rank whole, the others only where more than one thing shows: most of a picture has one)
        k = codes.shape[-1]
        c0, v0 = np.ascontiguousarray(codes[..., 0]), np.ascontiguousarray(cover[..., 0])
        fc, fv = codes.reshape(-1, k), cover.reshape(-1, k)
        multi = np.nonzero(np.any(fv[:, 1:] > 0.0, axis=1))[0]
        mc, mv = fc[multi, 1:], fv[multi, 1:]
        for c, layer in matte_layers(names).items():
            m = np.where(c0 == c, v0, np.float32(0.0))
            if multi.size:
                m.reshape(-1)[multi] += np.where(mc == c, mv, 0.0).sum(1)
            ch[layer + '.Y'] = m
    if 'crypto' in layers:
        for layer, table in zip(CRYPTO_LAYERS, (names.names, names.materials)):
            uniq = sorted({t for t in table if t})
            idx_of = np.array([uniq.index(t) if t else -1 for t in table] + [-1], np.int32)
            cc, aa = CM.channels(layer, idx_of[np.clip(codes, 0, len(table))], cover, uniq)
            ch.update(cc)
            keep32.update(cc)
            attrs.update(aa)
    return {k: np.asarray(v, np.float32) for k, v in ch.items()}, attrs
