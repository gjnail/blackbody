"""The stage (stage.wgsl): the set drawn in CG, as the background plate the fire and the liquid go over.

Without footage it is the floor (with the ground on), the objects in the shot in their materials and the sky,
or a flat background colour behind the objects. With footage, it is the footage with the objects drawn in CG
over it, and their shadows on it.

The composite and the liquid's march take it in place of the footage: it is already scene-linear, and its
alpha says how much of each pixel is CG, which the composite does not light again with the fire and the lamps
(the stage lit it).

Which objects are drawn (look):
- not in the shot (Hides fire off): never;
- In the footage: the real object is in the footage. It is not drawn, but it hides the CG behind it and shades it;
- CG: drawn in its material (scene/materials.py), or in its own colour;
- Automatic: CG without footage, and for things that fall or float (they cannot be in the footage).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

from . import camera as cam
from . import lume as LU
from .gpu import Uniforms
from .liquid_render import lava_table
from .solver import MAX_COLLIDERS, _mesh_ref, pack_colliders
from ..scene.materials import FLOORS, PATTERNS, material

NOT_DRAWN, IN_FOOTAGE, CG = 0, 1, 2
SUN_SHARPNESS = 24.0     # the key light's soft shadows: 1 / tan of its angular radius (about 2.4 degrees)
HORIZON_FADE = 150.0     # m: far off, the floor fades into the sky at the horizon over this distance
INPUT_LINEAR = 2         # composite.wgsl input transform: scene-linear
ROPE_ROW = MAX_COLLIDERS  # the material rows after the objects': rope (manila), then steel (cables, springs)
ROPE_LOOKS = (((0.456, 0.305, 0.127), 0.85, 0.0), ((0.55, 0.56, 0.57), 0.4, 1.0))   # (colour, roughness, metal)
LIGHTNING_ROW = ROPE_ROW + 2   # after them: lightning (its glow is per segment)
BOLT_GLOW = 0.02          # how much of lightning's core radiance it scatters into the air round it
GLOW_T0, GLOW_DT = 700.0, 100.0   # K: hot matter's blackbody table's first temperature and step (16 entries)
GLOW_BLOCK = 8            # matter grid nodes to a block of hot surface lighting what is round it (matter_glow.wgsl)
MATTER_LIGHTS = 256       # the most of those lights
STRANDS = {0: (3.0, 7.0), 1: (6.0, 9.0), 2: (0.0, 0.0)}   # ropes.ROPE, CABLE, SPRING: strands, twist (radii per turn)


def drawn(c, footage):
    """How object c (a collider) is drawn: NOT_DRAWN, IN_FOOTAGE or CG."""
    if not c.get('holdout', True):
        return NOT_DRAWN
    look = c.get('look', 'auto')
    if look == 'cg':
        return CG
    if look == 'footage':
        return IN_FOOTAGE
    moves = c.get('dynamic') or c.get('floating') or c.get('breakable') or c.get('joint', 'none') != 'none'
    return IN_FOOTAGE if footage and not moves else CG


def looks(scene, footage):
    """Per object in the shot (the enabled colliders, in the solver's order): (drawn, colour, roughness,
    metal, clear, pattern, inside colour, ior)."""
    out = []
    for c in [c for c in scene.colliders if c['enabled']][:MAX_COLLIDERS]:
        m = material(c.get('material', 'wood'))
        colour = tuple(c['colour']) if c.get('own_colour') else m.colour
        out.append((drawn(c, footage), colour, m.roughness, m.metal, m.clear, PATTERNS.get(m.pattern, 0),
                    m.inside or colour, m.ior))
    return out


def standin_colours(scene):
    """Per object (the solver's order), the colour a liquid scene's grey stand-in takes instead of the look's
    (r, g, b, 1), or (0, 0, 0, 0): things that fall or float, and objects set to CG, in their material's colour."""
    out = []
    for c in [c for c in scene.colliders if c['enabled']][:MAX_COLLIDERS]:
        own = c.get('dynamic') or c.get('floating') or c.get('joint', 'none') != 'none' or c.get('look') == 'cg'
        colour = tuple(c['colour']) if c.get('own_colour') else material(c.get('material', 'wood')).colour
        out.append((*colour, 1.0) if own else (0.0, 0.0, 0.0, 0.0))
    return out


def see_through(scene):
    """Indices (in the solver's list) of the objects light passes through (glass, ice): the fire behind them
    is seen through them, so they do not hold it out."""
    cols = [c for c in scene.colliders if c['enabled']][:MAX_COLLIDERS]
    return {i for i, c in enumerate(cols) if material(c.get('material', 'wood')).clear > 0.5 and c.get('look') != 'footage'}


def wanted(scene, footage, mode='composite', objects=True):
    """Whether the stage draws anything this render (objects: it draws the CG objects; a liquid scene with
    grey stand-ins draws its own)."""
    if mode != 'composite':
        return False
    any_cg = objects and any(row[0] == CG for row in looks(scene, footage))
    if footage:
        return any_cg
    return any_cg or scene.data['composite'].get('backdrop', 'stage') == 'stage'


@dataclass
class StageLight:
    """What lights the stage, in the units of the render it goes into."""
    sky: tuple = (0.1, 0.11, 0.13)      # sky radiance (rgb)
    sun: tuple = (0.0, 0.0, 0.0)        # key light: irradiance at normal incidence (rgb)
    sun_dir: tuple = (0.0, 1.0, 0.0)    # toward the key light (world)
    fire: tuple = (0.0, 0.0, 0.0)       # gain on the fire's point lights (rgb); 0: no fire light
    fire_shadows: float = 1.0           # how much objects and smoke shade it (0..1)
    lamps: int = 0                      # lights in the set in the renderer's lamp buffer
    lamp_gain: float = 1.0              # their light times this (the liquid render's exposure)
    env: object = None                  # an environment HDRI (texture), its rotation (degrees), strength
    env_rotation: float = 0.0
    env_strength: float = 1.0


def fire_light(scene, frame, look, comp):
    """The stage's light in a fire scene (the fire's Lighting, and the composite's fire light settings)."""
    gain = float(getattr(look, 'fire_scatter', 1.0)) * float(comp.surface_light)
    return StageLight(
        sky=tuple(float(x) for x in look.ambient),
        sun=tuple(float(x) * float(look.sun_intensity) for x in look.sun_color),
        sun_dir=tuple(cam.sun_direction(look.sun_azimuth, look.sun_elevation)),
        fire=tuple(gain * float(t) for t in comp.tint), fire_shadows=float(comp.surface_shadows))


def water_light(wlook, comp=None, fire_look=None):
    """The stage's light in a liquid scene: the water look's sky and key light, in its exposure; with fire
    (a fire and liquid scene), the fire's point lights too, as on the fabric."""
    gain = 2.0 ** float(wlook.exposure)
    fire = (0.0, 0.0, 0.0)
    if fire_look is not None and comp is not None:
        k = float(getattr(fire_look, 'fire_scatter', 1.0)) * float(comp.surface_light) * gain
        fire = tuple(k * float(t) for t in comp.tint)
    return StageLight(
        sky=tuple(float(x) * gain for x in wlook.sky), sun=tuple(float(x) * gain for x in wlook.sun),
        sun_dir=tuple(cam.sun_direction(wlook.sun_azimuth, wlook.sun_elevation)), fire=fire,
        fire_shadows=float(comp.surface_shadows) if comp is not None else 1.0, lamp_gain=gain)


_LUM = {}


def matter_glow_scale(look):
    """(the stage radiance of a blackbody at 1300 K, the dynamic range) as the fire's look shows blackbody light: a
    thick flame at Flame temperature is its Intensity times 2^Exposure, brighter and dimmer by the physical ratio of
    their luminances to the power Dynamic range."""
    from .lut import luminance
    fk = float(look.flame_k)
    if fk not in _LUM:
        _LUM[fk] = math.log10(max(luminance(fk), 1e-300) / luminance(1300.0))
    dr = float(look.dynamic_range)
    return float(look.intensity) * 2.0 ** float(look.exposure) * 10.0 ** (-_LUM[fk] * dr), dr


class Stage:
    def __init__(self, gpu):
        self.gpu = gpu
        self.k = gpu.kernel('stage.wgsl', ['utex3d', 'tex3d', 'tex3d', 'tex3d', 'tex3d', 'smp', 'smp', 'rbuf', 'rbuf',
                                           'rbuf', 'tex2d', 'tex2d', 'tex2d', 'tex3d', 'st2d:rgba16float:w',
                                           'rbuf', 'rbuf', 'rbuf', 'rbuf', 'st2d:rgba16float:w', 'tex3d', 'tex3d', 'tex2d',
                                           'utex3d', 'utex3d', 'rbuf', 'tex3d', 'rbuf', 'buf', 'buf', 'rbuf'],
                            workgroup=(8, 8, 1))
        mg = ['utex3d', 'utex3d', 'buf', 'buf']
        self.k_mglow = gpu.kernel('matter_glow.wgsl', mg, 'lights', workgroup=(4, 4, 4))
        self.k_mglow_finish = gpu.kernel('matter_glow.wgsl', mg, 'finish', workgroup=(1, 1, 1))
        self._ml = gpu.buffer((1 + 2 * MATTER_LIGHTS) * 16, 'stage-matter-lights')
        self._mlc = gpu.buffer(16, 'stage-matter-light-count')
        self._no_matter = gpu.texture3d((1, 1, 1), 'rgba16float', 'stage-no-matter')
        self.tex = None
        self.hold = None           # the holdouts it leaves for the march (pieces' distance, the footage's matte)
        self.has_pieces = False
        self.has_matter = False
        self._bufs = {}
        self._zero = gpu.buffer(64, 'stage-no-lights')
        self._env = None
        self._env_key = None
        self.env_sky = None     # the HDRI's average colour (the sky's light), when it has one
        self.lume = None        # Lume's state (lume.Lume), once a scene asks for Lume
        self._lume_off = [gpu.buffer(16, 'stage-no-acc'), gpu.buffer(32, 'stage-no-aov'), gpu.buffer(16, 'stage-no-env')]

    @property
    def lume_pending(self):
        """Whether the viewer should refine again: Lume wants more passes of the picture it is showing."""
        return self.lume is not None and self.lume.pending

    def _ensure(self, w, h):
        if self.tex is not None and self.tex.size[:2] == (w, h):
            return
        for t in (self.tex, self.hold):
            if t is not None:
                t.destroy()
        self.tex = self.gpu.texture2d(w, h, 'rgba16float', 'stage')
        self.hold = self.gpu.texture2d(w, h, 'rgba16float', 'stage-holdout')

    def _buffer(self, name, data):
        """Upload an array into a named storage buffer, grown as needed."""
        data = np.ascontiguousarray(data)
        size = max(int(data.nbytes), 16)
        b = self._bufs.get(name)
        if b is None or b.size < size:
            if b is not None:
                b.destroy()
            b = self._bufs[name] = self.gpu.buffer(max(size, 1024), f'stage-{name}')
        if data.nbytes:
            self.gpu.write_buffer(b, data)
        return b

    @staticmethod
    def piece_arrays(scene, pieces, shutter=0.0, ropes=None, ground_y=None, bolts=None):
        """The pieces of broken objects and the segments of ropes and springs, for the shader: (pieces (n, 5, 4),
        planes (p, 4), grid corner, cell size, grid dims, cells (g, 2) uint32, list uint32), or None.
        pieces: {collider index: Solids.piece_poses entry}; ropes: {collider index: Solids.rope_poses entry};
        ground_y: the ground's height (a snapped rope hangs down to it); bolts: lightning (Scene.bolts)."""
        from .ropes import num, prism_planes, rope_points, segments
        from .solids import fractured
        enabled = [i for i, c in enumerate(scene.colliders) if c['enabled']][:MAX_COLLIDERS]
        row_of = {ci: r for r, ci in enumerate(enabled)}
        P, PL, centres, radii = [], [], [], []
        first = 0
        for ci, pose in (pieces or {}).items():
            if ci not in row_of or ci >= len(scene.colliders):
                continue
            frac = fractured(scene.colliders[ci], pose['size'], num(pose.get('hollow', 0.0)))
            n = min(len(frac.pieces), len(pose['pos']))
            burn = pose.get('burn')
            for k in range(n):
                pc = frac.pieces[k]
                pos = np.asarray(pose['pos'][k], float)
                if pos[1] < -1.0e3:
                    continue                         # (burnt to ash)
                pl = pc.planes.copy()
                pl[:, 3] -= pl[:, :3] @ pc.centroid
                pl[pc.inner, :3] *= 2.0              # (a cut face: drawn in the inside colour)
                rad = float(np.linalg.norm(pc.verts - pc.centroid, axis=1).max())
                vel = np.asarray(pose['vel'][k], float)
                P.append([[*pos, len(pl)], [*np.asarray(pose['quat'][k], float)], [*vel, first],
                          [*np.asarray(pose['omega'][k], float), row_of[ci]], [*pc.centroid, rad],
                          [*(np.asarray(burn[k], float) if burn is not None and k < len(burn) else (0.0, 0.0, 0.0, 0.0))]])
                PL.append(pl)
                first += len(pl)
                centres.append(pos)
                radii.append(rad + float(np.linalg.norm(vel)) * 0.5 * shutter)
        # ropes and springs: straight segments along them (their r: strands, distance along it, twist, radius)
        for ci, rope in (ropes or {}).items():
            pts, vel = rope_points(rope, ground=ground_y)
            if len(pts) < 2:
                continue
            look = int(round(num(rope['look'])))
            row = ROPE_ROW + (0 if look == 0 else 1)
            strands, twist = STRANDS.get(look, (0.0, 0.0))
            rad = max(num(rope['radius']), 1e-4)
            for centre, quat, v, half, along in segments(pts, vel, rad):
                pl = prism_planes(rad, half)
                bound = math.hypot(half, rad)
                P.append([[*centre, len(pl)], [*quat], [*v, first], [0.0, 0.0, 0.0, row], [strands, along, twist, bound], [0.0] * 4])
                PL.append(pl)
                first += len(pl)
                centres.append(centre)
                radii.append(bound + float(np.linalg.norm(v)) * 0.5 * shutter)
        # lightning: glowing segments along its channels (their r: the glow, linear rgb)
        for pts, core, glow in (bolts or []):
            rad = max(float(core), 1e-4)
            for centre, quat, v, half, _along in segments(np.asarray(pts, float), np.zeros((len(pts), 3)), rad):
                pl = prism_planes(rad, half)
                bound = math.hypot(half, rad)
                P.append([[*centre, len(pl)], [*quat], [0.0, 0.0, 0.0, first], [0.0, 0.0, 0.0, LIGHTNING_ROW], [*glow, bound], [0.0] * 4])
                PL.append(pl)
                first += len(pl)
                centres.append(centre)
                radii.append(bound)
        if not P:
            return None
        P = np.asarray(P, np.float32)
        PL = np.concatenate(PL).astype(np.float32)
        c = np.asarray(centres)
        r = np.asarray(radii)[:, None]
        lo, hi = (c - r).min(0), (c + r).max(0)
        ext = float(np.max(hi - lo))
        cell = max(2.0 * float(np.median(r)), ext / 48.0, 1e-3)
        dims = np.minimum(np.maximum(np.ceil((hi - lo) / cell).astype(int), 1), 64)
        cell = max(cell, float(np.max((hi - lo) / dims)))
        lists = [[] for _ in range(int(np.prod(dims)))]
        a = np.clip(np.floor((c - r - lo) / cell).astype(int), 0, dims - 1)
        b = np.clip(np.floor((c + r - lo) / cell).astype(int), 0, dims - 1)
        for k in range(len(P)):
            for z in range(a[k, 2], b[k, 2] + 1):
                for y in range(a[k, 1], b[k, 1] + 1):
                    for x in range(a[k, 0], b[k, 0] + 1):
                        lists[x + dims[0] * (y + dims[1] * z)].append(k)
        counts = np.array([len(L) for L in lists], np.uint32)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.uint32)
        flat = np.array([k for L in lists for k in L], np.uint32)
        return P, PL, lo, cell, dims, np.stack([starts, counts], 1), flat

    def environment(self, path):
        """Load (or reuse) an HDRI for the sky (fire scenes; the liquid renderer has its own)."""
        if not path:
            self.env_sky = None
            return None
        try:
            key = (path, os.path.getmtime(path))
        except OSError:
            self.env_sky = None
            return None
        if key != self._env_key:
            from ..io.hdri import load_hdri, sky_average
            img = load_hdri(path)
            if self._env is not None:
                self._env.destroy()
            h, w = img.shape[:2]
            self._env = self.gpu.texture2d(w, h, 'rgba16float', 'stage-hdri')
            rgba = np.concatenate([img, np.ones((h, w, 1), np.float32)], -1)
            self.gpu.upload(self._env, np.minimum(rgba, 6.0e4).astype(np.float16))
            self.env_sky = tuple(float(x) for x in sky_average(img))
            self._env_key = key
        return self._env

    @staticmethod
    def _bound(cols, rows, meshes, shutter, extra=None):
        """A sphere round every object in the shot (fire-local centre, radius), moving ones over the shutter, and round
        `extra` ((lo, hi) fire-local m: the matter)."""
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        if extra is not None:
            lo, hi = np.minimum(lo, extra[0]), np.maximum(hi, extra[1])
        for c, row in zip(cols, rows):
            if row[0] == NOT_DRAWN:
                continue
            s = np.abs(np.asarray(c.size, float))
            if c.shape == 'sphere':
                r = s[0]
            elif c.shape == 'cylinder':
                r = math.hypot(s[0], s[1])
            elif c.shape == 'mesh':
                m0, m1 = _mesh_ref(meshes, c.mesh, c.mesh_frame, c.mesh_fps)[:2]
                corner = np.maximum(np.abs(np.asarray(m0[:3], float)), np.abs(np.asarray(m1[:3], float)))
                r = float(np.linalg.norm(corner)) * float(s.max()) if m0[3] >= 0 else 0.0
            else:
                r = float(np.linalg.norm(s))
            r += float(np.linalg.norm(c.vel)) * 0.5 * shutter + 1e-3
            p = np.asarray(c.pos, float)
            lo, hi = np.minimum(lo, p - r), np.maximum(hi, p + r)
        if not np.all(np.isfinite(lo)):
            return (0.0, 0.0, 0.0), 0.0
        centre = 0.5 * (lo + hi)
        return tuple(centre), float(np.linalg.norm(hi - lo) * 0.5)

    def draw(self, b, r, scene, camstate, fire, colliders, meshes, light: StageLight, comp, size, plate_fit=(1.0, 1.0),
             samples=1, shutter=0.0, footage=False, vol=None, ground_y=0.0, frame=0, objects=True, floor=True,
             pieces=None, ropes=None, matter=None, bolts=None, grass=None, burns=None, final=False):
        """Draw the stage into self.tex (size: the plate's, footage or output) and return it.
        r: the Renderer (its footage plate and holdouts, light volume, fire lights and lamp buffer);
        colliders: the objects as the solver has them (ColliderGPU, moving ones where they are this frame);
        vol: the fire's volume when this frame's light volume is lit (Renderer.light), else None;
        ground_y: the ground's height (fire-local m: the bottom of the simulation box);
        objects: draw the CG objects (else they only shade the floor: a liquid scene's grey stand-ins);
        floor: draw the floor (not under bottomless water);
        pieces: the pieces of broken objects, {collider index: Solids.piece_poses entry};
        ropes: the ropes and springs, {collider index: Solids.rope_poses entry};
        matter: the sand, snow, mud, jelly and clay (matter.Matter), or None; bolts: lightning (Scene.bolts);
        final: a final render (with Lume: all its passes now; else the viewer's, a few at a time)."""
        g = self.gpu
        pw, ph = int(size[0]), int(size[1])
        self._ensure(pw, ph)
        rows = looks(scene, footage)
        if not objects:
            rows = [(IN_FOOTAGE if row[0] == CG else row[0],) + tuple(row[1:]) for row in rows]
        cols = list(colliders or [])[:MAX_COLLIDERS]
        rows = rows[:len(cols)] + [(NOT_DRAWN, (0.5, 0.5, 0.5), 0.6, 0.0, 0.0, 0, (0.5, 0.5, 0.5), 1.5)] * (len(cols) - len(rows))
        cd = scene.data['composite']
        d = scene.data['domain']
        l2w = np.asarray(fire.local_to_world(), float)
        w2l = np.linalg.inv(l2w)
        view = np.asarray(camstate.view)
        fwd_w = -view[2, :3] / max(np.linalg.norm(view[2, :3]), 1e-12)
        p0 = camstate.inv_view_proj @ np.array([0.0, 0.0, 0.0, 1.0])
        near = float(np.dot(p0[:3] / p0[3] - np.asarray(camstate.eye, float), fwd_w))
        # the angle one plate pixel spans at the middle of the picture
        fit = (float(plate_fit[0]), float(plate_fit[1]))
        dx = 2.0 / max(pw * fit[0], 1.0)
        a = camstate.inv_view_proj @ np.array([0.0, 0.0, 1.0, 1.0])
        c = camstate.inv_view_proj @ np.array([dx, 0.0, 1.0, 1.0])
        e = np.asarray(camstate.eye, float)
        va, vc = a[:3] / a[3] - e, c[:3] / c[3] - e
        pix = float(np.linalg.norm(np.cross(va, vc)) / max(np.dot(va, vc), 1e-12))
        # what lights it
        sd_l = w2l[:3, :3] @ np.asarray(light.sun_dir, float)
        sd_l = sd_l / max(float(np.linalg.norm(sd_l)), 1e-12)
        env = light.env
        sky = light.sky
        vol_on = vol is not None and getattr(r, 'L0', None) is not None
        if vol_on:
            dims, hcell, org = vol.dims, vol.h, vol.origin
            ld = r.light_dims
            lsc = tuple(np.asarray(ld, float) / np.asarray(dims, float))
        else:
            hcell, org, ld, lsc = 1.0, (0.0, 0.0, 0.0), (1, 1, 1), (1.0, 1.0, 1.0)
        fire_on = vol_on and r.lights is not None and max(light.fire) > 0.0
        # the floor
        fl = FLOORS.get(cd.get('floor', 'concrete'), FLOORS['concrete'])
        tint = np.asarray(cd.get('floor_tint', (1.0, 1.0, 1.0)), float)
        stage_on = not footage and cd.get('backdrop', 'stage') == 'stage'
        floor_on = stage_on and bool(d['ground']) and floor
        # objects: how far ambient occlusion reaches (about half the size of a typical one)
        sizes = [float(np.max(np.abs(cg.size))) for cg, row in zip(cols, rows) if row[0] != NOT_DRAWN]
        ao_reach = float(np.clip(0.6 * np.median(sizes), 0.05, 1.5)) if sizes else 0.3
        surf = matter.surface() if matter is not None and matter.active else None
        mb = matter.world_bounds() if surf is not None else None
        if mb is None:
            surf = None
        centre, radius = self._bound(cols, rows, meshes, shutter, mb)
        # the footage under it
        matte_on, depth_on = r.hold_on if (footage and r.hold is not None) else (False, False)
        from .renderer import DEPTH_KINDS, INPUT_TRANSFORMS
        transform = INPUT_TRANSFORMS.get(comp.plate_transform, 0)
        vis = float(comp.visibility)
        ns = 1 if samples <= 1 else (8 if (shutter > 0.0 and samples >= 8) else 4)
        lume = LU.settings(scene)
        if lume.on:
            ns = 1      # (Lume: one path per pixel per pass, the passes added up)
        u = (Uniforms().m4(camstate.inv_view_proj).m4(w2l)
             .v4(pw, ph, ns, shutter)
             .v4(*fit, float(comp.lens_k1), pix)
             .v3(fwd_w, near)
             .v4(1.0 if floor_on else 0.0, float(ground_y), 1.0 if stage_on else 0.0, 1.0 if footage else 0.0)
             .v4(*(np.asarray(fl.colour, float) * tint), fl.roughness)
             .v4(fl.pattern, 1.0 if (footage and d['ground']) else 0.0, math.radians(fire.yaw), HORIZON_FADE)
             .v4(*comp.bg, 1.0 if comp.bg_checker else 0.0)
             .v4(*sky, max(8, ph / 60))
             .v4(*light.sun, SUN_SHARPNESS)
             .v3(sd_l, ao_reach)
             .v4(*light.fire, light.fire_shadows)
             .v4(*org, hcell)
             .v4(*ld, light.lamps)
             .v4(*lsc, 1.0 if vol_on else 0.0)
             .v4(*comp.atmos_colour, 3.912 / vis if vis > 0.0 else 0.0)
             .v4(transform, comp.plate_gain, 1.0 if matte_on else 0.0, 1.0 if depth_on else 0.0)
             .v4(DEPTH_KINDS.get(comp.depth_kind, 0), comp.depth_scale, frame, light.lamp_gain)
             .v4(1.0 if env is not None else 0.0, math.radians(light.env_rotation), light.env_strength, 0.0)
             .v4(r._shaper_lo, r._shaper_hi, 1.0 if r.lut_plate_log else 0.0, r.lut_size)
             .v4(*centre, radius))
        pa = self.piece_arrays(scene, pieces, shutter, ropes, ground_y, bolts) if (pieces or ropes or bolts) else None
        self.has_pieces = pa is not None
        if pa is not None:
            P, PL, glo, gcell, gdims, GC, GL = pa
            u.v4(*glo, gcell).v4(*gdims, len(P))
            nb = int(np.sum(P[:, 3, 3] == LIGHTNING_ROW))       # (lightning's segments come last)
            bufs = [self._buffer('pieces', P), self._buffer('planes', PL), self._buffer('cells', GC), self._buffer('list', GL)]
        else:
            nb = 0
            u.v4().v4()
            bufs = [self._buffer('pieces', np.zeros(24, np.float32)), self._buffer('planes', np.zeros(4, np.float32)),
                    self._buffer('cells', np.zeros(2, np.uint32)), self._buffer('list', np.zeros(1, np.uint32))]
        if surf is not None:
            phi, look0, _look1, look2 = surf
            last = matter.origin + (np.asarray(matter.dims, float) - 1.0) * matter.dx     # (its grid's last node)
            u.v4(*matter.origin, matter.dx).v4(*matter.dims, 1.0).v4(*matter.origin).v4(*last)
            mtex = [look0, phi]
        else:
            u.v4().v4(1.0, 1.0, 1.0, 0.0).v4().v4()
            mtex = [self._no_matter, self._no_matter]
            look2 = self._no_matter
        # lightning's glow: reaching about ten times its core's radius (at least 5 cm)
        reach = max(10.0 * max((float(c) for _p, c, _g in (bolts or [])), default=0.0), 0.05)
        u.v4(len(P) - nb if pa is not None else 0, nb, reach, BOLT_GLOW)
        # the grass on the ground (strands.py ground_map): (texture, corner, size)
        if grass is not None:
            u.v4(*grass[1], *grass[2])
        else:
            u.v4()
        # what the fire has done to the burnable floor and objects (Renderer SurfaceInputs: the frame's burn state)
        floor_burn = burns.burn if (burns is not None and vol is not None) else None
        obj_burn = burns.burn_obj if (burns is not None and burns.slots is not None) else None
        if floor_burn is not None:
            grid = tuple(burns.grid)
            u.v4(vol.origin[0], vol.origin[2], burns.cell, 1.0).v4(grid[0], grid[2], 1.0 if obj_burn is not None else 0.0)
        else:
            u.v4().v4(0.0, 0.0, 1.0 if obj_burn is not None else 0.0)
        # hot matter's glow: a blackbody, shown as the fire shows one (renderer.pack_look): as bright as a thick flame
        # at the same temperature (liquid_render.lava_table: 700 K on in 100 K steps)
        glow = surf is not None and getattr(matter, 'thermal', False)
        scale, dr = matter_glow_scale(scene.look(frame)) if glow else (0.0, 1.0)
        table = [(r_, g_, b_, w_ * dr) for r_, g_, b_, w_ in lava_table(GLOW_T0, GLOW_DT, 16)]
        u.v4(1.0 if glow else 0.0, scale, GLOW_T0, 1.0 / GLOW_DT)
        for row in table:
            u.v4(*row)
        # (and the light it casts round it: its glowing surface as point lights)
        b.clear_buffer(self._ml, 0, 16)
        if glow:
            b.clear_buffer(self._mlc)
            gu = (Uniforms().v4(*matter.dims, GLOW_BLOCK).v4(*matter.origin, matter.dx)
                  .v4(1.0, scale, GLOW_T0, 1.0 / GLOW_DT).v4(MATTER_LIGHTS))
            for row in table:
                gu.v4(*row)
            res = [phi, look2, self._ml, self._mlc]
            b.run(self.k_mglow, res, gu, groups=tuple(-(-int(d) // (4 * GLOW_BLOCK)) for d in matter.dims))
            b.run(self.k_mglow_finish, res, gu, groups=(1, 1, 1))
        self.has_matter = surf is not None
        pack_colliders(u, cols, meshes)
        for i in range(MAX_COLLIDERS):
            if i < len(rows):
                dr, colour, rough, metal, clear, pattern, inside, ior = rows[i]
                u.v4(*colour, rough).v4(metal, clear, pattern, dr).v4(*inside, ior)
            else:
                u.v4().v4().v4()
        for colour, rough, metal in ROPE_LOOKS:
            u.v4(*colour, rough).v4(metal, 0.0, 0.0, float(CG)).v4(*colour, 1.5)
        u.v4(0.0, 0.0, 0.0, 1.0).v4(0.0, 1.0, 0.0, float(CG)).v4(0.0, 0.0, 0.0, 1.0)   # (lightning: lets the light by)
        black = r._black
        lume_bufs = self._lume_off
        if lume.on:
            if self.lume is None:
                self.lume = LU.Lume(g)
            L = self.lume
            L.ensure(pw, ph)
            L.environment(scene.data['lighting'].get('environment', '') if env is not None else '')
            lume_bufs = [L.acc, L.aov, L.env_buffer()]
            # (the picture the passes so far are of: an edit is drawn live first, which starts them afresh)
            key = (int(frame), pw, ph, bytes(np.asarray(camstate.inv_view_proj, np.float32).tobytes()), bool(footage),
                   float(shutter))
            first, count = L.plan(lume, key, final, samples)
        elif self.lume is not None:
            self.lume.pending = False
        res = [meshes.atlas if meshes is not None else r._empty_r32,
               r.L0 if vol_on else r._empty, r.L1 if vol_on else r._empty, r.E if vol_on else r._empty,
               r.LT if (vol_on and light.lamps and r.LT is not None) else r._empty,
               g.linear, g.repeat,
               r.lights if fire_on else self._zero, r.light_count if fire_on else self._zero, r._lamp_buf,
               r.plate if (footage and r.plate is not None) else black,
               r.hold if (footage and r.hold is not None and any(r.hold_on)) else black,
               env if env is not None else black,
               r.lut_plate if (footage and r.lut_plate is not None) else r._lut_none,
               self.tex, *bufs, self.hold, *mtex, grass[0] if grass is not None else black,
               floor_burn if floor_burn is not None else r._empty, obj_burn if obj_burn is not None else r._empty,
               burns.slots if obj_burn is not None else r._no_slots, look2, self._ml, *lume_bufs]
        if not lume.on:
            b.run(self.k, res, u.v4().v4(), (pw, ph, 1))
            return self.tex
        # Lume: its passes, each a new path per pixel added to the ones before (lume.wgsl), then the denoiser
        base = list(u.data)
        # (Clamp bright paths: a bounce's light capped at that many times the sky's brightness; 0: none)
        cap = float(lume.clamp) * max(float(np.dot(np.asarray(sky, float), (0.2126, 0.7152, 0.0722))), 1e-4) if lume.clamp > 0 else 0.0
        ew, eh = L.env_dims if env is not None else (0, 0)
        for i in range(count):
            up = Uniforms()
            up.data = base + [1.0, float(first + i), float(lume.bounces), cap, float(ew), float(eh), 0.0, 0.0]
            b.run(self.k, res, up, (pw, ph, 1))
            if final and (i + 1) % LU.FINAL_SUBMIT == 0 and i + 1 < count:
                b.submit(restart=True)
        if count > 0 or final:
            L.finish(b, self.tex, first + count, lume.denoise)
        else:
            L.finish(b, self.tex, L.passes, False)
        return self.tex
