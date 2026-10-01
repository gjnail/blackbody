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
from .gpu import Uniforms
from .solver import MAX_COLLIDERS, _mesh_ref, pack_colliders
from ..scene.materials import FLOORS, PATTERNS, material

NOT_DRAWN, IN_FOOTAGE, CG = 0, 1, 2
SUN_SHARPNESS = 24.0     # the key light's soft shadows: 1 / tan of its angular radius (about 2.4 degrees)
HORIZON_FADE = 150.0     # m: far off, the floor fades into the sky at the horizon over this distance
INPUT_LINEAR = 2         # composite.wgsl input transform: scene-linear


def drawn(c, footage):
    """How object c (a collider) is drawn: NOT_DRAWN, IN_FOOTAGE or CG."""
    if not c.get('holdout', True):
        return NOT_DRAWN
    look = c.get('look', 'auto')
    if look == 'cg':
        return CG
    if look == 'footage':
        return IN_FOOTAGE
    return IN_FOOTAGE if footage and not (c.get('dynamic') or c.get('floating')) else CG


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
        own = c.get('dynamic') or c.get('floating') or c.get('look') == 'cg'
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


class Stage:
    def __init__(self, gpu):
        self.gpu = gpu
        self.k = gpu.kernel('stage.wgsl', ['utex3d', 'tex3d', 'tex3d', 'tex3d', 'tex3d', 'smp', 'smp', 'rbuf', 'rbuf',
                                           'rbuf', 'tex2d', 'tex2d', 'tex2d', 'tex3d', 'st2d:rgba16float:w'],
                            workgroup=(8, 8, 1))
        self.tex = None
        self._zero = gpu.buffer(64, 'stage-no-lights')
        self._env = None
        self._env_key = None
        self.env_sky = None     # the HDRI's average colour (the sky's light), when it has one

    def _ensure(self, w, h):
        if self.tex is not None and self.tex.size[:2] == (w, h):
            return
        if self.tex is not None:
            self.tex.destroy()
        self.tex = self.gpu.texture2d(w, h, 'rgba16float', 'stage')

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
    def _bound(cols, rows, meshes, shutter):
        """A sphere round every object in the shot (fire-local centre, radius), moving ones over the shutter."""
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
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
             samples=1, shutter=0.0, footage=False, vol=None, ground_y=0.0, frame=0, objects=True, floor=True):
        """Draw the stage into self.tex (size: the plate's, footage or output) and return it.
        r: the Renderer (its footage plate and holdouts, light volume, fire lights and lamp buffer);
        colliders: the objects as the solver has them (ColliderGPU, moving ones where they are this frame);
        vol: the fire's volume when this frame's light volume is lit (Renderer.light), else None;
        ground_y: the ground's height (fire-local m: the bottom of the simulation box);
        objects: draw the CG objects (else they only shade the floor: a liquid scene's grey stand-ins);
        floor: draw the floor (not under bottomless water)."""
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
        centre, radius = self._bound(cols, rows, meshes, shutter)
        # the footage under it
        matte_on, depth_on = r.hold_on if (footage and r.hold is not None) else (False, False)
        from .renderer import DEPTH_KINDS, INPUT_TRANSFORMS
        transform = INPUT_TRANSFORMS.get(comp.plate_transform, 0)
        vis = float(comp.visibility)
        ns = 1 if samples <= 1 else (8 if (shutter > 0.0 and samples >= 8) else 4)
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
        pack_colliders(u, cols, meshes)
        for i in range(MAX_COLLIDERS):
            if i < len(rows):
                dr, colour, rough, metal, clear, pattern, inside, ior = rows[i]
                u.v4(*colour, rough).v4(metal, clear, pattern, dr).v4(*inside, ior)
            else:
                u.v4().v4().v4()
        black = r._black
        b.run(self.k, [meshes.atlas if meshes is not None else r._empty_r32,
                       r.L0 if vol_on else r._empty, r.L1 if vol_on else r._empty, r.E if vol_on else r._empty,
                       r.LT if (vol_on and light.lamps and r.LT is not None) else r._empty,
                       g.linear, g.repeat,
                       r.lights if fire_on else self._zero, r.light_count if fire_on else self._zero, r._lamp_buf,
                       r.plate if (footage and r.plate is not None) else black,
                       r.hold if (footage and r.hold is not None and any(r.hold_on)) else black,
                       env if env is not None else black,
                       r.lut_plate if (footage and r.lut_plate is not None) else r._lut_none,
                       self.tex], u, (pw, ph, 1))
        return self.tex
