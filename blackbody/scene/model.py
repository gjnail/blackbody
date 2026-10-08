"""The scene: every setting of a shot, how it animates, and how it maps onto the engine."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from ..engine import SIM_VERSION
from ..engine import camera as cam
from ..engine.lut import flame_ev
from ..engine.embers import EmberParams
from ..engine.renderer import CompParams, LookParams
from ..engine.solver import MAX_COLLIDERS, MAX_EMITTERS, VOLUME_MODES, ColliderGPU, EmitterGPU, Solver, SolverParams, SpreadParams
from ..engine.mesh import is_numbered, mesh_deforms, sequence_files, split_source
from .anim import Curve
from .params import (COLLIDER_PARAMS, EMITTER_PARAMS, LOOK_KEYS, SECTIONS, SIM_SECTIONS, coerce, collider_defaults, defaults,
                     emitter_defaults, fabric_defaults, light_defaults,
                     matter_defaults, param, shot_defaults, strand_defaults)

FILE_VERSION = 1
# settings that only say where frames are kept: they do not change the simulation itself
CACHE_ONLY = {('domain', 'disk_cache'), ('domain', 'cache_dir'), ('domain', 'checkpoint_every')}
PROJECT_EXT = '.bbfire'


def _smoothstep(a, b, x):
    if b <= a:
        return 1.0 if x >= a else 0.0
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def _wander(t, seed):
    """Smooth pseudo-random signal in about [-1, 1] (sum of incommensurate sines)."""
    s = seed * 1.618
    return (0.5 * math.sin(t * 1.0 + s) + 0.3 * math.sin(t * 2.31 + 1.7 * s) + 0.2 * math.sin(t * 5.17 + 2.9 * s))


def base_width(e, pos, size, end):
    """Width (m) of the circle with the same area as an emitter's footprint on the ground; None for a mesh."""
    sx, sz = abs(size[0]), abs(size[2])
    shape = e['shape']
    if shape == 'mesh':
        return None
    if shape == 'box':
        area = 4.0 * sx * sz
    elif shape == 'capsule':
        run = math.hypot(end[0] - pos[0], end[2] - pos[2])
        area = 2.0 * sx * run + math.pi * sx * sx
    elif shape == 'ring':
        area = math.pi * (sx + abs(size[1])) ** 2
    else:  # sphere, cylinder, cone
        area = math.pi * sx * sz
    return 2.0 * math.sqrt(area / math.pi)


def puff_wave(t, width, seed):
    """(wave around 0, strength around 1) of the puffing of a fire base `width` m wide at time t (s): real pool
    fires puff about 1.5 / sqrt(D) times a second (Cetegen & Ahmed 1993). Filmed campfires show it as a broad
    peak, not a tone: the rate and strength drift from puff to puff, bulges swell then pinch off faster, and a
    faster flutter rides on top. Measured against footage of a real campfire (tools/fire_check.py)."""
    f = min(1.5 / math.sqrt(max(width, 0.02)), 8.0)
    phase = 2.0 * math.pi * f * t + 2.5 * _wander(t * 0.35 * f, seed)
    flutter = 0.3 * math.sin(2.0 * math.pi * 2.7 * f * t + 1.5 * _wander(t * f, seed + 7.0))
    wave = math.sin(phase) + 0.3 * math.sin(2.0 * phase + 0.6) + flutter
    return wave, 0.6 + 0.4 * _wander(t * 0.25 * f, seed + 3.0)


PUFF_BLEND = 1.2          # Puffing at 1: how firmly the gas over the base is steered into each surge (per frame, at its peak)
PUFF_HEAT = 3.0           # and how much hotter it gets at the peak of a surge
AUTO_PX_PER_CELL = 1.5    # Resolution from the shot: screen pixels per cell of the final grid
AUTO_MAX_UPRES = 3        # the detail upres it goes up to
AUTO_MAX_BYTES = 2.0e9    # within this much GPU memory for the grids (the simulation plus the upres grid)


def _positive(v):
    """A setting is in use if it is above zero, or if any of its keys is."""
    if isinstance(v, Curve):
        return any(float(k[1]) > 0 for k in v.keys)
    return float(v) > 0


def _to_json_value(v):
    if isinstance(v, Curve):
        return v.to_json()
    if isinstance(v, tuple):
        return list(v)
    return v


def _from_json_value(p, v):
    if isinstance(v, dict) and 'keys' in v:
        return Curve.from_json(v)
    return coerce(p, v)


BOLT_RADIANCE = 50.0   # lightning's channel at the peak of a flash (scene-linear: white is 1)


def quat_mul(a, b):
    """Quaternion product a b (x, y, z, w): b's turn, then a's."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def turn_quat(yaw, pitch=0.0, roll=0.0):
    """An object's orientation (x, y, z, w) from its Rotation about the vertical, then its Tilt about its own sideways
    axis (x) and its Roll about its own front-to-back axis (z), in radians: R = Ry(yaw) Rx(pitch) Rz(roll)."""
    q = (0.0, math.sin(0.5 * yaw), 0.0, math.cos(0.5 * yaw))
    q = quat_mul(q, (math.sin(0.5 * pitch), 0.0, 0.0, math.cos(0.5 * pitch)))
    return quat_mul(q, (0.0, 0.0, math.sin(0.5 * roll), math.cos(0.5 * roll)))


def quat_matrix(q):
    """The 3x3 rotation of quaternion q (x, y, z, w)."""
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_quat(m):
    """The quaternion (x, y, z, w, w >= 0) of a 3x3 rotation: quat_matrix undone."""
    m = np.asarray(m, float)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0.0:
        k = 2.0 * math.sqrt(1.0 + t)
        q = ((m[2, 1] - m[1, 2]) / k, (m[0, 2] - m[2, 0]) / k, (m[1, 0] - m[0, 1]) / k, 0.25 * k)
    else:   # (near a half turn: from the largest diagonal term, which keeps it accurate)
        i = int(np.argmax(np.diag(m)))
        j, l = (i + 1) % 3, (i + 2) % 3
        k = 2.0 * math.sqrt(max(1.0 + m[i, i] - m[j, j] - m[l, l], 1e-12))
        v = [0.0, 0.0, 0.0]
        v[i], v[j], v[l] = 0.25 * k, (m[j, i] + m[i, j]) / k, (m[l, i] + m[i, l]) / k
        q = (v[0], v[1], v[2], (m[l, j] - m[j, l]) / k)
    q = np.asarray(q) / max(float(np.linalg.norm(q)), 1e-12)
    return tuple(float(x) for x in (q if q[3] >= 0.0 else -q))


def turn_matrix(yaw, pitch=0.0, roll=0.0):
    """turn_quat as a 3x3 rotation (own frame -> fire-local), angles in degrees."""
    return quat_matrix(turn_quat(math.radians(yaw), math.radians(pitch), math.radians(roll)))


class Scene:
    def __init__(self):
        self.name = 'Untitled'
        self.data = {s: defaults(s) for s in SECTIONS}
        self.emitters = [dict(emitter_defaults(), name='Fuel bed')]
        self.colliders = []
        self.lights = []        # lamps in the set (they light the smoke; look only, not the simulation)
        self.fabrics = []       # cloth: curtains, flags, sheets (engine/cloth.py)
        self.matter = []        # sand, snow, mud, jelly and clay (engine/matter.py)
        self.strands = []       # grass and plants (engine/strands.py)
        self.shots = []         # guns firing: bullets and what they hit (engine/ballistics.py)
        self.footage = None     # {'path', 'fps', 'offset'}
        self.track = None       # {'points': {frame: [x, y]}}
        self.roto = []          # roto shapes over the footage (scene/roto.py): a holdout drawn in the app
        # the camera matched to the footage from the ground (scene/groundmatch.py): {'corners': 4 x [x, y] across and
        # down the frame (0..1), 'scale_by': 'height' | 'side', 'height': m, 'side': m, 'lens': 'picture' | 'known',
        # 'frame': the frame it was lined up on}. With it, the camera is the footage's, the same in every layer.
        self.ground = None
        # More effects in the same shot, each its own simulation (a Scene), drawn back to front with this
        # scene's own effect at base_index among them. They share the shot (SHARED_COMPOSITE, the render
        # settings, the footage, the roto and the track's points; see sync_layers).
        self.layers = []
        self.base_index = 0
        self.uid = 'base'
        self.enabled = True
        # Objects attached to others ({'child': [kind, name], 'parent': [kind, name], 'offset': [x, y, z]}): a child
        # goes where its parent goes, at its offset, whenever the scene changes (apply_links)
        self.links = []
        self.preset = None
        self.notes = ''
        self.path = None

    # -- addressing ------------------------------------------------------------------------------
    # A path is ('section', 'key') or ('emitter' / 'collider' / 'light' / 'fabric' / 'matter', index, 'key').

    def _slot(self, path):
        if path[0] == 'emitter':
            return self.emitters[path[1]], path[2]
        if path[0] == 'collider':
            return self.colliders[path[1]], path[2]
        if path[0] == 'light':
            return self.lights[path[1]], path[2]
        if path[0] == 'fabric':
            return self.fabrics[path[1]], path[2]
        if path[0] == 'matter':
            return self.matter[path[1]], path[2]
        if path[0] == 'strands':
            return self.strands[path[1]], path[2]
        if path[0] == 'shot':
            return self.shots[path[1]], path[2]
        return self.data[path[0]], path[1]

    @staticmethod
    def spec(path):
        return param(path[0], path[-1])

    def raw(self, path):
        d, k = self._slot(path)
        return d[k]

    def get(self, path, frame=None):
        v = self.raw(path)
        if isinstance(v, Curve):
            return v.eval(frame if frame is not None else self.data['render']['start'])
        return v

    def set(self, path, value, frame=None):
        """Set a value. If the parameter is animated and a frame is given, set a key instead."""
        d, k = self._slot(path)
        p = self.spec(path)
        value = coerce(p, value)
        cur = d[k]
        if isinstance(cur, Curve):
            if frame is None:
                d[k] = value
            else:
                cur.set(frame, value)
        else:
            d[k] = value

    def curve(self, path):
        v = self.raw(path)
        return v if isinstance(v, Curve) else None

    def set_key(self, path, frame, value=None):
        d, k = self._slot(path)
        p = self.spec(path)
        cur = d[k]
        val = coerce(p, value if value is not None else self.get(path, frame))
        if not isinstance(cur, Curve):
            d[k] = Curve([[float(frame), val, 'smooth']])
        else:
            cur.set(frame, val)

    def remove_key(self, path, frame):
        d, k = self._slot(path)
        cur = d[k]
        if isinstance(cur, Curve):
            cur.remove(frame)
            if not cur.keys:
                d[k] = coerce(self.spec(path), self.get(path, frame) if cur.keys else self.spec(path).default)

    def clear_anim(self, path, frame=None):
        d, k = self._slot(path)
        cur = d[k]
        if isinstance(cur, Curve):
            d[k] = coerce(self.spec(path), cur.eval(frame if frame is not None else 0))

    def key_frames(self):
        """Every keyed frame in the scene (for the timeline)."""
        out = set()
        for s, vals in self.data.items():
            for v in vals.values():
                if isinstance(v, Curve):
                    out.update(v.frames())
        for group in (self.emitters, self.colliders, self.lights, self.fabrics, getattr(self, 'matter', []),
                      getattr(self, 'strands', []), getattr(self, 'shots', [])):
            for d in group:
                for v in d.values():
                    if isinstance(v, Curve):
                        out.update(v.frames())
        for r in getattr(self, 'roto', None) or []:
            out.update(float(f) for f in r.get('keys', {}))
        return sorted(out)

    # -- time ------------------------------------------------------------------------------------

    @property
    def fps(self):
        return float(self.data['render']['fps'])

    @property
    def start(self):
        return int(self.data['render']['start'])

    @property
    def end(self):
        return int(self.data['render']['end'])

    def seconds(self, frame):
        """Shot time of a frame, in seconds from the first frame."""
        return (frame - self.start) / self.fps

    # -- emitters / colliders ------------------------------------------------------------------

    def add_emitter(self, **kw):
        e = emitter_defaults()
        e['name'] = f'Emitter {len(self.emitters) + 1}'
        for k, v in kw.items():
            e[k] = coerce(param('emitter', k), v)
        self.emitters.append(e)
        return len(self.emitters) - 1

    def add_collider(self, **kw):
        c = collider_defaults()
        c['name'] = f'Collider {len(self.colliders) + 1}'
        for k, v in kw.items():
            c[k] = coerce(param('collider', k), v)
        self.colliders.append(c)
        return len(self.colliders) - 1

    def add_light(self, **kw):
        d = light_defaults()
        d['name'] = f'Light {len(self.lights) + 1}'
        for k, v in kw.items():
            d[k] = coerce(param('light', k), v)
        self.lights.append(d)
        return len(self.lights) - 1

    def add_fabric(self, **kw):
        d = fabric_defaults()
        d['name'] = f'Fabric {len(self.fabrics) + 1}'
        for k, v in kw.items():
            d[k] = coerce(param('fabric', k), v)
        self.fabrics.append(d)
        return len(self.fabrics) - 1

    def add_matter(self, **kw):
        d = matter_defaults()
        for k, v in kw.items():
            d[k] = coerce(param('matter', k), v)
        if 'name' not in kw:
            from .params import MATTER_MATERIALS
            label = dict(MATTER_MATERIALS).get(d['material'], 'Matter')
            names = {m['name'] for m in self.matter}
            d['name'] = label if label not in names else next(f'{label} {n}' for n in range(2, 10000) if f'{label} {n}' not in names)
        self.matter.append(d)
        return len(self.matter) - 1

    def add_strands(self, **kw):
        """Add a patch of grass (engine/strands.py). Its blades grow as tall as its kind's unless Size says."""
        from .params import STRAND_KINDS
        from ..engine.strands import KINDS
        d = strand_defaults()
        for k, v in kw.items():
            d[k] = coerce(param('strands', k), v)
        if 'size' not in kw:
            s = d['size']
            d['size'] = (s[0], KINDS[d['kind']].height, s[2])
        if 'name' not in kw:
            label = dict(STRAND_KINDS).get(d['kind'], 'Grass')
            names = {m['name'] for m in self.strands}
            d['name'] = label if label not in names else next(f'{label} {n}' for n in range(2, 10000) if f'{label} {n}' not in names)
        self.strands.append(d)
        return len(self.strands) - 1

    def add_shot(self, **kw):
        """Add a gun firing (engine/ballistics.py): from its muzzle (Position) toward what it is aimed at (Aim)."""
        from .params import SHOT_ROUNDS
        d = shot_defaults()
        for k, v in kw.items():
            d[k] = coerce(param('shot', k), v)
        if 'name' not in kw:
            label = dict(SHOT_ROUNDS).get(d['round'], 'Shot').split(' (')[0]
            names = {m['name'] for m in self.shots}
            d['name'] = label if label not in names else next(f'{label} {n}' for n in range(2, 10000) if f'{label} {n}' not in names)
        self.shots.append(d)
        return len(self.shots) - 1

    def strand_specs(self):
        """The enabled patches of grass, as engine/strands.py builds them (at the first frame)."""
        from ..engine.strands import StrandSpec
        out = []
        for i, d in enumerate(getattr(self, 'strands', None) or []):
            if not d['enabled']:
                continue
            g = lambda k: self.get(('strands', i, k), self.start)
            colour = None
            if d.get('own_colour'):
                c = np.asarray(d['colour'], float)
                colour = tuple(float(x) for x in c)
            out.append(StrandSpec(kind=d['kind'], shape=d['shape'], pos=tuple(float(x) for x in g('position')),
                                  size=tuple(abs(float(x)) for x in g('size')), yaw=math.radians(float(g('yaw'))),
                                  thickness=float(d['thickness']), dryness=float(d['dryness']), burns=bool(d['burns']),
                                  on_objects=d['grows_on'] == 'everything', stiffness=float(d['stiffness']),
                                  width=float(d['blade_width']), colour=colour, seed=int(d['seed'])))
        return out

    def blasts(self):
        """The explosive charges (an emitter's Blast): [(the scene frame it goes off at (when it ignites), where (fire-local
        m), kg of TNT)]. One that ignites before the pre-roll never goes off."""
        out = []
        first = self.start - self.data['domain']['preroll'] * self.fps
        for i, e in enumerate(self.emitters):
            w = float(e.get('blast', 0.0) or 0.0)
            if not e['enabled'] or w <= 0.0:
                continue
            f = self.start + float(e['start']) * self.fps
            if f < first:
                continue
            out.append((f, tuple(float(x) for x in self.get(('emitter', i, 'position'), f)), w))
        return out

    def matter_specs(self):
        """The enabled sand, snow, mud, jelly and clay, as sources for engine/matter.py (where they start, in the box's
        frame, and what they are)."""
        from ..engine.matter import MatterSpec
        out = []
        for i, d in enumerate(getattr(self, 'matter', None) or []):
            if not d['enabled']:
                continue
            g = lambda k: self.get(('matter', i, k), self.start)
            colour = None
            if d.get('own_colour'):
                c = np.asarray(d['colour'], float)
                colour = tuple(float(x) for x in np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4))
            out.append(MatterSpec(material=d['material'], shape=d['shape'], pos=tuple(float(x) for x in g('position')),
                                  size=tuple(abs(float(x)) for x in g('size')), yaw=math.radians(float(g('yaw'))),
                                  velocity=tuple(float(x) for x in d['velocity']), release=float(d['release']),
                                  pour=bool(d['pours']), rate=float(d['rate']) / 1000.0, start=float(d['pour_start']),
                                  stop=float(d['pour_stop']), colour=colour, stiffness=float(d['stiffness']), seed=int(d['seed']),
                                  temperature=self._matter_kelvin(d),
                                  mesh=self.mesh_path(d.get('mesh', '')) if d['shape'] == 'mesh' else '',
                                  name=str(d.get('name', ''))))
        return out

    @staticmethod
    def _matter_kelvin(d):
        """A body of matter's temperature as it starts (K): its own, but a melt at least a little past its melting point
        (as it is poured: 12% hotter than it, so it runs before it sets)."""
        from ..engine.matter import material
        T = float(d.get('temperature', 20.0)) + 273.15
        m = material(d['material'])
        if m.freeze and m.melts_at > 0.0:
            T = max(T, m.melts_at * 1.12)
        return T

    def fabric_specs(self):
        """What the enabled fabrics are made of and how they are built (engine/cloth.py)."""
        from ..engine.cloth import MAX_FABRICS, FabricSpec
        out = []
        for d in self.fabrics:
            if not d['enabled'] or (d['shape'] == 'mesh' and not d['mesh']):
                continue
            out.append(FabricSpec(
                shape=d['shape'], mesh=self.mesh_path(d['mesh']) if d['shape'] == 'mesh' else '',
                width=float(d['width']), height=float(d['height']), fullness=float(d.get('fullness', 1.0)),
                detail=int(d['detail']), orientation=d['orientation'],
                pins=d['pins'], material=d['material'], stiffness=float(d['stiffness']), bend=float(d['bend']),
                weight=float(d['weight']), burnable=bool(d['burnable']), flammability=float(d['flammability']),
                wetness=float(d.get('wetness', 0.0)),
                colour=tuple(float(x) for x in d['colour']), self_collide=bool(d['self_collide']),
                tears=bool(d.get('tears', False)), tear_strength=float(d.get('tear_strength', 1.0))))
        return out[:MAX_FABRICS]

    def fabrics_at(self, frame, moved=None):
        """Where the enabled fabrics' pins are at `frame` (fractional frames allowed). moved: fabric index ->
        {position, yaw} for fabrics pinned to a falling or floating object (engine/solids.py attached)."""
        from ..engine.cloth import MAX_FABRICS, FabricPlace
        out = []
        t = self.seconds(frame)
        moved = moved or {}
        for i, d in enumerate(self.fabrics):
            if not d['enabled'] or (d['shape'] == 'mesh' and not d['mesh']):
                continue
            mv = moved.get(i)
            g = lambda k: mv[k] if (mv is not None and k in mv) else self.get(('fabric', i, k), frame)
            rel = float(d['release'])
            out.append(FabricPlace(pos=tuple(float(x) for x in g('position')), yaw=math.radians(g('yaw')),
                                   scale=tuple(float(x) for x in d['scale']) if d['shape'] == 'mesh' else (1.0, 1.0, 1.0),
                                   released=d['pins'] == 'none' or (rel >= 0.0 and t >= rel)))
        return out[:MAX_FABRICS]

    def lamps(self, frame, moved=None):
        """The enabled lights at `frame` for the renderer: fire-local position (m), unit aim, linear colour
        times intensity (lux at 1 m), radius (m), kind, cone cosines and whether smoke shadows them.
        moved: light index -> {position, direction} for lights carried by a falling or floating object."""
        out = []
        moved = moved or {}
        for i, d in enumerate(self.lights):
            if not d['enabled']:
                continue
            mv = moved.get(i)
            g = lambda k: mv[k] if (mv is not None and k in mv) else self.get(('light', i, k), frame)
            col = np.asarray(d['colour'], float)
            if d['temperature'] > 0:
                from ..engine.lut import blackbody_lut
                t = float(d['temperature'])
                bb = blackbody_lut(8, max(400.0, t - 1.0), t + 1.0)[4, :3]
                col = col * bb / max(float(bb.max()), 1e-9)
            aim = np.asarray(g('direction'), float)
            aim = aim / n if (n := float(np.linalg.norm(aim))) > 1e-9 else np.array([0.0, -1.0, 0.0])
            half = math.radians(max(0.5, float(d['cone'])))
            soft = min(max(float(d['softness']), 0.0), 1.0)
            if d['kind'] == 'lightning':
                out += self._lightning_lamps(i, d, frame, col, g)
                continue
            kind = d['kind']
            co, ci = math.cos(half), math.cos(half * (1.0 - soft))
            prof = self.light_profile(d) if kind in ('point', 'spot') else None
            peak = max(float(g('intensity')), 0.0)
            lumens = max(float(g('lumens')), 0.0) if 'lumens' in d else 0.0
            if prof is not None:
                kind = 'point'                          # (the profile is its shape: no cone as well)
            if lumens > 0.0 or (prof is not None and d.get('profile_brightness', True)):
                # in photometric units: the colour carries no brightness of its own (its luminance 1)
                from ..io.ies import flux_per_peak
                col = col / max(float(col @ np.array([0.2126, 0.7152, 0.0722])), 1e-9)
                peak = lumens / flux_per_peak(kind, co, ci, prof) if lumens > 0.0 else prof.peak
            out.append(dict(kind=kind, position=tuple(float(x) for x in g('position')), direction=tuple(float(x) for x in aim),
                            power=tuple(float(x) for x in col * peak),
                            radius=float(d['radius']), cos_outer=co, cos_inner=ci,
                            shadows=bool(d['shadows']), in_footage=bool(d.get('in_footage', True)),
                            width=float(d.get('width', 1.0)), height=float(d.get('height', 1.0)), spin=float(d.get('spin', 0.0)),
                            profile=prof))
        return out

    def light_profile(self, d):
        """A light's profile (io/ies.py Profile) from its IES file, or None (none, or it will not read)."""
        if not d.get('profile'):
            return None
        from ..io import ies
        try:
            return ies.load(self.mesh_path(d['profile']))
        except (OSError, ValueError, IndexError):
            return None

    LIGHTNING_LAMPS = 4

    def _lightning(self, i, d, frame):
        """A lightning light at `frame`: (its channels (engine/lightning.py bolt), how bright the flash is over the frame,
        how bright its first flash is), or None while it is dark."""
        from ..engine import lightning as LG
        b, first = LG.brightness(self.seconds(frame), float(d['strike_at']), int(d['strokes']), int(d['bolt_seed']),
                                 exposure=1.0 / self.fps)
        if b <= 1e-4:
            return None
        top = tuple(float(x) for x in self.get(('light', i, 'position'), frame))
        end = tuple(float(x) for x in self.get(('light', i, 'end'), frame))
        return LG.cached_bolt(top, end, int(d['bolt_seed']), float(d['branching'])), b, first

    def _lightning_lamps(self, i, d, frame, col, g):
        """A lightning light as point lamps along its main channel, as bright as its flash over the frame."""
        got = self._lightning(i, d, frame)
        if got is None:
            return []
        channels, b, _first = got
        main = channels[0][0]
        seg = np.linalg.norm(np.diff(main, axis=0), axis=1)
        along = np.concatenate([[0.0], np.cumsum(seg)])
        n = self.LIGHTNING_LAMPS
        power = col * max(float(g('intensity')), 0.0) * b / n
        out = []
        for k in range(n):
            s = (k + 0.5) / n * along[-1]
            j = min(int(np.searchsorted(along, s)), len(main) - 1)
            out.append(dict(kind='point', position=tuple(float(x) for x in main[j]), direction=(0.0, -1.0, 0.0),
                            power=tuple(float(x) for x in power), radius=max(along[-1] / (2.0 * n), 0.05), cos_outer=-1.0,
                            cos_inner=-1.0, shadows=bool(d['shadows']), in_footage=False))
        return out

    def bolts(self, frame):
        """The lightning to draw at `frame`: [(points (n, 3) fire-local m, core radius (m), glow (linear rgb, the radiance of
        its core))], each channel as bright as its share of the flash over the frame (the branches in the first flash only)."""
        out = []
        for i, d in enumerate(self.lights):
            if not d['enabled'] or d['kind'] != 'lightning':
                continue
            got = self._lightning(i, d, frame)
            if got is None:
                continue
            channels, b, first = got
            col = np.asarray(d['colour'], float)
            if d['temperature'] > 0:
                from ..engine.lut import blackbody_lut
                t = float(d['temperature'])
                bb = blackbody_lut(8, max(400.0, t - 1.0), t + 1.0)[4, :3]
                col = col * bb / max(float(bb.max()), 1e-9)
            glow = col * BOLT_RADIANCE        # (how it looks; its Intensity is how much it lights the set)
            core = 0.5 * float(d['thickness'])
            for pts, thick, bright, first_only in channels:
                level = (first if first_only else b) * bright
                if level > 1e-3:
                    out.append((pts, core * thick, tuple(float(x) for x in glow * level)))
        return out

    def lightning_fires(self, frame):
        """Where lightning that sets fire strikes, as emitters: a burst of flame for a quarter of a second from the stroke."""
        from ..engine.solver import EmitterGPU
        out = []
        if self.kind not in ('fire', 'both'):
            return out
        t = self.seconds(frame)
        for i, d in enumerate(self.lights):
            if not d['enabled'] or d['kind'] != 'lightning' or not d.get('ignites'):
                continue
            t0 = float(d['strike_at'])
            if not (t0 <= t < t0 + 0.25):
                continue
            end = tuple(float(x) for x in self.get(('light', i, 'end'), frame))
            env = 1.0 - (t - t0) / 0.25
            out.append(EmitterGPU(shape='sphere', pos=end, size=(0.15, 0.15, 0.15), fuel=15.0 * env, temp=1.1 * env, radial=1.5,
                                  vel_blend=0.3 * env, noise=0.8, noise_freq=6.0, seed=float(31 * i + 7)))
        return out

    # -- engine mapping --------------------------------------------------------------------------

    def v(self, section, key, frame):
        return self.get((section, key), frame)

    def domain_size(self):
        d = self.data['domain']
        return (float(d['size_x']), float(d['size_y']), float(d['size_z']))

    def sim_layout(self, final=False):
        """Grid dims, cell size (m) and grid corner (fire-local metres)."""
        d = self.data['domain']
        scale = self.data['render']['final_scale'] if final else d['preview_scale']
        res = max(16, int(round(d['resolution'] * scale)))
        auto = self.auto_detail() if final else None
        if auto:
            res = auto[0]
        dims, h = Solver.dims_for(self.domain_size(), res)
        origin = (-dims[0] * h / 2.0, 0.0, -dims[2] * h / 2.0)
        return dims, h, origin

    def wind(self, frame, fire_yaw_deg):
        m = self.data['motion']
        t = self.seconds(frame)
        speed = self.v('motion', 'wind_speed', frame)
        if speed <= 0:
            return (0.0, 0.0, 0.0)
        g = self.v('motion', 'gust', frame)
        f = m['gust_freq'] * 2 * math.pi
        seed = self.data['domain']['seed']
        s = speed * max(0.0, 1.0 + g * 0.8 * _wander(t * f, seed))
        ang = math.radians(self.v('motion', 'wind_dir', frame) + g * 25.0 * _wander(t * f * 0.7, seed + 11))
        w = np.array([math.sin(ang) * s, 0.0, math.cos(ang) * s])
        local = cam.rot_y(math.radians(fire_yaw_deg)).T @ w
        return tuple(float(x) for x in local)

    def solver_params(self, frame):
        f = frame
        d, c, m, sp = self.data['domain'], self.data['combustion'], self.data['motion'], self.data['spread']
        fire_yaw = self.v('camera', 'fire_yaw', f)
        feat = self.features()
        return SolverParams(
            ignition=c['ignition'], burn_rate=c['burn_rate'], heat=c['heat'], soot=c['soot'],
            flame_gain=c['flame_gain'], flame_life=c['flame_life'], expansion=self.v('combustion', 'expansion', f),
            radiative=c['radiative'], cooling=self.v('combustion', 'cooling', f),
            smoke_dissipation=self.v('combustion', 'smoke_dissipation', f), fuel_dissipation=c['fuel_dissipation'],
            temp_cap=c['temp_cap'], rich=c['rich'], expansion_cap=c['expansion_cap'], sponge=d['sponge'], sponge_strength=d['sponge_strength'],
            flame_speed=c['flame_speed'], fuel_weight=c['fuel_weight'], soot_stain=c['soot_stain'],
            oxygen=feat['oxygen'], air_use=c['air_use'], air_mixing=c['air_mixing'], vapour=feat['vapour'], steam_yield=c['steam_yield'],
            water_yield=c['water_yield'], vapour_dissipation=self.v('combustion', 'vapour_dissipation', f),
            boil_temp=self.boil_temp(), water_douse=c['water_douse'], latent_heat=c['latent_heat'],
            steam_expansion=c.get('steam_expansion', 0.5),
            thermal_expansion=c['thermal_expansion'], ambient_k=self.data['shading']['ambient_k'],
            upres_turbulence=self.data['render']['upres_turbulence'],
            flame_k=self.data['shading']['flame_k'], humidity=self.data['shading']['humidity'],
            buoyancy=self.v('motion', 'buoyancy', f), soot_weight=m['soot_weight'], damping=m['damping'],
            vorticity=self.v('motion', 'vorticity', f), vort_outside=m['vort_outside'],
            turbulence=self.v('motion', 'turbulence', f), turb_freq=m['turb_freq'], turb_rise=m['turb_rise'],
            turb_evolve=m['turb_evolve'], turb_octave2=m['turb_octave2'],
            disturbance=self.v('motion', 'disturbance', f), disturb_block=m['disturb_block'], disturb_rate=m['disturb_rate'],
            mask_temp=m['mask_temp'], mask_flame=m['mask_flame'], mask_smoke=m['mask_smoke'],
            wind=self.wind(f, fire_yaw), wind_relax=m['wind_relax'],
            maccormack=d['maccormack'], maccormack_vel=d['maccormack_vel'], mg_cycles=d['mg_cycles'],
            seed=d['seed'], open_sides=d['open_sides'], open_top=d['open_top'], ground=d['ground'],
            spread=SpreadParams(
                enabled=bool(sp['enabled']), ground=bool(sp['ground']) and bool(d['ground']), area=(sp['area_x'], sp['area_z']),
                coverage=sp['coverage'], patch_freq=sp['patch_freq'], burn_time=sp['burn_time'], fuel=sp['fuel'],
                heat=sp['heat'], smoke=sp['smoke'], catch_temp=sp['catch_temp'], catch_time=sp['catch_time'],
                creep=sp['creep'], smoulder=sp['smoulder'], smoulder_smoke=sp['smoulder_smoke'], dry_time=sp['dry_time'],
                spotting=sp['spotting'] if self.data['embers'].get('enabled', True) else 0.0, spot_temp=sp['spot_temp']))

    def boil_temp(self):
        """The boiling point of water on the temperature field's scale (0 = the air, 1 = the flame temperature)."""
        s = self.data['shading']
        return min(max((373.15 - s['ambient_k']) / max(s['flame_k'] - s['ambient_k'], 1.0), 0.0), 1.0)

    def features(self):
        """Optional simulation fields this scene needs: tracked air, water vapour, flame colourant,
        burnable surfaces. Each costs memory and time, so it is only allocated when something uses it."""
        c = self.data['combustion']
        ems = [e for e in self.emitters if e['enabled']]
        oxygen = c['air'] == 'tracked'
        vapour = (_positive(c['water_yield']) or any(_positive(e['vapour']) for e in ems)
                  or (_positive(c['steam_yield']) and any(_positive(e['douse']) for e in ems)))
        chem = any(_positive(e['color_amount']) for e in ems)
        burn = bool(self.data['spread']['enabled'])
        # liquid water sharing the box with the fire (domain kind 'both') puts it out and makes steam
        water = self.data['domain'].get('kind') == 'both'
        # wet cloth in the heat steams
        wet_cloth = any(d.get('enabled') and _positive(d.get('wetness', 0.0)) for d in self.fabrics)
        vapour = vapour or water or wet_cloth
        stain = _positive(c['soot_stain'])
        return {'oxygen': bool(oxygen), 'vapour': bool(vapour), 'chem': bool(chem), 'burn': burn,
                'aux': bool(oxygen or vapour), 'water': bool(water), 'stain': bool(stain)}

    def rate(self, path, frame):
        """How fast an animated setting changes, per second of simulation time (0 if it is not animated).
        Vectors give a tuple."""
        cur = self.curve(path)
        if cur is None:
            v = self.get(path, frame)
            return tuple(0.0 for _ in v) if isinstance(v, (tuple, list)) else 0.0
        a, b = cur.eval(frame - 0.5), cur.eval(frame + 0.5)
        k = self.fps / max(self.v('domain', 'time_scale', frame), 1e-3)
        if isinstance(a, (tuple, list)):
            return tuple((y - x) * k for x, y in zip(a, b))
        return (b - a) * k

    def mesh_path(self, p):
        """Where a mesh file really is: 'builtin:NAME' is one of Blackbody's own meshes; relative paths
        are relative to the project file; a path that has gone missing is looked for next to the project."""
        if not p:
            return ''
        f, prim = split_source(p)
        suffix = '#' + prim if prim else ''
        if f.startswith('builtin:'):
            return str(Path(__file__).resolve().parents[1] / 'assets' / 'meshes' / f[8:]) + suffix
        q = Path(f)
        base = Path(self.path).parent if self.path else Path.cwd()
        if not q.is_absolute():
            q = base / q
        found = (lambda x: bool(sequence_files(str(x)))) if is_numbered(f) else (lambda x: x.exists())
        if not found(q) and self.path:
            alt = base / Path(f).name
            if found(alt):
                q = alt
        return str(q) + suffix

    def envelope(self, e, frame):
        t = self.seconds(frame)
        start = e['start']
        env = 1.0 if start <= -99 else _smoothstep(start, start + max(e['fade_in'], 1e-3), t)
        if e['stop'] >= 0:
            env *= 1.0 - _smoothstep(e['stop'], e['stop'] + max(e['fade_out'], 1e-3), t)
        return env

    def emitter_turn(self, i, frame, carried=None):
        """Emitter i's orientation for the GPU at `frame`: (its yaw in radians, a quaternion (x, y, z, w) applied after
        it). One that takes its shape from a tipped-over collider (a link with 'shape': fire on solid letters) takes
        the collider's whole turn. carried: how far the falling or floating object it rides on has turned since the
        start (engine/solids.py attached), on top."""
        yaw, quat = math.radians(float(self.get(('emitter', i, 'yaw'), frame))), (0.0, 0.0, 0.0, 1.0)
        name = self.emitters[i]['name']
        for l in getattr(self, 'links', None) or []:
            if l.get('shape') and list(l['child']) == ['emitter', name] and l['parent'][0] == 'collider':
                pi, _ = self.find_object(*l['parent'])
                if pi is not None and self.tilted(pi):
                    f = self.start if carried is not None else frame   # (a carried one turns from where it started)
                    yaw, quat = 0.0, turn_quat(*(math.radians(float(self.get(('collider', pi, k), f)))
                                                 for k in ('yaw', 'pitch', 'roll')))
                break
        if carried is not None:
            quat = quat_mul(tuple(float(x) for x in carried), quat)
        return yaw, quat

    def emitters_gpu(self, frame, embers_only=False, substeps=6, moved=None):
        """Emitters at `frame`, which may be fractional (the solver evaluates them every substep so
        fast-moving emitters leave a continuous trail). substeps: how many steps the frame is taken in, so
        that steering the gas (puffing) is as firm per frame however many there are. moved: emitter index ->
        {position, end, velocity, turn} for emitters carried by a falling or floating object (engine/solids.py
        attached), in place of their keys: velocity is how fast the point it rides on moves, turn how far the
        object has turned since the start (a quaternion), which turns the emitter's shape and its own Velocity."""
        moved = moved or {}
        out = []
        master = self.v('combustion', 'fuel_scale', frame)
        seed = self.data['domain']['seed']
        both = self.kind == 'both'
        puffing = self.v('motion', 'puffing', frame)
        t_real = self.seconds(frame)  # the footage's seconds: puffing must match real fire whatever the time scale
        for i, e in enumerate(self.emitters):
            if not e['enabled'] or (embers_only and not e['embers']):
                continue
            if both and e.get('emits') in ('liquid', 'lava'):
                continue
            mv = moved.get(i)
            g = lambda k: mv[k] if (mv is not None and k in mv) else self.get(('emitter', i, k), frame)
            fill = e['shape'] == 'volume' and e.get('volume_mode', 'fill') == 'fill'
            if e['shape'] == 'volume' and e.get('volume_mode') == 'hold' and embers_only:
                continue
            if fill:
                # a volume that fills the box does it once, whole, in one substep, and throws no embers
                if embers_only or not self._fills_now(e, frame, substeps):
                    continue
                env = 1.0
            else:
                env = self.envelope(e, frame)
            if env <= 1e-4:
                continue
            # a moving emitter drags the air (and its embers) along with it
            motion = np.asarray(self.rate(('emitter', i, 'position'), frame), float)
            if e['shape'] == 'capsule':
                motion = 0.5 * (motion + np.asarray(self.rate(('emitter', i, 'end'), frame), float))
            carried = None
            if mv is not None:
                motion = np.asarray(mv['velocity'], float)
                carried = mv.get('turn')
            speed = float(np.linalg.norm(motion))
            inherit = float(e['inherit'])
            # its own jet (Velocity, not how fast it is carried), turned as far as the object carrying it has turned
            vel = np.asarray(self.get(('emitter', i, 'velocity'), frame), float)
            if carried is not None:
                vel = quat_matrix(carried) @ vel
            vel = vel + inherit * motion
            blend = max(float(e['vel_blend']), inherit * min(1.0, speed / 0.5))
            yaw, quat = self.emitter_turn(i, frame, carried)
            amount = g('color_amount') * env
            heat = g('temperature') * min(1.0, env * 4.0)
            if puffing > 0.0 and blend <= 0.0 and g('fuel') > 0.0:
                # puffing: the gas over the fire's base surges upward at the rate real fires puff, which rolls it
                # up into the bulges that pinch off the top of the flame. Only while it surges: between puffs the
                # gas is left to itself (steering it back down makes the flame stall)
                width = base_width(e, g('position'), g('size'), g('end'))
                if width is not None:
                    wave, strength = puff_wave(t_real, width, float(e['seed'] * 13 + seed * 7 + i))
                    vel = vel + np.array([0.0, 0.5 * math.sqrt(9.81 * width) * max(0.0, 1.0 + wave), 0.0])
                    surge = min(PUFF_BLEND * puffing * strength * max(0.0, wave), 0.95)
                    blend = 1.0 - (1.0 - surge) ** (1.0 / max(1, substeps))
                    heat *= 1.0 + PUFF_HEAT * puffing * strength * max(0.0, wave)
            out.append(EmitterGPU(
                shape=e['shape'], pos=tuple(g('position')), size=tuple(g('size')), p1=tuple(g('end')), soft=e['softness'],
                fuel=g('fuel') * env * master, temp=heat, smoke=g('smoke') * env,
                vel=tuple(float(x) for x in vel), radial=g('radial'), vel_blend=blend * env,
                noise=e['noise'], noise_freq=e['noise_freq'], noise_rise=e['noise_rise'], contrast=e['contrast'],
                seed=float(e['seed'] * 13 + seed * 7 + i), yaw=yaw, quat=quat,
                swirl=g('swirl') * env, swirl_width=e['swirl_width'], douse=g('douse') * env, vapour=g('vapour') * env,
                color=tuple(float(x) * amount for x in e['color']), thickness=e['thickness'],
                mesh=self.item_source(e), volume_mode=VOLUME_MODES.get(e.get('volume_mode', 'fill'), 0) if e['shape'] == 'volume' else 0,
                volume_vel=float(e.get('volume_velocity', 1.0)) * env if e['shape'] == 'volume' else 0.0,
                **self._mesh_time(e, frame)))
        if not embers_only and self.lights:
            out = out + self.lightning_fires(frame)     # (flames where lightning strikes)
        return out[:MAX_EMITTERS]

    def item_source(self, d):
        """The mesh library source of an emitter or collider: its mesh, or a Volume emitter's field."""
        if d.get('shape') == 'mesh' and d.get('mesh'):
            return self.mesh_path(d['mesh'])
        if d.get('shape') == 'volume' and d.get('volume'):
            from ..io.volume import field_source
            return field_source(self.mesh_path(d['volume']), bool(d.get('volume_zup')))
        return ''

    def _fills_now(self, e, frame, substeps):
        """Whether a Volume emitter that fills the box once does it in the substep at `frame`: the one
        its start time falls in (the simulation's first, if it starts before the simulation does)."""
        n = max(1, int(substeps))
        begin = self.start - int(round(self.data['domain']['preroll'] * self.fps)) - 1
        at = begin + 1e-6 if e['start'] <= -99 else max(self.start + e['start'] * self.fps, begin + 1e-6)
        return frame - 0.5 / n < at <= frame + 0.5 / n

    def _mesh_time(self, d, frame):
        """mesh_frame / mesh_fps for an emitter or collider whose mesh deforms (a numbered sequence
        or a USD prim), so the solver blends the frames around `frame`."""
        src = self.item_source(d)
        if d.get('shape') not in ('mesh', 'volume') or not src or not mesh_deforms(src):
            return {}
        fps = self.fps / max(self.v('domain', 'time_scale', frame), 1e-3)
        return {'mesh_frame': float(frame) - float(d.get('mesh_offset', 0.0)), 'mesh_fps': fps}

    def mesh_items(self, frame=None):
        """The meshes in use, as (source, frame) pairs for the mesh library (frame None when static)."""
        out = []
        for d in self.emitters + self.colliders:
            src = self.item_source(d) if d['enabled'] else ''
            if src:
                t = self._mesh_time(d, self.start if frame is None else frame)
                out.append((src, t.get('mesh_frame')))
        return out

    def colliders_gpu(self, frame=None, overrides=None):
        """Colliders at `frame` (fractional frames allowed), with the velocity of animated ones.
        `overrides` maps a collider index to ColliderGPU fields (pos, rot_y, vel, spin) that replace the
        keyframed ones: floating objects, which the liquid moves."""
        f = self.start if frame is None else frame
        out = []
        for i, c in enumerate(self.colliders):
            if not c['enabled']:
                continue
            g = lambda k: self.get(('collider', i, k), f)
            turn = dict(rot_y=math.radians(g('yaw')), spin=math.radians(self.rate(('collider', i, 'yaw'), f)))
            if self.tilted(i):
                # tipped over: its whole turn in its quaternion, and how fast it turns as its omega (world axes), so the
                # solids, the shaders and the motion blur all turn it about the same axis
                turn = dict(rot_y=0.0, spin=0.0, quat=turn_quat(*(math.radians(g(k)) for k in ('yaw', 'pitch', 'roll'))),
                            omega=self.turn_rate(i, f))
            out.append(ColliderGPU(
                shape=c['shape'], pos=tuple(g('position')), size=tuple(g('size')),
                vel=tuple(self.rate(('collider', i, 'position'), f)),
                burnable=bool(c['burnable']), mesh=self.mesh_path(c['mesh']) if c['shape'] == 'mesh' else '',
                hollow=g('hollow'), opening=tuple(g('opening')), opening_at=tuple(g('opening_at')), holdout=bool(c['holdout']),
                **turn, **self._mesh_time(c, f)))
        if overrides:
            import dataclasses
            idx = [i for i, c in enumerate(self.colliders) if c['enabled']]
            out = [dataclasses.replace(cg, **overrides[i]) if i in overrides else cg for i, cg in zip(idx, out)]
        return out[:MAX_COLLIDERS]

    def collider_heat(self):
        """Each enabled collider's surface temperature (K) and its material's thermal effusivity, in colliders_gpu's
        order: how it heats or chills the matter that touches it (mpm_heat.wgsl)."""
        from .materials import MATERIALS
        out = []
        for c in self.colliders:
            if c['enabled']:
                m = MATERIALS.get(c.get('material'), MATERIALS['wood'])
                out.append((float(c.get('temperature', 20.0)) + 273.15, float(m.effusivity)))
        return out[:MAX_COLLIDERS]

    def tilted(self, i):
        """Whether collider i is tipped over (Tilt or Roll), at any frame."""
        c = self.colliders[i]
        return any(isinstance(c.get(k), Curve) or float(c.get(k) or 0.0) != 0.0 for k in ('pitch', 'roll'))

    def turn(self, i, frame=None):
        """Collider i's orientation at `frame` as a 3x3 rotation (its own frame -> fire-local)."""
        f = self.start if frame is None else frame
        return turn_matrix(*(float(self.get(('collider', i, k), f)) for k in ('yaw', 'pitch', 'roll')))

    def turn_rate(self, i, frame):
        """How fast collider i turns from the keys of its Rotation, Tilt and Roll: (x, y, z, 0) rad/s, world axes."""
        if all(self.curve(('collider', i, k)) is None for k in ('yaw', 'pitch', 'roll')):
            return (0.0, 0.0, 0.0, 0.0)
        a, b = (turn_quat(*(math.radians(self.get(('collider', i, k), frame + s)) for k in ('yaw', 'pitch', 'roll')))
                for s in (-0.5, 0.5))
        x, y, z, w = quat_mul(b, (-a[0], -a[1], -a[2], a[3]))          # the turn from a to b, in world axes
        if w < 0.0:
            x, y, z, w = -x, -y, -z, -w
        sn = math.sqrt(x * x + y * y + z * z)
        if sn < 1e-12:
            return (0.0, 0.0, 0.0, 0.0)
        k = 2.0 * math.atan2(sn, w) / sn * self.fps / max(self.v('domain', 'time_scale', frame), 1e-3)
        return (x * k, y * k, z * k, 0.0)

    def colliders_animated(self):
        return (any(self.curve(('collider', i, k)) is not None for i, c in enumerate(self.colliders) if c['enabled']
                    for k in ('position', 'size', 'yaw', 'pitch', 'roll', 'hollow', 'opening', 'opening_at'))
                or any(self._mesh_time(c, self.start) for c in self.colliders if c['enabled']))

    # -- liquid ----------------------------------------------------------------------------------

    @property
    def kind(self):
        """'fire' (burning gas and smoke) or 'liquid'."""
        return self.data['domain'].get('kind', 'fire')

    def liquid_wind(self, frame):
        """Wind on the liquid (fire-local m/s)."""
        speed = self.v('liquid', 'wind_speed', frame)
        if speed <= 0:
            return (0.0, 0.0, 0.0)
        ang = math.radians(self.v('liquid', 'wind_dir', frame))
        w = np.array([math.sin(ang) * speed, 0.0, math.cos(ang) * speed])
        local = cam.rot_y(math.radians(self.v('camera', 'fire_yaw', frame))).T @ w
        return tuple(float(x) for x in local)

    def floating(self):
        """Indices and densities (kg/m^3) of the colliders the liquid moves."""
        if self.kind == 'fire':
            return []
        from .materials import resolved
        return [(i, resolved(c)['density']) for i, c in enumerate(self.colliders) if c['enabled'] and c.get('floating')]

    def liquid_params(self, frame):
        from ..engine.liquid import LiquidParams
        q, d = self.data['liquid'], self.data['domain']
        prm = LiquidParams(
            gravity=q['gravity'], flip=q['flip'], ppc=int(q['ppc']), surface_tension=q['surface_tension'],
            wall_drag=q['wall_drag'], drying=q['drying'], pressure_iters=int(q['pressure_iters']),
            volume_correction=q['volume_correction'], seed=float(d['seed']),
            whitewater=bool(q['whitewater']), ww_rate=25.0 * q['ww_amount'], ww_min_speed=q['ww_min_speed'],
            ww_turbulence=q['ww_turbulence'], ww_crests=q['ww_crests'], foam_life=q['foam_life'],
            bubble_rise=q['bubble_rise'], viscosity=q['viscosity'], rho=q['liquid_density'],
            contact_angle=q['contact_angle'], water_level=self.liquid_level(frame) if d['open_sides'] else 0.0,
            level_absorb=q['level_absorb'], wind=self.liquid_wind(frame), wind_surface=q['wind_surface'],
            narrow_band=bool(q['narrow_band']), band_width=int(q['band_width']),
            cooling=q.get('cooling', 0.0), solidify=q.get('solidify', 1000.0),
            current=self.liquid_current(frame),
            open_sides=d['open_sides'], open_top=d['open_top'], ground=d['ground'])
        for k, v in self.liquid_heat().items():
            setattr(prm, k, v)
        return prm

    def liquid_heat(self):
        """The liquid's heat settings (LiquidParams fields, liquid_thermal.py), with the air it meets: the
        scene's air for a liquid, the gas's for fire and liquid in one box. With heat on, wet ground dries
        at the rate a film of water (0.1 mm) evaporates, or boils off a hot ground."""
        q = self.data['liquid']
        if not q.get('thermal'):
            return {}
        from ..engine.liquid_thermal import boil_flux, evaporation_flux, vapour_sat
        s = self.data['shading']
        both = self.kind == 'both'
        air = s['ambient_k'] - 273.15 if both else float(q.get('air_temp', 20.0))
        rh = s['humidity'] if both else float(q.get('air_humidity', 50.0))
        temp = float(q.get('liquid_temp', 20.0))
        ground = float(q.get('ground_temp', 20.0))
        boil = float(q.get('boil_point', 100.0))
        speed = max(float(q.get('heat_speed', 1.0)), 1.0)
        cols = [float(c.get('temperature', 20.0)) for c in self.colliders if c['enabled']]
        srcs = [float(e.get('liquid_temp', temp)) for e in self.emitters
                if e['enabled'] and e.get('temp_own') and (not both or e.get('emits') == 'liquid')]
        vap = max(0.0, min(rh, 100.0)) / 100.0 * vapour_sat(air, air < 0.0)
        film = 0.1   # kg/m^2: the water a wet surface holds
        dry = max(evaporation_flux(min(ground, boil), air, vap), 0.0) / film
        dry = max(dry, boil_flux(ground - boil) / 2.26e6 / film)
        return dict(thermal=True, temp=temp, air_temp=air, humidity=rh, ground_temp=ground, collider_temps=tuple(cols),
                    min_source_temp=min([temp] + srcs), freeze_point=float(q.get('freeze_point', 0.0)), boil_point=boil,
                    supercool=float(q.get('supercool', 1.0)), heat_speed=speed, bubble_size=float(q.get('bubble_size', 2.5)),
                    gas_air_k=float(s['ambient_k']), gas_flame_k=float(s['flame_k']),
                    drying=min(max(float(q['drying']) if ground <= 0.0 else 0.0, dry * speed), 5.0))

    # -- clouds (kind 'cloud') ---------------------------------------------------------------------

    def cloud_params(self, frame):
        """The cloud simulation's settings (engine/cloud.py CloudParams)."""
        from ..engine.cloud import CloudParams
        a = self.data['atmosphere']
        return CloudParams(
            surface_t=float(a['surface_t']), surface_rh=float(a['surface_rh']) / 100.0, mixed_layer=float(a['mixed_layer']),
            lapse=float(a['lapse']), tropopause=float(a['tropopause']), free_rh=float(a['free_rh']) / 100.0,
            inversion_z=float(a['inversion_z']), inversion_dt=float(a['inversion_dt']), wind=float(a['wind']),
            wind_dir=float(a['wind_dir']), shear=float(a['shear']), veer=float(a['veer']), follow=bool(a.get('follow', False)),
            heat_flux=float(a['heat_flux']),
            evaporation=float(a['evaporation']), thermal_size=float(a['thermal_size']), patchy=float(a['patchy']),
            bubble=float(a['bubble']), bubble_r=float(a['bubble_r']), rain_threshold=float(a['rain_threshold']) * 1e-3,
            hail=float(a['hail']), glaciation=float(a['glaciation']), vorticity=float(a['vorticity']),
            seed=int(self.data['domain'].get('seed', 0)))

    def sky_look(self, frame, final=False):
        """How the sky looks (engine/cloud_render.py SkyLook): the sun from Lighting, the rest from Sky."""
        from ..engine.cloud_render import SkyLook
        k, l = self.data['sky'], self.data['lighting']
        sun = np.asarray(l['sun_color']) * (self.v('lighting', 'sun_intensity', frame) if l['sun_on'] else 0.0)
        return SkyLook(sun_azimuth=self.v('lighting', 'sun_azimuth', frame), sun_elevation=self.v('lighting', 'sun_elevation', frame),
                       sun=tuple(float(x) for x in sun), sky=tuple(float(x) for x in k['sky_color']),
                       horizon=tuple(float(x) for x in k['horizon_color']), ground=tuple(float(x) for x in k['ground_color']),
                       brightness=float(k['brightness']), skylight=float(k['skylight']), silver=float(k['silver']),
                       multiple=float(k['multiple']), visibility=float(k['visibility']) * 1000.0, draw_sky=bool(k['draw_sky']),
                       density=float(k['density']))

    # -- weather ---------------------------------------------------------------------------------

    def has_weather(self):
        """Precipitation is simulated (Weather section, liquid and fire-and-liquid scenes)."""
        w = self.data.get('weather', {})
        return self.kind in ('liquid', 'both') and w.get('precip', 'none') != 'none'

    def weather_params(self, frame, final=False):
        """The precipitation's settings at `frame` (engine/weather.py WeatherParams): the air at the ground
        is the liquid's (Liquid › Air temperature, humidity; the gas's in a fire-and-liquid box), its wind is
        the liquid's, and its area is centred on the box and covers it."""
        from ..engine.weather import WeatherParams
        w = self.data['weather']
        q, d = self.data['liquid'], self.data['domain']
        both = self.kind == 'both'
        s = self.data['shading']
        air = s['ambient_k'] - 273.15 if both else float(q.get('air_temp', 20.0))
        dims, h, origin = self.sim_layout(final)
        x0, z0 = origin[0], origin[2]
        x1, z1 = x0 + dims[0] * h, z0 + dims[2] * h
        cx, cz = 0.5 * (x0 + x1), 0.5 * (z0 + z1)
        half = 0.5 * max(float(w['area']), x1 - x0 + 0.2, z1 - z0 + 0.2)
        top = float(w['top']) if w['top'] > 0.0 else origin[1] + dims[1] * h + 1.0
        cols = [float(c.get('temperature', 20.0)) for c in self.colliders if c['enabled']]
        return WeatherParams(
            kind=w['precip'], rate=float(self.v('weather', 'rate', frame)), size=float(w['size']), ground_t=air,
            humidity=float(w['humidity']) / 100.0, warm_t=float(w['warm_t']), warm_z=float(w['warm_z']),
            wind=tuple(self.liquid_wind(frame)), gust=float(w['gust']), turbulence=float(w['turbulence']), eddy=float(w['eddy']),
            area=(cx - half, cz - half, cx + half, cz + half), top=top, ground=bool(d['ground']),
            ground_temp=float(q.get('ground_temp', 20.0)), collider_temps=tuple(cols), cover=bool(w['cover']),
            cover_cell=float(w['cover_cell']) / 100.0, start=float(w['starts']), lying=float(w['lying']) / 100.0,
            buildup=float(w['buildup']),
            heat_speed=max(float(q.get('heat_speed', 1.0)), 1.0) if q.get('thermal') else 1.0,
            freeze_point=float(q.get('freeze_point', 0.0)), capacity=int(float(w['limit']) * 1e6),
            seed=int(d.get('seed', 0)))

    def weather_look(self):
        w = self.data['weather']
        return dict(snow_bright=float(w['snow_bright']), sparkle=float(w['sparkle']), gloss=float(w['gloss']),
                    opacity=float(w['fall_opacity']))

    # -- lava in a fire-and-liquid scene ---------------------------------------------------------

    def has_lava(self):
        """A fire-and-liquid scene with a lava source: the lava is a liquid of its own beside the water."""
        return self.kind == 'both' and any(e['enabled'] and e.get('emits') == 'lava' for e in self.emitters)

    def lava_settings(self):
        """(liquid, look) settings of the lava: the Lava flow preset's (so the lava looks and moves as that
        preset's does), with any setting this scene's Lava section changes from its default over them."""
        from .presets import PRESETS   # (here: presets imports this module)
        base = PRESETS.get('lava', {})
        liq = {**defaults('liquid'), **copy.deepcopy(base.get('liquid', {}))}
        look = {**defaults('water'), **copy.deepcopy(base.get('water', {}))}
        dflt = defaults('lava')
        for k, v in self.data.get('lava', {}).items():
            if isinstance(v, Curve) or v != dflt.get(k):
                for d in (liq, look):
                    if k in d:
                        d[k] = v
        liq['whitewater'] = False
        # the scene's own: how its colliders are drawn, and what the backdrop is
        for k in ('colliders_look', 'standin_color', 'backdrop', 'reflect_footage', 'lens_drops'):
            if k in self.data['water']:
                look[k] = self.data['water'][k]
        return liq, look

    def _as_lava(self, fn):
        saved = self.data['liquid'], self.data['water']
        self.data['liquid'], self.data['water'] = self.lava_settings()
        try:
            return fn()
        finally:
            self.data['liquid'], self.data['water'] = saved

    def lava_params(self, frame):
        """The lava's LiquidParams at `frame` (it has no open water, sea, rain or whitewater of its own)."""
        prm = self._as_lava(lambda: self.liquid_params(frame))
        prm.water_level = 0.0
        prm.current = (0.0, 0.0, 0.0)
        prm.whitewater = False
        prm.ocean = None
        return prm

    def lava_look(self, frame, final=False):
        """The lava's WaterLook at `frame`: the Lava flow preset's look with this scene's Lava section."""
        look = self._as_lava(lambda: self.water_look(frame, final))
        look.whitewater = False
        return look

    def liquid_level(self, frame):
        """The open water's level at `frame` (m, 0: none). A tide keyframes it; while the open water is on
        it stays a little above the ground, so the box never loses its sides mid-shot."""
        q = self.data['liquid']
        if 'water_level' not in q:
            return 0.0
        v = float(self.v('liquid', 'water_level', frame))
        c = self.curve(('liquid', 'water_level'))
        top = max(float(k[1]) for k in c.keys) if c is not None and c.keys else v
        return max(v, 0.02) if top > 0.0 else 0.0

    def liquid_current(self, frame):
        """The open water's current (fire-local m/s)."""
        q = self.data['liquid']
        speed = self.v('liquid', 'current_speed', frame) if 'current_speed' in q else 0.0
        if speed <= 0 or self.liquid_level(frame) <= 0:
            return (0.0, 0.0, 0.0)
        ang = math.radians(self.v('liquid', 'current_dir', frame))
        w = np.array([math.sin(ang) * speed, 0.0, math.cos(ang) * speed])
        local = cam.rot_y(math.radians(self.v('camera', 'fire_yaw', frame))).T @ w
        return tuple(float(x) for x in local)

    def whitewater_capacity(self):
        q = self.data['liquid']
        return int(q['ww_max'] * 1e6) if q['whitewater'] else 0

    def liquid_capacity(self, final=False):
        from ..engine.liquid import LiquidSolver
        dims, _, _ = self.sim_layout(final)
        q = self.data['liquid']
        return LiquidSolver.capacity_for(dims, int(q['ppc']), q['max_particles'] * 1e6,
                                         band=int(q['band_width']) if q.get('narrow_band') else None)

    def sources_gpu(self, frame, filled=None, emits='liquid'):
        """Liquid sources at `frame` (fractional allowed). A volume ('fill') source pours once, the
        first time the shot reaches its start; `filled` holds the indices that already have (and is
        updated). In a fire-and-liquid scene, `emits` picks the liquid ('liquid': the water, or 'lava')."""
        from ..engine.liquid import source
        out = []
        t = self.seconds(frame)
        both = self.kind == 'both'
        ref = self.lava_settings()[0]['liquid_density'] if emits == 'lava' else self.data['liquid']['liquid_density']
        for i, e in enumerate(self.emitters):
            if not e['enabled'] or (both and e.get('emits', 'fire') != emits) or (not both and emits != 'liquid'):
                continue
            g = lambda k: self.get(('emitter', i, k), frame)
            fill = e['liquid_mode'] == 'fill'
            if fill:
                if filled is None or i in filled or (e['start'] > -99 and t < e['start']):
                    continue
                filled.add(i)
                strength = 1.0
            else:
                if e['start'] > -99 and t < e['start']:
                    continue
                if e['stop'] >= 0 and t >= e['stop']:
                    continue
                strength = g('flow')
                if strength <= 1e-4:
                    continue
            motion = np.asarray(self.rate(('emitter', i, 'position'), frame), float)
            vel = np.asarray(g('velocity'), float) + float(e['inherit']) * motion
            src = source(shape=e['shape'], pos=tuple(g('position')), size=tuple(g('size')), p1=tuple(g('end')),
                         vel=tuple(float(x) for x in vel), radial=g('radial'), strength=strength, fill=fill,
                         jitter=e['jitter'], vel_blend=0.0 if fill else e['vel_blend'], yaw=math.radians(g('yaw')),
                         dye=tuple(e.get('dye', (1.0, 1.0, 1.0))), dye_amount=float(e.get('dye_amount', 0.0)),
                         dye_cloud=float(e.get('dye_cloud', 0.5)),
                         density=(float(e.get('liquid_density', 0.0)) / max(float(ref), 1.0)
                                  if e.get('liquid_density', 0.0) > 0 else 1.0),
                         mesh=self.item_source(e), soft=e['softness'],
                         temp=float(e.get('liquid_temp', 20.0)) if e.get('temp_own') else None)
            if e['shape'] == 'volume':   # (a volume pours where it is dense, its water moving as the volume does)
                src.volume_vel = float(e.get('volume_velocity', 1.0))
            out.append(src)
        return out[:MAX_EMITTERS]

    def water_look(self, frame, final=False):
        from ..engine.liquid_render import WaterLook
        w, l = self.data['water'], self.data['lighting']
        sky = np.asarray(self.v('lighting', 'ambient', frame)) * self.v('lighting', 'ambient_intensity', frame)
        sun = np.asarray(l['sun_color']) * (self.v('lighting', 'sun_intensity', frame) if l['sun_on'] else 0.0)
        from ..engine import sky as sky_mod
        phys = sky_mod.of_scene(self, frame)
        sky_image, env_strength = None, l['env_strength']
        if phys is not None:
            # the physical sky (engine/sky.py): its light, the sun through the air, its HDRI to reflect
            sky = phys[1] * phys[4] * self.v('lighting', 'ambient_intensity', frame)
            sun = np.asarray(phys[2]) * (phys[4] if l['sun_on'] else 0.0)
            sky_image, env_strength = (phys[0], phys[3]), phys[4]
        return WaterLook(
            ior=w['ior'], color=tuple(w['color']), clarity=w['clarity'], murk=w['murk'], murk_color=tuple(w['murk_color']),
            roughness=w['roughness'], reflection=w['reflection'], reflect_footage=w['reflect_footage'],
            ripple=self.v('water', 'ripple', frame), ripple_freq=w['ripple_freq'], radius=w['radius'],
            smoothing=int(w['smoothing']), calm=int(w['calm']), surface_res=w['surface_res'] if final else min(w['surface_res'], 2.0),
            wet_darken=self.v('water', 'wet_darken', frame), wet_gloss=w['wet_gloss'], shadow=w['shadow'],
            backdrop=w['backdrop'], sky=tuple(float(x) for x in sky), sun=tuple(float(x) for x in sun),
            sun_azimuth=self.v('lighting', 'sun_azimuth', frame), sun_elevation=self.v('lighting', 'sun_elevation', frame),
            whitewater=bool(self.data['liquid']['whitewater']), foam=w['foam'], spray=w['spray'], bubbles=w['bubbles'],
            foam_color=tuple(w['foam_color']), step_out=w['step'], step_in=max(w['step'], 0.5),
            foam_scale=w['foam_scale'], foam_lace=w['foam_lace'], droplets=w['droplets'], droplet_size=w['droplet_size'],
            sheets=w['sheets'], caustics=w['caustics'], colliders_look=w['colliders_look'],
            environment=self.mesh_path(l['environment']) if l['environment'] else '',
            env_rotation=l['env_rotation'] if sky_image is None else 0.0,   # (the sky's image is the world's way round)
            env_strength=env_strength, env_sun=bool(l['env_sun']) and sky_image is None, wind=self.liquid_wind(frame),
            sky_image=sky_image,
            glow=w.get('glow', 0.0), glow_temp=w.get('glow_temp', 1300.0), crust=w.get('crust', 0.7),
            crust_scale=w.get('crust_scale', 0.08), whitecaps=w.get('whitecaps', 1.0),
            sea_foam=w.get('sea_foam', 1.0), sea_foam_life=w.get('sea_foam_life', 12.0), foam_streaks=w.get('foam_streaks', 0.7),
            crest_glow=w.get('crest_glow', 1.0), crest_glow_color=tuple(w.get('crest_glow_color', (0.1, 0.55, 0.45))),
            gusts=w.get('gusts', 0.3), standin_color=tuple(w.get('standin_color', (0.3, 0.3, 0.3))), bottomless=bool(w.get('bottomless', False)),
            rainbow=w.get('rainbow', 1.0), lens_drops=w.get('lens_drops', 0.0),
            crust_time=w.get('crust_time', 1.5), ropes=w.get('ropes', 1.0),
            crust_color=tuple(w.get('crust_color', (0.075, 0.073, 0.071))), crust_roughness=w.get('crust_roughness', 0.45),
            crust_kind=w.get('crust_kind', 'skin'),
            lava_light=w.get('lava_light', 1.0), lava_cooling=float(self.data['liquid'].get('cooling', 0.0)),
            **self._ice_look())

    def _ice_look(self):
        """The look of the liquid's ice (WaterLook fields), when it has heat."""
        q, w = self.data['liquid'], self.data['water']
        if not q.get('thermal'):
            return {}
        air = self.data['shading']['ambient_k'] - 273.15 if self.kind == 'both' else float(q.get('air_temp', 20.0))
        melt = min(max((air - float(q.get('freeze_point', 0.0))) / 3.0, 0.0), 1.0)
        return dict(ice=True, frost=float(w.get('frost', 1.0)), ice_cloud=float(w.get('ice_cloud', 1.0)),
                    frost_color=tuple(w.get('frost_color', (0.9, 0.93, 0.97))), crystal_size=float(w.get('crystal_size', 0.02)),
                    ice_melting=melt)

    def look(self, frame, final=False):
        s, l = self.data['shading'], self.data['lighting']
        amb = np.asarray(self.v('lighting', 'ambient', frame)) * self.v('lighting', 'ambient_intensity', frame)
        sun_color = tuple(l['sun_color'])
        from ..engine import sky as sky_mod
        phys = sky_mod.of_scene(self, frame)
        if phys is not None:
            # the physical sky (engine/sky.py): its light on an upward surface for the ambient, the sun through the air
            amb = phys[1] * phys[4] * self.v('lighting', 'ambient_intensity', frame)
            sun_color = tuple(float(x) for x in phys[2])
        return LookParams(
            ambient_k=s['ambient_k'], flame_k=self.v('shading', 'flame_k', frame), max_k=s['max_k'],
            dynamic_range=s['dynamic_range'], intensity=s['intensity'],
            exposure=self.v('shading', 'exposure', frame) + self.physical_ev(frame), colour_response=s['colour_response'],
            flame_density=s['flame_density'], flame_sharpness=s['flame_sharpness'], flame_threshold=s['flame_threshold'],
            soot_glow=s['soot_glow'], flame_absorption=s['flame_absorption'], flame_occlusion=s['flame_occlusion'], blue=s['blue'], blue_color=tuple(s['blue_color']),
            ignition=self.data['combustion']['ignition'], smoke_albedo=tuple(s['smoke_albedo']),
            smoke_density=self.v('shading', 'smoke_density', frame), ambient=tuple(amb), anisotropy=s['anisotropy'],
            sun_color=sun_color, sun_intensity=self.v('lighting', 'sun_intensity', frame) if l['sun_on'] else 0.0,
            sun_azimuth=self.v('lighting', 'sun_azimuth', frame), sun_elevation=self.v('lighting', 'sun_elevation', frame),
            fire_scatter=l['fire_scatter'], light_spread=l['light_spread'], shadow=l['shadow'],
            detail=s['detail'], detail_freq=s['detail_freq'], detail_disp=s['detail_disp'], detail_rise=s['detail_rise'],
            step=self.data['render']['final_step'] if final else s['step'], edge_fade=s['edge_fade'],
            colour_gain=s['colour_gain'], humidity=s['humidity'], steam_density=s['steam_density'],
            steam_albedo=tuple(s['steam_albedo']), multiple_scattering=s['multiple_scattering'],
            coal_bed=self.v('shading', 'coal_bed', frame), coal_k=s['coal_k'],
            coal_height=s['coal_height'], lamps=self.lamps(frame))

    def comp(self, frame, mode='composite'):
        c = self.data['composite']
        return CompParams(
            mode=mode, view=c['view'], knee=c['knee'], plate_gain=2.0 ** c['plate_exposure'],
            plate_transform=c['plate_transform'], fire_gain=1.0, smoke_opacity=self.v('composite', 'smoke_opacity', frame),
            bloom=self.v('composite', 'bloom', frame), bloom_radius=c['bloom_radius'],
            light_cast=self.v('composite', 'light_cast', frame), haze=self.v('composite', 'haze', frame),
            haze_freq=c['haze_freq'], haze_speed=c['haze_speed'], saturation=c['saturation'], grain=c['grain'],
            bg=tuple(c['bg']), bg_checker=c['bg_checker'], tint=tuple(c['tint']),
            surface_light=self.v('composite', 'surface_light', frame), scorch=c['scorch'],
            surface_shadows=c['surface_shadows'], soot=c['soot'], wet=c['wet'], depth_kind=c['depth_kind'],
            depth_scale=c['depth_scale'], highlight_white=c['highlight_white'], visibility=c['visibility'], atmos_colour=tuple(c['atmos_colour']),
            atmos_from_footage=c['atmos_from_footage'], grain_match=c['grain_match'],
            f_stop=c['f_stop'], focus_distance=self.v('composite', 'focus_distance', frame), softness=c['softness'],
            focal_mm=self.v('camera', 'focal_mm', frame), sensor_mm=self.v('camera', 'sensor_mm', frame),
            lens_k1=c['lens_k1'], fringing=c['fringing'], halation=c['halation'])

    def physical_ev(self, frame):
        """Stops Real-world brightness adds to Fire exposure: thick flame at the flame temperature shines as a
        blackbody of that temperature does, in footage exposed so that a grey card in the scene's light
        (Scene light, EV at ISO 100) comes out at 0.18."""
        if not self.data['shading']['exposure_physical']:
            return 0.0
        return math.log2(0.18) + flame_ev(self.v('shading', 'flame_k', frame)) - self.v('shading', 'scene_ev', frame)

    def ember_params(self, frame):
        e = self.data['embers']
        return EmberParams(**{k: e[k] for k in EmberParams.__dataclass_fields__ if k in e})

    def camera(self, frame):
        g = lambda k: self.v('camera', k, frame)
        anchor = (g('anchor_x'), g('anchor_y'))
        if self.track and self.track.get('points'):
            pt = track_point(self.track, frame)
            if pt is not None:
                off = self.track.get('offset', (0.0, 0.0))
                anchor = (pt[0] + off[0], pt[1] + off[1])
        spec = cam.CameraSpec(
            mode=g('mode'), yaw=g('yaw'), pitch=g('pitch'), distance=g('distance'), target_y=g('target_y'),
            position=tuple(g('position')), rotation=tuple(g('rotation')), focal_mm=g('focal_mm'), sensor_mm=g('sensor_mm'),
            anchor=anchor, scale=g('scale'), roll=g('roll'), use_anchor=g('use_anchor'), near=g('near'), far=g('far'))
        fire = cam.FireXform(position=tuple(g('fire_position')), yaw=g('fire_yaw'))
        return spec, fire

    def output_size(self):
        r = self.data['render']
        return int(r['width']), int(r['height'])

    def upres_for(self, final=False):
        """The detail upres factor: in final renders, and in the viewer with Render › Upres in the viewer."""
        r = self.data['render']
        auto = self.auto_detail() if final else None
        if auto:
            return auto[1]
        return max(1, int(r['upres'])) if (final or r.get('upres_preview')) else 1

    def substep_cap(self, frame, h=None):
        """The most substeps a frame may take: Domain › Max substeps, raised with the time scale (each frame then
        covers that much more simulated time) and with a grid finer than the hand resolution (cell size h, as
        Resolution from the shot makes it), so the gas moves no more cells per substep than it was tuned at."""
        d = self.data['domain']
        cap = d['substeps_max'] * max(1.0, self.v('domain', 'time_scale', frame))
        if h:
            cap *= max(1.0, max(self.domain_size()) / max(d['resolution'], 1) / h)
        return int(math.ceil(cap))

    def auto_detail(self):
        """Render › Resolution from the shot: (voxels on the longest side, upres) for final renders. The voxels
        stay as set; with Detail upres left at 1, the upres rises until a cell of the upres grid is about
        AUTO_PX_PER_CELL pixels on screen where the box is largest in the shot (checked at a few frames), within
        AUTO_MAX_UPRES and the memory budget. None when it is off or the box is not in view."""
        r, d = self.data['render'], self.data['domain']
        if not r.get('auto_resolution') or self.kind != 'fire':
            return None
        W, H = self.output_size()
        sx, sy, sz = self.domain_size()
        corners = np.array([(x, y, z) for x in (-sx / 2, sx / 2) for y in (0.0, sy) for z in (-sz / 2, sz / 2)])
        px_per_m = 0.0
        for f in sorted({self.start, (self.start + self.end) // 2, self.end}):
            spec, fire = self.camera(f)
            cs = cam.compute(spec, W / H, fire)
            world = corners @ cam.rot_y(math.radians(fire.yaw)).T + np.asarray(fire.position, float)
            clip = np.c_[world, np.ones(len(world))] @ np.asarray(cs.view_proj).T
            front = clip[:, 3] > 1e-6
            if front.sum() < 2:
                continue
            scr = clip[front, :2] / clip[front, 3:4] * 0.5 * np.array([W, H])
            ext = scr.max(axis=0) - scr.min(axis=0)
            px_per_m = max(px_per_m, ext[0] / max(sx, sz), ext[1] / sy)
        if px_per_m <= 0.0:
            return None
        cells = max(sx, sy, sz) * px_per_m / AUTO_PX_PER_CELL
        # Only the detail upres goes up. A finer simulation grid changes how the fire burns (the Campfire's
        # flames came out about a fifth shorter on a grid twice as fine) and costs about four times as much,
        # yet against footage of a real campfire its flame edges were no sharper; the upres grid only carries
        # the fire it simulates, and adds the fine detail.
        res = max(16, int(round(d['resolution'] * r['final_scale'])))
        hand_up = max(1, int(r['upres']))
        if hand_up > 1:
            return res, hand_up   # Detail upres set by hand is kept
        up = min(max(1, math.ceil(cells / res)), AUTO_MAX_UPRES)
        # within the memory budget (Solver.memory_bytes, about), never below the hand settings
        n = int(np.prod(Solver.dims_for(self.domain_size(), res)[0]))
        while up > hand_up and n * (62 + 28 * up ** 3) > AUTO_MAX_BYTES:
            up -= 1
        return res, up

    def sim_signature(self, final=False):
        """Changes whenever anything that affects the simulation changes."""
        blob = {
            'sections': {s: {k: _to_json_value(v) for k, v in self.data[s].items() if (s, k) not in CACHE_ONLY and (s, k) not in LOOK_KEYS}
                         for s in SIM_SECTIONS},
            # (Velocity from the volume moves nothing but a Volume emitter: left out for the others, their caches stand)
            'emitters': [{k: _to_json_value(v) for k, v in e.items()
                          if k != 'name' and (k != 'volume_velocity' or e.get('shape') == 'volume')} for e in self.emitters],
            'colliders': [{k: _to_json_value(v) for k, v in c.items() if k != 'name'} for c in self.colliders],
            'fabrics': [{k: _to_json_value(v) for k, v in f.items() if k != 'name'} for f in self.fabrics],
            # (lightning that sets fire adds flames where it strikes)
            'lightning': [{k: _to_json_value(l[k]) for k in ('position', 'end', 'strike_at', 'kind')}
                          for l in self.lights if l.get('kind') == 'lightning' and l.get('ignites') and l.get('enabled')],
            'matter': [{k: _to_json_value(v) for k, v in m.items() if k != 'name'} for m in getattr(self, 'matter', None) or []],
            'strands': [{k: _to_json_value(v) for k, v in m.items() if k not in ('name', 'own_colour', 'colour')}
                        for m in getattr(self, 'strands', None) or []],
            'shots': [{k: _to_json_value(v) for k, v in m.items() if k != 'name'}
                      for m in getattr(self, 'shots', None) or []],
            'fps': self.fps, 'start': self.start, 'layout': [list(x) if isinstance(x, tuple) else x for x in self.sim_layout(final)],
            'embers': {k: _to_json_value(v) for k, v in self.data['embers'].items()},
            'fire_yaw': _to_json_value(self.data['camera']['fire_yaw']),
            'meshes': self._mesh_stamps(),
            # dousing cools gas to the boiling point, and steam condenses (releasing heat) against the
            # air's temperature and humidity: with steam these look settings change the simulation too
            'steam': ([round(self.boil_temp(), 5)] + [self.data['shading'][k] for k in ('ambient_k', 'flame_k', 'humidity')]
                      if self.features()['vapour'] else None),
            'upres': self.upres_for(final),
            'engine': SIM_VERSION,     # (frames simulated by older code are another simulation's)
        }
        return hashlib.sha1(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def _mesh_stamps(self):
        """Modification times of the mesh files in use, so editing a mesh re-simulates."""
        from ..engine.mesh import MeshLibrary
        out = {}
        for d in (self.emitters + self.colliders + [f for f in self.fabrics if f.get('shape') == 'mesh']
                  + [m for m in (getattr(self, 'matter', None) or []) if m.get('shape') == 'mesh']):
            p = self.item_source(d)
            if p:
                try:
                    out[p] = MeshLibrary._stamp(p)
                except OSError:
                    out[p] = None
        return out

    # -- serialisation ---------------------------------------------------------------------------

    def to_dict(self):
        return {
            'format': 'blackbody-scene', 'version': FILE_VERSION, 'name': self.name, 'preset': self.preset,
            'notes': self.notes,
            'sections': {s: {k: _to_json_value(v) for k, v in vals.items()} for s, vals in self.data.items()},
            'emitters': [{k: _to_json_value(v) for k, v in e.items()} for e in self.emitters],
            'colliders': [{k: _to_json_value(v) for k, v in c.items()} for c in self.colliders],
            'lights': [{k: _to_json_value(v) for k, v in l.items()} for l in self.lights],
            'fabrics': [{k: _to_json_value(v) for k, v in f.items()} for f in self.fabrics],
            'matter': [{k: _to_json_value(v) for k, v in m.items()} for m in getattr(self, 'matter', None) or []],
            'strands': [{k: _to_json_value(v) for k, v in m.items()} for m in getattr(self, 'strands', None) or []],
            'shots': [{k: _to_json_value(v) for k, v in m.items()} for m in getattr(self, 'shots', None) or []],
            'footage': self.footage,
            'track': ({'points': {str(k): list(v) for k, v in self.track.get('points', {}).items()},
                       'offset': list(self.track.get('offset', (0, 0)))} if self.track else None),
            'roto': getattr(self, 'roto', None) or [],
            'ground': getattr(self, 'ground', None),
            'layers': [dict(l.to_dict(), uid=l.uid, enabled=bool(l.enabled)) for l in getattr(self, 'layers', None) or []],
            'base_index': int(getattr(self, 'base_index', 0)),
            'links': [dict(l) for l in getattr(self, 'links', None) or []],
        }

    @classmethod
    def from_dict(cls, d):
        s = cls()
        if d.get('format') not in (None, 'blackbody-scene'):
            raise ValueError('Not a Blackbody scene file')
        s.name = d.get('name', 'Untitled')
        s.preset = d.get('preset')
        s.notes = d.get('notes', '')
        for sec, vals in d.get('sections', {}).items():
            if sec not in s.data:
                continue
            for k, v in vals.items():
                try:
                    s.data[sec][k] = _from_json_value(param(sec, k), v)
                except KeyError:
                    pass  # parameter from a newer or older version
        if 'composite' in d.get('sections', {}) and 'backdrop' not in d['sections']['composite']:
            s.data['composite']['backdrop'] = 'colour'   # (saved before the stage: the flat background it was made on)
        if 'emitters' in d:
            s.emitters = []
            for e in d['emitters']:
                x = emitter_defaults()
                for k, v in e.items():
                    if k in x:
                        x[k] = _from_json_value(param('emitter', k), v)
                s.emitters.append(x)
        if 'colliders' in d:
            s.colliders = []
            for c in d['colliders']:
                x = collider_defaults()
                for k, v in c.items():
                    if k in x:
                        x[k] = _from_json_value(param('collider', k), v)
                if 'keeps_temperature' not in c and float(c.get('temperature', 20.0)) != 20.0:
                    x['keeps_temperature'] = True   # (saved before objects warmed and cooled: they held their heat)
                s.colliders.append(x)
        s.lights = []
        for l in d.get('lights', []):
            x = light_defaults()
            for k, v in l.items():
                if k in x:
                    x[k] = _from_json_value(param('light', k), v)
            if x['kind'] == 'area' and 'width' not in l:
                x['width'] = x['height'] = round(max(float(x['radius']), 0.01) * math.sqrt(math.pi), 3)
            s.lights.append(x)
        s.fabrics = []
        for f in d.get('fabrics', []):
            x = fabric_defaults()
            for k, v in f.items():
                if k in x:
                    x[k] = _from_json_value(param('fabric', k), v)
            s.fabrics.append(x)
        s.matter = []
        for m in d.get('matter', []):
            x = matter_defaults()
            for k, v in m.items():
                if k in x:
                    x[k] = _from_json_value(param('matter', k), v)
            s.matter.append(x)
        s.strands = []
        for m in d.get('strands', []):
            x = strand_defaults()
            for k, v in m.items():
                if k in x:
                    x[k] = _from_json_value(param('strands', k), v)
            s.strands.append(x)
        s.shots = []
        for m in d.get('shots', []):
            x = shot_defaults()
            for k, v in m.items():
                if k in x:
                    x[k] = _from_json_value(param('shot', k), v)
            s.shots.append(x)
        s.footage = d.get('footage')
        tr = d.get('track')
        if tr and tr.get('points'):
            s.track = {'points': {int(float(k)): tuple(v) for k, v in tr['points'].items()},
                       'offset': tuple(tr.get('offset', (0, 0)))}
        s.roto = [dict(r) for r in (d.get('roto') or []) if isinstance(r, dict)]
        g = d.get('ground')
        s.ground = dict(g) if isinstance(g, dict) and len(g.get('corners') or []) == 4 else None
        for i, ld in enumerate(d.get('layers') or []):
            if not isinstance(ld, dict):
                continue
            ld = dict(ld, layers=[])   # layers do not nest
            layer = cls.from_dict(ld)
            layer.uid = str(ld.get('uid') or f'layer{i + 1}')
            layer.enabled = bool(ld.get('enabled', True))
            s.layers.append(layer)
        s.base_index = max(0, min(int(d.get('base_index', 0) or 0), len(s.layers)))
        s.links = [dict(l) for l in (d.get('links') or []) if isinstance(l, dict) and l.get('child') and l.get('parent')]
        if s.layers:
            s.sync_layers()
        return s

    # -- attached objects ------------------------------------------------------------------------------

    def find_object(self, kind, name):
        """(index, dict) of the object of a kind with a name, or (None, None)."""
        items = {'emitter': self.emitters, 'collider': self.colliders, 'light': self.lights, 'fabric': self.fabrics,
                 'matter': getattr(self, 'matter', []), 'strands': getattr(self, 'strands', []),
                 'shot': getattr(self, 'shots', [])}.get(kind, [])
        for i, d in enumerate(items):
            if d.get('name') == name:
                return i, d
        return None, None

    def link_of(self, kind, name):
        return next((l for l in getattr(self, 'links', None) or [] if list(l['child']) == [kind, name]), None)

    def apply_links(self):
        """Put every attached object where its parent is, at its offset, over the whole shot (a parent's keys
        become the child's, moved by the offset). A link with 'shape' also gives the child the parent's mesh, size
        and rotation (fire on solid letters). Links whose objects are gone are dropped."""
        keep = []
        for l in getattr(self, 'links', None) or []:
            ci, child = self.find_object(*l['child'])
            pi, parent = self.find_object(*l['parent'])
            if child is None or parent is None or child is parent:
                continue
            keep.append(l)
            off = tuple(float(x) for x in l.get('offset', (0.0, 0.0, 0.0)))
            pp = parent['position']
            old = child['position']
            if isinstance(pp, Curve):
                child['position'] = Curve([[f, tuple(a + b for a, b in zip(v, off)), it] for f, v, it in pp.keys])
            else:
                child['position'] = tuple(a + b for a, b in zip(pp, off))
            if l['child'][0] == 'emitter' and child.get('shape') == 'capsule' and 'end_offset' in l:
                eo = tuple(float(x) for x in l['end_offset'])
                if isinstance(pp, Curve):
                    child['end'] = Curve([[f, tuple(a + b for a, b in zip(v, eo)), it] for f, v, it in pp.keys])
                else:
                    child['end'] = tuple(a + b for a, b in zip(pp, eo))
            if l.get('shape'):
                import copy
                for k in ('mesh', 'size', 'yaw'):
                    if k in parent and k in child:
                        child[k] = copy.deepcopy(parent[k])
            del old
        self.links = keep

    # -- layers ------------------------------------------------------------------------------------

    def layer_order(self, enabled_only=True):
        """The shot's effects back to front: [(uid, scene)] (this scene's own is 'base')."""
        out = [(l.uid, l) for l in getattr(self, 'layers', None) or []]
        out.insert(min(getattr(self, 'base_index', 0), len(out)), ('base', self))
        return [(u, s) for u, s in out if s.enabled or not enabled_only]

    def layer(self, uid):
        if uid in (None, 'base'):
            return self
        return next((l for l in self.layers if l.uid == uid), None)

    def sync_layers(self, source=None):
        """Give every layer the shot's shared settings, from `source` (the layer just edited) or this scene."""
        src = source or self
        scenes = [self] + list(self.layers)
        for l in self.layers:   # project-relative files in a layer are found from the project too
            l.path = self.path
        for s in scenes:
            if s is src:
                continue
            share_shot(src, s)

    def save(self, path):
        path = Path(path)
        data = self.to_dict()
        if self.footage and self.footage.get('path'):
            # keep the absolute path and one relative to the project, so the footage is found again
            # whether the project stays put or moves together with its footage
            ab = Path(self.footage['path']).resolve()
            f = dict(self.footage, path=str(ab))
            try:
                f['rel'] = os.path.relpath(ab, path.parent.resolve())
            except ValueError:  # different drive
                f.pop('rel', None)
            data['footage'] = f
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(json.dumps(data, indent=1), encoding='utf-8')
        tmp.replace(path)
        self.path = str(path)
        for l in getattr(self, 'layers', None) or []:
            l.path = self.path

    @classmethod
    def load(cls, path):
        path = Path(path)
        s = cls.from_dict(json.loads(path.read_text(encoding='utf-8')))
        if s.footage and s.footage.get('path'):
            s.footage['path'] = relink(s.footage, path.parent)
        s.path = str(path)
        for l in getattr(s, 'layers', None) or []:
            l.path = s.path
        for d in s.emitters + s.colliders + s.fabrics:
            for key in ('mesh', 'volume'):
                m = d.get(key)
                if m and not m.startswith('builtin:') and Path(m).is_absolute() and not Path(m).exists():
                    alt = path.parent / Path(m).name
                    if alt.exists():
                        d[key] = str(alt.resolve())
        return s

    def copy(self):
        return copy.deepcopy(self)


def relink(footage, project_dir):
    """Find footage again: the stored path, else the path relative to the project, else the file
    name next to the project. Returns the best candidate (which may still be missing)."""
    project_dir = Path(project_dir)
    cands = [Path(footage['path'])]
    if not cands[0].is_absolute():
        cands.insert(0, project_dir / cands[0])
    if footage.get('rel'):
        cands.append(project_dir / footage['rel'])
    cands.append(project_dir / Path(footage['path']).name)
    for c in cands:
        if c.exists():
            return str(c.resolve())
    return str(cands[0])


# What every layer of a shot has in common (the footage and how it is read, the frame, the output look,
# the holdouts): set on one, set on all
SHARED_COMPOSITE = ('plate_transform', 'ocio_plate', 'plate_exposure', 'view', 'ocio_config', 'ocio_working', 'ocio_display',
                    'ocio_view', 'ocio_look', 'exr_space', 'knee', 'highlight_white', 'holdout_matte', 'matte_channel',
                    'matte_invert', 'holdout_depth', 'depth_kind', 'depth_scale', 'grain_match', 'grain', 'bg', 'bg_checker')


MATCHED_CAMERA = ('mode', 'position', 'rotation', 'focal_mm', 'sensor_mm', 'use_anchor', 'roll', 'near', 'far')


def share_shot(src, dst):
    """Copy the shot's shared settings from one layer to another. The track's points are shared; where each
    layer sits on them (the track's offset) is its own."""
    import copy as _copy
    dst.footage = _copy.deepcopy(src.footage)
    dst.roto = _copy.deepcopy(getattr(src, 'roto', []))
    dst.ground = _copy.deepcopy(getattr(src, 'ground', None))
    if dst.ground:   # a camera matched to the footage is the same for every layer; where each effect stands is its own
        for k in MATCHED_CAMERA:
            dst.data['camera'][k] = _copy.deepcopy(src.data['camera'][k])
    dst.data['render'] = _copy.deepcopy(src.data['render'])
    for k in SHARED_COMPOSITE:
        if k in src.data['composite']:
            dst.data['composite'][k] = _copy.deepcopy(src.data['composite'][k])
    if src.track and src.track.get('points'):
        off = (dst.track or {}).get('offset', src.track.get('offset', (0.0, 0.0)))
        dst.track = {'points': dict(src.track['points']), 'offset': off}
    else:
        dst.track = None


def track_point(track, frame):
    """Tracked anchor at a frame, linearly interpolated between tracked frames; None outside."""
    pts = track.get('points') or {}
    if not pts:
        return None
    if frame in pts:
        return pts[frame]
    fs = sorted(pts)
    if frame < fs[0] or frame > fs[-1]:
        return pts[fs[0]] if frame < fs[0] else pts[fs[-1]]
    import bisect
    i = bisect.bisect_left(fs, frame)
    f0, f1 = fs[i - 1], fs[i]
    t = (frame - f0) / (f1 - f0)
    a, b = pts[f0], pts[f1]
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
