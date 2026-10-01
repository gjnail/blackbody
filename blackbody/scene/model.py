"""The scene: every setting of a shot, how it animates, and how it maps onto the engine."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from ..engine import camera as cam
from ..engine.lut import flame_ev
from ..engine.embers import EmberParams
from ..engine.renderer import CompParams, LookParams
from ..engine.solver import MAX_COLLIDERS, MAX_EMITTERS, VOLUME_MODES, ColliderGPU, EmitterGPU, Solver, SolverParams, SpreadParams
from ..engine.mesh import is_numbered, mesh_deforms, sequence_files, split_source
from .anim import Curve
from .params import (COLLIDER_PARAMS, EMITTER_PARAMS, LOOK_KEYS, SECTIONS, SIM_SECTIONS, coerce, collider_defaults, defaults,
                     emitter_defaults, fabric_defaults, light_defaults, param)

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


class Scene:
    def __init__(self):
        self.name = 'Untitled'
        self.data = {s: defaults(s) for s in SECTIONS}
        self.emitters = [dict(emitter_defaults(), name='Fuel bed')]
        self.colliders = []
        self.lights = []        # lamps in the set (they light the smoke; look only, not the simulation)
        self.fabrics = []       # cloth: curtains, flags, sheets (engine/cloth.py)
        self.footage = None     # {'path', 'fps', 'offset'}
        self.track = None       # {'points': {frame: [x, y]}}
        self.preset = None
        self.notes = ''
        self.path = None

    # -- addressing ------------------------------------------------------------------------------
    # A path is ('section', 'key') or ('emitter' / 'collider' / 'light' / 'fabric', index, 'key').

    def _slot(self, path):
        if path[0] == 'emitter':
            return self.emitters[path[1]], path[2]
        if path[0] == 'collider':
            return self.colliders[path[1]], path[2]
        if path[0] == 'light':
            return self.lights[path[1]], path[2]
        if path[0] == 'fabric':
            return self.fabrics[path[1]], path[2]
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
        for group in (self.emitters, self.colliders, self.lights, self.fabrics):
            for d in group:
                for v in d.values():
                    if isinstance(v, Curve):
                        out.update(v.frames())
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
                colour=tuple(float(x) for x in d['colour']), self_collide=bool(d['self_collide'])))
        return out[:MAX_FABRICS]

    def fabrics_at(self, frame):
        """Where the enabled fabrics' pins are at `frame` (fractional frames allowed)."""
        from ..engine.cloth import MAX_FABRICS, FabricPlace
        out = []
        t = self.seconds(frame)
        for i, d in enumerate(self.fabrics):
            if not d['enabled'] or (d['shape'] == 'mesh' and not d['mesh']):
                continue
            g = lambda k: self.get(('fabric', i, k), frame)
            rel = float(d['release'])
            out.append(FabricPlace(pos=tuple(float(x) for x in g('position')), yaw=math.radians(g('yaw')),
                                   scale=tuple(float(x) for x in d['scale']) if d['shape'] == 'mesh' else (1.0, 1.0, 1.0),
                                   released=d['pins'] == 'none' or (rel >= 0.0 and t >= rel)))
        return out[:MAX_FABRICS]

    def lamps(self, frame):
        """The enabled lights at `frame` for the renderer: fire-local position (m), unit aim, linear colour
        times intensity (lux at 1 m), radius (m), kind, cone cosines and whether smoke shadows them."""
        out = []
        for i, d in enumerate(self.lights):
            if not d['enabled']:
                continue
            g = lambda k: self.get(('light', i, k), frame)
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
            out.append(dict(kind=d['kind'], position=tuple(float(x) for x in g('position')), direction=tuple(float(x) for x in aim),
                            power=tuple(float(x) for x in col * max(float(g('intensity')), 0.0)),
                            radius=float(d['radius']), cos_outer=math.cos(half), cos_inner=math.cos(half * (1.0 - soft)),
                            shadows=bool(d['shadows']), in_footage=bool(d.get('in_footage', True))))
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

    def emitters_gpu(self, frame, embers_only=False, substeps=6):
        """Emitters at `frame`, which may be fractional (the solver evaluates them every substep so
        fast-moving emitters leave a continuous trail). substeps: how many steps the frame is taken in, so
        that steering the gas (puffing) is as firm per frame however many there are."""
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
            g = lambda k: self.get(('emitter', i, k), frame)
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
            speed = float(np.linalg.norm(motion))
            inherit = float(e['inherit'])
            vel = np.asarray(g('velocity'), float) + inherit * motion
            blend = max(float(e['vel_blend']), inherit * min(1.0, speed / 0.5))
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
                seed=float(e['seed'] * 13 + seed * 7 + i), yaw=math.radians(g('yaw')),
                swirl=g('swirl') * env, swirl_width=e['swirl_width'], douse=g('douse') * env, vapour=g('vapour') * env,
                color=tuple(float(x) * amount for x in e['color']), thickness=e['thickness'],
                mesh=self.item_source(e), volume_mode=VOLUME_MODES.get(e.get('volume_mode', 'fill'), 0) if e['shape'] == 'volume' else 0,
                **self._mesh_time(e, frame)))
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
            out.append(ColliderGPU(
                shape=c['shape'], pos=tuple(g('position')), size=tuple(g('size')), rot_y=math.radians(g('yaw')),
                vel=tuple(self.rate(('collider', i, 'position'), f)), spin=math.radians(self.rate(('collider', i, 'yaw'), f)),
                burnable=bool(c['burnable']), mesh=self.mesh_path(c['mesh']) if c['shape'] == 'mesh' else '',
                hollow=g('hollow'), opening=tuple(g('opening')), opening_at=tuple(g('opening_at')), holdout=bool(c['holdout']),
                **self._mesh_time(c, f)))
        if overrides:
            import dataclasses
            idx = [i for i, c in enumerate(self.colliders) if c['enabled']]
            out = [dataclasses.replace(cg, **overrides[i]) if i in overrides else cg for i, cg in zip(idx, out)]
        return out[:MAX_COLLIDERS]

    def colliders_animated(self):
        return (any(self.curve(('collider', i, k)) is not None for i, c in enumerate(self.colliders) if c['enabled']
                    for k in ('position', 'size', 'yaw', 'hollow', 'opening', 'opening_at'))
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
        return [(i, float(c['density'])) for i, c in enumerate(self.colliders) if c['enabled'] and c.get('floating')]

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
            out.append(source(shape=e['shape'], pos=tuple(g('position')), size=tuple(g('size')), p1=tuple(g('end')),
                              vel=tuple(float(x) for x in vel), radial=g('radial'), strength=strength, fill=fill,
                              jitter=e['jitter'], vel_blend=0.0 if fill else e['vel_blend'], yaw=math.radians(g('yaw')),
                              dye=tuple(e.get('dye', (1.0, 1.0, 1.0))), dye_amount=float(e.get('dye_amount', 0.0)),
                              dye_cloud=float(e.get('dye_cloud', 0.5)),
                              density=(float(e.get('liquid_density', 0.0)) / max(float(ref), 1.0)
                                       if e.get('liquid_density', 0.0) > 0 else 1.0),
                              mesh=self.mesh_path(e['mesh']) if e['shape'] == 'mesh' else '', soft=e['softness'],
                              temp=float(e.get('liquid_temp', 20.0)) if e.get('temp_own') else None))
        return out[:MAX_EMITTERS]

    def water_look(self, frame, final=False):
        from ..engine.liquid_render import WaterLook
        w, l = self.data['water'], self.data['lighting']
        sky = np.asarray(self.v('lighting', 'ambient', frame)) * self.v('lighting', 'ambient_intensity', frame)
        sun = np.asarray(l['sun_color']) * (self.v('lighting', 'sun_intensity', frame) if l['sun_on'] else 0.0)
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
            environment=self.mesh_path(l['environment']) if l['environment'] else '', env_rotation=l['env_rotation'],
            env_strength=l['env_strength'], env_sun=bool(l['env_sun']), wind=self.liquid_wind(frame),
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
        return LookParams(
            ambient_k=s['ambient_k'], flame_k=self.v('shading', 'flame_k', frame), max_k=s['max_k'],
            dynamic_range=s['dynamic_range'], intensity=s['intensity'],
            exposure=self.v('shading', 'exposure', frame) + self.physical_ev(frame), colour_response=s['colour_response'],
            flame_density=s['flame_density'], flame_sharpness=s['flame_sharpness'], flame_threshold=s['flame_threshold'],
            soot_glow=s['soot_glow'], flame_absorption=s['flame_absorption'], flame_occlusion=s['flame_occlusion'], blue=s['blue'], blue_color=tuple(s['blue_color']),
            ignition=self.data['combustion']['ignition'], smoke_albedo=tuple(s['smoke_albedo']),
            smoke_density=self.v('shading', 'smoke_density', frame), ambient=tuple(amb), anisotropy=s['anisotropy'],
            sun_color=tuple(l['sun_color']), sun_intensity=self.v('lighting', 'sun_intensity', frame) if l['sun_on'] else 0.0,
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
            'emitters': [{k: _to_json_value(v) for k, v in e.items() if k != 'name'} for e in self.emitters],
            'colliders': [{k: _to_json_value(v) for k, v in c.items() if k != 'name'} for c in self.colliders],
            'fabrics': [{k: _to_json_value(v) for k, v in f.items() if k != 'name'} for f in self.fabrics],
            'fps': self.fps, 'start': self.start, 'layout': [list(x) if isinstance(x, tuple) else x for x in self.sim_layout(final)],
            'embers': {k: _to_json_value(v) for k, v in self.data['embers'].items()},
            'fire_yaw': _to_json_value(self.data['camera']['fire_yaw']),
            'meshes': self._mesh_stamps(),
            # dousing cools gas to the boiling point, and steam condenses (releasing heat) against the
            # air's temperature and humidity: with steam these look settings change the simulation too
            'steam': ([round(self.boil_temp(), 5)] + [self.data['shading'][k] for k in ('ambient_k', 'flame_k', 'humidity')]
                      if self.features()['vapour'] else None),
            'upres': self.upres_for(final),
        }
        return hashlib.sha1(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def _mesh_stamps(self):
        """Modification times of the mesh files in use, so editing a mesh re-simulates."""
        from ..engine.mesh import MeshLibrary
        out = {}
        for d in self.emitters + self.colliders + [f for f in self.fabrics if f.get('shape') == 'mesh']:
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
            'footage': self.footage,
            'track': ({'points': {str(k): list(v) for k, v in self.track.get('points', {}).items()},
                       'offset': list(self.track.get('offset', (0, 0)))} if self.track else None),
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
                s.colliders.append(x)
        s.lights = []
        for l in d.get('lights', []):
            x = light_defaults()
            for k, v in l.items():
                if k in x:
                    x[k] = _from_json_value(param('light', k), v)
            s.lights.append(x)
        s.fabrics = []
        for f in d.get('fabrics', []):
            x = fabric_defaults()
            for k, v in f.items():
                if k in x:
                    x[k] = _from_json_value(param('fabric', k), v)
            s.fabrics.append(x)
        s.footage = d.get('footage')
        tr = d.get('track')
        if tr and tr.get('points'):
            s.track = {'points': {int(float(k)): tuple(v) for k, v in tr['points'].items()},
                       'offset': tuple(tr.get('offset', (0, 0)))}
        return s

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

    @classmethod
    def load(cls, path):
        path = Path(path)
        s = cls.from_dict(json.loads(path.read_text(encoding='utf-8')))
        if s.footage and s.footage.get('path'):
            s.footage['path'] = relink(s.footage, path.parent)
        s.path = str(path)
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
