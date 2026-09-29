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
from ..engine.embers import EmberParams
from ..engine.renderer import CompParams, LookParams
from ..engine.solver import MAX_COLLIDERS, MAX_EMITTERS, ColliderGPU, EmitterGPU, Solver, SolverParams, SpreadParams
from .anim import Curve
from .params import (COLLIDER_PARAMS, EMITTER_PARAMS, SECTIONS, SIM_SECTIONS, coerce, collider_defaults, defaults,
                     emitter_defaults, param)

FILE_VERSION = 1
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
        self.footage = None     # {'path', 'fps', 'offset'}
        self.track = None       # {'points': {frame: [x, y]}}
        self.preset = None
        self.notes = ''
        self.path = None

    # -- addressing ------------------------------------------------------------------------------
    # A path is ('section', 'key') or ('emitter', index, 'key') or ('collider', index, 'key').

    def _slot(self, path):
        if path[0] == 'emitter':
            return self.emitters[path[1]], path[2]
        if path[0] == 'collider':
            return self.colliders[path[1]], path[2]
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
        for group in (self.emitters, self.colliders):
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
            oxygen=feat['oxygen'], air_use=c['air_use'], air_mixing=c['air_mixing'], vapour=feat['vapour'], steam_yield=c['steam_yield'],
            water_yield=c['water_yield'], vapour_dissipation=self.v('combustion', 'vapour_dissipation', f),
            boil_temp=self.boil_temp(), water_douse=c['water_douse'], latent_heat=c['latent_heat'],
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
                creep=sp['creep'], smoulder=sp['smoulder'], smoulder_smoke=sp['smoulder_smoke']))

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
        vapour = vapour or water
        return {'oxygen': bool(oxygen), 'vapour': bool(vapour), 'chem': bool(chem), 'burn': burn,
                'aux': bool(oxygen or vapour), 'water': bool(water)}

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
        if p.startswith('builtin:'):
            return str(Path(__file__).resolve().parents[1] / 'assets' / 'meshes' / p[8:])
        q = Path(p)
        base = Path(self.path).parent if self.path else Path.cwd()
        if not q.is_absolute():
            q = base / q
        if not q.exists() and self.path:
            alt = base / Path(p).name
            if alt.exists():
                q = alt
        return str(q)

    def envelope(self, e, frame):
        t = self.seconds(frame)
        start = e['start']
        env = 1.0 if start <= -99 else _smoothstep(start, start + max(e['fade_in'], 1e-3), t)
        if e['stop'] >= 0:
            env *= 1.0 - _smoothstep(e['stop'], e['stop'] + max(e['fade_out'], 1e-3), t)
        return env

    def emitters_gpu(self, frame, embers_only=False):
        """Emitters at `frame`, which may be fractional (the solver evaluates them every substep so
        fast-moving emitters leave a continuous trail)."""
        out = []
        master = self.v('combustion', 'fuel_scale', frame)
        seed = self.data['domain']['seed']
        both = self.kind == 'both'
        for i, e in enumerate(self.emitters):
            if not e['enabled'] or (embers_only and not e['embers']):
                continue
            if both and e.get('emits') == 'liquid':
                continue
            g = lambda k: self.get(('emitter', i, k), frame)
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
            out.append(EmitterGPU(
                shape=e['shape'], pos=tuple(g('position')), size=tuple(g('size')), p1=tuple(g('end')), soft=e['softness'],
                fuel=g('fuel') * env * master, temp=g('temperature') * min(1.0, env * 4.0), smoke=g('smoke') * env,
                vel=tuple(float(x) for x in vel), radial=g('radial'), vel_blend=blend * env,
                noise=e['noise'], noise_freq=e['noise_freq'], noise_rise=e['noise_rise'], contrast=e['contrast'],
                seed=float(e['seed'] * 13 + seed * 7 + i), yaw=math.radians(g('yaw')),
                swirl=g('swirl') * env, swirl_width=e['swirl_width'], douse=g('douse') * env, vapour=g('vapour') * env,
                color=tuple(float(x) * amount for x in e['color']), thickness=e['thickness'],
                mesh=self.mesh_path(e['mesh']) if e['shape'] == 'mesh' else ''))
        return out[:MAX_EMITTERS]

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
                hollow=g('hollow'), opening=tuple(g('opening')), opening_at=tuple(g('opening_at')), holdout=bool(c['holdout'])))
        if overrides:
            import dataclasses
            idx = [i for i, c in enumerate(self.colliders) if c['enabled']]
            out = [dataclasses.replace(cg, **overrides[i]) if i in overrides else cg for i, cg in zip(idx, out)]
        return out[:MAX_COLLIDERS]

    def colliders_animated(self):
        return any(self.curve(('collider', i, k)) is not None for i, c in enumerate(self.colliders) if c['enabled']
                   for k in ('position', 'size', 'yaw', 'hollow', 'opening', 'opening_at'))

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
        return LiquidParams(
            gravity=q['gravity'], flip=q['flip'], ppc=int(q['ppc']), surface_tension=q['surface_tension'],
            wall_drag=q['wall_drag'], drying=q['drying'], pressure_iters=int(q['pressure_iters']),
            volume_correction=q['volume_correction'], seed=float(d['seed']),
            whitewater=bool(q['whitewater']), ww_rate=25.0 * q['ww_amount'], ww_min_speed=q['ww_min_speed'],
            ww_turbulence=q['ww_turbulence'], ww_crests=q['ww_crests'], foam_life=q['foam_life'],
            bubble_rise=q['bubble_rise'], viscosity=q['viscosity'], rho=q['liquid_density'],
            contact_angle=q['contact_angle'], water_level=q['water_level'] if d['open_sides'] else 0.0,
            level_absorb=q['level_absorb'], wind=self.liquid_wind(frame), wind_surface=q['wind_surface'],
            narrow_band=bool(q['narrow_band']), band_width=int(q['band_width']),
            open_sides=d['open_sides'], open_top=d['open_top'], ground=d['ground'])

    def whitewater_capacity(self):
        q = self.data['liquid']
        return int(q['ww_max'] * 1e6) if q['whitewater'] else 0

    def liquid_capacity(self, final=False):
        from ..engine.liquid import LiquidSolver
        dims, _, _ = self.sim_layout(final)
        q = self.data['liquid']
        return LiquidSolver.capacity_for(dims, int(q['ppc']), q['max_particles'] * 1e6)

    def sources_gpu(self, frame, filled=None):
        """Liquid sources at `frame` (fractional allowed). A volume ('fill') source pours once, the
        first time the shot reaches its start; `filled` holds the indices that already have (and is
        updated)."""
        from ..engine.liquid import source
        out = []
        t = self.seconds(frame)
        both = self.kind == 'both'
        for i, e in enumerate(self.emitters):
            if not e['enabled'] or (both and e.get('emits') != 'liquid'):
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
                              mesh=self.mesh_path(e['mesh']) if e['shape'] == 'mesh' else '', soft=e['softness']))
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
            env_strength=l['env_strength'], env_sun=bool(l['env_sun']), wind=self.liquid_wind(frame))

    def look(self, frame, final=False):
        s, l = self.data['shading'], self.data['lighting']
        amb = np.asarray(self.v('lighting', 'ambient', frame)) * self.v('lighting', 'ambient_intensity', frame)
        return LookParams(
            ambient_k=s['ambient_k'], flame_k=self.v('shading', 'flame_k', frame), max_k=s['max_k'],
            dynamic_range=s['dynamic_range'], intensity=s['intensity'], exposure=self.v('shading', 'exposure', frame),
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
            steam_albedo=tuple(s['steam_albedo']), multiple_scattering=s['multiple_scattering'])

    def comp(self, frame, mode='composite'):
        c = self.data['composite']
        return CompParams(
            mode=mode, view=c['view'], knee=c['knee'], plate_gain=2.0 ** c['plate_exposure'],
            plate_transform=c['plate_transform'], fire_gain=1.0, smoke_opacity=self.v('composite', 'smoke_opacity', frame),
            bloom=self.v('composite', 'bloom', frame), bloom_radius=c['bloom_radius'],
            light_cast=self.v('composite', 'light_cast', frame), haze=self.v('composite', 'haze', frame),
            haze_freq=c['haze_freq'], haze_speed=c['haze_speed'], saturation=c['saturation'], grain=c['grain'],
            bg=tuple(c['bg']), bg_checker=c['bg_checker'], tint=tuple(c['tint']),
            surface_light=self.v('composite', 'surface_light', frame), scorch=c['scorch'])

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

    def sim_signature(self, final=False):
        """Changes whenever anything that affects the simulation changes."""
        blob = {
            'sections': {s: {k: _to_json_value(v) for k, v in self.data[s].items()} for s in SIM_SECTIONS},
            'emitters': [{k: _to_json_value(v) for k, v in e.items() if k != 'name'} for e in self.emitters],
            'colliders': [{k: _to_json_value(v) for k, v in c.items() if k != 'name'} for c in self.colliders],
            'fps': self.fps, 'start': self.start, 'layout': [list(x) if isinstance(x, tuple) else x for x in self.sim_layout(final)],
            'embers': {k: _to_json_value(v) for k, v in self.data['embers'].items()},
            'fire_yaw': _to_json_value(self.data['camera']['fire_yaw']),
            'meshes': self._mesh_stamps(),
            # dousing cools gas to the boiling point, and steam condenses (releasing heat) against the
            # air's temperature and humidity: with steam these look settings change the simulation too
            'steam': ([round(self.boil_temp(), 5)] + [self.data['shading'][k] for k in ('ambient_k', 'flame_k', 'humidity')]
                      if self.features()['vapour'] else None),
            'upres': self.data['render']['upres'] if final else 1,
        }
        return hashlib.sha1(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def _mesh_stamps(self):
        """Modification times of the mesh files in use, so editing a mesh re-simulates."""
        out = {}
        for d in self.emitters + self.colliders:
            if d.get('shape') == 'mesh' and d.get('mesh'):
                p = self.mesh_path(d['mesh'])
                try:
                    out[p] = os.path.getmtime(p)
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
        for d in s.emitters + s.colliders:
            m = d.get('mesh')
            if m and not m.startswith('builtin:') and Path(m).is_absolute() and not Path(m).exists():
                alt = path.parent / Path(m).name
                if alt.exists():
                    d['mesh'] = str(alt.resolve())
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
