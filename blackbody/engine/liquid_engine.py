"""The engine's liquid path: preparing, stepping, caching, rendering and reporting a liquid scene.

Engine dispatches here when the scene's kind is 'liquid'. Frames are cached as packed particles
(16 bytes each) plus the ground wetness, and the surface is rebuilt from them at render time, so
changing the look re-renders any cached frame without re-simulating.
"""
from __future__ import annotations

import logging
import math
import time

import numpy as np

from types import SimpleNamespace

from . import camera as cam
from .cloth import STEPS_PER_SECOND
from .gpu import Uniforms
from .liquid import LiquidSolver
from .liquid_float import Floats
from .solids import Solids, attached
from .liquid_render import LiquidRenderer, LiquidView
from . import stage as stage_mod
from .renderer import INPUT_TRANSFORMS

log = logging.getLogger('blackbody.liquid')


def _halton(i, b):
    f, r = 1.0, 0.0
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r


class LiquidEngine:
    """Mixin for Engine (which owns gpu, solver, renderer, cache and the accumulation kernels)."""

    liquid = None
    liquid_r = None
    kind = 'fire'
    _rwet = None
    _floats = None
    _floats_ready = False
    weather = None          # precipitation (engine/weather.py), when the scene has it
    weather_r = None
    _wx_on = False
    _wx_surf = None         # (the surface map's maximum height, a host copy of it for the cache)

    def _liq(self):
        if self.liquid is None:
            self.liquid = LiquidSolver(self.gpu, self.solver.meshes)
            self.liquid_r = LiquidRenderer(self.gpu, self.renderer)
        return self.liquid

    def _prepare_liquid(self, scene, final=False, soft=False):
        sig = scene.sim_signature(final)
        dims, h, origin = scene.sim_layout(final)
        L = self._liq()
        # (meshes, and the fields of volumes that pour liquid)
        meshes = [scene.item_source(d) for d in scene.emitters + scene.colliders if d['enabled'] and scene.item_source(d)]
        if self.solver.set_meshes(meshes, scene.data['domain']['mesh_resolution'], cell=h):
            L.colliders = None  # the atlas changed: rewrite the solid distance field
        changed = L.configure(dims, h, origin, scene.liquid_capacity(final), scene.whitewater_capacity())
        L._prm = scene.liquid_params(scene.start)
        L.set_colliders(scene.colliders_gpu(scene.start))
        self._prepare_weather(scene, final)
        if self.cloth.configure(scene.fabric_specs()):
            changed = True
        if self.solids.configure(scene, (dims, h, origin)):
            changed = True
        if self._prepare_matter(scene, (dims, h, origin)):
            changed = True
        if self._prepare_strands(scene):
            changed = True
        if self._prepare_heat(scene):
            changed = True
        if self.kind != 'liquid':
            self.kind = 'liquid'
            changed = True
        if changed or final != self.final or self.sig is None:
            self.sig = sig
            self.final = final
            self.cache.clear()
            self.reset()
            return 'reset'
        if sig != self.sig:
            self.sig = sig
            self.cache.clear()
            if soft and self.sim_frame is not None:
                return 'soft'
            self.reset()
            return 'reset'
        return None

    _olayer = None
    _lview = None

    def _ocean_layer(self, scene):
        """The open water round the box (ocean_layer.OceanLayer), or None where there is none."""
        q, d = scene.data['liquid'], scene.data['domain']
        area = float(q.get('open_water_area', 6.0))
        if area <= 0.0 or not d['open_sides'] or scene.liquid_level(scene.start) <= 0.0:
            if self._olayer is not None:
                self._olayer.on = False
            return None
        if self._olayer is None:
            from .ocean_layer import OceanLayer
            self._olayer = OceanLayer(self.gpu, self.solver.meshes)
        return self._olayer

    def _step_ocean_layer(self, b, scene, frame, prm, fdt):
        layer = self._ocean_layer(scene)
        if layer is None:
            return
        q = scene.data['liquid']
        L = self.liquid
        n = 512 if int(q.get('ocean_detail', 256)) >= 512 else 256
        layer.place(L, float(q.get('open_water_area', 6.0)), n)
        wind = prm.wind
        spec = prm.ocean
        sea_depth = float(q.get('ocean_depth', 0.0))
        bed = bool(scene.data['domain']['ground']) and not (sea_depth > prm.water_level)
        layer.step(b, L, prm, fdt, spec, wind=(wind[0], wind[2]),
                   foam_life=float(scene.data['water'].get('sea_foam_life', 12.0)), ground_is_bed=bed,
                   sea_depth=sea_depth if sea_depth > 0.0 else None)

    def _prepare_weather(self, scene, final=False):
        """Precipitation for scenes that have it (Weather section)."""
        self._wx_on = bool(scene.has_weather())
        if not self._wx_on:
            return
        if self.weather is None:
            from .weather import Weather
            self.weather = Weather(self.gpu, self.solver.meshes)
        self.weather.configure(scene.weather_params(scene.start, final))

    def _step_weather(self, b, scene, frame, wprm, fs, dt, moving):
        """One substep of the precipitation, after the liquid's."""
        W = self.weather
        if self._wx_surf is None or moving:
            W.surface(b, wprm, self.liquid)
            self._wx_surf = 'pending'
        W.time = scene.seconds(fs)
        W.step(b, dt, wprm, self.liquid)

    def _weather_surface_ready(self):
        """After the first surface pass: its highest point (for the renderer) and a host copy (for the cache)."""
        if self._wx_surf == 'pending':
            surf = self.weather.read_surface()
            top = float(np.nanmax(np.where(surf[..., 0] > -100.0, surf[..., 0], 0.0))) if surf.size else 0.0
            self._wx_surf = (top, surf)

    def _weather_view(self, scene, frame, live, entry=None):
        """What the weather renderer draws for this frame (live or from the cache), or None."""
        if not scene.has_weather() or self.weather is None:
            return None
        from .weather_render import WeatherRenderer, WeatherView
        if self.weather_r is None:
            self.weather_r = WeatherRenderer(self.gpu, self.renderer)
        W = self.weather
        top = self._wx_surf[0] if isinstance(self._wx_surf, tuple) else 0.0
        yr = (-0.05, top + 0.6)
        look = scene.weather_look()
        haze = W.haze(scene.weather_params(frame))
        if live:
            return WeatherView(W.packed, W.packed_count, W.COVER, W.SURF, W.area, W.map_dims, yr, look, haze)
        if entry is None or 'wx' not in entry:
            return None
        cover = entry.get('wx_cover')
        surf = self._wx_surf[1] if isinstance(self._wx_surf, tuple) else None
        buf, n, ctex, stex = self.weather_r.upload(entry['wx'], None if cover is None else cover.astype(np.float32), surf)
        return WeatherView(buf, n, ctex, stex, tuple(entry.get('wx_area', W.area)), W.map_dims, yr, look, haze)

    def _reset_liquid(self):
        if self.weather is not None:
            self.weather.reset()
        self._wx_surf = None
        if self._olayer is not None:
            self._olayer.on = False
        self._filled = set()
        self._floats = None
        self._floats_ready = False
        self.cloth.reset()
        if self.liquid is not None and self.liquid.dims is not None:
            self.liquid.reset()

    def _ocean_spec(self, scene, frame):
        """The sea on the open water at `frame` (None when flat or without a water level)."""
        from .ocean import OCEAN_KEYS, spec_from
        d = scene.data['domain']
        q = scene.data['liquid']
        if not d['open_sides']:
            return None
        vals = {k: scene.v('liquid', k, frame) if k in q else None for k in
                OCEAN_KEYS}
        vals = {k: v for k, v in vals.items() if v is not None}
        return spec_from(vals, scene.liquid_level(frame), scene.liquid_wind(frame), seed=int(d['seed']),
                         yaw=scene.v('camera', 'fire_yaw', frame))

    @staticmethod
    def _sea_sides(scene, spec):
        """The open sides (-x, +x, -z, +z) the open water flows through (1): all of them, or (Waves come
        from the upwave side) those the waves come in through, as in a wave flume. The others are walls
        all the way up (a crest, a run-up, the water a breaking wave throws over a reef would spill out
        of the box, which would drain with every wave; it flows back out to sea along the bottom, as
        undertow): -1 along the waves, -2 past the shore. With a surge the side past the shore is a wall
        under the level only (0): the flood pours on inland over it."""
        if scene.data['liquid'].get('sea_from', 'all') != 'upwave' or spec is None or not spec.on:
            return (1.0, 1.0, 1.0, 1.0)
        # the waves' direction: the strongest of the wind sea, the swell and the surge
        cand = [(spec.height, spec.direction), (spec.swell, spec.swell_dir), (spec.surge, spec.surge_dir)]
        ang = math.radians(max(cand, key=lambda c: c[0])[1])
        dx, dz = math.sin(ang), math.cos(ang)
        # a side lets the sea in when the waves travel into the box through it
        # (-1 along the waves: the renderer mirrors the box's water past them, liq_march.wgsl mir;
        # -2 past the shore, a wall all the way up too, not mirrored: there is land past it)
        past = 0.0 if spec.surge > 1e-4 else -2.0
        sides = [1.0 if o > 0.3 else (past if o < -0.3 else -1.0) for o in (dx, -dx, dz, -dz)]
        # and a current (a river under a tidal bore) flows in through one side and out through another
        cur = scene.liquid_current(scene.start)
        speed = math.hypot(cur[0], cur[2])
        if speed > 1e-3:
            for i, o in enumerate((cur[0], cur[0], cur[2], cur[2])):
                if abs(o) > 0.3 * speed:
                    sides[i] = 1.0
        return tuple(sides)

    def _step_liquid(self, scene, frame):
        t0 = time.perf_counter()
        fdt = scene.v('domain', 'time_scale', frame) / scene.fps
        d = scene.data['domain']
        prm = scene.liquid_params(frame)
        if frame < scene.start and scene.data['liquid']['settle']:
            prm.damping = 6.0   # still liquid starts still: calm the filling transient before the shot
        L = self.liquid
        prm.ocean = self._ocean_spec(scene, frame)
        prm.sea_sides = self._sea_sides(scene, prm.ocean)
        tide = scene.curve(('liquid', 'water_level')) is not None and d['open_sides']
        if prm.ocean is not None:
            prm.damping = 0.0   # a sea is never still: settling would only calm its waves in the box
        prm.rain, prm.rain_drop = self._rain(scene, frame)
        L._prm = prm
        L.force_hook = self.cloth.liquid_hook if self.cloth.active else None   # (cloth pushes the water back)
        # surface tension can demand more substeps than the domain allows: stability comes first
        hi = min(40, max(d['substeps_max'], L.capillary_substeps(fdt)))
        n = L.substeps_for(fdt, cfl=d['cfl'], lo=d['substeps_min'], hi=hi)
        # rigid bodies (falling and floating objects) move through the frame first, pushed by the liquid as it
        # was measured over the frame before; the liquid then sees them where they are at each substep
        solids = self.solids if self.solids.active else None
        self._push_matter(L)
        self._cloth_meets_matter(fdt)   # (fabric and sand, snow, mud: each the other's surface this frame)
        if solids:
            self._objects_meet_cloth(fdt)
            self._shots_ahead(scene, fdt)     # (bullets: where their ways cross the water, the sand, snow and mud)
        poses = (solids.advance(scene, frame, fdt, n, couple=self._matter_couple(scene, frame, fdt, self.solver.meshes))
                 if solids else None)
        if solids:
            self._shots_kick(fdt)                # (and what they did to them)
        if solids:
            self._cloth_takes_objects(fdt)
        # the objects warm and cool (objheat_engine.py); the water and the weather meet them at their own temperatures
        lava_k = float(scene.water_look(frame).glow_temp) if prm.cooling > 0.0 else 0.0
        if self._heat_on():
            self._objheat.liquid_lava_k = lava_k
        self._step_heat(scene, frame, fdt, liquid=L, water_thermal=bool(prm.thermal))
        prm.collider_temps = self._object_temps_c(scene, prm.collider_temps)
        moving = scene.colliders_animated() or bool(solids)
        filled = getattr(self, '_filled', None)
        if filled is None:
            filled = self._filled = set()
        shift = self._follow_shift(scene, frame - 1, solids)
        wprm = scene.weather_params(frame) if (self._wx_on and self.weather is not None) else None
        if wprm is not None:
            wprm.collider_temps = self._object_temps_c(scene, wprm.collider_temps)
        cloth = self.cloth.active
        if cloth:
            # fabric in the liquid (cloth.py): its drag and buoyancy, and it soaks
            clook = scene.look(frame)
            if not self.cloth.placed:
                self.cloth.place(scene.fabrics_at(frame - 1), clook.ambient_k)
            cprm = SimpleNamespace(wind=tuple(scene.liquid_wind(frame)), ground=bool(d['ground']))
            cloth_steps = max(1, int(math.ceil(fdt * STEPS_PER_SECOND / n)))
        pieces = poses is not None and self._pieces_for(scene, L, 'liquid')
        grass = self.strands_on
        if grass:
            # grass on the banks: the wind (there is no gas in a liquid scene)
            glook = scene.look(frame)
            gwind = tuple(scene.liquid_wind(frame))
            ggust = float(scene.data['weather'].get('gust', 0.3)) if 'weather' in scene.data else 0.3
        with self.gpu.batch() as b:
            self._build_radiant(b, scene, frame, box=L)   # (what radiates heat this frame: radiant.py)
            msolid = self._matter_solid(b, L, 'liquid')   # (sand, snow and mud: solid to the water)
            if shift != (0, 0):
                L.shift(b, *shift, prm)
            self._footage_solid(b, scene, frame)
            regions = solids.regions(scene) if solids else []
            if regions:
                L.clear_float(b)
            for i in range(n):
                fs = frame - 1 + (i + 0.5) / n
                srcs = scene.sources_gpu(fs, filled)
                cols = scene.colliders_gpu(fs, poses[i] if poses else None) if moving else None
                prm.clock = scene.seconds(fs)
                if tide:
                    prm.water_level = scene.liquid_level(fs)   # a tide: the level at each substep
                L.pieces_step = self._solids_step(self.body_field_for('liquid') if pieces else None, msolid, i)
                L.step(b, fdt / n, prm, srcs, cols)
                L.pieces_step = None
                if regions:
                    L.float_forces(b, regions, i, fdt / n)
                if cloth:
                    carried = attached(scene, 'fabric', poses[i] if poses else None)
                    self.cloth.step(b, None, fdt / n, scene.fabrics_at(fs + 0.5 / n, moved=carried), cprm, clook,
                                    list(cols) if cols is not None else list(L.colliders or []), self.solver.meshes,
                                    steps=cloth_steps, liquid=L)
                if grass:
                    self._step_strands(b, scene, fs, fdt / n, None, list(cols) if cols is not None else list(L.colliders or []),
                                       self.solver.meshes, glook, bool(d['ground']), L.origin[1], gwind, ggust)
                if wprm is not None:
                    self._step_weather(b, scene, frame, wprm, fs, fdt / n, moving)
            L.pack(b)
            if wprm is not None:
                self.weather.pack(b)
            self._step_ocean_layer(b, scene, frame, prm, fdt)
        L.measure()
        self._water_heat_back(L)        # (what the water took from each object: objheat_engine.py)
        if lava_k > 0.0:
            self._liquid_radiates(L, lava_k, float(scene.look(frame).ambient_k))   # (the scene's own lava glows)
        if wprm is not None:
            self.weather.measure()
            self._weather_surface_ready()
        if regions:
            solids.liquid_measures(L.read_float(len(regions), n), fdt, n, L.h, prm.rho)
        self._step_matter(scene, frame, fdt, poses, n, self.solver.meshes)
        self._melt_matter(scene, frame, fdt, None, L)
        self._heat_matter(scene, frame, fdt, None, L)
        self._heat_back()
        self._wet_matter(fdt, L)
        self.sim_frame = frame
        self.last_substeps = n
        self.last_step_ms = (time.perf_counter() - t0) * 1000.0

    def _follow_shift(self, scene, frame, floats=None):
        """With Box follows on: the whole cells (x, z) to move the box by so it is centred on what it
        follows again, once that has drifted a couple of cells off centre."""
        mode = scene.data['liquid'].get('follow', 'off')
        L = self.liquid
        if mode == 'off' or not scene.data['domain']['open_sides'] or not L.dims:
            return 0, 0
        n, h, o = L.dims, L.h, L.origin
        if mode == 'liquid':
            if L.bbox is None:
                return 0, 0
            lo, hi = L.bbox
            cx, cz = o[0] + 0.5 * (lo[0] + hi[0]) * h, o[2] + 0.5 * (lo[2] + hi[2]) * h
        else:
            cols = scene.colliders_gpu(frame, floats.overrides() if floats else None)
            if not cols:
                return 0, 0
            cx, cz = float(cols[0].pos[0]), float(cols[0].pos[2])
        ex, ez = cx - (o[0] + 0.5 * n[0] * h), cz - (o[2] + 0.5 * n[2] * h)
        # a few cells at a time: a sudden jump would drop the box's sides into moving water
        dx = int(round(ex / h)) if abs(ex) > 2.0 * h else 0
        dz = int(round(ez / h)) if abs(ez) > 2.0 * h else 0
        return dx, dz

    def _footage_holdout(self, scene):
        """The scene's holdout matte / depth pass / roto shapes reader (io/holdout.py), kept while they stay the same;
        None without them."""
        import json
        c = scene.data['composite']
        roto = getattr(scene, 'roto', None) or []
        key = (id(scene), c.get('holdout_matte', ''), c.get('holdout_depth', ''), c.get('matte_channel'),
               c.get('matte_invert'), int((scene.footage or {}).get('offset', 0)), json.dumps(roto, sort_keys=True))
        if getattr(self, '_fh_key', None) != key:
            old = getattr(self, '_fh', None)
            if old is not None:
                old.close()
            self._fh = None
            if key[1] or key[2] or roto:
                from ..io.holdout import FootageHoldout
                fh = FootageHoldout(scene)
                self._fh = fh if fh.active else None
            self._fh_key = key
        return self._fh

    def _set_liquid_holdout(self, scene, frame):
        """The footage's holdouts for rendering the liquid at `frame` (Renderer.set_holdout)."""
        fh = self._footage_holdout(scene)
        hold = (None, None)
        if fh is not None:
            try:
                hold = fh.read(frame)
            except Exception as ex:   # a missing frame of the pass: render without it
                log.warning('Holdout frame %s unavailable: %s', frame, ex)
        self.renderer.set_holdout(*hold)

    def _footage_env(self, scene, wlook, frame, plate, cs, plate_fit, wanted=True):
        """With Lighting › Environment from the footage, the footage as the environment round the set (footage_env.py) in
        the water look, in place of the physical sky, as a fire scene has it (Engine._environment). True when it is. Its
        HDRI is built only when it is `wanted` (the set is drawn) or the key light comes from it; else the look has none
        (its light on the liquid is Match ambient to footage's either way)."""
        from . import footage_env as FE
        if not FE.applies(scene, plate, cs.view_proj):
            return False
        wlook.env_sun = bool(scene.data['lighting'].get('env_sun'))
        wlook.sky_image = FE.of_scene(scene, frame, plate, cs.view_proj, plate_fit) if (wanted or wlook.env_sun) else None
        wlook.env_strength, wlook.env_rotation = 1.0, 0.0   # (its strength in it, in the world's frame already)
        return True

    def _footage_solid(self, b, scene, frame):
        """With Hits the footage on, make the footage's surfaces (its depth pass) solid for the liquid."""
        L = self.liquid
        fh = self._footage_holdout(scene) if scene.data['liquid'].get('footage_collide') else None
        depth = None
        if fh is not None and fh.depth is not None:
            try:
                depth = fh.read(frame)[1]
            except Exception as ex:
                log.warning('Depth pass frame %s unavailable: %s', frame, ex)
        if depth is None:
            L.set_footage(b, None)
            return
        h, w = depth.shape[:2]
        tex = getattr(self, '_sim_depth', None)
        if tex is None or tex.size[:2] != (w, h):
            if tex is not None:
                tex.destroy()
            tex = self._sim_depth = self.gpu.texture2d(w, h, 'rgba16float', 'liquid-footage-depth')
        img = np.zeros((h, w, 4), np.float32)
        img[..., 1] = np.clip(np.nan_to_num(np.asarray(depth, np.float32), nan=0.0, posinf=0.0, neginf=0.0), 0.0, 65000.0)
        self.gpu.upload(tex, img.astype(np.float16))
        rw, rh = scene.data['render']['width'], scene.data['render']['height']
        spec, fire = scene.camera(frame)
        cs = cam.compute(spec, rw / max(rh, 1), fire)
        comp = scene.comp(frame, 'composite')
        L.set_footage(b, tex, cs, fire, comp.depth_kind, comp.depth_scale)

    @staticmethod
    def _rain(scene, frame):
        """Rain (mm/h, drop diameter mm) at this frame."""
        q = scene.data['liquid']
        if 'rain' not in q:
            return 0.0, 2.5
        return float(scene.v('liquid', 'rain', frame)), float(q.get('rain_drop', 2.5))

    def _snapshot_liquid(self):
        entry = {'liq': self.liquid.read_packed(), 'wet': self.liquid.read_wet(), 'ww': self.liquid.read_ww_packed(),
                 'origin': tuple(self.liquid.origin)}
        dye = self.liquid.read_dye()
        if dye is not None:
            entry['dye'] = dye
        crust = self.liquid.read_crust()
        if crust is not None:
            entry['crust'] = crust
        ice = self.liquid.read_ice()
        if ice is not None:
            entry['ice'] = ice
        band = self.liquid.read_band()
        if band is not None:
            entry['band'] = band
        if self.solids.active:
            entry['solids'] = self.solids.state()
        if self._olayer is not None and self._olayer.on:
            entry['sea_layer'] = self._olayer.read()
        if self._wx_on and self.weather is not None:
            entry['wx'] = self.weather.read_packed()
            cov = self.weather.read_cover()
            if cov is not None:
                entry['wx_cover'] = cov.astype(np.float16)
            entry['wx_area'] = tuple(self.weather.area)
        if self.kind == 'liquid' and self.cloth.active:
            for k, v in (self.cloth.snapshot() or {}).items():
                entry['cloth_' + k] = v
        return entry

    def floating_overrides(self, frame):
        """Where the rigid bodies (falling and floating objects) are at `frame`, as Scene.colliders_gpu
        overrides: None if there are none, or the frame is neither live nor cached. Any kind of scene."""
        solids = getattr(self, 'solids', None)
        if self.sim_frame == frame and solids is not None and solids.active:
            return solids.overrides()
        entry = self.cache.get(frame) if self.cache is not None else None
        if entry is not None and entry.get('solids'):
            return Solids.overrides_from(entry['solids'])
        if entry is not None and entry.get('floats'):
            return Floats.overrides_from(entry['floats'])   # (a cache from before rigid bodies)
        return None

    body_overrides = floating_overrides

    def piece_poses(self, frame):
        """The pieces of broken (breakable) objects at `frame`: {collider index: Solids.piece_poses entry}, or None."""
        solids = getattr(self, 'solids', None)
        if self.sim_frame == frame and solids is not None and (solids.sets or solids.asms):
            return solids.piece_poses() or None
        entry = self.cache.get(frame) if self.cache is not None else None
        st = entry.get('solids') if entry is not None else None
        if isinstance(st, dict) and st.get('pieces'):
            return st['pieces']
        return None

    def rope_poses(self, frame):
        """The ropes and springs at `frame`: {collider index: Solids.rope_poses entry}, or None."""
        solids = getattr(self, 'solids', None)
        if self.sim_frame == frame and solids is not None and solids.joints:
            return solids.rope_poses() or None
        entry = self.cache.get(frame) if self.cache is not None else None
        st = entry.get('solids') if entry is not None else None
        if isinstance(st, dict) and st.get('ropes'):
            return st['ropes']
        return None

    def _liquid_view(self, scene, frame):
        L = self.liquid
        ppc = int(scene.data['liquid']['ppc'])
        d = scene.data['domain']
        q = scene.data['liquid']
        bounds = (bool(d['open_sides']), bool(d['open_top']), not d['ground'])
        spec = self._ocean_spec(scene, frame)
        extra = dict(colliders=tuple(scene.colliders_gpu(frame, self.floating_overrides(frame))), meshes=self.solver.meshes,
                     level=scene.liquid_level(frame) if d['open_sides'] else 0.0, level_blend=q['level_absorb'],
                     ocean=spec, time=scene.seconds(frame), current=scene.liquid_current(frame),
                     sea_sides=self._sea_sides(scene, spec))
        if self.sim_frame == frame:
            layer = self._olayer if (self._olayer is not None and self._olayer.on) else None
            top = float(L.bbox[1][1]) if L.bbox is not None else -1.0
            return LiquidView(L.packed, L.packed_count, L.WET, L.dims, L.h, L.origin, ppc, L.wpacked, L.ww_count, bounds,
                              band=L.band_buffer, dye=L.dye_buffer, crust=L.crust_field, ice=L.ice_buffer, layer=layer,
                              liquid_top=top, **extra), True
        entry = self.cache.get(frame)
        if entry is None or 'liq' not in entry:
            return None, False
        buf = self.liquid_r.upload_particles(entry['liq'])
        nx, nz = L.dims[0], L.dims[2]
        if self._rwet is None or self._rwet.size[:2] != (nx, nz):
            if self._rwet is not None:
                self._rwet.destroy()
            self._rwet = self.gpu.texture2d(nx, nz, 'r32float', 'cached-wet')
        self.gpu.upload(self._rwet, entry['wet'].astype(np.float32)[..., None])
        ww = entry.get('ww')
        wbuf = self.liquid_r.upload_whitewater(ww) if ww is not None and len(ww) else None
        band = entry.get('band')
        bbuf = self.liquid_r.upload_band(band, int(np.prod(L.dims))) if band is not None else None
        dye = entry.get('dye')
        dbuf = self.liquid_r.upload_dye(dye) if dye is not None and len(dye) == len(entry['liq']) else None
        crust = entry.get('crust')
        cbuf = self.liquid_r.upload_crust(crust) if crust is not None and np.ndim(crust) == 4 else None
        ice = entry.get('ice')
        ibuf = self.liquid_r.upload_ice(ice) if ice is not None and len(ice) == len(entry['liq']) else None
        layer = None
        if entry.get('sea_layer'):
            if self._lview is None:
                from .ocean_layer import LayerView
                self._lview = LayerView(self.gpu)
            layer = self._lview.load(entry)
        from .liquid_render import liquid_top_of
        # (with the narrow band the deep liquid has no particles, but it lies under those that are)
        return LiquidView(buf, len(entry['liq']), self._rwet, L.dims, L.h, tuple(entry.get('origin', L.origin)), ppc, wbuf,
                          len(ww) if wbuf is not None else 0, bounds, band=bbuf, dye=dbuf, crust=cbuf, ice=ibuf, layer=layer,
                          liquid_top=liquid_top_of(entry['liq'], L.dims), **extra), False

    def _render_liquid(self, scene, frame, out_size, mode='composite', final=False, samples=1, motion_blur=False,
                       fire_scale=1.0, plate=None, plate_fit=(1.0, 1.0), seed=None):
        t0 = time.perf_counter()
        vol, live = self._liquid_view(scene, frame)
        if vol is None:
            raise RuntimeError(f'frame {frame} is neither simulated nor cached')
        W, H = out_size
        fw, fh = max(8, int(round(W * fire_scale))), max(8, int(round(H * fire_scale)))
        spec, fire = scene.camera(frame)
        cs = cam.compute(spec, W / H, fire)
        look = scene.water_look(frame, final)
        look.rain, look.rain_drop = self._rain(scene, frame)
        if plate is not None and scene.data['lighting'].get('ambient_from_footage', True):
            look.sky = self.footage_ambient(plate, scene, frame)
        comp = scene.comp(frame, mode)
        shutter = 0.0
        if motion_blur:
            shutter = scene.data['render']['shutter_angle'] / 360.0 / scene.fps * scene.v('domain', 'time_scale', frame)
        r = self.renderer
        if plate is not None:
            r.set_plate(plate)
        elif mode == 'composite':
            r.set_plate(None)
        ptex = r.plate if plate is not None else None
        self._set_liquid_holdout(scene, frame)
        base_seed = frame * 64 if seed is None else seed
        samples = max(1, int(samples))
        t = scene.seconds(frame)
        ground = scene.data['domain']['ground']
        LR = self.liquid_r
        # the set drawn in CG behind and under the liquid (stage.py): the liquid takes it as its footage
        footage = plate is not None
        objects = look.colliders_look != 'shaded'
        pieces = self.piece_poses(frame)   # (broken things: drawn, and holding out what is behind them, in every view)
        ropes = self.rope_poses(frame)
        matter = self.matter_for(frame) if mode == 'composite' else None
        bolts = scene.bolts(frame) if mode == 'composite' and scene.lights else None   # (lightning)
        shots = self.shot_view(frame) if mode == 'composite' else None    # (bullets: debris, sparks, holes)
        stage_on = (stage_mod.wanted(scene, footage, mode, objects=objects) or bool(pieces) or bool(ropes)
                    or matter is not None or bool(bolts) or bool(shots))
        # (the footage's HDRI only for a set drawn round the liquid: Lume's water, below, is never over footage)
        footage_env = self._footage_env(scene, look, frame, plate, cs, plate_fit, wanted=stage_on)
        env = LR.environment(look)
        if env is not None:
            # the HDRI is the set's own light: its sky for the ambient, its sun for the key light (the footage's leaves
            # the ambient to Match ambient to footage, as on the smoke)
            if not footage_env:
                look.sky = env[0]
            if look.env_sun:
                strength = max(look.sun) or 3.0
                look.sun_azimuth, look.sun_elevation = env[1], env[2]
                look.sun = tuple(c * strength for c in env[3])
        r.hold_stage = None
        p_transform, p_gain = INPUT_TRANSFORMS.get(comp.plate_transform, 0), comp.plate_gain
        standins = stage_mod.standin_colours(scene)

        drop_shutter = scene.data['render']['shutter_angle'] / 360.0 / scene.fps if motion_blur else 0.0
        lamps = scene.lamps(frame) if hasattr(scene, 'lamps') else []
        if any(l['child'][0] == 'light' for l in getattr(scene, 'links', None) or []):
            carried = attached(scene, 'light', self.floating_overrides(frame))
            if carried:
                lamps = scene.lamps(frame, moved=carried)

        wview = self._weather_view(scene, frame, live, None if live else self.cache.get(frame))
        cloth, clook = self._cloth_for_liquid(scene, frame, live, final, look)
        grass = self.strands_for(frame)
        if grass and clook is None:
            clook = self._raster_look(scene, frame, final, look)
        nlamps = r.pack_lamps(lamps) if (cloth or grass) else 0   # (the set's lights on the fabric and the grass)
        # Lume traces the water itself (stage.wgsl lume_water.wgsl): the set drawn round and under it (the floor, the
        # objects in CG), the liquid's own element left empty. Not yet with fabric or grass (the march sets them in the
        # liquid) or a liquid it does not trace (LiquidRenderer.lume_ok): then the march draws it over the set Lume lit.
        from . import lume as LU
        lume_water = (LU.settings(scene).on and mode == 'composite' and not footage and not (cloth or grass)
                      and LR.lume_ok(vol, look))
        if lume_water:
            stage_on = True
            objects = True

        def march(b, jit, s):
            if lume_water:
                LR.lume_clear(b, (fw, fh))
                if wview is not None:
                    self.weather_r.cover(b, wview, cs, fire, look, (fw, fh))
                LR.drops(b, vol, cs, fire, look, (fw, fh), jitter=jit, shutter=drop_shutter)
                LR.rain(b, cs, fire, look, (fw, fh), jitter=jit, shutter=drop_shutter, frame=frame,
                        wind=scene.liquid_params(frame).wind, ground=ground)
                if wview is not None:
                    self.weather_r.draw(b, wview, cs, fire, look, (fw, fh), jitter=jit, shutter=drop_shutter, time=t)
                return
            lay = None
            if cloth or grass:
                # the fabric and the grass, drawn first: the march sees them in front of the liquid and through it
                self._draw_raster(b, r, cs, fire, clook, (fw, fh), jit, 0.0, None, cloth, grass,
                                  light_gain=2.0 ** look.exposure, fire_lights=False, lamp_count=nlamps)
                lay = self.raster.layer(b)
            LR.march(b, vol, cs, fire, look, comp, (fw, fh), plate=ptex, plate_fit=plate_fit,
                     plate_transform=p_transform, plate_gain=p_gain,
                     jitter=jit, seed=s, shutter=shutter, ground=ground, time=t, lamps=lamps, cloth=lay,
                     standins=standins)
            if wview is not None:
                self.weather_r.cover(b, wview, cs, fire, look, (fw, fh))
            LR.drops(b, vol, cs, fire, look, (fw, fh), jitter=jit, shutter=drop_shutter)
            LR.rain(b, cs, fire, look, (fw, fh), jitter=jit, shutter=drop_shutter, frame=frame,
                    wind=scene.liquid_params(frame).wind, ground=ground)
            if wview is not None:
                self.weather_r.draw(b, wview, cs, fire, look, (fw, fh), jitter=jit, shutter=drop_shutter, time=t)

        r._ensure_fire(fw, fh)
        with self.gpu.batch() as b:
            # the liquid's surface and the key light's caustics under it first: Lume traces them in the stage
            LR.build(b, vol, look)
            LR.sea(b, vol, look)
            LR.caustics(b, vol, look, fire)
            stage = None
            if stage_on:
                light = stage_mod.water_light(look, comp)
                light.lamps = r.pack_lamps(lamps)
                if env is not None:   # (this render's: not a texture left from an earlier one)
                    light.env, light.env_rotation = LR.env_tex, float(look.env_rotation)
                    light.env_strength = float(look.env_strength) * 2.0 ** float(look.exposure)
                    light.env_image = look.sky_image if not look.environment else None
                    self.stage.env_sky = tuple(float(c) for c in LR.env_info[0])   # (for Lume's pick of it)
                    if footage_env:   # (the set's sky is the footage's light, as in a fire scene)
                        light.sky = tuple(float(c) * 2.0 ** float(look.exposure) for c in env[0])
                size = r.plate_size if footage else (W, H)
                self.stage.heat = self._stage_heat(frame, vol.colliders)   # (hot objects glow: objheat_engine.py)
                stage = ptex = self.stage.draw(b, r, scene, cs, fire, vol.colliders, vol.meshes, light, comp, size,
                                               plate_fit=plate_fit, samples=samples, shutter=shutter, footage=footage,
                                               ground_y=vol.origin[1], frame=frame, objects=objects,
                                               floor=not look.bottomless, pieces=pieces, ropes=ropes, matter=matter, bolts=bolts,
                                               grass=self._strands.ground_map(b) if grass else None, final=final, shots=shots,
                                               water=LR.lume_water(vol, look, t) if lume_water else None,
                                               colours=([tuple(c[:3]) if c[3] > 0.5 else tuple(look.standin_color) for c in standins]
                                                        if (lume_water and look.colliders_look == 'shaded') else None))
                if self.stage.has_pieces or self.stage.has_matter:   # the liquid is hidden behind the pieces and the matter
                    r.hold_stage = self.stage.hold
                    r.hold_stage_matte = bool(footage and r.hold is not None and r.hold_on[0])
                p_transform, p_gain = INPUT_TRANSFORMS['linear'], 1.0
            if samples == 1:
                march(b, (0.0, 0.0), base_seed)
            else:
                self._ensure_acc(fw, fh)
                self._ensure_acc2(fw, fh)
                set0, set1 = self._acc[0:3], self._acc[3:6]
                mat0, mat1 = self._acc2[0:3], self._acc2[3:6]
                mats = [r.mask, r.mask, r.mask]   # the mattes (the accumulator takes three inputs)
                for i in range(samples):
                    march(b, (_halton(i + 1, 2) - 0.5, _halton(i + 1, 3) - 0.5), base_seed + i)
                    src_in, dst = (set0, set1) if i % 2 == 0 else (set1, set0)
                    b.run(self.k_accum, [r.beauty, r.emit, r.aux, *src_in, *dst], Uniforms().v4(fw, fh, 0, i), (fw, fh, 1))
                    m_in, m_dst = (mat0, mat1) if i % 2 == 0 else (mat1, mat0)
                    b.run(self.k_accum, [*mats, *m_in, *m_dst], Uniforms().v4(fw, fh, 0, i), (fw, fh, 1))
                last, other = (set1, set0) if (samples - 1) % 2 == 0 else (set0, set1)
                b.run(self.k_accum, [r.beauty, r.emit, r.aux, *last, *other], Uniforms().v4(fw, fh, 1, samples), (fw, fh, 1))
                b.run(self.k_copy, [*other, r.beauty, r.emit, r.aux], Uniforms().v4(fw, fh), (fw, fh, 1))
                mlast, mother = (mat1, mat0) if (samples - 1) % 2 == 0 else (mat0, mat1)
                b.run(self.k_accum, [*mats, *mlast, *mother], Uniforms().v4(fw, fh, 1, samples), (fw, fh, 1))
                b.run(self.k_copy, [*mother, r.mask, r.surf, self._acc2_spare], Uniforms().v4(fw, fh), (fw, fh, 1))
            r.defocus(b, comp, cs, fire)
            r.bloom(b, comp.bloom_radius)
            hz = LR.lava_haze(b, vol, look, cs, fire, comp, time=t)
            r.composite(b, (W, H), comp, time=t, frame=frame, plate_fit=plate_fit, liquid=True, liquid_haze=hz,
                        hfov=cs.hfov, stage=stage)
            LR.lens_drops(b, look, (W, H), time=t)
        r.hold_stage = None
        self.last_render_ms = (time.perf_counter() - t0) * 1000.0
        return cs

    def _cloth_for_liquid(self, scene, frame, live, final, wlook):
        """(fabric in the shot, its look) for rendering a liquid scene's `frame`: the fire look's settings, lit by
        the water look's key light and sky (the cloth is drawn as the march sees the set)."""
        if not self.cloth.active:
            return False, None
        if live:
            if not self.cloth.placed:
                return False, None
            self.cloth.use_view(None)
        else:
            entry = self.cache.get(frame) if self.cache is not None else None
            cl = {k[6:]: v for k, v in (entry or {}).items() if k.startswith('cloth_') and k != 'cloth_state'}
            if not cl:
                return False, None
            self.cloth.use_view(cl)
        return True, self._raster_look(scene, frame, final, wlook)

    @staticmethod
    def _raster_look(scene, frame, final, wlook):
        """The look the cloth and the grass are drawn in in a liquid scene: the fire look's settings, lit by the water
        look's key light and sky."""
        clook = scene.look(frame, final)
        clook.sun_color, clook.sun_intensity = tuple(wlook.sun), 1.0
        clook.sun_azimuth, clook.sun_elevation = wlook.sun_azimuth, wlook.sun_elevation
        clook.ambient = tuple(wlook.sky)
        clook.time = scene.seconds(frame)
        return clook

    def _stats_liquid(self):
        L = self.liquid
        f = self.liquid_r.surface_dims(L.dims, 2.0) if L.dims else None
        return {
            'gpu': self.gpu.name, 'backend': self.gpu.backend, 'kind': 'liquid', 'dims': L.dims,
            'voxels': int(np.prod(L.dims)) if L.dims else 0, 'cell_mm': L.h * 1000 if L.h else 0,
            'max_speed': L.max_speed, 'substeps': self.last_substeps, 'sim_ms': self.last_step_ms,
            'render_ms': self.last_render_ms, 'cached': len(self.cache.items), 'cache_mb': self.cache.bytes / 1e6,
            'particles': L.count, 'whitewater': L.ww_count, 'embers': 0, 'burning': 0,
            'particle_limit': L.capacity and L.count >= 0.995 * L.capacity,
            'memory_mb': (L.memory_bytes() + (self.liquid_r.memory_bytes(L.dims, 2.0) if f else 0)) / 1e6 if L.dims else 0,
            'mesh_errors': dict(self.solver.meshes.errors),
        }
