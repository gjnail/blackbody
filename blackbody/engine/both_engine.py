"""The engine's path for fire and liquid in one box (domain kind 'both'): a hose on a fire, burning
fuel on water, a fire by a fountain, lava running into the sea.

The fire and the water (and the lava, when a source pours it) run on the same grid. Each substep:
- The lava moves first, as a liquid of its own (a second LiquidSolver, with the Lava flow preset's
  look and behaviour: Scene.lava_params / lava_look). To the water it is solid ground that moves:
  both_solid.wgsl lays its level set into the water's solid field, so water runs over and round it
  and is shoved aside where it advances. The lava only feels the water's heat.
- The water moves.
- both_lava.wgsl: the air over hot lava heats up (a shimmering updraft that sets burnable things
  alight) and water touching lava boils, about 0.4 kg of steam per square metre of contact per
  second, into the gas at the first open cell over the contact; both_quench.wgsl chills the lava's
  skin where the water touches it, so it crusts over black.
- both_wet.wgsl writes the fire solver's water field: where the liquid is (it smothers flame and
  fuel and cools the gas to boiling, the heat becoming steam), the steam off the lava, and the state
  of the fuel beds under the fire emitters. Water reaching a bed soaks it and it stops burning; the
  heat the bed stored while it burned boils the water off in a plume of steam; wet fuel dries out
  again in the heat of the flames around it, so a fire only partly doused creeps back.
- both_drag.wgsl lets the liquids carry the air: gas in and next to the water (or the lava) takes
  its velocity, so a stream drags a jet of air through the flames and smoke and the gas over a pool
  stays still.
- both_evap.wgsl boils off drops of spray flying through hot gas or landing on lava (not the bulk
  of the water: hot gas holds too little heat for its volume to boil a stream or a pool).
- The fire steps. Steam dilutes the air feeding the flames (they starve as it passes about a
  quarter of the gas) and swells as it is made, blowing the gas out where water hits the fire.

Rendering, per sample: the fire over the footage becomes the backdrop the liquids refract and reflect
(flames behind a sheet of water show through it). The lava is traced over it, the water over the
fire and the lava (so lava under the water shows through it, refracted), and the nearer of the two
kept per pixel; then the fire is marched again only up to the liquids' surface and laid over them.
The composite treats the result as one element, with the water's wet ground, shadows and caustics on
the footage.
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import camera as cam
from . import stage as stage_mod
from .solids import attached
from .cloth import STEPS_PER_SECOND
from .gpu import TU, Uniforms
from .liquid import LiquidSolver
from .liquid_render import LiquidRenderer, LiquidView, liquid_top_of
from .renderer import BB_DEFINES, INPUT_TRANSFORMS
from .solver import MAX_COLLIDERS, pack_emitters


def _halton(i, b):
    f, r = 1.0, 0.0
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r


class BothEngine:
    """Mixin for Engine, alongside LiquidEngine."""

    _both = None
    _both_size = None
    _bk = None           # the coupling kernels
    _bed = None          # the fuel beds under the fire emitters (x = soaked, y = heat), ping-pong on the fire grid
    _wraw = None         # the water field before steam made under the water is lifted to its surface (both_rise)
    lava = None          # the lava's own LiquidSolver (a fire-and-liquid scene with lava sources)
    lava_r = None        # and its renderer
    _lava_on = False
    _lava_field = None   # fire grid, rgba32float: lava temperature (K), lava share, steam into the gas, water around
    _no_field = None     # a 1-cell stand-in for it
    _lava_filled = None
    _rwet_lava = None
    EVAPORATION = 3.0    # 1/s: share of lone drops in gas at flame heat that boil away per second
    LAVA_DROPS = 2.0     # 1/s: share of lone drops on lava at full heat that sizzle away per second
    DRAG = 60.0          # 1/s: how fast gas in a face full of liquid takes the liquid's velocity
    BED_HEATING = 0.8    # 1/s: how fast a burning fuel bed heats toward the fire's temperature
    BED_QUENCH = 1.5     # 1/s: how fast water on a fully soaked bed takes its heat above boiling
    SOAK_WATER = 2.5     # a soaked bed holds enough water to quench itself from the fire's heat this many times over
    CHAR_SMOKE = 0.8     # 1/s: smoke from a doused bed still hot enough to char, at full heat
    DRY_STILL = 1.0 / 90.0  # 1/s: soaked fuel drying without flames around it
    LAVA_AIR_RATE = 20.0  # 1/s: how fast the air next to lava heats toward its boundary-layer temperature (it holds it against the gas cooling)
    LAVA_AIR_SHARE = 0.6  # the share of the lava surface's temperature (above the air's) that air right over it reaches
    STEAM_CONDENSE = 0.5  # per cell of water over a lava contact: the steam from there that condenses back into the sea
    LAVA_SOLID = True    # the water treats the lava as moving ground (both_solid.wgsl)
    SPATTER = 6e-4       # spray drops thrown up per g/m^3 of steam off the lava (whitewater), and their speed (m/s)
    SPATTER_SPEED = 1.6

    # -- simulation ------------------------------------------------------------------------------

    def _lava_solver(self):
        if self.lava is None:
            self.lava = LiquidSolver(self.gpu, self.solver.meshes)
        return self.lava

    def _lava_renderer(self):
        if self.lava_r is None:
            self.lava_r = LiquidRenderer(self.gpu, self.renderer)
        return self.lava_r

    def _prepare_both(self, scene, final=False, soft=False):
        sig = scene.sim_signature(final)
        dims, h, origin = scene.sim_layout(final)
        L = self._liq()
        meshes = [scene.mesh_path(d['mesh']) for d in scene.emitters + scene.colliders
                  if d['enabled'] and d['shape'] == 'mesh' and d['mesh']]
        if self.solver.set_meshes(meshes, scene.data['domain']['mesh_resolution']):
            L.colliders = None
            if self.lava is not None:
                self.lava.colliders = None
        changed = self.solver.configure(dims, h, origin, scene.features(), scene.data['render']['upres'] if final else 1)
        self.solver.set_colliders(scene.colliders_gpu(scene.start))
        changed = L.configure(dims, h, origin, scene.liquid_capacity(final), scene.whitewater_capacity()) or changed
        L._prm = scene.liquid_params(scene.start)
        L.set_colliders(scene.colliders_gpu(scene.start))
        L.gas = self.solver   # the liquid's thermal model reads the gas it is in (liquid_thermal.py)
        self._prepare_weather(scene, final)   # precipitation (weather.py)
        # fabric: in the air and the fire (it burns, and feeds the fire) and in the water (it soaks: cloth.py)
        if self.cloth.configure(scene.fabric_specs()):
            changed = True
        if self.solids.configure(scene, (dims, h, origin)):
            changed = True
        if self._prepare_matter(scene, (dims, h, origin)):
            changed = True
        if self._prepare_strands(scene):
            changed = True
        self.solver.cloth_hook = self._solver_hook()
        lava_on = scene.has_lava()
        if lava_on:
            V = self._lava_solver()
            changed = V.configure(dims, h, origin, scene.liquid_capacity(final), 0) or changed
            V._prm = scene.lava_params(scene.start)
            V.set_colliders(scene.colliders_gpu(scene.start))
        if lava_on != self._lava_on:
            self._lava_on = lava_on
            changed = True
        if self.kind != 'both':
            self.kind = 'both'
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

    def _reset_both(self):
        self.solver.reset()
        self.embers.reset()
        self._reset_liquid()
        self._lava_filled = set()
        if self.lava is not None and self.lava.dims is not None:
            self.lava.reset()
        with self.gpu.batch() as b:
            for t in list(self._bed or ()) + [x for x in (self._lava_field, self._wraw) if x is not None]:
                self.liquid.fill(b, t)

    # -- the water, the lava and the fire, each substep --------------------------------------------

    def _coupling(self):
        """The coupling kernels and fields (allocated outside the step's batch: a batch must not open
        inside another)."""
        g = self.gpu
        if self._bk is None:
            vf = g.vel_format
            P = (64, 1, 1)
            self._bk = {
                'wet': g.kernel('both_wet.wgsl', ['utex3d'] * 4 + ['st3d:rgba16float:w'] * 2 + ['utex3d'] * 2
                                + ['st3d:rgba16float:w']),
                'drag': g.kernel('both_drag.wgsl', ['utex3d'] * 3 + [f'st3d:{vf}:w'] + ['utex3d'] * 2, 'main', {'VELFMT': vf}),
                'evap': g.kernel('both_evap.wgsl', ['buf', 'buf', 'buf', 'utex3d', 'utex3d', 'utex3d'], workgroup=P),
                'lava': g.kernel('both_lava.wgsl', ['utex3d'] * 4 + ['st3d:rgba16float:w', 'st3d:rgba32float:w']),
                'quench': g.kernel('both_quench.wgsl', ['buf', 'utex3d', 'utex3d'], workgroup=P),
                'solid': g.kernel('both_solid.wgsl', ['utex3d', 'st3d:r32float:rw']),
                'spatter': g.kernel('both_spatter.wgsl', ['utex3d', 'buf', 'buf', 'buf']),
                'rise': g.kernel('both_rise.wgsl', ['utex3d', 'utex3d', 'st3d:rgba16float:w']),
                'displace': g.kernel('both_displace.wgsl', ['buf', 'buf', 'buf', 'utex3d'], workgroup=P),
            }
            self._no_field = g.texture3d((1, 1, 1), 'rgba32float', 'no-lava-field')
            g.upload(self._no_field, np.zeros((1, 1, 1, 4), np.float32))
        S = self.solver
        dims = tuple(S.dims)
        fresh = []
        if self._bed is None or self._bed[0].size != dims:
            for t in list(self._bed or ()) + ([self._wraw] if self._wraw is not None else []):
                t.destroy()
            self._bed = [g.texture3d(dims, 'rgba16float', f'fuel-bed{i}') for i in range(2)]
            self._wraw = g.texture3d(dims, 'rgba16float', 'water-raw')
            fresh += self._bed + [self._wraw]
        if self._lava_on and (self._lava_field is None or self._lava_field.size != dims):
            if self._lava_field is not None:
                self._lava_field.destroy()
            self._lava_field = g.texture3d(dims, 'rgba32float', 'lava-field')
            fresh.append(self._lava_field)
        if fresh:
            with g.batch() as b:
                for t in fresh:
                    self.liquid.fill(b, t)
        return self._bk

    def _lava_meets(self, b, scene, prm, dt, lava_k):
        """Record the lava heating the air and boiling the water it touches (both_lava.wgsl, which also
        writes the lava field), and the water chilling the lava's skin (both_quench.wgsl)."""
        S, L, V = self.solver, self.liquid, self.lava
        k = self._bk
        lv = scene.data.get('lava', {})
        rest_l, rest_w = float(V._prm.ppc), float(L._prm.ppc)
        u = (S._grid(dt, prm).v4(rest_l, lava_k, prm.ambient_k, prm.flame_k)
             .v4(self.LAVA_AIR_RATE * float(lv.get('air_heat', 1.0)), self.LAVA_AIR_SHARE * min(float(lv.get('air_heat', 1.0)), 1.5),
                 float(lv.get('boiling', 1.0)), self.STEAM_CONDENSE)
             .v4(rest_w))
        b.run(k['lava'], [V.DENS, V.HEAT, L.DENS, S.scal[0], S.scal[1], self._lava_field], u, S.dims)
        S.scal.reverse()
        if L._prm.whitewater and L.ww_capacity > 64 and self.SPATTER > 0.0:
            # the boiling line spits spray
            u = S._grid(dt, prm).v4(self.SPATTER * float(lv.get('boiling', 1.0)), self.SPATTER_SPEED, L.ww_capacity, 1.2).v4(L.steps * 7 + 5)
            b.run(k['spatter'], [self._lava_field, L.WA, L.WB, L.wctr], u, S.dims)
        quench = float(lv.get('quench', 4.0))
        if quench > 0.0:
            u = V._grid(dt).v4(V.capacity, rest_l, quench, rest_w)
            b.run_indirect(k['quench'], [V.parts, V.DENS, L.DENS], u, V.args, 0)

    def _water_on_fire(self, b, scene, prm, ems, dt):
        """Record, for one substep before the fire steps: the water field the fire reads (with the fuel
        beds soaking, heating and boiling off, and the steam off the lava), and the liquids dragging the
        air along."""
        S, L, V = self.solver, self.liquid, (self.lava if self._lava_on else None)
        k = self._bk
        c = scene.data['combustion']
        gas = getattr(L, 'gas_flux', None)   # the liquid's heat model: its vapour and heat into the gas
        rest = float(L._prm.ppc)
        heat = max(float(c.get('ember_heat', 150.0)), 1.0)
        u = (S._grid(dt, prm).v4(*L.dims, rest)
             .v4(float(c.get('soak', 6.0)), self.BED_HEATING, self.BED_QUENCH, heat)
             .v4(prm.steam_yield, prm.boil_temp, 1.0 / max(float(c.get('rekindle', 8.0)), 0.1), self.DRY_STILL)
             .v4(self.CHAR_SMOKE, self.SOAK_WATER * heat, 1.0 if V is not None else 0.0, 1.0 if gas is not None else 0.0))
        pack_emitters(u, ems, meshes=S.meshes)
        field = self._lava_field if V is not None else self._no_field
        b.run(k['wet'], [L.DENS, S.scal[0], self._bed[0], S.meshes.atlas, self._wraw, self._bed[1], field,
                         gas if gas is not None else self._no_field, S.scal[1]], u, S.dims)
        self._bed.reverse()
        S.scal.reverse()
        b.run(k['rise'], [self._wraw, field, S.water], S._grid(dt, prm).v4(1.0 if V is not None else 0.0), S.dims)
        u = S._grid(dt, prm).v4(*L.dims, rest).v4(self.DRAG, 1.0 if V is not None else 0.0, float(V._prm.ppc) if V is not None else rest)
        b.run(k['drag'], [S.vel[0], L.vel_tex, L.DENS, S.vel[1], V.vel_tex if V is not None else L.vel_tex,
                          V.DENS if V is not None else L.DENS], u, tuple(x + 1 for x in S.dims))
        S.vel.reverse()

    def _boil_drops(self, b, boil, dt, lava_k):
        """Record lone drops boiling away in the hot gas and on hot lava."""
        L, S = self.liquid, self.solver
        if getattr(L, 'gas_flux', None) is not None:
            return   # the liquid's heat model boils and evaporates its drops itself
        on = self._lava_on and self.lava is not None
        u = (L._grid(dt).v4(L.capacity, boil, self.EVAPORATION, L.steps + 17).v4(*S.dims, float(L._prm.ppc))
             .v4(1.0 if on else 0.0, self.LAVA_DROPS, lava_k))
        b.run_indirect(self._bk['evap'], [L.parts, L.ctr, L.freelist, S.scal[0], L.DENS,
                                          self._lava_field if on else self._no_field], u, L.args, 0)

    def _step_both(self, scene, frame):
        t0 = time.perf_counter()
        fdt = scene.v('domain', 'time_scale', frame) / scene.fps
        d = scene.data['domain']
        prm = scene.solver_params(frame)
        ep = scene.ember_params(frame)
        look = scene.look(frame)
        lprm = scene.liquid_params(frame)
        if frame < scene.start and scene.data['liquid']['settle']:
            lprm.damping = 6.0
        lprm.ocean = self._ocean_spec(scene, frame)
        lprm.sea_sides = self._sea_sides(scene, lprm.ocean)
        if lprm.ocean is not None:
            lprm.damping = 0.0
        lprm.rain, lprm.rain_drop = self._rain(scene, frame)
        L = self.liquid
        L._prm = lprm
        V = self.lava if self._lava_on else None
        vprm = None
        lava_k = 1300.0
        if V is not None:
            vprm = scene.lava_params(frame)
            V._prm = vprm
            lava_k = float(scene.lava_look(frame).glow_temp)
        n_fire = self.solver.substeps_for(fdt, cfl=d['cfl'], lo=d['substeps_min'], hi=scene.substep_cap(frame, self.solver.h))
        hi = min(40, max(d['substeps_max'], L.capillary_substeps(fdt)))
        n = max(n_fire, L.substeps_for(fdt, cfl=d['cfl'], lo=d['substeps_min'], hi=hi))
        if V is not None:
            n = max(n, V.substeps_for(fdt, cfl=d['cfl'], lo=d['substeps_min'], hi=hi))
        cam_g = self._cam_grid(scene, frame) if ep.enabled else (0.0, 0.0, 0.0)
        # rigid bodies move through the frame first (pushed by the air and the water as they were), then the gas
        # and the liquid see them where they are at each substep
        solids = self.solids if self.solids.active else None
        poses = None
        self._push_matter(L)
        self._cloth_meets_matter()      # (fabric and sand, snow, mud: each the other's surface this frame)
        if solids:
            self._air_for_solids()
            self._objects_meet_cloth(fdt)
            poses = solids.advance(scene, frame, fdt, n, couple=self._matter_couple(scene, frame, fdt, self.solver.meshes))
            self._cloth_takes_objects(fdt)
        moving = scene.colliders_animated() or bool(solids)
        wprm = scene.weather_params(frame) if (self._wx_on and self.weather is not None) else None
        filled = getattr(self, '_filled', None)
        if filled is None:
            filled = self._filled = set()
        if self._lava_filled is None:
            self._lava_filled = set()
        boil = scene.boil_temp()
        k = self._coupling()
        cloth = self.cloth.active
        if cloth:
            if not self.cloth.placed:
                self.cloth.place(scene.fabrics_at(frame - 1), look.ambient_k)
        self.cloth.prepare_frame(self.solver)   # (also clears the solver's steam-off-cloth flag without cloth)
        cloth_steps = max(1, int(math.ceil(fdt * STEPS_PER_SECOND / n)))
        grass = self.strands_on
        if grass:
            self._strands.prepare_frame(self.solver)
            gwind = scene.wind(frame, scene.v('camera', 'fire_yaw', frame))
            ggust = scene.v('motion', 'gust', frame)
        pieces = poses is not None and self._pieces_for(scene, self.solver)
        lpieces = poses is not None and self._pieces_for(scene, L, 'liquid')
        burning = poses is not None and self._burn_pieces(scene, fdt)   # (things that break and burn)
        dust = self.solids.dust(scene, fdt, n) if (poses is not None and self.solids.sets) else None
        with self.gpu.batch() as b:
            self._footage_solid(b, scene, frame)
            if burning:
                self.piece_fire.splat(b, self.solver)
            self._splat_matter_fire(b, scene, self.solver)       # (dry leaves, sawdust, coal: their flames)
            msolid = self._matter_solid(b, self.solver)          # (sand, snow and mud: solid to the gas
            lmsolid = self._matter_solid(b, L, 'liquid')        # and to the water)
            regions = solids.regions(scene) if solids else []
            if regions:
                L.clear_float(b)
            for i in range(n):
                fs = frame - 1 + (i + 0.5) / n
                dt = fdt / n
                cols = scene.colliders_gpu(fs, poses[i] if poses else None) if moving else None
                lprm.clock = scene.seconds(fs)
                srcs = scene.sources_gpu(fs, filled)
                if V is not None:
                    vprm.clock = lprm.clock
                    V.step(b, dt, vprm, scene.sources_gpu(fs, self._lava_filled, emits='lava'), cols)
                    # water the lava has flowed into is displaced (both_displace.wgsl), then the water's solid
                    # ground: its colliders, with the lava over them
                    if self.LAVA_SOLID:
                        b.run_indirect(k['displace'], [L.parts, L.ctr, L.freelist, V.DENS],
                                       L._grid(dt, lprm).v4(L.capacity, float(vprm.ppc), 0.5), L.args, 0)
                    if cols is not None:
                        L.colliders = list(cols)[:MAX_COLLIDERS]
                    L._write_sdf(b)
                    if self.LAVA_SOLID:
                        b.run(k['solid'], [V.DENS, L.SDF], L._grid(dt, lprm).v4(float(vprm.ppc)), L.dims)
                    L.pieces_step = self._solids_step(self.body_field_for('liquid') if lpieces else None, lmsolid, i)
                    L.step(b, dt, lprm, srcs, None)
                    L.pieces_step = None
                else:
                    L.pieces_step = self._solids_step(self.body_field_for('liquid') if lpieces else None, lmsolid, i)
                    L.step(b, dt, lprm, srcs, cols)
                    L.pieces_step = None
                if regions:
                    L.float_forces(b, regions, i, dt)
                if wprm is not None:
                    self._step_weather(b, scene, frame, wprm, fs, dt, moving)
                carried = poses[i] if poses else None   # (things attached to falling objects go with them)
                ems = scene.emitters_gpu(fs, substeps=n, moved=attached(scene, 'emitter', carried))
                if dust:
                    ems = ems + dust[i]    # (dust where things broke; the scene's own sources come first)
                if V is not None:
                    self._lava_meets(b, scene, prm, dt, lava_k)
                if self.solver.water is not None:
                    self._water_on_fire(b, scene, prm, ems, dt)
                self._boil_drops(b, boil, dt, lava_k)
                self.solver.pieces_step = self._solids_step(self.body_field if pieces else None, msolid, i)
                self.solver.step(b, dt, prm, ems, cols)
                self.solver.pieces_step = None
                if cloth:
                    # fabric moves in the air and the water just stepped, then spreads onto the gas
                    self.cloth.step(b, self.solver, dt, scene.fabrics_at(fs + 0.5 / n, moved=attached(scene, 'fabric', carried)), prm, look,
                                    list(cols) if cols is not None else self.solver.colliders, self.solver.meshes,
                                    steps=cloth_steps, liquid=L)
                    self.cloth.splat(b, self.solver, look)
                if grass:
                    self._step_strands(b, scene, fs, dt, self.solver, list(cols) if cols is not None else self.solver.colliders,
                                       self.solver.meshes, look, bool(scene.data['domain']['ground']), self.solver.origin[1],
                                       gwind, ggust)
                if ep.enabled:
                    ember_ems = scene.emitters_gpu(fs, embers_only=True, moved=attached(scene, 'emitter', carried))
                    if ember_ems or self.embers.count:
                        self.embers.step(b, self.solver, ep, ember_ems, dt, cam_g, look.smoke_density, look.ambient_k)
            L.pack(b)
            if V is not None:
                V.pack(b)
            if wprm is not None:
                self.weather.pack(b)
        L.lava_heat = self._lava_field if V is not None else None   # (for the liquid's thermal model)
        self.solver.measure()
        L.measure()
        if wprm is not None:
            self.weather.measure()
            self._weather_surface_ready()
        if V is not None:
            V.measure()
        if regions:
            solids.liquid_measures(L.read_float(len(regions), n), fdt, n, L.h, lprm.rho)
        self._step_matter(scene, frame, fdt, poses, n, self.solver.meshes)
        self._melt_matter(scene, frame, fdt, self.solver, L)
        self._heat_matter(scene, frame, fdt, self.solver, L)
        self._wet_matter(fdt, L)
        self.sim_frame = frame
        self.last_substeps = n
        self.last_step_ms = (time.perf_counter() - t0) * 1000.0

    def _snapshot_both(self, scene=None):
        """A cached frame: the fire, the water, and the lava's packed particles, ground and crust."""
        entry = self._snapshot_fire(scene)
        entry.update(self._snapshot_liquid())
        V = self.lava
        if self._lava_on and V is not None and V.dims is not None:
            entry['lava'] = V.read_packed()
            entry['lava_wet'] = V.read_wet()
            entry['lava_origin'] = tuple(V.origin)
            crust = V.read_crust() if hasattr(V, 'read_crust') else None
            if crust is not None:
                entry['lava_crust'] = crust
        return entry

    def _stats_both(self, stats):
        L = self.liquid
        V = self.lava if self._lava_on else None
        mem = L.memory_bytes() / 1e6 if L.dims else 0.0
        if V is not None and V.dims:
            mem += V.memory_bytes() / 1e6
        stats.update(kind='both', particles=L.count, whitewater=L.ww_count,
                     particle_limit=bool(L.capacity and L.count >= 0.995 * L.capacity),
                     memory_mb=stats.get('memory_mb', 0.0) + mem)
        if V is not None:
            stats['lava_particles'] = V.count
        return stats

    # -- rendering -------------------------------------------------------------------------------

    def _ensure_both(self, w, h):
        g = self.gpu
        if self._both is None:
            self.k_bfp = g.kernel('liq_both_fp.wgsl', ['tex2d', 'tex2d', 'smp', 'st2d:rgba16float:w'], workgroup=(8, 8, 1))
            self.k_bmerge = g.kernel('liq_both_merge.wgsl', ['utex2d'] * 6 + ['st2d:rgba16float:w'] * 3, workgroup=(8, 8, 1))
            self.k_bover = g.kernel('both_over.wgsl', ['utex2d'] * 2 + ['st2d:rgba16float:w'], workgroup=(8, 8, 1))
            self.k_bliq = g.kernel('both_liquids.wgsl', ['utex2d'] * 6 + ['st2d:rgba16float:w'] * 3, workgroup=(8, 8, 1))
            self.k_glow = g.kernel('both_glow.wgsl', ['tex3d'] * 3 + ['tex2d', 'smp', 'st3d:rgba16float:w'] + ['tex3d'] * 3,
                                   defines=BB_DEFINES)
        if self._both_size == (w, h):
            return
        for t in (self._both or {}).values():
            t.destroy()
        usage = TU.TEXTURE_BINDING | TU.STORAGE_BINDING | TU.COPY_DST | TU.COPY_SRC
        self._both = {k: g.texture2d(w, h, 'rgba16float', f'both-{k}', usage)
                      for k in ('fp', 'lb', 'le', 'la', 'fb', 'fe', 'fa', 'kb', 'ke', 'ka', 'kp')}
        self._both_size = (w, h)

    def _lava_view(self, scene, frame):
        """The lava of `frame` for its renderer (live or cached), or None without lava."""
        V = self.lava
        if not self._lava_on or V is None or V.dims is None:
            return None
        d = scene.data['domain']
        bounds = (bool(d['open_sides']), bool(d['open_top']), not d['ground'])
        extra = dict(colliders=tuple(scene.colliders_gpu(frame, self.floating_overrides(frame))), meshes=self.solver.meshes,
                     time=scene.seconds(frame))
        ppc = int(V._prm.ppc)
        if self.sim_frame == frame:
            top = float(V.bbox[1][1]) if V.bbox is not None else -1.0
            return LiquidView(V.packed, V.packed_count, V.WET, V.dims, V.h, V.origin, ppc, None, 0, bounds,
                              crust=getattr(V, 'crust_field', None), liquid_top=top, **extra)
        entry = self.cache.get(frame)
        if entry is None or 'lava' not in entry:
            return None
        VR = self._lava_renderer()
        buf = VR.upload_particles(entry['lava'])
        nx, nz = V.dims[0], V.dims[2]
        if self._rwet_lava is None or self._rwet_lava.size[:2] != (nx, nz):
            if self._rwet_lava is not None:
                self._rwet_lava.destroy()
            self._rwet_lava = self.gpu.texture2d(nx, nz, 'r32float', 'cached-lava-ground')
        self.gpu.upload(self._rwet_lava, entry['lava_wet'].astype(np.float32)[..., None])
        crust = entry.get('lava_crust')   # the crust field the lava's flow carries (sim grid, (nz, ny, nx, 4))
        cbuf = VR.upload_crust(crust) if crust is not None and np.ndim(crust) == 4 else None
        return LiquidView(buf, len(entry['lava']), self._rwet_lava, V.dims, V.h, tuple(entry.get('lava_origin', V.origin)), ppc,
                          None, 0, bounds, crust=cbuf, liquid_top=liquid_top_of(entry['lava'], V.dims), **extra)

    def _render_both(self, scene, frame, out_size, mode='composite', final=False, samples=1, motion_blur=False,
                     fire_scale=1.0, plate=None, plate_fit=(1.0, 1.0), seed=None):
        t0 = time.perf_counter()
        vol, live = self.volume_for(scene, frame)
        lv, lv_live = self._liquid_view(scene, frame)
        wview = self._weather_view(scene, frame, lv_live, None if lv_live else self.cache.get(frame))
        if vol is None or lv is None:
            raise RuntimeError(f'frame {frame} is neither simulated nor cached')
        kv = self._lava_view(scene, frame)
        W, H = out_size
        fw, fh = max(8, int(round(W * fire_scale))), max(8, int(round(H * fire_scale)))
        spec, fire = scene.camera(frame)
        cs = cam.compute(spec, W / H, fire)
        look = scene.look(frame, final)
        look.vapour = bool(vol.aux) and self.solver.features.get('vapour', False)
        look.colourant = bool(vol.chem)
        self._carried_lamps(scene, frame, look)
        wlook = scene.water_look(frame, final)
        wlook.rain, wlook.rain_drop = self._rain(scene, frame)
        klook = scene.lava_look(frame, final) if kv is not None else None
        lava_k = float(klook.glow_temp) if klook is not None else 1300.0
        if plate is not None and scene.data['lighting'].get('ambient_from_footage', True):
            look.ambient = self.footage_ambient(plate, scene, frame)
            wlook.sky = look.ambient
            if klook is not None:
                klook.sky = look.ambient
        env = self.liquid_r.environment(wlook)
        if env is not None:
            wlook.sky = env[0]
            if wlook.env_sun:
                strength = max(wlook.sun) or 3.0
                wlook.sun_azimuth, wlook.sun_elevation = env[1], env[2]
                wlook.sun = tuple(c * strength for c in env[3])
            if klook is not None:
                klook.sky, klook.sun_azimuth, klook.sun_elevation, klook.sun = wlook.sky, wlook.sun_azimuth, wlook.sun_elevation, wlook.sun
        VR = self._lava_renderer() if kv is not None else None
        if VR is not None:
            VR.environment(klook)
        comp = scene.comp(frame, mode)
        ep = scene.ember_params(frame)
        t = vol.time
        shutter = lshutter = drop_shutter = 0.0
        if motion_blur:
            drop_shutter = scene.data['render']['shutter_angle'] / 360.0 / scene.fps
            lshutter = drop_shutter * scene.v('domain', 'time_scale', frame)
            if live:
                shutter = lshutter
        r = self.renderer
        if plate is not None:
            r.set_plate(plate)
        elif mode == 'composite':
            r.set_plate(None)
        ptex = r.plate if plate is not None else None
        self._set_liquid_holdout(scene, frame)
        base_seed = frame * 64 if seed is None else seed
        samples = max(1, int(samples))
        surfaces = self.surfaces_for(scene, frame, vol, comp)
        ground = scene.data['domain']['ground']
        LR = self.liquid_r
        r._ensure_fire(fw, fh)
        self._ensure_both(fw, fh)
        # fabric (cloth.py): lit by the fire and shadowing it, seen by the water's march in front of the
        # liquids and through them
        cloth = self.cloth.active and vol.cloth is not None
        grass = self.strands_for(frame)
        self._cloth_drawn = False
        if cloth:
            self.cloth.use_view(None if isinstance(vol.cloth, str) else vol.cloth)
            self.cloth.prepare_light(r.light_dims_for(vol.dims))
        if cloth or grass:
            look.time = t
        T = self._both
        # the set drawn in CG behind and under it all (stage.py), in place of the footage
        footage = plate is not None
        objects = wlook.colliders_look != 'shaded'
        pieces = self.piece_poses(frame)   # (broken things: drawn, and holding out what is behind them, in every view)
        ropes = self.rope_poses(frame)
        matter = self.matter_for(frame) if mode == 'composite' else None
        bolts = scene.bolts(frame) if mode == 'composite' and scene.lights else None   # (lightning)
        stage_on = (stage_mod.wanted(scene, footage, mode, objects=objects) or bool(pieces) or bool(ropes)
                    or matter is not None or bool(bolts))
        r.hold_stage = None
        p_transform, p_gain = INPUT_TRANSFORMS.get(comp.plate_transform, 0), comp.plate_gain
        if stage_on:
            p_transform, p_gain = INPUT_TRANSFORMS['linear'], 1.0
        standins = stage_mod.standin_colours(scene)
        fp_u = (Uniforms().v4(fw, fh).v4(1.0 if (ptex is not None or stage_on) else 0.0, p_transform, *plate_fit)
                .v4(p_gain, comp.fire_gain, comp.smoke_opacity).v4(*comp.bg))
        size = (fw, fh, 1)

        # the fire lights the liquid as it lights the footage around it (composite.wgsl): its point lights,
        # scaled as the footage's light is (relative to the light already on it)
        lamps = look.lamps
        fire_lit, fire_gain = False, None
        if comp.surface_light > 0.0:
            sd = cam.sun_direction(wlook.sun_azimuth, wlook.sun_elevation)
            lum = lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
            ambient = lum(wlook.sky) + 0.5 * lum(wlook.sun) * max(float(sd[1]), 0.0)
            fire_gain = tuple(float(c) * comp.fire_gain * comp.surface_light * ambient for c in comp.tint)
            fire_lit = True

        def one(b, jit, s):
            # (the fire's point lights as r.light left them: it may have reallocated them for this box)
            fire_lights = (r.lights, r.light_count) if fire_lit and r.lights is not None else None
            # the fire, whole: behind the liquids it is seen through them
            r.march(b, vol, cs, fire, look, (fw, fh), jitter=jit, seed=s, shutter=shutter, ground=ground, time=t,
                    surfaces=surfaces)
            b.run(self.k_bfp, [r.beauty, ptex if ptex is not None else r._black, self.gpu.linear, T['fp']], fp_u, size)
            water_plate = T['fp']
            lay = None
            if cloth or grass:
                self._draw_raster(b, r, cs, fire, look, (fw, fh), jit, 0.0, vol, cloth, grass, light_gain=2.0 ** wlook.exposure)
                lay = self.raster.layer(b)
            if kv is not None:
                # the lava over the fire and the footage; then that is what the water refracts
                VR.march(b, kv, cs, fire, klook, comp, (fw, fh), plate=T['fp'], plate_fit=(1.0, 1.0),
                         plate_transform=INPUT_TRANSFORMS['linear'], plate_gain=1.0, jitter=jit, seed=s, shutter=lshutter,
                         ground=ground, time=t, lamps=lamps, fire_lights=fire_lights, fire_gain=fire_gain)
                for src, key in ((r.beauty, 'kb'), (r.emit, 'ke'), (r.aux, 'ka')):
                    b.copy_texture(src, T[key], size)
                b.run(self.k_bover, [T['kb'], T['fp'], T['kp']], Uniforms().v4(fw, fh), size)
                water_plate = T['kp']
            LR.march(b, lv, cs, fire, wlook, comp, (fw, fh), plate=water_plate, plate_fit=(1.0, 1.0),
                     plate_transform=INPUT_TRANSFORMS['linear'], plate_gain=1.0, jitter=jit, seed=s, shutter=lshutter,
                     ground=ground, time=t, lamps=lamps, fire_lights=fire_lights, fire_gain=fire_gain, cloth=lay,
                     standins=standins)
            if wview is not None:
                self.weather_r.cover(b, wview, cs, fire, wlook, (fw, fh))
            LR.drops(b, lv, cs, fire, wlook, (fw, fh), jitter=jit, shutter=drop_shutter)
            LR.rain(b, cs, fire, wlook, (fw, fh), jitter=jit, shutter=drop_shutter, frame=frame,
                    wind=scene.liquid_params(frame).wind, ground=ground)
            if wview is not None:
                self.weather_r.draw(b, wview, cs, fire, wlook, (fw, fh), jitter=jit, shutter=drop_shutter, time=t)
            if kv is not None:
                # the nearer of the lava and the water, per pixel
                for src, key in ((r.beauty, 'lb'), (r.emit, 'le'), (r.aux, 'la')):
                    b.copy_texture(src, T[key], size)
                b.run(self.k_bliq, [T['kb'], T['ke'], T['ka'], T['lb'], T['le'], T['la'], r.beauty, r.emit, r.aux],
                      Uniforms().v4(fw, fh), size)
            for src, key in ((r.beauty, 'lb'), (r.emit, 'le'), (r.aux, 'la')):
                b.copy_texture(src, T[key], size)
            # the fire in front of the liquids, over them
            r.march(b, vol, cs, fire, look, (fw, fh), jitter=jit, seed=s, shutter=shutter, ground=ground, time=t,
                    surfaces=surfaces, limit=T['la'])
            for src, key in ((r.beauty, 'fb'), (r.emit, 'fe'), (r.aux, 'fa')):
                b.copy_texture(src, T[key], size)
            b.run(self.k_bmerge, [T['fb'], T['fe'], T['fa'], T['lb'], T['le'], T['la'], r.beauty, r.emit, r.aux],
                  Uniforms().v4(fw, fh), size)

        def glow(b, u, res, ld):
            # the lava glows into the fire's light volume, so it lights the steam and smoke (and, through the
            # point lights, the water): both_glow.wgsl, from the lava's surface just built
            u.v4(VR.fscale * vol.h / kv.h, lava_k, float(scene.data.get('lava', {}).get('glow_light', 1.0)), 1.0)
            u.v4(vol.h, LR.fscale)
            b.run(self.k_glow, res + [VR.surf, VR.heat_tex, LR.surf], u, ld)

        with self.gpu.batch() as b:
            LR.build(b, lv, wlook)
            LR.sea(b, lv, wlook)
            LR.caustics(b, lv, wlook, fire)
            if kv is not None:
                VR.build(b, kv, klook)
            r.light(b, vol, look, fire, t, emit=glow if kv is not None else None,
                    occluder=self.cloth.occlusion if cloth else None, colliders=surfaces.colliders, meshes=surfaces.meshes)
            stage = None
            if stage_on:
                light = stage_mod.water_light(wlook, comp, look)
                light.lamps = r._lamps_on
                if LR.env_tex is not None and wlook.environment:
                    light.env, light.env_rotation = LR.env_tex, float(wlook.env_rotation)
                    light.env_strength = float(wlook.env_strength) * 2.0 ** float(wlook.exposure)
                ssize = r.plate_size if footage else (W, H)
                stage = ptex = self.stage.draw(b, r, scene, cs, fire, surfaces.colliders, surfaces.meshes, light, comp, ssize,
                                               plate_fit=plate_fit, samples=samples, shutter=lshutter, footage=footage,
                                               vol=vol, ground_y=vol.origin[1], frame=frame, objects=objects,
                                               floor=not wlook.bottomless, pieces=pieces, ropes=ropes, matter=matter, bolts=bolts,
                                               grass=self._strands.ground_map(b) if grass else None, burns=surfaces, final=final)
                if self.stage.has_pieces or self.stage.has_matter:   # the fire and the liquids stop at the pieces and the matter
                    r.hold_stage = self.stage.hold
                    r.hold_stage_matte = bool(footage and r.hold is not None and r.hold_on[0])
            if samples == 1:
                one(b, (0.0, 0.0), base_seed)
            else:
                self._ensure_acc(fw, fh)
                self._ensure_acc2(fw, fh)
                set0, set1 = self._acc[0:3], self._acc[3:6]
                sur0, sur1 = self._acc2[0:3], self._acc2[3:6]
                srcs2 = [r.surf, r.mask, r.mask]
                for i in range(samples):
                    one(b, (_halton(i + 1, 2) - 0.5, _halton(i + 1, 3) - 0.5), base_seed + i)
                    src_in, dst = (set0, set1) if i % 2 == 0 else (set1, set0)
                    b.run(self.k_accum, [r.beauty, r.emit, r.aux, *src_in, *dst], Uniforms().v4(fw, fh, 0, i), size)
                    s_in, s_dst = (sur0, sur1) if i % 2 == 0 else (sur1, sur0)
                    b.run(self.k_accum, [*srcs2, *s_in, *s_dst], Uniforms().v4(fw, fh, 0, i), size)
                last, other = (set1, set0) if (samples - 1) % 2 == 0 else (set0, set1)
                b.run(self.k_accum, [r.beauty, r.emit, r.aux, *last, *other], Uniforms().v4(fw, fh, 1, samples), size)
                b.run(self.k_copy, [*other, r.beauty, r.emit, r.aux], Uniforms().v4(fw, fh), size)
                slast, sother = (sur1, sur0) if (samples - 1) % 2 == 0 else (sur0, sur1)
                b.run(self.k_accum, [*srcs2, *slast, *sother], Uniforms().v4(fw, fh, 1, samples), size)
                b.run(self.k_copy, [*sother, r.surf, r.mask, self._acc2_spare], Uniforms().v4(fw, fh), size)
            if ep.enabled and self.embers.count:
                self.embers.draw(b, r, cs, fire, ep, look, (fw, fh), scene.fps)
            r.defocus(b, comp, cs, fire)
            r.bloom(b, comp.bloom_radius)
            r.composite(b, (W, H), comp, time=t, frame=frame, plate_fit=plate_fit, liquid='both', stage=stage)
            LR.lens_drops(b, wlook, (W, H), time=t)
        r.hold_stage = None
        self.last_render_ms = (time.perf_counter() - t0) * 1000.0
        return cs
