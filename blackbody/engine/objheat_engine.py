"""The engine's side of heat that crosses between things (objheat.py, radiant.py): the objects' own heat, and the frame's
shared radiant sources everything warmed by radiation reads.

Per frame, after the rigid bodies have moved and before the frame's batch:
  _step_heat     the objects warm and cool (they read back what they meet), and their faces go into the radiant sources
Inside the batch, before the substeps:
  _build_radiant the frame's radiant sources: the gas as it is, what matter and lava added last frame, the objects
And everything that touches an object reads its temperature from _object_heat_for (matter, the liquid's heat, the
weather's snow cover), the gas beside hot objects is heated through the solver hook (_heat_hook), and what the matter
and the water took from each object is handed back to it for the next frame.
"""
from __future__ import annotations

import numpy as np

from .gpu import Uniforms


class HeatEngine:
    _objheat = None
    _radiant = None
    _k_objair = None

    @property
    def objheat(self):
        if self._objheat is None:
            from .objheat import ObjectHeat
            self._objheat = ObjectHeat(self.gpu)
        return self._objheat

    @property
    def radiant(self):
        if self._radiant is None:
            from .radiant import Radiant
            self._radiant = Radiant(self.gpu)
        return self._radiant

    def _prepare_heat(self, scene):
        """The objects' heat for this scene; True if it changed (the simulation starts over)."""
        changed = self.objheat.configure(scene)
        self.cloth.radiant = self.radiant
        return changed

    def _reset_heat(self):
        if self._objheat is not None:
            self._objheat.reset()

    def _heat_on(self):
        return self._objheat is not None and self._objheat.active

    def _object_heat_for(self, scene):
        """Each object's (surface temperature K, effusivity), in colliders_gpu's order: its own heat when objects warm and
        cool, else its Temperature as set (Scene.collider_heat)."""
        if self._heat_on():
            return self._objheat.collider_heat()
        return scene.collider_heat()

    def _object_temps_c(self, scene, default):
        """Each object's surface temperature (C), for the liquid's heat and the weather (their collider_temps)."""
        if self._heat_on():
            return self._objheat.temps_c()
        return default

    def _ground_for(self, scene, origin_y):
        """(height m, temperature K, effusivity) of the ground under the box, or None without one."""
        if not scene.data['domain'].get('ground', True):
            return None
        oh = self.objheat
        if not oh.things:
            oh.configure(scene)
        return (float(origin_y), oh.ground_k, oh.ground_e)

    def _step_heat(self, scene, frame, fdt, gas=None, liquid=None, lava=None, water_thermal=False):
        """The objects through the frame (before its batch: reads back), where the rigid bodies have them at its end."""
        if not self._heat_on():
            self.radiant.set_objects([])
            return
        oh = self._objheat
        cols = scene.colliders_gpu(frame, self.solids.overrides() if self.solids.active else None)
        try:
            oh.wind = tuple(float(x) for x in scene.liquid_wind(frame)) if scene.kind != 'fire' else (0.0, 0.0, 0.0)
        except (AttributeError, KeyError, TypeError):
            oh.wind = (0.0, 0.0, 0.0)
        box = gas if (gas is not None and getattr(gas, 'dims', None)) else liquid
        oh.ground_y = float(box.origin[1]) if box is not None and getattr(box, 'dims', None) else 0.0
        oh.step(cols, fdt, gas, liquid, lava, self.radiant if self.radiant.built else None, water_thermal)
        self.radiant.set_objects(oh.radiant_sources())

    def _build_radiant(self, b, scene, frame, gas=None, box=None):
        """The frame's radiant sources, in batch b (before the substeps): over the gas's box (with its radiation), or
        over `box` (a Liquid, or anything with origin, dims and h) without gas."""
        look = scene.look(frame)
        r = self.radiant
        if gas is not None and getattr(gas, 'dims', None):
            r.layout(gas.origin, gas.dims, gas.h, gas=True)
            return r.build(b, gas.scal[0], gas.dims, float(look.ambient_k))
        if box is not None and getattr(box, 'dims', None):
            r.layout(box.origin, box.dims, box.h, gas=False)
            return r.build(b, None, None, float(look.ambient_k))
        return False

    def _heat_back(self):
        """After the frame: what the matter and the water took from each object, for its heat next frame."""
        if not self._heat_on():
            return
        oh = self._objheat
        m = getattr(self, '_matter', None)
        if m is not None and getattr(m, '_objects_heat_on', False):
            oh.matter_j = m.object_heat()
            m._objects_heat_on = False

    def _water_heat_back(self, liquid):
        if not self._heat_on() or liquid is None:
            return
        st = getattr(liquid, 'thermal_stats', None) or {}     # (a property: the last measured frame's)
        if 'object_heat_j' in st:
            self._objheat.liquid_j = np.asarray(st['object_heat_j'], float)

    _k_lavarad = None

    def _liquid_radiates(self, V, fresh_k, ambient_k):
        """A lava solver's glowing surface into the shared radiant sources for the next frame (lava_rad.wgsl): it
        scorches cloth, warms matter and heats objects beside it."""
        if self.radiant.grid is None or getattr(V, 'HEAT', None) is None:
            return
        if self._k_lavarad is None:
            self._k_lavarad = self.gpu.kernel('lava_rad.wgsl', ['utex3d', 'utex3d', 'buf'], workgroup=(4, 4, 4))
        u = V._grid(0.0).v4(ambient_k, fresh_k, float(V._prm.ppc))
        self.radiant.uniform_block(u)
        with self.gpu.batch() as b:
            b.run(self._k_lavarad, [V.DENS, V.HEAT, self.radiant.RA], u, V.dims)

    # -- the gas beside hot objects ----------------------------------------------------------------

    def _heat_hook_on(self):
        oh = self._objheat
        return oh is not None and oh.active and any(t.skin > oh.ambient_k + 20.0 for t in oh.things)

    def _heat_hook(self, b, solver, dt, stage):
        """Solver.cloth_hook: at the sources, the gas beside hot objects heated by them (obj_air.wgsl)."""
        if stage != 'sources' or not self._heat_hook_on():
            return
        from .radiant import FLAME_K, MAX_K
        from .solver import MAX_COLLIDERS, pack_colliders
        if self._k_objair is None:
            self._k_objair = self.gpu.kernel('obj_air.wgsl', ['utex3d', 'utex3d', 'st3d:rgba16float:w'], workgroup=(4, 4, 4))
        oh = self._objheat
        u = solver._grid(dt).v4(dt, oh.ambient_k, FLAME_K, MAX_K)
        cols = list(solver.colliders)[:MAX_COLLIDERS]
        pack_colliders(u, cols, solver.meshes)
        t = [x for x in oh.temps_k()[:len(cols)]] + [0.0] * MAX_COLLIDERS
        u.raw([float(x) for x in t[:MAX_COLLIDERS]])
        b.run(self._k_objair, [solver.scal[0], solver.meshes.atlas, solver.scal[1]], u, solver.dims)
        solver.scal.reverse()

    def _stage_heat(self, frame, cols):
        """What the stage needs to draw hot objects at `frame` (stage.py Stage.heat): each object's (surface K,
        emissivity), and the faces of the glowing ones where `cols` has them; None when none is hot enough to glow."""
        if not self._heat_on():
            return None
        entry = self.cache.get(frame) if getattr(self, 'cache', None) is not None else None
        temps = self.object_temps_at(entry)
        if not temps or max(temps) <= 700.0:
            return None
        oh = self._objheat
        return {'rows': [(T, t.eps) for T, t in zip(temps, oh.things)], 'faces': oh.faces_at(list(cols or []), temps)}

    # -- cache ---------------------------------------------------------------------------------------

    def _snapshot_heat(self, entry):
        if self._heat_on() and entry is not None:
            entry['objheat'] = self._objheat.state()

    def object_temps_at(self, entry=None):
        """The objects' surface temperatures (K) to draw: a cached frame's, or the live ones."""
        if entry is not None and 'objheat' in entry:
            return [float(x) for x in np.asarray(entry['objheat'])[:, 0]]
        return self._objheat.temps_k() if self._heat_on() else []
