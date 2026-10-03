"""Bullets in the engines (ballistics.py): flown with the rigid bodies (solids.py steps them), then what they did to the
rest of the shot: dust and gun smoke into the gas, the kick they give the matter and the water along their tracks
(bullet_matter.wgsl, bullet_liquid.wgsl), and what the stage draws of them (shot_draw.py), live and from the cache."""
from __future__ import annotations

import numpy as np

from .ballistics import SURFACES

DUST_LAST = 0.25      # s a hit's dust keeps coming
SMOKE_LAST = 0.35     # s a muzzle's smoke keeps coming


class BallisticsEngine:
    """Mixed into Engine (engine.py)."""

    @property
    def shooting(self):
        s = getattr(self, 'solids', None)
        return s is not None and s.shots is not None

    _bullet_media = None

    def _shots_ahead(self, scene, fdt):
        """Before the bullets fly this frame: where their ways cross the matter and the liquid (bullet_media.py)."""
        if not self.shooting:
            return
        m, L = self._shot_media()
        shots = self.solids.shots
        if m is None and L is None:
            shots.ahead = {}
            shots.reahead = None
            return
        if self._bullet_media is None:
            from .bullet_media import BulletMedia
            self._bullet_media = BulletMedia(self.gpu)
        self._bullet_media.look_ahead(shots, fdt, matter=m, liquid=L)
        # (one that turns this frame, off something it glanced off, is looked ahead on again along its new way)
        media = self._bullet_media
        shots.reahead = lambda b: media.reahead(shots, b, matter=m, liquid=L)

    def _shots_kick(self, fdt=None):
        """After: what they did along their tracks through the matter and the liquid goes into them, and their holes
        through the fabric."""
        if not self.shooting:
            return
        cloth = getattr(self, 'cloth', None)
        if (cloth is not None and cloth.active and getattr(cloth, 'placed', False) and fdt is not None
                and any(b.alive or b.trail for b in self.solids.shots.bullets)):
            if self._bullet_media is None:
                from .bullet_media import BulletMedia
                self._bullet_media = BulletMedia(self.gpu)
            self._bullet_media.cloth(self.solids.shots, cloth, fdt)
        if not self.solids.shots.media or self._bullet_media is None:
            self.solids.shots.media = []
            return
        m, L = self._shot_media()
        self._bullet_media.kick(self.solids.shots, matter=m, liquid=L)

    def _shot_media(self):
        m = getattr(self, '_matter', None)
        if m is None or not m.active or not m.count:
            m = None
        L = getattr(self, 'liquid', None) if getattr(self, 'kind', '') in ('liquid', 'both') else None
        if L is not None and (getattr(L, 'dims', None) is None or not L.h):
            L = None
        return m, L

    def shot_view(self, frame):
        """What the bullets leave to draw at `frame` (Ballistics.view): live, or from the cache. None: nothing."""
        solids = getattr(self, 'solids', None)
        v = None
        if self.sim_frame == frame and solids is not None and solids.shots is not None:
            v = solids.shots.view(solids) or None
        else:
            entry = self.cache.get(frame) if self.cache is not None else None
            st = entry.get('solids') if entry is not None else None
            if isinstance(st, dict) and st.get('shots_view'):
                v = st['shots_view']
        self._cloth_holes(v)
        return v

    def _cloth_holes(self, view):
        """The holes bullets made through the cloth at this frame, for its drawing (cloth_draw.wgsl: binding 19)."""
        cloth = getattr(self, 'cloth', None)
        if cloth is None or not getattr(cloth, 'active', False):
            return
        H = view.get('cloth_holes') if view else None
        n = 0 if H is None else len(H)
        if n == 0 and getattr(cloth, 'holes_buf', None) is None:
            return
        from .ballistics import CLOTH_HOLES_MOST
        if getattr(cloth, 'holes_buf', None) is None:
            cloth.holes_buf = self.gpu.buffer(16 * (1 + CLOTH_HOLES_MOST), 'cloth-holes')
        a = np.zeros((1 + CLOTH_HOLES_MOST, 4), np.float32)
        a[0, 0] = n
        if n:
            a[1:1 + n] = np.asarray(H, np.float32)[-CLOTH_HOLES_MOST:]
        self.gpu.write_buffer(cloth.holes_buf, a)

    def _shot_puffs(self, scene, fdt, substeps):
        """Dust where bullets hit (as much as the energy they left and the material's dustiness give: a cloud off
        concrete and plaster, a wisp off wood, none off steel) and gun smoke at the muzzles, for each substep of the
        frame just flown: [[EmitterGPU]] or None. (The gas carries them: fire and both scenes.)"""
        from ..scene.materials import material
        from .solver import EmitterGPU
        B = self.solids.shots if self.shooting else None
        if B is None:
            return None
        t1 = B.now
        t0 = t1 - fdt
        hits = [i for i in B.impacts if i.time > t0 - DUST_LAST and i.owner[0] not in ('water',)]
        smoke = [f for f in B.flashes if f[3] == 'muzzle' and f[0] > t0 - SMOKE_LAST]
        if not hits and not smoke:
            return None
        out = []
        for k in range(substeps):
            ti = t0 + (k + 0.5) * fdt / substeps
            ems = []
            for j, imp in enumerate(hits):
                if not (imp.time <= ti < imp.time + DUST_LAST):
                    continue
                s = SURFACES.get(imp.surface)
                if imp.owner[0] in ('collider', 'body', 'piece') and 0 <= int(imp.owner[1]) < len(scene.colliders):
                    dust = material(scene.colliders[int(imp.owner[1])].get('material', 'wood')).dust
                elif imp.owner[0] in ('ground', 'matter'):
                    dust = 1.5 if s is not None and s.debris == 'dirt' else 0.8
                else:
                    dust = 0.5
                if dust <= 0.0 or imp.energy < 1.0:
                    continue
                fade = 1.0 - (ti - imp.time) / DUST_LAST
                r = float(np.clip(0.02 + 0.004 * imp.energy ** 0.5, 0.025, 0.18))
                c = np.asarray(imp.pos) + np.asarray(imp.normal) * r
                ems.append(EmitterGPU(shape='sphere', pos=tuple(float(x) for x in c), size=(r, r, r), fuel=0.0, temp=0.0,
                                      smoke=float(np.clip(dust * imp.energy / 60.0, 0.3, 12.0)) * fade,
                                      vel=tuple(float(x) for x in np.asarray(imp.normal) * 4.0 * fade),
                                      radial=1.5, vel_blend=0.5 * fade, noise=0.9, noise_freq=12.0, seed=float(j * 13 + k)))
            for j, (t, pos, strength, kind, d) in enumerate(smoke):
                if not (t <= ti < t + SMOKE_LAST):
                    continue
                fade = 1.0 - (ti - t) / SMOKE_LAST
                c = np.asarray(pos) + np.asarray(d) * 0.08
                hot = ti < t + 0.004
                ems.append(EmitterGPU(shape='sphere', pos=tuple(float(x) for x in c), size=(0.04, 0.04, 0.04),
                                      fuel=4.0 if hot else 0.0, temp=1.0 if hot else 0.0,
                                      smoke=1.5 * fade * strength, vel=tuple(float(x) for x in np.asarray(d) * 6.0 * fade),
                                      radial=0.8, vel_blend=0.6, noise=0.8, noise_freq=10.0, seed=float(97 + j)))
            out.append(ems)
        return out if any(out) else None
