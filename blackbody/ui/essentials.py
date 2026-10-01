"""Essentials: the few settings that matter most for each kind of simulation, on one page.

Each entry is (section, key) or (section, key, label): the label replaces the registry's where a
plainer word helps. A group may carry a condition on the scene, so it only shows when it applies
(the sea settings only for a scene with waves, say). Everything else is a click away in its section
or a search away.
"""
from __future__ import annotations

import dataclasses

from ..scene.params import applies, param

PLACE = ('Place it in your shot', [('camera', 'scale'), ('camera', 'focal_mm'), ('camera', 'pitch')])
DETAIL = ('Speed and detail', [('domain', 'time_scale', 'Speed'), ('domain', 'resolution', 'Detail (voxels)'),
                               ('domain', 'preroll'), ('render', 'motion_blur')])

FIRE = [
    PLACE,
    ('The fire', [('combustion', 'fuel_scale', 'Amount of fire'), ('combustion', 'cooling'), ('combustion', 'soot', 'Smoke'),
                  ('motion', 'turbulence'), ('motion', 'puffing'), ('motion', 'buoyancy')]),
    ('Wind', [('motion', 'wind_speed'), ('motion', 'wind_dir'), ('motion', 'gust')]),
    ('Look', [('shading', 'exposure'), ('shading', 'flame_k'), ('shading', 'smoke_density'), ('shading', 'smoke_albedo'),
              ('embers', 'enabled', 'Embers'), ('embers', 'rate', 'Ember rate')]),
    ('Spreading fire', [('spread', 'creep'), ('spread', 'catch_time'), ('spread', 'burn_time'), ('spread', 'spotting')],
     lambda s: s.data['spread']['enabled']),
    ('Blend with the footage', [('composite', 'light_cast'), ('composite', 'haze'), ('composite', 'bloom'),
                                ('composite', 'visibility'), ('composite', 'grain')]),
    DETAIL,
]

LIQUID = [
    PLACE,
    ('The liquid', [('liquid', 'flip'), ('liquid', 'viscosity'), ('liquid', 'surface_tension'), ('liquid', 'whitewater'),
                    ('liquid', 'ww_amount', 'Whitewater amount')]),
    ('Sea', [('liquid', 'ocean_height'), ('liquid', 'ocean_length'), ('liquid', 'ocean_dir'), ('liquid', 'ocean_chop'),
             ('liquid', 'wind_speed'), ('water', 'whitecaps'), ('water', 'sea_foam')],
     lambda s: s.data['liquid']['ocean_height'] > 0 or s.data['liquid']['swell_height'] > 0),
    ('Weather', [('weather', 'precip'), ('weather', 'rate'), ('weather', 'size'), ('weather', 'gust')],
     lambda s: s.data['weather']['precip'] != 'none'),
    ('Heat, ice and steam', [('liquid', 'liquid_temp'), ('liquid', 'air_temp'), ('liquid', 'ground_temp'),
                             ('liquid', 'heat_speed')],
     lambda s: s.data['liquid']['thermal']),
    ('Look', [('water', 'color'), ('water', 'clarity'), ('water', 'murk'), ('water', 'roughness'), ('water', 'reflection'),
              ('water', 'foam'), ('water', 'spray')]),
    ('Blend with the footage', [('water', 'wet_darken'), ('water', 'shadow'), ('water', 'caustics'), ('composite', 'bloom'),
                                ('composite', 'grain')]),
    DETAIL,
]

BOTH = [
    PLACE,
    ('The fire', [('combustion', 'fuel_scale', 'Amount of fire'), ('combustion', 'cooling'), ('combustion', 'soot', 'Smoke'),
                  ('motion', 'turbulence')]),
    ('The liquid', [('liquid', 'flip'), ('liquid', 'viscosity'), ('liquid', 'whitewater')]),
    ('Water on fire', [('combustion', 'water_douse'), ('combustion', 'soak'), ('combustion', 'ember_heat'),
                       ('combustion', 'steam_expansion')]),
    ('Lava', [('lava', 'viscosity'), ('lava', 'cooling'), ('lava', 'glow'), ('lava', 'crust')],
     lambda s: any(e.get('emits') == 'lava' for e in s.emitters)),
    ('Look', [('shading', 'exposure'), ('water', 'color'), ('water', 'clarity'), ('shading', 'smoke_density')]),
    ('Blend with the footage', [('composite', 'light_cast'), ('composite', 'haze'), ('composite', 'bloom'),
                                ('composite', 'grain')]),
    DETAIL,
]

CLOUD = [
    ('The sky', [('atmosphere', 'time_lapse'), ('atmosphere', 'surface_t'), ('atmosphere', 'surface_rh'),
                 ('atmosphere', 'free_rh'), ('atmosphere', 'heat_flux'), ('atmosphere', 'wind')]),
    ('Look', [('sky', 'brightness'), ('sky', 'density'), ('sky', 'silver'), ('sky', 'visibility'), ('sky', 'draw_sky')]),
    ('Camera', [('camera', 'focal_mm'), ('camera', 'pitch'), ('camera', 'distance')]),
    ('Detail', [('domain', 'resolution', 'Detail (voxels)'), ('domain', 'preroll')]),
]

ESSENTIALS = {'fire': FIRE, 'liquid': LIQUID, 'both': BOTH, 'cloud': CLOUD}

INTRO = {
    'fire': 'The settings that matter most for fire and smoke.',
    'liquid': 'The settings that matter most for liquids.',
    'both': 'The settings that matter most for fire and liquid together.',
    'cloud': 'The settings that matter most for skies and storms.',
}


def groups(scene):
    """[(title, [(section, key, Param)])] for the scene's kind, leaving out what does not apply."""
    out = []
    for g in ESSENTIALS.get(scene.kind, FIRE):
        title, items = g[0], g[1]
        cond = g[2] if len(g) > 2 else None
        if cond is not None:
            try:
                if not cond(scene):
                    continue
            except (KeyError, TypeError):
                continue
        rows = []
        for it in items:
            sec, key = it[0], it[1]
            if not applies(sec, key, scene.kind):
                continue
            try:
                p = param(sec, key)
            except KeyError:
                continue
            if len(it) > 2:
                p = dataclasses.replace(p, label=it[2])
            rows.append((sec, key, p))
        if rows:
            out.append((title, rows))
    return out
