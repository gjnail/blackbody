"""Built-in presets where fire, water and lava meet (domain kind 'both'): water putting out a fire,
lava running into the sea, water thrown on lava, lava setting the grass alight.

presets.py registers them (both_presets(K) returns the specs, K being its keyframe helper). The fire,
the water and the lava share one grid, so each box is sized for the liquids' detail and tall enough
for the steam.
"""
from __future__ import annotations

BOTH_ORDER = ['hose_on_fire', 'lava_sea', 'lava_quench', 'lava_grass']

# the campfire under the hose sits here; its logs are the collider mesh (tools/make_firewood.py LOGS)
FIRE_AT = (0.3, 0.0, 0.0)


def both_presets(K):
    stream = dict(liquid_mode='stream', vel_blend=1.0, embers=False, noise=0.0)
    volume = dict(liquid_mode='fill', vel_blend=0.0, embers=False, noise=0.0)
    # dusk: the lava's glow does most of the lighting; a low, weak sun and a dim blue sky
    dusk = {'sun_on': True, 'sun_intensity': 0.8, 'sun_azimuth': -60.0, 'sun_elevation': 8.0,
            'ambient': (0.32, 0.38, 0.5), 'ambient_intensity': 0.8}
    # cool, damp air: steam off the water hangs white
    steam_air = {'ambient_k': 290.0, 'humidity': 70.0, 'steam_density': 0.5, 'multiple_scattering': 1.0,
                 'detail': 0.5, 'detail_freq': 9.0}
    basalt = {'colliders_look': 'shaded', 'standin_color': (0.05, 0.05, 0.055)}
    fx, fy, fz = FIRE_AT
    log = lambda a, b: dict(position=(a[0] + fx, a[1] + 0.02, a[2] + fz), end=(b[0] + fx, b[1] + 0.02, b[2] + fz))
    return {
        'hose_on_fire': {
            'name': 'Hose on a fire', 'category': 'Fire and liquid', 'size': '1 m campfire',
            'blurb': 'A campfire knocked down by a hose. The stream sweeps across the fire: where it lands the flames '
                     'collapse, the soaked logs stop burning and the hot embers boil the water off in a thick white '
                     'plume of steam that billows up through the smoke; a corner the water missed keeps burning, and '
                     'the water runs off across the ground. Flames behind the stream show through it, refracted.',
            'domain': {'kind': 'both', 'size_x': 3.2, 'size_y': 2.8, 'size_z': 2.0, 'resolution': 160, 'preroll': 2.0,
                       'substeps_max': 12, 'cfl': 1.5},
            'combustion': {'burn_rate': 5.0, 'heat': 0.6, 'soot': 0.35, 'cooling': 2.2, 'flame_life': 0.08, 'steam_yield': 300.0,
                           'vapour_dissipation': 2.2, 'soak': 2.0, 'ember_heat': 150.0, 'rekindle': 8.0, 'steam_expansion': 0.5},
            'motion': {'buoyancy': 5.5, 'turbulence': 3.0, 'turb_freq': 2.5, 'vorticity': 1.6, 'disturbance': 2.0, 'disturb_block': 0.04,
                       'mask_smoke': 0.5},
            'shading': {'flame_k': 1650, 'max_k': 2250, 'smoke_density': 3.0, 'smoke_albedo': (0.55, 0.54, 0.53), 'coal_bed': 1.0,
                        'coal_height': 0.08, 'flame_threshold': 0.1, 'flame_sharpness': 3.2, 'flame_absorption': 10.0,
                        'soot_glow': 0.1, 'detail': 0.5, 'ambient_k': 285.0, 'humidity': 70.0, 'steam_density': 0.5},
            'embers': {'rate': 60, 'launch': 1.6, 'lifetime': 2.0},
            'liquid': {'ww_min_speed': 1.5},
            'water': {'color': (0.75, 0.88, 0.9), 'clarity': 4.0, 'colliders_look': 'shaded', 'standin_color': (0.07, 0.06, 0.05)},
            'lighting': dusk,
            'camera': {'distance': 4.4, 'target_y': 0.7, 'pitch': 7, 'yaw': 18, 'anchor_x': 0.5, 'anchor_y': 0.8, 'focal_mm': 35},
            'colliders': [dict(name='Firewood', shape='mesh', mesh='builtin:firewood.obj', position=FIRE_AT, size=(1.0, 1.0, 1.0),
                               holdout=True)],
            'emitters': [
                dict(noise_rise=1.5, name='Log A', shape='capsule', **log((-0.42, 0.06, -0.10), (0.10, 0.13, 0.02)),
                     size=(0.09, 0.09, 0.09), fuel=10, temperature=0.45),
                dict(noise_rise=1.5, name='Log B', shape='capsule', **log((0.38, 0.06, -0.22), (-0.06, 0.15, 0.05)),
                     size=(0.09, 0.09, 0.09), fuel=10, temperature=0.45, seed=3),
                dict(noise_rise=1.5, name='Log C', shape='capsule', **log((0.16, 0.05, 0.42), (-0.02, 0.16, -0.06)),
                     size=(0.08, 0.08, 0.08), fuel=9, temperature=0.45, seed=5),
                dict(noise_rise=1.5, name='Coal bed', shape='cylinder', position=(fx, 0.04, fz), size=(0.36, 0.04, 0.36),
                     fuel=8, temperature=0.4, seed=7),
                # the hose, from off to the left, swept across the fire and back
                dict(name='Hose', shape='cylinder', position=(-1.45, 0.95, 0.3), size=(0.032, 0.032, 0.032), emits='liquid',
                     jitter=0.05, start=1.0, fade_in=0.0,
                     velocity=K((1.0, (3.7, 0.4, -0.2)), (2.0, (3.7, 0.4, -1.4)), (3.0, (3.6, 0.45, -0.1)), (4.0, (3.7, 0.4, -0.9))),
                     **stream),
            ],
        },
        'lava_sea': {
            'name': 'Lava into the sea', 'category': 'Fire and liquid', 'size': '4 m of shore',
            'blurb': 'Lava running off a rock shelf into the sea. Where it meets the water it boils it: a thick white '
                     'plume of steam billows up off the shoreline, and the lava front chills black and glassy while the '
                     'lava behind it keeps glowing. The hot air over the flow shimmers.',
            'domain': {'kind': 'both', 'size_x': 4.0, 'size_y': 2.6, 'size_z': 2.0, 'resolution': 128, 'preroll': 3.0,
                       'substeps_max': 12, 'cfl': 1.5},
            # (mixing dilutes heat and vapour alike; steady boiling along a shore swells gently, not in bursts)
            'combustion': {'vapour_dissipation': 1.2, 'cooling': 1.2, 'smoke_dissipation': 1.0, 'latent_heat': 1.0,
                           'steam_expansion': 0.15},
            # the warm steam churns like the hot gas does: its small-scale turbulence is masked by temperature
            'motion': {'buoyancy': 4.0, 'turbulence': 3.0, 'turb_freq': 2.0, 'vorticity': 1.6, 'mask_temp': 6.0},
            'shading': {**steam_air},
            'liquid': {'water_level': 0.35, 'settle': True, 'ww_min_speed': 1.0},
            'water': {'color': (0.25, 0.55, 0.55), 'clarity': 1.2, 'murk': 1.0, 'murk_color': (0.08, 0.18, 0.2),
                      'backdrop': 30.0, **basalt},
            'lava': {},
            'lighting': dusk,
            'camera': {'distance': 5.0, 'target_y': 0.5, 'pitch': 17, 'yaw': 32, 'anchor_x': 0.5, 'anchor_y': 0.62, 'focal_mm': 35},
            # a rough basalt bench with a channel down its middle, broken off into the sea (tools/make_lava_shore.py)
            'colliders': [dict(name='Shore', shape='mesh', mesh='builtin:lava_shore.png', size=(10.0, 0.8, 8.0),
                               position=(0.0, -0.016, 0.0), yaw=180.0)],
            'emitters': [dict(name='Lava', shape='box', position=(-1.2, 0.66, 0.0), size=(0.1, 0.07, 0.42),
                              velocity=(1.0, 0.0, 0.0), emits='lava', **stream)],
        },
        'lava_quench': {
            'name': 'Water on lava', 'category': 'Fire and liquid', 'size': '1.5 m lava pool',
            'blurb': 'A pool of lava fed from a vent, and a hose turned on it after a second: the water flashes to steam '
                     'where it lands, drops skitter and sizzle away on the hot surface, and the lava chills black and '
                     'glassy under the stream while the rest keeps glowing.',
            'domain': {'kind': 'both', 'size_x': 2.6, 'size_y': 1.8, 'size_z': 1.6, 'resolution': 136, 'preroll': 1.0,
                       'substeps_max': 12, 'cfl': 1.5},
            'combustion': {'vapour_dissipation': 1.2, 'cooling': 1.2, 'smoke_dissipation': 1.0, 'steam_expansion': 0.3},
            'motion': {'buoyancy': 4.0, 'turbulence': 2.5, 'turb_freq': 2.0, 'vorticity': 1.5, 'mask_temp': 6.0},
            'shading': {**steam_air},
            'liquid': {'ww_min_speed': 1.0},
            'water': {'color': (0.7, 0.85, 0.85), 'clarity': 3.0, **basalt},
            'lava': {},
            'lighting': dusk,
            'camera': {'distance': 3.4, 'target_y': 0.35, 'pitch': 16, 'yaw': 25, 'anchor_x': 0.5, 'anchor_y': 0.7, 'focal_mm': 35},
            'emitters': [
                dict(name='Lava pool', shape='box', position=(0.15, 0.06, 0.0), size=(0.55, 0.06, 0.45), emits='lava', start=-1.0,
                     **volume),
                dict(name='Vent', shape='box', position=(0.65, 0.1, 0.0), size=(0.07, 0.06, 0.2), velocity=(-0.5, 0.0, 0.0),
                     emits='lava', **stream),
                dict(name='Hose', shape='cylinder', position=(-1.15, 0.75, 0.0), size=(0.035, 0.035, 0.035),
                     velocity=(3.2, 0.9, 0.0), emits='liquid', jitter=0.04, start=1.0, **stream),
            ],
        },
        'lava_grass': {
            'name': 'Lava into grass', 'category': 'Fire and liquid', 'size': '4 m flow',
            'blurb': 'A lava flow creeping into dry grass. The air over the lava is hot enough to light the grass just '
                     'ahead of the front, so a line of flame and smoke runs ahead of it and the lava rolls on over the '
                     'burnt ground. Spreading fire is on: the ground is the grass.',
            'domain': {'kind': 'both', 'size_x': 4.0, 'size_y': 2.2, 'size_z': 2.4, 'resolution': 128, 'preroll': 1.5,
                       'substeps_max': 12, 'cfl': 1.5},
            'combustion': {'burn_rate': 6.0, 'heat': 0.6, 'soot': 0.45, 'cooling': 2.4, 'flame_life': 0.08},
            'motion': {'buoyancy': 5.0, 'turbulence': 3.0, 'turb_freq': 2.5, 'vorticity': 1.6, 'disturbance': 2.0, 'disturb_block': 0.04},
            'spread': {'enabled': True, 'ground': True, 'area_x': 3.8, 'area_z': 2.2, 'coverage': 0.8, 'patch_freq': 2.0,
                       'burn_time': 1.8, 'fuel': 10.0, 'heat': 0.5, 'smoke': 0.8, 'catch_temp': 0.28, 'catch_time': 0.3,
                       'creep': 0.15, 'smoulder': 4.0, 'smoulder_smoke': 1.2},
            'shading': {'flame_k': 1650, 'smoke_density': 4.0, 'smoke_albedo': (0.3, 0.29, 0.27)},
            'embers': {'rate': 60},
            'water': {**basalt},
            'lava': {},
            'lighting': dusk,
            'camera': {'distance': 4.6, 'target_y': 0.3, 'pitch': 16, 'yaw': 20, 'anchor_x': 0.5, 'anchor_y': 0.7, 'focal_mm': 35},
            'emitters': [dict(name='Lava', shape='box', position=(-1.75, 0.12, 0.0), size=(0.1, 0.1, 0.35),
                              velocity=(0.9, 0.0, 0.0), emits='lava', **stream)],
        },
    }
