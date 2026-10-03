"""Built-in presets with guns firing (engine/ballistics.py): a shooting range, a bullet through glass in slow motion,
bottles on a fence, a swinging steel target, a machine gun's tracers.

presets.py registers them (shot_presets(K) returns the specs). The shots sit outside the box, where the gun would be:
only what they hit needs to be in it. The gas is there for the dust the hits throw up and the guns' smoke.
"""
from __future__ import annotations

SHOT_ORDER = ['shooting_range', 'glass_slowmo', 'bottle_shoot', 'steel_gong', 'machine_gun']

# daylight on an outdoor range: a high sun, a bright sky
DAY = {'sun_on': True, 'sun_intensity': 3.0, 'sun_elevation': 40.0, 'sun_azimuth': 35.0, 'ambient_intensity': 2.5}
# still air that lets the dust hang: little buoyancy, gentle turbulence
DUST_AIR = {'buoyancy': 0.6, 'turbulence': 0.8, 'turb_freq': 3.0, 'vorticity': 1.0}
DUST_LOOK = {'smoke_density': 3.0, 'smoke_albedo': (0.55, 0.53, 0.5), 'exposure': 0.0}


def _bottle(name, x, y, z, colour=(0.18, 0.42, 0.2)):
    """A glass bottle standing at (x, y, z) (its foot): a hollow glass cylinder open at the top, in one piece until it
    is hit."""
    return dict(name=name, shape='cylinder', position=(x, y + 0.12, z), size=(0.037, 0.12, 0.037), hollow=0.003,
                opening=(0.045, 0.01, 0.045), opening_at=(0.0, 0.12, 0.0), material='glass', own_colour=True,
                colour=colour, dynamic=True, breakable=True, pieces=40, density=900.0)


def shot_presets(K):
    return {
        'shooting_range': {
            'name': 'Shooting range', 'category': 'Bullets', 'size': '9 mm pistol at 6 m',
            'blurb': 'A pistol works along a row of targets: a window pane takes a hole with its web of cracks and stays '
                     'standing, a steel plate rings and swings with a grey star of splashed lead on it and sparks off it, '
                     'a can is knocked off its post, a pine board is holed with splinters torn out of its back, and a '
                     'concrete block loses a crater in a puff of dust. Shot settings: Cartridge, Rounds, Rate of fire.',
            'render': {'end': 72},
            'domain': {'size_x': 4.0, 'size_y': 2.0, 'size_z': 2.0, 'resolution': 96, 'preroll': 0.0},
            'motion': DUST_AIR, 'shading': DUST_LOOK,
            'composite': {'backdrop': 'stage', 'floor': 'dirt'},
            'lighting': DAY,
            'camera': {'distance': 4.2, 'target_y': 0.75, 'pitch': 7.0, 'yaw': -18.0, 'focal_mm': 35, 'use_anchor': False},
            'emitters': [],
            'colliders': [
                dict(name='Pane', shape='box', position=(-1.5, 0.95, 0.0), size=(0.3, 0.3, 0.003), material='glass',
                     breakable=True, fracture='shards', pieces=60, held='edges'),
                dict(name='Pane frame', shape='box', position=(-1.5, 0.32, 0.0), size=(0.05, 0.33, 0.05), material='wood'),
                dict(name='Gong', shape='box', position=(-0.65, 0.95, 0.0), size=(0.18, 0.18, 0.006), material='steel',
                     own_colour=True, colour=(0.75, 0.73, 0.68), joint='hinge', joint_to='Gong bar',
                     joint_at=(0.0, 0.2, 0.0), joint_axis=(1.0, 0.0, 0.0), joint_friction=0.05),
                dict(name='Gong bar', shape='box', position=(-0.65, 1.17, 0.0), size=(0.25, 0.015, 0.015), material='steel'),
                dict(name='Gong post', shape='box', position=(-0.65, 0.58, -0.05), size=(0.02, 0.6, 0.02), material='steel'),
                dict(name='Post', shape='box', position=(0.15, 0.4, 0.0), size=(0.04, 0.4, 0.04), material='wood'),
                dict(name='Can', shape='cylinder', position=(0.15, 0.861, 0.0), size=(0.033, 0.06, 0.033), dynamic=True,
                     material='aluminium', density=37.0, own_colour=True, colour=(0.75, 0.08, 0.06)),
                dict(name='Board', shape='box', position=(0.85, 0.75, 0.0), size=(0.1, 0.45, 0.011), material='wood'),
                dict(name='Board foot', shape='box', position=(0.85, 0.15, -0.1), size=(0.15, 0.15, 0.1), material='concrete'),
                dict(name='Block', shape='box', position=(1.5, 0.3, 0.0), size=(0.25, 0.3, 0.2), material='concrete'),
            ],
            'shots': [dict(name='Pistol', round='9mm', position=(x * 0.35, 1.2, 6.0), aim=aim, start=t, count=n, rate=240.0,
                           scatter=0.15, seed=k)
                      for k, (x, aim, t, n) in enumerate(((-1.5, (-1.5, 0.98, 0.0), 0.3, 1), (-0.65, (-0.62, 0.95, 0.0), 0.9, 2),
                                                          (0.15, (0.15, 0.87, 0.0), 1.6, 1), (0.85, (0.85, 0.85, 0.0), 2.0, 2),
                                                          (1.5, (1.5, 0.35, 0.2), 2.5, 2)))],
        },
        'glass_slowmo': {
            'name': 'Bullet through glass', 'category': 'Bullets', 'size': '9 mm, slowed 400 times',
            'blurb': 'A 9 mm bullet through a window pane, slowed down 400 times: the bullet flies in, a white cone of '
                     'crushed glass bursts out of the back of the pane, cracks run out from the hole and splinters spray '
                     'after the bullet, which flies on a little slower. Domain › Time scale sets the slow motion.',
            'render': {'end': 60},
            'domain': {'size_x': 1.2, 'size_y': 1.2, 'size_z': 1.2, 'resolution': 48, 'preroll': 0.0, 'time_scale': 0.0025},
            'composite': {'backdrop': 'stage', 'floor': 'concrete'},
            'lighting': {**DAY, 'sun_azimuth': 120.0, 'sun_elevation': 30.0},
            'camera': {'distance': 0.9, 'target_y': 0.6, 'pitch': 3.0, 'yaw': 55.0, 'focal_mm': 50, 'use_anchor': False},
            'emitters': [],
            'colliders': [dict(name='Pane', shape='box', position=(0.0, 0.6, 0.0), size=(0.25, 0.25, 0.003), material='glass',
                               breakable=True, fracture='shards', pieces=70, held='edges')],
            'shots': [dict(name='Pistol', round='9mm', position=(0.02, 0.6, 2.0), aim=(0.0, 0.6, 0.0), start=0.004, scatter=0.0,
                           flash=False)],
        },
        'bottle_shoot': {
            'name': 'Bottles on a fence', 'category': 'Bullets', 'size': '.22 rifle at 10 m',
            'blurb': 'A .22 rifle picks off five glass bottles standing on a fence rail, one a second: each one bursts, '
                     'its pieces spraying off the rail and tumbling to the ground, where they lie.',
            'render': {'end': 132},
            'domain': {'size_x': 3.0, 'size_y': 1.6, 'size_z': 1.4, 'resolution': 80, 'preroll': 0.0},
            'motion': DUST_AIR, 'shading': DUST_LOOK,
            'composite': {'backdrop': 'stage', 'floor': 'grass'},
            'lighting': DAY,
            'camera': {'distance': 3.2, 'target_y': 0.95, 'pitch': 5.0, 'yaw': -12.0, 'focal_mm': 35, 'use_anchor': False},
            'emitters': [],
            'colliders': [dict(name='Rail', shape='box', position=(0.0, 0.9, 0.0), size=(1.2, 0.03, 0.05), material='wood'),
                          dict(name='Post', shape='box', position=(-1.1, 0.44, 0.0), size=(0.05, 0.44, 0.05), material='wood'),
                          dict(name='Post 2', shape='box', position=(1.1, 0.44, 0.0), size=(0.05, 0.44, 0.05), material='wood')]
                         + [_bottle(f'Bottle {k + 1}', x, 0.93, 0.0, c) for k, (x, c) in enumerate(
                             ((-0.8, (0.18, 0.42, 0.2)), (-0.4, (0.45, 0.25, 0.08)), (0.0, (0.82, 0.84, 0.8)),
                              (0.4, (0.18, 0.42, 0.2)), (0.8, (0.08, 0.2, 0.42))))],
            'shots': [dict(name=f'Rifle {k + 1}', round='22lr', position=(0.5, 1.1, 10.0), aim=(x, 1.02, 0.0), start=0.5 + k,
                           scatter=0.02, seed=k, flash=False)
                      for k, x in enumerate((-0.8, -0.4, 0.0, 0.4, 0.8))],
        },
        'steel_gong': {
            'name': 'Steel gong', 'category': 'Bullets', 'size': '9 mm at 7 m',
            'blurb': 'A hanging steel plate shot six times: each bullet flattens on it in a flash, splashes its lead in a '
                     'grey star across the plate and throws sparks; the plate swings back on its hinge with every hit.',
            'render': {'end': 96},
            'domain': {'size_x': 1.6, 'size_y': 1.6, 'size_z': 1.2, 'resolution': 64, 'preroll': 0.0},
            'motion': DUST_AIR, 'shading': DUST_LOOK,
            'composite': {'backdrop': 'stage', 'floor': 'dirt'},
            'lighting': {**DAY, 'sun_intensity': 1.5, 'ambient_intensity': 1.2},
            'camera': {'distance': 1.9, 'target_y': 0.85, 'pitch': 5.0, 'yaw': -25.0, 'focal_mm': 40, 'use_anchor': False},
            'emitters': [],
            'colliders': [dict(name='Gong', shape='box', position=(0.0, 0.9, 0.0), size=(0.2, 0.2, 0.008), material='steel',
                               own_colour=True, colour=(0.82, 0.8, 0.74), joint='hinge', joint_to='Bar',
                               joint_at=(0.0, 0.22, 0.0), joint_axis=(1.0, 0.0, 0.0), joint_friction=0.05),
                          dict(name='Bar', shape='box', position=(0.0, 1.14, 0.0), size=(0.35, 0.015, 0.015), material='steel'),
                          dict(name='Post', shape='box', position=(-0.33, 0.57, -0.02), size=(0.02, 0.57, 0.02), material='steel'),
                          dict(name='Post 2', shape='box', position=(0.33, 0.57, -0.02), size=(0.02, 0.57, 0.02), material='steel')],
            'shots': [dict(name='Pistol', round='9mm', position=(0.3, 1.2, 7.0), aim=(0.0, 0.92, 0.0), start=0.4, count=6,
                           rate=150.0, scatter=0.12)],
        },
        'machine_gun': {
            'name': 'Machine gun at dusk', 'category': 'Bullets', 'size': '5.56 mm, 900 rounds a minute',
            'blurb': 'A burst of 25 rounds with tracers at dusk, walked across a concrete wall and the dirt in front of it: '
                     'red streaks, sparks and chips off the concrete, bullets skipping off the ground, dust rising.',
            'render': {'end': 72},
            'domain': {'size_x': 5.0, 'size_y': 2.5, 'size_z': 3.0, 'resolution': 96, 'preroll': 0.0},
            'motion': DUST_AIR, 'shading': {**DUST_LOOK, 'exposure': 0.5},
            'composite': {'backdrop': 'stage', 'floor': 'dirt'},
            'lighting': {'sun_on': True, 'sun_intensity': 0.5, 'sun_elevation': 6.0, 'sun_azimuth': -70.0,
                         'ambient': (0.3, 0.36, 0.5), 'ambient_intensity': 0.7},
            'camera': {'distance': 6.0, 'target_y': 0.8, 'pitch': 8.0, 'yaw': -30.0, 'focal_mm': 35, 'use_anchor': False},
            'emitters': [],
            'colliders': [dict(name='Wall', shape='box', position=(0.0, 0.9, -1.0), size=(2.2, 0.9, 0.15), material='concrete')],
            'shots': [dict(name='Machine gun', round='556', position=K((0.0, (-1.0, 1.4, 12.0)), (2.0, (1.0, 1.4, 12.0)), interp='linear'),
                           aim=K((0.0, (-1.8, 0.2, 0.0)), (2.0, (1.8, 1.1, -1.0)), interp='linear'), start=0.3, count=25, rate=900.0,
                           scatter=0.25, tracer=True)],
        },
    }
