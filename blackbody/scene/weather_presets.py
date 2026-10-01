"""Built-in presets for weather (Weather section, engine/weather.py): snow, hail, sleet and freezing rain
falling on liquid scenes, at real sizes, rates and temperatures.

presets.py registers them (weather_presets(K) returns the specs, K being its keyframe helper).
"""
from __future__ import annotations

WEATHER_ORDER = ['snow_pond', 'hail_pond']


def weather_presets(K):
    winter = {'sun_on': True, 'sun_intensity': 1.6, 'sun_azimuth': -50.0, 'sun_elevation': 18.0, 'sun_color': (1.0, 0.9, 0.78),
              'ambient': (0.62, 0.66, 0.74)}
    summer = {'sun_on': True, 'sun_intensity': 1.2, 'sun_azimuth': 30.0, 'sun_elevation': 35.0, 'sun_color': (1.0, 0.97, 0.92),
              'ambient': (0.42, 0.45, 0.5)}
    volume = dict(liquid_mode='fill', vel_blend=0.0, embers=False, noise=0.0)
    basin = [dict(name='Banks', shape='mesh', mesh='builtin:pond_basin.png', size=(3.2, 0.4, 2.4), position=(0.0, -0.03, 0.0),
                  temperature=0.0),
             dict(name='Rock', shape='sphere', size=(0.14, 0.14, 0.14), position=(-0.35, 0.1, 0.2), temperature=-2.0),
             dict(name='Stone', shape='sphere', size=(0.08, 0.08, 0.08), position=(0.4, 0.15, -0.25), temperature=-2.0)]
    return {
        'snow_pond': {
            'name': 'Snow on a pond', 'category': 'Weather', 'size': '1.6 m pond, -4 C',
            'blurb': 'Steady snow on a small pond on a still, grey winter day (-4 C). The flakes drift down, tumbling and fluttering, and '
                     'settle into a white blanket on the frozen banks and the rocks, while those that reach the water (2 C) vanish into it '
                     'as they touch, cooling it.',
            'domain': {'kind': 'liquid', 'size_x': 1.6, 'size_y': 0.4, 'size_z': 1.2, 'resolution': 112, 'preroll': 0.3,
                       'open_sides': False, 'substeps_max': 10, 'cfl': 1.5},
            # (the ground frozen at the freezing point, as the banks are: snow lies on it and does not melt)
            'liquid': {'thermal': True, 'liquid_temp': 2.0, 'air_temp': -4.0, 'air_humidity': 90.0, 'ground_temp': 0.0,
                       'wind_speed': 1.0, 'wind_dir': 60.0, 'settle': True},
            # (dark, peaty winter water; the banks drawn snow-grey, as the water reflects them under their snow)
            'water': {'backdrop': 6.0, 'clarity': 0.15, 'color': (0.25, 0.4, 0.42), 'ripple': 0.2, 'colliders_look': 'shaded',
                      'standin_color': (0.62, 0.64, 0.68)},
            'weather': {'precip': 'snow', 'rate': 3.0, 'size': 6.0, 'area': 4.8, 'turbulence': 0.5, 'gust': 0.3, 'lying': 4.0},
            'lighting': dict(winter),
            'camera': {'distance': 2.0, 'target_y': 0.1, 'pitch': 36, 'yaw': 15, 'anchor_x': 0.5, 'anchor_y': 0.6, 'focal_mm': 35},
            # the pond's hollow (tools/make_pond_basin.py), its banks at the freezing point
            'colliders': [
                dict(name='Banks', shape='mesh', mesh='builtin:pond_basin.png', size=(3.2, 0.4, 2.4),
                     position=(0.0, -0.03, 0.0), temperature=0.0),
                dict(name='Rock', shape='sphere', size=(0.14, 0.14, 0.14), position=(-0.35, 0.1, 0.2), temperature=-2.0),
                dict(name='Stone', shape='sphere', size=(0.08, 0.08, 0.08), position=(0.4, 0.15, -0.25), temperature=-2.0),
            ],
            'emitters': [
                dict(name='Pond', shape='box', position=(0.0, 0.1, 0.0), size=(0.8, 0.1, 0.6), start=-1.0, **volume),
            ],
        },
        'hail_pond': {
            'name': 'Hailstorm', 'category': 'Weather', 'size': '18 mm hail, 1.6 m pond',
            'blurb': 'A summer thunderstorm drops hail the size of grapes on a garden pond (22 C). The stones fall at 15 to 20 m/s, '
                     'some partly melted on the way down from the freezing level high above; they crack into the water in crowns of '
                     'spray, bounce and skitter on the banks, and lie there in white drifts, melting slowly on the warm ground.',
            'render': {'end': 264},   # long enough to reach what its library picture shows
            'domain': {'kind': 'liquid', 'size_x': 1.6, 'size_y': 0.5, 'size_z': 1.2, 'resolution': 112, 'preroll': 0.3,
                       'open_sides': False, 'substeps_max': 10, 'cfl': 1.5},
            'liquid': {'thermal': True, 'liquid_temp': 18.0, 'air_temp': 22.0, 'air_humidity': 85.0, 'ground_temp': 18.0,
                       'wind_speed': 5.0, 'wind_dir': 40.0, 'settle': True, 'whitewater': True, 'ww_min_speed': 1.0},
            'water': {'backdrop': 6.0, 'clarity': 0.3, 'color': (0.3, 0.42, 0.3), 'ripple': 0.3, 'colliders_look': 'shaded',
                      'standin_color': (0.2, 0.22, 0.14)},
            'weather': {'precip': 'hail', 'rate': 50.0, 'size': 18.0, 'area': 4.8, 'turbulence': 1.5, 'gust': 0.6},
            'lighting': dict(summer),
            'camera': {'distance': 1.9, 'target_y': 0.1, 'pitch': 40, 'yaw': 15, 'anchor_x': 0.5, 'anchor_y': 0.6, 'focal_mm': 35},
            'colliders': [dict(c, temperature=18.0) for c in basin],
            'emitters': [
                dict(name='Pond', shape='box', position=(0.0, 0.1, 0.0), size=(0.8, 0.1, 0.6), start=-1.0, **volume),
            ],
        },
    }
