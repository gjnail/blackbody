"""Built-in presets for the sky (kind 'cloud', engine/cloud.py): clouds building from the warmed ground,
storms, their rain, snow and hail. The scene's metres are kilometres (Atmosphere › Scale 1000).

presets.py registers them (cloud_presets(K) returns the specs, K being its keyframe helper).
"""
from __future__ import annotations

CLOUD_ORDER = ['cumulus_day', 'thunderstorm']


def cloud_presets(K):
    afternoon = {'sun_on': True, 'sun_intensity': 3.2, 'sun_azimuth': 35.0, 'sun_elevation': 38.0, 'sun_color': (1.0, 0.96, 0.9)}
    return {
        'cumulus_day': {
            'name': 'Cumulus building', 'category': 'Sky and weather', 'size': '20 km of sky, time-lapse',
            'blurb': 'A summer afternoon (26 C at the ground, a light breeze), a minute to each second: thermals rise off the sun-warmed '
                     'fields, and where they pass their condensation level, about a kilometre up, cumulus bubble into being, flat-based '
                     'and sunlit, build as the latent heat of their condensing vapour lifts them, and flatten against the warm layer '
                     'of air above, drifting with the breeze and their shadows over the ground.',
            'render': {'end': 360},   # long enough to reach what its library picture shows
            'domain': {'kind': 'cloud', 'size_x': 20.0, 'size_y': 12.0, 'size_z': 20.0, 'resolution': 128, 'preroll': 20.0,
                       'substeps_max': 12, 'cfl': 1.0},
            # (a lid of warmer air at 2.4 km holds them to fair-weather cumulus; the biggest push into it)
            'atmosphere': {'scale': 1000.0, 'time_lapse': 60.0, 'surface_t': 26.0, 'surface_rh': 50.0, 'heat_flux': 220.0,
                           'evaporation': 100.0, 'wind': 3.0, 'shear': 1.0, 'lapse': 6.5, 'free_rh': 45.0,
                           'inversion_z': 2400.0, 'inversion_dt': 2.5},
            'sky': {},
            'lighting': dict(afternoon),
            # (from the ground, a few km inside the box, looking up across the field)
            'camera': {'distance': 13.0, 'target_y': 1.5, 'pitch': -6.2, 'yaw': 20.0, 'use_anchor': False, 'focal_mm': 28,
                       'far': 100000.0},
            'emitters': [],
            'colliders': [],
        },
        'thunderstorm': {
            'name': 'Thunderstorm', 'category': 'Sky and weather', 'size': '60 km of sky, time-lapse',
            'blurb': 'A single storm in a hot, humid, sheared afternoon air mass (about 3000 J/kg of CAPE), a minute to each second. '
                     'A warm bubble of air rises, condenses, and its latent heat drives an updraft of 40 to 60 m/s that towers into a '
                     'cumulonimbus, overshoots the tropopause at 12 km and spreads its anvil downwind; ice, snow and hail grow in it, '
                     'and grey shafts of rain and hail fall from its base, their cold outflow spreading over the ground.',
            'render': {'end': 720},   # long enough to reach what its library picture shows
            'domain': {'kind': 'cloud', 'size_x': 60.0, 'size_y': 16.0, 'size_z': 60.0, 'resolution': 192, 'preroll': 8.0,
                       'substeps_max': 12, 'cfl': 1.0},
            # (a thin warm layer at 1.1 km caps the air, as on a real storm day: the warm bubble breaks through it and
            # starts one storm, where without a cap any eddy would set off storms everywhere)
            'atmosphere': {'scale': 1000.0, 'time_lapse': 60.0, 'surface_t': 27.0, 'surface_rh': 62.0, 'lapse': 6.8,
                           'mixed_layer': 900.0, 'inversion_z': 1100.0, 'inversion_dt': 1.0,
                           'tropopause': 12000.0, 'heat_flux': 0.0, 'evaporation': 0.0, 'bubble': 2.5, 'bubble_r': 8000.0,
                           'wind': 3.0, 'shear': 2.5, 'veer': 40.0, 'follow': True, 'hail': 2.5},
            'sky': {'visibility': 120.0},
            'lighting': {'sun_on': True, 'sun_intensity': 3.0, 'sun_azimuth': -15.0, 'sun_elevation': 25.0,
                         'sun_color': (1.0, 0.94, 0.85)},
            # (from the ground 45 km away, the sun over the left shoulder; the box follows the storm)
            'camera': {'distance': 45.0, 'target_y': 7.5, 'pitch': -9.4, 'yaw': 30.0, 'use_anchor': False, 'focal_mm': 28,
                       'far': 200000.0},
            'emitters': [],
            'colliders': [],
        },
    }
