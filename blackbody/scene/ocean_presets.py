"""Built-in sea presets: open water from a glassy lake to a storm, each a complete starting point.

presets.py registers them (ocean_presets(K) returns the specs, K being its keyframe helper). The sea
(engine/ocean.py) runs on to the horizon; the box holds only the water that something disturbs, so
it can be small next to the waves as long as its water is deep enough for them (waves are held to
less than half the water level).
"""
from __future__ import annotations

OCEAN_ORDER = ['open_ocean', 'storm_sea', 'calm_lake', 'harbour_chop', 'beach_break', 'shore_break', 'reef_barrel', 'big_wave',
               'tsunami', 'tidal_bore', 'river_rocks']


SAND = (0.36, 0.31, 0.23)
REEF = (0.22, 0.21, 0.19)


def _bed(name, size, pos=(0.0, 0.0, 0.0)):
    """A built-in sea bed (tools/make_seabeds.py) as a collider: size (x, height, z) in metres."""
    return dict(name=name.capitalize(), shape='mesh', mesh=f'builtin:{name}.png', size=size, position=pos)


LAND = 8.0   # m: the tsunami's seafront (builtin coast.png), 2 m above its 6 m sea


def _building(name, x, z, hx, hy, hz):
    """A block of a town on the tsunami's seafront (half sizes in metres), standing on its land."""
    return dict(name=name, shape='box', size=(hx, hy, hz), position=(x, LAND + hy, z))


def _palm(name, x, z):
    return dict(name=name, shape='cylinder', size=(0.22, 4.0, 0.22), position=(x, LAND + 4.0, z))


def _car(name, x, z, yaw):
    """A car parked on the promenade: it floats off once the water is deep enough."""
    return dict(name=name, shape='box', size=(2.2, 0.75, 0.9), position=(x, LAND + 0.75, z), yaw=yaw,
                floating=True, density=400.0)


def ocean_presets(K):
    stream = dict(liquid_mode='stream', vel_blend=1.0, embers=False, noise=0.0)
    shallows = {'color': (0.35, 0.78, 0.72), 'clarity': 3.0, 'murk': 0.25, 'murk_color': (0.08, 0.2, 0.2),
                'colliders_look': 'shaded', 'reflect_footage': 0.0, 'ripple': 0.0, 'backdrop': 200.0, 'caustics': 1.0,
                'whitecaps': 1.0, 'crest_glow': 1.2, 'gusts': 0.3, 'bottomless': False}
    sunny = {'sun_on': True, 'sun_intensity': 3.0, 'sun_azimuth': 150.0, 'sun_elevation': 35.0, 'ambient': (0.55, 0.64, 0.78)}
    deep_blue = {'color': (0.05, 0.36, 0.46), 'clarity': 6.0, 'murk': 0.08, 'murk_color': (0.03, 0.14, 0.2),
                 'bottomless': True, 'colliders_look': 'shaded', 'reflect_footage': 0.0, 'ripple': 0.0,
                 'backdrop': 400.0, 'caustics': 0.0}
    return {
        'open_ocean': {
            'name': 'Open ocean', 'category': 'Sea', 'size': '12 m of sea, to the horizon',
            'blurb': 'Deep water in a fresh breeze: local wind waves on top of a long swell from a storm far away, '
                     'scattered whitecaps whose foam drifts off in streaks, sunlight glowing through the crests and a '
                     'glitter path under the sun. A buoy rides it.',
            'domain': {'kind': 'liquid', 'size_x': 12.0, 'size_y': 4.4, 'size_z': 12.0, 'resolution': 160, 'preroll': 1.0,
                       'substeps_max': 10, 'cfl': 1.5},
            'liquid': {'water_level': 3.0, 'narrow_band': False, 'level_absorb': 24.0, 'ocean_height': 1.1, 'ocean_length': 22.0, 'ocean_dir': 40.0,
                       'ocean_spread': 0.45, 'ocean_chop': 0.95, 'swell_height': 0.7, 'swell_length': 90.0,
                       'swell_dir': 80.0, 'ocean_depth': 200.0, 'wind_speed': 10.0, 'wind_dir': 40.0, 'ww_min_speed': 2.5},
            'water': dict(deep_blue, whitecaps=1.0, sea_foam=1.0, foam_streaks=0.7, crest_glow=1.0, gusts=0.3),
            'lighting': {'sun_on': True, 'sun_intensity': 3.0, 'sun_azimuth': 115.0, 'sun_elevation': 24.0,
                         'ambient': (0.5, 0.6, 0.75)},
            'camera': {'distance': 14.0, 'target_y': 3.2, 'pitch': 7, 'yaw': 0, 'anchor_x': 0.5, 'anchor_y': 0.62,
                       'focal_mm': 35},
            'colliders': [dict(name='Buoy', shape='cylinder', size=(0.35, 0.5, 0.35), position=(0.5, 3.1, 0.0),
                               floating=True, density=450.0)],
            'emitters': [],
        },
        'storm_sea': {
            'name': 'Storm at sea', 'category': 'Sea', 'size': '16 m of sea, 20 m/s wind',
            'blurb': 'A gale: steep, confused seas over a big swell, crests breaking all over into whitecaps, the foam '
                     'they leave drawn out into long streaks down the wind, gusts darkening patches of the water. '
                     'Overcast light. A crate is thrown about.',
            'domain': {'kind': 'liquid', 'size_x': 16.0, 'size_y': 9.0, 'size_z': 16.0, 'resolution': 160, 'preroll': 1.0,
                       'substeps_max': 10, 'cfl': 1.5},
            'liquid': {'water_level': 6.0, 'narrow_band': False, 'ocean_height': 2.4, 'ocean_length': 40.0, 'ocean_dir': 20.0,
                       'ocean_spread': 0.6, 'ocean_chop': 1.1, 'swell_height': 1.2, 'swell_length': 120.0,
                       'swell_dir': 50.0, 'ocean_depth': 300.0, 'wind_speed': 20.0, 'wind_dir': 20.0, 'ww_min_speed': 3.0},
            'water': {'color': (0.05, 0.3, 0.32), 'clarity': 4.0, 'murk': 0.15, 'murk_color': (0.04, 0.1, 0.11),
                      'bottomless': True, 'colliders_look': 'shaded', 'reflect_footage': 0.0, 'ripple': 0.0, 'backdrop': 400.0,
                      'caustics': 0.0, 'whitecaps': 1.0, 'sea_foam': 1.0, 'sea_foam_life': 14.0, 'foam_streaks': 1.0,
                      'crest_glow': 0.6, 'gusts': 0.6},
            'lighting': {'sun_on': True, 'sun_intensity': 0.35, 'sun_azimuth': 150.0, 'sun_elevation': 50.0,
                         'ambient': (0.45, 0.5, 0.55)},
            'camera': {'distance': 20.0, 'target_y': 6.4, 'pitch': 9, 'yaw': 0, 'anchor_x': 0.5, 'anchor_y': 0.6,
                       'focal_mm': 35},
            'colliders': [dict(name='Crate', shape='box', size=(0.6, 0.45, 0.8), position=(1.0, 6.2, -1.0), yaw=25.0,
                               floating=True, density=500.0)],
            'emitters': [],
        },
        'calm_lake': {
            'name': 'Calm lake', 'category': 'Sea', 'size': '5 m of lake',
            'blurb': 'A still lake late in the day: glassy water, a faint swell left by a boat long gone, and puffs of '
                     'breeze (cat’s paws) roughening drifting patches of ripples. A mooring post stands in it.',
            'domain': {'kind': 'liquid', 'size_x': 5.0, 'size_y': 1.4, 'size_z': 5.0, 'resolution': 176, 'preroll': 1.0,
                       'substeps_max': 10, 'cfl': 1.5},
            'liquid': {'water_level': 0.9, 'narrow_band': False, 'ocean_height': 0.03, 'ocean_length': 0.7, 'ocean_dir': 120.0,
                       'ocean_spread': 0.6, 'ocean_chop': 0.5, 'swell_height': 0.06, 'swell_length': 7.0, 'swell_dir': 200.0,
                       'ocean_depth': 12.0, 'wind_speed': 2.0, 'wind_dir': 120.0, 'ww_min_speed': 2.0},
            'water': {'color': (0.35, 0.55, 0.45), 'clarity': 2.5, 'murk': 0.3, 'murk_color': (0.08, 0.12, 0.09),
                      'bottomless': True, 'colliders_look': 'shaded', 'reflect_footage': 0.0, 'ripple': 0.0, 'backdrop': 200.0,
                      'caustics': 0.0, 'whitecaps': 0.0, 'crest_glow': 0.3, 'gusts': 0.9, 'roughness': 0.03},
            'lighting': {'sun_on': True, 'sun_intensity': 2.5, 'sun_color': (1.0, 0.72, 0.45), 'sun_azimuth': 175.0,
                         'sun_elevation': 9.0, 'ambient': (0.55, 0.55, 0.62)},
            'camera': {'distance': 5.5, 'target_y': 0.95, 'pitch': 6, 'yaw': 0, 'anchor_x': 0.5, 'anchor_y': 0.6,
                       'focal_mm': 40},
            'colliders': [dict(name='Post', shape='cylinder', size=(0.09, 0.9, 0.09), position=(-0.8, 0.9, 0.6))],
            'emitters': [],
        },
        'harbour_chop': {
            'name': 'Harbour chop', 'category': 'Sea', 'size': '5 m, pier posts',
            'blurb': 'Short, choppy waves running in from every side in a breezy harbour, slapping against the posts of a '
                     'pier and throwing up spray; a few whitecaps on the steepest crests.',
            'domain': {'kind': 'liquid', 'size_x': 5.0, 'size_y': 2.2, 'size_z': 4.0, 'resolution': 176, 'preroll': 1.5,
                       'substeps_max': 12, 'cfl': 1.5},
            'liquid': {'water_level': 1.2, 'narrow_band': False, 'ocean_height': 0.4, 'ocean_length': 3.2, 'ocean_dir': 70.0,
                       'ocean_spread': 0.75, 'ocean_chop': 1.0, 'ocean_depth': 5.0, 'wind_speed': 9.0, 'wind_dir': 70.0,
                       'ww_min_speed': 1.6},
            'water': {'color': (0.25, 0.5, 0.45), 'clarity': 1.8, 'murk': 0.5, 'murk_color': (0.05, 0.13, 0.12),
                      'bottomless': True, 'colliders_look': 'shaded', 'reflect_footage': 0.0, 'ripple': 0.0, 'backdrop': 150.0,
                      'caustics': 0.0, 'whitecaps': 1.0, 'crest_glow': 0.8, 'gusts': 0.4},
            'lighting': {'sun_on': True, 'sun_intensity': 3.0, 'sun_azimuth': -60.0, 'sun_elevation': 30.0,
                         'ambient': (0.55, 0.62, 0.72)},
            'camera': {'distance': 5.0, 'target_y': 1.2, 'pitch': 14, 'yaw': 20, 'anchor_x': 0.5, 'anchor_y': 0.6,
                       'focal_mm': 35},
            'colliders': [dict(name='Post', shape='cylinder', size=(0.12, 1.2, 0.12), position=(-0.6, 1.2, 0.0)),
                          dict(name='Post 2', shape='cylinder', size=(0.12, 1.2, 0.12), position=(0.9, 1.2, 0.0))],
            'emitters': [],
        },
        'beach_break': {
            'name': 'Beach break', 'category': 'Sea', 'size': '1 m surf on a sandy beach',
            'blurb': 'A swell rolling onto a sandy beach: the waves slow and steepen as they feel the bottom, spill over '
                     'into white water and run up the sand, and the foam drifts back. Waves come in from the sea side only '
                     '(the other sides are walls along them), as in a wave tank; the open water past the box shoals too.',
            'domain': {'kind': 'liquid', 'size_x': 24.0, 'size_y': 5.0, 'size_z': 12.0, 'resolution': 192, 'preroll': 3.0,
                       'substeps_max': 12, 'cfl': 1.5, 'mesh_resolution': 192},
            'liquid': {'water_level': 2.5, 'narrow_band': False, 'swell_height': 1.0, 'swell_length': 40.0, 'swell_dir': 90.0,
                       'ocean_height': 0.12, 'ocean_length': 3.0, 'ocean_dir': 80.0, 'ocean_spread': 0.5, 'ocean_chop': 0.8,
                       'sea_from': 'upwave', 'wind_speed': 4.0, 'wind_dir': 80.0, 'ww_min_speed': 2.2, 'ww_amount': 3.0,
                       'ww_turbulence': 3.0, 'ww_crests': 0.2, 'foam_life': 3.0},
            'water': dict(shallows, standin_color=SAND, spray=0.8, bubbles=2.0, foam=1.3, droplets=0.3, foam_scale=0.08),
            'lighting': dict(sunny),
            'camera': {'distance': 18.0, 'target_y': 2.2, 'pitch': 12, 'yaw': 65, 'anchor_x': 0.5, 'anchor_y': 0.6, 'focal_mm': 35},
            'colliders': [_bed('beach', (60.0, 7.0, 100.0), (5.0, 0.0, 0.0))],
            'emitters': [],
        },
        'shore_break': {
            'name': 'Shore break', 'category': 'Sea', 'size': '1 m wave, close up',
            'blurb': 'A wave dumping onto a steep beach, seen close: it rears up in the shallows, pitches its lip forward over '
                     'a hollow tube and crashes onto the sand, then rushes up the beach as white water. A small box with fine '
                     'cells: the lip and the tube stay crisp.',
            'domain': {'kind': 'liquid', 'size_x': 16.0, 'size_y': 4.0, 'size_z': 8.0, 'resolution': 256, 'preroll': 2.0,
                       'substeps_max': 16, 'cfl': 1.5, 'mesh_resolution': 192},
            'liquid': {'water_level': 2.2, 'narrow_band': False, 'swell_height': 0.9, 'swell_length': 28.0, 'swell_dir': 90.0,
                       'ocean_height': 0.08, 'ocean_length': 2.5, 'ocean_dir': 80.0, 'ocean_spread': 0.5, 'ocean_chop': 0.7,
                       'sea_from': 'upwave', 'wind_speed': 3.0, 'wind_dir': 270.0, 'ww_min_speed': 2.5, 'ww_amount': 3.0,
                       'ww_turbulence': 3.0, 'ww_crests': 0.2, 'foam_life': 2.0},
            'water': dict(shallows, standin_color=SAND, spray=0.8, bubbles=2.0, foam=1.3, droplets=0.3, foam_scale=0.05),
            'lighting': dict(sunny, sun_azimuth=-30.0, sun_elevation=35.0),
            'camera': {'mode': 'free', 'position': (7.5, 3.8, 6.0), 'rotation': (-6.22, 50.53, 0.0), 'focal_mm': 30,
                       'use_anchor': False},
            'colliders': [_bed('beach', (24.0, 5.0, 60.0), (3.0, 0.0, 0.0))],
            'emitters': [],
        },
        'reef_barrel': {
            'name': 'Reef barrel', 'category': 'Sea', 'size': '2 m tube over a slab reef',
            'blurb': 'A long-period groundswell meeting a slab reef: deep water right up to a steep reef face, so each '
                     'wave jacks up in a few metres, throws a thick lip out over a hollow tube and peels along the reef, '
                     'exploding into white water where the lip lands. Seen with a long lens from the channel beside the '
                     'break. Fine cells (7.5 cm) and every drop of the water simulated (no narrow band): a few seconds a '
                     'frame. The sides along the waves are mirrors, so the break carries on past the box.',
            'domain': {'kind': 'liquid', 'size_x': 24.0, 'size_y': 8.0, 'size_z': 16.0, 'resolution': 320, 'preroll': 3.0,
                       'substeps_max': 16, 'cfl': 1.5, 'mesh_resolution': 192},
            'liquid': {'water_level': 4.5, 'narrow_band': False, 'max_particles': 40.0,
                       'swell_height': 1.8, 'swell_length': 70.0, 'swell_dir': 90.0,
                       'ocean_height': 0.03, 'ocean_length': 2.0, 'ocean_dir': 90.0, 'ocean_spread': 0.3, 'ocean_chop': 0.5,
                       'sea_from': 'upwave', 'wind_speed': 4.0, 'wind_dir': 270.0,
                       'ww_amount': 14.0, 'ww_turbulence': 2.0, 'ww_crests': 0.0, 'ww_min_speed': 5.0, 'ww_max': 8.0,
                       'foam_life': 0.8},
            'water': dict(shallows, color=(0.1, 0.5, 0.5), clarity=4.0, murk=0.1, murk_color=(0.08, 0.22, 0.2),
                          standin_color=REEF, ripple=0.3, ripple_freq=6.0, bubbles=2.5, foam=1.5, spray=1.0, droplets=0.2,
                          foam_scale=0.15, smoothing=2, calm=24, rainbow=0.3, roughness=0.1),
            'lighting': dict(sunny, sun_azimuth=30.0, sun_elevation=35.0),
            'camera': {'mode': 'free', 'position': (8.0, 6.5, 12.0), 'rotation': (-3.28, 30.65, 0.0), 'focal_mm': 50,
                       'use_anchor': False},
            'colliders': [_bed('slab', (25.0, 3.637, 17.0))],
            'emitters': [],
        },
        'big_wave': {
            'name': 'Big wave', 'category': 'Sea', 'size': '7 m tube over an outer reef',
            'blurb': 'A giant winter swell meeting a deep outer slab: a mountain of water jacks up out of deep water, '
                     'throws a lip as thick as a house over a tube you could drive a truck through, and explodes into a '
                     'wall of white water. The Reef barrel scaled up four times, as a wave tank scales: every length times '
                     'four, every time twice as slow (Froude scaling), so it breaks the same way, at 30 cm cells.',
            'domain': {'kind': 'liquid', 'size_x': 96.0, 'size_y': 32.0, 'size_z': 64.0, 'resolution': 320, 'preroll': 6.0,
                       'substeps_max': 16, 'cfl': 1.5, 'mesh_resolution': 192},
            'liquid': {'water_level': 18.0, 'narrow_band': False, 'max_particles': 40.0,
                       'swell_height': 7.2, 'swell_length': 280.0, 'swell_dir': 90.0,
                       'ocean_height': 0.12, 'ocean_length': 8.0, 'ocean_dir': 90.0, 'ocean_spread': 0.3, 'ocean_chop': 0.5,
                       'sea_from': 'upwave', 'wind_speed': 8.0, 'wind_dir': 270.0,
                       'ww_amount': 14.0, 'ww_turbulence': 2.0, 'ww_crests': 0.0, 'ww_min_speed': 10.0, 'ww_max': 8.0,
                       'foam_life': 1.6},
            'water': dict(shallows, color=(0.07, 0.42, 0.48), clarity=8.0, murk=0.05, murk_color=(0.06, 0.2, 0.22),
                          standin_color=REEF, ripple=0.3, ripple_freq=1.5, bubbles=2.5, foam=1.5, spray=1.0, droplets=0.2,
                          droplet_size=0.008, foam_scale=0.6, smoothing=2, calm=24, rainbow=0.3, roughness=0.1),
            'lighting': dict(sunny, sun_azimuth=30.0, sun_elevation=35.0),
            'camera': {'mode': 'free', 'position': (32.0, 26.0, 48.0), 'rotation': (-3.28, 30.65, 0.0), 'focal_mm': 50,
                       'use_anchor': False},
            'colliders': [_bed('slab', (100.0, 14.548, 68.0))],
            'emitters': [],
        },
        'tsunami': {
            'name': 'Tsunami', 'category': 'Sea', 'size': '4 m tsunami onto a seafront',
            'blurb': 'A tsunami reaching a town’s seafront, seen from a hotel balcony: first the sea draws back, '
                     'baring the sea floor, then a wall of water rises out of the bay, breaks into a churning white bore, '
                     'runs up the beach and pours over the sea wall into the streets, lifting the boats and the cars on '
                     'the promenade. 13 seconds; every drop of the water simulated (no narrow band), 25 cm cells. The '
                     'surge is one long wave (Liquid, Surge shape Tsunami); the sides along it are mirrors, so the flood '
                     'runs on along the shore past the box.',
            'domain': {'kind': 'liquid', 'size_x': 96.0, 'size_y': 16.0, 'size_z': 64.0, 'resolution': 384, 'preroll': 1.0,
                       'substeps_max': 16, 'cfl': 1.5, 'mesh_resolution': 192},
            'liquid': {'water_level': 6.0, 'narrow_band': False, 'max_particles': 40.0,
                       'surge_height': 4.0, 'surge_length': 10.0, 'surge_dir': 90.0, 'surge_time': 11.0, 'surge_kind': 'tsunami',
                       'swell_height': 0.5, 'swell_length': 40.0, 'swell_dir': 95.0,
                       'ocean_height': 0.25, 'ocean_length': 6.0, 'ocean_dir': 80.0, 'ocean_spread': 0.4, 'ocean_chop': 0.7,
                       'sea_from': 'upwave', 'wind_speed': 6.0, 'wind_dir': 80.0,
                       'ww_min_speed': 3.0, 'ww_amount': 4.0, 'ww_turbulence': 3.0, 'ww_crests': 0.3, 'ww_max': 8.0,
                       'foam_life': 6.0},
            'water': dict(shallows, color=(0.25, 0.3, 0.22), clarity=0.6, murk=2.5, murk_color=(0.28, 0.25, 0.16),
                          standin_color=(0.46, 0.43, 0.37), foam_color=(0.82, 0.8, 0.72), bubbles=2.0, foam=1.5, spray=1.0,
                          droplets=0.3, foam_scale=0.2, caustics=0.0, backdrop=400.0),
            'lighting': dict(sunny, sun_azimuth=120.0, sun_elevation=40.0, sun_intensity=2.4, ambient=(0.55, 0.58, 0.64)),
            'camera': {'mode': 'free', 'position': (22.0, 16.0, 30.0), 'rotation': (-11.11, 46.59, 0.0), 'focal_mm': 28,
                       'use_anchor': False},
            'render': {'end': 312},
            'colliders': [_bed('coast', (98.0, 7.843, 66.0)), _bed('coast_wide', (300.0, 7.843, 300.0), (40.0, 0.0, 0.0)),
                          _building('Hotel', 24.0, -16.0, 5.0, 7.0, 6.0), _building('House', 22.0, 6.0, 4.0, 3.5, 4.5),
                          _building('Block', 38.0, 20.0, 6.0, 10.0, 7.0),
                          _palm('Palm', 14.0, -26.0), _palm('Palm 2', 14.5, -6.0), _palm('Palm 3', 14.0, 14.0),
                          _palm('Palm 4', 14.5, 27.0),
                          _car('Car', 16.0, -12.0, 5.0), _car('Car 2', 16.5, 0.0, -8.0), _car('Car 3', 15.5, 10.0, 2.0),
                          _car('Car 4', 30.0, 2.0, 90.0),
                          dict(name='Boat', shape='box', size=(3.0, 0.8, 1.2), position=(-12.0, 6.2, -6.0), yaw=20.0,
                               floating=True, density=300.0)],
            'emitters': [],
        },
        'tidal_bore': {
            'name': 'Tidal bore', 'category': 'Sea', 'size': '60 cm bore up a river',
            'blurb': 'The incoming tide running up a river as a single breaking front: the water behind it stands higher '
                     'and flows upstream against the river\u2019s current, churning white along its face and rolling up the '
                     'banks. A bore surge (Liquid, Surge shape).',
            'domain': {'kind': 'liquid', 'size_x': 24.0, 'size_y': 3.0, 'size_z': 8.0, 'resolution': 200, 'preroll': 1.0,
                       'substeps_max': 14, 'cfl': 1.5},
            'liquid': {'water_level': 1.0, 'surge_height': 0.6, 'surge_length': 1.5, 'surge_dir': 90.0,
                       'surge_time': 2.5, 'surge_kind': 'bore', 'current_speed': 0.6, 'current_dir': -90.0,
                       'sea_from': 'upwave', 'ocean_depth': 1.0, 'ww_min_speed': 2.2},
            'water': dict(shallows, color=(0.55, 0.55, 0.4), clarity=0.6, murk=2.0, murk_color=(0.35, 0.3, 0.2), bottomless=True,
                          caustics=0.0),
            'lighting': dict(sunny, sun_azimuth=100.0, sun_elevation=30.0),
            'camera': {'distance': 14.0, 'target_y': 1.0, 'pitch': 14, 'yaw': 35, 'anchor_x': 0.5, 'anchor_y': 0.6, 'focal_mm': 35},
            'colliders': [dict(name='Bank', shape='box', size=(12.0, 1.4, 0.8), position=(0.0, 0.7, -3.8)),
                          dict(name='Bank 2', shape='box', size=(12.0, 1.4, 0.8), position=(0.0, 0.7, 3.8))],
            'emitters': [],
        },
        'river_rocks': {
            'name': 'River past rocks', 'category': 'Sea', 'size': '1.8 m/s current',
            'blurb': 'A fast river running past boulders: the water piles up against them, pours round, and trails eddies '
                     'and lines of foam far downstream, past the edge of the box, where the open water carries the current '
                     'on round another rock.',
            'domain': {'kind': 'liquid', 'size_x': 8.0, 'size_y': 1.6, 'size_z': 5.0, 'resolution': 192, 'preroll': 2.0,
                       'substeps_max': 12, 'cfl': 1.5},
            'liquid': {'water_level': 0.8, 'current_speed': 1.8, 'current_dir': 90.0, 'ww_min_speed': 2.6,
                       'ocean_height': 0.03, 'ocean_length': 0.8, 'ocean_dir': 90.0, 'wind_speed': 2.0, 'wind_dir': 90.0,
                       'open_water_area': 5.0},
            'water': dict(shallows, color=(0.5, 0.66, 0.55), clarity=1.2, murk=0.8, murk_color=(0.16, 0.2, 0.14), bottomless=True,
                          caustics=0.0, sea_foam_life=25.0),
            'lighting': dict(sunny, sun_azimuth=60.0, sun_elevation=40.0),
            'camera': {'distance': 12.0, 'target_y': 0.8, 'pitch': 28, 'yaw': 30, 'anchor_x': 0.5, 'anchor_y': 0.5, 'focal_mm': 35},
            'colliders': [dict(name='Boulder', shape='sphere', size=(0.55, 0.55, 0.55), position=(-1.8, 0.6, 0.4)),
                          dict(name='Boulder 2', shape='sphere', size=(0.4, 0.4, 0.4), position=(0.8, 0.5, -1.2)),
                          dict(name='Rock downstream', shape='sphere', size=(0.8, 0.8, 0.8), position=(9.0, 0.5, 1.5))],
            'emitters': [],
        },
    }
