"""Built-in liquid presets: real water effects at real sizes, each a complete starting point.

presets.py registers them (liquid_presets(K) returns the specs, K being its keyframe helper).
Liquids need finer grids than fire, so the boxes are kept tight around the action.
"""
from __future__ import annotations

LIQUID_ORDER = ['water_pour', 'rock_splash', 'bucket_throw', 'fountain', 'hose', 'wave', 'waterfall', 'spill',
                'floating', 'pond_below', 'rain_pond', 'towel_dip', 'boat_wake', 'honey', 'lava', 'ink_tank', 'oil_water', 'river_post',
                'hose_on_fire', 'sea_swell']


def _fall(t0, y0, y_surface, y_rest, g=9.81):
    """Keyframes (seconds, position) for an object dropped at t0 from height y0: free fall to the
    water surface, then slowing to rest at y_rest."""
    keys = [(0.0, (0.0, y0, 0.0))]
    t_hit = (2.0 * (y0 - y_surface) / g) ** 0.5
    n = 8
    for i in range(n + 1):
        t = t_hit * i / n
        keys.append((t0 + t, (0.0, y0 - 0.5 * g * t * t, 0.0)))
    v = g * t_hit
    depth = y_surface - y_rest
    t_stop = 2.0 * depth / v * 2.5      # water brakes it hard: about 2.5x a constant-deceleration stop
    for f, s in ((0.15, 0.45), (0.4, 0.8), (1.0, 1.0)):
        keys.append((t0 + t_hit + t_stop * f, (0.0, y_surface - depth * s, 0.0)))
    return keys


def liquid_presets(K):
    # clear water outdoors: a key light for glints, a sky-coloured ambient
    sun = {'sun_on': True, 'sun_intensity': 3.0, 'sun_azimuth': -40.0, 'sun_elevation': 35.0, 'ambient': (0.55, 0.62, 0.72)}
    stream = dict(liquid_mode='stream', vel_blend=1.0, embers=False, noise=0.0)
    volume = dict(liquid_mode='fill', vel_blend=0.0, embers=False, noise=0.0)
    out = {
        'water_pour': {
            'name': 'Pouring water', 'category': 'Liquids', 'size': '80 cm pour',
            'blurb': 'A steady stream of water poured from knee height onto the ground, splashing and spreading into a puddle.',
            'domain': {'kind': 'liquid', 'size_x': 1.4, 'size_y': 1.0, 'size_z': 1.2, 'resolution': 176, 'preroll': 0.0,
                       'substeps_max': 12, 'cfl': 1.5},
            'liquid': {'wall_drag': 3.0},
            'water': {'backdrop': 6.0, 'ripple_freq': 40.0},
            'lighting': dict(sun),
            'camera': {'distance': 2.2, 'target_y': 0.42, 'pitch': 14, 'yaw': 15, 'anchor_x': 0.5, 'anchor_y': 0.86, 'focal_mm': 35},
            'emitters': [dict(name='Spout', shape='cylinder', position=(-0.3, 0.8, 0.0), size=(0.03, 0.02, 0.03),
                              velocity=(0.9, -0.3, 0.0), **stream)],
        },
        'rock_splash': {
            'name': 'Rock into a pond', 'category': 'Liquids', 'size': '15 cm rock',
            'blurb': 'A rock dropped into still water: the crown splash, the cavity closing into a rebound jet, rings spreading out. '
                     'The rock itself is a collider: put it in your footage or render it separately.',
            'domain': {'kind': 'liquid', 'size_x': 2.0, 'size_y': 0.9, 'size_z': 2.0, 'resolution': 208, 'preroll': 1.0,
                       'substeps_max': 14, 'cfl': 1.5},
            'liquid': {'ww_min_speed': 0.8, 'settle': True, 'water_level': 0.24},
            'water': {'backdrop': 5.0, 'clarity': 1.5, 'color': (0.55, 0.78, 0.74)},
            'lighting': dict(sun),
            'camera': {'distance': 1.7, 'target_y': 0.22, 'pitch': 20, 'anchor_x': 0.5, 'anchor_y': 0.62, 'focal_mm': 45},
            'colliders': [dict(name='Rock', shape='sphere', size=(0.075, 0.075, 0.075),
                               position=K(*_fall(0.2, 0.95, 0.24, 0.08), interp='linear'))],
            'emitters': [],
        },
        'bucket_throw': {
            'name': 'Thrown bucket', 'category': 'Liquids', 'size': '8 litres',
            'blurb': 'A bucket of water thrown forward: the mass spreads into sheets and drops in flight, then slaps onto the ground.',
            'domain': {'kind': 'liquid', 'size_x': 2.6, 'size_y': 1.1, 'size_z': 1.6, 'resolution': 256, 'preroll': 0.0,
                       'substeps_max': 14, 'cfl': 1.5},
            'water': {'backdrop': 8.0},
            'lighting': dict(sun),
            'camera': {'distance': 2.4, 'target_y': 0.45, 'pitch': 10, 'yaw': 20, 'anchor_x': 0.5, 'anchor_y': 0.8, 'focal_mm': 35},
            'emitters': [dict(name='Water', shape='sphere', position=(-0.95, 0.75, 0.0), size=(0.2, 0.1, 0.1),
                              velocity=(2.4, 0.9, 0.0), radial=0.7, start=0.05, **volume)],
        },
        'fountain': {
            'name': 'Fountain jet', 'category': 'Liquids', 'size': '1.2 m jet',
            'blurb': 'A vertical fountain jet that rises, breaks up at the top and rains back down onto the ground.',
            'domain': {'kind': 'liquid', 'size_x': 1.3, 'size_y': 1.5, 'size_z': 1.3, 'resolution': 192, 'preroll': 1.0,
                       'substeps_max': 12, 'cfl': 1.5},
            'liquid': {'wall_drag': 1.5},
            'water': {'backdrop': 10.0},
            'lighting': dict(sun),
            'camera': {'distance': 3.4, 'target_y': 0.7, 'pitch': 8, 'anchor_x': 0.5, 'anchor_y': 0.88, 'focal_mm': 35},
            'emitters': [dict(name='Nozzle', shape='cylinder', position=(0.0, 0.06, 0.0), size=(0.022, 0.03, 0.022),
                              velocity=(0.0, 4.8, 0.0), jitter=0.04, **stream)],
        },
        'hose': {
            'name': 'Hose on a wall', 'category': 'Liquids', 'size': '1.5 m jet',
            'blurb': 'A garden hose sprayed at a wall: the jet hits, fans out across the wall and runs down to the ground. '
                     'The wall is a collider: line it up with the wall in your footage.',
            'domain': {'kind': 'liquid', 'size_x': 1.8, 'size_y': 1.2, 'size_z': 1.2, 'resolution': 256, 'preroll': 0.5,
                       'substeps_max': 16, 'cfl': 1.5},
            'liquid': {'wall_drag': 1.0},
            'water': {'backdrop': 2.0},
            'lighting': dict(sun),
            'camera': {'distance': 2.8, 'target_y': 0.5, 'pitch': 8, 'yaw': 30, 'anchor_x': 0.5, 'anchor_y': 0.85, 'focal_mm': 35},
            'colliders': [dict(name='Wall', shape='box', position=(0.85, 0.6, 0.0), size=(0.08, 0.6, 0.6))],
            'emitters': [dict(name='Hose', shape='cylinder', position=(-0.8, 0.55, 0.0), size=(0.018, 0.018, 0.018),
                              velocity=(7.0, 1.0, 0.0), jitter=0.03, **stream)],
        },
        'wave': {
            'name': 'Breaking wave', 'category': 'Liquids', 'size': '4 m channel',
            'blurb': 'A wall of water released down a channel, crashing over a low obstacle. The channel sides are glass.',
            'domain': {'kind': 'liquid', 'size_x': 4.0, 'size_y': 1.5, 'size_z': 1.2, 'resolution': 224, 'preroll': 0.0,
                       'open_sides': False, 'substeps_max': 12, 'cfl': 1.5},
            'liquid': {'surface_tension': 0.0},
            'water': {'backdrop': 8.0, 'ripple': 0.35},
            'lighting': dict(sun),
            'camera': {'distance': 5.5, 'target_y': 0.4, 'pitch': 16, 'yaw': 20, 'anchor_x': 0.5, 'anchor_y': 0.8, 'focal_mm': 35},
            'colliders': [dict(name='Step', shape='box', position=(0.9, 0.12, 0.0), size=(0.12, 0.12, 0.6))],
            'emitters': [
                dict(name='Water wall', shape='box', position=(-1.62, 0.45, 0.0), size=(0.38, 0.45, 0.6), **volume),
                dict(name='Shallows', shape='box', position=(0.4, 0.03, 0.0), size=(1.6, 0.03, 0.6), **volume),
            ],
        },
        'waterfall': {
            'name': 'Waterfall', 'category': 'Liquids', 'size': '1.5 m drop',
            'blurb': 'A sheet of water flowing off a ledge and falling onto the ground below.',
            'domain': {'kind': 'liquid', 'size_x': 2.2, 'size_y': 1.9, 'size_z': 1.4, 'resolution': 192, 'preroll': 1.0,
                       'substeps_max': 12, 'cfl': 1.5},
            'liquid': {'wall_drag': 0.8},
            'water': {'backdrop': 6.0},
            'lighting': dict(sun),
            'camera': {'distance': 4.4, 'target_y': 0.85, 'pitch': 6, 'yaw': 30, 'anchor_x': 0.5, 'anchor_y': 0.88, 'focal_mm': 35},
            'colliders': [dict(name='Ledge', shape='box', position=(-0.7, 0.75, 0.0), size=(0.4, 0.75, 0.55))],
            'emitters': [dict(name='Flow', shape='box', position=(-0.9, 1.56, 0.0), size=(0.18, 0.04, 0.38),
                              velocity=(1.2, 0.0, 0.0), **stream)],
        },
        'spill': {
            'name': 'Spilled glass', 'category': 'Liquids', 'size': '25 cl',
            'blurb': 'A glass of water knocked over: it rushes out across the ground and settles into a puddle.',
            'domain': {'kind': 'liquid', 'size_x': 0.9, 'size_y': 0.25, 'size_z': 0.7, 'resolution': 176, 'preroll': 0.0,
                       'substeps_max': 16, 'cfl': 1.5},
            'liquid': {'wall_drag': 3.0},
            'water': {'backdrop': 1.5, 'ripple_freq': 80.0, 'ripple': 0.15},
            'lighting': dict(sun),
            'camera': {'distance': 1.3, 'target_y': 0.02, 'pitch': 28, 'anchor_x': 0.5, 'anchor_y': 0.7, 'focal_mm': 35},
            'emitters': [dict(name='Glass', shape='cylinder', position=(-0.3, 0.06, 0.0), size=(0.035, 0.06, 0.035),
                              velocity=(0.8, -0.1, 0.0), start=0.05, **volume)],
        },
        'floating': {
            'name': 'Things in a pond', 'category': 'Liquids', 'size': '2.4 m pond',
            'blurb': 'A crate, a ball and a stone dropped into open water: the crate plunges and bobs back up, the ball '
                     'barely dips, the stone sinks. The liquid moves them (Floats), and rings spread out past the frame.',
            'domain': {'kind': 'liquid', 'size_x': 2.4, 'size_y': 1.0, 'size_z': 2.4, 'resolution': 192, 'preroll': 1.0,
                       'substeps_max': 14, 'cfl': 1.5},
            'liquid': {'water_level': 0.35, 'settle': True, 'narrow_band': True, 'ww_min_speed': 0.8},
            'water': {'backdrop': 5.0, 'clarity': 2.0, 'color': (0.6, 0.8, 0.76), 'colliders_look': 'shaded'},
            'lighting': dict(sun),
            'camera': {'distance': 2.6, 'target_y': 0.3, 'pitch': 22, 'yaw': 10, 'anchor_x': 0.5, 'anchor_y': 0.6, 'focal_mm': 35},
            'colliders': [
                dict(name='Crate', shape='box', size=(0.12, 0.12, 0.12), position=(-0.45, 0.85, 0.1), yaw=20.0,
                     floating=True, density=550.0),
                dict(name='Ball', shape='sphere', size=(0.11, 0.11, 0.11), position=(0.2, 0.75, -0.25),
                     floating=True, density=150.0),
                dict(name='Stone', shape='sphere', size=(0.07, 0.07, 0.07), position=(0.45, 0.9, 0.35),
                     floating=True, density=2600.0),
            ],
            'emitters': [],
        },
        'boat_wake': {
            'name': 'Boat wake', 'category': 'Liquids', 'size': '1.2 m hull, 2.5 m/s',
            'blurb': 'A small hull running across a lake in a light breeze: bow wave, spray blown off by the wind, and a V '
                     'of wake that fans out behind it across the open water, long after the hull has gone by, with a trail '
                     'of foam behind it. The open water round the box carries the wake on past its edges.',
            'domain': {'kind': 'liquid', 'size_x': 12.0, 'size_y': 1.4, 'size_z': 5.0, 'resolution': 240, 'preroll': 1.0,
                       'substeps_max': 14, 'cfl': 1.5},
            'liquid': {'water_level': 0.8, 'narrow_band': True, 'wind_speed': 5.0, 'wind_dir': 60.0, 'ww_min_speed': 1.0,
                       'ocean_height': 0.08, 'ocean_length': 2.0, 'ocean_dir': 60.0, 'ocean_spread': 0.5, 'ocean_chop': 0.7,
                       'ocean_depth': 20.0, 'open_water_area': 6.0},
            'water': {'backdrop': 60.0, 'clarity': 3.0, 'color': (0.35, 0.62, 0.66), 'murk': 0.3, 'murk_color': (0.06, 0.14, 0.15),
                      'bottomless': True, 'colliders_look': 'shaded', 'ripple': 0.0, 'reflect_footage': 0.0, 'caustics': 0.0,
                      'whitecaps': 0.0},
            'lighting': {'sun_on': True, 'sun_intensity': 3.0, 'sun_azimuth': 110.0, 'sun_elevation': 30.0, 'ambient': (0.55, 0.62, 0.75)},
            'camera': {'distance': 11.0, 'target_y': 0.8, 'pitch': 24, 'yaw': 20, 'anchor_x': 0.5, 'anchor_y': 0.55, 'focal_mm': 35},
            'colliders': [dict(name='Hull', shape='box', size=(0.6, 0.12, 0.2),
                               position=K((0.0, (-5.2, 0.82, 0.0)), (4.2, (5.3, 0.82, 0.0)), interp='linear'))],
            'emitters': [],
        },
        'honey': {
            'name': 'Honey', 'category': 'Liquids', 'size': '45 cm pour',
            'blurb': 'A thin stream of honey poured onto a surface: it lands slowly and builds a thick, rounded pool that '
                     'creeps outward. Viscosity makes it thick; lower it for syrup or oil.',
            'domain': {'kind': 'liquid', 'size_x': 0.5, 'size_y': 0.5, 'size_z': 0.5, 'resolution': 160, 'preroll': 0.0,
                       'substeps_max': 16, 'cfl': 1.5},
            'liquid': {'viscosity': 10.0, 'liquid_density': 1420.0, 'surface_tension': 0.05, 'flip': 0.4,
                       'whitewater': False, 'wall_drag': 0.5},
            'water': {'ior': 1.49, 'color': (0.88, 0.52, 0.1), 'clarity': 0.03, 'backdrop': 1.5, 'ripple': 0.0,
                      'roughness': 0.03, 'caustics': 0.3},
            'lighting': dict(sun),
            'camera': {'distance': 0.9, 'target_y': 0.12, 'pitch': 14, 'yaw': 10, 'anchor_x': 0.5, 'anchor_y': 0.72, 'focal_mm': 50},
            'emitters': [dict(name='Pour', shape='cylinder', position=(0.0, 0.46, 0.0), size=(0.01, 0.01, 0.01),
                              velocity=(0.0, -0.35, 0.0), **stream)],
        },
        'lava': {
            'name': 'Lava flow', 'category': 'Liquids', 'size': '4 m flow',
            'blurb': 'Pahoehoe lava welling up at a vent at dusk and spreading over the ground in a lobe that buds '
                     'toes at its edge. Fresh melt glows orange through a thin glassy skin, in stretch marks drawn '
                     'out along the flow; within seconds the skin greys over and darkens, tearing where the flow '
                     'pulls it apart and roping up where it is squeezed. The foot of the advancing lobe glows where '
                     'it splits along the ground, the air over the lava shimmers and its glow lights the ground.',
            'domain': {'kind': 'liquid', 'size_x': 5.8, 'size_y': 0.8, 'size_z': 3.0, 'resolution': 208, 'preroll': 3.0,
                       'substeps_max': 12, 'cfl': 1.5},
            'liquid': {'viscosity': 80.0, 'liquid_density': 2600.0, 'surface_tension': 0.4, 'flip': 0.2,
                       'whitewater': False, 'wall_drag': 1.0, 'cooling': 0.4, 'solidify': 3000.0},
            'water': {'ior': 1.6, 'color': (0.2, 0.2, 0.2), 'clarity': 0.01, 'murk': 400.0, 'murk_color': (0.05, 0.045, 0.045),
                      'backdrop': 4.0, 'ripple': 0.0, 'roughness': 0.35, 'reflection': 0.4, 'caustics': 0.0,
                      'glow': 2.2, 'glow_temp': 1400.0, 'crust': 0.75, 'crust_scale': 0.12, 'crust_time': 0.8,
                      'ropes': 1.0, 'lava_light': 1.0, 'crust_roughness': 0.7, 'crust_color': (0.03, 0.029, 0.028)},
            'lighting': {'sun_on': True, 'sun_intensity': 0.35, 'sun_azimuth': 200.0, 'sun_elevation': 6.0,
                         'ambient': (0.1, 0.12, 0.2)},
            # photographed: the hottest melt near clipping, blooming, with a film's red halation round it
            'composite': {'bloom': 0.3, 'halation': 0.6},
            'camera': {'distance': 3.4, 'target_y': 0.1, 'pitch': 22, 'yaw': 62, 'anchor_x': 0.5, 'anchor_y': 0.62, 'focal_mm': 35},
            # the vent, and two breakouts: melt bursting through the crust at the lobe's edge, pushing out toes
            'emitters': [dict(name='Vent', shape='box', position=(-0.85, 0.045, 0.0), size=(0.18, 0.045, 0.26),
                              velocity=(0.4, 0.0, 0.0), **stream),
                         dict(name='Breakout', shape='sphere', position=(0.0, 0.06, 0.38), size=(0.08, 0.08, 0.08),
                              velocity=(0.25, 0.0, 0.35), start=0.8, **stream),
                         dict(name='Breakout 2', shape='sphere', position=(0.22, 0.06, -0.3), size=(0.08, 0.08, 0.08),
                              velocity=(0.35, 0.0, -0.25), start=2.2, **stream)],
        },
        'sea_swell': {
            'name': 'Crates at sea', 'category': 'Liquids', 'size': '6 m of sea',
            'blurb': 'A patch of open sea in a swell, running on to the horizon, with two crates riding the waves: '
                     'they bob, tilt and roll as the crests pass. Waves, whitecaps and wind chop on the open water; '
                     'deep water (Bottomless) seen from above.',
            'domain': {'kind': 'liquid', 'size_x': 6.0, 'size_y': 2.2, 'size_z': 6.0, 'resolution': 176, 'preroll': 1.5,
                       'substeps_max': 12, 'cfl': 1.5},
            'liquid': {'water_level': 1.3, 'narrow_band': True, 'ocean_height': 0.35, 'ocean_length': 5.0,
                       'ocean_dir': 60.0, 'ocean_spread': 0.35, 'ocean_chop': 0.8, 'wind_speed': 5.0, 'wind_dir': 60.0,
                       'ww_min_speed': 2.5},
            'water': {'color': (0.12, 0.42, 0.48), 'clarity': 2.5, 'murk': 0.5, 'murk_color': (0.05, 0.16, 0.2),
                      'backdrop': 60.0, 'bottomless': True, 'colliders_look': 'shaded', 'ripple': 0.3},
            'lighting': dict(sun),
            'camera': {'distance': 5.5, 'target_y': 1.2, 'pitch': 12, 'yaw': 25, 'anchor_x': 0.5, 'anchor_y': 0.6, 'focal_mm': 35},
            'colliders': [
                dict(name='Crate', shape='box', size=(0.25, 0.2, 0.3), position=(-0.6, 1.45, 0.3), yaw=30.0,
                     floating=True, density=450.0),
                dict(name='Barrel', shape='cylinder', size=(0.2, 0.3, 0.2), position=(0.9, 1.5, -0.6),
                     floating=True, density=350.0),
            ],
            'emitters': [],
        },
        'ink_tank': {
            'name': 'Ink in water', 'category': 'Liquids', 'size': '50 cm tank',
            'blurb': 'A thin stream of ink poured into a tank of still water: it plunges in a dark thread, curls into '
                     'mushroom clouds and billows, and slowly clouds the water blue. Dye and cloudiness are per source.',
            'domain': {'kind': 'liquid', 'size_x': 0.5, 'size_y': 0.45, 'size_z': 0.3, 'resolution': 160, 'preroll': 0.5,
                       'substeps_max': 12, 'cfl': 1.5, 'open_sides': False},
            'liquid': {'whitewater': False, 'settle': True},
            'water': {'backdrop': 1.5, 'clarity': 4.0, 'color': (0.9, 0.96, 0.97), 'ripple': 0.0},
            'lighting': dict(sun),
            'camera': {'distance': 1.35, 'target_y': 0.2, 'pitch': 8, 'yaw': 5, 'anchor_x': 0.5, 'anchor_y': 0.5, 'focal_mm': 50},
            'emitters': [
                dict(name='Water', shape='box', position=(0.0, 0.16, 0.0), size=(0.25, 0.16, 0.15), **volume),
                dict(name='Ink', shape='cylinder', position=(-0.05, 0.4, 0.0), size=(0.006, 0.01, 0.006),
                     velocity=(0.0, -0.8, 0.0), start=0.6, stop=1.6, dye=(0.02, 0.03, 0.12), dye_amount=120.0, dye_cloud=0.25,
                     **stream),
            ],
        },
        'oil_water': {
            'name': 'Oil on water', 'category': 'Liquids', 'size': '1.2 m pool',
            'blurb': 'Oil poured onto still water: lighter than the water, it dives in, bobs back up and spreads into '
                     'an amber slick on the surface. Each source can pour its own density and dye.',
            'domain': {'kind': 'liquid', 'size_x': 1.2, 'size_y': 0.6, 'size_z': 1.2, 'resolution': 160, 'preroll': 1.0,
                       'substeps_max': 12, 'cfl': 1.5},
            'liquid': {'water_level': 0.2, 'settle': True, 'ww_min_speed': 1.5},
            'water': {'backdrop': 4.0, 'clarity': 2.0, 'color': (0.6, 0.8, 0.76)},
            'lighting': dict(sun),
            'camera': {'distance': 1.5, 'target_y': 0.18, 'pitch': 24, 'yaw': 10, 'anchor_x': 0.5, 'anchor_y': 0.6, 'focal_mm': 45},
            'emitters': [
                dict(name='Oil', shape='cylinder', position=(0.0, 0.5, 0.0), size=(0.02, 0.01, 0.02),
                     velocity=(0.0, -1.0, 0.0), start=1.0, stop=2.5, dye=(0.75, 0.45, 0.08), dye_amount=25.0, dye_cloud=0.05,
                     liquid_density=900.0, **stream),
            ],
        },
        'river_post': {
            'name': 'River past a post', 'category': 'Liquids', 'size': '3 m of river',
            'blurb': 'A river flowing past a post: the water piles up against it, splits and trails a wake of eddies and '
                     'foam downstream. The open water carries on to the horizon, flowing with the current.',
            'domain': {'kind': 'liquid', 'size_x': 3.0, 'size_y': 0.9, 'size_z': 1.6, 'resolution': 176, 'preroll': 2.0,
                       'substeps_max': 12, 'cfl': 1.5},
            'liquid': {'water_level': 0.45, 'current_speed': 1.3, 'current_dir': 90.0, 'ww_min_speed': 1.2},
            'water': {'backdrop': 8.0, 'clarity': 0.8, 'color': (0.55, 0.7, 0.6), 'murk': 0.6, 'colliders_look': 'shaded'},
            'lighting': dict(sun),
            'camera': {'distance': 3.2, 'target_y': 0.4, 'pitch': 24, 'yaw': 30, 'anchor_x': 0.5, 'anchor_y': 0.6, 'focal_mm': 35},
            'colliders': [dict(name='Post', shape='cylinder', size=(0.08, 0.6, 0.08), position=(-0.6, 0.5, 0.0))],
            'emitters': [],
        },
        'towel_dip': {
            'name': 'Towel dipped in water', 'category': 'Liquids', 'size': '40 x 55 cm towel',
            'blurb': 'A cotton hand towel dunked in a pond and lifted out again. Below the water it darkens as it soaks, '
                     'and comes out heavy and limp, with a sharp line where the wet ends. Water runs down the cloth to '
                     'its hem and drips back into the pond, faster at first, then more and more slowly; the towel stays '
                     'damp. It is a Fabric whose pins are keyframed; set Wet at start to hang it wet.',
            'domain': {'kind': 'liquid', 'size_x': 1.8, 'size_y': 1.3, 'size_z': 1.4, 'resolution': 160, 'preroll': 0.5,
                       'substeps_max': 12, 'cfl': 1.5},
            'render': {'end': 216},
            'liquid': {'water_level': 0.5, 'settle': True, 'narrow_band': True, 'ww_min_speed': 0.8},
            'water': {'backdrop': 4.0, 'clarity': 2.0, 'color': (0.6, 0.8, 0.76)},
            'lighting': {**sun, 'sun_intensity': 2.2},
            'camera': {'distance': 2.1, 'target_y': 0.66, 'pitch': 10, 'yaw': 20, 'anchor_x': 0.5, 'anchor_y': 0.97, 'focal_mm': 35},
            'emitters': [],
            'fabrics': [dict(name='Towel', width=0.4, height=0.55, pins='top', material='cotton', weight=2.5,
                             colour=(0.16, 0.38, 0.62), detail=36, burnable=False,
                             position=K((0.0, (0.0, 0.95, 0.0)), (0.5, (0.0, 0.95, 0.0)), (1.6, (0.0, 0.38, 0.0)),
                                        (3.8, (0.0, 0.38, 0.0)), (5.0, (0.0, 0.88, 0.0))))],
        },
        # 'hose_on_fire' (kept in LIQUID_ORDER) is defined in both_presets.py, with the fire-and-water engine
    }
    # the pond, seen from under its surface: the sky only through Snell's window overhead, the surface
    # a mirror of the pond floor outside it, the floaters hanging from it
    fl = out['floating']
    out['pond_below'] = {
        **fl, 'name': 'Pond from below', 'size': '2.4 m pond',
        'blurb': 'The floating things seen from under the water: the sky through a bright circle overhead '
                 "(Snell's window), the surface a rippling mirror of the pond floor all around it, the crate and "
                 'the ball hanging from it, and the water fading green into the distance. The camera is under the '
                 'water level; the view follows it.',
        'camera': {'distance': 0.4, 'target_y': 0.3, 'pitch': -30, 'yaw': 150, 'focal_mm': 16, 'use_anchor': False},
    }
    out['rain_pond'] = {
        'name': 'Rain on a pond', 'category': 'Liquids', 'size': '3 m pond',
        'blurb': 'Steady rain on still water: rings spreading from every drop, little crowns of spray where they '
                 'land, streaks falling through the air. Rain sets the rate (mm/h), Raindrop size the drops.',
        'domain': {'kind': 'liquid', 'size_x': 3.0, 'size_y': 0.6, 'size_z': 3.0, 'resolution': 160, 'preroll': 1.0,
                   'substeps_max': 12, 'cfl': 1.5},
        'liquid': {'water_level': 0.25, 'settle': True, 'narrow_band': True, 'rain': 25.0, 'rain_drop': 2.5,
                   'ww_min_speed': 2.0},
        'water': {'backdrop': 10.0, 'clarity': 1.0, 'color': (0.35, 0.5, 0.5), 'ripple': 0.1},
        'lighting': {**sun, 'sun_intensity': 1.2, 'sun_elevation': 40.0, 'sun_azimuth': -70.0, 'ambient': (0.6, 0.64, 0.7)},
        'camera': {'distance': 1.8, 'target_y': 0.25, 'pitch': 10, 'yaw': 20, 'anchor_x': 0.5, 'anchor_y': 0.5, 'focal_mm': 35},
        'emitters': [],
    }
    return out
