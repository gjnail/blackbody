"""Built-in presets for liquids that freeze, melt, boil and evaporate (Liquid › Heat and phase changes,
engine/liquid_thermal.py), at real sizes and temperatures.

presets.py registers them (phase_presets(K) returns the specs, K being its keyframe helper). Scenes with
steam (boiling, evaporating into cold air) are fire-and-liquid boxes with no fire in them: the box's gas
is the air, which carries the vapour off and shows it as steam where it condenses.
"""
from __future__ import annotations

PHASE_ORDER = ['ice_cubes', 'ice_melt', 'pond_freeze', 'frozen_pour', 'boiling_pot', 'hot_plate', 'quench', 'steaming_pool',
               'boiling_throw']


def phase_presets(K):
    sun = {'sun_on': True, 'sun_intensity': 3.0, 'sun_azimuth': -40.0, 'sun_elevation': 35.0, 'ambient': (0.55, 0.62, 0.72)}
    winter = {'sun_on': True, 'sun_intensity': 2.2, 'sun_azimuth': -60.0, 'sun_elevation': 14.0, 'sun_color': (1.0, 0.86, 0.7),
              'ambient': (0.5, 0.58, 0.72)}
    indoor = {'sun_on': True, 'sun_intensity': 2.5, 'sun_azimuth': 30.0, 'sun_elevation': 55.0, 'sun_color': (1.0, 0.9, 0.78),
              'ambient': (0.42, 0.42, 0.44)}
    stream = dict(liquid_mode='stream', vel_blend=1.0, embers=False, noise=0.0)
    volume = dict(liquid_mode='fill', vel_blend=0.0, embers=False, noise=0.0)
    # fire and liquid in one box, with no fire: the gas is the air round the water, and shows its steam
    air_box = {'fuel_scale': 0.0, 'cooling': 0.6, 'smoke_dissipation': 1.0}
    return {
        'ice_cubes': {
            'name': 'Ice cubes in water', 'category': 'Ice and steam', 'size': '4 cm cubes, 24 cm tank',
            'blurb': 'Four ice cubes from the freezer (-18 C) dropped into a tank of water at room temperature. They '
                     'plunge trailing bubbles, bob back up and float with about a tenth of their height out of the water, '
                     'as ice does: clear at the rim, milky at the core where they froze last, wet and glossy as they start '
                     'to melt.',
            'domain': {'kind': 'liquid', 'size_x': 0.24, 'size_y': 0.24, 'size_z': 0.24, 'resolution': 128, 'preroll': 1.0,
                       'open_sides': False, 'substeps_max': 14, 'cfl': 1.5},
            'liquid': {'thermal': True, 'liquid_temp': 21.0, 'air_temp': 21.0, 'ground_temp': 21.0, 'settle': True,
                       'ww_min_speed': 0.4, 'surface_tension': 0.073},
            'water': {'backdrop': 1.0, 'clarity': 6.0, 'color': (0.8, 0.94, 0.96), 'ripple': 0.1, 'ripple_freq': 90.0,
                      'frost': 0.8, 'ice_cloud': 1.5, 'crystal_size': 0.006},
            'lighting': dict(indoor),
            'camera': {'distance': 0.5, 'target_y': 0.09, 'pitch': 20, 'yaw': 12, 'anchor_x': 0.5, 'anchor_y': 0.86, 'focal_mm': 50},
            'emitters': [
                dict(name='Water', shape='box', position=(0.0, 0.05, 0.0), size=(0.12, 0.05, 0.12), start=-1.0, **volume),
                dict(name='Ice cube 1', shape='box', position=(-0.04, 0.2, 0.03), size=(0.02, 0.02, 0.02), yaw=20.0,
                     velocity=(0.1, -0.4, 0.0), start=0.1, temp_own=True, liquid_temp=-18.0, **volume),
                dict(name='Ice cube 2', shape='box', position=(0.045, 0.2, -0.035), size=(0.02, 0.02, 0.02), yaw=-35.0,
                     velocity=(-0.1, -0.2, 0.05), start=0.4, temp_own=True, liquid_temp=-18.0, **volume),
                dict(name='Ice cube 3', shape='box', position=(0.0, 0.2, 0.05), size=(0.02, 0.02, 0.02), yaw=5.0,
                     velocity=(0.0, -0.6, -0.1), start=0.7, temp_own=True, liquid_temp=-18.0, **volume),
                dict(name='Ice cube 4', shape='box', position=(0.05, 0.21, 0.05), size=(0.02, 0.02, 0.02), yaw=40.0,
                     velocity=(-0.15, -0.3, -0.05), start=1.0, temp_own=True, liquid_temp=-18.0, **volume),
            ],
        },
        'ice_melt': {
            'name': 'Ice melting', 'category': 'Ice and steam', 'size': '4 cm cubes, time-lapse',
            'blurb': 'Four ice cubes standing in a shallow tray of warm water (30 C), heat sped up 40 times (5 s is 3 '
                     'minutes): their edges round off, they shrink and settle, and the meltwater runs down their sides and '
                     'spreads into the tray. The cold meltwater sinks under the warm water around them.',
            'domain': {'kind': 'liquid', 'size_x': 0.26, 'size_y': 0.1, 'size_z': 0.2, 'resolution': 128, 'preroll': 0.3,
                       'open_sides': False, 'substeps_max': 12, 'cfl': 1.5},
            'liquid': {'thermal': True, 'liquid_temp': 30.0, 'air_temp': 24.0, 'air_humidity': 45.0, 'ground_temp': 24.0,
                       'heat_speed': 40.0, 'settle': True, 'surface_tension': 0.073, 'contact_angle': 60.0},
            'water': {'backdrop': 1.0, 'clarity': 6.0, 'color': (0.8, 0.94, 0.96), 'ripple': 0.05, 'ripple_freq': 90.0,
                      'frost': 0.5, 'ice_cloud': 1.5, 'crystal_size': 0.006},
            'lighting': dict(indoor),
            'camera': {'distance': 0.42, 'target_y': 0.02, 'pitch': 24, 'yaw': 10, 'anchor_x': 0.5, 'anchor_y': 0.6, 'focal_mm': 50},
            'emitters': [
                dict(name='Water', shape='box', position=(0.0, 0.006, 0.0), size=(0.13, 0.006, 0.1), start=-1.0, **volume),
                dict(name='Ice cube 1', shape='box', position=(-0.06, 0.021, 0.02), size=(0.02, 0.02, 0.02), yaw=15.0,
                     start=-1.0, temp_own=True, liquid_temp=-18.0, **volume),
                dict(name='Ice cube 2', shape='box', position=(0.0, 0.021, -0.03), size=(0.02, 0.02, 0.02), yaw=-25.0,
                     start=-1.0, temp_own=True, liquid_temp=-18.0, **volume),
                dict(name='Ice cube 3', shape='box', position=(0.06, 0.021, 0.025), size=(0.02, 0.02, 0.02), yaw=40.0,
                     start=-1.0, temp_own=True, liquid_temp=-18.0, **volume),
                dict(name='Ice cube 4', shape='box', position=(-0.005, 0.021, 0.045), size=(0.02, 0.02, 0.02), yaw=5.0,
                     start=-1.0, temp_own=True, liquid_temp=-18.0, **volume),
            ],
        },
        'pond_freeze': {
            'name': 'Pond freezing over', 'category': 'Ice and steam', 'size': '1.6 m pond, time-lapse',
            'blurb': 'A small pond through a night of hard frost (-18 C, a light breeze), heat sped up 3000 times '
                     '(each second is 50 minutes): a skin of clear ice creeps out from the frozen banks and the rocks, '
                     'joins up across the pond and thickens, and the ripples die under it. Water is densest at 4 C, '
                     'so the pond freezes from the top down, over clear water that stays a little above freezing, '
                     'kept warm by the earth under it.',
            'domain': {'kind': 'liquid', 'size_x': 1.6, 'size_y': 0.4, 'size_z': 1.2, 'resolution': 160, 'preroll': 0.2,
                       'open_sides': False, 'substeps_max': 12, 'cfl': 1.5},
            'liquid': {'thermal': True, 'liquid_temp': 2.0, 'air_temp': -18.0, 'air_humidity': 75.0, 'ground_temp': 3.0,
                       'heat_speed': 3000.0, 'supercool': 0.3, 'wind_speed': 3.0, 'wind_dir': 60.0, 'settle': True},
            'water': {'backdrop': 6.0, 'clarity': 1.2, 'color': (0.45, 0.62, 0.62), 'ripple': 0.35, 'frost': 0.25, 'ice_cloud': 0.3,
                      'crystal_size': 0.03, 'colliders_look': 'shaded', 'standin_color': (0.24, 0.22, 0.2)},
            'lighting': dict(winter),
            'camera': {'distance': 2.1, 'target_y': 0.1, 'pitch': 28, 'yaw': 15, 'anchor_x': 0.5, 'anchor_y': 0.55, 'focal_mm': 35},
            # the pond's hollow (tools/make_pond_basin.py): frozen banks sloping down to a bed just under the
            # box's floor, which is the unfrozen mud at the ground's temperature. (A collider keeps its surface
            # at its temperature however much heat it takes, and has one temperature all over: the banks are
            # at the freezing point, frozen at the waterline, not a cold source under the water, and the rocks
            # just below it. They hold and seed the ice; the night air freezes the pond.)
            'colliders': [
                dict(name='Banks', shape='mesh', mesh='builtin:pond_basin.png', size=(3.2, 0.4, 2.4),
                     position=(0.0, -0.03, 0.0), temperature=0.0, keeps_temperature=True),
                dict(name='Rock', shape='sphere', size=(0.14, 0.14, 0.14), position=(-0.35, 0.08, 0.2), temperature=-1.0, keeps_temperature=True),
                dict(name='Stone', shape='sphere', size=(0.08, 0.08, 0.08), position=(0.4, 0.15, -0.25), temperature=-1.0, keeps_temperature=True),
            ],
            'emitters': [
                dict(name='Pond', shape='box', position=(0.0, 0.1, 0.0), size=(0.8, 0.1, 0.6), start=-1.0, **volume),
            ],
        },
        'frozen_pour': {
            'name': 'Water on dry ice', 'category': 'Ice and steam', 'size': '40 cm slab, -78 C',
            'blurb': 'Water poured onto a slab of dry ice on a frozen day: it freezes where it lands, first a glaze, '
                     'then a growing mound of white, rimed ice the rest of the stream runs over and freezes onto. Heat '
                     'sped up 20 times.',
            'domain': {'kind': 'liquid', 'size_x': 0.5, 'size_y': 0.45, 'size_z': 0.4, 'resolution': 128, 'preroll': 0.0,
                       'substeps_max': 14, 'cfl': 1.5},
            'liquid': {'thermal': True, 'liquid_temp': 4.0, 'air_temp': -20.0, 'air_humidity': 60.0, 'ground_temp': -20.0,
                       'heat_speed': 20.0, 'wall_drag': 4.0, 'ww_min_speed': 0.8},
            'water': {'backdrop': 3.0, 'clarity': 3.0, 'frost': 1.4, 'crystal_size': 0.008, 'colliders_look': 'shaded',
                      'standin_color': (0.42, 0.44, 0.47)},
            'lighting': dict(winter),
            'camera': {'distance': 0.95, 'target_y': 0.1, 'pitch': 20, 'yaw': 25, 'anchor_x': 0.5, 'anchor_y': 0.62, 'focal_mm': 40},
            # (the slab's top, 62.5 mm, is on a cell boundary at this box's resolutions: a top just above a
            # cell's centre leaves a sliver of the cell over it where water is solid to the solver, and
            # water poured into that sliver piles up and freezes there)
            'colliders': [
                dict(name='Dry ice', shape='box', size=(0.15, 0.03125, 0.12), position=(0.0, 0.03125, 0.0), temperature=-78.0, keeps_temperature=True),
            ],
            'emitters': [
                dict(name='Pour', shape='cylinder', position=(-0.07, 0.34, 0.0), size=(0.015, 0.01, 0.015),
                     velocity=(0.2, -0.5, 0.0), start=0.0, stop=4.0, jitter=0.02, **stream),
            ],
        },
        'boiling_pot': {
            'name': 'Boiling pot', 'category': 'Ice and steam', 'size': '20 cm pot, rolling boil',
            'blurb': 'A pot of water on full heat: steam bubbles stream up from spots on the hot bottom, the boil rolls '
                     'the surface, bursting bubbles spit drops, and steam rises off it, clear right over the water and '
                     'clouding as it cools in the kitchen air.',
            'domain': {'kind': 'both', 'size_x': 0.36, 'size_y': 0.45, 'size_z': 0.36, 'resolution': 112, 'preroll': 1.0,
                       'substeps_max': 14, 'cfl': 1.5},
            'combustion': dict(air_box, vapour_dissipation=1.2),
            'motion': {'buoyancy': 2.0, 'turbulence': 0.6, 'turb_freq': 12.0, 'vorticity': 1.0},
            'shading': {'ambient_k': 294.0, 'humidity': 55.0, 'steam_density': 0.35, 'detail': 0.3, 'detail_freq': 30.0},
            'liquid': {'thermal': True, 'liquid_temp': 99.0, 'ground_temp': 20.0, 'ww_min_speed': 0.6, 'bubble_size': 3.0},
            'water': {'backdrop': 1.5, 'clarity': 4.0, 'color': (0.82, 0.94, 0.95), 'bubbles': 1.2, 'ripple': 0.15,
                      'ripple_freq': 80.0, 'colliders_look': 'shaded', 'standin_color': (0.55, 0.56, 0.58)},
            'lighting': dict(indoor),
            'camera': {'distance': 0.75, 'target_y': 0.1, 'pitch': 38, 'yaw': 10, 'anchor_x': 0.5, 'anchor_y': 0.62, 'focal_mm': 40},
            'colliders': [
                # a pot on a burner: its base runs about 10 K over boiling (100 kW/m^2, a strong gas ring), its
                # walls, heated only by the water, at the water's temperature
                # (walls thicker than a real pot's so the grid holds them: a wall thinner than two cells leaks)
                dict(name='Pot', shape='cylinder', size=(0.108, 0.065, 0.108), position=(0.0, 0.065, 0.0), hollow=0.012,
                     opening=(0.12, 0.02, 0.12), opening_at=(0.0, 0.065, 0.0), temperature=100.0, keeps_temperature=True),
                dict(name='Pot base', shape='cylinder', size=(0.094, 0.004, 0.094), position=(0.0, 0.016, 0.0),
                     temperature=110.0, keeps_temperature=True),
            ],
            'emitters': [
                dict(name='Water', shape='cylinder', position=(0.0, 0.058, 0.0), size=(0.094, 0.038, 0.094), start=-1.0,
                     emits='liquid', **volume),
            ],
        },
        'hot_plate': {
            'name': 'Water on hot steel', 'category': 'Ice and steam', 'size': '2 plates, 150 C and 300 C',
            'blurb': 'A glass of water splashed across two steel plates. On the left one (150 C) it boils where it '
                     'lands, sizzling, spitting and steaming away in a few seconds. The right one (300 C) is past the '
                     'Leidenfrost point: the water rides on its own vapour, never quite touching the steel, so it glides '
                     'and skates across it in beads and puddles and boils off far more slowly.',
            'domain': {'kind': 'both', 'size_x': 0.9, 'size_y': 0.25, 'size_z': 0.45, 'resolution': 224, 'preroll': 0.0,
                       'substeps_max': 16, 'cfl': 1.5},
            'combustion': dict(air_box, vapour_dissipation=1.5),
            'motion': {'buoyancy': 2.5, 'turbulence': 0.5, 'turb_freq': 10.0},
            'shading': {'ambient_k': 294.0, 'humidity': 45.0, 'steam_density': 0.3, 'detail': 0.3, 'detail_freq': 30.0},
            'liquid': {'thermal': True, 'liquid_temp': 20.0, 'ground_temp': 20.0, 'ww_min_speed': 0.5, 'contact_angle': 70.0,
                       'surface_tension': 0.073, 'bubble_size': 1.5},
            'water': {'backdrop': 2.0, 'clarity': 5.0, 'colliders_look': 'shaded', 'standin_color': (0.12, 0.12, 0.13)},
            'lighting': dict(indoor),
            'camera': {'distance': 1.0, 'target_y': 0.03, 'pitch': 28, 'yaw': 0, 'anchor_x': 0.5, 'anchor_y': 0.62, 'focal_mm': 40},
            'colliders': [
                dict(name='Plate 150 C', shape='box', size=(0.2, 0.01, 0.2), position=(-0.22, 0.01, 0.0), temperature=150.0, keeps_temperature=True),
                dict(name='Plate 300 C', shape='box', size=(0.2, 0.01, 0.2), position=(0.22, 0.01, 0.0), temperature=300.0, keeps_temperature=True),
            ],
            'emitters': [
                dict(name='Splash left', shape='sphere', position=(-0.3, 0.13, 0.0), size=(0.028, 0.028, 0.028),
                     velocity=(0.9, -0.6, 0.0), radial=0.4, start=0.1, emits='liquid', **volume),
                dict(name='Splash right', shape='sphere', position=(0.14, 0.13, 0.0), size=(0.028, 0.028, 0.028),
                     velocity=(0.9, -0.6, 0.0), radial=0.4, start=0.1, emits='liquid', **volume),
            ],
        },
        'quench': {
            'name': 'Red-hot steel quenched', 'category': 'Ice and steam', 'size': '7 cm steel ball at 1100 C, 36 cm tank',
            'blurb': 'A ball of white-hot steel (1100 C) dropped into a tank of cold water in a dim workshop. The water flashes to steam round it '
                     'and a film of vapour holds it off the steel (film boiling); as the steel cools the film breaks and the '
                     'water boils hard against it, and steam pours off the surface. Its glow fades from orange to dull red '
                     'to nothing as the water takes its heat (Heat speed 12: about twelve times quicker than for real).',
            'domain': {'kind': 'both', 'size_x': 0.5, 'size_y': 0.5, 'size_z': 0.36, 'resolution': 128, 'preroll': 0.5,
                       'substeps_max': 14, 'cfl': 1.5, 'matter_heat_speed': 12.0},
            'combustion': dict(air_box, vapour_dissipation=1.0),
            'motion': {'buoyancy': 2.0, 'turbulence': 0.6, 'turb_freq': 12.0, 'vorticity': 1.0},
            # (the steel glows as bright as a flame that hot would in the Look: with no fire in the shot, Exposure is set for
            # the dim room, so 1100 C steel reads bright orange as it does to the eye, not a dull red)
            'shading': {'ambient_k': 294.0, 'humidity': 60.0, 'steam_density': 0.45, 'detail': 0.3, 'detail_freq': 30.0,
                        'exposure': 3.5},
            'liquid': {'thermal': True, 'liquid_temp': 18.0, 'ground_temp': 20.0, 'heat_speed': 12.0, 'ww_min_speed': 0.6,
                       'bubble_size': 3.0},
            'water': {'backdrop': 1.5, 'clarity': 4.0, 'color': (0.82, 0.94, 0.95), 'bubbles': 1.2, 'ripple': 0.15,
                      'ripple_freq': 80.0, 'colliders_look': 'holdout'},
            # (drawn on the stage: the tank in glass and the steel in CG, glowing as hot as it is: objheat.py)
            'composite': {'backdrop': 'stage', 'floor': 'concrete'},
            'lighting': {**indoor, 'sun_intensity': 0.35, 'ambient_intensity': 0.35},   # (a dim workshop: the glow shows)
            'camera': {'distance': 0.85, 'target_y': 0.1, 'pitch': 48, 'yaw': 15, 'anchor_x': 0.5, 'anchor_y': 0.55, 'focal_mm': 35},
            'colliders': [
                dict(name='Tank', shape='box', size=(0.18, 0.1, 0.13), position=(0.0, 0.1, 0.0), hollow=0.014,
                     opening=(0.17, 0.02, 0.12), opening_at=(0.0, 0.1, 0.0), material='glass'),
                dict(name='Steel ball', shape='sphere', size=(0.035, 0.035, 0.035), position=(0.02, 0.36, 0.0),
                     material='steel', temperature=1100.0, dynamic=True, release=0.6),
            ],
            'emitters': [
                dict(name='Water', shape='box', position=(0.0, 0.078, 0.0), size=(0.164, 0.064, 0.114), start=-0.5,
                     emits='liquid', **volume),
            ],
        },
        'steaming_pool': {
            'name': 'Hot pool in frost', 'category': 'Ice and steam', 'size': '2 m pool, -12 C air',
            'blurb': 'A hot spring pool (40 C) on a freezing morning: the water evaporates into the dry, cold air, which '
                     'cannot hold the vapour, so it condenses straight into a fog that rises off the whole surface and '
                     'drifts away on the breeze (sea smoke, steam fog).',
            'domain': {'kind': 'both', 'size_x': 2.4, 'size_y': 1.4, 'size_z': 1.8, 'resolution': 144, 'preroll': 2.0,
                       'open_sides': False, 'substeps_max': 10, 'cfl': 1.5},
            'combustion': dict(air_box, vapour_dissipation=0.4, latent_heat=1.0),
            'motion': {'buoyancy': 3.0, 'turbulence': 1.2, 'turb_freq': 3.0, 'vorticity': 1.4, 'wind_speed': 0.8, 'wind_dir': 70.0},
            'shading': {'ambient_k': 261.0, 'humidity': 70.0, 'steam_density': 0.6, 'detail': 0.4, 'detail_freq': 6.0},
            'liquid': {'thermal': True, 'liquid_temp': 40.0, 'ground_temp': -5.0, 'settle': True},
            'water': {'backdrop': 8.0, 'clarity': 2.5, 'color': (0.6, 0.85, 0.82), 'ripple': 0.2},
            'lighting': dict(winter),
            'camera': {'distance': 3.2, 'target_y': 0.3, 'pitch': 14, 'yaw': 10, 'anchor_x': 0.5, 'anchor_y': 0.7, 'focal_mm': 35},
            'emitters': [
                dict(name='Pool', shape='box', position=(0.0, 0.14, 0.0), size=(1.2, 0.14, 0.9), start=-2.0, emits='liquid',
                     **volume),
            ],
        },
        'boiling_throw': {
            'name': 'Boiling water into -30 C air', 'category': 'Ice and steam', 'size': '5 litres thrown',
            'blurb': 'A pot of boiling water flung into the air on a -30 C day. The spray is so hot and the air so cold '
                     'and dry that the drops evaporate as they fly, and the vapour freezes into a roaring cloud of ice '
                     'fog, while what is left of the water falls as a shower of cooling drops.',
            'domain': {'kind': 'both', 'size_x': 3.6, 'size_y': 3.2, 'size_z': 2.4, 'resolution': 192, 'preroll': 0.0,
                       'substeps_max': 14, 'cfl': 1.5},
            'combustion': dict(air_box, vapour_dissipation=0.6, latent_heat=1.0),
            'motion': {'buoyancy': 3.0, 'turbulence': 1.5, 'turb_freq': 3.0, 'vorticity': 1.6},
            'shading': {'ambient_k': 243.0, 'humidity': 60.0, 'steam_density': 0.7, 'detail': 0.4, 'detail_freq': 5.0},
            'liquid': {'thermal': True, 'liquid_temp': 97.0, 'ground_temp': -25.0, 'ww_min_speed': 1.0, 'ww_amount': 2.0},
            'water': {'backdrop': 15.0, 'clarity': 4.0},
            'lighting': dict(winter),
            'camera': {'distance': 6.5, 'target_y': 1.6, 'pitch': 3, 'yaw': 15, 'anchor_x': 0.5, 'anchor_y': 0.9, 'focal_mm': 28},
            'emitters': [
                dict(name='Thrown water', shape='sphere', position=(-1.0, 1.1, 0.0), size=(0.1, 0.1, 0.1),
                     velocity=(1.6, 6.0, 0.0), radial=2.0, start=0.1, emits='liquid', **volume),
            ],
        },
    }
