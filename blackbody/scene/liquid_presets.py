"""Built-in liquid presets: real water effects at real sizes, each a complete starting point.

presets.py registers them (liquid_presets(K) returns the specs, K being its keyframe helper).
Liquids need finer grids than fire, so the boxes are kept tight around the action.
"""
from __future__ import annotations

LIQUID_ORDER = ['water_pour', 'rock_splash', 'bucket_throw', 'fountain', 'hose', 'wave', 'waterfall', 'spill']


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
    return {
        'water_pour': {
            'name': 'Pouring water', 'category': 'Liquids', 'size': '80 cm pour',
            'blurb': 'A steady stream of water poured from knee height onto the ground, splashing and spreading into a puddle.',
            'domain': {'kind': 'liquid', 'size_x': 1.4, 'size_y': 1.0, 'size_z': 1.2, 'resolution': 176, 'preroll': 0.0,
                       'substeps_max': 12, 'cfl': 1.5},
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
    }
