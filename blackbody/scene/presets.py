"""Built-in presets: real fires at real sizes, each a complete starting point for a shot."""
from __future__ import annotations

from .anim import Curve
from .model import Scene
from .params import coerce, collider_defaults, emitter_defaults, param


class K:
    """Keyframes in a preset, timed in seconds from the first frame so they land on the same moments
    whatever the shot's frame rate and first frame: K((0.0, a), (1.5, b), interp='linear')."""

    def __init__(self, *keys, interp='smooth'):
        self.keys = keys
        self.interp = interp

    def curve(self, p, fps, start):
        return Curve([[start + t * fps, coerce(p, v), self.interp] for t, v in self.keys])


def _value(p, v, scene):
    return v.curve(p, scene.fps, scene.start) if isinstance(v, K) else coerce(p, v)


def _em(scene, **kw):
    e = emitter_defaults()
    e.update({k: _value(param('emitter', k), v, scene) for k, v in kw.items()})
    return e


def _col(scene, i, **kw):
    c = collider_defaults()
    c['name'] = f'Collider {i + 1}'
    c.update({k: _value(param('collider', k), v, scene) for k, v in kw.items()})
    return c


def _apply(scene: Scene, spec: dict):
    # the frame range first: keyframes are placed in it
    for sec in sorted((k for k in spec if isinstance(spec[k], dict)), key=lambda k: k != 'render'):
        for k, v in spec[sec].items():
            if isinstance(v, K):
                scene.data[sec][k] = v.curve(param(sec, k), scene.fps, scene.start)
            else:
                scene.set((sec, k), v)
    if 'emitters' in spec:
        scene.emitters = [_em(scene, **e) for e in spec['emitters']]
    if 'colliders' in spec:
        scene.colliders = [_col(scene, i, **c) for i, c in enumerate(spec['colliders'])]
    if 'fabrics' in spec:
        scene.fabrics = []
        for f in spec['fabrics']:
            i = scene.add_fabric(**{k: v for k, v in f.items() if not isinstance(v, K)})
            for k, v in f.items():
                if isinstance(v, K):
                    scene.fabrics[i][k] = v.curve(param('fabric', k), scene.fps, scene.start)
    if 'matter' in spec:
        scene.matter = []
        for m in spec['matter']:
            scene.add_matter(**m)
    if 'lights' in spec:
        scene.lights = []
        for l in spec['lights']:
            scene.add_light(**l)
    if 'links' in spec:
        import copy
        scene.links = copy.deepcopy(spec['links'])
    if 'strands' in spec:
        scene.strands = []
        for g in spec['strands']:
            scene.add_strands(**g)


PRESETS = {
    'campfire': {
        'name': 'Campfire', 'category': 'Fires', 'size': '1.2 m flames',
        'blurb': 'A wood campfire on the ground: lively flame tongues, grey smoke, a steady stream of embers.',
        'domain': {'size_x': 2.2, 'size_y': 3.6, 'size_z': 2.2, 'resolution': 144, 'preroll': 2.5, 'time_scale': 1.6, 'substeps_max': 12},
        'combustion': {'burn_rate': 5.0, 'heat': 0.6, 'soot': 0.35, 'cooling': 2.2, 'flame_life': 0.08},
        'motion': {'puffing': 0.6, 'buoyancy': 5.5, 'turbulence': 3.5, 'turb_freq': 2.5, 'vorticity': 1.6, 'disturbance': 2.5, 'disturb_block': 0.04},
        # flame shape checked against footage of a real wood campfire: opaque, sharp-edged tongues, little glowing soot
        'shading': {'flame_k': 1650, 'max_k': 2250, 'smoke_density': 2.0, 'smoke_albedo': (0.55, 0.54, 0.53), 'coal_bed': 1.0,
                    'coal_height': 0.08, 'flame_threshold': 0.1, 'flame_sharpness': 3.2, 'flame_absorption': 10.0,
                    'soot_glow': 0.1, 'detail': 0.5},
        'embers': {'rate': 90, 'launch': 1.6, 'lifetime': 2.4},
        'camera': {'distance': 6.5, 'target_y': 1.2, 'pitch': 5, 'anchor_x': 0.5, 'anchor_y': 0.9, 'focal_mm': 35},
        'emitters': [
            dict(noise_rise=1.5, name='Log A', shape='capsule', position=(-0.42, 0.07, -0.18), end=(0.4, 0.07, 0.2), size=(0.08, 0.08, 0.08), fuel=10, temperature=0.45),
            dict(noise_rise=1.5, name='Log B', shape='capsule', position=(-0.35, 0.09, 0.25), end=(0.38, 0.09, -0.22), size=(0.08, 0.08, 0.08), fuel=10, temperature=0.45, seed=3),
            dict(noise_rise=1.5, name='Coal bed', shape='cylinder', position=(0, 0.04, 0), size=(0.34, 0.04, 0.34), fuel=8, temperature=0.4, seed=7),
        ],
    },
    'bonfire': {
        'name': 'Bonfire', 'category': 'Fires', 'size': '5 m flames',
        'blurb': 'A large pile of burning wood: tall rolling flames, a thick smoke column and a heavy shower of embers.',
        'domain': {'size_x': 7.0, 'size_y': 13.0, 'size_z': 7.0, 'resolution': 160, 'preroll': 3.0, 'time_scale': 1.5, 'substeps_max': 12},
        'combustion': {'burn_rate': 4.0, 'heat': 0.45, 'soot': 0.3, 'cooling': 1.6, 'radiative': 0.3, 'flame_life': 0.1, 'smoke_dissipation': 0.35},
        'motion': {'puffing': 0.6, 'buoyancy': 6.5, 'turbulence': 4.0, 'turb_freq': 0.8, 'turb_rise': 3.0, 'vorticity': 2.0, 'disturbance': 3.0,
                   'disturb_block': 0.14},
        'shading': {'flame_k': 1600, 'max_k': 2200, 'smoke_density': 3.5, 'detail_freq': 1.2, 'smoke_albedo': (0.16, 0.155, 0.15),
                    'coal_bed': 1.0, 'coal_height': 0.25},
        'lighting': {'light_spread': 1.2},
        'embers': {'rate': 450, 'launch': 4.0, 'spread': 1.6, 'lifetime': 3.5, 'size_max': 0.016, 'turb_freq': 0.8, 'turbulence': 5.0},
        'camera': {'distance': 18.0, 'target_y': 3.4, 'pitch': 3, 'anchor_y': 0.92, 'focal_mm': 35},
        'emitters': [
            dict(name='Wood pile', shape='cone', position=(0, 0.45, 0), size=(1.5, 0.45, 1.5), fuel=12, temperature=0.45, noise_freq=1.2, noise_rise=2.5),
        ],
    },
    'torch': {
        'name': 'Torch', 'category': 'Fires', 'size': '40 cm flame',
        'blurb': 'A hand-held torch head in mid-air: a compact, fast-flickering flame with a thin smoke trail.',
        'domain': {'size_x': 0.7, 'size_y': 1.6, 'size_z': 0.7, 'resolution': 128, 'ground': False, 'preroll': 1.5, 'time_scale': 1.75, 'substeps_max': 12},
        'combustion': {'burn_rate': 7.0, 'heat': 0.55, 'soot': 0.3, 'cooling': 3.5, 'flame_life': 0.06},
        'motion': {'puffing': 0.6, 'buoyancy': 4.0, 'turbulence': 2.0, 'turb_freq': 8.0, 'turb_rise': 1.0, 'vorticity': 1.2, 'disturbance': 1.6,
                   'disturb_block': 0.012},
        'shading': {'flame_k': 1700, 'smoke_density': 6.0, 'detail_freq': 10.0, 'flame_absorption': 15.0},
        'lighting': {'light_spread': 0.12},
        'embers': {'rate': 12, 'launch': 0.8, 'lifetime': 1.2, 'size_max': 0.004, 'turb_freq': 6.0},
        'camera': {'distance': 2.4, 'target_y': 0.45, 'pitch': 2, 'anchor_y': 0.75, 'focal_mm': 35},
        'emitters': [
            dict(name='Torch head', shape='cylinder', position=(0, 0.1, 0), size=(0.055, 0.06, 0.055), fuel=18, temperature=0.5,
                 noise_freq=14.0, noise_rise=1.25, noise=0.7),
        ],
    },
    'candle': {
        'name': 'Candle', 'category': 'Small flames', 'size': '4 cm flame',
        'blurb': 'A single candle flame: calm and teardrop-shaped with a blue base. Pair with a slight breeze for flicker.',
        'domain': {'size_x': 0.08, 'size_y': 0.2, 'size_z': 0.08, 'resolution': 96, 'ground': False, 'preroll': 1.5, 'sponge': 6,
                   'maccormack': 0.3},
        'combustion': {'burn_rate': 25.0, 'heat': 0.5, 'soot': 0.02, 'cooling': 6.0, 'flame_life': 0.05, 'expansion': 0.3},
        'motion': {'buoyancy': 2.0, 'turbulence': 0.0, 'vorticity': 0.2, 'disturbance': 0.05, 'disturb_block': 0.004, 'damping': 0.4},
        'shading': {'flame_k': 1800, 'max_k': 2500, 'blue': 1.2, 'smoke_density': 2.0, 'detail': 0.0, 'flame_sharpness': 2.2,
                    'flame_absorption': 60.0, 'flame_occlusion': 0.2},
        'lighting': {'light_spread': 0.02},
        'embers': {'enabled': False},
        'camera': {'distance': 0.45, 'target_y': 0.05, 'pitch': 0, 'anchor_y': 0.7, 'focal_mm': 50},
        'emitters': [
            dict(name='Wick', shape='sphere', position=(0, 0.017, 0), size=(0.005, 0.009, 0.005), fuel=40, temperature=0.6, noise=0.1,
                 embers=False),
        ],
    },
    'gas_ring': {
        'name': 'Gas burner', 'category': 'Small flames', 'size': '10 cm flames',
        'blurb': 'A ring of clean gas flames: blue at the base, almost no smoke.',
        'domain': {'size_x': 0.35, 'size_y': 0.3, 'size_z': 0.35, 'resolution': 112, 'ground': True, 'preroll': 1.0},
        'combustion': {'burn_rate': 20.0, 'heat': 0.35, 'soot': 0.0, 'cooling': 8.0, 'flame_life': 0.05, 'flame_gain': 0.5},
        'motion': {'buoyancy': 2.5, 'turbulence': 0.6, 'turb_freq': 30.0, 'vorticity': 0.5, 'disturbance': 0.3, 'disturb_block': 0.004},
        'shading': {'flame_k': 1900, 'blue': 3.0, 'flame_density': 0.5, 'dynamic_range': 0.6, 'detail': 0.0},
        'lighting': {'light_spread': 0.04},
        'embers': {'enabled': False},
        'camera': {'distance': 0.9, 'target_y': 0.05, 'pitch': 18, 'anchor_y': 0.62, 'focal_mm': 50},
        'emitters': [
            dict(name='Burner ring', shape='ring', position=(0, 0.02, 0), size=(0.075, 0.008, 0.075), fuel=30, temperature=0.6,
                 velocity=(0, 0.6, 0), vel_blend=0.3, noise=0.4, noise_freq=40.0, embers=False),
        ],
    },
    'pool_fire': {
        'name': 'Fuel spill', 'category': 'Fires', 'size': '3 m flames',
        'blurb': 'Burning petrol spread on the ground: pulsing orange flames and thick black smoke.',
        'domain': {'size_x': 5.0, 'size_y': 9.0, 'size_z': 5.0, 'resolution': 144, 'preroll': 2.5, 'time_scale': 2.2, 'substeps_max': 12},
        'combustion': {'burn_rate': 4.0, 'heat': 0.5, 'soot': 1.2, 'cooling': 1.8, 'radiative': 0.25, 'flame_life': 0.1, 'smoke_dissipation': 0.2},
        'motion': {'puffing': 0.6, 'buoyancy': 6.0, 'turbulence': 3.0, 'turb_freq': 1.1, 'turb_rise': 2.0, 'vorticity': 2.0, 'disturbance': 2.5,
                   'disturb_block': 0.08},
        'shading': {'flame_k': 1550, 'max_k': 2100, 'smoke_density': 9.0, 'smoke_albedo': (0.05, 0.05, 0.05), 'detail_freq': 1.5},
        'lighting': {'light_spread': 0.8},
        'embers': {'rate': 20, 'launch': 2.0},
        'camera': {'distance': 16.0, 'target_y': 3.0, 'pitch': 4, 'anchor_y': 0.9, 'focal_mm': 35},
        'emitters': [
            dict(noise_rise=1.5, name='Spill', shape='cylinder', position=(0, 0.03, 0), size=(1.3, 0.03, 1.1), fuel=12, temperature=0.45, noise_freq=1.6),
        ],
    },
    'fire_line': {
        'name': 'Fire line', 'category': 'Fires', 'size': '6 m long',
        'blurb': 'A line of burning ground, like a fuel trail or the edge of a grass fire.',
        'domain': {'size_x': 8.0, 'size_y': 4.0, 'size_z': 2.4, 'resolution': 176, 'preroll': 2.0},
        'combustion': {'burn_rate': 5.0, 'heat': 0.55, 'soot': 0.4, 'cooling': 2.4, 'flame_life': 0.08},
        'motion': {'buoyancy': 5.0, 'turbulence': 3.0, 'turb_freq': 2.2, 'vorticity': 1.6, 'disturbance': 2.5, 'disturb_block': 0.04},
        'shading': {'flame_k': 1650},
        'embers': {'rate': 120},
        'camera': {'distance': 11.0, 'target_y': 1.0, 'pitch': 6, 'anchor_y': 0.8, 'focal_mm': 35},
        'emitters': [
            dict(name='Trail', shape='capsule', position=(-3.2, 0.05, 0), end=(3.2, 0.05, 0), size=(0.14, 0.14, 0.14), fuel=12,
                 temperature=0.45, noise_freq=2.5),
        ],
    },
    'fireball': {
        'name': 'Fireball', 'category': 'Explosions', 'size': '8 m ball',
        'blurb': 'A fuel-air fireball: a blazing burst that rolls up into a cauliflower ball of flame, then a dark smoke column.',
        'domain': {'size_x': 12.0, 'size_y': 16.0, 'size_z': 12.0, 'resolution': 176, 'preroll': 0.0, 'substeps_max': 10, 'mg_cycles': 3},
        'combustion': {'burn_rate': 6.0, 'heat': 1.1, 'soot': 0.6, 'cooling': 2.2, 'radiative': 0.35, 'flame_life': 0.3, 'flame_gain': 2.0,
                       'expansion': 1.0, 'rich': 3.0, 'smoke_dissipation': 0.2},
        'motion': {'buoyancy': 8.0, 'turbulence': 8.0, 'turb_freq': 0.8, 'turb_rise': 2.0, 'vorticity': 3.0, 'disturbance': 6.0,
                   'disturb_block': 0.3},
        'shading': {'flame_k': 1700, 'max_k': 2400, 'smoke_density': 4.0, 'smoke_albedo': (0.08, 0.075, 0.07), 'flame_occlusion': 0.6,
                    'flame_absorption': 1.5, 'detail_freq': 0.8},
        'lighting': {'light_spread': 2.0},
        'embers': {'rate': 800, 'launch': 12.0, 'spread': 10.0, 'lifetime': 1.6, 'gravity': 4.0, 'drag': 1.0, 'size_max': 0.02,
                   'turb_freq': 0.6, 'shutter': 0.25},
        'camera': {'distance': 34.0, 'target_y': 6.0, 'pitch': 3, 'anchor_y': 0.9, 'focal_mm': 35},
        'emitters': [
            dict(name='Burst', shape='sphere', position=(0, 1.2, 0), size=(1.0, 0.8, 1.0), fuel=25, temperature=1.2, radial=10.0,
                 vel_blend=0.5, start=0.0, stop=0.2, fade_in=0.02, fade_out=0.12, noise_freq=0.9, contrast=1.2),
        ],
    },
    'flamethrower': {
        'name': 'Flamethrower', 'category': 'Explosions', 'size': '7 m jet',
        'blurb': 'A pressurised jet of burning fuel that curls upward as it slows.',
        'domain': {'size_x': 9.0, 'size_y': 4.5, 'size_z': 3.0, 'resolution': 176, 'preroll': 1.5, 'substeps_max': 10, 'sponge': 4},
        'combustion': {'burn_rate': 3.0, 'heat': 0.55, 'soot': 0.9, 'cooling': 2.2, 'flame_life': 0.12, 'expansion': 0.8, 'flame_gain': 2.0,
                       'rich': 6.0},
        'motion': {'buoyancy': 5.0, 'turbulence': 7.0, 'turb_freq': 1.5, 'vorticity': 2.0, 'disturbance': 4.0, 'disturb_block': 0.06,
                   'damping': 0.3},
        'shading': {'flame_k': 1700, 'smoke_density': 5.0, 'smoke_albedo': (0.1, 0.1, 0.1)},
        'embers': {'rate': 60, 'launch': 0.5, 'spread': 2.0, 'shutter': 0.2},
        'camera': {'distance': 13.0, 'target_y': 1.6, 'pitch': 3, 'anchor_x': 0.5, 'anchor_y': 0.78, 'focal_mm': 35},
        'emitters': [
            dict(name='Nozzle', shape='capsule', position=(-3.7, 1.2, 0), end=(-3.1, 1.24, 0), size=(0.1, 0.1, 0.1), fuel=120,
                 temperature=0.9, velocity=(18.0, 1.0, 0.0), vel_blend=1.0, noise=0.5, noise_freq=4.0),
        ],
    },
    'vehicle_fire': {
        'name': 'Vehicle fire', 'category': 'Fires', 'size': 'car-sized',
        'blurb': 'Flames over a car-sized body with heavy black smoke. Place the collider box over the car in your footage.',
        'domain': {'size_x': 8.0, 'size_y': 12.0, 'size_z': 6.0, 'resolution': 160, 'preroll': 3.0, 'time_scale': 1.9, 'substeps_max': 12},
        'combustion': {'burn_rate': 4.0, 'heat': 0.5, 'soot': 1.4, 'cooling': 1.8, 'radiative': 0.25, 'flame_life': 0.1, 'smoke_dissipation': 0.18},
        'motion': {'puffing': 0.6, 'buoyancy': 6.0, 'turbulence': 3.5, 'turb_freq': 1.0, 'vorticity': 2.0, 'disturbance': 2.5, 'disturb_block': 0.08},
        'shading': {'flame_k': 1550, 'smoke_density': 10.0, 'smoke_albedo': (0.04, 0.04, 0.04), 'detail_freq': 1.5},
        'lighting': {'light_spread': 0.9},
        'embers': {'rate': 60},
        'camera': {'distance': 20.0, 'target_y': 3.5, 'pitch': 5, 'anchor_y': 0.9, 'focal_mm': 35},
        'colliders': [dict(name='Car body', shape='box', position=(0, 0.72, 0), size=(2.2, 0.72, 0.95))],
        'emitters': [
            dict(noise_rise=1.5, name='Cabin', shape='box', position=(0.3, 1.5, 0), size=(1.2, 0.12, 0.8), fuel=14, temperature=0.45, noise_freq=1.8),
            dict(noise_rise=1.5, name='Engine bay', shape='box', position=(-1.6, 1.25, 0), size=(0.6, 0.1, 0.8), fuel=10, temperature=0.45, seed=4),
        ],
    },
    'smoke_plume': {
        'name': 'Smoke plume', 'category': 'Smoke', 'size': '6 m column',
        'blurb': 'Smouldering smoke with no visible flame, for aftermath shots.',
        'domain': {'size_x': 4.0, 'size_y': 9.0, 'size_z': 4.0, 'resolution': 144, 'preroll': 4.0},
        'combustion': {'fuel_scale': 0.0, 'smoke_dissipation': 0.08, 'cooling': 0.8},
        'motion': {'buoyancy': 4.0, 'turbulence': 1.5, 'turb_freq': 1.2, 'vorticity': 1.5, 'mask_smoke': 1.0, 'disturbance': 0.5},
        'shading': {'smoke_density': 3.0, 'smoke_albedo': (0.45, 0.45, 0.46)},
        'lighting': {'sun_on': True, 'sun_intensity': 3.0},
        'embers': {'enabled': False},
        'camera': {'distance': 12.0, 'target_y': 3.0, 'pitch': 5, 'anchor_y': 0.92},
        'emitters': [
            dict(name='Smoulder', shape='cylinder', position=(0, 0.05, 0), size=(0.7, 0.05, 0.7), fuel=0.0, temperature=0.8, smoke=6.0,
                 embers=False, noise_freq=2.0),
        ],
    },
    # -- new kinds of fire ---------------------------------------------------------------------
    'fire_whirl': {
        'name': 'Fire whirl', 'category': 'Fires', 'size': '8 m column',
        'blurb': 'A pool of burning fuel drawn into a spinning column of flame: a fire tornado. The emitter\'s Swirl sets the spin.',
        'render': {'end': 168},   # long enough to reach what its library picture shows
        'domain': {'size_x': 5.0, 'size_y': 12.0, 'size_z': 5.0, 'resolution': 160, 'preroll': 4.0},
        'combustion': {'burn_rate': 2.2, 'heat': 0.6, 'soot': 0.45, 'cooling': 1.0, 'radiative': 0.2, 'flame_life': 0.14,
                       'smoke_dissipation': 0.3},
        'motion': {'buoyancy': 7.0, 'turbulence': 1.5, 'turb_freq': 1.2, 'turb_rise': 3.0, 'vorticity': 3.0, 'disturbance': 1.5,
                   'disturb_block': 0.08},
        'shading': {'flame_k': 1650, 'max_k': 2250, 'smoke_density': 4.0, 'smoke_albedo': (0.12, 0.115, 0.11), 'detail_freq': 1.5},
        'lighting': {'light_spread': 0.9},
        'embers': {'rate': 200, 'launch': 2.5, 'spread': 1.0, 'lifetime': 3.0, 'turbulence': 4.0},
        'camera': {'distance': 15.0, 'target_y': 4.0, 'pitch': 4, 'anchor_y': 0.93, 'focal_mm': 35},
        'emitters': [
            dict(name='Burning pool', shape='cylinder', position=(0, 0.04, 0), size=(0.9, 0.04, 0.9), fuel=14, temperature=0.5,
                 noise_freq=1.5, swirl=2.5, swirl_width=7.0),
        ],
    },
    'waved_torch': {
        'name': 'Waved torch', 'category': 'Fires', 'size': '40 cm flame, swinging',
        'blurb': 'A torch swung back and forth: the flame and its embers trail behind the moving head. Keyframe an emitter\'s position to move it.',
        'domain': {'size_x': 2.6, 'size_y': 2.2, 'size_z': 1.2, 'resolution': 160, 'ground': False, 'preroll': 1.0, 'substeps_max': 8},
        'combustion': {'burn_rate': 7.0, 'heat': 0.55, 'soot': 0.3, 'cooling': 3.5, 'flame_life': 0.06},
        'motion': {'buoyancy': 4.0, 'turbulence': 2.0, 'turb_freq': 8.0, 'turb_rise': 1.0, 'vorticity': 1.2, 'disturbance': 1.6,
                   'disturb_block': 0.012},
        'shading': {'flame_k': 1700, 'smoke_density': 6.0, 'detail_freq': 10.0, 'flame_absorption': 15.0},
        'lighting': {'light_spread': 0.15},
        'embers': {'rate': 30, 'launch': 0.6, 'lifetime': 1.4, 'size_max': 0.004, 'turb_freq': 6.0},
        'camera': {'distance': 3.8, 'target_y': 0.9, 'pitch': 2, 'anchor_y': 0.8, 'focal_mm': 35},
        'emitters': [
            dict(name='Torch head', shape='cylinder', size=(0.055, 0.06, 0.055), fuel=18, temperature=0.5, noise_freq=14.0,
                 noise_rise=0.5, noise=0.7,
                 position=K((0.0, (-0.8, 0.6, 0.0)), (0.7, (0.0, 0.85, 0.0)), (1.4, (0.8, 0.6, 0.0)), (2.1, (0.0, 0.85, 0.0)),
                            (2.8, (-0.8, 0.6, 0.0)), (3.5, (0.0, 0.85, 0.0)), (4.2, (0.8, 0.6, 0.0)), (4.9, (0.0, 0.85, 0.0)),
                            (5.6, (-0.8, 0.6, 0.0)))),
        ],
    },
    'hose_douse': {
        'name': 'Fire put out', 'category': 'Fires', 'size': '1.2 m flames, hosed',
        'blurb': 'A campfire hit by a hose after a second and a half: the flames collapse and the fire turns to billowing steam.',
        'domain': {'size_x': 3.2, 'size_y': 4.0, 'size_z': 2.4, 'resolution': 144, 'preroll': 2.5},
        'combustion': {'burn_rate': 5.0, 'heat': 0.6, 'soot': 0.35, 'cooling': 2.2, 'flame_life': 0.08, 'steam_yield': 400.0,
                       'vapour_dissipation': 1.0},
        'motion': {'buoyancy': 5.5, 'turbulence': 3.5, 'turb_freq': 2.5, 'vorticity': 1.6, 'disturbance': 2.5, 'disturb_block': 0.04,
                   'mask_smoke': 0.5},
        'shading': {'flame_k': 1650, 'max_k': 2250, 'smoke_density': 5.0, 'ambient_k': 285.0, 'humidity': 75.0, 'steam_density': 1.0},
        'lighting': {'ambient_intensity': 2.0},
        'embers': {'rate': 90, 'launch': 1.6, 'lifetime': 2.4},
        'camera': {'distance': 7.0, 'target_y': 1.2, 'pitch': 5, 'anchor_x': 0.5, 'anchor_y': 0.9, 'focal_mm': 35},
        'emitters': [
            dict(name='Log A', shape='capsule', position=(-0.42, 0.07, -0.18), end=(0.4, 0.07, 0.2), size=(0.08, 0.08, 0.08), fuel=10,
                 temperature=0.45, stop=1.9, fade_out=1.2),
            dict(name='Log B', shape='capsule', position=(-0.35, 0.09, 0.25), end=(0.38, 0.09, -0.22), size=(0.08, 0.08, 0.08), fuel=10,
                 temperature=0.45, seed=3, stop=1.9, fade_out=1.2),
            dict(name='Coal bed', shape='cylinder', position=(0, 0.04, 0), size=(0.34, 0.04, 0.34), fuel=8, temperature=0.4, seed=7,
                 stop=1.8, fade_out=0.8),
            dict(name='Hose spray', shape='capsule', position=(-1.4, 1.0, 0.0), end=(0.0, 0.25, 0.0), size=(0.35, 0.35, 0.35), fuel=0.0,
                 temperature=0.0, noise=0.4, embers=False, velocity=(5.0, -2.5, 0.0), vel_blend=0.15, douse=25.0,
                 start=1.5, fade_in=0.4),
        ],
    },
    'grinder_sparks': {
        'name': 'Grinder sparks', 'category': 'Sparks', 'size': 'angle-grinder cut',
        'blurb': 'A spray of white-hot metal sparks from a cut: launched in a narrow cone, falling at full gravity and skittering off the floor.',
        'domain': {'size_x': 4.0, 'size_y': 2.0, 'size_z': 2.0, 'resolution': 64, 'preroll': 1.0},
        'combustion': {'fuel_scale': 0.0},
        'motion': {'turbulence': 0.0, 'disturbance': 0.0, 'vorticity': 0.0},
        'embers': {'rate': 1500, 'count': 32768, 'launch': 14.0, 'spread': 1.2, 'direction': (1.0, -0.2, 0.0), 'cone': 12.0,
                   'drag': 0.35, 'gravity': 9.81, 'turbulence': 0.5, 'lifetime': 0.9, 'life_jitter': 0.7, 'temperature': 2300.0,
                   'cooling': 1.6, 'size_min': 0.0006, 'size_max': 0.0025, 'brightness': 1.8, 'bounce': 0.35, 'friction': 0.3,
                   'shutter': 0.5, 'fade_in': 0.0},
        'camera': {'distance': 3.2, 'target_y': 0.5, 'pitch': 6, 'anchor_x': 0.35, 'anchor_y': 0.85, 'focal_mm': 35},
        'emitters': [
            dict(name='Cut', shape='sphere', position=(-1.2, 0.8, 0.0), size=(0.01, 0.01, 0.01), fuel=0.0, temperature=0.0, noise=0.0),
        ],
    },
    'fireworks': {
        'name': 'Firework bursts', 'category': 'Sparks', 'size': '20 m bursts',
        'blurb': 'Three shells bursting into stars: red strontium, green copper and gold. Each star takes its emitter\'s colourant colour.',
        'domain': {'size_x': 16.0, 'size_y': 22.0, 'size_z': 10.0, 'resolution': 64, 'preroll': 0.0},
        'combustion': {'soot': 0.0, 'smoke_dissipation': 0.25},
        'motion': {'buoyancy': 1.0, 'turbulence': 0.5, 'disturbance': 0.0, 'mask_smoke': 1.0},
        'shading': {'smoke_density': 0.6, 'smoke_albedo': (0.5, 0.5, 0.52)},
        'lighting': {'ambient': (0.02, 0.025, 0.04)},
        'embers': {'rate': 30000, 'count': 16384, 'launch': 13.0, 'spread': 1.0, 'cone': 180.0, 'drag': 0.9, 'gravity': 6.0,
                   'turbulence': 0.6, 'lifetime': 2.4, 'life_jitter': 0.4, 'temperature': 2200.0, 'cooling': 0.5,
                   'size_min': 0.02, 'size_max': 0.04, 'brightness': 1.2, 'shutter': 0.7, 'fade_in': 0.0},
        'composite': {'bloom': 0.1},
        'camera': {'distance': 70.0, 'target_y': 13.0, 'pitch': -4, 'anchor_y': 0.95, 'focal_mm': 35},
        'emitters': [
            dict(name='Red shell', shape='sphere', position=(-3.0, 14.0, 0.0), size=(0.3, 0.3, 0.3), fuel=0.0, temperature=0.0, smoke=6.0,
                 noise=0.0, color=(1.0, 0.06, 0.04), color_amount=1.0, start=0.4, stop=0.46, fade_in=0.01, fade_out=0.01),
            dict(name='Green shell', shape='sphere', position=(3.5, 16.0, -1.0), size=(0.3, 0.3, 0.3), fuel=0.0, temperature=0.0, smoke=6.0,
                 noise=0.0, color=(0.1, 1.0, 0.3), color_amount=1.0, start=1.3, stop=1.36, fade_in=0.01, fade_out=0.01),
            dict(name='Gold shell', shape='sphere', position=(0.5, 12.0, 1.0), size=(0.3, 0.3, 0.3), fuel=0.0, temperature=0.0, smoke=6.0,
                 noise=0.0, start=2.2, stop=2.26, fade_in=0.01, fade_out=0.01),
        ],
    },
    'coloured_flames': {
        'name': 'Coloured flames', 'category': 'Small flames', 'size': '3 burners',
        'blurb': 'Three burners dosed with flame colourants: copper (green), strontium (red) and sodium (orange). Set any emitter\'s Colourant to colour its flame.',
        'domain': {'size_x': 1.2, 'size_y': 0.8, 'size_z': 0.5, 'resolution': 144, 'preroll': 1.5},
        'combustion': {'burn_rate': 12.0, 'heat': 0.45, 'soot': 0.03, 'cooling': 4.0, 'flame_life': 0.06, 'flame_gain': 0.25},
        'motion': {'buoyancy': 3.0, 'turbulence': 1.2, 'turb_freq': 12.0, 'vorticity': 0.8, 'disturbance': 0.6, 'disturb_block': 0.008},
        'shading': {'flame_k': 1800, 'blue': 1.0, 'colour_gain': 1.0, 'detail': 0.1, 'flame_absorption': 25.0},
        'lighting': {'light_spread': 0.06},
        'embers': {'enabled': False},
        'camera': {'distance': 2.2, 'target_y': 0.22, 'pitch': 6, 'anchor_y': 0.78, 'focal_mm': 50},
        'emitters': [
            dict(name='Copper', shape='cylinder', position=(-0.35, 0.02, 0), size=(0.05, 0.02, 0.05), fuel=25, temperature=0.6, noise=0.5,
                 noise_freq=20.0, color=(0.08, 1.0, 0.35), color_amount=3.0, embers=False),
            dict(name='Strontium', shape='cylinder', position=(0.0, 0.02, 0), size=(0.05, 0.02, 0.05), fuel=25, temperature=0.6, noise=0.5,
                 noise_freq=20.0, color=(1.0, 0.05, 0.03), color_amount=3.0, seed=2, embers=False),
            dict(name='Sodium', shape='cylinder', position=(0.35, 0.02, 0), size=(0.05, 0.02, 0.05), fuel=25, temperature=0.6, noise=0.5,
                 noise_freq=20.0, color=(1.0, 0.5, 0.04), color_amount=3.0, seed=4, embers=False),
        ],
    },
    'road_flare': {
        'name': 'Road flare', 'category': 'Small flames', 'size': '20 cm flame',
        'blurb': 'A burning road flare on the ground: a searing strontium-red flame, white smoke and red sparks.',
        'domain': {'size_x': 0.7, 'size_y': 1.4, 'size_z': 0.7, 'resolution': 128, 'preroll': 1.5},
        'combustion': {'burn_rate': 15.0, 'heat': 0.6, 'soot': 0.05, 'cooling': 4.0, 'flame_life': 0.05, 'smoke_dissipation': 0.3},
        'motion': {'buoyancy': 4.0, 'turbulence': 2.0, 'turb_freq': 10.0, 'vorticity': 1.0, 'disturbance': 1.2, 'disturb_block': 0.01,
                   'mask_smoke': 0.5},
        'shading': {'flame_k': 1900, 'colour_gain': 1.0, 'smoke_density': 4.0, 'smoke_albedo': (0.8, 0.8, 0.82), 'flame_absorption': 30.0},
        'lighting': {'light_spread': 0.12, 'fire_scatter': 2.0},
        'embers': {'rate': 60, 'launch': 1.2, 'spread': 0.8, 'lifetime': 0.8, 'temperature': 2100, 'size_max': 0.003, 'gravity': 4.0},
        'camera': {'distance': 2.4, 'target_y': 0.3, 'pitch': 8, 'anchor_y': 0.85, 'focal_mm': 35},
        'emitters': [
            dict(name='Flare head', shape='capsule', position=(-0.05, 0.02, 0), end=(0.03, 0.03, 0.0), size=(0.016, 0.016, 0.016),
                 fuel=40, temperature=0.9, smoke=3.0, noise=0.4, noise_freq=30.0, color=(1.0, 0.08, 0.12), color_amount=8.0),
        ],
    },
    'car_through_smoke': {
        'name': 'Car through smoke', 'category': 'Smoke', 'size': 'car at 30 km/h',
        'blurb': 'A car drives through a smoke column and drags it along in its wake. Keyframe a collider\'s position to move it.',
        'domain': {'size_x': 18.0, 'size_y': 8.0, 'size_z': 6.0, 'resolution': 176, 'preroll': 5.0, 'substeps_max': 8},
        'combustion': {'fuel_scale': 0.0, 'smoke_dissipation': 0.1, 'cooling': 0.8},
        'motion': {'buoyancy': 4.0, 'turbulence': 1.5, 'turb_freq': 1.2, 'vorticity': 1.5, 'mask_smoke': 1.0, 'disturbance': 0.5},
        'shading': {'smoke_density': 3.0, 'smoke_albedo': (0.45, 0.45, 0.46), 'soot_glow': 0.0},
        'lighting': {'sun_on': True, 'sun_intensity': 3.0},
        'embers': {'enabled': False},
        'camera': {'distance': 20.0, 'target_y': 2.5, 'pitch': 6, 'anchor_y': 0.88},
        'colliders': [dict(name='Car', shape='box', size=(2.2, 0.72, 0.9),
                           position=K((0.0, (-10.5, 0.72, 1.2)), (1.0, (-10.5, 0.72, 1.2)), (3.6, (10.5, 0.72, 1.2)), interp='linear'))],
        'emitters': [
            dict(name='Smoulder', shape='cylinder', position=(0, 0.25, 0), size=(0.9, 0.25, 0.9), fuel=0.0, temperature=0.8, smoke=6.0,
                 embers=False, noise_freq=2.0),
        ],
    },
    'grass_fire': {
        'name': 'Grass fire', 'category': 'Fires', 'size': 'spreading in wind',
        'blurb': 'A dropped torch lights dry grass and the wind drives the fire front across the ground. Spreading fire burns the floor by itself.',
        'render': {'end': 192},   # long enough to reach what its library picture shows
        'domain': {'size_x': 14.0, 'size_y': 4.0, 'size_z': 7.0, 'resolution': 192, 'preroll': 0.0},
        'combustion': {'burn_rate': 6.0, 'heat': 0.6, 'soot': 0.45, 'cooling': 2.4, 'flame_life': 0.08},
        'motion': {'buoyancy': 5.0, 'turbulence': 3.0, 'turb_freq': 2.0, 'vorticity': 1.6, 'disturbance': 2.5, 'disturb_block': 0.05,
                   'wind_speed': 3.0, 'wind_dir': 90.0, 'gust': 0.4},
        'spread': {'enabled': True, 'ground': True, 'area_x': 13.0, 'area_z': 6.5, 'coverage': 0.8, 'patch_freq': 1.0, 'burn_time': 1.6,
                   'fuel': 10.0, 'heat': 0.5, 'smoke': 0.8, 'catch_temp': 0.3, 'catch_time': 0.25, 'creep': 0.25, 'smoulder': 4.0,
                   'smoulder_smoke': 1.2},
        'shading': {'flame_k': 1650, 'smoke_density': 4.0, 'smoke_albedo': (0.3, 0.29, 0.27)},
        'embers': {'rate': 120},
        'camera': {'distance': 16.0, 'target_y': 0.6, 'pitch': 14, 'anchor_y': 0.75, 'focal_mm': 35},
        'emitters': [
            dict(name='Dropped torch', shape='sphere', position=(-5.5, 0.1, 0.0), size=(0.25, 0.1, 0.25), fuel=14, temperature=0.6,
                 stop=1.0, fade_out=0.3),
        ],
    },
    'curtain_fire': {
        'name': 'Curtain catching', 'category': 'Fires', 'size': '2.4 m curtain',
        'blurb': 'A small flame at the hem of a curtain climbs it and spreads across. The curtain is a collider marked Burnable.',
        'render': {'end': 168},   # long enough to reach what its library picture shows
        'domain': {'size_x': 3.0, 'size_y': 3.4, 'size_z': 1.6, 'resolution': 176, 'preroll': 0.0},
        'combustion': {'burn_rate': 6.0, 'heat': 0.6, 'soot': 0.6, 'cooling': 2.2, 'flame_life': 0.08, 'smoke_dissipation': 0.3},
        'motion': {'buoyancy': 5.0, 'turbulence': 2.5, 'turb_freq': 3.0, 'vorticity': 1.5, 'disturbance': 2.0, 'disturb_block': 0.03},
        'spread': {'enabled': True, 'ground': False, 'coverage': 1.0, 'burn_time': 5.0, 'fuel': 9.0, 'heat': 0.5, 'smoke': 1.0,
                   'catch_temp': 0.3, 'catch_time': 0.35, 'creep': 0.04, 'smoulder': 3.0},
        'shading': {'flame_k': 1650, 'smoke_density': 5.0, 'smoke_albedo': (0.15, 0.145, 0.14)},
        'embers': {'rate': 40},
        'camera': {'distance': 6.5, 'target_y': 1.4, 'pitch': 2, 'anchor_y': 0.92, 'focal_mm': 35},
        'colliders': [
            dict(name='Wall', shape='box', position=(0.0, 1.7, -0.62), size=(1.5, 1.7, 0.06)),
            dict(name='Curtain', shape='box', position=(0.0, 1.35, -0.45), size=(0.65, 1.2, 0.025), burnable=True),
        ],
        'emitters': [
            dict(name='Lighter', shape='sphere', position=(0.1, 0.2, -0.38), size=(0.04, 0.06, 0.04), fuel=20, temperature=0.7,
                 stop=2.0, fade_out=0.3, embers=False),
        ],
    },
    'fabric_curtain': {
        'name': 'Burning curtain', 'category': 'Fires', 'size': '2.4 m curtain',
        'blurb': 'A lighter at the hem of a real cotton curtain: it catches, the flames race up it, and it chars, burns through '
                 'and falls apart in about ten seconds, feeding the fire as it goes. The curtain is a Fabric: try silk, wool (it '
                 'puts itself out) or polyester (it shrinks away and melts).',
        'render': {'end': 264},   # long enough to reach what its library picture shows
        'domain': {'size_x': 3.0, 'size_y': 3.6, 'size_z': 1.8, 'resolution': 160, 'preroll': 0.0},
        'combustion': {'burn_rate': 6.0, 'heat': 0.6, 'soot': 0.6, 'cooling': 2.2, 'flame_life': 0.08, 'smoke_dissipation': 0.3},
        'motion': {'buoyancy': 5.0, 'turbulence': 2.5, 'turb_freq': 3.0, 'vorticity': 1.5, 'disturbance': 2.0, 'disturb_block': 0.03},
        'shading': {'flame_k': 1650, 'max_k': 2250, 'smoke_density': 4.0, 'smoke_albedo': (0.3, 0.28, 0.25),
                    'flame_threshold': 0.1, 'flame_sharpness': 3.2, 'flame_absorption': 10.0, 'soot_glow': 0.1, 'detail': 0.5},
        'embers': {'rate': 30},
        # a dim room lit from the side (a window), so the folds show before the fire takes over
        'lighting': {'ambient': (0.035, 0.038, 0.045), 'sun_on': True, 'sun_intensity': 0.9, 'sun_azimuth': 65.0,
                     'sun_elevation': 20.0},
        'camera': {'distance': 6.0, 'target_y': 1.5, 'pitch': 2, 'anchor_y': 0.92, 'focal_mm': 35},
        'colliders': [
            dict(name='Wall', shape='box', position=(0.0, 1.8, -0.7), size=(1.5, 1.8, 0.06)),
        ],
        'emitters': [
            dict(name='Lighter', shape='sphere', position=(0.15, 0.24, -0.5), size=(0.05, 0.07, 0.05), fuel=24, temperature=0.8,
                 start=0.2, stop=4.0, fade_out=0.4, embers=False, noise=0.3),
        ],
        'fabrics': [
            dict(name='Curtain', position=(0.0, 1.45, -0.5), width=1.3, height=2.3, fullness=2.0, pins='top', material='cotton',
                 colour=(0.72, 0.62, 0.45), detail=80),
        ],
    },
    'wet_towels': {
        'name': 'Wet and dry towels', 'category': 'Fires', 'size': '70 cm towels',
        'blurb': 'Two cotton tea towels hung over a fire on a cold, damp morning. The dry one catches and burns away in '
                 'seconds. The soaked one drips, steams and holds at 100 C while the fire boils its water off, and '
                 'outlasts it: it cannot catch until it has dried. Wet at start, on a Fabric, sets how soaked it is.',
        'domain': {'size_x': 2.0, 'size_y': 3.0, 'size_z': 1.4, 'resolution': 144, 'preroll': 1.0},
        'render': {'end': 288},
        'combustion': {'burn_rate': 5.0, 'heat': 0.6, 'soot': 0.35, 'cooling': 2.2, 'flame_life': 0.08, 'smoke_dissipation': 0.3},
        'motion': {'puffing': 0.4, 'buoyancy': 5.5, 'turbulence': 3.0, 'turb_freq': 2.5, 'vorticity': 1.6, 'disturbance': 2.0,
                   'disturb_block': 0.04},
        'shading': {'flame_k': 1650, 'max_k': 2250, 'smoke_density': 2.5, 'smoke_albedo': (0.55, 0.54, 0.53), 'flame_threshold': 0.1,
                    'flame_sharpness': 3.2, 'flame_absorption': 10.0, 'soot_glow': 0.1, 'detail': 0.5, 'ambient_k': 279.0,
                    'humidity': 85.0},
        'embers': {'rate': 25},
        'lighting': {'ambient': (0.16, 0.17, 0.2), 'sun_on': True, 'sun_intensity': 1.2, 'sun_azimuth': 55.0, 'sun_elevation': 18.0},
        'camera': {'distance': 4.0, 'target_y': 1.05, 'pitch': 4, 'anchor_y': 0.9, 'focal_mm': 35},
        'colliders': [
            dict(name='Rail', shape='box', position=(0.0, 1.415, 0.0), size=(0.62, 0.012, 0.012)),
        ],
        'emitters': [
            dict(noise_rise=1.5, name='Fire', shape='cylinder', position=(0.0, 0.05, 0.0), size=(0.36, 0.05, 0.24), fuel=10,
                 temperature=0.5, seed=5),
        ],
        'fabrics': [
            dict(name='Dry towel', position=(-0.27, 1.05, 0.0), width=0.45, height=0.7, fullness=1.25, pins='top',
                 material='cotton', colour=(0.3, 0.45, 0.66), detail=40),
            dict(name='Soaked towel', position=(0.27, 1.05, 0.0), width=0.45, height=0.7, fullness=1.25, pins='top',
                 material='cotton', colour=(0.3, 0.45, 0.66), detail=40, wetness=1.0),
        ],
    },
    'flag_wind': {
        'name': 'Flag in smoky wind', 'category': 'Smoke', 'size': '1.5 m flag',
        'blurb': 'A nylon flag on a pole streams and flutters in a gusting wind while the smoke of a fire upwind blows past it; '
                 'the flag stirs the smoke behind it. The flag is a Fabric held by one side: change its material, size or the wind.',
        'domain': {'size_x': 6.0, 'size_y': 3.6, 'size_z': 3.0, 'resolution': 144, 'preroll': 1.0},
        'combustion': {'soot': 0.9, 'smoke_dissipation': 0.1},
        'motion': {'buoyancy': 5.0, 'turbulence': 3.0, 'turb_freq': 2.0, 'disturbance': 2.0,
                   'wind_speed': 3.5, 'wind_dir': 90.0, 'gust': 0.5, 'wind_relax': 1.5},
        'shading': {'smoke_density': 2.5, 'smoke_albedo': (0.7, 0.7, 0.72)},
        'lighting': {'sun_on': True, 'sun_intensity': 3.0},
        'embers': {'enabled': False},
        'camera': {'distance': 7.0, 'target_y': 1.8, 'pitch': 4, 'anchor_y': 0.9, 'focal_mm': 35},
        'colliders': [
            dict(name='Pole', shape='cylinder', position=(-0.85, 1.1, 0.0), size=(0.025, 1.1, 0.025), holdout=False),
        ],
        'emitters': [
            dict(name='Fire', shape='cylinder', position=(-2.4, 0.05, 0.35), size=(0.45, 0.05, 0.45), fuel=14.0, temperature=0.7,
                 smoke=4.0, embers=False),
        ],
        'fabrics': [
            dict(name='Flag', position=(-0.08, 1.75, 0.0), width=1.5, height=1.0, pins='side', material='nylon',
                 colour=(0.75, 0.08, 0.06), detail=40, burnable=False),
        ],
    },
    'room_fire': {
        'name': 'Starved room fire', 'category': 'Fires', 'size': 'room, 2.6 m',
        'blurb': 'A fire in a closed room uses up its air and dies down, filling the room with hot smoke and unburnt fuel that leaks under the door. When the door slides open at 6 s, fresh air gets in and flames roll out of the doorway. Uses tracked air and a moving collider.',
        'render': {'end': 264},
        'domain': {'size_x': 6.4, 'size_y': 3.0, 'size_z': 3.2, 'resolution': 144, 'preroll': 0.0, 'substeps_max': 10, 'mg_cycles': 3},
        'combustion': {'air': 'tracked', 'air_use': 3.0, 'air_mixing': 0.05, 'thermal_expansion': 0.5, 'burn_rate': 6.0, 'rich': 8.0, 'fuel_dissipation': 0.02,
                       'heat': 0.6, 'cooling': 0.03, 'soot': 0.7, 'flame_life': 0.15, 'expansion': 0.5, 'smoke_dissipation': 0.05,
                       'flame_gain': 2.0},
        'motion': {'buoyancy': 8.0, 'turbulence': 4.0, 'turb_freq': 2.0, 'vorticity': 2.0, 'disturbance': 2.5, 'disturb_block': 0.06,
                   'mask_smoke': 0.3},
        'shading': {'flame_k': 1650, 'smoke_density': 1.5, 'smoke_albedo': (0.08, 0.075, 0.07), 'soot_glow': 0.3},
        'lighting': {'light_spread': 0.6, 'fire_scatter': 1.5},
        'embers': {'enabled': False},
        'camera': {'distance': 8.0, 'target_y': 1.2, 'yaw': 40, 'pitch': 4, 'anchor_x': 0.45, 'anchor_y': 0.85, 'focal_mm': 35},
        'colliders': [
            # one hollow box with a doorway cut out of its front wall, and a door that slides aside at 6 s;
            # they do not hide the fire so the demo shows inside (turn Hides fire on to match a real room)
            dict(name='Room', shape='box', position=(-1.1, 1.2, 0.0), size=(1.5, 1.2, 1.5), hollow=0.2,
                 opening=(0.3, 0.87, 0.46), opening_at=(1.4, -0.33, 0.0), holdout=False),
            dict(name='Door', shape='box', size=(0.05, 0.87, 0.47), holdout=False,
                 position=K((0.0, (0.3, 0.97, 0.0)), (6.0, (0.3, 0.97, 0.0)), (6.5, (0.3, 0.97, -0.92)), interp='linear')),
        ],
        'emitters': [
            dict(name='Sofa fire', shape='box', position=(-1.4, 0.25, 0.0), size=(0.6, 0.15, 0.6), fuel=20, temperature=0.6, noise_freq=2.0),
        ],
    },
    'kettle_steam': {
        'name': 'Kettle steam', 'category': 'Steam', 'size': 'spout plume',
        'blurb': 'Steam from a kettle spout: clear right at the spout, clouding over as it cools, then evaporating as it mixes into the room.',
        'domain': {'size_x': 0.6, 'size_y': 0.9, 'size_z': 0.5, 'resolution': 144, 'ground': False, 'preroll': 2.0},
        'combustion': {'fuel_scale': 0.0, 'cooling': 2.5, 'vapour_dissipation': 2.5, 'smoke_dissipation': 1.0},
        'motion': {'buoyancy': 35.0, 'turbulence': 1.5, 'turb_freq': 20.0, 'turb_rise': 0.3, 'vorticity': 1.0, 'disturbance': 0.8,
                   'disturb_block': 0.006, 'mask_temp': 20.0},
        'shading': {'ambient_k': 293.0, 'humidity': 45.0, 'steam_density': 0.8, 'detail': 0.3, 'detail_freq': 25.0},
        'lighting': {'ambient_intensity': 3.0, 'sun_on': True, 'sun_intensity': 3.0, 'sun_azimuth': 60, 'sun_elevation': 30},
        'embers': {'enabled': False},
        'camera': {'distance': 1.4, 'target_y': 0.25, 'pitch': 2, 'anchor_x': 0.4, 'anchor_y': 0.85, 'focal_mm': 50},
        'emitters': [
            dict(name='Spout', shape='sphere', position=(-0.15, 0.05, 0.0), size=(0.01, 0.01, 0.01), fuel=0.0, temperature=0.055,
                 vapour=590.0, velocity=(0.5, 1.2, 0.0), vel_blend=0.8, noise=0.2, embers=False),
        ],
    },
    'steam_vent': {
        'name': 'Steam vent', 'category': 'Steam', 'size': '6 m plume, cold day',
        'blurb': 'A pipe venting steam into freezing air: a dense white column that billows and thins as it rises.',
        'domain': {'size_x': 3.5, 'size_y': 9.0, 'size_z': 3.5, 'resolution': 144, 'preroll': 3.0},
        'combustion': {'fuel_scale': 0.0, 'cooling': 1.0, 'vapour_dissipation': 1.0, 'smoke_dissipation': 1.0},
        'motion': {'buoyancy': 25.0, 'turbulence': 2.0, 'turb_freq': 1.5, 'turb_rise': 2.0, 'vorticity': 1.5, 'disturbance': 1.0,
                   'disturb_block': 0.05, 'mask_temp': 10.0, 'wind_speed': 1.0, 'gust': 0.5},
        'shading': {'ambient_k': 272.0, 'humidity': 80.0, 'steam_density': 0.3, 'detail': 0.4, 'detail_freq': 2.0},
        'lighting': {'ambient_intensity': 2.0, 'sun_on': True, 'sun_intensity': 3.0, 'sun_elevation': 25},
        'embers': {'enabled': False},
        'camera': {'distance': 12.0, 'target_y': 3.5, 'pitch': 3, 'anchor_y': 0.93},
        'emitters': [
            dict(name='Vent', shape='cylinder', position=(0, 0.3, 0), size=(0.15, 0.1, 0.15), fuel=0.0, temperature=0.1, vapour=550.0,
                 velocity=(0.0, 6.0, 0.0), vel_blend=1.0, noise=0.3, embers=False),
        ],
    },
    'armchair_fire': {
        'name': 'Burning armchair', 'category': 'Fires', 'size': 'mesh, 1 m',
        'blurb': 'Fire spreading over an armchair from a flame on its seat. The chair is a mesh (OBJ) collider marked Burnable: use your own model the same way.',
        'render': {'end': 336},   # long enough to reach what its library picture shows
        'domain': {'size_x': 2.4, 'size_y': 3.6, 'size_z': 2.4, 'resolution': 160, 'preroll': 0.0},
        'combustion': {'burn_rate': 5.0, 'heat': 0.6, 'soot': 1.1, 'cooling': 2.0, 'flame_life': 0.1, 'smoke_dissipation': 0.25},
        'motion': {'buoyancy': 5.5, 'turbulence': 3.0, 'turb_freq': 3.0, 'vorticity': 1.6, 'disturbance': 2.0, 'disturb_block': 0.03},
        'spread': {'enabled': True, 'ground': False, 'coverage': 1.0, 'burn_time': 12.0, 'fuel': 8.0, 'heat': 0.5, 'smoke': 1.5,
                   'catch_temp': 0.3, 'catch_time': 0.5, 'creep': 0.03, 'smoulder': 6.0},
        'shading': {'flame_k': 1600, 'smoke_density': 6.0, 'smoke_albedo': (0.05, 0.05, 0.05)},
        'embers': {'rate': 30},
        'camera': {'distance': 6.0, 'target_y': 1.2, 'yaw': 25, 'pitch': 6, 'anchor_y': 0.9, 'focal_mm': 35},
        'colliders': [dict(name='Armchair', shape='mesh', mesh='builtin:armchair.obj', position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0),
                           burnable=True)],
        'emitters': [
            dict(name='Match', shape='sphere', position=(0.1, 0.44, 0.1), size=(0.05, 0.05, 0.05), fuel=18, temperature=0.7,
                 stop=2.5, fade_out=0.5, embers=False),
        ],
    },
    'flash_fire': {
        'name': 'Fuel spill flash fire', 'category': 'Fires', 'size': '3 m of spilled petrol',
        'blurb': 'Petrol vapour from a spill creeps along the ground, heavier than air. A spark at 3 s lights its edge and a flame front runs back across the vapour to the puddle, which keeps burning. Uses Flame speed and Fuel vapour weight.',
        'render': {'end': 168},
        'domain': {'size_x': 4.4, 'size_y': 2.6, 'size_z': 2.2, 'resolution': 176, 'preroll': 0.0, 'substeps_max': 8},
        'combustion': {'flame_speed': 4.0, 'fuel_weight': 1.0, 'fuel_dissipation': 0.02, 'burn_rate': 6.0, 'heat': 0.6, 'soot': 0.8,
                       'cooling': 2.0, 'flame_life': 0.1, 'expansion': 1.0, 'rich': 4.0},
        'motion': {'buoyancy': 5.5, 'turbulence': 3.0, 'turb_freq': 3.0, 'vorticity': 1.6, 'disturbance': 2.0, 'disturb_block': 0.03},
        'shading': {'flame_k': 1700, 'smoke_density': 5.0, 'smoke_albedo': (0.06, 0.06, 0.06)},
        'embers': {'enabled': False},
        'camera': {'distance': 6.0, 'target_y': 0.5, 'yaw': 10, 'pitch': 12, 'anchor_y': 0.8, 'focal_mm': 35},
        'emitters': [
            dict(name='Petrol spill', shape='box', position=(-1.2, 0.02, 0.0), size=(0.5, 0.02, 0.45), fuel=4.0, temperature=0.0,
                 noise=0.3, noise_freq=4.0, embers=False),
            dict(name='Spark', shape='sphere', position=(-0.4, 0.05, 0.1), size=(0.05, 0.05, 0.05), fuel=0.0, temperature=1.5,
                 start=3.0, stop=3.5, fade_in=0.05, fade_out=0.1, embers=False),
        ],
    },
    'gas_cloud': {
        'name': 'Gas cloud ignition', 'category': 'Explosions', 'size': '5 m cloud',
        'blurb': 'A burst of propane spreads into a low cloud that drifts onto a pilot flame at about 2 s: a flame front burns through the whole cloud in half a second and it swells into a fireball. Uses Flame speed, Fuel vapour weight and Heat expansion.',
        'render': {'end': 96},
        'domain': {'size_x': 7.0, 'size_y': 6.0, 'size_z': 5.0, 'resolution': 176, 'preroll': 0.0, 'substeps_max': 10, 'mg_cycles': 3},
        'combustion': {'flame_speed': 6.0, 'fuel_weight': 0.6, 'fuel_dissipation': 0.05, 'burn_rate': 8.0, 'heat': 0.9, 'soot': 0.9,
                       'cooling': 1.6, 'flame_life': 0.2, 'expansion': 2.0, 'thermal_expansion': 0.8, 'rich': 6.0, 'flame_gain': 1.5},
        'motion': {'buoyancy': 6.0, 'turbulence': 5.0, 'turb_freq': 1.2, 'vorticity': 2.0, 'disturbance': 3.0, 'disturb_block': 0.08},
        'shading': {'flame_k': 1800, 'max_k': 2300, 'smoke_density': 3.0, 'smoke_albedo': (0.07, 0.065, 0.06)},
        'embers': {'enabled': False},
        'camera': {'distance': 13.0, 'target_y': 1.6, 'pitch': 6, 'anchor_y': 0.85, 'focal_mm': 35},
        'emitters': [
            dict(name='Leak', shape='sphere', position=(-0.8, 0.3, 0.0), size=(0.4, 0.3, 0.4), fuel=14.0, temperature=0.0, radial=3.0,
                 vel_blend=0.4, stop=1.0, fade_out=0.3, embers=False),
            dict(name='Pilot flame', shape='sphere', position=(0.4, 0.15, 0.2), size=(0.06, 0.06, 0.06), fuel=0.0, temperature=1.5,
                 start=1.2, stop=2.5, fade_in=0.05, fade_out=0.1, embers=False),
        ],
    },
    'backdraft': {
        'name': 'Backdraft', 'category': 'Fires', 'size': 'small room, 1.6 m',
        'blurb': 'A fire in a shut room uses up the air and dies back, filling the room with hot, fuel-rich smoke. When the door slides open at 4 s the smoke pours out of the doorway, meets fresh air and ignites, and the flames roll back in along the air coming in. Soot blackens the ceiling and the door frame. Uses tracked air, Flame speed and Soot stains.',
        'render': {'end': 192},
        'domain': {'size_x': 5.0, 'size_y': 2.6, 'size_z': 2.4, 'resolution': 128, 'preroll': 0.0, 'substeps_max': 10, 'mg_cycles': 3},
        'combustion': {'air': 'tracked', 'air_use': 2.5, 'air_mixing': 0.6, 'thermal_expansion': 0.4, 'flame_speed': 4.0,
                       'burn_rate': 6.0, 'rich': 8.0, 'fuel_dissipation': 0.01, 'heat': 0.7, 'cooling': 0.05, 'soot': 0.8,
                       'flame_life': 0.18, 'expansion': 0.3, 'smoke_dissipation': 0.05, 'flame_gain': 2.0, 'soot_stain': 0.4},
        'motion': {'buoyancy': 8.0, 'turbulence': 4.0, 'turb_freq': 2.5, 'vorticity': 2.0, 'disturbance': 2.5, 'disturb_block': 0.05,
                   'mask_smoke': 0.3},
        'shading': {'flame_k': 1700, 'smoke_density': 1.5, 'smoke_albedo': (0.08, 0.075, 0.07), 'soot_glow': 0.3},
        'lighting': {'light_spread': 0.5, 'fire_scatter': 1.5},
        'embers': {'enabled': False},
        'camera': {'distance': 6.0, 'target_y': 0.8, 'yaw': 45, 'pitch': 5, 'anchor_x': 0.45, 'anchor_y': 0.8, 'focal_mm': 35},
        'colliders': [
            # a hollow room with a doorway in its front wall (+x), and a door that slides aside at 4 s; a
            # 4 cm gap under the door lets the room breathe a little, as real rooms do. They do not hide
            # the fire, so the demo shows inside (turn Hides fire on to match a real room)
            dict(name='Room', shape='box', position=(-1.0, 0.8, 0.0), size=(0.9, 0.8, 0.9), hollow=0.15,
                 opening=(0.3, 0.7, 0.35), opening_at=(0.825, -0.1, 0.0), holdout=False),
            dict(name='Door', shape='box', size=(0.04, 0.7, 0.36), holdout=False,
                 position=K((0.0, (-0.055, 0.74, 0.0)), (4.0, (-0.055, 0.74, 0.0)), (4.25, (-0.055, 0.74, -0.75)), interp='linear')),
        ],
        'emitters': [
            dict(name='Burning furniture', shape='box', position=(-1.3, 0.2, 0.0), size=(0.35, 0.12, 0.35), fuel=30, temperature=0.6,
                 noise_freq=2.0),
        ],
    },
    'hillside_fire': {
        'name': 'Fire running uphill', 'category': 'Fires', 'size': '8 m slope',
        'blurb': 'A fire lit at the foot of a grassy slope runs up it, driven by the breeze that climbs the hill and by its flames leaning onto the unburnt grass above. The slope is a heightfield: a greyscale image used as a Burnable mesh collider.',
        'render': {'end': 216},
        'domain': {'size_x': 9.0, 'size_y': 6.0, 'size_z': 9.0, 'resolution': 176, 'preroll': 0.0, 'mesh_resolution': 128},
        'combustion': {'burn_rate': 6.0, 'heat': 0.6, 'soot': 0.45, 'cooling': 2.4, 'flame_life': 0.08},
        'motion': {'buoyancy': 5.0, 'turbulence': 3.0, 'turb_freq': 2.0, 'vorticity': 1.6, 'disturbance': 2.5, 'disturb_block': 0.05,
                   'wind_speed': 2.0, 'wind_dir': 180.0, 'gust': 0.3},
        'spread': {'enabled': True, 'ground': False, 'coverage': 0.85, 'patch_freq': 1.2, 'burn_time': 2.0, 'fuel': 10.0, 'heat': 0.5,
                   'smoke': 0.8, 'catch_temp': 0.25, 'catch_time': 0.3, 'creep': 0.12, 'smoulder': 4.0, 'smoulder_smoke': 1.2},
        'shading': {'flame_k': 1650, 'smoke_density': 4.0, 'smoke_albedo': (0.3, 0.29, 0.27)},
        'embers': {'rate': 80},
        'camera': {'distance': 15.0, 'target_y': 1.6, 'yaw': 20, 'pitch': 12, 'anchor_y': 0.8, 'focal_mm': 35},
        'colliders': [dict(name='Hillside', shape='mesh', mesh='builtin:hillside.png', position=(0.0, 0.0, 0.0),
                           size=(8.0, 3.2, 8.0), burnable=True)],
        'emitters': [
            dict(name='Dropped torch', shape='sphere', position=(-0.5, 0.25, 3.4), size=(0.3, 0.15, 0.3), fuel=14, temperature=0.6,
                 stop=1.5, fade_out=0.3),
        ],
    },
    'spot_fires': {
        'name': 'Spot fires', 'category': 'Fires', 'size': 'wind-driven embers',
        'blurb': 'A strong wind throws burning embers ahead of a grass fire; where hot ones land they start new fires, which the wind then joins up. Spreading fire › Spot fires.',
        'render': {'end': 216},
        'domain': {'size_x': 16.0, 'size_y': 5.0, 'size_z': 8.0, 'resolution': 192, 'preroll': 0.0},
        'combustion': {'burn_rate': 6.0, 'heat': 0.6, 'soot': 0.45, 'cooling': 2.4, 'flame_life': 0.08},
        'motion': {'buoyancy': 5.0, 'turbulence': 3.0, 'turb_freq': 2.0, 'vorticity': 1.6, 'disturbance': 2.5, 'disturb_block': 0.05,
                   'wind_speed': 6.0, 'wind_dir': 90.0, 'gust': 0.5},
        'spread': {'enabled': True, 'ground': True, 'area_x': 15.0, 'area_z': 7.5, 'coverage': 0.8, 'patch_freq': 1.0, 'burn_time': 1.6,
                   'fuel': 10.0, 'heat': 0.5, 'smoke': 0.8, 'catch_temp': 0.3, 'catch_time': 0.25, 'creep': 0.1, 'smoulder': 4.0,
                   'smoulder_smoke': 1.2, 'spotting': 0.06, 'spot_temp': 800.0},
        'shading': {'flame_k': 1650, 'smoke_density': 4.0, 'smoke_albedo': (0.3, 0.29, 0.27)},
        'embers': {'rate': 400, 'launch': 3.0, 'lifetime': 4.0, 'cooling': 0.25},
        'camera': {'distance': 18.0, 'target_y': 0.6, 'pitch': 16, 'anchor_y': 0.75, 'focal_mm': 35},
        'emitters': [
            dict(name='Dropped torch', shape='sphere', position=(-6.5, 0.1, 0.0), size=(0.3, 0.1, 0.3), fuel=14, temperature=0.6,
                 stop=1.0, fade_out=0.3),
        ],
    },
    # -- things that fall (engine/solids.py) ----------------------------------------------------------------------------
    'tower_knockdown': {
        'name': 'Knocking down a tower', 'category': 'Things that fall', 'size': '1.2 m tower',
        'blurb': 'A bowling ball thrown into a tower of wooden blocks beside a run of dominoes, on the stage in sunlight: '
                 'the blocks tumble and bounce, the dominoes fall in turn. Objects with Falls on, drawn in their materials.',
        'render': {'end': 96},
        'domain': {'size_x': 5.2, 'size_y': 2.4, 'size_z': 2.6, 'resolution': 64, 'preroll': 0.0},
        'composite': {'backdrop': 'stage', 'floor': 'boards'},
        'lighting': {'sun_on': True, 'sun_intensity': 3.0, 'sun_elevation': 35.0, 'sun_azimuth': -40.0, 'ambient_intensity': 2.5},
        'camera': {'distance': 5.2, 'target_y': 0.45, 'pitch': 14, 'yaw': 20, 'focal_mm': 35},
        'emitters': [],
        'colliders': [dict(name=f'Block {k + 1}', shape='box', position=(0.6, 0.1 + 0.2 * k, 0.0), size=(0.1, 0.1, 0.1),
                           yaw=(7.0 if k % 2 else -4.0), dynamic=True, material='wood') for k in range(6)]
                     + [dict(name='Bowling ball', shape='sphere', position=(-2.2, 0.9, 0.0), size=(0.109, 0.109, 0.109), dynamic=True,
                             material='plastic', density=1300.0, own_colour=True, colour=(0.02, 0.05, 0.25),
                             start_velocity=(5.5, 2.0, 0.0))]
                     + [dict(name=f'Domino {k + 1}', shape='box', position=(-1.1 + 0.16 * k, 0.12, 0.9), size=(0.02, 0.12, 0.06),
                             dynamic=True, material='wood',
                             **({'start_spin': (0.0, 0.0, -160.0), 'start_velocity': (0.34, 0.06, 0.0), 'release': 0.6} if k == 0 else {}))
                        for k in range(9)],
    },
    'wall_smash': {
        'name': 'Ball through a brick wall', 'category': 'Things that fall', 'size': '1.6 m wall',
        'blurb': 'A 260 kg steel ball into a brick wall: it punches through, the mortar gives way brick by brick, the '
                 'wall above it caves in and the dust rolls out. A breakable box in Bricks.',
        'render': {'end': 72},
        'domain': {'size_x': 4.0, 'size_y': 2.4, 'size_z': 3.4, 'resolution': 96, 'preroll': 0.0},
        'shading': {'smoke_albedo': (0.62, 0.57, 0.5), 'smoke_density': 0.8},
        'motion': {'buoyancy': 0.5, 'turbulence': 1.5, 'vorticity': 1.0},
        'composite': {'backdrop': 'stage', 'floor': 'concrete'},
        'lighting': {'sun_on': True, 'sun_intensity': 3.0, 'sun_elevation': 35.0, 'sun_azimuth': -140.0, 'ambient_intensity': 2.5},
        'camera': {'distance': 4.6, 'target_y': 0.6, 'pitch': 10, 'yaw': 200, 'focal_mm': 35},
        'emitters': [],
        'colliders': [
            dict(name='Brick wall', shape='box', position=(0.0, 0.6, 0.0), size=(0.8, 0.6, 0.05), material='brick',
                 breakable=True, fracture='bricks'),
            dict(name='Steel ball', shape='sphere', position=(0.0, 0.55, -1.4), size=(0.2, 0.2, 0.2), dynamic=True,
                 material='steel', start_velocity=(0.0, 0.6, 5.0), release=0.25),
        ],
    },
    'chain_swing': {
        'name': 'Chain and rope', 'category': 'Things that fall', 'size': '1.4 m chain',
        'blurb': 'A steel weight on a 12 mm chain swings down from a beam into a stack of crates and knocks it over, the '
                 'chain going slack and snapping taut as it goes, while a crate on a rope thrown over the same beam '
                 'drops, is caught and swings under it. Objects on a rope (Properties › Joint): Rope is Chain, and a '
                 'rope over a post.',
        'render': {'end': 120},
        'domain': {'size_x': 5.0, 'size_y': 3.6, 'size_z': 3.0, 'resolution': 48, 'preroll': 0.0},
        'composite': {'backdrop': 'stage', 'floor': 'boards'},
        'lighting': {'sun_on': True, 'sun_intensity': 3.0, 'sun_elevation': 40.0, 'sun_azimuth': -130.0, 'ambient_intensity': 2.5},
        'camera': {'distance': 6.2, 'target_y': 1.3, 'pitch': 6, 'yaw': -12, 'focal_mm': 35},
        'emitters': [],
        'colliders': [
            dict(name='Post', shape='box', position=(-0.1, 1.2, -1.05), size=(0.06, 1.2, 0.06), material='wood'),
            dict(name='Post 2', shape='box', position=(-0.1, 1.2, 1.05), size=(0.06, 1.2, 0.06), material='wood'),
            dict(name='Beam', shape='cylinder', position=(-0.1, 2.46, 0.0), size=(0.07, 1.12, 0.07), pitch=90.0,
                 material='wood'),
            dict(name='Weight', shape='sphere', position=(-1.45, 2.2, -0.45), size=(0.14, 0.14, 0.14), dynamic=True,
                 material='steel', joint='rope', rope_look='chain', rope_thickness=0.012, joint_anchor=(-0.1, 2.38, -0.45),
                 rope_length=1.45, release=0.3),
            *[dict(name=f'Crate {k + 1}', shape='box', position=(0.75 + dx, 0.15 + 0.3 * k, -0.45), size=(0.15, 0.15, 0.15),
                   yaw=yaw, dynamic=True, material='wood', density=300.0, friction=0.7, bounce=0.05)
              for k, (dx, yaw) in enumerate(((0.0, 0.0), (0.03, 12.0), (-0.02, -8.0), (0.02, 20.0)))],
            dict(name='Hanging crate', shape='box', position=(-0.75, 1.5, 0.5), size=(0.18, 0.18, 0.18), dynamic=True,
                 material='wood', joint='rope', rope_look='rope', rope_thickness=0.02, joint_anchor=(0.55, 0.0, 0.5),
                 rope_length=4.1, release=0.6),
        ],
    },
    'wrecking_ball': {
        'name': 'Wrecking ball', 'category': 'Things that fall', 'size': '2 m wall',
        'blurb': 'A 900 kg wrecking ball on a crane\N{RIGHT SINGLE QUOTATION MARK}s cable swings down into a brick wall at 6 m/s '
                 'and bursts through it in a cloud of dust. An object joined to the crane by a rope (Properties › Joint) '
                 'and a breakable box in Bricks.',
        'render': {'end': 96},
        'domain': {'size_x': 8.0, 'size_y': 6.0, 'size_z': 4.4, 'resolution': 96, 'preroll': 0.0},
        'shading': {'smoke_albedo': (0.62, 0.57, 0.5), 'smoke_density': 0.8},
        'motion': {'buoyancy': 0.5, 'turbulence': 1.5, 'vorticity': 1.0},
        'composite': {'backdrop': 'stage', 'floor': 'dirt'},
        'lighting': {'sun_on': True, 'sun_intensity': 3.0, 'sun_elevation': 35.0, 'sun_azimuth': -130.0, 'ambient_intensity': 2.5},
        'camera': {'distance': 11.5, 'target_y': 2.5, 'pitch': 5, 'yaw': 25, 'focal_mm': 35},
        'emitters': [],
        'colliders': [
            dict(name='Brick wall', shape='box', position=(0.62, 0.75, 0.0), size=(1.0, 0.75, 0.06), yaw=90.0, material='brick',
                 breakable=True, fracture='bricks'),
            dict(name='Crane mast', shape='box', position=(0.0, 2.7, -1.6), size=(0.12, 2.7, 0.12), material='steel',
                 own_colour=True, colour=(0.75, 0.5, 0.04)),
            dict(name='Crane jib', shape='box', position=(0.0, 5.45, -0.75), size=(0.1, 0.1, 0.95), material='steel',
                 own_colour=True, colour=(0.75, 0.5, 0.04)),
            dict(name='Wrecking ball', shape='sphere', position=(-3.727, 2.740, 0.0), size=(0.3, 0.3, 0.3), material='steel',
                 joint='rope', joint_to='Crane jib', joint_to_at=(0.0, -0.1, 0.75), rope_length=4.25, rope_look='cable',
                 rope_thickness=0.03, release=0.3),
        ],
    },
    'sand_hopper': {
        'name': 'Sand from a hopper', 'category': 'Sand, snow and mud', 'size': '40 cm hopper',
        'blurb': 'Forty litres of dry sand run out through a hole in the bottom of a hopper on legs and build a heap at its '
                 'angle of repose round the legs. Matter in Sand, filling a hollow box with an opening in its floor.',
        'render': {'end': 144},
        'domain': {'size_x': 1.6, 'size_y': 1.5, 'size_z': 1.6, 'resolution': 32, 'preroll': 0.0, 'matter_detail': 128},
        'composite': {'backdrop': 'stage', 'floor': 'concrete'},
        'lighting': {'sun_on': True, 'sun_intensity': 3.0, 'sun_elevation': 40.0, 'sun_azimuth': -130.0, 'ambient_intensity': 2.5},
        'camera': {'distance': 3.1, 'target_y': 0.65, 'pitch': 10, 'yaw': 25, 'focal_mm': 35},
        'emitters': [],
        'colliders': [
            dict(name='Hopper', shape='box', position=(0.0, 1.05, 0.0), size=(0.2, 0.2, 0.2), hollow=0.015,
                 opening=(0.06, 0.03, 0.06), opening_at=(0.0, -0.2, 0.0), material='painted', own_colour=True,
                 colour=(0.15, 0.3, 0.45)),
        ] + [dict(name=f'Leg {k + 1}', shape='box', position=(sx * 0.18, 0.43, sz * 0.18), size=(0.015, 0.43, 0.015),
                  material='steel') for k, (sx, sz) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1)))],
        'matter': [dict(name='Sand', material='sand', shape='box', position=(0.0, 1.03, 0.0), size=(0.17, 0.16, 0.17))],
    },
    'snowballs': {
        'name': 'Snowballs at a wall', 'category': 'Sand, snow and mud', 'size': '18 cm snowballs',
        'blurb': 'Three snowballs of packing snow thrown one after another at a brick wall: each squashes where it hits '
                 'and sticks to the bricks. Matter in Packing snow, thrown.',
        'render': {'end': 60},
        'domain': {'size_x': 3.0, 'size_y': 1.8, 'size_z': 1.8, 'resolution': 32, 'preroll': 0.0, 'matter_detail': 160},
        'composite': {'backdrop': 'stage', 'floor': 'concrete'},
        'lighting': {'sun_on': True, 'sun_intensity': 2.5, 'sun_elevation': 30.0, 'sun_azimuth': -60.0, 'ambient_intensity': 2.0},
        'camera': {'distance': 3.0, 'target_y': 0.7, 'pitch': 6, 'yaw': -50, 'focal_mm': 35},
        'emitters': [],
        'colliders': [dict(name='Brick wall', shape='box', position=(0.6, 0.8, 0.0), size=(0.06, 0.8, 0.8), material='brick')],
        'matter': [dict(name=f'Snowball {k + 1}', material='packing_snow', shape='sphere', position=(-1.25, y, z),
                        size=(0.09, 0.09, 0.09), velocity=(7.0, vy, -0.4 * z), release=t)
                   for k, (y, z, vy, t) in enumerate(((1.0, 0.0, 1.3, 0.0), (1.15, 0.35, 0.9, 0.5), (0.9, -0.3, 1.7, 1.0)))],
    },
    'jelly_ball': {
        'name': 'Ball dropped on jelly', 'category': 'Sand, snow and mud', 'size': '34 cm block',
        'blurb': 'A 7 kg steel ball dropped onto a block of jelly: the jelly squashes deep, throws the ball back up and '
                 'wobbles. Matter in Jelly and an object that falls, pushing each other.',
        'render': {'end': 72},
        'domain': {'size_x': 1.4, 'size_y': 1.4, 'size_z': 1.4, 'resolution': 32, 'preroll': 0.0, 'matter_detail': 128},
        'composite': {'backdrop': 'stage', 'floor': 'tiles'},
        'lighting': {'sun_on': True, 'sun_intensity': 3.0, 'sun_elevation': 45.0, 'sun_azimuth': -130.0, 'ambient_intensity': 2.5},
        'camera': {'distance': 1.8, 'target_y': 0.3, 'pitch': 14, 'yaw': 25, 'focal_mm': 35},
        'emitters': [],
        'colliders': [dict(name='Steel ball', shape='sphere', position=(0.03, 0.8, 0.0), size=(0.06, 0.06, 0.06), dynamic=True,
                           material='steel')],
        'matter': [dict(name='Jelly', material='jelly', shape='box', position=(0.0, 0.08, 0.0), size=(0.17, 0.08, 0.17))],
    },
    'mud_drag': {
        'name': 'Crate through mud', 'category': 'Sand, snow and mud', 'size': '1.6 m of mud',
        'blurb': 'A crate dragged through a bed of thick mud: it ploughs a trench, pushes up a bow wave that slumps back, and '
                 'leaves ridges. Matter in Mud, and an object moved by its keys.',
        'render': {'end': 72},
        'domain': {'size_x': 2.2, 'size_y': 1.0, 'size_z': 1.2, 'resolution': 32, 'preroll': 0.0, 'matter_detail': 160},
        'composite': {'backdrop': 'stage', 'floor': 'dirt'},
        'lighting': {'sun_on': True, 'sun_intensity': 3.0, 'sun_elevation': 35.0, 'sun_azimuth': -120.0, 'ambient_intensity': 2.5},
        'camera': {'distance': 2.4, 'target_y': 0.15, 'pitch': 22, 'yaw': 30, 'focal_mm': 35},
        'emitters': [],
        'colliders': [dict(name='Crate', shape='box', size=(0.12, 0.12, 0.12), material='wood',
                           position=K((0.0, (-0.95, 0.1, 0.0)), (0.3, (-0.95, 0.1, 0.0)), (2.7, (0.95, 0.1, 0.0)), interp='linear'))],
        'matter': [dict(name='Mud', material='mud', shape='box', position=(0.0, 0.05, 0.0), size=(0.8, 0.05, 0.45))],
    },
    'sand_castle': {
        'name': 'Sand castle and a wave', 'category': 'Sand, snow and mud', 'size': '32 cm castle',
        'blurb': 'A wall of water let go at a castle of damp sand: the wave breaks over it and the castle stands, then the '
                 'water soaks into it, undermines it and slumps it into a mound that the sloshing water carries. Matter in '
                 'Wet sand and a block of water in a closed box.',
        'render': {'end': 192},
        'domain': {'kind': 'liquid', 'size_x': 2.4, 'size_y': 1.0, 'size_z': 1.4, 'resolution': 144, 'preroll': 0.0,
                   'open_sides': False, 'matter_detail': 160, 'substeps_max': 12, 'cfl': 1.5},
        'water': {'backdrop': 6.0},
        'composite': {'backdrop': 'stage', 'floor': 'tiles'},
        'lighting': {'sun_on': True, 'sun_intensity': 3.0, 'sun_elevation': 40.0, 'sun_azimuth': -130.0, 'ambient_intensity': 2.5},
        'camera': {'distance': 1.7, 'target_y': 0.14, 'pitch': 18, 'yaw': -35, 'focal_mm': 35},
        'emitters': [dict(name='Water', shape='box', position=(-0.9, 0.3, 0.0), size=(0.25, 0.3, 0.65), liquid_mode='fill',
                          vel_blend=0.0, embers=False, noise=0.0, start=0.0)],
        'matter': [dict(name='Castle', material='wet_sand', shape='box', position=(0.0, 0.07, 0.0), size=(0.16, 0.07, 0.16)),
                   dict(name='Keep', material='wet_sand', shape='cylinder', position=(0.0, 0.24, 0.0), size=(0.065, 0.1, 0.065))]
                  + [dict(name=f'Tower {k + 1}', material='wet_sand', shape='cylinder', position=(sx * 0.115, 0.215, sz * 0.115),
                          size=(0.04, 0.075, 0.04)) for k, (sx, sz) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1)))],
    },
    'iron_pour': {
        'name': 'Pouring molten iron', 'category': 'Sand, snow and mud', 'size': '3 litres of iron',
        'blurb': 'Molten iron at 1320 °C poured from a ladle into a mould: it glows orange and lights the floor round it, '
                 'fills the mould, and dims to red as its skin cools and sets. Matter in Molten iron, poured.',
        'render': {'end': 192},
        'domain': {'size_x': 1.2, 'size_y': 0.9, 'size_z': 0.8, 'resolution': 32, 'preroll': 0.0, 'matter_detail': 192},
        'composite': {'backdrop': 'stage', 'floor': 'concrete'},
        'lighting': {'sun_on': True, 'sun_intensity': 0.4, 'sun_elevation': 25.0, 'sun_azimuth': -60.0, 'ambient_intensity': 0.4},
        'camera': {'distance': 1.3, 'target_y': 0.2, 'pitch': 14, 'yaw': 20, 'focal_mm': 35},
        'emitters': [],
        'colliders': [
            dict(name='Ladle', shape='cylinder', position=(-0.2, 0.42, 0.0), size=(0.07, 0.08, 0.07), hollow=0.01,
                 opening=(0.06, 0.02, 0.06), opening_at=(0.0, 0.08, 0.0), pitch=60.0, yaw=90.0, material='steel'),
            dict(name='Mould', shape='box', position=(0.08, 0.08, 0.0), size=(0.18, 0.08, 0.14), hollow=0.03,
                 opening=(0.15, 0.045, 0.11), opening_at=(0.0, 0.08, 0.0), material='earth'),
        ],
        'matter': [dict(name='Molten iron', material='molten_iron', pours=True, position=(-0.13, 0.46, 0.0),
                        size=(0.014, 0.014, 0.014), velocity=(0.45, 0.0, 0.0), rate=0.6, pour_start=0.2, pour_stop=3.7)],
    },
    'sheet_rip': {
        'name': 'Ball through a sheet', 'category': 'Things that fall', 'size': '2 m sheet',
        'blurb': 'A crate dropped onto a cotton sheet laced into a frame lands in it and rests, the sheet sagging under it. '
                 'Then a steel ball dropped beside it rips a ragged hole: the crate tips in after it, both fall through, '
                 'and the torn flaps hang down. Fabric held by all its edges, with Tears on; objects with Falls on.',
        'render': {'end': 120},
        'domain': {'size_x': 4.0, 'size_y': 4.4, 'size_z': 4.0, 'resolution': 48, 'preroll': 0.0},
        'composite': {'backdrop': 'stage', 'floor': 'concrete'},
        'lighting': {'sun_on': True, 'sun_intensity': 3.0, 'sun_elevation': 45.0, 'sun_azimuth': -130.0, 'ambient_intensity': 2.5},
        'camera': {'distance': 4.5, 'target_y': 1.15, 'pitch': 27, 'yaw': 30, 'focal_mm': 35},
        'emitters': [],
        'colliders': [
            *[dict(name=f'Post {k + 1}', shape='box', position=(sx * 1.08, 0.62, sz * 1.08), size=(0.05, 0.62, 0.05),
                   material='wood') for k, (sx, sz) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1)))],
            *[dict(name=f'Rail {k + 1}', shape='box', position=(sx * 1.06, 1.2, 0.0), size=(0.04, 0.04, 1.04), material='wood')
              for k, sx in enumerate((-1, 1))],
            *[dict(name=f'Rail {k + 3}', shape='box', position=(0.0, 1.2, sz * 1.06), size=(1.04, 0.04, 0.04), material='wood')
              for k, sz in enumerate((-1, 1))],
            dict(name='Crate', shape='box', position=(-0.38, 1.7, 0.22), size=(0.18, 0.18, 0.18), yaw=25.0, dynamic=True,
                 material='wood', density=300.0, friction=0.8, bounce=0.05, release=0.2),
            dict(name='Steel ball', shape='sphere', position=(0.12, 2.7, -0.08), size=(0.24, 0.24, 0.24), dynamic=True,
                 material='steel', release=1.6),
        ],
        'fabrics': [dict(name='Sheet', position=(0.0, 1.2, 0.0), width=2.0, height=2.0, orientation='lying', pins='edges',
                         material='cotton', colour=(0.82, 0.8, 0.74), detail=96, tears=True, tear_strength=1.2)],
    },
    'sand_sling': {
        'name': 'Sand into a sling', 'category': 'Sand, snow and mud', 'size': '90 cm sheet',
        'blurb': 'Sand poured onto a cotton sheet tied to four posts: it heaps in the dip it makes, the sheet sagging '
                 'under its weight, and not a grain gets through. Matter in Sand, poured; Fabric held by its four corners.',
        'render': {'end': 120},
        'domain': {'size_x': 1.4, 'size_y': 1.2, 'size_z': 1.4, 'resolution': 40, 'preroll': 0.0, 'matter_detail': 112},
        'composite': {'backdrop': 'stage', 'floor': 'concrete'},
        'lighting': {'sun_on': True, 'sun_intensity': 3.0, 'sun_elevation': 40.0, 'sun_azimuth': -130.0, 'ambient_intensity': 2.5},
        'camera': {'distance': 2.1, 'target_y': 0.42, 'pitch': 22, 'yaw': 25, 'focal_mm': 35},
        'emitters': [],
        'colliders': [dict(name=f'Post {k + 1}', shape='cylinder', position=(sx * 0.47, 0.25, sz * 0.47),
                           size=(0.02, 0.25, 0.02), material='wood') for k, (sx, sz) in
                      enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1)))],
        'fabrics': [dict(name='Sling', position=(0.0, 0.5, 0.0), width=0.9, height=0.9, orientation='lying', pins='corners',
                         material='cotton', colour=(0.62, 0.15, 0.1))],
        'matter': [dict(name='Sand', material='sand', pours=True, position=(0.0, 0.95, 0.0), size=(0.03, 0.03, 0.03),
                        velocity=(0.0, -0.5, 0.0), rate=1.0, pour_start=0.0, pour_stop=3.0)],
    },
    'chocolate_pan': {
        'name': 'Chocolate in a hot pan', 'category': 'Sand, snow and mud', 'size': '24 cm pan',
        'blurb': 'Squares of chocolate dropped into a steel pan at 180 °C: they melt from the bottom where they touch it, '
                 'slump into glossy pools and run together. Matter in Chocolate; the pan’s Temperature 180 °C; Heat speed '
                 '60 (it melts some sixty times quicker than for real).',
        'render': {'end': 240},
        'domain': {'size_x': 0.6, 'size_y': 0.3, 'size_z': 0.5, 'resolution': 32, 'preroll': 0.0, 'matter_detail': 192,
                   'matter_heat_speed': 60.0},
        'composite': {'backdrop': 'stage', 'floor': 'boards'},
        'lighting': {'sun_on': True, 'sun_intensity': 2.5, 'sun_elevation': 40.0, 'sun_azimuth': -120.0, 'ambient_intensity': 1.8},
        'camera': {'distance': 0.55, 'target_y': 0.0, 'pitch': 45, 'yaw': 25, 'anchor_y': 0.6, 'focal_mm': 35},
        'emitters': [],
        'colliders': [
            dict(name='Pan', shape='cylinder', position=(0.0, 0.02, 0.0), size=(0.12, 0.02, 0.12), hollow=0.004,
                 opening=(0.13, 0.01, 0.13), opening_at=(0.0, 0.02, 0.0), material='steel', temperature=180.0),
            dict(name='Handle', shape='box', position=(0.21, 0.03, 0.0), size=(0.09, 0.006, 0.013), material='steel'),
        ],
        'matter': [dict(name=f'Square {k + 1}', material='chocolate', shape='box', position=(x, y, z),
                        size=(0.02, 0.004, 0.02), yaw=a) for k, (x, y, z, a) in enumerate((
                            (-0.05, 0.010, 0.03, 10.0), (0.03, 0.010, 0.04, -20.0), (0.0, 0.010, -0.05, 35.0),
                            (-0.01, 0.019, 0.0, 5.0), (0.06, 0.010, -0.02, 50.0)))],
    },
    'chocolate_fire': {
        'name': 'Chocolate by a fire', 'category': 'Sand, snow and mud', 'size': '20 cm flames',
        'blurb': 'Three pieces of chocolate on a slab beside a small fire: its radiant heat softens the side of each that '
                 'faces it, the nearest first, and they slump and run into glossy puddles. Matter in Chocolate, with '
                 'Heat speed 30 (it melts some thirty times quicker than for real).',
        'render': {'end': 240},
        'domain': {'size_x': 1.0, 'size_y': 1.0, 'size_z': 0.8, 'resolution': 64, 'preroll': 0.5, 'matter_detail': 160,
                   'matter_heat_speed': 30.0},
        'combustion': {'burn_rate': 8.0, 'heat': 0.9, 'soot': 0.2},
        'shading': {'exposure': -1.5},
        'composite': {'backdrop': 'stage', 'floor': 'tiles'},
        'lighting': {'sun_on': True, 'sun_intensity': 2.0, 'sun_elevation': 35.0, 'sun_azimuth': -120.0, 'ambient_intensity': 1.6},
        'camera': {'distance': 0.65, 'target_y': 0.07, 'pitch': 20, 'yaw': 15, 'focal_mm': 35},
        'emitters': [dict(name='Fire', shape='cylinder', position=(-0.2, 0.03, 0.0), size=(0.07, 0.03, 0.1), fuel=8,
                          temperature=0.6)],
        'matter': [dict(name='Bar', material='chocolate', shape='box', position=(-0.05, 0.02, -0.1), size=(0.06, 0.02, 0.035),
                        yaw=25.0),
                   dict(name='Truffle', material='chocolate', shape='sphere', position=(-0.06, 0.035, 0.09), size=(0.035, 0.035, 0.035)),
                   dict(name='Block', material='chocolate', shape='box', position=(0.08, 0.04, 0.0), size=(0.045, 0.04, 0.045))],
    },
    'yard_blast': {
        'name': 'Blast in a yard', 'category': 'Things that fall', 'size': '2 kg charge',
        'blurb': 'Two kilograms of explosive go off among crates, barrels, a brick wall and a heap of sand: a fireball, the '
                 'crates and barrels thrown clear, the wall blown down brick by brick, the sand flattened, then a column of '
                 'smoke. An emitter with Blast set.',
        'render': {'end': 96},
        'domain': {'size_x': 7.0, 'size_y': 5.0, 'size_z': 7.0, 'resolution': 112, 'preroll': 0.0, 'matter_detail': 128,
                   'substeps_max': 10},
        'combustion': {'burn_rate': 6.0, 'heat': 1.1, 'soot': 0.6, 'cooling': 2.2, 'radiative': 0.35, 'flame_life': 0.3,
                       'flame_gain': 2.0, 'expansion': 1.5, 'rich': 3.0, 'smoke_dissipation': 0.2},
        'motion': {'buoyancy': 8.0, 'turbulence': 6.0, 'turb_freq': 1.2, 'turb_rise': 2.0, 'vorticity': 3.0, 'disturbance': 4.0,
                   'disturb_block': 0.2},
        'shading': {'flame_k': 1700, 'max_k': 2400, 'smoke_density': 4.0, 'smoke_albedo': (0.1, 0.095, 0.09), 'flame_occlusion': 0.6,
                    'flame_absorption': 1.5, 'detail_freq': 1.0, 'exposure': -1.0},
        'embers': {'rate': 600, 'launch': 10.0, 'spread': 8.0, 'lifetime': 1.4, 'gravity': 4.0, 'drag': 1.0, 'size_max': 0.015,
                   'shutter': 0.25},
        'composite': {'backdrop': 'stage', 'floor': 'dirt'},
        'lighting': {'sun_on': True, 'sun_intensity': 3.0, 'sun_elevation': 35.0, 'sun_azimuth': -130.0, 'ambient_intensity': 2.5},
        'camera': {'distance': 7.5, 'target_y': 0.9, 'pitch': 10, 'yaw': 200, 'focal_mm': 35},
        'emitters': [dict(name='Explosion', shape='sphere', position=(0.0, 0.3, 0.0), size=(0.3, 0.3, 0.3), fuel=25.0, temperature=1.2,
                          radial=12.0, vel_blend=0.6, noise_freq=2.0, contrast=1.2, start=0.4, stop=0.5, fade_in=0.01, fade_out=0.08,
                          blast=2.0)],
        'colliders': [dict(name=f'Crate {k + 1}', shape='box', position=(x, 0.2, z), size=(0.2, 0.2, 0.2), yaw=20.0 * k,
                           dynamic=True, material='wood', density=300.0)
                      for k, (x, z) in enumerate(((1.3, 0.5), (-1.1, 0.9), (0.4, -1.3)))]
                     + [dict(name=f'Barrel {k + 1}', shape='cylinder', position=(x, 0.44, z), size=(0.29, 0.44, 0.29), dynamic=True,
                             material='steel', density=170.0, own_colour=True, colour=(0.05, 0.12, 0.32))
                        for k, (x, z) in enumerate(((-1.3, -0.9), (1.5, -0.8)))]
                     + [dict(name='Brick wall', shape='box', position=(0.0, 0.6, 2.1), size=(1.1, 0.6, 0.06), material='brick',
                             breakable=True, fracture='bricks')],
        'matter': [dict(name='Sand', material='sand', shape='pile', position=(-2.2, 0.2, 0.2), size=(0.5, 0.2, 0.5))],
    },
    'lightning_strike': {
        'name': 'Lightning strikes a post', 'category': 'Fires', 'size': '4 m bolt',
        'blurb': 'At dusk, lightning strikes a wooden post: three flashes down a branching channel light up the yard, and the '
                 'post catches and burns. A light of the Lightning kind, and Spreading fire.',
        'render': {'end': 144},
        'domain': {'size_x': 3.0, 'size_y': 4.6, 'size_z': 3.0, 'resolution': 112, 'preroll': 0.0},
        'combustion': {'burn_rate': 5.0, 'heat': 0.6, 'soot': 0.35, 'cooling': 2.2, 'flame_life': 0.08},
        'motion': {'buoyancy': 5.5, 'turbulence': 3.0, 'turb_freq': 2.5, 'vorticity': 1.6},
        'spread': {'enabled': True, 'ground': False, 'coverage': 1.0, 'burn_time': 20.0, 'fuel': 9.0, 'heat': 0.5, 'smoke': 1.0,
                   'catch_temp': 0.3, 'catch_time': 0.2, 'creep': 0.08, 'smoulder': 4.0},
        'shading': {'flame_k': 1650, 'max_k': 2250, 'smoke_density': 2.0, 'smoke_albedo': (0.5, 0.49, 0.48)},
        'composite': {'backdrop': 'stage', 'floor': 'dirt'},
        'lighting': {'sun_on': True, 'sun_intensity': 0.08, 'sun_elevation': 6.0, 'sun_azimuth': -60.0, 'ambient': (0.2, 0.25, 0.4),
                     'ambient_intensity': 0.25},
        'camera': {'distance': 6.5, 'target_y': 1.9, 'pitch': 4, 'yaw': 15, 'focal_mm': 35},
        'emitters': [],
        'colliders': [dict(name='Post', shape='cylinder', position=(0.3, 0.6, 0.0), size=(0.08, 0.6, 0.08), material='wood',
                           burnable=True),
                      dict(name='Crate', shape='box', position=(-0.9, 0.25, 0.4), size=(0.25, 0.25, 0.25), material='wood')],
        'lights': [dict(name='Lightning', kind='lightning', position=(-0.6, 4.2, -0.3), end=(0.3, 1.2, 0.0), intensity=600000.0,
                        strike_at=0.25, strokes=3, branching=0.6, thickness=0.03, colour=(0.8, 0.85, 1.0))],
    },
    'window_smash': {
        'name': 'Stone through a window', 'category': 'Things that fall', 'size': '1 m pane',
        'blurb': 'A stone thrown through a pane of window glass held in its frame: it punches a hole, shards break away '
                 'round it and fall, the rest stays in the frame. A breakable thin box of glass in Shards.',
        'render': {'end': 48},
        'domain': {'size_x': 2.4, 'size_y': 2.0, 'size_z': 2.4, 'resolution': 48, 'preroll': 0.0},
        'composite': {'backdrop': 'stage', 'floor': 'tiles'},
        'lighting': {'sun_on': True, 'sun_intensity': 3.0, 'sun_elevation': 40.0, 'sun_azimuth': -120.0, 'ambient_intensity': 2.5},
        'camera': {'distance': 2.6, 'target_y': 0.75, 'pitch': 6, 'yaw': 215, 'focal_mm': 35},
        'emitters': [],
        'colliders': [
            dict(name='Glass pane', shape='box', position=(0.0, 0.9, 0.0), size=(0.5, 0.5, 0.004), material='glass',
                 breakable=True, fracture='shards', pieces=40, held='edges'),
            dict(name='Frame top', shape='box', position=(0.0, 1.43, 0.0), size=(0.56, 0.03, 0.04), material='wood',
                 own_colour=True, colour=(0.75, 0.74, 0.7)),
            dict(name='Frame bottom', shape='box', position=(0.0, 0.37, 0.0), size=(0.56, 0.03, 0.04), material='wood',
                 own_colour=True, colour=(0.75, 0.74, 0.7)),
            dict(name='Frame left', shape='box', position=(-0.53, 0.9, 0.0), size=(0.03, 0.5, 0.04), material='wood',
                 own_colour=True, colour=(0.75, 0.74, 0.7)),
            dict(name='Frame right', shape='box', position=(0.53, 0.9, 0.0), size=(0.03, 0.5, 0.04), material='wood',
                 own_colour=True, colour=(0.75, 0.74, 0.7)),
            dict(name='Stone', shape='sphere', position=(-0.1, 0.95, -1.0), size=(0.045, 0.045, 0.045), dynamic=True,
                 material='stone', start_velocity=(0.1, 0.4, 9.0), release=0.3),
        ],
    },
    'vase_drop': {
        'name': 'Vase off a table', 'category': 'Things that fall', 'size': '35 cm vase',
        'blurb': 'A pottery vase knocked off the edge of a table: it tips, falls a metre, lands on its rim and shatters '
                 'across the tiles. A breakable hollow cylinder of Ceramic.',
        'render': {'end': 48},
        'domain': {'size_x': 2.4, 'size_y': 2.0, 'size_z': 2.4, 'resolution': 48, 'preroll': 0.0},
        'composite': {'backdrop': 'stage', 'floor': 'tiles'},
        'lighting': {'sun_on': True, 'sun_intensity': 2.5, 'sun_elevation': 45.0, 'sun_azimuth': -60.0, 'ambient_intensity': 2.5},
        'camera': {'distance': 2.4, 'target_y': 0.55, 'pitch': 14, 'yaw': 30, 'focal_mm': 35},
        'emitters': [],
        'colliders': [
            dict(name='Table', shape='box', position=(-0.45, 0.375, 0.0), size=(0.4, 0.375, 0.35), material='wood'),
            dict(name='Vase', shape='cylinder', position=(-0.12, 0.92, 0.0), size=(0.09, 0.17, 0.09), hollow=0.007,
                 opening=(0.12, 0.01, 0.12), opening_at=(0.0, 0.17, 0.0),   # (open at the top)
                 material='ceramic', own_colour=True, colour=(0.12, 0.25, 0.55), dynamic=True, breakable=True, pieces=40,
                 start_velocity=(0.7, 0.0, 0.0), release=0.3),
        ],
    },
    'crash_test': {
        'name': 'Crash test', 'category': 'Things that fall', 'size': '4.4 m car at 30 km/h',
        'blurb': 'A car driven at 30 km/h into a brick wall, a crash-test figure standing by it: the figure is hit, goes '
                 'limp and folds over the bonnet, and the car knocks the wall down, runs on over the bricks and brakes. A '
                 'car (its Speed keyed) and a person (Physics > Build), a brick wall that breaks.',
        'render': {'end': 96},
        'domain': {'size_x': 16.0, 'size_y': 4.0, 'size_z': 8.0, 'resolution': 96, 'preroll': 0.0},
        'composite': {'backdrop': 'stage', 'floor': 'concrete'},
        'lighting': {'sun_on': True, 'sun_intensity': 3.0, 'sun_elevation': 38.0, 'sun_azimuth': -130.0, 'ambient_intensity': 2.5},
        'camera': {'distance': 10.0, 'target_y': 0.8, 'pitch': 10, 'yaw': -30, 'focal_mm': 35},
        'emitters': [],
        'colliders': [
            dict(name='Car', shape='box', position=(-5.0, 0.75, 0.0), size=(2.2, 0.75, 0.9), build='car', material='painted',
                 own_colour=True, colour=(0.55, 0.06, 0.04), drive='rear', start_velocity=(8.3, 0.0, 0.0),
                 # (30 km/h through the wall, then braked to a stop past it)
                 drive_speed=K((0.0, 30.0), (1.3, 30.0), (1.45, 0.0), interp='linear')),
            dict(name='Person', shape='box', position=(1.2, 0.9, 0.6), size=(0.25, 0.9, 0.15), build='figure', material='person'),
            dict(name='Wall', shape='box', position=(3.2, 0.6, 0.0), size=(0.1, 0.6, 1.6), breakable=True, fracture='bricks',
                 material='brick', held='base'),
        ],
    },
    'stunt_fall': {
        'name': 'Stunt fall', 'category': 'Things that fall', 'size': '1.5 m drop',
        'blurb': 'A crash-test figure standing braced on a concrete block is hit in the chest by a thrown ball: it goes limp, '
                 'tumbles off the edge and lands in a heap past a stack of boxes. A person (Physics > Build) that stands '
                 'until it is hit hard.',
        'render': {'end': 72},
        'domain': {'size_x': 8.0, 'size_y': 4.0, 'size_z': 4.0, 'resolution': 64, 'preroll': 0.0},
        'composite': {'backdrop': 'stage', 'floor': 'concrete'},
        'lighting': {'sun_on': True, 'sun_intensity': 3.0, 'sun_elevation': 40.0, 'sun_azimuth': -120.0, 'ambient_intensity': 2.5},
        'camera': {'distance': 7.0, 'target_y': 1.3, 'pitch': 8, 'yaw': -15, 'focal_mm': 35},
        'emitters': [],
        'colliders': [
            dict(name='Block', shape='box', position=(0.0, 0.75, 0.0), size=(0.8, 0.75, 0.8), material='concrete'),
            dict(name='Person', shape='box', position=(0.35, 2.4, 0.0), size=(0.25, 0.9, 0.15), build='figure', material='person',
                 yaw=90.0),
            dict(name='Ball', shape='sphere', position=(-3.0, 2.9, 0.0), size=(0.22, 0.22, 0.22), dynamic=True, material='rubber',
                 density=600.0, start_velocity=(7.5, 2.2, 0.0), release=0.6),
            *[dict(name=f'Box {k + 1}', shape='box', position=(1.5, 0.25, 0.55 * (k - 1)), size=(0.25, 0.25, 0.25),
                   dynamic=True, material='cardboard') for k in range(3)],
        ],
    },
    'cart_jump': {
        'name': 'Cart off a ramp', 'category': 'Things that fall', 'size': '1 m cart, 6 m run',
        'blurb': 'A cart with a fire on its back, its four wheels turned by motors, races up a ramp, jumps off its end and '
                 'bowls over a tower of blocks, trailing flame and smoke, and brakes into a barrier. Motors on hinges (Joint › '
                 'Motor speed, keyed down to brake), a tilted plank (Shape › Roll) and a fire attached to the cart.',
        'render': {'end': 120},
        'domain': {'size_x': 7.6, 'size_y': 3.0, 'size_z': 2.4, 'resolution': 176, 'preroll': 0.0, 'substeps_max': 10},
        'combustion': {'burn_rate': 6.0, 'heat': 0.6, 'soot': 0.35, 'cooling': 2.5, 'flame_life': 0.07},
        'motion': {'buoyancy': 5.0, 'turbulence': 2.5, 'turb_freq': 4.0, 'vorticity': 1.4, 'disturbance': 1.6,
                   'disturb_block': 0.02},
        'shading': {'flame_k': 1650, 'max_k': 2250, 'smoke_density': 3.0, 'smoke_albedo': (0.5, 0.49, 0.48),
                    'flame_absorption': 12.0},
        'lighting': {'sun_on': True, 'sun_intensity': 1.2, 'sun_azimuth': -50.0, 'sun_elevation': 14.0,
                     'ambient': (0.42, 0.46, 0.55), 'ambient_intensity': 0.7},
        'composite': {'backdrop': 'stage', 'floor': 'concrete'},
        'embers': {'rate': 50, 'launch': 1.0, 'lifetime': 1.5},
        'camera': {'distance': 8.0, 'target_y': 0.8, 'pitch': 8, 'yaw': 0, 'anchor_x': 0.5, 'anchor_y': 0.62, 'focal_mm': 35},
        'colliders': [
            # (driving along +x: its wheels' axles along -z, so turning forward rolls it forward)
            dict(name='Cart', shape='box', position=(-3.0, 0.25, 0.0), size=(0.5, 0.06, 0.3), material='wood', dynamic=True),
            *[dict(name=f'Cart wheel {k + 1}', shape='cylinder', position=(-3.0 + sx * 0.35, 0.15, sz * 0.36),
                   size=(0.15, 0.04, 0.15), pitch=-90.0, material='rubber', own_colour=True, colour=(0.05, 0.05, 0.05),
                   joint='hinge', joint_to='Cart', joint_axis=(0.0, 1.0, 0.0), joint_friction=0.0, motor_torque=20.0,
                   # (flat out, then it brakes once it has hit the tower)
                   motor_speed=K((0.0, 150.0), (2.3, 150.0), (2.7, 0.0), interp='linear'))
              for k, (sx, sz) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1)))],
            # a 2 m plank at 15 degrees, its low end's top flush with the ground, propped at the top
            dict(name='Ramp', shape='box', position=(-0.4, 0.23, 0.0), size=(1.0, 0.03, 0.45), roll=15.0, material='wood'),
            dict(name='Ramp prop', shape='box', position=(0.4, 0.193, 0.0), size=(0.1, 0.193, 0.35), material='wood'),
            *[dict(name=f'Block {k + 1}', shape='box', position=(2.2, 0.1 + 0.2 * k, 0.0), size=(0.1, 0.1, 0.1),
                   yaw=(7.0 if k % 2 else -4.0), dynamic=True, material='wood') for k in range(6)],
            # (what stops it, however the crash goes)
            dict(name='Barrier', shape='box', position=(3.35, 0.4, 0.0), size=(0.15, 0.4, 1.0), material='concrete'),
        ],
        'emitters': [
            # (fed hard while it races along, where the wind of its going stretches the flame thin; less once it stands)
            dict(name='Cart fire', shape='sphere', position=(-3.0, 0.47, 0.0), size=(0.16, 0.16, 0.16), temperature=0.6,
                 fuel=K((0.0, 22.0), (2.6, 22.0), (3.4, 10.0)), noise_freq=8.0, noise=0.6, noise_rise=0.8),
        ],
        'links': [{'child': ['emitter', 'Cart fire'], 'parent': ['collider', 'Cart'], 'offset': [0.0, 0.22, 0.0]}],
    },
    'meadow_fire': {
        'name': 'Meadow fire', 'category': 'Fires', 'size': '8 m field of long grass',
        'blurb': 'A line of fire lit along the edge of a field of dry long grass: the wind drives the front across it, the '
                 'grass bending ahead of the flames, catching, burning down to black stubble and feeding the fire as it '
                 'goes. Grass & plants: Long grass, dried.',
        'render': {'end': 192},
        'domain': {'size_x': 9.0, 'size_y': 3.2, 'size_z': 6.0, 'resolution': 176, 'preroll': 0.0},
        'combustion': {'burn_rate': 6.0, 'heat': 0.6, 'soot': 0.4, 'cooling': 2.4, 'flame_life': 0.08},
        'motion': {'buoyancy': 5.0, 'turbulence': 3.0, 'turb_freq': 2.0, 'vorticity': 1.6, 'disturbance': 2.5, 'disturb_block': 0.05,
                   'wind_speed': 3.0, 'wind_dir': 90.0, 'gust': 0.4, 'wind_relax': 2.0},
        'shading': {'flame_k': 1650, 'smoke_density': 3.5, 'smoke_albedo': (0.45, 0.43, 0.4)},
        'lighting': {'sun_on': True, 'sun_intensity': 1.6, 'sun_azimuth': -50.0, 'sun_elevation': 18.0,
                     'ambient': (0.45, 0.5, 0.6), 'ambient_intensity': 0.7},
        'composite': {'backdrop': 'stage', 'floor': 'dirt'},
        'embers': {'rate': 80},
        'camera': {'distance': 10.5, 'target_y': 0.5, 'pitch': 14, 'yaw': 12, 'anchor_x': 0.5, 'anchor_y': 0.7, 'focal_mm': 35},
        'emitters': [
            dict(name='Fire line', shape='capsule', position=(-3.7, 0.1, -1.7), end=(-3.7, 0.1, 1.7), size=(0.15, 0.15, 0.15),
                 fuel=12, temperature=0.6, stop=1.0, fade_out=0.3),
        ],
        'strands': [dict(name='Dry grass', kind='meadow', position=(0.0, 0.0, 0.0), size=(4.2, 0.45, 2.6), dryness=0.85)],
    },
    'shed_fire': {
        'name': 'Shed on fire', 'category': 'Things that fall', 'size': '2 m wooden shed',
        'blurb': 'A fire inside a wooden shed: its walls and posts catch, char and weaken until the posts burn through, '
                 'the roof falls in and the walls break up, the burnt pieces smouldering and crumbling to ash. Objects '
                 'that are both Breakable and Burnable burn piece by piece.',
        'render': {'end': 480},
        'domain': {'size_x': 4.2, 'size_y': 4.4, 'size_z': 3.6, 'resolution': 160, 'preroll': 0.0},
        'combustion': {'burn_rate': 6.0, 'heat': 0.6, 'soot': 0.45, 'cooling': 2.2, 'flame_life': 0.08},
        'motion': {'buoyancy': 5.5, 'turbulence': 3.0, 'turb_freq': 2.0, 'vorticity': 1.6, 'disturbance': 2.0,
                   'disturb_block': 0.04},
        'spread': {'enabled': True, 'ground': False, 'coverage': 1.0, 'burn_time': 3.5, 'fuel': 6.0, 'heat': 0.5, 'smoke': 1.4,
                   'catch_temp': 0.32, 'catch_time': 1.0, 'creep': 0.05, 'smoulder': 6.0, 'smoulder_smoke': 1.0},
        'shading': {'flame_k': 1650, 'max_k': 2250, 'smoke_density': 4.5, 'smoke_albedo': (0.22, 0.21, 0.2), 'exposure': -1.0},
        'lighting': {'sun_on': True, 'sun_intensity': 0.5, 'sun_azimuth': -60.0, 'sun_elevation': 8.0,
                     'ambient': (0.32, 0.38, 0.5), 'ambient_intensity': 0.45},
        'composite': {'backdrop': 'stage', 'floor': 'dirt'},
        'embers': {'rate': 90, 'launch': 1.6, 'lifetime': 2.4},
        'camera': {'distance': 7.0, 'target_y': 1.2, 'pitch': 8, 'yaw': 28, 'anchor_x': 0.5, 'anchor_y': 0.62, 'focal_mm': 35},
        'colliders': [
            *[dict(name=f'Post {k + 1}', shape='box', position=(sx * 0.95, 1.0, sz * 0.75), size=(0.06, 1.0, 0.06), material='wood',
                   breakable=True, burnable=True, pieces=8, held='base')
              for k, (sx, sz) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1)))],
            dict(name='Back wall', shape='box', position=(0.0, 0.97, -0.75), size=(0.885, 0.95, 0.025), material='wood',
                 breakable=True, burnable=True, fracture='splinters', pieces=24, held='base'),
            dict(name='Left wall', shape='box', position=(-0.95, 0.97, 0.0), size=(0.025, 0.95, 0.685), material='wood',
                 breakable=True, burnable=True, fracture='splinters', pieces=20, held='base'),
            dict(name='Right wall', shape='box', position=(0.95, 0.97, 0.0), size=(0.025, 0.95, 0.685), material='wood',
                 breakable=True, burnable=True, fracture='splinters', pieces=20, held='base'),
            # (resting on the posts, held by nothing else)
            dict(name='Roof', shape='box', position=(0.0, 2.045, 0.0), size=(1.08, 0.04, 0.88), material='wood',
                 breakable=True, burnable=True, fracture='splinters', pieces=28, held='free'),
        ],
        'emitters': [
            dict(name='Fire inside', shape='cylinder', position=(0.2, 0.06, -0.45), size=(0.2, 0.06, 0.2), fuel=8,
                 temperature=0.5, noise_rise=1.5),
        ],
    },
    'crates_in_fire': {
        'name': 'Crates into a fire', 'category': 'Things that fall', 'size': '1 m campfire',
        'blurb': 'Three wooden crates dropped onto a campfire one after another: they land on the burning logs, shove '
                 'the smoke aside, catch and burn. Burnable objects with Falls on, and Spreading fire.',
        'render': {'end': 144},
        'domain': {'size_x': 2.6, 'size_y': 3.2, 'size_z': 2.4, 'resolution': 144, 'preroll': 2.0, 'time_scale': 1.4,
                   'substeps_max': 12},
        'combustion': {'burn_rate': 5.0, 'heat': 0.6, 'soot': 0.35, 'cooling': 2.2, 'flame_life': 0.08},
        'motion': {'puffing': 0.6, 'buoyancy': 5.5, 'turbulence': 3.0, 'turb_freq': 2.5, 'vorticity': 1.6, 'disturbance': 2.0,
                   'disturb_block': 0.04},
        'spread': {'enabled': True, 'ground': False, 'coverage': 1.0, 'burn_time': 10.0, 'fuel': 8.0, 'heat': 0.5, 'smoke': 1.0,
                   'catch_temp': 0.3, 'catch_time': 0.4, 'creep': 0.05, 'smoulder': 4.0},
        'shading': {'flame_k': 1650, 'max_k': 2250, 'smoke_density': 2.5, 'smoke_albedo': (0.5, 0.49, 0.48), 'coal_bed': 1.0,
                    'coal_height': 0.08, 'flame_threshold': 0.1, 'flame_sharpness': 3.2, 'flame_absorption': 10.0,
                    'soot_glow': 0.1, 'detail': 0.5},
        # dusk: a low, weak sun and a dim blue sky, the fire lighting the crates
        'lighting': {'sun_on': True, 'sun_intensity': 0.6, 'sun_azimuth': -60.0, 'sun_elevation': 8.0,
                     'ambient': (0.32, 0.38, 0.5), 'ambient_intensity': 0.5},
        'composite': {'backdrop': 'stage', 'floor': 'dirt'},
        'embers': {'rate': 70, 'launch': 1.6, 'lifetime': 2.2},
        'camera': {'distance': 4.4, 'target_y': 0.75, 'pitch': 10, 'yaw': 18, 'anchor_x': 0.5, 'anchor_y': 0.75, 'focal_mm': 35},
        'colliders': [
            dict(name='Firewood', shape='mesh', mesh='builtin:firewood.obj', position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0)),
            # (slatted crates: light, rough, and they hardly bounce)
            dict(name='Crate 1', shape='box', position=(0.1, 0.8, 0.05), size=(0.15, 0.15, 0.15), yaw=20.0, dynamic=True,
                 material='wood', density=300.0, friction=0.8, bounce=0.05, burnable=True, release=0.3),
            dict(name='Crate 2', shape='box', position=(-0.3, 1.0, 0.2), size=(0.15, 0.15, 0.15), yaw=-35.0, dynamic=True,
                 material='wood', density=300.0, friction=0.8, bounce=0.05, burnable=True, release=1.3),
            dict(name='Crate 3', shape='box', position=(0.3, 1.2, -0.22), size=(0.15, 0.15, 0.15), yaw=60.0, dynamic=True,
                 material='wood', density=300.0, friction=0.8, bounce=0.05, burnable=True, release=2.3),
        ],
        'emitters': [
            dict(noise_rise=1.5, name='Log A', shape='capsule', position=(-0.42, 0.08, -0.10), end=(0.10, 0.15, 0.02),
                 size=(0.09, 0.09, 0.09), fuel=10, temperature=0.45),
            dict(noise_rise=1.5, name='Log B', shape='capsule', position=(0.38, 0.08, -0.22), end=(-0.06, 0.17, 0.05),
                 size=(0.09, 0.09, 0.09), fuel=10, temperature=0.45, seed=3),
            dict(noise_rise=1.5, name='Log C', shape='capsule', position=(0.16, 0.07, 0.42), end=(-0.02, 0.18, -0.06),
                 size=(0.08, 0.08, 0.08), fuel=9, temperature=0.45, seed=5),
            dict(noise_rise=1.5, name='Coal bed', shape='cylinder', position=(0.0, 0.04, 0.0), size=(0.36, 0.04, 0.36),
                 fuel=8, temperature=0.4, seed=7),
        ],
    },
}

ORDER = ['campfire', 'bonfire', 'torch', 'candle', 'gas_ring', 'pool_fire', 'fire_line', 'fireball', 'flamethrower',
         'vehicle_fire', 'smoke_plume', 'fire_whirl', 'waved_torch', 'hose_douse', 'grass_fire', 'spot_fires', 'hillside_fire',
         'curtain_fire', 'fabric_curtain', 'wet_towels', 'armchair_fire', 'room_fire', 'backdraft', 'flash_fire', 'gas_cloud', 'coloured_flames', 'road_flare',
         'grinder_sparks', 'fireworks', 'car_through_smoke', 'flag_wind', 'kettle_steam', 'steam_vent',
         'crates_in_fire', 'tower_knockdown', 'wall_smash', 'wrecking_ball', 'chain_swing', 'window_smash', 'vase_drop',
         'yard_blast', 'lightning_strike', 'sheet_rip', 'crash_test', 'stunt_fall', 'cart_jump', 'meadow_fire', 'shed_fire', 'sand_hopper', 'snowballs', 'jelly_ball', 'mud_drag', 'sand_castle', 'sand_sling', 'iron_pour', 'chocolate_fire', 'chocolate_pan']


def make(name: str, fps=None, start=None) -> Scene:
    """A fresh scene built from a preset. `fps` and `start` set the frame rate and first frame the
    preset's keyframes are placed for (they default to the preset's own)."""
    spec = PRESETS[name]
    s = Scene()
    r = s.data['render']
    if fps is not None:
        r['fps'] = float(fps)
    if start is not None:
        length = r['end'] - r['start']
        r['start'], r['end'] = int(start), int(start) + length
    _apply(s, spec)
    for c, cs in zip(s.colliders, spec.get('colliders', [])):
        if 'material' not in cs and c['name'] in MATERIAL_BY_NAME:
            m = MATERIAL_BY_NAME[c['name']]
            c['material'] = m[0]
            if len(m) > 1:
                c['own_colour'], c['colour'] = True, m[1]
    # (the presets' looks were made on the flat background; a preset can ask for the stage)
    if 'backdrop' not in spec.get('composite', {}):
        s.data['composite']['backdrop'] = 'colour'
    s.name = spec['name']
    s.preset = name
    return s


# What the presets' objects are made of, by name (unless the preset says), with a colour of their own
MATERIAL_BY_NAME = {
    'Rock': ('stone',), 'Stone': ('stone',), 'Boulder': ('stone',), 'Boulder 2': ('stone',), 'Rock downstream': ('stone',),
    'Ledge': ('stone',), 'Slab': ('stone',), 'Shore': ('stone', (0.05, 0.05, 0.055)),
    'Crate': ('wood',), 'Post': ('wood',), 'Post 2': ('wood',), 'Palm': ('wood',), 'Firewood': ('wood',), 'Door': ('wood',),
    'Ball': ('plastic',), 'Buoy': ('plastic',), 'Barrel': ('plastic', (0.04, 0.12, 0.45)),
    'Wall': ('plaster',), 'Room': ('plaster',), 'Step': ('concrete',), 'Hotel': ('concrete',), 'House': ('plaster',),
    'Block': ('concrete',),
    'Hull': ('painted', (0.75, 0.75, 0.72)), 'Car body': ('painted', (0.06, 0.07, 0.08)), 'Car': ('painted', (0.4, 0.42, 0.45)),
    'Pot': ('steel',), 'Pot base': ('steel',), 'Plate 150 C': ('steel', (0.12, 0.12, 0.13)),
    'Plate 300 C': ('steel', (0.12, 0.12, 0.13)), 'Rail': ('steel',), 'Pole': ('steel',),
    'Curtain': ('fabric',), 'Armchair': ('fabric', (0.25, 0.14, 0.08)),
    'Bank': ('earth',), 'Bank 2': ('earth',), 'Banks': ('earth',), 'Hillside': ('earth', (0.09, 0.1, 0.05)),
    'Coast': ('earth',), 'Coast_wide': ('earth',), 'Beach': ('earth', (0.45, 0.38, 0.27)),
    'Dry ice': ('ceramic', (0.85, 0.87, 0.9)),
}


def apply_to(scene: Scene, name: str, keep_camera=True, keep_render=True):
    """Load a preset's fire into an existing scene, keeping its footage, framing and output settings."""
    fresh = make(name, fps=scene.fps, start=scene.start) if keep_render else make(name)
    keep = {}
    if keep_camera:
        keep['camera'] = dict(scene.data['camera'])
        # (the clip planes go with the effect's scale: a sky preset works in kilometres)
        if getattr(scene, 'ground', None):   # a camera matched to the footage keeps its lens; the clip planes only widen
            c = keep['camera']
            c['near'] = min(float(c['near']), float(fresh.data['camera']['near']))
            c['far'] = max(float(c['far']), float(fresh.data['camera']['far']))
        else:
            for k in ('distance', 'target_y', 'focal_mm', 'near', 'far'):
                keep['camera'][k] = fresh.data['camera'][k]
    if keep_render:
        keep['render'] = dict(scene.data['render'])
    keep['composite'] = dict(scene.data['composite'])
    footage, track = scene.footage, scene.track
    scene.data = fresh.data
    for k, v in keep.items():
        scene.data[k] = v
    scene.emitters = fresh.emitters
    scene.colliders = fresh.colliders
    scene.fabrics = fresh.fabrics
    scene.matter = fresh.matter
    scene.strands = fresh.strands
    scene.links = fresh.links       # (what is attached to what among the effect's objects, which have all changed)
    # the shot's own lights stay (they are its set); a preset's lightning is part of its effect
    scene.lights = [l for l in scene.lights if l.get('kind') != 'lightning'] + [l for l in fresh.lights if l.get('kind') == 'lightning']
    scene.footage, scene.track = footage, track
    scene.preset = name
    return scene


# Liquid presets live in their own module; they use K for keyframes.
from .liquid_presets import LIQUID_ORDER, liquid_presets  # noqa: E402

PRESETS.update(liquid_presets(K))
ORDER.extend(LIQUID_ORDER)

# Sea presets (the open water of engine/ocean.py) too.
from .ocean_presets import OCEAN_ORDER, ocean_presets  # noqa: E402

PRESETS.update(ocean_presets(K))
ORDER.extend(OCEAN_ORDER)

# Liquids that freeze, melt, boil and evaporate (engine/liquid_thermal.py).
from .phase_presets import PHASE_ORDER, phase_presets  # noqa: E402

PRESETS.update(phase_presets(K))
ORDER.extend(PHASE_ORDER)

# Weather: snow, hail, sleet and freezing rain (engine/weather.py).
from .weather_presets import WEATHER_ORDER, weather_presets  # noqa: E402

PRESETS.update(weather_presets(K))
ORDER.extend(WEATHER_ORDER)

# The sky: clouds, storms (engine/cloud.py).
from .cloud_presets import CLOUD_ORDER, cloud_presets  # noqa: E402

PRESETS.update(cloud_presets(K))
ORDER.extend(CLOUD_ORDER)

# Fire, water and lava together (engine/both_engine.py).
from .both_presets import BOTH_ORDER, both_presets  # noqa: E402

PRESETS.update(both_presets(K))
ORDER.extend(n for n in BOTH_ORDER if n not in ORDER)
