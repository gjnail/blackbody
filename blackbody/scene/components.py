"""Building blocks for making a scene from scratch: fire and liquid sources, objects, fabric, lights
and forces, each a few ready-set emitters, colliders, fabrics or lights (and sometimes a scene-wide
setting) that can be added to any scene.

Values come from the presets, where they were checked against real fires and liquids. Generic
sources (`scales`) are sized for a 2 m box and scale with the scene's box; real objects (a torch, a
hose nozzle, a curtain) keep their real size, and the box grows to fit them. Adding a liquid to a fire
scene (or fire to a liquid scene) turns it into a fire-and-liquid scene, so the two meet.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

from .model import Scene, turn_matrix
from .params import defaults

REF_WIDTH = 2.0   # the box width the scaling blocks are sized for


@dataclass
class Component:
    key: str
    name: str
    group: str
    tip: str
    glyph: str
    need: str = 'any'          # 'fire', 'liquid', 'lava' or 'any': what the scene must simulate
    objects: list = field(default_factory=list)   # [(kind, dict)]: kind emitter / collider / light / fabric / shot
    scene: dict = field(default_factory=dict)     # {section: {key: value}} set when added
    scales: bool = False       # sized for a 2 m box: scaled to the scene's box
    room: tuple = (0.0, 0.0, 0.0)   # space it needs around its base: half width, height, half depth (m)
    pick: str = ''             # 'mesh': ask for a file for the first object; 'text': ask for words, 'shape': for a picture
                               # (every mesh object gets it)
    origin: bool = False       # objects are placed about the block's own origin (your blocks), not the first object
    links: list = field(default_factory=list)     # attachments between its objects, by name
    file: str = ''             # your block's .bbblock file


def _E(**kw):
    return ('emitter', kw)


def _C(**kw):
    return ('collider', kw)


def _L(**kw):
    return ('light', kw)


def _F(**kw):
    return ('fabric', kw)


def _M(**kw):
    return ('matter', kw)


def _S(**kw):
    return ('strands', kw)


def _G(**kw):
    return ('shot', kw)


SPARKS = {'embers': {'enabled': True, 'rate': 1500.0, 'count': 32768, 'lifetime': 0.9, 'life_jitter': 0.7, 'launch': 14.0,
                     'spread': 1.2, 'direction': (1.0, -0.2, 0.0), 'cone': 12.0, 'drag': 0.35, 'gravity': 9.81, 'turbulence': 0.5,
                     'temperature': 2300.0, 'cooling': 1.6, 'size_min': 0.0006, 'size_max': 0.0025, 'brightness': 1.8,
                     'fade_in': 0.0, 'bounce': 0.35, 'friction': 0.3}}

COMPONENTS = [
    # -- fire ---------------------------------------------------------------------------------------------
    Component('burner', 'Fire', 'Fire', 'A patch of burning fuel on the ground: the basic flame. Drag it out, '
              'make it bigger with its square handle.', 'flame', 'fire', scales=True, room=(0.8, 2.6, 0.8),
              objects=[_E(name='Fire', shape='cylinder', position=(0.0, 0.04, 0.0), size=(0.3, 0.04, 0.3), fuel=12.0, temperature=0.45,
                          noise_freq=2.5, noise_rise=1.5)]),
    Component('campfire', 'Campfire', 'Fire', 'Two burning logs crossed over a coal bed, as in the Campfire preset.', 'flame', 'fire',
              scales=True, room=(0.9, 3.0, 0.9),
              objects=[_E(name='Log A', shape='capsule', position=(-0.42, 0.07, -0.18), end=(0.4, 0.07, 0.2), size=(0.08, 0.08, 0.08),
                          fuel=10.0, temperature=0.45, noise_rise=1.5),
                       _E(name='Log B', shape='capsule', position=(-0.35, 0.09, 0.25), end=(0.38, 0.09, -0.22), size=(0.08, 0.08, 0.08),
                          fuel=10.0, temperature=0.45, noise_rise=1.5, seed=3),
                       _E(name='Coal bed', shape='cylinder', position=(0.0, 0.04, 0.0), size=(0.34, 0.04, 0.34), fuel=8.0,
                          temperature=0.4, noise_rise=1.5, seed=7)]),
    Component('pool', 'Fuel pool', 'Fire', 'A pool of burning petrol or oil: a wide, smoky flame.', 'flame', 'fire', scales=True,
              room=(1.2, 3.2, 1.2),
              objects=[_E(name='Fuel pool', shape='cylinder', position=(0.0, 0.03, 0.0), size=(0.65, 0.03, 0.55), fuel=12.0,
                          smoke=1.5, noise_freq=1.6, noise_rise=1.5)]),
    Component('line', 'Fire line', 'Fire', 'A burning line along the ground: a trail of fuel, a grass-fire front.', 'flame', 'fire',
              scales=True, room=(1.2, 2.0, 0.6),
              objects=[_E(name='Fire line', shape='capsule', position=(-0.8, 0.05, 0.0), end=(0.8, 0.05, 0.0), size=(0.07, 0.07, 0.07),
                          fuel=12.0, noise_freq=2.5)]),
    Component('torch', 'Torch', 'Fire', 'A hand-held torch head in mid-air, 1 m up (real size).', 'flame', 'fire',
              room=(0.3, 1.8, 0.3),
              objects=[_E(name='Torch', shape='cylinder', position=(0.0, 1.0, 0.0), size=(0.055, 0.06, 0.055), fuel=18.0,
                          temperature=0.5, noise_freq=14.0, noise_rise=1.25, noise=0.7)]),
    Component('gas', 'Gas burner', 'Fire', 'A blue gas ring, 15 cm across (real size). Turn the base glow up in Shading for the blue.',
              'flame', 'fire', room=(0.2, 0.4, 0.2),
              objects=[_E(name='Gas ring', shape='ring', position=(0.0, 0.02, 0.0), size=(0.075, 0.008, 0.075), fuel=30.0,
                          temperature=0.6, velocity=(0.0, 0.6, 0.0), vel_blend=0.3, noise=0.4, noise_freq=40.0, embers=False)]),
    Component('fireball', 'Fireball', 'Fire', 'A burst of fuel that goes up in a rolling fireball at the start of the shot.', 'flame',
              'fire', scales=True, room=(1.6, 3.0, 1.6),
              objects=[_E(name='Fireball', shape='sphere', position=(0.0, 0.3, 0.0), size=(0.18, 0.14, 0.18), fuel=25.0, temperature=1.2,
                          radial=4.0, vel_blend=0.5, noise_freq=4.0, contrast=1.2, start=0.0, stop=0.2, fade_in=0.02, fade_out=0.12)],
              scene={'combustion': {'expansion': 2.0}}),
    Component('explosion', 'Explosion', 'Fire', 'A 2 kg charge going off at half a second: a fireball, and a blast wave that '
              'throws the things that fall, blows breakable things apart and scatters sand and snow. Set its size in Blast.', 'burst',
              'fire', room=(2.0, 4.0, 2.0),
              objects=[_E(name='Explosion', shape='sphere', position=(0.0, 0.3, 0.0), size=(0.3, 0.3, 0.3), fuel=25.0, temperature=1.2,
                          radial=12.0, vel_blend=0.6, noise_freq=2.0, contrast=1.2, start=0.5, stop=0.6, fade_in=0.01, fade_out=0.08,
                          blast=2.0)],
              scene={'combustion': {'expansion': 2.0, 'soot': 0.6, 'smoke_dissipation': 0.2},
                     'shading': {'smoke_albedo': (0.08, 0.075, 0.07), 'flame_occlusion': 0.6, 'flame_absorption': 1.5}}),
    Component('jet', 'Flame jet', 'Fire', 'A jet of burning fuel shot sideways, like a flamethrower, curling up as it burns.',
              'flame', 'fire', scales=True, room=(1.6, 1.6, 0.5),
              objects=[_E(name='Flame jet', shape='capsule', position=(-0.8, 0.5, 0.0), end=(-0.67, 0.51, 0.0), size=(0.022, 0.022, 0.022),
                          fuel=120.0, temperature=0.9, velocity=(8.0, 0.5, 0.0), vel_blend=1.0, noise=0.5, noise_freq=12.0)]),
    Component('whirl', 'Fire whirl', 'Fire', 'A pool fire with the air spinning around it, which twists the flames into a column.',
              'flame', 'fire', scales=True, room=(1.0, 3.2, 1.0),
              objects=[_E(name='Fire whirl', shape='cylinder', position=(0.0, 0.04, 0.0), size=(0.36, 0.04, 0.36), fuel=14.0,
                          temperature=0.5, swirl=2.5, swirl_width=7.0, noise_freq=2.5)]),
    Component('colour', 'Coloured flame', 'Fire', 'A flame tinted by a metal salt: copper green. Change its Colourant colour for others.',
              'flame', 'fire', scales=True, room=(0.4, 1.2, 0.4),
              objects=[_E(name='Coloured flame', shape='cylinder', position=(0.0, 0.03, 0.0), size=(0.12, 0.03, 0.12), fuel=20.0,
                          temperature=0.5, color_amount=8.0, color=(0.08, 1.0, 0.35), noise_freq=8.0, embers=False)]),
    Component('burnable', 'Burnable block', 'Fire', 'A block that catches fire where flames touch it, burns and spreads over itself '
              '(turns on Spreading fire). Put a fire next to it.', 'cube', 'fire', scales=True, room=(0.6, 2.0, 0.6),
              objects=[_C(name='Burnable block', shape='box', position=(0.45, 0.25, 0.0), size=(0.25, 0.25, 0.25), burnable=True)],
              scene={'spread': {'enabled': True}}),
    Component('text_fire', 'Burning text', 'Fire', 'Type words and they burn: solid letters in any font on this computer, '
              'with fire all over them. Right-click them to edit the text.', 'text', 'fire', pick='text',
              objects=[_E(name='Fire', shape='mesh', position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), thickness=0.015, fuel=10.0,
                          temperature=0.55, noise=0.6, noise_freq=10.0, noise_rise=1.5),
                       _C(name='Text', shape='mesh', position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))]),
    Component('logo_fire', 'Burning logo', 'Fire', 'A logo or picture (SVG, PNG, JPG) made solid, with fire all over it: '
              'its shape is traced from the picture.', 'shape', 'fire', pick='shape',
              objects=[_E(name='Fire', shape='mesh', position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), thickness=0.015, fuel=10.0,
                          temperature=0.55, noise=0.6, noise_freq=10.0, noise_rise=1.5),
                       _C(name='Logo', shape='mesh', position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))]),
    Component('armchair', 'Armchair', 'Fire', 'A burnable armchair (a built-in mesh) with a small fire on its seat (turns on Spreading fire).',
              'cube', 'fire', room=(0.8, 2.4, 0.8),
              objects=[_C(name='Armchair', shape='mesh', mesh='builtin:armchair.obj', position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0),
                          burnable=True),
                       _E(name='Seat fire', shape='sphere', position=(0.0, 0.5, 0.1), size=(0.12, 0.06, 0.12), fuel=12.0, temperature=0.6,
                          stop=2.0, fade_out=0.5)],
              scene={'spread': {'enabled': True}}),
    # -- smoke, steam and sparks --------------------------------------------------------------------------------
    Component('smoke', 'Smoke', 'Smoke, steam & sparks', 'Smouldering smoke with no flame: a smoke column, a smoke machine.', 'cloud', 'fire',
              scales=True, room=(0.8, 3.2, 0.8),
              objects=[_E(name='Smoke', shape='cylinder', position=(0.0, 0.05, 0.0), size=(0.25, 0.05, 0.25), fuel=0.0, temperature=0.8,
                          smoke=6.0, noise_freq=4.0, embers=False)]),
    Component('steam', 'Steam vent', 'Smoke, steam & sparks', 'Steam blowing up out of a vent: clear at the vent, clouding as it cools.',
              'cloud', 'fire', scales=True, room=(0.6, 3.2, 0.6),
              objects=[_E(name='Steam vent', shape='cylinder', position=(0.0, 0.1, 0.0), size=(0.06, 0.04, 0.06), fuel=0.0,
                          temperature=0.1, velocity=(0.0, 3.0, 0.0), vel_blend=1.0, vapour=550.0, noise=0.3, embers=False)]),
    Component('kettle', 'Kettle steam', 'Smoke, steam & sparks', 'A wisp of steam from a kettle spout (real size).', 'cloud', 'fire',
              room=(0.3, 0.8, 0.25),
              objects=[_E(name='Spout', shape='sphere', position=(0.0, 0.25, 0.0), size=(0.01, 0.01, 0.01), fuel=0.0, temperature=0.055,
                          velocity=(0.5, 1.2, 0.0), vel_blend=0.8, vapour=590.0, noise=0.2, embers=False)]),
    Component('sparks', 'Sparks', 'Smoke, steam & sparks', 'A shower of hot metal sparks, as from an angle grinder. This sets how every '
              'ember in the scene flies: fast, under full gravity, bouncing off the floor.', 'sparks', 'fire', room=(1.5, 1.2, 0.8),
              objects=[_E(name='Sparks', shape='sphere', position=(-0.6, 0.8, 0.0), size=(0.01, 0.01, 0.01), fuel=0.0, temperature=0.0,
                          noise=0.0, embers=True)],
              scene=SPARKS),
    # -- liquids ------------------------------------------------------------------------------------------
    Component('pour', 'Pour', 'Liquids', 'A stream of water pouring from a spout 80 cm up, as from a jug.', 'drop', 'liquid',
              room=(0.7, 1.0, 0.5),
              scales=True, objects=[_E(name='Pour', shape='cylinder', position=(-0.3, 0.8, 0.0), size=(0.03, 0.02, 0.03), velocity=(0.9, -0.3, 0.0),
                          vel_blend=1.0, noise=0.0, embers=False)]),
    Component('hose', 'Hose jet', 'Liquids', 'A hose jet aimed sideways at 7 m/s. Point it with its Velocity.', 'drop', 'liquid',
              room=(1.2, 1.2, 0.6),
              scales=True, objects=[_E(name='Hose', shape='sphere', position=(-0.8, 0.55, 0.0), size=(0.018, 0.018, 0.018), velocity=(7.0, 1.0, 0.0),
                          vel_blend=1.0, noise=0.0, embers=False, jitter=0.03)]),
    Component('fountain', 'Fountain', 'Liquids', 'A jet straight up from the ground that breaks up and rains back down.', 'drop', 'liquid',
              room=(0.6, 1.5, 0.6),
              scales=True, objects=[_E(name='Fountain', shape='cylinder', position=(0.0, 0.06, 0.0), size=(0.022, 0.03, 0.022), velocity=(0.0, 4.8, 0.0),
                          vel_blend=1.0, noise=0.0, embers=False, jitter=0.04)]),
    Component('block', 'Block of water', 'Liquids', 'A block of water that is there at the start and collapses: a dam break.', 'drop',
              'liquid', scales=True, room=(0.8, 0.8, 0.6),
              objects=[_E(name='Water', shape='box', position=(-0.5, 0.25, 0.0), size=(0.25, 0.25, 0.3), noise=0.0, embers=False,
                          liquid_mode='fill', start=0.0)]),
    Component('text_water', 'Water letters', 'Liquids', 'Words made of water, in any font on this computer: they hang in '
              'the air for a moment, then fall and splash.', 'text', 'liquid', pick='text',
              objects=[_E(name='Water', shape='mesh', position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), thickness=0.0, noise=0.0,
                          embers=False, liquid_mode='fill', start=0.0)]),
    Component('logo_water', 'Water shape', 'Liquids', 'A logo or picture (SVG, PNG, JPG) made of water: it hangs in the air '
              'for a moment, then falls and splashes.', 'shape', 'liquid', pick='shape',
              objects=[_E(name='Water', shape='mesh', position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), thickness=0.0, noise=0.0,
                          embers=False, liquid_mode='fill', start=0.0)]),
    Component('throw', 'Thrown water', 'Liquids', 'A bucketful of water thrown through the air.', 'drop', 'liquid', room=(1.2, 1.0, 0.6),
              scales=True, objects=[_E(name='Thrown water', shape='sphere', position=(-0.9, 0.75, 0.0), size=(0.2, 0.1, 0.1), velocity=(2.4, 0.9, 0.0),
                          radial=0.7, noise=0.0, start=0.05, embers=False, liquid_mode='fill')]),
    Component('flow', 'Waterfall', 'Liquids', 'A sheet of water flowing off a ledge.', 'drop', 'liquid', room=(1.0, 1.7, 0.6),
              scales=True, objects=[_E(name='Flow', shape='box', position=(-0.9, 1.56, 0.0), size=(0.18, 0.04, 0.38), velocity=(1.2, 0.0, 0.0),
                          vel_blend=1.0, noise=0.0, embers=False),
                       _C(name='Ledge', shape='box', position=(-0.7, 0.75, 0.0), size=(0.4, 0.75, 0.55))]),
    Component('pond', 'Pond', 'Liquids', 'Still water filling the bottom of the box, 25 cm deep, for things to splash into.', 'waves',
              'liquid', room=(0.0, 0.5, 0.0), scene={'liquid': {'water_level': 0.25, 'settle': True}}),
    Component('rain', 'Rain', 'Weather', 'Rain falling over the whole box (a heavy shower). Rain on a fire puts it out.', 'weather',
              'liquid', scene={'liquid': {'rain': 25.0}}),
    Component('ink', 'Ink', 'Liquids', 'A drip of ink that clouds in the water. Change its Dye colour.', 'drop', 'liquid',
              room=(0.3, 0.6, 0.3),
              scales=True, objects=[_E(name='Ink', shape='sphere', position=(0.0, 0.45, 0.0), size=(0.006, 0.01, 0.006), velocity=(0.0, -0.8, 0.0),
                          vel_blend=1.0, noise=0.0, start=0.3, stop=1.3, embers=False, dye=(0.02, 0.03, 0.12), dye_amount=120.0,
                          dye_cloud=0.25)]),
    Component('lava', 'Lava', 'Liquids', 'Molten rock pouring out of a vent: glowing, crusting over as it cools, boiling any water it meets.',
              'lava', 'lava', room=(1.2, 0.8, 0.8),
              scales=True, objects=[_E(name='Lava', shape='box', position=(-0.8, 0.05, 0.0), size=(0.12, 0.05, 0.2), velocity=(0.6, 0.0, 0.0),
                          vel_blend=1.0, noise=0.0, embers=False)]),
    Component('snow', 'Snow', 'Weather', 'Snow falling on the scene and lying on the ground.', 'snow', 'liquid',
              scene={'weather': {'precip': 'snow'}}),
    Component('sleet', 'Sleet', 'Weather', 'Ice pellets bouncing off the ground.', 'weather', 'liquid',
              scene={'weather': {'precip': 'sleet'}}),
    Component('hail', 'Hail', 'Weather', 'Hailstones that bounce, roll and pile up.', 'weather', 'liquid',
              scene={'weather': {'precip': 'hail', 'size': 0.018}}),
    Component('freezing', 'Freezing rain', 'Weather', 'Rain that freezes where it lands, glazing everything in ice.', 'weather',
              'liquid', scene={'weather': {'precip': 'freezing_rain'}}),
    # -- objects ------------------------------------------------------------------------------------------------------------------
    Component('box', 'Box', 'Objects', 'A solid box the fire or water flows around. In your shot it hides what is behind it.', 'cube',
              scales=True, room=(0.4, 0.5, 0.4),
              objects=[_C(name='Box', shape='box', position=(0.6, 0.2, 0.0), size=(0.2, 0.2, 0.2))]),
    Component('ball', 'Ball', 'Objects', 'A solid sphere.', 'cube', scales=True, room=(0.4, 0.5, 0.4),
              objects=[_C(name='Ball', shape='sphere', position=(0.6, 0.2, 0.0), size=(0.2, 0.2, 0.2))]),
    Component('pillar', 'Pillar', 'Objects', 'A solid cylinder standing on the ground.', 'cube', scales=True, room=(0.3, 1.2, 0.3),
              objects=[_C(name='Pillar', shape='cylinder', position=(0.6, 0.5, 0.0), size=(0.1, 0.5, 0.1))]),
    Component('wall', 'Wall', 'Objects', 'A wall behind the effect: smoke pools against it, water hits it.', 'cube', scales=True,
              room=(1.0, 1.4, 0.1),
              objects=[_C(name='Wall', shape='box', position=(0.0, 0.6, -0.7), size=(0.9, 0.6, 0.04))]),
    Component('ramp', 'Ramp', 'Objects', 'A 2 m plank propped up at 20°: roll a ball down it, or drive a cart up '
              'it. Tilt and Roll (Properties › Shape) tip any object over like this.', 'ramp',
              scales=True, room=(1.1, 0.8, 0.5),
              # (its low end's top flush with the ground; its far end on a block)
              objects=[_C(name='Ramp', shape='box', position=(0.0, 0.314, 0.0), size=(1.0, 0.03, 0.4), roll=20.0, material='wood'),
                       _C(name='Ramp prop', shape='box', position=(0.8, 0.268, 0.0), size=(0.1, 0.268, 0.3), material='wood')]),
    Component('room', 'Room with a door', 'Objects', 'A closed room (hollow walls) with a door cut in one side: fire inside starves '
              'of air and flares when it gets out. Turn on tracked air in Combustion for that.', 'cube', scales=True,
              room=(1.1, 1.6, 1.0),
              objects=[_C(name='Room', shape='box', position=(0.0, 0.8, 0.0), size=(0.9, 0.8, 0.8), hollow=0.08,
                          opening=(0.1, 0.55, 0.25), opening_at=(0.85, -0.25, 0.0), holdout=False)]),
    Component('car', 'Car body', 'Objects', 'A car-sized block (4.4 m long, real size).', 'cube', room=(2.6, 2.0, 1.2),
              objects=[_C(name='Car body', shape='box', position=(0.0, 0.72, 0.0), size=(2.2, 0.72, 0.95))]),
    Component('crate', 'Floating crate', 'Objects', 'A wooden crate that floats, bobs and drifts on the water.', 'cube', 'liquid',
              room=(0.3, 0.8, 0.3),
              objects=[_C(name='Crate', shape='box', position=(0.0, 0.6, 0.0), size=(0.12, 0.12, 0.12), yaw=20.0, floating=True,
                          material='wood')]),
    Component('stone', 'Falling stone', 'Objects', 'A stone dropped from 60 cm: it splashes and sinks.', 'cube', 'liquid',
              room=(0.3, 0.8, 0.3),
              objects=[_C(name='Stone', shape='sphere', position=(0.0, 0.6, 0.0), size=(0.07, 0.07, 0.07), floating=True,
                          material='stone')]),
    Component('hill', 'Hillside', 'Objects', 'A built-in hillside (8 m across) for fire to climb or water to run down.', 'cube',
              room=(4.2, 3.6, 4.2),
              objects=[_C(name='Hillside', shape='mesh', mesh='builtin:hillside.png', position=(0.0, 0.0, 0.0), size=(8.0, 3.2, 8.0))]),
    Component('basin', 'Pond basin', 'Objects', 'A built-in pond basin with sloping banks (3.2 m across).', 'cube',
              room=(1.7, 0.6, 1.3),
              objects=[_C(name='Banks', shape='mesh', mesh='builtin:pond_basin.png', position=(0.0, 0.0, 0.0), size=(3.2, 0.4, 2.4))]),
    Component('logs', 'Firewood', 'Objects', 'A built-in pile of firewood logs (a solid mesh).', 'cube', room=(0.6, 0.6, 0.6),
              objects=[_C(name='Firewood', shape='mesh', mesh='builtin:firewood.obj', position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))]),
    Component('hot_plate', 'Hot plate', 'Objects', 'A 30 cm steel plate at 200 °C: chocolate, wax and '
              'metal on it melt, snow on it melts from below, and water on it boils (with Heat and phase changes on).',
              'cube', room=(0.15, 0.02, 0.15),
              objects=[_C(name='Hot plate', shape='box', position=(0.0, 0.01, 0.0), size=(0.15, 0.01, 0.15), material='steel',
                          temperature=200.0, keeps_temperature=True)]),
    Component('text', 'Text', 'Objects', 'Solid letters in any font on this computer. Set them on fire (they catch all '
              'over, flare up and burn out), float them, pour water over them.', 'text', pick='text',
              objects=[_C(name='Text', shape='mesh', position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))]),
    Component('logo', 'Logo or picture', 'Objects', 'Any logo or picture (SVG, PNG, JPG) as a solid shape, traced from it. '
              'Set it on fire, float it, pour water over it.', 'shape', pick='shape',
              objects=[_C(name='Logo', shape='mesh', position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))]),
    Component('mesh', 'Your mesh…', 'Objects', 'Any OBJ or STL model, a numbered mesh sequence, a USD mesh or a greyscale heightfield '
              'image (terrain).', 'cube', pick='mesh',
              objects=[_C(name='Mesh', shape='mesh', position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))]),
    # -- things that fall (engine/solids.py: real size, in every kind of scene) ------------------------------------------------
    Component('drop_box', 'Falling box', 'Things that fall', 'A 30 cm wooden box dropped from 1.2 m: it lands on a corner, '
              'tumbles and settles. Smoke and water push it; set its material for a heavier or bouncier one.', 'crate',
              room=(0.5, 1.4, 0.5),
              objects=[_C(name='Box', shape='box', position=(0.0, 1.2, 0.0), size=(0.15, 0.15, 0.15), yaw=30.0, dynamic=True,
                          material='wood', start_spin=(40.0, 0.0, 25.0))]),
    Component('bouncy_ball', 'Bouncy ball', 'Things that fall', 'A rubber ball dropped from 1.5 m: it bounces, lower each time.',
              'ball', room=(0.4, 1.7, 0.4),
              objects=[_C(name='Rubber ball', shape='sphere', position=(0.0, 1.5, 0.0), size=(0.06, 0.06, 0.06), dynamic=True,
                          material='rubber', own_colour=True, colour=(0.7, 0.12, 0.05))]),
    Component('boulder', 'Falling boulder', 'Things that fall', 'A 1 m granite boulder dropped from 3 m: about 1.4 tonnes. It '
              'flattens what it lands on and blasts the smoke or the water aside.', 'ball', room=(0.8, 3.8, 0.8),
              objects=[_C(name='Boulder', shape='sphere', position=(0.0, 3.0, 0.0), size=(0.5, 0.5, 0.5), dynamic=True,
                          material='stone')]),
    Component('cannonball', 'Thrown ball', 'Things that fall', 'A 20 cm steel ball thrown sideways at 7 m/s: aim it at a tower '
              'of blocks or a stack of crates.', 'ball', room=(0.4, 1.2, 0.4),
              objects=[_C(name='Steel ball', shape='sphere', position=(0.0, 0.9, 0.0), size=(0.1, 0.1, 0.1), dynamic=True,
                          material='steel', start_velocity=(7.0, 1.0, 0.0))]),
    Component('drum', 'Steel drum', 'Things that fall', 'A standing 200 litre steel drum (empty: 20 kg). It tips over when '
              'something hits it.', 'pillar', room=(0.4, 1.0, 0.4),
              objects=[_C(name='Drum', shape='cylinder', position=(0.0, 0.44, 0.0), size=(0.29, 0.44, 0.29), dynamic=True,
                          material='steel', density=170.0, own_colour=True, colour=(0.05, 0.12, 0.32))]),
    Component('dominoes', 'Domino run', 'Things that fall', 'Ten big wooden dominoes in a row, the first one flicked over: each '
              'knocks the next down.', 'wall', room=(1.4, 0.4, 0.2),
              # (the flick: turning about its foot at 160 degrees a second, so its middle moves at 34 cm/s)
              objects=[_C(name=f'Domino {k + 1}', shape='box', position=(-1.2 + 0.16 * k, 0.12, 0.0), size=(0.02, 0.12, 0.06),
                          dynamic=True, material='wood',
                          **({'start_spin': (0.0, 0.0, -160.0), 'start_velocity': (0.34, 0.06, 0.0)} if k == 0 else {}))
                       for k in range(10)]),
    Component('tower', 'Tower of blocks', 'Things that fall', 'Six 20 cm wooden blocks stacked 1.2 m high. Throw a ball at it, '
              'or put it on a moving collider.', 'wall', room=(0.3, 1.3, 0.3),
              objects=[_C(name=f'Block {k + 1}', shape='box', position=(0.0, 0.1 + 0.2 * k, 0.0), size=(0.1, 0.1, 0.1),
                          yaw=(7.0 if k % 2 else -4.0), dynamic=True, material='wood') for k in range(6)]),
    Component('crate_stack', 'Stack of crates', 'Things that fall', 'Six 50 cm crates stacked three, two, one. Knock out the '
              'bottom one, or push the stack with a moving collider.', 'crate', room=(0.9, 1.6, 0.4),
              objects=[_C(name=f'Crate {k + 1}', shape='box', position=pos, size=(0.25, 0.25, 0.25), dynamic=True,
                          material='wood', density=150.0)
                       for k, pos in enumerate(((-0.52, 0.25, 0.0), (0.0, 0.25, 0.0), (0.52, 0.25, 0.0), (-0.26, 0.75, 0.0),
                                                (0.26, 0.75, 0.0), (0.0, 1.25, 0.0)))]),
    Component('ramp_ball', 'Ball down a ramp', 'Things that fall', 'A steel ball let go at the top of a 2 m ramp: it rolls down '
              'and bowls over a row of dominoes.', 'ball', room=(2.4, 0.8, 0.5),
              objects=[_C(name='Ramp', shape='box', position=(0.0, 0.314, 0.0), size=(1.0, 0.03, 0.4), roll=20.0, material='wood'),
                       _C(name='Ramp prop', shape='box', position=(0.8, 0.268, 0.0), size=(0.1, 0.268, 0.3), material='wood'),
                       _C(name='Ball', shape='sphere', position=(0.76, 0.71, 0.0), size=(0.08, 0.08, 0.08), dynamic=True,
                          material='steel')]
              + [_C(name=f'Domino {k + 1}', shape='box', position=(round(-1.3 - 0.16 * k, 2), 0.12, 0.0), size=(0.02, 0.12, 0.06),
                    dynamic=True, material='wood') for k in range(6)]),
    # -- things that break (engine/fracture.py, solids.py) --------------------------------------------------------------------
    Component('brick_wall', 'Brick wall', 'Things that fall', 'A 1.6 m brick wall, half a brick thick, in running bond: it '
              'stands until something hits it hard enough, then it comes apart at the mortar, brick by brick, in a puff of '
              'dust. Throw a ball at it.', 'wall', room=(0.9, 1.3, 0.2),
              objects=[_C(name='Brick wall', shape='box', position=(0.0, 0.6, 0.0), size=(0.8, 0.6, 0.05), material='brick',
                          breakable=True, fracture='bricks')]),
    Component('glass_pane', 'Glass pane', 'Things that fall', 'A 1 m pane of window glass held in its frame: something thrown '
              'through it punches a hole and the shards fall out.', 'window', room=(0.6, 1.5, 0.1),
              objects=[_C(name='Glass pane', shape='box', position=(0.0, 0.9, 0.0), size=(0.5, 0.5, 0.004), material='glass',
                          breakable=True, fracture='shards', pieces=40, held='edges')]),
    Component('pillar_concrete', 'Concrete pillar', 'Things that fall', 'A 2 m concrete pillar standing on the ground: hit it hard '
              'and it breaks into chunks and topples.', 'pillar', room=(0.3, 2.1, 0.3),
              objects=[_C(name='Concrete pillar', shape='box', position=(0.0, 1.0, 0.0), size=(0.15, 1.0, 0.15), material='concrete',
                          breakable=True, pieces=30)]),
    Component('crate_break', 'Wooden crate (breaks)', 'Things that fall', 'A hollow wooden crate dropped from 2 m: it lands '
              'on a corner and splinters.', 'crate', room=(0.4, 2.4, 0.4),
              objects=[_C(name='Crate', shape='box', position=(0.0, 2.0, 0.0), size=(0.22, 0.18, 0.18), yaw=25.0, hollow=0.015,
                          material='wood', dynamic=True, breakable=True, fracture='splinters', pieces=36, strength=0.2,
                          start_spin=(60.0, 0.0, 40.0))]),
    Component('vase', 'Vase', 'Things that fall', 'A pottery vase knocked off a table (1.1 m): it lands on its rim and shatters.',
              'pillar', room=(0.3, 1.5, 0.3),
              objects=[_C(name='Vase', shape='cylinder', position=(0.0, 1.3, 0.0), size=(0.1, 0.17, 0.1), hollow=0.008,
                          opening=(0.13, 0.01, 0.13), opening_at=(0.0, 0.17, 0.0),   # (open at the top)
                          material='ceramic', own_colour=True, colour=(0.12, 0.25, 0.55), dynamic=True, breakable=True,
                          pieces=40, start_spin=(90.0, 0.0, 40.0))]),
    # -- people and cars (engine/assemblies.py) --------------------------------------------------------------------------
    Component('person', 'Person', 'People and cars', 'A crash-test figure, 1.8 m, standing braced: it stays on its feet until '
              'something hits it hard, then falls as a body does.', 'person', room=(0.8, 2.0, 0.6),
              objects=[_C(name='Person', shape='box', position=(0.0, 0.9, 0.0), size=(0.25, 0.9, 0.15), build='figure',
                          material='person')]),
    Component('car', 'Car', 'People and cars', 'A 4.4 m car on sprung wheels, driven at 30 km/h (its Speed, keyed: 0 '
              'brakes it) and steered (Steer).', 'car', room=(6.0, 1.6, 2.2),
              objects=[_C(name='Car', shape='box', position=(0.0, 0.75, 0.0), size=(2.2, 0.75, 0.9), build='car',
                          material='painted', own_colour=True, colour=(0.55, 0.06, 0.04), drive='rear', drive_speed=30.0)]),
    # -- ropes, springs and hinges (engine/solids.py Joint) -------------------------------------------------------------
    Component('wrecking_ball', 'Wrecking ball', 'Ropes and hinges', 'A 900 kg steel ball on a crane\N{RIGHT SINGLE QUOTATION MARK}s cable, '
              'pulled back and let go: it swings through at 6 m/s. Put a brick wall at the bottom of its swing.', 'ball',
              room=(4.2, 5.6, 1.1),
              objects=[_C(name='Crane mast', shape='box', position=(0.0, 2.7, -0.9), size=(0.12, 2.7, 0.12), material='steel',
                          own_colour=True, colour=(0.75, 0.5, 0.04)),
                       _C(name='Crane jib', shape='box', position=(0.0, 5.45, -0.4), size=(0.1, 0.1, 0.62), material='steel',
                          own_colour=True, colour=(0.75, 0.5, 0.04)),
                       _C(name='Wrecking ball', shape='sphere', position=(-3.727, 2.740, 0.0), size=(0.3, 0.3, 0.3), material='steel',
                          joint='rope', joint_to='Crane jib', joint_to_at=(0.0, -0.1, 0.4), rope_length=4.25, rope_look='cable',
                          rope_thickness=0.03)]),
    Component('rope_swing', 'Rope swing', 'Ropes and hinges', 'A wooden disc seat on a 2.6 m rope from a branch, pulled back '
              'and let go.', 'ball', room=(1.6, 3.4, 0.3),
              objects=[_C(name='Branch', shape='box', position=(0.0, 3.2, 0.0), size=(0.9, 0.07, 0.07), material='wood'),
                       _C(name='Swing seat', shape='cylinder', position=(-1.3125, 0.857, 0.0), size=(0.18, 0.025, 0.18), material='wood',
                          joint='rope', joint_to='Branch', joint_to_at=(0.0, -0.07, 0.0), rope_length=2.6, rope_thickness=0.03)]),
    Component('door_hinged', 'Door on hinges', 'Ropes and hinges', 'A door in its frame, pushed open: it swings on its hinges and '
              'slows to a stop. Smoke and water push it; a blast slams it.', 'wall', room=(0.6, 2.2, 1.0),
              objects=[_C(name='Door', shape='box', position=(0.0, 1.015, 0.0), size=(0.44, 1.0, 0.02), material='painted',
                          own_colour=True, colour=(0.55, 0.12, 0.08), joint='hinge', joint_at=(-0.44, 0.0, 0.0),
                          joint_axis=(0.0, 1.0, 0.0), joint_friction=0.5, start_spin=(0.0, 75.0, 0.0)),
                       _C(name='Door frame left', shape='box', position=(-0.5, 1.035, 0.0), size=(0.03, 1.035, 0.05), material='wood'),
                       _C(name='Door frame right', shape='box', position=(0.5, 1.035, 0.0), size=(0.03, 1.035, 0.05), material='wood'),
                       _C(name='Door frame top', shape='box', position=(0.0, 2.1, 0.0), size=(0.53, 0.03, 0.05), material='wood')]),
    Component('hanging_lamp', 'Hanging lamp', 'Ropes and hinges', 'A lamp on a 75 cm flex from the ceiling, swaying: its light '
              'swings with it through the smoke.', 'bulb', room=(0.5, 2.7, 0.5),
              objects=[_C(name='Lamp shade', shape='cylinder', position=(0.0, 1.75, 0.0), size=(0.16, 0.07, 0.16), material='painted',
                          own_colour=True, colour=(0.08, 0.25, 0.14), joint='rope', joint_to='Ceiling rose',
                          joint_to_at=(0.0, -0.02, 0.0), rope_look='cable', rope_thickness=0.006, start_velocity=(0.8, 0.0, 0.3)),
                       _C(name='Ceiling rose', shape='box', position=(0.0, 2.6, 0.0), size=(0.06, 0.02, 0.06), material='painted'),
                       _L(name='Lamp light', kind='point', position=(0.0, 1.66, 0.0), intensity=130.0, temperature=2700.0,
                          colour=(1.0, 1.0, 1.0))],
              links=[{'child': ['light', 'Lamp light'], 'parent': ['collider', 'Lamp shade'], 'offset': [0.0, -0.09, 0.0]}]),
    Component('spring_weight', 'Weight on a spring', 'Ropes and hinges', 'A 38 kg steel weight bouncing on a spring under a bracket, '
              'once a second.', 'drop', room=(0.3, 2.4, 0.2),
              objects=[_C(name='Weight', shape='box', position=(0.0, 1.05, 0.0), size=(0.1, 0.06, 0.1), material='steel',
                          joint='spring', joint_to='Bracket', joint_to_at=(0.0, -0.04, 0.0), rope_length=0.6, spring_k=1500.0,
                          rope_thickness=0.01),
                       _C(name='Bracket', shape='box', position=(0.0, 2.2, 0.0), size=(0.25, 0.04, 0.12), material='steel')]),
    Component('seesaw', 'Seesaw', 'Ropes and hinges', 'A plank on a pivot with a rubber ball on one end; a steel ball dropped on '
              'the other end flips it into the air.', 'wall', room=(1.3, 1.7, 0.3),
              objects=[_C(name='Plank', shape='box', position=(0.0, 0.43, 0.0), size=(1.2, 0.025, 0.15), material='wood',
                          joint='hinge', joint_at=(0.0, -0.025, 0.0), joint_axis=(0.0, 0.0, 1.0), joint_friction=0.1),
                       _C(name='Pivot', shape='box', position=(0.0, 0.2, 0.0), size=(0.08, 0.2, 0.2), material='wood'),
                       _C(name='Rubber ball', shape='sphere', position=(-1.05, 0.52, 0.0), size=(0.065, 0.065, 0.065), dynamic=True,
                          material='rubber', own_colour=True, colour=(0.7, 0.12, 0.05)),
                       _C(name='Steel ball', shape='sphere', position=(1.0, 1.5, 0.0), size=(0.12, 0.12, 0.12), dynamic=True,
                          material='steel')]),
    # -- machines: hinges driven by motors (Properties › Joint › Motor speed) ---------------------------------------------
    Component('cart', 'Motor cart', 'Machines', 'A 1 m wooden cart on four rubber wheels, each turned by a motor at 30 rpm: it '
              'drives off at half a metre a second. Change its wheels’ Motor speed, or put a ramp in its way.', 'cart',
              room=(0.6, 0.4, 0.5),
              # (its axles along -z: Tilt -90 turns a wheel's own axis onto it, so turning forward drives it along +x)
              objects=[_C(name='Cart', shape='box', position=(0.0, 0.25, 0.0), size=(0.5, 0.06, 0.3), material='wood', dynamic=True)]
              + [_C(name=f'Cart wheel {k + 1}', shape='cylinder', position=(sx * 0.35, 0.15, sz * 0.36), size=(0.15, 0.04, 0.15),
                    pitch=-90.0, material='rubber', own_colour=True, colour=(0.05, 0.05, 0.05), joint='hinge', joint_to='Cart',
                    joint_axis=(0.0, 1.0, 0.0), motor_speed=30.0, motor_torque=20.0, joint_friction=0.0)
                 for k, (sx, sz) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1)))]),
    Component('turntable', 'Turntable', 'Machines', 'A wooden turntable a metre across that a motor spins up to 45 rpm, with three '
              'blocks on it: they ride round, then slide off and fly.', 'turntable', room=(1.6, 0.5, 1.6),
              objects=[_C(name='Turntable', shape='cylinder', position=(0.0, 0.33, 0.0), size=(0.5, 0.03, 0.5), material='wood',
                          joint='hinge', joint_axis=(0.0, 1.0, 0.0), motor_speed=45.0, motor_torque=6.0, joint_friction=0.0),
                       # (a centimetre below it: rubbing on its stand would stall it)
                       _C(name='Turntable stand', shape='cylinder', position=(0.0, 0.145, 0.0), size=(0.12, 0.145, 0.12),
                          material='steel')]
              + [_C(name=f'Block {k + 1}', shape='box', position=pos, size=(0.06, 0.06, 0.06), yaw=yaw, dynamic=True, material='wood')
                 for k, (pos, yaw) in enumerate((((0.32, 0.42, 0.0), 0.0), ((-0.16, 0.42, 0.277), -120.0),
                                                 ((-0.16, 0.42, -0.277), 120.0)))]),
    Component('windmill', 'Windmill', 'Machines', 'A 4 m windmill whose sails a motor turns at 12 rpm: they stir the smoke and '
              'knock what comes near.', 'windmill', room=(1.4, 4.2, 0.4),
              # (two balanced bars of sails crossed on the hub, one just behind the other: four sails on hinges of their own
              # would each be pulled out of step by their weight)
              objects=[_C(name='Windmill tower', shape='box', position=(0.0, 1.45, 0.0), size=(0.15, 1.45, 0.15), material='wood'),
                       _C(name='Windmill hub', shape='cylinder', position=(0.0, 2.8, 0.28), size=(0.1, 0.02, 0.1), pitch=90.0,
                          material='wood')]
              + [_C(name=f'Sails {k + 1}', shape='box', position=(0.0, 2.8, z), size=(1.25, 0.12, 0.01), roll=roll,
                    material='painted', own_colour=True, colour=(0.85, 0.82, 0.72), joint='hinge', joint_axis=(0.0, 0.0, 1.0),
                    motor_speed=12.0, motor_torque=400.0)
                 for k, (z, roll) in enumerate(((0.2, 45.0), (0.235, 135.0)))]),
    # -- grass and plants (engine/strands.py: real size, in every kind of scene but the sky) ---------------------------------
    Component('lawn', 'Lawn', 'Grass & plants', 'A 2 m square of short, thick lawn grass: it ripples in the wind and is '
              'flattened where things roll over it.', 'grass', room=(1.1, 0.2, 1.1),
              objects=[_S(name='Lawn', kind='lawn', position=(0.0, 0.0, 0.0), size=(1.0, 0.08, 1.0))]),
    Component('meadow', 'Long grass', 'Grass & plants', 'A 3 \N{MULTIPLICATION SIGN} 2 m patch of long grass, 45 cm tall: it bends '
              'in the wind in waves, parts round what moves through it, and fire runs through it, faster downwind.', 'grass',
              room=(1.6, 0.6, 1.1),
              objects=[_S(name='Long grass', kind='meadow', position=(0.0, 0.0, 0.0), size=(1.5, 0.45, 1.0))]),
    Component('dry_grass', 'Dry grass', 'Grass & plants', 'A 4 \N{MULTIPLICATION SIGN} 3 m patch of grass dried to straw: a spark '
              'sets it alight and the wind drives the fire through it, leaving black stubble. Drop a torch in it.', 'grass',
              room=(2.1, 0.6, 1.6),
              objects=[_S(name='Dry grass', kind='meadow', position=(0.0, 0.0, 0.0), size=(2.0, 0.4, 1.5), dryness=0.9)]),
    Component('wheat', 'Wheat', 'Grass & plants', 'A 3 \N{MULTIPLICATION SIGN} 2 m patch of ripe wheat, 90 cm tall with its ears: '
              'it sways stiffly in the wind and burns fast.', 'wheat', room=(1.6, 1.1, 1.1),
              objects=[_S(name='Wheat', kind='wheat', position=(0.0, 0.0, 0.0), size=(1.5, 0.9, 1.0), dryness=0.95)]),
    Component('reeds', 'Reeds', 'Grass & plants', 'A clump of reeds 1.5 m tall on a 1.2 m disc: they lean and whip in the wind. '
              'Put them at the water\N{RIGHT SINGLE QUOTATION MARK}s edge.', 'reeds', room=(0.7, 1.7, 0.7),
              objects=[_S(name='Reeds', kind='reeds', shape='disc', position=(0.0, 0.0, 0.0), size=(0.6, 1.5, 0.6))]),
    Component('grassy_hill', 'Grassy hillside', 'Grass & plants', 'The built-in hillside (8 m across) covered in long grass that '
              'grows on its slopes: a hillside for a grass fire to climb.', 'hill', room=(4.2, 3.8, 4.2),
              objects=[_C(name='Hillside', shape='mesh', mesh='builtin:hillside.png', position=(0.0, 0.0, 0.0), size=(8.0, 3.2, 8.0),
                          material='earth'),
                       _S(name='Hillside grass', kind='meadow', position=(0.0, 0.0, 0.0), size=(3.9, 0.4, 3.9), grows_on='everything',
                          thickness=0.6, dryness=0.6)]),
    # -- sand, snow, mud, jelly, clay (engine/matter.py: real size, in every kind of scene but the sky) -----------------------
    Component('sand_pile', 'Sand pile', 'Sand, snow & mud', 'A heap of dry sand 60 cm across, at its angle of repose: knock '
              'into it or drop something on it and it slides.', 'matter', room=(0.4, 0.3, 0.4),
              objects=[_M(name='Sand pile', material='sand', shape='pile', position=(0.0, 0.1, 0.0), size=(0.3, 0.1, 0.3))]),
    Component('sand_pour', 'Sand pour', 'Sand, snow & mud', 'Sand poured from 80 cm up at a litre a second for four seconds: '
              'it builds a heap that slides as it grows.', 'pour', room=(0.4, 1.0, 0.4),
              objects=[_M(name='Sand pour', material='sand', pours=True, position=(0.0, 0.8, 0.0), size=(0.03, 0.03, 0.03),
                          velocity=(0.0, -0.5, 0.0), rate=1.0, pour_start=0.0, pour_stop=4.0)]),
    Component('sand_column', 'Sand column', 'Sand, snow & mud', 'A 60 cm column of sand held up, then let go at half a second: '
              'it collapses and spreads.', 'pillar', room=(0.6, 0.7, 0.6),
              objects=[_M(name='Sand column', material='sand', shape='cylinder', position=(0.0, 0.3, 0.0), size=(0.12, 0.3, 0.12),
                          release=0.5)]),
    Component('sand_castle', 'Sand castle', 'Sand, snow & mud', 'A castle of damp sand 32 cm across, with four towers and a '
              'keep: it stands until water soaks into it, then slumps. Give it a pond, a pour or a wave.', 'matter',
              room=(0.4, 0.4, 0.4),
              objects=[_M(name='Castle', material='wet_sand', shape='box', position=(0.0, 0.07, 0.0), size=(0.16, 0.07, 0.16)),
                       _M(name='Keep', material='wet_sand', shape='cylinder', position=(0.0, 0.24, 0.0), size=(0.065, 0.1, 0.065))]
              + [_M(name=f'Tower {k + 1}', material='wet_sand', shape='cylinder', position=(sx * 0.115, 0.215, sz * 0.115),
                    size=(0.04, 0.075, 0.04)) for k, (sx, sz) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1)))]),
    Component('sand_sling', 'Sand into a sling', 'Sand, snow & mud', 'A cotton sheet tied to four posts 50 cm up, with sand '
              'poured onto it from 95 cm for three seconds: it heaps in the dip it makes and weighs the sheet down.', 'fabric',
              room=(0.5, 1.0, 0.5),
              objects=[_C(name=f'Post {k + 1}', shape='cylinder', position=(sx * 0.47, 0.25, sz * 0.47), size=(0.02, 0.25, 0.02),
                          material='wood') for k, (sx, sz) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1)))]
              + [_F(name='Sling', position=(0.0, 0.5, 0.0), width=0.9, height=0.9, orientation='lying', pins='corners',
                    material='cotton', colour=(0.62, 0.15, 0.1)),
                 _M(name='Sand', material='sand', pours=True, position=(0.0, 0.95, 0.0), size=(0.03, 0.03, 0.03),
                    velocity=(0.0, -0.5, 0.0), rate=1.0, pour_start=0.0, pour_stop=3.0)]),
    Component('chocolate_pan', 'Chocolate in a hot pan', 'Sand, snow & mud', 'Squares of chocolate in a 24 cm steel pan at '
              '180 °C on the hob: they melt where they touch it and run into glossy pools (Heat speed 60: some sixty times '
              'quicker than for real).', 'matter', room=(0.3, 0.05, 0.13), scene={'domain': {'matter_heat_speed': 60.0}},
              objects=[_C(name='Pan', shape='cylinder', position=(0.0, 0.02, 0.0), size=(0.12, 0.02, 0.12), hollow=0.004,
                          opening=(0.13, 0.01, 0.13), opening_at=(0.0, 0.02, 0.0), material='steel', temperature=180.0,
                          keeps_temperature=True),
                       _C(name='Handle', shape='box', position=(0.21, 0.03, 0.0), size=(0.09, 0.006, 0.013), material='steel')]
              + [_M(name=f'Square {k + 1}', material='chocolate', shape='box', position=(x, y, z), size=(0.02, 0.004, 0.02),
                    yaw=a) for k, (x, y, z, a) in enumerate(((-0.05, 0.010, 0.03, 10.0), (0.03, 0.010, 0.04, -20.0),
                                                            (0.0, 0.010, -0.05, 35.0), (-0.01, 0.019, 0.0, 5.0),
                                                            (0.06, 0.010, -0.02, 50.0)))]),
    Component('matter_shape', 'Matter shape', 'Sand, snow & mud', 'A mesh of your own (OBJ, STL or a USD prim) filled with '
              'clay: choose what it is made of in Properties (chocolate, wax, jelly, sand, snow...).', 'matter', pick='mesh',
              objects=[_M(name='Shape', material='clay', shape='mesh', position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))]),
    Component('text_chocolate', 'Chocolate letters', 'Sand, snow & mud', 'Words in chocolate, in any font on this computer, '
              'standing on the ground: put a fire beside them and they soften and run.', 'text', pick='text',
              objects=[_M(name='Chocolate', material='chocolate', shape='mesh', position=(0.0, 0.0, 0.0),
                          size=(1.0, 1.0, 1.0))]),
    Component('snowball', 'Snowball', 'Sand, snow & mud', 'A 14 cm snowball of packing snow thrown at 6 m/s: it splats on '
              'what it hits and breaks into lumps.', 'snowball', room=(0.3, 0.8, 0.3),
              objects=[_M(name='Snowball', material='packing_snow', shape='sphere', position=(0.0, 0.7, 0.0),
                          size=(0.07, 0.07, 0.07), velocity=(6.0, 1.0, 0.0))]),
    Component('snow_drift', 'Snow drift', 'Sand, snow & mud', 'A bank of fresh snow 25 cm deep: things that land in it sink in '
              'and pack it.', 'snow', room=(0.7, 0.3, 0.4),
              objects=[_M(name='Snow drift', material='snow', shape='box', position=(0.0, 0.125, 0.0), size=(0.6, 0.125, 0.3))]),
    Component('mud', 'Mud', 'Sand, snow & mud', 'A heap of thick mud let go: it slumps and flows until it is thin enough to '
              'stop.', 'mud', room=(0.8, 0.4, 0.8),
              objects=[_M(name='Mud', material='mud', shape='pile', position=(0.0, 0.15, 0.0), size=(0.3, 0.15, 0.3))]),
    Component('jelly_block', 'Jelly', 'Sand, snow & mud', 'A 16 cm block of jelly dropped from 30 cm: it squashes, bounces, '
              'wobbles and settles.', 'jelly', room=(0.3, 0.6, 0.3),
              objects=[_M(name='Jelly', material='jelly', shape='box', position=(0.0, 0.38, 0.0), size=(0.08, 0.08, 0.08))]),
    Component('clay_lump', 'Lump of clay', 'Sand, snow & mud', 'A 16 cm lump of clay dropped from 30 cm: it lands with a flat '
              'base and keeps it.', 'clay', room=(0.3, 0.6, 0.3),
              objects=[_M(name='Clay', material='clay', shape='box', position=(0.0, 0.38, 0.0), size=(0.08, 0.08, 0.08))]),
    # -- fabric -------------------------------------------------------------------------------------------------------------
    Component('curtain', 'Curtain', 'Fabric', 'A cotton curtain hanging from a rail (real size). It blows in the air and burns.',
              'fabric', room=(0.8, 2.4, 0.9),
              objects=[_F(name='Curtain', width=1.2, height=2.0, position=(0.0, 1.2, -0.6), pins='top', material='cotton',
                          colour=(0.62, 0.12, 0.1))]),
    Component('flag', 'Flag', 'Fabric', 'A nylon flag held along one side, streaming in the wind (give the scene some wind).', 'fabric',
              room=(1.6, 2.6, 0.8),
              objects=[_F(name='Flag', width=1.5, height=1.0, position=(0.75, 2.2, 0.0), pins='side', material='nylon',
                          colour=(0.75, 0.08, 0.06), detail=40)]),
    Component('banner', 'Banner', 'Fabric', 'A banner hung by its top corners.', 'fabric', room=(1.2, 2.4, 0.9),
              objects=[_F(name='Banner', width=2.0, height=0.8, position=(0.0, 2.2, -0.8), pins='top_corners', material='polyester',
                          colour=(0.85, 0.82, 0.75))]),
    Component('sheet', 'Falling sheet', 'Fabric', 'A sheet dropped flat from 1.5 m: it drapes over whatever is under it.', 'fabric',
              room=(1.0, 1.8, 1.0),
              objects=[_F(name='Sheet', width=1.6, height=1.6, orientation='lying', position=(0.0, 1.5, 0.0), pins='none',
                          material='cotton', colour=(0.82, 0.8, 0.76))]),
    Component('towel', 'Wet towel', 'Fabric', 'A soaked tea towel: it drips, steams over a fire and will not burn until it dries.',
              'fabric', room=(0.5, 1.4, 0.5),
              objects=[_F(name='Wet towel', width=0.5, height=0.7, position=(0.0, 1.2, 0.0), pins='top', material='cotton',
                          colour=(0.85, 0.85, 0.8), wetness=1.0)]),
    Component('canopy', 'Canopy', 'Fabric', 'A cloth held up by its four corners, sagging in the middle.', 'fabric',
              room=(1.0, 2.4, 1.0),
              objects=[_F(name='Canopy', width=1.6, height=1.6, orientation='lying', position=(0.0, 2.0, 0.0), pins='corners',
                          material='canvas', colour=(0.78, 0.74, 0.62))]),
    Component('tablecloth', 'Tablecloth', 'Fabric', 'A tablecloth dropped onto a table: it drapes over the edges.', 'fabric',
              room=(0.9, 1.4, 0.9),
              objects=[_F(name='Tablecloth', width=1.4, height=1.4, orientation='lying', position=(0.0, 1.0, 0.0), pins='none',
                          material='linen', colour=(0.9, 0.88, 0.82)),
                       _C(name='Table', shape='box', position=(0.0, 0.37, 0.0), size=(0.45, 0.37, 0.45))]),
    Component('cloth_mesh', 'Your cloth mesh…', 'Fabric', 'Any OBJ model simulated as cloth: a shirt, a tent, a sail.', 'mesh',
              pick='mesh', objects=[_F(name='Cloth', shape='mesh', position=(0.0, 1.0, 0.0), pins='none')]),
    # -- forces: air pushed around (it carries smoke, flame, steam, embers and cloth) ------------------------
    Component('fan', 'Fan', 'Forces', 'A box that blows air one way, like a fan or a gust through a door: it pushes smoke, flame '
              'and cloth. Turn it to aim it.', 'wind', 'fire', scales=True, room=(1.0, 1.2, 0.6),
              objects=[_E(name='Fan', shape='box', position=(-0.7, 0.5, 0.0), size=(0.15, 0.3, 0.3), fuel=0.0, temperature=0.0,
                          velocity=(3.0, 0.0, 0.0), vel_blend=0.35, noise=0.0, embers=False)]),
    Component('updraft', 'Updraft', 'Forces', 'A column of rising air that lifts smoke, sparks and light cloth.', 'wind', 'fire',
              scales=True, room=(0.6, 2.0, 0.6),
              objects=[_E(name='Updraft', shape='cylinder', position=(0.0, 0.3, 0.0), size=(0.25, 0.3, 0.25), fuel=0.0,
                          temperature=0.0, velocity=(0.0, 3.0, 0.0), vel_blend=0.3, noise=0.0, embers=False)]),
    Component('suction', 'Suction', 'Forces', 'A point that draws the air in from all round, like a vent or an extractor.', 'wind',
              'fire', scales=True, room=(0.8, 1.4, 0.8),
              objects=[_E(name='Suction', shape='sphere', position=(0.0, 0.8, 0.0), size=(0.2, 0.2, 0.2), fuel=0.0, temperature=0.0,
                          radial=-3.0, vel_blend=0.4, noise=0.0, embers=False)]),
    # -- lights ---------------------------------------------------------------------------------------------------------
    Component('lamp', 'Lamp', 'Lights', 'A bare bulb in the set, 2 m up: it lights the smoke and the smoke shadows it.', 'bulb',
              scales=True, objects=[_L(name='Lamp', kind='point', position=(1.2, 2.0, 0.8), intensity=130.0, temperature=2700.0,
                          colour=(1.0, 1.0, 1.0))]),
    Component('spot', 'Spotlight', 'Lights', 'A stage spot aimed down at the effect: its beam shows in the smoke.', 'bulb',
              scales=True, objects=[_L(name='Spotlight', kind='spot', position=(1.5, 3.0, 1.0), direction=(-0.45, -0.8, -0.3), intensity=20000.0,
                          cone=18.0, colour=(1.0, 0.95, 0.88))]),
    Component('lightning', 'Lightning', 'Lights', 'A bolt of lightning from 4 m up striking the ground at half a second: it '
              'flashes three times down a branching channel and lights the set, and sets fire to anything burnable it strikes '
              '(move where it Strikes).', 'sparks', scales=True,
              objects=[_L(name='Lightning', kind='lightning', position=(-0.5, 4.0, -0.3), end=(0.0, 0.0, 0.0), intensity=600000.0,
                          strike_at=0.5, strokes=3, branching=0.6, thickness=0.03, colour=(0.8, 0.85, 1.0))]),
    Component('window', 'Window light', 'Lights', 'Soft light from a window to one side (an area light).', 'bulb',
              scales=True, objects=[_L(name='Window', kind='area', position=(-2.0, 1.6, 0.5), direction=(1.0, -0.2, 0.0), intensity=3000.0,
                          radius=0.6, width=1.0, height=1.2, colour=(0.85, 0.9, 1.0))]),
    Component('wind', 'Wind', 'Forces', 'A steady breeze blowing to screen right. Turn it in Motion › Wind (or Liquid › Wind).', 'wind',
              scene={'motion': {'wind_speed': 1.5}, 'liquid': {'wind_speed': 4.0}, 'atmosphere': {'wind': 8.0}}),
    Component('vortex', 'Vortex', 'Forces', 'Air spinning around a vertical axis: it twists smoke and flame into a column.', 'wind',
              'fire', scales=True, room=(0.8, 2.0, 0.8),
              objects=[_E(name='Vortex', shape='cylinder', position=(0.0, 0.05, 0.0), size=(0.3, 0.05, 0.3), fuel=0.0, temperature=0.0,
                          swirl=2.5, swirl_width=6.0, noise=0.0, embers=False)]),
]

# Guns (engine/ballistics.py): each fires at the spot it is put on, from where a shooter would stand; its muzzle is out
# of the box (only what the bullets hit needs to be in it)
COMPONENTS += [
    Component('pistol', 'Pistol shot', 'Guns & bullets', 'A 9 mm pistol fired once from 5 m at what is on this spot, at '
              'chest height. Its bullet holes glass, wood and thin metal and goes on, splashes on steel, chips concrete '
              'and brick, and knocks over what it hits.', 'pistol', origin=True, room=(0.3, 1.3, 0.3),
              objects=[_G(name='Pistol', round='9mm', position=(0.4, 1.3, 5.0), aim=(0.0, 1.1, 0.0), start=0.5,
                          scatter=0.1)]),
    Component('rifle', 'Rifle shot', 'Guns & bullets', 'A .308 rifle fired once from 25 m: over twice the pistol’s speed, '
              'through a brick, a car door or a tree trunk.', 'rifle', origin=True,
              room=(0.3, 1.3, 0.3),
              objects=[_G(name='Rifle', round='308', position=(1.0, 1.5, 25.0), aim=(0.0, 1.1, 0.0), start=0.5, scatter=0.02,
                          flash=False)]),
    Component('shotgun', 'Shotgun blast', 'Guns & bullets', 'Nine 00 buckshot pellets from 6 m, in a fist-sized cone.', 'shotgun',
              origin=True, room=(0.3, 1.3, 0.3),
              objects=[_G(name='Shotgun', round='buck', position=(0.3, 1.2, 6.0), aim=(0.0, 1.0, 0.0), start=0.5)]),
    Component('burst', 'Machine-gun burst', 'Guns & bullets', 'Twelve 5.56 mm rounds at 750 a minute, with tracers, from '
              '15 m, scattering over a metre.', 'mgburst', origin=True, room=(0.6, 1.3, 0.6),
              objects=[_G(name='Machine gun', round='556', position=(1.5, 1.4, 15.0), aim=(0.0, 0.8, 0.0), start=0.5,
                          count=12, rate=750.0, scatter=0.6, tracer=True)]),
    Component('bottle', 'Glass bottle', 'Guns & bullets', 'A glass bottle standing on the ground, in one piece until it is '
              'hit: then it bursts.', 'pillar', room=(0.1, 0.3, 0.1),
              objects=[_C(name='Bottle', shape='cylinder', position=(0.0, 0.12, 0.0), size=(0.037, 0.12, 0.037), hollow=0.003,
                          opening=(0.045, 0.01, 0.045), opening_at=(0.0, 0.12, 0.0), material='glass', own_colour=True,
                          colour=(0.18, 0.42, 0.2), dynamic=True, breakable=True, pieces=40, density=900.0)]),
    Component('steel_target', 'Steel target', 'Guns & bullets', 'A steel plate hung from a bar on a post: it swings back '
              'when hit and wears a grey star of splashed lead for each bullet, which flashes and throws sparks.', 'wall',
              room=(0.4, 1.4, 0.3),
              objects=[_C(name='Steel target', shape='box', position=(0.0, 0.95, 0.0), size=(0.2, 0.2, 0.008), material='steel',
                          own_colour=True, colour=(0.82, 0.8, 0.74), joint='hinge', joint_to='Target bar',
                          joint_at=(0.0, 0.22, 0.0), joint_axis=(1.0, 0.0, 0.0), joint_friction=0.05),
                       _C(name='Target bar', shape='box', position=(0.0, 1.19, 0.0), size=(0.3, 0.015, 0.015), material='steel'),
                       _C(name='Target post', shape='box', position=(0.0, 0.6, -0.05), size=(0.02, 0.6, 0.02), material='steel')]),
]

# the tile pictures (ui/icons.GLYPHS); blocks not listed keep the one they were made with
GLYPHS = {'campfire': 'logs', 'pool': 'pool', 'line': 'line', 'torch': 'torch', 'gas': 'ring', 'fireball': 'burst', 'jet': 'jet',
          'whirl': 'spiral', 'burnable': 'crate', 'steam': 'steam', 'kettle': 'steam', 'pour': 'pour', 'hose': 'jet',
          'fountain': 'fountain', 'block': 'crate', 'flow': 'waves', 'pond': 'waves', 'ball': 'ball', 'pillar': 'pillar',
          'wall': 'wall', 'room': 'house', 'car': 'car', 'crate': 'crate', 'stone': 'ball', 'hill': 'hill', 'basin': 'hill',
          'logs': 'logs', 'mesh': 'mesh', 'flag': 'flag', 'banner': 'flag', 'spot': 'spot', 'window': 'window', 'vortex': 'spiral',
          'fan': 'jet', 'updraft': 'steam', 'suction': 'burst', 'text_fire': 'text', 'text': 'text', 'text_water': 'text'}
for _c in COMPONENTS:
    _c.glyph = GLYPHS.get(_c.key, _c.glyph)

BY_KEY = {c.key: c for c in COMPONENTS}


def get(key):
    """A building block by key: a built-in one, or one of yours ('user:<name>', scene/blocks.py)."""
    if key in BY_KEY:
        return BY_KEY[key]
    from . import blocks
    c = blocks.get(key)
    if c is None:
        raise KeyError(key)
    return c


def _values(v):
    """The values a setting takes: its keys' if it is animated."""
    from .anim import Curve
    return [k[1] for k in v.keys] if isinstance(v, Curve) else [v]
GROUPS = ['Fire', 'Smoke, steam & sparks', 'Liquids', 'Fabric', 'Weather', 'Forces', 'Objects', 'Things that fall',
          'Ropes and hinges', 'Machines', 'Sand, snow & mud', 'Grass & plants', 'Guns & bullets', 'Lights']


# Making a scene from scratch: (label, box width in metres)
SCALES = {'small': ('Tabletop', 0.6), 'person': ('Person-sized', 2.0), 'large': ('Car or room', 6.0), 'huge': ('Building', 20.0)}
KINDS = {'fire': 'Fire and smoke', 'liquid': 'Liquid', 'both': 'Fire and liquid', 'cloud': 'Sky and clouds'}
KIND_BADGES = {'fire': 'Fire and smoke', 'liquid': 'Liquid', 'both': 'Fire and liquid', 'cloud': 'Sky'}


def has_liquid(scene):
    """Whether a scene has any liquid in it: a source, standing water, rain, snow or something floating."""
    if scene.kind == 'fire':
        return False
    if scene.kind == 'both':
        return any(e.get('emits') in ('liquid', 'lava') for e in scene.emitters) or scene.data['liquid']['water_level'] > 0
    return bool(scene.emitters or scene.data['liquid']['water_level'] > 0 or scene.data['liquid']['rain'] > 0
                or scene.data['weather']['precip'] != 'none' or any(c.get('floating') for c in scene.colliders))


def has_fire(scene):
    """Whether a scene has any fire, smoke or steam source in it."""
    if scene.kind == 'fire':
        return bool(scene.emitters)
    if scene.kind == 'both':
        return any(e.get('emits', 'fire') == 'fire' for e in scene.emitters)
    return False


def target_kind(scene, comp):
    """What the scene must simulate to take a block: the least it can be. An empty scene becomes whatever
    the first source needs; fire and liquid together make a fire-and-liquid scene. None if it cannot."""
    kind, need = scene.kind, comp.need
    if kind == 'cloud':
        return kind if comp.key in IN_SKY else None
    if need == 'any' or kind == need or kind == 'both':
        return kind
    if need == 'lava':
        return 'both'
    if kind == 'fire' and need == 'liquid' and not has_fire(scene):
        return 'liquid'
    if kind == 'liquid' and need == 'fire' and not has_liquid(scene):
        return 'fire'
    return 'both'


def _liquid_look(scene):
    """Light a scene that has just become a liquid one as the liquid scenes are lit (sky and sun for glints),
    unless its lighting was already changed."""
    lt = scene.data['lighting']
    base = defaults('lighting')
    if all(lt[k] == base[k] for k in ('ambient', 'sun_on')):
        lt.update(ambient=(0.55, 0.62, 0.72), sun_on=True, sun_azimuth=-40.0, sun_elevation=35.0)
    scene.data['liquid']['wall_drag'] = max(scene.data['liquid']['wall_drag'], 3.0)
    scene.data['water']['backdrop'] = round(max(1.5, 2.0 * scene_width(scene)), 2)


IN_SKY = {'hill', 'mesh', 'wind'}   # a sky is kilometres across: only terrain and wind make sense in one


def scene_kind_for(kind, need, key=None):
    """The kind a scene must become to take a block that needs `need` (or None if it cannot)."""
    if kind == 'cloud':
        return kind if key in IN_SKY else None
    if need == 'any' or kind == need or kind == 'both':
        return kind
    if need == 'lava':
        return 'both'
    return 'both'   # fire into a liquid scene, a liquid into a fire scene


def scene_width(scene):
    """How big the effect is meant to be (m), from the size of its turbulence, which new_scene and the
    presets set by the size of the fire. Unlike the box, it does not change as blocks make the box grow."""
    tf = float(scene.data['motion']['turb_freq'])
    box = scene.domain_size()[0] * 1.2
    return min((5.0 / max(tf, 1e-3)) ** (1 / 0.9), box) if tf > 0 else box


def _scale_factor(scene):
    return min(max(scene_width(scene) / REF_WIDTH, 0.05), 50.0)


_BOUNDS = {}


def mesh_box(mesh):
    """(min, max) of an OBJ mesh in its own units (a built-in one or a file), or None if it cannot be read cheaply."""
    import numpy as np
    from pathlib import Path
    if not mesh or '#' in str(mesh) or not str(mesh).lower().endswith('.obj'):
        return None
    f = Path(__file__).resolve().parents[1] / 'assets' / 'meshes' / mesh[8:] if mesh.startswith('builtin:') else Path(mesh)
    try:
        key = (str(f), f.stat().st_mtime)
    except OSError:
        return None
    if key not in _BOUNDS:
        try:
            with open(f, 'rb') as fh:
                rows = [ln.split()[1:4] for ln in fh if ln.startswith(b'v ')]
            v = np.array(rows, np.float64)
            _BOUNDS[key] = (v.min(0), v.max(0)) if len(v) else None
        except (OSError, ValueError):
            _BOUNDS[key] = None
    return _BOUNDS[key]


def text_info(mesh):
    """What a Text block's mesh says about itself (text, font, height, depth, extent), or None."""
    import json
    from pathlib import Path
    if not mesh or not str(mesh).lower().endswith('.obj') or str(mesh).startswith('builtin:'):
        return None
    try:
        meta = json.loads(Path(mesh).with_suffix('.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) and ('text' in meta or meta.get('kind') == 'image') else None


def text_label(text):
    """An object name for some words: the first 24 characters, on one line, in quotes."""
    t = ' '.join(str(text).split())
    return f'“{t[:24]}{"…" if len(t) > 24 else ""}”'


def _extent(kind, d):
    """Where an object reaches (fire-local min and max corners), roughly; for an animated one, everywhere it goes."""
    import numpy as np
    from .anim import Curve
    if any(isinstance(d.get(k), Curve) for k in ('position', 'end', 'size')):
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for pos in _values(d.get('position', (0.0, 0.0, 0.0))):
            for size in _values(d.get('size', (0.1, 0.1, 0.1))):
                for end in _values(d['end']) if 'end' in d else [None]:
                    s = dict(d, position=pos, size=size)
                    if end is not None:
                        s['end'] = end
                    a, b = _extent(kind, s)
                    lo, hi = np.minimum(lo, a), np.maximum(hi, b)
        return lo, hi
    p = np.asarray(d.get('position', (0.0, 0.0, 0.0)), float)
    if kind == 'collider' and any(isinstance(d.get(k), Curve) or d.get(k) for k in ('pitch', 'roll')):
        # tipped over: the box round its turned corners (at every key of its turn)
        s = np.abs(np.asarray(d.get('size', (0.1, 0.1, 0.1)), float))
        box = mesh_box(d.get('mesh')) if d.get('shape') == 'mesh' else None
        if box is not None:
            c, r = (box[0] + box[1]) / 2 * s, (box[1] - box[0]) / 2 * s
        else:
            c = np.zeros(3)
            r = np.array([s[0]] * 3) if d.get('shape') == 'sphere' else (np.array([s[0], s[1], s[0]]) if d.get('shape') == 'cylinder' else s)
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for yaw in _values(d.get('yaw', 0.0)):
            for pitch in _values(d.get('pitch', 0.0)):
                for roll in _values(d.get('roll', 0.0)):
                    R = turn_matrix(float(yaw), float(pitch), float(roll))
                    cw, rw = R @ c, np.abs(R) @ r
                    lo, hi = np.minimum(lo, p + cw - rw), np.maximum(hi, p + cw + rw)
        return lo, hi
    if kind == 'emitter' or kind == 'collider':
        s = np.abs(np.asarray(d.get('size', (0.1, 0.1, 0.1)), float))
        box = mesh_box(d.get('mesh')) if d.get('shape') == 'mesh' else None
        if box is not None:
            c, r = (box[0] + box[1]) / 2 * s, (box[1] - box[0]) / 2 * s
            if d.get('yaw'):   # turned: the box round its turned corners
                r = np.array([math.hypot(r[0], r[2]), r[1], math.hypot(r[0], r[2])])
            return p + c - r, p + c + r
        if d.get('shape') == 'mesh' and str(d.get('mesh', '')).endswith('.png'):
            s = s * np.array([0.5, 1.0, 0.5])   # a heightfield's size is its full width, height and depth, from the ground
            return p - s * np.array([1, 0, 1]), p + s
        if d.get('shape') == 'mesh':
            s = s * 0.5   # a model's size is a scale: take it as about a metre across
        lo, hi = p - s, p + s
        if d.get('shape') == 'capsule' and 'end' in d:
            e = np.asarray(d['end'], float)
            lo, hi = np.minimum(lo, e - s), np.maximum(hi, e + s)
        return lo, hi
    if kind == 'fabric':
        w, h = float(d.get('width', 1.0)), float(d.get('height', 1.0))
        if d.get('orientation') == 'lying':
            r = np.array([w / 2, 0.05, h / 2])
            return p - r, p + r
        return p - np.array([w / 2, h / 2, 0.1]), p + np.array([w / 2, h / 2, 0.1])   # a panel's position is its middle
    if kind == 'strands':
        s = np.abs(np.asarray(d.get('size', (1.0, 0.45, 1.0)), float))
        r = np.array([s[0], 0.0, s[0] if d.get('shape') == 'disc' else s[2]])
        if d.get('yaw') and d.get('shape') != 'disc':   # turned: round its turned corners
            r[0] = r[2] = math.hypot(r[0], r[2])
        return p - r, p + r + np.array([0.0, s[1], 0.0])
    if kind == 'shot':
        # (the gun is outside the box, where a shooter stands: only where it is aimed needs to be inside)
        a = np.asarray(d.get('aim', (0.0, 0.0, 0.0)), float)
        return a, a
    if kind == 'matter':
        s = np.abs(np.asarray(d.get('size', (0.25, 0.25, 0.25)), float))
        if d.get('shape') == 'mesh':   # a mesh it fills: Size scales it, as a collider's mesh
            box = mesh_box(d.get('mesh'))
            if box is None:
                return p - s * 0.5, p + s * 0.5   # unread: about a metre across
            c, r = (box[0] + box[1]) / 2 * s, (box[1] - box[0]) / 2 * s
            if d.get('yaw'):   # turned: the box round its turned corners
                r = np.array([math.hypot(r[0], r[2]), r[1], math.hypot(r[0], r[2])])
            return p + c - r, p + c + r
        if d.get('shape') == 'sphere':
            s = np.array([s[0]] * 3)
        elif d.get('shape') in ('cylinder', 'pile'):
            s = np.array([s[0], s[1], s[0]])
        if d.get('pours'):
            # a stream: from the nozzle down to the ground, spreading into a heap round it
            r = max(0.3, 0.6 * float(p[1]))
            return np.array([p[0] - r, 0.0, p[2] - r]), np.array([p[0] + r, p[1] + s[0], p[2] + r])
        return p - s, p + s
    return p, p


def convert_kind(scene: Scene, kind):
    """Turn a scene into another kind, keeping what each emitter emits."""
    old = scene.kind
    if old == kind:
        return
    if kind == 'both':
        for e in scene.emitters:
            if old == 'liquid':
                e['emits'] = 'liquid'
    scene.data['domain']['kind'] = kind


def add(scene: Scene, key, at=None, mesh=None):
    """Add a building block to the scene. `at` is a fire-local ground point (x, z) to put its base on.
    Returns ([(kind, index)], [notes for the user]). Raises ValueError if the scene cannot take it."""
    from .anim import Curve
    comp = get(key)
    notes = []
    target = target_kind(scene, comp)
    if target is None:
        raise ValueError(f'{comp.name} cannot go in a sky scene: skies are kilometres across. Start a fire or liquid scene for it '
                         '(Create › Start from scratch).')
    if target != scene.kind:
        was = scene.kind
        empty = not (scene.emitters or scene.colliders or scene.fabrics or scene.matter)
        convert_kind(scene, target)
        if empty:   # nothing in it yet: the box takes the shape and grid that kind of simulation wants
            _domain_for(scene, target, round(scene_width(scene), 3))
        if target in ('liquid', 'both') and was == 'fire':
            _liquid_look(scene)
        if target == 'both':
            notes.append('Fire and liquid now simulate together in this scene.')
    f = _scale_factor(scene) if comp.scales else 1.0
    if comp.file:
        from . import blocks
        objs = blocks.unpacked(comp)   # its meshes unpacked beside your blocks
    else:
        objs = copy.deepcopy(comp.objects)
    room = comp.room
    info = None
    if comp.pick == 'mesh':
        if not mesh:
            raise ValueError('No mesh chosen.')
        from pathlib import Path
        objs[0][1]['mesh'] = mesh
        objs[0][1]['name'] = Path(mesh.split('#')[0]).stem.rstrip('.#_') or objs[0][1].get('name', 'Mesh')
    elif comp.pick in ('text', 'shape'):
        if not mesh:
            raise ValueError('No text given.' if comp.pick == 'text' else 'No picture chosen.')
        info = text_info(mesh) or {}
        h = float(info.get('height', 0.4))
        if info.get('kind') == 'image':
            from pathlib import Path
            label = text_label(Path(info.get('image', 'shape')).stem)
        else:
            label = text_label(info.get('text', 'Text'))
        dom = scene.data['domain']
        cell = max(scene.domain_size()) / max(16.0, dom['resolution'] * dom['preview_scale'])
        for kind, d in objs:
            if d.get('shape') == 'mesh':
                d['mesh'] = mesh
                d['name'] = label if kind == 'collider' or len(objs) == 1 else f'{label} fire'
                if kind == 'emitter' and d.get('thickness'):
                    d['thickness'] = round(max(0.04 * h, cell), 4)   # a skin of fire about a cell deep
                if kind == 'emitter' and d.get('noise_freq'):
                    d['noise_freq'] = round(d['noise_freq'] * 0.4 / h, 3)   # clumps sized to the letters
        if comp.need == 'liquid':   # water letters hang well above the ground, then fall
            y = round(max(0.75 * h, 0.4 * scene.domain_size()[1]), 3)
            for kind, d in objs:
                d['position'] = (d['position'][0], y, d['position'][2])
        room = (0.0, (1.2 if comp.need == 'liquid' else 5.0) * h, 0.0)   # head room for the flames (or the fall)
    anchor = (0.0, 0.0) if comp.origin else None
    for kind, d in objs:
        if 'position' in d and anchor is None:
            anchor = (_values(d['position'])[0][0] * f, _values(d['position'])[0][2] * f)
            break
    dx = dy = dz = 0.0
    if at is not None and len(at) == 3:   # (x, y, z): a point with a height (a table top, a step, a slope)
        dy = float(at[1])
        at = (at[0], at[2])
    if at is not None and anchor is not None:
        dx, dz = at[0] - anchor[0], at[1] - anchor[1]
    elif at is not None:
        anchor = (0.0, 0.0)
    added = []
    lists = {'emitter': scene.emitters, 'collider': scene.colliders, 'light': scene.lights, 'fabric': scene.fabrics,
             'matter': scene.matter, 'strands': scene.strands, 'shot': scene.shots}
    renamed = {}
    for kind, d in objs:
        d = dict(d)
        for k in ('position', 'end', 'joint_anchor', 'aim'):
            if k in d:
                v = d[k]
                if isinstance(v, Curve):
                    d[k] = Curve([[fr, (x[0] * f + dx, x[1] * f + dy, x[2] * f + dz), it] for fr, x, it in v.keys])
                else:
                    d[k] = (v[0] * f + dx, v[1] * f + dy, v[2] * f + dz)
        if f != 1.0:
            if 'size' in d and not (d.get('shape') == 'mesh'):
                d['size'] = tuple(x * f for x in d['size'])
            if kind == 'collider' and d.get('joint', 'none') != 'none':
                for k in ('joint_at', 'joint_to_at'):
                    if k in d:
                        d[k] = tuple(x * f for x in d[k])
                for k in ('rope_length', 'rope_thickness'):
                    if k in d:
                        d[k] = d[k] * f
            if kind == 'collider' and d.get('hollow'):
                d['hollow'] = d['hollow'] * f
                d['opening'] = tuple(x * f for x in d.get('opening', (0, 0, 0)))
                d['opening_at'] = tuple(x * f for x in d.get('opening_at', (0, 0, 0)))
            if kind == 'emitter' and d.get('noise_freq'):
                d['noise_freq'] = d['noise_freq'] / f
            if kind == 'emitter' and 'velocity' in d:
                d['velocity'] = tuple(x * math.sqrt(f) for x in d['velocity'])   # Froude scaling: the same arc at any size
            if kind == 'emitter' and d.get('radial'):
                d['radial'] = d['radial'] * math.sqrt(f)
            if kind == 'light':
                d['intensity'] = d.get('intensity', 2000.0) * f * f   # as bright at the effect from f times as far
                d['radius'] = d.get('radius', 0.1) * f
                for k in ('width', 'height'):
                    if k in d:
                        d[k] = d[k] * f
        if kind == 'emitter' and scene.kind == 'both' and not comp.origin:   # your blocks keep what each source emits
            d['emits'] = {'fire': 'fire', 'liquid': 'liquid', 'lava': 'lava'}.get(comp.need, 'fire')
        if kind == 'emitter' and (comp.need in ('liquid', 'lava') or d.get('vapour')) and 'size' in d and d.get('shape') != 'mesh' \
                and not comp.origin and not isinstance(d['size'], Curve):
            # a source smaller than the grid's cells pours next to nothing: at least about one cell across
            dom = scene.data['domain']
            cell = max(scene.domain_size()) / max(16.0, dom['resolution'] * dom['preview_scale'])
            small = [x for x in d['size'] if x < cell]
            if small:
                d['size'] = tuple(max(x, cell) for x in d['size'])
                notes.append(f'{d.get("name", comp.name)} is made {2 * cell * 100:.0f} cm across, two cells of the simulation grid, '
                             'so it pours properly; raise Domain › Voxels for a finer grid and a finer source.')
        names = {o['name'] for o in lists[kind]}
        base, n = d.get('name', comp.name), 2
        while d.get('name', base) in names:
            d['name'] = f'{base} {n}'
            n += 1
        renamed[(kind, base)] = d.get('name', base)
        curves = {k: v for k, v in d.items() if isinstance(v, Curve)}   # animated settings go in as they are
        i = {'emitter': scene.add_emitter, 'collider': scene.add_collider, 'light': scene.add_light,
             'fabric': scene.add_fabric, 'matter': scene.add_matter,
             'strands': scene.add_strands, 'shot': scene.add_shot}[kind](**{k: v for k, v in d.items() if k not in curves})
        lists[kind][i].update(curves)
        added.append((kind, i))
    for kind, i in added:   # its joints, to its objects as they are named here
        c = lists[kind][i]
        if kind == 'collider' and c.get('joint_to') and ('collider', c['joint_to']) in renamed:
            c['joint_to'] = renamed[('collider', c['joint_to'])]
    for l in comp.links:   # its attachments, between the objects as they are named here
        c, p = tuple(l['child']), tuple(l['parent'])
        if c in renamed and p in renamed:
            scene.links.append(dict(l, child=[c[0], renamed[c]], parent=[p[0], renamed[p]]))
    # scene-wide settings that come with it (only the sections the scene has)
    from .params import applies
    for sec, vals in comp.scene.items():
        if not applies(sec, None, scene.kind):
            continue
        for k, v in vals.items():
            if applies(sec, k, scene.kind):
                scene.set((sec, k), v)
    if comp.scene.get('spread', {}).get('enabled') and comp.need in ('fire', 'both'):
        notes.append('Spreading fire is on: fire spreads over anything marked Burnable.')
    if comp.pick in ('text', 'shape'):
        text_detail(scene, mesh)
        emitters = [i for k, i in added if k == 'emitter']
        colliders = [i for k, i in added if k == 'collider']
        if emitters and colliders:   # the fire keeps to the letters wherever they are moved, stretched or turned
            scene.links.append({'child': ['emitter', scene.emitters[emitters[0]]['name']],
                                'parent': ['collider', scene.colliders[colliders[0]]['name']], 'offset': [0.0, 0.0, 0.0],
                                'shape': True})
    # grow the box to fit (it keeps its cell count, so detail drops a little)
    grown = _fit_box(scene, [(k, lists[k][i]) for k, i in added], room, f, (anchor[0] + dx, anchor[1] + dz) if anchor else (0.0, 0.0))
    if grown:
        notes.append(f'The simulation box grew to {grown[0]:.2g} × {grown[1]:.2g} × {grown[2]:.2g} m to fit it.')
    return added, notes


# what a source can be turned into (Turn into, in the viewer's menus): the building block it takes its settings from
TURN_INTO = [('burner', 'Fire'), ('smoke', 'Smoke'), ('steam', 'Steam'), ('pour', 'Water'), ('lava', 'Lava'), ('fan', 'Fan'),
             ('updraft', 'Updraft'), ('suction', 'Suction'), ('vortex', 'Vortex')]
GEOMETRY = {'name', 'enabled', 'shape', 'mesh', 'volume', 'volume_mode', 'volume_zup', 'position', 'size', 'end', 'yaw', 'pitch',
            'roll', 'thickness', 'mesh_offset'}


def turn_into(scene: Scene, i, key):
    """Make source i the kind of source the block `key` is (fire, water, a fan...), keeping its shape and place.
    Returns notes. Raises ValueError if the scene cannot take it."""
    comp = BY_KEY[key]
    src = next(d for kind, d in comp.objects if kind == 'emitter')
    notes = []
    target = target_kind(scene, comp)
    if target is None:
        raise ValueError(f'A sky scene has no {comp.name.lower()} sources.')
    if target != scene.kind:
        was = scene.kind
        # this source is changing, so it does not count as fire or liquid that is already there
        others = [e for j, e in enumerate(scene.emitters) if j != i]
        keep = scene.emitters
        scene.emitters = others
        target = target_kind(scene, comp) or target
        scene.emitters = keep
        convert_kind(scene, target)
        if target in ('liquid', 'both') and was == 'fire':
            _liquid_look(scene)
        if target == 'both':
            notes.append('Fire and liquid now simulate together in this scene.')
    from .params import emitter_defaults
    e = scene.emitters[i]
    base = emitter_defaults()
    for k, v in base.items():
        if k not in GEOMETRY:
            e[k] = v
    for k, v in src.items():
        if k not in GEOMETRY:
            e[k] = v
    if scene.kind == 'both':
        e['emits'] = {'fire': 'fire', 'liquid': 'liquid', 'lava': 'lava'}.get(comp.need, 'fire')
    if comp.need in ('liquid', 'lava') and 'velocity' not in src:
        e['velocity'] = (0.0, -0.5, 0.0)
    return notes


def make_dynamic(scene: Scene, i, on=True, release=0.0):
    """Make object (collider) i fall (Make it fall, in the viewer's menus): a free rigid body that falls,
    tumbles, slides and bounces, pushed by the smoke and the water and knocking into other things. Its keys now
    set only where it starts; it is let go `release` seconds after the first frame. on=False keeps it where its
    keys put it again. Returns notes for the user."""
    c = scene.colliders[i]
    notes = []
    c['dynamic'] = bool(on)
    if not on:
        c['floating'] = False
        if c.get('joint', 'none') != 'none':
            c['joint'] = 'none'
            notes.append(f'{c["name"]} is no longer joined to anything.')
        return notes
    c['release'] = float(release)
    c['holdout'] = True   # it is a real thing in the shot
    if c['shape'] == 'mesh':
        notes.append(f'{c["name"]} falls as its convex hull: hollows and dents in it do not catch on things.')
    if float(c.get('hollow', 0.0) or 0.0) > 0.0:
        notes.append(f'{c["name"]} falls as a solid: a hollow thing falls like a full one of its weight (lower its density).')
    from ..engine.solver import MAX_COLLIDERS
    if sum(1 for d in scene.colliders if d['enabled']) > MAX_COLLIDERS:
        notes.append(f'Only the first {MAX_COLLIDERS} objects take part in the simulation.')
    return notes


JOINTS = [('rope', 'A rope'), ('spring', 'A spring'), ('hinge', 'A hinge'), ('ball', 'A ball joint')]


def add_joint(scene: Scene, i, kind='rope', to=None, on=True):
    """Hang object (collider) i on a rope or a spring, or put it on a hinge or a ball joint (Hang it on a rope, Hinge it,
    Join it to..., in the viewer's menus): to object `to` (an index) or, without one, to a fixed point. It falls from
    then on, held by its joint. A rope or a spring goes up 1.5 m from its top; a hinge goes along a box's side (a
    door) or its back edge if it is flat (a lid), through the middle of anything else (a wheel); a ball joint is at
    its top. on=False takes the joint away (it still falls). Returns notes for the user."""
    import numpy as np
    from ..engine.solids import surface_toward
    c = scene.colliders[i]
    if not on or kind in (None, 'none'):
        c['joint'] = 'none'
        return [f'{c["name"]} is no longer joined to anything: it falls freely.']
    if kind not in dict(JOINTS):
        raise ValueError(f'No such joint: {kind}')
    notes = []
    c['joint'] = kind
    c['dynamic'] = True
    c['holdout'] = True
    c.setdefault('release', 0.0)
    pos = np.asarray(scene.get(('collider', i, 'position'), scene.start), float)
    size = np.abs(np.asarray(scene.get(('collider', i, 'size'), scene.start), float))
    top = float(size[0]) if c['shape'] == 'sphere' else float(size[1])
    if c['shape'] == 'mesh':
        top = float(size[1]) * 0.5
    other = None
    if to is not None and 0 <= to < len(scene.colliders) and to != i:
        other = scene.colliders[to]
        c['joint_to'] = other['name']
    else:
        c['joint_to'] = ''
    c['rope_length'] = 0.0
    if kind in ('rope', 'spring'):
        c['joint_at'] = (0.0, 0.0, 0.0)          # (where its surface faces the other end)
        if other is not None:
            op = np.asarray(scene.get(('collider', to, 'position'), scene.start), float)
            osz = np.abs(np.asarray(scene.get(('collider', to, 'size'), scene.start), float))
            u = scene.turn(to, scene.start).T @ (pos - op)          # (into its own frame)
            n = float(np.linalg.norm(u))
            c['joint_to_at'] = tuple(float(x) for x in (surface_toward(other['shape'], osz, u / n) if n > 1e-9 else np.zeros(3)))
            notes.append(f'{c["name"]} hangs on a {kind} from {other["name"]}.')
        else:
            c['joint_anchor'] = (float(pos[0]), float(pos[1] + top + 1.5), float(pos[2]))
            notes.append(f'{c["name"]} hangs on a {kind} from a point 1.5 m above it: move Anchor (Properties › Joint) to hang '
                         'it from somewhere else, or join it to an object.')
    elif kind == 'hinge':
        if c['shape'] == 'box' and size[1] <= min(size[0], size[2]):
            c['joint_at'], c['joint_axis'] = (0.0, 0.0, -float(size[2])), (1.0, 0.0, 0.0)          # a lid, a flap
        elif c['shape'] == 'box':
            c['joint_at'], c['joint_axis'] = ((-float(size[0]), 0.0, 0.0) if size[0] >= size[2] else (0.0, 0.0, -float(size[2])),
                                              (0.0, 1.0, 0.0))                                        # a door, a gate
        else:
            c['joint_at'], c['joint_axis'] = (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)                        # a wheel, a turntable
        notes.append(f'{c["name"]} turns on a hinge{" on " + other["name"] if other is not None else ""}: give it a push '
                     '(Physics › Spinning at), or let something knock it.')
    else:
        c['joint_at'] = (0.0, top, 0.0)
        notes.append(f'{c["name"]} swings on a ball joint at its top{" on " + other["name"] if other is not None else ""}.')
    return notes


def make_breakable(scene: Scene, i, on=True, pieces=None, fracture=None):
    """Make object (collider) i breakable (Make it breakable, in the viewer's menus): cut beforehand into pieces
    glued together, which come apart where it is hit or loaded harder than its material holds. The pattern is the
    one its material breaks into unless given: bricks for a brick box, shards for glass, splinters for wood, chunks
    for the rest. Returns notes for the user."""
    from ..scene.materials import material
    import numpy as np
    c = scene.colliders[i]
    notes = []
    if on and c['shape'] == 'mesh':
        notes.append(f'{c["name"]} is a mesh: its pieces are cut from its inside, each a convex chunk (a hollow or a dent in '
                     'one is filled in). Whole, it is drawn as itself.')
    c['breakable'] = bool(on)
    if not on:
        return notes
    m = c.get('material', 'wood')
    if fracture is None:
        if m == 'brick' and c['shape'] == 'box':
            fracture = 'bricks'
        elif material(m).clear > 0.5:
            fracture = 'shards'
        elif m == 'wood' or getattr(material(m), 'pattern', '').startswith('wood'):
            fracture = 'splinters'
        elif material(m).yields > 0.0:
            fracture = 'bends'
        else:
            fracture = 'voronoi'
    c['fracture'] = fracture
    if pieces is not None:
        c['pieces'] = int(pieces)
    c['holdout'] = True
    if c['shape'] != 'box' and fracture == 'bricks':
        c['fracture'] = 'voronoi'
    if fracture == 'shards' and float(np.min(np.abs(np.asarray(c['size'], float)))) > 0.03:
        notes.append(f'{c["name"]} is thick for a pane: glass breaks best as a thin box.')
    if not c.get('dynamic'):
        notes.append(f'{c["name"]} stands where it is until it breaks, held by {dict(base="its base", edges="its edges", free="nothing")[c.get("held", "base")]} '
                     '(Breaking > Held by).')
    return notes


def text_detail(scene, mesh):
    """Fine enough mesh detail for the strokes of a Text block's letters: about three cells across a stroke."""
    info = text_info(mesh)
    if not info:
        return
    ext = info.get('extent') or (1.0, 0.4, 0.1)
    need = int(math.ceil(max(ext) / (0.03 * float(info.get('height', 0.4))))) + 4
    d = scene.data['domain']
    d['mesh_resolution'] = int(min(192, max(int(d['mesh_resolution']), need)))


def _fit_box(scene, objs, room, f, base):
    import numpy as np
    sx, sy, sz = scene.domain_size()
    need = np.array([sx / 2, sy, sz / 2])
    for kind, d in objs:
        if kind == 'light':
            continue   # a light may stand outside the box: it lights it from there
        lo, hi = _extent(kind, d)
        need = np.maximum(need, [max(abs(lo[0]), abs(hi[0])) * 1.1, hi[1] * 1.08, max(abs(lo[2]), abs(hi[2])) * 1.1])
    rx, ry, rz = (x * f for x in room)
    if rx or ry or rz:
        need = np.maximum(need, [abs(base[0]) + rx, ry, abs(base[1]) + rz])
    new = (float(need[0] * 2), float(need[1]), float(need[2] * 2))
    new = tuple(max(round(n + 0.005, 2), o) if n > o * 1.01 else o for n, o in zip(new, (sx, sy, sz)))
    if new == (sx, sy, sz):
        return None
    d = scene.data['domain']
    d['size_x'], d['size_y'], d['size_z'] = new
    return new


def new_scene(kind='auto', scale='person'):
    """An empty scene to build in: the box, solver and look set up for the kind and size of effect. 'auto'
    leaves the kind to what is added first (it starts as the light gas solver, which also moves fabric)."""
    if kind == 'auto':
        s = new_scene('fire', scale)
        s.name = 'New scene'
        return s
    if kind == 'cloud':
        from . import presets
        s = presets.make('cumulus_day')
        s.name = 'Sky'
        s.preset = None
        return s
    w = SCALES[scale][1]
    s = Scene()
    s.emitters = []
    s.name = {'fire': 'New fire', 'liquid': 'New liquid', 'both': 'New fire and liquid'}[kind]
    _domain_for(s, kind, w)
    m = s.data['motion']
    m['turb_freq'] = round(5.0 / w ** 0.9, 3)   # also how big the scene is meant to be (scene_width)
    if kind in ('fire', 'both'):
        # the scale of the turbulence and detail follow the size of the fire (the presets' values, by size)
        m['disturb_block'] = round(0.02 * w, 4)
        m['puffing'] = 0.6
        s.data['shading']['detail_freq'] = round(6.0 / w, 3)
        s.data['embers']['turb_freq'] = round(4.0 / w ** 0.9, 3)
        s.data['lighting']['light_spread'] = round(0.18 * w, 3)
    if kind in ('liquid', 'both'):
        s.data['lighting'].update(ambient=(0.55, 0.62, 0.72), sun_on=True, sun_azimuth=-40.0, sun_elevation=35.0)
        s.data['liquid']['wall_drag'] = 3.0
        s.data['water']['backdrop'] = round(max(1.5, 4.0 * w), 2)
    return s


def _domain_for(s, kind, w):
    """The box, grid, pre-roll and camera for a kind of simulation of an effect w metres across."""
    d = s.data['domain']
    d['kind'] = kind
    if kind == 'fire':
        d['size_x'], d['size_y'], d['size_z'] = w, round(w * 1.6, 3), w
        d['resolution'] = 128
        d['preroll'] = 1.5
        d['substeps_max'] = 12
    elif kind == 'liquid':
        d['size_x'], d['size_y'], d['size_z'] = w, round(w * 0.6, 3), round(w * 0.8, 3)
        d['resolution'] = 160
        d['preroll'] = 0.0
        d['substeps_max'] = 12
        d['cfl'] = 1.5
    else:
        d['size_x'], d['size_y'], d['size_z'] = w, round(w * 1.3, 3), round(w * 0.8, 3)
        d['resolution'] = 144
        d['preroll'] = 1.0
        d['substeps_max'] = 12
        d['cfl'] = 1.5
    c = s.data['camera']
    h = d['size_y']
    c['distance'] = round(max(h * 1.8, w * 1.6), 3)
    c['target_y'] = round(h * 0.4, 3)
    c['near'] = round(max(0.002, 0.01 * w), 4)
    if not (s.footage or s.track):   # with a shot, its matched angle and placement stay
        c['pitch'] = 6.0 if kind == 'fire' else 16.0
        c['anchor_x'], c['anchor_y'] = 0.5, 0.86 if kind == 'fire' else 0.72


def fresh_defaults_check():
    """(for tests) every block's settings are real settings."""
    from .params import param
    for c in COMPONENTS:
        for kind, d in c.objects:
            for k in d:
                param(kind, k)
        for sec, vals in c.scene.items():
            for k in vals:
                param(sec, k)
    return True


__all__ = ['Component', 'COMPONENTS', 'BY_KEY', 'GROUPS', 'SCALES', 'KINDS', 'add', 'new_scene', 'scene_kind_for',
           'convert_kind']
