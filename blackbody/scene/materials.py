"""Real materials for solid objects: how heavy, slippery, bouncy and strong they are, and how they look.

The numbers are handbook values for the material against itself or a hard floor:
- density in kg/m^3;
- friction is the Coulomb coefficient;
- bounce is the coefficient of restitution (how much of the speed comes back off a hard floor);
- strength is the stress, in Pa, at which glued pieces of it come apart (breakable objects).

Hollow everyday objects (a cardboard box, a plastic crate) use an effective density, their weight over the
space they take up. The look is used when an object is drawn as a CG object:
- colour: albedo, linear;
- roughness: 0 for a mirror, 1 for chalk;
- metal: 0 or 1;
- clear: how much light passes through, for glass and ice, and ior its index of refraction;
- pattern: a procedural texture;
- inside: the colour of broken faces.

FLOORS are the stage's floors, for shots without footage (engine/stage.py).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    key: str
    label: str
    density: float      # kg/m^3
    friction: float     # Coulomb coefficient
    bounce: float       # coefficient of restitution
    strength: float     # Pa: where glued pieces come apart
    colour: tuple       # linear albedo
    roughness: float = 0.6
    metal: float = 0.0
    clear: float = 0.0
    pattern: str = 'plain'   # plain, wood, stone, concrete, brick, metal, glass
    inside: tuple = None     # colour of broken faces (None: the colour)
    ior: float = 1.5         # index of refraction (clear materials)


MATERIALS = {m.key: m for m in (
    Material('wood', 'Wood (pine)', 550.0, 0.45, 0.35, 4.0e6, (0.45, 0.29, 0.15), 0.65, pattern='wood', inside=(0.62, 0.45, 0.27)),
    Material('stone', 'Stone (granite)', 2650.0, 0.6, 0.25, 8.0e6, (0.32, 0.31, 0.3), 0.8, pattern='stone', inside=(0.42, 0.41, 0.39)),
    Material('concrete', 'Concrete', 2400.0, 0.65, 0.2, 3.0e6, (0.42, 0.41, 0.38), 0.9, pattern='concrete', inside=(0.5, 0.48, 0.44)),
    Material('brick', 'Brick', 1900.0, 0.6, 0.15, 2.0e6, (0.38, 0.13, 0.07), 0.85, pattern='brick', inside=(0.5, 0.2, 0.11)),
    Material('steel', 'Steel', 7850.0, 0.45, 0.45, 4.0e8, (0.56, 0.57, 0.58), 0.35, metal=1.0, pattern='metal'),
    Material('aluminium', 'Aluminium', 2700.0, 0.45, 0.45, 9.0e7, (0.91, 0.92, 0.92), 0.3, metal=1.0, pattern='metal'),
    Material('glass', 'Glass', 2500.0, 0.5, 0.5, 3.0e7, (0.9, 0.95, 0.94), 0.03, clear=0.92, pattern='glass', inside=(0.75, 0.85, 0.82)),
    Material('ice', 'Ice', 917.0, 0.05, 0.25, 1.0e6, (0.8, 0.9, 0.95), 0.08, clear=0.7, pattern='glass', ior=1.31),
    Material('plastic', 'Plastic', 950.0, 0.35, 0.5, 2.0e7, (0.75, 0.1, 0.06), 0.4),
    Material('rubber', 'Rubber', 1100.0, 0.9, 0.8, 1.5e7, (0.04, 0.04, 0.04), 0.7),
    Material('ceramic', 'Ceramic (pottery)', 2300.0, 0.5, 0.35, 2.0e7, (0.8, 0.77, 0.72), 0.25, inside=(0.66, 0.5, 0.38)),
    Material('cardboard', 'Cardboard box', 120.0, 0.5, 0.15, 5.0e5, (0.48, 0.33, 0.17), 0.9, inside=(0.58, 0.45, 0.28)),
    Material('foam', 'Foam', 40.0, 0.7, 0.3, 2.0e5, (0.85, 0.85, 0.8), 0.95),
    # (a car body or a boat hull is a shell: its weight over the space it takes up)
    Material('painted', 'Painted metal', 300.0, 0.5, 0.4, 4.0e8, (0.32, 0.03, 0.025), 0.25, inside=(0.56, 0.57, 0.58)),
    Material('plaster', 'Plaster wall', 900.0, 0.6, 0.2, 1.0e6, (0.7, 0.68, 0.64), 0.9, inside=(0.8, 0.79, 0.76)),
    Material('fabric', 'Fabric (upholstery)', 250.0, 0.8, 0.2, 1.0e6, (0.36, 0.08, 0.06), 0.95),
    Material('earth', 'Earth (soil)', 1600.0, 0.65, 0.1, 2.0e5, (0.16, 0.12, 0.085), 0.95, pattern='concrete'),
)}

OPTIONS = tuple((k, m.label) for k, m in MATERIALS.items())

# the shader's pattern numbers (stage.wgsl)
PATTERNS = {'plain': 0, 'wood': 1, 'stone': 2, 'concrete': 3, 'brick': 4, 'metal': 5, 'glass': 0}


@dataclass(frozen=True)
class Floor:
    key: str
    label: str
    colour: tuple       # linear albedo
    roughness: float
    pattern: int        # stage.wgsl floor_look


FLOORS = {f.key: f for f in (
    Floor('studio', 'Studio grey', (0.2, 0.2, 0.2), 0.6, 0),
    Floor('concrete', 'Concrete', (0.33, 0.32, 0.3), 0.85, 1),
    Floor('boards', 'Wooden boards', (0.34, 0.21, 0.11), 0.55, 2),
    Floor('tiles', 'Tiles', (0.6, 0.58, 0.54), 0.3, 3),
    Floor('dirt', 'Dirt', (0.17, 0.13, 0.09), 0.95, 4),
    Floor('grass', 'Grass', (0.06, 0.11, 0.035), 0.9, 5),
    Floor('sand', 'Sand', (0.45, 0.38, 0.27), 0.9, 6),
    Floor('checker', 'Checker (1 m)', (0.5, 0.5, 0.5), 0.6, 7),
)}

FLOOR_OPTIONS = tuple((k, f.label) for k, f in FLOORS.items())


def material(key) -> Material:
    """The material for a key (wood when the key is unknown)."""
    return MATERIALS.get(key, MATERIALS['wood'])


def resolved(collider: dict) -> dict:
    """The physical values a collider uses: its own where set, else its material's.
    A density of 0, and a friction or bounce below 0, mean "the material's"."""
    m = material(collider.get('material', 'wood'))
    dens = float(collider.get('density', 0.0) or 0.0)
    fr = float(collider.get('friction', -1.0))
    bo = float(collider.get('bounce', -1.0))
    return dict(density=dens if dens > 0.0 else m.density, friction=fr if fr >= 0.0 else m.friction,
                bounce=bo if bo >= 0.0 else m.bounce, strength=m.strength)
