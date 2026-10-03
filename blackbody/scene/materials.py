"""Real materials for solid objects: how heavy, slippery, bouncy and strong they are, and how they look.

The numbers are handbook values for the material against itself or a hard floor:
- density in kg/m^3;
- friction is the Coulomb coefficient;
- bounce is the coefficient of restitution (how much of the speed comes back off a hard floor);
- strength is the stress, in Pa, at which glued pieces of it come apart under a load (breakable objects): pulled,
  sheared or bent. It is an effective strength (pottery's and glass's near the handbook's, the rest lower), set so
  that a wall stands under its own weight and a wrecking ball brings it down, and a vase tipped off a table holds
  until it lands;
- impact is the speed, in m/s, of a hit that breaks a piece of it away (solids.py: how fast what meets it closes on
  it), so that a stone goes through a window and a dropped vase shatters where it lands, and a touch does not.

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
    dust: float = 0.0        # how much dust it throws up where it breaks (mortar, plaster, soil: about 1)
    effusivity: float = 1500.0  # W s^0.5/m^2/K, sqrt(conductivity x density x heat capacity): how readily it gives heat
                                # to (or takes it from) what touches it (steel a pan's 14000, wood 400, foam 40)
    impact: float = 0.0         # m/s: hit this fast (what meets it closing on it, or it stopped short), a piece of it
                                # breaks away (a pane or a pot dropped on a hard floor from about 30 cm, a brick wall's
                                # mortar from 20 cm). A hit sends a stress wave through it of about density x sound
                                # speed x that speed; this is where that reaches what it holds. 0: never
    brittle: float = 0.0        # how far a crack runs on from a hit (0..1): of a hit that breaks a piece away, the
                                # share that carries to the pieces glued to it, a piece's size further on (glass and
                                # pottery shatter right through, wood splits off where it is hit, metal dents)
    yields: float = 0.0         # Pa: loaded past this (but short of its strength) a joint of it bends and stays bent
                                # (metal, plastic); 0: it never bends, it breaks (stone, glass, wood)
    ductility: float = 0.0      # radians: how far a joint of it bends in all before it tears
    heat_capacity: float = 1000.0   # J/kg/K: the heat it takes to warm a kilogram of it a degree (steel 490, wood 1700)
    emissivity: float = 0.9         # how well its surface radiates heat, and so absorbs it (Kirchhoff): most things 0.9,
                                    # hot oxidised steel 0.6, bare aluminium 0.1

MATERIALS = {m.key: m for m in (
    Material('wood', 'Wood (pine)', 550.0, 0.45, 0.35, 2.0e+06, (0.45, 0.29, 0.15), 0.65, pattern='wood', inside=(0.62, 0.45, 0.27), dust=0.15, effusivity=400.0, impact=7.0, brittle=0.25, heat_capacity=1700.0, emissivity=0.9),
    Material('stone', 'Stone (granite)', 2650.0, 0.6, 0.25, 2.0e+06, (0.32, 0.31, 0.3), 0.8, pattern='stone', inside=(0.42, 0.41, 0.39), dust=0.6, effusivity=2400.0, impact=6.0, brittle=0.55, heat_capacity=790.0, emissivity=0.9),
    Material('concrete', 'Concrete', 2400.0, 0.65, 0.2, 1.0e+06, (0.42, 0.41, 0.38), 0.9, pattern='concrete', inside=(0.5, 0.48, 0.44), dust=1.0, effusivity=1900.0, impact=4.0, brittle=0.5, heat_capacity=880.0, emissivity=0.9),
    Material('brick', 'Brick', 1900.0, 0.6, 0.15, 1.5e+06, (0.38, 0.13, 0.07), 0.85, pattern='brick', inside=(0.5, 0.2, 0.11), dust=1.0, effusivity=1100.0, impact=2.0, brittle=0.35, heat_capacity=840.0, emissivity=0.9),
    Material('steel', 'Steel', 7850.0, 0.45, 0.45, 4.0e+08, (0.56, 0.57, 0.58), 0.35, metal=1.0, pattern='metal', effusivity=14000.0, impact=60.0, brittle=0.0, yields=2.5e+08, ductility=1.2, heat_capacity=490.0, emissivity=0.6),
    Material('aluminium', 'Aluminium', 2700.0, 0.45, 0.45, 2.5e+08, (0.91, 0.92, 0.92), 0.3, metal=1.0, pattern='metal', effusivity=24000.0, impact=40.0, brittle=0.0, yields=1.5e+08, ductility=0.8, heat_capacity=900.0, emissivity=0.1),
    Material('glass', 'Glass', 2500.0, 0.5, 0.5, 2.0e+07, (0.9, 0.95, 0.94), 0.03, clear=0.92, pattern='glass', inside=(0.75, 0.85, 0.82), dust=0.03, effusivity=1400.0, impact=2.5, brittle=0.7, heat_capacity=840.0, emissivity=0.9),
    Material('ice', 'Ice', 917.0, 0.05, 0.25, 3.0e+05, (0.8, 0.9, 0.95), 0.08, clear=0.7, pattern='glass', ior=1.31, dust=0.1, effusivity=2100.0, impact=3.0, brittle=0.75, heat_capacity=2100.0, emissivity=0.97),
    Material('plastic', 'Plastic', 950.0, 0.35, 0.5, 3.0e+06, (0.75, 0.1, 0.06), 0.4, effusivity=600.0, impact=15.0, brittle=0.3, yields=2.0e+06, ductility=1.5, heat_capacity=1500.0, emissivity=0.9),
    Material('rubber', 'Rubber', 1100.0, 0.9, 0.8, 5.0e+06, (0.04, 0.04, 0.04), 0.7, effusivity=600.0, impact=0.0, brittle=0.0, heat_capacity=1900.0, emissivity=0.9),
    Material('ceramic', 'Ceramic (pottery)', 2300.0, 0.5, 0.35, 3.0e+06, (0.8, 0.77, 0.72), 0.25, inside=(0.66, 0.5, 0.38), dust=0.4, effusivity=1400.0, impact=2.5, brittle=0.8, heat_capacity=900.0, emissivity=0.9),
    Material('cardboard', 'Cardboard box', 120.0, 0.5, 0.15, 1.0e+05, (0.48, 0.33, 0.17), 0.9, inside=(0.58, 0.45, 0.28), dust=0.3, effusivity=150.0, impact=6.0, brittle=0.1, heat_capacity=1400.0, emissivity=0.9),
    Material('foam', 'Foam', 40.0, 0.7, 0.3, 5.0e+04, (0.85, 0.85, 0.8), 0.95, dust=0.2, effusivity=40.0, impact=0.0, brittle=0.0, heat_capacity=1300.0, emissivity=0.9),
    # (a car body or a boat hull is a shell: its weight over the space it takes up)
    Material('painted', 'Painted metal', 300.0, 0.5, 0.4, 1.6e+07, (0.32, 0.03, 0.025), 0.25, inside=(0.56, 0.57, 0.58), effusivity=14000.0, impact=60.0, brittle=0.0, yields=1.0e+07, ductility=1.0, heat_capacity=490.0, emissivity=0.9),
    Material('plaster', 'Plaster wall', 900.0, 0.6, 0.2, 3.0e+05, (0.7, 0.68, 0.64), 0.9, inside=(0.8, 0.79, 0.76), dust=1.5, effusivity=600.0, impact=2.0, brittle=0.6, heat_capacity=1000.0, emissivity=0.9),
    Material('fabric', 'Fabric (upholstery)', 250.0, 0.8, 0.2, 5.0e+05, (0.36, 0.08, 0.06), 0.95, effusivity=150.0, impact=0.0, brittle=0.0, heat_capacity=1300.0, emissivity=0.9),
    Material('earth', 'Earth (soil)', 1600.0, 0.65, 0.1, 5.0e+04, (0.16, 0.12, 0.085), 0.95, pattern='concrete', dust=2.0, effusivity=800.0, impact=1.5, brittle=0.3, heat_capacity=800.0, emissivity=0.95),
    Material('person', 'Person (crash-test figure)', 1010.0, 0.8, 0.15, 2.0e+06, (0.78, 0.6, 0.16), 0.55, effusivity=1300.0, heat_capacity=1500.0, emissivity=0.9),
)}

def _woods():
    """The woods other than pine ('wood'), from their species (engine/wood.py): their density; their strength, the
    speed of a hit that breaks them and how readily they give off heat in proportion to pine's; their look (the stage
    draws their grain: wood.wgsl)."""
    from ..engine.wood import SPECIES
    pine = SPECIES['pine']
    base = MATERIALS['wood']
    out = []
    for k, sp in SPECIES.items():
        if k == 'pine':
            continue
        r = sp.mor / pine.mor
        label = sp.label if k in ('plywood', 'mdf') else f'Wood ({sp.label if k == "fir" else sp.label.lower()})'
        out.append(Material(k, label, sp.density,
                            base.friction, base.bounce, base.strength * r,
                            tuple(round(0.5 * (a + b), 3) for a, b in zip(sp.early, sp.late)), sp.roughness,
                            pattern=f'wood_{k}', inside=tuple(min(1.0, round(1.15 * c, 3)) for c in sp.early),
                            dust=base.dust * (2.0 if k == 'mdf' else 1.0), effusivity=base.effusivity * (sp.density / pine.density) ** 0.5,
                            impact=base.impact * r ** 0.5, brittle=base.brittle,
                            heat_capacity=base.heat_capacity, emissivity=base.emissivity))
    return out


MATERIALS.update({m.key: m for m in _woods()})

# (for the pickers: the woods together, pine first)
_WOODS = [k for k, m in MATERIALS.items() if m.pattern == 'wood' or m.pattern.startswith('wood_')]
OPTIONS = tuple((k, MATERIALS[k].label) for k in _WOODS + [k for k in MATERIALS if k not in _WOODS])

# the shader's pattern numbers (stage.wgsl)
PATTERNS = {'plain': 0, 'wood': 1, 'stone': 2, 'concrete': 3, 'brick': 4, 'metal': 5, 'glass': 0}
# (the woods' own grain, wood.wgsl: wood_<species> from 11 on, in engine/wood.py SPECIES' order)
PATTERNS.update({'wood_spruce': 11, 'wood_fir': 12, 'wood_oak': 13, 'wood_ash': 14, 'wood_maple': 15, 'wood_birch': 16,
                 'wood_walnut': 17, 'wood_cherry': 18, 'wood_mahogany': 19, 'wood_teak': 20, 'wood_cedar': 21,
                 'wood_balsa': 22, 'wood_plywood': 23, 'wood_mdf': 24})


@dataclass(frozen=True)
class Floor:
    key: str
    label: str
    colour: tuple       # linear albedo
    roughness: float
    pattern: int        # stage.wgsl floor_look
    effusivity: float = 1500.0   # W s^0.5/m^2/K: how readily the ground under it takes heat from what lies on it


FLOORS = {f.key: f for f in (
    Floor('studio', 'Studio grey', (0.2, 0.2, 0.2), 0.6, 0, effusivity=1900.0),
    Floor('concrete', 'Concrete', (0.33, 0.32, 0.3), 0.85, 1, effusivity=1900.0),
    Floor('boards', 'Wooden boards', (0.34, 0.21, 0.11), 0.55, 2, effusivity=400.0),
    Floor('tiles', 'Tiles', (0.6, 0.58, 0.54), 0.3, 3, effusivity=1500.0),
    Floor('dirt', 'Dirt', (0.17, 0.13, 0.09), 0.95, 4, effusivity=800.0),
    Floor('grass', 'Grass', (0.06, 0.11, 0.035), 0.9, 5, effusivity=700.0),
    Floor('sand', 'Sand', (0.45, 0.38, 0.27), 0.9, 6, effusivity=600.0),
    Floor('checker', 'Checker (1 m)', (0.5, 0.5, 0.5), 0.6, 7, effusivity=1900.0),
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
                bounce=bo if bo >= 0.0 else m.bounce, strength=m.strength, impact=m.impact,
                brittle=m.brittle, yields=m.yields, ductility=m.ductility)
