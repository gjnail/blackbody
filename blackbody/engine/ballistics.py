"""Bullets: what a shot does to everything it meets, in every kind of scene.

A shot (Scene.shots: a gun's muzzle, where it is aimed, the cartridge, when and how often it fires) sends bullets
along their real paths: slowed by the air (drag), dropping under gravity, through the rigid world of solids.py
(MuJoCo's ray casts against the ground, the walls, every object, falling body and broken piece as they are at that
step), and into the sand, snow, mud and jelly (matter.py) and the water (liquid.py) through the kernels in
bullet_*.wgsl.

Where a bullet meets a surface it either glances off (ricochet: when it comes in flatter than the surface's critical
angle), stops in it, or goes through and on, slower. Inside, it is slowed as a projectile in a dense medium is
(Poncelet's law, the one ballistics uses for penetration):

    m v dv/ds = -A (R + 1/2 Cd rho v^2)

A the area it presents, R the target's resistance to being pushed aside (its strength, as the pressure a cavity in it
takes to open), rho its density, Cd the bullet's nose. And it changes on the way, as real bullets do:
- it flattens or mushrooms when the pressure on its nose passes what it can take (soft lead easily, hollow points in
  anything wet, jacketed bullets only on hard things: steel, stone, concrete), presenting more area;
- a long pointed bullet turns sideways (yaws) after a distance in a dense medium, and above a speed then breaks up
  (5.56 mm and 7.62 mm in gel and water, at the depths gel tests show);
- going out of the far side, the last of the thickness gives way early (a plug pushed out of a plate, the back of a
  concrete wall scabbing off, splinters torn from the back of a board), taking some of its momentum.

R is calibrated per surface against typical published penetration figures (gel, water, pine boards, mild steel plate,
concrete, brick, sand, snow: tests/test_ballistics.py lists them); the depths come out within about a third.

The effects of each hit (Impact) are left for the others: the momentum to the body it hit (solids), a breakable
object's pieces broken where it went in (solids), debris, sparks and dust, the hole or dent left on the surface
(marks.wgsl), the kick it gives sand, jelly and water along its track (bullet_matter.wgsl, bullet_liquid.wgsl).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

AIR_DENSITY = 1.2      # kg/m^3
SPEED_OF_SOUND = 343.0 # m/s
STOP_SPEED = 12.0      # m/s: a bullet slower than this inside something has stopped (in water it sinks)
LEAD = 11340.0         # kg/m^3
SPARK_LIFE = 0.12      # s a spark glows, on average
BORES_MOST = 128       # the most holes and craters carved at once (the newest)
CLOTH_HOLES_MOST = 64  # the most holes drawn through cloth (the newest: cloth_draw.wgsl)
PUSH_MOST = 40.0       # m/s: the most speed a bullet gives a body or a piece it hits
PUSH_SPIN = 300.0      # rad/s: and spin
RIM = 0.006            # m: a bullet meets what comes within this of its axis at least (breakable things' pieces have
                       # gaps between them a millimetre wide, wider toward the point of a narrow one)
STEPS = 400            # the integration's steps through the deepest it can go
GEL_RHO, GEL_R = 1040.0, 4.0e6   # ordnance gel (10%): where bullets' yawing depths are measured
YAW_R = 1.0e7          # Pa: a medium this strong turns a bullet sideways twice as soon as a weak one as dense
YAW_MOST = 4.0         # (and the hardest no more than 4 times as soon: in them the bullet is flattened and worn away first)
HARD = 0.3             # a hard target (past 100 MPa) this strong for its build flattens a bullet on contact: a jacketed
                       # bullet on concrete, brick or steel
TUMBLE = 0.55          # a tumbling bullet presents this share of its side on average
PLUG_SHEAR = 0.5       # a plug's sides shear through at this share of the surface's resistance


@dataclass(frozen=True)
class Round:
    """A cartridge's bullet (or shot): what leaves the muzzle."""
    key: str
    label: str
    mass: float            # kg (each pellet's, for shot)
    calibre: float         # m: its diameter
    speed: float           # m/s at the muzzle
    length: float          # m: the bullet's length (it presents length x calibre when it turns sideways)
    build: str             # lead, fmj (full metal jacket), hp (hollow point), steel (steel core), ap (hardened core)
    cd_air: float          # drag coefficient in air (pointed rifle bullets low, round balls high)
    cd: float = 0.3        # its nose's drag coefficient in a dense medium (round nose 0.3, flat or flattened 0.8-1)
    yaw_depth: float = 0.0     # m: in gel (1040 kg/m^3) a pointed bullet turns sideways after this (0: it stays nose on)
    frag_speed: float = 0.0    # m/s: turned sideways faster than this, it breaks up (0: it holds together)
    pellets: int = 1           # shot: this many pellets in a cone
    spread: float = 0.0        # degrees: the cone's half angle (shot), or the gun's scatter
    tracer: bool = False       # it burns red along its path (a tracer round)


ROUNDS = {r.key: r for r in (
    Round('pellet', 'Air rifle pellet (.177)', 0.00053, 0.0045, 175.0, 0.0055, 'lead', 0.6, cd=0.6),
    Round('22lr', '.22 LR', 0.0026, 0.0057, 330.0, 0.0115, 'lead', 0.35, cd=0.6),
    Round('9mm', '9 mm pistol (FMJ)', 0.0080, 0.0090, 360.0, 0.0155, 'fmj', 0.32, cd=0.3, yaw_depth=0.45),
    Round('9mm_hp', '9 mm pistol (hollow point)', 0.0080, 0.0090, 360.0, 0.0155, 'hp', 0.32, cd=0.3),
    Round('45acp', '.45 ACP (FMJ)', 0.0149, 0.0115, 260.0, 0.0172, 'fmj', 0.33, cd=0.3, yaw_depth=0.55),
    Round('556', '5.56 mm rifle (M193)', 0.0036, 0.0057, 990.0, 0.0190, 'fmj', 0.27, cd=0.25, yaw_depth=0.12,
          frag_speed=700.0),
    Round('762x39', '7.62x39 mm rifle', 0.0079, 0.0079, 715.0, 0.0265, 'steel', 0.30, cd=0.25, yaw_depth=0.26),
    Round('308', '.308 / 7.62x51 mm rifle', 0.0097, 0.0078, 850.0, 0.0290, 'fmj', 0.27, cd=0.25, yaw_depth=0.16,
          frag_speed=740.0),
    Round('slug', '12 gauge slug', 0.0283, 0.0185, 470.0, 0.0180, 'lead', 0.45, cd=0.5),
    Round('buck', '12 gauge 00 buckshot (9 pellets)', 0.0035, 0.0084, 400.0, 0.0084, 'lead', 0.47, cd=0.47,
          pellets=9, spread=1.4),
    Round('50bmg', '.50 BMG rifle', 0.0420, 0.0129, 900.0, 0.0580, 'ap', 0.28, cd=0.25, yaw_depth=0.45),
)}

ROUND_OPTIONS = tuple((k, r.label) for k, r in ROUNDS.items())

# How each build of bullet takes the pressure on its nose (the target's resistance plus 1/2 rho v^2): past `deforms` Pa
# it flattens (or mushrooms) to `widens` times its area, nose drag `flat_cd`; a hollow point opens in anything wet.
BUILDS = {
    'lead':  dict(deforms=1.2e8, widens=2.2, flat_cd=0.8),
    'fmj':   dict(deforms=6.0e8, widens=1.8, flat_cd=0.8),
    'hp':    dict(deforms=2.5e7, widens=2.1, flat_cd=0.45),
    'steel': dict(deforms=1.2e9, widens=1.3, flat_cd=0.6),
    'ap':    dict(deforms=4.0e9, widens=1.1, flat_cd=0.4),
}


@dataclass(frozen=True)
class Surface:
    """How a surface takes a bullet."""
    key: str
    rho: float             # kg/m^3
    R: float               # Pa: its resistance to a cavity opening in it (calibrated: see the module's notes)
    crit: float            # degrees: a bullet coming in flatter than this glances off
    keep: float            # of its speed, what a glancing bullet keeps
    leave: float           # the angle it leaves at, as a share of the angle it came in at (hard flat things < 1: it
                           # skims along them; soft ones > 1: it digs in and the dent throws it up)
    back: float            # calibres: the far side gives way when this much is left (a plug, scabbing, splinters)
    debris: str            # what flies off: chips, splinters, shards, sparks, flakes, dirt, spray, fluff, none
    mark: int              # the mark it leaves (marks.wgsl): 0 none, 1 crater, 2 hole, 3 dent, 4 web, 5 splintered
    shell: float = 0.0     # m: a hollow thing's skin (a car's body, a cardboard box): only this much of it is solid
    flash: float = 0.0     # how bright a hit flashes (lead on steel splashes white-hot)


# Ricochet angles and what keeps are after Haag (Shooting Incident Reconstruction): bullets skim concrete and steel,
# skip off water below about 7 degrees, dig into soil and wood.
SURFACES = {s.key: s for s in (
    # the materials (scene/materials.py)
    Surface('wood', 550.0, 5.9e7, 8.0, 0.55, 1.6, 0.7, 'splinters', 5),
    Surface('stone', 2650.0, 6.0e8, 26.0, 0.65, 0.35, 2.5, 'chips', 1),
    Surface('concrete', 2400.0, 2.15e8, 22.0, 0.65, 0.4, 2.5, 'chips', 1),
    Surface('brick', 1900.0, 1.8e8, 18.0, 0.6, 0.5, 2.0, 'chips', 1),
    Surface('steel', 7850.0, 4.0e9, 30.0, 0.8, 0.6, 0.6, 'sparks', 3, flash=1.0),
    Surface('aluminium', 2700.0, 1.6e9, 20.0, 0.7, 0.8, 0.6, 'flakes', 3, flash=0.5),
    Surface('glass', 2500.0, 1.5e8, 12.0, 0.6, 0.6, 3.0, 'shards', 4),
    Surface('ice', 917.0, 1.0e8, 10.0, 0.65, 0.6, 2.5, 'shards', 4),
    Surface('plastic', 950.0, 5.0e7, 8.0, 0.6, 1.2, 0.8, 'chips', 2),
    Surface('rubber', 1100.0, 2.5e7, 6.0, 0.5, 1.5, 0.3, 'none', 2),
    Surface('ceramic', 2300.0, 4.0e8, 18.0, 0.6, 0.5, 3.0, 'shards', 4),
    Surface('cardboard', 700.0, 8.0e6, 4.0, 0.5, 1.5, 0.3, 'fluff', 2, shell=0.005),
    Surface('foam', 40.0, 2.0e5, 3.0, 0.4, 2.0, 0.0, 'fluff', 2),
    Surface('painted', 7850.0, 4.0e9, 25.0, 0.75, 0.8, 0.6, 'flakes', 2, shell=0.0008, flash=0.6),
    Surface('plaster', 900.0, 1.5e7, 10.0, 0.5, 1.2, 1.5, 'chips', 1),
    Surface('fabric', 250.0, 1.0e6, 3.0, 0.4, 2.0, 0.0, 'fluff', 2),
    Surface('earth', 1600.0, 2.5e7, 12.0, 0.5, 1.8, 0.0, 'dirt', 1),
    # the matter (matter.py) and the water
    Surface('sand', 1600.0, 2.2e7, 12.0, 0.5, 1.8, 0.0, 'dirt', 0),
    Surface('snow', 400.0, 2.4e6, 6.0, 0.5, 2.0, 0.0, 'dirt', 0),
    Surface('mud', 1700.0, 2.0e6, 8.0, 0.4, 2.0, 0.0, 'dirt', 0),
    Surface('clay', 1800.0, 8.0e6, 10.0, 0.4, 2.0, 0.0, 'dirt', 0),
    Surface('gel', 1040.0, 4.0e6, 6.0, 0.5, 2.0, 0.0, 'spray', 0),
    Surface('water', 1000.0, 0.0, 7.0, 0.85, 1.5, 0.0, 'spray', 0),
    Surface('wax', 900.0, 1.0e7, 8.0, 0.5, 1.5, 0.5, 'chips', 0),
)}

def _woods():
    """The other woods (engine/wood.py SPECIES) as surfaces: pine's, its resistance in proportion to their density
    (penetration in wood goes about as its density to the 1.25)."""

    from .wood import SPECIES
    pine = SPECIES['pine']
    return {k: replace(SURFACES['wood'], key=k, rho=sp.density, R=SURFACES['wood'].R * (sp.density / pine.density) ** 1.25)
            for k, sp in SPECIES.items() if k != 'pine'}


SURFACES.update(_woods())
# a car's cabin: a shell of window glass round air (its laminated windscreen and side windows about 5 mm)
SURFACES['windows'] = replace(SURFACES['glass'], key='windows', shell=0.005)

# The stage's floors (materials.FLOORS) as surfaces
FLOOR_SURFACES = {'studio': 'concrete', 'concrete': 'concrete', 'boards': 'wood', 'tiles': 'ceramic', 'dirt': 'earth',
                  'grass': 'earth', 'sand': 'sand', 'checker': 'concrete'}

# The matter's models (matter.MATTERS' model, or its key where that says more) as surfaces
MATTER_SURFACES = {'sand': 'sand', 'wet_sand': 'sand', 'soaked_sand': 'mud', 'snow': 'snow', 'packing_snow': 'snow',
                   'mud': 'mud', 'jelly': 'gel', 'gel': 'gel', 'clay': 'clay', 'leaves': 'snow', 'sawdust': 'snow', 'coal': 'sand',
                   'ash': 'snow', 'wax': 'wax', 'chocolate': 'wax', 'aluminium': 'aluminium', 'iron': 'steel',
                   'molten_wax': 'water', 'molten_chocolate': 'water', 'molten_aluminium': 'water', 'molten_iron': 'water'}


def surface(key):
    """The surface for a material, floor or matter key (wood when unknown)."""
    if key in SURFACES:
        return SURFACES[key]
    if key in MATTER_SURFACES:
        return SURFACES[MATTER_SURFACES[key]]
    return SURFACES.get(FLOOR_SURFACES.get(key, 'wood'), SURFACES['wood'])


@dataclass
class Bullet:
    """A bullet in flight (or what is left of it)."""
    pos: np.ndarray            # m (fire-local)
    vel: np.ndarray            # m/s
    mass: float                # kg
    calibre: float             # m
    length: float              # m
    build: str
    cd_air: float
    cd: float
    yaw_depth: float = 0.0
    frag_speed: float = 0.0
    tracer: bool = False
    born: float = 0.0          # s (simulation time) it left the muzzle
    shot: int = 0              # its shot (index into Scene.shots)
    n: int = 0                 # its number among the shot's bullets
    alive: bool = True
    widen: float = 1.0         # its presented area over its own (flattened, mushroomed, sideways)
    flat: bool = False         # flattened or mushroomed
    yawed: bool = False        # turned sideways
    broken: bool = False       # broken up
    dense: float = 0.0         # m: how far it has gone through dense stuff (gel-equivalent: for turning sideways)
    trail: list = field(default_factory=list)   # (time, position) where it has been, for drawing it and its tracer
    hits: int = 0              # surfaces met
    t: float = 0.0             # s: the time it has been flown to
    skip: tuple = None         # (owner, distance): what it has just come out of, not to be met again that close
    inside: tuple = None       # (where it went in, where it came out or stopped, when it went in, when it got there): its
                               # way through the last thing it went into, drawn as it goes (a bullet seen through gel)
    lodged: bool = False       # it stopped in something clear, where it is seen (gel): drawn there from then on

    @property
    def speed(self):
        return float(np.linalg.norm(self.vel))

    @property
    def area(self):
        return math.pi * 0.25 * self.calibre * self.calibre * self.widen

    @property
    def energy(self):
        return 0.5 * self.mass * float(self.vel @ self.vel)


def bullet_of(r: Round, pos, direction, born=0.0, shot=0, n=0, speed=None, tracer=None) -> Bullet:
    d = np.asarray(direction, float)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    v = float(r.speed if speed is None else speed)
    return Bullet(pos=np.asarray(pos, float).copy(), vel=d * v, mass=r.mass, calibre=r.calibre, length=r.length,
                  build=r.build, cd_air=r.cd_air, cd=r.cd, yaw_depth=r.yaw_depth, frag_speed=r.frag_speed,
                  tracer=r.tracer if tracer is None else bool(tracer), born=float(born), shot=shot, n=n)


def air_step(b: Bullet, h, gravity=9.81, rho=AIR_DENSITY):
    """The bullet's velocity after h seconds in still air (drag, and gravity down -y): its new position and velocity,
    by the midpoint rule (a bullet's drag changes slowly over a step)."""
    def acc(v):
        s = float(np.linalg.norm(v))
        k = 0.5 * rho * b.cd_air * b.area / max(b.mass, 1e-9)
        return -k * s * v + np.array([0.0, -gravity, 0.0])
    v0 = b.vel
    vm = v0 + 0.5 * h * acc(v0)
    v1 = v0 + h * acc(vm)
    return b.pos + h * vm, v1


@dataclass
class Passage:
    """What happened to a bullet going into a surface (through, stopping in it, or glancing off)."""
    outcome: str               # 'through', 'stop' or 'glance'
    depth: float               # m it went in (through: the whole thickness)
    speed_in: float            # m/s
    speed_out: float           # m/s (0 when it stopped)
    mass_in: float
    mass_out: float            # kg: what is left of the bullet (the rest splashed or broke away)
    energy: float              # J it left in the surface (and in its own deformation)
    plug: float = 0.0          # m: the far side pushed out ahead of it (its depth)
    flat: bool = False         # it flattened or mushroomed
    yawed: bool = False        # it turned sideways inside
    broken: bool = False       # it broke up inside
    angle: float = 0.0         # degrees: how flat it came in (0 grazing, 90 square on)
    track: list = field(default_factory=list)   # (depth m, speed m/s, presented area m^2) along its way in


def _nose_pressure(b: Bullet, s: Surface, v):
    return s.R + 0.5 * s.rho * v * v


def penetrate(b: Bullet, s: Surface, thickness, record=False) -> Passage:
    """Push the bullet into `thickness` metres of surface s, square on (its path through it is that long), changing it
    as it goes (its speed, mass, flattening, turning sideways, breaking up). The bullet is changed in place; returns
    what happened. Its direction is left to the caller."""
    v = b.speed
    m_in, v_in = b.mass, v
    cal = b.calibre
    build = BUILDS.get(b.build, BUILDS['fmj'])
    thick = max(float(thickness), 0.0)
    if s.shell > 0.0:
        thick = min(thick, 2.0 * s.shell)
    # the far side gives way when there is this much left (a plug, a scab, splinters)
    back = s.back * cal
    # step: fine enough to resolve the stop in the hardest surfaces (a few mm in steel)
    ds = max(min(thick, 4.0) / STEPS, 2.0e-5)
    track = []
    depth = 0.0
    plug = 0.0
    out = 'stop'
    # (a long bullet turns sideways sooner the denser and the stronger what it is in: by the pressure on its nose, gel's
    # at the same speed the reference)
    gel_scale = s.rho / GEL_RHO * min(1.0 + s.R / YAW_R, YAW_MOST) / (1.0 + GEL_R / YAW_R)
    while True:
        if depth >= thick - 1e-12:
            out = 'through'
            break
        rem = thick - depth
        if back > 0.0 and depth > 0.0 and rem <= back * math.sqrt(b.widen):
            # the rest is pushed out ahead of it as a plug: it shares its momentum with the plug, and the plug's sides
            # must shear through (PLUG_SHEAR of the resistance, over the plug's side as it slides out)
            d_eff = cal * math.sqrt(b.widen)
            m_plug = s.rho * b.area * rem
            v_shared = v * b.mass / (b.mass + m_plug)
            # (a brittle thing's back breaks out as a cone of cracks, for little more than the cracks' energy)
            shear = PLUG_SHEAR * (0.2 if s.debris in ('chips', 'shards') else 1.0)
            work = 0.5 * (shear * s.R) * math.pi * d_eff * rem * rem
            e_left = 0.5 * (b.mass + m_plug) * v_shared * v_shared - work
            if e_left > 0.0:
                plug = rem
                v = math.sqrt(2.0 * e_left / (b.mass + m_plug))
                depth = thick
                out = 'through'
                break
        p = _nose_pressure(b, s, v)
        if not b.flat and (p > build['deforms'] or s.R > max(HARD * build['deforms'], 1.0e8)):
            b.flat = True
            b.widen *= build['widens']
            b.cd = max(b.cd, build['flat_cd'])
            # (a jacketed bullet flattening on something hard loses its lead core's front: it splashes)
            if p > 3.0 * build['deforms'] and b.build in ('lead', 'fmj', 'hp'):
                b.mass *= 0.7
        if b.yaw_depth > 0.0 and not b.yawed and b.dense >= b.yaw_depth:
            b.yawed = True
            # (tumbling, it presents on average about half its side: length x calibre)
            side = b.length / max(cal, 1e-6)
            b.widen = max(b.widen, side * TUMBLE)
            b.cd = max(b.cd, 0.9)
            if b.frag_speed > 0.0 and v > b.frag_speed and not b.broken:
                # it breaks at the cannelure: the front stays together, the rest goes as fragments
                b.broken = True
                b.mass *= 0.6
                b.widen *= 0.8
        k = 0.5 * b.cd * s.rho
        a = b.area / max(b.mass, 1e-9)
        # m v dv/ds = -A (R + k v^2), exactly over the step: v^2 -> (v^2 + R/k) exp(-2 a k ds) - R/k
        if k > 0.0:
            e = math.exp(-2.0 * a * k * ds)
            v2 = (v * v + s.R / k) * e - s.R / k
        else:
            v2 = v * v - 2.0 * a * s.R * ds
        if record:
            track.append((depth, v, b.area))
        if v2 <= STOP_SPEED * STOP_SPEED:
            # stops within this step: where exactly
            if k > 0.0:
                stop = math.log((v * v + s.R / k) / (STOP_SPEED * STOP_SPEED + s.R / k)) / (2.0 * a * k)
            else:
                stop = (v * v - STOP_SPEED * STOP_SPEED) / (2.0 * a * max(s.R, 1e-9))
            depth += min(max(stop, 0.0), ds)
            v = 0.0
            out = 'stop'
            break
        v = math.sqrt(v2)
        depth += ds
        b.dense += ds * gel_scale
    if record:
        track.append((depth, v, b.area))
    energy = 0.5 * m_in * v_in * v_in - 0.5 * b.mass * v * v
    return Passage(out, depth, v_in, v, m_in, b.mass, max(energy, 0.0), plug=plug, flat=b.flat, yawed=b.yawed,
                   broken=b.broken, track=track)


def _transit(pas: Passage):
    """How long a bullet takes over its way into a surface (s): its depth at the speeds it went there."""
    tr = pas.track
    if len(tr) < 2:
        v = max(0.5 * (pas.speed_in + pas.speed_out), 1.0)
        return pas.depth / v
    s = np.array([t[0] for t in tr], float)
    v = np.maximum(np.array([t[1] for t in tr], float), STOP_SPEED)
    return float(np.sum(np.diff(s) / (0.5 * (v[:-1] + v[1:]))))


def depth_in(r: Round, key, speed=None):
    """How far a round goes into a semi-infinite block of surface `key` fired into it square on (m)."""
    b = bullet_of(r, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), speed=speed)
    return penetrate(b, surface(key), 50.0).depth


def plate_limit(r: Round, key, speed=None, lo=1e-4, hi=0.2):
    """The thickest plate of surface `key` the round goes right through, square on (m)."""
    def goes(t):
        b = bullet_of(r, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), speed=speed)
        return penetrate(b, surface(key), t).outcome == 'through'
    if not goes(lo):
        return 0.0
    if goes(hi):
        return hi
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        if goes(mid):
            lo = mid
        else:
            hi = mid
    return lo


def glances(b: Bullet, s: Surface, angle_deg, rng) -> bool:
    """Whether a bullet coming in at `angle_deg` to the surface (0 grazing, 90 square on) glances off: below the
    surface's critical angle it does, with a couple of degrees either way of chance (the surface's roughness, the
    bullet's wobble). Faster bullets dig in a little more readily (the critical angle falls as 1/sqrt(v/400))."""
    crit = s.crit * min(1.3, max(0.6, math.sqrt(400.0 / max(b.speed, 1.0))))
    if b.build in ('steel', 'ap'):
        crit *= 1.2        # (hard cores do not dig in: they skip)
    return angle_deg < crit + float(rng.normal(0.0, 1.5))


def glance(b: Bullet, s: Surface, d_in, n, angle_deg, rng):
    """A bullet glancing off a surface (unit normal n facing it; d_in its unit direction in): its new direction and
    speed. It leaves at `leave` times the angle it came in at, a little to either side, and keeps `keep` of its speed
    (less the squarer it came in). It flattens against anything hard."""
    tang = d_in - n * float(d_in @ n)
    tl = float(np.linalg.norm(tang))
    tang = tang / tl if tl > 1e-9 else np.array([1.0, 0.0, 0.0])
    side = np.cross(n, tang)
    out_angle = math.radians(max(0.3, angle_deg * s.leave * (1.0 + 0.25 * float(rng.normal()))))
    yaw = math.radians(float(rng.normal(0.0, 2.5 + 0.2 * angle_deg)))
    d = (tang * math.cos(out_angle) + n * math.sin(out_angle)) * math.cos(yaw) + side * math.sin(yaw)
    keep = s.keep * (1.0 - 0.25 * angle_deg / max(s.crit, 1.0))
    if s.R > 1e8 and b.build in ('lead', 'fmj', 'hp') and not b.flat:
        b.flat = True
        b.widen *= 1.4
        b.cd = max(b.cd, 0.6)
    return d / max(float(np.linalg.norm(d)), 1e-9), max(keep, 0.2) * b.speed


def deflect(d_in, n, angle_deg, s: Surface, rng):
    """Its direction going on through: bent a little toward the surface's normal (the near side bites first), and
    scattered by its own wobble, more in soft things that turn it."""
    tilt = math.radians(90.0 - angle_deg) * (0.08 if s.R > 1e8 else 0.15)
    t = d_in + n * float(d_in @ n)          # (n faces back toward where it came from)
    d = d_in * math.cos(tilt) - (t / max(float(np.linalg.norm(t)), 1e-9)) * math.sin(tilt) if np.linalg.norm(t) > 1e-6 else d_in
    jitter = rng.normal(0.0, 0.01 if s.R > 1e8 else 0.02, 3)
    d = d + jitter - d * float(d @ jitter)
    return d / max(float(np.linalg.norm(d)), 1e-9)


def crater_radius(energy, s: Surface, calibre, through=False, exit_side=False):
    """The radius of what a hit leaves on the surface (m): the hole or dent for tough things, the spalled crater for
    brittle ones (about as big as the energy left there breaks out: concrete loses a crater 3-4 cm across to a pistol,
    8-10 cm to a rifle), the splintered hole for wood (the far side's bigger)."""
    e = max(float(energy), 0.0)
    base = 0.5 * calibre
    if s.mark == 2:
        # tough: a hole about the bullet's size, the far side torn a little wider
        return base * ((1.5 if exit_side else 1.05) + 0.15 * math.sqrt(e / 500.0))
    if s.mark == 4:
        # glass, pottery, ice: the hole about the bullet's size, its crushed cone's mouth twice that, wider out the back
        return base * ((2.4 if exit_side else 1.6) + 0.3 * math.sqrt(e / 500.0))
    if s.debris in ('chips', 'shards'):
        # brittle: a cone of it breaks out, wider the more energy was left there, as craters in concrete are measured:
        # about 3.5 cm across from a pistol (500 J), 8-9 cm from a rifle (3500 J): across ~ E^0.45; smaller in
        # stronger stuff (by the cube root of its strength), bigger in weaker
        r = 0.0175 * (max(e, 1.0) / 500.0) ** 0.45 * (2.15e8 / max(s.R, 6.0e7)) ** (1.0 / 3.0)
        if exit_side:
            r *= 1.6
        return max(base * 1.2, min(r, (14.0 if exit_side else 8.0) * calibre))
    if s.debris == 'splinters':
        return base * (1.0 if not exit_side else 2.5 + 2.0e-4 * e)
    if s.debris in ('sparks', 'flakes'):
        if through:
            return base * 1.15
        return base * (1.2 + 0.4 * math.sqrt(e / 500.0))
    if s.debris == 'dirt':
        return base * 4.0 + 0.01 * math.sqrt(e / 500.0)
    return base


def splash_radius(energy, calibre):
    """How far the lead splashes round a bullet that stops on hard metal (m): the grey star a steel target wears."""
    return calibre * (1.8 + 1.4 * math.sqrt(max(energy, 0.0) / 500.0))


# ---- shots in a scene, flown through the rigid world -----------------------------------------------------------

# what a hit is in, for how its holes look broken (marks.wgsl CLASS)
CLASS = {'concrete': 1, 'stone': 2, 'brick': 3, 'plaster': 4, 'wood': 5, 'steel': 6, 'aluminium': 6, 'painted': 7, 'glass': 8,
         'ice': 8, 'windows': 8, 'ceramic': 9, 'earth': 10, 'plastic': 11, 'rubber': 11, 'cardboard': 11, 'foam': 11, 'fabric': 11, 'wax': 11}
CLASS.update({k: 5 for k in ('spruce', 'fir', 'oak', 'ash', 'maple', 'birch', 'walnut', 'cherry', 'mahogany', 'teak', 'cedar',
                             'balsa', 'plywood', 'mdf')})
# marks' styles (marks.wgsl), beyond Surface.mark: a bullet's skid where it glanced off, the lead's star on steel
SKID, SPLASH = 6, 7
# debris rows that are not an object's material
GROUND_ROW, LEAD_ROW, SPARK_ROW = -1, -2, -3


@dataclass
class ShotSpec:
    """A shot as the simulation fires it (Scene.shots)."""
    index: int
    round: Round
    speed: float           # m/s at the muzzle
    first: float           # the scene frame its first round goes off at
    count: int             # rounds
    interval: float        # s between rounds (simulation time)
    scatter: float         # radians: the gun's scatter (one sigma)
    tracer: bool
    flash: bool
    seed: int


def shot_specs(scene):
    """The enabled shots of a scene (Scene.shots). One whose first round goes before the pre-roll never fires."""
    out = []
    first = scene.start - float(scene.data['domain']['preroll']) * scene.fps
    for i, d in enumerate(getattr(scene, 'shots', None) or []):
        if not d.get('enabled', True):
            continue
        r = ROUNDS.get(d.get('round', '9mm'), ROUNDS['9mm'])
        f = scene.start + float(d.get('start', 0.0)) * scene.fps
        if f < first:
            continue
        speed = float(d.get('speed', 0.0) or 0.0)
        out.append(ShotSpec(i, r, speed if speed > 0.0 else r.speed, f, max(1, int(d.get('count', 1))),
                            60.0 / max(float(d.get('rate', 600.0)), 1e-3), math.radians(float(d.get('scatter', 0.05))),
                            bool(d.get('tracer', False)) or r.tracer, bool(d.get('flash', True)), int(d.get('seed', 0))))
    return out


@dataclass
class Impact:
    """One bullet meeting one surface."""
    time: float            # s (simulation)
    pos: np.ndarray        # where (fire-local m)
    normal: np.ndarray     # the surface's normal there, toward the bullet
    dir: np.ndarray        # the bullet's direction coming in
    speed: float           # m/s coming in
    out: float             # m/s going on (0: it stopped there)
    kind: str              # 'glance', 'stop', 'through'
    surface: str           # Surface key
    owner: tuple           # ('collider', i), ('body', i, n), ('piece', i, k, set), ('ground',), ('wall',), ('matter', key), ('water',)
    energy: float          # J it left there
    calibre: float
    shot: int
    exit: np.ndarray = None   # where it came out (through)
    depth: float = 0.0     # m it went in


@dataclass
class Mark:
    """What a hit leaves on a surface, kept in the frame of the MuJoCo body it is on (the world's for fixed things)."""
    body: int
    pos: np.ndarray        # in the body's frame
    normal: np.ndarray
    dir: np.ndarray        # the bullet's direction (skids and splinters run along it)
    style: int             # Surface.mark, SKID, SPLASH
    radius: float          # m
    seed: float
    exit: bool = False     # the far side (bigger, torn outward)
    grain: np.ndarray = None   # wood: the grain's direction (body frame)
    cls: int = 0           # what it is in (CLASS)


@dataclass
class Bore:
    """A hole or crater a bullet made, carved out of what it is in (the stage traces the object less every bore in it:
    marks.wgsl): a cone from a (radius ra) to b (radius rb), kept in the frame of the MuJoCo body it is in."""
    body: int
    a: np.ndarray
    b: np.ndarray
    ra: float
    rb: float
    grain: np.ndarray = None    # wood's grain (body frame): torn longer along it
    seed: float = 0.0
    elong: float = 0.0          # how much longer than wide it is along the grain (at its far end)
    rough: float = 0.1          # how ragged its edge is
    cls: int = 0                # what it is in (CLASS)
    lobes: float = 6.0          # how many ragged lobes round it
    row: int = -1               # the stage's material row of what it is in (splinters standing out of it are drawn in
                                # its wood: marks.wgsl fray_*), -1 the ground


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


class Ballistics:
    """The bullets of a scene's shots, flown with the rigid bodies (solids.py calls step() before each of its steps),
    what each did where it hit, and the debris, marks and flashes that came of it."""

    def __init__(self, scene, solids=None):
        from .debris import Debris
        self.scene = scene
        self.specs = shot_specs(scene)
        cd = scene.data.get('composite', {})
        self.floor = surface(cd.get('floor', 'concrete'))
        self.debris = Debris(seed=sum(s.seed for s in self.specs) + 17)
        self._tab = None
        self.ahead = {}          # (shot, n) -> [(A, B, surface key, medium)]: matter and water on this frame's path
        self.reset()

    @staticmethod
    def wanted(scene):
        return bool(shot_specs(scene))

    def reset(self):
        self.bullets: list[Bullet] = []
        self.impacts: list[Impact] = []
        self.marks: list[Mark] = []
        self.bores: list[Bore] = []
        self.flashes = []        # (time, position, strength, kind, direction): muzzle flashes, lead flashing on steel
        self.media = []          # bullets through matter and water, for their kernels (bullet_matter/liquid.wgsl)
        self.cloth_holes = []    # (u, v, radius, fabric): holes through the cloth, in its weave's coordinates (m)
        self.fired = [0] * len(self.specs)
        self.t_first = [None] * len(self.specs)
        self.rngs = [np.random.default_rng(1000 + 7919 * s.seed + s.index) for s in self.specs]
        self.debris.clear()
        self.now = 0.0

    # -- the world -------------------------------------------------------------------------------------------

    def bind(self, solids):
        """What each of the model's geoms is (solids.py's names): its owner and its surface."""
        import mujoco
        m = solids.model
        set_of = {ps.index: si for si, ps in enumerate(getattr(solids, 'sets', []) or [])}
        owners, surfs = [], []
        cols = self.scene.colliders
        for g in range(int(m.ngeom)):
            gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
            b = int(m.geom_bodyid[g])
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ''
            gt = int(m.geom_type[g])
            owner, key = ('other',), 'concrete'
            if bname.startswith('piece'):
                i, n = (int(x) for x in bname[5:].split('_')[:2])
                owner, key = ('piece', i, n, set_of.get(i, -1)), cols[i].get('material', 'wood')
            elif gname.startswith('fixed'):
                i = int(gname[5:].split('_')[0])
                owner, key = ('collider', i), cols[i].get('material', 'wood')
            elif bname.startswith('asm'):
                # a part of a person or a car (assemblies.py): a car's tyres rubber, its cabin glass, its body its own
                # material; a crash-test figure's vinyl and foam, as plastic
                from .assemblies import kind_of
                i = int(bname[3:].split('_')[0])
                look = None
                if gt == mujoco.mjtGeom.mjGEOM_MESH:
                    mesh = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[g])) or ''
                    a = next((a for a in getattr(solids, 'asms', []) or [] if a.get('index') == i), None)
                    try:
                        look = a['A'].looks[int(mesh.split('_')[-1])] if a is not None else None
                    except (ValueError, IndexError, KeyError):
                        look = None
                key = look or ('plastic' if kind_of(cols[i]) == 'figure' else cols[i].get('material', 'painted'))
                if key == 'glass':
                    key = 'windows'        # (a car's cabin: its windows, with the air of the cabin behind them)
                owner = ('collider', i)
            elif bname.startswith('body'):
                n = int(bname[4:].split('_')[0])
                bodies = getattr(solids, 'bodies', [])
                i = bodies[n].index if n < len(bodies) else -1
                owner, key = ('body', i, n), cols[i].get('material', 'wood') if i >= 0 else 'wood'
            elif gt == mujoco.mjtGeom.mjGEOM_PLANE:
                up = float(np.asarray(solids.data.geom_xmat[g]).reshape(3, 3)[1, 2])
                owner = ('ground',) if up > 0.9 else ('wall',)
                key = self.floor.key if up > 0.9 else 'concrete'
            elif gt == mujoco.mjtGeom.mjGEOM_HFIELD:
                owner, key = ('ground',), 'earth'
            owners.append(owner)
            surfs.append(key)
        self._tab = (owners, surfs)

    def _owner(self, g):
        return self._tab[0][g], surface(self._tab[1][g])

    # -- firing ----------------------------------------------------------------------------------------------

    def _fire(self, f0, f1, t0, h):
        """The rounds that go off in this step (scene frames f0 to f1, simulation time t0 to t0 + h)."""
        for si, sp in enumerate(self.specs):
            while self.fired[si] < sp.count:
                k = self.fired[si]
                if k == 0:
                    if f0 <= sp.first < f1:
                        self.t_first[si] = t0 + (sp.first - f0) / max(f1 - f0, 1e-12) * h
                    elif sp.first < f0:
                        self.t_first[si] = t0       # (a start inside the step before: it goes now)
                    else:
                        break
                    t = self.t_first[si]
                else:
                    t = self.t_first[si] + k * sp.interval
                if t >= t0 + h:
                    break
                frame = f0 + (max(t, t0) - t0) / max(h, 1e-12) * (f1 - f0)
                self._launch(si, sp, k, max(t, t0), frame)
                self.fired[si] += 1

    def _launch(self, si, sp, k, t, frame):
        sc = self.scene
        muzzle = np.asarray(sc.get(('shot', sp.index, 'position'), frame), float)
        aim = np.asarray(sc.get(('shot', sp.index, 'aim'), frame), float)
        d = _unit(aim - muzzle)
        if not np.any(d):
            d = np.array([0.0, 0.0, -1.0])
        rng = self.rngs[si]
        t1 = _unit(np.cross(d, [0.0, 1.0, 0.0] if abs(d[1]) < 0.95 else [1.0, 0.0, 0.0]))
        t2 = np.cross(d, t1)
        r = sp.round
        base = d + (t1 * rng.normal() + t2 * rng.normal()) * sp.scatter
        for p in range(r.pellets):
            dd = base
            if r.pellets > 1:
                # shot: a cone of pellets, spread evenly over it
                rr = math.radians(r.spread) * math.sqrt(rng.random())
                ph = rng.uniform(0.0, 2.0 * math.pi)
                dd = base + (t1 * math.cos(ph) + t2 * math.sin(ph)) * rr
            b = bullet_of(r, muzzle, dd, born=t, shot=si, n=k * r.pellets + p, speed=sp.speed, tracer=sp.tracer)
            b.t = t
            b.trail = [(t, muzzle.copy())]
            self.bullets.append(b)
        if sp.flash:
            self.flashes.append((t, muzzle + d * 0.03, 1.0, 'muzzle', d))

    # -- flying ----------------------------------------------------------------------------------------------

    def step(self, solids, f0, f1, t0, h):
        """Fire and fly every bullet through one of the rigid bodies' steps (before MuJoCo takes it, so what a bullet
        gives a body goes into this step). f0, f1: the step's start and end in scene frames; t0: its start in
        simulation seconds; h: its length."""
        if self._tab is None:
            self.bind(solids)
        self.now = t0
        self._fire(f0, f1, t0, h)
        t1 = t0 + h
        g = float(getattr(solids, 'gravity', 9.81))
        for b in self.bullets:
            if b.alive:
                self._fly(solids, b, t1, g)
        self.now = t1

    def _fly(self, solids, b, t1, gravity):
        lo, hi = getattr(solids, 'box', (None, None))
        guard = 0
        while b.alive and b.t < t1 - 1e-12 and guard < 32:
            guard += 1
            dt = t1 - b.t
            p1, v1 = air_step(b, dt, gravity)
            seg = p1 - b.pos
            L = float(np.linalg.norm(seg))
            if L < 1e-9:
                b.t = t1
                break
            u = seg / L
            hit = self._cast(solids, b, u, L)
            if hit is None:
                b.pos, b.vel, b.t = p1, v1, t1
                b.trail.append((b.t, b.pos.copy()))
                b.skip = None
                continue
            dist, g, nrm, medium = hit
            f = dist / L
            b.pos = b.pos + u * dist
            b.vel = b.vel + (v1 - b.vel) * f
            b.t += dt * f
            b.trail.append((b.t, b.pos.copy()))
            self._hit(solids, b, g, nrm, medium)
        # gone: spent, or well out of the box
        if b.alive and (b.speed < 25.0 or b.t - b.born > 4.0):
            b.alive = False
        if b.alive and lo is not None:
            span = float(np.max(np.asarray(hi) - np.asarray(lo)))
            if np.any(b.pos < np.asarray(lo) - 2.0 * span - 20.0) or np.any(b.pos > np.asarray(hi) + 2.0 * span + 20.0):
                b.alive = False
        if len(b.trail) > 64:
            del b.trail[:-64]

    def _cast(self, solids, b, u, L):
        """The first thing on the bullet's way over L metres along u: (distance, geom, normal, medium) or None. Matter
        and water come from this frame's look ahead (self.ahead): the stretches of them on its way."""
        import mujoco
        m, d = solids.model, solids.data
        gid = np.array([-1], np.int32)
        nrm = np.zeros(3)
        best = None
        # (a bullet is as wide as its calibre: rays along its middle and round its edge, so it cannot slip through the
        # hair's-breadth gaps between a breakable thing's pieces, or past the edge of something it clips)
        for off in self._rim(u, max(0.5 * b.calibre, RIM)):
            start = b.pos + u * 2.0e-5 + off
            dist = mujoco.mj_ray(m, d, start, u, None, 1, -1, gid, nrm)
            if dist >= 0.0 and dist + 2.0e-5 <= L and gid[0] >= 0 and (best is None or dist + 2.0e-5 < best[0]):
                if b.skip is not None and self._tab[0][int(gid[0])][:2] == b.skip[0] and dist < b.skip[1]:
                    continue
                best = (dist + 2.0e-5, int(gid[0]), nrm.copy(), None)
        for A, B, key, medium in self.ahead.get((b.shot, b.n), ()):
            a = np.asarray(A, float) - b.pos
            s = float(a @ u)
            if s < -1e-6 or s > L or (best is not None and s >= best[0]):
                continue
            if float(np.linalg.norm(a - u * s)) > 0.03:
                continue        # (it has turned off the path it was looked ahead on)
            best = (max(s, 0.0), -1, None, (np.asarray(A, float), np.asarray(B, float), key, medium))
        return best

    def _turned(self, solids, b):
        """A bullet that turned off its way this frame (a glance, a deflection) is looked ahead on again along its new
        one, so the matter and the water there meet it now, not a frame later (the engine sets `reahead`:
        bullet_media.py)."""
        f = getattr(self, 'reahead', None)
        if f is not None and not getattr(solids, 'rehearsal', False) and b.speed > 1.0:
            f(b)

    @staticmethod
    def _rim(u, r):
        """Offsets from a bullet's axis (along unit u) to its middle and eight points round its edge, radius r."""
        t1 = np.cross(u, [0.0, 1.0, 0.0] if abs(u[1]) < 0.9 else [1.0, 0.0, 0.0])
        t1 /= max(float(np.linalg.norm(t1)), 1e-12)
        t2 = np.cross(u, t1)
        c = 0.7071067811865476
        return (np.zeros(3), t1 * r, -t1 * r, t2 * r, -t2 * r, (t1 + t2) * c * r, (t1 - t2) * c * r, (t2 - t1) * c * r,
                -(t1 + t2) * c * r)

    def _span(self, solids, g, at, u):
        """Where the line through `at` along u enters and leaves geom g, and any of the same owner it passes straight
        on into (a mesh's boxes): distances along u from `at` (None, None: it misses). Planes and height fields:
        endless."""
        import mujoco
        m, d = solids.model, solids.data
        owner = self._tab[0][g]
        t_in = t_out = None
        gg, base = g, 0.0
        for _ in range(16):
            t = int(m.geom_type[gg])
            if t in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD):
                return (0.0 if t_in is None else t_in), 50.0
            far = 2.0 * float(m.geom_rbound[gg]) + float(np.linalg.norm(d.geom_xpos[gg] - at)) + 0.01
            here = at + u * base

            def ray(q, v):
                if t == mujoco.mjtGeom.mjGEOM_MESH:
                    return mujoco.mj_rayMesh(m, d, gg, q, v)
                return mujoco.mju_rayGeom(d.geom_xpos[gg], d.geom_xmat[gg], m.geom_size[gg], q, v, t)
            tf = ray(here - u * far, u)
            tb = ray(here + u * far, -u)
            if tf < 0.0 or tb < 0.0:
                break
            if t_in is None:
                t_in = base + tf - far
            t_out = base + far - tb
            # straight on into more of the same thing?
            gid = np.array([-1], np.int32)
            nx = mujoco.mj_ray(m, d, at + u * (t_out + 1.0e-5), u, None, 1, -1, gid, None)
            if gid[0] < 0 or nx < 0.0 or nx > 2.0e-3 or self._tab[0][int(gid[0])][:2] != owner[:2]:
                break
            gg = int(gid[0])
            base = t_out + nx
        return t_in, t_out

    def _run(self, solids, g, P, u, r):
        """How much of what geom g is a bullet of radius r entering it at P along u goes through, and how far on its
        far side is (m from P): the most over the lines along its middle and round its edge (a bullet half over an edge
        still goes through all of what it meets)."""
        thick = exit_d = 0.0
        for off in self._rim(u, r):
            t_in, t_out = self._span(solids, g, P + off, u)
            if t_in is None:
                continue
            thick = max(thick, t_out - max(t_in, 0.0))
            exit_d = max(exit_d, t_out)
        return thick, exit_d

    # -- hits ------------------------------------------------------------------------------------------------

    def _hit(self, solids, b, g, nrm, medium):
        rng = self.rngs[b.shot]
        u = _unit(b.vel)
        P = b.pos.copy()
        if medium is not None:
            A, B, key, which = medium
            s = surface(key)
            owner = ('matter', key) if which == 'matter' else ('water',)
            thick = float(np.linalg.norm(B - A))
            far_side = thick
            solid = thick
            n = -u
            body = -1
        else:
            owner, s = self._owner(g)
            thick, far_side = self._run(solids, g, P, u, max(0.5 * b.calibre, RIM))
            solid = thick * self._fill(owner)
            n = _unit(nrm)
            body = int(solids.model.geom_bodyid[g])
        if float(n @ u) > 0.0:
            n = -n
        angle = math.degrees(math.asin(min(1.0, abs(float(n @ u)))))
        v_in, m_in, cal = b.vel.copy(), b.mass, b.calibre
        b.hits += 1
        if getattr(solids, 'rehearsal', False) and medium is None:
            self._note(solids, g, P, b.speed)
        if glances(b, s, angle, rng):
            d_out, sp = glance(b, s, u, n, angle, rng)
            b.vel = d_out * sp
            b.pos = P + n * 2.0e-4
            self._turned(solids, b)
            e = 0.5 * m_in * float(v_in @ v_in) - b.energy
            imp = Impact(b.t, P, n, u, float(np.linalg.norm(v_in)), sp, 'glance', s.key, owner, max(e, 0.0), cal, b.shot)
            if medium is None:
                self._push(solids, g, P, m_in * v_in - b.mass * b.vel)
            else:
                # (skipping off it, it throws up a little of it: a short track at its speed)
                self.media.append(dict(time=b.t, at=P, dir=u, track=[(0.0, imp.speed, b.area), (0.02, sp, b.area)],
                                       surface=s.key, medium=medium[3], speed=imp.speed, energy=imp.energy, calibre=cal,
                                       outcome='glance', exit=None, out=d_out, mass=m_in))
            self._effects(solids, imp, body, rng, d_out=d_out)
            return
        pas = penetrate(b, s, solid, record=True)
        transit = _transit(pas)
        t_in = b.t
        if pas.outcome == 'through':
            Q = P + u * far_side if medium is None else np.asarray(medium[1], float)
            d_out = deflect(u, n, angle, s, rng)
            b.vel = d_out * pas.speed_out
            b.pos = Q + d_out * 2.0e-5
            self._turned(solids, b)
            if medium is None:      # (not to meet the same thing again on its way out)
                b.skip = (self._tab[0][g][:2], float(np.linalg.norm(b.pos - P)) + 0.01)
            # (it comes out when it has had time to go through: it flies on from then)
            b.t = t_in + transit
            b.inside = (P.copy(), np.asarray(Q, float).copy(), t_in, t_in + transit)
            plug_p = s.rho * math.pi * 0.25 * cal * cal * pas.plug * pas.speed_out * u
        else:
            Q = None
            b.alive = False
            b.pos = P + u * pas.depth
            b.vel = np.zeros(3)
            plug_p = np.zeros(3)
            b.inside = (P.copy(), b.pos.copy(), t_in, t_in + transit)
            b.lodged = s.key == 'gel'
        imp = Impact(b.t, P, n, u, pas.speed_in, pas.speed_out, pas.outcome, s.key, owner, pas.energy, cal, b.shot,
                     exit=Q, depth=pas.depth)
        if medium is None:
            self._push(solids, g, P, m_in * v_in - b.mass * b.vel - plug_p)
        else:
            self.media.append(dict(time=b.t, at=np.asarray(medium[0], float), dir=u, track=pas.track, surface=s.key,
                                   medium=medium[3], speed=pas.speed_in, energy=pas.energy, calibre=cal,
                                   outcome=pas.outcome, exit=Q, out=None, mass=m_in))
        self._effects(solids, imp, body, rng, pas=pas, g=g)

    def _fill(self, owner):
        """How much of an object is solid, for a bullet going through it: an object lighter than its material (a can,
        a crate, a hollow shell given its weight over the space it takes up: Density below its material's) is that
        much of it, and the bullet meets that share of its thickness."""
        if owner[0] not in ('collider', 'body', 'piece') or int(owner[1]) < 0:
            return 1.0
        from ..scene.materials import material
        c = self.scene.colliders[int(owner[1])]
        own = float(c.get('density', 0.0) or 0.0)
        full = material(c.get('material', 'wood')).density
        return 1.0 if own <= 0.0 else float(min(1.0, own / max(full, 1e-6)))

    def _note(self, solids, g, P, speed):
        """A rehearsal (solids._rehearse): where a breakable is hit, in its own frame, as a contact closing at the
        bullet's speed (nothing else hits as hard: its cracks crowd round the bullet's hole)."""
        i = getattr(solids, '_reh_geom', {}).get(g)
        if i is None:
            return
        got = solids.hits.get(i)
        if got is not None and got[1] >= speed:
            return
        pos, R = getattr(solids, '_reh_frame', {}).get(i, (None, None))
        if pos is None:
            bid = int(solids.model.geom_bodyid[g])
            pos, R = solids.data.xpos[bid], solids.data.xmat[bid].reshape(3, 3)
        solids.hits[i] = [np.round(np.asarray(R).T @ (np.asarray(P) - np.asarray(pos)), 4), float(speed), False]

    def _push(self, solids, g, P, J):
        """Give the body geom g is on the impulse J (N s) at P: through MuJoCo's mass matrix, so a free body, a piece
        welded to others or a door on its hinge each takes it as it would a real knock."""
        import mujoco
        if g < 0 or getattr(solids, 'rehearsal', False):
            return
        m, d = solids.model, solids.data
        bid = int(m.geom_bodyid[g])
        if bid == 0 or m.body_mocapid[bid] >= 0 or m.body_jntnum[bid] == 0:
            return
        for bd in getattr(solids, 'bodies', []):
            if bd.body_id == bid and bd.held:
                return
        for ps in getattr(solids, 'sets', []):
            if ps.held and bid in set(int(x) for x in ps.bodies):
                return
        qf = np.zeros(int(m.nv))
        mujoco.mj_applyFT(m, d, np.asarray(J, float), np.zeros(3), np.asarray(P, float), bid, qf)
        dq = np.zeros((1, int(m.nv)))
        mujoco.mj_solveM(m, d, dq, qf.reshape(1, -1))
        # (a tiny splinter given a bullet's whole momentum would fly off at hundreds of m/s and break the step: what
        # a body takes is capped at PUSH_MOST m/s; the rest went into the bits that flew off it)
        j = int(m.body_jntadr[bid])
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            a = int(m.jnt_dofadr[j])
            lin = float(np.linalg.norm(dq[0, a:a + 3]))
            ang = float(np.linalg.norm(dq[0, a + 3:a + 6]))
            k = min(1.0, PUSH_MOST / max(lin, 1e-9), PUSH_SPIN / max(ang, 1e-9))
            dq *= k
        d.qvel[:] += dq[0]

    # -- what a hit leaves -----------------------------------------------------------------------------------

    def _effects(self, solids, imp: Impact, body, rng, pas=None, g=-1, d_out=None):
        self.impacts.append(imp)
        if len(self.impacts) > 4096:
            del self.impacts[:-4096]
        if getattr(solids, 'rehearsal', False) or imp.owner[0] in ('matter', 'water'):
            return       # (the kernels splash the matter and the water: bullet_matter.wgsl, bullet_liquid.wgsl)
        s = SURFACES.get(imp.surface, SURFACES['wood'])
        E = imp.energy
        row = self._row(imp.owner)
        self._cls = CLASS.get(s.key, 5 if s.debris == 'splinters' else 0)
        self._mrow = self._mat_row(imp.owner)
        self._rng = rng
        grain = self._grain(solids, imp.owner, body)
        self._grain_now = grain
        n, u, P = imp.normal, imp.dir, imp.pos
        if imp.kind == 'glance':
            r_skid = imp.calibre * (1.5 + 0.004 * E ** 0.5)
            self._mark(solids, body, P, n, d_out, SKID, r_skid, rng, grain=grain)
            if s.R > 2e7 and s.debris != 'none':
                # the groove it ploughed along the surface: a round-bottomed furrow, a bullet's width, shallow
                along = _unit(d_out - n * float(d_out @ n))
                rr = 0.6 * imp.calibre
                self._bore(solids, body, P - along * 0.3 * r_skid - n * 0.6 * rr, P + along * 4.0 * r_skid - n * 0.75 * rr,
                           rr, 0.7 * rr, rough=0.2, lobes=5.0)
            if s.debris == 'sparks' or s.R > 2e8:
                self._spawn('sparks', P, n, d_out, E * 0.3, row, rng, glance=True)
            if s.debris != 'sparks':
                self._spawn(s.debris, P, n, _unit(n + d_out), E * 0.4, row, rng, grain=grain)
            return
        through = imp.kind == 'through'
        # the way in
        r_in = crater_radius(E * (0.4 if through else 1.0), s, imp.calibre, through=through)
        style = s.mark
        if s.debris in ('sparks', 'flakes') and not through:
            style = 3     # (dented, not holed)
        self._mark(solids, body, P, n, u, style, r_in, rng, grain=grain)
        self._carve(solids, body, imp, s, r_in, style)
        if s.debris == 'sparks' and not through:
            self._mark(solids, body, P + n * 1e-4, n, u, SPLASH, splash_radius(E, imp.calibre), rng)
            self.flashes.append((imp.time, P + n * 0.01, min(3.0, E / 400.0) * s.flash, 'hit', n))
        elif s.flash > 0.0:
            self.flashes.append((imp.time, P + n * 0.01, 0.4 * s.flash, 'hit', n))
        # what flies off the way in: back out of the crater, round the normal
        back = _unit(n * 1.5 - u * 0.5)
        share = 0.35 if through else 1.0
        self._spawn(s.debris, P, n, back, E * share * (0.6 if s.debris == 'splinters' else 1.0), row, rng, grain=grain)
        if through and imp.exit is not None:
            Q = imp.exit
            r_out = crater_radius(E, s, imp.calibre, through=True, exit_side=True)
            self._mark(solids, body, Q, u, u, style, r_out, rng, exit=True, grain=grain)
            # the far side spalls forward: splinters, shards, a cone of chips
            self._spawn(s.debris, Q, u, u, E * (1.4 if s.debris in ('splinters', 'shards') else 0.7), row, rng,
                        grain=grain, exit=True)
        if imp.owner[0] == 'piece':
            self._crack(solids, imp, s, rng)

    def _carve(self, solids, body, imp, s, r_in, style):
        """The hole or crater a hit leaves, as bores (Bore): through a board, a can, a car's door or a pane, a hole
        that flares out the back (wood's torn splinters, glass's spalled cone); into concrete, stone, brick or the
        ground, a cone-shaped crater; into steel it did not go through, a shallow dish; into wood or plastic it stopped
        in, a deep narrow hole."""
        P, n, u = imp.pos, imp.normal, imp.dir
        a = 0.5 * imp.calibre
        lip = 0.0015
        if imp.kind == 'through' and imp.exit is not None:
            Q = np.asarray(imp.exit, float)
            r_out = crater_radius(imp.energy, s, imp.calibre, through=True, exit_side=True)
            if s.debris in ('chips',):
                # brittle and thick: a crater going in, a bore through, a wider scab breaking out of the back
                depth = float(np.linalg.norm(Q - P))
                self._bore(solids, body, P + n * lip, P + u * min(0.45 * r_in, 0.5 * depth), r_in, 1.2 * a, rough=0.3,
                           lobes=7.0)
                self._bore(solids, body, P, Q + u * lip, 1.2 * a, r_out, rough=0.35, lobes=7.0)
            elif s.mark == 5:
                # wood: in through a hole a little narrower than the bullet (the fibres close behind it), out through a
                # split torn along the grain, ragged with fibres
                # split torn along the grain, ragged with fibres: the bullet's own channel through the board, and
                # out of its back a shallow scoop broken out, longer along the grain, its torn fibres facing out
                r1 = max(a, min(r_out, 8.0 * a)) * 0.5
                thick = float(np.linalg.norm(Q - P))
                scoop = min(0.4 * thick, 0.8 * r1)
                g_now = self._grain_now
                self._grain_now = None
                self._bore(solids, body, P - u * lip, Q + u * lip, 0.9 * a, a, rough=0.2, lobes=9.0)
                self._grain_now = g_now
                self._bore(solids, body, Q - u * scoop, Q + u * lip, 0.9 * a, r1, elong=1.5, rough=0.4, lobes=16.0)
            elif s.mark == 4:
                # glass: the bullet's hole, its far side a cone of glass spalled out (conchoidal, ragged)
                self._bore(solids, body, P - u * lip, Q + u * lip, 1.15 * a, max(1.2 * a, min(r_out, 12.0 * a)), rough=0.25,
                           lobes=9.0)
            else:
                # tough (metal, plastic, cardboard): about the bullet's size, its far side petalled out
                metal = self._cls in (6, 7)
                self._bore(solids, body, P - u * lip, Q + u * lip, 1.05 * a, 1.3 * a, rough=0.3 if metal else 0.15,
                           lobes=5.0 if metal else 8.0)
            return
        depth = max(float(imp.depth), 0.0)
        if style == 1 or s.debris == 'dirt':
            # a crater: a shallow cone spalled out round where it went in, broken into facets, and at its floor the
            # bullet's own pocket, narrower, as deep as it went
            cone = max(min(0.4 * r_in, depth), a)
            self._bore(solids, body, P + n * lip, P + u * cone, r_in, max(0.9 * a, 0.3 * r_in), rough=0.35, lobes=6.0)
            if depth > cone + 0.5 * a:
                self._bore(solids, body, P + u * (0.6 * cone), P + u * depth, 0.85 * a, 0.55 * a, rough=0.2, lobes=5.0)
        elif style == 3:
            # a dent: a shallow dish pushed into the metal
            self._bore(solids, body, P + n * (0.3 * r_in), P + u * max(min(depth, 0.25 * r_in), 2.0e-4), r_in, 0.4 * r_in,
                       rough=0.04, lobes=3.0)
        elif depth > 0.0:
            # a hole it stopped in (wood, plastic): narrow and deep, dark at its end
            self._bore(solids, body, P - u * lip, P + u * depth, 0.95 * a, 0.8 * a, rough=0.12, lobes=10.0)

    def _bore(self, solids, body, A, B, ra, rb, elong=0.0, rough=0.1, lobes=6.0):
        if body is None or body < 0:
            return
        R = np.asarray(solids.data.xmat[body]).reshape(3, 3)
        x = np.asarray(solids.data.xpos[body])
        g = getattr(self, '_grain_now', None)
        rng = getattr(self, '_rng', None)
        self.bores.append(Bore(body, R.T @ (np.asarray(A, float) - x), R.T @ (np.asarray(B, float) - x), float(ra), float(rb),
                               None if g is None else R.T @ np.asarray(g, float), float(rng.random()) if rng is not None else 0.0,
                               float(elong if g is not None else 0.0), float(rough), int(getattr(self, '_cls', 0)),
                               float(lobes), int(getattr(self, '_mrow', -1))))
        if len(self.bores) > BORES_MOST:
            del self.bores[:-BORES_MOST]

    def _row(self, owner):
        if owner[0] in ('collider', 'body', 'piece'):
            return int(owner[1])
        return GROUND_ROW

    def _mat_row(self, owner):
        """The stage's material row of what was hit (stage.py: the enabled objects in order), or -1."""
        if owner[0] not in ('collider', 'body', 'piece'):
            return -1
        enabled = [i for i, c in enumerate(self.scene.colliders) if c['enabled']][:16]
        ci = int(owner[1])
        return enabled.index(ci) if ci in enabled else -1

    def _grain(self, solids, owner, body):
        """Wood's grain where it was hit (world): along the object's longest side, as the stage draws its rings."""
        if owner[0] not in ('collider', 'body', 'piece') or int(owner[1]) < 0:
            return None
        from .wood import grain_axis, is_wood
        c = self.scene.colliders[int(owner[1])]
        if not is_wood(c.get('material', '')) or c.get('material') == 'mdf':
            return None     # (MDF is a felt of fibres: no grain to split along)
        ax = grain_axis(c.get('shape', 'box'), c.get('size', (1.0, 1.0, 1.0)))
        e = np.zeros(3)
        e[ax] = 1.0
        if body is None or body < 0:
            return e
        if body == 0:
            from .solids import turn_wxyz, xyzw
            from .liquid_float import q_rot
            enabled = [i for i, cc in enumerate(self.scene.colliders) if cc['enabled']]
            if int(owner[1]) in enabled:
                cg = self.scene.colliders_gpu(self.scene.start)[enabled.index(int(owner[1]))]
                return q_rot(xyzw(turn_wxyz(cg))) @ e
            return e
        return np.asarray(solids.data.xmat[body]).reshape(3, 3) @ e

    def _mark(self, solids, body, P, n, u, style, radius, rng, exit=False, grain=None):
        if style <= 0 or body is None or body < 0:
            return
        R = np.asarray(solids.data.xmat[body]).reshape(3, 3)
        x = np.asarray(solids.data.xpos[body])
        g = None if grain is None else R.T @ np.asarray(grain, float)
        self.marks.append(Mark(body, R.T @ (np.asarray(P) - x), R.T @ np.asarray(n), R.T @ np.asarray(u), int(style),
                               float(radius), float(rng.random()), bool(exit), g, int(getattr(self, '_cls', 0))))
        if len(self.marks) > 512:
            del self.marks[:-512]

    def _spawn(self, kind, at, normal, out_dir, energy, row, rng, glance=False, grain=None, exit=False):
        """Debris for `energy` J left at a hit (debris.py): how much, how fast and how big by what it is."""
        from . import debris as D
        if energy <= 0.5 or kind in ('none', 'spray'):
            return
        e = float(energy)
        now = self.now
        if kind == 'sparks':
            # (bright and brief: hot fragments of jacket and steel, glowing for a tenth of a second; off a glance on along
            # its way, off a square hit fanned out over the plate's face)
            n_sp = int(min(30, 5 + e / 60.0))
            self.debris.spawn(D.SPARK, at, normal, out_dir, n_sp, 22.0, 0.0006, SPARK_ROW, now,
                              spread=0.3, life=SPARK_LIFE, fan=not glance)
            if not glance:
                # the bullet's lead, splashed flat out along the plate
                self.debris.spawn(D.LEAD, at, normal, normal, int(min(30, 4 + e / 40.0)), 35.0, 0.0012, LEAD_ROW, now,
                                  fan=True)
            return
        k = D.KIND_OF.get(kind, D.CHIP)
        if kind == 'chips':
            count, speed, size, spread = min(50, 4 + e / 30.0), 14.0, 0.0035, 0.75
        elif kind == 'splinters':
            count, speed, size, spread = min(40, 3 + e / 35.0), 18.0, 0.0018, 0.5
        elif kind == 'shards':
            count, speed, size, spread = min(60, 5 + e / 15.0), 16.0, 0.004, 0.55
        elif kind == 'flakes':
            count, speed, size, spread = min(30, 3 + e / 40.0), 10.0, 0.0025, 0.9
        elif kind == 'dirt':
            count, speed, size, spread = min(50, 5 + e / 20.0), 8.0, 0.004, 0.6
        else:
            count, speed, size, spread = min(20, 2 + e / 60.0), 4.0, 0.004, 1.0
        if exit:
            count *= 1.3
            spread *= 0.8
        self.debris.spawn(k, at, normal, out_dir, int(count), speed, size, row, now, spread=spread, grain=grain)

    def _crack(self, solids, imp, s, rng):
        """A breakable object hit: the pieces in the bullet's way break away (a hole through a pane with its web of
        cracks left standing round it, a bite out of a brick), and a small brittle thing (a bottle, a vase, a plate)
        shatters through; the pieces that go are thrown on along its way and out from its path."""
        si = int(imp.owner[3]) if len(imp.owner) > 3 else -1
        if si < 0 or si >= len(getattr(solids, 'sets', [])) or not hasattr(solids, 'break_off'):
            return
        ps = solids.sets[si]
        d = solids.data
        cen = d.xpos[ps.bodies]
        P = imp.pos
        Q = imp.exit if imp.exit is not None else P + imp.dir * imp.depth
        ext = float(np.ptp(cen, axis=0).max()) + 2.0 * float(np.max(np.abs(ps.size)))
        brittle = s.debris == 'shards'
        r_hole = crater_radius(imp.energy, s, imp.calibre, through=imp.kind == 'through', exit_side=True)
        # each piece's centre's distance from the bullet's path through it
        seg = Q - P
        L2 = max(float(seg @ seg), 1e-12)
        t = np.clip((cen - P) @ seg / L2, 0.0, 1.0)
        dist = np.linalg.norm(cen - (P + t[:, None] * seg[None]), axis=1)
        rad = np.array([float(np.linalg.norm(p.verts - p.centroid, axis=1).max()) for p in ps.frac.pieces[:len(ps.bodies)]])
        if brittle and ext < 25.0 * r_hole and imp.energy > 20.0:
            go = np.arange(len(ps.bodies))        # (small and brittle: it all goes)
        else:
            go = np.nonzero(dist < r_hole + 0.35 * rad)[0]
            k = int(imp.owner[2])
            if brittle or rad[k] < 3.0 * r_hole:
                go = np.union1d(go, [k])
        if not len(go):
            return
        solids.break_off(si, go)
        # thrown: along the bullet's way and out from its path, harder the closer to it
        for j in go:
            vadr = int(ps.vadr[j])
            away = cen[j] - (P + t[j] * seg)
            a = float(np.linalg.norm(away))
            away = away / a if a > 1e-9 else _unit(rng.normal(size=3))
            near = 1.0 / (1.0 + dist[j] / max(r_hole, 1e-3))
            kick = (0.05 * imp.speed * near * (0.6 + 0.8 * rng.random()) if brittle else 2.0 * near)
            d.qvel[vadr:vadr + 3] += imp.dir * kick * 0.6 + away * kick * 0.5
            d.qvel[vadr + 3:vadr + 6] += rng.normal(0.0, 6.0 * near, 3)

    # -- frames ----------------------------------------------------------------------------------------------

    def frame_end(self, solids, fdt, substeps=4):
        """After the rigid bodies' frame: the debris flies on through it (fdt seconds), bouncing off what it meets."""
        import mujoco
        if solids.model is None or not len(self.debris):
            self._forget()
            return
        m, d = solids.model, solids.data
        gid = np.zeros(1, np.int32)
        nrm = np.zeros(3)

        def ray(pts, vec):
            frac = np.full(len(pts), 2.0)
            nn = np.zeros((len(pts), 3))
            for k in range(len(pts)):
                L = float(np.linalg.norm(vec[k]))
                if L < 1e-9:
                    continue
                dist = mujoco.mj_ray(m, d, pts[k], vec[k] / L, None, 1, -1, gid, nrm)
                if 0.0 <= dist <= L:
                    frac[k] = dist / L
                    nn[k] = nrm
            return frac, nn

        h = fdt / max(1, substeps)
        t = self.now - fdt
        for _ in range(max(1, substeps)):
            t += h
            self.debris.step(h, t, ray=ray, gravity=float(getattr(solids, 'gravity', 9.81)))
        self._forget()

    def _forget(self):
        """Drop what no longer matters: long-dead bullets (their tracers gone; not those lodged where they are seen),
        old flashes."""
        now = self.now
        self.bullets = [b for b in self.bullets if b.alive or b.lodged or now - b.t < 0.25 or
                        (b.inside is not None and now < b.inside[3])]
        self.flashes = [f for f in self.flashes if now - f[0] < 0.5]

    # -- state -----------------------------------------------------------------------------------------------

    def state(self):
        import copy
        return dict(bullets=copy.deepcopy(self.bullets), impacts=list(self.impacts), marks=list(self.marks),
                    bores=list(self.bores),
                    flashes=list(self.flashes), fired=list(self.fired), t_first=list(self.t_first),
                    cloth_holes=list(self.cloth_holes),
                    rngs=[r.bit_generator.state for r in self.rngs], debris=self.debris.state(), now=self.now)

    def load_state(self, st):
        import copy
        if not st:
            self.reset()
            return
        self.bullets = copy.deepcopy(st['bullets'])
        self.impacts = list(st['impacts'])
        self.marks = list(st['marks'])
        self.bores = list(st.get('bores', []))
        self.flashes = list(st['flashes'])
        self.cloth_holes = list(st.get('cloth_holes', []))
        self.fired = list(st['fired'])
        self.t_first = list(st['t_first'])
        for r, s in zip(self.rngs, st['rngs']):
            r.bit_generator.state = s
        self.debris.load_state(st['debris'])
        self.now = float(st['now'])
        self.media = []

    # -- drawing ---------------------------------------------------------------------------------------------

    def view(self, solids, shutter=0.0):
        """What to draw now (kept with each cached frame): the bullets in flight and their tracers, the debris, the
        marks where they are now (world: they move with what they are on), and the flashes. Plain arrays."""
        d = solids.data if solids is not None else None
        now = self.now
        out = {}
        rows = []
        for b in self.bullets:
            ins = b.inside
            if ins is not None and ins[2] <= now < ins[3]:
                # on its way through something: drawn where it has got to, at the speed it goes there
                f = (now - ins[2]) / max(ins[3] - ins[2], 1e-12)
                d = ins[1] - ins[0]
                rows.append([*(ins[0] + d * f), b.calibre, *(d / max(ins[3] - ins[2], 1e-12)), b.length, 0.0, now, 0.0, 0.0])
            elif b.alive and (ins is None or now >= ins[3]):
                rows.append([*b.pos, b.calibre, *b.vel, b.length, float(b.tracer), b.t, 0.0, 0.0])
            elif b.lodged and ins is not None and now >= ins[3]:
                rows.append([*ins[1], b.calibre, 0.0, 0.0, 1e-3, b.length, 0.0, b.t, 0.0, 0.0])
        if rows:
            out['bullets'] = np.array(rows, np.float32)
        tr = []
        for b in self.bullets:
            if not b.tracer or len(b.trail) < 2:
                continue
            pts = [p for (t, p) in b.trail if now - t < 0.06]
            if len(pts) >= 2:
                tr.append(np.asarray(pts, np.float32))
        if tr:
            out['tracers'] = tr
        db = self.debris
        if len(db):
            out['debris'] = dict(pos=db.pos.astype(np.float32), quat=db.quat.astype(np.float32),
                                 vel=db.vel.astype(np.float32), omega=db.omega.astype(np.float32),
                                 row=db.row.astype(np.int32), kind=db.kind.astype(np.int32),
                                 glow=db.glow(now).astype(np.float32), planes=[p.astype(np.float32) for p in db.planes])
        if self.marks and d is not None:
            M = np.zeros((len(self.marks), 4, 4), np.float32)
            for k, mk in enumerate(self.marks):
                R = np.asarray(d.xmat[mk.body]).reshape(3, 3)
                x = np.asarray(d.xpos[mk.body])
                M[k, 0, :3] = R @ mk.pos + x
                M[k, 0, 3] = mk.radius
                M[k, 1, :3] = R @ mk.normal
                M[k, 1, 3] = mk.style
                M[k, 2, :3] = R @ mk.dir
                M[k, 2, 3] = mk.seed
                M[k, 3, :3] = (R @ mk.grain) if mk.grain is not None else 0.0
                M[k, 3, 3] = (1.0 if mk.exit else 0.0) + 2.0 * mk.cls
            out['marks'] = M
        if self.bores and d is not None:
            Bv = np.zeros((len(self.bores), 4, 4), np.float32)
            for k, bo in enumerate(self.bores):
                R = np.asarray(d.xmat[bo.body]).reshape(3, 3)
                x = np.asarray(d.xpos[bo.body])
                Bv[k, 0, :3] = R @ bo.a + x
                Bv[k, 0, 3] = bo.ra
                Bv[k, 1, :3] = R @ bo.b + x
                Bv[k, 1, 3] = bo.rb
                Bv[k, 2, :3] = (R @ bo.grain) if bo.grain is not None else 0.0
                Bv[k, 2, 3] = bo.seed
                Bv[k, 3] = (bo.elong, bo.rough, bo.cls + 16 * (bo.row + 1), bo.lobes)   # (marks.wgsl bore_cls, bore_row)
            out['bores'] = Bv
        fl = [f for f in self.flashes if now - f[0] < 0.004 + 0.5 * shutter]
        if fl:
            out['flashes'] = np.array([[*p, s, *dd, 1.0 if k == 'muzzle' else 0.0] for (t, p, s, k, dd) in fl], np.float32)
        if self.cloth_holes:
            out['cloth_holes'] = np.asarray(self.cloth_holes[-CLOTH_HOLES_MOST:], np.float32)
        return out
