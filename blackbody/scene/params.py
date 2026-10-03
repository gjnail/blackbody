"""Parameter registry: every user-facing setting with its range, unit and tooltip.

The Properties panel, the command line (--set section.key=value), presets and project files all
read from this one table, so a parameter added here shows up everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .materials import FLOOR_OPTIONS
from .materials import OPTIONS as MATERIAL_OPTIONS


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    kind: str = 'float'        # float, int, bool, enum, color, vec3, str, file
    default: object = 0.0
    lo: float = 0.0            # slider range
    hi: float = 1.0
    hard_lo: float | None = None
    hard_hi: float | None = None
    unit: str = ''
    decimals: int = 2
    options: tuple = ()        # enum: ((value, label), ...)
    anim: bool = False
    tip: str = ''
    group: str = ''
    log: bool = False
    advanced: bool = False


def F(key, label, default, lo, hi, unit='', decimals=2, anim=False, tip='', group='', log=False, advanced=False,
      hard_lo=None, hard_hi=None):
    return Param(key, label, 'float', default, lo, hi, hard_lo, hard_hi, unit, decimals, (), anim, tip, group, log, advanced)


def I(key, label, default, lo, hi, unit='', tip='', group='', advanced=False, hard_lo=None, hard_hi=None):
    return Param(key, label, 'int', default, lo, hi, hard_lo, hard_hi, unit, 0, (), False, tip, group, False, advanced)


def B(key, label, default, tip='', group='', advanced=False):
    return Param(key, label, 'bool', default, tip=tip, group=group, advanced=advanced)


def E(key, label, default, options, tip='', group='', advanced=False):
    return Param(key, label, 'enum', default, options=tuple(options), tip=tip, group=group, advanced=advanced)


def C(key, label, default, anim=False, tip='', group='', advanced=False):
    return Param(key, label, 'color', tuple(default), anim=anim, tip=tip, group=group, advanced=advanced)


def V(key, label, default, lo, hi, unit='', anim=False, tip='', group='', decimals=2, advanced=False):
    return Param(key, label, 'vec3', tuple(default), lo, hi, None, None, unit, decimals, (), anim, tip, group, False, advanced)


SECTIONS = {
    'domain': [
        E('kind', 'Simulation', 'fire', (('fire', 'Fire and smoke'), ('liquid', 'Liquid'), ('both', 'Fire and liquid'),
                                          ('cloud', 'Clouds and weather')),
          tip='What this shot simulates: burning gas, a liquid such as water, both in the same box (a hose putting out a fire, burning fuel on water), '
              'or the sky: clouds building from the warmed ground, storms, their rain, snow and hail (kilometres across: see Atmosphere › Scale).',
          group='Simulation'),
        F('size_x', 'Width', 2.0, 0.1, 20.0, 'm', tip='Simulation box width. Fire outside the box is lost, so give it room.', group='Box', log=True, hard_lo=0.02),
        F('size_y', 'Height', 3.2, 0.1, 40.0, 'm', tip='Simulation box height, measured up from the fire base.', group='Box', log=True, hard_lo=0.02),
        F('size_z', 'Depth', 2.0, 0.1, 20.0, 'm', tip='Simulation box depth (toward the camera).', group='Box', log=True, hard_lo=0.02),
        I('resolution', 'Voxels (longest side)', 128, 32, 384, tip='Cells along the longest side of the box for final renders. Detail and memory grow with the cube of this.', group='Resolution', hard_lo=16, hard_hi=768),
        F('preview_scale', 'Interactive resolution', 0.75, 0.25, 1.0, '×', tip='Fraction of the final resolution used while you work. Final renders always use the full resolution.', group='Resolution'),
        B('grow', 'Grow to fit', False, tip='The box starts small around the fire and grows, keeping its cell size, wherever the smoke gets near an open side. Detail stays high near the fire, and no memory goes on empty air until the smoke needs it.', group='Box'),
        F('grow_limit', 'Grow up to', 3.0, 1.0, 8.0, '×', 1, tip='How far the box may grow, as a multiple of its size on each side.', group='Box'),
        B('ground', 'Solid ground', True, tip='The bottom of the box is a floor. Turn off for fire in mid-air.', group='Boundaries'),
        B('open_sides', 'Open sides', True, tip='Gas can leave through the sides. Turn off for fire inside a room or container.', group='Boundaries'),
        B('open_top', 'Open top', True, tip='Gas can leave through the top.', group='Boundaries'),
        F('sponge', 'Edge absorption', 8.0, 0.0, 24.0, 'cells', 0, tip='Width of the layer at open edges where smoke and heat fade out, so the box edge never shows in renders.', group='Boundaries'),
        F('sponge_strength', 'Absorption strength', 6.0, 0.0, 30.0, '/s', 1, group='Boundaries', advanced=True),
        F('time_scale', 'Time scale', 1.0, 0.1, 3.0, '×', anim=True, tip='Below 1 slows the fire down (for slow-motion shots); above 1 speeds it up.', group='Time'),
        F('preroll', 'Pre-roll', 2.0, 0.0, 10.0, 's', 1, tip='Seconds simulated before the first frame so the fire is already burning when the shot starts.', group='Time'),
        I('seed', 'Seed', 1, 0, 9999, tip='Changes the turbulence and emission noise for a different take with the same settings.', group='Time'),
        I('substeps_min', 'Min substeps', 1, 1, 8, group='Solver', advanced=True),
        I('substeps_max', 'Max substeps', 6, 1, 16, tip='Upper limit on simulation steps per frame. Fast fire (explosions) needs more.', group='Solver'),
        F('cfl', 'CFL', 2.0, 0.5, 6.0, 'cells', 1, tip='How many cells gas may move per substep before another substep is added. Lower is more accurate.', group='Solver', advanced=True),
        I('mg_cycles', 'Pressure accuracy', 2, 1, 6, tip='Multigrid cycles per substep. More keeps the gas incompressible in fast, violent flow.', group='Solver'),
        F('maccormack', 'Scalar sharpness', 1.0, 0.0, 1.0, '', 2, tip='MacCormack correction for fire and smoke transport. 1 keeps fine detail; 0 is softer.', group='Solver', advanced=True),
        F('maccormack_vel', 'Velocity sharpness', 0.5, 0.0, 1.0, '', 2, tip='MacCormack correction for the air itself. Higher keeps more swirl but can become unstable.', group='Solver', advanced=True),
        I('mesh_resolution', 'Mesh detail', 96, 32, 192, tip='Resolution of the distance field built from each mesh emitter or collider (cells along its longest side).', group='Solver', advanced=True),
        I('matter_detail', 'Matter detail', 128, 32, 384, tip='How fine sand, snow, mud, jelly and clay are simulated: grid nodes across the box’s longest side, eight particles to a cell. Finer is slower.', group='Solver', advanced=True),
        I('matter_particles', 'Most matter particles', 2000000, 10000, 20000000, tip='The most particles sand, snow, mud, jelly and clay are made of (each takes 128 bytes of GPU memory).', group='Solver', advanced=True),
        F('matter_heat_speed', 'Heat speed', 4.0, 1.0, 50.0, '×', 1, tip='How many times faster than for real wax, chocolate and metal warm, cool and melt, and objects warm and cool: 1 is real time (a bar of chocolate beside a campfire takes a minute or so to melt, a steel bar in a forge minutes to glow), 4 melts the chocolate in a few seconds.', group='Solver', log=True),
        B('disk_cache', 'Disk cache', False, tip='Also keep every simulated frame on disk (next to the project, in NAME.bbcache): frames survive closing the app, a stopped simulation resumes from its last checkpoint, and render farm machines can share one simulation.', group='Cache'),
        Param('cache_dir', 'Cache folder', 'str', '', tip='Where the disk cache goes. Empty: next to the project file (or the user cache folder for an unsaved scene).', group='Cache', advanced=True),
        I('checkpoint_every', 'Checkpoint every', 10, 1, 500, 'frames', tip='How often the whole simulation state is saved, so a simulation can resume from there. Checkpoints are several times larger than ordinary cached frames.', group='Cache', advanced=True),
    ],
    'combustion': [
        F('fuel_scale', 'Master fuel', 1.0, 0.0, 3.0, '×', anim=True, tip='Scales every emitter. Keyframe it to ignite, grow or die down the fire.', group='Fuel'),
        F('ignition', 'Ignition temperature', 0.25, 0.0, 1.5, '', 2, tip='Fuel above this temperature burns. Emitters inject heat to light their own fuel.', group='Fuel'),
        F('burn_rate', 'Burn rate', 6.0, 0.5, 40.0, '/s', 1, tip='How quickly fuel burns once lit. Lower lets fuel travel further before burning: taller, lazier flames.', group='Fuel', log=True),
        F('fuel_dissipation', 'Fuel dissipation', 0.5, 0.0, 5.0, '/s', 2, group='Fuel', advanced=True),
        F('heat', 'Heat release', 0.6, 0.0, 3.0, '', 2, tip='Heat added per unit of burned fuel. Drives buoyancy and flame colour.', group='Heat'),
        F('cooling', 'Cooling', 2.0, 0.0, 10.0, '/s', 2, anim=True, tip='How fast hot gas cools by mixing with air. Higher gives shorter flames.', group='Heat'),
        F('radiative', 'Radiative cooling', 0.0, 0.0, 2.0, '', 2, tip='Extra cooling that grows with the fourth power of temperature: sharper flame tips.', group='Heat'),
        F('temp_cap', 'Temperature limit', 3.0, 1.0, 10.0, '', 1, group='Heat', advanced=True),
        F('flame_gain', 'Flame amount', 1.0, 0.0, 4.0, '', 2, tip='Visible flame produced per unit of burned fuel.', group='Flame'),
        F('flame_life', 'Flame lifetime', 0.08, 0.01, 0.6, 's', 3, tip='How long visible flame lingers after the fuel burns. Longer gives fuller, softer flames.', group='Flame', log=True),
        F('soot', 'Soot yield', 0.3, 0.0, 3.0, '', 2, tip='Smoke produced per unit of burned fuel. Wood 0.2–0.5, rubber or oil 1–3, gas near 0.', group='Smoke'),
        F('smoke_dissipation', 'Smoke dissipation', 0.4, 0.0, 4.0, '/s', 2, anim=True, tip='How fast smoke thins out.', group='Smoke'),
        F('soot_stain', 'Soot stains', 0.0, 0.0, 2.0, '/s', 2, tip='Smoke that touches walls, ceilings, colliders and the floor leaves soot on them, heaviest where hot smoke pools under a ceiling or pours out of a doorway. It shows in the composite (Composite › Soot) and as a soot layer in EXRs. 0 is off.', group='Smoke'),
        F('expansion', 'Gas expansion', 0.5, 0.0, 6.0, '', 2, anim=True, tip='Volume released by burning fuel. High values make fireballs billow outward.', group='Expansion'),
        F('thermal_expansion', 'Heat expansion', 0.0, 0.0, 1.0, '', 2, tip='Gas swells as it heats and shrinks as it cools, as real air does (1 is physical). Fireballs billow harder and a hot closed room pushes smoke out of every gap; 0 leaves the swelling to Gas expansion.', group='Expansion'),
        F('flame_speed', 'Flame speed', 0.0, 0.0, 20.0, 'm/s', 2, tip='Premixed flame fronts: fuel that has already mixed with air catches from any flame next to it and a front runs through it at this speed, however cold it is. A flash fire across a spilled-fuel vapour, a gas cloud going up, or the fireball when air gets into a smoke-filled room (backdraft). Hydrocarbon vapours 2–8 m/s in turbulent air. 0 is off.', group='Flame front'),
        F('fuel_weight', 'Fuel vapour weight', 0.0, 0.0, 5.0, 'm/s²', 2, tip='How much heavier than air unburnt fuel vapour is: petrol, propane and butane vapour sink and spread along the ground, pooling in low places, until something lights them. 0 is as light as air (natural gas is lighter still).', group='Fuel'),
        F('rich', 'Oxygen limit', 4.0, 0.2, 20.0, '', 2, tip='Fuel level at which lack of air halves the burn rate. Lower it for fuel-rich fireballs that burn from the outside in.', group='Fuel', log=True),
        F('expansion_cap', 'Expansion limit', 40.0, 1.0, 200.0, '/s', 1, tip='Upper limit on how fast burning gas can expand in one place. Keeps violent bursts stable.', group='Expansion', advanced=True),
        E('air', 'Air supply', 'open', (('open', 'Open air'), ('tracked', 'Tracked (rooms, backdraft)')),
          tip='Open air: the fire always finds oxygen; the oxygen limit only models mixing. Tracked: burning uses up the air, so a fire in a closed space starves, unburnt fuel builds up, and fresh air let in later can make it flash over (backdraft).', group='Air'),
        F('air_use', 'Air use', 1.0, 0.0, 5.0, '', 2, tip='With tracked air: how much of the air in a place each unit of burned fuel uses up. Higher starves a closed-room fire sooner.', group='Air'),
        F('air_mixing', 'Air mixing', 0.03, 0.0, 0.3, 'm²/s', 3, tip='With tracked air: how strongly used-up air mixes with the air around it (turbulent mixing too fine for the grid). It lets starved, fuel-rich gas that meets fresh air ignite. Real fires: 0.01–0.1.', group='Air', advanced=True),
        F('steam_yield', 'Steam from dousing', 150.0, 0.0, 1000.0, 'g/m³', 0, tip='Water vapour made when an emitter with Put out takes heat from the gas, per unit of temperature removed. Water on a fire flashes to steam.', group='Steam'),
        F('water_yield', 'Water from burning', 0.0, 0.0, 200.0, 'g/m³', 0, tip='Water vapour made per unit of burned fuel. Real flames make plenty; it shows as condensing steam in cold, humid air.', group='Steam'),
        F('vapour_dissipation', 'Vapour mixing', 1.5, 0.0, 10.0, '/s', 2, anim=True, tip='How fast water vapour mixes away into the surrounding air. Steam evaporates at its edges as it thins.', group='Steam'),
        F('water_douse', 'Water on flames', 40.0, 0.0, 200.0, '/s', 1, tip='With liquid in the same box: how fast a place full of water puts out the flames and hot gas in it (and turns their heat to steam).', group='Water on fire'),
        F('soak', 'Soaking', 6.0, 0.0, 40.0, '/s', 1, tip='With liquid in the same box: how fast water reaching burning fuel soaks it. Soaked fuel stops burning; once the water stops it dries out and can catch again.', group='Water on fire'),
        F('ember_heat', 'Ember heat', 150.0, 0.0, 400.0, '×', 0, tip="With liquid in the same box: how much heat a burning fuel bed holds, as a multiple of the air's (a bed of glowing charcoal about 150, a thin layer of burning leaves 20). Water on hot embers boils off in a thick white plume of steam that goes on after the flames are out. 0 makes no steam from the embers.", group='Water on fire'),
        F('rekindle', 'Drying in flames', 8.0, 0.5, 120.0, 's', 1, tip='With liquid in the same box: how long soaked fuel takes to dry out with flames around it before it can burn again. Short, and a doused corner of the fire soon catches again from the rest; long, and dousing puts it out for good.', group='Water on fire', log=True),
        F('steam_expansion', 'Steam burst', 0.5, 0.0, 2.0, '', 2, tip='With liquid in the same box: how much water flashing to steam swells and pushes the gas out (1 is physical): the gust of steam, smoke and sparks when water hits a fire.', group='Water on fire'),
        F('latent_heat', 'Latent heat', 1.0, 0.0, 2.0, '', 2, tip='Heat given off as steam condenses (1 is physical). It warms the cloud so steam billows upward like a cumulus; 0 lets it drift.', group='Steam'),
    ],
    'motion': [
        F('buoyancy', 'Buoyancy', 5.0, 0.0, 30.0, 'm/s²', 1, anim=True, tip='Upward acceleration of hot gas per unit of temperature.', group='Buoyancy'),
        F('soot_weight', 'Smoke weight', 0.02, 0.0, 1.0, 'm/s²', 3, tip='Downward pull of soot. Keep small: cooled smoke is close to neutrally buoyant.', group='Buoyancy', advanced=True),
        F('damping', 'Air drag', 0.05, 0.0, 2.0, '/s', 2, group='Buoyancy', advanced=True),
        F('puffing', 'Puffing', 0.0, 0.0, 1.0, '', 2, anim=True, tip='How strongly the flames pulse. Real fires bulge and pinch off at the tip about 1.5/√D times a second, where D is the width of the base in metres: twice a second for a campfire, under once a second for a bonfire. The grid is too coarse to form these eddies by itself, so the heat and the rise of the gas over each emitter surge at the rate for its size (the flame is a little brighter for it). 0 is off.', group='Puffing'),
        F('vorticity', 'Vorticity', 1.5, 0.0, 10.0, '', 2, anim=True, tip='Vorticity confinement: restores small swirls lost to numerical smoothing.', group='Detail'),
        F('vort_outside', 'Vorticity in smoke', 0.3, 0.0, 1.0, '×', 2, tip='Fraction of the vorticity applied outside hot or burning gas.', group='Detail'),
        F('turbulence', 'Turbulence', 3.0, 0.0, 30.0, 'm/s²', 1, anim=True, tip='Divergence-free noise force in hot gas. Makes flames lick and roll.', group='Turbulence'),
        F('turb_freq', 'Turbulence frequency', 2.5, 0.1, 20.0, '/m', 2, tip='Spatial frequency of the turbulence. Scale with fire size: higher for small fires.', group='Turbulence', log=True),
        F('turb_rise', 'Turbulence rise', 1.5, 0.0, 10.0, 'm/s', 1, tip='How fast the turbulence pattern travels upward with the flames.', group='Turbulence'),
        F('turb_evolve', 'Turbulence evolution', 0.8, 0.0, 5.0, '/s', 2, group='Turbulence'),
        F('turb_octave2', 'Fine turbulence', 0.5, 0.0, 1.0, '×', 2, group='Turbulence'),
        F('disturbance', 'Disturbance', 2.0, 0.0, 20.0, 'm/s²', 1, anim=True, tip='Small random kicks that break up smooth flame sheets into flickering detail.', group='Disturbance'),
        F('disturb_block', 'Disturbance size', 0.05, 0.005, 1.0, 'm', 3, group='Disturbance', log=True),
        F('disturb_rate', 'Disturbance rate', 24.0, 1.0, 120.0, 'Hz', 0, group='Disturbance', advanced=True),
        F('mask_temp', 'Mask: temperature', 1.0, 0.0, 4.0, '×', 2, group='Force masks', advanced=True),
        F('mask_flame', 'Mask: flame', 1.0, 0.0, 4.0, '×', 2, group='Force masks', advanced=True),
        F('mask_smoke', 'Mask: smoke', 0.0, 0.0, 4.0, '×', 2, group='Force masks', advanced=True),
        F('wind_speed', 'Wind speed', 0.0, 0.0, 20.0, 'm/s', 2, anim=True, tip='Steady wind. Light breeze 1–3 m/s, strong wind 8+.', group='Wind'),
        F('wind_dir', 'Wind direction', 90.0, -180.0, 180.0, '°', 0, anim=True, tip='Direction the wind blows toward, around the vertical axis. 90° blows to screen right in the default camera.', group='Wind'),
        F('gust', 'Gustiness', 0.3, 0.0, 1.0, '', 2, anim=True, tip='How much the wind speed and direction wander.', group='Wind'),
        F('gust_freq', 'Gust frequency', 0.4, 0.05, 3.0, 'Hz', 2, group='Wind'),
        F('wind_relax', 'Wind coupling', 0.4, 0.0, 5.0, '/s', 2, tip='How strongly the air in the box is pulled toward the wind speed.', group='Wind'),
    ],
    'shading': [
        F('flame_k', 'Flame temperature', 1650.0, 900.0, 3000.0, 'K', 0, anim=True, tip='Colour temperature of flame at the reference heat. Wood fire 1400–1800 K, candle 1500–1900 K.', group='Flame colour'),
        F('max_k', 'Hottest core', 2250.0, 1000.0, 6000.0, 'K', 0, tip='Colour temperature the hottest gas approaches.', group='Flame colour'),
        F('dynamic_range', 'Physical brightness', 0.75, 0.0, 1.5, '', 2, tip='1 follows real blackbody brightness (cool parts go dark red fast); lower flattens it for a softer look.', group='Flame colour'),
        B('exposure_physical', 'Real-world brightness', False, tip='Make flame as bright as real flame of its temperature, against how brightly lit the scene in the footage is (Scene light). Fire exposure then adjusts it from there. Real flame is many stops brighter than a scene at night, a few brighter than one at dusk and about as bright as one in full sun.', group='Flame brightness'),
        F('scene_ev', 'Scene light', 9.0, 0.0, 17.0, 'EV', 1, anim=True, tip='With Real-world brightness: how brightly lit the scene in the footage is, as a light meter would read it (EV at ISO 100). About 3 to 5 for a street at night, 6 to 8 indoors, 9 to 11 at dusk, 12 to 13 on an overcast day, 15 in full sun.', group='Flame brightness'),
        F('exposure', 'Fire exposure', 0.0, -6.0, 6.0, 'EV', 2, anim=True, tip='Brightness of the fire in stops. At 0, thick flame at the flame temperature renders at full white.', group='Flame brightness'),
        F('intensity', 'Emission', 1.0, 0.0, 10.0, '×', 2, group='Flame brightness', advanced=True),
        F('flame_density', 'Flame density', 1.0, 0.0, 4.0, '×', 2, group='Flame shape'),
        F('flame_sharpness', 'Flame sharpness', 1.6, 0.5, 4.0, '', 2, tip='Contrast of flame edges. Higher gives crisper, more defined tongues.', group='Flame shape'),
        F('flame_threshold', 'Flame threshold', 0.02, 0.0, 0.5, '', 3, tip='Trims faint wisps of flame.', group='Flame shape'),
        F('flame_absorption', 'Flame thickness', 4.0, 0.2, 200.0, '/m', 2, tip='How quickly flame becomes optically thick. Thick flame glows at full blackbody brightness; thin flame is translucent and dimmer. Small flames need higher values.', group='Flame shape', log=True),
        F('flame_occlusion', 'Flame occlusion', 0.35, 0.0, 1.0, '', 2, tip='How much flames hide what is behind them in the alpha channel.', group='Flame shape'),
        F('soot_glow', 'Glowing soot', 0.25, 0.0, 3.0, '', 2, tip='Incandescence of hot smoke just above the flames.', group='Flame shape'),
        F('blue', 'Base glow', 0.0, 0.0, 5.0, '', 2, tip='Glow where fuel is burning: the blue chemiluminescence of gas and alcohol flames. Change its colour for other chemistry.', group='Flame colour'),
        E('colour_response', 'Colour as seen by', 'camera', (('camera', 'A camera'), ('eye', 'The eye')), tip='Blackbody colours as a typical camera sensor records them (a little more yellow, and always within the colours a display can show) or as the eye sees them.', group='Flame colour'),
        C('blue_color', 'Base glow colour', (0.12, 0.30, 1.0), tip='Colour of the base glow. Blue for gas and alcohol; green for copper or boron, red for strontium, orange for sodium.', group='Flame colour'),
        F('colour_gain', 'Colourant brightness', 1.0, 0.0, 10.0, '×', 2, tip='Brightness of the flame colourants released by emitters (Emitter › Colour).', group='Flame colour'),
        F('smoke_density', 'Smoke density', 6.0, 0.0, 60.0, '/m', 1, anim=True, tip='How thick the smoke looks.', group='Smoke'),
        C('smoke_albedo', 'Smoke colour', (0.22, 0.21, 0.20), tip='Scattering colour of the smoke. Wood smoke is mid grey; oil and rubber smoke is near black.', group='Smoke'),
        F('anisotropy', 'Forward scattering', 0.35, -0.9, 0.9, '', 2, tip='How much light scatters forward through smoke (backlit smoke glows).', group='Smoke'),
        F('multiple_scattering', 'Multiple scattering', 1.0, 0.0, 1.0, '', 2, tip='Light bouncing around inside the smoke more than once (1 is physical). Thick, pale smoke and steam glow white inside instead of going grey; dark smoke barely changes.', group='Smoke'),
        F('detail', 'Detail noise', 0.35, 0.0, 1.0, '', 2, tip='Adds fine render-time detail beyond the simulation grid.', group='Detail'),
        F('detail_freq', 'Detail frequency', 3.0, 0.2, 30.0, '/m', 2, group='Detail', log=True),
        F('detail_disp', 'Detail displacement', 1.2, 0.0, 4.0, 'cells', 2, group='Detail'),
        F('detail_rise', 'Detail rise', 1.0, 0.0, 10.0, 'm/s', 2, group='Detail', advanced=True),
        F('step', 'Ray step', 0.7, 0.25, 2.0, 'cells', 2, tip='Sampling step of the renderer. Smaller is cleaner and slower.', group='Quality', advanced=True),
        F('edge_fade', 'Edge fade', 10.0, 0.0, 30.0, 'cells', 0, group='Quality', advanced=True),
        F('ambient_k', 'Air temperature', 300.0, 250.0, 400.0, 'K', 0, tip='Temperature of the surrounding air (300 K is 27 °C, 273 K is freezing). Steam condenses far more readily in cold air: breath and exhaust on a winter day. Also the colour of the coolest gas.', group='Steam'),
        F('humidity', 'Air humidity', 50.0, 0.0, 100.0, '%', 0, tip='Relative humidity of the surrounding air. Humid air lets steam linger; dry air evaporates it quickly.', group='Steam'),
        F('steam_density', 'Steam density', 0.4, 0.0, 5.0, '/m', 2, tip='How thick condensed steam looks, per gram of water droplets per cubic metre.', group='Steam'),
        C('steam_albedo', 'Steam colour', (0.92, 0.93, 0.95), tip='Scattering colour of the water droplets. Near white.', group='Steam'),
        F('coal_bed', 'Coal bed', 0.0, 0.0, 4.0, '', 2, anim=True, tip='Glowing coals on the ground where the flames burn down onto it, over charred ground: the ember bed of a campfire or bonfire, for footage that has none. At 1 the coals glow at about half the brightness of thick flame. 0 is none.', group='Coal bed'),
        F('coal_k', 'Coal temperature', 1250.0, 800.0, 1600.0, 'K', 0, tip='How hot the coals glow: 900 K dull red, 1100 K cherry red, 1250 K orange, 1400 K yellow-orange.', group='Coal bed'),
        F('coal_height', 'Coal bed height', 0.08, 0.01, 0.5, 'm', 3, tip='How high the coals are heaped at the middle of the bed: a few centimetres for a campfire, 20 cm or more under a bonfire.', group='Coal bed'),
    ],
    'lighting': [
        C('ambient', 'Ambient light', (0.10, 0.11, 0.13), anim=True, tip='Light reaching the smoke from the surroundings (sky, room). Match it to your footage.', group='Ambient'),
        F('ambient_intensity', 'Ambient intensity', 1.0, 0.0, 10.0, '×', 2, anim=True, group='Ambient'),
        E('sky', 'Sky', 'colour', (('colour', 'Ambient colour'), ('physical', 'Physical (sun and air)')),
          tip='Physical: the sky and the sun\'s light worked out from the air itself for where the key light is: a deep blue '
          'sky over a high sun, a pale haze at the horizon, the sun yellowing and reddening as it sets and the sky glowing '
          'round it, all in step. The key light becomes the sun (its colour from the air; Key intensity is its strength '
          'above the air) and the sky lights everything, seen behind the set too. An Environment (HDRI) file overrides it.',
          group='Sky'),
        F('haze', 'Haze', 0.1, 0.01, 1.0, '', 3, log=True, tip='How much haze (dust, smoke, water droplets) the air holds: '
          'its optical depth straight up. 0.02 clear mountain air, 0.1 a clear day, 0.3 hazy, 0.6 a thick summer haze. '
          'Haze whitens the sky, brightens it round the sun and dims and yellows the sun.', group='Sky'),
        F('ground_albedo', 'Ground brightness', 0.3, 0.0, 1.0, '', 2, tip='How much light the land round about sends back '
          'up into the sky: 0.1 forest, 0.3 fields, 0.8 fresh snow.', group='Sky'),
        F('altitude', 'Altitude', 0.0, 0.0, 6000.0, 'm', 0, tip='The height above the sea: higher up, the sky is a deeper '
          'blue and the sun whiter.', group='Sky'),
        B('ambient_from_footage', 'Match ambient to footage', True, tip='Light the smoke with the average colour and brightness of the footage, so it sits in the scene. Ambient intensity scales it.', group='Ambient'),
        B('sun_on', 'Key light', False, tip='A directional light (sun, moon, lamp) that lights and shadows the smoke.', group='Key light'),
        F('sun_azimuth', 'Key azimuth', 35.0, -180.0, 180.0, '°', 0, anim=True, group='Key light'),
        F('sun_elevation', 'Key elevation', 40.0, -10.0, 90.0, '°', 0, anim=True, group='Key light'),
        C('sun_color', 'Key colour', (1.0, 0.95, 0.88), group='Key light'),
        F('sun_intensity', 'Key intensity', 3.0, 0.0, 20.0, '×', 2, anim=True, group='Key light'),
        F('shadow', 'Shadow density', 1.0, 0.0, 4.0, '×', 2, group='Key light'),
        F('fire_scatter', 'Fire lights smoke', 1.0, 0.0, 5.0, '×', 2, tip='How brightly the flames light the smoke around them.', group='Fire light'),
        Param('environment', 'Environment (HDRI)', 'file', '', tip='A panoramic HDR image of the set (latitude-longitude .hdr or .exr), shot where the effect sits. Liquids reflect and refract it; empty uses the footage and the ambient colour.', group='Environment'),
        F('env_rotation', 'Environment rotation', 0.0, -180.0, 180.0, '°', 1, tip='Turns the environment around the vertical axis to line it up with the set.', group='Environment'),
        F('env_strength', 'Environment strength', 1.0, 0.0, 10.0, '×', 2, tip='Brightness of the environment.', group='Environment', log=True),
        B('env_sun', 'Key light from environment', False, tip='Aim the key light at the brightest spot of the environment (the sun), with its colour.', group='Environment'),
        F('light_spread', 'Fire light reach', 0.35, 0.05, 3.0, 'm', 2, tip='How far the fire light spreads through the smoke.', group='Fire light', log=True),
    ],
    'embers': [
        B('enabled', 'Embers', True, group='Emission'),
        F('rate', 'Rate', 60.0, 0.0, 2000.0, '/s', 0, tip='Embers launched per second across all emitters that emit embers.', group='Emission', log=True),
        I('count', 'Pool size', 16384, 1024, 262144, tip='Most embers alive at once. The oldest are recycled.', group='Emission', advanced=True),
        F('lifetime', 'Lifetime', 2.2, 0.1, 10.0, 's', 2, group='Emission'),
        F('life_jitter', 'Lifetime variation', 0.6, 0.0, 1.0, '', 2, group='Emission', advanced=True),
        F('launch', 'Launch speed', 1.8, 0.0, 40.0, 'm/s', 2, group='Launch'),
        F('spread', 'Spread', 0.7, 0.0, 10.0, 'm/s', 2, tip='Random extra speed in every direction at launch.', group='Launch'),
        V('direction', 'Launch direction', (0.0, 1.0, 0.0), -1.0, 1.0, tip='Direction particles are launched in (fire-local). Straight up for embers; along the cut for grinder sparks.', group='Launch'),
        F('cone', 'Launch cone', 0.0, 0.0, 180.0, '°', 1, tip='Half-angle of the cone particles are launched into around the launch direction. 180 sprays every way, as in a firework burst.', group='Launch'),
        F('drag', 'Air drag', 3.0, 0.1, 20.0, '/s', 2, tip='How closely embers follow the air. Small embers ride the updraft; hot metal sparks barely feel it.', group='Motion', log=True),
        F('gravity', 'Gravity', 1.0, 0.0, 9.81, 'm/s²', 2, tip='Light, buoyant embers fall slowly (about 1). Sparks of metal fall at full gravity (9.81).', group='Motion'),
        F('turbulence', 'Turbulence', 3.0, 0.0, 30.0, 'm/s²', 1, group='Motion'),
        F('turb_freq', 'Turbulence frequency', 2.0, 0.1, 20.0, '/m', 2, group='Motion', advanced=True),
        F('temperature', 'Launch temperature', 1850.0, 900.0, 3000.0, 'K', 0, group='Look'),
        F('cooling', 'Cooling', 0.8, 0.0, 5.0, '/s', 2, group='Look'),
        F('size_min', 'Size min', 0.002, 0.0005, 0.05, 'm', 4, group='Look', log=True),
        F('size_max', 'Size max', 0.010, 0.0005, 0.1, 'm', 4, group='Look', log=True),
        F('brightness', 'Brightness', 5.0, 0.0, 50.0, '×', 2, group='Look', log=True),
        F('min_width_px', 'Min width', 1.3, 0.5, 4.0, 'px', 2, group='Look', advanced=True),
        F('fade_in', 'Fade in', 0.06, 0.0, 1.0, 's', 2, group='Look', advanced=True),
        F('shutter', 'Streak length', 0.5, 0.0, 1.0, 'frame', 2, tip='Motion blur of each ember, as a fraction of a frame (0.5 = 180° shutter).', group='Look'),
        B('collide', 'Hit colliders', True, tip='Embers and sparks bounce off colliders as well as the ground.', group='Collisions'),
        F('bounce', 'Bounce', 0.25, 0.0, 1.0, '', 2, tip='How much speed a particle keeps off a surface, straight back out. Sparks skitter; embers settle.', group='Collisions'),
        F('friction', 'Friction', 0.5, 0.0, 1.0, '', 2, tip='How much sliding speed a particle loses when it hits a surface.', group='Collisions'),
    ],
    'spread': [
        B('enabled', 'Spreading fire', False, tip='Let fire spread by itself across burnable surfaces: the ground, and colliders marked Burnable. A surface catches where hot gas touches it, flames for a while, smoulders, then is burnt out. Light it with any emitter, even one that only burns for a moment.', group='Spreading fire'),
        B('ground', 'Ground burns', True, tip='The floor of the box is burnable: grass, brush, a fuel spill. Needs Domain › Solid ground.', group='Ground'),
        F('area_x', 'Burnable width', 0.0, 0.0, 40.0, 'm', 2, tip='Width of the burnable patch of ground, centred on the fire base. 0 uses the whole floor.', group='Ground'),
        F('area_z', 'Burnable depth', 0.0, 0.0, 40.0, 'm', 2, tip='Depth of the burnable patch of ground, centred on the fire base. 0 uses the whole floor.', group='Ground'),
        F('coverage', 'Coverage', 0.85, 0.0, 1.0, '', 2, tip='Share of the surface that carries fuel. Lower leaves bare patches the fire has to go around.', group='Fuel bed'),
        F('patch_freq', 'Patch frequency', 1.5, 0.1, 20.0, '/m', 2, tip='Size of the fuel patches: higher gives smaller, more broken patches.', group='Fuel bed', log=True),
        F('burn_time', 'Burn time', 3.0, 0.1, 60.0, 's', 2, tip='How long a spot flames once it catches. Dry grass 1–3 s, wood and furniture much longer.', group='Fuel bed', log=True),
        F('fuel', 'Fuel', 8.0, 0.0, 100.0, '/s', 1, tip='Fuel a burning spot gives off per second: the size of its flames.', group='Fuel bed'),
        F('heat', 'Heat', 0.45, 0.0, 3.0, '', 2, tip='Temperature of a burning surface. Must exceed the ignition temperature to light its own fuel.', group='Fuel bed'),
        F('smoke', 'Smoke', 0.5, 0.0, 20.0, '/s', 2, tip='Smoke a burning spot gives off directly, on top of the smoke from its flames.', group='Fuel bed'),
        F('catch_temp', 'Catches at', 0.35, 0.0, 2.0, '', 2, tip='Gas temperature at which a surface starts heating toward catching fire. Flame is about 1.', group='Catching'),
        F('catch_time', 'Catch time', 0.4, 0.01, 10.0, 's', 2, tip='How long a surface must sit in hot gas before it catches. Longer makes the fire slower to spread.', group='Catching', log=True),
        F('creep', 'Creep speed', 0.05, 0.0, 2.0, 'm/s', 3, tip='Spread from a burning spot to the spots next to it by contact and radiant heat, even in still air. Wind-blown flames spread much faster through the gas.', group='Catching'),
        F('smoulder', 'Smoulder time', 4.0, 0.0, 60.0, 's', 1, tip='After flaming, a spot smoulders (smoke and a little heat) for this long, then is burnt out for good.', group='Burning out'),
        F('smoulder_smoke', 'Smoulder smoke', 1.0, 0.0, 20.0, '/s', 2, group='Burning out'),
        F('burn_speed', 'Burn speed-up', 1.0, 1.0, 1000.0, '×', 0, anim=True, tip='Wood catches, chars and burns this many '
          'times faster than in reality, while the flames and the smoke move at their own speed: a time-lapse of a shed '
          'burning down (real wood takes minutes to catch all over and tens of minutes to burn through). 1 is real time. '
          'Keyframe it to let a fire grow at its own pace, then speed through the burning down. Wood only: other burnable '
          'things go by Burn time and Catch time.', group='Burning out', log=True),
        F('dry_time', 'Drying time', 20.0, 0.0, 600.0, 's', 1, tip='Water that puts a surface out soaks it: it cannot catch again until it has dried, which takes this long (less in hot gas). Wet surfaces show darker in the composite. 0: it can catch again at once.', group='Water'),
        F('spotting', 'Spot fires', 0.0, 0.0, 1.0, '', 3, tip='Chance that a hot ember landing on the burnable ground starts a new fire there, ahead of the main fire. Wind-blown embers then carry the fire across gaps. Needs Embers.', group='Spot fires'),
        F('spot_temp', 'Ember heat needed', 900.0, 400.0, 2000.0, 'K', 0, tip='Embers that have cooled below this land without starting anything.', group='Spot fires', advanced=True),
    ],
    'liquid': [
        F('gravity', 'Gravity', 9.81, 0.0, 30.0, 'm/s²', 2, tip='Downward acceleration. 9.81 is Earth.', group='Forces'),
        F('flip', 'Splashiness', 0.9, 0.0, 1.0, '', 2, tip='How much each particle keeps its own motion (FLIP) versus the smooth average of its neighbours (PIC). High is lively and splashy, like water; low is calm and damped, like syrup.', group='Behaviour'),
        F('surface_tension', 'Surface tension', 0.073, 0.0, 0.3, 'N/m', 3, tip='Pulls the surface smooth and beads small drops. Water 0.073, soapy water 0.03, oils 0.02 to 0.03. It matters at centimetre scale and below; on big shots it has no visible effect.', group='Behaviour'),
        F('wall_drag', 'Grip on surfaces', 2.0, 0.0, 20.0, 'm/s²', 2, tip='How strongly surfaces slow the liquid touching them. Fast sheets still slide on; slow films stop and pin, so spills settle into puddles instead of spreading forever. 0 is perfectly slippery.', group='Behaviour'),
        F('drying', 'Drying', 0.02, 0.0, 1.0, '/s', 3, tip='How fast wet ground dries after the liquid has gone. 0 stays wet.', group='Behaviour'),
        F('viscosity', 'Viscosity', 0.0, 0.0, 200.0, 'Pa·s', 3, tip='How thick the liquid is. Water 0.001 (no visible effect), olive oil 0.08, syrup 2 to 5, honey 10, molten lava 100 and up. Thick liquids coil, fold and flow slowly.', group='Behaviour', log=True),
        F('liquid_density', 'Density', 1000.0, 500.0, 3000.0, 'kg/m³', 0, tip='Mass per volume. Water 1000, oil 900, honey 1400, lava 2600. Sets what floats in it.', group='Behaviour', log=True),
        F('contact_angle', 'Contact angle', 90.0, 0.0, 180.0, '°', 0, tip='How the liquid meets surfaces: below 90 it wets them and spreads (clean glass, concrete), above 90 it beads up (a waxed car, a leaf). Works with surface tension.', group='Behaviour'),
        F('cooling', 'Cooling', 0.0, 0.0, 2.0, '/s', 3, tip="How fast a molten liquid (lava, wax, molten glass) loses its heat where it meets the air and the ground. As it cools its glow dims and reddens, a crust covers it and it stiffens until it sets. The inside stays hot much longer. 0 never cools.", group='Molten'),
        F('solidify', 'Setting', 1000.0, 1.0, 100000.0, '×', 0, tip='How much thicker the liquid is once cooled than when molten. 1000 and up and it sets solid: flows stop where they have cooled, and a crust holds.', group='Molten', log=True),
        B('thermal', 'Heat and phase changes', False, tip="Give the liquid a temperature. Water then freezes into ice (which floats and moves as solid pieces, and freezes onto cold surfaces), melts, boils in bubbles on hot surfaces and evaporates, with the real heat capacities and latent heats of water. Set the temperatures here, on sources (Heat) and on colliders (Heat). With fire in the box the fire heats it too.", group='Heat'),
        F('liquid_temp', 'Liquid temperature', 20.0, -40.0, 100.0, '°C', 1, tip='Temperature of the liquid as sources pour it (unless a source sets its own) and of the open water. Below freezing a source pours ice.', group='Heat'),
        F('air_temp', 'Air temperature', 20.0, -60.0, 60.0, '°C', 1, tip='Temperature of the air round the liquid. Water loses heat to cold air and evaporates into dry air. In a fire and liquid box the gas is the air instead (Shading: Air temperature, Air humidity).', group='Heat'),
        F('air_humidity', 'Air humidity', 50.0, 0.0, 100.0, '%', 0, tip='Relative humidity of that air. Dry air evaporates water fast (and cools it); saturated air not at all.', group='Heat'),
        F('ground_temp', 'Ground temperature', 20.0, -200.0, 600.0, '°C', 0, tip='The ground, closed walls and the surfaces in the footage. A hot plate 150 to 250 (water on it boils; past about 210 drops dance on their own vapour), a stove top 400, frozen ground -10, dry ice -78, liquid nitrogen -196 (water on it freezes where it lands).', group='Heat'),
        F('heat_speed', 'Heat speed-up', 1.0, 1.0, 10000.0, '×', 0, tip='Heat flows this many times faster than in reality, while the liquid still moves at its real speed: a time-lapse of freezing, melting and drying. Real ice takes minutes to hours: a pond skins over in about 10 minutes of hard frost, an ice cube melts in a drink in 10 to 20. 1 is physical.', group='Heat', log=True),
        F('supercool', 'Supercooling', 1.0, 0.0, 40.0, 'K', 1, tip='How far below freezing still water can cool before ice forms in it by itself. Ice next to it, a cold surface or a disturbance seeds it sooner. Pure, undisturbed water holds 10 to 20: seeded, it then flashes to slush as dendrites race through it.', group='Heat', advanced=True),
        F('freeze_point', 'Freezing point', 0.0, -30.0, 30.0, '°C', 1, tip='Temperature the liquid freezes at. Fresh water 0, sea water -1.9, brine lower.', group='Heat', advanced=True),
        F('boil_point', 'Boiling point', 100.0, 50.0, 200.0, '°C', 1, tip='Temperature the liquid boils at. Water 100 at sea level, 93 at 2000 m, 120 in a pressure cooker.', group='Heat', advanced=True),
        F('bubble_size', 'Steam bubble size', 2.5, 0.5, 10.0, 'mm', 1, tip='Diameter of the steam bubbles boiling water sends up from a hot surface. Hard boiling merges them into bigger ones.', group='Heat', advanced=True),
        B('footage_collide', 'Hits the footage', False, tip='The liquid collides with the scene in your footage, using its depth pass (Composite › Holdout depth): water splashes against real walls, runs down real steps. Needs a depth pass of the footage; a matte alone does not tell how far things are.', group='Behaviour'),
        F('rain', 'Rain', 0.0, 0.0, 200.0, 'mm/h', 1, anim=True, tip='Rain falling on the scene: streaks through the air, and ring ripples and splashes where drops hit water. Drizzle 1, steady rain 5, heavy 20, a downpour 50 and up.', group='Rain'),
        F('rain_drop', 'Raindrop size', 2.5, 0.5, 6.0, 'mm', 1, tip='Typical raindrop diameter. Drizzle 0.5, rain 2 to 3, a thunderstorm 4 to 5.', group='Rain', advanced=True),
        F('water_level', 'Water level', 0.0, 0.0, 40.0, 'm', 3, anim=True, tip='Water that carries on past the open sides of the box up to this height (a pond, a flooded street, the sea). It fills the box to this level at the start, keeps it topped up at the sides, lets waves run out, and is drawn out to the horizon. Keyframe it for a tide (slowly: a fast rise is a bore, see Surge). 0 is off.', group='Open water'),
        F('level_absorb', 'Edge absorption', 8.0, 0.0, 32.0, 'cells', 0, tip='Width of the layer at the open sides where waves are calmed so they do not bounce back off the box edge.', group='Open water', advanced=True),
        F('ocean_height', 'Wave height', 0.0, 0.0, 10.0, 'm', 2, anim=True, tip="Waves on the open water: a random sea, as tall as this from trough to crest on average (the significant wave height). A calm lake 0.05, a breezy bay 0.3, a rough sea 1.5 and up. The box's own water rides them, and they carry on to the horizon. Needs a water level; waves are held to less than half of the water's depth, as deeper ones would break. 0 is flat.", group='Open water'),
        F('ocean_length', 'Wavelength', 6.0, 0.3, 200.0, 'm', 2, anim=True, tip='Distance between the crests of the main waves. Ripples on a pond 0.5, a harbour chop 3, an ocean swell 50 to 150.', group='Open water', log=True),
        F('ocean_dir', 'Wave direction', 90.0, -180.0, 180.0, '°', 0, anim=True, tip='Direction the waves travel toward, around the vertical axis (as the wind direction).', group='Open water'),
        F('ocean_spread', 'Crest spread', 0.3, 0.0, 1.0, '', 2, tip='How much the waves come from different directions: 0 long, parallel crests (a swell from a far storm), 1 a confused, choppy sea.', group='Open water'),
        F('ocean_chop', 'Choppiness', 0.6, 0.0, 1.5, '', 2, tip='Sharpens the crests and flattens the troughs, as real waves are. High values fold the crests over into whitecaps.', group='Open water'),
        F('swell_height', 'Swell height', 0.0, 0.0, 10.0, 'm', 2, anim=True, tip="A swell from a storm far away, on top of the local waves: long, smooth, parallel crests running in from their own direction. Most real seas have one. 0 is none. Like Wave height, the significant height.", group='Open water'),
        F('swell_length', 'Swell wavelength', 60.0, 2.0, 400.0, 'm', 1, anim=True, tip='Distance between the crests of the swell: 30 on a lake after a blow, 80 to 300 on the open ocean.', group='Open water', log=True),
        F('swell_dir', 'Swell direction', 90.0, -180.0, 180.0, '°', 0, anim=True, tip='Direction the swell travels toward, around the vertical axis. It need not follow the wind.', group='Open water'),
        F('ocean_depth', 'Sea depth', 0.0, 0.0, 1000.0, 'm', 1, tip="How deep the sea is, for how its waves travel: in shallow water they slow, shorten and steepen. 0 takes the water level (the box's floor is the sea bed). Set it deeper for the open sea, where the box is only its top layer.", group='Open water', advanced=True),
        F('open_water_area', 'Open water area', 6.0, 0.0, 20.0, '×', 1, tip="How far round the box the open water carries on what happens in it: the box's wakes and ripples spread out across it, its foam drifts off with the current, the current flows round islands and posts in it, and the sea bed under it makes waves shoal and break into surf. As a multiple of the box's size; 0 turns it off.", group='Open water', advanced=True),
        E('ocean_detail', 'Wave detail', '256', (('128', 'Draft (128)'), ('256', 'Standard (256)'), ('512', 'Fine (512)')), tip='Resolution of each of the three scales of waves (from the swell down to the ripples). Fine costs more memory and time for crisper close-ups.', group='Open water', advanced=True),
        E('sea_from', 'Waves come in', 'all', (('all', 'Through every open side'), ('upwave', 'Only from where they come from')),
          tip="Where the sea's waves enter the box. Every open side suits open water. For waves running up a beach or breaking over a reef, let them in only from the side they come from: the other sides become walls along the waves (as in a wave tank), so the breaking surf is not pulled back toward the open sea's shape.", group='Open water'),
        F('surge_height', 'Surge', 0.0, 0.0, 30.0, 'm', 2, tip='A single long wave rolling in on top of the sea: a tsunami, a solitary wave, a flood or tidal bore. How high it stands above the water. It runs up any beach or shore in the box. 0 is none.', group='Surge'),
        F('surge_length', 'Surge length', 40.0, 1.0, 5000.0, 'm', 1, tip='How long the surge is, front to back (or how steep its front is, for a bore). A tsunami is kilometres long; a tidal bore front a few metres.', group='Surge', log=True),
        F('surge_dir', 'Surge direction', 90.0, -180.0, 180.0, '°', 0, tip='Direction the surge travels toward, around the vertical axis.', group='Surge'),
        F('surge_time', 'Surge arrives', 2.0, -60.0, 600.0, 's', 2, tip='When the crest (or front) of the surge passes the middle of the box, in seconds from the first frame. It travels at the speed of a long wave in water of the sea depth.', group='Surge'),
        E('surge_kind', 'Surge shape', 'wave', (('wave', 'A wave that passes (solitary wave)'), ('bore', 'A front: the water stays raised behind it (bore, flood)'),
                                                 ('tsunami', 'A tsunami: the sea draws back, then a front floods in')),
          tip='A wave is a hump of water that passes by. A bore is a step: the water stays higher behind its front, as a tidal bore running up a river or a flood surge. A tsunami is a bore the sea draws back from first, as most real tsunamis do: the water drains off the shore, baring the sea bed, then the front floods in.', group='Surge'),
        F('current_speed', 'Current', 0.0, 0.0, 5.0, 'm/s', 2, anim=True, tip='The open water flows past: a river, a tidal channel. The water in the box is carried along with it, around rocks and posts.', group='Open water'),
        F('current_dir', 'Current direction', 90.0, -180.0, 180.0, '°', 0, anim=True, tip='Direction the current flows toward, around the vertical axis.', group='Open water'),
        F('wind_speed', 'Wind speed', 0.0, 0.0, 30.0, 'm/s', 2, anim=True, tip='Wind on the liquid: it blows spray downwind, drifts foam and drags the surface.', group='Wind'),
        F('wind_dir', 'Wind direction', 90.0, -180.0, 180.0, '°', 0, anim=True, tip='Direction the wind blows toward, around the vertical axis.', group='Wind'),
        F('wind_surface', 'Surface drag', 1.0, 0.0, 5.0, '×', 2, tip='How strongly the wind drags the liquid surface (a real surface is dragged about 1/1000 as hard as spray is).', group='Wind', advanced=True),
        B('settle', 'Settle during pre-roll', False, tip='Calms the liquid during the pre-roll, so a pond or tank filled at the start is perfectly still when the shot begins. Leave off for flows that should already be running (a waterfall, a fountain).', group='Behaviour'),
        I('ppc', 'Particles per cell', 8, 1, 27, tip='Liquid particles per grid cell. 8 is standard; 27 gives smoother surfaces and finer splashes at three times the cost.', group='Quality', advanced=True, hard_lo=1, hard_hi=64),
        I('pressure_iters', 'Pressure accuracy', 10, 2, 40, tip='Solver iterations per substep. More keeps the liquid from compressing in violent impacts.', group='Quality', advanced=True),
        F('volume_correction', 'Volume preservation', 0.3, 0.0, 1.0, '', 2, tip='Pushes particles apart where they have bunched up, so the liquid keeps its volume over long shots.', group='Quality', advanced=True),
        B('narrow_band', 'Narrow band', False, tip='Keep particles only near the surface and let the grid carry the deep liquid. A pond, a pool or the sea then costs particles for its surface only, not its whole volume.', group='Quality', advanced=True),
        I('band_width', 'Band width', 4, 2, 12, 'cells', tip='Depth of the layer under the surface that keeps particles when Narrow band is on.', group='Quality', advanced=True),
        E('follow', 'Box follows', 'off', (('off', 'Nothing (stays put)'), ('liquid', 'The liquid'), ('collider', 'The first collider')), tip="The simulation box moves along with the action, in whole cells, so a small box can cover a long run: a flood running down a street, a boat crossing the sea (it follows the first collider, floating or animated). The liquid keeps its place in the world as the box moves; what it leaves behind is dropped, and open water fills in ahead. Needs open sides. Not with fire in the box.", group='Quality', advanced=True),
        F('max_particles', 'Particle limit', 16.0, 0.5, 200.0, 'M', 1, tip='Most particles alive at once, in millions. Sources stop adding liquid when it is reached.', group='Quality', advanced=True, log=True),
        B('whitewater', 'Whitewater', True, tip='Spray, foam and bubbles where the liquid moves fast and churns or breaks, as in any real splash.', group='Whitewater'),
        F('ww_amount', 'Amount', 1.0, 0.0, 10.0, '×', 2, tip='How much whitewater a churning or breaking liquid releases.', group='Whitewater', log=True),
        F('ww_min_speed', 'Minimum speed', 1.2, 0.0, 10.0, 'm/s', 2, tip='Liquid slower than this releases no whitewater. Lower it for small, gentle splashes.', group='Whitewater'),
        F('ww_turbulence', 'From churning', 1.0, 0.0, 5.0, '×', 2, tip='Whitewater from turbulent, swirling liquid: impacts, plunging jets, where air gets trapped.', group='Whitewater', advanced=True),
        F('ww_crests', 'From crests', 1.0, 0.0, 5.0, '×', 2, tip='Whitewater from breaking crests and the rims of splashes.', group='Whitewater', advanced=True),
        F('foam_life', 'Foam lifetime', 2.5, 0.1, 30.0, 's', 1, tip='How long foam lasts on the surface before it pops.', group='Whitewater', log=True),
        F('bubble_rise', 'Bubble buoyancy', 6.0, 0.0, 20.0, 'm/s²', 1, tip='How quickly bubbles rise back to the surface.', group='Whitewater', advanced=True),
        F('ww_max', 'Whitewater limit', 4.0, 0.25, 50.0, 'M', 2, tip='Most whitewater particles alive at once, in millions. The oldest are recycled.', group='Whitewater', advanced=True, log=True),
    ],
    'water': [
        C('color', 'Colour', (0.70, 0.90, 0.93), tip='The colour white light takes on after passing through the liquid for the clarity distance. Clear water is a faint cyan; tea, oil or muddy water much deeper.', group='Liquid'),
        F('clarity', 'Clarity', 3.0, 0.01, 100.0, 'm', 2, tip='Distance at which the liquid reaches its colour. Tap water several metres; wine or coffee a few millimetres.', group='Liquid', log=True),
        F('murk', 'Murkiness', 0.0, 0.0, 200.0, '/m', 1, tip='Light scattered by particles in the liquid (silt, milk, bubbles). 0 is perfectly clear.', group='Liquid'),
        C('murk_color', 'Murk colour', (0.55, 0.65, 0.60), tip='Colour of the light the murk scatters back.', group='Liquid'),
        F('ior', 'Refractive index', 1.333, 1.0, 2.0, '', 3, tip='How strongly the liquid bends light. Water 1.333, alcohol 1.36, oil and glycerine 1.47.', group='Liquid'),
        F('roughness', 'Surface roughness', 0.06, 0.0, 0.5, '', 3, tip='Micro-roughness that spreads the sun glints. 0 is a perfect mirror.', group='Surface'),
        F('ripple', 'Ripples', 0.25, 0.0, 1.0, '', 2, anim=True, tip='Fine ripples finer than the simulation, strongest where the liquid moves fast.', group='Surface'),
        F('ripple_freq', 'Ripple frequency', 25.0, 2.0, 200.0, '/m', 1, tip='Scale of the ripples: higher is finer. Scale with the shot: higher for small splashes.', group='Surface', log=True),
        F('sheets', 'Thin sheets', 0.6, 0.0, 1.0, '', 2, tip='Stretches each particle along the sheet or stream it is part of, so thin sheets, crowns and ribbons stay crisp and continuous instead of beading up.', group='Surface'),
        F('radius', 'Particle size', 1.0, 0.5, 2.0, '×', 2, tip='Size of each particle in the surface, in particle spacings. Larger joins thin sheets and drops into smoother liquid.', group='Surface', advanced=True),
        I('smoothing', 'Smoothing', 1, 0, 8, tip='Passes of smoothing on the whole liquid surface. More removes particle graininess but rounds off fine detail.', group='Surface'),
        I('calm', 'Calm surfaces', 12, 0, 32, tip='Passes of smoothing along the surface of the bulk of the liquid (not thin sheets or drops), so still and slow water is glassy instead of grainy. It never moves or shrinks the surface.', group='Surface'),
        F('surface_res', 'Surface detail', 2.0, 1.0, 3.0, '×', 1, tip='Resolution of the rendered surface relative to the simulation grid.', group='Surface', advanced=True),
        F('reflection', 'Reflection', 1.0, 0.0, 2.0, '×', 2, tip='Strength of reflections. 1 is physically correct.', group='Look'),
        F('reflect_footage', 'Footage in reflections', 0.75, 0.0, 1.0, '', 2, tip='How much of the reflected surroundings comes from your footage rather than the ambient sky colour.', group='Look'),
        F('backdrop', 'Background distance', 15.0, 0.2, 500.0, 'm', 1, tip='How far behind the liquid the scene seen through it is. Close for a table top, far for an open landscape. Sets how much the footage bends through the liquid.', group='Look', log=True),
        F('wet_darken', 'Wet ground', 0.5, 0.0, 1.0, '', 2, anim=True, tip='How much darker the ground in your footage becomes where the liquid has wetted it.', group='Ground'),
        F('wet_gloss', 'Wet shine', 0.5, 0.0, 2.0, '', 2, tip='Reflections on wet ground.', group='Ground'),
        F('shadow', 'Shadow', 1.0, 0.0, 2.0, '×', 2, tip='Shadow the liquid casts on the ground from the key light (Lighting). Clear water casts little; murky or coloured liquid more.', group='Ground'),
        F('caustics', 'Caustics', 1.0, 0.0, 4.0, '×', 2, tip='The rippling bright pattern the key light makes on the ground through the water surface. 1 is physically correct.', group='Ground'),
        F('whitecaps', 'Whitecaps', 1.0, 0.0, 4.0, '×', 2, tip="How easily the open water's crests break into foam where they fold over (Liquid › Wave height, Choppiness). 0 is none.", group='Sea'),
        F('sea_foam', 'Sea foam', 1.0, 0.0, 4.0, '×', 2, tip='How much of the foam the breaking crests leave on the open water shows: the white of fresh whitecaps and the thin, lacy foam they leave behind.', group='Sea'),
        F('sea_foam_life', 'Sea foam lifetime', 12.0, 1.0, 120.0, 's', 1, tip='How long the thin foam a breaking crest leaves lasts before it clears. A few seconds on fresh water, 10 to 30 on the sea, minutes in a storm.', group='Sea', log=True),
        F('foam_streaks', 'Foam streaks', 0.7, 0.0, 1.0, '', 2, tip='The thin foam gathers into long streaks along the wind, as on any windy sea. 0 leaves it in patches.', group='Sea'),
        F('crest_glow', 'Crest glow', 1.0, 0.0, 4.0, '×', 2, tip='Sunlight passing through the thin tops of the waves glows green-blue, strongest looking toward a low sun. 1 is about right for clear sea water.', group='Sea'),
        C('crest_glow_color', 'Crest glow colour', (0.1, 0.55, 0.45), tip='Colour of the light through the crests: the water seen through a thin layer. Greener for coastal water, bluer for the open ocean.', group='Sea', advanced=True),
        F('gusts', 'Gusts', 0.3, 0.0, 1.0, '', 2, tip='Patches of stronger wind drifting over the water (cat\u2019s paws), roughening the ripples into darker, drifting patches. 0 is an even wind.', group='Sea'),
        F('rainbow', 'Rainbow', 1.0, 0.0, 3.0, '×', 2, tip='The rainbow sunlight makes in spray and droplets, about 42 degrees away from the point opposite the sun (with a fainter second bow outside it). Only where the key light is behind the camera. 1 is physical.', group='Whitewater'),
        F('lens_drops', 'Drops on the lens', 0.0, 0.0, 3.0, '×', 2, tip='Spray, splashes and rain that reach the camera leave drops on the lens, which bend the picture through them. 0 is a dry lens.', group='Lens'),
        B('bottomless', 'Bottomless', False, tip="The bottom is out of sight (the sea, a deep lake): looking into the open water you see the water's own colour deepening, not the ground under the box. Use Murkiness and its colour for the colour of deep water.", group='Colour'),
        C('standin_color', 'Stand-in colour', (0.3, 0.3, 0.3), tip='Colour of the colliders drawn as stand-ins (Colliders: Grey stand-ins): sand for a beach or sea bed, rock for a reef.', group='Look', advanced=True),
        E('colliders_look', 'Colliders', 'holdout', (('holdout', 'In the footage (hold out)'), ('shaded', 'Grey stand-ins')),
          tip='Colliders are real objects in your footage: they hide the liquid behind them and show their own pixels when seen through or in the liquid. Grey stand-ins draws them as plain grey objects, for shots without footage.', group='Look'),
        F('foam', 'Foam', 1.0, 0.0, 4.0, '×', 2, tip='How dense and white the foam on the surface looks.', group='Whitewater'),
        F('foam_scale', 'Foam bubble size', 0.012, 0.002, 0.1, 'm', 3, tip='Size of the cells in the foam lace.', group='Whitewater', log=True),
        F('foam_lace', 'Foam lace', 0.6, 0.0, 1.0, '', 2, tip='How much thin foam breaks up into a lacy network, showing the water through its holes.', group='Whitewater'),
        F('droplets', 'Spray droplets', 1.0, 0.0, 4.0, '×', 2, tip='How visible the individual spray droplets are, drawn as small motion-blurred drops.', group='Whitewater'),
        F('droplet_size', 'Droplet size', 0.003, 0.0005, 0.02, 'm', 4, tip='Typical size of a spray droplet.', group='Whitewater', log=True),
        F('spray', 'Spray', 0.35, 0.0, 4.0, '×', 2, tip='How dense the spray in the air looks.', group='Whitewater'),
        F('bubbles', 'Bubbles', 0.6, 0.0, 4.0, '×', 2, tip='How much bubbles whiten the liquid.', group='Whitewater'),
        C('foam_color', 'Foam colour', (0.92, 0.94, 0.95), group='Whitewater', advanced=True),
        F('glow', 'Glow', 0.0, 0.0, 10.0, '', 2, anim=True, tip='Incandescence of a molten liquid (lava, molten metal, glass): light from its own heat. Above 0 the liquid is drawn as molten: opaque, glowing from its temperature through a skin or crust that rides the flow, lighting the ground around it and shimmering the air over it (Composite › Heat haze). 1 is about the brightness of basalt lava at 1300 K beside the Key light; 0 for anything cold.', group='Molten'),
        F('glow_temp', 'Glow temperature', 1300.0, 700.0, 2500.0, 'K', 0, anim=True, tip='Temperature of the fresh melt, which sets its colour and brightness as a blackbody: about 1000 K dull red, 1300 K orange (basalt lava, 1400 K as it erupts), 1800 K yellow-white (molten steel). As it cools it reddens and dims the way real lava does.', group='Molten'),
        F('crust', 'Crust', 0.6, 0.0, 1.0, '', 2, tip='How thick and dark the crust a cooling surface grows: 0 it stays bare glowing melt, 1 a black crust with only thin glowing cracks. The cracks open wherever the flow pulls the crust apart.', group='Molten'),
        F('crust_scale', 'Crust plate size', 0.08, 0.01, 1.0, 'm', 3, tip='Size of the crust plates between the glowing cracks. They ride the flow, break into smaller plates where it spreads and fold into ropes where it piles up.', group='Molten', log=True),
        F('crust_time', 'Crust forms in', 1.5, 0.1, 60.0, 's', 2, tip='How long a fresh molten surface takes to skin over and darken once it is out in the air (Liquid › Cooling sets how fast it loses its heat and stiffens). Real basalt lava darkens over tens of seconds; shorter keeps a young flow crusted in a short shot.', group='Molten', log=True),
        F('ropes', 'Ropes', 1.0, 0.0, 3.0, '×', 2, tip='How strongly the crust wrinkles into ropes where the flow squeezes it (pahoehoe). 0 keeps it flat.', group='Molten'),
        C('crust_color', 'Crust colour', (0.075, 0.073, 0.071), tip='Colour of the cooled crust in daylight. Fresh basalt rind is nearly black (about 0.03): bubbly glass, which glitters in the light rather than shining; greyer for older, weathered lava, browner for oxidised.', group='Molten', advanced=True),
        E('crust_kind', 'Crust kind', 'skin', (('skin', 'A skin (pahoehoe)'), ('plates', 'Plates (a lava lake, a channel)')),
          tip='A skin: the thin skin of a pahoehoe flow, black glass when fresh, greying as it cools. The glow shows through it where it is young and thin, in stretch marks drawn out along the flow, and stays longest in the troughs of its wrinkles; it splits into ragged glowing tears where the flow pulls it apart hard, ropes up where it is squeezed, and the foot of an advancing lobe glows in a broken seam where its front splits along the ground. Plates: the crust of a lava lake or a lava channel, broken into plates with glowing cracks between them that ride the flow.', group='Molten'),
        F('crust_roughness', 'Crust roughness', 0.45, 0.05, 1.0, '', 2, tip='Crust kind Plates: how rough the cooled plates are, low for glassy crust that shines silver in the light, high for rubbly crust. (A skin is glassy where fresh and dulls in patches as it ages.)', group='Molten', advanced=True),
        F('lava_light', 'Glow on surroundings', 1.0, 0.0, 4.0, '×', 2, tip='How strongly the glow lights the ground and the footage around the molten liquid. 1 is physical.', group='Molten'),
        F('frost', 'Frost', 1.0, 0.0, 2.0, '×', 2, tip='How white frost and rime make ice: ice frozen fast from drops and spray is white rime, and ice left below freezing grows feathers of hoarfrost. Melting ice is wet and glossy instead.', group='Ice'),
        F('ice_cloud', 'Ice cloudiness', 1.0, 0.0, 3.0, '×', 2, tip='How milky ice looks inside. Ice frozen fast traps air and is white; ice frozen slowly is clear (black ice on a pond). 0 makes all ice clear.', group='Ice'),
        C('frost_color', 'Frost colour', (0.9, 0.93, 0.97), group='Ice', advanced=True),
        F('crystal_size', 'Ice crystal size', 0.02, 0.002, 0.2, 'm', 3, tip='Size of the facets of the ice crystals and of the frost feathers on them.', group='Ice', advanced=True, log=True),
        F('step', 'Ray step', 0.35, 0.15, 1.0, 'cells', 2, group='Quality', advanced=True),
    ],
    'weather': [
        E('precip', 'Precipitation', 'none', (('none', 'None'), ('snow', 'Snow'), ('rain', 'Rain'), ('sleet', 'Sleet (ice pellets)'),
                                               ('freezing_rain', 'Freezing rain'), ('graupel', 'Graupel (soft hail)'), ('hail', 'Hail'),
                                               ('sky', 'From the temperatures aloft')),
          tip='What falls on the scene, simulated piece by piece: it melts, refreezes and evaporates on its way down as the air makes it, '
              'is blown about by the wind, settles on the ground and the colliders, and falls into the liquid. What arrives depends on the air '
              'too: snow falling through air above freezing (Liquid › Air temperature) arrives wet or as rain; sleet and freezing rain are '
              'snow melted in warmer air aloft, and need air below freezing at the ground. "From the temperatures aloft" melts snow in the '
              'layer of warm air given below.', group='Precipitation'),
        F('rate', 'Rate', 2.0, 0.05, 150.0, 'mm/h', 1, anim=True, log=True,
          tip='How much falls, as the water it holds: light snow 0.5 (about 5 mm of snow an hour), heavy snow 3, a downpour 30, a hailstorm 20 to 80.',
          group='Precipitation'),
        F('size', 'Size', 0.0, 0.0, 80.0, 'mm', 1,
          tip='Typical size of what forms: snowflakes 2 to 15 mm, graupel 2 to 5, hailstones 5 (a pea) to 45 (a golf ball) and over. 0 takes '
              'the size the rate gives (hail: 15).', group='Precipitation'),
        F('starts', 'Starts at', 0.0, -60.0, 3600.0, 's', 1, tip='When it begins, in seconds of the shot (it is falling throughout from 0).',
          group='Precipitation', advanced=True),
        F('warm_t', 'Warmest air aloft', 2.0, -20.0, 15.0, '°C', 1,
          tip='For "From the temperatures aloft": the warmest the air gets in a layer above the cold air at the ground. Above about +1 C the '
              'snow falling through it melts: partly melted, it refreezes below into ice pellets (sleet); melted through, it stays liquid '
              'below freezing and freezes onto what it hits (freezing rain).', group='Air aloft', advanced=True),
        F('warm_z', 'At height', 1500.0, 200.0, 5000.0, 'm', 0, tip='How high that warm layer is.', group='Air aloft', advanced=True),
        F('humidity', 'Humidity aloft', 95.0, 20.0, 100.0, '%', 0,
          tip='Of the air the precipitation falls through (over ice below freezing). Drier air evaporates it on the way down (and cools it: '
              'snow survives warmer air when the air is dry).', group='Air aloft', advanced=True),
        F('gust', 'Gusts', 0.3, 0.0, 1.0, '', 2, tip='How much the wind (Liquid › Wind) comes and goes.', group='Wind'),
        F('turbulence', 'Eddies', 0.4, 0.0, 5.0, 'm/s', 2, tip='Swirls in the air that toss what falls about (snow most).', group='Wind'),
        F('eddy', 'Eddy size', 1.5, 0.1, 20.0, 'm', 1, tip='Size of those swirls.', group='Wind', advanced=True, log=True),
        F('area', 'Area', 8.0, 0.5, 60.0, 'm', 1, log=True,
          tip='Width of the ground round the box that precipitation is simulated over (it covers the box at least). Make it reach as far as '
              'the camera sees falling snow or hail: beyond it, only the haze of the fall shows.', group='Area'),
        F('top', 'From height', 0.0, 0.0, 60.0, 'm', 1,
          tip='Height it starts falling from, inside the simulation (above that the air column decides what it is). 0: a metre over the box.',
          group='Area', advanced=True),
        F('limit', 'Particle limit', 2.0, 0.1, 16.0, 'M', 1, tip='Most falling pieces simulated at once (millions).', group='Area',
          advanced=True),
        F('lying', 'Already lying', 0.0, 0.0, 100.0, 'cm', 1,
          tip='Snow on the ground (and the colliders\' tops) from before the shot, where they are cold enough to keep it.', group='Ground'),
        F('buildup', 'Build-up speed-up', 1.0, 1.0, 1000.0, '×', 0, log=True,
          tip='Makes what lands build up (and melt) this many times faster, as in a time-lapse, while it still falls at its real speed: '
              'an hour of snow in a few seconds. 1 is real time (3 mm/h of water lays down about 3 cm of snow an hour).',
          group='Ground'),
        B('cover', 'Lies on the ground', True,
          tip='Snow, graupel and sleet build up where they land (on the ground and the colliders\' tops), settle and melt; freezing rain glazes '
              'them. Off: what lands is gone.', group='Ground'),
        F('cover_cell', 'Cover detail', 2.0, 0.5, 20.0, 'cm', 1, tip='Size of the cells the ground cover is kept in.', group='Ground',
          advanced=True),
        F('snow_bright', 'Snow brightness', 1.0, 0.2, 2.0, '', 2, tip='Brightness of falling and lying snow (1: fresh snow).', group='Look'),
        F('sparkle', 'Sparkle', 1.0, 0.0, 3.0, '', 2, tip='Glints of the sun on the crystals of lying snow.', group='Look'),
        F('gloss', 'Glaze and wet shine', 1.0, 0.0, 2.0, '', 2, tip='How strongly glaze ice and wet ground reflect.', group='Look'),
        F('fall_opacity', 'Falling opacity', 1.0, 0.1, 3.0, '', 2, tip='Opacity of the falling pieces as drawn.', group='Look', advanced=True),
    ],
    'atmosphere': [
        F('scale', 'Scale', 1000.0, 1.0, 10000.0, 'm per m', 0, log=True,
          tip='Metres of sky for each metre of the scene: at 1000 the box, the camera and the terrain are measured in kilometres.',
          group='Scale'),
        F('time_lapse', 'Time-lapse', 20.0, 1.0, 600.0, 's per s', 0, log=True,
          tip='Seconds of sky for each second of the shot: clouds take 10 to 30 minutes to build, a storm an hour. 1 is real time.',
          group='Scale'),
        F('surface_t', 'Air at the ground', 28.0, -40.0, 45.0, '°C', 1, tip='Temperature of the air near the ground.', group='Air column'),
        F('surface_rh', 'Humidity at the ground', 65.0, 5.0, 100.0, '%', 0, group='Air column'),
        F('mixed_layer', 'Mixed layer', 1200.0, 0.0, 4000.0, 'm', 0,
          tip='Depth of the layer the sun-warmed ground keeps stirred (its air cools 9.8 C per km as it rises): cloud bases sit near its top.',
          group='Air column'),
        F('lapse', 'Lapse rate', 6.8, 3.0, 9.8, 'K/km', 1,
          tip='How fast the air cools with height above that layer. Steeper (7 to 9) is unstable: towering clouds and storms; 5 to 6 is stable: '
              'flat clouds.', group='Air column'),
        F('free_rh', 'Humidity aloft', 55.0, 5.0, 100.0, '%', 0, tip='Humidity above the mixed layer: dry air aloft eats the clouds away.',
          group='Air column'),
        F('tropopause', 'Tropopause', 11000.0, 3000.0, 18000.0, 'm', 0,
          tip='Above it the air is stable: a storm\'s updraft stops there and spreads into the anvil.', group='Air column'),
        F('inversion_z', 'Inversion height', 0.0, 0.0, 6000.0, 'm', 0,
          tip='A layer of warmer air that caps the clouds at its height (fair-weather cumulus, a flat sheet of stratocumulus). 0: none.',
          group='Air column'),
        F('inversion_dt', 'Inversion strength', 3.0, 0.0, 15.0, 'K', 1, group='Air column'),
        F('wind', 'Wind', 4.0, 0.0, 40.0, 'm/s', 1, tip='Wind at the ground.', group='Wind'),
        F('wind_dir', 'Wind direction', 0.0, -180.0, 180.0, '°', 0, group='Wind'),
        F('shear', 'Shear', 2.0, 0.0, 10.0, 'm/s per km', 1,
          tip='How much stronger the wind is per km up: shear tilts clouds and organises storms (strong shear: long-lived supercells).',
          group='Wind'),
        F('veer', 'Veer', 0.0, -180.0, 180.0, '°', 0, tip='How far the wind turns from the ground to the tropopause (storms that rotate).',
          group='Wind', advanced=True),
        B('follow', 'Follow the storm', False,
          tip='The box moves with the clouds (at the mean wind of the lowest 6 km), so a storm stays in the middle of it instead of '
              'drifting out of the side; the ground slides under it. Not with terrain.', group='Wind'),
        F('heat_flux', 'Ground heating', 300.0, 0.0, 800.0, 'W/m²', 0,
          tip='Heat the sun-warmed ground gives the air: it sets off the thermals that build cumulus (a summer afternoon 200 to 400, a cloudy '
              'or cold day 0 to 50).', group='Thermals'),
        F('evaporation', 'Ground moisture', 150.0, 0.0, 600.0, 'W/m²', 0,
          tip='Latent heat of the water evaporating from the ground (moist fields, forest, the sea): more makes lower, wetter clouds.',
          group='Thermals'),
        F('thermal_size', 'Thermal size', 1500.0, 200.0, 10000.0, 'm', 0, tip='Size of the patches of warmer ground thermals rise from.',
          group='Thermals', log=True),
        F('patchy', 'Patchiness', 0.7, 0.0, 1.0, '', 2, tip='How unevenly the ground warms.', group='Thermals'),
        F('bubble', 'Warm bubble', 0.0, 0.0, 6.0, 'K', 1,
          tip='Starts a single storm: a bubble of air this much warmer, in the middle of the box (0: none).', group='Thermals'),
        F('bubble_r', 'Bubble size', 5000.0, 1000.0, 20000.0, 'm', 0, group='Thermals', advanced=True),
        F('rain_threshold', 'Rain needs', 1.0, 0.1, 3.0, 'g/kg', 2,
          tip='Cloud water before the droplets start to coalesce into rain: 0.5 over the sea (big droplets), 1 to 2 over land.',
          group='Precipitation'),
        F('hail', 'Hail fall speed', 1.0, 0.5, 4.0, '×', 2, tip='Graupel and hail fall this much faster: 2 to 4 for big hail.',
          group='Precipitation'),
        F('glaciation', 'Glaciation time', 900.0, 60.0, 3600.0, 's', 0,
          tip='How long cloud water below freezing takes to turn to ice (seeded clouds glaciate fast).', group='Precipitation',
          advanced=True),
        F('vorticity', 'Detail', 0.0, 0.0, 0.05, '', 3,
          tip='Restores the small eddies the grid smears (cauliflower edges). Strong values make the air unstable.', group='Detail',
          advanced=True),
    ],
    'sky': [
        F('brightness', 'Cloud brightness', 1.0, 0.2, 3.0, '', 2, group='Clouds'),
        F('density', 'Cloud density', 1.0, 0.1, 5.0, '', 2, log=True, tip='How opaque a given amount of cloud water looks.', group='Clouds'),
        F('silver', 'Silver lining', 1.0, 0.0, 1.2, '', 2,
          tip='How sharply cloud droplets throw sunlight forward: the bright rim of a cloud in front of the sun.', group='Clouds'),
        F('multiple', 'Multiple scattering', 0.8, 0.0, 1.0, '', 2,
          tip='Light scattered many times inside thick cloud: keeps their sunlit sides white.', group='Clouds'),
        F('skylight', 'Skylight', 1.0, 0.0, 3.0, '', 2, tip='Blue light from the sky on the clouds\' shaded sides.', group='Clouds'),
        F('visibility', 'Visibility', 60.0, 2.0, 300.0, 'km', 0, log=True,
          tip='How far one sees through the air before it fades into the haze of the horizon.', group='Air'),
        B('draw_sky', 'Draw the sky', True, tip='The sky and the ground behind the clouds. Off: only the clouds, over the footage.',
          group='Air'),
        C('sky_color', 'Sky', (0.25, 0.42, 0.85), tip='Colour of the sky overhead.', group='Air'),
        C('horizon_color', 'Horizon haze', (0.72, 0.8, 0.92), group='Air'),
        C('ground_color', 'Ground', (0.22, 0.25, 0.17), group='Air'),
    ],
    'lava': [
        F('viscosity', 'Viscosity', 150.0, 1.0, 100000.0, 'Pa·s', 1, tip='How thick the molten rock is. Runny pahoehoe basalt 100 to 1000, a’a lava 10,000 and up. The lava is as thick as this where it is fresh and hot, and stiffens as it cools (Setting).', group='Lava', log=True),
        F('liquid_density', 'Density', 2600.0, 1500.0, 3500.0, 'kg/m³', 0, tip='Mass per volume of the lava. Basalt melt about 2600: it sinks in water and pushes it aside.', group='Lava'),
        F('cooling', 'Cooling', 0.35, 0.0, 2.0, '/s', 3, tip='How fast the lava loses its heat to the air and the ground where it is exposed, so it crusts over and stiffens as it spreads.', group='Lava'),
        F('solidify', 'Setting', 1000.0, 1.0, 100000.0, '×', 0, tip='How much thicker the lava is once cooled than when molten. 1000 and up and it sets solid.', group='Lava', log=True),
        F('quench', 'Quenching', 4.0, 0.0, 30.0, '/s', 2, tip='How fast water chills the lava it touches: the skin goes black and glassy within a second or so, while the lava under it stays molten and glowing.', group='Lava meets water'),
        F('boiling', 'Boiling', 1.0, 0.0, 5.0, '×', 2, tip='How fiercely water touching hot lava boils off into steam (1 is about real: a metre of lava shore makes a thick white plume).', group='Lava meets water'),
        F('air_heat', 'Heats the air', 1.0, 0.0, 3.0, '×', 2, tip='Heat the lava gives the air over it: the shimmering updraft above a flow, steam off wet ground, and burnable things catching fire when it reaches them.', group='Lava meets water'),
        F('glow', 'Glow', 1.0, 0.0, 10.0, '', 2, anim=True, tip='Incandescence of the molten lava: light from its own heat, strongest where it is freshest.', group='Look'),
        F('glow_temp', 'Glow temperature', 1300.0, 700.0, 2500.0, 'K', 0, anim=True, tip='Colour of the glow of fresh lava: about 1000 K dull red, 1300 K orange, 1500 K yellow.', group='Look'),
        F('crust', 'Crust', 0.6, 0.0, 1.0, '', 2, tip='How much of the surface is dark cooled crust, 0 bare glowing melt, 1 nearly all crust with glowing cracks.', group='Look'),
        F('crust_scale', 'Crust plate size', 0.07, 0.01, 1.0, 'm', 3, tip='Size of the crust plates between the glowing cracks.', group='Look', log=True),
        F('glow_light', 'Glow lights the steam', 1.0, 0.0, 3.0, '×', 2, tip='How much the glowing lava lights the steam, smoke and water around it (1 is physical): the plume over a lava shore lit orange from underneath.', group='Look'),
    ],
    'camera': [
        E('mode', 'Camera', 'orbit', (('orbit', 'Orbit the fire'), ('free', 'Free / tracked')), group='Camera'),
        F('focal_mm', 'Focal length', 35.0, 8.0, 300.0, 'mm', 1, anim=True, tip='Match your camera lens.', group='Lens', log=True),
        F('sensor_mm', 'Sensor width', 36.0, 5.0, 70.0, 'mm', 2, tip='36 = full frame, 24.89 = Super 35, 23.5 = APS-C, 17.3 = Micro Four Thirds.', group='Lens'),
        F('yaw', 'Orbit', 0.0, -180.0, 180.0, '°', 1, anim=True, tip='Walk the camera around the fire.', group='Orbit'),
        F('pitch', 'Camera height angle', 6.0, -30.0, 89.0, '°', 1, anim=True, tip='Positive looks down on the fire.', group='Orbit'),
        F('distance', 'Distance', 6.0, 0.3, 100.0, 'm', 2, anim=True, tip='Camera distance. Changes perspective; use Scale in frame to change size.', group='Orbit', log=True),
        F('target_y', 'Look-at height', 1.2, 0.0, 20.0, 'm', 2, anim=True, group='Orbit'),
        V('position', 'Position', (0.0, 1.6, 6.0), -100.0, 100.0, 'm', anim=True, group='Free camera'),
        V('rotation', 'Rotation', (0.0, 0.0, 0.0), -180.0, 180.0, '°', anim=True, group='Free camera'),
        B('use_anchor', 'Place in frame', True, tip='Pin the fire base to a point in the frame. Turn off when using a tracked camera.', group='Placement'),
        F('anchor_x', 'Base X', 0.5, 0.0, 1.0, '', 3, anim=True, tip='Horizontal position of the fire base as a fraction of frame width.', group='Placement'),
        F('anchor_y', 'Base Y', 0.85, 0.0, 1.0, '', 3, anim=True, tip='Vertical position of the fire base as a fraction of frame height (0 = top).', group='Placement'),
        F('scale', 'Scale in frame', 1.0, 0.05, 10.0, '×', 3, anim=True, group='Placement', log=True),
        F('roll', 'Roll', 0.0, -45.0, 45.0, '°', 1, anim=True, tip='Match a tilted horizon.', group='Placement'),
        V('fire_position', 'Fire position', (0.0, 0.0, 0.0), -100.0, 100.0, 'm', anim=True, group='Fire in the scene'),
        F('fire_yaw', 'Fire rotation', 0.0, -180.0, 180.0, '°', 1, anim=True, group='Fire in the scene'),
        F('near', 'Near clip', 0.05, 0.001, 10.0, 'm', 3, group='Clipping', advanced=True),
        F('far', 'Far clip', 2000.0, 10.0, 100000.0, 'm', 0, group='Clipping', advanced=True),
    ],
    'composite': [
        E('plate_transform', 'Footage colour space', 'srgb', (('srgb', 'sRGB (most video)'), ('rec709', 'Rec.709 / BT.1886'), ('linear', 'Linear'), ('acescg', 'ACEScg'), ('ocio', 'OCIO colour space')),
          tip='How your footage is encoded. Video from phones and most cameras is sRGB or Rec.709; EXR plates are usually linear. OCIO: any colour space of the OpenColorIO config (camera log, for example), chosen below.', group='Footage'),
        Param('ocio_plate', 'Footage OCIO space', 'str', '', tip='With Footage colour space set to OCIO: the colour space of the footage in the OpenColorIO config.', group='Footage'),
        F('plate_exposure', 'Footage exposure', 0.0, -4.0, 4.0, 'EV', 2, group='Footage'),
        E('view', 'View transform', 'standard', (('standard', 'Standard (footage unchanged)'), ('agx', 'AgX filmic'), ('aces', 'ACES fit'), ('raw', 'Raw (clip)'), ('ocio', 'OCIO display and view')),
          tip='How linear light becomes display pixels. Standard leaves footage untouched and rolls off only fire highlights. OCIO uses a display and view from an OpenColorIO config (ACES, AgX, a show LUT).', group='Output look'),
        Param('ocio_config', 'OCIO config', 'file', '', tip='An OpenColorIO config (config.ocio). Empty uses the OCIO environment variable, or the built-in ACES studio config.', group='OCIO'),
        Param('ocio_working', 'Working space', 'str', '', tip='The colour space in the config that matches Blackbody\u2019s scene-linear Rec.709 working space. Empty finds it (Linear Rec.709 (sRGB), lin_rec709, and so on).', group='OCIO'),
        Param('ocio_display', 'Display', 'str', '', tip='OCIO display (sRGB, Rec.1886, P3-D65...). Empty: the config\u2019s default.', group='OCIO'),
        Param('ocio_view', 'View', 'str', '', tip='OCIO view transform for the display (ACES 1.0 SDR-video, Filmic, Un-tone-mapped...). Empty: the display\u2019s default.', group='OCIO'),
        Param('ocio_look', 'Look', 'str', '', tip='Optional OCIO look (a show grade) applied before the view.', group='OCIO', advanced=True),
        Param('exr_space', 'EXR colour space', 'str', '', tip='OCIO colour space EXR renders are written in (ACEScg, ACES2065-1...). Empty writes Blackbody\u2019s scene-linear Rec.709.', group='OCIO'),
        F('knee', 'Highlight roll-off', 0.8, 0.5, 1.0, '', 2, tip='Where highlights start to compress in the Standard view.', group='Output look'),
        F('highlight_white', 'Highlights to white', 0.25, 0.0, 0.9, '', 2, tip='Standard view: how over-bright highlights lose their colour on the way to white, as on a camera sensor, where each colour channel also catches some of the light meant for the others. Over-bright flame goes orange, then yellow, then white at the hottest cores. 0 compresses each channel on its own, so over-bright flame goes flat yellow. Footage below the roll-off is never changed.', group='Output look'),
        F('smoke_opacity', 'Smoke opacity', 1.0, 0.0, 2.0, '×', 2, anim=True, group='Fire'),
        F('saturation', 'Fire saturation', 1.0, 0.0, 2.0, '', 2, group='Fire'),
        C('tint', 'Fire colour balance', (1.0, 1.0, 1.0), group='Fire'),
        F('bloom', 'Glow', 0.15, 0.0, 2.0, '', 2, anim=True, tip='Soft glow around bright flame, as a lens would produce.', group='Glow'),
        F('bloom_radius', 'Glow radius', 1.0, 0.25, 2.0, '', 2, group='Glow'),
        F('light_cast', 'Fire light on footage', 0.6, 0.0, 4.0, '', 2, anim=True, tip='How much the fire lights up the scene around it in the footage.', group='Interaction'),
        F('surface_light', 'Fire light on surfaces', 1.0, 0.0, 4.0, '', 2, anim=True, tip='Lights the ground and every collider marked Hides fire the way the fire would light the real ones in your footage: brightest facing the flames, falling off with distance.', group='Interaction'),
        F('surface_shadows', 'Shadows from the fire', 1.0, 0.0, 1.0, '', 2, tip='Colliders in the shot and thick smoke between the fire and a surface shade the fire light on it. 0 lights every surface as if nothing were in the way.', group='Interaction'),
        F('scorch', 'Scorch', 0.8, 0.0, 1.0, '', 2, tip='With Spreading fire: how dark the burnt ground and burnt objects go in the composite.', group='Interaction'),
        F('soot', 'Soot', 0.8, 0.0, 1.0, '', 2, tip='With Combustion › Soot stains: how dark the soot left on walls, ceilings and the ground goes in the composite.', group='Interaction'),
        F('wet', 'Wet surfaces', 0.5, 0.0, 1.0, '', 2, tip='With Spreading fire: how much darker surfaces soaked by a hose or a liquid look while they dry.', group='Interaction'),
        Param('holdout_matte', 'Holdout matte', 'file', '', tip='An image sequence (name.####.png or .exr, numbered like the footage) with a matte of the objects in front of the fire: the fire, smoke and embers go behind them. Any roto or keyed matte works.', group='Holdouts from footage'),
        E('matte_channel', 'Matte channel', 'alpha', (('alpha', 'Alpha'), ('luma', 'Brightness'), ('red', 'Red'), ('green', 'Green'), ('blue', 'Blue')), tip='Which channel of the matte holds the objects.', group='Holdouts from footage'),
        B('matte_invert', 'Invert matte', False, group='Holdouts from footage'),
        Param('holdout_depth', 'Depth pass', 'file', '', tip='An EXR sequence with the depth of the footage (from a 3D scene, a depth estimator, or a lidar camera): fire behind a surface in the depth pass is hidden, fire in front of it stays, so it can burn around objects.', group='Holdouts from footage'),
        E('depth_kind', 'Depth is', 'z', (('z', 'Distance along the view axis (Z)'), ('distance', 'Distance from the camera'), ('inverse', 'Inverse depth (1/Z)')), tip='What the depth pass stores. Renderers usually write Z; depth estimators often write inverse depth.', group='Holdouts from footage'),
        F('depth_scale', 'Depth units', 1.0, 0.0001, 1000.0, 'm', 4, tip='Metres per unit of the depth pass: 1 for metres, 0.01 for centimetres, 0.3048 for feet.', group='Holdouts from footage', log=True, hard_lo=1e-6),
        F('haze', 'Heat haze', 1.0, 0.0, 5.0, '', 2, anim=True, tip='Shimmer of the hot air bending the light from the footage behind it, traced through the simulated heat so it rises with the plume and ends where the hot air ends; over a molten liquid, the air its surface heats (bending the liquid seen through it too). 1 is its physical strength: stronger with long lenses and small eddies.', group='Interaction'),
        F('haze_freq', 'Haze size', 1.0, 0.25, 4.0, '', 2, tip='Fineness of the small eddies in the hot air: higher is finer.', group='Interaction', advanced=True),
        F('haze_speed', 'Haze speed', 1.0, 0.0, 4.0, '', 2, tip='How fast the small eddies change as they drift with the air.', group='Interaction', advanced=True),
        F('f_stop', 'Aperture', 0.0, 0.0, 22.0, 'f/', 1, tip='Depth of field: the f-number the footage was shot at (f/1.4 very shallow, f/16 deep). The fire blurs by how far it is from the focus distance. 0 keeps it sharp.', group='Lens'),
        F('focus_distance', 'Focus distance', 0.0, 0.0, 100.0, 'm', 2, anim=True, tip='Where the lens was focused. 0 focuses on the fire.', group='Lens'),
        F('softness', 'Softness', 0.0, 0.0, 5.0, 'px', 2, tip='Blur the fire as softly as the footage is: a soft lens, compression or slightly missed focus. Compare an edge in the footage with the edges of the fire (pixels at 1080p).', group='Lens'),
        F('lens_k1', 'Lens distortion', 0.0, -0.3, 0.3, '', 3, tip='The distortion of the lens the footage was shot with, so the fire bends with the picture toward the frame edges: negative for barrel (wide lenses), positive for pincushion. At -0.1 the frame corners are pulled 10% toward the centre.', group='Lens'),
        F('fringing', 'Colour fringing', 0.0, 0.0, 4.0, 'px', 2, tip='Lateral chromatic aberration: red and blue images a little different in size, so bright edges get coloured fringes toward the frame corners (pixels at the corners, at 1080p).', group='Lens'),
        F('halation', 'Halation', 0.0, 0.0, 2.0, '', 2, tip='A red glow around the brightest flame, as film shows (light scattering back through the film base) and to a lesser degree many digital cameras.', group='Lens'),
        F('visibility', 'Visibility', 0.0, 0.0, 1000.0, 'm', 0, tip='How far you can see through the air in the shot: about 50 m in fog, 200–1000 m in mist, smog or dusk haze, 10 km and more on a clear day. The fire and smoke fade into the haze over their distance from the camera, as everything in the footage does. 0 is perfectly clear air.', group='Atmosphere', hard_hi=100000.0),
        B('atmos_from_footage', 'Haze colour from footage', True, tip='Take the colour of the haze from the footage: the average of its haziest bright spots, usually the sky near the horizon.', group='Atmosphere'),
        C('atmos_colour', 'Haze colour', (0.55, 0.6, 0.7), tip='Colour of the haze when it is not taken from the footage (scene-linear).', group='Atmosphere'),
        B('grain_match', 'Match footage noise', True, tip='Measure the noise in the footage (how strong it is at each brightness, how coloured and how coarse) and give the fire the same, so the fire is no cleaner than the picture around it. Grain adds film grain on top.', group='Match'),
        F('grain', 'Grain', 0.0, 0.0, 1.0, '', 2, tip='Film grain on the fire, on top of the noise matched from the footage (Match footage noise).', group='Match'),
        E('backdrop', 'Backdrop', 'stage', (('stage', 'Floor and sky'), ('colour', 'Background colour')),
          tip='Without footage: a floor out to the horizon (with the ground on) under a sky in the ambient light of Lighting '
          '(or the environment HDRI), lit by the fire, the key light and the lamps; or a flat background colour.', group='No footage'),
        E('floor', 'Floor', 'concrete', FLOOR_OPTIONS, tip='What the floor is made of, without footage.', group='No footage'),
        C('floor_tint', 'Floor tint', (1.0, 1.0, 1.0), tip='Multiplies the colours of the floor. White leaves them as they are.',
          group='No footage'),
        C('bg', 'Background', (0.0, 0.0, 0.0), group='No footage'),
        B('bg_checker', 'Checkerboard', False, group='No footage'),
    ],
    'render': [
        I('width', 'Width', 1920, 64, 7680, 'px', group='Frame'),
        I('height', 'Height', 1080, 64, 4320, 'px', group='Frame'),
        F('fps', 'Frame rate', 24.0, 1.0, 120.0, 'fps', 3, group='Frame'),
        I('start', 'First frame', 1, -10000, 100000, group='Range'),
        I('end', 'Last frame', 120, -10000, 100000, group='Range'),
        F('final_scale', 'Final resolution', 1.0, 0.25, 3.0, '×', 2, tip='Multiplies the voxel resolution for final renders.', group='Quality'),
        I('aa_samples', 'Samples per pixel', 4, 1, 64, tip='Anti-aliasing and noise-free volume sampling. 4 is clean; 16 for hero shots.', group='Quality'),
        B('auto_resolution', 'Resolution from the shot', True, tip='Final renders, with Detail upres left at 1: pick the detail upres (up to 3) that makes its cells about a pixel and a half on screen, from how big the fire appears in the shot. The voxels stay as set: a finer simulation grid changes how the fire burns.', group='Quality'),
        I('upres', 'Detail upres', 1, 1, 3, tip='Final renders: carry the fire and smoke on a grid this many times finer, moved by the simulated air plus extra small-scale turbulence. Much finer detail for a fraction of the cost of simulating at that resolution. 1 is off.', group='Quality', hard_lo=1, hard_hi=4),
        F('upres_turbulence', 'Upres turbulence', 1.0, 0.0, 4.0, '', 2, tip='Strength of the small swirls the upres adds, scaled by how turbulent the simulated air is.', group='Quality'),
        B('upres_preview', 'Upres in the viewer', False, tip='Show Detail upres while you work too, not only in final renders. Slower.', group='Quality'),
        F('final_step', 'Final ray step', 0.5, 0.2, 1.0, 'cells', 2, group='Quality', advanced=True),
        B('motion_blur', 'Motion blur', True, group='Motion blur'),
        F('shutter_angle', 'Shutter angle', 180.0, 0.0, 360.0, '°', 0, tip='180° matches most film and video. Match your footage.', group='Motion blur'),
    ],
    # Lume: path-traced light on the set drawn in CG (lume.wgsl)
    'lume': [
        E('engine', 'Lighting engine', 'classic', (('classic', 'Classic (fast)'), ('lume', 'Lume (path traced)')),
          tip='Classic lights the floor and the objects drawn in CG directly: the key light, the sky, the fire and the lamps, '
          'with soft shadows. Lume traces the light as it really travels: it bounces from surface to surface (a red wall '
          'tints the floor beside it, a room is lit by its walls), shadows are as soft as each light is big, glass, ice and '
          'jelly bend and tint what is seen through them, and the fire lights the set from each of its flames, through the '
          'smoke. Smoke, steam, cloth and grass take the same light: the fire\'s from inside, the sky\'s, the sunlit '
          'ground\'s from below, bounce after bounce (smoke looks as dark as its Smoke colour makes it). Slower: the viewer '
          'sharpens over a few seconds once you stop.', group='Lume'),
        I('bounces', 'Bounces', 4, 1, 16, tip='How many times light bounces from surface to surface. 3–4 for most shots; '
          'more for rooms and glass, which pass light on many times.', group='Lume'),
        I('samples', 'Samples (final)', 256, 4, 8192, tip='Light paths traced per pixel in final renders. More is cleaner and '
          'slower: 128 for a preview, 256–1024 for hero shots.', group='Lume', hard_hi=65536),
        I('viewer_samples', 'Samples (viewer)', 64, 4, 4096, tip='Light paths per pixel the viewer gathers once you stop '
          'moving or editing, a few at a time.', group='Lume', hard_hi=65536),
        B('denoise', 'Denoise', True, tip='Smooth away the grain of the remaining noise, keeping edges, textures and '
          'shadow edges (guided by the surfaces\' colour, facing and distance).', group='Lume'),
        F('clamp', 'Clamp bright paths', 0.0, 0.0, 1000.0, '×', 1, tip='Bright single paths (fireflies: the sun glancing '
          'off a small shiny thing, seen in a mirror) are capped at this many times the sky\'s brightness. 0 keeps every path '
          'as it is (exact, but slower to clear).', group='Lume', advanced=True),
    ],
}

MESH_TIP = ('A triangle mesh in OBJ or STL format, in metres with y up (the usual export settings); it need not be watertight. '
            'A numbered sequence (name.####.obj) deforms over time; a USD prim (file.usd#/World/Car) comes from a USD scene; '
            'a greyscale image (PNG, TIFF, EXR) is a heightfield: terrain, scaled by Size (width, height, depth in metres).')

VOLUME_TIP = ('An OpenVDB volume (.vdb) from Houdini, Blender, EmberGen or Blackbody itself, a numbered sequence of them '
              '(smoke.####.vdb, one # per digit) or a Volume prim in a USD file. Its density grid (density, smoke or the first float '
              'grid) is the smoke; a temperature grid, if there is one, is its heat. Metres; level sets become their inside.')

EMITTER_PARAMS = [
    Param('name', 'Name', 'str', 'Emitter'),
    B('enabled', 'Enabled', True),
    E('shape', 'Shape', 'cylinder', (('sphere', 'Sphere'), ('box', 'Box'), ('cylinder', 'Disc / cylinder'), ('capsule', 'Line'),
                                     ('ring', 'Ring'), ('cone', 'Cone'), ('mesh', 'Mesh'), ('volume', 'Volume (VDB)')), group='Shape'),
    Param('mesh', 'Mesh file', 'file', '', tip=MESH_TIP + ' The object burns over its surface.', group='Shape'),
    Param('volume', 'Volume file', 'file', '', tip=VOLUME_TIP, group='Shape'),
    E('volume_mode', 'Volume', 'fill', (('fill', 'Fills the box with its smoke, once'), ('hold', 'Keeps it topped up'),
                                          ('source', 'Releases where it is dense')),
      tip='Fills: at Ignite at, the volume\'s smoke (and its heat and fuel) goes into the simulation as it is, and then moves with the air: '
          'a cloud of smoke, a gas leak, haze. Keeps it topped up: while the emitter is on, the smoke is kept at least as thick as the '
          'volume every step, so an animated VDB sequence (an explosion from another program) drives it and the simulation carries '
          'it on. For both, Smoke and Fuel are the amount where the volume is densest (as much as a second of release) and Heat its '
          'temperature. Releases: the volume is a source, releasing at the emitter\'s rates wherever it is dense.', group='Shape'),
    B('volume_zup', 'Z-up file', False, tip='The VDB was saved with Z up (Blender does): turn it to y up. USD volumes are turned by the stage\'s own up axis.', group='Shape'),
    V('position', 'Position', (0.0, 0.08, 0.0), -50.0, 50.0, 'm', anim=True, decimals=3,
      tip='Keyframe it to move the emitter: a waved torch, a running stuntman, falling debris. The fire trails behind it.', group='Shape'),
    V('size', 'Size', (0.32, 0.08, 0.32), 0.001, 20.0, 'm', anim=True, decimals=3,
      tip='Sphere: radii. Box: half-sizes. Disc: radius, half-height, radius. Line: thickness (x). Ring: radius, tube radius. Cone: base radius, half-height. Mesh: scale on each axis (1 = as modelled). Volume: scale on each axis (1 = as in the file).', group='Shape'),
    V('end', 'Line end', (1.0, 0.08, 0.0), -50.0, 50.0, 'm', anim=True, decimals=3, tip='Second point of a Line emitter.', group='Shape'),
    F('yaw', 'Rotation', 0.0, -180.0, 180.0, '°', 1, anim=True, tip='Turns the emitter about the vertical axis.', group='Shape'),
    F('thickness', 'Surface depth', 0.04, 0.0, 1.0, 'm', 3, tip='Mesh: fuel comes out within this distance of the surface, so the outside of the object burns. 0 fills the whole inside.', group='Shape'),
    F('mesh_offset', 'Mesh frame offset', 0.0, -10000.0, 10000.0, 'frames', 1, tip='A deforming mesh (sequence or animated USD) plays this many frames late (negative: early).', group='Shape', advanced=True),
    F('softness', 'Edge softness', 0.0, 0.0, 0.5, 'm', 3, group='Shape'),
    F('fuel', 'Fuel', 14.0, 0.0, 200.0, '/s', 1, anim=True, tip='Fuel released per second inside the emitter.', group='Emission', log=True),
    F('temperature', 'Heat', 0.45, 0.0, 3.0, '', 2, anim=True, tip='Temperature injected with the fuel. Needs to exceed the ignition temperature to light it.', group='Emission'),
    F('smoke', 'Smoke', 0.0, 0.0, 50.0, '/s', 2, anim=True, tip='Smoke released directly, on top of smoke from combustion.', group='Emission'),
    V('velocity', 'Velocity', (0.0, 0.0, 0.0), -50.0, 50.0, 'm/s', anim=True, group='Motion'),
    F('radial', 'Burst speed', 0.0, -20.0, 80.0, 'm/s', 1, anim=True, tip='Outward speed from the emitter centre (explosions).', group='Motion'),
    F('blast', 'Blast', 0.0, 0.0, 100.0, 'kg TNT', 2, tip='An explosive charge that goes off when it ignites (Ignite at): its blast wave throws the things that fall, blows broken objects apart and scatters sand and snow, harder the nearer they are. In kilograms of TNT: a firework 0.05, a hand grenade 0.2, a car bomb 100.', group='Motion'),
    F('vel_blend', 'Velocity strength', 0.0, 0.0, 1.0, '', 2, tip='How strongly the emitter forces its velocity onto the air.', group='Motion'),
    F('inherit', 'Motion inheritance', 1.0, 0.0, 1.0, '', 2, tip='How much a moving emitter (keyframed position) drags the air and its embers along with it, so the fire trails behind.', group='Motion'),
    F('swirl', 'Swirl', 0.0, -20.0, 20.0, 'm/s', 2, anim=True, tip='Spins the air around the emitter\'s vertical axis, at this speed at its edge. Rising hot gas stretches the spin into a fire whirl or dust devil. Positive turns anticlockwise seen from above.', group='Swirl'),
    F('swirl_width', 'Swirl width', 4.0, 1.0, 20.0, '×', 1, tip='How far out the swirl reaches, in emitter radii. The spin falls off with distance beyond the emitter, as in a real vortex.', group='Swirl'),
    F('douse', 'Put out', 0.0, 0.0, 50.0, '/s', 1, anim=True, tip='Cools the gas and removes fuel and flame inside the emitter, and puts out burning surfaces: a hose, an extinguisher, rain. The heat it takes turns to steam (Combustion › Steam).', group='Extinguish'),
    F('vapour', 'Steam', 0.0, 0.0, 600.0, 'g/m³', 0, anim=True, tip='Water vapour in the gas the emitter releases. Steam straight off boiling water is about 600 g/m³; exhaust and breath 20–60. It stays clear while hot and condenses into visible steam as it cools. Give it Heat (about 0.055 is 100 °C) so it rises.', group='Steam'),
    F('color_amount', 'Colourant', 0.0, 0.0, 20.0, '/s', 2, anim=True, tip='Flame colourant (a metal salt) released with the fuel. It glows in its own colour wherever the gas is hot, and colours this emitter\'s embers and sparks. 0 is a plain flame.', group='Colour'),
    C('color', 'Colourant colour', (0.08, 1.0, 0.35), tip='Copper: green. Boron: bright green. Strontium: red. Lithium: crimson. Sodium: orange-yellow. Potassium: lilac.', group='Colour'),
    F('noise', 'Patchiness', 0.9, 0.0, 1.0, '', 2, tip='Breaks the emission into clumps so flames grow from distinct spots.', group='Patchiness'),
    F('noise_freq', 'Patch frequency', 3.5, 0.2, 40.0, '/m', 2, group='Patchiness', log=True),
    F('noise_rise', 'Patch drift', 0.6, 0.0, 10.0, 'm/s', 2, group='Patchiness'),
    F('contrast', 'Patch contrast', 1.6, 0.0, 4.0, '', 2, group='Patchiness'),
    I('seed', 'Seed', 0, 0, 9999, group='Patchiness'),
    F('start', 'Ignite at', -100.0, -100.0, 1000.0, 's', 2, tip='Seconds from the first frame. Negative means already burning (pre-roll).', group='Timing'),
    F('stop', 'Stop at', -1.0, -1.0, 1000.0, 's', 2, tip='Seconds from the first frame when emission stops. -1 burns forever.', group='Timing'),
    F('fade_in', 'Fade in', 0.3, 0.0, 10.0, 's', 2, group='Timing'),
    F('fade_out', 'Fade out', 0.8, 0.0, 10.0, 's', 2, group='Timing'),
    B('embers', 'Throws embers', True, group='Timing'),
    E('emits', 'Emits', 'fire', (('fire', 'Fire (fuel, heat, smoke)'), ('liquid', 'Liquid (water)'), ('lava', 'Lava (molten rock)')),
      tip='In a fire-and-liquid scene: whether this emitter feeds the fire, pours the liquid (water), or pours lava. Lava is its own liquid (see the Lava settings): it pushes the water aside, boils it where they meet and crusts over black, and heats the air over it.', group='Liquid'),
    E('liquid_mode', 'Pours', 'stream', (('stream', 'A stream (continuous)'), ('fill', 'A volume, once')),
      tip='A stream keeps pouring at its velocity: the flow is speed times area. A volume fills the shape with liquid once, at Ignite at (a pool, a thrown bucket of water, a wave).', group='Liquid'),
    F('flow', 'Flow', 1.0, 0.0, 1.0, '', 2, anim=True, tip='Fraction of the source that pours. Keyframe it to open and close a tap.', group='Liquid'),
    F('jitter', 'Breakup', 0.02, 0.0, 0.5, '', 3, tip='Random variation of the velocity at the source, which breaks a stream up into drops sooner.', group='Liquid'),
    C('dye', 'Dye', (0.2, 0.05, 0.05), tip='Colour of a dye this source carries (ink, blood, paint, mud): the liquid from here takes on this colour, and clouds into the rest as it mixes.', group='Dye and density'),
    F('dye_amount', 'Dye strength', 0.0, 0.0, 200.0, '/m', 1, tip='How strongly the dye colours the liquid: the colour builds up over 1/strength metres of it. Clear water 0, diluted ink 5, blood or paint 100 and up.', group='Dye and density', log=True),
    F('dye_cloud', 'Cloudiness', 0.5, 0.0, 1.0, '', 2, tip='0 a clear dye (it only tints, as wine or tea), 1 a cloudy one that scatters light (milk, mud, ink clouds).', group='Dye and density'),
    B('temp_own', 'Own temperature', False, tip='Pour the liquid at its own temperature instead of the liquid one (Liquid: Heat). Needs Heat and phase changes on.', group='Heat'),
    F('liquid_temp', 'Temperature', 20.0, -200.0, 100.0, '°C', 1, tip='Temperature of the liquid this source pours: boiling water 98 to 100, tap water 10 to 15, below freezing it pours ice (an ice cube placed as a volume, crushed ice, snow).', group='Heat'),
    F('liquid_density', 'Density', 0.0, 0.0, 20000.0, 'kg/m³', 0, tip='Density of the liquid from this source, if it differs from the rest: oil (900) floats on water, brine (1200) sinks under it, molten metal far more. 0 is the same as the liquid.', group='Dye and density'),
]

COLLIDER_PARAMS = [
    Param('name', 'Name', 'str', 'Collider'),
    B('enabled', 'Enabled', True),
    E('shape', 'Shape', 'box', (('box', 'Box'), ('sphere', 'Sphere'), ('cylinder', 'Cylinder'), ('mesh', 'Mesh')), group='Shape'),
    Param('mesh', 'Mesh file', 'file', '', tip=MESH_TIP, group='Shape'),
    V('position', 'Position', (0.0, 0.5, 0.0), -50.0, 50.0, 'm', anim=True, decimals=3,
      tip='Keyframe it to move the object (a passing car, a door swinging open): it pushes the gas out of its way.', group='Shape'),
    V('size', 'Size', (0.5, 0.5, 0.5), 0.01, 20.0, 'm', anim=True, decimals=3,
      tip='Box: half-sizes. Sphere: radius (x). Cylinder: radius, half-height. Mesh: scale on each axis (1 = as modelled).', group='Shape'),
    F('yaw', 'Rotation', 0.0, -180.0, 180.0, '°', 1, anim=True, group='Shape'),
    F('pitch', 'Tilt', 0.0, -180.0, 180.0, '°', 1, anim=True, tip='Tips it over about its own sideways axis, after its '
      'Rotation: its top leans toward its front. A ramp, a leaning wall, a wheel on its side.', group='Shape'),
    F('roll', 'Roll', 0.0, -180.0, 180.0, '°', 1, anim=True, tip='Rolls it about its own front-to-back axis, after its '
      'Tilt: its top leans to its left.', group='Shape'),
    F('mesh_offset', 'Mesh frame offset', 0.0, -10000.0, 10000.0, 'frames', 1, tip='A deforming mesh (sequence or animated USD) plays this many frames late (negative: early).', group='Shape', advanced=True),
    F('hollow', 'Hollow walls', 0.0, 0.0, 1.0, 'm', 3, anim=True, tip='Makes the collider hollow, with walls this thick: a room, a tank, a pipe. 0 is solid.', group='Walls and openings'),
    V('opening', 'Opening size', (0.0, 0.0, 0.0), 0.0, 10.0, 'm', anim=True, decimals=3, tip='Half size of a box cut out of the collider: a door, a window, a vent. Keyframe it to open a door. 0 is none. Make it a little deeper than the wall it cuts through: one that stops flush with the wall’s inside can leave a film there that sand and the like catch on.', group='Walls and openings'),
    V('opening_at', 'Opening at', (0.0, 0.0, 0.0), -20.0, 20.0, 'm', anim=True, decimals=3, tip='Centre of the opening, measured from the collider\'s centre in its own frame (it turns with the collider).', group='Walls and openings'),
    B('holdout', 'Hides fire', True, tip='The object blocks the view of fire behind it, as the real object in your footage would. Turn off for helper colliders that are not in the shot.', group='Rendering'),
    E('look', 'Look', 'auto', (('auto', 'Automatic'), ('cg', 'CG'), ('footage', 'In the footage')),
      tip='CG draws it in its material, lit by the fire, the sky, the key light and the lamps, with shadows. In the footage '
      'leaves it to the real object in your footage: it only hides the fire behind it. Automatic is CG without footage, '
      'and for things that fall or float (they are not in the footage).', group='Rendering'),
    B('own_colour', 'Own colour', False, tip='Draw it in its own colour instead of the colour of its material (CG look).', group='Rendering'),
    C('colour', 'Colour', (0.6, 0.6, 0.6), tip='Its colour with Own colour on (scene-linear).', group='Rendering'),
    B('burnable', 'Burnable', False, tip='With Spreading fire on, this object catches where hot gas touches it and fire spreads across its surface: a curtain, furniture, a wooden wall. It can move while it burns.', group='Burning'),
    F('temperature', 'Temperature', 20.0, -200.0, 1500.0, '°C', 0, tip='How hot the object is as it starts: a hot pan 180, a stove 400, a red-hot steel bar 800 (it glows from about 500), ice-cold metal -20, dry ice -78, liquid nitrogen -196. It then warms and cools as real things do, by what it is made of and how big it is: in the fire\'s heat and its radiation, by what it touches (the ground, other objects, water, wax, chocolate and metal, sand and snow) and in the air. Water (with Heat and phase changes on) boils on it, dances on its vapour past about 210, or freezes onto it; wax, chocolate and metal touching it warm or cool, the faster the better it conducts (steel quickly, wood slowly).', group='Heat'),
    B('keeps_temperature', 'Keeps its temperature', False, tip='Held at its Temperature by something the shot does not show: a hot plate on its element, a stove, a freezer\'s shelf, ground frozen hard. It gives and takes heat but never warms or cools itself. Off, it starts at its Temperature and changes with what it meets.', group='Heat'),
    B('dynamic', 'Falls', False, tip='A free rigid body: it falls, tumbles, slides, bounces and knocks into other things, and the gas and '
      'the water push it. A blast blows a light one away, and in a liquid it floats or sinks by its density. Its keys set only where '
      'it starts.', group='Physics'),
    E('material', 'Material', 'wood', MATERIAL_OPTIONS, tip='What it is made of: how heavy, slippery, bouncy and strong it is, '
      'and how it looks as a CG object.', group='Physics'),
    F('density', 'Density', 0.0, 0.0, 8000.0, 'kg/m³', 0, tip='Mass per volume. 0 uses the material\'s. Below the liquid\'s density it '
      'floats (pine 500, oak 750, ice 920, plastic 950); above it sinks (stone 2600, steel 7800). A hollow thing weighs less than its '
      'material: a cardboard box 120, a plastic crate 200.', group='Physics'),
    F('friction', 'Friction', -1.0, -1.0, 1.5, '', 2, tip='How it grips what it slides on, as a Coulomb coefficient: ice 0.05, '
      'wood 0.45, rubber 0.9. -1 uses the material\'s.', group='Physics', hard_lo=-1.0),
    F('bounce', 'Bounce', -1.0, -1.0, 0.95, '', 2, tip='How much of its speed comes back off a hard floor (restitution): clay 0, '
      'brick 0.15, wood 0.35, a rubber ball 0.8. -1 uses the material\'s.', group='Physics', hard_lo=-1.0, hard_hi=0.95),
    V('start_velocity', 'Thrown at', (0.0, 0.0, 0.0), -50.0, 50.0, 'm/s', tip='Its velocity when the simulation starts, '
      'such as a thrown ball or a launched crate. A falling object with keys also starts at the speed its keys give it.', group='Physics'),
    V('start_spin', 'Spinning at', (0.0, 0.0, 0.0), -720.0, 720.0, '°/s', tip='Its spin about each axis when the simulation starts.',
      group='Physics', advanced=True),
    F('release', 'Falls from', 0.0, -10.0, 60.0, 's', 2, tip='Seconds from the first frame when it is let go. Until then it is held '
      'where its keys put it, so it can be carried, lifted or placed and then dropped. Negative values let it go during the '
      'pre-roll, so it has already landed when the shot starts.', group='Physics'),
    E('build', 'Build', 'none', (('none', 'As it is'), ('figure', 'A person'), ('car', 'A car')),
      tip='Builds it as a thing of many parts on joints, in its box: a person (a crash-test figure, 1.8 m in a box of '
      '0.5 x 1.8 x 0.3 m), who stands braced until something hits it hard and then falls as a body does, or a car (4.4 m '
      'in a box of 4.4 x 1.5 x 1.8 m) on four sprung wheels that its motor drives and that steer. It falls (it is a body '
      'of its own) and its parts are drawn in its material.', group='Physics'),
    E('stance', 'Stance', 'stands', (('stands', 'Stands braced'), ('limp', 'Limp')),
      tip='A person: stands braced until hit hard (then goes limp), or limp from the start.', group='Physics'),
    E('drive', 'Drive', 'rear', (('rear', 'Rear wheels'), ('all', 'All four'), ('off', 'Off (rolls)')),
      tip='A car: which wheels its motor drives, or none (it rolls freely, pushed or down a slope).', group='Physics'),
    F('drive_speed', 'Speed', 0.0, -60.0, 200.0, 'km/h', 1, anim=True,
      tip='A car: the speed its motor drives it at (keyframe it: 0 brakes it, below 0 reverses).', group='Physics'),
    F('steer', 'Steer', 0.0, -35.0, 35.0, '°', 1, anim=True,
      tip='A car: how far its front wheels turn (left is positive; keyframe it).', group='Physics'),
    B('breakable', 'Breaks', False, tip='It is made of pieces glued together, which come apart where it is hit or loaded harder '
      'than its material holds: a wall knocked through, a pane shattered, a crate smashed. Without Falls it stands where it '
      'is, held by what Held by says, until it breaks.', group='Breaking'),
    E('fracture', 'Breaks into', 'voronoi', (('voronoi', 'Chunks'), ('bricks', 'Bricks'), ('shards', 'Shards (glass)'),
                                              ('splinters', 'Splinters (wood)'), ('bends', 'Bends (metal)')),
      tip='How it comes apart: irregular chunks (stone, concrete, pottery), the bricks of a wall (a box), slivers radiating '
      'from the middle of a pane (glass), pieces long along the grain (wood), or, for metal and plastic, where it creases: '
      'a bar or a post along its length, a sheet across it, so it bends and stays bent and tears once bent too far.',
      group='Breaking'),
    I('pieces', 'Pieces', 24, 2, 300, tip='How many pieces it breaks into (Bricks: as many as fit). More pieces take longer to '
      'simulate.', group='Breaking'),
    F('strength', 'Strength', 1.0, 0.01, 100.0, '×', 2, tip='How strongly the pieces hold together, times the material\N{RIGHT SINGLE QUOTATION MARK}s '
      'strength (for bricks, the mortar\N{RIGHT SINGLE QUOTATION MARK}s). Lower breaks it more easily.', group='Breaking', log=True),
    E('held', 'Held by', 'base', (('base', 'Its base'), ('edges', 'Its edges'), ('free', 'Nothing')),
      tip='What holds an object that breaks but does not fall: glued to the ground along its base (a wall), held round its '
      'edges (a pane in its frame), or nothing (it stands on its own weight).', group='Breaking'),
    I('fracture_seed', 'Pattern seed', 0, 0, 9999, tip='Another number gives another pattern of pieces.', group='Breaking',
      advanced=True),
    E('joint', 'Joined by', 'none', (('none', 'Nothing'), ('rope', 'A rope'), ('hinge', 'A hinge'), ('ball', 'A ball joint'),
                                       ('spring', 'A spring')),
      tip='What holds it to another object or to a fixed point: a rope (it hangs, swings and goes slack), a hinge (a door, a '
      'lid, a seesaw), a ball joint (it swings any way about a point) or a spring. An object with a joint falls.', group='Joint'),
    Param('joint_to', 'Joined to', 'str', '', tip='The name of the object it is joined to. Empty: a fixed point in the world '
          '(Anchor).', group='Joint'),
    V('joint_at', 'Joint on it', (0.0, 0.0, 0.0), -10.0, 10.0, 'm', tip='Where on this object the joint is, in its own frame: '
      '(0, 0, 0) is its middle. A rope or a spring joined at its middle is tied where it meets the object’s surface.',
      group='Joint'),
    V('joint_anchor', 'Anchor', (0.0, 2.0, 0.0), -1000.0, 1000.0, 'm', tip='The fixed point a rope or a spring hangs from when '
      'it is not joined to an object.', group='Joint'),
    V('joint_to_at', 'Joint on the other', (0.0, 0.0, 0.0), -10.0, 10.0, 'm', tip='Where a rope or a spring is tied on the '
      'object it is joined to, in that object’s own frame.', group='Joint'),
    V('joint_axis', 'Hinge axis', (0.0, 1.0, 0.0), -1.0, 1.0, tip='The line a hinge turns about, in the object’s own '
      'frame: (0, 1, 0) for a door, (1, 0, 0) for a seesaw or a lid.', group='Joint'),
    F('rope_length', 'Rope length', 0.0, 0.0, 100.0, 'm', 2, tip='How long the rope is (a spring: how long it is at rest). '
      '0: as long as it is from end to end at the start.', group='Joint'),
    E('rope_look', 'Rope is', 'rope', (('rope', 'Rope'), ('cable', 'Steel cable'), ('chain', 'Chain')),
      tip='What the rope is: a rope or a steel cable (weightless, wrapping round a post or a ball in its way), or a chain '
      'of steel links that weigh what they do and catch on, drape over and pile on what they meet.',
      group='Joint'),
    F('rope_thickness', 'Thickness', 0.025, 0.002, 0.2, 'm', 3, tip='How thick the rope, or the spring’s wire, is.',
      group='Joint'),
    F('spring_k', 'Spring stiffness', 500.0, 1.0, 1.0e6, 'N/m', 0, tip='How hard the spring pulls per metre it is stretched.',
      group='Joint', log=True),
    F('joint_friction', 'Joint friction', 0.2, 0.0, 10.0, '', 2, tip='How much a hinge or a ball joint resists turning: 0 '
      'swings for ever, 1 settles in a second or two.', group='Joint', advanced=True),
    F('joint_break', 'Breaks at', 0.0, 0.0, 1.0e7, 'N', 0, tip='The pull that snaps the rope or the spring, or tears the hinge '
      'out. 0: it never breaks.', group='Joint'),
    F('motor_speed', 'Motor speed', 0.0, -600.0, 600.0, 'rpm', 0, anim=True, tip='A motor turns the hinge at this many '
      'turns a minute, anticlockwise looking down its Hinge axis (negative: the other way): a wheel, a fan, a windmill, a '
      'turntable. It pushes against what it is joined to. 0: no motor.', group='Joint'),
    F('motor_torque', 'Motor strength', 50.0, 0.0, 1.0e5, 'N·m', 1, tip='The most turning force the motor has. Too little '
      'and it stalls under its load (a wheel on a slope, a fan in water).', group='Joint', log=True),
    B('floating', 'Floats', False, tip='The liquid moves it: it floats or sinks by its density, bobs, drifts with the flow, tips and '
      'turns. Falls does the same, and also lets it fall through the air. Its keyframes set only where it starts.', group='Liquid'),
]

# Lights in the set (lamps, street lights, stage spots, windows): they light the smoke and steam, and
# the smoke shadows them. A real one (in the footage) loses the light the smoke blocks on the footage's
# surfaces; one added in CG lights them too.
LIGHT_PARAMS = [
    Param('name', 'Name', 'str', 'Light'),
    B('enabled', 'Enabled', True),
    E('kind', 'Kind', 'point', (('point', 'Point (a bulb, a lamp)'), ('spot', 'Spot'), ('area', 'Area (a window, a panel)'),
                               ('lightning', 'Lightning (a bolt or an arc)')),
      tip='A point light shines every way; a spot in a cone along its aim; an area light from a flat panel facing its aim (Panel width and height), brightest straight in front, its shadows as soft as the panel is big. Lightning is a bolt from Position to where it Strikes: a branching channel that flashes, lighting the set.', group='Light'),
    V('position', 'Position', (1.5, 2.0, 1.0), -100.0, 100.0, 'm', anim=True, decimals=3, group='Light'),
    V('direction', 'Aim', (0.0, -1.0, 0.0), -1.0, 1.0, '', anim=True, decimals=3, tip='The direction a spot or area light shines (any length).', group='Light'),
    C('colour', 'Colour', (1.0, 0.86, 0.68), group='Light'),
    F('temperature', 'Colour temperature', 0.0, 0.0, 12000.0, 'K', 0, tip='0 uses the colour as it is; otherwise the colour of a light at this temperature (tungsten 2700–3200 K, daylight 5600–6500 K), times the colour.', group='Light'),
    F('intensity', 'Intensity at 1 m', 2000.0, 0.0, 200000.0, 'lx', 0, anim=True, log=True, tip='How bright the light is, as the illuminance one metre in front of it: a 100 W bulb is about 130, a car headlight 20 000, a stage spot 50 000 and up. The key light at intensity 3 is bright daylight, about 50 000.', group='Light', hard_lo=0.0),
    F('radius', 'Size', 0.1, 0.0, 5.0, 'm', 3, tip='Radius of the light: it softens the falloff close to it.', group='Light'),
    F('width', 'Panel width', 1.0, 0.01, 20.0, 'm', 2, tip='Area: how wide the panel is (a window, a softbox, an LED panel), '
      'level across its aim. Lume traces the panel itself (its soft shadows, its shape in reflections); the classic engine '
      'lights by it as by a disc as big.', group='Light'),
    F('height', 'Panel height', 1.0, 0.01, 20.0, 'm', 2, tip='Area: how tall the panel is.', group='Light'),
    F('spin', 'Turn', 0.0, -180.0, 180.0, '°', 1, tip='Turns an area light\'s panel, or a light profile\'s pattern, about '
      'its aim.', group='Light'),
    Param('profile', 'Light profile (IES)', 'file', '', tip='A photometric file (.ies) from the maker of a real light: how '
          'bright it is in each direction (a downlight\'s beam, a street light\'s spread, a wall washer\'s throw), in '
          'candela. Its aim is the light\'s Aim (straight down in the file). It takes the place of a point light\'s even '
          'spread or a spot\'s cone; area lights do not use it.', group='Light'),
    B('profile_brightness', 'Brightness from the profile', True, tip='Takes the light\'s brightness from its profile (the '
      'real light\'s candela) instead of Intensity, unless Output is set.', group='Light'),
    F('lumens', 'Output', 0.0, 0.0, 1.0e6, 'lm', 0, anim=True, log=True, tip='The light\'s whole output in lumens, as on '
      'the box (an LED bulb like a 60 W one 800, a car headlight 1 500, a stage spot 10 000, a stadium light 100 000): '
      'sets its brightness for its kind, cone, panel or profile, in place of Intensity. 0: Intensity sets it.',
      group='Light', hard_lo=0.0),
    F('cone', 'Cone angle', 30.0, 1.0, 90.0, '°', 1, tip='Spot: half-angle of the beam.', group='Light'),
    F('softness', 'Cone edge', 0.25, 0.0, 1.0, '', 2, tip='Spot: how soft the edge of the beam is.', group='Light'),
    B('shadows', 'Smoke shadows', True, tip='Smoke and steam between the light and a point shade it (beams through smoke).', group='Light'),
    V('end', 'Strikes', (0.0, 0.0, 0.0), -1000.0, 1000.0, 'm', decimals=3, tip='Lightning: where the bolt strikes. It runs '
      'from Position to here, jagged, branching on the way.', group='Lightning'),
    F('strike_at', 'Strikes at', 0.5, -10.0, 600.0, 's', 2, tip='Lightning: when it strikes, in seconds from the first frame.',
      group='Lightning'),
    I('strokes', 'Flashes', 3, 1, 12, tip='Lightning: how many times it flashes down the same channel (its return strokes), '
      'tens of milliseconds apart.', group='Lightning'),
    F('branching', 'Branches', 0.5, 0.0, 1.0, '', 2, tip='Lightning: how many branches leave its channel (they show in the '
      'first flash only).', group='Lightning'),
    F('thickness', 'Thickness', 0.03, 0.002, 2.0, 'm', 3, tip='Lightning: how wide the bright core of its channel is; its glow '
      'spreads much wider.', group='Lightning', log=True),
    I('bolt_seed', 'Bolt seed', 0, 0, 9999, tip='Lightning: another number gives another bolt.', group='Lightning'),
    B('ignites', 'Sets fire where it strikes', True, tip='Lightning: a burst of flame where it strikes (in a scene with fire), '
      'which catches whatever there will burn.', group='Lightning'),
    B('in_footage', 'In the footage', True, tip='On for a real light on the set: the footage already shows its light, so the smoke only takes away the light it shadows from the ground and objects. Off for a light added in CG: it lights the ground and the objects in the shot as well, shadowed by them and by the smoke.', group='Light'),
]

# Fabric (engine/cloth.py): curtains, flags, sheets, banners, or any mesh, as real cloth. It hangs from
# its pins, drapes over objects, blows in the fire's own air and the wind, holds the air back, and
# (if burnable) catches, chars and burns through, feeding the fire.
FABRIC_MATERIALS = (('cotton', 'Cotton (shirting, sheets)'), ('linen', 'Linen'), ('silk', 'Silk'), ('chiffon', 'Chiffon'),
                    ('wool', 'Wool (puts itself out)'), ('denim', 'Denim'), ('canvas', 'Canvas (tents, sails)'),
                    ('velvet', 'Velvet'), ('polyester', 'Polyester (shrinks and melts)'), ('nylon', 'Nylon (flags; melts)'))

FABRIC_PARAMS = [
    Param('name', 'Name', 'str', 'Fabric'),
    B('enabled', 'Enabled', True),
    E('shape', 'Shape', 'panel', (('panel', 'Panel (curtain, flag, sheet)'), ('mesh', 'Mesh')), group='Shape'),
    Param('mesh', 'Mesh file', 'file', '', tip='A triangle mesh (OBJ, STL or a USD prim) to make of cloth: a garment, a tablecloth '
          'modelled over its table, a tarp. Its own shape is its rest shape.', group='Shape'),
    F('width', 'Width', 1.2, 0.05, 20.0, 'm', 2, tip='Panel: across (along its top edge).', group='Shape'),
    F('height', 'Height', 2.0, 0.05, 20.0, 'm', 2, tip='Panel: down from its top edge (or front to back, lying).', group='Shape'),
    F('fullness', 'Gathered', 1.0, 1.0, 3.0, '×', 2, tip='Panel, hanging: how much cloth is gathered into its width, as a '
      'curtain on its rail (1 flat; curtains 1.5 to 2.5). It hangs in pleats that fall open lower down.', group='Shape'),
    E('orientation', 'Hangs', 'hanging', (('hanging', 'Upright (a curtain, a flag)'), ('lying', 'Flat (a sheet, a tablecloth)')),
      group='Shape'),
    V('position', 'Position', (0.0, 1.2, -0.6), -50.0, 50.0, 'm', anim=True, decimals=3,
      tip='The panel\'s centre (a mesh\'s origin). Keyframe it to move what holds the pins: a curtain being drawn, a flag carried.',
      group='Shape'),
    F('yaw', 'Rotation', 0.0, -180.0, 180.0, '°', 1, anim=True, group='Shape'),
    V('scale', 'Scale', (1.0, 1.0, 1.0), 0.01, 100.0, '', tip='Mesh: scale on each axis (1 = as modelled).', group='Shape', decimals=3),
    E('pins', 'Held by', 'top', (('top', 'Its top edge (a curtain)'), ('side', 'One side (a flag on a pole)'),
                                  ('top_corners', 'Its top corners (a banner)'), ('corners', 'Four corners (a canopy)'),
                                  ('edges', 'All its edges (a trampoline, a sheet laced into a frame)'), ('none', 'Nothing (it falls)')),
      group='Shape'),
    F('release', 'Let go at', -1.0, -1.0, 1000.0, 's', 2, tip='Seconds from the first frame when the pins let go and the fabric falls '
      '(a curtain rod giving way). -1 never.', group='Shape'),
    I('detail', 'Detail', 48, 8, 200, tip='Cells across the panel\'s longer side. More gives finer folds and costs more.', group='Shape'),
    E('material', 'Material', 'cotton', FABRIC_MATERIALS, tip='A real fabric: its weight, how it stretches and bends, how it moves in '
      'the air, how it looks, and how it catches and burns.', group='Fabric'),
    C('colour', 'Colour', (0.62, 0.12, 0.1), group='Fabric'),
    F('weight', 'Weight', 1.0, 0.1, 10.0, '×', 2, tip='Times the material\'s weight per area.', group='Fabric', log=True),
    F('stiffness', 'Stretch', 1.0, 0.01, 10.0, '×', 2, tip='Times the material\'s resistance to stretching and shearing.',
      group='Fabric', log=True, advanced=True),
    F('bend', 'Stiffness', 1.0, 0.01, 100.0, '×', 2, tip='Times the material\'s bending stiffness: higher drapes in bigger, rounder '
      'folds (starched, waxed); lower crumples finely.', group='Fabric', log=True),
    B('self_collide', 'Folds on itself', True, tip='The cloth cannot pass through itself as it folds and crumples.', group='Fabric',
      advanced=True),
    B('tears', 'Tears', False, tip='It rips where it is pulled too far: caught on something moving through it (a wrecking ball, a '
      'falling crate), or overloaded (sand heaped on a sling). The rip runs on from where it starts, ragged.', group='Fabric'),
    F('tear_strength', 'Tear strength', 1.0, 0.2, 5.0, '×', 2, tip='Times how far its threads stretch before they break (canvas '
      'and nylon ripstop hold out longest, chiffon and linen give soonest).', group='Fabric', log=True),
    B('burnable', 'Burnable', True, tip='It catches where the gas around it is hot enough, burns (feeding the fire), chars and burns '
      'through. Synthetics shrink away from the heat and melt first.', group='Burning'),
    F('flammability', 'Flammability', 1.0, 0.1, 5.0, '×', 2, tip='How readily it catches and how much fuel it gives off, times the '
      'material\'s (flame-retardant cloth is about 0.3).', group='Burning', log=True),
    F('wetness', 'Wet at start', 0.0, 0.0, 1.0, '', 2, tip='How soaked it starts: 1 dripping wet (a towel just out of the water). '
      'Wet cloth is heavier and limper, darker and glossier; it drips until it has drained, and in heat it steams and stays '
      'at 100 C until it has dried, so it cannot catch (a wet blanket over a fire smothers it). It soaks in a liquid too.',
      group='Water'),
]

SECTION_TITLES = {
    'domain': 'Domain', 'combustion': 'Combustion', 'motion': 'Motion', 'shading': 'Shading', 'lighting': 'Lighting',
    'embers': 'Embers', 'spread': 'Spreading fire', 'camera': 'Camera', 'composite': 'Composite', 'render': 'Render',
    'liquid': 'Liquid', 'water': 'Liquid look', 'lava': 'Lava', 'weather': 'Weather', 'atmosphere': 'Atmosphere', 'sky': 'Sky',
    'lume': 'Lume',
}

# Which sections change the simulation (and so invalidate cached frames) versus only how it looks.
SIM_SECTIONS = ('domain', 'combustion', 'motion', 'spread', 'liquid', 'lava', 'weather', 'atmosphere')
# Settings in the simulation sections that only change how it looks (they do not re-simulate).
LOOK_KEYS = {('lava', 'glow'), ('lava', 'glow_temp'), ('lava', 'crust'), ('lava', 'crust_scale'), ('lava', 'glow_light'),
             ('weather', 'snow_bright'), ('weather', 'sparkle'), ('weather', 'gloss'), ('weather', 'fall_opacity')}

# Which settings apply to which kind of simulation (domain 'kind'). Sections and keys not listed here
# apply to both.
KIND_SECTIONS = {'fire': ('combustion', 'motion', 'shading', 'embers', 'spread'), 'liquid': ('liquid', 'water', 'weather'),
                 'both': ('lava',), 'cloud': ('atmosphere', 'sky')}
FIRE_ONLY_KEYS = {
    'domain': {'sponge', 'sponge_strength', 'mg_cycles', 'maccormack', 'maccormack_vel'},
    'composite': {'smoke_opacity', 'saturation', 'tint', 'light_cast', 'haze', 'haze_freq', 'haze_speed', 'surface_shadows', 'soot',
                  'wet', 'holdout_matte', 'matte_channel', 'matte_invert', 'holdout_depth', 'depth_kind', 'depth_scale'},
    'emitter': {'fuel', 'temperature', 'smoke', 'swirl', 'swirl_width', 'douse', 'vapour', 'color_amount', 'color', 'noise',
                'noise_freq', 'noise_rise', 'contrast', 'seed', 'fade_in', 'fade_out', 'embers'},
    'collider': {'burnable'},
}
LIQUID_ONLY_KEYS = {'emitter': {'liquid_mode', 'flow', 'jitter', 'dye', 'dye_amount', 'dye_cloud', 'liquid_density', 'temp_own',
                                'liquid_temp'},
                    'collider': {'floating', 'density', 'temperature'},
                    'lighting': {'environment', 'env_rotation', 'env_strength', 'env_sun'}}
BOTH_ONLY_KEYS = {'emitter': {'emits'}, 'combustion': {'water_douse', 'soak', 'ember_heat', 'rekindle', 'steam_expansion'}}


# Liquid settings a fire-and-liquid box takes from its gas instead (the gas is the air round the liquid)
NOT_BOTH_KEYS = {'liquid': {'air_temp', 'air_humidity'}}


def applies(section, key, kind):
    """Whether a setting (or, with key None, a whole section) matters for a simulation kind."""
    if kind == 'both':
        return section not in KIND_SECTIONS['cloud'] and key not in NOT_BOTH_KEYS.get(section, ())
    if key in BOTH_ONLY_KEYS.get(section, ()):
        return False
    for k, secs in KIND_SECTIONS.items():
        if section in secs and k != kind:
            return False
    if key is None:
        return True
    if kind in ('liquid', 'cloud') and key in FIRE_ONLY_KEYS.get(section, ()):
        return False
    if kind != 'liquid' and key in LIQUID_ONLY_KEYS.get(section, ()):
        return False
    return True

_INDEX = {(s, p.key): p for s, ps in SECTIONS.items() for p in ps}
_EMITTER_INDEX = {p.key: p for p in EMITTER_PARAMS}
_COLLIDER_INDEX = {p.key: p for p in COLLIDER_PARAMS}
_LIGHT_INDEX = {p.key: p for p in LIGHT_PARAMS}
_FABRIC_INDEX = {p.key: p for p in FABRIC_PARAMS}

# Matter: sand, snow, mud, jelly and clay (engine/matter.py), simulated as particles in every kind of scene.
MATTER_MATERIALS = (('sand', 'Sand'), ('wet_sand', 'Wet sand'), ('snow', 'Snow'), ('packing_snow', 'Packing snow'), ('mud', 'Mud'),
                    ('jelly', 'Jelly'), ('gel', 'Ballistic gel'), ('clay', 'Clay'), ('wax', 'Wax'), ('chocolate', 'Chocolate'),
                    ('aluminium', 'Aluminium'), ('iron', 'Iron'), ('molten_wax', 'Molten wax'),
                    ('molten_chocolate', 'Melted chocolate'), ('molten_aluminium', 'Molten aluminium'),
                    ('molten_iron', 'Molten iron'), ('leaves', 'Dry leaves'), ('sawdust', 'Sawdust'), ('coal', 'Coal'),
                    ('ash', 'Ash'))
MATTER_PARAMS = [
    Param('name', 'Name', 'str', 'Sand'),
    B('enabled', 'Enabled', True),
    E('material', 'Made of', 'sand', MATTER_MATERIALS, tip='Sand piles up at its angle of repose and pours; wet sand holds a '
      'steeper shape and clumps; snow packs where it is squeezed and breaks up where it is pulled; packing snow makes '
      'snowballs; mud slumps and flows until it is thin enough to stop; jelly wobbles and springs back; clay squashes and '
      'stays squashed. Wax, chocolate, aluminium and iron are solid until the fire heats them past their melting point '
      '(60, 34, 660 and 1150 °C), then run, and set again where they cool; molten ones are poured hot and set as they '
      'cool. Dry leaves, sawdust and coal catch where the fire’s heat reaches them and burn down to ash: leaves in a '
      'flash of flame, sawdust smouldering, coal glowing for a long time.', group='Matter'),
    E('shape', 'Shape', 'box', (('box', 'Box'), ('sphere', 'Ball'), ('cylinder', 'Cylinder'), ('pile', 'Pile (a cone)'),
                               ('mesh', 'Mesh')),
      tip='The shape of the body of it at the start (a pour: the nozzle is a disc of its Size’s first value across). A '
      'mesh: it fills its Mesh file (a chocolate bunny, a sand sculpture, a jelly from a mould).', group='Matter'),
    Param('mesh', 'Mesh file', 'file', '', tip='A closed triangle mesh in OBJ or STL format (or a USD prim) that it fills, '
          'in metres with y up; its own origin goes at Position and Size scales it on each axis (1: as modelled).',
          group='Matter'),
    V('position', 'Position', (0.0, 0.5, 0.0), -50.0, 50.0, 'm', decimals=3, tip='Its middle, whatever its shape (a pile’s base '
      'is Size’s height below it); a pour: the nozzle; a mesh: where its own origin goes.', group='Matter'),
    V('size', 'Size', (0.25, 0.25, 0.25), 0.005, 20.0, 'm', decimals=3, tip='Half its width, height and depth (a ball: its '
      'radius; a cylinder or a pile: its radius and half its height; a mesh: its scale on each axis, 1 as modelled).',
      group='Matter'),
    F('yaw', 'Rotation', 0.0, -180.0, 180.0, '°', 1, group='Matter'),
    V('velocity', 'Thrown at', (0.0, 0.0, 0.0), -50.0, 50.0, 'm/s', tip='How fast it moves as it starts (a thrown snowball), or '
      'as it leaves the nozzle (a pour).', group='Matter'),
    F('release', 'Let go at', 0.0, -10.0, 60.0, 's', 2, tip='It is held where it is until then (a column of sand let go to '
      'collapse). Negative: during the pre-roll.', group='Matter'),
    B('pours', 'Pours', False, tip='A stream poured from a nozzle at Position (its Size’s first value is the nozzle’s '
      'radius), instead of a body of it at the start.', group='Pour'),
    F('rate', 'Flow', 0.5, 0.001, 1000.0, 'L/s', 3, tip='How much it pours, in litres a second.', group='Pour', log=True),
    F('pour_start', 'Pours from', 0.0, -10.0, 60.0, 's', 2, group='Pour'),
    F('pour_stop', 'Pours until', 5.0, 0.0, 600.0, 's', 2, group='Pour'),
    B('own_colour', 'Own colour', False, tip='Draw it in a colour of its own instead of its material’s.', group='Look'),
    C('colour', 'Colour', (0.55, 0.42, 0.25), group='Look'),
    F('stiffness', 'Stiffness', 1.0, 0.1, 10.0, '×', 2, tip='Times its material’s stiffness: stiffer jelly wobbles faster; '
      'stiffer anything takes longer to simulate.', group='Matter', log=True, advanced=True),
    I('seed', 'Seed', 0, 0, 9999, tip='Another number gives other grains in it.', group='Matter', advanced=True),
    F('temperature', 'Temperature', 20.0, -50.0, 1800.0, '°C', 0, tip='How hot it is as it starts. Wax, chocolate and '
      'metal warm in the fire and cool in the air and the water; hot metal glows (dull red from about 600 °C, orange at '
      '1000 °C). Molten ones start hotter than their melting point whatever this says.', group='Matter'),
]
_MATTER_INDEX = {p.key: p for p in MATTER_PARAMS}

# Grass and plants (engine/strands.py): patches of blades that sway in the wind and the fire's draught, part round
# what moves through them, and burn.
STRAND_KINDS = (('lawn', 'Lawn'), ('meadow', 'Long grass'), ('wheat', 'Wheat'), ('reeds', 'Reeds'))
STRAND_PARAMS = [
    Param('name', 'Name', 'str', 'Grass'),
    B('enabled', 'Enabled', True),
    E('kind', 'Kind', 'meadow', STRAND_KINDS, tip='A lawn is short and thick; long grass sways and bends over in the wind; '
      'wheat stands stiff with its ears; reeds are tall and sparse.', group='Grass'),
    E('shape', 'Shape', 'box', (('box', 'Rectangle'), ('disc', 'Disc')), group='Grass'),
    V('position', 'Position', (0.0, 0.0, 0.0), -50.0, 50.0, 'm', decimals=3, tip='The middle of the patch, at the foot of '
      'the blades.', group='Grass'),
    V('size', 'Size', (1.0, 0.45, 1.0), 0.01, 50.0, 'm', decimals=3, tip='Half its width and depth (a disc: its radius, the '
      'first value), and how tall the blades grow (the second).', group='Grass'),
    F('yaw', 'Rotation', 0.0, -180.0, 180.0, '°', 1, group='Grass'),
    F('thickness', 'Thickness', 1.0, 0.05, 4.0, '×', 2, tip='How thickly it grows: times the usual '
      'number of blades of its kind (a lawn about 4000 a square metre, long grass 900, wheat 350, reeds 120).', group='Grass',
      log=True),
    F('dryness', 'Dryness', 0.3, 0.0, 1.0, '', 2, tip='0: fresh and green, hard to light; 1: dry as straw, it catches at '
      'a spark and burns fast.', group='Grass'),
    B('burns', 'Burns', True, tip='Fire sets it alight: it chars, burns down to stubble and feeds the flames, so fire runs '
      'through it, faster downwind.', group='Grass'),
    E('grows_on', 'Grows on', 'ground', (('ground', 'The ground'), ('everything', 'The ground and objects')),
      tip='The ground: at Position’s height, but not under objects. The ground and objects: on whatever is below, '
      'a hillside, a mound, the top of a wall (not on things that move).', group='Grass'),
    F('stiffness', 'Stiffness', 1.0, 0.2, 5.0, '×', 2, tip='Times its kind’s stiffness: '
      'stiffer blades bend less in the wind and spring back faster.', group='Grass', log=True, advanced=True),
    F('blade_width', 'Blade width', 1.0, 0.25, 4.0, '×', 2, tip='Times its kind’s blade '
      'width.', group='Look', log=True, advanced=True),
    B('own_colour', 'Own colour', False, tip='Draw it in a colour of its own instead of its kind’s (dried toward '
      'straw by its Dryness).', group='Look'),
    C('colour', 'Colour', (0.1, 0.24, 0.04), group='Look'),
    I('seed', 'Seed', 0, 0, 9999, tip='Another number gives other blades.', group='Grass', advanced=True),
]
_STRAND_INDEX = {p.key: p for p in STRAND_PARAMS}

# Shots: a gun's muzzle and what it is aimed at, the cartridge, and when and how often it fires (engine/ballistics.py)
SHOT_ROUNDS = (('pellet', 'Air rifle pellet (.177)'), ('22lr', '.22 LR'), ('9mm', '9 mm pistol (FMJ)'),
               ('9mm_hp', '9 mm pistol (hollow point)'), ('45acp', '.45 ACP (FMJ)'), ('556', '5.56 mm rifle (M193)'),
               ('762x39', '7.62x39 mm rifle'), ('308', '.308 / 7.62x51 mm rifle'), ('slug', '12 gauge slug'),
               ('buck', '12 gauge 00 buckshot (9 pellets)'), ('50bmg', '.50 BMG rifle'))
SHOT_PARAMS = [
    Param('name', 'Name', 'str', 'Shot'),
    B('enabled', 'Enabled', True),
    E('round', 'Cartridge', '9mm', SHOT_ROUNDS, tip='What it fires: its bullet’s weight, size, speed and build. A '
      'hollow point opens up in anything wet and stops sooner; a rifle bullet turns sideways and breaks up in gel and '
      'water; shot spreads in a cone.', group='Gun'),
    V('position', 'Muzzle', (0.0, 1.0, 1.5), -50.0, 50.0, 'm', anim=True, decimals=3, tip='Where the bullets leave the '
      'gun.', group='Gun'),
    V('aim', 'Aimed at', (0.0, 1.0, 0.0), -50.0, 50.0, 'm', anim=True, decimals=3, tip='The point it is aimed at: the '
      'bullets fly toward it (dropping a little over long distances) and on past it.', group='Gun'),
    F('start', 'Fires at', 0.5, 0.0, 60.0, 's', 2, tip='When the first round goes off, from the start of the shot.',
      group='Firing'),
    I('count', 'Rounds', 1, 1, 100, tip='How many rounds it fires.', group='Firing', hard_lo=1, hard_hi=1000),
    F('rate', 'Rate of fire', 600.0, 30.0, 1200.0, 'rounds/min', 0, tip='How fast it fires more than one: a pistol '
      'fired as fast as it can be about 300 a minute, an assault rifle 600-900, a machine gun 1000 or more.',
      group='Firing', log=True),
    F('scatter', 'Scatter', 0.05, 0.0, 5.0, '°', 2, tip='How far each round strays from the aim (one standard '
      'deviation): a rifle 0.02, a pistol at arm’s length 0.1, firing from the hip 1 or more.', group='Firing'),
    F('speed', 'Muzzle speed', 0.0, 0.0, 1500.0, 'm/s', 0, tip='The bullet’s speed leaving the muzzle; 0 takes '
      'the cartridge’s (a short barrel is slower, a long one faster).', group='Gun', advanced=True),
    B('tracer', 'Tracer', False, tip='Tracer rounds: each burns bright red along its path.', group='Look'),
    B('flash', 'Muzzle flash', True, tip='A flash of light at the muzzle as each round goes off.', group='Look'),
    I('seed', 'Seed', 0, 0, 9999, tip='Another number scatters the rounds, and what flies off where they hit, '
      'differently.', group='Firing', advanced=True),
]
_SHOT_INDEX = {p.key: p for p in SHOT_PARAMS}


def param(section, key):
    if section == 'emitter':
        return _EMITTER_INDEX[key]
    if section == 'light':
        return _LIGHT_INDEX[key]
    if section == 'fabric':
        return _FABRIC_INDEX[key]
    if section == 'matter':
        return _MATTER_INDEX[key]
    if section == 'strands':
        return _STRAND_INDEX[key]
    if section == 'shot':
        return _SHOT_INDEX[key]
    if section == 'collider':
        return _COLLIDER_INDEX[key]
    return _INDEX[(section, key)]


def defaults(section):
    return {p.key: p.default for p in SECTIONS[section]}


def emitter_defaults():
    return {p.key: p.default for p in EMITTER_PARAMS}


def fabric_defaults():
    return {p.key: p.default for p in FABRIC_PARAMS}


def matter_defaults():
    return {p.key: p.default for p in MATTER_PARAMS}


def strand_defaults():
    return {p.key: p.default for p in STRAND_PARAMS}


def shot_defaults():
    return {p.key: p.default for p in SHOT_PARAMS}


def light_defaults():
    return {p.key: p.default for p in LIGHT_PARAMS}


def collider_defaults():
    return {p.key: p.default for p in COLLIDER_PARAMS}


def coerce(p: Param, value):
    """Convert a value (possibly from JSON or the command line) to the parameter's type and hard range."""
    if p.kind == 'float':
        v = float(value)
        if p.hard_lo is not None:
            v = max(v, p.hard_lo)
        if p.hard_hi is not None:
            v = min(v, p.hard_hi)
        return v
    if p.kind == 'int':
        v = int(round(float(value)))
        if p.hard_lo is not None:
            v = max(v, int(p.hard_lo))
        if p.hard_hi is not None:
            v = min(v, int(p.hard_hi))
        return v
    if p.kind == 'bool':
        if isinstance(value, str):
            return value.strip().lower() in ('1', 'true', 'yes', 'on')
        return bool(value)
    if p.kind == 'enum':
        vals = [o[0] for o in p.options]
        return value if value in vals else p.default
    if p.kind in ('color', 'vec3'):
        if isinstance(value, str):
            value = [float(x) for x in value.replace(',', ' ').split()]
        v = tuple(float(x) for x in value)
        return (v + tuple(p.default))[:3] if len(v) < 3 else v[:3]
    if p.kind in ('str', 'file'):
        return '' if value is None else str(value)
    return value
