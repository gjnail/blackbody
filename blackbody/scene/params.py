"""Parameter registry: every user-facing setting with its range, unit and tooltip.

The Properties panel, the command line (--set section.key=value), presets and project files all
read from this one table, so a parameter added here shows up everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field


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
        E('kind', 'Simulation', 'fire', (('fire', 'Fire and smoke'), ('liquid', 'Liquid'), ('both', 'Fire and liquid')),
          tip='What this shot simulates: burning gas, a liquid such as water, or both in the same box (a hose putting out a fire, burning fuel on water).', group='Simulation'),
        F('size_x', 'Width', 2.0, 0.1, 20.0, 'm', tip='Simulation box width. Fire outside the box is lost, so give it room.', group='Box', log=True, hard_lo=0.02),
        F('size_y', 'Height', 3.2, 0.1, 40.0, 'm', tip='Simulation box height, measured up from the fire base.', group='Box', log=True, hard_lo=0.02),
        F('size_z', 'Depth', 2.0, 0.1, 20.0, 'm', tip='Simulation box depth (toward the camera).', group='Box', log=True, hard_lo=0.02),
        I('resolution', 'Voxels (longest side)', 128, 32, 384, tip='Cells along the longest side of the box for final renders. Detail and memory grow with the cube of this.', group='Resolution', hard_lo=16, hard_hi=768),
        F('preview_scale', 'Interactive resolution', 0.75, 0.25, 1.0, '×', tip='Fraction of the final resolution used while you work. Final renders always use the full resolution.', group='Resolution'),
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
        F('expansion', 'Gas expansion', 0.5, 0.0, 6.0, '', 2, anim=True, tip='Volume released by burning fuel. High values make fireballs billow outward.', group='Expansion'),
        F('thermal_expansion', 'Heat expansion', 0.0, 0.0, 1.0, '', 2, tip='Gas swells as it heats and shrinks as it cools, as real air does (1 is physical). Fireballs billow harder and a hot closed room pushes smoke out of every gap; 0 leaves the swelling to Gas expansion.', group='Expansion'),
        F('rich', 'Oxygen limit', 4.0, 0.2, 20.0, '', 2, tip='Fuel level at which lack of air halves the burn rate. Lower it for fuel-rich fireballs that burn from the outside in.', group='Fuel', log=True),
        F('expansion_cap', 'Expansion limit', 40.0, 1.0, 200.0, '/s', 1, tip='Upper limit on how fast burning gas can expand in one place. Keeps violent bursts stable.', group='Expansion', advanced=True),
        E('air', 'Air supply', 'open', (('open', 'Open air'), ('tracked', 'Tracked (rooms, backdraft)')),
          tip='Open air: the fire always finds oxygen; the oxygen limit only models mixing. Tracked: burning uses up the air, so a fire in a closed space starves, unburnt fuel builds up, and fresh air let in later can make it flash over (backdraft).', group='Air'),
        F('air_use', 'Air use', 1.0, 0.0, 5.0, '', 2, tip='With tracked air: how much of the air in a place each unit of burned fuel uses up. Higher starves a closed-room fire sooner.', group='Air'),
        F('air_mixing', 'Air mixing', 0.03, 0.0, 0.3, 'm²/s', 3, tip='With tracked air: how strongly used-up air mixes with the air around it (turbulent mixing too fine for the grid). It lets starved, fuel-rich gas that meets fresh air ignite. Real fires: 0.01–0.1.', group='Air', advanced=True),
        F('steam_yield', 'Steam from dousing', 150.0, 0.0, 1000.0, 'g/m³', 0, tip='Water vapour made when an emitter with Put out takes heat from the gas, per unit of temperature removed. Water on a fire flashes to steam.', group='Steam'),
        F('water_yield', 'Water from burning', 0.0, 0.0, 200.0, 'g/m³', 0, tip='Water vapour made per unit of burned fuel. Real flames make plenty; it shows as condensing steam in cold, humid air.', group='Steam'),
        F('vapour_dissipation', 'Vapour mixing', 1.5, 0.0, 10.0, '/s', 2, anim=True, tip='How fast water vapour mixes away into the surrounding air. Steam evaporates at its edges as it thins.', group='Steam'),
        F('water_douse', 'Water on fire', 40.0, 0.0, 200.0, '/s', 1, tip='With liquid in the same box: how fast a place full of water puts the fire out (and turns its heat to steam).', group='Steam'),
        F('latent_heat', 'Latent heat', 1.0, 0.0, 2.0, '', 2, tip='Heat given off as steam condenses (1 is physical). It warms the cloud so steam billows upward like a cumulus; 0 lets it drift.', group='Steam'),
    ],
    'motion': [
        F('buoyancy', 'Buoyancy', 5.0, 0.0, 30.0, 'm/s²', 1, anim=True, tip='Upward acceleration of hot gas per unit of temperature.', group='Buoyancy'),
        F('soot_weight', 'Smoke weight', 0.02, 0.0, 1.0, 'm/s²', 3, tip='Downward pull of soot. Keep small: cooled smoke is close to neutrally buoyant.', group='Buoyancy', advanced=True),
        F('damping', 'Air drag', 0.05, 0.0, 2.0, '/s', 2, group='Buoyancy', advanced=True),
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
        F('exposure', 'Fire exposure', 0.0, -6.0, 6.0, 'EV', 2, anim=True, tip='Brightness of the fire in stops. At 0, thick flame at the flame temperature renders at full white.', group='Flame brightness'),
        F('intensity', 'Emission', 1.0, 0.0, 10.0, '×', 2, group='Flame brightness', advanced=True),
        F('flame_density', 'Flame density', 1.0, 0.0, 4.0, '×', 2, group='Flame shape'),
        F('flame_sharpness', 'Flame sharpness', 1.6, 0.5, 4.0, '', 2, tip='Contrast of flame edges. Higher gives crisper, more defined tongues.', group='Flame shape'),
        F('flame_threshold', 'Flame threshold', 0.02, 0.0, 0.5, '', 3, tip='Trims faint wisps of flame.', group='Flame shape'),
        F('flame_absorption', 'Flame thickness', 4.0, 0.2, 200.0, '/m', 2, tip='How quickly flame becomes optically thick. Thick flame glows at full blackbody brightness; thin flame is translucent and dimmer. Small flames need higher values.', group='Flame shape', log=True),
        F('flame_occlusion', 'Flame occlusion', 0.35, 0.0, 1.0, '', 2, tip='How much flames hide what is behind them in the alpha channel.', group='Flame shape'),
        F('soot_glow', 'Glowing soot', 0.25, 0.0, 3.0, '', 2, tip='Incandescence of hot smoke just above the flames.', group='Flame shape'),
        F('blue', 'Base glow', 0.0, 0.0, 5.0, '', 2, tip='Glow where fuel is burning: the blue chemiluminescence of gas and alcohol flames. Change its colour for other chemistry.', group='Flame colour'),
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
    ],
    'lighting': [
        C('ambient', 'Ambient light', (0.10, 0.11, 0.13), anim=True, tip='Light reaching the smoke from the surroundings (sky, room). Match it to your footage.', group='Ambient'),
        F('ambient_intensity', 'Ambient intensity', 1.0, 0.0, 10.0, '×', 2, anim=True, group='Ambient'),
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
        F('water_level', 'Water level', 0.0, 0.0, 20.0, 'm', 3, tip='Water that carries on past the open sides of the box up to this height (a pond, a flooded street, the sea). It fills the box to this level at the start, keeps it topped up at the sides, lets waves run out, and is drawn out to the horizon. 0 is off.', group='Open water'),
        F('level_absorb', 'Edge absorption', 8.0, 0.0, 32.0, 'cells', 0, tip='Width of the layer at the open sides where waves are calmed so they do not bounce back off the box edge.', group='Open water', advanced=True),
        F('wind_speed', 'Wind speed', 0.0, 0.0, 30.0, 'm/s', 2, anim=True, tip='Wind on the liquid: it blows spray downwind, drifts foam and drags the surface.', group='Wind'),
        F('wind_dir', 'Wind direction', 90.0, -180.0, 180.0, '°', 0, anim=True, tip='Direction the wind blows toward, around the vertical axis.', group='Wind'),
        F('wind_surface', 'Surface drag', 1.0, 0.0, 5.0, '×', 2, tip='How strongly the wind drags the liquid surface (a real surface is dragged about 1/1000 as hard as spray is).', group='Wind', advanced=True),
        B('settle', 'Settle during pre-roll', False, tip='Calms the liquid during the pre-roll, so a pond or tank filled at the start is perfectly still when the shot begins. Leave off for flows that should already be running (a waterfall, a fountain).', group='Behaviour'),
        I('ppc', 'Particles per cell', 8, 1, 27, tip='Liquid particles per grid cell. 8 is standard; 27 gives smoother surfaces and finer splashes at three times the cost.', group='Quality', advanced=True, hard_lo=1, hard_hi=64),
        I('pressure_iters', 'Pressure accuracy', 10, 2, 40, tip='Solver iterations per substep. More keeps the liquid from compressing in violent impacts.', group='Quality', advanced=True),
        F('volume_correction', 'Volume preservation', 0.3, 0.0, 1.0, '', 2, tip='Pushes particles apart where they have bunched up, so the liquid keeps its volume over long shots.', group='Quality', advanced=True),
        B('narrow_band', 'Narrow band', False, tip='Keep particles only near the surface and let the grid carry the deep liquid. A pond, a pool or the sea then costs particles for its surface only, not its whole volume.', group='Quality', advanced=True),
        I('band_width', 'Band width', 4, 2, 12, 'cells', tip='Depth of the layer under the surface that keeps particles when Narrow band is on.', group='Quality', advanced=True),
        B('disk_cache', 'Cache to disk', False, tip='Also keep every simulated frame on disk (next to the project, or in the app data folder for an unsaved scene), so final renders and reopened projects re-render without re-simulating.', group='Quality'),
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
        F('step', 'Ray step', 0.35, 0.15, 1.0, 'cells', 2, group='Quality', advanced=True),
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
        E('plate_transform', 'Footage colour space', 'srgb', (('srgb', 'sRGB (most video)'), ('rec709', 'Rec.709 / BT.1886'), ('linear', 'Linear'), ('acescg', 'ACEScg')),
          tip='How your footage is encoded. Video from phones and most cameras is sRGB or Rec.709; EXR plates are usually linear.', group='Footage'),
        F('plate_exposure', 'Footage exposure', 0.0, -4.0, 4.0, 'EV', 2, group='Footage'),
        E('view', 'View transform', 'standard', (('standard', 'Standard (footage unchanged)'), ('agx', 'AgX filmic'), ('aces', 'ACES fit'), ('raw', 'Raw (clip)')),
          tip='How linear light becomes display pixels. Standard leaves footage untouched and rolls off only fire highlights.', group='Output look'),
        F('knee', 'Highlight roll-off', 0.8, 0.5, 1.0, '', 2, tip='Where highlights start to compress in the Standard view.', group='Output look'),
        F('smoke_opacity', 'Smoke opacity', 1.0, 0.0, 2.0, '×', 2, anim=True, group='Fire'),
        F('saturation', 'Fire saturation', 1.0, 0.0, 2.0, '', 2, group='Fire'),
        C('tint', 'Fire colour balance', (1.0, 1.0, 1.0), group='Fire'),
        F('bloom', 'Glow', 0.15, 0.0, 2.0, '', 2, anim=True, tip='Soft glow around bright flame, as a lens would produce.', group='Glow'),
        F('bloom_radius', 'Glow radius', 1.0, 0.25, 2.0, '', 2, group='Glow'),
        F('light_cast', 'Fire light on footage', 0.6, 0.0, 4.0, '', 2, anim=True, tip='How much the fire lights up the scene around it in the footage.', group='Interaction'),
        F('surface_light', 'Fire light on surfaces', 1.0, 0.0, 4.0, '', 2, anim=True, tip='Lights the ground and every collider marked Hides fire the way the fire would light the real ones in your footage: brightest facing the flames, falling off with distance.', group='Interaction'),
        F('scorch', 'Scorch', 0.8, 0.0, 1.0, '', 2, tip='With Spreading fire: how dark the burnt ground and burnt objects go in the composite.', group='Interaction'),
        F('haze', 'Heat haze', 1.0, 0.0, 5.0, '', 2, anim=True, tip='Shimmer of hot air distorting the footage behind and above the fire.', group='Interaction'),
        F('haze_freq', 'Haze size', 1.0, 0.25, 4.0, '', 2, group='Interaction', advanced=True),
        F('haze_speed', 'Haze speed', 1.0, 0.0, 4.0, '', 2, group='Interaction', advanced=True),
        F('grain', 'Grain', 0.0, 0.0, 1.0, '', 2, tip='Film grain on the fire so it matches noisy footage.', group='Match'),
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
        I('upres', 'Detail upres', 1, 1, 3, tip='Final renders: carry the fire and smoke on a grid this many times finer, moved by the simulated air plus extra small-scale turbulence. Much finer detail for a fraction of the cost of simulating at that resolution. 1 is off.', group='Quality', hard_lo=1, hard_hi=4),
        F('upres_turbulence', 'Upres turbulence', 1.0, 0.0, 4.0, '', 2, tip='Strength of the small swirls the upres adds, scaled by how turbulent the simulated air is.', group='Quality'),
        F('final_step', 'Final ray step', 0.5, 0.2, 1.0, 'cells', 2, group='Quality', advanced=True),
        B('motion_blur', 'Motion blur', True, group='Motion blur'),
        F('shutter_angle', 'Shutter angle', 180.0, 0.0, 360.0, '°', 0, tip='180° matches most film and video. Match your footage.', group='Motion blur'),
    ],
}

MESH_TIP = 'A triangle mesh in OBJ format, in metres with y up (the usual export settings). It need not be watertight.'

EMITTER_PARAMS = [
    Param('name', 'Name', 'str', 'Emitter'),
    B('enabled', 'Enabled', True),
    E('shape', 'Shape', 'cylinder', (('sphere', 'Sphere'), ('box', 'Box'), ('cylinder', 'Disc / cylinder'), ('capsule', 'Line'),
                                     ('ring', 'Ring'), ('cone', 'Cone'), ('mesh', 'Mesh')), group='Shape'),
    Param('mesh', 'Mesh file', 'file', '', tip=MESH_TIP + ' The object burns over its surface.', group='Shape'),
    V('position', 'Position', (0.0, 0.08, 0.0), -50.0, 50.0, 'm', anim=True, decimals=3,
      tip='Keyframe it to move the emitter: a waved torch, a running stuntman, falling debris. The fire trails behind it.', group='Shape'),
    V('size', 'Size', (0.32, 0.08, 0.32), 0.001, 20.0, 'm', anim=True, decimals=3,
      tip='Sphere: radii. Box: half-sizes. Disc: radius, half-height, radius. Line: thickness (x). Ring: radius, tube radius. Cone: base radius, half-height. Mesh: scale on each axis (1 = as modelled).', group='Shape'),
    V('end', 'Line end', (1.0, 0.08, 0.0), -50.0, 50.0, 'm', anim=True, decimals=3, tip='Second point of a Line emitter.', group='Shape'),
    F('yaw', 'Rotation', 0.0, -180.0, 180.0, '°', 1, anim=True, tip='Turns the emitter about the vertical axis.', group='Shape'),
    F('thickness', 'Surface depth', 0.04, 0.0, 1.0, 'm', 3, tip='Mesh: fuel comes out within this distance of the surface, so the outside of the object burns. 0 fills the whole inside.', group='Shape'),
    F('softness', 'Edge softness', 0.0, 0.0, 0.5, 'm', 3, group='Shape'),
    F('fuel', 'Fuel', 14.0, 0.0, 200.0, '/s', 1, anim=True, tip='Fuel released per second inside the emitter.', group='Emission', log=True),
    F('temperature', 'Heat', 0.45, 0.0, 3.0, '', 2, anim=True, tip='Temperature injected with the fuel. Needs to exceed the ignition temperature to light it.', group='Emission'),
    F('smoke', 'Smoke', 0.0, 0.0, 50.0, '/s', 2, anim=True, tip='Smoke released directly, on top of smoke from combustion.', group='Emission'),
    V('velocity', 'Velocity', (0.0, 0.0, 0.0), -50.0, 50.0, 'm/s', anim=True, group='Motion'),
    F('radial', 'Burst speed', 0.0, -20.0, 80.0, 'm/s', 1, anim=True, tip='Outward speed from the emitter centre (explosions).', group='Motion'),
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
    E('emits', 'Emits', 'fire', (('fire', 'Fire (fuel, heat, smoke)'), ('liquid', 'Liquid')),
      tip='In a fire-and-liquid scene: whether this emitter feeds the fire or pours liquid.', group='Liquid'),
    E('liquid_mode', 'Pours', 'stream', (('stream', 'A stream (continuous)'), ('fill', 'A volume, once')),
      tip='A stream keeps pouring at its velocity: the flow is speed times area. A volume fills the shape with liquid once, at Ignite at (a pool, a thrown bucket of water, a wave).', group='Liquid'),
    F('flow', 'Flow', 1.0, 0.0, 1.0, '', 2, anim=True, tip='Fraction of the source that pours. Keyframe it to open and close a tap.', group='Liquid'),
    F('jitter', 'Breakup', 0.02, 0.0, 0.5, '', 3, tip='Random variation of the velocity at the source, which breaks a stream up into drops sooner.', group='Liquid'),
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
    F('hollow', 'Hollow walls', 0.0, 0.0, 1.0, 'm', 3, anim=True, tip='Makes the collider hollow, with walls this thick: a room, a tank, a pipe. 0 is solid.', group='Walls and openings'),
    V('opening', 'Opening size', (0.0, 0.0, 0.0), 0.0, 10.0, 'm', anim=True, decimals=3, tip='Half size of a box cut out of the collider: a door, a window, a vent. Keyframe it to open a door. 0 is none.', group='Walls and openings'),
    V('opening_at', 'Opening at', (0.0, 0.0, 0.0), -20.0, 20.0, 'm', anim=True, decimals=3, tip='Centre of the opening, measured from the collider\'s centre in its own frame (it turns with the collider).', group='Walls and openings'),
    B('holdout', 'Hides fire', True, tip='The object blocks the view of fire behind it, as the real object in your footage would. Turn off for helper colliders that are not in the shot.', group='Rendering'),
    B('burnable', 'Burnable', False, tip='With Spreading fire on, this object catches where hot gas touches it and fire spreads across its surface: a curtain, furniture, a wooden wall. It can move while it burns.', group='Burning'),
    B('floating', 'Floats', False, tip='The liquid moves it: it floats or sinks, bobs, drifts with the flow and turns. It stays upright (it turns only about the vertical). Its keyframes set only where it starts.', group='Liquid'),
    F('density', 'Density', 600.0, 20.0, 8000.0, 'kg/m³', 0, tip='Mass per volume of a floating object. Below the liquid\'s density it floats: pine 500, oak 750, ice 920, plastic 950; above it sinks: stone 2600, steel 7800.', group='Liquid', log=True),
]

SECTION_TITLES = {
    'domain': 'Domain', 'combustion': 'Combustion', 'motion': 'Motion', 'shading': 'Shading', 'lighting': 'Lighting',
    'embers': 'Embers', 'spread': 'Spreading fire', 'camera': 'Camera', 'composite': 'Composite', 'render': 'Render',
    'liquid': 'Liquid', 'water': 'Liquid look',
}

# Which sections change the simulation (and so invalidate cached frames) versus only how it looks.
SIM_SECTIONS = ('domain', 'combustion', 'motion', 'spread', 'liquid')

# Which settings apply to which kind of simulation (domain 'kind'). Sections and keys not listed here
# apply to both.
KIND_SECTIONS = {'fire': ('combustion', 'motion', 'shading', 'embers', 'spread'), 'liquid': ('liquid', 'water')}
FIRE_ONLY_KEYS = {
    'domain': {'sponge', 'sponge_strength', 'mg_cycles', 'maccormack', 'maccormack_vel'},
    'composite': {'smoke_opacity', 'saturation', 'tint', 'light_cast', 'haze', 'haze_freq', 'haze_speed'},
    'emitter': {'fuel', 'temperature', 'smoke', 'swirl', 'swirl_width', 'douse', 'vapour', 'color_amount', 'color', 'noise',
                'noise_freq', 'noise_rise', 'contrast', 'seed', 'fade_in', 'fade_out', 'embers'},
    'collider': {'burnable'},
}
LIQUID_ONLY_KEYS = {'emitter': {'liquid_mode', 'flow', 'jitter'}, 'collider': {'floating', 'density'},
                    'lighting': {'environment', 'env_rotation', 'env_strength', 'env_sun'}}
BOTH_ONLY_KEYS = {'emitter': {'emits'}}


def applies(section, key, kind):
    """Whether a setting (or, with key None, a whole section) matters for a simulation kind."""
    if kind == 'both':
        return True
    if key in BOTH_ONLY_KEYS.get(section, ()):
        return False
    for k, secs in KIND_SECTIONS.items():
        if section in secs and k != kind:
            return False
    if key is None:
        return True
    if kind == 'liquid' and key in FIRE_ONLY_KEYS.get(section, ()):
        return False
    if kind != 'liquid' and key in LIQUID_ONLY_KEYS.get(section, ()):
        return False
    return True

_INDEX = {(s, p.key): p for s, ps in SECTIONS.items() for p in ps}
_EMITTER_INDEX = {p.key: p for p in EMITTER_PARAMS}
_COLLIDER_INDEX = {p.key: p for p in COLLIDER_PARAMS}


def param(section, key):
    if section == 'emitter':
        return _EMITTER_INDEX[key]
    if section == 'collider':
        return _COLLIDER_INDEX[key]
    return _INDEX[(section, key)]


def defaults(section):
    return {p.key: p.default for p in SECTIONS[section]}


def emitter_defaults():
    return {p.key: p.default for p in EMITTER_PARAMS}


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
