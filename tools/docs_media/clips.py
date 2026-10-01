"""The showcase clips for the README and the site: a preset, from when (`at`, seconds into the preset's own
timeline), how many frames, `every` (simulated frames per played frame: 2 is a 2x time-lapse), a gentle
camera move (keyframes in played seconds) built from the preset's own camera, overrides and a backplate."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))

from blackbody.scene import presets                   # noqa: E402

NIGHT = dict(kind='dirt', night=True)
NIGHT_CONCRETE = dict(kind='concrete', night=True)
DAY_PEBBLES = dict(kind='pebbles', night=False, fog=25.0)
DAY_STONE = dict(kind='stone', night=False, fog=25.0)
DAY_SKY = dict(kind='sky', night=False)
DAY_FINE = dict(kind='concrete', night=False, fog=25.0)   # tabletop scenes: pebbles would look like boulders
DAY_DIRT = dict(kind='dirt', night=False, fog=25.0)
SHARP = {'flame_sharpness': 2.4, 'exposure': -0.5}
LIT = {'light_cast': 1.4, 'surface_light': 1.4}
DUSK = {'sun_on': True, 'sun_intensity': 0.15, 'sun_elevation': 8.0, 'ambient': (0.05, 0.06, 0.09)}


def drift(preset, seconds, dyaw=6.0, push=0.08):
    """A slow orbit and push-in from the preset's own camera over the clip."""
    c = presets.make(preset).data['camera']
    if c.get('mode') == 'free':
        return {}
    y, d = float(c['yaw']), float(c['distance'])
    return {'yaw': [(0.0, y - dyaw / 2), (seconds, y + dyaw / 2)],
            'distance': [(0.0, d), (seconds, d * (1.0 - push))]}


def clip(preset, at, frames, every=1, plate=None, plate_args=None, dyaw=6.0, push=0.08, **kw):
    secs = frames / 24.0
    c = dict(preset=preset, at=at, frames=frames, every=every, samples=kw.pop('samples', 3),
             camera_keys=kw.pop('camera_keys', None) or drift(preset, secs, dyaw, push))
    if plate:
        c['plate'] = plate
        c['plate_args'] = plate_args or {}
    c.update(kw)
    return c


CLIPS = {
    # ---- fire --------------------------------------------------------------------------------------------
    'campfire': clip('campfire', 1.5, 120, plate='ground', plate_args=NIGHT,
                     set={'composite': LIT, 'shading': {'flame_sharpness': 3.8, 'detail': 0.8, 'exposure': -0.6}},
                     camera=dict(pitch=4, target_y=0.75, anchor_y=0.84, focal_mm=35),
                     camera_keys={'yaw': [(0.0, -8.0), (5.0, 6.0)], 'distance': [(0.0, 4.2), (5.0, 3.6)]}),
    'fire_whirl': clip('fire_whirl', 3.0, 96, plate='ground', plate_args=NIGHT, set={'shading': SHARP, 'composite': LIT}),
    'fabric_curtain': clip('fabric_curtain', 0.5, 168, plate='ground', plate_args=NIGHT_CONCRETE, dyaw=4.0),
    'wet_towels': clip('wet_towels', 2.0, 120, plate='ground', plate_args=NIGHT_CONCRETE, dyaw=4.0),
    'flag_wind': clip('flag_wind', 1.0, 96, plate='ground', plate_args=NIGHT),
    'hillside_fire': clip('hillside_fire', 2.5, 120, plate='ground', plate_args=NIGHT, set={'composite': LIT}),
    'fireworks': clip('fireworks', 0.3, 96, plate='ground',
                      plate_args=dict(NIGHT, horizon=(0.006, 0.007, 0.011), zenith=(0.0015, 0.002, 0.005), gain=0.12)),
    'coloured_flames': clip('coloured_flames', 1.0, 96),
    'hose_douse': clip('hose_douse', 1.5, 120, plate='ground', plate_args=NIGHT, set={'composite': LIT}),
    'car_through_smoke': clip('car_through_smoke', 0.5, 96, plate='ground', plate_args=dict(DAY_STONE, fog=60.0), dyaw=4.0),
    'flamethrower': clip('flamethrower', 0.8, 96, plate='ground', plate_args=NIGHT_CONCRETE,
                         set={'shading': SHARP, 'composite': LIT}),
    'gas_cloud': clip('gas_cloud', 1.0, 72, plate='ground', plate_args=NIGHT_CONCRETE, set={'composite': LIT}),
    'kettle_steam': clip('kettle_steam', 1.5, 96),
    'backdraft': clip('backdraft', 2.0, 144, plate='ground', plate_args=NIGHT_CONCRETE, dyaw=4.0, set={'composite': LIT}),
    'spot_fires': clip('spot_fires', 2.0, 120, plate='ground', plate_args=NIGHT, set={'composite': LIT}),
    # ---- things that fall (on the stage: no plate) --------------------------------------------------------------
    'tower_knockdown': clip('tower_knockdown', 0.0, 96, dyaw=8.0, push=0.06),
    'crates_in_fire': clip('crates_in_fire', 0.0, 144, dyaw=6.0, set={'composite': LIT}),
    'wall_smash': clip('wall_smash', 0.0, 72, dyaw=8.0, push=0.06),
    'wrecking_ball': clip('wrecking_ball', 0.0, 96, dyaw=6.0, push=0.05),
    'window_smash': clip('window_smash', 0.0, 48, dyaw=6.0, push=0.04),
    'vase_drop': clip('vase_drop', 0.0, 48, dyaw=6.0, push=0.04),
    # ---- sand, snow and mud (on the stage) -------------------------------------------------------------------------
    'sand_hopper': clip('sand_hopper', 0.0, 144, dyaw=6.0, push=0.04),
    'snowballs': clip('snowballs', 0.0, 60, dyaw=4.0, push=0.04),
    'jelly_ball': clip('jelly_ball', 0.0, 72, dyaw=6.0, push=0.04),
    'mud_drag': clip('mud_drag', 0.0, 72, dyaw=6.0, push=0.04),
    # ---- water ----------------------------------------------------------------------------------------------
    'floating': clip('floating', 0.2, 96, plate='ground', plate_args=DAY_PEBBLES),
    'ink_tank': clip('ink_tank', 0.3, 120, plate='ground', plate_args=DAY_FINE, dyaw=8.0),
    'oil_water': clip('oil_water', 0.3, 120, plate='ground', plate_args=DAY_FINE),
    'honey': clip('honey', 0.5, 120, plate='ground', plate_args=DAY_STONE),
    'boat_wake': clip('boat_wake', 0.5, 96, plate='ground', plate_args=DAY_SKY, dyaw=4.0, push=0.04),
    'river_rocks': clip('river_rocks', 1.0, 96, plate='ground', plate_args=DAY_SKY, dyaw=4.0),
    'hose_on_fire': clip('hose_on_fire', 1.5, 96, plate='ground', plate_args=NIGHT_CONCRETE,
                         set={'lighting': DUSK, 'composite': LIT, 'shading': SHARP}),
    'fountain': clip('fountain', 0.5, 96, plate='ground', plate_args=DAY_PEBBLES),
    'waterfall': clip('waterfall', 0.5, 96, plate='ground', plate_args=DAY_PEBBLES),
    'rock_splash': clip('rock_splash', 0.4, 72, plate='ground', plate_args=DAY_PEBBLES),
    # ---- the sea ---------------------------------------------------------------------------------------------
    'storm_sea': clip('storm_sea', 1.0, 96, plate='ground', plate_args=DAY_SKY, dyaw=4.0, push=0.04, draft=True),
    'calm_lake': clip('calm_lake', 1.0, 96, plate='ground', plate_args=DAY_SKY, dyaw=4.0, push=0.04),
    'beach_break': clip('beach_break', 1.5, 120, plate='ground', plate_args=DAY_SKY, dyaw=4.0, push=0.04),
    'sea_swell': clip('sea_swell', 0.5, 96, plate='ground', plate_args=DAY_SKY, dyaw=4.0, push=0.04),
    'harbour_chop': clip('harbour_chop', 0.5, 96, plate='ground', plate_args=DAY_SKY, dyaw=4.0, push=0.04),
    'tsunami': clip('tsunami', 4.0, 96, every=2, plate='ground', plate_args=DAY_SKY, draft=True),
    # ---- lava ------------------------------------------------------------------------------------------------
    'lava': clip('lava', 1.5, 120, plate='ground', plate_args=NIGHT, set={'lighting': DUSK}),
    'lava_sea': clip('lava_sea', 1.0, 96, plate='ground', plate_args=DAY_SKY),
    'lava_grass': clip('lava_grass', 1.0, 96, plate='ground', plate_args=NIGHT, set={'lighting': DUSK}),
    'lava_quench': clip('lava_quench', 0.8, 96, plate='ground', plate_args=DAY_DIRT),
    # ---- heat: ice, boiling, steam ------------------------------------------------------------------------------
    'ice_cubes': clip('ice_cubes', 0.3, 96, plate='ground', plate_args=DAY_FINE),
    'boiling_pot': clip('boiling_pot', 0.8, 96, plate='ground', plate_args=DAY_FINE),
    'pond_freeze': clip('pond_freeze', 0.0, 120, plate='ground', plate_args=DAY_SKY),
    'hot_plate': clip('hot_plate', 0.2, 96, plate='ground', plate_args=DAY_STONE),
    'boiling_throw': clip('boiling_throw', 0.1, 96, plate='ground', plate_args=dict(DAY_SKY)),
    'frozen_pour': clip('frozen_pour', 0.5, 96, plate='ground', plate_args=DAY_FINE),
    'steaming_pool': clip('steaming_pool', 0.5, 96, plate='ground', plate_args=DAY_SKY),
    # ---- weather and clouds ---------------------------------------------------------------------------------------
    'snow_pond': clip('snow_pond', 0.5, 120, plate='ground', plate_args=DAY_SKY, draft=True),
    'hail_pond': clip('hail_pond', 3.0, 120, plate='ground', plate_args=DAY_SKY, draft=True),
    'cumulus_day': clip('cumulus_day', 0.0, 150, every=2, dyaw=4.0, push=0.0),
    'thunderstorm': clip('thunderstorm', 0.0, 150, every=4, dyaw=4.0, push=0.0),
}
