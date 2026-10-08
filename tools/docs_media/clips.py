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
    'sheet_rip': clip('sheet_rip', 0.0, 108, dyaw=6.0, push=0.05),     # the crate rests by 1 s, the ball rips through at 2.2 s
    # the car hits the figure at 0.5 s and the wall at 0.7 s and runs on to stop 7.6 m on: a still camera pans after it
    'crash_test': clip('crash_test', 0.0, 72,
                       camera=dict(mode='free', position=(4.0, 3.0, 11.0), rotation=(-10.82, 26.57, 0.0), focal_mm=40,
                                   use_anchor=False),
                       camera_keys={'rotation': [(0.0, (-10.82, 26.57, 0.0)), (0.8, (-11.83, 11.31, 0.0)),
                                                 (1.3, (-12.06, 0.0, 0.0)), (1.8, (-11.94, -8.28, 0.0)),
                                                 (2.6, (-11.81, -11.81, 0.0))]}),
    'stunt_fall': clip('stunt_fall', 0.0, 72, dyaw=4.0, push=0.04),     # the ball hits at 0.8 s, it lands by 2 s
    'chain_swing': clip('chain_swing', 0.0, 96, dyaw=5.0, push=0.04),   # the weight hits the crates at 1.2 s
    'yard_blast': clip('yard_blast', 0.0, 96, dyaw=6.0, push=0.05),
    'cart_jump': clip('cart_jump', 0.4, 100, dyaw=4.0, push=0.04),   # the action runs 0.5-4.5 s
    'meadow_fire': clip('meadow_fire', 0.5, 168, dyaw=4.0, push=0.04),   # the front crosses the field in about 8 s
    # (the whole burn-down, 40 s, played twice as fast: the corner fire grows, flames roll out under the roof, the walls
    # catch one by one, then 200x faster it chars through and falls in)
    'shed_fire': clip('shed_fire', 0.0, 480, every=2, dyaw=4.0, push=0.03),
    'lightning_strike': clip('lightning_strike', 0.0, 96, dyaw=4.0, push=0.04),   # the flash is at frames 6-10
    'window_smash': clip('window_smash', 0.0, 48, dyaw=6.0, push=0.04),
    'vase_drop': clip('vase_drop', 0.0, 48, dyaw=6.0, push=0.04),
    # ---- sand, snow and mud (on the stage) -------------------------------------------------------------------------
    'sand_hopper': clip('sand_hopper', 0.0, 144, dyaw=6.0, push=0.04),
    'snowballs': clip('snowballs', 0.0, 60, dyaw=4.0, push=0.04),
    'jelly_ball': clip('jelly_ball', 0.0, 72, dyaw=6.0, push=0.04),
    'mud_drag': clip('mud_drag', 0.0, 72, dyaw=6.0, push=0.04),
    'sand_castle': clip('sand_castle', 0.5, 180, dyaw=6.0, push=0.08),   # the wave at 1 s, slumped by about 6 s
    'sand_sling': clip('sand_sling', 0.0, 120, dyaw=6.0, push=0.06),    # poured from 0 to 3 s, settled in the sheet after
    'iron_pour': clip('iron_pour', 0.0, 192, dyaw=6.0, push=0.06),      # poured from 0.2 s to 3.7 s, setting after
    'chocolate_fire': clip('chocolate_fire', 0.5, 216, dyaw=4.0, push=0.06),   # the nearest piece runs by about 4 s
    'chocolate_pan': clip('chocolate_pan', 0.0, 240, dyaw=4.0, push=0.05),     # soft by 3 s, pools run together by 8 s
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
    # (on its own stage, the floor and sky, where the steel glows: let go at 0.6 s, in the water at about 0.9 s, held off
    # it by its steam, then boiling it hard as its glow fades; the steam peaks at about 1.2-1.5 s)
    'quench': clip('quench', 0.5, 72, camera=dict(pitch=40, target_y=0.2, anchor_y=0.5),
                   camera_keys={'yaw': [(0.0, 13.5), (3.0, 16.5)], 'distance': [(0.0, 1.3), (3.0, 1.22)]}),
    'boiling_throw': clip('boiling_throw', 0.1, 96, plate='ground', plate_args=dict(DAY_SKY)),
    'frozen_pour': clip('frozen_pour', 0.5, 96, plate='ground', plate_args=DAY_FINE),
    'steaming_pool': clip('steaming_pool', 0.5, 96, plate='ground', plate_args=DAY_SKY),
    # ---- bullets and wood (each on its own stage) ------------------------------------------------------------
    'shooting_range': clip('shooting_range', 0.0, 96, dyaw=3.0, push=0.04),   # pane ~7.6, gong 22/28, can 39, board 48/54, block 60/66, dust
    'steel_gong': clip('steel_gong', 4 / 24, 72, camera=dict(pitch=4, target_y=0.9),
                       camera_keys={'yaw': [(0.0, -41.5), (3.0, -38.5)], 'distance': [(0.0, 1.32), (3.0, 1.27)]}),   # all six hits (sparks last a frame: GIF at half speed)
    'glass_slowmo': clip('glass_slowmo', 2 / 24, 72, dyaw=3.0, push=0.04),    # slowed 400x: in view ~10, hits ~19, shards drift 40-59
    'bottle_shoot': clip('bottle_shoot', 6 / 24, 126, camera=dict(pitch=6, target_y=0.85),
                         camera_keys={'yaw': [(0.0, -13.5), (5.25, -10.5)], 'distance': [(0.0, 2.45), (5.25, 2.33)]}),   # bursts at 13, 37, 61, 85, 109
    'machine_gun': clip('machine_gun', 0.0, 72, dyaw=3.0, push=0.04),         # skips off the dirt from ~8, then walks up the wall
    'board_break': clip('board_break', 1.0, 60, dyaw=3.0, push=0.04),         # slowed 4x: it snaps at frames 48-50
    'boards_shot': clip('boards_shot', 0.0, 72, dyaw=3.0, push=0.04),         # from behind: the pistol's exit at ~10, the rifle's at 29
    'wood_lineup': clip('wood_lineup', 0.0, 96, dyaw=24.0, push=0.12),        # (nothing moves: the camera turns along the woods)
    # ---- the materials meeting (wind on snow, a log on lava) ------------------------------------------------------
    'snow_drift': clip('snow_drift', 0.8, 120, dyaw=4.0, push=0.04),          # the spindrift builds from about 1 s
    'lava_raft': clip('lava_raft', 1.0, 72, plate='ground', plate_args=NIGHT, set={'lighting': DUSK}, dyaw=4.0,
                      push=0.04),                                         # the log floats off at the flow's front
    # ---- weather and clouds ---------------------------------------------------------------------------------------
    'snow_pond': clip('snow_pond', 0.5, 120, plate='ground', plate_args=DAY_SKY, draft=True),
    'hail_pond': clip('hail_pond', 3.0, 120, plate='ground', plate_args=DAY_SKY, draft=True),
    'cumulus_day': clip('cumulus_day', 0.0, 150, every=2, dyaw=4.0, push=0.0),
    'thunderstorm': clip('thunderstorm', 0.0, 150, every=4, dyaw=4.0, push=0.0),
}
