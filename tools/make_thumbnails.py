"""Render the preset library thumbnails (fire over black; liquids over a neutral ground; final quality)."""
import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from blackbody.engine.engine import Engine
from blackbody.scene import presets

FRAMES = {'campfire': 60, 'bonfire': 72, 'torch': 48, 'candle': 48, 'gas_ring': 36, 'pool_fire': 72, 'fire_line': 60,
          'fireball': 22, 'flamethrower': 48, 'vehicle_fire': 72, 'smoke_plume': 96,
          'fire_whirl': 144, 'waved_torch': 42, 'hose_douse': 72, 'grass_fire': 168, 'curtain_fire': 144, 'armchair_fire': 312,
          'room_fire': 170, 'coloured_flames': 48, 'road_flare': 48, 'grinder_sparks': 24, 'fireworks': 30, 'car_through_smoke': 72,
          'kettle_steam': 60, 'steam_vent': 96, 'flash_fire': 92, 'gas_cloud': 60, 'backdraft': 132, 'hillside_fire': 192,
          'spot_fires': 144, 'fabric_curtain': 192, 'wet_towels': 120, 'flag_wind': 72, 'towel_dip': 124,
          'water_pour': 45, 'rock_splash': 19, 'bucket_throw': 12, 'fountain': 48, 'hose': 36, 'wave': 30, 'waterfall': 36,
          'spill': 9, 'floating': 30, 'boat_wake': 45, 'honey': 72, 'lava': 96, 'hose_on_fire': 50,
          'sea_swell': 60, 'ink_tank': 48, 'oil_water': 72, 'river_post': 60, 'rain_pond': 30, 'pond_below': 40,
          'open_ocean': 60, 'storm_sea': 72, 'calm_lake': 48, 'harbour_chop': 60, 'beach_break': 96, 'shore_break': 96, 'reef_barrel': 40,
          'big_wave': 80, 'tsunami': 216, 'tidal_bore': 72, 'river_rocks': 72,
          'ice_cubes': 110, 'ice_melt': 60, 'pond_freeze': 72, 'frozen_pour': 96, 'boiling_pot': 48, 'hot_plate': 36, 'steaming_pool': 72,
          'boiling_throw': 36, 'snow_pond': 36, 'hail_pond': 240, 'cumulus_day': 300, 'thunderstorm': 600, 'lava_sea': 84, 'lava_quench': 56, 'lava_grass': 96,
          'tower_knockdown': 15, 'crates_in_fire': 110, 'wall_smash': 22, 'wrecking_ball': 44, 'yard_blast': 16, 'lightning_strike': 7, 'window_smash': 14, 'vase_drop': 24,
          'sand_hopper': 100, 'snowballs': 16, 'jelly_ball': 20, 'mud_drag': 44, 'sand_castle': 34, 'sand_sling': 66, 'iron_pour': 66, 'chocolate_fire': 140, 'chocolate_pan': 240, 'sheet_rip': 58,
          'crash_test': 18, 'stunt_fall': 30, 'chain_swing': 34}

# presets shown over an old lava field instead of paving slabs
FIELD_PLATES = {'lava'}

# presets shown over a photograph (tools/plates, SOURCES.txt there), the shot's camera matched to it: the
# photograph's lens (focal length, sensor width), its horizon row (the tilt) and the camera's height (a guess:
# a hand-held camera), the crop's top row for the shot's aspect; where the scene stands on the ground (how far
# ahead along the view, and where across the frame) and which way it faces (the bearing an orbit camera would
# have); the footage exposure (a grade); the scene's lights matched to the photograph's (the sun's bearing
# measured from the camera's, so it stays where the photograph has it however the scene turns)
PHOTO_PLATES = {
    'lava': dict(file='spatter_ground_kilauea_2024.jpg', focal_mm=4.25, sensor_mm=5.64, horizon=100, height=1.5,
                 top=40, distance=5.5, across=0.6, yaw=62.0, exposure=0.0, sun_bearing=210.0,
                 lighting=dict(sun_elevation=40.0, sun_intensity=3.0, ambient=(0.5, 0.57, 0.68))),
}


def photo_plate(sc, name, W, H):
    """Match the scene's camera and lights to a photograph (PHOTO_PLATES[name]) and return the photograph
    cropped to the shot's aspect and size, as footage (sRGB, 8-bit RGBA)."""
    from blackbody.engine import camera as cam
    p = PHOTO_PLATES[name]
    im = Image.open(Path(__file__).resolve().parent / 'plates' / p['file']).convert('RGB')
    pw, ph = im.size
    f_px = pw * p['focal_mm'] / p['sensor_mm']
    tilt = math.atan((ph / 2 - p['horizon']) / f_px)        # down from level
    Y = math.radians(p['yaw'])
    fwd = np.array([-math.sin(Y), 0.0, -math.cos(Y)])
    right = np.array([math.cos(Y), 0.0, -math.sin(Y)])
    D, Hc = p['distance'], p['height']
    depth = D * math.cos(tilt) + Hc * math.sin(tilt)         # the scene's base along the lens axis
    L = (p['across'] - 0.5) * pw * depth / f_px
    eye = -(fwd * D + right * L) + np.array([0.0, Hc, 0.0])
    rot = (-math.degrees(tilt), p['yaw'], 0.0)
    # where the scene's base lands in the photograph, and so in the crop: the crop's offset from the
    # photograph's centre is a lens shift, which Place in frame applies
    spec = cam.CameraSpec(mode='free', position=tuple(eye), rotation=rot, focal_mm=p['focal_mm'],
                          sensor_mm=p['sensor_mm'], use_anchor=False)
    px, _ = cam.project(cam.compute(spec, pw / ph), [(0.0, 0.0, 0.0)], pw, ph)
    crop_h = pw * H / W
    for k, v in dict(mode='free', position=tuple(float(x) for x in eye), rotation=rot, focal_mm=p['focal_mm'],
                     sensor_mm=p['sensor_mm'], use_anchor=True, anchor_x=float(px[0, 0] / pw),
                     anchor_y=float((px[0, 1] - p['top']) / crop_h), scale=1.0, roll=0.0).items():
        sc.set(('camera', k), v)
    sc.set(('composite', 'plate_exposure'), p['exposure'])
    sc.set(('lighting', 'ambient_from_footage'), False)
    for k, v in p['lighting'].items():
        sc.set(('lighting', k), v)
    sc.set(('lighting', 'sun_azimuth'), (p['yaw'] + p['sun_bearing']) % 360.0)
    top = int(round(p['top']))
    crop = im.crop((0, top, pw, top + int(round(crop_h)))).resize((W, H), Image.LANCZOS)
    out = np.full((H, W, 4), 255, np.uint8)
    out[..., :3] = np.asarray(crop)
    return out


def ground_plate(cs, W, H, dusk=False):
    """A neutral backplate seen through the shot camera (paving slabs to the horizon, a pale sky), so
    clear liquid, which shows only what it bends and reflects, has something to show."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64) + 0.5
    ndc = np.stack([xs / W * 2 - 1, 1 - ys / H * 2], -1)
    ones = np.ones((H, W, 1))
    a = np.concatenate([ndc, 0 * ones, ones], -1) @ cs.inv_view_proj.T
    b = np.concatenate([ndc, ones, ones], -1) @ cs.inv_view_proj.T
    a, b = a[..., :3] / a[..., 3:], b[..., :3] / b[..., 3:]
    d = b - a
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    t = -a[..., 1] / np.where(np.abs(d[..., 1]) < 1e-9, -1e-9, d[..., 1])
    hit = (d[..., 1] < 0) & (t > 0)
    q = a + d * t[..., None]
    u, v = q[..., 0] / 0.4, q[..., 2] / 0.4
    fu, fv = u - np.floor(u), v - np.floor(v)
    joint = np.minimum(np.minimum(fu, 1 - fu), np.minimum(fv, 1 - fv)) < 0.02
    rnd = (np.sin(np.floor(u) * 12.9898 + np.floor(v) * 78.233) * 43758.5453) % 1.0
    slab = np.stack([0.34 + 0.06 * rnd, 0.32 + 0.05 * rnd, 0.29 + 0.04 * rnd], -1)
    ground = np.where(joint[..., None], 0.16, slab)
    fog = np.exp(-np.maximum(t, 0) / 60.0)[..., None]
    sky = np.stack([0.52 + 0.2 * d[..., 1], 0.6 + 0.2 * d[..., 1], 0.72 + 0.18 * d[..., 1]], -1)
    img = np.where(hit[..., None], ground * fog + sky * 0.7 * (1 - fog), sky)
    if dusk:   # the same slabs at dusk, for scenes lit for it (molten liquids glowing on the ground)
        img = img * np.array([0.16, 0.17, 0.22])
    out = np.full((H, W, 4), 255, np.uint8)
    out[..., :3] = (np.clip(img, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    return out


class _Noise2:
    """Value noise on a periodic random lattice (quintic fade), about -1..1."""

    def __init__(self, seed, n=256):
        self.g = np.random.default_rng(seed).random((n, n)).astype(np.float32) * 2.0 - 1.0
        self.n = n

    def __call__(self, x, y):
        xi, yi = np.floor(x), np.floor(y)
        fx, fy = (x - xi).astype(np.float32), (y - yi).astype(np.float32)
        xi, yi = xi.astype(np.int64) % self.n, yi.astype(np.int64) % self.n
        x1, y1 = (xi + 1) % self.n, (yi + 1) % self.n
        u = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
        v = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
        g = self.g
        a = g[yi, xi] + (g[yi, x1] - g[yi, xi]) * u
        b = g[y1, xi] + (g[y1, x1] - g[y1, xi]) * u
        return a + (b - a) * v


def lava_field_plate(cs, W, H, sun_dir=(0.0, 0.1, -1.0), sun=(0.35, 0.33, 0.3), sky=(0.1, 0.12, 0.2), seed=7):
    """A backplate of an old pahoehoe lava field seen through the shot camera, for molten presets: dark
    basalt in lumps and lobes a metre or so across, with clefts between them, patches of ropes, glassy
    silver-grey and oxidised rust-brown patches; lit by the scene's sky and low sun, fading into dusk haze.
    Display-referred sRGB, as footage is."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64) + 0.5
    ndc = np.stack([xs / W * 2 - 1, 1 - ys / H * 2], -1)
    ones = np.ones((H, W, 1))
    a = np.concatenate([ndc, 0 * ones, ones], -1) @ cs.inv_view_proj.T
    b = np.concatenate([ndc, ones, ones], -1) @ cs.inv_view_proj.T
    a, b = a[..., :3] / a[..., 3:], b[..., :3] / b[..., 3:]
    d = b - a
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    t = -a[..., 1] / np.where(np.abs(d[..., 1]) < 1e-9, -1e-9, d[..., 1])
    hit = (d[..., 1] < 0) & (t > 0)
    t = np.where(hit, t, 1e3)
    q = a + d * t[..., None]
    x, z = q[..., 0], q[..., 2]
    # a pixel's footprint on the ground (m): detail finer than about two of them fades out
    pix = 2.0 / H / cs.proj[1, 1] if hasattr(cs, 'proj') else 1.2 / H
    foot = t * pix / np.maximum(-d[..., 1], 0.05)
    n1, n2, n3, n4 = (_Noise2(seed + k) for k in range(4))

    def fade(lam):
        return np.clip(1.5 - foot * 3.0 / lam, 0.0, 1.0)

    def height(x, z):
        # pillowy lobes and toes (billows: rounded tops, creased where they meet), ropes in patches, a rough rind
        h = np.zeros_like(x)
        for k, (lam, amp) in enumerate(((1.6, 0.16), (0.7, 0.07), (0.3, 0.03), (0.13, 0.01))):
            wx = 0.35 * n3(x / lam + 2.1 * k, z / lam + 5.3 * k)
            h += amp * (np.abs(n1(x / lam + 13.1 * k + wx, z / lam + 7.7 * k - wx)) - 0.5) * fade(lam)
        ang = n3(x / 1.6, z / 1.6) * 2.5
        warp = n3(x / 0.25 + 5.0, z / 0.25 + 2.0) + 0.5 * n2(x / 0.1, z / 0.1)
        ph = (x * np.cos(ang) + z * np.sin(ang)) / 0.04 + warp * 2.0
        rope_mask = np.clip(n4(x / 0.6, z / 0.6) * 2.5 - 0.8, 0.0, 1.0) * (0.5 + 0.5 * n2(x / 0.12, z / 0.12))
        h += 0.003 * np.sin(ph * 2 * np.pi) * rope_mask * fade(0.04)
        for k, (lam, amp) in enumerate(((0.05, 0.004), (0.02, 0.0018), (0.008, 0.0007))):
            h += amp * n4(x / lam + 3.3 * k, z / lam + 9.1 * k) * fade(lam)
        return h

    e = np.maximum(foot * 0.7, 0.003)
    h0 = height(x, z)
    hx = (height(x + e, z) - h0) / e
    hz = (height(x, z + e) - h0) / e
    nrm = np.stack([-hx, np.ones_like(hx), -hz], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    # creases and hollows: lower than the ground around them, darker
    cav = np.clip(0.55 + h0 / 0.12, 0.2, 1.0)
    glass = np.clip(n2(x / 0.7 + 4.0, z / 0.7 + 1.0) * 1.8, 0.0, 1.0)          # fresher, glassier skin
    rust = np.clip(n3(x / 1.3 + 8.0, z / 1.3 + 3.0) * 2.0 - 0.6, 0.0, 1.0)      # oxidised
    grain = 0.5 + 0.5 * n4(x / 0.03 + 1.0, z / 0.03 + 7.0) * fade(0.03)
    alb = np.stack([0.05, 0.048, 0.046], 0)[None, None] * (0.7 + 0.6 * grain)[..., None]
    alb = alb * (1 - glass[..., None] * 0.25) + rust[..., None] * np.array([0.025, 0.012, 0.005])
    sd = np.asarray(sun_dir, float)
    sd = sd / np.linalg.norm(sd)
    sky = np.asarray(sky, float)
    sun = np.asarray(sun, float)
    ndl = np.clip(nrm @ sd, 0.0, 1.0)
    diff = alb * (sky[None, None] * (0.55 + 0.45 * nrm[..., 1:2]) * cav[..., None] + sun * ndl[..., None])
    # the glassy rind's sheen: the sky in it, brightest toward grazing, and the low sun's glints
    v = -d
    ndv = np.clip(np.sum(nrm * v, -1), 0.0, 1.0)
    fr = 0.04 + 0.96 * (1 - ndv) ** 5
    sheen = (0.1 + 0.4 * glass) * fr
    hv = sd[None, None] + v
    hv /= np.linalg.norm(hv, axis=-1, keepdims=True)
    ndh = np.clip(np.sum(nrm * hv, -1), 0.0, 1.0)
    rough = 0.35 - 0.15 * glass
    a2 = rough ** 4
    D = a2 / (np.pi * (ndh * ndh * (a2 - 1) + 1) ** 2)
    spec = (D * 0.05 * ndl * 0.25)[..., None] * sun
    grey_sky = sky.mean() * 0.5 + sky * 0.5
    col = diff + sheen[..., None] * grey_sky * 1.3 * cav[..., None] + spec
    # sky: a dusk gradient, and haze over distance
    up = np.clip(d[..., 1], 0.0, 1.0)[..., None]
    sky_img = sky * 1.6 * (1.0 - up) + sky * np.array([0.6, 0.65, 0.9]) * up + np.array([0.02, 0.012, 0.006]) * (1 - up) ** 6
    fog = np.exp(-np.maximum(t, 0) / 60.0)[..., None]
    img = np.where(hit[..., None], col * fog + sky * 1.4 * (1 - fog), sky_img)
    out = np.full((H, W, 4), 255, np.uint8)
    out[..., :3] = (np.clip(img, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('names', nargs='*')
    ap.add_argument('--out', default=str(ROOT / 'blackbody' / 'assets' / 'presets'))
    ap.add_argument('--size', default='640x360')
    ap.add_argument('--samples', type=int, default=4)
    ap.add_argument('--mode', default='fire')
    ap.add_argument('--draft', action='store_true')
    ap.add_argument('--frame', type=int, default=None)
    ap.add_argument('--photo', action='store_true', help='presets with a photograph (PHOTO_PLATES): over it, the '
                    'camera and lights matched to it, instead of over a generated plate')
    args = ap.parse_args()
    W, H = (int(x) for x in args.size.split('x'))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eng = Engine()
    for name in args.names or presets.ORDER:
        t0 = time.perf_counter()
        sc = presets.make(name)
        sc.data['render']['width'], sc.data['render']['height'] = W, H
        if sc.kind in ('liquid', 'both') and sc.colliders:
            sc.data['water']['colliders_look'] = 'shaded'   # the plate has no wall or rock: show stand-ins
        f = sc.start + (args.frame or FRAMES.get(name, 60)) - 1
        eng.prepare(sc, final=not args.draft)
        eng.simulate_to(sc, f, cache=False)
        if sc.kind in ('liquid', 'both'):
            from blackbody.engine import camera as cam
            plate = photo_plate(sc, name, W, H) if (args.photo and name in PHOTO_PLATES) else None
            spec, fire = sc.camera(f)
            cs = cam.compute(spec, W / H, fire)
            if plate is not None:
                pass
            elif name in FIELD_PLATES:   # lava on an old lava field, lit as the scene is (so not lit from it)
                sc.set(('lighting', 'ambient_from_footage'), False)
                L = sc.data['lighting']
                plate = lava_field_plate(cs, W, H, cam.sun_direction(L['sun_azimuth'], L['sun_elevation']),
                                         tuple(np.array([1.0, 0.93, 0.85]) * L['sun_intensity']), L['ambient'])
            else:
                plate = ground_plate(cs, W, H, dusk=sc.data['water'].get('glow', 0.0) > 0.0)
            eng.render(sc, f, (W, H), mode='composite', final=not args.draft, samples=args.samples, motion_blur=True,
                       plate=plate)
        else:
            # (one on the stage shows it: its floor, sky and objects are in the composite)
            mode = 'composite' if sc.data['composite'].get('backdrop') == 'stage' else args.mode
            eng.render(sc, f, (W, H), mode=mode, final=not args.draft, samples=args.samples, motion_blur=True)
        img = eng.display_image()
        Image.fromarray(np.ascontiguousarray(img[..., :3])).save(out / f'{name}.png')
        st = eng.stats()
        print(f'{name:14s} frame {f} dims {st["dims"]} {time.perf_counter() - t0:.1f}s')


if __name__ == '__main__':
    main()
