"""Fitting the fire into footage: highlights that go to white like a camera's, haze by distance, the
footage's own noise, and the coal bed."""
import numpy as np
import pytest

from blackbody.io.colour import view_transform
from blackbody.io.images import camera_rolloff, rolloff
from blackbody.io.platestats import NOISE_BINS, NOISE_LO, haze_colour, measure_noise, plate_linear
from blackbody.scene import presets


def _noise_plate(h, w, blur=0.0, corr=0.6, seed=1):
    """A smooth gradient plate (linear) with noise whose variance grows with brightness, and the truth."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, w)[None, :, None]
    base = np.broadcast_to(0.002 * 400.0 ** x, (h, w, 3)).astype(np.float32).copy()
    n = rng.standard_normal((h, w, 3))
    if blur > 0:
        r = int(3 * blur) + 1
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / blur) ** 2)
        k /= k.sum()
        n = np.apply_along_axis(lambda v: np.convolve(v, k, 'same'), 1, n)
        n = np.apply_along_axis(lambda v: np.convolve(v, k, 'same'), 0, n)
        n /= n.std()
    c = np.array([[1.0, corr, corr], [corr, 1.0, corr], [corr, corr, 1.0]])
    n = n @ np.linalg.cholesky(c).T
    sigma = lambda v: np.sqrt(1e-6 + 2e-4 * v)
    return (base + sigma(base) * n).astype(np.float32), sigma


def test_highlights_leave_footage_alone_and_go_to_white():
    rng = np.random.default_rng(0)
    footage = rng.uniform(0.0, 0.8, (64, 64, 3)).astype(np.float32)
    assert np.abs(camera_rolloff(footage, 0.8, 0.25) - footage).max() < 1e-6, 'nothing below the knee changes'
    grey = np.full((4, 3), 3.0, np.float32)
    assert np.allclose(camera_rolloff(grey, 0.8, 0.25), rolloff(grey, 0.8), atol=1e-6), 'neutrals roll off as before'
    flame = np.array([1.0, 0.25, 0.007], np.float32)
    plain = np.clip(rolloff(flame * 16.0, 0.8), 0, 1)
    cam = np.clip(camera_rolloff(flame * 16.0, 0.8, 0.25), 0, 1)
    assert plain[2] < 0.15, 'per channel, over-bright flame clips to a flat yellow'
    assert cam[2] > 0.5 and cam.min() > 0.5, 'with crosstalk it goes nearly white'
    blues = [np.clip(camera_rolloff(flame * e, 0.8, 0.25), 0, 1)[2] for e in (1.0, 2.0, 4.0, 8.0, 16.0)]
    assert np.all(np.diff(blues) > 0), 'whitening grows with exposure'
    # the export path matches
    out = view_transform(flame[None, None] * 16.0, 'standard', 0.8, 0.25)
    assert out.min() > 0.7


@pytest.mark.parametrize('blur', [0.0, 1.2])
def test_noise_is_measured(blur):
    img, sigma = _noise_plate(240, 480, blur=blur)
    m = measure_noise(img)
    centres = 2.0 ** (NOISE_LO + np.arange(NOISE_BINS))
    ok = (centres > 0.004) & (centres < 0.5)
    assert np.allclose(m.sigma[ok, 1], sigma(centres[ok]), rtol=0.12), 'strength at each brightness'
    assert m.size == pytest.approx(blur, abs=0.3), 'grain size'
    c = m.chol @ m.chol.T
    assert c[0, 1] == pytest.approx(0.6, abs=0.08) and c[1, 2] == pytest.approx(0.6, abs=0.08), 'channel correlation'
    assert np.allclose(m.sigma_at(centres[ok][:, None].repeat(3, 1))[:, 1], m.sigma[ok, 1], rtol=1e-5)


def test_noise_under_one_code_value_is_still_measured():
    """8-bit footage whose noise is under a code value: most neighbour differences are exactly zero."""
    rng = np.random.default_rng(2)
    code = 120.0 + rng.standard_normal((200, 300, 1)) * 0.45
    img = np.clip(np.round(np.repeat(code, 3, axis=2)), 0, 255).astype(np.uint8)
    m = measure_noise(plate_linear(img, 'srgb'))
    assert m is not None and m.sigma.max() > 0.0


def test_haze_colour_is_the_hazy_sky():
    rng = np.random.default_rng(3)
    img = np.zeros((180, 320, 3), np.float32)
    img[:80] = (0.30, 0.26, 0.38)                                    # hazy sky
    img[80:] = rng.uniform(0.0, 0.2, (100, 320, 3))                   # dark, textured ground
    img[120:125, 150:155] = (1.0, 0.9, 0.2)                           # a small bright lamp
    assert np.allclose(haze_colour(img), (0.30, 0.26, 0.38), atol=0.02)


def test_eight_bit_linearisation_matches_the_formula():
    codes = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, 2)
    x = codes.astype(np.float32) / 255.0
    ref = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    assert np.allclose(plate_linear(codes, 'srgb', 2.0), ref * 2.0, atol=1e-6)


# -- on the GPU ---------------------------------------------------------------------------------------

def _campfire(engine, res=48):
    """The campfire with dark smoke and plain motion, so the tests have dense smoke to look through."""
    sc = presets.make('campfire')
    sc.data['domain'].update(resolution=res, time_scale=1.0)
    sc.data['motion']['puffing'] = 0.0
    sc.data['shading'].update(smoke_albedo=(0.22, 0.21, 0.20), smoke_density=5.0)
    sc.data['embers']['enabled'] = False
    engine.invalidate()
    engine.prepare(sc)
    f = sc.start + 30
    engine.simulate_to(sc, f, cache=True)
    return sc, f


def test_coal_bed_glows_at_the_base(engine):
    sc, f = _campfire(engine)
    sc.data['camera']['pitch'] = 35.0  # look down on the bed
    out = {}
    for cb in (0.0, 1.0, 2.0):
        sc.data['shading']['coal_bed'] = cb
        engine.render(sc, f, (256, 256), mode='fire')
        aov = engine.aovs()
        out[cb] = (aov['emission'][..., :3].astype(np.float32).sum(-1), aov['mask'][..., 0].astype(np.float32),
                   aov['beauty'][..., 3].astype(np.float32))
    bed = out[1.0][1] > 0.2
    assert bed.sum() > 50, 'the ground under the coals is charred'
    assert out[0.0][1].max() == 0.0
    assert (out[1.0][2] - out[0.0][2]).max() > 0.9, 'the coals are solid: they hide the footage'
    assert out[0.0][0][bed].sum() < out[1.0][0][bed].sum() < out[2.0][0][bed].sum(), 'and they glow'


def test_haze_fades_the_fire_toward_the_haze_colour(engine):
    sc, f = _campfire(engine)
    sc.data['composite']['atmos_from_footage'] = False
    sc.data['composite']['atmos_colour'] = (0.5, 0.5, 0.5)
    out = {}
    for vis in (0.0, 20.0):
        sc.data['composite']['visibility'] = vis
        engine.render(sc, f, (192, 192), mode='fire')
        out[vis] = engine.renderer.read_linear()[..., :3].astype(np.float32)
        alpha = engine.aovs()['beauty'][..., 3].astype(np.float32)
    smoke = (alpha > 0.5) & (out[0.0].max(-1) < 0.1)
    assert smoke.sum() > 50
    assert out[20.0][smoke].mean() > out[0.0][smoke].mean() + 0.05, 'dark smoke lifts toward grey haze'


def test_fire_gets_the_footage_noise(engine):
    sc, f = _campfire(engine)
    sc.data['composite']['bloom'] = 0.0
    sc.data['composite']['light_cast'] = 0.0
    sc.data['composite']['surface_light'] = 0.0
    sc.data['shading']['coal_bed'] = 0.0
    rng = np.random.default_rng(5)
    plate = np.clip(np.round(60.0 + rng.standard_normal((192, 192, 1)) * 6.0), 0, 255).astype(np.uint8)
    plate = np.concatenate([np.repeat(plate, 3, 2), np.full((192, 192, 1), 255, np.uint8)], -1)
    lin = plate_linear(plate, 'srgb')
    lin_mean = sum(np.roll(np.roll(lin, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0
    plate_std = float((lin - lin_mean).std())
    std = {}
    for match in (False, True):
        sc.data['composite']['grain_match'] = match
        sc.data['composite']['grain'] = 0.0
        engine.render(sc, f, (192, 192), mode='composite', plate=plate)
        comp = engine.renderer.read_linear()[..., :3].astype(np.float32)
        alpha = engine.aovs()['beauty'][..., 3].astype(np.float32)
        smoke = (alpha > 0.97) & (comp.max(-1) < 0.2)
        # noise: difference from the 3x3 mean, where the smoke hides the footage
        mean = sum(np.roll(np.roll(comp, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0
        std[match] = float((comp - mean)[smoke].std()) if smoke.sum() > 30 else None
    assert std[False] is not None, 'the fire has dense smoke to measure over'
    added = np.sqrt(max(std[True] ** 2 - std[False] ** 2, 0.0))
    assert added == pytest.approx(plate_std, rel=0.3), 'the smoke picks up as much noise as the footage has'
