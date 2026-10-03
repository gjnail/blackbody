"""The physical sky (engine/sky.py; Lighting › Sky: Physical): its shape against the CIE's standard clear sky, its
brightness against what clear skies measure, the sun reddening as it sets, haze whitening the sky, the HDRI's
convention, and the scene's look and renders taking it up."""
import math

import numpy as np
import pytest

from blackbody.engine import sky
from blackbody.scene import presets

ZENITH = np.array([[0.0, 1.0, 0.0]])
SUN_LUX = 128000.0          # the sun's illuminance outside the air (lux)


def _lum(rgb):
    return np.asarray(rgb) @ sky.LUMA


def _cie_clear(D, s):
    """The CIE standard clear sky (type 12, ISO 15469): luminance over the zenith's along directions D."""
    a, b, c, d, e = -1.0, -0.32, 10.0, -3.0, 0.45
    Z, Zs = np.arccos(np.clip(D[:, 1], -1, 1)), math.acos(s[1])
    chi = np.arccos(np.clip(D @ s, -1, 1))
    phi = lambda z: 1.0 + a * np.exp(b / np.cos(z))
    f = lambda x: 1.0 + c * (np.exp(d * x) - math.exp(d * math.pi / 2)) + e * np.cos(x) ** 2
    return f(chi) * phi(Z) / (f(Zs) * phi(0.0))


def _sky_dirs():
    az, alt = np.meshgrid(np.radians(np.arange(0, 360, 10)), np.radians(np.arange(5, 86, 8)))
    return np.stack([np.cos(alt) * np.sin(az), np.sin(alt), -np.cos(alt) * np.cos(az)], -1).reshape(-1, 3)


@pytest.mark.parametrize('elevation', [60, 30, 15])
def test_the_sky_has_the_shape_of_the_cie_clear_sky(elevation):
    s = sky.sun_direction(30, elevation)
    D = _sky_dirs()
    D = D[np.degrees(np.arccos(np.clip(D @ s, -1, 1))) > 10]          # (the sun's own glare aside)
    rel = _lum(sky.radiance(D, s)) / _lum(sky.radiance(ZENITH, s))[0]
    ref = _cie_clear(D, s)
    lr, lc = np.log(rel), np.log(ref)
    assert np.corrcoef(lr, lc)[0, 1] > 0.88                          # bright round the sun and at the horizon, darkest
    assert np.sqrt(np.mean((lr - lc) ** 2)) < 0.32                  # opposite the sun, as the CIE's has it


def test_the_zenith_and_the_diffuse_share_are_what_clear_skies_measure():
    s = sky.sun_direction(0, 60)
    Lz = _lum(sky.radiance(ZENITH, s))[0] * SUN_LUX / sky.LUMA.sum()
    assert 2500 < Lz < 7000                                          # cd/m2: a clear sky's zenith, the sun at 60°
    shares = []
    for el in (60, 30, 15, 4):
        s = sky.sun_direction(0, el)
        dif = math.pi * _lum(sky.ambient(s))
        shares.append(dif / (dif + _lum(sky.sun_irradiance(s)) * s[1]))
    assert 0.1 < shares[0] < 0.25                                    # the sky a sixth of the light at noon,
    assert all(a < b for a, b in zip(shares, shares[1:]))            # more as the sun sinks,
    assert shares[-1] > 0.6                                          # most of it at sunrise


def test_the_sun_yellows_and_reddens_as_it_sets_and_goes_out_below_the_horizon():
    ratios = []
    for el in (60, 20, 5, 1):
        e = sky.sun_irradiance(sky.sun_direction(0, el))
        assert (e > 0).all() and (e <= 1).all()
        ratios.append(e[0] / e[2])
    assert all(a < b for a, b in zip(ratios, ratios[1:])) and ratios[0] < 1.5 and ratios[-1] > 4
    assert np.all(sky.sun_irradiance(sky.sun_direction(0, -2)) == 0)
    assert _lum(sky.ambient(sky.sun_direction(0, -10))) < 1e-3 * _lum(sky.ambient(sky.sun_direction(0, 30)))


def test_haze_whitens_the_sky_and_dims_the_sun():
    s = sky.sun_direction(0, 45)
    blue, suns = [], []
    for h in (0.03, 0.1, 0.6):
        a = sky.Air(haze=h)
        z = sky.radiance(ZENITH, s, a)[0]
        blue.append(z[2] / z[0])
        suns.append(_lum(sky.sun_irradiance(s, a)))
    assert blue[0] > blue[1] > blue[2] > 1.0                         # a deep blue in clear air, paler in haze
    assert suns[0] > suns[1] > suns[2]


def test_the_hdri_is_brightest_round_the_sun_where_the_hdri_convention_puts_it():
    s = sky.sun_direction(40, 20)
    img = sky.image(s, sky.Air(), 128)
    assert img.shape == (64, 128, 3) and np.isfinite(img).all() and (img >= 0).all()
    i, j = np.unravel_index(np.argmax(_lum(img)), img.shape[:2])
    T, P = (i + 0.5) / 64 * math.pi, ((j + 0.5) / 128 - 0.5) * 2 * math.pi
    d = np.array([math.sin(T) * math.sin(P), math.cos(T), -math.sin(T) * math.cos(P)])   # (io/hdri.py, lume env_table)
    assert math.degrees(math.acos(min(1.0, d @ s))) < 6
    assert _lum(img[:32]).mean() > _lum(img[32:]).mean()             # the sky over the ground below the horizon


def test_the_scene_takes_its_light_from_the_physical_sky():
    sc = presets.make('vase_drop', fps=24)
    l = sc.data['lighting']
    l.update(sun_on=True, sun_intensity=6.0, environment='', sun_elevation=8.0)
    assert sky.of_scene(sc, sc.start) is None                       # Sky: Ambient colour, as before
    plain = sc.look(sc.start)
    l['sky'] = 'physical'
    key, amb, sun, img, k = sky.of_scene(sc, sc.start)
    assert k == 6.0 and img.ndim == 3
    low = sc.look(sc.start)
    assert low.sun_color != plain.sun_color and low.sun_color[0] > low.sun_color[2]       # a low sun: warm
    assert np.allclose(low.ambient, np.asarray(amb) * 6.0 * sc.v('lighting', 'ambient_intensity', sc.start))
    l['sun_elevation'] = 60.0
    high = sc.look(sc.start)
    assert sum(high.ambient) > sum(low.ambient) and high.sun_color[2] > low.sun_color[2]
    wl = sc.water_look(sc.start)
    assert wl.sky_image is not None and not wl.env_sun and wl.env_strength == 6.0
    l['environment'] = 'some.hdr'                                    # an HDRI file wins
    assert sky.of_scene(sc, sc.start) is None


def _render(eng, sc, f, **lume):
    sc.data['lume'].update(lume)
    eng.render(sc, f, (160, 90), mode='composite', final=True, samples=2, motion_blur=False)
    return eng.gpu.read(eng.stage.tex).astype(np.float32)[..., :3]


@pytest.mark.parametrize('lume', [dict(engine='classic'), dict(engine='lume', samples=16, bounces=3, denoise=True)])
def test_a_set_under_the_physical_sky_renders_and_its_light_follows_the_sun(engine, lume):
    sc = presets.make('vase_drop', fps=24)
    sc.data['render']['width'], sc.data['render']['height'] = 160, 90
    sc.data['lighting'].update(sky='physical', sun_on=True, sun_intensity=8.0, environment='')
    sc.data['composite']['backdrop'] = 'stage'
    f = sc.start + 2
    engine.prepare(sc, final=True)
    engine.simulate_to(sc, f, cache=False)
    out = {}
    for el in (50.0, 3.0):
        sc.data['lighting']['sun_elevation'] = el
        img = _render(engine, sc, f, **lume)
        assert np.isfinite(img).all() and (img >= 0).all()
        assert engine.stage._env_key[0] == 'physical-sky'            # the sky behind the set is the physical one
        out[el] = img
    hi, lo = out[50.0].reshape(-1, 3).mean(0), out[3.0].reshape(-1, 3).mean(0)
    assert _lum(hi) > 1.5 * _lum(lo)                                 # the set dims as the sun sets
    assert lo[0] / lo[2] > hi[0] / hi[2]                             # and warms
