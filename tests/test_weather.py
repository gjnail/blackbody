"""Weather: the atmosphere's physics (engine/atmos.py), what the air column delivers (snow, sleet, freezing
rain, hail), and the precipitation particles on the GPU (engine/weather.py): their rate, what they build up
on the ground, and the presets."""
import math

import numpy as np
import pytest

from blackbody.engine import atmos as A


# -- physics -------------------------------------------------------------------------------------

def test_saturation_and_lapse_rates():
    assert abs(float(A.es_water(0.0)) - 611.2) < 0.5
    assert abs(float(A.es_water(20.0)) - 2339.0) < 10.0
    assert abs(float(A.es_ice(-10.0)) - 259.9) < 1.0
    # the saturated lapse rate: about 4.8 K/km in warm air, near the dry 9.8 in the cold upper air
    assert 4.5e-3 < A.moist_lapse(15.0, 1e5) < 5.1e-3
    assert 7.5e-3 < A.moist_lapse(-20.0, 6e4) < 8.5e-3


def test_fall_speeds():
    assert 6.3 < float(A.rain_speed(2.0)) < 6.8            # a 2 mm raindrop
    assert 0.9 < float(A.snow_speed(5.0)) < 1.2            # a snowflake aggregate
    assert 2.0 < float(A.graupel_speed(3.0)) < 3.2
    assert 16.0 < float(A.sphere_speed(20.0, A.RHO_I)) < 21.0   # 2 cm hail
    # thinner air aloft: everything falls faster
    assert float(A.rain_speed(2.0, 0.7)) > float(A.rain_speed(2.0, 1.2))


def _arrive(snd, kind, rate=2.0, size=None):
    return A.arrivals(snd, kind, rate, size, samples=48, dt=1.0).summary


def test_column_types():
    """The air column decides what arrives: snow when it is cold all the way down; wet snow just above
    freezing; sleet when a weak warm layer aloft half-melts it over a deep cold layer; freezing rain when a
    strong one melts it right through (the drops stay liquid below freezing: no nucleus); rain when the
    ground is mild."""
    s = _arrive(A.winter_sounding(-5.0), A.SNOW)
    assert s['snow'] > 0.99
    s = _arrive(A.winter_sounding(0.5), A.SNOW)
    assert s['snow'] > 0.9 and s['wet_snow'] > 0.1
    s = _arrive(A.winter_sounding(-6.0, 1.0, 1200.0), A.SNOW)
    assert s['ice pellets'] > 0.5
    s = _arrive(A.winter_sounding(-1.0, 5.0, 800.0), A.SNOW)
    assert s['freezing_rain'] > 0.95
    s = _arrive(A.winter_sounding(4.0), A.SNOW)
    assert s['rain'] > 0.95


def test_hail_melts_on_the_way_down():
    """From a freezing level near 4 km, small hail melts into rain; big stones arrive."""
    snd = A.convective_sounding(28.0)
    small = _arrive(snd, A.HAIL, 30.0, 8.0)
    big = _arrive(snd, A.HAIL, 30.0, 40.0)
    assert small['rain'] > big['rain']
    assert big['hail'] > 0.9


def test_dry_air_evaporates_snow():
    """Snow falling through dry air sublimates (virga); in saturated air it arrives."""
    dry = _arrive(A.winter_sounding(2.0, rh=0.5), A.SNOW)
    wet = _arrive(A.winter_sounding(-3.0, rh=1.0), A.SNOW)
    assert dry['lost'] > 0.9
    assert wet['lost'] < 0.3


# -- on the GPU ----------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def gpu(engine):
    return engine.gpu


def _run(W, prm, seconds, sub=4, fps=24):
    g = W.gpu
    with g.batch() as b:
        W.surface(b, prm)
    for _ in range(int(seconds * fps)):
        with g.batch() as b:
            for _ in range(sub):
                W.step(b, 1.0 / fps / sub, prm)
            W.pack(b)
        W.measure()


def test_snow_builds_up_at_its_rate(gpu):
    from blackbody.engine.weather import Weather, WeatherParams
    W = Weather(gpu)
    prm = WeatherParams(kind='snow', rate=3.0, ground_t=-4.0, humidity=0.95, area=(-1.5, -1.5, 1.5, 1.5), top=3.0,
                        capacity=300000, cover_cell=0.05)
    W.configure(prm)
    _run(W, prm, 6.0)
    cov = W.read_cover()
    swe = float(cov[..., 1].mean())
    want = 3.0 / 3600.0 * 6.0
    assert 0.7 * want < swe < 1.4 * want
    # fresh snow at -4 C is light: about 80 kg/m^3
    rho = swe / max(float(cov[..., 0].mean()), 1e-9)
    assert 60.0 < rho < 110.0
    assert W.stats['landed']['snow'] > 0


def test_hail_bounces_then_rests(gpu):
    from blackbody.engine.weather import Weather, WeatherParams
    W = Weather(gpu)
    prm = WeatherParams(kind='hail', rate=60.0, size=15.0, ground_t=20.0, humidity=0.8, area=(-1.0, -1.0, 1.0, 1.0),
                        top=3.0, capacity=20000, ground_temp=20.0)
    W.configure(prm)
    _run(W, prm, 3.0)
    pk = W.read_packed()
    resting = pk[:, 10] > 0.5
    assert resting.sum() > 5
    # resting stones lie on the ground, about their own radius up
    y = pk[resting, 1]
    r = pk[resting, 3] * 0.5e-3
    assert np.all(np.abs(y - r) < 0.02)


def test_freezing_rain_glazes(gpu):
    """Supercooled drops freeze where they land: the glaze grows at the rate the rain falls."""
    from blackbody.engine.weather import Weather, WeatherParams
    W = Weather(gpu)
    prm = WeatherParams(kind='freezing_rain', rate=5.0, ground_t=-2.0, humidity=0.95, area=(-1.0, -1.0, 1.0, 1.0),
                        top=3.0, capacity=300000, ground_temp=-2.0, cover_cell=0.05)
    W.configure(prm)
    _run(W, prm, 4.0)
    glaze = float(W.read_cover()[..., 3].mean())       # m of ice
    want = 5.0 / 3600.0 * 4.0 / 917.0
    assert 0.6 * want < glaze < 1.5 * want


@pytest.mark.parametrize('name', ['snow_pond', 'hail_pond'])
def test_weather_presets_run(engine, name):
    from blackbody.scene import presets
    sc = presets.make(name)
    sc.data['domain']['resolution'] = 48
    sc.data['domain']['preroll'] = 0.0
    sc.data['render']['width'], sc.data['render']['height'] = 160, 90
    engine.prepare(sc)
    engine.simulate_to(sc, sc.start + 6, cache=False)
    engine.render(sc, sc.start + 6, (160, 90))
    assert np.isfinite(engine.aovs()['beauty'].astype(np.float32)).all()
    assert engine.weather.stats['alive'] > 0
