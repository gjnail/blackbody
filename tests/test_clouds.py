"""Clouds (kind 'cloud', engine/cloud.py): the air column, a quiet sky that stays quiet, a storm that grows
from a warm bubble and rains, water conserved by the microphysics, and the presets."""
import numpy as np
import pytest

from blackbody.engine import atmos as A
from blackbody.engine.cloud import CloudParams, CloudSolver, base_state, parcel

STORM = dict(surface_t=27.0, surface_rh=0.62, lapse=6.8, mixed_layer=900.0, inversion_z=1100.0, inversion_dt=1.0,
             tropopause=12000.0, heat_flux=0.0, evaporation=0.0, shear=2.5, veer=40.0, wind=3.0, hail=2.5)


def test_parcel_diagnostics():
    """A hot, humid afternoon sounding: its condensation level about a kilometre up and some 2000 to 3500
    J/kg of CAPE (a severe-storm day); a cooler, drier one barely any."""
    p = parcel(CloudParams(**STORM))
    assert 800.0 < p['lcl'] < 1300.0
    assert 1500.0 < p['cape'] < 3500.0
    assert p['top'] > 10000.0
    q = parcel(CloudParams(surface_t=24.0, surface_rh=0.55, lapse=6.5))
    assert q['cape'] < 1000.0


def test_background_column():
    """Hydrostatic, stable above the mixed layer, and the mixed layer well mixed (its vapour the same all
    the way up: humidity rising toward its top, not falling, which would make it overturn by itself)."""
    prm = CloudParams(**STORM)
    arr, d = base_state(prm, 64, 250.0)
    theta, qv, z = d['theta'], d['qv'], d['z']
    above = z > 1500.0
    assert np.all(np.diff(theta[above]) > 0.0)
    ml = z < prm.mixed_layer
    assert np.ptp(qv[ml]) / qv[ml].mean() < 0.01
    # pressure falls about 12 % per km near the ground
    assert 0.85 < d['p'][4] / d['p'][0] < 0.92


def test_follow_the_storm():
    """Following the storm takes the mean wind of the lowest 6 km off every height: the shear is kept, the
    drift is gone."""
    still, d0 = base_state(CloudParams(**STORM), 64, 250.0)
    moving, d1 = base_state(CloudParams(**STORM, follow=True), 64, 250.0)
    sel = d1['z'] <= 6000.0
    for k in range(2):
        assert abs(np.average(d1['wind'][k][sel], weights=d1['rho'][sel])) < 1e-6
        assert np.allclose(np.diff(d1['wind'][k]), np.diff(d0['wind'][k]))


@pytest.fixture(scope='module')
def sky(engine):
    from blackbody.engine.mesh import MeshLibrary
    C = CloudSolver(engine.gpu, MeshLibrary(engine.gpu))
    return C


def _run(C, prm, minutes, size=(40.0, 16.0, 40.0), res=96):
    h = max(size) / res
    dims = tuple(int(round(s / h)) for s in size)
    C.configure(dims, h, (-size[0] / 2, 0.0, -size[2] / 2), 1000.0)
    C.set_colliders([])
    C.reset(prm)
    frame = 2.5
    peak = {'max_updraft': 0.0, 'cloud_top': 0.0, 'rain_peak': 0.0, 'cover': 0.0}
    while C.time < minutes * 60.0:
        n = C.substeps_for(frame, cfl=1.0, hi=12)
        with C.gpu.batch() as b:
            for _ in range(n):
                C.step(b, frame / n, prm)
        st = C.measure()
        for k in peak:
            peak[k] = max(peak[k], st[k])
    return peak


def test_capped_sky_stays_quiet(sky):
    """Under the cap, with nothing to break it, the sky stays clear."""
    peak = _run(sky, CloudParams(**STORM, bubble=0.0), 12.0)
    assert peak['cover'] == 0.0
    assert peak['max_updraft'] < 5.0


def test_bubble_grows_a_storm(sky):
    """A warm bubble breaks the cap: a cumulonimbus towers to the tropopause on an updraft of tens of m/s,
    and rains on the ground within half an hour."""
    peak = _run(sky, CloudParams(**STORM, bubble=2.5, bubble_r=8000.0), 30.0)
    assert peak['max_updraft'] > 20.0
    assert peak['cloud_top'] > 9000.0
    assert peak['rain_peak'] > 5.0


def test_microphysics_conserves_water(sky):
    """Every change of phase moves water between vapour, cloud, rain, ice, snow and hail: none is made or lost."""
    prm = CloudParams(**STORM, bubble=2.5, bubble_r=8000.0)
    _run(sky, prm, 22.0, res=64)
    C = sky
    g = C.gpu
    rho = C.base_info['rho'][None, :, None]

    def total(ta, tb):
        a = g.read(ta)
        b = g.read(tb)
        return float(((a[..., 1] + a[..., 2] + a[..., 3] + b[..., 0] + b[..., 1] + b[..., 2]) * rho).sum())
    before = total(C.A[0], C.B[0])
    with g.batch() as b:
        b.run(C._k['micro'], [C.A[0], C.B[0], C.A[1], C.B[1], C.base], C._u(2.5, prm), C.dims)
    after = total(C.A[1], C.B[1])
    assert abs(after - before) / before < 1e-5


@pytest.mark.parametrize('name', ['cumulus_day', 'thunderstorm'])
def test_cloud_presets_run(engine, name):
    from blackbody.scene import presets
    sc = presets.make(name)
    sc.data['domain']['resolution'] = 48
    sc.data['domain']['preroll'] = 2.0
    sc.data['render']['width'], sc.data['render']['height'] = 160, 90
    engine.prepare(sc)
    engine.simulate_to(sc, sc.start + 6, cache=False)
    engine.render(sc, sc.start + 6, (160, 90))
    img = engine.aovs()['beauty'].astype(np.float32)
    assert np.isfinite(img).all() and img[..., :3].mean() > 0.01
    assert engine.stats()['sky_minutes'] > 1.0


def test_lume_lights_the_clouds(engine):
    # With Lighting engine: Lume the light a cloud scatters many times is traced (cloud_lume.wgsl: its mean and flow on a
    # coarse grid), converging over its passes: finite, the cloud lit inside (the sky's and the sun's light carried in),
    # dark where there is no light, and the frame's sky as the classic engine draws it
    from blackbody.engine import cloud_render as CR
    from blackbody.scene import presets
    sc = presets.make('cumulus_day')
    sc.data['domain']['resolution'] = 48
    sc.data['domain']['preroll'] = 2.0
    sc.data['render']['width'], sc.data['render']['height'] = 160, 90
    engine.prepare(sc)
    f = sc.start + 1
    engine.simulate_to(sc, f, cache=False)
    # (a cloud put in the sky by hand: a ball of cloud water 2 km across, 2 km up, in the camera's view)
    C = engine.cloud
    A = engine.gpu.read(C.A[0]).astype(np.float32)
    zz, yy, xx = np.meshgrid(*(np.arange(d) for d in C.dims[::-1]), indexing='ij')
    c = (np.array([C.dims[0] / 2, 2000.0 / C.h, C.dims[2] / 2]))
    ball = ((xx - c[0]) ** 2 + (yy - c[1]) ** 2 + (zz - c[2]) ** 2) * C.h ** 2 < 1000.0 ** 2
    A[ball, 2] = 1.0e-3
    engine.gpu.upload(C.A[0], A)
    sc.data['lume']['engine'] = 'classic'
    engine.render(sc, f, (160, 90), final=True)
    classic = engine.aovs()['beauty'].astype(np.float32)
    sc.data['lume']['engine'] = 'lume'
    engine.render(sc, f, (160, 90), final=True)
    lume = engine.aovs()['beauty'].astype(np.float32)
    cr = engine.cloud_r
    lv = engine.gpu.read(cr.LV[0]).astype(np.float32)
    assert np.isfinite(lume).all() and np.isfinite(lv).all() and (lv[..., :3] >= 0).all()
    assert lv[..., :3].max() > 0.05                                 # light carried into the cloud
    sky = np.abs(lume - classic).mean(-1) < 1e-3
    assert 0.2 < sky.mean() < 1.0                                    # the sky drawn alike, the cloud not
    lvk, k = cr.lv_dims
    assert max(lvk) <= CR.LUME_MAX and k >= 1
