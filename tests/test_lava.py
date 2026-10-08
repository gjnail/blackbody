"""GPU tests for molten liquids: the crust field carried along by the flow (liq_crust_adv.wgsl), the crust and
glow drawn on the surface (liq_lava_shade.wgsl), and the light the glow casts on the footage around it."""
import numpy as np

from blackbody.engine.liquid import LiquidParams, LiquidSolver, crust_restarts, source
from blackbody.engine.liquid_render import lava_table
from blackbody.scene import presets


def _scene(res=48, w=320, h=180, preroll=0.0):
    sc = presets.make('lava')
    sc.data['domain']['resolution'] = res
    sc.data['domain']['preroll'] = preroll
    sc.data['render']['width'], sc.data['render']['height'] = w, h
    return sc


def test_blackbody_table_follows_planck():
    t = np.array(lava_table(500.0, 50.0, 32))
    assert t.shape == (32, 4)
    T = 500.0 + 50.0 * np.arange(32)
    i1300 = int(np.argmin(abs(T - 1300.0)))
    assert abs(t[i1300, 3]) < 1e-6, 'luminance is relative to 1300 K'
    assert np.all(np.diff(t[:, 3]) > 0), 'hotter is brighter'
    # 1100 K is a few percent of 1300 K, 900 K well under a thousandth: cooled crust goes black fast
    assert 0.01 < 10 ** t[int(np.argmin(abs(T - 1100.0))), 3] < 0.08
    assert 10 ** t[int(np.argmin(abs(T - 900.0))), 3] < 1e-3
    # red-orange: red over green over blue, reddest when coolest
    assert np.all(t[:, 0] >= t[:, 1]) and np.all(t[:, 1] >= t[:, 2])
    assert t[4, 1] / t[4, 0] < t[20, 1] / t[20, 0]


def test_crust_moments_are_irregular_and_repeatable():
    a, b = crust_restarts(0.0)
    assert a == 0.0 and 0.3 <= b <= 0.8
    gaps = []
    t = 0.0
    for _ in range(12):
        last, nxt = crust_restarts(t)
        gaps.append(nxt - last)
        t = nxt + 1e-6
    assert min(gaps) >= 0.3 and max(gaps) <= 0.8 and len(set(round(g, 3) for g in gaps)) > 3
    assert crust_restarts(5.0) == crust_restarts(5.0)


def test_the_crust_rides_the_flow(engine):
    s = LiquidSolver(engine.gpu)
    dims, h = LiquidSolver.dims_for((1.6, 0.3, 0.8), 64)
    origin = (-dims[0] * h / 2, 0.0, -dims[2] * h / 2)
    s.configure(dims, h, origin, LiquidSolver.capacity_for(dims, 8, 4_000_000), 0)
    prm = LiquidParams(viscosity=50.0, rho=2600.0, flip=0.2, whitewater=False, cooling=0.5, solidify=100.0,
                       surface_tension=0.0)
    vent = source('box', (-0.6, 0.05, 0.0), (0.05, 0.05, 0.12), vel=(0.5, 0.0, 0.0))
    for f in range(45):
        n = s.substeps_for(1 / 30, cfl=1.5, lo=1, hi=8)
        with s.gpu.batch() as b:
            for _ in range(n):
                s.step(b, 1 / 30 / n, prm, [vent])
            s.pack(b)
        s.measure()
    field = s.read_crust()
    assert field is not None and field.shape == (dims[2], dims[1], dims[0], 4)
    assert np.isfinite(field).all()
    dens = engine.gpu.read(s.DENS)[..., 0]
    liquid = dens > 0.3 * 8
    surf = liquid & ~np.roll(liquid, -1, axis=1)          # the top cell of each column of liquid
    d, age = field[..., :3], field[..., 3]
    formed = surf & (np.abs(d).sum(-1) > 1e-3)
    assert formed.sum() > 20, 'crust has formed on the surface'
    # carried along by the flow (+x): each bit of crust started where it formed, upstream of where it is now
    assert np.median(d[formed][:, 0]) < -0.5
    # and it is smooth: neighbouring cells of formed crust agree, apart from the seams between batches
    dx = np.abs(np.diff(d[..., 0], axis=2))
    both = formed[..., 1:] & formed[..., :-1]
    assert np.median(dx[both]) < 1.2
    # the skin is older downstream, fresh at the vent
    ix = np.arange(dims[0])[None, None, :] * np.ones_like(age)
    near = surf & (ix < dims[0] * 0.3)
    far = surf & (ix > dims[0] * 0.5)
    if near.any() and far.any():
        assert np.median(age[far]) > np.median(age[near])


def test_lava_is_opaque_and_glows_through_cracks(engine):
    sc = _scene(res=96)
    engine.invalidate()
    engine.prepare(sc, final=False)
    f = sc.start + 48
    engine.simulate_to(sc, f, cache=False)
    plate = np.full((180, 320, 4), 60, np.uint8)
    engine.render(sc, f, (320, 180), plate=plate)
    beauty = engine.aovs()['beauty'].astype(np.float32)
    assert np.isfinite(beauty).all()
    lava = beauty[..., 3] > 0.99
    assert lava.sum() > 200, 'the flow is in the shot'
    rgb = beauty[..., :3][lava]
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    # glowing cracks, pores and fresh melt, and dark crust between them: a wide spread of brightness (a skin
    # tears only where it is pulled apart hard, so at this coarse grid the glow is mostly pores and thin skin)
    assert np.percentile(lum, 98) > 12 * max(np.percentile(lum, 20), 1e-4)
    hot = rgb[lum > np.percentile(lum, 95)]
    assert (hot[:, 0] > hot[:, 1]).all() and (hot[:, 1] >= hot[:, 2]).all(), 'the glow is orange'


def test_the_glow_lights_the_ground_around_it(engine):
    plate = np.full((180, 320, 4), 90, np.uint8)
    out = {}
    for light in (0.0, 1.0):
        sc = _scene(res=96)
        sc.data['water']['lava_light'] = light
        engine.invalidate()
        engine.prepare(sc, final=False)
        f = sc.start + 36
        engine.simulate_to(sc, f, cache=False)
        engine.render(sc, f, (320, 180), plate=plate)
        out[light] = engine.display_image()[..., :3].astype(np.float32)
    diff = out[1.0] - out[0.0]
    assert diff.max() > 10, 'the footage near the lava is lit by it'
    lit = diff.sum(-1) > 6
    assert diff[lit][:, 0].mean() > diff[lit][:, 2].mean(), 'with orange light'


def test_the_crust_survives_the_cache(engine):
    sc = _scene(res=48)
    engine.invalidate()
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 30, cache=True)
    entry = engine.cache.get(sc.start + 20)
    assert entry is not None, 'the frame should be in the frame cache'
    assert np.ndim(entry['crust']) == 4 and entry['crust'].shape[-1] == 4
    engine.render(sc, sc.start + 20, (160, 90))      # a cached frame, drawn from its own crust field
    assert np.isfinite(engine.aovs()['beauty'].astype(np.float32)).all()
