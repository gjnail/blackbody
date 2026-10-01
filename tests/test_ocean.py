"""The sea (engine/ocean.py, engine/fft.py) and the open water round the box (engine/ocean_layer.py)."""
import math

import numpy as np
import pytest

from blackbody.engine import ocean
from blackbody.engine.ocean import OceanSpec


@pytest.fixture(scope='module')
def gpu():
    try:
        from blackbody.engine.gpu import GPU
        return GPU()
    except Exception as ex:  # no GPU on this machine
        pytest.skip(f'GPU unavailable: {ex}')


SEA = OceanSpec(height=0.35, length=5.0, direction=60.0, spread=0.35, chop=0.8, wind=5.0, depth=1.3, cap_depth=1.3)


# -- the spectrum -------------------------------------------------------------------------------------

def test_the_sea_is_as_tall_as_asked():
    for spec in (SEA, OceanSpec(height=2.0, length=40.0, wind=14.0, depth=200.0),
                 OceanSpec(height=0.8, length=20.0, swell=1.2, swell_length=90.0, wind=8.0, depth=50.0)):
        sp = ocean.spectrum(spec)
        var = sum(float((2.0 * np.abs(sp['h0'][c]) ** 2).sum()) for c in range(ocean.CASCADES))
        asked = math.hypot(spec.height, spec.swell)
        assert abs(4.0 * math.sqrt(var) - asked) < 0.1 * asked


def test_waves_are_held_within_the_water_depth():
    sp = ocean.spectrum(OceanSpec(height=3.0, length=30.0, depth=2.0, cap_depth=2.0))
    var = sum(float((2.0 * np.abs(sp['h0'][c]) ** 2).sum()) for c in range(ocean.CASCADES))
    assert 4.0 * math.sqrt(var) <= 0.45 * 2.0 * 1.05


def test_wind_roughens_the_sea_like_the_real_one():
    """The mean square slope grows with the wind (Cox and Munk: 0.003 + 0.00512 U), and without
    wind the short waves are gone (glassy water)."""
    def mss(wind):
        sp = ocean.spectrum(OceanSpec(height=1.0, length=40.0, wind=wind, depth=100.0))
        return sum(sp['var']) + sp['tail']
    calm, breeze, gale = mss(0.0), mss(6.0), mss(15.0)
    assert calm < breeze < gale
    assert 0.5 * (0.003 + 0.00512 * 15.0) < gale < 2.5 * (0.003 + 0.00512 * 15.0)
    assert ocean.spectrum(OceanSpec(height=1.0, length=40.0, wind=0.0, depth=100.0))['tail'] == 0.0


def test_the_cascades_reach_down_to_ripples():
    spec = OceanSpec(height=1.1, length=22.0, swell=0.7, swell_length=90.0, wind=10.0, depth=200.0)
    t = spec.tiles
    assert len(t) == ocean.CASCADES and all(a > b for a, b in zip(t, t[1:]))
    assert t[-1] / (spec.n / 2) < 0.03     # the finest waves drawn are a few centimetres long


# -- the GPU sea --------------------------------------------------------------------------------------

def test_fft_matches_numpy(gpu):
    from blackbody.engine.fft import FFT2D
    f = FFT2D(gpu, 64, 2)
    x = np.random.default_rng(3).standard_normal((2, 64, 64, 4)).astype(np.float32)
    gpu.upload(f.a, x)
    with gpu.batch() as b:
        f.run(b)
    y = gpu.read(f.a)
    for l in range(2):
        ref = np.fft.ifft2(x[l, ..., 0] + 1j * x[l, ..., 1]) * 64 * 64
        got = y[l, ..., 0] + 1j * y[l, ..., 1]
        assert np.abs(got - ref).max() < 1e-4 * np.abs(ref).max()
    f.destroy()


def test_the_rendered_sea_matches_the_sum_of_its_waves(gpu):
    S = ocean.Sea(gpu)
    t, cur = 3.7, (0.3, -0.2)
    with gpu.batch() as b:
        S.update(b, SEA, t, 0.0455, current=cur, wind=(2.5, 4.3))
    W = gpu.read(S.W).astype(np.float64)
    rng = np.random.default_rng(0)
    for c in range(ocean.CASCADES):
        L = S.info['tiles'][c]
        ij = rng.integers(0, S.n, (24, 2))
        x, z = (ij[:, 0] + 0.5) * L / S.n, (ij[:, 1] + 0.5) * L / S.n
        ref = ocean.surface_at(SEA, x, z, t, current=cur, layers=[c])
        got = W[2 * c, ij[:, 1], ij[:, 0], :3].T
        assert np.abs(got - ref).max() < 3e-3 * max(np.abs(ref).max(), 1e-6)
    S.release()


def test_the_simulation_sees_every_wave_its_grid_carries(gpu):
    from blackbody.engine.gpu import Uniforms
    B = ocean.SeaBoundary(gpu)
    h, dims, org = 0.0455, (48, 16, 40), (-1.0, 0.0, -0.8)
    ocn = gpu.texture2d(dims[0], dims[2], 'rgba32float', 'test-ocn')
    gb = Uniforms().v4(*dims, h).v4(*org, 0.0).v4(1, 1, 0, 0.01).tobytes()
    t = 2.3
    with gpu.batch() as b:
        B.write(b, ocn, gb, dims, 1.3 / h, SEA, t, (0.0, 0.0), h, org)
    got = (gpu.read(ocn)[..., 0].astype(np.float64) - 1.3 / h) * h
    lo0, lom = ocean.low_parts(SEA, h)
    xs = org[0] + (np.arange(dims[0]) + 0.5) * h
    zs = org[2] + (np.arange(dims[2]) + 0.5) * h
    X, Z = np.meshgrid(xs, zs)
    ref = np.zeros(X.shape)
    for j in range(ocean.LOW_CASCADES):
        kx, kz, _, _ = ocean._lattice(SEA.n, SEA.tiles[j])
        w = ocean.dispersion(np.hypot(kx, kz), SEA.depth)
        ht = lo0[j] * np.exp(-1j * w * t) + lom[j] * np.exp(1j * w * t)
        m = np.abs(ht) > 0
        ref += np.real(np.exp(1j * (np.outer(X.ravel(), kx[m]) + np.outer(Z.ravel(), kz[m]))) @ ht[m]).reshape(X.shape)
    assert np.abs(got - ref).max() < 0.01 * np.abs(ref).max()
    ocn.destroy()
    B.release()


def test_whitecaps_grow_with_the_wind():
    covers = [ocean.whitecap_cover(SEA, u) for u in (4.0, 10.0, 14.0, 20.0)]
    assert covers == sorted(covers)
    assert abs(covers[1] - 0.01) < 0.004 and abs(covers[3] - 0.10) < 0.03   # Monahan: ~1 % at 10 m/s, ~10 % at 20


def test_foam_forms_in_a_gale_and_not_in_a_breeze(gpu):
    def foam(wind):
        spec = OceanSpec(height=0.55, length=6.0, direction=60.0, spread=0.35, chop=1.0, wind=wind, depth=50.0)
        S = ocean.Sea(gpu)
        with gpu.batch() as b:
            S.update(b, spec, 40.0, 0.05, wind=(wind, 0.0))
        F = gpu.read(S.F[S._fi]).astype(np.float64)[0]
        S.release()
        return (F[..., 0] > 0.3).mean(), F[..., 1].mean()
    fresh_b, aged_b = foam(4.0)
    fresh_g, aged_g = foam(18.0)
    assert fresh_g > 5.0 * max(fresh_b, 1e-4) and aged_g > aged_b
    assert 0.005 < fresh_g < 0.4


def test_foam_is_the_same_however_the_frame_is_reached(gpu):
    """The foam is stepped on a fixed lattice: a jump (pre-roll) and stepping frame by frame agree
    on the fresh foam, and nearly on the aged foam (which remembers further back)."""
    spec = OceanSpec(height=0.55, length=6.0, chop=1.1, wind=14.0, depth=50.0)
    fp = ocean.SeaFoam(life=4.0)
    a, b = ocean.Sea(gpu), ocean.Sea(gpu)
    with gpu.batch() as bb:
        a.update(bb, spec, 30.0, 0.05, wind=(14.0, 0.0), foam=fp)
    for i in range(24):
        with gpu.batch() as bb:
            b.update(bb, spec, 29.0 + (i + 1) / 24.0, 0.05, wind=(14.0, 0.0), foam=fp)
    fa, fb = gpu.read(a.F[a._fi]).astype(np.float64)[0], gpu.read(b.F[b._fi]).astype(np.float64)[0]
    assert np.abs(fa[..., 0] - fb[..., 0]).max() < 0.02
    assert np.abs(fa[..., 1] - fb[..., 1]).mean() < 0.02
    a.release()
    b.release()


# -- the surge ------------------------------------------------------------------------------------------

def test_a_surge_travels_at_the_long_wave_speed():
    spec = OceanSpec(surge=1.0, surge_length=20.0, surge_dir=90.0, surge_time=2.0, depth=4.0)
    c = math.sqrt(9.81 * 5.0)
    assert abs(spec.surge_speed - c) < 1e-9
    x = np.linspace(-50.0, 50.0, 2001)
    for t in (0.0, 2.0, 5.0):
        eta = spec.surge_at(x, np.zeros_like(x), t)
        assert abs(x[np.argmax(eta)] - c * (t - 2.0)) < 0.1 and abs(eta.max() - 1.0) < 1e-3
    bore = OceanSpec(surge=0.5, surge_length=2.0, surge_kind='bore', surge_time=0.0, depth=1.0)
    eta = bore.surge_at(x, np.zeros_like(x), 1.0)
    assert eta[0] > 0.49 and eta[-1] < 0.01          # raised behind the front, still ahead of it


def test_the_surge_reaches_the_simulation(gpu):
    from blackbody.engine.gpu import Uniforms
    spec = OceanSpec(surge=0.4, surge_length=6.0, surge_dir=90.0, surge_time=1.0, depth=1.0)
    B = ocean.SeaBoundary(gpu)
    h, dims, org = 0.05, (80, 16, 8), (-2.0, 0.0, -0.2)
    ocn = gpu.texture2d(dims[0], dims[2], 'rgba32float', 'test-ocn-surge')
    gb = Uniforms().v4(*dims, h).v4(*org, 0.0).v4(1, 1, 0, 0.01).tobytes()
    with gpu.batch() as b:
        kp = B.write(b, ocn, gb, dims, 1.0 / h, spec, 1.0, (0.0, 0.0), h, org)
    o = gpu.read(ocn).astype(np.float64)
    eta = (o[4, :, 0] - 1.0 / h) * h
    xs = org[0] + (np.arange(dims[0]) + 0.5) * h
    assert abs(eta.max() - 0.4) < 0.02 and abs(xs[np.argmax(eta)]) < 0.1
    assert o[4, np.argmax(eta), 1] > 0.5          # its water moves with it
    assert kp < 0.01                              # and all the way down
    ocn.destroy()
    B.release()


# -- the open water round the box ------------------------------------------------------------------

def test_waves_spread_across_the_open_water_at_their_speed(gpu):
    """eWave: a bump of water spreads into rings whose energy moves out at about the group speed."""
    from blackbody.engine.gpu import Uniforms
    from blackbody.engine.mesh import MeshLibrary
    from blackbody.engine.ocean_layer import OceanLayer
    lay = OceanLayer(gpu, MeshLibrary(gpu))
    n, size = 256, 32.0
    lay._alloc(n)
    lay.corner, lay.size = (-16.0, -16.0), size
    gpu.upload(lay.DEP, np.tile(np.array([50.0, 1.0e4, 0.0, 0.0], np.float16), (n, n, 1)))
    lay.COLS = gpu.texture2d(8, 8, 'rgba32float', 'test-cols')
    gpu.upload(lay.COLS, np.full((8, 8, 4), -1.0e4, np.float32))

    class Box:
        origin, h, dims = (100.0, 0.0, 100.0), 0.05, (8, 8, 8)
    dl = size / n
    x = (np.arange(n) + 0.5) * dl - 16.0
    X, Z = np.meshgrid(x, x)
    st = np.zeros((1, n, n, 4), np.float32)
    st[0, ..., 0] = 0.05 * np.exp(-(X ** 2 + Z ** 2) / (2 * 0.3 ** 2))
    gpu.upload(lay.fft.a, st)
    dt = 1.0 / 48.0
    wu = Uniforms().raw(lay._layer_u(Box, 8.0, dt).data).v4(50.0, 9.81, 0.0, 0.0).v4(1.5, 0.5 * dl * dl / math.pi ** 2, 1.0, 0.0)
    for _ in range(96):
        with gpu.batch() as b:
            b.run(lay.k_pre, [lay.fft.a, lay.COLS, lay.DEP, lay.TMP], wu, (n, n, 1))
            b.copy_texture(lay.TMP, lay.fft.a)
            lay.fft.run(b, inverse=False)
            b.run(lay.k_prop, [lay.fft.a, lay.COLS, lay.DEP, lay.TMP], wu, (n, n, 1))
            b.copy_texture(lay.TMP, lay.fft.a)
            lay.fft.run(b, inverse=True)
    a = gpu.read(lay.fft.a)[0, ..., 0].astype(np.float64)
    e = a * a
    r = float((e * np.hypot(X, Z)).sum() / e.sum())
    cg = 0.5 * math.sqrt(9.81 / (1.0 / 0.3))       # the group speed of the bump's dominant waves
    assert 0.6 * cg * 2.0 < r < 2.0 * cg * 2.0
    assert np.isfinite(a).all()
    lay.release()


def test_waves_shoal_and_break_over_shallows():
    """The shoaling the renderer uses (ocn_sample.wgsl oc_shoal), mirrored: nothing in deep water,
    taller in shallow water, as the group speed falls (Green's law, H ~ d^-1/4, near the shore)."""
    def shoal(k0, d):
        x = k0 * d
        if x > 6.0:
            return 1.0
        k = k0 / math.sqrt(math.tanh(x))
        kd = k * d
        nn = 0.5 * (1.0 + 2.0 * kd / math.sinh(min(2.0 * kd, 40.0)))
        return math.sqrt(0.5 * math.sqrt(9.81 / k0) / (nn * math.sqrt(9.81 * math.tanh(x) / k0)))
    k0 = 2.0 * math.pi / 50.0
    assert shoal(k0, 100.0) == 1.0
    # (at middling depth waves first lose a little height, about a tenth, before they grow)
    assert shoal(k0, 1.0) > shoal(k0, 3.0) > shoal(k0, 10.0) > 0.85
    ratio = shoal(k0, 0.25) / shoal(k0, 1.0)
    assert abs(ratio - 4.0 ** 0.25) < 0.1


# -- the settings ------------------------------------------------------------------------------------

def test_every_sea_setting_exists_and_reaches_the_sea():
    from blackbody.scene import params
    keys = {p.key for p in params.SECTIONS['liquid']}
    assert set(ocean.OCEAN_KEYS) <= keys
    for k in ('sea_from', 'open_water_area', 'water_level'):
        assert k in keys
    water = {p.key for p in params.SECTIONS['water']}
    for k in ('whitecaps', 'sea_foam', 'sea_foam_life', 'foam_streaks', 'crest_glow', 'gusts', 'standin_color'):
        assert k in water
    q = {'ocean_height': 0.5, 'ocean_length': 8.0, 'swell_height': 1.0, 'swell_length': 80.0, 'swell_dir': 30.0,
         'ocean_depth': 40.0, 'ocean_detail': '512', 'surge_height': 0.7, 'surge_kind': 'bore'}
    spec = ocean.spec_from(q, 3.0, yaw=10.0)
    assert spec.swell == 1.0 and spec.depth == 40.0 and spec.n == 512 and spec.surge_kind == 'bore'
    assert abs(spec.swell_dir - 20.0) < 1e-9
    assert ocean.spec_from({'ocean_height': 0.0}, 1.0) is None


def test_the_sea_presets_load():
    from blackbody.scene import presets
    from blackbody.scene.ocean_presets import OCEAN_ORDER
    for name in OCEAN_ORDER:
        sc = presets.make(name)
        assert sc.data['domain']['kind'] == 'liquid' and sc.liquid_level(sc.start) > 0.0
        spec = ocean.spec_from({k: sc.data['liquid'].get(k) for k in ocean.OCEAN_KEYS if k in sc.data['liquid']},
                               sc.liquid_level(sc.start))
        assert spec is not None and spec.on


def test_a_tide_keyframes_the_level():
    from blackbody.scene import presets
    sc = presets.make('calm_lake')
    s = sc.start
    sc.set_key(('liquid', 'water_level'), s, 0.9)
    sc.set_key(('liquid', 'water_level'), s + 48, 1.1)
    assert abs(sc.liquid_level(s) - 0.9) < 1e-6 and abs(sc.liquid_level(s + 48) - 1.1) < 1e-6
    assert sc.liquid_params(s + 24).water_level > 0.9
    sc.set_key(('liquid', 'water_level'), s + 96, 0.0)
    assert sc.liquid_level(s + 96) > 0.0          # a tide never takes the open water away mid-shot


def test_a_tsunami_draws_the_sea_back_before_its_front():
    spec = OceanSpec(surge=4.0, surge_length=20.0, surge_dir=90.0, surge_time=0.0, surge_kind='tsunami', depth=6.0)
    x = np.linspace(-200.0, 200.0, 4001)
    eta = spec.surge_at(x, np.zeros_like(x), 0.0)
    assert eta[x < -40.0].min() > 3.9                    # the flood behind its front
    i = np.argmin(eta)
    assert abs(x[i] - 60.0) < 8.0 and eta[i] < -2.0      # the sea drawn back, three lengths ahead of it


def test_a_wave_flume_lets_the_sea_in_where_its_waves_come_from():
    from dataclasses import replace
    from types import SimpleNamespace
    from blackbody.engine.liquid_engine import LiquidEngine

    def scene(current=0.0):
        return SimpleNamespace(data={'liquid': {'sea_from': 'upwave'}}, start=1,
                               liquid_current=lambda f: (current, 0.0, 0.0))

    waves = OceanSpec(swell=1.0, swell_length=60.0, swell_dir=90.0, depth=4.0)
    # in at -x; walls all the way up along the waves (-1, mirrored in the render) and past the shore (-2)
    assert LiquidEngine._sea_sides(scene(), waves) == (1.0, -2.0, -1.0, -1.0)
    # with a surge the flood pours on over the side past the shore (a wall under the level only)
    assert LiquidEngine._sea_sides(scene(), replace(waves, surge=2.0)) == (1.0, 0.0, -1.0, -1.0)
    # a river under a bore flows in at one end and out at the other
    assert LiquidEngine._sea_sides(scene(-0.6), replace(waves, surge=0.5)) == (1.0, 1.0, -1.0, -1.0)


def test_a_wave_flume_keeps_the_seas_level(engine):
    """Waves let in at one side of a box (the others walls, a beach at the far end): the box's water stays
    at the sea's level (it drained, or piled up, when the side only pushed the sea's orbital motion in and
    the calming layer held back the water a swell brings in and takes out)."""
    from blackbody.scene import presets
    presets.PRESETS['_flume_test'] = {
        'name': 'Flume', 'category': 'Test', 'size': '', 'blurb': '',
        'domain': {'kind': 'liquid', 'size_x': 8.0, 'size_y': 2.0, 'size_z': 1.0, 'resolution': 96, 'preroll': 1.0,
                   'substeps_max': 12, 'cfl': 1.5},
        'liquid': {'water_level': 1.0, 'narrow_band': False, 'swell_height': 0.25, 'swell_length': 3.0,
                   'swell_dir': 90.0, 'sea_from': 'upwave', 'whitewater': False},
        'water': {}, 'lighting': {}, 'camera': {},
        'colliders': [dict(name='Beach', shape='mesh', mesh='builtin:beach.png', size=(8.0, 1.6, 2.0),
                           position=(1.5, 0.0, 0.0))],
        'emitters': [],
    }
    try:
        sc = presets.make('_flume_test')
        engine.prepare(sc, final=True)
        L = engine.liquid
        tops = []
        for f in range(1, 97):
            engine.simulate_to(sc, sc.start + f - 1, cache=False)
            if f > 24:
                pos, _ = L.read_particles()
                deep = pos[(pos[:, 0] > -3.8) & (pos[:, 0] < -2.2)]
                cols = np.floor((deep[:, 0] + 3.8) / 0.1).astype(int)
                top = np.full(16, -1.0)
                np.maximum.at(top, cols, deep[:, 1])
                tops.append(top[top > 0].mean())
        off = float(np.mean(tops)) - 1.0
        assert abs(off) < 0.06, f'the box sits {off:+.3f} m off the sea level'
    finally:
        del presets.PRESETS['_flume_test']
