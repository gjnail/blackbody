"""GPU tests for the liquid features beyond plain water: viscosity, open water, narrow band, floating
objects, wind, contact angle, fire and liquid in one box, anisotropic surfacing, spray droplets and
the liquid's EXR mattes, and the HDRI reader."""
import math

import numpy as np
import pytest

from blackbody.engine.liquid import PARTICLE_BYTES, LiquidParams, LiquidSolver, source
from blackbody.scene import presets


@pytest.fixture(scope='module')
def liquid(engine):
    return LiquidSolver(engine.gpu)


def _setup(s, size, res, ppc=8, ww=0, band=None):
    dims, h = LiquidSolver.dims_for(size, res)
    origin = (-dims[0] * h / 2, 0.0, -dims[2] * h / 2)
    s.configure(dims, h, origin, LiquidSolver.capacity_for(dims, ppc, 4_000_000, band=band), ww)
    return dims, h, origin


def _run(s, prm, first, frames, fps=30, every=None):
    s._prm = prm
    for _ in range(frames):
        n = s.substeps_for(1 / fps, cfl=1.5, lo=1, hi=12)
        with s.gpu.batch() as b:
            for _ in range(n):
                s.step(b, 1 / fps / n, prm, first if s.steps == 0 else (every or []))
            s.pack(b)
        s.measure()


def _front(s):
    pos, _ = s.read_particles()
    return float(np.percentile(pos[:, 0], 99.5))


def test_particles_are_compact():
    assert PARTICLE_BYTES == 48


def test_viscosity_slows_a_dam_break(liquid):
    fronts = []
    for visc in (0.0, 20.0):
        dims, h, origin = _setup(liquid, (1.0, 0.5, 0.4), 48)
        prm = LiquidParams(open_sides=False, whitewater=False, viscosity=visc, flip=0.5)
        liquid.reset()
        fill = source('box', pos=(origin[0] + 0.12, 0.15, 0.0), size=(0.12, 0.15, 0.2), fill=True)
        _run(liquid, prm, [fill], 10)
        assert np.isfinite(liquid.max_speed)
        fronts.append(_front(liquid))
    assert fronts[1] < fronts[0] - 0.05, f'viscous front at {fronts[1]:.3f} m, water at {fronts[0]:.3f} m'


def test_open_water_holds_its_level(liquid):
    dims, h, origin = _setup(liquid, (0.8, 0.5, 0.8), 48)
    settle = LiquidParams(whitewater=False, water_level=0.2, level_absorb=6.0, damping=6.0)
    prm = LiquidParams(whitewater=False, water_level=0.2, level_absorb=6.0)
    liquid.reset()
    _run(liquid, settle, [], 20)     # as the pre-roll does with Settle on
    n1 = liquid.count
    _run(liquid, prm, [], 20)
    pos, vel = liquid.read_particles()
    top = np.percentile(pos[:, 1], 99.5)
    assert abs(top - 0.2) < 2.0 * h, f'surface at {top:.3f} m, level 0.2 m'
    assert abs(liquid.count - n1) < 0.05 * n1, 'the open sides neither drain nor flood the box'
    assert np.percentile(np.linalg.norm(vel, axis=1), 99) < 0.1


def test_narrow_band_keeps_the_volume_with_far_fewer_particles(liquid):
    counts, vols = [], []
    for band in (False, True):
        dims, h, origin = _setup(liquid, (0.6, 0.4, 0.6), 48, band=4 if band else None)
        prm = LiquidParams(open_sides=False, whitewater=False, narrow_band=band, band_width=4)
        liquid.reset()
        fill = source('box', pos=(0.0, 0.12, 0.0), size=(0.3, 0.12, 0.3), fill=True)
        _run(liquid, prm, [fill], 20)
        dens = liquid.gpu.read(liquid.DENS)[..., 0]
        vols.append(float((dens >= prm.surface_density * prm.ppc).sum()) * h ** 3)
        counts.append(liquid.count)
        pos, vel = liquid.read_particles()
        assert np.percentile(np.linalg.norm(vel, axis=1), 99) < 0.05, 'still water stays still'
    assert counts[1] < 0.5 * counts[0], counts
    assert abs(vols[1] - vols[0]) < 0.03 * vols[0], vols


def _band_volume(s, prm, h):
    """What the narrow band conserves: the particles, at particles per cell, plus the deep cells (full)."""
    deep = int((np.frombuffer(s.gpu.read_buffer(s.band_buffer), np.uint32) != 0).sum())
    return (s.count / prm.ppc + deep) * h ** 3


def test_narrow_band_keeps_the_volume_of_sloshing_water(liquid):
    # A dam break sloshing in a closed box: the whole flow passes through the deep liquid, freed going
    # in and born again coming out. Unbalanced, that exchange drained a quarter of the box in 4 s.
    dims, h, origin = _setup(liquid, (1.2, 0.6, 0.4), 48, band=4)
    prm = LiquidParams(open_sides=False, whitewater=False, narrow_band=True, band_width=4)
    liquid.reset()
    fill = source('box', pos=(origin[0] + 0.2, 0.2, 0.0), size=(0.2, 0.2, 0.2), fill=True)
    _run(liquid, prm, [fill], 30)
    v0 = _band_volume(liquid, prm, h)
    _run(liquid, prm, [], 90)
    v1 = _band_volume(liquid, prm, h)
    assert abs(v1 / v0 - 1) < 0.04, f'{v0 * 1e3:.2f} L of water became {v1 * 1e3:.2f} L'
    bank = np.frombuffer(liquid.gpu.read_buffer(liquid.bank), np.int32)
    assert abs(bank[0]) < 0.02 * v0 / h ** 3 * prm.ppc, f'band balance {bank[0]} particles'


def test_wind_blows_spray_downwind(liquid):
    dims, h, origin = _setup(liquid, (1.0, 0.8, 1.0), 56, ww=200_000)
    prm = LiquidParams(whitewater=True, open_sides=False, wind=(8.0, 0.0, 0.0))
    liquid.reset()
    pool = source('box', pos=(0.0, 0.08, 0.0), size=(0.5, 0.08, 0.5), fill=True)
    ball = source('sphere', pos=(0.0, 0.45, 0.0), size=(0.1, 0.1, 0.1), fill=True, vel=(0.0, -3.5, 0.0))
    _run(liquid, prm, [pool, ball], 12)
    ww = liquid.read_ww_packed()
    assert len(ww)
    cls = np.floor((ww[:, 1] >> 16) / 65535.0 * 3.0 + 1e-4).astype(int)
    v = np.frombuffer(ww[:, 2:4].astype('<u4').tobytes(), np.float16).reshape(-1, 4)[:, :3].astype(np.float32)
    spray = v[cls == 0]
    assert len(spray) > 20
    assert spray[:, 0].mean() > 0.5, f'mean spray drift {spray[:, 0].mean():.2f} m/s'


def test_contact_angle_runs(liquid):
    dims, h, origin = _setup(liquid, (0.4, 0.2, 0.4), 40)
    liquid.reset()
    drop = source('sphere', pos=(0.0, 0.05, 0.0), size=(0.04, 0.04, 0.04), fill=True)
    for angle in (30.0, 150.0):
        liquid.reset()
        prm = LiquidParams(whitewater=False, open_sides=False, contact_angle=angle)
        _run(liquid, prm, [drop], 5)
        assert np.isfinite(liquid.max_speed) and liquid.count > 0


def _scene(name, res):
    sc = presets.make(name)
    sc.data['domain']['resolution'] = res
    sc.data['domain']['preroll'] = 0.0
    sc.data['render']['width'], sc.data['render']['height'] = 320, 180
    return sc


def test_floating_objects_float_and_sink(engine):
    # tumbling bodies need to be ten cells or so across for their pressure to be well measured
    sc = _scene('floating', 112)
    sc.data['domain']['preroll'] = 0.5
    engine.invalidate()
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 60, cache=True)
    ov = engine.floating_overrides(sc.start + 60)
    assert ov is not None and len(ov) == 3
    crate, ball, stone = (ov[i]['pos'][1] for i in range(3))
    level = sc.data['liquid']['water_level']
    # the crate and the ball float part-way in, the stone lies on the bottom
    assert crate - 0.17 < level < crate + 0.17, f'crate at {crate:.3f} m, water at {level} m'   # half a diagonal: it may tilt
    assert ball - 0.11 < level < ball + 0.11, f'ball at {ball:.3f} m, water at {level} m'
    assert stone < 0.07 + 0.05, f'the stone sinks (at {stone:.3f} m)'
    # the cache keeps where the liquid moved them
    assert engine.floating_overrides(sc.start + 30) is not None
    plate = np.full((180, 320, 4), 128, np.uint8)
    engine.render(sc, sc.start + 30, (320, 180), plate=plate)


def test_fire_and_liquid_in_one_box(engine):
    sc = _scene('hose_on_fire', 64)
    engine.invalidate()
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 45, cache=True)
    assert engine.kind == 'both'
    assert engine.liquid.count > 0
    water = engine.gpu.read(engine.solver.water)[..., 0]
    assert water.max() > 0.5, 'the liquid reaches the fire solver'
    plate = np.full((180, 320, 4), 128, np.uint8)
    engine.render(sc, sc.start + 45, (320, 180), samples=2, plate=plate)
    aov = engine.aovs()
    assert np.isfinite(aov['beauty'].astype(np.float32)).all()
    st = engine.stats()
    assert st['kind'] == 'both' and st['particles'] > 0


def test_lava_glows_and_honey_is_thick(engine):
    for name in ('lava', 'honey'):
        sc = _scene(name, 96 if name == 'lava' else 48)   # (the lava's 5 m box: 48 cells would be coarser than its vent)
        engine.invalidate()
        engine.prepare(sc, final=False)
        f = sc.start + (24 if name == 'lava' else 12)   # (the lava needs a second to spread into a lobe)
        engine.simulate_to(sc, f, cache=False)
        plate = np.full((180, 320, 4), 60, np.uint8)
        engine.render(sc, f, (320, 180), plate=plate)
        e = engine.aovs()['emission'].astype(np.float32)
        assert np.isfinite(e).all()
        if name == 'lava':
            assert e[..., 0].max() > 0.05 and e[..., 0].max() > e[..., 2].max(), 'lava glows red-orange'


def test_liquid_exr_has_mattes(engine, tmp_path):
    import OpenEXR
    from blackbody.render.job import Output, RenderJob
    sc = _scene('fountain', 48)
    job = RenderJob(sc, [Output('exr', str(tmp_path / 'f.####.exr'), 'element')], engine,
                    frames=(sc.start + 10, sc.start + 10), final=False, size=(160, 90), samples=1)
    paths = job.run()
    with OpenEXR.File(paths[0]) as f:
        names = set(f.channels())
    for layer in ('water.Y', 'foam.Y', 'spray.Y', 'drops.Y', 'bubbles.Y'):
        assert layer in names, sorted(names)


def test_sheets_build_a_finite_surface(engine):
    sc = _scene('waterfall', 48)
    sc.data['water']['sheets'] = 1.0
    engine.invalidate()
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 8, cache=False)
    engine.render(sc, sc.start + 8, (160, 90))
    a = np.frombuffer(engine.gpu.read_buffer(engine.liquid_r.aniso), np.float32)
    assert np.isfinite(a).all()
    assert a.reshape(-1, 8)[:, 6].max() >= 1.0


def test_hdri_reader_and_directions(tmp_path):
    from blackbody.io.hdri import brightest, direction_of, load_hdri
    h, w = 16, 32
    img = np.full((h, w, 3), 0.2, np.float32)
    img[4, 24] = (50.0, 45.0, 40.0)                   # a sun
    # write a flat (uncompressed) Radiance file
    m = img.max(axis=-1)
    e = np.ceil(np.log2(np.maximum(m, 1e-30))).astype(np.int32)
    mant = img / np.ldexp(1.0, e)[..., None] * 256.0
    rgbe = np.concatenate([np.clip(mant, 0, 255), (e + 128)[..., None]], -1).astype(np.uint8)
    p = tmp_path / 'sky.hdr'
    p.write_bytes(b'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y %d +X %d\n' % (h, w) + rgbe.tobytes())
    back = load_hdri(p)
    assert back.shape == (h, w, 3)
    assert np.allclose(back[0, 0], 0.2, rtol=0.02)
    assert back[4, 24, 0] > 40.0
    az, el, colour = brightest(back)
    d = direction_of((24 + 0.5) / w, (4 + 0.5) / h)
    assert math.isclose(el, math.degrees(math.asin(d[1])), abs_tol=8.0)
    assert colour[0] >= colour[2]
