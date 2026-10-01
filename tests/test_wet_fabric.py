"""Wet fabric (engine/cloth.py, cloth_wick/cloth_drip/cloth_drops/cloth_steam/cloth_rad*.wgsl): water held in
the cloth wicks and drains to the hem, drips off it as drops that fall and land, wet cloth is heavier,
and in the heat (the fire's gas and its radiation) it steams, holds at the boil and dries before it can
catch."""
from types import SimpleNamespace

import numpy as np
import pytest

from blackbody.engine import cloth as C
from blackbody.scene import presets

LOOK = SimpleNamespace(ambient_k=293.0, flame_k=1700.0, max_k=2000.0)
HANG = C.FabricPlace(pos=(0.0, 1.5, 0.0))


@pytest.fixture(scope='module')
def gpu(engine):
    return engine.gpu


def _cloth(gpu, spec, place=HANG):
    cl = C.Cloth(gpu)
    assert cl.configure([spec])
    cl.place([place])
    return cl


def _run(cl, gpu, seconds, place=HANG):
    chunk = 20
    with gpu.batch() as b:
        for _ in range(max(1, int(seconds * C.STEPS_PER_SECOND / chunk))):
            cl.step(b, None, chunk / C.STEPS_PER_SECOND, [place], None, LOOK, [], None, steps=chunk)


def _rd(cl, name, rows=None):
    a = np.frombuffer(cl.gpu.read_buffer(cl.bufs[name]), np.float32).reshape(-1, 4)
    return a[:rows or cl.built.n].copy()


def _water_kg(cl):
    """kg of water the cloth holds, and the per-vertex wetness."""
    m = C.MATERIALS[cl.specs[0].material]
    p, nrm = _rd(cl, 'P'), _rd(cl, 'N')
    return float((p[:, 3] * m.absorb * m.density * nrm[:, 3]).sum()), p[:, 3]


def _drops(cl):
    d = np.frombuffer(cl.gpu.read_buffer(cl.bufs['DR']), np.float32)
    made = int(d[:1].view(np.uint32)[0])
    live = d[4:].reshape(-1, 8)
    return made, live[live[:, 3] > 0.0]


def test_wet_materials_hold_water_as_real_fabrics_do():
    m = C.MATERIALS
    # dripping wet, cotton holds about twice its weight and keeps about half that once drained; synthetics little
    assert 1.6 <= m['cotton'].absorb <= 2.4 and 0.4 <= m['cotton'].retain <= 0.65
    assert m['polyester'].absorb < 0.5 * m['cotton'].absorb and m['polyester'].retain < m['cotton'].retain
    assert m['wool'].wick < m['cotton'].wick and m['nylon'].wick < m['cotton'].wick
    for k, v in m.items():
        assert 1e-7 < v.wick < 1e-4 and 0.0 < v.retain < 1.0 and v.absorb > 0.0, k


def test_a_soaked_hanging_cloth_drains_to_its_hem_and_drips(gpu):
    """Hung up dripping wet, free water runs down to the hem and drips off it; the top drains to what the
    fabric keeps, not to dry; the drops fall and land on the floor."""
    cl = _cloth(gpu, C.FabricSpec(width=0.5, height=0.7, material='cotton', detail=24, burnable=False, wetness=1.0))
    w0, _ = _water_kg(cl)
    _run(cl, gpu, 0.5)
    made0, live0 = _drops(cl)
    assert made0 > 10, 'it drips at once'
    assert len(live0) > 0 and (live0[:, 5] < -0.5).any(), 'the drops are falling'
    _run(cl, gpu, 5.5)
    w1, wet = _water_kg(cl)
    x = _rd(cl, 'X')
    made1, live1 = _drops(cl)
    assert np.isfinite(x).all() and np.isfinite(wet).all()
    assert made1 > made0 + 50, 'and keeps dripping'
    assert 0.6 * w0 < w1 < 0.97 * w0, f'it has lost some of its water, not all ({w0 * 1000:.0f} -> {w1 * 1000:.0f} g)'
    y = x[:, 1]
    top, hem = wet[y > np.percentile(y, 85)].mean(), wet[y < np.percentile(y, 15)].mean()
    keep = C.MATERIALS['cotton'].retain
    assert hem > top + 0.08, f'the hem is wetter than the top ({hem:.2f} vs {top:.2f})'
    assert top > 0.8 * keep, 'and the top stays damp'
    assert (live1[:, 1] > -0.01).all(), 'no drop falls through the floor'
    # the water that left the cloth left as drops (each 2-2.8 mm across its radius)
    r = 0.0024
    assert abs((made1 * 1000.0 * 4.18879 * r ** 3) - (w0 - w1)) < 0.5 * (w0 - w1)


def test_water_wicks_slowly_into_dry_cloth_lying_flat(gpu):
    """Lying flat, there is no running down: water spreads by wicking alone, a wet front creeping a few
    centimetres in seconds (as the square root of time), and none is lost."""
    spec = C.FabricSpec(width=0.4, height=0.4, material='cotton', detail=40, orientation='lying', pins='none',
                        burnable=False)
    flat = C.FabricPlace(pos=(0.0, 0.002, 0.0))
    cl = _cloth(gpu, spec, flat)
    p = _rd(cl, 'P')
    x0 = p[:, 0].copy()
    wet0 = x0 < x0.min() + 0.06          # a 6 cm strip along one edge, soaked
    p[wet0, 3] = 1.0
    gpu.write_buffer(cl.bufs['P'], np.ascontiguousarray(p))
    cl._reset_water(p[:, 3])
    cl._may_be_wet = True
    w0, _ = _water_kg(cl)
    _run(cl, gpu, 2.0, flat)
    _, wa = _water_kg(cl)
    _run(cl, gpu, 6.0, flat)
    w1, wb = _water_kg(cl)
    # how far the water has reached past the strip: the wetness beyond it, per row of the cloth (m)
    rows = len(np.unique(np.round(p[:, 2], 4)))
    dx = 0.4 / 40
    past_a = float(wa[~wet0].sum()) * dx / rows
    past_b = float(wb[~wet0].sum()) * dx / rows
    far = x0[wb > 0.02].max() - (x0.min() + 0.06)
    assert 0.002 < past_a < past_b, f'the water wicks out ({past_a * 100:.2f} cm, then {past_b * 100:.2f} cm)'
    assert 1.4 < past_b / past_a < 2.6, f'as the square root of time ({past_b / past_a:.2f}x in 4x the time)'
    assert far < 0.06, f'a few centimetres in seconds ({far * 100:.1f} cm)'
    assert abs(w1 - w0) < 0.01 * w0, 'and no water is lost'


def test_wet_cloth_hangs_heavier(gpu):
    """A soaked sheet held at its two top corners sags lower between them than the same sheet dry."""
    def sag(wetness):
        cl = _cloth(gpu, C.FabricSpec(width=1.0, height=0.6, material='cotton', detail=24, pins='top_corners',
                                       burnable=False, wetness=wetness))
        _run(cl, gpu, 2.5)
        x = _rd(cl, 'X')
        r = _rd(cl, 'R')
        top = r[:, 1] > r[:, 1].max() - 1e-4
        return float(1.5 + 0.3 - x[top, 1].min())   # how far its top edge hangs below its pins
    assert sag(1.0) > sag(0.0) + 0.002


def _towel_scene(wetness, res=48, x=0.0):
    sc = presets.make('campfire')
    sc.data['domain'].update(resolution=res, preroll=0.0)
    if 'puffing' in sc.data['motion']:
        sc.data['motion']['puffing'] = 0.0
    sc.data['embers']['enabled'] = False
    sc.add_fabric(position=(x, 1.1, 0.0), width=0.5, height=0.7, material='cotton', pins='top', detail=24,
                  wetness=wetness)
    return sc


def test_a_soaked_towel_in_the_flames_steams_and_holds_at_the_boil(engine):
    """Hung in a campfire's flames, a soaked towel holds at 100 C while the heat boils its water off as steam
    into the gas (the vapour the gas carries, which cools it), and it does not catch while it is wet."""
    sc = _towel_scene(1.0)
    engine.invalidate()
    engine.prepare(sc)
    engine.simulate_to(sc, sc.start + 72, cache=False)
    st = engine.cloth.state()
    w = _rd(engine.cloth, 'P')[:, 3]
    assert st[:, 0].max() < 374.0, 'held at the boil'
    assert not (st[:, 2] > 0.5).any(), 'not caught'
    assert w.min() < 0.9, 'and drying'
    aux = engine.gpu.read(engine.solver.aux[0]).astype(np.float32)
    assert aux[..., 1].max() > 2.0, f'steam in the gas: {aux[..., 1].max():.2f} g/m^3'


def test_the_fires_radiation_heats_a_towel_beside_it_and_water_holds_a_wet_one_cool(engine):
    """Beside a campfire, out of its flames, a towel is heated by the fire's radiation: dry, the side toward
    the fire scorches hot while its far side stays cool; soaked, its water holds it at or below 100 C."""
    def run(wetness):
        sc = _towel_scene(wetness, x=0.6)
        engine.invalidate()
        engine.prepare(sc)
        engine.simulate_to(sc, sc.start + 96, cache=False)
        q = _rd(engine.cloth, 'QR')[:, 0]
        st = engine.cloth.state()
        hot, cold = q > np.percentile(q, 80), q < np.percentile(q, 20)
        return q, st[:, 0], hot, cold, st
    q, T, hot, cold, st = run(0.0)
    assert q.max() > 5000.0, f'the fire radiates on it ({q.max():.0f} W/m^2)'
    assert T[hot].mean() > T[cold].mean() + 60.0, 'dry, the side toward the fire heats'
    assert T.max() > 420.0, 'and toasts'
    qw, Tw, hotw, _, stw = run(1.0)
    assert Tw.max() < 373.5, 'soaked, it stays at or below the boil'
    assert Tw[hotw].mean() < T[hot].mean() - 40.0, 'far cooler than the dry one'
    assert not (stw[:, 2] > 0.5).any()
