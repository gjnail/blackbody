"""Heat and phase changes of liquids (engine/liquid_thermal.py): the physics tables, and on the GPU water
that boils at the rate its heat allows, evaporates into dry air, cools below freezing only until it is
seeded, freezes into ice that keeps its shape and floats, and the presets that show them."""
import math

import numpy as np
import pytest

from blackbody.engine import liquid_thermal as lt
from blackbody.engine.liquid import LiquidParams, LiquidSolver, source


# -- the tables ----------------------------------------------------------------------------------

def test_enthalpy_round_trip():
    for t in (-30.0, -1.0, 0.5, 20.0, 99.0):
        frozen = t < 0.0
        h = lt.enthalpy(t, frozen=frozen)
        assert lt.temperature(h, seeded=frozen) == pytest.approx(t, abs=1e-9)
    # melting takes the latent heat at the freezing point
    assert lt.enthalpy(0.0, frozen=False) - lt.enthalpy(-1e-9, frozen=True) == pytest.approx(lt.LF, rel=1e-6)
    # supercooled water is still liquid until seeded; seeded, the same enthalpy is part ice at 0 C
    h = lt.enthalpy(-5.0, frozen=False)
    assert lt.temperature(h, seeded=False) == pytest.approx(-5.0)
    assert lt.temperature(h, seeded=True) == pytest.approx(0.0)
    assert 1.0 - h / lt.LF == pytest.approx(5.0 * lt.CW / lt.LF)    # about 6 % slush


def test_water_is_densest_at_4c():
    d = {t: lt.water_density(t) for t in (0.0, 2.0, 4.0, 6.0, 20.0, 100.0)}
    assert max(d, key=d.get) == 4.0
    assert d[20.0] == pytest.approx(998.2, abs=0.2)
    assert d[100.0] == pytest.approx(958.4, abs=0.5)
    assert d[0.0] < d[4.0] and d[0.0] > d[20.0]


def test_saturation_vapour():
    assert lt.vapour_sat(20.0) == pytest.approx(17.3, rel=0.03)
    assert lt.vapour_sat(100.0) == pytest.approx(598.0, rel=0.03)
    assert lt.vapour_sat(-10.0, over_ice=True) < lt.vapour_sat(-10.0)


def test_boiling_curve():
    # nucleate boiling rises steeply to the critical heat flux, falls through the transition regime to the
    # Leidenfrost minimum, then film boiling climbs slowly
    q = [lt.boil_flux(d) for d in (2, 5, 10, 20, 60, 109, 111, 200)]
    assert q[0] == 0.0
    assert q[1] < q[2] < q[3] == pytest.approx(1.1e6, rel=0.01)
    assert q[3] > q[4] > q[5]
    assert 1.5e4 < q[5] < 3.5e4 and 1.5e4 < q[6] < 3.5e4
    assert q[7] > q[6]


def test_evaporation_direction():
    assert lt.evaporation_flux(40.0, 10.0, 5.0) > 0.0                # warm water, dry air: evaporates
    assert lt.evaporation_flux(5.0, 25.0, 23.0) < 0.0               # cold water in wet warm air: dew
    fast = lt.evaporation_flux(40.0, 10.0, 5.0, wind=5.0)
    assert fast > 2.0 * lt.evaporation_flux(40.0, 10.0, 5.0)


# -- on the GPU -------------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def liquid(engine):
    return LiquidSolver(engine.gpu)


def _setup(s, size, res, ppc=8, ww=0):
    dims, h = LiquidSolver.dims_for(size, res)
    origin = (-dims[0] * h / 2, 0.0, -dims[2] * h / 2)
    s.configure(dims, h, origin, LiquidSolver.capacity_for(dims, ppc, 4_000_000), ww)
    return dims, h, origin


def _run(s, prm, first, frames, fps=30, cb=None):
    s._prm = prm
    for f in range(frames):
        n = s.substeps_for(1 / fps, cfl=1.5, lo=2, hi=12)
        with s.gpu.batch() as b:
            for _ in range(n):
                s.step(b, 1 / fps / n, prm, first if s.steps == 0 else [])
            s.pack(b)
        s.measure()
        if cb:
            cb(f)


def _ice_mask(s):
    ice = s.read_ice()
    return (ice & 0xFFFF) / 65535.0 > 0.5


def _ice_share(s):
    """Each live particle's frozen share, from its enthalpy (the packed ice is the renderer's: slush packs
    as water)."""
    cap = s.capacity
    parts = np.frombuffer(s.gpu.read_buffer(s.parts, cap * 48), np.float32).reshape(cap, 12)
    th = np.frombuffer(s.gpu.read_buffer(s.thermal.therm, cap * 8), np.float32).reshape(cap, 2)
    th = th[parts[:, 3] >= 0.0]
    return np.where(th[:, 1] >= 0.0, np.clip(1.0 - th[:, 0] / lt.LF, 0.0, 1.0), 0.0)


def test_boiling_takes_the_heat_it_is_given(liquid):
    """Water at its boiling point on a plate 8 K hotter boils off at the rate nucleate boiling brings heat
    in (Rohsenow, about 70 kW/m^2): mass lost = heat / latent heat."""
    dims, h, _ = _setup(liquid, (0.2, 0.2, 0.2), 32, ww=100_000)
    prm = LiquidParams(thermal=True, temp=100.0, ground_temp=108.0, air_temp=20.0, humidity=100.0, open_sides=False,
                       open_top=False)
    src = [source('box', (0.0, 0.05, 0.0), (0.1, 0.05, 0.1), fill=True, jitter=0.0, vel_blend=0.0)]
    _run(liquid, prm, src, 5)
    n0 = liquid.count
    _run(liquid, prm, [], 30)
    lost = (n0 - liquid.count) * prm.rho * h ** 3 / prm.ppc          # kg in 1 s (boiled and evaporated)
    # off the floor and the closed walls (the ground's temperature too) up to the water's depth
    want = lt.boil_flux(8.0) * (0.2 * 0.2 + 4 * 0.2 * 0.1) / 2.257e6
    assert 0.5 * want < lost < 3.0 * want
    assert liquid.ww_count > 100           # steam bubbles
    st = liquid.thermal_stats
    assert st['max_temp'] < 101.0 and st['boiling_cells'] > 0


def test_evaporation_into_dry_air(liquid):
    """A shallow pan of 60 C water in dry 20 C air gives off vapour at the rate the Lewis analogy says."""
    dims, h, _ = _setup(liquid, (0.2, 0.1, 0.2), 32)
    prm = LiquidParams(thermal=True, temp=60.0, ground_temp=60.0, air_temp=20.0, humidity=0.0, open_sides=False)
    src = [source('box', (0.0, 0.02, 0.0), (0.1, 0.02, 0.1), fill=True, jitter=0.0, vel_blend=0.0)]
    _run(liquid, prm, src, 3)
    got = []
    _run(liquid, prm, [], 15, cb=lambda f: got.append(liquid.thermal_stats['vapour_g']))
    rate = sum(got) / 0.5 * 1e-3                                     # kg/s
    want = lt.evaporation_flux(60.0, 20.0, 0.0) * 0.2 * 0.2
    assert 0.5 * want < rate < 3.0 * want


def test_supercooled_water_needs_a_seed(liquid):
    """Still water at -3 C stays liquid while it can hold that much supercooling, and freezes (in part, to
    slush) once it cannot."""
    _setup(liquid, (0.16, 0.16, 0.16), 24)
    base = dict(thermal=True, temp=-3.0, ground_temp=-3.0, air_temp=-3.0, open_sides=False, min_source_temp=-3.0)
    src = [source('box', (0.0, 0.03, 0.0), (0.06, 0.03, 0.06), fill=True, jitter=0.0, vel_blend=0.0)]
    _run(liquid, LiquidParams(supercool=10.0, **base), src, 10)
    assert liquid.thermal_stats['ice_cells'] == 0
    assert liquid.thermal_stats['min_temp'] < -2.0
    # it cannot hold that much any more: it seeds itself and flashes to slush, about 6 % ice at 0 C (the
    # latent heat of the ice warms the rest to the freezing point)
    _run(liquid, LiquidParams(supercool=2.0, **base), [], 10)
    st = liquid.thermal_stats
    assert st['min_temp'] > -1.0 and st['max_temp'] < 0.5
    ice = _ice_share(liquid)
    assert 0.03 < float(ice.mean()) < 0.12


def test_ice_cube_keeps_its_shape_and_floats(liquid):
    dims, h, _ = _setup(liquid, (0.2, 0.25, 0.2), 48)
    prm = LiquidParams(thermal=True, temp=20.0, ground_temp=20.0, air_temp=20.0, open_sides=False, min_source_temp=-10.0)
    src = [source('box', (0.0, 0.04, 0.0), (0.1, 0.04, 0.1), fill=True, jitter=0.0, vel_blend=0.0),
           source('box', (0.0, 0.15, 0.0), (0.02, 0.02, 0.02), fill=True, jitter=0.0, vel_blend=0.0, temp=-10.0)]
    _run(liquid, prm, src, 1)
    n_ice = int(_ice_mask(liquid).sum())
    assert n_ice > 1000
    spread, above = [], []

    def cb(f):
        pos, _ = liquid.read_particles()
        m = _ice_mask(liquid)
        p = pos[m]
        c = np.median(p, 0)
        spread.append(np.percentile(np.linalg.norm(p - c, axis=1), 99))
        if f >= 60:
            # the water's level away from the cube, and how much of the cube stands above it
            far = pos[~m]
            far = far[np.hypot(far[:, 0] - c[0], far[:, 2] - c[2]) > 0.05]
            level = np.percentile(far[:, 1], 97) + 0.25 * h
            top, bottom = p[:, 1].max() + 0.25 * h, p[:, 1].min() - 0.25 * h
            above.append((top - level) / (top - bottom))
    _run(liquid, prm, [], 120, cb=cb)
    # it keeps its shape: a rigid 4 cm cube has all of itself within its half-diagonal of its middle
    assert max(spread) < 0.02 * math.sqrt(3) + 1.5 * h
    # it barely melts in 4 s of 20 C water (a real cube takes many minutes)
    assert _ice_mask(liquid).sum() > 0.9 * n_ice
    # and floats with about a tenth of itself above the water (ice 917 kg/m^3: 8 %), bobbing
    assert 0.0 < float(np.mean(above)) < 0.25


def test_heat_off_changes_nothing(liquid):
    _setup(liquid, (0.16, 0.16, 0.16), 24)
    prm = LiquidParams(open_sides=False)
    src = [source('box', (0.0, 0.03, 0.0), (0.06, 0.03, 0.06), fill=True, jitter=0.0, vel_blend=0.0)]
    _run(liquid, prm, src, 2)
    assert liquid.gas_flux is None and liquid.ice_buffer is None and liquid.read_ice() is None
    assert liquid.thermal_stats == {}


PHASE = ['ice_cubes', 'ice_melt', 'pond_freeze', 'frozen_pour', 'boiling_pot', 'hot_plate', 'steaming_pool', 'boiling_throw']


@pytest.mark.parametrize('name', PHASE)
def test_phase_presets_run(engine, name):
    from blackbody.scene import presets
    sc = presets.make(name)
    sc.data['domain']['resolution'] = 48
    sc.data['domain']['preroll'] = 0.0
    sc.data['render']['width'], sc.data['render']['height'] = 160, 90
    assert sc.data['liquid']['thermal']
    engine.prepare(sc)
    engine.simulate_to(sc, sc.start + 3, cache=False)
    engine.render(sc, sc.start + 3, (160, 90))
    assert np.isfinite(engine.aovs()['beauty'].astype(np.float32)).all()
    # (the liquid is there: at this coarse a grid a thrown blob is spray, filling no cell)
    assert engine.liquid.count > 0 and 'liquid_cells' in engine.liquid.thermal_stats
