"""The GPU's memory: what the card has (engine/gpu.py card_memory), what is made (GPU.allocated) against what was
estimated (Solver.estimate, Scene.memory_estimate), a scene fitted to the card before its grids are made
(Scene.memory_plan, with a notice) from the card's size alone, a disk cache keeping the layout it was simulated at
(Scene.cache_fit), Grow to fit held by memory and texture size, and the app making its engine again after the GPU fails
(ui/worker.py _recover): a lost device, or one out of memory. GPU-free unless a test takes the engine fixture."""
import gc
import os
import types

import numpy as np
import pytest
import wgpu

from blackbody.engine import gpu as G
from blackbody.engine.solver import Solver, SolverParams
from blackbody.io.simcache import CacheMismatch, SimCache
from blackbody.scene import model, presets
from blackbody.scene.params import applies


@pytest.fixture
def card(monkeypatch):
    """Stand in for the GPU in use (what the scene's plan reads): card(plan in GB, side), and `cut_gb`: the plan cut to
    that after running out of memory (the card's own plan stays `plan_gb`)."""
    def put(plan_gb=None, side=None, name='Test card', cut_gb=None):
        monkeypatch.setitem(G._CARD, 'card_plan', None if plan_gb is None else plan_gb * 1e9)
        monkeypatch.setitem(G._CARD, 'plan', None if plan_gb is None else (cut_gb or plan_gb) * 1e9)
        monkeypatch.setitem(G._CARD, 'side', side)
        monkeypatch.setitem(G._CARD, 'name', name)
    put()
    return put


class FakeDevice:
    def __init__(self, fail=None):
        self.fail = fail
        self.made = []

    def create_texture(self, **k):
        if self.fail:
            raise self.fail
        t = types.SimpleNamespace(create_view=lambda: 'view', destroy=lambda: None, **k)
        self.made.append(t)
        return t

    def create_buffer(self, **k):
        if self.fail:
            raise self.fail
        return types.SimpleNamespace(destroy=lambda: None, **k)


def fake_gpu(ceiling=None, plan=None, device=None):
    g = G.GPU.__new__(G.GPU)
    g.allocated, g.ceiling, g.plan, g.device, g._kernels = 0, ceiling, plan, device or FakeDevice(), {}
    return g


# -- what failed ---------------------------------------------------------------------------------------------------------

def test_a_lost_device_and_an_allocation_that_failed_are_told_apart():
    assert G.failure(wgpu.GPUValidationError('Validation Error\n\nCaused by:\n  In wgpuDeviceCreateTexture\n    Parent device is lost')) == 'lost'
    assert G.failure(wgpu.GPUValidationError('In wgpuDeviceCreateTexture\n    Not enough memory left.')) == 'memory'
    assert G.failure(wgpu.GPUOutOfMemoryError('out of memory')) == 'memory'
    assert G.failure(G.GPUOutOfMemory(1e9, 2e9, 2.5e9, 'scal0')) == 'memory'
    try:   # (a kernel that fails to compile on a lost device says so only in its cause)
        try:
            raise wgpu.GPUValidationError('Parent device is lost')
        except Exception as ex:
            raise RuntimeError('WGSL compile failed in adv_scalar.wgsl:sl') from ex
    except RuntimeError as ex:
        assert G.failure(ex) == 'lost'
    assert G.failure(ValueError('bad value')) is None
    assert G.failure(RuntimeError('frame 12 is neither simulated nor cached')) is None


# -- what the card has, and what is made ---------------------------------------------------------------------------------

def test_the_cards_memory_can_be_given(monkeypatch):
    monkeypatch.setenv(G.MEMORY_ENV, '6')     # (in GB as the system shows a card's: a 6 GB card)
    six = 6 * 2 ** 30
    assert G.card_memory({'vendor_id': 0x10DE, 'device_id': 1}) == {'total': six, 'budget': six, 'free': six,
                                                                       'source': G.MEMORY_ENV}
    monkeypatch.setenv(G.MEMORY_ENV, 'lots')
    monkeypatch.setattr(G, '_dxgi_memory', lambda info: {'total': 8e9, 'budget': 7e9, 'source': 'DXGI'})
    monkeypatch.setattr(G, '_nvml_memory', lambda info: {'total': 8e9, 'budget': 8e9, 'free': 5e9, 'source': 'NVML'})
    m = G.card_memory({'vendor_id': 0x10DE, 'device_id': 1})
    assert m['source'] == 'DXGI' and m['budget'] == 7e9 and m['free'] == 5e9, 'what other programs hold is left out'
    monkeypatch.setattr(G, '_dxgi_memory', lambda info: 1 / 0)    # (a reader that fails is passed over)
    assert G.card_memory({'vendor_id': 0x10DE, 'device_id': 1})['source'] == 'NVML'


def test_the_plan_comes_from_the_cards_size_not_what_is_free():
    """What happens to be free as the app starts (a browser holding some of the card) must not change the plan: it shapes
    the scene's layout and so its disk cache. DXGI and NVML say a little differently what one card has: the same plan."""
    dxgi = {'total': 25503465472.0, 'budget': 24698159104.0, 'free': 23.0e9, 'source': 'DXGI'}   # (an RTX 3090)
    nvml = {'total': 25769803776.0, 'budget': 25769803776.0, 'free': 7.0e9, 'source': 'NVML'}
    assert G.card_size(dxgi['total']) == G.card_size(nvml['total']) == 24 * 2 ** 30
    assert G.plan_for(dxgi) == G.plan_for(nvml) == G.plan_for(dict(dxgi, free=2e9))
    assert G.plan_for(dxgi) == pytest.approx(G.PLAN_SHARE * 24 * 2 ** 30 - G.PLAN_RESERVE)
    assert G.plan_for(None) is None and G.plan_for({'total': 0.3e9, 'free': 0.3e9}) == G.MIN_PLAN


def test_a_size_given_in_gb_is_planned_as_given(monkeypatch):
    """BLACKBODY_GPU_MEMORY names the card to plan for: it is not rounded to whole GB as a card's own reading is (2.5 is
    not planned as a 2 GB card, nor 3.5 as a 4 GB one)."""
    for gb in ('2.5', '3.5', '7.5'):
        monkeypatch.setenv(G.MEMORY_ENV, gb)
        m = G.card_memory({'vendor_id': 0x10DE, 'device_id': 1})
        assert G.plan_for(m) == pytest.approx(G.PLAN_SHARE * float(gb) * G.GB - G.PLAN_RESERVE), gb


def test_memory_is_said_in_the_gb_the_system_shows():
    """A 24 GB card is 24 GB (of 2**30 bytes), not 25.8."""
    ex = G.GPUOutOfMemory(2 * 2 ** 30, 20 * 2 ** 30, 24 * 2 ** 30, 'scal0')
    assert 'scal0 (2.00 GB)' in str(ex) and '20.00 GB of the 24.0 GB' in str(ex)


@pytest.mark.skipif(os.name != 'nt', reason='DXGI is Windows')
def test_dxgi_finds_no_card_that_is_not_there():
    assert G._dxgi_memory({'vendor_id': 0xFFFF, 'device_id': 0xFFFF}) is None


def test_textures_and_buffers_are_counted_until_they_go():
    g = fake_gpu(ceiling=1e6)
    t = G.Texture(g, (16, 16, 16), 'rgba16float', label='scal')
    b = G.Buffer(g, 1024, label='stats')
    assert g.allocated == 16 ** 3 * 8 + 1024
    t.destroy()
    t.destroy()                     # (once only)
    assert g.allocated == 1024
    del b
    gc.collect()
    assert g.allocated == 0, 'one let go without destroy() is counted out too'
    G.Texture(g, (4, 4, 1), 'depth32float', dim='2d')   # (a format that is never read back)
    gc.collect()
    assert g.allocated == 0


def test_an_allocation_past_what_the_system_gives_is_refused_before_the_driver_is_asked():
    g = fake_gpu(ceiling=1e6)
    keep = G.Texture(g, (32, 32, 32), 'rgba16float', label='vel0')   # 262 kB
    with pytest.raises(G.GPUOutOfMemory, match='no room for scal1'):
        G.Texture(g, (64, 64, 32), 'rgba16float', label='scal1')     # 1 MB more
    assert g.allocated == keep.nbytes and len(g.device.made) == 1
    # the driver's own refusal is raised the same way, and counted out
    g = fake_gpu(device=FakeDevice(wgpu.GPUValidationError('In wgpuDeviceCreateTexture\n    Not enough memory left.')))
    with pytest.raises(G.GPUOutOfMemory) as got:
        G.Texture(g, (64, 64, 64), 'r32float', label='sdf')
    assert got.value.driver and g.allocated == 0 and G.failure(got.value) == 'memory'
    # a stand-in GPU without the count still makes them (tests/test_suite.py)
    G.Buffer(types.SimpleNamespace(device=FakeDevice()), 16)


def test_running_out_of_memory_plans_within_less_each_time(monkeypatch):
    monkeypatch.setattr(G, '_CARD', dict(G._CARD))
    g = fake_gpu(ceiling=10e9, plan=8e9)
    p1 = g.out_of_memory(G.GPUOutOfMemory(4e9, 7e9, 10e9))         # (refused at the ceiling: it held at most that)
    assert p1 == pytest.approx(min(0.75 * 10e9 - G.PLAN_RESERVE, 0.75 * 8e9))
    p2 = g.out_of_memory(G.GPUOutOfMemory(1e9, 2e9, 10e9, driver=True))   # (the driver: it could not hold 3 GB)
    assert p2 == pytest.approx(0.75 * 3e9 - G.PLAN_RESERVE) and G.card()['plan'] == p2
    assert g.out_of_memory() >= 0.25e9, 'never planned down to nothing'
    g.allocated = int(g.plan)
    assert g.room() == pytest.approx(G.PLAN_RESERVE)


# -- the estimate --------------------------------------------------------------------------------------------------------

def test_the_estimate_counts_what_the_solver_makes(engine):
    """Solver.estimate against what configure really makes (every texture and buffer counted as it is made), for a few
    grids, optional fields and upres."""
    g = engine.gpu
    vb = 16 if g.vel_format == 'rgba32float' else 8
    for dims, feats, up in (((48, 80, 48), {}, 1), ((64, 104, 64), {'aux': True, 'chem': True, 'burn': True, 'stain': True}, 1),
                            ((40, 64, 40), {'water': True}, 2), ((32, 56, 40), {'aux': True}, 3)):
        gc.collect()
        s = Solver(g)
        a0 = g.allocated
        s.configure(dims, 0.02, (0.0, 0.0, 0.0), feats, up)
        made = g.allocated - a0
        assert made == pytest.approx(Solver.estimate(dims, s.features, up, vb), rel=0.005), (dims, feats, up)
        assert s.memory_bytes() == Solver.estimate(dims, s.features, up, vb)
        s._release()
        assert g.allocated - a0 < 0.02 * made, 'and gives it back'


def test_the_scene_estimate_is_near_what_an_engine_makes(engine):
    sc = presets.make('campfire')
    sc.data['domain'].update(resolution=64, preroll=0.0)
    sc.data['render'].update(width=320, height=180)
    gc.collect()
    from blackbody.engine.engine import Engine
    g = engine.gpu
    a0 = g.allocated
    e = Engine(g)
    e.prepare(sc)
    e.simulate_to(sc, sc.start + 1)
    e.render(sc, sc.start + 1, (320, 180), samples=4)
    made = g.allocated - a0
    est = sc.memory_estimate(False)
    assert est['simulation'] == pytest.approx(e.solver.memory_bytes(), rel=0.01)
    assert est['total'] == pytest.approx(made, rel=0.35), (est, made)
    del e
    gc.collect()


# -- a scene fitted to the card ------------------------------------------------------------------------------------------

def _big():
    sc = presets.make('campfire')
    sc.data['domain']['resolution'] = 384
    sc.data['render'].update(upres=3, auto_resolution=False, width=1280, height=720)
    return sc


def test_a_scene_is_taken_as_set_while_it_fits(card):
    sc = _big()
    asked = sc._asked_detail(True)
    assert sc.memory_plan(True)[:2] == asked and sc.memory_notes(True) == [], 'no GPU yet: as set'
    card(plan_gb=200.0, side=16384)
    assert sc.memory_plan(True)[:2] == asked and sc.memory_notes(True) == []


def test_past_the_cards_memory_the_detail_goes_first_then_the_voxels(card):
    sc = _big()
    need = sc.memory_estimate(True)['total']
    up2 = sc.memory_estimate(True, upres=2)
    card(plan_gb=(up2['simulation'] + up2['drawing'] + up2['pictures']) / 1e9 * 1.01, side=16384)
    res, up, got, note = sc.memory_plan(True)
    assert (res, up) == (384, 2) and got <= G.card()['plan']
    assert 'Final renders simulate with Detail upres 2 instead of 3' in note and f'{need / G.GB:.1f} GB' in note
    assert sc.upres_for(True) == 2 and sc.memory_notes(True) == [note]
    card(plan_gb=1.0, side=16384)
    res, up, got, note = sc.memory_plan(True)
    assert up == 1 and res < 384 and 'voxels on the longest side instead of 384' in note
    e = sc.memory_estimate(True, res, up)
    assert e['simulation'] + e['drawing'] <= 1e9 - e['pictures']
    dims, h, _ = sc.sim_layout(True)
    assert dims == Solver.dims_for(sc.domain_size(), res)[0], 'the layout (and so the cache signature) follows the plan'
    # the viewer is fitted too, at its own resolution
    assert sc.memory_plan(False)[3].startswith('The viewer simulates with')


def test_a_large_output_does_not_take_the_grids_down_to_nothing(card):
    sc = _big()
    sc.data['render'].update(width=7680, height=4320)        # (8K: about 10 GB of passes)
    card(plan_gb=4.0, side=16384)
    res, up, _need, note = sc.memory_plan(True)
    e = sc.memory_estimate(True, res, up)
    assert e['simulation'] + e['drawing'] <= 0.25 * 4e9 and res > 100
    assert 'the 7680×4320 output alone takes about' in note


def test_grids_are_kept_within_the_largest_3d_texture(card):
    sc = _big()
    card(plan_gb=None, side=1024)        # (384 voxels at upres 3 is 1,152 cells a side)
    res, up, _need, note = sc.memory_plan(True)
    assert (res, up) == (384, 2) and 'makes 3-D textures at most 1024 cells a side' in note
    card(plan_gb=None, side=256)
    res, up, _need, _note = sc.memory_plan(True)
    assert up == 1 and max(Solver.dims_for(sc.domain_size(), res)[0]) < 256


def test_a_liquids_surface_grid_is_kept_within_the_largest_3d_texture(card):
    """The liquid's surface grid is Surface detail times finer than its box (up to 3): on Direct3D 12 (2,048 a side) 768
    voxels at 3 would not be made."""
    from blackbody.engine.liquid_render import LiquidRenderer
    for name in ('rock_splash', 'hose_on_fire'):
        sc = presets.make(name)
        sc.data['domain']['resolution'] = 768
        sc.data['water']['surface_res'] = 3.0
        card(plan_gb=None, side=2048)
        res, _up, _need, note = sc.memory_plan(True)
        dims = Solver.dims_for(sc.domain_size(), res)[0]
        assert max(LiquidRenderer.surface_dims(dims, 3.0)) <= 2048 < max(LiquidRenderer.surface_dims(
            Solver.dims_for(sc.domain_size(), res + 8)[0], 3.0)), name
        assert 'at most 2048 cells a side' in note
        card()


def test_the_liquid_estimates_need_no_liquid(card):
    """The scene's estimate asks the liquid's sizes of the classes themselves, not of a stand-in instance."""
    from blackbody.engine.liquid import LiquidSolver
    from blackbody.engine.liquid_render import LiquidRenderer
    assert LiquidSolver.estimate((32, 24, 32), 1000, 64) > 0 and LiquidRenderer.memory_bytes((32, 24, 32), 2.0) > 0
    s = LiquidSolver.__new__(LiquidSolver)
    s.dims, s.capacity, s.ww_capacity = (32, 24, 32), 1000, 64
    assert s.memory_bytes() == LiquidSolver.estimate((32, 24, 32), 1000, 64)


def test_the_note_says_what_picked_the_detail_and_what_can_be_lowered(card):
    sc = presets.make('campfire')                      # (Resolution from the shot picks upres 3 at 144 voxels)
    assert sc.memory_plan(True)[1] == 3 and int(sc.data['render']['upres']) == 1
    card(plan_gb=None, side=400)                       # (the upres grid would be 432 cells a side)
    note = sc.memory_plan(True)[3]
    assert 'Detail upres 2 instead of the 3 Resolution from the shot picked' in note, note
    card(plan_gb=1.4, side=16384)
    sc.data['domain']['resolution'] = 256
    note = sc.memory_plan(True)[3]
    assert 'Lower Voxels yourself' in note and 'Detail upres yourself' not in note, note   # (Detail upres is at 1)


def test_liquid_and_sky_scenes_are_fitted_by_their_voxels(card):
    for name in ('rock_splash', 'cumulus_day', 'hose_on_fire'):
        sc = presets.make(name)
        sc.data['domain']['resolution'] = 256
        e = sc.memory_estimate(False)
        assert e['simulation'] > 0 and e['total'] > e['simulation']
        card(plan_gb=e['total'] / 2e9, side=16384)
        res, up, _need, note = sc.memory_plan(False)
        assert res < sc._asked_detail(False)[0] and up == sc._asked_detail(False)[1], name
        assert 'voxels on the longest side' in note
        card()


def test_resolution_from_the_shot_follows_a_smaller_card(card):
    """The detail upres Resolution from the shot picks keeps within 2 GB, and within half the plan on a card with less:
    not more on a larger one (upres 3 for 2 costs two to three times the final render)."""
    sc = presets.make('campfire')
    sc.data['domain']['resolution'] = 176
    assert model.auto_max_bytes() == model.AUTO_MAX_BYTES, 'a card not known: 2 GB'
    known = sc.auto_detail()
    card(plan_gb=40.0, side=16384)
    assert model.auto_max_bytes() == model.AUTO_MAX_BYTES and sc.auto_detail() == known == (176, 3)
    card(plan_gb=3.0, side=16384)
    assert model.auto_max_bytes() == pytest.approx(model.AUTO_SHARE * 3.0e9) and sc.auto_detail() == (176, 2)
    card(plan_gb=1.0, side=16384)
    assert sc.auto_detail()[1] == 1


def test_resolution_from_the_shot_picks_on_a_large_card_what_it_did_before(card):
    """With room on the card, the detail upres it picks is what it was before the card was read (a rule of thumb per
    cell, within 2 GB), so no final render loses detail at the Voxels a user sets. At these, Solver.estimate (a little
    higher) would have picked one less."""
    card(plan_gb=G.plan_for({'total': 24 * 2 ** 30}) / 1e9, side=16384)
    for name, res, up in (('campfire', 184, 3), ('campfire', 256, 2), ('fireball', 224, 2), ('flamethrower', 248, 3)):
        sc = presets.make(name)
        sc.data['domain']['resolution'] = res
        assert sc.auto_detail() == (res, up), name


def test_the_final_layout_is_the_same_on_any_card_it_fits(card):
    """The four presets whose detail upres a 24 GB card would have raised (to their final render's two to three times the
    time) keep it, and their signature, on every card they fit: 8, 12 and 24 GB."""
    for name in ('fireball', 'hillside_fire', 'gas_cloud', 'shed_fire'):
        sc = presets.make(name)
        sc._mesh_stamps = lambda: {}
        sigs = set()
        for gb in (8, 12, 24):
            card(plan_gb=G.plan_for({'total': gb * 2 ** 30}) / 1e9, side=16384)
            assert sc.upres_for(True) == 2 and sc.memory_notes(True) == [], (name, gb)
            sigs.add(sc.sim_signature(True))
        card()
        assert sc.upres_for(True) == 2 and sigs == {sc.sim_signature(True)}, name


# -- a disk cache keeps the layout it was simulated at ---------------------------------------------------------------------

def _cached(sc, tmp_path, final=True):
    """Open the scene's disk cache as the engine does (Engine._attach_disk), with one frame in it."""
    sc.data['domain'].update(disk_cache=True, cache_dir=str(tmp_path))
    c = SimCache(sc.cache_folder(final), sc.sim_signature(final), about=sc.cache_about(final))
    c.put(5, {'x': np.arange(4.0)}, wait=True)
    return c


def _reopened(sc, final=True, readonly=False):
    return SimCache(sc.cache_folder(final), sc.sim_signature(final), readonly=readonly, about=sc.cache_about(final))


def test_a_disk_cache_survives_a_start_with_a_smaller_plan(card, monkeypatch, tmp_path):
    """Simulated with room for upres 3 (Resolution from the shot), started again with a plan that would pick 2 but still
    holds 3: the layout, the signature and the frames are kept."""
    sc = presets.make('campfire')
    sc.data['domain']['resolution'] = 184
    card(plan_gb=6.0, side=16384)
    assert sc.memory_plan(True)[:2] == (184, 3)
    sig = sc.sim_signature(True)
    _cached(sc, tmp_path)
    # (a start with less, where 3 still fits but on its own the scene would take upres 2: Resolution from the shot's
    # share of the plan cut, as on a card where that share falls short of 3 while the plan holds it)
    card(plan_gb=4.0, side=16384)
    monkeypatch.setattr(model, 'AUTO_SHARE', 0.4)
    assert sc.auto_detail() == (184, 2) and sc.cache_fit(True) == (184, 3)
    assert sc.memory_plan(True)[:2] == (184, 3) and sc.memory_notes(True) == []
    assert sc.sim_signature(True) == sig and 5 in _reopened(sc), 'the cache is kept'


def test_a_disk_cache_cut_to_fit_is_kept_on_a_start_with_more_room(card, tmp_path):
    """A scene cut to fit (after running out of memory, or on a smaller card) and cached so keeps that layout when the
    next start has room for more, with a notice; deleting the cache lets it simulate as set."""
    sc = _big()
    card(plan_gb=1.0, side=16384)
    cut = sc.memory_plan(True)[:2]
    assert cut != (384, 3)
    sig = sc.sim_signature(True)
    _cached(sc, tmp_path)
    assert 'as set, the scene needs about' in sc.memory_notes(True)[0], 'what this card picks: the cut, not the cache'
    card(plan_gb=200.0, side=16384)
    assert sc.memory_plan(True)[:2] == cut and sc.sim_signature(True) == sig and 5 in _reopened(sc)
    note = sc.memory_notes(True)[0]
    assert 'the disk cache holds the simulation so' in note and str(sc.cache_folder(True)) in note
    assert note.endswith('to simulate it as set.')
    sc.from_cache = True                                  # (a render from the cache: no advice to simulate again)
    assert sc.memory_notes(True)[0].startswith('Final renders read the disk cache with ')
    # the viewer's own cache is another one
    assert sc.cache_fit(False) is None


def test_a_cached_layout_that_no_longer_fits_is_fitted_again(card, tmp_path):
    sc = _big()
    card(plan_gb=200.0, side=16384)
    _cached(sc, tmp_path)
    card(plan_gb=1.0, side=16384)
    res, up, _need, note = sc.memory_plan(True)
    assert (res, up) != (384, 3) and 'instead of 384' in note
    assert 5 not in _reopened(sc), 'simulated again, at what fits'


def test_a_render_from_the_cache_takes_its_layout_on_any_card(card, tmp_path):
    """A farm machine with a smaller card renders a cache as it was simulated, instead of refusing it as simulated with
    other settings; one that simulates (or the app) fits the scene to its own card."""
    sc = _big()
    card(plan_gb=200.0, side=16384)
    _cached(sc, tmp_path)
    card(plan_gb=1.0, side=16384)
    with pytest.raises(CacheMismatch):
        _reopened(sc, readonly=True)
    sc.from_cache = True
    assert sc.memory_plan(True)[:2] == (384, 3) and sc.sim_layout(True)[0] == Solver.dims_for(sc.domain_size(), 384)[0]
    assert 5 in _reopened(sc, readonly=True)
    assert 'more than Test card has room for, so it may run out of memory' in sc.memory_notes(True)[0]


def test_a_plan_cut_after_running_out_of_memory_keeps_a_disk_caches_frames(card, tmp_path):
    """Running out of memory cuts the plan for the rest of the session. A disk cache is judged by the card's own plan,
    not the cut one: its layout and frames are kept (not cleared to simulate the scene within less), and a notice says
    what to do if it runs out again. A cache with no frames in it has nothing to lose, and is fitted within the cut."""
    sc = _big()
    card(plan_gb=200.0, side=16384)
    sig = sc.sim_signature(True)
    _cached(sc, tmp_path)
    card(plan_gb=200.0, side=16384, cut_gb=1.0)
    assert sc.memory_plan(True)[:2] == (384, 3) and sc.sim_signature(True) == sig and 5 in _reopened(sc)
    note = sc.memory_notes(True)[0]
    assert note.startswith('Final renders keep the layout the disk cache holds (384 voxels on the longest side, Detail '
                           'upres 3) and its frames') and 'since it ran out of memory' in note, note
    assert str(sc.cache_folder(True)) in note
    card(plan_gb=1.0, side=16384)                           # (a card without room for it: fitted again, as before)
    assert sc.memory_plan(True)[:2] != (384, 3)
    empty = _big()
    empty.data['domain'].update(disk_cache=True, cache_dir=str(tmp_path / 'empty'))
    card(plan_gb=200.0, side=16384)
    SimCache(empty.cache_folder(True), empty.sim_signature(True), about=empty.cache_about(True))
    card(plan_gb=200.0, side=16384, cut_gb=1.0)
    assert empty.cache_fit(True) is None and empty.memory_plan(True)[:2] != (384, 3)


def test_a_plan_cut_on_a_card_whose_memory_is_not_known_keeps_a_disk_caches_frames(card, monkeypatch, tmp_path):
    """On a card whose memory cannot be read (Metal on a discrete card, Linux Intel) the scene is laid out as set; a
    plan cut after running out of memory keeps a disk cache's layout and frames there too, not cleared to simulate the
    scene within less."""
    sc = _big()
    card(plan_gb=None, side=16384)
    sig = sc.sim_signature(True)
    _cached(sc, tmp_path)
    monkeypatch.setitem(G._CARD, 'plan', 1e9)   # (out of memory: the plan cut, the card's own still unknown)
    assert sc.memory_plan(True)[:2] == (384, 3) and sc.sim_signature(True) == sig and 5 in _reopened(sc)
    assert 'since it ran out of memory' in sc.memory_notes(True)[0]


def test_a_cache_signed_with_more_than_the_scene_keeps_its_layout(card, tmp_path):
    """The engine may sign a disk cache with more than the scene's own signature (with solids, the bodies' fingerprint
    after it): the layout the cache holds is still found, by the scene's own signature recorded beside it."""
    sc = _big()
    card(plan_gb=1.0, side=16384)
    cut = sc.memory_plan(True)[:2]
    sc.data['domain'].update(disk_cache=True, cache_dir=str(tmp_path))
    c = SimCache(sc.cache_folder(True), f'{sc.sim_signature(True)}-0f1e2d3c', about=sc.cache_about(True))
    c.put(5, {'x': np.arange(4.0)}, wait=True)
    card(plan_gb=200.0, side=16384)
    assert sc.cache_fit(True) == cut and sc.memory_plan(True)[:2] == cut


def test_a_render_from_a_cache_that_records_no_layout_reads_it_as_set(card, tmp_path):
    """A cache written before layouts were recorded (meta.json holds only its signature) is rendered from as the scene
    is set, on a card without room for that, rather than refused as simulated with other settings."""
    sc = _big()
    sc.data['domain'].update(disk_cache=True, cache_dir=str(tmp_path))
    card(plan_gb=200.0, side=16384)
    c = SimCache(sc.cache_folder(True), sc.sim_signature(True))    # (no `about`: as caches were written before)
    c.put(5, {'x': np.arange(4.0)}, wait=True)
    card(plan_gb=1.0, side=16384)
    assert sc.cache_fit(True) is None, 'simulating, the scene is fitted to this card'
    sc.from_cache = True
    assert sc.memory_plan(True)[:2] == (384, 3) and 5 in _reopened(sc, readonly=True)


def test_a_changed_setting_or_shot_drops_the_cached_layout(card, tmp_path):
    sc = _big()
    card(plan_gb=1.0, side=16384)
    _cached(sc, tmp_path)
    card(plan_gb=200.0, side=16384)
    assert sc.cache_fit(True) is not None
    sc.emitters[0]['fuel'] = sc.emitters[0]['fuel'] * 1.5        # (another simulation)
    assert sc.cache_fit(True) is None and sc.memory_plan(True)[:2] == (384, 3)
    sc = presets.make('campfire')                                # (Resolution from the shot: the shot asks for more)
    sc.data['domain']['resolution'] = 176
    card(plan_gb=3.0, side=16384)
    _cached(sc, tmp_path / 'shot')
    assert sc.cache_fit(True) == (176, 2)
    card(plan_gb=200.0, side=16384)
    sc.data['camera']['distance'] *= 3.0                          # (further away: the shot wants less detail)
    assert sc._wanted_detail(True) != (176, 3) and sc.cache_fit(True) is None


def test_an_engines_disk_cache_survives_another_start(engine, monkeypatch, tmp_path):
    """Through the engine: final frames simulated into the disk cache with the scene cut to fit are still there when the
    app starts again with room for more (another plan: BLACKBODY_GPU_MEMORY, another card, the plan cut after running
    out of memory gone), and are simulated again only when the cached layout no longer fits."""
    from blackbody.engine.engine import Engine
    monkeypatch.setattr(G, '_CARD', dict(G._CARD))
    sc = presets.make('campfire')
    sc.data['domain'].update(resolution=64, preroll=0.0, disk_cache=True, cache_dir=str(tmp_path))
    sc.data['render'].update(width=160, height=90, upres=3, auto_resolution=False)
    sc.data['embers']['enabled'] = False
    e2 = sc.memory_estimate(True, 64, 2)
    small = e2['total'] * 1.01                              # (room for upres 2, not 3: a card that small)
    G._CARD.update(plan=small, card_plan=small, side=16384, name='Test card')
    e = Engine(engine.gpu)
    e.prepare(sc, final=True)
    assert e.solver.upres == 2
    e.simulate_to(sc, sc.start + 2)
    e.cache.disk.flush()
    frames = e.cache.disk.frames()
    assert frames == [sc.start, sc.start + 1, sc.start + 2]
    del e
    gc.collect()
    G._CARD.update(plan=100e9, card_plan=100e9)             # (started again with room for the scene as set)
    e = Engine(engine.gpu)
    e.prepare(sc.copy(), final=True)
    assert e.solver.upres == 2 and e.cache.disk.frames() == frames, 'the cache and its layout are kept'
    assert any('the disk cache holds the simulation so' in n for n in e.notices())
    del e
    gc.collect()
    G._CARD.update(plan=small * 0.6, card_plan=small * 0.6)   # (a card with no room for it: fitted again, simulated again)
    e = Engine(engine.gpu)
    e.prepare(sc.copy(), final=True)
    assert e.solver.upres == 1 and e.cache.disk.frames() == []
    e.cache.attach(None)
    del e
    gc.collect()


def test_the_layout_is_worked_out_once_while_the_engine_prepares(card, tmp_path, monkeypatch):
    sc = _big()
    _cached(sc, tmp_path)
    calls = []
    real = model.Scene._memory_plan
    monkeypatch.setattr(model.Scene, '_memory_plan', lambda self, final: calls.append(final) or real(self, final))
    with sc.planned():
        sc.sim_signature(True), sc.sim_layout(True), sc.upres_for(True), sc.memory_notes(True), sc.cache_about(True)
        with sc.planned():
            sc.sim_layout(True)
        assert calls == [True]
        c = sc.copy()                     # (a copy made meanwhile is not held: it may be changed)
        c.data['domain']['resolution'] = 64
        assert c.memory_plan(True)[0] == 64 and calls == [True, True]
    sc.sim_layout(True)
    assert calls == [True, True, True], 'and again outside it'


# -- Grow to fit -----------------------------------------------------------------------------------------------------------

def _solver(dims=(64, 96, 64), upres=1):
    """A Solver without a GPU, for growth_needed (it reads only its layout and the smoke's box)."""
    s = Solver.__new__(Solver)
    s.gpu = types.SimpleNamespace(vel_format='rgba32float')
    s.dims, s.upres, s.features = dims, upres, {}
    s.bbox = ((2, 0, 2), (dims[0] - 2, dims[1] - 2, dims[2] - 2))   # (smoke near every open side)
    s._carried = lambda: {}
    return s


def test_grow_to_fit_stops_at_the_largest_3d_texture():
    s = _solver()
    lo, hi = s.growth_needed(12, 16, (192, 288, 192), SolverParams())
    assert list(lo) == [16, 0, 16] and list(hi) == [16, 16, 16] and s.grow_held is None
    lo, hi = s.growth_needed(12, 16, (192, 288, 192), SolverParams(), max_side=105)
    assert list(lo) == [16, 0, 16] and list(hi) == [16, 8, 16], 'the velocity grid is a cell larger than the box'
    s.dims = (64, 104, 64)
    lo, hi = s.growth_needed(12, 16, (192, 288, 192), SolverParams(), max_side=105)
    assert hi[1] == 0 and s.grow_held == 'texture'
    s = _solver(upres=3)                # (the upres grid is three times finer: 105 // 3 = 35 < 64 already)
    lo, hi = s.growth_needed(12, 16, (192, 288, 192), SolverParams(), max_side=105)
    assert not (lo.any() or hi.any()) and s.grow_held == 'texture'
    s = _solver()
    s.growth_needed(12, 16, (64, 96, 64), SolverParams(), max_side=105)
    assert s.grow_held is None, 'Grow up to stopping it is no notice'


def test_grow_to_fit_stops_where_the_memory_runs_out():
    s = _solver()
    now = s.footprint()
    lo, hi = s.growth_needed(12, 16, (192, 288, 192), SolverParams(), max_bytes=now * 10)
    assert hi[1] == 16 and s.grow_held is None
    lo, hi = s.growth_needed(12, 16, (192, 288, 192), SolverParams(), max_bytes=s.footprint((72, 104, 64)))
    assert s.grow_held == 'memory' and (lo + hi).sum() > 0, 'smaller steps, fewer sides'
    assert s.footprint(tuple(np.asarray(s.dims) + lo + hi)) <= s.footprint((72, 104, 64))
    assert hi[1] == 8, 'up goes last'
    lo, hi = s.growth_needed(12, 16, (192, 288, 192), SolverParams(), max_bytes=now)
    assert not (lo.any() or hi.any()) and s.grow_held == 'memory'


def test_grow_to_fit_is_offered_in_fire_scenes_only():
    for key in ('grow', 'grow_limit'):
        assert applies('domain', key, 'fire')
        assert not any(applies('domain', key, k) for k in ('liquid', 'both', 'cloud')), key


def test_grow_to_fit_is_held_by_the_scenes_room_not_what_else_is_on_the_gpu(card):
    """How far a box grows shapes the simulation: it is held by what the scene's grids may take on this card
    (Scene.memory_room), the same on every run, not by what other engines or a large render happen to hold now."""
    from blackbody.engine.engine import Engine
    sc = presets.make('campfire')
    card(plan_gb=6.0, side=16384)
    seen = []
    for free in (0.0, 5e9):
        E = Engine.__new__(Engine)
        E.final = True
        E.gpu = types.SimpleNamespace(max_side=16384, room=lambda free=free: free)
        s = _solver()
        s.base, s.grow_held = None, None

        def needed(margin, step, limit, prm, max_bytes=None, max_side=None, s=s):
            seen.append(max_bytes)
            return np.zeros(3, int), np.zeros(3, int)
        s.growth_needed = needed
        E.solver = s
        E._grow(sc, SolverParams())
    assert seen == [sc.memory_room(True)] * 2 and seen[0] == pytest.approx(6e9 - 1920 * 1080 * model.PICTURE_BYTES)


def test_grow_to_fit_is_turned_off_when_a_fire_scene_takes_liquid():
    """Grow to fit is hidden outside fire scenes, so it could not be turned off there: adding water turns it off, once."""
    from blackbody.scene import components as C
    sc = presets.make('campfire')
    sc.data['domain']['grow'] = True
    _added, notes = C.add(sc, 'pour')
    assert sc.kind in ('liquid', 'both') and not sc.data['domain']['grow']
    assert any('Grow to fit is off' in n for n in notes)
    sc = presets.make('campfire')                       # (and nothing said when it was off)
    _added, notes = C.add(sc, 'pour')
    assert not any('Grow to fit' in n for n in notes)


def test_making_something_float_in_a_fire_scene_says_grow_to_fit_is_off():
    """Make float turns a fire scene into a fire-and-liquid one (a pond to float in): that Grow to fit is off is said with
    the rest, as when water is added."""
    _app()
    from blackbody.ui import actions
    sc = presets.make('campfire')
    sc.data['domain']['grow'] = True
    sc.add_collider(name='Crate', position=(0.4, 0.3, 0.0))
    said = []
    doc = types.SimpleNamespace(scene=sc, frame=sc.start, edit=lambda text, fn, structure=False: fn(sc),
                                set_playing=lambda on: None)
    actions.make_float(types.SimpleNamespace(doc=doc, msg=types.SimpleNamespace(setText=said.append)), 0, 600.0)
    assert sc.kind == 'both' and not sc.data['domain']['grow']
    assert said[-1].startswith('Crate floats') and 'Grow to fit is off' in said[-1], said


def test_the_engine_says_what_stopped_the_box_growing():
    from blackbody.engine.engine import Engine
    E = Engine.__new__(Engine)
    E.kind, E._cap_notes, E._col_names, E._matter, E._strands, E.weather, E._wx_on = 'fire', [], [], None, None, None, False
    E.solids = types.SimpleNamespace(warnings=[], capped={})
    E.solver = types.SimpleNamespace(meshes=types.SimpleNamespace(errors={}), upres=2)
    E.gpu = types.SimpleNamespace(max_side=2048)
    assert E.notices() == []
    E._grow_held = ('memory', (96, 160, 96))
    assert E.notices()[0].startswith("Grow to fit stopped at 96×160×96 cells: a larger box would not fit in the GPU's memory")
    E._grow_held = ('texture', (96, 160, 96))
    assert 'at most 2048 cells a side (the detail upres grid is 2 times finer)' in E.notices()[0]


def test_the_box_grows_on_the_gpu_as_it_did_through_memory(engine):
    """A growth copied on the GPU (Solver.grow) leaves every field as the one through the computer's memory did."""
    g = engine.gpu
    rng = np.random.default_rng(3)
    out = []
    for room in (None, 0):              # (no room for both boxes at once: through the computer's memory)
        s = Solver(g)
        s.configure((32, 48, 40), 0.02, (-0.32, 0.0, -0.4), {'aux': True, 'burn': True, 'stain': True}, 2)
        s._prm = SolverParams()
        for t in [s.vel[0], s.scal[0], s.P[0], s.aux[0], s.burn[0], s.stain, s.scal_fine[0]]:
            dtype, ch, _ = G.FORMATS[t.format]
            g.upload(t, rng.random(t.size[::-1] + (ch,)).astype(dtype))
        s.time, s.steps = 1.25, 40
        s.grow((8, 0, 0), (0, 8, 16), room)
        g.sync()
        assert s.dims == (40, 56, 56) and s.origin[0] == pytest.approx(-0.48) and s.time == 1.25 and s.steps == 40
        out.append({k: g.read(t) for k, t in (('vel', s.vel[0]), ('scal', s.scal[0]), ('p', s.P[0]), ('aux', s.aux[0]),
                                                 ('burn', s.burn[0]), ('burn1', s.burn[1]), ('stain', s.stain),
                                                 ('fine', s.scal_fine[0]))})
        s._release()
        rng = np.random.default_rng(3)
    for k in out[0]:
        assert np.array_equal(out[0][k], out[1][k]), k


# -- the engine made again after the GPU fails -----------------------------------------------------------------------------

class _Engine:
    """Stands in for engine.Engine in the worker (no GPU)."""
    made = []

    def __init__(self, gpu, cache_bytes=None):
        self.gpu, self.cache_bytes, self.carried = gpu, cache_bytes, None
        self.cache = types.SimpleNamespace(items={1: 'a', 2: 'b'})
        _Engine.made.append(self)

    def carry(self, old):
        self.carried = old

    def invalidate(self):
        pass


def _app():
    """The Qt application the worker's signals need: a full one, as the other UI tests make it (a bare QCoreApplication
    made first would leave them none to make their widgets in)."""
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _GPU:
    made = []
    keep_plan = G.GPU.keep_plan

    def __init__(self, ok=True):
        self.ok, self.oom, self.forgot, self.on_compile, self.plan = ok, None, False, None, 20e9
        _GPU.made.append(self)

    def alive(self):
        return self.ok

    def lost_reason(self):
        return 'The WGPU device was lost (Unknown):\nDevice dropped.'

    def out_of_memory(self, ex=None):
        self.oom = ex
        return 3.2 * G.GB

    def forget_bindings(self):
        self.forgot = True


@pytest.fixture
def worker(monkeypatch):
    _app()
    import blackbody.engine.engine as E
    from blackbody.render.layers import LayerEngines
    from blackbody.ui.worker import EngineWorker
    monkeypatch.setattr(E, 'Engine', _Engine)
    monkeypatch.setattr(G, 'GPU', _GPU)
    _Engine.made, _GPU.made = [], []
    w = EngineWorker()
    w.engine = _Engine(_GPU())
    w.layer_engines = LayerEngines(w.engine, cache_bytes=123)
    w.layer_engines.extra['smoke'] = _Engine(w.engine.gpu, 123)
    said = []
    w.message.connect(said.append)
    w.said = said
    return w


def test_a_lost_device_makes_the_engines_again_on_a_new_one(worker):
    old, layer = worker.engine, worker.layer_engines.extra['smoke']
    assert worker._recover(wgpu.GPUValidationError('Parent device is lost'))
    new = worker.engine
    assert new is not old and new.carried is old and new.gpu is not old.gpu and new.gpu is _GPU.made[-1]
    assert worker.layer_engines.base is new and worker.layer_engines.cache_bytes == 123
    assert worker.layer_engines.extra['smoke'].carried is layer and worker.layer_engines.extra['smoke'].gpu is new.gpu
    assert new.gpu.on_compile == worker._compiling
    assert 'The GPU stopped responding (its device was lost: Device dropped.)' in worker.said[-1]
    assert 'kept the 2 cached frames' in worker.said[-1]


def test_running_out_of_memory_makes_the_engines_again_on_the_same_gpu_within_less(worker):
    g = worker.engine.gpu
    ex = G.GPUOutOfMemory(2e9, 5e9, 6e9, 'scal-fine0')
    assert worker._recover(ex)
    assert worker.engine.gpu is g and g.oom is ex and g.forgot
    assert 'The GPU ran out of memory' in worker.said[-1] and '3.2 GB' in worker.said[-1]
    assert worker._recover(ex) and worker._recover(ex) and not worker._recover(ex), 'three times, then given up'
    assert 'ran out of memory again' in worker._advice and 'close other programs that use the GPU' in worker._advice


def test_a_new_gpu_after_a_lost_device_keeps_a_plan_cut_after_running_out(worker, monkeypatch):
    """A plan cut after running out of memory stays cut on the new device: the scene is laid out as it was, so the
    frames carried over still belong to it."""
    monkeypatch.setattr(G, '_CARD', dict(G._CARD))
    worker.engine.gpu.plan = 3.2e9
    assert worker._recover(wgpu.GPUValidationError('Parent device is lost'))
    assert worker.engine.gpu.plan == 3.2e9 and G.card()['plan'] == 3.2e9


def test_other_failures_are_left_to_the_error_and_a_gpu_that_keeps_failing_is_given_up(worker):
    assert not worker._recover(ValueError('a bug')), 'the GPU still answers: not its failure'
    worker.engine.gpu.ok = False
    assert worker._recover(ValueError('garbage read back')), 'the GPU no longer answers: lost'
    for _ in range(2):
        assert worker._recover(wgpu.GPUValidationError('Parent device is lost'))
    # the frame loop's handler (EngineWorker.run): the restart advice is in what it says, not hidden behind it
    worker._frame_failed(wgpu.GPUValidationError('Parent device is lost'))
    assert worker.said[-1].startswith('Render failed: Parent device is lost')
    assert 'save your work and restart the app' in worker.said[-1]


def test_a_gpu_that_cannot_be_started_again_is_said_so(worker, monkeypatch):
    def broken(self, ok=True):
        raise G.GPUUnavailable('No GPU adapter found.')
    monkeypatch.setattr(_GPU, '__init__', broken)
    old = worker.engine
    worker._frame_failed(wgpu.GPUValidationError('Parent device is lost'))
    assert worker.engine is old and 'could not start it again (No GPU adapter found.): restart the app' in worker.said[-1]
    # Cache range and Export frame say it too
    worker._cache_range = types.MethodType(type(worker)._cache_range, worker)
    worker.scene = presets.make('campfire')
    import blackbody.render.layers as L
    monkeypatch.setattr(L, 'simulate', lambda *a, **k: (_ for _ in ()).throw(wgpu.GPUValidationError('Parent device is lost')))
    worker._cache_range()
    assert worker.said[-1].startswith('Caching stopped:') and 'restart the app' in worker.said[-1]
    got = []
    worker.stillDone.connect(lambda result, text: got.append(text))
    worker._still({'scene': worker.scene, 'frame': 1})
    assert 'restart the app' in got[-1]


def test_a_lost_device_is_recovered_with_the_cache_kept(engine, monkeypatch):
    """A simulated device loss (the device destroyed under a running simulation): the worker makes the engine again on a
    new device, shows the cached frames as they were and simulates on."""
    _app()
    from blackbody.engine.engine import Engine
    from blackbody.ui.worker import EngineWorker
    monkeypatch.setattr(G, '_CARD', dict(G._CARD))     # (the GPUs made here are not the one the other tests use)
    sc = presets.make('campfire')
    sc.data['domain'].update(resolution=40, preroll=0.0)
    sc.data['render'].update(width=160, height=90)
    sc.data['embers']['enabled'] = False
    g = G.GPU()
    w = EngineWorker()
    w.engine = Engine(g)
    w.scene = sc
    shown = []
    w.frameReady.connect(lambda img, f, st: shown.append((f, st)))
    assert w._show(sc.start + 2)
    before = w.engine.cache.get(sc.start + 1)['scal'].copy()
    g.device.destroy()                                 # (as a driver reset leaves it)
    with pytest.raises(Exception) as got:
        w._show(sc.start + 3)
    assert G.failure(got.value) == 'lost' or not g.alive()
    assert w._recover(got.value)
    assert w.engine.gpu is not g and w.engine.gpu.alive()
    assert w._show(sc.start + 1), 'a cached frame, drawn on the new device'
    assert w.engine.sim_frame is None and np.array_equal(w.engine.cache.get(sc.start + 1)['scal'], before)
    assert w._show(sc.start + 3), 'and the simulation carries on'
    assert shown[-1][0] == sc.start + 3 and w.engine.sim_frame == sc.start + 3
    del w
    gc.collect()


def test_an_allocation_past_the_ceiling_is_recovered_with_a_smaller_grid(engine, monkeypatch):
    """Out of memory as the grids are made: refused at the GPU's ceiling, the engine is made again on the same device
    within less, and the scene is fitted to that, with a notice."""
    _app()
    from blackbody.engine.engine import Engine
    from blackbody.ui.worker import EngineWorker
    monkeypatch.setattr(G, '_CARD', dict(G._CARD))     # (this GPU is not the one the other tests use)
    monkeypatch.setattr(G, 'PLAN_RESERVE', 0.0)
    g = G.GPU()
    g.plan, g.ceiling = 100e9, 0.6e9       # (the plan lets the scene be made as set, and the allocation is what fails)
    G._CARD.update(plan=g.plan)
    sc = presets.make('campfire')
    sc.data['domain'].update(resolution=440, preroll=0.0)    # (330 voxels in the viewer: about 0.8 GB of grids)
    sc.data['render'].update(width=160, height=90)
    sc.data['embers']['enabled'] = False
    w = EngineWorker()
    w.engine = Engine(g)
    w.scene = sc
    try:                                   # (as the worker's loop has it)
        w._show(sc.start)
        pytest.fail('made past the ceiling')
    except G.GPUOutOfMemory as ex:
        assert w._recover(ex)
    assert w.engine.gpu is g and g.plan == pytest.approx(0.45e9) and g.allocated < 0.05e9, 'the old engine let go'
    assert w._show(sc.start)
    assert max(w.engine.solver.dims) < 330 and g.allocated <= g.ceiling
    assert any('The viewer simulates with' in n and 'instead of 330' in n for n in w.engine.notices())
    del w
    gc.collect()


def test_running_out_of_memory_keeps_the_disk_cache(engine, monkeypatch, tmp_path):
    """Through the worker: frames simulated into the disk cache, then the driver refuses an allocation (other programs
    holding the card). The engine made again within the cut plan keeps the cached layout and every frame on disk, says
    what to do if it runs out again, and simulates on: it does not clear the cache to simulate within less."""
    _app()
    from blackbody.engine.engine import Engine
    from blackbody.ui.worker import EngineWorker
    monkeypatch.setattr(G, '_CARD', dict(G._CARD))     # (this GPU is not the one the other tests use)
    monkeypatch.setattr(G, 'MIN_PLAN', 1e6)            # (a cut below even this small scene)
    g = G.GPU()
    sc = presets.make('campfire')
    sc.data['domain'].update(resolution=64, preroll=0.0, disk_cache=True, cache_dir=str(tmp_path))
    sc.data['render'].update(width=160, height=90)
    sc.data['embers']['enabled'] = False
    w = EngineWorker()
    w.engine = Engine(g)
    w.scene = sc
    assert w._show(sc.start + 2)
    w.engine.cache.disk.flush()
    frames, dims = w.engine.cache.disk.frames(), w.engine.solver.dims
    assert frames == [sc.start, sc.start + 1, sc.start + 2]
    assert w._recover(G.GPUOutOfMemory(0.05 * G.GB, g.allocated, g.ceiling, 'scal1', driver=True))
    assert g.plan < sc.memory_estimate(False)['total'] < G.plan_for(g.memory), 'the scene as cached is past the cut plan'
    assert w._show(sc.start + 3)
    assert w.engine.solver.dims == dims and w.engine.cache.disk.frames() == frames + [sc.start + 3]
    assert any(n.startswith('The viewer keeps the layout the disk cache holds') for n in w.engine.notices())
    w.engine.cache.attach(None)
    del w
    gc.collect()
