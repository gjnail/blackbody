"""Matter (engine/matter.py): sand, snow, mud, jelly and clay as MPM particles on the GPU, checked against what they do
for real. (Its tests that need no GPU are in test_matter_model.py.)"""
import math

import numpy as np
import pytest

from blackbody.engine.matter import MatterSpec
from blackbody.scene.model import Scene


# ---- on the GPU ---------------------------------------------------------------------------------------------------


@pytest.fixture
def gpu(engine):
    return engine.gpu


def matter(gpu, specs, box=((-0.5, 0.0, -0.5), (1.0, 0.8, 1.0)), res=64, secs=10.0):
    from blackbody.engine.matter import Matter
    M = Matter(gpu)
    M.configure(specs, box[0], box[1], resolution=res, duration=secs)
    return M


def run(M, secs, every=None):
    out = []
    for f in range(int(round(secs * 24))):
        M.advance(1.0 / 24.0)
        if every and f % every == every - 1:
            out.append(M.read_particles().copy())
    return out


def live(M):
    P = M.read_particles()
    P = P[P[:, 3] >= 0.0]
    return P, M.origin + P[:, :3].astype(float) * M.dx


def test_poured_sand_heaps_at_its_angle_of_repose(gpu):
    M = matter(gpu, [MatterSpec(material='sand', pos=(0.0, 0.3, 0.0), size=(0.025,) * 3, velocity=(0.0, -0.3, 0.0), pour=True,
                                rate=6e-4, start=0.0, stop=3.0)], res=96, secs=4.0)
    run(M, 4.0)
    _, x = live(M)
    r = np.percentile(np.hypot(x[:, 0], x[:, 2]), 99.5)
    angle = math.degrees(math.atan(x[:, 1].max() / r))
    assert 30.0 < angle < 38.0          # (dry sand: 34 degrees)


def test_a_column_of_sand_collapses_and_none_is_lost(gpu):
    M = matter(gpu, [MatterSpec(material='sand', shape='cylinder', pos=(0.0, 0.3, 0.0), size=(0.1, 0.3, 0.1))])
    n0 = M.count
    run(M, 1.5)
    P, x = live(M)
    assert len(P) == n0
    assert x[:, 1].max() < 0.25 and np.ptp(x[:, 0]) > 0.4       # it spread out, low
    assert M.max_speed < 0.2                                      # and stopped


def test_jelly_keeps_its_volume_and_springs_back(gpu):
    M = matter(gpu, [MatterSpec(material='jelly', shape='box', pos=(0.0, 0.4, 0.0), size=(0.08, 0.08, 0.08))])
    heights = []
    for f in range(48):
        M.advance(1.0 / 24.0)
        _, x = live(M)
        heights.append(np.ptp(x[:, 1]))
    P, x = live(M)
    F = P[:, 20:32].reshape(-1, 3, 4)[:, :, :3]
    assert abs(np.linalg.det(F).mean() - 1.0) < 0.02             # (nearly incompressible)
    assert min(heights) < 0.92 * heights[0]                      # squashed as it landed
    assert heights[-1] > 0.9 * heights[0]                        # and stood up again (sagging a little: rho g h / E)


def test_mud_slumps_and_stops(gpu):
    M = matter(gpu, [MatterSpec(material='mud', shape='cylinder', pos=(0.0, 0.2, 0.0), size=(0.1, 0.2, 0.1))])
    run(M, 1.0)
    _, x1 = live(M)
    run(M, 1.0)
    _, x2 = live(M)
    assert x2[:, 1].max() < 0.2                                  # slumped from 40 cm
    assert x2[:, 1].max() > 0.02                                 # but held up by its yield stress
    assert abs(x2[:, 1].max() - x1[:, 1].max()) < 0.01           # and it has stopped


def test_clay_keeps_the_shape_it_is_squashed_into(gpu):
    M = matter(gpu, [MatterSpec(material='clay', shape='box', pos=(0.0, 0.35, 0.0), size=(0.08, 0.08, 0.08))])
    run(M, 1.0)
    _, x1 = live(M)
    run(M, 1.0)
    _, x2 = live(M)
    assert 0.6 * 0.16 < np.ptp(x2[:, 1]) < 0.98 * 0.16           # squashed a little as it landed
    assert abs(np.ptp(x2[:, 1]) - np.ptp(x1[:, 1])) < 0.003      # and stays so


def test_snow_packs_where_it_is_squeezed(gpu):
    from blackbody.engine.solver import ColliderGPU, pack_colliders
    M = matter(gpu, [MatterSpec(material='packing_snow', shape='sphere', pos=(-0.25, 0.4, 0.0), size=(0.07,) * 3,
                                velocity=(6.0, 0.5, 0.0))])
    wall = [ColliderGPU(shape='box', pos=(0.3, 0.4, 0.0), size=(0.05, 0.4, 0.4))]
    for f in range(12):
        M.advance(1.0 / 24.0, lambda fr: wall, None, pack_colliders)
    P, x = live(M)
    assert P[:, 7].min() < 0.9                                   # packed (its volume, Jp, down)
    assert x[:, 0].max() < 0.26                                  # and stopped by the wall


def test_matter_is_the_same_every_time(gpu):
    spec = [MatterSpec(material='sand', shape='cylinder', pos=(0.0, 0.25, 0.0), size=(0.08, 0.25, 0.08))]
    a = matter(gpu, spec)
    b = matter(gpu, spec)
    run(a, 0.5)
    run(b, 0.5)
    assert np.array_equal(a.read_particles(), b.read_particles())


def test_a_falling_crate_lands_on_sand_and_a_heavy_ball_sinks_into_it(engine):
    s = Scene()
    s.emitters = []
    s.data['domain'].update(size_x=1.6, size_y=1.0, size_z=1.0, resolution=32, preroll=0.0, matter_detail=64,
                            open_sides=False)
    s.add_matter(material='sand', shape='box', position=(0.0, 0.1, 0.0), size=(0.6, 0.1, 0.35))
    s.add_collider(name='Crate', shape='box', position=(-0.3, 0.5, 0.0), size=(0.1, 0.1, 0.1), dynamic=True, material='wood')
    s.add_collider(name='Ball', shape='sphere', position=(0.3, 0.7, 0.0), size=(0.07, 0.07, 0.07), dynamic=True,
                   material='steel')
    engine.invalidate()
    engine.prepare(s, final=False)
    engine.simulate_to(s, s.start + 36, cache=False)
    o = engine.solids.overrides()
    crate, ball = o[0]['pos'][1] - 0.1, o[1]['pos'][1] - 0.07     # their bottoms
    assert 0.12 < crate < 0.25          # on the sand (20 cm deep), not through it
    assert 0.0 < ball < crate - 0.03    # the steel ball, much heavier for its size, sank into it


def test_a_cached_frame_is_drawn_as_it_was(engine):
    s = Scene()
    s.emitters = []
    s.data['domain'].update(size_x=1.2, size_y=1.0, size_z=1.2, resolution=32, preroll=0.0, matter_detail=48)
    s.data['camera'].update(distance=2.0, target_y=0.2, pitch=15.0, use_anchor=False)
    s.add_matter(material='clay', shape='box', position=(0.0, 0.4, 0.0), size=(0.1, 0.1, 0.1))
    engine.invalidate()
    engine.prepare(s, final=False)
    engine.simulate_to(s, s.start + 6, cache=True)
    engine.render(s, s.start + 6, (160, 90))
    live_img = engine.display_image().astype(int)
    engine.simulate_to(s, s.start + 10, cache=True)
    engine.render(s, s.start + 6, (160, 90))                    # (from the cache now)
    cached = engine.display_image().astype(int)
    assert np.abs(live_img - cached).max() <= 2
    assert np.abs(live_img[30:70, 60:100] - live_img[5:10, 60:100].mean((0, 1))).mean() > 10   # (the clay is there)


def test_a_blast_throws_sand_away_from_it(gpu):
    M = matter(gpu, [MatterSpec(material='sand', shape='box', pos=(0.15, 0.05, 0.0), size=(0.15, 0.05, 0.15))])
    run(M, 0.2)
    P0, x0 = live(M)
    M.blast((-0.2, 0.05, 0.0), 0.05)                            # (a large firework, 20 cm off)
    run(M, 0.4)
    P1, x1 = live(M)
    assert len(P1) == len(P0)
    shift = x1[:, 0] - x0[:, 0]
    assert shift.mean() > 0.03                                   # pushed away from it (the near sand shoving the far along)
    assert shift.min() > -0.01                                   # and none of it toward it


# ---- solid to the smoke and the water (matter_field.py) -------------------------------------------------------------

def test_smoke_goes_round_a_sand_heap_not_through_it(engine):
    from blackbody.scene import components
    inside_smoke = []
    for heap in (False, True):
        sc = components.new_scene('fire', 'person')
        sc.emitters = []
        sc.add_emitter(name='Smoke', shape='sphere', position=(-0.8, 0.2, 0.0), size=(0.15, 0.15, 0.15), fuel=0.0, temperature=0.05,
                       smoke=5.0, velocity=(1.5, 0.0, 0.0), embers=False)
        if heap:
            sc.add_matter(name='Heap', material='sand', shape='pile', position=(0.2, 0.2, 0.0), size=(0.45, 0.2, 0.45))
        sc.data['domain'].update(size_x=2.4, size_y=1.6, size_z=1.6, resolution=64, preroll=0.0)
        sc.data['motion'].update(wind_speed=1.5, wind_dir=90.0, gust=0.0, wind_relax=2.0, buoyancy=0.5)
        engine.prepare(sc, final=True)
        engine.simulate_to(sc, sc.start + 48, cache=False)
        s = engine.solver
        scal = engine.gpu.read(s.scal[0]).astype(np.float32)
        nz, ny, nx = scal.shape[:3]
        Z, Y, X = np.meshgrid(*(s.origin[a] + (np.arange(n) + 0.5) * s.h for a, n in ((2, nz), (1, ny), (0, nx))), indexing='ij')
        inside = (np.hypot(X - 0.2, Z) < 0.25) & (Y < 0.08)
        inside_smoke.append(float(scal[..., 2][inside].mean()))
    assert inside_smoke[0] > 0.01 and inside_smoke[1] < 0.02 * inside_smoke[0]


def test_water_runs_off_a_sand_heap_not_through_it(engine):
    from blackbody.scene import components
    deep = []
    for heap in (False, True):
        sc = components.new_scene('liquid', 'person')
        sc.emitters = []
        components.add(sc, 'pour', at=(0.15, 0.0))
        if heap:
            sc.add_matter(name='Heap', material='wet_sand', shape='pile', position=(0.2, 0.25, 0.0), size=(0.5, 0.25, 0.5))
        engine.prepare(sc, final=False)
        engine.simulate_to(sc, sc.start + 40, cache=False)
        x = engine.liquid.read_particles()[0]
        deep.append(int(((np.hypot(x[:, 0] - 0.2, x[:, 2]) < 0.3) & (x[:, 1] < 0.12)).sum()))
    assert deep[0] > 100 and deep[1] == 0


# ---- melting (mpm_melt.wgsl) ----------------------------------------------------------------------------------------

def melting(engine, material, kind='both', secs=4.0):
    from blackbody.scene import components
    sc = components.new_scene(kind, 'person')
    sc.emitters = []
    sc.add_emitter(name='Fire', shape='cylinder', position=(-0.12, 0.05, 0.0), size=(0.2, 0.05, 0.3), fuel=12, temperature=0.6,
                   **({'emits': 'fire'} if kind == 'both' else {}))
    sc.add_matter(name='Heap', material=material, shape='pile', position=(0.25, 0.15, 0.0), size=(0.3, 0.15, 0.3))
    sc.data['domain'].update(size_x=1.8, size_y=1.6, size_z=1.4, resolution=64, preroll=0.0)
    engine.prepare(sc, final=False)
    first = None
    for f in range(sc.start, sc.start + int(secs * sc.fps) + 1):
        engine.simulate_to(sc, f, cache=False)
        if first is None:
            first = int((engine.matter.read_particles()[:, 3] >= 0.0).sum())
    left = int((engine.matter.read_particles()[:, 3] >= 0.0).sum())
    water = len(engine.liquid.read_particles()[0]) if kind == 'both' else 0
    return first, left, water


def test_snow_the_flames_touch_melts_into_water(engine):
    first, left, water = melting(engine, 'snow')
    assert left < 0.99 * first and water > 20          # its flank melts, and the water runs off
    assert first - left < 0.5 * first                  # (only where the flames touch it)


def test_sand_does_not_melt_and_snow_in_a_fire_box_just_goes(engine):
    first, left, _ = melting(engine, 'sand')
    assert left == first
    first, left, _ = melting(engine, 'snow', kind='fire')
    assert left < 0.99 * first


# ---- getting wet (mpm_wet.wgsl) -------------------------------------------------------------------------------------

def test_water_poured_on_sand_wets_the_sand_it_runs_over(engine):
    from blackbody.scene import components
    sc = components.new_scene('liquid', 'person')
    sc.emitters = []
    components.add(sc, 'pour', at=(0.1, 0.0))
    sc.add_matter(name='Heap', material='sand', shape='pile', position=(0.2, 0.25, 0.0), size=(0.5, 0.25, 0.5))
    engine.prepare(sc, final=False)
    m, L = engine.matter, engine.liquid
    assert [x.key for x in m._mats] == ['sand', 'wet_sand', 'soaked_sand']
    engine.simulate_to(sc, sc.start + 48, cache=False)
    P = m.read_particles()
    wet = P[np.round(P[:, 3]) >= 1, :3] * m.dx + np.asarray(m.origin)
    assert 0.002 * len(P) < len(wet) < 0.2 * len(P)       # the sand the water runs over, not the whole heap
    # (and only there: every wet grain is within a few of the liquid's cells of some water)
    near = np.zeros(tuple(int(n) for n in L.dims), bool)
    c = np.floor((L.read_particles()[0] - np.asarray(L.origin)) / L.h).astype(int)
    c = c[np.all((c >= 0) & (c < near.shape), axis=1)]
    near[c[:, 0], c[:, 1], c[:, 2]] = True
    for _ in range(3):
        grown = near.copy()
        for a in range(3):
            grown |= np.roll(near, 1, a) | np.roll(near, -1, a)
        near = grown
    g = np.clip(np.floor((wet - np.asarray(L.origin)) / L.h).astype(int), 0, np.asarray(near.shape) - 1)
    assert near[g[:, 0], g[:, 1], g[:, 2]].mean() > 0.9


def test_a_pour_digs_into_a_heap_of_dry_sand(engine):
    from blackbody.scene import components
    sc = components.new_scene('liquid', 'person')
    sc.emitters = []
    components.add(sc, 'pour', at=(0.1, 0.0))
    sc.add_matter(name='Heap', material='sand', shape='pile', position=(0.2, 0.25, 0.0), size=(0.5, 0.25, 0.5))
    engine.prepare(sc, final=False)
    m = engine.matter
    engine.simulate_to(sc, sc.start, cache=False)
    P0 = m.read_particles()[:, :3].copy()
    engine.simulate_to(sc, sc.start + 96, cache=False)
    P = m.read_particles()
    moved = np.linalg.norm(P[:len(P0), :3] - P0, axis=1) * m.dx
    assert int((np.round(P[:, 3]) == 2).sum()) > 500       # the water soaks through where it runs
    assert 500 < int((moved > 0.02).sum()) < 0.02 * len(P0)  # and carries it off there: a crater and a gully


def castle(engine, until):
    """A block of damp sand (a sand castle) that a wall of water let go beside it floods: [(seconds, slots of its grains,
    how far each has moved (m))] each second."""
    from blackbody.scene import components
    sc = components.new_scene('liquid', 'person')
    sc.emitters = []
    sc.add_emitter(name='Water', shape='box', position=(-0.7, 0.3, 0.0), size=(0.25, 0.3, 0.5), noise=0.0, embers=False,
                   liquid_mode='fill', start=0.0)
    sc.add_matter(name='Castle', material='wet_sand', shape='box', position=(0.3, 0.12, 0.0), size=(0.12, 0.12, 0.2))
    sc.data['domain'].update(open_sides=False)
    engine.prepare(sc, final=False)
    m = engine.matter
    engine.simulate_to(sc, sc.start, cache=False)
    P0 = m.read_particles()[:, :3].copy()
    out = []
    for f in range(sc.start, sc.start + int(until * sc.fps) + 1, int(sc.fps)):
        engine.simulate_to(sc, f, cache=False)
        P = m.read_particles()
        out.append(((f - sc.start) / sc.fps, np.round(P[:, 3]).astype(int),
                    np.linalg.norm(P[:len(P0), :3] - P0, axis=1) * m.dx, P[:, :3] * m.dx + np.asarray(m.origin)))
    return out


def test_damp_sand_holds_when_the_water_hits_it_then_soaks_through_and_slumps(engine):
    run = castle(engine, 8.0)
    t, slot, moved, x = run[1]
    assert t == 1.0 and np.mean(moved > 0.02) < 0.01         # the wave has hit it: it stands
    t, slot, moved, x = run[-1]
    assert np.mean(slot == 1) > 0.4                           # soaked through
    assert np.mean(moved > 0.02) > 0.25                       # and slumped
    assert x[:, 0].min() > -0.3 and x[:, 0].max() < 0.9       # where it stood


def test_sand_slumping_into_water_does_not_blow_the_water_apart(engine):
    # (water caught between the slumping sand and the floor once made the pressure solve fail)
    from blackbody.scene import components
    sc = components.new_scene('liquid', 'person')
    sc.emitters = []
    components.add(sc, 'block', at=(0.0, 0.0))
    sc.add_matter(name='Heap', material='sand', shape='box', position=(0.3, 0.12, 0.0), size=(0.12, 0.12, 0.2))
    engine.prepare(sc, final=False)
    for f in range(sc.start + 1, sc.start + 13):
        engine.simulate_to(sc, f, cache=False)
        v = engine.liquid.read_particles()[1]
        assert len(v) and np.percentile(np.linalg.norm(v, axis=1), 99.9) < 6.0, f


def test_sand_in_a_fire_box_has_no_wet_sand(engine):
    from blackbody.scene import components
    sc = components.new_scene('fire', 'person')
    sc.add_matter(name='Heap', material='sand', shape='pile', position=(0.2, 0.25, 0.0), size=(0.5, 0.25, 0.5))
    engine.prepare(sc, final=False)
    assert [x.key for x in engine.matter._mats] == ['sand'] and not engine.matter.wets()
