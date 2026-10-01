"""Matter (engine/matter.py): sand, snow, mud, jelly and clay as MPM particles, checked against what they do for real."""
import math

import numpy as np
import pytest

from blackbody.engine.matter import MatterSpec, fill_points, friction_angle
from blackbody.scene.model import Scene


def test_a_shape_is_filled_with_eight_particles_to_a_cell():
    rng = np.random.default_rng(0)
    assert len(fill_points('box', (0.1, 0.1, 0.1), 0.01, rng)) == 8000
    assert abs(len(fill_points('sphere', (0.1, 0.1, 0.1), 0.01, rng)) / (4.0 / 3.0 * math.pi * 1000.0) - 1.0) < 0.04
    assert abs(len(fill_points('cylinder', (0.1, 0.1, 0.1), 0.01, rng)) / (math.pi * 100.0 * 20.0) - 1.0) < 0.04
    # a pile: a cone 40 cm across and 20 cm high
    assert abs(len(fill_points('pile', (0.2, 0.1, 0.2), 0.01, rng)) / (math.pi * 400.0 * 20.0 / 3.0) - 1.0) < 0.05


def test_the_angle_of_repose_asked_for_sets_the_friction_angle():
    assert abs(friction_angle(28.7) - 33.0) < 1e-9
    a = [friction_angle(x) for x in (15.0, 25.0, 30.0, 35.0, 40.0, 45.0)]
    assert all(b > c for b, c in zip(a[1:], a[:-1]))
    assert all(f >= r for f, r in zip(a, (15.0, 25.0, 30.0, 35.0, 40.0, 45.0)))


def test_matter_is_saved_with_the_scene():
    s = Scene()
    i = s.add_matter(material='snow', shape='sphere', position=(0.0, 1.0, 0.0), size=(0.1, 0.1, 0.1), velocity=(3.0, 1.0, 0.0))
    j = s.add_matter(material='sand', pours=True, rate=2.0, own_colour=True, colour=(1.0, 0.5, 0.0))
    assert s.matter[i]['name'] == 'Snow' and s.matter[j]['name'] == 'Sand'
    s2 = Scene.from_dict(s.to_dict())
    assert [m['material'] for m in s2.matter] == ['snow', 'sand']
    assert s.sim_signature() == s2.sim_signature()
    specs = s2.matter_specs()
    assert specs[0].shape == 'sphere' and specs[0].velocity == (3.0, 1.0, 0.0)
    assert specs[1].pour and abs(specs[1].rate - 0.002) < 1e-12             # (litres a second: m^3/s)
    assert np.allclose(specs[1].colour, (1.0, 0.214, 0.0), atol=1e-3)      # (sRGB as it is picked: linear)
    s2.matter[0]['material'] = 'packing_snow'
    assert s.sim_signature() != s2.sim_signature()


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
