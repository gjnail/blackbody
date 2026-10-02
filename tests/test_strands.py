"""Grass and plants (engine/strands.py) on the GPU: blades that grow where they are put (not inside things; on what
stays put when they grow on objects), stand, bend in the wind and spring back, are pushed aside by what moves through
them, burn and feed the fire, are cached as bytes and drawn. (Its tests that need no GPU are in
test_strands_model.py.)"""
import numpy as np
import pytest

from blackbody.engine.solver import ColliderGPU
from blackbody.engine.strands import POINTS, Strands, StrandSpec


class Look:   # what Strands.step reads of a look
    ambient_k = 293.0
    flame_k = 1650.0


@pytest.fixture
def gpu(engine):
    return engine.gpu


def grown(gpu, specs, colliders=(), fixed=None):
    S = Strands(gpu)
    S.configure(specs)
    with gpu.batch() as b:
        S.place(b, list(colliders), None, True, 0.0, (1 << len(colliders)) - 1 if fixed is None else fixed)
    return S


def run(gpu, S, secs, wind=(0.0, 0.0, 0.0), colliders=(), at=None, fps=24):
    for f in range(int(round(secs * fps))):
        with gpu.batch() as b:
            S.step(b, 1.0 / fps, None, at(f) if at else list(colliders), None, wind, 0.0, Look, True, 0.0)
    return S


def tips(S):
    x, alive = S.positions()
    return x[alive], alive


def test_a_patch_grows_on_the_ground_where_it_is_put_and_stands(gpu):
    S = grown(gpu, [StrandSpec(kind='meadow', pos=(0.5, 0.0, -0.2), size=(1.0, 0.45, 0.6))])
    x, alive = tips(S)
    assert alive.all() and len(x) == S.n > 1500
    root = x[:, 0]
    assert np.allclose(root[:, 1], 0.0) and np.abs(root[:, 0] - 0.5).max() <= 1.0 + 1e-5 and np.abs(root[:, 2] + 0.2).max() <= 0.6 + 1e-5
    rest = x[:, -1, 1].mean()
    assert 0.6 * 0.45 < rest < 0.45                         # standing, arched over a little
    seg = np.linalg.norm(np.diff(x, axis=1), axis=2)
    assert np.allclose(seg, seg[:, :1], rtol=1e-3)          # (its segments all the same length)
    run(gpu, S, 2.0)                                         # left alone, it stays as it stood
    y, _ = tips(S)
    assert np.abs(y - x).max() < 0.005


def test_nothing_grows_inside_an_object_and_grass_can_grow_on_one_that_stays_put(gpu):
    box = ColliderGPU(shape='box', pos=(0.0, 0.3, 0.0), size=(0.3, 0.3, 0.3))
    S = grown(gpu, [StrandSpec(kind='lawn', size=(1.0, 0.08, 1.0))], [box])
    x, alive = S.positions()
    root = x[:, 0]
    under = (np.abs(root[:, 0]) < 0.28) & (np.abs(root[:, 2]) < 0.28)
    assert under.any() and not alive[under].any() and alive[~under & (np.abs(root[:, 0]) > 0.32)].all()
    on = grown(gpu, [StrandSpec(kind='lawn', size=(1.0, 0.08, 1.0), on_objects=True)], [box])
    x, alive = on.positions()
    top = alive & (np.abs(x[:, 0, 0]) < 0.28) & (np.abs(x[:, 0, 2]) < 0.28)
    assert top.sum() > 100 and np.allclose(x[top, 0, 1], 0.6, atol=0.004)   # on its top
    # (not on one that moves: the mask leaves it out)
    moving = grown(gpu, [StrandSpec(kind='lawn', size=(1.0, 0.08, 1.0), on_objects=True)], [box], fixed=0)
    x, alive = moving.positions()
    assert not (alive & (x[:, 0, 1] > 0.1)).any()


def test_the_wind_bends_it_downwind_harder_the_stronger_it_blows_and_it_springs_back(gpu):
    lean = {}
    for speed in (2.0, 5.0):
        S = grown(gpu, [StrandSpec(kind='meadow', size=(0.6, 0.45, 0.6))])
        x0, _ = tips(S)
        run(gpu, S, 2.0, wind=(speed, 0.0, 0.0))
        x1, _ = tips(S)
        lean[speed] = float((x1[:, -1, 0] - x0[:, -1, 0]).mean())
        run(gpu, S, 4.0)
        x2, _ = tips(S)
        assert abs(float((x2[:, -1, 0] - x0[:, -1, 0]).mean())) < 0.1 * lean[speed]   # back up when it stops
    assert 0.03 < lean[2.0] < lean[5.0] < 0.45
    stiff = grown(gpu, [StrandSpec(kind='meadow', size=(0.6, 0.45, 0.6), stiffness=4.0)])
    x0, _ = tips(stiff)
    run(gpu, stiff, 2.0, wind=(5.0, 0.0, 0.0))
    x1, _ = tips(stiff)
    assert float((x1[:, -1, 0] - x0[:, -1, 0]).mean()) < 0.7 * lean[5.0]


def test_something_moving_through_it_pushes_it_aside(gpu):
    S = grown(gpu, [StrandSpec(kind='meadow', size=(1.0, 0.45, 0.4))])

    def ball(f):   # a 30 cm ball rolling through at a metre a second
        return [ColliderGPU(shape='sphere', pos=(-1.2 + f / 24.0, 0.15, 0.0), size=(0.15, 0.15, 0.15), vel=(1.0, 0.0, 0.0))]

    worst = 0.0
    for f in range(48):
        run(gpu, S, 1.0 / 24.0, at=lambda _f, f=f: ball(f))
        x, _ = tips(S)
        c = np.array(ball(f)[0].pos)
        worst = max(worst, float(0.15 - np.linalg.norm(x[:, 1:] - c, axis=2).min()))
    assert worst < 0.01          # (no blade inside it, give or take a centimetre at the substep it moved into)


def test_the_cache_keeps_the_blades_to_a_fraction_of_their_height(gpu):
    S = grown(gpu, [StrandSpec(kind='meadow', size=(0.6, 0.45, 0.6))])
    run(gpu, S, 0.5, wind=(4.0, 0.0, 1.0))
    snap = S.snapshot()
    assert snap['off'].dtype == np.int8 and snap['off'].nbytes == S.n * (POINTS - 1) * 3
    x_live, _ = S.positions()
    S.use_view(snap)
    x_view = np.frombuffer(gpu.read_buffer(S._view['X']), np.float32).reshape(S.n, POINTS, 4)[:, :, :3]
    h = S.BL[:, 1, 0][:, None, None]
    assert (np.abs(x_view - x_live) <= h / 127.0 * 0.51 + 1e-6).all()
    S.use_view(None)


def test_grass_is_the_same_every_time(gpu):
    a = run(gpu, grown(gpu, [StrandSpec(kind='meadow', size=(0.5, 0.45, 0.5))]), 1.0, wind=(3.0, 0.0, 0.0))
    b = run(gpu, grown(gpu, [StrandSpec(kind='meadow', size=(0.5, 0.45, 0.5))]), 1.0, wind=(3.0, 0.0, 0.0))
    assert np.array_equal(a.positions()[0], b.positions()[0])


# ---- with the fire ---------------------------------------------------------------------------------------------------

def field(dryness, grass=True, res=64):
    from blackbody.scene import components
    sc = components.new_scene('fire', 'person')
    sc.emitters = []
    sc.add_emitter(name='Torch', shape='sphere', position=(-1.6, 0.1, 0.0), size=(0.2, 0.1, 0.2), fuel=14, temperature=0.6,
                   start=0.0, stop=1.0, fade_out=0.3)
    if grass:
        sc.add_strands(kind='meadow', size=(1.8, 0.45, 1.0), dryness=dryness)
    sc.data['domain'].update(size_x=4.4, size_y=2.4, size_z=2.8, resolution=res, preroll=0.0)
    sc.data['motion'].update(wind_speed=2.5, wind_dir=90.0, gust=0.3, wind_relax=2.0)
    return sc


def burn(engine, sc, secs):
    engine.prepare(sc, final=True)
    burning = []
    for f in range(sc.start, sc.start + int(secs * sc.fps) + 1):
        engine.simulate_to(sc, f, cache=False)
        burning.append(engine.stats().get('burning', 0))
    return burning


def test_a_dry_field_burns_through_from_a_torch_downwind_and_feeds_the_fire(engine):
    sc = field(0.85)
    gas = burn(engine, sc, 6.0)
    S = engine.strands
    st = S.state()
    x, alive = S.positions()
    burnt = alive & (st[:, 1] >= 1.0)
    assert burnt.mean() > 0.3                                   # the fire ran through it
    assert x[burnt, 0, 0].max() > 1.0                           # downwind, from the torch at the far side
    assert np.allclose(np.linalg.norm(x[burnt, -1] - x[burnt, 0], axis=1), 0.0, atol=0.15 * 0.45 + 0.01)   # stubble
    bare = burn(engine, field(0.85, grass=False), 6.0)
    assert sum(gas[48:]) > 3 * max(sum(bare[48:]), 1)           # it was the grass burning, long after the torch went out


def test_fresh_grass_is_hard_to_light(engine):
    burn(engine, field(0.0), 4.0)
    st = engine.strands.state()
    assert (st[:, 1] >= 1.0).mean() < 0.15


def test_grass_is_drawn_and_the_cache_draws_it_again(engine):
    from blackbody.scene import components
    sc = components.new_scene('fire', 'person')
    sc.emitters = []
    sc.add_strands(kind='meadow', size=(1.0, 0.45, 0.6))
    sc.data['composite'].update(backdrop='stage', floor='dirt')
    sc.data['camera'].update(distance=3.0, target_y=0.3, pitch=10.0, yaw=0.0, use_anchor=False)
    engine.prepare(sc, final=False)
    for f in range(sc.start, sc.start + 4):
        engine.simulate_to(sc, f, cache=True)
    engine.render(sc, sc.start + 3, (160, 90), samples=1)
    live = engine.display_image()[..., :3].astype(float)
    engine.render(sc, sc.start + 1, (160, 90), samples=1)       # a cached frame
    cached = engine.display_image()[..., :3].astype(float)
    sc.strands[0]['enabled'] = False
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 3, cache=False)
    engine.render(sc, sc.start + 3, (160, 90), samples=1)
    bare = engine.display_image()[..., :3].astype(float)
    grass = np.abs(live - bare).sum(axis=2) > 12.0
    assert grass[45:, :].mean() > 0.2                           # the lower half of the frame is grass
    assert np.abs(cached - live).sum(axis=2)[grass].mean() < 12.0   # (and so is the cached frame's)


def test_the_ground_under_grass_is_shaded_and_black_where_it_has_burnt(gpu):
    S = grown(gpu, [StrandSpec(kind='meadow', pos=(0.5, 0.0, 0.0), size=(1.0, 0.45, 0.6))])
    st = np.zeros((S.n, 4), np.float32)
    x, _ = S.positions()
    st[x[:, 0, 0] < 0.5, 1] = 1.5                 # the left half has burnt
    gpu.write_buffer(S.bufs['ST'], st)
    with gpu.batch() as b:
        tex, lo, size = S.ground_map(b)
    m = gpu.read(tex).astype(np.float32)         # (z, x, 4): how burnt, how thick
    assert lo[0] <= -0.5 and lo[1] <= -0.6 and lo[0] + size[0] >= 1.5 and lo[1] + size[1] >= 0.6   # (round the patch)
    zs, xs = np.meshgrid((np.arange(m.shape[0]) + 0.5) / m.shape[0] * size[1] + lo[1],
                         (np.arange(m.shape[1]) + 0.5) / m.shape[1] * size[0] + lo[0], indexing='ij')
    inside = (np.abs(xs - 0.5) < 0.8) & (np.abs(zs) < 0.4)
    left, right = inside & (xs < 0.3), inside & (xs > 0.7)
    assert m[left, 0].mean() > 0.9 and m[right, 0].mean() < 0.05     # black where it burnt
    assert m[right, 1].mean() > 2.0 * m[left, 1].mean() > 0.0         # the thick grass shades it more than stubble
