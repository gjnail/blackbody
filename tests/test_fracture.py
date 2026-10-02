"""Breaking things: cutting objects into pieces (engine/fracture.py) and the pieces coming apart under load
(engine/solids.py). No GPU: MuJoCo runs on its own."""
import math

import numpy as np
import pytest

from blackbody.engine.fracture import bricks, fracture, make_piece, shape_planes

SHAPES = [('box', (0.5, 0.3, 0.2)), ('sphere', (0.2, 0.2, 0.2)), ('cylinder', (0.15, 0.3, 0.15))]


def whole(shape, size):
    pl = shape_planes(shape, size)
    return make_piece(pl, np.zeros(len(pl), bool))[0]


def connected(frac):
    """Whether every piece is glued, through its neighbours, to every other."""
    seen = {0}
    todo = [0]
    nbr = {}
    for b in frac.bonds:
        nbr.setdefault(b.i, []).append(b.j)
        nbr.setdefault(b.j, []).append(b.i)
    while todo:
        for j in nbr.get(todo.pop(), []):
            if j not in seen:
                seen.add(j)
                todo.append(j)
    return len(seen) == len(frac.pieces)


@pytest.mark.parametrize('shape,size', SHAPES)
@pytest.mark.parametrize('pattern', ['voronoi', 'splinters'])
def test_the_pieces_fill_the_object_exactly(shape, size, pattern):
    f = fracture(shape, size, 24, pattern, seed=3)
    assert len(f.pieces) >= 12
    assert abs(f.volume - whole(shape, size).volume) < 1e-6 * whole(shape, size).volume + 1e-12
    for p in f.pieces:
        assert np.all(p.verts @ p.planes[:, :3].T <= p.planes[:, 3] + 1e-6)     # convex
        assert p.inner.any()                                                     # each has a cut face
    assert connected(f)
    assert all(b.area > 0.0 for b in f.bonds)


def test_glass_breaks_into_slivers_through_its_thickness():
    f = fracture('box', (0.5, 0.5, 0.004), 40, 'shards', seed=1)
    assert abs(f.volume - 0.5 * 0.5 * 0.004 * 8) < 1e-9
    # every cut runs straight through the pane: no cut face has a normal along its thin axis
    for p in f.pieces:
        assert np.all(np.abs(p.planes[p.inner, 2]) < 1e-9)


def test_a_wall_is_cut_into_bricks_in_running_bond():
    f = bricks((0.8, 0.6, 0.05))
    assert abs(f.volume - 1.6 * 1.2 * 0.1) < 1e-9
    xs = {}
    for p in f.pieces:
        lo = p.verts.min(0)
        xs.setdefault(round(lo[1], 6), []).append(round(lo[0], 6))
    rows = [sorted(v) for _, v in sorted(xs.items())]
    assert len(rows) == 16
    # the joints of each course fall half a brick from the joints of the course below
    assert rows[0][1] - rows[1][1] != 0.0 and abs(abs(rows[0][1] - rows[1][1]) - 0.5 * (rows[0][2] - rows[0][1])) < 1e-6
    assert connected(f)


def test_a_hollow_box_breaks_as_its_walls():
    f = fracture('box', (0.25, 0.25, 0.25), 30, 'splinters', seed=2, hollow=0.02)
    assert abs(f.volume - (0.5 ** 3 - 0.46 ** 3)) < 1e-9
    assert connected(f)


def test_a_vase_breaks_as_strips_of_its_wall():
    R, H, t = 0.1, 0.17, 0.008
    f = fracture('cylinder', (R, H, R), 40, 'voronoi', seed=2, hollow=t)
    shell = math.pi * R * R * 2 * H - math.pi * (R - t) ** 2 * (2 * H - t)
    assert shell <= f.volume < 1.3 * shell      # (each strip's inside is flat)
    assert connected(f)


# ---- coming apart -----------------------------------------------------------------------------------------------

pytest.importorskip('mujoco')

from blackbody.engine.solids import Solids  # noqa: E402
from blackbody.scene.model import Scene  # noqa: E402


def scene_of(*cols, fps=24):
    s = Scene()
    s.data['domain'].update(ground=True, open_sides=True, preroll=0.0)
    s.data['render']['fps'] = float(fps)
    s.emitters = []
    for c in cols:
        s.add_collider(**c)
    return s


def run(scene, seconds):
    S = Solids()
    S.configure(scene, ((96, 96, 96), 12.0 / 96, (-6.0, 0.0, -6.0)))
    S.reset()
    start = {k: v['pos'].copy() for k, v in S.piece_poses(whole=True).items()}
    for f in range(scene.start + 1, scene.start + 1 + int(round(seconds * scene.fps))):
        S.advance(scene, f, 1.0 / scene.fps, 1)
    moved = {k: np.linalg.norm(v['pos'] - start[k], axis=1) for k, v in S.piece_poses(whole=True).items()}
    return S, moved


WALL = dict(name='Wall', shape='box', position=(0.0, 0.6, 0.0), size=(0.8, 0.6, 0.05), breakable=True, fracture='bricks',
            material='brick')


def ball(v):
    return dict(name='Ball', shape='sphere', position=(0.0, 0.6, -1.0), size=(0.2, 0.2, 0.2), dynamic=True, material='steel',
                start_velocity=(0.0, 0.0, v))


def test_a_brick_wall_stands_until_it_is_hit():
    S, moved = run(scene_of(WALL), 2.0)
    assert not S.breaks
    assert moved[0].max() < 0.005


def test_a_soft_knock_cracks_a_wall_and_a_hard_one_knocks_it_down():
    S1, soft = run(scene_of(WALL, ball(1.0)), 2.0)
    S2, hard = run(scene_of(WALL, ball(6.0)), 2.0)
    assert (soft[0] > 0.05).sum() <= 3
    assert len(S2.breaks) > 3 * max(len(S1.breaks), 1)
    assert (hard[0] > 0.05).sum() > 50


def test_a_stone_goes_through_a_window():
    pane = dict(name='Pane', shape='box', position=(0.0, 0.9, 0.0), size=(0.5, 0.5, 0.004), breakable=True, fracture='shards',
                material='glass', pieces=40, held='edges')
    stone = dict(name='Stone', shape='sphere', position=(0.0, 0.9, -0.8), size=(0.04, 0.04, 0.04), dynamic=True,
                 material='stone', start_velocity=(0.0, 0.3, 9.0))
    S, moved = run(scene_of(pane, stone), 1.0)
    assert S.overrides()[1]['pos'][2] > 0.3          # it went through
    assert 3 <= (moved[0] > 0.05).sum() < 30          # a hole: some shards fall, the rest stay in the frame


def test_a_vase_set_down_holds_and_one_dropped_shatters():
    vase = dict(name='Vase', shape='cylinder', size=(0.1, 0.17, 0.1), hollow=0.008, breakable=True, dynamic=True,
                material='ceramic', pieces=40)
    S1, _ = run(scene_of(dict(vase, position=(0.0, 0.171, 0.0))), 1.0)
    S2, _ = run(scene_of(dict(vase, position=(0.0, 1.3, 0.0), start_spin=(90.0, 0.0, 40.0))), 1.5)
    assert len(S1.breaks) == 0
    assert len(S2.breaks) > 20


def test_metal_bends_and_stays_bent_and_holds_its_own_weight():
    post = dict(name='Post', shape='box', position=(0.0, 1.0, 0.0), size=(0.04, 1.0, 0.04), breakable=True, fracture='bends',
                material='steel', pieces=12, held='base')
    S0, _ = run(scene_of(post), 1.0)
    assert not S0.sets[0].bent and S0.whole(0)                       # a steel post stands as it is
    ball = dict(name='Ball', shape='sphere', position=(-0.61, 1.2, 0.0), size=(0.21, 0.21, 0.21), dynamic=True, material='steel',
                start_velocity=(6.0, 0.0, 0.0))
    S, _ = run(scene_of(post, ball), 2.0)
    top = S.piece_poses()[0]['pos'][-1]
    assert S.sets[0].bent and not S.whole(0)                         # a 300 kg ball at 6 m/s bends it
    assert 0.2 < top[0] < 1.0                                        # over, and it stays over (not sprung back)
    assert (S._w['over'] >= 0).all()                                 # without tearing
    # a cached frame carries the bend: loaded back, it is as bent
    st = S.state()
    S.reset()
    assert not S.sets[0].bent
    assert S.load_state(st) and S.sets[0].bent
    assert np.allclose(S.model.eq_data[S._w['eq'], 3:10], st['bend'][0])


def test_a_mesh_breaks_into_pieces_that_glue_and_do_not_overlap():
    import os
    from blackbody.engine.fracture import mesh_pieces
    from blackbody.engine.mesh import load_mesh
    from blackbody.engine.solids import mesh_occupancy
    path = os.path.join(os.path.dirname(__file__), '..', 'blackbody', 'assets', 'meshes', 'armchair.obj')
    v, t = load_mesh(path)
    v = np.asarray(v, float) * 0.5
    f = mesh_pieces(v, t, 24, np.random.default_rng(1))
    P = f.pieces
    lo, hi = v.min(0), v.max(0)
    cell = float((hi - lo).max()) / 48
    vol = mesh_occupancy(v, t, lo, hi, cell).sum() * cell ** 3
    assert 15 <= len(P) <= 80
    assert 0.9 < sum(p.volume for p in P) / vol < 1.6                # convex pieces fill a little of its hollows in
    for i in range(len(P)):                                          # no two overlap (they would be thrown apart)
        for j in range(len(P)):
            if i != j:
                assert ((P[i].verts @ P[j].planes[:, :3].T - P[j].planes[:, 3]).max(1) > -2e-3).all()
    par = list(range(len(P)))
    def find(a):
        while par[a] != a:
            a = par[a]
        return a
    for bd in f.bonds:
        par[find(bd.i)] = find(bd.j)
    assert len({find(i) for i in range(len(P))}) == 1               # glued into one chair
    chair = dict(name='Chair', shape='mesh', mesh=os.path.abspath(path), position=(0.0, 2.5, 0.0), size=(0.5, 0.5, 0.5),
                 dynamic=True, breakable=True, material='stone', pieces=24, start_spin=(40.0, 0.0, 30.0))
    S, _ = run(scene_of(chair), 2.0)
    assert len(S.breaks) > 20 and not S.whole(0)                     # dropped 2.5 m, a stone chair shatters


def test_breaking_is_deterministic():
    S1, m1 = run(scene_of(WALL, ball(4.0)), 1.0)
    S2, m2 = run(scene_of(WALL, ball(4.0)), 1.0)
    assert len(S1.breaks) == len(S2.breaks)
    assert np.array_equal(m1[0], m2[0])


# ---- breaking and burning (Breakable and Burnable: each piece burns on its own) ----------------------------------------

SPREAD = dict(catch_temp=0.3, catch_time=0.4, creep=0.06, burn_time=2.0, smoulder=3.0, fuel=8.0, heat=0.5, smoke=1.0,
              smoulder_smoke=1.0)
POST = dict(name='Post', shape='box', position=(0.0, 0.8, 0.0), size=(0.07, 0.8, 0.07), material='wood', breakable=True,
            burnable=True, pieces=14, held='base')


def burning_post(**kw):
    sc = scene_of(dict(POST, **kw))
    sc.data['spread']['enabled'] = True
    S = Solids()
    S.configure(sc, ((96, 96, 96), 12.0 / 96, (-6.0, 0.0, -6.0)))
    S.reset()
    return sc, S


def flames_at_foot(S, height=0.3):
    """The gas temperature at fire_points(): a flame round the post's foot, up to `height`."""
    pts, own = S.fire_points()
    return np.where(pts[:, 1] < height, 1.0, 0.0), own


def test_a_burnable_breakable_burns_piece_by_piece_and_falls_when_its_foot_burns_through():
    sc, S = burning_post()
    ps = S.sets[0]
    assert ps.burnable and S.burning and (ps.fire[:, 0] == 1.0).all()
    for f in range(sc.start + 1, sc.start + 1 + 24 * 20):
        T, own = flames_at_foot(S)
        S.burn(1.0 / 24.0, T, own, SPREAD)
        S.advance(sc, f, 1.0 / 24.0, 1)
        if f == sc.start + 24:
            y = S.data.xipos[ps.bodies][:, 1]
            alight = (ps.fire[:, 1] >= 1.0) & (ps.fire[:, 0] > 0.0)
            assert alight[y < 0.3].all() and not alight[y > 0.9].any()   # its foot caught in the flames within a second
    assert (ps.fire[:, 2] >= 1.5).all()                              # the fire crept up it, and it burnt through
    assert S.data.xipos[ps.bodies[~ps.gone]][:, 1].max() < 0.3        # it fell
    assert 0 < ps.gone.sum() < len(ps.gone)                          # most crumbled to ash, some charcoal is left
    assert (S.piece_poses()[0]['pos'][ps.gone][:, 1] < -1.0e3).all()  # (the ash is drawn nowhere)


def test_a_breakable_that_does_not_burn_or_is_not_near_fire_stays_whole():
    sc, S = burning_post(burnable=False)
    assert not S.burning
    sc, S = burning_post()
    for f in range(sc.start + 1, sc.start + 1 + 24 * 3):
        pts, own = S.fire_points()
        S.burn(1.0 / 24.0, np.zeros(len(pts)), own, SPREAD)
        S.advance(sc, f, 1.0 / 24.0, 1)
    ps = S.sets[0]
    assert (ps.fire[:, 1] == 0.0).all() and S.data.xipos[ps.bodies][:, 1].max() > 1.4     # (still standing, unlit)


def test_burning_pieces_feed_the_fire_and_their_glue_weakens_with_their_char():
    sc, S = burning_post()
    ps = S.sets[0]
    for _ in range(24):
        T, own = flames_at_foot(S)
        S.burn(1.0 / 24.0, T, own, SPREAD)
    pts, val = S.fuel_points(SPREAD)
    alight = ((ps.fire[:, 1] >= 1.0) & (ps.fire[:, 0] > 0.0)).sum()
    assert len(pts) == 7 * alight > 0 and (val[:, 0] > 0.0).all() and (val[:, 1] == 0.5).all()
    for _ in range(24 * 2):
        T, own = flames_at_foot(S)
        S.burn(1.0 / 24.0, T, own, SPREAD)
    W = S._w
    char = 1.0 - ps.fire[:, 0]
    worst = np.maximum(char[W['first']], np.where(W['other'] >= 0, char[np.maximum(W['other'], 0)], 0.0))
    assert np.allclose(W['strength'], W['strength0'] * (1.0 - worst) ** 2)
    assert W['strength'].min() < 0.5 * W['strength0'].max()


def test_burning_is_kept_with_the_state_and_undone_by_a_reset():
    sc, S = burning_post()
    ps = S.sets[0]
    for f in range(sc.start + 1, sc.start + 1 + 24 * 12):
        T, own = flames_at_foot(S)
        S.burn(1.0 / 24.0, T, own, SPREAD)
        S.advance(sc, f, 1.0 / 24.0, 1)
    assert ps.gone.any()
    st = S.state()
    fire, gone = ps.fire.copy(), ps.gone.copy()
    S.reset()
    assert not ps.gone.any() and (ps.fire[:, 0] == 1.0).all() and np.allclose(S._w['strength'], S._w['strength0'])
    g = int(S.model.body_geomadr[int(ps.bodies[np.nonzero(gone)[0][0]])])
    assert S.model.geom_contype[g] != 0                              # (the ash is whole again)
    assert S.load_state(st)
    assert np.array_equal(ps.fire, fire) and np.array_equal(ps.gone, gone)
    assert S.model.geom_contype[g] == 0
