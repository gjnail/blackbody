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


def test_chunks_stand_whole_at_rest_and_break_when_hit():
    # (every weld was as stiff as the next, so the solver loaded a sliver of a bond as much as any, past what its area
    # holds: the Concrete pillar dropped three welds and an anchor as the shot began, puffing dust, nothing near it)
    pillar = dict(name='Concrete pillar', shape='box', position=(0.0, 1.0, 0.0), size=(0.15, 1.0, 0.15), material='concrete',
                  breakable=True, pieces=30)
    S, moved = run(scene_of(pillar), 1.5)
    assert not S.breaks and moved[0].max() < 0.002
    ball = dict(name='Ball', shape='sphere', position=(-1.0, 1.4, 0.0), size=(0.2, 0.2, 0.2), dynamic=True, material='steel',
                start_velocity=(8.0, 0.0, 0.0))
    S, moved = run(scene_of(pillar, ball), 1.5)
    assert len(S.breaks) > 15 and (moved[0] > 0.1).sum() > 5          # (hit hard, it breaks into chunks)


def test_a_stone_mesh_stands_and_lies_whole():
    # (a mesh's welds too; and standing, it was glued to the ground by every face of its pieces that looked down, under
    # its seat and its arms as well as its feet, each a sliver)
    import os
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'blackbody', 'assets', 'meshes', 'armchair.obj'))
    from blackbody.engine.mesh import load_mesh
    v, _t = load_mesh(path)
    low = float(np.asarray(v, float)[:, 1].min()) * 0.6
    chair = dict(name='Chair', shape='mesh', mesh=path, size=(0.6, 0.6, 0.6), material='stone', breakable=True, pieces=24)
    S, _ = run(scene_of(dict(chair, position=(0.0, -low + 0.001, 0.0), dynamic=True)), 1.0)
    assert not S.breaks
    S, _ = run(scene_of(dict(chair, position=(0.0, -low, 0.0))), 1.0)
    assert not S.breaks


def test_a_pane_hit_at_its_corner_is_cut_into_pieces_the_physics_takes():
    # (a hit near a corner cut wedges against the frame a tenth of a millimetre thin, and moved in by the whole gap for
    # the physics, small splinters came out as thin: MuJoCo refused the model, mjMINVAL)
    from blackbody.engine.fracture import inset, web
    from blackbody.engine.solids import GAP
    for size, n, at in (((0.5, 0.5, 0.004), 40, (-0.995, -0.95)), ((0.08, 0.1, 0.002), 40, (0.995, 0.95)),
                        ((0.3, 0.4, 0.003), 60, (-0.9, 0.98))):
        hit = np.round([at[0] * size[0], at[1] * size[1], 0.0], 4)
        f = web(size, n, np.random.default_rng(7919), hit)
        assert all(p.volume >= 4e-4 * p.face_area.max() for p in f.pieces)
        assert connected(f)
        for p in f.pieces:
            pl = p.planes.copy()
            pl[p.inner, 3] -= GAP
            q, _ = make_piece(pl, p.inner)
            v = inset(p, GAP)
            if q is not None and q.volume >= min(2.5e-4, 0.5 * p.volume / p.face_area.max()) * q.face_area.max():
                assert np.array_equal(v, q.verts)            # (one the whole gap leaves thick enough is moved in by all of it)
            else:
                assert len(v) >= 4
        sc = scene_of(dict(name='Pane', shape='box', position=(0.0, 1.2, 0.0), size=size, material='glass', breakable=True,
                           fracture='shards', pieces=n, held='edges'))
        S = Solids()
        S._rehearse = lambda scene, idx, layout, hit=hit: {0: hit}
        S.configure(sc, ((96, 96, 96), 12.0 / 96, (-6.0, 0.0, -6.0)))     # (MuJoCo takes it)
        assert len(S.sets[0].bodies) == len(f.pieces)


def test_breaking_is_deterministic():
    S1, m1 = run(scene_of(WALL, ball(4.0)), 1.0)
    S2, m2 = run(scene_of(WALL, ball(4.0)), 1.0)
    assert len(S1.breaks) == len(S2.breaks)
    assert np.array_equal(m1[0], m2[0])


LAYOUT = ((96, 96, 96), 12.0 / 96, (-6.0, 0.0, -6.0))
PANE = dict(name='Pane', shape='box', position=(0.0, 0.9, 0.0), size=(0.5, 0.5, 0.004), breakable=True, fracture='shards',
            material='glass', pieces=40, held='edges')
THROWN = dict(name='Stone', shape='sphere', position=(0.25, 0.9, -1.8), size=(0.04, 0.04, 0.04), dynamic=True,
              material='stone', start_velocity=(0.0, 1.0, 9.0))       # (it strikes the pane off centre at frame 6)


def test_where_a_thing_is_hit_does_not_depend_on_how_fast_the_machine_is(monkeypatch):
    # (the rehearsal that finds where a breakable is hit, which its cracks crowd round, stopped after 15 s of the
    # machine's time as well: on a slow or busy machine bottle_shoot's last bottle was never hit in it, and broke
    # differently from one run to the next)
    import time
    from blackbody.engine import solids as SO
    sc = scene_of(PANE, THROWN)
    idx = Solids.wanted(sc)
    SO._HITS.clear()
    hit = Solids()._rehearse(sc, idx, LAYOUT)
    assert list(hit) == [0] and abs(hit[0][0] - 0.25) < 0.02 and abs(hit[0][1]) < 0.03
    clock = [0.0]

    def slow():                     # a machine a thousand times slower
        clock[0] += 1000.0
        return clock[0]
    monkeypatch.setattr(time, 'perf_counter', slow)
    SO._HITS.clear()
    again = Solids()._rehearse(sc, idx, LAYOUT)
    assert list(again) == [0] and np.array_equal(again[0], hit[0])


def test_a_rehearsal_is_bounded_by_its_work_and_kept_for_its_scene(monkeypatch):
    from blackbody.engine import solids as SO
    from blackbody.engine.solids import _HITS
    sc = scene_of(PANE, THROWN)
    idx = Solids.wanted(sc)
    _HITS.clear()
    full = Solids()._rehearse(sc, idx, LAYOUT)
    work = SO.REHEARSE_WORK
    monkeypatch.setattr(SO, 'REHEARSE_WORK', 3000)   # (a big scene's worth: it stops before the stone arrives)
    assert Solids()._rehearse(sc, idx, LAYOUT) == {}
    monkeypatch.setattr(SO, 'REHEARSE_WORK', work)
    back = Solids()._rehearse(sc, idx, LAYOUT)
    assert len(_HITS) == 2 and np.array_equal(back[0], full[0])     # (each kept by what it ran to)
    # a curve's key moved is another scene: the model, and where things are hit, are not kept for the old keys
    k = Solids._key(sc, idx, LAYOUT)
    sc.set_key(('collider', 1, 'position'), sc.start, (0.25, 0.9, -1.8))
    k1 = Solids._key(sc, idx, LAYOUT)
    sc.set_key(('collider', 1, 'position'), sc.start, (-0.25, 0.9, -1.8))     # (the same curve, changed in place)
    assert len({k, k1, Solids._key(sc, idx, LAYOUT)}) == 3
    moved = Solids()._rehearse(sc, idx, LAYOUT)
    assert abs(moved[0][0] + 0.25) < 0.02


def test_a_model_cut_otherwise_has_another_fingerprint(monkeypatch):
    # (the disk cache's signature carries it: frames simulated with the objects cut otherwise, by older code or round
    # another hit, are simulated again rather than shown with these pieces)
    from blackbody.engine import fracture as F
    from blackbody.engine import solids as SO

    def fingerprint():
        SO._HITS.clear()
        SO._FRACTURES.clear()
        S = Solids()
        S.configure(scene_of(PANE, THROWN), LAYOUT)
        return S.fingerprint()
    a = fingerprint()
    assert fingerprint() == a
    monkeypatch.setattr(F, 'SLIVER_BOND', 0.0)        # (as before slivers of contact were dropped: more welds)
    b = fingerprint()
    monkeypatch.undo()
    monkeypatch.setattr(SO, 'REHEARSE_WORK', 3000)    # (a rehearsal stopped before the stone came: cut round nothing)
    c = fingerprint()
    monkeypatch.undo()
    SO._FRACTURES.clear()                             # (none of these cuts kept for other tests)
    assert len({a, b, c}) == 3


# ---- breaking and burning (Breakable and Burnable: each piece burns on its own) ----------------------------------------

SPREAD = dict(catch_temp=0.3, catch_time=0.4, creep=0.06, burn_time=2.0, smoulder=3.0, fuel=8.0, heat=0.5, smoke=1.0,
              smoulder_smoke=1.0, burn_speed=1.0)
POST = dict(name='Post', shape='box', position=(0.0, 0.8, 0.0), size=(0.07, 0.8, 0.07), material='wood', breakable=True,
            burnable=True, pieces=14, held='base')


def burning_post(**kw):
    sc = scene_of(dict(POST, **kw))
    sc.data['spread']['enabled'] = True
    S = Solids()
    S.configure(sc, ((96, 96, 96), 12.0 / 96, (-6.0, 0.0, -6.0)))
    S.reset()
    return sc, S


def flames(S, lo=-1.0, hi=0.3):
    """The gas temperature at fire_points(): flames (field temperature 1) round the post from height lo to hi."""
    pts, own = S.fire_points()
    return np.where((pts[:, 1] >= lo) & (pts[:, 1] < hi), 1.0, 0.0), own


def burn_for(sc, S, seconds, speed, lo=-1.0, hi=0.3, fall=True, every=None):
    sp = dict(SPREAD, burn_speed=speed)
    f0 = sc.start + 1
    for i, f in enumerate(range(f0, f0 + int(seconds * 24))):
        T, own = flames(S, lo, hi)
        S.burn(1.0 / 24.0, T, own, sp)
        if fall:
            S.advance(sc, f, 1.0 / 24.0, 1)
        if every is not None:
            every(i + 1)


def test_wood_catches_in_flames_and_chars_at_its_charring_rate():
    from blackbody.engine import wood_fire as WF
    sc, S = burning_post()
    ps = S.sets[0]
    assert ps.wood is not None and ps.wstate is not None
    # in its flames (about 50 kW/m^2) pine catches within ten seconds or so; then chars at about 0.65 mm a minute
    caught = []
    burn_for(sc, S, 20.0, 1.0, hi=9.0, fall=False, every=lambda i: caught.append((ps.fire[:, 1] >= 1.0).mean()))
    first = (np.array(caught) >= 1.0).argmax() / 24.0
    assert 2.0 < first < 12.0, first
    q = WF.flux(1.0)
    assert 40.0e3 < q < 60.0e3
    rate = ps.wood.char_rate * np.sqrt(np.clip(q / WF.Q_REF, 0.5, 2.0))
    lit = ps.wstate['lit']
    assert np.allclose(ps.wstate['char'], rate * lit, rtol=0.02)
    assert 0.5 / 60000.0 < ps.wood.char_rate < 0.8 / 60000.0


def test_a_flame_spreads_up_wood_far_faster_than_down():
    sc, S = burning_post()
    ps = S.sets[0]
    y = S.data.xipos[ps.bodies][:, 1]
    mid = np.argsort(np.abs(y - 0.8))[0]
    ps.spots[mid * 54:(mid + 1) * 54, 1] = 1.0  # (one piece in the middle alight, no flames round it)
    for _ in range(int(24 * 40)):
        pts, own = S.fire_points()
        T = np.where(own[:, 1] == mid, 1.0, 0.0)   # (its own flames, and nothing more)
        S.burn(1.0 / 24.0, T, own, dict(SPREAD, burn_speed=5.0))
    al = ps.fire[:, 1] >= 1.0
    up, down = al[y > y[mid] + 0.01].sum(), al[y < y[mid] - 0.01].sum()
    assert up >= 3 and up > 2 * down, (up, down)


def test_wood_goes_out_where_too_little_heat_reaches_it_and_glows():
    from blackbody.engine import wood_fire as WF
    sc, S = burning_post()
    ps = S.sets[0]
    burn_for(sc, S, 0.5, 20.0, hi=9.0, fall=False)          # (all of it alight in its flames)
    assert (ps.fire[:, 1] >= 1.0).all()
    burn_for(sc, S, 2.0 * max(WF.SUSTAIN_TIME / 20.0, WF.SUSTAIN_GAS), 20.0, lo=9.0, hi=9.0, fall=False)   # (then no flames)
    assert not (ps.fire[:, 1] >= 1.0).any() and (ps.fire[:, 3] > 0.0).all() and (ps.fire[:, 0] > 0.0).all()


def test_a_board_burning_on_one_face_chars_through_its_whole_thickness():
    from blackbody.engine import wood_fire as WF
    wf = WF.props('wood', 500.0)
    # (a spot alight in flames, and its twin straight through the 19 mm board on the other face, in the cold)
    X = np.zeros((2, 4))
    X[:, 0], X[:, 2], X[0, 1] = 1.0, 1.0, 1.0
    st = {k: np.zeros(2) for k in ('char', 'lit', 'low')}
    args = (np.array([1.0, 0.0]), np.zeros((2, 3)), np.zeros((0, 2), np.int64), np.zeros(0), np.full(2, 0.019))
    t = 0
    while st['char'][0] < 0.6 * 0.019:
        WF.step_spots(wf, X, st, *args, 1.0, 1.0, np.array([1, 0]))
        t += 1
    assert (X[:, 2] < 1.5).all()                        # past half of it from one face, not through (from both, it would be)
    assert np.isclose(X[0, 0], X[1, 0]) and 0.3 < X[0, 0] < 0.45      # (what is left of it there, the same either side)
    while X[0, 2] < 1.5:
        WF.step_spots(wf, X, st, *args, 1.0, 1.0, np.array([1, 0]))
        t += 1
    assert 0.0185 < st['char'][0] < 0.0195 and st['char'][1] == 0.0
    assert X[1, 2] >= 1.5 and X[1, 1] >= 1.0           # through: spent on both faces, its char showing on the cold one
    assert 20.0 < t / 60.0 < 35.0                       # (about half an hour in flames, at its charring rate)


def test_wood_is_drawn_with_its_char_running_on_across_the_seams_between_its_boards():
    sc = scene_of(dict(name='Wall', shape='box', position=(0.0, 0.97, 0.0), size=(0.6, 0.95, 0.0095), material='wood',
                       breakable=True, burnable=True, fracture='splinters', pieces=8, held='base'))
    sc.data['spread']['enabled'] = True
    S = Solids()
    S.configure(sc, ((96, 96, 96), 12.0 / 96, (-6.0, 0.0, -6.0)))
    S.reset()
    ps = S.sets[0]
    assert len(ps.spot_blend) > 0
    k = int(ps.pairs[0, 0])
    ps.spots[k * 54:(k + 1) * 54, 1] = 1.5              # (one board's spots alight, the rest cold)
    drawn = S._spots_drawn(ps)
    piece = np.arange(len(ps.spots)) // 54
    nb = set(int(x) for x in ps.pairs[(ps.pairs == k).any(1)].ravel()) - {k}
    beside = np.isin(piece, list(nb))
    assert (drawn[beside, 1] > 0.05).any()              # its char runs on across the seams to the boards beside it
    assert (drawn[~beside & (piece != k), 1] == 0.0).all()    # (and no further)
    assert (ps.spots[piece != k, 1] == 0.0).all()       # (where it burns is still each spot's own)


def test_a_burnable_breakable_burns_piece_by_piece_and_falls_when_its_foot_burns_through():
    sc, S = burning_post()
    ps = S.sets[0]
    assert ps.burnable and S.burning and (ps.fire[:, 0] == 1.0).all()
    y = S.data.xipos[ps.bodies][:, 1]
    burn_for(sc, S, 1.0, 60.0, hi=0.3)
    alight = (ps.fire[:, 1] >= 1.0) & (ps.fire[:, 0] > 0.0)
    assert alight[y < 0.3].all() and not alight[y > 0.9].any()   # its foot caught in the flames, its top not yet
    burn_for(sc, S, 24.0, 600.0, hi=9.0)
    assert (ps.fire[:, 2] >= 1.5).all()                              # in its flames it burnt through
    assert S.data.xipos[ps.bodies[~ps.gone]][:, 1].max() < 0.3        # it fell
    assert 0 < ps.gone.sum() < len(ps.gone)                          # most crumbled to ash, some charcoal is left
    assert (S.piece_poses()[0]['pos'][ps.gone][:, 1] < -1.0e3).all()  # (the ash is drawn nowhere)


def test_a_breakable_that_does_not_burn_or_is_not_near_fire_stays_whole():
    sc, S = burning_post(burnable=False)
    assert not S.burning
    sc, S = burning_post()
    burn_for(sc, S, 3.0, 100.0, lo=9.0, hi=9.0)
    ps = S.sets[0]
    assert (ps.fire[:, 1] == 0.0).all() and S.data.xipos[ps.bodies][:, 1].max() > 1.4     # (still standing, unlit)


def test_burning_pieces_feed_the_fire_and_their_glue_weakens_with_their_char():
    sc, S = burning_post()
    ps = S.sets[0]
    burn_for(sc, S, 0.5, 60.0, fall=False)
    pts, val = S.fuel_points(SPREAD)
    X = ps.spots
    alight = ((X[:, 1] >= 1.0) & (X[:, 0] > 0.0) & (X[:, 2] < 1.5)).sum()       # (wood: its flames off its burning spots)
    assert len(pts) == alight > 0 and (val[:, 0] > 0.0).all() and (val[:, 1] == 0.5).all()
    burn_for(sc, S, 3.0, 600.0, fall=False)
    W = S._w
    char = 1.0 - ps.fire[:, 0]
    worst = np.maximum(char[W['first']], np.where(W['other'] >= 0, char[np.maximum(W['other'], 0)], 0.0))
    assert np.allclose(W['strength'], W['strength0'] * (1.0 - worst) ** 2)
    assert W['strength'].min() < 0.5 * W['strength0'].max()


def test_burning_is_kept_with_the_state_and_undone_by_a_reset():
    sc, S = burning_post()
    ps = S.sets[0]
    burn_for(sc, S, 16.0, 800.0, hi=9.0)
    assert ps.gone.any()
    st = S.state()
    fire, gone, char = ps.fire.copy(), ps.gone.copy(), ps.wstate['char'].copy()
    S.reset()
    assert not ps.gone.any() and (ps.fire[:, 0] == 1.0).all() and np.allclose(S._w['strength'], S._w['strength0'])
    assert (ps.wstate['char'] == 0.0).all()
    g = int(S.model.body_geomadr[int(ps.bodies[np.nonzero(gone)[0][0]])])
    assert S.model.geom_contype[g] != 0                              # (the ash is whole again)
    assert S.load_state(st)
    assert np.allclose(ps.fire, fire) and np.array_equal(ps.gone, gone) and np.allclose(ps.wstate['char'], char)
    assert np.array_equal(ps.fire, fire) and np.array_equal(ps.gone, gone)
    assert S.model.geom_contype[g] == 0
