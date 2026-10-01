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
    start = {k: v['pos'].copy() for k, v in S.piece_poses().items()}
    for f in range(scene.start + 1, scene.start + 1 + int(round(seconds * scene.fps))):
        S.advance(scene, f, 1.0 / scene.fps, 1)
    moved = {k: np.linalg.norm(v['pos'] - start[k], axis=1) for k, v in S.piece_poses().items()}
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


def test_breaking_is_deterministic():
    S1, m1 = run(scene_of(WALL, ball(4.0)), 1.0)
    S2, m2 = run(scene_of(WALL, ball(4.0)), 1.0)
    assert len(S1.breaks) == len(S2.breaks)
    assert np.array_equal(m1[0], m2[0])
