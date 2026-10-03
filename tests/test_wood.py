"""Wood (engine/wood.py): its species, cutting it along its grain, and how a board breaks; the stage's wood
(wood.wgsl) kept to the same species. No GPU: MuJoCo runs on its own."""
import math
import re
from pathlib import Path

import numpy as np
import pytest

from blackbody.engine.fracture import make_piece, shape_planes
from blackbody.engine.wood import ALONG, SPECIES, bond_strengths, fibres, grain_axis, is_wood, species_of
from blackbody.scene.materials import MATERIALS, PATTERNS

WGSL = Path(__file__).resolve().parents[1] / 'blackbody' / 'engine' / 'wgsl' / 'wood.wgsl'


def whole(shape, size):
    pl = shape_planes(shape, size)
    return make_piece(pl, np.zeros(len(pl), bool))[0]


def connected(frac, bonds=None):
    bonds = frac.bonds if bonds is None else bonds
    seen, todo, nbr = {0}, [0], {}
    for b in bonds:
        nbr.setdefault(b.i, []).append(b.j)
        nbr.setdefault(b.j, []).append(b.i)
    while todo:
        for j in nbr.get(todo.pop(), []):
            if j not in seen:
                seen.add(j)
                todo.append(j)
    return len(seen) == len(frac.pieces)


def test_every_species_is_a_material_with_its_own_grain():
    for k, sp in SPECIES.items():
        key = 'wood' if k == 'pine' else k
        m = MATERIALS[key]
        assert is_wood(key) and species_of(key) is sp
        assert abs(m.density - sp.density) < 1e-9 if key != 'wood' else True
        assert PATTERNS[m.pattern] == sp.pattern
    assert MATERIALS['oak'].strength > MATERIALS['wood'].strength > MATERIALS['balsa'].strength
    assert not is_wood('steel')


def test_the_stage_draws_the_same_species():
    """wood.wgsl's table: each species' pattern number, colours and grain as wood.py has them, its base colour the
    material's (the pattern is a multiple of it)."""
    src = WGSL.read_text(encoding='utf-8')
    rows = {}
    for m in re.finditer(r'case (\d+): \{ return WoodSp\((.*?)\); \}', src):
        rows[int(m.group(1))] = m.group(2)
    m = re.search(r'default: \{ return WoodSp\((.*?)\); \}', src)
    rows[1] = m.group(1)
    for k, sp in SPECIES.items():
        nums = [float(x) for x in re.findall(r'-?\d+\.?\d*(?:e-?\d+)?', rows[sp.pattern].replace('vec3<f32>', ''))]
        key = 'wood' if k == 'pine' else k
        want = [*sp.early, *sp.late, *MATERIALS[key].colour, *sp.heart, sp.ring, sp.late_share, sp.sharp, sp.rays,
                sp.pores, sp.knots, sp.figure]
        assert np.allclose(nums, want, atol=1e-3), (k, nums, want)


@pytest.mark.parametrize('shape,size', [('box', (0.6, 0.0095, 0.1)), ('box', (0.05, 1.2, 0.05)), ('box', (0.3, 0.04, 0.6)),
                                        ('cylinder', (0.15, 0.3, 0.15)), ('cylinder', (0.03, 0.6, 0.03))])
def test_wood_is_cut_along_its_grain_and_fills_itself(shape, size):
    f = fibres(shape, size, 48, seed=5)
    assert len(f.pieces) >= 12
    assert abs(f.volume - whole(shape, size).volume) < 1e-6 * whole(shape, size).volume
    assert connected(f)
    ax = grain_axis(shape, size)
    # its pieces are long along the grain
    ext = np.array([np.ptp(p.verts, axis=0) for p in f.pieces])
    others = [k for k in range(3) if k != ax]
    assert np.median(ext[:, ax] / ext[:, others].max(1)) > 1.2
    # each bond is either end grain to end grain (strong) or side by side (weaker, the wider the weaker)
    along, across = bond_strengths(SPECIES['pine'])
    for b in f.bonds:
        end = abs(float(np.asarray(b.normal)[ax])) > 0.6
        assert (b.k == along) if end else (b.k <= across + 1e-9)


def test_a_board_breaks_jagged_not_square():
    """The cuts between lengths are staggered from bundle to bundle and slanted: across a board's width they are not at
    one place, so a break steps along the grain."""
    f = fibres('box', (0.6, 0.0095, 0.1), 60, seed=2)
    ax = 0
    ends = [b for b in f.bonds if abs(float(np.asarray(b.normal)[ax])) > 0.6]
    xs = np.array([b.centre[ax] for b in ends])
    # (a square cut puts every bundle's end at the same x: here they spread)
    gaps = np.diff(np.sort(xs))
    assert len(ends) >= 10 and np.median(gaps) > 0.005
    slanted = [b for b in ends if abs(float(np.asarray(b.normal)[ax])) < 0.97]
    assert len(slanted) > 0.7 * len(ends)


def test_the_grain_runs_as_the_stage_draws_it():
    assert grain_axis('box', (1.0, 0.1, 0.2)) == 0
    assert grain_axis('box', (0.1, 1.0, 0.2)) == 1
    assert grain_axis('box', (0.2, 0.2, 0.2)) == 1          # (of equal sides y, as wood.wgsl wood_axis)
    assert grain_axis('cylinder', (0.5, 0.1, 0.5)) == 1     # (a log's slice too)


pytest.importorskip('mujoco')

from blackbody.engine.solids import Solids  # noqa: E402
from blackbody.scene.model import Scene  # noqa: E402


def drop_on_board(mass, height, material='wood'):
    """A 19 mm board of `material` across a 30 cm span between two blocks, a steel ball of `mass` kg dropped on its middle
    from `height` m: the sizes (in pieces) of what is still glued together after a second."""
    s = Scene()
    s.data['domain'].update(ground=True, open_sides=True, preroll=0.0)
    s.data['render']['fps'] = 24.0
    s.emitters = []
    s.add_collider(name='Left', shape='box', position=(-0.2, 0.2, 0.0), size=(0.05, 0.2, 0.12), material='concrete')
    s.add_collider(name='Right', shape='box', position=(0.2, 0.2, 0.0), size=(0.05, 0.2, 0.12), material='concrete')
    s.add_collider(name='Board', shape='box', position=(0.0, 0.4105, 0.0), size=(0.21, 0.0095, 0.075), material=material,
                   breakable=True, fracture='splinters', pieces=60, dynamic=True)
    r = (3 * mass / (4 * math.pi * 7850)) ** (1 / 3)
    s.add_collider(name='Weight', shape='sphere', position=(0.0, 0.421 + r + height, 0.0), size=(r, r, r), material='steel',
                   dynamic=True, density=7850.0)
    S = Solids()
    S.configure(s, ((96, 96, 96), 4.0 / 96, (-2.0, 0.0, -2.0)))
    S.reset()
    for f in range(s.start + 1, s.start + 25):
        S.advance(s, f, 1.0 / 24, 1)
    W = S._w
    n = len(S.sets[0].bodies)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for r_ in range(len(W['over'])):
        if W['over'][r_] >= 0 and W['other'][r_] >= 0:
            parent[find(int(W['first'][r_]))] = find(int(W['other'][r_]))
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted((len(g) for g in groups.values()), reverse=True)


def test_a_light_knock_leaves_a_board_whole_and_a_hard_one_snaps_it_in_two():
    assert len(drop_on_board(1.0, 0.3)) == 1
    snapped = drop_on_board(5.0, 0.5)
    assert len(snapped) >= 2 and snapped[1] >= 0.3 * sum(snapped)      # (two big halves, not a heap of sticks)


def test_oak_takes_more_than_balsa():
    assert len(drop_on_board(2.0, 0.4, 'balsa')) >= 2
    assert len(drop_on_board(2.0, 0.4, 'oak')) == 1


def _groups(S):
    W = S._w
    n = len(S.sets[0].bodies)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for r_ in range(len(W['over'])):
        if W['over'][r_] >= 0 and W['other'][r_] >= 0:
            parent[find(int(W['first'][r_]))] = find(int(W['other'][r_]))
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted((len(g) for g in groups.values()), reverse=True)


def test_a_long_plank_sags_under_a_load_as_wood_does_and_holds_it():
    # a 2 m span of 19 x 140 mm pine, 10 kg in its middle: beam theory sags it 2.6 cm (with its own weight); it holds
    s = Scene()
    s.data['domain'].update(ground=True, open_sides=True, preroll=0.0)
    s.data['render']['fps'] = 24.0
    s.emitters = []
    s.add_collider(name='Left', shape='box', position=(-1.05, 0.3, 0.0), size=(0.08, 0.3, 0.12), material='concrete')
    s.add_collider(name='Right', shape='box', position=(1.05, 0.3, 0.0), size=(0.08, 0.3, 0.12), material='concrete')
    s.add_collider(name='Plank', shape='box', position=(0.0, 0.6105, 0.0), size=(1.15, 0.0095, 0.07), material='wood',
                   breakable=True, fracture='splinters', pieces=60, dynamic=True)
    s.add_collider(name='Load', shape='box', position=(0.0, 0.745, 0.0), size=(0.12, 0.12, 0.12), material='steel',
                   dynamic=True, density=10.0 / 0.24 ** 3)
    S = Solids()
    S.configure(s, ((96, 96, 96), 4.0 / 96, (-2.0, 0.0, -2.0)))
    S.reset()
    for f in range(s.start + 1, s.start + 37):
        S.advance(s, f, 1.0 / 24, 1)
    ps = S.sets[0]
    x = S.data.xpos[ps.bodies]
    sag = 0.6105 - float(x[np.abs(x[:, 0]) < 0.25, 1].mean())
    assert 0.012 < sag < 0.05, sag
    assert len(_groups(S)) == 1


def wall_at_a_small_step():
    """A wall of 19 mm pine boards standing on its base, and a door header's small pieces 1.5 m off making the step
    short: (Solids, the wall's pieces where they start) once configured."""
    s = Scene()
    s.data['domain'].update(ground=True, open_sides=True, preroll=0.0)
    s.data['render']['fps'] = 24.0
    s.emitters = []
    s.add_collider(name='Wall', shape='box', position=(0.0, 0.97, 0.0), size=(0.905, 0.95, 0.0095), material='wood',
                   breakable=True, fracture='splinters', pieces=24, held='base')
    s.add_collider(name='Header', shape='box', position=(0.0, 1.81, 1.5), size=(0.4, 0.11, 0.0095), material='wood',
                   breakable=True, fracture='splinters', pieces=4, held='edges')
    S = Solids()
    S.configure(s, ((96, 96, 96), 4.0 / 96, (-2.0, 0.0, -2.0)))
    S.reset()
    return s, S, S.data.xpos[S.sets[0].bodies].copy()


def test_a_wall_of_boards_stands_still_at_a_small_step():
    # (its bending joints, made as soft as the wood by a weak hold on a stiff spring, rang up at a short step and burst
    # it in a tenth of a second: wood.BEND_IMPEDANCE)
    s, S, x0 = wall_at_a_small_step()
    assert S.model.opt.timestep < 1.2e-3
    for f in range(s.start + 1, s.start + 13):
        S.advance(s, f, 1.0 / 24, 1)
    W = S._w
    assert (S.data.eq_active[W['eq'][W['set'] == 0]] == 1).all()          # whole
    assert np.abs(S.data.xpos[S.sets[0].bodies] - x0).max() < 0.002      # and still


def test_an_edge_driven_into_a_log_splits_it_along_its_grain():
    def strike(v):
        s = Scene()
        s.data['domain'].update(ground=True, open_sides=True, preroll=0.0)
        s.data['render']['fps'] = 24.0
        s.emitters = []
        s.add_collider(name='Log', shape='cylinder', position=(0.0, 0.2, 0.0), size=(0.12, 0.2, 0.12), material='wood',
                       breakable=True, fracture='splinters', pieces=48, dynamic=True)
        h = 0.05
        s.add_collider(name='Wedge', shape='box', position=(0.0, 0.4 + h * 1.42 + 0.003, 0.0), size=(h, h, 0.08), roll=45.0,
                       material='steel', dynamic=True, density=3.0 / (2 * h * 2 * h * 0.16), start_velocity=(0.0, -v, 0.0))
        S = Solids()
        S.configure(s, ((96, 96, 96), 4.0 / 96, (-2.0, 0.0, -2.0)))
        S.reset()
        for f in range(s.start + 1, s.start + 13):
            S.advance(s, f, 1.0 / 24, 1)
        return _groups(S)
    # (a 3 kg wedge: at 8 m/s, an axe's blow, about 100 J, it splits the log in two; at 3 m/s it only checks its top)
    split = strike(8.0)
    assert len(split) >= 2 and split[1] >= 0.25 * sum(split)
    assert len(strike(3.0)) == 1
