"""Text blocks: the letters' geometry (closed, holes kept) and how the blocks go into a scene."""
import json
from collections import Counter

import numpy as np
import pytest

from blackbody.scene import components as C

T = pytest.importorskip('blackbody.ui.textmesh')


def _ring(cx, cy, r, n, ccw=True):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])
    return p if ccw else p[::-1]


def _closed(tris):
    edges = Counter()
    for a, b, c in tris:
        for x, y in ((a, b), (b, c), (c, a)):
            edges[(int(x), int(y))] += 1
    return all(n == 1 and edges.get((y, x), 0) == 1 for (x, y), n in edges.items())


def test_letter_with_holes_is_a_closed_solid():
    # an "8": one outline, two holes, given in any winding
    outer = np.array([[0, 0], [1, 0], [1, 2], [0, 2]], float)
    holes = [_ring(0.5, 0.5, 0.25, 12), _ring(0.5, 1.5, 0.25, 12, ccw=False)]
    groups = T.faces([outer[::-1]] + holes)
    assert len(groups) == 1 and len(groups[0][1]) == 2
    o, hs = groups[0]
    V = np.concatenate([o] + hs)
    area = sum(0.5 * T._cross(V[a], V[b], V[c]) for a, b, c in T.triangulate(o, hs))
    want = 2.0 - sum(abs(T._area(h)) for h in holes)
    assert area == pytest.approx(want, rel=1e-9)
    v, t = T.extrude(groups, 0.1)
    assert _closed(t)
    p0, p1, p2 = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    vol = np.einsum('ij,ij->i', p0, np.cross(p1, p2)).sum() / 6
    assert vol == pytest.approx(want * 0.1, rel=1e-9)   # wound outwards: positive volume


def test_concave_outline_and_island_in_a_hole():
    # a C shape (concave) and, separately, a ring with a dot inside its hole (an outline in a hole is its own face)
    c = np.array([[0, 0], [1, 0], [1, 0.3], [0.3, 0.3], [0.3, 0.7], [1, 0.7], [1, 1], [0, 1]], float)
    ring_o, ring_h, dot = _ring(3, 0.5, 0.5, 16), _ring(3, 0.5, 0.35, 16, ccw=False), _ring(3, 0.5, 0.1, 8)
    groups = T.faces([c, ring_o, ring_h, dot])
    assert len(groups) == 3
    v, t = T.extrude(groups, 0.05)
    assert _closed(t)
    for o, hs in groups:
        V = np.concatenate([o] + hs)
        assert all(T._cross(V[a], V[b], V[cc]) > -1e-12 for a, b, cc in T.triangulate(o, hs))


def _fake_text_mesh(tmp_path, text='FIRE', height=0.4):
    v, t = T.extrude(T.faces([np.array([[-0.6, 0], [0.6, 0], [0.6, height], [-0.6, height]], float)]), 0.1)
    path = tmp_path / 'FIRE_test.obj'
    path.write_text(''.join(f'v {x} {y} {z}\n' for x, y, z in v) + ''.join(f'f {a + 1} {b + 1} {c + 1}\n' for a, b, c in t))
    path.with_suffix('.json').write_text(json.dumps({'text': text, 'font': 'Arial', 'bold': False, 'italic': False,
                                                     'height': height, 'depth': 0.1, 'extent': [1.2, height, 0.1], 'triangles': len(t)}))
    return str(path)


def test_burning_text_block(tmp_path):
    mesh = _fake_text_mesh(tmp_path)
    sc = C.new_scene('fire', 'person')
    added, notes = C.add(sc, 'text_fire', mesh=mesh)
    assert [k for k, _ in added] == ['emitter', 'collider']
    e, c = sc.emitters[0], sc.colliders[0]
    assert e['mesh'] == c['mesh'] == mesh and c['name'] == '“FIRE”' and e['name'] == '“FIRE” fire'
    assert sc.links == [{'child': ['emitter', e['name']], 'parent': ['collider', c['name']], 'offset': [0.0, 0.0, 0.0], 'shape': True}]
    assert sc.data['domain']['mesh_resolution'] >= 96
    # the fire keeps to the letters when they are moved, stretched and turned
    c['position'], c['size'], c['yaw'] = (0.3, 0.0, 0.2), (1.5, 1.0, 1.0), 20.0
    sc.apply_links()
    assert tuple(e['position']) == (0.3, 0.0, 0.2) and tuple(e['size']) == (1.5, 1.0, 1.0) and e['yaw'] == 20.0
    lo, hi = C._extent('collider', c)   # read from the mesh, not guessed
    assert hi[0] - lo[0] == pytest.approx(1.2 * 1.5, rel=0.3)


def test_text_blocks_in_other_scenes(tmp_path):
    mesh = _fake_text_mesh(tmp_path, 'SPLASH', 0.25)
    sc = C.new_scene('liquid', 'person')
    C.add(sc, 'text_water', mesh=mesh)
    e = sc.emitters[0]
    assert e['shape'] == 'mesh' and e['liquid_mode'] == 'fill' and e['position'][1] > 0.1   # hangs above the ground
    with pytest.raises(ValueError):
        C.add(C.new_scene('cloud', 'person'), 'text', mesh=mesh)
