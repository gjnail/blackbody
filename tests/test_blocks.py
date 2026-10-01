"""Your own blocks: save objects from a scene, add them to another, share the file."""
import json
import zipfile

import pytest

from blackbody.scene import blocks, components as C
from blackbody.scene.anim import Curve


@pytest.fixture
def block_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(blocks, 'DIR', str(tmp_path / 'blocks'))
    return tmp_path


def _obj(path):
    path.write_text('v -0.5 0 -0.1\nv 0.5 0 -0.1\nv 0 0.6 0.1\nv 0 0 0.3\nf 1 2 3\nf 1 3 4\nf 2 4 3\nf 1 4 2\n')
    return str(path)


def test_save_and_add_a_block(block_dir):
    sc = C.new_scene('fire', 'person')
    mesh = _obj(block_dir / 'sign.obj')
    C.add(sc, 'burner', at=(0.8, 0.5))
    i = sc.add_collider(name='Sign', shape='mesh', mesh=mesh, position=(1.0, 0.0, 0.5), burnable=True)
    j = sc.add_emitter(name='Sign fire', shape='mesh', mesh=mesh, position=(1.0, 0.0, 0.5), fuel=8.0)
    sc.links.append({'child': ['emitter', 'Sign fire'], 'parent': ['collider', 'Sign'], 'offset': [0, 0, 0], 'shape': True})
    sc.emitters[j]['fuel'] = Curve([[1.0, 2.0, 'smooth'], [24.0, 10.0, 'smooth']])
    path = blocks.save(sc, [('collider', i)], 'Burning sign', 'A sign on fire')   # the fire comes along: it is attached
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        data = json.loads(z.read('block.json'))
    assert 'block.json' in names and any(n.endswith('sign.obj') for n in names)
    assert data['need'] == 'fire' and data['scene'] == {'spread': {'enabled': True}}
    assert [e['name'] for e in data['objects']['emitters']] == ['Sign fire'] and len(data['links']) == 1
    # kept about its own base: the middle of its footprint
    assert abs(data['objects']['colliders'][0]['position'][0]) < 1e-6

    keys = [c.key for c in blocks.all_blocks()]
    assert keys == ['user:Burning_sign']
    other = C.new_scene('liquid', 'person')
    C.add(other, 'pour')   # water in it already: with the fire it becomes a fire-and-liquid scene
    added, notes = C.add(other, 'user:Burning_sign', at=(-0.3, 0.2))
    assert other.kind == 'both' and [k for k, _ in added] == ['emitter', 'collider']
    e, c = other.emitters[added[0][1]], other.colliders[added[1][1]]
    lo, hi = C._extent('collider', c)   # the middle of its footprint is where it was dropped
    assert ((lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2) == pytest.approx((-0.3, 0.2), abs=0.06)
    assert c['position'][1] == 0.0
    assert e['emits'] == 'fire' and isinstance(e['fuel'], Curve)
    assert c['mesh'] != mesh and c['mesh'].endswith('sign.obj') and __import__('os').path.exists(c['mesh'])
    assert other.links[-1]['child'] == ['emitter', e['name']] and other.links[-1]['parent'] == ['collider', c['name']]
    # added twice: the second copy's names and links are its own
    added2, _ = C.add(other, 'user:Burning_sign')
    e2 = other.emitters[added2[0][1]]
    assert e2['name'] == 'Sign fire 2' and other.links[-1]['child'] == ['emitter', 'Sign fire 2']


def test_moving_things_and_liquid_blocks(block_dir):
    sc = C.new_scene('liquid', 'person')
    C.add(sc, 'pour')
    sc.emitters[0]['position'] = Curve([[1.0, (0.2, 0.8, 0.0), 'smooth'], [48.0, (0.6, 0.8, 0.0), 'smooth']])
    path = blocks.save(sc, [('emitter', 0)], 'Moving tap')
    data = blocks.read(path)
    assert data['need'] == 'liquid' and data['objects']['emitters'][0]['emits'] == 'liquid'
    fire = C.new_scene('fire', 'person')
    C.add(fire, 'burner')
    added, _ = C.add(fire, 'user:Moving_tap', at=(1.0, 0.0))
    e = fire.emitters[added[0][1]]
    assert fire.kind == 'both' and e['emits'] == 'liquid'
    xs = [k[1][0] for k in e['position'].keys]
    assert xs == pytest.approx([1.0, 1.4])   # it starts where it is dropped, and the path keeps its shape


def test_install_and_delete(block_dir):
    sc = C.new_scene('fire', 'person')
    C.add(sc, 'box')
    path = blocks.save(sc, [('collider', 0)], 'Crate', path=block_dir / 'shared.bbblock')
    key = blocks.install(path)
    assert key == 'user:shared' and C.get(key).name == 'Crate'
    assert blocks.install(path) == 'user:shared'   # the same file: kept, not copied again
    blocks.delete(key)
    assert blocks.all_blocks() == []
    with pytest.raises(KeyError):
        C.get(key)
