"""Pack project: every file a shot uses is copied beside it, and the project still opens after it moves."""
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from blackbody.scene import components as C, pack
from blackbody.scene.model import Scene

OBJ = 'v -0.5 0 -0.1\nv 0.5 0 -0.1\nv 0 0.6 0.1\nv 0 0 0.3\nf 1 2 3\nf 1 3 4\nf 2 4 3\nf 1 4 2\n'


def _files(tmp):
    src = tmp / 'elsewhere'
    (src / 'seq').mkdir(parents=True)
    (src / 'chair.obj').write_text(OBJ)
    for f in (1, 2, 3):
        (src / 'seq' / f'burn.{f:04d}.obj').write_text(OBJ)
    (src / 'FIRE_x.obj').write_text(OBJ)
    (src / 'FIRE_x.json').write_text(json.dumps({'text': 'FIRE', 'font': 'Arial', 'height': 0.4, 'depth': 0.1}))
    (src / 'plate').mkdir()
    for f in (1, 2, 3):
        Image.fromarray(np.full((8, 8, 3), 40 * f, np.uint8)).save(src / 'plate' / f'shot.{f:04d}.png')
        Image.fromarray(np.full((8, 8), 255, np.uint8)).save(src / 'plate' / f'matte.{f:04d}.png')
    return src


def test_pack_and_move(tmp_path):
    src = _files(tmp_path)
    shot = C.new_scene('fire', 'person')
    shot.add_collider(name='Chair', shape='mesh', mesh=str(src / 'chair.obj'))
    shot.add_emitter(name='Burning', shape='mesh', mesh=str(src / 'seq' / 'burn.####.obj'))
    shot.add_collider(name='“FIRE”', shape='mesh', mesh=str(src / 'FIRE_x.obj'))
    shot.add_emitter(name='Fire on chair', shape='mesh', mesh=str(src / 'chair.obj'))   # the same file twice: copied once
    shot.footage = {'path': str(src / 'plate' / 'shot.0001.png')}
    shot.data['composite']['holdout_matte'] = str(src / 'plate' / 'matte.####.png')
    layer = C.new_scene('fire', 'person')
    layer.uid = 'l1'
    layer.add_collider(name='Layer chair', shape='mesh', mesh=str(src / 'chair.obj'))
    shot.layers = [layer]
    proj = tmp_path / 'work' / 'my shot.bbfire'
    proj.parent.mkdir()
    shot.save(proj)

    refs = pack.gather(shot)
    whats = sorted((r.what, len(r.files)) for r in refs)
    assert whats == [('footage', 3), ('holdout_matte', 3), ('mesh', 1), ('mesh', 2), ('mesh', 3)]
    chair = next(r for r in refs if r.real.endswith('chair.obj'))
    assert sorted(chair.users) == ['Chair', 'Fire on chair', 'Layer chair']

    mapping = pack.copy(refs, proj)
    n = pack.apply(shot, mapping)
    assert n == 7
    assert shot.colliders[0]['mesh'] == 'my shot files/meshes/chair.obj'
    assert shot.emitters[-2]['mesh'] == 'my shot files/meshes/burn.####.obj'
    assert layer.colliders[0]['mesh'] == 'my shot files/meshes/chair.obj'
    assert (pack.folder_for(proj) / 'meshes' / 'FIRE_x.json').exists()   # Edit text still works on the copy
    shot.save(proj)

    # move everything somewhere else, and lose the originals
    moved = tmp_path / 'other computer'
    shutil.copytree(proj.parent, moved)
    shutil.rmtree(proj.parent)
    shutil.rmtree(src)
    s = Scene.load(moved / 'my shot.bbfire')
    for d in s.colliders + s.emitters:
        if d.get('shape') == 'mesh':
            assert Path(s.mesh_path(d['mesh']).replace('####', '0001')).exists(), d['name']
    assert Path(s.footage['path']).exists() and moved in Path(s.footage['path']).parents
    assert Path(s.mesh_path(s.data['composite']['holdout_matte']).replace('####', '0002')).exists()
    lay = s.layers[0]
    assert Path(lay.mesh_path(lay.colliders[0]['mesh'])).exists()   # a layer finds project-relative files too


def test_pack_again_and_name_clashes(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.mkdir()
    b.mkdir()
    (a / 'rock.obj').write_text(OBJ)
    (b / 'rock.obj').write_text(OBJ.replace('0.6', '0.9'))   # another file of the same name
    shot = C.new_scene('fire', 'person')
    shot.add_collider(name='Rock A', shape='mesh', mesh=str(a / 'rock.obj'))
    shot.add_collider(name='Rock B', shape='mesh', mesh=str(b / 'rock.obj'))
    proj = tmp_path / 'p.bbfire'
    shot.save(proj)
    pack.apply(shot, pack.copy(pack.gather(shot), proj))
    ma, mb = shot.colliders[0]['mesh'], shot.colliders[1]['mesh']
    assert ma != mb and Path(shot.mesh_path(ma)).read_text() != Path(shot.mesh_path(mb)).read_text()
    # packing a packed project changes nothing
    again = pack.copy(pack.gather(shot), proj)
    assert pack.apply(shot, again) == 0
