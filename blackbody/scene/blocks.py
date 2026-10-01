"""Your own building blocks: things built in a scene (a burning sign, a torch rig, a dripping tap) saved as a
block, to add to any scene from Create like the built-in ones and to share as one file.

A block is a zip (.bbblock): block.json (the objects, the links between them, what the scene needs for them)
and the meshes and volumes they use, under files/. Objects are kept relative to the block's base: the middle
of their footprint, on the ground they stood on. Adding a block unpacks its files once, into a folder beside
the blocks.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

from .anim import Curve
from .files import object_sources, rebase, source_files
from .model import Scene

EXT = '.bbblock'
FORMAT = 'blackbody-block'
DIR = None          # set by the app: where your blocks live (a folder of .bbblock files)
_CACHE = {}         # path -> (mtime, Component)
LISTS = ('emitters', 'colliders', 'lights', 'fabrics', 'matter')
KIND_OF = {'emitters': 'emitter', 'colliders': 'collider', 'lights': 'light', 'fabrics': 'fabric', 'matter': 'matter'}


def folder():
    if DIR is None:
        raise ValueError('No folder for blocks has been set.')
    d = Path(DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def slug(name):
    return re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')[:40] or 'block'


def _items(scene, kind):
    return {'emitter': scene.emitters, 'collider': scene.colliders, 'light': scene.lights, 'fabric': scene.fabrics,
            'matter': scene.matter}[kind]


def with_attached(scene, sel):
    """The objects sel = [(kind, i)] plus everything attached to them (and to those), in scene order."""
    names = {(k, _items(scene, k)[i]['name']) for k, i in sel}
    grow = True
    while grow:
        grow = False
        for l in getattr(scene, 'links', None) or []:
            c, p = tuple(l['child']), tuple(l['parent'])
            if p in names and c not in names:
                names.add(c)
                grow = True
    out = []
    for kind in ('emitter', 'collider', 'fabric', 'light'):
        for i, d in enumerate(_items(scene, kind)):
            if (kind, d['name']) in names:
                out.append((kind, i))
    return out


def _shift(v, dx, dz):
    if isinstance(v, Curve):
        return Curve([[f, (x[0] + dx, x[1], x[2] + dz), it] for f, x, it in v.keys])
    return (float(v[0]) + dx, float(v[1]), float(v[2]) + dz)


def _still(scene, kind, i, d, frame):
    """A copy of an object with its animated place and size taken at a frame (for measuring it)."""
    s = dict(d)
    for k in ('position', 'end', 'size'):
        if k in s and isinstance(s[k], Curve):
            s[k] = tuple(scene.get((kind, i, k), frame))
    return s


def save(scene, sel, name, tip='', frame=None, path=None):
    """Save objects sel = [(kind, i)] of a scene (with what is attached to them) as a block. Returns its path."""
    from . import components as C
    sel = with_attached(scene, sel)
    if not sel:
        raise ValueError('Select something to save as a block.')
    frame = scene.start if frame is None else frame
    import numpy as np
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for kind, i in sel:
        if kind == 'light':
            continue
        a, b = C._extent(kind, _still(scene, kind, i, _items(scene, kind)[i], frame))
        lo, hi = np.minimum(lo, a), np.maximum(hi, b)
    if not np.isfinite(lo).all():   # lights only
        ps = np.array([scene.get((k, i, 'position'), frame) for k, i in sel], float)
        lo, hi = ps.min(0), ps.max(0)
    cx, cz = float(lo[0] + hi[0]) / 2, float(lo[2] + hi[2]) / 2
    top = float(hi[1])

    block = Scene()
    block.emitters, block.colliders, block.lights, block.fabrics = [], [], [], []
    files = []        # (name in the zip, file on disk)
    sources = {}      # resolved source -> its name in the block
    need = set()
    for kind, i in sel:
        d = copy.deepcopy(_items(scene, kind)[i])
        for k in ('position', 'end'):
            if k in d:
                d[k] = _shift(d[k], -cx, -cz)
        for k, src in object_sources(kind, d):
            real = scene.mesh_path(src)
            if real not in sources:
                sub = f'files/{len(sources)}'
                found = source_files(real)
                if not found:
                    raise ValueError(f'{d["name"]} uses {Path(src.split("#")[0]).name}, which cannot be found.')
                for p in found:
                    files.append((f'{sub}/{p.name}', p))
                sources[real] = 'block:' + rebase(real, sub).replace('\\', '/')
            d[k] = sources[real]
        if kind == 'emitter':
            if scene.kind == 'liquid':
                d['emits'] = 'liquid'
            elif scene.kind in ('fire', 'cloud'):
                d['emits'] = 'fire'
            need.add(d['emits'])
        _items(block, kind).append(d)
    names = {(k, _items(scene, k)[i]['name']) for k, i in sel}
    links = [dict(l) for l in getattr(scene, 'links', None) or []
             if tuple(l['child']) in names and tuple(l['parent']) in names]
    burnable = any(c.get('burnable') for c in block.colliders)
    if 'fire' in need and need & {'liquid', 'lava'}:
        want = 'both'
    elif need:
        want = 'lava' if need == {'lava'} else ('liquid' if need == {'liquid'} else 'fire')
    else:
        want = 'any'
    glyph = ('flame' if 'fire' in need else 'lava' if 'lava' in need else 'drop' if 'liquid' in need else
             'fabric' if block.fabrics else 'cube' if block.colliders else 'bulb')
    height = max(top - float(lo[1]), 0.1)
    room = (0.0, top + 2.5 * height, 0.0) if 'fire' in need else (0.0, 0.0, 0.0)
    objs = {k: v for k, v in block.to_dict().items() if k in LISTS}
    data = {'format': FORMAT, 'version': 1, 'name': name.strip() or 'My block', 'tip': tip.strip(), 'need': want,
            'glyph': glyph, 'room': list(room), 'scene': {'spread': {'enabled': True}} if burnable else {},
            'made_in': scene.kind, 'objects': objs, 'links': links}
    path = Path(path) if path else folder() / f'{slug(data["name"])}{EXT}'
    tmp = path.with_suffix('.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('block.json', json.dumps(data, indent=1))
        for arc, p in files:
            z.write(p, arc)
    tmp.replace(path)
    _CACHE.pop(str(path), None)
    return path


def read(path):
    with zipfile.ZipFile(path) as z:
        data = json.loads(z.read('block.json').decode('utf-8'))
    if data.get('format') != FORMAT:
        raise ValueError(f'{Path(path).name} is not a Blackbody block.')
    return data


def component(path):
    """A block file as a building block (scene/components.Component) for Create, key 'user:<file stem>'."""
    from .components import Component
    path = Path(path)
    st = path.stat().st_mtime
    hit = _CACHE.get(str(path))
    if hit and hit[0] == st:
        return hit[1]
    data = read(path)
    s = Scene.from_dict(dict(data['objects'], format='blackbody-scene'))
    objs = [(KIND_OF[lst], d) for lst in LISTS for d in getattr(s, lst)]
    n = len(objs)
    what = f'{n} thing{"s" if n != 1 else ""}: ' + ', '.join(d['name'] for _, d in objs[:4]) + ('…' if n > 4 else '')
    c = Component(f'user:{path.stem}', data.get('name', path.stem), 'Mine', data.get('tip') or what, data.get('glyph', 'cube'),
                  data.get('need', 'any'), objs, data.get('scene', {}), room=tuple(data.get('room', (0.0, 0.0, 0.0))))
    c.origin = True
    c.links = data.get('links', [])
    c.file = str(path)
    _CACHE[str(path)] = (st, c)
    return c


def all_blocks():
    """Your blocks, newest first. Unreadable files are skipped."""
    if DIR is None or not Path(DIR).exists():
        return []
    out = []
    for p in sorted(Path(DIR).glob(f'*{EXT}'), key=lambda q: q.stat().st_mtime, reverse=True):
        try:
            out.append(component(p))
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            continue
    return out


def get(key):
    if not key.startswith('user:') or DIR is None:
        return None
    p = Path(DIR) / f'{key[5:]}{EXT}'
    return component(p) if p.exists() else None


def unpacked(comp):
    """The block's objects with their files unpacked (once) and their sources pointing at them."""
    src = Path(comp.file)
    h = hashlib.sha1(src.read_bytes()).hexdigest()[:10]
    out = folder() / '.files' / f'{src.stem}-{h}'
    objs = copy.deepcopy(comp.objects)
    if any(isinstance(v, str) and v.startswith('block:') for _, d in objs for v in d.values()):
        if not out.exists():
            tmp = out.with_name(out.name + '.tmp')
            shutil.rmtree(tmp, ignore_errors=True)
            with zipfile.ZipFile(src) as z:
                z.extractall(tmp, [n for n in z.namelist() if n.startswith('files/')])
            tmp.replace(out)
        for _, d in objs:
            for k, v in list(d.items()):
                if isinstance(v, str) and v.startswith('block:'):
                    f, sep, prim = v[6:].partition('#/')
                    d[k] = str(out / f) + (sep + prim if sep else '')
    return objs


def install(path):
    """Copy a block file into your blocks (import). Returns its key."""
    path = Path(path)
    read(path)   # it is a block
    dst = folder() / path.name
    if dst.resolve() != path.resolve():
        data = path.read_bytes()
        n = 2
        while dst.exists() and dst.read_bytes() != data:   # the same block imported again is not copied twice
            dst = folder() / f'{path.stem}_{n}{EXT}'
            n += 1
        if not dst.exists():
            shutil.copy2(path, dst)
    return f'user:{dst.stem}'


def delete(key):
    p = Path(folder()) / f'{key[5:]}{EXT}'
    _CACHE.pop(str(p), None)
    p.unlink(missing_ok=True)


def shift_value(v, dx, dz):
    """A place moved across the ground (keys and all)."""
    return _shift(v, dx, dz)
