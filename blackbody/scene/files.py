"""The files a scene's objects use: meshes, volumes and their sequences, USD files and the notes beside a Text
block's letters. Saving a block and packing a project both copy them."""
from __future__ import annotations

from pathlib import Path

FILE_KEYS = {'emitter': ('mesh', 'volume'), 'collider': ('mesh',), 'fabric': ('mesh',), 'light': (), 'matter': ('mesh',)}


def object_sources(kind, d):
    """[(key, source)] of the files an object uses (only the ones its shape reads), built-in meshes left out."""
    out = []
    shape = d.get('shape')
    for k in FILE_KEYS.get(kind, ()):
        v = d.get(k)
        if not v or not isinstance(v, str) or v.startswith('builtin:'):
            continue
        if (k == 'mesh' and shape != 'mesh') or (k == 'volume' and shape != 'volume'):
            continue
        out.append((k, v))
    return out


def source_files(source):
    """The files on disk behind a resolved source (absolute paths): every frame of a numbered sequence, the USD file
    of a prim, and a Text block's .json beside its OBJ."""
    from ..engine.mesh import is_numbered, sequence_files, split_source
    f, prim = split_source(source)
    if is_numbered(f):
        return [Path(p) for _, p in sorted(sequence_files(f).items())]
    p = Path(f)
    if not p.exists():
        return []
    out = [p]
    side = p.with_suffix('.json')
    if p.suffix.lower() == '.obj' and side.exists():
        out.append(side)
    return out


def rebase(source, folder):
    """The same source with its file(s) in another folder: 'a/b/fire.####.obj#/x' -> 'folder/fire.####.obj#/x'."""
    from ..engine.mesh import split_source
    f, prim = split_source(source)
    new = str(Path(folder) / Path(f).name)
    return new + ('#' + prim if prim else '')
