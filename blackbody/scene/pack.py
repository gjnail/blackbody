"""Pack a project: copy every file it uses (meshes and their sequences, Text and logo shapes, VDB volumes, the
HDRI, holdout passes and, if wanted, the footage) into a folder beside the project file, and point the project at
the copies by paths relative to it. The project and its folder can then move to another computer together."""
from __future__ import annotations

import filecmp
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .files import object_sources, rebase, source_files

SUBS = {'mesh': 'meshes', 'volume': 'volumes', 'environment': 'pictures', 'holdout_matte': 'holdouts', 'holdout_depth': 'holdouts',
        'footage': 'footage'}
LABELS = {'mesh': 'Meshes and shapes', 'volume': 'Volumes', 'environment': 'Environment', 'holdout_matte': 'Holdout mattes',
          'holdout_depth': 'Depth passes', 'footage': 'Footage'}


@dataclass
class Ref:
    what: str          # mesh, volume, environment, holdout_matte, holdout_depth, footage
    source: str        # as the project has it
    real: str          # resolved
    files: list = field(default_factory=list)   # on disk
    users: list = field(default_factory=list)   # names of the things that use it

    @property
    def size(self):
        return sum(p.stat().st_size for p in self.files if p.exists())


def _scenes(shot):
    return [shot] + [l for l in getattr(shot, 'layers', None) or []]


def _footage_files(path):
    from ..io.footage import find_sequence
    p = Path(path)
    if not p.exists():
        return []
    seq = find_sequence(p) if p.suffix.lower() not in ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.mxf') else None
    if not seq:
        return [p]
    nums, pattern = seq
    return [Path(pattern.format(n)) for n in nums if Path(pattern.format(n)).exists()]


def gather(shot):
    """Every file the shot uses, once each: [Ref]. Built-in meshes and missing files are left out."""
    refs = {}

    def add(what, source, real, files, user):
        if not files:
            return
        key = (what, real)
        r = refs.setdefault(key, Ref(what, source, real, list(files)))
        if user not in r.users:
            r.users.append(user)

    for sc in _scenes(shot):
        for kind, items in (('emitter', sc.emitters), ('collider', sc.colliders), ('fabric', sc.fabrics)):
            for d in items:
                for k, src in object_sources(kind, d):
                    real = sc.mesh_path(src)
                    add(k, src, real, source_files(real), d['name'])
        env = sc.data.get('lighting', {}).get('environment', '')
        if env:
            real = sc.mesh_path(env)
            add('environment', env, real, source_files(real), 'Lighting')
        for k in ('holdout_matte', 'holdout_depth'):
            src = sc.data['composite'].get(k, '')
            if src:
                real = sc.mesh_path(src)
                add(k, src, real, source_files(real), 'Composite')
        if sc.footage and sc.footage.get('path'):
            add('footage', sc.footage['path'], str(Path(sc.footage['path'])), _footage_files(sc.footage['path']), 'Footage')
    return list(refs.values())


def folder_for(project):
    project = Path(project)
    return project.parent / f'{project.stem} files'


def copy(refs, project, progress=None):
    """Copy the files of refs into the project's folder. Returns {(what, real): new source}: relative to the project
    (forward slashes), or an absolute path for footage (the project keeps a relative one beside it)."""
    project = Path(project)
    base = folder_for(project)
    out = {}
    done = 0
    total = sum(len(r.files) for r in refs) or 1
    for k, r in enumerate(refs):
        sub = base / SUBS[r.what]
        if r.files and r.files[0].resolve().parent == sub.resolve():
            dst_dir = sub   # already packed
        else:
            dst_dir = sub
            if any((dst_dir / p.name).exists() and not filecmp.cmp(p, dst_dir / p.name, shallow=False) for p in r.files):
                dst_dir = sub / str(k)   # another file of the same name is there: a folder of its own
        dst_dir.mkdir(parents=True, exist_ok=True)
        for p in r.files:
            dst = dst_dir / p.name
            if p.resolve() != dst.resolve() and not (dst.exists() and dst.stat().st_size == p.stat().st_size and filecmp.cmp(p, dst)):
                shutil.copy2(p, dst)
            done += 1
            if progress is not None and progress(done / total, p.name) is False:
                raise InterruptedError('Packing was stopped.')
        if r.what == 'footage':
            out[(r.what, r.real)] = str((dst_dir / Path(r.real).name).resolve())
        else:
            new = rebase(r.real, dst_dir)
            from ..engine.mesh import split_source
            f, prim = split_source(new)
            rel = Path(f).resolve().relative_to(project.parent.resolve()).as_posix()
            out[(r.what, r.real)] = rel + ('#' + prim if prim else '')
    return out


def apply(shot, mapping):
    """Point the shot's objects and settings at the copies (mapping from copy()). Returns how many were changed."""
    n = 0
    for sc in _scenes(shot):
        for kind, items in (('emitter', sc.emitters), ('collider', sc.colliders), ('fabric', sc.fabrics)):
            for d in items:
                for k, src in object_sources(kind, d):
                    new = mapping.get((k, sc.mesh_path(src)))
                    if new is not None and new != src:
                        d[k] = new
                        n += 1
        lit = sc.data.get('lighting', {})
        if lit.get('environment'):
            new = mapping.get(('environment', sc.mesh_path(lit['environment'])))
            if new is not None and new != lit['environment']:
                lit['environment'] = new
                n += 1
        comp = sc.data['composite']
        for k in ('holdout_matte', 'holdout_depth'):
            if comp.get(k):
                new = mapping.get((k, sc.mesh_path(comp[k])))
                if new is not None and new != comp[k]:
                    comp[k] = new
                    n += 1
        if sc.footage and sc.footage.get('path'):
            new = mapping.get(('footage', str(Path(sc.footage['path']))))
            if new is not None and new != sc.footage['path']:
                sc.footage = dict(sc.footage, path=new)
                n += 1
    return n


def human(n):
    for unit in ('bytes', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024 or unit == 'TB':
            return f'{n:.0f} {unit}' if unit == 'bytes' else f'{n:.1f} {unit}'
        n /= 1024.0

