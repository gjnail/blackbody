"""Triangle meshes as emitters and colliders.

A mesh (OBJ or STL, metres, y up) is baked once into a signed distance grid on the GPU. Small
meshes use the exact distance everywhere and the generalised winding number for inside/outside,
which copes with holes and overlapping parts. Dense meshes use a fast bake: exact distances in a
band near the surface and signed ray crossings for inside/outside. Bakes are kept on disk, so a
mesh is only baked once per machine. All meshes in use share one atlas texture that the solver
kernels sample.

A mesh source is a file path (OBJ, STL, or a greyscale image read as a heightfield: terrain), a
numbered sequence ('burning_man.####.obj', one file per frame) or a USD prim ('shot.usd#/World/Car').
Sequences and animated USD prims deform: every frame is baked on one grid that holds all of them,
and the atlas carries the frame before and after the current time, so the shape blends smoothly
between frames and its surface moves (pushing the gas) at the speed it deforms.
"""
from __future__ import annotations

import logging
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .gpu import GPU, ceil_div, Uniforms

log = logging.getLogger('blackbody.mesh')

SLICE = 256          # triangles per bake dispatch (keeps each dispatch short on slow GPUs)
SUBMIT_EVERY = 16    # dispatches per submission
FAST_ABOVE = 3000    # triangles; denser meshes use the banded bake
BAND = 6.0           # cells of exact distance around the surface in the banded bake
FAST_SLICE = 20000   # triangles per dispatch of the banded bake
BAKE_VERSION = 3     # bump when the bake changes, so stale disk caches are ignored
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.exr')
HEIGHTFIELD_MAX = 192  # heightfield images are resampled to at most this many points a side


class MeshError(ValueError):
    pass


def load_obj(path):
    """Vertices (n, 3) and triangles (m, 3) from an OBJ file. Polygons are split into fans;
    texture and normal indices, groups and materials are ignored."""
    verts, faces = [], []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith('f '):
                idx = []
                for tok in line.split()[1:]:
                    i = int(tok.split('/')[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(verts, np.float32).reshape(-1, 3), np.asarray(faces, np.int64).reshape(-1, 3)


def load_stl(path):
    """Vertices and triangles from a binary or ASCII STL file."""
    data = Path(path).read_bytes()
    if len(data) >= 84:
        n = struct.unpack_from('<I', data, 80)[0]
        if 84 + n * 50 == len(data):
            rec = np.frombuffer(data, dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]), count=n, offset=84)
            v = rec['v'].reshape(-1, 3).astype(np.float32)
            return v, np.arange(len(v), dtype=np.int64).reshape(-1, 3)
    pts = [tuple(float(x) for x in ln.split()[1:4]) for ln in data.decode('utf-8', 'replace').splitlines()
           if ln.strip().startswith('vertex')]
    v = np.asarray(pts, np.float32).reshape(-1, 3)
    return v, np.arange(len(v) - len(v) % 3, dtype=np.int64).reshape(-1, 3)


def _read_grey(path):
    """A greyscale image as float32 in [0, 1] (EXR and 16-bit images keep their precision)."""
    p = Path(path)
    if p.suffix.lower() == '.exr':
        import OpenEXR
        with OpenEXR.File(str(p)) as f:
            ch = f.channels()
            for name in ('Y', 'R', 'RGB', 'RGBA'):
                if name in ch:
                    a = np.asarray(ch[name].pixels, np.float32)
                    break
            else:
                a = np.asarray(next(iter(ch.values())).pixels, np.float32)
        if a.ndim == 3:
            a = a[..., 0]
        a = np.nan_to_num(a)
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / max(hi - lo, 1e-12)
    from PIL import Image
    im = Image.open(p)
    if im.mode in ('I;16', 'I;16B', 'I', 'F'):
        a = np.asarray(im, np.float32)
        return a / max(float(a.max()), 1.0) if im.mode != 'F' else np.clip(a, 0.0, None) / max(float(a.max()), 1e-12)
    return np.asarray(im.convert('L'), np.float32) / 255.0


def load_heightfield(path):
    """A terrain solid from a greyscale image. The image spans x and z from -0.5 to 0.5 (its top edge
    toward -z, the back); black is 0.02 and white 1.02 above a flat base at 0. Scale it with the
    collider's Size: width, height, depth in metres."""
    img = _read_grey(path)
    step = max(1, int(math.ceil(max(img.shape) / HEIGHTFIELD_MAX)))
    if step > 1:  # box-filter down to the point budget
        hh, ww = (img.shape[0] // step) * step, (img.shape[1] // step) * step
        img = img[:hh, :ww].reshape(hh // step, step, ww // step, step).mean(axis=(1, 3))
    nz, nx = img.shape
    if nx < 2 or nz < 2:
        raise MeshError(f'{Path(path).name} is too small for a heightfield')
    xs = np.linspace(-0.5, 0.5, nx, dtype=np.float32)
    zs = np.linspace(-0.5, 0.5, nz, dtype=np.float32)
    X, Z = np.meshgrid(xs, zs)
    top = np.stack([X, 0.02 + img.astype(np.float32), Z], -1).reshape(-1, 3)
    bot = np.stack([X, np.zeros_like(X), Z], -1).reshape(-1, 3)
    verts = np.concatenate([top, bot])
    nb = nx * nz
    idx = np.arange(nb).reshape(nz, nx)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel()
    tris = [np.stack([a, b, c], 1), np.stack([c, b, d], 1),                       # top, facing up
            np.stack([a + nb, c + nb, b + nb], 1), np.stack([c + nb, d + nb, b + nb], 1)]  # base, facing down
    # the four sides: walk each boundary edge and close it down to the base, facing outward
    def side(line, flip):
        t0, t1 = line[:-1], line[1:]
        q = np.stack([t0, t1, t1 + nb], 1), np.stack([t0, t1 + nb, t0 + nb], 1)
        return [x[:, ::-1] for x in q] if flip else list(q)
    tris += side(idx[0, :], False) + side(idx[-1, :], True) + side(idx[:, 0], True) + side(idx[:, -1], False)
    return verts, np.concatenate(tris).astype(np.int64)


def split_source(source):
    """(file, prim) for 'file.usd#/prim' sources, else (source, '')."""
    if '#/' in source:
        f, _, prim = source.partition('#/')
        return f, '/' + prim
    return source, ''


def is_numbered(source):
    """True for numbered sequences: 'fire.####.obj' (one # per digit)."""
    return bool(source) and '#/' not in source and '#' in Path(source).name


def frame_file(source, frame):
    """The file of a numbered sequence for a frame: 'fire.####.obj' -> 'fire.0012.obj'."""
    p = Path(source)
    n = p.name.count('#')
    i = p.name.find('#' * n)
    return str(p.with_name(p.name[:i] + f'{int(frame):0{n}d}' + p.name[i + n:]))


def sequence_files(source):
    """{frame: path} of a numbered sequence's files on disk."""
    import re
    p = Path(source)
    n = p.name.count('#')
    pat = re.compile('^' + re.escape(p.name).replace(re.escape('#' * n), f'(-?\\d{{{n},}})') + '$')
    out = {}
    if p.parent.exists():
        for q in p.parent.iterdir():
            m = pat.match(q.name)
            if m:
                out[int(m.group(1))] = str(q)
    return out


def load_mesh(path, frame=None):
    """Vertices and triangles of a mesh source (see the module notes); `frame` picks the frame of a
    sequence or the time of an animated USD prim."""
    f, prim = split_source(path)
    if prim:
        from ..io.usd import load_usd_mesh
        return load_usd_mesh(f, prim, frame)
    if is_numbered(path):
        files = sequence_files(path)
        if not files:
            raise MeshError(f'No files match {Path(path).name}')
        fr = int(math.floor(frame)) if frame is not None else min(files)
        fr = min(max(fr, min(files)), max(files))  # hold the first and last frame
        while fr not in files:  # a gap in the numbering: use the frame before it
            fr -= 1
        path = files[fr]
    p = Path(path)
    if not p.exists():
        raise MeshError(f'Mesh not found: {p}')
    ext = p.suffix.lower()
    if ext == '.obj':
        v, t = load_obj(p)
    elif ext == '.stl':
        v, t = load_stl(p)
    elif ext in IMAGE_EXTS:
        v, t = load_heightfield(p)
    else:
        raise MeshError(f'{p.name}: meshes must be OBJ, STL, a heightfield image or a USD prim')
    if len(t) == 0 or len(v) == 0:
        raise MeshError(f'{p.name} has no triangles')
    if t.min() < 0 or t.max() >= len(v):
        raise MeshError(f'{p.name} has faces that refer to missing vertices')
    return v, t


@dataclass
class MeshSDF:
    """A baked distance grid: `data` is (nz, ny, nx) signed distances in mesh units, negative inside.
    The grid spans [bmin, bmax] with cell centres at bmin + (i + 0.5) * cell."""
    path: str
    bmin: tuple
    bmax: tuple
    dims: tuple
    data: np.ndarray
    triangles: int
    mesh_min: tuple
    mesh_max: tuple


def grid_for(vmin, vmax, resolution):
    """A grid around the mesh bounds with two cells of margin on every side."""
    vmin = np.asarray(vmin, float)
    vmax = np.asarray(vmax, float)
    ext = np.maximum(vmax - vmin, 1e-6)
    res = max(8, int(resolution))
    cell = float(ext.max()) / (res - 4)
    dims = np.maximum(np.ceil(ext / cell).astype(int) + 4, 4)
    centre = 0.5 * (vmin + vmax)
    lo = centre - dims * cell / 2
    return lo, lo + dims * cell, tuple(int(x) for x in dims), cell


def bake(gpu: GPU, verts, tris, resolution=96, path='', method=None, grid=None):
    """Signed distance grid of a mesh, computed on the GPU. `method` is 'exact', 'band' or None
    (exact for small meshes, banded for dense ones). `grid` = (lo, dims, cell) bakes onto a given
    grid (the shared grid of a deforming mesh's frames)."""
    verts = np.asarray(verts, np.float32)
    tris = np.asarray(tris, np.int64)
    vmin, vmax = verts[tris.reshape(-1)].min(0), verts[tris.reshape(-1)].max(0)
    if grid is None:
        lo, hi, dims, cell = grid_for(vmin, vmax, resolution)
    else:
        lo, dims, cell = np.asarray(grid[0], float), tuple(int(x) for x in grid[1]), float(grid[2])
        hi = lo + np.asarray(dims) * cell
    method = method or ('band' if len(tris) > FAST_ABOVE else 'exact')
    if method == 'band':
        sdf = _bake_band(gpu, verts, tris, lo, dims, cell)
    else:
        sdf = _bake_exact(gpu, verts, tris, lo, dims, cell)
    return MeshSDF(path=str(path), bmin=tuple(float(x) for x in lo), bmax=tuple(float(x) for x in hi), dims=dims,
                   data=sdf, triangles=int(len(tris)), mesh_min=tuple(float(x) for x in vmin),
                   mesh_max=tuple(float(x) for x in vmax))


def _corners(verts, tris):
    corners = np.zeros((len(tris), 3, 4), np.float32)
    corners[:, :, :3] = verts[tris]
    return corners


def _bake_band(gpu, verts, tris, lo, dims, cell):
    """Exact distance within BAND cells of the surface, inside/outside from signed ray crossings."""
    nx, ny, nz = dims
    nvox = nx * ny * nz
    corners = _corners(verts, tris)
    tri_buf = gpu.buffer(max(16, corners.nbytes), 'mesh-tris')
    gpu.write_buffer(tri_buf, corners.reshape(-1, 4))
    dist = gpu.buffer(nvox * 4, 'mesh-dist')
    hits = gpu.buffer(nvox * 4, 'mesh-hits')
    out = gpu.buffer(nvox * 4, 'mesh-sdf')
    binds = ['rbuf', 'buf', 'buf', 'buf']
    k_clear = gpu.kernel('mesh_band.wgsl', binds, 'clear', workgroup=(64, 1, 1))
    k_band = gpu.kernel('mesh_band.wgsl', binds, 'band', workgroup=(64, 1, 1))
    k_cross = gpu.kernel('mesh_band.wgsl', binds, 'crossings', workgroup=(8, 8, 1))
    k_finish = gpu.kernel('mesh_band.wgsl', binds, 'finish', workgroup=(8, 8, 1))
    res = [tri_buf, dist, hits, out]
    ntri = len(tris)
    try:
        b = gpu.batch()
        g = ceil_div(nvox, 64)
        base = lambda first=0, count=0, gx=0: Uniforms().v4(nx, ny, nz, BAND).v4(*lo, cell).v4(first, count, gx)
        b.run(k_clear, res, base(), groups=(min(g, 65535), ceil_div(g, 65535), 1))
        first = 0
        while first < ntri:
            count = min(FAST_SLICE, ntri - first)
            gx = min(count, 65535)
            b.run(k_band, res, base(first, count, gx), groups=(gx, ceil_div(count, gx), 1))
            b.run(k_cross, res, base(first, count), (ny, nz, 1))
            first += count
            b.submit(restart=True)
            gpu.sync()
        b.run(k_finish, res, base(), (ny, nz, 1))
        b.submit()
        data = np.frombuffer(gpu.read_buffer(out), np.float32).reshape(nz, ny, nx).copy()
    finally:
        for buf in res:
            buf.destroy()
    return data


def _bake_exact(gpu, verts, tris, lo, dims, cell):
    """Exact distance everywhere; inside where the generalised winding number is over one half."""
    nx, ny, nz = dims
    corners = _corners(verts, tris)
    tri_buf = gpu.buffer(max(16, corners.nbytes), 'mesh-tris')
    gpu.write_buffer(tri_buf, corners.reshape(-1, 4))
    nvox = nx * ny * nz
    acc = gpu.buffer(nvox * 8, 'mesh-acc')
    k = gpu.kernel('mesh_bake.wgsl', ['rbuf', 'buf'], workgroup=(4, 4, 4))
    ntri = len(tris)
    first = 0
    b = gpu.batch()
    try:
        n_disp = 0
        while first < ntri:
            count = min(SLICE, ntri - first)
            u = Uniforms().v4(nx, ny, nz, count).v4(*lo, cell).v4(first, 1.0 if first == 0 else 0.0)
            b.run(k, [tri_buf, acc], u, dims)
            first += count
            n_disp += 1
            if n_disp % SUBMIT_EVERY == 0:
                b.submit(restart=True)
                gpu.sync()
        b.submit()
        raw = np.frombuffer(gpu.read_buffer(acc), np.float32).reshape(nz, ny, nx, 2)
    finally:
        tri_buf.destroy()
        acc.destroy()
    dist = np.sqrt(np.maximum(raw[..., 0], 0.0))
    winding = raw[..., 1] / (4.0 * math.pi)
    return np.where(np.abs(winding) > 0.5, -dist, dist).astype(np.float32)


def cache_dir():
    """Where baked meshes are kept between sessions (BLACKBODY_CACHE overrides it)."""
    env = os.environ.get('BLACKBODY_CACHE')
    if env:
        return Path(env) / 'meshes'
    if os.name == 'nt':
        return Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local')) / 'Blackbody' / 'meshes'
    return Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'blackbody' / 'meshes'


def _cache_file(path, resolution, extra=''):
    import hashlib
    f, prim = split_source(path)
    h = hashlib.sha1(Path(f).read_bytes() + prim.encode() + extra.encode()).hexdigest()[:20]
    return cache_dir() / f'{h}_{int(resolution)}_v{BAKE_VERSION}.npz'


def load_or_bake(gpu, path, resolution=96, frame=None, grid=None):
    """A mesh's distance grid from the disk cache, or baked now (and cached). For a frame of a
    deforming mesh, `path` is the frame's file (or the USD prim with `frame`) and `grid` the shared grid."""
    f = None
    try:
        extra = '' if grid is None else f'{[round(float(x), 6) for x in grid[0]]}{list(grid[1])}{grid[2]:.6g}'
        if frame is not None and split_source(path)[1]:
            extra += f'@{float(frame):.4f}'
        f = _cache_file(path, resolution, extra)
        if f.exists():
            z = np.load(f)
            m = z['meta']
            return MeshSDF(path=str(path), bmin=tuple(m[0:3]), bmax=tuple(m[3:6]), dims=tuple(int(x) for x in z['dims']),
                           data=z['data'], triangles=int(m[12]), mesh_min=tuple(m[6:9]), mesh_max=tuple(m[9:12]))
    except Exception as ex:  # a damaged or unreadable cache just means baking again
        log.info('Mesh cache unusable (%s); baking', ex)
    v, t = load_mesh(path, frame)
    sdf = bake(gpu, v, t, resolution, path, grid=grid)
    if f is not None:
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            meta = np.array(list(sdf.bmin) + list(sdf.bmax) + list(sdf.mesh_min) + list(sdf.mesh_max) + [sdf.triangles], np.float64)
            tmp = f.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, data=sdf.data, dims=np.array(sdf.dims), meta=meta)
            tmp.replace(f)
        except OSError as ex:
            log.info('Could not cache the baked mesh: %s', ex)
    return sdf


def mesh_bounds(source):
    """Bounding box (min, max) of a mesh source over all of its frames."""
    f, prim = split_source(source)
    if prim:
        from ..io.usd import usd_mesh_bounds
        return usd_mesh_bounds(f, prim)
    files = sequence_files(source).values() if is_numbered(source) else [source]
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for q in files:
        if Path(q).suffix.lower() in IMAGE_EXTS:
            v = np.array([[-0.5, 0.0, -0.5], [0.5, 1.02, 0.5]])
        elif Path(q).suffix.lower() == '.obj':
            with open(q, 'rb') as fh:
                rows = [ln.split()[1:4] for ln in fh if ln.startswith(b'v ')]
            v = np.array(rows, np.float64) if rows else np.zeros((1, 3))
        else:
            v, _ = load_mesh(q)
        lo, hi = np.minimum(lo, v.min(0)), np.maximum(hi, v.max(0))
    return lo, hi


def animated(source):
    """True for mesh sources that change from frame to frame."""
    from ..io.volume import field_animated, is_field
    if is_field(source):
        return field_animated(source)
    if is_numbered(source):
        return True
    f, prim = split_source(source)
    if prim:
        from ..io.usd import usd_prim_animated
        return usd_prim_animated(f, prim)
    return False


_deforms = {}


def mesh_deforms(source):
    """Whether a mesh source changes over time (a numbered sequence or an animated USD prim); cached."""
    if not source:
        return False
    from ..io.volume import is_field, split_field
    base = split_field(source)[0] if is_field(source) else source
    if is_numbered(base):
        return True
    f, prim = split_source(base)
    if not prim:
        return False
    try:
        key = (source, os.path.getmtime(f))
    except OSError:
        return False
    v = _deforms.get(key)
    if v is None:
        try:
            v = bool(animated(source))
        except Exception as ex:
            log.warning('Cannot read %s: %s', source, ex)
            v = False
        _deforms[key] = v
    return v


class MeshLibrary:
    """The meshes a scene uses, baked on first use and packed into one atlas texture. A deforming
    mesh contributes the two frames around the current time. Volume fields (io/volume.py: the smoke
    of a VDB, for Volume emitters) go in the same atlas; a changing one contributes its current frame."""

    def __init__(self, gpu: GPU):
        self.gpu = gpu
        self._baked = {}     # (source, frame or None, stamp, resolution) -> MeshSDF
        self._seq = {}       # (source, stamp, resolution) -> shared grid (lo, dims, cell)
        self._anim = {}      # (source, stamp) -> animated?
        self._field_ref = {}  # field source -> (density, temperature) scale shared by all its frames
        self.slots = {}      # source or (source, frame) -> (z offset, MeshSDF)
        self.errors = {}     # source -> message
        self._key = ()
        self.atlas = gpu.texture3d((1, 1, 1), 'r32float', 'mesh-atlas')
        gpu.upload(self.atlas, np.full((1, 1, 1, 1), 1.0e6, np.float32))

    @staticmethod
    def _stamp(source):
        from ..io.volume import is_field, split_field
        if is_field(source):
            source = split_field(source)[0]
        f, _ = split_source(source)
        if is_numbered(source):
            files = sequence_files(source)
            return (len(files), max((os.path.getmtime(q) for q in files.values()), default=0.0))
        return os.path.getmtime(f)

    def is_animated(self, source):
        try:
            key = (source, self._stamp(source))
        except OSError:
            return False
        if key not in self._anim:
            try:
                self._anim[key] = animated(source)
            except Exception:
                self._anim[key] = False
        return self._anim[key]

    def _grid(self, source, stamp, resolution):
        key = (source, stamp, int(resolution))
        g = self._seq.get(key)
        if g is None:
            lo, hi = mesh_bounds(source)
            lo_g, _, dims, cell = grid_for(lo, hi, resolution)
            g = self._seq[key] = (tuple(lo_g), dims, cell)
        return g

    def require(self, items, resolution=96):
        """Make sure every mesh in `items` is baked and in the atlas. Items are sources, or
        (source, frame) pairs for meshes that may deform. True if the atlas changed."""
        from ..io.volume import is_field
        want = []
        for it in items:
            src, frame = (it, None) if isinstance(it, str) else (it[0], it[1])
            if not src:
                continue
            if frame is not None and self.is_animated(src):
                fi = int(math.floor(frame))
                want += [(src, fi)] if is_field(src) else [(src, fi), (src, fi + 1)]
            else:
                want.append((src, None))
        from ..io.volume import is_field, load_field
        want = sorted(set(want), key=lambda k: (k[0], -1e18 if k[1] is None else k[1]))
        entries = []
        errors = {}
        for src, fr in want:
            try:
                stamp = self._stamp(src)
            except OSError:
                errors[src] = f'Mesh not found: {src}'
                continue
            key = (src, fr, stamp, int(resolution))
            sdf = self._baked.get(key)
            if sdf is None:
                try:
                    import time
                    t0 = time.perf_counter()
                    if is_field(src):
                        cells = max(96, min(2 * int(resolution), 192))
                        sdf, ref = load_field(src, fr, cells, self._field_ref.get(src))
                        self._field_ref.setdefault(src, ref)
                    elif fr is None:
                        sdf = load_or_bake(self.gpu, src, resolution)
                    else:
                        grid = self._grid(src, stamp, resolution)
                        f, prim = split_source(src)
                        path = src if prim else frame_file(src, self._held(src, fr))
                        sdf = load_or_bake(self.gpu, path, resolution, frame=fr if prim else None, grid=grid)
                    log.info('Mesh %s%s: %d triangles, %s grid in %.2f s', Path(split_source(src)[0]).name,
                             '' if fr is None else f' frame {fr}', sdf.triangles, sdf.dims, time.perf_counter() - t0)
                except Exception as ex:
                    errors[src] = str(ex)
                    continue
                if fr is None:
                    for k in [k for k in self._baked if k[0] == src and k[1] is None]:
                        del self._baked[k]
                elif len([k for k in self._baked if k[0] == src]) > 8:  # keep only a few frames around
                    for k in sorted((k for k in self._baked if k[0] == src and k[1] is not None),
                                    key=lambda k: abs(k[1] - fr), reverse=True)[:4]:
                        del self._baked[k]
                self._baked[key] = sdf
            entries.append(((src if fr is None else (src, fr)), sdf))
        self.errors = errors
        for p, msg in errors.items():
            log.warning('%s', msg)
        key = tuple((p, id(s)) for p, s in entries)
        if key == self._key:
            return False
        self._key = key
        self._build_atlas(entries)
        return True

    @staticmethod
    def _held(source, frame):
        """A sequence frame clamped to the files that exist (the first and last hold)."""
        files = sequence_files(source)
        if not files:
            return frame
        fr = min(max(int(frame), min(files)), max(files))
        while fr not in files:
            fr -= 1
        return fr

    def _build_atlas(self, entries):
        self.slots = {}
        if not entries:
            w = h = d = 1
            data = np.full((1, 1, 1), 1.0e6, np.float32)
        else:
            w = max(s.dims[0] for _, s in entries)
            h = max(s.dims[1] for _, s in entries)
            d = sum(s.data.shape[0] for _, s in entries)
            lim = int(self.gpu.limits.get('max-texture-dimension-3d', 2048))
            if d > lim:
                raise RuntimeError(f'Too many meshes for the GPU at this mesh detail ({d} > {lim} cells); '
                                   'lower Domain › Mesh detail or use fewer meshes.')
            data = np.full((d, h, w), 1.0e6, np.float32)
            z = 0
            for p, s in entries:
                nx, ny, _ = s.dims
                depth = s.data.shape[0]   # a field's layers are stacked along z
                data[z:z + depth, :ny, :nx] = s.data
                self.slots[p] = (z, s)
                z += depth
        self.atlas.destroy()
        self.atlas = self.gpu.texture3d((w, h, d), 'r32float', 'mesh-atlas')
        self.gpu.upload(self.atlas, data[..., None])

    def ref(self, path):
        """(m0, m1, m2) vec4 triples for a static mesh in the atlas (or the first frame of a
        deforming one), or None if it is not available."""
        slot = self.slots.get(path) if path else None
        if slot is None and path:
            frames = sorted(k[1] for k in self.slots if isinstance(k, tuple) and k[0] == path)
            slot = self.slots.get((path, frames[0])) if frames else None
        if slot is None:
            return None
        z, s = slot
        # m1.w: a field's layers past the density (1: it has a temperature)
        return (s.bmin + (float(z),), s.bmax + (float(getattr(s, 'layers', 1) - 1),), tuple(float(x) for x in s.dims) + (0.0,))

    def ref_at(self, path, frame=None):
        """(m0, m1, m2, anim) for a mesh at a (fractional) frame. anim = (atlas z offset of the next
        frame, blend toward it, 0, 0), or z offset -1 for a mesh that does not deform."""
        if frame is not None and path and (path, int(math.floor(frame))) in self.slots:
            fi = int(math.floor(frame))
            za, sa = self.slots[(path, fi)]
            if hasattr(sa, 'layers'):   # a field: the frame as it is, no blending
                return (sa.bmin + (float(za),), sa.bmax + (float(sa.layers - 1),), tuple(float(x) for x in sa.dims) + (0.0,),
                        (-1.0, 0.0, 0.0, 0.0))
            zb, _ = self.slots.get((path, fi + 1), (za, sa))
            return (sa.bmin + (float(za),), sa.bmax + (0.0,), tuple(float(x) for x in sa.dims) + (0.0,),
                    (float(zb), float(frame - fi), 0.0, 0.0))
        r = self.ref(path)
        return None if r is None else r + ((-1.0, 0.0, 0.0, 0.0),)

    def bounds(self, path):
        """The mesh's own bounding box (mesh units), or None."""
        slot = self.slots.get(path) if path else None
        if slot is None and path:
            frames = [k for k in self.slots if isinstance(k, tuple) and k[0] == path]
            slot = self.slots.get(frames[0]) if frames else None
        return None if slot is None else (slot[1].mesh_min, slot[1].mesh_max)


NO_MESH = ((0.0, 0.0, 0.0, -1.0), (0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 0.0))
NO_ANIM = (-1.0, 0.0, 0.0, 0.0)
