"""Triangle meshes as emitters and colliders.

A mesh (OBJ or STL, metres, y up) is baked once into a signed distance grid on the GPU. Small
meshes use the exact distance everywhere and the generalised winding number for inside/outside,
which copes with holes and overlapping parts. Dense meshes use a fast bake: exact distances in a
band near the surface and signed ray crossings for inside/outside. Bakes are kept on disk, so a
mesh is only baked once per machine. All meshes in use share one atlas texture that the solver
kernels sample.
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
BAKE_VERSION = 2     # bump when the bake changes, so stale disk caches are ignored


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


def load_mesh(path):
    p = Path(path)
    if not p.exists():
        raise MeshError(f'Mesh not found: {p}')
    ext = p.suffix.lower()
    if ext == '.obj':
        v, t = load_obj(p)
    elif ext == '.stl':
        v, t = load_stl(p)
    else:
        raise MeshError(f'{p.name}: meshes must be OBJ or STL files')
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


def bake(gpu: GPU, verts, tris, resolution=96, path='', method=None):
    """Signed distance grid of a mesh, computed on the GPU. `method` is 'exact', 'band' or None
    (exact for small meshes, banded for dense ones)."""
    verts = np.asarray(verts, np.float32)
    tris = np.asarray(tris, np.int64)
    vmin, vmax = verts[tris.reshape(-1)].min(0), verts[tris.reshape(-1)].max(0)
    lo, hi, dims, cell = grid_for(vmin, vmax, resolution)
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


def _cache_file(path, resolution):
    import hashlib
    h = hashlib.sha1(Path(path).read_bytes()).hexdigest()[:20]
    return cache_dir() / f'{h}_{int(resolution)}_v{BAKE_VERSION}.npz'


def load_or_bake(gpu, path, resolution=96):
    """A mesh's distance grid from the disk cache, or baked now (and cached)."""
    f = None
    try:
        f = _cache_file(path, resolution)
        if f.exists():
            z = np.load(f)
            m = z['meta']
            return MeshSDF(path=str(path), bmin=tuple(m[0:3]), bmax=tuple(m[3:6]), dims=tuple(int(x) for x in z['dims']),
                           data=z['data'], triangles=int(m[12]), mesh_min=tuple(m[6:9]), mesh_max=tuple(m[9:12]))
    except Exception as ex:  # a damaged or unreadable cache just means baking again
        log.info('Mesh cache unusable (%s); baking', ex)
    v, t = load_mesh(path)
    sdf = bake(gpu, v, t, resolution, path)
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


class MeshLibrary:
    """The meshes a scene uses, baked on first use and packed into one atlas texture."""

    def __init__(self, gpu: GPU):
        self.gpu = gpu
        self._baked = {}     # (path, mtime, resolution) -> MeshSDF
        self.slots = {}      # path -> (z offset, MeshSDF)
        self.errors = {}     # path -> message
        self._key = ()
        self.atlas = gpu.texture3d((1, 1, 1), 'r32float', 'mesh-atlas')
        gpu.upload(self.atlas, np.full((1, 1, 1, 1), 1.0e6, np.float32))

    def require(self, paths, resolution=96):
        """Make sure every mesh in `paths` is baked and in the atlas. True if the atlas changed."""
        want = sorted({p for p in paths if p})
        entries = []
        errors = {}
        for p in want:
            try:
                mtime = os.path.getmtime(p)
            except OSError:
                errors[p] = f'Mesh not found: {p}'
                continue
            key = (p, mtime, int(resolution))
            sdf = self._baked.get(key)
            if sdf is None:
                try:
                    import time
                    t0 = time.perf_counter()
                    sdf = load_or_bake(self.gpu, p, resolution)
                    log.info('Mesh %s: %d triangles, %s grid in %.2f s', Path(p).name, sdf.triangles, sdf.dims, time.perf_counter() - t0)
                except Exception as ex:
                    errors[p] = str(ex)
                    continue
                for k in [k for k in self._baked if k[0] == p]:
                    del self._baked[k]
                self._baked[key] = sdf
            entries.append((p, sdf))
        self.errors = errors
        for p, msg in errors.items():
            log.warning('%s', msg)
        key = tuple((p, id(s)) for p, s in entries)
        if key == self._key:
            return False
        self._key = key
        self._build_atlas(entries)
        return True

    def _build_atlas(self, entries):
        self.slots = {}
        if not entries:
            w = h = d = 1
            data = np.full((1, 1, 1), 1.0e6, np.float32)
        else:
            w = max(s.dims[0] for _, s in entries)
            h = max(s.dims[1] for _, s in entries)
            d = sum(s.dims[2] for _, s in entries)
            lim = int(self.gpu.limits.get('max-texture-dimension-3d', 2048))
            if d > lim:
                raise RuntimeError(f'Too many meshes for the GPU at this mesh detail ({d} > {lim} cells); '
                                   'lower Domain › Mesh detail or use fewer meshes.')
            data = np.full((d, h, w), 1.0e6, np.float32)
            z = 0
            for p, s in entries:
                nx, ny, nz = s.dims
                data[z:z + nz, :ny, :nx] = s.data
                self.slots[p] = (z, s)
                z += nz
        self.atlas.destroy()
        self.atlas = self.gpu.texture3d((w, h, d), 'r32float', 'mesh-atlas')
        self.gpu.upload(self.atlas, data[..., None])

    def ref(self, path):
        """(m0, m1, m2) vec4 triples for a mesh in the atlas, or None if it is not available."""
        slot = self.slots.get(path) if path else None
        if slot is None:
            return None
        z, s = slot
        return (s.bmin + (float(z),), s.bmax + (0.0,), tuple(float(x) for x in s.dims) + (0.0,))

    def bounds(self, path):
        """The mesh's own bounding box (mesh units), or None."""
        slot = self.slots.get(path) if path else None
        return None if slot is None else (slot[1].mesh_min, slot[1].mesh_max)


NO_MESH = ((0.0, 0.0, 0.0, -1.0), (0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 0.0))
