"""Volumes (OpenVDB files, numbered VDB sequences and USD Volume prims) as fields for Volume emitters:
the smoke or gas a volume holds goes into the simulation as smoke (and heat and fuel), not as a solid.

A field source is a volume source with FIELD_TAG after it: 'smoke.vdb|field', 'fire.####.vdb|field',
'shot.usd#/World/Cloud|field' (the prim's own frame) or 'shot.usd#/World/Cloud?world|field' (world
space). ':zup' after the tag turns a Z-up VDB (Blender's) to y-up.

load_field gives the density on an axis-aligned grid in the volume's own frame (metres, y up), scaled so
that its densest values are about 1, and its temperature as a second layer when the volume has one
(above the grid's background, scaled the same way). Level sets become 1 inside and 0 outside, with a
one-voxel ramp. The grid is at most `max_cells` a side (averaged down), with a border of empty cells.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FIELD_TAG = '|field'
DENSITY_NAMES = ('density', 'smoke', 'soot', 'fog', 'fuel')
TEMPERATURE_NAMES = ('temperature', 'heat', 'temp')
REFERENCE_PERCENTILE = 99.5   # this share of a field's non-empty voxels is at or below 1 after scaling
FIELD_VERSION = 1             # bump when load_field changes, so stale disk caches are ignored


class VolumeError(ValueError):
    pass


def is_field(source):
    return bool(source) and FIELD_TAG in source


def field_source(source, zup=False):
    """The field source for a volume file or prim."""
    return f'{source}{FIELD_TAG}' + (':zup' if zup else '')


def split_field(source):
    """(volume source, z-up?) of a field source."""
    base, _, opts = source.partition(FIELD_TAG)
    return base, 'zup' in opts


@dataclass
class FieldGrid:
    """A volume's field in the mesh atlas (same fields as mesh.MeshSDF): `data` is (layers * nz, ny, nx),
    density first, then temperature; the grid spans [bmin, bmax] (metres, the volume's own frame)."""
    path: str
    bmin: tuple
    bmax: tuple
    dims: tuple
    data: np.ndarray
    triangles: int
    mesh_min: tuple
    mesh_max: tuple
    layers: int = 1


def _pick(names, prefs):
    low = {n.lower(): n for n in names}
    for p in prefs:
        if p in low:
            return low[p]
    return None


def _signed_permutation(m, tol=1e-6):
    """(perm, sign, scale) if the 3x3 matrix m only scales, flips and swaps axes, else None: output
    axis a takes input axis perm[a], times sign[a] * scale[a]."""
    perm, sign, scale = [], [], []
    for a in range(3):
        row = m[a]
        j = int(np.argmax(np.abs(row)))
        big = abs(row[j])
        if big <= 0 or np.abs(np.delete(row, j)).max() > tol * big:
            return None
        perm.append(j)
        sign.append(1.0 if row[j] > 0 else -1.0)
        scale.append(big)
    if sorted(perm) != [0, 1, 2]:
        return None
    return perm, sign, scale


def _trilinear(a, idx):
    """Values of a (nx, ny, nz) at fractional index points idx (..., 3); 0 outside."""
    n = np.asarray(a.shape)
    i0 = np.floor(idx).astype(np.int64)
    f = idx - i0
    out = np.zeros(idx.shape[:-1], np.float32)
    for k in range(8):
        o = np.array([k & 1, (k >> 1) & 1, k >> 2])
        c = i0 + o
        w = np.prod(np.where(o == 1, f, 1.0 - f), axis=-1)
        ok = np.all((c >= 0) & (c < n), axis=-1)
        cc = np.clip(c, 0, n - 1)
        out += np.where(ok, a[cc[..., 0], cc[..., 1], cc[..., 2]], 0.0) * w
    return out


def _to_frame(layers, lo, m, max_cells):
    """Place index-space arrays (x, y, z; voxel centres at lo + i) through the 4x4 index -> frame
    transform m onto an axis-aligned grid in that frame: (arrays, grid corner (m), cell size per axis)."""
    lin, tr = m[:3, :3], m[:3, 3]
    sp = _signed_permutation(lin)
    if sp is not None:
        perm, sign, scale = sp
        out = []
        for a in layers:
            b = np.transpose(a, perm)
            for ax in range(3):
                if sign[ax] < 0:
                    b = b[(slice(None),) * ax + (slice(None, None, -1),)]
            out.append(np.ascontiguousarray(b))
        # the corner: the transformed index-space box's lowest corner
        n = np.asarray(layers[0].shape, float)
        c0 = lin @ (np.asarray(lo, float) - 0.5) + tr
        c1 = lin @ (np.asarray(lo, float) - 0.5 + n) + tr
        return out, np.minimum(c0, c1), np.asarray(scale, float)
    # a rotated grid: resample on axis-aligned cells as fine as its finest axis
    n = np.asarray(layers[0].shape, float)
    corners = np.array([[x, y, z] for x in (0, n[0]) for y in (0, n[1]) for z in (0, n[2])]) + np.asarray(lo) - 0.5
    w = corners @ lin.T + tr
    wlo, whi = w.min(0), w.max(0)
    cell = float(np.min(np.linalg.norm(lin, axis=0)))
    dims = np.maximum(np.ceil((whi - wlo) / cell).astype(int), 1)
    if dims.max() > 2 * max_cells:
        cell *= dims.max() / (2 * max_cells)
        dims = np.maximum(np.ceil((whi - wlo) / cell).astype(int), 1)
    g = np.stack(np.meshgrid(*[wlo[a] + (np.arange(dims[a]) + 0.5) * cell for a in range(3)], indexing='ij'), -1)
    inv = np.linalg.inv(lin)
    idx = (g - tr) @ inv.T - np.asarray(lo, float)
    return [_trilinear(a, idx) for a in layers], wlo, np.full(3, cell)


def _downsample(a, step):
    if step <= 1:
        return a
    pad = [(0, (-s) % step) for s in a.shape]
    b = np.pad(a, pad)
    s = b.shape
    return b.reshape(s[0] // step, step, s[1] // step, step, s[2] // step, step).mean(axis=(1, 3, 5))


def _reference(a):
    pos = a[a > 1e-9]
    if not len(pos):
        return 1.0
    return max(float(np.percentile(pos, REFERENCE_PERCENTILE)), 1e-9)


def _read_vdb(path):
    """(density, temperature or None, index lo, 4x4 index -> world) from a VDB file, both on the
    density grid's voxels."""
    from .vdbread import read_float_grid, vdb_grid_names
    names = [n for n, t in vdb_grid_names(path) if 'float' in t.lower() or 'half' in t.lower()]
    if not names:
        raise VolumeError(f'{Path(path).name} has no float grid')
    dname = _pick(names, DENSITY_NAMES) or names[0]
    dense, lo, xf, cls = read_float_grid(path, dname, max_cells=400_000_000, with_class=True)
    dense = dense.astype(np.float32)
    if cls == 'level set':
        vox = float(np.mean(np.linalg.norm(xf[:3, :3], axis=0)))
        dense = np.clip(0.5 - dense / (2.0 * max(vox, 1e-9)), 0.0, 1.0).astype(np.float32)
    temp = None
    tname = _pick(names, TEMPERATURE_NAMES)
    if tname is not None and tname != dname:
        t, tlo, txf, _, bg = read_float_grid(path, tname, max_cells=400_000_000, with_class=True, with_background=True)
        t = np.maximum(t.astype(np.float32) - bg, 0.0)   # heat above the air around it
        if np.allclose(txf, xf, atol=1e-9):
            # onto the density grid's voxels
            temp = np.zeros_like(dense)
            a = np.asarray(tlo) - np.asarray(lo)
            s0 = np.maximum(a, 0)
            s1 = np.minimum(a + np.asarray(t.shape), np.asarray(dense.shape))
            if (s1 > s0).all():
                temp[s0[0]:s1[0], s0[1]:s1[1], s0[2]:s1[2]] = t[s0[0] - a[0]:s1[0] - a[0], s0[1] - a[1]:s1[1] - a[1],
                                                                s0[2] - a[2]:s1[2] - a[2]]
        else:
            idx = np.stack(np.meshgrid(*[np.arange(n) for n in dense.shape], indexing='ij'), -1) + np.asarray(lo)
            wpos = idx @ xf[:3, :3].T + xf[:3, 3]
            ti = (wpos - txf[:3, 3]) @ np.linalg.inv(txf[:3, :3]).T - np.asarray(tlo)
            temp = _trilinear(t, ti)
    return dense, temp, np.asarray(lo), xf


def _usd_volume(source, frame):
    """(VDB path, frame -> transform from the VDB's world into the volume's frame, metres y-up)."""
    from . import usd as U
    f, prim_path = source.split('#', 1)
    pp, world = U._split(prim_path)
    stage = U.open_stage(f)
    prim = U._prim(stage, pp)
    tc = U.time_code(stage, frame)
    path, _grid = U._volume_field(prim, tc)
    if not path or not Path(path).exists():
        raise VolumeError(f'{pp}: the VDB file {path or "(none)"} is missing')
    m = U._world(prim, tc)
    if not world:
        m = np.linalg.inv(U._world(prim, tc)) @ m
    return path, U.axis_matrix(stage) @ m


def load_field(source, frame=None, max_cells=128, reference=None):
    """A field source as a FieldGrid (see the module notes), and the (density, temperature) scale used,
    so that the frames of a sequence share one: pass it back as `reference` for the next frames."""
    from ..engine.mesh import frame_file, is_numbered, sequence_files
    base, zup = split_field(source)
    to_frame = np.eye(4)
    if '#/' in base:
        path, to_frame = _usd_volume(base, frame)
    elif is_numbered(base):
        files = sequence_files(base)
        if not files:
            raise VolumeError(f'No files match {Path(base).name}')
        fr = int(math.floor(frame)) if frame is not None else min(files)
        fr = min(max(fr, min(files)), max(files))
        while fr not in files:
            fr -= 1
        path = files[fr]
    else:
        path = base
    if not Path(path).exists():
        raise VolumeError(f'Volume not found: {path}')
    if Path(path).suffix.lower() != '.vdb':
        raise VolumeError(f'{Path(path).name}: volumes must be OpenVDB (.vdb) files or USD Volume prims')
    cached = _cache_path(path, (base, zup, int(max_cells), np.round(to_frame, 9).tolist(), reference))
    if cached is not None and cached.exists():
        try:
            z = np.load(cached)
            meta = z['meta']
            return FieldGrid(path=str(path), bmin=tuple(meta[0:3]), bmax=tuple(meta[3:6]), dims=tuple(int(x) for x in z['dims']),
                             data=z['data'], triangles=0, mesh_min=tuple(meta[6:9]), mesh_max=tuple(meta[9:12]),
                             layers=int(meta[12])), (float(meta[13]), float(meta[14]))
        except Exception:
            pass   # a damaged cache just means reading the VDB again
    grid, ref = _load(path, zup, to_frame, max_cells, reference)
    if cached is not None:
        try:
            cached.parent.mkdir(parents=True, exist_ok=True)
            meta = np.array(list(grid.bmin) + list(grid.bmax) + list(grid.mesh_min) + list(grid.mesh_max)
                            + [grid.layers, ref[0], ref[1]], np.float64)
            tmp = cached.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, data=grid.data, dims=np.array(grid.dims), meta=meta)
            tmp.replace(cached)
        except OSError:
            pass
    return grid, ref


def _cache_path(path, key):
    import hashlib
    from ..engine.mesh import cache_dir
    try:
        h = hashlib.sha1(Path(path).read_bytes() + repr(key).encode()).hexdigest()[:24]
    except OSError:
        return None
    return cache_dir() / f'{h}_field_v{FIELD_VERSION}.npz'


def _load(path, zup, to_frame, max_cells, reference):
    dense, temp, lo, xf = _read_vdb(path)
    m = to_frame @ xf
    if zup:
        m = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float) @ m
    layers = [dense] + ([temp] if temp is not None else [])
    step = max(1, int(math.ceil(max(dense.shape) / max(8, int(max_cells) - 2))))
    if step > 1:
        layers = [_downsample(a, step) for a in layers]
        m = m.copy()
        m[:3, :3] = m[:3, :3] * step
        lo = (np.asarray(lo, float) - 0.5) / step + 0.5
    arrays, corner, cell = _to_frame(layers, lo, m, max_cells)
    if reference is None:
        reference = (_reference(arrays[0]), _reference(arrays[1]) if len(arrays) > 1 else 1.0)
    out = [np.clip(arrays[0] / reference[0], 0.0, 4.0)]
    if len(arrays) > 1:
        out.append(np.clip(arrays[1] / reference[1], 0.0, 4.0))
    out = [np.pad(a.astype(np.float32), 1) for a in out]   # an empty border: the field fades out at its edge
    corner = corner - cell
    nx, ny, nz = out[0].shape
    data = np.concatenate([np.transpose(a, (2, 1, 0)) for a in out], axis=0)   # (layers * nz, ny, nx)
    hi = corner + cell * np.array([nx, ny, nz])
    occupied = np.argwhere(out[0] > 1e-3)
    if len(occupied):
        mlo = corner + occupied.min(0) * cell
        mhi = corner + (occupied.max(0) + 1) * cell
    else:
        mlo, mhi = corner, hi
    grid = FieldGrid(path=str(path), bmin=tuple(float(x) for x in corner), bmax=tuple(float(x) for x in hi),
                     dims=(int(nx), int(ny), int(nz)), data=np.ascontiguousarray(data, np.float32), triangles=0,
                     mesh_min=tuple(float(x) for x in mlo), mesh_max=tuple(float(x) for x in mhi), layers=len(out))
    return grid, reference


def field_animated(source):
    """Whether a field source changes from frame to frame (a numbered sequence, or a USD volume whose
    file or transform is animated)."""
    from ..engine.mesh import is_numbered, split_source
    base, _ = split_field(source)
    if is_numbered(base):
        return True
    f, prim = split_source(base)
    if prim:
        from .usd import usd_prim_animated
        return usd_prim_animated(f, prim)
    return False
