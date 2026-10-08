"""Volumes (OpenVDB files, numbered VDB sequences, USD Volume prims and USD Points prims) as fields for Volume
emitters: the smoke or gas a volume holds goes into the simulation as smoke (and heat and fuel), not as a solid, and
moves off with the velocity it had; a liquid source fills where it is dense.

A field source is a volume source with FIELD_TAG after it: 'smoke.vdb|field', 'fire.####.vdb|field',
'shot.usd#/World/Cloud|field' (the prim's own frame) or 'shot.usd#/World/Cloud?world|field' (world
space). ':zup' after the tag turns a Z-up VDB (Blender's) to y-up. A USD Points prim (a particle cache: sparks, dust,
a FLIP sim's particles) is splatted into a field: each point a soft ball as wide as its width, carrying its velocity.

load_field gives the field on an axis-aligned grid in the volume's own frame (metres, y up), in layers: the density,
scaled so that its densest values are about 1; its temperature when the volume has one (above the grid's background,
scaled the same way); and its velocity (three layers, m/s in the volume's frame) when it has a vel grid (or vel.x,
vel.y and vel.z float grids). Level sets become 1 inside and 0 outside, with a one-voxel ramp. The grid is no finer
than the simulation can use (`finest`: FIELD_PER_CELL cells to a simulation cell), at most `max_cells` a side and
FIELD_MAX_VOXELS in all (averaged down), with a border of empty cells, and its layers fit one atlas column
(FIELD_DEPTH). A volume turned other than by quarter turns (a tilted USD prim) is resampled on the axis-aligned box
around it, and it is that box that keeps to these limits.

A changing field (a sequence, an animated USD prim) puts all its frames on one grid (field_grid: the union of the
frames' boxes, read from the VDB headers without reading the frames), so the solver can blend from one frame to the
next.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FIELD_TAG = '|field'
DENSITY_NAMES = ('density', 'smoke', 'soot', 'fog', 'fuel')
TEMPERATURE_NAMES = ('temperature', 'heat', 'temp')
VELOCITY_NAMES = ('vel', 'v', 'velocity')
REFERENCE_PERCENTILE = 99.5   # this share of a field's non-empty voxels is at or below 1 after scaling
FIELD_VERSION = 3             # bump when load_field changes, so stale disk caches are ignored
FIELD_MAX_CELLS = 512         # a field is at most this many cells a side (its border included; a rotated one's box) ...
FIELD_MAX_VOXELS = 24_000_000  # ... and this many in all, a layer (each takes 4 bytes a cell in the mesh atlas)
FIELD_PER_CELL = 2.0          # a field's cells across one simulation cell at most: finer adds nothing it can use
FIELD_DEPTH = 2048            # cells: all of a field's layers stacked along z fit one column of the mesh atlas
POINT_SPREAD = 1.0            # cells: a point smaller than its field's cells is spread over a ball this big
POINTS_MAX = 4_000_000        # points splatted at most (more are thinned evenly)
SEQUENCE_FRAMES = 2000        # frames of a changing USD field looked at for its one grid at most (VDB headers) ...
POINT_FRAMES = 200            # ... or of moving USD points (their every point is read)


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
    """A volume's field in the mesh atlas (same fields as mesh.MeshSDF): `data` is (layers * nz, ny, nx): the density,
    then the temperature (`temp`), then the velocity's x, y and z (`vel`); the grid spans [bmin, bmax] (metres, the
    volume's own frame)."""
    path: str
    bmin: tuple
    bmax: tuple
    dims: tuple
    data: np.ndarray
    triangles: int
    mesh_min: tuple
    mesh_max: tuple
    layers: int = 1
    temp: bool = False
    vel: bool = False

    @property
    def flags(self):
        """What it holds past its density, as the emitter's m1.w carries it: 1 a temperature, 2 a velocity."""
        return (1 if self.temp else 0) | (2 if self.vel else 0)


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


def _sampler(arrays):
    """Trilinear lookup into `arrays` (all (nx, ny, nz)): a function of fractional index points idx (..., 3) giving
    each array's values there, 0 outside. The arrays are interleaved once, so that a corner is one gather for them all,
    and the points are taken in chunks, to keep the memory down on big grids."""
    n = np.asarray(arrays[0].shape[:3])
    table = np.stack([np.asarray(a, np.float32).reshape(-1) for a in arrays], -1)
    stride = (int(n[1] * n[2]), int(n[2]), 1)

    def sample(idx, chunk=1 << 20):
        flat = np.asarray(idx, np.float32).reshape(-1, 3)
        out = np.zeros((len(flat), table.shape[1]), np.float32)
        for s in range(0, len(flat), chunk):
            p = flat[s:s + chunk]
            i0 = np.floor(p)
            f = p - i0
            i0 = i0.astype(np.int64)
            ends = []   # per axis: the (offset, weight) of its lower and upper neighbour
            for ax in range(3):
                both = []
                for d, wt in ((0, 1.0 - f[:, ax]), (1, f[:, ax])):
                    c = i0[:, ax] + d
                    both.append((np.clip(c, 0, n[ax] - 1) * stride[ax], wt * ((c >= 0) & (c < n[ax]))))
                ends.append(both)
            acc = out[s:s + chunk]
            for xo, xw in ends[0]:
                for yo, yw in ends[1]:
                    xy, wxy = xo + yo, xw * yw
                    for zo, zw in ends[2]:
                        acc += table[xy + zo] * (wxy * zw)[:, None]
        shape = np.shape(idx)[:-1]
        return [out[:, k].reshape(shape) for k in range(table.shape[1])]
    return sample


def _resample(arrays, corner, cell, dims, to_index):
    """Arrays sampled trilinearly at the cell centres of an axis-aligned grid (corner, cell (per axis), dims), a slab
    along x at a time (a whole grid's points would take gigabytes): to_index maps points (..., 3) of the grid's frame
    to fractional indices into the arrays."""
    corner, cell = np.asarray(corner, float), np.broadcast_to(np.asarray(cell, float), (3,))
    sample = _sampler(arrays)
    jj, kk = np.meshgrid(corner[1] + (np.arange(dims[1]) + 0.5) * cell[1],
                         corner[2] + (np.arange(dims[2]) + 0.5) * cell[2], indexing='ij')
    out = [np.zeros(tuple(int(d) for d in dims), np.float32) for _ in arrays]
    for i in range(int(dims[0])):
        pts = np.stack([np.full_like(jj, corner[0] + (i + 0.5) * cell[0]), jj, kk], -1)
        for o, v in zip(out, sample(to_index(pts))):
            o[i] = v
    return out


def _side(layers, max_cells):
    """Cells a side a field may have, its border included: max_cells, and all its layers within one atlas column."""
    return min(int(max_cells), FIELD_DEPTH // max(int(layers), 1))


def _to_frame(layers, lo, m, max_cells, finest=None):
    """Place index-space arrays (x, y, z; voxel centres at lo + i) through the 4x4 index -> frame
    transform m onto an axis-aligned grid in that frame: (arrays, grid corner (m), cell size per axis). A rotated grid
    is resampled on cells no finer than `finest` (m), within the field's limits."""
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
    # a rotated grid: resample on axis-aligned cells as fine as its finest axis, or as the field's limits allow (_step
    # has averaged it down by the whole part of that)
    n = np.asarray(layers[0].shape, float)
    corners = np.array([[x, y, z] for x in (0, n[0]) for y in (0, n[1]) for z in (0, n[2])]) + np.asarray(lo) - 0.5
    w = corners @ lin.T + tr
    wlo, whi = w.min(0), w.max(0)
    ext = np.maximum(whi - wlo, 1e-12)
    side = max(_side(len(layers), max_cells) - 2, 1)   # (room for its border)
    cell = max(float(np.min(np.linalg.norm(lin, axis=0))), finest or 0.0, float(ext.max()) / side * (1 + 1e-9))
    dims = np.maximum(np.ceil(ext / cell).astype(int), 1)
    while np.prod(dims + 2) > FIELD_MAX_VOXELS:
        cell *= 1.02
        dims = np.maximum(np.ceil(ext / cell).astype(int), 1)
    inv = np.linalg.inv(lin)
    lo = np.asarray(lo, float)
    return _resample(layers, wlo, cell, dims, lambda p: (p - tr) @ inv.T - lo), wlo, np.full(3, cell)


def _onto(arrays, corner, cell, grid):
    """Arrays on an axis-aligned grid (corner, cell) placed on another one, grid = (corner, cell, dims): copied across
    when the cells line up (a sequence's frames on its lattice), else sampled trilinearly. Empty where they do not
    reach."""
    g0, gc, gd = np.asarray(grid[0], float), np.asarray(grid[1], float), tuple(int(x) for x in grid[2])
    off = (np.asarray(corner, float) - g0) / gc
    if np.allclose(cell, gc, rtol=1e-5) and np.allclose(off, np.round(off), atol=1e-3):
        o = np.round(off).astype(int)
        out = []
        for a in arrays:
            b = np.zeros(gd, np.float32)
            s0 = np.maximum(o, 0)
            s1 = np.minimum(o + np.asarray(a.shape), gd)
            if (s1 > s0).all():
                b[s0[0]:s1[0], s0[1]:s1[1], s0[2]:s1[2]] = a[s0[0] - o[0]:s1[0] - o[0], s0[1] - o[1]:s1[1] - o[1],
                                                             s0[2] - o[2]:s1[2] - o[2]]
            out.append(b)
        return out
    corner, cell = np.asarray(corner, float), np.asarray(cell, float)
    return _resample(arrays, g0, gc, gd, lambda p: (p - corner) / cell - 0.5)


def _downsample(a, step):
    if step <= 1:
        return a
    pad = [(0, (-s) % step) for s in a.shape[:3]] + [(0, 0)] * (a.ndim - 3)
    b = np.pad(a, pad)
    s = b.shape
    return b.reshape(s[0] // step, step, s[1] // step, step, s[2] // step, step, *s[3:]).mean(axis=(1, 3, 5))


def _aligned(a, lo, step):
    """An index-space array (voxel lo + i) padded at the front so it starts on a multiple of `step`: (array, its lo).
    Averaged down from there, every frame of a sequence lands on the same lattice."""
    lo = np.asarray(lo, np.int64)
    al = (lo // step) * step
    front = lo - al
    if step <= 1 or not front.any():
        return a, lo
    return np.pad(a, [(int(f), 0) for f in front] + [(0, 0)] * (a.ndim - 3)), al


def _reference(a):
    pos = a[a > 1e-9]
    if not len(pos):
        return 1.0
    return max(float(np.percentile(pos, REFERENCE_PERCENTILE)), 1e-9)


# -- reading a frame -----------------------------------------------------------------------------------

def _vdb_names(path):
    """(density grid, temperature grid or None, velocity: a vec3 grid name, a triple of float grid names, or None) of
    a VDB file, from its headers."""
    from .vdbread import vdb_grid_info
    info = vdb_grid_info(path)
    floats = [n for n, g in info.items() if 'float' in g['type'].lower() or 'half' in g['type'].lower()]
    if not floats:
        raise VolumeError(f'{Path(path).name} has no float grid')
    dname = _pick(floats, DENSITY_NAMES) or floats[0]
    tname = _pick(floats, TEMPERATURE_NAMES)
    vecs = [n for n, g in info.items() if 'vec3' in g['type'].lower()]
    vel = _pick(vecs, VELOCITY_NAMES)
    if vel is None:
        for base in VELOCITY_NAMES:
            for sep in ('.', '_'):
                trio = [_pick(floats, (f'{base}{sep}{c}',)) for c in 'xyz']
                if all(trio):
                    vel = tuple(trio)
                    break
            if vel is not None:
                break
    return dname, (tname if tname != dname else None), vel, info


def _layer_onto(arr, lo, xf, dense_shape, dense_lo, dense_xf, step):
    """Another grid of the file (arr: (nx, ny, nz[, 3]) from voxel lo, transform xf) on the density's voxels, averaged
    down by `step` like it: lined up and copied across when both grids share a transform, else sampled trilinearly."""
    tail = arr.shape[3:]
    out = np.zeros(tuple(dense_shape) + tail, np.float32)
    if np.allclose(xf, dense_xf, atol=1e-9):
        a, alo = _aligned(arr, lo, step)
        a = _downsample(a, step)
        o = (alo - dense_lo) // step
        s0 = np.maximum(o, 0)
        s1 = np.minimum(o + np.asarray(a.shape[:3]), np.asarray(dense_shape))
        if (s1 > s0).all():
            out[s0[0]:s1[0], s0[1]:s1[1], s0[2]:s1[2]] = a[s0[0] - o[0]:s1[0] - o[0], s0[1] - o[1]:s1[1] - o[1],
                                                           s0[2] - o[2]:s1[2] - o[2]]
        return out
    # its own voxels: averaged down to about the density's cells, then looked up at their centres (a slab at a time)
    ratio = np.linalg.norm(dense_xf[:3, :3], axis=0).min() * step / max(np.linalg.norm(xf[:3, :3], axis=0).min(), 1e-12)
    own = max(1, int(round(ratio)))
    a, alo = _aligned(arr, lo, own)
    a = _downsample(a, own)
    alo_c = (np.asarray(alo, float) - 0.5) / own + 0.5
    inv = np.linalg.inv(xf[:3, :3] * own)
    sample = _sampler([a[..., c] for c in range(tail[0])] if tail else [a])
    jj, kk = np.meshgrid(np.arange(dense_shape[1]), np.arange(dense_shape[2]), indexing='ij')
    for i in range(dense_shape[0]):
        cells = np.stack([np.full_like(jj, i), jj, kk], -1)
        fine = (np.asarray(dense_lo, float) - 0.5) + (cells + 0.5) * step   # the cells' centres, in density voxels
        ti = (fine @ dense_xf[:3, :3].T + dense_xf[:3, 3] - xf[:3, 3]) @ inv.T - alo_c
        vals = sample(ti)
        out[i] = np.stack(vals, -1) if tail else vals[0]
    return out


def _read_vdb(path, step_for):
    """A VDB frame on the density grid's voxels, averaged down by step_for(density shape, its lo, its transform, the
    layer count): {'layers': [density, temperature?, vx?, vy?, vz? (velocity in the file's world axes)], 'temp',
    'vel', 'lo' (coarse index of the first cell), 'xf' (coarse index -> file world)}."""
    from .vdbread import read_float_grid
    dname, tname, vname, _info = _vdb_names(path)
    dense, lo, xf, cls = read_float_grid(path, dname, max_cells=400_000_000, with_class=True)
    dense = dense.astype(np.float32)
    if cls == 'level set':
        vox = float(np.mean(np.linalg.norm(xf[:3, :3], axis=0)))
        dense = np.clip(0.5 - dense / (2.0 * max(vox, 1e-9)), 0.0, 1.0).astype(np.float32)
    step = int(step_for(dense.shape, lo, xf, 1 + (tname is not None) + 3 * (vname is not None)))
    d, dlo = _aligned(dense, lo, step)
    d = _downsample(d, step)
    out = {'layers': [d], 'temp': False, 'vel': False}
    if tname is not None:
        t, tlo, txf, _, bg = read_float_grid(path, tname, max_cells=400_000_000, with_class=True, with_background=True)
        t = np.maximum(t.astype(np.float32) - bg, 0.0)   # heat above the air around it
        out['layers'].append(_layer_onto(t, tlo, txf, d.shape, dlo, xf, step))
        out['temp'] = True
    if vname is not None:
        if isinstance(vname, tuple):
            comps = []
            for n in vname:
                c, clo, cxf = read_float_grid(path, n, max_cells=400_000_000)
                comps.append(_layer_onto(c.astype(np.float32), clo, cxf, d.shape, dlo, xf, step))
            v = np.stack(comps, -1)
        else:
            raw, vlo, vxf, meta = read_float_grid(path, vname, max_cells=400_000_000, components=True, with_meta=True)
            v = _layer_onto(raw.astype(np.float32), vlo, vxf, d.shape, dlo, xf, step)
            if meta.get('is_local_space'):
                v = v @ vxf[:3, :3].T    # (vectors stored in index space: the grid's own axes)
        out['layers'] += [v[..., 0], v[..., 1], v[..., 2]]
        out['vel'] = True
    m = xf.copy()
    m[:3, :3] = m[:3, :3] * step
    out['lo'] = (np.asarray(dlo, float) - 0.5) / step + 0.5
    out['xf'] = m
    return out


def _usd_prim(base, frame):
    """(stage, prim, its type, frame -> transform from the prim's space into the volume's frame, metres y-up)."""
    from . import usd as U
    f, prim_path = base.split('#', 1)
    pp, world = U._split(prim_path)
    stage = U.open_stage(f)
    prim = U._prim(stage, pp)
    tc = U.time_code(stage, frame)
    m = U._world(prim, tc) if world else np.eye(4)
    return stage, prim, prim.GetTypeName(), U.axis_matrix(stage) @ m


def _usd_volume(base, frame):
    """(VDB path, transform from the VDB's world into the volume's frame) of a USD Volume prim."""
    from . import usd as U
    stage, prim, _t, to_frame = _usd_prim(base, frame)
    path, _grid = U._volume_field(prim, U.time_code(stage, frame))
    if not path or not Path(path).exists():
        raise VolumeError(f'{prim.GetPath()}: the VDB file {path or "(none)"} is missing')
    return path, to_frame


def _points(prim, frame):
    """(positions (n, 3), radii (n,), velocities (n, 3) or None) of a USD Points prim at a frame, in its own space
    (stage units, per second), thinned evenly past POINTS_MAX."""
    from pxr import UsdGeom
    from . import usd as U
    tc = U.time_code(prim.GetStage(), frame)
    pt = UsdGeom.Points(prim)
    p = np.asarray(pt.GetPointsAttr().Get(tc) or [], np.float64).reshape(-1, 3)
    r = U._point_radii(prim, tc, len(p))
    v = np.asarray(pt.GetVelocitiesAttr().Get(tc) or [], np.float64).reshape(-1, 3)
    v = v if len(v) == len(p) else None
    if len(p) > POINTS_MAX:
        keep = np.linspace(0, len(p) - 1, POINTS_MAX).astype(np.int64)
        p, r = p[keep], r[keep]
        v = None if v is None else v[keep]
    return p, np.maximum(r, 0.0), v


def _point_voxel(r, lo, hi, max_cells, finest_local):
    """The splat voxel (stage units) for points of radii r spanning lo..hi: as fine as the typical point, but no finer
    than the field's limits allow."""
    ext = float(np.max(hi - lo)) if len(r) else 1.0
    typical = float(np.median(r)) if len(r) else ext / 16
    vox = max(typical, ext / max(8, max_cells - 4), 1e-6)
    if finest_local:
        vox = max(vox, finest_local)
    while (np.ceil((hi - lo) / vox) + 4).prod() > FIELD_MAX_VOXELS:
        vox *= 1.25
    return vox


def splat_points(p, r, v, vox):
    """Points (positions p (n, 3), radii r (n,), velocities v (n, 3) or None) splatted on the lattice of voxel `vox`
    (index i centred on i * vox): (density (nx, ny, nz), velocity (nx, ny, nz, 3) or None, index of the first voxel).
    A point is a soft ball (1 - (d / R)^2)^2 of radius R = max(r, POINT_SPREAD voxels), 1 at its centre (less for a
    point smaller than its ball, by volume); a voxel's velocity is the weighted mean of its points'."""
    if not len(p):
        return np.zeros((1, 1, 1), np.float32), (np.zeros((1, 1, 1, 3), np.float32) if v is not None else None), \
            np.zeros(3, np.int64)
    R = np.maximum(r, POINT_SPREAD * vox)
    lo = np.floor((p - R[:, None]).min(0) / vox).astype(np.int64) - 1
    hi = np.ceil((p + R[:, None]).max(0) / vox).astype(np.int64) + 2
    shape = tuple(int(x) for x in hi - lo)
    dens = np.zeros(int(np.prod(shape)), np.float64)
    wsum = np.zeros_like(dens) if v is not None else None
    vacc = np.zeros((len(dens), 3), np.float64) if v is not None else None
    K = int(np.ceil(R.max() / vox))
    o = np.stack(np.meshgrid(*[np.arange(-K, K + 1)] * 3, indexing='ij'), -1).reshape(-1, 3)
    per = max(1, (1 << 22) // len(o))
    for s in range(0, len(p), per):
        ps, rs, Rs = p[s:s + per], r[s:s + per], R[s:s + per]
        c = np.round(ps / vox).astype(np.int64)
        cells = c[:, None, :] + o[None]                                   # (k, offsets, 3)
        d = np.linalg.norm(cells * vox - ps[:, None, :], axis=-1)
        w = np.clip(1.0 - (d / Rs[:, None]) ** 2, 0.0, None) ** 2
        w *= np.minimum(rs / Rs, 1.0)[:, None] ** 3                     # (a small point deposits its share)
        idx = cells - lo
        flat = (idx[..., 0] * shape[1] + idx[..., 1]) * shape[2] + idx[..., 2]
        np.add.at(dens, flat.ravel(), w.ravel())
        if v is not None:
            ww = np.clip(1.0 - (d / Rs[:, None]) ** 2, 0.0, None) ** 2
            np.add.at(wsum, flat.ravel(), ww.ravel())
            for a in range(3):
                np.add.at(vacc[:, a], flat.ravel(), (ww * v[s:s + per, a][:, None]).ravel())
    vel = None
    if v is not None:
        vel = (vacc / np.maximum(wsum, 1e-12)[:, None]).reshape(shape + (3,)).astype(np.float32)
    return dens.reshape(shape).astype(np.float32), vel, lo


def _read_points(base, frame, max_cells, finest, vox=None):
    """A USD Points prim's frame, splatted: the same layout as _read_vdb, plus 'to_frame'."""
    stage, prim, _t, to_frame = _usd_prim(base, frame)
    p, r, v = _points(prim, frame)
    scale = float(np.mean(np.linalg.norm(to_frame[:3, :3], axis=0)))
    if vox is None:
        lo, hi = (p.min(0), p.max(0)) if len(p) else (np.zeros(3), np.ones(3))
        vox = _point_voxel(r, lo - r.max(initial=0.0), hi + r.max(initial=0.0), max_cells,
                           None if not finest else finest / max(scale, 1e-12))
    dens, vel, lo = splat_points(p, r, v, vox)
    xf = np.diag([vox, vox, vox, 1.0])
    out = {'layers': [dens], 'temp': False, 'vel': vel is not None, 'lo': lo.astype(float), 'xf': xf,
           'to_frame': to_frame}
    if vel is not None:
        out['layers'] += [vel[..., 0], vel[..., 1], vel[..., 2]]
    return out


def _frame_file(base, frame):
    """(the VDB file of a frame, transform from its world into the volume's frame), or (None, None) for USD points."""
    from ..engine.mesh import is_numbered, sequence_files
    if '#/' in base:
        _stage, _prim, t, _m = _usd_prim(base, frame)
        if t == 'Points':
            return None, None
        return _usd_volume(base, frame)
    if is_numbered(base):
        files = sequence_files(base)
        if not files:
            raise VolumeError(f'No files match {Path(base).name}')
        fr = int(math.floor(frame)) if frame is not None else min(files)
        fr = min(max(fr, min(files)), max(files))
        while fr not in files:
            fr -= 1
        return files[fr], np.eye(4)
    return base, np.eye(4)


ZUP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)


def _step(shape, lin, layers, max_cells, finest):
    """How many voxels a side to average into one cell: few enough cells a side (max_cells, and all the layers within
    one atlas column) and in all (FIELD_MAX_VOXELS), and none finer than `finest` (m) needs. `lin` turns the voxels
    into the volume's frame. A rotated grid is resampled on the axis-aligned box around it (_to_frame), up to about
    1.7 times as many cells a side, so that box is what must keep to the limits; as it is resampled anyway, it is
    averaged down by the whole part of what they ask only, and the resample's cells (less than twice as coarse) take
    the rest, rather than halving a grid that is only just over a limit."""
    n = np.asarray(shape[:3], float)
    vox = max(float(np.min(np.linalg.norm(lin, axis=0))), 1e-12)
    rotated = _signed_permutation(lin) is None
    if rotated:
        n = np.abs(lin) @ n / vox   # (the box around it, in cells as fine as its finest axis)
    side = max(8, _side(layers, max_cells) - 6)   # (room for its border, and a rotated or changing grid's rounding)
    r = max(1.0, n.max() / side, (finest or 0.0) / vox - 1e-6)
    if rotated:
        while np.prod(np.ceil(n / r) + 6) > FIELD_MAX_VOXELS:
            r *= 1.01
        return max(1, int(math.floor(r + 1e-6)))
    step = int(math.ceil(r - 1e-9))
    while np.prod(np.ceil(n / step) + 6) > FIELD_MAX_VOXELS:
        step += 1
    return step


# -- loading ---------------------------------------------------------------------------------------------

def load_field(source, frame=None, max_cells=FIELD_MAX_CELLS, reference=None, finest=None, grid=None):
    """A field source as a FieldGrid (see the module notes), and the (density, temperature) scale used, so that the
    frames of a sequence share one: pass it back as `reference` for the next frames. `finest` (m): the finest cell
    worth keeping (FIELD_PER_CELL to a simulation cell); `grid`: the one grid of a changing field (field_grid), which
    every frame is put on."""
    base, zup = split_field(source)
    path, to_frame = _frame_file(base, frame)
    if path is not None:
        if not Path(path).exists():
            raise VolumeError(f'Volume not found: {path}')
        if Path(path).suffix.lower() != '.vdb':
            raise VolumeError(f'{Path(path).name}: volumes must be OpenVDB (.vdb) files or USD Volume or Points prims')
        keyed, extra = path, ''
    else:
        keyed, extra = base.split('#', 1)[0], f'{base}@{frame}'
    key = (base, zup, int(max_cells), None if to_frame is None else np.round(to_frame, 9).tolist(), reference,
           None if finest is None else round(float(finest), 9), _grid_key(grid), extra)
    cached = _cache_path(keyed, key)
    if cached is not None and cached.exists():
        try:
            z = np.load(cached)
            meta = z['meta']
            return FieldGrid(path=str(path or base), bmin=tuple(meta[0:3]), bmax=tuple(meta[3:6]),
                             dims=tuple(int(x) for x in z['dims']), data=z['data'], triangles=0,
                             mesh_min=tuple(meta[6:9]), mesh_max=tuple(meta[9:12]), layers=int(meta[12]),
                             temp=bool(meta[15]), vel=bool(meta[16])), (float(meta[13]), float(meta[14]))
        except Exception:
            pass   # a damaged cache just means reading the volume again
    out, ref = _load(base, frame, path, to_frame, zup, max_cells, reference, finest, grid)
    if cached is not None:
        try:
            cached.parent.mkdir(parents=True, exist_ok=True)
            meta = np.array(list(out.bmin) + list(out.bmax) + list(out.mesh_min) + list(out.mesh_max)
                            + [out.layers, ref[0], ref[1], float(out.temp), float(out.vel)], np.float64)
            tmp = cached.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, data=out.data, dims=np.array(out.dims), meta=meta)
            tmp.replace(cached)
        except OSError:
            pass
    return out, ref


def _grid_key(grid):
    if grid is None:
        return None
    return [np.round(np.asarray(grid[0], float), 9).tolist(), np.round(np.asarray(grid[1], float), 12).tolist(),
            [int(x) for x in grid[2]], int(grid[3])]


def _cache_path(path, key):
    import hashlib
    from ..engine.mesh import cache_dir
    try:
        h = hashlib.sha1(Path(path).read_bytes() + repr(key).encode()).hexdigest()[:24]
    except OSError:
        return None
    return cache_dir() / f'{h}_field_v{FIELD_VERSION}.npz'


def _load(base, frame, path, to_frame, zup, max_cells, reference, finest, grid):
    turn = ZUP if zup else np.eye(4)
    if path is None:
        vox = None
        if grid is not None:
            vox = grid[4]
        fr = _read_points(base, frame, max_cells, finest, vox)
        to_frame = fr['to_frame']
        m = turn @ to_frame @ fr['xf']
        if grid is None:
            step = _step(fr['layers'][0].shape, m[:3, :3], len(fr['layers']), max_cells, finest)
        else:
            step = int(grid[3])
        if step > 1:
            lo0 = np.asarray(fr['lo'], np.int64)
            new = []
            for a in fr['layers']:
                b, blo = _aligned(a, lo0, step)
                new.append(_downsample(b, step))
            fr['layers'] = new
            fr['lo'] = (np.asarray(blo, float) - 0.5) / step + 0.5
            fr['xf'] = fr['xf'].copy()
            fr['xf'][:3, :3] *= step
    else:
        def step_for(shape, lo, xf, layers):
            if grid is not None:
                return int(grid[3])
            return _step(shape, (turn @ to_frame @ xf)[:3, :3], layers, max_cells, finest)
        fr = _read_vdb(path, step_for)
    m = turn @ to_frame @ fr['xf']
    layers = fr['layers']
    if fr['vel']:
        # the velocity into the volume's frame (m/s): the file's world axes turned and scaled as its points are
        lin = (turn @ to_frame)[:3, :3]
        v = np.stack(layers[-3:], -1).astype(np.float32, copy=False) @ lin.T.astype(np.float32)
        layers = layers[:-3] + [v[..., 0], v[..., 1], v[..., 2]]
        del v
    arrays, corner, cell = _to_frame(layers, fr['lo'], m, max_cells, finest)
    layers = fr['layers'] = None   # (the file's own voxels are done with)
    if reference is None:
        reference = (_reference(arrays[0]), _reference(arrays[1]) if fr['temp'] else 1.0)
    for k, ref in enumerate(reference[:1 + fr['temp']]):   # the density and temperature, scaled; the velocity as it is
        arrays[k] = np.clip(arrays[k] / ref, 0.0, 4.0)
    arrays = [np.asarray(a, np.float32) for a in arrays]
    if grid is not None:
        arrays = _onto(arrays, corner, cell, grid)
        corner, cell, pad = np.asarray(grid[0], float), np.asarray(grid[1], float), 0
    else:
        corner, pad = corner - cell, 1   # an empty border: the field fades out at its edge
    nx, ny, nz = (int(s) + 2 * pad for s in arrays[0].shape)
    count = len(arrays)
    data = np.zeros((count * nz, ny, nx), np.float32)   # (layers * nz, ny, nx), filled a layer at a time
    for k in range(count):
        data[k * nz + pad:(k + 1) * nz - pad, pad:ny - pad, pad:nx - pad] = np.transpose(arrays[k], (2, 1, 0))
        arrays[k] = None
    hi = corner + cell * np.array([nx, ny, nz])
    dense = data[:nz] > 1e-3   # (z, y, x)
    if dense.any():
        span = [np.flatnonzero(dense.any(axis=ax)) for ax in ((0, 1), (0, 2), (1, 2))]   # (x, y, z)
        mlo = corner + np.array([s[0] for s in span]) * cell
        mhi = corner + (np.array([s[-1] for s in span]) + 1) * cell
    else:
        mlo, mhi = corner, hi
    field = FieldGrid(path=str(path or base), bmin=tuple(float(x) for x in corner), bmax=tuple(float(x) for x in hi),
                      dims=(int(nx), int(ny), int(nz)), data=data, triangles=0,
                      mesh_min=tuple(float(x) for x in mlo), mesh_max=tuple(float(x) for x in mhi), layers=count,
                      temp=fr['temp'], vel=fr['vel'])
    return field, reference


# -- a changing field's one grid -------------------------------------------------------------------------

def _frames_of(base, limit=SEQUENCE_FRAMES):
    """The frames a changing field has: a sequence's files, or the USD stage's frames, `limit` of them at most, evenly
    spread ([None]: it does not change)."""
    from ..engine.mesh import is_numbered, sequence_files, split_source
    if is_numbered(base):
        return sorted(sequence_files(base))
    f, prim = split_source(base)
    if prim:
        from . import usd as U
        stage = U.open_stage(f)
        r = U.frame_range(stage)
        if r is None:
            return [None]
        a, b = r
        step = max(1, (b - a) // limit)
        return list(range(a, b + 1, step)) + ([b] if (b - a) % step else [])
    return [None]


def _box_corners(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def field_grid(source, max_cells=FIELD_MAX_CELLS, finest=None):
    """The one grid every frame of a changing field goes on, so the solver can blend between frames: (corner (3,), cell
    (3,), dims (3,), step, splat voxel) in the volume's own frame (metres y-up), around the union of the frames' boxes.
    A VDB frame's box comes from its header (its data is read only when the writer did not store one); a USD points
    frame's from its points. The cells are the first frame's voxels averaged down by `step`, on their lattice, so the
    frames copy across without resampling when their transforms agree."""
    from .vdbread import read_float_grid, vdb_grid_info
    base, zup = split_field(source)
    turn = ZUP if zup else np.eye(4)
    lo_all, hi_all = np.full(3, np.inf), np.full(3, -np.inf)
    first = None
    layers = 1
    vox = None
    points = '#/' in base and _usd_prim(base, None)[2] == 'Points'
    for f in _frames_of(base, POINT_FRAMES if points else SEQUENCE_FRAMES):
        path, to_frame = _frame_file(base, f)
        if path is None:   # USD points: their box, on the splat lattice of the first frame's voxel
            _stage, prim, _t, to_frame = _usd_prim(base, f)
            p, r, v = _points(prim, f)
            if not len(p):
                continue
            if vox is None:
                scale = float(np.mean(np.linalg.norm(to_frame[:3, :3], axis=0)))
                vox = _point_voxel(r, p.min(0) - r.max(), p.max(0) + r.max(), max_cells,
                                   None if not finest else finest / max(scale, 1e-12))
                layers = 1 + 3 * (v is not None)
            R = np.maximum(r, POINT_SPREAD * vox)
            ilo = np.floor((p - R[:, None]).min(0) / vox) - 1
            ihi = np.ceil((p + R[:, None]).max(0) / vox) + 2
            xf = np.diag([vox, vox, vox, 1.0])
        else:
            if not Path(path).exists():
                continue
            dname, tname, vname, info = _vdb_names(path)
            g = info[dname]
            xf = g['xform']
            if g['bbox'] is not None:
                ilo, ihi = g['bbox']
            else:
                dense, ilo, xf = read_float_grid(path, dname, max_cells=400_000_000)
                ihi = np.asarray(ilo) + np.asarray(dense.shape)
            if first is None:
                layers = 1 + (tname is not None) + 3 * (vname is not None)
        m = turn @ to_frame @ xf
        w = (_box_corners(np.asarray(ilo, float) - 0.5, np.asarray(ihi, float) - 0.5) @ m[:3, :3].T) + m[:3, 3]
        lo_all, hi_all = np.minimum(lo_all, w.min(0)), np.maximum(hi_all, w.max(0))
        if first is None:
            first = m
    if first is None:
        raise VolumeError(f'{Path(base.split("#")[0]).name}: no frame of the volume could be read')
    lin, tr = first[:3, :3], first[:3, 3]
    sp = _signed_permutation(lin)
    vs = np.asarray(sp[2], float) if sp is not None else np.full(3, float(np.min(np.linalg.norm(lin, axis=0))))
    shape = np.ceil((hi_all - lo_all) / vs)
    step = _step(shape, np.diag(vs), layers, max_cells, finest)
    cell = vs * step
    anchor = tr + lin @ np.full(3, -0.5)   # the frame-space corner of voxel (0, 0, 0): the lattice the cells sit on
    g0 = anchor + np.floor((lo_all - anchor) / cell - 1e-6) * cell - cell
    g1 = anchor + np.ceil((hi_all - anchor) / cell + 1e-6) * cell + cell
    dims = np.maximum(np.round((g1 - g0) / cell).astype(int), 1)
    return g0, cell, tuple(int(x) for x in dims), int(step), vox


def field_animated(source):
    """Whether a field source changes from frame to frame (a numbered sequence, or a USD volume whose
    file or transform is animated, or USD points that move)."""
    from ..engine.mesh import is_numbered, split_source
    base, _ = split_field(source)
    if is_numbered(base):
        return True
    f, prim = split_source(base)
    if prim:
        from .usd import usd_prim_animated
        return usd_prim_animated(f, prim)
    return False
