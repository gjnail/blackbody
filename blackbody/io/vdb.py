"""OpenVDB writer (file format 224, standard 5-4-3 trees, ZIP-compressed buffers).

Writes sparse float grids (density, temperature, flame, fuel) and a vec3 velocity grid, so a
simulation can be rendered in Blender, Houdini, Maya, Cinema 4D, Unreal or anything that reads VDB.
Only 8^3 leaves that contain data are stored.
"""
from __future__ import annotations

import io
import math
import os
import struct
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

FILE_VERSION = 224
LIB_MAJOR, LIB_MINOR = 10, 0
MAGIC = 0x56444220
COMPRESS_NONE, COMPRESS_ZIP = 0, 1
NO_MASK_AND_ALL_VALS = 6


def _s(buf, text):
    b = text.encode('utf-8')
    buf.write(struct.pack('<I', len(b)))
    buf.write(b)


def _meta(buf, items):
    buf.write(struct.pack('<I', len(items)))
    for name, kind, value in items:
        _s(buf, name)
        _s(buf, kind)
        if kind == 'string':
            b = value.encode('utf-8')
            buf.write(struct.pack('<I', len(b)))
            buf.write(b)
        elif kind == 'vec3i':
            buf.write(struct.pack('<I', 12))
            buf.write(struct.pack('<3i', *value))
        elif kind == 'int64':
            buf.write(struct.pack('<I', 8))
            buf.write(struct.pack('<q', int(value)))
        elif kind == 'float':
            buf.write(struct.pack('<I', 4))
            buf.write(struct.pack('<f', float(value)))
        elif kind == 'bool':
            buf.write(struct.pack('<I', 1))
            buf.write(struct.pack('<?', bool(value)))
        else:
            raise ValueError(kind)


def _mask(bits_on, nbits):
    """Node mask as little-endian uint64 words (bit i of word w is voxel 64 w + i)."""
    bits = np.zeros(nbits, bool)
    bits[np.asarray(bits_on, np.int64)] = True
    return np.packbits(bits, bitorder='little').tobytes()


def _values_bytes(data, z, compress):
    """writeCompressedValues without mask compression: a metadata byte, then the raw or zipped values."""
    head = struct.pack('<b', NO_MASK_AND_ALL_VALS)
    if not compress & COMPRESS_ZIP:
        return head + data
    if len(z) < len(data):
        return head + struct.pack('<q', len(z)) + z
    return head + struct.pack('<q', -len(data)) + data


def _values(buf, arr, compress):
    data = np.ascontiguousarray(arr).tobytes()
    buf.write(_values_bytes(data, zlib.compress(data, 6) if compress & COMPRESS_ZIP else None, compress))


_WORKERS = min(8, os.cpu_count() or 1)
_pool = None


def _zip_all(chunks, compress):
    """Compress many leaf buffers on worker threads (zlib releases the GIL), in batches. The pool
    lives for the whole process so a long sequence does not start threads for every grid."""
    global _pool
    if not compress & COMPRESS_ZIP:
        return [_values_bytes(c, None, compress) for c in chunks]
    if _WORKERS < 2 or len(chunks) < 256:
        return [_values_bytes(c, zlib.compress(c, 6), compress) for c in chunks]
    if _pool is None:
        _pool = ThreadPoolExecutor(_WORKERS, thread_name_prefix='vdb-zip')
    step = -(-len(chunks) // (_WORKERS * 4))
    batches = [chunks[i:i + step] for i in range(0, len(chunks), step)]
    done = _pool.map(lambda b: [_values_bytes(c, zlib.compress(c, 6), compress) for c in b], batches)
    return [v for part in done for v in part]


def _grid_block(data, threshold):
    """Split a (nx, ny, nz[, 3]) array (x-major index order) into the 8^3 leaves that hold active voxels.
    Returns leaf origins (n, 3), values (n, 512[, 3]) with inactive voxels zeroed, and active (n, 512)."""
    vec = data.ndim == 4
    nx, ny, nz = data.shape[:3]
    px, py, pz = (-nx) % 8, (-ny) % 8, (-nz) % 8
    pad = ((0, px), (0, py), (0, pz)) + (((0, 0),) if vec else ())
    d = np.pad(data, pad)
    X, Y, Z = d.shape[0] // 8, d.shape[1] // 8, d.shape[2] // 8
    if vec:
        blocks = d.reshape(X, 8, Y, 8, Z, 8, 3).transpose(0, 2, 4, 1, 3, 5, 6).reshape(X, Y, Z, 512, 3)
        mag = np.abs(blocks).max(axis=-1)
    else:
        blocks = d.reshape(X, 8, Y, 8, Z, 8).transpose(0, 2, 4, 1, 3, 5).reshape(X, Y, Z, 512)
        mag = np.abs(blocks)
    active = mag > threshold
    bx, by, bz = np.nonzero(active.any(axis=-1))
    act = active[bx, by, bz]
    vals = blocks[bx, by, bz]
    vals = np.where(act[..., None] if vec else act, vals, np.float32(0.0)).astype(np.float32)
    return np.stack([bx, by, bz], -1).astype(np.int64) * 8, vals, act


def _write_grid(f, name, data, voxel_size, translation, rotation_y, threshold, grid_class, compress, extra_meta):
    vec = data.ndim == 4
    gtype = 'Tree_vec3s_5_4_3' if vec else 'Tree_float_5_4_3'
    origins, vals, act = _grid_block(data.astype(np.float32), threshold)

    body = io.BytesIO()
    body.write(struct.pack('<I', compress))
    nx, ny, nz = data.shape[:3]
    meta = [('class', 'string', grid_class), ('name', 'string', name),
            ('file_bbox_min', 'vec3i', (0, 0, 0)), ('file_bbox_max', 'vec3i', (nx - 1, ny - 1, nz - 1)),
            ('is_local_space', 'bool', False)]
    if vec:
        meta.append(('vector_type', 'string', 'invariant'))
    meta += extra_meta
    _meta(body, meta)
    # transform: index -> world. Voxel centres sit at integer index coordinates.
    if abs(rotation_y) < 1e-9:
        # ScaleTranslateMap serialises translation, scale, voxel size and three cached inverses
        _s(body, 'UniformScaleTranslateMap')
        v = float(voxel_size)
        body.write(struct.pack('<3d', *translation))
        for x in (v, v, 1.0 / v, 1.0 / (v * v), 0.5 / v):
            body.write(struct.pack('<3d', x, x, x))
    else:
        c, s = math.cos(rotation_y), math.sin(rotation_y)
        m = np.eye(4)
        m[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) * voxel_size
        m[:3, 3] = translation
        _s(body, 'AffineMap')
        body.write(m.T.astype('<f8').tobytes())  # OpenVDB Mat4d is row-vector convention: translation in the last row

    # topology
    body.write(struct.pack('<i', 1))  # buffer count
    bg = (0.0, 0.0, 0.0) if vec else (0.0,)
    body.write(struct.pack('<%df' % len(bg), *bg))
    n = len(origins)
    if not n:
        body.write(struct.pack('<II', 0, 0))
        block_off = body.tell()
    else:
        body.write(struct.pack('<II', 0, 1))
        body.write(struct.pack('<3i', 0, 0, 0))  # the one upper node covers index space [0, 4096)
        ox, oy, oz = origins[:, 0], origins[:, 1], origins[:, 2]
        up = ((ox >> 7) << 10) | ((oy >> 7) << 5) | (oz >> 7)                           # lower node's slot in the upper
        low = (((ox & 127) >> 3) << 8) | (((oy & 127) >> 3) << 4) | ((oz & 127) >> 3)    # leaf's slot in its lower
        order = np.lexsort((low, up))
        up, low = up[order], low[order]
        masks = np.packbits(act[order], axis=1, bitorder='little')                       # (n, 64) leaf value masks
        lowers, starts = np.unique(up, return_index=True)
        ends = np.append(starts[1:], n)
        body.write(_mask(lowers, 32768))
        body.write(_mask([], 32768))
        _values(body, np.zeros((32768, 3) if vec else 32768, np.float32), compress)
        zero_lower = io.BytesIO()
        _values(zero_lower, np.zeros((4096, 3) if vec else 4096, np.float32), compress)
        zero_lower, no_values = zero_lower.getvalue(), _mask([], 4096)
        for s, e in zip(starts, ends):
            body.write(_mask(low[s:e], 4096))
            body.write(no_values)
            body.write(zero_lower)
            body.write(masks[s:e].tobytes())
        block_off = body.tell()
        leaf_vals = vals[order]
        for m, v in zip(masks, _zip_all([leaf_vals[i].tobytes() for i in range(n)], compress)):
            body.write(m.tobytes())
            body.write(v)
    blob = body.getvalue()

    # descriptor header, then offsets (absolute file positions), then the grid
    head = io.BytesIO()
    _s(head, name)
    _s(head, gtype)
    _s(head, '')
    grid_pos = f.tell() + len(head.getvalue()) + 24
    head.write(struct.pack('<3q', grid_pos, grid_pos + block_off, grid_pos + len(blob)))
    f.write(head.getvalue())
    f.write(blob)


def write_vdb(path, grids, voxel_size, translation, rotation_y=0.0, threshold=1e-4, compress=COMPRESS_ZIP, file_meta=None):
    """grids: {name: array}, float arrays (nx, ny, nz) or vec3 arrays (nx, ny, nz, 3), x-major.
    translation: world position of voxel (0,0,0)'s centre."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(struct.pack('<q', MAGIC))
        f.write(struct.pack('<III', FILE_VERSION, LIB_MAJOR, LIB_MINOR))
        f.write(struct.pack('<b', 1))  # has grid offsets
        f.write(str(uuid.uuid4()).encode('ascii'))
        meta = [('creator', 'string', 'Blackbody')] + list(file_meta or [])
        buf = io.BytesIO()
        _meta(buf, meta)
        f.write(buf.getvalue())
        f.write(struct.pack('<i', len(grids)))
        for name, data in grids.items():
            cls = 'fog volume' if data.ndim == 3 else 'unknown'
            _write_grid(f, name, data, voxel_size, translation, rotation_y, threshold, cls, compress, [])


def write_vdb_frame(path, solver, scene, frame=None):
    """Write the solver's current fields as a VDB: density, temperature, flame, fuel and vel, plus
    the optional fields a scene uses: steam (condensed water, g/m^3), vapour (g/m^3), oxygen_used
    (0..1) and color (flame colourant, vec3). With upres, every grid is written at the finer voxel
    size: the fire's own fields as carried on the finer grid, the others repeated onto it."""
    k = max(1, int(getattr(solver, 'upres', 1)))
    fine = (lambda a: a) if k == 1 else (lambda a: np.repeat(np.repeat(np.repeat(a, k, 0), k, 1), k, 2))
    sc = (solver.read_scalars_fine() if k > 1 else solver.read_scalars()).astype(np.float32)  # (z, y, x, 4)
    vel = fine(solver.read_velocity_centres().astype(np.float32))  # (z, y, x, 3), fire-local
    to_x = lambda a: np.ascontiguousarray(np.transpose(a, (2, 1, 0) + tuple(range(3, a.ndim))))
    grids = {'density': to_x(sc[..., 2]), 'temperature': to_x(sc[..., 0]), 'flame': to_x(sc[..., 3]),
             'fuel': to_x(sc[..., 1]), 'vel': to_x(vel)}
    feats = getattr(solver, 'features', {}) or {}
    aux = solver.read_aux() if hasattr(solver, 'read_aux') else None
    if aux is not None:
        aux = fine(aux.astype(np.float32))
        if feats.get('vapour'):
            from ..engine.renderer import vapour_saturation
            shade = scene.data['shading']
            amb, flame_k, max_k = shade['ambient_k'], shade['flame_k'], shade['max_k']
            t = np.maximum(sc[..., 0], 0.0)
            kelvin = amb + (flame_k - amb) * np.minimum(t, 1.0) + (max(max_k, flame_k + 1.0) - flame_k) * (1.0 - np.exp(-np.maximum(t - 1.0, 0.0)))
            tc = kelvin - 273.15
            sat = 610.94 * np.exp(np.minimum(17.625 * tc / np.maximum(tc + 243.04, 1.0), 60.0)) / (461.5 * kelvin) * 1000.0
            q_air = min(max(shade['humidity'], 0.0), 100.0) / 100.0 * vapour_saturation(amb)
            grids['vapour'] = to_x(np.maximum(aux[..., 1], 0.0))
            grids['steam'] = to_x(np.maximum(q_air + aux[..., 1] - sat, 0.0).astype(np.float32))
        if feats.get('oxygen'):
            grids['oxygen_used'] = to_x(aux[..., 0])
    chem = solver.read_chem() if hasattr(solver, 'read_chem') else None
    if chem is not None:
        grids['color'] = to_x(fine(chem[..., :3].astype(np.float32)))  # Blender and Houdini look for 'color'
    spec, fire = scene.camera(frame if frame is not None else scene.start)
    h = solver.h / k
    yaw = math.radians(fire.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    corner_local = np.array(solver.origin) + 0.5 * h
    rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    translation = tuple(rot @ corner_local + np.array(fire.position))
    look = scene.data['shading']
    meta = [('blackbody_temperature_1_kelvin', 'float', float(look['flame_k'])),
            ('blackbody_ambient_kelvin', 'float', float(look['ambient_k']))]
    write_vdb(path, grids, h, translation, yaw, file_meta=meta)


def write_liquid_vdb_frame(path, engine, scene, frame):
    """Write a liquid frame as a VDB on the surface grid: 'density' (1 inside the liquid, 0 outside,
    with a one-voxel ramp across the surface, so the 0.5 iso-surface is the liquid surface: mesh it
    with Blender's Volume to Mesh or Houdini's Convert VDB), 'vel' (m/s, for motion blur) and the
    whitewater densities 'spray', 'foam' and 'bubbles'."""
    vol, _ = engine.volume_for(scene, frame)
    if vol is None:
        raise RuntimeError(f'frame {frame} is neither simulated nor cached')
    look = scene.water_look(frame, final=True)
    lr = engine.liquid_r
    with engine.gpu.batch() as b:
        lr.build(b, vol, look)
    surf = engine.gpu.read(lr.surf).astype(np.float32)     # (z, y, x, 4): distance (surface cells), velocity
    ww = engine.gpu.read(lr.ww_tex).astype(np.float32)     # (z, y, x, 4): spray, foam, bubbles, density
    to_x = lambda a: np.ascontiguousarray(np.transpose(a, (2, 1, 0) + tuple(range(3, a.ndim))))
    density = np.clip(0.5 - surf[..., 0], 0.0, 1.0)
    grids = {'density': to_x(density), 'vel': to_x(surf[..., 1:4] * (density[..., None] > 0.0))}
    for i, name in enumerate(('spray', 'foam', 'bubbles')):
        if ww[..., i].max() > 1e-4:
            grids[name] = to_x(ww[..., i])
    spec, fire = scene.camera(frame)
    n = vol.dims
    vs = vol.h * n[0] / lr.nf[0]
    yaw = math.radians(fire.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    corner_local = np.array(vol.origin) + 0.5 * vs
    rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    translation = tuple(rot @ corner_local + np.array(fire.position))
    meta = [('blackbody_liquid_surface_iso', 'float', 0.5)]
    write_vdb(path, grids, vs, translation, yaw, threshold=1e-3, file_meta=meta)


# ---------------------------------------------------------------------------------------------
# Reader (for verification and for re-importing Blackbody caches)
# ---------------------------------------------------------------------------------------------

def read_vdb(path):
    """Read grids written by write_vdb (and simple standard VDBs with the same features)."""
    data = Path(path).read_bytes()
    pos = 0

    def take(n):
        nonlocal pos
        b = data[pos:pos + n]
        pos += n
        return b

    def u32():
        return struct.unpack('<I', take(4))[0]

    def string():
        return take(u32()).decode('utf-8')

    def read_meta():
        out = {}
        for _ in range(u32()):
            name, kind = string(), string()
            size = u32()
            raw = take(size)
            out[name] = raw.decode('utf-8', 'replace') if kind == 'string' else raw
        return out

    magic = struct.unpack('<q', take(8))[0]
    assert magic == MAGIC, 'not a VDB file'
    version, _, _ = struct.unpack('<III', take(12))
    has_offsets = struct.unpack('<b', take(1))[0]
    take(36)
    file_meta = read_meta()
    n = struct.unpack('<i', take(4))[0]
    grids = {}
    for _ in range(n):
        name, gtype, _parent = string(), string(), string()
        gpos, bpos, epos = struct.unpack('<3q', take(24))
        comp = u32()
        meta = read_meta()
        mtype = string()
        if mtype in ('UniformScaleTranslateMap', 'ScaleTranslateMap'):
            tr = struct.unpack('<3d', take(24))
            sc = struct.unpack('<3d', take(24))
            take(24 * 4)  # voxel size and cached inverses
            xform = {'translation': tr, 'voxel': sc[0]}
        elif mtype in ('UniformScaleMap', 'ScaleMap'):
            sc = struct.unpack('<3d', take(24))
            take(24 * 4)
            xform = {'translation': (0.0, 0.0, 0.0), 'voxel': sc[0]}
        else:
            m = np.frombuffer(take(128), '<f8').reshape(4, 4).T
            xform = {'matrix': m}
        vec = 'vec3' in gtype
        vsize = 12 if vec else 4
        take(4)
        take(vsize)  # background
        tiles, children = struct.unpack('<II', take(8))

        def values(count):
            take(1)
            nbytes = count * vsize
            if comp & COMPRESS_ZIP:
                k = struct.unpack('<q', take(8))[0]
                raw = take(k) if k > 0 else take(-k)
                buf = zlib.decompress(raw) if k > 0 else raw
            else:
                buf = take(nbytes)
            return np.frombuffer(buf, np.float32).reshape((count, 3) if vec else (count,))

        def mask_bits(nbits):
            w = np.frombuffer(take(nbits // 8), np.uint64)
            return [i for i in range(nbits) if (int(w[i >> 6]) >> (i & 63)) & 1]

        leaves = []
        for _c in range(children):
            ox, oy, oz = struct.unpack('<3i', take(12))
            upper_children = mask_bits(32768)
            mask_bits(32768)
            values(32768)
            for uc in upper_children:
                lx = ox + ((uc >> 10) & 31) * 128
                ly = oy + ((uc >> 5) & 31) * 128
                lz = oz + (uc & 31) * 128
                lower_children = mask_bits(4096)
                mask_bits(4096)
                values(4096)
                for lc in lower_children:
                    leaves.append((lx + ((lc >> 8) & 15) * 8, ly + ((lc >> 4) & 15) * 8, lz + (lc & 15) * 8))
                    mask_bits(512)
        vox = {}
        for o in leaves:
            mask_bits(512)
            vox[o] = values(512)
        grids[name] = {'type': gtype, 'meta': meta, 'xform': xform, 'leaves': vox, 'compression': comp}
        pos = epos
    return {'version': version, 'meta': file_meta, 'grids': grids}


def dense_from_leaves(leaves, shape, vec=False):
    out = np.zeros(tuple(shape) + ((3,) if vec else ()), np.float32)
    for (x, y, z), vals in leaves.items():
        block = vals.reshape((8, 8, 8, 3) if vec else (8, 8, 8))
        sx, sy, sz = min(8, shape[0] - x), min(8, shape[1] - y), min(8, shape[2] - z)
        if sx > 0 and sy > 0 and sz > 0:
            out[x:x + sx, y:y + sy, z:z + sz] = block[:sx, :sy, :sz]
    return out
