"""Reading OpenVDB files written by other programs (Houdini, Blender, EmberGen, OpenVDB itself).

io/vdb.py writes VDBs and reads back its own. This reader covers what other writers produce for
float grids (fog volumes, level sets): root tiles, the per-node compression of inactive values
(OpenVDB's seven "mask compression" modes), grids saved as half floats, and ZIP or Blosc compression
(Blosc with its LZ4 or zlib codecs, byte-shuffled or not, decoded here in pure Python). Vector
grids are read as their three components. Zstd-compressed Blosc and pre-2015 files (format < 222)
are not supported and say so.

read_float_grid(path, name) gives a grid as a dense array over its active bounding box, with the
index-to-world transform, which is all a USD volume or a mesh bake needs.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

MAGIC = 0x56444220
COMPRESS_ZIP = 0x1
COMPRESS_ACTIVE_MASK = 0x2
COMPRESS_BLOSC = 0x4
# per-node metadata: how the inactive values of a node were stored
NO_MASK_OR_INACTIVE_VALS, NO_MASK_AND_MINUS_BG, NO_MASK_AND_ONE_INACTIVE_VAL, MASK_AND_NO_INACTIVE_VALS, \
    MASK_AND_ONE_INACTIVE_VAL, MASK_AND_TWO_INACTIVE_VALS, NO_MASK_AND_ALL_VALS = range(7)
LOG2 = (5, 4, 3)   # the standard tree: 32^3 upper internal nodes, 16^3 lower, 8^3 leaves


class VDBError(ValueError):
    pass


# -- Blosc and LZ4 -----------------------------------------------------------------------------------------

def lz4_block(src, size):
    """Decode one raw LZ4 block to `size` bytes."""
    src = memoryview(src)
    out = bytearray()
    i, n = 0, len(src)
    while i < n:
        token = src[i]
        i += 1
        lit = token >> 4
        if lit == 15:
            while True:
                b = src[i]
                i += 1
                lit += b
                if b != 255:
                    break
        out += src[i:i + lit]
        i += lit
        if i >= n or len(out) >= size:
            break
        off = src[i] | (src[i + 1] << 8)
        i += 2
        ml = token & 15
        if ml == 15:
            while True:
                b = src[i]
                i += 1
                ml += b
                if b != 255:
                    break
        ml += 4
        start = len(out) - off
        if off <= 0 or start < 0:
            raise VDBError('corrupt LZ4 data')
        if off >= ml:
            out += out[start:start + ml]
        else:   # an overlapping copy repeats the last `off` bytes
            pat = bytes(out[start:])
            out += (pat * (ml // off + 1))[:ml]
    if len(out) != size:
        raise VDBError(f'LZ4 block decoded to {len(out)} bytes, expected {size}')
    return bytes(out)


def _unshuffle(buf, typesize):
    b = np.frombuffer(buf, np.uint8)
    n = len(b) // typesize
    if typesize <= 1 or n == 0:
        return bytes(buf)
    body = b[:n * typesize].reshape(typesize, n).T.reshape(-1)
    return body.tobytes() + bytes(b[n * typesize:])


def blosc_decompress(buf):
    """Decode a Blosc 1 buffer (blosclz is not supported; LZ4, LZ4HC and zlib are)."""
    buf = memoryview(buf)
    if len(buf) < 16:
        raise VDBError('truncated Blosc buffer')
    flags, typesize = buf[2], buf[3]
    nbytes, blocksize, cbytes = struct.unpack_from('<III', buf, 4)
    if flags & 0x2 or nbytes == 0:   # copied as is (small buffers are), never shuffled
        return bytes(buf[16:16 + nbytes])
    blocksize = max(blocksize, 1)
    codec = (flags >> 5) & 0x7
    if flags & 0x4:
        raise VDBError('bit-shuffled Blosc data is not supported')
    nblocks = (nbytes + blocksize - 1) // blocksize
    starts = struct.unpack_from(f'<{nblocks}I', buf, 16)
    dont_split = bool(flags & 0x10)
    out = bytearray()
    for k in range(nblocks):
        last = k == nblocks - 1 and nbytes % blocksize
        bsize = (nbytes % blocksize) if last else blocksize
        nsplits = typesize if (not dont_split and not last and typesize <= 16 and bsize // typesize >= 128) else 1
        neblock = bsize // nsplits
        pos = starts[k]
        block = bytearray()
        for _ in range(nsplits):
            csize = struct.unpack_from('<i', buf, pos)[0]
            pos += 4
            chunk = bytes(buf[pos:pos + csize])
            pos += csize
            if csize == neblock:
                block += chunk
            elif codec == 1:
                block += lz4_block(chunk, neblock)
            elif codec == 3:
                block += zlib.decompress(chunk)
            elif codec == 4:
                raise VDBError('Zstd-compressed Blosc data is not supported: save the VDB with ZIP or Blosc LZ4 compression')
            else:
                raise VDBError(f'Blosc codec {codec} is not supported')
        out += _unshuffle(bytes(block), typesize) if flags & 0x1 else block
    return bytes(out[:nbytes])


# -- the file format -------------------------------------------------------------------------------------

class _Reader:
    def __init__(self, data):
        self.d = data
        self.p = 0

    def take(self, n):
        b = self.d[self.p:self.p + n]
        if len(b) < n:
            raise VDBError('the VDB file is truncated')
        self.p += n
        return b

    def u32(self):
        return struct.unpack('<I', self.take(4))[0]

    def i32(self):
        return struct.unpack('<i', self.take(4))[0]

    def i64(self):
        return struct.unpack('<q', self.take(8))[0]

    def string(self):
        return self.take(self.u32()).decode('utf-8', 'replace')

    def meta(self):
        out = {}
        for _ in range(self.u32()):
            name, kind = self.string(), self.string()
            raw = self.take(self.u32())
            if kind == 'string':
                out[name] = raw.decode('utf-8', 'replace')
            elif kind == 'bool':
                out[name] = bool(raw[0]) if raw else False
            elif kind in ('int32', 'int64', 'float', 'double'):
                fmt = {'int32': '<i', 'int64': '<q', 'float': '<f', 'double': '<d'}[kind]
                out[name] = struct.unpack(fmt, raw)[0]
            else:
                out[name] = raw
        return out

    def mask(self, nbits):
        words = np.frombuffer(self.take(nbits // 8), '<u8')
        return np.unpackbits(words.view(np.uint8), bitorder='little').astype(bool)


class _Grid:
    def __init__(self, r: _Reader, gtype, compression, half):
        self.r = r
        self.comp = compression
        self.half = half
        self.vec = 'vec3' in gtype
        self.dtype_size = 12 if self.vec else 4
        self.comps = 3 if self.vec else 1

    def _data(self, count):
        """`count` values (each self.comps floats) as stored: compressed or not, as half or float."""
        r = self.r
        width = 2 if self.half else 4
        nbytes = count * self.comps * width
        if self.comp & (COMPRESS_ZIP | COMPRESS_BLOSC):
            k = r.i64()
            if k <= 0:
                raw = r.take(-k)
            elif self.comp & COMPRESS_BLOSC:
                raw = blosc_decompress(r.take(k))
            else:
                raw = zlib.decompress(r.take(k))
        else:
            raw = r.take(nbytes)
        a = np.frombuffer(raw[:nbytes], '<f2' if self.half else '<f4').astype(np.float32)
        return a.reshape(count, self.comps) if self.vec else a

    def values(self, count, value_mask, background):
        """OpenVDB's readCompressedValues: a node's `count` values given its value mask."""
        r = self.r
        meta = r.take(1)[0]
        bg = np.asarray(background, np.float32)
        inactive0 = bg if meta == NO_MASK_OR_INACTIVE_VALS else -bg
        inactive1 = np.zeros_like(bg)
        if meta in (NO_MASK_AND_ONE_INACTIVE_VAL, MASK_AND_ONE_INACTIVE_VAL, MASK_AND_TWO_INACTIVE_VALS):
            inactive0 = np.frombuffer(r.take(self.dtype_size), '<f4').copy()
            if meta == MASK_AND_TWO_INACTIVE_VALS:
                inactive1 = np.frombuffer(r.take(self.dtype_size), '<f4').copy()
            elif meta == MASK_AND_ONE_INACTIVE_VAL:
                inactive1 = bg
        elif meta == MASK_AND_NO_INACTIVE_VALS:
            inactive0, inactive1 = bg, -bg
        select = r.mask(count) if meta in (MASK_AND_NO_INACTIVE_VALS, MASK_AND_ONE_INACTIVE_VAL, MASK_AND_TWO_INACTIVE_VALS) else None
        if (self.comp & COMPRESS_ACTIVE_MASK) and meta != NO_MASK_AND_ALL_VALS:
            active = self._data(int(value_mask.sum()))
            shape = (count, 3) if self.vec else (count,)
            out = np.empty(shape, np.float32)
            fill = np.where((select if select is not None else np.zeros(count, bool))[:, None] if self.vec else
                            (select if select is not None else np.zeros(count, bool)),
                            inactive1 if not self.vec else inactive1.reshape(1, -1),
                            inactive0 if not self.vec else inactive0.reshape(1, -1))
            out[:] = fill
            out[value_mask] = active
            return out
        return self._data(count)


def _read_map(r: _Reader):
    kind = r.string()
    if kind in ('UniformScaleTranslateMap', 'ScaleTranslateMap'):
        tr = struct.unpack('<3d', r.take(24))
        sc = struct.unpack('<3d', r.take(24))
        r.take(24 * 4)   # voxel size and cached inverses
        m = np.diag([sc[0], sc[1], sc[2], 1.0])
        m[:3, 3] = tr
        return m
    if kind in ('UniformScaleMap', 'ScaleMap'):
        sc = struct.unpack('<3d', r.take(24))
        r.take(24 * 4)
        return np.diag([sc[0], sc[1], sc[2], 1.0])
    if kind == 'TranslationMap':
        tr = struct.unpack('<3d', r.take(24))
        m = np.eye(4)
        m[:3, 3] = tr
        return m
    if kind == 'AffineMap':
        return np.frombuffer(r.take(128), '<f8').reshape(4, 4).T.copy()
    raise VDBError(f'VDB transform "{kind}" is not supported')


def vdb_grid_names(path):
    """[(name, type)] of the grids in a VDB file, from its header (the grids themselves are not read)."""
    r = _Reader(Path(path).read_bytes())
    if struct.unpack('<q', r.take(8))[0] != MAGIC:
        raise VDBError(f'{Path(path).name} is not a VDB file')
    r.u32()
    r.take(8)
    has_offsets = r.take(1)[0]
    r.take(36)
    r.meta()
    if not has_offsets:
        raise VDBError('VDB files without grid offsets are not supported')
    out = []
    for _ in range(r.i32()):
        name = r.string().split('\x1e')[0]
        gtype = r.string()
        r.string()
        r.i64(), r.i64()
        r.p = r.i64()
        out.append((name, gtype))
    return out


def read_vdb_grids(path, names=None):
    """{grid name: {'type', 'xform' (4x4 index -> world), 'background', 'leaves': {origin: values (512,) or (512, 3)},
    'tiles': [(origin, size, value)]}} for the float or vec3 grids of a VDB file (only `names`, if given)."""
    r = _Reader(Path(path).read_bytes())
    if struct.unpack('<q', r.take(8))[0] != MAGIC:
        raise VDBError(f'{Path(path).name} is not a VDB file')
    version = r.u32()
    if version < 222:
        raise VDBError(f'{Path(path).name}: VDB format {version} is too old (resave it with a current OpenVDB)')
    r.take(8)                    # library version
    has_offsets = r.take(1)[0]
    r.take(36)                   # uuid
    r.meta()                     # file metadata
    if not has_offsets:
        raise VDBError('VDB files without grid offsets are not supported')
    grids = {}
    for _ in range(r.i32()):
        name = r.string().split('\x1e')[0]
        gtype = r.string()
        r.string()               # instance parent
        gpos, bpos, epos = r.i64(), r.i64(), r.i64()
        keep = ('float' in gtype.lower() or 'vec3' in gtype.lower()) and (names is None or name in names)
        if not keep:
            r.p = epos
            continue
        r.p = gpos
        comp = r.u32()
        meta = r.meta()
        xform = _read_map(r)
        half = bool(meta.get('is_saved_as_half_float', False))
        g = _Grid(r, gtype, comp, half)
        r.take(4)                # buffer count
        bg = np.frombuffer(r.take(g.dtype_size), '<f4').copy()
        n_tiles, n_children = r.u32(), r.u32()
        tiles = []
        for _t in range(n_tiles):
            o = struct.unpack('<3i', r.take(12))
            v = np.frombuffer(r.take(g.dtype_size), '<f4').copy()
            active = r.take(1)[0]
            if active:
                tiles.append((o, 1 << sum(LOG2), v))
        leaf_masks = []          # (origin, value mask), in file order
        for _c in range(n_children):
            o = struct.unpack('<3i', r.take(12))
            _read_internal(r, g, o, 0, bg, tiles, leaf_masks)
        leaves = {}
        for o, _m in leaf_masks:
            vm = r.mask(512)
            leaves[o] = g.values(512, vm, bg)
        grids[name] = {'type': gtype, 'xform': xform, 'background': bg, 'leaves': leaves, 'tiles': tiles,
                       'class': str(meta.get('class', ''))}
        r.p = epos
    return grids


def _read_internal(r, g, origin, level, bg, tiles, leaf_masks):
    log2 = LOG2[level]
    n = 1 << (3 * log2)
    child = r.mask(n)
    value = r.mask(n)
    vals = g.values(n, value, bg)
    child_log = sum(LOG2[level + 1:])
    size = 1 << child_log
    idx = np.nonzero(value & ~child)[0]
    for i in idx:   # active tiles inside the node
        tiles.append((_offset(origin, i, log2, child_log), size, vals[i]))
    for i in np.nonzero(child)[0]:
        o = _offset(origin, i, log2, child_log)
        if level + 1 < len(LOG2) - 1:
            _read_internal(r, g, o, level + 1, bg, tiles, leaf_masks)
        else:
            leaf_masks.append((o, r.mask(512)))


def _offset(origin, i, log2, child_log):
    i = int(i)
    x = (i >> (2 * log2)) & ((1 << log2) - 1)
    y = (i >> log2) & ((1 << log2) - 1)
    z = i & ((1 << log2) - 1)
    return (origin[0] + (x << child_log), origin[1] + (y << child_log), origin[2] + (z << child_log))


def read_float_grid(path, name=None, max_cells=64_000_000, with_class=False, with_background=False):
    """(dense (nx, ny, nz) float32 over the active bounding box, index of its first cell (3,), 4x4
    index -> world transform[, grid class][, background]) of a float grid (the first float grid if `name`
    is None). Vector grids give their length. Voxels outside every leaf and tile hold the background."""
    grids = read_vdb_grids(path, None if name is None else [name])
    if not grids:
        raise VDBError(f'{Path(path).name} has no {"grid " + name if name else "float grid"}')
    g = grids[name] if name in grids else next(iter(grids.values()))
    lo, hi = np.full(3, 2**31 - 1), np.full(3, -2**31)
    for o in g['leaves']:
        lo, hi = np.minimum(lo, o), np.maximum(hi, np.asarray(o) + 8)
    for o, size, _v in g['tiles']:
        lo, hi = np.minimum(lo, o), np.maximum(hi, np.asarray(o) + size)
    bg = float(np.linalg.norm(g['background'])) if g['background'].size > 1 else float(g['background'][0])
    extra = ((g['class'],) if with_class else ()) + ((bg,) if with_background else ())
    if (hi <= lo).any():
        return (np.full((1, 1, 1), bg, np.float32), np.zeros(3, int), g['xform']) + extra
    shape = hi - lo
    if int(np.prod(shape)) > max_cells:
        raise VDBError(f'{Path(path).name}: the grid is too large to read densely ({tuple(shape)} voxels)')
    out = np.full(tuple(shape), bg, np.float32)
    for o, size, v in g['tiles']:
        a = np.asarray(o) - lo
        v = np.atleast_1d(v)
        out[a[0]:a[0] + size, a[1]:a[1] + size, a[2]:a[2] + size] = np.linalg.norm(v) if v.size > 1 else v[0]
    for o, vals in g['leaves'].items():
        a = np.asarray(o) - lo
        block = (np.linalg.norm(vals, axis=-1) if vals.ndim > 1 else vals).reshape(8, 8, 8)
        out[a[0]:a[0] + 8, a[1]:a[1] + 8, a[2]:a[2] + 8] = block
    return (out, lo, g['xform']) + extra


def voxel_surface(occ):
    """Closed, outward-facing triangles around the true cells of a boolean grid, in index space (cell
    (i, j, k) spans i..i+1): the surface a volume's iso-level makes, cell by cell."""
    occ = np.pad(np.asarray(occ, bool), 1)
    corners = []
    for axis in range(3):
        u, w = [a for a in range(3) if a != axis]
        d = np.diff(occ.astype(np.int8), axis=axis)
        idx = np.argwhere(d != 0)           # the face between padded cells c and c + 1 along the axis
        if not len(idx):
            continue
        base = idx - 1                       # unpadded: the cell's corner ...
        base[:, axis] = idx[:, axis]         # ... on the face's plane
        eu, ew = np.eye(3, dtype=int)[u], np.eye(3, dtype=int)[w]
        corners.append(np.stack([base, base + eu, base + eu + ew, base + ew], 1))
    if not corners:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    q = np.concatenate(corners)                            # (quads, 4, 3)
    v, inv = np.unique(q.reshape(-1, 3), axis=0, return_inverse=True)
    quad = inv.reshape(-1, 4)
    t = np.concatenate([quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]])
    return v.astype(np.float64), _orient_out(v.astype(np.float64), t, occ)


def _orient_out(v, t, occ_padded):
    """Turn every face to point out of the solid: a probe a little along its normal must be outside."""
    p = v[t]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    c = p.mean(1)
    probe = np.floor(c + 0.25 * n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)).astype(int) + 1
    probe = np.clip(probe, 0, np.array(occ_padded.shape) - 1)
    inside = occ_padded[probe[:, 0], probe[:, 1], probe[:, 2]]
    t = t.copy()
    t[inside] = t[inside][:, ::-1]
    return t
