"""Cryptomatte (Psyop's specification, version 1.2): ID mattes any compositor with a Cryptomatte node can pick from.

A Cryptomatte layer named N is RANKS (id, coverage) pairs per pixel, the most covering first, two to each RGBA layer:
N00.R = the first id, N00.G = its coverage, N00.B = the second id, N00.A = its coverage, N01 the next two, and so on,
in 32-bit float (an id is the bits of a 32-bit hash of its name). The EXR's header holds, under
cryptomatte/<key>/ (key: the first 7 hex digits of the hash of N), its name, the hash (MurmurHash3_32), the
conversion (uint32_to_float32) and the manifest: every name and its hash in hex, as JSON. Nuke's Cryptomatte node,
Fusion's, Houdini's and Blender's read them.
"""
from __future__ import annotations

import json
import struct

import numpy as np

RANKS = 6    # (id, coverage) pairs per pixel: three RGBA layers


def murmur3_32(data, seed=0):
    """MurmurHash3_x86_32 of bytes (or a str, as UTF-8): an unsigned 32-bit int."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    c1, c2, m = 0xcc9e2d51, 0x1b873593, 0xffffffff
    h = seed & m
    n = len(data) // 4 * 4
    for i in range(0, n, 4):
        k = int.from_bytes(data[i:i + 4], 'little')
        k = (k * c1) & m
        k = ((k << 15) | (k >> 17)) & m
        k = (k * c2) & m
        h ^= k
        h = ((h << 13) | (h >> 19)) & m
        h = (h * 5 + 0xe6546b64) & m
    k = 0
    tail = data[n:]
    if len(tail) >= 3:
        k ^= tail[2] << 16
    if len(tail) >= 2:
        k ^= tail[1] << 8
    if tail:
        k ^= tail[0]
        k = (k * c1) & m
        k = ((k << 15) | (k >> 17)) & m
        k = (k * c2) & m
        h ^= k
    h ^= len(data)
    h ^= h >> 16
    h = (h * 0x85ebca6b) & m
    h ^= h >> 13
    h = (h * 0xc2b2ae35) & m
    h ^= h >> 16
    return h


def name_bits(name):
    """A name's id as the bits of a 32-bit float (uint32): its hash, its exponent moved off all zeros and all ones
    (no denormals, infinities or NaNs, which a compositor's maths would mangle)."""
    h = murmur3_32(name)
    e = (h >> 23) & 0xff
    if e in (0, 255):
        h ^= 1 << 23
    return h


def name_id(name):
    """A name's id as the float a Cryptomatte layer holds."""
    return struct.unpack('<f', struct.pack('<I', name_bits(name)))[0]


def layer_key(layer):
    """The 7 hex digits a layer's metadata is filed under."""
    return f'{murmur3_32(layer):08x}'[:7]


def manifest(names):
    """{name: hex id} for the header, as JSON."""
    return json.dumps({n: f'{name_bits(n):08x}' for n in sorted(set(names))}, separators=(',', ':'))


def header(layer, names):
    """The header attributes of Cryptomatte layer `layer` over these names."""
    key = f'cryptomatte/{layer_key(layer)}/'
    return {key + 'name': layer, key + 'hash': 'MurmurHash3_32', key + 'conversion': 'uint32_to_float32',
            key + 'manifest': manifest(names)}


def merge_ranks(index, cover, ranks=RANKS):
    """Per pixel, the coverage of the same thing added up and the most covering first: index (..., k) int (negative:
    nothing), cover (..., k) -> (index, cover), each (..., ranks)."""
    idx = np.array(index, np.int64, copy=True)
    cov = np.where(idx >= 0, np.asarray(cover, np.float32), 0.0).astype(np.float32)
    k = idx.shape[-1]
    for j in range(1, k):
        for i in range(j):
            same = (idx[..., i] == idx[..., j]) & (idx[..., j] >= 0) & (cov[..., j] > 0.0)
            cov[..., i] += np.where(same, cov[..., j], 0.0)
            cov[..., j] = np.where(same, 0.0, cov[..., j])
    cov = np.where(cov > 0.0, cov, 0.0)
    order = np.argsort(-cov, axis=-1, kind='stable')[..., :ranks]
    idx = np.take_along_axis(idx, order, -1)
    cov = np.take_along_axis(cov, order, -1)
    idx = np.where(cov > 0.0, idx, -1)
    if idx.shape[-1] < ranks:
        pad = ranks - idx.shape[-1]
        idx = np.concatenate([idx, np.full(idx.shape[:-1] + (pad,), -1, np.int64)], -1)
        cov = np.concatenate([cov, np.zeros(cov.shape[:-1] + (pad,), np.float32)], -1)
    return idx, cov


def channels(layer, index, cover, names, ranks=RANKS):
    """Cryptomatte layer `layer`: (channels {name: (h, w) float32}, header attributes). index (h, w, k): per pixel what
    it sees (an index into names, negative: nothing; several may share a name, their coverage is added), cover (h, w, k)
    its coverage of the pixel. (Only the pixels where more than one thing shows are merged and sorted: the rest, most
    of a picture, keep their one.)"""
    index = np.asarray(index)
    cover = np.asarray(cover, np.float32)
    h, w, k = index.shape
    fi, fc = index.reshape(-1, k), cover.reshape(-1, k)
    multi = np.nonzero(np.any(fc[:, 1:] > 0.0, axis=1))[0] if k > 1 else np.zeros(0, np.int64)
    idx = np.full((ranks, h * w), -1, np.int32)
    cov = np.zeros((ranks, h * w), np.float32)
    has = (fc[:, 0] > 0.0) & (fi[:, 0] >= 0)
    idx[0] = np.where(has, fi[:, 0], -1)
    cov[0] = np.where(has, fc[:, 0], np.float32(0.0))
    if multi.size:
        mi, mc = merge_ranks(fi[multi], fc[multi], ranks)
        idx[:, multi], cov[:, multi] = mi.T, mc.T
    bits = np.array([name_bits(n) for n in names] + [0], np.uint32)   # (the last: nothing, id 0)
    ids = bits[np.where(idx >= 0, idx, len(names))].view(np.float32)
    out = {}
    for r in range(0, ranks, 2):
        pre = f'{layer}{r // 2:02d}.'
        out[pre + 'R'] = ids[r].reshape(h, w)
        out[pre + 'G'] = cov[r].reshape(h, w)
        out[pre + 'B'] = ids[r + 1].reshape(h, w)
        out[pre + 'A'] = cov[r + 1].reshape(h, w)
    return out, header(layer, [n for n in names if n])


def decode(chans, attrs, layer):
    """What a Cryptomatte layer says (a reader, as a compositor reads one): (names {float id bits: name} from its
    manifest, coverage {name: (h, w) matte}) - for checking a written file."""
    key = f'cryptomatte/{layer_key(layer)}/'
    man = json.loads(attrs[key + 'manifest'])
    by_bits = {int(h, 16): n for n, h in man.items()}
    mattes = {}
    r = 0
    while f'{layer}{r:02d}.R' in chans:
        pre = f'{layer}{r:02d}.'
        for c_id, c_cov in (('R', 'G'), ('B', 'A')):
            b = np.asarray(chans[pre + c_id], np.float32).view(np.uint32)
            c = np.asarray(chans[pre + c_cov], np.float32)
            for bits in np.unique(b[c > 0.0]):
                name = by_bits.get(int(bits))
                if name is None:
                    continue
                m = mattes.setdefault(name, np.zeros(c.shape, np.float32))
                m += np.where(b == bits, c, 0.0)
        r += 1
    return by_bits, mattes
