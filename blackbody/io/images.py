"""Still-image writers: OpenEXR (multi-layer, half or float) and PNG (8 or 16 bit)."""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

import blackbody

EXR_COMPRESSION = {'none': 'NO_COMPRESSION', 'zip': 'ZIP_COMPRESSION', 'zips': 'ZIPS_COMPRESSION',
                   'piz': 'PIZ_COMPRESSION', 'dwaa': 'DWAA_COMPRESSION', 'dwab': 'DWAB_COMPRESSION',
                   'rle': 'RLE_COMPRESSION', 'zstd': 'ZSTD_COMPRESSION'}


def write_exr(path, channels, compression='zip', attrs=None):
    """Write an OpenEXR file. `channels` maps names ('R', 'A', 'emission.R', 'heat.Y') to (h, w) arrays
    of float16 or float32; names that share a prefix become layers in Nuke and After Effects."""
    import OpenEXR
    comp = getattr(OpenEXR, EXR_COMPRESSION.get(compression, 'ZIP_COMPRESSION'), OpenEXR.ZIP_COMPRESSION)
    header = {'compression': comp, 'type': OpenEXR.scanlineimage}
    for k, v in (attrs or {}).items():
        header[k] = v
    ch = {k: np.ascontiguousarray(v) for k, v in channels.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with OpenEXR.File(header, ch) as f:
        f.write(str(path))


def _chunk(kind, data):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xFFFFFFFF)


def encode_png(img, level=6):
    """PNG bytes from (h, w[, c]) uint8 or uint16, c in 1-4."""
    a = np.asarray(img)
    if a.ndim == 2:
        a = a[..., None]
    h, w, c = a.shape
    bits = 16 if a.dtype == np.uint16 else 8
    if bits == 8 and a.dtype != np.uint8:
        raise TypeError('PNG needs uint8 or uint16 data')
    color = {1: 0, 2: 4, 3: 2, 4: 6}[c]
    raw = (a.astype('>u2') if bits == 16 else a).reshape(h, -1).view(np.uint8)
    up = raw.copy()
    up[1:] = raw[1:] - raw[:-1]  # "Up" filter; uint8 arithmetic wraps as PNG expects
    rows = np.concatenate([np.full((h, 1), 2, np.uint8), up], axis=1)
    ihdr = struct.pack('>IIBBBBB', w, h, bits, color, 0, 0, 0)
    text = b'Software\x00' + f'{blackbody.APP_NAME} {blackbody.__version__}'.encode()
    return (b'\x89PNG\r\n\x1a\n' + _chunk(b'IHDR', ihdr) + _chunk(b'tEXt', text)
            + _chunk(b'IDAT', zlib.compress(rows.tobytes(), level)) + _chunk(b'IEND', b''))


def write_png(path, img, level=6):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(encode_png(img, level))


def float_to_uint(img, bits):
    """[0, 1] float -> uint8 / uint16 with rounding."""
    m = 255.0 if bits == 8 else 65535.0
    return (np.clip(np.asarray(img, np.float32), 0.0, 1.0) * m + 0.5).astype(np.uint8 if bits == 8 else np.uint16)


def linear_to_srgb(x):
    x = np.maximum(np.asarray(x, np.float32), 0.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def srgb_to_linear(x):
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def rolloff(x, knee=0.8):
    x = np.asarray(x, np.float32)
    r = 1.0 - knee
    return np.where(x > knee, knee + r * (1.0 - np.exp(-(x - knee) / r)), x)


def element_to_display(rgb_premult, alpha, knee=0.8, mode='premultiplied'):
    """Scene-linear premultiplied fire element -> display-encoded RGBA in [0, 1].

    premultiplied: colour stays premultiplied (interpret as 'premultiplied' / 'matted with black').
    straight: alpha is raised to cover emitted light (luma key) and colour is divided through, so the
              element drops onto footage with an ordinary Normal blend.
    """
    rgb = rolloff(np.asarray(rgb_premult, np.float32), knee)
    a = np.clip(np.asarray(alpha, np.float32), 0.0, 1.0)
    if mode == 'straight':
        lum = np.clip(rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32), 0.0, 1.0)
        a2 = np.maximum(a, np.clip(lum * 1.2, 0.0, 1.0))
        rgb = np.where(a2[..., None] > 1e-5, rgb / np.maximum(a2[..., None], 1e-5), 0.0)
        a = a2
    return np.concatenate([linear_to_srgb(np.clip(rgb, 0.0, 1.0)), a[..., None]], -1)
