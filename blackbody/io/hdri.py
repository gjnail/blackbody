"""Environment maps (latitude-longitude HDR panoramas) for liquid reflections.

Reads Radiance .hdr (RGBE, run-length encoded or flat), OpenEXR, and ordinary images (sRGB).
The panorama's centre column looks down -z, u runs around the vertical axis and v from straight up
(top row) to straight down.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

MAX_WIDTH = 2048


def _read_hdr(path):
    data = Path(path).read_bytes()
    pos = 0

    def line():
        nonlocal pos
        end = data.index(b'\n', pos)
        s = data[pos:end].decode('ascii', 'replace')
        pos = end + 1
        return s

    head = line()
    if not head.startswith('#?'):
        raise ValueError('not a Radiance HDR file')
    while line().strip():
        pass
    dims = line().split()
    if len(dims) != 4:
        raise ValueError('unsupported HDR resolution line')
    flip_y = dims[0] == '+Y'
    h, w = int(dims[1]), int(dims[3])
    out = np.zeros((h, w, 4), np.uint8)
    buf = np.frombuffer(data, np.uint8)
    for y in range(h):
        if w >= 8 and w < 32768 and buf[pos] == 2 and buf[pos + 1] == 2 and (buf[pos + 2] & 0x80) == 0:
            pos += 4
            for c in range(4):
                x = 0
                while x < w:
                    n = int(buf[pos])
                    pos += 1
                    if n > 128:
                        n -= 128
                        out[y, x:x + n, c] = buf[pos]
                        pos += 1
                    else:
                        out[y, x:x + n, c] = buf[pos:pos + n]
                        pos += n
                    x += n
        else:
            out[y] = buf[pos:pos + 4 * w].reshape(w, 4)
            pos += 4 * w
    if flip_y:
        out = out[::-1]
    e = out[..., 3].astype(np.int32)
    scale = np.where(e > 0, np.ldexp(1.0, e - 136), 0.0).astype(np.float32)
    return out[..., :3].astype(np.float32) * scale[..., None]


def _read_exr(path):
    import OpenEXR
    with OpenEXR.File(str(path)) as f:
        ch = f.channels()
        if 'RGB' in ch:
            rgb = np.asarray(ch['RGB'].pixels, np.float32)
        elif 'RGBA' in ch:
            rgb = np.asarray(ch['RGBA'].pixels, np.float32)[..., :3]
        else:
            rgb = np.stack([np.asarray(ch[k].pixels, np.float32) for k in ('R', 'G', 'B')], -1)
    return rgb


def _read_ldr(path):
    from PIL import Image
    a = np.asarray(Image.open(path).convert('RGB'), np.float32) / 255.0
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4).astype(np.float32)


def load_hdri(path, max_width=MAX_WIDTH):
    """(h, w, 3) float32 scene-linear panorama, halved until it is at most max_width wide."""
    ext = Path(path).suffix.lower()
    if ext in ('.hdr', '.pic', '.rgbe'):
        img = _read_hdr(path)
    elif ext == '.exr':
        img = _read_exr(path)
    else:
        img = _read_ldr(path)
    img = np.nan_to_num(np.maximum(img, 0.0), posinf=6.0e4)
    while img.shape[1] > max_width and img.shape[0] % 2 == 0 and img.shape[1] % 2 == 0:
        img = 0.25 * (img[0::2, 0::2] + img[1::2, 0::2] + img[0::2, 1::2] + img[1::2, 1::2])
    return np.ascontiguousarray(img, np.float32)


def direction_of(u, v):
    """World direction of panorama coordinates (u, v in 0..1)."""
    phi = (u - 0.5) * 2.0 * math.pi
    theta = v * math.pi
    return np.array([math.sin(theta) * math.sin(phi), math.cos(theta), -math.sin(theta) * math.cos(phi)])


def sky_average(img):
    """Average light arriving on an upward-facing surface, as an ambient colour (cosine-weighted
    over the upper hemisphere)."""
    h, w = img.shape[:2]
    v = (np.arange(h) + 0.5) / h
    theta = v * math.pi
    wgt = np.maximum(np.cos(theta), 0.0) * np.sin(theta)
    s = (img.mean(axis=1) * wgt[:, None]).sum(axis=0) / max(wgt.sum(), 1e-9)
    return tuple(float(x) for x in s)


def brightest(img):
    """(azimuth, elevation in degrees, colour) of the brightest spot (the sun), after a blur so a
    single hot pixel does not win. Azimuth follows the key light's convention."""
    h, w = img.shape[:2]
    lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    k = max(1, w // 256)
    small = lum[: h - h % k, : w - w % k].reshape(h // k, k, w // k, k).mean(axis=(1, 3))
    y, x = np.unravel_index(np.argmax(small), small.shape)
    u, v = (x + 0.5) / small.shape[1], (y + 0.5) / small.shape[0]
    d = direction_of(u, v)
    az = math.degrees(math.atan2(d[0], d[2]))
    el = math.degrees(math.asin(max(-1.0, min(1.0, d[1]))))
    y0, x0 = int(v * h), int(u * w)
    patch = img[max(0, y0 - 2 * k): y0 + 2 * k + 1, max(0, x0 - 2 * k): x0 + 2 * k + 1].reshape(-1, 3)
    c = patch.mean(axis=0)
    c = c / max(float(c.max()), 1e-9)
    return az, el, tuple(float(x) for x in c)
