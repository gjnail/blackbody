"""The footage as the set's environment (Lighting › Environment from the footage, with Lume): an HDRI of what the camera
sees, for Lume to light the CG by and for it to reflect, so a CG object takes the colours and the light of the shot it
is put in. Each direction the camera saw is the plate there (scene-linear, at its exposure); past the frame's edges the
footage's own light is carried on (pulled in from the nearest of what was seen, blurred wider the further out), so the
sky behind the camera is the sky at the top of the frame and the ground behind it the ground at the bottom.

In the HDRI convention Blackbody's take (io/hdri.py; lume.py env_table): row 0 straight up, the middle column -z, in the
world's frame.
"""
from __future__ import annotations

import math

import numpy as np

WIDTH = 256      # the HDRI's width (its height half that): the footage's light, which Lume needs smooth more than sharp


def linear(plate, kind):
    """The footage (h, w, >=3: uint8 display-encoded, or float) in the working space, as engine.footage_ambient has it."""
    a = np.asarray(plate)[..., :3].astype(np.float32)
    x = a / 255.0 if np.asarray(plate).dtype == np.uint8 else a
    if kind == 'srgb':
        return np.where(x <= 0.04045, x / 12.92, ((np.maximum(x, 0.0) + 0.055) / 1.055) ** 2.4)
    if kind == 'rec709':
        return np.maximum(x, 0.0) ** 2.4
    if kind == 'acescg':
        m = np.array([[1.70505, -0.62179, -0.08326], [-0.13026, 1.14080, -0.01055], [-0.02400, -0.12897, 1.15297]])
        return x @ m.T
    return x


def _blur(a, sigma):
    """A separable Gaussian blur of a lat-long image (h x w x c): round in longitude, held at the poles."""
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = sum(np.roll(a, i, 1) * k[i + r] for i in range(-r, r + 1))
    pad = np.pad(out, ((r, r), (0, 0), (0, 0)), mode='edge')
    return sum(pad[r + i:r + i + a.shape[0]] * k[i + r] for i in range(-r, r + 1))


def _fill(values, mask, sigma):
    """values where mask; elsewhere the footage's light carried on: near the frame, its own pixels blurred outward;
    further, the footage's mean at that elevation (each row of the panorama), above and below the frame its top and
    bottom rows'."""
    m = mask.astype(np.float32)[..., None]
    rows = (values * m).sum(1) / np.maximum(m.sum(1), 1e-6)
    has = m[..., 0].sum(1) > 0
    if not has.any():
        return values
    idx = np.where(has)[0]
    near = idx[np.clip(np.searchsorted(idx, np.arange(len(has))), 0, len(idx) - 1)]
    near = np.where(np.abs(idx[np.clip(np.searchsorted(idx, np.arange(len(has))) - 1, 0, len(idx) - 1)] - np.arange(len(has)))
                    < np.abs(near - np.arange(len(has))), idx[np.clip(np.searchsorted(idx, np.arange(len(has))) - 1, 0, len(idx) - 1)], near)
    row_fill = rows[near][:, None, :] * np.ones_like(values)
    bv, bm = _blur(values * m, sigma), _blur(m, sigma)
    local = bv / np.maximum(bm, 1e-6)
    c = np.clip(bm * 2.0, 0.0, 1.0)
    out = c * local + (1.0 - c) * row_fill
    return np.where(mask[..., None], values, out)


def image(plate, kind, view_proj, exposure=1.0, width=WIDTH, plate_fit=(1.0, 1.0)):
    """The HDRI (h x w x 3) of the footage seen through the camera view_proj (world -> clip), at `exposure` (times the
    plate's scene-linear values). plate_fit: the plate's scale in the picture (stage.wgsl U.fit: plate uv = (picture uv -
    0.5) x fit + 0.5)."""
    lin = linear(plate, kind)
    ph, pw = lin.shape[:2]
    step = max(1, int(round(pw / 512)))
    lin = lin[::step, ::step]
    ph, pw = lin.shape[:2]
    w, h = int(width), int(width) // 2
    v = (np.arange(h) + 0.5) / h
    u = (np.arange(w) + 0.5) / w
    T, P = np.meshgrid(v * math.pi, (u - 0.5) * 2.0 * math.pi, indexing='ij')
    d = np.stack([np.sin(T) * np.sin(P), np.cos(T), -np.sin(T) * np.cos(P)], -1)   # (lume.py env convention)
    clip = np.einsum('ij,hwj->hwi', np.asarray(view_proj, float), np.concatenate([d, np.zeros(d.shape[:2] + (1,))], -1))
    cw = clip[..., 3]
    front = cw > 1e-6
    nx = np.where(front, clip[..., 0] / np.where(front, cw, 1.0), 9.0)
    ny = np.where(front, clip[..., 1] / np.where(front, cw, 1.0), 9.0)
    pu = ((nx + 1.0) * 0.5 - 0.5) * float(plate_fit[0]) + 0.5
    pv = ((1.0 - ny) * 0.5 - 0.5) * float(plate_fit[1]) + 0.5
    seen = front & (pu >= 0.0) & (pu <= 1.0) & (pv >= 0.0) & (pv <= 1.0)
    px = np.clip(pu * pw - 0.5, 0, pw - 1)
    py = np.clip(pv * ph - 0.5, 0, ph - 1)
    x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
    x1, y1 = np.minimum(x0 + 1, pw - 1), np.minimum(y0 + 1, ph - 1)
    fx, fy = (px - x0)[..., None], (py - y0)[..., None]
    val = (lin[y0, x0] * (1 - fx) * (1 - fy) + lin[y0, x1] * fx * (1 - fy) + lin[y1, x0] * (1 - fx) * fy + lin[y1, x1] * fx * fy)
    out = _fill(np.where(seen[..., None], val, 0.0), seen, sigma=w / 64.0)
    return (np.maximum(out, 0.0) * float(exposure)).astype(np.float32)


_last = None     # (the last (key, HDRI): a render asks for it for its key light, then for its set)


def applies(scene, plate, view_proj):
    """Whether the footage is a scene's environment: Lighting › Environment from the footage, with no HDRI file and
    footage to take it from."""
    lt = scene.data['lighting']
    return bool(lt.get('env_from_footage')) and not lt.get('environment') and plate is not None and view_proj is not None


def of_scene(scene, frame, plate, view_proj, plate_fit=(1.0, 1.0)):
    """The footage as a scene's environment at a frame (where it applies), or None: (key, HDRI), at the plate's exposure
    times Environment strength (so drawn at strength 1, unrotated: it is in the world's frame already). Fire scenes
    (Engine._environment) and liquid ones (LiquidEngine._footage_env) take it alike."""
    global _last
    if not applies(scene, plate, view_proj):
        return None
    lt, comp = scene.data['lighting'], scene.data['composite']
    gain = 2.0 ** float(comp.get('plate_exposure', 0.0)) * float(lt.get('env_strength', 1.0))
    kind = comp.get('plate_transform', 'srgb')
    p = np.asarray(plate)
    key = ('footage-env', frame, p.shape, float(p[::64, ::64].astype(np.float32).sum()),
           bytes(np.asarray(view_proj, np.float32).tobytes()), gain, kind, tuple(float(x) for x in plate_fit))
    got = _last     # (read once: the viewer and a render job may both be asking, each for its own)
    if got is None or got[0] != key:
        got = _last = key, image(plate, kind, view_proj, exposure=gain, plate_fit=plate_fit)
    return got
