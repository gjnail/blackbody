"""View transforms in numpy, matching the composite shader, for high-bit-depth exports."""
from __future__ import annotations

import numpy as np

from .images import camera_rolloff, linear_to_srgb

_AGX = np.array([[0.842479062253094, 0.0423282422610123, 0.0423756549057051],
                 [0.0784335999999992, 0.878468636469772, 0.0784336],
                 [0.0792237451477643, 0.0791661274605434, 0.879142973793104]], np.float32)
_AGX_INV = np.array([[1.19687900512017, -0.0528968517574562, -0.0529716355144438],
                     [-0.0980208811401368, 1.15190312990417, -0.0980434501171241],
                     [-0.0990297440797205, -0.0989611768448433, 1.15107367264116]], np.float32)


def agx(rgb):
    # WGSL/GLSL mat3 constructors are column-major, so v' = M v means rows of _AGX are columns there
    v = np.maximum(rgb, 1e-10) @ _AGX
    mn, mx = -12.47393, 4.026069
    v = (np.clip(np.log2(v), mn, mx) - mn) / (mx - mn)
    x2 = v * v
    x4 = x2 * x2
    v = 15.5 * x4 * x2 - 40.14 * x4 * v + 31.96 * x4 - 6.868 * x2 * v + 0.4298 * x2 + 0.1191 * v - 0.00232
    v = v @ _AGX_INV
    return np.power(np.maximum(v, 0.0), 2.2)


def aces_fit(x):
    x = x * 0.8
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


def view_transform(rgb, view='standard', knee=0.8, white=0.0):
    """Scene-linear RGB -> display-encoded sRGB in [0, 1]. white: the Standard view's highlights to white."""
    rgb = np.asarray(rgb, np.float32)
    if view == 'agx':
        out = agx(rgb)
    elif view == 'aces':
        out = aces_fit(rgb)
    elif view == 'raw':
        out = np.clip(rgb, 0.0, 1.0)
    else:
        out = camera_rolloff(rgb, knee, white)
    return linear_to_srgb(np.clip(out, 0.0, 1.0))
