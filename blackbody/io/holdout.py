"""Holdouts from the footage: a matte of the objects in front of the fire and/or a depth pass.

Both are image sequences (or videos) numbered like the footage, set in Composite › Holdouts from
footage. The matte hides the fire, smoke and embers wherever it is set; the depth pass hides them
behind the footage's surfaces only, so fire can burn in front of an object and behind it, and the fire
light lands on the footage's own surfaces (their shape comes from the depth pass).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger('blackbody.holdout')

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _channel(img, which):
    a = np.asarray(img)
    x = a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else a.astype(np.float32)
    if which == 'alpha':
        return x[..., 3] if x.shape[-1] > 3 else np.ones(x.shape[:2], np.float32)
    if which == 'luma':
        return x[..., :3] @ LUMA
    return x[..., {'red': 0, 'green': 1, 'blue': 2}.get(which, 0)]


class FootageHoldout:
    """Reads a scene's holdout matte and depth pass frame by frame (None where not set)."""

    def __init__(self, scene):
        from .footage import Footage
        c = scene.data['composite']
        self.scene = scene
        self.channel = c.get('matte_channel', 'alpha')
        self.invert = bool(c.get('matte_invert', False))
        self.matte = self.depth = None
        self.errors = []
        for attr, key in (('matte', 'holdout_matte'), ('depth', 'holdout_depth')):
            p = c.get(key, '')
            if not p:
                continue
            path = Path(scene.mesh_path(p))   # resolves project-relative paths the same way
            if '#' in path.name:
                from ..engine.mesh import sequence_files
                files = sequence_files(str(path))
                if files:
                    path = Path(files[min(files)])
            try:
                setattr(self, attr, Footage(path))
            except Exception as ex:
                self.errors.append(f'{key}: {ex}')
                log.warning('Holdout %s unavailable: %s', key, ex)

    @property
    def active(self):
        return self.matte is not None or self.depth is not None

    def index(self, frame):
        off = int((self.scene.footage or {}).get('offset', 0))
        return frame - self.scene.start + off

    def read(self, frame):
        """(matte (h, w) 0..1 or None, depth (h, w) in the pass's own units or None) for a frame."""
        i = self.index(frame)
        m = d = None
        if self.matte is not None:
            m = np.clip(_channel(self.matte.read(i), self.channel), 0.0, 1.0)
            if self.invert:
                m = 1.0 - m
        if self.depth is not None:
            d = _channel(self.depth.read(i), 'red')
        return m, d

    def close(self):
        for f in (self.matte, self.depth):
            if f is not None:
                f.close()
