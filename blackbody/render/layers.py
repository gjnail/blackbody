"""Layers: several simulations in one shot, each in its own engine, composited back to front.

Each layer is drawn over the composite of the ones behind it as if that were its footage, so its heat
haze, glow and light land on what is behind it, effects included. Every layer has its own box, scale and
settings, and its own cache. (An effect that has to touch another, a hose on a fire, belongs in one
simulation: layers only overlap in the picture.)
"""
from __future__ import annotations

import copy

import numpy as np


class LayerEngines:
    """The engines of a shot's layers: the base layer's engine is given, the others are made as layers
    appear (on the same GPU) and dropped when they go."""

    def __init__(self, base, cache_bytes=1536 << 20):
        self.base = base
        self.extra = {}
        self.cache_bytes = cache_bytes

    def get(self, uid):
        if uid == 'base':
            return self.base
        e = self.extra.get(uid)
        if e is None:
            from ..engine.engine import Engine
            e = self.extra[uid] = Engine(self.base.gpu, cache_bytes=self.cache_bytes)
        return e

    def keep(self, uids):
        for u in list(self.extra):
            if u not in uids:
                del self.extra[u]


def over_linear(scene):
    """A light copy of a layer, to draw over a scene-linear composite (the layers behind it) instead of the
    footage: the plate is read as linear, as it is, and its noise is not matched again."""
    c = copy.copy(scene)
    c.data = dict(scene.data)
    comp = dict(scene.data['composite'])
    comp.update(plate_transform='linear', ocio_plate='', plate_exposure=0.0, grain_match=False, grain=0.0)
    c.data['composite'] = comp
    return c


def simulate(engines, order, frame, final=False, soft=False, progress=None, cancelled=None, cache=False):
    """Bring every layer's simulation to `frame`. False if it was cancelled."""
    for uid, sc in order:
        eng = engines.get(uid)
        eng.prepare(sc, final=final, soft=soft)
        if frame in eng.cache or eng.sim_frame == frame:
            continue
        kw = {'cache': cache} if cache else {}
        if not eng.simulate_to(sc, frame, progress=progress, cancelled=cancelled, **kw):
            return False
    return True


def render(engines, order, frame, size, mode='composite', final=False, samples=1, motion_blur=False, plate=None,
           plate_fit=(1.0, 1.0), holdout=None):
    """Render the layers back to front, each over the last one's composite. Returns the front layer's engine,
    whose display and linear images are the shot's. In views other than the composite, the layers' elements
    are not merged: only the last layer in `order` is drawn."""
    prev = None
    eng = None
    if mode != 'composite':
        order = order[-1:]
    for k, (uid, sc) in enumerate(order):
        eng = engines.get(uid)
        hold = holdout if sc.kind != 'liquid' else None
        if k == 0:
            eng.render(sc, frame, size, mode=mode, final=final, samples=samples, motion_blur=motion_blur, plate=plate,
                       plate_fit=plate_fit, holdout=hold)
        else:
            eng.render(over_linear(sc), frame, size, mode='composite', final=final, samples=samples, motion_blur=motion_blur,
                       plate=prev, plate_fit=(1.0, 1.0), holdout=hold)
        if k < len(order) - 1:
            prev = np.asarray(eng.linear_comp(), np.float32)
            prev[..., 3] = 1.0
    return eng


def merge_elements(elements):
    """Premultiplied RGBA elements (back to front) merged into one: each in front of the ones before it."""
    out = None
    for e in elements:
        e = np.asarray(e, np.float32)
        if out is None:
            out = e.copy()
            continue
        if e.shape != out.shape:
            continue
        out = e + out * (1.0 - e[..., 3:4])
    return out
