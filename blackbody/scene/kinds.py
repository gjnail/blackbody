"""The kinds of object a scene holds (sources, objects, lights, fabric, matter) and the list each lives in: one table,
so the interface handles a new kind of object once it is added here."""
from __future__ import annotations

KINDS = ('emitter', 'collider', 'light', 'fabric', 'matter')
LISTS = {'emitter': 'emitters', 'collider': 'colliders', 'light': 'lights', 'fabric': 'fabrics', 'matter': 'matter'}
NOUNS = {'emitter': 'Source', 'collider': 'Object', 'light': 'Light', 'fabric': 'Fabric', 'matter': 'Matter'}
TITLES = {'emitter': 'Sources', 'collider': 'Objects (colliders)', 'light': 'Lights', 'fabric': 'Fabric',
          'matter': 'Sand, snow & mud'}


def items(scene, kind):
    """The scene's own list of objects of a kind (changing it changes the scene)."""
    got = getattr(scene, LISTS[kind], None)
    if got is None:   # a scene from before that kind existed
        got = []
        setattr(scene, LISTS[kind], got)
    return got


def lists(scene):
    """{kind: the scene's list of them}, in KINDS order."""
    return {k: items(scene, k) for k in KINDS}


def count(scene):
    return sum(len(v) for v in lists(scene).values())
