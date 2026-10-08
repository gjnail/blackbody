"""What the engine leaves out of a scene, worked out from the scene alone (no GPU): its fixed caps on objects (16), on
sources that emit at once (16 in each list, the water level's and lightning's fires among them), on lights in the set
(8, a lightning bolt taking 4 while it flashes), on fabrics (16), on grass (8 patches, 400,000 blades) and on matter's
material slots (15).

`notices(scene)` says what is cut off, in words. The engine adds them to what its parts say as they run
(Engine.notices), the viewer shows them and the command line prints them. `room(scene, items)` and `over(...)` are
Repeat's check before it makes copies past a cap."""
from __future__ import annotations

import math

from ..engine.cloth import MAX_FABRICS
from ..engine.renderer import MAX_LAMPS
from ..engine.solver import MAX_COLLIDERS, MAX_EMITTERS
from ..engine.strands import MAX_BLADES, MAX_PATCHES

LEVEL_SLOTS = 4            # the water level's top-up along the open sides (engine/liquid.py _level_sources), one a side
LIGHTNING_FIRE = 0.25      # s a strike that sets fire burns as a source (Scene.lightning_fires)


def _who(names):
    """'A', 'A and B', 'A, B and C'; past four, the first three and how many more."""
    names = list(names)
    if len(names) > 4:
        names = names[:3] + [f'{len(names) - 3} more']
    return names[0] if len(names) == 1 else ', '.join(names[:-1]) + ' and ' + names[-1]


def _is(n):
    return 'is' if n == 1 else 'are'


# -- sources ---------------------------------------------------------------------------------------------------------

def _span(scene):
    """The simulation's time (shot seconds): from the start of the pre-roll to the last frame."""
    return -float(scene.data['domain']['preroll']), scene.seconds(scene.end)


def _when(e, t0, liquid):
    """When a source emits (shot seconds, from and to); a source that fills the box once: the moment it does."""
    start, stop = float(e['start']), float(e['stop'])
    once = e.get('liquid_mode') == 'fill' if liquid else (e['shape'] == 'volume' and e.get('volume_mode', 'fill') == 'fill')
    if once:
        at = t0 if start <= -99 else max(start, t0)
        return at, at
    a = -math.inf if start <= -99 else start
    if stop < 0:
        return a, math.inf
    # (a fire source fades out after its stop; a liquid one stops at it)
    return a, (stop - 1e-6 if liquid else stop + max(float(e.get('fade_out', 0.0)), 1e-3))


def source_lists(scene):
    """The scene's lists of sources the engine caps at MAX_EMITTERS each: {list: (what they are, [(name, from, to)] in
    the engine's order, the slots the engine takes first at time t)}."""
    t0, _t1 = _span(scene)
    kind = scene.kind
    out = {}
    if kind in ('fire', 'both'):
        rows = [(e['name'], *_when(e, t0, False)) for e in scene.emitters
                if e['enabled'] and not (kind == 'both' and e.get('emits') in ('liquid', 'lava'))]
        for d in scene.lights:     # (flames where lightning that sets fire strikes: after the scene's own)
            if d['enabled'] and d['kind'] == 'lightning' and d.get('ignites'):
                t = float(d['strike_at'])
                rows.append((f'the fire {d["name"]} sets', t, t + LIGHTNING_FIRE))
        out['fire'] = ('sources', rows, lambda t: 0)
    if kind in ('liquid', 'both'):
        rows = [(e['name'], *_when(e, t0, True)) for e in scene.emitters
                if e['enabled'] and (kind == 'liquid' or e.get('emits') == 'liquid')]
        level = bool(scene.data['domain']['open_sides']) and scene.liquid_level(scene.start) > 0.0
        # (the water level's sources go first: the box filled to it at the first step, then topped up along the sides)
        out['liquid'] = ('liquid sources', rows, (lambda t: LEVEL_SLOTS + (1 if t <= t0 else 0)) if level else (lambda t: 0))
    if kind == 'both':
        rows = [(e['name'], *_when(e, t0, True)) for e in scene.emitters if e['enabled'] and e.get('emits') == 'lava']
        out['lava'] = ('lava sources', rows, lambda t: 0)
    return out


def _busiest(rows, cap, taken, t0, t1):
    """(the most slots in use at once, the names left out at some moment): rows [(name, from, to)] in the engine's
    order, `taken(t)` slots the engine fills first."""
    times = sorted({min(max(a, t0), t1) for _, a, b in rows if b >= t0 and a <= t1} | {t0})
    most, cut = 0, {}
    for t in times:
        on = [n for n, a, b in rows if a <= t <= b]
        first = taken(t)
        most = max(most, first + len(on))
        for n in on[max(cap - first, 0):]:
            cut[n] = True
    return most, list(cut)


# -- lights ----------------------------------------------------------------------------------------------------------

def _flash(scene, d):
    """When a lightning light lights the set (shot seconds, from and to): its strokes and their tails, round a frame."""
    from ..engine import lightning as LG
    ss = LG.strokes(float(d['strike_at']), int(d['strokes']), int(d['bolt_seed']))
    half = 0.5 / scene.fps
    return ss[0][0] - half, ss[-1][0] + LG.DECAY * (3.0 + math.log(1e4)) + half


def _lamp_cuts(scene):
    """(the lights left out always, those left out only while lightning flashes, the most lamp slots in use at once)."""
    lights = [d for d in scene.lights if d['enabled']]
    each = getattr(scene, 'LIGHTNING_LAMPS', 4)
    t1 = scene.seconds(scene.end)
    flashes = {id(d): _flash(scene, d) for d in lights if d['kind'] == 'lightning'}

    def cut(on):
        used, out = 0, []
        for d in lights:
            n = (each if id(d) in on else 0) if d['kind'] == 'lightning' else 1
            if n and used + n > MAX_LAMPS:
                out.append(d['name'])
            used += n
        return used, out

    most, always = cut(set())
    flashing = {}
    for a, b in flashes.values():
        t = max(a, 0.0)
        if b < 0.0 or t > t1:
            continue
        used, names = cut({k for k, (a2, b2) in flashes.items() if a2 <= t <= b2})
        most = max(most, used)
        flashing.update((n, True) for n in names if n not in always)
    return always, list(flashing), most


# -- the check -------------------------------------------------------------------------------------------------------

def notices(scene):
    """What the engine's caps leave out of the scene, in words (empty when nothing is)."""
    out = []
    cols = [c['name'] for c in scene.colliders if c['enabled']]
    if len(cols) > MAX_COLLIDERS:
        extra = cols[MAX_COLLIDERS:]
        out.append(f'Only the first {MAX_COLLIDERS} objects take part (simulated and drawn): {_who(extra)} '
                   f'{_is(len(extra))} left out.')
    if scene.kind == 'cloud':
        return out
    t0, t1 = _span(scene)
    for key, (what, rows, taken) in source_lists(scene).items():
        _most, cut = _busiest(rows, MAX_EMITTERS, taken, t0, t1)
        if not cut:
            continue
        level = taken(t1) > 0
        out.append(f'At most {MAX_EMITTERS} {what} {"pour" if key != "fire" else "emit"} at once'
                   + (f', and the water level keeps {LEVEL_SLOTS} of them for itself' if level else '')
                   + f': at the busiest moment the last in the list {"are" if len(cut) > 1 else "is"} left out ({_who(cut)}).')
    always, flashing, _most = _lamp_cuts(scene)
    if always:
        out.append(f'Only the first {MAX_LAMPS} lights light the scene: {_who(always)} light{"s" if len(always) == 1 else ""} '
                   f'nothing.')
    if flashing:
        each = getattr(scene, 'LIGHTNING_LAMPS', 4)
        out.append(f'Lightning takes {each} of the {MAX_LAMPS} lights while it flashes: then {_who(flashing)} '
                   f'light{"s" if len(flashing) == 1 else ""} nothing.')
    fabrics = [d['name'] for d in scene.fabrics if d['enabled'] and not (d['shape'] == 'mesh' and not d['mesh'])]
    if len(fabrics) > MAX_FABRICS:
        extra = fabrics[MAX_FABRICS:]
        out.append(f'Only the first {MAX_FABRICS} fabrics are simulated: {_who(extra)} {_is(len(extra))} left out.')
    from ..engine import strands
    specs = scene.strand_specs()
    out += strands.notes(specs, [d['name'] for d in getattr(scene, 'strands', None) or [] if d['enabled']])
    from ..engine.matter import material_slots
    out += material_slots(scene.matter_specs(), wets=scene.kind in ('liquid', 'both'))[3]
    return out


# -- Repeat ----------------------------------------------------------------------------------------------------------

def room(scene, items):
    """What copies of `items` ([(kind, index)], with what is attached to them) take from the caps they count against:
    [(what, in use at the busiest moment, each copy's share, cap, left out or thinned)]. Disabled things copy disabled
    and take nothing."""
    from . import kinds as K
    from ..engine.strands import blades_wanted
    per = {}

    def add(key, n=1):
        per[key] = per.get(key, 0) + n

    lists = source_lists(scene) if scene.kind != 'cloud' else {}
    specs = scene.strand_specs() if scene.kind != 'cloud' else []
    grows = {id(d): s for d, s in zip([d for d in getattr(scene, 'strands', None) or [] if d['enabled']], specs)}
    for kind, i in items:
        d = K.items(scene, kind)[i]
        if not d.get('enabled', True):
            continue
        if kind == 'collider':
            add('objects')
        elif kind == 'emitter' and lists:
            emits = {'fire': 'fire', 'liquid': 'liquid'}.get(scene.kind) or (d.get('emits') or 'fire')
            if emits in lists:
                add(emits)
        elif kind == 'light' and scene.kind != 'cloud':
            add('lights', getattr(scene, 'LIGHTNING_LAMPS', 4) if d['kind'] == 'lightning' else 1)
        elif kind == 'fabric' and scene.kind != 'cloud' and not (d['shape'] == 'mesh' and not d['mesh']):
            add('fabrics')
        elif kind == 'strands' and id(d) in grows:
            add('patches')
            add('blades', blades_wanted(grows[id(d)]))
    out = []
    if 'objects' in per:
        out.append(('objects', sum(1 for c in scene.colliders if c['enabled']), per['objects'], MAX_COLLIDERS, 'left out'))
    t0, t1 = _span(scene)
    for key, (what, rows, taken) in lists.items():
        if key in per:
            out.append((f'{what} at once', _busiest(rows, MAX_EMITTERS, taken, t0, t1)[0], per[key], MAX_EMITTERS, 'left out'))
    if 'lights' in per:
        bolt = any(d['enabled'] and d['kind'] == 'lightning' for d in scene.lights)
        what = f'lights (a lightning bolt takes {getattr(scene, "LIGHTNING_LAMPS", 4)} while it flashes)' if bolt else 'lights'
        out.append((what, _lamp_cuts(scene)[2], per['lights'], MAX_LAMPS, 'left out'))
    if 'fabrics' in per:
        n = sum(1 for d in scene.fabrics if d['enabled'] and not (d['shape'] == 'mesh' and not d['mesh']))
        out.append(('fabrics', n, per['fabrics'], MAX_FABRICS, 'left out'))
    if 'patches' in per:
        out.append(('patches of grass', len(specs), per['patches'], MAX_PATCHES, 'left out'))
        out.append(('blades of grass', sum(blades_wanted(s) for s in specs[:MAX_PATCHES]), per['blades'], MAX_BLADES, 'thinned'))
    return out


def over(rows, copies):
    """What `copies` more copies would take past the caps (room's rows), in words, for Repeat (empty when they fit)."""
    out = []
    for what, used, each, cap, how in rows:
        total = used + copies * each
        if total <= cap or each <= 0:
            continue
        if how == 'thinned':
            out.append(f'That makes {total:,} {what}, past the {cap:,} there is room for: all the grass would be thinned to '
                       f'{cap / total:.0%}.')
        else:
            out.append(f'That makes {total} {what}, past the {cap} there is room for: {total - max(cap, used)}'
                       f'{" more" if used > cap else ""} would be {how}.')
    return out
