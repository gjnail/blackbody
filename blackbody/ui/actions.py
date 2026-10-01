"""What you can do to a thing in the scene: set it on fire, make it float, soak it, turn a source into
another kind, attach it to something else, send it along a path... One list, used by the viewer's and the
object list's right-click menus and by the buttons at the top of an object's settings."""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QMenu, QMessageBox

from ..scene import components as C
from . import icons, theme

MATERIALS = [('cotton', 'Cotton'), ('linen', 'Linen'), ('silk', 'Silk'), ('chiffon', 'Chiffon'), ('wool', 'Wool'), ('denim', 'Denim'),
             ('canvas', 'Canvas'), ('velvet', 'Velvet'), ('polyester', 'Polyester'), ('nylon', 'Nylon')]
PINS = [('top', 'By its top edge'), ('side', 'By one side'), ('top_corners', 'By its top corners'), ('corners', 'By four corners'),
        ('none', 'Not at all (it falls)')]


def _items(sc, kind):
    return {'emitter': sc.emitters, 'collider': sc.colliders, 'light': sc.lights, 'fabric': sc.fabrics}[kind]


def _say(win, text):
    if hasattr(win, 'msg'):
        win.msg.setText(text)


def _t(doc):
    return round(doc.scene.seconds(doc.frame), 3)


# -- the actions ------------------------------------------------------------------------------------------------

def set_on_fire(win, kind, i):
    """Make it burnable (spreading fire on) and light it at its base with a flame that dies after a few seconds:
    from there the fire is the object's own."""
    doc = win.doc
    sc = doc.scene
    d = _items(sc, kind)[i]
    pos = np.asarray(sc.get((kind, i, 'position'), doc.frame), float)
    if kind == 'fabric':
        h = float(d.get('height', 1.0))
        w = float(d.get('width', 1.0))
        at = pos + (np.array([0.0, -h / 2 + 0.02, 0.0]) if d.get('orientation') != 'lying' else np.array([0.0, 0.02, 0.0]))
        r = max(0.03, 0.06 * w)
    else:
        size = np.asarray(sc.get((kind, i, 'size'), doc.frame), float)
        bottom = pos[1] - (size[1] if d.get('shape') != 'mesh' else 0.0)
        at = np.array([pos[0], max(0.02, bottom + 0.03), pos[2]])
        r = max(0.04, 0.35 * float(min(size[0], size[2] if len(size) > 2 else size[0])))
    target = C.target_kind(sc, C.BY_KEY['burner'])
    if target is None:
        raise ValueError('Nothing burns in a sky scene.')

    def fn(s):
        if target != s.kind:
            C.convert_kind(s, target)
        dd = _items(s, kind)[i]
        dd['burnable'] = True
        if kind == 'collider':
            s.data['spread']['enabled'] = True
        e = dict(name=f'Flame on {dd["name"]}', shape='sphere', position=tuple(float(x) for x in at), size=(r, r * 0.6, r),
                 fuel=14.0, temperature=0.6, start=_t(doc), stop=_t(doc) + 3.0, fade_out=0.6, noise_freq=4.0)
        if s.kind == 'both':
            e['emits'] = 'fire'
        s.add_emitter(**e)
    doc.edit(f'Set {d["name"]} on fire', fn, structure=True)
    _say(win, f'{d["name"]} catches at frame {doc.frame}: the flame at its base lasts 3 s, then the fire is its own'
              + (' (Spreading fire is on).' if kind == 'collider' else '.'))
    doc.set_playing(True)


def make_float(win, i, density):
    doc = win.doc
    sc = doc.scene
    d = sc.colliders[i]
    target = C.target_kind(sc, C.BY_KEY['crate'])

    def fn(s):
        if target and target != s.kind:
            C.convert_kind(s, target)
        c = s.colliders[i]
        c['floating'] = True
        c['density'] = float(density)
        if not C.has_liquid(s):   # nothing to float in: a pond under it
            s.data['liquid']['water_level'] = round(max(0.1, 0.3 * s.domain_size()[1]), 3)
            s.data['liquid']['settle'] = True
    doc.edit(f'{"Float" if density < 1000 else "Sink"} {d["name"]}', fn, structure=True)
    _say(win, f'{d["name"]} {"floats" if density < 1000 else "sinks"} ({density:.0f} kg/m³)'
              + ('; there was no water, so a pond fills the bottom of the box.' if not C.has_liquid(sc) else '.'))
    doc.set_playing(True)


def make_temperature(win, i, celsius):
    doc = win.doc

    def fn(s):
        s.colliders[i]['temperature'] = float(celsius)
        if s.kind in ('liquid', 'both'):
            s.data['liquid']['thermal'] = True
    doc.edit('Change temperature', fn, structure=True)
    _say(win, f'{doc.scene.colliders[i]["name"]} is {celsius:.0f} °C: liquid on it ' +
         ('boils.' if celsius >= 100 else 'freezes.' if celsius < 0 else 'warms or cools.'))


def set_value(win, kind, i, key, value, text):
    win.doc.set((kind, i, key), value, merge=False)
    _say(win, text)


def turn_into(win, i, key, label):
    doc = win.doc
    notes = []

    def fn(s):
        notes.extend(C.turn_into(s, i, key))
    try:
        doc.edit(f'Turn into {label.lower()}', fn, structure=True)
    except ValueError as ex:
        QMessageBox.information(win, 'Turn into', str(ex))
        return
    doc.sceneReplaced.emit()
    _say(win, f'{doc.scene.emitters[i]["name"]} is now {label.lower()}. ' + ' '.join(notes))
    doc.set_playing(True)


def drop_to_ground(win, kind, i):
    doc = win.doc
    sc = doc.scene
    d = _items(sc, kind)[i]
    pos = np.asarray(sc.get((kind, i, 'position'), doc.frame), float)
    if kind == 'fabric':
        y = 0.01 if d.get('orientation') == 'lying' else float(d.get('height', 1.0)) / 2 + 0.01
    elif kind == 'light':
        return
    else:
        size = np.asarray(sc.get((kind, i, 'size'), doc.frame), float)
        shape = d.get('shape')
        y = 0.0 if shape == 'mesh' else float(size[0] if (shape == 'sphere' and kind == 'collider') or shape == 'capsule' else size[1])
    new = (float(pos[0]), y, float(pos[2]))
    doc.set((kind, i, 'position'), new, merge=False)
    if kind == 'emitter' and d.get('shape') == 'capsule':
        end = np.asarray(sc.get(('emitter', i, 'end'), doc.frame), float)
        doc.set(('emitter', i, 'end'), (float(end[0]), y, float(end[2])), merge=False)
    _say(win, f'{d["name"]} stands on the ground.')


def duplicate(win, kind, i):
    doc = win.doc
    if kind == 'emitter':
        doc.duplicate_emitter(i)
        return
    if kind == 'fabric':
        doc.duplicate_fabric(i)
        return
    import copy
    sc = doc.scene
    w = max(sc.domain_size()) * 0.08

    def fn(s):
        items = _items(s, kind)
        d = copy.deepcopy(items[i])
        d['name'] = d['name'] + ' copy'
        p = d['position']
        if not hasattr(p, 'keys'):
            d['position'] = (p[0] + w, p[1], p[2])
        items.insert(i + 1, d)
    doc.edit('Duplicate', fn, structure=True)
    doc.select((kind, i + 1), force=True)


def delete(win, kind, i):
    doc = win.doc
    {'emitter': doc.remove_emitter, 'collider': doc.remove_collider, 'light': doc.remove_light,
     'fabric': doc.remove_fabric}[kind](i)


def frame_it(win, kind, i):
    doc = win.doc
    if doc.work_view is None:
        if hasattr(win, 'set_workspace'):
            win.set_workspace('build')
    wv = doc.work_view
    if wv is None:
        return
    sc = doc.scene
    pos = np.asarray(sc.get((kind, i, 'position'), doc.frame), float)
    d = _items(sc, kind)[i]
    if kind in ('emitter', 'collider'):
        r = float(np.max(np.abs(sc.get((kind, i, 'size'), doc.frame))))
    elif kind == 'fabric':
        r = max(float(d.get('width', 1.0)), float(d.get('height', 1.0))) / 2
    else:
        r = 0.3
    wv.target = tuple(float(x) for x in pos)
    wv.distance = max(0.15, r * 5.0)
    doc.move_work_view()


def aim_light(win, i):
    doc = win.doc
    sc = doc.scene
    pos = np.asarray(sc.get(('light', i, 'position'), doc.frame), float)
    tgt = np.array([0.0, sc.domain_size()[1] * 0.3, 0.0])
    d = tgt - pos
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    doc.set(('light', i, 'direction'), tuple(float(x) for x in d), merge=False)
    _say(win, 'Aimed at the middle of the scene.')


def hollow(win, i):
    doc = win.doc
    sc = doc.scene
    size = np.asarray(sc.get(('collider', i, 'size'), doc.frame), float)
    doc.set(('collider', i, 'hollow'), float(max(0.01, 0.1 * float(size.min()))), merge=False)
    _say(win, f'{sc.colliders[i]["name"]} is hollow now: a container, a room or a pipe. Cut a door or window in it with '
              'Opening size and Opening at.')


# -- the menu ----------------------------------------------------------------------------------------------------

def fill_menu(m: QMenu, win, sel, path_mode=None):
    """The actions for object sel = (kind, i) into menu m."""
    doc = win.doc
    kind, i = sel
    sc = doc.scene
    items = _items(sc, kind)
    if i >= len(items):
        return
    d = items[i]

    def act(text, fn, tip='', glyph=None, enabled=True):
        a = m.addAction(icons.glyph_icon(glyph, theme.MUTED, 16) if glyph else icons.glyph_icon('plus', 'transparent', 16), text, fn)
        if tip:
            a.setToolTip(tip)
        a.setEnabled(enabled)
        return a

    def guarded(fn):
        def run():
            try:
                fn()
            except ValueError as ex:
                QMessageBox.information(win, 'Not here', str(ex))
        return run

    head = m.addAction(d.get('name', kind))
    head.setEnabled(False)
    if kind == 'emitter':
        sub = m.addMenu(icons.glyph_icon('spark' if False else 'sparks', theme.ACCENT, 16), 'Turn into')
        for key, label in C.TURN_INTO:
            ok = C.target_kind(sc, C.BY_KEY[key]) is not None
            a = sub.addAction(icons.glyph_icon(C.BY_KEY[key].glyph, theme.MUTED, 16), label, lambda k=key, l=label: turn_into(win, i, k, l))
            a.setEnabled(ok)
        act(f'Start at this frame ({doc.frame})', lambda: set_value(win, 'emitter', i, 'start', _t(doc), f'It starts at frame {doc.frame}.'))
        act(f'Stop at this frame ({doc.frame})', lambda: set_value(win, 'emitter', i, 'stop', _t(doc), f'It stops at frame {doc.frame}.'))
        act('A short burst from here', lambda: (doc.set(('emitter', i, 'start'), _t(doc), merge=False),
                                                set_value(win, 'emitter', i, 'stop', _t(doc) + 0.2, 'A fifth of a second, from this frame.')))
    if kind == 'collider':
        act('Set it on fire', guarded(lambda: set_on_fire(win, 'collider', i)), 'It catches here and the fire spreads over it', 'flame')
        act('Make it float', lambda: make_float(win, i, 500.0), 'It floats on the water (a pond is added if there is none)', 'waves')
        act('Make it sink', lambda: make_float(win, i, 2500.0), 'It falls in the water and sinks', 'drop')
        act('Make it hot (300 °C)', lambda: make_temperature(win, i, 300.0), 'Water on it boils', 'flame')
        act('Make it freezing (-30 °C)', lambda: make_temperature(win, i, -30.0), 'Water on it freezes', 'snow')
        act('Make it hollow', lambda: hollow(win, i), 'Walls only: a tank, a room, a pipe', 'house')
        act('Hide it in the render' if d.get('holdout', True) else 'Show it in the render',
            lambda: set_value(win, 'collider', i, 'holdout', not d.get('holdout', True),
                              'It is in the shot now: it hides the effect behind it.' if not d.get('holdout', True)
                              else 'It is a helper now: it shapes the effect but is not drawn.'))
    if kind == 'fabric':
        act('Set it on fire', guarded(lambda: set_on_fire(win, 'fabric', i)), 'A flame at its lower edge catches it', 'flame')
        if float(d.get('wetness', 0.0)) < 0.5:
            act('Soak it', lambda: set_value(win, 'fabric', i, 'wetness', 1.0, 'Soaked: it drips, steams in the heat and will not burn '
                                                                             'until it dries.'), 'Wet through', 'drop')
        else:
            act('Dry it', lambda: set_value(win, 'fabric', i, 'wetness', 0.0, 'Dry.'), '', 'flame')
        act(f'Let go at this frame ({doc.frame})', lambda: set_value(win, 'fabric', i, 'release', _t(doc), f'It falls from frame {doc.frame}.'),
            'What holds it lets go and it falls', 'fabric')
        sub = m.addMenu('Held')
        for k, label in PINS:
            a = sub.addAction(label, lambda k=k, l=label: set_value(win, 'fabric', i, 'pins', k, f'Held {l.lower()}.'))
            a.setCheckable(True)
            a.setChecked(d.get('pins') == k)
        sub = m.addMenu('Material')
        for k, label in MATERIALS:
            a = sub.addAction(label, lambda k=k, l=label: set_value(win, 'fabric', i, 'material', k, f'{l}.'))
            a.setCheckable(True)
            a.setChecked(d.get('material') == k)
    if kind == 'light':
        act('Aim it at the middle', lambda: aim_light(win, i), '', 'spot')
    m.addSeparator()
    others = [(k2, j, x['name']) for k2 in ('emitter', 'collider', 'fabric', 'light') for j, x in enumerate(_items(sc, k2))
              if not (k2 == kind and j == i)]
    link = sc.link_of(kind, d['name'])
    if link is not None:
        act(f'Detach from {link["parent"][1]}', lambda: (doc.detach(kind, i), _say(win, f'{d["name"]} moves on its own now.')),
            'It stops following', 'copy')
    sub = m.addMenu(icons.glyph_icon('copy', theme.MUTED, 16), 'Attach to')
    sub.setToolTip('It goes wherever the other one goes: a torch in a moving hand, a flag on a moving pole, fire on a car')
    if not others:
        a = sub.addAction('Nothing else in the scene yet')
        a.setEnabled(False)
    for k2, j, name in others:
        sub.addAction(icons.glyph_icon(k2, theme.OBJECT_COLOURS[k2], 16), name,
                      lambda k2=k2, j=j, name=name: (doc.attach((kind, i), (k2, j)),
                                                     _say(win, f'{d["name"]} is attached to {name}: it goes where {name} goes.')))
    if path_mode is not None:
        act('Move along a path…', lambda: path_mode(kind, i), 'Click points on the ground to draw where it goes, over the next seconds',
            'line')
    act('Drop to the ground', lambda: drop_to_ground(win, kind, i), '', 'fit', enabled=kind != 'light')
    act('Look at it', lambda: frame_it(win, kind, i), 'The Build camera turns to it', 'camera')
    m.addSeparator()
    act('Duplicate', lambda: duplicate(win, kind, i), '', 'copy')
    act('Delete', lambda: delete(win, kind, i), '', 'trash')


def quick_actions(win, sel):
    """The few most useful actions as (label, glyph, callable) for buttons on the object's page."""
    kind, i = sel
    sc = win.doc.scene
    d = _items(sc, kind)[i]
    out = []
    if kind == 'collider':
        out += [('Set on fire', 'flame', lambda: _guard(win, lambda: set_on_fire(win, 'collider', i))),
                ('Float', 'waves', lambda: make_float(win, i, 500.0)), ('Hot', 'flame', lambda: make_temperature(win, i, 300.0))]
    elif kind == 'fabric':
        out += [('Set on fire', 'flame', lambda: _guard(win, lambda: set_on_fire(win, 'fabric', i))),
                ('Soak' if float(d.get('wetness', 0.0)) < 0.5 else 'Dry', 'drop',
                 lambda: set_value(win, 'fabric', i, 'wetness', 1.0 if float(d.get('wetness', 0.0)) < 0.5 else 0.0, 'Done.')),
                ('Let go now', 'fabric', lambda: set_value(win, 'fabric', i, 'release', _t(win.doc), f'It falls from frame {win.doc.frame}.'))]
    elif kind == 'emitter':
        out += [('Start here', 'play', lambda: set_value(win, 'emitter', i, 'start', _t(win.doc), f'It starts at frame {win.doc.frame}.')),
                ('Stop here', 'stop', lambda: set_value(win, 'emitter', i, 'stop', _t(win.doc), f'It stops at frame {win.doc.frame}.'))]
    out.append(('Ground', 'fit', lambda: drop_to_ground(win, kind, i)))
    return out


def _guard(win, fn):
    try:
        fn()
    except ValueError as ex:
        QMessageBox.information(win, 'Not here', str(ex))


def path_keys(scene, kind, i, pts, start_frame, seconds):
    """Position keys that take object i along the ground points pts (local x, y, z) from start_frame over
    `seconds`, spaced by distance so it moves at a steady speed."""
    from ..scene.anim import Curve
    p = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = max(float(s[-1]), 1e-9)
    fps = scene.fps
    keys = [[float(start_frame + seconds * fps * si / total), tuple(float(x) for x in q), 'smooth'] for si, q in zip(s, p)]
    return Curve(keys)

