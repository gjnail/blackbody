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


def _unique(items, name):
    names = {o['name'] for o in items}
    base, n = name, 2
    while name in names:
        name = f'{base} {n}'
        n += 1
    return name


def text_spec(d, scene=None):
    """The words (or picture), font and size of an object made by a Text or logo block, or None. With the scene, a
    mesh path relative to the project (a packed project) is found."""
    if d.get('shape') != 'mesh' or not d.get('mesh'):
        return None
    from . import textmesh
    return textmesh.spec_for(scene.mesh_path(d['mesh']) if scene is not None else d['mesh'])


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
    spec = text_spec(d, sc) if kind == 'collider' else None

    def fn(s):
        if target != s.kind:
            C.convert_kind(s, target)
        dd = _items(s, kind)[i]
        dd['burnable'] = True
        if kind == 'collider':
            s.data['spread']['enabled'] = True
        if spec is not None:   # letters catch all over at once: a flame in their shape for 2 s, then the fire is their own
            h = float(spec['height'])
            dom = s.data['domain']
            cell = max(s.domain_size()) / max(16.0, dom['resolution'] * dom['preview_scale'])
            e = dict(name=_unique(s.emitters, f'Flame on {dd["name"]}'), shape='mesh', mesh=dd['mesh'], position=tuple(float(x) for x in pos),
                     size=(1.0, 1.0, 1.0), thickness=round(max(0.04 * h, cell), 4), fuel=14.0, temperature=0.6, start=_t(doc),
                     stop=_t(doc) + 2.0, fade_out=0.6, noise=0.4, noise_freq=round(3.2 / h, 3))
        else:
            e = dict(name=_unique(s.emitters, f'Flame on {dd["name"]}'), shape='sphere', position=tuple(float(x) for x in at),
                     size=(r, r * 0.6, r), fuel=14.0, temperature=0.6, start=_t(doc), stop=_t(doc) + 3.0, fade_out=0.6, noise_freq=4.0)
        if s.kind == 'both':
            e['emits'] = 'fire'
        s.add_emitter(**e)
        if spec is not None:
            s.links.append({'child': ['emitter', e['name']], 'parent': ['collider', dd['name']], 'offset': [0.0, 0.0, 0.0],
                            'shape': True})
    doc.edit(f'Set {d["name"]} on fire', fn, structure=True)
    if spec is not None:
        steady = 'Burning logo' if spec.get('kind') == 'image' else 'Burning text'
        _say(win, f'{d["name"]} catches all over at frame {doc.frame}: it flares up and burns out (Spreading fire is on). For '
                  f'fire that keeps burning on it, use Create › {steady}.')
    else:
        _say(win, f'{d["name"]} catches at frame {doc.frame}: the flame at its base lasts 3 s, then the fire is its own'
                  + (' (Spreading fire is on).' if kind == 'collider' else '.'))
    doc.set_playing(True)


def repeat(win, sel=None):
    """Copies of the selection (or sel) in a row, a ring or scattered, from the Repeat dialog."""
    from ..scene import arrange as A, blocks
    from . import repeatdialog
    doc = win.doc
    sel = sel or doc.selected_objects()
    if not sel:
        QMessageBox.information(win, 'Repeat', 'Select what to repeat first (click it in the viewer or the object list).')
        return
    whole = blocks.with_attached(doc.scene, sel)
    lo, hi = A.footprint(doc.scene, whole, doc.frame)
    got = repeatdialog.ask(win, len(whole), (float(hi[0] - lo[0]), float(hi[2] - lo[2])))
    if not got:
        return
    places, ring = got
    new = doc.repeat(sel, places, ring=ring)
    _say(win, f'{len(new)} new things ({len(places) - (1 if ring else 0)} copies). They are selected: drag one to move them all, '
              'Ctrl+Z to undo.')
    doc.set_playing(True)


def group(win, sel=None):
    doc = win.doc
    sel = sel or doc.selected_objects()
    if len(sel) < 2:
        return
    doc.group(sel)
    name = _items(doc.scene, sel[0][0])[sel[0][1]]['name']
    _say(win, f'Grouped: the others are attached to {name} and go wherever it goes. Ungroup to move them on their own again.')


def select_group(win, kind, i):
    """Select a thing with everything attached to it (and what it is attached to, up to the top)."""
    from ..scene import blocks
    doc = win.doc
    sc = doc.scene
    top = (kind, i)
    seen = set()
    while True:
        d = _items(sc, top[0])[top[1]]
        link = sc.link_of(top[0], d['name'])
        if link is None or tuple(link['parent']) in seen:
            break
        seen.add(tuple(link['parent']))
        pi, _ = sc.find_object(*link['parent'])
        if pi is None:
            break
        top = (link['parent'][0], pi)
    doc.set_selected(blocks.with_attached(sc, [top]), top)


def save_block(win, sel):
    """Save objects sel = [(kind, i)] (and what is attached to them) as one of your blocks, under Yours in Create."""
    from pathlib import Path
    from ..scene import blocks
    from . import blockdialog
    sc = win.doc.scene
    sel = [s for s in sel if s[0] in ('emitter', 'collider', 'light', 'fabric')]
    if not sel:
        QMessageBox.information(win, 'Save as a block', 'Select what you built first (click it in the viewer or the object list).')
        return
    got = blockdialog.ask(win, sc, sel)
    if not got:
        return
    name, tip = got
    path = blocks.folder() / f'{blocks.slug(name)}{blocks.EXT}'
    if path.exists() and QMessageBox.question(win, 'Save as a block', f'You have a block called {name} already. Replace it?') \
            != QMessageBox.Yes:
        return
    try:
        path = blocks.save(sc, sel, name, tip, frame=win.doc.frame, path=path)
    except (ValueError, OSError) as ex:
        QMessageBox.information(win, 'Save as a block', str(ex))
        return
    if hasattr(win, 'create'):
        win.create.reload_blocks()
    _say(win, f'Saved {name} as a block: it is under Yours in Create (and {Path(path).name} is the file to share).')


def edit_text(win, kind, i):
    """Change the words, font or size of letters made by a Text block. Everything made from the same letters (the
    fire on them, the flame that lit them) changes with them."""
    from . import textdialog, textmesh
    doc = win.doc
    sc = doc.scene
    d = _items(sc, kind)[i]
    spec = text_spec(d, sc)
    if spec is None:
        return
    old_mesh = d['mesh']
    users = [(k2, o) for k2 in ('emitter', 'collider') for o in _items(sc, k2) if o.get('shape') == 'mesh' and o.get('mesh') == old_mesh]
    if any(k2 == 'emitter' and o.get('fuel', 0.0) > 0 and o.get('emits', 'fire') == 'fire' for k2, o in users):
        look = 'fire'
    elif any(k2 == 'emitter' for k2, o in users):
        look = 'water'
    else:
        look = 'solid'
    from . import shapedialog
    image = spec.get('kind') == 'image'
    new = (shapedialog if image else textdialog).ask(win, spec, width=C.scene_width(sc), look=look, edit=True)
    if not new or new == spec:
        return
    try:
        path = textmesh.make(new)[0]
    except (ValueError, OSError) as ex:
        QMessageBox.information(win, 'Shape' if image else 'Text', str(ex))
        return
    old_label, new_label = textmesh.label_of(spec), textmesh.label_of(new)

    def fn(s):
        renamed = []
        for k2 in ('emitter', 'collider'):
            items = _items(s, k2)
            for o in items:
                if o.get('shape') == 'mesh' and o.get('mesh') == old_mesh:
                    o['mesh'] = path
                    if old_label in o['name'] and old_label != new_label:
                        name = _unique(items, o['name'].replace(old_label, new_label))
                        renamed.append((k2, o['name'], name))
                        o['name'] = name
        for l in s.links:
            for end in ('child', 'parent'):
                for k2, a, b in renamed:
                    if list(l[end]) == [k2, a]:
                        l[end] = [k2, b]
        C.text_detail(s, path)
    doc.edit('Edit text', fn, structure=True)
    _say(win, ('The shape is changed.' if image else f'The letters read {new_label} now.') if old_label != new_label
         else ('The shape is changed.' if image else 'The letters are changed.'))
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

    many = doc.selected_objects() if (kind, i) in doc.selected_objects() else [(kind, i)]
    if len(many) > 1:
        head = m.addAction(f'{len(many)} selected')
        head.setEnabled(False)
        act('Group (attach to ' + d.get('name', kind) + ')', lambda: group(win, many),
            'They go wherever this one goes: move it and they all move', 'layers')
        act('Repeat…', lambda: repeat(win, many), 'Copies of all of them in a row, a ring or scattered', 'grid')
        act(f'Duplicate the {len(many)}', lambda: doc.duplicate_objects(many), '', 'copy')
        act('Save them as a block…', lambda: save_block(win, many), 'Keep them under Yours in Create', 'save')
        act(f'Delete the {len(many)}', lambda: doc.delete_objects(many), '', 'trash')
        m.addSeparator()
    head = m.addAction(d.get('name', kind))
    head.setEnabled(False)
    if kind in ('emitter', 'collider') and text_spec(d, sc) is not None:
        image = text_spec(d, sc).get('kind') == 'image'
        act('Edit shape…' if image else 'Edit text…', lambda: edit_text(win, kind, i),
            'Change the picture, its size or what of it is the shape' if image else 'Change the words, the font or the size',
            'shape' if image else 'text')
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
    attached = [l for l in sc.links if list(l['parent']) == [kind, d['name']]]
    if attached or link is not None:
        act('Select its group', lambda: select_group(win, kind, i), 'It and everything attached to it', 'layers')
    if attached:
        act(f'Ungroup ({len(attached)} attached)', lambda: (doc.ungroup((kind, i)), _say(win, 'They move on their own again.')),
            'What is attached to it moves on its own again', 'copy')
    act('Repeat…', lambda: repeat(win, [(kind, i)] if len(many) == 1 else many),
        'Copies of it in a row (torches down a path), a ring (jets round a stage) or scattered (spot fires)', 'grid')
    if path_mode is not None:
        act('Move along a path…', lambda: path_mode(kind, i), 'Click points on the ground to draw where it goes, over the next seconds',
            'line')
    act('Drop to the ground', lambda: drop_to_ground(win, kind, i), '', 'fit', enabled=kind != 'light')
    act('Look at it', lambda: frame_it(win, kind, i), 'The Build camera turns to it', 'camera')
    m.addSeparator()
    act('Save as a block…', lambda: save_block(win, many),
        'Keep it (and what is attached to it) under Yours in Create, to add to any scene or share as a file', 'save')
    act('Duplicate', lambda: duplicate(win, kind, i), '', 'copy')
    act('Delete', lambda: delete(win, kind, i), '', 'trash')


def selected_actions(win):
    """(label, glyph, callable) for the bar over a several-things selection's settings."""
    doc = win.doc
    many = doc.selected_objects()
    if len(many) < 2:
        return []
    return [('Group', 'layers', lambda: group(win, many)), ('Repeat…', 'grid', lambda: repeat(win, many)),
            ('Save block…', 'save', lambda: save_block(win, many)), ('Delete all', 'trash', lambda: doc.delete_objects(many))]


def quick_actions(win, sel):
    """The few most useful actions as (label, glyph, callable) for buttons on the object's page."""
    kind, i = sel
    sc = win.doc.scene
    d = _items(sc, kind)[i]
    out = []
    if kind in ('emitter', 'collider') and text_spec(d, sc) is not None:
        image = text_spec(d, sc).get('kind') == 'image'
        out.append(('Edit shape' if image else 'Edit text', 'shape' if image else 'text', lambda: edit_text(win, kind, i)))
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

