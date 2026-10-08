"""What you can do to a thing in the scene: set it on fire, make it float, soak it, turn a source into
another kind, attach it to something else, send it along a path... One list, used by the viewer's and the
object list's right-click menus and by the buttons at the top of an object's settings."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtWidgets import QMenu, QMessageBox

from ..scene import components as C
from ..scene import kinds as K
from . import icons, theme

MATERIALS = [('cotton', 'Cotton'), ('linen', 'Linen'), ('silk', 'Silk'), ('chiffon', 'Chiffon'), ('wool', 'Wool'), ('denim', 'Denim'),
             ('canvas', 'Canvas'), ('velvet', 'Velvet'), ('polyester', 'Polyester'), ('nylon', 'Nylon')]
PINS = [('top', 'By its top edge'), ('side', 'By one side'), ('top_corners', 'By its top corners'), ('corners', 'By four corners'),
        ('edges', 'By all its edges'), ('none', 'Not at all (it falls)')]


def _items(sc, kind):
    return K.items(sc, kind)


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
    elif kind == 'strands':   # grass: a flame at one end of the patch, among the blades' feet
        size = np.abs(np.asarray(sc.get((kind, i, 'size'), doc.frame), float))
        yaw = math.radians(float(sc.get((kind, i, 'yaw'), doc.frame)))
        off = np.array([-0.85 * size[0], 0.0, 0.0])
        at = pos + np.array([off[0] * math.cos(yaw), 0.05, -off[0] * math.sin(yaw)])
        r = max(0.06, min(0.25, 0.25 * size[1] + 0.05))
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
        if kind == 'strands':
            dd['burns'] = True
        else:
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
                  + (' (Spreading fire is on).' if kind == 'collider' else '.')
                  + (' It is breakable too, so it burns piece by piece and falls in.' if kind == 'collider' and d.get('breakable')
                     else ''))
    doc.set_playing(True)


def repeat(win, sel=None):
    """Copies of the selection (or sel) in a row, a ring or scattered, from the Repeat dialog."""
    from ..scene import arrange as A, blocks, caps
    from . import repeatdialog
    doc = win.doc
    sel = sel or doc.selected_objects()
    if not sel:
        QMessageBox.information(win, 'Repeat', 'Select what to repeat first (click it in the viewer or the object list).')
        return
    whole = blocks.with_attached(doc.scene, sel)
    lo, hi = A.footprint(doc.scene, whole, doc.frame)
    got = repeatdialog.ask(win, len(whole), (float(hi[0] - lo[0]), float(hi[2] - lo[2])), caps.room(doc.scene, whole))
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
    sel = [s for s in sel if s[0] in K.KINDS]
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


def make_fall(win, i, on=True, at_frame=False):
    """Make object i fall (a rigid body: it falls, tumbles and knocks into things), from the start or from this frame;
    on=False holds it where its keys put it again."""
    doc = win.doc
    sc = doc.scene
    d = sc.colliders[i]
    release = _t(doc) if at_frame else 0.0
    out = {}

    def fn(s):
        out['notes'] = C.make_dynamic(s, i, on=on, release=release)
    doc.edit(f'{"Make" if on else "Stop"} {d["name"]} {"fall" if on else "falling"}', fn)
    if on:
        when = f'from frame {doc.frame}' if at_frame else 'from the start'
        _say(win, f'{d["name"]} falls {when}: it tumbles, bounces and knocks into things, and the smoke and water push it. '
                  'Its keys now only set where it starts. ' + ' '.join(out.get('notes') or []))
    else:
        _say(win, f'{d["name"]} stays where its keys put it again.')
    doc.set_playing(True)


BREAKS_INTO = [('voronoi', 'Chunks'), ('bricks', 'Bricks'), ('shards', 'Shards (glass)'), ('splinters', 'Splinters (wood)')]


def make_breakable(win, i, on=True, fracture=None):
    """Make object i breakable (cut beforehand into pieces that come apart where it is hit hard enough), in the
    pattern its material breaks into unless given; on=False makes it whole again."""
    doc = win.doc
    d = doc.scene.colliders[i]
    out = {}

    def fn(s):
        out['notes'] = C.make_breakable(s, i, on=on, fracture=fracture)
    doc.edit(f'{"Make" if on else "Stop"} {d["name"]} {"breakable" if on else "breaking"}', fn)
    c = doc.scene.colliders[i]
    notes = ' '.join(out.get('notes') or [])
    if on and c.get('breakable'):
        into = dict(BREAKS_INTO).get(c.get('fracture'), 'pieces').lower()
        burns = (' It is burnable too: set it on fire and it burns piece by piece and falls in.' if c.get('burnable') else
                 ' Set it on fire too, and it burns piece by piece and falls in.')
        _say(win, f'{d["name"]} breaks into {into} where something hits it hard enough. Throw something at it, drop it, '
                  f'or Make it fall.{burns} {notes}'.strip())
    elif on:
        _say(win, notes or f'{d["name"]} cannot break.')
    else:
        _say(win, f'{d["name"]} is whole again.')
    doc.set_playing(True)


JOINT_ACTIONS = [('rope', 'Hang it on a rope', 'From a point 1.5 m above it: it swings, and the rope goes slack or snaps'),
                 ('spring', 'Put it on a spring', 'It bounces up and down on a spring from a point above it'),
                 ('hinge', 'Hinge it', 'Along a side (a door), its back edge (a lid), or through its middle (a wheel)'),
                 ('ball', 'Put it on a ball joint', 'It swings any way about a point at its top')]


def add_joint(win, i, kind, to=None, on=True):
    """Hang object i on a rope or a spring, or hinge it (to object `to`, or to a fixed point); on=False takes it off."""
    doc = win.doc
    d = doc.scene.colliders[i]
    other = doc.scene.colliders[to]['name'] if to is not None else None
    out = {}

    def fn(s):
        out['notes'] = C.add_joint(s, i, kind=kind, to=to, on=on)
    what = {'rope': 'on a rope', 'spring': 'on a spring', 'hinge': 'on a hinge', 'ball': 'on a ball joint'}.get(kind, '')
    doc.edit(f'{d["name"]} {what}' if on else f'Take {d["name"]} off its joint', fn)
    notes = ' '.join(out.get('notes') or [])
    if on:
        _say(win, (f'{d["name"]} hangs {what}' + (f' from {other}' if other else '') + '. It swings and falls from the start; '
                   'drag it to where it starts. ' + notes).strip())
    else:
        _say(win, notes or f'{d["name"]} is no longer joined to anything.')
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


def make_of(win, i, material):
    """What matter i is made of (sand, snow, mud...): its name follows if it was named after the old one."""
    from ..scene.params import MATTER_MATERIALS
    doc = win.doc
    d = doc.scene.matter[i]
    names = dict(MATTER_MATERIALS)
    old, new = names.get(d.get('material'), ''), names.get(material, material)

    def fn(s):
        m = s.matter[i]
        m['material'] = material
        if old and m['name'].startswith(old):
            m['name'] = new + m['name'][len(old):]
    doc.edit(f'Make it {new.lower()}', fn, structure=True)
    _say(win, f'It is {new.lower()} now.')
    doc.set_playing(True)


def grass_kind(win, i, kind):
    """What grass patch i is (lawn, long grass, wheat, reeds): its blades grow as tall as that kind's, and its name
    follows if it was named after the old kind."""
    from ..scene.params import STRAND_KINDS
    from ..engine.strands import KINDS as GROW
    doc = win.doc
    d = doc.scene.strands[i]
    names = dict(STRAND_KINDS)
    old, new = names.get(d.get('kind'), ''), names.get(kind, kind)

    def fn(s):
        g = s.strands[i]
        g['kind'] = kind
        sz = g['size']
        if not hasattr(sz, 'keys'):
            g['size'] = (float(sz[0]), float(GROW[kind].height), float(sz[2]))
        if old and g['name'].startswith(old):
            g['name'] = new + g['name'][len(old):]
    doc.edit(f'Make it {new.lower()}', fn, structure=True)
    _say(win, f'It is {new.lower()} now.')
    doc.set_playing(True)


def pour(win, i, on=True):
    """Matter i poured from a nozzle where it is (from this frame, if the shot has started), or a body of it again."""
    doc = win.doc
    d = doc.scene.matter[i]
    t = max(0.0, _t(doc))

    def fn(s):
        m = s.matter[i]
        m['pours'] = bool(on)
        if on:
            m['pour_start'] = t
            if float(m.get('pour_stop', 5.0)) <= t:
                m['pour_stop'] = t + 4.0
            sz = m['size']
            if not hasattr(sz, 'keys') and float(sz[0]) > 0.1:   # a body's size would be a very wide nozzle
                m['size'] = (0.04, 0.04, 0.04)
            v = m['velocity']
            if not hasattr(v, 'keys') and not any(float(x) for x in v):
                m['velocity'] = (0.0, -0.5, 0.0)
    doc.edit(f'{d["name"]} pours' if on else f'{d["name"]} stops pouring', fn, structure=True)
    _say(win, f'{d["name"]} pours from here at frame {doc.frame}: drag its arrow tip in Properties (Thrown at) to aim it.'
         if on else f'{d["name"]} is a body of it again.')
    doc.set_playing(True)


def drop_to_ground(win, kind, i):
    doc = win.doc
    sc = doc.scene
    d = _items(sc, kind)[i]
    pos = np.asarray(sc.get((kind, i, 'position'), doc.frame), float)
    if kind == 'fabric':
        y = 0.01 if d.get('orientation') == 'lying' else float(d.get('height', 1.0)) / 2 + 0.01
    elif kind == 'light' or (kind == 'matter' and d.get('pours')):
        return
    elif kind == 'strands':   # its position is its blades' feet
        y = 0.0
    else:
        size = np.asarray(sc.get((kind, i, 'size'), doc.frame), float)
        shape = d.get('shape')
        y = 0.0 if shape == 'mesh' else float(size[0] if (shape == 'sphere' and kind in ('collider', 'matter')) or shape == 'capsule'
                                              else size[1])
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
    doc.remove_object(kind, i)


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
    if kind in ('emitter', 'collider', 'matter', 'strands'):
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
        if d.get('dynamic') or d.get('joint', 'none') not in (None, 'none'):
            act('Stop it falling', lambda: make_fall(win, i, on=False), 'It stays where its keys put it (and comes off its joint)', 'stop')
        else:
            act('Make it fall', lambda: make_fall(win, i), 'A real object: it falls, tumbles, bounces and knocks into other things',
                'ball')
            act(f'Drop it at this frame ({doc.frame})', lambda: make_fall(win, i, at_frame=True),
                'Held where it is until this frame, then it falls', 'ball')
        joined = d.get('joint', 'none') not in (None, 'none')
        if joined:
            act('Take it off its joint', lambda: add_joint(win, i, d.get('joint'), on=False), 'It falls freely', 'stop')
        else:
            for key, label, tip in JOINT_ACTIONS:
                act(label, lambda k=key: add_joint(win, i, k), tip, 'line')
        others = [(j, x['name']) for j, x in enumerate(sc.colliders) if j != i]
        if others:
            for key, label in (('rope', 'Tie it with a rope to'), ('hinge', 'Hinge it to')):
                sub = m.addMenu(icons.glyph_icon('line', theme.MUTED, 16), label)
                for j, name in others:
                    sub.addAction(name, lambda k=key, j=j: add_joint(win, i, k, to=j))
        if d.get('breakable'):
            act('Stop it breaking', lambda: make_breakable(win, i, on=False), 'It stays whole', 'stop')
        else:
            mesh = d.get('shape') == 'mesh'
            a = act('Make it breakable', lambda: make_breakable(win, i), 'Cut into pieces that come apart where it is hit hard '
                    'enough: bricks for brick, shards for glass, splinters for wood, chunks for the rest. Burnable too: it burns '
                    'piece by piece and falls in', 'burst', enabled=not mesh)
            if mesh:
                a.setToolTip('Meshes cannot break yet: boxes, balls and cylinders can')
            sub = m.addMenu(icons.glyph_icon('burst', theme.MUTED, 16), 'Make it break into')
            sub.setEnabled(not mesh)
            for key, label in BREAKS_INTO:
                sub.addAction(label, lambda k=key: make_breakable(win, i, fracture=k))
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
    if kind == 'strands':
        from ..scene.params import STRAND_KINDS
        act('Set it on fire', guarded(lambda: set_on_fire(win, 'strands', i)), 'A flame at one end: the fire runs through it, '
            'faster downwind', 'flame')
        dry = float(d.get('dryness', 0.3))
        if dry < 0.7:
            act('Dry it out', lambda: set_value(win, 'strands', i, 'dryness', 0.9, 'Dry as straw: it catches at a spark.'),
                'Dry as straw: it catches easily and burns fast', 'flame')
        else:
            act('Make it green', lambda: set_value(win, 'strands', i, 'dryness', 0.1, 'Fresh and green: hard to light.'),
                'Fresh and green: hard to light', 'drop')
        sub = m.addMenu(icons.glyph_icon('grass', theme.MUTED, 16), 'Kind')
        for k, label in STRAND_KINDS:
            a = sub.addAction(label, lambda k=k: grass_kind(win, i, k))
            a.setCheckable(True)
            a.setChecked(d.get('kind') == k)
        on_all = d.get('grows_on') == 'everything'
        act('Grow on the ground only' if on_all else 'Grow on objects too',
            lambda: set_value(win, 'strands', i, 'grows_on', 'ground' if on_all else 'everything',
                              'It grows on the ground only.' if on_all else 'It grows on whatever is under it: a hillside, a mound.'),
            'On a hillside or a mound under it as well as the ground', 'hill')
    if kind == 'matter':
        from ..scene.params import MATTER_MATERIALS
        sub = m.addMenu(icons.glyph_icon('matter', theme.MUTED, 16), 'Made of')
        for k, label in MATTER_MATERIALS:
            a = sub.addAction(label, lambda k=k: make_of(win, i, k))
            a.setCheckable(True)
            a.setChecked(d.get('material') == k)
        if d.get('pours'):
            act(f'Start pouring at this frame ({doc.frame})', lambda: set_value(win, 'matter', i, 'pour_start', max(0.0, _t(doc)),
                                                                                  f'It pours from frame {doc.frame}.'), '', 'play')
            act(f'Stop pouring at this frame ({doc.frame})', lambda: set_value(win, 'matter', i, 'pour_stop', max(0.0, _t(doc)),
                                                                                 f'It stops pouring at frame {doc.frame}.'), '', 'stop')
            act('Make it a body of it', lambda: pour(win, i, on=False), 'A heap, a block or a ball of it at the start, instead of a pour',
                'matter')
        else:
            act(f'Let go at this frame ({doc.frame})', lambda: set_value(win, 'matter', i, 'release', _t(doc),
                                                                           f'It is held where it is until frame {doc.frame}, then let go.'),
                'Held in its shape until then (a column of sand that collapses)', 'matter')
            act('Pour it from here', lambda: pour(win, i), 'A stream from a nozzle where it is, from this frame', 'pour')
    m.addSeparator()
    others = [(k2, j, x['name']) for k2 in K.KINDS for j, x in enumerate(_items(sc, k2))
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
                ('Stop falling', 'stop', lambda: make_fall(win, i, on=False))
                if d.get('dynamic') or d.get('joint', 'none') not in (None, 'none') else ('Fall', 'ball', lambda: make_fall(win, i)),
                ('Whole', 'stop', lambda: make_breakable(win, i, on=False)) if d.get('breakable') else
                ('Breakable', 'burst', lambda: make_breakable(win, i)),
                ('Float', 'waves', lambda: make_float(win, i, 500.0)), ('Hot', 'flame', lambda: make_temperature(win, i, 300.0))]
    elif kind == 'fabric':
        out += [('Set on fire', 'flame', lambda: _guard(win, lambda: set_on_fire(win, 'fabric', i))),
                ('Soak' if float(d.get('wetness', 0.0)) < 0.5 else 'Dry', 'drop',
                 lambda: set_value(win, 'fabric', i, 'wetness', 1.0 if float(d.get('wetness', 0.0)) < 0.5 else 0.0, 'Done.')),
                ('Let go now', 'fabric', lambda: set_value(win, 'fabric', i, 'release', _t(win.doc), f'It falls from frame {win.doc.frame}.'))]
    elif kind == 'strands':
        dry = float(d.get('dryness', 0.3))
        out += [('Set on fire', 'flame', lambda: _guard(win, lambda: set_on_fire(win, 'strands', i))),
                ('Dry', 'flame', lambda: set_value(win, 'strands', i, 'dryness', 0.9, 'Dry as straw.')) if dry < 0.7 else
                ('Green', 'drop', lambda: set_value(win, 'strands', i, 'dryness', 0.1, 'Fresh and green.'))]
    elif kind == 'matter':
        if d.get('pours'):
            out += [('Pour from here', 'play', lambda: set_value(win, 'matter', i, 'pour_start', max(0.0, _t(win.doc)),
                                                                  f'It pours from frame {win.doc.frame}.')),
                    ('Stop here', 'stop', lambda: set_value(win, 'matter', i, 'pour_stop', max(0.0, _t(win.doc)),
                                                            f'It stops pouring at frame {win.doc.frame}.'))]
        else:
            out += [('Let go now', 'matter', lambda: set_value(win, 'matter', i, 'release', _t(win.doc),
                                                               f'It is let go at frame {win.doc.frame}.')),
                    ('Pour', 'pour', lambda: pour(win, i))]
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

