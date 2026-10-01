"""The open document: scene, selection, current frame and undo history.

Every edit goes through here. Edits are recorded as before/after scene snapshots (scenes are small),
consecutive edits to the same parameter merge into one undo step, and the engine thread gets a copy
of the scene once per event-loop turn however many edits happened in it.
"""
from __future__ import annotations

import json

import numpy as np
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QUndoCommand, QUndoStack

from ..scene import PROJECT_EXT, Scene, presets
from ..scene.anim import Curve


class Snapshot(QUndoCommand):
    def __init__(self, doc, before, after, text, merge_key=None):
        super().__init__(text)
        self.doc = doc
        self.before = before
        self.after = after
        self.merge_key = merge_key
        self._first = True

    def id(self):
        return 7 if self.merge_key else -1

    def mergeWith(self, other):
        if other.merge_key is None or other.merge_key != self.merge_key:
            return False
        self.after = other.after
        return True

    def redo(self):
        if self._first:  # the edit was already applied live
            self._first = False
            return
        self.doc._restore(self.after)

    def undo(self):
        self.doc._restore(self.before)


class Document(QObject):
    sceneReplaced = Signal()          # whole scene changed (load, undo, preset): rebuild UI
    paramChanged = Signal(object)     # one parameter path changed
    structureChanged = Signal()       # emitters/colliders added or removed, renamed
    selectionChanged = Signal(object)
    frameChanged = Signal(int)
    dirtyChanged = Signal(bool)
    footageChanged = Signal(object)   # footage info dict or None
    playingChanged = Signal(bool)
    loadStarted = Signal(str)         # a different scene replaced this one (preset, file, new): its name
    viewChanged = Signal()            # the work view turned on or off, or moved
    layersChanged = Signal()          # a layer added, removed, renamed, hidden, moved, or another one picked

    def __init__(self, worker=None, parent=None):
        super().__init__(parent)
        self.worker = worker
        self.active = 'base'    # the layer being edited (its uid); self.scene is that layer, self.shot the whole shot
        self.scene = presets.make('campfire')
        self.undo = QUndoStack(self)
        self.undo.cleanChanged.connect(self._clean_changed)
        self.frame = self.scene.start
        self.selection = ('emitter', 0)
        self.footage_info = None
        self.picked = []   # other objects selected with the main one (Ctrl+click, a box drag); their settings do not show
        self.playing = False
        self.load_seq = 0   # counts scene loads; the engine tags its frames with it, so the viewer knows a frame is stale
        self.baseline = self.scene.copy()   # the scene as loaded (preset or file): settings changed from it are marked
        self.work_view = None   # ui/workview.WorkView while the viewer looks through a camera of its own
        self._push_timer = QTimer(self)
        self._push_timer.setSingleShot(True)
        self._push_timer.setInterval(0)
        self._push_timer.timeout.connect(self._push)
        self._gen = 0

    # -- the shot and its layers -------------------------------------------------------------------------

    @property
    def scene(self):
        """The layer being edited (the shot itself when it has no other layers)."""
        return self.shot.layer(self.active) or self.shot

    @scene.setter
    def scene(self, value):
        """A whole new shot (a file opened, a new scene): edit its base layer."""
        self.shot = value
        self.active = 'base'

    def layers(self):
        """[(uid, scene)] back to front, hidden ones too."""
        return self.shot.layer_order(enabled_only=False)

    def set_active(self, uid):
        """Edit another layer of the shot."""
        if uid == self.active or self.shot.layer(uid) is None:
            return
        self.active = uid
        self.baseline = self.scene.copy()
        self.selection = ('emitter', 0) if self.scene.emitters else ('section', 'domain')
        if self.work_view is not None:
            from .workview import WorkView
            self.work_view = WorkView.behind_shot(self.scene, self.frame)
        self.sceneReplaced.emit()
        self.layersChanged.emit()
        self.push_soon()

    def add_layer(self, scale='person'):
        """A new, empty layer in front of the others, to build or load an effect into. Returns its uid."""
        import copy as _copy
        import uuid
        from ..scene import components
        from ..scene.model import share_shot
        uid = 'l' + uuid.uuid4().hex[:8]

        def fn(s):
            shot = self.shot
            lay = components.new_scene('auto', scale)
            lay.uid = uid
            lay.name = f'Layer {len(shot.layers) + 2}'
            share_shot(shot, lay)
            cam = _copy.deepcopy(shot.data['camera'])   # placed where the base layer is, a little to the side
            for k in ('distance', 'target_y', 'near', 'far'):
                cam[k] = lay.data['camera'][k]
            if not isinstance(cam.get('anchor_x'), dict) and isinstance(cam.get('anchor_x'), float):
                cam['anchor_x'] = min(0.85, cam['anchor_x'] + 0.2) if cam['anchor_x'] < 0.6 else max(0.15, cam['anchor_x'] - 0.2)
            lay.data['camera'] = cam
            shot.layers.append(lay)
        self.edit('Add layer', fn, structure=True)
        self.set_active(uid)
        self.layersChanged.emit()
        return uid

    def remove_layer(self, uid):
        if uid == 'base':
            return
        if self.active == uid:
            self.set_active('base')
        self.edit('Delete layer', lambda s: setattr(self.shot, 'layers', [l for l in self.shot.layers if l.uid != uid]), structure=True)
        self.layersChanged.emit()

    def layer_set(self, uid, key, value, text):
        """Rename a layer ('name') or show/hide it ('enabled')."""
        def fn(s):
            lay = self.shot.layer(uid)
            if lay is not None:
                setattr(lay, key, value)
        self.edit(text, fn, structure=True)
        self.layersChanged.emit()

    def move_layer(self, uid, delta):
        """Move a layer back (-1) or forward (+1) among the others: the last one is in front."""
        order = [u for u, _ in self.layers()]
        if uid not in order:
            return
        i = order.index(uid)
        j = max(0, min(len(order) - 1, i + delta))
        if i == j:
            return
        order.insert(j, order.pop(i))

        def fn(s):
            shot = self.shot
            by = {l.uid: l for l in shot.layers}
            shot.layers = [by[u] for u in order if u != 'base']
            shot.base_index = order.index('base')
        self.edit('Move layer', fn, structure=True)
        self.layersChanged.emit()

    def duplicate_layer(self, uid):
        import uuid
        new = 'l' + uuid.uuid4().hex[:8]

        def fn(s):
            src = self.shot.layer(uid)
            lay = src.copy()
            lay.layers, lay.base_index = [], 0
            lay.uid = new
            lay.name = f'{src.name} copy'
            lay.enabled = True
            self.shot.layers.append(lay)
        self.edit('Duplicate layer', fn, structure=True)
        self.set_active(new)
        self.layersChanged.emit()

    # -- engine sync ----------------------------------------------------------------------------

    def _push(self):
        if self.worker is not None:
            c = self.shot.copy()
            c.load_seq = self.load_seq
            c.active_uid = self.active
            if self.work_view is not None:   # the work view shows the layer being edited, on its own
                c.view_only = self.active
                lay = c.layer(self.active) or c
                self.work_view.apply(lay)
                c.footage = None
            self.worker.post('scene', c)

    def camera(self, frame):
        """The camera the viewer looks through at a frame: the shot's, or the work view's."""
        spec, fire = self.scene.camera(frame)
        if self.work_view is not None:
            spec = self.work_view.spec(spec, fire)
        return spec, fire

    def set_work_view(self, wv):
        """Look through a work view (None: back to the shot's camera). Only the viewer changes."""
        self.work_view = wv
        self.viewChanged.emit()
        self.push_soon()

    def move_work_view(self):
        """The work view moved: show the scene through it (cached frames are re-drawn, not re-simulated)."""
        self.viewChanged.emit()
        self.push_soon()

    def push_soon(self):
        self._push_timer.start()

    def _clean_changed(self, clean):
        try:
            self.dirtyChanged.emit(not clean)
        except RuntimeError:   # the window is going away
            pass

    # -- the camera matched to the footage ----------------------------------------------------------------

    def ground_match(self, g=None):
        """The camera that ground match g (or the shot's) makes, as a scene/groundmatch.Match. Raises ValueError."""
        from ..scene import groundmatch as GM
        sc = self.scene
        g = g or sc.ground
        if not g:
            raise ValueError('The ground has not been lined up.')
        W, H = sc.output_size()
        f = int(g.get('frame', self.frame))
        sensor = float(sc.v('camera', 'sensor_mm', f))
        focal = float(g.get('focal_mm') or sc.v('camera', 'focal_mm', f))
        by_side = g.get('scale_by') == 'side'
        verts = [[(x * W, y * H) for x, y in seg] for seg in g['verticals']] if g.get('verticals') else None
        if g.get('mode') == 'horizon':
            hz = [(x * W, y * H) for x, y in g['horizon']]
            b = g.get('base', (0.5, 0.8))
            return GM.solve_horizon(hz, (b[0] * W, b[1] * H), (W, H), focal * W / sensor, float(g.get('height', 1.6)), verts)
        return GM.solve([(x * W, y * H) for x, y in g['corners']], (W, H), focal_px=focal * W / sensor,
                        height=None if by_side else float(g.get('height', 1.6)), side=float(g.get('side', 2.0)) if by_side else None,
                        use_solved_lens=g.get('lens', 'picture') == 'picture', verticals=verts)

    def set_ground(self, g, merge=True):
        """Line up the ground: keep g and point the camera (every layer's: it is the footage's) the way it says. The
        first time, the effect goes to the middle of the rectangle. Returns the Match; raises ValueError, changing
        nothing, when g cannot be a rectangle on the ground."""
        from ..scene.anim import Curve
        m = self.ground_match(g)
        W = self.scene.output_size()[0]
        first = not self.scene.ground
        g = dict(g)
        g['plane'] = list(m.plane) if m.slope > 1.5 else [0.0, 1.0, 0.0]
        g['slope'] = round(float(m.slope), 2)
        g['surfaces'] = list((self.scene.ground or {}).get('surfaces') or [])

        def fn(s):
            for x in [self.shot] + list(self.shot.layers):
                c = x.data['camera']
                for k in ('position', 'rotation', 'focal_mm', 'roll'):
                    if isinstance(c.get(k), Curve):
                        x.clear_anim(('camera', k))   # a still camera: tracking the camera move keys it again
                c['mode'] = 'free'
                c['use_anchor'] = False
                c['roll'] = 0.0
                c['position'] = m.position
                c['rotation'] = m.rotation
                if m.focal_solved:
                    c['focal_mm'] = round(m.focal_px * float(c['sensor_mm']) / W, 2)
                elif g.get('focal_mm'):
                    c['focal_mm'] = float(g['focal_mm'])
                c['near'] = min(float(c['near']), 0.05)
                if first:
                    c['fire_position'] = (0.0, 0.0, 0.0)
                    c['fire_yaw'] = 0.0
                x.ground = dict(g)
        self.edit('Line up the ground', fn, merge_key=('ground', self._gen) if merge else None, path=('camera', 'position'))
        return m

    def finish_ground(self):
        """After lining up: a sloping ground becomes a solid slope (a surface) in the scene, so water runs down it."""
        from . import surfaces
        g = self.scene.ground or {}
        old = [s_ for s_ in g.get('surfaces') or [] if s_.get('kind') == 'slope']
        if not old and g.get('slope', 0.0) <= 1.5:
            return

        def fn(s):
            for x in [self.shot] + list(self.shot.layers):
                if not x.ground:
                    continue
                x.colliders = [c for c in x.colliders if not any(c.get('mesh') == o['mesh'] for o in old)]
                x.ground['surfaces'] = [s_ for s_ in x.ground.get('surfaces') or [] if s_.get('kind') != 'slope']
            if g.get('slope', 0.0) > 1.5:
                sx, sy, sz = self.scene.domain_size()
                srf = surfaces.slope_surface(g['plane'], max(4.0, 1.5 * max(sx, sz)))
                self._add_surface(srf, 'Sloping ground')
        self.edit('Sloping ground', fn, structure=True)

    def _add_surface(self, srf, name):
        from . import surfaces
        path, centre = surfaces.write(srf)
        entry = {'name': name, 'kind': srf['kind'], 'top': srf['top'], 'mesh': path, 'centre': list(centre)}
        if srf.get('treads'):
            entry['treads'] = srf['treads']
        for x in [self.shot] + list(self.shot.layers):
            if not x.ground:
                continue
            names = {c['name'] for c in x.colliders}
            n, base = 2, name
            while entry['name'] in names:
                entry['name'] = f'{base} {n}'
                n += 1
            x.ground = dict(x.ground, surfaces=list(x.ground.get('surfaces') or []) + [dict(entry)])
            x.add_collider(**surfaces.collider_of(x, entry))
        return entry

    def add_surface(self, srf):
        """A real surface lined up in the footage (ui/surfaces.py) becomes a solid in every layer. Returns its name."""
        from . import surfaces
        name = surfaces.NAMES.get(srf['kind'], 'Surface')
        out = {}
        self.edit(f'Add {name.lower()}', lambda s: out.update(self._add_surface(srf, name)), structure=True)
        i = next((k for k, c in enumerate(self.scene.colliders) if c.get('mesh') == out.get('mesh')), None)
        if i is not None:
            self.select(('collider', i), force=True)
        return out.get('name', name)

    def clear_ground(self):
        """Back to pinning the effect in the frame (2D), without a matched camera."""
        def fn(s):
            for x in [self.shot] + list(self.shot.layers):
                x.ground = None
                c = x.data['camera']
                c['mode'] = 'orbit'
                c['use_anchor'] = True
        self.edit('Pin in the frame', fn, path=('camera', 'mode'))

    def shot_from_view(self, key=False):
        """Point the shot's camera (the active layer's) the way the Build view looks. key=False: a still camera (any
        camera animation goes); key=True: a key at this frame, so keys at two frames make a camera move. Returns a
        note for the user; raises ValueError when the camera is matched to footage."""
        from ..scene.anim import Curve
        sc = self.scene
        if self.work_view is None:
            raise ValueError('Look around in Build first: the shot’s camera takes the Build view.')
        if sc.footage or sc.track:
            raise ValueError('The shot’s camera matches your footage, so it is not moved from Build. Place the effect in the '
                             'shot instead: switch to Shot and drag it, or set Camera › Placement.')
        vals = self.work_view.shot_values(sc, self.frame)
        f = self.frame
        anim = ('position', 'rotation', 'focal_mm', 'roll')

        def fn(s):
            c = s.data['camera']
            fresh = c['mode'] != 'free' or bool(c.get('use_anchor'))
            if fresh or not key:   # a camera of a different kind (or a still one): its old animation goes
                for k in anim:
                    if isinstance(c.get(k), Curve):
                        s.clear_anim(('camera', k), f)
            c['mode'] = 'free'
            c['use_anchor'] = False
            c['roll'] = 0.0
            c['near'] = min(float(c['near']), vals['near'])
            rot = list(vals['rotation'])
            if key and isinstance(c.get('rotation'), Curve):
                # the same turn by the short way round, so the camera does not spin between keys
                prev = s.get(('camera', 'rotation'), f)
                rot = [r + 360.0 * round((p - r) / 360.0) for r, p in zip(rot, prev)]
            for k, v in (('position', vals['position']), ('rotation', tuple(rot)), ('focal_mm', vals['focal_mm'])):
                if key:
                    s.set_key(('camera', k), f, v)
                else:
                    s.set(('camera', k), v)
        self.edit('Key the shot camera' if key else 'Shot camera from this view', fn, structure=True)
        if not key:
            return 'The shot’s camera now sees what you see here. Tab shows the shot.'
        c = self.scene.data['camera']
        n = len(c['position'].keys) if isinstance(c['position'], Curve) else 1
        return (f'Shot camera keyed at frame {f} ({n} keys): it moves between them.' if n > 1 else
                f'Shot camera keyed at frame {f}. Go to another frame, look from somewhere else and key again for a camera move.')

    # -- undo plumbing -----------------------------------------------------------------------------

    def _snap(self):
        return json.dumps({'shot': self.shot.to_dict(), 'active': self.active})

    def _counts(self):
        sc = self.scene
        return (len(sc.emitters), len(sc.colliders), len(sc.lights), len(sc.fabrics), self.active)

    def _restore(self, snap):
        counts = self._counts()
        path = self.shot.path
        old = (self.scene.preset, self.scene.kind)
        d = json.loads(snap)
        active = d.get('active', 'base')
        self.scene = Scene.from_dict(d['shot'] if 'shot' in d else d)
        self.shot.path = path
        for l in self.shot.layers:
            l.path = path
        if self.shot.layer(active) is not None:
            self.active = active
        self.layersChanged.emit()
        if (self.scene.preset, self.scene.kind) != old:   # undoing or redoing a preset load
            self._begin_load(self.scene.name)
            self.baseline = self.scene.copy()
        n = len(self.scene.emitters)
        if self.selection[0] == 'emitter' and self.selection[1] >= n:
            self.selection = ('emitter', max(0, n - 1)) if n else ('section', 'domain')
        if self.selection[0] == 'collider' and self.selection[1] >= len(self.scene.colliders):
            self.selection = ('section', 'domain')
        if self.selection[0] == 'light' and self.selection[1] >= len(self.scene.lights):
            self.selection = ('section', 'lighting')
        if self.selection[0] == 'fabric' and self.selection[1] >= len(self.scene.fabrics):
            self.selection = ('section', 'domain')
        if counts != self._counts():   # things came or went: only the main selection stays
            self.picked = []
        self.sceneReplaced.emit()
        self.push_soon()

    def edit(self, text, fn, merge_key=None, structure=False, path=None):
        """Apply fn(scene) as one undoable step."""
        before = self._snap()
        kind_before = self.scene.kind
        fn(self.scene)
        if getattr(self.scene, 'links', None):
            self.scene.apply_links()   # attached objects follow what they are attached to
        if (self.shot.ground or {}).get('surfaces'):   # real surfaces stay where they are in the world
            from . import surfaces
            for x in [self.shot] + list(self.shot.layers):
                surfaces.sync(x)
        if self.shot.layers:
            self.shot.sync_layers(source=self.scene)   # what one layer changes of the shot, every layer has
        after = self._snap()
        if before == after:
            return
        self.undo.push(Snapshot(self, before, after, text, merge_key))
        if self.scene.kind != kind_before:
            self.sceneReplaced.emit()   # what the scene simulates changed: the panels change with it
        if structure:
            self.picked = []   # indices may have moved: the main selection stays, the others go
            self.structureChanged.emit()
        if path is not None:
            self.paramChanged.emit(path)
        self.push_soon()

    # -- parameters --------------------------------------------------------------------------------

    def value(self, path):
        return self.scene.get(path, self.frame)

    def is_animated(self, path):
        return self.scene.curve(path) is not None

    def has_key(self, path):
        c = self.scene.curve(path)
        return c is not None and c.has_key(self.frame)

    def set(self, path, value, merge=True):
        label = Scene.spec(path).label
        key = ('set', path, self._gen) if merge else None
        link = None
        if len(path) == 3 and path[2] in ('position', 'end') and getattr(self.scene, 'links', None):
            items = {'emitter': self.scene.emitters, 'collider': self.scene.colliders, 'light': self.scene.lights,
                     'fabric': self.scene.fabrics}[path[0]]
            link = self.scene.link_of(path[0], items[path[1]].get('name'))

        def fn(s):
            if link is not None:   # an attached object moved by hand: it keeps the new place relative to its parent
                pi, parent = s.find_object(*link['parent'])
                if parent is not None:
                    pp = s.get((link['parent'][0], pi, 'position'), self.frame)
                    if path[2] == 'position':
                        link['offset'] = [float(v) - float(q) for v, q in zip(value, pp)]
                    else:
                        link['end_offset'] = [float(v) - float(q) for v, q in zip(value, pp)]
            s.set(path, value, self.frame)
        self.edit(f'Change {label}', fn, merge_key=key, path=path)

    def toggle_key(self, path):
        label = Scene.spec(path).label
        if self.has_key(path):
            self.edit(f'Remove key: {label}', lambda s: s.remove_key(path, self.frame), path=path)
        else:
            self.edit(f'Set key: {label}', lambda s: s.set_key(path, self.frame), path=path)

    def clear_animation(self, path):
        self.edit('Clear animation', lambda s: s.clear_anim(path, self.frame), path=path)

    def baseline_value(self, path):
        """The value a setting had when the scene was loaded, or None if there is nothing to compare with
        (an object added since, or one whose place in the list changed)."""
        b = self.baseline
        if b is None:
            return None
        try:
            if path[0] in ('emitter', 'collider', 'light', 'fabric'):
                lists = {'emitter': 'emitters', 'collider': 'colliders', 'light': 'lights', 'fabric': 'fabrics'}[path[0]]
                cur, base = getattr(self.scene, lists), getattr(b, lists)
                if path[1] >= len(base) or base[path[1]].get('name') != cur[path[1]].get('name'):
                    return None
            return b.get(path, self.frame)
        except (KeyError, IndexError, TypeError, AttributeError):
            return None

    # -- keys (the animation editor) -------------------------------------------------------------------

    def _keys_edit(self, text, keys, fn):
        self.edit(text, fn)
        for path in sorted({k[0] for k in keys}, key=str):
            self.paramChanged.emit(path)

    def move_keys(self, keys, df):
        """Move keys [(path, frame)] by df frames as one undo step (a key landing on another replaces it).
        Returns where they are now."""
        moved_to = []

        def fn(s):
            by = {}
            for path, f in keys:
                by.setdefault(path, []).append(f)
            for path, frames in by.items():
                c = s.curve(path)
                if c is None:
                    continue
                take = [k for k in c.keys if any(abs(k[0] - f) < 1e-6 for f in frames)]
                c.keys = [k for k in c.keys if k not in take]
                for f, v, interp in take:
                    c.set(f + df, v, interp)
                    moved_to.append((path, float(f + df)))
        self._keys_edit('Move keys' if len(keys) > 1 else 'Move key', keys, fn)
        return moved_to

    def delete_keys(self, keys):
        def fn(s):
            for path, f in keys:
                if s.curve(path) is not None:
                    s.remove_key(path, f)
        self._keys_edit('Delete keys' if len(keys) > 1 else 'Delete key', keys, fn)

    def add_key(self, path, frame):
        self._keys_edit('Set key', [(path, frame)], lambda s: s.set_key(path, frame))

    def set_key_interp(self, keys, interp):
        def fn(s):
            for path, f in keys:
                c = s.curve(path)
                for k in (c.keys if c is not None else []):
                    if abs(k[0] - f) < 1e-6:
                        k[2] = interp
        self._keys_edit('Change easing', keys, fn)

    def end_drag(self):
        """Stop merging: the next edit of the same parameter becomes its own undo step."""
        self._gen += 1

    # -- structure ---------------------------------------------------------------------------------

    def add_emitter(self, shape='sphere', **extra):
        def fn(s):
            base = dict(position=(0.0, 0.3, 0.0), size=(0.25, 0.25, 0.25), shape=shape)
            if shape == 'mesh':
                base.update(position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), name=Path(extra.get('mesh') or 'Mesh').stem)
            elif shape == 'volume':
                # the volume where it is in its file, as smoke that fills the box at the start
                src = (extra.get('volume') or 'Volume').split('#/')
                name = src[-1].rstrip('?world').split('/')[-1] if len(src) > 1 else Path(src[0]).stem.rstrip('.#')
                base.update(position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), name=name, fuel=0.0, temperature=0.0, smoke=4.0,
                            noise=0.0, start=0.0, embers=False)
            elif shape == 'capsule':
                base.update(position=(-0.5, 0.1, 0.0), end=(0.5, 0.1, 0.0), size=(0.1, 0.1, 0.1))
            elif shape == 'ring':
                base.update(position=(0.0, 0.05, 0.0), size=(0.3, 0.03, 0.3))
            elif shape in ('cylinder', 'box'):
                base.update(position=(0.0, 0.05, 0.0), size=(0.3, 0.05, 0.3))
            if s.kind == 'liquid':
                # a liquid source: a small nozzle up in the air that sets the velocity of what it pours
                base.update(name=f'Source {len(s.emitters) + 1}', vel_blend=1.0, embers=False, noise=0.0)
                if shape in ('cylinder', 'sphere', 'box', 'cone'):
                    base.update(position=(0.0, 0.6, 0.0), size=(0.04, 0.03, 0.04), velocity=(0.0, -0.5, 0.0))
            base.update(extra)
            s.add_emitter(**base)
        self.edit('Add source' if self.scene.kind == 'liquid' else 'Add emitter', fn, structure=True)
        self.select(('emitter', len(self.scene.emitters) - 1))

    def duplicate_emitter(self, i):
        import copy

        def fn(s):
            e = copy.deepcopy(s.emitters[i])
            e['name'] = e['name'] + ' copy'
            e['seed'] = int(e.get('seed', 0)) + 1
            s.emitters.insert(i + 1, e)
        self.edit('Duplicate emitter', fn, structure=True)
        self.select(('emitter', i + 1))

    def remove_emitter(self, i):
        self.edit('Delete emitter', lambda s: s.emitters.pop(i), structure=True)
        n = len(self.scene.emitters)
        self.select(('emitter', min(i, n - 1)) if n else ('section', 'combustion'))

    def add_collider(self, shape='box', **extra):
        def fn(s):
            base = dict(shape=shape)
            if shape == 'mesh':
                base.update(position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), name=Path(extra.get('mesh') or 'Mesh').stem)
            base.update(extra)
            s.add_collider(**base)
        self.edit('Add collider', fn, structure=True)
        self.select(('collider', len(self.scene.colliders) - 1))

    def remove_collider(self, i):
        self.edit('Delete collider', lambda s: s.colliders.pop(i), structure=True)
        self.select(('section', 'domain'))

    def add_light(self, kind='point', **extra):
        def fn(s):
            s.add_light(kind=kind, **extra)
        self.edit('Add light', fn, structure=True)
        self.select(('light', len(self.scene.lights) - 1))

    def remove_light(self, i):
        self.edit('Delete light', lambda s: s.lights.pop(i), structure=True)
        self.select(('section', 'lighting'))

    def add_fabric(self, **extra):
        def fn(s):
            s.add_fabric(**extra)
        self.edit('Add fabric', fn, structure=True)
        self.select(('fabric', len(self.scene.fabrics) - 1))

    def duplicate_fabric(self, i):
        import copy

        def fn(s):
            d = copy.deepcopy(s.fabrics[i])
            d['name'] = d['name'] + ' copy'
            s.fabrics.insert(i + 1, d)
        self.edit('Duplicate fabric', fn, structure=True)
        self.select(('fabric', i + 1))

    def remove_fabric(self, i):
        self.edit('Delete fabric', lambda s: s.fabrics.pop(i), structure=True)
        self.select(('section', 'domain'))

    def add_component(self, key, at=None, mesh=None):
        """Add a building block (scene/components.py) as one undoable step, select what it added and return
        notes for the user. Raises ValueError if the scene cannot take it."""
        from ..scene import components
        comp = components.get(key)
        result = {}

        def fn(s):
            result['added'], result['notes'] = components.add(s, key, at=at, mesh=mesh)
        kind_before = self.scene.kind
        probe = self.scene.copy()
        components.add(probe, key, at=at, mesh=mesh)   # raises before anything changes if it cannot go in
        self.edit(f'Add {comp.name}', fn, structure=True)
        if self.scene.kind != kind_before:
            self.sceneReplaced.emit()   # the panels change with the kind of simulation
        added = result.get('added') or []
        if added:
            self.select(added[-1], force=True)
        return result.get('notes', [])

    def new_from_scratch(self, kind='fire', scale='person'):
        """Replace the scene with an empty one of a kind and size, as one undoable step. With a shot to keep
        (footage, a track, an animated camera), the footage, framing, range and output settings stay."""
        from ..scene import components
        keep = self.has_shot()

        def fn(s):
            fresh = components.new_scene(kind, scale)
            if keep:
                cam = dict(s.data['camera'])
                for k in ('distance', 'target_y', 'near', 'far', 'pitch'):
                    cam[k] = fresh.data['camera'][k]
                fresh.data['camera'] = cam
                fresh.data['render'] = dict(s.data['render'])
                fresh.data['composite'] = dict(s.data['composite'])
                fresh.footage, fresh.track, fresh.roto = s.footage, s.track, list(getattr(s, 'roto', []))
            saved = {'path': s.path} if s.path else {}
            saved.update(layers=s.layers, base_index=s.base_index, uid=s.uid, enabled=s.enabled)
            if s is not self.shot:
                saved['name'] = s.name
            s.__dict__.update(fresh.__dict__)
            s.__dict__.update(saved)
            if s is not self.shot:
                from ..scene.model import share_shot
                share_shot(self.shot, s)
        name = components.new_scene(kind, scale).name
        self._begin_load(name)
        self.edit('New sky scene' if kind == 'cloud' else 'New scene', fn, structure=True)
        self._loaded()
        if not self.scene.emitters:
            self.selection = ('section', 'domain')

    def roto_edit(self, text, fn):
        """An undoable change to the roto shapes: fn(list of shapes)."""
        self.edit(text, lambda s: fn(s.roto), structure=True)

    def attach(self, child, parent):
        """Attach object child (kind, i) to parent (kind, i): from now on it goes where the parent goes."""
        sc = self.scene
        lists = {'emitter': sc.emitters, 'collider': sc.colliders, 'light': sc.lights, 'fabric': sc.fabrics}
        c = lists[child[0]][child[1]]
        p = lists[parent[0]][parent[1]]
        cp = np.asarray(sc.get((child[0], child[1], 'position'), self.frame), float)
        pp = np.asarray(sc.get((parent[0], parent[1], 'position'), self.frame), float)

        def fn(s):
            s.links = [l for l in s.links if list(l['child']) != [child[0], c['name']]]
            link = {'child': [child[0], c['name']], 'parent': [parent[0], p['name']], 'offset': [float(x) for x in cp - pp]}
            if child[0] == 'emitter' and c.get('shape') == 'capsule':
                link['end_offset'] = [float(x) for x in np.asarray(s.get(('emitter', child[1], 'end'), self.frame), float) - pp]
            s.links.append(link)
        self.edit(f'Attach {c["name"]} to {p["name"]}', fn, structure=True)

    def detach(self, kind, i):
        sc = self.scene
        items = {'emitter': sc.emitters, 'collider': sc.colliders, 'light': sc.lights, 'fabric': sc.fabrics}[kind]
        name = items[i]['name']
        self.edit(f'Detach {name}', lambda s: setattr(s, 'links', [l for l in s.links if list(l['child']) != [kind, name]]),
                  structure=True)

    def rename(self, kind, i, name):
        def fn(s):
            items = {'emitter': s.emitters, 'collider': s.colliders, 'light': s.lights, 'fabric': s.fabrics}[kind]
            old = items[i]['name']
            items[i]['name'] = name
            for l in getattr(s, 'links', None) or []:   # links follow the name
                for end in ('child', 'parent'):
                    if list(l[end]) == [kind, old]:
                        l[end] = [kind, name]
        self.edit('Rename', fn, structure=True)

    def select(self, sel, force=False):
        """Select an object or a section. `force` tells the panels again even if it is already selected
        (a click on it in the viewer brings its settings up)."""
        if sel != self.selection or force or self.picked:
            self.selection = sel
            self.picked = []
            self.selectionChanged.emit(sel)

    # -- several things at once -------------------------------------------------------------------------

    def _lists(self):
        sc = self.scene
        return {'emitter': sc.emitters, 'collider': sc.colliders, 'light': sc.lights, 'fabric': sc.fabrics}

    def selected_objects(self):
        """The objects selected: the main one (whose settings show) first, then the others selected with it."""
        lists = self._lists()
        out = []
        for s in [self.selection] + list(self.picked):
            if s and s[0] in lists and 0 <= s[1] < len(lists[s[0]]) and s not in out:
                out.append(tuple(s))
        return out

    def set_selected(self, sels, primary=None):
        """Select several objects; `primary` (or the first) is the one whose settings show."""
        sels = list(dict.fromkeys(tuple(s) for s in sels))
        if not sels:
            self.select(('section', 'domain'))
            return
        primary = tuple(primary) if primary is not None and tuple(primary) in sels else sels[0]
        self.selection = primary
        self.picked = [s for s in sels if s != primary]
        self.selectionChanged.emit(primary)

    def pick(self, sel):
        """Ctrl+click: add an object to the selection (it becomes the main one), or take it out."""
        cur = self.selected_objects()
        sel = tuple(sel)
        if sel in cur:
            if len(cur) == 1:
                return
            cur.remove(sel)
            self.set_selected(cur, self.selection if self.selection != sel else cur[0])
        else:
            self.set_selected(cur + [sel], sel)

    def selection_state(self):
        """The selected objects (and what is attached to them) as they are now, for moving or turning them together."""
        import copy
        from ..scene import blocks
        lists = self._lists()
        return {s: copy.deepcopy(lists[s[0]][s[1]]) for s in blocks.with_attached(self.scene, self.selected_objects())}

    def arrange(self, starts, pivot=(0.0, 0.0, 0.0), theta=0.0, move=(0.0, 0.0, 0.0), label='Move together'):
        """Turn (theta degrees about the vertical through pivot) and move the objects of a selection_state() from where
        they were then: the whole of a path moves. One undo step per drag."""
        import copy
        from ..scene import arrange as A

        def fn(s):
            names = {(k, d['name']) for (k, _), d in starts.items()}
            lists = {'emitter': s.emitters, 'collider': s.colliders, 'light': s.lights, 'fabric': s.fabrics}
            for (kind, i), d0 in starts.items():
                items = lists[kind]
                if i >= len(items):
                    continue
                link = s.link_of(kind, d0['name'])
                if link is not None and tuple(link['parent']) in names:
                    continue   # it goes with what it is attached to
                d = copy.deepcopy(d0)
                A.transform(kind, d, pivot, theta, move)
                for k in ('position', 'end', 'yaw', 'direction', 'velocity'):
                    if k in d:
                        items[i][k] = d[k]
                if link is not None:   # attached to something that stays: it keeps its new place relative to it
                    pi, parent = s.find_object(*link['parent'])
                    if parent is not None:
                        pp = s.get((link['parent'][0], pi, 'position'), self.frame)
                        here = s.get((kind, i, 'position'), self.frame)
                        link['offset'] = [float(v) - float(q) for v, q in zip(here, pp)]
        self.edit(label, fn, merge_key=(label, self._gen))

    def delete_objects(self, sels):
        sels = [tuple(s) for s in sels]
        n = len(sels)

        def fn(s):
            lists = {'emitter': s.emitters, 'collider': s.colliders, 'light': s.lights, 'fabric': s.fabrics}
            for kind, i in sorted(sels, key=lambda x: (x[0], -x[1])):
                if i < len(lists[kind]):
                    lists[kind].pop(i)
        self.edit(f'Delete {n} things' if n > 1 else 'Delete', fn, structure=True)
        self.select(('section', 'domain'))

    def duplicate_objects(self, sels):
        """Copies of the objects (and what is attached to them) beside them, selected."""
        from ..scene import arrange as A, blocks
        sel = blocks.with_attached(self.scene, sels)
        lo, hi = A.footprint(self.scene, sel, self.frame)
        step = float(max(hi[0] - lo[0], 0.05)) * 1.15
        out = {}

        def fn(s):
            out['new'] = A.copy_objects(s, sel, move=(step, 0.0, 0.0))
        self.edit('Duplicate', fn, structure=True)
        self.set_selected(out['new'], out['new'][0] if out['new'] else None)
        return out['new']

    def group(self, sels):
        """Attach the others to the main one: they go wherever it goes."""
        sels = [tuple(s) for s in sels]
        if len(sels) < 2:
            return
        import numpy as np
        lists = self._lists()
        parent = sels[0]
        pname = lists[parent[0]][parent[1]]['name']
        f = self.frame

        def fn(s):
            pp = np.asarray(s.get((parent[0], parent[1], 'position'), f), float)
            for kind, i in sels[1:]:
                c = {'emitter': s.emitters, 'collider': s.colliders, 'light': s.lights, 'fabric': s.fabrics}[kind][i]
                s.links = [l for l in s.links if list(l['child']) != [kind, c['name']]]
                cp = np.asarray(s.get((kind, i, 'position'), f), float)
                link = {'child': [kind, c['name']], 'parent': [parent[0], pname], 'offset': [float(x) for x in cp - pp]}
                if kind == 'emitter' and c.get('shape') == 'capsule':
                    link['end_offset'] = [float(x) for x in np.asarray(s.get(('emitter', i, 'end'), f), float) - pp]
                s.links.append(link)
        self.edit(f'Group with {pname}', fn)
        self.set_selected(sels, parent)

    def ungroup(self, sel):
        name = self._lists()[sel[0]][sel[1]]['name']
        self.edit(f'Ungroup {name}', lambda s: setattr(s, 'links', [l for l in s.links if list(l['parent']) != [sel[0], name]]))

    def repeat(self, sels, places, ring=False):
        """Copies of the objects (and what is attached to them) at places [(dx, dz, turn)] about their middle (see
        scene/arrange.layout); with ring, the originals take the first place. The originals and copies end up selected."""
        from ..scene import arrange as A, blocks
        sel = blocks.with_attached(self.scene, sels)
        lo, hi = A.footprint(self.scene, sel, self.frame)
        pivot = ((lo[0] + hi[0]) / 2, 0.0, (lo[2] + hi[2]) / 2)
        out = {}

        def fn(s):
            new = []
            rest = places
            if ring and places:
                dx, dz, th = places[0]
                for kind, i in sel:
                    A.transform(kind, A.items(s, kind)[i], pivot, th, (dx, 0.0, dz))
                rest = places[1:]
            for k, (dx, dz, th) in enumerate(rest):
                new += A.copy_objects(s, sel, pivot, th, (dx, 0.0, dz), seed=k + 1)
            out['new'] = new
        self.edit(f'Repeat ({len(places)})', fn, structure=True)
        self.set_selected(sel + out['new'], sels[0] if sels else None)
        return out['new']

    # -- time ----------------------------------------------------------------------------------------

    def set_frame(self, f):
        f = int(max(self.scene.start, min(self.scene.end, f)))
        if f != self.frame:
            self.frame = f
            self.frameChanged.emit(f)
        if self.worker is not None and not self.playing:
            self.worker.post('frame', f)

    def frame_from_engine(self, f):
        """The engine advanced during playback."""
        if f != self.frame:
            self.frame = f
            self.frameChanged.emit(f)

    def set_playing(self, on):
        on = bool(on)
        if on == self.playing:
            return
        self.playing = on
        if self.worker is not None:
            if not on:
                self.worker.post('frame', self.frame)
            self.worker.post('play', on)
        self.playingChanged.emit(on)

    # -- files ------------------------------------------------------------------------------------------

    def _begin_load(self, name):
        self.picked = []
        self.load_seq += 1
        self.loadStarted.emit(name or 'Untitled')

    def _reframe_work_view(self):
        if self.work_view is not None:
            from .workview import WorkView
            self.work_view = WorkView.framing(self.scene)
            self.viewChanged.emit()

    def has_shot(self):
        """The scene is matched to a real shot (footage, a 2D track or an animated camera), which a preset
        loaded with Keep my shot keeps. Without one, a preset comes with its own camera and frame range."""
        s = self.scene
        return bool(s.footage or s.track or any(isinstance(v, Curve) for v in s.data['camera'].values()))

    def new(self, preset='campfire'):
        self.scene = presets.make(preset)
        self.baseline = self.scene.copy()
        self._begin_load(self.scene.name)
        self.undo.clear()
        self.frame = self.scene.start
        self.selection = ('emitter', 0)
        self._set_footage_path(None)
        self.sceneReplaced.emit()
        self.frameChanged.emit(self.frame)
        self.push_soon()

    def open(self, path):
        s = Scene.load(path)
        self.scene = s
        self.baseline = s.copy()
        self._reframe_work_view()
        self._begin_load(s.name)
        self.undo.clear()
        self.frame = s.start
        self.selection = ('emitter', 0) if s.emitters else ('section', 'domain')
        self._set_footage_path((s.footage or {}).get('path'))
        self.sceneReplaced.emit()
        self.frameChanged.emit(self.frame)
        self.push_soon()

    def save(self, path=None):
        path = path or self.shot.path
        if not path:
            raise ValueError('no path')
        if not str(path).lower().endswith(PROJECT_EXT):
            path = str(path) + PROJECT_EXT
        self.shot.name = self.shot.name if self.shot.name not in ('', 'Untitled') else Path(path).stem
        self.shot.save(path)
        self.undo.setClean()
        return path

    def load_preset(self, name, keep_shot=True):
        """Replace the effect with a built-in preset, from its first frame. With keep_shot and a shot to keep
        (has_shot), the footage, camera placement, frame range and output settings stay."""
        keep_shot = keep_shot and self.has_shot()

        def fn(s):
            if keep_shot:
                presets.apply_to(s, name)
            else:
                fresh = presets.make(name)
                saved = {'path': s.path, 'name': s.name} if s.path else {}   # a saved project stays that project
                saved.update(layers=s.layers, base_index=s.base_index, uid=s.uid, enabled=s.enabled)
                if s is not self.shot:
                    saved.pop('name', None)
                s.__dict__.update(fresh.__dict__)
                s.__dict__.update(saved)
                if s is not self.shot:   # a layer keeps the shot it is in
                    from ..scene.model import share_shot
                    share_shot(self.shot, s)
        self._begin_load(presets.PRESETS[name]['name'])
        self.edit(f'Preset: {presets.PRESETS[name]["name"]}', fn, structure=True)
        self._loaded()

    def load_user_preset(self, path, keep_shot=True):
        """Replace the effect with a preset the user saved (a scene file), as load_preset does."""
        other = Scene.load(path)
        keep_sections = ('camera', 'render', 'composite') if keep_shot and self.has_shot() else ()

        def fn(s):
            saved = {k: dict(s.data[k]) for k in keep_sections}
            s.data = other.data
            for k, v in saved.items():
                s.data[k] = v
            s.emitters = other.emitters
            s.colliders = other.colliders
            s.lights = other.lights
            s.fabrics = other.fabrics
            s.preset = other.preset
        self._begin_load(Path(path).stem)
        self.edit(f'Preset: {Path(path).stem}', fn, structure=True)
        self._loaded()

    def _loaded(self):
        """After a preset load: select its first emitter and start from the first frame."""
        self.baseline = self.scene.copy()
        self._reframe_work_view()
        self.selection = ('emitter', 0) if self.scene.emitters else ('section', 'domain')
        self.frame = self.scene.start
        self.sceneReplaced.emit()
        self.frameChanged.emit(self.frame)
        if self.worker is not None:   # the new scene first, then the frame, so the engine never runs the old one there
            self._push_timer.stop()
            self._push()
            self.worker.post('frame', self.frame)

    # -- footage ------------------------------------------------------------------------------------------

    def _set_footage_path(self, path):
        if self.worker is not None:
            self.worker.post('footage', path)
        if not path:
            self.footage_info = None
            self.footageChanged.emit(None)

    def import_footage(self, path):
        self._set_footage_path(str(Path(path).resolve()))

    def footage_opened(self, info):
        """Engine thread opened (or failed to open) footage: fit the shot to it."""
        self.footage_info = info
        if info and 'error' not in info:
            new_clip = not (self.scene.footage and self.scene.footage.get('path') == info['path'])

            def fn(s):
                prev = s.footage or {}
                s.footage = {'path': info['path'], 'offset': int(prev.get('offset', 0)) if prev.get('path') == info['path'] else 0}
                if new_clip and info.get('focal_35') and not s.ground:   # the lens the file says it was shot with
                    s.data['camera']['focal_mm'] = float(info['focal_35'])
                    s.data['camera']['sensor_mm'] = 36.0
                r = s.data['render']
                r['width'], r['height'] = int(info['width']), int(info['height'])
                r['fps'] = float(round(info['fps'], 3))
                r['end'] = r['start'] + int(info['frames']) - 1
                if info.get('linear'):
                    s.data['composite']['plate_transform'] = 'linear'
                elif s.data['composite']['plate_transform'] == 'linear':
                    s.data['composite']['plate_transform'] = 'srgb'
            if not (self.scene.footage and self.scene.footage.get('path') == info['path']
                    and self.scene.data['render']['width'] == info['width']):
                self.edit('Import footage', fn, structure=True)
            else:
                fn(self.scene)
                if self.shot.layers:
                    self.shot.sync_layers(source=self.scene)
            if self.baseline is not None:
                fn(self.baseline)   # the size, rate and range the footage sets are the shot's, not changes to the effect
            self.sceneReplaced.emit()
            if self.frame > self.scene.end:
                self.set_frame(self.scene.end)
        self.footageChanged.emit(info)

    def remove_footage(self):
        def fn(s):
            s.footage = None
            s.track = None
        self.edit('Remove footage', fn, structure=True)
        self._set_footage_path(None)
        self.sceneReplaced.emit()
