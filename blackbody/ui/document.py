"""The open document: scene, selection, current frame and undo history.

Every edit goes through here. Edits are recorded as before/after scene snapshots (scenes are small),
consecutive edits to the same parameter merge into one undo step, and the engine thread gets a copy
of the scene once per event-loop turn however many edits happened in it.
"""
from __future__ import annotations

import json
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

    def __init__(self, worker=None, parent=None):
        super().__init__(parent)
        self.worker = worker
        self.scene = presets.make('campfire')
        self.undo = QUndoStack(self)
        self.undo.cleanChanged.connect(lambda clean: self.dirtyChanged.emit(not clean))
        self.frame = self.scene.start
        self.selection = ('emitter', 0)
        self.footage_info = None
        self.playing = False
        self.load_seq = 0   # counts scene loads; the engine tags its frames with it, so the viewer knows a frame is stale
        self.baseline = self.scene.copy()   # the scene as loaded (preset or file): settings changed from it are marked
        self._push_timer = QTimer(self)
        self._push_timer.setSingleShot(True)
        self._push_timer.setInterval(0)
        self._push_timer.timeout.connect(self._push)
        self._gen = 0

    # -- engine sync ----------------------------------------------------------------------------

    def _push(self):
        if self.worker is not None:
            c = self.scene.copy()
            c.load_seq = self.load_seq
            self.worker.post('scene', c)

    def push_soon(self):
        self._push_timer.start()

    # -- undo plumbing -----------------------------------------------------------------------------

    def _snap(self):
        return json.dumps(self.scene.to_dict())

    def _restore(self, snap):
        path = self.scene.path
        old = (self.scene.preset, self.scene.kind)
        self.scene = Scene.from_dict(json.loads(snap))
        self.scene.path = path
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
        self.sceneReplaced.emit()
        self.push_soon()

    def edit(self, text, fn, merge_key=None, structure=False, path=None):
        """Apply fn(scene) as one undoable step."""
        before = self._snap()
        fn(self.scene)
        after = self._snap()
        if before == after:
            return
        self.undo.push(Snapshot(self, before, after, text, merge_key))
        if structure:
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
        self.edit(f'Change {label}', lambda s: s.set(path, value, self.frame), merge_key=key, path=path)

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
        comp = components.BY_KEY[key]
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
                fresh.footage, fresh.track = s.footage, s.track
            saved = {'path': s.path} if s.path else {}
            s.__dict__.update(fresh.__dict__)
            s.__dict__.update(saved)
        name = components.new_scene(kind, scale).name
        self._begin_load(name)
        self.edit('New sky scene' if kind == 'cloud' else 'New scene', fn, structure=True)
        self._loaded()
        if not self.scene.emitters:
            self.selection = ('section', 'domain')

    def rename(self, kind, i, name):
        def fn(s):
            {'emitter': s.emitters, 'collider': s.colliders, 'light': s.lights, 'fabric': s.fabrics}[kind][i]['name'] = name
        self.edit('Rename', fn, structure=True)

    def select(self, sel, force=False):
        """Select an object or a section. `force` tells the panels again even if it is already selected
        (a click on it in the viewer brings its settings up)."""
        if sel != self.selection or force:
            self.selection = sel
            self.selectionChanged.emit(sel)

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
        self.load_seq += 1
        self.loadStarted.emit(name or 'Untitled')

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
        self._begin_load(s.name)
        self.undo.clear()
        self.frame = s.start
        self.selection = ('emitter', 0) if s.emitters else ('section', 'domain')
        self._set_footage_path((s.footage or {}).get('path'))
        self.sceneReplaced.emit()
        self.frameChanged.emit(self.frame)
        self.push_soon()

    def save(self, path=None):
        path = path or self.scene.path
        if not path:
            raise ValueError('no path')
        if not str(path).lower().endswith(PROJECT_EXT):
            path = str(path) + PROJECT_EXT
        self.scene.name = self.scene.name if self.scene.name not in ('', 'Untitled') else Path(path).stem
        self.scene.save(path)
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
                s.__dict__.update(fresh.__dict__)
                s.__dict__.update(saved)
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
            def fn(s):
                prev = s.footage or {}
                s.footage = {'path': info['path'], 'offset': int(prev.get('offset', 0)) if prev.get('path') == info['path'] else 0}
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
