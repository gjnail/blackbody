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
        self._push_timer = QTimer(self)
        self._push_timer.setSingleShot(True)
        self._push_timer.setInterval(0)
        self._push_timer.timeout.connect(self._push)
        self._gen = 0

    # -- engine sync ----------------------------------------------------------------------------

    def _push(self):
        if self.worker is not None:
            self.worker.post('scene', self.scene.copy())

    def push_soon(self):
        self._push_timer.start()

    # -- undo plumbing -----------------------------------------------------------------------------

    def _snap(self):
        return json.dumps(self.scene.to_dict())

    def _restore(self, snap):
        path = self.scene.path
        self.scene = Scene.from_dict(json.loads(snap))
        self.scene.path = path
        n = len(self.scene.emitters)
        if self.selection[0] == 'emitter' and self.selection[1] >= n:
            self.selection = ('emitter', max(0, n - 1)) if n else ('section', 'domain')
        if self.selection[0] == 'collider' and self.selection[1] >= len(self.scene.colliders):
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

    def end_drag(self):
        """Stop merging: the next edit of the same parameter becomes its own undo step."""
        self._gen += 1

    # -- structure ---------------------------------------------------------------------------------

    def add_emitter(self, shape='sphere', **extra):
        def fn(s):
            base = dict(position=(0.0, 0.3, 0.0), size=(0.25, 0.25, 0.25), shape=shape)
            if shape == 'mesh':
                base.update(position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), name=Path(extra.get('mesh') or 'Mesh').stem)
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

    def rename(self, kind, i, name):
        def fn(s):
            (s.emitters if kind == 'emitter' else s.colliders)[i]['name'] = name
        self.edit('Rename', fn, structure=True)

    def select(self, sel):
        if sel != self.selection:
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

    def new(self, preset='campfire'):
        self.scene = presets.make(preset)
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
        def fn(s):
            if keep_shot:
                presets.apply_to(s, name)
            else:
                fresh = presets.make(name)
                s.__dict__.update(fresh.__dict__)
        self.edit(f'Preset: {presets.PRESETS[name]["name"]}', fn, structure=True)
        self.selection = ('emitter', 0) if self.scene.emitters else ('section', 'domain')
        self.sceneReplaced.emit()

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
