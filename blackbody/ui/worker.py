"""The engine thread. All GPU work happens here; the UI talks to it through a command queue and
gets frames, progress and results back as Qt signals."""
from __future__ import annotations

import logging
import queue
import threading
import time
import traceback

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

log = logging.getLogger('blackbody.worker')


def to_qimage(rgba):
    h, w = rgba.shape[:2]
    a = np.ascontiguousarray(rgba)
    return QImage(a.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()


def plate_fit(fw, fh, ow, oh):
    fa, oa = fw / fh, ow / oh
    if abs(fa - oa) < 1e-3:
        return (1.0, 1.0)
    return (oa / fa, 1.0) if fa > oa else (1.0, fa / oa)


class EngineWorker(QThread):
    ready = Signal(dict)                     # GPU info once the engine is up
    failed = Signal(str)                     # the engine could not start
    frameReady = Signal(object, int, dict)   # QImage, frame, stats
    simProgress = Signal(float, int)         # while catching up: fraction, frame
    cacheChanged = Signal(object)            # sorted cached frames
    footageInfo = Signal(object)             # dict describing the footage, or None
    message = Signal(str)
    jobProgress = Signal(float, str, object)  # fraction, text, preview QImage or None
    jobDone = Signal(object, bool, str)      # written files, cancelled, error text
    stillDone = Signal(object, str)          # final-quality frame arrays, error text
    status = Signal(str, float)              # what holds up the next frame: text, fraction (-1 unknown, -2 an error)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._q = queue.Queue()
        self.scene = None
        self.frame = 1
        self.playing = False
        self.mode = 'composite'
        self.quality = 0.5
        self.refine = True
        self.live = True
        self.footage = None
        self.engine = None
        self._hold = None       # (settings key, FootageHoldout) for the scene's holdout matte and depth pass
        self._need = True
        self._refined = False
        self._shown = None
        self._job_cancel = threading.Event()
        self._last_cache = None
        self._last_progress = 0.0
        self._load_seq = 0      # which scene load (Document.load_seq) the engine is on
        self.layer_engines = None   # render/layers.LayerEngines: the engines of the shot's other layers

    # -- API for the GUI thread -------------------------------------------------------------------

    def post(self, kind, value=None):
        self._q.put((kind, value))

    def cancel_job(self):
        self._job_cancel.set()

    def stop(self):
        self.post('quit')
        self.wait(5000)

    # -- thread -------------------------------------------------------------------------------------

    def run(self):
        try:
            from ..engine.engine import Engine
            from ..engine.gpu import GPU
            g = GPU()
            g.on_compile = self._compiling
            self.engine = Engine(g)
            self.ready.emit({'name': g.name, 'backend': g.backend})
        except Exception as ex:
            self.failed.emit(f'{ex}\n\n{traceback.format_exc()}')
            return
        while True:
            idle = not self.playing and not self._need
            timeout = None
            if idle:
                timeout = 0.2 if (self.refine and not self._refined and self._shown is not None) else None
            try:
                item = self._q.get(timeout=timeout) if idle else self._q.get_nowait()
            except queue.Empty:
                item = None
            items = [item] if item else []
            while True:
                try:
                    items.append(self._q.get_nowait())
                except queue.Empty:
                    break
            for kind, value in items:
                if kind == 'quit':
                    self._close_footage()
                    return
                self._handle(kind, value)
            if self.scene is None:
                continue
            try:
                if self.playing:
                    f = self.frame
                    if self._shown == f:
                        f = f + 1 if f < self.scene.end else self.scene.start
                    self.frame = f
                    self._show(f, refine=False)
                elif self._need:
                    if self._show(self.frame, refine=False):
                        self._need = False
                        self._refined = False
                elif self.refine and not self._refined and item is None:
                    self._show(self.frame, refine=True)
                    self._refined = True
            except Exception as ex:
                log.exception('render failed')
                self.message.emit(f'Render failed: {ex}')
                self.status.emit(f'This frame could not be simulated or drawn:\n{ex}', -2.0)
                self.playing = False
                self._need = False
                self._refined = True

    def _handle(self, kind, value):
        if kind == 'scene':
            self.scene = value
            self._need = True
            seq = getattr(value, 'load_seq', 0)
            if seq != self._load_seq:   # a different scene (preset, file): nothing of the last one is on screen
                self._load_seq = seq
                self._shown = None
                self._last_cache = None
                self.cacheChanged.emit([])
                self.status.emit('Setting up the simulation', -1.0)
        elif kind == 'frame':
            if value != self.frame or self._shown != value:
                self.frame = int(value)
                self._need = True
        elif kind == 'play':
            self.playing = bool(value)
            self._need = True
        elif kind == 'mode':
            self.mode = value
            self._need = True
        elif kind == 'quality':
            self.quality = float(value)
            self._need = True
        elif kind == 'refine':
            self.refine = bool(value)
        elif kind == 'live':
            self.live = bool(value)
        elif kind == 'footage':
            self._open_footage(value)
            self._need = True
        elif kind == 'restart':
            self.engine.invalidate()
            for e in (self.layer_engines.extra.values() if self.layer_engines is not None else ()):
                e.invalidate()
            self._need = True
        elif kind == 'cache_range':
            self._cache_range()
        elif kind == 'job':
            self._run_job(value)
            self._need = True
        elif kind == 'still':
            self._still(value)

    # -- footage ------------------------------------------------------------------------------------

    def _close_footage(self):
        if self.footage is not None:
            self.footage.close()
            self.footage = None

    def _open_footage(self, path):
        self._close_footage()
        if not path:
            self.footageInfo.emit(None)
            return
        try:
            from ..io.footage import Footage
            self.footage = Footage(path)
            f = self.footage
            self.footageInfo.emit({'path': path, 'width': f.width, 'height': f.height, 'fps': f.fps, 'frames': f.frames,
                                   'linear': f.linear, 'audio': f.audio, 'describe': f.describe(), 'kind': f.kind,
                                   'focal_35': f.focal_35})
        except Exception as ex:
            self.footage = None
            self.footageInfo.emit({'error': str(ex), 'path': path})

    def _holdout(self, frame):
        """(matte, depth) from the footage holdout settings for a frame, or None."""
        sc = self.scene
        c = sc.data['composite']
        import json
        roto = getattr(sc, 'roto', None) or []
        key = (c.get('holdout_matte'), c.get('holdout_depth'), c.get('matte_channel'), c.get('matte_invert'),
               (sc.footage or {}).get('offset', 0), sc.start, sc.path, json.dumps(roto, sort_keys=True), sc.output_size())
        if not (c.get('holdout_matte') or c.get('holdout_depth') or roto):
            return None   # (a liquid layer is drawn without it: render/layers.py)
        if self._hold is None or self._hold[0] != key:
            from ..io.holdout import FootageHoldout
            if self._hold is not None:
                self._hold[1].close()
            self._hold = (key, FootageHoldout(sc))
            for e in self._hold[1].errors:
                self.message.emit(f'Holdout: {e}')
        h = self._hold[1]
        try:
            return h.read(frame) if h.active else None
        except Exception as ex:
            self.message.emit(f'Holdout: {ex}')
            return None

    def _plate(self, frame):
        if self.footage is None or not self.scene.footage:
            return None
        off = int(self.scene.footage.get('offset', 0))
        return self.footage.read(frame - self.scene.start + off)

    # -- frames -------------------------------------------------------------------------------------

    def _interrupted(self):
        return not self._q.empty()

    def _compiling(self, source_file):
        """GPU.on_compile: a kernel is about to compile (seconds to minutes when its shader is new to the driver)."""
        self.status.emit(f'Compiling GPU shaders: {source_file.rsplit("/", 1)[-1]}', -1.0)

    def _progress(self, frac, frame):
        now = time.perf_counter()
        if now - self._last_progress > 0.1:
            self._last_progress = now
            self.simProgress.emit(float(frac), int(frame))

    def _layers(self):
        """The shot's layers to draw [(uid, scene)] back to front, and the one being edited (uid, scene)."""
        from ..render.layers import LayerEngines
        sc = self.scene
        if self.layer_engines is None or self.layer_engines.base is not self.engine:
            self.layer_engines = LayerEngines(self.engine)
        active_uid = getattr(sc, 'active_uid', 'base')
        active = sc.layer(active_uid) or sc
        if active is sc:
            active_uid = 'base'
        self.layer_engines.keep({u for u, _ in sc.layer_order(enabled_only=False)})
        only = getattr(sc, 'view_only', None)
        if only is not None or self.mode != 'composite':
            return [(active_uid, active)], (active_uid, active)   # the work view, and the passes: the layer being edited
        order = sc.layer_order() or [('base', sc)]
        return order, (active_uid, active)

    def _show(self, frame, refine=False):
        sc = self.scene
        order, (active_uid, active) = self._layers()
        LE = self.layer_engines
        for uid, lay in order:
            eng = LE.get(uid)
            eng.prepare(lay, final=False, soft=self.live)
            if frame not in eng.cache and eng.sim_frame != frame:
                far = eng.sim_frame is None or frame < eng.sim_frame or frame - eng.sim_frame > 2
                # a long catch-up (pre-roll, a jump) gives way to the UI even while playing, so Pause and edits work
                ok = eng.simulate_to(lay, frame, progress=self._progress if far else None,
                                     cancelled=self._interrupted if (far or not self.playing) else None)
                if not ok:
                    return False
                if far:
                    self.simProgress.emit(1.0, frame)
        W, H = sc.output_size()
        q = 1.0 if refine else self.quality
        w, h = max(16, int(W * q)), max(16, int(H * q))
        # a liquid element refracts the footage behind it, so it needs the footage too
        wants_plate = self.mode == 'composite' or (active.kind == 'liquid' and self.mode == 'fire')
        plate = self._plate(frame) if wants_plate else None
        fit = plate_fit(self.footage.width, self.footage.height, W, H) if plate is not None else (1.0, 1.0)
        # the refined frame gets motion blur too (from the velocity cached with the frame)
        from ..render.layers import render as render_layers
        front = render_layers(LE, order, frame, (w, h), mode=self.mode, final=False, samples=4 if refine else 1,
                              motion_blur=refine and bool(sc.data['render']['motion_blur']), plate=plate, plate_fit=fit,
                              holdout=self._holdout(frame))
        img = to_qimage(front.display_image())
        self._shown = frame
        eng = LE.get(active_uid)
        sc = active
        st = eng.stats() if eng.sim_frame is not None or eng.cache.frames() else front.stats()
        st['refined'] = refine
        st['preview'] = (w, h)
        # things that fall or float, where the simulation put them (drawn and picked there in the viewer)
        st['floats'] = eng.floating_overrides(frame) if hasattr(eng, 'floating_overrides') else None
        st['frame'] = frame
        st['load_seq'] = getattr(self.scene, 'load_seq', 0)
        st['layers'] = len(order)
        self.frameReady.emit(img, frame, st)
        frames = eng.cache.frames()
        if frames != self._last_cache:
            self._last_cache = frames
            self.cacheChanged.emit(frames)
        return True

    def _cache_range(self):
        sc = self.scene
        if sc is None:
            return
        eng = self.engine
        from ..render.layers import LayerEngines, simulate
        if self.layer_engines is None:
            self.layer_engines = LayerEngines(eng)
        self.message.emit('Simulating the frame range…')
        ok = simulate(self.layer_engines, sc.layer_order() or [('base', sc)], sc.end, progress=self._progress,
                      cancelled=self._interrupted)
        self.simProgress.emit(1.0, eng.sim_frame or sc.start)
        self.cacheChanged.emit(eng.cache.frames())
        self.message.emit('Frame range cached.' if ok else 'Caching stopped.')

    def _still(self, spec):
        """Render one frame at final quality and hand back the arrays (for 'Export frame')."""
        try:
            sc = spec['scene']
            from ..render.layers import LayerEngines, render as render_layers, simulate
            LE = LayerEngines(self.engine) if self.layer_engines is None else self.layer_engines
            mode = spec.get('mode', 'composite')
            order = sc.layer_order() or [('base', sc)]
            if mode != 'composite':
                order = [('base', sc)]
            simulate(LE, order, spec['frame'], final=True, progress=self._progress)
            W, H = sc.output_size()
            plate = self._plate(spec['frame'])
            fit = plate_fit(self.footage.width, self.footage.height, W, H) if plate is not None else (1.0, 1.0)
            eng = render_layers(LE, order, spec['frame'], (W, H), mode=mode, final=True,
                                samples=sc.data['render']['aa_samples'], motion_blur=sc.data['render']['motion_blur'],
                                plate=plate, plate_fit=fit, holdout=self._holdout(spec['frame']))
            result = {'display': eng.display_image(), 'aov': eng.aovs(), 'linear': eng.linear_comp(),
                      'frame': spec['frame'], 'path': spec.get('path'), 'mode': spec.get('mode', 'composite')}
            self.stillDone.emit(result, '')
        except Exception as ex:
            self.stillDone.emit(None, f'{ex}')
        finally:
            self.engine.invalidate()
            for e in (self.layer_engines.extra.values() if self.layer_engines is not None else ()):
                e.invalidate()

    def _run_job(self, spec):
        from ..render.job import RenderJob
        self._job_cancel.clear()
        sc = spec['scene']
        written, cancelled, err = [], False, ''
        try:
            job = RenderJob(sc, spec['outputs'], self.engine, frames=spec['frames'], final=True,
                            footage=self.footage if sc.footage else None, samples=spec.get('samples'),
                            motion_blur=spec.get('motion_blur'), engines=self.layer_engines)
            last_preview = [0.0]

            def progress(frac, text):
                now = time.perf_counter()
                img = None
                if now - last_preview[0] > 0.5:
                    last_preview[0] = now
                    try:
                        img = to_qimage(self.engine.display_image())
                    except Exception:
                        img = None
                self.jobProgress.emit(float(frac), text, img)

            written = job.run(progress=progress, cancelled=self._job_cancel.is_set)
            cancelled = job.cancelled
        except Exception as ex:
            err = f'{ex}\n\n{traceback.format_exc()}'
        finally:
            self.engine.invalidate()
        self.jobDone.emit(written, cancelled, err)
