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
        self._need = True
        self._refined = False
        self._shown = None
        self._job_cancel = threading.Event()
        self._last_cache = None
        self._last_progress = 0.0

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
            self.engine = Engine()
            g = self.engine.gpu
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
                self.playing = False
                self._need = False
                self._refined = True

    def _handle(self, kind, value):
        if kind == 'scene':
            self.scene = value
            self._need = True
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
                                   'linear': f.linear, 'audio': f.audio, 'describe': f.describe(), 'kind': f.kind})
        except Exception as ex:
            self.footage = None
            self.footageInfo.emit({'error': str(ex), 'path': path})

    def _plate(self, frame):
        if self.footage is None or not self.scene.footage:
            return None
        off = int(self.scene.footage.get('offset', 0))
        return self.footage.read(frame - self.scene.start + off)

    # -- frames -------------------------------------------------------------------------------------

    def _interrupted(self):
        return not self._q.empty()

    def _progress(self, frac, frame):
        now = time.perf_counter()
        if now - self._last_progress > 0.1:
            self._last_progress = now
            self.simProgress.emit(float(frac), int(frame))

    def _show(self, frame, refine=False):
        sc = self.scene
        eng = self.engine
        eng.prepare(sc, final=False, soft=self.live)
        if frame not in eng.cache and eng.sim_frame != frame:
            far = eng.sim_frame is None or frame < eng.sim_frame or frame - eng.sim_frame > 2
            ok = eng.simulate_to(sc, frame, progress=self._progress if far else None,
                                 cancelled=self._interrupted if not self.playing else None)
            if far:
                self.simProgress.emit(1.0, frame)
            if not ok:
                return False
        W, H = sc.output_size()
        q = 1.0 if refine else self.quality
        w, h = max(16, int(W * q)), max(16, int(H * q))
        # a liquid element refracts the footage behind it, so it needs the footage too
        wants_plate = self.mode == 'composite' or (sc.kind == 'liquid' and self.mode == 'fire')
        plate = self._plate(frame) if wants_plate else None
        fit = plate_fit(self.footage.width, self.footage.height, W, H) if plate is not None else (1.0, 1.0)
        eng.render(sc, frame, (w, h), mode=self.mode, final=False, samples=4 if refine else 1,
                   motion_blur=False, plate=plate, plate_fit=fit)
        img = to_qimage(eng.display_image())
        self._shown = frame
        st = eng.stats()
        st['refined'] = refine
        st['preview'] = (w, h)
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
        eng.prepare(sc, final=False, soft=False)
        self.message.emit('Simulating the frame range…')
        ok = eng.simulate_to(sc, sc.end, progress=self._progress, cancelled=self._interrupted)
        self.simProgress.emit(1.0, eng.sim_frame or sc.start)
        self.cacheChanged.emit(eng.cache.frames())
        self.message.emit('Frame range cached.' if ok else 'Caching stopped.')

    def _still(self, spec):
        """Render one frame at final quality and hand back the arrays (for 'Export frame')."""
        try:
            sc = spec['scene']
            eng = self.engine
            eng.prepare(sc, final=True)
            eng.simulate_to(sc, spec['frame'], progress=self._progress)
            W, H = sc.output_size()
            plate = self._plate(spec['frame'])
            fit = plate_fit(self.footage.width, self.footage.height, W, H) if plate is not None else (1.0, 1.0)
            eng.render(sc, spec['frame'], (W, H), mode=spec.get('mode', 'composite'), final=True,
                       samples=sc.data['render']['aa_samples'], motion_blur=sc.data['render']['motion_blur'],
                       plate=plate, plate_fit=fit)
            result = {'display': eng.display_image(), 'aov': eng.aovs(), 'linear': eng.linear_comp(),
                      'frame': spec['frame'], 'path': spec.get('path'), 'mode': spec.get('mode', 'composite')}
            self.stillDone.emit(result, '')
        except Exception as ex:
            self.stillDone.emit(None, f'{ex}')
        finally:
            self.engine.invalidate()

    def _run_job(self, spec):
        from ..render.job import RenderJob
        self._job_cancel.clear()
        sc = spec['scene']
        written, cancelled, err = [], False, ''
        try:
            job = RenderJob(sc, spec['outputs'], self.engine, frames=spec['frames'], final=True,
                            footage=self.footage if sc.footage else None, samples=spec.get('samples'),
                            motion_blur=spec.get('motion_blur'))
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
