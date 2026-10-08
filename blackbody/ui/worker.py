"""The engine thread. All GPU work happens here; the UI talks to it through a command queue and
gets frames, progress and results back as Qt signals.

When the GPU fails under it (its device lost to a driver reset or a time-out, or out of memory),
the thread makes its engines again, on a new device if the old one is gone, keeping the frames
they had cached, says so, and carries on (_recover): no restart of the app."""
from __future__ import annotations

import gc
import logging
import queue
import threading
import time
import traceback

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

log = logging.getLogger('blackbody.worker')

RECOVER_TRIES = 3   # times in a row the engine is made again after a GPU failure (no frame drawn between) before it stops


def to_qimage(rgba):
    h, w = rgba.shape[:2]
    a = np.ascontiguousarray(rgba)
    return QImage(a.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()


def LU_on(scene):
    """Whether a scene lights its set with Lume."""
    return (scene.data.get('lume') or {}).get('engine', 'classic') == 'lume'


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
        self._precompile = None     # the background compile of the slow shaders (engine/precompile.py)
        self._recoveries = 0    # engines made again after a GPU failure since a frame was last drawn

    # -- API for the GUI thread -------------------------------------------------------------------

    def post(self, kind, value=None):
        self._q.put((kind, value))

    def cancel_job(self):
        self._job_cancel.set()

    def stop(self):
        self.post('quit')
        self.wait(5000)
        from ..engine import precompile
        precompile.stop(self._precompile)   # (what it finished stays compiled; the rest is done next time)

    # -- thread -------------------------------------------------------------------------------------

    def run(self):
        try:
            from ..engine.engine import Engine
            from ..engine.gpu import GPU
            g = GPU()
            g.on_compile = g.on_wait = self._compiling
            self.engine = Engine(g)
            self.ready.emit({'name': g.name, 'backend': g.backend})
            from ..engine import precompile
            self._precompile = precompile.start_background(g)   # (the liquids' march, in a thread, on this GPU)
        except Exception as ex:
            self.failed.emit(f'{ex}\n\n{traceback.format_exc()}')
            return
        while True:
            idle = not self.playing and not self._need
            timeout = None
            if idle:
                timeout = 0.2 if (self.refine and not self._refined and self._shown is not None) else None
                if timeout is not None and self._lume_pending():
                    timeout = 0.005   # (Lume is gathering light paths: on at once, unless an edit comes in)
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
                self._collected()
                self._handle(kind, value)
            if self.scene is None:
                continue
            self._collected()
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
                    # (Lume adds a few light paths per pixel each refinement: refine again until it has them all)
                    self._refined = not self._lume_pending()
            except Exception as ex:
                self._frame_failed(ex)

    def _frame_failed(self, ex):
        """A frame could not be simulated or drawn (`ex`): start the GPU's engines again if it was the GPU (_recover), to
        draw the frame again; else say so, with the advice to restart the app when the GPU keeps failing."""
        if self._recover(ex):
            self._need = True        # (the same frame again, on the engine made anew)
            self._refined = False
            return
        log.error('render failed', exc_info=ex)
        self.message.emit(f'Render failed: {ex}{self._advice}')
        self.status.emit(f'This frame could not be simulated or drawn:\n{ex}{self._advice}', -2.0)
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

    # -- GPU failures ---------------------------------------------------------------------------------

    def _recover(self, ex):
        """After `ex` from a simulation or a render: if the GPU failed (engine/gpu.py failure: its device lost, or out of
        memory; or it no longer answers), make the engines again with what they had cached (Engine.carry), on a new device
        if the old one is gone, and say so. On running out of memory the GPU first plans within less, so the scene's grids
        are made to fit (scene/model.py memory_plan, with a notice of what was cut). False when it was not the GPU, or it
        failed RECOVER_TRIES times in a row, or could not be started again: then `_advice` says to restart the app, for
        the caller to add to what it says (not said here: the caller's own message would replace it)."""
        from ..engine import gpu as G
        from ..engine.engine import Engine
        from ..render.layers import LayerEngines
        self._advice = ''
        old = self.engine
        if old is None:
            return False
        why = G.failure(ex)
        if why is None:
            if old.gpu.alive():
                return False
            why = 'lost'
        self._recoveries += 1
        if self._recoveries > RECOVER_TRIES:
            self._advice = (' The GPU ran out of memory again after Blackbody started it afresh (the notices say what the '
                            'scene needs): save your work, close other programs that use the GPU and restart the app.'
                            if why == 'memory' else
                            ' The GPU failed again after Blackbody started it afresh: save your work and restart the app.')
            return False
        log.warning('GPU failure (%s): %s; making the engine again', why, ex)
        g = old.gpu
        LE = self.layer_engines
        olds = dict(LE.extra) if LE is not None else {}
        frames = len(old.cache.items)
        try:
            if why == 'lost':
                reason = g.lost_reason()
                g = G.GPU()
                g.keep_plan(old.gpu)   # (a plan cut after running out of memory stays cut: the scene's layout holds)
                g.on_compile = self._compiling
            else:
                plan = g.out_of_memory(ex)
                g.forget_bindings()   # (so what the old engines made is let go)
            engine = Engine(g)
            engine.carry(old)
            engines = LayerEngines(engine, **({'cache_bytes': LE.cache_bytes} if LE is not None else {}))
            for uid, e in olds.items():
                engines.extra[uid] = n = Engine(g, cache_bytes=engines.cache_bytes)
                n.carry(e)
        except Exception as ex2:
            log.exception('the engine could not be made again')
            self._advice = f' The GPU failed and Blackbody could not start it again ({ex2}): restart the app.'
            return False
        self.engine, self.layer_engines = engine, engines
        # let the old engines go now: the exception's traceback holds them (and, on the same device, their memory)
        seen = set()
        while ex is not None and id(ex) not in seen:
            seen.add(id(ex))
            ex.__traceback__ = None
            ex = ex.__cause__ or ex.__context__
        del old, olds, LE
        gc.collect()
        self._collect = True   # (and again once the caller has let go of what it held: a render job, the engines' cycles)
        if why == 'lost':
            said = f': {reason.strip().splitlines()[-1]}' if reason.strip() else ''
            kept = f' and kept the {frames} cached frame{"" if frames == 1 else "s"}' if frames else ''
            self.message.emit(f'The GPU stopped responding (its device was lost{said}). Blackbody started it again{kept}; '
                              'frames not cached are simulated again.')
        else:
            # (a disk cache's frames are never simulated again for it: scene/model.py memory_plan keeps their layout)
            self.message.emit(f'The GPU ran out of memory. Blackbody freed it and now fits the scene in about '
                              f'{plan / G.GB:.1f} GB, keeping what a disk cache holds: the notices say what that changes.')
        self._shown = None
        self._last_cache = None
        return True

    _collect = False
    _advice = ''   # after _recover gave up: to restart the app (' …'), for the caller's message

    def _collected(self):
        """After the engines were made again: collect the old ones before anything is allocated on the GPU."""
        if self._collect:
            self._collect = False
            gc.collect()

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

    def _lume_pending(self):
        """Whether any layer's Lume wants more light paths of the picture it is showing (refine again)."""
        LE = self.layer_engines
        if LE is None:
            return False
        engines = [LE.base] + list(LE.extra.values())
        return any(getattr(getattr(e, '_stage', None), 'lume_pending', False)
                   or getattr(getattr(e, '_lvol', None), 'pending', False) for e in engines)

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
        lu = getattr(getattr(eng, '_stage', None), 'lume', None)
        st['lume'] = (lu.passes, lu.target) if (lu is not None and LU_on(active)) else None
        st['preview'] = (w, h)
        # things that fall or float, where the simulation put them (drawn and picked there in the viewer)
        st['floats'] = eng.floating_overrides(frame) if hasattr(eng, 'floating_overrides') else None
        st['ropes'] = eng.rope_poses(frame) if hasattr(eng, 'rope_poses') else None   # ropes and springs, for drawing
        st['frame'] = frame
        st['load_seq'] = getattr(self.scene, 'load_seq', 0)
        st['layers'] = len(order)
        self.frameReady.emit(img, frame, st)
        frames = eng.cache.frames()
        if frames != self._last_cache:
            self._last_cache = frames
            self.cacheChanged.emit(frames)
        self._recoveries = 0
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
        try:
            ok = simulate(self.layer_engines, sc.layer_order() or [('base', sc)], sc.end, progress=self._progress,
                          cancelled=self._interrupted)
        except Exception as ex:
            if not self._recover(ex):   # (made again: the frames cached so far are kept, and Cache range carries on)
                log.exception('caching failed')
                self.message.emit(f'Caching stopped: {ex}{self._advice}')
                return
            self.post('cache_range')
            return
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
            text = f'{ex}'
            if self._recover(ex):
                text += '\n\nThe GPU failed under it: Blackbody started the engine again. Try exporting the frame again.'
            elif self._advice:
                text += '\n\n' + self._advice.strip()
            self.stillDone.emit(None, text)
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
            if self._recover(ex):
                err = (f'{ex}\n\nThe GPU failed under the render: Blackbody started the engine again, and the frames '
                       'written so far are kept. Render again to carry on.')
            elif self._advice:
                err = f'{ex}\n\n{self._advice.strip()}\n\n{traceback.format_exc()}'
        finally:
            self.engine.invalidate()
        self.jobDone.emit(written, cancelled, err)
