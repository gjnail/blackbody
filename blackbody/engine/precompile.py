"""Compiling the slow GPU kernels ahead of time, so the first liquid scene does not wait for them.

The graphics driver compiles each kernel once and keeps the result on disk, by the exact shader: after that it takes
a fraction of a second. The first time is cold, and one kernel is slow: the liquid renderer's march (liq_march.wgsl),
which every liquid, sea, lava and ice scene needs (the other liquid kernels together take seconds). It is cold after
an install, after an update that changes it or one of its includes, after a driver update, and when the driver has
dropped it from a full cache.

The app starts this as it opens (start_background). It compiles the built-in kernels (the march) every time: whether
the driver still has them only the driver knows, and when it does they take a moment. Then it compiles any kernel this
machine compiled before that took SLOW seconds or more, has used within RECENT and has changed since (an update), as
the record of compiled kernels (gpu.ShaderRecord) tells; it reads the record only once the march is done, so reading
it holds nothing up. It runs in a thread of the app's own process, on the app's own GPU device: a device does not see
what another process compiled after it was made (measured with NVIDIA's Vulkan driver), and wgpu takes a device's calls
from several threads (measured: the app's work on the device went on during a cold compile of the march, with a
hitch or two of up to about 0.4 s, as on a device of its own). A scene that needs a kernel the precompile is compiling
at that moment waits for it and takes the pipeline it made (gpu.Kernel), so nothing is compiled twice, whatever the
driver keeps. The compile runs in the driver, outside Python's lock, so the app runs on meanwhile; it takes one core.
`blackbody precompile` does the same in the foreground, for installers and render farms: processes started after it
find the kernels in the driver's cache.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time

log = logging.getLogger('blackbody.precompile')

SLOW = 2.0                  # s: a kernel whose first compile here took this long is compiled again ahead of time when
                            # it changes
RECENT = 60 * 24 * 3600.0   # s: ... if this machine has used it within this long (older rows are removed)


def builtin():
    """Kernels compiled ahead of time on every machine, each time the app opens: the liquid renderer's march."""
    from .liquid_render import march_spec
    return [march_spec()]


SPEC_KEYS = ('file', 'entry', 'bindings', 'defines', 'workgroup')   # (gpu.kernel_spec)


def _identity(spec):
    return json.dumps([spec[k] for k in SPEC_KEYS])   # (the defines in their order: another order is another shader)


def current_key(spec):
    """The shader key a kernel spec (gpu.kernel_spec) has with the shaders as they are now; None if its source is gone
    (a kernel of an older version) or the spec is not one."""
    from .gpu import load_wgsl, shader_key
    try:
        code = load_wgsl(spec['file'], spec.get('defines') or None)
        return shader_key(code, spec.get('entry', 'main'), spec['bindings'], spec['workgroup'])
    except (OSError, KeyError, TypeError, ValueError, AttributeError):
        return None


def _prune_others(record, since):
    """Remove the records of the other adapters and drivers (folders beside this one) that no kernel has been used on
    since `since`: a driver updated since then, its compiled code gone with it."""
    from .gpu import ShaderRecord
    try:
        others = [f.path for f in os.scandir(record.folder.parent) if f.is_dir() and f.name != record.folder.name]
    except OSError:
        return
    for path in others:
        old = ShaderRecord(path)
        if old.last_used() < since:
            old.entries(since)   # (removes its files)
            try:
                os.rmdir(path)
            except OSError:   # (something else in it)
                pass


def changed(record, now=None):
    """The slow kernels this machine has compiled and used lately whose shaders have changed since (an update), as
    specs, slowest first, the built-in ones aside: rows of SLOW seconds or more, used within RECENT, whose shader as it is
    now this adapter and driver have not compiled. Rows not used within RECENT are removed on the way, and so are other
    adapters' records not used within it. Each spec carries the keys of the rows it came from ('rows'), for
    compile_specs to forget them if it no longer builds."""
    now = time.time() if now is None else now
    since = now - RECENT
    _prune_others(record, since)
    skip = {_identity(s) for s in builtin()}
    found = {}   # identity -> (seconds its first compile took, spec, the rows' keys)
    for key, row, used in record.entries(since):
        if not isinstance(row, dict) or any(k not in row for k in SPEC_KEYS):
            continue   # (not a kernel's row)
        try:
            secs = float(row.get('seconds') or 0.0)
        except (TypeError, ValueError):
            continue
        if secs < SLOW:
            continue
        ident = _identity(row)
        if ident in skip:
            continue
        keys = found[ident][2] if ident in found else []
        keys.append(key)
        if ident not in found or found[ident][0] < secs:
            found[ident] = (secs, {k: row[k] for k in SPEC_KEYS}, keys)
    todo = []
    for secs, spec, keys in found.values():
        spec['rows'] = keys
        key = current_key(spec)
        if key is not None and not record.has(key):
            todo.append((secs, spec))
    todo.sort(key=lambda r: -r[0])
    return [spec for _, spec in todo]


def pending(record, now=None):
    """Everything to compile ahead of time, as specs: the built-in kernels, then the slow ones that have changed
    (changed; none without a record)."""
    return builtin() + (changed(record, now) if record is not None else [])


def compile_specs(gpu, specs, say=None):
    """Compile these kernels (gpu.kernel, quietly: each recorded and kept for the app, and one another thread is
    compiling waited for). say(i, n, spec) before each and say(i, n, spec, seconds) after. Returns how many compiled."""
    done = 0
    for i, spec in enumerate(specs):
        if say is not None:
            say(i, len(specs), spec)
        try:
            k = gpu.kernel(spec['file'], spec['bindings'], spec.get('entry', 'main'), spec.get('defines') or None,
                           tuple(spec['workgroup']), quiet=True)
        except Exception as ex:   # (a kernel of an older version whose resources no longer match its shader)
            log.warning('Could not precompile %s: %s', spec.get('file'), str(ex).splitlines()[0] if str(ex) else ex)
            record = getattr(gpu, 'record', None)
            if record is not None:   # (its rows would list it again at every start, and fail again)
                for key in spec.get('rows', ()):
                    record.forget(key)
            continue
        done += 1
        if say is not None:
            say(i, len(specs), spec, getattr(k, 'seconds', 0.0))
    return done


class Background(threading.Thread):
    """The precompile in a thread of the app, on the app's GPU (its kernels kept there for the app to take): the
    built-in kernels, then the slow ones that have changed (read from the record once those are done). stop() keeps it
    from starting another (one under way finishes in the driver)."""

    def __init__(self, gpu):
        super().__init__(name='blackbody-precompile', daemon=True)
        self.gpu = gpu
        self.todo = builtin()   # (what it has found to compile; the changed ones are added once these are done)
        self.done = 0
        self.error = None
        self._cancel = threading.Event()   # (not _stop: Thread has one)

    def run(self):
        try:
            self._compile(self.gpu, self.todo)
            record = getattr(self.gpu, 'record', None)
            if record is not None and not self._cancel.is_set():
                more = changed(record)
                self.todo = self.todo + more
                self._compile(self.gpu, more)
            log.info('Precompiled %d of %d GPU kernels', self.done, len(self.todo))
        except Exception as ex:   # (never in the way of the app: a scene compiles what it needs itself)
            self.error = ex
            log.info('Background shader compile stopped: %s', ex)

    def _compile(self, g, specs):
        for spec in specs:
            if self._cancel.is_set():
                break
            self.done += compile_specs(g, [spec])

    def stop(self):
        self._cancel.set()


def start_background(gpu):
    """Start the precompile in a thread of its own; the thread, or None when it is turned off (BLACKBODY_PRECOMPILE=0).
    The app calls this as it opens, and stop() as it closes."""
    if os.environ.get('BLACKBODY_PRECOMPILE', '1') == '0':
        return None
    try:
        job = Background(gpu)
        job.start()
    except Exception as ex:   # (never in the way of the app starting: a scene compiles what it needs itself)
        log.info('No background shader compile: %s', ex)
        return None
    return job


def stop(job):
    """Stop a background compile start_background started (the app is closing): it starts no other kernel, and what it
    finished stays compiled."""
    if job is not None:
        job.stop()


def main(quiet=False):
    """`blackbody precompile`: compile now, in the foreground, what the app compiles in the background."""
    from .gpu import GPU
    gpu = GPU()
    todo = pending(gpu.record)
    say = None
    if not quiet:
        print(f'{gpu.name} ({gpu.backend}): {len(todo)} kernel{"" if len(todo) == 1 else "s"} to compile (a moment each '
              f'if the driver has it already)', flush=True)

        def say(i, n, spec, seconds=None):
            name = os.path.basename(str(spec['file']))
            if seconds is None:
                print(f'[{i + 1}/{n}] {name} ...', end=' ', flush=True)
            else:
                print(f'{seconds:.1f} s', flush=True)
    t0 = time.perf_counter()
    done = compile_specs(gpu, todo, say)
    if not quiet:
        print(f'Compiled {done} of {len(todo)} in {time.perf_counter() - t0:.1f} s. The driver keeps them: scenes that '
              f'use them start without compiling.', flush=True)
    return 0 if done == len(todo) else 1
