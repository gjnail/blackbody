"""Shared fixtures and the test tiers (see pytest.ini):

- `gpu` is set on every test that uses the engine, so `pytest -m "not gpu"` runs all the others (what CI runs);
- `quick` is set on the tests that need no GPU or took under QUICK_SECONDS when they last ran here (`pytest -m quick`);
- `slow` tests (every preset simulated, the command-line renderer) run only when asked for (`pytest -m slow`).

A test that runs longer than the `timeout` in pytest.ini (time spent compiling shaders aside) stops the run, unless the
run is being debugged."""
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GPU_FIXTURES = {'engine', 'gpu'}   # a test that uses one of these (itself or through another fixture) needs the GPU
QUICK_SECONDS = 10.0                # a test that took longer when it last ran here is left out of the quick tier
COMPILE_SECONDS = 1800.0            # shader compile time a test may take on top of its timeout (a cold march: 8-20 min)
DEBUGGERS = ('bdb', 'pdb', 'pydevd', 'debugpy')   # modules whose trace function means a debugger is attached

_took = {}         # node id -> seconds each test of this run took (its setup and call; not session fixtures)
_shared = {}       # node id -> seconds of its setup spent starting session fixtures (the engine: paid once a run)
_running = None    # node id of the test being run
_debugging = False  # set once a debugger has stopped the run (--pdb at a failure, breakpoint()): no time limits after


@pytest.fixture(scope='session')
def engine():
    try:
        from blackbody.engine.engine import Engine
        return Engine()
    except Exception as ex:
        if os.environ.get('BLACKBODY_ALLOW_NO_GPU') == '1':   # a machine without a GPU (CI): the GPU tests skip
            pytest.skip(f'GPU engine unavailable: {ex}')
        raise RuntimeError('The GPU engine did not start. On a machine without a GPU run pytest -m "not gpu", '
                           'or set BLACKBODY_ALLOW_NO_GPU=1 to skip the GPU tests.') from ex


@pytest.fixture(scope='session', autouse=True)
def _own_cache(tmp_path_factory):
    """Mesh bakes and disk caches go to a folder of this run's own, not the app's: a bake left there by older code
    could pass here and fail on a clean machine."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('BLACKBODY_CACHE', str(tmp_path_factory.mktemp('cache')))
        yield


# -- tiers ------------------------------------------------------------------------------------------------------------

@pytest.hookimpl(tryfirst=True)   # (before -m deselects by these marks)
def pytest_collection_modifyitems(config, items):
    cache = getattr(config, 'cache', None)
    took = cache.get('blackbody/took', {}) if cache is not None else {}
    for item in items:
        gpu = not GPU_FIXTURES.isdisjoint(getattr(item, 'fixturenames', ()))
        if gpu:
            item.add_marker(pytest.mark.gpu)
        t = took.get(item.nodeid)
        if item.get_closest_marker('slow') is None and (not gpu if t is None else t < QUICK_SECONDS):
            item.add_marker(pytest.mark.quick)


class _SessionFixtureClock:
    """Times the session fixtures each test starts. (pytest asks the session's hooks to set these up, and those leave
    out a conftest.py below the root, so this is a plugin of its own.)"""

    @pytest.hookimpl(hookwrapper=True)
    def pytest_fixture_setup(self, fixturedef, request):
        t0 = time.perf_counter()
        yield
        if fixturedef.scope == 'session' and _running is not None:
            _shared[_running] = _shared.get(_running, 0.0) + time.perf_counter() - t0


def pytest_configure(config):
    config.pluginmanager.register(_SessionFixtureClock(), 'blackbody-session-fixture-clock')


def pytest_runtest_logreport(report):
    # (a module's fixture counts against the first test that uses it: run on its own, each test pays for it)
    if report.when in ('setup', 'call'):
        t = report.duration - (_shared.pop(report.nodeid, 0.0) if report.when == 'setup' else 0.0)
        _took[report.nodeid] = round(_took.get(report.nodeid, 0.0) + max(t, 0.0), 2)


def pytest_sessionfinish(session):
    cache = getattr(session.config, 'cache', None)
    if cache is not None and _took:
        took = cache.get('blackbody/took', {})
        took.update(_took)
        cache.set('blackbody/took', took)


# -- timeout ----------------------------------------------------------------------------------------------------------

def pytest_addoption(parser, pluginmanager):
    if not pluginmanager.hasplugin('timeout'):   # (pytest-timeout, if installed, reads the same setting)
        parser.addini('timeout', 'seconds a test may run before the run is stopped (0: no limit)', default='0')


def _limit(item):
    m = item.get_closest_marker('timeout')
    if m is not None and m.args:
        return float(m.args[0])
    try:
        return float(item.config.getini('timeout') or 0)
    except ValueError:   # (this conftest was not loaded at start-up, so the setting is not registered)
        return 0.0


def _compiling(thread_id):
    """True while that thread is compiling a shader (a Kernel) or starting the device (the GPU) in engine/gpu.py; its
    other allocations count as running."""
    g = sys.modules.get('blackbody.engine.gpu')
    if g is None:
        return False
    codes = (g.Kernel.__init__.__code__, g.GPU.__init__.__code__)
    f = sys._current_frames().get(thread_id)
    while f is not None:
        if f.f_code in codes:
            return True
        f = f.f_back
    return False


def pytest_enter_pdb(config, pdb):
    global _debugging
    _debugging = True


def _debugged(config):
    """True when the run is being debugged: --pdb or --trace, a debugger already entered, or an IDE's attached (a trace
    function of its own, or the debugger's slot in sys.monitoring; coverage's trace function does not count)."""
    if _debugging or config.getoption('usepdb', False) or config.getoption('trace', False):
        return True
    mon = getattr(sys, 'monitoring', None)
    if mon is not None and mon.get_tool(mon.DEBUGGER_ID) is not None:
        return True
    f = sys.gettrace()
    if f is None:
        return False
    module = getattr(f, '__module__', None) or type(f).__module__ or ''
    return any(part.startswith(DEBUGGERS) for part in module.split('.'))


def _stack(frame):
    """A thread's stack from below pytest's own frames (the test, its fixtures and what they call)."""
    lines = traceback.format_stack(frame)
    inner = [i for i, ln in enumerate(lines) if f'{os.sep}_pytest{os.sep}' in ln or f'{os.sep}pluggy{os.sep}' in ln]
    return ''.join(lines[inner[-1] + 1 if inner else 0:]).rstrip()


def _time_out(item, limit):
    """Say which test hung and where every thread is, then end the run: a call stuck in the GPU driver cannot be
    interrupted (this is what pytest-timeout's thread method does)."""
    out = err = ''
    capman = item.config.pluginmanager.getplugin('capturemanager')
    try:
        if capman is not None:
            capman.suspend_global_capture(in_=True)
            out, err = capman.read_global_capture()
    except Exception:
        pass
    names = {t.ident: t.name for t in threading.enumerate()}
    lines = [f'Timeout: {item.nodeid} ran for more than {limit:g} s (shader compiles aside)']
    for title, text in (('captured stdout', out), ('captured stderr', err)):
        if text:
            lines += [f'--- {title} ---', text.rstrip()]
    for tid, frame in sys._current_frames().items():
        if tid != threading.get_ident():
            lines += [f'--- thread {names.get(tid, tid)} ---', _stack(frame)]
    try:
        tw = item.config.get_terminal_writer()
        tw.line()
        tw.sep('+', 'Timeout')
        for ln in lines:
            tw.line(ln)
        tw.flush()
    except Exception:
        sys.__stderr__.write('\n'.join(lines) + '\n')
    sys.__stderr__.flush()
    os._exit(1)


class _Watchdog(threading.Thread):
    """Ends the run when its test runs past the limit (time spent compiling shaders aside, up to COMPILE_SECONDS)."""

    def __init__(self, item, limit):
        super().__init__(name='test-timeout', daemon=True)
        self.item, self.limit = item, limit
        self.main = threading.get_ident()
        self.done = threading.Event()

    def run(self):
        ran = compiled = 0.0
        last = time.monotonic()
        while not self.done.wait(0.5):
            if _debugging:   # (stopped at a breakpoint)
                return
            now = time.monotonic()
            if compiled < COMPILE_SECONDS and _compiling(self.main):
                compiled += now - last
            else:
                ran += now - last
            last = now
            if ran > self.limit:
                _time_out(self.item, self.limit)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    global _running
    _running = item.nodeid
    limit = 0.0 if item.config.pluginmanager.hasplugin('timeout') or _debugged(item.config) else _limit(item)
    dog = _Watchdog(item, limit) if limit > 0 else None
    if dog is not None:
        dog.start()
    yield
    if dog is not None:
        dog.done.set()
