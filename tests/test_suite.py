"""The test suite's own machinery (conftest.py, pytest.ini) and CI's command-line check. GPU-free: the checks run a
small pytest of their own in a folder with copies of conftest.py and pytest.ini."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from blackbody.scene import presets

ROOT = Path(__file__).resolve().parents[1]

INNER = '''
import pytest


@pytest.fixture
def gpu(engine):
    return engine.gpu


def test_free():
    pass


def test_uses_the_engine(engine):
    pass


def test_uses_it_through_another_fixture(gpu):
    pass


@pytest.mark.slow
def test_long_by_design():
    pass
'''


def _suite(tmp_path, files):
    (tmp_path / 'tests').mkdir()
    shutil.copy(ROOT / 'tests' / 'conftest.py', tmp_path / 'tests' / 'conftest.py')
    shutil.copy(ROOT / 'pytest.ini', tmp_path / 'pytest.ini')
    for name, text in files.items():
        (tmp_path / 'tests' / name).write_text(text)
    return tmp_path


def _pytest(where, *args, cache=False, env=None):
    e = {k: v for k, v in os.environ.items() if k != 'BLACKBODY_ALLOW_NO_GPU'}
    e['PYTHONPATH'] = str(ROOT)
    e.update(env or {})
    # (-p no:timeout: with pytest-timeout installed conftest.py leaves time limits to it, and these check conftest's own)
    cmd = [sys.executable, '-m', 'pytest', '-q', '-rs', '-p', 'no:timeout', *([] if cache else ['-p', 'no:cacheprovider']),
           *args]
    return subprocess.run(cmd, cwd=str(where), capture_output=True, text=True, env=e, timeout=300)


def _collected(r):
    return sorted(ln.split('::')[1] for ln in r.stdout.splitlines() if '::' in ln)


def test_every_preset_is_registered_once_and_has_its_picture():
    # (the pictures are the anchor: ORDER and PRESETS are filled together, category by category, so a category left
    # out shrinks both)
    pictures = {p.stem for p in (ROOT / 'blackbody' / 'assets' / 'presets').glob('*.png')}
    twice = sorted({n for n in presets.ORDER if presets.ORDER.count(n) > 1})
    assert not twice, f'listed twice in presets.ORDER: {twice}'
    assert set(presets.PRESETS) == set(presets.ORDER)
    assert not pictures - set(presets.ORDER), f'pictures of presets that are not registered: {sorted(pictures - set(presets.ORDER))}'
    assert not set(presets.ORDER) - pictures, (f'presets without a picture (tools/make_thumbnails.py): '
                                               f'{sorted(set(presets.ORDER) - pictures)}')


def test_cli_lists_every_preset():
    r = subprocess.run([sys.executable, '-m', 'blackbody', 'presets'], cwd=str(ROOT), capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert [ln.split()[0] for ln in r.stdout.splitlines() if ln.strip()] == list(presets.ORDER)


def test_tests_that_use_the_engine_are_marked_gpu_and_slow_ones_run_only_when_asked(tmp_path):
    s = _suite(tmp_path, {'test_inner.py': INNER})
    assert _collected(_pytest(s, '--collect-only', '-m', 'gpu')) == ['test_uses_it_through_another_fixture',
                                                                    'test_uses_the_engine']
    assert _collected(_pytest(s, '--collect-only', '-m', 'not gpu')) == ['test_free', 'test_long_by_design']
    assert _collected(_pytest(s, '--collect-only')) == ['test_free', 'test_uses_it_through_another_fixture',
                                                       'test_uses_the_engine']
    assert _collected(_pytest(s, '--collect-only', '-m', 'slow')) == ['test_long_by_design']


def test_the_quick_tier_follows_how_long_each_test_last_took(tmp_path):
    s = _suite(tmp_path, {'test_inner.py': INNER + '''

def test_free_but_long():
    pass
'''})
    # untimed, a test is quick if it needs no GPU
    assert _collected(_pytest(s, '--collect-only', '-m', 'quick', cache=True)) == ['test_free', 'test_free_but_long']
    took = s / '.pytest_cache' / 'v' / 'blackbody' / 'took'
    took.parent.mkdir(parents=True, exist_ok=True)
    took.write_text(json.dumps({'tests/test_inner.py::test_uses_the_engine': 2.0,
                                'tests/test_inner.py::test_free_but_long': 25.0}))
    assert _collected(_pytest(s, '--collect-only', '-m', 'quick', cache=True)) == ['test_free', 'test_uses_the_engine']
    # a run records how long its tests took, their own fixtures included but not the ones made once a run
    (s / 'tests' / 'test_fixtures.py').write_text('''
import time

import pytest


@pytest.fixture(scope='module')
def slow_to_make():
    time.sleep(0.5)


@pytest.fixture(scope='session')
def made_once_a_run():
    time.sleep(0.5)


def test_with_a_slow_fixture(slow_to_make):
    pass


def test_with_one_made_once_a_run(made_once_a_run):
    pass
''')
    r = _pytest(s, '-m', 'not gpu', cache=True)
    assert r.returncode == 0, r.stdout[-2000:]
    t = json.loads(took.read_text())
    assert t['tests/test_inner.py::test_free_but_long'] < 10.0 and 'tests/test_inner.py::test_free' in t
    assert t['tests/test_inner.py::test_uses_the_engine'] == 2.0, 'tests that did not run keep their time'
    assert t['tests/test_fixtures.py::test_with_a_slow_fixture'] >= 0.45
    assert t['tests/test_fixtures.py::test_with_one_made_once_a_run'] < 0.25


def test_an_engine_that_cannot_start_fails_the_gpu_tests(tmp_path):
    s = _suite(tmp_path, {'test_inner.py': '''
import sys
import types

broken = types.ModuleType('blackbody.engine.engine')


class Engine:
    def __init__(self):
        raise RuntimeError('device creation failed')


broken.Engine = Engine
sys.modules['blackbody.engine.engine'] = broken


def test_uses_the_engine(engine):
    pass
'''})
    r = _pytest(s)
    assert r.returncode == 1 and '1 error' in r.stdout, r.stdout[-2000:]
    assert 'The GPU engine did not start' in r.stdout and 'device creation failed' in r.stdout
    r = _pytest(s, env={'BLACKBODY_ALLOW_NO_GPU': '1'})
    assert r.returncode == 0 and '1 skipped' in r.stdout, r.stdout[-2000:]
    assert 'device creation failed' in r.stdout


def test_a_hung_test_stops_the_run_but_a_shader_compile_does_not_count(tmp_path):
    s = _suite(tmp_path, {'test_inner.py': '''
import time
from pathlib import Path

import pytest

from blackbody.engine import gpu as G


@pytest.mark.timeout(1)
def test_a_long_compile(monkeypatch):
    monkeypatch.setattr(G, 'load_wgsl', lambda *a: time.sleep(3) or '')

    class Device:
        def create_shader_module(self, **k):
            raise ValueError('compiled')

    with pytest.raises(RuntimeError, match='compiled'):
        G.Kernel(type('FakeGPU', (), {'device': Device()})(), 'x.wgsl', [])
    Path(__file__).with_name('compiled.txt').write_text('')


@pytest.mark.timeout(1)
def test_hangs():
    class Device:
        def create_buffer(self, **k):
            time.sleep(120)

    G.Buffer(type('FakeGPU', (), {'device': Device()})(), 16)   # (stuck in an allocation in gpu.py: no compile)


def test_never_reached():
    pass
'''})
    t0 = time.monotonic()
    r = _pytest(s)
    assert time.monotonic() - t0 < 60
    out = r.stdout + r.stderr
    assert r.returncode == 1, out[-2000:]
    assert (s / 'tests' / 'compiled.txt').exists(), 'a 3 s shader compile is not held against a 1 s limit'
    assert 'Timeout: tests/test_inner.py::test_hangs ran for more than 1 s' in out, out[-2000:]
    assert 'time.sleep(120)' in out, 'the stack of the hung test'
    assert 'passed' not in out and 'failed' not in out, 'the run ended there'


def test_a_run_being_debugged_has_no_time_limit(tmp_path):
    s = _suite(tmp_path, {'test_inner.py': '''
import time

import pytest


@pytest.mark.timeout(1)
def test_long():
    time.sleep(2)
'''})
    r = _pytest(s, '--pdb')
    assert r.returncode == 0 and '1 passed' in r.stdout, r.stdout[-2000:]
    r = _pytest(s)
    assert r.returncode == 1 and 'Timeout: tests/test_inner.py::test_long' in r.stdout + r.stderr, r.stdout[-2000:]
    # stopped at a breakpoint() part way through
    (s / 'tests' / 'test_inner.py').write_text('''
import time

import pytest


@pytest.mark.timeout(1)
def test_long(request):
    request.config.hook.pytest_enter_pdb(config=request.config, pdb=None)   # (what breakpoint() calls, less its prompt)
    time.sleep(2)
''')
    r = _pytest(s)
    assert r.returncode == 0 and '1 passed' in r.stdout, r.stdout[-2000:]


def test_the_tests_bake_meshes_into_a_cache_of_their_own(tmp_path_factory):
    from blackbody.engine.mesh import cache_dir
    from blackbody.io.simcache import default_root
    run = tmp_path_factory.getbasetemp().resolve()
    assert run in cache_dir().resolve().parents
    assert run in default_root().resolve().parents
