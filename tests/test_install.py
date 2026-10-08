"""A clean install works: requirements.txt and pyproject.toml declare the same packages (USD and OpenColorIO
among them), the Python range the docs state is the one the launchers accept, and the launchers install again
after a failed or interrupted install instead of starting a half-installed environment. GPU-free."""
import os
import re
import shutil
import subprocess
import sys
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
CHECK = re.compile(r'sys\.exit\(not \(\((\d+), (\d+)\) <= sys\.version_info\[:2\] <= \((\d+), (\d+)\)')


def _pins():
    lines = (ROOT / 'requirements.txt').read_text(encoding='utf-8').splitlines()
    reqs = [Requirement(s) for s in (ln.strip() for ln in lines) if s and not s.startswith('#')]
    return {canonicalize_name(r.name): r for r in reqs}


def _project():
    return tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']


def _launcher_range():
    found = {tuple(map(int, CHECK.search((ROOT / f).read_text(encoding='utf-8')).groups()))
             for f in ('Blackbody.bat', 'blackbody.sh')}
    assert len(found) == 1, f'the launchers accept different Pythons: {found}'
    lo0, lo1, hi0, hi1 = found.pop()
    return (lo0, lo1), (hi0, hi1)


def test_requirements_and_pyproject_agree():
    pins = _pins()
    deps = {canonicalize_name(r.name): r for r in map(Requirement, _project()['dependencies'])}
    assert {'usd-core', 'opencolorio'} <= set(pins), 'USD and OCIO features need these installed'
    assert set(pins) == set(deps), f'only in requirements.txt: {set(pins) - set(deps)}, ' \
                                   f'only in pyproject.toml: {set(deps) - set(pins)}'
    for name, pin in pins.items():
        (spec,) = pin.specifier
        assert spec.operator == '==', f'{name} is not pinned in requirements.txt'
        assert deps[name].specifier.contains(spec.version), f'{name} {spec.version} is outside pyproject\'s {deps[name].specifier}'
        assert str(pin.marker or '') == str(deps[name].marker or ''), f'{name} is installed on different platforms'


def test_python_range_is_stated_the_same_everywhere():
    lo, hi = _launcher_range()
    assert _project()['requires-python'] == f'>={lo[0]}.{lo[1]}'
    stated = f'Python {lo[0]}.{lo[1]} or {hi[0]}.{hi[1]}'
    for doc in ('README.md', 'CONTRIBUTING.md', 'docs/getting-started.md', 'site/index.html'):
        text = (ROOT / doc).read_text(encoding='utf-8')
        assert stated in text, f'{doc} does not say "{stated}"'
        older = [m for m in re.findall(r'Python (3\.\d+)\+?', text) if tuple(map(int, m.split('.'))) < lo]
        assert not older, f'{doc} still advertises Python {older}'


def test_windows_launcher_loads_every_package_after_installing():
    """The app runs under pythonw (no console), so the launcher imports each package once after installing,
    where an error can still be shown."""
    bat = (ROOT / 'Blackbody.bat').read_text(encoding='utf-8')
    imported = {m.strip().split('.')[0] for m in re.search(r'-c "import ([^"]+,[^"]+)"', bat)[1].split(',')}
    modules = {}
    for mod, dists in packages_distributions().items():
        for d in dists:
            modules.setdefault(canonicalize_name(d), set()).add(mod)
    windows = {'platform_machine': 'AMD64', 'sys_platform': 'win32', 'platform_system': 'Windows', 'os_name': 'nt'}
    for name, pin in _pins().items():
        if name in modules and (pin.marker is None or pin.marker.evaluate(windows)):
            assert modules[name] & imported, f'Blackbody.bat does not check that {name} loads'


# --- the launchers themselves, on a copy with stand-ins for Python and the app -----------------------------

def _bash():
    """Git's bash on Windows (System32's bash.exe starts WSL, another machine), else the one on the PATH."""
    if os.name != 'nt':
        return shutil.which('bash')
    git = shutil.which('git')
    cands = [p / 'bin' / 'bash.exe' for p in Path(git).parents[:3]] if git else []
    cands.append(Path(shutil.which('bash') or 'none'))
    windir = os.environ.get('SystemRoot', r'C:\Windows').lower()
    return next((str(c) for c in cands if c.is_file() and not str(c).lower().startswith(windir)), None)


FAKE_PYTHON = r'''#!/usr/bin/env bash
# stand-in for python: logs each call; FAKE_BAD=1 makes it the wrong version, FAKE_PIP_FAIL=1 makes pip fail
echo "$(basename "$0") $*" >> "$FAKE_LOG"
case "$1" in
  -c) [ "${FAKE_BAD:-0}" = 0 ] ;;
  --version) echo "Python 3.9.0" ;;
  -m) case "$2" in
        venv) d="${@: -1}"; mkdir -p "$d/bin"; cp "$0" "$d/bin/python"; chmod +x "$d/bin/python" ;;
        pip) [ "${FAKE_PIP_FAIL:-0}" = 0 ] ;;
        blackbody) shift 2; echo "blackbody ran: $*" ;;
      esac ;;
esac
'''


def test_shell_launcher_retries_a_failed_install(tmp_path):
    bash = _bash()
    if not bash:
        pytest.skip('no bash')
    shutil.copy(ROOT / 'blackbody.sh', tmp_path)
    (tmp_path / 'requirements.txt').write_text('numpy==1.0\n')
    fake = tmp_path / 'fakebin'
    fake.mkdir()
    for name in ('python3.13', 'python3.12', 'python3'):
        (fake / name).write_bytes(FAKE_PYTHON.encode())
        (fake / name).chmod(0o755)
    log = tmp_path / 'calls.log'

    def launch(**env):
        log.write_text('')
        e = dict(os.environ, FAKE_LOG=log.as_posix(), PATH=str(fake) + os.pathsep + os.environ['PATH'], **env)
        r = subprocess.run([bash, (tmp_path / 'blackbody.sh').as_posix(), '--version', 'a b'], cwd=tmp_path, env=e,
                           stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
        return r, log.read_text().splitlines()

    marker = tmp_path / '.venv' / '.installed'
    r, calls = launch(FAKE_PIP_FAIL='1')
    assert r.returncode != 0 and 'tries again' in r.stderr
    assert 'python3.13 -m venv .venv' in calls and not marker.exists()
    assert not any('blackbody' in c for c in calls)

    r, calls = launch()                                  # the venv is reused, the install retried
    assert r.returncode == 0, r.stderr
    assert 'blackbody ran: --version a b' in r.stdout
    assert not any('venv' in c for c in calls) and 'python -m pip install -r requirements.txt' in calls
    assert marker.read_bytes() == (tmp_path / 'requirements.txt').read_bytes()

    r, calls = launch()                                  # installed: straight to the app
    assert r.returncode == 0 and calls == ['python -m blackbody --version a b']

    (tmp_path / 'requirements.txt').write_text('numpy==1.0\nusd-core==2.0\n')
    r, calls = launch()                                  # requirements changed: installs again
    assert r.returncode == 0 and 'python -m pip install -r requirements.txt' in calls

    (tmp_path / 'requirements.txt').write_text('numpy==2.0\n')
    r, calls = launch(FAKE_BAD='1')                      # only the wrong Python version around
    assert r.returncode != 0 and 'Python 3.12 or 3.13' in r.stderr and 'Python 3.9.0' in r.stderr
    assert not (tmp_path / '.venv').exists() and not any('pip' in c for c in calls)


@pytest.mark.skipif(os.name != 'nt', reason='Windows launcher')
def test_windows_launcher_retries_a_failed_install(tmp_path):
    lo, hi = _launcher_range()
    if not lo <= sys.version_info[:2] <= hi:
        pytest.skip('this Python is outside the range the launcher accepts')
    shutil.copy(ROOT / 'Blackbody.bat', tmp_path)
    app = tmp_path / 'blackbody'
    app.mkdir()
    (app / '__init__.py').write_text('')
    (app / '__main__.py').write_text('import sys\nprint("blackbody ran:", *sys.argv[1:])\n'
                                     'sys.exit(3 if "fail" in sys.argv else 0)\n')
    # a venv without pip that borrows this interpreter's packages, so pip can run offline and change nothing
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(tmp_path / '.venv')], check=True)
    sites = [p for p in sys.path if p.endswith('site-packages') and os.path.isdir(p)]
    (tmp_path / '.venv' / 'Lib' / 'site-packages' / 'borrowed.pth').write_text('\n'.join(sites) + '\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith('PIP_') and k != 'PYTHONPATH'}
    env.update(PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1')

    def launch(*args):
        return subprocess.run(['cmd', '/d', '/c', str(tmp_path / 'Blackbody.bat'), *args], env=env,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300)

    marker = tmp_path / '.venv' / '.installed'
    (tmp_path / 'requirements.txt').write_text('no-such-package-for-blackbody-tests==1.0\n')
    r = launch('render', 'x')
    assert r.returncode != 0 and 'Installing the dependencies failed' in r.stdout and 'Press any key' in r.stdout
    assert 'blackbody ran' not in r.stdout and not marker.exists()

    (tmp_path / 'requirements.txt').write_text('pip\n')
    marker.write_text('pip==1.0\n')                     # from an install before requirements.txt changed
    (tmp_path / 'mujoco.py').write_text('raise ImportError("DLL load failed while importing _mujoco")\n')
    r = launch('render', 'x')                            # installed, but a package cannot load
    assert r.returncode != 0 and 'cannot load them' in r.stdout and 'DLL load failed' in r.stderr
    assert not marker.exists()

    (tmp_path / 'mujoco.py').unlink()
    r = launch('render', 'x')
    assert r.returncode == 0 and 'blackbody ran: render x' in r.stdout and 'Setting up' in r.stdout
    assert marker.read_bytes() == (tmp_path / 'requirements.txt').read_bytes()

    r = launch('--version')                              # installed: straight to the app, in this console
    assert r.returncode == 0 and r.stdout.strip() == 'blackbody ran: --version'
    assert launch('render', 'fail').returncode == 3      # a render's exit code reaches the farm
