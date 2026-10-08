"""The same scene simulates the same way in another process: a small scene of each solver family (tests/determinism_run.py)
simulated in two processes at once, each loading the GPU while the other runs, and their states compared exactly. This
sees what varies with the GPU's scheduling from run to run (the order its threads happen to run in), which a render farm
meets and a test running a scene twice in one process may not. It cannot see what one run leaves in the engine for the
next (memos, buffers not cleared): both processes run the families in the same order, so both are left the same; that
needs a fresh run compared with a repeat in one process. About a minute (two engines start and compile their shaders)."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import determinism_run  # noqa: E402

RUNNER = Path(determinism_run.__file__)

# families that still vary from run to run, and why (each passes once its cause is fixed: then take it off this list)
VARIES = {
    'lava_sea': 'liquid kernels draw their random numbers from the particle\'s slot (both_evap.wgsl, liq_therm_g2p.wgsl, '
                'liq_steam.wgsl, liq_g2p.wgsl), and slots are handed out in the order the threads run: once the lava '
                'boils the sea, which drops boil away differs',
}


def _wait_for_all(procs, timeout):
    """Each process's (stdout, stderr), all within `timeout` seconds in all (not each: the test's own limit must not
    end the run before the kill); any still running when one hangs past it (or the wait fails) is killed, not left
    holding the GPU."""
    end = time.monotonic() + timeout
    try:
        return [p.communicate(timeout=max(1.0, end - time.monotonic())) for p in procs]
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
                p.communicate()


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    """Each family's fingerprint from two processes simulating at the same time."""
    env = dict(os.environ, BLACKBODY_CACHE=str(tmp_path_factory.mktemp('determinism-cache')))
    procs = [subprocess.Popen([sys.executable, str(RUNNER)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              env=env, cwd=str(RUNNER.parents[1])) for _ in range(2)]
    out = []
    for p, (stdout, stderr) in zip(procs, _wait_for_all(procs, timeout=850)):
        if p.returncode != 0 and 'GPU' in stderr and os.environ.get('BLACKBODY_ALLOW_NO_GPU') == '1':
            pytest.skip('GPU engine unavailable')
        assert p.returncode == 0, stderr[-4000:]
        out.append(json.loads(stdout.strip().splitlines()[-1]))
    return out


@pytest.mark.gpu
@pytest.mark.timeout(900)
@pytest.mark.parametrize('family', [pytest.param(f, marks=pytest.mark.xfail(reason=VARIES[f], strict=False))
                                    if f in VARIES else f for f in determinism_run.FAMILIES])
def test_a_scene_simulates_the_same_in_another_process(runs, family):
    a, b = runs[0][family], runs[1][family]
    differ = sorted(k for k in a if a[k] != b.get(k))
    assert not differ, f'{family}: {", ".join(differ)} differ between two runs of {determinism_run.FAMILIES[family][0]}'


def test_every_family_says_what_it_reads():
    """(GPU-free) The runner's families: each a preset that exists, read through parts it knows."""
    from blackbody.scene import presets
    known = {'gas', 'liquid', 'lava', 'fabric', 'weather', 'objects'}
    for fam, (name, res, frames, parts, change) in determinism_run.FAMILIES.items():
        assert name in presets.ORDER, fam
        assert set(parts) <= known and res >= 16 and frames > 0, fam
    assert set(VARIES) <= set(determinism_run.FAMILIES)
    for need in ('fire', 'liquid', 'cloth', 'weather', 'fire_water', 'lava'):
        assert need in determinism_run.FAMILIES and need not in VARIES, need


def test_a_run_that_hangs_is_not_left_running():
    """(GPU-free) A run that hangs: the wait gives up at its time limit, and neither it nor the one after it (never waited
    for) is left running on the GPU."""
    import time
    procs = [subprocess.Popen([sys.executable, '-c', f'import time; time.sleep({s})']) for s in (0.1, 60, 60)]
    t0 = time.perf_counter()
    with pytest.raises(subprocess.TimeoutExpired):
        _wait_for_all(procs, timeout=2)
    assert time.perf_counter() - t0 < 30
    assert all(p.poll() is not None for p in procs)
