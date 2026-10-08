"""Shader compiles: the record of what this adapter and driver have compiled (gpu.ShaderRecord), a kernel another
thread is compiling waited for rather than compiled twice (gpu.Kernel), the background precompile of the slow kernels
(engine/precompile.py) and the liquid march kept small enough to compile in a reasonable time. GPU-free but for one."""
import json
import os
import re
import threading
import time
import types
from functools import lru_cache

import pytest

from blackbody.engine import gpu as G
from blackbody.engine import precompile


def test_a_kernels_key_follows_everything_the_driver_compiles():
    k = G.shader_key('fn main() {}', 'main', ['buf'], (8, 8, 1))
    assert k == G.shader_key('fn main() {}', 'main', ['buf'], [8, 8, 1])
    assert len({k, G.shader_key('fn main() { }', 'main', ['buf'], (8, 8, 1)),
                G.shader_key('fn main() {}', 'other', ['buf'], (8, 8, 1)),
                G.shader_key('fn main() {}', 'main', ['rbuf'], (8, 8, 1)),
                G.shader_key('fn main() {}', 'main', ['buf'], (4, 4, 4))}) == 5


def test_an_adapter_key_changes_with_the_driver():
    a = {'vendor': 'NVIDIA', 'device': 'RTX 3090', 'description': '616.64', 'backend_type': 'Vulkan'}
    assert G.adapter_key(a) == G.adapter_key(dict(a))
    assert G.adapter_key(a) != G.adapter_key(dict(a, description='617.10'))
    assert G.adapter_key(a) != G.adapter_key(dict(a, backend_type='D3D12'))


def test_the_record_keeps_the_first_compile_of_each_kernel(tmp_path):
    rec = G.ShaderRecord(tmp_path / 'adapter')
    spec = G.kernel_spec('x.wgsl', ['buf'], defines={'B': 2, 'A': 'rgba16float'}, workgroup=(64, 1, 1))
    assert not rec.has('k1') and rec.get('k1') is None and rec.entries() == []
    rec.note('k1', spec, 95.25)
    rec.note('k1', spec, 0.2)   # (the second compile is the driver's cache: not a cold time)
    row = rec.get('k1')
    assert row['seconds'] == 95.25 and row['file'] == 'x.wgsl' and row['workgroup'] == [64, 1, 1]
    assert list(row['defines']) == ['B', 'A']   # (in the order load_wgsl writes them)
    assert [k for k, *_ in rec.entries()] == ['k1']
    assert not list(tmp_path.rglob('*.tmp'))


def test_rows_not_used_lately_are_removed_rather_than_read(tmp_path, monkeypatch):
    rec = G.ShaderRecord(tmp_path)
    for k in ('new', 'old'):
        rec.note(k, G.kernel_spec(f'{k}.wgsl', ['buf']), 30.0)
    (tmp_path / 'left.1.2.tmp').write_text('{', encoding='utf-8')   # (a write cut off long ago)
    long_ago = time.time() - precompile.RECENT - 3600
    for name in ('old.json', 'left.1.2.tmp'):
        os.utime(tmp_path / name, (long_ago, long_ago))
    read = []
    text = G.Path.read_text
    monkeypatch.setattr(G.Path, 'read_text', lambda self, *a, **k: read.append(self.name) or text(self, *a, **k))
    assert [k for k, *_ in rec.entries(time.time() - precompile.RECENT)] == ['new']
    assert read == ['new.json'] and sorted(p.name for p in tmp_path.iterdir()) == ['new.json']


def test_a_record_that_cannot_be_written_does_not_stop_a_compile(tmp_path):
    blocked = tmp_path / 'file'
    blocked.write_text('')
    rec = G.ShaderRecord(blocked / 'adapter')   # (a folder under a file: cannot be made)
    rec.note('k', G.kernel_spec('x.wgsl', []), 1.0)
    assert not rec.has('k') and rec.entries() == []


class _Device:
    def __init__(self, calls, seconds=0.0):
        self.calls = calls
        self.seconds = seconds   # (how long its driver takes)

    def create_shader_module(self, **k):
        self.calls.append(('module', time.monotonic()))
        time.sleep(self.seconds)
        return 'module'

    def create_bind_group_layout(self, **k):
        return 'layout0'

    def create_pipeline_layout(self, **k):
        return 'layout'

    def create_compute_pipeline(self, **k):
        self.calls.append(('pipeline', time.monotonic()))
        return 'pipe'


def _fake_gpu(calls, record, seconds=0.0):
    gpu = type('FakeGPU', (), {})()
    gpu.device = _Device(calls, seconds)
    gpu.arena = type('Arena', (), {'layout': 'arena'})()
    gpu.record = record
    gpu.on_wait = lambda f: calls.append(('wait', f))
    return gpu


def _app_gpu(calls, record, seconds=0.0):
    """A fake GPU with GPU.kernel's cache of kernels, as the app and its background precompile share it."""
    gpu = _fake_gpu(calls, record, seconds)
    gpu._kernels, gpu._by_shader = {}, {}
    gpu.on_compile = lambda f: calls.append(('compile', f))
    gpu.kernel = types.MethodType(G.GPU.kernel, gpu)
    return gpu


def test_the_app_takes_the_kernel_the_precompile_is_compiling_on_its_device(tmp_path, monkeypatch):
    """The precompile compiles on the app's own device, in a thread. The app asking for the same kernel meanwhile waits
    for it and takes the pipeline it made: one compile, whatever the driver keeps. Only the app's ask shows the
    'Compiling' card."""
    monkeypatch.setattr(G, 'load_wgsl', lambda name, defines=None: f'// {name}')
    calls = []
    gpu = _app_gpu(calls, G.ShaderRecord(tmp_path), seconds=0.6)
    spec = G.kernel_spec('x.wgsl', ['buf'], workgroup=(64, 1, 1))
    bg = threading.Thread(target=lambda: precompile.compile_specs(gpu, [spec]))
    bg.start()
    while not calls:
        time.sleep(0.01)
    k = gpu.kernel('x.wgsl', ['buf'], workgroup=(64, 1, 1))
    bg.join()
    assert [c[0] for c in calls] == ['module', 'compile', 'wait', 'pipeline']
    assert k.pipe == 'pipe' and gpu.kernel('x.wgsl', ['buf'], workgroup=(64, 1, 1)) is k
    assert gpu._by_shader == {k.key: ('layout0', 'pipe')} and not G._busy


def test_the_precompile_waits_for_the_app_without_a_card(tmp_path, monkeypatch):
    """The other way round: the app is compiling a kernel when the precompile reaches it. The precompile waits and takes
    the app's pipeline, and does not tell on_wait (the card is the app's own; no one waits on the precompile)."""
    monkeypatch.setattr(G, 'load_wgsl', lambda name, defines=None: f'// {name}')
    calls = []
    gpu = _app_gpu(calls, G.ShaderRecord(tmp_path), seconds=0.6)
    app = threading.Thread(target=lambda: gpu.kernel('x.wgsl', ['buf'], workgroup=(64, 1, 1)))
    app.start()
    while not [c for c in calls if c[0] == 'module']:
        time.sleep(0.01)
    assert precompile.compile_specs(gpu, [G.kernel_spec('x.wgsl', ['buf'], workgroup=(64, 1, 1))]) == 1
    app.join()
    assert [c[0] for c in calls] == ['compile', 'module', 'pipeline']
    assert len(gpu._kernels) == 1 and not G._busy


def test_the_app_waits_for_a_kernel_another_device_is_compiling(tmp_path, monkeypatch):
    """A thread compiling a kernel on a device of its own: the app asking for the same one meanwhile waits for it
    (every device of a process shares what the driver compiled), then compiles it in a moment."""
    monkeypatch.setattr(G, 'load_wgsl', lambda name, defines=None: f'// {name}')
    rec = G.ShaderRecord(tmp_path)
    bg_calls, app_calls = [], []
    bg = threading.Thread(target=lambda: G.Kernel(_fake_gpu(bg_calls, rec, seconds=0.6), 'x.wgsl', ['buf'],
                                                  workgroup=(64, 1, 1)))
    bg.start()
    while not bg_calls:
        time.sleep(0.01)
    k = G.Kernel(_fake_gpu(app_calls, rec), 'x.wgsl', ['buf'], workgroup=(64, 1, 1))
    bg.join()
    assert app_calls[0] == ('wait', 'x.wgsl')
    assert app_calls[1][0] == 'module' and app_calls[1][1] >= bg_calls[-1][1]   # (once the other was done)
    assert k.pipe == 'pipe' and k.seconds < 0.5
    assert rec.get(k.key)['seconds'] >= 0.6   # (the cold time is the precompile's)
    assert not G._busy
    # another kernel does not wait
    G.Kernel(_fake_gpu(app_calls, rec), 'y.wgsl', ['buf'])
    assert [c for c in app_calls if c[0] == 'wait'] == [('wait', 'x.wgsl')]


def test_a_kernel_is_recorded_once_compiled_and_let_go_when_it_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(G, 'load_wgsl', lambda name, defines=None: f'// {name} {defines}')
    rec = G.ShaderRecord(tmp_path)
    calls = []
    k = G.Kernel(_fake_gpu(calls, rec), 'x.wgsl', ['buf', 'rbuf'], defines={'N': 3}, workgroup=(8, 8, 1))
    row = rec.get(k.key)
    assert row['file'] == 'x.wgsl' and row['bindings'] == ['buf', 'rbuf'] and row['defines'] == {'N': 3}
    assert k.seconds >= 0.0 and not [c for c in calls if c[0] == 'wait']

    gpu = _fake_gpu(calls, rec)
    gpu.device.create_shader_module = lambda **k: (_ for _ in ()).throw(ValueError('bad shader'))
    with pytest.raises(RuntimeError, match='bad shader'):
        G.Kernel(gpu, 'y.wgsl', ['buf'])
    assert not rec.has(G.shader_key("// y.wgsl None", 'main', ['buf'], (8, 8, 4)))
    assert not G._busy   # (no one waits for it)


def _march_key():
    return precompile.current_key(precompile.builtin()[0])


def test_the_march_is_compiled_ahead_every_time(tmp_path):
    """Whether the driver still has it only the driver knows (it drops old code from a full cache): the record saying
    it was compiled does not take it off the list. Warm, it takes a moment."""
    rec = G.ShaderRecord(tmp_path)
    todo = precompile.pending(rec)
    assert [s['file'] for s in todo] == ['liq_march.wgsl']
    rec.note(_march_key(), todo[0], 95.0)
    assert precompile.pending(rec) == todo and precompile.pending(None) == todo
    assert precompile.changed(rec) == []


def test_slow_kernels_used_here_are_compiled_again_when_they_change(tmp_path):
    rec = G.ShaderRecord(tmp_path)
    rec.note(_march_key(), precompile.builtin()[0], 95.0)
    stage = G.kernel_spec('stage.wgsl', ['buf'], 'lume_main', {'F_WOOD': 'false'}, (8, 8, 1))
    rec.note('old-stage', stage, 24.0)                                       # slow, its shader since changed
    rec.note('old-fast', G.kernel_spec('liq_p2g.wgsl', ['buf']), 0.3)       # fast: not worth it
    rec.note('old-gone', G.kernel_spec('no_such_file.wgsl', ['buf']), 30.0)  # an older version's
    rec.note('old-unused', G.kernel_spec('raymarch.wgsl', ['buf']), 9.0)     # not used for months
    month = 31 * 24 * 3600
    os.utime(tmp_path / 'old-unused.json', (time.time() - 3 * month, time.time() - 3 * month))
    todo = precompile.changed(rec)
    assert [(s['file'], s['entry'], s['defines']) for s in todo] == [('stage.wgsl', 'lume_main', {'F_WOOD': 'false'})]
    assert precompile.pending(rec)[1:] == todo
    assert not (tmp_path / 'old-unused.json').exists()   # (removed, not kept for ever)
    rec.note(precompile.current_key(todo[0]), todo[0], 2.5)
    assert precompile.changed(rec) == []


def test_a_row_that_no_longer_builds_is_forgotten(tmp_path):
    """A row from an older version whose kernel's resources have since changed fails to build from it: it is taken
    off the record, not listed and failed again at every start."""
    rec = G.ShaderRecord(tmp_path)
    rec.note('old-blur', G.kernel_spec('liq_surf_blur.wgsl', ['tex_wrong']), 30.0)
    rec.note('older-blur', G.kernel_spec('liq_surf_blur.wgsl', ['tex_wrong']), 12.0)   # (the same kernel, once more)
    todo = precompile.changed(rec)
    assert [s['file'] for s in todo] == ['liq_surf_blur.wgsl'] and sorted(todo[0]['rows']) == ['old-blur', 'older-blur']
    gpu = types.SimpleNamespace(record=rec)
    gpu.kernel = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('Validation Error: binding 0'))
    assert precompile.compile_specs(gpu, todo) == 0
    assert not rec.has('old-blur') and not rec.has('older-blur')
    assert precompile.changed(rec) == []


def test_a_kernel_compiled_with_defines_in_any_order_is_not_compiled_again(tmp_path, monkeypatch):
    """load_wgsl writes numeric defines in the order they are given (BB_DEFINES: BB_MIN, then BB_MAX), so the record
    keeps that order: rebuilt in another order the shader would be another, and the precompile would compile a kernel
    the app never uses."""
    from blackbody.engine.renderer import BB_DEFINES
    assert list(BB_DEFINES) != sorted(BB_DEFINES)
    monkeypatch.setattr(precompile, 'SLOW', 0.05)
    rec = G.ShaderRecord(tmp_path)
    k = G.Kernel(_fake_gpu([], rec, seconds=0.1), 'raymarch.wgsl', ['buf'], defines=BB_DEFINES, workgroup=(8, 8, 1))
    assert rec.get(k.key)['seconds'] >= 0.1
    assert precompile.changed(rec) == []


def test_a_damaged_record_is_passed_over(tmp_path):
    rec = G.ShaderRecord(tmp_path)
    rec.note(_march_key(), precompile.builtin()[0], 95.0)
    (tmp_path / 'half.json').write_text('{"file": "stage.wgsl", "seconds": 30', encoding='utf-8')     # (cut off)
    (tmp_path / 'list.json').write_text('[1, 2]', encoding='utf-8')
    (tmp_path / 'nokeys.json').write_text('{"file": "stage.wgsl", "seconds": 30}', encoding='utf-8')
    (tmp_path / 'badsecs.json').write_text(json.dumps(dict(G.kernel_spec('stage.wgsl', ['buf']), seconds='slow')),
                                           encoding='utf-8')
    (tmp_path / 'baddefs.json').write_text(json.dumps(dict(G.kernel_spec('stage.wgsl', ['buf']), defines=[1],
                                                           seconds=30)), encoding='utf-8')
    assert precompile.changed(rec) == []


def test_other_drivers_records_go_once_unused(tmp_path):
    """A driver update starts a record of its own (gpu.adapter_key); the old one goes once no kernel has been used on it
    within RECENT. One still in use (another GPU of this machine) stays."""
    rec = G.ShaderRecord(tmp_path / 'new-driver')
    rec.note(_march_key(), precompile.builtin()[0], 95.0)
    for name in ('old-driver', 'other-gpu'):
        G.ShaderRecord(tmp_path / name).note('k', G.kernel_spec('stage.wgsl', ['buf']), 30.0)
    long_ago = time.time() - precompile.RECENT - 3600
    os.utime(tmp_path / 'old-driver' / 'k.json', (long_ago, long_ago))
    precompile.changed(rec)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['new-driver', 'other-gpu']


def test_the_slowest_kernels_go_first(tmp_path):
    rec = G.ShaderRecord(tmp_path)
    rec.note('a', G.kernel_spec('stage.wgsl', ['buf'], 'lume_main'), 5.0)
    rec.note('b', G.kernel_spec('stage.wgsl', ['buf'], 'main'), 30.0)
    rec.note('c', precompile.builtin()[0], 500.0)   # (the march at an older version: cold took this long)
    assert [(s['file'], s['entry']) for s in precompile.pending(rec)] == [
        ('liq_march.wgsl', 'main'), ('stage.wgsl', 'main'), ('stage.wgsl', 'lume_main')]


def test_the_app_starts_the_precompile_unless_it_is_turned_off(tmp_path, monkeypatch):
    """Every time the app opens (the march is compiled ahead every time), without reading the record on the engine's
    thread: the background thread reads it."""
    started = []

    class Job:
        def __init__(self, gpu):
            self.todo, self.stopped = precompile.builtin(), False
            started.append(self)

        def start(self):
            pass

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(precompile, 'Background', Job)
    monkeypatch.setattr(precompile, 'changed', lambda record, now=None: 1 / 0)   # (not on this thread)
    gpu = type('FakeGPU', (), {'record': G.ShaderRecord(tmp_path / 'adapter'), 'name': 'GPU', 'backend': 'Vulkan'})()
    gpu.record.note(_march_key(), precompile.builtin()[0], 95.0)
    monkeypatch.setenv('BLACKBODY_PRECOMPILE', '0')
    assert precompile.start_background(gpu) is None
    monkeypatch.delenv('BLACKBODY_PRECOMPILE')
    job = precompile.start_background(gpu)
    assert job is started[-1] and [s['file'] for s in job.todo] == ['liq_march.wgsl']
    precompile.stop(job)
    assert job.stopped
    precompile.stop(None)
    monkeypatch.setattr(Job, 'start', lambda self: 1 / 0)
    assert precompile.start_background(gpu) is None   # (never in the way of the app starting)


def test_the_background_compiles_the_march_then_reads_the_record(monkeypatch):
    """The built-in kernels first, then the slow ones that have changed, found in the record once those are done (the
    record may be large, and the march is the one a scene may be waiting for)."""
    order = []
    march, stage = G.kernel_spec('liq_march.wgsl', ['buf']), G.kernel_spec('stage.wgsl', ['buf'])
    monkeypatch.setattr(precompile, 'builtin', lambda: [march])
    monkeypatch.setattr(precompile, 'changed', lambda record, now=None: order.append(('read', record)) or [stage])
    monkeypatch.setattr(precompile, 'compile_specs', lambda g, specs, say=None: order.append(specs[0]['file']) or 1)
    job = precompile.Background(type('AppGPU', (), {'record': 'the record'})())
    job.start()
    job.join(5)
    assert order == ['liq_march.wgsl', ('read', 'the record'), 'stage.wgsl']
    assert job.todo == [march, stage] and job.done == 2 and job.error is None


def test_the_background_compile_stops_between_kernels(monkeypatch):
    """stop() keeps it from starting another kernel (one under way finishes in the driver); it compiles on the app's
    own GPU, where the app finds what it made."""
    compiled = []
    gate = threading.Event()

    def compile_one(g, specs, say=None):
        compiled.append((g, specs[0]['file']))
        gate.wait(5)
        return 1

    monkeypatch.setattr(precompile, 'compile_specs', compile_one)
    monkeypatch.setattr(precompile, 'builtin', lambda: [G.kernel_spec(f'{n}.wgsl', ['buf']) for n in 'abc'])
    monkeypatch.setattr(precompile, 'changed', lambda record, now=None: 1 / 0)
    app = type('AppGPU', (), {'record': None})()
    job = precompile.Background(app)
    job.start()
    while not compiled:
        time.sleep(0.01)
    job.stop()
    gate.set()
    job.join(5)
    assert compiled == [(app, 'a.wgsl')] and job.done == 1 and job.error is None and job.daemon


def test_a_machine_without_a_home_folder_still_starts(tmp_path, monkeypatch):
    """The record's folder comes from BLACKBODY_CACHE, else the system's cache folder; the home folder is looked up only
    when neither is set (a farm node's user may have none). With nowhere to keep it there is no record, and kernels
    compile all the same."""
    def no_home(*a):
        raise RuntimeError('Could not determine home directory.')

    monkeypatch.setattr(G.Path, 'home', staticmethod(no_home))
    monkeypatch.setattr(G, 'load_wgsl', lambda name, defines=None: f'// {name}')
    monkeypatch.delenv('BLACKBODY_CACHE', raising=False)
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path))
    assert str(G.shader_dir()).startswith(str(tmp_path))
    assert G.shader_record('adapter').folder.parent == G.shader_dir()
    monkeypatch.delenv('LOCALAPPDATA')
    monkeypatch.delenv('XDG_CACHE_HOME')
    assert G.shader_record('adapter') is None
    k = G.Kernel(_fake_gpu([], None), 'x.wgsl', ['buf'])
    assert k.pipe == 'pipe' and [s['file'] for s in precompile.pending(None)] == ['liq_march.wgsl']


def test_the_command_line_has_precompile(monkeypatch, capsys):
    from blackbody import cli
    seen = []
    monkeypatch.setattr(precompile, 'main', lambda quiet=False: seen.append(quiet) or 0)
    assert cli.main(['precompile']) == 0
    assert cli.main(['precompile', '-q']) == 0
    assert seen == [False, True]


def test_a_liquid_scene_compiles_the_march_the_precompile_builds(engine):
    """The march is compiled when the first frame is drawn (LiquidRenderer.k_march), from the spec the precompile
    compiles ahead of time (the same shader to the driver), and the record has it."""
    from blackbody.scene import presets
    sc = presets.make('water_pour')
    sc.data['domain']['resolution'] = 32
    engine.invalidate()
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start, cache=False)
    engine.render(sc, sc.start, (64, 36))
    rec = engine.gpu.record
    key = _march_key()
    assert engine.liquid_r.k_march.key == key and rec.has(key)
    assert precompile.changed(rec) == []


def test_the_march_spec_matches_the_shader():
    """march_spec's resources are the ones liq_march.wgsl declares (group 0)."""
    from blackbody.engine.liquid_render import MARCH_BINDINGS, march_spec
    code = G.load_wgsl('liq_march.wgsl')
    slots = sorted(int(n) for n in re.findall(r'@group\(0\)\s*@binding\((\d+)\)', code))
    assert slots == list(range(len(MARCH_BINDINGS)))
    assert march_spec()['bindings'] == MARCH_BINDINGS and march_spec()['workgroup'] == [8, 8, 1]


# -- the march's size -------------------------------------------------------------------------------------------------

MARCH_BUDGET = 16000   # statements in the march with every call inlined (12,729 now; see the test below)


def inlined_size(name, entry='main'):
    """Statements in a kernel with every function call inlined, as GPU drivers compile it: each function's own
    statements plus, for each call it makes, the callee's inlined size. Loops are not unrolled (a driver may)."""
    src = re.sub(r'//[^\n]*', '', G.load_wgsl(name))
    funcs = {}
    for m in re.finditer(r'\bfn\s+(\w+)\s*\(', src):
        i, depth = src.index('{', m.end()), 0
        for j in range(i, len(src)):
            depth += {'{': 1, '}': -1}.get(src[j], 0)
            if depth == 0:
                break
        funcs[m.group(1)] = src[i:j + 1]
    calls = {f: [c for c in re.findall(r'\b(\w+)\s*\(', b) if c in funcs and c != f] for f, b in funcs.items()}

    @lru_cache(None)
    def size(f):
        return funcs[f].count(';') + sum(size(c) for c in calls[f])

    return size(entry)


def test_the_march_stays_quick_to_compile():
    """The driver inlines every call of a function, so each call of a large one (what a ray lands on, the colliders'
    distance) is a copy of it, and the copies are what it compiles. Cold on an RTX 3090 (16-thread CPU): at 66,000
    inlined statements the march took 16 minutes and 22 GB of memory to compile, at 19,000 110 s and 3.5 GB, at
    16,500 85 s, at 12,700 45 s and 1.8 GB. Before adding a call of a large function, look for one already on that
    path to share (land(), through(), the loops of solid_n and normal_at)."""
    n = inlined_size('liq_march.wgsl')
    assert n < MARCH_BUDGET, f'liq_march.wgsl inlines to {n} statements (budget {MARCH_BUDGET})'


def test_no_large_loop_of_the_march_writes_at_a_varying_index():
    """Direct3D's shader compiler (FXC, which wgpu uses there) unrolls a loop that writes a vector or array at a varying
    index, and gives up on one whose body calls the march's functions: too large once inlined ('unable to unroll
    loop'; the six taps of normal_at and solid_n did). Such a loop selects instead."""
    src = re.sub(r'//[^\n]*', '', G.load_wgsl('liq_march.wgsl'))
    names = set(re.findall(r'\bfn\s+(\w+)\s*\(', src))
    bad = []
    for m in re.finditer(r'\bfor\s*\(', src):
        i, depth = src.index('{', m.end()), 0
        for j in range(i, len(src)):
            depth += {'{': 1, '}': -1}.get(src[j], 0)
            if depth == 0:
                break
        body = src[i:j + 1]
        writes = re.findall(r'\b\w+\[[^\]]*[A-Za-z_][^\]]*\]\s*[-+*/]?=(?!=)', body)
        if writes and any(c in names for c in re.findall(r'\b(\w+)\s*\(', body)):
            bad.append(writes[0])
    assert not bad, f'loops of liq_march.wgsl that write at a varying index and call its functions: {bad}'
