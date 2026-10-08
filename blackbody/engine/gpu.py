"""GPU context and compute helpers on top of wgpu (WebGPU on Vulkan, Direct3D 12 or Metal).

Every simulation and render pass is a compute kernel. A kernel's resources live in bind group 0
and its parameters in bind group 1, a dynamic-offset window into a per-submission uniform arena,
so one command buffer can carry hundreds of dispatches that each see their own parameters.

The GPU also knows the card's memory (read from the system: card_memory) and counts what its
textures and buffers take (GPU.allocated). A scene's grids are fitted to the card before they are
made (scene/model.py memory_plan, through card()): to a plan made from the card's size alone, never
from what happens to be free, so a scene is laid out (and its cache signed) the same on every start.
An allocation past what the system gives the app is refused before the driver is asked
(GPUOutOfMemory), and failure() tells a lost device or an allocation the card could not make from
other errors, so the app can start the engine again (ui/worker.py) instead of needing a restart.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import wgpu

log = logging.getLogger('blackbody.gpu')

TU = wgpu.TextureUsage
BU = wgpu.BufferUsage
SS = wgpu.ShaderStage

WGSL_DIR = Path(__file__).with_name('wgsl')

# format -> (numpy dtype, channels, bytes per texel)
FORMATS = {
    'rgba32float': (np.float32, 4, 16),
    'rgba16float': (np.float16, 4, 8),
    'rg32float': (np.float32, 2, 8),
    'rg16float': (np.float16, 2, 4),
    'r32float': (np.float32, 1, 4),
    'r16float': (np.float16, 1, 2),
    'rgba8unorm': (np.uint8, 4, 4),
    'r32uint': (np.uint32, 1, 4),
}

DEFAULT_USAGE = TU.TEXTURE_BINDING | TU.STORAGE_BINDING | TU.COPY_SRC | TU.COPY_DST
GB = 2 ** 30   # a GB of GPU memory as the system shows it (a 24 GB card has 24 of these)
# bytes per texel of formats made without being read back (FORMATS has the rest)
OTHER_TEXELS = {'depth32float': 4, 'depth24plus': 4, 'depth24plus-stencil8': 4, 'bgra8unorm': 4, 'rgba8unorm-srgb': 4,
                'r8unorm': 1, 'rg8unorm': 2, 'r16uint': 2, 'r32sint': 4, 'rg32uint': 8, 'rgba32uint': 16}


def texel_bytes(fmt):
    f = FORMATS.get(fmt)
    return f[2] if f is not None else OTHER_TEXELS.get(fmt, 16)


class GPUUnavailable(RuntimeError):
    """No usable GPU adapter was found."""


class GPUOutOfMemory(RuntimeError):
    """An allocation the card has no room for: refused before the driver was asked (past what the system gives the app),
    or refused by the driver."""

    def __init__(self, wanted, allocated, budget, label='', driver=False):
        self.wanted, self.allocated, self.budget, self.label, self.driver = int(wanted), int(allocated), budget, label, driver
        what = f'{label} ({wanted / GB:.2f} GB)' if label else f'{wanted / GB:.2f} GB more'
        if driver or budget is None:
            text = f'The GPU had no memory left for {what}, with {allocated / GB:.2f} GB in use.'
        else:
            text = f'The GPU has no room for {what}: {allocated / GB:.2f} GB of the {budget / GB:.1f} GB it gives Blackbody are in use.'
        super().__init__(text)


def failure(ex):
    """What kind of GPU failure an exception is: 'lost' (the device is gone: a driver reset, a time-out, a card that went
    away), 'memory' (an allocation the card could not make) or None (anything else). wgpu reports both as validation
    errors ('Parent device is lost', 'Not enough memory left'), so the words are what tells them apart; the causes of an
    exception are looked at too."""
    seen = set()
    while ex is not None and id(ex) not in seen:
        seen.add(id(ex))
        if isinstance(ex, (GPUOutOfMemory, wgpu.GPUOutOfMemoryError)):
            return 'memory'
        if isinstance(ex, (wgpu.GPUError, RuntimeError)):
            text = str(ex).lower()
            if any(w in text for w in ('device is lost', 'device lost', 'device was lost', 'devicelost', 'device_lost')):
                return 'lost'
            if any(w in text for w in ('not enough memory', 'out of memory', 'outofmemory', 'out_of_memory')):
                return 'memory'
        ex = ex.__cause__ or ex.__context__
    return None


class _LostWatch(logging.Handler):
    """wgpu says a device was lost, and why, only in its log (its device-lost callback cannot raise): keep it, for what
    the app says when it starts the GPU again (GPU.lost_reason)."""

    count = 0
    text = ''

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return
        if 'device was lost' in msg:
            _LostWatch.count += 1
            _LostWatch.text = msg


logging.getLogger('wgpu').addHandler(_LostWatch(logging.ERROR))


def ceil_div(a, b):
    return -(-int(a) // int(b))


def groups_1d(n, size=64):
    """Workgroup counts for n invocations of a 1D kernel, spilling into y past WebGPU's 65535 per
    dimension (the kernel recovers its index as x + y * num_workgroups.x * size)."""
    g = max(ceil_div(n, size), 1)
    return (min(g, 65535), ceil_div(g, 65535), 1)


def _made(obj, gpu, nbytes, label, make):
    """Make a texture or buffer through `make()`, counted in the GPU's allocated bytes: refused past its ceiling, and an
    allocation the driver could not make raised as GPUOutOfMemory. (A stand-in GPU without the count just makes it.)"""
    take = getattr(gpu, 'take', None)
    if take is None:
        return make()
    take(nbytes, label)
    try:
        out = make()
    except Exception as ex:
        gpu.allocated -= nbytes
        if failure(ex) == 'memory':
            raise GPUOutOfMemory(nbytes, gpu.allocated, gpu.ceiling, label, driver=True) from ex
        raise
    obj._bytes = nbytes
    return out


def _free(obj):
    """Take a texture or buffer out of its GPU's count (once: on destroy, or when it is collected without one)."""
    n = obj._bytes
    if n:
        obj._bytes = 0
        obj._gpu.allocated -= n


class Texture:
    """A texture, its default view and its shape. `size` is (w, h, d); 2D textures have d == 1."""

    __slots__ = ('tex', 'view', 'size', 'format', 'dim', 'label', '_gpu', '_bytes')

    def __init__(self, gpu, size, fmt, dim='3d', usage=DEFAULT_USAGE, label=''):
        w, h, d = (int(size[0]), int(size[1]), int(size[2]) if len(size) > 2 else 1)
        self._gpu = gpu
        self._bytes = 0
        self.size = (w, h, d)
        self.format = fmt
        self.dim = dim
        self.label = label
        self.tex = _made(self, gpu, w * h * d * texel_bytes(fmt), label,
                         lambda: gpu.device.create_texture(size=(w, h, d), dimension=dim, format=fmt, usage=usage, label=label))
        self.view = self.tex.create_view()

    @property
    def nbytes(self):
        w, h, d = self.size
        return w * h * d * texel_bytes(self.format)

    def destroy(self):
        _free(self)
        try:
            self.tex.destroy()
        except Exception:  # already destroyed or device lost
            pass

    def __del__(self):
        try:
            _free(self)
        except Exception:   # (half made, or the interpreter is shutting down)
            pass

    def __repr__(self):
        return f'<Texture {self.label} {self.size} {self.format}>'


class Buffer:
    __slots__ = ('buf', 'size', 'label', '_gpu', '_bytes')

    def __init__(self, gpu, size, usage=BU.STORAGE | BU.COPY_DST | BU.COPY_SRC, label=''):
        self.size = int(size)
        self.label = label
        self._gpu = gpu
        self._bytes = 0
        self.buf = _made(self, gpu, self.size, label, lambda: gpu.device.create_buffer(size=self.size, usage=usage, label=label))

    def destroy(self):
        _free(self)
        try:
            self.buf.destroy()
        except Exception:
            pass

    def __del__(self):
        try:
            _free(self)
        except Exception:
            pass


class Uniforms:
    """Builds a kernel's parameter block. Every field is a vec4 (16 bytes), matching the WGSL
    structs, which are written with vec4 members only so no padding rules can bite."""

    __slots__ = ('data',)

    def __init__(self):
        self.data = []

    def v4(self, x=0.0, y=0.0, z=0.0, w=0.0):
        self.data.extend((float(x), float(y), float(z), float(w)))
        return self

    def v3(self, v, w=0.0):
        return self.v4(v[0], v[1], v[2], w)

    def m4(self, m):
        """A 4x4 matrix given row-major in numpy; WGSL mat4x4 is column-major."""
        self.data.extend(np.asarray(m, np.float32).T.reshape(-1).tolist())
        return self

    def raw(self, values):
        vals = [float(v) for v in values]
        assert len(vals) % 4 == 0, 'uniform block must be vec4 aligned'
        self.data.extend(vals)
        return self

    def tobytes(self):
        return np.asarray(self.data, np.float32).tobytes()


class UniformArena:
    """One large uniform buffer per submission; each dispatch gets a 256-byte aligned window."""

    ALIGN = 256
    WINDOW = 8192  # the largest parameter block a kernel may use

    def __init__(self, device, size=8 << 20):
        self.device = device
        self.size = size
        self.buf = device.create_buffer(size=size, usage=BU.UNIFORM | BU.COPY_DST, label='uniform-arena')
        self.cpu = bytearray(size)
        self.off = 0
        vis = SS.COMPUTE | SS.VERTEX | SS.FRAGMENT
        self.layout = device.create_bind_group_layout(entries=[{
            'binding': 0, 'visibility': vis,
            'buffer': {'type': 'uniform', 'has_dynamic_offset': True, 'min_binding_size': 0}}])
        self.group = device.create_bind_group(layout=self.layout, entries=[
            {'binding': 0, 'resource': {'buffer': self.buf, 'offset': 0, 'size': self.WINDOW}}])

    def fits(self, n):
        return self.off + max(n, 16) + self.WINDOW <= self.size

    def push(self, data: bytes):
        n = len(data)
        if n > self.WINDOW:
            raise ValueError(f'uniform block of {n} bytes exceeds the {self.WINDOW} byte window')
        off = self.off
        self.cpu[off:off + n] = data
        self.off = off + ceil_div(max(n, 16), self.ALIGN) * self.ALIGN
        return off

    def flush(self, queue):
        if self.off:
            queue.write_buffer(self.buf, 0, memoryview(self.cpu)[:self.off])
        self.off = 0


# ---------------------------------------------------------------------------------------------
# Shader sources
# ---------------------------------------------------------------------------------------------

_INCLUDE = re.compile(r'^\s*//!include\s+([\w./-]+)\s*$', re.M)


def load_wgsl(name, defines=None, _seen=None):
    """Read a WGSL file, expanding `//!include file.wgsl` lines (each file at most once).
    String defines replace `${NAME}` tokens (texel formats, which WGSL constants cannot express);
    numeric and boolean defines become `const NAME = value;` declarations."""
    seen = set() if _seen is None else _seen

    def expand(fname):
        if fname in seen:
            return ''
        seen.add(fname)
        text = (WGSL_DIR / fname).read_text(encoding='utf-8')
        return _INCLUDE.sub(lambda m: expand(m.group(1)), text)

    body = expand(name)
    head = ''
    for k, v in (defines or {}).items():
        if isinstance(v, str):
            body = body.replace('${' + k + '}', v)
        elif isinstance(v, bool):
            head += f'const {k}: bool = {"true" if v else "false"};\n'
        elif isinstance(v, int):
            head += f'const {k}: i32 = {v};\n'
        else:
            head += f'const {k}: f32 = {float(v)!r};\n'
    return head + body


# ---------------------------------------------------------------------------------------------
# The record of compiled kernels
# ---------------------------------------------------------------------------------------------
#
# WebGPU has no pipeline cache Blackbody could keep (wgpu-native offers none), but the graphics
# driver keeps the code it compiles on disk, by the exact shader: a kernel compiled once compiles
# again in a fraction of a second, until the shader or the driver changes. A large one compiles
# cold for a minute or more. A device does not see what another process compiled after it was made
# (measured with NVIDIA's Vulkan driver), so the background precompile (precompile.py) runs in the
# app's own process, in a thread, on the app's own device (wgpu takes a device's calls from several
# threads; the app's work on it goes on meanwhile). A kernel it is compiling is waited for, and the
# pipeline it made taken as it is (_busy, GPU._by_shader): nothing is compiled twice, whatever the
# driver keeps. (On another device, the CLI's, the driver's cache is what is shared.) The record keeps,
# per adapter and driver, which kernels have been compiled, how long the first compile took and when
# each was last used, so the precompile can find the slow ones whose shader an update has changed.
# It says what was compiled, not what the driver still has: the driver drops old code when its
# cache is full, and only the driver knows.

SHADER_WAIT = 1800.0    # s a kernel waits for another thread compiling the same one, before compiling it itself

_busy = {}                   # shader key -> threading.Event: a kernel some thread of this process is compiling
_busy_lock = threading.Lock()


def shader_dir():
    """Where the record of compiled kernels is kept (BLACKBODY_CACHE overrides it). The home folder is looked up only
    when neither that nor the system's cache folder is set (Path.home raises for a user without one: a farm node's)."""
    env = os.environ.get('BLACKBODY_CACHE')
    if env:
        return Path(env) / 'shaders'
    if os.name == 'nt':
        return Path(os.environ.get('LOCALAPPDATA') or Path.home() / 'AppData' / 'Local') / 'Blackbody' / 'shaders'
    return Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'blackbody' / 'shaders'


def shader_record(adapter):
    """The record of what one adapter and driver (adapter_key) have compiled; None when there is nowhere to keep it
    (kernels compile all the same, and the precompile still does the built-in ones)."""
    try:
        return ShaderRecord(shader_dir() / adapter)
    except (RuntimeError, KeyError, OSError) as ex:   # (no cache folder set and no home folder)
        log.info('No record of compiled shaders: %s', ex)
        return None


def shader_key(code, entry, bindings, workgroup):
    """A kernel's identity as the driver sees it: its source (includes and defines expanded), entry point, resource
    layout and workgroup size."""
    h = hashlib.sha1(code.encode('utf-8'))
    h.update(repr((entry, tuple(bindings), tuple(workgroup))).encode('utf-8'))
    return h.hexdigest()[:24]


def adapter_key(info):
    """An adapter and its driver (GPUAdapterInfo, or any mapping like it), and the wgpu that translates the shaders:
    what the driver's own cache of compiled code is good for."""
    import wgpu
    parts = [str(info.get(k, '')) for k in ('vendor', 'device', 'description', 'backend_type', 'vendor_id', 'device_id')]
    parts.append(str(getattr(wgpu, '__version__', '')))
    return hashlib.sha1('|'.join(parts).encode('utf-8')).hexdigest()[:16]


class ShaderRecord:
    """The kernels compiled on one adapter and driver: a small file per kernel, named by its shader_key, holding how
    to build it again (its spec) and how long its first compile took."""

    def __init__(self, folder):
        self.folder = Path(folder)

    def get(self, key):
        try:
            return json.loads((self.folder / f'{key}.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None

    def has(self, key):
        return (self.folder / f'{key}.json').is_file()

    def note(self, key, spec, seconds):
        """Record a kernel compiled for the first time here. A second compile of it is the driver's cache: only the
        time it was last used is kept (the file's own, which entries reads)."""
        if self.has(key):
            try:
                os.utime(self.folder / f'{key}.json')
            except OSError:
                pass
            return
        try:
            import blackbody
            self.folder.mkdir(parents=True, exist_ok=True)
            row = dict(spec, seconds=round(float(seconds), 3), time=round(time.time()), version=blackbody.__version__)
            tmp = self.folder / f'{key}.{os.getpid()}.{threading.get_ident()}.tmp'
            tmp.write_text(json.dumps(row, indent=1), encoding='utf-8')
            os.replace(tmp, self.folder / f'{key}.json')
        except (OSError, TypeError, ValueError) as ex:   # (a read-only cache folder, a define that is not plain data)
            log.debug('shader record: %s', ex)

    def forget(self, key):
        """Take a kernel off the record (one that no longer builds from its row: precompile.compile_specs)."""
        try:
            os.remove(self.folder / f'{key}.json')
        except OSError:
            pass

    def entries(self, since=None):
        """Every kernel recorded: (key, row, time last used). With `since` (a time), the files not used since then are
        deleted, not read (kernels of older versions, and ones no scene has asked for in a long while): the record
        stays the size of what this machine uses."""
        try:
            files = sorted(os.scandir(self.folder), key=lambda f: f.name)
        except OSError:
            return []
        out = []
        for f in files:
            if not f.name.endswith(('.json', '.tmp')):   # (only its own files)
                continue
            try:
                used = f.stat().st_mtime
                if since is not None and used < since:
                    os.remove(f.path)
                elif f.name.endswith('.json'):
                    out.append((f.name[:-5], json.loads(Path(f.path).read_text(encoding='utf-8')), used))
            except (OSError, ValueError):
                pass
        return out

    def last_used(self):
        """When a kernel of this record was last used (0 when it has none)."""
        try:
            return max((f.stat().st_mtime for f in os.scandir(self.folder) if f.name.endswith('.json')), default=0.0)
        except OSError:
            return 0.0


def kernel_spec(source_file, bindings, entry='main', defines=None, workgroup=(8, 8, 4)):
    """How to build a kernel again (GPU.kernel's arguments), as plain data. The defines keep their order: load_wgsl
    writes them in it, so another order is another shader to the driver."""
    return {'file': str(source_file), 'entry': entry, 'bindings': list(bindings), 'defines': dict(defines or {}),
            'workgroup': list(workgroup)}


# ---------------------------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------------------------

def _layout_entry(i, spec):
    """Binding spec strings:
    tex3d / tex2d            sampled float texture (filterable)
    utex3d / utex2d          sampled unfilterable-float texture (textureLoad only)
    st3d:FMT:ACC / st2d:...  storage texture, ACC in w, r, rw
    smp / smpn               filtering / non-filtering sampler
    buf / rbuf               read-write / read-only storage buffer
    """
    vis = SS.COMPUTE
    kind, *rest = spec.split(':')
    e = {'binding': i, 'visibility': vis}
    if kind in ('tex3d', 'tex2d', 'utex3d', 'utex2d'):
        dim = '3d' if kind.endswith('3d') else '2d'
        e['texture'] = {'sample_type': 'unfilterable-float' if kind.startswith('u') else 'float',
                        'view_dimension': dim, 'multisampled': False}
    elif kind in ('st3d', 'st2d'):
        fmt, acc = rest
        e['storage_texture'] = {'access': {'w': 'write-only', 'r': 'read-only', 'rw': 'read-write'}[acc],
                                'format': fmt, 'view_dimension': '3d' if kind == 'st3d' else '2d'}
    elif kind == 'smp':
        e['sampler'] = {'type': 'filtering'}
    elif kind == 'smpn':
        e['sampler'] = {'type': 'non-filtering'}
    elif kind == 'buf':
        e['buffer'] = {'type': 'storage'}
    elif kind == 'rbuf':
        e['buffer'] = {'type': 'read-only-storage'}
    else:
        raise ValueError(f'unknown binding spec {spec!r}')
    return e


class Kernel:
    """A compute pipeline with an explicit layout: group 0 = resources, group 1 = parameters."""

    def __init__(self, gpu, source_file, bindings, entry='main', defines=None, workgroup=(8, 8, 4), label=None,
                 quiet=False):
        self.gpu = gpu
        self.label = label or f'{source_file}:{entry}'
        self.bindings = list(bindings)
        self.workgroup = workgroup
        code = load_wgsl(source_file, defines)
        self.key = shader_key(code, entry, self.bindings, workgroup)
        self.seconds = 0.0   # how long it took to compile (the driver's: cold the first time, a moment after)
        record = getattr(gpu, 'record', None)
        made = getattr(gpu, '_by_shader', None)   # (what any thread has compiled on this device)
        # another thread (the background precompile) compiling this very kernel: wait for it, and take the pipeline
        # it made (on another device the driver has it ready). quiet: no one is waiting on this one (the precompile
        # waiting for the app), so on_wait is not told
        t_end = time.monotonic() + SHADER_WAIT
        told = quiet
        while True:
            with _busy_lock:
                other = _busy.get(self.key)
                if other is None:
                    mine = _busy[self.key] = threading.Event()
                    break
            if not told and getattr(gpu, 'on_wait', None) is not None:
                gpu.on_wait(source_file)
            told = True
            if not other.wait(max(t_end - time.monotonic(), 0.0)):
                mine = None   # (it is taking too long: compile it here as well)
                break
        dev = gpu.device
        t0 = time.perf_counter()
        try:
            twin = made.get(self.key) if made is not None else None
            if twin is not None:   # (compiled on this device already, by the thread waited for: the same pipeline)
                self.layout0, self.pipe = twin
            else:
                try:
                    module = dev.create_shader_module(code=code, label=self.label)
                except Exception as ex:  # surface the WGSL error with the kernel name
                    raise RuntimeError(f'WGSL compile failed in {self.label}:\n{ex}') from None
                self.layout0 = dev.create_bind_group_layout(entries=[_layout_entry(i, s)
                                                                     for i, s in enumerate(self.bindings)])
                layout = dev.create_pipeline_layout(bind_group_layouts=[self.layout0, gpu.arena.layout])
                self.pipe = dev.create_compute_pipeline(layout=layout, compute={'module': module, 'entry_point': entry},
                                                        label=self.label)
                self.seconds = time.perf_counter() - t0
                if made is not None:
                    made[self.key] = (self.layout0, self.pipe)   # (not the kernel: its bind groups hold resources)
            if record is not None:
                record.note(self.key, kernel_spec(source_file, bindings, entry, defines, workgroup), self.seconds)
        finally:
            if mine is not None:
                with _busy_lock:
                    _busy.pop(self.key, None)
                mine.set()
        self._groups = {}

    def bind(self, resources):
        key = tuple(id(r) for r in resources)
        g = self._groups.get(key)
        if g is not None:
            return g
        if len(resources) != len(self.bindings):
            raise ValueError(f'{self.label}: expected {len(self.bindings)} resources, got {len(resources)}')
        entries = []
        for i, (spec, r) in enumerate(zip(self.bindings, resources)):
            if isinstance(r, Texture):
                res = r.view
            elif isinstance(r, Buffer):
                res = {'buffer': r.buf, 'offset': 0, 'size': r.size}
            else:
                res = r  # sampler, view or raw binding
            entries.append({'binding': i, 'resource': res})
        g = self.gpu.device.create_bind_group(layout=self.layout0, entries=entries, label=self.label)
        if len(self._groups) > 64:  # textures get reallocated when a domain is resized
            self._groups.clear()
        self._groups[key] = (g, tuple(resources))  # keep the resources alive while cached
        return self._groups[key]

    def groups_for(self, size):
        wx, wy, wz = self.workgroup
        return (ceil_div(size[0], wx), ceil_div(size[1], wy), ceil_div(size[2] if len(size) > 2 else 1, wz))


class Batch:
    """Records dispatches into one command buffer. Use as a context manager; it submits on exit."""

    def __init__(self, gpu):
        self.gpu = gpu
        self.enc = gpu.device.create_command_encoder()
        self.cp = None
        self.count = 0

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            self.submit()
        else:
            self._end_pass()
            self.gpu.arena.off = 0

    def _end_pass(self):
        if self.cp is not None:
            self.cp.end()
            self.cp = None

    def run(self, kernel: Kernel, resources, uniforms, size=None, groups=None):
        """Dispatch `kernel` over `size` threads (or explicit workgroup counts)."""
        data = uniforms.tobytes() if isinstance(uniforms, Uniforms) else (uniforms or b'\0' * 16)
        arena = self.gpu.arena
        if not arena.fits(len(data)):
            self.submit(restart=True)
        off = arena.push(data)
        if self.cp is None:
            self.cp = self.enc.begin_compute_pass()
        g, _ = kernel.bind(resources)
        self.cp.set_pipeline(kernel.pipe)
        self.cp.set_bind_group(0, g)
        self.cp.set_bind_group(1, arena.group, [off])
        wg = groups or kernel.groups_for(size)
        if min(wg) > 0:
            self.cp.dispatch_workgroups(*wg)
        self.count += 1

    def run_indirect(self, kernel: Kernel, resources, uniforms, args: 'Buffer', offset=0):
        """Dispatch `kernel` with workgroup counts read from `args` on the GPU (three u32 at `offset`),
        for work whose size is only known on the GPU (live particle counts)."""
        data = uniforms.tobytes() if isinstance(uniforms, Uniforms) else (uniforms or b'\0' * 16)
        arena = self.gpu.arena
        if not arena.fits(len(data)):
            self.submit(restart=True)
        off = arena.push(data)
        if self.cp is None:
            self.cp = self.enc.begin_compute_pass()
        g, _ = kernel.bind(resources)
        self.cp.set_pipeline(kernel.pipe)
        self.cp.set_bind_group(0, g)
        self.cp.set_bind_group(1, arena.group, [off])
        self.cp.dispatch_workgroups_indirect(args.buf, offset)
        self.count += 1

    def clear_buffer(self, buf: 'Buffer', offset=0, size=None):
        self._end_pass()
        self.enc.clear_buffer(buf.buf, offset, size)

    def run_program(self, prog: 'Program'):
        """Replay a pre-bound sequence of dispatches (see Program)."""
        if self.cp is None:
            self.cp = self.enc.begin_compute_pass()
        cp = self.cp
        group = prog.group
        last = None
        for pipe, g0, off, wg in prog.items:
            if pipe is not last:
                cp.set_pipeline(pipe)
                last = pipe
            cp.set_bind_group(0, g0)
            cp.set_bind_group(1, group, [off])
            cp.dispatch_workgroups(*wg)
        self.count += len(prog.items)

    def render_pass(self, **desc):
        """Close the compute pass and open a render pass; the caller must call .end() on it."""
        self._end_pass()
        return self.enc.begin_render_pass(**desc)

    def uniform_offset(self, uniforms):
        data = uniforms.tobytes() if isinstance(uniforms, Uniforms) else uniforms
        if not self.gpu.arena.fits(len(data)):
            self.submit(restart=True)
        return self.gpu.arena.push(data)

    def copy_buffer(self, src: 'Buffer', dst: 'Buffer', src_offset=0, dst_offset=0, size=None):
        self._end_pass()
        self.enc.copy_buffer_to_buffer(src.buf, src_offset, dst.buf, dst_offset, size if size is not None else src.size)

    def copy_texture(self, src: Texture, dst: Texture, size=None, src_origin=(0, 0, 0), dst_origin=(0, 0, 0)):
        self._end_pass()
        self.enc.copy_texture_to_texture({'texture': src.tex, 'origin': src_origin, 'mip_level': 0},
                                         {'texture': dst.tex, 'origin': dst_origin, 'mip_level': 0},
                                         size or src.size)

    def submit(self, restart=False):
        self._end_pass()
        self.gpu.arena.flush(self.gpu.queue)
        self.gpu.queue.submit([self.enc.finish()])
        if restart:
            self.enc = self.gpu.device.create_command_encoder()


class Program:
    """A fixed sequence of dispatches whose resources and parameters never change (an iterative
    solver's inner loop). Everything is bound once and the parameter blocks live in their own
    uniform buffer, so replaying it costs only the raw dispatch calls.

    ops: (kernel, resources, uniforms, size) or (kernel, resources, uniforms, None, groups)."""

    def __init__(self, gpu, ops):
        self.gpu = gpu
        blocks, self.items, self._keep = [], [], []
        off = 0
        for op in ops:
            kernel, resources, uniforms, size = op[:4]
            groups = op[4] if len(op) > 4 else None
            data = uniforms.tobytes() if isinstance(uniforms, Uniforms) else (uniforms or b'\0' * 16)
            g0, keep = kernel.bind(resources)
            self._keep.append(keep)
            wg = groups or kernel.groups_for(size)
            if min(wg) <= 0:
                continue
            blocks.append((off, data))
            self.items.append((kernel.pipe, g0, off, tuple(wg)))
            off += ceil_div(max(len(data), 16), UniformArena.ALIGN) * UniformArena.ALIGN
        size = off + UniformArena.WINDOW
        cpu = bytearray(size)
        for o, d in blocks:
            cpu[o:o + len(d)] = d
        dev = gpu.device
        self.buf = dev.create_buffer(size=size, usage=BU.UNIFORM | BU.COPY_DST, label='program-uniforms')
        gpu.queue.write_buffer(self.buf, 0, cpu)
        self.group = dev.create_bind_group(layout=gpu.arena.layout, entries=[
            {'binding': 0, 'resource': {'buffer': self.buf, 'offset': 0, 'size': UniformArena.WINDOW}}])

    def __len__(self):
        return len(self.items)

    def destroy(self):
        try:
            self.buf.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------------------------------
# The card's memory
# ---------------------------------------------------------------------------------------------

MEMORY_ENV = 'BLACKBODY_GPU_MEMORY'   # GB: take the card to have this much memory (in place of what the system says)
PLAN_SHARE = 0.8        # of the card's memory, the share a scene is planned to fill (the rest: the desktop, other programs)
PLAN_RESERVE = 0.4e9    # and what a scene's estimate leaves of that for what it does not count: shaders, meshes, fabric,
#                         grass, matter, Lume, and the driver's allocations in blocks of 256 MB
MIN_PLAN = 0.25e9       # the least a scene is ever planned within
UNIFIED_SHARE = 0.65    # of the computer's memory a GPU that shares it (Apple silicon) may use: Metal's working set
OOM_SHARE = 0.75        # after an allocation the card could not make: plan within this share of what it then held


def card_memory(info):
    """What the system says the card has: {'total': bytes on the card, 'budget': bytes it lets this process use, 'free':
    bytes no process is using now, 'source': where that was read}, or None when it cannot be read. `info` is the
    adapter's info (vendor_id, device_id, adapter_type, backend_type). BLACKBODY_GPU_MEMORY (in GB) stands in for all."""
    env = os.environ.get(MEMORY_ENV, '').strip()
    if env:
        try:
            b = float(env) * GB
            if b > 0:
                return {'total': b, 'budget': b, 'free': b, 'source': MEMORY_ENV}
        except ValueError:
            log.warning('%s=%r is not a number of GB; reading the card instead', MEMORY_ENV, env)
    for read in (_dxgi_memory, _nvml_memory, _sysfs_memory, _unified_memory):
        m = _read(read, info)
        if m and m.get('total', 0) > 0:
            m['budget'] = min(float(m.get('budget') or m['total']), float(m['total']))
            if 'free' not in m:
                # (DXGI's budget leaves out what other programs hold on the card; NVML says what is free)
                other = _read(_nvml_memory, info) if read is not _nvml_memory else None
                m['free'] = min(other['free'], m['budget']) if other else m['budget']
            return m
    return None


def card_size(total):
    """The card's memory in whole GB (bytes): what the system says, rounded (DXGI and NVML differ by a few hundred MB on
    one card, and the plan must not), unless that is under a GB."""
    return float(round(total / GB) * GB) if total >= 0.75 * GB else float(total)


def plan_for(memory):
    """What a scene's estimate is fitted within on a card with `memory` (card_memory; None: not known): PLAN_SHARE of the
    card's size less PLAN_RESERVE. From the size alone, not what is free as the app starts: the plan shapes the
    simulation (its layout and detail, so its disk cache), which must come out the same on every start. A size given
    with BLACKBODY_GPU_MEMORY is taken as it is (not rounded to whole GB as a card's reading is)."""
    if not memory:
        return None
    size = memory['total'] if memory.get('source') == MEMORY_ENV else card_size(memory['total'])
    return max(PLAN_SHARE * size - PLAN_RESERVE, MIN_PLAN)


def _read(read, info):
    try:
        return read(info)
    except Exception as ex:   # (a reader that fails is just not the one that knows)
        log.debug('%s: %s', read.__name__, ex)
        return None


def _com(obj, index, restype, *argtypes):
    """Method `index` of a COM object's table (ctypes)."""
    import ctypes
    table = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(table[index])


def _dxgi_memory(info):
    """Windows: the adapter's memory from DXGI (any API's adapter: matched by vendor and device id), and the budget the
    system gives this process on it now (IDXGIAdapter3.QueryVideoMemoryInfo). An integrated GPU's memory is shared."""
    if sys.platform != 'win32':
        return None
    import ctypes

    class GUID(ctypes.Structure):
        _fields_ = [('a', ctypes.c_uint32), ('b', ctypes.c_uint16), ('c', ctypes.c_uint16), ('d', ctypes.c_ubyte * 8)]

    def guid(s):
        s = s.replace('-', '')
        return GUID(int(s[:8], 16), int(s[8:12], 16), int(s[12:16], 16), (ctypes.c_ubyte * 8)(*bytes.fromhex(s[16:])))

    class Desc1(ctypes.Structure):
        _fields_ = [('description', ctypes.c_wchar * 128), ('vendor', ctypes.c_uint), ('device', ctypes.c_uint),
                    ('subsys', ctypes.c_uint), ('revision', ctypes.c_uint), ('dedicated', ctypes.c_size_t),
                    ('dedicated_system', ctypes.c_size_t), ('shared', ctypes.c_size_t), ('luid_lo', ctypes.c_uint32),
                    ('luid_hi', ctypes.c_int32), ('flags', ctypes.c_uint)]

    class VideoMemory(ctypes.Structure):
        _fields_ = [('budget', ctypes.c_uint64), ('usage', ctypes.c_uint64), ('reservable', ctypes.c_uint64),
                    ('reserved', ctypes.c_uint64)]

    PP = ctypes.POINTER(ctypes.c_void_p)
    release = lambda o: _com(o, 2, ctypes.c_ulong)(o)
    factory = ctypes.c_void_p()
    if ctypes.WinDLL('dxgi').CreateDXGIFactory1(ctypes.byref(guid('770aae78-f26f-4dba-a829-253c83d1b387')),
                                                ctypes.byref(factory)) != 0:
        return None
    try:
        i = 0
        while True:
            ad = ctypes.c_void_p()
            if _com(factory, 12, ctypes.c_long, ctypes.c_uint, PP)(factory, i, ctypes.byref(ad)) != 0:   # EnumAdapters1
                return None
            i += 1
            try:
                d = Desc1()
                _com(ad, 10, ctypes.c_long, ctypes.POINTER(Desc1))(ad, ctypes.byref(d))   # GetDesc1
                if d.flags & 2 or d.vendor != int(info.get('vendor_id', -1)) or d.device != int(info.get('device_id', -1)):
                    continue   # (a software adapter, or another card)
                total = float(d.dedicated)
                if info.get('adapter_type') == 'IntegratedGPU' or d.dedicated < 1 << 30:
                    total += float(d.shared)
                out = {'total': total, 'budget': total, 'source': 'DXGI'}
                a3 = ctypes.c_void_p()
                if _com(ad, 0, ctypes.c_long, ctypes.POINTER(GUID), PP)(
                        ad, ctypes.byref(guid('645967a4-1392-4310-a798-8053ce3e93fd')), ctypes.byref(a3)) == 0:
                    try:
                        vm = VideoMemory()
                        if _com(a3, 14, ctypes.c_long, ctypes.c_uint, ctypes.c_int, ctypes.POINTER(VideoMemory))(
                                a3, 0, 0, ctypes.byref(vm)) == 0 and vm.budget > 0:   # QueryVideoMemoryInfo, local
                            out['budget'] = float(vm.budget)
                    finally:
                        release(a3)
                return out
            finally:
                release(ad)
    finally:
        release(factory)


def _nvml_memory(info):
    """An NVIDIA card (Linux, or Windows without DXGI's answer): its memory and how much is free now, from NVML."""
    if int(info.get('vendor_id', 0)) != 0x10DE:
        return None
    import ctypes
    lib = ctypes.WinDLL('nvml.dll') if sys.platform == 'win32' else ctypes.CDLL('libnvidia-ml.so.1')

    class Pci(ctypes.Structure):
        _fields_ = [('bus_legacy', ctypes.c_char * 16), ('domain', ctypes.c_uint), ('bus', ctypes.c_uint),
                    ('device', ctypes.c_uint), ('pci_device_id', ctypes.c_uint), ('subsystem', ctypes.c_uint),
                    ('bus_id', ctypes.c_char * 32)]

    class Mem(ctypes.Structure):
        _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]

    if lib.nvmlInit_v2() != 0:
        return None
    try:
        n = ctypes.c_uint()
        if lib.nvmlDeviceGetCount_v2(ctypes.byref(n)) != 0:
            return None
        want = (int(info.get('device_id', 0)) << 16) | 0x10DE
        for i in range(n.value):
            h, pci, mem = ctypes.c_void_p(), Pci(), Mem()
            if (lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)) != 0 or lib.nvmlDeviceGetPciInfo_v3(h, ctypes.byref(pci)) != 0
                    or pci.pci_device_id != want or lib.nvmlDeviceGetMemoryInfo(h, ctypes.byref(mem)) != 0):
                continue
            return {'total': float(mem.total), 'budget': float(mem.total), 'free': float(mem.free), 'source': 'NVML'}
        return None
    finally:
        lib.nvmlShutdown()


def _sysfs_memory(info):
    """Linux, AMD (and others whose driver says): the card's memory and what is in use, from /sys/class/drm."""
    if not sys.platform.startswith('linux'):
        return None
    for dev in sorted(Path('/sys/class/drm').glob('card[0-9]*/device')):
        try:
            if (int((dev / 'vendor').read_text(), 16) != int(info.get('vendor_id', -1))
                    or int((dev / 'device').read_text(), 16) != int(info.get('device_id', -1))):
                continue
            total = float((dev / 'mem_info_vram_total').read_text())
            used = float((dev / 'mem_info_vram_used').read_text())
        except (OSError, ValueError):
            continue
        return {'total': total, 'budget': total, 'free': max(total - used, 0.0), 'source': 'sysfs'}
    return None


def _unified_memory(info):
    """Apple silicon (Metal, integrated): the GPU shares the computer's memory, and may use about UNIFIED_SHARE of it."""
    if info.get('backend_type') != 'Metal' or info.get('adapter_type') != 'IntegratedGPU':
        return None
    ram = float(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'))
    return {'total': ram * UNIFIED_SHARE, 'budget': ram * UNIFIED_SHARE, 'source': 'unified memory'}


_CARD = {'plan': None, 'card_plan': None, 'side': None, 'name': None}   # the GPU in use (the last one made): see card()


def card():
    """The GPU in use, for what has no GPU at hand (the scene's memory plan): {'plan': bytes a scene's estimate may take
    (None: the card's memory is not known), 'card_plan': the plan from the card's size alone (plan_for), which a plan
    cut after running out of memory leaves as it was, 'side': the largest 3-D texture's side (None before a GPU is
    made), 'name'}."""
    return dict(_CARD)


class GPU:
    """Owns the adapter, device, samplers and the kernel cache, and counts the memory its textures and buffers take."""

    def __init__(self, adapter_name=None, backend=None, power='high-performance'):
        self.adapter = self._pick_adapter(adapter_name, backend, power)
        info = self.adapter.info
        self.name = info.get('device', 'GPU')
        self.backend = info.get('backend_type', '')
        self.allocated = 0          # bytes in the textures and buffers made through this GPU (and not yet freed)
        self.memory = card_memory(info)
        # a scene's estimate is fitted within `plan` (from the card's size: plan_for); past `ceiling` an allocation is
        # refused: what the system gives the app as it starts, but never less than the plan promises
        self.plan = self.card_plan = plan_for(self.memory)   # (`plan` is cut after running out of memory: out_of_memory)
        self.ceiling = max(self.memory['budget'], self.plan + PLAN_RESERVE) if self.memory else None
        self._lost_at = _LostWatch.count
        feats = set(self.adapter.features)
        want = [f for f in ('float32-filterable', 'timestamp-query') if f in feats]
        self.float32_filterable = 'float32-filterable' in feats
        alim = self.adapter.limits
        # counts: ask for what the kernels need; sizes: take whatever the adapter offers
        wanted = {'max-storage-textures-per-shader-stage': 8, 'max-sampled-textures-per-shader-stage': 24,
                  'max-storage-buffers-per-shader-stage': 16, 'max-storage-buffer-binding-size': None,
                  'max-buffer-size': None, 'max-texture-dimension-2d': None, 'max-texture-dimension-3d': None,
                  'max-compute-workgroup-storage-size': None}
        limits = {}
        for k, v in wanted.items():
            if k in alim:
                limits[k] = int(alim[k]) if v is None else min(int(alim[k]), v)
        self.device = self.adapter.request_device_sync(required_features=want, required_limits=limits, label='blackbody')
        self.queue = self.device.queue
        self.limits = dict(self.device.limits)
        self.arena = UniformArena(self.device)
        d = self.device
        self.linear = d.create_sampler(mag_filter='linear', min_filter='linear', mipmap_filter='nearest',
                                       address_mode_u='clamp-to-edge', address_mode_v='clamp-to-edge', address_mode_w='clamp-to-edge')
        self.nearest = d.create_sampler(mag_filter='nearest', min_filter='nearest',
                                        address_mode_u='clamp-to-edge', address_mode_v='clamp-to-edge', address_mode_w='clamp-to-edge')
        self.repeat = d.create_sampler(mag_filter='linear', min_filter='linear', mipmap_filter='nearest',
                                       address_mode_u='repeat', address_mode_v='repeat', address_mode_w='repeat')
        self._kernels = {}
        self._by_shader = {}     # shader key -> (layout0, pipe) made on this device, by any thread (the precompile's too)
        self.on_compile = None   # called with the shader file before a new kernel compiles (the app shows it)
        self.on_wait = None      # called with it when another thread (the background precompile) is compiling it
        self.adapter_key = adapter_key(info)
        self.record = shader_record(self.adapter_key)   # what this adapter and driver have compiled (or None)
        self.vel_format ='rgba32float' if self.float32_filterable else 'rgba16float'
        self.max_side = int(self.limits.get('max-texture-dimension-3d', 2048))
        _CARD.update(plan=self.plan, card_plan=self.card_plan, side=self.max_side, name=self.name)
        m = self.memory
        log.info('GPU: %s (%s), %s', self.name, self.backend,
                 f'{m["total"] / GB:.1f} GB, {m["free"] / GB:.1f} GB free; scenes fitted within {self.plan / GB:.1f} GB '
                 f'({m["source"]})' if m else 'memory unknown')

    # -- memory -------------------------------------------------------------------------------

    def take(self, nbytes, label=''):
        """Count an allocation in, or refuse it (GPUOutOfMemory) when it would take more than the system gives the app:
        a request past that can bring the device down, or page the grids out to the computer's memory and crawl."""
        if self.ceiling is not None and self.allocated + nbytes > self.ceiling:
            raise GPUOutOfMemory(nbytes, self.allocated, self.ceiling, label)
        self.allocated += nbytes

    def room(self):
        """Bytes still free of what the app plans to use (the plan with its reserve, less after running out of memory),
        for choosing how a growing box is copied (Solver.grow). None: not known."""
        return None if self.plan is None else max(self.plan + PLAN_RESERVE - self.allocated, 0.0)

    def out_of_memory(self, ex=None):
        """An allocation failed (GPUOutOfMemory `ex`): from now on, plan the scene's grids within OOM_SHARE of what the
        card then held, and of the plan before (so each failure asks less), so the engine made again fits
        (scene/model.py memory_plan; a layout a disk cache holds, which fits card_plan, is kept). Returns the new plan
        (bytes)."""
        held = float(self.allocated)
        if isinstance(ex, GPUOutOfMemory):
            held = float(ex.allocated + ex.wanted)
            if not ex.driver and self.ceiling is not None:
                held = min(held, self.ceiling)
        cut = OOM_SHARE * held - PLAN_RESERVE
        if self.plan is not None:
            cut = min(cut, OOM_SHARE * self.plan)
        self.plan = max(cut, MIN_PLAN)
        _CARD.update(plan=self.plan)
        return self.plan

    def keep_plan(self, old):
        """Plan within no more than another GPU did (`old`, one whose device was lost: ui/worker.py makes this one in its
        place), so a plan cut after running out of memory stays cut and the scene is laid out as it was."""
        if old.plan is not None and (self.plan is None or old.plan < self.plan):
            self.plan = old.plan
            _CARD.update(plan=self.plan)

    def forget_bindings(self):
        """Drop the kernels' cached bind groups (they keep the textures in them alive): before what an engine made is
        let go, so its memory comes back."""
        for k in self._kernels.values():
            k._groups.clear()

    def alive(self):
        """Whether the device still works: a lost one fails the smallest request."""
        try:
            b = self.device.create_buffer(size=16, usage=BU.COPY_DST | BU.COPY_SRC)
            self.queue.read_buffer(b, 0, 4)
            b.destroy()
            return True
        except Exception:
            return False

    def lost_reason(self):
        """What the driver said when a device was last lost since this GPU was made ('' if nothing was: wgpu only logs it)."""
        return _LostWatch.text if _LostWatch.count != self._lost_at else ''

    @staticmethod
    def adapters():
        out = []
        for a in wgpu.gpu.enumerate_adapters_sync():
            i = a.info
            out.append({'name': i.get('device', '?'), 'backend': i.get('backend_type', '?'), 'type': i.get('adapter_type', '?')})
        return out

    @staticmethod
    def _pick_adapter(name, backend, power):
        name = name or os.environ.get('BLACKBODY_GPU')
        backend = backend or os.environ.get('BLACKBODY_BACKEND')
        if name or backend:
            for a in wgpu.gpu.enumerate_adapters_sync():
                i = a.info
                if name and name.lower() not in i.get('device', '').lower():
                    continue
                if backend and backend.lower() != i.get('backend_type', '').lower():
                    continue
                if i.get('adapter_type') == 'CPU':
                    continue
                return a
            log.warning('No adapter matched name=%r backend=%r; using the default', name, backend)
        a = wgpu.gpu.request_adapter_sync(power_preference=power)
        if a is None:
            raise GPUUnavailable('No GPU adapter found. Blackbody needs a GPU with Vulkan, Direct3D 12 or Metal.')
        return a

    # -- resources ----------------------------------------------------------------------------

    def texture3d(self, size, fmt, label='', usage=DEFAULT_USAGE):
        return Texture(self, size, fmt, '3d', usage, label)

    def texture2d(self, w, h, fmt, label='', usage=DEFAULT_USAGE):
        return Texture(self, (w, h, 1), fmt, '2d', usage, label)

    def buffer(self, size, label='', usage=BU.STORAGE | BU.COPY_DST | BU.COPY_SRC):
        return Buffer(self, size, usage, label)

    def kernel(self, source_file, bindings, entry='main', defines=None, workgroup=(8, 8, 4), quiet=False):
        """The kernel, compiled the first time it is asked for. quiet: neither on_compile nor on_wait is told (the
        background precompile, which no one is waiting on). Its thread may ask for a kernel as the app does: the first
        one kept is the one every caller gets."""
        key = (source_file, entry, tuple(bindings), tuple(sorted((defines or {}).items())), workgroup)
        k = self._kernels.get(key)
        if k is None:
            if self.on_compile is not None and not quiet:
                self.on_compile(source_file)
            k = self._kernels.setdefault(key, Kernel(self, source_file, bindings, entry, defines, workgroup,
                                                     quiet=quiet))
        return k

    def batch(self):
        return Batch(self)

    # -- transfers ----------------------------------------------------------------------------

    def upload(self, tex: Texture, array):
        dtype, ch, bpt = FORMATS[tex.format]
        a = np.ascontiguousarray(array, dtype=dtype)
        w, h, d = tex.size
        self.queue.write_texture({'texture': tex.tex, 'origin': (0, 0, 0), 'mip_level': 0}, a,
                                 {'offset': 0, 'bytes_per_row': w * bpt, 'rows_per_image': h}, (w, h, d))

    def read(self, tex: Texture, origin=(0, 0, 0), size=None):
        """Read a texture (or a box of it) back as (d, h, w, c) for 3D or (h, w, c) for 2D."""
        dtype, ch, bpt = FORMATS[tex.format]
        w, h, d = size or tex.size
        data = self.queue.read_texture({'texture': tex.tex, 'origin': origin, 'mip_level': 0},
                                       {'offset': 0, 'bytes_per_row': w * bpt, 'rows_per_image': h}, (w, h, d))
        arr = np.frombuffer(data, dtype=dtype).reshape(d, h, w, ch)
        return arr[0] if tex.dim == '2d' else arr

    def read_buffer(self, buf: Buffer, size=None, offset=0):
        return self.queue.read_buffer(buf.buf, offset, size)

    def write_buffer(self, buf: Buffer, data, offset=0):
        self.queue.write_buffer(buf.buf, offset, np.ascontiguousarray(data))

    def sync(self):
        """Block until the GPU has finished everything submitted so far."""
        tmp = getattr(self, '_sync_buf', None)
        if tmp is None:
            tmp = self._sync_buf = self.device.create_buffer(size=16, usage=BU.COPY_DST | BU.COPY_SRC)
        self.queue.read_buffer(tmp, 0, 4)
