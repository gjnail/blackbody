"""The simulation cache on disk.

Every simulated frame the engine caches (a dict of numpy arrays and small values: fire fields,
embers, liquid particles, and at checkpoint frames the whole solver state) can also be written to a
folder, one file per frame. Frames survive closing the app, a simulation can resume from the last
checkpoint instead of starting over, and a render farm can simulate once and render frame ranges on
many machines from the same folder (`blackbody simulate` / `blackbody render --from-cache`).

File layout: an 8-byte magic, a little-endian u32 header length, a JSON header (values, and for
each array its dtype, shape, offset and compressed length), then the arrays, each zlib-compressed.
Files are written to a temporary name and renamed, so a reader never sees half a frame.
"""
from __future__ import annotations

import json
import logging
import os
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

log = logging.getLogger('blackbody.simcache')

MAGIC = b'BBCACHE1'
LEVEL = 1  # zlib level: smoke is mostly empty space, so even the fastest level shrinks it a lot


def _encode(value, arrays):
    """JSON-able form of a value; numpy arrays are pulled out into `arrays` and referenced by index."""
    if isinstance(value, np.ndarray):
        arrays.append(value)
        return {'__array__': len(arrays) - 1}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            return {k: _encode(v, arrays) for k, v in value.items()}
        return {'__dict__': [[_encode(k, arrays), _encode(v, arrays)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {'__tuple__': [_encode(v, arrays) for v in value]}
    if isinstance(value, list):
        return [_encode(v, arrays) for v in value]
    return value


def _decode(value, arrays):
    if isinstance(value, dict):
        if '__array__' in value:
            return arrays[value['__array__']]
        if '__tuple__' in value:
            return tuple(_decode(v, arrays) for v in value['__tuple__'])
        if '__dict__' in value:
            return {_decode(k, arrays): _decode(v, arrays) for k, v in value['__dict__']}
        return {k: _decode(v, arrays) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v, arrays) for v in value]
    return value


def write_entry(path, entry):
    """Write one cache entry (dict) to `path`, atomically."""
    arrays = []
    body = _encode({k: v for k, v in entry.items() if not k.startswith('_')}, arrays)
    blobs, table, offset = [], [], 0
    for a in arrays:
        a = np.ascontiguousarray(a)
        z = zlib.compress(a.tobytes(), LEVEL)
        table.append({'dtype': a.dtype.str, 'shape': list(a.shape), 'offset': offset, 'size': len(z)})
        blobs.append(z)
        offset += len(z)
    head = json.dumps({'entry': body, 'arrays': table}).encode('utf-8')
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<I', len(head)))
        f.write(head)
        for z in blobs:
            f.write(z)
    os.replace(tmp, path)


def read_entry(path):
    """Read a cache entry written by write_entry."""
    with open(path, 'rb') as f:
        data = f.read()
    if data[:8] != MAGIC:
        raise ValueError(f'{path} is not a Blackbody cache file')
    n = struct.unpack_from('<I', data, 8)[0]
    head = json.loads(data[12:12 + n].decode('utf-8'))
    base = 12 + n
    arrays = []
    for t in head['arrays']:
        raw = zlib.decompress(data[base + t['offset']: base + t['offset'] + t['size']])
        arrays.append(np.frombuffer(raw, dtype=np.dtype(t['dtype'])).reshape(t['shape']).copy())
    return _decode(head['entry'], arrays)


class CacheMismatch(ValueError):
    pass


_META = {}   # meta.json read, by path: ((modified, size), its contents)


def read_meta(folder):
    """A cache folder's meta.json ({} when there is none): its signature and, from caches that record it, the layout the
    scene was fitted to the GPU at (scene/model.py Scene.cache_fit). Read again only when the file changes."""
    p = Path(folder) / 'meta.json'
    try:
        st = p.stat()
    except OSError:
        return {}
    stamp = (st.st_mtime_ns, st.st_size)
    hit = _META.get(str(p))
    if hit is not None and hit[0] == stamp:
        return hit[1]
    try:
        meta = json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        meta = {}
    _META[str(p)] = (stamp, meta)
    return meta


def has_frames(folder):
    """Whether a cache folder holds any simulated frame (looks no further than the first)."""
    try:
        with os.scandir(folder) as it:
            return any(e.name.startswith('f') and e.name.endswith('.bbc') for e in it)
    except OSError:
        return False


class SimCache:
    """A folder of cached frames for one simulation (one signature). Read-only caches (render farm
    machines rendering what another machine simulated) never write, and refuse a folder simulated
    with other settings instead of clearing it. `about` goes into meta.json beside the signature
    (the layout the scene was fitted to the GPU at, and what it asked for)."""

    def __init__(self, folder, signature=None, readonly=False, about=None):
        self.folder = Path(folder)
        self.readonly = bool(readonly)
        if not self.readonly:
            self.folder.mkdir(parents=True, exist_ok=True)
        meta = self.folder / 'meta.json'
        if signature is not None:
            old = json.loads(meta.read_text(encoding='utf-8')) if meta.exists() else {}
            if old.get('signature') not in (None, signature):
                if self.readonly:
                    raise CacheMismatch(f'The disk cache in {self.folder} was simulated with other settings; '
                                        'render with the same project and settings, or simulate again.')
                # the settings changed since these frames were simulated: they no longer apply
                self.clear()
            if not self.readonly:
                meta.write_text(json.dumps({'signature': signature, 'format': 1, **(about or {})}), encoding='utf-8')
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='simcache')
        self._pending = {}
        self._lock = threading.Lock()

    def _path(self, frame):
        return self.folder / f'f{int(frame):+08d}.bbc'

    def put(self, frame, entry, wait=False):
        """Write a frame in the background (the entry must not be changed afterwards)."""
        if self.readonly:
            return
        with self._lock:
            fut = self._pool.submit(self._write, frame, entry)
            self._pending[frame] = fut
        if wait:
            fut.result()

    def _write(self, frame, entry):
        try:
            write_entry(self._path(frame), entry)
        except Exception as ex:  # a full disk must not stop the simulation
            log.warning('Could not write frame %s to the disk cache: %s', frame, ex)
        finally:
            with self._lock:
                self._pending.pop(frame, None)

    def flush(self):
        with self._lock:
            futs = list(self._pending.values())
        for f in futs:
            f.result()

    def get(self, frame):
        with self._lock:
            fut = self._pending.get(frame)
        if fut is not None:
            fut.result()
        p = self._path(frame)
        if not p.exists():
            return None
        try:
            return read_entry(p)
        except Exception as ex:
            log.warning('Unreadable cache frame %s: %s', frame, ex)
            return None

    def __contains__(self, frame):
        return frame in self._pending or self._path(frame).exists()

    def frames(self):
        out = []
        for p in self.folder.glob('f*.bbc'):
            try:
                out.append(int(p.stem[1:]))
            except ValueError:
                pass
        with self._lock:
            out += list(self._pending)
        return sorted(set(out))

    def checkpoints(self):
        """Frames that hold a whole solver state (so a simulation can resume from them)."""
        return [f for f in self.frames() if (self.folder / f'c{int(f):+08d}').exists()]

    def mark_checkpoint(self, frame):
        if not self.readonly:
            (self.folder / f'c{int(frame):+08d}').touch()

    def clear(self):
        if self.readonly:
            return
        self.flush() if hasattr(self, '_pool') else None
        for p in list(self.folder.glob('f*.bbc')) + list(self.folder.glob('c*')) + list(self.folder.glob('*.tmp')):
            try:
                p.unlink()
            except OSError:
                pass

    def size_bytes(self):
        return sum(p.stat().st_size for p in self.folder.glob('f*.bbc'))


def default_root(project_path=None):
    """Where disk caches live: next to the project (<project>.bbcache), else the user cache folder."""
    env = os.environ.get('BLACKBODY_CACHE')
    if project_path:
        p = Path(project_path)
        return p.with_name(p.stem + '.bbcache')
    if env:
        return Path(env) / 'sims'
    if os.name == 'nt':
        return Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local')) / 'Blackbody' / 'sims'
    return Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'blackbody' / 'sims'
