"""Footage reader: video files (anything FFmpeg decodes), image sequences and still images.

Frames come back as (h, w, 4) arrays: uint8 for 8-bit display-referred sources, float16 for deeper
sources (10/12-bit video, 16-bit PNG/TIFF) and for EXR (scene-linear).
"""
from __future__ import annotations

import logging
import re
import threading
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path

import numpy as np

log = logging.getLogger('blackbody.footage')

VIDEO_EXT = {'.mov', '.mp4', '.m4v', '.mkv', '.avi', '.mxf', '.webm', '.mts', '.m2ts', '.wmv', '.mpg', '.mpeg', '.flv',
             '.3gp', '.ts', '.y4m', '.gif'}
IMAGE_EXT = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.exr', '.dpx', '.bmp', '.tga', '.webp'}
_DIGITS = re.compile(r'^(.*?)(\d+)$')


def _rgba8(a):
    if a.ndim == 2:
        a = np.stack([a, a, a], -1)
    if a.shape[2] == 3:
        a = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, a.dtype)], -1)
    return np.ascontiguousarray(a)


def _to_half(a, maxv):
    a = a.astype(np.float32) / maxv
    if a.ndim == 2:
        a = np.stack([a, a, a], -1)
    if a.shape[2] == 3:
        a = np.concatenate([a, np.ones(a.shape[:2] + (1,), np.float32)], -1)
    return np.ascontiguousarray(a.astype(np.float16))


def find_sequence(path: Path):
    """Given one file of a numbered sequence, return (frame numbers, path pattern using {:0Nd})."""
    stem, ext = path.stem, path.suffix
    m = _DIGITS.match(stem)
    if not m:
        return None
    prefix, digits = m.group(1), m.group(2)
    rx = re.compile('^' + re.escape(prefix) + r'(\d+)' + re.escape(ext) + '$', re.I)
    nums = []
    for p in path.parent.iterdir():
        mm = rx.match(p.name)
        if mm:
            nums.append(int(mm.group(1)))
    if len(nums) < 2:
        return None
    nums.sort()
    pad = len(digits) if digits.startswith('0') or len(set(len(str(n)) for n in nums)) > 1 else 0
    return nums, str(path.parent / (prefix + '{:0%dd}' % pad + ext)) if pad else str(path.parent / (prefix + '{:d}' + ext))


def read_image(path):
    """Read one image file into RGBA uint8 or float16."""
    path = str(path)
    ext = Path(path).suffix.lower()
    if ext == '.exr':
        import OpenEXR
        with OpenEXR.File(path) as f:
            ch = f.channels()
            if 'RGBA' in ch:
                a = ch['RGBA'].pixels
            elif 'RGB' in ch:
                rgb = ch['RGB'].pixels
                a = np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,), rgb.dtype)], -1)
            else:
                first = next(iter(ch.values())).pixels
                a = np.stack([first] * 3 + [np.ones_like(first)], -1) if first.ndim == 2 else first
        return np.ascontiguousarray(a.astype(np.float16))
    if ext == '.dpx':
        import av
        with av.open(path) as c:
            fr = next(c.decode(video=0))
            a = fr.to_ndarray(format='rgba64le')
        return _to_half(a, 65535.0)
    from PIL import Image
    im = Image.open(path)
    if im.mode in ('I;16', 'I;16B', 'I;16L', 'I'):
        return _to_half(np.asarray(im, np.uint16 if im.mode != 'I' else np.int32).astype(np.float32), 65535.0)
    arr = np.asarray(im)
    if arr.dtype == np.uint16:
        return _to_half(arr, 65535.0)
    im = im.convert('RGBA')
    return np.ascontiguousarray(np.asarray(im))


class Footage:
    """A video, image sequence or still, read frame by frame with a small decoded-frame cache."""

    def __init__(self, path, cache_frames=48):
        self.path = str(path)
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(self.path)
        self._lock = threading.RLock()
        self._cache = OrderedDict()
        self._cache_max = cache_frames
        self.audio = False
        self.linear = False
        self.deep = False
        self.codec = ''
        self.rotation = 0
        self.first_number = 0
        ext = p.suffix.lower()
        if ext in VIDEO_EXT:
            self.kind = 'video'
            self._open_video()
        elif ext in IMAGE_EXT:
            seq = find_sequence(p)
            if seq:
                self.kind = 'sequence'
                self._nums, self._pattern = seq
                self.first_number = self._nums[0]
                self.frames = len(self._nums)
                self.fps = 24.0
            else:
                self.kind = 'still'
                self.frames = 1
                self.fps = 24.0
            first = self._read_file(0)
            self.height, self.width = first.shape[:2]
            self.linear = ext == '.exr'
            self.deep = first.dtype != np.uint8
            self._put(0, first)
            self.codec = ext[1:].upper()
        else:
            raise ValueError(f'Unsupported footage type: {ext}')

    # -- video ----------------------------------------------------------------------------------

    def _open_video(self):
        import av
        self._av = av
        self._c = av.open(self.path)
        vs = [s for s in self._c.streams.video]
        if not vs:
            raise ValueError('No video stream in this file')
        self._s = vs[0]
        self._s.thread_type = 'AUTO'
        s = self._s
        rate = s.average_rate or s.guessed_rate or Fraction(24, 1)
        self.fps = float(rate)
        self._tb = float(s.time_base) if s.time_base else 1.0 / self.fps
        self._start = float(s.start_time * s.time_base) if s.start_time is not None and s.time_base else 0.0
        dur = None
        if s.duration and s.time_base:
            dur = float(s.duration * s.time_base)
        elif self._c.duration:
            dur = self._c.duration / 1e6
        self.frames = int(s.frames) if s.frames else max(1, int(round((dur or 1.0) * self.fps)))
        cc = s.codec_context
        self.width, self.height = cc.width, cc.height
        self.codec = cc.name
        pf = str(cc.pix_fmt or '')
        self.deep = any(b in pf for b in ('10', '12', '16', 'p16', '48', '64'))
        self.audio = len(self._c.streams.audio) > 0
        self._next_index = None
        self._dec = None
        rot = 0
        try:
            r = s.metadata.get('rotate')
            if r:
                rot = int(float(r)) % 360
        except Exception:
            pass
        self.rotation = rot
        # decode the first frame to settle size (rotation) and sanity-check
        first = self._decode_to(0)
        self.height, self.width = first.shape[:2]

    def _frame_array(self, fr):
        if self.deep:
            a = fr.to_ndarray(format='rgba64le')
            a = _to_half(a, 65535.0)
        else:
            a = fr.to_ndarray(format='rgba')
        rot = self.rotation
        try:
            if getattr(fr, 'rotation', 0):
                rot = int(fr.rotation) % 360
        except Exception:
            pass
        if rot:
            # rotation is counter-clockwise in the display matrix convention
            k = {90: 1, 180: 2, 270: 3}.get(rot, 0)
            a = np.ascontiguousarray(np.rot90(a, k))
        return a

    def _index_of(self, fr):
        t = float(fr.pts * fr.time_base) if fr.pts is not None and fr.time_base else (fr.time or 0.0)
        return int(round((t - self._start) * self.fps))

    def _decode_to(self, index):
        """Decode forward to `index`, seeking first unless we are just behind it."""
        if self._next_index is None or index < self._next_index or index > self._next_index + 12:
            ts = int((self._start + max(index - 1, 0) / self.fps) / self._tb)
            self._c.seek(ts, stream=self._s, backward=True, any_frame=False)
            self._dec = self._c.decode(self._s)
            self._next_index = -1
        last = None
        for fr in self._dec:
            i = self._index_of(fr)
            self._next_index = i + 1
            if i >= index - 2:
                arr = self._frame_array(fr)
                self._put(i, arr)
                last = (i, arr)
            if i >= index:
                return arr
        # ran off the end: return the last frame we saw (or the nearest cached)
        if last is not None:
            return last[1]
        near = min(self._cache, key=lambda k: abs(k - index)) if self._cache else None
        if near is not None:
            return self._cache[near]
        raise IndexError(f'frame {index} not found')

    # -- sequences --------------------------------------------------------------------------------

    def _read_file(self, index):
        if self.kind == 'still':
            return read_image(self.path)
        return read_image(self._pattern.format(self._nums[index]))

    # -- cache ------------------------------------------------------------------------------------

    def _put(self, i, arr):
        self._cache[i] = arr
        self._cache.move_to_end(i)
        while len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)

    def read(self, index):
        """Frame `index` (0-based, clamped to the clip)."""
        index = int(min(max(index, 0), self.frames - 1))
        with self._lock:
            a = self._cache.get(index)
            if a is not None:
                self._cache.move_to_end(index)
                return a
            if self.kind == 'video':
                return self._decode_to(index)
            a = self._read_file(index)
            self._put(index, a)
            return a

    def describe(self):
        kind = {'video': 'Video', 'sequence': 'Image sequence', 'still': 'Still image'}[self.kind]
        depth = 'float' if self.linear else ('10+ bit' if self.deep else '8 bit')
        return (f'{kind} · {self.width}×{self.height} · {self.fps:.3f} fps · {self.frames} frames · {self.codec} · {depth}'
                + (' · audio' if self.audio else ''))

    def close(self):
        with self._lock:
            if self.kind == 'video' and getattr(self, '_c', None) is not None:
                try:
                    self._c.close()
                except Exception:
                    pass
                self._c = None
            self._cache.clear()
