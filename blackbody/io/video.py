"""Video writers through FFmpeg (PyAV): ProRes, DNxHR, H.264/H.265 and VP9, with the source audio
carried over from the footage when there is any."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

log = logging.getLogger('blackbody.video')


@dataclass(frozen=True)
class VideoProfile:
    label: str
    codec: str
    ext: str
    pix_fmt: str
    options: dict = field(default_factory=dict)
    alpha: bool = False
    deep: bool = False      # feed 16-bit RGB so 10-bit codecs get real precision
    fallback: str | None = None


PROFILES = {
    'prores4444': VideoProfile('ProRes 4444 with alpha (.mov)', 'prores_ks', '.mov', 'yuva444p10le',
                               {'profile': '4', 'vendor': 'apl0', 'bits_per_mb': '8000'}, alpha=True, deep=True),
    'prores422hq': VideoProfile('ProRes 422 HQ (.mov)', 'prores_ks', '.mov', 'yuv422p10le',
                                {'profile': '3', 'vendor': 'apl0'}, deep=True),
    'dnxhr_hq': VideoProfile('DNxHR HQ (.mov)', 'dnxhd', '.mov', 'yuv422p', {'profile': 'dnxhr_hq'}),
    'dnxhr_444': VideoProfile('DNxHR 444 (.mov)', 'dnxhd', '.mov', 'yuv444p10le', {'profile': 'dnxhr_444'}, deep=True),
    'h264': VideoProfile('H.264 (.mp4)', 'libx264', '.mp4', 'yuv420p', {'crf': '16', 'preset': 'slow'}, fallback='h264_nvenc'),
    'h265': VideoProfile('H.265 10-bit (.mp4)', 'libx265', '.mp4', 'yuv420p10le',
                         {'crf': '18', 'preset': 'medium', 'x265-params': 'log-level=error'}, deep=True),
    'vp9_alpha': VideoProfile('VP9 with alpha (.webm)', 'libvpx-vp9', '.webm', 'yuva420p',
                              {'crf': '20', 'b': '0', 'row-mt': '1', 'deadline': 'good'}, alpha=True),
}


def available_profiles():
    import av
    out = {}
    for k, p in PROFILES.items():
        try:
            av.codec.Codec(p.codec, 'w')
            out[k] = p
        except Exception:
            if p.fallback:
                try:
                    av.codec.Codec(p.fallback, 'w')
                    out[k] = p
                except Exception:
                    pass
    return out


class VideoWriter:
    def __init__(self, path, profile: str, width: int, height: int, fps: float, audio_source=None, audio_start=0.0,
                 audio_duration=None, quality=None):
        import av
        from av.video.reformatter import ColorRange, Colorspace
        self.av = av
        self.p = PROFILES[profile]
        self.path = Path(path)
        if self.path.suffix.lower() != self.p.ext:
            self.path = self.path.with_suffix(self.p.ext)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # even sizes for 4:2:x chroma
        if '420' in self.p.pix_fmt or '422' in self.p.pix_fmt:
            width, height = width - width % 2, height - height % 2
        self.width, self.height = width, height
        self.fps = fps
        rate = Fraction(fps).limit_denominator(1001)
        self.c = av.open(str(self.path), 'w')
        codec = self.p.codec
        try:
            av.codec.Codec(codec, 'w')
        except Exception:
            codec = self.p.fallback
        self.s = self.c.add_stream(codec, rate=rate)
        self.s.width, self.s.height = width, height
        self.s.pix_fmt = self.p.pix_fmt
        opts = dict(self.p.options)
        if quality is not None and 'crf' in opts:
            opts['crf'] = str(int(quality))
        if codec == 'h264_nvenc':
            opts = {'preset': 'p6', 'cq': '18', 'rc': 'vbr'}
        self.s.options = opts
        cc = self.s.codec_context
        try:
            cc.color_primaries = 1   # BT.709
            cc.color_trc = 1
            cc.colorspace = 1
            cc.color_range = 1       # studio range (MPEG)
        except Exception:
            pass
        if self.p.ext == '.mp4' and codec in ('libx265', 'hevc_nvenc'):
            try:
                cc.codec_tag = 'hvc1'  # plays in QuickTime and Apple apps
            except Exception:
                pass
        self._cs = Colorspace.ITU709
        self._cr = ColorRange.MPEG
        self.frame_index = 0
        self._audio = None
        if audio_source:
            try:
                self._setup_audio(audio_source, audio_start, audio_duration)
            except Exception as ex:  # audio is a nicety; never fail a render over it
                log.warning('Audio not copied: %s', ex)
                self._audio = None

    # -- audio -------------------------------------------------------------------------------

    def _setup_audio(self, src, start, duration):
        av = self.av
        ic = av.open(str(src))
        if not ic.streams.audio:
            ic.close()
            return
        ia = ic.streams.audio[0]
        name = ia.codec_context.name
        copy_ok = {'.mov': {'aac', 'alac', 'mp3', 'ac3', 'eac3'} | {n for n in [name] if name.startswith('pcm_')},
                   '.mp4': {'aac', 'mp3', 'ac3', 'eac3', 'alac', 'opus', 'flac'},
                   '.webm': {'opus', 'vorbis'}}[self.p.ext]
        if name in copy_ok:
            oa = self.c.add_stream_from_template(ia)
            mode = 'copy'
        else:
            # .mov masters (ProRes, DNxHR) get lossless 24-bit PCM, as editorial expects
            target = {'.mov': 'pcm_s24le', '.webm': 'libopus'}.get(self.p.ext, 'aac')
            oa = self.c.add_stream(target, rate=ia.codec_context.sample_rate or 48000)
            try:
                oa.layout = ia.codec_context.layout.name
            except Exception:
                pass
            mode = 'transcode'
        self._audio = {'ic': ic, 'ia': ia, 'oa': oa, 'mode': mode, 'start': float(start),
                       'end': float(start) + float(duration) if duration else None, 'it': None, 'pending': None,
                       'resampler': None, 'done': False}

    def _pump_audio(self, until_seconds):
        a = self._audio
        if a is None or a['done']:
            return
        av = self.av
        if a['it'] is None:
            ia = a['ia']
            try:
                a['ic'].seek(int(max(a['start'] - 0.5, 0) / float(ia.time_base)), stream=ia, backward=True)
            except Exception:
                pass
            a['it'] = a['ic'].demux(ia)
        ia, oa = a['ia'], a['oa']
        tb = float(ia.time_base)
        while True:
            pk = a['pending']
            if pk is None:
                try:
                    pk = next(a['it'])
                except StopIteration:
                    a['done'] = True
                    self._flush_audio_encoder()
                    return
                if pk.dts is None or pk.pts is None:
                    continue
            t = float(pk.pts) * tb
            if a['end'] is not None and t >= a['end']:
                a['done'] = True
                self._flush_audio_encoder()
                return
            if t - a['start'] > until_seconds:
                a['pending'] = pk
                return
            a['pending'] = None
            if a['mode'] == 'copy':
                if t < a['start'] - 1e-6:
                    continue
                shift = int(round(a['start'] / tb))
                pk.pts -= shift
                pk.dts -= shift
                pk.stream = oa
                self.c.mux(pk)
            else:
                if pk.duration and (pk.pts + pk.duration) * tb <= a['start']:
                    continue
                for fr in pk.decode():
                    fr = self._trim_audio(fr, a)
                    if fr is None:
                        continue
                    if a['resampler'] is None:
                        a['resampler'] = av.AudioResampler(format=oa.codec_context.format.name if oa.codec_context.format else 'fltp',
                                                           layout=oa.codec_context.layout.name if oa.codec_context.layout else 'stereo',
                                                           rate=oa.codec_context.sample_rate)
                    for rf in a['resampler'].resample(fr):
                        rf.pts = None
                        for opk in oa.encode(rf):
                            self.c.mux(opk)

    def _trim_audio(self, fr, a):
        """Cut a decoded frame to [start, end) so transcoded audio lasts exactly as long as the video."""
        if fr.pts is None or fr.time_base is None:
            return fr
        sr = fr.sample_rate
        t0 = float(fr.pts * fr.time_base)
        i0 = max(0, int(round((a['start'] - t0) * sr)))
        i1 = fr.samples if a['end'] is None else min(fr.samples, int(round((a['end'] - t0) * sr)))
        if i1 <= i0:
            return None
        if i0 == 0 and i1 == fr.samples:
            return fr
        arr = fr.to_ndarray()
        if not fr.format.is_planar:  # packed: one row of interleaved samples
            ch = len(fr.layout.channels)
            i0, i1 = i0 * ch, i1 * ch
        out = self.av.AudioFrame.from_ndarray(np.ascontiguousarray(arr[:, i0:i1]), format=fr.format.name,
                                              layout=fr.layout.name)
        out.sample_rate = sr
        return out

    def _flush_audio_encoder(self):
        a = self._audio
        if a and a['mode'] == 'transcode':
            try:
                for opk in a['oa'].encode(None):
                    self.c.mux(opk)
            except Exception:
                pass

    # -- video -------------------------------------------------------------------------------

    def write(self, rgba):
        """Write one frame: (h, w, 4) uint8, or uint16 for deep profiles (float in [0,1] is accepted too)."""
        av = self.av
        a = np.asarray(rgba)
        if a.shape[0] != self.height or a.shape[1] != self.width:
            a = a[:self.height, :self.width]
        if a.dtype.kind == 'f':
            a = (np.clip(a, 0, 1) * (65535.0 if self.p.deep else 255.0) + 0.5).astype(np.uint16 if self.p.deep else np.uint8)
        if self.p.deep and a.dtype != np.uint16:
            a = a.astype(np.uint16) * 257
        if not self.p.deep and a.dtype != np.uint8:
            a = (a >> 8).astype(np.uint8)
        fmt = 'rgba64le' if a.dtype == np.uint16 else 'rgba'
        fr = av.VideoFrame.from_ndarray(np.ascontiguousarray(a), format=fmt)
        dst_range = self._cr if not self.p.alpha else self._cr
        fr = fr.reformat(format=self.p.pix_fmt, dst_colorspace=self._cs, dst_color_range=dst_range)
        fr.pts = self.frame_index
        fr.time_base = Fraction(1, 1) / Fraction(self.fps).limit_denominator(1001)
        for pk in self.s.encode(fr):
            self.c.mux(pk)
        self.frame_index += 1
        if self._audio is not None:
            self._pump_audio(self.frame_index / self.fps)

    def close(self):
        for pk in self.s.encode(None):
            self.c.mux(pk)
        if self._audio is not None:
            self._pump_audio(1e9)
            try:
                self._audio['ic'].close()
            except Exception:
                pass
        self.c.close()
        return self.path
