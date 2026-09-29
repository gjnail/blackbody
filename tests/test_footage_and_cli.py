"""Footage decoding is frame accurate; the command line renders every output type."""
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from blackbody.io.footage import Footage

ROOT = Path(__file__).resolve().parents[1]


def _make_numbered_video(path, n=48, fps=24, w=320, h=180):
    """Each frame's brightness encodes its index, so decoding can be checked frame by frame."""
    c = av.open(str(path), 'w')
    s = c.add_stream('libx264', rate=fps)
    s.width, s.height, s.pix_fmt = w, h, 'yuv420p'
    s.options = {'crf': '8', 'preset': 'ultrafast', 'g': '12'}
    for i in range(n):
        img = np.full((h, w, 3), int(20 + i * 4), np.uint8)
        for pk in s.encode(av.VideoFrame.from_ndarray(img, format='rgb24')):
            c.mux(pk)
    for pk in s.encode(None):
        c.mux(pk)
    c.close()


def test_video_frames_are_exact(tmp_path):
    p = tmp_path / 'num.mp4'
    _make_numbered_video(p)
    f = Footage(p)
    assert (f.width, f.height, f.frames) == (320, 180, 48)
    assert f.fps == pytest.approx(24.0)
    order = [0, 1, 2, 30, 31, 5, 47, 12, 13, 40, 0]  # sequential runs, backward seeks, jumps
    for i in order:
        v = int(np.median(f.read(i)[..., 0]))
        assert abs(v - (20 + i * 4)) <= 2, f'frame {i} decoded as brightness {v}'
    f.close()


def test_image_sequence(tmp_path):
    from PIL import Image
    for i in range(1001, 1006):
        Image.fromarray(np.full((20, 30, 3), i - 1000, np.uint8)).save(tmp_path / f'shot.{i:04d}.png')
    f = Footage(tmp_path / 'shot.1003.png')
    assert f.kind == 'sequence' and f.frames == 5 and f.first_number == 1001
    assert int(f.read(2)[0, 0, 0]) == 3


@pytest.mark.slow
def test_cli_renders_every_format(tmp_path, engine):
    out = tmp_path / 'r'
    cmd = [sys.executable, '-m', 'blackbody', 'render', '--preset', 'torch', '--draft', '--frames', '1-3',
           '--size', '320x180', '--samples', '1', '-q',
           '-o', str(out / 'e.####.exr'), '-o', str(out / 'p.####.png'), '-o', str(out / 'a.mov'),
           '-o', str(out / 'v.####.vdb')]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert len(list(out.glob('e.*.exr'))) == 3
    assert len(list(out.glob('p.*.png'))) == 3
    assert len(list(out.glob('v.*.vdb'))) == 3
    with av.open(str(out / 'a.mov')) as c:
        s = c.streams.video[0]
        assert s.codec_context.name == 'prores' and 'yuva444' in s.codec_context.pix_fmt
        assert sum(1 for _ in c.decode(video=0)) == 3


def test_cli_set_is_strict():
    from blackbody.cli import _apply_setting
    from blackbody.scene import presets
    s = presets.make('campfire')
    assert _apply_setting(s, 'motion.wind_speed=2.5') is None
    assert s.get(('motion', 'wind_speed'), s.start) == pytest.approx(2.5)
    assert _apply_setting(s, 'emitter.0.fuel=14') is None
    assert s.emitters[0]['fuel'] == pytest.approx(14.0)
    assert _apply_setting(s, 'composite.view=agx') is None
    assert 'wind_speed' in _apply_setting(s, 'motion.wnd_speed=3')           # suggests the right name
    assert 'not one of' in _apply_setting(s, 'composite.view=agxx')          # enums never fall back quietly
    assert 'not a number' in _apply_setting(s, 'motion.buoyancy=lots')
    assert 'three numbers' in _apply_setting(s, 'shading.smoke_albedo=0.1 0.2')
    assert 'on or off' in _apply_setting(s, 'domain.ground=maybe')
    assert 'index' in _apply_setting(s, 'emitter.fuel=20')
    assert 'no emitter.9' in _apply_setting(s, 'emitter.9.fuel=20')
    assert s.data['composite']['view'] == 'agx'                               # failed sets changed nothing


def test_cli_argument_errors_exit_2(tmp_path, capsys):
    from blackbody.cli import main
    out = str(tmp_path / 'x.####.exr')
    assert main(['render', '--preset', 'torchy', '-o', out]) == 2
    assert main(['render', str(tmp_path / 'missing.bbfire'), '-o', out]) == 2
    assert main(['render', '--preset', 'torch', '--frames', '5-2', '-o', out]) == 2
    assert main(['render', '--preset', 'torch', '-o', str(tmp_path / 'x.tif')]) == 2
    assert main(['render', '--preset', 'torch', '-o', str(tmp_path / 'x.mov'), '--format', 'prores9']) == 2
    assert main(['render', '--preset', 'torch', '--footage', str(tmp_path / 'nope.mov'), '-o', out]) == 2
    err = capsys.readouterr().err
    assert err.count('Error: ') == 6 and 'Traceback' not in err
    assert not list(tmp_path.glob('*.exr'))
    assert main(['settings', 'motion']) == 0
    assert 'motion.wind_speed' in capsys.readouterr().out


def test_mov_audio_is_lossless_and_frame_exact(tmp_path):
    """Audio a .mov cannot hold as-is (FLAC) becomes 24-bit PCM cut to exactly the rendered frames."""
    from blackbody.io.video import VideoWriter
    tone = (0.3 * 32767 * np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)).astype(np.int16)
    src = tmp_path / 'src.mkv'
    with av.open(str(src), 'w') as c:
        v = c.add_stream('mpeg4', rate=30)
        v.width, v.height, v.pix_fmt = 64, 48, 'yuv420p'
        a = c.add_stream('flac', rate=48000)
        a.layout = 'stereo'
        for _ in range(30):
            for p in v.encode(av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), np.uint8), format='rgb24')):
                c.mux(p)
        for k in range(0, 48000, 4800):
            fr = av.AudioFrame.from_ndarray(np.stack([tone[k:k + 4800]] * 2).T.reshape(1, -1).copy(), format='s16', layout='stereo')
            fr.sample_rate, fr.pts = 48000, k
            for p in a.encode(fr):
                c.mux(p)
        for s in (v, a):
            for p in s.encode():
                c.mux(p)
    out = tmp_path / 'comp.mov'
    w = VideoWriter(out, 'prores422hq', 64, 48, 30.0, audio_source=src, audio_start=0.1, audio_duration=10 / 30)
    for _ in range(10):
        w.write(np.zeros((48, 64, 4), np.uint8))
    w.close()
    with av.open(str(out)) as c:
        assert c.streams.audio[0].codec_context.name == 'pcm_s24le'
        frames = list(c.decode(audio=0))
    left = np.concatenate([f.to_ndarray().reshape(f.samples, -1)[:, 0] for f in frames]) / 2.0 ** 31
    assert len(left) == 16000                                        # 10 frames at 30 fps, 48 kHz
    assert np.abs(left - tone[4800:20800] / 32768.0).max() < 1e-6     # the same samples, not shifted
