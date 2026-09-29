import io
import math

import numpy as np
import pytest
from PIL import Image

from blackbody.io.chan import _matrix, _to_xyz_euler, chan_keys
from blackbody.io.images import element_to_display, encode_png, write_exr
from blackbody.io.vdb import dense_from_leaves, read_vdb, write_vdb
from blackbody.engine.camera import euler_xyz


@pytest.mark.parametrize('dtype,channels', [(np.uint8, 3), (np.uint8, 4), (np.uint16, 4), (np.uint16, 3)])
def test_png_round_trip(tmp_path, dtype, channels):
    """Exact round trip through FFmpeg's PNG decoder (Pillow drops 16-bit colour to 8 bits)."""
    import av
    rng = np.random.default_rng(3)
    hi = 255 if dtype == np.uint8 else 65535
    img = rng.integers(0, hi + 1, (37, 53, channels)).astype(dtype)
    p = tmp_path / 'x.png'
    p.write_bytes(encode_png(img))
    fmt = {(np.uint8, 3): 'rgb24', (np.uint8, 4): 'rgba', (np.uint16, 3): 'rgb48le', (np.uint16, 4): 'rgba64le'}[(dtype, channels)]
    with av.open(str(p)) as c:
        back = next(c.decode(video=0)).to_ndarray(format=fmt)
    assert np.array_equal(back, img)


def test_png_round_trip_8bit_exact():
    rng = np.random.default_rng(4)
    img = rng.integers(0, 256, (21, 34, 4)).astype(np.uint8)
    back = np.asarray(Image.open(io.BytesIO(encode_png(img))))
    assert np.array_equal(back, img)


def test_exr_layers(tmp_path):
    import OpenEXR
    rng = np.random.default_rng(1)
    b = rng.random((16, 24, 4)).astype(np.float16)
    heat = rng.random((16, 24)).astype(np.float32)
    p = tmp_path / 't.exr'
    write_exr(p, {'R': b[..., 0], 'G': b[..., 1], 'B': b[..., 2], 'A': b[..., 3], 'heat.Y': heat}, 'piz')
    with OpenEXR.File(str(p)) as f:
        ch = f.channels()
        assert np.array_equal(ch['RGBA'].pixels, b)
        assert np.array_equal(ch['heat.Y'].pixels, heat)


def test_vdb_round_trip(tmp_path):
    rng = np.random.default_rng(2)
    d = np.zeros((40, 70, 30), np.float32)
    d[5:30, 10:60, 3:25] = rng.random((25, 50, 22))
    v = np.zeros((40, 70, 30, 3), np.float32)
    v[10:20, 20:30, 5:15] = rng.standard_normal((10, 10, 10, 3))
    p = tmp_path / 'g.vdb'
    write_vdb(p, {'density': d, 'vel': v}, 0.1, (0.0, 0.05, 0.0))
    r = read_vdb(p)
    assert r['version'] == 224
    dd = dense_from_leaves(r['grids']['density']['leaves'], d.shape)
    vv = dense_from_leaves(r['grids']['vel']['leaves'], d.shape, vec=True)
    assert np.allclose(dd, np.where(np.abs(d) > 1e-4, d, 0))
    assert np.allclose(vv, v)
    assert r['grids']['density']['xform']['voxel'] == pytest.approx(0.1)


def test_euler_decomposition_round_trip():
    rng = np.random.default_rng(5)
    for _ in range(50):
        rx, ry, rz = rng.uniform(-80, 80, 3)
        m = euler_xyz(rx, ry, rz)
        e = _to_xyz_euler(m)
        assert np.allclose(euler_xyz(*e), m, atol=1e-9)


def test_chan_import(tmp_path):
    p = tmp_path / 'cam.chan'
    p.write_text('1 0 1.6 6 0 0 0 40\n2 0.1 1.6 6 0 5 0 40\n3 0.2 1.6 6 0 10 0 40\n')
    pos, rot, focal = chan_keys(str(p), 'ZXY', 36.0, 16 / 9)
    assert len(pos) == 3 and pos[1][1] == (0.1, 1.6, 6.0)
    assert rot[2][1][1] == pytest.approx(10.0)
    # 40 degree vertical fov at 16:9 on a 36 mm sensor
    hfov = 2 * math.atan(math.tan(math.radians(20)) * 16 / 9)
    assert focal[0][1] == pytest.approx(36 / (2 * math.tan(hfov / 2)))


def test_element_alpha_modes():
    rgb = np.array([[[2.0, 1.0, 0.3], [0.0, 0.0, 0.0]]], np.float32)
    a = np.array([[0.1, 0.0]], np.float32)
    pre = element_to_display(rgb, a, mode='premultiplied')
    st = element_to_display(rgb, a, mode='straight')
    assert pre[0, 0, 3] == pytest.approx(0.1)
    assert st[0, 0, 3] > 0.5          # emitted light raises alpha in straight mode
    assert np.all(pre[0, 1] == 0.0)
