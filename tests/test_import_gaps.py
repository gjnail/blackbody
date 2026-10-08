"""Import gaps closed: the mesh atlas packed in columns within any GPU's 2048-cell limit; VDB velocity (at its own
strength), blending between a sequence's frames, a field as fine as the simulation, a tilted volume keeping to the field's
limits, bit-shuffled and Zstd Blosc data, volumes that pour liquid; USD points as particles (one-frame caches too) or
solid balls, IES profiles on panels, portal and geometry lights, shadow switches, and a frame offset and frame range
taken by the command line too. Synthetic USD files (pxr) and VDBs (io/vdb.py, or written byte
by byte as other programs do). Mostly GPU-free; the GPU tests are small."""
import argparse
import functools
import math
import struct
import sys
import types

import numpy as np
import pytest

from blackbody.scene import presets


def _quiet(sc, res=40):
    sc.data['domain'].update(time_scale=1.0, resolution=res, preroll=0.0)
    sc.data['motion'].update(turbulence=0.0, disturbance=0.0, vorticity=0.0, wind_speed=0.0, puffing=0.0)
    sc.data['embers']['enabled'] = False
    sc.emitters = []
    sc.colliders = []
    return sc


def _ball(n=24, radius=10.0):
    idx = np.indices((n, n, n)).transpose(1, 2, 3, 0).astype(float)
    r = np.linalg.norm(idx - (n - 1) / 2, axis=-1)
    return np.clip(1.0 - r / radius, 0.0, None).astype(np.float32) * 3.0


def _ball_vdb(path, origin=(-0.6, 0.8, -0.6), voxel=0.05, vel=None, heat=None, zup=False, n=24):
    from blackbody.io.vdb import write_vdb
    d = _ball(n)
    grids = {'density': d}
    if heat is not None:
        grids['temperature'] = (d / 3.0 * heat).astype(np.float32)
    if vel is not None:
        v = np.zeros(d.shape + (3,), np.float32)
        v[...] = vel
        grids['vel'] = v
    write_vdb(path, grids, voxel, origin)
    return str(path)


# -- the mesh atlas ------------------------------------------------------------------------------------------

def _overlap(a, ca, b, cb):
    return all(ca[k] < cb[k] + b[k] and cb[k] < ca[k] + a[k] for k in range(3))


def test_the_atlas_packs_in_columns_within_the_2048_cell_limit():
    """20 meshes at Mesh detail 96 and 5 VDBs with a temperature stacked 3540 cells deep (the old layout: past D3D12's
    and Metal's 2048, the whole frame failed). Packed in columns they all fit, none overlapping, every side within
    2048, and each grid's corner travels as a float32 exactly."""
    from blackbody.engine.mesh import ATLAS_TILE, atlas_code, atlas_corner, pack_atlas
    boxes = [(96, 60, 80)] * 20 + [(194, 194, 2 * 194)] * 5
    assert sum(b[2] for b in boxes) > 2048
    size, corners = pack_atlas(boxes, limit=2048)
    assert all(c is not None for c in corners) and max(size) <= 2048
    old = max(b[0] for b in boxes) * max(b[1] for b in boxes) * sum(b[2] for b in boxes)
    assert np.prod(size) <= old, 'no bigger than one stack would have been'
    for i, (b, c) in enumerate(zip(boxes, corners)):
        assert c[0] % ATLAS_TILE == 0 and c[1] % ATLAS_TILE == 0
        assert all(c[k] + b[k] <= size[k] for k in range(3))
        code = atlas_code(c)
        assert float(np.float32(code)) == code and atlas_corner(code) == tuple(c)
        for j in range(i):
            assert not _overlap(b, c, boxes[j], corners[j]), (i, j)
    # a few small meshes still share one stack, as before
    size, corners = pack_atlas([(40, 30, 50), (60, 20, 40)])
    assert size == (60, 30, 90) and corners == [(0, 0, 0), (0, 0, 50)]


def test_the_atlas_leaves_out_whole_sources_last_first():
    from blackbody.engine.mesh import pack_atlas
    boxes = [(100, 100, 100), (100, 100, 100), (100, 100, 100), (100, 100, 100)]
    groups = ['a', 'b', 'c', 'c']    # c: a deforming mesh's two frames
    size, corners = pack_atlas(boxes, groups, budget=4 * 100 * 100 * 250)
    assert corners[0] is not None and corners[1] is not None
    assert corners[2] is None and corners[3] is None, 'both frames of the last source go'
    assert 4 * np.prod(size) <= 4 * 100 * 100 * 250
    _, corners = pack_atlas([(3000, 10, 10), (10, 10, 10)])
    assert corners[0] is None and corners[1] is not None, 'a grid too big for any column is left out alone'


def _mesh_scene(tmp_path, n=6):
    from blackbody.scene.presets import _col
    sc = _quiet(presets.make('campfire'), 40)
    sc.data['domain'].update(size_x=2.4, size_y=1.6, size_z=2.4, mesh_resolution=32)
    for i in range(n):
        p = tmp_path / f'box{i}.obj'
        h = 0.08 + 0.02 * i
        v = [(sx * h, sy * h * 0.6, sz * h) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        faces = ((1, 3, 4, 2), (5, 6, 8, 7), (1, 2, 6, 5), (3, 7, 8, 4), (1, 5, 7, 3), (2, 4, 8, 6))
        p.write_text('\n'.join([f'v {x} {y} {z}' for x, y, z in v] + ['f ' + ' '.join(map(str, f)) for f in faces]))
        sc.colliders.append(_col(sc, i, shape='mesh', mesh=str(p), position=(-0.9 + 0.36 * i, 0.4, 0.1 * i),
                                 size=(1.0, 1.0, 1.0), holdout=False))
    return sc


def test_meshes_in_atlas_columns_sample_as_in_one_stack(engine, tmp_path, monkeypatch):
    """The shaders find each grid wherever its column stands: the solid field is the same with the grids spread over
    several columns in several rows (the layout a GPU limited to 2048 cells gets once they no longer fit one stack),
    so both halves of an atlas code, the x tile and the y tile, are taken apart right on the GPU."""
    from blackbody.engine import mesh as M
    sc = _mesh_scene(tmp_path)
    engine.invalidate()
    engine.prepare(sc)
    s = engine.solver
    one = s.gpu.read(s.sdf)[..., 0].copy()
    assert s.meshes.atlas.size[0] < 2 * M.ATLAS_TILE, 'one stack'
    boxes = [(g.dims[0], g.dims[1], g.data.shape[0]) for _, g in s.meshes.slots.values()]
    depth = max(b[2] for b in boxes)                                  # (a grid to a column) ...
    limit = 2 * M._tiled(max(b[0] for b in boxes))                    # ... two columns to a row
    assert len(boxes) == 6 and 3 * M._tiled(max(b[1] for b in boxes)) <= limit, boxes
    real = M.pack_atlas
    monkeypatch.setattr(M, 'pack_atlas', lambda b, groups=None, **kw: real(b, groups, limit=limit, depth=depth))
    s.meshes._key = ()
    engine.invalidate()
    engine.prepare(sc)
    assert not s.meshes.errors
    corners = [M.atlas_corner(code) for code, _ in s.meshes.slots.values()]
    assert any(c[0] > 0 for c in corners) and any(c[1] > 0 for c in corners), corners
    assert max(s.meshes.atlas.size[:2]) <= limit
    cols = s.gpu.read(s.sdf)[..., 0]
    assert (one < 0).sum() > 50
    assert np.array_equal(one < 0, cols < 0) and np.allclose(one, cols, atol=1e-4)


def test_meshes_the_atlas_has_no_room_for_are_left_out_with_a_notice(engine, tmp_path, monkeypatch):
    from blackbody.engine import mesh as M
    sc = _mesh_scene(tmp_path, 4)
    engine.invalidate()
    engine.prepare(sc)
    s = engine.solver
    boxes = [(sd.dims[0], sd.dims[1], sd.data.shape[0]) for _, sd in
             (s.meshes.slots[sc.mesh_path(c['mesh'])] for c in sc.colliders)]
    room = 4 * int(np.prod(M.pack_atlas(boxes[:2])[0]))   # (room for the first two)
    monkeypatch.setattr(M, 'pack_atlas', functools.partial(M.pack_atlas, budget=room))
    s.meshes._key = ()
    engine.invalidate()
    engine.prepare(sc)
    last = sc.mesh_path(sc.colliders[-1]['mesh'])
    assert last in s.meshes.errors and 'no room' in s.meshes.errors[last]
    assert s.meshes.ref(sc.mesh_path(sc.colliders[0]['mesh'])) is not None, 'the first keep their room'
    assert s.meshes.ref(last) is None
    assert any('no room' in n for n in engine.notices())
    engine.simulate_to(sc, sc.start + 2, cache=False)   # the frame goes on without it


def test_the_guides_give_the_limits_the_code_has():
    import pathlib
    from blackbody.engine import mesh as M
    from blackbody.io import volume as V
    docs = pathlib.Path(__file__).resolve().parents[1] / 'docs'
    t = ' '.join((docs / 'troubleshooting.md').read_text(encoding='utf-8').split())
    s = ' '.join((docs / 'scene-import.md').read_text(encoding='utf-8').split())
    assert f'{V.FIELD_MAX_CELLS} cells a side' in t and f'{V.FIELD_MAX_CELLS} cells a side' in s
    assert f'{V.FIELD_MAX_VOXELS // 1_000_000} million' in t and f'{V.FIELD_MAX_VOXELS // 1_000_000} million' in s
    assert f'{V.POINTS_MAX // 1_000_000} million points' in t and f'{V.POINT_FRAMES} of its frames' in t
    assert f'up to {round(M.ATLAS_MAX_BYTES / 2 ** 30)} GB' in t and f'{M.ATLAS_ZSPAN} cells a side' in t
    assert V.FIELD_PER_CELL == 2.0 and 'half a simulation cell' in t and 'half a simulation cell' in s


# -- VDB: Blosc variants -------------------------------------------------------------------------------------

def _lz4_literals(data):
    n = len(data)
    head = bytearray([0xF0 if n >= 15 else n << 4])
    if n >= 15:
        rest = n - 15
        while rest >= 255:
            head.append(255)
            rest -= 255
        head.append(rest)
    return bytes(head) + bytes(data)


def _bitshuffle_ref(buf, typesize):
    """bitshuffle's bshuf_trans_bit_elem as its scalar code does it: bytes transposed, then each 8-byte word's 8x8 bit
    matrix (TRANS_BIT_8X8) into rows, then the rows regrouped by byte."""
    n = len(buf) // typesize
    a = bytes(buf[i * typesize + j] for j in range(typesize) for i in range(n))
    nbyte = n * typesize
    rows = nbyte // 8
    tmp = bytearray(nbyte)
    for ii in range(rows):
        x = int.from_bytes(a[8 * ii:8 * ii + 8], 'little')
        t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AA
        x = x ^ t ^ (t << 7)
        t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCC
        x = x ^ t ^ (t << 14)
        t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0
        x = (x ^ t ^ (t << 28)) & ((1 << 64) - 1)
        for kk in range(8):
            tmp[kk * rows + ii] = x & 0xFF
            x >>= 8
    nb = n // 8
    out = bytearray(nbyte)
    for kk in range(8):
        for j in range(typesize):
            out[(j * 8 + kk) * nb:(j * 8 + kk + 1) * nb] = tmp[(kk * typesize + j) * nb:(kk * typesize + j + 1) * nb]
    return bytes(out)


def _blosc_block(body, n, typesize, flags, version=2):
    """A one-block Blosc 1 buffer around a codec body (one split)."""
    return struct.pack('<BBBBIII', version, 1, flags, typesize, n, n, 16 + 4 + 4 + len(body)) + struct.pack('<I', 20) \
        + struct.pack('<i', len(body)) + body


def test_blosc_bit_shuffled_data_is_read():
    from blackbody.io.vdbread import blosc_decompress
    rng = np.random.default_rng(1)
    raw = rng.random(256).astype(np.float32).tobytes()          # 256 elements: a multiple of 8
    z = _lz4_literals(_bitshuffle_ref(raw, 4))
    assert blosc_decompress(_blosc_block(z, len(raw), 4, 0x10 | 0x04 | (1 << 5))) == raw
    # format 3 and on: the first multiple of 8 elements is shuffled, the rest stays as it is
    odd = rng.random(259).astype(np.float32).tobytes()
    k = (259 // 8) * 8 * 4
    z = _lz4_literals(_bitshuffle_ref(odd[:k], 4) + odd[k:])
    assert blosc_decompress(_blosc_block(z, len(odd), 4, 0x10 | 0x04 | (1 << 5), version=3)) == odd


def test_zstd_blosc_goes_through_the_zstandard_package(monkeypatch):
    from blackbody.io.vdbread import VDBError, blosc_decompress
    raw = np.arange(64, dtype=np.float32).tobytes()
    shuffled = np.frombuffer(raw, np.uint8).reshape(64, 4).T.reshape(-1).tobytes()
    buf = _blosc_block(b'ZSTD' + shuffled, len(raw), 4, 0x10 | 0x01 | (4 << 5))
    monkeypatch.setitem(sys.modules, 'compression', None)    # (no Python 3.14 zstd either)
    monkeypatch.setitem(sys.modules, 'zstandard', None)
    with pytest.raises(VDBError, match='zstandard'):
        blosc_decompress(buf)
    calls = []

    class Decompressor:
        def decompress(self, data, max_output_size=0):
            calls.append(max_output_size)
            assert data[:4] == b'ZSTD'
            return data[4:]
    monkeypatch.setitem(sys.modules, 'zstandard', types.SimpleNamespace(ZstdDecompressor=Decompressor))
    assert blosc_decompress(buf) == raw and calls == [len(raw)]


# -- VDB: velocity, layers, resolution, sequences --------------------------------------------------------------

def test_a_vdb_brings_its_velocity_and_heat_as_layers(tmp_path):
    from blackbody.io.volume import field_source, load_field
    p = _ball_vdb(tmp_path / 'puff.vdb', vel=(0.0, 0.0, 2.0), heat=1.5, zup=True)
    g, _ = load_field(field_source(p, zup=True), None)
    assert g.layers == 5 and g.temp and g.vel and g.flags == 3
    nz = g.dims[2]
    dens, heat = g.data[:nz], g.data[nz:2 * nz]
    vx, vy, vz = (g.data[k * nz:(k + 1) * nz] for k in (2, 3, 4))
    core = dens > 0.5
    assert heat[core].max() > 0.5
    # the file's +z is up in a Z-up file: y up here
    assert vy[core] == pytest.approx(2.0, abs=1e-5) and np.abs(vx[core]).max() < 1e-5 and np.abs(vz[core]).max() < 1e-5
    # Blender-style split grids (vel.x, vel.y, vel.z) are read too
    from blackbody.io.vdb import write_vdb
    d = _ball()
    write_vdb(tmp_path / 'split.vdb', {'density': d, 'vel.x': np.full_like(d, -1.0), 'vel.y': np.zeros_like(d),
                                       'vel.z': np.zeros_like(d)}, 0.05, (0.0, 0.0, 0.0))
    g, _ = load_field(field_source(str(tmp_path / 'split.vdb')), None)
    assert g.vel and not g.temp
    nz = g.dims[2]
    assert g.data[nz:2 * nz][g.data[:nz] > 0.5] == pytest.approx(-1.0)


def test_the_vdb_header_gives_each_grids_box_without_its_data(tmp_path):
    from blackbody.io.vdbread import read_float_grid, vdb_grid_info
    p = _ball_vdb(tmp_path / 'b.vdb', vel=(1.0, 0.0, 0.0))
    info = vdb_grid_info(p)
    assert set(info) == {'density', 'vel'} and 'vec3' in info['vel']['type']
    lo, hi = info['density']['bbox']
    dense, dlo, xf = read_float_grid(p, 'density')
    assert tuple(lo) == (0, 0, 0) and tuple(hi) == (24, 24, 24)
    assert np.allclose(info['density']['xform'], xf)


def test_a_field_is_as_fine_as_the_simulation_can_use(tmp_path):
    """No longer 96 to 192 cells whatever the simulation: as fine as the file, down to half a simulation cell."""
    from blackbody.io.vdb import write_vdb
    from blackbody.io.volume import field_source, load_field
    slab = np.ones((220, 8, 8), np.float32)
    write_vdb(tmp_path / 'slab.vdb', {'density': slab}, 0.01, (0.0, 0.0, 0.0))
    src = field_source(str(tmp_path / 'slab.vdb'))
    fine, _ = load_field(src, None)
    assert fine.dims[0] - 2 == 224, 'every voxel of its 8-voxel leaves (the old cap: at most 192)'
    coarse, _ = load_field(src, None, finest=0.04)
    assert coarse.dims[0] - 2 == 56, 'averaged down to the 4 cm the simulation can use'
    assert (fine.bmax[0] - fine.bmin[0]) == pytest.approx(coarse.bmax[0] - coarse.bmin[0] - 0.06, abs=1e-6)


def _tilted_volume(tmp_path, n=40):
    """A USD Volume prim turned 45 degrees about y and 35 about x, its VDB (n voxels of 2.5 cm a side) holding a ball of
    density with a temperature and a velocity of 2 m/s along its own x: (field source in world space, its rotation)."""
    from pxr import Sdf, Usd, UsdGeom, UsdVol
    from blackbody.io.vdb import write_vdb
    from blackbody.io.volume import field_source
    idx = np.indices((n, n, n)).transpose(1, 2, 3, 0).astype(float)
    d = np.clip(1.0 - np.linalg.norm(idx - (n - 1) / 2, axis=-1) / (0.45 * n), 0.0, None).astype(np.float32)
    v = np.zeros(d.shape + (3,), np.float32)
    v[..., 0] = 2.0
    write_vdb(tmp_path / 'tilted.vdb', {'density': d, 'temperature': 2.0 * d, 'vel': v}, 0.025, (0.0, 0.0, 0.0))
    path = tmp_path / 'tilted.usda'
    st = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    vol = UsdVol.Volume.Define(st, '/W/Cloud')
    UsdGeom.Xformable(vol).AddRotateYOp().Set(45.0)
    UsdGeom.Xformable(vol).AddRotateXOp().Set(35.0)
    f = UsdVol.OpenVDBAsset.Define(st, '/W/Cloud/density')
    f.CreateFilePathAttr(Sdf.AssetPath(str(tmp_path / 'tilted.vdb')))
    f.CreateFieldNameAttr('density')
    vol.CreateFieldRelationship('density', f.GetPath())
    st.GetRootLayer().Save()
    a, b = math.radians(45.0), math.radians(35.0)
    ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    return field_source(f'{path}#/W/Cloud?world'), ry @ rx


def test_a_tilted_usd_volume_keeps_to_the_field_limits(tmp_path, monkeypatch):
    """A tilted Volume prim is resampled on the upright box around it, up to about 1.7 times as many cells a side as the
    file has: it is that box that keeps to the cells a side, the cells in all and one atlas column for all its layers.
    (Before, only the file's own voxels were checked, and the box could reach 1024 cells a side: a 200-voxel VDB with a
    temperature and a velocity took 2 minutes and came out 25 million cells a layer; from about 240 voxels its five
    layers no longer fitted one column, and the atlas left it out.)"""
    pytest.importorskip('pxr')
    from blackbody.io import volume as V
    monkeypatch.setattr(V, '_cache_path', lambda *a, **k: None)
    src, rot = _tilted_volume(tmp_path)
    # as fine as the file where the limits allow: cells of its own 2.5 cm voxels
    g, ref = V.load_field(src, None)
    assert g.layers == 5 and g.temp and g.vel
    assert (g.bmax[0] - g.bmin[0]) / g.dims[0] == pytest.approx(0.025, rel=1e-6)
    # tight limits: on cells a side (max_cells), in all, and on the atlas column's depth
    monkeypatch.setattr(V, 'FIELD_DEPTH', 180)
    monkeypatch.setattr(V, 'FIELD_MAX_VOXELS', 25_000)
    for cells in (48, 30):
        t, _ = V.load_field(src, None, max_cells=cells, reference=ref)
        nx, ny, nz = t.dims
        assert max(t.dims) <= min(cells, V.FIELD_DEPTH // t.layers), t.dims
        assert nx * ny * nz <= V.FIELD_MAX_VOXELS and t.layers * nz <= V.FIELD_DEPTH and t.data.shape[0] == 5 * nz
        cell = (t.bmax[0] - t.bmin[0]) / nx
        dens = t.data[:nz]
        # the ball is all there, and moves at the file's 2 m/s along its own x, turned with the prim
        assert dens.sum() * cell ** 3 == pytest.approx(g.data[:g.dims[2]].sum() * 0.025 ** 3, rel=0.03)
        core = dens > 0.5
        vel = np.stack([t.data[k * nz:(k + 1) * nz][core] for k in (2, 3, 4)], -1)
        assert vel.mean(0) == pytest.approx(rot @ (2.0, 0.0, 0.0), abs=0.02)


def test_a_sequence_shares_one_grid_so_its_frames_blend(tmp_path):
    from blackbody.io.volume import field_grid, field_source, load_field
    for f in (1, 2, 3):
        _ball_vdb(tmp_path / f'seq.{f:04d}.vdb', origin=(-0.6 + 0.1 * f, 0.8, -0.6))
    src = field_source(str(tmp_path / 'seq.####.vdb'))
    grid = field_grid(src)
    frames = [load_field(src, f, grid=grid)[0] for f in (1, 2, 3)]
    assert len({(g.bmin, g.dims) for g in frames}) == 1, 'one grid'
    c = [np.argwhere(g.data > 1.0).mean(0) for g in frames]
    assert c[2][2] - c[0][2] == pytest.approx(0.2 / grid[1][0], abs=0.01), 'each frame where its file puts it'


def test_volume_emitters_carry_their_velocity_strength_and_pour_liquid():
    sc = presets.make('campfire')
    sc.add_emitter(shape='volume', volume='puff.vdb', volume_velocity=0.7, start=-100.0, volume_mode='hold')
    em = [e for e in sc.emitters_gpu(sc.start) if e.shape == 'volume']
    assert em and em[0].volume_vel == pytest.approx(0.7) and em[0].mesh.endswith('puff.vdb|field')
    from blackbody.engine.solver import Uniforms, pack_emitters
    u = pack_emitters(Uniforms(), em)
    assert u is not None
    liq = presets.make('water_pour')
    assert liq.kind == 'liquid'
    liq.add_emitter(shape='volume', volume='splash.vdb', liquid_mode='fill', start=0.0)
    srcs = [s for s in liq.sources_gpu(liq.start, filled=set()) if s.shape == 'volume']
    assert srcs and srcs[0].mesh.endswith('splash.vdb|field') and srcs[0].volume_vel == pytest.approx(1.0)


def test_velocity_from_the_volume_leaves_other_emitters_caches_standing():
    """Any other emitter simulates as before, bit for bit (the GPU test below), so its disk cache stands: Velocity from
    the volume, which every emitter now carries, counts in the simulation's signature only for a Volume emitter (before,
    every scene with an emitter simulated again once)."""
    sc = presets.make('campfire')
    assert sc.emitters and all(e['shape'] != 'volume' for e in sc.emitters)
    sig = sc.sim_signature()
    for e in sc.emitters:
        del e['volume_velocity']
    assert sc.sim_signature() == sig, 'a scene simulated before it existed keeps its cache'
    sc.emitters[0]['volume_velocity'] = 0.3
    assert sc.sim_signature() == sig
    sc.add_emitter(shape='volume', volume='puff.vdb', start=0.0)
    sig = sc.sim_signature()
    sc.emitters[-1]['volume_velocity'] = 0.3
    assert sc.sim_signature() != sig, "a Volume emitter's does count"


# -- VDB on the GPU --------------------------------------------------------------------------------------------

def _volume_scene(engine, vdb, **kw):
    sc = _quiet(presets.make('smoke_plume'), 40)
    args = dict(shape='volume', volume=str(vdb), position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), smoke=4.0,
                temperature=0.0, fuel=0.0, noise=0.0, start=0.0, volume_mode='fill', embers=False)
    args.update(kw)
    sc.add_emitter(**args)
    engine.invalidate()
    engine.prepare(sc)
    return sc


def _soot_centre(engine):
    s = engine.solver
    soot = s.read_scalars()[..., 2].astype(np.float64)
    z, y, x = np.indices(soot.shape)
    ijk = np.array([(soot * x).sum(), (soot * y).sum(), (soot * z).sum()]) / max(soot.sum(), 1e-12)
    return soot.sum(), np.asarray(s.origin) + (ijk + 0.5) * s.h


def test_a_vdbs_velocity_moves_its_smoke_off(engine, tmp_path):
    """An explosion's vel grid: the smoke goes in moving as it did in the file (here 3 m/s sideways), where before it
    started still. With Velocity from the volume at 0 it starts still again."""
    vdb = _ball_vdb(tmp_path / 'puff.vdb', origin=(-0.6, 0.8, -0.6), vel=(3.0, 0.0, 0.0))
    xs = {}
    for k in (0.0, 1.0):
        sc = _volume_scene(engine, vdb, volume_velocity=k)
        engine.simulate_to(sc, sc.start + 8, cache=False)
        total, c = _soot_centre(engine)
        assert total > 0.0
        xs[k] = c[0]
    assert xs[1.0] - xs[0.0] > 0.15, xs


def test_velocity_from_the_volume_counts_at_its_own_strength(engine, tmp_path):
    """Velocity from the volume scales the volume's own velocity whatever the emitter's Velocity strength: at 0.2 the
    air that fills with the smoke takes a fifth of the file's 3 m/s, with Velocity strength 1 as with 0 (before, at
    Velocity strength 1 it took all of it)."""
    vdb = _ball_vdb(tmp_path / 'puff.vdb', origin=(-0.6, 0.8, -0.6), vel=(3.0, 0.0, 0.0))
    u = {}
    for blend, k in ((1.0, 1.0), (1.0, 0.2), (0.0, 0.2)):
        sc = _volume_scene(engine, vdb, volume_velocity=k, vel_blend=blend, velocity=(0.0, 0.0, 0.0))
        engine.simulate_to(sc, sc.start + 1, cache=False)
        u[(blend, k)] = float(engine.solver.read_velocity_centres()[..., 0].astype(np.float64).sum())
    full = u[(1.0, 1.0)]
    assert full > 0.0
    assert u[(1.0, 0.2)] / full == pytest.approx(0.2, abs=0.06), u
    assert u[(0.0, 0.2)] / full == pytest.approx(0.2, abs=0.06), u


_VEL_TARGET_KERNEL = """
struct EmitterBlock { cnt: vec4<f32>, em: array<Emitter, MAX_EMITTERS> };
@group(0) @binding(0) var atlas: texture_3d<f32>;
@group(0) @binding(1) var<storage, read> E: EmitterBlock;
@group(0) @binding(2) var<storage, read> masks: array<f32>;
@group(0) @binding(3) var<storage, read_write> outv: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = arrayLength(&masks);
  if (id.x >= n) { return; }
  for (var j = 0; j < i32(E.cnt.x); j++) {
    let e = E.em[j];
    let w = e.a.xyz + vec3<f32>(0.01 * f32(id.x % 7u), 0.0, 0.0);
    let o = 2u * (u32(j) * n + id.x);
    outv[o] = emitter_vel_target(e, w, masks[id.x]);
    outv[o + 1u] = vec4<f32>(emitter_velocity(e, w), 0.0);
  }
}
"""


def test_an_emitter_without_a_volume_sets_the_velocity_exactly_as_before(engine):
    """Velocity from the volume leaves every other emitter as it was, bit for bit: the flow is mixed toward its own
    Velocity at mask x Velocity strength (emitters.wgsl emitter_vel_target, which apply.wgsl and liq_forces.wgsl use),
    and embers and liquid start with its own Velocity. Mixing it as (a own + b volume) / max(a, b) gave (a own) / a,
    which in float32 is not always own: every preset that sets a velocity, and every moving emitter, simulated a little
    differently (campfire's smoke by 2.5%). Run on the GPU for many masks, at a carried emitter's small strength and a
    jet's large one."""
    from blackbody.engine.gpu import Uniforms, _layout_entry, load_wgsl
    from blackbody.engine.solver import EMITTER_VEC4, pack_emitters
    sc = _quiet(presets.make('smoke_plume'))
    sc.add_emitter(shape='capsule', position=(0.1, 0.3, -0.2), velocity=(0.3, 1.7, -2.9), vel_blend=0.047,
                   start=0.0, fade_in=0.0)
    sc.add_emitter(shape='box', position=(0.0, 0.5, 0.0), velocity=(-7.3, 11.1, 0.9), vel_blend=0.83,
                   start=0.0, fade_in=0.0)
    em = sc.emitters_gpu(sc.start + 5)
    assert len(em) == 2 and all(e.shape != 'volume' and e.radial == 0.0 for e in em)
    packed = np.frombuffer(pack_emitters(Uniforms(), em).tobytes(), np.float32)
    own = [packed[4 + j * 4 * EMITTER_VEC4 + 16:][:3] for j in range(2)]          # (each one's e.xyz, after the count)
    blend = [packed[4 + j * 4 * EMITTER_VEC4 + 15] for j in range(2)]             # (and its d.w)
    assert blend[0] == pytest.approx(0.047) and blend[1] == pytest.approx(0.83)
    n = 1024
    masks = np.random.default_rng(3).uniform(0.0, 1.0, n).astype(np.float32)
    seen = set()
    code = ''.join(load_wgsl(f, _seen=seen) for f in ('common.wgsl', 'noise.wgsl', 'meshsdf.wgsl', 'emitters.wgsl'))
    import wgpu
    dev = engine.gpu.device
    BU = wgpu.BufferUsage
    layout = dev.create_bind_group_layout(entries=[_layout_entry(i, s) for i, s in
                                                   enumerate(('utex3d', 'rbuf', 'rbuf', 'buf'))])
    pipe = dev.create_compute_pipeline(layout=dev.create_pipeline_layout(bind_group_layouts=[layout]), compute={
        'module': dev.create_shader_module(code=code + _VEL_TARGET_KERNEL), 'entry_point': 'main'})
    atlas = dev.create_texture(size=(1, 1, 1), dimension='3d', format='r32float', usage=wgpu.TextureUsage.TEXTURE_BINDING)
    bufs = [dev.create_buffer_with_data(data=packed.tobytes(), usage=BU.STORAGE),
            dev.create_buffer_with_data(data=masks.tobytes(), usage=BU.STORAGE),
            dev.create_buffer(size=2 * 2 * n * 16, usage=BU.STORAGE | BU.COPY_SRC)]
    group = dev.create_bind_group(layout=layout, entries=[{'binding': 0, 'resource': atlas.create_view()}] + [
        {'binding': i + 1, 'resource': {'buffer': b, 'offset': 0, 'size': b.size}} for i, b in enumerate(bufs)])
    enc = dev.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipe)
    cp.set_bind_group(0, group)
    cp.dispatch_workgroups(n // 64)
    cp.end()
    dev.queue.submit([enc.finish()])
    out = np.frombuffer(dev.queue.read_buffer(bufs[2]), np.float32).reshape(2, n, 2, 4)
    for j in range(2):
        target, start = out[j, :, 0], out[j, :, 1]
        firm = np.clip(masks * blend[j], np.float32(0.0), np.float32(1.0))
        off = int((target[:, :3] != own[j]).any(axis=1).sum())
        assert off == 0, f'emitter {j}: {off} of {n} targets are not its own Velocity'
        assert np.array_equal(target[:, 3], firm), 'mixed in at mask x Velocity strength'
        assert np.array_equal(start[:, :3], np.broadcast_to(own[j], (n, 3))), 'what it releases starts at its Velocity'


def test_a_sequence_blends_between_its_frames(engine, tmp_path):
    """A volume that fills the box half way between two frames of its sequence: half of each frame (before, the first
    frame alone, whole)."""
    sc0 = presets.make('smoke_plume')
    f = sc0.start
    _ball_vdb(tmp_path / f'seq.{f:04d}.vdb', origin=(-0.9, 0.8, -0.6), n=24)
    _ball_vdb(tmp_path / f'seq.{f + 1:04d}.vdb', origin=(0.3, 0.8, -0.6), n=24)
    half = 0.5 / sc0.fps
    sc = _volume_scene(engine, tmp_path / 'seq.####.vdb', start=half)
    s = engine.solver
    engine.simulate_to(sc, sc.start + 2, cache=False)
    assert not s.meshes.errors
    soot = s.read_scalars()[..., 2].astype(np.float64)
    x = (np.arange(s.dims[0]) + 0.5) * s.h + s.origin[0]
    left, right = soot[..., x < 0.0].sum(), soot[..., x > 0.0].sum()
    assert left > 0.2 * (left + right) and right > 0.2 * (left + right), (left, right)
    peak = float(soot.max())
    assert peak < 3.5, 'each half as thick, not whole'


def test_a_volume_pours_liquid_where_it_is_dense_moving_as_it_does(engine, tmp_path):
    from blackbody.engine.liquid import LiquidParams, LiquidSolver, source
    from blackbody.engine.mesh import MeshLibrary
    from blackbody.io.volume import field_source
    # a ball 0.25 m in radius (its densest quarter within 0.19 m of its centre), moving at 1.5 m/s along z
    vdb = _ball_vdb(tmp_path / 'blob.vdb', origin=(-0.3, 0.2, -0.3), voxel=0.025, vel=(0.0, 0.0, 1.5))
    centre = np.array([-0.3, 0.2, -0.3]) + 11.5 * 0.025
    src = field_source(vdb)
    meshes = MeshLibrary(engine.gpu)
    meshes.require([src])
    L = LiquidSolver(engine.gpu, meshes)
    dims, h = LiquidSolver.dims_for((1.2, 1.0, 1.2), 40)
    L.configure(dims, h, (-dims[0] * h / 2, 0.0, -dims[2] * h / 2), LiquidSolver.capacity_for(dims, 8, 4_000_000), 0)
    prm = LiquidParams(open_sides=False, open_top=False, whitewater=False)
    out = {}
    for k in (0.0, 1.0):
        L.reset()
        s = source('volume', pos=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), fill=True, jitter=0.0, mesh=src)
        s.volume_vel = k
        L._prm = prm
        with L.gpu.batch() as b:
            L.step(b, 1.0 / 120.0, prm, [s])
            L.pack(b)
        L.measure()
        pos, vel = L.read_particles()
        out[k] = (len(pos), pos, vel)
    n, pos, vel = out[1.0]
    assert n > 1000 and n == out[0.0][0]
    r = np.linalg.norm(pos - centre, axis=1)
    assert np.percentile(r, 99) < 0.19 + 1.5 * h, 'where the volume is dense'
    assert pos.mean(0) == pytest.approx(centre, abs=1.5 * h)
    assert vel[:, 2].mean() > 1.2 and abs(out[0.0][2][:, 2].mean()) < 0.2, 'starting with the volume\'s velocity'


# -- USD -------------------------------------------------------------------------------------------------------

def _stage(path, start=1001, end=1010, fps=25):
    from pxr import Usd, UsdGeom
    st = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    st.SetStartTimeCode(start)
    st.SetEndTimeCode(end)
    st.SetFramesPerSecond(fps)
    st.SetTimeCodesPerSecond(fps)
    return st


def test_usd_points_become_particles_or_solid_balls(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Gf, UsdGeom
    from blackbody.io import usd as U
    from blackbody.io.volume import load_field
    path = tmp_path / 'pts.usda'
    st = _stage(path)
    gravel = UsdGeom.Points.Define(st, '/Set/Gravel')
    gravel.CreatePointsAttr([(1, 0.05, 1), (1.2, 0.05, 1)])
    gravel.CreateWidthsAttr([0.1, 0.1])
    sparks = UsdGeom.Points.Define(st, '/Set/Sparks')
    pts = sparks.CreatePointsAttr()
    for f in (1001, 1010):
        pts.Set([Gf.Vec3f(0, 1 + 0.05 * (f - 1001), 0), Gf.Vec3f(0.3, 1, 0)], f)
    sparks.CreateWidthsAttr([0.2, 0.2])
    sparks.CreateVelocitiesAttr([(0, 2, 0), (0, 2, 0)])
    st.GetRootLayer().Save()
    info = U.scan(path)
    kinds = {i.path: i for i in info.meshes()}
    assert not kinds['/Set/Gravel'].particles and kinds['/Set/Sparks'].particles
    # still points: balls, round and as big as the balls they stand for (an octahedron held a third of it)
    v, t = U.load_usd_mesh(path, '/Set/Gravel?world')
    assert len(t) == 2 * 80
    one = v[:len(v) // 2] - (1.0, 0.05, 1.0)
    p = one[t[:80]]
    vol = abs(np.einsum('ij,ij->i', p[:, 0], np.cross(p[:, 1], p[:, 2])).sum()) / 6.0
    assert vol == pytest.approx(4.0 / 3.0 * math.pi * 0.05 ** 3, rel=1e-6)
    sc = presets.make('campfire')
    n_e, n_c = len(sc.emitters), len(sc.colliders)
    report = U.import_usd(sc, path)
    assert len(sc.emitters) == n_e + 1 and len(sc.colliders) == n_c + 1, report
    e = sc.emitters[-1]
    assert e['shape'] == 'volume' and e['volume_mode'] == 'hold' and e['volume'].endswith('#/Set/Sparks?world')
    assert any('particles as smoke' in r for r in report)
    # the particles' field: each point a soft ball, moving at its velocity
    g, _ = load_field(sc.item_source(e), 1001)
    nz = g.dims[2]
    dens, vy = g.data[:nz], g.data[2 * nz:3 * nz]
    assert g.vel and dens.max() > 0.9 and vy[dens > 0.5] == pytest.approx(2.0)
    # Solid objects: the particles become balls too
    sc2 = presets.make('campfire')
    U.import_usd(sc2, path, volumes='solid')
    assert len(sc2.emitters) == n_e and len(sc2.colliders) == n_c + 2
    # in a liquid scene, particles become water poured once
    liq = presets.make('water_pour')
    U.import_usd(liq, path)
    assert liq.emitters[-1]['shape'] == 'volume' and liq.emitters[-1]['liquid_mode'] == 'fill'
    # a one-frame cache (a common export): its points and velocities one time sample each, no default value
    one = tmp_path / 'one.usda'
    st = _stage(one)
    cache = UsdGeom.Points.Define(st, '/Set/Cache')
    cache.CreatePointsAttr().Set([Gf.Vec3f(0, 1, 0), Gf.Vec3f(0.3, 1, 0)], 1001)
    cache.CreateVelocitiesAttr().Set([Gf.Vec3f(0, 3, 0), Gf.Vec3f(0, 3, 0)], 1001)
    cache.CreateWidthsAttr([0.2, 0.2])
    st.GetRootLayer().Save()
    assert U.scan(one).meshes()[0].particles, 'its velocities make it particles, not still balls'
    sc3 = presets.make('campfire')
    n_e, n_c = len(sc3.emitters), len(sc3.colliders)
    U.import_usd(sc3, one)
    assert len(sc3.emitters) == n_e + 1 and len(sc3.colliders) == n_c
    g, _ = load_field(sc3.item_source(sc3.emitters[-1]), None)
    nz = g.dims[2]
    assert g.vel and g.data[2 * nz:3 * nz][g.data[:nz] > 0.5] == pytest.approx(3.0)


def test_usd_portal_geometry_and_ies_lights_come_in(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Gf, Sdf, UsdGeom, UsdLux
    from blackbody.io import usd as U
    ies = tmp_path / 'down.ies'
    ies.write_text('IESNA:LM-63-2002\nTILT=NONE\n1 1000 1 3 1 1 2 0.1 0.1 0\n1 1 50\n0 45 90\n0\n1000 800 100\n')
    path = tmp_path / 'lights.usda'
    st = _stage(path)
    fix = UsdLux.RectLight.Define(st, '/Set/Fixture')
    fix.CreateWidthAttr(0.6)
    fix.CreateHeightAttr(0.6)
    fix.CreateIntensityAttr(1000.0)
    UsdLux.ShapingAPI.Apply(fix.GetPrim()).CreateShapingIesFileAttr(Sdf.AssetPath(str(ies)))
    UsdLux.ShadowAPI.Apply(fix.GetPrim()).CreateShadowEnableAttr(False)
    UsdGeom.Xformable(fix).AddTranslateOp().Set(Gf.Vec3d(0, 3, 0))
    UsdGeom.Xformable(fix).AddRotateXOp().Set(-90.0)
    UsdLux.DomeLight.Define(st, '/Set/Sky').CreateIntensityAttr(2.0)
    portal = UsdLux.PortalLight.Define(st, '/Set/Window')
    portal.CreateWidthAttr(2.0)
    portal.CreateHeightAttr(1.0)
    UsdGeom.Xformable(portal).AddTranslateOp().Set(Gf.Vec3d(0, 1.5, -3))
    sign = UsdGeom.Cube.Define(st, '/Set/Sign')
    sign.CreateSizeAttr(1.0)
    UsdGeom.Xformable(sign).AddTranslateOp().Set(Gf.Vec3d(2, 1, 0))
    gl = UsdLux.GeometryLight.Define(st, '/Set/SignLight')
    gl.CreateIntensityAttr(10.0)
    gl.CreateGeometryRel().SetTargets([sign.GetPath()])
    panel = UsdGeom.Mesh.Define(st, '/Set/Panel')
    panel.CreatePointsAttr([(-1, 0.5, -1), (1, 0.5, -1), (1, 0.5, 1), (-1, 0.5, 1)])
    panel.CreateFaceVertexCountsAttr([4])
    panel.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    UsdLux.MeshLightAPI.Apply(panel.GetPrim())
    UsdLux.LightAPI(panel.GetPrim()).CreateIntensityAttr(5.0)
    st.GetRootLayer().Save()
    sc = presets.make('campfire')
    sc.lights = []
    report = U.import_usd(sc, path)
    L = {l['name']: l for l in sc.lights}
    fx = L['Fixture']
    assert fx['kind'] == 'point' and fx['profile'].endswith('down.ies') and not fx['profile_brightness']
    assert fx['intensity'] == pytest.approx(1000.0 * 0.36) and fx['shadows'] is False
    assert fx['direction'] == pytest.approx((0.0, -1.0, 0.0), abs=1e-6)
    w = L['Window']
    assert w['kind'] == 'area' and (w['width'], w['height']) == pytest.approx((2.0, 1.0))
    assert w['intensity'] == pytest.approx(2.0 * 2.0), "the dome's radiance through its 2 m²"
    assert w['direction'] == pytest.approx((0.0, 0.0, -1.0), abs=1e-6) and w['position'] == pytest.approx((0, 1.5, -3))
    s = L['SignLight']
    assert s['kind'] == 'point' and s['position'] == pytest.approx((2.0, 1.0, 0.0))
    assert s['intensity'] == pytest.approx(10.0 * 6.0 / 4.0) and s['radius'] == pytest.approx(0.5)
    p = L['Panel']
    assert p['position'] == pytest.approx((0.0, 0.5, 0.0)) and p['intensity'] == pytest.approx(5.0 * 4.0 / 4.0)
    assert not any('not imported' in r for r in report), report


def test_usd_frame_offset_and_range(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Gf, UsdGeom
    from blackbody.io import usd as U
    path = tmp_path / 'shot.usda'
    st = _stage(path, 1001, 1010, 25)
    cam = UsdGeom.Camera.Define(st, '/Shot/Cam')
    op = UsdGeom.Xformable(cam).AddTranslateOp()
    op.Set(Gf.Vec3d(0, 1, 5), 1001)
    op.Set(Gf.Vec3d(1, 1, 5), 1010)
    blob = UsdGeom.Mesh.Define(st, '/Shot/Blob')
    blob.CreateFaceVertexCountsAttr([3])
    blob.CreateFaceVertexIndicesAttr([0, 1, 2])
    pts = blob.CreatePointsAttr()
    pts.Set([(0, 0, 0), (1, 0, 0), (0, 1, 0)], 1001)
    pts.Set([(0, 0, 0), (2, 0, 0), (0, 1, 0)], 1010)
    st.GetRootLayer().Save()
    sc = presets.make('campfire')
    report = U.import_usd(sc, path, offset=1000, match_range=True)
    r = sc.data['render']
    assert (r['start'], r['end'], r['fps']) == (1, 10, 25.0) and report[0].startswith('frames 1-10 at 25 fps')
    assert sc.v('camera', 'position', 1) == pytest.approx((0.0, 1.0, 5.0))
    assert sc.v('camera', 'position', 10) == pytest.approx((1.0, 1.0, 5.0))
    c = sc.colliders[-1]
    assert c['mesh_offset'] == -1000.0
    assert sc._mesh_time(c, 10)['mesh_frame'] == 1010.0, 'the deforming mesh is read at its own USD frame'
    from blackbody.engine.mesh import load_mesh
    v, _ = load_mesh(sc.item_source(c), sc._mesh_time(c, 10)['mesh_frame'])
    assert v[:, 0].max() == pytest.approx(2.0)
    # time-code numbered shots: the offset lands in Mesh frame offset, whose limits take all the dialog and
    # --usd-offset allow (past them, the import says so rather than leave objects out of step)
    from blackbody.scene.params import COLLIDER_PARAMS, EMITTER_PARAMS, FRAME_OFFSET_MAX
    sc3 = presets.make('campfire')
    U.import_usd(sc3, path, offset=86400)
    assert sc3.colliders[-1]['mesh_offset'] == -86400.0
    for p in (p for p in EMITTER_PARAMS + COLLIDER_PARAMS if p.key == 'mesh_offset'):
        assert p.lo <= -FRAME_OFFSET_MAX and p.hi >= FRAME_OFFSET_MAX, 'dragging it keeps it, not the slider end'
        assert (p.hard_lo, p.hard_hi) == (-FRAME_OFFSET_MAX, FRAME_OFFSET_MAX)
    with pytest.raises(U.USDError, match='out of range'):
        U.import_usd(presets.make('campfire'), path, offset=int(FRAME_OFFSET_MAX) + 1)
    # with footage, only where the shot starts (the footage keeps its length and rate)
    sc2 = presets.make('campfire')
    sc2.data['render'].update(start=1, end=48, fps=24.0)
    U.import_usd(sc2, path, match_range='start')
    assert (sc2.data['render']['start'], sc2.data['render']['end'], sc2.data['render']['fps']) == (1001, 1048, 24.0)


def test_the_command_line_takes_the_usd_range_and_offset(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Gf, UsdGeom
    from blackbody.cli import _usd_and_cache
    path = tmp_path / 'shot.usda'
    st = _stage(path, 1001, 1100, 25)
    op = UsdGeom.Xformable(UsdGeom.Camera.Define(st, '/Shot/Cam')).AddTranslateOp()
    op.Set(Gf.Vec3d(0, 1, 5), 1001)
    op.Set(Gf.Vec3d(1, 1, 5), 1100)
    st.GetRootLayer().Save()

    def run(**kw):
        sc = presets.make('campfire')
        a = dict(usd=str(path), usd_offset=0, usd_keep_range=False, footage=None, quiet=True, cache=None, from_cache=None)
        a.update(kw)
        assert _usd_and_cache(sc, argparse.Namespace(**a)) is None
        return sc
    sc = run()
    assert (sc.start, sc.end, sc.fps) == (1001, 1100, 25.0), 'the preset no longer starts the shot at its own frame'
    assert sc.v('camera', 'position', 1001) == pytest.approx((0.0, 1.0, 5.0))
    sc = run(usd_offset=1000)
    assert (sc.start, sc.end) == (1, 100) and sc.v('camera', 'position', 100) == pytest.approx((1.0, 1.0, 5.0))
    base = presets.make('campfire')
    sc = run(usd_keep_range=True)
    assert (sc.start, sc.end, sc.fps) == (base.start, base.end, base.fps)
