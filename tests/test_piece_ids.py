"""Broken pieces told apart by number (bodyfield.py, bodies_sdf.wgsl): the matter reads back what it gave each piece by
the number written in its field, which must stay exact however many pieces there are. The GPU-free ones first."""
from types import SimpleNamespace

import numpy as np

from blackbody.engine import bodyfield
from blackbody.engine.gpu import Uniforms

H = 0.01   # (the grid's cell, m)


def _cubes(centres, half=0.4 * H):
    """piece_records for little cubes (half-size `half`) at these centres, each its own collider's only piece."""
    def records(scene, pieces, rows=None, owners=None):
        c = np.asarray(centres, float)
        planes = np.array([[1, 0, 0, half], [-1, 0, 0, half], [0, 1, 0, half], [0, -1, 0, half],
                           [0, 0, 1, half], [0, 0, -1, half]], np.float32)
        rad = float(np.sqrt(3.0) * half)
        P = np.zeros((len(c), 5, 4), np.float32)
        P[:, 0, :3], P[:, 0, 3] = c, len(planes)
        P[:, 1, 3] = 1.0            # (unturned)
        P[:, 4, 3] = rad
        if owners is not None:
            owners.extend((k, 0) for k in range(len(c)))
        return P, planes, c, np.full(len(c), rad)
    return records


class _FakeGPU:
    def kernel(self, *a, **k):
        return None

    def buffer(self, size, label=''):
        return SimpleNamespace(size=size, destroy=lambda: None)

    def write_buffer(self, buf, data, offset=0):
        pass


def test_only_the_pieces_that_reach_the_grid_are_numbered(monkeypatch):
    # (2,000 pieces in the scene, three of them by the grid: those are 0, 1, 2, and the rest take no numbers)
    far = [(5.0 + 0.1 * k, 5.0, 5.0) for k in range(2000)]
    near = [(0.05, 0.05, 0.05), (0.15, 0.05, 0.05), (0.25, 0.1, 0.05)]
    centres = far[:700] + near[:1] + far[700:1500] + near[1:] + far[1500:]
    monkeypatch.setattr(bodyfield, 'piece_records', _cubes(centres))
    bf = bodyfield.BodyField(_FakeGPU())
    assert bf.prepare(None, [{0: None}], (32, 16, 16), H, (0.0, 0.0, 0.0))
    assert bf.owners[0] == [(700, 0), (1501, 0), (1502, 0)]
    assert bf.steps[0][2] == 3


def test_the_matters_push_rows_grow_with_the_pieces():
    from blackbody.engine.matter import Matter, PIECE_ROWS
    m = SimpleNamespace(gpu=_FakeGPU(), _buf={'react_i': SimpleNamespace(size=(16 + PIECE_ROWS) * 24, destroy=lambda: None)})
    Matter._react_rows(m, 100)
    assert m._buf['react_i'].size == (16 + PIECE_ROWS) * 24        # (room enough already)
    Matter._react_rows(m, 9000)
    assert m._buf['react_i'].size >= (16 + 9000) * 24


# ---- on the GPU -------------------------------------------------------------------------------------------------------

class _Grid:
    """A grid as BodyField.bake takes it (matter.py _PieceGrid): its distance to solids and solid velocity."""

    def __init__(self, gpu, dims, fmt):
        self.dims, self.h, self.origin = dims, H, np.zeros(3)
        self.sdf = gpu.texture3d(dims, 'r32float', 'test-sdf')
        gpu.upload(self.sdf, np.full((dims[2], dims[1], dims[0], 1), 1.0e6, np.float32))
        self._svel = gpu.texture3d(dims, fmt, 'test-svel')

    def _grid(self, dt):
        return Uniforms().v4(*self.dims, self.h).v4(*self.origin, 0.0).v4(0.0, 0.0, 0.0, dt)

    def solid_vel(self):
        return self._svel


def test_every_one_of_thousands_of_pieces_keeps_its_own_number(engine, monkeypatch):
    # 2,704 little cubes, one in each cell of a 52 x 52 layer: each cell must name its own (a half float's whole numbers
    # stop at 2048, past which the matter's push went to the piece next to the one it pushed)
    gpu = engine.gpu
    n = 52
    ij = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing='ij'), -1).reshape(-1, 2)   # (k = i * n + j)
    centres = np.column_stack([(ij[:, 0] + 0.5) * H, (ij[:, 1] + 0.5) * H, np.full(len(ij), 0.5 * H)])
    monkeypatch.setattr(bodyfield, 'piece_records', _cubes(centres))
    named = {}
    for fmt in ('rgba32float', 'rgba16float'):
        grid = _Grid(gpu, (n, n, 1), fmt)
        bf = bodyfield.BodyField(gpu, fmt)
        assert bf.prepare(None, [{0: None}], grid.dims, H, grid.origin)
        with gpu.batch() as b:
            assert bf.bake(b, grid, 0)
        w = gpu.read(grid.solid_vel()).astype(np.float64)[0, :, :, 3]   # (y, x)
        named[fmt] = w[ij[:, 1], ij[:, 0]] - 1.0
    k = np.arange(len(ij))
    assert np.array_equal(named['rgba32float'], k)                  # (the matter's field: every piece its own)
    assert (named['rgba16float'] >= 0.0).all()                       # (the solvers' read only that a piece is there,
    assert not np.array_equal(named['rgba16float'], k)               # which is all a half float can tell past 2048)
