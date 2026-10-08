"""Lists the GPU fills, in an order that is the same every run: places handed out by counts per block and their running
totals (block_scan.wgsl), glowing matter's lights (the brightest kept past the cap: matter_glow.wgsl), the pieces of ice
(numbered by their roots' bricks of cells: liq_ice.wgsl) and the radiant sources (in the order of their cells:
rad_build.wgsl). An atomic append hands places out in the order the threads happen to run, which differs from run to
run; past a cap it kept whichever came first."""
import numpy as np
import pytest

from blackbody.engine.gpu import Uniforms


@pytest.fixture(scope='module')
def gpu(engine):
    return engine.gpu


def test_block_starts_are_the_running_totals_of_the_counts(gpu):
    k = gpu.kernel('block_scan.wgsl', ['rbuf', 'buf'], workgroup=(256, 1, 1))
    rng = np.random.default_rng(4)
    for n in (1, 5, 255, 1000, 31251):
        cnt = np.zeros(-(-n // 4) * 4, np.uint32)
        cnt[:n] = rng.integers(0, 65, n)
        a = gpu.buffer(cnt.nbytes, 'scan-counts')
        o = gpu.buffer((n + 1) * 4, 'scan-starts')
        gpu.write_buffer(a, cnt)
        for size in (0, 64):
            with gpu.batch() as b:
                b.run(k, [a, o], Uniforms().v4(n, size, 64 * n - 7), groups=(1, 1, 1))
            got = np.frombuffer(gpu.read_buffer(o), np.uint32)
            c = cnt[:n].astype(np.int64)
            if size:   # (what each block holds besides its count: its width, the last one 57 wide here)
                width = np.minimum(64, 64 * n - 7 - 64 * np.arange(n))
                c = width - np.minimum(c, width)
            assert np.array_equal(got, np.concatenate([[0], np.cumsum(c)])), (n, size)
        a.destroy()
        o.destroy()


def _glow(gpu, st, dims, temps, first=0):
    """Matter's glow lights from a heap whose every node is on its surface at these temperatures (K); `first` hot
    objects' lights before them. (lights, the counters, each block's light)"""
    from blackbody.engine import stage as S
    from blackbody.engine.liquid_render import lava_table
    phi = gpu.texture3d(dims, 'r32float', 'glow-test-phi')
    look = gpu.texture3d(dims, 'r32float', 'glow-test-look')
    gpu.upload(phi, np.zeros(dims[::-1] + (1,), np.float32))
    gpu.upload(look, temps.astype(np.float32).reshape(dims[::-1] + (1,)))
    blocks = [-(-d // S.GLOW_BLOCK) for d in dims]
    nb = int(np.prod(blocks))
    if st._mlb is None or st._mlb.size < nb * 32:
        st._mlb = gpu.buffer(nb * 32, 'glow-test-blocks')
    gpu.write_buffer(st._mlc, np.array([first, 0, 0, 0], np.uint32))
    gu = (Uniforms().v4(*dims, S.GLOW_BLOCK).v4(0.0, 0.0, 0.0, 0.01).v4(1.0, 1.0, S.GLOW_T0, 1.0 / S.GLOW_DT)
          .v4(S.MATTER_LIGHTS, nb))
    for row in lava_table(S.GLOW_T0, S.GLOW_DT, 16):
        gu.v4(*row)
    res = [phi, look, st._ml, st._mlc, st._mlb]
    with gpu.batch() as b:
        b.run(st.k_mglow, res, gu, groups=tuple(-(-n // 4) for n in blocks))
        b.run(st.k_mglow_finish, res, gu, groups=(1, 1, 1))
    st._ml_ran = True
    ml = np.frombuffer(gpu.read_buffer(st._ml), np.float32).reshape(-1, 4).copy()
    mlc = np.frombuffer(gpu.read_buffer(st._mlc), np.uint32).copy()
    bl = np.frombuffer(gpu.read_buffer(st._mlb, nb * 32), np.float32).reshape(nb, 2, 4).copy()
    return ml, mlc, bl


def test_glowing_matter_keeps_its_brightest_lights_the_same_every_run(gpu):
    from blackbody.engine import stage as S
    st = S.Stage(gpu)
    dims = (96, 64, 96)                       # 12 x 8 x 12 = 1152 blocks, all glowing
    rng = np.random.default_rng(5)
    temps = 900.0 + 800.0 * rng.random(dims[::-1])
    runs = [_glow(gpu, st, dims, temps, first=10) for _ in range(3)]
    ml, mlc, bl = runs[0]
    assert all(np.array_equal(ml, r[0]) for r in runs[1:]), 'the same lights, in the same order, every run'
    room = S.MATTER_LIGHTS - 10
    assert int(ml[0, 0]) == S.MATTER_LIGHTS and mlc[1] == 1152 and mlc[2] == 1152 - room
    keep = np.sort(np.argsort(-bl[:, 1, 3], kind='stable')[:room])
    got = ml[1 + 2 * 10:1 + 2 * S.MATTER_LIGHTS:2]
    assert np.allclose(got[:, :3], bl[keep, 0, :3]) and np.allclose(ml[2 + 2 * 10::2][:room, :3], bl[keep, 1, :3]), \
        'the brightest, in the order of their blocks'
    assert st.matter_lights_cut() == 1152 - room
    st.clear_notes()
    assert st.matter_lights_cut() == 0
    # few enough to fit: every glowing block, in order (cold blocks give none)
    temps[:, :, :80] = 300.0                  # (two of the twelve columns of blocks glowing: 192)
    ml, mlc, bl = _glow(gpu, st, dims, temps)
    glowing = np.nonzero(bl[:, 1, 3] > 0.0)[0]
    assert int(ml[0, 0]) == len(glowing) == mlc[1] == 192 and mlc[2] == 0
    assert np.allclose(ml[1:1 + 2 * 192:2, :3], bl[glowing, 0, :3])
    assert st.matter_lights_cut() == 0
    # no room left after the hot objects: none
    ml, mlc, _ = _glow(gpu, st, dims, temps, first=S.MATTER_LIGHTS)
    assert int(ml[0, 0]) == S.MATTER_LIGHTS and mlc[2] == 192


def test_pieces_of_ice_are_numbered_the_same_every_run(gpu):
    from blackbody.engine.liquid import LiquidParams, LiquidSolver, source
    from blackbody.engine.liquid_thermal import MAX_BODIES
    L = LiquidSolver(gpu)
    dims, h = LiquidSolver.dims_for((0.4, 0.4, 0.4), 64)
    L.configure(dims, h, (-dims[0] * h / 2, 0.0, -dims[2] * h / 2), LiquidSolver.capacity_for(dims, 8, 4_000_000), 0)
    prm = LiquidParams(thermal=True, temp=-5.0, ground_temp=-5.0, air_temp=-5.0, open_sides=False, min_source_temp=-5.0)
    L._prm = prm
    with gpu.batch() as b:
        L.step(b, 1 / 60, prm, [source('box', (0.0, 0.05, 0.0), (0.1, 0.05, 0.1), fill=True, jitter=0.0, vel_blend=0.0)])
    th = L.thermal
    nx, ny, nz = dims
    cells = np.zeros((nz, ny, nx, 4), np.float32)
    cells[::2, ::2, ::2, 2:] = 1.0            # frozen cells on their own: 32^3 pieces, four times the most bodies
    outs = []
    for _ in range(3):
        gpu.upload(th.cells, cells)
        with gpu.batch() as b:
            th.ice_rigid(b, prm, L.VA, L.VB)
        outs.append(np.frombuffer(gpu.read_buffer(th.bodycell), np.uint32).copy())
    assert all(np.array_equal(outs[0], o) for o in outs[1:])
    # numbered by their bricks of 8 x 8 x 4 cells, then the cells in each; past MAX_BODIES none
    i = np.arange(nx * ny * nz)
    x, y, z = i % nx, (i // nx) % ny, i // (nx * ny)
    bx, by = -(-nx // 8), -(-ny // 8)
    brick = x // 8 + bx * (y // 8 + by * (z // 4))
    local = x % 8 + 8 * (y % 8 + 8 * (z % 4))
    roots = (x % 2 == 0) & (y % 2 == 0) & (z % 2 == 0)
    order = i[roots][np.lexsort((local[roots], brick[roots]))]
    want = np.full(i.size, 0xFFFFFFFF, np.uint32)
    want[order[:MAX_BODIES]] = np.arange(MAX_BODIES)
    assert np.array_equal(outs[0], want)
    assert th.most_pieces() == 32 ** 3


def test_radiant_sources_are_listed_in_the_order_of_their_cells(gpu):
    from blackbody.engine.radiant import Radiant
    R = Radiant(gpu)
    dims = (48, 64, 48)
    R.layout((0.0, 0.0, 0.0), dims, 0.05, gas=True)
    _, cs, cd, bs, _ = R.grid
    rng = np.random.default_rng(2)
    heat = np.where(rng.random(dims[::-1]) < 0.1, rng.random(dims[::-1]) * 1.5, 0.0).astype(np.float32)
    scal = gpu.texture3d(dims, 'rgba32float', 'radiant-test-gas')
    a = np.zeros(dims[::-1] + (4,), np.float32)
    a[..., 0] = heat
    gpu.upload(scal, a)
    R.set_objects([((0.1 * j, 0.2, 0.3), 0.01, 50.0 + j, (0.0, 1.0, 0.0), 0) for j in range(5)])
    lists = []
    for _ in range(3):
        with gpu.batch() as b:
            R.build(b, scal, dims, 293.0)
        n = int(np.frombuffer(gpu.read_buffer(R.RLC, 4), np.uint32)[0])
        lists.append(np.frombuffer(gpu.read_buffer(R.RL, n * 48), np.float32).reshape(n, 3, 4).copy())
    assert all(np.array_equal(lists[0], o) for o in lists[1:]), 'the same list, in the same order, every run'
    rl = lists[0]
    # the coarse cells that radiate, in order (z, then y, then x), then the five objects' faces
    hot = (heat > 0.05).reshape(cd[2], bs, cd[1], bs, cd[0], bs).any(axis=(1, 3, 5))
    assert len(rl) == int(hot.sum()) + 5
    cells = rl[:-5, 0, :3] / cs                               # (each source's centre lies in its cell)
    idx = (np.floor(cells[:, 2]) * cd[1] + np.floor(cells[:, 1])) * cd[0] + np.floor(cells[:, 0])
    assert np.all(np.diff(idx) > 0)
    assert np.allclose(rl[-5:, 0, 0], [0.1 * j for j in range(5)], atol=1e-6)
