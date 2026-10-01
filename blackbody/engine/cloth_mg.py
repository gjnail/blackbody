"""Fabric bending solved by multigrid (engine/cloth.py).

Bending is the fourth-order part of cloth: the constraint passes carry its stiffness only a cell or
so per pass, so on a fine cloth a stiff fabric would need hundreds of passes to stand up as it should.
For a panel (its cloth flat at rest, so the isometric bending energy is exactly quadratic), each
substep's bending is one linear system per coordinate,

    (M / h^2 + Q) x = M x_predicted / h^2,      Q = sum over hinges of k_h K_h K_h^T,

the same equation the bending passes converge to (the same Bergou weights K and compliance), and
multigrid solves it to convergence in about a V-cycle a substep at any detail.

The fine level is the cloth itself, applied hinge by hinge (so hinges with a burnt-away vertex drop
out). Coarser levels keep every other row and column of each panel's grid; their operators are the
Galerkin products P^T A P (bilinear P in the grid's index space, nothing prolonged onto pins), kept as
a mass part and a bending part so the substep length can change, which makes the two-grid correction
convergent whatever the smoother. Damped Jacobi smooths, its weight from a bound on each level's
largest eigenvalue. Levels are built here (numpy) and run on the GPU (cloth_mg.wgsl), all panels at
once: level L holds every panel's level-L grid, block by block.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MIN_SIDE = 3          # stop coarsening at this many vertices across
MAX_LEVELS = 7
SMOOTH_OMEGA = 1.3    # Jacobi weight times the inverse of the largest eigenvalue of D^-1 A
CHEB_MARGIN = 1.1     # Chebyshev smoothing: over-estimate of the largest eigenvalue of D^-1 A
CHEB_RATIO = 12.0     # and the part of the spectrum it damps: [lmax / CHEB_RATIO, lmax]


# -- small sparse helpers (numpy only) ----------------------------------------------------------------

def coo_sum(rows, cols, vals, n_rows):
    """Duplicates summed; sorted by row, then column. Returns (rows, cols, vals)."""
    if len(rows) == 0:
        return rows.astype(np.int64), cols.astype(np.int64), vals
    key = rows.astype(np.int64) * (int(cols.max()) + 1) + cols.astype(np.int64)
    order = np.argsort(key, kind='stable')
    key = key[order]
    vals = vals[order]
    first = np.concatenate([[True], key[1:] != key[:-1]])
    idx = np.nonzero(first)[0]
    v = np.add.reduceat(vals, idx, axis=0)
    r = rows[order][idx]
    c = cols[order][idx]
    return r.astype(np.int64), c.astype(np.int64), v


def csr(rows, cols, vals, n):
    """(row starts (n + 1), cols, vals) from sorted COO."""
    start = np.searchsorted(rows, np.arange(n + 1))
    return start, cols, vals


def csr_mv(A, x):
    start, cols, vals = A
    y = vals[:, None] * x[cols] if x.ndim == 2 else vals * x[cols]
    n = len(start) - 1
    out = np.zeros((n,) + x.shape[1:])
    rowid = np.repeat(np.arange(n), np.diff(start))
    np.add.at(out, rowid, y)
    return out


def galerkin(rows, cols, vals, P, nc):
    """P^T A P for A in COO (fine indices) and P = (fine rows, coarse cols, weights)."""
    pr, pc, pw = P
    order = np.argsort(pr, kind='stable')
    pr, pc, pw = pr[order], pc[order], pw[order]
    nf = int(max(rows.max() if len(rows) else 0, pr.max() if len(pr) else 0)) + 1
    start = np.searchsorted(pr, np.arange(nf + 1))
    cnt = np.diff(start)
    # expand every nonzero (i, j, a) into its P_i x P_j coarse pairs
    ci = cnt[rows]
    cj = cnt[cols]
    k = ci * cj
    keep = k > 0
    rows, cols, vals, ci, cj, k = rows[keep], cols[keep], vals[keep], ci[keep], cj[keep], k[keep]
    tot = int(k.sum())
    ent = np.repeat(np.arange(len(rows)), k)
    off = np.arange(tot) - np.repeat(np.cumsum(k) - k, k)
    a = off // cj[ent]
    b = off % cj[ent]
    ia = start[rows[ent]] + a
    ib = start[cols[ent]] + b
    r = pc[ia]
    c = pc[ib]
    wgt = pw[ia] * pw[ib]
    v = vals[ent] * (wgt[:, None] if vals.ndim == 2 else wgt)
    return coo_sum(r, c, v, nc)


@dataclass
class Level:
    n: int
    pinned: np.ndarray                       # (n,) bool: held (not solved for)
    mass: np.ndarray = None                  # level 0: lumped (n,)
    hv: np.ndarray = None                    # level 0: hinges (h, 4), weights (h, 4), stiffness (h,)
    hk: np.ndarray = None
    hs: np.ndarray = None
    A: tuple = None                          # coarse: CSR (start, cols, vals (nnz, 2): mass, bending)
    lam_m: float = 1.0                       # largest eigenvalue of diag(M)^-1 M
    lam_q: float = 1.0                       # largest eigenvalue of diag(Q)^-1 Q
    P: tuple = None                          # prolongation from the next level: fine rows, coarse cols, weights
    fabric: np.ndarray = None                # (n,) which fabric each vertex is of


def level0_apply(lv, x, h):
    v = np.einsum('hk,hkd->hd', lv.hk, x[lv.hv]) * lv.hs[:, None]
    out = lv.mass[:, None] / (h * h) * x
    for c in range(4):
        np.add.at(out, lv.hv[:, c], lv.hk[:, c:c + 1] * v)
    return out


def coarse_apply(lv, x, h):
    s, c, v = lv.A
    return csr_mv((s, c, v[:, 0] / (h * h) + v[:, 1]), x)


def apply(lv, x, h):
    return level0_apply(lv, x, h) if lv.A is None else coarse_apply(lv, x, h)


def diag(lv, h):
    if lv.A is None:
        d = lv.mass / (h * h)
        dq = np.zeros(lv.n)
        for c in range(4):
            np.add.at(dq, lv.hv[:, c], lv.hs * lv.hk[:, c] ** 2)
        return d + dq
    s, c, v = lv.A
    rowid = np.repeat(np.arange(lv.n), np.diff(s))
    on = rowid == c
    d = np.zeros(lv.n)
    np.add.at(d, rowid[on], v[on, 0] / (h * h) + v[on, 1])
    return d


def smoother_weight(lv):
    return SMOOTH_OMEGA / max(lv.lam_m, lv.lam_q, 1.0)


def _power(mv, d, pinned, iters=40):
    n = len(d)
    rng = np.random.default_rng(1)
    x = rng.standard_normal(n)
    x[pinned] = 0
    lam = 1.0
    for _ in range(iters):
        y = mv(x) / np.maximum(d, 1e-30)
        y[pinned] = 0
        nx = np.linalg.norm(x)
        lam = float(np.linalg.norm(y) / max(nx, 1e-30))
        x = y / max(np.linalg.norm(y), 1e-30)
    return lam


def _grid_of(uv):
    us = np.unique(np.round(uv[:, 0], 9))
    vs = np.unique(np.round(uv[:, 1], 9))
    i = np.searchsorted(us, np.round(uv[:, 0], 9))
    j = np.searchsorted(vs, np.round(uv[:, 1], 9))
    return len(us), len(vs), i, j


def _coarse_index(n):
    c = list(range(0, n, 2))
    if c[-1] != n - 1:
        c.append(n - 1)
    return np.array(c)


def _interp(fine_n, keep):
    a = np.searchsorted(keep, np.arange(fine_n), side='right') - 1
    a = np.clip(a, 0, len(keep) - 2)
    t = (np.arange(fine_n) - keep[a]) / np.maximum(keep[a + 1] - keep[a], 1)
    return a, a + 1, t


def _prolong(grid, ku, kv, cnu):
    """Bilinear prolongation for one panel's grid (fine vertex ids in `grid`, (nv, nu))."""
    fnv, fnu = grid.shape
    au, bu, tu = _interp(fnu, ku)
    av, bv, tv = _interp(fnv, kv)
    fj, fi = np.meshgrid(np.arange(fnv), np.arange(fnu), indexing='ij')
    fine = grid[fj, fi].reshape(-1)
    rows, cols, wts = [], [], []
    for ca, wa in ((au, 1.0 - tu), (bu, tu)):
        for cb, wb in ((av, 1.0 - tv), (bv, tv)):
            w = (wa[fi] * wb[fj]).reshape(-1)
            c = (cb[fj] * cnu + ca[fi]).reshape(-1)
            k = w > 1e-9
            rows.append(fine[k])
            cols.append(c[k])
            wts.append(w[k])
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)


@dataclass
class Hierarchy:
    levels: list = field(default_factory=list)
    vertices: np.ndarray = None   # level 0's vertices in the cloth (panels only)


def build(panels, n_cloth):
    """Multigrid for the panels of a cloth. `panels`: per panel (cloth vertex ids (m,), uv (m, 2),
    mass (m,), pinned (m,), hinge vertices (h, 4) in cloth ids, hinge weights (h, 4), hinge stiffness (h,)).
    Level 0 works on cloth vertex ids directly; returns None when there are no panels."""
    if not panels:
        return None
    # level 0: the whole cloth, panel vertices free (unless pinned), everything else held
    pinned0 = np.ones(n_cloth, bool)
    mass0 = np.ones(n_cloth)
    hv, hk, hs = [], [], []
    for p in panels:
        ids, uv, mass, pin, h_v, h_k, h_s = p
        pinned0[ids] = pin
        mass0[ids] = mass
        hv.append(h_v)
        hk.append(h_k)
        hs.append(h_s)
    lv0 = Level(n_cloth, pinned0, mass=mass0, hv=np.concatenate(hv), hk=np.concatenate(hk), hs=np.concatenate(hs))
    d_q = diag(lv0, 1e30)
    lv0.lam_q = _power(lambda x: level0_apply(lv0, np.stack([x] * 3, -1), 1e30)[:, 0], d_q, pinned0)
    levels = [lv0]
    # the panels' grids, coarsened together
    grids = []
    for p in panels:
        ids, uv = p[0], p[1]
        nu, nv, i, j = _grid_of(uv)
        g = np.full((nv, nu), -1, np.int64)
        g[j, i] = ids
        grids.append(g)
    # A of level 0 as COO, mass and bending parts
    rows_m = np.arange(n_cloth)
    cols_m = np.arange(n_cloth)
    vals_m = np.stack([mass0, np.zeros(n_cloth)], -1)
    hr = np.repeat(lv0.hv, 4, axis=1).reshape(-1)
    hc = np.tile(lv0.hv, (1, 4)).reshape(-1)
    hq = (lv0.hk[:, :, None] * lv0.hk[:, None, :] * lv0.hs[:, None, None]).reshape(-1)
    A_rows = np.concatenate([rows_m, hr])
    A_cols = np.concatenate([cols_m, hc])
    A_vals = np.concatenate([vals_m, np.stack([np.zeros_like(hq), hq], -1)])
    cur_pin = pinned0
    for level in range(1, MAX_LEVELS):
        if all(min(g.shape) <= MIN_SIDE for g in grids):
            break
        rows, cols, wts, new_grids = [], [], [], []
        base = 0
        for g in grids:
            fnv, fnu = g.shape
            ku = _coarse_index(fnu) if fnu > MIN_SIDE else np.arange(fnu)
            kv = _coarse_index(fnv) if fnv > MIN_SIDE else np.arange(fnv)
            r, c, w = _prolong(g, ku, kv, len(ku))
            rows.append(r)
            cols.append(c + base)
            wts.append(w)
            new_grids.append(np.arange(len(ku) * len(kv)).reshape(len(kv), len(ku)) + base)
            base += len(ku) * len(kv)
        nc = base
        rows, cols, wts = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
        free = ~cur_pin[rows]                       # nothing is prolonged onto a held vertex
        P = (rows[free], cols[free], wts[free])
        levels[-1].P = P
        r, c, v = galerkin(A_rows, A_cols, A_vals, P, nc)
        # a coarse vertex no free fine vertex depends on is held
        pin_c = np.ones(nc, bool)
        pin_c[np.unique(P[1])] = False
        lv = Level(nc, pin_c, A=csr(r, c, v, nc))
        s, cc, vv = lv.A
        d_m = np.zeros(nc)
        d_q = np.zeros(nc)
        rowid = np.repeat(np.arange(nc), np.diff(s))
        on = rowid == cc
        np.add.at(d_m, rowid[on], vv[on, 0])
        np.add.at(d_q, rowid[on], vv[on, 1])
        lv.lam_m = _power(lambda x: csr_mv((s, cc, vv[:, 0]), x), d_m, pin_c) if (d_m > 0).any() else 1.0
        lv.lam_q = _power(lambda x: csr_mv((s, cc, vv[:, 1]), x), d_q, pin_c) if (d_q > 0).any() else 1.0
        levels.append(lv)
        A_rows, A_cols, A_vals = r, c, v
        grids = new_grids
        cur_pin = pin_c
    return Hierarchy(levels, np.concatenate([p[0] for p in panels]))


# -- a CPU reference of the cycle the GPU runs ----------------------------------------------------------

def jacobi(lv, x, b, h, sweeps):
    d = diag(lv, h)
    w = smoother_weight(lv)
    for _ in range(sweeps):
        r = b - apply(lv, x, h)
        r[lv.pinned] = 0
        x = x + w * r / np.where(d > 0, d, 1.0)[:, None]
    return x


def chebyshev(lv, x, b, h, degree):
    """Chebyshev polynomial smoothing of D^-1 A over [lmax / CHEB_RATIO, lmax]: one product with A a step,
    like Jacobi, but every high-frequency error damped at once."""
    d = diag(lv, h)
    dinv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)[:, None]
    lmax = CHEB_MARGIN * max(lv.lam_m, lv.lam_q, 1.0)
    lmin = lmax / CHEB_RATIO
    theta, delta = 0.5 * (lmax + lmin), 0.5 * (lmax - lmin)
    sigma = theta / delta
    rho = 1.0 / sigma
    r = (b - apply(lv, x, h)) * dinv
    r[lv.pinned] = 0
    dx = r / theta
    for _ in range(degree):
        x = x + dx
        r = (b - apply(lv, x, h)) * dinv
        r[lv.pinned] = 0
        rho_n = 1.0 / (2.0 * sigma - rho)
        dx = rho_n * rho * dx + 2.0 * rho_n / delta * r
        rho = rho_n
    return x + dx


def vcycle(levels, k, x, b, h, pre=3, post=3, coarse=30, smooth=None):
    smooth = smooth or chebyshev
    lv = levels[k]
    if k == len(levels) - 1:
        return chebyshev(lv, x, b, h, coarse)
    x = smooth(lv, x, b, h, pre)
    r = b - apply(lv, x, h)
    r[lv.pinned] = 0
    rows, cols, wts = lv.P
    nc = levels[k + 1].n
    rc = np.zeros((nc, 3))
    np.add.at(rc, cols, wts[:, None] * r[rows])
    ec = vcycle(levels, k + 1, np.zeros((nc, 3)), rc, h, pre, post, coarse, smooth)
    e = np.zeros_like(x)
    np.add.at(e, rows, wts[:, None] * ec[cols])
    e[lv.pinned] = 0
    return smooth(lv, x + e, b, h, post)


# -- the GPU side ---------------------------------------------------------------------------------------

def _cheb_coefs(lv, steps):
    """(weight of the last step, weight of D^-1 r) for each of `steps` Chebyshev dispatches."""
    lmax = CHEB_MARGIN * max(lv.lam_m, lv.lam_q, 1.0)
    lmin = lmax / CHEB_RATIO
    theta, delta = 0.5 * (lmax + lmin), 0.5 * (lmax - lmin)
    sigma = theta / delta
    rho = 1.0 / sigma
    out = [(0.0, 1.0 / theta)]
    for _ in range(steps - 1):
        rho_n = 1.0 / (2.0 * sigma - rho)
        out.append((rho_n * rho, 2.0 * rho_n / delta))
        rho = rho_n
    return out


def _vertex_hinges(hv, n):
    """Per vertex: (first, count) then (hinge << 2 | slot) entries, as u32."""
    h = np.repeat(np.arange(len(hv), dtype=np.int64), 4)
    s = np.tile(np.arange(4, dtype=np.int64), len(hv))
    v = hv.reshape(-1).astype(np.int64)
    order = np.argsort(v, kind='stable')
    v, e = v[order], (h[order] << 2) | s[order]
    cnt = np.bincount(v, minlength=n)
    first = 2 * n + np.concatenate([[0], np.cumsum(cnt)[:-1]])
    return np.concatenate([np.stack([first, cnt], -1).reshape(-1), e]).astype(np.uint32)


def _prolong_slots(P, n_fine):
    """Fixed four (coarse, weight) slots per fine vertex."""
    rows, cols, wts = P
    pc = np.full((n_fine, 4), 0xFFFFFFFF, np.uint32)
    pw = np.zeros((n_fine, 4), np.float32)
    order = np.argsort(rows, kind='stable')
    rows, cols, wts = rows[order], cols[order], wts[order]
    start = np.searchsorted(rows, np.arange(n_fine + 1))
    slot = np.arange(len(rows)) - start[rows]
    assert slot.max(initial=0) < 4
    pc[rows, slot] = cols
    pw[rows, slot] = wts
    return pc, pw


def _restrict_csr(P, n_coarse):
    """Per coarse vertex: head (first, count as f32 bits), then (fine index bits, weight)."""
    rows, cols, wts = P
    order = np.argsort(cols, kind='stable')
    rows, cols, wts = rows[order], cols[order], wts[order]
    cnt = np.bincount(cols, minlength=n_coarse)
    first = n_coarse + np.concatenate([[0], np.cumsum(cnt)[:-1]])
    head = np.zeros((n_coarse, 4), np.float32)
    head[:, 0] = first.astype(np.uint32).view(np.float32)
    head[:, 1] = cnt.astype(np.uint32).view(np.float32)
    ent = np.zeros((len(rows), 4), np.float32)
    ent[:, 0] = rows.astype(np.uint32).view(np.float32)
    ent[:, 1] = wts
    return np.concatenate([head, ent])


class Runner:
    """Records the multigrid bending solve of one substep on the GPU (see the module notes)."""

    PRE = 3        # Chebyshev dispatches before and after the coarse correction
    POST = 3
    COARSEST = 10

    def __init__(self, gpu, hier: Hierarchy, hinge_v, mask):
        self.gpu = gpu
        self.hier = hier
        self.mask = int(mask)
        L = hier.levels
        n0 = L[0].n
        l0b = ['rbuf', 'buf', 'buf', 'buf', 'rbuf', 'rbuf', 'rbuf', 'rbuf']
        self.k_l0 = {e: gpu.kernel('cloth_mg.wgsl', l0b, entry=e, workgroup=(64, 1, 1))
                     for e in ('l0_start', 'l0_cheb', 'l0_residual')}
        self.k_restrict = gpu.kernel('cloth_mgc.wgsl', ['rbuf', 'rbuf', 'buf', 'buf', 'buf'], entry='gather_residual',
                                     workgroup=(64, 1, 1))
        cb = ['rbuf', 'buf', 'rbuf', 'buf', 'rbuf', 'rbuf', 'rbuf']
        self.k_c = {e: gpu.kernel('cloth_mgc_solve.wgsl', cb, entry=e, workgroup=(64, 1, 1)) for e in ('cheb', 'resid')}
        self.k_p0 = gpu.kernel('cloth_mg_prolong0.wgsl', ['buf', 'rbuf', 'rbuf', 'rbuf', 'rbuf', 'rbuf', 'rbuf'],
                               workgroup=(64, 1, 1))
        self.k_pc = gpu.kernel('cloth_mg_prolongc.wgsl', ['buf', 'rbuf', 'rbuf', 'rbuf', 'rbuf'], workgroup=(64, 1, 1))
        self.bufs = []
        self.VH = self._buf(_vertex_hinges(hinge_v, n0), 'mg-vh')
        self.lv = []
        for k, lv in enumerate(L):
            d = {'n': lv.n}
            if k == 0:
                for name in ('XT', 'DX', 'RES'):
                    d[name] = gpu.buffer(max(16, n0 * 16), f'mg-{name.lower()}0')
                    self.bufs.append(d[name])
            else:
                s, c, v = lv.A
                a = np.zeros((len(c), 4), np.float32)
                a[:, 0] = c.astype(np.uint32).view(np.float32)
                a[:, 1:3] = v
                dg = np.zeros((lv.n, 4), np.float32)
                rowid = np.repeat(np.arange(lv.n), np.diff(s))
                on = rowid == c
                np.add.at(dg[:, 0], rowid[on], v[on, 0])
                np.add.at(dg[:, 1], rowid[on], v[on, 1])
                dg[:, 2] = lv.pinned
                d['AS'] = self._buf(s.astype(np.uint32), f'mg-as{k}')
                d['A'] = self._buf(a, f'mg-a{k}')
                d['DG'] = self._buf(dg, f'mg-dg{k}')
                d['RT'] = self._buf(_restrict_csr(L[k - 1].P, lv.n), f'mg-rt{k}')
                for name in ('X', 'X2', 'B', 'DC', 'RES'):
                    d[name] = gpu.buffer(max(16, lv.n * 16), f'mg-{name.lower()}{k}')
                    self.bufs.append(d[name])
            if lv.P is not None:
                pc, pw = _prolong_slots(lv.P, lv.n)
                d['PC'] = self._buf(pc, f'mg-pc{k}')
                d['PW'] = self._buf(pw, f'mg-pw{k}')
            d['coef_pre'] = _cheb_coefs(lv, self.PRE)
            d['coef_post'] = _cheb_coefs(lv, self.POST)
            d['coef_last'] = _cheb_coefs(lv, self.COARSEST)
            self.lv.append(d)

    def _buf(self, arr, name):
        arr = np.ascontiguousarray(arr)
        b = self.gpu.buffer(max(16, arr.nbytes), name)
        self.gpu.write_buffer(b, arr)
        self.bufs.append(b)
        return b

    def destroy(self):
        for b in self.bufs:
            b.destroy()
        self.bufs = []

    def cycle(self, b, h, cb):
        """One V-cycle for this substep (length h); `cb` is the cloth's buffer dict (X and XB are swapped)."""
        from .gpu import Uniforms
        L = self.lv
        n0 = L[0]['n']

        def U(n, c1=0.0, c2=0.0):
            return Uniforms().v4(n, h, c1, c2).v4(self.mask)

        def l0(entry, xo, c1=0.0, c2=0.0):
            b.run(self.k_l0[entry], [cb['X'], xo, L[0]['XT'], L[0]['DX'], cb['S'], cb['V'], cb['H'], self.VH],
                  U(n0, c1, c2), (n0, 1, 1))

        def l0_smooth(coefs):
            for c1, c2 in coefs:
                l0('l0_cheb', cb['XB'], c1, c2)
                cb['X'], cb['XB'] = cb['XB'], cb['X']

        def c_smooth(k, coefs):
            d = L[k]
            for c1, c2 in coefs:
                b.run(self.k_c['cheb'], [d['X'], d['X2'], d['B'], d['DC'], d['AS'], d['A'], d['DG']], U(d['n'], c1, c2),
                      (d['n'], 1, 1))
                d['X'], d['X2'] = d['X2'], d['X']

        l0('l0_start', cb['XB'])
        l0_smooth(L[0]['coef_pre'])
        l0('l0_residual', L[0]['RES'])
        last = len(L) - 1
        for k in range(1, last + 1):
            d = L[k]
            b.run(self.k_restrict, [L[k - 1]['RES'], d['RT'], d['B'], d['X'], d['DC']], U(d['n']), (d['n'], 1, 1))
            if k == last:
                c_smooth(k, d['coef_last'])
                break
            c_smooth(k, d['coef_pre'])
            b.run(self.k_c['resid'], [d['X'], d['RES'], d['B'], d['DC'], d['AS'], d['A'], d['DG']], U(d['n']),
                  (d['n'], 1, 1))
        for k in range(last - 1, 0, -1):
            d = L[k]
            b.run(self.k_pc, [d['X'], L[k + 1]['X'], d['PC'], d['PW'], d['DG']], U(d['n']), (d['n'], 1, 1))
            c_smooth(k, d['coef_post'])
        b.run(self.k_p0, [cb['X'], L[1]['X'], L[0]['PC'], L[0]['PW'], cb['S'], cb['V'], cb['HOLED']], U(n0), (n0, 1, 1))
        l0_smooth(L[0]['coef_post'])
