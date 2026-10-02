"""Lume, the path-traced lighting engine (engine/lume.py, wgsl/lume.wgsl, wgsl/lume_denoise.wgsl)."""
import math

import numpy as np
import pytest

from blackbody.engine import lume as LU
from blackbody.scene import presets
from blackbody.scene.model import Scene


def test_settings_default_to_classic_and_old_projects_stay_classic():
    sc = Scene()
    s = LU.settings(sc)
    assert not s.on and s.bounces == 4 and s.samples >= 4 and s.denoise and s.clamp == 0.0
    sc.data.pop('lume', None)      # a project from before Lume
    assert not LU.settings(sc).on
    sc.data['lume'] = {'engine': 'lume'}
    assert LU.settings(sc).on


def _sky(w=256, h=128, sun_dir=(0.3, 0.55, -0.78), sun_deg=2.0, sun=5000.0):
    v = (np.arange(h) + 0.5) / h
    u = (np.arange(w) + 0.5) / w
    T, P = np.meshgrid(v * math.pi, (u - 0.5) * 2 * math.pi, indexing='ij')
    d = np.stack([np.sin(T) * np.sin(P), np.cos(T), -np.sin(T) * np.cos(P)], -1)
    img = np.where(d[..., 1:2] > 0, np.array([0.2, 0.35, 0.8], np.float32), np.array([0.1, 0.1, 0.1], np.float32))
    sd = np.asarray(sun_dir, float) / np.linalg.norm(sun_dir)
    img = img + (d @ sd > math.cos(math.radians(sun_deg)))[..., None] * sun
    return img.astype(np.float32), sd


def test_env_table_is_a_distribution_that_finds_the_sun():
    img, sd = _sky()
    table, w, h = LU.env_table(img)
    marg, cond, pdf = table[:h], table[h:h + w * h].reshape(h, w), table[h + w * h:].reshape(h, w)
    assert (w, h) == (256, 128) and len(table) == h + 2 * w * h
    assert np.all(np.diff(marg) >= 0) and marg[-1] == pytest.approx(1.0)
    assert np.all(np.diff(cond, axis=1) >= -1e-6) and np.allclose(cond[:, -1], 1.0)
    assert pdf.mean() == pytest.approx(1.0, rel=1e-4)            # a pdf over the picture: averages 1
    assert pdf.min() > 0.0                                        # every direction can be picked
    # the sun's pixels hold most of the light, so most picks go there
    lum = img @ np.array([0.2126, 0.7152, 0.0722])
    st = np.sin((np.arange(h) + 0.5) / h * math.pi)
    sun = lum > 1000
    assert (pdf * sun).sum() / pdf.sum() > 0.8
    # its pdf over directions integrates to 1 (pdf / (2 pi^2 sin theta) over each pixel's solid angle)
    dom = st[:, None] * (math.pi / h) * (2 * math.pi / w)
    assert (pdf / (2 * math.pi ** 2 * st[:, None]) * dom).sum() == pytest.approx(1.0, rel=1e-4)


def test_env_table_shrinks_a_big_hdri():
    img = np.ones((1024, 2048, 3), np.float32)
    table, w, h = LU.env_table(img)
    assert (w, h) == (LU.ENV_WIDTH, LU.ENV_WIDTH // 2)
    # an even sky: every direction as likely as any other (per steradian: 1 / 4 pi)
    st = np.sin((np.arange(h) + 0.5) / h * math.pi)
    per_sr = table[h + w * h:].reshape(h, w) / (2 * math.pi ** 2 * st[:, None])
    assert np.allclose(per_sr, 1 / (4 * math.pi), rtol=2e-3)


class _Plan(LU.Lume):
    def __init__(self):   # (no GPU: only the pass bookkeeping)
        self.key, self.passes, self.target, self.pending = None, 0, 0, False


def test_the_viewer_gathers_passes_until_its_samples_are_in():
    s = LU.LumeSettings(on=True, viewer_samples=6, samples=100)
    L = _Plan()
    assert L.plan(s, 'a', final=False, samples=1) == (0, 1)      # live: one pass, afresh
    L.passes = 1
    got = []
    for _ in range(5):
        first, n = L.plan(s, 'a', final=False, samples=4)
        got.append((first, n))
        L.passes = first + n
    assert got[:3] == [(0, 2), (2, 2), (4, 2)] and got[3] == (6, 0) and got[4] == (6, 0)
    assert L.plan(s, 'b', final=False, samples=4) == (0, 2)      # another picture: afresh
    assert L.plan(s, 'b', final=True, samples=4) == (0, 100)     # a final render: all of them now


@pytest.fixture(scope='module')
def vase(engine):
    sc = presets.make('vase_drop', fps=24)
    sc.data['render']['width'], sc.data['render']['height'] = 160, 90
    f = sc.start + 7
    engine.prepare(sc, final=True)
    engine.simulate_to(sc, f, cache=False)
    return engine, sc, f


def _render(eng, sc, f, final=True, aa=4, **lume):
    """Render the frame with these Lume settings; aa: the render's own samples (1: the viewer's live picture)."""
    sc.data['lume'].update(lume)
    eng.render(sc, f, (160, 90), mode='composite', final=final, samples=aa, motion_blur=False)
    return eng.gpu.read(eng.stage.tex).astype(np.float32)


def test_lume_lights_the_set_about_as_classic_does(vase):
    eng, sc, f = vase
    classic = _render(eng, sc, f, engine='classic')
    lume = _render(eng, sc, f, engine='lume', samples=64, bounces=4, denoise=True)
    assert np.isfinite(lume).all() and (lume[..., :3] >= 0).all()
    a, b = classic[..., :3].mean(), lume[..., :3].mean()
    assert 0.7 < b / a < 1.3                                      # the same light, traced (bounce light adds a little)
    assert np.allclose(lume[..., 3], classic[..., 3], atol=1e-3)  # and the same CG share of each pixel


def test_more_samples_converge_and_the_denoiser_keeps_the_brightness(vase):
    eng, sc, f = vase
    ref = _render(eng, sc, f, engine='lume', samples=512, bounces=4, denoise=False)[..., :3]
    lo = _render(eng, sc, f, engine='lume', samples=8, bounces=4, denoise=False)[..., :3]
    dn = _render(eng, sc, f, engine='lume', samples=8, bounces=4, denoise=True)[..., :3]
    err = lambda a: float(np.sqrt(((a - ref) ** 2).mean()))
    assert err(dn) < 0.75 * err(lo)                               # the denoiser takes away much of the grain
    assert dn.mean() == pytest.approx(ref.mean(), rel=0.05)      # without darkening or brightening the picture


def test_the_viewer_refines_progressively(vase):
    eng, sc, f = vase
    sc.data['lume'].update(engine='lume', viewer_samples=6, denoise=True)
    _render(eng, sc, f, final=False, aa=1)
    L = eng.stage.lume
    assert L.passes == 1 and not eng.stage.lume_pending
    seen = []
    for _ in range(4):
        _render(eng, sc, f, final=False, aa=4)
        seen.append((L.passes, eng.stage.lume_pending))
    assert seen == [(2, True), (4, True), (6, False), (6, False)]
    sc.data['lume']['engine'] = 'classic'
    _render(eng, sc, f, final=False, aa=4)
    assert not eng.stage.lume_pending
