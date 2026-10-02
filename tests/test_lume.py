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


def _bench_scene(spec, monkeypatch):
    """A scene of the benchmark (tools/lume_bench) with its exact materials, for this test only: the benchmark patches
    the stage's floors, looks and horizon for its whole process, and monkeypatch puts them back afterwards (left
    patched, the next tests' objects all took the benchmark's looks)."""
    import tempfile
    from pathlib import Path
    from blackbody.engine import stage
    from blackbody.scene import materials
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'tools' / 'lume_bench'))
    import lume_side
    monkeypatch.setattr(stage, 'HORIZON_FADE', stage.HORIZON_FADE)
    monkeypatch.setattr(stage, 'looks', stage.looks)
    monkeypatch.setitem(materials.FLOORS, 'bench', None)
    monkeypatch.setitem(stage.FLOORS, 'bench', None)
    sky = Path(tempfile.mkdtemp()) / 'uniform.hdr'
    lume_side.uniform_hdr(sky)
    sc, rows = lume_side.build(spec, sky)
    lume_side.patch(spec, rows)
    return sc


def _floor_only(**kw):
    spec = dict(size=(96, 64), camera=dict(eye=(0.0, 1.2, 2.0), target=(0.0, 0.0, 0.0), hfov=40.0), sky=0.5,
                floor=dict(alb=(0.6, 0.6, 0.6), rough=0.9), bounces=1, objects=[], lamps=[])
    spec.update(kw)
    return spec


def _stage(engine, sc, spec, **lume):
    f = sc.start
    engine.prepare(sc, final=True)
    engine.simulate_to(sc, f, cache=False)
    sc.data['lume'].update(engine='lume', denoise=False, clamp=0.0, **lume)
    engine.render(sc, f, spec['size'], mode='composite', final=True, samples=4, motion_blur=False)
    return engine.gpu.read(engine.stage.tex).astype(np.float32)[..., :3]


def test_the_last_bounce_still_finds_the_sky(engine, monkeypatch):
    # an open floor under a uniform sky: nothing to bounce off, so one bounce lights it as fully as four (the sky half
    # left to the bounce by the HDRI's picked light must still be traced at the last bounce)
    spec = _floor_only()
    sc = _bench_scene(spec, monkeypatch)
    one = _stage(engine, sc, spec, samples=256, bounces=1)[40:, :].mean()
    four = _stage(engine, sc, spec, samples=256, bounces=4)[40:, :].mean()
    assert one == pytest.approx(four, rel=0.01)


def test_a_lamp_lights_the_floor_below_it_as_a_sphere_does(engine, monkeypatch):
    # a lamp 1 m over a matte floor, no sky: straight below it the floor's radiance is its albedo / pi times the lamp's
    # intensity / d^2 (a sphere's irradiance), times what the floor's highlight leaves the diffuse (1 - its rough Fresnel)
    I, d, alb, rough = 2.0, 1.0, 0.6, 0.9
    spec = _floor_only(sky=0.0, camera=dict(eye=(0.0, 0.6, 0.6), target=(0.0, 0.0, 0.0), hfov=10.0),
                       floor=dict(alb=(alb,) * 3, rough=rough),
                       lamps=[dict(pos=(0.0, d, 0.0), radius=0.05, power=(I, I, I))])
    sc = _bench_scene(spec, monkeypatch)
    img = _stage(engine, sc, spec, samples=1024, bounces=1)
    got = float(img[28:36, 44:52].mean())
    # seen from the camera (45 degrees down), lit from straight above: Lume's material (lume.wgsl lu_eval) by hand
    v = np.array([0.0, 0.6, 0.6]) / math.hypot(0.6, 0.6)
    l = np.array([0.0, 1.0, 0.0])
    nv, nl = v[1], 1.0
    fv = 0.04 + (max(1.0 - rough, 0.04) - 0.04) * (1.0 - nv) ** 5
    diffuse = alb / math.pi * (1.0 - fv)
    h = (v + l) / np.linalg.norm(v + l)
    a2 = (rough * rough) ** 2
    D = a2 / (math.pi * (h[1] ** 2 * (a2 - 1.0) + 1.0) ** 2)
    lv, ll = math.sqrt(a2 + (1 - a2) * nv * nv), math.sqrt(a2 + (1 - a2) * nl * nl)
    G2 = 2 * nl * nv / (nl * lv + nv * ll)
    F = 0.04 + 0.96 * (1.0 - float(v @ h)) ** 5
    highlight = F * D * G2 / (4.0 * nv * nl)
    assert got == pytest.approx((diffuse + highlight) * I / d ** 2, rel=0.015)


def test_a_glass_ball_focuses_the_lamp_into_a_caustic(engine, monkeypatch):
    # a glass ball on a floor under a lamp (no sky): straight below it, opposite the lamp, the light it focuses is brighter
    # than the open floor beside it (traced from the lamp: lume.wgsl caustics); the shadow round it is dark
    spec = dict(size=(160, 120), camera=dict(eye=(0.0, 2.2, 0.6), target=(0.0, 0.0, 0.0), hfov=40.0), sky=0.0,
                floor=dict(alb=(0.6, 0.6, 0.6), rough=0.8), bounces=4,
                objects=[dict(shape='sphere', pos=(0.0, 0.25, 0.0), size=(0.25,) * 3, alb=(1.0, 1.0, 1.0), rough=0.03,
                              clear=1.0, ior=1.5)],
                lamps=[dict(pos=(0.0, 2.5, 0.0), radius=0.05, power=(4.0, 4.0, 4.0))])
    sc = _bench_scene(spec, monkeypatch)
    img = _stage(engine, sc, spec, samples=256, bounces=4).mean(-1)
    from blackbody.engine import camera as cam
    spec_c, fire = sc.camera(sc.start)
    cs = cam.compute(spec_c, 160 / 120, fire)
    def at(p):
        px, ok = cam.project(cs, np.array([p], float), 160, 120)
        x, y = int(px[0][0]), int(px[0][1])
        return float(img[y - 1:y + 2, x - 1:x + 2].mean())
    under = at((0.0, 0.0, 0.0))        # (seen past the ball? the camera looks down at it from above and in front)
    open_floor = at((0.8, 0.0, 0.0))
    assert engine.stage.lume is not None
    assert max(under, at((0.0, 0.0, 0.06)), at((0.0, 0.0, -0.06))) > 1.5 * open_floor
