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


def test_caustics_aim_at_the_clear_things_and_mirrors_first_then_the_biggest():
    # nine glazed pots and painted floats (coats), each bigger than the last, listed before a glass ball, a mirror ball
    # and an ice block: the glass and the mirror are traced at all the same, and of the coats the biggest; still at most
    # TARGETS of them
    from types import SimpleNamespace
    from blackbody.engine.stage import looks
    sc = Scene()
    sc.colliders.clear()
    for k in range(9):
        sc.add_collider(shape='sphere', position=(k * 0.5, 0.2, 0.0), size=(0.1 + 0.01 * k,) * 3,
                        material='ceramic' if k % 2 else 'painted')
    glass = sc.add_collider(shape='sphere', position=(0.0, 0.2, 1.0), size=(0.08,) * 3, material='glass')
    mirror = sc.add_collider(shape='sphere', position=(1.0, 0.2, 1.0), size=(0.3,) * 3, material='aluminium')
    sc.add_collider(shape='box', position=(2.0, 0.2, 1.0), size=(0.2,) * 3, material='ice')   # (flat: its shadow is exact)
    cols = [SimpleNamespace(shape=c['shape'], size=c['size'], pos=c['position']) for c in sc.colliders]
    rows = looks(sc, False)
    assert [LU.focusing(r, c) for r, c in zip(rows, cols)][:9] == ['coat'] * 9
    targets, mask = LU.caustic_targets(sc, cols, rows, None)
    assert len(targets) == LU.TARGETS
    assert targets[0][:3] == pytest.approx((0.0, 0.2, 1.0)) and targets[1][:3] == pytest.approx((1.0, 0.2, 1.0))
    picked = [k for k in range(16) if mask >> k & 1]
    assert glass in picked and mirror in picked and 11 not in picked
    assert sorted(picked) == sorted([glass, mirror] + list(range(3, 9)))      # (the six biggest coats; not the two least)
    # a big clear thing is taken before a small one, and a clear one before a mirror however big
    assert LU.focused_light('glass', rows[glass], 0.2) > LU.focused_light('glass', rows[glass], 0.1)
    few, _ = LU.caustic_targets(sc, cols[9:11], rows[9:11], None)
    assert [t[:3] for t in few] == [pytest.approx((0.0, 0.2, 1.0)), pytest.approx((1.0, 0.2, 1.0))]


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


def _ready(vase):
    """The vase scene, simulated to its frame again if a test that failed before this one left the engine rebuilt
    (conftest.py)."""
    eng, sc, f = vase
    if eng.sim_frame != f:
        eng.prepare(sc, final=True)
        eng.simulate_to(sc, f, cache=False)
    return vase


def _render(eng, sc, f, final=True, aa=4, **lume):
    """Render the frame with these Lume settings; aa: the render's own samples (1: the viewer's live picture)."""
    sc.data['lume'].update(lume)
    eng.render(sc, f, (160, 90), mode='composite', final=final, samples=aa, motion_blur=False)
    return eng.gpu.read(eng.stage.tex).astype(np.float32)


def test_lume_lights_the_set_about_as_classic_does(vase):
    eng, sc, f = _ready(vase)
    classic = _render(eng, sc, f, engine='classic')
    lume = _render(eng, sc, f, engine='lume', samples=64, bounces=4, denoise=True)
    assert np.isfinite(lume).all() and (lume[..., :3] >= 0).all()
    a, b = classic[..., :3].mean(), lume[..., :3].mean()
    assert 0.7 < b / a < 1.3                                      # the same light, traced (bounce light adds a little)
    assert np.allclose(lume[..., 3], classic[..., 3], atol=1e-3)  # and the same CG share of each pixel


def test_more_samples_converge_and_the_denoiser_keeps_the_brightness(vase):
    eng, sc, f = _ready(vase)
    ref = _render(eng, sc, f, engine='lume', samples=512, bounces=4, denoise=False)[..., :3]
    lo = _render(eng, sc, f, engine='lume', samples=8, bounces=4, denoise=False)[..., :3]
    dn = _render(eng, sc, f, engine='lume', samples=8, bounces=4, denoise=True)[..., :3]
    err = lambda a: float(np.sqrt(((a - ref) ** 2).mean()))
    assert err(dn) < 0.75 * err(lo)                               # the denoiser takes away much of the grain
    assert dn.mean() == pytest.approx(ref.mean(), rel=0.05)      # without darkening or brightening the picture


def test_the_viewer_refines_progressively(vase):
    eng, sc, f = _ready(vase)
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
    # a glass ball on a floor under a lamp off to one side (no sky): in its shadow, where the line from the lamp through
    # the ball meets the floor, the light it focuses is brighter than the open floor (traced from the lamp: lume.wgsl
    # caustics); without the caustics, the ball's shadow lets the lamp's light straight through it, unfocused. (Seen
    # from the front and low, as the benchmark sees it: from above, the camera sees that spot only through the ball,
    # where its focus is not traced.)
    from blackbody.engine import camera as cam
    from blackbody.engine import lume as LU
    spec = dict(size=(160, 120), camera=dict(eye=(0.0, 0.8, 2.6), target=(0.0, 0.25, 0.0), hfov=40.0), sky=0.0,
                floor=dict(alb=(0.6, 0.6, 0.6), rough=0.7), bounces=4,
                objects=[dict(shape='sphere', pos=(-0.25, 0.25, 0.0), size=(0.25,) * 3, alb=(1.0, 1.0, 1.0), rough=0.03,
                              clear=1.0, ior=1.5)],
                lamps=[dict(pos=(-1.5, 2.5, 0.5), radius=0.1, power=(8.0, 8.0, 8.0))])
    sc = _bench_scene(spec, monkeypatch)
    spec_c, fire = sc.camera(sc.start)
    cs = cam.compute(spec_c, 160 / 120, fire)

    def px(p):
        q, ok = cam.project(cs, np.array([p], float), 160, 120)
        return int(q[0][0]), int(q[0][1])

    # where the line from the lamp through the ball's middle meets the floor
    lamp, ball = np.array([-1.5, 2.5, 0.5]), np.array([-0.25, 0.25, 0.0])
    d = (ball - lamp) / np.linalg.norm(ball - lamp)
    spot = ball + d * (ball[1] / -d[1])
    x, y = px(spot)
    ox, oy = px((0.6, 0.0, -0.3))   # (open floor, lit)
    assert 1 <= x < 159 and 1 <= y < 119 and 1 <= ox < 159 and 1 <= oy < 119, (x, y, ox, oy)
    on = _stage(engine, sc, spec, samples=256, bounces=4).mean(-1)
    assert engine.stage.lume is not None
    monkeypatch.setattr(LU, 'caustic_targets', lambda *a, **k: ([], 0))
    off = _stage(engine, sc, spec, samples=256, bounces=4).mean(-1)
    # (the brightest place near there with the caustics, and the same place without them)
    w = on[y - 6:y + 7, x - 8:x + 9]
    cy, cx = np.unravel_index(int(np.argmax(w)), w.shape)
    cy, cx = y - 6 + int(cy), x - 8 + int(cx)
    at = lambda img, u, v: float(img[v - 1:v + 2, u - 1:u + 2].mean())
    open_floor = at(on, ox, oy)
    assert at(on, cx, cy) > 1.5 * open_floor
    assert at(on, cx, cy) > 3.0 * at(off, cx, cy)   # (without them: the light the glass lets straight through)


def test_dispersion_splits_the_caustic_into_colours_and_keeps_its_light(engine, monkeypatch):
    # a glass ball focusing a white lamp onto the floor (as above): with dispersion each wavelength focuses at its own
    # depth, so the caustic's rim turns from white to coloured (its colour spread grows), and no light is made or lost
    spec = dict(size=(160, 120), camera=dict(eye=(0.0, 0.8, 2.6), target=(0.0, 0.25, 0.0), hfov=40.0), sky=0.0,
                floor=dict(alb=(0.6, 0.6, 0.6), rough=0.7), bounces=4,
                objects=[dict(shape='sphere', pos=(-0.25, 0.25, 0.0), size=(0.25,) * 3, alb=(1.0, 1.0, 1.0), rough=0.03,
                              clear=1.0, ior=1.5)],
                lamps=[dict(pos=(-1.5, 2.5, 0.5), radius=0.1, power=(8.0, 8.0, 8.0))])
    sc = _bench_scene(spec, monkeypatch)
    white = _stage(engine, sc, spec, samples=512, bounces=4, dispersion=0.0)
    split = _stage(engine, sc, spec, samples=512, bounces=4, dispersion=8.0)
    assert np.isfinite(split).all() and (split >= 0).all()
    assert split.mean() == pytest.approx(white.mean(), rel=0.01)                       # the same light, spread
    lit = white.mean(-1) > 2.0 * np.median(white.mean(-1))                            # (the caustic and the highlights)
    chroma = lambda im: (im.max(-1) - im.min(-1)) / np.maximum(im.mean(-1), 1e-4)
    assert chroma(split)[lit].mean() > 1.5 * chroma(white)[lit].mean()


def test_a_mirror_ball_throws_the_lamp_onto_the_floor(engine, monkeypatch):
    # a mirror ball (bare smooth metal) beside a lamp: the light it reflects onto the floor is traced from the lamp
    # (lume.wgsl caustics) and agrees with the camera's own paths finding it by bouncing (the caustics off): the same
    # light, with far less noise
    from blackbody.engine import lume as LU
    spec = dict(size=(128, 96), camera=dict(eye=(0.0, 1.6, 2.4), target=(0.0, 0.0, 0.0), hfov=50.0), sky=0.0,
                floor=dict(alb=(0.6, 0.6, 0.6), rough=0.8), bounces=2,
                objects=[dict(shape='sphere', pos=(0.0, 0.3, 0.0), size=(0.3,) * 3, alb=(0.95, 0.95, 0.95), rough=0.05,
                              metal=1.0)],
                lamps=[dict(pos=(-0.9, 0.5, 0.0), radius=0.1, power=(2.0, 2.0, 2.0))])
    sc = _bench_scene(spec, monkeypatch)
    on = _stage(engine, sc, spec, samples=256, bounces=2).mean(-1)
    monkeypatch.setattr(LU, 'caustic_targets', lambda *a, **k: ([], 0))
    off = _stage(engine, sc, spec, samples=2048, bounces=2).mean(-1)
    # the floor to the right of the ball, away from the lamp, lit only by what the ball reflects
    band = (slice(60, 90), slice(80, 120))
    assert float(on[band].mean()) > 0.0
    assert abs(float(on[band].mean()) / float(off[band].mean()) - 1.0) < 0.1
    # (and with an eighth of the samples, less noise: the bounce finds a lamp in a mirror rarely, in fireflies)
    def noise(img):
        b = img[band]
        return float(np.abs(b[1:-1, 1:-1] - (b[:-2, 1:-1] + b[2:, 1:-1] + b[1:-1, :-2] + b[1:-1, 2:]) / 4).mean())
    assert noise(on) < noise(off)


def test_a_caustic_seen_through_its_glass_is_focused(engine, monkeypatch):
    # a glass ball just off a floor under a lamp straight above it, seen from above: through the ball the camera sees the
    # floor below it, where the ball focuses the lamp; that light comes from the caustic cache (lume.wgsl lu_cache_get),
    # so the ball shows the focused light, far brighter than the open floor. Without the cache, the light comes straight
    # through the glass, unfocused, about as bright as the open floor
    from blackbody.engine import lume as LU
    spec = dict(size=(160, 120), camera=dict(eye=(0.0, 2.2, 0.6), target=(0.0, 0.0, 0.0), hfov=40.0), sky=0.0,
                floor=dict(alb=(0.6, 0.6, 0.6), rough=0.8), bounces=8,
                objects=[dict(shape='sphere', pos=(0.0, 0.26, 0.0), size=(0.25,) * 3, alb=(1.0, 1.0, 1.0), rough=0.03,
                              clear=1.0, ior=1.5)],
                lamps=[dict(pos=(0.0, 2.5, 0.0), radius=0.15, power=(4.0, 4.0, 4.0))])
    sc = _bench_scene(spec, monkeypatch)
    on = _stage(engine, sc, spec, samples=256, bounces=8).mean(-1)
    monkeypatch.setattr(LU, 'cache_cell', lambda targets: 0.0)
    off = _stage(engine, sc, spec, samples=256, bounces=8).mean(-1)
    ball = (slice(60, 85), slice(60, 100))   # (the lower half of the ball, as the camera sees it)
    open_floor = float(on[100:115, 5:30].mean())
    bright_on = float(np.percentile(on[ball], 75))
    bright_off = float(np.percentile(off[ball], 75))
    assert bright_on > 3.0 * open_floor
    assert bright_on > 2.0 * bright_off


def test_coloured_glass_casts_a_coloured_shadow(engine, monkeypatch):
    # a red glass block on a floor under a lamp off to one side (no sky): the light its shadow rays let through takes the glass's tint
    # (lume.wgsl lu_shadow: its colour put back by lu_direct, g_wcol), not its grey; the shadow is red
    spec = dict(size=(96, 64), camera=dict(eye=(0.0, 1.6, 1.6), target=(0.0, 0.0, 0.0), hfov=40.0), sky=0.0,
                floor=dict(alb=(0.6, 0.6, 0.6), rough=0.9), bounces=1,
                objects=[dict(shape='box', pos=(0.0, 0.3, 0.0), size=(0.25, 0.05, 0.25), alb=(0.9, 0.25, 0.2), rough=0.03,
                              clear=1.0, ior=1.5)],
                lamps=[dict(pos=(-1.0, 2.0, 0.0), radius=0.05, power=(3.0, 3.0, 3.0))])
    sc = _bench_scene(spec, monkeypatch)
    img = _stage(engine, sc, spec, samples=256, bounces=1)
    from blackbody.engine import camera as cam
    spec_c, fire = sc.camera(sc.start)
    cs = cam.compute(spec_c, 96 / 64, fire)
    # (the floor in its shadow, off to the side away from the lamp, seen past the block, not through it)
    q, _ok = cam.project(cs, np.array([[0.176, 0.0, 0.0]]), 96, 64)
    x, y = int(q[0][0]), int(q[0][1])
    c = img[y - 2:y + 3, x - 2:x + 3].reshape(-1, 3).mean(0)
    assert c[0] > 2.0 * c[2] and c[0] > 2.0 * c[1], c
