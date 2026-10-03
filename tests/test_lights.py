"""Lights in the set in physical terms: light profiles (IES files, io/ies.py), lumens, area lights as real panels
(Lume traces the panel: lume.wgsl lu_rect, sampled over the solid angle it covers), old projects' area lights."""
import math

import numpy as np
import pytest

from blackbody.io import ies


def _ies_text(va, ha, cd, mult=1.0):
    """An IES LM-63-2002 file: vertical angles va, horizontal ha, candela cd[h][v]."""
    lines = ['IESNA:LM-63-2002', '[TEST] test', 'TILT=NONE',
             f'1 -1 {mult} {len(va)} {len(ha)} 1 2 0.1 0.1 0.0', '1.0 1.0 20',
             ' '.join(f'{a:g}' for a in va), ' '.join(f'{a:g}' for a in ha)]
    for row in cd:
        lines.append(' '.join(f'{c:g}' for c in row))
    return '\n'.join(lines) + '\n'


def _cosine_downlight(peak=1000.0):
    va = np.arange(0, 181, 5.0)
    cd = [[peak * max(math.cos(math.radians(a)), 0.0) for a in va]]
    return ies.parse(_ies_text(va, [0.0], cd), 'cosine')


def test_a_profile_reads_its_peak_and_lumens_and_points_down_its_aim():
    p = _cosine_downlight(1000.0)
    assert p.peak == pytest.approx(1000.0)
    assert p.lumens == pytest.approx(math.pi * 1000.0, rel=2e-3)        # a Lambertian downlight: pi times its peak
    assert p.table.shape == (ies.PROF_V, ies.PROF_H)
    assert p.table[0].min() == pytest.approx(1.0) and p.table[-1].max() == 0.0   # brightest along its aim, dark behind
    aim = np.array([0.0, -1.0, 0.0])
    axes = ies.frame(aim)
    w = np.array([math.sin(math.radians(60)), -math.cos(math.radians(60)), 0.0])
    assert ies.value(p.table, aim, axes, w) == pytest.approx(0.5, abs=0.01)


def test_a_profile_symmetric_in_quarters_and_its_multiplier():
    va = [0.0, 45.0, 90.0]
    ha = [0.0, 90.0]                                       # quarters alike: along its length bright, across dim
    cd = [[200.0, 200.0, 0.0], [100.0, 100.0, 0.0]]
    p = ies.parse(_ies_text(va, ha, cd, mult=2.0))
    assert p.peak == pytest.approx(400.0)                  # (times the file's candela multiplier)
    j90 = ies.PROF_H // 4
    row = p.table[10]                                      # (about 29 degrees off the aim)
    assert row[0] == pytest.approx(1.0) and row[j90] == pytest.approx(0.5)
    assert row[2 * j90] == pytest.approx(1.0) and row[3 * j90] == pytest.approx(0.5)   # mirrored into every quarter
    with pytest.raises(ValueError):
        ies.parse('no tilt here')


def test_lumens_set_the_brightness_for_each_kind_of_light():
    from blackbody.scene.model import Scene
    sc = Scene()
    sc.lights.clear()
    sc.add_light(kind='point', lumens=800.0, colour=(1.0, 1.0, 1.0), temperature=0.0)
    sc.add_light(kind='area', lumens=800.0, colour=(1.0, 1.0, 1.0), temperature=0.0, width=0.6, height=0.6)
    sc.add_light(kind='spot', lumens=800.0, colour=(1.0, 1.0, 1.0), temperature=0.0, cone=30.0, softness=0.0)
    L = sc.lamps(sc.start)
    lum = lambda l: float(np.asarray(l['power']) @ np.array([0.2126, 0.7152, 0.0722]))
    assert lum(L[0]) == pytest.approx(800.0 / (4.0 * math.pi), rel=1e-3)   # candela, every way
    assert lum(L[1]) == pytest.approx(800.0 / math.pi, rel=1e-3)           # a panel: straight in front
    cone = 2.0 * math.pi * (1.0 - math.cos(math.radians(30.0)))
    assert lum(L[2]) == pytest.approx(800.0 / cone, rel=0.01)               # a hard-edged spot: over its cone
    assert L[1]['width'] == 0.6 and L[1]['height'] == 0.6


def test_a_profile_file_sets_the_light_and_its_shape(tmp_path):
    from blackbody.scene.model import Scene
    va = np.arange(0, 181, 5.0)
    f = tmp_path / 'down.ies'
    f.write_text(_ies_text(va, [0.0], [[1500.0 * max(math.cos(math.radians(a)), 0.0) ** 2 for a in va]]))
    sc = Scene()
    sc.lights.clear()
    sc.add_light(kind='spot', profile=str(f), colour=(1.0, 1.0, 1.0), temperature=0.0)
    (L,) = sc.lamps(sc.start)
    assert L['kind'] == 'point' and L['profile'] is not None               # the profile is its shape (no cone)
    assert float(np.asarray(L['power']).max()) == pytest.approx(1500.0, rel=1e-3)   # its candela, from the file
    sc.lights[0]['profile_brightness'] = False
    sc.lights[0]['intensity'] = 300.0
    assert float(np.asarray(sc.lamps(sc.start)[0]['power']).max()) == pytest.approx(300.0)
    sc.lights[0]['profile'] = str(tmp_path / 'missing.ies')
    assert sc.lamps(sc.start)[0]['profile'] is None                         # a file that will not read: a plain light


def test_an_old_projects_area_light_becomes_a_panel_as_big():
    from blackbody.scene.model import Scene
    sc = Scene()
    sc.lights.clear()
    sc.add_light(kind='area', radius=0.5)
    d = sc.to_dict()
    for l in d['lights']:
        for k in ('width', 'height'):
            l.pop(k, None)
    s2 = Scene.from_dict(d)
    assert s2.lights[0]['width'] == pytest.approx(0.5 * math.sqrt(math.pi), abs=1e-3)


def test_lamp_rows_pack_the_panel_and_the_profile():
    from blackbody.engine.renderer import LAMP_ROWS, MAX_LAMPS, LAMP_PROF_V, lamp_rows
    p = _cosine_downlight()
    lamps = [dict(kind='area', position=(0, 1, 0), direction=(0, -1, 0), power=(1, 1, 1), radius=0.1, cos_outer=0.5,
                  cos_inner=0.6, width=1.2, height=0.4, spin=0.0, profile=None),
             dict(kind='point', position=(0, 1, 0), direction=(0, -1, 0), power=(1, 1, 1), radius=0.1, cos_outer=-1,
                  cos_inner=-1, profile=p)]
    rows = lamp_rows(lamps)
    assert rows.shape == (LAMP_ROWS, 16)
    assert rows[0, 11] == pytest.approx(0.6) and rows[0, 12] == pytest.approx(0.2)        # half width, half height
    assert rows[0, 3] == pytest.approx(math.sqrt(1.2 * 0.4 / math.pi), rel=1e-5)        # the disc as big (classic)
    assert rows[MAX_LAMPS, 3] == pytest.approx(0.6) and rows[MAX_LAMPS, 7] == pytest.approx(0.2)
    assert rows[1, 15] == 1.0 and rows[0, 15] == 0.0                                      # which has a profile
    s = 2 * MAX_LAMPS + LAMP_PROF_V
    assert np.allclose(rows[s:s + LAMP_PROF_V], p.table)


# -- traced ------------------------------------------------------------------------------------------------------------------

def _bench(monkeypatch, spec):
    from test_lume import _bench_scene
    return _bench_scene(spec, monkeypatch)


def _floor(**kw):
    spec = dict(size=(96, 64), camera=dict(eye=(0.0, 0.3, 1.0), target=(0.0, 0.0, 0.0), hfov=10.0), sky=0.0,
                floor=dict(alb=(0.6, 0.6, 0.6), rough=0.9), bounces=1, objects=[], lamps=[])
    spec.update(kw)
    return spec


def _render(engine, sc, spec, samples=1024):
    from test_lume import _stage
    return _stage(engine, sc, spec, samples=samples, bounces=int(spec['bounces']))


def _brdf(v, l, alb=0.6, rough=0.9):
    """Lume's material (lume.wgsl lu_eval), as test_lume works it out: diffuse and highlight, over cos."""
    nv, nl = v[1], l[1]
    fv = 0.04 + (max(1.0 - rough, 0.04) - 0.04) * (1.0 - nv) ** 5
    h = (v + l) / np.linalg.norm(v + l)
    a2 = (rough * rough) ** 2
    D = a2 / (math.pi * (h[1] ** 2 * (a2 - 1.0) + 1.0) ** 2)
    lv, ll = math.sqrt(a2 + (1 - a2) * nv * nv), math.sqrt(a2 + (1 - a2) * nl * nl)
    G2 = 2 * nl * nv / (nl * lv + nv * ll)
    F = 0.04 + 0.96 * (1.0 - float(v @ h)) ** 5
    return alb / math.pi * (1.0 - fv) + F * D * G2 / (4.0 * nv * nl)


def test_a_panel_lights_the_floor_as_its_solid_angle_says(engine, monkeypatch):
    # a 1 m square panel 1 m over a matte floor, facing down, no sky: the floor's radiance under its middle is the panel's
    # radiance (intensity / area) times Lume's material integrated over the directions the panel covers (by hand, a
    # 200 x 200 grid over the panel). A point light as bright straight on would give 1.33 times as much.
    I, d, a = 2.0, 1.0, 1.0
    spec = _floor(lamps=[dict(pos=(0.0, d, 0.0), panel=(a, a), aim=(0.0, -1.0, 0.0), spin=0.0, power=(I, I, I))])
    sc = _bench(monkeypatch, spec)
    img = _render(engine, sc, spec)
    got = float(img[28:36, 44:52].mean())
    v = np.array([0.0, 0.3, 1.0]) / math.hypot(0.3, 1.0)
    g = (np.arange(200) + 0.5) / 200 * a - a / 2
    X, Z = np.meshgrid(g, g)
    P = np.stack([X, np.full_like(X, d), Z], -1).reshape(-1, 3)
    r2 = (P ** 2).sum(-1)
    L = P / np.sqrt(r2)[:, None]
    dA = (a / 200) ** 2
    want = sum(_brdf(v, l) * l[1] * (I / (a * a)) * (l[1] * dA / q) for l, q in zip(L, r2))   # (dw = cos' dA / r^2)
    assert got == pytest.approx(want, rel=0.015)


def test_a_panel_seen_from_behind_gives_no_light_and_blocks_the_light_behind_it(engine, monkeypatch):
    spec = _floor(sky=0.5, lamps=[dict(pos=(0.0, 0.6, 0.0), panel=(1.0, 1.0), aim=(0.0, 1.0, 0.0), spin=0.0,
                                      power=(5.0, 5.0, 5.0))])
    sc = _bench(monkeypatch, spec)
    under = float(_render(engine, sc, spec, samples=256)[28:36, 44:52].mean())
    spec2 = dict(spec, lamps=[])
    sc2 = _bench(monkeypatch, spec2)
    open_sky = float(_render(engine, sc2, spec2, samples=256)[28:36, 44:52].mean())
    assert under < 0.75 * open_sky                         # the panel shades the sky above it, and sends no light down


def test_a_profile_shapes_a_lamps_light_on_the_floor(engine, monkeypatch, tmp_path):
    # a point lamp 1 m up with a cos^2 profile: the floor 1 m to the side (45 degrees off its aim) gets the profile's
    # share there (0.5) of what the same lamp without a profile gives it
    va = np.arange(0, 181, 2.5)
    f = tmp_path / 'cos2.ies'
    f.write_text(_ies_text(va, [0.0], [[100.0 * max(math.cos(math.radians(x)), 0.0) ** 2 for x in va]]))
    out = []
    for prof in ('', str(f)):
        spec = _floor(camera=dict(eye=(1.0, 0.3, 1.0), target=(1.0, 0.0, 0.0), hfov=10.0),
                      lamps=[dict(pos=(0.0, 1.0, 0.0), radius=0.03, power=(2.0, 2.0, 2.0))])
        sc = _bench(monkeypatch, spec)
        sc.lights[0].update(profile=prof, profile_brightness=False, direction=(0.0, -1.0, 0.0))
        out.append(float(_render(engine, sc, spec, samples=256)[28:36, 44:52].mean()))
    assert out[1] / out[0] == pytest.approx(0.5, rel=0.03)


# -- per-light passes ----------------------------------------------------------------------------------------------------------

def test_the_light_passes_add_up_to_the_picture_and_each_holds_its_own_light(engine, monkeypatch):
    # a glossy ball and a matte block on a floor, lit by the key light, the sky and a lamp (Lume › Light passes): the passes
    # add up to the picture, and each light's pass is that light's alone (the lamp's goes when the lamp does)
    spec = dict(size=(96, 64), camera=dict(eye=(0.0, 1.0, 2.6), target=(0.0, 0.3, 0.0), hfov=40.0), sky=0.4,
                floor=dict(alb=(0.6, 0.6, 0.6), rough=0.8), bounces=3,
                objects=[dict(shape='sphere', pos=(0.35, 0.3, 0.0), size=(0.3, 0.3, 0.3), alb=(0.7, 0.3, 0.2), rough=0.25),
                         dict(shape='box', pos=(-0.4, 0.2, -0.1), size=(0.2, 0.2, 0.2), alb=(0.6, 0.6, 0.6), rough=0.9)],
                lamps=[dict(pos=(-0.8, 1.2, 0.8), radius=0.05, power=(1.5, 1.4, 1.2))])
    sc = _bench(monkeypatch, spec)
    sc.data['lighting'].update(sun_on=True, sun_intensity=1.5, sun_elevation=35.0, sun_azimuth=40.0)
    sc.data['lume']['light_passes'] = True
    img = _render(engine, sc, spec, samples=256)
    lp = engine.stage.light_passes()
    assert sorted(lp) == ['light_fire', 'light_key', 'light_lamps', 'light_sky']
    low = slice(28, 64)                                   # (the floor and the objects: no sky seen straight)
    tot = sum(lp.values())
    assert float(tot[low].mean()) == pytest.approx(float(img[low].mean()), rel=0.01)
    for k in ('light_key', 'light_sky', 'light_lamps'):
        assert float(lp[k][low].mean()) > 0.03 * float(img[low].mean()), k
    assert float(lp['light_fire'].max()) == 0.0
    key = float(lp['light_key'][low].mean())
    sc.lights.clear()
    img2 = _render(engine, sc, spec, samples=256)
    lp2 = engine.stage.light_passes()
    assert float(lp2['light_lamps'].max()) == 0.0
    assert float(lp2['light_key'][low].mean()) == pytest.approx(key, rel=0.02)


# -- the footage as the environment ----------------------------------------------------------------------------------------

def test_the_footage_becomes_the_environment_seen_through_the_camera():
    # a plate of sky over ground with a red patch, a camera looking along -z: ahead, the panorama is the plate where the
    # camera saw it (the patch where it is); past the frame, the footage's light at each height carried on round the set
    # (its sky above, behind the camera too; its ground below)
    from blackbody.engine import camera as cam
    from blackbody.engine import footage_env as FE
    H, W = 360, 640
    p = np.zeros((H, W, 3), np.float32)
    p[:180] = (0.4, 0.6, 0.9)
    p[180:] = (0.3, 0.2, 0.1)
    p[150:210, 200:260] = (1.0, 0.0, 0.0)
    view = cam.look_at((0, 1.5, 0), (0, 1.5, -10))
    P = cam.perspective(math.radians(40), W / H, 0.1, 100)
    img = FE.image(p, 'linear', P @ view, exposure=2.0, width=256)

    def at(d):
        d = np.asarray(d, float) / np.linalg.norm(d)
        th, ph = math.acos(d[1]), math.atan2(d[0], -d[2])
        return img[min(int(th / math.pi * 128), 127), min(int((ph / (2 * math.pi) + 0.5) * 256), 255)]

    assert np.allclose(at((0, 0.2, -1)), 2.0 * np.array([0.4, 0.6, 0.9]), atol=1e-3)
    assert np.allclose(at((0, -0.2, -1)), 2.0 * np.array([0.3, 0.2, 0.1]), atol=1e-3)
    assert np.allclose(at((-0.18, 0.0, -1)), (2.0, 0.0, 0.0), atol=1e-3)                   # where the camera saw it
    for d in ((0, 1, 0), (0, 0.4, 1), (1, 0.3, 0)):
        assert np.allclose(at(d), 2.0 * np.array([0.4, 0.6, 0.9]), atol=0.05), d             # its sky, round the set
    for d in ((0, -1, 0), (0, -0.4, 1)):
        assert np.allclose(at(d), 2.0 * np.array([0.3, 0.2, 0.1]), atol=0.05), d             # its ground
