"""Settings shown where the engines read them: the environment (an HDRI, its rotation and strength, the key light from
it, the footage as the environment) in fire, liquid and fire-and-liquid scenes; the footage's holdouts (and roto shapes)
in liquid scenes; and every setting a tooltip names shown wherever the tooltip is. The environment's light on the smoke
and the water looks, GPU-free (the engine's helpers on an engine made without a GPU)."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from blackbody.scene import presets
from blackbody.scene.params import (COLLIDER_PARAMS, EMITTER_PARAMS, SECTION_TITLES, SECTIONS, applies)

KINDS = ('fire', 'liquid', 'both', 'cloud')
ENV_KEYS = ('environment', 'env_rotation', 'env_strength', 'env_sun', 'env_from_footage')
HOLDOUT_KEYS = ('holdout_matte', 'matte_channel', 'matte_invert', 'holdout_depth', 'depth_kind', 'depth_scale')


def test_the_environment_shows_in_every_scene_that_draws_a_set():
    for key in ENV_KEYS:
        for kind in ('fire', 'liquid', 'both'):
            assert applies('lighting', key, kind), (key, kind)
        assert not applies('lighting', key, 'cloud'), key            # (the sky renderer draws no set)
    for key in ('sky', 'haze', 'ground_albedo', 'altitude'):         # the physical sky: the same
        assert applies('lighting', key, 'fire') and applies('lighting', key, 'liquid') and not applies('lighting', key, 'cloud')
    assert applies('lighting', 'sun_azimuth', 'cloud')               # (the sun lights the clouds)


def test_the_holdouts_from_the_footage_show_in_liquid_scenes():
    for key in HOLDOUT_KEYS:
        for kind in ('fire', 'liquid', 'both'):
            assert applies('composite', key, kind), (key, kind)
        assert not applies('composite', key, 'cloud'), key
    for key in ('haze', 'haze_freq', 'haze_speed'):                  # (a molten liquid's heat haze)
        assert applies('composite', key, 'liquid') and not applies('composite', key, 'cloud')
    assert not applies('composite', 'smoke_opacity', 'liquid')       # (still the fire's own)


# -- tooltips name only what is shown with them ----------------------------------------------------------------------

def _tables():
    out = {sec: params for sec, params in SECTIONS.items()}
    out['emitter'], out['collider'] = EMITTER_PARAMS, COLLIDER_PARAMS
    titles = {title: sec for sec, title in SECTION_TITLES.items()}
    titles.update(Emitter='emitter', Collider='collider')
    return out, titles


def _refs(tip):
    """The settings a tooltip names as 'Section › Setting' (or 'Section › Group'): [(text, section, keys)]."""
    tables, titles = _tables()
    out, i = [], tip.find(' › ')
    while i >= 0:
        title = max((t for t in titles if tip[:i].endswith(t)), key=len, default=None)
        assert title is not None, f'no section before › in: {tip}'
        sec, rest = titles[title], tip[i + 3:]
        params = tables[sec]
        groups = {}
        for p in params:
            groups.setdefault(p.group, []).append(p.key)
        names = {**groups, **{p.label: [p.key] for p in params}}
        label = max((n for n in names if n and rest.startswith(n)), key=len, default=None)
        assert label is not None, f'{title} › has no setting or group for: {rest[:40]!r}'
        out.append((label, sec, names[label]))
        i = tip.find(' › ', i + 3)
    return out


def _shown(sec, keys, kind):
    return any(applies(sec, k, kind) for k in keys)


def test_every_setting_a_tooltip_names_shows_where_the_tooltip_does():
    tables, _ = _tables()
    bad = []
    for sec, params in tables.items():
        for p in params:
            if not p.tip or ' › ' not in p.tip or (sec, p.key) == ('domain', 'kind'):   # (the kind describes the others)
                continue
            refs = _refs(p.tip)
            for kind in KINDS:
                if not applies(sec, p.key, kind):
                    continue
                for label, rsec, keys in refs:
                    # (or the tooltip names another setting of that name that is shown: one for each kind of scene)
                    if not _shown(rsec, keys, kind) and not any(l == label and _shown(s, k, kind) for l, s, k in refs):
                        bad.append(f'{kind}: {sec}.{p.key} names {SECTION_TITLES.get(rsec, rsec)} › {label}')
    assert not bad, '\n'.join(bad)


# -- the environment's light, GPU-free ---------------------------------------------------------------------------------

def _engine():
    from blackbody.engine.engine import Engine
    return object.__new__(Engine)      # (its environment helpers keep their state as they go: no GPU needed)


def _write_hdr(path, img):
    """A flat (uncompressed) Radiance file."""
    h, w = img.shape[:2]
    m = img.max(axis=-1)
    e = np.ceil(np.log2(np.maximum(m, 1e-30))).astype(np.int32)
    mant = img / np.ldexp(1.0, e)[..., None] * 256.0
    rgbe = np.concatenate([np.clip(mant, 0, 255), (e + 128)[..., None]], -1).astype(np.uint8)
    path.write_bytes(b'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y %d +X %d\n' % (h, w) + rgbe.tobytes())


def _sky_with_a_sun(tmp_path, at=(10, 48)):
    h, w = 32, 64
    img = np.full((h, w, 3), 0.25, np.float32)
    img[h // 2:] = 0.05                                       # (the ground darker)
    img[at[0] - 1:at[0] + 2, at[1] - 1:at[1] + 2] = (20.0, 16.0, 10.0)   # a warm sun, its middle brightest
    img[at] = (60.0, 50.0, 30.0)
    p = tmp_path / 'set.hdr'
    _write_hdr(p, img)
    return p


def _uv_of(d, rotation_deg):
    """Where the stage and the liquid's march look up world direction d in the HDRI (stage.wgsl, liq_march.wgsl hdri)."""
    c, s = math.cos(math.radians(rotation_deg)), math.sin(math.radians(rotation_deg))
    r = np.array([c * d[0] - s * d[2], d[1], s * d[0] + c * d[2]])
    return math.atan2(r[0], -r[2]) / (2.0 * math.pi) + 0.5, math.acos(max(-1.0, min(1.0, r[1]))) / math.pi


@pytest.mark.parametrize('rotation', [0.0, 40.0, -125.0])
def test_a_fire_scenes_key_light_comes_from_the_hdris_sun(tmp_path, rotation):
    from blackbody.engine import camera as cam
    from blackbody.io.hdri import load_hdri, sky_average
    p = _sky_with_a_sun(tmp_path)
    sc = presets.make('campfire')
    assert sc.kind == 'fire'
    f = sc.start
    sc.data['lighting'].update(environment=str(p), env_rotation=rotation, env_strength=2.0, env_sun=True, sun_on=False,
                               sky='colour')
    eng = _engine()
    look = sc.look(f)
    eng._env_look(sc, look, f)
    # the key light is aimed where the stage draws the sun: through the rotation, at the bright texel
    u, v = _uv_of(cam.sun_direction(look.sun_azimuth, look.sun_elevation), rotation)
    assert abs(u * 64 - 48.5) < 0.05 and abs(v * 32 - 10.5) < 0.05
    assert look.sun_color[0] == pytest.approx(1.0) and look.sun_color[0] > look.sun_color[1] > look.sun_color[2]
    assert look.sun_intensity == pytest.approx(3.0)                  # (the key light off: as the liquid engine has it)
    # its average lights the smoke, as it does the set
    assert np.allclose(look.ambient, np.asarray(sky_average(load_hdri(p))) * 2.0, rtol=1e-5)
    sc.data['lighting'].update(sun_on=True, sun_intensity=5.0)
    look = sc.look(f)
    eng._env_look(sc, look, f)
    assert look.sun_intensity == pytest.approx(5.0)                  # Key intensity sets its strength


def test_without_key_light_from_environment_the_key_light_stays_and_a_relative_path_is_found(tmp_path):
    p = _sky_with_a_sun(tmp_path)
    sc = presets.make('campfire')
    sc.path = str(tmp_path / 'shot.bbx')
    f = sc.start
    sc.data['lighting'].update(environment=p.name, env_sun=False, sun_on=True, sky='colour')
    eng = _engine()
    plain = sc.look(f)
    look = sc.look(f)
    eng._env_look(sc, look, f)
    assert (look.sun_azimuth, look.sun_elevation, look.sun_color) == (plain.sun_azimuth, plain.sun_elevation, plain.sun_color)
    assert look.ambient != plain.ambient                             # (the HDRI's light, found next to the project)
    what, key, img, rot, k = eng._environment(sc, f)
    assert what == 'file' and key[0] == str(p) and img.shape == (32, 64, 3)
    sc.data['lighting']['environment'] = ''
    assert eng._environment(sc, f) is None


def _footage(view):
    from blackbody.engine import camera as cam
    H, W = 90, 160
    plate = np.zeros((H, W, 3), np.float32)
    plate[:45] = (0.4, 0.6, 0.9)
    plate[45:] = (0.3, 0.2, 0.1)
    return plate, cam.perspective(math.radians(40), W / H, 0.1, 100) @ view


def _counting(monkeypatch):
    """Counts the footage's HDRIs built (footage_env.image), from a fresh start."""
    from blackbody.engine import footage_env as FE
    real, built = FE.image, []
    monkeypatch.setattr(FE, 'image', lambda *a, **k: built.append(1) or real(*a, **k))
    monkeypatch.setattr(FE, '_last', None)
    return built


def test_the_footage_is_the_environment_in_every_kind_of_scene_with_a_set(monkeypatch):
    from blackbody.engine import camera as cam
    plate, vp = _footage(cam.look_at((0, 1.5, 0), (0, 1.5, -10)))
    cs = SimpleNamespace(view_proj=vp)
    eng = _engine()
    built = _counting(monkeypatch)
    for name, kind in (('campfire', 'fire'), ('hose', 'liquid'), ('hose', 'both')):
        sc = presets.make(name)
        sc.data['domain']['kind'] = kind
        f = sc.start
        sc.data['lighting'].update(env_from_footage=True, environment='', env_strength=2.0, env_rotation=30.0,
                                   sky='physical', env_sun=False)
        what, key, img, rot, k = eng._environment(sc, f, plate=plate, cs=cs)
        assert what == 'footage' and rot == 0.0 and k == 1.0         # (its strength in it; in the world's frame already)
        if kind == 'fire':
            continue
        wl = sc.water_look(f)
        assert eng._footage_env(sc, wl, f, plate, cs, (1.0, 1.0))
        assert wl.sky_image[0] == key and wl.sky_image[1] is img     # the liquid's set takes the same one
        assert wl.env_strength == 1.0 and wl.env_rotation == 0.0 and not wl.env_sun
        # with no set drawn and the key light its own, no HDRI is built (nor the physical sky's taken in its place)
        n = len(built)
        sc.data['lighting']['env_strength'] = 3.0                    # (a new one, were it built)
        wl = sc.water_look(f)
        assert eng._footage_env(sc, wl, f, plate, cs, (1.0, 1.0), wanted=False)
        assert wl.sky_image is None and len(built) == n
        sc.data['lighting']['env_sun'] = True                        # the key light from it: built
        wl = sc.water_look(f)
        assert eng._footage_env(sc, wl, f, plate, cs, (1.0, 1.0), wanted=False)
        assert wl.sky_image is not None and wl.env_sun and len(built) == n + 1
        sc.data['lighting'].update(env_from_footage=False, env_sun=False)
        wl = sc.water_look(f)                                        # the physical sky, unrotated too
        assert wl.sky_image is not None and wl.sky_image[0][0] == 'physical-sky' and wl.env_rotation == 0.0
        assert not eng._footage_env(sc, wl, f, plate, cs, (1.0, 1.0)) and wl.sky_image[0][0] == 'physical-sky'
        sc.data['lighting'].update(env_from_footage=True, environment='set.hdr')   # an HDRI file wins
        wl = sc.water_look(f)
        assert not eng._footage_env(sc, wl, f, plate, cs, (1.0, 1.0)) and wl.env_rotation == 30.0


def test_a_fire_scene_builds_the_footages_hdri_for_the_smoke_only_for_its_key_light(monkeypatch):
    from blackbody.engine import camera as cam
    plate, vp = _footage(cam.look_at((0, 1.5, 0), (0, 1.5, -10)))
    cs = SimpleNamespace(view_proj=vp)
    eng = _engine()
    built = _counting(monkeypatch)
    sc = presets.make('campfire')
    f = sc.start
    sc.data['lighting'].update(env_from_footage=True, environment='', env_sun=False, sun_on=True)
    look = sc.look(f)
    plain = (look.ambient, look.sun_azimuth, look.sun_elevation, look.sun_color)
    eng._env_look(sc, look, f, plate=plate, cs=cs)
    assert not built and (look.ambient, look.sun_azimuth, look.sun_elevation, look.sun_color) == plain
    sc.data['lighting']['env_sun'] = True
    eng._env_look(sc, look, f, plate=plate, cs=cs)
    assert len(built) == 1 and look.ambient == plain[0]              # (the ambient left to Match ambient to footage)
    assert (look.sun_azimuth, look.sun_elevation) != plain[1:3] and look.sun_elevation > 0.0   # (its sky the brightest)


def test_the_footages_hdri_goes_with_its_own_key(monkeypatch):
    from blackbody.engine import camera as cam
    from blackbody.engine import footage_env as FE
    plate, vp = _footage(cam.look_at((0, 1.5, 0), (0, 1.5, -10)))
    other = plate * 0.25
    sc = presets.make('campfire')
    f = sc.start
    sc.data['lighting'].update(env_from_footage=True, environment='')
    kind = sc.data['composite'].get('plate_transform', 'srgb')
    real = FE.image

    def image(p, *a, **k):
        img = real(p, *a, **k)
        if p is plate:      # another render asks for its own meanwhile (the viewer's worker and a render job)
            FE.of_scene(sc, f + 1, other, vp)
        return img
    monkeypatch.setattr(FE, 'image', image)
    monkeypatch.setattr(FE, '_last', None)
    key, img = FE.of_scene(sc, f, plate, vp)
    assert key[1] == f and np.array_equal(img, real(plate, kind, vp))
    assert FE.of_scene(sc, f + 1, other, vp)[1] is FE._last[1]      # (the last one kept)


class _GPU:
    """Enough of a GPU for the environment caches: counts what is uploaded."""
    def __init__(self):
        self.uploads = 0

    def _res(self, *a, **k):
        return SimpleNamespace(destroy=lambda: None)
    texture2d = buffer = _res

    def upload(self, *a):
        self.uploads += 1
    write_buffer = upload


def test_the_renderers_take_an_environment_again_after_a_render_without_it(tmp_path):
    from blackbody.engine.liquid_render import LiquidRenderer, WaterLook
    from blackbody.engine.lume import Lume
    p = str(_sky_with_a_sun(tmp_path))
    LR = object.__new__(LiquidRenderer)
    LR.gpu, LR.env_tex, LR._env_key, LR.env_info = _GPU(), None, None, None
    on, off = WaterLook(environment=p), WaterLook()
    first = LR.environment(on)
    assert LR.environment(off) is None and LR.environment(on) == first   # (measured again, not left None)
    assert LR.gpu.uploads == 2
    L = object.__new__(Lume)
    L.gpu, L._env, L._env_key, L.env_dims, L.env_traced = _GPU(), None, None, (0, 0), False
    L.environment(p)
    dims = L.env_dims
    assert dims[0] > 0
    L.environment('')
    assert L.env_dims == (0, 0)
    L.environment(p)
    assert L.env_dims == dims                                        # Lume's light from it back too


def test_roto_shapes_hide_the_liquid(tmp_path):
    from blackbody.scene.roto import new_shape
    sc = presets.make('hose')
    assert sc.kind == 'liquid'
    sc.data['render']['width'], sc.data['render']['height'] = 64, 36
    eng = _engine()
    assert eng._footage_holdout(sc) is None                          # nothing to hold out
    shape = new_shape()
    shape['feather'] = 0.0
    shape['keys'] = {sc.start: [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]}
    sc.roto = [shape]
    fh = eng._footage_holdout(sc)
    assert fh is not None
    matte, depth = fh.read(sc.start)
    assert depth is None and matte.shape == (36, 64)
    assert matte[:, :24].min() > 0.9 and matte[:, 40:].max() < 0.1   # the left half hidden
    shape['enabled'] = False                                         # (a change to the shapes is taken up)
    assert eng._footage_holdout(sc) is None


# -- on the GPU: the set lit by them --------------------------------------------------------------------------------------

def _set_scene(kind):
    from blackbody.scene.model import Scene
    sc = Scene()
    sc.data['domain'].update(kind=kind, resolution=24)
    sc.data['render'].update(width=64, height=36, motion_blur=False)
    sc.data['composite'].update(backdrop='stage')
    sc.add_collider(name='Box', shape='box', position=(0.6, 0.3, 0.0), size=(0.2, 0.2, 0.2), material='plaster', look='cg')
    return sc


def _ready(engine, sc, f):
    engine.invalidate()
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, f, cache=False)


@pytest.mark.parametrize('kind', ['fire', 'liquid', 'both'])
def test_the_set_takes_each_environment_again_after_it_is_turned_off(engine, tmp_path, kind):
    """An HDRI file, the physical sky and the footage round the set, each on, off and on again at the same frame: the
    same picture as at first (nothing kept of it while it was off but its texture)."""
    p = _sky_with_a_sun(tmp_path)
    sc = _set_scene(kind)
    f = sc.start + 1
    _ready(engine, sc, f)
    plate = np.zeros((36, 64, 4), np.uint8)
    plate[:18], plate[18:] = (110, 150, 220, 255), (90, 70, 50, 255)
    loaded = (lambda: engine.stage._env_key) if kind == 'fire' else (lambda: engine.liquid_r._env_key)
    for name, on, off, pl, key in (('file', dict(environment=str(p)), dict(environment=''), None, str(p)),
                                   ('sky', dict(sky='physical'), dict(sky='colour'), None, 'physical-sky'),
                                   ('footage', dict(env_from_footage=True), dict(env_from_footage=False), plate,
                                    'footage-env')):
        sc.data['lighting'].update(environment='', sky='colour', env_from_footage=False, env_sun=True, env_rotation=40.0)
        shots = []
        for st in (on, off, on):
            sc.data['lighting'].update(st)
            engine.render(sc, f, (64, 36), plate=pl)
            shots.append(engine.gpu.read(engine.stage.tex).astype(np.float32))
        assert loaded()[0] == key, name                              # round the set
        assert np.isfinite(shots[2]).all() and np.array_equal(shots[2], shots[0]), name
        assert not np.array_equal(shots[1], shots[0]), name          # (and it was off between)


@pytest.mark.parametrize('kind', ['fire', 'both'])
def test_the_smoke_is_lit_by_the_hdri(engine, tmp_path, monkeypatch, kind):
    from blackbody.io.hdri import brightest, load_hdri, sky_average
    p = _sky_with_a_sun(tmp_path)
    sc = _set_scene(kind)
    sc.data['lighting'].update(environment=str(p), env_sun=True, env_rotation=40.0, env_strength=2.0, sky='colour',
                               sun_on=False)
    f = sc.start + 1
    _ready(engine, sc, f)
    seen = []
    light = engine.renderer.light
    monkeypatch.setattr(engine.renderer, 'light', lambda b, vol, look, *a, **k: seen.append(look) or light(b, vol, look, *a, **k))
    engine.render(sc, f, (64, 36))
    look = seen[-1]                                                  # the smoke's look, as it is lit
    img = load_hdri(p)
    az, el, colour = brightest(img)
    assert np.allclose(look.ambient, np.asarray(sky_average(img)) * 2.0, rtol=1e-5)          # its average
    assert look.sun_azimuth == pytest.approx(az + 40.0) and look.sun_elevation == pytest.approx(el)   # its sun
    assert np.allclose(look.sun_color, colour) and look.sun_intensity == pytest.approx(3.0)
    sc.data['lighting']['env_sun'] = False                           # the key light its own again
    engine.render(sc, f, (64, 36))
    assert seen[-1].sun_intensity == 0.0 and np.allclose(seen[-1].ambient, look.ambient)


def test_a_roto_shape_hides_the_liquid(engine):
    from blackbody.scene.roto import new_shape
    sc = presets.make('rock_splash')                                 # (a pool across the shot)
    sc.data['domain'].update(resolution=32, preroll=0.0)
    sc.data['render']['width'], sc.data['render']['height'] = 64, 36
    f = sc.start + 2
    _ready(engine, sc, f)
    plate = np.full((36, 64, 4), 110, np.uint8)
    shots = {}
    for roto in (False, True):
        shape = new_shape()
        shape['feather'] = 0.0
        shape['keys'] = {sc.start: [[-0.1, -0.1], [0.5, -0.1], [0.5, 1.1], [-0.1, 1.1]]}   # (the left half)
        sc.roto = [shape] if roto else []
        engine.render(sc, f, (64, 36), plate=plate)
        shots[roto] = engine.display_image()[..., :3].astype(np.float32)
    seen = np.abs(shots[False] - plate[..., :3]).max(axis=-1) > 8.0  # the liquid, where it is in the shot
    assert seen[:, :28].sum() > 20 and seen[:, 36:].sum() > 20
    hidden = np.abs(shots[True] - plate[..., :3]).max(axis=-1) > 8.0
    assert not hidden[:, :28].any(), 'the roto shape hides it'
    assert np.array_equal(hidden[:, 36:], seen[:, 36:]), 'and only there'
