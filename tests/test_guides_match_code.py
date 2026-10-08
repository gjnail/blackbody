"""The guides say what the code does: facts in docs/*.md checked against the code they describe, so a guide does not go
on describing a feature after it has changed (the camera tracker, the travel solve, matter that burns, what Lume traces,
the presets they name). GPU-free."""
import inspect
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'


def _doc(name):
    """A guide's text with its lines joined (a sentence may be wrapped anywhere)."""
    return ' '.join((DOCS / name).read_text(encoding='utf-8').split())


def _src(rel):
    return (ROOT / 'blackbody' / rel).read_text(encoding='utf-8')


def test_the_camera_tracker_is_described_as_it_works():
    """Track follows many spots and solves a 3-D camera once the ground is lined up (io/tracker.MultiTracker,
    scene/camsolve.py); only without a line-up does it pin the effect to one point in 2D."""
    from blackbody.io.tracker import MultiTracker
    n = inspect.signature(MultiTracker.__init__).parameters['points'].default
    for name in ('compositing.md', 'troubleshooting.md'):
        assert f'about {n} ' in _doc(name), name
    assert 'The tracker follows one point in 2D' not in _doc('troubleshooting.md')


def test_the_travel_solve_is_described_as_it_works():
    """A travelling camera is solved a frame at a time, its spots placed where their rays meet (camsolve.solve_travel):
    there is no bundle adjustment refining the whole shot together, and the guides say so."""
    bundle = 'bundle' in _src('scene/camsolve.py').lower()
    for name in ('compositing.md', 'troubleshooting.md'):
        t = _doc(name).lower()
        assert ('bundle adjustment' in t) == bundle, name
        assert ('whole shot together' in t) != bundle, name


def test_the_matter_guide_matches_its_materials():
    from blackbody.engine.matter import MATTERS
    from blackbody.scene.params import MATTER_MATERIALS
    t = _doc('matter.md')
    # every material Made of offers has its row in the table
    table = ' '.join(l for l in (DOCS / 'matter.md').read_text(encoding='utf-8').splitlines() if l.startswith('| '))
    missing = [label for _key, label in MATTER_MATERIALS if label not in table]
    assert not missing, missing
    # what burns is said to burn, and its limits do not say it does not
    burns = [m for m in MATTERS.values() if m.burns_at > 0.0]
    if burns:
        assert 'It does not burn' not in t
        section = t.split('## Things that burn')[1].split(' ## ')[0].lower()
        assert all(m.label.lower() in section for m in burns), [m.label for m in burns]
    # a snowball squashes and sticks: the table does not have it splat where the limits say it does not
    if 'does not splat' in t:
        assert not re.search(r'\bsplats\b', t)


def test_lume_fire_caustics_are_described_as_traced():
    """The fire's light is traced through glass and off mirrors (lume.wgsl lu_fire_cau) while its shadows are on
    (stage.py fire_cau): the guide neither lists it as untraced nor leaves out the condition."""
    t = _doc('lume.md')
    traced = 'fn lu_fire_cau' in _src('engine/wgsl/lume.wgsl')
    m = re.search(r'What it does not yet trace from the lights:([^.]*)\.', t)
    assert m is not None
    assert ('fire' in m.group(1)) != traced
    if traced and re.search(r'fire_cau\s*=.*fire_shadows', _src('engine/stage.py')):
        assert 'Shadows from the fire' in t.split('**Caustics.**')[1].split(' - **')[0]


def test_lume_says_where_it_traces_water_and_caustics():
    """Lume traces the water only where liquid_engine.py's lume_water lets it (no footage, no fabric or grass, and a
    liquid LiquidRenderer.lume_ok takes), and caustics only without footage (stage.py caustic_targets)."""
    t = _doc('lume.md')
    bullet = t.split('**Water traced as it is.**')[1].split(' - **')[0]
    later = t.split('Liquids Lume does not trace yet')[1]
    cond = re.search(r'lume_water = \((.*?)\)\s*if lume_water', _src('engine/liquid_engine.py'), re.S).group(1)
    assert ('not footage' in cond) == ('water over footage' in bullet) == ('water over footage' in later)
    if 'cloth or grass' in cond:
        assert 'fabric or grass' in bullet and 'fabric or grass' in later
    ok = _src('engine/liquid_render.py').split('def lume_ok')[1].split('def ')[0]
    for word, says in (('glow', 'lava'), ('ice', 'ice'), ('ocean', "sea's waves")):
        if word in ok:
            assert says in bullet and says in later, says
    footage_off = re.search(r'caustic_targets\([^\n]*\) if not footage else', _src('engine/stage.py')) is not None
    caustics = t.split('**Caustics.**')[1].split(' - **')[0]
    assert footage_off == ('Not yet over footage' in caustics)


def test_presets_the_guides_name_exist():
    """Every preset a guide names (Preset: *...*, Presets: *...*, the *...* preset) is one, by its name in the list."""
    from blackbody.scene.presets import PRESETS
    names = {p.get('name', k) for k, p in PRESETS.items()}
    named, bad = 0, []
    for f in sorted(DOCS.glob('*.md')):
        t = _doc(f.name)
        found = [n for m in re.finditer(r'Presets?: ([^.]*)', t) for n in re.findall(r'\*([^*]+)\*', m.group(1))]
        found += re.findall(r'\*([^*]+)\* preset\b', t)
        named += len(found)
        bad += [f'{f.name}: {n}' for n in found if n not in names]
    assert named > 20
    assert not bad, bad


def test_the_guides_give_the_gpu_lists_caps():
    """The matter guide gives its glow's most lights and the limits the most pieces of ice, as the code has them."""
    from blackbody.engine.liquid_thermal import MAX_BODIES
    from blackbody.engine.stage import MATTER_LIGHTS
    assert f'at most {MATTER_LIGHTS} patches of its surface' in _doc('matter.md')
    assert f'at most {MAX_BODIES:,} separate pieces' in _doc('troubleshooting.md')
    assert f'const MAX_BODIES: u32 = {MAX_BODIES}u;' in _src('engine/wgsl/liq_therm_common.wgsl')   # (its twin)
