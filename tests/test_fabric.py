"""Fabric (engine/cloth.py): how it hangs, bends, drapes, moves in the air, burns, renders, caches and
exports."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from blackbody.engine import cloth as C
from blackbody.scene import presets
from blackbody.scene.params import collider_defaults

LOOK = SimpleNamespace(ambient_k=293.0, flame_k=1700.0, max_k=2000.0)


@pytest.fixture(scope='module')
def gpu(engine):
    return engine.gpu


def _cloth(gpu, spec, place=C.FabricPlace(pos=(0.0, 1.5, 0.0))):
    cl = C.Cloth(gpu)
    assert cl.configure([spec])
    cl.place([place])
    return cl


def _run(cl, gpu, seconds, place=C.FabricPlace(pos=(0.0, 1.5, 0.0)), colliders=(), meshes=None):
    chunk = 20
    with gpu.batch() as b:
        for _ in range(max(1, int(seconds * C.STEPS_PER_SECOND / chunk))):
            cl.step(b, None, chunk / C.STEPS_PER_SECOND, [place], None, LOOK, list(colliders), meshes, steps=chunk)
    return cl.positions()


def _plain(sc, res=40):
    sc.data['domain'].update(resolution=res, preroll=0.0, time_scale=1.0)
    if 'puffing' in sc.data['motion']:
        sc.data['motion']['puffing'] = 0.0
    sc.data['embers']['enabled'] = False
    return sc


def peirce_angle(l_over_c):
    """Peirce's cantilever: the angle (degrees) a strip of free length l droops, with c its bending length."""
    target = (1.0 / l_over_c) ** 3
    lo, hi = 1e-4, math.radians(89.9)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if math.cos(mid / 2) / (8 * math.tan(mid)) > target:
            lo = mid
        else:
            hi = mid
    return math.degrees(0.5 * (lo + hi))


def test_fabric_materials_are_real_fabrics():
    for name, m in C.MATERIALS.items():
        c = (m.bend / (m.density * 9.81)) ** (1 / 3)
        assert 0.005 < c < 0.06, (name, c)   # bending lengths of real woven cloth: chiffon a few mm .. canvas several cm
        assert 0.02 < m.density < 0.6 and m.stretch > 100 * m.shear * 0.1
    assert C.MATERIALS['wool'].burn_temp < 0.85 * C.MATERIALS['wool'].ignition, 'wool cannot keep itself burning'
    assert C.MATERIALS['cotton'].burn_temp > C.MATERIALS['cotton'].ignition
    assert C.MATERIALS['polyester'].melt > 0 and C.MATERIALS['polyester'].melt < C.MATERIALS['polyester'].ignition


def test_a_hanging_curtain_barely_stretches(gpu):
    spec = C.FabricSpec(width=1.0, height=1.5, detail=40, pins='top', material='cotton', burnable=False, self_collide=False)
    cl = _cloth(gpu, spec)
    x, gone = _run(cl, gpu, 2.0)
    assert np.isfinite(x).all() and not gone.any()
    top = 1.5 + 0.75
    drop = top - x[:, 1].min()
    assert 1.5 <= drop < 1.5 * 1.012, f'hangs its own length (stretched {drop / 1.5 - 1:.4f})'
    assert np.abs(x[:, 2]).max() < 0.01, 'still air: it hangs flat'


@pytest.mark.parametrize('l_over_c, det', [(2.0, 16), (3.0, 16), (2.0, 40)])
def test_bending_matches_peirces_cantilever(gpu, l_over_c, det):
    """A strip clamped flat over a ledge droops by the angle Peirce's cantilever test gives for its bending
    rigidity and weight (the standard measurement of a fabric's bending stiffness), however finely it is
    meshed (a stiff fabric keeps its stiffness at high detail)."""
    m = C.MATERIALS['cotton']
    mult = 2300.0                   # a stiff strip, so that the overhang spans many cells
    c = (m.bend * mult / (m.density * 9.81)) ** (1 / 3)
    L = l_over_c * c
    spec = C.FabricSpec(width=L * (1 + 2.0 / det), height=0.3 * L, detail=det, orientation='lying', pins='side', pin_rows=2,
                        material='cotton', bend=mult, burnable=False, self_collide=False)
    place = C.FabricPlace(pos=(0.0, 1.0, 0.0))
    cl = _cloth(gpu, spec, place)
    x, _ = _run(cl, gpu, 3.0, place)
    r = cl.built.rest
    cols = np.unique(np.round(r[:, 0], 9))
    clamp = x[np.isclose(r[:, 0], cols[1])].mean(0)
    tip = x[np.isclose(r[:, 0], cols[-1])].mean(0)
    theta = math.degrees(math.atan2(clamp[1] - tip[1], tip[0] - clamp[0]))
    assert theta == pytest.approx(peirce_angle((cols[-1] - cols[1]) / c), abs=6.0)


def test_silk_drapes_over_a_ball_without_passing_through(gpu, engine):
    from blackbody.engine.solver import ColliderGPU
    spec = C.FabricSpec(width=1.6, height=1.6, detail=48, orientation='lying', pins='none', material='silk', burnable=False)
    place = C.FabricPlace(pos=(0.0, 1.5, 0.0))
    cl = _cloth(gpu, spec, place)
    ball = ColliderGPU(shape='sphere', pos=(0.0, 0.7, 0.0), size=(0.45, 0.45, 0.45))
    x, _ = _run(cl, gpu, 2.5, place, [ball], engine.solver.meshes)
    d = np.linalg.norm(x - np.array([0.0, 0.7, 0.0]), axis=1)
    assert d.min() > 0.45 - 0.004, 'nothing inside the ball'
    assert x[:, 1].max() < 0.45 + 0.7 + 0.02, 'it lies on the ball'
    assert x[:, 1].min() < 0.5, 'and hangs down around it'
    assert np.ptp(x[:, 0]) < 1.3, 'gathered into folds, not spread flat'


def test_a_sheet_folding_onto_itself_does_not_pass_through(gpu, engine):
    """A sheet dropped over a narrow post folds and piles up around it without passing through itself."""
    from blackbody.engine.solver import ColliderGPU
    spec = C.FabricSpec(width=1.4, height=1.4, detail=36, orientation='lying', pins='none', material='cotton', burnable=False)
    place = C.FabricPlace(pos=(0.03, 1.0, 0.02))
    cl = _cloth(gpu, spec, place)
    post = ColliderGPU(shape='box', pos=(0.0, 0.3, 0.0), size=(0.08, 0.3, 0.08))
    x, _ = _run(cl, gpu, 3.0, place, [post], engine.solver.meshes)
    assert np.isfinite(x).all() and x[:, 1].min() >= 0.0
    assert x[:, 1].max() < 0.66, 'it has fallen onto the post'
    B = cl.built
    close = 0
    for k in range(0, len(x), 37):
        dx = np.linalg.norm(x - x[k], axis=1)
        dr = np.linalg.norm(B.rest - B.rest[k], axis=1)
        close += int(((dx < 0.5 * B.radius) & (dr > 4 * B.mean_edge)).sum())
    assert close == 0, 'no two far-apart parts of the sheet sit inside each other'


def test_a_fine_sheet_dropped_fast_onto_a_coarse_one_stays_on_it(gpu, engine):
    """A finely meshed sheet falling a metre lands on a coarse one held at its corners and stays on top: it
    moves further than its thickness each step, and the coarse one's vertices are far apart."""
    coarse = C.FabricSpec(width=1.2, height=1.2, detail=10, orientation='lying', pins='corners', material='canvas',
                          burnable=False)
    fine = C.FabricSpec(width=0.6, height=0.6, detail=36, orientation='lying', pins='none', material='silk', burnable=False)
    cl = C.Cloth(gpu)
    assert cl.configure([coarse, fine])
    places = [C.FabricPlace(pos=(0.0, 0.6, 0.0)), C.FabricPlace(pos=(0.02, 1.6, 0.01))]
    cl.place(places)
    chunk = 20
    with gpu.batch() as b:
        for _ in range(int(1.5 * C.STEPS_PER_SECOND / chunk)):
            cl.step(b, None, chunk / C.STEPS_PER_SECOND, places, None, LOOK, [], engine.solver.meshes, steps=chunk)
    x, _ = cl.positions()
    (a0, n0), (a1, n1) = cl.built.ranges
    xc, xf = x[a0:a0 + n0], x[a1:a1 + n1]
    assert np.isfinite(x).all()
    assert xf[:, 1].max() < 0.75, 'it has landed'
    # the coarse sheet's height under each point of the fine one (its nearest vertices, weighted)
    d = np.linalg.norm(xf[:, None, [0, 2]] - xc[None, :, [0, 2]], axis=2)
    near = np.argsort(d, axis=1)[:, :3]
    w = 1.0 / (np.take_along_axis(d, near, 1) + 1e-4)
    under = (xc[near, 1] * w).sum(1) / w.sum(1)
    assert (xf[:, 1] > under - 0.01).mean() > 0.995, 'it lies on the coarse sheet, not through it'


def test_many_fabrics(gpu):
    """More than eight fabrics in one scene, each with its own material and thickness."""
    names = list(C.MATERIALS)
    specs = [C.FabricSpec(width=0.3, height=0.4, detail=8, pins='top', material=names[i % len(names)], burnable=False,
                          self_collide=False) for i in range(12)]
    cl = C.Cloth(gpu)
    assert cl.configure(specs)
    places = [C.FabricPlace(pos=(0.4 * (i % 4) - 0.6, 1.2, 0.4 * (i // 4) - 0.4)) for i in range(12)]
    cl.place(places)
    with gpu.batch() as b:
        for _ in range(30):
            cl.step(b, None, 20 / C.STEPS_PER_SECOND, places, None, LOOK, [], None, steps=20)
    x, _ = cl.positions()
    assert np.isfinite(x).all()
    for (a, n), p in zip(cl.built.ranges, places):
        assert x[a:a + n, 1].min() == pytest.approx(1.2 - 0.4 + 0.2, abs=0.02), 'each hangs its own length from its own pins'


def _flag_scene(wind):
    sc = _plain(presets.make('smoke_plume'))
    sc.data['motion'].update(turbulence=0.0, disturbance=0.0, wind_speed=wind, wind_dir=90.0)
    for e in sc.emitters:
        e['enabled'] = False
    sc.add_fabric(position=(-0.2, 2.2, 0.0), width=1.5, height=1.0, material='nylon', pins='side', detail=32)
    return sc


def test_a_flag_streams_out_and_flutters_in_the_wind(engine):
    sc = _flag_scene(6.0)
    engine.invalidate()
    engine.prepare(sc)
    engine.simulate_to(sc, sc.start + 72, cache=False)
    x1, _ = engine.cloth.positions()
    engine.simulate_to(sc, sc.start + 75, cache=False)
    x2, _ = engine.cloth.positions()
    pole = -0.95
    assert x1[:, 0].max() > pole + 1.2, 'it streams out downwind'
    assert np.abs(x2 - x1).max() > 0.02, 'and keeps moving (flutters)'
    sc0 = _flag_scene(0.0)
    engine.invalidate()
    engine.prepare(sc0)
    engine.simulate_to(sc0, sc0.start + 72, cache=False)
    x0, _ = engine.cloth.positions()
    assert x0[:, 0].max() < pole + 0.6, 'in still air it hangs down the pole'


def test_fabric_slows_the_air_it_hangs_in(engine):
    def wind_behind(with_cloth):
        sc = _plain(presets.make('smoke_plume'))
        sc.data['motion'].update(turbulence=0.0, disturbance=0.0, wind_speed=4.0, wind_dir=90.0)
        for e in sc.emitters:
            e['enabled'] = False
        if with_cloth:
            sc.add_fabric(position=(0.0, 2.0, 0.0), width=2.5, height=3.0, material='canvas', pins='corners', detail=24,
                          yaw=90.0)
        engine.invalidate()
        engine.prepare(sc)
        engine.simulate_to(sc, sc.start + 60, cache=False)
        s = engine.solver
        v = s.read_velocity().astype(np.float32)
        ix = int((0.6 - s.origin[0]) / s.h)
        iy0, iy1 = int((1.0 - s.origin[1]) / s.h), int((3.0 - s.origin[1]) / s.h)
        iz0, iz1 = int((-0.8 - s.origin[2]) / s.h), int((0.8 - s.origin[2]) / s.h)
        return float(v[iz0:iz1, iy0:iy1, ix, 0].mean())
    free, held = wind_behind(False), wind_behind(True)
    assert held < 0.8 * free, (free, held)


def _curtain_scene(material, res=48):
    sc = _plain(presets.make('campfire'), res)
    sc.add_fabric(position=(0.0, 1.05, -0.18), width=1.0, height=1.5, material=material, pins='top', detail=36)
    return sc


def test_a_curtain_by_the_fire_catches_burns_through_and_feeds_the_fire(engine):
    sc = _curtain_scene('cotton')
    engine.invalidate()
    engine.prepare(sc)
    engine.simulate_to(sc, sc.start + 96, cache=False)
    st = engine.cloth.state()
    assert (st[:, 2] > 0.5).any() or (st[:, 1] > 0.5).any(), 'it caught'
    fuel = float(engine.solver.read_scalars()[..., 1].astype(np.float32).sum())
    engine.simulate_to(sc, sc.start + 240, cache=False)
    st = engine.cloth.state()
    assert (st[:, 1] >= 1.0).mean() > 0.5, 'most of it has burnt through'
    plain = _plain(presets.make('campfire'), 48)
    engine.invalidate()
    engine.prepare(plain)
    engine.simulate_to(plain, plain.start + 96, cache=False)
    assert fuel > 1.15 * float(engine.solver.read_scalars()[..., 1].astype(np.float32).sum()), 'it feeds the fire'


def test_wool_puts_itself_out_where_cotton_burns_away(engine):
    """A brief flame against the hem: cotton keeps burning up the curtain, wool does not."""
    gone = {}
    for mat in ('cotton', 'wool'):
        sc = _plain(presets.make('campfire'), 40)
        for e in sc.emitters:
            e['stop'] = 2.5
        sc.add_fabric(position=(0.0, 1.05, -0.18), width=1.0, height=1.5, material=mat, pins='top', detail=28)
        engine.invalidate()
        engine.prepare(sc)
        engine.simulate_to(sc, sc.start + 240, cache=False)
        gone[mat] = float((engine.cloth.state()[:, 1] >= 1.0).mean())
    assert gone['cotton'] > 0.15 and gone['wool'] < 0.3 * gone['cotton'], gone


def test_polyester_shrinks_and_melts_away_from_the_heat(engine):
    sc = _curtain_scene('polyester', 40)
    engine.invalidate()
    engine.prepare(sc)
    engine.simulate_to(sc, sc.start + 72, cache=False)
    st = engine.cloth.state()
    assert st[:, 3].max() > 0.5, 'it softened and shrank where it is hot'
    assert (st[:, 1] >= 1.0).any(), 'and melted through'


def test_fabric_renders_under_the_fire_and_behind_objects_in_the_shot(engine):
    sc = _plain(presets.make('campfire'), 40)
    sc.add_fabric(position=(0.0, 1.0, -0.6), width=1.2, height=1.6, material='cotton', pins='top', detail=24, colour=(0.8, 0.8, 0.8))
    engine.invalidate()
    engine.prepare(sc)
    f = sc.start + 24
    engine.simulate_to(sc, f, cache=True)
    engine.render(sc, f, (200, 150), mode='fire', samples=2)
    aov = engine.aovs()
    fab = aov['fabric'].astype(np.float32)
    seen = fab[..., 3] > 0.5
    assert seen.sum() > 200, 'the fabric is drawn'
    b = aov['beauty'].astype(np.float32)
    assert (b[..., 3][seen] > 0.99).mean() > 0.95, 'it is opaque'
    # a box in the shot in front of half of it hides that half
    box = dict(collider_defaults(), name='Box', shape='box', position=(0.4, 1.0, 0.2), size=(0.4, 0.8, 0.1), holdout=True)
    sc.colliders.append(box)
    engine.invalidate()
    engine.prepare(sc)
    engine.simulate_to(sc, f, cache=True)
    engine.render(sc, f, (200, 150), mode='fire', samples=2)
    seen2 = engine.aovs()['fabric'].astype(np.float32)[..., 3] > 0.5
    assert seen2.sum() < 0.85 * seen.sum(), 'the box hides part of it'


def test_fabric_casts_shadows(engine):
    """A canopy over the ground shades the footage under it from the key light; a curtain behind a box
    gets less of the fire's light than one in the open."""
    sc = _plain(presets.make('campfire'), 40)
    sc.data['lighting']['sun_elevation'] = 65.0
    sc.data['camera'].update(pitch=40.0, anchor_y=0.6)
    sc.add_fabric(position=(0.55, 0.9, 0.0), width=0.7, height=0.7, material='canvas', pins='corners',
                  orientation='lying', detail=20, burnable=False)
    engine.invalidate()
    engine.prepare(sc)
    f = sc.start + 12
    engine.simulate_to(sc, f, cache=True)
    plate = np.full((150, 200, 4), 0.45, np.float32)
    engine.render(sc, f, (200, 150), mode='composite', plate=plate)
    lamps = engine.aovs()['lamps'][..., :3].astype(np.float32).mean(-1)
    assert lamps.min() < -0.3, 'the ground under the canopy is in its shadow'
    assert (lamps < -0.1).sum() > 50
    assert lamps.max() <= 1e-3, 'and nothing is lit up by it'

    def fire_lit(box):
        s = _plain(presets.make('campfire'), 40)
        s.data['lighting'].update(sun_intensity=0.0, ambient_intensity=0.0)   # the fire's light alone
        s.add_fabric(position=(0.0, 0.8, -0.7), width=1.2, height=1.2, material='cotton', pins='top', detail=20,
                     burnable=False, colour=(0.8, 0.8, 0.8))
        if box:   # between the fire and the curtain, not in the shot (the curtain is seen through it)
            s.colliders.append(dict(collider_defaults(), name='Box', shape='box', position=(0.0, 0.6, -0.4),
                                    size=(0.35, 0.35, 0.05), holdout=False))
        engine.invalidate()
        engine.prepare(s)
        g = s.start + 24
        engine.simulate_to(s, g, cache=True)
        engine.render(s, g, (200, 150), mode='fire', samples=1)
        fab = engine.aovs()['fabric'].astype(np.float32)
        seen = fab[..., 3] > 0.5
        return fab[..., :3].mean(-1)[seen].mean()

    shaded, open_ = fire_lit(True), fire_lit(False)
    assert shaded < 0.7 * open_, f'the box shades the curtain from the fire ({shaded:.4f} vs {open_:.4f})'


def test_fabric_floats_soaks_and_sinks_in_water(engine):
    """Dropped on a pond, dry cotton floats on the air in its weave, soaks, and slowly sinks; it is seen on
    the water from above."""
    sc = presets.make('floating')
    sc.data['domain'].update(resolution=80, preroll=0.0)
    lvl = sc.liquid_level(sc.start)
    sc.add_fabric(position=(0.0, lvl + 0.15, 0.3), width=0.6, height=0.6, material='cotton', pins='none',
                  orientation='lying', detail=16, burnable=False)
    engine.invalidate()
    engine.prepare(sc)
    wet = lambda: np.frombuffer(engine.gpu.read_buffer(engine.cloth.bufs['P']), np.float32).reshape(-1, 4)[:engine.cloth.built.n, 3]
    f1 = sc.start + int(0.75 * sc.fps)
    engine.simulate_to(sc, f1, cache=True)
    x1, _ = engine.cloth.positions()
    w1 = wet()
    assert np.isfinite(x1).all()
    assert abs(float(np.median(x1[:, 1])) - lvl) < 0.06, 'it floats at the surface'
    assert 0.0 < w1.mean() < 0.6, 'and has begun to soak'
    engine.render(sc, f1, (240, 135), mode='composite')
    lay = engine.gpu.read(engine.cloth.lq).astype(np.float32)
    seen = lay[..., 3] >= 0.0
    img = np.asarray(engine.linear_comp(), np.float32)
    assert seen.sum() > 200, 'it is drawn'
    red = img[..., 0][seen].mean() / max(img[..., 1][seen].mean(), 1e-6)
    assert red > 1.5, f'and seen (the fabric is red; the water and the sky are not): {red:.2f}'
    f2 = sc.start + int(3.0 * sc.fps)
    engine.simulate_to(sc, f2, cache=True)
    x2, _ = engine.cloth.positions()
    w2 = wet()
    assert w2.mean() > max(0.5, w1.mean() + 0.2), 'it soaks through'
    assert float(np.median(x2[:, 1])) < float(np.median(x1[:, 1])) - 0.005, 'and, soaked, it sinks'


def test_wet_fabric_does_not_catch(engine):
    """A soaked curtain by the fire stays at the boil while its water boils off, and does not catch."""
    sc = _curtain_scene('cotton')
    engine.invalidate()
    engine.prepare(sc)
    engine.simulate_to(sc, sc.start + 1, cache=False)
    p = np.frombuffer(engine.gpu.read_buffer(engine.cloth.bufs['P']), np.float32).reshape(-1, 4).copy()
    p[:, 3] = 1.0
    engine.gpu.write_buffer(engine.cloth.bufs['P'], p)
    engine.simulate_to(sc, sc.start + 96, cache=False)
    st = engine.cloth.state()
    w = np.frombuffer(engine.gpu.read_buffer(engine.cloth.bufs['P']), np.float32).reshape(-1, 4)[:engine.cloth.built.n, 3]
    assert not (st[:, 2] > 0.5).any() and not (st[:, 1] > 0.0).any(), 'nothing caught (dry, it would have)'
    assert st[:, 0].max() < 374.0, 'held at the boil'
    assert w.min() < 0.98, 'and the fire is drying it'


def test_fabric_from_the_cache_looks_as_it_did_live(engine):
    sc = _curtain_scene('cotton', 40)
    engine.invalidate()
    engine.prepare(sc)
    f = sc.start + 48
    engine.simulate_to(sc, f, cache=True)
    engine.render(sc, f, (160, 120), mode='fire', samples=1)
    live = engine.aovs()['fabric'].astype(np.float32)
    engine.simulate_to(sc, f + 6, cache=True)
    engine.render(sc, f, (160, 120), mode='fire', samples=1)
    cached = engine.aovs()['fabric'].astype(np.float32)
    assert (live[..., 3] > 0.5).sum() > 100
    assert np.abs(live - cached).max() < 0.02


def test_fabric_is_saved_with_the_scene_and_changes_the_simulation():
    sc = presets.make('campfire')
    sig = sc.sim_signature()
    sc.add_fabric(material='silk', pins='side', width=0.9)
    assert sc.sim_signature() != sig
    back = type(sc).from_dict(sc.to_dict())
    assert back.fabrics[0]['material'] == 'silk' and back.fabrics[0]['pins'] == 'side'
    assert back.fabric_specs()[0].width == pytest.approx(0.9)


def test_fabric_exports_as_meshes(engine, tmp_path):
    from blackbody.io import fabric_mesh as FM
    sc = _curtain_scene('cotton', 40)
    engine.invalidate()
    engine.prepare(sc)
    f = sc.start + 216
    engine.simulate_to(sc, f, cache=True)
    meshes = FM.fabric_meshes(engine, sc, f)
    assert len(meshes) == 1
    m = meshes[0]
    n = engine.cloth.built.n
    assert m['points'].shape == (n, 3) and m['faces'].max() < n
    assert len(m['faces']) < len(engine.cloth.built.tris), 'burnt-through triangles are left out'
    p = FM.write_obj(tmp_path / 'cloth.0001.obj', meshes)
    text = p.read_text()
    assert text.count('\nv ') == n and text.count('\nf ') == len(m['faces'])
    pytest.importorskip('pxr')
    w = FM.UsdWriter(tmp_path / 'cloth.usda', sc.fps)
    w.add(f, meshes)
    w.close()
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(str(tmp_path / 'cloth.usda'))
    prim = st.GetPrimAtPath('/World/Fabric/' + FM._safe(m['name']))
    assert prim and len(UsdGeom.Mesh(prim).GetPointsAttr().Get(f)) == n


def _box_through_a_sheet(engine, tears):
    from blackbody.scene import components
    sc = components.new_scene('fire', 'person')
    sc.emitters = []
    sc.add_fabric(name='Sheet', width=1.0, height=1.2, position=(0.0, 1.3, 0.0), pins='top', material='cotton', tears=tears)
    sc.add_collider(name='Box', shape='box', position=(0.0, 1.15, -0.6), size=(0.15, 0.15, 0.15), material='steel')
    sc.set_key(('collider', 0, 'position'), sc.start, (0.0, 1.15, -0.6))
    sc.set_key(('collider', 0, 'position'), sc.start + 36, (0.0, 1.15, 0.9))
    sc.data['domain'].update(size_x=1.6, size_y=2.2, size_z=2.0, resolution=32, preroll=0.0)
    engine.invalidate()
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 48, cache=False)
    st = engine.cloth.state()
    return st, engine.cloth.positions()[0]


def test_a_box_rips_through_a_sheet_that_tears_and_slides_off_one_that_does_not(engine):
    st, X = _box_through_a_sheet(engine, True)
    torn = st[:, 2] < -0.5
    assert torn.sum() > 30                                # ripped where the box went through
    assert (st[torn, 1] >= 1.0).all()                    # (gone, as burnt-away cloth is)
    assert (np.abs(X[:, 2]) > 1.5).mean() < 0.1          # and most of it still hangs (a flap may go with the box)
    st, _ = _box_through_a_sheet(engine, False)
    assert not (st[:, 2] < -0.5).any() and not (st[:, 1] >= 1.0).any()   # whole


def test_a_sling_that_tears_holds_its_sand_unless_it_is_weak(engine):
    from blackbody.scene import presets
    out = {}
    for strength in (1.0, 0.2):
        sc = presets.make('sand_sling')
        sc.fabrics[0]['tears'], sc.fabrics[0]['tear_strength'] = True, strength
        engine.invalidate()
        engine.prepare(sc, final=False)
        engine.simulate_to(sc, sc.start + 84, cache=False)
        x = engine.matter.positions()[0]
        out[strength] = (int((engine.cloth.state()[:, 2] < -0.5).sum()), float((x[:, 1] < 0.3).mean()))
    assert out[1.0] == (0, 0.0)                           # cotton holds 10 litres of sand
    assert out[0.2][0] > 0 and out[0.2][1] > 0.5          # a weak one rips and the sand pours through
