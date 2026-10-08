"""GPU tests for the third round of liquid features: the sea, bodies that tip and roll, the liquid as
a mesh, molten liquids that cool, dye and liquids of other densities, rain, currents, the camera
under water, lamps, drops on the lens, the footage's depth pass as holdout and collider, and a box
that follows the action."""
import math

import numpy as np
import pytest

from blackbody.engine.liquid import LiquidParams, LiquidSolver, rain_drops_per_m2s, rain_speed, source
from blackbody.scene import presets


@pytest.fixture(scope='module')
def liquid(engine):
    return LiquidSolver(engine.gpu)


def _setup(s, size, res, ppc=8, ww=0, band=None):
    dims, h = LiquidSolver.dims_for(size, res)
    origin = (-dims[0] * h / 2, 0.0, -dims[2] * h / 2)
    s.configure(dims, h, origin, LiquidSolver.capacity_for(dims, ppc, 4_000_000, band=band), ww)
    return dims, h, origin


def _run(s, prm, first, frames, fps=30, every=None):
    s._prm = prm
    for _ in range(frames):
        n = s.substeps_for(1 / fps, cfl=1.5, lo=1, hi=12)
        with s.gpu.batch() as b:
            for _ in range(n):
                s.step(b, 1 / fps / n, prm, first if s.steps == 0 else (every or []))
            s.pack(b)
        s.measure()


def _scene(name, res, w=320, h=180):
    sc = presets.make(name)
    sc.data['domain']['resolution'] = res
    sc.data['domain']['preroll'] = 0.0
    sc.data['render']['width'], sc.data['render']['height'] = w, h
    return sc


def _render(engine, sc, frame, plate_value=110, w=320, h=180):
    plate = np.full((h, w, 4), plate_value, np.uint8)
    engine.render(sc, frame, (w, h), plate=plate)
    img = engine.display_image()[..., :3].astype(np.float32)
    assert np.isfinite(engine.aovs()['beauty'].astype(np.float32)).all()
    return img


# -- the sea -----------------------------------------------------------------------------------------

def test_no_sea_without_waves():
    from blackbody.engine.ocean import spec_from
    q = {'ocean_height': 0.0, 'ocean_length': 5.0, 'ocean_dir': 0.0, 'ocean_spread': 0.3, 'ocean_chop': 0.8}
    assert spec_from(q, 1.0) is None


def test_the_sea_runs_and_carries_the_crates(engine):
    sc = _scene('sea_swell', 64)
    engine.invalidate()
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 24, cache=True)
    assert np.isfinite(engine.liquid.max_speed) and engine.liquid.count > 0
    ov = engine.floating_overrides(sc.start + 24)
    level = sc.data['liquid']['water_level']
    for body in (ov.values() if isinstance(ov, dict) else ov):
        assert abs(body['pos'][1] - level) < 0.45, f'a crate left the sea surface: {body["pos"]}'
    _render(engine, sc, sc.start + 24)


# -- the liquid as a mesh ------------------------------------------------------------------------------

def test_surface_nets_make_a_closed_outward_surface():
    from blackbody.io.liquid_mesh import surface_nets
    n = 24
    g = np.stack(np.meshgrid(*(np.arange(n, dtype=np.float32),) * 3, indexing='ij'), -1)   # z, y, x
    phi = np.linalg.norm(g - 11.5, axis=-1) - 7.0
    v, q, _ = surface_nets(phi)
    assert len(v) > 100 and len(q) > 100
    # closed: every edge is shared by exactly two quads
    e = np.concatenate([q[:, [0, 1]], q[:, [1, 2]], q[:, [2, 3]], q[:, [3, 0]]])
    _, counts = np.unique(np.sort(e, axis=1), axis=0, return_counts=True)
    assert (counts == 2).all()
    # outward: each quad's normal points away from the centre
    a, b, c = v[q[:, 0]], v[q[:, 1]], v[q[:, 2]]
    nrm = np.cross(b - a, c - a)
    out = (a + c) / 2 - 11.5
    assert (np.einsum('ij,ij->i', nrm, out) > 0).mean() > 0.99


def test_liquid_mesh_export(engine, tmp_path):
    from blackbody.io.liquid_mesh import liquid_surface_mesh, write_obj
    sc = _scene('water_pour', 48)
    engine.invalidate()
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 20, cache=True)
    m = liquid_surface_mesh(engine, sc, sc.start + 20)
    assert len(m['points']) > 0 and len(m['faces']) > 0 and np.isfinite(m['points']).all()
    p = write_obj(tmp_path / 'pour.obj', m)
    assert p.stat().st_size > 1000
    pytest.importorskip('pxr')
    from blackbody.io.liquid_mesh import UsdWriter
    w = UsdWriter(tmp_path / 'pour.usda', 24)
    w.add(sc.start + 20, m)
    w.close()
    assert (tmp_path / 'pour.usda').stat().st_size > 1000


# -- molten liquids ----------------------------------------------------------------------------------

def test_lava_cools_and_dims(engine):
    sc = _scene('lava', 96)   # (48 cells on the lava's 5 m box: the ground layer that cools by contact covers its vent)
    sc.data['liquid']['cooling'] = 1.5
    engine.invalidate()
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 40, cache=False)
    heat = engine.gpu.read(engine.liquid.HEAT)[..., 0]
    dens = engine.gpu.read(engine.liquid.DENS)[..., 0]
    wet = dens > 1.0
    assert wet.any()
    assert heat[wet].min() < 0.7, 'the lava cools where it meets the air and the ground'
    assert heat[wet].max() > 0.8, 'fresh lava from the vent is still hot'


# -- dye and two liquids -------------------------------------------------------------------------------

def test_dye_is_carried_and_drawn(engine):
    sc = _scene('ink_tank', 48)
    engine.invalidate()
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 30, cache=True)
    L = engine.liquid
    assert L.dye_on
    dye = L.read_dye()
    d = np.frombuffer(dye.astype('<u4').tobytes(), np.float16).reshape(-1, 4).astype(np.float32)
    inked = (d[:, :3].sum(1) > 1.0).sum()
    assert 0 < inked < 0.5 * len(d), 'some of the water carries the ink, not all'
    clear = _render(engine, sc, sc.start + 30)
    # the cached frame keeps its dye
    assert 'dye' in engine.cache.get(sc.start + 30)
    assert engine.liquid_r.dye_tex is not engine.liquid_r._no_dye
    assert clear.mean() > 5


def test_a_lighter_liquid_rises_through_the_water(liquid):
    dims, h, origin = _setup(liquid, (0.3, 0.4, 0.3), 40)
    prm = LiquidParams(open_sides=False, whitewater=False, flip=0.9)
    liquid.reset()
    oil = source('box', pos=(0.0, 0.06, 0.0), size=(0.06, 0.05, 0.06), fill=True, dye=(0.9, 0.6, 0.1),
                 dye_amount=20.0, density=0.7)
    water = source('box', pos=(0.0, 0.15, 0.0), size=(0.15, 0.15, 0.15), fill=True)

    def oil_height():
        pos, _ = liquid.read_particles()
        d = np.frombuffer(liquid.read_dye().astype('<u4').tobytes(), np.float16).reshape(-1, 4).astype(np.float32)
        return float(pos[d[:, :3].sum(1) > 0.5, 1].mean())

    _run(liquid, prm, [oil, water], 2)
    y0 = oil_height()
    _run(liquid, prm, [], 40)
    y1 = oil_height()
    assert y1 > y0 + 0.05, f'the oil rose from {y0:.3f} m to {y1:.3f} m only'


# -- rain --------------------------------------------------------------------------------------------

def test_rain_numbers():
    assert math.isclose(rain_speed(2.5), 7.35, abs_tol=0.1)          # terminal speed of a 2.5 mm drop
    assert 150 < rain_drops_per_m2s(5.0, 2.5) < 190                   # 5 mm/h of 2.5 mm drops
    from blackbody.engine.liquid_render import rain_rings
    cell, life, speed, _ = rain_rings(50.0, 2.5)
    assert speed * life <= cell + 1e-6, 'heavy rain: rings stay within their cells'
    assert rain_rings(0.0, 2.5)[0] == 0.0


def test_rain_splashes_throw_spray(liquid):
    counts = []
    for rain in (0.0, 60.0):
        _setup(liquid, (0.6, 0.4, 0.6), 40, ww=200_000)
        prm = LiquidParams(water_level=0.15, rain=rain, ww_min_speed=50.0)
        liquid.reset()
        _run(liquid, prm, [], 15)
        counts.append(liquid.ww_count)
    assert counts[1] > counts[0] + 200, f'spray without rain {counts[0]}, with {counts[1]}'


# -- currents ----------------------------------------------------------------------------------------

def test_a_current_carries_the_open_water(liquid):
    _setup(liquid, (0.8, 0.4, 0.5), 40)
    prm = LiquidParams(water_level=0.2, whitewater=False, current=(0.6, 0.0, 0.0))
    liquid.reset()
    _run(liquid, prm, [], 45)
    pos, vel = liquid.read_particles()
    assert liquid.count > 0 and np.isfinite(vel).all()
    assert np.median(vel[:, 0]) > 0.3, f'the water flows at {np.median(vel[:, 0]):.2f} m/s'
    assert abs(np.percentile(pos[:, 1], 99.5) - 0.2) < 0.06, 'and keeps its level'


# -- rendering: under water, lamps, lens drops ---------------------------------------------------------

def test_a_camera_under_water(engine):
    sc = _scene('floating', 48)
    sc.data['camera'].update(target_y=0.2, pitch=-4.0, distance=0.8)
    engine.invalidate()
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 6, cache=True)
    img = _render(engine, sc, sc.start + 6)
    aux = engine.aovs()
    assert img.mean() > 20, 'under water is lit, not black'
    # the water fills the whole frame
    assert (engine.renderer.gpu.read(engine.renderer.aux)[..., 3].astype(np.float32) > 0.99).mean() > 0.95
    assert aux is not None


def test_a_lamp_lights_the_water(engine):
    sc = _scene('floating', 48)
    engine.invalidate()
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 6, cache=True)
    dark = _render(engine, sc, sc.start + 6)
    sc.add_light(kind='point', position=(0.0, 0.8, 0.0), intensity=40000.0, colour=(1.0, 0.4, 0.1))
    lit = _render(engine, sc, sc.start + 6)
    assert lit[..., 0].mean() > dark[..., 0].mean() + 0.5, 'the lamp adds its glints and light'


def test_drops_on_the_lens_bend_the_picture(engine):
    sc = _scene('fountain', 48)
    engine.invalidate()
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 6, cache=True)
    a = _render(engine, sc, sc.start + 6)
    sc.data['water']['lens_drops'] = 2.0
    b = _render(engine, sc, sc.start + 6)
    changed = (np.abs(a - b).sum(-1) > 6).mean()
    assert 0.01 < changed < 0.6, f'{changed:.1%} of the picture changed'
    lin = engine.renderer.gpu.read(engine.renderer.lin).astype(np.float32)
    assert np.isfinite(lin).all()


# -- the footage's depth pass ---------------------------------------------------------------------------

def _wall_depth(sc, wall_x, tmp_path, frames=40):
    """A depth pass (distance) of a wall across the box at fire-local x = wall_x, and the ground."""
    from blackbody.engine import camera as cam
    from blackbody.io.images import write_exr
    W, H = sc.data['render']['width'], sc.data['render']['height']
    spec, fire = sc.camera(sc.start)
    cs = cam.compute(spec, W / H, fire)
    ys, xs = np.mgrid[0:H, 0:W]
    ndc = np.stack([(xs + 0.5) / W * 2 - 1, 1 - (ys + 0.5) / H * 2], -1).reshape(-1, 2)

    def unproj(z):
        p = np.concatenate([ndc, np.full((len(ndc), 1), z), np.ones((len(ndc), 1))], 1) @ cs.inv_view_proj.T
        return p[:, :3] / p[:, 3:]
    rd = unproj(1.0) - unproj(0.0)
    rd /= np.linalg.norm(rd, axis=1, keepdims=True)
    w2l = np.linalg.inv(fire.local_to_world())
    ro = (np.concatenate([np.asarray(cs.eye, float), [1.0]]) @ w2l.T)[:3]
    rl = rd @ w2l[:3, :3].T
    tw = (wall_x - ro[0]) / np.where(np.abs(rl[:, 0]) > 1e-9, rl[:, 0], 1e-9)
    pw = ro + rl * tw[:, None]
    hit_w = (tw > 0) & (pw[:, 1] > 0) & (pw[:, 1] < 1.0)
    tg = -ro[1] / np.where(np.abs(rl[:, 1]) > 1e-9, rl[:, 1], 1e-9)
    t = np.where(hit_w, tw, np.where(tg > 0, tg, 0.0))
    t = np.where(hit_w & (tg > 0), np.minimum(tw, tg), t).reshape(H, W).astype(np.float32)
    for f in range(1, frames + 1):
        write_exr(tmp_path / f'depth.{f:04d}.exr', {'R': t, 'G': t, 'B': t})
    sc.data['composite']['holdout_depth'] = str(tmp_path / 'depth.####.exr')
    sc.data['composite']['depth_kind'] = 'distance'
    sc.data['composite']['depth_scale'] = 1.0


def test_the_footage_depth_stops_the_liquid(engine, tmp_path):
    fronts = []
    for collide in (False, True):
        sc = _scene('water_pour', 64)
        sc.data['camera'].update(distance=1.5, target_y=0.05, pitch=35, yaw=-20, anchor_y=0.5)
        _wall_depth(sc, 0.15, tmp_path)
        sc.data['liquid']['footage_collide'] = collide
        engine.invalidate()
        engine.prepare(sc, final=False)
        engine.simulate_to(sc, sc.start + 40, cache=True)
        pos, _ = engine.liquid.read_particles()
        fronts.append(float(np.percentile(pos[:, 0], 99.5)))
        _render(engine, sc, sc.start + 40)
    assert fronts[0] > 0.2, f'without it the water runs on past the wall ({fronts[0]:.3f} m)'
    assert fronts[1] < 0.15 + 0.03, f'with it the wall stops the water ({fronts[1]:.3f} m)'


def _box_mask(sc, centre, half, W, H):
    """Which pixels (H, W) see an upright box (fire-local centre and half size), and how far away it is there (m)."""
    from blackbody.engine import camera as cam
    spec, fire = sc.camera(sc.start)
    cs = cam.compute(spec, W / H, fire)
    ys, xs = np.mgrid[0:H, 0:W]
    ndc = np.stack([(xs + 0.5) / W * 2 - 1, 1 - (ys + 0.5) / H * 2], -1).reshape(-1, 2)

    def unproj(z):
        p = np.concatenate([ndc, np.full((len(ndc), 1), z), np.ones((len(ndc), 1))], 1) @ cs.inv_view_proj.T
        return p[:, :3] / p[:, 3:]
    rd = unproj(1.0) - unproj(0.0)
    rd /= np.linalg.norm(rd, axis=1, keepdims=True)
    w2l = np.linalg.inv(fire.local_to_world())
    ro = (np.concatenate([np.asarray(cs.eye, float), [1.0]]) @ w2l.T)[:3]
    rl = rd @ w2l[:3, :3].T
    inv = 1.0 / np.where(np.abs(rl) > 1e-9, rl, 1e-9)
    ta = (np.asarray(centre) - np.asarray(half) - ro) * inv
    tb = (np.asarray(centre) + np.asarray(half) - ro) * inv
    t0, t1 = np.minimum(ta, tb).max(1), np.maximum(ta, tb).min(1)
    hit = (t1 > np.maximum(t0, 0.0))
    return hit.reshape(H, W), t0.reshape(H, W)


def test_the_water_shows_the_ground_behind_a_rock_not_a_ghost_of_it(engine, monkeypatch):
    """A rock standing in a pond, in the footage. Rays through the water that reach the floor behind it look the footage
    up where that floor is on screen, which is where the rock is: they showed the rock's own pixels there, a ghost of it
    on the water beside it. The footage there is filled in from beside the rock now (liq_clean.wgsl)."""
    from blackbody.engine.liquid_render import LiquidRenderer
    W, H = 320, 180
    sc = _scene('floating', 64, W, H)
    centre, half = (0.15, 0.25, 0.0), (0.15, 0.25, 0.15)
    sc.colliders = sc.colliders[:1]
    sc.colliders[0].update(shape='box', position=centre, size=half, yaw=0.0, floating=False, density=0.0)
    sc.data['water'].update(colliders_look='holdout', ripple=0.0)
    engine.invalidate()
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 6, cache=True)
    rock, _ = _box_mask(sc, centre, half, W, H)
    assert 200 < rock.sum() < 0.3 * W * H
    plate = np.full((H, W, 4), 110, np.uint8)
    plate[rock] = (220, 30, 30, 255)   # (the rock is red in the footage)

    def red_behind(clean):
        if not clean:
            monkeypatch.setattr(LiquidRenderer, '_clean_plate', lambda self, b, *a: self._no_clean)
        engine.render(sc, sc.start + 6, (W, H), plate=plate)
        monkeypatch.undo()
        beauty = engine.aovs()['beauty'].astype(np.float32)
        aux = engine.aovs()['aux'].astype(np.float32)
        # the water behind the rock: above it on screen, where it shows the floor behind it (below and round it the
        # water mirrors the rock and shows its foot under the surface: the rock's own red, rightly)
        rows = np.arange(H)[:, None]
        top = np.where(rock.any(0), np.argmax(rock, 0), -1)[None, :]
        water = (aux[..., 3] > 0.5) & (rows < top - 2)
        red = np.maximum(beauty[..., 0] - 0.5 * (beauty[..., 1] + beauty[..., 2]), 0.0)
        return float(red[water].sum()), int(water.sum())
    ghost, n0 = red_behind(clean=False)
    fixed, n1 = red_behind(clean=True)
    assert n0 > 300 and n1 == n0
    assert ghost > 5.0, f'the scene shows the ghost without the fix ({ghost:.2f})'
    assert fixed < 0.1 * ghost, f'red behind the rock: {fixed:.2f} (was {ghost:.2f})'
    near, clean = (engine.gpu.read(t) for t in engine.liquid_r._clean)
    covered = np.abs(near[..., 0]) < 60000.0
    assert covered[rock].mean() > 0.9 and covered[~rock].mean() < 0.02, 'the rock is found where it is in the footage'


def test_a_real_wall_behind_the_water_stays_in_its_reflection(engine, tmp_path, monkeypatch):
    """A wall in the footage's depth pass standing behind a pond, seen from low down: the water mirrors it. A reflection
    (or a ray that leaves the water for the backdrop) may well meet such a wall, so for those rays only the colliders and
    the matte are filled in, not the depth pass's surfaces: the reflection is the same with the filled-in footage as
    without it. (With the wall filled in for them too, a tenth of its blue went from the water.)"""
    from blackbody.engine import camera as cam
    from blackbody.engine.liquid_render import LiquidRenderer
    from blackbody.io.images import write_exr
    W, H = 320, 180
    sc = _scene('floating', 64, W, H)
    sc.data['camera']['pitch'] = 8.0
    spec, fire = sc.camera(sc.start)
    cs = cam.compute(spec, W / H, fire)
    ys, xs = np.mgrid[0:H, 0:W]
    ndc = np.stack([(xs + 0.5) / W * 2 - 1, 1 - (ys + 0.5) / H * 2], -1).reshape(-1, 2)

    def unproj(z):
        p = np.concatenate([ndc, np.full((len(ndc), 1), z), np.ones((len(ndc), 1))], 1) @ cs.inv_view_proj.T
        return p[:, :3] / p[:, 3:]
    rd = unproj(1.0) - unproj(0.0)
    rd /= np.linalg.norm(rd, axis=1, keepdims=True)
    w2l = np.linalg.inv(fire.local_to_world())
    ro = (np.concatenate([np.asarray(cs.eye, float), [1.0]]) @ w2l.T)[:3]
    rl = rd @ w2l[:3, :3].T
    # a wall 4 m wide and 3 m high, upright and square to the view, a little past the far side of the box
    half = np.array([sc.data['domain'][k] for k in ('size_x', 'size_y', 'size_z')]) * 0.5
    fh = -ro * np.array([1.0, 0.0, 1.0])
    fh /= np.linalg.norm(fh)
    p0 = fh * (np.abs(fh) @ half + 0.3)
    tw = ((p0 - ro) @ fh) / np.where(np.abs(rl @ fh) > 1e-9, rl @ fh, 1e-9)
    pw = ro + rl * tw[:, None]
    tg = -ro[1] / np.where(np.abs(rl[:, 1]) > 1e-9, rl[:, 1], 1e-9)
    wall = ((tw > 0) & (pw[:, 1] > 0) & (pw[:, 1] < 3.0) & (np.abs((pw - p0) @ np.cross([0.0, 1.0, 0.0], fh)) < 2.0)
            & ((tg <= 0) | (tw < tg)))
    t = np.where(wall, tw, np.where(tg > 0, tg, 0.0)).reshape(H, W).astype(np.float32)
    for f in range(1, 8):
        write_exr(tmp_path / f'depth.{f:04d}.exr', {'R': t, 'G': t, 'B': t})
    sc.data['composite'].update(holdout_depth=str(tmp_path / 'depth.####.exr'), depth_kind='distance', depth_scale=1.0)
    wall = wall.reshape(H, W)
    plate = np.full((H, W, 4), 110, np.uint8)
    plate[wall] = (30, 30, 220, 255)   # (the wall is blue in the footage)
    engine.invalidate()
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 6, cache=True)

    def blue(clean):
        if not clean:
            monkeypatch.setattr(LiquidRenderer, '_clean_plate', lambda self, b, *a: self._no_clean)
        engine.render(sc, sc.start + 6, (W, H), plate=plate)
        monkeypatch.undo()
        beauty = engine.aovs()['beauty'].astype(np.float32)
        water = (engine.aovs()['aux'][..., 3].astype(np.float32) > 0.5) & ~wall
        return float(np.maximum(beauty[..., 2] - 0.5 * (beauty[..., 0] + beauty[..., 1]), 0.0)[water].sum())
    plain, kept = blue(False), blue(True)
    assert plain > 500.0, f'the water mirrors the wall ({plain:.1f})'
    assert kept > 0.99 * plain, f'its reflection with the filled-in footage: {kept:.1f} of {plain:.1f}'


# -- a box that follows --------------------------------------------------------------------------------

def test_the_box_follows_the_boat(engine):
    sc = _scene('boat_wake', 40)
    sc.data['domain']['size_x'] = 3.0
    sc.data['liquid']['follow'] = 'collider'
    engine.invalidate()
    engine.prepare(sc, final=False)
    engine.simulate_to(sc, sc.start + 10, cache=True)
    x0 = engine.liquid.origin[0]
    n0 = engine.liquid.count
    engine.simulate_to(sc, sc.start + 40, cache=True)
    x1 = engine.liquid.origin[0]
    # the box keeps up with the hull, to within a few cells
    hull = [sc.colliders_gpu(sc.start + f)[0].pos[0] for f in (9, 39)]
    moved, want = x1 - x0, hull[1] - hull[0]
    assert want > 1.0 and abs(moved - want) < 4.0 * engine.liquid.h, f'the box moved {moved:.2f} m, the hull {want:.2f} m'
    # and keeps its water (it does not drain as it goes)
    assert abs(engine.liquid.count - n0) < 0.03 * n0, f'{n0} particles became {engine.liquid.count}'
    assert engine.liquid.count > 0 and np.isfinite(engine.liquid.max_speed)
    # each cached frame keeps where the box was
    assert engine.cache.get(sc.start + 10)['origin'][0] < engine.cache.get(sc.start + 40)['origin'][0]
    _render(engine, sc, sc.start + 10)
    _render(engine, sc, sc.start + 40)
