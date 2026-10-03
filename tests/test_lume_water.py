"""Lume tracing the water (engine/wgsl/lume_water.wgsl, lume.wgsl lu_path; liquid_render.LumeWater): checked against
Lume's own glass, which the benchmark checks against Mitsuba (tools/lume_bench): a box of clear water held in the air
is a ball of glass of water's index of refraction, so the two must give the same picture. (Lit by the sky alone: the
key light through a glass ball would come as caustics, traced from it, and through water by the caustic map.)"""
import math

import numpy as np
import pytest

from blackbody.engine import stage as stage_mod
from blackbody.scene.model import Scene

C = np.array([0.0, 0.55, 0.0])   # the ball's middle (fire-local m)
R = 0.3                           # its radius (m)
IOR = 1.333
N = 32                            # the water's grid (cells a side)
H = 0.8 / N                       # its cell (m)
ORG = C - 0.4


def _euler_looking(eye, target):
    from blackbody.scene.groundmatch import euler_xyz
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, (0.0, 1.0, 0.0))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return euler_xyz(np.column_stack([r, u, -f]))


def _scene(box):
    sc = Scene()
    for k in ('emitters', 'colliders', 'lights', 'fabrics', 'matter', 'strands'):   # (a new scene has a fire in it)
        getattr(sc, k, []).clear()
    sc.data['domain'].update(size_x=6.0, size_y=3.0, size_z=6.0, resolution=24, ground=True)
    sc.data['render'].update(width=160, height=90, motion_blur=False)
    eye = (1.3, 1.5, 1.7)
    sc.data['camera'].update(mode='free', position=eye, rotation=tuple(_euler_looking(eye, C)), focal_mm=18.0 / math.tan(math.radians(20.0)),
                             sensor_mm=36.0, use_anchor=False, fire_position=(0.0, 0.0, 0.0), fire_yaw=0.0, scale=1.0, roll=0.0,
                             near=0.02)
    sc.data['lighting'].update(sun_on=False, environment='')
    sc.data['composite'].update(backdrop='stage', floor='concrete', floor_tint=(1.0, 1.0, 1.0), visibility=0.0)
    if box:
        sc.add_collider(name='Ball', shape='sphere', position=tuple(C), size=(R, R, R), material='plaster', look='cg')
    return sc


def _water():
    """A ball of clear water the size of the glass one, as the liquid's surface grid holds it: its signed distance
    (grid cells, negative in it)."""
    from blackbody.engine.liquid_render import LumeWater
    idx = np.stack(np.meshgrid(*(np.arange(N),) * 3, indexing='ij'), -1)[..., ::-1]   # z y x -> (x, y, z)
    p = ORG + (idx + 0.5) * H
    sdf = np.linalg.norm(p - C, axis=-1) - R
    surf = np.zeros((N, N, N, 4), np.float32)
    surf[..., 0] = sdf / H
    data = [*ORG, H, N, N, N, 1.0,
            0.0, 0.0, 0.0, 0.0,     1.0, 1.0, 1.0, IOR,
            1.0, 0.0, -1.0, 8.0,    0.0, 25.0, 2.0, 0.0,
            1.0, 1.0, 1.0, 1.0,     0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 0.012,   0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,     -1.0, 0.0, 0.0, 0.0]
    return surf, data


def _render(engine, sc, water=None):
    from blackbody.engine.liquid_render import LumeWater
    f = sc.start + 1
    sc.data['lume'].update(engine='lume', samples=512, bounces=6, denoise=False)
    engine.prepare(sc, final=True)
    engine.simulate_to(sc, f, cache=False)
    orig = engine.stage.draw
    if water is not None:
        g = engine.gpu
        surf, data = water
        t = g.texture3d((N, N, N), 'rgba16float', 'test-water')
        g.upload(t, surf.astype(np.float16))
        r = engine.renderer
        lw = LumeWater(t, r._empty, r._black, r._black, data)
        engine.stage.draw = lambda b, *a, **k: orig(b, *a, **{**k, 'water': lw})
    try:
        engine.render(sc, f, (160, 90), mode='composite', final=True, samples=4, motion_blur=False)
    finally:
        if water is not None:
            del engine.stage.draw
    return engine.gpu.read(engine.stage.tex).astype(np.float32)[..., :3]


def test_a_ball_of_water_in_the_air_is_lit_and_seen_as_a_ball_of_glass_of_waters_index_is(engine, monkeypatch):
    glass_row = (stage_mod.CG, (1.0, 1.0, 1.0), 0.03, 0.0, 1.0, -2, (1.0, 1.0, 1.0), IOR)
    monkeypatch.setattr(stage_mod, 'looks', lambda scene, footage: [glass_row] * len(scene.colliders))
    empty = _render(engine, _scene(False))
    glass = _render(engine, _scene(True))
    water = _render(engine, _scene(False), water=_water())
    assert np.isfinite(water).all()
    luma = np.array([0.2126, 0.7152, 0.0722])
    lg, lw, le = glass @ luma, water @ luma, empty @ luma
    ball = np.abs(lg - le) > 0.02 * le.mean()                  # (where the ball changes the picture: it and its shade)
    assert ball.mean() > 0.05
    assert lw[ball].mean() == pytest.approx(lg[ball].mean(), rel=0.02)
    assert lw.mean() == pytest.approx(lg.mean(), rel=0.01)
    # and pixel by pixel, but for the grain
    assert np.median(np.abs(lw[ball] - lg[ball]) / np.maximum(lg[ball], 1e-3)) < 0.02


def test_a_liquid_scene_with_lume_has_lume_trace_its_water_and_leaves_lava_to_the_march(engine):
    from blackbody.scene import presets
    got = []
    orig = engine.stage.draw

    def draw(b, *a, **k):
        got.append(k.get('water'))
        return orig(b, *a, **k)

    for name, traced in (('floating', True), ('lava', False)):
        sc = presets.make(name, fps=24)
        sc.data['render'].update(width=160, height=90)
        f = sc.start + 20
        engine.prepare(sc, final=True)
        engine.simulate_to(sc, f, cache=False)
        sc.data['lume'].update(engine='lume', samples=16, bounces=4, denoise=True)
        got.clear()
        engine.stage.draw = draw
        try:
            engine.render(sc, f, (160, 90), mode='composite', final=True, samples=1, motion_blur=False)
        finally:
            del engine.stage.draw
        img = engine.gpu.read(engine.renderer.lin).astype(np.float32)
        assert np.isfinite(img).all()
        handed = bool(got) and got[-1] is not None   # (the stage drew it with the water to trace)
        assert handed == traced, name
