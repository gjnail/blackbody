"""Lume for the smoke (engine/lume_volume.py, wgsl/lume_volume.wgsl): the light it traces through the volume, against
what is known exactly."""
import math

import numpy as np
import pytest

from blackbody.engine import camera as cam
from blackbody.engine import lume_volume as LVM
from blackbody.engine.engine import VolumeView
from blackbody.engine.renderer import LookParams

N, H = 48, 0.05          # a 2.4 m box
ORG = (-N * H / 2, 0.0, -N * H / 2)


def _vol(engine, scal):
    t = engine.gpu.texture3d((N, N, N), 'rgba16float', 'test-scal')
    engine.gpu.upload(t, scal.astype(np.float16))
    return VolumeView(scal=[t], vel=[t], dims=(N, N, N), h=H, origin=ORG)


def _trace(engine, vol, look, ground=False, floor=(0.0, 0.0, 0.0), passes=1, rays=256, monkeypatch=None):
    """The light Lume finds in each cell (mean radiance, z y x rgb) and the way it comes from (z y x xyz), with its
    grid's dims."""
    monkeypatch.setattr(LVM, 'FINAL', (passes, rays))
    monkeypatch.setattr(LVM, 'FRESH_FINAL', passes)
    r = engine.renderer
    lv = LVM.LumeVolume(engine.gpu)
    fire = cam.FireXform()
    with engine.gpu.batch() as b:
        r.light(b, vol, look, fire, 0.0)
        lv.compute(b, r, vol, look, fire, [], None, [], floor, ground, 0.0, True)
    r.L0_march = r.L1_march = None
    rd = lambda t: engine.gpu.read(t).astype(np.float32)
    return rd(lv.lv[0])[..., :3], rd(lv.lvd[0])[..., :3], lv.dims


def _centre(dims, i):
    """Fire-local centre of LV cell i (x, y, z)."""
    return np.asarray(ORG) + (np.asarray(i) + 0.5) * (N / np.asarray(dims, float)) * H


def _dark_look(**kw):
    return LookParams(**{'ambient': (0.0, 0.0, 0.0), 'sun_intensity': 0.0, **kw})


def test_empty_air_under_an_even_sky_sees_just_the_sky(engine, monkeypatch):
    look = _dark_look()
    look.ambient = (0.2, 0.3, 0.5)
    lv, w, dims = _trace(engine, _vol(engine, np.zeros((N, N, N, 4))), look, monkeypatch=monkeypatch)
    assert np.allclose(lv, look.ambient, rtol=0.01)
    assert np.abs(w).max() < 0.02                      # from every way alike


def test_over_open_ground_half_the_light_is_the_sky_and_half_the_lit_ground(engine, monkeypatch):
    """No smoke, a grey ground under a high sun and a grey sky: from above, the sky; from below, the ground, as bright as
    the sun and the sky's light on it make it (albedo / pi x irradiance). Inside the box and past its sides alike."""
    look = _dark_look(sun_intensity=3.0, sun_elevation=90.0, sun_color=(1.0, 1.0, 1.0))
    look.ambient = (0.3, 0.3, 0.3)
    a = 0.4
    lv, w, dims = _trace(engine, _vol(engine, np.zeros((N, N, N, 4))), look, ground=True, floor=(a, a, a), passes=4,
                         monkeypatch=monkeypatch)
    ground = a / math.pi * (3.0 + math.pi * 0.3)
    want = 0.5 * (0.3 + ground)
    mid = lv[:, dims[1] // 2]
    assert np.allclose(mid, want, rtol=0.03), (mid.mean(), want)
    # and the moment: the sky brighter than the ground, the light comes from above, by (sky - ground) / (2 (sky + ground))
    assert np.allclose(w[:, dims[1] // 2, :, 1], (0.3 - ground) / (2 * (0.3 + ground)), atol=0.02)


@pytest.mark.parametrize('sigma', [1.0, 30.0])
def test_inside_a_glowing_ball_the_light_is_the_flame_seen_through_itself(engine, monkeypatch, sigma):
    """A ball of flame gas at the flame temperature, no smoke, black sky: each ray from a point inside sees the flame's
    radiance B times 1 - exp(-sigma s), s its way out of the ball (an optically thick flame glows at B, a thin one less:
    Kirchhoff's law). (All the box at the flame temperature, and the flame's absorption linear in the flame field, so
    the field's blending between cells blurs the ball's edge without dimming it.)"""
    look = _dark_look(flame_sharpness=1.0, flame_threshold=0.0)
    R = 0.6
    c = np.asarray(ORG) + N * H / 2                          # (the box's centre)
    c[1] = N * H / 2
    x = np.asarray(ORG)[None, None, None, :] + (np.stack(np.meshgrid(*(np.arange(N),) * 3, indexing='ij'), -1)[..., ::-1]
                                               + 0.5) * H      # z y x -> (x, y, z)
    inside = np.clip((R - np.linalg.norm(x - c, axis=-1)) / H + 0.5, 0.0, 1.0)
    fl = ((sigma / look.flame_absorption) ** (1.0 / look.flame_sharpness) + look.flame_threshold) / look.flame_density
    scal = np.zeros((N, N, N, 4))
    scal[..., 0] = 1.0
    scal[..., 3] = fl * inside
    lv, _, dims = _trace(engine, _vol(engine, scal), look, monkeypatch=monkeypatch)
    lut = engine.renderer._lut
    u = (look.flame_k - 400.0) / (6500.0 - 400.0) * len(lut) - 0.5
    i = int(math.floor(u))
    B = lut[i, :3] * (1 - (u - i)) + lut[i + 1, :3] * (u - i)
    # the mean of 1 - exp(-sigma s) over the ways out, from the cells round the centre
    k = np.arange(4000)
    z = 1 - (2 * k + 1) / 4000
    dirs = np.stack([np.sqrt(1 - z * z) * np.cos(2.39996 * k), z, np.sqrt(1 - z * z) * np.sin(2.39996 * k)], -1)
    for idx in [(dims[0] // 2, dims[1] // 2, dims[2] // 2), (dims[0] // 2 - 1, dims[1] // 2, dims[2] // 2)]:
        p = _centre(dims, idx) - c
        pd = dirs @ p
        s = -pd + np.sqrt(pd * pd - (p @ p - R * R))
        want = B * float(np.mean(1 - np.exp(-sigma * s)))
        got = lv[idx[2], idx[1], idx[0]]
        assert np.allclose(got, want, rtol=0.03), (sigma, idx, got, want)


def test_white_smoke_under_an_even_sky_is_lit_as_the_sky_all_through(engine, monkeypatch):
    """Smoke that scatters all it stops (albedo 1) under an even sky: the light inside is the sky's, everywhere (what
    one way loses, the others give back). Each pass is a bounce more; with enough of them Lume finds it."""
    look = _dark_look(smoke_albedo=(1.0, 1.0, 1.0))
    look.ambient = (0.3, 0.3, 0.3)
    c = np.asarray(ORG) + N * H / 2
    c[1] = N * H / 2
    x = np.asarray(ORG)[None, None, None, :] + (np.stack(np.meshgrid(*(np.arange(N),) * 3, indexing='ij'), -1)[..., ::-1]
                                               + 0.5) * H
    inside = np.clip((0.6 - np.linalg.norm(x - c, axis=-1)) / H + 0.5, 0.0, 1.0)
    scal = np.zeros((N, N, N, 4))
    scal[..., 2] = 3.0 / look.smoke_density * inside          # 3 /m: 1.8 deep from the edge to the middle
    vol = _vol(engine, scal)
    few, _, dims = _trace(engine, vol, look, passes=2, monkeypatch=monkeypatch)
    many, _, _ = _trace(engine, vol, look, passes=16, monkeypatch=monkeypatch)
    mid = tuple(d // 2 for d in dims[::-1])
    assert few[mid][0] < 0.27                                 # (two bounces are not enough deep inside)
    assert np.allclose(many[mid], 0.3, rtol=0.03), many[mid]


# Mitsuba 3.9 (volpath, 1024 spp a pixel, six 90 degree cameras at each probe) on the blob below: the mean of the light
# reaching each probe from every way but the sun's own beam (what Lume's grid holds). Classic's estimate there (its
# sky term and multiple-scattering octaves) is 2 to 5 times this.
MITSUBA_PROBES = [((0.0, 1.2, 0.0), (0.2105, 0.2175, 0.2383)), ((0.1318, 1.3928, 0.1883), (0.3888, 0.3919, 0.4102)),
                  ((-0.1318, 1.0072, -0.1883), (0.1788, 0.1921, 0.2247)), ((0.0, 1.2, 0.45), (0.4274, 0.4341, 0.461)),
                  ((0.0, 0.75, 0.0), (0.2253, 0.243, 0.2858)), ((0.0, 1.75, 0.0), (0.4498, 0.4617, 0.4998)),
                  ((0.6, 1.2, 0.0), (0.4162, 0.4321, 0.4772)), ((-0.9, 1.2, 0.0), (0.3061, 0.3336, 0.3986))]


def test_a_thick_pale_cloud_in_sun_and_sky_is_lit_as_mitsuba_lights_it(engine):
    """A lumpy blob of pale smoke (albedo 0.93, forward scattering 0.35, up to 10 /m: thick steam) under a low sun and a
    blue sky, traced at a final render's settings from nothing: within a tenth or so of Mitsuba's light at each probe."""
    n, h = 96, 0.025
    org = np.array([-n * h / 2, 0.0, -n * h / 2])
    c = np.array([0.0, 1.2, 0.0])
    x = org + (np.stack(np.meshgrid(*(np.arange(n),) * 3, indexing='ij'), -1)[..., ::-1] + 0.5) * h
    r = np.linalg.norm(x - c, axis=-1)
    lump = 1.0 + 0.35 * np.sin(7.0 * x[..., 0] + 1.3) * np.sin(6.0 * x[..., 1] + 0.4) * np.sin(8.0 * x[..., 2] + 2.1)
    sig = 10.0 * np.clip((0.75 - r) / 0.25, 0.0, 1.0) ** 1.5 * lump
    look = LookParams(ambient=(0.3, 0.33, 0.4), sun_intensity=3.0, sun_color=(1.0, 0.95, 0.88), sun_azimuth=35.0,
                      sun_elevation=40.0, smoke_albedo=(0.93,) * 3, anisotropy=0.35, smoke_density=6.0)
    scal = np.zeros((n, n, n, 4))
    scal[..., 2] = sig / look.smoke_density
    t = engine.gpu.texture3d((n, n, n), 'rgba16float', 'test-blob')
    engine.gpu.upload(t, scal.astype(np.float16))
    vol = VolumeView(scal=[t], vel=[t], dims=(n, n, n), h=h, origin=tuple(org))
    r_ = engine.renderer
    lv = LVM.LumeVolume(engine.gpu)
    fire = cam.FireXform()
    with engine.gpu.batch() as b:
        r_.light(b, vol, look, fire, 0.0)
        lv.compute(b, r_, vol, look, fire, [], None, [], (0.0, 0.0, 0.0), False, 0.0, True)
    r_.L0_march = r_.L1_march = r_.LV_lume = r_.LVD_lume = None
    grid = engine.gpu.read(lv.lv[0]).astype(np.float32)[..., :3]
    dims = np.asarray(lv.dims, float)
    luma = np.array([0.2126, 0.7152, 0.0722])
    logs = []
    for p, want in MITSUBA_PROBES:
        q = np.clip((np.asarray(p) - org) / (n * h) * dims - 0.5, 0.0, dims - 1.0)     # (trilinear, cell-centred)
        i = np.minimum(np.floor(q).astype(int), dims.astype(int) - 2)
        f = q - i
        got = sum(grid[i[2] + dz, i[1] + dy, i[0] + dx] * ((f[0] if dx else 1 - f[0]) * (f[1] if dy else 1 - f[1])
                                                            * (f[2] if dz else 1 - f[2]))
                  for dz in (0, 1) for dy in (0, 1) for dx in (0, 1))
        ratio = float(got @ luma) / float(np.asarray(want) @ luma)
        assert 0.75 < ratio < 1.3, (p, ratio)
        logs.append(math.log(ratio))
    assert math.sqrt(np.mean(np.square(logs))) < 0.13


def test_the_viewer_refines_a_still_picture_until_it_settles_and_starts_over_on_a_new_one(engine):
    look = _dark_look()
    look.ambient = (0.2, 0.3, 0.5)
    vol = _vol(engine, np.zeros((N, N, N, 4)))
    r = engine.renderer
    lv = LVM.LumeVolume(engine.gpu)
    fire = cam.FireXform()

    def view(key):
        with engine.gpu.batch() as b:
            r.light(b, vol, look, fire, 0.0)
            lv.compute(b, r, vol, look, fire, [], None, [], (0.0, 0.0, 0.0), False, 0.0, False, key=key)
        r.L0_march = r.L1_march = r.LV_lume = r.LVD_lume = None

    got = []
    for _ in range(LVM.VIEWER_SETTLE + 2):
        view('a')
        got.append(lv.pending)
    assert all(got[:LVM.VIEWER_SETTLE - 1]) and not any(got[LVM.VIEWER_SETTLE:])
    view('b')                                   # another picture: refined afresh
    assert lv.pending
    sky = engine.gpu.read(lv.lv[0]).astype(np.float32)[..., :3]
    assert np.allclose(sky, look.ambient, rtol=0.02)
