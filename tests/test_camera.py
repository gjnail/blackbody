import numpy as np
import pytest

from blackbody.engine import camera as cam


def test_anchor_places_fire_base():
    spec = cam.CameraSpec(anchor=(0.3, 0.7), scale=1.4, roll=5.0, yaw=20, pitch=10, distance=8)
    fire = cam.FireXform(position=(1.0, 0.0, -2.0), yaw=30)
    cs = cam.compute(spec, 16 / 9, fire)
    px, ok = cam.project(cs, [fire.position], 1920, 1080)
    assert ok[0]
    assert px[0] == pytest.approx((0.3 * 1920, 0.7 * 1080), abs=1e-3)


def test_pixel_ray_round_trip():
    spec = cam.CameraSpec(yaw=-35, pitch=12, distance=5, focal_mm=50, anchor=(0.6, 0.8))
    cs = cam.compute(spec, 1.5, cam.FireXform())
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.uniform(-1, 1, 3) + np.array([0, 1, 0])
        px, ok = cam.project(cs, [p], 1500, 1000)
        o, d = cam.pixel_ray(cs, px[0][0], px[0][1], 1500, 1000)
        # the point lies on the ray
        v = p - o
        assert np.linalg.norm(np.cross(v, d)) < 1e-6 * max(1.0, np.linalg.norm(v))


def test_world_to_grid():
    fire = cam.FireXform(position=(2.0, 0.0, 1.0), yaw=90)
    m = cam.world_to_grid(fire, (-1.0, 0.0, -1.0), 0.1)
    # the fire base (world 2,0,1) is at the grid's centre in x/z, bottom in y
    g = m @ np.array([2.0, 0.0, 1.0, 1.0])
    assert g[:3] == pytest.approx((10.0, 0.0, 10.0))


def test_focal_length_sets_field_of_view():
    wide = cam.hfov_of(cam.CameraSpec(focal_mm=18))
    tele = cam.hfov_of(cam.CameraSpec(focal_mm=85))
    assert wide > tele
    assert np.degrees(cam.hfov_of(cam.CameraSpec(focal_mm=36, sensor_mm=36))) == pytest.approx(53.13, abs=0.01)
