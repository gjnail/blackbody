"""Realism checks: the footage's lens on the rendered element (depth of field, softness, distortion,
colour fringing, halation), heat haze traced through the simulated heat, camera colour, real-world
brightness, resolution from the shot, and puffing at the rate real fires puff."""
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from blackbody.engine.lut import blackbody_lut, flame_ev
from blackbody.scene import presets
from blackbody.scene.model import base_width, puff_wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))


# -- no GPU ----------------------------------------------------------------------------------------------

def test_camera_colour_is_a_touch_more_yellow_and_in_gamut():
    eye, camera = blackbody_lut(), blackbody_lut(response='camera')
    i = int((1650 - 400) / (6500 - 400) * (len(eye) - 1))
    assert camera[i, 1] / camera[i, 0] > eye[i, 1] / eye[i, 0], 'flame reads a little more yellow on camera'
    assert (camera[:, :3] >= 0).all()
    assert np.array_equal(camera[:, 3], eye[:, 3]), 'brightness stays as the eye sees it'


def test_real_world_brightness_follows_temperature_and_scene_light():
    assert flame_ev(1300) == pytest.approx(12.3, abs=0.2)   # the soot of a wood fire
    assert flame_ev(1650) == pytest.approx(18.1, abs=0.2)
    sc = presets.make('campfire')
    assert sc.physical_ev(sc.start) == 0.0, 'off unless asked for'
    sc.data['shading']['exposure_physical'] = True
    dusk = sc.physical_ev(sc.start)
    sc.data['shading']['scene_ev'] = 15.0
    assert sc.physical_ev(sc.start) == pytest.approx(dusk - 6.0), 'a brighter scene makes the fire relatively dimmer'
    assert sc.look(sc.start).exposure == pytest.approx(sc.data['shading']['exposure'] + dusk - 6.0)


def test_resolution_from_the_shot_follows_how_big_the_fire_is():
    sc = presets.make('campfire')
    sc.data['render'].update(width=1280, height=720)
    sc.data['domain']['resolution'] = 64
    sc.data['camera']['distance'] = 40.0
    far = sc.auto_detail()
    sc.data['camera']['distance'] = 8.0
    near = sc.auto_detail()
    assert near[0] * near[1] > far[0] * far[1], 'a fire that fills more of the frame gets finer cells'
    sc.data['domain']['resolution'] = 300
    assert sc.auto_detail()[0] >= 300, 'never lower than set by hand'
    sc.data['render']['auto_resolution'] = False
    assert sc.auto_detail() is None
    assert sc.upres_for(True) == sc.data['render']['upres']


def test_puffing_pulses_the_base_at_the_rate_real_fires_puff():
    sc = presets.make('campfire')
    e = sc.emitters[2]                                          # the coal bed, 0.68 m across
    width = base_width(e, e['position'], e['size'], e['end'])
    assert width == pytest.approx(0.68, abs=0.01)
    fps = 200.0
    wave = np.array([puff_wave(i / fps, width, 1.0)[0] for i in range(int(20 * fps))])
    spec = np.abs(np.fft.rfft(wave - wave.mean()))
    f = np.fft.rfftfreq(len(wave), 1.0 / fps)[np.argmax(spec)]
    assert f == pytest.approx(1.5 / math.sqrt(width), rel=0.1)
    # and it reaches the solver as a surging upflow over the base
    sc.data['motion']['puffing'] = 0.6
    ups = [sc.emitters_gpu(sc.start + i / 4.0)[2].vel[1] for i in range(40)]
    assert max(ups) > 1.0 and min(ups) < 0.2


# -- on the GPU ------------------------------------------------------------------------------------------

def _campfire(engine, res=48):
    sc = presets.make('campfire')
    sc.data['domain'].update(resolution=res, time_scale=1.0)
    sc.data['motion']['puffing'] = 0.0
    sc.data['embers']['enabled'] = False
    sc.data['composite']['bloom'] = 0.0
    engine.invalidate()
    engine.prepare(sc)
    f = sc.start + 30
    engine.simulate_to(sc, f, cache=True)
    return sc, f


def _render(engine, sc, f, size=(256, 256), mode='fire', plate=None, **comp):
    base = sc.comp
    sc.comp = lambda fr, m='composite': dataclasses.replace(base(fr, m), **comp)
    try:
        engine.render(sc, f, size, mode=mode, plate=plate)
    finally:
        sc.comp = base
    return engine.renderer.read_linear().astype(np.float32)


def test_depth_of_field_blurs_the_fire_out_of_focus(engine):
    sc, f = _campfire(engine)
    sharp = _render(engine, sc, f)[..., 3]
    blur = _render(engine, sc, f, f_stop=1.4, focus_distance=1.0)[..., 3]
    soft = _render(engine, sc, f, softness=3.0)[..., 3]
    detail = lambda a: float((np.diff(a, axis=0) ** 2).sum() + (np.diff(a, axis=1) ** 2).sum())
    assert detail(blur) < 0.6 * detail(sharp), 'out of focus, the fire loses its fine detail'
    assert detail(soft) < 0.8 * detail(sharp)
    assert blur.sum() == pytest.approx(sharp.sum(), rel=0.05), 'blur spreads the fire, it does not remove it'
    focused = _render(engine, sc, f, f_stop=1.4)[..., 3]      # focused on the fire itself
    assert detail(focused) > detail(blur)


def test_lens_distortion_pulls_the_fire_toward_the_centre(engine):
    sc, f = _campfire(engine)
    sc.data['camera'].update(anchor_x=0.25, anchor_y=0.8)
    centroid = lambda a: np.array(np.nonzero(a > 0.2)).mean(axis=1)
    plain = _render(engine, sc, f)[..., 3]
    barrel = _render(engine, sc, f, lens_k1=-0.2)[..., 3]
    mid = np.array(plain.shape) / 2.0
    assert np.linalg.norm(centroid(barrel) - mid) < np.linalg.norm(centroid(plain) - mid) - 2.0


def test_fringing_and_halation_colour_the_edges(engine):
    sc, f = _campfire(engine)
    plain = _render(engine, sc, f, mode='composite')
    halo = _render(engine, sc, f, mode='composite', halation=2.0) - plain
    added = halo[..., :3].reshape(-1, 3).sum(axis=0)
    assert added[0] > 0.0 and added[0] > 2.0 * added[1], 'halation adds red light'
    fringe = _render(engine, sc, f, mode='composite', fringing=4.0)
    d = np.abs(fringe - plain)[..., :3]
    assert d[..., 0].sum() > 0.0 and d[..., 1].sum() < 0.05 * d[..., 0].sum(), 'only red and blue move'


def test_heat_haze_bends_the_footage_only_near_the_fire(engine):
    sc, f = _campfire(engine)
    yy, xx = np.mgrid[0:256, 0:256]
    stripes = ((np.sin(xx * 0.9) + np.sin(yy * 0.7)) * 60 + 128).astype(np.uint8)
    plate = np.dstack([stripes, stripes, stripes, np.full_like(stripes, 255)])
    off = _render(engine, sc, f, mode='composite', plate=plate, haze=0.0)
    on = _render(engine, sc, f, mode='composite', plate=plate, haze=3.0)
    d = np.abs(on - off)[..., :3].max(axis=-1)
    alpha = engine.aovs()['beauty'][..., 3].astype(np.float32)
    assert d.max() > 0.01, 'the hot air bends the footage behind it'
    assert d[:20, :20].max() == 0.0 and d[:20, -20:].max() == 0.0, 'but not far from the fire'
    assert d[alpha < 0.01].max() > 0.0, 'including where there is no smoke or flame, only hot air'


def test_substep_cap_follows_time_scale_and_grid():
    sc = presets.make('campfire')
    sc.data['domain'].update(substeps_max=6, time_scale=1.0, resolution=144)
    h_hand = max(sc.domain_size()) / 144
    assert sc.substep_cap(sc.start, h_hand) == 6
    sc.data['domain']['time_scale'] = 1.6
    assert sc.substep_cap(sc.start, h_hand) == 10, 'each frame covers 1.6x the time'
    assert sc.substep_cap(sc.start, h_hand / 2) == 20, 'and a grid twice as fine needs twice the substeps'


def test_retuned_campfire_stays_stable_on_a_fine_grid(engine):
    """Time scale 1.6 with puffing on the grid Resolution from the shot picks: the gas must not run away."""
    sc = presets.make('campfire')
    sc.data['domain'].update(resolution=176, preview_scale=1.0)
    sc.data['embers']['enabled'] = False
    engine.invalidate()
    engine.prepare(sc)
    fastest = 0.0
    for f in range(sc.start, sc.start + 120):
        engine.simulate_to(sc, f, cache=False)
        fastest = max(fastest, engine.solver.max_speed)
    assert fastest < 15.0


def test_the_campfire_puffs_at_the_rate_real_fires_do(engine):
    from fire_check import measure, puff_law
    sc = presets.make('campfire')     # at its own viewer resolution: much coarser grids cannot hold the puffs
    sc.data['embers']['enabled'] = False
    m = measure(engine, sc, seconds=8.0)
    assert m['puff'] / puff_law(m['D']) == pytest.approx(1.0, abs=0.45)
