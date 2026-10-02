"""The stage (engine/stage.py): the set drawn in CG for shots without footage, and CG objects over footage."""
import numpy as np

from blackbody.scene.model import Scene


def scene_of(*cols, fps=24):
    s = Scene()
    s.data['domain'].update(ground=True, open_sides=True, preroll=0.0, size_x=2.0, size_y=2.0, size_z=2.0, resolution=48)
    s.data['render']['fps'] = float(fps)
    s.data['camera'].update(distance=4.0, target_y=0.4, pitch=8.0)
    for c in cols:
        s.add_collider(**c)
    return s


def test_the_stage_draws_cg_objects_in_their_material(engine):
    """Without footage the floor, the sky and a CG box are drawn; the box in its wood's colour, lit."""
    s = scene_of(dict(name='Box', shape='box', position=(0.0, 0.3, 0.0), size=(0.3, 0.3, 0.3), material='wood'))
    s.data['lighting'].update(sun_on=True, sun_intensity=3.0, ambient_intensity=3.0)
    engine.invalidate()
    engine.prepare(s, final=False)
    engine.simulate_to(s, s.start, cache=False)
    engine.render(s, s.start, (320, 180))
    st = engine.gpu.read(engine.stage.tex).astype(np.float32)
    assert np.all(st[..., 3] > 0.99)                 # all CG
    box = st[85:100, 150:170, :3].mean(axis=(0, 1))
    floor = st[170:178, 10:310, :3].mean(axis=(0, 1))
    sky = st[2:10, 10:310, :3].mean(axis=(0, 1))
    assert box[0] > box[1] > box[2] > 0.0, box       # wood: warm
    assert floor.min() > 0.0 and sky.min() > 0.0
    assert sky[2] > sky[0]                           # the default ambient is a little blue


def test_cg_objects_go_over_the_footage_with_their_shadows(engine):
    """With footage a falling (so CG) box is drawn over it and darkens the footage's ground where its shadow falls;
    the footage elsewhere is left as it was."""
    s = scene_of(dict(name='Box', shape='box', position=(0.0, 0.3, 0.0), size=(0.3, 0.3, 0.3), material='wood', dynamic=True))
    s.data['lighting'].update(sun_on=True, sun_intensity=3.0, sun_elevation=30.0, sun_azimuth=90.0,
                              ambient_from_footage=False)
    plate = np.full((180, 320, 4), 128, np.uint8)
    engine.invalidate()
    engine.prepare(s, final=False)
    engine.simulate_to(s, s.start, cache=False)
    engine.render(s, s.start, (320, 180), plate=plate)
    st = engine.gpu.read(engine.stage.tex).astype(np.float32)
    cg = st[..., 3] > 0.5
    assert 0.02 < cg.mean() < 0.4
    grey = (128 / 255 + 0.055) / 1.055
    grey **= 2.4
    away = st[2:10, 2:40, :3]                         # high in the sky of the footage: untouched
    assert np.allclose(away, grey, atol=0.01)
    ground = st[120:178, :, :3][~cg[120:178]]
    assert ground.min() < 0.8 * grey                  # its shadow on the footage's ground


def test_a_rope_is_drawn_from_its_anchor_down_to_what_hangs_on_it(engine):
    """A steel cable from a fixed point down to a ball: drawn as a thin upright line over the stage, and held out of
    the fire and the liquid behind it (the stage's hold: its distance from the camera)."""
    s = scene_of(dict(name='Ball', shape='sphere', position=(0.0, 0.3, 0.0), size=(0.12, 0.12, 0.12), material='steel',
                      joint='rope', joint_anchor=(0.0, 1.6, 0.0), rope_look='cable', rope_thickness=0.04))
    engine.invalidate()
    engine.prepare(s, final=False)
    engine.simulate_to(s, s.start, cache=False)
    engine.render(s, s.start, (320, 180))
    hold = engine.gpu.read(engine.stage.hold).astype(np.float32)[..., 1]
    ys, xs = np.nonzero(hold > 0.0)
    assert len(ys) > 30
    assert np.ptp(ys) > 40 and np.ptp(xs) < 12          # a thin upright line
    assert abs(float(np.median(hold[ys, xs])) - 4.0) < 0.6   # about as far off as the ball (4 m)


def test_lightning_is_drawn_glowing_down_its_channel(engine):
    """A bolt from high above down to the ground, at its first flash: a bright line down the picture with a glow round it,
    the set lit, and nothing drawn before it strikes."""
    s = scene_of()
    s.data['lighting'].update(sun_on=False, ambient_intensity=0.2)
    s.add_light(kind='lightning', position=(0.0, 1.9, 0.0), end=(0.1, 0.0, 0.0), intensity=2e5, strike_at=0.2, branching=0.0,
                thickness=0.03)
    at = s.start + int(round(0.2 * s.fps))
    engine.invalidate()
    engine.prepare(s, final=False)
    engine.simulate_to(s, at, cache=False)
    engine.render(s, at, (320, 180))
    st = engine.gpu.read(engine.stage.tex).astype(np.float32)[..., :3].max(-1)
    col = st[20:160].max(0)                                      # the brightest down each column
    assert col.max() > 5.0                                       # its core: far brighter than anything lit
    assert np.sum(col > 1.0) < 80                                # a line, wandering as lightning does (its glow dimmer)
    glow = st[90, int(np.argmax(col)) + 15]
    assert 0.02 < glow < col.max()
    engine.simulate_to(s, at - 3, cache=False)
    engine.render(s, at - 3, (320, 180))
    before = engine.gpu.read(engine.stage.tex).astype(np.float32)[..., :3].max()
    assert before < 2.0
