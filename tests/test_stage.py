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
