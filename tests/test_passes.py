"""The EXRs' compositing passes (render/passes.py, stage_ids.wgsl, raymarch.wgsl's vectors, io/cryptomatte.py): motion
vectors, normals and positions, per-object mattes and Cryptomatte. The GPU-free ones first."""
import json
import math
import os

import numpy as np
import pytest

from blackbody.engine import camera as cam
from blackbody.io import cryptomatte as CM
from blackbody.render import passes as P
from blackbody.render.job import (Output, PASSES, RenderJob, exr_compression, frame_path, output_notes,
                                  wants_passes)
from blackbody.scene import presets
from blackbody.scene.model import Scene


# -- Cryptomatte (GPU-free) -------------------------------------------------------------------------------------------

def test_murmur3_matches_the_reference():
    # MurmurHash3_x86_32's published values (what Cryptomatte hashes names with)
    assert CM.murmur3_32('') == 0
    assert CM.murmur3_32('', 1) == 0x514E28B7
    assert CM.murmur3_32('hello') == 0x248BFA47
    assert CM.murmur3_32('foo') == 0xF6A5C420
    assert CM.murmur3_32('The quick brown fox jumps over the lazy dog') == 0x2E4FF723


def test_cryptomatte_ids_are_plain_floats_and_the_manifest_holds_them():
    names = [f'Object {i}' for i in range(2000)] + ['Floor', 'Wood (oak)', 'é ñ']
    ids = np.array([CM.name_id(n) for n in names], np.float32)
    assert np.isfinite(ids).all()
    tiny = np.abs(ids) < np.finfo(np.float32).tiny            # (no denormals: a compositor would flush them to 0)
    assert not tiny.any()
    head = CM.header('CryptoObject', names)
    key = CM.layer_key('CryptoObject')
    assert len(key) == 7 and head[f'cryptomatte/{key}/name'] == 'CryptoObject'
    assert head[f'cryptomatte/{key}/hash'] == 'MurmurHash3_32'
    assert head[f'cryptomatte/{key}/conversion'] == 'uint32_to_float32'
    man = json.loads(head[f'cryptomatte/{key}/manifest'])
    assert set(man) == set(names)
    # the manifest's hex is the very float in the image
    for n in ('Floor', 'é ñ'):
        assert np.float32(CM.name_id(n)).view(np.uint32) == int(man[n], 16)


def test_cryptomatte_ranks_round_trip_and_merge():
    # three pixels: one thing; two things; the same material on two codes (added up), most covering first
    index = np.array([[[0, -1, -1], [1, 0, -1], [2, 2, 1]]])
    cover = np.array([[[1.0, 0.0, 0.0], [0.25, 0.75, 0.0], [0.25, 0.25, 0.5]]], np.float32)
    names = ['Crate', 'Ball', 'Glass']
    ch, attrs = CM.channels('CryptoMaterial', index, cover, names)
    assert {f'CryptoMaterial0{r}.{c}' for r in range(3) for c in 'RGBA'} == set(ch)
    assert all(v.dtype == np.float32 for v in ch.values())
    by_bits, mattes = CM.decode(ch, attrs, 'CryptoMaterial')
    assert sorted(by_bits.values()) == sorted(names)
    assert np.allclose(mattes['Crate'][0], [1.0, 0.75, 0.0])
    assert np.allclose(mattes['Ball'][0], [0.0, 0.25, 0.5])
    assert np.allclose(mattes['Glass'][0], [0.0, 0.0, 0.5])
    # most covering first: the second pixel's first rank is Crate (0.75), the third's is a tie kept in order
    assert ch['CryptoMaterial00.G'][0, 1] == pytest.approx(0.75)
    assert ch['CryptoMaterial00.R'][0, 1] == np.float32(CM.name_id('Crate'))


# -- motion and names (GPU-free) --------------------------------------------------------------------------------------

def _scene(fps=24):
    s = Scene()
    s.data['domain'].update(ground=True, open_sides=True, preroll=0.0, size_x=2.0, size_y=2.0, size_z=2.0, resolution=40)
    s.data['render']['fps'] = float(fps)
    s.data['camera'].update(distance=4.0, target_y=0.4, pitch=8.0)
    return s


def test_screen_motion_points_the_way_things_move():
    s = _scene()
    f = s.start + 5
    size = (320, 180)
    mats = P.clip_matrices(s, f, size[0] / size[1])
    dt = P.frame_seconds(s, f)
    p = np.array([[0.0, 0.5, 0.0]] * 3)
    v = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    m = P.screen_motion(p, v, mats, dt, size)
    assert m[0, 0] > 1.0 and abs(m[0, 1]) < 0.1 * m[0, 0]      # to the right: +u
    assert m[1, 1] > 1.0 and abs(m[1, 0]) < 0.1 * m[1, 1]      # up: +v (Nuke's y is up)
    assert np.allclose(m[:, 2:], -m[:, :2], rtol=0.02, atol=1e-3)   # backward: the other way (as far as perspective lets)
    assert np.allclose(m[2], 0.0)
    # the camera orbiting the fire: a still point off its axis moves on the screen; one on the axis stays put
    s.set_key(('camera', 'yaw'), s.start, 0.0)
    s.set_key(('camera', 'yaw'), s.start + 24, 24.0)
    mats = P.clip_matrices(s, s.start + 12, size[0] / size[1])
    m = P.screen_motion([[0.0, 0.5, 1.0], [0.0, 0.5, 0.0]], np.zeros((2, 3)), mats, dt, size)
    assert abs(m[0, 0]) > 1.0 and np.allclose(m[1], 0.0, atol=1e-3)
    # and at the shot's first frame the camera is still before it
    m0 = P.screen_motion([[0.0, 0.5, 1.0]], np.zeros((1, 3)), P.clip_matrices(s, s.start, size[0] / size[1]), dt, size)
    assert np.allclose(m0[0, 2:], 0.0) and abs(m0[0, 0]) > 0.01


def test_set_names_are_the_shots_things_each_once():
    s = _scene()
    s.add_collider(name='Box', shape='box', position=(0.0, 0.3, 0.0), size=(0.2, 0.2, 0.2), material='wood')
    s.add_collider(name='Box', shape='box', position=(0.6, 0.3, 0.0), size=(0.2, 0.2, 0.2), material='glass')
    s.add_collider(name='Hidden', shape='box', position=(0.6, 0.3, 0.6), size=(0.2, 0.2, 0.2), holdout=False)
    n = P.set_names(s, footage=False)
    listed = [n.names[c] for c in n.listed()]
    assert listed == ['Box', 'Box 2', 'Floor']                 # (not in the shot: not named)
    assert n.rows == {0: 1, 1: 2, 2: 3} and n.special['floor'] == len(n.names) - 1
    assert n.materials[2] == 'Glass' and n.code(('object', 1)) == 2 and n.code('debris') == 0
    # over footage the floor is the footage's: no floor matte
    assert 'Floor' not in [n.names[c] for c in P.set_names(s, footage=True).listed()]
    assert set(P.matte_layers(n).values()) == {'matte_Box', 'matte_Box_2', 'matte_Floor'}
    assert P.safe_name('3 crates.left') == 'n3_crates_left'
    # fabric (drawn in one layer with any grass) is named once, after the set's own things
    s.add_fabric()
    n = P.set_names(s, footage=False)
    assert [n.names[c] for c in n.listed()][-1] == 'Fabric' and n.special['fabric'] == len(n.names) - 1


def test_exr_outputs_want_passes_but_a_sky_has_none():
    o = Output('exr', 'x.####.exr')
    assert set(PASSES) <= set(o.layers)
    assert wants_passes(_scene(), o)
    assert not wants_passes(presets.make('cumulus_day'), o)
    assert not wants_passes(_scene(), Output('exr', 'x.####.exr', layers=('emission',)))


def test_the_lens_bends_the_vectors_as_it_bends_the_picture():
    w, h = 64, 36
    mv = np.zeros((h, w, 4), np.float32)
    mv[..., 0] = 2.0                                           # everything 2 px to the right
    front = {'share': np.ones((h, w), np.float32), 'motion': mv}
    assert P.through_lens(front, 0.0, (w, h)) is front
    bent = P.through_lens({k: v.copy() for k, v in front.items()}, 0.2, (w, h))   # pincushion: magnified out at the edges
    centre = bent['motion'][h // 2, w // 2, 0]
    edge = bent['motion'][h // 2, w - 2, 0]
    assert centre == pytest.approx(2.0, rel=0.05) and edge > centre


def test_the_render_window_puts_the_passes_in_both_exrs():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    QtWidgets = pytest.importorskip('PySide6.QtWidgets')
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from blackbody.ui.document import Document
    from blackbody.ui.export_dialog import ExportDialog

    def dialog(scene):
        d = Document()
        d.scene = scene
        return ExportDialog(d)

    dlg = dialog(presets.make('campfire'))
    dlg.exr.setChecked(True)
    dlg.comp_exr.setChecked(True)
    dlg.passes.setChecked(True)
    exrs = {o.content: o for o in dlg.outputs() if o.kind == 'exr'}
    assert set(PASSES) <= set(exrs['element'].layers) and set(exrs['composite'].layers) == set(PASSES)
    # DWAA would garble the Cryptomatte: the window says those EXRs are written with ZIP
    dlg.exr_comp.setCurrentIndex(dlg.exr_comp.findData('dwaa'))
    assert 'ZIP, not DWAA' in dlg.preview_paths.text()
    dlg.passes.setChecked(False)
    assert not any(set(PASSES) & set(o.layers) for o in dlg.outputs() if o.kind == 'exr')
    assert 'ZIP, not DWAA' not in dlg.preview_paths.text()
    sky = dialog(presets.make('cumulus_day'))
    assert not sky.passes.isEnabled()


def _fake_passes(scene, w=64, h=48):
    """A frame's passes as frame_passes makes them, of a crate, a ball and the floor side by side (the ball's pixels
    shared with the floor), with N, P and vectors all different."""
    names = P.set_names(scene)
    crate, ball, floor = names.rows[0], names.rows[1], names.special['floor']
    codes, cover = np.zeros((h, w, 6), np.int32), np.zeros((h, w, 6), np.float32)
    a, b = w // 3, 2 * w // 3
    codes[:, :a, 0], codes[:, a:b, 0], codes[:, b:, 0] = crate, ball, floor
    cover[..., 0] = 1.0
    codes[:, a:b, 1] = floor
    cover[:, a:b, 0], cover[:, a:b, 1] = 0.7, 0.3
    rng = np.random.default_rng(3)
    n = rng.normal(size=(h, w, 3))
    return names, {'codes': codes, 'cover': cover, 'N': (n / np.linalg.norm(n, axis=-1, keepdims=True)).astype(np.float32),
                   'P': rng.uniform(-5.0, 5.0, (h, w, 3)).astype(np.float32), 'seen': np.ones((h, w), np.float32),
                   'motion': rng.normal(0.0, 3.0, (h, w, 4)).astype(np.float32)}


def test_a_lossy_exr_keeps_its_cryptomatte_whole(tmp_path):
    """DWAA's lossy maths works on every channel named R, G, B or Y, 32-bit float ones too, and would garble the
    Cryptomatte ids: an EXR with the passes is written with ZIP instead (the render's notes say so), its ids exactly as
    made; without the passes DWAA stays."""
    import OpenEXR
    s = _scene()
    s.add_collider(name='Crate', shape='box', position=(-0.6, 0.3, 0.0), size=(0.25, 0.25, 0.25), material='wood')
    s.add_collider(name='Ball', shape='sphere', position=(0.6, 0.25, 0.4), size=(0.2, 0.2, 0.2), material='steel')
    names, passes = _fake_passes(s)
    h, w = passes['seen'].shape
    o = Output('exr', str(tmp_path / 'c.####.exr'), 'composite', layers=PASSES, compression='dwaa')
    assert exr_compression(s, o) == 'zip'
    assert any('ZIP, not DWAA' in n for n in output_notes(s, [o]))
    plain = Output('exr', str(tmp_path / 'p.####.exr'), 'element', layers=('emission',), compression='dwab')
    assert exr_compression(s, plain) == 'dwab' and not output_notes(s, [plain])
    job = RenderJob(s, [o], None, frames=(s.start, s.start), size=(w, h))
    job.names = names
    comp = np.random.default_rng(4).uniform(0.0, 4.0, (h, w, 3)).astype(np.float32)
    job._write_comp(o, s.start, comp, {}, None, None, passes)
    path = frame_path(o.path, s.start)
    with OpenEXR.File(path) as fh:
        assert fh.header()['compression'] == OpenEXR.ZIP_COMPRESSION
    ch, attrs = _channels(path)
    want = P.channels(passes, names, PASSES, set())[0]
    for k in [k for k in want if k.startswith('Crypto')]:
        assert np.array_equal(ch[k].view(np.uint32), want[k].view(np.uint32)), k   # (bit for bit)
    assert np.array_equal(ch['P.R'], want['P.R'].astype(np.float16))
    mattes = CM.decode(ch, attrs, 'CryptoObject')[1]
    assert set(mattes) == {'Crate', 'Ball', 'Floor'}
    assert np.allclose(mattes['Ball'][:, w // 2], 0.7) and np.allclose(mattes['Floor'][:, w // 2], 0.3)


def test_a_rope_is_named_apart_from_what_it_hangs_on():
    """A lamp hanging on a chain: the chain has its own name and code (its links are owned by it), so picking the lamp
    in Cryptomatte does not pick its chain."""
    from blackbody.engine.ropes import CHAIN
    from blackbody.engine.stage import Stage
    s = _scene()
    ci = s.add_collider(name='Lamp', shape='sphere', position=(0.0, 1.0, 0.0), size=(0.2, 0.2, 0.2), material='glass',
                        joint='rope', rope_look='chain', joint_anchor=(0.0, 2.0, 0.0))
    s.add_collider(name='Post', shape='box', position=(1.0, 0.5, 0.0), size=(0.1, 1.0, 0.1), joint='spring')
    n = P.set_names(s)
    listed = [n.names[c] for c in n.listed()]
    assert {'Lamp', 'Lamp chain', 'Post', 'Post spring'} <= set(listed)
    rc = n.code(('rope', ci))
    assert rc and rc != n.code(('object', ci)) and n.materials[rc] == 'Steel'
    assert n.names[n.code(('rope', ci + 1))] == 'Post spring'
    rope = dict(a=np.array([0.0, 2.0, 0.0]), b=np.array([0.0, 1.1, 0.0]), va=np.zeros(3), vb=np.array([0.5, 0.0, 0.0]),
                length=np.float32(1.0), radius=np.float32(0.01), look=np.float32(CHAIN), broken=np.float32(0.0))
    owners = []
    assert Stage.piece_arrays(s, None, ropes={ci: rope}, owners=owners) is not None
    assert owners and set(owners) == {('rope', ci)}


# -- on the GPU -------------------------------------------------------------------------------------------------------

def _channels(path):
    import OpenEXR
    with OpenEXR.File(path) as fh:
        attrs = dict(fh.header())
        out = {}
        for k, v in fh.channels().items():
            px = np.asarray(v.pixels)
            if px.ndim == 3:
                letters = 'RGBA' if px.shape[2] == 4 else 'RGB'
                for j in range(px.shape[2]):
                    out[f'{k}.{letters[j]}' if k not in ('RGBA', 'RGB') else letters[j]] = px[..., j]
            else:
                out[k] = px
    return out, attrs


def test_exr_passes_name_the_objects_and_move_with_them(engine, tmp_path):
    """A crate sliding to the right past a still ball, in a fire: the element and composite EXRs have every pass layer;
    Cryptomatte names them (its manifest decodes to the mattes); the crate's vectors point right, the ball's are still,
    and the fire's rise."""
    s = _scene()
    i = s.add_collider(name='Crate', shape='box', position=(-0.6, 0.3, 0.0), size=(0.25, 0.25, 0.25), material='wood')
    s.set_key(('collider', i, 'position'), s.start, (-0.6, 0.3, 0.0))
    s.set_key(('collider', i, 'position'), s.start + 24, (0.6, 0.3, 0.0))
    s.add_collider(name='Ball', shape='sphere', position=(0.6, 0.25, 0.4), size=(0.2, 0.2, 0.2), material='steel')
    f = s.start + 6
    engine.invalidate()
    paths = RenderJob(s, [Output('exr', str(tmp_path / 'e.####.exr'), 'element'),
                          Output('exr', str(tmp_path / 'c.####.exr'), 'composite')], engine, frames=(f, f), final=False,
                      size=(192, 108), samples=1).run()
    for p in (frame_path(str(tmp_path / 'e.####.exr'), f), frame_path(str(tmp_path / 'c.####.exr'), f)):
        assert p in paths
        ch, attrs = _channels(p)
        want = {'forward.u', 'forward.v', 'backward.u', 'backward.v', 'N.R', 'N.G', 'N.B', 'P.R', 'P.G', 'P.B',
                'matte_Crate.Y', 'matte_Ball.Y', 'matte_Floor.Y'}
        want |= {f'Crypto{k}0{r}.{c}' for k in ('Object', 'Material') for r in range(3) for c in 'RGBA'}
        assert want <= set(ch), sorted(want - set(ch))
        assert ch['CryptoObject00.R'].dtype == np.float32 and ch['P.R'].dtype == np.float16   # (ids whole; P as the EXR)
        _bits, mattes = CM.decode(ch, attrs, 'CryptoObject')
        assert set(mattes) == {'Crate', 'Ball', 'Floor'}
        assert np.allclose(mattes['Crate'], ch['matte_Crate.Y'].astype(np.float32), atol=2e-3)
        assert set(CM.decode(ch, attrs, 'CryptoMaterial')[1]) == {'Wood (pine)', 'Steel', 'Concrete'}
        crate, ball = mattes['Crate'] > 0.99, mattes['Ball'] > 0.99
        assert crate.sum() > 100 and ball.sum() > 50
        fu, fv = ch['forward.u'].astype(np.float32), ch['forward.v'].astype(np.float32)
        assert fu[crate].mean() > 1.0 and abs(fv[crate].mean()) < 0.2 * fu[crate].mean()
        assert np.abs(ch['backward.u'].astype(np.float32)[crate].mean() + fu[crate].mean()) < 0.1
        assert np.median(np.abs(fu[ball])) < 1e-3 and np.median(np.abs(fv[ball])) < 1e-3   # (but where fire is in front)
        floor = (mattes['Floor'] > 0.99) & (ch['N.G'].astype(np.float32) > 0.0)
        assert np.allclose(ch['N.G'].astype(np.float32)[floor], 1.0, atol=1e-2)
        assert ch['P.G'][crate].max() == pytest.approx(0.55, abs=0.02)   # (the crate's top)
        if 'A' in ch:
            fire = (ch['A'].astype(np.float32) > 0.3) & ~(mattes['Crate'] > 0.0)
            assert fire.sum() > 20 and fv[fire].mean() > 0.2, 'the fire rises'


def test_pieces_are_named_as_their_object(engine, monkeypatch):
    """A broken box's pieces, falling: their pixels are named as the box (a piece is its object in the mattes), and
    their vectors point down."""
    from blackbody.engine.solids import fractured
    s = _scene()
    ci = s.add_collider(name='Vase', shape='box', position=(0.0, 0.4, 0.0), size=(0.25, 0.25, 0.25), material='brick',
                        breakable=True, holdout=False)          # (the whole of it not drawn: only its pieces)
    engine.invalidate()
    engine.prepare(s, final=False)
    engine.simulate_to(s, s.start, cache=False)
    size = np.array(s.colliders[ci]['size'], float)
    frac = fractured(s.colliders[ci], size, 0.0, None, s)
    n = len(frac.pieces)
    assert n > 3
    pos = np.array([pc.centroid for pc in frac.pieces]) + (0.0, 0.4, 0.0)
    poses = {ci: dict(pos=pos, quat=np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)), vel=np.tile([0.0, -2.0, 0.0], (n, 1)),
                      omega=np.zeros((n, 3)), size=size, hollow=np.float32(0.0))}
    monkeypatch.setattr(engine, 'piece_poses', lambda frame: poses)
    names = P.set_names(s)
    assert 'Vase' in names.names
    out = P.trace_set(engine, s, s.start, (160, 90), names)
    vase = np.where(out['codes'] == names.rows[ci], out['cover'], 0.0).sum(-1)
    assert len(engine.stage.owners) == n and all(o == ('object', ci) for o in engine.stage.owners)
    on = vase > 0.99
    assert on.sum() > 50
    assert out['motion'][on, 1].mean() < -0.5 and out['motion'][on, 3].mean() > 0.5   # down, and up from the last
    assert out['over'] == 0


def test_a_liquid_scene_names_its_liquid_and_moves_with_it(engine, tmp_path):
    sc = presets.make('sand_castle')
    sc.data['domain'].update(resolution=32)
    f = sc.start + 8
    engine.invalidate()
    paths = RenderJob(sc, [Output('exr', str(tmp_path / 'l.####.exr'), 'element')], engine, frames=(f, f), final=False,
                      size=(160, 90), samples=1).run()
    ch, attrs = _channels(paths[0])
    assert {'matte_Liquid.Y', 'forward.u', 'N.R'} <= set(ch)
    mattes = CM.decode(ch, attrs, 'CryptoObject')[1]
    water = ch['water.Y'].astype(np.float32) > 0.99
    assert water.sum() > 50
    assert (mattes['Liquid'][water] > 0.98).all()
    speed = np.hypot(ch['forward.u'].astype(np.float32), ch['forward.v'].astype(np.float32))[water]
    assert speed.mean() > 0.5, 'the water moves'


def test_the_fires_vectors_gather_every_anti_aliasing_pass(engine):
    """With several anti-aliasing passes the fire's vectors are every pass's, each weighed by the fire's share of the
    pixel in it, and that share (what they are laid over the set by) is the passes' together, as the beauty's is."""
    s = _scene()
    f = s.start + 6
    size = (160, 90)
    engine.invalidate()
    engine.prepare(s, final=False)
    engine.simulate_to(s, f, cache=False)
    r = engine.renderer
    r.motion = P.march_motion(s, f, size)
    try:
        engine.render(s, f, size, mode='fire', samples=4)
        vec, share = r.read_vectors()
        beauty = engine.gpu.read(r.beauty).astype(np.float32)
    finally:
        r.motion = None
    full = np.clip(beauty[..., 3] + beauty[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32), 0.0, 1.0)
    assert r._vec_passes == 4
    # (each pass's share clamped on its own: at most the beauty's, and close to it)
    assert (share <= full + 0.01).all() and np.abs(share - full).mean() < 0.01
    fire = share > 0.3
    assert fire.sum() > 20 and vec[fire, 1].mean() > 0.2, 'the fire rises'


def test_the_liquid_is_sampled_where_it_is_seen(engine, tmp_path, monkeypatch):
    """A liquid's vectors and normals come from its surface grid sampled on the GPU at the pixels that see it (the grid
    is never read back whole: at final quality it is hundreds of megabytes), as the grid filtered there says."""
    sc = presets.make('sand_castle')
    sc.data['domain'].update(resolution=32)
    f = sc.start + 8
    engine.invalidate()
    read, whole = engine.gpu.read, []

    def watched(tex, *a, **k):
        if str(getattr(tex, 'label', '')).startswith('liq-surface'):
            whole.append(tex.label)
        return read(tex, *a, **k)

    monkeypatch.setattr(engine.gpu, 'read', watched)
    RenderJob(sc, [Output('exr', str(tmp_path / 'l.####.exr'), 'element')], engine, frames=(f, f), final=False,
              size=(160, 90), samples=1).run()
    assert not whole
    monkeypatch.setattr(engine.gpu, 'read', read)
    lr = engine.liquid_r
    grid = read(lr.surf).astype(np.float64)                     # (z, y, x, 4)
    nz, ny, nx = grid.shape[:3]
    rng = np.random.default_rng(5)
    q = rng.uniform([1.0, 1.0, 1.0], [nx - 1.0, ny - 1.0, nz - 1.0], (500, 3))
    vel, grad = P.surface_at(engine.gpu, lr.surf, lr.nf, q)

    def trilinear(p, c):
        # (cell centres at i + 0.5)
        i = np.clip(p - 0.5, 0.0, [nx - 1, ny - 1, nz - 1])
        i0 = np.minimum(np.floor(i).astype(int), [nx - 2, ny - 2, nz - 2])
        t = i - i0
        out = 0.0
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    wgt = ((t[:, 0] if dx else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1])
                           * (t[:, 2] if dz else 1 - t[:, 2]))
                    out = out + grid[i0[:, 2] + dz, i0[:, 1] + dy, i0[:, 0] + dx, c] * wgt[:, None]
        return out

    want = trilinear(q, slice(1, 4))
    assert np.abs(vel - want).max() <= 0.01 * max(np.abs(want).max(), 1e-3) + 1e-4
    e = np.eye(3) * 0.75
    gw = np.stack([trilinear(q + e[k], slice(0, 1))[:, 0] - trilinear(q - e[k], slice(0, 1))[:, 0] for k in range(3)], 1)
    assert np.abs(grad - gw).max() <= 0.01 * max(np.abs(gw).max(), 1e-3) + 1e-4


def test_the_footage_holdouts_are_where_the_plate_is(engine):
    """Over footage of another shape than the picture (a wide plate, its middle shown), the footage's matte holds the
    set out where the composite has it: looked up in the plate, not at the picture's own place."""
    s = _scene()
    s.add_collider(name='Wall', shape='box', position=(0.0, 1.0, -1.5), size=(30.0, 30.0, 0.2), material='concrete',
                   look='cg')
    f = s.start
    engine.invalidate()
    engine.prepare(s, final=False)
    engine.simulate_to(s, f, cache=False)
    size = (160, 90)
    fa, oa = 4.0, size[0] / size[1]
    fit = (oa / fa, 1.0)                                          # (RenderJob._plate_fit: the plate is wider)
    matte = np.zeros((100, 400), np.float32)
    matte[:, :160] = 1.0                                          # (the plate's left 40 %)
    names = P.set_names(s, footage=True)
    r = engine.renderer
    r.set_holdout(matte, None)
    try:
        out = P.trace_set(engine, s, f, size, names, footage=True, plate_fit=fit)
    finally:
        r.set_holdout(None, None)
    wall = np.where(out['codes'] == names.rows[0], out['cover'], 0.0).sum(-1).mean(0)   # (per column)
    edge = (0.5 - 0.1 / fit[0]) * size[0]                         # (the plate's 40 % in the picture: 27.5 %)
    assert wall[int(edge) + 3:].min() > 0.95, 'beside the matte: the wall'
    assert wall[:int(edge) - 3].max() < 0.05, 'under the matte: held out'
