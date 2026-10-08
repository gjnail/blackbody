"""GPU test: the grain motion blur leaves in a final fire render has no pattern in it (raymarch.wgsl pixel_hash)."""
import numpy as np

from blackbody.scene import presets

LUMA = np.array([0.2126, 0.7152, 0.0722])


def _peak(img):
    """A picture's strongest frequency past the lowest ones, over the median power there, and where it is (cycles per
    pixel, x and y). Grain with no pattern stays near the floor of a white spectrum's largest bin (some 13 here)."""
    r = img - img.mean()
    w = np.hanning(r.shape[0])[:, None] * np.hanning(r.shape[1])[None, :]
    P = np.abs(np.fft.fftshift(np.fft.fft2(r * w))) ** 2
    f = (np.arange(r.shape[0]) - r.shape[0] // 2) / r.shape[0]
    band = np.hypot(f[:, None], f[None, :]) > 0.15
    k = np.unravel_index(np.argmax(np.where(band, P, 0.0)), P.shape)
    return float(P[k] / np.median(P[band])), (float(f[k[1]]), float(f[k[0]]))


def test_motion_blur_grain_has_no_hatching(engine):
    # A campfire seen close, so it fills the picture. With the shutter offset from interleaved gradient noise, the
    # passes left a diagonal hatching about 3 px apart at (0.22, 0.24) cycles per pixel: 100 times the floor with 4
    # passes, 250 with 8. With a hash the grain is white (under 20) and as strong as before.
    sc = presets.make('campfire')
    sc.data['domain']['resolution'] = 48
    sc.data['camera'].update(distance=1.6, target_y=0.6)
    engine.invalidate()
    engine.prepare(sc, final=False)
    f = sc.start + 24
    engine.simulate_to(sc, f, cache=False)

    def luma(samples):
        engine.render(sc, f, (160, 160), mode='fire', final=True, samples=samples, motion_blur=True)
        return engine.aovs()['beauty'].astype(np.float64)[..., :3] @ LUMA

    ref = luma(96)
    assert ref.mean() > 0.05, 'the fire should fill much of the picture'
    for n in (4, 8):
        grain = luma(n) - ref
        peak, where = _peak(grain)
        assert peak < 40.0, f'{n} passes: a pattern in the grain at {where} cycles per pixel ({peak:.0f} x the median)'
        assert np.sqrt((grain ** 2).mean()) < 0.6 / np.sqrt(n) * ref.mean()
