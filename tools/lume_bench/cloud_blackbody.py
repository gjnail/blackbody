"""Lume's clouds against Mitsuba, the Blackbody side (README: Clouds). Run with Blackbody's own environment:

    python tools/lume_bench/cloud_blackbody.py OUT.npz [frame] [width height] [lit|back|below]

One cumulus of the cumulus_day preset alone (its condensate kept in a box round it and the rest of the sky's taken away,
in the fields themselves, so both renderers see the same cloud), its sub-grid detail and the haze off, seen from its sunlit
side at its own height (lit), from behind it (back) or from the ground under it (below, with the ground). Writes the
cloud renderer's pictures, classic (img) and with Lume (img_lume), and what cloud_mitsuba.py needs to path trace it: the
extinction grid as the march draws it (cropped to the box: Mitsuba's tracking crosses only that), the camera, the sun and
the sky (an envmap of the renderer's sky in Mitsuba's convention)."""
import math
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402

from blackbody.engine import camera as cam  # noqa: E402
from blackbody.engine.engine import Engine  # noqa: E402
from blackbody.scene import presets  # noqa: E402

DENSE = 0.05     # (cloud_march.wgsl: the densest a cloud is drawn)


def biggest_cloud(cond, thr=2e-5):
    """The cells of the cloud holding the most condensate (6-connected), and the labels."""
    mask = cond > thr
    lab = np.zeros(cond.shape, np.int32)
    n, best, bk = 0, 0.0, 0
    for start in zip(*np.nonzero(mask)):
        if lab[start]:
            continue
        n += 1
        q = deque([start])
        lab[start] = n
        tot = 0.0
        while q:
            z, y, x = q.popleft()
            tot += cond[z, y, x]
            for dz, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                c = (z + dz, y + dy, x + dx)
                if all(0 <= c[i] < cond.shape[i] for i in range(3)) and mask[c] and not lab[c]:
                    lab[c] = n
                    q.append(c)
        if tot > best:
            best, bk = tot, n
    return lab, bk


def main():
    out = sys.argv[1]
    F = int(sys.argv[2]) if len(sys.argv) > 2 else 240
    W, H = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (240, 160)
    side = sys.argv[5] if len(sys.argv) > 5 else 'lit'
    sc = presets.make('cumulus_day', fps=24)
    sc.data['render'].update(width=W, height=H)
    eng = Engine()
    eng.prepare(sc, final=True)
    eng.simulate_to(sc, F, cache=False)
    C, g = eng.cloud, eng.gpu
    A0 = g.read(C.A[0]).astype(np.float32)   # (nz, ny, nx, 4): theta', vapour, cloud water, rain
    B0 = g.read(C.B[0]).astype(np.float32)   # cloud ice, snow, graupel and hail, the cloud's water carried (all count)
    cond = A0[..., 2] + A0[..., 3] + B0.sum(-1)
    lab, k = biggest_cloud(cond)
    zz, yy, xx = np.nonzero(lab == k)
    lo = np.maximum(np.array([xx.min(), yy.min(), zz.min()]) - 2, 0)
    hi = np.minimum(np.array([xx.max(), yy.max(), zz.max()]) + 3, np.array(C.dims))
    keep = np.zeros_like(cond, bool)
    keep[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] = True
    keep &= (lab == k) | (lab == 0)
    A1, B1 = A0.copy(), B0.copy()
    A1[~keep, 2:4] = 0.0
    B1[~keep, :] = 0.0
    ta = g.texture3d(tuple(C.dims), 'rgba32float', 'one-a')
    tb = g.texture3d(tuple(C.dims), 'rgba32float', 'one-b')
    g.upload(ta, A1)
    g.upload(tb, B1)

    cell = float(C.h)
    org = np.asarray(C.origin, float)
    c_m = org + (lo + hi) / 2.0 * cell              # the box's middle (m of sky)
    size = (hi - lo) * cell
    _spec, fire = sc.camera(F)
    look = sc.sky_look(F, True)
    l2w = fire.local_to_world()
    w2l = np.linalg.inv(l2w)
    sd_l = w2l[:3, :3] @ np.asarray(cam.sun_direction(look.sun_azimuth, look.sun_elevation), float)
    sd_l /= np.linalg.norm(sd_l)
    flat = np.array([sd_l[0], 0.0, sd_l[2]])
    flat /= np.linalg.norm(flat)
    if side == 'back':
        flat = -flat
    dist = (1.2 if side == 'below' else 2.5) * float(max(size[0], size[2]))
    eye_sky = c_m + flat * dist
    eye_sky[1] = 30.0 if side == 'below' else c_m[1]
    scale = float(C.scale)
    eye_w = (l2w @ np.append(eye_sky / scale, 1.0))[:3]
    view = cam.look_at(eye_w, (l2w @ np.append(c_m / scale, 1.0))[:3])
    hfov = math.radians(70.0) if side == 'below' else 2.0 * math.atan(0.75 * float(max(size)) / dist)
    P = cam.perspective(2.0 * math.atan(math.tan(hfov / 2.0) * H / W), W / H, 0.01, 1.0e4)
    vp = P @ view
    cs = cam.CameraState(view=view, proj=P, view_proj=vp, inv_view_proj=np.linalg.inv(vp), eye=eye_w, aspect=W / H, hfov=hfov)
    look.detail, look.visibility, look.exposure, look.brightness, look.skylight = 0.0, 1.0e12, 0.0, 1.0, 1.0
    look.ground = tuple(look.ground) if side == 'below' else (0.0, 0.0, 0.0)
    look.draw_ground = side == 'below'
    r = eng.renderer
    r.set_plate(None)
    r._ensure_fire(W, H)

    def shot(lume):
        acc = None
        t0 = time.perf_counter()
        for i in range(16):
            with g.batch() as b:
                eng.cloud_r.render(b, C, ta, tb, C.PRECIP, cs, fire, look, (W, H), seed=1000 + i * 7919, step=0.2,
                                   light=(i == 0), lume=lume)
            img = g.read(r.beauty).astype(np.float32)
            acc = img if acc is None else acc + img
        print('lume' if lume else 'classic', round(time.perf_counter() - t0, 2), 's', flush=True)
        return acc / 16

    img, img_lume = shot(None), shot((0, True))
    LIGHT = g.read(eng.cloud_r.LIGHT).astype(np.float32)
    sigma = (np.minimum(LIGHT[..., 2], DENSE) + LIGHT[..., 3])[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]

    def ray(nx, ny):
        pn = cs.inv_view_proj @ np.array([nx, ny, 0.0, 1.0])
        pf = cs.inv_view_proj @ np.array([nx, ny, 1.0, 1.0])
        a = (w2l @ np.append(pn[:3] / pn[3], 1.0))[:3]
        d = (w2l @ np.append(pf[:3] / pf[3], 1.0))[:3] - a
        return a * scale, d / np.linalg.norm(d)

    eye, fwd = ray(0.0, 0.0)
    _, upd = ray(0.0, 0.5)
    up = upd - fwd * float(upd @ fwd)
    up /= np.linalg.norm(up)
    sun = np.asarray(look.sun, float)
    level = max(float(np.mean(look.sun)), 1e-3)
    sky, hor = np.asarray(look.sky) * level * 0.22, np.asarray(look.horizon) * level * 0.3

    def hg(c, gg):
        return (1 - gg * gg) / (4 * math.pi * np.maximum(1 + gg * gg - 2 * gg * c, 1e-4) ** 1.5)

    # the sky (cloud_march.wgsl sky_col) in Mitsuba's envmap convention: u = atan2(x, -z) / 2 pi, v = acos(y) / pi
    PH, TH = np.meshgrid((np.arange(512) + 0.5) / 512 * 2 * math.pi, (np.arange(256) + 0.5) / 256 * math.pi)
    d = np.stack([np.sin(TH) * np.sin(PH), np.cos(TH), -np.sin(TH) * np.cos(PH)], -1)
    upw = np.maximum(d[..., 1], 0.0) ** 0.45
    env = hor * (1 - upw[..., None]) + sky * upw[..., None]
    cc = d @ sd_l
    env = env + sun * (0.04 * hg(cc, 0.76) + 0.02 * hg(cc, 0.3))[..., None]
    np.savez_compressed(out, img=img, img_lume=img_lume, sigma=sigma, org=org + lo * cell, cell=cell,
                        dims=np.array(sigma.shape[::-1]), eye=eye, fwd=fwd, up=up, hfov=hfov, sun_dir=sd_l, sun=sun,
                        env=env.astype(np.float32), size=np.array([W, H]),
                        ground=np.asarray(look.ground, float) if side == 'below' else np.zeros(3))
    print('saved', out)


if __name__ == '__main__':
    main()
