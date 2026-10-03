"""Lume's clouds against Mitsuba, the Mitsuba side (README: Clouds). Run with the Mitsuba environment:

    mitsuba-env\\Scripts\\python tools/lume_bench/cloud_mitsuba.py IN.npz OUT.npz [spp] [all|sun|sky]

Path traces the cloud cloud_blackbody.py wrote (volpath, every bounce): its extinction grid as a heterogeneous medium
(trilinear, cell-centred, as the march reads it), albedo 1, the droplets' phase (0.2 HG -0.25 + 0.8 HG 0.85: cloud_march.wgsl
phase), the sun a directional light of irradiance pi times the renderer's sun (that is the radiance of a white wall facing
it), the sky the renderer's sky as an envmap, and with `below` the ground as a diffuse plane. sun / sky: only that light."""
import sys
import time

import numpy as np


def main():
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')
    z = np.load(sys.argv[1])
    spp = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    mode = sys.argv[4] if len(sys.argv) > 4 else 'all'
    W, H = (int(x) for x in z['size'])
    org, cell, dims = z['org'], float(z['cell']), z['dims']
    ext = dims * cell
    T = mi.ScalarTransform4f
    eye, fwd, up = z['eye'], z['fwd'], z['up']
    env = z['env'].astype(np.float32) * (0.0 if mode == 'sun' else 1.0) + (1e-6 if mode == 'sun' else 0.0)
    scene = {
        'type': 'scene',
        'integrator': {'type': 'volpath', 'max_depth': -1, 'rr_depth': 400},
        'sensor': {'type': 'perspective', 'fov': float(np.degrees(z['hfov'])), 'fov_axis': 'x', 'near_clip': 1.0,
                   'far_clip': 1.0e6, 'to_world': T().look_at(origin=list(eye), target=list(eye + fwd * 1000.0), up=list(up)),
                   'sampler': {'type': 'independent', 'sample_count': spp},
                   'film': {'type': 'hdrfilm', 'width': W, 'height': H, 'rfilter': {'type': 'box'}, 'pixel_format': 'rgb'}},
        'sky': {'type': 'envmap', 'bitmap': mi.Bitmap(env)},
        'sun': {'type': 'directional', 'direction': [float(-x) for x in z['sun_dir']],
                'irradiance': {'type': 'rgb', 'value': [float(x) for x in z['sun'] * np.pi]}},
        'cloud': {
            'type': 'cube', 'bsdf': {'type': 'null'},
            'to_world': T().translate(list(org + ext / 2)).scale(list(ext / 2)),
            'interior': {
                'type': 'heterogeneous', 'albedo': 1.0, 'scale': 1.0,
                'sigma_t': {'type': 'gridvolume', 'grid': mi.VolumeGrid(z['sigma'].astype(np.float32)[..., None]),
                            'filter_type': 'trilinear', 'to_world': T().translate(list(org)).scale(list(ext))},
                'phase': {'type': 'blendphase', 'weight': 0.8,
                          'phase1': {'type': 'hg', 'g': -0.25}, 'phase2': {'type': 'hg', 'g': 0.85}},
            },
        },
    }
    if float(np.max(z['ground'])) > 0.0:
        scene['ground'] = {'type': 'rectangle',
                           'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [float(x) for x in z['ground']]}},
                           'to_world': T().translate([float(eye[0]), 0.0, float(eye[2])]).rotate([1, 0, 0], -90).scale([2.0e5, 2.0e5, 1.0])}
    if mode == 'sky':
        del scene['sun']
    t0 = time.perf_counter()
    img = np.array(mi.render(mi.load_dict(scene), spp=spp), np.float32)
    print('mitsuba', spp, 'spp', round(time.perf_counter() - t0, 1), 's')
    np.savez_compressed(sys.argv[2], img=img)


def compare(bb, ref):
    """Lume and the classic estimate against the reference, over the cloud's pixels (where they differ)."""
    z = np.load(bb)
    m = np.load(ref)['img'][..., :3]
    c, lu = z['img'][..., :3], z['img_lume'][..., :3]
    cl = np.abs(c - lu).mean(-1) >= 1e-4
    for name, x in (('classic', c), ('lume', lu)):
        r = np.log2(np.maximum(x.mean(-1), 1e-4) / np.maximum(m.mean(-1), 1e-4))[cl]
        print(f'{name}: over the cloud {x[cl].sum() / m[cl].sum():.3f} of the reference; per pixel median {np.median(r):+.3f} stops, '
              f'rms {np.sqrt((r * r).mean()):.3f}, within 10%: {100 * (np.abs(r) < np.log2(1.1)).mean():.0f}%')


if __name__ == '__main__':
    if sys.argv[1] == 'compare':
        compare(sys.argv[2], sys.argv[3])
    else:
        main()
