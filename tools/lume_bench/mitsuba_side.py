"""The reference side of the benchmark: each scene of scenes.py rendered by Mitsuba 3 (its path tracer, on the GPU with
OptiX), with Lume's own material as a Mitsuba BSDF, so any difference from Lume is in how Lume carries the light, not
in the material. Run with an environment that has Mitsuba 3 (pip install mitsuba):

    python tools/lume_bench/mitsuba_side.py OUT_DIR [SCENE ...] [--ref 16384] [--spp 4,16,64,256,1024]

Writes OUT_DIR/mitsuba_SCENE.npz: the reference picture (`ref`, at --ref samples per pixel) and, for the equal-time
comparison, Mitsuba's own pictures and seconds at each of --spp."""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scenes import SCENES                                          # noqa: E402

import drjit as dr                                                  # noqa: E402
import mitsuba as mi                                                # noqa: E402

mi.set_variant('cuda_ad_rgb')
INV_PI = 1.0 / math.pi
LUMA = (0.2126, 0.7152, 0.0722)


def _luma(c):
    return c.x * LUMA[0] + c.y * LUMA[1] + c.z * LUMA[2]


class LumeBSDF(mi.BSDF):
    """Lume's material (lume.wgsl lu_mat / lu_eval / the bounce): Lambert diffuse times (1 - rough Fresnel at the view)
    plus a GGX highlight (height-correlated Smith, Schlick's Fresnel at the facet), one-sided; sampled as Lume samples it
    (the highlight by its visible normals with probability ps, else the cosine)."""

    def __init__(self, props):
        mi.BSDF.__init__(self, props)
        self.alb = mi.Color3f(props['ar'], props['ag'], props['ab'])
        self.f0 = mi.Color3f(props['fr'], props['fg'], props['fb'])
        r = min(max(float(props['rough']), 0.03), 1.0)
        self.r = r
        self.a = r * r
        flags = mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide
        self.m_components = [flags]
        self.m_flags = flags

    def _fresnel_rough(self, nv):
        return self.f0 + (dr.maximum(mi.Color3f(1.0 - self.r), self.f0) - self.f0) * dr.power(1.0 - dr.clip(nv, 0.0, 1.0), 5.0)

    def _ps(self, nv):
        fe = _luma(self._fresnel_rough(nv))
        de = _luma(self.alb) * (1.0 - fe)
        ps = dr.clip(fe / dr.maximum(fe + de, 1e-6), 0.1, 0.95)
        return dr.select(de <= 1e-6, 1.0, ps)

    def _eval_pdf(self, wi, wo, active):
        nv = dr.maximum(wi.z, 1e-4)
        nl = wo.z
        ok = active & (wi.z > 0.0) & (nl > 0.0)
        a2 = self.a * self.a
        fv = self._fresnel_rough(nv)
        f = self.alb * (nl * INV_PI) * (1.0 - fv)
        h = dr.normalize(wi + wo)
        vh = dr.maximum(dr.dot(wi, h), 1e-4)
        k = h.z * h.z * (a2 - 1.0) + 1.0
        D = a2 / dr.maximum(math.pi * k * k, 1e-12)
        lam_v = dr.sqrt(a2 + (1.0 - a2) * nv * nv)
        lam_l = dr.sqrt(a2 + (1.0 - a2) * nl * nl)
        G2 = 2.0 * nl * nv / dr.maximum(nl * lam_v + nv * lam_l, 1e-9)
        G1 = 2.0 * nv / dr.maximum(nv + lam_v, 1e-9)
        F = self.f0 + (1.0 - self.f0) * dr.power(1.0 - vh, 5.0)
        f = f + F * (D * G2 / (4.0 * nv))
        ps = self._ps(nv)
        pdf = (1.0 - ps) * nl * INV_PI + ps * D * G1 / (4.0 * nv)
        return dr.select(ok, f, 0.0), dr.select(ok, pdf, 0.0)

    def sample(self, ctx, si, sample1, sample2, active):
        wi = si.wi
        nv = dr.maximum(wi.z, 1e-4)
        ps = self._ps(nv)
        spec = sample1 < ps
        # the highlight: a visible normal (Dupuy and Benyoub 2023), the view reflected in it
        a = self.a
        wh = dr.normalize(mi.Vector3f(a * wi.x, a * wi.y, wi.z))
        phi = 2.0 * math.pi * sample2.x
        z = (1.0 - sample2.y) * (1.0 + wh.z) - wh.z
        st = dr.sqrt(dr.clip(1.0 - z * z, 0.0, 1.0))
        hh = mi.Vector3f(st * dr.cos(phi), st * dr.sin(phi), z) + wh
        m = dr.normalize(mi.Vector3f(a * hh.x, a * hh.y, dr.maximum(hh.z, 1e-6)))
        wo_spec = 2.0 * dr.dot(wi, m) * m - wi
        wo_diff = mi.warp.square_to_cosine_hemisphere(sample2)
        wo = dr.select(spec, wo_spec, wo_diff)
        value, pdf = self._eval_pdf(wi, wo, active)
        bs = mi.BSDFSample3f()
        bs.wo = wo
        bs.pdf = pdf
        bs.eta = 1.0
        bs.sampled_component = mi.UInt32(0)
        bs.sampled_type = dr.select(spec, mi.UInt32(+mi.BSDFFlags.GlossyReflection), mi.UInt32(+mi.BSDFFlags.DiffuseReflection))
        ok = active & (pdf > 0.0)
        return bs, dr.select(ok, value / dr.maximum(pdf, 1e-12), 0.0)

    def eval(self, ctx, si, wo, active):
        return self._eval_pdf(si.wi, wo, active)[0]

    def pdf(self, ctx, si, wo, active):
        return self._eval_pdf(si.wi, wo, active)[1]

    def eval_pdf(self, ctx, si, wo, active):
        return self._eval_pdf(si.wi, wo, active)

    def to_string(self):
        return 'LumeBSDF'


mi.register_bsdf('lume', lambda props: LumeBSDF(props))


def material(o):
    """The Mitsuba BSDF of a scene object (or the floor), as Lume sees it (stage.wgsl surface_at)."""
    if float(o.get('clear', 0.0)) >= 1.0:
        return {'type': 'dielectric', 'int_ior': float(o.get('ior', 1.5)), 'ext_ior': 1.0}
    alb = np.minimum(np.asarray(o['alb'], float), 0.95)
    metal = float(o.get('metal', 0.0))
    diff = alb * (1.0 - metal)
    f0 = 0.04 * (1.0 - metal) + np.minimum(np.asarray(o['alb'], float), 1.0) * metal
    return {'type': 'lume', 'ar': float(diff[0]), 'ag': float(diff[1]), 'ab': float(diff[2]),
            'fr': float(f0[0]), 'fg': float(f0[1]), 'fb': float(f0[2]), 'rough': float(o.get('rough', 0.5))}


def build(spec, spp):
    T = mi.ScalarTransform4f
    W, H = spec['size']
    cam = spec['camera']
    d = {'type': 'scene',
         'integrator': {'type': 'path', 'max_depth': int(spec['bounces']) + 1, 'hide_emitters': False},
         'sensor': {'type': 'perspective', 'fov': float(cam['hfov']), 'fov_axis': 'x', 'near_clip': 0.01,
                    'to_world': T().look_at(origin=list(cam['eye']), target=list(cam['target']), up=[0, 1, 0]),
                    'sampler': {'type': 'independent', 'sample_count': int(spp)},
                    'film': {'type': 'hdrfilm', 'width': W, 'height': H, 'rfilter': {'type': 'box'}, 'pixel_format': 'rgb'}},
         'sky': {'type': 'constant', 'radiance': {'type': 'rgb', 'value': [float(spec['sky'])] * 3}}}
    if spec['floor'] is not None:
        d['floor'] = {'type': 'rectangle', 'to_world': T().rotate([1, 0, 0], -90).scale(2000.0), 'bsdf': material(spec['floor'])}
    for i, o in enumerate(spec['objects']):
        if o['shape'] == 'sphere':
            d[f'o{i}'] = {'type': 'sphere', 'center': list(o['pos']), 'radius': float(o['size'][0]), 'bsdf': material(o)}
        else:
            d[f'o{i}'] = {'type': 'cube', 'to_world': T().translate(list(o['pos'])).scale(list(o['size'])), 'bsdf': material(o)}
    for i, L in enumerate(spec['lamps']):
        r = float(L['radius'])
        rad = np.asarray(L['power'], float) / (math.pi * r * r)      # a sphere of radiance L: intensity L pi r^2
        d[f'lamp{i}'] = {'type': 'sphere', 'center': list(L['pos']), 'radius': r,
                         'emitter': {'type': 'area', 'radiance': {'type': 'rgb', 'value': [float(x) for x in rad]}}}
    return mi.load_dict(d)


def render(spec, spp, seed=0):
    sc = build(spec, spp)
    t0 = time.perf_counter()
    img = np.array(mi.render(sc, spp=int(spp), seed=seed), np.float32)
    return img, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('scenes', nargs='*')
    ap.add_argument('--ref', type=int, default=16384)
    ap.add_argument('--spp', default='4,16,64,256,1024')
    ap.add_argument('--repeat', type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spps = [int(x) for x in args.spp.split(',')]
    for name in args.scenes or list(SCENES):
        spec = SCENES[name]
        # the reference, in chunks of 1024 samples (each its own seed), averaged
        chunks = max(1, args.ref // 1024)
        acc = None
        t_ref = 0.0
        for k in range(chunks):
            img, dt = render(spec, min(args.ref, 1024), seed=1000 + k)
            acc = img if acc is None else acc + img
            t_ref += dt
        ref = acc / chunks
        print(f'mitsuba {name} reference {chunks * min(args.ref, 1024)} spp: {t_ref:.1f} s', flush=True)
        render(spec, spps[0])     # (warm the JIT up: not timed)
        imgs, secs = [], []
        for spp in spps:
            dt = float('inf')
            for _ in range(args.repeat):   # (the fastest of a few: the GPU may be shared)
                img, t = render(spec, spp, seed=7)
                dt = min(dt, t)
            imgs.append(img)
            secs.append(dt)
            print(f'mitsuba {name} {spp} spp: {dt:.2f} s', flush=True)
        np.savez_compressed(out / f'mitsuba_{name}.npz', ref=ref, spp=np.array(spps), secs=np.array(secs), imgs=np.stack(imgs))


if __name__ == '__main__':
    main()
