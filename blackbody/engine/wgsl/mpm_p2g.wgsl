// Matter, particle to grid: each particle adds its mass, its momentum (with its APIC affine velocity) and the push
// of its stress to the 27 nodes round it (MLS-MPM).
//!include mpm_common.wgsl

struct Params {
  n: vec4<f32>,      // grid nodes (x, y, z), particles (count)
  s: vec4<f32>,      // step (s), node spacing (m), _, simulation time (s)
  mats: array<MMat, MAX_MATS>,
};

@group(0) @binding(0) var<storage, read> P: array<MParticle>;
@group(0) @binding(1) var<storage, read_write> G: array<atomic<i32>>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.n.w)) { return; }
  let p = P[i];
  if (p.x.w < 0.0) { return; }
  let m = U.mats[u32(p.x.w)];
  let n = vec3<i32>(U.n.xyz);
  let dt = U.s.x;
  let dx = U.s.y;
  // masses and momenta are in units of a particle of water (1000 kg/m^3 times its volume)
  let mass = m.a.w / 1000.0;
  let held = p.c1.w > U.s.w;
  // the push of its stress, and its affine velocity, as one affine momentum about it
  var affine = mass * from_rows(p.c0, p.c1, p.c2);
  if (!held) {
    let tau = kirchhoff(m, from_rows(p.f0, p.f1, p.f2), p.v.w);
    affine = affine - (dt * 4.0 / (dx * dx) / 1000.0) * tau;
  }
  let v = select(p.v.xyz, vec3<f32>(0.0), held);
  let base = vec3<i32>(floor(p.x.xyz - vec3<f32>(0.5)));
  let fx = p.x.xyz - vec3<f32>(base);
  let w = bspline(fx);
  for (var a = 0; a < 3; a++) {
    for (var b = 0; b < 3; b++) {
      for (var c = 0; c < 3; c++) {
        let node = base + vec3<i32>(a, b, c);
        if (any(node < vec3<i32>(0)) || any(node >= n)) { continue; }
        let wt = w[a].x * w[b].y * w[c].z;
        let dpos = (vec3<f32>(f32(a), f32(b), f32(c)) - fx) * dx;
        let mom = wt * (mass * v + affine * dpos);
        let k = nidx(node, n) * NODE;
        let q = clamp(mom * FX_P, vec3<f32>(-2.0e9), vec3<f32>(2.0e9));
        atomicAdd(&G[k], i32(round(q.x)));
        atomicAdd(&G[k + 1u], i32(round(q.y)));
        atomicAdd(&G[k + 2u], i32(round(q.z)));
        atomicAdd(&G[k + 3u], i32(round(wt * mass * FX_MASS)));
      }
    }
  }
}
