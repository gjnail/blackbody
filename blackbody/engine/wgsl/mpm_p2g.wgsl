// Matter, particle to grid: each particle adds its mass, its momentum (with its APIC affine velocity) and the push
// of its stress to the 27 nodes round it (MLS-MPM): by a sheet of fabric, only to the nodes on its own side of it
// (mpm_common.wgsl SHEET_N): what of its push would have gone to the others, the sheet takes as far as it holds (the
// weight of sand lying on a sling), and the rest comes back to its own nodes (sheet_hold: a heap under a draped cloth
// is not pressed down by its own weight).
//!include mpm_common.wgsl

struct Params {
  n: vec4<f32>,      // grid nodes (x, y, z), particles (count)
  s: vec4<f32>,      // step (s), node spacing (m), fabric (mpm_common.wgsl SHEET_N), simulation time (s)
  mats: array<MMat, MAX_MATS>,
};

@group(0) @binding(0) var<storage, read> P: array<MParticle>;
@group(0) @binding(1) var<storage, read_write> G: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read> CF: array<i32>;   // the fabric round each node (cloth_matter.wgsl)
@group(0) @binding(3) var<storage, read_write> CT: array<atomic<i32>>;   // per node: momentum the fabric took (3)
@group(1) @binding(0) var<uniform> U: Params;

fn sheet(k: u32, tdx: f32) -> Sheet {
  return sheet_at(vec4<i32>(CF[k], CF[k + 1u], CF[k + 2u], CF[k + 3u]), vec4<i32>(CF[k + 4u], CF[k + 5u], CF[k + 6u],
                  CF[k + 7u]), CF[k + 8u], tdx, U.s.y);
}

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
  let side = select(0.0, p.f2.w, U.s.z > 0.5);
  let tdx = (U.s.z - 1.0) / dx;
  let base = vec3<i32>(floor(p.x.xyz - vec3<f32>(0.5)));
  let fx = p.x.xyz - vec3<f32>(base);
  let w = bspline(fx);
  // by a sheet of fabric: the nodes across it (far), the push it would have given them that comes back to its own as
  // far as the sheet gives way (back), and its own nodes' weight (near)
  var far: array<bool, 27>;
  var back = vec3<f32>(0.0);
  var near = 0.0;
  if (side != 0.0) {
    for (var j = 0; j < 27; j++) {
      let o = vec3<i32>(j / 9, (j / 3) % 3, j % 3);
      let node = base + o;
      if (any(node < vec3<i32>(0)) || any(node >= n)) { continue; }
      let wt = w[o.x].x * w[o.y].y * w[o.z].z;
      let sh = sheet(nidx(node, n) * SHEET_N, tdx);
      if (sh.side != 0.0 && sh.side != side) {
        far[j] = true;
        let push = wt * (affine * ((vec3<f32>(o) - fx) * dx));
        let hold = sheet_hold(sh, side);
        back += (1.0 - hold) * push;
        let q = clamp(hold * push * SHEET_FX_T, vec3<f32>(-2.0e9), vec3<f32>(2.0e9));
        if (dot(q, q) >= 0.25) {
          let kt = 3u * nidx(node, n);
          atomicAdd(&CT[kt], i32(round(q.x)));
          atomicAdd(&CT[kt + 1u], i32(round(q.y)));
          atomicAdd(&CT[kt + 2u], i32(round(q.z)));
        }
      } else {
        near += wt;
      }
    }
  }
  for (var j = 0; j < 27; j++) {
    let o = vec3<i32>(j / 9, (j / 3) % 3, j % 3);
    let node = base + o;
    if (any(node < vec3<i32>(0)) || any(node >= n) || far[j]) { continue; }
    let wt = w[o.x].x * w[o.y].y * w[o.z].z;
    let dpos = (vec3<f32>(o) - fx) * dx;
    let mom = wt * (mass * v + affine * dpos) + back * (wt / max(near, 1.0e-6));
    let k = nidx(node, n) * NODE;
    let q = clamp(mom * FX_P, vec3<f32>(-2.0e9), vec3<f32>(2.0e9));
    atomicAdd(&G[k], i32(round(q.x)));
    atomicAdd(&G[k + 1u], i32(round(q.y)));
    atomicAdd(&G[k + 2u], i32(round(q.z)));
    atomicAdd(&G[k + 3u], i32(round(wt * mass * FX_MASS)));
  }
}
