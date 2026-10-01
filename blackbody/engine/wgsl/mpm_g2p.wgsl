// Matter, grid to particle: each particle picks up the grid's new velocity and the affine velocity round it (APIC),
// deforms with it (F <- (I + dt C) F), gives way where its material yields, and moves. One that leaves the grid
// (through an open side, the top, or the bottom of a box without ground) is gone.
//!include mpm_common.wgsl

struct Params {
  n: vec4<f32>,      // grid nodes (x, y, z), particles (count)
  s: vec4<f32>,      // step (s), node spacing (m), _, simulation time at the start of the step (s)
  mats: array<MMat, MAX_MATS>,
};

@group(0) @binding(0) var<storage, read_write> P: array<MParticle>;
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<atomic<u32>>;   // fastest speed (f32 bits), bounds (grid units), least Jp
@group(1) @binding(0) var<uniform> U: Params;

// The bounds of the matter (grid units, kept as integers shifted by 2^20 so they stay positive).
fn bound(x: vec3<f32>) {
  let lo = vec3<u32>(floor(x) + vec3<f32>(1048576.0));
  let hi = vec3<u32>(ceil(x) + vec3<f32>(1048576.0));
  atomicMin(&stats[1], lo.x);
  atomicMin(&stats[2], lo.y);
  atomicMin(&stats[3], lo.z);
  atomicMax(&stats[4], hi.x);
  atomicMax(&stats[5], hi.y);
  atomicMax(&stats[6], hi.z);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.n.w)) { return; }
  var p = P[i];
  if (p.x.w < 0.0) { return; }
  let n = vec3<i32>(U.n.xyz);
  let dt = U.s.x;
  let dx = U.s.y;
  if (p.c1.w > U.s.w) {
    // held where it is until it is let go (keeping the speed it is thrown at then)
    bound(p.x.xyz);
    return;
  }
  let base = vec3<i32>(floor(p.x.xyz - vec3<f32>(0.5)));
  let fx = p.x.xyz - vec3<f32>(base);
  let w = bspline(fx);
  var v = vec3<f32>(0.0);
  var B = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  for (var a = 0; a < 3; a++) {
    for (var b = 0; b < 3; b++) {
      for (var c = 0; c < 3; c++) {
        let node = clamp(base + vec3<i32>(a, b, c), vec3<i32>(0), n - vec3<i32>(1));
        let wt = w[a].x * w[b].y * w[c].z;
        let dpos = (vec3<f32>(f32(a), f32(b), f32(c)) - fx) * dx;
        let vi = textureLoad(vel, node, 0).xyz;
        v += wt * vi;
        B += wt * mat3x3<f32>(vi * dpos.x, vi * dpos.y, vi * dpos.z);
      }
    }
  }
  let C = (4.0 / (dx * dx)) * B;
  let m = U.mats[u32(p.x.w)];
  let y = plastic(m, (eye3() + dt * C) * from_rows(p.f0, p.f1, p.f2), p.v.w, dt);
  let Ct = transpose(C);
  let Ft = transpose(y.f);
  p.c0 = vec4<f32>(Ct[0], p.c0.w);
  p.c1 = vec4<f32>(Ct[1], p.c1.w);
  p.c2 = vec4<f32>(Ct[2], p.c2.w);
  p.f0 = vec4<f32>(Ft[0], p.f0.w);
  p.f1 = vec4<f32>(Ft[1], p.f1.w);
  p.f2 = vec4<f32>(Ft[2], p.f2.w);
  p.v = vec4<f32>(v, y.jp);
  let x = p.x.xyz + v * (dt / dx);
  if (any(x < vec3<f32>(0.6)) || any(x >= vec3<f32>(n) - vec3<f32>(1.6))) {
    p.x.w = -1.0;              // gone out of the grid
    P[i] = p;
    return;
  }
  p.x = vec4<f32>(x, p.x.w);
  P[i] = p;
  atomicMax(&stats[0], bitcast<u32>(length(v)));
  bound(x);
  atomicMin(&stats[7], bitcast<u32>(max(y.jp, 0.0)));     // (the most packed snow: how hard it is)
}
