// Upres: fire and smoke carried on a grid several times finer than the simulation, moved by the
// simulated air (sampled from the simulation's velocity) plus small swirls whose strength follows
// how much the air is spinning there. The reaction kernel then burns fuel on the fine grid too, so
// flames get fine detail without a full-resolution pressure solve. Semi-Lagrangian with an
// optional MacCormack correction, like adv_scalar.wgsl.
//!include common.wgsl
//!include noise.wgsl

struct Params {
  g: Grid,          // the fine grid
  up: vec4<f32>,    // fine cells per simulation cell, turbulence strength, seed, MacCormack strength
  lo: vec4<f32>,    // simulation grid dims; w = simulation cell size (m)
};

@group(0) @binding(0) var vel: texture_3d<f32>;   // simulation grid, face velocities
@group(0) @binding(1) var curl: texture_3d<f32>;  // simulation grid, vorticity (w = magnitude)
@group(0) @binding(2) var src: texture_3d<f32>;
@group(0) @binding(3) var fwd: texture_3d<f32>;
@group(0) @binding(4) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var lin: sampler;
@group(1) @binding(0) var<uniform> U: Params;

// Velocity (m/s) at fine-grid position p (cells).
fn fine_vel(p: vec3<f32>) -> vec3<f32> {
  let pl = p / U.up.x;
  var v = vel_at(vel, lin, pl, U.lo.xyz);
  if (U.up.y > 0.0) {
    // swirls about two fine cells across, as fast as the simulated spin turns over a simulation cell
    let spin = samp_c(curl, lin, pl, U.lo.xyz).w;
    let f = U.up.x / (2.0 * U.lo.w);
    let wp = world_of(U.g, p);
    let q = wp * f + vec3<f32>(U.up.z, -U.g.org.w * 0.7, U.up.z * 0.37);
    v += curl_noise(q) * (U.up.y * 0.35 * spin * U.lo.w);
  }
  return v;
}

fn trace(p: vec3<f32>, k: f32) -> vec3<f32> {
  let v0 = fine_vel(p);
  let v1 = fine_vel(p + 0.5 * k * v0);
  return p + k * v1;
}

// The velocity the fine grid moves with, at its cell centres (m/s): written out with VDB exports.
@compute @workgroup_size(8, 8, 4)
fn velocity(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = U.g.n.xyz;
  let c = vec3<i32>(id);
  if (any(c >= vec3<i32>(n))) { return; }
  textureStore(dst, c, vec4<f32>(fine_vel(vec3<f32>(c) + 0.5), 0.0));
}

@compute @workgroup_size(8, 8, 4)
fn sl(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = U.g.n.xyz;
  let c = vec3<i32>(id);
  if (any(c >= vec3<i32>(n))) { return; }
  let k = U.g.bc.w / U.g.n.w;
  textureStore(dst, c, samp_c(src, lin, trace(vec3<f32>(c) + 0.5, -k), n));
}

@compute @workgroup_size(8, 8, 4)
fn mc(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = U.g.n.xyz;
  let c = vec3<i32>(id);
  if (any(c >= vec3<i32>(n))) { return; }
  let k = U.g.bc.w / U.g.n.w;
  let p = vec3<f32>(c) + 0.5;
  let pb = trace(p, -k);
  let pf = trace(p, k);
  let ahead = textureLoad(fwd, c, 0);
  let back = samp_c(fwd, lin, pf, n);
  let phi = textureLoad(src, c, 0);
  let r = ahead + 0.5 * U.up.w * (phi - back);
  let q = clamp(pb, vec3<f32>(0.5), n - vec3<f32>(0.5)) - vec3<f32>(0.5);
  let i0 = vec3<i32>(floor(q));
  let hi = vec3<i32>(n) - vec3<i32>(1);
  var lo = vec4<f32>(3.0e38);
  var up = vec4<f32>(-3.0e38);
  for (var j = 0; j < 8; j++) {
    let s = textureLoad(src, min(i0 + vec3<i32>(j & 1, (j >> 1) & 1, j >> 2), hi), 0);
    lo = min(lo, s);
    up = max(up, s);
  }
  let bad = (r < lo) | (r > up);
  textureStore(dst, c, select(r, ahead, bad));
}
