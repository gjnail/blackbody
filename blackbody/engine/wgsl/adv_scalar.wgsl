// Scalar advection (temperature, fuel, smoke, flame) with an optional MacCormack correction.
//!include common.wgsl

struct Params {
  g: Grid,
  a: vec4<f32>,  // x = MacCormack strength (0 = semi-Lagrangian, 1 = full correction)
};

@group(0) @binding(0) var vel: texture_3d<f32>;
@group(0) @binding(1) var src: texture_3d<f32>;
@group(0) @binding(2) var fwd: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var lin: sampler;
@group(1) @binding(0) var<uniform> U: Params;

// Second-order Runge-Kutta trace; k = signed dt / h (negative traces backwards in time).
fn trace(p: vec3<f32>, n: vec3<f32>, k: f32) -> vec3<f32> {
  let v0 = vel_at(vel, lin, p, n);
  let v1 = vel_at(vel, lin, p + 0.5 * k * v0, n);
  return p + k * v1;
}

@compute @workgroup_size(8, 8, 4)
fn sl(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = U.g.n.xyz;
  let c = vec3<i32>(id);
  if (any(c >= vec3<i32>(n))) { return; }
  let k = U.g.bc.w / U.g.n.w;
  let pb = trace(vec3<f32>(c) + 0.5, n, -k);
  textureStore(dst, c, samp_c(src, lin, pb, n));
}

@compute @workgroup_size(8, 8, 4)
fn mc(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = U.g.n.xyz;
  let c = vec3<i32>(id);
  if (any(c >= vec3<i32>(n))) { return; }
  let k = U.g.bc.w / U.g.n.w;
  let p = vec3<f32>(c) + 0.5;
  let pb = trace(p, n, -k);
  let pf = trace(p, n, k);
  let ahead = textureLoad(fwd, c, 0);   // semi-Lagrangian result here
  let back = samp_c(fwd, lin, pf, n);   // that result carried back again
  let phi = textureLoad(src, c, 0);
  let r = ahead + 0.5 * U.a.x * (phi - back);
  // limiter: stay within the values the backtrace interpolated between
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
  // revert to the semi-Lagrangian value per channel where the correction overshoots
  let bad = (r < lo) | (r > up);
  textureStore(dst, c, select(r, ahead, bad));
}
