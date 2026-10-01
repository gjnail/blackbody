// Clouds: the fields A and B ride the wind (engine/cloud.py). Semi-Lagrangian, with a MacCormack
// correction clamped to the neighbourhood it came from (sharp cloud edges, no new extremes). Air drawn in
// through the open sides is the background air: no excess heat, the environment's vapour, no condensate.
// What air carries is its whole potential temperature, not its excess over the background: interpolated as
// the whole and turned back into the excess at the cell it arrives in (a parcel lifted through stable air
// arrives colder than the air round it: the stratification that holds the sky still).
//   sl:  dst = the field carried back along the wind (phi^)
//   mc:  dst = phi^ + (phi - phi~) / 2, phi~ being phi^ carried forward again, clamped
//!include cloud_common.wgsl

@group(0) @binding(0) var vel: texture_3d<f32>;
@group(0) @binding(1) var src: texture_3d<f32>;
@group(0) @binding(2) var src2: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba32float, write>;
@group(0) @binding(4) var<storage, read> base: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> U: Cloud;

// the background's potential temperature at height y (cells), linear between levels
fn theta0(y: f32, n: vec3<i32>) -> f32 {
  let q = clamp(y - 0.5, 0.0, f32(n.y - 1));
  let j = i32(floor(q));
  let j1 = min(j + 1, n.y - 1);
  return mix(base[u32(j) * 2u].x, base[u32(j1) * 2u].x, q - f32(j));
}

fn inflow(p: vec3<f32>, n: vec3<i32>, is_a: bool) -> vec4<f32> {
  if (!is_a) { return vec4<f32>(0.0); }
  let j = clamp(i32(floor(p.y)), 0, n.y - 1);
  return vec4<f32>(theta0(p.y, n), base[u32(j) * 2u].w, 0.0, 0.0);
}

// the field at p, with A's first component as the whole potential temperature
fn fetch(t: texture_3d<f32>, p: vec3<f32>, n: vec3<i32>, is_a: bool) -> vec4<f32> {
  // past the open sides: the background air
  let nf = vec3<f32>(n);
  if (U.g.bc.x > 0.5 && (p.x < 0.0 || p.x > nf.x || p.z < 0.0 || p.z > nf.z)) { return inflow(p, n, is_a); }
  var v = c_cell(t, p, n);
  if (is_a) { v.x += theta0(p.y, n); }
  return v;
}

fn whole(t: texture_3d<f32>, c: vec3<i32>, n: vec3<i32>, is_a: bool) -> vec4<f32> {
  var v = textureLoad(t, c, 0);
  if (is_a) { v.x += theta0(f32(c.y) + 0.5, n); }
  return v;
}

fn excess(v: vec4<f32>, c: vec3<i32>, n: vec3<i32>, is_a: bool) -> vec4<f32> {
  var r = v;
  if (is_a) { r.x -= theta0(f32(c.y) + 0.5, n); }
  return r;
}

fn back(c: vec3<i32>, n: vec3<i32>, dir: f32) -> vec3<f32> {
  let p = vec3<f32>(c) + vec3<f32>(0.5);
  let dt = U.g.bc.w;
  let h = U.g.n.w;
  let v = c_vel(vel, p, n);
  let mid = p - 0.5 * dir * v * dt / h;
  let v2 = c_vel(vel, mid, n);
  return p - dir * v2 * dt / h;
}

fn sl_main(id: vec3<u32>, is_a: bool) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  textureStore(dst, c, excess(fetch(src, back(c, n, 1.0), n, is_a), c, n, is_a));
}

fn mc_main(id: vec3<u32>, is_a: bool) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let phi = whole(src, c, n, is_a);
  let hat = whole(src2, c, n, is_a);
  // carried forward again
  let fwd = fetch(src2, back(c, n, -1.0), n, is_a);
  var r = hat + 0.5 * (phi - fwd);
  // clamp to the cells round where it came from
  let p = back(c, n, 1.0) - vec3<f32>(0.5);
  let i = vec3<i32>(floor(p));
  var lo = vec4<f32>(1.0e9);
  var hi = vec4<f32>(-1.0e9);
  for (var o = 0; o < 8; o++) {
    let q = clamp(i + vec3<i32>(o & 1, (o >> 1) & 1, o >> 2), vec3<i32>(0), n - vec3<i32>(1));
    let v = whole(src, q, n, is_a);
    lo = min(lo, v);
    hi = max(hi, v);
  }
  let pp = p + vec3<f32>(0.5);
  let nf = vec3<f32>(n);
  if (U.g.bc.x > 0.5 && (pp.x < 0.0 || pp.x > nf.x || pp.z < 0.0 || pp.z > nf.z)) {
    let e = inflow(pp, n, is_a);
    lo = min(lo, e);
    hi = max(hi, e);
  }
  r = clamp(r, lo, hi);
  textureStore(dst, c, excess(r, c, n, is_a));
}

@compute @workgroup_size(8, 8, 4)
fn sl_a(@builtin(global_invocation_id) id: vec3<u32>) { sl_main(id, true); }
@compute @workgroup_size(8, 8, 4)
fn sl_b(@builtin(global_invocation_id) id: vec3<u32>) { sl_main(id, false); }
@compute @workgroup_size(8, 8, 4)
fn mc_a(@builtin(global_invocation_id) id: vec3<u32>) { mc_main(id, true); }
@compute @workgroup_size(8, 8, 4)
fn mc_b(@builtin(global_invocation_id) id: vec3<u32>) { mc_main(id, false); }
