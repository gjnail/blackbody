// Velocity self-advection on the MAC grid, each face component traced from its own face centre.
//
// Air traced back to beyond the box's open sides or top comes in no faster than the air out there
// blows in: not at all in still air (a fire still draws air in, through the pressure the projection
// supplies every step), at most at the wind's speed on the windward side. Carrying the edge's own
// inflow in again instead lets an inflow feed itself step after step; on fine grids that grew into a
// gale through the whole box. The bottom of a box without a floor is left as it was: it usually sits
// just under the fire (a torch, a candle), in the updraft the fire draws in.
//!include common.wgsl

struct Params {
  g: Grid,
  a: vec4<f32>,    // x = MacCormack strength
  amb: vec4<f32>,  // xyz = the air outside the box (the wind, m/s), _
};

@group(0) @binding(0) var vel: texture_3d<f32>;
@group(0) @binding(1) var fwd: texture_3d<f32>;
@group(0) @binding(2) var dst: texture_storage_3d<${VELFMT}, write>;
@group(0) @binding(3) var lin: sampler;
@group(1) @binding(0) var<uniform> U: Params;

fn trace(p: vec3<f32>, n: vec3<f32>, k: f32) -> vec3<f32> {
  let v0 = vel_at(vel, lin, p, n);
  let v1 = vel_at(vel, lin, p + 0.5 * k * v0, n);
  return p + k * v1;
}

fn comp_at(t: texture_3d<f32>, p: vec3<f32>, n: vec3<f32>, axis: i32) -> f32 {
  if (axis == 0) { return u_at(t, lin, p, n); }
  if (axis == 1) { return v_at(t, lin, p, n); }
  return w_at(t, lin, p, n);
}

fn pick(v: vec4<f32>, axis: i32) -> f32 {
  if (axis == 0) { return v.x; }
  if (axis == 1) { return v.y; }
  return v.z;
}

// Offset from cell space to texel space for a component's lattice, and its largest valid texel centre.
fn lattice_off(axis: i32) -> vec3<f32> {
  if (axis == 0) { return vec3<f32>(0.5, 0.0, 0.0); }
  if (axis == 1) { return vec3<f32>(0.0, 0.5, 0.0); }
  return vec3<f32>(0.0, 0.0, 0.5);
}

fn lattice_hi(axis: i32, n: vec3<f32>) -> vec3<f32> {
  return n - vec3<f32>(0.5) + lattice_off(axis) * 2.0;
}

fn face_pos(c: vec3<i32>, axis: i32) -> vec3<f32> {
  return vec3<f32>(c) + vec3<f32>(0.5) - lattice_off(axis);
}

// A velocity component traced back to p: its inflow through an open side or the top is held to the
// outside air's (see the notes at the top).
fn from_outside(v: f32, p: vec3<f32>, axis: i32, n: vec3<f32>) -> f32 {
  let w = pick(U.amb, axis);
  if (axis == 1) {
    if (U.g.bc.y > 0.5 && p.y > n.y) { return max(v, min(w, 0.0)); }
    return v;
  }
  if (U.g.bc.x < 0.5) { return v; }
  let x = select(p.z, p.x, axis == 0);
  let hi = select(n.z, n.x, axis == 0);
  if (x < 0.0) { return min(v, max(w, 0.0)); }
  if (x > hi) { return max(v, min(w, 0.0)); }
  return v;
}

fn advect_sl(c: vec3<i32>, axis: i32, n: vec3<f32>, k: f32) -> f32 {
  let pb = trace(face_pos(c, axis), n, -k);
  return from_outside(comp_at(vel, pb, n, axis), pb, axis, n);
}

fn advect_mc(c: vec3<i32>, axis: i32, n: vec3<f32>, k: f32) -> f32 {
  let p = face_pos(c, axis);
  let pb = trace(p, n, -k);
  let pf = trace(p, n, k);
  let ahead = pick(textureLoad(fwd, c, 0), axis);
  let back = comp_at(fwd, pf, n, axis);
  let phi = pick(textureLoad(vel, c, 0), axis);
  let r = ahead + 0.5 * U.a.x * (phi - back);
  let hi_t = lattice_hi(axis, n);
  let q = clamp(pb + lattice_off(axis), vec3<f32>(0.5), hi_t) - vec3<f32>(0.5);
  let i0 = vec3<i32>(floor(q));
  let hi = vec3<i32>(hi_t - vec3<f32>(0.5));
  var lo = 3.0e38;
  var up = -3.0e38;
  for (var j = 0; j < 8; j++) {
    let s = pick(textureLoad(vel, min(i0 + vec3<i32>(j & 1, (j >> 1) & 1, j >> 2), hi), 0), axis);
    lo = min(lo, s);
    up = max(up, s);
  }
  // Selle et al.: where the correction would create a new extremum, fall back to the
  // semi-Lagrangian value instead of clamping (clamping can pump energy into divergent flow)
  if (r < lo || r > up) { return ahead; }
  return from_outside(r, pb, axis, n);
}

@compute @workgroup_size(8, 8, 4)
fn sl(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = U.g.n.xyz;
  let d = vec3<i32>(n);
  let c = vec3<i32>(id);
  if (any(c > d)) { return; }
  let k = U.g.bc.w / U.g.n.w;
  var o = vec4<f32>(0.0);
  if (c.y < d.y && c.z < d.z) { o.x = advect_sl(c, 0, n, k); }
  if (c.x < d.x && c.z < d.z) { o.y = advect_sl(c, 1, n, k); }
  if (c.x < d.x && c.y < d.y) { o.z = advect_sl(c, 2, n, k); }
  textureStore(dst, c, o);
}

@compute @workgroup_size(8, 8, 4)
fn mc(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = U.g.n.xyz;
  let d = vec3<i32>(n);
  let c = vec3<i32>(id);
  if (any(c > d)) { return; }
  let k = U.g.bc.w / U.g.n.w;
  var o = vec4<f32>(0.0);
  if (c.y < d.y && c.z < d.z) { o.x = advect_mc(c, 0, n, k); }
  if (c.x < d.x && c.z < d.z) { o.y = advect_mc(c, 1, n, k); }
  if (c.x < d.x && c.y < d.y) { o.z = advect_mc(c, 2, n, k); }
  textureStore(dst, c, o);
}
