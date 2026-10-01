// Clouds: the wind carries itself (engine/cloud.py), each MAC face traced back along the wind at its own
// place, semi-Lagrangian with a clamped MacCormack correction. Past the open sides the air moves with the
// background wind of its height.
//   sl:  dst = the velocity carried back
//   mc:  dst = corrected (src = the old velocity, src2 = the sl result)
//!include cloud_common.wgsl

@group(0) @binding(0) var vel: texture_3d<f32>;
@group(0) @binding(1) var src: texture_3d<f32>;
@group(0) @binding(2) var src2: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<${VELFMT}, write>;
@group(0) @binding(4) var<storage, read> base: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> U: Cloud;

fn wind0(y: f32, n: vec3<i32>) -> vec3<f32> {
  let j = clamp(i32(floor(y)), 0, n.y - 1);
  let w = base[u32(j) * 2u + 1u];
  return vec3<f32>(w.x, 0.0, w.y);
}

fn comp(t: texture_3d<f32>, p: vec3<f32>, k: u32, n: vec3<i32>) -> f32 {
  let nf = vec3<f32>(n);
  if (U.g.bc.x > 0.5 && (p.x < 0.0 || p.x > nf.x || p.z < 0.0 || p.z > nf.z)) { return wind0(p.y, n)[k]; }
  return c_face(t, p, k, n);
}

fn trace(p: vec3<f32>, n: vec3<i32>, dir: f32) -> vec3<f32> {
  let dt = U.g.bc.w;
  let h = U.g.n.w;
  let v = c_vel(vel, p, n);
  let mid = p - 0.5 * dir * v * dt / h;
  return p - dir * c_vel(vel, mid, n) * dt / h;
}

fn face_pos(c: vec3<i32>, k: u32) -> vec3<f32> {
  var o = vec3<f32>(0.5);
  o[k] = 0.0;
  return vec3<f32>(c) + o;
}

fn in_faces(c: vec3<i32>, k: u32, n: vec3<i32>) -> bool {
  var lim = n - vec3<i32>(1);
  lim[k] = n[k];
  return all(c <= lim);
}

@compute @workgroup_size(8, 8, 4)
fn sl(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c > n)) { return; }
  var out = vec4<f32>(0.0);
  for (var k = 0u; k < 3u; k++) {
    if (!in_faces(c, k, n)) { continue; }
    let p = face_pos(c, k);
    out[k] = comp(src, trace(p, n, 1.0), k, n);
  }
  textureStore(dst, c, out);
}

@compute @workgroup_size(8, 8, 4)
fn mc(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c > n)) { return; }
  var out = vec4<f32>(0.0);
  for (var k = 0u; k < 3u; k++) {
    if (!in_faces(c, k, n)) { continue; }
    let p = face_pos(c, k);
    let phi = textureLoad(src, c, 0)[k];
    let hat = textureLoad(src2, c, 0)[k];
    let fwd = comp(src2, trace(p, n, -1.0), k, n);
    var r = hat + 0.5 * (phi - fwd);
    // clamp to the faces round where it came from
    let q = trace(p, n, 1.0);
    var o = vec3<f32>(0.5);
    o[k] = 0.0;
    var lim = n - vec3<i32>(1);
    lim[k] = n[k];
    let i = vec3<i32>(floor(clamp(q - o, vec3<f32>(0.0), vec3<f32>(lim))));
    var lo = 1.0e9;
    var hi = -1.0e9;
    for (var m = 0; m < 8; m++) {
      let f = min(i + vec3<i32>(m & 1, (m >> 1) & 1, m >> 2), lim);
      let v = textureLoad(src, f, 0)[k];
      lo = min(lo, v);
      hi = max(hi, v);
    }
    let nf = vec3<f32>(n);
    if (U.g.bc.x > 0.5 && (q.x < 0.0 || q.x > nf.x || q.z < 0.0 || q.z > nf.z)) {
      let e = wind0(q.y, n)[k];
      lo = min(lo, e);
      hi = max(hi, e);
    }
    out[k] = clamp(r, lo, hi);
  }
  textureStore(dst, c, out);
}
