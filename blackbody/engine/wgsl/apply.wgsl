// Applies forces to face velocities: buoyancy, the cell-centred force field, wind, damping, emitter
// velocities and swirl. Faces on closed domain walls are zero; faces touching solids take the
// velocity of the collider's surface there (zero for still colliders), so moving colliders push
// the gas out of their way.
//!include common.wgsl
//!include noise.wgsl
//!include meshsdf.wgsl
//!include emitters.wgsl
//!include colliders.wgsl

struct Params {
  g: Grid,
  b: vec4<f32>,     // buoyancy (m/s^2 per unit temperature), soot weight (m/s^2 per unit smoke), damping (1/s), vapour lift (m/s^2 per g/m^3)
  b2: vec4<f32>,    // fuel weight (m/s^2 per unit fuel: heavier-than-air vapour hugs the ground), _, _, _
  wind: vec4<f32>,  // wind velocity (m/s), relaxation toward it (1/s)
  ccnt: vec4<f32>,  // collider count, any collider moving (1/0), broken pieces in svel (1/0)
  col: array<Collider, MAX_COLLIDERS>,
  cnt: vec4<f32>,   // emitter count, any swirl (1/0)
  em: array<Emitter, MAX_EMITTERS>,
};

const SWIRL_RATE: f32 = 3.0;  // 1/s, how quickly the air takes up an emitter's swirl

@group(0) @binding(0) var vel: texture_3d<f32>;
@group(0) @binding(1) var scal: texture_3d<f32>;
@group(0) @binding(2) var force: texture_3d<f32>;
@group(0) @binding(3) var sdf: texture_3d<f32>;
@group(0) @binding(4) var atlas: texture_3d<f32>;
@group(0) @binding(5) var aux: texture_3d<f32>;
@group(0) @binding(6) var dst: texture_storage_3d<${VELFMT}, write>;
@group(0) @binding(7) var svel: texture_3d<f32>;   // broken pieces' velocity in the cells inside them (w = 1; bodyfield.py)
@group(1) @binding(0) var<uniform> U: Params;

fn ld(t: texture_3d<f32>, c: vec3<i32>, d: vec3<i32>) -> vec4<f32> {
  return textureLoad(t, clamp(c, vec3<i32>(0), d - vec3<i32>(1)), 0);
}

// face velocity texel (the velocity texture is one larger than the grid on each axis)
fn vl(c: vec3<i32>, d: vec3<i32>) -> vec4<f32> {
  return textureLoad(vel, clamp(c, vec3<i32>(0), d), 0);
}

fn solid(c: vec3<i32>, d: vec3<i32>) -> bool {
  if (!in_grid(c, d)) { return false; }
  return textureLoad(sdf, c, 0).x < 0.0;
}

fn emit_vel(cur: f32, wp: vec3<f32>, axis: i32, h: f32) -> f32 {
  var v = cur;
  let cnt = i32(U.cnt.x);
  for (var i = 0; i < cnt; i++) {
    let e = U.em[i];
    if (emitter_vel_blend(e) <= 0.0) { continue; }
    let m = emitter_mask(e, wp, h);
    if (m <= 0.0) { continue; }
    let tv = emitter_vel_target(e, wp, m);
    var t = tv.z;
    if (axis == 0) { t = tv.x; } else if (axis == 1) { t = tv.y; }
    v = mix(v, t, tv.w);
  }
  return v;
}

// Relax the horizontal velocity toward each emitter's swirl. `vx`, `vz` are the full horizontal
// velocity at this face (one component is interpolated); returns the component the face holds.
fn swirl_vel(vx0: f32, vz0: f32, wp: vec3<f32>, axis: i32, dt: f32) -> f32 {
  var vx = vx0;
  var vz = vz0;
  let cnt = i32(U.cnt.x);
  for (var i = 0; i < cnt; i++) {
    let s = swirl_at(U.em[i], wp);
    if (s.w <= 0.0) { continue; }
    let cur = vx * s.x + vz * s.y;
    let dv = (s.z - cur) * (1.0 - exp(-SWIRL_RATE * s.w * dt));
    vx += s.x * dv;
    vz += s.y * dv;
  }
  return select(vz, vx, axis == 0);
}

// Velocity component `axis` of the solid surface at a face touching a collider.
fn solid_vel(wp: vec3<f32>, axis: i32) -> f32 {
  if (U.ccnt.y < 0.5) { return 0.0; }
  var best = 1.0e9;
  var v = vec3<f32>(0.0);
  let cnt = i32(U.ccnt.x);
  for (var i = 0; i < cnt; i++) {
    let d = col_sdf(U.col[i], wp);
    if (d < best) {
      best = d;
      v = col_velocity(U.col[i], wp);
    }
  }
  if (axis == 0) { return v.x; }
  if (axis == 1) { return v.y; }
  return v.z;
}

// Velocity component `axis` of the solid at the face between cells a and b: a broken piece's, where one of
// them is inside a piece, else the colliders'.
fn solid_vel_at(a: vec3<i32>, b: vec3<i32>, d: vec3<i32>, wp: vec3<f32>, axis: i32) -> f32 {
  if (U.ccnt.z > 0.5) {
    var pv = vec4<f32>(0.0);
    if (in_grid(a, d)) { pv = textureLoad(svel, a, 0); }
    if (pv.w < 0.5 && in_grid(b, d)) { pv = textureLoad(svel, b, 0); }
    if (pv.w > 0.5) { return select(select(pv.z, pv.y, axis == 1), pv.x, axis == 0); }
  }
  return solid_vel(wp, axis);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let d = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c > d)) { return; }
  let dt = U.g.bc.w;
  let h = U.g.n.w;
  var v = textureLoad(vel, c, 0);
  let relax = 1.0 - exp(-U.wind.w * dt);
  let damp = exp(-U.b.z * dt);
  let pc = vec3<f32>(c);
  let swirl = U.cnt.y > 0.5;

  // u: face between cells c - x and c
  if (c.y < d.y && c.z < d.z) {
    let a = c - vec3<i32>(1, 0, 0);
    let wall = (c.x == 0 || c.x == d.x) && U.g.bc.x < 0.5;
    let wp = world_of(U.g, pc + vec3<f32>(0.0, 0.5, 0.5));
    if (wall) {
      v.x = 0.0;
    } else if (solid(a, d) || solid(c, d)) {
      v.x = solid_vel_at(a, c, d, wp, 0);
    } else {
      v.x += dt * 0.5 * (ld(force, a, d).x + ld(force, c, d).x);
      v.x += (U.wind.x - v.x) * relax;
      v.x *= damp;
      if (swirl) {
        let wz = 0.25 * (vl(a, d).z + vl(a + vec3<i32>(0, 0, 1), d).z + vl(c, d).z + vl(c + vec3<i32>(0, 0, 1), d).z);
        v.x = swirl_vel(v.x, wz, wp, 0, dt);
      }
      v.x = emit_vel(v.x, wp, 0, h);
    }
  } else {
    v.x = 0.0;
  }

  // v: face between cells c - y and c; buoyancy acts here
  if (c.x < d.x && c.z < d.z) {
    let a = c - vec3<i32>(0, 1, 0);
    let wall = (c.y == 0 && U.g.bc.z < 0.5) || (c.y == d.y && U.g.bc.y < 0.5);
    let wp = world_of(U.g, pc + vec3<f32>(0.5, 0.0, 0.5));
    if (wall) {
      v.y = 0.0;
    } else if (solid(a, d) || solid(c, d)) {
      v.y = solid_vel_at(a, c, d, wp, 1);
    } else {
      let sa = ld(scal, a, d);
      let sb = ld(scal, c, d);
      let T = 0.5 * (sa.x + sb.x);
      let S = 0.5 * (sa.z + sb.z);
      let f = 0.5 * (ld(force, a, d).y + ld(force, c, d).y);
      var lift = U.b.x * T - U.b.y * S - U.b2.x * 0.5 * (sa.y + sb.y);
      if (U.b.w > 0.0) { lift += U.b.w * 0.5 * (ld(aux, a, d).y + ld(aux, c, d).y); }
      v.y += dt * (f + lift);
      v.y += (U.wind.y - v.y) * relax * 0.25;
      v.y *= damp;
      v.y = emit_vel(v.y, wp, 1, h);
    }
  } else {
    v.y = 0.0;
  }

  // w: face between cells c - z and c
  if (c.x < d.x && c.y < d.y) {
    let a = c - vec3<i32>(0, 0, 1);
    let wall = (c.z == 0 || c.z == d.z) && U.g.bc.x < 0.5;
    let wp = world_of(U.g, pc + vec3<f32>(0.5, 0.5, 0.0));
    if (wall) {
      v.z = 0.0;
    } else if (solid(a, d) || solid(c, d)) {
      v.z = solid_vel_at(a, c, d, wp, 2);
    } else {
      v.z += dt * 0.5 * (ld(force, a, d).z + ld(force, c, d).z);
      v.z += (U.wind.z - v.z) * relax;
      v.z *= damp;
      if (swirl) {
        let ux = 0.25 * (vl(a, d).x + vl(a + vec3<i32>(1, 0, 0), d).x + vl(c, d).x + vl(c + vec3<i32>(1, 0, 0), d).x);
        v.z = swirl_vel(ux, v.z, wp, 2, dt);
      }
      v.z = emit_vel(v.z, wp, 2, h);
    }
  } else {
    v.z = 0.0;
  }
  textureStore(dst, c, vec4<f32>(v.xyz, 0.0));
}
