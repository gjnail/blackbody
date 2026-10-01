// Drops of water dripping off wet cloth (cloth_drip.wgsl lets them go): they fall, held back by the air
// (a 5 mm drop falls no faster than about 9 m/s, and wind carries it), until they land on the floor or
// an object, or fall into a liquid.
//!include common.wgsl
//!include meshsdf.wgsl
//!include colliders.wgsl

const RHO_W: f32 = 1000.0;
const RHO_AIR: f32 = 1.2;
const CD: f32 = 0.6;           // drag coefficient of a small falling drop

struct Params {
  a: vec4<f32>,      // dt, drops (capacity), ground (1/0), gravity (m/s^2)
  wind: vec4<f32>,   // wind (fire-local m/s), _
  lq: vec4<f32>,     // the liquid's grid: dims, cell size (m)
  lo: vec4<f32>,     // its corner (fire-local m), liquid on (1/0)
  ccnt: vec4<f32>,
  col: array<Collider, MAX_COLLIDERS>,
};

struct Drops {
  head: u32,
  p1: u32,
  p2: u32,
  p3: u32,
  d: array<vec4<f32>>,   // per drop: position, life left (s); velocity, radius (m)
};

@group(0) @binding(0) var<storage, read_write> DR: Drops;
@group(0) @binding(1) var atlas: texture_3d<f32>;
@group(0) @binding(2) var LTYPE: texture_3d<f32>;
@group(1) @binding(0) var<uniform> U: Params;

fn in_liquid(p: vec3<f32>) -> bool {
  if (U.lo.w < 0.5) { return false; }
  let n = vec3<i32>(U.lq.xyz);
  let c = vec3<i32>(floor((p - U.lo.xyz) / U.lq.w));
  if (any(c < vec3<i32>(0)) || any(c >= n)) { return false; }
  let t = textureLoad(LTYPE, c, 0).x;
  return t > 0.5 && t < 1.5;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = id.x;
  if (k >= u32(U.a.y)) { return; }
  let p = DR.d[2u * k];
  if (p.w <= 0.0) { return; }
  let v4 = DR.d[2u * k + 1u];
  let r = v4.w;
  let dt = U.a.x;
  let g = U.a.w;
  // gravity, then the air's drag (quadratic, solved implicitly so it settles at the terminal speed)
  let vt2 = 8.0 / 3.0 * (RHO_W / RHO_AIR) * g * r / CD;
  var v = v4.xyz + vec3<f32>(0.0, -g * dt, 0.0);
  let rel = v - U.wind.xyz;
  v = U.wind.xyz + rel / (1.0 + g * length(rel) / vt2 * dt);
  let x = p.xyz + v * dt;
  var life = p.w - dt;
  if (U.a.z > 0.5 && x.y < r) { life = 0.0; }
  for (var c = 0; c < i32(U.ccnt.x); c++) {
    if (col_sdf(U.col[c], x) < r) { life = 0.0; }
  }
  if (in_liquid(x)) { life = 0.0; }
  DR.d[2u * k] = vec4<f32>(x, life);
  DR.d[2u * k + 1u] = vec4<f32>(v, r);
}
