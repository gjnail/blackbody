// Gas <- grass: the wind's drag on the blades (strand_canopy.wgsl's area of them a cubic metre, a), taken out of the
// gas through the canopy: dv/dt = -Cd a |v| v / 2, done implicitly so it never turns the flow round. A long meadow
// (a few square metres of blade a cubic metre) halves a breeze through it within a second or so; the air over the grass
// runs on, so the wind blows across the top of a field and stalls inside it.
//!include common.wgsl

const AREA_K: f32 = 65536.0;
const CD: f32 = 0.3;           // the drag coefficient of a grass blade (flexible: it bends with the flow)

struct Params {
  g: vec4<f32>,      // simulation grid dims, cell size (m)
  cg: vec4<f32>,     // coupling grid dims, its cell size (m)
  k: vec4<f32>,      // dt (s), _
};

@group(0) @binding(0) var vel: texture_3d<f32>;
@group(0) @binding(1) var<storage, read> CA: array<i32>;
@group(0) @binding(2) var dst: texture_storage_3d<${VELFMT}, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn area_at(p: vec3<f32>) -> f32 {
  let d = vec3<i32>(U.cg.xyz);
  let q = p * (U.g.w / U.cg.w) - vec3<f32>(0.5);
  let b = vec3<i32>(floor(q));
  let f = q - floor(q);
  var a = 0.0;
  for (var k = 0; k < 8; k++) {
    let o = vec3<i32>(k & 1, (k >> 1) & 1, k >> 2);
    let c = b + o;
    if (any(c < vec3<i32>(0)) || any(c >= d)) { continue; }
    let w3 = mix(vec3<f32>(1.0) - f, f, vec3<f32>(o));
    a += w3.x * w3.y * w3.z * f32(CA[u32((c.z * d.y + c.y) * d.x + c.x)]);
  }
  return a / AREA_K / (U.cg.w * U.cg.w * U.cg.w);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(U.g.xyz);
  let c = vec3<i32>(id);
  if (any(c > n)) { return; }
  var v = textureLoad(vel, c, 0);
  let pc = vec3<f32>(c) + vec3<f32>(0.5);
  let speed = length(v.xyz);
  // each face's own area (faces sit half a cell back along their axis)
  let ax = area_at(pc - vec3<f32>(0.5, 0.0, 0.0));
  let ay = area_at(pc - vec3<f32>(0.0, 0.5, 0.0));
  let az = area_at(pc - vec3<f32>(0.0, 0.0, 0.5));
  let k = 0.5 * CD * speed * U.k.x;
  v.x /= 1.0 + k * ax;
  v.y /= 1.0 + k * ay;
  v.z /= 1.0 + k * az;
  textureStore(dst, c, v);
}
