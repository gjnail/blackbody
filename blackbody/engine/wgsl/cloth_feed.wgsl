// Gas <- burning fabric: the fuel, heat and smoke the burning cloth gives off (gathered on the
// coupling grid by cloth_splat.wgsl) go into the gas, on the simulation grid or its finer upres grid.
//!include common.wgsl

const FUEL_K: f32 = 1.0e9;
const HEAT_K: f32 = 1.0e4;

struct Params {
  t: vec4<f32>,      // target grid dims, its cell size (m)
  cg: vec4<f32>,     // coupling grid dims, its cell size (m)
  k: vec4<f32>,      // dt, fuel density at which the heat is fully there (F/s), _, _
};

@group(0) @binding(0) var src: texture_3d<f32>;
@group(0) @binding(1) var<storage, read> G: array<i32>;
@group(0) @binding(2) var dst: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn gat(c: vec3<i32>, ch: u32) -> f32 {
  let d = vec3<i32>(U.cg.xyz);
  if (any(c < vec3<i32>(0)) || any(c >= d)) { return 0.0; }
  return f32(G[u32((c.z * d.y + c.y) * d.x + c.x) * 13u + ch]);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(U.t.xyz);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  var s = textureLoad(src, c, 0);
  let q = (vec3<f32>(c) + vec3<f32>(0.5)) * (U.t.w / U.cg.w) - vec3<f32>(0.5);
  let b = vec3<i32>(floor(q));
  let f = q - floor(q);
  var fuel = 0.0;
  var smoke = 0.0;
  var heat = 0.0;
  for (var k = 0; k < 8; k++) {
    let o = vec3<i32>(k & 1, (k >> 1) & 1, k >> 2);
    let w3 = mix(vec3<f32>(1.0) - f, f, vec3<f32>(o));
    let w = w3.x * w3.y * w3.z;
    fuel += w * gat(b + o, 6u);
    smoke += w * gat(b + o, 8u);
    heat = max(heat, gat(b + o, 7u));
  }
  if (fuel > 0.0) {
    let vol = U.cg.w * U.cg.w * U.cg.w;
    let rate = fuel / FUEL_K / vol;          // F per second here
    let dt = U.k.x;
    s.y += rate * dt;
    s.z += smoke / FUEL_K / vol * dt;
    s.x = max(s.x, heat / HEAT_K * clamp(rate / U.k.y, 0.0, 1.0));
  }
  textureStore(dst, c, s);
}
