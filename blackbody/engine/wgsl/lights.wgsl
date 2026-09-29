// A short list of point lights standing in for the fire, for lighting surfaces: the half-resolution
// emission volume is cut into blocks, and every block that glows becomes a light at its
// emission-weighted centre with its total power.
//!include common.wgsl

struct Params {
  ln: vec4<f32>,    // emission volume dims; w = its cell size (m)
  org: vec4<f32>,   // grid corner (fire-local m); w = block size (cells)
};

@group(0) @binding(0) var E: texture_3d<f32>;
@group(0) @binding(1) var<storage, read_write> lights: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> light_count: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let b = i32(U.org.w);
  let dims = vec3<i32>(U.ln.xyz);
  let lo = vec3<i32>(id) * b;
  if (any(lo >= dims)) { return; }
  let hi = min(lo + vec3<i32>(b), dims);
  let cell = U.ln.w;
  var power = vec3<f32>(0.0);
  var centre = vec3<f32>(0.0);
  var weight = 0.0;
  for (var z = lo.z; z < hi.z; z++) {
    for (var y = lo.y; y < hi.y; y++) {
      for (var x = lo.x; x < hi.x; x++) {
        let e = textureLoad(E, vec3<i32>(x, y, z), 0).rgb;
        let l = luma(e);
        if (l <= 0.0) { continue; }
        power += e;
        centre += (vec3<f32>(f32(x), f32(y), f32(z)) + 0.5) * l;
        weight += l;
      }
    }
  }
  if (weight <= 1e-6) { return; }
  let vol = cell * cell * cell;
  let i = atomicAdd(&light_count[0], 1u);
  if (2u * i + 1u >= arrayLength(&lights)) { return; }
  lights[2u * i] = vec4<f32>(U.org.xyz + centre / weight * cell, 0.5 * f32(b) * cell);
  lights[2u * i + 1u] = vec4<f32>(power * vol, 0.0);
}
