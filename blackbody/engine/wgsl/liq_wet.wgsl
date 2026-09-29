// Wetness of the ground under the domain: liquid in the bottom cells wets it, then it dries.
//!include common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,  // density that counts as touching (particles), drying rate (1/s)
};

@group(0) @binding(0) var dens: texture_3d<f32>;
@group(0) @binding(1) var wet: texture_storage_2d<r32float, read_write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec2<i32>(id.xy);
  if (c.x >= n.x || c.y >= n.z) { return; }
  let d0 = textureLoad(dens, vec3<i32>(c.x, 0, c.y), 0).x;
  let d1 = textureLoad(dens, vec3<i32>(c.x, 1, c.y), 0).x;
  let touch = clamp(max(d0, 0.5 * d1) / max(U.k.x, 1e-3), 0.0, 1.0);
  let w = textureLoad(wet, c).x * exp(-U.k.y * U.g.bc.w);
  textureStore(wet, c, vec4<f32>(max(w, touch), 0.0, 0.0, 0.0));
}
