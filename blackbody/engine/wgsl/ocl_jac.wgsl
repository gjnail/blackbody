// One Jacobi sweep of the layer current's pressure (m^2/s): solid neighbours reflect it, and the
// layer's edge is open water (0).

struct Params { L: Layer, };

@group(0) @binding(0) var prs: texture_2d<f32>;
@group(0) @binding(1) var div: texture_2d<f32>;
@group(0) @binding(2) var dep: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<r32float, write>;
@group(1) @binding(0) var<uniform> U: Params;
//!include ocl_common.wgsl

fn p_at(c: vec2<i32>, self_p: f32) -> f32 {
  let n = lay_n();
  if (any(c < vec2<i32>(0)) || any(c >= vec2<i32>(n))) { return 0.0; }
  if (lay_blocked(textureLoad(dep, c, 0))) { return self_p; }
  return textureLoad(prs, c, 0).x;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec2<i32>(id.xy);
  if (c.x >= lay_n() || c.y >= lay_n()) { return; }
  let p0 = textureLoad(prs, c, 0).x;
  let s = p_at(c + vec2<i32>(1, 0), p0) + p_at(c - vec2<i32>(1, 0), p0) + p_at(c + vec2<i32>(0, 1), p0) + p_at(c - vec2<i32>(0, 1), p0);
  let dl = U.L.map.z;
  textureStore(dst, c, vec4<f32>(0.25 * (s - textureLoad(div, c, 0).x * dl * dl), 0.0, 0.0, 0.0));
}
