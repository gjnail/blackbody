// The layer current's divergence (how much it piles water up or draws it away), per cell (1/s).

struct Params { L: Layer, };

@group(0) @binding(0) var vel: texture_2d<f32>;
@group(0) @binding(1) var dep: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<r32float, write>;
@group(1) @binding(0) var<uniform> U: Params;
//!include ocl_common.wgsl

fn vel_at(c: vec2<i32>) -> vec2<f32> {
  let n = lay_n();
  let cc = clamp(c, vec2<i32>(0), vec2<i32>(n - 1));
  if (lay_blocked(textureLoad(dep, cc, 0))) { return vec2<f32>(0.0); }
  return textureLoad(vel, cc, 0).xy;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec2<i32>(id.xy);
  if (c.x >= lay_n() || c.y >= lay_n()) { return; }
  let d = (vel_at(c + vec2<i32>(1, 0)).x - vel_at(c - vec2<i32>(1, 0)).x + vel_at(c + vec2<i32>(0, 1)).y - vel_at(c - vec2<i32>(0, 1)).y)
          / (2.0 * U.L.map.z);
  textureStore(dst, c, vec4<f32>(d, 0.0, 0.0, 0.0));
}
