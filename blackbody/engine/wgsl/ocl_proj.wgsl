// The layer current made free of divergence: less the pressure's gradient; still in solids; the box's
// own flow kept inside the box.

struct Params { L: Layer, };

@group(0) @binding(0) var vel: texture_2d<f32>;
@group(0) @binding(1) var prs: texture_2d<f32>;
@group(0) @binding(2) var dep: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rg32float, write>;
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
  if (lay_blocked(textureLoad(dep, c, 0))) {
    textureStore(dst, c, vec4<f32>(0.0));
    return;
  }
  let p0 = textureLoad(prs, c, 0).x;
  let g = vec2<f32>(p_at(c + vec2<i32>(1, 0), p0) - p_at(c - vec2<i32>(1, 0), p0),
                    p_at(c + vec2<i32>(0, 1), p0) - p_at(c - vec2<i32>(0, 1), p0)) / (2.0 * U.L.map.z);
  let v = textureLoad(vel, c, 0).xy;
  let bs = box_share(lay_pos(c));
  textureStore(dst, c, vec4<f32>(v - g * (1.0 - bs), 0.0, 0.0));
}
