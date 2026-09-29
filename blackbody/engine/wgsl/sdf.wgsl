// Collider signed distance at cell centres, in cells (negative inside a solid).
// Rewritten every substep while a collider is animated.
//!include common.wgsl
//!include meshsdf.wgsl
//!include colliders.wgsl

struct Params {
  g: Grid,
  cnt: vec4<f32>,
  col: array<Collider, MAX_COLLIDERS>,
};

@group(0) @binding(0) var atlas: texture_3d<f32>;
@group(0) @binding(1) var dst: texture_storage_3d<r32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let d = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= d)) { return; }
  let w = world_of(U.g, vec3<f32>(c) + 0.5);
  var s = 1.0e6;
  let cnt = i32(U.cnt.x);
  for (var i = 0; i < cnt; i++) {
    s = min(s, col_sdf(U.col[i], w));
  }
  textureStore(dst, c, vec4<f32>(s / U.g.n.w, 0.0, 0.0, 0.0));
}
