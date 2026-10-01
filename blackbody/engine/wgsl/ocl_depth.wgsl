// The sea bed under the open-water layer: per layer cell, how deep the water is (from the level down
// to the first solid: a collider, a terrain, the ground) and how far the nearest solid is at the water
// line (negative inside: an island, a pier, a hull). Recomputed when the colliders change.
//!include meshsdf.wgsl
//!include colliders.wgsl

struct Params {
  L: Layer,
  k: vec4<f32>,       // level (m), the ground is the sea bed (1/0), depth where nothing is found (m), _
  ccnt: vec4<f32>,
  col: array<Collider, MAX_COLLIDERS>,
};

@group(0) @binding(0) var atlas: texture_3d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;
//!include ocl_common.wgsl

fn solid_at(w: vec3<f32>) -> f32 {
  var d = 1.0e6;
  for (var i = 0; i < i32(U.ccnt.x); i++) { d = min(d, col_sdf(U.col[i], w)); }
  if (U.k.y > 0.5) { d = min(d, w.y); }
  return d;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec2<i32>(id.xy);
  if (c.x >= lay_n() || c.y >= lay_n()) { return; }
  let x = lay_pos(c);
  let level = U.k.x;
  let at_level = solid_at(vec3<f32>(x.x, level, x.y));
  var depth = U.k.z;
  if (at_level <= 0.0) {
    depth = 0.0;
  } else {
    // down from the water line to the first solid
    var y = level;
    let floor_y = level - U.k.z;
    for (var i = 0; i < 64; i++) {
      let d = solid_at(vec3<f32>(x.x, y, x.y));
      if (d < 0.01) { break; }
      y -= max(d, 0.01 + 0.004 * (level - y));
      if (y <= floor_y) { y = floor_y; break; }
    }
    depth = clamp(level - y, 0.0, U.k.z);
  }
  textureStore(dst, c, vec4<f32>(depth, at_level, 0.0, 0.0));
}
