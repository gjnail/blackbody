// Looking up a burnable collider's surface in the object-burn atlas. Needs colliders.wgsl and
// burn_common.wgsl; the includer declares `burn_obj: texture_3d<f32>` and
// `slots: array<BurnSlot>` (storage).

// The atlas cell of collider k (region s) under world point w, or (-1, -1, -1) outside the region.
// (The region grows and shrinks with the collider: a burn laid out on it stays where it is on its
// surface when its Size is animated.)
fn obj_burn_cell(k: Collider, s: BurnSlot, w: vec3<f32>) -> vec3<i32> {
  let q = burn_to_layout(col_to_local(k, w), col_scale(k), s);
  let ci = vec3<i32>(floor((q - s.lo.xyz) / s.cell.xyz));
  if (any(ci < vec3<i32>(0)) || any(ci >= vec3<i32>(s.dims.xyz))) { return vec3<i32>(-1); }
  return vec3<i32>(ci.x, ci.y, ci.z + i32(s.lo.w));
}

fn obj_burn_at(k: Collider, s: BurnSlot, w: vec3<f32>) -> vec4<f32> {
  let ci = obj_burn_cell(k, s, w);
  if (ci.x < 0) { return vec4<f32>(0.0); }
  return textureLoad(burn_obj, ci, 0);
}

// The spot on collider k's surface at world point w (outward normal n): looked up `off` (m) out from it
// at the size the region was laid out at, and as much further as the collider has grown since (the layer
// the burn lives in grows and shrinks with it).
fn obj_burn_on(k: Collider, s: BurnSlot, w: vec3<f32>, n: vec3<f32>, off: f32) -> vec4<f32> {
  return obj_burn_at(k, s, w + n * (off * burn_grown(col_scale(k), s)));
}
