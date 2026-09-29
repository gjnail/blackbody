// Looking up a burnable collider's surface in the object-burn atlas. Needs colliders.wgsl and
// burn_common.wgsl; the includer declares `burn_obj: texture_3d<f32>` and
// `slots: array<BurnSlot>` (storage).

// The atlas cell of collider k (region s) under world point w, or (-1, -1, -1) outside the region.
fn obj_burn_cell(k: Collider, s: BurnSlot, w: vec3<f32>) -> vec3<i32> {
  let q = yaw_to_local(w - k.a.xyz, k.b.w);
  let ci = vec3<i32>(floor((q - s.lo.xyz) / s.dims.w));
  if (any(ci < vec3<i32>(0)) || any(ci >= vec3<i32>(s.dims.xyz))) { return vec3<i32>(-1); }
  return vec3<i32>(ci.x, ci.y, ci.z + i32(s.lo.w));
}

fn obj_burn_at(k: Collider, s: BurnSlot, w: vec3<f32>) -> vec4<f32> {
  let ci = obj_burn_cell(k, s, w);
  if (ci.x < 0) { return vec4<f32>(0.0); }
  return textureLoad(burn_obj, ci, 0);
}
