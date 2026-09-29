// Mesh signed distance fields. Every mesh in use is baked once into a distance grid in mesh units,
// and all of them are stacked along z in one r32float atlas. The including file declares
// `atlas: texture_3d<f32>` (a 1x1x1 texture when no mesh is in use).
//
// A mesh reference is three vec4s: m0 = bounding box min, atlas z offset (negative: no mesh);
// m1 = bounding box max, _; m2 = grid dims (cells), _.

// Rotation about the vertical axis: world offset -> object space, and back.
fn yaw_to_local(r: vec3<f32>, yaw: f32) -> vec3<f32> {
  let cs = cos(yaw);
  let sn = sin(yaw);
  return vec3<f32>(cs * r.x - sn * r.z, r.y, sn * r.x + cs * r.z);
}

fn yaw_to_world(q: vec3<f32>, yaw: f32) -> vec3<f32> {
  let cs = cos(yaw);
  let sn = sin(yaw);
  return vec3<f32>(cs * q.x + sn * q.z, q.y, -sn * q.x + cs * q.z);
}

fn atlas_at(c: vec3<i32>, dims: vec3<i32>, zoff: i32) -> f32 {
  let q = clamp(c, vec3<i32>(0), dims - vec3<i32>(1));
  return textureLoad(atlas, vec3<i32>(q.x, q.y, q.z + zoff), 0).x;
}

// Signed distance (mesh units) at q (mesh space). Outside the baked box, the distance to the box
// is added to the value at its edge, which stays a good estimate far from the mesh.
fn mesh_sdf(q: vec3<f32>, m0: vec4<f32>, m1: vec4<f32>, m2: vec4<f32>) -> f32 {
  if (m0.w < 0.0) { return 1.0e9; }
  let dims = vec3<i32>(m2.xyz);
  let cell = (m1.xyz - m0.xyz) / max(m2.xyz, vec3<f32>(1.0));
  let qc = clamp(q, m0.xyz + 0.5 * cell, m1.xyz - 0.5 * cell);
  let outside = length(q - qc);
  let t = (qc - m0.xyz) / cell - vec3<f32>(0.5);
  let i0 = vec3<i32>(floor(t));
  let f = t - floor(t);
  let z = i32(m0.w);
  let c00 = mix(atlas_at(i0, dims, z), atlas_at(i0 + vec3<i32>(1, 0, 0), dims, z), f.x);
  let c10 = mix(atlas_at(i0 + vec3<i32>(0, 1, 0), dims, z), atlas_at(i0 + vec3<i32>(1, 1, 0), dims, z), f.x);
  let c01 = mix(atlas_at(i0 + vec3<i32>(0, 0, 1), dims, z), atlas_at(i0 + vec3<i32>(1, 0, 1), dims, z), f.x);
  let c11 = mix(atlas_at(i0 + vec3<i32>(0, 1, 1), dims, z), atlas_at(i0 + vec3<i32>(1, 1, 1), dims, z), f.x);
  let d = mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
  return d + outside;
}

// Mesh placed in the world: position p0, per-axis scale s, rotation about y. Metres.
fn placed_mesh_sdf(w: vec3<f32>, p0: vec3<f32>, s: vec3<f32>, yaw: f32, m0: vec4<f32>, m1: vec4<f32>, m2: vec4<f32>) -> f32 {
  let sc = max(s, vec3<f32>(1e-4));
  let q = yaw_to_local(w - p0, yaw) / sc;
  return mesh_sdf(q, m0, m1, m2) * min(sc.x, min(sc.y, sc.z));
}
