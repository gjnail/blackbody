// Mesh signed distance fields. Every mesh in use is baked once into a distance grid in mesh units,
// and all of them share one r32float atlas: stacked along z in columns that stand side by side in x
// and y (engine/mesh.py pack_atlas). The including file declares `atlas: texture_3d<f32>` (a 1x1x1
// texture when no mesh is in use).
//
// A mesh reference is three vec4s: m0 = bounding box min, atlas code (where its grid starts in the
// atlas, atlas_org; negative: no mesh); m1 = bounding box max, _; m2 = grid dims (cells), _. A
// deforming mesh (a numbered sequence or an animated USD prim) adds an = (atlas code of the next frame
// (negative: the mesh does not deform), blend toward it (0..1), frames per second, _): both frames
// share one grid.

const ATLAS_TILE: i32 = 32;     // matches mesh.ATLAS_TILE: columns start on this grid in x and y
const ATLAS_TILES: i32 = 64;    // matches mesh.ATLAS_TILES
const ATLAS_ZSPAN: i32 = 2048;  // matches mesh.ATLAS_ZSPAN

// The atlas cell a grid starts at, from its atlas code: z + ATLAS_ZSPAN * (x tile + ATLAS_TILES * y tile).
fn atlas_org(code: f32) -> vec3<i32> {
  let k = i32(round(code));
  let col = k / ATLAS_ZSPAN;
  return vec3<i32>((col % ATLAS_TILES) * ATLAS_TILE, (col / ATLAS_TILES) * ATLAS_TILE, k - col * ATLAS_ZSPAN);
}

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

// A grid's cell c (clamped to the grid), the grid starting at atlas cell org.
fn atlas_at(c: vec3<i32>, dims: vec3<i32>, org: vec3<i32>) -> f32 {
  let q = clamp(c, vec3<i32>(0), dims - vec3<i32>(1));
  return textureLoad(atlas, q + org, 0).x;
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
  let z = atlas_org(m0.w);
  let c00 = mix(atlas_at(i0, dims, z), atlas_at(i0 + vec3<i32>(1, 0, 0), dims, z), f.x);
  let c10 = mix(atlas_at(i0 + vec3<i32>(0, 1, 0), dims, z), atlas_at(i0 + vec3<i32>(1, 1, 0), dims, z), f.x);
  let c01 = mix(atlas_at(i0 + vec3<i32>(0, 0, 1), dims, z), atlas_at(i0 + vec3<i32>(1, 0, 1), dims, z), f.x);
  let c11 = mix(atlas_at(i0 + vec3<i32>(0, 1, 1), dims, z), atlas_at(i0 + vec3<i32>(1, 1, 1), dims, z), f.x);
  let d = mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
  return d + outside;
}

// A deforming mesh between two of its frames.
fn mesh_sdf_anim(q: vec3<f32>, m0: vec4<f32>, m1: vec4<f32>, m2: vec4<f32>, an: vec4<f32>) -> f32 {
  let d0 = mesh_sdf(q, m0, m1, m2);
  if (an.x < 0.0 || m0.w < 0.0 || an.y <= 0.0) { return d0; }
  let d1 = mesh_sdf(q, vec4<f32>(m0.xyz, an.x), m1, m2);
  return mix(d0, d1, an.y);
}

// Mesh placed in the world: position p0, per-axis scale s, rotation about y. Metres.
fn placed_mesh_sdf(w: vec3<f32>, p0: vec3<f32>, s: vec3<f32>, yaw: f32, m0: vec4<f32>, m1: vec4<f32>, m2: vec4<f32>,
                   an: vec4<f32>) -> f32 {
  let sc = max(s, vec3<f32>(1e-4));
  let q = yaw_to_local(w - p0, yaw) / sc;
  return mesh_sdf_anim(q, m0, m1, m2, an) * min(sc.x, min(sc.y, sc.z));
}

// The same, for a point already in the object's own frame (metres, before scaling): for placements
// that turn it by more than a yaw (tumbling colliders).
fn placed_mesh_sdf_local(ql: vec3<f32>, s: vec3<f32>, m0: vec4<f32>, m1: vec4<f32>, m2: vec4<f32>, an: vec4<f32>) -> f32 {
  let sc = max(s, vec3<f32>(1e-4));
  return mesh_sdf_anim(ql / sc, m0, m1, m2, an) * min(sc.x, min(sc.y, sc.z));
}

// A deforming mesh's surface velocity at a point in its own frame, returned in its own frame (the
// caller turns it to world).
fn mesh_deform_velocity_local(ql: vec3<f32>, s: vec3<f32>, m0: vec4<f32>, m1: vec4<f32>, m2: vec4<f32>,
                              an: vec4<f32>) -> vec3<f32> {
  return mesh_deform_velocity(ql, vec3<f32>(0.0), s, 0.0, m0, m1, m2, an);
}

// How fast a deforming mesh's surface moves at world point w (m/s, along the surface normal): the
// change of distance between its two frames, times the frame rate.
fn mesh_deform_velocity(w: vec3<f32>, p0: vec3<f32>, s: vec3<f32>, yaw: f32, m0: vec4<f32>, m1: vec4<f32>,
                        m2: vec4<f32>, an: vec4<f32>) -> vec3<f32> {
  if (an.x < 0.0 || m0.w < 0.0 || an.z <= 0.0) { return vec3<f32>(0.0); }
  let sc = max(s, vec3<f32>(1e-4));
  let q = yaw_to_local(w - p0, yaw) / sc;
  let k = min(sc.x, min(sc.y, sc.z));
  let m0b = vec4<f32>(m0.xyz, an.x);
  let rate = (mesh_sdf(q, m0b, m1, m2) - mesh_sdf(q, m0, m1, m2)) * k * an.z;  // m/s the distance grows
  let e = 0.75 * (m1.xyz - m0.xyz) / max(m2.xyz, vec3<f32>(1.0));
  let g = vec3<f32>(
    mesh_sdf_anim(q + vec3<f32>(e.x, 0.0, 0.0), m0, m1, m2, an) - mesh_sdf_anim(q - vec3<f32>(e.x, 0.0, 0.0), m0, m1, m2, an),
    mesh_sdf_anim(q + vec3<f32>(0.0, e.y, 0.0), m0, m1, m2, an) - mesh_sdf_anim(q - vec3<f32>(0.0, e.y, 0.0), m0, m1, m2, an),
    mesh_sdf_anim(q + vec3<f32>(0.0, 0.0, e.z), m0, m1, m2, an) - mesh_sdf_anim(q - vec3<f32>(0.0, 0.0, e.z), m0, m1, m2, an)) / (2.0 * e);
  let gw = yaw_to_world(g / sc, yaw);
  let l = length(gw);
  if (l < 1e-6) { return vec3<f32>(0.0); }
  // the surface moves out (toward the gas) where the distance shrinks
  return -(gw / l) * clamp(rate, -20.0, 20.0);
}
