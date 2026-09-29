// Collider shapes and motion, shared by the SDF, force, surface, particle and render kernels.
// Needs meshsdf.wgsl.

const MAX_COLLIDERS: u32 = 16u;  // matches solver.MAX_COLLIDERS

struct Collider {
  a: vec4<f32>,   // position (world m), shape id (0 sphere, 1 box, 2 cylinder, 3 mesh)
  b: vec4<f32>,   // size (m): radius or half extents, or the mesh scale; w = rotation about y (radians)
  v: vec4<f32>,   // velocity (m/s); w = spin about y (radians/s)
  m0: vec4<f32>,  // mesh: bounding box min, atlas z offset (negative: no mesh)
  m1: vec4<f32>,  // mesh: bounding box max; w = burnable (1/0)
  m2: vec4<f32>,  // mesh: grid dims (cells); w = burn atlas slot (negative: none)
  x: vec4<f32>,   // opening centre (object space, m); w = hollow wall thickness (m, 0 = solid)
  y: vec4<f32>,   // opening half size (m, 0 = none); w = hides the fire behind it (1/0)
};

fn box_sdf(q: vec3<f32>, r: vec3<f32>) -> f32 {
  let d = abs(q) - r;
  return length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
}

// The collider's own shape, before hollowing and openings.
fn col_shape_sdf(k: Collider, q: vec3<f32>, w: vec3<f32>) -> f32 {
  let shape = i32(k.a.w + 0.5);
  if (shape == 3) { return placed_mesh_sdf(w, k.a.xyz, k.b.xyz, k.b.w, k.m0, k.m1, k.m2); }
  if (shape == 0) { return length(q) - k.b.x; }
  if (shape == 1) { return box_sdf(q, k.b.xyz); }
  let d = vec2<f32>(length(q.xz) - k.b.x, abs(q.y) - k.b.y);
  return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

fn col_sdf(k: Collider, w: vec3<f32>) -> f32 {
  let q = yaw_to_local(w - k.a.xyz, k.b.w);
  var d = col_shape_sdf(k, q, w);
  // hollow: only a wall of this thickness inside the surface is solid (a room, a tank, a pipe)
  if (k.x.w > 0.0) { d = max(d, -(d + k.x.w)); }
  // an opening cut through it (a door, a window, a vent)
  if (all(k.y.xyz > vec3<f32>(0.0))) { d = max(d, -box_sdf(q - k.x.xyz, k.y.xyz)); }
  return d;
}

// Velocity of the collider's surface at world point w: its own motion plus its spin about y.
fn col_velocity(k: Collider, w: vec3<f32>) -> vec3<f32> {
  let r = w - k.a.xyz;
  return k.v.xyz + k.v.w * vec3<f32>(r.z, 0.0, -r.x);
}
