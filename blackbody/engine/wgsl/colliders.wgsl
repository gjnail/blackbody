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
  s: vec4<f32>,   // deforming mesh: next frame's atlas z offset (negative: none), blend, frames per second, _
  r: vec4<f32>,   // orientation quaternion (x, y, z, w), applied after the yaw: a tumbling object (identity: none)
  o: vec4<f32>,   // angular velocity (radians/s, world axes), on top of the spin about y
};

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}

// World point w in collider k's own frame (object space, metres): undo its orientation, then its yaw.
fn col_to_local(k: Collider, w: vec3<f32>) -> vec3<f32> {
  return yaw_to_local(quat_rotate(vec4<f32>(-k.r.xyz, k.r.w), w - k.a.xyz), k.b.w);
}

// How big collider k is along its own axes (m): what its shape is scaled by. Things kept on its
// surface in coordinates divided by this (its soot) grow and shrink with it.
fn col_scale(k: Collider) -> vec3<f32> {
  let shape = i32(k.a.w + 0.5);
  if (shape == 0) { return vec3<f32>(max(k.b.x, 1e-4)); }
  if (shape == 2) { return max(vec3<f32>(k.b.x, k.b.y, k.b.x), vec3<f32>(1e-4)); }
  return max(k.b.xyz, vec3<f32>(1e-4));
}

// A point in collider k's own frame (object space, metres), in the world.
fn col_to_world(k: Collider, q: vec3<f32>) -> vec3<f32> {
  return k.a.xyz + col_to_world_dir(k, q);
}

// A direction in collider k's own frame, in world axes.
fn col_to_world_dir(k: Collider, d: vec3<f32>) -> vec3<f32> {
  return quat_rotate(k.r, yaw_to_world(d, k.b.w));
}

fn box_sdf(q: vec3<f32>, r: vec3<f32>) -> f32 {
  let d = abs(q) - r;
  return length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
}

// The collider's own shape, before hollowing and openings.
fn col_shape_sdf(k: Collider, q: vec3<f32>, w: vec3<f32>) -> f32 {
  let shape = i32(k.a.w + 0.5);
  if (shape == 3) { return placed_mesh_sdf_local(q, k.b.xyz, k.m0, k.m1, k.m2, k.s); }
  if (shape == 0) { return length(q) - k.b.x; }
  if (shape == 1) { return box_sdf(q, k.b.xyz); }
  let d = vec2<f32>(length(q.xz) - k.b.x, abs(q.y) - k.b.y);
  return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

fn col_sdf(k: Collider, w: vec3<f32>) -> f32 {
  let q = col_to_local(k, w);
  var d = col_shape_sdf(k, q, w);
  // hollow: only a wall of this thickness inside the surface is solid (a room, a tank, a pipe)
  if (k.x.w > 0.0) { d = max(d, -(d + k.x.w)); }
  // an opening cut through it (a door, a window, a vent)
  if (all(k.y.xyz > vec3<f32>(0.0))) { d = max(d, -box_sdf(q - k.x.xyz, k.y.xyz)); }
  return d;
}

// Velocity of the collider's surface at world point w: its own motion, its spin about y and its
// tumbling, and for a deforming mesh how fast its surface moves as it changes shape.
fn col_velocity(k: Collider, w: vec3<f32>) -> vec3<f32> {
  let r = w - k.a.xyz;
  var v = k.v.xyz + k.v.w * vec3<f32>(r.z, 0.0, -r.x) + cross(k.o.xyz, r);
  if (k.s.x >= 0.0 && i32(k.a.w + 0.5) == 3) {
    v += col_to_world_dir(k, mesh_deform_velocity_local(col_to_local(k, w), k.b.xyz, k.m0, k.m1, k.m2, k.s));
  }
  return v;
}
