// Subtract the pressure gradient from face velocities. Faces on closed walls stay zero; faces
// touching solids keep the collider's surface velocity set by the force pass.
// Open boundaries hold p = 0 on the boundary face (ghost = -p inside), matching the solver, so gas
// flows freely in and out.
//!include common.wgsl

struct Params { g: Grid };

@group(0) @binding(0) var vel: texture_3d<f32>;
@group(0) @binding(1) var P: texture_3d<f32>;
@group(0) @binding(2) var sdf: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<${VELFMT}, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn solid(c: vec3<i32>, d: vec3<i32>) -> bool {
  if (!in_grid(c, d)) { return false; }
  return textureLoad(sdf, c, 0).x < 0.0;
}

fn pr(c: vec3<i32>, d: vec3<i32>) -> f32 {
  if (in_grid(c, d)) { return textureLoad(P, c, 0).x; }
  return 0.0;
}

// Pressure difference across the face between cells a and b (b - a); one may be a ghost.
fn dp(a: vec3<i32>, b: vec3<i32>, d: vec3<i32>) -> f32 {
  let ia = in_grid(a, d);
  let ib = in_grid(b, d);
  let pa = pr(a, d);
  let pb = pr(b, d);
  if (!ia) { return 2.0 * pb; }
  if (!ib) { return -2.0 * pa; }
  return pb - pa;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let d = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c > d)) { return; }
  var v = textureLoad(vel, c, 0);
  let ih = 1.0 / U.g.n.w;

  if (c.y < d.y && c.z < d.z) {
    let a = c - vec3<i32>(1, 0, 0);
    let wall = (c.x == 0 || c.x == d.x) && U.g.bc.x < 0.5;
    if (wall) { v.x = 0.0; } else if (!(solid(a, d) || solid(c, d))) { v.x -= dp(a, c, d) * ih; }
  } else { v.x = 0.0; }

  if (c.x < d.x && c.z < d.z) {
    let a = c - vec3<i32>(0, 1, 0);
    let wall = (c.y == 0 && U.g.bc.z < 0.5) || (c.y == d.y && U.g.bc.y < 0.5);
    if (wall) { v.y = 0.0; } else if (!(solid(a, d) || solid(c, d))) { v.y -= dp(a, c, d) * ih; }
  } else { v.y = 0.0; }

  if (c.x < d.x && c.y < d.y) {
    let a = c - vec3<i32>(0, 0, 1);
    let wall = (c.z == 0 || c.z == d.z) && U.g.bc.x < 0.5;
    if (wall) { v.z = 0.0; } else if (!(solid(a, d) || solid(c, d))) { v.z -= dp(a, c, d) * ih; }
  } else { v.z = 0.0; }

  textureStore(dst, c, vec4<f32>(v.xyz, 0.0));
}
