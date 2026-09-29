// Floating objects: what the liquid does to one collider, gathered over the cells around it. For
// every solid cell of the body, each liquid neighbour presses on the face between them with its
// pressure (the solve's x = p dt / rho), which sums to the force (buoyancy, the push of waves and
// impacts) and its turning moment about the vertical axis; the liquid's velocity next to the body
// is averaged for the drift. Sums go to one slot per substep (fixed point, deterministic), read
// back once per frame.
//!include common.wgsl
//!include liq_common.wgsl
//!include meshsdf.wgsl
//!include colliders.wgsl

const FX_F: f32 = 65536.0;   // pressure sums
const FX_T: f32 = 4096.0;    // moment sums (cells x pressure)
const FX_V: f32 = 1024.0;    // velocity sums (m/s)
const SLOT: u32 = 12u;       // i32 per body and substep

struct Params {
  g: Grid,
  r0: vec4<f32>,   // region corner (cells), body (collider index)
  r1: vec4<f32>,   // region size (cells), output slot
  ccnt: vec4<f32>,
  col: array<Collider, MAX_COLLIDERS>,
};

@group(0) @binding(0) var X: texture_3d<f32>;
@group(0) @binding(1) var T: texture_3d<f32>;
@group(0) @binding(2) var vel: texture_3d<f32>;
@group(0) @binding(3) var atlas: texture_3d<f32>;
@group(0) @binding(4) var<storage, read_write> acc: array<atomic<i32>>;
@group(1) @binding(0) var<uniform> U: Params;

fn add(k: u32, v: f32, s: f32) {
  let q = i32(clamp(v * s, -2.0e9, 2.0e9));
  if (q != 0) { atomicAdd(&acc[u32(U.r1.w) * SLOT + k], q); }
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(vec3<f32>(id) >= U.r1.xyz)) { return; }
  let n = gdim(U.g);
  let c = vec3<i32>(U.r0.xyz) + vec3<i32>(id);
  if (!in_grid(c, n)) { return; }
  let body = U.col[u32(U.r0.w)];
  let h = U.g.n.w;
  if (col_sdf(body, world_of(U.g, vec3<f32>(c) + vec3<f32>(0.5))) >= 0.0) { return; }
  add(11u, 1.0, 1.0);  // solid cells of the body
  let centre = (body.a.xyz - U.g.org.xyz) / h;
  var f = vec3<f32>(0.0);
  var tq = 0.0;
  var vs = vec3<f32>(0.0);
  var nv = 0.0;
  for (var i = 0; i < 6; i++) {
    var d = vec3<i32>(0);
    d[i >> 1] = select(-1, 1, (i & 1) == 1);
    let q = c + d;
    if (!in_grid(q, n)) { continue; }
    if (textureLoad(T, q, 0).x < 0.5 || textureLoad(T, q, 0).x > 1.5) { continue; }
    let x = textureLoad(X, q, 0).x;
    let ff = -vec3<f32>(d) * x;   // pressure pushes into the body
    f += ff;
    let r = vec3<f32>(c) + vec3<f32>(0.5) + 0.5 * vec3<f32>(d) - centre;
    tq += r.z * ff.x - r.x * ff.z;
    vs += mac_vel(vel, vec3<f32>(q) + vec3<f32>(0.5), n);
    nv += 1.0;
  }
  if (nv == 0.0) { return; }
  add(0u, f.x, FX_F);
  add(1u, f.y, FX_F);
  add(2u, f.z, FX_F);
  add(3u, tq, FX_T);
  add(4u, vs.x, FX_V);
  add(5u, vs.y, FX_V);
  add(6u, vs.z, FX_V);
  add(7u, nv, 1.0);
}
