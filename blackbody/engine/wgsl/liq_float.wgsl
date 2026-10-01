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
const SLOT: u32 = 16u;       // i32 per body and substep:
// 0-2 pressure force, 3-5 its moment about the centre, 6-8 liquid velocity sum, 9 its count,
// 10 cells with nothing liquid under them, 11 pressure beside those, 12 its count, 13 solid cells

struct Params {
  g: Grid,
  r0: vec4<f32>,   // region corner (cells), body (collider index)
  r1: vec4<f32>,   // region size (cells), output slot
  k: vec4<f32>,    // hydrostatic change of x over half a cell (g dt h / 2)
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
  add(13u, 1.0, 1.0);  // solid cells of the body
  let centre = (body.a.xyz - U.g.org.xyz) / h;
  var f = vec3<f32>(0.0);
  var tq = vec3<f32>(0.0);
  var vs = vec3<f32>(0.0);
  var nv = 0.0;
  for (var i = 0; i < 6; i++) {
    var d = vec3<i32>(0);
    d[i >> 1] = select(-1, 1, (i & 1) == 1);
    let q = c + d;
    if (!in_grid(q, n)) { continue; }
    if (textureLoad(T, q, 0).x < 0.5 || textureLoad(T, q, 0).x > 1.5) { continue; }
    // pressure at the face, half a cell up or down from the liquid cell's centre
    let x = textureLoad(X, q, 0).x + U.k.x * f32(d.y);
    let ff = -vec3<f32>(d) * x;   // pressure pushes into the body
    f += ff;
    let r = vec3<f32>(c) + vec3<f32>(0.5) + 0.5 * vec3<f32>(d) - centre;
    tq += cross(r, ff);
    vs += mac_vel(vel, vec3<f32>(q) + vec3<f32>(0.5), n);
    nv += 1.0;
  }
  // nothing liquid under this cell (the ground, another solid, or the one-cell film of air a body
  // leaves as it lifts off): water seeps under a real object, so count these cells and the
  // pressure of the liquid beside them, for the push from below the grid misses
  let below = c - vec3<i32>(0, 1, 0);
  var exposed = false;
  if (below.y < 0) {
    exposed = U.g.bc.z < 0.5;
  } else {
    let tb = textureLoad(T, below, 0).x;
    exposed = tb < 0.5 || (tb > 1.5 && col_sdf(body, world_of(U.g, vec3<f32>(below) + vec3<f32>(0.5))) >= 0.0);
  }
  if (exposed) {
    add(10u, 1.0, 1.0);
    for (var layer = 0; layer < 2; layer++) {
      let base = select(c, below, layer == 1);
      if (base.y < 0) { continue; }
      for (var i = 0; i < 6; i++) {
        if ((i >> 1) == 1) { continue; }
        var d = vec3<i32>(0);
        d[i >> 1] = select(-1, 1, (i & 1) == 1);
        let q = base + d;
        if (!in_grid(q, n)) { continue; }
        let tt = textureLoad(T, q, 0).x;
        if (tt < 0.5 || tt > 1.5) { continue; }
        add(11u, textureLoad(X, q, 0).x, FX_F);
        add(12u, 1.0, 1.0);
      }
    }
  }
  if (nv == 0.0) { return; }
  add(0u, f.x, FX_F);
  add(1u, f.y, FX_F);
  add(2u, f.z, FX_F);
  add(3u, tq.x, FX_T);
  add(4u, tq.y, FX_T);
  add(5u, tq.z, FX_T);
  add(6u, vs.x, FX_V);
  add(7u, vs.y, FX_V);
  add(8u, vs.z, FX_V);
  add(9u, nv, 1.0);
}
