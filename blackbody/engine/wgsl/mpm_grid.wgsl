// Matter, the grid: each node's velocity from the momentum and mass the particles gave it, then gravity, and the
// ground, the box's walls and the objects in the scene, which stop it moving into them and drag it along their
// surface with friction. What the objects do to the matter, they feel back: the momentum each one takes from it is
// summed (react) for the rigid bodies (solids.py).
//!include mpm_common.wgsl
//!include meshsdf.wgsl
//!include colliders.wgsl

struct Params {
  n: vec4<f32>,      // grid nodes (x, y, z), _
  s: vec4<f32>,      // step (s), node spacing (m), gravity (m/s^2), friction of the ground and the walls
  o: vec4<f32>,      // the grid's corner (node 0, fire-local m), the ground's height (fire-local m; very low: none)
  w: vec4<f32>,      // closed sides (1/0), the box's edge (nodes in from the grid's), _, _
  ccnt: vec4<f32>,   // objects (count), _
  col: array<Collider, MAX_COLLIDERS>,
};

@group(0) @binding(0) var atlas: texture_3d<f32>;
@group(0) @binding(1) var<storage, read> G: array<i32>;
@group(0) @binding(2) var vel: texture_storage_3d<rgba32float, write>;
@group(0) @binding(3) var<storage, read_write> react: array<atomic<i32>>;   // per object: momentum taken (xyz), its moment (xyz)
@group(1) @binding(0) var<uniform> U: Params;

const FX_R: f32 = 64.0;   // react's fixed point (momentum in particles of water times m/s)

// Stop v moving into a surface (normal n) that moves at vc, with friction mu (Coulomb): it may leave freely.
fn boundary(v: vec3<f32>, vc: vec3<f32>, n: vec3<f32>, mu: f32) -> vec3<f32> {
  let rel = v - vc;
  let vn = dot(rel, n);
  if (vn >= 0.0) { return v; }
  let vt = rel - vn * n;
  let lt = length(vt);
  let slow = lt + mu * vn;
  return vc + select(vec3<f32>(0.0), vt * (slow / max(lt, 1e-12)), slow > 0.0);
}

fn col_normal(k: Collider, p: vec3<f32>, e: f32) -> vec3<f32> {
  let a = vec3<f32>(1.0, -1.0, -1.0);
  let b = vec3<f32>(-1.0, -1.0, 1.0);
  let c = vec3<f32>(-1.0, 1.0, -1.0);
  let d = vec3<f32>(1.0, 1.0, 1.0);
  let g = a * col_sdf(k, p + a * e) + b * col_sdf(k, p + b * e) + c * col_sdf(k, p + c * e) + d * col_sdf(k, p + d * e);
  let l = length(g);
  return select(vec3<f32>(0.0, 1.0, 0.0), g / l, l > 1e-12);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(U.n.xyz);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let k = nidx(c, n) * NODE;
  let mass = f32(G[k + 3u]) / FX_MASS;
  if (mass <= 1.0e-9) {
    textureStore(vel, c, vec4<f32>(0.0));
    return;
  }
  let dt = U.s.x;
  let dx = U.s.y;
  var v = vec3<f32>(f32(G[k]), f32(G[k + 1u]), f32(G[k + 2u])) / FX_P / mass;
  v.y -= U.s.z * dt;
  let p = U.o.xyz + vec3<f32>(c) * dx;      // fire-local m
  let mu = U.s.w;
  // the ground
  if (p.y - U.o.w < 0.25 * dx) {
    v = boundary(v, vec3<f32>(0.0), vec3<f32>(0.0, 1.0, 0.0), mu);
  }
  // the objects
  for (var i = 0; i < i32(U.ccnt.x); i++) {
    let o = U.col[i];
    let d = col_sdf(o, p);
    if (d >= 0.25 * dx) { continue; }
    let before = v;
    v = boundary(v, col_velocity(o, p), col_normal(o, p, 0.5 * dx), mu);
    let dp = mass * (before - v);              // the momentum the object took
    if (dot(dp, dp) > 0.0) {
      let arm = p - o.a.xyz;
      let q = clamp(dp * FX_R, vec3<f32>(-2.0e9), vec3<f32>(2.0e9));
      let t = clamp(cross(arm, dp) * FX_R, vec3<f32>(-2.0e9), vec3<f32>(2.0e9));
      let r = u32(i) * 6u;
      atomicAdd(&react[r], i32(round(q.x)));
      atomicAdd(&react[r + 1u], i32(round(q.y)));
      atomicAdd(&react[r + 2u], i32(round(q.z)));
      atomicAdd(&react[r + 3u], i32(round(t.x)));
      atomicAdd(&react[r + 4u], i32(round(t.y)));
      atomicAdd(&react[r + 5u], i32(round(t.z)));
    }
  }
  // the box's closed sides (through open ones, the top, or a box without ground, it leaves: mpm_g2p.wgsl)
  if (U.w.x > 0.5) {
    let mg = i32(U.w.y);
    if (c.x <= mg) { v.x = max(v.x, 0.0); }
    if (c.x >= n.x - 1 - mg) { v.x = min(v.x, 0.0); }
    if (c.z <= mg) { v.z = max(v.z, 0.0); }
    if (c.z >= n.z - 1 - mg) { v.z = min(v.z, 0.0); }
  }
  textureStore(vel, c, vec4<f32>(v, mass));
}
