// Matter, the grid: each node's velocity from the momentum and mass the particles gave it, then gravity, the liquid's
// push and drag (mpm_liquid.wgsl), and the ground, the box's walls and the objects in the scene, which stop it moving
// into them and drag it along their surface with friction. What the objects do to the matter, they feel back: the momentum each one takes from it is
// summed (react) for the rigid bodies (solids.py).
//!include mpm_common.wgsl
//!include meshsdf.wgsl
//!include colliders.wgsl

struct Params {
  n: vec4<f32>,      // grid nodes (x, y, z), fabric (cloth_matter.wgsl's sheet in CF: mpm_common.wgsl SHEET_N)
  s: vec4<f32>,      // step (s), node spacing (m), gravity (m/s^2), friction of the ground and the walls
  o: vec4<f32>,      // the grid's corner (node 0, fire-local m), the ground's height (fire-local m; very low: none)
  w: vec4<f32>,      // closed sides (1/0), the box's edge (nodes in from the grid's), the liquid's push on (1/0), the
                     // least density a node is taken to have for the liquid's push (kg/m^3: its lightest material's;
                     // a node at the matter's edge is part water, but its grains sink or float by their own density)
  ccnt: vec4<f32>,   // objects (count), _
  col: array<Collider, MAX_COLLIDERS>,
};

@group(0) @binding(0) var atlas: texture_3d<f32>;
@group(0) @binding(1) var<storage, read> G: array<i32>;
@group(0) @binding(2) var vel: texture_storage_3d<rgba32float, write>;
@group(0) @binding(3) var<storage, read_write> react: array<atomic<i32>>;   // per object: momentum taken (xyz), its moment
                                                                            // (xyz); then per broken objects' piece
@group(0) @binding(4) var push: texture_3d<f32>;   // the liquid's force on it (N/m^3); w: 1 where the liquid is
@group(0) @binding(5) var flow: texture_3d<f32>;   // the liquid's velocity there (m/s); w: its drag (kg/m^4)
@group(0) @binding(6) var psdf: texture_3d<f32>;   // broken objects' pieces (bodies_sdf.wgsl): the distance to them (nodes)
@group(0) @binding(7) var psvel: texture_3d<f32>;  // their velocity at the nodes in or by them; w: 1 + which piece
@group(0) @binding(8) var<storage, read> CF: array<i32>;   // the fabric round each node (cloth_matter.wgsl): weight,
                                                           // w velocity (3), w normal (3), w how far in front of it
@group(0) @binding(9) var<storage, read_write> CT: array<atomic<i32>>;   // per node: momentum the fabric took (3)
@group(1) @binding(0) var<uniform> U: Params;

const FX_R: f32 = 64.0;   // react's fixed point (momentum in particles of water times m/s)

fn sheet(k: u32, tdx: f32) -> Sheet {
  return sheet_at(vec4<i32>(CF[k], CF[k + 1u], CF[k + 2u], CF[k + 3u]), vec4<i32>(CF[k + 4u], CF[k + 5u], CF[k + 6u],
                  CF[k + 7u]), CF[k + 8u], tdx, U.s.y);
}
const RHO: f32 = 125.0;   // a node's density (kg/m^3) per unit of its mass (particles of water, eight to a node)

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
  // the liquid: its pressure, then its drag (as fast as the flow at most)
  if (U.w.z > 0.5) {
    let lp = textureLoad(push, c, 0);
    if (lp.w > 0.0) {
      let rho = max(mass * RHO, U.w.w);
      v += lp.xyz * (dt / rho);
      let lf = textureLoad(flow, c, 0);
      let du = lf.xyz - v;
      v += du * min(lf.w * length(du) * dt / rho, 1.0);
    }
  }
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
  // broken objects' pieces (in the colliders' count's z: pieces on): as the objects, each feeling what it took
  if (U.ccnt.z > 0.5) {
    let ps = textureLoad(psdf, c, 0).x;
    if (ps < 0.25) {
      let pv = textureLoad(psvel, c, 0);
      let lo = max(c - vec3<i32>(1), vec3<i32>(0));
      let hi = min(c + vec3<i32>(1), n - vec3<i32>(1));
      let g = vec3<f32>(textureLoad(psdf, vec3<i32>(hi.x, c.y, c.z), 0).x - textureLoad(psdf, vec3<i32>(lo.x, c.y, c.z), 0).x,
                        textureLoad(psdf, vec3<i32>(c.x, hi.y, c.z), 0).x - textureLoad(psdf, vec3<i32>(c.x, lo.y, c.z), 0).x,
                        textureLoad(psdf, vec3<i32>(c.x, c.y, hi.z), 0).x - textureLoad(psdf, vec3<i32>(c.x, c.y, lo.z), 0).x);
      let gl = length(g);
      let nrm = select(vec3<f32>(0.0, 1.0, 0.0), g / gl, gl > 1.0e-6);
      let before = v;
      v = boundary(v, pv.xyz, nrm, mu);
      let dp = mass * (before - v);
      if (pv.w > 0.5 && dot(dp, dp) > 0.0) {
        let r = 96u + 6u * u32(pv.w - 0.5);     // (after the 16 objects')
        let q = clamp(dp * FX_R, vec3<f32>(-2.0e9), vec3<f32>(2.0e9));
        let t = clamp(cross(p, dp) * FX_R, vec3<f32>(-2.0e9), vec3<f32>(2.0e9));   // (about the origin)
        if (r + 5u < arrayLength(&react)) {
          atomicAdd(&react[r], i32(round(q.x)));
          atomicAdd(&react[r + 1u], i32(round(q.y)));
          atomicAdd(&react[r + 2u], i32(round(q.z)));
          atomicAdd(&react[r + 3u], i32(round(t.x)));
          atomicAdd(&react[r + 4u], i32(round(t.y)));
          atomicAdd(&react[r + 5u], i32(round(t.z)));
        }
      }
    }
  }
  // fabric: a thin sheet moving as the cloth moves (mpm_common.wgsl SHEET_N). A node by it holds the matter of its
  // own side only, and the nearest ones cannot move into it, as far as it holds (sheet_hold); what they take, the
  // cloth feels (cloth_matter.wgsl gather)
  if (U.n.w > 0.5) {
    let sh = sheet(nidx(c, n) * SHEET_N, (U.n.w - 1.0) / dx);
    if (sh.side != 0.0 && abs(sh.off) < 0.5 && dot(sh.n, sh.n) > 0.5) {
      let before = v;
      v = mix(v, boundary(v, sh.v, sh.n * sh.side, mu), sheet_hold(sh, sh.side));
      let dp = mass * (before - v);
      if (dot(dp, dp) > 0.0) {
        let q = clamp(dp * SHEET_FX_T, vec3<f32>(-2.0e9), vec3<f32>(2.0e9));
        let kt = 3u * nidx(c, n);
        atomicAdd(&CT[kt], i32(round(q.x)));
        atomicAdd(&CT[kt + 1u], i32(round(q.y)));
        atomicAdd(&CT[kt + 2u], i32(round(q.z)));
      }
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
