// Water in fabric, after cloth_wick.wgsl: the cloth takes its new wetness; the water boiled off it is
// kept (IMP.w, kg) for cloth_splat.wgsl to give to the gas as steam; and the water running off it
// gathers at each spot until there is a drop's worth (2-2.8 mm across the radius, as water drips off
// cloth), when it falls (cloth_drops.wgsl). Where the water runs in faster than single drops can carry
// it, it falls in a stream of them. Cloth hanging in a liquid gives its water back to it instead.
//!include cloth_common.wgsl

const RHO_W: f32 = 1000.0;
const DROP_LIFE: f32 = 4.0;    // s (a drop that has not landed by then is let go)

struct Params {
  a: vec4<f32>,    // dt, vertex count, drops (capacity), substep (for the drops' sizes)
  lq: vec4<f32>,   // the liquid's grid: dims, cell size (m)
  lo: vec4<f32>,   // its corner (fire-local m), liquid on (1/0)
};

struct Drops {
  head: atomic<u32>,   // drops made so far (the next one goes in slot head % capacity)
  p1: u32,
  p2: u32,
  p3: u32,
  d: array<vec4<f32>>, // per drop: position (fire-local m), life left (s); velocity (m/s), radius (m)
};

@group(0) @binding(0) var<storage, read_write> P: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> W: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> IMP: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> N: array<vec4<f32>>;    // w: area (m^2)
@group(0) @binding(4) var<storage, read> M: array<Mat>;
@group(0) @binding(5) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> DR: Drops;
@group(0) @binding(8) var LTYPE: texture_3d<f32>;   // the liquid's cells: 0 air, 1 liquid, 2 solid
@group(1) @binding(0) var<uniform> U: Params;

fn in_liquid(p: vec3<f32>) -> bool {
  if (U.lo.w < 0.5) { return false; }
  let n = vec3<i32>(U.lq.xyz);
  let c = vec3<i32>(floor((p - U.lo.xyz) / U.lq.w));
  if (any(c < vec3<i32>(0)) || any(c >= n)) { return false; }
  let t = textureLoad(LTYPE, c, 0).x;
  return t > 0.5 && t < 1.5;
}

fn drip_hash(n: u32) -> vec3<f32> {
  var s = n * 747796405u + 2891336453u;
  s = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  let a = (s >> 22u) ^ s;
  let b = a * 1664525u + 1013904223u;
  let c = b * 1664525u + 1013904223u;
  return vec3<f32>(f32(a), f32(b), f32(c)) * (1.0 / 4294967296.0);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.a.y)) { return; }
  let base = 2u * i;
  let w = W[base];
  let p = P[i];
  P[i] = vec4<f32>(p.xyz, w.x);
  let mat = M[u32(V[i].w + 0.5)];
  // kg of water a soaking of this spot of cloth holds
  let kg = mat.f.y * mat.a.x * max(N[i].w, 0.0);
  if (w.y > 0.0) {
    let imp = IMP[i];
    IMP[i] = vec4<f32>(imp.xyz, imp.w + w.y * kg);
  }
  var q = W[base + 1u].x + w.z * kg;
  if (q <= 0.0) { return; }
  let x = X[i].xyz;
  if (in_liquid(x)) {
    W[base + 1u] = vec4<f32>(0.0);
    return;
  }
  let cap = u32(U.a.z);
  let seed = u32(U.a.w);
  let vel = V[i].xyz;
  for (var k = 0u; k < 4u; k++) {
    let h = drip_hash(i * 9781u + seed * 6271u + k * 31u);
    let r = mix(0.0020, 0.0028, h.x);
    let m = RHO_W * 4.18879 * r * r * r;
    if (q < m) { break; }
    q -= m;
    let slot = atomicAdd(&DR.head, 1u) % cap;
    // it hangs from the cloth's underside as it lets go; a stream's drops follow one another down
    let pos = x - vec3<f32>(0.0, r + mat.a.w + f32(k) * 3.0 * r, 0.0);
    let jit = (h.yzy - vec3<f32>(0.5)) * vec3<f32>(0.04, 0.0, 0.04);
    DR.d[2u * slot] = vec4<f32>(pos, DROP_LIFE);
    DR.d[2u * slot + 1u] = vec4<f32>(vel + jit - vec3<f32>(0.0, 0.05 * f32(k), 0.0), r);
  }
  W[base + 1u] = vec4<f32>(q, 0.0, 0.0, 0.0);
}
