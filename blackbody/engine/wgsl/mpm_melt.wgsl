// Snow melting (matter.py): each snow particle warms in the gas round it, the hotter the faster (in flames about a second,
// in a warm room minutes), and where it touches an object warmer than freezing (a hot plate, a heated pipe), as the
// object gives it heat; when it has taken in its latent heat it melts: it leaves the matter, and in a box with a
// liquid the water it held joins the liquid (whole liquid particles, the share left over by chance), in free slots as
// liq_spawn.wgsl takes them, moving as it was.
//!include common.wgsl
//!include liq_common.wgsl
//!include meshsdf.wgsl
//!include colliders.wgsl

struct MParticle {
  x: vec4<f32>,   // position (the matter's grid units); w = material slot, negative: an empty slot
  v: vec4<f32>,   // velocity (m/s)
  c0: vec4<f32>,  // w = its own random number
  c1: vec4<f32>,
  c2: vec4<f32>,  // w = how far it has melted (0..1)
  f0: vec4<f32>,
  f1: vec4<f32>,
  f2: vec4<f32>,
};

struct Params {
  m: vec4<f32>,      // the matter's grid: node 0 (fire-local m), node spacing (m)
  g: Grid,           // the gas's grid
  lq: vec4<f32>,     // the liquid's grid corner (fire-local m), its cell size (m)
  k: vec4<f32>,      // particles (count), dt (s), the liquid's slot capacity (0: no liquid to join), seed
  t: vec4<f32>,      // ambient K, flame K, melting (per s per K above freezing), gas on (1/0)
  melt: array<vec4<f32>, 4>,   // per material slot (16): liquid particles a particle of it melts into (0: it does not melt)
  ccnt: vec4<f32>,             // the objects (solver.pack_colliders)
  col: array<Collider, MAX_COLLIDERS>,
  touch: array<vec4<f32>, MAX_COLLIDERS>,   // per object: its temperature (K), its effusivity (W s^0.5/m^2/K)
};

@group(0) @binding(0) var<storage, read_write> P: array<MParticle>;
@group(0) @binding(1) var scal: texture_3d<f32>;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var<storage, read_write> parts: array<Particle>;
@group(0) @binding(4) var<storage, read_write> ctr: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read> freelist: array<u32>;
@group(0) @binding(6) var atlas: texture_3d<f32>;   // the meshes' distance fields (meshsdf.wgsl)
@group(1) @binding(0) var<uniform> U: Params;

fn alloc() -> i32 {
  let f = atomicSub(&ctr[C_FREE], 1);
  if (f > 0) { return i32(freelist[u32(f - 1)]); }
  atomicAdd(&ctr[C_FREE], 1);
  let s = atomicAdd(&ctr[C_COUNT], 1);
  if (s >= i32(U.k.z)) {
    atomicSub(&ctr[C_COUNT], 1);
    return -1;
  }
  return s;
}

const SNOW_EFFUSIVITY: f32 = 400.0;   // W s^0.5/m^2/K

fn rnd(i: u32, j: u32) -> f32 {
  var h = i * 747796405u + j * 2891336453u + u32(U.k.w) * 277803737u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  return f32((h >> 22u) ^ h) / 4294967296.0;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = id.x + id.y * nwg.x * 64u;
  if (i >= u32(U.k.x)) { return; }
  var p = P[i];
  if (p.x.w < 0.0) { return; }
  let slot = u32(p.x.w + 0.5);
  if (slot >= 15u) { return; }
  let water = U.melt[slot / 4u][slot % 4u];
  if (water <= 0.0) { return; }
  let pos = U.m.xyz + p.x.xyz * U.m.w;
  // the air's temperature next to it (K): the hottest of the gas's a cell round it (inside a heap of snow the cells are
  // solid, with no gas: only the snow at its surface feels the flames), the ambient outside the box
  var tk = U.t.x;
  if (U.t.w > 0.5) {
    let n = U.g.n.xyz;
    let q = (pos - U.g.org.xyz) / U.g.n.w;
    if (all(q >= vec3<f32>(-1.0)) && all(q <= n + vec3<f32>(1.0))) {
      var T = 0.0;
      for (var a = 0; a < 7; a++) {
        var o = vec3<f32>(0.0);
        if (a > 0) { o[(a - 1) / 2] = select(-1.0, 1.0, (a & 1) == 0); }
        T = max(T, samp_c(scal, lin, q + o, n).x);
      }
      tk = U.t.x + (U.t.y - U.t.x) * min(T, 1.0);
    }
  }
  var over = tk - 273.15;
  // an object it touches (within a node spacing of it, as snow resting on it is), warmer than freezing: as the air
  // twice as far above freezing would, by the share of the heat at the face between them the object gives (its
  // effusivity against snow's, some 400)
  if (U.ccnt.x > 0.5) {
    var nearest = U.m.w;
    var heat = 0.0;
    for (var c = 0; c < i32(U.ccnt.x); c++) {
      let d = col_sdf(U.col[c], pos);
      if (d < nearest) {
        nearest = d;
        heat = 2.0 * max(U.touch[c].x - 273.15, 0.0) * U.touch[c].y / (U.touch[c].y + SNOW_EFFUSIVITY);
      }
    }
    over = max(over, 0.0) + heat;
  }
  if (over <= 0.0) { return; }
  p.c2.w += U.k.y * U.t.z * over;
  if (p.c2.w < 1.0) {
    P[i].c2 = p.c2;
    return;
  }
  // melted: out of the matter
  P[i].x = vec4<f32>(p.x.xyz, -1.0);
  if (U.k.z < 0.5) { return; }
  // its water: whole liquid particles, and one more with the chance of the share left over
  var n = u32(floor(water));
  if (rnd(i, 1u) < water - floor(water)) { n += 1u; }
  let x = (pos - U.lq.xyz) / U.lq.w;
  for (var j = 0u; j < min(n, 8u); j++) {
    let s = alloc();
    if (s < 0) { break; }
    let jit = vec3<f32>(rnd(i, 3u * j + 2u), rnd(i, 3u * j + 3u), rnd(i, 3u * j + 4u)) - vec3<f32>(0.5);
    parts[u32(s)] = new_particle(x + 0.3 * jit, p.v.xyz);
  }
}
