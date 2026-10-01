// Narrow band, pass 1 after the move: the deep liquid gives back what flowed out of it. Particles
// that move into the deep liquid are freed (pass 4), so where the grid carried liquid out of the deep
// region into the band this step, as many particles are born just inside the band as the outflow
// moved (rest density times the face's flux). The deep region is the one the particles moved
// through (the flags before this step's update, cells about to leave it included), and the births
// count in cellcount, which decides the cells that join or leave the deep liquid next. Without this
// the band on the outflow side thins out and the box drains.
//!include common.wgsl
//!include liq_common.wgsl
//!include noise.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,   // particles per cell, slot capacity, step seed, particles carry dye and density (1/0)
};

@group(0) @binding(0) var<storage, read_write> parts: array<Particle>;
@group(0) @binding(1) var<storage, read_write> ctr: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read> freelist: array<u32>;
@group(0) @binding(3) var<storage, read> band: array<u32>;
@group(0) @binding(4) var sdf: texture_3d<f32>;
@group(0) @binding(5) var vel: texture_3d<f32>;
@group(0) @binding(6) var<storage, read_write> attr: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read_write> cellcount: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> bank: array<atomic<i32>, 8>;   // see liq_band_bank.wgsl
@group(1) @binding(0) var<uniform> U: Params;

fn alloc() -> i32 {
  let f = atomicSub(&ctr[C_FREE], 1);
  if (f > 0) { return i32(freelist[u32(f - 1)]); }
  atomicAdd(&ctr[C_FREE], 1);
  let s = atomicAdd(&ctr[C_COUNT], 1);
  if (s >= i32(U.k.y)) {
    atomicSub(&ctr[C_COUNT], 1);
    return -1;
  }
  return s;
}

fn is_deep(c: vec3<i32>, n: vec3<i32>) -> bool {
  if (!in_grid(c, n)) { return false; }
  return band[nidx(c, n)] != 0u;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  if (band[nidx(c, n)] != 0u || textureLoad(sdf, c, 0).x < 0.0) { return; }
  let dt = U.g.bc.w;
  let ih = 1.0 / U.g.n.w;
  // the outflow through each face shared with a deep cell, in cells of liquid this step
  var flux = array<f32, 6>();
  var total = 0.0;
  for (var i = 0u; i < 6u; i++) {
    let k = i >> 1u;
    let hi = (i & 1u) == 1u;
    var e = vec3<i32>(0);
    e[k] = 1;
    if (is_deep(select(c - e, c + e, hi), n)) {
      // the face is this cell's low face, or the high neighbour's low face
      let v = textureLoad(vel, select(c, c + e, hi), 0)[k];
      flux[i] = clamp(select(v, -v, hi) * dt * ih, 0.0, 1.0);   // > 0: out of the deep cell into this one
      total += flux[i];
    }
  }
  if (total <= 0.0) { return; }
  let q = total * U.k.x;   // the particles the outflow carried
  atomicAdd(&bank[2], i32(q * 256.0));
  // half the band's balance back, in this cell's share of last step's outflow
  let owed = -0.5 * f32(atomicLoad(&bank[1])) * q / max(f32(atomicLoad(&bank[3])) / 256.0, 1.0);
  let seed = nidx(c, n) * 2246822519u + u32(U.k.z) * 3266489917u;
  let want = u32(clamp(floor(q + clamp(owed, -q, U.k.x) + rand1(seed)), 0.0, 2.0 * U.k.x));
  for (var j = 0u; j < want; j++) {
    // a face in proportion to its outflow, then a point in the slab the outflow crossed this step
    var pick = rand1(seed + 7919u * (j + 1u)) * total;
    var i = 0u;
    for (; i < 5u; i++) {
      if (pick < flux[i]) { break; }
      pick -= flux[i];
    }
    let k = i >> 1u;
    let r = rand3(seed, j + 17u);
    var x = vec3<f32>(c) + r;
    let depth = r[k] * flux[i];
    x[k] = select(f32(c[k]) + depth, f32(c[k] + 1) - depth, (i & 1u) == 1u);
    let slot = alloc();
    if (slot < 0) { return; }
    parts[u32(slot)] = new_particle(x, mac_vel(vel, x, n));
    if (U.k.w > 0.5) { attr[u32(slot)] = vec4<u32>(0u); }
    atomicAdd(&cellcount[nidx(c, n)], 1u);
    atomicAdd(&bank[0], 1);
  }
}
