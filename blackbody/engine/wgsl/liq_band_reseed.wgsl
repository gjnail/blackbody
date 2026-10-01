// Narrow band, pass 5: where the surface has come down to what was deep liquid (a trough, a cavity,
// a splash crater), the cell gets its particles back, moving with the grid velocity there. With no
// slots left the cell stays deep liquid: left empty it would turn to air inside the water, the deep
// liquid round it would count as near a surface and empty in turn, and the box would drain.
// Only what the cell lacks is added: particles that moved into it this step already count (a full
// batch on top of them overfills the cell, and the volume correction then swells the liquid).
//!include common.wgsl
//!include liq_common.wgsl
//!include noise.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,   // particles per cell, slot capacity, step seed, sub-cells per axis
  a: vec4<f32>,   // particles carry dye and density (1/0)
};

@group(0) @binding(0) var<storage, read_write> parts: array<Particle>;
@group(0) @binding(1) var<storage, read_write> ctr: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read> freelist: array<u32>;
@group(0) @binding(3) var<storage, read> band_old: array<u32>;
@group(0) @binding(4) var<storage, read_write> band_new: array<u32>;
@group(0) @binding(5) var sdf: texture_3d<f32>;
@group(0) @binding(6) var vel: texture_3d<f32>;
@group(0) @binding(7) var<storage, read_write> attr: array<vec4<u32>>;
@group(0) @binding(8) var<storage, read> cellcount: array<u32>;   // particles per cell after the move
@group(0) @binding(9) var<storage, read_write> bank: array<atomic<i32>, 8>;   // see liq_band_bank.wgsl
@group(1) @binding(0) var<uniform> U: Params;

fn sub_cell(k: u32, m: f32) -> vec3<u32> {
  let mi = u32(m);
  let j = k % (mi * mi * mi);
  return vec3<u32>(j % mi, (j / mi) % mi, j / (mi * mi));
}

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

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let i = nidx(c, n);
  if (band_old[i] == 0u || band_new[i] != 0u) { return; }
  // a full cell less of deep liquid (a collider that moved in displaces it: the bank puts it back)
  atomicAdd(&bank[0], -i32(U.k.x));
  if (textureLoad(sdf, c, 0).x < 0.0) { return; }
  let want = u32(U.k.x);
  let seed = i * 2654435761u + u32(U.k.z) * 97u;
  for (var k = min(cellcount[i], want); k < want; k++) {
    let r = rand3(seed, k + 11u);
    let x = vec3<f32>(c) + (vec3<f32>(sub_cell(k, U.k.w)) + r) / U.k.w;
    let slot = alloc();
    if (slot < 0) {
      band_new[i] = 1u;
      atomicAdd(&bank[0], i32(U.k.x));   // it stays deep liquid after all
      return;
    }
    parts[u32(slot)] = new_particle(x, mac_vel(vel, x, n));
    if (U.a.x > 0.5) { attr[u32(slot)] = vec4<u32>(0u); }
    atomicAdd(&bank[0], 1);
  }
}
