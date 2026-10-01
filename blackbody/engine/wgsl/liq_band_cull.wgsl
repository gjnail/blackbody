// Narrow band, pass 4: particles that have sunk into the deep liquid are freed (the grid carries
// the liquid there). A cell that has just joined the deep liquid (it holds at least a full cell, see
// pass 3) hands over exactly one cell of particles, since it counts as full from now on: any beyond
// that step across into a neighbouring cell of the band instead of vanishing, so the deep liquid
// grows without taking liquid out of the box.
//!include common.wgsl
//!include liq_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,   // slot capacity, particles per cell
};

@group(0) @binding(0) var<storage, read_write> parts: array<Particle>;
@group(0) @binding(1) var<storage, read_write> ctr: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> freelist: array<u32>;
@group(0) @binding(3) var<storage, read> band: array<u32>;
@group(0) @binding(4) var<storage, read> band_old: array<u32>;
@group(0) @binding(5) var<storage, read_write> cellcount: array<atomic<u32>>;
@group(0) @binding(6) var sdf: texture_3d<f32>;
@group(0) @binding(7) var<storage, read_write> bank: array<atomic<i32>, 8>;   // see liq_band_bank.wgsl
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.k.x)) { return; }
  var P = parts[i];
  if (!alive(P)) { return; }
  let n = gdim(U.g);
  let c = clamp(vec3<i32>(floor(P.p.xyz)), vec3<i32>(0), n - vec3<i32>(1));
  let ci = nidx(c, n);
  if (band[ci] == 0u) { return; }
  if (band_old[ci] == 0u && atomicSub(&cellcount[ci], 1u) > u32(U.k.y)) {
    // one more than the cell hands over: into a neighbour still in the band, mirrored across the face
    let start = (i * 2654435761u) >> 29u;
    for (var j = 0u; j < 6u; j++) {
      let d = (start + j) % 6u;
      let a = d >> 1u;
      var o = vec3<i32>(0);
      o[a] = select(-1, 1, (d & 1u) == 1u);
      let q = c + o;
      if (!in_grid(q, n) || band[nidx(q, n)] != 0u || textureLoad(sdf, q, 0).x < 0.0) { continue; }
      var x = P.p.xyz;
      let face = f32(select(c[a], c[a] + 1, o[a] > 0));
      x[a] = clamp(2.0 * face - x[a], f32(q[a]) + 0.001, f32(q[a]) + 0.999);
      P.p = vec4<f32>(x, P.p.w);
      parts[i] = P;
      atomicAdd(&cellcount[nidx(q, n)], 1u);
      return;
    }
  }
  P.p.w = -1.0;
  parts[i] = P;
  let k = atomicAdd(&ctr[C_FREE], 1);
  freelist[u32(k)] = i;
  atomicAdd(&bank[0], -1);
}
