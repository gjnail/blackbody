// Fire, water and lava in one box: lava advancing into the water takes its place. The water treats
// the lava as solid ground (both_solid.wgsl), but that ground moves, and water caught where the lava
// has just flowed would be squeezed out by the water's volume correction, which, pushing as hard as
// it must, flings it about. Here, before the water steps, water particles well inside the lava are
// freed instead: the lava has displaced them (the open water around the box keeps the level up).
//!include common.wgsl
//!include liq_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,  // slot capacity, the lava's rest density (particles per cell), share of lava in which water is displaced
};

@group(0) @binding(0) var<storage, read_write> parts: array<Particle>;   // the water's
@group(0) @binding(1) var<storage, read_write> ctr: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> freelist: array<u32>;
@group(0) @binding(3) var lava_dens: texture_3d<f32>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.k.x)) { return; }
  var P = parts[i];
  if (!alive(P)) { return; }
  let n = gdim(U.g);
  let lava = interp_cell(lava_dens, P.p.xyz, n).x / max(U.k.y, 1e-3);
  if (lava < U.k.z) { return; }
  P.p.w = -1.0;
  parts[i] = P;
  let k = atomicAdd(&ctr[C_FREE], 1);
  freelist[u32(k)] = i;
}
