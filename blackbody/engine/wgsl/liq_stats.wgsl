// Particle statistics: live count, top speed, and the bounding box of the liquid in cells.
// out: [0] live, [1] max speed (float bits), [2..4] min cell, [5..7] max cell
//!include common.wgsl
//!include liq_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,  // slot capacity
};

@group(0) @binding(0) var<storage, read> parts: array<Particle>;
@group(0) @binding(1) var<storage, read_write> out: array<atomic<u32>, 8>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.k.x)) { return; }
  let P = parts[i];
  if (P.p.w < 0.5) { return; }
  atomicAdd(&out[0], 1u);
  atomicMax(&out[1], bitcast<u32>(length(P.v.xyz)));
  let c = vec3<u32>(max(floor(P.p.xyz), vec3<f32>(0.0)));
  for (var a = 0; a < 3; a++) {
    atomicMin(&out[2 + a], c[a]);
    atomicMax(&out[5 + a], c[a]);
  }
}
