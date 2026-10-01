// Every particle slot free (age -1). A cleared buffer's zeros would read as live particles (age 0) at
// the grid's corner, and the particle kernels run over whole rows of 65535 workgroups, past the
// slots in use: over 4 million particles, millions of those would join the liquid there.
//!include common.wgsl
//!include liq_common.wgsl

struct Params {
  k: vec4<f32>,   // slot capacity
};

@group(0) @binding(0) var<storage, read_write> parts: array<Particle>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.k.x)) { return; }
  parts[i].p.w = -1.0;
}
