// Per-workgroup partial sums of a . b over the grid.
//!include liq_cg_common.wgsl
//!include liq_cg_reduce.wgsl

@group(0) @binding(0) var A: texture_3d<f32>;
@group(0) @binding(1) var B: texture_3d<f32>;
@group(0) @binding(2) var<storage, read_write> partials: array<f32>;
@group(1) @binding(0) var<uniform> U: CGParams;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32,
        @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let c = vec3<i32>(id);
  var v = 0.0;
  if (all(c < vec3<i32>(U.n.xyz))) {
    v = textureLoad(A, c, 0).x * textureLoad(B, c, 0).x;
  }
  reduce_store(li, v, wg_index(wid, nwg));
}
