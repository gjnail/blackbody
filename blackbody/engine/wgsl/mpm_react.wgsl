// Matter: add this step's momentum taken by each object (fixed point, mpm_grid.wgsl) to the frame's total (f32), and
// clear it for the next step.

@group(0) @binding(0) var<storage, read_write> step_sum: array<i32>;
@group(0) @binding(1) var<storage, read_write> total: array<f32>;
@group(1) @binding(0) var<uniform> U: vec4<f32>;    // slots, 1 / fixed point, _, _

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.x)) { return; }
  total[i] += f32(step_sum[i]) * U.y;
  step_sum[i] = 0;
}
