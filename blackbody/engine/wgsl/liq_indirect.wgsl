// Workgroup counts for particle kernels (64 threads each) from a GPU-side counter, in 2D past
// 65535 workgroups.
struct Params {
  k: vec4<f32>,  // counter index, capacity, dispatch-args offset (u32 words), _
};

@group(0) @binding(0) var<storage, read> ctr: array<i32>;
@group(0) @binding(1) var<storage, read_write> args: array<u32>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(1, 1, 1)
fn main() {
  let count = clamp(ctr[u32(U.k.x)], 0, i32(U.k.y));
  let o = u32(U.k.z);
  let groups = u32((count + 63) / 64);
  args[o] = min(groups, 65535u);
  args[o + 1u] = max((groups + 65534u) / 65535u, 1u);
  args[o + 2u] = 1u;
}
