// The box's foam particles per column of cells (for ocl_cols.wgsl).
//!include common.wgsl
//!include liq_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,   // whitewater capacity
};

@group(0) @binding(0) var<storage, read> WA: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> WB: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> cnt: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.k.x)) { return; }
  let a = WA[i];
  if (a.w <= 0.0) { return; }
  if (abs(WB[i].w - 1.0) > 0.5) { return; }   // foam only
  let n = gdim(U.g);
  let c = vec2<i32>(floor(a.xz));
  if (any(c < vec2<i32>(0)) || c.x >= n.x || c.y >= n.z) { return; }
  atomicAdd(&cnt[u32(c.x + n.x * c.y)], 1u);
}
