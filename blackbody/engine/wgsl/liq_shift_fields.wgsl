// The box moves by whole cells: the fields that are not rebuilt from the particles every step move
// with it. whitewater: positions (cells) of the spray, foam and bubbles; band: the deep-liquid flags
// (one u32 per cell; the cells the box moves over start as not deep, the new open water's fill marks
// them); wet: the ground's wetness (one value per column).
//!include common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,  // shift of the box (cells, x y z), whitewater capacity
};

@group(0) @binding(0) var<storage, read_write> A: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn ww(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = id.x + id.y * nwg.x * 64u;
  if (i >= u32(U.k.w)) { return; }
  var a = A[i];
  if (a.w <= 0.0) { return; }
  a = vec4<f32>(a.xyz - U.k.xyz, a.w);
  if (any(a.xyz < vec3<f32>(0.0)) || any(a.xyz > vec3<f32>(gdim(U.g)))) { a.w = 0.0; }
  A[i] = a;
}
