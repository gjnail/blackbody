// p = z + beta p
//!include liq_cg_common.wgsl

@group(0) @binding(0) var Z: texture_3d<f32>;
@group(0) @binding(1) var P: texture_storage_3d<r32float, read_write>;
@group(0) @binding(2) var<storage, read> S: array<f32>;
@group(1) @binding(0) var<uniform> U: CGParams;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (any(c >= vec3<i32>(U.n.xyz))) { return; }
  let p = textureLoad(Z, c, 0).x + S[2] * textureLoad(P, c).x;
  textureStore(P, c, vec4<f32>(p, 0.0, 0.0, 0.0));
}
