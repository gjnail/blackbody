// The cloth's and the colliders' extinction (cloth_occlude.wgsl) into the light volume's spare channels:
// L1.z = cloth, L1.w = objects (1/m). Written to T2, copied back over L1.

struct Params {
  ld: vec4<f32>,
};

@group(0) @binding(0) var L1: texture_3d<f32>;
@group(0) @binding(1) var CO: texture_3d<f32>;
@group(0) @binding(2) var T2: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (any(c >= vec3<i32>(U.ld.xyz))) { return; }
  let l = textureLoad(L1, c, 0);
  let o = textureLoad(CO, c, 0);
  textureStore(T2, c, vec4<f32>(l.x, l.y, o.x, o.y));
}
