// Fill a 3D texture with a constant.

struct Params {
  n: vec4<f32>,
  value: vec4<f32>,
};

@group(0) @binding(0) var dst: texture_storage_3d<${FMT}, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (any(c >= vec3<i32>(U.n.xyz))) { return; }
  textureStore(dst, c, U.value);
}
