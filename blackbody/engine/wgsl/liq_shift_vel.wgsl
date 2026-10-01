// The box moves by whole cells: the grid velocity the deep liquid keeps between steps (narrow band)
// moves with it. The box's new edge takes the velocity just inside it.
//!include common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,  // shift of the box (cells, x y z)
};

@group(0) @binding(0) var src: texture_3d<f32>;
@group(0) @binding(1) var dst: texture_storage_3d<rgba32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let m = gdim(U.g) + vec3<i32>(1);
  let c = vec3<i32>(id);
  if (any(c >= m)) { return; }
  let s = clamp(c + vec3<i32>(U.k.xyz), vec3<i32>(0), m - vec3<i32>(1));
  textureStore(dst, c, textureLoad(src, s, 0));
}
