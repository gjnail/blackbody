// With Lume tracing the water (stage.wgsl lume_water.wgsl), the liquid's own element is empty: the stage's picture has
// the water in it already. Its passes are cleared to that (no liquid, the footage as it is) for the drops, the rain and
// the composite to go over.

@group(0) @binding(0) var out_beauty: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var out_emit: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var out_aux: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var out_mask: texture_storage_2d<rgba16float, write>;

struct Params {
  res: vec4<f32>,   // width, height
};
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec2<i32>(id.xy);
  if (c.x >= i32(U.res.x) || c.y >= i32(U.res.y)) { return; }
  textureStore(out_beauty, c, vec4<f32>(0.0));
  textureStore(out_emit, c, vec4<f32>(0.0));
  textureStore(out_aux, c, vec4<f32>(1.0, 0.0, 0.0, 0.0));   // (x: the footage times 1; no liquid here)
  textureStore(out_mask, c, vec4<f32>(0.0));
}
