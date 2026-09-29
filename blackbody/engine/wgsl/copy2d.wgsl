// Copy three float textures into the render buffers (used after resolving accumulated samples).

struct Params {
  n: vec4<f32>,
};

@group(0) @binding(0) var a: texture_2d<f32>;
@group(0) @binding(1) var b: texture_2d<f32>;
@group(0) @binding(2) var c: texture_2d<f32>;
@group(0) @binding(3) var oa: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var ob: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var oc: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let px = vec2<i32>(id.xy);
  if (f32(px.x) >= U.n.x || f32(px.y) >= U.n.y) { return; }
  textureStore(oa, px, textureLoad(a, px, 0));
  textureStore(ob, px, textureLoad(b, px, 0));
  textureStore(oc, px, textureLoad(c, px, 0));
}
