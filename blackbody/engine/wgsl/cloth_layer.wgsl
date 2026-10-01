// The drawn cloth as one layer for the liquid's march (liq_march.wgsl): rgb = its colour, a = its
// distance along the pixel's ray from where the march starts (m), or -1 where there is no cloth.

struct Params { s: vec4<f32> };   // width, height

@group(0) @binding(0) var col: texture_2d<f32>;
@group(0) @binding(1) var aux: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let px = vec2<i32>(id.xy);
  if (f32(px.x) >= U.s.x || f32(px.y) >= U.s.y) { return; }
  let a = textureLoad(aux, px, 0);
  let c = textureLoad(col, px, 0);
  textureStore(dst, px, select(vec4<f32>(0.0, 0.0, 0.0, -1.0), vec4<f32>(c.rgb, a.y), a.w > 0.5));
}
