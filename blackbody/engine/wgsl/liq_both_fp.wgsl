// Fire and liquid in one box: the fire over the footage, as the backdrop the liquid refracts and
// reflects (so flames behind a sheet of water show through it).
//!include colour.wgsl

struct Params {
  res: vec4<f32>,    // width, height
  plate: vec4<f32>,  // footage on (1/0), input transform, fit (x, y)
  g: vec4<f32>,      // footage gain, fire gain, smoke opacity
  bg: vec4<f32>,     // background colour without footage
};

@group(0) @binding(0) var beauty: texture_2d<f32>;
@group(0) @binding(1) var plate: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let px = vec2<i32>(id.xy);
  if (f32(px.x) >= U.res.x || f32(px.y) >= U.res.y) { return; }
  let uv = (vec2<f32>(px) + vec2<f32>(0.5)) / U.res.xy;
  var back = U.bg.rgb;
  if (U.plate.x > 0.5) {
    let puv = (uv - vec2<f32>(0.5)) * U.plate.zw + vec2<f32>(0.5);
    back = input_transform(textureSampleLevel(plate, lin, puv, 0.0).rgb, i32(U.plate.y)) * U.g.x;
  }
  let b = textureLoad(beauty, px, 0);
  let a = clamp(b.a * U.g.z, 0.0, 1.0);
  textureStore(dst, px, vec4<f32>(b.rgb * U.g.y + back * (1.0 - a), 1.0));
}
