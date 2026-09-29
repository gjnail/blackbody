// Tent-filtered upsample of the coarser level, added to this level's downsample (weighted).

struct Params {
  low: vec4<f32>,  // coarse width, height, 1/width, 1/height
  dst: vec4<f32>,  // destination width, height, weight of this level, filter radius (coarse texels)
};

@group(0) @binding(0) var low: texture_2d<f32>;
@group(0) @binding(1) var here: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let px = vec2<i32>(id.xy);
  if (f32(px.x) >= U.dst.x || f32(px.y) >= U.dst.y) { return; }
  let uv = (vec2<f32>(px) + 0.5) / U.dst.xy;
  let t = U.low.zw * U.dst.w;
  var up = textureSampleLevel(low, lin, uv, 0.0).rgb * 4.0;
  up += (textureSampleLevel(low, lin, uv + vec2<f32>(t.x, 0.0), 0.0).rgb
       + textureSampleLevel(low, lin, uv - vec2<f32>(t.x, 0.0), 0.0).rgb
       + textureSampleLevel(low, lin, uv + vec2<f32>(0.0, t.y), 0.0).rgb
       + textureSampleLevel(low, lin, uv - vec2<f32>(0.0, t.y), 0.0).rgb) * 2.0;
  up += textureSampleLevel(low, lin, uv + t, 0.0).rgb
      + textureSampleLevel(low, lin, uv - t, 0.0).rgb
      + textureSampleLevel(low, lin, uv + vec2<f32>(t.x, -t.y), 0.0).rgb
      + textureSampleLevel(low, lin, uv + vec2<f32>(-t.x, t.y), 0.0).rgb;
  up *= 1.0 / 16.0;
  let h = textureLoad(here, px, 0).rgb;
  textureStore(dst, px, vec4<f32>(h * U.dst.z + up, 1.0));
}
