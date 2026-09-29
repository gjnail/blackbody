// 13-tap downsample (Jimenez 2014). The first level uses a Karis average against fireflies.

struct Params {
  src: vec4<f32>,  // source width, height, 1/width, 1/height
  dst: vec4<f32>,  // destination width, height, first level (1/0), _
};

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var lin: sampler;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn s(uv: vec2<f32>) -> vec3<f32> { return textureSampleLevel(src, lin, uv, 0.0).rgb; }

fn karis(c: vec3<f32>) -> f32 { return 1.0 / (1.0 + dot(c, vec3<f32>(0.2126, 0.7152, 0.0722))); }

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let px = vec2<i32>(id.xy);
  if (f32(px.x) >= U.dst.x || f32(px.y) >= U.dst.y) { return; }
  let uv = (vec2<f32>(px) + 0.5) / U.dst.xy;
  let t = U.src.zw;
  let a = s(uv + t * vec2<f32>(-2.0, -2.0));
  let b = s(uv + t * vec2<f32>(0.0, -2.0));
  let c = s(uv + t * vec2<f32>(2.0, -2.0));
  let d = s(uv + t * vec2<f32>(-1.0, -1.0));
  let e = s(uv + t * vec2<f32>(1.0, -1.0));
  let f = s(uv + t * vec2<f32>(-2.0, 0.0));
  let g = s(uv);
  let h = s(uv + t * vec2<f32>(2.0, 0.0));
  let i = s(uv + t * vec2<f32>(-1.0, 1.0));
  let j = s(uv + t * vec2<f32>(1.0, 1.0));
  let k = s(uv + t * vec2<f32>(-2.0, 2.0));
  let l = s(uv + t * vec2<f32>(0.0, 2.0));
  let m = s(uv + t * vec2<f32>(2.0, 2.0));
  var o: vec3<f32>;
  if (U.dst.z > 0.5) {
    let g0 = (d + e + i + j) * 0.25;
    let g1 = (a + b + f + g) * 0.25;
    let g2 = (b + c + g + h) * 0.25;
    let g3 = (f + g + k + l) * 0.25;
    let g4 = (g + h + l + m) * 0.25;
    let w0 = karis(g0) * 0.5;
    let w1 = karis(g1) * 0.125;
    let w2 = karis(g2) * 0.125;
    let w3 = karis(g3) * 0.125;
    let w4 = karis(g4) * 0.125;
    o = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
  } else {
    o = (d + e + i + j) * 0.125 + (a + c + k + m) * 0.03125 + (b + f + h + l) * 0.0625 + g * 0.125;
  }
  textureStore(dst, px, vec4<f32>(max(o, vec3<f32>(0.0)), 1.0));
}
