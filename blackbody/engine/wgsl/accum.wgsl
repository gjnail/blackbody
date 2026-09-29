// Accumulate render samples (anti-aliasing, motion blur) and resolve them.
// mode 0: acc_out = acc_in + src    mode 1: out = acc_in / count (written to acc_out)

struct Params {
  n: vec4<f32>,  // width, height, mode, count
};

@group(0) @binding(0) var src_b: texture_2d<f32>;
@group(0) @binding(1) var src_e: texture_2d<f32>;
@group(0) @binding(2) var src_x: texture_2d<f32>;
@group(0) @binding(3) var in_b: texture_2d<f32>;
@group(0) @binding(4) var in_e: texture_2d<f32>;
@group(0) @binding(5) var in_x: texture_2d<f32>;
@group(0) @binding(6) var out_b: texture_storage_2d<rgba32float, write>;
@group(0) @binding(7) var out_e: texture_storage_2d<rgba32float, write>;
@group(0) @binding(8) var out_x: texture_storage_2d<rgba32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let px = vec2<i32>(id.xy);
  if (f32(px.x) >= U.n.x || f32(px.y) >= U.n.y) { return; }
  if (U.n.z < 0.5) {
    let first = U.n.w < 0.5;
    var ab = textureLoad(src_b, px, 0);
    var ae = textureLoad(src_e, px, 0);
    var ax = textureLoad(src_x, px, 0);
    if (!first) {
      ab += textureLoad(in_b, px, 0);
      ae += textureLoad(in_e, px, 0);
      ax += textureLoad(in_x, px, 0);
    }
    textureStore(out_b, px, ab);
    textureStore(out_e, px, ae);
    textureStore(out_x, px, ax);
  } else {
    let k = 1.0 / max(U.n.w, 1.0);
    textureStore(out_b, px, textureLoad(in_b, px, 0) * k);
    textureStore(out_e, px, textureLoad(in_e, px, 0) * k);
    textureStore(out_x, px, textureLoad(in_x, px, 0) * k);
  }
}
