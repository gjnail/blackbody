// Fire, water and lava in one box: the lava laid over the fire and the footage (premultiplied), as
// the backdrop the water refracts and reflects (lava under the water shows through it).

@group(0) @binding(0) var top_t: texture_2d<f32>;    // premultiplied rgba: the lava
@group(0) @binding(1) var back_t: texture_2d<f32>;   // opaque: the fire over the footage
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;

struct Params {
  res: vec4<f32>,
};
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let px = vec2<i32>(id.xy);
  if (f32(px.x) >= U.res.x || f32(px.y) >= U.res.y) { return; }
  let a = textureLoad(top_t, px, 0);
  let b = textureLoad(back_t, px, 0);
  textureStore(dst, px, vec4<f32>(a.rgb + b.rgb * (1.0 - clamp(a.a, 0.0, 1.0)), 1.0));
}
