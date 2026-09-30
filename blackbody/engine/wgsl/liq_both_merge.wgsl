// Fire and liquid in one box: the fire in front of the liquid over the liquid layer (which already
// shows the fire behind it, refracted). aux keeps the fire's heat, depth and temperature, with the
// liquid's depth where it is the nearer surface, and w = the liquid's multiplier on the footage (wet
// ground, shadows and caustics).

@group(0) @binding(0) var fb: texture_2d<f32>;
@group(0) @binding(1) var fe: texture_2d<f32>;
@group(0) @binding(2) var fa: texture_2d<f32>;
@group(0) @binding(3) var lb: texture_2d<f32>;
@group(0) @binding(4) var le: texture_2d<f32>;
@group(0) @binding(5) var la: texture_2d<f32>;
@group(0) @binding(6) var out_b: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var out_e: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var out_a: texture_storage_2d<rgba16float, write>;

struct Params {
  res: vec4<f32>,
};
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let px = vec2<i32>(id.xy);
  if (f32(px.x) >= U.res.x || f32(px.y) >= U.res.y) { return; }
  let f = textureLoad(fb, px, 0);
  let l = textureLoad(lb, px, 0);
  let e = textureLoad(fe, px, 0);
  let le_ = textureLoad(le, px, 0);
  let x = textureLoad(fa, px, 0);
  let y = textureLoad(la, px, 0);
  let t = 1.0 - f.a;
  textureStore(out_b, px, vec4<f32>(f.rgb + l.rgb * t, f.a + l.a * t));
  textureStore(out_e, px, vec4<f32>(e.rgb + le_.rgb * t, e.a));
  let liquid_near = y.w > 0.5 && y.y > 0.0 && (x.y <= 0.0 || y.y < x.y);
  textureStore(out_a, px, vec4<f32>(x.x, select(x.y, y.y, liquid_near), x.z, y.x));
}
