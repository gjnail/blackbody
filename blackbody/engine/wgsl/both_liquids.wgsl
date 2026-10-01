// Fire, water and lava in one box: the two liquids as one layer, the nearer of them per pixel. The
// water was traced over the lava (it shows the lava behind and under it, refracted), so where the
// water is nearer, or the lava is not there, the water's pixel stands, laid over the lava where the
// water ray missed (spray over lava); where the lava is nearer, it hides the water behind it.
// aux (both): x = the multiplier on the footage (wet ground, shadows), y = depth (m, from where the
// ray enters the box), z = _, w = coverage.

@group(0) @binding(0) var kb: texture_2d<f32>;   // the lava: beauty (premultiplied), emission, aux
@group(0) @binding(1) var ke: texture_2d<f32>;
@group(0) @binding(2) var ka: texture_2d<f32>;
@group(0) @binding(3) var lb: texture_2d<f32>;   // the water, likewise
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
  let b1 = textureLoad(kb, px, 0);
  let e1 = textureLoad(ke, px, 0);
  let a1 = textureLoad(ka, px, 0);
  let b2 = textureLoad(lb, px, 0);
  let e2 = textureLoad(le, px, 0);
  let a2 = textureLoad(la, px, 0);
  let lava_hit = a1.w > 0.5 && a1.y > 0.0;
  let water_hit = a2.w > 0.5 && a2.y > 0.0;
  if (lava_hit && (!water_hit || a1.y < a2.y)) {
    textureStore(out_b, px, b1);
    textureStore(out_e, px, e1);
    textureStore(out_a, px, vec4<f32>(a1.x, a1.y, a1.z, a1.w));
    return;
  }
  if (water_hit) {
    textureStore(out_b, px, b2);
    textureStore(out_e, px, e2);
    textureStore(out_a, px, a2);
    return;
  }
  // neither is solidly there: the water's spray (and wet ground) over the lava's
  let t = 1.0 - clamp(b2.a, 0.0, 1.0);
  textureStore(out_b, px, vec4<f32>(b2.rgb + b1.rgb * t, b2.a + b1.a * t));
  textureStore(out_e, px, vec4<f32>(e2.rgb + e1.rgb * t, max(e2.a, e1.a)));
  let depth = select(a1.y, a2.y, a2.y > 0.0);
  textureStore(out_a, px, vec4<f32>(a1.x * a2.x, depth, a2.z, 1.0 - (1.0 - a1.w) * (1.0 - a2.w)));
}
