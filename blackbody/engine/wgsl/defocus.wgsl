// Depth of field for the rendered element (beauty, emission, aux): each pixel is blurred over the circle
// of confusion of its own depth, the lens aperture seen from that far out of focus, plus the footage's
// own softness. Pixels with nothing rendered take the fire's distance, so the blur spreads past the
// element's edge as a lens spreads it. A sample only counts where its own blur reaches this pixel, so
// something sharp does not smear over a blurred neighbour.

struct Params {
  n: vec4<f32>,      // width, height, _, _
  lens: vec4<f32>,   // circle of confusion scale A (px; coc = A |d - s| / d), focus distance s (m), softness (px), fire distance (m)
};

@group(0) @binding(0) var sb: texture_2d<f32>;   // beauty (premultiplied), alpha
@group(0) @binding(1) var se: texture_2d<f32>;   // emission
@group(0) @binding(2) var sx: texture_2d<f32>;   // aux: heat, depth (m), max Kelvin / 1000, alpha
@group(0) @binding(3) var lin: sampler;
@group(0) @binding(4) var ob: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var oe: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var ox: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

const MAX_R: f32 = 48.0;
const GOLDEN: f32 = 2.39996323;

// Blur radius (px) for a pixel whose aux is x: half the circle of confusion, with the softness added in quadrature.
fn radius_of(x: vec4<f32>) -> f32 {
  let d = select(U.lens.w, x.y, x.w > 0.01 && x.y > 0.0);
  let coc = U.lens.x * abs(d - U.lens.y) / max(d, 1e-3);
  return min(sqrt(0.25 * coc * coc + U.lens.z * U.lens.z), MAX_R);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let px = vec2<i32>(id.xy);
  if (f32(px.x) >= U.n.x || f32(px.y) >= U.n.y) { return; }
  let x0 = textureLoad(sx, px, 0);
  let r = radius_of(x0);
  if (r < 0.35) {
    textureStore(ob, px, textureLoad(sb, px, 0));
    textureStore(oe, px, textureLoad(se, px, 0));
    textureStore(ox, px, x0);
    return;
  }
  let inv = 1.0 / U.n.xy;
  let centre = vec2<f32>(px) + vec2<f32>(0.5);
  let count = clamp(i32(ceil(r * r * 0.8)), 12, 192);
  var sum_b = vec4<f32>(0.0);
  var sum_e = vec4<f32>(0.0);
  var sum_x = vec4<f32>(0.0);
  var sum_d = 0.0;
  var w_d = 0.0;
  var w_sum = 0.0;
  for (var i = 0; i < count; i++) {
    // a Vogel spiral covers the disc evenly
    let rr = r * sqrt((f32(i) + 0.5) / f32(count));
    let th = f32(i) * GOLDEN;
    let q = centre + rr * vec2<f32>(cos(th), sin(th));
    let uv = q * inv;
    let xq = textureSampleLevel(sx, lin, uv, 0.0);
    let w = clamp(radius_of(xq) - rr + 1.0, 0.0, 1.0);
    if (w <= 0.0) { continue; }
    sum_b += w * textureSampleLevel(sb, lin, uv, 0.0);
    sum_e += w * textureSampleLevel(se, lin, uv, 0.0);
    sum_x += w * xq;
    sum_d += w * xq.w * xq.y;   // depth, weighted by how much is there
    w_d += w * xq.w;
    w_sum += w;
  }
  let k = 1.0 / max(w_sum, 1e-6);
  var xo = sum_x * k;
  xo.y = select(0.0, sum_d / max(w_d, 1e-6), w_d > 1e-6);
  textureStore(ob, px, sum_b * k);
  textureStore(oe, px, sum_e * k);
  textureStore(ox, px, xo);
}
