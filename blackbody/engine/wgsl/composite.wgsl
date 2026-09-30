// Final composite: footage (decoded to linear), heat haze, fire light cast onto the footage,
// the fire element over it, bloom, grain, then a view transform.
// out_disp: display-encoded 8-bit (viewport, video, PNG); out_lin: scene-linear half float (EXR).
//!include common.wgsl
//!include noise.wgsl
//!include colour.wgsl

struct Params {
  res: vec4<f32>,    // output width, height, footage present (1/0), view mode
  plate: vec4<f32>,  // footage gain, input transform (0 sRGB, 1 Rec.709, 2 linear, 3 ACEScg), fit scale x, fit scale y
  fire: vec4<f32>,   // fire gain, smoke opacity, bloom strength, light cast strength
  haze: vec4<f32>,   // strength (px), frequency (per px), speed, time (s)
  grade: vec4<f32>,  // saturation, grain, frame seed, highlight knee
  view: vec4<f32>,   // view transform (0 standard, 1 AgX, 2 ACES fit, 3 raw), background mode, checker size, depth range (m)
  bg: vec4<f32>,     // background colour (linear) when there is no footage
  tint: vec4<f32>,   // fire colour balance (rgb multipliers), light-cast falloff exponent
  liq: vec4<f32>,    // x = 1 liquid element (aux.x multiplies the footage: wet ground, shadow; no haze),
                     //     2 fire and liquid (fire aux, with aux.w the liquid's multiplier on the footage)
  srf: vec4<f32>,    // fire light on surfaces (strength), scorch (strength), _, _
};

@group(0) @binding(0) var plate: texture_2d<f32>;
@group(0) @binding(1) var beauty: texture_2d<f32>;
@group(0) @binding(2) var emit: texture_2d<f32>;
@group(0) @binding(3) var aux: texture_2d<f32>;
@group(0) @binding(4) var bloom: texture_2d<f32>;
@group(0) @binding(5) var glow: texture_2d<f32>;
@group(0) @binding(6) var lin: sampler;
@group(0) @binding(7) var out_disp: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(8) var out_lin: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var surf: texture_2d<f32>;   // fire light on the surface seen (rgb), holdout coverage (a)
@group(0) @binding(10) var mask: texture_2d<f32>;  // scorch, holdout depth * coverage, coverage
@group(1) @binding(0) var<uniform> U: Params;

// Per-channel soft shoulder above the knee: identity below it, so footage is untouched.
fn rolloff(c: vec3<f32>, k: f32) -> vec3<f32> {
  let r = 1.0 - k;
  let over = max(c - vec3<f32>(k), vec3<f32>(0.0));
  let soft = vec3<f32>(k) + r * (vec3<f32>(1.0) - exp(-over / r));
  return select(c, soft, c > vec3<f32>(k));
}

fn agx(v_in: vec3<f32>) -> vec3<f32> {
  let m = mat3x3<f32>(0.842479062253094, 0.0423282422610123, 0.0423756549057051,
                      0.0784335999999992, 0.878468636469772, 0.0784336,
                      0.0792237451477643, 0.0791661274605434, 0.879142973793104);
  let mi = mat3x3<f32>(1.19687900512017, -0.0528968517574562, -0.0529716355144438,
                       -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
                       -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
  let min_ev = -12.47393;
  let max_ev = 4.026069;
  var v = m * max(v_in, vec3<f32>(1e-10));
  v = clamp(log2(v), vec3<f32>(min_ev), vec3<f32>(max_ev));
  v = (v - vec3<f32>(min_ev)) / (max_ev - min_ev);
  let x2 = v * v;
  let x4 = x2 * x2;
  v = 15.5 * x4 * x2 - 40.14 * x4 * v + 31.96 * x4 - 6.868 * x2 * v + 0.4298 * x2 + 0.1191 * v - vec3<f32>(0.00232);
  v = mi * v;
  return pow(max(v, vec3<f32>(0.0)), vec3<f32>(2.2));  // back to linear for the sRGB encode
}

fn aces_fit(x: vec3<f32>) -> vec3<f32> {
  return clamp((x * (2.51 * x + vec3<f32>(0.03))) / (x * (2.43 * x + vec3<f32>(0.59)) + vec3<f32>(0.14)), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn view_transform(c: vec3<f32>) -> vec3<f32> {
  let v = i32(U.view.x);
  if (v == 1) { return agx(c); }
  if (v == 2) { return aces_fit(c * 0.8); }
  if (v == 3) { return clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)); }
  return rolloff(c, U.grade.w);
}

fn heatmap(x: f32) -> vec3<f32> {
  let t = clamp(x, 0.0, 1.0);
  return clamp(vec3<f32>(1.5 - abs(4.0 * t - 3.0), 1.5 - abs(4.0 * t - 2.0), 1.5 - abs(4.0 * t - 1.0)), vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let px = vec2<i32>(id.xy);
  if (f32(px.x) >= U.res.x || f32(px.y) >= U.res.y) { return; }
  let uv = (vec2<f32>(px) + 0.5) / U.res.xy;
  let mode = i32(U.res.w);

  let b = textureSampleLevel(beauty, lin, uv, 0.0);
  let e = textureSampleLevel(emit, lin, uv, 0.0);
  let x = textureSampleLevel(aux, lin, uv, 0.0);
  let bl = textureSampleLevel(bloom, lin, uv, 0.0).rgb;
  let gw = textureSampleLevel(glow, lin, uv, 0.0).rgb;
  let tint = U.tint.rgb;

  // heat haze: displace the footage by the gradient of rising noise, scaled by the heat above it
  let liquid = U.liq.x > 0.5 && U.liq.x < 1.5;
  let both = U.liq.x > 1.5;
  let hf = select(1.0 - exp(-x.x * 1.5), 0.0, liquid);
  var off = vec2<f32>(0.0);
  if (U.haze.x > 0.0 && hf > 1e-3) {
    let q = vec3<f32>(vec2<f32>(px) * U.haze.y + vec2<f32>(0.0, U.haze.w * U.haze.z), U.haze.w * 0.7);
    let nd = gnoise_d(q);
    off = nd.yz * (U.haze.x * hf) / U.res.xy;
  }

  var back = U.bg.rgb;
  if (U.res.z > 0.5) {
    let puv = (uv + off - vec2<f32>(0.5)) * U.plate.zw + vec2<f32>(0.5);
    back = input_transform(textureSampleLevel(plate, lin, puv, 0.0).rgb, i32(U.plate.y)) * U.plate.x;
  } else if (U.view.y > 0.5) {
    let cs = max(U.view.z, 4.0);
    let chk = (i32(floor(f32(px.x) / cs)) + i32(floor(f32(px.y) / cs))) & 1;
    back = select(vec3<f32>(0.18), vec3<f32>(0.24), chk == 1);
  }

  // burnt ground and objects darken; the fire lights the ground and the objects in the shot, facing it
  let sl = textureSampleLevel(surf, lin, uv, 0.0);
  let mk = textureSampleLevel(mask, lin, uv, 0.0);
  back = back * (1.0 - clamp(mk.x * U.srf.y, 0.0, 1.0) * 0.9);
  let ls = sl.rgb * tint * U.fire.x * U.srf.x;
  back = back + back * ls / (vec3<f32>(1.0) + 0.25 * ls);  // a soft shoulder: at most about 4x the footage

  // the fire lights the scene around it
  let spill = gw * tint * U.fire.x * U.fire.w;
  var lit = back + back * spill;
  if (liquid) { lit = back * x.x; }
  if (both) { lit = lit * x.w; }

  let fire_rgb = b.rgb * tint * U.fire.x;
  let alpha = clamp(b.a * U.fire.y, 0.0, 1.0);
  var grain = 0.0;
  if (U.grade.y > 0.0) {
    let r = rand1(u32(px.x) * 1973u + u32(px.y) * 9277u + u32(U.grade.z) * 26699u);
    grain = (r - 0.5) * U.grade.y;
  }
  var fire_g = fire_rgb * (1.0 + grain * inverseSqrt(max(luma(fire_rgb), 1e-3)) * 0.25);
  let bloom_rgb = bl * tint * U.fire.x * U.fire.z;

  // saturation of the fire contribution
  let fl = luma(fire_g);
  fire_g = mix(vec3<f32>(fl), fire_g, U.grade.x);

  var comp = lit * (1.0 - alpha) + fire_g + bloom_rgb;
  var disp = vec3<f32>(0.0);
  var out_a = 1.0;
  if (mode == 0) {
    disp = view_transform(comp);
  } else if (mode == 1) {
    comp = fire_g + bloom_rgb;
    disp = view_transform(comp + U.bg.rgb * (1.0 - alpha));
    out_a = alpha;
  } else if (mode == 2) {
    disp = vec3<f32>(alpha);
  } else if (mode == 3) {
    disp = view_transform(e.rgb * tint * U.fire.x + bloom_rgb);
  } else if (mode == 4) {
    disp = select(heatmap(hf), vec3<f32>(clamp(x.x, 0.0, 1.0)), liquid);
  } else if (mode == 5) {
    let dn = select(0.0, 1.0 - clamp(x.y / max(U.view.w, 1e-3), 0.0, 1.0), x.w > 0.01);
    disp = vec3<f32>(dn);
  } else if (liquid) {
    disp = heatmap(x.z) * step(0.01, x.w);
  } else {
    disp = heatmap((x.z - 0.8) / 1.8) * step(0.01, x.z);
  }
  textureStore(out_disp, px, vec4<f32>(linear_to_srgb(disp), out_a));
  textureStore(out_lin, px, vec4<f32>(comp, out_a));
}
