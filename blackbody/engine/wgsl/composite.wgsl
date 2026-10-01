// Final composite: footage (decoded to linear), heat haze, fire light cast onto the footage,
// the fire element over it (faded into the air's haze by distance), bloom, grain and the footage's own
// noise, then a view transform. The element is seen through the footage's lens: its distortion and
// colour fringing, and halation around the brightest light.
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
  liq: vec4<f32>,    // x = 1 liquid element (aux.x multiplies the footage: wet ground, shadow; haze only its own),
                     //     2 fire and liquid (fire aux, with aux.w the liquid's multiplier on the footage)
  srf: vec4<f32>,    // fire light on surfaces (strength), scorch (strength), _, _
  sw: vec4<f32>,     // soot (strength), wet surfaces (strength), _, _
  oc: vec4<f32>,     // OCIO LUTs: log shaper low and high (log2), footage LUT uses the shaper (1/0), LUT size
  hl: vec4<f32>,     // Standard view: highlights to white (channel crosstalk, 0..0.9), _, _, _
  atm: vec4<f32>,    // atmosphere: haze colour (linear rgb), extinction (1/m; 0 is clear air)
  gr: vec4<f32>,     // footage noise: on (1/0), grain size (px: the Gaussian its noise looks blurred by), _, _
  gchol: array<vec4<f32>, 3>,           // columns of the lower Cholesky factor of the noise's channel correlation
  gsig: array<vec4<f32>, NOISE_BINS>,   // noise standard deviation (linear rgb) at 2^(NOISE_LO + i)
  lens: vec4<f32>,   // radial distortion k1 (of the half-diagonal), colour fringing (scale at the corners), halation, _
  hz: vec4<f32>,     // heat haze: focal length (output px), from the simulation (1/0), the element is behind the
                     //   hot air too (1/0: a molten liquid's own hot air), _
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
@group(0) @binding(9) var surf: texture_2d<f32>;   // fire light on the surface seen (rgb), wetness (a)
@group(0) @binding(10) var mask: texture_2d<f32>;  // scorch, holdout depth * coverage, coverage, soot
@group(0) @binding(11) var lut_view: texture_3d<f32>;   // OCIO display and view (scene-linear through the log shaper -> display-encoded)
@group(0) @binding(12) var lut_plate: texture_3d<f32>;  // OCIO footage colour space -> scene-linear
@group(0) @binding(13) var opl: texture_2d<f32>;   // heat haze: extra optical path (um), depth of the hot air (m), weight
@group(0) @binding(14) var lamp_surf: texture_2d<f32>;  // lights in the set on the surface seen: the footage times 1 + rgb
@group(1) @binding(0) var<uniform> U: Params;

// Per-channel soft shoulder above the knee: identity below it, so footage is untouched.
fn rolloff(c: vec3<f32>, k: f32) -> vec3<f32> {
  let r = 1.0 - k;
  let over = max(c - vec3<f32>(k), vec3<f32>(0.0));
  let soft = vec3<f32>(k) + r * (vec3<f32>(1.0) - exp(-over / r));
  return select(c, soft, c > vec3<f32>(k));
}

// The Standard view's roll-off, applied the way a camera sensor clips: each of its colour channels also
// sees some of the others' light (crosstalk s), so a channel nearing clipping drags the others up with it
// and over-bright flame goes from orange through yellow to white instead of clipping to a flat yellow.
// Identity wherever no channel passes the knee, so footage stays untouched; s = 0 is the plain roll-off.
fn camera_rolloff(c: vec3<f32>, k: f32, s: f32) -> vec3<f32> {
  let raw = c * (1.0 - s) + vec3<f32>(s * (c.r + c.g + c.b) / 3.0);
  let y = rolloff(raw, k);
  return (y - vec3<f32>(s * (y.r + y.g + y.b) / 3.0)) / (1.0 - s);
}

// Noise standard deviation of the footage at linear values v, per channel.
fn plate_sigma(v: vec3<f32>) -> vec3<f32> {
  var s = vec3<f32>(0.0);
  for (var c = 0; c < 3; c++) {
    let f = clamp(log2(max(v[c], 1e-12)) - NOISE_LO, 0.0, f32(NOISE_BINS - 1));
    let i = min(i32(floor(f)), NOISE_BINS - 2);
    s[c] = mix(U.gsig[i][c], U.gsig[i + 1][c], f - f32(i));
  }
  return s;
}

// Three independent standard normals for an integer lattice point.
fn gauss3(c: vec2<i32>, seed: u32) -> vec3<f32> {
  let k = vec3<u32>(bitcast<u32>(c.x), bitcast<u32>(c.y), seed);
  let a = vec3<f32>(pcg3d(k)) * (1.0 / 4294967296.0);
  let b = vec3<f32>(pcg3d(k ^ vec3<u32>(0x9e3779b9u, 0x85ebca6bu, 0xc2b2ae35u))) * (1.0 / 4294967296.0);
  return sqrt(-2.0 * log(max(a, vec3<f32>(1e-7)))) * cos(6.2831853 * b);
}

// Unit-variance noise blurred by a Gaussian of `size` pixels, as footage grain is (0 = every pixel its own).
fn grain_field(px: vec2<i32>, size: f32, seed: u32) -> vec3<f32> {
  if (size < 0.3) { return gauss3(px, seed); }
  let r = min(i32(ceil(2.5 * size)), 4);
  var v = vec3<f32>(0.0);
  var ww = 0.0;
  for (var dy = -r; dy <= r; dy++) {
    for (var dx = -r; dx <= r; dx++) {
      let w = exp(-f32(dx * dx + dy * dy) / (2.0 * size * size));
      v += w * gauss3(px + vec2<i32>(dx, dy), seed);
      ww += w * w;
    }
  }
  return v * inverseSqrt(ww);
}

// The footage's own noise where the fire has changed the picture: just enough that each pixel ends up as
// noisy as footage of its brightness is, counting the footage noise still showing through (plate_part is
// what is left of the footage in the composite, plate its value as decoded).
fn footage_grain(comp: vec3<f32>, plate: vec3<f32>, plate_part: vec3<f32>, px: vec2<i32>) -> vec3<f32> {
  let want = plate_sigma(comp);
  let present = plate_sigma(plate) * min(plate_part / max(plate, vec3<f32>(1e-5)), vec3<f32>(16.0));
  let add = sqrt(max(want * want - present * present, vec3<f32>(0.0)));
  if (max(add.r, max(add.g, add.b)) <= 0.0) { return vec3<f32>(0.0); }
  let z = grain_field(px, U.gr.y, u32(U.grade.z) * 2654435761u + 7u);
  return (mat3x3<f32>(U.gchol[0].xyz, U.gchol[1].xyz, U.gchol[2].xyz) * z) * add;
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
  return camera_rolloff(c, U.grade.w, U.hl.x);
}

// A 3D LUT at c (0..1 on each axis), sampled between its texel centres.
fn lut3(t: texture_3d<f32>, c: vec3<f32>) -> vec3<f32> {
  let n = U.oc.w;
  let q = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)) * ((n - 1.0) / n) + vec3<f32>(0.5 / n);
  return textureSampleLevel(t, lin, q, 0.0).rgb;
}

fn log_shaper(x: vec3<f32>) -> vec3<f32> {
  let lo = U.oc.x;
  return (log2(max(x, vec3<f32>(0.0)) + vec3<f32>(exp2(lo))) - vec3<f32>(lo)) / (U.oc.y - lo);
}

// Scene-linear to display-encoded pixels: the chosen view transform, or an OCIO display and view.
fn view_encode(c: vec3<f32>) -> vec3<f32> {
  if (i32(U.view.x) == 4) { return lut3(lut_view, log_shaper(c)); }
  return linear_to_srgb(view_transform(c));
}

fn decode_plate(c: vec3<f32>) -> vec3<f32> {
  if (i32(U.plate.y) == 4) {
    if (U.oc.z > 0.5) { return lut3(lut_plate, log_shaper(c)); }
    return lut3(lut_plate, c);
  }
  return input_transform(c, i32(U.plate.y));
}

// Where the element, rendered through a perfect pinhole lens, has what the footage's lens shows at uv:
// radial distortion x_d = x_u (1 + k1 r_u^2) with r relative to the half-diagonal, solved for x_u.
fn undistort(uv: vec2<f32>, k1: f32) -> vec2<f32> {
  if (k1 == 0.0) { return uv; }
  let asp = vec2<f32>(U.res.x / U.res.y, 1.0);
  let hd = 0.5 * length(asp);
  let d = (uv - vec2<f32>(0.5)) * asp / hd;
  var u = d;
  for (var i = 0; i < 5; i++) { u = d / (1.0 + k1 * dot(u, u)); }
  return u * hd / asp + vec2<f32>(0.5);
}

// A texture seen through a lens with lateral colour fringing: red a touch larger than green, blue smaller.
fn sample_fringed(t: texture_2d<f32>, uv: vec2<f32>, s: f32) -> vec4<f32> {
  let g = textureSampleLevel(t, lin, uv, 0.0);
  if (s == 0.0) { return g; }
  let r = textureSampleLevel(t, lin, vec2<f32>(0.5) + (uv - vec2<f32>(0.5)) / (1.0 + s), 0.0).r;
  let b = textureSampleLevel(t, lin, vec2<f32>(0.5) + (uv - vec2<f32>(0.5)) / (1.0 - s), 0.0).b;
  return vec4<f32>(r, g.g, b, g.a);
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

  // the element as the footage's lens sees it (the footage already has its own lens)
  let euv = undistort(uv, U.lens.x);
  let liquid = U.liq.x > 0.5 && U.liq.x < 1.5;
  let both = U.liq.x > 1.5;

  // heat haze: the hot air bends the light from the footage behind it by the gradient of the extra
  // optical path across the picture (haze_opl.wgsl); the bend grows with focal length and shrinks with
  // the hot air's distance. A molten liquid's hot air is over it, so it bends the liquid too.
  var off = vec2<f32>(0.0);
  if (U.haze.x > 0.0 && U.hz.y > 0.5 && (!liquid || U.hz.z > 0.5)) {
    let ho = textureSampleLevel(opl, lin, euv, 0.0);
    if (ho.z > 0.0 && ho.y > 0.0) {
      let dt = 1.0 / vec2<f32>(textureDimensions(opl));
      let gx = textureSampleLevel(opl, lin, euv + vec2<f32>(dt.x, 0.0), 0.0).x - textureSampleLevel(opl, lin, euv - vec2<f32>(dt.x, 0.0), 0.0).x;
      let gy = textureSampleLevel(opl, lin, euv + vec2<f32>(0.0, dt.y), 0.0).x - textureSampleLevel(opl, lin, euv - vec2<f32>(0.0, dt.y), 0.0).x;
      let g_px = vec2<f32>(gx, gy) / (2.0 * dt) / U.res.xy;   // micrometres of optical path per output pixel
      off = g_px * (1.0e-6 * U.hz.x * U.hz.x / ho.y * U.haze.x) / U.res.xy;
    }
  }

  let ruv = euv + select(vec2<f32>(0.0), off, U.hz.z > 0.5);
  let b = sample_fringed(beauty, ruv, U.lens.y);
  let e = sample_fringed(emit, ruv, U.lens.y);
  let x = textureSampleLevel(aux, lin, ruv, 0.0);
  let bl = sample_fringed(bloom, ruv, U.lens.y).rgb;
  let gw = textureSampleLevel(glow, lin, euv, 0.0).rgb;
  let tint = U.tint.rgb;
  let hf = select(1.0 - exp(-x.x * 1.5), 0.0, liquid);

  var back = U.bg.rgb;
  var plate_lin = vec3<f32>(0.0);  // the footage as decoded, before the fire lights or scorches it
  if (U.res.z > 0.5) {
    let puv = (uv + off - vec2<f32>(0.5)) * U.plate.zw + vec2<f32>(0.5);
    back = decode_plate(textureSampleLevel(plate, lin, puv, 0.0).rgb) * U.plate.x;
    plate_lin = back;
  } else if (U.view.y > 0.5) {
    let cs = max(U.view.z, 4.0);
    let chk = (i32(floor(f32(px.x) / cs)) + i32(floor(f32(px.y) / cs))) & 1;
    back = select(vec3<f32>(0.18), vec3<f32>(0.24), chk == 1);
  }

  // burnt ground and objects darken; the fire lights the ground and the objects in the shot, facing it
  let sl = textureSampleLevel(surf, lin, euv, 0.0);
  let mk = textureSampleLevel(mask, lin, euv, 0.0);
  back = back * (1.0 - clamp(mk.x * U.srf.y, 0.0, 1.0) * 0.9);
  back = back * (1.0 - clamp(mk.w * U.sw.x, 0.0, 1.0) * 0.85);   // soot on walls, ceilings and the ground
  back = back * (1.0 - clamp(sl.a * U.sw.y, 0.0, 1.0) * 0.45);   // soaked surfaces go darker
  // lights in the set: CG lights brighten the footage (softly limited far up), smoke shadows real ones
  let lc = textureSampleLevel(lamp_surf, lin, euv, 0.0).rgb;
  let lup = max(lc, vec3<f32>(0.0));
  back = back * max(vec3<f32>(1.0) + lup / (vec3<f32>(1.0) + lup / 16.0) + min(lc, vec3<f32>(0.0)), vec3<f32>(0.0));
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
  // halation: light from the brightest flame scatters back through the film (or sensor stack) as a red halo
  let halo = vec3<f32>(1.0, 0.3, 0.08) * (0.2 * U.lens.z * max(luma(bl) - 0.5, 0.0));
  let bloom_rgb = (bl * U.fire.z + halo) * tint * U.fire.x;

  // saturation of the fire contribution
  let fl = luma(fire_g);
  fire_g = mix(vec3<f32>(fl), fire_g, U.grade.x);

  // the air between the camera and the fire: the fire fades into the haze over its distance (aux.y), as
  // the footage behind it already has; the haze's own light replaces what the fire hides
  if (U.atm.w > 0.0) {
    let t_air = exp(-U.atm.w * max(x.y, 0.0));
    fire_g = fire_g * t_air + U.atm.rgb * ((1.0 - t_air) * alpha);
  }

  var comp = lit * (1.0 - alpha) + fire_g + bloom_rgb;
  if (mode == 0 && U.gr.x > 0.5 && U.res.z > 0.5 && !liquid) {
    comp += footage_grain(comp, plate_lin, lit * (1.0 - alpha), px);
  }
  var disp = vec3<f32>(0.0);
  var enc = vec3<f32>(-1.0);  // display-encoded already (the views); otherwise disp is encoded below
  var out_a = 1.0;
  if (mode == 0) {
    enc = view_encode(comp);
  } else if (mode == 1) {
    comp = fire_g + bloom_rgb;
    enc = view_encode(comp + U.bg.rgb * (1.0 - alpha));
    out_a = alpha;
  } else if (mode == 2) {
    disp = vec3<f32>(alpha);
  } else if (mode == 3) {
    enc = view_encode(e.rgb * tint * U.fire.x + bloom_rgb);
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
  if (enc.x < 0.0) { enc = linear_to_srgb(disp); }
  textureStore(out_disp, px, vec4<f32>(enc, out_a));
  textureStore(out_lin, px, vec4<f32>(comp, out_a));
}
