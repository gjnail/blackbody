// A molten liquid's skin in relief: lumps and swells a few centimetres high, tens of centimetres across, on
// the surface the particles left (the surface builder smooths the liquid to a gloss, which suits water; a
// pahoehoe skin is lumpy wherever it has been out in the air a while: it sags and swells unevenly as the lobe
// inflates under it, and folds where it is squeezed). Laid out in the crust coordinate (liq_crust_adv.wgsl),
// so the lumps ride the flow; fresh melt is smooth, and they grow in over Crust forms in. They are moved into
// the signed distance itself, so the march finds them: they shape the silhouette and catch the light, and
// the finer relief is left to the shading (liq_lava_shade.wgsl). Not at the ground, so the lobe's foot
// stays on it; and only on bulk lava: a film or a stray blob a few centimetres thin (lava spreading over or
// under water, over a ledge) would break up into shards under lumps that size, so there they fade out (a
// probe a few centimetres in from the surface must still be well inside). A skin only (not a crust of plates).
//!include common.wgsl
//!include noise.wgsl

struct Params {
  n: vec4<f32>,      // simulation grid dims, metres per cell
  nf: vec4<f32>,     // surface grid dims, surface cells per grid cell
  org: vec4<f32>,    // grid corner (fire-local m), relief strength
  lava: array<vec4<f32>, 8>,
  lava_bb: array<vec4<f32>, 32>,
};

@group(0) @binding(0) var src: texture_3d<f32>;
@group(0) @binding(1) var crust_t: texture_3d<f32>;   // crust coordinate less place (cells), skin age (s)
@group(0) @binding(2) var dst: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn crust_samp(p: vec3<f32>) -> vec4<f32> {
  let n = vec3<i32>(textureDimensions(crust_t));
  let q = p - vec3<f32>(0.5);
  let b = vec3<i32>(floor(q));
  let f = q - floor(q);
  var s = vec4<f32>(0.0);
  for (var o = 0; o < 8; o++) {
    let oo = vec3<i32>(o & 1, (o >> 1) & 1, o >> 2);
    let c = clamp(b + oo, vec3<i32>(0), n - vec3<i32>(1));
    let wv = mix(vec3<f32>(1.0) - f, f, vec3<f32>(oo));
    s += wv.x * wv.y * wv.z * textureLoad(crust_t, c, 0);
  }
  return s;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let nf = vec3<i32>(U.nf.xyz);
  let c = vec3<i32>(id);
  if (c.x >= nf.x || c.y >= nf.y || c.z >= nf.z) { return; }
  let s = textureLoad(src, c, 0);
  // only near the surface (the relief moves it by a couple of cells at most)
  if (abs(s.x) > 4.0 || U.lava[1].w > 0.5) {
    textureStore(dst, c, s);
    return;
  }
  let ds = U.n.w / U.nf.w;                                   // metres per surface cell
  // thick enough? the surface's normal from the distance's gradient, and the distance a few centimetres in
  let gx = textureLoad(src, clamp(c + vec3<i32>(1, 0, 0), vec3<i32>(0), nf - 1), 0).x
         - textureLoad(src, clamp(c - vec3<i32>(1, 0, 0), vec3<i32>(0), nf - 1), 0).x;
  let gy = textureLoad(src, clamp(c + vec3<i32>(0, 1, 0), vec3<i32>(0), nf - 1), 0).x
         - textureLoad(src, clamp(c - vec3<i32>(0, 1, 0), vec3<i32>(0), nf - 1), 0).x;
  let gz = textureLoad(src, clamp(c + vec3<i32>(0, 0, 1), vec3<i32>(0), nf - 1), 0).x
         - textureLoad(src, clamp(c - vec3<i32>(0, 0, 1), vec3<i32>(0), nf - 1), 0).x;
  let g = vec3<f32>(gx, gy, gz);
  if (dot(g, g) < 1e-4) {
    textureStore(dst, c, s);
    return;
  }
  let nrm = normalize(g);
  let probe = vec3<f32>(c) - nrm * (s.x + 0.035 / ds);       // 3.5 cm in from the surface point
  let phi_in = textureLoad(src, clamp(vec3<i32>(round(probe)), vec3<i32>(0), nf - 1), 0).x;
  let bulk = smoothstep(-0.3, -1.2, phi_in);
  if (bulk <= 0.0) {
    textureStore(dst, c, s);
    return;
  }
  let p = (vec3<f32>(c) + vec3<f32>(0.5)) * (U.n.xyz / U.nf.xyz);   // simulation grid cells
  let k = crust_samp(p);
  let X = U.org.xyz + (p + k.xyz) * U.n.w;                    // the crust coordinate (m)
  let tskin = max(U.lava[3].x, 1e-3);
  let grown = 0.45 + 0.55 * smoothstep(0.0, 2.0 * tskin, k.w);
  let hy = (f32(c.y) + 0.5) * ds;
  let foot = smoothstep(0.015, 0.05, hy);
  // (gradient noise spreads about +-0.3: these are lumps of about +-2 cm, +-1 cm and +-3 mm)
  let d = 0.065 * gnoise(X / 0.32 + vec3<f32>(6.6, 3.1, 0.2))
        + 0.032 * gnoise(X / 0.13 + vec3<f32>(2.4, 7.3, 5.8))
        + 0.011 * gnoise(X / 0.055 + vec3<f32>(8.8, 1.6, 4.0));
  let disp = d * grown * foot * bulk * U.org.w;               // metres, outward
  textureStore(dst, c, vec4<f32>(s.x - disp / ds, s.yzw));
}
