// Ice: frost, rime and the sparkle of its crystals, over what the march drew (liq_march.wgsl leaves the
// point, normal and ray of every pixel whose surface is frozen in the surface-hit buffer, kind 2;
// liq_lava.wgsl). Kept out of the march, whose compile every loop there lengthens by minutes.
//
// - Frost: ice frozen fast from drops and spray is white rime, and ice left below freezing grows
//   feathers of hoarfrost: a matte white coat over the glassy ice, lit by the sky and the key light.
//   Melting ice (the air above freezing) is wet and glossy, with almost none.
// - Crystals: the flat facets of the ice's crystals (the gradient of the distance to the nearest
//   crystal centre) catch the key light as sparkles, most on frost.
//
// - Ice under water: ice (n 1.31) in water (n 1.333) is nearly invisible straight on, but where a ray
//   through the water meets an ice face at a grazing angle it reflects, up to totally (water to ice is a
//   drop in index): the faces of a submerged cube flash and outline it. For every pixel where the camera
//   sees the water's surface (the march's depth and matte), the refracted ray is followed a little way
//   through the frozen-share field to the first ice face, and that reflection is laid over the pixel.
//
// The ice's own look through it (clear or milky as it froze) is the march's, from the dye field.
//!include noise.wgsl

struct Params {
  inv_vp: mat4x4<f32>,  // clip -> world
  w2g: mat4x4<f32>,     // world -> grid cells
  n: vec4<f32>,      // simulation grid dims, metres per cell
  res: vec4<f32>,    // width, height
  org: vec4<f32>,    // grid corner (fire-local m)
  sky: vec4<f32>,    // ambient light (rgb)
  sun: vec4<f32>,    // key light (rgb, colour x intensity)
  lg: vec4<f32>,     // toward the key light (grid axes)
  ice: vec4<f32>,    // on (1/0), frost amount, _, melting (0..1)
  ice2: vec4<f32>,   // frost colour (rgb), crystal size (m)
  sub: vec4<f32>,    // ice under water: on (1/0), surface grid cells per simulation cell, steps, index of refraction
};

@group(0) @binding(0) var heat_t: texture_3d<f32>;    // surface grid: w = frozen share
@group(0) @binding(1) var dye_t: texture_3d<f32>;     // simulation grid: w = scattering (1/m), the ice's cloudiness
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var<storage, read> gbuf: array<vec4<f32>>;
@group(0) @binding(4) var beauty_in: texture_2d<f32>;
@group(0) @binding(5) var emit_in: texture_2d<f32>;
@group(0) @binding(6) var out_beauty: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var out_emit: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var aux_t: texture_2d<f32>;     // the march's aux: y = depth (m)
@group(0) @binding(9) var mask_t: texture_2d<f32>;    // the march's mattes: x = the liquid
@group(0) @binding(10) var surf_t: texture_3d<f32>;   // the surface's signed distance (x)
@group(1) @binding(0) var<uniform> U: Params;

fn worley(p: vec3<f32>) -> vec2<f32> {
  let i = floor(p);
  let f = p - i;
  var d1 = 8.0;
  var d2 = 8.0;
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let c = vec3<f32>(f32(x), f32(y), f32(z));
        let h = vec3<f32>(pcg3d(vec3<u32>(vec3<i32>(i + c) + vec3<i32>(1048576)))) * (1.0 / 4294967296.0);
        let r = c + h - f;
        let d = dot(r, r);
        if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) { d2 = d; }
      }
    }
  }
  return sqrt(vec2<f32>(d1, d2));
}

fn world(p: vec3<f32>) -> vec3<f32> { return U.org.xyz + p * U.n.w; }

// The flat facets of the ice's crystals tilt the normal: the gradient of the distance to the nearest
// crystal centre.
fn ice_facets(nrm: vec3<f32>, p: vec3<f32>, amt: f32) -> vec3<f32> {
  let q = world(p) / max(U.ice2.w, 1e-4);
  let e = 0.08;
  let f0 = worley(q).x;
  let g = vec3<f32>(worley(q + vec3<f32>(e, 0.0, 0.0)).x - f0, worley(q + vec3<f32>(0.0, e, 0.0)).x - f0,
                    worley(q + vec3<f32>(0.0, 0.0, e)).x - f0) / e;
  let tg = g - nrm * dot(nrm, g);
  return normalize(nrm - 0.35 * amt * tg);
}

// Frost over the ice at p: a matte coat where it froze white, and feathers of hoarfrost.
fn frost_cover(p: vec3<f32>, cloud: f32) -> f32 {
  let amt = U.ice.y * (1.0 - 0.85 * U.ice.w);
  if (amt <= 0.0) { return 0.0; }
  let q = world(p) / max(U.ice2.w, 1e-4) * 0.7;
  let warp = vec3<f32>(gnoise(q * 0.5), gnoise(q * 0.5 + vec3<f32>(5.2, 1.3, 7.1)), gnoise(q * 0.5 + vec3<f32>(9.4, 3.7, 2.2)));
  var r = 0.0;
  var a = 0.5;
  var f = 1.0;
  for (var o = 0; o < 4; o++) {
    r += a * (1.0 - abs(gnoise(q * f + warp * 1.5)));
    f *= 2.07;
    a *= 0.5;
  }
  let feathers = smoothstep(0.62, 0.88, r);
  return clamp(amt * (0.35 * feathers * (1.0 - U.ice.w) + 0.9 * cloud), 0.0, 1.0);
}

fn ice_s(q: vec3<f32>) -> f32 { return textureSampleLevel(heat_t, lin, q / U.n.xyz, 0.0).w; }

fn fresnel(cosi: f32, n1: f32, n2: f32) -> f32 {
  let eta = n1 / n2;
  let sint2 = eta * eta * (1.0 - cosi * cosi);
  if (sint2 >= 1.0) { return 1.0; }
  let cost = sqrt(1.0 - sint2);
  let rs = (n1 * cosi - n2 * cost) / (n1 * cosi + n2 * cost);
  let rp = (n1 * cost - n2 * cosi) / (n1 * cost + n2 * cosi);
  return 0.5 * (rs * rs + rp * rp);
}

// Where the camera sees the water's surface: the ray refracted into it, followed to the first ice face.
fn under_water(px: vec2<i32>) {
  let a = textureLoad(aux_t, px, 0);
  let mk = textureLoad(mask_t, px, 0);
  if (mk.x < 0.5 || a.y <= 0.0) { return; }
  let uv = (vec2<f32>(px) + vec2<f32>(0.5)) / U.res.xy;
  let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let pn = U.inv_vp * vec4<f32>(ndc, 0.0, 1.0);
  let pf = U.inv_vp * vec4<f32>(ndc, 1.0, 1.0);
  let ro_w = pn.xyz / pn.w;
  let rd_w = normalize(pf.xyz / pf.w - ro_w);
  let p0 = (U.w2g * vec4<f32>(ro_w + rd_w * a.y, 1.0)).xyz;
  let rd = normalize((U.w2g * vec4<f32>(rd_w, 0.0)).xyz);
  // the surface's normal there, from its distance field
  let e = 0.5 / U.sub.y;
  let g = vec3<f32>(textureSampleLevel(surf_t, lin, (p0 + vec3<f32>(e, 0.0, 0.0)) / U.n.xyz, 0.0).x -
                    textureSampleLevel(surf_t, lin, (p0 - vec3<f32>(e, 0.0, 0.0)) / U.n.xyz, 0.0).x,
                    textureSampleLevel(surf_t, lin, (p0 + vec3<f32>(0.0, e, 0.0)) / U.n.xyz, 0.0).x -
                    textureSampleLevel(surf_t, lin, (p0 - vec3<f32>(0.0, e, 0.0)) / U.n.xyz, 0.0).x,
                    textureSampleLevel(surf_t, lin, (p0 + vec3<f32>(0.0, 0.0, e)) / U.n.xyz, 0.0).x -
                    textureSampleLevel(surf_t, lin, (p0 - vec3<f32>(0.0, 0.0, e)) / U.n.xyz, 0.0).x);
  if (dot(g, g) < 1e-12) { return; }
  var nrm = normalize(g);
  if (dot(nrm, rd) > 0.0) { nrm = -nrm; }
  let t = refract(rd, nrm, 1.0 / U.sub.w);
  if (dot(t, t) < 1e-6) { return; }
  // find the first ice face along it (the frozen share crossing a half)
  let stp = 0.5 / U.sub.y;
  var q = p0 - nrm * (0.5 / U.sub.y);
  var prev = ice_s(q);
  if (prev > 0.5) { return; }   // the surface here is ice itself (the frost pass's)
  let ns = i32(U.sub.z);
  for (var i = 0; i < ns; i++) {
    q += t * stp;
    if (any(q < vec3<f32>(0.0)) || any(q > U.n.xyz)) { return; }
    if (textureSampleLevel(surf_t, lin, q / U.n.xyz, 0.0).x > 0.0) { return; }   // out of the water again
    let s = ice_s(q);
    if (s > 0.5) {
      // the ice's face: its normal from the frozen share's gradient, facing the ray
      let d = 0.75 / U.sub.y;
      var fa = -vec3<f32>(ice_s(q + vec3<f32>(d, 0.0, 0.0)) - ice_s(q - vec3<f32>(d, 0.0, 0.0)),
                          ice_s(q + vec3<f32>(0.0, d, 0.0)) - ice_s(q - vec3<f32>(0.0, d, 0.0)),
                          ice_s(q + vec3<f32>(0.0, 0.0, d)) - ice_s(q - vec3<f32>(0.0, 0.0, d)));
      if (dot(fa, fa) < 1e-8) { return; }
      fa = normalize(fa);
      if (dot(fa, t) > 0.0) { fa = -fa; }
      let R = fresnel(clamp(-dot(t, fa), 0.0, 1.0), U.sub.w, 1.31);
      if (R < 0.002) { return; }
      // what the face reflects: mostly the bright water surface and sky seen from below
      let rdir = reflect(t, fa);
      let lg = normalize(U.lg.xyz);
      let refl = U.sky.rgb * (0.6 + 0.4 * max(rdir.y, 0.0)) + U.sun.rgb * 0.15 * max(dot(rdir, lg), 0.0);
      let b = textureLoad(beauty_in, px, 0);
      let keep = mk.x;
      let depth_fade = exp(-f32(i) * stp * U.n.w * 0.5);
      let r = R * depth_fade;
      textureStore(out_beauty, px, vec4<f32>(mix(b.rgb, refl * keep, r), b.a));
      return;
    }
    prev = s;
  }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let px = vec2<i32>(id.xy);
  let w = i32(U.res.x);
  if (px.x >= w || px.y >= i32(U.res.y)) { return; }
  let k = u32(px.x + px.y * w) * 3u;
  let g2 = gbuf[k + 2u];
  if (u32(g2.w + 0.5) != 2u) {
    if (U.sub.x > 0.5) { under_water(px); }
    return;
  }
  let p = gbuf[k].xyz;
  let nrm = normalize(gbuf[k + 1u].xyz);
  let keep = gbuf[k + 1u].w;
  let rd = normalize(g2.xyz);
  let uvw = p / U.n.xyz;
  let ic = clamp(textureSampleLevel(heat_t, lin, uvw, 0.0).w, 0.0, 1.0);
  if (ic <= 0.02) { return; }
  let cloud = clamp(textureSampleLevel(dye_t, lin, uvw, 0.0).w / 250.0, 0.0, 1.0);
  let fc = frost_cover(p, cloud) * ic;
  let nf = ice_facets(nrm, p, ic);
  let lg = normalize(U.lg.xyz);
  let light = U.sky.rgb * 0.95 + U.sun.rgb * (0.25 + 0.6 * max(dot(nf, lg), 0.0));
  // crystals catching the key light, glittering most on frost
  let hv = normalize(lg - rd);
  let sparkle = U.sun.rgb * (pow(max(dot(nf, hv), 0.0), 600.0) * (0.4 + 2.0 * fc) * ic * select(0.0, 1.0, dot(nrm, lg) > 0.0));
  let b = textureLoad(beauty_in, px, 0);
  let e = textureLoad(emit_in, px, 0);
  let col = mix(b.rgb, U.ice2.rgb * light * keep, fc) + sparkle * keep;
  textureStore(out_beauty, px, vec4<f32>(col, b.a));
  textureStore(out_emit, px, vec4<f32>(e.rgb * (1.0 - fc) + sparkle * (0.5 * keep), e.a));
}
