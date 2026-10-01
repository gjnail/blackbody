// Liquid renderer: ray tracing of the liquid surface (a signed distance on the surface grid) as a
// dielectric in front of the footage.
//
// At the first surface the ray splits by exact Fresnel into a reflection and a refraction. The
// reflection sees the ground (the footage, where the camera sees the ground), a collider, the
// environment or the liquid itself. The refraction travels through the body, absorbing per
// Beer-Lambert and scattering if the liquid is murky, reflects totally inside where it must, and
// leaves (and may re-enter other drops) until it reaches the ground, a collider or the backdrop.
// All three are looked up in the footage at the screen position of the point the ray reaches, so
// the footage bends through the liquid the way a real scene would. A key light adds glints (GGX
// microfacets), and caustics on the ground (a light map traced through the surface beforehand).
//
// Colliders are real objects in the footage: seen directly they hold the liquid out (it does not
// show through them), seen through or in the liquid they show their own pixels. For shots without
// footage they can be drawn as grey stand-ins instead.
//
// Open water: with a water level, the liquid carries on past the open sides of the box as a flat
// sheet out to the horizon, and rays that leave the box under water carry on through it.
//
// Dye: where sources pour a dye (ink, blood, milk, mud) the refracted rays through the liquid add
// its absorption and scattering along their way (a field of the particles' dye on the simulation
// grid), so it colours and clouds the liquid where it has spread.
//
// Lamps: the lights in the set (point, spot, area) and, with fire in the shot, the point lights
// standing in for the flames (lights.wgsl) light the liquid too: a glint of each on the surface
// (wider for a bigger lamp) and their light on foam, spray and murky water. No shadows.
//
// Holdouts from the footage: its matte hides the liquid wherever it is set, its depth pass hides the
// liquid behind the footage's own surfaces (a real wall, a real step), so the liquid runs behind and
// in front of them.
//
// Under water: a camera below the surface (in the box's liquid, or under the open water) looks out
// through it from inside: the sky only through Snell's window overhead, the surface a mirror of the
// water outside it, everything hazed by the water, and shafts of sunlight where the waves focus it.
//
// Whitewater: spray scatters light along the camera ray as a white mist, foam covers the surface
// as a lacy white diffuse layer (and hides the glints under it), bubbles scatter inside the body.
//
// The environment for reflections is an HDRI of the set if there is one, else a sky from the
// ambient colour; directions in front of the camera see the footage at the far point in that
// direction, directions behind it see the footage mirrored.
//
// beauty: radiance (the footage seen through the liquid included), alpha = liquid coverage
// emit:   key-light glints, compressed (drives bloom); alpha = foam coverage
// aux:    x = multiplier on the footage outside the liquid (wet ground, shadow, caustics), y = depth (m),
//         z = liquid speed / 10 (m/s), w = coverage
//!include common.wgsl
//!include noise.wgsl
//!include colour.wgsl
//!include meshsdf.wgsl
//!include colliders.wgsl
//!include liq_ice_look.wgsl

struct Params {
  inv_vp: mat4x4<f32>,  // clip -> world
  vp: mat4x4<f32>,      // world -> clip
  w2g: mat4x4<f32>,     // world -> grid cells
  g2w: mat4x4<f32>,     // grid cells -> world
  n: vec4<f32>,         // grid dims, metres per cell
  nf: vec4<f32>,        // surface grid dims, surface cells per grid cell
  res: vec4<f32>,       // width, height, sub-pixel jitter (px)
  frame: vec4<f32>,     // noise seed, shutter offset (s), motion blur on, time (s)
  plate: vec4<f32>,     // footage present, input transform, fit scale x, fit scale y
  back: vec4<f32>,      // background colour without footage (linear rgb), checkerboard
  scene: vec4<f32>,     // camera position (world), backdrop distance (m)
  fwd: vec4<f32>,       // camera forward (world), ground on
  optic: vec4<f32>,     // index of refraction, roughness, reflection gain, footage in reflections (0..1)
  absorb: vec4<f32>,    // absorption (1/m, rgb), murk extinction (1/m)
  murk: vec4<f32>,      // murk scattering colour (rgb), footage gain
  sky: vec4<f32>,       // sky / ambient light (rgb), checker size (px)
  sun: vec4<f32>,       // key light (rgb, colour x intensity), apparent sun radius (rad)
  sundir: vec4<f32>,    // toward the key light (world), shadow strength
  ripple: vec4<f32>,    // amount, frequency (1/m), speed of full ripples (m/s), _
  wet: vec4<f32>,       // wet darkening, wet gloss, crust plate size (m), bottomless (1: the bottom is out of sight)
  q: vec4<f32>,         // max steps, min step outside (surface cells), min step inside (surface cells), max interface events
  ww: vec4<f32>,        // whitewater: spray optical depth per density per cell, foam coverage per density, bubble optical depth per density per cell, on
  foam: vec4<f32>,      // foam colour (rgb), foam lace (0..1)
  org: vec4<f32>,       // grid corner (fire-local m), colliders drawn as grey stand-ins (1/0)
  lvl: vec4<f32>,       // water level (grid cells), open water on (1/0), caustics strength, blend width at the sides (cells)
  envp: vec4<f32>,      // HDRI on (1/0), rotation (rad), strength, foam cell size (m)
  glow: vec4<f32>,      // incandescence of a molten liquid (rgb radiance: blackbody colour x strength), crust cover (0..1)
  ocn: vec4<f32>,       // sea waves on (1/0), 1/swell tile, 1/chop tile (1/m), water level (m)
  ocn2: vec4<f32>,      // _, whitecaps, crest bound (m), rainbow
  ocx: array<vec4<f32>, 15>,    // the sea's layers, foam and looks (ocn_sample.wgsl); [14].x the liquid's top (cells, -1 unknown)
  dye: vec4<f32>,       // dye on (1/0), most samples along a path; the open water's current (m/s, grid x and z)
  rain: vec4<f32>,      // rain rings: impact cell size (m, 0 = no rain), ring life (s), ring speed (m/s), ring slope
  ice: vec4<f32>,       // ice (liq_ice_look.wgsl): on (1/0), frost amount, _, melting (0..1)
  ice2: vec4<f32>,      // frost colour (rgb), crystal size (m)
  lcnt: vec4<f32>,      // lamps in the set, the fire's lights on (1/0)
  lfire: vec4<f32>,     // the fire's lights: gain from their power to key-light units (rgb)
  lamp: array<vec4<f32>, 32>,   // 8 lamps: position (fire-local m), radius (m); power at 1 m (key-light
                                // units, rgb), kind (0 point, 1 spot, 2 area); aim, cos outer; cos inner
  lava: array<vec4<f32>, 8>,    // a molten liquid's look (liq_lava.wgsl; LiquidRenderer.lava_uniforms)
  lava_bb: array<vec4<f32>, 32>,  // blackbody: colour (rgb, luminance 1), log10 luminance relative to 1300 K; from
                                  // lava[4].x K in steps of 1 / lava[4].y
  hold: vec4<f32>,      // footage holdouts: matte on, depth pass on, depth kind (0 Z, 1 distance, 2 inverse Z), metres per unit
  hold2: vec4<f32>,     // footage fit scale (x, y)
  ccnt: vec4<f32>,      // collider count
  col: array<Collider, MAX_COLLIDERS>,
  calb: array<vec4<f32>, MAX_COLLIDERS>,   // a stand-in's own colour (rgb; w = 1: use it): things that fall or float, in their material
};

@group(0) @binding(0) var surf_t: texture_3d<f32>;
@group(0) @binding(1) var plate: texture_2d<f32>;
@group(0) @binding(2) var wet_t: texture_2d<f32>;
@group(0) @binding(3) var lin: sampler;
@group(0) @binding(4) var out_beauty: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var out_emit: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var out_aux: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var ww_t: texture_3d<f32>;
@group(0) @binding(8) var atlas: texture_3d<f32>;
@group(0) @binding(9) var caus_t: texture_2d<f32>;
@group(0) @binding(10) var env_t: texture_2d<f32>;
@group(0) @binding(11) var rep: sampler;
@group(0) @binding(12) var out_mask: texture_storage_2d<rgba16float, write>;  // mattes: water, spray, bubbles, (droplets)
@group(0) @binding(13) var oc_w: texture_3d<f32>;    // the sea's layers (ocn_sample.wgsl)
@group(0) @binding(14) var oc_f: texture_3d<f32>;    // the sea's foam
@group(0) @binding(15) var oc_l1: texture_2d<f32>;   // the open-water layer around the box (sea_layer.py)
@group(0) @binding(16) var oc_l2: texture_2d<f32>;   // the sea's height over the box (ocn_boxmap.wgsl)
@group(0) @binding(17) var heat_t: texture_3d<f32>;   // the liquid's heat at the surface (0 cooled .. 1 molten)
@group(0) @binding(18) var dye_t: texture_3d<f32>;    // dye per simulation cell: absorption (1/m, rgb), scattering (1/m)
@group(0) @binding(19) var<storage, read> fire_lights: array<vec4<f32>>;   // (position fire-local m, radius), (power, _)
@group(0) @binding(20) var<storage, read> fire_count: array<u32>;
@group(0) @binding(21) var hold_t: texture_2d<f32>;   // holdouts from the footage: x = matte, y = depth pass
// bindings 22, 23: surface hits for the passes after the march, and the lights a molten liquid's glow
// makes (liq_lava.wgsl)
@group(0) @binding(24) var cloth_t: texture_2d<f32>;   // fabric drawn before the march (cloth_layer.wgsl): colour,
                                                       // distance from where the ray starts (m; -1 none); 1x1 without
//!include ocn_sample.wgsl
//!include ocn_shade.wgsl
//!include liq_lava.wgsl
@group(1) @binding(0) var<uniform> U: Params;

const PI: f32 = 3.14159265;

var<private> g_shift: f32 = 0.0;  // motion-blur time offset of this sample (s)
var<private> g_under = false;     // the camera is under water
var<private> g_lamp = vec3<f32>(0.0);   // light of the lamps (and the fire) at the shading point, from all sides
var<private> g_near = 0.0;        // camera to where this pixel's ray starts (m)

// Scattering of the water (1/m). Seen from under water even clear water scatters a little (the fine
// particles all natural water holds), which is what gives it its colour into the distance.
fn murk_sc() -> f32 { return select(U.absorb.w, max(U.absorb.w, 0.04), g_under); }

fn ign(p: vec2<f32>) -> f32 {
  return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

// ---- a wave flume's mirror image ------------------------------------------------------------------
// With the sea let in only where its waves come from, the box's sides along the waves are frictionless
// walls (the sea's sides -1 in ocx[13]): mirror planes for the water. Past one the box's own water,
// reflected, carries on one box width (the break peeling on along the reef, the run-up along the beach,
// the flood over the land); cut off there, the box ended in a slab of glass where its breaking wave
// met the open sea's plain swell.
fn mir_lo() -> vec3<f32> {   // the corners of the box with its mirror images (cells)
  let s = U.ocx[13];
  return vec3<f32>(select(0.0, -U.n.x, abs(s.x + 1.0) < 0.5), 0.0, select(0.0, -U.n.z, abs(s.z + 1.0) < 0.5));
}
fn mir_hi() -> vec3<f32> {
  let s = U.ocx[13];
  return vec3<f32>(select(U.n.x, 2.0 * U.n.x, abs(s.y + 1.0) < 0.5), U.n.y, select(U.n.z, 2.0 * U.n.z, abs(s.w + 1.0) < 0.5));
}

// Grid point p folded back into the box across the mirrored sides.
fn mir(p: vec3<f32>) -> vec3<f32> {
  var q = p;
  let lo = mir_lo();
  let hi = mir_hi();
  if (q.x < 0.0 && lo.x < 0.0) { q.x = -q.x; }
  if (q.x > U.n.x && hi.x > U.n.x) { q.x = 2.0 * U.n.x - q.x; }
  if (q.z < 0.0 && lo.z < 0.0) { q.z = -q.z; }
  if (q.z > U.n.z && hi.z > U.n.z) { q.z = 2.0 * U.n.z - q.z; }
  return q;
}

// -1 along the axes p was folded across (motion there turns round in the mirror), else 1.
fn mir_sign(p: vec3<f32>) -> vec3<f32> {
  return select(vec3<f32>(1.0), vec3<f32>(-1.0), mir(p) != p);
}

// ---- the surface ------------------------------------------------------------------------------

// Raw sample at grid position p: distance in grid cells, velocity (m/s).
fn surf_raw(p: vec3<f32>) -> vec4<f32> {
  let s = textureSampleLevel(surf_t, lin, mir(p) / U.n.xyz, 0.0);
  return vec4<f32>(s.x / U.nf.w, s.yzw * mir_sign(p));
}

// Sample at p at this sample's moment in the shutter (the surface moved with its velocity).
fn surf(p: vec3<f32>) -> vec4<f32> {
  if (U.frame.z < 0.5) { return surf_raw(p); }
  let s0 = surf_raw(p);
  return surf_raw(p - s0.yzw * (g_shift / U.n.w));
}

fn open_water() -> bool { return U.lvl.y > 0.5; }

// Signed distance to the domain box (negative inside): the liquid is cut flat where it meets the
// box, as against the glass of a tank, instead of ending in the ragged edge of the surface grid.
// With open water the sides are not walls: the liquid carries on past them.
fn box_d(p: vec3<f32>) -> f32 {
  let d = max(-p, p - U.n.xyz);
  if (open_water()) { return d.y; }
  return max(d.x, max(d.y, d.z));
}

// Distance in from the side walls (negative outside the box; past a mirrored side, from the far edge of
// its mirror image).
fn side_in(p: vec3<f32>) -> f32 {
  let lo = mir_lo();
  let hi = mir_hi();
  return min(min(p.x - lo.x, hi.x - p.x), min(p.z - lo.z, hi.z - p.z));
}

// Fire-local metres (x, z) of grid point p, and how much of the short waves to keep there (they
// fade with distance from the camera, where they would only alias).
fn sea_xz(p: vec3<f32>) -> vec2<f32> { return U.org.xz + p.xz * U.n.w; }
fn sea_fade(p: vec3<f32>) -> f32 { return sea_footprint(p); }   // a pixel's footprint on the water (m)

// The open water's surface height at p (grid cells): the level, with the sea's waves.
fn lvl_at(p: vec3<f32>) -> f32 {
  if (!ocean_on()) { return U.lvl.x; }
  return U.lvl.x + ocean_eta(sea_xz(p), sea_fade(p)) / U.n.w;
}

// The same over the box's footprint, from the map of it made each frame: one texture read (the
// blend below runs at every step of every ray in the box).
fn lvl_box(p: vec3<f32>) -> f32 {
  if (!ocean_on()) { return U.lvl.x; }
  return U.lvl.x + textureSampleLevel(oc_l2, lin, mir(p).xz / U.n.xz, 0.0).x / U.n.w;
}

// How far the sea bed stands out of the water here (a beach, an island in the box, the swash zone
// on a beach): 1 where there is no open water to blend to, 0 on the sea.
fn box_dry(p: vec3<f32>) -> f32 {
  if (!ocean_on()) { return 0.0; }
  return textureSampleLevel(oc_l2, lin, mir(p).xz / U.n.xz, 0.0).y;
}

// The simulated surface, blended into the open water toward (and past) the open sides.
fn with_level(p: vec3<f32>, s: f32) -> f32 {
  if (!open_water()) { return s; }
  // (the outermost cells of the box sit a little low, topped up a cell short of the level: leave them out)
  // (no open water over dry land: a beach in the box)
  let w = max(smoothstep(1.5, U.lvl.w + 1.5, side_in(p)), box_dry(p));
  if (w >= 1.0) { return s; }
  return mix(p.y - lvl_box(p), s, w);
}

fn phi(p: vec3<f32>) -> f32 { return with_level(p, max(surf(p).x, box_d(p))); }

// Whitewater densities at p: x spray, y foam, z bubbles.
fn ww_at(p_in: vec3<f32>) -> vec3<f32> {
  let p = mir(p_in);
  if (any(p < vec3<f32>(0.0)) || any(p > U.n.xyz)) { return vec3<f32>(0.0); }   // none past the box
  return textureSampleLevel(ww_t, lin, p / U.n.xyz, 0.0).xyz;
}

// Light a white scattering medium (spray, foam, bubbles) receives: sky, plus the key light.
fn ww_light(nrm: vec3<f32>, lg: vec3<f32>) -> vec3<f32> {
  return U.sky.rgb * 0.95 + U.sun.rgb * (0.25 + 0.6 * max(dot(nrm, lg), 0.0)) + g_lamp * 0.6;
}

// ---- lamps ---------------------------------------------------------------------------------------

struct LampHit {
  l: vec3<f32>,    // toward the lamp
  e: vec3<f32>,    // its light here (key-light units)
  size: f32,       // its apparent radius (rad)
};

// Lamp i of the set, seen from x (fire-local m).
fn lamp_at(i: i32, x: vec3<f32>) -> LampHit {
  let a = U.lamp[4 * i];
  let b = U.lamp[4 * i + 1];
  let c = U.lamp[4 * i + 2];
  let d = U.lamp[4 * i + 3];
  let dv = a.xyz - x;
  let r2 = dot(dv, dv);
  let l = dv * inverseSqrt(max(r2, 1e-8));
  var e = b.rgb / (r2 + a.w * a.w);
  let kind = i32(b.w + 0.5);
  if (kind == 1) { e *= smoothstep(c.w, d.x, dot(-l, c.xyz)); }
  else if (kind == 2) { e *= max(dot(-l, c.xyz), 0.0); }
  return LampHit(l, e, a.w * inverseSqrt(max(r2, 1e-8)));
}

// Fire light i, seen from x (fire-local m).
fn fire_light_at(i: u32, x: vec3<f32>) -> LampHit {
  let a = fire_lights[2u * i];
  let dv = a.xyz - x;
  let r2 = dot(dv, dv);
  let l = dv * inverseSqrt(max(r2, 1e-8));
  return LampHit(l, fire_lights[2u * i + 1u].rgb * U.lfire.rgb / (r2 + a.w * a.w), a.w * inverseSqrt(max(r2, 1e-8)));
}

fn fire_lights_n() -> u32 {
  if (U.lcnt.y < 0.5) { return 0u; }
  return min(fire_count[0], min(64u, arrayLength(&fire_lights) / 2u));
}

// All the lamps' light at x from every side (for foam, spray and murk).
fn lamps_ambient(x: vec3<f32>) -> vec3<f32> {
  var e = vec3<f32>(0.0);
  for (var i = 0; i < i32(U.lcnt.x); i++) { e += lamp_at(i, x).e; }
  let nf = fire_lights_n();
  for (var i = 0u; i < nf; i++) { e += fire_light_at(i, x).e; }
  return e;
}

// The lamps' glints on a surface at x with normal nrm, seen from direction v (toward the eye).
fn glint_sized(nrm: vec3<f32>, v: vec3<f32>, l: vec3<f32>, size: f32) -> f32 {
  let nl = dot(nrm, l);
  if (nl <= 0.0) { return 0.0; }
  let nv = max(dot(nrm, v), 1e-4);
  let hv = normalize(l + v);
  let nh = max(dot(nrm, hv), 0.0);
  let r = max(U.optic.y, 0.02);
  let a2 = r * r * r * r + size * size;
  let dd = nh * nh * (a2 - 1.0) + 1.0;
  let D = a2 / (PI * dd * dd);
  let vis = 0.5 / (nl * sqrt(nv * nv * (1.0 - a2) + a2) + nv * sqrt(nl * nl * (1.0 - a2) + a2));
  return D * vis * fresnel(dot(v, hv), 1.0, U.optic.x) * nl;
}

fn lamps_glint(x: vec3<f32>, nrm: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
  var s = vec3<f32>(0.0);
  for (var i = 0; i < i32(U.lcnt.x); i++) {
    let h = lamp_at(i, x);
    s += h.e * glint_sized(nrm, v, h.l, min(h.size, 0.5));
  }
  let nf = fire_lights_n();
  for (var i = 0u; i < nf; i++) {
    let h = fire_light_at(i, x);
    s += h.e * glint_sized(nrm, v, h.l, min(h.size, 0.5));
  }
  return s * PI;
}

fn lamps_on() -> bool { return U.lcnt.x > 0.5 || U.lcnt.y > 0.5; }

// Cubic B-spline filtered distance (eight trilinear taps): a smooth surface for normals.
fn phi_cubic(p_in: vec3<f32>) -> f32 {
  var p = mir(p_in);
  if (U.frame.z > 0.5) { p = p - surf_raw(p).yzw * (g_shift / U.n.w); }
  let coord = p * (U.nf.xyz / U.n.xyz) - vec3<f32>(0.5);
  let idx = floor(coord);
  let f = coord - idx;
  let f2 = f * f;
  let f3 = f2 * f;
  let w0 = (vec3<f32>(1.0) - 3.0 * f + 3.0 * f2 - f3) / 6.0;
  let w1 = (vec3<f32>(4.0) - 6.0 * f2 + 3.0 * f3) / 6.0;
  let w2 = (vec3<f32>(1.0) + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
  let w3 = f3 / 6.0;
  let g0 = w0 + w1;
  let g1 = w2 + w3;
  let h0 = (idx - vec3<f32>(0.5) + w1 / g0) / U.nf.xyz;
  let h1 = (idx + vec3<f32>(1.5) + w3 / g1) / U.nf.xyz;
  var s = 0.0;
  for (var j = 0; j < 8; j++) {
    let o = vec3<i32>(j & 1, (j >> 1) & 1, j >> 2);
    let uvw = select(h0, h1, o == vec3<i32>(1));
    let g = select(g0, g1, o == vec3<i32>(1));
    s += g.x * g.y * g.z * textureSampleLevel(surf_t, lin, uvw, 0.0).x;
  }
  return with_level(p_in, max(s / U.nf.w, box_d(p_in)));
}

fn normal_at(p: vec3<f32>) -> vec3<f32> {
  let e = 0.6 / U.nf.w;
  let g = vec3<f32>(phi_cubic(p + vec3<f32>(e, 0.0, 0.0)) - phi_cubic(p - vec3<f32>(e, 0.0, 0.0)),
                    phi_cubic(p + vec3<f32>(0.0, e, 0.0)) - phi_cubic(p - vec3<f32>(0.0, e, 0.0)),
                    phi_cubic(p + vec3<f32>(0.0, 0.0, e)) - phi_cubic(p - vec3<f32>(0.0, 0.0, e)));
  let l = length(g);
  if (l < 1e-8) { return vec3<f32>(0.0, 1.0, 0.0); }
  return g / l;
}

// Small ripples the grid cannot hold, stronger where the liquid moves fast. With a current they
// ride along with it, on the open water too.
fn ripple(nrm: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
  if (U.ripple.x <= 0.0) { return nrm; }
  let cur = vec3<f32>(U.dye.z, 0.0, U.dye.w);
  let sp = max(length(surf(p).yzw), length(cur));
  let amt = U.ripple.x * clamp(sp / max(U.ripple.z, 1e-3), 0.0, 1.0);
  if (amt <= 0.0) { return nrm; }
  let t = U.frame.w;
  let q = (p * U.n.w - cur * t) * U.ripple.y + vec3<f32>(0.37 * t, -0.9 * t, 0.21 * t);
  let g = gnoise_d(q).yzw + 0.5 * gnoise_d(q * 2.13 + vec3<f32>(13.1, 7.7, 3.3)).yzw;
  let tg = g - nrm * dot(nrm, g);
  return normalize(nrm - 0.35 * amt * tg);
}

// Rain rings: every raindrop that lands on the water leaves a ring that spreads and fades. The
// impacts are scattered over two offset lattices of cells, one per cell per ring life at a random
// place and time (the cell size is set so there are as many per square metre and second as the
// rain brings down). Returns the slope of the rings (d/dx, d/dz) at xz (fire-local m).
fn rain_hash(c: vec2<i32>, k: i32, l: u32) -> vec3<f32> {
  var s = u32(c.x) * 73856093u ^ u32(c.y) * 19349663u ^ u32(k) * 83492791u ^ l * 2654435761u;
  var o = vec3<f32>(0.0);
  for (var i = 0; i < 3; i++) {
    s = s * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    o[i] = f32((w >> 22u) ^ w) * (1.0 / 4294967296.0);
  }
  return o;
}

fn rain_slope(xz: vec2<f32>) -> vec2<f32> {
  let S = U.rain.x;
  if (S <= 0.0) { return vec2<f32>(0.0); }
  let life = U.rain.y;
  let t = U.frame.w;
  var slope = vec2<f32>(0.0);
  for (var l = 0u; l < 2u; l++) {
    let off = vec2<f32>(0.5 * S * f32(l), 0.37 * S * f32(l));
    let base = vec2<i32>(floor((xz - off) / S));
    for (var j = -1; j <= 1; j++) {
      for (var i = -1; i <= 1; i++) {
        let c = base + vec2<i32>(i, j);
        let ph = rain_hash(c, 0, l).z * life;
        let k = i32(floor((t + ph) / life));
        let age = t + ph - f32(k) * life;
        let h = rain_hash(c, k, l);
        let at = (vec2<f32>(c) + vec2<f32>(0.1) + 0.8 * h.xy) * S + off;
        let d = xz - at;
        let r = length(d);
        let w = 0.004 + 0.012 * age;
        let x = (r - U.rain.z * age) / w;
        if (abs(x) > 3.0 || r < 1e-5) { continue; }
        // drops differ in size: most leave faint rings, a few strong ones
        let env = (1.0 - age / life) * (1.0 - age / life) * (0.25 + 0.75 * h.z * h.z);
        let dh = env * (3.14159 * cos(3.14159 * x) - 2.0 * x * sin(3.14159 * x)) * exp(-x * x);
        slope += dh * (d / r) * (0.004 / w);
      }
    }
  }
  return slope * U.rain.w;
}

fn rain_rings(nrm: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
  if (U.rain.x <= 0.0 || nrm.y <= 0.2) { return nrm; }
  let s = rain_slope(U.org.xz + p.xz * U.n.w) * nrm.y;
  return normalize(nrm - vec3<f32>(s.x, 0.0, s.y));
}

fn box_range(ro: vec3<f32>, rd: vec3<f32>) -> vec2<f32> {
  let safe = select(rd, vec3<f32>(1e-8), abs(rd) < vec3<f32>(1e-8));
  let inv = 1.0 / safe;
  let ta = (mir_lo() - ro) * inv;   // (with the mirror images past a flume's walls)
  let tb = (mir_hi() - ro) * inv;
  let t0 = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), max(min(ta.z, tb.z), 0.0));
  let t1 = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z));
  return vec2<f32>(t0, t1);
}

// ---- colliders ----------------------------------------------------------------------------------

fn solid_d(p: vec3<f32>) -> f32 {
  let cnt = i32(U.ccnt.x);
  if (cnt == 0) { return 1.0e6; }
  let w = U.org.xyz + p * U.n.w;
  // past a flume's walls the colliders in the box are mirrored too: the water there is the box's own,
  // reflected, flowing round their reflections (a reef, a town's houses)
  let q = mir(p);
  let mirrored = any(q != p);
  let wm = U.org.xyz + q * U.n.w;
  var d = 1.0e9;
  for (var i = 0; i < cnt; i++) {
    if (U.col[i].y.w < 0.5) { continue; }  // a helper, not in the shot: invisible
    d = min(d, col_sdf(U.col[i], w));
    if (mirrored) { d = min(d, col_sdf(U.col[i], wm)); }
  }
  return d / U.n.w;
}

fn solid_n(p: vec3<f32>) -> vec3<f32> {
  let e = 0.5;
  let g = vec3<f32>(solid_d(p + vec3<f32>(e, 0.0, 0.0)) - solid_d(p - vec3<f32>(e, 0.0, 0.0)),
                    solid_d(p + vec3<f32>(0.0, e, 0.0)) - solid_d(p - vec3<f32>(0.0, e, 0.0)),
                    solid_d(p + vec3<f32>(0.0, 0.0, e)) - solid_d(p - vec3<f32>(0.0, 0.0, e)));
  let l = length(g);
  if (l < 1e-8) { return vec3<f32>(0.0, 1.0, 0.0); }
  return g / l;
}

// First collider along the ray within tmax (grid cells), or -1.
fn trace_solid(ro: vec3<f32>, rd: vec3<f32>, tmax: f32) -> f32 {
  if (U.ccnt.x < 0.5) { return -1.0; }
  var t = 0.0;
  var d = 1.0e9;
  for (var i = 0; i < 256; i++) {
    d = solid_d(ro + rd * t);
    if (d < 0.02) { return t; }
    // (a ray grazing a big terrain creeps along it: the step grows with the distance travelled)
    t += max(d * 0.9, 0.05 + 0.002 * t);
    if (t >= tmax) { return -1.0; }
  }
  // out of steps while skimming a surface: it has reached it
  return select(-1.0, t, d < 0.5);
}

// ---- tracing -----------------------------------------------------------------------------------

// First surface along the ray in [t0, t1] (grid cells): x = distance (-1 for none), y = what was hit
// (1 liquid, 2 collider).
fn trace_out(ro: vec3<f32>, rd: vec3<f32>, t0_in: f32, t1_in: f32, max_steps: i32) -> vec2<f32> {
  let minstep = U.q.y / U.nf.w;
  // only the slab of the box the liquid reaches up to: above it there is only air, and over a big box a
  // ray creeping through it a fraction of a cell a step ran out of steps before it reached the water
  var t0 = t0_in;
  var t1 = t1_in;
  let top = U.ocx[14].x;
  if (top > 0.0) {
    if (rd.y < -1e-6) {
      if (ro.y > top) { t0 = max(t0, (top - ro.y) / rd.y); }
    } else if (rd.y > 1e-6) {
      t1 = min(t1, (top - ro.y) / rd.y);
    } else if (ro.y > top) {
      return vec2<f32>(-1.0, 0.0);
    }
    if (t0 >= t1) { return vec2<f32>(-1.0, 0.0); }
  }
  var t = t0;
  var d = phi(ro + rd * t);
  var s = solid_d(ro + rd * t);
  if (s < 0.02) { return vec2<f32>(t0, 2.0); }
  if (d < 0.0) { return vec2<f32>(t0, 1.0); }
  for (var i = 0; i < max_steps; i++) {
    let tp = t;
    t += max(min(d, s) * 0.85, minstep);
    if (t >= t1) { return vec2<f32>(-1.0, 0.0); }
    let p = ro + rd * t;
    s = solid_d(p);
    if (s < 0.02) { return vec2<f32>(t, 2.0); }
    d = phi(p);
    if (d < 0.0) {
      var a = tp;
      var b = t;
      for (var k = 0; k < 7; k++) {
        let m = 0.5 * (a + b);
        if (phi(ro + rd * m) < 0.0) { b = m; } else { a = m; }
      }
      return vec2<f32>(0.5 * (a + b), 1.0);
    }
  }
  return vec2<f32>(-1.0, 0.0);
}

// Exit from the liquid along the ray (grid cells). x = distance; y = 1 if a surface was found, 0 if
// the ray left the box (or ran out of steps) still inside, 2 if it hit a collider; z = bubble
// density integrated on the way (density x cells).
fn trace_in(ro: vec3<f32>, rd: vec3<f32>, t1: f32, max_steps: i32) -> vec3<f32> {
  let minstep = U.q.z / U.nf.w;
  var t = 0.0;
  var d = phi(ro);
  var bub = 0.0;
  let use_ww = U.ww.w > 0.5 && U.ww.z > 0.0;
  if (d > 0.0) { return vec3<f32>(0.0, 1.0, 0.0); }
  for (var i = 0; i < max_steps; i++) {
    let tp = t;
    let s = solid_d(ro + rd * t);
    t += max(min(-d, max(s, 0.0)) * 0.85, minstep);
    if (use_ww) { bub += ww_at(ro + rd * (0.5 * (tp + min(t, t1)))).z * (min(t, t1) - tp); }
    if (t >= t1) { return vec3<f32>(t1, 0.0, bub); }
    if (solid_d(ro + rd * t) < 0.02) { return vec3<f32>(t, 2.0, bub); }
    d = phi(ro + rd * t);
    if (d > 0.0) {
      var a = tp;
      var b = t;
      for (var k = 0; k < 7; k++) {
        let m = 0.5 * (a + b);
        if (phi(ro + rd * m) > 0.0) { b = m; } else { a = m; }
      }
      // (the liquid's surface keeps a thin gap over a floor or a sea bed: going down into it, the
      // ray has reached the bed, not the underside of the water)
      let q = ro + rd * (0.5 * (a + b));
      if (solid_d(q) < 1.5 && dot(rd, solid_n(q)) < 0.0) { return vec3<f32>(0.5 * (a + b), 2.0, bub); }
      // (the same over the ground: on to it, as a ray leaving the liquid downward)
      if (U.fwd.w > 0.5 && q.y < 1.5 && rd.y < 0.0) { return vec3<f32>(0.5 * (a + b), 0.0, bub); }
      return vec3<f32>(0.5 * (a + b), 1.0, bub);
    }
  }
  return vec3<f32>(t, 0.0, bub);
}

// Spray along the camera ray between t0 and t1 (grid cells): (in-scattered light, transmittance).
// Sunlight a cloud of water drops sends back toward the eye along view direction v (unit, world),
// relative to the key light: the primary bow about 42 degrees from the point opposite the sun, red
// outside and violet inside, the fainter secondary near 51 degrees with its colours reversed, and
// the brighter sky inside the primary (light leaving the drops at every angle under the bow's).
fn rainbow(v: vec3<f32>, to_sun: vec3<f32>) -> vec3<f32> {
  let a = degrees(acos(clamp(dot(v, -to_sun), -1.0, 1.0)));
  let p1 = vec3<f32>(42.2, 41.3, 40.6);
  let p2 = vec3<f32>(50.4, 51.4, 53.0);
  let d1 = (vec3<f32>(a) - p1) / 0.75;
  let d2 = (vec3<f32>(a) - p2) / 0.9;
  let bow = exp(-0.5 * d1 * d1) + 0.35 * exp(-0.5 * d2 * d2);
  let inner = 0.12 * smoothstep(40.5, 30.0, a) * smoothstep(0.0, 8.0, a);
  return bow + vec3<f32>(inner);
}

fn spray_along(ro: vec3<f32>, rd: vec3<f32>, t0: f32, t1: f32, lg: vec3<f32>, seed: f32) -> vec4<f32> {
  if (U.ww.w < 0.5 || U.ww.x <= 0.0 || t1 <= t0) { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
  let st = max(0.75, (t1 - t0) / 160.0);
  var t = t0 + st * seed;
  var tr = 1.0;
  var col = vec3<f32>(0.0);
  var light = ww_light(vec3<f32>(0.0, 1.0, 0.0), lg);
  if (U.ocn2.w > 0.0) { light += U.sun.rgb * rainbow(to_world_dir(rd), U.sundir.xyz) * (0.6 * U.ocn2.w); }
  for (var i = 0; i < 160; i++) {
    if (t >= t1 || tr < 0.01) { break; }
    let s = ww_at(ro + rd * t).x;
    if (s > 1e-4) {
      let a = 1.0 - exp(-s * U.ww.x * st);
      col += tr * a * light;
      tr *= 1.0 - a;
    }
    t += st;
  }
  return vec4<f32>(col, tr);
}

// ---- optics -------------------------------------------------------------------------------------

fn fresnel(cosi: f32, eta_i: f32, eta_t: f32) -> f32 {
  let ci = clamp(cosi, 0.0, 1.0);
  let st = eta_i / eta_t * sqrt(max(0.0, 1.0 - ci * ci));
  if (st >= 1.0) { return 1.0; }
  let ct = sqrt(max(0.0, 1.0 - st * st));
  let rs = (eta_i * ci - eta_t * ct) / (eta_i * ci + eta_t * ct);
  let rp = (eta_t * ci - eta_i * ct) / (eta_t * ci + eta_i * ct);
  return 0.5 * (rs * rs + rp * rp);
}

// Key-light glint: GGX microfacet reflection of a sun disc (its size widens the lobe).
fn glint(nrm: vec3<f32>, v: vec3<f32>, l: vec3<f32>) -> f32 {
  let nl = dot(nrm, l);
  if (nl <= 0.0) { return 0.0; }
  let nv = max(dot(nrm, v), 1e-4);
  let hv = normalize(l + v);
  let nh = max(dot(nrm, hv), 0.0);
  let r = max(U.optic.y, 0.02);
  let a2 = r * r * r * r + U.sun.w * U.sun.w;
  let dd = nh * nh * (a2 - 1.0) + 1.0;
  let D = a2 / (PI * dd * dd);
  let vis = 0.5 / (nl * sqrt(nv * nv * (1.0 - a2) + a2) + nv * sqrt(nl * nl * (1.0 - a2) + a2));
  return D * vis * fresnel(dot(v, hv), 1.0, U.optic.x) * nl;
}

// Cellular (Worley) noise: distances to the nearest and second-nearest feature point.
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

// Foam cover at surface point p (grid cells) for foam density fd: thick foam is solid white, thin
// foam breaks into a lace of bubble walls stretched along the flow.
fn foam_cover(p: vec3<f32>, fd: f32, vel: vec3<f32>) -> f32 {
  let cov = 1.0 - exp(-fd * U.ww.y);
  if (cov <= 0.001 || U.foam.w <= 0.0) { return cov; }
  var q = (U.org.xyz + p * U.n.w) / max(U.envp.w, 1e-4);
  let sp = length(vel);
  if (sp > 0.05) {
    let a = vel / sp;
    q -= a * dot(q, a) * 0.55;  // cells stretched along the flow
  }
  q += vec3<f32>(0.0, U.frame.w * 0.15, 0.0);
  let w = worley(q);
  let wall = 1.0 - smoothstep(0.03, 0.22, w.y - w.x);   // 1 on the bubble walls
  let lace = mix(1.0, wall * (0.75 + 0.5 * w.x), U.foam.w * (1.0 - smoothstep(0.55, 0.95, cov)));
  return clamp(cov * lace * 1.15, 0.0, 1.0);
}

// ---- what the rays see -------------------------------------------------------------------------

fn to_world_dir(d: vec3<f32>) -> vec3<f32> { return normalize((U.g2w * vec4<f32>(d, 0.0)).xyz); }
fn to_world(p: vec3<f32>) -> vec3<f32> { return (U.g2w * vec4<f32>(p, 1.0)).xyz; }

// Screen position of a world point: uv and how far inside the frame it is (0 outside, 1 inside).
fn screen_of(w: vec3<f32>) -> vec3<f32> {
  let c = U.vp * vec4<f32>(w, 1.0);
  if (c.w <= 1e-4) { return vec3<f32>(0.5, 0.5, 0.0); }
  let ndc = c.xy / c.w;
  let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let e = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
  return vec3<f32>(uv, smoothstep(-0.03, 0.02, e));
}

// The footage at a screen position (or the background shown when there is none).
fn backdrop(uv: vec2<f32>) -> vec3<f32> {
  if (U.plate.x > 0.5) {
    let q = (uv - vec2<f32>(0.5)) * U.plate.zw + vec2<f32>(0.5);
    let puv = clamp(vec2<f32>(1.0) - abs(vec2<f32>(1.0) - abs(q)), vec2<f32>(0.0), vec2<f32>(1.0));
    return input_transform(textureSampleLevel(plate, lin, puv, 0.0).rgb, i32(U.plate.y)) * U.murk.w;
  }
  if (U.back.w > 0.5) {
    let px = uv * U.res.xy;
    let cs = max(U.sky.w, 4.0);
    let chk = (i32(floor(px.x / cs)) + i32(floor(px.y / cs))) & 1;
    return select(vec3<f32>(0.18), vec3<f32>(0.24), chk == 1);
  }
  return U.back.rgb;
}

// The HDRI (latitude-longitude) in world direction d.
fn hdri(d: vec3<f32>) -> vec3<f32> {
  let c = cos(U.envp.y);
  let s = sin(U.envp.y);
  let r = vec3<f32>(c * d.x - s * d.z, d.y, s * d.x + c * d.z);
  let u = atan2(r.x, -r.z) * (0.5 / PI) + 0.5;
  let v = acos(clamp(r.y, -1.0, 1.0)) / PI;
  return textureSampleLevel(env_t, rep, vec2<f32>(u, v), 0.0).rgb * U.envp.z;
}

fn sky(d: vec3<f32>) -> vec3<f32> {
  if (U.envp.x > 0.5) { return hdri(d); }
  let s = U.sky.rgb;
  let up = d.y;
  let above = mix(s * 1.35, s * 0.9, clamp(up, 0.0, 1.0));
  let below = s * 0.3;
  return mix(below, above, smoothstep(-0.06, 0.04, up));
}

// Environment seen in a reflection, world direction d.
fn env(d: vec3<f32>) -> vec3<f32> {
  var col = sky(d);
  let a = U.optic.w;
  if (a > 0.0) {
    let f = U.fwd.xyz;
    var dd = d;
    var gain = 1.0;
    let k = dot(dd, f);
    if (k < 0.15) {
      if (U.envp.x > 0.5) { return col; }  // behind the camera the HDRI knows better than a mirror
      dd = normalize(dd + 2.0 * (0.15 - k) * f);
      gain = 0.8;
    }
    let s = screen_of(U.scene.xyz + dd * 1000.0);
    col = mix(col, backdrop(s.xy) * gain, a * s.z);
  }
  return col;
}

fn wet_at(pg: vec3<f32>) -> f32 {
  let n = vec2<i32>(i32(U.n.x), i32(U.n.z));
  let q = pg.xz - vec2<f32>(0.5);
  let i0 = vec2<i32>(floor(q));
  let f = q - floor(q);
  var s = 0.0;
  for (var j = 0; j < 4; j++) {
    let o = vec2<i32>(j & 1, j >> 1);
    let c = i0 + o;
    if (any(c < vec2<i32>(0)) || any(c >= n)) { continue; }
    let w = mix(1.0 - f, f, vec2<f32>(o));
    s += w.x * w.y * textureLoad(wet_t, c, 0).x;
  }
  // fade out toward the box edges: the simulation knows nothing past them, and a film that ran out
  // of the box would otherwise end in a straight line there. A wide, wobbly fade reads as the ragged
  // edge of a wet patch.
  let edge = min(min(pg.x, U.n.x - pg.x), min(pg.z, U.n.z - pg.z));
  let fw = max(6.0, 0.25 * min(U.n.x, U.n.z));
  let wob = gnoise_d(vec3<f32>(pg.x, 3.7, pg.z) * (3.0 / fw)).x;
  return s * smoothstep(0.0, fw, edge + wob * 0.35 * fw);
}

// Direct sunlight at a ground point relative to the sunlight with no liquid in the way: caustics
// where the surface focuses it, shadow where the liquid absorbs or scatters it. 1 outside the map.
fn caustic_at(pg: vec3<f32>, lg: vec3<f32>) -> f32 {
  let uv = mir(pg).xz / U.n.xz;
  if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
    // past the map: under the open water, sunlight that crossed its flat surface
    if (!open_water() || pg.y >= U.lvl.x) { return 1.0; }
    let tr = refract(-lg, vec3<f32>(0.0, 1.0, 0.0), 1.0 / U.optic.x);
    let L = (U.lvl.x - pg.y) / max(-tr.y, 1e-3) * U.n.w;
    let ext = dot(U.absorb.rgb, vec3<f32>(0.33333)) + U.absorb.w;
    return (1.0 - fresnel(lg.y, 1.0, U.optic.x)) * exp(-ext * L);
  }
  return textureSampleLevel(caus_t, lin, uv, 0.0).x;
}

// Multiplier on the footage at a ground point: darker where wet; caustics and shadow in the part of
// its light that comes straight from the key light.
fn ground_shade(pg: vec3<f32>, lg: vec3<f32>) -> f32 {
  // wet ground darkens where it is only wet, not under the liquid (seen through it, it is just the ground)
  var m = 1.0;
  if (phi(pg + vec3<f32>(0.0, 0.6 / U.nf.w, 0.0)) > 0.0) { m = 1.0 - U.wet.x * wet_at(pg); }
  if (U.lvl.z > 0.0 && lg.y > 0.02) {
    let es = luma(U.sun.rgb) * lg.y;
    let frac = es / max(es + luma(U.sky.rgb), 1e-4);
    let c = caustic_at(pg, lg);
    let k = select(U.lvl.z, U.sundir.w, c < 1.0);
    m *= 1.0 + (c - 1.0) * k * frac;
  }
  return m;
}

// The collider nearest grid point p (in the shot), or -1.
fn solid_id(p: vec3<f32>) -> i32 {
  let w = U.org.xyz + p * U.n.w;
  let wm = U.org.xyz + mir(p) * U.n.w;
  var best = 1.0e9;
  var id = -1;
  for (var i = 0; i < i32(U.ccnt.x); i++) {
    if (U.col[i].y.w < 0.5) { continue; }
    let d = min(col_sdf(U.col[i], w), col_sdf(U.col[i], wm));
    if (d < best) { best = d; id = i; }
  }
  return id;
}

// What a collider shows at grid point p seen along d: its own pixels in the footage, or a plain grey
// stand-in (in its material's colour, for one that falls or floats).
fn stand_in(p: vec3<f32>, d: vec3<f32>, lg: vec3<f32>) -> vec3<f32> {
  if (U.org.w > 0.5 || U.plate.x < 0.5) {
    let nn = solid_n(p);
    let nw = to_world_dir(nn);
    var albedo = select(vec3<f32>(0.3), U.ocx[12].rgb, U.ocx[12].w > 0.5);   // the stand-in colour (sand for a sea bed)
    let id = solid_id(p);
    if (id >= 0 && U.calb[id].w > 0.5) { albedo = U.calb[id].rgb; }
    var e = sky(nw) * (0.55 + 0.45 * nw.y) + U.sun.rgb * max(dot(nn, lg), 0.0);
    if (lava_on()) { e += lava_light_at(U.org.xyz + p * U.n.w, nn); }   // a molten liquid's glow (liq_lava.wgsl)
    return albedo * e;
  }
  let s = screen_of(to_world(p));
  return mix(sky(to_world_dir(d)) * 0.3, backdrop(s.xy), s.z);
}

// Where a ray leaving the liquid ends up: a collider, the ground (footage at that point, shaded) or
// the backdrop at the backdrop distance.
fn background(p: vec3<f32>, d: vec3<f32>, lg: vec3<f32>) -> vec3<f32> {
  var tg = 1.0e9;
  if (U.fwd.w > 0.5 && d.y < -1e-4) { tg = -p.y / d.y; }
  let ts = trace_solid(p, d, min(tg, U.scene.w / U.n.w));
  if (ts >= 0.0) { return stand_in(p + d * ts, d, lg); }
  if (tg < 1.0e8) {
    let pg = p + d * tg;
    let wg = to_world(pg);
    let s = screen_of(wg);
    let dw = to_world_dir(d);
    var seen = backdrop(s.xy);
    if ((U.vp * vec4<f32>(wg, 1.0)).w <= 1e-4) { seen = sky(dw) * 0.35; }
    return seen * ground_shade(pg, lg);
  }
  let dw = to_world_dir(d);
  let far = to_world(p) + dw * U.scene.w;
  let s = screen_of(far);
  return mix(env(dw), backdrop(s.xy), s.z);
}

// Height of q over the sea (cells), and far above it inside the box's footprint.
fn sheet_f(q: vec3<f32>) -> f32 {
  if (side_in(q) > 1.0) { return 1.0e4; }
  return q.y - lvl_at(q);
}

// Where the ray leaves the part of the box's footprint that its own liquid covers (grid cells along it).
fn footprint_exit(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
  let lo = 1.0;
  let b0 = mir_lo();
  let b1 = mir_hi();
  var te = 1.0e7;
  for (var a = 0; a < 3; a += 2) {
    if (rd[a] > 1e-6) { te = min(te, (b1[a] - lo - ro[a]) / rd[a]); }
    if (rd[a] < -1e-6) { te = min(te, (b0[a] + lo - ro[a]) / rd[a]); }
  }
  return te + 0.01;
}

// Open water beyond the box: the sheet at the water level, or the sea. Distance along the ray to it,
// if the ray meets it outside the box footprint (from above), else -1.
fn open_sheet(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
  if (!open_water()) { return -1.0; }
  if (!ocean_on()) {
    if (rd.y > -1e-5 || ro.y < U.lvl.x) { return -1.0; }
    let t = (U.lvl.x - ro.y) / rd.y;
    if (t <= 0.0) { return -1.0; }
    if (side_in(ro + rd * t) > 1.0) { return -1.0; }
    return t;
  }
  // the sea: march the ray down through the band the crests and troughs can reach
  let crest = U.ocn2.z / U.n.w;
  let top = U.lvl.x + crest;
  let bottom = U.lvl.x - crest;
  if (ro.y < lvl_at(ro)) { return -1.0; }
  if (ro.y > top && rd.y > -1e-5) { return -1.0; }
  var t = 0.0;
  if (ro.y > top) { t = (top - ro.y) / rd.y; }
  var t_end = 1.0e7;
  if (rd.y < -1e-5) { t_end = (bottom - ro.y) / rd.y; }
  t_end = min(t_end, t + 4.0 * U.scene.w / U.n.w);
  var f = sheet_f(ro + rd * t);
  var hit = false;
  var tp = t;
  for (var i = 0; i < 128; i++) {
    if (f < 0.0) { hit = true; break; }
    tp = t;
    if (side_in(ro + rd * t) > 1.0) {
      t = max(footprint_exit(ro, rd), t + 0.05);   // the box's own liquid is traced on its own
    } else {
      t += max(f * 0.5, 0.05 + 0.0015 * t);
    }
    if (t >= t_end) {
      t = t_end;
      f = sheet_f(ro + rd * t);
      hit = f < 0.0;
      break;
    }
    f = sheet_f(ro + rd * t);
  }
  if (!hit) {
    // far out, past where the march reaches: the level itself (the waves are too small to see there)
    if (rd.y > -1e-5) { return -1.0; }
    let tf = (U.lvl.x - ro.y) / rd.y;
    if (tf <= 0.0 || side_in(ro + rd * tf) > 1.0) { return -1.0; }
    return tf;
  }
  var a = tp;
  var b = t;
  for (var k = 0; k < 8; k++) {
    let m = 0.5 * (a + b);
    let q = ro + rd * m;
    if (q.y - lvl_at(q) < 0.0) { b = m; } else { a = m; }
  }
  let th = 0.5 * (a + b);
  if (side_in(ro + rd * th) > 1.0) { return -1.0; }
  return th;
}

// Light scattered into a ray over L metres of liquid (murk lit by the sky and the key light), and the
// transmittance over it. An endless path gives water's own colour, light x murk / extinction.
fn murk_path(L: f32) -> array<vec3<f32>, 2> {
  let ms = murk_sc();
  let sig = U.absorb.rgb + vec3<f32>(ms);
  let tr = exp(-sig * L);
  let light = U.murk.rgb * (U.sky.rgb + (U.sun.rgb + g_lamp) * 0.25) * (ms / max(sig, vec3<f32>(1e-5)));
  return array<vec3<f32>, 2>(light * (vec3<f32>(1.0) - tr), tr);
}

// Dye along a path inside the liquid from ro over t grid cells: its optical depth (absorption rgb,
// scattering), about a sample a cell.
fn dye_along(ro: vec3<f32>, rd: vec3<f32>, t: f32) -> vec4<f32> {
  if (U.dye.x < 0.5 || t <= 0.0) { return vec4<f32>(0.0); }
  let ns = i32(clamp(ceil(t), 1.0, U.dye.y));
  let dt = t / f32(ns);
  var s = vec4<f32>(0.0);
  for (var i = 0; i < ns; i++) {
    s += textureSampleLevel(dye_t, lin, mir(ro + rd * ((f32(i) + 0.5) * dt)) / U.n.xyz, 0.0);
  }
  return s * (dt * U.n.w);
}

// murk_path with a dye on the way (optical depth from dye_along): the dye absorbs, and scatters the
// same light the murk does, its colour what its absorption leaves.
fn murk_dye(L: f32, dye: vec4<f32>) -> array<vec3<f32>, 2> {
  let ms = murk_sc();
  let tau = (U.absorb.rgb + vec3<f32>(ms)) * L + dye.rgb + vec3<f32>(dye.w);
  let tr = exp(-tau);
  let sc = U.murk.rgb * (ms * L) + vec3<f32>(dye.w);
  let light = (U.sky.rgb + (U.sun.rgb + g_lamp) * 0.25) * sc / max(tau, vec3<f32>(1e-5));
  return array<vec3<f32>, 2>(light * (vec3<f32>(1.0) - tr), tr);
}

fn deep_colour() -> vec3<f32> {
  let ms = murk_sc();
  let sig = U.absorb.rgb + vec3<f32>(ms);
  return U.murk.rgb * (U.sky.rgb + U.sun.rgb * 0.25) * (ms / max(sig, vec3<f32>(1e-5)));
}

// A ray under the open water (outside the box, or leaving it under water): on to the ground, or up
// through the flat surface. Returns what it sees, attenuated by the water on the way.
fn under_open(p: vec3<f32>, d: vec3<f32>, lg: vec3<f32>) -> vec3<f32> {
  if (d.y < -1e-4) {
    if (U.fwd.w < 0.5 || U.wet.w > 0.5) {
      return deep_colour();   // bottomless: the water's own colour, all the way down
    }
    let mp = murk_path(max(-p.y / d.y, 0.0) * U.n.w);
    return background(p, d, lg) * mp[1] + mp[0];
  }
  // up through the surface (the flat level, or the sea's waves where the ray meets them)
  var L = max((U.lvl.x - p.y) / max(d.y, 1e-4), 0.0);
  L = max((lvl_at(p + d * L) - p.y) / max(d.y, 1e-4), 0.0);
  let mp = murk_path(L * U.n.w);
  var sn = vec3<f32>(0.0, 1.0, 0.0);
  if (ocean_on()) { sn = ocean_normal(sea_xz(p + d * L), sea_fade(p + d * L)); }
  let out_dir = refract(d, -sn, U.optic.x);
  var seen = deep_colour();   // totally reflected: the surface mirrors the water under it
  if (dot(out_dir, out_dir) > 1e-6) {
    seen = env(to_world_dir(out_dir)) * (1.0 - fresnel(dot(d, sn), U.optic.x, 1.0))
         + deep_colour() * fresnel(dot(d, sn), U.optic.x, 1.0);
  }
  return seen * mp[1] + mp[0];
}

// Radiance arriving along a reflected ray from p: ground, a collider, the environment or more liquid.
fn reflected(p: vec3<f32>, d: vec3<f32>, lg: vec3<f32>) -> vec3<f32> {
  let r = box_range(p, d);
  if (r.y > r.x) {
    let hit = trace_out(p, d, r.x, r.y, 64);
    if (hit.y > 1.5) { return stand_in(p + d * hit.x, d, lg); }
    if (hit.y > 0.5) {
      // the reflection sees more liquid: its surface reflects the sky and transmits a little
      let dw = to_world_dir(d);
      return env(dw) * 0.25 + background(p + d * hit.x, d, lg) * 0.35 * exp(-U.absorb.rgb * 0.3);
    }
  }
  return background(p, d, lg);
}

// ---- shading a liquid surface point -------------------------------------------------------------

struct Shade {
  col: vec3<f32>,
  spec: vec3<f32>,
  bub: f32,         // how much the bubbles inside hide (0..1)
};

fn shade_liquid(p: vec3<f32>, rd: vec3<f32>, nrm: vec3<f32>, lg: vec3<f32>, max_steps: i32) -> Shade {
  let h = U.n.w;
  let ior = U.optic.x;
  let eps = 0.35 / U.nf.w;
  let cosi = clamp(-dot(rd, nrm), 0.0, 1.0);
  let F = fresnel(cosi, 1.0, ior);

  // reflection and glint
  var rr = reflect(rd, nrm);
  if (open_water() && abs(p.y - lvl_box(p)) < 2.0 * max(U.ocn2.z / U.n.w, 1.0)) {
    // off the open water's surface a reflection pointing down would meet the next wave: the sky above it
    rr = normalize(vec3<f32>(rr.x, max(rr.y, 0.02), rr.z));
  }
  // where the surface is the sea's own it is shaded as the open water is (ocn_shade.wgsl)
  let share = sea_share(p);
  var col = F * U.optic.z * sea_reflected(p + nrm * eps, rr, lg, share);
  var spec = U.sun.rgb * (sea_glint_box(p, nrm, -rd, lg, share) * PI);
  if (lamps_on()) { spec += lamps_glint(U.org.xyz + p * U.n.w, nrm, -rd); }
  col += spec;

  // refraction through the body, out the far side, possibly through more drops
  var s = through(p - nrm * eps, refract(rd, nrm, 1.0 / ior), vec3<f32>(1.0 - F), lg, max_steps);
  s.col += col + (1.0 - F) * share * sea_crest_glow(p, nrm, rd, lg);
  s.spec = spec;
  return s;
}

// A ray inside the liquid at pos going dir, carrying thr of the light: on through the body, out
// through its surface (or totally reflected back in), through more drops, to what it finally sees.
fn through(pos_in: vec3<f32>, dir_in: vec3<f32>, thr_in: vec3<f32>, lg: vec3<f32>, max_steps: i32) -> Shade {
  let h = U.n.w;
  let ior = U.optic.x;
  let eps = 0.35 / U.nf.w;
  var col = vec3<f32>(0.0);
  var thr = thr_in;
  var bub = 0.0;
  var pos = pos_in;
  var dir = dir_in;
  var inside = true;
  var done = false;
  let events = i32(U.q.w);
  for (var ev = 0; ev < events; ev++) {
    if (inside) {
      let rb = box_range(pos, dir);
      let ex = trace_in(pos, dir, max(rb.y, 0.0), max_steps);
      let cl = cloth_along(pos, dir, ex.x);
      if (cl.w >= 0.0) {
        // fabric under the liquid: seen through the water before it, and lit through the water over it
        // (the light down to it dimmed about as much as the light back up)
        let mc = murk_dye(cl.w * h, dye_along(pos, dir, cl.w));
        col += thr * mc[0];
        thr *= mc[1];
        col += thr * cl.rgb * mc[1];
        done = true;
        break;
      }
      let L = ex.x * h;
      let mp = murk_dye(L, dye_along(pos, dir, ex.x));
      col += thr * mp[0];
      thr *= mp[1];
      if (ex.z > 0.0) {
        let tb = exp(-ex.z * U.ww.z);
        col += thr * (1.0 - tb) * ww_light(vec3<f32>(0.0, 1.0, 0.0), lg) * 0.8;
        thr *= tb;
        bub = 1.0 - (1.0 - bub) * tb;
      }
      pos = pos + dir * ex.x;
      if (ex.y > 1.5) {
        col += thr * stand_in(pos, dir, lg);
        done = true;
        break;
      }
      if (ex.y < 0.5) {
        // left the box still in the liquid: on through the open water, or out at the box wall
        // (under the sea's surface there, not only the flat level: a ray out of a crest is still in the sea)
        if (open_water() && pos.y < max(U.lvl.x, lvl_box(pos) + 0.5)) { col += thr * under_open(pos, dir, lg); }
        else { col += thr * background(pos, dir, lg); }
        done = true;
        break;
      }
      // leaving the liquid the normal faces along the ray (a drop a cell or two across can have its
      // estimated normal the wrong way round, which would count as grazing and lose all the light)
      var n2 = normal_at(pos);
      if (dot(dir, n2) < 0.0) { n2 = -n2; }
      let c2 = clamp(dot(dir, n2), 0.0, 1.0);
      let out_dir = refract(dir, -n2, ior);
      if (dot(out_dir, out_dir) < 1e-6) {
        dir = reflect(dir, -n2);
        pos -= n2 * eps;
        continue;
      }
      thr *= 1.0 - fresnel(c2, ior, 1.0);
      dir = out_dir;
      pos += n2 * eps;
      inside = false;
    } else {
      let rb = box_range(pos, dir);
      var hit = vec2<f32>(-1.0, 0.0);
      if (rb.y > rb.x) { hit = trace_out(pos, dir, rb.x, rb.y, max_steps / 2); }
      let reach = select(max(rb.y, 0.0) + 2.0 * max(U.n.x, max(U.n.y, U.n.z)), hit.x, hit.y > 0.5);
      let cl = cloth_along(pos, dir, reach);
      if (cl.w >= 0.0) {
        col += thr * cl.rgb;   // fabric past the liquid (behind a drop)
        done = true;
        break;
      }
      if (hit.y > 1.5) {
        col += thr * stand_in(pos + dir * hit.x, dir, lg);
        done = true;
        break;
      }
      if (hit.y < 0.5) {
        col += thr * background(pos, dir, lg);
        done = true;
        break;
      }
      pos = pos + dir * hit.x;
      var n3 = normal_at(pos);
      if (dot(dir, n3) > 0.0) { n3 = -n3; }   // entering, it faces the ray
      let F3 = fresnel(-dot(dir, n3), 1.0, ior);
      col += thr * F3 * env(to_world_dir(reflect(dir, n3)));
      thr *= 1.0 - F3;
      dir = refract(dir, n3, 1.0 / ior);
      pos -= n3 * eps;
      inside = true;
    }
  }
  if (!done) { col += thr * background(pos, dir, lg); }
  var s: Shade;
  s.col = col;
  s.spec = vec3<f32>(0.0);
  s.bub = bub;
  return s;
}

// Shafts of sunlight in the water along a ray from ro over t grid cells: the key light scattered
// toward the eye (strongly forward, as by the fine particles in natural water) from where it comes
// down through the surface, in the pattern the waves focus it into (the caustics map, followed up the
// light to each point), dimmed by the water on its way in and on to the eye.
fn shafts(ro: vec3<f32>, rd: vec3<f32>, t: f32, lg: vec3<f32>, seed: f32) -> vec3<f32> {
  if (lg.y <= 0.02 || t <= 0.0) { return vec3<f32>(0.0); }
  let h = U.n.w;
  let scat = murk_sc();
  let sig = U.absorb.rgb + vec3<f32>(scat);
  let ext = dot(U.absorb.rgb, vec3<f32>(0.33333)) + U.absorb.w;
  let tr = refract(-lg, vec3<f32>(0.0, 1.0, 0.0), 1.0 / U.optic.x);
  let g = 0.8;
  let ct = dot(rd, -tr);
  let phase = (1.0 - g * g) / (12.566 * pow(1.0 + g * g - 2.0 * g * ct, 1.5));
  let tm = min(t, 60.0 / h);
  let ns = 24;
  let dt = tm / f32(ns);
  var s = vec3<f32>(0.0);
  for (var i = 0; i < ns; i++) {
    let q = ro + rd * ((f32(i) + seed) * dt);
    let down = max(lvl_at(q) - q.y, 0.0) / max(-tr.y, 0.05);         // cells of water the light crossed
    if (lvl_at(q) - q.y <= 0.0 && phi(q) > 0.0) { continue; }          // in the air
    var focus = 1.0;
    if (U.fwd.w > 0.5 && q.y > 0.0) {
      let pg = q + tr * (q.y / max(-tr.y, 0.05));
      let lg_ground = (1.0 - fresnel(lg.y, 1.0, U.optic.x)) * exp(-ext * (down + q.y / max(-tr.y, 0.05)) * h);
      focus = clamp(caustic_at(pg, lg) / max(lg_ground, 1e-3), 0.0, 6.0);
    }
    s += exp(-sig * (down + (f32(i) + seed) * dt) * h) * focus;
  }
  return U.sun.rgb * U.murk.rgb * (scat * phase * dt * h) * s;
}

// What a camera under water sees along rd (grid cells, from ro under the surface).
fn underwater(ro: vec3<f32>, rd: vec3<f32>, lg: vec3<f32>, max_steps: i32, seed: f32) -> Shade {
  let h = U.n.w;
  var s: Shade;
  s.spec = vec3<f32>(0.0);
  s.bub = 0.0;
  var pos = ro;
  var col = vec3<f32>(0.0);
  var thr = vec3<f32>(1.0);
  let rb = box_range(ro, rd);
  let in_box = all(ro >= mir_lo()) && all(ro <= mir_hi());
  if (!in_box) {
    // under the open water past the box: through it to the box, or on without reaching it
    if (rb.y <= rb.x) {
      var far = 1.0e4;
      if (rd.y < -1e-4) { far = -ro.y / rd.y; }
      else if (rd.y > 1e-4) { far = (U.lvl.x - ro.y) / rd.y; }
      s.col = under_open(ro, rd, lg) + shafts(ro, rd, far, lg, seed);
      return s;
    }
    let mp = murk_path(rb.x * h);
    col = mp[0] + shafts(ro, rd, rb.x, lg, seed);
    thr = mp[1];
    pos = ro + rd * (rb.x + 1e-3);
    if (phi(pos) >= 0.0) {
      s.col = col + thr * background(pos, rd, lg);
      return s;
    }
  }
  // the shafts along the way to where the ray leaves the water (the first surface, or the box)
  let ex = trace_in(pos, rd, max(box_range(pos, rd).y, 0.0), max_steps);
  col += thr * shafts(pos, rd, ex.x, lg, seed);
  let r = through(pos, rd, thr, lg, max_steps);
  s.col = col + r.col;
  s.bub = r.bub;
  return s;
}

// How much of what lies dist_m (metres from the camera) along the pixel's ray (uv, world direction
// rd_w) the footage lets show: its matte, and in front of its own surfaces only.
fn hold_keep(uv: vec2<f32>, rd_w: vec3<f32>, dist_m: f32) -> f32 {
  if (U.hold.x < 0.5 && U.hold.y < 0.5) { return 1.0; }
  let huv = (uv - vec2<f32>(0.5)) * U.hold2.xy + vec2<f32>(0.5);
  let hv = textureSampleLevel(hold_t, lin, huv, 0.0);
  var keep = 1.0;
  if (U.hold.x > 0.5) { keep = 1.0 - clamp(hv.x, 0.0, 1.0); }
  if (U.hold.y > 0.5 && hv.y > 0.0 && hv.y < 1.0e20) {
    let kind = i32(U.hold.z + 0.5);
    var d = hv.y * U.hold.w;
    if (kind == 2) { d = U.hold.w / hv.y; }
    if (kind != 1) { d = d / max(dot(rd_w, U.fwd.xyz), 1e-3); }   // Z along the view axis -> along the ray
    keep *= smoothstep(-0.03, 0.03, d - dist_m);
  }
  return keep;
}

// ---- fabric (cloth.py) ---------------------------------------------------------------------------

fn cloth_on() -> bool { return textureDimensions(cloth_t).x > 1u; }

// The fabric at screen position uv: its colour and its distance from the camera (m), or w < 0.
fn cloth_px(uv: vec2<f32>) -> vec4<f32> {
  if (any(uv < vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0))) { return vec4<f32>(-1.0); }
  let c = textureLoad(cloth_t, vec2<i32>(uv * vec2<f32>(textureDimensions(cloth_t))), 0);
  if (c.w < 0.0) { return vec4<f32>(-1.0); }
  return vec4<f32>(c.rgb, c.w + g_near);
}

// How far behind the drawn fabric (m) the ray's point at t (cells) lies, where the screen shows fabric
// there (y = 1; 0 where it shows none).
fn cloth_rel(p: vec3<f32>, d: vec3<f32>, t: f32) -> vec2<f32> {
  let w = to_world(p + d * t);
  let sc = screen_of(w);
  if (sc.z <= 0.0) { return vec2<f32>(0.0); }
  let c = cloth_px(sc.xy);
  if (c.w < 0.0) { return vec2<f32>(0.0); }
  return vec2<f32>(length(w - U.scene.xyz) - c.w, 1.0);
}

fn cloth_hit(p: vec3<f32>, d: vec3<f32>, t: f32) -> vec4<f32> {
  let c = cloth_px(screen_of(to_world(p + d * t)).xy);
  return vec4<f32>(c.rgb, t);
}

// The fabric a ray at grid point p going d meets within tmax cells, found in the drawn cloth's depth (as
// screen-space reflections are): its colour and how far along the ray (cells), or w < 0. The ray meets it
// where it passes from in front of the drawn surface to just behind it (a few cells: the sheet's depth).
fn cloth_along(p: vec3<f32>, d: vec3<f32>, tmax: f32) -> vec4<f32> {
  if (!cloth_on() || tmax <= 0.0) { return vec4<f32>(-1.0); }
  let steps = 24;
  let thick = 3.0 * U.n.w;
  var t0 = 0.0;
  var r0 = cloth_rel(p, d, 0.0);
  for (var i = 1; i <= steps; i++) {
    let t1 = tmax * f32(i) / f32(steps);
    let r1 = cloth_rel(p, d, t1);
    if (r1.y > 0.5 && r1.x >= -0.5 * thick) {
      if (r1.x <= thick) { return cloth_hit(p, d, t1); }
      if (r0.y < 0.5 || r0.x < 0.0) {
        // it passed behind the sheet within this step: where (bisection), if the sheet is there
        var lo = t0;
        var hi = t1;
        for (var k = 0; k < 6; k++) {
          let m = 0.5 * (lo + hi);
          let rm = cloth_rel(p, d, m);
          if (rm.y > 0.5 && rm.x >= 0.0) { hi = m; } else { lo = m; }
        }
        let rh = cloth_rel(p, d, hi);
        if (rh.y > 0.5 && rh.x <= thick) { return cloth_hit(p, d, hi); }
      }
    }
    t0 = t1;
    r0 = r1;
  }
  return vec4<f32>(-1.0);
}

// ---- main ----------------------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let px = vec2<i32>(id.xy);
  if (f32(px.x) >= U.res.x || f32(px.y) >= U.res.y) { return; }
  let uv = (vec2<f32>(px) + vec2<f32>(0.5) + U.res.zw) / U.res.xy;
  let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let pn = U.inv_vp * vec4<f32>(ndc, 0.0, 1.0);
  let pf = U.inv_vp * vec4<f32>(ndc, 1.0, 1.0);
  let ro_w = pn.xyz / pn.w;
  let rd_w = normalize(pf.xyz / pf.w - ro_w);
  let ro = (U.w2g * vec4<f32>(ro_w, 1.0)).xyz;
  let rdg = (U.w2g * vec4<f32>(rd_w, 0.0)).xyz;
  let cells_per_m = length(rdg);
  let rd = rdg / cells_per_m;
  let lg = normalize((U.w2g * vec4<f32>(U.sundir.xyz, 0.0)).xyz);
  g_near = length(ro_w - U.scene.xyz);
  if (U.frame.z > 0.5) {
    g_shift = U.frame.y * (fract(ign(vec2<f32>(px) + vec2<f32>(17.0, 59.0)) + U.frame.x * 0.7548776662) - 0.5);
  }
  let max_steps = i32(U.q.x);
  let seed = fract(ign(vec2<f32>(px) + vec2<f32>(3.0, 7.0)) + U.frame.x * 0.381966);

  // the camera under water: in the box's liquid, or under the open water past it
  var under = false;
  if (all(ro > mir_lo()) && all(ro < mir_hi())) { under = phi(ro) < 0.0; }
  else if (open_water()) { under = ro.y < lvl_at(ro); }
  if (under) {
    g_under = true;
    if (lamps_on()) { g_lamp = lamps_ambient(U.org.xyz + ro * U.n.w); }
    let uw = underwater(ro, rd, lg, max_steps, seed);
    textureStore(out_beauty, px, vec4<f32>(uw.col, 1.0));
    textureStore(out_emit, px, vec4<f32>(0.0));
    textureStore(out_aux, px, vec4<f32>(1.0, 0.001, 0.0, 1.0));   // the water is right at the lens
    textureStore(out_mask, px, vec4<f32>(1.0, 0.0, uw.bub, 0.0));
    return;
  }

  let r = box_range(ro, rd);
  var hit = vec2<f32>(-1.0, 0.0);
  if (r.y > r.x) {
    let jit = fract(ign(vec2<f32>(px)) + U.frame.x * 0.61803398875) * U.q.y / U.nf.w;
    hit = trace_out(ro, rd, r.x + jit, r.y, max_steps);
  }
  // colliders outside the box still hold the liquid out
  var t_solid = -1.0;
  if (hit.y > 1.5) { t_solid = hit.x; }
  if (U.ccnt.x > 0.5) {
    // a solid may come before what the march found in the box: before the box (a sea bed or a bank
    // running out past it, the ray already inside it where it enters the box), or above the slab the
    // liquid reaches up to (trace_out marches only that: a house the ray passes on its way down)
    let ts = trace_solid(ro, rd, select(1.0e5, hit.x, hit.y > 0.5));
    if (ts >= 0.0 && (t_solid < 0.0 || ts < t_solid)) { t_solid = ts; }
  }
  // the open water past the box: first when the ray crosses it before it reaches any liquid in the
  // box (a ray that goes under the open water and on into the box stays with the open water)
  var t_open = -1.0;
  var t_sheet = open_sheet(ro, rd);
  // (the open water never lies inside a solid: a beach or bank standing above the level)
  if (t_sheet >= 0.0 && U.ccnt.x > 0.5 && solid_d(ro + rd * t_sheet) < 0.0) { t_sheet = -1.0; }
  if (t_sheet >= 0.0 && (hit.y < 0.5 || t_sheet < hit.x - 0.5)) {
    if (t_solid < 0.0 || t_sheet < t_solid) {
      t_open = t_sheet;
      t_solid = -1.0;
      hit = vec2<f32>(-1.0, 0.0);
    }
  }

  if (cloth_on()) {
    // fabric drawn before the march (cloth.py): in front of all the rest here it is what is seen
    let cl = textureLoad(cloth_t, px, 0);
    if (cl.w >= 0.0) {
      let t_cl = cl.w * cells_per_m;
      var first = 1.0e9;
      if (t_solid >= 0.0) { first = t_solid; }
      if (hit.y > 0.5) { first = min(first, hit.x); }
      if (t_open >= 0.0) { first = min(first, t_open); }
      if (t_cl < first) {
        var sp = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        if (r.y > r.x) { sp = spray_along(ro, rd, r.x, min(t_cl, r.y), lg, seed); }
        let keep = hold_keep(uv, rd_w, g_near + cl.w);
        textureStore(out_beauty, px, vec4<f32>(sp.rgb + cl.rgb * sp.w, 1.0) * keep);
        textureStore(out_emit, px, vec4<f32>(0.0));
        textureStore(out_aux, px, vec4<f32>(1.0, cl.w, 0.0, keep));
        textureStore(out_mask, px, vec4<f32>(0.0, (1.0 - sp.w) * keep, 0.0, 0.0));
        return;
      }
    }
  }

  if (t_solid >= 0.0) {
    // a collider in front: it is in the footage (hold out), or a grey stand-in
    let sp = spray_along(ro, rd, r.x, min(t_solid, r.y), lg, seed);
    let depth = t_solid / cells_per_m;
    if (U.org.w > 0.5 || U.plate.x < 0.5) {
      let c = stand_in(ro + rd * t_solid, rd, lg);
      textureStore(out_beauty, px, vec4<f32>(sp.rgb + c * sp.w, 1.0));
      textureStore(out_aux, px, vec4<f32>(1.0, depth, 0.0, 1.0));
    } else {
      var lit = vec3<f32>(0.0);
      if (lava_on() && U.lava[3].w > 0.0) {
        // the object in the footage, lit by a molten liquid's glow (liq_lava.wgsl)
        let ps = ro + rd * t_solid;
        lit = lava_relight(uv, U.org.xyz + ps * U.n.w, solid_n(ps)) * sp.w;
      }
      textureStore(out_beauty, px, vec4<f32>(sp.rgb + lit, 1.0 - sp.w));
      textureStore(out_aux, px, vec4<f32>(1.0, depth, 0.0, 1.0 - sp.w));
    }
    textureStore(out_emit, px, vec4<f32>(0.0));
    textureStore(out_mask, px, vec4<f32>(0.0, 1.0 - sp.w, 0.0, 0.0));
    return;
  }

  if (hit.y < 0.5 && t_open < 0.0) {
    // no liquid here: the footage may still be wet, shadowed or lit by caustics where the ray
    // meets the ground, and spray may hang in the air
    var add = vec3<f32>(0.0);
    var mult = 1.0;
    var spray = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if (r.y > r.x) { spray = spray_along(ro, rd, r.x, r.y, lg, seed); }
    // spray in front of the footage's surfaces only
    let sk = hold_keep(uv, rd_w, length(ro_w - U.scene.xyz) + max(r.x, 0.0) / cells_per_m);
    spray = vec4<f32>(spray.rgb * sk, 1.0 - (1.0 - spray.w) * sk);
    if (U.fwd.w > 0.5 && rd.y < -1e-5) {
      let tg = -ro.y / rd.y;
      let pg = ro + rd * tg;
      if (tg > 0.0 && pg.x > -2.0 && pg.z > -2.0 && pg.x < U.n.x + 2.0 && pg.z < U.n.z + 2.0) {
        mult = mix(1.0, ground_shade(pg, lg), hold_keep(uv, rd_w, length(ro_w - U.scene.xyz) + tg / cells_per_m));
        let w = wet_at(pg);
        if (w > 0.0 && U.wet.y > 0.0) {
          let rw = reflect(rd_w, vec3<f32>(0.0, 1.0, 0.0));
          let fr = fresnel(abs(rd_w.y), 1.0, U.optic.x);
          add = env(rw) * (w * U.wet.y * fr);
        }
      }
    }
    if (lava_on() && U.lava[3].w > 0.0 && U.fwd.w > 0.5 && U.plate.x > 0.5 && rd.y < -1e-5) {
      // a molten liquid's glow lights the ground around it (liq_lava.wgsl)
      let tl = -ro.y / rd.y;
      if (tl > 0.0) {
        let keep = hold_keep(uv, rd_w, length(ro_w - U.scene.xyz) + tl / cells_per_m);
        add += lava_relight(uv, U.org.xyz + (ro + rd * tl) * U.n.w, vec3<f32>(0.0, 1.0, 0.0)) * (mult * keep);
      }
    }
    let sa = 1.0 - spray.w;
    textureStore(out_beauty, px, vec4<f32>(spray.rgb + add * spray.w, sa));
    textureStore(out_emit, px, vec4<f32>(0.0));
    textureStore(out_aux, px, vec4<f32>(mix(mult, 1.0, sa), 0.0, 0.0, sa));
    textureStore(out_mask, px, vec4<f32>(0.0, sa, 0.0, 0.0));
    return;
  }

  var t_hit = hit.x;
  var p = vec3<f32>(0.0);
  var nrm = vec3<f32>(0.0, 1.0, 0.0);
  var sp = 0.0;
  var sh: Shade;
  var ic = 0.0;       // frozen share of the surface here (ice)
  if (lamps_on()) {
    let tl = select(t_open, hit.x, hit.y > 0.5);
    g_lamp = lamps_ambient(U.org.xyz + (ro + rd * tl) * U.n.w);
  }
  if (hit.y > 0.5) {
    p = ro + rd * t_hit;
    sp = length(surf(p).yzw);
    nrm = normal_at(p);
    nrm = sea_facing(sea_box_normal(p, nrm), rd);   // the sea's waves the simulation cannot carry, where it is the sea
    let n_ice = nrm;
    nrm = rain_rings(ripple(nrm, p), p);
    // ice keeps the shape it froze in: no ripples or rain rings on it (its frost: liq_ice_shade.wgsl)
    ic = ice_at(p);
    if (ic > 0.02) { nrm = normalize(mix(nrm, n_ice, ic)); }
    if (lava_on()) {
      sh = Shade(vec3<f32>(0.0), vec3<f32>(0.0), 0.0);   // opaque and glowing: liq_lava_shade.wgsl shades it
    } else {
      sh = shade_liquid(p, rd, nrm, lg, max_steps);
    }
  } else {
    // the open water past the box
    t_hit = t_open;
    p = ro + rd * t_hit;
    nrm = rain_rings(ripple(sea_normal(p), p), p);
    sh = sea_open_shade(p, rd, nrm, lg);   // ocn_shade.wgsl
  }
  var col = sh.col;
  var spec_out = sh.spec;

  // foam on the surface: a lacy white diffuse layer that hides the water (and its glints) under it
  var cover = 0.0;
  if (U.ww.w > 0.5 && U.ww.y > 0.0) {
    let fd = ww_at(p + nrm * (0.3 / U.nf.w)).y;
    cover = foam_cover(p, fd, surf(p).yzw) * (1.0 - ic);
  }
  // the sea's whitecaps and foam, with the simulation's own foam (ocn_shade.wgsl)
  let sf = sea_surface(p, nrm, rd, lg, col, cover);
  col = sf.rgb;
  cover = sf.w;
  spec_out *= 1.0 - cover;
  // spray between the camera and the surface
  let spray = spray_along(ro, rd, r.x, min(t_hit, r.y), lg, seed);
  col = spray.rgb + col * spray.w;

  let keep = hold_keep(uv, rd_w, length(ro_w - U.scene.xyz) + t_hit / cells_per_m);
  if (hit.y > 0.5) {
    // surfaces shaded by a pass of their own over this (liq_lava.wgsl): molten, ice
    let fpm = t_hit / cells_per_m * U.lava[5].x;
    if (lava_on()) { gbuf_hit(px, p, nrm, rd, fpm, keep, 1.0); }
    else if (ic > 0.5) { gbuf_hit(px, p, nrm, rd, fpm, keep, 2.0); }
  }
  textureStore(out_beauty, px, vec4<f32>(col, 1.0) * keep);
  // glints feed the bloom, compressed: a lens spreads a sun glint into a glow, not a white-out
  let sl = max(max(spec_out.r, spec_out.g), spec_out.b);
  let bloom_in = spec_out * (spray.w * 8.0 * (1.0 - exp(-sl / 8.0)) / max(sl, 1e-6));
  textureStore(out_emit, px, vec4<f32>(bloom_in, cover) * keep);
  textureStore(out_aux, px, vec4<f32>(1.0, t_hit / cells_per_m, sp * 0.1, keep));
  textureStore(out_mask, px, vec4<f32>(1.0, 1.0 - spray.w, sh.bub * spray.w * (1.0 - cover), 0.0) * keep);
}
