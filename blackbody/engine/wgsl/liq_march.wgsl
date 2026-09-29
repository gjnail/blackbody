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
  wet: vec4<f32>,       // wet darkening, wet gloss, _, _
  q: vec4<f32>,         // max steps, min step outside (surface cells), min step inside (surface cells), max interface events
  ww: vec4<f32>,        // whitewater: spray optical depth per density per cell, foam coverage per density, bubble optical depth per density per cell, on
  foam: vec4<f32>,      // foam colour (rgb), foam lace (0..1)
  org: vec4<f32>,       // grid corner (fire-local m), colliders drawn as grey stand-ins (1/0)
  lvl: vec4<f32>,       // water level (grid cells), open water on (1/0), caustics strength, blend width at the sides (cells)
  envp: vec4<f32>,      // HDRI on (1/0), rotation (rad), strength, foam cell size (m)
  ccnt: vec4<f32>,      // collider count
  col: array<Collider, MAX_COLLIDERS>,
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
@group(1) @binding(0) var<uniform> U: Params;

const PI: f32 = 3.14159265;

var<private> g_shift: f32 = 0.0;  // motion-blur time offset of this sample (s)

fn ign(p: vec2<f32>) -> f32 {
  return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

// ---- the surface ------------------------------------------------------------------------------

// Raw sample at grid position p: distance in grid cells, velocity (m/s).
fn surf_raw(p: vec3<f32>) -> vec4<f32> {
  let s = textureSampleLevel(surf_t, lin, p / U.n.xyz, 0.0);
  return vec4<f32>(s.x / U.nf.w, s.yzw);
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

// Distance in from the side walls (negative outside the box).
fn side_in(p: vec3<f32>) -> f32 {
  return min(min(p.x, U.n.x - p.x), min(p.z, U.n.z - p.z));
}

// The simulated surface, blended into the flat open water toward (and past) the open sides.
fn with_level(p: vec3<f32>, s: f32) -> f32 {
  if (!open_water()) { return s; }
  let w = smoothstep(0.0, U.lvl.w, side_in(p));
  return mix(p.y - U.lvl.x, s, w);
}

fn phi(p: vec3<f32>) -> f32 { return with_level(p, max(surf(p).x, box_d(p))); }

// Whitewater densities at p: x spray, y foam, z bubbles.
fn ww_at(p: vec3<f32>) -> vec3<f32> {
  return textureSampleLevel(ww_t, lin, p / U.n.xyz, 0.0).xyz;
}

// Light a white scattering medium (spray, foam, bubbles) receives: sky, plus the key light.
fn ww_light(nrm: vec3<f32>, lg: vec3<f32>) -> vec3<f32> {
  return U.sky.rgb * 0.95 + U.sun.rgb * (0.25 + 0.6 * max(dot(nrm, lg), 0.0));
}

// Cubic B-spline filtered distance (eight trilinear taps): a smooth surface for normals.
fn phi_cubic(p_in: vec3<f32>) -> f32 {
  var p = p_in;
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

// Small ripples the grid cannot hold, stronger where the liquid moves fast.
fn ripple(nrm: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
  if (U.ripple.x <= 0.0) { return nrm; }
  let sp = length(surf(p).yzw);
  let amt = U.ripple.x * clamp(sp / max(U.ripple.z, 1e-3), 0.0, 1.0);
  if (amt <= 0.0) { return nrm; }
  let t = U.frame.w;
  let q = p * U.n.w * U.ripple.y + vec3<f32>(0.37 * t, -0.9 * t, 0.21 * t);
  let g = gnoise_d(q).yzw + 0.5 * gnoise_d(q * 2.13 + vec3<f32>(13.1, 7.7, 3.3)).yzw;
  let tg = g - nrm * dot(nrm, g);
  return normalize(nrm - 0.35 * amt * tg);
}

fn box_range(ro: vec3<f32>, rd: vec3<f32>) -> vec2<f32> {
  let n = U.n.xyz;
  let safe = select(rd, vec3<f32>(1e-8), abs(rd) < vec3<f32>(1e-8));
  let inv = 1.0 / safe;
  let ta = (vec3<f32>(0.0) - ro) * inv;
  let tb = (n - ro) * inv;
  let t0 = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), max(min(ta.z, tb.z), 0.0));
  let t1 = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z));
  return vec2<f32>(t0, t1);
}

// ---- colliders ----------------------------------------------------------------------------------

fn solid_d(p: vec3<f32>) -> f32 {
  let cnt = i32(U.ccnt.x);
  if (cnt == 0) { return 1.0e6; }
  let w = U.org.xyz + p * U.n.w;
  var d = 1.0e9;
  for (var i = 0; i < cnt; i++) { d = min(d, col_sdf(U.col[i], w)); }
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
  for (var i = 0; i < 160; i++) {
    let d = solid_d(ro + rd * t);
    if (d < 0.02) { return t; }
    t += max(d * 0.9, 0.05);
    if (t >= tmax) { return -1.0; }
  }
  return -1.0;
}

// ---- tracing -----------------------------------------------------------------------------------

// First surface along the ray in [t0, t1] (grid cells): x = distance (-1 for none), y = what was hit
// (1 liquid, 2 collider).
fn trace_out(ro: vec3<f32>, rd: vec3<f32>, t0: f32, t1: f32, max_steps: i32) -> vec2<f32> {
  let minstep = U.q.y / U.nf.w;
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
      return vec3<f32>(0.5 * (a + b), 1.0, bub);
    }
  }
  return vec3<f32>(t, 0.0, bub);
}

// Spray along the camera ray between t0 and t1 (grid cells): (in-scattered light, transmittance).
fn spray_along(ro: vec3<f32>, rd: vec3<f32>, t0: f32, t1: f32, lg: vec3<f32>, seed: f32) -> vec4<f32> {
  if (U.ww.w < 0.5 || U.ww.x <= 0.0 || t1 <= t0) { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
  let st = max(0.75, (t1 - t0) / 160.0);
  var t = t0 + st * seed;
  var tr = 1.0;
  var col = vec3<f32>(0.0);
  let light = ww_light(vec3<f32>(0.0, 1.0, 0.0), lg);
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
  // fade out toward the box edges: liquid that ran out of the box wetted the ground beyond it too,
  // so a hard edge there would be wrong
  let edge = min(min(pg.x, U.n.x - pg.x), min(pg.z, U.n.z - pg.z));
  return s * smoothstep(0.0, 6.0, edge);
}

// Direct sunlight at a ground point relative to the sunlight with no liquid in the way: caustics
// where the surface focuses it, shadow where the liquid absorbs or scatters it. 1 outside the map.
fn caustic_at(pg: vec3<f32>, lg: vec3<f32>) -> f32 {
  let uv = pg.xz / U.n.xz;
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
  var m = 1.0 - U.wet.x * wet_at(pg);
  if (U.lvl.z > 0.0 && lg.y > 0.02) {
    let es = luma(U.sun.rgb) * lg.y;
    let frac = es / max(es + luma(U.sky.rgb), 1e-4);
    let c = caustic_at(pg, lg);
    let k = select(U.lvl.z, U.sundir.w, c < 1.0);
    m *= 1.0 + (c - 1.0) * k * frac;
  }
  return m;
}

// What a collider shows at grid point p seen along d: its own pixels in the footage, or a plain grey
// stand-in.
fn stand_in(p: vec3<f32>, d: vec3<f32>, lg: vec3<f32>) -> vec3<f32> {
  if (U.org.w > 0.5 || U.plate.x < 0.5) {
    let nn = solid_n(p);
    let nw = to_world_dir(nn);
    return vec3<f32>(0.3) * (sky(nw) * (0.55 + 0.45 * nw.y) + U.sun.rgb * max(dot(nn, lg), 0.0));
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

// Open water beyond the box: the flat sheet at the water level. Distance along the ray to it, if
// the ray meets it outside the box footprint (from above), else -1.
fn open_sheet(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
  if (!open_water() || rd.y > -1e-5 || ro.y < U.lvl.x) { return -1.0; }
  let t = (U.lvl.x - ro.y) / rd.y;
  if (t <= 0.0) { return -1.0; }
  if (side_in(ro + rd * t) > 0.5 * U.lvl.w) { return -1.0; }
  return t;
}

// A ray under the open water (outside the box, or leaving it under water): on to the ground, or up
// through the flat surface. Returns what it sees, attenuated by the water on the way.
fn under_open(p: vec3<f32>, d: vec3<f32>, lg: vec3<f32>) -> vec3<f32> {
  var L = 0.0;
  var col = vec3<f32>(0.0);
  if (d.y < -1e-4 && U.fwd.w > 0.5) {
    L = -p.y / d.y;
    let tr = exp(-(U.absorb.rgb + vec3<f32>(U.absorb.w)) * L * U.n.w);
    col = background(p, d, lg) * tr + (1.0 - exp(-U.absorb.w * L * U.n.w)) * U.murk.rgb * U.sky.rgb;
  } else {
    L = (U.lvl.x - p.y) / max(d.y, 1e-4);
    let tr = exp(-(U.absorb.rgb + vec3<f32>(U.absorb.w)) * L * U.n.w);
    let out_dir = refract(d, vec3<f32>(0.0, -1.0, 0.0), U.optic.x);
    var seen = sky(to_world_dir(reflect(d, vec3<f32>(0.0, -1.0, 0.0)))) * 0.2;
    if (dot(out_dir, out_dir) > 1e-6) { seen = env(to_world_dir(out_dir)); }
    col = seen * tr + (1.0 - exp(-U.absorb.w * L * U.n.w)) * U.murk.rgb * U.sky.rgb;
  }
  return col;
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
};

fn shade_liquid(p: vec3<f32>, rd: vec3<f32>, nrm: vec3<f32>, lg: vec3<f32>, max_steps: i32) -> Shade {
  let h = U.n.w;
  let ior = U.optic.x;
  let eps = 0.35 / U.nf.w;
  let cosi = clamp(-dot(rd, nrm), 0.0, 1.0);
  let F = fresnel(cosi, 1.0, ior);

  // reflection and glint
  var col = F * U.optic.z * reflected(p + nrm * eps, reflect(rd, nrm), lg);
  let spec = U.sun.rgb * (glint(nrm, -rd, lg) * PI);
  col += spec;

  // refraction through the body, out the far side, possibly through more drops
  var thr = vec3<f32>(1.0 - F);
  var pos = p - nrm * eps;
  var dir = refract(rd, nrm, 1.0 / ior);
  var inside = true;
  var done = false;
  let events = i32(U.q.w);
  for (var ev = 0; ev < events; ev++) {
    if (inside) {
      let rb = box_range(pos, dir);
      let ex = trace_in(pos, dir, max(rb.y, 0.0), max_steps);
      let L = ex.x * h;
      let tr_s = exp(-U.absorb.w * L);
      col += thr * (1.0 - tr_s) * U.murk.rgb * (U.sky.rgb + U.sun.rgb * 0.25);
      thr *= exp(-U.absorb.rgb * L) * tr_s;
      if (ex.z > 0.0) {
        let tb = exp(-ex.z * U.ww.z);
        col += thr * (1.0 - tb) * ww_light(vec3<f32>(0.0, 1.0, 0.0), lg) * 0.8;
        thr *= tb;
      }
      pos = pos + dir * ex.x;
      if (ex.y > 1.5) {
        col += thr * stand_in(pos, dir, lg);
        done = true;
        break;
      }
      if (ex.y < 0.5) {
        // left the box still in the liquid: on through the open water, or out at the box wall
        if (open_water() && pos.y < U.lvl.x) { col += thr * under_open(pos, dir, lg); }
        else { col += thr * background(pos, dir, lg); }
        done = true;
        break;
      }
      let n2 = normal_at(pos);
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
      let n3 = normal_at(pos);
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
  s.spec = spec;
  return s;
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
  if (U.frame.z > 0.5) {
    g_shift = U.frame.y * (fract(ign(vec2<f32>(px) + vec2<f32>(17.0, 59.0)) + U.frame.x * 0.7548776662) - 0.5);
  }
  let max_steps = i32(U.q.x);
  let seed = fract(ign(vec2<f32>(px) + vec2<f32>(3.0, 7.0)) + U.frame.x * 0.381966);

  let r = box_range(ro, rd);
  var hit = vec2<f32>(-1.0, 0.0);
  if (r.y > r.x) {
    let jit = fract(ign(vec2<f32>(px)) + U.frame.x * 0.61803398875) * U.q.y / U.nf.w;
    hit = trace_out(ro, rd, r.x + jit, r.y, max_steps);
  }
  // colliders outside the box still hold the liquid out
  var t_solid = -1.0;
  if (hit.y > 1.5) { t_solid = hit.x; }
  else if (U.ccnt.x > 0.5) {
    let ts = trace_solid(ro, rd, select(1.0e5, hit.x, hit.y > 0.5));
    if (ts >= 0.0) { t_solid = ts; }
  }
  // the open water past the box: first when the ray crosses it before it reaches any liquid in the
  // box (a ray that goes under the open water and on into the box stays with the open water)
  var t_open = -1.0;
  let t_sheet = open_sheet(ro, rd);
  if (t_sheet >= 0.0 && (hit.y < 0.5 || t_sheet < hit.x - 0.5)) {
    if (t_solid < 0.0 || t_sheet < t_solid) {
      t_open = t_sheet;
      t_solid = -1.0;
      hit = vec2<f32>(-1.0, 0.0);
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
      textureStore(out_beauty, px, vec4<f32>(sp.rgb, 1.0 - sp.w));
      textureStore(out_aux, px, vec4<f32>(1.0, depth, 0.0, 1.0 - sp.w));
    }
    textureStore(out_emit, px, vec4<f32>(0.0));
    return;
  }

  if (hit.y < 0.5 && t_open < 0.0) {
    // no liquid here: the footage may still be wet, shadowed or lit by caustics where the ray
    // meets the ground, and spray may hang in the air
    var add = vec3<f32>(0.0);
    var mult = 1.0;
    var spray = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if (r.y > r.x) { spray = spray_along(ro, rd, r.x, r.y, lg, seed); }
    if (U.fwd.w > 0.5 && rd.y < -1e-5) {
      let tg = -ro.y / rd.y;
      let pg = ro + rd * tg;
      if (tg > 0.0 && pg.x > -2.0 && pg.z > -2.0 && pg.x < U.n.x + 2.0 && pg.z < U.n.z + 2.0) {
        mult = ground_shade(pg, lg);
        let w = wet_at(pg);
        if (w > 0.0 && U.wet.y > 0.0) {
          let rw = reflect(rd_w, vec3<f32>(0.0, 1.0, 0.0));
          let fr = fresnel(abs(rd_w.y), 1.0, U.optic.x);
          add = env(rw) * (w * U.wet.y * fr);
        }
      }
    }
    let sa = 1.0 - spray.w;
    textureStore(out_beauty, px, vec4<f32>(spray.rgb + add * spray.w, sa));
    textureStore(out_emit, px, vec4<f32>(0.0));
    textureStore(out_aux, px, vec4<f32>(mix(mult, 1.0, sa), 0.0, 0.0, sa));
    return;
  }

  var t_hit = hit.x;
  var p = vec3<f32>(0.0);
  var nrm = vec3<f32>(0.0, 1.0, 0.0);
  var sp = 0.0;
  var sh: Shade;
  if (hit.y > 0.5) {
    p = ro + rd * t_hit;
    sp = length(surf(p).yzw);
    nrm = ripple(normal_at(p), p);
    sh = shade_liquid(p, rd, nrm, lg, max_steps);
  } else {
    // the flat open water past the box
    t_hit = t_open;
    p = ro + rd * t_hit;
    nrm = ripple(vec3<f32>(0.0, 1.0, 0.0), p);
    let F = fresnel(clamp(-dot(rd, nrm), 0.0, 1.0), 1.0, U.optic.x);
    let spec = U.sun.rgb * (glint(nrm, -rd, lg) * PI);
    let below = under_open(p, refract(rd, nrm, 1.0 / U.optic.x), lg);
    sh.col = F * U.optic.z * background(p + nrm * 0.01, reflect(rd, nrm), lg) + spec + (1.0 - F) * below;
    sh.spec = spec;
  }
  var col = sh.col;
  var spec_out = sh.spec;

  // foam on the surface: a lacy white diffuse layer that hides the water (and its glints) under it
  var cover = 0.0;
  if (U.ww.w > 0.5 && U.ww.y > 0.0) {
    let fd = ww_at(p + nrm * (0.3 / U.nf.w)).y;
    cover = foam_cover(p, fd, surf(p).yzw);
    col = mix(col, U.foam.rgb * ww_light(nrm, lg), cover);
    spec_out *= 1.0 - cover;
  }
  // spray between the camera and the surface
  let spray = spray_along(ro, rd, r.x, min(t_hit, r.y), lg, seed);
  col = spray.rgb + col * spray.w;

  textureStore(out_beauty, px, vec4<f32>(col, 1.0));
  // glints feed the bloom, compressed: a lens spreads a sun glint into a glow, not a white-out
  let sl = max(max(spec_out.r, spec_out.g), spec_out.b);
  let bloom_in = spec_out * (spray.w * 8.0 * (1.0 - exp(-sl / 8.0)) / max(sl, 1e-6));
  textureStore(out_emit, px, vec4<f32>(bloom_in, cover));
  textureStore(out_aux, px, vec4<f32>(1.0, t_hit / cells_per_m, sp * 0.1, 1.0));
}
