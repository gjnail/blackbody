// Volume ray march of the fire domain, and the surfaces the fire sits among.
// beauty: premultiplied radiance (emission + scattered light), alpha = 1 - transmittance
// emit:   emission only (premultiplied), alpha = flame coverage
// aux:    x = heat (integrated temperature, m), y = depth (m, opacity-weighted), z = max Kelvin / 1000
// surf:   rgb = fire light arriving at the surface seen in this pixel (the ground, a collider in the
//         shot, or the footage's own surface from its depth pass), a = how wet it is (0..1)
// mask:   x = scorch (0 untouched .. 1 burnt out), y = holdout depth (clip w) * coverage, z = coverage
//         (colliders in the shot and the footage's matte or depth pass hide what is behind them),
//         w = soot left on the surface (0..1)
// lamp:   rgb = how the lights in the set change the light on the surface seen, relative to the light the
//         footage shows there (the footage is multiplied by 1 + lamp): lights added in CG brighten it,
//         and the smoke's shadow takes away some of the light of the real lamps in the footage
// deep:   with deep output on, up to deepp.x depth bins per pixel, shared by all the anti-aliasing
//         passes: the colour and opacity each bin adds to the pixel (absolute, summed over the passes),
//         then its front and back depth (m, along the view axis). The first pass that sees anything
//         places the bins where the opacity builds up; later passes add into the bin at their depth, and
//         open bins of their own for what lies where no bin is yet (a wisp only they catch).
//
// Where flames burn down onto the ground there can be a bed of glowing coals: a layer a few centimetres
// deep of lumps that glow at their own temperature (hottest deep in the bed, duller on top) with gaps
// between them, marched with the gas, over charred ground (mask.x).
//
// Colliders marked "hides fire" are found by sphere tracing their exact shapes (so they work outside
// the simulation box too); the volume march stops at them, and at the footage's own surfaces when it
// has a depth pass. Fire light on surfaces is gathered from a short list of point lights built from
// the fire's emission (lights.wgsl): brightest facing the flames, falling off with the square of the
// distance, and shaded by colliders and smoke between the surface and the fire (one soft shadow ray
// toward the lights' weighted centre, its penumbra as wide as the fire).
//
// Lights in the set (lamps.wgsl) are worked out exactly at every step of the march: their falloff, a
// spot's cone and an area light's facing, scattered by the smoke's phase function; only the smoke's
// shadow on them comes from the light volume.
//!include common.wgsl
//!include noise.wgsl
//!include meshsdf.wgsl
//!include colliders.wgsl
//!include burn_common.wgsl
//!include burnobj.wgsl

@group(0) @binding(0) var scal: texture_3d<f32>;
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var L0: texture_3d<f32>;
@group(0) @binding(3) var L1: texture_3d<f32>;
@group(0) @binding(4) var noise: texture_3d<f32>;
@group(0) @binding(5) var bb: texture_2d<f32>;
@group(0) @binding(6) var lin: sampler;
@group(0) @binding(7) var rep: sampler;
@group(0) @binding(8) var aux: texture_3d<f32>;
@group(0) @binding(9) var chem: texture_3d<f32>;
@group(0) @binding(10) var out_beauty: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var out_emit: texture_storage_2d<rgba16float, write>;
@group(0) @binding(12) var out_aux: texture_storage_2d<rgba16float, write>;
@group(0) @binding(13) var out_surf: texture_storage_2d<rgba16float, write>;
@group(0) @binding(14) var out_mask: texture_storage_2d<rgba16float, write>;
@group(0) @binding(15) var<storage, read> lights: array<vec4<f32>>;   // per light: position (fire-local m), softening (m); power (rgb)
@group(0) @binding(16) var<storage, read> light_count: array<u32>;
@group(0) @binding(17) var burn: texture_3d<f32>;                     // burnable floor (simulation grid)
@group(0) @binding(18) var burn_obj: texture_3d<f32>;
@group(0) @binding(19) var<storage, read> slots: array<BurnSlot>;
@group(0) @binding(20) var atlas: texture_3d<f32>;
@group(0) @binding(21) var limit: texture_2d<f32>;                    // liquid in front: y = its depth (m), w = coverage
@group(0) @binding(22) var stain: texture_3d<f32>;                    // soot left on surfaces (simulation grid, r32float)
@group(0) @binding(23) var hold: texture_2d<f32>;                     // holdouts from the footage: x = matte, y = depth pass
@group(0) @binding(24) var<storage, read_write> deep: array<vec4<f32>>;  // deep bins, 2 vec4 each
@group(0) @binding(25) var stain_obj: texture_3d<f32>;                // soot on colliders, each in its own frame
@group(0) @binding(26) var<storage, read> stain_slots: array<SootSlot>;  // one region per collider
@group(0) @binding(27) var LT: texture_3d<f32>;                       // lamps' transmittance through the smoke (lamps.wgsl)
@group(0) @binding(28) var<storage, read> lamps: array<Lamp>;         // the lights in the set (fire-local m)
@group(0) @binding(29) var out_lamp: texture_storage_2d<rgba16float, write>;

struct Lamp { p: vec4<f32>, c: vec4<f32>, d: vec4<f32>, e: vec4<f32> };  // see lamps.wgsl

//!include shade.wgsl
//!include lamp_shape.wgsl

struct Params {
  inv_vp: mat4x4<f32>,   // clip -> world
  w2g: mat4x4<f32>,      // world -> grid (cells)
  vp: mat4x4<f32>,       // world -> clip
  n: vec4<f32>,          // grid dims, metres per cell
  ln: vec4<f32>,         // light-volume dims, ground (1 = no fade at the floor)
  res: vec4<f32>,        // width, height, sub-pixel jitter (px)
  frame: vec4<f32>,      // noise seed, shutter (s), motion blur on, max steps
  look: Look,
  org: vec4<f32>,        // grid corner (fire-local m); w = grid cells per velocity cell (upres)
  surf: vec4<f32>,       // holdouts on (1/0), ground in the shot (1/0), surfaces lit (1/0), scorch on (1/0)
  bgrid: vec4<f32>,      // burnable floor grid dims (simulation grid); w = its cell size (m)
  shad: vec4<f32>,       // x = shadows in the fire light (0..1), y = soot on (1/0), z = wetness on (1/0), w = soot per unit stain
  hold: vec4<f32>,       // footage holdouts: x = matte on, y = depth pass on, z = depth kind (0 Z, 1 distance, 2 inverse Z), w = metres per unit
  hold2: vec4<f32>,      // footage fit scale (x, y), collider soot regions, lights in the set (count)
  fwd: vec4<f32>,        // camera forward (world, unit); w = near plane distance (m)
  deepp: vec4<f32>,      // deep bins: x = most per pixel (0 = off), y = image width, z = opacity at which a bin is split off, w = pass
  ccnt: vec4<f32>,
  col: array<Collider, MAX_COLLIDERS>,
  lim: vec4<f32>,        // x = stop the march at the liquid surface in `limit` (fire and liquid in one box);
                         // y = cloth in the light volume (L1.z: its extinction, 1/m; it shadows the surfaces)
  coal: vec4<f32>,       // coal bed: x = brightness (0 = none), y = temperature (K), z = lump frequency (1/m), w = height (m)
};
@group(1) @binding(0) var<uniform> U: Params;

const COAL_SIGMA: f32 = 120.0;  // extinction of a solid lump of coal (1/m): opaque within a centimetre or two
const COAL_ALBEDO: f32 = 0.12;  // charcoal and ash
// Exposure lift for the coals. Next to flame (1 = thick flame at the flame temperature) a bed at its true
// blackbody brightness would be invisible against the footage; with this, a bed at coal temperature reads
// at about half the brightness of thick flame, as it does in footage exposed for the fire.
const COAL_GAIN: f32 = 15.0;

fn ign(p: vec2<f32>) -> f32 {
  return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

// Nearest collider in the shot at fire-local point p: (distance (m), index).
fn holdout_sdf(p: vec3<f32>) -> vec2<f32> {
  var best = vec2<f32>(1.0e9, -1.0);
  for (var i = 0; i < i32(U.ccnt.x); i++) {
    if (U.col[i].y.w < 0.5) { continue; }
    let d = col_sdf(U.col[i], p);
    if (d < best.x) { best = vec2<f32>(d, f32(i)); }
  }
  return best;
}

fn col_normal(k: Collider, p: vec3<f32>, e: f32) -> vec3<f32> {
  let g = vec3<f32>(col_sdf(k, p + vec3<f32>(e, 0.0, 0.0)) - col_sdf(k, p - vec3<f32>(e, 0.0, 0.0)),
                    col_sdf(k, p + vec3<f32>(0.0, e, 0.0)) - col_sdf(k, p - vec3<f32>(0.0, e, 0.0)),
                    col_sdf(k, p + vec3<f32>(0.0, 0.0, e)) - col_sdf(k, p - vec3<f32>(0.0, 0.0, e)));
  let l = length(g);
  return select(vec3<f32>(0.0, 1.0, 0.0), g / l, l > 1e-8);
}

// How much of the fire's light gets from the fire (centre pc, size spread, fire-local m) to point p:
// colliders in the shot cast soft shadows, and smoke between them dims the light.
fn fire_visibility(p: vec3<f32>, pc: vec3<f32>, spread: f32) -> f32 {
  let h = U.bgrid.w;
  let d = pc - p;
  let dist = length(d);
  if (dist < 1e-4) { return 1.0; }
  let dir = d / dist;
  let reach = max(dist - 0.5 * spread, 0.0);   // stop where the fire itself begins
  var vis = 1.0;
  if (U.surf.x > 0.5 && U.ccnt.x > 0.5) {
    var t = 1.5 * h;
    for (var i = 0; i < 40; i++) {
      if (t >= reach) { break; }
      let dh = holdout_sdf(p + dir * t).x;
      let pen = max(spread * t / dist, 0.5 * h);   // the penumbra widens with distance from the surface
      vis = min(vis, clamp(0.5 + 0.5 * dh / pen, 0.0, 1.0));
      if (vis < 0.01) { return 0.0; }
      t += clamp(dh, 0.4 * h, 0.25 * dist);
    }
  }
  // smoke and steam in the way (not the flames: that is where the light comes from)
  let n = U.n.xyz;
  let hv = U.n.w;
  var od = 0.0;
  let steps = 16;
  let dt_m = reach / f32(steps);
  for (var i = 0; i < steps; i++) {
    let q = (p + dir * ((f32(i) + 0.5) * dt_m) - U.org.xyz) / hv;
    if (any(q < vec3<f32>(0.0)) || any(q > n)) { continue; }
    let s = samp_c(scal, lin, q, n);
    var ax = vec4<f32>(0.0);
    if (U.look.air.z > 0.5) { ax = samp_c(aux, lin, q, n); }
    od += (smoke_extinction(s, U.look) + steam_extinction(s, ax, U.look)) * dt_m;
  }
  return vis * exp(-od - cloth_depth(p, dir, reach));
}

// Optical depth of the cloth (the light volume's L1.z, cloth.py occlusion) from fire-local point p along
// unit dir for `reach` metres, in steps under a light cell (a sheet is only a couple of cells thick there).
fn cloth_depth(p: vec3<f32>, dir: vec3<f32>, reach: f32) -> f32 {
  if (U.lim.y < 0.5) { return 0.0; }
  let ln = U.ln.xyz;
  let lc = U.n.w * U.n.xyz / ln;   // metres per light cell, per axis
  let o = (p - U.org.xyz) / lc;
  let dl = dir / lc;
  let safe = select(dl, vec3<f32>(1e-8), abs(dl) < vec3<f32>(1e-8));
  let ta = (vec3<f32>(0.0) - o) / safe;
  let tb = (ln - o) / safe;
  let t0 = max(max(max(min(ta.x, tb.x), min(ta.y, tb.y)), min(ta.z, tb.z)), 0.0);
  let t1 = min(min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z)), reach);
  if (t1 <= t0) { return 0.0; }
  let dt = max((t1 - t0) / 96.0, 0.75 * min(lc.x, min(lc.y, lc.z)));
  var od = 0.0;
  var t = t0 + 0.5 * dt;
  for (var i = 0; i < 96; i++) {
    if (t >= t1) { break; }
    od += samp_c(L1, lin, o + dl * t, ln).z;
    t += dt;
  }
  return od * dt;
}

// Fire light arriving at a surface point with normal nrm (fire-local metres).
fn surface_light(p: vec3<f32>, nrm: vec3<f32>) -> vec3<f32> {
  var e = vec3<f32>(0.0);
  var c = vec3<f32>(0.0);
  var c2 = 0.0;
  var cw = 0.0;
  var r = 0.0;
  let count = light_count[0];
  for (var i = 0u; i < count; i++) {
    let a = lights[2u * i];
    let d = a.xyz - p;
    let r2 = dot(d, d);
    let cosine = dot(nrm, d) * inverseSqrt(max(r2, 1e-8));
    // a light the size of its block: wrap the cosine a little and soften the falloff near it
    let facing = clamp((cosine + 0.2) / 1.2, 0.0, 1.0);
    let l = lights[2u * i + 1u].rgb * (facing / (r2 + a.w * a.w));
    e += l;
    let w = luma(l);
    c += a.xyz * w;
    c2 += dot(a.xyz, a.xyz) * w;
    r += a.w * w;
    cw += w;
  }
  if (U.shad.x <= 0.0 || cw <= 1e-12) { return e; }
  // one shadow ray toward the centre of the light reaching this point; the fire's spread around that
  // centre sets how soft the shadows are
  let pc = c / cw;
  let spread = sqrt(max(c2 / cw - dot(pc, pc), 0.0)) + r / cw;
  return e * mix(1.0, fire_visibility(p, pc, spread), U.shad.x);
}

// -- lights in the set ------------------------------------------------------------------------------

struct LampAt { dir: vec3<f32>, f: f32 };

// Lamp L seen from fire-local point p: the unit direction toward it, and how much of its power arrives
// (per square metre, before shadows): the inverse square softened by its size, within a spot's cone
// (exactly, so a narrow beam is as sharp as the march) or in front of an area light.
fn lamp_at(k: i32, p: vec3<f32>) -> LampAt {
  let L = lamps[k];
  let v = L.p.xyz - p;
  let d2 = dot(v, v);
  let dir = v * inverseSqrt(max(d2, 1e-12));
  return LampAt(dir, lamp_shape(k, -dir) / (d2 + L.p.w * L.p.w));
}

// The lamps' transmittance at light-grid point pl, for lamps 4 * block .. 4 * block + 3.
fn lamp_tr4(pl: vec3<f32>, block: f32) -> vec4<f32> {
  let ln = U.ln.xyz;
  let blocks = select(1.0, 2.0, U.hold2.w > 4.5);
  let q = clamp(pl, vec3<f32>(0.5), ln - vec3<f32>(0.5));
  return textureSampleLevel(LT, lin, vec3<f32>(q.x / ln.x, q.y / ln.y, (q.z + block * ln.z) / (ln.z * blocks)), 0.0);
}

fn pick4(v: vec4<f32>, j: i32) -> f32 {
  return select(select(v.w, v.z, j == 2), select(v.y, v.x, j == 0), j < 2);
}

// Optical depth (smoke and steam) from fire-local point p along unit dir for `reach` metres, marched only
// where the path is inside the simulation box.
fn smoke_depth(p: vec3<f32>, dir: vec3<f32>, reach: f32) -> f32 {
  let n = U.n.xyz;
  let hv = U.n.w;
  let o = (p - U.org.xyz) / hv;
  let safe = select(dir, vec3<f32>(1e-8), abs(dir) < vec3<f32>(1e-8));
  let ta = (vec3<f32>(0.0) - o) / safe;
  let tb = (n - o) / safe;
  let t0 = max(max(max(min(ta.x, tb.x), min(ta.y, tb.y)), min(ta.z, tb.z)), 0.0);
  let t1 = min(min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z)), reach / hv);
  if (t1 <= t0) { return 0.0; }
  let steps = 20;
  let dt = (t1 - t0) / f32(steps);
  var od = 0.0;
  for (var i = 0; i < steps; i++) {
    let q = o + dir * (t0 + (f32(i) + 0.5) * dt);
    let s = samp_c(scal, lin, q, n);
    var ax = vec4<f32>(0.0);
    if (U.look.air.z > 0.5) { ax = samp_c(aux, lin, q, n); }
    od += (smoke_extinction(s, U.look) + steam_extinction(s, ax, U.look)) * dt * hv;
  }
  return od;
}

// How much light from a lamp at `dist` metres along dir reaches p past the colliders in the shot: soft
// shadows, their penumbra as wide as the lamp.
fn lamp_blockers(p: vec3<f32>, dir: vec3<f32>, dist: f32, radius: f32) -> f32 {
  if (U.surf.x < 0.5 || U.ccnt.x < 0.5) { return 1.0; }
  let h = U.bgrid.w;
  var vis = 1.0;
  var t = 1.5 * h;
  let reach = dist - radius;
  for (var i = 0; i < 40; i++) {
    if (t >= reach) { break; }
    let dh = holdout_sdf(p + dir * t).x;
    let pen = max(radius * t / max(dist, 1e-4), 0.5 * h);
    vis = min(vis, clamp(0.5 + 0.5 * dh / pen, 0.0, 1.0));
    if (vis < 0.01) { return 0.0; }
    t += clamp(dh, 0.4 * h, max(0.25 * dist, 0.4 * h));
  }
  return vis;
}

// How the lights in the set change the light on a surface at p (normal nrm), relative to the light the
// footage shows there. A light added in CG adds its light, shadowed by colliders and smoke. A real lamp
// in the footage already lights the surface, but not through smoke that is only in the simulation: its
// light blocked by the smoke is taken away. The footage's own light there is taken as the key light and
// the sky (the Lighting settings) plus the real lamps' own light.
fn lamp_surface(p: vec3<f32>, nrm: vec3<f32>) -> vec3<f32> {
  let count = i32(U.hold2.w + 0.5);
  var add = vec3<f32>(0.0);
  var cut = vec3<f32>(0.0);
  var real = vec3<f32>(0.0);
  for (var i = 0; i < count; i++) {
    let Lm = lamps[i];
    let a = lamp_at(i, p);
    let e = Lm.c.rgb * (a.f * max(dot(nrm, a.dir), 0.0));
    if (max(e.x, max(e.y, e.z)) <= 0.0) { continue; }
    let dist = length(Lm.p.xyz - p);
    let vis = lamp_blockers(p, a.dir, dist, Lm.p.w);
    var tr = 1.0;
    if (Lm.e.y > 0.5) { tr = exp(-smoke_depth(p, a.dir, dist) * U.look.sundir.w - cloth_depth(p, a.dir, dist)); }
    if (Lm.e.z > 0.5) {
      real += e * vis;
      cut += e * (vis * (1.0 - tr));
    } else {
      add += e * (vis * tr);
    }
  }
  let sd = normalize((U.w2g * vec4<f32>(U.look.sundir.xyz, 0.0)).xyz);   // toward the key light, fire-local
  let sun_e = U.look.sun.rgb * max(dot(nrm, sd), 0.0);
  let sky_e = 3.14159265 * U.look.amb.rgb * (0.6 + 0.4 * nrm.y);
  var base = sun_e + sky_e + real;
  base = max(base, vec3<f32>(0.02 * luma(base) + 1e-4));
  if (U.lim.y > 0.5) {
    // cloth only in the simulation shadows the footage's own key light and sky (from straight up; the
    // sky is all around, so not all of it)
    let reach = 4.0 * U.n.w * max(U.n.x, max(U.n.y, U.n.z));
    cut += sun_e * (1.0 - exp(-cloth_depth(p, sd, reach)));
    cut += sky_e * (0.7 * (1.0 - exp(-cloth_depth(p, vec3<f32>(0.0, 1.0, 0.0), reach))));
  }
  return (add - cut) / base;
}

// Soot left on surfaces at fire-local point p (the simulation grid's r32float stain, trilinear).
fn soot_at(p: vec3<f32>) -> f32 {
  let d = vec3<i32>(U.bgrid.xyz);
  let q = (p - U.org.xyz) / U.bgrid.w - vec3<f32>(0.5);
  let i0 = vec3<i32>(floor(q));
  let f = q - floor(q);
  var acc = 0.0;
  for (var k = 0; k < 8; k++) {
    let o = vec3<i32>(k & 1, (k >> 1) & 1, k >> 2);
    let w = mix(vec3<f32>(1.0) - f, f, vec3<f32>(o));
    acc += textureLoad(stain, clamp(i0 + o, vec3<i32>(0), d - vec3<i32>(1)), 0).x * w.x * w.y * w.z;
  }
  return 1.0 - exp(-max(acc, 0.0) * U.shad.w);
}

// A collider's region of the soot atlas, in size-relative coordinates (stain_obj.wgsl).
struct SootSlot { lo: vec4<f32>, dims: vec4<f32>, cell: vec4<f32> };

// Soot on collider i at world point p (its own soot region, trilinear), 0..1.
fn collider_soot(i: i32, p: vec3<f32>) -> f32 {
  if (f32(i) >= U.hold2.z) { return 0.0; }
  let k = U.col[i];
  let s = stain_slots[i];
  let d = vec3<i32>(s.dims.xyz);
  let q = (col_to_local(k, p) / col_scale(k) - s.lo.xyz) / s.cell.xyz - vec3<f32>(0.5);
  let i0 = vec3<i32>(floor(q));
  let f = q - floor(q);
  var acc = 0.0;
  for (var j = 0; j < 8; j++) {
    let o = vec3<i32>(j & 1, (j >> 1) & 1, j >> 2);
    let w = mix(vec3<f32>(1.0) - f, f, vec3<f32>(o));
    let c = clamp(i0 + o, vec3<i32>(0), d - vec3<i32>(1));
    acc += textureLoad(stain_obj, vec3<i32>(c.x, c.y, c.z + i32(s.lo.w)), 0).x * w.x * w.y * w.z;
  }
  return 1.0 - exp(-max(acc, 0.0) * U.shad.w);
}

// The footage's depth pass at uv as a distance along the ray from the camera (m), or 0 where it has none.
fn footage_distance(huv: vec2<f32>, rd_w: vec3<f32>) -> f32 {
  let v = textureSampleLevel(hold, lin, huv, 0.0).y;
  if (!(v > 0.0) || v > 1.0e20) { return 0.0; }
  let kind = i32(U.hold.z + 0.5);
  var d = v * U.hold.w;
  if (kind == 2) { d = U.hold.w / v; }
  if (kind != 1) { d = d / max(dot(rd_w, U.fwd.xyz), 1e-3); }   // Z along the view axis -> along the ray
  return d;
}

// How much of the ground at grid point g (cells) is under coals, 0..1: where the gas just above the
// bed is hot, averaged over a few centimetres so the bed follows the base of the fire, not every lick.
fn coal_cover(g: vec3<f32>) -> f32 {
  let n = U.n.xyz;
  let h = U.n.w;
  let y = max(1.0, (U.coal.w + 0.02) / h);
  let r = max(1.5, 0.06 / h);
  var heat = samp_c(scal, lin, vec3<f32>(g.x, y, g.z), n).x;
  heat += samp_c(scal, lin, vec3<f32>(g.x + r, y, g.z), n).x;
  heat += samp_c(scal, lin, vec3<f32>(g.x - r, y, g.z), n).x;
  heat += samp_c(scal, lin, vec3<f32>(g.x, y, g.z + r), n).x;
  heat += samp_c(scal, lin, vec3<f32>(g.x, y, g.z - r), n).x;
  return smoothstep(0.05, 0.45, max(heat, 0.0) * 0.2);
}

// The coal bed at grid point p (cells): x = extinction (1/m), y = temperature (K); zero outside it.
// Heaped highest where it is most covered; lumps from the detail noise, shifting over a few seconds.
fn coal_at(p: vec3<f32>, L: Look) -> vec2<f32> {
  let h = U.n.w;
  let n = U.n.xyz;
  if (p.x < 0.0 || p.z < 0.0 || p.x > n.x || p.z > n.z) { return vec2<f32>(0.0); }
  let cover = coal_cover(p);
  let top = U.coal.w * sqrt(cover);
  let y = p.y * h;
  if (cover <= 0.0 || y >= top) { return vec2<f32>(0.0); }
  let hgt = y / top;
  let pw = U.org.xyz + p * h;
  let nz = textureSampleLevel(noise, rep, pw * U.coal.z + vec3<f32>(0.0, 0.0, L.misc.w * 0.05), 0.0);
  let lump = smoothstep(0.40 + 0.2 * hgt, 0.55 + 0.2 * hgt, nz.x);
  let k = U.coal.y * (1.05 - 0.15 * hgt) * (0.8 + 0.4 * nz.y);
  return vec2<f32>(COAL_SIGMA * lump * cover, k);
}

// Light the coals give off per metre: like flame, in proportion to how strongly they absorb (Kirchhoff).
fn coal_emission(c: vec2<f32>, L: Look) -> vec3<f32> {
  if (c.x <= 0.0) { return vec3<f32>(0.0); }
  let b = bb_lookup(c.y);
  return b.rgb * min(pow(10.0, (b.a - L.misc.x) * L.fire.w), 1.0e4) * L.fire2.x * COAL_GAIN * U.coal.x * c.x;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let px = vec2<i32>(id.xy);
  if (f32(px.x) >= U.res.x || f32(px.y) >= U.res.y) { return; }
  let L = U.look;
  let n = U.n.xyz;
  let h = U.n.w;

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

  let safe = select(rd, vec3<f32>(1e-8), abs(rd) < vec3<f32>(1e-8));
  let inv = 1.0 / safe;
  let ta = (vec3<f32>(0.0) - ro) * inv;
  let tb = (n - ro) * inv;
  let t_in = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), max(min(ta.z, tb.z), 0.0));
  var t_out = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z));

  // -- surfaces: colliders in the shot and the ground --------------------------------------------
  let far = max(t_out, 0.0) + 2.0 * max(n.x, max(n.y, n.z));   // cells: the box and as far again beyond
  var t_obj = 1.0e9;
  var hit = -1;
  if (U.surf.x > 0.5 && U.ccnt.x > 0.5) {
    var t = 0.0;
    for (var i = 0; i < 160; i++) {
      let pl = U.org.xyz + (ro + rd * t) * h;
      let dh = holdout_sdf(pl);
      if (dh.x < 0.3 * h) {
        let k = U.col[i32(dh.y)];
        if (k.a.w > 2.5 || k.x.w > 0.0 || all(k.y.xyz > vec3<f32>(0.0))) {
          t_obj = t;
          hit = i32(dh.y);
          break;
        }
        // a plain sphere, box or cylinder: on onto its surface itself, so that what is behind its edge (cloth, the
        // footage) is hidden to its silhouette and not in a band 0.3 cells round it; a ray only passing near goes on
        var tt = t;
        var dd = dh.x;
        var on = true;
        for (var j = 0; j < 24; j++) {
          if (dd < 1.0e-3) { break; }
          let q = col_sdf(k, U.org.xyz + (ro + rd * (tt + dd / h)) * h);
          if (q >= dd) { on = false; break; }
          tt += dd / h;
          dd = q;
        }
        if (on) {
          t_obj = tt;
          hit = i32(dh.y);
          break;
        }
        t = tt;
      }
      t += max(dh.x / h, 0.25);
      if (t > far) { break; }
    }
  }
  var t_gnd = 1.0e9;
  if (U.surf.y > 0.5 && rd.y < -1e-6 && ro.y > 0.0) {
    t_gnd = -ro.y / rd.y;
  }
  // holdouts from the footage: a matte of what is in front of the fire, and its depth pass
  let near = U.fwd.w / max(dot(rd_w, U.fwd.xyz), 1e-3);   // camera to where the ray starts (m)
  var matte = 0.0;
  var t_hold = 1.0e9;
  let huv = (uv - vec2<f32>(0.5)) * U.hold2.xy + vec2<f32>(0.5);
  if (U.hold.x > 0.5) { matte = clamp(textureSampleLevel(hold, lin, huv, 0.0).x, 0.0, 1.0); }
  if (U.hold.y > 0.5) {
    let fd = footage_distance(huv, rd_w);
    if (fd > 0.0) { t_hold = max(fd - near, 0.0) * cells_per_m; }
  }
  var surf = vec4<f32>(0.0);
  var mask = vec4<f32>(0.0);
  var lamp_s = vec3<f32>(0.0);
  let t_s = min(min(t_obj, t_gnd), t_hold);
  let bh = U.bgrid.w;
  if (t_s < 1.0e8) {
    let ps = U.org.xyz + (ro + rd * t_s) * h;
    var nrm = vec3<f32>(0.0, 1.0, 0.0);
    let clip = U.vp * vec4<f32>(ro_w + rd_w * (t_s / cells_per_m), 1.0);
    if (t_hold <= t_s) {
      // the footage's own surface: its normal from the depth pass one pixel either side
      let du = vec2<f32>(1.0 / U.res.x, 0.0);
      let dv = vec2<f32>(0.0, 1.0 / U.res.y);
      let ndx = vec2<f32>(2.0 / U.res.x, 0.0);
      let ndy = vec2<f32>(0.0, -2.0 / U.res.y);
      let rx = normalize((U.inv_vp * vec4<f32>(ndc + ndx, 1.0, 1.0)).xyz / (U.inv_vp * vec4<f32>(ndc + ndx, 1.0, 1.0)).w - ro_w);
      let ry = normalize((U.inv_vp * vec4<f32>(ndc + ndy, 1.0, 1.0)).xyz / (U.inv_vp * vec4<f32>(ndc + ndy, 1.0, 1.0)).w - ro_w);
      let dx = footage_distance(huv + du * U.hold2.x, rx);
      let dy = footage_distance(huv + dv * U.hold2.y, ry);
      let wp = ro_w + rd_w * (t_s / cells_per_m);
      if (dx > 0.0 && dy > 0.0) {
        let wx = ro_w + rx * (dx - near);
        let wy = ro_w + ry * (dy - near);
        var nw = normalize(cross(wx - wp, wy - wp));
        if (dot(nw, rd_w) > 0.0) { nw = -nw; }
        // world -> fire-local: the grid transform's rotation, undone of its scale
        let gl = (U.w2g * vec4<f32>(nw, 0.0)).xyz;
        nrm = normalize(gl);
      } else {
        nrm = normalize(-rd);
      }
      mask = vec4<f32>(0.0, clip.w, 1.0, 0.0);
    } else if (hit >= 0 && t_obj <= t_gnd) {
      let k = U.col[hit];
      nrm = col_normal(k, ps, 0.5 * h);
      mask = vec4<f32>(0.0, clip.w, 1.0, 0.0);
      if (i32(k.m2.w) >= 0) {
        let b = obj_burn_at(k, slots[i32(k.m2.w)], ps + nrm * (0.75 * h));
        if (U.surf.w > 0.5) { mask.x = burn_char(b); }
        if (U.shad.z > 0.5) { surf.a = burn_wet(b); }
      }
      if (U.shad.y > 0.5) { mask.w = collider_soot(hit, ps + nrm * (0.75 * bh)); }
    } else {
      // the burnable floor is the bottom layer of the simulation grid
      let cg = vec3<i32>(floor((ps - U.org.xyz) / bh));
      if (cg.x >= 0 && cg.z >= 0 && cg.x < i32(U.bgrid.x) && cg.z < i32(U.bgrid.z)) {
        let b = textureLoad(burn, vec3<i32>(cg.x, 0, cg.z), 0);
        if (U.surf.w > 0.5) { mask.x = burn_char(b); }
        if (U.shad.z > 0.5) { surf.a = burn_wet(b); }
      }
      if (U.shad.y > 0.5) { mask.w = soot_at(vec3<f32>(ps.x, U.org.y + 0.5 * bh, ps.z)); }
      let gc = ro + rd * t_s;
      if (U.coal.x > 0.0 && gc.x >= 0.0 && gc.z >= 0.0 && gc.x <= U.n.x && gc.z <= U.n.z) {
        mask.x = max(mask.x, 0.6 * coal_cover(gc));  // charred under the coals
      }
    }
    if (U.surf.z > 0.5) { surf = vec4<f32>(surface_light(ps + nrm * (0.5 * h), nrm), surf.a); }
    if (U.hold2.w > 0.5 || U.lim.y > 0.5) { lamp_s = lamp_surface(ps + nrm * (0.5 * h), nrm); }
  }
  if (matte > 0.0) {
    // an object in the footage's matte is in front: the ground behind it is not seen, nor are embers
    surf = vec4<f32>(surf.rgb * (1.0 - matte), surf.a * (1.0 - matte));
    mask = vec4<f32>(mask.x * (1.0 - matte), mask.y * (1.0 - matte), max(mask.z, matte), mask.w * (1.0 - matte));
    lamp_s *= 1.0 - matte;
  }
  textureStore(out_surf, px, surf);
  textureStore(out_mask, px, mask);
  textureStore(out_lamp, px, vec4<f32>(lamp_s, 0.0));

  t_out = min(min(t_out, t_obj), t_hold);
  let deep_n = min(u32(U.deepp.x), 16u);
  let deep_base = (u32(px.y) * u32(U.deepp.y) + u32(px.x)) * deep_n * 2u;
  let zk = dot(rd_w, U.fwd.xyz) / cells_per_m;   // grid cells along the ray -> metres along the view axis
  var dsum: array<vec4<f32>, 16>;   // per bin: colour and opacity it adds to the pixel (all passes so far)
  var dz: array<vec2<f32>, 16>;     // per bin: front and back depth (m)
  var dcount = 0u;
  if (deep_n > 0u) {
    if (U.deepp.w > 0.5) {
      for (var k = 0u; k < deep_n; k++) {
        dsum[k] = deep[deep_base + 2u * k];
        dz[k] = deep[deep_base + 2u * k + 1u].xy;
      }
      dcount = u32(deep[deep_base + 1u].z);
    } else {
      for (var k = 0u; k < deep_n; k++) {
        deep[deep_base + 2u * k] = vec4<f32>(0.0);
        deep[deep_base + 2u * k + 1u] = vec4<f32>(0.0);
      }
    }
  }
  let defining = dcount == 0u;   // this pass places the bins (nothing earlier saw anything here)
  if (U.lim.x > 0.5) {
    // only the fire in front of the liquid: the liquid layer already shows the fire behind it
    let lq = textureLoad(limit, px, 0);
    if (lq.w > 0.5 && lq.y > 0.0) { t_out = min(t_out, lq.y * cells_per_m); }
  }
  if (t_out <= t_in) {
    textureStore(out_beauty, px, vec4<f32>(0.0));
    textureStore(out_emit, px, vec4<f32>(0.0));
    textureStore(out_aux, px, vec4<f32>(0.0));
    return;
  }

  // -- the volume ---------------------------------------------------------------------------------
  let step = L.misc.z;
  let jit = fract(ign(vec2<f32>(px)) + U.frame.x * 0.61803398875);
  var t = t_in + step * jit;
  let ds = step * h;
  let cos_sun = dot(rd_w, L.sundir.xyz);
  let phase = hg(cos_sun, L.amb.w);
  let lscale = U.ln.xyz / n;
  let fade_cells = max(L.misc.y, 1e-3);
  let shutter = U.frame.y * (fract(ign(vec2<f32>(px) + vec2<f32>(17.0, 59.0)) + U.frame.x * 0.7548776662) - 0.5);
  let rise = vec3<f32>(0.0, -L.misc.w * L.detail.w * L.detail.y, 0.0);
  let max_steps = i32(U.frame.w);
  let vk = max(U.org.w, 1.0);

  var tr = 1.0;   // transmittance for light transport
  var tra = 1.0;  // transmittance of the background (alpha): flames occlude only partly
  var col = vec3<f32>(0.0);
  var emit = vec3<f32>(0.0);
  var flame_a = 0.0;
  var heat = 0.0;
  var dsum_depth = 0.0;
  var dw = 0.0;
  var kmax = 0.0;
  // the deep bin being gathered (when placing bins): where it started, and the colour and alpha then
  var dk = 0u;
  var d_t0 = t;
  var d_col = vec3<f32>(0.0);
  var d_tra = 1.0;

  let keep = 1.0 - matte;   // the fire behind an object in the footage's matte is hidden
  for (var i = 0; i < max_steps; i++) {
    if (t >= t_out || (tr < 0.002 && tra < 0.002)) { break; }
    let p = ro + rd * t;
    let col_prev = col;
    let tra_prev = tra;
    let t_prev = t;
    let pl = p * lscale;
    let l1 = samp_c(L1, lin, pl, U.ln.xyz);
    if (l1.y < 1e-5) {
      t += step * 3.0;
      continue;
    }
    var q = p;
    if (U.frame.z > 0.5) {
      q -= vel_at(vel, lin, q / vk, n / vk) * (shutter / h);
    }
    var dmod = 1.0;
    if (L.detail.x > 0.0) {
      let nz = textureSampleLevel(noise, rep, q * h * L.detail.y + rise, 0.0);
      q += (nz.yzw - vec3<f32>(0.5)) * L.detail.z;
      dmod = max(1.0 + L.detail.x * (nz.x - 0.5) * 3.0, 0.0);
    }
    let s = samp_c(scal, lin, q, n);

    var edge = min(min(p.x, n.x - p.x), min(p.z, n.z - p.z));
    edge = min(edge, n.y - p.y);
    if (U.ln.w < 0.5) { edge = min(edge, p.y); }
    let fade = smoothstep(0.0, fade_cells, edge);

    var ax = vec4<f32>(0.0);
    if (L.air.z > 0.5) { ax = samp_c(aux, lin, q, n); }
    var ch = vec4<f32>(0.0);
    if (L.air.w > 0.5) { ch = samp_c(chem, lin, q, n); }

    let dens = smoke_extinction(s, L) * dmod * fade;
    let wet = steam_extinction(s, ax, L) * dmod * fade;
    let sf = flame_sigma(s, L) * fade;
    var cl = vec2<f32>(0.0);
    if (U.coal.x > 0.0 && p.y * h < U.coal.w) { cl = coal_at(p, L); }
    let sig = dens + wet + sf + cl.x;
    let e = emission_k(s, ch, sf, dens, L) + coal_emission(cl, L);
    let l0 = samp_c(L0, lin, pl, U.ln.xyz);
    var lin_in = L.amb.rgb * l1.x + L.sun.rgb * (l0.a * phase) + l0.rgb * L.sun.w;
    // the lights in the set: their light here, exactly, shadowed by the smoke on the way (from the light
    // volume) and scattered toward the camera by the phase function (a spot's beam glows brightest
    // looking into it); each multiple-scattering octave sees them as it sees the key light
    var lamp_o = array<vec3<f32>, 3>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    let n_lamps = i32(U.hold2.w + 0.5);
    let scatters = dens + wet > 1e-5;
    if (n_lamps > 0 && scatters) {
      let pm = U.org.xyz + p * h;
      let t0 = lamp_tr4(pl, 0.0);
      var t1 = vec4<f32>(1.0);
      if (n_lamps > 4) { t1 = lamp_tr4(pl, 1.0); }
      for (var li = 0; li < n_lamps; li++) {
        let Lm = lamps[li];
        let la = lamp_at(li, pm);
        if (la.f <= 0.0) { continue; }
        let ltr = pick4(select(t0, t1, li >= 4), li & 3);
        let pw = Lm.c.rgb * la.f;
        let cs = dot(rd, la.dir);
        lin_in += pw * (ltr * hg(cs, L.amb.w));
        if (L.ms.x > 0.0) {
          var b_l = 0.5;
          var c_l = 0.5;
          for (var o = 0; o < 3; o++) {
            lamp_o[o] += pw * (pow(ltr, b_l) * hg(cs, L.amb.w * c_l));
            b_l *= 0.5;
            c_l *= 0.5;
          }
        }
      }
    }
    let scat = dens * L.smoke.rgb + wet * L.steam.rgb;
    if (L.ms.x > 0.0 && scatters) {
      // multiple scattering (after Wrenninge): each further bounce keeps the albedo's share of the
      // light, sees the medium thinner and scatters more evenly, so pale smoke and steam stay bright inside
      let albedo = scat / (dens + wet);
      var a_o = albedo * L.ms.x;
      var b_o = 0.5;
      var c_o = 0.5;
      for (var o = 0; o < 3; o++) {
        lin_in += a_o * (L.amb.rgb * pow(l1.x, b_o) + L.sun.rgb * (pow(l0.a, b_o) * hg(cos_sun, L.amb.w * c_o))
                         + l0.rgb * L.sun.w + lamp_o[o]);
        a_o *= albedo * L.ms.x;
        b_o *= 0.5;
        c_o *= 0.5;
      }
    }
    let a = exp(-sig * ds);
    let w = select(ds, (1.0 - a) / sig, sig > 1e-6);
    col += tr * (e + (scat + vec3<f32>(COAL_ALBEDO * cl.x)) * lin_in) * w;
    emit += tr * e * w;
    let fa = 1.0 - exp(-sf * ds);
    flame_a += tr * fa * (1.0 - flame_a);
    heat += max(s.x, 0.0) * fade * ds;
    if (luma(e) > 1e-4) { kmax = max(kmax, kelvin(s.x, L)); }
    let dop = tr * (1.0 - a);
    dsum_depth += dop * t;
    dw += dop;
    tr *= a;
    tra *= exp(-(dens + wet + sf * L.fire2.w + cl.x) * ds);
    t += step;
    if (deep_n > 0u) {
      if (defining) {
        if (dk + 1u < deep_n) {
          // split off a bin once enough opacity (or light) has built up since the last one; a bin
          // starts where something is (empty air before it is skipped)
          let seg_a = 1.0 - tra / max(d_tra, 1e-8);
          let seg_c = (col - d_col) / max(d_tra, 1e-8);
          if (seg_a < 1e-6 && luma(seg_c) < 1e-7) {
            d_t0 = t;
          } else if (seg_a > U.deepp.z || luma(seg_c) > 4.0 * U.deepp.z) {
            dsum[dk] = vec4<f32>(col - d_col, d_tra - tra) * keep;
            dz[dk] = vec2<f32>(d_t0 * zk + near, t * zk + near);
            dk += 1u;
            d_t0 = t;
            d_col = col;
            d_tra = tra;
          }
        }
      } else {
        // add this step into the bin at its depth; where no bin reaches, open one (while there is room),
        // else stretch the nearest
        let dc = col - col_prev;
        let da = tra_prev - tra;
        if (da > 0.0 || luma(dc) > 0.0) {
          let z0 = t_prev * zk + near;
          let z1 = t * zk + near;
          let tol = 1.5 * step * zk;
          var j = -1;
          var near_j = 0u;
          var near_d = 1.0e30;
          for (var m = 0u; m < dcount; m++) {
            if (z0 >= dz[m].x - tol && z0 <= dz[m].y + tol) { j = i32(m); break; }
            let gap = max(dz[m].x - z0, z0 - dz[m].y);
            if (gap < near_d) { near_d = gap; near_j = m; }
          }
          if (j < 0 && dcount < deep_n) {
            j = i32(dcount);
            dsum[dcount] = vec4<f32>(0.0);
            dz[dcount] = vec2<f32>(z0, z1);
            dcount += 1u;
          }
          if (j < 0) { j = i32(near_j); }
          dsum[j] += vec4<f32>(dc, da) * keep;
          dz[j] = vec2<f32>(min(dz[j].x, z0), max(dz[j].y, z1));
        }
      }
    }
  }
  if (deep_n > 0u) {
    if (defining) {
      let seg_a = 1.0 - tra / max(d_tra, 1e-8);
      let seg_c = (col - d_col) / max(d_tra, 1e-8);
      if (seg_a > 1e-5 || luma(seg_c) > 1e-6) {
        dsum[dk] = vec4<f32>(col - d_col, d_tra - tra) * keep;
        dz[dk] = vec2<f32>(d_t0 * zk + near, min(t, t_out) * zk + near);
        dk += 1u;
      }
      dcount = dk;
    }
    for (var k = 0u; k < deep_n; k++) {
      deep[deep_base + 2u * k] = dsum[k];
      deep[deep_base + 2u * k + 1u] = vec4<f32>(dz[k], select(0.0, f32(dcount), k == 0u), 0.0);
    }
  }

  let depth = select(0.0, dsum_depth / max(dw, 1e-6) / cells_per_m, dw > 1e-4);
  textureStore(out_beauty, px, vec4<f32>(col, 1.0 - tra) * keep);
  textureStore(out_emit, px, vec4<f32>(emit, clamp(flame_a, 0.0, 1.0)) * keep);
  textureStore(out_aux, px, vec4<f32>(heat * keep, depth, kmax * 0.001, (1.0 - tra) * keep));
}
