// Fabric, drawn before the fire is marched (the march stops at it, and cloth_merge.wgsl puts it under
// the fire). Shaded as cloth, in the same light as the smoke: the key light (shadowed by the smoke,
// from the light volume), the sky, the fire (its short list of point lights), and the lights in the
// set (shadowed by the smoke). Cloth is:
// - diffuse, with the soft sheen of fibres at grazing angles (the Charlie sheen of Estevez and Kulla
//   2017), and, for silk, satin and nylon, a highlight stretched along the threads (anisotropic GGX);
// - thin: light from behind comes through it, tinted by its colour (a curtain glows with the fire
//   behind it);
// - woven: its threads are drawn where they are big enough on screen (plain, twill, pile, ripstop);
// - burning as real cloth does, in sharp, ragged bands (burn_bands): toasted brown ahead of the flame,
//   a thin bright line of glowing fibre where it is catching, black char behind with embers
//   smouldering in it, greying to ash, and a dull glowing rim where it crumbles into a hole.
//
// The bands follow the burn coordinate (vs): below 0 how far a spot is from catching (by its heat),
// 0 where it catches, 1 where it has burnt through. Interpolated across a triangle its contours
// cross the triangle where the flame front really is, so the bands are as sharp as the pixels
// however coarse the cloth, and the holes open along curves, not along the triangles' edges.
//
// Outputs: col = colour (premultiplied, alpha 1); aux = view depth (clip w), distance along the ray
// from where the march starts (m), _, coverage (the march's `limit`); glow = its own light (emission).
//!include common.wgsl
//!include shade.wgsl
//!include cloth_common.wgsl
//!include lume_light.wgsl
//!include lamp_shape.wgsl

struct DrawMat {
  c0: vec4<f32>,   // colour (linear), sheen strength
  c1: vec4<f32>,   // sheen roughness, highlight strength, highlight roughness along and across the threads
  c2: vec4<f32>,   // translucency, thread spacing (m), weave (0 none, 1 plain, 2 twill, 3 pile, 4 ripstop), weave depth
  c3: vec4<f32>,   // burning temperature (K), melts (1/0), ignition temperature (K), share of a soaking it keeps drained
};

struct Lamp { p: vec4<f32>, c: vec4<f32>, d: vec4<f32>, e: vec4<f32> };

struct Params {
  vp: mat4x4<f32>,     // world -> clip
  l2w: mat4x4<f32>,    // fire-local -> world
  res: vec4<f32>,      // width, height, sub-pixel jitter (px)
  eye: vec4<f32>,      // camera (fire-local), distance from the camera to where the march starts, along the view axis
  fwd: vec4<f32>,      // camera forward (fire-local), shutter offset (s: motion blur)
  sund: vec4<f32>,     // toward the key light (fire-local), strength of the fire's light (Lighting, as on the smoke)
  lg: vec4<f32>,       // simulation grid corner (fire-local m), its cell size (m)
  ln: vec4<f32>,       // light-volume dims, lights in the set (count)
  lsc: vec4<f32>,      // light-volume cells per simulation cell (xyz), light volume present (1/0; 2: with the
                       // cloth's and the objects' extinction in L1.z, L1.w)
  look: Look,
  xp: vec4<f32>,       // its reflected light times this (the liquid render's exposure; 1 with the fire), lit by Lume (1/0)
};

@group(0) @binding(0) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> N: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> S: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> T: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> UV: array<vec4<f32>>;   // weave coordinates (m), fabric
@group(0) @binding(5) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> DM: array<DrawMat>;
@group(0) @binding(7) var L0: texture_3d<f32>;
@group(0) @binding(8) var L1: texture_3d<f32>;
@group(0) @binding(9) var bb: texture_2d<f32>;
@group(0) @binding(10) var lin: sampler;
@group(0) @binding(11) var<storage, read> lights: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read> light_count: array<u32>;
@group(0) @binding(13) var<storage, read> lamps: array<Lamp>;
@group(0) @binding(14) var LT: texture_3d<f32>;
@group(0) @binding(15) var E: texture_3d<f32>;   // the light volume's emission and extinction (smoke and cloth)
@group(0) @binding(16) var<storage, read> PW: array<vec4<f32>>;   // w: how wet it is (0 dry .. 1 soaked)
@group(0) @binding(17) var LV: texture_3d<f32>;    // Lume's light (lume_light.wgsl)
@group(0) @binding(18) var LVD: texture_3d<f32>;
@group(0) @binding(19) var<storage, read> HL: array<vec4<f32>>;   // bullet holes: [0].x how many; u, v (m), radius, fabric
@group(1) @binding(0) var<uniform> U: Params;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) p: vec3<f32>,
  @location(1) n: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) @interpolate(flat) fab: u32,
  @location(4) st: vec4<f32>,
  @location(5) cw: f32,
  @location(6) bc: f32,
  @location(7) wet: f32,
};

// A vertex's burn coordinate: below 0 it is heating toward its ignition temperature (-1 cold), from 0
// as it catches to 1 as it burns through (its burn progress).
fn burn_coord(st: vec4<f32>, dm: DrawMat) -> f32 {
  if (st.y > 0.0 || st.z > 0.5) { return st.y; }
  let ign = max(dm.c3.z, 400.0);
  return clamp((st.x - ign) / (ign - 300.0), -1.0, 0.0);
}

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VOut {
  var o: VOut;
  let t = T[vid / 3u];
  let c = vid % 3u;
  let i = select(select(t.z, t.y, c == 1u), t.x, c == 0u);
  // burnt away: a triangle goes when all of it has, or when what has burnt off it has drifted away
  // as ash (it would stretch across the gap); otherwise its hole opens inside it (the fragments)
  let g3 = vec3<bool>(gone(S[t.x]), gone(S[t.y]), gone(S[t.z]));
  // (torn away, cloth_tear.wgsl: a rip opens at once, with no burnt edge)
  var cut = all(g3) || S[t.x].z < -0.5 || S[t.y].z < -0.5 || S[t.z].z < -0.5;
  if (any(g3) && !cut) {
    let e0 = length(X[t.x].xyz - X[t.y].xyz) / max(length(UV[t.x].xy - UV[t.y].xy), 1e-6);
    let e1 = length(X[t.y].xyz - X[t.z].xyz) / max(length(UV[t.y].xy - UV[t.z].xy), 1e-6);
    let e2 = length(X[t.z].xyz - X[t.x].xyz) / max(length(UV[t.z].xy - UV[t.x].xy), 1e-6);
    cut = max(e0, max(e1, e2)) > 1.6;
  }
  if (cut) {
    o.pos = vec4<f32>(0.0, 0.0, -2.0, 1.0);   // outside the view
    return o;
  }
  let p = X[i].xyz + V[i].xyz * U.fwd.w;
  var clip = U.vp * (U.l2w * vec4<f32>(p, 1.0));
  clip.x -= 2.0 * U.res.z / U.res.x * clip.w;
  clip.y += 2.0 * U.res.w / U.res.y * clip.w;
  o.pos = clip;
  o.p = p;
  o.n = N[i].xyz;
  o.uv = UV[i].xy;
  o.fab = u32(UV[i].z + 0.5);
  o.st = S[i];
  // (cloth burnt away still counts as burning for the line where the char meets fresh cloth: the
  // front creeps on from the edge of the hole)
  if (gone(S[i])) { o.st.z = 1.0; }
  o.cw = clip.w;
  o.bc = burn_coord(S[i], DM[o.fab]);
  o.wet = clamp(PW[i].w, 0.0, 1.0);
  return o;
}

struct FOut {
  @location(0) col: vec4<f32>,
  @location(1) aux: vec4<f32>,
  @location(2) glow: vec4<f32>,
};

fn hash2(p: vec2<f32>) -> f32 {
  return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn vnoise(p: vec2<f32>) -> f32 {
  let i = floor(p);
  let f = p - i;
  let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash2(i), hash2(i + vec2<f32>(1.0, 0.0)), u.x), mix(hash2(i + vec2<f32>(0.0, 1.0)), hash2(i + vec2<f32>(1.0, 1.0)), u.x), u.y);
}

// The weave's height at uv (m) for a pattern: which thread is on top, and its rounded profile.
// Returns (height -0.5..0.5, 1 where the warp is on top).
fn weave(uv: vec2<f32>, pitch: f32, kind: f32) -> vec2<f32> {
  let g = uv / pitch;
  let cell = floor(g);
  let f = g - cell;
  let k = i32(kind + 0.5);
  var warp_up = (i32(cell.x) + i32(cell.y)) % 2 == 0;
  if (k == 2) { warp_up = ((i32(cell.x) + i32(cell.y) + 1000) % 3) != 0; }   // twill: diagonal ribs
  let pu = sin(3.14159265 * f.x);   // across a warp thread
  let pv = sin(3.14159265 * f.y);
  var h = select(pv - 0.5, pu - 0.5, warp_up) * 0.5 + select(-0.25, 0.25, warp_up);
  if (k == 4) {
    // ripstop: a heavier thread every eight
    let r = min(abs(fract(g.x / 8.0) - 0.5), abs(fract(g.y / 8.0) - 0.5));
    h += 0.4 * (1.0 - smoothstep(0.0, 0.06, r));
  }
  return vec2<f32>(h, select(0.0, 1.0, warp_up));
}

// Charlie sheen (Estevez and Kulla 2017) with Neubelt's visibility.
fn sheen(nh: f32, nl: f32, nv: f32, rough: f32) -> f32 {
  let a = max(rough * rough, 0.02);
  let s2 = max(1.0 - nh * nh, 0.0);
  let d = (2.0 + 1.0 / a) * pow(s2, 0.5 / a) / (2.0 * 3.14159265);
  return d / (4.0 * max(nl + nv - nl * nv, 1e-3));
}

// Anisotropic GGX highlight along tangent t (roughness au along it, av across).
fn aniso_spec(n: vec3<f32>, t: vec3<f32>, l: vec3<f32>, v: vec3<f32>, au: f32, av: f32) -> f32 {
  let h = normalize(l + v);
  let b = cross(n, t);
  let th = dot(t, h) / au;
  let bh = dot(b, h) / av;
  let nh = dot(n, h);
  let d = 1.0 / (3.14159265 * au * av * pow(th * th + bh * bh + nh * nh, 2.0));
  let nl = max(dot(n, l), 1e-3);
  let nv = max(dot(n, v), 1e-3);
  let f = 0.04 + 0.96 * pow(1.0 - max(dot(h, v), 0.0), 5.0);
  return d * f / (4.0 * nl * nv) * min(1.0, 2.0 * nh * min(nl, nv) / max(dot(v, h), 1e-3));
}

// Light from one direction l (unit) with irradiance e (at normal incidence) on the cloth: its colour
// seen from v, lit from the front or showing through from behind.
fn light_cloth(e: vec3<f32>, l: vec3<f32>, n: vec3<f32>, v: vec3<f32>, t: vec3<f32>, alb: vec3<f32>, dm: DrawMat,
               sheen_on: f32, spec_on: f32) -> vec3<f32> {
  let nl = dot(n, l);
  let nv = max(dot(n, v), 1e-3);
  if (nl <= 0.0) {
    // from behind: diffuse transmission through the thin sheet
    return e * alb * (dm.c2.x * (-nl) / 3.14159265);
  }
  let h = normalize(l + v);
  var c = alb * (1.0 - dm.c2.x) / 3.14159265;
  c += alb * dm.c0.w * sheen_on * sheen(max(dot(n, h), 0.0), nl, nv, dm.c1.x);
  if (dm.c1.y > 0.0) { c += vec3<f32>(dm.c1.y * spec_on * aniso_spec(n, t, l, v, dm.c1.z, dm.c1.w)); }
  return e * c * nl;
}

fn lamp_tr(pl: vec3<f32>, i: i32) -> f32 {
  let ln = U.ln.xyz;
  let blocks = select(1.0, 2.0, U.ln.w > 4.5);
  let q = clamp(pl, vec3<f32>(0.5), ln - vec3<f32>(0.5));
  let t = textureSampleLevel(LT, lin, vec3<f32>(q.x / ln.x, q.y / ln.y, (q.z + f32(i / 4) * ln.z) / (ln.z * blocks)), 0.0);
  let j = i & 3;
  return select(select(t.w, t.z, j == 2), select(t.y, t.x, j == 0), j < 2);
}

fn in_light(pl: vec3<f32>) -> bool {
  return U.lsc.w > 0.5 && all(pl >= vec3<f32>(0.0)) && all(pl <= U.ln.xyz);
}

// Where to look up the light volume for light from direction l on the cloth at light-volume point pl
// (normal n): the cloth's own shadow is spread over a couple of light cells round its sheet, so off
// the sheet on the side the light comes from, and a little toward the light.
fn off_sheet(pl: vec3<f32>, n: vec3<f32>, l: vec3<f32>) -> vec3<f32> {
  return pl + n * select(-1.5, 1.5, dot(n, l) >= 0.0) + l * 0.75;
}

// The objects' shadow (L1.w) on light from direction l reaching light-volume point pl, within `reach`
// light cells: a light cell a step along the major axis.
fn obj_shadow(pl: vec3<f32>, l: vec3<f32>, reach: f32) -> f32 {
  if (U.lsc.w < 1.5) { return 1.0; }
  let st = l / max(abs(l.x), max(abs(l.y), max(abs(l.z), 1e-6)));
  let len = length(st);
  var od = 0.0;
  var q = pl;
  var t = 0.0;
  for (var k = 0; k < 128; k++) {
    q += st;
    t += len;
    if (t > reach || any(q < vec3<f32>(0.0)) || any(q > U.ln.xyz)) { break; }
    od += samp_c(L1, lin, q, U.ln.xyz).w;
  }
  return exp(-od * length(st * U.lg.w / U.lsc.xyz));
}

// How much of the fire's light gets to the cloth at fire-local p (normal n) from the fire's centre pc
// (size spread): the smoke, other cloth (the light volume's extinction, E.a) and the objects (L1.w) in
// between, marched from just off the sheet on the fire's side.
fn fire_shadow(p: vec3<f32>, n: vec3<f32>, pc: vec3<f32>, spread: f32) -> f32 {
  if (U.lsc.w < 0.5) { return 1.0; }
  let d = pc - p;
  let dist = length(d);
  if (dist < 1e-4) { return 1.0; }
  let dir = d / dist;
  let lc = U.lg.w / U.lsc.xyz;   // metres per light cell, per axis
  let cell = max(lc.x, max(lc.y, lc.z));
  let p0 = p + (n * select(-1.5, 1.5, dot(n, dir) >= 0.0) + dir * 0.75) * cell;
  let reach = max(dist - 0.5 * spread - 0.75 * cell, 0.0);
  let steps = clamp(i32(ceil(reach / (0.75 * cell))), 1, 48);
  let dt = reach / f32(steps);
  let ln = U.ln.xyz;
  var od = 0.0;
  for (var k = 0; k < steps; k++) {
    let q = (p0 + dir * ((f32(k) + 0.5) * dt) - U.lg.xyz) / lc;
    if (any(q < vec3<f32>(0.0)) || any(q > ln)) { continue; }
    od += samp_c(E, lin, q, ln).a + samp_c(L1, lin, q, ln).w;
  }
  return exp(-od * dt);
}

// A glowing solid at temperature k (K), in the fire's exposure (as bright as a thick flame that hot).
fn glow_at(k: f32, L: Look) -> vec3<f32> {
  let bbv = bb_lookup(k);
  return bbv.rgb * min(pow(10.0, (bbv.a - L.misc.x) * L.fire.w), 1.0e4) * L.fire2.x;
}

// The light burning cloth gives off itself, at burn coordinate b (ragged), burning share z (0..1), weave
// coordinates uv (m):
// - where it catches, a thin line of fibre glowing at its burning temperature, flickering;
// - behind it, in the black char, sparse embers smouldering out as the char cools;
// - where the char crumbles into a hole, a dull glowing rim.
fn burn_glow(b: f32, z: f32, uv: vec2<f32>, dm: DrawMat, L: Look) -> vec3<f32> {
  if (b < -0.03) { return vec3<f32>(0.0); }
  let kb = max(dm.c3.x, 700.0);
  let t = L.misc.w;
  let flick = 0.9 + 0.2 * vnoise(uv * 25.0 + vec2<f32>(t * 3.1, t * 1.7));
  var g = vec3<f32>(0.0);
  // the catching line: brightest a hair behind where it catches, gone a little further back
  let line = smoothstep(-0.03, 0.0, b) * (1.0 - smoothstep(0.015, 0.06, b)) * smoothstep(0.05, 0.4, z);
  if (line > 0.0) { g += glow_at(kb * flick, L) * (4.0 * line); }
  // embers in the char: sparse, finer where they are dying out
  let fresh = 1.0 - smoothstep(0.05, 0.6, b);
  let speck = smoothstep(0.62, 0.9, vnoise(uv * 160.0 + vec2<f32>(t * 0.4, 0.0)) * vnoise(uv * 45.0 + vec2<f32>(5.0, t * 0.25)) * 1.6);
  if (b > 0.02 && fresh > 0.0) { g += glow_at(kb * 0.8 * flick, L) * (2.0 * speck * fresh); }
  // the crumbling rim of a hole
  let rim = smoothstep(0.7, 0.86, b) * (1.0 - smoothstep(0.88, 0.9, b));
  if (rim > 0.0) {
    let rk = kb * mix(0.72, 0.9, smoothstep(0.82, 0.89, b)) * (0.9 + 0.2 * vnoise(uv * 70.0 + vec2<f32>(t, 0.0)));
    g += glow_at(rk, L) * (4.0 * rim);
  }
  return g;
}

// ---- ash and embers: cloth that has burnt away -------------------------------------------------------
// Each burnt-off vertex is a few flakes of char (cloth_predict.wgsl carries it on the air): ragged,
// a few millimetres to a couple of centimetres, tumbling as they go. Fresh from the flame they glow
// at their temperature (cloth_finish.wgsl cools them), flickering as they turn edge-on; then they
// are black char, and grey ash as they age, crumbling away after FLAKE_LIFE seconds. A synthetic's
// drops of melt are dark and do not glow.
const FLAKES: u32 = 3u;           // flakes drawn per burnt-off vertex (matches cloth.FLAKES)
const FLAKE_LIFE: f32 = 7.0;      // s

struct FlakeOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) q: vec2<f32>,                       // across the flake (-1..1)
  @location(1) @interpolate(flat) k: f32,          // temperature (K)
  @location(2) @interpolate(flat) fab: u32,
  @location(3) @interpolate(flat) age: f32,        // s
  @location(4) @interpolate(flat) seed: f32,
  @location(5) cw: f32,
  @location(6) p: vec3<f32>,
  @location(7) @interpolate(flat) face: f32,       // how squarely it faces the eye (0 edge-on .. 1)
};

fn flake_hash(n: u32) -> vec3<f32> {
  var s = n * 747796405u + 2891336453u;
  s = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  let a = (s >> 22u) ^ s;
  let b = a * 1664525u + 1013904223u;
  let c = b * 1664525u + 1013904223u;
  return vec3<f32>(f32(a), f32(b), f32(c)) * (1.0 / 4294967296.0);
}

@vertex
fn vs_flake(@builtin(vertex_index) vid: u32) -> FlakeOut {
  var o: FlakeOut;
  let fl = vid / 6u;
  let vi = fl / FLAKES;
  let st = S[vi];
  o.pos = vec4<f32>(0.0, 0.0, -2.0, 1.0);
  if (!gone(st) || st.w > FLAKE_LIFE) { return o; }
  let h0 = flake_hash(fl * 2654435761u + 17u);
  let h1 = flake_hash(fl * 2246822519u + 101u);
  let melts = DM[u32(UV[vi].z + 0.5)].c3.y > 0.5;
  // its size, shrinking as it crumbles at the end; scattered over the patch of cloth it came from
  var size = select(mix(0.003, 0.012, h0.x * h0.x), mix(0.002, 0.005, h0.x), melts);
  size *= 1.0 - smoothstep(0.7 * FLAKE_LIFE, FLAKE_LIFE, st.w);
  let t = U.look.misc.w;
  let pc = X[vi].xyz + V[vi].xyz * U.fwd.w + (h1 - vec3<f32>(0.5)) * 0.02;
  // tumbling: a plane turning about a random axis
  let ax = normalize(flake_hash(fl * 3266489917u + 7u) - vec3<f32>(0.5) + vec3<f32>(1e-3, 0.0, 0.0));
  let ang = t * mix(2.0, 9.0, h0.y) + 6.2831853 * h0.z;
  var a = normalize(cross(ax, vec3<f32>(0.3, 1.0, 0.2)));
  var b = cross(ax, a);
  let ca = cos(ang);
  let sa = sin(ang);
  let a2 = a * ca + b * sa;
  let b2 = ax * 0.6 + cross(ax, a2) * 0.8;
  var corners = array<vec2<f32>, 6>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
                                    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0));
  let corner = corners[vid % 6u];
  let p = pc + (a2 * corner.x + b2 * corner.y) * size;
  var clip = U.vp * (U.l2w * vec4<f32>(p, 1.0));
  clip.x -= 2.0 * U.res.z / U.res.x * clip.w;
  clip.y += 2.0 * U.res.w / U.res.y * clip.w;
  o.pos = clip;
  o.q = corner;
  o.k = select(st.x, 0.0, melts);
  o.fab = u32(UV[vi].z + 0.5);
  o.age = st.w;
  o.seed = h1.z * 100.0;
  o.cw = clip.w;
  o.p = p;
  let nrm = normalize(cross(a2, b2));
  o.face = abs(dot(nrm, normalize(U.eye.xyz - pc)));
  return o;
}

@fragment
fn fs_flake(i: FlakeOut) -> FOut {
  // a ragged outline
  let ang = atan2(i.q.y, i.q.x);
  let r = length(i.q);
  let edge = 0.55 + 0.3 * vnoise(vec2<f32>(ang * 1.6 + i.seed, i.seed * 0.37)) + 0.15 * vnoise(vec2<f32>(ang * 5.0, i.seed));
  if (r > edge) { discard; }
  let L = U.look;
  let dm = DM[i.fab];
  let melts = dm.c3.y > 0.5;
  // black char, greying to ash as it ages (a synthetic's melt stays dark brown)
  let ashy = smoothstep(1.0, 4.0, i.age) * (0.5 + 0.5 * vnoise(i.q * 3.0 + vec2<f32>(i.seed, 0.0)));
  var alb = select(mix(vec3<f32>(0.03, 0.028, 0.026), vec3<f32>(0.32, 0.31, 0.29), ashy), vec3<f32>(0.06, 0.04, 0.025), melts);
  // lit by the sky and the fire, from either side
  var e = L.amb.rgb * 0.7;
  let nf = light_count[0];
  for (var k = 0u; k < nf; k++) {
    let lp = lights[2u * k];
    let d = lp.xyz - i.p;
    e += lights[2u * k + 1u].rgb * (U.sund.w / (dot(d, d) + lp.w * lp.w)) * 0.5;
  }
  var c = alb * e / 3.14159265;
  // its own glow while it is hot: brightest face-on, a rim of it edge-on
  var glow = vec3<f32>(0.0);
  if (i.k > 720.0) {
    let hot = smoothstep(720.0, 900.0, i.k);
    let spot = 0.6 + 0.4 * vnoise(i.q * 4.0 + vec2<f32>(i.seed, L.misc.w * 2.0));
    glow = glow_at(i.k, L) * (2.0 * hot * spot * (0.35 + 0.65 * i.face));
  }
  var o: FOut;
  o.col = vec4<f32>(c + glow, 1.0);
  let v = normalize(U.eye.xyz - i.p);
  let dist = max(length(i.p - U.eye.xyz) - U.eye.w / max(dot(-v, U.fwd.xyz), 1e-3), 0.0);
  o.aux = vec4<f32>(i.cw, dist, 0.0, 1.0);
  o.glow = vec4<f32>(glow, 1.0);
  return o;
}

// ---- drops of water dripping off wet cloth (cloth_drip.wgsl, cloth_drops.wgsl) ----------------------
// Each is a little ball of water: it mirrors the sky (most round its rim) and the lights as pin-point
// glints, and, as a ball lens, shows the world behind it upside down: the bright sky in its lower half.
// A drop between the eye and a light flashes with the light it focuses. Its own bind group (binding
// 17 only; the cloth's vertex stage has no storage buffers to spare).
struct DropsR {
  head: vec4<u32>,
  d: array<vec4<f32>>,   // per drop: position (fire-local m), life left (s); velocity (m/s), radius (m)
};
@group(0) @binding(17) var<storage, read> DRW: DropsR;

struct DropOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) q: vec2<f32>,                         // across the drop (-1..1)
  @location(1) p: vec3<f32>,
  @location(2) cw: f32,
  @location(3) @interpolate(flat) ax: vec3<f32>,     // its billboard's axes (fire-local)
  @location(4) @interpolate(flat) ay: vec3<f32>,
};

@vertex
fn vs_drop(@builtin(vertex_index) vid: u32) -> DropOut {
  var o: DropOut;
  o.pos = vec4<f32>(0.0, 0.0, -2.0, 1.0);
  let k = vid / 6u;
  let p4 = DRW.d[2u * k];
  if (p4.w <= 0.0) { return o; }
  let v4 = DRW.d[2u * k + 1u];
  let pc = p4.xyz + v4.xyz * U.fwd.w;
  let toe = normalize(U.eye.xyz - pc);
  var side = cross(vec3<f32>(0.0, 1.0, 0.0), toe);
  if (dot(side, side) < 1e-6) { side = vec3<f32>(1.0, 0.0, 0.0); }
  let ax = normalize(side);
  let ay = cross(toe, ax);
  var corners = array<vec2<f32>, 6>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
                                    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0));
  let corner = corners[vid % 6u];
  let p = pc + (ax * corner.x + ay * corner.y) * v4.w;
  var clip = U.vp * (U.l2w * vec4<f32>(p, 1.0));
  clip.x -= 2.0 * U.res.z / U.res.x * clip.w;
  clip.y += 2.0 * U.res.w / U.res.y * clip.w;
  o.pos = clip;
  o.q = corner;
  o.p = p;
  o.cw = clip.w;
  o.ax = ax;
  o.ay = ay;
  return o;
}

@fragment
fn fs_drop(i: DropOut) -> FOut {
  let r2 = dot(i.q, i.q);
  if (r2 > 1.0) { discard; }
  let v = normalize(U.eye.xyz - i.p);
  let n = normalize(i.ax * i.q.x + i.ay * i.q.y + cross(i.ax, i.ay) * sqrt(1.0 - r2));
  let nv = max(dot(n, v), 0.0);
  let fr = 0.02 + 0.98 * pow(1.0 - nv, 5.0);
  let L = U.look;
  // the sky, mirrored, and seen through it upside down (bright below, the darker ground above); round its
  // rim the light it bends comes from far off to the side, and much of it is reflected inside: a dark ring
  let rd = reflect(-v, n);
  let rim = 1.0 - 0.55 * smoothstep(0.55, 1.0, r2);
  var c = L.amb.rgb * (fr * mix(0.25, 1.0, smoothstep(-0.2, 0.3, rd.y))
                       + (1.0 - fr) * rim * mix(0.45, 1.25, smoothstep(0.3, -0.5, n.y)));
  // the key light (in the smoke's shadow), the fire, the lights in the set: glints, and the light it
  // focuses toward the eye when lit from behind
  let pl = (i.p - U.lg.xyz) / U.lg.w * U.lsc.xyz;
  var sun_tr = 1.0;
  if (in_light(pl)) { sun_tr = samp_c(L0, lin, pl, U.ln.xyz).a; }
  let t = i.ax;
  var e = L.sun.rgb * sun_tr;
  var l = U.sund.xyz;
  c += e * (0.5 * aniso_spec(n, t, l, v, 0.03, 0.03) * max(dot(n, l), 0.0) + 0.6 * pow(max(dot(-v, l), 0.0), 40.0));
  let nf = light_count[0];
  for (var k = 0u; k < nf; k++) {
    let a = lights[2u * k];
    let d = a.xyz - i.p;
    let r2l = dot(d, d);
    e = lights[2u * k + 1u].rgb * (U.sund.w / (r2l + a.w * a.w));
    l = d * inverseSqrt(max(r2l, 1e-8));
    c += e * (0.5 * aniso_spec(n, t, l, v, 0.05, 0.05) * max(dot(n, l), 0.0) + 0.6 * pow(max(dot(-v, l), 0.0), 20.0));
  }
  for (var k = 0; k < i32(U.ln.w); k++) {
    let lm = lamps[k];
    let d = lm.p.xyz - i.p;
    let d2 = dot(d, d);
    l = d * inverseSqrt(max(d2, 1e-12));
    let f = lamp_shape(k, -l) / (d2 + lm.p.w * lm.p.w);
    if (f <= 0.0) { continue; }
    c += lm.c.rgb * f * (0.5 * aniso_spec(n, t, l, v, 0.03, 0.03) * max(dot(n, l), 0.0) + 0.6 * pow(max(dot(-v, l), 0.0), 40.0));
  }
  var o: FOut;
  o.col = vec4<f32>(c * U.xp.x, 1.0);
  let dist = max(length(i.p - U.eye.xyz) - U.eye.w / max(dot(-v, U.fwd.xyz), 1e-3), 0.0);
  o.aux = vec4<f32>(i.cw, dist, 0.0, 1.0);
  o.glow = vec4<f32>(0.0, 0.0, 0.0, 1.0);
  return o;
}

@fragment
fn fs(i: VOut) -> FOut {
  // wet: water in the weave darkens it (less light scatters back out of the fibres), lets more light
  // through (a soaked shirt shows what is behind it), mats the fibres' sheen and gives it the gloss of
  // the water's film
  let wet = i.wet;
  var dm = DM[i.fab];
  var film = 0.0;
  if (wet > 0.0) {
    dm.c0 = vec4<f32>(pow(max(dm.c0.rgb, vec3<f32>(1e-4)), vec3<f32>(1.0 + 1.2 * wet)), dm.c0.w * (1.0 - 0.7 * wet));
    dm.c1 = vec4<f32>(dm.c1.x, max(dm.c1.y, 0.35 * wet), mix(dm.c1.z, 0.15, wet), mix(dm.c1.w, 0.2, wet));
    dm.c2.x = mix(dm.c2.x, min(2.5 * dm.c2.x + 0.15, 0.85), wet);
    // dripping wet (water beyond what it keeps once drained stands in the weave): a smooth, bright film
    film = smoothstep(dm.c3.w, 1.0, wet) * select(0.0, 1.0, dm.c3.w > 0.0);
    dm.c1 = vec4<f32>(dm.c1.x, mix(dm.c1.y, 0.9, film), mix(dm.c1.z, 0.06, film), mix(dm.c1.w, 0.08, film));
    dm.c0 = vec4<f32>(dm.c0.rgb * (1.0 - 0.15 * film), dm.c0.w);
  }
  let L = U.look;
  let st = i.st;
  // the burn coordinate, made ragged at the scales burning cloth is (a few cm down to the threads)
  let rag = vnoise(i.uv * 60.0) * 0.6 + vnoise(i.uv * 220.0) * 0.4;
  let rg2 = 0.5 * vnoise(i.uv * 38.0) + 0.3 * vnoise(i.uv * 140.0 + vec2<f32>(7.3, 1.1)) + 0.2 * vnoise(i.uv * 520.0);
  let bcn = i.bc + 0.12 * (rg2 - 0.5);
  if (bcn > 0.9) { discard; }   // burnt through
  // bullet holes (bullet_media.py): through the weave, as fine as the pixels however coarse the cloth; their edge
  // ragged at the threads' scale, a few threads left across them, the weave round them pulled and darker
  var holed = 0.0;
  let nh = u32(HL[0].x + 0.5);
  let tp = max(DM[i.fab].c2.y, 2.0e-4);
  for (var k = 0u; k < nh; k++) {
    let hr = HL[1u + k];
    if (u32(hr.w + 0.5) != i.fab) { continue; }
    let dv = i.uv - hr.xy;
    let d = length(dv);
    if (d > 2.0 * hr.z) { continue; }
    let ang = atan2(dv.y, dv.x);
    let edge = hr.z * (1.0 + 0.3 * (vnoise(vec2<f32>(ang * 1.7, hr.x * 997.0)) - 0.5) + 0.35 * (vnoise(i.uv / tp) - 0.5));
    let th = fract(i.uv / tp);
    let left = select(0.0, 1.0, min(th.x, th.y) < 0.18 && hash2(floor(i.uv / tp) + hr.xy * 513.0) > 0.88 && d > 0.45 * edge);
    if (d < edge && left < 0.5) { discard; }
    holed = max(holed, 1.0 - smoothstep(edge, edge * 1.7, d));
  }
  let v = normalize(U.eye.xyz - i.p);
  var n = normalize(i.n);
  if (dot(n, v) < 0.0) { n = -n; }   // both sides of the sheet
  // tangent frame from the weave coordinates
  let dpx = dpdx(i.p);
  let dpy = dpdy(i.p);
  let dux = dpdx(i.uv);
  let duy = dpdy(i.uv);
  var t = dpx * duy.y - dpy * dux.y;
  t = normalize(t - n * dot(n, t) + vec3<f32>(1e-6, 0.0, 0.0));
  let b = cross(n, t);
  // the weave, faded out where its threads are smaller than a pixel
  var alb = dm.c0.rgb * (1.0 - 0.3 * holed);
  var nn = n;
  let pitch = max(dm.c2.y, 1e-5);
  let foot = max(length(dux) + length(duy), 1e-9) / pitch;
  let no_weave = dm.c2.z < 0.5 || (dm.c2.z > 2.5 && dm.c2.z < 3.5);   // none, or pile (velvet)
  let wv = (1.0 - smoothstep(0.35, 1.0, foot)) * select(1.0, 0.0, no_weave);
  if (wv > 0.0) {
    let e = pitch * 0.25;
    let h0 = weave(i.uv, pitch, dm.c2.z);
    let hu = weave(i.uv + vec2<f32>(e, 0.0), pitch, dm.c2.z).x - h0.x;
    let hv = weave(i.uv + vec2<f32>(0.0, e), pitch, dm.c2.z).x - h0.x;
    nn = normalize(n - (t * hu + b * hv) * (dm.c2.w * wv * 4.0));
    alb *= 1.0 + wv * (0.16 * h0.x + 0.05 * (h0.y - 0.5));
  }
  // slubs and uneven dye: a little variation over the cloth
  alb *= 0.94 + 0.12 * vnoise(i.uv * 7.0);
  // heat: toasted brown ahead of the flame (by its heat, and by how long it has been hot), black char
  // right behind where it catches, greying to ash before it crumbles away
  let melts = dm.c3.y > 0.5;
  // (cotton browns from about 130 C, well before it catches at 350 C: a halo round the char)
  let toast = max(clamp(st.w, 0.0, 1.0), pow(smoothstep(-0.75, -0.02, bcn), 1.5) * select(1.0, 0.0, bcn > 0.0) * (1.0 - smoothstep(0.0, 0.2, wet)));
  let brown = select(toast, 0.0, melts);
  alb = mix(alb, alb * vec3<f32>(0.5, 0.31, 0.14), brown * 0.85);
  let char = smoothstep(-0.01, 0.06, bcn);
  let char_col = select(vec3<f32>(0.022, 0.019, 0.017), vec3<f32>(0.05, 0.035, 0.02), melts) * (0.8 + 0.4 * vnoise(i.uv * 300.0));
  alb = mix(alb, char_col, char);
  let ash = smoothstep(0.5, 0.82, bcn) * smoothstep(0.35, 0.75, vnoise(i.uv * 30.0 + vec2<f32>(3.1, 9.4)));
  alb = mix(alb, vec3<f32>(0.3, 0.29, 0.27), ash * select(0.65, 0.0, melts));
  let sheen_on = 1.0 - char;
  let spec_on = 1.0 - char;

  // light
  var c = vec3<f32>(0.0);
  let lg = (i.p - U.lg.xyz) / U.lg.w;           // simulation cells
  let pl = lg * U.lsc.xyz;                       // light-volume cells
  // (the smoke's, other cloth's and the objects' shadows: the light volume, looked up off the sheet)
  var sun_tr = 1.0;
  var sky = 1.0;
  let ps = off_sheet(pl, n, U.sund.xyz);
  if (in_light(ps)) { sun_tr = samp_c(L0, lin, ps, U.ln.xyz).a * obj_shadow(ps, U.sund.xyz, 1.0e9); }
  let pk = pl + n * 1.5;   // the sky over the side seen (under a canopy, the canopy hides it)
  if (in_light(pk)) { sky = samp_c(L1, lin, pk, U.ln.xyz).x; }
  c += light_cloth(L.sun.rgb * sun_tr, U.sund.xyz, nn, v, t, alb, dm, sheen_on, spec_on);
  let sheen_k = 1.0 + 0.5 * dm.c0.w * sheen_on * pow(1.0 - max(dot(nn, v), 0.0), 3.0);
  let fres = film * spec_on * (0.02 + 0.98 * pow(1.0 - max(dot(nn, v), 0.0), 5.0));
  if (U.xp.y > 0.5) {
    // Lume's light, every way but the key light's and the lamps' beams (the fire's, the sky's, what the ground, the
    // objects and the smoke send on), on the side seen and through from behind; a film of water mirrors it
    let sim = U.ln.xyz / U.lsc.xyz;
    let e_f = lume_irradiance(LV, LVD, lg, sim, nn);
    let e_b = lume_irradiance(LV, LVD, lg, sim, -nn);
    c += alb * 0.31830988 * (e_f * (1.0 - 0.5 * dm.c2.x) + e_b * (0.5 * dm.c2.x)) * sheen_k;
    if (film > 0.0) { c += e_f * (0.31830988 * fres); }
  } else {
  // the sky: from above, dimmed where the smoke hides it; a little comes through from behind
  c += L.amb.rgb * sky * alb * ((0.6 + 0.4 * nn.y) * (1.0 - 0.5 * dm.c2.x) + 0.15 * dm.c2.x) * sheen_k;
  // a film of water mirrors the sky, most at grazing angles
  if (film > 0.0) { c += L.amb.rgb * sky * fres; }
  // the fire: one shadow toward the centre of its light here, soft by the fire's spread around it
  let nf = light_count[0];
  var cf = vec3<f32>(0.0);
  var fc = vec3<f32>(0.0);
  var fc2 = 0.0;
  var fr = 0.0;
  var fw = 0.0;
  for (var k = 0u; k < nf; k++) {
    let a = lights[2u * k];
    let d = a.xyz - i.p;
    let r2 = dot(d, d);
    let e = lights[2u * k + 1u].rgb * (U.sund.w / (r2 + a.w * a.w));
    cf += light_cloth(e, d * inverseSqrt(max(r2, 1e-8)), nn, v, t, alb, dm, sheen_on, spec_on);
    let w = luma(e);
    fc += a.xyz * w;
    fc2 += dot(a.xyz, a.xyz) * w;
    fr += a.w * w;
    fw += w;
  }
  if (fw > 1e-12) {
    let pc = fc / fw;
    cf *= fire_shadow(i.p, n, pc, sqrt(max(fc2 / fw - dot(pc, pc), 0.0)) + fr / fw);
  }
  c += cf;
  }
  // the lights in the set
  for (var k = 0; k < i32(U.ln.w); k++) {
    let lm = lamps[k];
    let d = lm.p.xyz - i.p;
    let d2 = dot(d, d);
    let dir = d * inverseSqrt(max(d2, 1e-12));
    let f = lamp_shape(k, -dir) / (d2 + lm.p.w * lm.p.w);
    if (f <= 0.0) { continue; }
    var tr = 1.0;
    let pq = off_sheet(pl, n, dir);
    if (in_light(pq)) {
      let lc = U.lg.w / U.lsc.xyz;
      tr = lamp_tr(pq, k) * obj_shadow(pq, dir, length(d / lc));
    }
    c += light_cloth(lm.c.rgb * (f * tr), dir, nn, v, t, alb, dm, sheen_on, spec_on);
  }
  // its own light where it burns
  let glow = burn_glow(bcn, st.z, i.uv, dm, L);
  var o: FOut;
  o.col = vec4<f32>(c * U.xp.x + glow, 1.0);
  let dist = max(length(i.p - U.eye.xyz) - U.eye.w / max(dot(-v, U.fwd.xyz), 1e-3), 0.0);
  o.aux = vec4<f32>(i.cw, dist, 0.0, 1.0);
  o.glow = vec4<f32>(glow, 1.0);
  return o;
}
