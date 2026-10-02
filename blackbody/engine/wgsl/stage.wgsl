// The stage: the set drawn in CG, as a background plate the fire and the liquid are put over.
//
// Without footage, every pixel is traced: the floor (the ground plane, with a procedural floor out to
// the horizon), the objects in the shot in their materials (scene/materials.py: wood, stone, concrete,
// brick, metal, glass, ...) and the sky behind them (the Lighting's ambient light, brighter toward the
// horizon, or the environment HDRI), or the background colour.
//
// With footage, only the objects drawn in CG are traced over it: hidden behind the footage's own
// surfaces (its depth pass) and its matte, and behind the objects that are in the footage. Their shadows
// darken the footage where they fall on the ground or on an object in the footage (a shadow catcher).
//
// Objects are found by sphere tracing their exact shapes (colliders.wgsl), as the volume march does to
// hide the fire behind them, so the two always agree. Surfaces are lit in the same light as the smoke
// and the fabric (cloth_draw.wgsl):
// - the key light, with the objects' soft shadows and the smoke's (the light volume's transmittance);
// - the sky, occluded by the objects nearby (ambient occlusion) and by the smoke overhead;
// - the fire: its short list of point lights (Renderer.lights), with one soft shadow toward their
//   centre through the objects and the smoke (as the march lights surfaces);
// - the lights in the set, shadowed by the objects and the smoke.
// Diffuse is Lambert, highlights GGX with Schlick's Fresnel; metals tint their reflection; glass and ice
// refract what is behind them (one surface in, one out) and reflect the sky.
//
// Each pixel takes res.z samples spread over it and over the shutter (objects moved along their
// velocity: motion blur).
//
// The pieces of broken objects (engine/fracture.py, any number of them) are traced exactly as convex polyhedra
// (each bounded by planes in its own frame), found through a coarse grid of the cells they overlap. Their cut faces
// are drawn in the material's inside colour.
//
// With Lighting engine Lume, each pixel's light is path traced instead (lume.wgsl): one path per pixel per pass, the
// passes added up in ACC (with the first surface's albedo, normal and distance in AOV, for the denoiser).
//
// out: rgb = the plate (scene-linear), a = how much of the pixel is CG. The composite does not light CG
// pixels again with the fire and the lamps (they are lit here); footage pixels it lights as before.
// out_hold: the holdouts the march and the liquid use: x = the footage's matte, y = the distance from the camera to
// the nearest piece, or to the footage's own surface (0: none).
//!include common.wgsl
//!include noise.wgsl
//!include colour.wgsl
//!include meshsdf.wgsl
//!include colliders.wgsl
//!include burn_common.wgsl
//!include burnobj.wgsl
//!include lume.wgsl

const PI: f32 = 3.14159265;
const FLOOR: i32 = 100;
const PIECE: i32 = 1000;   // a hit on piece k has id PIECE + k
const ROPE_ROW: i32 = 16;  // = MAX_COLLIDERS: the material rows of ropes (then steel cables and springs) follow the objects'
const MAT_ROWS: u32 = 19u;
const LIGHTNING_ROW: i32 = 18;   // lightning: segments that glow (their r.xyz), casting no shadow
const MATTER: i32 = 200;   // a hit on the matter (sand, snow, mud, jelly, clay: matter.py)

struct Mat {
  c: vec4<f32>,   // albedo (linear rgb), roughness
  d: vec4<f32>,   // metal (0..1), clear (share of light through it), pattern, drawn (0 not in the shot, 1 in the footage, 2 CG)
  e: vec4<f32>,   // colour of broken faces (linear rgb), index of refraction
};

struct Lamp { p: vec4<f32>, c: vec4<f32>, d: vec4<f32>, e: vec4<f32> };

struct Params {
  inv_vp: mat4x4<f32>,  // clip -> world
  w2l: mat4x4<f32>,     // world -> fire-local
  res: vec4<f32>,       // plate width, height (px), samples per pixel, shutter (s)
  fit: vec4<f32>,       // footage fit scale (x, y: picture uv -> plate uv), lens distortion k1, pixel size at 1 m (m)
  fwd: vec4<f32>,       // camera forward (world), camera to the near plane along it (m)
  stage: vec4<f32>,     // floor drawn (1/0), floor height (fire-local m), sky (1) or the background colour (0), footage (1/0)
  floor_c: vec4<f32>,   // floor albedo (linear rgb), roughness
  floor_d: vec4<f32>,   // floor pattern, the footage's ground takes shadows (1/0), fire yaw (rad), horizon fade (m)
  bg: vec4<f32>,        // background colour (linear rgb), checkerboard (1/0)
  sky: vec4<f32>,       // sky radiance (rgb), checker size (px)
  sun: vec4<f32>,       // key light (irradiance, rgb), shadow sharpness
  sund: vec4<f32>,      // toward the key light (fire-local), ambient occlusion reach (m)
  fire: vec4<f32>,      // fire light gain (rgb), shadows (0..1)
  lv: vec4<f32>,        // simulation grid corner (fire-local m), cell (m)
  ln: vec4<f32>,        // light volume dims, lights in the set (count)
  lsc: vec4<f32>,       // light cells per simulation cell (xyz), light volume present (1/0)
  atm: vec4<f32>,       // haze colour (linear rgb), extinction (1/m)
  foot: vec4<f32>,      // footage: input transform (4 = OCIO), gain, matte on (1/0), depth pass on (1/0)
  depth: vec4<f32>,     // depth pass kind (0 Z, 1 distance, 2 inverse Z), metres per unit, frame (seed), lamps' gain
  envp: vec4<f32>,      // environment on (1/0), rotation (rad), strength, _
  oc: vec4<f32>,        // OCIO: log shaper low and high (log2), footage LUT uses the shaper (1/0), LUT size
  bound: vec4<f32>,     // a sphere round every object (fire-local centre, radius; radius 0: none)
  pg: vec4<f32>,        // the pieces' grid: corner (fire-local m), cell size (m)
  pn: vec4<f32>,        // its dims (cells), pieces (count)
  mo: vec4<f32>,        // the matter's grid: node 0 (fire-local m), node spacing (m)
  mn: vec4<f32>,        // its nodes (x, y, z), matter drawn (1/0)
  mlo: vec4<f32>,       // the matter's grid's extent (fire-local m: its first and last nodes), _
  mhi: vec4<f32>,
  bolt: vec4<f32>,      // lightning: its first segment (piece index), how many, its glow's reach (m), its glow's strength
  gm: vec4<f32>,        // grass on the ground (strands.py): its map's corner (fire-local x, z), size (m; 0: none)
  bf: vec4<f32>,        // the burnable floor (the simulation grid's bottom layer): its corner (fire-local x, z), cell (m), on (1/0)
  bf2: vec4<f32>,       // its cells (x, z), burnable objects (the object-burn atlas) on (1/0), _
  mg: vec4<f32>,        // hot matter's glow: on (1/0), its radiance at 1300 K, the table's first temperature (K), 1 / its step
  mgb: array<vec4<f32>, 16>,   // a blackbody's colour (luminance 1) and log10 of its luminance relative to 1300 K
  ccnt: vec4<f32>,      // objects (count), _
  col: array<Collider, MAX_COLLIDERS>,
  mat: array<Mat, MAT_ROWS>,
  lume: vec4<f32>,      // Lume (lume.wgsl): on (1/0), this pass (0, 1, ...), bounces, the cap on a bounce's light (0: none)
  lume2: vec4<f32>,     // the HDRI's brightness map to pick directions from (ENV): width, height (0: none); anything clear in the set (1/0), _
};

@group(0) @binding(0) var atlas: texture_3d<f32>;       // mesh distance fields (meshsdf.wgsl)
@group(0) @binding(1) var L0: texture_3d<f32>;          // light volume: fire light (rgb), key-light transmittance (a)
@group(0) @binding(2) var L1: texture_3d<f32>;          // x = sky transmittance (looking straight up)
@group(0) @binding(3) var E: texture_3d<f32>;           // a = extinction (smoke, cloth), 1/m
@group(0) @binding(4) var LT: texture_3d<f32>;          // the lamps' transmittance through the smoke
@group(0) @binding(5) var lin: sampler;
@group(0) @binding(6) var rep: sampler;
@group(0) @binding(7) var<storage, read> lights: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> light_count: array<u32>;
@group(0) @binding(9) var<storage, read> lamps: array<Lamp>;
@group(0) @binding(10) var plate: texture_2d<f32>;      // the footage
@group(0) @binding(11) var hold: texture_2d<f32>;       // its holdouts: x = matte, y = depth pass
@group(0) @binding(12) var env_t: texture_2d<f32>;      // environment (latitude-longitude)
@group(0) @binding(13) var lut_plate: texture_3d<f32>;  // OCIO: footage colour space -> scene-linear
@group(0) @binding(14) var out_plate: texture_storage_2d<rgba16float, write>;
// pieces: a = centre (fire-local m), planes (count); q = orientation (x y z w); v = velocity (m/s), first plane;
// o = spin (rad/s, world), material row; r = its centre in the object's frame before it broke, bounding radius.
// A segment of a rope or a spring (material row ROPE_ROW on) is a prism along its y axis, its first planes round
// its sides; its r = strands (0: a plain wire), its middle's distance along the rope (m), twist (radii per turn), _.
struct PieceG { a: vec4<f32>, q: vec4<f32>, v: vec4<f32>, o: vec4<f32>, r: vec4<f32>, b: vec4<f32> };   // b: its fire (burn_common.wgsl)
@group(0) @binding(15) var<storage, read> PC: array<PieceG>;
@group(0) @binding(16) var<storage, read> PL: array<vec4<f32>>;   // n, d in the piece's frame (n . x <= d); a cut face's n is 2 long
@group(0) @binding(17) var<storage, read> GC: array<vec2<u32>>;   // per grid cell: first, count in GL
@group(0) @binding(18) var<storage, read> GL: array<u32>;         // piece indices
@group(0) @binding(19) var out_hold: texture_storage_2d<rgba16float, write>;
@group(0) @binding(20) var m_look: texture_3d<f32>;   // the matter's albedo (linear rgb), roughness
@group(0) @binding(21) var m_phi: texture_3d<f32>;
@group(0) @binding(22) var gmap: texture_2d<f32>;     // the grass on the ground: how burnt, how thick (strand_map.wgsl)
@group(0) @binding(23) var burn: texture_3d<f32>;     // the burnable floor's state (burn_common.wgsl), on the simulation grid
@group(0) @binding(24) var burn_obj: texture_3d<f32>; // the burnable objects', each in its own frame (burnobj.wgsl)
@group(0) @binding(25) var<storage, read> slots: array<BurnSlot>;
@group(0) @binding(26) var m_glow: texture_3d<f32>;   // the matter's temperature (K) and how metallic it is
@group(0) @binding(27) var<storage, read> ML: array<vec4<f32>>;   // hot matter's lights (matter_glow.wgsl): [0].x how many    // its distance to the surface (m), clear, sparkle, wrap
@group(0) @binding(28) var<storage, read_write> ACC: array<vec4<f32>>;   // Lume: the passes' light added up (rgb), CG share
@group(0) @binding(29) var<storage, read_write> AOV: array<vec4<f32>>;   // Lume: per pixel, albedo + distance, normal + _
@group(0) @binding(30) var<storage, read> ENV: array<f32>;               // Lume: the HDRI's cumulative sums and pdf
@group(1) @binding(0) var<uniform> U: Params;

var<private> g_tau: f32;   // this sample's time in the shutter (s): objects are moved along their motion
var<private> g_over: vec4<f32>;   // a colour a pattern puts in place of the material's (mortar), and how much
var<private> g_plane: i32;        // the plane a piece was hit on (trace)
var<private> g_opaque: bool;      // only what stops light counts (shadows, the sky's occlusion)
var<private> g_skip_plain: bool;  // the march leaves the plain shapes out (trace() finds them exactly)
var<private> g_jit: f32;          // this sample's random number in [0, 1) (soft shadows start their march a little apart)
var<private> t_piece: f32;        // how far along the camera's ray the first piece is (see), -1: none
var<private> t_footage: f32;      // and the footage's own surface (see), 1e9: none

// ---- the objects -----------------------------------------------------------------------------------

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

// Object i where it is at this sample's time.
fn obj(i: i32) -> Collider {
  var k = U.col[i];
  if (g_tau != 0.0) {
    k.a = vec4<f32>(k.a.xyz + k.v.xyz * g_tau, k.a.w);
    k.b = vec4<f32>(k.b.xyz, k.b.w + k.v.w * g_tau);
    let w = k.o.xyz * g_tau;
    let ang = length(w);
    if (ang > 1e-6) { k.r = normalize(quat_mul(vec4<f32>(w / ang * sin(0.5 * ang), cos(0.5 * ang)), k.r)); }
  }
  return k;
}

// ---- the matter -----------------------------------------------------------------------------------------

// The light of matter at T (K): a blackbody, from the table (U.mgb), in the stage's units (U.mg.y at 1300 K).
fn matter_glow(T: f32) -> vec3<f32> {
  let f = (T - U.mg.z) * U.mg.w;
  if (f <= 0.0) { return vec3<f32>(0.0); }
  let fc = min(f, 14.999);
  let i = u32(fc);
  let t = fc - f32(i);
  let a = U.mgb[i];
  let b = U.mgb[i + 1u];
  var lg = mix(a.w, b.w, t);
  if (f > 15.0) { lg += (f - 15.0) * (b.w - a.w); }
  return mix(a.rgb, b.rgb, t) * pow(10.0, lg) * U.mg.y;
}

fn matter_uvw(p: vec3<f32>) -> vec3<f32> {
  return ((p - U.mo.xyz) / U.mo.w + vec3<f32>(0.5)) / U.mn.xyz;
}

// The distance from fire-local p to the matter's surface (m): its field (distances from everywhere in its grid:
// mpm_jfa.wgsl), and outside the grid, the distance to it plus the field at its edge.
fn matter_d(p: vec3<f32>) -> f32 {
  let q = clamp(p, U.mlo.xyz, U.mhi.xyz);
  return length(p - q) + textureSampleLevel(m_phi, lin, matter_uvw(q), 0.0).x;
}

fn matter_normal(p: vec3<f32>) -> vec3<f32> {
  let e = 0.5 * U.mo.w;
  let g = vec3<f32>(matter_d(p + vec3<f32>(e, 0.0, 0.0)) - matter_d(p - vec3<f32>(e, 0.0, 0.0)),
                    matter_d(p + vec3<f32>(0.0, e, 0.0)) - matter_d(p - vec3<f32>(0.0, e, 0.0)),
                    matter_d(p + vec3<f32>(0.0, 0.0, e)) - matter_d(p - vec3<f32>(0.0, 0.0, e)));
  let l = length(g);
  return select(vec3<f32>(0.0, 1.0, 0.0), g / l, l > 1e-12);
}

// Where a ray inside the matter leaves it: its distance along the ray.
fn matter_exit_t(ro: vec3<f32>, rd: vec3<f32>, eps: f32) -> f32 {
  var t = 2.0 * eps;
  for (var j = 0; j < 96; j++) {
    let d = matter_d(ro + rd * t);
    if (d > 0.0) { return t; }   // (only out of it for good: dips in its field inside are not surfaces)
    t += max(-d, max(eps, 0.25 * U.mo.w));
  }
  return t;
}

// Anything to trace: objects, or the matter.
fn objects_on() -> bool {
  return U.ccnt.x > 0.5 || U.mn.w > 0.5;
}

// The nearest object at fire-local p whose drawn flag is at least `want` (1: in the shot, 2: CG):
// (distance (m), index).
fn scene_d(p: vec3<f32>, want: f32) -> vec2<f32> {
  var best = vec2<f32>(1.0e9, -1.0);
  if (U.mn.w > 0.5) {
    best = vec2<f32>(matter_d(p), f32(MATTER));
  }
  for (var i = 0; i < i32(U.ccnt.x); i++) {
    if (U.mat[i].d.w < want) { continue; }
    if (g_opaque && U.mat[i].d.y > 0.5) { continue; }   // (glass lets the light through)
    if (g_skip_plain && plain(i)) { continue; }
    let d = col_sdf(obj(i), p);
    if (d < best.x) { best = vec2<f32>(d, f32(i)); }
  }
  return best;
}

fn obj_normal(i: i32, p: vec3<f32>, e: f32) -> vec3<f32> {
  let k = obj(i);
  let a = vec3<f32>(1.0, -1.0, -1.0);
  let b = vec3<f32>(-1.0, -1.0, 1.0);
  let c = vec3<f32>(-1.0, 1.0, -1.0);
  let d = vec3<f32>(1.0, 1.0, 1.0);
  let g = a * col_sdf(k, p + a * e) + b * col_sdf(k, p + b * e) + c * col_sdf(k, p + c * e) + d * col_sdf(k, p + d * e);
  let l = length(g);
  return select(vec3<f32>(0.0, 1.0, 0.0), g / l, l > 1e-12);
}

// Where the ray enters and leaves the sphere round all the objects (t0 > t1: it misses).
fn bound_span(ro: vec3<f32>, rd: vec3<f32>) -> vec2<f32> {
  if (U.bound.w <= 0.0) { return vec2<f32>(1.0, 0.0); }
  let oc = ro - U.bound.xyz;
  let b = dot(oc, rd);
  let c = dot(oc, oc) - U.bound.w * U.bound.w;
  let disc = b * b - c;
  if (disc < 0.0) { return vec2<f32>(1.0, 0.0); }
  let s = sqrt(disc);
  return vec2<f32>(max(-b - s, 0.0), -b + s);
}

struct Hit { t: f32, id: i32 };

// ---- pieces ---------------------------------------------------------------------------------------------------------

// Piece k where it is at this sample's time: centre, orientation.
fn piece_pose(k: u32) -> array<vec4<f32>, 2> {
  let P = PC[k];
  var q = P.q;
  if (g_tau != 0.0) {
    let w = P.o.xyz * g_tau;
    let ang = length(w);
    if (ang > 1e-6) { q = normalize(quat_mul(vec4<f32>(w / ang * sin(0.5 * ang), cos(0.5 * ang)), q)); }
  }
  return array<vec4<f32>, 2>(vec4<f32>(P.a.xyz + P.v.xyz * g_tau, 0.0), q);
}

struct PHit { t0: f32, t1: f32, k0: i32, k1: i32 };

// Where a ray (fire-local) enters and leaves piece k (t0 > t1: it misses), and the planes it crosses there.
fn piece_hit(k: u32, ro: vec3<f32>, rd: vec3<f32>) -> PHit {
  let P = PC[k];
  let pose = piece_pose(k);
  let qi = vec4<f32>(-pose[1].xyz, pose[1].w);
  let o = quat_rotate(qi, ro - pose[0].xyz);
  let d = quat_rotate(qi, rd);
  var t0 = -1.0e30;
  var t1 = 1.0e30;
  var k0 = -1;
  var k1 = -1;
  let first = u32(P.v.w);
  let n = u32(P.a.w);
  for (var j = 0u; j < n; j++) {
    let pl = PL[first + j];
    let nn = normalize(pl.xyz);
    let dn = dot(nn, d);
    let dist = pl.w - dot(nn, o);
    if (abs(dn) < 1e-12) {
      if (dist < 0.0) { return PHit(1.0, 0.0, -1, -1); }
      continue;
    }
    let t = dist / dn;
    if (dn < 0.0) {
      if (t > t0) { t0 = t; k0 = i32(j); }
    } else if (t < t1) {
      t1 = t;
      k1 = i32(j);
    }
    if (t0 > t1) { return PHit(1.0, 0.0, -1, -1); }
  }
  return PHit(t0, t1, k0, k1);
}

// The nearest piece along a ray from t0 to tmax: (t, piece, plane) through the pieces' grid, or piece -1.
fn trace_pieces(ro: vec3<f32>, rd: vec3<f32>, t0: f32, tmax: f32) -> vec3<f32> {
  var best = vec3<f32>(tmax, -1.0, -1.0);
  if (U.pn.w < 0.5) { return best; }
  let gd = vec3<i32>(U.pn.xyz);
  let cs = U.pg.w;
  let lo = U.pg.xyz;
  let hi = lo + vec3<f32>(gd) * cs;
  let safe = select(rd, vec3<f32>(1e-12), abs(rd) < vec3<f32>(1e-12));
  let inv = 1.0 / safe;
  let ta = (lo - ro) * inv;
  let tb = (hi - ro) * inv;
  let tin = max(max(max(min(ta.x, tb.x), min(ta.y, tb.y)), min(ta.z, tb.z)), t0);
  let tout = min(min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z)), tmax);
  if (tin > tout) { return best; }
  let p = ro + rd * (tin + 1e-6 * cs);
  var c = clamp(vec3<i32>(floor((p - lo) / cs)), vec3<i32>(0), gd - vec3<i32>(1));
  let stp = vec3<i32>(sign(rd));
  let dt = abs(cs * inv);
  var tnext = (lo + (vec3<f32>(c) + select(vec3<f32>(0.0), vec3<f32>(1.0), rd > vec3<f32>(0.0))) * cs - ro) * inv;
  tnext = select(tnext, vec3<f32>(1.0e30), abs(rd) < vec3<f32>(1e-12));
  for (var it = 0; it < 256; it++) {
    let ci = u32(c.x + gd.x * (c.y + gd.y * c.z));
    let r = GC[ci];
    for (var j = r.x; j < r.x + r.y; j++) {
      let k = GL[j];
      if (g_opaque && U.mat[i32(PC[k].o.w + 0.5)].d.y > 0.5) { continue; }
      let h = piece_hit(k, ro, rd);
      if (h.t0 <= h.t1 && h.t0 > t0 && h.t0 < best.x) { best = vec3<f32>(h.t0, f32(k), f32(h.k0)); }
    }
    let tn = min(tnext.x, min(tnext.y, tnext.z));
    if (tn > best.x || tn > tout) { break; }
    if (tnext.x <= tnext.y && tnext.x <= tnext.z) {
      c.x += stp.x;
      tnext.x += dt.x;
    } else if (tnext.y <= tnext.z) {
      c.y += stp.y;
      tnext.y += dt.y;
    } else {
      c.z += stp.z;
      tnext.z += dt.z;
    }
    if (any(c < vec3<i32>(0)) || any(c >= gd)) { break; }
  }
  return best;
}

// The first thing along the ray from t0 to tmax: an object (drawn at least `want`), the floor, or none (-1).
// Whether object i is a plain shape (a solid sphere, box or cylinder, nothing cut out of it): trace() finds it exactly,
// with one test, instead of marching toward it.
fn plain(i: i32) -> bool {
  let k = U.col[i];
  return i32(k.a.w + 0.5) <= 2 && k.x.w <= 0.0 && !all(k.y.xyz > vec3<f32>(0.0));
}

// Where a ray (fire-local) enters plain object k past t0: its distance along the ray, or -1 where it misses it or starts
// inside it (a ray goes on through what it starts in, as the march does).
fn plain_hit(k: Collider, ro: vec3<f32>, rd: vec3<f32>, t0: f32) -> f32 {
  let o = col_to_local(k, ro);
  let d = yaw_to_local(quat_rotate(vec4<f32>(-k.r.xyz, k.r.w), rd), k.b.w);
  let shape = i32(k.a.w + 0.5);
  var tn = -1.0;
  var tf = -1.0;
  if (shape == 0) {
    let b = dot(o, d);
    let h = b * b - (dot(o, o) - k.b.x * k.b.x);
    if (h < 0.0) { return -1.0; }
    let sq = sqrt(h);
    tn = -b - sq;
    tf = -b + sq;
  } else if (shape == 1) {
    let inv = vec3<f32>(1.0) / select(d, vec3<f32>(1e-12), abs(d) < vec3<f32>(1e-12));
    let t1 = (-k.b.xyz - o) * inv;
    let t2 = (k.b.xyz - o) * inv;
    let lo = min(t1, t2);
    let hi = max(t1, t2);
    tn = max(max(lo.x, lo.y), lo.z);
    tf = min(min(hi.x, hi.y), hi.z);
    if (tf < tn) { return -1.0; }
  } else {
    // a cylinder round y: inside its round side and between its ends
    let r = k.b.x;
    let a = d.x * d.x + d.z * d.z;
    let bq = o.x * d.x + o.z * d.z;
    let cq = o.x * o.x + o.z * o.z - r * r;
    var r0 = -1.0e30;
    var r1 = 1.0e30;
    if (a > 1e-12) {
      let h = bq * bq - a * cq;
      if (h < 0.0) { return -1.0; }
      let sq = sqrt(h);
      r0 = (-bq - sq) / a;
      r1 = (-bq + sq) / a;
    } else if (cq > 0.0) {
      return -1.0;
    }
    var y0 = -1.0e30;
    var y1 = 1.0e30;
    if (abs(d.y) > 1e-12) {
      let ta = (-k.b.y - o.y) / d.y;
      let tb = (k.b.y - o.y) / d.y;
      y0 = min(ta, tb);
      y1 = max(ta, tb);
    } else if (abs(o.y) > k.b.y) {
      return -1.0;
    }
    tn = max(r0, y0);
    tf = min(r1, y1);
    if (tf < tn) { return -1.0; }
  }
  if (tn > t0) { return tn; }
  return -1.0;
}

fn trace(ro: vec3<f32>, rd: vec3<f32>, t0: f32, tmax: f32, want: f32, floor_on: bool) -> Hit {
  var h = Hit(1.0e9, -1);
  // (a ray from a hair under the floor going down, as from the underside of a ball resting on it, is in the ground)
  if (floor_on && U.stage.x > 0.5 && rd.y < -1e-6 && ro.y > U.stage.y - 0.02) {
    let tf = max((U.stage.y - ro.y) / rd.y, t0 + 1e-6);
    if (tf < tmax) { h = Hit(tf, FLOOR); }
  }
  let pc = trace_pieces(ro, rd, t0, min(tmax, h.t));
  if (pc.y >= 0.0) {
    h = Hit(pc.x, PIECE + i32(pc.y));
    g_plane = i32(pc.z);
  }
  if (!objects_on()) { return h; }
  // the plain shapes, exactly; only the rest (meshes, hollow things, things with openings, the matter) are marched
  var rest = U.mn.w > 0.5;
  for (var i = 0; i < i32(U.ccnt.x); i++) {
    if (U.mat[i].d.w < want) { continue; }
    if (g_opaque && U.mat[i].d.y > 0.5) { continue; }
    if (!plain(i)) {
      rest = true;
      continue;
    }
    let tp = plain_hit(obj(i), ro, rd, t0);
    if (tp > 0.0 && tp < min(tmax, h.t)) { h = Hit(tp, i); }
  }
  if (!rest) { return h; }
  let span = bound_span(ro, rd);
  var t = max(t0, span.x);
  let lim = min(min(tmax, h.t), span.y);
  let hp = h;
  g_skip_plain = true;
  var escaping = true;   // a ray that starts inside an object goes on through it
  var t_out = t;         // the last point found outside everything
  for (var i = 0; i < 192; i++) {
    if (t >= lim) { break; }
    let d = scene_d(ro + rd * t, want);
    let eps = max(1.0e-4, 0.3 * U.fit.w * t);
    if (escaping) {
      if (d.x < 0.0) { t += max(-d.x, eps); continue; }
      escaping = false;
    }
    if (d.x < eps) {
      if (d.x < -eps && i32(d.y) == MATTER) {
        // stepped past the matter's surface (its field is not an exact distance): back to where the ray crosses it
        var a = t_out;
        var b = t;
        for (var k = 0; k < 8; k++) {
          let m = 0.5 * (a + b);
          if (matter_d(ro + rd * m) > 0.0) { a = m; } else { b = m; }
        }
        g_skip_plain = false;
        return Hit(b, MATTER);
      }
      g_skip_plain = false;
      return Hit(t, i32(d.y));
    }
    t_out = t;
    t += max(d.x, 0.4 * eps);
  }
  g_skip_plain = false;
  return hp;
}

// How much of a light of angular size k (sharpness: 1 / tan of its angular radius) in direction l gets
// to p past the objects (soft shadow; Quilez's).
fn soft_shadow(p: vec3<f32>, l: vec3<f32>, tmax: f32, k: f32, want: f32, t0: f32) -> f32 {
  g_opaque = true;
  let r = soft_shadow_opaque(p, l, tmax, k, want, t0);
  g_opaque = false;
  return r;
}

fn soft_shadow_opaque(p: vec3<f32>, l: vec3<f32>, tmax: f32, k: f32, want: f32, t0: f32) -> f32 {
  if (trace_pieces(p, l, t0, tmax).y >= 0.0) { return 0.0; }
  if (!objects_on()) { return 1.0; }
  let span = bound_span(p, l);
  if (span.x > span.y) { return 1.0; }
  var res = 1.0;
  // (the first step jittered per sample: what bands are left round a heap become noise the samples average away)
  var t = max(t0, span.x) * (1.0 + 0.5 * g_jit);
  let lim = min(tmax, span.y);
  var ph = 1.0e20;
  for (var i = 0; i < 64; i++) {
    if (t >= lim) { break; }
    let d = scene_d(p + l * t, want).x;
    // the closest the ray comes to what is near it between this step and the last (Quilez's improved soft shadow), not
    // only at the steps: the steps' pattern changes from pixel to pixel and drew contours round a heap
    let y = select(d * d / (2.0 * ph), 0.0, i == 0);
    let dd = sqrt(max(d * d - y * y, 0.0));
    res = min(res, k * dd / max(t - y, 1e-4));
    if (res < 0.002) { return 0.0; }
    ph = d;
    t += clamp(d, 0.5 * t0, 0.25 * max(lim, 0.1));
  }
  let s = clamp(res, 0.0, 1.0);
  return s * s * (3.0 - 2.0 * s);
}

// Ambient occlusion by the objects over the hemisphere round n at p (1 open .. 0 enclosed).
fn ambient_occ(p: vec3<f32>, n: vec3<f32>, want: f32) -> f32 {
  g_opaque = true;
  let r = ambient_occ_opaque(p, n, want);
  g_opaque = false;
  return r;
}

fn ambient_occ_opaque(p: vec3<f32>, n: vec3<f32>, want: f32) -> f32 {
  if (!objects_on()) { return 1.0; }
  let reach = U.sund.w;
  if (length(p - U.bound.xyz) > U.bound.w + reach) { return 1.0; }
  var occ = 0.0;
  var w = 1.0;
  var ws = 0.0;
  for (var i = 1; i <= 5; i++) {
    let hgt = reach * f32(i) / 5.0;
    let d = scene_d(p + n * hgt, want).x;
    occ += w * clamp((hgt - d) / hgt, 0.0, 1.0);
    ws += w;
    w *= 0.75;
  }
  return clamp(1.0 - 1.4 * occ / ws, 0.0, 1.0);
}

// ---- the sky and the footage -------------------------------------------------------------------------

// A fire-local direction in world axes (the fire is turned about the vertical).
fn to_world_dir(d: vec3<f32>) -> vec3<f32> {
  let c = cos(U.floor_d.z);
  let s = sin(U.floor_d.z);
  return vec3<f32>(c * d.x + s * d.z, d.y, -s * d.x + c * d.z);
}

fn hdri(dw: vec3<f32>) -> vec3<f32> {
  let c = cos(U.envp.y);
  let s = sin(U.envp.y);
  let r = vec3<f32>(c * dw.x - s * dw.z, dw.y, s * dw.x + c * dw.z);
  let u = atan2(r.x, -r.z) * (0.5 / PI) + 0.5;
  let v = acos(clamp(r.y, -1.0, 1.0)) / PI;
  return textureSampleLevel(env_t, rep, vec2<f32>(u, v), 0.0).rgb * U.envp.z;
}

// The sky in fire-local direction d (as the liquid renderer's sky): brighter toward the horizon, dim below.
fn sky_dir(d: vec3<f32>) -> vec3<f32> {
  if (U.envp.x > 0.5) { return hdri(to_world_dir(d)); }
  let s = U.sky.rgb;
  let above = mix(s * 1.35, s * 0.9, clamp(d.y, 0.0, 1.0));
  return mix(s * 0.3, above, smoothstep(-0.06, 0.04, d.y));
}

fn lut3(t: texture_3d<f32>, c: vec3<f32>) -> vec3<f32> {
  let n = U.oc.w;
  let q = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)) * ((n - 1.0) / n) + vec3<f32>(0.5 / n);
  return textureSampleLevel(t, lin, q, 0.0).rgb;
}

fn footage_at(puv: vec2<f32>) -> vec3<f32> {
  let c = textureSampleLevel(plate, lin, puv, 0.0).rgb;
  var v = input_transform(c, i32(U.foot.x));
  if (i32(U.foot.x) == 4) {
    if (U.oc.z > 0.5) {
      let lo = U.oc.x;
      v = lut3(lut_plate, (log2(max(c, vec3<f32>(0.0)) + vec3<f32>(exp2(lo))) - vec3<f32>(lo)) / (U.oc.y - lo));
    } else {
      v = lut3(lut_plate, c);
    }
  }
  return v * U.foot.y;
}

// The footage's depth pass at plate uv as a distance along the ray from the camera (m), or 0 where it has none.
fn footage_distance(puv: vec2<f32>, rd_w: vec3<f32>) -> f32 {
  let v = textureSampleLevel(hold, lin, puv, 0.0).y;
  if (!(v > 0.0) || v > 1.0e20) { return 0.0; }
  let kind = i32(U.depth.x + 0.5);
  var d = v * U.depth.y;
  if (kind == 2) { d = U.depth.y / v; }
  if (kind != 1) { d = d / max(dot(rd_w, U.fwd.xyz), 1e-3); }
  return d;
}

// ---- materials -----------------------------------------------------------------------------------------

fn hash31(p: vec3<f32>) -> f32 {
  return f32(pcg3d(vec3<u32>(vec3<i32>(floor(p)) + vec3<i32>(1048576))).x) * (1.0 / 4294967296.0);
}

// Noise that fades to its mean (0) where its features are smaller than the pixel (fw, in its own units).
fn fnoise(q: vec3<f32>, fw: f32) -> f32 {
  return gnoise(q) * (1.0 - smoothstep(0.3, 0.9, fw));
}

// A line of half-width w round each integer of x, softened over the pixel (fw): 1 on the line.
fn grid_line(x: f32, w: f32, fw: f32) -> f32 {
  let d = abs(fract(x + 0.5) - 0.5);
  return (1.0 - smoothstep(w - fw, w + fw, d)) * (1.0 - smoothstep(0.15, 0.45, fw));
}

// Surface detail of a pattern at object-space q (m), object size s (half extents), object-space normal n,
// pixel footprint fw (m): rgb = albedo multiplier, w = roughness offset (and g_over).
fn pattern(kind: i32, q: vec3<f32>, s: vec3<f32>, n: vec3<f32>, fw: f32) -> vec4<f32> {
  g_over = vec4<f32>(0.0);
  if (kind == 1) {
    // wood: growth rings round the long axis, wavy, with fine fibres along it
    var ax = 0;
    if (s.y >= s.x && s.y >= s.z) { ax = 1; } else if (s.z >= s.x && s.z >= s.y) { ax = 2; }
    var along = q.x;
    var across = q.yz;
    if (ax == 1) { along = q.y; across = q.xz; } else if (ax == 2) { along = q.z; across = q.xy; }
    let warp = gnoise(vec3<f32>(across * 3.0, along * 0.6));
    let r = length(across + vec2<f32>(0.37, -0.83) * max(s.x + s.y + s.z, 0.1)) * 160.0 + warp * 5.0;
    let ring = smoothstep(0.0, 0.25, fract(r)) * (1.0 - smoothstep(0.55, 1.0, fract(r)));
    let fade = smoothstep(0.002, 0.006, fw);
    let rings = mix(mix(0.6, 1.0, ring), 0.85, fade);
    let fib = fnoise(vec3<f32>(across * 400.0, along * 12.0), fw * 400.0);
    let knot = smoothstep(0.55, 0.8, gnoise(vec3<f32>(across * 2.0, along * 1.3) + vec3<f32>(5.1)));
    let m = rings * (1.0 + 0.12 * fib) * (1.0 - 0.45 * knot);
    return vec4<f32>(vec3<f32>(m * 1.15, m * 1.12, m * 1.1), 0.08 * fib);
  }
  if (kind == 2) {
    // stone (granite): mottled, with light and dark crystals
    let mott = fbm3(q * 5.0, 3);
    let sp = fnoise(q * 160.0, fw * 160.0);
    let dark = smoothstep(0.25, 0.45, sp);
    let light = smoothstep(0.3, 0.5, -sp);
    var c = vec3<f32>(1.0 + 0.5 * mott);
    c = mix(c, vec3<f32>(0.35), dark * 0.8);
    c = mix(c, vec3<f32>(1.45, 1.3, 1.25), light * 0.6);
    return vec4<f32>(c, 0.1 * sp);
  }
  if (kind == 3) {
    // concrete: blotchy, with small dark pores
    let bl = fbm3(q * 2.5, 3);
    let pores = smoothstep(0.42, 0.55, fnoise(q * 120.0, fw * 120.0));
    let grain = fnoise(q * 500.0, fw * 500.0);
    return vec4<f32>(vec3<f32>((1.0 + 0.45 * bl) * (1.0 - 0.45 * pores) * (1.0 + 0.1 * grain)), 0.05 * grain);
  }
  if (kind == 4) {
    // brick: running bond (21.5 x 6.5 cm faces, 1 cm mortar) on the vertical faces, headers on top
    var u = q.x;
    var v = q.y;
    var len = 0.225;
    if (abs(n.x) > abs(n.z)) { u = q.z; }
    if (abs(n.y) > 0.7) { u = q.x; v = q.z; len = 0.1125; }
    let row = floor(v / 0.075);
    let uu = u / len + 0.5 * (row - 2.0 * floor(row * 0.5));
    let col = floor(uu);
    let mortar = max(grid_line(v / 0.075, 0.5 * 0.01 / 0.075, fw / 0.075), grid_line(uu, 0.5 * 0.01 / len, fw / len));
    let hb = hash31(vec3<f32>(col, row, 7.0));
    let c = vec3<f32>(0.8 + 0.4 * hb, 0.8 + 0.35 * hb, 0.85 + 0.3 * hb) * (1.0 + 0.15 * fnoise(q * 60.0, fw * 60.0));
    g_over = vec4<f32>(0.5, 0.48, 0.44, mortar);
    return vec4<f32>(c, 0.1 * mortar);
  }
  if (kind == 5) {
    // brushed metal: fine streaks along one axis
    let st = fnoise(vec3<f32>(q.x * 3.0, q.y * 900.0, q.z * 900.0), fw * 900.0);
    return vec4<f32>(vec3<f32>(1.0 + 0.04 * st), 0.12 * st);
  }
  // plain: a slight unevenness
  return vec4<f32>(vec3<f32>(1.0 + 0.06 * fnoise(q * 8.0, fw * 8.0)), 0.0);
}

// The floor's albedo (rgb) and roughness (w) at fire-local (x, z), footprint fw (m).
fn floor_look(xz: vec2<f32>, fw: f32) -> vec4<f32> {
  let kind = i32(U.floor_d.x + 0.5);
  let base = U.floor_c.rgb;
  var rough = U.floor_c.w;
  var m = vec3<f32>(1.0);
  let q = vec3<f32>(xz.x, 0.0, xz.y);
  if (kind == 1) {
    // concrete slab, with saw-cut joints every 3 m
    m = pattern(3, q, vec3<f32>(10.0), vec3<f32>(0.0, 1.0, 0.0), fw).rgb;
    let joint = max(grid_line(xz.x / 3.0, 0.004 / 3.0, fw / 3.0), grid_line(xz.y / 3.0, 0.004 / 3.0, fw / 3.0));
    m *= 1.0 - 0.6 * joint;
  } else if (kind == 2) {
    // wooden boards 14 cm wide, of staggered lengths, along x
    let row = floor(xz.y / 0.14);
    let lenb = 1.2 + 1.2 * hash31(vec3<f32>(row, 3.0, 1.0));
    let sh = hash31(vec3<f32>(row, 5.0, 2.0)) * lenb;
    let plank = floor((xz.x + sh) / lenb);
    let hb = hash31(vec3<f32>(plank, row, 9.0));
    let gap = max(grid_line(xz.y / 0.14, 0.0015 / 0.14, fw / 0.14), grid_line((xz.x + sh) / lenb, 0.0015 / lenb, fw / lenb));
    let wq = vec3<f32>((xz.x + sh) - plank * lenb, row * 0.37 + hb * 3.0, fract(xz.y / 0.14) * 0.14);
    m = pattern(1, vec3<f32>(wq.x, wq.y * 0.02, wq.z) , vec3<f32>(2.0, 0.01, 0.07), vec3<f32>(0.0, 1.0, 0.0), fw).rgb;
    m *= (0.8 + 0.4 * hb) * (1.0 - 0.75 * gap);
  } else if (kind == 3) {
    // 30 cm tiles with grout
    let t = floor(xz / 0.3);
    let hb = hash31(vec3<f32>(t, 4.0));
    let grout = max(grid_line(xz.x / 0.3, 0.002 / 0.3, fw / 0.3), grid_line(xz.y / 0.3, 0.002 / 0.3, fw / 0.3));
    m = vec3<f32>(0.94 + 0.12 * hb) * (1.0 + 0.04 * fnoise(q * 20.0, fw * 20.0));
    m = mix(m, vec3<f32>(0.55), grout);
    rough = mix(rough, 0.85, grout);
  } else if (kind == 4) {
    // dirt: lumpy, with pebbles
    let lump = fbm3(q * 1.5, 4);
    let peb = smoothstep(0.35, 0.5, fnoise(q * 40.0, fw * 40.0));
    m = vec3<f32>(1.0 + 0.6 * lump) * (1.0 + 0.5 * peb);
  } else if (kind == 5) {
    // grass: patchy, with blades
    let tuft = fbm3(q * 0.7, 3);
    let blades = fnoise(vec3<f32>(xz.x * 300.0, 0.0, xz.y * 80.0), fw * 300.0);
    m = vec3<f32>(1.0 + 0.5 * tuft, 1.0 + 0.35 * tuft, 1.0 + 0.2 * tuft) * (1.0 + 0.3 * blades);
  } else if (kind == 6) {
    // sand: fine grains and wind ripples
    let rip = sin(xz.x * 40.0 + 3.0 * gnoise(q * 0.8)) * (1.0 - smoothstep(0.03, 0.12, fw));
    m = vec3<f32>(1.0 + 0.08 * rip + 0.1 * fnoise(q * 700.0, fw * 700.0) + 0.15 * fbm3(q * 1.2, 2));
  } else if (kind == 7) {
    // a 1 m checker, for judging scale
    let ck = (i32(floor(xz.x)) + i32(floor(xz.y))) & 1;
    m = vec3<f32>(select(0.45, 1.0, ck == 1));
    let fade = smoothstep(0.2, 0.6, fw);
    m = mix(m, vec3<f32>(0.725), fade);
  } else {
    m = vec3<f32>(1.0 + 0.03 * fnoise(q * 4.0, fw * 4.0));
  }
  return vec4<f32>(base * max(m, vec3<f32>(0.0)), clamp(rough, 0.02, 1.0));
}

// ---- light ------------------------------------------------------------------------------------------------

fn ggx(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, rough: f32) -> f32 {
  let h = normalize(l + v);
  let a = max(rough * rough, 0.002);
  let a2 = a * a;
  let nh = max(dot(n, h), 0.0);
  let nl = max(dot(n, l), 1e-4);
  let nv = max(dot(n, v), 1e-4);
  let dd = nh * nh * (a2 - 1.0) + 1.0;
  let D = a2 / (PI * dd * dd);
  let k = a * 0.5;
  let G = (nl / (nl * (1.0 - k) + k)) * (nv / (nv * (1.0 - k) + k));
  return D * G / (4.0 * nl * nv);
}

fn fresnel(f0: vec3<f32>, c: f32) -> vec3<f32> {
  return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - clamp(c, 0.0, 1.0), 5.0);
}

// Fresnel for light from all round a rough surface: a rough one reflects much less at grazing angles
// (Lagarde's fit of the split-sum environment BRDF).
fn fresnel_rough(f0: vec3<f32>, c: f32, rough: f32) -> vec3<f32> {
  return f0 + (max(vec3<f32>(1.0 - rough), f0) - f0) * pow(1.0 - clamp(c, 0.0, 1.0), 5.0);
}

fn light_cell(p: vec3<f32>) -> vec3<f32> { return (p - U.lv.xyz) / U.lv.w * U.lsc.xyz; }

fn in_light(pl: vec3<f32>) -> bool {
  return U.lsc.w > 0.5 && all(pl >= vec3<f32>(0.0)) && all(pl <= U.ln.xyz);
}

fn lamp_tr(pl: vec3<f32>, i: i32) -> f32 {
  let ln = U.ln.xyz;
  let blocks = select(1.0, 2.0, U.ln.w > 4.5);
  let q = clamp(pl, vec3<f32>(0.5), ln - vec3<f32>(0.5));
  let t = textureSampleLevel(LT, lin, vec3<f32>(q.x / ln.x, q.y / ln.y, (q.z + f32(i / 4) * ln.z) / (ln.z * blocks)), 0.0);
  let j = i & 3;
  return select(select(t.w, t.z, j == 2), select(t.y, t.x, j == 0), j < 2);
}

// The smoke between p and a point `reach` metres along dir (the light volume's extinction): transmittance.
fn smoke_tr(p: vec3<f32>, dir: vec3<f32>, reach: f32) -> f32 {
  if (U.lsc.w < 0.5) { return 1.0; }
  let lc = U.lv.w / U.lsc.xyz;   // metres per light cell, per axis
  let cell = max(lc.x, max(lc.y, lc.z));
  let steps = clamp(i32(ceil(reach / (0.75 * cell))), 1, 40);
  let dt = reach / f32(steps);
  let ln = U.ln.xyz;
  var od = 0.0;
  for (var k = 0; k < steps; k++) {
    let q = (p + dir * ((f32(k) + 0.5) * dt) - U.lv.xyz) / lc;
    if (any(q < vec3<f32>(0.0)) || any(q > ln)) { continue; }
    od += samp_c(E, lin, q, ln).a;
  }
  return exp(-od * dt);
}

struct Surf {
  p: vec3<f32>,      // fire-local (m)
  n: vec3<f32>,
  alb: vec3<f32>,    // diffuse albedo
  f0: vec3<f32>,     // reflectance at normal incidence
  rough: f32,
  eps: f32,          // how far off the surface shadow rays start (m)
  want: f32,         // what casts shadows on it (1: everything in the shot, 2: CG objects only)
  wrap: f32,         // light wrapping into it past where it faces away (snow)
  glint: f32,        // the glint of its grains or crystals toward the key light
  gn: vec3<f32>,     // the normal of the grain that glints
  em: vec3<f32>,     // light it gives off (lightning)
  self_glow: f32,    // 1: the hot matter itself (its glow's lights do not light it again)
};

// Light reflected toward v by surface s: the key light, the sky, the fire, the lights in the set.
fn shade(s: Surf, v: vec3<f32>) -> vec3<f32> {
  let n = s.n;
  let nv = max(dot(n, v), 1e-4);
  let po = s.p + n * s.eps;
  let pl = light_cell(s.p);
  var diff = vec3<f32>(0.0);   // irradiance on the diffuse part
  var spec = vec3<f32>(0.0);   // reflected radiance, before the Fresnel term
  // the key light
  if (max(U.sun.r, max(U.sun.g, U.sun.b)) > 0.0) {
    let l = U.sund.xyz;
    let nl = dot(n, l);
    let nlw = (nl + s.wrap) / (1.0 + s.wrap);
    if (nlw > 0.0) {
      var vis = soft_shadow(po, l, 1.0e4, U.sun.w, s.want, 2.0 * s.eps);
      let ps = pl + n * 1.5 + l * 0.75;
      if (vis > 0.0 && in_light(ps)) { vis *= samp_c(L0, lin, ps, U.ln.xyz).a; }
      diff += U.sun.rgb * (nlw * vis);
      if (nl > 0.0) {
        var sp = ggx(n, v, l, s.rough);
        if (s.glint > 0.0) {
          // a grain turned just so flashes the sun at the camera
          sp += s.glint * pow(max(dot(s.gn, normalize(l + v)), 0.0), 600.0) * 40.0;
        }
        spec += U.sun.rgb * (nl * vis * sp);
      }
    }
  }
  // the sky: from above, hidden by objects nearby and by smoke overhead
  var sky = ambient_occ(s.p, n, s.want);
  let pk = pl + n * 1.5;
  if (in_light(pk)) { sky *= samp_c(L1, lin, pk, U.ln.xyz).x; }
  diff += PI * U.sky.rgb * ((0.6 + 0.4 * n.y) * sky);
  let r = reflect(-v, n);
  var env = sky_dir(r);
  if (r.y < 0.0 && U.stage.x > 0.5) {
    // the floor seen in it, lit by the sky
    env = mix(env, U.floor_c.rgb * U.sky.rgb * 0.8, smoothstep(0.0, -0.1, r.y));
  }
  // rough surfaces see the sky's average more than a sharp reflection of it
  env = mix(env, U.sky.rgb * (0.6 + 0.4 * n.y), clamp(s.rough * s.rough * 1.3, 0.0, 1.0));
  // the fire: its point lights, one soft shadow toward their centre, as wide as the fire round it
  let nf = light_count[0];
  if (nf > 0u && max(U.fire.r, max(U.fire.g, U.fire.b)) > 0.0) {
    var fd = vec3<f32>(0.0);
    var fs = vec3<f32>(0.0);
    var fc = vec3<f32>(0.0);
    var fc2 = 0.0;
    var fr = 0.0;
    var fw = 0.0;
    for (var k = 0u; k < nf; k++) {
      let a = lights[2u * k];
      let d = a.xyz - s.p;
      let r2 = dot(d, d);
      let l = d * inverseSqrt(max(r2, 1e-8));
      let e = lights[2u * k + 1u].rgb / (r2 + a.w * a.w);
      let c = dot(n, l);
      // a light the size of its block: the cosine wrapped a little (as the march lights surfaces)
      let facing = clamp((c + 0.2) / 1.2, 0.0, 1.0);
      fd += e * facing;
      if (c > 0.0) { fs += e * (c * ggx(n, v, l, max(s.rough, 0.15))); }
      let wgt = luma(e) * facing;
      fc += a.xyz * wgt;
      fc2 += dot(a.xyz, a.xyz) * wgt;
      fr += a.w * wgt;
      fw += wgt;
    }
    if (fw > 1e-12) {
      let pc = fc / fw;
      let spread = sqrt(max(fc2 / fw - dot(pc, pc), 0.0)) + fr / fw;
      let dv = pc - po;
      let dist = length(dv);
      var vis = 1.0;
      if (dist > 1e-4 && U.fire.w > 0.0) {
        let dir = dv / dist;
        let reach = max(dist - 0.5 * spread, 0.0);
        var sh = soft_shadow(po, dir, reach, dist / max(spread, 0.05), s.want, 2.0 * s.eps);
        sh *= smoke_tr(po, dir, reach);
        vis = mix(1.0, sh, U.fire.w);
      }
      diff += fd * U.fire.rgb * vis;
      spec += fs * U.fire.rgb * vis;
    }
  }
  // hot matter's glow: its point lights (matter_glow.wgsl), without shadows
  let nm = u32(ML[0].x);
  if (nm > 0u && s.self_glow < 0.5) {
    for (var k = 0u; k < nm; k++) {
      let a = ML[1u + 2u * k];
      let d = a.xyz - s.p;
      let r2 = dot(d, d);
      let l = d * inverseSqrt(max(r2, 1e-8));
      let e = ML[2u + 2u * k].rgb / (r2 + a.w * a.w);
      let c = dot(n, l);
      diff += e * clamp((c + 0.2) / 1.2, 0.0, 1.0);
      if (c > 0.0) { spec += e * (c * ggx(n, v, l, max(s.rough, 0.15))); }
    }
  }
  // the lights in the set
  for (var k = 0; k < i32(U.ln.w); k++) {
    let lm = lamps[k];
    let d = lm.p.xyz - s.p;
    let d2 = dot(d, d);
    let dir = d * inverseSqrt(max(d2, 1e-12));
    var f = 1.0 / (d2 + lm.p.w * lm.p.w);
    let kind = i32(lm.c.w + 0.5);
    let facing = dot(-dir, lm.d.xyz);
    if (kind == 1) { f *= smoothstep(lm.d.w, lm.e.x, facing); }
    if (kind == 2) { f *= max(facing, 0.0); }
    let nl = dot(n, dir);
    if (f <= 0.0 || nl <= 0.0) { continue; }
    let dist = sqrt(d2);
    var vis = soft_shadow(po, dir, max(dist - lm.p.w, 0.0), dist / max(lm.p.w, 0.01), s.want, 2.0 * s.eps);
    if (lm.e.y > 0.5) {
      let pq = pl + n * 1.5 + dir * 0.75;
      if (in_light(pq)) { vis *= lamp_tr(pq, k); }
    }
    let e = lm.c.rgb * (f * vis * U.depth.w);
    diff += e * nl;
    spec += e * (nl * ggx(n, v, dir, s.rough));
  }
  let fv = fresnel(s.f0, nv);
  let fe = fresnel_rough(s.f0, nv, s.rough);
  return s.alb / PI * diff * (vec3<f32>(1.0) - fe) + spec * fv + env * sky * fe;
}

// ---- one ray ----------------------------------------------------------------------------------------------

struct Seen { c: vec3<f32>, cg: f32, t: f32 };

fn haze(c: vec3<f32>, dist: f32) -> vec3<f32> {
  if (U.atm.w <= 0.0) { return c; }
  let tr = exp(-U.atm.w * dist);
  return c * tr + U.atm.rgb * (1.0 - tr);
}

fn background(rd: vec3<f32>, px: vec2<f32>) -> vec3<f32> {
  if (U.stage.z > 0.5) {
    var c = sky_dir(rd);
    if (U.envp.x < 0.5 && max(U.sun.r, max(U.sun.g, U.sun.b)) > 0.0) {
      // a glow round the key light
      c += U.sun.rgb * (0.02 * pow(max(dot(rd, U.sund.xyz), 0.0), 64.0));
    }
    if (U.atm.w > 0.0) { c = U.atm.rgb; }
    return c;
  }
  if (U.bg.w > 0.5) {
    let cs = max(U.sky.w, 4.0);
    let chk = (i32(floor(px.x / cs)) + i32(floor(px.y / cs))) & 1;
    return select(vec3<f32>(0.18), vec3<f32>(0.24), chk == 1);
  }
  return U.bg.rgb;
}

// A rope or a spring's wire at a hit on segment kp: round (its normal from its axis, not its flat sides), the rope
// twisted from strands that catch the light on their crowns and are dark in the grooves between them.
fn rope_surface(s0: Surf, kp: u32, pose: array<vec4<f32>, 2>, pl: vec4<f32>, row: i32, fw: f32) -> Surf {
  var s = s0;
  let P = PC[kp];
  if (row == LIGHTNING_ROW) {
    // lightning: all glow, brightest in the middle of its channel
    let qi = vec4<f32>(-pose[1].xyz, pose[1].w);
    let ql = quat_rotate(qi, s.p - pose[0].xyz);
    let rad = max(PL[u32(P.v.w)].w, 1e-5);
    let core = 1.0 - clamp(length(ql.xz) / rad, 0.0, 1.0);
    s.n = quat_rotate(pose[1], normalize(vec3<f32>(ql.x, 0.0, ql.z) + vec3<f32>(1e-6, 0.0, 0.0)));
    s.alb = vec3<f32>(0.0);
    s.f0 = vec3<f32>(0.0);
    s.rough = 1.0;
    s.em = P.r.xyz * (0.6 + 0.8 * core);
    return s;
  }
  let m = U.mat[row];
  let qi = vec4<f32>(-pose[1].xyz, pose[1].w);
  let ql = quat_rotate(qi, s.p - pose[0].xyz);
  let rad = max(PL[u32(P.v.w)].w, 1e-5);
  var nl = normalize(pl.xyz);
  let side = abs(nl.y) < 0.5;
  if (side) { nl = normalize(vec3<f32>(ql.x, 0.0, ql.z)); }
  var shade = 1.0;
  let strands = P.r.x;
  if (strands > 0.5 && side) {
    let ang = atan2(ql.z, ql.x);
    let along = ql.y + P.r.y;
    let pitch = max(P.r.z * rad, 1e-4);
    let ph = strands * (ang - 6.2831853 * along / pitch);
    // (faded out where a pixel is wider than a strand)
    let fade = clamp(2.0 - fw * strands / (3.0 * rad), 0.0, 1.0);
    let crown = 0.5 + 0.5 * cos(ph);
    shade = mix(0.82, 0.5 + 0.5 * sqrt(crown), fade);
    let tang = vec3<f32>(-sin(ang), 0.0, cos(ang));
    let grad = normalize(tang * (strands / rad) - vec3<f32>(0.0, 6.2831853 * strands / pitch, 0.0));
    nl = normalize(nl - grad * (0.45 * sin(ph) * fade));
    if (m.d.x < 0.5) {   // fibres
      shade *= 0.85 + 0.3 * fnoise(vec3<f32>(ang * rad, along, 0.0) * (60.0 / rad), fw * 60.0 / rad);
    }
  }
  s.n = quat_rotate(pose[1], nl);
  let metal = clamp(m.d.x, 0.0, 1.0);
  let base = m.c.rgb * shade;
  s.rough = clamp(m.c.w, 0.02, 1.0);
  s.alb = min(base, vec3<f32>(0.95)) * (1.0 - metal);
  s.f0 = mix(vec3<f32>(0.04), min(base, vec3<f32>(1.0)), metal);
  return s;
}

// A burnable surface as the fire has left it (b: its spot's state, burn_common.wgsl): browned where it is heating, black
// with char where it has burnt, flecked with grey ash where its fuel is nearly gone, glowing in cracks where it burns and
// a dull red where it smoulders.
fn burnt(s_in: Surf, b: vec4<f32>) -> Surf {
  return burnt_at(s_in, b, s_in.p);
}

// The same, its flecks and embers placed by q (a piece's own frame, so they stay on it as it falls).
fn burnt_at(s_in: Surf, b: vec4<f32>, q: vec3<f32>) -> Surf {
  var s = s_in;
  if (b.z < 0.5) { return s; }
  // (burning wood blackens at once; it goes on charring deeper as its fuel goes)
  let ch = max(burn_char(b), select(0.0, 0.8, burn_alight(b)));
  let toast = select(0.0, clamp(b.y, 0.0, 1.0), b.y > 0.0 && b.y < 1.0 && b.z < 1.5);
  s.alb = mix(s.alb, s.alb * vec3<f32>(0.45, 0.3, 0.14), 0.8 * toast);
  let fleck = hash31(floor(q / 0.012) + vec3<f32>(3.0, 1.0, 7.0));
  let ash = smoothstep(0.7, 1.0, ch) * smoothstep(0.65, 0.92, fleck) * 0.8;
  s.alb = mix(s.alb, mix(vec3<f32>(0.026, 0.023, 0.02), vec3<f32>(0.25, 0.24, 0.23), ash), ch);
  s.rough = mix(s.rough, 0.95, ch);
  s.f0 = mix(s.f0, vec3<f32>(0.02), ch);
  let flick = 0.75 + 0.5 * hash31(floor(q / 0.03) + vec3<f32>(floor(U.depth.z * 0.5), 5.0, 0.0));
  if (burn_alight(b)) {
    s.em += vec3<f32>(2.2, 0.55, 0.08) * (0.02 + 1.2 * smoothstep(0.78, 0.97, fleck)) * flick;
  } else if (b.w > 0.0) {
    s.em += vec3<f32>(0.8, 0.14, 0.02) * (b.w * smoothstep(0.75, 0.95, fleck)) * flick;
  }
  return s;
}

// The surface at a hit: where, which way it faces, and its material there.
fn surface_at(h: Hit, ro: vec3<f32>, rd: vec3<f32>, want: f32) -> Surf {
  var s: Surf;
  s.p = ro + rd * h.t;
  let fw = max(U.fit.w * h.t, 1e-5);
  s.eps = max(2.0 * fw, 5.0e-4);
  s.want = want;
  if (h.id == FLOOR) {
    s.n = vec3<f32>(0.0, 1.0, 0.0);
    let fl = floor_look(s.p.xz, fw / sqrt(max(abs(rd.y), 0.03)));
    s.alb = fl.rgb;
    s.rough = fl.w;
    if (U.gm.z > 0.0) {
      // under grass: in the shade of thick grass; where it has burnt, black with char and flecked with grey ash
      let uv = (s.p.xz - U.gm.xy) / U.gm.zw;
      if (all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0))) {
        let gg = textureSampleLevel(gmap, lin, uv, 0.0);
        s.alb *= 1.0 - 0.6 * gg.y;
        let ash = 0.7 * smoothstep(0.86, 0.97, hash31(vec3<f32>(floor(s.p.xz / 0.012), 7.0)));
        s.alb = mix(s.alb, mix(vec3<f32>(0.028, 0.025, 0.022), vec3<f32>(0.22, 0.21, 0.2), ash), 0.94 * gg.x);
        s.rough = mix(s.rough, 0.95, gg.x);
      }
    }
    if (U.bf.w > 0.5) {
      // the burnable floor (Spreading fire, Ground): its spot under here
      let cg = vec2<i32>(floor((s.p.xz - U.bf.xy) / U.bf.z));
      if (cg.x >= 0 && cg.y >= 0 && cg.x < i32(U.bf2.x) && cg.y < i32(U.bf2.y)) {
        s = burnt(s, textureLoad(burn, vec3<i32>(cg.x, 0, cg.y), 0));
      }
    }
    s.f0 = vec3<f32>(0.04);
    return s;
  }
  if (h.id == MATTER) {
    s.n = matter_normal(s.p);
    // (its field is not an exact distance: the trace can stop a little inside it. Out onto the surface, and shadow and
    // sky rays start a third of a node off it, so it does not shade itself)
    s.p = s.p - s.n * min(matter_d(s.p), 0.0);
    s.eps = max(s.eps, 0.33 * U.mo.w);
    let uvw = matter_uvw(s.p);
    let lk = textureSampleLevel(m_look, lin, uvw, 0.0);
    let ph = textureSampleLevel(m_phi, lin, uvw, 0.0);
    // grains: a fine mottle, and a few turned to glint
    let grain = 0.0025;
    let gq = s.p / max(grain, fw);
    let mot = fnoise(s.p / grain * 0.5, fw / grain * 0.5);
    s.alb = min(lk.rgb * (1.0 + 0.12 * mot), vec3<f32>(0.95)) * (1.0 - clamp(ph.y, 0.0, 1.0));
    s.rough = clamp(lk.a, 0.02, 1.0);
    s.f0 = vec3<f32>(0.04);
    // metal reflects in its own colour; hot matter glows as a blackbody at its temperature
    let gl = textureSampleLevel(m_glow, lin, uvw, 0.0);
    let metal = clamp(gl.y, 0.0, 1.0);
    s.f0 = mix(s.f0, min(lk.rgb, vec3<f32>(1.0)), metal);
    s.alb *= 1.0 - metal;
    if (U.mg.x > 0.5) { s.em += matter_glow(gl.x); }
    s.self_glow = 1.0;
    s.wrap = ph.w;
    let hr = vec3<f32>(hash31(gq), hash31(gq + vec3<f32>(17.0)), hash31(gq + vec3<f32>(41.0))) - vec3<f32>(0.5);
    s.gn = normalize(s.n + 0.7 * hr);
    s.glint = ph.z * select(0.0, 1.0, hash31(gq + vec3<f32>(73.0)) > 0.8) * (1.0 - smoothstep(0.004, 0.02, fw));
    return s;
  }
  if (h.id >= PIECE) {
    let kp = u32(h.id - PIECE);
    let P = PC[kp];
    let pose = piece_pose(kp);
    let pl = PL[u32(P.v.w) + u32(max(g_plane, 0))];
    let cut = length(pl.xyz) > 1.5;
    s.n = quat_rotate(pose[1], normalize(pl.xyz));
    let row = i32(P.o.w + 0.5);
    if (row >= ROPE_ROW) { return rope_surface(s, kp, pose, pl, row, fw); }
    let m = U.mat[row];
    // the pattern runs on through the pieces as it did through the whole object
    let qi = vec4<f32>(-pose[1].xyz, pose[1].w);
    let q = quat_rotate(qi, s.p - pose[0].xyz) + P.r.xyz;
    let pt = pattern(i32(m.d.z + 0.5), q, col_scale(U.col[row]), normalize(pl.xyz), fw);
    let metal = clamp(m.d.x, 0.0, 1.0);
    var base = mix(max(m.c.rgb * pt.rgb, vec3<f32>(0.0)), g_over.rgb, g_over.a);
    if (cut) { base = m.e.rgb * (0.85 + 0.3 * fnoise(q * 40.0, fw * 40.0)); }
    s.rough = clamp(select(m.c.w + pt.w, 0.9, cut), 0.02, 1.0);
    s.alb = min(base, vec3<f32>(0.95)) * ((1.0 - metal) * (1.0 - clamp(m.d.y, 0.0, 1.0)));
    s.f0 = mix(vec3<f32>(0.04), min(base, vec3<f32>(1.0)), metal);
    if (P.b.z > 0.5) { s = burnt_at(s, P.b, q); }   // (a piece of something that burns: its own fire)
    return s;
  }
  let k = obj(h.id);
  let m = U.mat[h.id];
  s.n = obj_normal(h.id, s.p, max(0.5 * fw, 2.0e-4));
  if (dot(s.n, rd) > 0.0 && m.d.y <= 0.0) { s.n = -s.n; }
  let q = col_to_local(k, s.p);
  let qn = normalize(col_to_local(k, s.p + s.n) - q);
  let pt = pattern(i32(m.d.z + 0.5), q, col_scale(k), qn, fw);
  let metal = clamp(m.d.x, 0.0, 1.0);
  let base = mix(max(m.c.rgb * pt.rgb, vec3<f32>(0.0)), g_over.rgb, g_over.a);
  s.rough = clamp(m.c.w + pt.w, 0.02, 1.0);
  s.alb = min(base, vec3<f32>(0.95)) * ((1.0 - metal) * (1.0 - clamp(m.d.y, 0.0, 1.0)));
  s.f0 = mix(vec3<f32>(0.04), min(base, vec3<f32>(1.0)), metal);
  if (U.bf2.z > 0.5 && i32(k.m2.w) >= 0) {
    // a burnable object: its spot just off the surface (its region of the object-burn atlas, in its own frame)
    let slot = slots[i32(k.m2.w)];
    s = burnt(s, obj_burn_at(k, slot, s.p + s.n * (0.75 * slot.dims.w)));
  }
  return s;
}

// Where a ray inside object i leaves it: its distance along the ray.
fn exit_t(i: i32, ro: vec3<f32>, rd: vec3<f32>, eps: f32) -> f32 {
  let k = obj(i);
  var t = 2.0 * eps;
  for (var j = 0; j < 96; j++) {
    let d = col_sdf(k, ro + rd * t);
    if (d > -eps) { return t; }
    t += max(-d, eps);
  }
  return t;
}

// Lightning's glow along a ray from ro (fire-local) as far as tmax: the light its channel scatters in the air round it.
// Each segment adds its glow over its length, falling off as 1 / (1 + (d / G)^2)^1.5 with the ray's distance d from it:
// summed along a long straight channel, that is 2 / (1 + (d / G)^2).
fn bolt_glow(ro: vec3<f32>, rd: vec3<f32>, tmax: f32) -> vec3<f32> {
  var g = vec3<f32>(0.0);
  let n = u32(U.bolt.y);
  if (n == 0u) { return g; }
  let G = U.bolt.z;
  let first = u32(U.bolt.x);
  for (var j = 0u; j < n; j++) {
    let P = PC[first + j];
    let pose = piece_pose(first + j);
    let half = PL[u32(P.v.w) + 8u].w;              // (its end plane: half its length)
    let ax = quat_rotate(pose[1], vec3<f32>(0.0, 1.0, 0.0));
    // the closest the ray (ro + t rd, 0 <= t <= tmax) and the segment (c + s ax, |s| <= half) come
    let w0 = ro - pose[0].xyz;
    let b = dot(rd, ax);
    let d = dot(rd, w0);
    let e = dot(ax, w0);
    let den = 1.0 - b * b;
    var t = 0.0;
    var s = 0.0;
    if (den > 1e-6) {
      t = clamp((b * e - d) / den, 0.0, tmax);
    }
    s = clamp(dot(ro + rd * t - pose[0].xyz, ax), -half, half);
    t = clamp(dot(pose[0].xyz + ax * s - ro, rd), 0.0, tmax);
    let dist = length(ro + rd * t - (pose[0].xyz + ax * s));
    let q = dist / G;
    g += P.r.xyz * ((2.0 * half / G) / pow(1.0 + q * q, 1.5));
  }
  return g * U.bolt.w;
}

// Past the CG along a ray (h: what it hit, if anything, that is not drawn): the footage (with the CG objects' shadows on
// it, for the camera's own ray), or the sky. t_foot: the footage's surface along the camera's ray; t_all: how far the
// ray has come since.
fn past_cg(ro: vec3<f32>, rd: vec3<f32>, h: Hit, ro0: vec3<f32>, px: vec2<f32>, puv: vec2<f32>, t_foot: f32, t_all: f32,
           first: bool) -> vec3<f32> {
  let footage = U.stage.w > 0.5;
  if (footage) {
    var catcher = 1.0;
    var pr = vec3<f32>(0.0);
    var nr = vec3<f32>(0.0, 1.0, 0.0);
    var have = false;
    if (h.id >= 0 && h.id < FLOOR) {
      pr = ro + rd * h.t;
      nr = obj_normal(h.id, pr, max(0.5 * U.fit.w * h.t, 2.0e-4));
      have = true;
    } else if (t_foot < 1.0e8) {
      pr = ro + rd * (t_foot - t_all);
      have = true;
      nr = vec3<f32>(0.0, 1.0, 0.0);
    } else if (U.floor_d.y > 0.5 && rd.y < -1e-6 && ro.y > U.stage.y) {
      pr = ro + rd * ((U.stage.y - ro.y) / rd.y);
      have = true;
    }
    if (have && first) {
      // the CG objects' shadows: how much of the light that lit it they take away
      let eps = max(2.0 * U.fit.w * length(pr - ro0), 5.0e-4);
      let po = pr + nr * eps;
      let sun_e = U.sun.rgb * max(dot(nr, U.sund.xyz), 0.0);
      let sky_e = PI * U.sky.rgb * (0.6 + 0.4 * nr.y);
      var lit = sky_e * ambient_occ(pr, nr, 2.0);
      if (max(sun_e.r, max(sun_e.g, sun_e.b)) > 0.0) { lit += sun_e * soft_shadow(po, U.sund.xyz, 1.0e4, U.sun.w, 2.0, 2.0 * eps); }
      catcher = luma(lit) / max(luma(sun_e + sky_e), 1e-6);
    }
    return footage_at(puv) * catcher;
  } else {
    return background(rd, px);
  }
}

// What the camera sees along one ray (fire-local), px the plate pixel, puv its plate uv, rd_w the ray
// in world axes.
fn see(ro0: vec3<f32>, rd0: vec3<f32>, px: vec2<f32>, puv: vec2<f32>, rd_w: vec3<f32>) -> Seen {
  if (U.lume.x > 0.5) { return lume_see(ro0, rd0, px, puv, rd_w); }
  let footage = U.stage.w > 0.5;
  t_piece = -1.0;
  t_footage = 1.0e9;
  var ro = ro0;
  var rd = rd0;
  var thr = vec3<f32>(1.0);
  var col = vec3<f32>(0.0);
  var t_all = 0.0;
  // the footage's own surfaces and matte hide what is behind them
  var t_foot = 1.0e9;
  var matte = 0.0;
  if (footage) {
    if (U.foot.w > 0.5) {
      let fd = footage_distance(puv, rd_w);
      if (fd > 0.0) { t_foot = max(fd - U.fwd.w / max(dot(rd_w, U.fwd.xyz), 1e-3), 0.0); }
    }
    t_footage = t_foot;
    if (U.foot.z > 0.5) { matte = clamp(textureSampleLevel(hold, lin, puv, 0.0).x, 0.0, 1.0); }
  }
  for (var bounce = 0; bounce < 3; bounce++) {
    let h = trace(ro, rd, 0.0, select(1.0e5, t_foot - t_all, footage), 1.0, !footage);
    var drawn = false;
    if (h.id == FLOOR || h.id >= PIECE || h.id == MATTER) { drawn = true; }
    if (h.id >= 0 && h.id < FLOOR) { drawn = U.mat[h.id].d.w > 1.5; }
    if ((h.id >= PIECE || h.id == MATTER) && bounce == 0) { t_piece = h.t; }
    if (!drawn) {
      // past the CG: the footage (with the CG objects' shadows on it), or the sky
      col += thr * past_cg(ro, rd, h, ro0, px, puv, t_foot, t_all, bounce == 0);
      break;
    }
    t_all += h.t;
    let s = surface_at(h, ro, rd, 1.0);
    let v = -rd;
    var c = shade(s, v) + s.em;
    let dist = length(s.p - ro0);
    if (h.id == FLOOR) {
      // far off, the floor fades into the sky at the horizon
      let fade = 1.0 - exp(-dist / max(U.floor_d.w, 1.0));
      c = mix(c, sky_dir(vec3<f32>(rd.x, 0.0, rd.z)) * 0.9, fade);
    }
    col += thr * haze(c, dist);
    var row = h.id;
    if (h.id >= PIECE) { row = i32(PC[u32(h.id - PIECE)].o.w + 0.5); }
    var clear = 0.0;
    var tint_c = vec3<f32>(1.0);
    var ior = 1.0;
    if (h.id == MATTER) {
      // jelly
      let uvw = matter_uvw(s.p);
      clear = clamp(textureSampleLevel(m_phi, lin, uvw, 0.0).y, 0.0, 1.0);
      tint_c = textureSampleLevel(m_look, lin, uvw, 0.0).rgb;
      ior = 1.35;
    } else if ((h.id < FLOOR || h.id >= PIECE) && row != LIGHTNING_ROW) {
      clear = clamp(U.mat[row].d.y, 0.0, 1.0);
      tint_c = U.mat[row].c.rgb;
      ior = max(U.mat[row].e.w, 1.0);
    }
    if (clear <= 0.0 || bounce == 2) { break; }
    // glass, ice, jelly: its reflection (above) and then the light refracted through it
    let n = s.n;
    let nv = max(dot(n, v), 1e-4);
    let fr = fresnel(vec3<f32>(pow((ior - 1.0) / (ior + 1.0), 2.0)), nv).x;
    var rin = refract(rd, n, 1.0 / ior);
    if (dot(rin, rin) < 0.5) { rin = rd; }
    let pin = s.p - n * s.eps;
    var tx = 0.0;
    var nout = vec3<f32>(0.0, 1.0, 0.0);
    if (h.id == MATTER) {
      tx = matter_exit_t(pin, rin, s.eps);
      nout = matter_normal(pin + rin * tx);
    } else if (h.id >= PIECE) {
      // a piece is a convex polyhedron: the refracted ray leaves it through one of its planes
      let kp = u32(h.id - PIECE);
      let ph = piece_hit(kp, pin, rin);
      tx = max(ph.t1, s.eps);
      let pose = piece_pose(kp);
      nout = quat_rotate(pose[1], normalize(PL[u32(PC[kp].v.w) + u32(max(ph.k1, 0))].xyz));
    } else {
      tx = exit_t(h.id, pin, rin, s.eps);
      nout = obj_normal(h.id, pin + rin * tx, s.eps);
    }
    let pout = pin + rin * tx;
    var rout = refract(rin, -nout, ior);
    if (dot(rout, rout) < 0.5) { rout = reflect(rin, -nout); }
    // tinted by its colour over the path through it (Beer-Lambert per 10 cm)
    let tint = pow(max(tint_c, vec3<f32>(1e-3)), vec3<f32>(tx / 0.1));
    thr *= clear * (1.0 - fr) * tint;
    ro = pout + nout * (2.0 * s.eps);
    rd = rout;
    t_all += tx;
  }
  var cg = 1.0;
  if (footage) {
    cg = 0.0;
    if (t_all > 0.0 && matte > 0.0) {
      // a real object in the footage's matte is in front of the CG
      col = mix(col, footage_at(puv), matte);
    }
    if (t_all > 0.0) { cg = 1.0 - matte; }
  }
  return Seen(col, cg, t_all);
}

// The element's pinhole uv for picture uv, through the footage's lens (as the composite's undistort).
fn undistort(uv: vec2<f32>, k1: f32) -> vec2<f32> {
  if (k1 == 0.0) { return uv; }
  let asp = vec2<f32>(U.res.x / U.res.y, 1.0);
  let hd = 0.5 * length(asp);
  let d = (uv - vec2<f32>(0.5)) * asp / hd;
  var u = d;
  for (var i = 0; i < 5; i++) { u = d / (1.0 + k1 * dot(u, u)); }
  return u * hd / asp + vec2<f32>(0.5);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let px = vec2<i32>(id.xy);
  if (f32(px.x) >= U.res.x || f32(px.y) >= U.res.y) { return; }
  let ns = max(i32(U.res.z + 0.5), 1);
  var acc = vec3<f32>(0.0);
  var cov = 0.0;
  var near_sum = 0.0;       // the pieces (or footage surfaces) the samples saw: their distance from the camera (m)
  var near_n = 0.0;
  let seed = u32(px.x) * 1973u + u32(px.y) * 9277u + u32(U.depth.z) * 26699u;
  let lume = U.lume.x > 0.5;
  var aov_a = vec4<f32>(0.0);
  var aov_n = vec4<f32>(0.0);
  if (lume) { lu_seed(px, 0u); }
  for (var i = 0; i < ns; i++) {
    // samples on a rotated grid over the pixel, each at its own time in the shutter (Lume: anywhere in the pixel and the
    // shutter, a new place every pass)
    var off = vec2<f32>(0.0);
    if (ns > 1) {
      off = vec2<f32>(fract(f32(i) * 0.7548776662 + 0.25), fract(f32(i) * 0.5698402910 + 0.6)) - vec2<f32>(0.5);
    }
    if (lume) { off = lu_rand2() - vec2<f32>(0.5); }
    if (lume) { g_jit = lu_rand(); } else { g_jit = rand1(seed + u32(i) * 31337u + 17u); }
    g_tau = 0.0;
    if (U.res.w > 0.0) {
      var jit = 0.0;
      if (lume) { jit = lu_rand(); } else { jit = rand1(seed + u32(i) * 7919u); }
      g_tau = U.res.w * ((f32(i) + jit) / f32(ns) - 0.5);
    }
    let pp = vec2<f32>(px) + vec2<f32>(0.5) + off;
    let puv = pp / U.res.xy;
    // the plate pixel's place in the picture, and through the lens to the element's pinhole camera
    let uv = undistort((puv - vec2<f32>(0.5)) / U.fit.xy + vec2<f32>(0.5), U.fit.z);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let pn = U.inv_vp * vec4<f32>(ndc, 0.0, 1.0);
    let pf = U.inv_vp * vec4<f32>(ndc, 1.0, 1.0);
    let ro_w = pn.xyz / pn.w;
    let rd_w = normalize(pf.xyz / pf.w - ro_w);
    let ro = (U.w2l * vec4<f32>(ro_w, 1.0)).xyz;
    let rd = normalize((U.w2l * vec4<f32>(rd_w, 0.0)).xyz);
    let s = see(ro, rd, pp, puv, rd_w);
    acc += s.c + bolt_glow(ro, rd, select(1.0e4, s.t, s.t > 0.0));
    if (lume) {
      aov_a += vec4<f32>(lu_alb, lu_dist);
      aov_n += vec4<f32>(lu_nrm, 0.0);
    }
    cov += s.cg;
    let to_near = U.fwd.w / max(dot(rd_w, U.fwd.xyz), 1e-3);   // camera to where the ray starts (m)
    var near = 1.0e9;
    if (t_piece >= 0.0) { near = t_piece + to_near; }
    if (t_footage < 1.0e8) { near = min(near, t_footage + to_near); }
    if (near < 1.0e8) {
      near_sum += near;
      near_n += 1.0;
    }
  }
  let inv = 1.0 / f32(ns);
  var outc = vec4<f32>(acc * inv, cov * inv);
  if (lume) {
    // this pass added to the ones before it (pass 0 starts afresh), and their average shown
    let k = u32(px.y) * u32(U.res.x) + u32(px.x);
    var a = outc;
    var al = aov_a * inv;
    var nn = vec4<f32>(aov_n.xyz * inv, pow(luma(acc * inv), 2.0));   // (w: the pass's brightness squared, for its noise)
    if (U.lume.y > 0.5) {
      a += ACC[k];
      al += AOV[2u * k];
      nn += AOV[2u * k + 1u];
    }
    ACC[k] = a;
    AOV[2u * k] = al;
    AOV[2u * k + 1u] = nn;
    outc = a / (U.lume.y + 1.0);
  }
  textureStore(out_plate, px, outc);
  var matte = 0.0;
  if (U.stage.w > 0.5 && U.foot.z > 0.5) {
    matte = clamp(textureSampleLevel(hold, lin, (vec2<f32>(px) + vec2<f32>(0.5)) / U.res.xy, 0.0).x, 0.0, 1.0);
  }
  // (a pixel holds out what is behind it where most of its samples saw something: its edge is where the CG's is)
  let held = near_n >= 0.5 * f32(ns);
  textureStore(out_hold, px, vec4<f32>(matte, select(0.0, near_sum / max(near_n, 1.0), held), 0.0, 0.0));
}
