// Lume: the light on the stage traced as it really travels (stage.wgsl takes this in place of its direct shading when
// Lighting engine is Lume).
//
// From each surface the camera sees, a path of light is followed back toward where it came from, bounce by bounce:
// - at every bounce, the light that comes straight from one of the lights (next-event estimation), picked by how much
//   light each brings here (and a share evenly): the key light, picked within the sun's disc (so shadows are as soft as
//   the sun is wide); one of the fire's point lights or one of the hot matter's, picked by how much light each brings,
//   toward a point within it (so each flame casts its own soft shadow, through the smoke); a lamp, within the solid
//   angle it covers; or the HDRI, picked by its brightness (a small bright sun in it is found at once); weighed against
//   finding it by bouncing (multiple importance sampling);
// - then the path bounces on: off the surface as its material scatters light (diffuse, or a GGX highlight sampled by
//   its visible normals), or through glass, ice and jelly (reflected or refracted by its Fresnel term, tinted over its
//   path through it), so light reaches the camera after bouncing from wall to floor to object;
// - a path that leaves the set sees the sky (or the HDRI), through the smoke overhead.
// Paths end by Russian roulette (from the fourth bounce) once they carry little light, and indirect light is capped
// (Clamp bright paths) so a stray glint does not leave a firefly. The random numbers are a low-discrepancy sequence
// (Owen-scrambled Sobol) for the camera and the first four surfaces, so a pixel's samples spread evenly.
//
// Caustics: the light a curved clear thing (a glass ball, an ice cube, jelly) or a mirror (bare smooth metal) focuses is
// all but never found by a path from the camera. So each pass also traces light paths from the lights (caustics): from
// a lamp, the key light or an HDRI with a sun in it, aimed at those things (the targets); through them (refracted and
// reflected by Fresnel, tinted) and off them; and where one lands on a surface, its light is sent to the camera (a
// straight line, nothing in the way: light tracing) and added to that pixel (CAU), and it goes on two diffuse bounces.
// The camera's paths leave exactly that light to them (lu_path): from surfaces seen straight from the camera, the lights
// found by way of the targets. Elsewhere (a surface seen through glass or in a mirror) the shadow rays go straight
// through curved glass, as through flat glass (a pane, a box, a shard), whose straight-through shadow is exact. The fire's
// lights keep the straight-through shadow through curved glass too.
//
// Every pass traces a path per pixel (more in a final render) at a new place in the pixel and the shutter; the stage
// adds the passes up (ACC) with the first surface's albedo, facing and distance, and each pass's brightness squared
// (AOV), for the denoiser (lume_denoise.wgsl).

const LU_INV_PI: f32 = 0.31830988;

var<private> lu_rng: u32;
var<private> lu_alb: vec3<f32>;    // the first surface's albedo (diffuse + highlight): the denoiser's guide
var<private> lu_nrm: vec3<f32>;    // its normal (0: the camera saw no surface)
var<private> lu_dist: f32;         // its distance from the camera (m)

fn lu_seed(px: vec2<i32>, i: u32) {
  lu_rng = pcg1(u32(px.x) * 1973u + u32(px.y) * 9277u + u32(U.lume.y + 0.5) * 26699u + i * 104729u + 7u);
}

// A PCG stream (O'Neill): the state steps as a linear congruential generator and each number is its permuted output, so the
// numbers are evenly spread in every dimension (hashing the last output instead clusters them); 24 bits, in [0, 1).
fn lu_pcg() -> f32 {
  lu_rng = lu_rng * 747796405u + 2891336453u;
  var w = ((lu_rng >> ((lu_rng >> 28u) + 4u)) ^ lu_rng) * 277803737u;
  w = (w >> 22u) ^ w;
  return f32(w >> 8u) * (1.0 / 16777216.0);
}

// ---- the random numbers: low-discrepancy where it counts ---------------------------------------------------------------
//
// A pixel's samples, over all its passes, are points of an Owen-scrambled Sobol sequence (Burley 2020: "Practical
// hash-based Owen scrambling"): however many there are, they spread evenly over the pixel, the lights and the directions
// a bounce can take, without the clumps and gaps of independent random numbers (on a set lit straight by a lamp, a
// fifteenth of the error after the same samples, as Mitsuba's stratified samplers have). Each random choice has its own
// dimension (a pair of them for a 2-d choice), its own scrambling and its own shuffle of the samples' order, so the
// dimensions do not line up with each other; the camera's come first (LU_DIMS_CAM), then LU_DIMS_PER for each surface
// along the path, each kept to one use whatever the path did before (lu_at). Past the first four surfaces, and for the
// caustics' light paths, PCG.

var<private> lu_n: u32;      // this sample's number in its pixel, over all its passes (its place in the sequences)
var<private> lu_pix: u32;    // the pixel's own scrambling
var<private> lu_dim: u32;    // the next random number's dimension (from LU_QMC_DIMS on: PCG)
var<private> lu_base: u32;   // the first dimension of the surface the path is on
var<private> lu_q: array<vec2<f32>, 6>;   // its numbers (lu_fill)
var<private> lu_nq: u32;                  // how many

// The camera's dimensions (+0 the place in the pixel, 2-d; +1 the soft shadows' jitter; +2 the time in the shutter), then
// each surface's: +0 clear or not; through glass +1.. its Fresnel choices; off a surface +1 which light and its sample
// (2-d), +2 the lobe and the bounce's direction (2-d), +3 and +4 the rest of the light's sample, +5 Russian roulette.
// Only those that count most are low-discrepancy (the camera's place in the pixel, and its time in the shutter with
// motion blur; a surface's first LU_FILL_PER): the rest are PCG (each low-discrepancy number costs, on every path).
const LU_DIMS_CAM: u32 = 3u;
const LU_DIMS_PER: u32 = 6u;
const LU_FILL_PER: u32 = 3u;
const LU_QMC_DIMS: u32 = 27u;  // (the camera's and 4 surfaces' worth)

// The numbers for the count dimensions from base on, all at once: one copy of the sequences' code in the kernel, not one
// for every random choice (which slowed every path by a fifth).
fn lu_fill(base: u32, count: u32) {
  lu_base = base;
  lu_dim = base;
  lu_nq = select(count, 0u, base >= LU_QMC_DIMS);
  for (var k = 0u; k < lu_nq; k++) { lu_q[k] = lu_qmc2(base + k); }
}

// This sample's numbers start: pixel px, sample n of it over all its passes.
fn lu_sample(px: vec2<i32>, n: u32) {
  lu_pix = lu_hash(u32(px.x) * 0x8da6b343u ^ u32(px.y) * 0xd8163841u ^ u32(U.depth.z + 0.5) * 0xcb1ab31fu);
  lu_n = n;
  lu_fill(0u, select(1u, LU_DIMS_CAM, U.res.w > 0.0));
}

// The next numbers are the path's surface j's (counting glass and every bounce), its k-th dimension.
fn lu_surface(j: i32) { lu_fill(LU_DIMS_CAM + LU_DIMS_PER * u32(j), LU_FILL_PER); }
fn lu_at(k: u32) { lu_dim = lu_base + k; }

fn lu_hash(x0: u32) -> u32 {   // (lowbias32, Wellons)
  var x = x0;
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  x ^= x >> 16u;
  return x;
}

// A random permutation of 32-bit numbers in which each bit depends only on itself and the bits below it (Laine and
// Karras, with Vegdahl's constants): applied to a number bit-reversed, Owen's nested uniform scrambling of its digits.
fn lu_lk(x0: u32, seed: u32) -> u32 {
  var x = x0;
  x ^= x * 0x3d20adeau;
  x += seed;
  x *= (seed >> 16u) | 1u;
  x ^= x * 0x05526c56u;
  x ^= x * 0x53a22864u;
  return x;
}

// Sobol's second dimension at index i, bit-reversed: its generator is Pascal's triangle mod 2, so each bit is the XOR of
// the index's bits at the positions that contain it (five shifts).
fn lu_sob1r(i: u32) -> u32 {
  var x = i;
  x ^= (x >> 1u) & 0x55555555u;
  x ^= (x >> 2u) & 0x33333333u;
  x ^= (x >> 4u) & 0x0f0f0f0fu;
  x ^= (x >> 8u) & 0x00ff00ffu;
  x ^= (x >> 16u) & 0x0000ffffu;
  return x;
}

// Dimension d's pair for this sample: the sample's index shuffled (Owen) for this dimension, Sobol's first two
// dimensions there, each scrambled (Owen) its own way. (One hash for the dimension; the three scramblings' seeds from it
// by multiplying: a hash for each cost a fifth of a path's time.)
fn lu_qmc2(d: u32) -> vec2<f32> {
  let seed = lu_hash(lu_pix ^ (d * 0x9e3779b9u + 0x632be5abu));
  let i = reverseBits(lu_lk(reverseBits(lu_n), seed));
  let x = reverseBits(lu_lk(i, seed * 0x2c1b3c6du + 0x297a2d39u));
  let y = reverseBits(lu_lk(lu_sob1r(i), seed * 0x7feb352du + 0x6a09e667u));
  return vec2<f32>(f32(x >> 8u), f32(y >> 8u)) * (1.0 / 16777216.0);
}

fn lu_rand2() -> vec2<f32> {
  let k = lu_dim - lu_base;
  if (k < lu_nq) {
    lu_dim += 1u;
    return lu_q[k];
  }
  let a = lu_pcg();
  return vec2<f32>(a, lu_pcg());
}

fn lu_rand() -> f32 {
  let k = lu_dim - lu_base;
  if (k < lu_nq) {
    lu_dim += 1u;
    return lu_q[k].x;
  }
  return lu_pcg();
}

// An orthonormal frame round n (Duff et al. 2017): (t, b, n).
fn lu_frame(n: vec3<f32>) -> mat3x3<f32> {
  let s = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (s + n.z);
  let b = n.x * n.y * a;
  return mat3x3<f32>(vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x), vec3<f32>(b, s + n.y * n.y * a, -n.y), n);
}

fn lu_to_local(f: mat3x3<f32>, v: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(dot(v, f[0]), dot(v, f[1]), dot(v, f[2]));
}

fn lu_cosine(u: vec2<f32>) -> vec3<f32> {
  let r = sqrt(u.x);
  let ph = 2.0 * PI * u.y;
  return vec3<f32>(r * cos(ph), r * sin(ph), sqrt(max(1.0 - u.x, 0.0)));
}

// A direction in the cone of half-angle acos(cmax) round the local z axis, uniformly.
fn lu_cone(u: vec2<f32>, cmax: f32) -> vec3<f32> {
  let c = 1.0 - u.x * (1.0 - cmax);
  let s = sqrt(max(1.0 - c * c, 0.0));
  let ph = 2.0 * PI * u.y;
  return vec3<f32>(s * cos(ph), s * sin(ph), c);
}

// A point in the unit ball, uniformly.
fn lu_ball() -> vec3<f32> {
  let d = lu_cone(lu_rand2(), -1.0);
  return d * pow(lu_rand(), 1.0 / 3.0);
}

// ---- the material: diffuse (Lambert) and a GGX highlight -----------------------------------------------------------

fn lu_d(nh: f32, a2: f32) -> f32 {
  let k = nh * nh * (a2 - 1.0) + 1.0;
  return a2 / max(PI * k * k, 1e-12);
}

fn lu_lambda(c: f32, a2: f32) -> f32 { return sqrt(a2 + (1.0 - a2) * c * c); }

fn lu_g1(c: f32, a2: f32) -> f32 { return 2.0 * c / max(c + lu_lambda(c, a2), 1e-9); }

fn lu_g2(nl: f32, nv: f32, a2: f32) -> f32 {
  return 2.0 * nl * nv / max(nl * lu_lambda(nv, a2) + nv * lu_lambda(nl, a2), 1e-9);
}

// A visible normal of a GGX surface of roughness a, seen from vl (local, z up): Dupuy and Benyoub 2023.
fn lu_vndf(vl: vec3<f32>, a: f32, u: vec2<f32>) -> vec3<f32> {
  let vh = normalize(vec3<f32>(a * vl.x, a * vl.y, vl.z));
  let ph = 2.0 * PI * u.x;
  let z = (1.0 - u.y) * (1.0 + vh.z) - vh.z;
  let st = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
  let h = vec3<f32>(st * cos(ph), st * sin(ph), z) + vh;
  return normalize(vec3<f32>(a * h.x, a * h.y, max(h.z, 1e-6)));
}

struct LuMat { alb: vec3<f32>, f0: vec3<f32>, a: f32, ps: f32, wrap: f32 };

fn lu_mat(s: Surf, nv: f32) -> LuMat {
  var m: LuMat;
  m.alb = s.alb;
  m.f0 = s.f0;
  let r = clamp(s.rough, 0.03, 1.0);
  m.a = r * r;
  m.wrap = s.wrap;
  // how often a bounce samples the highlight: as much as it reflects against what it scatters
  let fe = luma(fresnel_rough(s.f0, nv, r));
  let de = luma(s.alb) * (1.0 - fe);
  m.ps = clamp(fe / max(fe + de, 1e-6), 0.1, 0.95);
  if (de <= 1e-6) { m.ps = 1.0; }
  return m;
}

// Light from l reflected toward v (both local, z the normal), times the cosine, in its two parts: the diffuse (fd) and
// the highlight (fs), and each part's own pdf (pd by the cosine, ph by the visible normals), not yet weighed by the
// chance a bounce takes it (m.ps).
struct LuEv { fd: vec3<f32>, fs: vec3<f32>, pd: f32, ph: f32 };

fn lu_eval2(m: LuMat, vl: vec3<f32>, ll: vec3<f32>) -> LuEv {
  var e = LuEv(vec3<f32>(0.0), vec3<f32>(0.0), 0.0, 0.0);
  if (F_WATER && lu_med) {
    // a point in the water's murk: it scatters the same every way (its colour is in the path's throughput already)
    e.fd = m.alb * (0.25 * LU_INV_PI);
    e.pd = 0.25 * LU_INV_PI;
    return e;
  }
  let nv = max(vl.z, 1e-4);
  let nl = ll.z;
  // snow and grains let light wrap round a little past where they face away
  let nlw = (nl + m.wrap) / (1.0 + m.wrap);
  if (nlw <= 0.0) { return e; }
  let fv = fresnel_rough(m.f0, nv, sqrt(m.a));
  e.fd = m.alb * (LU_INV_PI * nlw) * (vec3<f32>(1.0) - fv);
  e.pd = max(nl, 0.0) * LU_INV_PI;
  if (nl > 0.0) {
    let h = normalize(vl + ll);
    let vh = max(dot(vl, h), 1e-4);
    let a2 = m.a * m.a;
    let D = lu_d(h.z, a2);
    let F = fresnel(m.f0, vh);
    e.fs = F * (D * lu_g2(nl, nv, a2) / (4.0 * nv));
    e.ph = D * lu_g1(nv, a2) / (4.0 * nv);
  }
  return e;
}

// The same, whole: (BSDF value, its pdf as sampled).
fn lu_eval(m: LuMat, vl: vec3<f32>, ll: vec3<f32>) -> vec4<f32> {
  let e = lu_eval2(m, vl, ll);
  return vec4<f32>(e.fd + e.fs, (1.0 - m.ps) * e.pd + m.ps * e.ph);
}

// ---- lights -----------------------------------------------------------------------------------------------------------

// Whether nothing that stops light lies between p and p + d * tmax (1 open, 0 hidden).
fn lu_open(p: vec3<f32>, d: vec3<f32>, tmax: f32, eps: f32, want: f32) -> f32 {
  g_opaque = true;
  let h = trace(p, d, 2.0 * eps, tmax, want, true);
  g_opaque = false;
  return select(1.0, 0.0, h.id >= 0);
}

// Whether caustics are traced this pass (lume.wgsl caustics): then curved clear things stop the shadow rays of the lights
// traced through them, and their light through them comes in as caustics instead.
fn lu_caustics_on() -> bool { return U.lume3.x > 0.5; }

// Whether the HDRI's light is traced for caustics (bit 17 of U.lume3.w: its light is gathered in a small bright part of
// it, a sun, lume.py env_peaked). Else its light through curved glass and off mirrors is found by bouncing, as an even
// sky's is found well that way (a light path from it would be a waste: Mitsuba's paths found that light faster).
fn lu_env_cau() -> bool { return (u32(U.lume3.w + 0.5) & 0x20000u) != 0u; }

// Whether the fire's light is traced for caustics too (bit 18 of U.lume3.w: stage.py, when the fire lights the set and
// its light is shadowed: without shadow rays, nothing would leave the light through the glass to the caustics).
fn lu_fire_cau() -> bool {
  return (u32(U.lume3.w + 0.5) & 0x40000u) != 0u && light_count[0] > 0u && max(U.fire.r, max(U.fire.g, U.fire.b)) > 0.0
         && U.fire.w > 0.0;
}

// Whether h is on one of the things the caustics' light paths are aimed at (U.lume3.w: a bit for each object row, bit
// 16 the matter; lume.py caustic_targets). Only those focus light as caustics: others the camera's paths find as before.
fn lu_target(h: Hit) -> bool {
  let mask = u32(U.lume3.w + 0.5);
  if (h.id == MATTER) { return (mask & 0x10000u) != 0u; }
  if (h.id >= 0 && h.id < FLOOR) { return (mask & (1u << u32(h.id))) != 0u; }
  return false;
}

// Curved glass whose light comes as caustics.
fn lu_xglass(h: Hit) -> bool { return lu_curved(h) && lu_target(h); }

const LU_MIRROR_ROUGH: f32 = 0.35;   // metal this smooth focuses light as caustics (lume.py MIRROR_ROUGH)
const LU_SHARP: f32 = 0.15;   // a mirror at least this rough (glossy) also takes the light paths that come to it off other
                              // mirrors or through glass, where the camera sees it straight: a sharp highlight seen in its
                              // blurred reflection, found by bouncing, is a sparkle

// Whether surface s, where h hit, is a mirror whose light comes as caustics: bare metal (nothing diffuse), smooth, one
// of the targets. (Its highlight seen from a diffuse surface is a lamp shrunk to a point: a path from the camera all but
// never finds it, and when it does, a firefly.)
fn lu_mirror(h: Hit, s: Surf) -> bool {
  if (h.id < 0 || h.id >= FLOOR || !lu_target(h) || U.mat[h.id].d.y > 0.0) { return false; }
  return luma(s.alb) <= 1e-6 && s.rough < LU_MIRROR_ROUGH;
}

// Whether surface s, where h hit, is a clear coat over colour whose highlight focuses light as caustics: a sharp highlight
// on something not bare metal (plastic, paint, lacquer), one of the targets. Its highlight is taken as a mirror's, its
// colour as a matte surface's (lu_path, caustics: split by lobe).
fn lu_coat(h: Hit, s: Surf) -> bool {
  if (h.id < 0 || h.id >= FLOOR || !lu_target(h) || U.mat[h.id].d.y > 0.0) { return false; }
  return luma(s.alb) > 1e-6 && s.rough < LU_MIRROR_ROUGH;
}

const LU_ROUGH_D: f32 = 0.35;   // rough enough that light landing on it from any way spreads evenly (lu_rough)

// Whether surface s is rough: a light path that lands on it may go on, bounced off it, and land again, and be sent to
// the camera from there. A glossy one (smoother) only ends a light path's landings: sent to the camera from it, light
// that landed there from a random way is a sparkle (its peaked highlight), so the camera's paths that reach a surface
// by way of it take that light from the caustic cache instead.
fn lu_rough(s: Surf) -> bool { return s.rough >= LU_ROUGH_D; }

// Whether what h hit bends light into a caustic: a ball, a cylinder, a mesh, the matter (jelly). Not the floor, a box or a
// piece of something broken, whose flat faces only shift the light (their straight-through shadow is exact).
fn lu_curved(h: Hit) -> bool {
  if (h.id == MATTER) { return true; }
  if (h.id >= 0 && h.id < FLOOR) {
    let sh = i32(U.col[h.id].a.w + 0.5);
    return sh == 0 || sh == 2 || sh == 3;
  }
  return false;
}

// How much light gets from p to p + d * tmax: 0 past anything that stops light; through glass, ice and jelly what their
// surfaces let in and out (Fresnel, straight through: a slab's shadow exactly, a ball's without its focused caustic) and
// their tint over the way through. caustic: 0 at curved glass among the targets (the light it focuses comes as
// caustics: lu_xglass). glow_r > 0: the light is a patch of hot matter that size, tmax away: the ray may end on the
// hot matter itself (the patch, within reach of it: that is where the light comes from), and nothing else may stand in
// the way.
//
// (One trace() in it, in a loop: every trace() written in Lume is a copy of it in the kernel, and the copies slow every
// path whether they run or not: four kinds of light with two traces each made Lume half again as slow.)
fn lu_shadow(p: vec3<f32>, d: vec3<f32>, tmax: f32, eps: f32, want: f32, caustic: bool, glow_r: f32) -> f32 {
  var tr = 1.0;
  g_wcol = vec3<f32>(1.0);
  var q = p;
  var left = tmax;
  for (var j = 0; j < 5; j++) {
    // first anything that stops light (glass, ice and jelly let through), then the clear things on the way, one at a time
    g_opaque = j == 0;
    let h = trace(q, d, 2.0 * eps, left, want, true);
    g_opaque = false;
    if (j == 0) {
      if (h.id >= 0) {
        return select(0.0, 1.0, glow_r > 0.0 && h.id == MATTER && h.t > tmax - 2.0 * glow_r - 4.0 * eps);
      }
      // a lamp is a solid bulb or panel: it shades what is behind it (but not the light picked on its own surface, at tmax)
      for (var k = 0; k < i32(U.ln.w); k++) {
        let t = lu_lamp_t(p, d, k);
        if (t > 2.0 * eps && t < tmax * (1.0 - 1e-3) - eps) { return 0.0; }
      }
      if (water_on() && glow_r <= 0.0) {
        // through the water's surfaces, straight (their Fresnel terms), and its absorption (its colour: g_wcol); the key
        // light on the floor by the caustic map (lume_water.wgsl w_shadow)
        tr = w_shadow(p, d, tmax, eps, lu_sun_floor);
        if (tr <= 0.0) { return 0.0; }
      }
      if (U.lume2.z < 0.5 || glow_r > 0.0) { return tr; }   // (nothing clear in the set)
      continue;
    }
    if (h.id < 0 || h.id == FLOOR) { break; }
    let ph = q + d * h.t;
    let cl = lu_clear(h, ph);
    if (cl.clear <= 0.0) { return 0.0; }
    if (caustic && lu_xglass(h)) { return 0.0; }   // (its light through it is traced from the light: caustics)
    let n = lu_hit_normal(h, ph, eps);
    let fin = lu_fresnel(max(abs(dot(n, d)), 1e-4), cl.ior);
    let ex = lu_exit(h, ph + d * eps, d, eps);
    let co = max(abs(dot(ex.xyz, d)), 1e-4);
    let fout = lu_fresnel(co, 1.0 / cl.ior);
    // (its tint over the way through: the grey here, its colour into g_wcol, put back by lu_direct, as the water's)
    let tc = pow(max(cl.tint, vec3<f32>(1e-3)), vec3<f32>(ex.w / 0.1));
    let tg = max(luma(tc), 1e-8);
    g_wcol *= tc / tg;
    tr *= cl.clear * (1.0 - fin) * (1.0 - fout) * tg;
    if (tr < 1e-4) { return 0.0; }
    let adv = h.t + ex.w + 2.0 * eps;
    q = q + d * adv;
    left -= adv;
    if (left <= 0.0) { break; }
  }
  return tr;
}

// The outward normal of what h hit at p.
fn lu_hit_normal(h: Hit, p: vec3<f32>, eps: f32) -> vec3<f32> {
  if (F_MARCH && h.id == MATTER) { return matter_normal(p); }
  if (F_SHOTS && h.id == FRAY) { return fray_normal(p, eps); }
  if (F_PIECES && h.id >= PIECE) {
    let kp = u32(h.id - PIECE);
    let pose = piece_pose(kp);
    return quat_rotate(pose[1], normalize(PL[u32(PC[kp].v.w) + u32(max(g_plane, 0))].xyz));
  }
  if (plain(h.id)) { return plain_normal(h.id, p); }
  return obj_normal(h.id, p, eps);
}

// A sphere light of radius r at c, seen from p: (the cosine of its half-angle, the solid angle it covers), or
// (-2, 0) from inside it.
fn lu_cone_of(p: vec3<f32>, c: vec3<f32>, r: f32) -> vec2<f32> {
  let dc = c - p;
  let s2 = r * r / max(dot(dc, dc), 1e-12);
  if (s2 >= 1.0) { return vec2<f32>(-2.0, 0.0); }
  let cmax = sqrt(1.0 - s2);
  return vec2<f32>(cmax, 2.0 * PI * s2 / (1.0 + cmax));   // (2 pi (1 - cmax), without the cancellation)
}

// How far along l from p the sphere (c, r) is, or -1.
fn lu_sphere_t(p: vec3<f32>, l: vec3<f32>, c: vec3<f32>, r: f32) -> f32 {
  let oc = p - c;
  let b = dot(oc, l);
  let h = b * b - (dot(oc, oc) - r * r);
  if (h < 0.0) { return -1.0; }
  let sq = sqrt(h);
  if (-b - sq > 0.0) { return -b - sq; }
  if (-b + sq > 0.0) { return -b + sq; }
  return -1.0;
}

// The radiance of lamp k seen along l (toward it): a bulb's intensity spread over its disc, through a spot's cone or by its
// light profile (lamp_shape.wgsl); an area light's over its panel, as bright from every way in front (its intensity
// falling with the cosine as the panel is seen edge on), dark behind.
fn lu_lamp_radiance(k: i32, l: vec3<f32>) -> vec3<f32> {
  let lm = lamps[k];
  if (lu_panel(k)) {
    let ax = lamps[LAMP_MAX + k];
    return lm.c.rgb * (U.depth.w / max(4.0 * ax.p.w * ax.c.w, 1e-8)) * select(0.0, 1.0, dot(l, lm.d.xyz) < 0.0);
  }
  let r = max(lm.p.w, 1e-3);
  return lm.c.rgb * (U.depth.w / (PI * r * r)) * lamp_shape(k, -l);
}

// Lamp k is a panel (an area light with a size: renderer.lamp_rows).
fn lu_panel(k: i32) -> bool {
  return i32(lamps[k].c.w + 0.5) == 2 && lamps[LAMP_MAX + k].p.w > 0.0;
}

// How far along d from p lamp k is (its bulb, or its panel from either side), or -1.
fn lu_lamp_t(p: vec3<f32>, d: vec3<f32>, k: i32) -> f32 {
  let lm = lamps[k];
  if (!lu_panel(k)) { return lu_sphere_t(p, d, lm.p.xyz, max(lm.p.w, 1e-3)); }
  let ax = lamps[LAMP_MAX + k];
  let dn = dot(d, lm.d.xyz);
  if (abs(dn) < 1e-9) { return -1.0; }
  let t = dot(lm.p.xyz - p, lm.d.xyz) / dn;
  let q = p + d * t - lm.p.xyz;
  if (t <= 0.0 || abs(dot(q, ax.p.xyz)) > ax.p.w || abs(dot(q, ax.c.xyz)) > ax.c.w) { return -1.0; }
  return t;
}

// A panel seen from p, as the spherical rectangle it covers (Urena, Fajardo and King 2013, "An Area-Preserving
// Parametrization for Spherical Rectangles"): its corner's coordinates in the panel's frame from p, and the solid angle S.
struct LuRect { o: vec3<f32>, x: vec3<f32>, y: vec3<f32>, z: vec3<f32>, x0: f32, x1: f32, y0: f32, y1: f32, z0: f32,
                b0: f32, b1: f32, k: f32, s: f32 };

const LU_RECT_MIN: f32 = 2e-3;   // (smaller (far off, edge on): the panel is sampled by area, the rectangle's sums lose digits)

fn lu_rect(p: vec3<f32>, k: i32) -> LuRect {
  let lm = lamps[k];
  let ax = lamps[LAMP_MAX + k];
  var r: LuRect;
  r.o = p;
  r.x = ax.p.xyz;
  r.y = ax.c.xyz;
  r.z = cross(r.x, r.y);
  let d = lm.p.xyz - r.x * ax.p.w - r.y * ax.c.w - p;
  r.x0 = dot(d, r.x);
  r.y0 = dot(d, r.y);
  r.z0 = dot(d, r.z);
  if (r.z0 > 0.0) {
    r.z0 = -r.z0;
    r.z = -r.z;
  }
  r.x1 = r.x0 + 2.0 * ax.p.w;
  r.y1 = r.y0 + 2.0 * ax.c.w;
  let v00 = vec3<f32>(r.x0, r.y0, r.z0);
  let v01 = vec3<f32>(r.x0, r.y1, r.z0);
  let v10 = vec3<f32>(r.x1, r.y0, r.z0);
  let v11 = vec3<f32>(r.x1, r.y1, r.z0);
  let n0 = normalize(cross(v00, v10));
  let n1 = normalize(cross(v10, v11));
  let n2 = normalize(cross(v11, v01));
  let n3 = normalize(cross(v01, v00));
  let g0 = acos(clamp(-dot(n0, n1), -1.0, 1.0));
  let g1 = acos(clamp(-dot(n1, n2), -1.0, 1.0));
  let g2 = acos(clamp(-dot(n2, n3), -1.0, 1.0));
  let g3 = acos(clamp(-dot(n3, n0), -1.0, 1.0));
  r.b0 = n0.z;
  r.b1 = n2.z;
  r.k = 2.0 * PI - g2 - g3;
  r.s = g0 + g1 - r.k;
  return r;
}

// The point on the panel along a direction picked evenly over the solid angle it covers (u in the unit square).
fn lu_rect_point(r: LuRect, u: vec2<f32>) -> vec3<f32> {
  let au = u.x * r.s + r.k;
  let fu = (cos(au) * r.b0 - r.b1) / sin(au);
  let cu = clamp(select(-1.0, 1.0, fu > 0.0) / sqrt(fu * fu + r.b0 * r.b0), -1.0, 1.0);
  let xu = clamp(-(cu * r.z0) / max(sqrt(1.0 - cu * cu), 1e-7), r.x0, r.x1);
  let dd = sqrt(xu * xu + r.z0 * r.z0);
  let h0 = r.y0 / sqrt(dd * dd + r.y0 * r.y0);
  let h1 = r.y1 / sqrt(dd * dd + r.y1 * r.y1);
  let hv = h0 + u.y * (h1 - h0);
  let hv2 = hv * hv;
  let yv = select(r.y1, hv * dd / sqrt(max(1.0 - hv2, 1e-12)), hv2 < 1.0 - 1e-6);
  return r.o + r.x * xu + r.y * yv + r.z * r.z0;
}

// The pdf (over solid angle) of lu_ls_lamp picking direction d from p, toward lamp k, where it meets it t away.
fn lu_lamp_pdf(p: vec3<f32>, d: vec3<f32>, t: f32, k: i32) -> f32 {
  let lm = lamps[k];
  if (!lu_panel(k)) {
    let cone = lu_cone_of(p, lm.p.xyz, max(lm.p.w, 1e-3));
    return select(0.0, 1.0 / max(cone.y, 1e-12), cone.x > -1.5);
  }
  let r = lu_rect(p, k);
  if (r.s > LU_RECT_MIN) { return 1.0 / r.s; }
  let ax = lamps[LAMP_MAX + k];
  return t * t / max(4.0 * ax.p.w * ax.c.w * abs(dot(d, lm.d.xyz)), 1e-12);
}

// The HDRI as a picture to pick directions from: (its pdf over the picture, which pixel), and back.
fn lu_env_dir(uv: vec2<f32>) -> vec3<f32> {
  let ph = (uv.x - 0.5) * 2.0 * PI;
  let th = uv.y * PI;
  let r = vec3<f32>(sin(th) * sin(ph), cos(th), -sin(th) * cos(ph));
  let c = cos(U.envp.y);
  let s = sin(U.envp.y);
  let dw = vec3<f32>(c * r.x + s * r.z, r.y, -s * r.x + c * r.z);
  let c2 = cos(U.floor_d.z);
  let s2 = sin(U.floor_d.z);
  return vec3<f32>(c2 * dw.x - s2 * dw.z, dw.y, s2 * dw.x + c2 * dw.z);
}

fn lu_env_uv(d: vec3<f32>) -> vec2<f32> {
  let dw = to_world_dir(d);
  let c = cos(U.envp.y);
  let s = sin(U.envp.y);
  let r = vec3<f32>(c * dw.x - s * dw.z, dw.y, s * dw.x + c * dw.z);
  return vec2<f32>(atan2(r.x, -r.z) * (0.5 / PI) + 0.5, acos(clamp(r.y, -1.0, 1.0)) / PI);
}

fn lu_env_on() -> bool { return U.envp.x > 0.5 && U.lume2.x > 0.5; }

// The pdf (per steradian) of picking direction d from the HDRI.
fn lu_env_pdf(d: vec3<f32>) -> f32 {
  let w = u32(U.lume2.x);
  let h = u32(U.lume2.y);
  let uv = lu_env_uv(d);
  let x = min(u32(uv.x * f32(w)), w - 1u);
  let y = min(u32(uv.y * f32(h)), h - 1u);
  let st = sin(uv.y * PI);
  return ENV[h + w * h + y * w + x] / max(2.0 * PI * PI * st, 1e-6);
}

// A direction picked from the HDRI by its brightness, and its pdf (per steradian).
fn lu_env_pick(u: vec2<f32>) -> vec4<f32> {
  let w = u32(U.lume2.x);
  let h = u32(U.lume2.y);
  // the row (marginal), then the pixel in it (conditional): binary searches in their cumulative sums
  var lo = 0u;
  var hi = h - 1u;
  while (lo < hi) {
    let m = (lo + hi) / 2u;
    if (ENV[m] < u.y) { lo = m + 1u; } else { hi = m; }
  }
  let y = lo;
  let row = h + y * w;
  lo = 0u;
  hi = w - 1u;
  while (lo < hi) {
    let m = (lo + hi) / 2u;
    if (ENV[row + m] < u.x) { lo = m + 1u; } else { hi = m; }
  }
  let x = lo;
  let jit = lu_rand2();
  let uv = vec2<f32>((f32(x) + jit.x) / f32(w), (f32(y) + jit.y) / f32(h));
  let d = lu_env_dir(uv);
  let st = sin(uv.y * PI);
  let pdf = ENV[h + w * h + y * w + x] / max(2.0 * PI * PI * st, 1e-6);
  return vec4<f32>(d, pdf);
}

fn lu_power(pdf_a: f32, pdf_b: f32) -> f32 {
  let a = pdf_a * pdf_a;
  return a / max(a + pdf_b * pdf_b, 1e-20);
}

// One light's sample at a surface (lu_direct): the direction to it (l), how far its shadow ray goes (tmax; 0: none),
// its light over the chance of this sample, before the material and the shadow (c), the pdf (per steradian) to weigh it
// against a bounce finding it by (pdf; 0: none can), and what it is (kind: 0 the key light, 1 the fire, 2 hot matter,
// 3 a lamp, 4 the HDRI; k: which lamp; r: a glowing patch's radius).
struct LuLs { l: vec3<f32>, tmax: f32, c: vec3<f32>, kind: i32, k: i32, r: f32, pdf: f32 };

// The key light: a direction within the sun's disc.
fn lu_ls_sun(sel: f32, u: vec2<f32>) -> LuLs {
  var ls: LuLs;
  if (max(U.sun.r, max(U.sun.g, U.sun.b)) <= 0.0) { return ls; }
  let cmax = cos(atan(1.0 / max(U.sun.w, 1.0)));
  let sf = lu_frame(U.sund.xyz);
  let lc = lu_cone(u, cmax);
  ls.l = sf[0] * lc.x + sf[1] * lc.y + sf[2] * lc.z;
  ls.c = U.sun.rgb / sel;
  ls.tmax = 1.0e4;
  return ls;
}

// The fire: one of its point lights, picked by how much light it brings here, toward a point within it.
fn lu_ls_fire(s: Surf, po: vec3<f32>, sel: f32, u: vec2<f32>) -> LuLs {
  var ls: LuLs;
  ls.kind = 1;
  let nf = light_count[0];
  if (nf == 0u || max(U.fire.r, max(U.fire.g, U.fire.b)) <= 0.0) { return ls; }
  var tot = 0.0;
  for (var k = 0u; k < nf; k++) {
    let a = lights[2u * k];
    let d = a.xyz - s.p;
    tot += luma(lights[2u * k + 1u].rgb) / (dot(d, d) + a.w * a.w);
  }
  if (tot <= 0.0) { return ls; }
  let pick = u.x * tot;
  var run = 0.0;
  var k = 0u;
  for (; k < nf - 1u; k++) {
    let a = lights[2u * k];
    let d = a.xyz - s.p;
    run += luma(lights[2u * k + 1u].rgb) / (dot(d, d) + a.w * a.w);
    if (run >= pick) { break; }
  }
  let a = lights[2u * k];
  let d0 = a.xyz - s.p;
  let wk = luma(lights[2u * k + 1u].rgb) / (dot(d0, d0) + a.w * a.w);
  let q = a.xyz + lu_ball() * a.w;
  let dv = q - po;
  let dist = length(dv);
  if (dist <= 1e-4 || wk <= 0.0) { return ls; }
  ls.l = dv / dist;
  let rad = lights[2u * k + 1u].rgb / (dot(d0, d0) + a.w * a.w);
  ls.c = U.fire.rgb * rad * (tot / (wk * sel));
  ls.tmax = select(0.0, max(dist - a.w, 0.0), U.fire.w > 0.0);   // (Fire shadows off: none)
  return ls;
}

// Hot matter: one of its glowing patches (it does not light itself).
fn lu_ls_glow(s: Surf, po: vec3<f32>, sel: f32, u: vec2<f32>) -> LuLs {
  var ls: LuLs;
  ls.kind = 2;
  let nm = u32(ML[0].x);
  if (nm == 0u || s.self_glow > 0.5) { return ls; }
  var tot = 0.0;
  for (var k = 0u; k < nm; k++) {
    let a = ML[1u + 2u * k];
    let d = a.xyz - s.p;
    tot += luma(ML[2u + 2u * k].rgb) / (dot(d, d) + a.w * a.w);
  }
  if (tot <= 0.0) { return ls; }
  let pick = u.x * tot;
  var run = 0.0;
  var k = 0u;
  for (; k < nm - 1u; k++) {
    let a = ML[1u + 2u * k];
    let d = a.xyz - s.p;
    run += luma(ML[2u + 2u * k].rgb) / (dot(d, d) + a.w * a.w);
    if (run >= pick) { break; }
  }
  let a = ML[1u + 2u * k];
  let d0 = a.xyz - s.p;
  let wk = luma(ML[2u + 2u * k].rgb) / (dot(d0, d0) + a.w * a.w);
  let q = a.xyz + lu_ball() * a.w;
  let dv = q - po;
  let dist = length(dv);
  if (dist <= 1e-4 || wk <= 0.0) { return ls; }
  ls.l = dv / dist;
  ls.c = ML[2u + 2u * k].rgb / (dot(d0, d0) + a.w * a.w) * (tot / (wk * sel));
  ls.tmax = dist;
  ls.r = a.w;
  return ls;
}

// Lamp k: a sphere, a direction picked within the solid angle it covers (its soft shadow exactly), weighed against a
// bounce finding it (lu_path).
fn lu_ls_lamp(k: i32, s: Surf, po: vec3<f32>, sel: f32, u: vec2<f32>) -> LuLs {
  var ls: LuLs;
  ls.kind = 3;
  ls.k = k;
  let lm = lamps[k];
  if (lu_panel(k)) {
    // a panel: a point on it along a direction picked evenly over the solid angle it covers (its soft shadow with
    // little noise however near), or by its area when it is small to the eye; none from behind it
    if (dot(s.p - lm.p.xyz, lm.d.xyz) <= 0.0) { return ls; }
    let rc = lu_rect(po, k);
    var q: vec3<f32>;
    if (rc.s > LU_RECT_MIN) {
      q = lu_rect_point(rc, u);
    } else {
      let ax = lamps[LAMP_MAX + k];
      q = lm.p.xyz + ax.p.xyz * ((2.0 * u.x - 1.0) * ax.p.w) + ax.c.xyz * ((2.0 * u.y - 1.0) * ax.c.w);
    }
    let dv = q - po;
    let dist = length(dv);
    if (dist <= 1e-6) { return ls; }
    ls.l = dv / dist;
    ls.pdf = sel * select(dist * dist / max(4.0 * lamps[LAMP_MAX + k].p.w * lamps[LAMP_MAX + k].c.w * abs(dot(ls.l, lm.d.xyz)),
                                            1e-12), 1.0 / rc.s, rc.s > LU_RECT_MIN);
    ls.c = lu_lamp_radiance(k, ls.l) / max(ls.pdf, 1e-12);
    ls.tmax = dist;
    return ls;
  }
  let r = max(lm.p.w, 1e-3);
  let cone = lu_cone_of(s.p, lm.p.xyz, r);   // (the solid angle as the surface itself sees it)
  if (cone.x < -1.5) {
    // inside the lamp: its light all round, as if from its middle
    let dc = lm.p.xyz - s.p;
    ls.l = normalize(dc + vec3<f32>(0.0, 1e-6, 0.0));
    ls.c = lu_lamp_radiance(k, ls.l) * (PI * r * r / (dot(dc, dc) + r * r)) / sel;
    return ls;
  }
  let lf = lu_frame(normalize(lm.p.xyz - s.p));
  let lc = lu_cone(u, cone.x);
  ls.l = lf[0] * lc.x + lf[1] * lc.y + lf[2] * lc.z;
  ls.pdf = sel / max(cone.y, 1e-12);
  ls.c = lu_lamp_radiance(k, ls.l) / ls.pdf;
  ls.tmax = max(lu_sphere_t(po, ls.l, lm.p.xyz, r), 0.0);
  return ls;
}

// The HDRI: a direction picked by its brightness, weighed against finding it by bouncing.
fn lu_ls_env(sel: f32, u: vec2<f32>) -> LuLs {
  var ls: LuLs;
  ls.kind = 4;
  if (!lu_env_on()) { return ls; }
  let pk = lu_env_pick(u);
  if (pk.w <= 0.0) { return ls; }
  ls.l = pk.xyz;
  ls.pdf = sel * pk.w;
  ls.c = sky_dir(ls.l) / ls.pdf;
  ls.tmax = 1.0e4;
  return ls;
}

// How much light slot j of the lights (0 the key light, 1 the fire, 2 hot matter, 3.. the lamps, then the HDRI) brings
// to a surface at p facing n, roughly (unshadowed, its irradiance, a little even where it is behind): what lu_direct
// picks one by. 0: there is none (or it does not light this surface).
fn lu_wt(j: i32, p: vec3<f32>, n: vec3<f32>, self_glow: f32) -> f32 {
  let nl = i32(U.ln.w);
  if (j == 0) {
    if (max(U.sun.r, max(U.sun.g, U.sun.b)) <= 0.0) { return 0.0; }
    return max(luma(U.sun.rgb), 1e-9) * (max(dot(n, U.sund.xyz), 0.0) + 0.05);
  }
  if (j == 1) {
    let nf = light_count[0];
    if (nf == 0u || max(U.fire.r, max(U.fire.g, U.fire.b)) <= 0.0) { return 0.0; }
    var tot = 0.0;
    for (var k = 0u; k < nf; k++) {
      let a = lights[2u * k];
      let d = a.xyz - p;
      tot += luma(lights[2u * k + 1u].rgb) / (dot(d, d) + a.w * a.w);
    }
    return luma(U.fire.rgb) * tot;
  }
  if (j == 2) {
    let nm = u32(ML[0].x);
    if (nm == 0u || self_glow > 0.5) { return 0.0; }
    var tot = 0.0;
    for (var k = 0u; k < nm; k++) {
      let a = ML[1u + 2u * k];
      let d = a.xyz - p;
      tot += luma(ML[2u + 2u * k].rgb) / (dot(d, d) + a.w * a.w);
    }
    return tot;
  }
  if (j < 3 + nl) {
    let lm = lamps[j - 3];
    let d = lm.p.xyz - p;
    let d2 = dot(d, d);
    let r = max(lm.p.w, 1e-3);
    if (lu_panel(j - 3) && dot(d, lm.d.xyz) >= 0.0) { return 0.0; }   // (behind a panel)
    let w = max(luma(lm.c.rgb) * U.depth.w, 1e-9) / max(d2, r * r) * (max(dot(n, d) / sqrt(max(d2, 1e-12)), 0.0) + 0.05);
    return w * (lamp_shape(j - 3, -d / sqrt(max(d2, 1e-12))) + 0.05);   // (a spot's cone, an area light's facing, a profile)
  }
  if (!lu_env_on()) { return 0.0; }
  return max(U.lume2.w, 1e-9);   // (the HDRI's light on an upward surface: stage.py)
}

const LU_SEL_EVEN: f32 = 0.2;   // of the chance to pick each light, this share is spread evenly over the lights there are

// The lights' weights (lu_wt) at the surface the path is on, their sum and how many there are: lu_direct sets them, and the
// bounce off the same surface weighs what it finds against them (one copy of lu_wt in the kernel, see lu_shadow).
var<private> lu_w: array<f32, 12>;   // (3 + the stage's 8 lamps + 1)
var<private> lu_wtot: f32;
var<private> lu_wcnt: f32;
var<private> lu_cov: bool;   // this surface's light from the lamps, the key light and the HDRI comes as caustics (lu_path)
var<private> lu_cau_sh: bool;
var<private> lu_coat_on: bool;   // the surface is a coat over colour (lu_coat): its parts weighed apart (lu_direct)
var<private> lu_med: bool;
var<private> lu_keep0: vec3<f32> = vec3<f32>(1.0);   // (what the camera ray's own scattering in the water kept: lume_see)        // the path is at a point in the water's murk, not on a surface (lume_water.wgsl)
var<private> lu_on_floor: bool;   // the surface the path is on is the floor (the key light on it under water: w_shadow)
var<private> lu_sun_floor: bool;  // ... and this shadow ray is the key light's
var<private> lu_coat_ex: bool;   // ... and its highlight's light from the traced lights comes as caustics   // its shadow rays stop at curved glass (its light through that comes as caustics): else straight through

// The chance lu_direct picked slot j at this surface: by lu_wt, with a share evenly (so a light it rates low, that this
// surface's highlight would show bright, is still picked often enough).
fn lu_sel(j: i32) -> f32 {
  let w = lu_w[j];
  if (w <= 0.0) { return 0.0; }
  return (1.0 - LU_SEL_EVEN) * w / lu_wtot + LU_SEL_EVEN / lu_wcnt;
}

// Per-light passes (F_LPASS): the light a camera path brings back, split by where it came from (0 the key light, 1 the sky,
// 2 the fire, hot matter and anything glowing, 3 the lamps), and how much of it the pixel shows (lu_lp_k: what the floor's
// fade, the haze, the footage and a background colour over it leave). lu_direct leaves its light's group (lu_dg) and the
// share it took from the caustic cache (lu_dc: split by U.lp, as the caustics are).
var<private> lu_lp: array<vec3<f32>, 4>;
var<private> lu_lp_k: f32;
var<private> lu_dg: i32;
var<private> lu_dc: vec3<f32>;

fn lu_lp_add(g: i32, c: vec3<f32>) {
  if (F_LPASS) { lu_lp[g] += c; }
}

// Light arriving straight from the lights at surface s (seen from v), times its BSDF: one of the lights (the key light,
// the fire, hot matter, a lamp, the HDRI) picked by how much light each brings here (lu_sel), a sample of it, and one
// shadow ray (one copy of it in the kernel, see lu_shadow) for how much of it gets here. (One light a bounce, as Mitsuba
// does: a shadow ray for every light each bounce took Lume twice as long a path for little less noise.)
fn lu_direct(s: Surf, m: LuMat, fr: mat3x3<f32>, vl: vec3<f32>, cached: bool, depth: i32) -> vec3<f32> {
  let po = s.p + s.n * s.eps;
  let pl = light_cell(s.p);
  let nl = i32(U.ln.w);
  lu_wtot = 0.0;
  lu_wcnt = 0.0;
  for (var i = 0; i < 4 + nl; i++) {
    let w = lu_wt(i, s.p, s.n, s.self_glow);
    lu_w[i] = w;
    if (w > 0.0) {
      lu_wtot += w;
      lu_wcnt += 1.0;
    }
  }
  if (lu_wcnt <= 0.0) { return vec3<f32>(0.0); }
  // one 2-d number picks the light (its x) and, rescaled within that light's share, is the light's own sample: so the
  // samples each light gets keep their even spread (two numbers, one to pick and one to sample, would leave each light
  // a random part of its sequence)
  lu_at(1u);
  let us = lu_rand2();
  var run = 0.0;
  var start = 0.0;
  var j = -1;
  var sel = 0.0;
  for (var i = 0; i < 4 + nl; i++) {
    if (lu_w[i] <= 0.0) { continue; }
    j = i;
    sel = lu_sel(i);
    start = run;
    run += sel;
    if (us.x < run) { break; }
  }
  let u2 = vec2<f32>(clamp((us.x - start) / max(sel, 1e-12), 0.0, 0.99999994), us.y);
  var ls: LuLs;
  lu_at(3u);
  if (j == 0) {
    ls = lu_ls_sun(sel, u2);
  } else if (j == 1) {
    ls = lu_ls_fire(s, po, sel, u2);
  } else if (j == 2) {
    ls = lu_ls_glow(s, po, sel, u2);
  } else if (j < 3 + nl) {
    ls = lu_ls_lamp(j - 3, s, po, sel, u2);
  } else {
    ls = lu_ls_env(sel, u2);
  }
  let traced = ls.kind == 0 || ls.kind == 3 || (ls.kind == 4 && lu_env_cau()) || (ls.kind == 1 && lu_fire_cau());   // (the caustics' lights)
  if (F_LPASS) {
    lu_dg = select(select(select(2, 3, ls.kind == 3), 1, ls.kind == 4), 0, ls.kind == 0);
    lu_dc = vec3<f32>(0.0);
  }
  if (lu_cov && traced) { ls.c = vec3<f32>(0.0); }
  // the light's sample, then the caustic cache's light here (cached: lu_path), each through the material: one
  // evaluation of it in the kernel for both (each copy of it slows every path)
  var out = vec3<f32>(0.0);
  var ce = vec3<f32>(0.0);
  for (var it = 0; it < 2; it++) {
    var l = ls.l;
    if (it == 1) {
      if (!cached) { break; }
      let cg = lu_cache_get(s, depth);
      ce = cg.e;
      if (max(ce.r, max(ce.g, ce.b)) <= 0.0) { break; }
      l = cg.l;
    } else if (max(ls.c.r, max(ls.c.g, ls.c.b)) <= 0.0) {
      continue;
    }
    var ll = lu_to_local(fr, l);
    if (it == 1 && ll.z <= 0.02) { ll = vec3<f32>(0.0, 0.0, 1.0); }   // (its light from no clear way: as from above)
    let e = lu_eval2(m, vl, ll);
    if (it == 1) {
      // (a coat's colour only: its highlight's caustic light, as a mirror's, is the camera's own)
      let cc = select(e.fd + e.fs, e.fd, lu_coat_on) / max(ll.z, 1e-4) * ce;
      out += cc;
      if (F_LPASS) { lu_dc = cc; }
      continue;
    }
    var c: vec3<f32>;
    if (lu_coat_on) {
      var wd = 1.0;
      var wh = 1.0;
      if (ls.pdf > 0.0) {
        wd = lu_power(ls.pdf, (1.0 - m.ps) * e.pd);
        wh = lu_power(ls.pdf, m.ps * e.ph);
      }
      c = ls.c * (e.fd * wd + select(e.fs * wh, vec3<f32>(0.0), lu_coat_ex && traced));
    } else {
      c = ls.c * (e.fd + e.fs);
      if (ls.pdf > 0.0) { c *= lu_power(ls.pdf, (1.0 - m.ps) * e.pd + m.ps * e.ph); }
    }
    if (max(c.r, max(c.g, c.b)) <= 0.0) { continue; }
    var vis = 1.0;
    if (ls.tmax > 0.0) {
      // (through curved glass: the caustics' lights stop there where the caustics have their light, else go straight
      // through; the HDRI not traced stops there always, and a bounce finds it through the glass; the fire's and hot
      // matter's go straight through, no caustics from them)
      lu_sun_floor = ls.kind == 0 && lu_on_floor;
      vis = lu_shadow(po, ls.l, ls.tmax, s.eps, s.want, (traced && lu_cau_sh) || (ls.kind == 4 && !lu_env_cau()),
                      select(0.0, ls.r, ls.kind == 2));
    }
    // the smoke on the way
    if (ls.kind == 1) {
      if (U.fire.w > 0.0) { vis = mix(1.0, vis * smoke_tr(po, ls.l, ls.tmax), U.fire.w); }
    } else if (vis > 0.0) {
      if (ls.kind == 0) {
        let ps = pl + s.n * 1.5 + ls.l * 0.75;
        if (in_light(ps)) { vis *= samp_c(L0, lin, ps, U.ln.xyz).a; }
      } else if (ls.kind == 3) {
        let pq = pl + s.n * 1.5 + ls.l * 0.75;
        if (lamps[ls.k].e.y > 0.5 && in_light(pq)) { vis *= lamp_tr(pq, ls.k); }
      } else if (ls.kind == 4) {
        let pk2 = pl + s.n * 1.5;
        if (in_light(pk2)) { vis *= samp_c(L1, lin, pk2, U.ln.xyz).x; }
      }
    }
    out += c * vis * g_wcol;   // (through water: in its colour)
    g_wcol = vec3<f32>(1.0);
  }
  return out;
}

// The sky a path sees when it leaves the set at p going d, after a bounce whose pdf was pdf (0: the camera's own ray)
// off a surface where lu_direct picks the HDRI with chance sel.
fn lu_sky(p: vec3<f32>, d: vec3<f32>, pdf: f32, sel: f32) -> vec3<f32> {
  var c = sky_dir(d);
  if (lu_env_on() && pdf > 0.0) {
    c *= lu_power(pdf, sel * lu_env_pdf(d));
  }
  // through the smoke overhead
  let pk = light_cell(p);
  if (in_light(pk)) { c *= samp_c(L1, lin, pk, U.ln.xyz).x; }
  return c;
}

// ---- glass, ice and jelly ------------------------------------------------------------------------------------------------

struct LuClear { clear: f32, ior: f32, tint: vec3<f32> };

// ---- dispersion (Lume › Dispersion: U.lu4.x, how many times the materials' own) ------------------------------------------
// A path takes one wavelength at its first refraction (lu_ior), picked in 10 nm bins by how much each adds to the picture
// (LU_LAM_CDF: the bins' brightness in the working space, under daylight), and carries its colour over the chance of
// picking it (LU_LAM_W: the bin's rgb over its pick's pdf; over the picks they average to white). Then every refraction
// on the path bends it by the index at that wavelength.
const LU_LAM_W = array<vec3<f32>, 35>(
  vec3<f32>(0.0000, 0.1154, 2.8846), vec3<f32>(0.0341, 0.0004, 2.9655), vec3<f32>(0.3984, 0.0000, 2.6016),
  vec3<f32>(0.4102, 0.0000, 2.5898), vec3<f32>(0.2826, 0.0000, 2.7174), vec3<f32>(0.2692, 0.0000, 2.7308),
  vec3<f32>(0.2553, 0.0000, 2.7447), vec3<f32>(0.1042, 0.0000, 2.8958), vec3<f32>(0.0000, 0.0086, 2.9914),
  vec3<f32>(0.0000, 0.2907, 2.7093), vec3<f32>(0.0000, 0.9636, 2.0364), vec3<f32>(0.0000, 1.8333, 1.1667),
  vec3<f32>(0.0000, 2.5531, 0.4469), vec3<f32>(0.0000, 2.9570, 0.0430), vec3<f32>(0.0000, 3.0000, 0.0000),
  vec3<f32>(0.0000, 3.0000, 0.0000), vec3<f32>(0.0000, 3.0000, 0.0000), vec3<f32>(0.2345, 2.7655, 0.0000),
  vec3<f32>(0.9511, 2.0489, 0.0000), vec3<f32>(1.5757, 1.4243, 0.0000), vec3<f32>(2.1046, 0.8954, 0.0000),
  vec3<f32>(2.5474, 0.4526, 0.0000), vec3<f32>(2.8999, 0.1001, 0.0000), vec3<f32>(3.0000, 0.0000, 0.0000),
  vec3<f32>(3.0000, 0.0000, 0.0000), vec3<f32>(3.0000, 0.0000, 0.0000), vec3<f32>(3.0000, 0.0000, 0.0000),
  vec3<f32>(3.0000, 0.0000, 0.0000), vec3<f32>(3.0000, 0.0000, 0.0000), vec3<f32>(3.0000, 0.0000, 0.0000),
  vec3<f32>(2.9008, 0.0992, 0.0000), vec3<f32>(2.5578, 0.4422, 0.0000), vec3<f32>(2.0933, 0.9067, 0.0000),
  vec3<f32>(1.4902, 1.5098, 0.0000), vec3<f32>(0.7421, 2.2579, 0.0000));
const LU_LAM_CDF = array<f32, 36>(0.00000, 0.00040, 0.00157, 0.00589, 0.02065, 0.05718, 0.11506, 0.17827, 0.23693, 0.28567,
  0.32243, 0.35031, 0.37545, 0.40294, 0.43637, 0.47744, 0.52145, 0.56434, 0.60635, 0.65359, 0.70496, 0.75823, 0.81065,
  0.85928, 0.90284, 0.93851, 0.96446, 0.98130, 0.99106, 0.99610, 0.99842, 0.99939, 0.99979, 0.99993, 0.99998, 1.00000);

var<private> lu_lam: f32 = 0.0;       // the path's wavelength (nm), once it has one (0: white still)
var<private> lu_lam_w: vec3<f32> = vec3<f32>(1.0);   // its colour, for the path to take on (lu_lam_take)

// The index of refraction n_d (at 587.6 nm) at the path's wavelength, picking one if it has none yet. Cauchy's formula
// n = A + B / lambda^2 through n_d with the Abbe number of glass like it (water and ice 55, crown glass 59, flint glass
// and gems 36), its spread times U.lu4.x.
fn lu_ior(n_d: f32) -> f32 {
  if (U.lu4.x <= 0.0 || n_d <= 1.0001) { return n_d; }
  if (lu_lam <= 0.0) {
    let u = lu_rand();
    var i = 0;
    for (; i < 34; i++) {
      if (u < LU_LAM_CDF[i + 1]) { break; }
    }
    let f = clamp((u - LU_LAM_CDF[i]) / max(LU_LAM_CDF[i + 1] - LU_LAM_CDF[i], 1e-9), 0.0, 1.0);
    lu_lam = 380.0 + 10.0 * (f32(i) + f);
    lu_lam_w = LU_LAM_W[i];
  }
  let v = select(select(36.0, 59.0, n_d < 1.6), 55.0, n_d < 1.45);
  let b = (n_d - 1.0) / v / (1.0 / (0.4861 * 0.4861) - 1.0 / (0.6563 * 0.6563)) * U.lu4.x;
  let l = lu_lam * 1e-3;
  return n_d + b * (1.0 / (l * l) - 1.0 / (0.5876 * 0.5876));
}

// The colour the path takes on with its wavelength, once (then white).
fn lu_lam_take() -> vec3<f32> {
  let w = lu_lam_w;
  lu_lam_w = vec3<f32>(1.0);
  return w;
}

fn lu_clear(h: Hit, p: vec3<f32>) -> LuClear {
  var c = LuClear(0.0, 1.0, vec3<f32>(1.0));
  var row = h.id;
  if (h.id >= PIECE) { row = i32(PC[u32(h.id - PIECE)].o.w + 0.5); }
  if (F_MARCH && h.id == MATTER) {
    let uvw = matter_uvw(p);
    c.clear = clamp(textureSampleLevel(m_phi, lin, uvw, 0.0).y, 0.0, 1.0);
    c.tint = textureSampleLevel(m_look, lin, uvw, 0.0).rgb;
    c.ior = 1.35;
  } else if (h.id != FLOOR && (h.id < FLOOR || h.id >= PIECE) && row != LIGHTNING_ROW) {
    c.clear = clamp(U.mat[row].d.y, 0.0, 1.0);
    c.tint = U.mat[row].c.rgb;
    c.ior = max(U.mat[row].e.w, 1.0);
  }
  return c;
}

fn lu_fresnel(ci: f32, eta: f32) -> f32 {
  // unpolarised Fresnel reflectance of a dielectric, entering a medium eta times as dense
  let st2 = (1.0 - ci * ci) / (eta * eta);
  if (st2 >= 1.0) { return 1.0; }
  let ct = sqrt(1.0 - st2);
  let rs = (ci - eta * ct) / (ci + eta * ct);
  let rp = (eta * ci - ct) / (eta * ci + ct);
  return 0.5 * (rs * rs + rp * rp);
}

// Where a ray inside what was hit (h) leaves it, and the normal there (pointing out).
fn lu_exit(h: Hit, pin: vec3<f32>, rin: vec3<f32>, eps: f32) -> vec4<f32> {
  var tx = 0.0;
  var nout = vec3<f32>(0.0, 1.0, 0.0);
  if (F_MARCH && h.id == MATTER) {
    tx = matter_exit_t(pin, rin, eps);
    nout = matter_normal(pin + rin * tx);
  } else if (F_PIECES && h.id >= PIECE) {
    let kp = u32(h.id - PIECE);
    let ph = piece_hit(kp, pin, rin);
    tx = max(ph.t1, eps);
    let pose = piece_pose(kp);
    nout = quat_rotate(pose[1], normalize(PL[u32(PC[kp].v.w) + u32(max(ph.k1, 0))].xyz));
  } else if (plain(h.id) || !F_MARCH) {
    tx = max(plain_span(obj(h.id), pin, rin).y, eps);   // (a plain shape's far side, exactly: not marched to)
    nout = plain_normal(h.id, pin + rin * tx);
  } else {
    tx = exit_t(h.id, pin, rin, eps);
    nout = obj_normal(h.id, pin + rin * tx, eps);
  }
  return vec4<f32>(nout, tx);
}

// ---- one path ----------------------------------------------------------------------------------------------------------

// What a path finds where it hits h (surface_at), with what Lume needs exact: a plain shape's normal (not its distance
// field's slope a pixel round, which rounds its edges that wide), and the rays off a surface found exactly (the floor, a
// plain shape, a piece) a hair's offset, not the stage's pixel or two (which would put their start that much nearer the
// lights).
fn lu_surf(h: Hit, ro: vec3<f32>, rd: vec3<f32>) -> Surf {
  if (F_WATER && h.id == MEDIUM) {
    // a point in the water's murk: no surface, a scatterer facing back the way the path came
    var m: Surf;
    m.p = ro + rd * h.t;
    m.n = -rd;
    m.alb = vec3<f32>(1.0);
    m.f0 = vec3<f32>(0.0);
    m.rough = 1.0;
    m.eps = 1.0e-5;
    m.want = 1.0;
    m.gn = m.n;
    return m;
  }
  g_exact_n = true;   // (a plain shape's exact normal: plain_normal)
  var s = surface_at(h, ro, rd, 1.0);
  if (h.id == FLOOR || h.id >= PIECE || (h.id >= 0 && h.id < FLOOR && plain(h.id))) { s.eps = min(s.eps, 1.0e-4); }
  return s;
}

// Whether a hit is on something drawn in CG (else the footage or the sky is seen there).
fn lu_drawn(h: Hit) -> bool {
  if (h.id == FLOOR || h.id >= PIECE || h.id == MATTER) { return true; }
  if (F_WATER && (h.id == WATER || h.id == MEDIUM)) { return true; }
  if (F_SHOTS && h.id == FRAY) { return true; }   // (a splinter standing out of a bullet's way out of wood: marks.wgsl)
  if (h.id >= 0 && h.id < FLOOR) { return U.mat[h.id].d.w > 1.5; }
  return false;
}

// The light coming back along a camera ray that first hits h (drawn), traced on from there.
fn lu_path(h0: Hit, ro0: vec3<f32>, rd0: vec3<f32>) -> vec3<f32> {
  var h = h0;
  var ro = ro0;
  var rd = rd0;
  var thr = vec3<f32>(1.0);
  var L = vec3<f32>(0.0);
  var pdf_last = 0.0;      // the pdf of the bounce that led here (0: the camera, or glass)
  let bounces = max(i32(U.lume.z + 0.5), 1);
  let cap = U.lume.w;
  var depth = 0;
  var inner = 0;           // reflections inside glass so far
  // Which light this path leaves to others. The caustics' light paths (caustics, below) carry light from the lamps, the
  // key light and an HDRI with a sun that meets mirrors or curved glass (the targets) and lands on a surface: sent to the
  // camera in a straight line where it can see that surface (and on up to LU_CAU_BOUNCES more bounces off such surfaces),
  // and left in the caustic cache where the light first lands. So at every diffuse or glossy surface, this path leaves
  // to them the lights it would find by way of mirrors and curved glass: where it has met only such surfaces seen
  // straight from the camera, the light paths' splats have that light; anywhere else (a surface seen through glass or in
  // a mirror, or past the light paths' bounces) it takes it from the cache (a caustic seen through its glass or in a
  // mirror: a path from the camera to a lamp that way, found by bouncing, is a firefly).
  var clean = true;        // every surface so far, from the camera, a diffuse or glossy one: no glass, no mirror
  var nd = 0;              // how many
  var prough = true;       // ... and all rough (lu_rough): the light paths go on from a landing only off rough surfaces
  var cov_d = false;       // the last diffuse or glossy surface is one the caustics' light paths land on
  var xs = 0;              // mirrors and curved glass (targets) since that surface
  var glass_s = false;     // through glass since the last bounce off a surface: the lights found then, the shadow rays
  var glass_x = false;     // had (straight through), or the caustics have (x: curved glass among the targets)
  var scattered = false;   // the path has been reflected or refracted (a lamp is seen in glass, not by the camera itself)
  // in the water (lume_water.wgsl): what it crosses absorbs, its murk scatters (a camera under the surface starts in it)
  var in_water = water_on() && w_phi(ro0) < 0.0;
  if (in_water) { thr *= exp(-w_sigma() * h0.t); }
  thr *= lu_keep0;   // (the camera's first look scattered in the water's murk, bubbles or spray: lume_see)
  lu_keep0 = vec3<f32>(1.0);
  for (var guard = 0; guard < 64; guard++) {
    lu_surface(guard);
    var s = lu_surf(h, ro, rd);
    let v = -rd;
    // (with a background colour behind the set, the floor is there only under the water: seen through the water past it,
    // the camera sees the background, as the liquid's own renderer draws it)
    if (F_WATER && scattered && depth == 0 && h.id == FLOOR && !in_water && U.stage.z < 0.5 && water_on()) {
      L += thr * select(U.bg.rgb, vec3<f32>(0.21), U.bg.w > 0.5);
      break;
    }
    if (depth == 0) {
      lu_nrm = s.n;
      lu_alb = s.alb + s.f0;
      lu_dist = length(s.p - ro0);
    }
    // light given off: lightning always; hot matter only as seen (its glow already lights the rest as its patches)
    if (depth == 0 || s.self_glow < 0.5) {
      L += thr * s.em;
      lu_lp_add(2, thr * s.em);
    }
    // glass, ice, jelly: reflected or refracted at random by Fresnel, tinted over its path through it
    lu_med = F_WATER && h.id == MEDIUM;
    lu_on_floor = h.id == FLOOR;
    // the water's surface: foam on it (a white, rough surface: below, as any), or the surface itself, reflecting or
    // refracting by the exact Fresnel term at a microfacet of its roughness
    var through_water = false;
    if (F_WATER && h.id == WATER) {
      if (depth == 0) { lu_alb = vec3<f32>(1.0); }
      let nn = select(-s.n, s.n, dot(s.n, v) > 0.0);   // (the normal on the side the ray came from)
      let fc = w_foam(s.p);
      if (fc > 0.0 && lu_rand() < fc) {
        s.n = nn;
        s.alb = U.wat[8].xyz;
        s.f0 = vec3<f32>(0.02);
        s.rough = 0.9;
        if (depth == 0) {
          lu_alb = s.alb + s.f0;
          lu_nrm = s.n;
        }
      } else {
        through_water = true;
        clean = false;
        let into = dot(s.n, v) > 0.0;   // (from the air)
        var mn = nn;
        if (s.rough > 0.002) {
          let f0 = lu_frame(nn);
          let hl = lu_vndf(lu_to_local(f0, v), max(s.rough * s.rough, 1e-4), lu_rand2());
          mn = normalize(f0[0] * hl.x + f0[1] * hl.y + f0[2] * hl.z);
        }
        let wn = lu_ior(w_ior());
        thr *= lu_lam_take();
        let eta = select(1.0 / wn, wn, into);   // (how many times denser the other side is)
        var refl = lu_rand() < lu_fresnel(max(dot(v, mn), 1e-4), eta);
        var rt = vec3<f32>(0.0);
        if (!refl) {
          rt = refract(rd, mn, 1.0 / eta);
          refl = dot(rt, rt) < 0.5;   // (all of it reflected inside)
        }
        if (refl) {
          rd = reflect(rd, mn);
          if (dot(rd, nn) <= 0.0) { rd = reflect(rd, nn); }
          ro = s.p + nn * s.eps;
        } else {
          rd = normalize(rt);
          if (dot(rd, nn) >= 0.0) { rd = rd - 2.0 * dot(rd, nn) * nn; }
          ro = s.p - nn * s.eps;
          in_water = into;
          glass_s = true;   // (the lights past it come through the shadow rays, straight: w_shadow)
        }
        pdf_last = 0.0;
        scattered = true;
      }
    }
    var cl = lu_clear(h, s.p);
    if (F_SHOTS) { cl.clear *= 1.0 - g_frost; }   // (glass crushed or cracked by a bullet: frosted, marks.wgsl)
    if (through_water) {
      // (done: on to what it meets)
    } else if (cl.clear > 0.0 && lu_rand() < cl.clear) {
      let n = s.n;
      let ci = max(dot(n, v), 1e-4);
      cl.ior = lu_ior(cl.ior);
      thr *= lu_lam_take();
      let fr = lu_fresnel(ci, cl.ior);
      var go_on = true;
      clean = false;
      let xg = lu_xglass(h);
      if (xg) { xs += 1; }
      if (lu_rand() < fr) {
        rd = reflect(rd, n);
        ro = s.p + n * s.eps;
      } else {
        glass_x = glass_x || xg;
        glass_s = glass_s || !xg;
        // through it: in, across (bouncing inside while the light cannot get out), and out
        var rin = refract(rd, n, 1.0 / cl.ior);
        if (dot(rin, rin) < 0.5) { rin = rd; }
        var pin = s.p - n * s.eps;
        var out = false;
        for (var j = 0; j < 4; j++) {
          let ex = lu_exit(h, pin, rin, s.eps);
          let nout = ex.xyz;
          let pout = pin + rin * ex.w;
          thr *= pow(max(cl.tint, vec3<f32>(1e-3)), vec3<f32>(ex.w / 0.1));
          let co = max(dot(-rin, -nout), 1e-4);
          let fo = lu_fresnel(co, 1.0 / cl.ior);
          let rout = refract(rin, -nout, cl.ior);
          if (dot(rout, rout) > 0.5 && lu_rand() >= fo) {
            rd = rout;
            ro = pout + nout * (2.0 * s.eps);
            out = true;
            break;
          }
          rin = reflect(rin, -nout);
          pin = pout - nout * (2.0 * s.eps);
          inner += 1;
        }
        if (!out) { go_on = false; }
      }
      if (!go_on) { break; }
      pdf_last = 0.0;
      scattered = true;
    } else {
      // an opaque surface (or the part of a clear one that is not): its light from the lights, then a bounce
      var sv = s;
      if (cl.clear > 0.0) { sv.alb = s.alb / max(1.0 - cl.clear, 1e-3); }
      let fr = lu_frame(sv.n);
      let vl = lu_to_local(fr, v);
      if (vl.z <= 0.0) { break; }
      var m = lu_mat(sv, vl.z);
      if (lu_med) { m.ps = 0.0; }
      var cached = false;   // its caustic light from the cache (lu_direct)
      // a coat over colour (lu_coat) is its colour (a diffuse or glossy surface) or its highlight (a mirror), as its
      // bounce takes one or the other: its colour's state is kept aside until then (coat_*)
      let coat = lu_coat(h, sv);
      let prev_cov = cov_d;
      var coat_nd = nd;
      var coat_rough = prough;
      var coat_cov = cov_d;
      if (lu_mirror(h, sv)) {
        if (clean && nd == 0 && sv.rough >= LU_SHARP && lu_caustics_on() && lu_cache_on()) {
          // a glossy mirror, the first thing seen: the light that comes to it off mirrors or through glass is from the
          // caustic cache (caustics, LU_SHARP); what it reflects of other surfaces is as off any mirror
          cov_d = true;
          xs = 0;
          cached = true;
        } else {
          xs += 1;
        }
        clean = false;
      } else {
        if (clean) {
          coat_nd = nd + 1;
          coat_rough = prough && lu_rough(sv);
        }
        // (sent straight to the camera by the light paths: the first surface it sees, or one past rough ones only)
        let splat = clean && coat_nd <= LU_CAU_BOUNCES + 1 && (coat_nd == 1 || coat_rough);
        coat_cov = lu_caustics_on() && (splat || lu_cache_on());
        cached = coat_cov && !splat;
        if (!coat) {
          nd = coat_nd;
          prough = coat_rough;
          cov_d = coat_cov;
          xs = 0;
        }
      }
      if (lu_med) {
        // (a point in the murk: no caustics land there, nothing cached)
        cached = false;
        cov_d = false;
        coat_cov = false;
      }
      lu_cov = cov_d && xs > 0;
      lu_coat_on = coat;
      lu_coat_ex = coat && prev_cov;
      lu_cau_sh = select(cov_d, coat_cov, coat);
      var dl = lu_direct(sv, m, fr, vl, cached, depth);
      var glint = vec3<f32>(0.0);
      if (s.glint > 0.0 && max(U.sun.r, max(U.sun.g, U.sun.b)) > 0.0) {
        // a grain turned just so flashes the sun at the camera
        let gl = s.glint * pow(max(dot(s.gn, normalize(U.sund.xyz + v)), 0.0), 600.0) * 40.0;
        glint = U.sun.rgb * gl * max(dot(s.n, U.sund.xyz), 0.0);
        dl += glint;
      }
      var add = thr * dl;
      var ksc = 1.0;
      if (depth > 0 && cap > 0.0) {
        let y = luma(add);
        if (y > cap) {
          ksc = cap / y;
          add *= ksc;
        }
      }
      L += add;
      if (F_LPASS) {
        let tk = thr * ksc;
        lu_lp[lu_dg] += tk * (dl - glint - lu_dc);
        lu_lp[0] += tk * (glint + lu_dc * U.lp.x);
        lu_lp[1] += tk * lu_dc * U.lp.y;
        lu_lp[3] += tk * lu_dc * U.lp.w;
      }
      // the bounce: the highlight (by its visible normals) or diffuse (by the cosine). It is traced even after the
      // last bounce, for the sky it may reach: that is this surface's light too, the half of the sky that the light
      // picked from the HDRI above leaves to it (multiple importance sampling)
      let last = depth + 1 >= bounces;
      var ll: vec3<f32>;
      lu_at(2u);
      let ub = lu_rand2();   // (the lobe by its x, then rescaled, the direction: as lu_direct picks its light)
      if (F_WATER && lu_med) {
        // the murk scatters it any way alike
        let z = 1.0 - 2.0 * ub.x;
        let rr = sqrt(max(1.0 - z * z, 0.0));
        ll = vec3<f32>(rr * cos(6.2831853 * ub.y), rr * sin(6.2831853 * ub.y), z);
      } else if (ub.x < m.ps) {
        let hl = lu_vndf(vl, m.a, vec2<f32>(min(ub.x / m.ps, 0.99999994), ub.y));
        ll = reflect(-vl, hl);
      } else {
        ll = lu_cosine(vec2<f32>(min((ub.x - m.ps) / max(1.0 - m.ps, 1e-6), 0.99999994), ub.y));
      }
      if (ll.z <= 0.0 && !lu_med) { break; }
      let e2 = lu_eval2(m, vl, ll);
      var ef = e2.fd + e2.fs;
      var epdf = (1.0 - m.ps) * e2.pd + m.ps * e2.ph;
      if (coat) {
        // a coat: by the lobe taken alone (its highlight a mirror's, its colour a matte surface's: the caustics split so)
        if (ub.x < m.ps) {
          ef = e2.fs;
          epdf = m.ps * e2.ph;
          clean = false;
          xs += 1;
        } else {
          ef = e2.fd;
          epdf = (1.0 - m.ps) * e2.pd;
          nd = coat_nd;
          prough = coat_rough;
          cov_d = coat_cov;
          xs = 0;
        }
      }
      if (epdf <= 0.0) { break; }
      thr *= ef / epdf;
      pdf_last = epdf;
      glass_s = false;
      glass_x = false;
      scattered = true;
      rd = fr[0] * ll.x + fr[1] * ll.y + fr[2] * ll.z;
      ro = s.p + s.n * s.eps;
      depth += 1;
      // Russian roulette, from the fourth bounce: paths that carry little light end, the rest carry more (ending them
      // sooner saves little time and adds noise: measured against Mitsuba, from the second it doubled a glossy set's)
      if (depth >= 4 && !last) {
        lu_at(5u);
        let q = clamp(max(thr.r, max(thr.g, thr.b)), 0.05, 0.95);
        if (lu_rand() >= q) { break; }
        thr /= q;
      }
    }
    if (inner > 8) { break; }
    // on to the next surface, or out to the sky
    var hn = trace_w(ro, rd, 0.0, 1.0e5, 1.0, true);
    if (water_on()) {
      // the water on the way: its absorption, and what may scatter the path first (a point there, MEDIUM: lit and
      // bounced as a surface is, with the medium's phase): its murk, the bubbles in it, the spray over it
      let tseg = select(60.0, hn.t, hn.id >= 0);
      lu_at(4u);
      let ev = w_event(ro, rd, tseg, in_water, lu_rand2(), lu_rand());
      if (in_water) { thr *= exp(-w_sigma() * select(tseg, ev.x, ev.x >= 0.0) - g_wdye); }   // (and the dye's: g_wdye)
      if (ev.x >= 0.0) {
        thr *= ev.yzw;
        hn = Hit(ev.x, MEDIUM);
      }
    }
    // a lamp in the way (once reflected or refracted: the camera does not see the lamps themselves, but their highlights in
    // glass), weighed against having picked it
    if (scattered) {
      var tl = 1.0e9;
      var kl = -1;
      for (var k = 0; k < i32(U.ln.w); k++) {
        let t = lu_lamp_t(ro, rd, k);
        if (t > 0.0 && t < tl) {
          tl = t;
          kl = k;
        }
      }
      if (kl >= 0 && (hn.id < 0 || tl < hn.t)) {
        // (through glass after a bounce off a surface: in that surface's shadow rays, or the caustics')
        if (!(depth > 0 && (glass_s || glass_x)) && !(cov_d && xs > 0)) {
          var w = 1.0;
          if (pdf_last > 0.0) { w = lu_power(pdf_last, lu_sel(3 + kl) * lu_lamp_pdf(ro, rd, tl, kl)); }
          var add = thr * lu_lamp_radiance(kl, rd) * w;
          if (cap > 0.0) {
            let y = luma(add);
            if (y > cap) { add *= cap / y; }
          }
          L += add;
          lu_lp_add(3, add);
        }
        break;
      }
    }
    if (!lu_drawn(hn)) {
      var sky = lu_sky(ro, rd, pdf_last, select(0.0, lu_sel(3 + i32(U.ln.w)), pdf_last > 0.0));
      // (the HDRI through glass: in the shadow rays already, or the caustics'; by way of mirrors from a surface the
      // caustics land on: theirs. Not traced: only through glass that does not focus, which its shadow rays go through)
      if (depth > 0 && lu_env_on()) {
        if (lu_env_cau()) {
          if (glass_s || glass_x || (cov_d && xs > 0)) { sky = vec3<f32>(0.0); }
        } else if (glass_s && !glass_x) {
          sky = vec3<f32>(0.0);
        }
      }
      // (with a background colour behind the set, what the camera sees straight through water and clear things is that
      // colour, as it sees it past them; the sky still lights the set)
      let backdrop = depth == 0 && U.stage.z < 0.5;
      if (backdrop) { sky = select(U.bg.rgb, vec3<f32>(0.21), U.bg.w > 0.5); }
      var add = thr * sky;
      if (depth > 0 && cap > 0.0) {
        let y = luma(add);
        if (y > cap) { add *= cap / y; }
      }
      L += add;
      if (!backdrop) { lu_lp_add(1, add); }
      break;
    }
    if (depth >= bounces) { break; }   // (past the last bounce: only the sky counted)
    h = hn;
  }
  return L;
}

// Lume's camera kernel: the stage runs it in place of its main when Lighting engine is Lume (stage.wgsl stage_pixel).
@compute @workgroup_size(8, 8, 1)
fn lume_main(@builtin(global_invocation_id) id: vec3<u32>) { stage_pixel(id, true); }

// What the camera sees along one ray with Lume (see() for the rest: the footage, its matte and holdouts, the sky).
fn lume_see(ro0: vec3<f32>, rd0: vec3<f32>, px: vec2<f32>, puv: vec2<f32>, rd_w: vec3<f32>) -> Seen {
  let footage = U.stage.w > 0.5;
  t_piece = -1.0;
  t_footage = 1.0e9;
  lu_nrm = vec3<f32>(0.0);
  lu_alb = vec3<f32>(1.0);
  lu_dist = 1.0e4;
  lu_lam = 0.0;
  lu_lam_w = vec3<f32>(1.0);
  if (F_LPASS) {
    for (var g = 0; g < 4; g++) { lu_lp[g] = vec3<f32>(0.0); }
    lu_lp_k = 1.0;
  }
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
  var h = trace_w(ro0, rd0, 0.0, select(1.0e5, t_foot, footage), 1.0, !footage);
  if ((h.id >= PIECE || h.id == MATTER)) { t_piece = h.t; }
  let cam_wet = w_inside(ro0);
  lu_keep0 = vec3<f32>(1.0);
  if (water_on()) {
    // the spray in the air (or under the surface, the murk and the bubbles) before the first thing the camera sees, or
    // in front of the background: it may scatter the camera's ray there
    let ev = w_event(ro0, rd0, select(1.0e4, h.t, h.id >= 0), cam_wet, lu_rand2(), lu_rand());
    if (ev.x >= 0.0) {
      h = Hit(ev.x, MEDIUM);
      lu_keep0 = ev.yzw;
    }
    lu_keep0 *= exp(-g_wdye);   // (the dye in the water on the camera's first stretch)
  }
  // (with a background colour behind the set, the floor is seen only through the water, and where the liquid left it
  // wet: its film's sheen, over the colour)
  var wet_seen = 1.0;
  if (water_on() && h.id == FLOOR && U.stage.z < 0.5 && !cam_wet) { wet_seen = w_wet(ro0 + rd0 * h.t); }
  if (!lu_drawn(h) || wet_seen <= 0.001) {
    let c = past_cg(ro0, rd0, h, ro0, px, puv, t_foot, 0.0, true);
    lu_alb = max(c, vec3<f32>(1e-3));
    return Seen(c, select(1.0, 0.0, footage), 0.0);
  }
  var col = lu_path(h, ro0, rd0);
  if (wet_seen < 1.0) { col = mix(background(rd0, px), col, wet_seen); }
  let dist = h.t;
  var keep = wet_seen;   // (of the path's light, what the pixel shows: per-light passes)
  if (h.id == FLOOR && !cam_wet) {
    // far off, the floor fades into the sky at the horizon
    let fade = 1.0 - exp(-dist / max(U.floor_d.w, 1.0));
    // (into the sky; with a background colour behind the set, into that: the floor drawn under the water, Lume's)
    let far = select(background(rd0, px), sky_dir(vec3<f32>(rd0.x, 0.0, rd0.z)) * 0.9, U.stage.z > 0.5);
    col = mix(col, far, fade);
    keep *= 1.0 - fade;
  }
  col = haze(col, dist);
  if (U.atm.w > 0.0) { keep *= exp(-U.atm.w * dist); }
  var cg = 1.0;
  if (footage) {
    cg = 1.0 - matte;
    if (matte > 0.0) { col = mix(col, footage_at(puv), matte); }
    keep *= 1.0 - matte;
  }
  if (F_LPASS) { lu_lp_k = keep; }
  return Seen(col, cg, h.t);
}


// ---- caustics: light traced from the lights through curved glass, to the camera ---------------------------------------

const LU_CAU_SCALE: f32 = 1024.0;   // CAU holds radiance times this, rounded (atomics are integers)

// The caustics this pass put in pixel k (radiance, rgb).
fn lu_caustic_at(k: u32) -> vec3<f32> {
  return vec3<f32>(f32(atomicLoad(&CAU[3u * k])), f32(atomicLoad(&CAU[3u * k + 1u])),
                   f32(atomicLoad(&CAU[3u * k + 2u]))) / LU_CAU_SCALE;
}

struct LuThru { ro: vec3<f32>, rd: vec3<f32>, tint: vec3<f32>, ok: bool };

// Light travelling along rd that meets clear thing h at p from outside: reflected off it or refracted in, across
// (reflected inside while it cannot get out) and out, at random by Fresnel; where it goes on from, which way, and the tint
// of its way through. (ok false: it was lost inside.)
fn lu_through(h: Hit, p: vec3<f32>, rd: vec3<f32>, cl: LuClear, eps: f32) -> LuThru {
  var n = lu_hit_normal(h, p, eps);
  if (dot(n, rd) > 0.0) { n = -n; }
  let ci = max(dot(n, -rd), 1e-4);
  if (lu_rand() < lu_fresnel(ci, cl.ior)) {
    return LuThru(p + n * eps, reflect(rd, n), vec3<f32>(1.0), true);
  }
  var rin = refract(rd, n, 1.0 / cl.ior);
  if (dot(rin, rin) < 0.5) { rin = rd; }
  var pin = p - n * eps;
  var tint = vec3<f32>(1.0);
  for (var j = 0; j < 6; j++) {
    let ex = lu_exit(h, pin, rin, eps);
    let nout = ex.xyz;
    let pout = pin + rin * ex.w;
    tint *= pow(max(cl.tint, vec3<f32>(1e-3)), vec3<f32>(ex.w / 0.1));
    let fo = lu_fresnel(max(dot(rin, nout), 1e-4), 1.0 / cl.ior);
    let rout = refract(rin, -nout, cl.ior);
    if (dot(rout, rout) > 0.5 && lu_rand() >= fo) {
      return LuThru(pout + nout * (2.0 * eps), rout, tint, true);
    }
    rin = reflect(rin, -nout);
    pin = pout - nout * (2.0 * eps);
  }
  return LuThru(pin, rin, tint, false);
}

const LU_CAU_BOUNCES: i32 = 2;   // a caustic's light is followed this many bounces on from where it lands

// ---- the caustic cache: the light the caustics' paths land with, where they land ------------------------------------
//
// A light path that has come through curved glass or off a mirror and lands on a surface also leaves its light there,
// in a cell of the cache (CAU, after the pixels' splats): the irradiance it brings, in two parts (by way of glass only,
// or off a mirror too: its bounces count), and the way it came. The camera's paths that reach a surface the light paths
// cannot send to the camera in a straight line (seen through glass or in a mirror, or after more bounces than the light
// paths follow) take that surface's caustic light from its cell: the caustic a glass ball makes, seen through the ball,
// focused (a path from the camera through the glass to a lamp, found by bouncing, is a firefly).
//
// Cells are U.ceye.w wide across a surface and four times that through it, one set for each of the six ways a surface
// faces, hashed into LU_CACHE_SLOTS slots (a slot's first word holds its cell's key; a cell taken goes on to the next
// slots), shifted by a random offset each pass so that over the passes a cell's edges blur away. Each pass starts it
// afresh with the splats.
const LU_CACHE_SLOTS: u32 = 262144u;   // (lume.py CACHE_SLOTS)
const LU_CACHE_WORDS: u32 = 10u;       // key; irradiance by way of glass (rgb); off a mirror too (rgb); the way it came (xyz)
const LU_CACHE_SCALE: f32 = 16384.0;   // (fixed point: irradiance times this)

fn lu_cache_on() -> bool { return U.ceye.w > 0.0; }

const LU_CACHE_SHRINK: f32 = 0.1666667;   // the cells shrink as the passes add up, as passes^-this (their blur goes)

// This pass's cell size: U.ceye.w at the first, shrinking (progressive photon mapping: the cache's blur goes to nothing
// as passes add up, while each pass's cells still hold enough light paths, their noise averaged away by the passes).
fn lu_cache_h() -> f32 { return U.ceye.w * pow(U.lume.y + 1.0, -LU_CACHE_SHRINK); }

// The cache's cell for a surface at p facing n: (its slot, its key).
fn lu_cell(p: vec3<f32>, n: vec3<f32>) -> vec2<u32> {
  let ps = u32(U.lume.y + 0.5) * 3u;
  let jit = vec3<f32>(f32(lu_hash(ps + 101u) >> 8u), f32(lu_hash(ps + 102u) >> 8u), f32(lu_hash(ps + 103u) >> 8u)) / 16777216.0;
  let q = p / lu_cache_h() + jit;
  let a = abs(n);
  var face = 0u;
  var c = vec3<f32>(q.y, q.z, q.x * 0.25);
  if (a.x >= a.y && a.x >= a.z) {
    face = select(0u, 1u, n.x > 0.0);
  } else if (a.y >= a.z) {
    face = select(2u, 3u, n.y > 0.0);
    c = vec3<f32>(q.x, q.z, q.y * 0.25);
  } else {
    face = select(4u, 5u, n.z > 0.0);
    c = vec3<f32>(q.x, q.y, q.z * 0.25);
  }
  let ci = vec3<i32>(floor(c));
  var k = lu_hash(bitcast<u32>(ci.x) ^ (face * 0x9e3779b9u));
  k = lu_hash(k ^ (bitcast<u32>(ci.y) * 0x85ebca6bu));
  k = lu_hash(k ^ (bitcast<u32>(ci.z) * 0xc2b2ae35u));
  let key = k | 1u;
  return vec2<u32>(lu_hash(key ^ 0x27d4eb2du) & (LU_CACHE_SLOTS - 1u), key);
}

// The cache's first word in CAU.
fn lu_cache_base() -> u32 { return 3u * u32(U.res.x) * u32(U.res.y); }

// A light path's light landing at surface s, travelling rd, carrying beta (one path among U.lume3.x): into its cell.
fn lu_cache_add(s: Surf, rd: vec3<f32>, beta: vec3<f32>, mirrored: bool) {
  let c = lu_cell(s.p, s.n);
  let a = abs(s.n);
  // (its power, over the surface a cell holds: h^2 across, over the surface's slope to the cell's face)
  let h = lu_cache_h();
  let e = beta * (max(a.x, max(a.y, a.z)) / (max(U.lume3.x, 1.0) * h * h));
  let q = min(e * LU_CACHE_SCALE, vec3<f32>(3.0e7));
  let dq = -rd * min(luma(e) * LU_CACHE_SCALE, 3.0e7);
  let base = lu_cache_base();
  for (var j = 0u; j < 8u; j++) {
    let w0 = base + ((c.x + j) & (LU_CACHE_SLOTS - 1u)) * LU_CACHE_WORDS;
    var r = atomicCompareExchangeWeak(&CAU[w0], 0u, c.y);
    if (!r.exchanged && r.old_value == 0u) { r = atomicCompareExchangeWeak(&CAU[w0], 0u, c.y); }   // (a weak one may fail)
    if (r.exchanged || r.old_value == c.y) {
      let o = w0 + select(1u, 4u, mirrored);
      // (rounded at random, up or down by how near: so many small parts add up to their sum)
      atomicAdd(&CAU[o], u32(q.r + lu_pcg()));
      atomicAdd(&CAU[o + 1u], u32(q.g + lu_pcg()));
      atomicAdd(&CAU[o + 2u], u32(q.b + lu_pcg()));
      atomicAdd(&CAU[w0 + 7u], bitcast<u32>(i32(round(dq.x))));
      atomicAdd(&CAU[w0 + 8u], bitcast<u32>(i32(round(dq.y))));
      atomicAdd(&CAU[w0 + 9u], bitcast<u32>(i32(round(dq.z))));
      return;
    }
  }
}

// The caustic light arriving at surface s, from its cell: its irradiance, and the way it came (s.n where it has none).
// depth: the bounces off surfaces before this one (the light off a mirror is one more, when the set allows it).
struct LuCached { e: vec3<f32>, l: vec3<f32> };

fn lu_cache_get(s: Surf, depth: i32) -> LuCached {
  var out = LuCached(vec3<f32>(0.0), s.n);
  let c = lu_cell(s.p, s.n);
  let base = lu_cache_base();
  for (var j = 0u; j < 8u; j++) {
    let w0 = base + ((c.x + j) & (LU_CACHE_SLOTS - 1u)) * LU_CACHE_WORDS;
    let k = atomicLoad(&CAU[w0]);
    if (k == 0u) { break; }
    if (k == c.y) {
      var e = vec3<f32>(f32(atomicLoad(&CAU[w0 + 1u])), f32(atomicLoad(&CAU[w0 + 2u])), f32(atomicLoad(&CAU[w0 + 3u])));
      if (depth + 2 <= max(i32(U.lume.z + 0.5), 1)) {
        e += vec3<f32>(f32(atomicLoad(&CAU[w0 + 4u])), f32(atomicLoad(&CAU[w0 + 5u])), f32(atomicLoad(&CAU[w0 + 6u])));
      }
      out.e = e / LU_CACHE_SCALE;
      let d = vec3<f32>(f32(bitcast<i32>(atomicLoad(&CAU[w0 + 7u]))), f32(bitcast<i32>(atomicLoad(&CAU[w0 + 8u]))),
                        f32(bitcast<i32>(atomicLoad(&CAU[w0 + 9u]))));
      if (dot(d, d) > 1e-6) { out.l = normalize(d); }
      break;
    }
  }
  return out;
}

// Light that came along rd to surface s, carrying beta, as the camera sees it (colour_only: a coat's colour, not its
// highlight): sent along a straight line to the camera
// (nothing in the way, glass too), into the pixel it is seen in (light tracing; U.lume3.z: 1 / (paths x a pixel's area)).
fn lu_splat(s: Surf, rd: vec3<f32>, beta: vec3<f32>, colour_only: bool) {
  let wc = U.ceye.xyz - s.p;
  let dist = length(wc);
  let wn = wc / max(dist, 1e-6);
  let nx = dot(s.n, wn);
  if (nx <= 0.0 || dot(s.n, -rd) <= 0.0) { return; }
  let hv = trace(s.p + s.n * max(s.eps, 1.0e-4), wn, 0.0, dist, 1.0, true);
  if (hv.id >= 0) { return; }
  let clip = U.cvp * (U.cl2w * vec4<f32>(s.p, 1.0));
  if (clip.w <= 1e-6) { return; }
  let ndc = clip.xy / clip.w;
  let px = vec2<i32>(floor(vec2<f32>(0.5 * (ndc.x + 1.0), 0.5 * (1.0 - ndc.y)) * U.res.xy));
  if (px.x < 0 || px.y < 0 || f32(px.x) >= U.res.x || f32(px.y) >= U.res.y) { return; }
  let fr = lu_frame(s.n);
  let vl = lu_to_local(fr, wn);
  let ll = lu_to_local(fr, -rd);
  let m = lu_mat(s, vl.z);
  let e = lu_eval2(m, vl, ll);
  let f = select(e.fd + e.fs, e.fd, colour_only) / max(ll.z, 1e-4);   // (a coat: its colour only, lu_coat)
  // the camera's axis (fire-local) and how far off it the surface is seen
  let fwd = normalize((U.w2l * vec4<f32>(U.fwd.xyz, 0.0)).xyz);
  let cc = max(dot(-wn, fwd), 1e-4);
  let val = beta * f * (nx / (dist * dist * cc * cc * cc) * U.lume3.z);
  let k = u32(px.y) * u32(U.res.x) + u32(px.x);
  let q = min(val * LU_CAU_SCALE, vec3<f32>(3.0e7));
  atomicAdd(&CAU[3u * k], u32(q.r + 0.5));
  atomicAdd(&CAU[3u * k + 1u], u32(q.g + 0.5));
  atomicAdd(&CAU[3u * k + 2u], u32(q.b + 0.5));
}

// A light path from one of the lights, aimed at one of the curved clear things: (where from, which way, the light it
// carries: its share of the light's power, for one path among U.lume3.x), or a zero light when there is nothing to send.
struct LuEmit { ro: vec3<f32>, rd: vec3<f32>, beta: vec3<f32> };

fn lu_emit() -> LuEmit {
  var e = LuEmit(vec3<f32>(0.0), vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0));
  let nl = i32(U.ln.w);
  let sun_on = max(U.sun.r, max(U.sun.g, U.sun.b)) > 0.0;
  let env_on = lu_env_on() && lu_env_cau();
  let fire_on = lu_fire_cau();
  let kinds = nl + select(0, 1, sun_on) + select(0, 1, env_on) + select(0, 1, fire_on);
  let nt = i32(U.lume3.y);
  if (kinds == 0 || nt == 0) { return e; }
  let pick = min(i32(lu_rand() * f32(kinds)), kinds - 1);
  let tg = U.ctg[min(i32(lu_rand() * f32(nt)), nt - 1)];
  let sel = f32(kinds * nt);   // (1 / the chance of this light and this target)
  if (pick < nl) {
    // a lamp: from a point on its surface (the part from which some of the target is in front of it: a cap round its side
    // facing the target, wider by the angle the target fills), a direction within the cone the target fills seen from
    // there; its light by its radiance and the cosine there. (From its middle, as a point, a big lamp's caustic was too
    // sharp and too small: the measured caustic seen through its glass, wrong at its edges.)
    let lm = lamps[pick];
    let r = max(lm.p.w, 1e-3);
    let c = lm.p.xyz;
    let toward = tg.xyz - c;
    let dist = length(toward);
    let sa = clamp(tg.w / max(dist, 1e-6), 0.0, 1.0);                   // (sin of the angle the target fills)
    let cmin = -sa;                                                     // (the cap: out to 90 degrees and that angle more)
    let lf = lu_frame(toward / max(dist, 1e-6));
    let nc = lu_cone(lu_rand2(), cmin);                                // (a point on the cap, uniformly by area)
    var nx = lf[0] * nc.x + lf[1] * nc.y + lf[2] * nc.z;
    var x = c + nx * r;
    var cap = 2.0 * PI * r * r * (1.0 - cmin);
    if (lu_panel(pick)) {
      // a panel: a point on its face, evenly
      let ax = lamps[LAMP_MAX + pick];
      let up = lu_rand2();
      nx = lm.d.xyz;
      x = c + ax.p.xyz * ((2.0 * up.x - 1.0) * ax.p.w) + ax.c.xyz * ((2.0 * up.y - 1.0) * ax.c.w);
      cap = 4.0 * ax.p.w * ax.c.w;
    }
    let cone = lu_cone_of(x, tg.xyz, tg.w);
    var l = vec3<f32>(0.0, -1.0, 0.0);
    var omega = 4.0 * PI;
    if (cone.x < -1.5) {
      l = lu_cone(lu_rand2(), -1.0);
    } else {
      let fr = lu_frame(normalize(tg.xyz - x));
      let lc = lu_cone(lu_rand2(), cone.x);
      l = fr[0] * lc.x + fr[1] * lc.y + fr[2] * lc.z;
      omega = cone.y;
    }
    let cosx = dot(nx, l);
    if (cosx <= 0.0) { return e; }
    e.ro = x + nx * 1.0e-4;
    e.rd = l;
    e.beta = lu_lamp_radiance(pick, -l) * (cosx * cap * omega * sel);
    return e;
  }
  if (fire_on && pick == kinds - 1) {
    // the fire: one of its lights, picked by its power, from a point in its ball (as its shadow rays go to), a direction
    // within the cone the target fills; through the smoke on the way to the target
    let nf = light_count[0];
    var tot = 0.0;
    for (var k = 0u; k < nf; k++) { tot += luma(lights[2u * k + 1u].rgb); }
    if (tot <= 0.0) { return e; }
    let pk = lu_rand() * tot;
    var run = 0.0;
    var k = 0u;
    for (; k < nf - 1u; k++) {
      run += luma(lights[2u * k + 1u].rgb);
      if (run >= pk) { break; }
    }
    let a = lights[2u * k];
    let pw = luma(lights[2u * k + 1u].rgb);
    let x = a.xyz + lu_ball() * a.w;
    let cone = lu_cone_of(x, tg.xyz, tg.w);
    if (cone.x < -1.5 || pw <= 0.0) { return e; }
    let fr = lu_frame(normalize(tg.xyz - x));
    let lc = lu_cone(lu_rand2(), cone.x);
    e.ro = x;
    e.rd = fr[0] * lc.x + fr[1] * lc.y + fr[2] * lc.z;
    e.beta = U.fire.rgb * lights[2u * k + 1u].rgb * (cone.y * sel * tot / pw);
    // (the smoke from the edge of its ball, as its shadow rays see it: lu_ls_fire)
    if (U.fire.w > 0.0) {
      e.beta *= mix(1.0, smoke_tr(x + e.rd * a.w, e.rd, max(length(tg.xyz - x) - a.w - tg.w, 0.0)), U.fire.w);
    }
    return e;
  }
  // the key light or the HDRI: parallel light over a disc as wide as the target, from far off
  var dir = vec3<f32>(0.0, 1.0, 0.0);   // toward the light
  var power = vec3<f32>(0.0);           // what crosses a unit area facing it
  if (sun_on && pick == nl) {
    let cmax = cos(atan(1.0 / max(U.sun.w, 1.0)));
    let sf = lu_frame(U.sund.xyz);
    let lc = lu_cone(lu_rand2(), cmax);
    dir = sf[0] * lc.x + sf[1] * lc.y + sf[2] * lc.z;
    power = U.sun.rgb;
  } else {
    let pk = lu_env_pick(lu_rand2());
    if (pk.w <= 0.0) { return e; }
    dir = pk.xyz;
    power = sky_dir(dir) / pk.w;
  }
  if (U.stage.x > 0.5 && dir.y < 0.0) { return e; }   // (light from below the horizon: the ground stops it)
  let travel = -dir;
  let fr = lu_frame(travel);
  let u = lu_rand2();
  let rr = sqrt(u.x) * tg.w;
  let ph = 2.0 * PI * u.y;
  // (from just outside the target, if nothing between there and the light stops it: a roof over the glass shades it)
  e.ro = tg.xyz + (fr[0] * cos(ph) + fr[1] * sin(ph)) * rr - travel * (2.0 * tg.w);
  e.rd = travel;
  if (lu_open(e.ro, dir, 1.0e5, 1.0e-4, 1.0) <= 0.0) { return e; }
  e.beta = power * (PI * tg.w * tg.w * sel);
  return e;
}

@compute @workgroup_size(64, 1, 1)
fn caustics(@builtin(global_invocation_id) id: vec3<u32>) {
  if (f32(id.x) >= U.lume3.x) { return; }
  lu_rng = pcg1(id.x * 9781u + u32(U.lume.y + 0.5) * 6271u + 17u);
  lu_base = 0u;   // (PCG: these paths are not a pixel's samples)
  lu_dim = 0u;
  lu_nq = 0u;
  lu_lam = 0.0;
  lu_lam_w = vec3<f32>(1.0);
  g_tau = U.res.w * (lu_rand() - 0.5);
  g_jit = lu_rand();
  let em = lu_emit();
  if (max(em.beta.r, max(em.beta.g, em.beta.b)) <= 0.0) { return; }
  var ro = em.ro;
  var rd = em.rd;
  var beta = em.beta;
  var curved = 0;
  var landed = 0;
  var opq = 0;   // bounces off surfaces so far (mirrors and landings: Bounces counts them, glass not)
  let bounces = max(i32(U.lume.z + 0.5), 1);
  let eps = 1.0e-4;
  for (var j = 0; j < 16; j++) {
    let h = trace(ro, rd, 0.0, 2.0e5, 1.0, true);
    if (h.id < 0 || !lu_drawn(h)) { return; }
    let p = ro + rd * h.t;
    let cl = lu_clear(h, p);
    if (cl.clear > 0.0) {
      if (landed > 0) { return; }   // (once landed, only on off diffuse and glossy surfaces: as lu_path leaves them)
      if (lu_rand() >= cl.clear) { return; }   // (its cloudy part)
      if (!lu_xglass(h)) {
        // glass that does not focus: straight through by its Fresnel and tint, as its shadow is
        let n = lu_hit_normal(h, p, eps);
        let ex = lu_exit(h, p + rd * eps, rd, eps);
        let t = (1.0 - lu_fresnel(max(abs(dot(n, rd)), 1e-4), cl.ior)) * (1.0 - lu_fresnel(max(abs(dot(ex.xyz, rd)), 1e-4), 1.0 / cl.ior));
        beta *= t * pow(max(cl.tint, vec3<f32>(1e-3)), vec3<f32>(ex.w / 0.1));
        ro = p + rd * (ex.w + 2.0 * eps);
        continue;
      }
      var cd = cl;
      cd.ior = lu_ior(cl.ior);
      beta *= lu_lam_take();
      let th = lu_through(h, p, rd, cd, eps);
      if (!th.ok) { return; }
      beta *= th.tint;
      ro = th.ro;
      rd = th.rd;
      curved += 1;
      continue;
    }
    let s = lu_surf(h, ro, rd);
    if (lu_mirror(h, s)) {
      // off a mirror: a direction by its highlight's visible normals, as seen from where the light came (bare metal:
      // its highlight is all of it, and symmetric)
      if (landed > 0) { return; }
      opq += 1;
      if (opq > bounces) { return; }
      // (a glossy mirror: the light that came off mirrors or through glass, left in the cache for the camera's paths
      // that see it straight: lu_path)
      if (curved > 0 && s.rough >= LU_SHARP && lu_cache_on()) { lu_cache_add(s, rd, beta, opq > 1); }
      if (opq >= bounces) { return; }   // (it still has to land)
      let fr = lu_frame(s.n);
      let li = lu_to_local(fr, -rd);
      if (li.z <= 0.0) { return; }
      let m = lu_mat(s, li.z);
      let lo = reflect(-li, lu_vndf(li, m.a, lu_rand2()));
      if (lo.z <= 0.0) { return; }
      let ev = lu_eval(m, li, lo);
      if (ev.w <= 0.0) { return; }
      beta *= ev.xyz / ev.w;
      ro = s.p + s.n * max(s.eps, 1.0e-4);
      rd = fr[0] * lo.x + fr[1] * lo.y + fr[2] * lo.z;
      curved += 1;
      continue;
    }
    var land = beta;
    let coat = lu_coat(h, s);
    if (coat) {
      // a coat over colour: off its highlight, as off a mirror, with the chance a bounce takes it; else into its colour
      if (landed > 0) { return; }
      let fr = lu_frame(s.n);
      let li = lu_to_local(fr, -rd);
      if (li.z <= 0.0) { return; }
      let m = lu_mat(s, li.z);
      if (lu_rand() < m.ps) {
        opq += 1;
        if (opq >= bounces) { return; }
        let lo = reflect(-li, lu_vndf(li, m.a, lu_rand2()));
        if (lo.z <= 0.0) { return; }
        let ev = lu_eval2(m, li, lo);
        if (ev.ph <= 0.0) { return; }
        beta *= ev.fs / (m.ps * ev.ph);
        ro = s.p + s.n * max(s.eps, 1.0e-4);
        rd = fr[0] * lo.x + fr[1] * lo.y + fr[2] * lo.z;
        curved += 1;
        continue;
      }
      land = beta / max(1.0 - m.ps, 1e-4);
    }
    if (curved == 0) { return; }   // (straight from the light: the camera's own paths have that light)
    if (landed > 0 && !lu_rough(s)) { return; }   // (landing again: only on rough surfaces, as lu_path counts them)
    // it has landed: the light this surface sends the camera (light tracing), then on, bounced off it (the light the
    // caustic throws round it, onto the floor beside it and the glass's underside: lu_path leaves these to it)
    opq += 1;
    if (opq > bounces) { return; }
    if (landed == 0 && lu_cache_on()) { lu_cache_add(s, rd, land, opq > 1); }
    lu_splat(s, rd, land, coat);
    landed += 1;
    beta = land;
    if (landed > LU_CAU_BOUNCES || !lu_rough(s)) { return; }   // (on only off rough surfaces)
    let fr = lu_frame(s.n);
    let li = lu_to_local(fr, -rd);
    if (li.z <= 0.0) { return; }
    let lo = lu_cosine(lu_rand2());   // (diffuse: f cos / pdf = f pi)
    let m = lu_mat(s, lo.z);
    let ev = lu_eval(m, lo, li);
    beta *= ev.xyz / max(li.z, 1e-4) * PI;
    let q = clamp(max(beta.r, max(beta.g, beta.b)) / max(max(em.beta.r, max(em.beta.g, em.beta.b)), 1e-12) * 4.0, 0.05, 1.0);
    if (lu_rand() >= q) { return; }   // (Russian roulette on what is left of its light)
    beta /= q;
    ro = s.p + s.n * max(s.eps, 1.0e-4);
    rd = fr[0] * lo.x + fr[1] * lo.y + fr[2] * lo.z;
  }
}
