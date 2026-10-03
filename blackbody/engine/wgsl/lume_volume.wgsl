// Lume for the smoke and steam: the light that reaches each point of the volume and is scattered there toward the
// camera, all but the key light's (and the lamps') first scattering, which the ray march works out exactly with its
// phase function. It is found by tracing rays from the point in every direction through the smoke, as Lume traces the
// set: along each, the flames' own light (dimmed by the smoke on the way, and stopped by the set), the key light
// scattered there on the way, the light the last pass found there scattered on (each pass a bounce more, the passes
// carried from frame to frame), and where the ray ends, the sky, or a surface of the set lit by the key light and by
// the light the last pass found in front of it (the fire's, the sky's, the smoke's). So smoke glows from inside where
// the fire lights it, takes the colour of the sunlit ground under it, and a wall between them keeps the fire's light
// off it. The key light's first scattering on the way takes the smoke's phase function; the light scattered more than
// once is taken as the same every way (by the second bounce it nearly is: against Mitsuba, within a tenth or so inside
// thick smoke and steam, where folding the forward scattering in by the similarity relation was further off). The light
// falling on a surface from a cell takes the way most of it comes from into account too (the first moment of what the
// rays found: the direction grid), so a floor under a fire gets the fire's light from above, not the floor's own.
//
// Half of each cell's rays go every way (an even spiral), half toward the fire (each in a cone round one of the fire's
// blocks, picked by how much light it sends here), the two weighted by the balance heuristic: smoke far from a small
// fire finds it without grain, and nothing is counted twice.
//
// trace: one thread a cell of this grid (LV); compose: the light volume the ray march reads, with LV in its fire-light
// channel and the sky's term taken out (LV holds the sky's light already).
//!include common.wgsl
//!include meshsdf.wgsl
//!include colliders.wgsl

@group(0) @binding(0) var scal: texture_3d<f32>;
@group(0) @binding(1) var aux: texture_3d<f32>;
@group(0) @binding(2) var chem: texture_3d<f32>;
@group(0) @binding(3) var bb: texture_2d<f32>;
@group(0) @binding(4) var lin: sampler;
@group(0) @binding(5) var L0: texture_3d<f32>;       // the light volume: fire light (rgb), the key light's transmittance (a)
@group(0) @binding(6) var atlas: texture_3d<f32>;    // meshes' distance (colliders.wgsl)
@group(0) @binding(7) var prev: texture_3d<f32>;     // LV as the last pass left it
@group(0) @binding(8) var prevd: texture_3d<f32>;    // the way its light comes from (first moment over the mean)
@group(0) @binding(9) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(10) var dstd: texture_storage_3d<rgba16float, write>;
@group(0) @binding(11) var<storage, read> lights: array<vec4<f32>>;   // the fire's blocks: centre, radius (m); power (rgb)
@group(0) @binding(12) var<storage, read> light_count: array<u32>;
@group(0) @binding(13) var L1c: texture_3d<f32>;     // the light volume's L1: z, the cloth's extinction (1/m)

//!include shade.wgsl

struct Params {
  g: vec4<f32>,        // simulation grid dims, its cell (m)
  org: vec4<f32>,      // its corner (fire-local m), the ground's height (fire-local m)
  lv: vec4<f32>,       // this grid's dims, rays a cell
  ld: vec4<f32>,       // the light volume's dims, the ground on (1/0)
  sun: vec4<f32>,      // toward the key light (fire-local), how much of the new pass to keep (1: all, below: blended in)
  flr: vec4<f32>,      // the ground's albedo (rgb), seed
  sx: vec4<f32>,       // the light volume's key-light steps' length (its cells: its optical depths are per cell), _
  look: Look,
  ccnt: vec4<f32>,
  col: array<Collider, MAX_COLLIDERS>,
  alb: array<vec4<f32>, MAX_COLLIDERS>,   // each object's albedo (rgb), whether rays stop at it (1/0)
};
@group(1) @binding(0) var<uniform> U: Params;

const PI: f32 = 3.14159265;
const LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

fn hash3(c: vec3<u32>, s: u32) -> u32 {
  var x = c.x * 1597334677u ^ c.y * 3812015801u ^ c.z * 2798796415u ^ s * 1979697957u;
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  x ^= x >> 16u;
  return x;
}

fn u01(x: u32) -> f32 { return f32(x >> 8u) / 16777216.0; }

// The k-th of n directions spread evenly over the sphere (a Fibonacci spiral), turned by `turn`.
fn spiral(k: u32, n: u32, turn: vec2<f32>) -> vec3<f32> {
  let z = 1.0 - (2.0 * (f32(k) + turn.x)) / f32(n);
  let r = sqrt(max(1.0 - z * z, 0.0));
  let a = 2.39996323 * f32(k) + 6.2831853 * turn.y;
  return vec3<f32>(r * cos(a), z, r * sin(a));
}

fn lv_q(p: vec3<f32>) -> vec3<f32> { return (p - U.org.xyz) / U.g.w * (U.lv.xyz / U.g.xyz); }

// The last pass's light at fire-local point p (the mean of what reaches it from every way).
fn prev_at(p: vec3<f32>) -> vec3<f32> {
  return samp_c(prev, lin, lv_q(p), U.lv.xyz).rgb;
}

// The last pass's light falling on a surface at p facing n (irradiance), from its mean and first moment (an L1
// spherical harmonic: exact for light from one side, as a floor's sky or a fire above it).
fn prev_irradiance(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
  let q = lv_q(p);
  let m = samp_c(prev, lin, q, U.lv.xyz).rgb;
  let w = samp_c(prevd, lin, q, U.lv.xyz).xyz;
  return PI * m * max(1.0 + 2.0 * dot(w, n), 0.0);
}

// The key light's transmittance through the smoke to fire-local point p: the light volume's, its optical depth taken
// over the length of its steps (light_shadow.wgsl counts each step toward the light as one cell, though an oblique one
// is up to the root of 3 of them).
fn sun_tr(p: vec3<f32>) -> f32 {
  let q = (p - U.org.xyz) / U.g.w * (U.ld.xyz / U.g.xyz);
  return pow(max(samp_c(L0, lin, q, U.ld.xyz).a, 1e-30), U.sx.x);
}

fn obj_normal(k: Collider, p: vec3<f32>, e: f32) -> vec3<f32> {
  let g = vec3<f32>(col_sdf(k, p + vec3<f32>(e, 0.0, 0.0)) - col_sdf(k, p - vec3<f32>(e, 0.0, 0.0)),
                    col_sdf(k, p + vec3<f32>(0.0, e, 0.0)) - col_sdf(k, p - vec3<f32>(0.0, e, 0.0)),
                    col_sdf(k, p + vec3<f32>(0.0, 0.0, e)) - col_sdf(k, p - vec3<f32>(0.0, 0.0, e)));
  let l = length(g);
  return select(vec3<f32>(0.0, 1.0, 0.0), g / l, l > 1e-9);
}

// Whether the key light reaches fire-local point p past the set's objects (1/0): a ray toward it, stepped by the
// objects' distance.
fn sun_clear(p: vec3<f32>, step: f32) -> f32 {
  let reach = length(U.g.xyz) * U.g.w;
  var t = step;
  for (var i = 0; i < 48; i++) {
    if (t > reach) { break; }
    let q = p + U.sun.xyz * t;
    var dmin = 1e9;
    for (var o = 0; o < i32(U.ccnt.x); o++) {
      if (U.alb[o].w < 0.5) { continue; }
      dmin = min(dmin, col_sdf(U.col[o], q));
    }
    if (dmin < 0.0) { return 0.0; }
    t += max(dmin, 0.5 * step);
  }
  return 1.0;
}

// What a lit surface at p (facing n, albedo a) sends back the way it was seen from: the key light on it, through the
// smoke and past the objects, and the light the last pass found just in front of it (the sky's, the fire's, the
// smoke's, the other surfaces').
fn surface_light(p: vec3<f32>, n: vec3<f32>, a: vec3<f32>, step: f32) -> vec3<f32> {
  let L = U.look;
  let ps = p + n * step;
  let cs = max(dot(n, U.sun.xyz), 0.0);
  var e_sun = vec3<f32>(0.0);
  if (cs > 0.0 && max(L.sun.r, max(L.sun.g, L.sun.b)) > 0.0) {
    e_sun = L.sun.rgb * cs * sun_tr(ps) * sun_clear(ps, step);
  }
  return a / PI * (e_sun + prev_irradiance(ps, n));
}

const CONE_RAYS_MAX: u32 = 32u;
const CONE_GROW: f32 = 1.5;      // a block's cone: round its emission-weighted centre, this much wider than its half-size

// The light along one ray from p0 the way d, starting t0 in: the flames' (dimmed on the way), the key light and the
// last pass's light scattered on the way, and where it ends, a lit surface or the sky.
fn march_ray(p0: vec3<f32>, d: vec3<f32>, ds: f32, steps: u32, t0: f32) -> vec3<f32> {
  let L = U.look;
  let n = U.g.xyz;
  let h = U.g.w;
  var t = t0;
  var tr = vec3<f32>(1.0);
  var got = vec3<f32>(0.0);
  var ended = false;
  for (var s = 0u; s < steps; s++) {
    let p = p0 + d * t;
    // the ground
    if (U.ld.w > 0.5 && p.y < U.org.w) {
      got += tr * surface_light(vec3<f32>(p.x, U.org.w, p.z), vec3<f32>(0.0, 1.0, 0.0), U.flr.rgb, ds);
      ended = true;
      break;
    }
    // the set's objects
    var hit = -1;
    for (var o = 0; o < i32(U.ccnt.x); o++) {
      if (U.alb[o].w < 0.5) { continue; }
      if (col_sdf(U.col[o], p) < 0.0) {
        hit = o;
        break;
      }
    }
    if (hit >= 0) {
      let nrm = obj_normal(U.col[hit], p, 0.5 * ds);
      got += tr * surface_light(p, nrm, U.alb[hit].rgb, ds);
      ended = true;
      break;
    }
    // out of the volume: the open ground beyond it (lit by the key light and the sky), or the sky
    let q = (p - U.org.xyz) / h;
    if (any(q < vec3<f32>(0.0)) || any(q > n)) {
      if (U.ld.w > 0.5 && d.y < -1e-4) {
        got += tr * U.flr.rgb / PI * (L.sun.rgb * max(U.sun.y, 0.0) + PI * L.amb.rgb);
        ended = true;
      }
      break;
    }
    let sm = samp_c(scal, lin, q, n);
    var ax = vec4<f32>(0.0);
    if (L.air.z > 0.5) { ax = samp_c(aux, lin, q, n); }
    var ch = vec4<f32>(0.0);
    if (L.air.w > 0.5) { ch = samp_c(chem, lin, q, n); }
    let dens = smoke_extinction(sm, L);
    let wet = steam_extinction(sm, ax, L);
    let sf = flame_sigma(sm, L);
    // (the cloth: it stops the light, its own light left to cloth_draw)
    let cl = samp_c(L1c, lin, q * (U.ld.xyz / n), U.ld.xyz).z;
    let sig = dens + wet + sf + cl;
    if (sig > 1e-6 || sm.x > 0.02) {
      let scat = dens * L.smoke.rgb + wet * L.steam.rgb;
      let e = emission_k(sm, ch, sf, dens, L) * L.sun.w;            // (the flames' light, as Fire light scatter sets)
      // (the key light scattered here toward the ray's start, by the phase function; the light the last pass found
      // here, every way alike)
      let inscat = prev_at(p) + L.sun.rgb * (sun_tr(p) * hg(dot(d, U.sun.xyz), L.amb.w));
      let a = exp(-sig * ds);
      let w = select(ds, (1.0 - a) / sig, sig > 1e-6);
      got += tr * (scat * inscat + e) * w;
      tr *= a;
      if (max(tr.x, max(tr.y, tr.z)) < 0.01) {
        ended = true;
        break;
      }
    }
    t += ds;
  }
  if (!ended) {
    got += tr * L.amb.rgb;
  }
  return got;
}

// A block of the fire as seen from p: how much light it sends here (luminance, for picking it), its cone's axis and
// the cosine of its half-angle.
struct Cone { w: f32, axis: vec3<f32>, cmax: f32 };

fn cone_of(i: u32, p: vec3<f32>) -> Cone {
  let a = lights[2u * i];
  let v = a.xyz - p;
  let r2 = dot(v, v);
  let rad = CONE_GROW * a.w;
  var k: Cone;
  k.w = dot(lights[2u * i + 1u].rgb, LUMA) / (r2 + a.w * a.w);
  k.axis = select(vec3<f32>(0.0, 1.0, 0.0), v * inverseSqrt(max(r2, 1e-12)), r2 > 1e-12);
  k.cmax = select(-1.0, sqrt(max(1.0 - rad * rad / max(r2, 1e-12), 0.0)), r2 > rad * rad);
  return k;
}

// The density over directions (per steradian) of picking way d by the fire's cones from p (W: their weights' sum).
fn cones_pdf(p: vec3<f32>, d: vec3<f32>, count: u32, W: f32) -> f32 {
  var pdf = 0.0;
  for (var i = 0u; i < count; i++) {
    let k = cone_of(i, p);
    if (k.w <= 0.0 || dot(d, k.axis) < k.cmax) { continue; }
    pdf += k.w / (2.0 * PI * (1.0 - k.cmax));
  }
  return pdf / W;
}

fn cone_dir(k: Cone, u: vec2<f32>) -> vec3<f32> {
  let ct = 1.0 - u.x * (1.0 - k.cmax);
  let st = sqrt(max(1.0 - ct * ct, 0.0));
  let ph = 2.0 * PI * u.y;
  let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(k.axis.y) > 0.9);
  let bx = normalize(cross(up, k.axis));
  let by = cross(k.axis, bx);
  return normalize(bx * (st * cos(ph)) + by * (st * sin(ph)) + k.axis * ct);
}

@compute @workgroup_size(4, 4, 4)
fn trace(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (any(c >= vec3<i32>(U.lv.xyz))) { return; }
  let n = U.g.xyz;
  let h = U.g.w;
  let lcell = U.g.xyz / U.lv.xyz;                     // (this grid's cell, in simulation cells)
  let p0 = U.org.xyz + (vec3<f32>(c) + 0.5) * lcell * h;
  let rays = u32(U.lv.w);
  let seed = u32(U.flr.w);
  let turn = vec2<f32>(u01(hash3(vec3<u32>(id), seed)), u01(hash3(vec3<u32>(id), seed + 7u)));
  let ds = max(min(min(lcell.x, lcell.y), lcell.z), 1.0) * h;   // (a step: this grid's cell, at least a simulation cell)
  let steps = u32(min(ceil(length(n) * h / ds), 256.0));
  // the fire's blocks, weighed from here
  let count = min(light_count[0], arrayLength(&lights) / 2u);
  var W = 0.0;
  for (var i = 0u; i < count; i++) { W += max(cone_of(i, p0).w, 0.0); }
  var n_c = 0u;
  if (W > 0.0) { n_c = min(rays / 2u, CONE_RAYS_MAX); }
  let n_u = rays - n_c;
  // the cone rays' blocks: stratified picks along the weights' running sum, in one walk
  var pick: array<u32, 32>;
  if (n_c > 0u) {
    let off = u01(hash3(vec3<u32>(id), seed + 13u));
    var acc = 0.0;
    var j = 0u;
    for (var i = 0u; i < count && j < n_c; i++) {
      acc += max(cone_of(i, p0).w, 0.0);
      while (j < n_c && (f32(j) + off) / f32(n_c) * W <= acc) {
        pick[j] = i;
        j++;
      }
    }
    for (; j < n_c; j++) { pick[j] = count - 1u; }
  }
  var sum = vec3<f32>(0.0);
  var mom = vec3<f32>(0.0);     // (the first moment of what the rays found, by luminance)
  for (var k = 0u; k < rays; k++) {
    var d: vec3<f32>;
    if (k < n_u) {
      d = spiral(k, n_u, turn);
    } else {
      let r = k - n_u;
      d = cone_dir(cone_of(pick[r], p0), vec2<f32>(u01(hash3(vec3<u32>(id), seed * 31u + r + 501u)),
                                                   u01(hash3(vec3<u32>(id), seed * 31u + r + 907u))));
    }
    // the balance heuristic over both: (1 / 4 pi) L / (n_u / 4 pi + n_c pdf_cones)
    var wt = 1.0 / f32(max(n_u, 1u));
    if (n_c > 0u) { wt = 1.0 / (f32(n_u) + 4.0 * PI * f32(n_c) * cones_pdf(p0, d, count, W)); }
    let got = march_ray(p0, d, ds, steps, ds * u01(hash3(vec3<u32>(id), k + 101u))) * wt;
    sum += got;
    mom += dot(got, LUMA) * d;
  }
  let lv = sum;
  var w = mom / max(dot(sum, LUMA), 1e-12);
  w *= min(1.0, 1.0 / max(length(w), 1e-6));
  let keep = clamp(U.sun.w, 0.0, 1.0);
  let old = textureLoad(prev, c, 0).rgb;
  let oldw = textureLoad(prevd, c, 0).xyz;
  textureStore(dst, c, vec4<f32>(mix(old, lv, keep), 1.0));
  textureStore(dstd, c, vec4<f32>(mix(oldw, w, keep), 0.0));
}
