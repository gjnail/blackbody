// Clouds lit by Lume (engine/cloud_render.py, with Lighting engine: Lume): the light a cloud scatters more than once,
// traced. Each cell of a coarser grid (the light scattered many times is smooth) holds the radiance reaching it from
// every way but the sun's straight beam: its mean (LV, rgb) and which way it flows (LD: its first moment, in luminance),
// for the march to scatter toward the eye by the cloud's phase (cloud_march.wgsl lume_toward): the sky and the ground
// seen through the cloud, and the light the cloud round it scatters, however many times.
//
// One pass of this kernel traces, from each cell, rays in every direction as light comes back along them, exactly as
// the cloud scatters it: each goes until the cloud scatters it (its optical depth, stepped through the grid) or it leaves
// the box (the sky, or the ground lit by the sun through the cloud's shadow and by the sky). Where it scatters, the sun's
// light there scattered back along it (its phase: droplets throw most light forward, cloud_march.wgsl phase) is added,
// and it goes on in a direction picked by the same phase, up to U.sky.w times; then it takes the light the last pass
// left there, scattered back along it by the phase (its mean and flow). Passes repeated converge on every order of
// scattering; a pass's long walks carry the light far through a thick cloud, so a few dozen do.
//
// Checked against Mitsuba 3 (volpath, every bounce) on a cumulus of the cumulus preset seen from the sunlit side, with
// isotropic scattering (the transport) and with the droplets' phase (the whole): see docs/lume.md. Lengths in metres of
// sky; grid cells inside.
//!include common.wgsl
//!include noise.wgsl

struct LumeParams {
  g: Grid,             // the cloud's grid: cells, cell size (m), origin (m)
  lv: vec4<f32>,       // this grid's cells (xyz), the cloud's cells per cell of it
  sun: vec4<f32>,      // toward the sun (unit), the pass's seed
  sun_e: vec4<f32>,    // the sun's irradiance above the cloud (rgb), rays per cell
  sky: vec4<f32>,      // the sky's radiance overhead (rgb), scatterings per ray
  hor: vec4<f32>,      // the sky's at the horizon (rgb), share of the last pass's estimate kept (0: replaced)
  ground: vec4<f32>,   // the ground's albedo (rgb), ground there (1/0; 0: the sky below the horizon too)
  glow: vec4<f32>,     // the sun's colour for the glow round it in the sky (cloud_march.wgsl sky_col), the silver lining
};

@group(0) @binding(0) var S: texture_3d<f32>;        // cloud_sig: extinction (1/m) of the cloud (x), of the precipitation (y)
@group(0) @binding(1) var M: texture_3d<f32>;        // cloud_sig: the ice share of the cloud (x), of the precipitation (y)
@group(0) @binding(2) var LT: texture_3d<f32>;       // cloud_light: optical depth to the sun (x)
@group(0) @binding(3) var LVin: texture_3d<f32>;     // the last pass's light (this grid): its mean (rgb)
@group(0) @binding(4) var LVout: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var LDin: texture_3d<f32>;     // ... its first moment (luminance: the mean of L w over the ways w it
@group(0) @binding(6) var LDout: texture_storage_3d<rgba16float, write>;   // comes from, xyz), its mean luminance (w)
@group(1) @binding(0) var<uniform> U: LumeParams;

const PI: f32 = 3.14159265;
const DENSE: f32 = 0.05;   // (cloud_march.wgsl: the densest a cloud is drawn)

var<private> rng: u32;

fn rnd() -> f32 {
  rng = pcg1(rng);
  return f32(rng) * (1.0 / 4294967296.0);
}

fn tri(t: texture_3d<f32>, p: vec3<f32>, n: vec3<i32>) -> vec4<f32> {
  let q = clamp(p - vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(n - vec3<i32>(1)));
  let i = vec3<i32>(floor(q));
  let f = q - vec3<f32>(i);
  let i1 = min(i + vec3<i32>(1), n - vec3<i32>(1));
  let c000 = textureLoad(t, i, 0);
  let c100 = textureLoad(t, vec3<i32>(i1.x, i.y, i.z), 0);
  let c010 = textureLoad(t, vec3<i32>(i.x, i1.y, i.z), 0);
  let c110 = textureLoad(t, vec3<i32>(i1.x, i1.y, i.z), 0);
  let c001 = textureLoad(t, vec3<i32>(i.x, i.y, i1.z), 0);
  let c101 = textureLoad(t, vec3<i32>(i1.x, i.y, i1.z), 0);
  let c011 = textureLoad(t, vec3<i32>(i.x, i1.y, i1.z), 0);
  let c111 = textureLoad(t, i1, 0);
  return mix(mix(mix(c000, c100, f.x), mix(c010, c110, f.x), f.y),
             mix(mix(c001, c101, f.x), mix(c011, c111, f.x), f.y), f.z);
}

// The cloud's extinction at p (cells), as the march draws it (1/m)
fn sig_t(p: vec3<f32>, n: vec3<i32>) -> f32 {
  let s = tri(S, p, n);
  return min(s.x, DENSE) + s.y;
}

fn hg(c: f32, g: f32) -> f32 {
  let g2 = g * g;
  return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * c, 1e-4), 1.5));
}

// The cloud's phase (cloud_march.wgsl phase): droplets a strong forward lobe and a little back; ice crystals broader
fn phase(c: f32, ice: f32) -> f32 {
  let k = U.glow.w;
  return mix(mix(hg(c, -0.25 * k), hg(c, 0.85 * k), 0.8), mix(hg(c, -0.2 * k), hg(c, 0.7 * k), 0.75), ice);
}

// Its mean cosine: the first moment, which scatters the flow of the light (lume_toward)
fn g_mean(ice: f32) -> f32 { return mix(0.63, 0.475, ice) * U.glow.w; }

// A cosine picked by the Henyey-Greenstein lobe g
fn hg_cos(g: f32, u: f32) -> f32 {
  if (abs(g) < 1e-3) { return 1.0 - 2.0 * u; }
  let s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u);
  return clamp((1.0 + g * g - s * s) / (2.0 * g), -1.0, 1.0);
}

// A direction picked by the phase round d (its lobes by their weights)
fn phase_dir(d: vec3<f32>, ice: f32) -> vec3<f32> {
  let k = U.glow.w;
  var g = 0.0;
  if (rnd() < ice) {
    g = select(-0.2 * k, 0.7 * k, rnd() < 0.75);
  } else {
    g = select(-0.25 * k, 0.85 * k, rnd() < 0.8);
  }
  let c = hg_cos(g, rnd());
  let s = sqrt(max(1.0 - c * c, 0.0));
  let a = 2.0 * PI * rnd();
  let t = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(d.x) > 0.9);
  let b1 = normalize(cross(d, t));
  let b2 = cross(d, b1);
  return normalize(d * c + (b1 * cos(a) + b2 * sin(a)) * s);
}

// The sky's radiance along d (cloud_march.wgsl sky_col, without the sun's disc).
fn sky_col(d: vec3<f32>) -> vec3<f32> {
  let up = max(d.y, 0.0);
  var s = mix(U.hor.rgb, U.sky.rgb, pow(up, 0.45));
  let c = dot(d, U.sun.xyz);
  s += U.glow.rgb * (0.04 * hg(c, 0.76) + 0.02 * hg(c, 0.3));
  return s;
}

// What a ray leaving the cloud's box at p going d brings back: the sky, or the ground (sunlit through the cloud's shadow,
// and lit by the sky) where it goes down to it
fn outside(p: vec3<f32>, d: vec3<f32>, n: vec3<i32>) -> vec3<f32> {
  if (U.ground.w > 0.5 && d.y < 0.0) {
    let t = -p.y / d.y;
    let pg = p + d * t;
    var shade = 1.0;
    if (all(pg.xz >= vec2<f32>(0.0)) && all(pg.xz <= vec2<f32>(n.xz))) {
      shade = exp(-tri(LT, vec3<f32>(pg.x, 0.5, pg.z), n).x);
    }
    let e = U.sun_e.rgb * max(U.sun.y, 0.0) * shade + PI * mix(U.sky.rgb, U.hor.rgb, 0.5);
    return U.ground.rgb / PI * e;
  }
  return sky_col(d);
}

fn sphere_dir() -> vec3<f32> {
  let z = 1.0 - 2.0 * rnd();
  let r = sqrt(max(1.0 - z * z, 0.0));
  let a = 2.0 * PI * rnd();
  return vec3<f32>(r * cos(a), z, r * sin(a));
}

// The last pass's light at p (cells) scattered back along -d by the phase there: its mean and flow (L0 + 3 g m.d)
fn cached(p: vec3<f32>, d: vec3<f32>, k: f32, nl: vec3<i32>, ice: f32) -> vec3<f32> {
  let q = p / k;
  let l0 = tri(LVin, q, nl).rgb;
  let m = tri(LDin, q, nl);
  return l0 * clamp(1.0 + 3.0 * g_mean(ice) * dot(m.xyz, d) / max(m.w, 1e-6), 0.0, 4.0);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let nl = vec3<i32>(U.lv.xyz);
  let c = vec3<i32>(id);
  if (any(c >= nl)) { return; }
  let n = gdim(U.g);
  let k = U.lv.w;
  let h = U.g.n.w;
  rng = pcg1(id.x * 73856093u ^ id.y * 19349663u ^ id.z * 83492791u ^ u32(U.sun.w));
  let rays = max(i32(U.sun_e.w), 1);
  let events = max(i32(U.sky.w), 1);
  var acc = vec3<f32>(0.0);
  var mom = vec3<f32>(0.0);
  for (var r = 0; r < rays; r++) {
    // from a point in the cell (cloud cells), every way
    var p = (vec3<f32>(c) + vec3<f32>(rnd(), rnd(), rnd())) * k;
    var d = sphere_dir();
    let d0 = d;
    var lr = vec3<f32>(0.0);
    for (var e = 0; e < events; e++) {
      // to where the cloud scatters it: its optical depth, half a cell a step (the first jittered), each step's
      // extinction at its middle
      let want = -log(max(1.0 - rnd(), 1e-12));
      var tau = 0.0;
      var t = 0.0;
      var st = 0.5 * rnd();
      var hit = false;
      for (var i = 0; i < 1024; i++) {
        let q = p + d * (t + 0.5 * st);
        if (any(q < vec3<f32>(0.0)) || any(q > vec3<f32>(n))) { break; }
        let dt = st * sig_t(q, n) * h;
        if (tau + dt >= want) {
          p = p + d * (t + st * (want - tau) / max(dt, 1e-12));
          hit = true;
          break;
        }
        tau += dt;
        t += st;
        st = 0.5;
      }
      if (!hit) {
        lr += outside(p + d * t, d, n);
        break;
      }
      // scattered at p back along -d: the sun's light there (its beam through the cloud, by the phase), then on
      // (light from further along d, by the phase), or the last pass's light there
      let ice = tri(M, p, n).x;
      lr += U.sun_e.rgb * exp(-tri(LT, p, n).x) * phase(dot(U.sun.xyz, d), ice);
      if (e == events - 1) {
        lr += cached(p, d, k, nl, ice);
        break;
      }
      d = phase_dir(d, ice);
    }
    acc += lr;
    mom += d0 * dot(lr, vec3<f32>(0.2126, 0.7152, 0.0722));
  }
  let est = acc / f32(rays);
  let est_d = vec4<f32>(mom / f32(rays), dot(est, vec3<f32>(0.2126, 0.7152, 0.0722)));
  let keep = clamp(U.hor.w, 0.0, 1.0);
  textureStore(LVout, c, vec4<f32>(mix(est, textureLoad(LVin, c, 0).rgb, keep), 1.0));
  textureStore(LDout, c, mix(est_d, textureLoad(LDin, c, 0), keep));
}
