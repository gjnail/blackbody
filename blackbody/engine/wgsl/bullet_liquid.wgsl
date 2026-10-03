// Bullets into water (engine/bullet_media.py, ballistics.py): what a bullet does to the liquid along its track, as far
// as it has gone this frame (s from s0 to s1 along it). The energy it leaves per metre (dE/ds) opens a cavity behind it:
// the water round its path is driven outward as an incompressible radial flow, u = K / r out to the cavity's radius R
// (K^2 = eta dE/ds / (pi rho ln(R / a))), dragged a little forward near the path. At the entry the flow turns up and
// back out of the hole: the crown of the splash, and the jet thrown back along the way the bullet came; where it comes
// out of the water (a shallow layer, a tank's far side) the last of it sprays on after it. The cavity then closes on
// itself under the water's pressure (the solver), throwing up the column of the splash.
//!include common.wgsl
//!include liq_common.wgsl

const TRACK: u32 = 32u;

struct KickU {
  g: Grid,
  a: vec4<f32>,                // the entry point (fire-local m), s0 (m)
  u: vec4<f32>,                // the bullet's direction (unit), s1 (m)
  b: vec4<f32>,                // bullet radius (m), the flow's core (m), the fastest it moves water (m/s), particle slots
  e: vec4<f32>,                // splash: depth (m) it sprays from, share turned back out, how far it went in all (m), exit (1/0)
  t: array<vec4<f32>, TRACK>,  // along the track: s (m), K (m^2/s), R (m), the bullet's speed (m/s)
};

@group(0) @binding(0) var<storage, read_write> PART: array<Particle>;
@group(1) @binding(0) var<uniform> K: KickU;

fn track_at(s: f32) -> vec3<f32> {
  var prev = K.t[0];
  if (s <= prev.x) { return prev.yzw; }
  for (var j = 1u; j < TRACK; j++) {
    let q = K.t[j];
    if (q.x < prev.x) { break; }
    if (s <= q.x) {
      let f = (s - prev.x) / max(q.x - prev.x, 1e-9);
      return mix(prev.yzw, q.yzw, f);
    }
    prev = q;
  }
  return vec3<f32>(0.0);
}

fn pcg1(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

fn hash11(x: u32) -> f32 { return f32(pcg1(x)) * (1.0 / 4294967296.0); }

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = id.x + id.y * nwg.x * 64u;
  if (i >= u32(K.b.w)) { return; }
  var P = PART[i];
  if (!alive(P)) { return; }
  let w = K.g.org.xyz + P.p.xyz * K.g.n.w;
  let u = K.u.xyz;
  let s = dot(w - K.a.xyz, u);
  if (s < -0.03 || s > K.u.w + 0.01) { return; }
  if (s >= 0.0 && s < K.a.w) { return; }
  if (s < 0.0 && K.a.w > 0.0) { return; }      // (the water round the hole is thrown once, as the bullet goes in)
  let off = w - K.a.xyz - u * s;
  let r = length(off);
  let reach = K.e.z;
  let tr = track_at(clamp(s, 0.0, reach));
  let R = tr.y;
  if (r > R || R <= 0.0) { return; }
  let a = K.b.x;
  let rc = max(K.b.y, 1e-4);
  var radial = off / max(r, 1e-6);
  if (r < 1e-5) { radial = normalize(vec3<f32>(hash11(i * 3u + 1u), hash11(i * 3u + 2u), hash11(i * 3u + 3u)) - vec3<f32>(0.5)); }
  // (u = K / r outside the core, K r / core^2 inside it: bullet_media.track_samples)
  let prof = select(rc / max(r, 1e-6), r / rc, r < rc);
  let sp = min(tr.x / rc * prof * (1.0 - smoothstep(0.6 * R, R, r)), K.b.z);
  var dv = radial * sp + u * (0.1 * tr.z * exp(-r / (2.0 * max(a, rc))));
  // the splash: near the entry the flow turns up and back out of the hole
  let depth = max(s, 0.0);
  let up = K.e.y * (1.0 - smoothstep(0.0, K.e.x, depth));
  if (up > 0.0) {
    let back = normalize(-u + radial * (0.8 + 0.6 * hash11(i * 7u + 11u)) + vec3<f32>(0.0, 1.0, 0.0));
    dv = mix(dv, back * (sp + 0.02 * tr.z) * (0.4 + 0.8 * hash11(i * 5u + 17u)), up);
  }
  if (K.e.w > 0.5 && s > reach - 1.5 * K.e.x) {
    dv += u * (0.08 * tr.z) * (0.5 + hash11(i * 13u + 5u));
  }
  P.v = P.v + dv;
  PART[i] = P;
}
