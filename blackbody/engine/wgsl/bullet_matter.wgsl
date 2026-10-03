// Bullets in sand, snow, mud, jelly and clay (engine/bullet_media.py, ballistics.py).
//
// kick: what a bullet does to the matter along its track, as far as it has gone this frame (s from s0 to s1 along it).
// Inside, the energy it leaves per metre (dE/ds, from how it slows: ballistics.penetrate) drives the matter round its
// path outward as an incompressible radial flow, u = K / r out to the cavity's radius R (Gibbs' "temporary cavity":
// K^2 = eta dE/ds / (pi rho ln(R / a)), so the flow carries a share eta of the energy), with a little of the bullet's
// own speed forward along its path near it. Jelly and clay spring back from it (their own elasticity: the cavity
// swells and collapses); sand, snow and mud are thrown, and near the entry where the matter is shallow the throw turns
// back out of the hole as a spray of ejecta, a cone round the way the bullet came.
//!include mpm_common.wgsl

const TRACK: u32 = 32u;   // samples along a track

struct KickU {
  n: vec4<f32>,                // grid nodes, particles (count)
  o: vec4<f32>,                // grid corner (fire-local m), node spacing (m)
  a: vec4<f32>,                // the entry point (fire-local m), s0 (m): how far along the track was kicked before
  u: vec4<f32>,                // the bullet's direction (unit), s1 (m): how far along it is now
  b: vec4<f32>,                // bullet radius (m), the flow's core (m), the fastest it moves matter (m/s), seed
  e: vec4<f32>,                // ejecta: depth (m) it sprays from, share turned back out, how far it went in all (m), exit (1/0)
  t: array<vec4<f32>, TRACK>,  // along the track: s (m), K (m^2/s), R (m), the bullet's speed (m/s)
  mats: array<MMat, MAX_MATS>,
};

@group(0) @binding(0) var<storage, read_write> P: array<MParticle>;
@group(1) @binding(0) var<uniform> K: KickU;

// the track's samples at s: (K, R, bullet speed), by linear interpolation
fn track_at(s: f32) -> vec3<f32> {
  var prev = K.t[0];
  if (s <= prev.x) { return prev.yzw; }
  for (var j = 1u; j < TRACK; j++) {
    let q = K.t[j];
    if (q.x < prev.x) { break; }           // (past its last sample)
    if (s <= q.x) {
      let f = (s - prev.x) / max(q.x - prev.x, 1e-9);
      return mix(prev.yzw, q.yzw, f);
    }
    prev = q;
  }
  return vec3<f32>(0.0, 0.0, 0.0);
}

fn hash11(x: u32) -> f32 { return f32(pcg1(x)) * (1.0 / 4294967296.0); }

fn pcg1(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}


@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(K.n.w)) { return; }
  var p = P[i];
  if (p.x.w < 0.0) { return; }
  let m = K.mats[u32(p.x.w + 0.5)];
  let w = K.o.xyz + p.x.xyz * K.o.w;
  let u = K.u.xyz;
  let s = dot(w - K.a.xyz, u);
  let reach = K.e.z;
  if (s < -0.05 || s > K.u.w + 0.01) { return; }
  let off = w - K.a.xyz - u * s;
  let r = length(off);
  let tr = track_at(clamp(s, 0.0, reach));
  let R = tr.y;
  if (r > R || R <= 0.0) { return; }
  // (only the stretch the bullet went through this frame, and the matter just ahead of the entry for the spray)
  if (s >= 0.0 && (s < K.a.w || s > K.u.w)) { return; }
  if (s < 0.0 && K.a.w > 0.0) { return; }      // (the matter round the hole is thrown once, as the bullet goes in)
  let model = i32(m.a.x + 0.5);
  let rho = m.a.w;
  let a = K.b.x;
  let rc = max(K.b.y, 1e-4);
  let radial = select(off / max(r, 1e-6), normalize(vec3<f32>(hash11(i * 3u + 1u), hash11(i * 3u + 2u), hash11(i * 3u + 3u)) - vec3<f32>(0.5)), r < 1e-5);
  // the radial flow, fading to nothing at the cavity's edge
  // (u = K / r outside the core, K r / core^2 inside it: bullet_media.track_samples)
  let prof = select(rc / max(r, 1e-6), r / rc, r < rc);
  let sp = min(tr.x / rc * prof * (1.0 - smoothstep(0.6 * R, R, r)), K.b.z);
  var dv = radial * sp;
  // dragged forward close to the path
  dv += u * (0.12 * tr.z * exp(-r / (2.0 * max(a, rc))));
  // near the entry the flow turns back out of the hole: ejecta (more of it in loose matter), and before the entry
  // (s < 0: the matter lying on the surface round the hole) it is thrown up and out
  let depth = max(s, 0.0);
  let eject = K.e.y * (1.0 - smoothstep(0.0, K.e.x, depth));
  if (eject > 0.0) {
    let back = normalize(-u * 1.2 + radial * (0.6 + 0.6 * hash11(i * 7u + 11u)) + vec3<f32>(0.0, 0.5, 0.0));
    dv = mix(dv, back * (0.6 * sp + 0.04 * tr.z) * (0.5 + hash11(i * 5u + 17u)), eject);
  }
  // out of the far side: the last of it sprays on after the bullet
  if (K.e.w > 0.5 && s > reach - 1.5 * K.e.x) {
    dv += u * (0.1 * tr.z) * (0.5 + hash11(i * 13u + 5u));
  }
  // jelly and clay hold together: the flow is gentler there, their elasticity makes the cavity
  if (model == 0 || model == 4) { dv *= 0.85; }
  // (a particle held still until its release takes the kick when it is let go: it is free from now)
  p.c1 = vec4<f32>(p.c1.xyz, 0.0);
  p.v = vec4<f32>(p.v.xyz + dv, p.v.w);
  P[i] = p;
}
