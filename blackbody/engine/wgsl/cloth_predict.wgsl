// Fabric (engine/cloth.py), start of a substep: the forces on every vertex (gravity, and the air
// moving past the cloth: pressure drag across it and skin friction along it, from the simulated gas
// or the wind outside the box; under a liquid, the water's drag and buoyancy), then the predicted
// position. Pinned vertices go where their pin is.
//
// In a liquid the cloth soaks (P.w, 0 dry .. 1 soaked, over its soak time). Dry cloth floats on the
// air in its weave; as water takes the air's place it sinks, slowly, by what its fibres weigh more
// than water. Soaked cloth carries its water: the air moves it less.
//
// Buffers (vec4 per vertex): X position (fire-local m), inverse mass (0 while pinned); P position at
// the substep's start, wetness; V velocity, fabric index; R rest position in the fabric's own frame,
// inverse mass (negative: pinned); S state (temperature K, burn progress 0..1 (1 = gone), burning
// (1/0), melt / heat dose); N normal, area (m^2).
//!include common.wgsl
//!include cloth_common.wgsl

struct Params {
  sim: vec4<f32>,      // dt, vertex count, gravity (m/s^2, down), gas on (1/0)
  g: Grid,             // the simulation grid (fire-local metres)
  air: vec4<f32>,      // wind outside the box (fire-local m/s), ambient K
  air2: vec4<f32>,     // flame K, max K, _, _
  fab: array<Fab, MAX_FABRICS>,
  lq: vec4<f32>,       // the liquid's grid: dims, cell size (m)
  lo: vec4<f32>,       // its corner (fire-local m), liquid on (1/0)
  lw: vec4<f32>,       // its density (kg/m^3), _
};

@group(0) @binding(0) var<storage, read_write> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> P: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> V: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> R: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> S: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> N: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> M: array<Mat>;
@group(0) @binding(7) var vel: texture_3d<f32>;
@group(0) @binding(8) var scal: texture_3d<f32>;
@group(0) @binding(9) var lin: sampler;
@group(0) @binding(10) var<storage, read_write> IMP: array<vec4<f32>>;   // momentum given to the air (kg m/s), gathered over the substeps;
                                                                          // w: steam boiled off (kg, cloth_drip.wgsl)
@group(0) @binding(11) var LVEL: texture_3d<f32>;    // the liquid's velocity (MAC faces, m/s: liquid.py vel_tex)
@group(0) @binding(12) var LTYPE: texture_3d<f32>;   // its cells: 0 air, 1 liquid, 2 solid
@group(0) @binding(13) var<storage, read> MP: array<vec4<f32>>;   // the matter's push on each vertex (N: cloth_matter.wgsl),
                                                                   // the mass riding on it (kg)
@group(0) @binding(15) var<storage, read_write> LIMP: array<vec4<f32>>;   // momentum it took from the liquid (kg m/s),
                                                                         // gathered until the liquid takes it back
                                                                         // (cloth_liquid.wgsl)
@group(0) @binding(14) var<storage, read> OP: array<vec4<f32>>;   // the push of things lying on it or hitting it this frame
                                                                   // (N: solids.py), the mass riding on it (kg)
@group(1) @binding(0) var<uniform> U: Params;

const MATTER_DAMP: f32 = 20.0;   // 1/s, fully laden

// The air at fire-local point p: its velocity and density (kg/m^3; hot gas is thinner).
fn air_at(p: vec3<f32>) -> vec4<f32> {
  let n = U.g.n.xyz;
  let q = (p - U.g.org.xyz) / U.g.n.w;
  if (U.sim.w < 0.5 || any(q < vec3<f32>(0.0)) || any(q > n)) {
    return vec4<f32>(U.air.xyz, 1.2);
  }
  let v = vel_at(vel, lin, q, n);
  let T = max(samp_c(scal, lin, q, n).x, 0.0);
  let amb = U.air.w;
  let tk = amb + (U.air2.x - amb) * min(T, 1.0) + (U.air2.y - U.air2.x) * (1.0 - exp(-max(T - 1.0, 0.0)));
  return vec4<f32>(v, 1.2 * amb / max(tk, amb));
}

// Velocity component k of the liquid at x (its cells), trilinear over its MAC faces.
fn liquid_face(x: vec3<f32>, k: u32, n: vec3<i32>) -> f32 {
  var off = vec3<f32>(0.5);
  off[k] = 0.0;
  let q = x - off;
  let b = vec3<i32>(floor(q));
  let f = q - floor(q);
  var lim = n - vec3<i32>(1);
  lim[k] = n[k];
  var v = 0.0;
  for (var o = 0; o < 8; o++) {
    let oo = vec3<i32>(o & 1, (o >> 1) & 1, o >> 2);
    let w = mix(vec3<f32>(1.0) - f, f, vec3<f32>(oo));
    v += w.x * w.y * w.z * textureLoad(LVEL, clamp(b + oo, vec3<i32>(0), lim), 0)[k];
  }
  return v;
}

// The liquid at fire-local point p: its velocity (m/s) and how far the point is in it (0..1: the share
// of the cells round it the liquid fills).
fn liquid_at(p: vec3<f32>) -> vec4<f32> {
  if (U.lo.w < 0.5) { return vec4<f32>(0.0); }
  let n = vec3<i32>(U.lq.xyz);
  let x = (p - U.lo.xyz) / U.lq.w;
  if (any(x < vec3<f32>(0.0)) || any(x > vec3<f32>(n))) { return vec4<f32>(0.0); }
  let q = x - vec3<f32>(0.5);
  let b = vec3<i32>(floor(q));
  let f = q - floor(q);
  var share = 0.0;
  for (var o = 0; o < 8; o++) {
    let oo = vec3<i32>(o & 1, (o >> 1) & 1, o >> 2);
    let w = mix(vec3<f32>(1.0) - f, f, vec3<f32>(oo));
    let t = textureLoad(LTYPE, clamp(b + oo, vec3<i32>(0), n - vec3<i32>(1)), 0).x;
    share += w.x * w.y * w.z * select(0.0, 1.0, t > 0.5 && t < 1.5);
  }
  if (share <= 1e-3) { return vec4<f32>(0.0); }
  return vec4<f32>(liquid_face(x, 0u, n), liquid_face(x, 1u, n), liquid_face(x, 2u, n), share);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.sim.y)) { return; }
  let dt = U.sim.x;
  let fi = u32(V[i].w + 0.5);
  let fb = U.fab[fi];
  let mat = M[fi];
  var x = X[i];
  var wet = clamp(P[i].w, 0.0, 1.0);
  let r = R[i];
  let st = S[i];
  let gone = st.y >= 1.0;
  let lq = liquid_at(x.xyz);
  let sub = lq.w;
  if (!gone && sub > 0.0) {
    // soaking: water takes the place of the air in the weave
    wet += (1.0 - wet) * (1.0 - exp(-sub * dt / max(mat.f.z, 0.05)));
  }
  P[i] = vec4<f32>(x.xyz, wet);
  if (gone && st.z < -0.5) {
    // torn away (cloth_tear.wgsl): it stays where it tore, out of everything (as if pinned, so nothing moves it)
    V[i] = vec4<f32>(0.0, 0.0, 0.0, V[i].w);
    X[i] = vec4<f32>(x.xyz, 0.0);
    return;
  }
  if (r.w < 0.0 && fb.scl.w < 0.5 && !gone) {
    // pinned: it follows its pin (the fabric's placement, animated or not)
    let goal = fabric_to_world(fb, r.xyz);
    V[i] = vec4<f32>((goal - x.xyz) / max(dt, 1e-6), V[i].w);
    X[i] = vec4<f32>(goal, 0.0);
    return;
  }
  var v = V[i].xyz;
  // per kg of fibre: it weighs 1 + the water it holds; under the liquid it is buoyed up by the water its
  // fibres and its pores (whether air or water is in them) push aside
  let held = wet * mat.f.y;
  let carry = 1.0 + held;
  var down = carry;
  if (sub > 0.0 && !gone) { down -= sub * (U.lw.x / max(mat.f.x, 100.0) + mat.f.y); }
  // sand, snow, mud and the like lying on it or landing on it: their push through the last frame, and their weight's
  // worth of mass, which rides on it (a sling of sand moves as the sand does: pushed alone, the light cloth would fly)
  let own = carry / max(abs(r.w), 1e-9);       // (kg: its fibre and the water in it)
  let load = select(0.0, MP[i].w + OP[i].w, !gone);
  v.y -= U.sim.z * dt * down / carry * (own / (own + load));
  if (!gone) { v += (MP[i].xyz + OP[i].xyz) * (dt / (own + load)); }
  let nn = N[i];
  let area = nn.w;
  if (area > 0.0 && !gone) {
    // drag: pressure drag across the sheet and friction along it (Cn, Ct), of the air and, where it is
    // in it, the liquid, each applied exactly over the step (it can only bring the cloth up to the
    // fluid's speed, never past it)
    let a = air_at(x.xyz);
    let mass = own + load;
    let flow = mix(a.xyz, lq.xyz, sub);
    let rho = mix(a.w, U.lw.x, sub);
    let rel = flow - v;
    let n = nn.xyz;
    let vn = dot(rel, n);
    let vt = rel - vn * n;
    let kn = 0.5 * rho * mat.b.x * abs(vn) * area / mass;
    let kt = 0.5 * rho * mat.b.y * length(vt) * area / mass;
    let dv = n * (vn * (1.0 - exp(-kn * dt))) + vt * (1.0 - exp(-kt * dt));
    v += dv;
    // what the cloth gains from the air, the air loses (cloth_air.wgsl gives it back to the gas), and the water what
    // it gains from the water (cloth_liquid.wgsl)
    IMP[i] = vec4<f32>(IMP[i].xyz - dv * mass * (1.0 - sub), IMP[i].w);
    if (sub > 0.0) { LIMP[i] = vec4<f32>(LIMP[i].xyz + dv * mass * sub, 0.0); }
  }
  v *= exp(-mat.a.y * dt);
  // and where it carries matter, the grains' friction takes the swing out of it (a sling of sand does not bounce)
  v *= exp(-MATTER_DAMP * dt * select(0.0, MP[i].w, !gone) / (own + load));   // (sand, not a thing that fell on it)
  // a vertex that has burnt away is a flake of ash: light, and carried by the air (or the water)
  if (gone) {
    let a = air_at(x.xyz);
    v = mix(v, mix(a.xyz, lq.xyz, sub), 1.0 - exp(-8.0 * dt));
  }
  // (in the constraints, a laden vertex weighs at most ten times its own: past that the threads converge too
  // slowly and a laden sling stretches like elastic)
  let m0 = 1.0 / max(abs(r.w), 1e-9);
  X[i] = vec4<f32>(x.xyz + v * dt, 1.0 / (m0 + min(load, 10.0 * m0)));
  V[i] = vec4<f32>(v, V[i].w);
}
