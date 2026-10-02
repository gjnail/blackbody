// Fabric, end of a substep: the velocity from how far each vertex moved, then how hot the cloth is and
// whether it burns. The cloth heats toward the gas around it (thin cloth quickly, heavy cloth slowly:
// its heat capacity over the heat the air brings to both faces) and spreads its heat along itself.
// Past its ignition temperature it catches and burns at its own burning temperature, giving off fuel
// (cloth_couple.wgsl feeds it to the fire), charring, and burning away into holes. It goes out if the
// heat around it cannot keep it above its ignition temperature (wool puts itself out). A synthetic
// shrinks away as it softens and melts away before it burns. Wet cloth (P.w) heats slowly, its water
// taking the heat too, and stops at the boil until the water has boiled off: it cannot catch until it
// is dry, and a soaked spot goes out.
//!include common.wgsl
//!include cloth_common.wgsl

struct Params {
  sim: vec4<f32>,    // dt, vertex count, gas on (1/0), _
  g: Grid,
  air: vec4<f32>,    // ambient K, flame K, max K, _
};

@group(0) @binding(0) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> P: array<vec4<f32>>;   // (w: wetness)
@group(0) @binding(2) var<storage, read_write> V: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> R: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> S_in: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> S_out: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> M: array<Mat>;
@group(0) @binding(7) var<storage, read> NB: array<u32>;   // per vertex: first neighbour, count, then the neighbour lists
@group(0) @binding(8) var scal: texture_3d<f32>;
@group(0) @binding(9) var lin: sampler;
@group(1) @binding(0) var<uniform> U: Params;

fn gas_kelvin(p: vec3<f32>) -> f32 {
  let n = U.g.n.xyz;
  let q = (p - U.g.org.xyz) / U.g.n.w;
  let amb = U.air.x;
  if (U.sim.z < 0.5 || any(q < vec3<f32>(0.0)) || any(q > n)) { return amb; }
  let T = max(samp_c(scal, lin, q, n).x, 0.0);
  return amb + (U.air.y - amb) * min(T, 1.0) + (U.air.z - U.air.y) * (1.0 - exp(-max(T - 1.0, 0.0)));
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.sim.y)) { return; }
  let dt = U.sim.x;
  let x = X[i].xyz;
  let fi = V[i].w;
  V[i] = vec4<f32>((x - P[i].xyz) / max(dt, 1e-9), fi);
  var st = S_in[i];
  if (gone(st)) {
    // burnt off: a flake of char (or a drop of melt) carried by the air. It cools toward the gas round
    // it in a moment, glowing as it goes (in the flames it stays hot); w counts its age (s). (Torn away,
    // cloth_tear.wgsl: z -1, and no flake.)
    let tg0 = gas_kelvin(X[i].xyz);
    S_out[i] = vec4<f32>(tg0 + (st.x - tg0) * exp(-dt / 0.6), st.y, select(0.0, -1.0, st.z < -0.5), st.w + dt);
    return;
  }
  let mat = M[u32(fi + 0.5)];
  let tg = gas_kelvin(x);
  let burning = st.z > 0.5;
  let ign = mat.b.z / max(mat.e.z, 0.05);
  var env = tg;
  if (burning) { env = max(tg, mat.c.x); }
  var wet = clamp(P[i].w, 0.0, 1.0);
  let held = wet * mat.f.y;                     // kg of water per kg of fibre
  let cap = 1.0 + held * C_WATER / C_FIBRE;     // its heat capacity, relative to dry
  var T = st.x + (env - st.x) * (1.0 - exp(-dt / max(mat.c.w * cap, 1e-3)));
  // heat spreads along the cloth to its neighbours (conduction, and the flame's heat just past its edge)
  let first = NB[i * 2u];
  let cnt = NB[i * 2u + 1u];
  var lap = 0.0;
  for (var k = 0u; k < cnt; k++) {
    let j = NB[first + k];
    let sj = S_in[j];
    let tj = select(sj.x, max(sj.x, mat.c.x), sj.z > 0.5);
    let l2 = max(dot(R[j].xyz - R[i].xyz, R[j].xyz - R[i].xyz), 1e-8);
    lap += (tj - st.x) / l2;
  }
  T += clamp(mat.e.x * lap * dt / cap, -0.2 * abs(T - U.air.x) - 50.0, 400.0);
  if (held > 0.0) {
    // drying: below the boil the water evaporates slowly, faster the hotter it is; at the boil the heat
    // that would warm it further boils water off instead
    var boiled = held * dt / 900.0 * exp((min(T, 373.15) - 373.15) / 22.0);
    if (T > 373.15) {
      boiled += (T - 373.15) * C_FIBRE * cap / LATENT;
      T = 373.15;
    }
    wet = max(held - boiled, 0.0) / max(mat.f.y, 1e-6);
    if (wet < 1e-6) { wet = 0.0; }   // (only a trace: the front of water wicking into dry cloth is far wetter)
  }
  P[i] = vec4<f32>(P[i].xyz, wet);
  var prog = st.y;
  var on = st.z;
  var w4 = st.w;
  if (mat.d.w > 0.5) {
    if (!burning && T > ign && wet <= 0.0) { on = 1.0; }
    if (wet > 0.2) { on = 0.0; }   // a soaked spot cannot keep burning
    if (on > 0.5) {
      prog += dt / max(mat.b.w, 1e-3);
      // it goes out when the heat around it cannot keep it burning
      if (T < 0.85 * ign && tg < ign) { on = 0.0; }
    }
  }
  if (mat.d.y > 0.0) {
    // synthetics: shrink (for good) as they soften, and melt away
    w4 = max(w4, smoothstep(mat.d.x, mat.d.y, T));
    if (T > mat.d.y) { prog += dt / 0.4; }
  } else {
    // scorching: browns with its time above about 150 C, before it chars
    w4 = min(1.0, w4 + dt * max(T - 420.0, 0.0) / 1500.0);
  }
  if (prog >= 1.0) {
    prog = 1.0;
    on = 0.0;
    w4 = 0.0;   // (from now on its age as a flake)
  }
  S_out[i] = vec4<f32>(T, prog, on, w4);
}
