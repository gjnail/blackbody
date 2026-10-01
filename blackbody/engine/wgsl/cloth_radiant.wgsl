// Fire -> fabric, radiation (3 of 3), every substep after cloth_finish.wgsl: the cloth absorbs the
// radiation falling on it (cloth_rad.wgsl; textiles absorb most of a fire's infrared, whatever their
// colour) and radiates its own heat away, to the cooler surroundings beyond the gas it is in. The air
// carrying heat off both faces is cloth_finish.wgsl's, so a cloth near a fire settles where the two
// balance: about 15 kW/m^2 (the fire's radiation some 30 cm from campfire flames) scorches cotton, and
// about 25 kW/m^2 lights it. Wet cloth cannot pass the boil: the heat boils its water off instead (the
// steam leaves it into the cool air beside the fire, where it shows), and it holds at 100 C until dry.
// A burning or burnt-away spot is cloth_finish.wgsl's.
//!include common.wgsl
//!include cloth_common.wgsl

const SIGMA: f32 = 5.670e-8;
const ABSORB: f32 = 0.85;      // share of a fire's radiation a textile absorbs; also how well it radiates

struct Params {
  sim: vec4<f32>,    // dt, vertex count, gas on (1/0), _
  g: Grid,
  air: vec4<f32>,    // ambient K, flame K, max K, _
};

@group(0) @binding(0) var<storage, read_write> S: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> P: array<vec4<f32>>;   // w: wetness
@group(0) @binding(2) var<storage, read> QR: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> M: array<Mat>;
@group(0) @binding(4) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(5) var scal: texture_3d<f32>;
@group(0) @binding(6) var lin: sampler;
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
  let st = S[i];
  if (gone(st) || st.z > 0.5) { return; }
  let dt = U.sim.x;
  let mat = M[u32(V[i].w + 0.5)];
  let p = P[i];
  var wet = clamp(p.w, 0.0, 1.0);
  let held = wet * mat.f.y;
  let cap = 1.0 + held * C_WATER / C_FIBRE;
  var T = st.x;
  // it radiates its own heat away (from both faces) when hotter than its surroundings: the cool air and
  // walls beyond, or the gas it is in (what the gas radiates onto it is already in QR, from the flames'
  // real temperature)
  let te = max(gas_kelvin(p.xyz), U.air.x);
  let net = ABSORB * QR[i].x - 2.0 * ABSORB * SIGMA * max(pow(T, 4.0) - pow(te, 4.0), 0.0);
  T = min(T + net * dt / max(mat.a.x * C_FIBRE * cap, 1e-6), U.air.z);
  if (held > 0.0 && T > 373.15) {
    // at the boil the heat boils water off instead of warming it further
    let boiled = (T - 373.15) * C_FIBRE * cap / LATENT;
    wet = max(held - boiled, 0.0) / max(mat.f.y, 1e-6);
    if (wet < 1e-3) { wet = 0.0; }
    T = 373.15;
    P[i] = vec4<f32>(p.xyz, wet);
  }
  S[i] = vec4<f32>(T, st.y, st.z, st.w);
}
