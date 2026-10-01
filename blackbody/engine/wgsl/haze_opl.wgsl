// Heat haze from the simulation: how much the fire's hot air changes the optical path of light crossing it,
// for each pixel. Hot air is thinner than the air around it, so its refractive index is lower (n - 1 falls
// as 1/T), and the composite displaces the footage behind by the gradient of this path across the picture.
// Eddies too small for the grid stir the hot air in proportion to how hot it is and drift with the
// simulated flow, so the shimmer rises with the plume and ends where the hot air ends.
// out: x = extra optical path (micrometres, negative through hot air), y = depth of the hot air (m), z = weight
//!include common.wgsl

struct Params {
  inv_vp: mat4x4<f32>,   // clip -> world
  w2g: mat4x4<f32>,      // world -> grid (cells)
  n: vec4<f32>,          // grid dims, metres per cell
  res: vec4<f32>,        // width, height of this pass
  k: vec4<f32>,          // ambient K, flame K (temperature 1), hottest K
  turb: vec4<f32>,       // small eddies: strength, frequency (1/m), speed, time (s)
  org: vec4<f32>,        // grid corner (fire-local m), w = grid cells per velocity cell
};

@group(0) @binding(0) var scal: texture_3d<f32>;
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var noise: texture_3d<f32>;
@group(0) @binding(3) var lin: sampler;
@group(0) @binding(4) var rep: sampler;
@group(0) @binding(5) var out_opl: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

const N_AIR: f32 = 0.0844;     // (n - 1) T of air (K): n - 1 = 2.93e-4 at 288 K
const EDDY_PERIOD: f32 = 0.4;  // s: each eddy pattern drifts this long before fading into the next

// Field temperature -> Kelvin, as shade.wgsl.
fn kelvin(t_in: f32) -> f32 {
  let t = max(t_in, 0.0);
  return U.k.x + (U.k.y - U.k.x) * min(t, 1.0) + (U.k.z - U.k.y) * (1.0 - exp(-max(t - 1.0, 0.0)));
}

fn eddy_offset(c: f32) -> vec3<f32> {
  return fract(vec3<f32>(c * 0.6180339, c * 0.4142135, c * 0.7320508)) * vec3<f32>(17.0, 23.0, 31.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let px = vec2<i32>(id.xy);
  if (f32(px.x) >= U.res.x || f32(px.y) >= U.res.y) { return; }
  let n = U.n.xyz;
  let h = U.n.w;
  let uv = (vec2<f32>(px) + vec2<f32>(0.5)) / U.res.xy;
  let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let pn = U.inv_vp * vec4<f32>(ndc, 0.0, 1.0);
  let pf = U.inv_vp * vec4<f32>(ndc, 1.0, 1.0);
  let ro_w = pn.xyz / pn.w;
  let rd_w = normalize(pf.xyz / pf.w - ro_w);
  let ro = (U.w2g * vec4<f32>(ro_w, 1.0)).xyz;
  let rdg = (U.w2g * vec4<f32>(rd_w, 0.0)).xyz;
  let cells_per_m = length(rdg);
  let rd = rdg / cells_per_m;
  let safe = select(rd, vec3<f32>(1e-8), abs(rd) < vec3<f32>(1e-8));
  let inv = 1.0 / safe;
  let ta = (vec3<f32>(0.0) - ro) * inv;
  let tb = (n - ro) * inv;
  let t_in = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), max(min(ta.z, tb.z), 0.0));
  let t_out = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z));
  if (t_out <= t_in) {
    textureStore(out_opl, px, vec4<f32>(0.0));
    return;
  }

  let amb = N_AIR / U.k.x;
  let vk = max(U.org.w, 1.0);
  let f = U.turb.y;
  let tau = U.turb.w * U.turb.z / EDDY_PERIOD;
  let ph0 = fract(tau);
  let ph1 = fract(tau + 0.5);
  let off0 = eddy_offset(floor(tau));
  let off1 = eddy_offset(floor(tau + 0.5) + 0.5);
  let w0 = 1.0 - abs(2.0 * ph0 - 1.0);
  let ds = h;   // one cell per step
  var opl = 0.0;
  var dsum = 0.0;
  var wsum = 0.0;
  var t = t_in + 0.5;
  for (var i = 0; i < 1024; i++) {
    if (t >= t_out) { break; }
    let p = ro + rd * t;
    let dn = N_AIR / kelvin(samp_c(scal, lin, p, n).x) - amb;   // negative where the air is hot
    if (dn < -1e-8) {
      var e = dn;
      if (U.turb.x > 0.0) {
        // two eddy patterns carried by the local flow, each fading out as the other fades in
        let pw = U.org.xyz + p * h;
        let v = vel_at(vel, lin, p / vk, n / vk);
        let a = textureSampleLevel(noise, rep, (pw - v * (ph0 * EDDY_PERIOD)) * f + off0, 0.0).x;
        let b = textureSampleLevel(noise, rep, (pw - v * (ph1 * EDDY_PERIOD)) * f + off1, 0.0).x;
        let nz = ((a - 0.5) * w0 + (b - 0.5) * (1.0 - w0)) * 5.5;   // about unit spread
        e = dn * (1.0 + U.turb.x * nz);
      }
      opl += e * ds;
      dsum += -dn * ds * (t / cells_per_m);
      wsum += -dn * ds;
    }
    t += 1.0;
  }
  textureStore(out_opl, px, vec4<f32>(opl * 1.0e6, select(0.0, dsum / wsum, wsum > 0.0), wsum * 1.0e6, 0.0));
}
