// Molten liquids: temperatures and blackbody light, shared by the march (liq_lava.wgsl) and the light
// the molten surface casts on its surroundings (liq_lava_cols.wgsl). Needs U.lava and U.lava_bb
// (LiquidRenderer.lava_uniforms).

// Radiance of a blackbody at T (K), in the look's units (glow x exposure at 1300 K).
fn lava_bb(T: f32) -> vec3<f32> {
  let f = (T - U.lava[4].x) * U.lava[4].y;
  if (f <= 0.0) { return vec3<f32>(0.0); }
  let top = U.lava[4].z - 1.0;
  let fc = min(f, top - 1e-3);
  let i = u32(fc);
  let t = fc - f32(i);
  let a = U.lava_bb[i];
  let b = U.lava_bb[i + 1u];
  var lg = mix(a.w, b.w, t);
  if (f > top) { lg += (f - top) * (b.w - a.w); }    // past the table: on up at its last slope
  return mix(a.rgb, b.rgb, t) * pow(10.0, lg) * U.lava[0].y;
}

struct LavaT {
  Tc: f32,      // the melt just under the surface (K)
  Ts: f32,      // the skin on top (K)
  young: f32,   // 1 fresh melt .. 0 old crust
};

// Temperatures from the heat just under the surface (hc) and how long the skin has been out in the air
// (tau, s): the skin cools far faster than the simulation's cells can show, darkening over Crust forms in.
fn lava_temps(hc: f32, tau: f32) -> LavaT {
  let Ta = U.lava[0].w;
  let Tc = Ta + (U.lava[0].z - Ta) * hc;
  let young = exp(-tau / max(U.lava[3].x, 1e-3));
  let Ts = Tc - (Tc - Ta) * clamp(U.lava[1].y, 0.0, 1.0) * (1.0 - young) * 0.88;
  return LavaT(Tc, Ts, young);
}

// The skin's age (s) read back from the heat it has lost at the simulation's cooling rate (heat =
// exp(-cooling t) out in the air), where the particles' own count of it is missing.
fn lava_tau_from_heat(hs: f32) -> f32 {
  if (U.lava[3].y <= 0.0) { return 0.0; }
  return -log(clamp(hs, 1e-4, 1.0)) / U.lava[3].y;
}

// Mean radiance of a molten surface: its skin, and the share of it that is glowing cracks.
fn lava_mean_glow(t: LavaT) -> vec3<f32> {
  let f = mix(1.0, 0.1, clamp(U.lava[1].y, 0.0, 1.0) * (1.0 - t.young));
  return lava_bb(t.Ts) * (1.0 - f) + lava_bb(mix(t.Ts, t.Tc, 0.85)) * f;
}
