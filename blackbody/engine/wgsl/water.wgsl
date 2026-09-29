// Water: saturation vapour density, shared by the reaction (condensation and its latent heat) and
// the renderer (visible steam).

// Saturation vapour density of water (g/m^3) at a temperature in Kelvin (Magnus formula).
fn vapour_saturation(tk: f32) -> f32 {
  let tc = tk - 273.15;
  let es = 610.94 * exp(min(17.625 * tc / max(tc + 243.04, 1.0), 60.0));
  return es / (461.5 * max(tk, 1.0)) * 1000.0;
}
