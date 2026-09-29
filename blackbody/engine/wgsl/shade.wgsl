//!include water.wgsl
// Fire and smoke shading shared by the light-volume and ray-march kernels.
// The including file declares `bb` (blackbody LUT, texture_2d) and `lin` (linear sampler), and
// defines BB_MIN / BB_MAX (Kelvin range of the LUT).

struct Look {
  fire: vec4<f32>,    // ambient K, flame K at temperature 1, max K, dynamic range (1 = physical)
  fire2: vec4<f32>,   // radiance of an optically thick flame at flame K (exposure included), flame absorption (1/m), soot glow, flame occlusion (0..1)
  blue: vec4<f32>,    // blue-core colour * strength, ignition temperature
  shape: vec4<f32>,   // flame sharpness (exponent), flame threshold, flame density, _
  smoke: vec4<f32>,   // smoke albedo (rgb), extinction per unit soot (1/m)
  amb: vec4<f32>,     // ambient/sky light (rgb), phase anisotropy g
  sun: vec4<f32>,     // key light colour * intensity, fire-light scatter
  sundir: vec4<f32>,  // direction toward the key light (world), shadow density multiplier
  detail: vec4<f32>,  // amount, frequency (1/m), displacement (cells), rise speed (m/s)
  misc: vec4<f32>,    // log10 luminance at flame K, edge fade (cells), step (cells), time (s)
  steam: vec4<f32>,   // steam albedo (rgb), extinction per g/m^3 of droplets (1/m)
  air: vec4<f32>,     // the air's own water vapour (g/m^3), colourant radiance gain, vapour on (1/0), colourant on (1/0)
  ms: vec4<f32>,      // multiple scattering (share of the light kept per bounce), _, _, _
};

// Field temperature -> Kelvin: linear up to the flame temperature at T = 1, then easing toward max.
fn kelvin(T: f32, L: Look) -> f32 {
  let t = max(T, 0.0);
  return L.fire.x + (L.fire.y - L.fire.x) * min(t, 1.0) + (L.fire.z - L.fire.y) * (1.0 - exp(-max(t - 1.0, 0.0)));
}

fn bb_lookup(Tk: f32) -> vec4<f32> {
  let u = clamp((Tk - BB_MIN) / (BB_MAX - BB_MIN), 0.0, 1.0);
  return textureSampleLevel(bb, lin, vec2<f32>(u, 0.5), 0.0);
}

// Normalised blackbody radiance: chromaticity times brightness relative to the flame temperature.
fn bb_radiance(T: f32, L: Look) -> vec3<f32> {
  let b = bb_lookup(kelvin(T, L));
  return b.rgb * min(pow(10.0, (b.a - L.misc.x) * L.fire.w), 1.0e4);
}

// Absorption coefficient of visible flame (1/m).
fn flame_sigma(s: vec4<f32>, L: Look) -> f32 {
  let fl = max(max(s.w, 0.0) * L.shape.z - L.shape.y, 0.0);
  return pow(fl, L.shape.x) * L.fire2.y;
}

// Emitted radiance per metre. Following Kirchhoff's law, flame and hot soot emit in proportion to
// how strongly they absorb, so an optically thick flame saturates at its blackbody radiance: big
// fires do not blow out and small, thin flames stay translucent, as real ones do.
// Flame colourants (metal salts) glow in their own line colours wherever the gas is hot.
fn colourant_emission(s: vec4<f32>, ch: vec4<f32>, L: Look) -> vec3<f32> {
  if (L.air.w < 0.5) { return vec3<f32>(0.0); }
  return max(ch.rgb, vec3<f32>(0.0)) * L.air.y * smoothstep(L.blue.w * 0.6, L.blue.w * 1.6 + 0.05, s.x);
}

fn emission_k(s: vec4<f32>, ch: vec4<f32>, sf: f32, ss: f32, L: Look) -> vec3<f32> {
  let B = bb_radiance(s.x, L) * L.fire2.x;
  let blue = L.blue.rgb * max(s.y, 0.0) * smoothstep(L.blue.w, L.blue.w + 0.25, s.x);
  return B * (sf + ss * L.fire2.z * clamp(s.x, 0.0, 1.0)) + blue + colourant_emission(s, ch, L);
}

fn emission(s: vec4<f32>, ch: vec4<f32>, L: Look) -> vec3<f32> {
  return emission_k(s, ch, flame_sigma(s, L), smoke_extinction(s, L), L);
}


// Condensed water (g/m^3): vapour beyond what the air can hold at the local temperature. Hot vapour
// stays clear; it clouds over as it cools, and evaporates again as it mixes into drier air.
fn condensate(s: vec4<f32>, a: vec4<f32>, L: Look) -> f32 {
  if (L.air.z < 0.5) { return 0.0; }
  return max(L.air.x + max(a.y, 0.0) - vapour_saturation(kelvin(s.x, L)), 0.0);
}

fn steam_extinction(s: vec4<f32>, a: vec4<f32>, L: Look) -> f32 {
  return condensate(s, a, L) * L.steam.w;
}

fn smoke_extinction(s: vec4<f32>, L: Look) -> f32 {
  return max(s.z, 0.0) * L.smoke.w;
}

// Henyey-Greenstein phase function.
fn hg(cos_t: f32, g: f32) -> f32 {
  let g2 = g * g;
  return (1.0 - g2) / (12.566370614 * pow(max(1.0 + g2 - 2.0 * g * cos_t, 1e-4), 1.5));
}
