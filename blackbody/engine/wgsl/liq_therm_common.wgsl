// Heat and phase changes of a liquid (water): shared definitions.
//
// Every particle carries its specific enthalpy h (kJ/kg) in a buffer parallel to the particles
// (therm[i].x), measured from ice at the freezing point, so one number holds both its temperature and
// how much of it is frozen, with the real latent heats between them:
//   ice             h < 0          T = Tf + h / c_ice                   all frozen
//   melting ice     0 <= h <= Lf   T = Tf                               frozen share 1 - h / Lf
//   water           h > Lf         T = Tf + (h - Lf) / c_water          liquid
//   boiling         h = hb         T = Tb (anything above boils off as vapour)
// therm[i].y says whether the particle can freeze: water with no ice in it cannot, until something
// seeds it (ice next to it, a cold enough surface, supercooling past what the water can hold). Until
// then it stays liquid below the freezing point (supercooled: h < Lf and still all water), and once
// seeded the same enthalpy turns part of it straight to ice at the freezing point, as real supercooled
// water flashes to slush. y < 0: not seeded; y >= 0: seeded, y = how cloudy its ice is (0 clear .. 1
// white: ice frozen fast traps air and grows as many small crystals).

struct Therm {
  a: vec4<f32>,   // freezing point (C), boiling point (C), supercooling water holds before it freezes by itself (K), heat speed-up
  b: vec4<f32>,   // air temperature (C), water vapour in the air (g/m^3), ground and walls temperature (C), liquid temperature (C)
  c: vec4<f32>,   // rest density (particles per cell), liquid density (kg/m^3), heat through the gas (1/0), lava (1/0)
  d: vec4<f32>,   // the gas's temperature scale: air (K), flame (K); reference temperature for buoyancy (C), bubble size (mm)
};

const LF: f32 = 333.55;          // kJ/kg, latent heat of melting
const CW: f32 = 4.18;            // kJ/(kg K), water
const CI: f32 = 2.05;            // kJ/(kg K), ice
const K_WATER: f32 = 0.6;        // W/(m K)
const K_ICE: f32 = 2.2;          // W/(m K)
const RHO_ICE: f32 = 916.7;      // kg/m^3
const RHO_VAP: f32 = 0.598;      // kg/m^3, steam at 100 C
const SIGMA_SB: f32 = 5.670e-8;  // W/(m^2 K^4)
const KELVIN: f32 = 273.15;
const FX_H: f32 = 16384.0;       // enthalpy sums (kJ/kg, 2^14)
const FX_F: f32 = 65536.0;       // frozen share sums (2^16)

fn t_freeze(th: Therm) -> f32 { return th.a.x; }
fn t_boil(th: Therm) -> f32 { return th.a.y; }

// Latent heat of boiling at T (C), kJ/kg.
fn lv_at(T: f32) -> f32 { return max(2501.0 - 2.37 * T, 1500.0); }

fn h_boil(th: Therm) -> f32 { return LF + CW * (th.a.y - th.a.x); }

fn seeded(y: f32) -> bool { return y >= 0.0; }

fn temp_of(th: Therm, h: f32, y: f32) -> f32 {
  let tf = th.a.x;
  if (!seeded(y)) { return tf + (h - LF) / CW; }
  if (h < 0.0) { return tf + h / CI; }
  if (h <= LF) { return tf; }
  return tf + (h - LF) / CW;
}

fn ice_of(h: f32, y: f32) -> f32 {
  if (!seeded(y)) { return 0.0; }
  return clamp(1.0 - h / LF, 0.0, 1.0);
}

// Enthalpy of the liquid at T (C), frozen if it is below freezing and seeded.
fn h_of_temp(th: Therm, T: f32, frozen: bool) -> f32 {
  let tf = th.a.x;
  if (frozen && T < tf) { return CI * (T - tf); }
  return LF + CW * (T - tf);
}

// A cell's temperature from its mean enthalpy and frozen share (any ice in it: it is seeded).
fn cell_temp(th: Therm, H: f32, F: f32) -> f32 {
  return temp_of(th, H, select(-1.0, 0.0, F > 1e-3));
}

// Water vapour a saturated air holds at T (C), g/m^3: over water, or over ice below freezing.
fn vapour_sat(T_in: f32, over_ice: bool) -> f32 {
  let T = clamp(T_in, -80.0, 374.0);
  var e = 6.112 * exp(17.67 * T / (T + 243.5));             // hPa (Bolton)
  if (over_ice && T < 0.0) { e = 6.112 * exp(22.46 * T / (T + 272.62)); }
  return e * 216.7 / (T + KELVIN);
}

// Density of liquid water at T (C), kg/m^3 (Tanaka et al. 2001): largest at 4 C, so near-freezing water
// floats on warmer water, and a pond cools to 4 C throughout before its top can freeze.
fn water_density(T_in: f32) -> f32 {
  let T = clamp(T_in, -30.0, 150.0);
  let a = T - 3.983035;
  return 999.97495 * (1.0 - a * a * (T + 301.797) / (522528.9 * (T + 69.34881)));
}

// Rotate v about unit axis k by angle a (Rodrigues).
fn rotate_axis(v: vec3<f32>, k: vec3<f32>, a: f32) -> vec3<f32> {
  let c = cos(a);
  let s = sin(a);
  return v * c + cross(k, v) * s + k * dot(k, v) * (1.0 - c);
}

// Ice bodies: connected frozen cells move as one rigid piece.
const NO_BODY: u32 = 0xFFFFFFFFu;
const MAX_BODIES: u32 = 8192u;
const BODY_WORDS: u32 = 36u;     // per body in the sums: 16 64-bit sums (lo, hi), flags, root cell, _
const FX_BM: f32 = 256.0;        // body sums: mass (share of a cell)
const FX_BV: f32 = 16384.0;      // velocity in the momentum
const FX_BW: f32 = 256.0;        // velocity in the angular momentum

// Gas accumulators (one buffer, regions in this order, see liquid_thermal.py): vapour into the gas (g),
// heat into the gas (J), heat out of lava (J) on the gas grid; heat back to the liquid from bubbles
// that collapsed (J) and the volume of steam bubbles (share of a cell) on the liquid grid; statistics.
const FX_G: f32 = 16777216.0;    // g (2^24: gentle evaporation gives a cell micrograms a step)
const FX_S: f32 = 4194304.0;     // g, the statistics' vapour (2^22)
const FX_E: f32 = 4096.0;        // J
const FX_R: f32 = 65536.0;       // J (heat returned)
const FX_V: f32 = 1048576.0;     // share of a cell
