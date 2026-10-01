// Weather: the physics of hydrometeors (raindrops, snowflakes, graupel, hail, ice pellets) falling through
// the air, on the GPU. The twin of engine/atmos.py: keep the two in step (tests/test_weather.py checks
// they agree). SI units, temperatures in C.
//
// A particle's state: what it is (kind: the shape it keeps while it holds ice), its ice and water (kg),
// temperature, dry size d0 (mm: a snowflake's aggregate size), whether its water can freeze (it holds
// ice or a nucleus has found it), and the most of it ever melted (fmax: a melted flake collapses, and
// stays collapsed if it refreezes, into an ice pellet).

const WX_RAIN: u32 = 0u;
const WX_SNOW: u32 = 1u;
const WX_GRAUPEL: u32 = 2u;
const WX_HAIL: u32 = 3u;
const WX_PELLET: u32 = 4u;

const WX_G: f32 = 9.80665;
const WX_RD: f32 = 287.05;
const WX_RV: f32 = 461.5;
const WX_LF: f32 = 3.336e5;
const WX_CW: f32 = 4186.0;
const WX_CI: f32 = 2106.0;
const WX_RHO_W: f32 = 1000.0;
const WX_RHO_I: f32 = 917.0;
const WX_RHO_GR: f32 = 300.0;
const WX_K: f32 = 273.15;
const WX_PI: f32 = 3.14159265;
// Bigg (1953) immersion freezing: rate per volume B (exp(-a T) - 1)
const WX_BIGG_B: f32 = 100.0;
const WX_BIGG_A: f32 = 0.66;

struct WxState {
  kind: u32,
  ice: f32,
  water: f32,
  t: f32,
  d0: f32,
  nucleated: bool,
  fmax: f32,
};

fn wx_lv(t: f32) -> f32 { return 2.501e6 - 2370.0 * t; }
fn wx_ls(t: f32) -> f32 { return 2.834e6 - 290.0 * min(t, 0.0); }
fn wx_es_water(t: f32) -> f32 { return 611.2 * exp(17.67 * t / (t + 243.5)); }
fn wx_es_ice(t: f32) -> f32 { return 611.15 * exp(22.452 * t / (t + 272.55)); }
fn wx_rho_vs(t: f32, ice: bool) -> f32 {
  return select(wx_es_water(t), wx_es_ice(t), ice) / (WX_RV * (t + WX_K));
}
fn wx_air_density(t: f32, p: f32) -> f32 { return p / (WX_RD * (t + WX_K)); }
fn wx_conductivity(t: f32) -> f32 { return 2.40e-2 + 7.73e-5 * t; }
fn wx_diffusivity(t: f32, p: f32) -> f32 { return 2.11e-5 * pow((t + WX_K) / WX_K, 1.94) * (101325.0 / p); }
fn wx_viscosity(t: f32) -> f32 {
  let tk = t + WX_K;
  return 1.458e-6 * pow(tk, 1.5) / (tk + 110.4);
}

// Saturation vapour density (kg/m^3) of the air at t: over ice below freezing (where snow keeps the air
// near ice saturation), over water above, times the relative humidity rh.
fn wx_vapour(t: f32, rh: f32) -> f32 { return rh * wx_rho_vs(t, t < 0.0); }

fn wx_mass(s: WxState) -> f32 { return s.ice + s.water; }
fn wx_melted(s: WxState) -> f32 {
  let m = wx_mass(s);
  return select(0.0, s.water / max(m, 1e-30), m > 0.0);
}
fn wx_drop_d(mass: f32) -> f32 { return pow(6.0 * max(mass, 0.0) / (WX_PI * WX_RHO_W), 1.0 / 3.0) * 1e3; }

// Size now (mm): melting snow collapses toward its drop (Mitra et al. 1990), graupel is a ball of rime,
// a stone or a frozen drop a sphere of its ice with a film of meltwater.
fn wx_diameter(s: WxState) -> f32 {
  let m = wx_mass(s);
  let dw = wx_drop_d(m);
  if (s.ice <= 0.0) { return dw; }
  let f = wx_melted(s);
  if (s.kind == WX_SNOW) { return dw + (max(s.d0, dw) - dw) * pow(1.0 - max(f, s.fmax), 0.7); }
  if (s.kind == WX_GRAUPEL) { return max(pow(6.0 * s.ice / (WX_PI * WX_RHO_GR), 1.0 / 3.0) * 1e3, dw); }
  return pow(6.0 * (s.ice / WX_RHO_I + s.water / WX_RHO_W) / WX_PI, 1.0 / 3.0) * 1e3;
}

fn wx_rain_speed(d: f32, rho: f32) -> f32 {
  var v = max(9.65 - 10.3 * exp(-0.6 * d), 0.0);
  if (d < 0.3) { v = 12.0 * pow(d, 1.2); }
  return v * pow(1.2 / rho, 0.4);
}
fn wx_snow_speed(d: f32, rho: f32) -> f32 { return 0.8 * pow(max(d, 1e-3), 0.16) * pow(1.2 / rho, 0.4); }
fn wx_graupel_speed(d: f32, rho: f32) -> f32 { return 1.3 * pow(max(d, 1e-3), 0.66) * pow(1.2 / rho, 0.4); }
fn wx_sphere_speed(d: f32, rho_p: f32, rho: f32) -> f32 { return sqrt(4.0 * rho_p * WX_G * d * 1e-3 / (3.0 * 0.6 * rho)); }

// Terminal fall speed (m/s) in air of density rho.
fn wx_speed(s: WxState, rho: f32) -> f32 {
  let d = wx_diameter(s);
  let f = wx_melted(s);
  let vr = wx_rain_speed(wx_drop_d(wx_mass(s)), rho);
  if (s.ice <= 0.0) { return vr; }
  if (s.kind == WX_SNOW) {
    let vs = wx_snow_speed(max(s.d0, d), rho);
    let fm = max(f, s.fmax);
    return vs + (vr - vs) * fm * fm;
  }
  if (s.kind == WX_GRAUPEL) {
    let vg = wx_graupel_speed(d, rho);
    return vg + (vr - vg) * f * f;
  }
  return wx_sphere_speed(d, select(WX_RHO_W, WX_RHO_I, f < 0.5), rho);
}

fn wx_capacitance(s: WxState) -> f32 {
  let d = wx_diameter(s) * 1e-3;
  let fluffy = s.kind == WX_SNOW && s.ice > 0.0 && max(wx_melted(s), s.fmax) < 0.5;
  return select(0.5 * d, 0.3 * d, fluffy);
}

fn wx_ventilation(d: f32, v: f32, t: f32, p: f32) -> f32 {
  let rho = wx_air_density(t, p);
  let nu = wx_viscosity(t) / rho;
  let re = max(v * d * 1e-3 / nu, 0.0);
  let sc = nu / wx_diffusivity(t, p);
  return 0.78 + 0.308 * pow(sc, 1.0 / 3.0) * sqrt(re);
}

// Advance a particle's heat and water by dt in air at ta (C), vapour rv (kg/m^3) and pressure pa, with an
// extra heat flow `extra` (W: from a surface it rests on). u is a uniform random number (for nucleation).
// The temperature of one phase is stepped implicitly about its linearised heat balance.
fn wx_exchange(s: ptr<function, WxState>, ta: f32, rv: f32, pa: f32, dt: f32, extra: f32, u: f32) {
  var S = *s;
  let m = wx_mass(S);
  if (m <= 0.0) { return; }
  let rho = wx_air_density(ta, pa);
  let v = wx_speed(S, rho);
  let d = wx_diameter(S);
  let fv = wx_ventilation(d, v, ta, pa);
  let C = wx_capacitance(S);
  let has_ice = S.ice > 0.0;
  let wet = S.water > 0.0;
  let mixed = has_ice && wet;
  let ts = select(S.t, 0.0, mixed);
  let over_ice = has_ice && !wet;
  let rvs = wx_rho_vs(ts, over_ice);
  let L = select(wx_lv(ts), wx_ls(ts), over_ice);
  let A = 4.0 * WX_PI * C * wx_conductivity(ta) * fv;
  let B = 4.0 * WX_PI * C * wx_diffusivity(ta, pa) * fv;
  let tk = ts + WX_K;
  let sl = rvs * (L / (WX_RV * tk * tk) - 1.0 / tk);
  let heat0 = A * (ta - ts) + L * B * (rv - rvs) + extra;
  let cm = S.ice * WX_CI + S.water * WX_CW;
  let K = A + L * B * sl;
  var t1 = ts + heat0 * dt / max(cm + K * dt, 1e-30);
  if (mixed) { t1 = 0.0; }
  let rvs1 = select(rvs + sl * (t1 - ts), rvs, mixed);
  let dm = B * (rv - rvs1) * dt;
  if (wet) { S.water = max(S.water + dm, 0.0); } else { S.ice = max(S.ice + dm, 0.0); }
  if (mixed) {
    let q = heat0 * dt;
    if (q > 0.0) {
      let melt = min(q / WX_LF, S.ice);
      S.ice -= melt;
      S.water += melt;
    } else {
      let frz = min(-q / WX_LF, S.water);
      S.water -= frz;
      S.ice += frz;
    }
  }
  if (has_ice && !wet && t1 > 0.0) {
    let melt = min(t1 * S.ice * WX_CI / WX_LF, S.ice);
    S.ice -= melt;
    S.water += melt;
    t1 = 0.0;
  }
  S.t = t1;
  // supercooled water freezes once it holds ice or a nucleus finds it (Bigg)
  if (S.water > 0.0 && S.t < 0.0) {
    if (S.ice <= 0.0 && !S.nucleated) {
      let rate = WX_BIGG_B * (S.water / WX_RHO_W) * (exp(-WX_BIGG_A * S.t) - 1.0);
      if (u < 1.0 - exp(-rate * dt)) { S.nucleated = true; }
    }
    if (S.ice > 0.0 || S.nucleated) {
      let cm2 = S.ice * WX_CI + S.water * WX_CW;
      let q2 = min(S.water, cm2 * (-S.t) / WX_LF);
      S.water -= q2;
      S.ice += q2;
      if (S.water > 0.0) { S.t = 0.0; }
    }
  }
  S.fmax = max(S.fmax, wx_melted(S));
  if (S.kind == WX_SNOW && S.fmax > 0.4 && wx_melted(S) < 0.05 && S.ice > 0.0) { S.kind = WX_PELLET; }
  if (S.ice <= 0.0 && S.water > 0.0 && S.kind != WX_RAIN) {
    S.nucleated = false;
    S.kind = WX_RAIN;
  }
  *s = S;
}
