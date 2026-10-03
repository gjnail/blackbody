// Heat, pass 3: the heat each cell of liquid gains or loses this step, from physical rates.
//
// - Inside the liquid, heat spreads by conduction (water 0.6, ice 2.2 W/(m K)) and, far faster, by
//   the churning of the flow (an eddy diffusivity from how fast neighbouring cells move past each other).
// - At the air: convection (natural, or forced by the wind and the liquid's own motion), thermal
//   radiation, and evaporation (sublimation from ice) at the rate the air's vapour deficit and the
//   same convection allow (the Lewis analogy), which carries off the latent heat. Wet air slows it,
//   dry or hot air speeds it; humid air on cold water condenses onto it instead.
// - At solids (colliders at their own temperature, the ground and closed walls at the ground's): natural
//   and forced convection, conduction through ice frozen onto them, and above the boiling point the
//   boiling curve of water: nucleate boiling (Rohsenow) up to the critical heat flux, the transition
//   regime, and film boiling past the Leidenfrost point, where a layer of vapour insulates the liquid.
// - With fire in the box, the gas next to the liquid is its air (temperature, vapour); with lava, lava
//   cells are a hot surface.
// - Steam bubbles that collapsed in cooler liquid last step give their latent heat back.
//
// What each collider gives the liquid (or takes from it) is added up for its own heat (objheat.py): a red-hot
// bar dropped in boils the water round it and cools as it does.
//
// Linear exchanges (conduction to walls and air, radiation) are capped at bringing the cell to the
// temperature they pull it toward, so no step overshoots; conduction between cells is limited to a
// stable share per step. Everything but the collapse of bubbles runs at the heat speed-up.
//
// dh: x = enthalpy change this step (kJ/kg), y = share of the cell's liquid that evaporated, z = wall
//     superheat where the wall boils the liquid (K, for the bubbles' nucleation), w = film boiling (1/0)
//!include common.wgsl
//!include liq_common.wgsl
//!include meshsdf.wgsl
//!include colliders.wgsl
//!include liq_therm_common.wgsl

struct Params {
  g: Grid,
  th: Therm,
  e: vec4<f32>,     // gas grid dims (cells), open water level (cells, < 0 off)
  w: vec4<f32>,     // wind (m/s, grid axes), gas cell size (m)
  r: vec4<f32>,     // accumulator regions: gas cells, liquid cells, _, _
  ccnt: vec4<f32>,  // collider count
  col: array<Collider, MAX_COLLIDERS>,
  ctemp: array<vec4<f32>, 4>,   // collider temperatures (C)
};

@group(0) @binding(0) var therm_t: texture_3d<f32>;
@group(0) @binding(1) var dens: texture_3d<f32>;
@group(0) @binding(2) var sdf: texture_3d<f32>;
@group(0) @binding(3) var vel: texture_3d<f32>;
@group(0) @binding(4) var atlas: texture_3d<f32>;
@group(0) @binding(5) var gas_t: texture_3d<f32>;     // fire scalars: x = temperature (0 air .. 1 flame)
@group(0) @binding(6) var gas_aux: texture_3d<f32>;   // fire aux: y = water vapour (g/m^3)
@group(0) @binding(7) var lava_t: texture_3d<f32>;    // lava (gas grid): x = temperature (K), y = share of the cell
@group(0) @binding(8) var dh: texture_storage_3d<rgba32float, write>;
@group(0) @binding(9) var<storage, read_write> gacc: array<atomic<i32>>;
@group(1) @binding(0) var<uniform> U: Params;

fn collider_temp(k: u32) -> f32 { return U.ctemp[k / 4u][k % 4u]; }

// Temperature (C) of the solid at world point w: the nearest collider's, or the ground's; and which collider (-1 the
// ground).
fn solid_temp(w: vec3<f32>) -> vec2<f32> {
  var best = U.g.n.w;
  var T = vec2<f32>(U.th.b.z, -1.0);
  let cnt = u32(U.ccnt.x);
  for (var k = 0u; k < cnt; k++) {
    let d = col_sdf(U.col[k], w);
    if (d < best) {
      best = d;
      T = vec2<f32>(collider_temp(k), f32(k));
    }
  }
  return T;
}

// The gas cell over liquid cell q (clamped to the gas grid).
fn gas_cell(q: vec3<i32>, n: vec3<i32>) -> vec3<i32> {
  let ng = vec3<i32>(U.e.xyz);
  let g = vec3<i32>(floor((vec3<f32>(q) + vec3<f32>(0.5)) * U.e.xyz / vec3<f32>(n)));
  return clamp(g, vec3<i32>(0), ng - vec3<i32>(1));
}

fn gas_index(g: vec3<i32>) -> u32 {
  let ng = vec3<i32>(U.e.xyz);
  return u32(g.x + ng.x * (g.y + ng.y * g.z));
}

fn lava_share(q: vec3<i32>, n: vec3<i32>) -> vec2<f32> {
  if (U.th.c.w < 0.5 || !in_grid(q, n)) { return vec2<f32>(0.0); }
  let l = textureLoad(lava_t, gas_cell(q, n), 0);
  return vec2<f32>(l.y, l.x - KELVIN);
}

// Boiling curve of water: heat flux (W/m^2) from a wall dTe above the boiling point, and the regime
// (0 no boiling, 1 nucleate or transition: bubbles, 2 film boiling: a vapour layer).
fn boil_flux(dTe: f32, Tw: f32, Tb: f32) -> vec2<f32> {
  if (dTe <= 3.0) { return vec2<f32>(0.0, 0.0); }
  let q_chf = 1.1e6;          // critical heat flux (W/m^2), reached about 20 K over boiling
  let q_min = 2.0e4;          // the Leidenfrost minimum
  let d_leid = 110.0;         // superheat of the Leidenfrost point (about 210 C for water on metal)
  if (dTe <= 20.0) {
    let r = dTe / 10.0;
    return vec2<f32>(min(1.37e5 * r * r * r, q_chf), 1.0);   // Rohsenow, water on metal
  }
  if (dTe < d_leid) {
    let s = (dTe - 20.0) / (d_leid - 20.0);
    return vec2<f32>(exp(mix(log(q_chf), log(q_min), s)), 1.0);
  }
  let a = Tw + KELVIN;
  let b = Tb + KELVIN;
  return vec2<f32>(250.0 * dTe + 0.9 * SIGMA_SB * (a * a * a * a - b * b * b * b), 2.0);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let th = textureLoad(therm_t, c, 0);
  let fl = th.w;
  if (fl < 1e-3) {
    textureStore(dh, c, vec4<f32>(0.0));
    return;
  }
  let h = U.g.n.w;
  let dt = U.g.bc.w;
  let speed = max(U.th.a.w, 1e-3);
  let dts = dt * speed;
  let H = th.x;
  let T = th.y;
  let F = th.z;
  let Tf = t_freeze(U.th);
  let Tb = t_boil(U.th);
  let rho = U.th.c.y;
  let m = rho * h * h * h * fl;                  // kg of liquid in the cell
  let cm = (CW * (1.0 - F) + CI * F) * 1000.0;   // J/(kg K)
  let vc = vel_centre(vel, c);
  let wp = world_of(U.g, vec3<f32>(c) + vec3<f32>(0.5));
  let ng = gas_cell(c, n);
  let gas_on = U.th.c.z > 0.5;
  let A1 = h * h;

  var G = 0.0;          // linear conductances to fixed temperatures (W/K) ...
  var GT = 0.0;         // ... and their temperatures, weighted
  var Qx = 0.0;         // heat (W) that does not depend on the cell's temperature (boiling, evaporation)
  var Kd = 0.0;         // conductance to the neighbouring liquid (W/K) ...
  var Qd = 0.0;         // ... and the heat it carries (W)
  var A_air = 0.0;       // free surface (m^2): each face to the air, by how much fuller this cell is
  var any_air = false;
  var nb_full = 0.0;     // the fullest liquid neighbour
  var air = c;
  var air_up = false;
  var superheat = 0.0;
  var film = 0.0;
  var lava_q = 0.0;     // W taken from lava
  var q_boil = 0.0;     // W of it by boiling (statistics)

  for (var f = 0; f < 6; f++) {
    var e = vec3<i32>(0);
    e[f / 2] = select(-1, 1, (f & 1) == 1);
    let q = c + e;
    var kind = 1;       // 0 liquid, 1 air, 2 solid
    var Tn = U.th.b.x;
    var Fn = 0.0;
    var fq = 0.0;
    if (!in_grid(q, n)) {
      if (e.y < 0) { kind = select(1, 2, U.g.bc.z < 0.5); }
      else if (e.y > 0) { kind = select(1, 2, U.g.bc.y < 0.5); }
      else if (U.e.w >= 0.0 && f32(c.y) + 0.5 < U.e.w) {
        kind = 0;        // the open water past the side
        Tn = U.th.b.w;
      } else { kind = select(1, 2, U.g.bc.x < 0.5); }
      if (kind == 2) { Tn = U.th.b.z; }
    } else if (textureLoad(sdf, q, 0).x < 0.0) {
      kind = 4;          // a collider: its contact is counted once, below, from this cell's own distance to it
    } else {
      let tq = textureLoad(therm_t, q, 0);
      fq = tq.w;
      nb_full = max(nb_full, tq.w);
      if (tq.w > 0.3) {
        kind = 0;
        Tn = tq.y;
        Fn = tq.z;
      }
    }
    // lava next to it is a hot surface
    let lv = lava_share(q, n);
    if (lv.x > 0.5) {
      kind = 3;
      Tn = lv.y;
    }
    if (kind == 4) { continue; }
    if (kind == 0) {
      // conduction, and the flow's churning (a mixing length of a fiftieth of a cell per unit of shear).
      // The churning is the flow's own doing, so like the flow it runs at the real speed, not the heat
      // speed-up (sped up, the slight stir of still water mixed a freezing pond to the bottom, turning it
      // to slush, where a real one keeps its warmer water under the ice)
      var dv = 0.0;
      if (in_grid(q, n)) { dv = length(vel_centre(vel, q) - vc); }
      let k = mix(K_WATER, K_ICE, 0.5 * (F + Fn));
      let turb = 0.02 * h * dv * (1.0 - max(F, Fn)) / speed;
      let K = (k + rho * CW * 1000.0 * turb) * h;
      Kd += K;
      Qd += K * (Tn - T);
    } else if (kind == 1) {
      // the surface between this cell and the emptier one lies across the face in proportion to the
      // difference (a flat surface through two cells counts once)
      A_air += A1 * clamp(min(fl, 1.0) - fq, 0.0, 1.0);
      any_air = true;
      if (e.y > 0 || !air_up) {
        air = q;
        air_up = e.y > 0;
      }
    } else {
      // a solid (or lava) face
      let dT = Tn - T;
      var q_w = 0.0;
      if (F > 0.5) {
        // ice frozen against it: conduction through half a cell of ice
        let K = 2.0 * K_ICE / h * A1;
        G += K;
        GT += K * Tn;
        q_w = K * dT / A1;
      } else {
        let hn = 220.0 * pow(abs(dT), 1.0 / 3.0) + 600.0 * length(vc);   // natural + forced convection in water
        let bf = boil_flux(Tn - Tb, Tn, Tb);
        if (bf.x > hn * max(dT, 0.0)) {
          Qx += bf.x * A1;
          q_boil += bf.x * A1;
          q_w = bf.x;
          if (bf.y > 1.5) { film = 1.0; } else { superheat = max(superheat, Tn - Tb); }
        } else {
          G += hn * A1;
          GT += hn * A1 * Tn;
          q_w = hn * dT;
        }
      }
      if (kind == 3) { lava_q += max(q_w, 0.0) * A1; }
    }
  }
  // a collider's surface: the layer of cells whose centres lie within a cell of it touches it over a face
  // each (every surface counted once, however the cells straddle it)
  let sd_c = textureLoad(sdf, c, 0).x;
  if (sd_c >= 0.0 && sd_c < 1.0) {
    let st = solid_temp(wp);
    let Tw = st.x;
    let hn = select(220.0 * pow(abs(Tw - T), 1.0 / 3.0), 2.0 * K_ICE / h, F > 0.5);
    let bf = boil_flux(Tw - Tb, Tw, Tb);
    var q_col = 0.0;     // W from the collider
    if (F <= 0.5 && bf.x > hn * max(Tw - T, 0.0)) {
      Qx += bf.x * A1;
      q_boil += bf.x * A1;
      q_col = bf.x * A1;
      if (bf.y > 1.5) { film = 1.0; } else { superheat = max(superheat, Tw - Tb); }
    } else {
      G += hn * A1;
      GT += hn * A1 * Tw;
      q_col = hn * A1 * (Tw - T);
    }
    if (st.y >= 0.0) {
      let oi = 3u * u32(U.r.x) + 2u * u32(U.r.y) + 12u + u32(st.y);
      atomicAdd(&gacc[oi], i32(round(clamp(q_col * dts * FX_E, -2.0e9, 2.0e9))));
    }
  }

  // the air: its surface, or for spray and thin sheets the surface of the drops themselves (liquid broken up
  // finer than the grid is spray of drops about 2 mm across, 3000 m^2 of surface per m^3; or at least the
  // particles' own spheres)
  let ppc = U.th.c.x;
  // (only drops in the air: a thin patch inside churning water, or the thin edge over a surface, is not spray)
  let spray = select(0.0, 1.0 - smoothstep(0.3, 0.6, nb_full), any_air);
  let A_drops = max(fl * h * 3000.0, fl * 4.84 * pow(ppc, 1.0 / 3.0)) * A1 * (1.0 - smoothstep(0.08, 0.3, fl)) * spray;
  let A = A_air + A_drops;
  var evap_kg = 0.0;
  if (A > 0.0) {
    var Ta = U.th.b.x;
    var rv = U.th.b.y;
    let gq = gas_cell(air, n);
    if (gas_on) {
      let s = textureLoad(gas_t, gq, 0).x;
      Ta = U.th.d.x + s * (U.th.d.y - U.th.d.x) - KELVIN;
      rv = U.th.b.y + max(textureLoad(gas_aux, gq, 0).y, 0.0);
    }
    let vrel = length(vc - U.w.xyz);
    let hc = max(1.52 * pow(abs(T - Ta), 1.0 / 3.0), 5.7 + 3.8 * vrel);
    let TsK = T + KELVIN;
    let TaK = Ta + KELVIN;
    let eps = mix(0.95, 0.5, smoothstep(400.0, 900.0, TaK));
    let hr = eps * SIGMA_SB * (TsK * TsK + TaK * TaK) * (TsK + TaK);
    G += (hc + hr) * A;
    GT += (hc + hr) * A * Ta;
    // evaporation (sublimation from ice), or condensation from air wetter than the surface
    let on_ice = F > 0.5;
    let rs = vapour_sat(min(T, Tb), on_ice);
    let mflux = hc / 1206.0 * (rs - rv) * 1e-3;                // kg/(m^2 s), Lewis analogy
    let L = (lv_at(T) + select(0.0, LF, on_ice)) * 1000.0;
    Qx -= mflux * A * L;
    evap_kg = max(mflux, 0.0) * A * dts;
    // what the gas gets: the vapour, and the heat convection carries into it
    let gi = gas_index(gq);
    if (evap_kg > 0.0) { atomicAdd(&gacc[gi], i32(round(min(evap_kg * 1000.0 * FX_G, 2.0e9)))); }
    let e_gas = hc * A * (T - Ta) * dts;
    atomicAdd(&gacc[u32(U.r.x) + gi], i32(round(clamp(e_gas * FX_E, -2.0e9, 2.0e9))));
  }
  if (lava_q > 0.0) {
    atomicAdd(&gacc[2u * u32(U.r.x) + gas_index(ng)], i32(round(min(lava_q * dts * FX_E, 2.0e9))));
  }

  // conduction between cells, limited to a stable share per step
  let beta = Kd * dts / max(m * cm, 1e-12);
  let lim = select(1.0, 0.4 / beta, beta > 0.4);
  var dH = (Qx + Qd * lim) * dts / (m * 1000.0);
  // the linear exchanges, capped at reaching the temperature they pull toward
  if (G > 0.0) {
    let Tt = GT / G;
    let seeded_cell = F > 1e-3;
    let Heq = h_of_temp(U.th, Tt, seeded_cell);
    let dq = G * (Tt - T) * dts / (m * 1000.0);
    let d_eq = Heq - H;
    dH += clamp(dq, min(0.0, d_eq), max(0.0, d_eq));
  }
  // steam bubbles that condensed in this cell last step
  let ri = 3u * u32(U.r.x) + nidx(c, n);
  let ret = f32(atomicLoad(&gacc[ri])) / FX_R;
  dH += ret / (m * 1000.0);

  let e_share = clamp(evap_kg / m, 0.0, 1.0);
  textureStore(dh, c, vec4<f32>(dH, e_share, superheat, film));

  // statistics (liquid_thermal.py Thermal.stats)
  let st = 3u * u32(U.r.x) + 2u * u32(U.r.y);
  atomicAdd(&gacc[st], 1);
  if (F > 0.5) { atomicAdd(&gacc[st + 1u], 1); }
  atomicAdd(&gacc[st + 2u], i32(round(clamp(T, -900.0, 2000.0) * 2.0)));
  if (superheat > 0.0 || film > 0.0 || T >= Tb - 0.5) { atomicAdd(&gacc[st + 4u], 1); }
  atomicMax(&gacc[st + 5u], i32((clamp(T, -900.0, 2000.0) + 1000.0) * 100.0));
  atomicMax(&gacc[st + 6u], i32((1000.0 - clamp(T, -900.0, 2000.0)) * 100.0));
  if (evap_kg > 0.0) { atomicAdd(&gacc[st + 7u], i32(round(min(evap_kg * 1000.0 * FX_S, 2.0e9)))); }
  if (q_boil > 0.0) { atomicAdd(&gacc[st + 10u], i32(round(min(q_boil * dts * 256.0, 2.0e9)))); }
}
