// Sources, combustion and dissipation at cell centres.
// scal: x = temperature (0 = ambient, about 1 = flame), y = fuel, z = soot/smoke, w = flame.
// aux (optional): x = share of the air's oxygen used up (0 = fresh air), y = water vapour above the
//   air's own (g/m^3), z = condensed water (g/m^3, for latent heat). chem (optional): rgb = flame colourant.
// burn (optional): the burnable floor; burn_obj + slots: burnable colliders (see burn_common.wgsl).
// water (optional): share of each cell filled with liquid water (from a liquid simulation in the same box).
//
// The same kernel runs on the finer upres grid (lo.x = how many times finer): there it reads the
// optional fields from the simulation grid and writes only the scalars.
//!include common.wgsl
//!include noise.wgsl
//!include meshsdf.wgsl
//!include emitters.wgsl
//!include colliders.wgsl
//!include burn_common.wgsl
//!include burnobj.wgsl
//!include water.wgsl

struct Params {
  g: Grid,
  comb: vec4<f32>,   // ignition temperature, burn rate (1/s), heat release, soot yield
  comb2: vec4<f32>,  // flame gain, flame lifetime (s), gas expansion, radiative cooling
  decay: vec4<f32>,  // cooling (1/s), smoke dissipation (1/s), fuel dissipation (1/s), temperature cap
  comb3: vec4<f32>,  // rich limit (fuel level where lack of mixing halves burning), expansion cap (1/s), air use, air tracked (1/0)
  wet: vec4<f32>,    // vapour on (1/0), steam per unit of heat removed, water per unit of fuel burned, vapour mixing (1/s)
  surf: vec4<f32>,   // burning surfaces: fuel (1/s), heat, smoke (1/s), smoulder smoke (1/s)
  feat: vec4<f32>,   // colourant on, surfaces on, air mixing (turbulent diffusivity, m^2/s), boiling point of water (field temperature)
  wat: vec4<f32>,    // water on (1/0), how fast a cell full of water puts fire out (1/s), _, _
  therm: vec4<f32>,  // heat expansion (share of physical), air temperature (K), flame minus air temperature (K), latent heat (field temperature per g/m^3)
  lo: vec4<f32>,     // x = cells of this grid per simulation cell (1, or the upres factor), y = air's own vapour (g/m^3)
  cnt: vec4<f32>,    // emitter count, sponge width (cells), sponge strength (1/s), _
  em: array<Emitter, MAX_EMITTERS>,
  ccnt: vec4<f32>,   // collider count
  col: array<Collider, MAX_COLLIDERS>,
};

@group(0) @binding(0) var src: texture_3d<f32>;
@group(0) @binding(1) var sdf: texture_3d<f32>;
@group(0) @binding(2) var aux_src: texture_3d<f32>;
@group(0) @binding(3) var chem_src: texture_3d<f32>;
@group(0) @binding(4) var burn: texture_3d<f32>;
@group(0) @binding(5) var atlas: texture_3d<f32>;
@group(0) @binding(6) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(7) var expo: texture_storage_3d<r32float, write>;
@group(0) @binding(8) var aux_dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(9) var chem_dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(10) var water: texture_3d<f32>;
@group(0) @binding(11) var burn_obj: texture_3d<f32>;
@group(0) @binding(12) var<storage, read> slots: array<BurnSlot>;
@group(1) @binding(0) var<uniform> U: Params;

fn kelvin_of(T: f32) -> f32 { return U.therm.y + U.therm.z * max(T, 0.0); }

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = U.g.n.xyz;
  let c = vec3<i32>(id);
  if (any(c >= vec3<i32>(n))) { return; }
  let h = U.g.n.w;
  let dt = U.g.bc.w;
  let time = U.g.org.w;
  let fine = i32(U.lo.x);
  let main_grid = fine <= 1;
  let cl = c / max(fine, 1);  // the simulation cell this cell lies in
  let s = textureLoad(src, c, 0);
  var T = s.x;
  var F = s.y;
  var S = s.z;
  var Fl = s.w;
  let T0 = T;
  let tracked = U.comb3.w > 0.5;
  let wet_on = U.wet.x > 0.5;
  let chem_on = U.feat.x > 0.5;
  var ax = vec4<f32>(0.0);
  if (tracked || wet_on) { ax = textureLoad(aux_src, cl, 0); }
  if (main_grid && tracked && U.feat.z > 0.0) {
    // turbulent mixing finer than the grid, as a diffusion of used-up air (walls and box sides
    // excepted, so a sealed room keeps its total): starved, fuel-rich gas that meets fresh air burns
    var sum = 0.0;
    var k = 0.0;
    for (var i = 0; i < 6; i++) {
      let o = select(-1, 1, (i & 1) == 1);
      let a = i >> 1;
      let q = c + vec3<i32>(select(0, o, a == 0), select(0, o, a == 1), select(0, o, a == 2));
      if (!in_grid(q, vec3<i32>(n))) { continue; }
      if (textureLoad(sdf, q, 0).x < 0.0) { continue; }
      sum += textureLoad(aux_src, q, 0).x;
      k += 1.0;
    }
    if (k > 0.0) { ax.x = mix(ax.x, sum / k, 1.0 - exp(-k * U.feat.z * dt / (h * h))); }
  }
  var ch = vec4<f32>(0.0);
  if (chem_on && main_grid) { ch = textureLoad(chem_src, c, 0); }
  let wp = world_of(U.g, vec3<f32>(c) + 0.5);

  var removed = 0.0;  // heat taken out by extinguishing emitters and water
  let cnt = i32(U.cnt.x);
  for (var i = 0; i < cnt; i++) {
    let e = U.em[i];
    let w = emitter_weight(e, wp, h, time);
    if (w > 0.0) {
      F += e.d.x * w * dt;
      T = max(T, e.d.y * w);
      S += e.d.z * w * dt;
      ch = vec4<f32>(ch.rgb + e.col.rgb * (w * dt), ch.w);
      ax.y = max(ax.y, e.col.w * w);
    }
    if (e.k.w > 0.0) {
      // water or extinguishing agent: smothers fuel and flame, and cools the gas toward the boiling
      // point of water (the heat goes into turning the water to steam, which stays hot and rises)
      let m = emitter_mask(e, wp, h);
      if (m > 0.0) {
        let q = exp(-e.k.w * m * dt);
        let boil = U.feat.w;
        if (T > boil) {
          removed += (T - boil) * (1.0 - q);
          T = boil + (T - boil) * q;
        }
        F *= q;
        Fl *= q;
      }
    }
  }

  // liquid water in the cell works like a hose: it smothers fuel and flame and boils away the heat
  if (U.wat.x > 0.5) {
    let wf = clamp(textureLoad(water, cl, 0).x, 0.0, 1.0);
    if (wf > 0.0) {
      let q = exp(-U.wat.y * wf * dt);
      let boil = U.feat.w;
      if (T > boil) {
        removed += (T - boil) * (1.0 - q);
        T = boil + (T - boil) * q;
      }
      F *= q;
      Fl *= q;
    }
  }

  // burning and smouldering surfaces release fuel, heat and smoke into the air next to them:
  // the floor, and burnable colliders (looked up in their own frame, so they can move)
  if (U.feat.y > 0.5) {
    var em = burn_emission(textureLoad(burn, cl, 0), U.surf);
    let nc = i32(U.ccnt.x);
    for (var i = 0; i < nc; i++) {
      let k = U.col[i];
      let slot = i32(k.m2.w);
      if (slot < 0) { continue; }
      let e2 = burn_emission(obj_burn_at(k, slots[slot], wp), U.surf);
      em = vec3<f32>(em.x + e2.x, max(em.y, e2.y), em.z + e2.z);
    }
    F += em.x * dt;
    T = max(T, em.y);
    S += em.z * dt;
  }

  // combustion: fuel above the ignition temperature burns into heat, soot and flame
  let ign = U.comb.x;
  let lit = smoothstep(ign * 0.7, ign * 1.3 + 1e-4, T);
  // mixing limit: a fuel-rich core has no air mixed into it, so it burns at its surface as air mixes in
  let rich = max(U.comb3.x, 1e-3);
  let oxy = 1.0 / (1.0 + (F / rich) * (F / rich));
  var burned = F * (1.0 - exp(-U.comb.y * dt)) * lit * oxy;
  if (tracked) {
    // tracked air: flames die once most of the oxygen is used (real flames go out below ~13% O2),
    // and burning can never use more oxygen than is left
    let avail = clamp(1.0 - ax.x, 0.0, 1.0);
    let per_fuel = max(U.comb3.z, 1e-4);
    burned = min(burned * smoothstep(0.2, 0.5, avail), avail / per_fuel);
    ax.x = min(ax.x + burned * per_fuel, 1.0);
  }
  F -= burned;
  T += burned * U.comb.z;
  S += burned * U.comb.w;
  Fl = Fl * exp(-dt / max(U.comb2.y, 1e-3)) + burned * U.comb2.x;
  ax.y += burned * U.wet.z + removed * U.wet.y;

  // cooling: exponential mixing loss plus radiative loss (exact step of dT/dt = -k T^4)
  T = T * exp(-U.decay.x * dt);
  T = T / pow(1.0 + 3.0 * U.comb2.w * dt * T * T * T, 1.0 / 3.0);
  S = S * exp(-U.decay.y * dt);
  F = F * exp(-U.decay.z * dt);
  T = min(T, U.decay.w);
  ch = ch * exp(-U.decay.y * dt);
  ax.y = ax.y * exp(-U.wet.w * dt);

  // latent heat: vapour condensing into droplets warms the gas (so steam billows up like a cloud),
  // droplets evaporating cool it. aux.z holds the condensed water, whose heat is already in T. The
  // balance is stiff (a few g/m^3 of droplets warm the air by several Kelvin), so the condensed
  // amount is solved for with Newton's method: c = total water - saturation(temperature with c's heat).
  if (wet_on && main_grid && U.therm.w > 0.0) {
    let lf = U.therm.w;               // field temperature per g/m^3 condensed
    let lk = lf * U.therm.z;          // Kelvin per g/m^3 condensed
    let q = U.lo.y + ax.y;            // all the water: the air's own plus what was added
    let t_dry = max(T - ax.z * lf, 0.0);
    var cond = 0.0;
    if (q > vapour_saturation(kelvin_of(t_dry))) {
      cond = max(ax.z, 0.0);
      for (var it = 0; it < 5; it++) {
        let tk = kelvin_of(t_dry + lf * cond);
        let sat = vapour_saturation(tk);
        let slope = vapour_saturation(tk + 0.5) - vapour_saturation(tk - 0.5);
        cond = max(cond - (cond - (q - sat)) / (1.0 + slope * lk), 0.0);
      }
    }
    T = t_dry + lf * cond;
    ax.z = cond;
  }

  // heat expansion: gas swells as it heats and shrinks as it cools (ideal gas at constant pressure:
  // div u = (1 / T) dT/dt, in Kelvin); measured before the sponge, which is not a real cooling
  var ex = burned / max(dt, 1e-6) * U.comb2.z;
  if (U.therm.x > 0.0) {
    ex += U.therm.x * (T - T0) * U.therm.z / (max(kelvin_of(0.5 * (T + T0)), 1.0) * max(dt, 1e-6));
  }
  ex = clamp(ex, -U.comb3.y, U.comb3.y);

  // sponge layer: gas fades out as it nears an open boundary, so the domain edge never shows;
  // there the used-up air is replaced by fresh air
  let sw = U.cnt.y * f32(max(fine, 1));
  if (sw > 0.0) {
    var dist = 1.0e6;
    if (U.g.bc.x > 0.5) { dist = min(min(f32(c.x), n.x - 1.0 - f32(c.x)), min(f32(c.z), n.z - 1.0 - f32(c.z))); }
    if (U.g.bc.y > 0.5) { dist = min(dist, n.y - 1.0 - f32(c.y)); }
    if (U.g.bc.z > 0.5) { dist = min(dist, f32(c.y)); }
    if (dist < sw) {
      let k = 1.0 - dist / sw;
      let fade = exp(-U.cnt.z * k * k * dt);
      S *= fade; T *= fade; Fl *= fade; F *= fade;
      ax *= fade;
      ch *= fade;
    }
  }

  if (textureLoad(sdf, c, 0).x < 0.0) {
    T = 0.0; F = 0.0; S = 0.0; Fl = 0.0; ex = 0.0;
    ax = vec4<f32>(0.0);
    ch = vec4<f32>(0.0);
  }
  textureStore(dst, c, vec4<f32>(max(T, 0.0), max(F, 0.0), max(S, 0.0), max(Fl, 0.0)));
  if (main_grid) {
    textureStore(expo, c, vec4<f32>(ex, 0.0, 0.0, 0.0));
    if (tracked || wet_on) { textureStore(aux_dst, c, max(ax, vec4<f32>(0.0))); }
    if (chem_on) { textureStore(chem_dst, c, max(ch, vec4<f32>(0.0))); }
  }
}
