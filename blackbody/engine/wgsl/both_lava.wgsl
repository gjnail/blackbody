// Fire, water and lava in one box: the lava, as the air and the water meet it. Per fire cell (the
// three grids share their layout), before the fire steps:
//
// - The air over hot lava heats up: a boundary layer at a good part of the lava surface's
//   temperature, which rises off the flow as a shimmering updraft and sets burnable things alight.
// - Water touching lava hotter than boiling boils. Lava shores and lava under a shallow sea boil at
//   around a megawatt per square metre of contact (film boiling), about 0.4 kg of steam per square
//   metre per second: a thick white plume. The steam goes into the gas at the first open cell above
//   the contact (neither full of water nor of lava); on its way up through water, part of it
//   condenses back into the water, so lava deep under the sea makes little steam at the surface.
//
// Writes the fire's scalars (temperature raised near the lava) and the lava field the other
// kernels read: x = lava temperature (K, 0 where there is none), y = lava share of the cell,
// z = steam into the gas here (g/m^3 per second), w = water share of the cell and around it.
//!include common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,    // rest density (particles per cell), temperature of fresh lava (K), air temperature (K), flame temperature (K)
  k2: vec4<f32>,   // air heating rate (1/s), share of the lava surface's temperature the air over it reaches, boiling (x physical), steam condensing per cell of water above
  k3: vec4<f32>,   // the water's rest density (particles per cell)
};

@group(0) @binding(0) var lava_dens: texture_3d<f32>;
@group(0) @binding(1) var lava_heat: texture_3d<f32>;
@group(0) @binding(2) var water_dens: texture_3d<f32>;
@group(0) @binding(3) var scal_in: texture_3d<f32>;
@group(0) @binding(4) var scal_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var field: texture_storage_3d<rgba32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn lava_share(c: vec3<i32>, n: vec3<i32>) -> f32 {
  if (!in_grid(c, n)) { return 0.0; }
  return clamp(textureLoad(lava_dens, c, 0).x / max(U.k.x, 1e-3), 0.0, 1.0);
}

fn water_share(c: vec3<i32>, n: vec3<i32>) -> f32 {
  if (!in_grid(c, n)) { return 0.0; }
  return clamp(textureLoad(water_dens, c, 0).x / max(U.k3.x, 1e-3), 0.0, 1.0);
}

// Heat of the lava in cell c (0..1: 1 as poured), 0 where there is none.
fn lava_heat_at(c: vec3<i32>, n: vec3<i32>) -> f32 {
  if (lava_share(c, n) < 0.02) { return 0.0; }
  return clamp(textureLoad(lava_heat, c, 0).x, 0.0, 1.0);
}

// Lava temperature (K) for a heat.
fn kelvin_of_heat(heat: f32) -> f32 { return U.k.z + (U.k.y - U.k.z) * heat; }

// Share of the lava in c that is past boiling, from its temperature.
fn over_boiling(heat: f32) -> f32 {
  return clamp((kelvin_of_heat(heat) - 373.15) / max(U.k.y - 373.15, 1.0), 0.0, 1.0);
}

// Steam (g/m^3/s) boiled where the lava in c touches water (in c or next to it).
fn boil_at(c: vec3<i32>, n: vec3<i32>) -> f32 {
  let fl = lava_share(c, n);
  if (fl < 0.02) { return 0.0; }
  var wf = water_share(c, n);
  for (var i = 0; i < 6; i++) {
    var d = vec3<i32>(0);
    d[i >> 1] = select(-1, 1, (i & 1) == 1);
    wf = max(wf, water_share(c + d, n));
  }
  if (wf <= 0.0) { return 0.0; }
  // film boiling, about 1 MW per m^2 of contact at lava temperature: 1e6 / 2.6e6 J/kg over a cell
  let per_cell = 1.0e6 / 2.6e6 * 1000.0 / U.g.n.w;
  return U.k2.z * per_cell * over_boiling(lava_heat_at(c, n)) * min(fl * 2.0, 1.0) * wf;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let dt = U.g.bc.w;
  var s = textureLoad(scal_in, c, 0);
  let fl = lava_share(c, n);
  let wf = water_share(c, n);

  // the hottest lava surface at or next to this cell
  var near = fl * lava_heat_at(c, n);
  var near_share = fl;
  var wn = wf;
  for (var i = 0; i < 6; i++) {
    var d = vec3<i32>(0);
    d[i >> 1] = select(-1, 1, (i & 1) == 1);
    let q = c + d;
    let f = lava_share(q, n);
    near = max(near, f * lava_heat_at(q, n));
    near_share = max(near_share, f);
    wn = max(wn, water_share(q, n));
  }

  // the air over hot lava heats toward a good part of the surface's temperature
  if (near > 0.0 && wf < 0.5) {
    let span = max(U.k.w - U.k.z, 1.0);
    let target_t = U.k2.y * (kelvin_of_heat(near) - U.k.z) / span;
    if (s.x < target_t) {
      s.x += (target_t - s.x) * (1.0 - exp(-U.k2.x * near_share * dt));
    }
  }

  // steam from lava boiling water: into the gas at the first open cell over the contact
  var steam = 0.0;
  let is_open = wf + fl < 0.6;
  if (is_open) {
    steam = boil_at(c, n);
    var under_water = 0.0;
    for (var j = 1; j <= 6; j++) {
      let q = c - vec3<i32>(0, j, 0);
      if (!in_grid(q, n)) { break; }
      let wq = water_share(q, n);
      let fq = lava_share(q, n);
      if (wq + fq < 0.6) { break; }   // an open cell under this one takes its own
      steam += boil_at(q, n) * exp(-U.k2.w * under_water);
      under_water += select(0.0, 1.0, wq > 0.5);
    }
  }

  textureStore(scal_out, c, s);
  textureStore(field, c, vec4<f32>(select(0.0, kelvin_of_heat(lava_heat_at(c, n)), fl >= 0.02), fl, steam, wn));
}
