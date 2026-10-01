// Heat, pass 6: what the liquid gave the gas this step, per gas cell, as rates for the fire solver
// (both_engine.py feeds it into the gas):
//   x = water vapour into the gas (g/m^3 per second): evaporation, boiling, bursting steam bubbles
//   y = change of the gas's temperature (its field units per second; negative where the liquid takes heat)
//   z = heat the water took from lava in the cell (kW per m^3 of cell)
//!include liq_therm_common.wgsl

struct Params {
  n: vec4<f32>,    // gas grid dims; w = gas cell size (m)
  k: vec4<f32>,    // step (s), the gas's temperature span (K), gas cells
};

@group(0) @binding(0) var<storage, read> gacc: array<i32>;
@group(0) @binding(1) var dst: texture_storage_3d<rgba32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(U.n.xyz);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let i = u32(c.x + n.x * (c.y + n.y * c.z));
  let nf = u32(U.k.z);
  let vol = U.n.w * U.n.w * U.n.w;
  let dt = max(U.k.x, 1e-6);
  let vap = f32(gacc[i]) / FX_G / (vol * dt);
  let heat = f32(gacc[nf + i]) / FX_E / (1.2 * 1006.0 * vol * max(U.k.y, 1.0) * dt);
  let lava = f32(gacc[2u * nf + i]) / FX_E / (vol * dt) * 1e-3;
  textureStore(dst, c, vec4<f32>(vap, heat, lava, 0.0));
}
