// Heat, pass 4: buoyancy from heat, ice and steam, added to that between liquids of different density
// (liq_attr_buoy.wgsl), as the upward share of gravity per cell that liq_forces.wgsl applies.
//
// - Warm water is lighter than cold (and water is densest at 4 C, so meltwater and near-freezing water
//   rise through a 4 C pond): the Boussinesq share (rho_ref - rho(T)) / rho_ref, rho_ref at the liquid's
//   mean temperature, so a uniformly warm liquid feels nothing and only its differences drive it.
// - Ice (916.7 kg/m^3) in water: the pressure solve weighs it as water, so it gets the difference as
//   lift wherever there is water around it to float in (not in the air: it falls at full g). An ice
//   body then floats with the right share above the water.
// - Steam bubbles: a share a of a cell's volume that is vapour lifts the liquid around it by
//   a / (1 - a) (the bubbles rise at their own terminal speed and drag it up): the rolling boil.
//!include common.wgsl
//!include liq_common.wgsl
//!include liq_therm_common.wgsl

struct Params {
  g: Grid,
  th: Therm,
  r: vec4<f32>,     // accumulator regions: gas cells, liquid cells; z = buoyancy between liquids on (1/0)
};

@group(0) @binding(0) var therm_t: texture_3d<f32>;
@group(0) @binding(1) var other: texture_3d<f32>;     // liq_attr_buoy.wgsl's (two liquids), or zero
@group(0) @binding(2) var<storage, read> gacc: array<i32>;
@group(0) @binding(3) var buoy: texture_storage_3d<r32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn water_around(c: vec3<i32>, n: vec3<i32>) -> f32 {
  var s = 0.0;
  for (var f = 0; f < 6; f++) {
    for (var d = 1; d <= 2; d++) {
      var e = vec3<i32>(0);
      e[f / 2] = select(-d, d, (f & 1) == 1);
      let q = c + e;
      if (!in_grid(q, n)) { continue; }
      let t = textureLoad(therm_t, q, 0);
      s = max(s, t.w * (1.0 - t.z));
    }
  }
  return smoothstep(0.2, 0.5, s);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  var b = 0.0;
  if (U.r.z > 0.5) { b = textureLoad(other, c, 0).x; }
  let t = textureLoad(therm_t, c, 0);
  if (t.w > 1e-3) {
    let ref_rho = water_density(U.th.d.z);
    let F = t.z;
    b += (ref_rho - water_density(t.y)) / ref_rho * (1.0 - F);
    if (F > 0.0) { b += F * (1.0 - RHO_ICE / ref_rho) * water_around(c, n); }
    let vi = 3u * u32(U.r.x) + u32(U.r.y) + nidx(c, n);
    let a = clamp(f32(gacc[vi]) / FX_V, 0.0, 0.6);
    b += a / (1.0 - a);
  }
  textureStore(buoy, c, vec4<f32>(clamp(b, -1.0, 10.0), 0.0, 0.0, 0.0));
}
