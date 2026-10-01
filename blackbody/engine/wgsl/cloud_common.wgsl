// Clouds (engine/cloud.py): what the cloud kernels share. The air is a perturbation on a background
// column at rest (the sounding, hydrostatic): per level j of the grid, base[2j] = potential temperature
// (K), Exner function, density (kg/m^3), environment vapour (kg/kg); base[2j + 1] = background wind x, z
// (m/s), temperature (K), pressure (Pa). The fields, per cell:
//   A = (potential temperature excess over the background (K), vapour, cloud water, rain)   (kg/kg)
//   B = (cloud ice, snow, graupel and hail, _)                                                 (kg/kg)
// Velocity on the MAC faces (m/s). Lengths in metres (the sky's, not the scene's).
//!include common.wgsl

struct Cloud {
  g: Grid,          // cells, cell size (m), origin (m), boundaries and dt
  k: vec4<f32>,     // time (s), step seed, sponge depth (cells), relaxation at the open sides (1/s)
  micro: vec4<f32>, // cloud water before it rains (kg/kg), autoconversion rate (1/s), hail fall factor, ice glaciation time (s)
  heat: vec4<f32>,  // surface heat flux (W/m^2), surface evaporation (kg/m^2/s), size of thermals (m), how patchy (0..1)
  vort: vec4<f32>,  // vorticity confinement, eddies (m/s), eddy size (m), condensation on (1/0)
};

const CG: f32 = 9.80665;
const CRD: f32 = 287.05;
const CRV: f32 = 461.5;
const CCP: f32 = 1004.6;
const CLV: f32 = 2.501e6;
const CLS: f32 = 2.834e6;
const CLF: f32 = 3.336e5;
const CEPS: f32 = 0.622;
const CK: f32 = 273.15;

fn c_es_w(tc: f32) -> f32 { return 611.2 * exp(17.67 * tc / (tc + 243.5)); }
fn c_es_i(tc: f32) -> f32 { return 611.15 * exp(22.452 * tc / (tc + 272.55)); }
// saturation mixing ratio (kg/kg) over water / ice at tc (C) and p (Pa)
fn c_qvs_w(tc: f32, p: f32) -> f32 { let e = min(c_es_w(tc), 0.5 * p); return CEPS * e / (p - e); }
fn c_qvs_i(tc: f32, p: f32) -> f32 { let e = min(c_es_i(tc), 0.5 * p); return CEPS * e / (p - e); }

// share of new condensate that is liquid: all of it above freezing, none below -40 C (between, more ice the
// colder; the ice then grows at the liquid's expense, Bergeron)
fn c_liquid_share(tc: f32) -> f32 {
  let f = clamp((tc + 40.0) / 40.0, 0.0, 1.0);
  return f * f;
}

// manual trilinear fetch of a cell-centred field at p (cell coordinates, centres at i + 0.5), clamped
fn c_cell(t: texture_3d<f32>, p: vec3<f32>, n: vec3<i32>) -> vec4<f32> {
  let q = clamp(p - vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(n - vec3<i32>(1)));
  let i = vec3<i32>(floor(q));
  let f = q - vec3<f32>(i);
  let hi = n - vec3<i32>(1);
  let i1 = min(i + vec3<i32>(1), hi);
  let c000 = textureLoad(t, i, 0);
  let c100 = textureLoad(t, vec3<i32>(i1.x, i.y, i.z), 0);
  let c010 = textureLoad(t, vec3<i32>(i.x, i1.y, i.z), 0);
  let c110 = textureLoad(t, vec3<i32>(i1.x, i1.y, i.z), 0);
  let c001 = textureLoad(t, vec3<i32>(i.x, i.y, i1.z), 0);
  let c101 = textureLoad(t, vec3<i32>(i1.x, i.y, i1.z), 0);
  let c011 = textureLoad(t, vec3<i32>(i.x, i1.y, i1.z), 0);
  let c111 = textureLoad(t, i1, 0);
  return mix(mix(mix(c000, c100, f.x), mix(c010, c110, f.x), f.y),
             mix(mix(c001, c101, f.x), mix(c011, c111, f.x), f.y), f.z);
}

// one component k of the MAC velocity at p (cell coordinates); component k lives on faces at integer k
fn c_face(t: texture_3d<f32>, p: vec3<f32>, k: u32, n: vec3<i32>) -> f32 {
  var o = vec3<f32>(0.5);
  o[k] = 0.0;
  var lim = n - vec3<i32>(1);
  lim[k] = n[k];
  let q = clamp(p - o, vec3<f32>(0.0), vec3<f32>(lim));
  let i = vec3<i32>(floor(q));
  let f = q - vec3<f32>(i);
  let i1 = min(i + vec3<i32>(1), lim);
  let c000 = textureLoad(t, i, 0)[k];
  let c100 = textureLoad(t, vec3<i32>(i1.x, i.y, i.z), 0)[k];
  let c010 = textureLoad(t, vec3<i32>(i.x, i1.y, i.z), 0)[k];
  let c110 = textureLoad(t, vec3<i32>(i1.x, i1.y, i.z), 0)[k];
  let c001 = textureLoad(t, vec3<i32>(i.x, i.y, i1.z), 0)[k];
  let c101 = textureLoad(t, vec3<i32>(i1.x, i.y, i1.z), 0)[k];
  let c011 = textureLoad(t, vec3<i32>(i.x, i1.y, i1.z), 0)[k];
  let c111 = textureLoad(t, i1, 0)[k];
  return mix(mix(mix(c000, c100, f.x), mix(c010, c110, f.x), f.y),
             mix(mix(c001, c101, f.x), mix(c011, c111, f.x), f.y), f.z);
}

fn c_vel(t: texture_3d<f32>, p: vec3<f32>, n: vec3<i32>) -> vec3<f32> {
  return vec3<f32>(c_face(t, p, 0u, n), c_face(t, p, 1u, n), c_face(t, p, 2u, n));
}
