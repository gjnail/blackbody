// Clouds: precipitation falling through the air (engine/cloud.py), each species at its own mass-weighted
// fall speed relative to the air (rain 4 to 9 m/s, snow about 1, graupel and hail 3 to 15, cloud ice a
// fraction of a metre a second), all faster in the thin air aloft. A vertical semi-Lagrangian shift: each
// cell takes what was above it; what leaves the bottom is the precipitation reaching the ground, kept per
// column as a rate (kg/m^2/s) of rain, snow and graupel/hail for the renderer.
//!include cloud_common.wgsl

@group(0) @binding(0) var A_in: texture_3d<f32>;
@group(0) @binding(1) var B_in: texture_3d<f32>;
@group(0) @binding(2) var A_out: texture_storage_3d<rgba32float, write>;
@group(0) @binding(3) var B_out: texture_storage_3d<rgba32float, write>;
@group(0) @binding(4) var<storage, read> base: array<vec4<f32>>;
@group(0) @binding(5) var precip: texture_storage_2d<rgba32float, write>;
@group(1) @binding(0) var<uniform> U: Cloud;

fn rho_at(j: i32) -> f32 { return base[u32(j) * 2u].z; }

// mass-weighted fall speeds (m/s)
fn v_rain(q: f32, rho: f32) -> f32 { return 5.0 * pow(max(rho * q / 1.0e-3, 1.0e-6), 0.1346) * sqrt(1.2 / rho); }
fn v_snow(q: f32, rho: f32) -> f32 { return 1.0 * pow(max(rho * q / 1.0e-3, 1.0e-6), 0.06) * sqrt(1.2 / rho); }
fn v_graupel(q: f32, rho: f32) -> f32 {
  return 4.0 * U.micro.z * pow(max(rho * q / 1.0e-3, 1.0e-6), 0.2) * sqrt(1.2 / rho);
}
fn v_ice(rho: f32) -> f32 { return 0.3 * sqrt(1.2 / rho); }

// linear fetch of a field in the column at height y (cells)
fn col_at(t: texture_3d<f32>, x: i32, z: i32, y: f32, n: vec3<i32>) -> vec4<f32> {
  let q = y - 0.5;
  if (q >= f32(n.y - 1)) { return vec4<f32>(0.0); }   // nothing falls in through the top
  let j = i32(floor(max(q, 0.0)));
  let f = clamp(q - f32(j), 0.0, 1.0);
  let a = textureLoad(t, vec3<i32>(x, j, z), 0);
  let b = textureLoad(t, vec3<i32>(x, min(j + 1, n.y - 1), z), 0);
  return mix(a, select(b, vec4<f32>(0.0), j + 1 > n.y - 1), f);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let dt = U.g.bc.w;
  let h = U.g.n.w;
  let a = textureLoad(A_in, c, 0);
  let b = textureLoad(B_in, c, 0);
  let rho = rho_at(c.y);
  let y = f32(c.y) + 0.5;
  // each species from as far above as it falls in a step (its speed where it is)
  let vr = v_rain(a.w, rho);
  let vs = v_snow(b.y, rho);
  let vg = v_graupel(b.z, rho);
  let vi = v_ice(rho);
  let ar = col_at(A_in, c.x, c.z, y + vr * dt / h, n);
  let bs = col_at(B_in, c.x, c.z, y + vs * dt / h, n);
  let bg = col_at(B_in, c.x, c.z, y + vg * dt / h, n);
  let bi = col_at(B_in, c.x, c.z, y + vi * dt / h, n);
  // (the air above is thinner: the same mass per volume is less per kg of the denser air below)
  let ja = min(i32(floor(y + vr * dt / h)), n.y - 1);
  let rr = rho_at(ja) / rho;
  let rs = rho_at(min(i32(floor(y + vs * dt / h)), n.y - 1)) / rho;
  let rg = rho_at(min(i32(floor(y + vg * dt / h)), n.y - 1)) / rho;
  textureStore(A_out, c, vec4<f32>(a.x, a.y, a.z, ar.w * rr));
  textureStore(B_out, c, vec4<f32>(bi.x * rho_at(min(i32(floor(y + vi * dt / h)), n.y - 1)) / rho, bs.y * rs, bg.z * rg, b.w));
  if (c.y == 0) {
    // through the ground: the precipitation rate there
    textureStore(precip, vec2<i32>(c.x, c.z), vec4<f32>(rho * a.w * vr, rho * b.y * vs, rho * b.z * vg, 0.0));
  }
}
