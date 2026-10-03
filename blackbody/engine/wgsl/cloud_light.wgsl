// Clouds: how much sunlight and skylight reaches each cell (engine/cloud_render.py), one thread per cell,
// through the cloud as it is drawn (cloud_sig.wgsl): the optical depth between the cell and the sun (x),
// and how much of the open sky the cell sees through the cloud round it (y: up, and slanting out to the
// sides, the light scattered many times inside the cloud getting out more easily than the direct beam);
// and the cell's own extinction (1/m) of its cloud (z) and of its precipitation (w), from cloud_sig. The
// renderer turns them into the light each part of the cloud gets: a thick cloud's far side and its base
// are in its own shadow, its sides lit by the sky round them.
//!include common.wgsl

@group(0) @binding(0) var S: texture_3d<f32>;
@group(0) @binding(1) var out_t: texture_storage_3d<rgba32float, write>;
@group(1) @binding(0) var<uniform> U: LightParams;

struct LightParams {
  g: Grid,
  sun: vec4<f32>,   // toward the sun (unit, the simulation's frame)
  ext: vec4<f32>,   // (unused here)
  ext2: vec4<f32>,
};

// extinction (1/m) of the cloud and precipitation in a cell, as drawn
// (with Lume, the whole cloud as the march draws it, its ragged edge too, as the light is traced through it; else its solid
// part and the precipitation: cloud_sig.wgsl)
fn sigma(c: vec3<i32>) -> f32 {
  let s = textureLoad(S, c, 0);
  return select(s.z, min(s.x, 0.05) + s.y, U.ext2.z > 0.5);
}

fn sigma_at(p: vec3<f32>, n: vec3<i32>) -> f32 {
  let c = vec3<i32>(floor(p));
  if (any(c < vec3<i32>(0)) || any(c >= n)) { return 0.0; }
  return sigma(c);
}

// the extinction at p (cells), interpolated between the cell centres
fn sigma_lin(p: vec3<f32>, n: vec3<i32>) -> f32 {
  let q = clamp(p - vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(n - vec3<i32>(1)));
  let i = vec3<i32>(floor(q));
  let f = q - vec3<f32>(i);
  let j = min(i + vec3<i32>(1), n - vec3<i32>(1));
  let x00 = mix(sigma(i), sigma(vec3<i32>(j.x, i.y, i.z)), f.x);
  let x10 = mix(sigma(vec3<i32>(i.x, j.y, i.z)), sigma(vec3<i32>(j.x, j.y, i.z)), f.x);
  let x01 = mix(sigma(vec3<i32>(i.x, i.y, j.z)), sigma(vec3<i32>(j.x, i.y, j.z)), f.x);
  let x11 = mix(sigma(vec3<i32>(i.x, j.y, j.z)), sigma(j), f.x);
  return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}

// optical depth from the cell centre p0 toward the sun (unit d, cells), interpolated (so neighbouring
// cells' paths, through the same cloud, agree: no stair-stepped shadows)
fn depth_sun(p0: vec3<f32>, d: vec3<f32>, n: vec3<i32>, h: f32) -> f32 {
  var tau = 0.0;
  var p = p0 + 0.5 * d;
  for (var i = 0; i < 1024; i++) {
    if (any(p < vec3<f32>(0.0)) || any(p >= vec3<f32>(n))) { break; }
    tau += sigma_lin(p, n) * 0.5 * h;
    p += 0.5 * d;
  }
  return tau;
}

// optical depth from the cell centre p0 along d (cells) until it leaves the box
fn depth_along(p0: vec3<f32>, d: vec3<f32>, n: vec3<i32>, h: f32) -> f32 {
  var tau = 0.0;
  var p = p0;
  let step = length(d) * h;
  for (var i = 0; i < 512; i++) {
    p += d;
    if (any(p < vec3<f32>(0.0)) || any(p >= vec3<f32>(n))) { break; }
    tau += sigma_at(p, n) * step;
  }
  return tau;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let h = U.g.n.w;
  let p0 = vec3<f32>(c) + vec3<f32>(0.5);
  let own = 0.5 * sigma(c) * h;
  // toward the sun, a cell at a time until it leaves the box
  let tau_sun = 0.25 * sigma(c) * h + depth_sun(p0, U.sun.xyz, n, h);
  // the sky seen: straight up, and out at 45 degrees to the four sides (the sky's light comes mostly from
  // above); through the cloud at the reduced extinction of light scattered many times
  let k = 0.35;
  let s = 0.7071;
  var vis = 0.4 * exp(-k * (own + depth_along(p0, vec3<f32>(0.0, 1.0, 0.0), n, h)));
  vis += 0.15 * exp(-k * (own + depth_along(p0, vec3<f32>(s, s, 0.0), n, h)));
  vis += 0.15 * exp(-k * (own + depth_along(p0, vec3<f32>(-s, s, 0.0), n, h)));
  vis += 0.15 * exp(-k * (own + depth_along(p0, vec3<f32>(0.0, s, s), n, h)));
  vis += 0.15 * exp(-k * (own + depth_along(p0, vec3<f32>(0.0, s, -s), n, h)));
  let own_s = textureLoad(S, c, 0);
  textureStore(out_t, c, vec4<f32>(tau_sun, vis, own_s.x, own_s.y));
}
