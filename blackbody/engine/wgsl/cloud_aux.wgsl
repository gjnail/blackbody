// Clouds: small passes (engine/cloud.py).
//   curl:    the air's spin at each cell centre (xyz) and its strength (w), for vorticity confinement
//   source:  the sun-warmed ground: heat and water vapour rise from it into the lowest layer of air, unevenly
//            (fields, roads and woods heat differently: patches the size of the thermals they set off),
//            so thermals form, rise, and where they pass their condensation level build cumulus
//   init:    the air at rest in its background state (with a little noise), and, if asked, a warm bubble
//            (the classic start of a storm in a model)
//   init_vel: the background wind
//!include cloud_common.wgsl
//!include noise.wgsl

@group(0) @binding(0) var vel: texture_3d<f32>;
@group(0) @binding(1) var A_in: texture_3d<f32>;
@group(0) @binding(2) var out4: texture_storage_3d<rgba32float, write>;
@group(0) @binding(3) var<storage, read> base: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> U: Cloud;

fn cell_vel(c: vec3<i32>) -> vec3<f32> {
  let a = textureLoad(vel, c, 0).xyz;
  let ux = textureLoad(vel, c + vec3<i32>(1, 0, 0), 0).x;
  let vy = textureLoad(vel, c + vec3<i32>(0, 1, 0), 0).y;
  let wz = textureLoad(vel, c + vec3<i32>(0, 0, 1), 0).z;
  return 0.5 * vec3<f32>(a.x + ux, a.y + vy, a.z + wz);
}

@compute @workgroup_size(8, 8, 4)
fn curl(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let h = U.g.n.w;
  let lo = max(c - vec3<i32>(1), vec3<i32>(0));
  let hi = min(c + vec3<i32>(1), n - vec3<i32>(1));
  let dx = cell_vel(vec3<i32>(hi.x, c.y, c.z)) - cell_vel(vec3<i32>(lo.x, c.y, c.z));
  let dy = cell_vel(vec3<i32>(c.x, hi.y, c.z)) - cell_vel(vec3<i32>(c.x, lo.y, c.z));
  let dz = cell_vel(vec3<i32>(c.x, c.y, hi.z)) - cell_vel(vec3<i32>(c.x, c.y, lo.z));
  let sx = f32(hi.x - lo.x) * h;
  let sy = f32(hi.y - lo.y) * h;
  let sz = f32(hi.z - lo.z) * h;
  let w = vec3<f32>(dy.z / sy - dz.y / sz, dz.x / sz - dx.z / sx, dx.y / sx - dy.x / sy);
  textureStore(out4, c, vec4<f32>(w, length(w)));
}

@compute @workgroup_size(8, 8, 4)
fn source(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  var a = textureLoad(A_in, c, 0);
  if (c.y == 0) {
    let b0 = base[0u];
    let h = U.g.n.w;
    let dt = U.g.bc.w;
    // patches of warmer and cooler ground, drifting slowly with the wind
    let wp = (vec3<f32>(c) + vec3<f32>(0.5)) * h;
    let size = max(U.heat.z, h);
    let t = U.k.x;
    let w0 = base[1u];
    let q = vec3<f32>(wp.x - w0.x * t * 0.3, t * 0.0003 * size, wp.z - w0.y * t * 0.3) / size;
    let pat = clamp(0.5 + 0.9 * gnoise(q) + 0.35 * gnoise(q * 2.7 + vec3<f32>(5.1)), 0.0, 1.5);
    let k = mix(1.0, pat * 1.6, U.heat.w);
    a.x += U.heat.x * k / (b0.z * CCP * h) * dt / b0.y;
    a.y += U.heat.y * k / (b0.z * h) * dt;
  }
  textureStore(out4, c, a);
}

@compute @workgroup_size(8, 8, 4)
fn init(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let b0 = base[u32(c.y) * 2u];
  let h = U.g.n.w;
  let p = (vec3<f32>(c) + vec3<f32>(0.5)) * h;
  let seed = u32(c.x) * 73856093u ^ u32(c.y) * 19349663u ^ u32(c.z) * 83492791u ^ u32(U.k.y);
  var th = (rand1(seed) - 0.5) * 0.1;
  // a warm bubble: heat.z = its radius (m), heat.w = how much warmer (K), at the box's middle, 1.5 km up
  // (only when U.vort.z < 0, the init's marker)
  if (U.vort.z < 0.0) {
    let ctr = vec3<f32>(0.5 * f32(n.x) * h, 1500.0, 0.5 * f32(n.z) * h);
    let r = length((p - ctr) / vec3<f32>(1.0, 0.6, 1.0)) / max(U.heat.z, 1.0);
    if (r < 1.0) { th += U.heat.w * cos(1.5707963 * r) * cos(1.5707963 * r); }
  }
  textureStore(out4, c, vec4<f32>(th, b0.w * (1.0 + (rand1(seed ^ 0x5bd1e995u) - 0.5) * 0.002), 0.0, 0.0));
}

@compute @workgroup_size(8, 8, 4)
fn init_vel(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c > n)) { return; }
  let j = clamp(c.y, 0, n.y - 1);
  let w = base[u32(j) * 2u + 1u];
  textureStore(out4, c, vec4<f32>(w.x, 0.0, w.y, 0.0));
}
