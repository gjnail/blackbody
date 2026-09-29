// Cell-centred body forces: vorticity confinement, curl-noise turbulence and disturbance.
// All three are masked to hot or burning gas, where real flames get their small-scale motion.
//!include common.wgsl
//!include noise.wgsl

struct Params {
  g: Grid,
  f: vec4<f32>,   // confinement strength, turbulence (m/s^2), disturbance (m/s^2), confinement outside the mask
  t: vec4<f32>,   // turbulence frequency (1/m), turbulence rise (m/s), turbulence evolution (1/s), disturbance block (m)
  m: vec4<f32>,   // mask weights: temperature, flame, smoke; disturbance rate (Hz)
  t2: vec4<f32>,  // second-octave gain, seed, _, _
};

@group(0) @binding(0) var curl: texture_3d<f32>;
@group(0) @binding(1) var scal: texture_3d<f32>;
@group(0) @binding(2) var dst: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn cw(c: vec3<i32>, d: vec3<i32>) -> f32 {
  return textureLoad(curl, clamp(c, vec3<i32>(0), d - vec3<i32>(1)), 0).w;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let d = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= d)) { return; }
  let h = U.g.n.w;
  let time = U.g.org.w;
  let s = textureLoad(scal, c, 0);
  let mask = clamp(s.x * U.m.x + s.w * U.m.y + s.z * U.m.z, 0.0, 1.0);
  var force = vec3<f32>(0.0);

  if (U.f.x > 0.0) {
    let w = textureLoad(curl, c, 0).xyz;
    let gw = 0.5 * vec3<f32>(cw(c + vec3<i32>(1, 0, 0), d) - cw(c - vec3<i32>(1, 0, 0), d),
                             cw(c + vec3<i32>(0, 1, 0), d) - cw(c - vec3<i32>(0, 1, 0), d),
                             cw(c + vec3<i32>(0, 0, 1), d) - cw(c - vec3<i32>(0, 0, 1), d));
    let len = length(gw);
    if (len > 1e-6) {
      force += U.f.x * h * cross(gw / len, w) * mix(U.f.w, 1.0, mask);
    }
  }

  let wp = world_of(U.g, vec3<f32>(c) + 0.5);
  if (U.f.y > 0.0 && mask > 0.001) {
    let q = wp * U.t.x - vec3<f32>(0.0, time * U.t.y * U.t.x, 0.0) + vec3<f32>(U.t2.y);
    let e = time * U.t.z;
    var cn = curl_noise(q + vec3<f32>(0.0, 0.0, e));
    cn += U.t2.x * curl_noise(q * 2.13 + vec3<f32>(5.2, 1.3, -1.7 * e));
    force += U.f.y * cn * mask;
  }

  if (U.f.z > 0.0 && mask > 0.001) {
    let cell = vec3<i32>(floor(wp / max(U.t.w, 1e-4))) + vec3<i32>(65536);
    let tick = u32(max(floor(time * U.m.w), 0.0));
    let hsh = pcg3d(vec3<u32>(cell) ^ vec3<u32>(tick * 2654435761u, tick * 1013904223u, tick));
    let r = vec3<f32>(hsh) * (2.0 / 4294967295.0) - vec3<f32>(1.0);
    force += U.f.z * r * mask;
  }
  textureStore(dst, c, vec4<f32>(force, mask));
}
