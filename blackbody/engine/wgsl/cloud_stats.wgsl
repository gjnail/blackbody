// Clouds: a frame's numbers (engine/cloud.py measure()), one thread per column: the fastest wind, the
// strongest updraft, the highest cloud (m), how many columns have cloud, and the most cloud water, cloud ice
// and graupel/hail anywhere (kg/kg). (Positive floats compare as their bits: atomicMax on u32.)
//!include cloud_common.wgsl

@group(0) @binding(0) var vel: texture_3d<f32>;
@group(0) @binding(1) var A: texture_3d<f32>;
@group(0) @binding(2) var B: texture_3d<f32>;
@group(0) @binding(3) var<storage, read_write> st: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> base: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> U: Cloud;

fn amax(i: u32, v: f32) { atomicMax(&st[i], bitcast<u32>(max(v, 0.0))); }

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  if (i32(id.x) >= n.x || i32(id.z) >= n.z || id.y != 0u) { return; }
  var speed = 0.0;
  var up = 0.0;
  var top = 0.0;
  var cloudy = false;
  var qc = 0.0;
  var qi = 0.0;
  var qg = 0.0;
  for (var j = 0; j < n.y; j++) {
    let c = vec3<i32>(i32(id.x), j, i32(id.z));
    let v = textureLoad(vel, c, 0).xyz;
    speed = max(speed, length(v));
    up = max(up, v.y);
    let a = textureLoad(A, c, 0);
    let b = textureLoad(B, c, 0);
    let cond = a.z + b.x;
    if (cond > 1.0e-5) {
      cloudy = true;
      top = max(top, (f32(j) + 0.5) * U.g.n.w);
    }
    qc = max(qc, a.z);
    qi = max(qi, b.x + b.y);
    qg = max(qg, b.z);
  }
  amax(0u, speed);
  amax(1u, up);
  amax(2u, top);
  if (cloudy) { atomicAdd(&st[3], 1u); }
  amax(4u, qc);
  amax(5u, qi);
  amax(6u, qg);
}
