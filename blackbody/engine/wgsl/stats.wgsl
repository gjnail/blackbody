// Grid statistics in one pass: max speed (for adaptive substeps) and the bounding box of visible
// content (for render bounds and auto-framing). Results are atomics in a small buffer:
// [0] max speed as float bits, [1..3] min cell, [4..6] max cell, [7] burning cell count.
//!include common.wgsl

struct Params {
  g: Grid,
  th: vec4<f32>,  // x = visibility threshold on smoke, y = on flame, z = on temperature
};

@group(0) @binding(0) var vel: texture_3d<f32>;
@group(0) @binding(1) var scal: texture_3d<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<atomic<u32>, 8>;
@group(1) @binding(0) var<uniform> U: Params;

var<workgroup> red: array<f32, 256>;
var<workgroup> vis_lo: array<atomic<u32>, 3>;
var<workgroup> vis_hi: array<atomic<u32>, 3>;
var<workgroup> burning: atomic<u32>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let d = gdim(U.g);
  let c = vec3<i32>(id);
  if (li == 0u) {
    for (var i = 0; i < 3; i++) {
      atomicStore(&vis_lo[i], 0xffffffffu);
      atomicStore(&vis_hi[i], 0u);
    }
    atomicStore(&burning, 0u);
  }
  workgroupBarrier();
  var speed = 0.0;
  if (all(c < d)) {
    speed = length(vel_centre(vel, c));
    let s = textureLoad(scal, c, 0);
    if (s.z > U.th.x || s.w > U.th.y || s.x > U.th.z) {
      for (var i = 0; i < 3; i++) {
        atomicMin(&vis_lo[i], u32(c[i]));
        atomicMax(&vis_hi[i], u32(c[i]));
      }
    }
    if (s.w > U.th.y) { atomicAdd(&burning, 1u); }
  }
  red[li] = speed;
  workgroupBarrier();
  var stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (li < stride) { red[li] = max(red[li], red[li + stride]); }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (li == 0u) {
    atomicMax(&out[0], bitcast<u32>(red[0]));
    for (var i = 0; i < 3; i++) {
      let lo = atomicLoad(&vis_lo[i]);
      if (lo != 0xffffffffu) {
        atomicMin(&out[1 + i], lo);
        atomicMax(&out[4 + i], atomicLoad(&vis_hi[i]));
      }
    }
    atomicAdd(&out[7], atomicLoad(&burning));
  }
}
