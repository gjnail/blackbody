// Semi-Lagrangian advection does not conserve what it carries: where heavy fuel vapour settles onto
// the floor, the cells there keep drawing fuel from above and the total grows. With heavy vapour or
// flame fronts on, the fuel's total is measured before and after advection (per-workgroup partial
// sums, then one total), and the reaction scales the fuel back, never up, to what was there.
//!include common.wgsl

struct Params {
  g: Grid,
  s: vec4<f32>,   // x = which total this is (0 before advection, 1 after)
};

@group(0) @binding(0) var scal: texture_3d<f32>;
@group(0) @binding(1) var<storage, read_write> part: array<f32>;
@group(0) @binding(2) var<storage, read_write> tot: array<f32>;
@group(1) @binding(0) var<uniform> U: Params;

var<workgroup> acc: array<f32, 256>;

fn groups() -> vec3<u32> { return (vec3<u32>(U.g.n.xyz) + vec3<u32>(7u, 7u, 3u)) / vec3<u32>(8u, 8u, 4u); }

@compute @workgroup_size(8, 8, 4)
fn partial(@builtin(global_invocation_id) id: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>,
           @builtin(local_invocation_index) li: u32) {
  let d = gdim(U.g);
  let c = vec3<i32>(id);
  var v = 0.0;
  if (all(c < d)) { v = max(textureLoad(scal, c, 0).y, 0.0); }
  acc[li] = v;
  workgroupBarrier();
  var stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (li < stride) { acc[li] = acc[li] + acc[li + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (li == 0u) {
    let g = groups();
    part[wg.x + wg.y * g.x + wg.z * g.x * g.y] = acc[0];
  }
}

@compute @workgroup_size(256, 1, 1)
fn total(@builtin(local_invocation_index) li: u32) {
  let g = groups();
  let count = g.x * g.y * g.z;
  var v = 0.0;
  for (var i = li; i < count; i += 256u) { v += part[i]; }
  acc[li] = v;
  workgroupBarrier();
  var stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (li < stride) { acc[li] = acc[li] + acc[li + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (li == 0u) { tot[u32(U.s.x)] = acc[0]; }
}
