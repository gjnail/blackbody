// A box closed on every side can only hold an expansion that sums to zero, so for a closed box the
// mean of the pressure equation's right-hand side over the gas is removed: whatever swelling is
// left over just raises the box's pressure evenly. Three passes: per-workgroup partial sums, one
// total, then the subtraction.
//!include common.wgsl

struct Params { g: Grid };

@group(0) @binding(0) var rhs: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var sdf: texture_3d<f32>;
@group(0) @binding(2) var<storage, read_write> part: array<vec2<f32>>;  // per workgroup: sum, gas cells; the last entry holds the mean
@group(1) @binding(0) var<uniform> U: Params;

var<workgroup> acc: array<vec2<f32>, 256>;

fn groups() -> vec3<u32> { return (vec3<u32>(U.g.n.xyz) + vec3<u32>(7u, 7u, 3u)) / vec3<u32>(8u, 8u, 4u); }

@compute @workgroup_size(8, 8, 4)
fn partial(@builtin(global_invocation_id) id: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>,
           @builtin(local_invocation_index) li: u32) {
  let d = gdim(U.g);
  let c = vec3<i32>(id);
  var v = vec2<f32>(0.0);
  if (all(c < d) && textureLoad(sdf, c, 0).x >= 0.0) { v = vec2<f32>(textureLoad(rhs, c).x, 1.0); }
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
  var v = vec2<f32>(0.0);
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
  if (li == 0u) { part[count] = vec2<f32>(acc[0].x / max(acc[0].y, 1.0), 0.0); }
}

@compute @workgroup_size(8, 8, 4)
fn subtract(@builtin(global_invocation_id) id: vec3<u32>) {
  let d = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= d)) { return; }
  if (textureLoad(sdf, c, 0).x < 0.0) { return; }
  let g = groups();
  let mean = part[g.x * g.y * g.z].x;
  textureStore(rhs, c, vec4<f32>(textureLoad(rhs, c).x - mean, 0.0, 0.0, 0.0));
}
