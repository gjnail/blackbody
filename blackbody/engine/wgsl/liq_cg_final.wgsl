// Sum the partial dot products and update the CG scalars.
//   mode 0: rz_new = sum; beta = first ? 0 : rz_new / rz; rz = rz_new
//   mode 1: pAp = sum; alpha = rz / pAp
//!include liq_cg_common.wgsl
//!include liq_cg_reduce.wgsl

@group(0) @binding(0) var<storage, read_write> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> S: array<f32>;
@group(1) @binding(0) var<uniform> U: CGParams;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_index) li: u32) {
  let count = u32(U.n.w);
  var s = 0.0;
  for (var i = li; i < count; i += 256u) {
    s += partials[i];
  }
  red[li] = s;
  workgroupBarrier();
  var stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (li < stride) { red[li] = red[li] + red[li + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (li == 0u) {
    let total = red[0];
    if (U.k.x < 0.5) {
      let rz = S[0];
      var beta = 0.0;
      if (U.k.y < 0.5 && abs(rz) > 1e-30) { beta = total / rz; }
      S[0] = total;
      S[2] = beta;
    } else {
      S[3] = total;
      var alpha = 0.0;
      if (abs(total) > 1e-30) { alpha = S[0] / total; }
      S[1] = alpha;
    }
  }
}
