// Ap = A p for the liquid pressure matrix (diagonal and liquid-neighbour mask in CO), with the
// per-workgroup partial sums of p . Ap.
//!include liq_cg_common.wgsl
//!include liq_cg_reduce.wgsl

@group(0) @binding(0) var P: texture_3d<f32>;
@group(0) @binding(1) var CO: texture_3d<f32>;
@group(0) @binding(2) var AP: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var<storage, read_write> partials: array<f32>;
@group(1) @binding(0) var<uniform> U: CGParams;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32,
        @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let c = vec3<i32>(id);
  var v = 0.0;
  if (all(c < vec3<i32>(U.n.xyz))) {
    let co = textureLoad(CO, c, 0);
    var ap = 0.0;
    if (co.x > 0.0) {
      let p = textureLoad(P, c, 0).x;
      let mask = u32(co.y);
      var s = 0.0;
      for (var i = 0u; i < 6u; i++) {
        if (((mask >> i) & 1u) != 0u) {
          var d = vec3<i32>(0);
          d[i >> 1u] = select(-1, 1, (i & 1u) == 1u);
          s += textureLoad(P, c + d, 0).x;
        }
      }
      ap = co.x * p - s;
      v = p * ap;
    }
    textureStore(AP, c, vec4<f32>(ap, 0.0, 0.0, 0.0));
  }
  reduce_store(li, v, wg_index(wid, nwg));
}
