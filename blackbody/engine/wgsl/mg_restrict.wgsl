// Restrict the fine residual to the coarse right-hand side and clear the coarse correction.
// With rhs scaled by h^2, the coarse rhs is (2h)^2 / h^2 = 4 times the mean child residual.
//!include mg_common.wgsl

@group(0) @binding(0) var Rf: texture_3d<f32>;
@group(0) @binding(1) var Rc: texture_storage_3d<r32float, write>;
@group(0) @binding(2) var Pc: texture_storage_3d<r32float, write>;
@group(1) @binding(0) var<uniform> U: MG;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dc = vec3<i32>(U.nc.xyz);
  let df = vec3<i32>(U.n.xyz);
  let C = vec3<i32>(id);
  if (any(C >= dc)) { return; }
  var sum = 0.0;
  var cnt = 0.0;
  for (var j = 0; j < 8; j++) {
    let f = 2 * C + vec3<i32>(j & 1, (j >> 1) & 1, j >> 2);
    if (all(f < df)) {
      sum += textureLoad(Rf, f, 0).x;
      cnt += 1.0;
    }
  }
  textureStore(Rc, C, vec4<f32>(4.0 * sum / max(cnt, 1.0), 0.0, 0.0, 0.0));
  textureStore(Pc, C, vec4<f32>(0.0));
}
