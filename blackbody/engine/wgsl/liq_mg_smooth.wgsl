// One red/black Gauss-Seidel half-sweep of A z = rhs, in place.
//!include liq_mg_common.wgsl

@group(0) @binding(0) var Z: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var RHS: texture_3d<f32>;
@group(0) @binding(2) var CO: texture_3d<f32>;
@group(1) @binding(0) var<uniform> U: LMG;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let d = vec3<i32>(U.n.xyz);
  let par = i32(U.n.w);
  let c = vec3<i32>(2 * i32(id.x) + ((i32(id.y) + i32(id.z) + par) & 1), i32(id.y), i32(id.z));
  if (any(c >= d)) { return; }
  let co = textureLoad(CO, c, 0);
  if (co.x <= 0.0) { return; }
  let mask = u32(co.y);
  var s = 0.0;
  for (var i = 0u; i < 6u; i++) {
    if (((mask >> i) & 1u) != 0u) { s += textureLoad(Z, c + mg_dir6(i)).x; }
  }
  textureStore(Z, c, vec4<f32>((textureLoad(RHS, c, 0).x + s) / co.x, 0.0, 0.0, 0.0));
}
