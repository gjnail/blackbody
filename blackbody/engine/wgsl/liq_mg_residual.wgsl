// res = rhs - A z
//!include liq_mg_common.wgsl

@group(0) @binding(0) var Z: texture_3d<f32>;
@group(0) @binding(1) var RHS: texture_3d<f32>;
@group(0) @binding(2) var CO: texture_3d<f32>;
@group(0) @binding(3) var RES: texture_storage_3d<r32float, write>;
@group(1) @binding(0) var<uniform> U: LMG;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let d = vec3<i32>(U.n.xyz);
  let c = vec3<i32>(id);
  if (any(c >= d)) { return; }
  let co = textureLoad(CO, c, 0);
  var r = 0.0;
  if (co.x > 0.0) {
    let mask = u32(co.y);
    var s = 0.0;
    for (var i = 0u; i < 6u; i++) {
      if (((mask >> i) & 1u) != 0u) { s += textureLoad(Z, c + mg_dir6(i), 0).x; }
    }
    r = textureLoad(RHS, c, 0).x - (co.x * textureLoad(Z, c, 0).x - s);
  }
  textureStore(RES, c, vec4<f32>(r, 0.0, 0.0, 0.0));
}
