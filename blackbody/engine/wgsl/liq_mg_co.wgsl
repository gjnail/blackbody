// Matrix of a coarse level from its cell types: 1 toward liquid and air neighbours (air is
// Dirichlet at the neighbour's centre), 2 toward an open domain boundary, 0 toward solids.
//!include liq_mg_common.wgsl

@group(0) @binding(0) var T: texture_3d<f32>;
@group(0) @binding(1) var CO: texture_storage_3d<rg32float, write>;
@group(1) @binding(0) var<uniform> U: LMG;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let d = vec3<i32>(U.n.xyz);
  let c = vec3<i32>(id);
  if (any(c >= d)) { return; }
  let t = textureLoad(T, c, 0).x;
  if (t < 0.5 || t > 1.5) {
    textureStore(CO, c, vec4<f32>(0.0));
    return;
  }
  var diag = 0.0;
  var mask = 0u;
  for (var i = 0u; i < 6u; i++) {
    let q = c + mg_dir6(i);
    if (any(q < vec3<i32>(0)) || any(q >= d)) {
      if (mg_outside_open(q, d, U.bc)) { diag += 2.0; }
      continue;
    }
    let tq = textureLoad(T, q, 0).x;
    if (tq < 0.5) {
      diag += 1.0;
    } else if (tq < 1.5) {
      diag += 1.0;
      mask |= 1u << i;
    }
  }
  textureStore(CO, c, vec4<f32>(diag, f32(mask), 0.0, 0.0));
}
