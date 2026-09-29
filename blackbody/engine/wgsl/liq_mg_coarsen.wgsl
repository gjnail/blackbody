// Coarse cell types from fine ones (0 air, 1 liquid, 2 solid) with Dirichlet dominance.
//!include liq_mg_common.wgsl

@group(0) @binding(0) var Tf: texture_3d<f32>;
@group(0) @binding(1) var Tc: texture_storage_3d<r32float, write>;
@group(1) @binding(0) var<uniform> U: LMG;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let df = vec3<i32>(U.n.xyz);
  let dc = vec3<i32>(U.nc.xyz);
  let C = vec3<i32>(id);
  if (any(C >= dc)) { return; }
  var air = false;
  var liq = false;
  for (var j = 0; j < 8; j++) {
    let f = 2 * C + vec3<i32>(j & 1, (j >> 1) & 1, j >> 2);
    if (any(f >= df)) { continue; }
    let t = textureLoad(Tf, f, 0).x;
    if (t < 0.5) { air = true; } else if (t < 1.5) { liq = true; }
  }
  let t = select(select(2.0, 1.0, liq), 0.0, air);
  textureStore(Tc, C, vec4<f32>(t, 0.0, 0.0, 0.0));
}
