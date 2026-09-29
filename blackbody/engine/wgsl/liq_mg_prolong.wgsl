// Add the trilinearly interpolated coarse correction to the fine solution (liquid cells only;
// coarse cells outside the grid count as zero, which keeps this the transpose of restriction).
//!include liq_mg_common.wgsl

@group(0) @binding(0) var Z: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var Zc: texture_3d<f32>;
@group(0) @binding(2) var CO: texture_3d<f32>;
@group(1) @binding(0) var<uniform> U: LMG;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let df = vec3<i32>(U.n.xyz);
  let dc = vec3<i32>(U.nc.xyz);
  let c = vec3<i32>(id);
  if (any(c >= df)) { return; }
  if (textureLoad(CO, c, 0).x <= 0.0) { return; }
  let x = (vec3<f32>(c) + vec3<f32>(0.5)) * 0.5 - vec3<f32>(0.5);
  let fl = floor(x);
  let t = x - fl;
  let i0 = vec3<i32>(fl);
  var e = 0.0;
  for (var j = 0; j < 8; j++) {
    let o = vec3<i32>(j & 1, (j >> 1) & 1, j >> 2);
    let q = i0 + o;
    if (any(q < vec3<i32>(0)) || any(q >= dc)) { continue; }
    let wv = mix(vec3<f32>(1.0) - t, t, vec3<f32>(o));
    e += wv.x * wv.y * wv.z * textureLoad(Zc, q, 0).x;
  }
  textureStore(Z, c, vec4<f32>(textureLoad(Z, c).x + e, 0.0, 0.0, 0.0));
}
