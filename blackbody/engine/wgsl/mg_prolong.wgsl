// Add the trilinearly interpolated coarse correction to the fine pressure.
//!include mg_common.wgsl

@group(0) @binding(0) var Pf: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var Pc: texture_3d<f32>;
@group(0) @binding(2) var S: texture_3d<f32>;
@group(1) @binding(0) var<uniform> U: MG;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let df = vec3<i32>(U.n.xyz);
  let dc = vec3<i32>(U.nc.xyz);
  let c = vec3<i32>(id);
  if (any(c >= df)) { return; }
  if (U.bc.w > 0.5 && textureLoad(S, c, 0).x < 0.0) { return; }
  // fine centre in coarse index space (coarse centres at integers)
  let x = (vec3<f32>(c) + vec3<f32>(0.5)) * 0.5 - vec3<f32>(0.5);
  let fl = floor(x);
  let t = x - fl;
  let i0 = vec3<i32>(fl);
  let hi = dc - vec3<i32>(1);
  var e = 0.0;
  for (var j = 0; j < 8; j++) {
    let o = vec3<i32>(j & 1, (j >> 1) & 1, j >> 2);
    let q = clamp(i0 + o, vec3<i32>(0), hi);
    let w = select(1.0 - t.x, t.x, o.x == 1) * select(1.0 - t.y, t.y, o.y == 1) * select(1.0 - t.z, t.z, o.z == 1);
    e += w * textureLoad(Pc, q, 0).x;
  }
  let p = textureLoad(Pf, c).x;
  textureStore(Pf, c, vec4<f32>(p + e, 0.0, 0.0, 0.0));
}
