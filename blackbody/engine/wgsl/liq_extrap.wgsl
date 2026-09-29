// One layer of velocity extrapolation: faces without fluid data take the mean of their valid
// neighbours on the same component lattice, so particles near the surface and in the air see a
// continuous velocity field. Fixed (wall) faces are neither changed nor used.
//!include common.wgsl
//!include liq_common.wgsl

struct Params { g: Grid };

@group(0) @binding(0) var src: texture_3d<f32>;
@group(0) @binding(1) var dst: texture_storage_3d<rgba32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let m = n + vec3<i32>(1);
  let c = vec3<i32>(id);
  if (any(c >= m)) { return; }
  let a = textureLoad(src, c, 0);
  let mask = u32(a.w);
  var v = a.xyz;
  var nm = mask;
  for (var k = 0u; k < 3u; k++) {
    if (flag_valid(mask, k) || flag_fixed(mask, k)) { continue; }
    var sum = 0.0;
    var cnt = 0.0;
    for (var i = 0; i < 6; i++) {
      let s = select(-1, 1, (i & 1) == 1);
      let ax = i >> 1;
      var d = vec3<i32>(0);
      d[ax] = s;
      let q = c + d;
      if (any(q < vec3<i32>(0)) || any(q >= m)) { continue; }
      let b = textureLoad(src, q, 0);
      let bm = u32(b.w);
      if (flag_valid(bm, k) && !flag_fixed(bm, k)) {
        sum += b[k];
        cnt += 1.0;
      }
    }
    if (cnt > 0.0) {
      v[k] = sum / cnt;
      nm |= 1u << k;
    }
  }
  textureStore(dst, c, vec4<f32>(v, f32(nm)));
}
