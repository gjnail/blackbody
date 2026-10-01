// Narrow band, pass 1 (after the particles are splatted): the deep liquid has no particles, only
// the grid. Its cells count as full of liquid, and faces no particle reached next to it keep the
// velocity the grid had at the end of the last step.
//!include common.wgsl
//!include liq_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,   // rest density (particles)
};

@group(0) @binding(0) var vold: texture_3d<f32>;
@group(0) @binding(1) var vprev: texture_3d<f32>;
@group(0) @binding(2) var<storage, read> band: array<u32>;
@group(0) @binding(3) var dens: texture_storage_3d<r32float, read_write>;
@group(0) @binding(4) var dst: texture_storage_3d<rgba32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn is_deep(c: vec3<i32>, n: vec3<i32>) -> bool {
  if (!in_grid(c, n)) { return false; }
  return band[nidx(c, n)] != 0u;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c > n)) { return; }
  if (is_deep(c, n)) {
    // exactly full: more (the splat of dense neighbours reaching in) would set the volume correction
    // pushing liquid out of the deep region, a source the band then turns into particles
    textureStore(dens, c, vec4<f32>(U.k.x, 0.0, 0.0, 0.0));
  }
  let a = textureLoad(vold, c, 0);
  var v = a.xyz;
  var mask = u32(a.w);
  let p = textureLoad(vprev, c, 0).xyz;
  for (var k = 0u; k < 3u; k++) {
    if (any(c > face_lim(k, n)) || flag_valid(mask, k)) { continue; }
    var e = vec3<i32>(0);
    e[k] = 1;
    if (is_deep(c, n) || is_deep(c - e, n)) {
      v[k] = p[k];
      mask |= 1u << k;
    }
  }
  textureStore(dst, c, vec4<f32>(v, f32(mask)));
}
