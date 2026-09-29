// Grid velocities from the particle sums: momentum / weight per face, with a flag for faces that
// received particle data. Also writes the particle density (particles per cell) at cell centres.
//!include common.wgsl
//!include liq_common.wgsl

struct Params { g: Grid };

@group(0) @binding(0) var<storage, read> acc: array<i32>;
@group(0) @binding(1) var vold: texture_storage_3d<rgba32float, write>;
@group(0) @binding(2) var dens: texture_storage_3d<r32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let m = n + vec3<i32>(1);
  let c = vec3<i32>(id);
  if (any(c >= m)) { return; }
  let i = nidx(c, m) * ACC;
  var v = vec3<f32>(0.0);
  var mask = 0u;
  for (var k = 0u; k < 3u; k++) {
    let den = acc[i + 2u * k + 1u];
    if (den > 0) {
      v[k] = (f32(acc[i + 2u * k]) / FX_M) / (f32(den) / FX_W);
      mask |= 1u << k;
    }
  }
  textureStore(vold, c, vec4<f32>(v, f32(mask)));
  if (all(c < n)) {
    textureStore(dens, c, vec4<f32>(f32(acc[i + 6u]) / FX_W, 0.0, 0.0, 0.0));
  }
}
