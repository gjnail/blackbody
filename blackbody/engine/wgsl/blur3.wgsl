// Separable Gaussian blur of a 3D volume along one axis, with zero padding outside the volume.

struct Params {
  n: vec4<f32>,    // dims
  ax: vec4<f32>,   // axis (unit vector), radius in cells
  k: vec4<f32>,    // sigma in cells
};

@group(0) @binding(0) var src: texture_3d<f32>;
@group(0) @binding(1) var dst: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let d = vec3<i32>(U.n.xyz);
  let c = vec3<i32>(id);
  if (any(c >= d)) { return; }
  let ax = vec3<i32>(U.ax.xyz);
  let r = i32(U.ax.w);
  let inv = -0.5 / max(U.k.x * U.k.x, 1e-4);
  var acc = vec4<f32>(0.0);
  var wsum = 0.0;
  for (var i = -r; i <= r; i++) {
    let w = exp(f32(i * i) * inv);
    let q = c + ax * i;
    if (all(q >= vec3<i32>(0)) && all(q < d)) {
      acc += w * textureLoad(src, q, 0);
    }
    wsum += w;
  }
  textureStore(dst, c, acc / wsum);
}
