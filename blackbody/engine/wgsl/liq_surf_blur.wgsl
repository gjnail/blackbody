// Surface builder, pass 3: a 3x3x3 binomial blur of the distance (and velocity), which smooths the
// last of the particle graininess out of the surface.

struct Params {
  nf: vec4<f32>,   // surface grid dims
};

@group(0) @binding(0) var src: texture_3d<f32>;
@group(0) @binding(1) var dst: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let ni = vec3<i32>(U.nf.xyz);
  let c = vec3<i32>(id);
  if (any(c >= ni)) { return; }
  var s = vec4<f32>(0.0);
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let o = vec3<i32>(x, y, z);
        let w = f32((2 - abs(x)) * (2 - abs(y)) * (2 - abs(z)));
        s += w * textureLoad(src, clamp(c + o, vec3<i32>(0), ni - vec3<i32>(1)), 0);
      }
    }
  }
  textureStore(dst, c, s / 64.0);
}
