// Matter's surface, for drawing: smooth the distance near the surface (a light blur over each node and its six
// neighbours, in the band round the particles only), so it shows the lie of the matter rather than single particles.

struct Params {
  n: vec4<f32>,      // grid nodes (x, y, z), _
  b: vec4<f32>,      // the band's width (m), how much of its neighbours' average a node takes, _, _
};

@group(0) @binding(0) var src: texture_3d<f32>;
@group(0) @binding(1) var dst: texture_storage_3d<r32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(U.n.xyz);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let d = textureLoad(src, c, 0).x;
  if (d >= 0.999 * U.b.x) {
    textureStore(dst, c, vec4<f32>(d));   // (away from the matter: as it is)
    return;
  }
  var sum = 0.0;
  for (var a = 0; a < 3; a++) {
    for (var s = -1; s <= 1; s += 2) {
      var o = vec3<i32>(0);
      o[a] = s;
      sum += textureLoad(src, clamp(c + o, vec3<i32>(0), n - vec3<i32>(1)), 0).x;
    }
  }
  textureStore(dst, c, vec4<f32>(min(mix(d, sum / 6.0, U.b.y), U.b.x)));
}
