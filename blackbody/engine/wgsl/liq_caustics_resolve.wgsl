// Caustics, pass 2: gathered photon energy relative to what the texel receives with no liquid,
// with a small binomial blur against the photon grid's noise.

struct RParams {
  cm: vec4<f32>,   // caustic map dims (x, z), photons per texel (total)
};

@group(0) @binding(0) var<storage, read> acc: array<u32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: RParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let cm = vec2<i32>(U.cm.xy);
  let c = vec2<i32>(id.xy);
  if (any(c >= cm)) { return; }
  var s = 0.0;
  var wsum = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let q = c + vec2<i32>(x, y);
      if (any(q < vec2<i32>(0)) || any(q >= cm)) { continue; }
      let w = f32((2 - abs(x)) * (2 - abs(y)));
      s += w * f32(acc[u32(q.x + cm.x * q.y)]);
      wsum += w;
    }
  }
  let ratio = s / max(wsum, 1.0) / (U.cm.z * 4096.0);
  textureStore(dst, c, vec4<f32>(ratio, 0.0, 0.0, 0.0));
}
