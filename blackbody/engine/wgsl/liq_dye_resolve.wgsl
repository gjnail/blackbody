// Dye for the renderer, pass 2: the mean dye of the liquid in each cell (absorption per metre rgb,
// scattering per metre). A cell without particles (the rim the surface reaches past them) takes the
// mean of its neighbours that have some, so the dye does not fade out at the surface.

const FX: f32 = 65536.0;

struct Params {
  n: vec4<f32>,   // simulation grid dims
};

@group(0) @binding(0) var<storage, read> acc: array<i32>;
@group(0) @binding(1) var dst: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn mean_at(c: vec3<i32>, n: vec3<i32>) -> vec4<f32> {
  let s = u32(c.x + n.x * (c.y + n.y * c.z)) * 5u;
  let w = f32(acc[s]) / FX;
  if (w < 1e-3) { return vec4<f32>(-1.0); }
  return vec4<f32>(f32(acc[s + 1u]), f32(acc[s + 2u]), f32(acc[s + 3u]), f32(acc[s + 4u])) / FX / w;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(U.n.xyz);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  var m = mean_at(c, n);
  if (m.x < 0.0) {
    var s = vec4<f32>(0.0);
    var k = 0.0;
    for (var a = 0; a < 3; a++) {
      for (var sg = -1; sg <= 1; sg += 2) {
        var e = vec3<i32>(0);
        e[a] = sg;
        let q = c + e;
        if (any(q < vec3<i32>(0)) || any(q >= n)) { continue; }
        let mq = mean_at(q, n);
        if (mq.x >= 0.0) {
          s += mq;
          k += 1.0;
        }
      }
    }
    m = select(vec4<f32>(0.0), s / max(k, 1.0), k > 0.0);
  }
  textureStore(dst, c, max(m, vec4<f32>(0.0)));
}
