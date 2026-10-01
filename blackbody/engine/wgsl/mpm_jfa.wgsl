// Matter's surface, for drawing: the distance to it from every node of the grid (jump flooding: Rong and Tan 2006), so
// that rays, shadows and the sky's occlusion through the empty grid see how far the matter really is.
// Each node near the matter (a particle within R) is a seed: its own position and its distance to the surface. Each
// pass, every node takes the best of the seeds its neighbours k nodes away have, for k halving down to 1.

struct Params {
  n: vec4<f32>,      // grid nodes (x, y, z), node spacing (m)
  b: vec4<f32>,      // the band's width (m: nodes with no particle within R have it), step k (nodes), _, _
};

@group(0) @binding(0) var src: texture_3d<f32>;
@group(0) @binding(1) var dst: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

// The seeds: near the matter, the node itself (xyz, grid units) and its distance (w, m); elsewhere none (w < 0).
@compute @workgroup_size(8, 8, 4)
fn init(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(U.n.xyz);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let d = textureLoad(src, c, 0).x;
  if (d < 0.999 * U.b.x) {
    textureStore(dst, c, vec4<f32>(vec3<f32>(c), max(d, 0.0)));
  } else {
    textureStore(dst, c, vec4<f32>(0.0, 0.0, 0.0, -1.0));
  }
}

@compute @workgroup_size(8, 8, 4)
fn step(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(U.n.xyz);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let k = i32(U.b.y);
  let h = U.n.w;
  var best = textureLoad(src, c, 0);
  var bd = 1.0e9;
  if (best.w >= 0.0) { bd = length(vec3<f32>(c) - best.xyz) * h + best.w; }
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let o = c + vec3<i32>(x, y, z) * k;
        if (any(o < vec3<i32>(0)) || any(o >= n)) { continue; }
        let s = textureLoad(src, o, 0);
        if (s.w < 0.0) { continue; }
        let d = length(vec3<f32>(c) - s.xyz) * h + s.w;
        if (d < bd) {
          bd = d;
          best = s;
        }
      }
    }
  }
  textureStore(dst, c, best);
}
