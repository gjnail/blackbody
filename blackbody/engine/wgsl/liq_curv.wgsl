// Surface tension, pass 2: mean curvature kappa = div(outward normal) (1/m), positive on convex
// liquid (a droplet), limited to what the grid can resolve.
//!include common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,  // minimum gradient strength for a usable normal
};

@group(0) @binding(0) var nrm: texture_3d<f32>;
@group(0) @binding(1) var dst: texture_storage_3d<r32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn n_at(q: vec3<i32>, n: vec3<i32>, fallback: vec3<f32>) -> vec3<f32> {
  if (!in_grid(q, n)) { return fallback; }
  let a = textureLoad(nrm, q, 0);
  return select(fallback, a.xyz, a.w > U.k.x);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let a = textureLoad(nrm, c, 0);
  var k = 0.0;
  if (a.w > U.k.x) {
    let f = a.xyz;
    let dx = n_at(c + vec3<i32>(1, 0, 0), n, f).x - n_at(c - vec3<i32>(1, 0, 0), n, f).x;
    let dy = n_at(c + vec3<i32>(0, 1, 0), n, f).y - n_at(c - vec3<i32>(0, 1, 0), n, f).y;
    let dz = n_at(c + vec3<i32>(0, 0, 1), n, f).z - n_at(c - vec3<i32>(0, 0, 1), n, f).z;
    let h = U.g.n.w;
    k = clamp((dx + dy + dz) / (2.0 * h), -1.0 / h, 1.0 / h);
  }
  textureStore(dst, c, vec4<f32>(k, 0.0, 0.0, 0.0));
}
