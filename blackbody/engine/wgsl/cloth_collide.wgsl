// Fabric collisions: the self-collision pushes (cloth_self.wgsl), then the ground and the colliders
// (their exact shapes, moving or not), each keeping the cloth its thickness away and holding it with
// friction against the surface's own motion, so cloth lies on a table, drapes over a chair and is
// carried along by a moving object. Wet cloth clings: the water between it and the surface holds it
// (more friction, and a film that draws it down onto a surface it is within a couple of millimetres of).
//!include common.wgsl
//!include meshsdf.wgsl
//!include colliders.wgsl
//!include cloth_common.wgsl

struct Params {
  sim: vec4<f32>,    // dt, vertex count, ground (1/0), self-collision on (1/0)
  ccnt: vec4<f32>,
  col: array<Collider, MAX_COLLIDERS>,
};

@group(0) @binding(0) var<storage, read_write> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> P: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> D: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> M: array<Mat>;
@group(0) @binding(5) var atlas: texture_3d<f32>;
@group(1) @binding(0) var<uniform> U: Params;

fn col_normal(k: Collider, p: vec3<f32>, e: f32) -> vec3<f32> {
  let g = vec3<f32>(col_sdf(k, p + vec3<f32>(e, 0.0, 0.0)) - col_sdf(k, p - vec3<f32>(e, 0.0, 0.0)),
                    col_sdf(k, p + vec3<f32>(0.0, e, 0.0)) - col_sdf(k, p - vec3<f32>(0.0, e, 0.0)),
                    col_sdf(k, p + vec3<f32>(0.0, 0.0, e)) - col_sdf(k, p - vec3<f32>(0.0, 0.0, e)));
  let l = length(g);
  return select(vec3<f32>(0.0, 1.0, 0.0), g / l, l > 1e-9);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.sim.y)) { return; }
  let x4 = X[i];
  if (x4.w <= 0.0) { return; }   // pinned
  let dt = U.sim.x;
  let mat = M[u32(V[i].w + 0.5)];
  let wet = clamp(P[i].w, 0.0, 1.0);
  let mu = mat.a.z * (1.0 + 1.5 * wet);
  let thick = mat.a.w;
  let cling = 0.002 * wet;   // m: how far off a surface the water film still reaches
  let p0 = P[i].xyz;
  var x = x4.xyz;
  if (U.sim.w > 0.5) { x += D[i].xyz; }
  if (U.sim.z > 0.5 && cling > 0.0 && x.y >= thick && x.y < thick + cling) { x.y -= (x.y - thick) * 0.2 * wet; }
  if (U.sim.z > 0.5 && x.y < thick) {
    let pen = thick - x.y;
    x.y = thick;
    let t = vec2<f32>(x.x - p0.x, x.z - p0.z);
    let tl = length(t);
    if (tl > 1e-9) {
      let keep = 1.0 - min(1.0, mu * pen / tl);
      x.x = p0.x + t.x * keep;
      x.z = p0.z + t.y * keep;
    }
  }
  for (var c = 0; c < i32(U.ccnt.x); c++) {
    let k = U.col[c];
    let d = col_sdf(k, x) - thick;
    if (d >= 0.0 && d < cling) {
      // the water film draws it onto the surface
      x -= col_normal(k, x, max(0.25 * thick, 1e-3)) * (d * 0.2 * wet);
      continue;
    }
    if (d >= 0.0) { continue; }
    let n = col_normal(k, x, max(0.25 * thick, 1e-3));
    x -= n * d;
    // friction against the surface as it moves
    let rel = (x - p0) - col_velocity(k, x) * dt;
    let rt = rel - dot(rel, n) * n;
    let tl = length(rt);
    if (tl > 1e-9) { x -= rt * min(1.0, mu * (-d) / tl); }
  }
  X[i] = vec4<f32>(x, x4.w);
}
