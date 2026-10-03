// Bullets through fabric (engine/bullet_media.py): where a bullet's way this frame passes through the cloth it leaves a
// hole (the vertices it passed within the hole's radius of are torn away, as a rip tears them: cloth_tear.wgsl, their
// triangles opening at once), and the cloth round the hole is pushed along its way: a curtain or a flag flicks where it
// is shot through.
//!include cloth_common.wgsl

const SEGS: u32 = 32u;

struct Params {
  k: vec4<f32>,                     // vertices, segments, the hole's radius (m), the push's reach (m)
  a: array<vec4<f32>, SEGS>,        // each segment's start (fire-local m), the push (m/s)
  b: array<vec4<f32>, SEGS>,        // and its end
};

@group(0) @binding(0) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> S: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> V: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.k.x)) { return; }
  let st = S[i];
  if (gone(st)) { return; }
  let p = X[i].xyz;
  var push = vec3<f32>(0.0);
  for (var k = 0u; k < u32(U.k.y); k++) {
    let a = U.a[k].xyz;
    let b = U.b[k].xyz;
    let ab = b - a;
    let L2 = max(dot(ab, ab), 1e-12);
    let t = clamp(dot(p - a, ab) / L2, 0.0, 1.0);
    let d = length(p - (a + ab * t));
    if (d < U.k.z) {
      S[i] = vec4<f32>(st.x, 1.0, -1.0, 1.0e9);      // (torn away: the hole)
      return;
    }
    if (d < U.k.w) {
      push += normalize(ab) * U.a[k].w * (1.0 - d / U.k.w);
    }
  }
  if (dot(push, push) > 0.0) {
    V[i] = vec4<f32>(V[i].xyz + push, V[i].w);
  }
}
