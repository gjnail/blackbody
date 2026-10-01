// Fabric: each vertex's normal and area (a third of the triangles around it that are still there),
// for the air's drag on it, the coupling to the gas and shading.
//!include cloth_common.wgsl

struct Params {
  sim: vec4<f32>,   // vertex count, _, _, _
};

@group(0) @binding(0) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> S: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> T: array<vec4<u32>>;   // triangles: three vertices, fabric
@group(0) @binding(3) var<storage, read> VT: array<u32>;        // per vertex: first triangle, count, then the lists
@group(0) @binding(4) var<storage, read_write> N: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.sim.x)) { return; }
  let first = VT[i * 2u];
  let cnt = VT[i * 2u + 1u];
  var n = vec3<f32>(0.0);
  var a = 0.0;
  for (var k = 0u; k < cnt; k++) {
    let t = T[VT[first + k]];
    if (gone(S[t.x]) || gone(S[t.y]) || gone(S[t.z])) { continue; }
    let c = cross(X[t.y].xyz - X[t.x].xyz, X[t.z].xyz - X[t.x].xyz);
    n += c;
    a += length(c);
  }
  let l = length(n);
  N[i] = vec4<f32>(select(vec3<f32>(0.0, 0.0, 1.0), n / l, l > 1e-12), a / 6.0);
}
