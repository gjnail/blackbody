// Fabric bending by multigrid, for wet cloth (engine/cloth_mg.py): the water it holds adds to its mass,
// so over the same substep its bending moves it less. The solve (cloth_mg.wgsl) is built on the dry mass;
// with bending as weak against the cloth's inertia over a substep as it is, scaling its correction by
// dry over wet mass gives the solve with the wet mass.
//!include cloth_common.wgsl

struct Params {
  a: vec4<f32>,   // vertex count, _, _, _
  b: vec4<f32>,   // fabrics using multigrid (bitmask), _, _, _
};

@group(0) @binding(0) var<storage, read_write> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> XT: array<vec4<f32>>;   // positions predicted for this substep
@group(0) @binding(2) var<storage, read> P: array<vec4<f32>>;    // w: wetness
@group(0) @binding(3) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> M: array<Mat>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.a.x)) { return; }
  let wet = P[i].w;
  if (wet <= 0.0) { return; }
  let fi = u32(V[i].w + 0.5);
  if ((u32(U.b.x + 0.5) & (1u << fi)) == 0u) { return; }
  let x = X[i];
  let t = XT[i].xyz;
  X[i] = vec4<f32>(t + (x.xyz - t) / (1.0 + wet * M[fi].f.y), x.w);
}
