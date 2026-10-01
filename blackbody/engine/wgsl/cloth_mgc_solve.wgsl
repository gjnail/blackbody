// Fabric bending multigrid, a coarse level (see cloth_mg.wgsl): Chebyshev smoothing of A e = b and its
// residual, A = M / h^2 + Q in CSR (each entry: column bits, mass part, bending part).

struct Params {
  a: vec4<f32>,   // vertex count (this level), substep (s), Chebyshev: weight of the last step, weight of the residual
  b: vec4<f32>,   // fabrics using multigrid (bitmask), _, _, _
};

@group(0) @binding(0) var<storage, read> XI: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> XO: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> B: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> DC: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> AS: array<u32>;          // row starts
@group(0) @binding(5) var<storage, read> A: array<vec4<f32>>;     // column bits, mass part, bending part, _
@group(0) @binding(6) var<storage, read> DG: array<vec4<f32>>;    // diagonal mass part, diagonal bending part, held (1/0), _
@group(1) @binding(0) var<uniform> U: Params;

fn residual(c: u32) -> vec3<f32> {
  let ih2 = 1.0 / (U.a.y * U.a.y);
  var ax = vec3<f32>(0.0);
  for (var k = AS[c]; k < AS[c + 1u]; k++) {
    let e = A[k];
    ax += XI[bitcast<u32>(e.x)].xyz * (e.y * ih2 + e.z);
  }
  return B[c].xyz - ax;
}

@compute @workgroup_size(64, 1, 1)
fn cheb(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = id.x;
  if (c >= u32(U.a.x)) { return; }
  let g = DG[c];
  let x = XI[c];
  if (g.z > 0.5) {
    XO[c] = x;
    return;
  }
  let d = g.x / (U.a.y * U.a.y) + g.y;
  let dx = DC[c].xyz * U.a.z + residual(c) / max(d, 1e-30) * U.a.w;
  DC[c] = vec4<f32>(dx, 0.0);
  XO[c] = vec4<f32>(x.xyz + dx, 0.0);
}

@compute @workgroup_size(64, 1, 1)
fn resid(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = id.x;
  if (c >= u32(U.a.x)) { return; }
  if (DG[c].z > 0.5) {
    XO[c] = vec4<f32>(0.0);
    return;
  }
  XO[c] = vec4<f32>(residual(c), 0.0);
}
