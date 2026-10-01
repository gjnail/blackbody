// Fabric bending multigrid, the coarse levels (see cloth_mg.wgsl): restriction of the finer level's
// residual, Chebyshev smoothing and residuals with the Galerkin operator (CSR, mass and bending parts),
// and prolongation of the correction back onto the finer level.

struct Params {
  a: vec4<f32>,   // vertex count (this level), substep (s), Chebyshev: weight of the last step, weight of the residual
  b: vec4<f32>,   // fabrics using multigrid (bitmask), _, _, _
};

// -- restrict: per coarse vertex, b = sum of w * (finer residual); x = 0; direction = 0 -------------------
@group(0) @binding(0) var<storage, read> RF: array<vec4<f32>>;        // the finer level's residual
@group(0) @binding(1) var<storage, read> RT: array<vec4<f32>>;        // per coarse vertex: (first, count) bits, then (fine index bits, weight)
@group(0) @binding(2) var<storage, read_write> B: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> XC: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> DC: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn gather_residual(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = id.x;
  if (c >= u32(U.a.x)) { return; }
  let head = RT[c];
  let first = bitcast<u32>(head.x);
  let cnt = bitcast<u32>(head.y);
  var b = vec3<f32>(0.0);
  for (var k = 0u; k < cnt; k++) {
    let e = RT[first + k];
    b += RF[bitcast<u32>(e.x)].xyz * e.y;
  }
  B[c] = vec4<f32>(b, 0.0);
  XC[c] = vec4<f32>(0.0);
  DC[c] = vec4<f32>(0.0);
}
