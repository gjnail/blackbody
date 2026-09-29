// Multigrid V-cycle used as the preconditioner of the liquid pressure CG. Every level stores its
// matrix the same way as the finest one: CO.x = diagonal (0 outside the liquid), CO.y = bit mask
// of the liquid neighbours (-x, +x, -y, +y, -z, +z), each coupled with -1. Coarse levels are
// built from cell types with Dirichlet dominance: a coarse cell is air if any child is air, else
// liquid if any child is liquid, else solid. Restriction is the transpose of trilinear
// prolongation, and smoothing is symmetric red/black Gauss-Seidel, so the V-cycle is a symmetric
// operator, as conjugate gradients requires.

struct LMG {
  n: vec4<f32>,   // this level's dims; w = red/black parity
  nc: vec4<f32>,  // the next coarser level's dims
  bc: vec4<f32>,  // sides open, top open, bottom open
};

fn mg_dir6(i: u32) -> vec3<i32> {
  var d = vec3<i32>(0);
  d[i >> 1u] = select(-1, 1, (i & 1u) == 1u);
  return d;
}

fn mg_outside_open(q: vec3<i32>, n: vec3<i32>, bc: vec4<f32>) -> bool {
  if (q.y < 0) { return bc.z > 0.5; }
  if (q.y >= n.y) { return bc.y > 0.5; }
  return bc.x > 0.5;
}
