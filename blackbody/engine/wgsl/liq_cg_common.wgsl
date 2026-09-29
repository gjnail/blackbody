// Preconditioned conjugate gradient for the liquid pressure, entirely on the GPU. Dot products
// reduce per workgroup into `partials` (declared by the including kernel); liq_cg_final sums
// those and derives the CG scalars in S, which the update kernels read, so no value ever has to
// come back to the CPU.
//
// S: [0] r.z, [1] alpha, [2] beta, [3] p.Ap

struct CGParams {
  n: vec4<f32>,   // grid dims; w = number of partial sums (final)
  k: vec4<f32>,   // final: x = mode (0: r.z -> beta, 1: p.Ap -> alpha), y = first iteration (1/0)
};
