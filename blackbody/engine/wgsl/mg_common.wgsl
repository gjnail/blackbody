// Geometric multigrid for the pressure Poisson equation  sum(p_nb) - k p = rhs  (rhs = h^2 div).
// Fluid neighbours add p_nb to the sum and 1 to k. Open boundaries are Dirichlet on the boundary
// face itself (ghost = -p), which adds 2 to k; this puts the boundary at the same place on every
// level, so coarse corrections stay consistent. Solid cells (level 0) and closed walls are Neumann:
// they drop out of the sum and of k.

struct MG {
  n: vec4<f32>,    // this level's dims; w = red/black parity
  bc: vec4<f32>,   // sides open, top open, bottom open, solids enabled (level 0)
  k: vec4<f32>,    // x = relaxation factor
  nc: vec4<f32>,   // coarse level dims (restriction / prolongation)
};

fn in_box(c: vec3<i32>, d: vec3<i32>) -> bool {
  return all(c >= vec3<i32>(0)) && all(c < d);
}

// 0 = outside, closed wall; 1 = outside, open; 2 = inside
fn nb_kind(q: vec3<i32>, d: vec3<i32>, bc: vec4<f32>) -> i32 {
  if (in_box(q, d)) { return 2; }
  var open = bc.x;
  if (q.y < 0) { open = bc.z; }
  if (q.y >= d.y) { open = bc.y; }
  return select(0, 1, open > 0.5);
}

// The six face neighbours: -x, +x, -y, +y, -z, +z.
fn mg_dir(i: i32) -> vec3<i32> {
  let s = select(-1, 1, (i & 1) == 1);
  let a = i >> 1;
  return vec3<i32>(select(0, s, a == 0), select(0, s, a == 1), select(0, s, a == 2));
}
