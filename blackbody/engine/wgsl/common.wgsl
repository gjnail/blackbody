// Shared solver helpers.
//
// Positions are in cell units: cell (i,j,k) spans [i, i+1) on each axis and its centre is at
// i + 0.5. Velocity lives on cell faces (a MAC grid) packed into one texture of size n + 1:
// texel (i,j,k).x is u on the -x face of cell (i,j,k), .y is v on its -y face and .z is w on its
// -z face. Scalars live at cell centres.

struct Grid {
  n: vec4<f32>,    // nx, ny, nz, h (metres per cell)
  org: vec4<f32>,  // world position of the grid corner; w = simulation time (s)
  bc: vec4<f32>,   // x = sides open, y = top open, z = bottom open (0 means solid ground), w = dt (s)
};

fn gdim(g: Grid) -> vec3<i32> { return vec3<i32>(g.n.xyz); }

fn in_grid(c: vec3<i32>, d: vec3<i32>) -> bool {
  return all(c >= vec3<i32>(0)) && all(c < d);
}

fn world_of(g: Grid, p: vec3<f32>) -> vec3<f32> { return g.org.xyz + p * g.n.w; }

// Cell-centred field at p, clamped to the domain.
fn samp_c(t: texture_3d<f32>, s: sampler, p: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
  let q = clamp(p, vec3<f32>(0.5), n - vec3<f32>(0.5));
  return textureSampleLevel(t, s, q / n, 0.0);
}

// Face velocity components at p. Each component is sampled on its own staggered lattice and
// clamped to the texels that hold valid faces for it.
fn u_at(t: texture_3d<f32>, s: sampler, p: vec3<f32>, n: vec3<f32>) -> f32 {
  let q = clamp(p + vec3<f32>(0.5, 0.0, 0.0), vec3<f32>(0.5), vec3<f32>(n.x + 0.5, n.y - 0.5, n.z - 0.5));
  return textureSampleLevel(t, s, q / (n + vec3<f32>(1.0)), 0.0).x;
}

fn v_at(t: texture_3d<f32>, s: sampler, p: vec3<f32>, n: vec3<f32>) -> f32 {
  let q = clamp(p + vec3<f32>(0.0, 0.5, 0.0), vec3<f32>(0.5), vec3<f32>(n.x - 0.5, n.y + 0.5, n.z - 0.5));
  return textureSampleLevel(t, s, q / (n + vec3<f32>(1.0)), 0.0).y;
}

fn w_at(t: texture_3d<f32>, s: sampler, p: vec3<f32>, n: vec3<f32>) -> f32 {
  let q = clamp(p + vec3<f32>(0.0, 0.0, 0.5), vec3<f32>(0.5), vec3<f32>(n.x - 0.5, n.y - 0.5, n.z + 0.5));
  return textureSampleLevel(t, s, q / (n + vec3<f32>(1.0)), 0.0).z;
}

fn vel_at(t: texture_3d<f32>, s: sampler, p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(u_at(t, s, p, n), v_at(t, s, p, n), w_at(t, s, p, n));
}

// Velocity at a cell centre from the six faces around it (exact average, no filtering).
fn vel_centre(t: texture_3d<f32>, c: vec3<i32>) -> vec3<f32> {
  let a = textureLoad(t, c, 0).xyz;
  let bx = textureLoad(t, c + vec3<i32>(1, 0, 0), 0).x;
  let by = textureLoad(t, c + vec3<i32>(0, 1, 0), 0).y;
  let bz = textureLoad(t, c + vec3<i32>(0, 0, 1), 0).z;
  return 0.5 * vec3<f32>(a.x + bx, a.y + by, a.z + bz);
}

fn luma(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)); }
