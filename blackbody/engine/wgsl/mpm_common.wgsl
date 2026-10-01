// Matter (engine/matter.py): sand, snow, mud, jelly and clay as MPM particles (MLS-MPM, Hu et al. 2018). Each
// particle carries its position, its velocity, the affine part of the velocity round it (APIC) and its elastic
// deformation gradient F. Every step the particles add their momentum and the push of their stress to the grid's
// nodes, the grid moves (gravity, the ground, the objects), and the particles pick up its motion again, deform with
// it, and give way where their material yields (sand slides, snow packs, mud and clay flow).
//
// Particle-to-grid sums are fixed-point integer atomics: integer sums do not depend on the order the threads add
// them in, so the result is the same on any GPU.

struct MParticle {
  x: vec4<f32>,   // position (grid units: node i is at i); w = material (0..15), negative: an empty slot
  v: vec4<f32>,   // velocity (m/s); w = plastic state: snow's Jp (its volume, packed), 1 as it falls
  c0: vec4<f32>,  // C, the affine velocity round it (APIC), by rows (1/s); w = a random number of its own (its look)
  c1: vec4<f32>,  // w = it is held still until this time (s of simulation), then let go
  c2: vec4<f32>,  // w = spare
  f0: vec4<f32>,  // F, the elastic deformation gradient, by rows; w = spare
  f1: vec4<f32>,
  f2: vec4<f32>,
};

// A material (matter.py MATTERS).
struct MMat {
  a: vec4<f32>,   // model (0 jelly, 1 sand, 2 snow, 3 mud, 4 clay), mu (Pa), lambda (Pa), density (kg/m^3)
  b: vec4<f32>,   // sand: Drucker-Prager alpha, cohesion (strain); snow: theta_c, theta_s, hardening xi;
                  // mud, clay: yield stress (Pa), share of the excess relaxed per second (mud), holds tension (1/0)
  c: vec4<f32>,   // friction against the ground and objects, largest hardening (snow), _, _
  d: vec4<f32>,   // look: albedo (linear rgb), roughness
  e: vec4<f32>,   // look: clear (jelly: light through it), sparkle, wrap (light into it: snow), colour variation
};

const MAX_MATS: u32 = 16u;
const FX_P: f32 = 16384.0;       // momentum fixed point (2^14): m/s times mass (in particles of 1000 kg/m^3)
const FX_MASS: f32 = 1048576.0;  // mass fixed point (2^20)
const NODE: u32 = 4u;            // accumulator slots per node: momentum x, y, z, mass

// Particle kernels run 64 threads per workgroup, dispatched in 2D past 65535 workgroups.
fn lin_id(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
  return gid.x + gid.y * nwg.x * 64u;
}

fn nidx(c: vec3<i32>, n: vec3<i32>) -> u32 {
  return u32(c.x + n.x * (c.y + n.y * c.z));
}

// A matrix from its rows (WGSL builds matrices from columns).
fn from_rows(a: vec4<f32>, b: vec4<f32>, c: vec4<f32>) -> mat3x3<f32> {
  return transpose(mat3x3<f32>(a.xyz, b.xyz, c.xyz));
}

fn eye3() -> mat3x3<f32> {
  return mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
}

fn diag3(s: vec3<f32>) -> mat3x3<f32> {
  return mat3x3<f32>(vec3<f32>(s.x, 0.0, 0.0), vec3<f32>(0.0, s.y, 0.0), vec3<f32>(0.0, 0.0, s.z));
}

// Quadratic B-spline weights of the three nodes base, base + 1, base + 2 along each axis, for a particle at
// fx = x - base (fx in [0.5, 1.5)).
fn bspline(fx: vec3<f32>) -> array<vec3<f32>, 3> {
  let a = vec3<f32>(1.5) - fx;
  let b = fx - vec3<f32>(1.0);
  let c = fx - vec3<f32>(0.5);
  return array<vec3<f32>, 3>(0.5 * a * a, vec3<f32>(0.75) - b * b, 0.5 * c * c);
}

// ---- the singular value decomposition of a 3x3 matrix --------------------------------------------------------------

struct SVD { u: mat3x3<f32>, s: vec3<f32>, v: mat3x3<f32> };

// One Jacobi rotation of the symmetric matrix A in the (p, q) plane, zeroing A[p][q]; V gathers the rotations.
fn jacobi(A: ptr<function, mat3x3<f32>>, V: ptr<function, mat3x3<f32>>, p: i32, q: i32) {
  let apq = (*A)[q][p];
  if (abs(apq) < 1.0e-24) { return; }
  let theta = ((*A)[q][q] - (*A)[p][p]) / (2.0 * apq);
  let t = select(1.0, -1.0, theta < 0.0) / (abs(theta) + sqrt(theta * theta + 1.0));
  let c = inverseSqrt(t * t + 1.0);
  let s = t * c;
  var J = eye3();
  J[p][p] = c;
  J[q][q] = c;
  J[q][p] = s;     // (row p, column q)
  J[p][q] = -s;    // (row q, column p)
  *A = transpose(J) * (*A) * J;
  *V = (*V) * J;
}

// F = U diag(s) V^T with U and V rotations, s largest first; the last one is negative if F turns inside out.
fn svd3(F: mat3x3<f32>) -> SVD {
  var A = transpose(F) * F;
  var V = eye3();
  for (var sweep = 0; sweep < 5; sweep++) {
    jacobi(&A, &V, 0, 1);
    jacobi(&A, &V, 0, 2);
    jacobi(&A, &V, 1, 2);
  }
  // the eigenvalues of F^T F, largest first, their eigenvectors with them
  var e = vec3<f32>(A[0][0], A[1][1], A[2][2]);
  if (e.x < e.y) { e = e.yxz; V = mat3x3<f32>(V[1], V[0], V[2]); }
  if (e.x < e.z) { e = e.zyx; V = mat3x3<f32>(V[2], V[1], V[0]); }
  if (e.y < e.z) { e = e.xzy; V = mat3x3<f32>(V[0], V[2], V[1]); }
  if (determinant(V) < 0.0) { V[2] = -V[2]; }
  let B = F * V;          // its columns are s_i u_i
  var u0 = B[0];
  let l0 = length(u0);
  u0 = select(vec3<f32>(1.0, 0.0, 0.0), u0 / l0, l0 > 1.0e-12);
  var u1 = B[1] - dot(B[1], u0) * u0;
  let l1 = length(u1);
  if (l1 > 1.0e-12) {
    u1 = u1 / l1;
  } else {
    u1 = normalize(select(cross(u0, vec3<f32>(1.0, 0.0, 0.0)), cross(u0, vec3<f32>(0.0, 1.0, 0.0)), abs(u0.x) > 0.9));
  }
  let u2 = cross(u0, u1);
  return SVD(mat3x3<f32>(u0, u1, u2), vec3<f32>(l0, dot(B[1], u1), dot(B[2], u2)), V);
}

// ---- the materials ------------------------------------------------------------------------------------------------

// Snow's hardening: packed snow (Jp < 1) is stiffer, loose snow softer (Stomakhin et al. 2013).
fn hardening(m: MMat, jp: f32) -> f32 {
  return clamp(exp(m.b.z * (1.0 - jp)), 0.2, max(m.c.y, 1.0));
}

// The Kirchhoff stress (tau = P F^T, Pa) of a particle of material m with elastic deformation F.
fn kirchhoff(m: MMat, F: mat3x3<f32>, jp: f32) -> mat3x3<f32> {
  let model = i32(m.a.x + 0.5);
  let mu = m.a.y;
  let la = m.a.z;
  if (model == 0) {
    // jelly: neo-Hookean
    let J = max(determinant(F), 1.0e-4);
    return mu * (F * transpose(F) - eye3()) + la * log(J) * eye3();
  }
  let d = svd3(F);
  if (model == 2) {
    // snow: fixed corotated, harder as it packs
    let h = hardening(m, jp);
    let R = d.u * transpose(d.v);
    let J = d.s.x * d.s.y * d.s.z;
    return 2.0 * mu * h * (F - R) * transpose(F) + la * h * (J - 1.0) * J * eye3();
  }
  // sand, mud, clay: St Venant-Kirchhoff in the logarithmic (Hencky) strain
  let eps = log(max(abs(d.s), vec3<f32>(1.0e-4)));
  let tr = eps.x + eps.y + eps.z;
  return d.u * diag3(2.0 * mu * eps + vec3<f32>(la * tr)) * transpose(d.u);
}

struct Yield { f: mat3x3<f32>, jp: f32 };

// Where the material gives way: F projected back onto what it can hold (sand slides past its friction angle, snow
// packs or breaks up, mud and clay flow past their yield stress). Returns the new F and plastic state.
fn plastic(m: MMat, F: mat3x3<f32>, jp: f32, dt: f32) -> Yield {
  let model = i32(m.a.x + 0.5);
  if (model == 0) { return Yield(F, jp); }
  let d = svd3(F);
  if (model == 2) {
    // snow: the stretch it holds is clamped; what is squeezed past it packs it (Jp)
    let s = clamp(d.s, vec3<f32>(1.0 - m.b.x), vec3<f32>(1.0 + m.b.y));
    let jn = clamp(jp * (d.s.x * d.s.y * d.s.z) / (s.x * s.y * s.z), 0.6, 20.0);
    return Yield(d.u * diag3(s) * transpose(d.v), jn);
  }
  let mu = m.a.y;
  let la = m.a.z;
  var eps = log(max(abs(d.s), vec3<f32>(1.0e-4)));
  if (model == 1) {
    // sand: Drucker-Prager (Klar et al. 2016); cohesion lets wet sand hold a little tension
    let c = m.b.y;
    var e = eps - vec3<f32>(c);
    let tr = e.x + e.y + e.z;
    if (tr >= 0.0) {
      e = vec3<f32>(0.0);       // pulled apart: it separates
    } else {
      let dev = e - vec3<f32>(tr / 3.0);
      let nd = length(dev);
      let dg = nd + (3.0 * la + 2.0 * mu) / (2.0 * mu) * tr * m.b.x;
      if (dg > 0.0 && nd > 1.0e-12) { e = e - dg * dev / nd; }
    }
    eps = e + vec3<f32>(c);
  } else {
    // mud, clay: von Mises in the deviatoric strain; mud relaxes toward it over time (viscoplastic)
    let tr = eps.x + eps.y + eps.z;
    let dev = eps - vec3<f32>(tr / 3.0);
    let nd = length(dev);
    let lim = sqrt(2.0 / 3.0) * m.b.x / (2.0 * mu);
    var dn = dev;
    if (nd > lim) {
      let keep = dev * (lim / nd);
      let share = select(1.0, 1.0 - exp(-m.b.y * dt), model == 3);
      dn = dev + (keep - dev) * share;
    }
    var trn = tr;
    if (m.b.z < 0.5 && tr > 0.0) { trn = 0.0; }      // (mud holds no tension: it tears)
    eps = dn + vec3<f32>(trn / 3.0);
  }
  return Yield(d.u * diag3(exp(eps)) * transpose(d.v), jp);
}
