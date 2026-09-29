// Liquid solver: shared definitions.
//
// The liquid is a set of particles (FLIP / APIC) that carry position, velocity and the affine part
// of the velocity field around them. Each substep they are splatted onto the MAC grid, the grid is
// made incompressible with a free-surface pressure solve, and the particles pick the new velocity
// back up and move through it.
//
// Particle-to-grid transfers add into fixed-point integer atomics. Integer sums do not depend on
// the order in which threads add them, so the simulation is deterministic on any GPU.

struct Particle {
  p: vec4<f32>,   // position (cells); w = 1 alive, 0 free slot
  v: vec4<f32>,   // velocity (m/s); w = age (s)
  cx: vec4<f32>,  // APIC affine velocity: gradient of u (m/s per cell)
  cy: vec4<f32>,  // gradient of v
  cz: vec4<f32>,  // gradient of w
};

const FX_M: f32 = 262144.0;     // momentum fixed-point scale (2^18)
const FX_W: f32 = 16777216.0;   // weight fixed-point scale (2^24)
const ACC: u32 = 8u;            // accumulator slots per grid node: u, wu, v, wv, w, ww, density, spare

// counters buffer
const C_COUNT: u32 = 0u;        // high-water mark of used particle slots
const C_FREE: u32 = 1u;         // entries on the free-slot stack

// Particle kernels run 64 threads per workgroup, dispatched in 2D once there are more than 65535
// workgroups (WebGPU's limit per dimension): this is the particle index of an invocation.
fn lin_id(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
  return gid.x + gid.y * nwg.x * 64u;
}

// Velocity texel .w holds bit flags per component k: bit k = valid (has fluid data),
// bit k + 3 = fixed (solid wall face, never overwritten).
fn flag_valid(m: u32, k: u32) -> bool { return ((m >> k) & 1u) != 0u; }
fn flag_fixed(m: u32, k: u32) -> bool { return ((m >> (k + 3u)) & 1u) != 0u; }

// Linear index of a node in a lattice of size m.
fn nidx(c: vec3<i32>, m: vec3<i32>) -> u32 {
  return u32(c.x + m.x * (c.y + m.y * c.z));
}

// Offset of component k's sample points from the cell corner: u faces sit at (i, j+.5, k+.5).
fn face_off(k: u32) -> vec3<f32> {
  var o = vec3<f32>(0.5);
  o[k] = 0.0;
  return o;
}

// Highest valid lattice index of component k (faces run to n on their own axis, n - 1 on the others).
fn face_lim(k: u32, n: vec3<i32>) -> vec3<i32> {
  var l = n - vec3<i32>(1);
  l[k] = n[k];
  return l;
}

// Trilinear interpolation of velocity component k at x (cells) from a MAC texture, with the
// gradient of the interpolant (for APIC). Returns (value, d/dx, d/dy, d/dz) per cell.
fn interp_comp(t: texture_3d<f32>, x: vec3<f32>, k: u32, n: vec3<i32>) -> vec4<f32> {
  let q = x - face_off(k);
  let fq = floor(q);
  let f = q - fq;
  let b = vec3<i32>(fq);
  let lim = face_lim(k, n);
  var val = 0.0;
  var grad = vec3<f32>(0.0);
  for (var o = 0; o < 8; o++) {
    let oo = vec3<i32>(o & 1, (o >> 1) & 1, o >> 2);
    let node = clamp(b + oo, vec3<i32>(0), lim);
    let fo = vec3<f32>(oo);
    let wv = mix(vec3<f32>(1.0) - f, f, fo);
    let s = fo * 2.0 - vec3<f32>(1.0);
    let a = textureLoad(t, node, 0)[k];
    val += wv.x * wv.y * wv.z * a;
    grad += vec3<f32>(s.x * wv.y * wv.z, wv.x * s.y * wv.z, wv.x * wv.y * s.z) * a;
  }
  return vec4<f32>(val, grad);
}

fn interp_val(t: texture_3d<f32>, x: vec3<f32>, k: u32, n: vec3<i32>) -> f32 {
  let q = x - face_off(k);
  let fq = floor(q);
  let f = q - fq;
  let b = vec3<i32>(fq);
  let lim = face_lim(k, n);
  var val = 0.0;
  for (var o = 0; o < 8; o++) {
    let oo = vec3<i32>(o & 1, (o >> 1) & 1, o >> 2);
    let node = clamp(b + oo, vec3<i32>(0), lim);
    let wv = mix(vec3<f32>(1.0) - f, f, vec3<f32>(oo));
    val += wv.x * wv.y * wv.z * textureLoad(t, node, 0)[k];
  }
  return val;
}

fn mac_vel(t: texture_3d<f32>, x: vec3<f32>, n: vec3<i32>) -> vec3<f32> {
  return vec3<f32>(interp_val(t, x, 0u, n), interp_val(t, x, 1u, n), interp_val(t, x, 2u, n));
}

// Cell-centred scalar (channel 0) at x with its gradient, clamped to the grid.
fn interp_cell(t: texture_3d<f32>, x: vec3<f32>, n: vec3<i32>) -> vec4<f32> {
  let q = x - vec3<f32>(0.5);
  let fq = floor(q);
  let f = q - fq;
  let b = vec3<i32>(fq);
  let lim = n - vec3<i32>(1);
  var val = 0.0;
  var grad = vec3<f32>(0.0);
  for (var o = 0; o < 8; o++) {
    let oo = vec3<i32>(o & 1, (o >> 1) & 1, o >> 2);
    let node = clamp(b + oo, vec3<i32>(0), lim);
    let fo = vec3<f32>(oo);
    let wv = mix(vec3<f32>(1.0) - f, f, fo);
    let s = fo * 2.0 - vec3<f32>(1.0);
    let a = textureLoad(t, node, 0).x;
    val += wv.x * wv.y * wv.z * a;
    grad += vec3<f32>(s.x * wv.y * wv.z, wv.x * s.y * wv.z, wv.x * wv.y * s.z) * a;
  }
  return vec4<f32>(val, grad);
}
