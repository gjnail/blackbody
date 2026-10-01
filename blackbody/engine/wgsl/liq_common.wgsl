// Liquid solver: shared definitions.
//
// The liquid is a set of particles (FLIP / APIC) that carry position, velocity and the affine part
// of the velocity field around them. Each substep they are splatted onto the MAC grid, the grid is
// made incompressible with a free-surface pressure solve, and the particles pick the new velocity
// back up and move through it.
//
// Particle-to-grid transfers add into fixed-point integer atomics. Integer sums do not depend on
// the order in which threads add them, so the simulation is deterministic on any GPU.

// 48 bytes. The APIC affine velocity C (rows: the gradients of u, v and w, in m/s per cell) is kept
// as nine half floats: c0 holds C00 and C01, c the rest in row order; the last half of c.w is the
// particle's heat (1 molten as poured .. 0 cooled to the air), for liquids that cool and set.
struct Particle {
  p: vec4<f32>,   // position (cells); w = age (s), negative for a free slot
  v: vec3<f32>,   // velocity (m/s)
  c0: u32,
  c: vec4<u32>,
};

fn alive(P: Particle) -> bool { return P.p.w >= 0.0; }

fn apic_cx(P: Particle) -> vec3<f32> { return vec3<f32>(unpack2x16float(P.c0), unpack2x16float(P.c.x).x); }
fn apic_cy(P: Particle) -> vec3<f32> { return vec3<f32>(unpack2x16float(P.c.x).y, unpack2x16float(P.c.y)); }
fn apic_cz(P: Particle) -> vec3<f32> { return vec3<f32>(unpack2x16float(P.c.z), unpack2x16float(P.c.w).x); }

fn apic_set(P: ptr<function, Particle>, cx_in: vec3<f32>, cy_in: vec3<f32>, cz_in: vec3<f32>) {
  let lim = vec3<f32>(6.0e4);
  let cx = clamp(cx_in, -lim, lim);
  let cy = clamp(cy_in, -lim, lim);
  let cz = clamp(cz_in, -lim, lim);
  let heat = unpack2x16float((*P).c.w).y;
  (*P).c0 = pack2x16float(cx.xy);
  (*P).c = vec4<u32>(pack2x16float(vec2<f32>(cx.z, cy.x)), pack2x16float(cy.yz), pack2x16float(cz.xy),
                     pack2x16float(vec2<f32>(cz.z, heat)));
}

fn heat_of(P: Particle) -> f32 { return unpack2x16float(P.c.w).y; }

fn set_heat(P: ptr<function, Particle>, heat: f32) {
  (*P).c.w = pack2x16float(vec2<f32>(unpack2x16float((*P).c.w).x, heat));
}

// A new particle, as hot as its source pours it (1).
fn new_particle(x: vec3<f32>, v: vec3<f32>) -> Particle {
  return Particle(vec4<f32>(x, 0.0), v, 0u, vec4<u32>(0u, 0u, 0u, pack2x16float(vec2<f32>(0.0, 1.0))));
}

// What a particle carries besides its motion, in a buffer parallel to the particles (only when a
// source pours a dye or a liquid of another density): x, y = its dye as half floats (absorption per
// metre rgb, scattering per metre), z = its density relative to the liquid, minus 1 (f32 bits), so
// all zeros is clear liquid of the liquid's own density.
fn attr_dye(a: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(a.x), unpack2x16float(a.y)); }
fn attr_rho(a: vec4<u32>) -> f32 { return 1.0 + bitcast<f32>(a.z); }
fn make_attr(dye: vec4<f32>, rho: f32) -> vec4<u32> {
  let d = clamp(dye, vec4<f32>(0.0), vec4<f32>(6.0e4));
  return vec4<u32>(pack2x16float(d.xy), pack2x16float(d.zw), bitcast<u32>(rho - 1.0), 0u);
}
const ATTR_SLOTS: u32 = 5u;     // per cell: sum w (density ratio - 1), sum w dye (rgba)
const FX_A: f32 = 65536.0;      // their fixed-point scale (2^16)

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

// A wave flume's walls: q, a cell just outside the grid, lies past a side whose bit (-x 1, +x 2,
// -z 4, +z 8) is set in mask: a wall all the way up, not open above the open water's level.
fn flume_wall(q: vec3<i32>, n: vec3<i32>, mask: f32) -> bool {
  let m = u32(mask + 0.5);
  if (m == 0u) { return false; }
  if (q.x < 0) { return (m & 1u) != 0u; }
  if (q.x >= n.x) { return (m & 2u) != 0u; }
  if (q.z < 0) { return (m & 4u) != 0u; }
  if (q.z >= n.z) { return (m & 8u) != 0u; }
  return false;
}
