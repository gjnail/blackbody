// Anisotropic surfacing, pass 2 (after Yu and Turk): per block, the covariance of the particles
// around it (its 3x3x3 neighbourhood of blocks, about 3 cells out). Its principal axes give each
// particle's kernel an ellipsoid shape, stretched along a thin sheet or jet (never thinner than
// round across it), so sheets and jets come out whole and smooth instead of holed and beaded. Few
// particles (spray) or an even spread (the bulk) stay round.
// out, per block: the matrix M (xx, xy, xz, yy, yz, zz) taking an offset to the round kernel's space,
// and the longest semi-axis (to size the splat), 8 floats.

const FX_M: f32 = 65536.0;

struct Params {
  n: vec4<f32>,   // simulation grid dims
  k: vec4<f32>,   // strength (0 round .. 1), longest axis allowed, shortest allowed, particles needed
};

@group(0) @binding(0) var<storage, read> mom: array<i32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(1) @binding(0) var<uniform> U: Params;

fn store_round(bi: u32) {
  let o = bi * 8u;
  dst[o] = 1.0; dst[o + 1u] = 0.0; dst[o + 2u] = 0.0;
  dst[o + 3u] = 1.0; dst[o + 4u] = 0.0; dst[o + 5u] = 1.0;
  dst[o + 6u] = 1.0; dst[o + 7u] = 0.0;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let nb = (vec3<i32>(U.n.xyz) + vec3<i32>(1)) / 2;
  let b = vec3<i32>(id);
  if (any(b >= nb)) { return; }
  let bi = u32(b.x + nb.x * (b.y + nb.y * b.z));
  var s0 = 0.0;
  var s1 = vec3<f32>(0.0);
  var sxx = 0.0; var sxy = 0.0; var sxz = 0.0; var syy = 0.0; var syz = 0.0; var szz = 0.0;
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let q = b + vec3<i32>(x, y, z);
        if (any(q < vec3<i32>(0)) || any(q >= nb)) { continue; }
        let base = u32(q.x + nb.x * (q.y + nb.y * q.z)) * 10u;
        let c = f32(mom[base]);
        if (c <= 0.0) { continue; }
        let m1 = vec3<f32>(f32(mom[base + 1u]), f32(mom[base + 2u]), f32(mom[base + 3u])) / FX_M;
        let d = vec3<f32>(f32(x), f32(y), f32(z)) * 2.0;   // that block's centre, from this one's (cells)
        // offsets re-centred on this block: o = o' + d
        s0 += c;
        s1 += m1 + c * d;
        sxx += f32(mom[base + 4u]) / FX_M + 2.0 * m1.x * d.x + c * d.x * d.x;
        sxy += f32(mom[base + 5u]) / FX_M + m1.x * d.y + m1.y * d.x + c * d.x * d.y;
        sxz += f32(mom[base + 6u]) / FX_M + m1.x * d.z + m1.z * d.x + c * d.x * d.z;
        syy += f32(mom[base + 7u]) / FX_M + 2.0 * m1.y * d.y + c * d.y * d.y;
        syz += f32(mom[base + 8u]) / FX_M + m1.y * d.z + m1.z * d.y + c * d.y * d.z;
        szz += f32(mom[base + 9u]) / FX_M + 2.0 * m1.z * d.z + c * d.z * d.z;
      }
    }
  }
  if (s0 < U.k.w) { store_round(bi); return; }
  let mu = s1 / s0;
  // covariance, then its eigenvectors by cyclic Jacobi rotations
  var a = mat3x3<f32>(
    vec3<f32>(sxx / s0 - mu.x * mu.x, sxy / s0 - mu.x * mu.y, sxz / s0 - mu.x * mu.z),
    vec3<f32>(sxy / s0 - mu.x * mu.y, syy / s0 - mu.y * mu.y, syz / s0 - mu.y * mu.z),
    vec3<f32>(sxz / s0 - mu.x * mu.z, syz / s0 - mu.y * mu.z, szz / s0 - mu.z * mu.z));
  var v = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
  var P = array<i32, 3>(0, 0, 1);
  var Q = array<i32, 3>(1, 2, 2);
  for (var sweep = 0; sweep < 6; sweep++) {
    for (var k = 0; k < 3; k++) {
      let p = P[k];
      let q = Q[k];
      let apq = a[q][p];
      if (abs(apq) < 1e-9) { continue; }
      let theta = 0.5 * atan2(2.0 * apq, a[q][q] - a[p][p]);
      let c = cos(theta);
      let s = sin(theta);
      // rotation in the (p, q) plane: J_pp = J_qq = c, J_pq = s, J_qp = -s (a[col][row])
      var rot = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
      rot[p][p] = c;
      rot[q][q] = c;
      rot[q][p] = s;
      rot[p][q] = -s;
      a = transpose(rot) * a * rot;
      v = v * rot;
    }
  }
  var lam = max(vec3<f32>(a[0][0], a[1][1], a[2][2]), vec3<f32>(1e-6));
  let lmin = min(lam.x, min(lam.y, lam.z));
  // stretched along the liquid's long directions, never thinner than round across it (a kernel
  // flattened below the particle spacing would let a one-particle sheet fall apart)
  var ax = clamp(sqrt(lam / lmin), vec3<f32>(1.0), vec3<f32>(4.0));
  ax = clamp(pow(ax, vec3<f32>(0.5 * U.k.x)), vec3<f32>(U.k.z), vec3<f32>(U.k.y));
  let inv = vec3<f32>(1.0) / ax;
  // M = V diag(1/ax) V^T
  let m = v * mat3x3<f32>(vec3<f32>(inv.x, 0.0, 0.0), vec3<f32>(0.0, inv.y, 0.0), vec3<f32>(0.0, 0.0, inv.z)) * transpose(v);
  let o = bi * 8u;
  dst[o] = m[0][0]; dst[o + 1u] = m[1][0]; dst[o + 2u] = m[2][0];
  dst[o + 3u] = m[1][1]; dst[o + 4u] = m[2][1]; dst[o + 5u] = m[2][2];
  dst[o + 6u] = max(ax.x, max(ax.y, ax.z)); dst[o + 7u] = 0.0;
}
