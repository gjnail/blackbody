// Pressure system for the free surface. A cell is liquid when its particle density is above the
// surface threshold. The Poisson matrix A (symmetric positive definite) has, per liquid cell, one
// coefficient per open face: 1 toward a liquid neighbour, 1/theta toward air (ghost fluid: the
// surface sits a fraction theta of the way to the air cell, found from the density, and p = 0
// there, plus the surface-tension jump sigma * kappa), none toward solids and closed walls. Right-hand side b = -h^2 (div u - c), where c is a
// small expansion wherever particles have bunched up clearly denser than at rest, which keeps the
// liquid's volume from drifting. Writes cell types, the matrix (diagonal, liquid-neighbour mask)
// and the starting residual r = b - A x for the warm-started pressure x.
//!include common.wgsl
//!include liq_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,   // surface density (particles), rest density (particles), volume correction (per step), minimum theta
  k2: vec4<f32>,  // density ratio above which the volume correction acts, surface tension sigma / rho * dt (m^3/s)
  k3: vec4<f32>,  // open water level (cells, < 0 off), gravity * dt (m/s)
};

@group(0) @binding(0) var vel: texture_3d<f32>;
@group(0) @binding(1) var dens: texture_3d<f32>;
@group(0) @binding(2) var sdf: texture_3d<f32>;
@group(0) @binding(3) var X: texture_3d<f32>;
@group(0) @binding(7) var kappa: texture_3d<f32>;
@group(0) @binding(4) var out_type: texture_storage_3d<r32float, write>;
@group(0) @binding(5) var out_co: texture_storage_3d<rg32float, write>;
@group(0) @binding(6) var out_r: texture_storage_3d<r32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn dir6(i: i32) -> vec3<i32> {
  var d = vec3<i32>(0);
  d[i >> 1] = select(-1, 1, (i & 1) == 1);
  return d;
}

// Pressure (scaled like x) of the open water outside the sides at the height of cell c: at rest
// under the level, so the liquid in the box meets still water there instead of air.
fn level_x(c: vec3<i32>) -> f32 {
  let y = f32(c.y) + 0.5;
  if (U.k3.x < 0.0 || U.g.bc.x < 0.5 || y >= U.k3.x) { return -1.0; }
  return U.k3.y * (U.k3.x - y) * U.g.n.w;
}

fn outside_open(q: vec3<i32>, n: vec3<i32>) -> bool {
  if (q.y < 0) { return U.g.bc.z > 0.5; }
  if (q.y >= n.y) { return U.g.bc.y > 0.5; }
  return U.g.bc.x > 0.5;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let rho_t = U.k.x;
  let rho0 = U.k.y;
  let rho = textureLoad(dens, c, 0).x;
  let sol = textureLoad(sdf, c, 0).x < 0.0;
  if (sol || rho < rho_t) {
    textureStore(out_type, c, vec4<f32>(select(0.0, 2.0, sol), 0.0, 0.0, 0.0));
    textureStore(out_co, c, vec4<f32>(0.0));
    textureStore(out_r, c, vec4<f32>(0.0));
    return;
  }
  let phi_c = (rho_t - rho) / rho0;
  var diag = 0.0;
  var mask = 0u;
  var ax_nb = 0.0;
  var jump = 0.0;
  let xg = U.k2.y * textureLoad(kappa, c, 0).x;  // pressure at the surface: sigma kappa (scaled)
  for (var i = 0; i < 6; i++) {
    let q = c + dir6(i);
    if (!in_grid(q, n)) {
      if (outside_open(q, n)) {
        diag += 2.0;  // p = 0 on the open boundary face (or the open water's pressure there)
        if (q.y >= 0 && q.y < n.y) {
          let xl = level_x(c);
          if (xl >= 0.0) { jump += 2.0 * xl; }
        }
      }
      continue;
    }
    if (textureLoad(sdf, q, 0).x < 0.0) { continue; }
    let rq = textureLoad(dens, q, 0).x;
    if (rq >= rho_t) {
      diag += 1.0;
      mask |= 1u << u32(i);
      ax_nb += textureLoad(X, q, 0).x;
    } else {
      let phi_q = (rho_t - rq) / rho0;
      let theta = clamp(phi_c / min(phi_c - phi_q, -1e-6), U.k.w, 1.0);
      diag += 1.0 / theta;
      jump += xg / theta;
    }
  }
  let h = U.g.n.w;
  let a = textureLoad(vel, c, 0).xyz;
  let ux = textureLoad(vel, c + vec3<i32>(1, 0, 0), 0).x;
  let vy = textureLoad(vel, c + vec3<i32>(0, 1, 0), 0).y;
  let wz = textureLoad(vel, c + vec3<i32>(0, 0, 1), 0).z;
  let div = (ux - a.x + vy - a.y + wz - a.z) / h;
  let dt = max(U.g.bc.w, 1e-6);
  let corr = min(U.k.z * max(rho / rho0 - U.k2.x, 0.0) / dt, 0.5 / dt);
  let b = -h * h * (div - corr) + jump;
  let r = b - (diag * textureLoad(X, c, 0).x - ax_nb);
  textureStore(out_type, c, vec4<f32>(1.0, 0.0, 0.0, 0.0));
  textureStore(out_co, c, vec4<f32>(diag, f32(mask), 0.0, 0.0));
  textureStore(out_r, c, vec4<f32>(r, 0.0, 0.0, 0.0));
}
