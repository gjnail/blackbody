// Subtract the pressure gradient. Between two liquid cells it is the plain difference; between
// liquid and air the air side is a ghost value extrapolated to the surface pressure (zero, plus
// the surface-tension jump) a fraction theta of the way across. Afterwards a face is valid if it touches liquid or carried particle
// data before the solve (spray in the air keeps its own velocity); the rest is extrapolated.
//!include common.wgsl
//!include liq_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,   // surface density (particles), rest density (particles), _, minimum theta
  k2: vec4<f32>,  // _, surface tension sigma / rho * dt (m^3/s)
  k3: vec4<f32>,  // open water level (cells, < 0 off), gravity * dt (m/s)
};

@group(0) @binding(0) var vel: texture_3d<f32>;
@group(0) @binding(1) var vold: texture_3d<f32>;
@group(0) @binding(2) var X: texture_3d<f32>;
@group(0) @binding(3) var T: texture_3d<f32>;
@group(0) @binding(4) var dens: texture_3d<f32>;
@group(0) @binding(5) var dst: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var kappa: texture_3d<f32>;
@group(1) @binding(0) var<uniform> U: Params;

// 0 air, 1 liquid, 2 solid; outside the grid: air if that boundary is open, else solid
fn ctype(c: vec3<i32>, n: vec3<i32>) -> i32 {
  if (!in_grid(c, n)) {
    var open = U.g.bc.x;
    if (c.y < 0) { open = U.g.bc.z; }
    if (c.y >= n.y) { open = U.g.bc.y; }
    return select(2, 0, open > 0.5);
  }
  return i32(textureLoad(T, c, 0).x + 0.5);
}

fn phi(c: vec3<i32>, n: vec3<i32>) -> f32 {
  if (!in_grid(c, n)) { return 1.0e6; }
  return (U.k.x - textureLoad(dens, c, 0).x) / U.k.y;
}

// fraction of the way from liquid cell l to air cell a where the surface lies
fn theta(l: vec3<i32>, a: vec3<i32>, n: vec3<i32>) -> f32 {
  if (!in_grid(a, n)) { return 0.5; }
  let pl = phi(l, n);
  let pa = phi(a, n);
  return clamp(pl / min(pl - pa, -1e-6), U.k.w, 1.0);
}

fn xp(c: vec3<i32>) -> f32 { return textureLoad(X, c, 0).x; }

// surface pressure next to liquid cell c (scaled like x)
fn xg(c: vec3<i32>) -> f32 { return U.k2.y * textureLoad(kappa, c, 0).x; }

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c > n)) { return; }
  let ih = 1.0 / U.g.n.w;
  let a4 = textureLoad(vel, c, 0);
  let mask = u32(a4.w);
  let pm = u32(textureLoad(vold, c, 0).w);
  var v = a4.xyz;
  var nm = 0u;
  for (var k = 0u; k < 3u; k++) {
    if (flag_fixed(mask, k)) {
      nm |= 8u << k;
      continue;
    }
    if (any(c > face_lim(k, n))) { continue; }
    var e = vec3<i32>(0);
    e[k] = 1;
    let ca = c - e;
    let ta = ctype(ca, n);
    let tb = ctype(c, n);
    // open side under the water level: the still water's pressure half a cell out
    let yc = f32(c.y) + 0.5;
    if (k != 1u && U.k3.x >= 0.0 && U.g.bc.x > 0.5 && yc < U.k3.x && (c[k] == 0 || c[k] == n[k])) {
      let xl = U.k3.y * (U.k3.x - yc) * U.g.n.w;
      if (c[k] == 0 && tb == 1) {
        v[k] -= ((xp(c) - xl) / 0.5) * ih;
        nm |= 1u << k;
        continue;
      }
      if (c[k] == n[k] && ta == 1) {
        v[k] -= ((xl - xp(ca)) / 0.5) * ih;
        nm |= 1u << k;
        continue;
      }
    }
    if (ta == 1 && tb == 1) {
      v[k] -= (xp(c) - xp(ca)) * ih;
      nm |= 1u << k;
    } else if (ta == 1 && tb == 0) {
      v[k] -= ((xg(ca) - xp(ca)) / theta(ca, c, n)) * ih;
      nm |= 1u << k;
    } else if (ta == 0 && tb == 1) {
      v[k] -= ((xp(c) - xg(c)) / theta(c, ca, n)) * ih;
      nm |= 1u << k;
    } else if (flag_valid(pm, k)) {
      nm |= 1u << k;
    }
  }
  textureStore(dst, c, vec4<f32>(v, f32(nm)));
}
