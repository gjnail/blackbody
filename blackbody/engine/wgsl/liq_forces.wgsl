// Body forces and boundary conditions on the grid velocity: gravity, source nozzles pushing their
// velocity onto the liquid, and solid faces (colliders, closed walls) held at the solid's own
// velocity, so a moving collider pushes the liquid. Solid faces are marked fixed. Wind drags the
// exposed surface along (form drag on cell-sized lumps of liquid, scaled by the surface setting),
// and with an open water level a layer along the open sides calms the flow toward rest, so waves
// run out into the open water instead of reflecting off the edge of the box.
//!include common.wgsl
//!include liq_common.wgsl
//!include noise.wgsl
//!include meshsdf.wgsl
//!include emitters.wgsl
//!include colliders.wgsl

struct Params {
  g: Grid,
  f: vec4<f32>,     // gravity (m/s^2, downward)
  w: vec4<f32>,     // wind (m/s, grid axes); w = drag rate per m/s of wind (1/m)
  lv: vec4<f32>,    // open water level (cells, < 0 off), calming layer width (cells), its rate (1/s), surface density
  ecnt: vec4<f32>,  // source count
  em: array<Emitter, MAX_EMITTERS>,
  ccnt: vec4<f32>,  // collider count, any moving (1/0)
  col: array<Collider, MAX_COLLIDERS>,
};

@group(0) @binding(0) var vold: texture_3d<f32>;
@group(0) @binding(1) var sdf: texture_3d<f32>;
@group(0) @binding(2) var atlas: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba32float, write>;
@group(0) @binding(4) var dens: texture_3d<f32>;
@group(1) @binding(0) var<uniform> U: Params;

fn solid(c: vec3<i32>, n: vec3<i32>) -> bool {
  if (!in_grid(c, n)) { return false; }
  return textureLoad(sdf, c, 0).x < 0.0;
}

fn wall(c: vec3<i32>, k: u32, n: vec3<i32>) -> bool {
  if (k == 1u) {
    return (c.y == 0 && U.g.bc.z < 0.5) || (c.y == n.y && U.g.bc.y < 0.5);
  }
  return (c[k] == 0 || c[k] == n[k]) && U.g.bc.x < 0.5;
}

// Velocity of the solid at world point w: that of the nearest collider (zero for walls).
fn solid_velocity(w: vec3<f32>) -> vec3<f32> {
  if (U.ccnt.y < 0.5) { return vec3<f32>(0.0); }
  var best = 1.0e9;
  var v = vec3<f32>(0.0);
  let cnt = i32(U.ccnt.x);
  for (var i = 0; i < cnt; i++) {
    let d = col_sdf(U.col[i], w);
    if (d < best) {
      best = d;
      v = col_velocity(U.col[i], w);
    }
  }
  return v;
}

fn dens_at(c: vec3<i32>, n: vec3<i32>) -> f32 {
  if (!in_grid(c, n)) { return 0.0; }
  return textureLoad(dens, c, 0).x;
}

// How much of face k at c is liquid with air next to it (0..1).
fn exposure(c: vec3<i32>, k: u32, n: vec3<i32>) -> f32 {
  var e = vec3<i32>(0);
  e[k] = 1;
  let ra = dens_at(c - e, n);
  let rb = dens_at(c, n);
  let thr = max(U.lv.w, 1e-3);
  let liquid = clamp(0.5 * (ra + rb) / thr - 0.5, 0.0, 1.0);
  if (liquid <= 0.0) { return 0.0; }
  let up = vec3<i32>(0, 1, 0);
  let air = min(min(ra, rb), min(dens_at(c - e + up, n), dens_at(c + up, n)));
  return liquid * clamp(1.5 - air / thr, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c > n)) { return; }
  let dt = U.g.bc.w;
  let h = U.g.n.w;
  let a = textureLoad(vold, c, 0);
  var v = a.xyz;
  var mask = u32(a.w);
  let cnt = i32(U.ecnt.x);
  for (var k = 0u; k < 3u; k++) {
    if (any(c > face_lim(k, n))) { continue; }
    var e = vec3<i32>(0);
    e[k] = 1;
    let wp = world_of(U.g, vec3<f32>(c) + face_off(k));
    if (wall(c, k, n)) {
      v[k] = 0.0;
      mask = (mask & ~(1u << k)) | (8u << k);
      continue;
    }
    if (solid(c - e, n) || solid(c, n)) {
      v[k] = solid_velocity(wp)[k];
      mask = (mask & ~(1u << k)) | (8u << k);
      continue;
    }
    if (k == 1u) { v[k] -= U.f.x * dt; }
    if (U.w.w > 0.0) {
      let ex = exposure(c, k, n);
      if (ex > 0.0) { v[k] += (U.w[k] - v[k]) * (1.0 - exp(-U.w.w * ex * dt)); }
    }
    if (U.lv.x >= 0.0 && U.g.bc.x > 0.5) {
      let p = vec3<f32>(c) + face_off(k);
      let ds = min(min(p.x, f32(n.x) - p.x), min(p.z, f32(n.z) - p.z));
      if (ds < U.lv.y && p.y < U.lv.x + 0.5 * U.lv.y) {
        let s = 1.0 - ds / U.lv.y;
        v[k] *= exp(-U.lv.z * s * s * dt);
      }
    }
    for (var s = 0; s < cnt; s++) {
      let em = U.em[s];
      if (em.d.w <= 0.0 || em.d.x <= 0.0) { continue; }
      let msk = emitter_mask(em, wp, h);
      if (msk <= 0.0) { continue; }
      let tv = emitter_velocity(em, wp);
      v[k] = mix(v[k], tv[k], clamp(msk * em.d.w, 0.0, 1.0));
      mask |= 1u << k;
    }
  }
  textureStore(dst, c, vec4<f32>(v, f32(mask)));
}
