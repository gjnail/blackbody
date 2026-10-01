// Body forces and boundary conditions on the grid velocity: gravity, source nozzles pushing their
// velocity onto the liquid, and solid faces (colliders, closed walls) held at the solid's own
// velocity, so a moving collider pushes the liquid. Solid faces are marked fixed. Wind drags the
// exposed surface along (form drag on cell-sized lumps of liquid, scaled by the surface setting),
// and with an open water level a layer along the open sides calms the flow toward rest, so waves
// run out into the open water instead of reflecting off the edge of the box. (With a current the
// open water, and so that layer, flows.) Liquids of different densities push each other up or down
// (liq_attr_buoy.wgsl).
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
  lv: vec4<f32>,    // open water on (>= 0), calming layer width (cells), its rate (1/s), surface density
  sea: vec4<f32>,   // the sea's peak wavenumber (1/m), current (m/s, grid x and z), buoyancy between liquids on (1/0)
  sides: vec4<f32>, // the open sides the open water flows through (-x, +x, -z, +z; 1); the rest are walls under the
                    // level (0), or all the way up (-1 a wave flume's walls along the waves, -2 past the shore)
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
@group(0) @binding(5) var ocn_t: texture_2d<f32>;
@group(0) @binding(6) var buoy: texture_3d<f32>;   // upward share of gravity per cell (two liquids)
//!include liq_level.wgsl
@group(1) @binding(0) var<uniform> U: Params;

fn solid(c: vec3<i32>, n: vec3<i32>) -> bool {
  if (!in_grid(c, n)) { return false; }
  return textureLoad(sdf, c, 0).x < 0.0;
}

fn wall(c: vec3<i32>, k: u32, n: vec3<i32>) -> bool {
  if (k == 1u) {
    return (c.y == 0 && U.g.bc.z < 0.5) || (c.y == n.y && U.g.bc.y < 0.5);
  }
  // an open side under the open water level holds the liquid in (the water past it is still), and
  // with the sea let in only where its waves come from, the other sides are walls all the way up
  let under = U.lv.x >= 0.0 && f32(c.y) + 0.5 < level_at(vec2<f32>(c.xz) + vec2<f32>(0.5));
  let side = select(select(3u, 2u, c[k] == 0), select(1u, 0u, c[k] == 0), k == 0u);
  let flume = U.lv.x >= 0.0 && U.sides[side] < -0.5;
  return (c[k] == 0 || c[k] == n[k]) && (U.g.bc.x < 0.5 || under || flume);
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

// Active absorption at a side the sea flows through (as in a numerical wave tank): a few cells in
// from the side, at column q, the box's water stands higher or lower than the sea's own surface
// there, and a long wave carries a flow of sqrt(g / depth) per metre it stands above the level, so
// the side lets that much more water in (or out). With the sea's orbital flow alone the side was a
// wavemaker that reflected everything coming back at it: a wave off a beach bounced to and fro in
// the box, and its water drifted well away from the sea's level. Returns m/s, inward positive.
fn absorb_flow(q: vec3<i32>, n: vec3<i32>, h: f32) -> f32 {
  let sea_top = level_at(vec2<f32>(q.xz) + vec2<f32>(0.5));   // the sea's surface (cells)
  var bed = 0;
  while (bed < n.y && solid(vec3<i32>(q.x, bed, q.z), n)) { bed++; }
  if (bed >= n.y - 1) { return 0.0; }
  // the box's surface: down from the top to the first of two liquid cells one over the other (a
  // drop of spray is not the surface)
  var top = -1.0;
  for (var y = n.y - 2; y >= bed; y--) {
    let d0 = dens_at(vec3<i32>(q.x, y, q.z), n);
    if (d0 >= U.lv.w && dens_at(vec3<i32>(q.x, max(y - 1, bed), q.z), n) >= U.lv.w) {
      top = f32(y) + clamp(d0 / max(2.5 * U.lv.w, 1e-3), 0.0, 1.0);
      break;
    }
  }
  if (top < 0.0) { top = f32(bed); }
  let depth = max(U.lv.x - f32(bed), 2.0);   // still-water depth (cells)
  return clamp(sqrt(U.f.x * h / depth) * (sea_top - top), -2.0, 2.0);
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
      // under the open water the sea flows in and out through the open sides: its waves' orbital
      // motion and the current (not through a flume's walls)
      let side = select(2 * i32(k), 2 * i32(k) + 1, c[k] == n[k]) - select(0, 2, k == 2u);
      if (k != 1u && U.lv.x >= 0.0 && U.g.bc.x > 0.5 && U.sides[min(side, 3)] > 0.5) {
        v[k] = sea_velocity(vec3<f32>(c) + face_off(k), U.sea.x, h)[k] + vec3<f32>(U.sea.y, 0.0, U.sea.z)[k];
        // and the water that evens out the box's level with the sea's (absorb_flow), measured three
        // cells in
        var q = c;
        let lo = c[k] == 0;
        q[k] = select(n[k] - 4, 3, lo);
        q = clamp(q, vec3<i32>(0), n - vec3<i32>(1));
        v[k] += select(-1.0, 1.0, lo) * absorb_flow(q, n, h);
      }
      mask = (mask & ~(1u << k)) | (8u << k);
      continue;
    }
    if (solid(c - e, n) || solid(c, n)) {
      v[k] = solid_velocity(wp)[k];
      mask = (mask & ~(1u << k)) | (8u << k);
      continue;
    }
    // the calming layer along the open sides draws the liquid toward the open water's own motion
    // (still water, or the sea's orbital velocity), acting on the motion the liquid has before
    // gravity's pull this step (which the pressure balances exactly in still water: blending it
    // unevenly would leave a flow for the pressure solve to make)
    if (U.lv.x >= 0.0 && U.g.bc.x > 0.5) {
      let p = vec3<f32>(c) + face_off(k);
      // (distance to the nearest side the open water flows through)
      let big = 1.0e6;
      let ds = min(min(select(big, p.x, U.sides.x > 0.5), select(big, f32(n.x) - p.x, U.sides.y > 0.5)),
                   min(select(big, p.z, U.sides.z > 0.5), select(big, f32(n.z) - p.z, U.sides.w > 0.5)));
      if (ds < U.lv.y && p.y < level_at(p.xz) + 0.5 * U.lv.y) {
        let s = 1.0 - ds / U.lv.y;
        var goal = sea_velocity(p, U.sea.x, h)[k] + vec3<f32>(U.sea.y, 0.0, U.sea.z)[k];
        // and, across the side it is nearest, the flow that evens out the box's level with the sea's
        // (absorb_flow): drawn toward the sea's orbital motion alone, the layer held back the water
        // a long swell brings in and takes out, and the box's level lagged far behind the sea's
        let dxl = select(big, p.x, U.sides.x > 0.5);
        let dxh = select(big, f32(n.x) - p.x, U.sides.y > 0.5);
        let dzl = select(big, p.z, U.sides.z > 0.5);
        let dzh = select(big, f32(n.z) - p.z, U.sides.w > 0.5);
        let ax = select(2u, 0u, min(dxl, dxh) <= min(dzl, dzh));
        if (k == ax) {
          let inward = select(select(-1.0, 1.0, dzl <= dzh), select(-1.0, 1.0, dxl <= dxh), ax == 0u);
          goal += inward * absorb_flow(clamp(c, vec3<i32>(0), n - vec3<i32>(1)), n, h);
        }
        v[k] = goal + (v[k] - goal) * exp(-U.lv.z * s * s * dt);
      }
    }
    if (k == 1u) {
      v[k] -= U.f.x * dt;
      if (U.sea.w > 0.5) {
        let da = dens_at(c - e, n);
        let db = dens_at(c, n);
        if (da + db > 1e-4) {
          let ba = select(0.0, textureLoad(buoy, clamp(c - e, vec3<i32>(0), n - vec3<i32>(1)), 0).x, da > 1e-4);
          let bb = select(0.0, textureLoad(buoy, clamp(c, vec3<i32>(0), n - vec3<i32>(1)), 0).x, db > 1e-4);
          v[k] += U.f.x * dt * (ba * da + bb * db) / (da + db);
        }
      }
    }
    if (U.w.w > 0.0) {
      let ex = exposure(c, k, n);
      if (ex > 0.0) { v[k] += (U.w[k] - v[k]) * (1.0 - exp(-U.w.w * ex * dt)); }
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
