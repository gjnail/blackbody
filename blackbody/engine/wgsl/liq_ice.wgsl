// Ice bodies: frozen liquid moves as rigid pieces.
//
// The pressure solve moves the ice as if it were liquid (so the water pushes on it and it pushes on the
// water, and its lighter weight floats it, liq_therm_buoy.wgsl); then the velocity inside each piece of
// ice is replaced by the rigid motion closest to it (after Carlson et al. 2004, "Rigid fluid"): the
// momentum and angular momentum of the piece, spread over it as one translation and one rotation. The
// frozen particles then move with that rigid motion exactly (liq_ice_parts.wgsl), so ice keeps its
// shape: it falls, floats, tips over, drifts and rolls as a solid. A piece frozen onto a surface below
// the freezing point (the ground, a collider, a closed wall) is held there.
//
// Pieces are the connected components of frozen cells (a cell is frozen where most of its liquid is),
// found each step by union-find on the grid (Playne and Hawick 2018: every frozen cell links to its
// frozen neighbours with atomic minimums, then points straight at its root). Each piece's mass, centre,
// momentum, angular momentum and inertia are summed from its cells in 64-bit fixed point (two 32-bit
// words with the carry), so the sums are exact whatever order the threads add in.
//
// Entry points, in order: label_init, label_union, label_flatten, label_roots, label_assign, body_sums
// (all over cells), body_solve (over bodies), body_faces (over the velocity's faces).
//!include common.wgsl
//!include liq_common.wgsl
//!include meshsdf.wgsl
//!include colliders.wgsl
//!include liq_therm_common.wgsl

struct Params {
  g: Grid,
  th: Therm,
  k: vec4<f32>,     // share of a cell frozen for it to be ice, _, _, _
  ccnt: vec4<f32>,  // collider count
  col: array<Collider, MAX_COLLIDERS>,
  ctemp: array<vec4<f32>, 4>,   // collider temperatures (C)
};

@group(0) @binding(0) var therm_t: texture_3d<f32>;
@group(0) @binding(1) var sdf: texture_3d<f32>;
@group(0) @binding(2) var atlas: texture_3d<f32>;
@group(0) @binding(3) var vin: texture_3d<f32>;
@group(0) @binding(4) var vout: texture_storage_3d<rgba32float, write>;
@group(0) @binding(5) var<storage, read_write> lab: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> bodycell: array<u32>;
@group(0) @binding(7) var<storage, read_write> sums: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> bodies: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> U: Params;

const COUNTER: u32 = MAX_BODIES * BODY_WORDS;

fn is_ice(c: vec3<i32>, n: vec3<i32>) -> bool {
  if (!in_grid(c, n)) { return false; }
  let t = textureLoad(therm_t, c, 0);
  return t.z >= U.k.x && t.w > 0.02;
}

fn cell_of(i: u32, n: vec3<i32>) -> vec3<i32> {
  let nx = u32(n.x);
  let ny = u32(n.y);
  return vec3<i32>(i32(i % nx), i32((i / nx) % ny), i32(i / (nx * ny)));
}

fn find(x_in: u32) -> u32 {
  var x = x_in;
  loop {
    let p = atomicLoad(&lab[x]);
    if (p == x) { return x; }
    x = p;
  }
  return x;
}

fn union_cells(a_in: u32, b_in: u32) {
  var a = a_in;
  var b = b_in;
  loop {
    a = find(a);
    b = find(b);
    if (a == b) { return; }
    if (a < b) {
      let t = a;
      a = b;
      b = t;
    }
    // hang the larger root under the smaller; if another thread got there first, carry on from where
    // it hung it
    let old = atomicMin(&lab[a], b);
    if (old == a) { return; }
    a = old;
  }
}

@compute @workgroup_size(8, 8, 4)
fn label_init(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let i = nidx(c, n);
  atomicStore(&lab[i], select(NO_BODY, i, is_ice(c, n)));
  bodycell[i] = NO_BODY;
}

@compute @workgroup_size(8, 8, 4)
fn label_union(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n) || !is_ice(c, n)) { return; }
  let i = nidx(c, n);
  for (var a = 0; a < 3; a++) {
    var e = vec3<i32>(0);
    e[a] = 1;
    let q = c + e;
    if (is_ice(q, n)) { union_cells(i, nidx(q, n)); }
  }
}

@compute @workgroup_size(8, 8, 4)
fn label_flatten(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n) || !is_ice(c, n)) { return; }
  let i = nidx(c, n);
  atomicStore(&lab[i], find(i));
}

@compute @workgroup_size(8, 8, 4)
fn label_roots(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n) || !is_ice(c, n)) { return; }
  let i = nidx(c, n);
  if (atomicLoad(&lab[i]) != i) { return; }
  let b = atomicAdd(&sums[COUNTER], 1u);
  if (b < MAX_BODIES) {
    bodycell[i] = b;
    atomicStore(&sums[b * BODY_WORDS + 33u], i);
  }
}

@compute @workgroup_size(8, 8, 4)
fn label_assign(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n) || !is_ice(c, n)) { return; }
  let i = nidx(c, n);
  let r = atomicLoad(&lab[i]);
  if (r != i) { bodycell[i] = bodycell[r]; }
}

fn add64(slot: u32, v: i32) {
  let lo = bitcast<u32>(v);
  let hi = select(0u, 0xFFFFFFFFu, v < 0);
  let old = atomicAdd(&sums[slot], lo);
  let carry = select(0u, 1u, old + lo < old);
  atomicAdd(&sums[slot + 1u], hi + carry);
}

fn read64(slot: u32) -> f32 {
  let lo = atomicLoad(&sums[slot]);
  let hi = bitcast<i32>(atomicLoad(&sums[slot + 1u]));
  return f32(hi) * 4294967296.0 + f32(lo);
}

fn collider_temp(k: u32) -> f32 { return U.ctemp[k / 4u][k % 4u]; }

fn solid_temp(w: vec3<f32>) -> f32 {
  var best = U.g.n.w;
  var T = U.th.b.z;
  let cnt = u32(U.ccnt.x);
  for (var k = 0u; k < cnt; k++) {
    let d = col_sdf(U.col[k], w);
    if (d < best) {
      best = d;
      T = collider_temp(k);
    }
  }
  return T;
}

// A frozen cell against a surface at or below freezing is frozen onto it.
fn frozen_on(c: vec3<i32>, n: vec3<i32>) -> bool {
  let h = U.g.n.w;
  let tf = t_freeze(U.th) + 0.5;
  let wp = world_of(U.g, vec3<f32>(c) + vec3<f32>(0.5));
  for (var f = 0; f < 6; f++) {
    var e = vec3<i32>(0);
    e[f / 2] = select(-1, 1, (f & 1) == 1);
    let q = c + e;
    if (!in_grid(q, n)) {
      var wall = false;
      if (e.y < 0) { wall = U.g.bc.z < 0.5; }
      else if (e.y > 0) { wall = U.g.bc.y < 0.5; }
      else { wall = U.g.bc.x < 0.5; }
      if (wall && U.th.b.z <= tf) { return true; }
    } else if (textureLoad(sdf, q, 0).x < 0.0) {
      if (solid_temp(wp + vec3<f32>(e) * (0.5 * h)) <= tf) { return true; }
    }
  }
  return false;
}

@compute @workgroup_size(8, 8, 4)
fn body_sums(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let i = nidx(c, n);
  let b = bodycell[i];
  if (b == NO_BODY) { return; }
  let base = b * BODY_WORDS;
  let t = textureLoad(therm_t, c, 0);
  let m = clamp(t.w * t.z, 0.0, 1.5);
  let rc = cell_of(atomicLoad(&sums[base + 33u]), n);
  let r = c - rc;
  let mf = i32(round(m * FX_BM));
  let v = clamp(vel_centre(vin, c), vec3<f32>(-20.0), vec3<f32>(20.0));
  add64(base + 0u, mf);
  add64(base + 2u, mf * r.x);
  add64(base + 4u, mf * r.y);
  add64(base + 6u, mf * r.z);
  add64(base + 8u, i32(round(m * v.x * FX_BV)));
  add64(base + 10u, i32(round(m * v.y * FX_BV)));
  add64(base + 12u, i32(round(m * v.z * FX_BV)));
  add64(base + 14u, mf * r.x * r.x);
  add64(base + 16u, mf * r.y * r.y);
  add64(base + 18u, mf * r.z * r.z);
  add64(base + 20u, mf * r.x * r.y);
  add64(base + 22u, mf * r.x * r.z);
  add64(base + 24u, mf * r.y * r.z);
  let rv = cross(vec3<f32>(r), v) * m;
  add64(base + 26u, i32(round(rv.x * FX_BW)));
  add64(base + 28u, i32(round(rv.y * FX_BW)));
  add64(base + 30u, i32(round(rv.z * FX_BW)));
  if (frozen_on(c, n)) { atomicOr(&sums[base + 32u], 1u); }
}

fn inverse3(m: mat3x3<f32>) -> mat3x3<f32> {
  let a = m[0];
  let b = m[1];
  let c = m[2];
  let r0 = cross(b, c);
  let r1 = cross(c, a);
  let r2 = cross(a, b);
  let det = dot(a, r0);
  if (abs(det) < 1e-12) { return mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0)); }
  return transpose(mat3x3<f32>(r0, r1, r2)) * (1.0 / det);
}

@compute @workgroup_size(64, 1, 1)
fn body_solve(@builtin(global_invocation_id) id: vec3<u32>) {
  let b = id.x;
  let count = min(atomicLoad(&sums[COUNTER]), MAX_BODIES);
  if (b >= count) { return; }
  let n = gdim(U.g);
  let base = b * BODY_WORDS;
  let M = read64(base) / FX_BM;
  if (M < 1e-6) {
    bodies[b * 3u] = vec4<f32>(0.0);
    bodies[b * 3u + 1u] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    bodies[b * 3u + 2u] = vec4<f32>(0.0);
    return;
  }
  let cr = vec3<f32>(read64(base + 2u), read64(base + 4u), read64(base + 6u)) / FX_BM / M;
  let P = vec3<f32>(read64(base + 8u), read64(base + 10u), read64(base + 12u)) / FX_BV;
  let sxx = read64(base + 14u) / FX_BM - M * cr.x * cr.x;
  let syy = read64(base + 16u) / FX_BM - M * cr.y * cr.y;
  let szz = read64(base + 18u) / FX_BM - M * cr.z * cr.z;
  let sxy = read64(base + 20u) / FX_BM - M * cr.x * cr.y;
  let sxz = read64(base + 22u) / FX_BM - M * cr.x * cr.z;
  let syz = read64(base + 24u) / FX_BM - M * cr.y * cr.z;
  let tr = sxx + syy + szz;
  let s6 = M / 6.0;    // each cell's own inertia (a unit cube)
  let I = mat3x3<f32>(vec3<f32>(tr - sxx + s6, -sxy, -sxz),
                      vec3<f32>(-sxy, tr - syy + s6, -syz),
                      vec3<f32>(-sxz, -syz, tr - szz + s6));
  let Lr = vec3<f32>(read64(base + 26u), read64(base + 28u), read64(base + 30u)) / FX_BW;
  let L = Lr - cross(cr, P);
  var V = P / M;
  var w = inverse3(I) * L;
  // a piece of a few cells (a frozen drop) has almost no moment of inertia: its angular momentum is noise
  // from the flow round it, so it only translates; bigger ones turn at most about 20 rad/s (three turns a
  // second), which a piece of ice in water never exceeds
  let wmax = 20.0 * U.g.n.w * smoothstep(2.0, 6.0, M);
  let wl = length(w);
  if (wl > wmax) { w *= wmax / max(wl, 1e-9); }
  let held = (atomicLoad(&sums[base + 32u]) & 1u) != 0u;
  if (held) {
    V = vec3<f32>(0.0);
    w = vec3<f32>(0.0);
  }
  let rc = vec3<f32>(cell_of(atomicLoad(&sums[base + 33u]), n));
  bodies[b * 3u] = vec4<f32>(rc + cr + vec3<f32>(0.5), M);
  bodies[b * 3u + 1u] = vec4<f32>(V, select(0.0, 1.0, held));
  // w: a piece under three cells (a frozen drop, a pellet) is left to the liquid's own motion: rigid, it
  // would only add noise
  bodies[b * 3u + 2u] = vec4<f32>(w, select(0.0, 1.0, M < 3.0));
}

// The piece of ice free to move in cell c (a piece held on a surface leaves the grid as the solve made it:
// written into the grid after the solve, a wall there would break the liquid's incompressibility, and
// its water, still flowing in, would be thrown back out in jets; its particles simply stay put).
fn body_at(c: vec3<i32>, n: vec3<i32>) -> u32 {
  if (!in_grid(c, n)) { return NO_BODY; }
  let b = bodycell[nidx(c, n)];
  if (b == NO_BODY || bodies[b * 3u + 1u].w > 0.5 || bodies[b * 3u + 2u].w > 0.5) { return NO_BODY; }
  return b;
}

// Velocity of body b at x (cells): its translation and its rotation (w in m/s per cell) about its centre.
fn rigid_at(b: u32, x: vec3<f32>) -> vec3<f32> {
  let com = bodies[b * 3u].xyz;
  return bodies[b * 3u + 1u].xyz + cross(bodies[b * 3u + 2u].xyz, x - com);
}

@compute @workgroup_size(8, 8, 4)
fn body_faces(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let m = n + vec3<i32>(1);
  let c = vec3<i32>(id);
  if (any(c >= m)) { return; }
  var t = textureLoad(vin, c, 0);
  let flags = u32(t.w + 0.5);
  var fl = flags;
  for (var k = 0u; k < 3u; k++) {
    if (flag_fixed(flags, k)) { continue; }
    var e = vec3<i32>(0);
    e[k] = 1;
    let ba = body_at(c - e, n);
    let bb = body_at(c, n);
    if (ba == NO_BODY && bb == NO_BODY) { continue; }
    let x = vec3<f32>(c) + face_off(k);
    var s = 0.0;
    var wsum = 0.0;
    if (ba != NO_BODY) {
      s += rigid_at(ba, x)[k];
      wsum += 1.0;
    }
    if (bb != NO_BODY) {
      s += rigid_at(bb, x)[k];
      wsum += 1.0;
    }
    t[k] = s / wsum;
    fl |= 1u << k;
  }
  textureStore(vout, c, vec4<f32>(t.xyz, f32(fl)));
}
