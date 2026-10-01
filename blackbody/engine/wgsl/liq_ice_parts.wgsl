// Ice bodies, the particles: frozen particles move with their piece of ice exactly (liq_ice.wgsl).
//
// hide: before the liquid's particles take the grid velocity and move (liq_g2p.wgsl), frozen particles
//   in a piece of ice are set aside (age stored as -2 - age, which reads as not alive), so that step
//   leaves them where they are.
// move: afterwards each is carried by its piece's rigid motion (turned about the piece's centre by the
//   exact rotation, and moved by its translation), takes the piece's velocity and the matching affine
//   field, is kept out of solids and walls, and gets its age back. It is counted in its cell as the
//   others were by their move (a source tops cells up to its count: uncounted, the ice under a pour
//   would have water packed into it).
//!include common.wgsl
//!include liq_common.wgsl
//!include liq_therm_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,    // slot capacity, collision radius (cells)
};

@group(0) @binding(0) var<storage, read_write> parts: array<Particle>;
@group(0) @binding(1) var<storage, read> therm: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> bodycell: array<u32>;
@group(0) @binding(3) var<storage, read> bodies: array<vec4<f32>>;
@group(0) @binding(4) var sdf: texture_3d<f32>;
@group(0) @binding(5) var<storage, read_write> ctr: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> freelist: array<u32>;
@group(0) @binding(7) var<storage, read_write> cellcount: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> U: Params;

// The piece of ice a frozen particle at x belongs to: its cell's, or the nearest frozen cell's around it
// (a particle at the edge of a piece can sit in a cell that is mostly water). Pieces under three cells
// (frozen drops) are not rigid: their particles move with the liquid's own motion.
fn body_of(x: vec3<f32>, n: vec3<i32>) -> u32 {
  let b = body_near(x, n);
  if (b == NO_BODY || bodies[b * 3u + 2u].w > 0.5) { return NO_BODY; }
  return b;
}

fn body_near(x: vec3<f32>, n: vec3<i32>) -> u32 {
  let c = clamp(vec3<i32>(floor(x)), vec3<i32>(0), n - vec3<i32>(1));
  let b0 = bodycell[nidx(c, n)];
  if (b0 != NO_BODY) { return b0; }
  var best = NO_BODY;
  var bd = 1.0e9;
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var xx = -1; xx <= 1; xx++) {
        let q = c + vec3<i32>(xx, y, z);
        if (!in_grid(q, n)) { continue; }
        let b = bodycell[nidx(q, n)];
        if (b == NO_BODY) { continue; }
        let d = length(vec3<f32>(q) + vec3<f32>(0.5) - x);
        if (d < bd) {
          bd = d;
          best = b;
        }
      }
    }
  }
  return best;
}

@compute @workgroup_size(64, 1, 1)
fn hide(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.k.x)) { return; }
  var P = parts[i];
  if (!alive(P)) { return; }
  let s = therm[i];
  if (ice_of(s.x, s.y) < 0.5) { return; }
  if (body_of(P.p.xyz, gdim(U.g)) == NO_BODY) { return; }
  P.p.w = -2.0 - P.p.w;
  parts[i] = P;
}

@compute @workgroup_size(64, 1, 1)
fn move_ice(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.k.x)) { return; }
  var P = parts[i];
  if (P.p.w > -1.5) { return; }      // alive and not set aside, or a free slot
  let n = gdim(U.g);
  let nf = vec3<f32>(n);
  let h = U.g.n.w;
  let dt = U.g.bc.w;
  let x = P.p.xyz;
  let age = -P.p.w - 2.0 + dt;
  var p = x;
  var v = P.v;
  let b = body_of(x, n);
  if (b != NO_BODY) {
    let com = bodies[b * 3u].xyz;
    let V = bodies[b * 3u + 1u].xyz;
    let w = bodies[b * 3u + 2u].xyz;     // m/s per cell
    var r = x - com;
    let wl = length(w);
    if (wl > 1e-9) { r = rotate_axis(r, w / wl, wl * dt / h); }
    p = com + r + V * (dt / h);
    v = V + cross(w, r);
    // the affine field of a rigid rotation: the velocity's gradient (m/s per cell) is the cross product with w
    apic_set(&P, vec3<f32>(0.0, -w.z, w.y), vec3<f32>(w.z, 0.0, -w.x), vec3<f32>(-w.y, w.x, 0.0));
  }
  // solids and the ground
  let rc = U.k.y;
  var sd = interp_cell(sdf, p, n);
  if (U.g.bc.z < 0.5 && p.y < sd.x) { sd = vec4<f32>(p.y, 0.0, 1.0, 0.0); }
  let gl = length(sd.yzw);
  if (gl > 1e-6 && sd.x < rc) {
    let nrm = sd.yzw / gl;
    p += (rc - sd.x) * nrm;
    let vn = dot(v, nrm);
    if (vn < 0.0) { v -= vn * nrm; }
  }
  // the box's walls: closed ones hold it, it leaves through open ones
  var dead = false;
  for (var a = 0; a < 3; a++) {
    if (p[a] < rc) {
      let open_lo = select(U.g.bc.x > 0.5, U.g.bc.z > 0.5, a == 1);
      if (open_lo) {
        if (p[a] < 0.0) { dead = true; }
      } else {
        p[a] = rc;
        v[a] = max(v[a], 0.0);
      }
    }
    if (p[a] > nf[a] - rc) {
      let open_hi = select(U.g.bc.x > 0.5, U.g.bc.y > 0.5, a == 1);
      if (open_hi) {
        if (p[a] > nf[a]) { dead = true; }
      } else {
        p[a] = nf[a] - rc;
        v[a] = min(v[a], 0.0);
      }
    }
  }
  if (dead || !(all(abs(p) < vec3<f32>(1.0e6)) && all(abs(v) < vec3<f32>(1.0e6)))) {
    P.p.w = -1.0;
    parts[i] = P;
    let k = atomicAdd(&ctr[C_FREE], 1);
    freelist[u32(k)] = i;
    return;
  }
  P.p = vec4<f32>(p, age);
  P.v = v;
  parts[i] = P;
  let cc = clamp(vec3<i32>(floor(p)), vec3<i32>(0), n - vec3<i32>(1));
  atomicAdd(&cellcount[nidx(cc, n)], 1u);
}
