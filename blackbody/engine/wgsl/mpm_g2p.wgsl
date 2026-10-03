// Matter, grid to particle: each particle picks up the grid's new velocity and the affine velocity round it (APIC),
// deforms with it (F <- (I + dt C) F), gives way where its material yields, and moves. One that leaves the grid
// (through an open side, the top, or the bottom of a box without ground) is gone. By a sheet of fabric, it takes the
// sheet's motion for the nodes on its far side, stays on its own side, and keeps that side for the next step
// (mpm_common.wgsl SHEET_N). Where it lies open to the air, the wind moves it (mpm_air.wgsl's air): past the friction
// speed that starts its grains moving, its top layer carries the mass a wind of that strength moves (the saltation
// flux, Kawamura's law with White's constant: sand drifts and creeps downwind, snow streams off a ridge), and grains
// in the air are carried by it at their settling speed (spray from a blast drifts off on the wind). The particles are
// far bigger than grains, so the flux is set, not the force: a force on a layer of them is held by their friction.
//!include mpm_common.wgsl

struct Params {
  n: vec4<f32>,      // grid nodes (x, y, z), particles (count)
  s: vec4<f32>,      // step (s), node spacing (m), fabric (mpm_common.wgsl SHEET_N), simulation time at the start of
                     // the step (s)
  mats: array<MMat, MAX_MATS>,
  air: vec4<f32>,    // the wind: on (1/0), how far off the surface it is taken (m), the air's density (kg/m^3), the
                     // ground's height (nodes; very low: none)
  dg: vec4<f32>,     // the dust grid (the gas's cells two to a side): its corner (m from the grid's node 0), cell size (m)
  dn: vec4<f32>,     // its cells, on (1/0)
};

@group(0) @binding(0) var<storage, read_write> P: array<MParticle>;
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<atomic<u32>>;   // fastest speed (f32 bits), bounds (grid units), least Jp
@group(0) @binding(3) var<storage, read> CF: array<i32>;                  // the fabric round each node (cloth_matter.wgsl)
@group(0) @binding(4) var<storage, read_write> CT: array<atomic<i32>>;    // per node: momentum the fabric took (3)
@group(0) @binding(5) var air: texture_3d<f32>;   // the air's velocity at the nodes (m/s); w: 1 under the liquid
@group(0) @binding(6) var<storage, read_write> DU: array<atomic<i32>>;   // the dust shed over the frame (kg, DUST_K)
@group(1) @binding(0) var<uniform> U: Params;

const RHO_NODE: f32 = 125.0;   // a node's density (kg/m^3) per unit of its mass (particles of water, eight to a node)
const DUST_K: f32 = 1.0e11;    // dust (kg) fixed point (mpm_dust.wgsl)

// Fine dust shed at p (kg; m from the grid's node 0), onto the dust grid, rounded up or down at random so a trickle still
// counts.
fn shed(p: vec3<f32>, kg: f32, r: f32) {
  if (U.dn.w < 0.5 || kg <= 0.0) { return; }
  let d = vec3<i32>(U.dn.xyz);
  let c = vec3<i32>(floor((p - U.dg.xyz) / U.dg.w));
  if (any(c < vec3<i32>(0)) || any(c >= d)) { return; }
  atomicAdd(&DU[u32((c.z * d.y + c.y) * d.x + c.x)], i32(floor(min(kg * DUST_K, 1.0e9) + r)));
}

// The air at x (grid units), trilinear.
fn air_at(x: vec3<f32>) -> vec4<f32> {
  let n = vec3<i32>(U.n.xyz);
  let q = clamp(x, vec3<f32>(0.0), vec3<f32>(n) - vec3<f32>(1.001));
  let b = vec3<i32>(floor(q));
  let f = q - vec3<f32>(b);
  var a = vec4<f32>(0.0);
  for (var k = 0; k < 8; k++) {
    let o = vec3<i32>(k & 1, (k >> 1) & 1, k >> 2);
    let w3 = mix(vec3<f32>(1.0) - f, f, vec3<f32>(o));
    a += w3.x * w3.y * w3.z * textureLoad(air, min(b + o, n - vec3<i32>(1)), 0);
  }
  return a;
}

fn sheet(k: u32, tdx: f32) -> Sheet {
  return sheet_at(vec4<i32>(CF[k], CF[k + 1u], CF[k + 2u], CF[k + 3u]), vec4<i32>(CF[k + 4u], CF[k + 5u], CF[k + 6u],
                  CF[k + 7u]), CF[k + 8u], tdx, U.s.y);
}

// The bounds of the matter (grid units, kept as integers shifted by 2^20 so they stay positive).
fn bound(x: vec3<f32>) {
  let lo = vec3<u32>(floor(x) + vec3<f32>(1048576.0));
  let hi = vec3<u32>(ceil(x) + vec3<f32>(1048576.0));
  atomicMin(&stats[1], lo.x);
  atomicMin(&stats[2], lo.y);
  atomicMin(&stats[3], lo.z);
  atomicMax(&stats[4], hi.x);
  atomicMax(&stats[5], hi.y);
  atomicMax(&stats[6], hi.z);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.n.w)) { return; }
  var p = P[i];
  if (p.x.w < 0.0) { return; }
  let n = vec3<i32>(U.n.xyz);
  let dt = U.s.x;
  let dx = U.s.y;
  if (p.c1.w > U.s.w) {
    // held where it is until it is let go (keeping the speed it is thrown at then)
    bound(p.x.xyz);
    return;
  }
  let base = vec3<i32>(floor(p.x.xyz - vec3<f32>(0.5)));
  let fx = p.x.xyz - vec3<f32>(base);
  let w = bspline(fx);
  let m = U.mats[u32(p.x.w)];
  let side = select(0.0, p.f2.w, U.s.z > 0.5);
  let tdx0 = (U.s.z - 1.0) / dx;
  let tdx1 = tdx0 + dt / dx;
  // the sheet of fabric round it, if any, weighted: how far in front of it the particle is (nodes) now and where the
  // sheet will be once the step is over, its normal, its velocity
  var sw = 0.0;
  var sd0 = 0.0;
  var sd = 0.0;
  var sn = vec3<f32>(0.0);
  var sv = vec3<f32>(0.0);
  var hold = 0.0;
  var v = vec3<f32>(0.0);
  var B = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  var fill = 0.0;                 // the matter round it (its nodes' mass, weighted), and which way it lies
  var mgrad = vec3<f32>(0.0);
  for (var a = 0; a < 3; a++) {
    for (var b = 0; b < 3; b++) {
      for (var c = 0; c < 3; c++) {
        let node = clamp(base + vec3<i32>(a, b, c), vec3<i32>(0), n - vec3<i32>(1));
        let wt = w[a].x * w[b].y * w[c].z;
        let dpos = (vec3<f32>(f32(a), f32(b), f32(c)) - fx) * dx;
        let vn = textureLoad(vel, node, 0);
        var vi = vn.xyz;
        fill += wt * vn.w;
        mgrad += wt * vn.w * dpos;
        if (U.s.z > 0.5) {
          let kc = nidx(node, n) * SHEET_N;
          let sh = sheet(kc, tdx0);
          if (sh.side != 0.0) {
            let ww = wt * sh.w;
            let along = dot(p.x.xyz - vec3<f32>(node), sh.n);
            sw += ww;
            sd0 += ww * (along + sh.off);
            sd += ww * (along + sheet_off(sh, tdx1));
            sn += ww * sh.n;
            sv += ww * sh.v;
            hold += ww * sheet_hold(sh, select(sh.side, side, side != 0.0));
            if (side != 0.0 && sh.side != side) {
              // a node on the far side: as if the sheet were there instead, moving as it does (as far as it holds),
              // and what it stops of the particle's motion, the cloth takes
              let vg = mix(p.v.xyz, boundary(p.v.xyz, sh.v, sh.n * side, m.c.x), sheet_hold(sh, side));
              let dp = (wt * m.a.w / 1000.0) * (p.v.xyz - vg);
              if (dot(dp, dp) > 0.0) {
                let q = clamp(dp * SHEET_FX_T, vec3<f32>(-2.0e9), vec3<f32>(2.0e9));
                let kt = 3u * nidx(node, n);
                atomicAdd(&CT[kt], i32(round(q.x)));
                atomicAdd(&CT[kt + 1u], i32(round(q.y)));
                atomicAdd(&CT[kt + 2u], i32(round(q.z)));
              }
              vi = vg;
            }
          }
        }
        v += wt * vi;
        B += wt * mat3x3<f32>(vi * dpos.x, vi * dpos.y, vi * dpos.z);
      }
    }
  }
  let C = (4.0 / (dx * dx)) * B;
  let y = plastic(m, (eye3() + dt * C) * from_rows(p.f0, p.f1, p.f2), p.v.w, dt);
  let Ct = transpose(C);
  let Ft = transpose(y.f);
  p.c0 = vec4<f32>(Ct[0], p.c0.w);
  p.c1 = vec4<f32>(Ct[1], p.c1.w);
  p.c2 = vec4<f32>(Ct[2], p.c2.w);
  p.f0 = vec4<f32>(Ft[0], p.f0.w);
  p.f1 = vec4<f32>(Ft[1], p.f1.w);
  // the wind, where it lies open to the air (its nodes part empty, the matter's mass lying to one side of it)
  if (U.air.x > 0.5 && (m.f.x > 0.0 || m.f.y > 0.0)) {
    let full = fill * RHO_NODE / m.a.w;
    let gl = length(mgrad);
    if (full < 0.85 && gl > 1.0e-9) {
      let nrm = -mgrad / gl;
      let z = U.air.y;
      let ua = air_at(p.x.xyz + nrm * (z / dx));
      if (ua.w < 0.5) {
        if (m.f.y > 0.0 && full < 0.15 && p.x.y - U.air.w > 1.0) {
          // off the ground, clear of the rest: carried toward the air's speed as fast as its grains settle (a loose
          // stream across only: it falls as a stream does, the air in it falling with it)
          var dv = (ua.xyz - v) * (1.0 - exp(-9.81 * dt / m.f.y));
          if (full > 0.07) { dv.y = 0.0; }
          v += dv;
          // (and sheds the fine dust on it, a third of it a second, as it flies)
          let hr = fract(sin(f32(i) * 12.9898 + U.s.w * 78.233) * 43758.5453);
          shed(p.x.xyz * dx, m.f.w * m.a.w * 0.125 * dx * dx * dx * 0.3 * dt, hr);
        } else if (m.f.x > 0.0) {
          // on the surface: the wind along it (a log layer above grains of its roughness) past the friction speed that
          // starts its grains moving; its top layer (half a node deep) carries the saltation flux, q = 2.78 rho_air /
          // g u*^3 (1 - r)(1 + r)^2, r = u*t / u*
          let ut = ua.xyz - dot(ua.xyz, nrm) * nrm;
          let lu = length(ut);
          let ustar = 0.4 * lu / clamp(log(z / max(m.f.z, 1.0e-6)), 3.0, 25.0);
          if (ustar > m.f.x && lu > 1.0e-6) {
            let r = m.f.x / ustar;
            let q = 2.78 * U.air.z / 9.81 * ustar * ustar * ustar * (1.0 - r) * (1.0 + r) * (1.0 + r);
            let us = q / (m.a.w * 0.5 * dx) * smoothstep(0.85, 0.55, full);
            let dir = ut / lu;
            let along = dot(v, dir);
            if (along < us) { v += dir * (us - along); }     // (each step: the grid shares it with the layer under)
            // the dust in what it moves rises off it as it goes: its fine share of the flux over a hop's length (30 cm)
            let hr = fract(sin(f32(i) * 12.9898 + U.s.w * 78.233) * 43758.5453);
            shed(p.x.xyz * dx, m.f.w * q / 0.3 * 0.25 * dx * dx * smoothstep(0.85, 0.55, full) * dt, hr);
          }
        }
      }
    }
  }
  var x = p.x.xyz + v * (dt / dx);
  // by fabric: never across the sheet, nor nearer it than SHEET_GAP where it holds the particle up (the side it was on,
  // or, just come by it, the side it is on), kept for the next step
  var keep = 0.0;
  let snl = length(sn);
  if (sw > 1.0e-6 && snl > 1.0e-6) {
    let nrm = sn / snl;
    let s = select(select(-1.0, 1.0, sd0 >= 0.0), side, side != 0.0);
    let d1 = sd / sw + dot(x - p.x.xyz, nrm);       // (nodes in front of it, once moved)
    let gap = SHEET_GAP * hold / sw;
    if (d1 * s < gap) {
      x += (gap - d1 * s) * s * nrm;
      let vn = dot(v - sv / sw, nrm) * s;
      if (vn < 0.0) { v -= vn * s * nrm; }
    }
    keep = s;
  }
  p.f2 = vec4<f32>(Ft[2], keep);
  p.v = vec4<f32>(v, y.jp);
  if (any(x < vec3<f32>(0.6)) || any(x >= vec3<f32>(n) - vec3<f32>(1.6))) {
    p.x.w = -1.0;              // gone out of the grid
    P[i] = p;
    return;
  }
  p.x = vec4<f32>(x, p.x.w);
  P[i] = p;
  atomicMax(&stats[0], bitcast<u32>(length(v)));
  bound(x);
  atomicMin(&stats[7], bitcast<u32>(max(y.jp, 0.0)));     // (the most packed snow: how hard it is)
}
