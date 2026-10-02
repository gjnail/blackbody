// Matter, grid to particle: each particle picks up the grid's new velocity and the affine velocity round it (APIC),
// deforms with it (F <- (I + dt C) F), gives way where its material yields, and moves. One that leaves the grid
// (through an open side, the top, or the bottom of a box without ground) is gone. By a sheet of fabric, it takes the
// sheet's motion for the nodes on its far side, stays on its own side, and keeps that side for the next step
// (mpm_common.wgsl SHEET_N).
//!include mpm_common.wgsl

struct Params {
  n: vec4<f32>,      // grid nodes (x, y, z), particles (count)
  s: vec4<f32>,      // step (s), node spacing (m), fabric (mpm_common.wgsl SHEET_N), simulation time at the start of
                     // the step (s)
  mats: array<MMat, MAX_MATS>,
};

@group(0) @binding(0) var<storage, read_write> P: array<MParticle>;
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<atomic<u32>>;   // fastest speed (f32 bits), bounds (grid units), least Jp
@group(0) @binding(3) var<storage, read> CF: array<i32>;                  // the fabric round each node (cloth_matter.wgsl)
@group(0) @binding(4) var<storage, read_write> CT: array<atomic<i32>>;    // per node: momentum the fabric took (3)
@group(1) @binding(0) var<uniform> U: Params;

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
  for (var a = 0; a < 3; a++) {
    for (var b = 0; b < 3; b++) {
      for (var c = 0; c < 3; c++) {
        let node = clamp(base + vec3<i32>(a, b, c), vec3<i32>(0), n - vec3<i32>(1));
        let wt = w[a].x * w[b].y * w[c].z;
        let dpos = (vec3<f32>(f32(a), f32(b), f32(c)) - fx) * dx;
        var vi = textureLoad(vel, node, 0).xyz;
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
