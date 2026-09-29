// Grid to particle, then move. Each particle takes the new grid velocity (PIC) blended with its
// own velocity plus the grid's change this step (FLIP), and the gradient of the grid velocity as
// its APIC affine field. Inside the liquid it then moves through the grid velocity (midpoint
// rule); where particles are sparse (spray, droplets, thin sheets) it flies on its own velocity,
// because a midpoint sample taken across the surface would pin it in place while its velocity
// kept growing. It is pushed out of solids and walls, and freed if it leaves through an open
// side. Survivors are counted per cell for the sources.
//
// Whitewater is born here too (after Ihmsen et al. 2012): where the liquid is fast and either
// traps air (liquid colliding with liquid or a surface: the flow converges before the pressure
// solve), churns (strong vorticity) or breaks (a convex crest stretched outward), a particle releases
// whitewater particles into a ring buffer.
//!include common.wgsl
//!include liq_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,  // slot capacity, FLIP ratio, APIC on (1/0), speed limit (m/s)
  c: vec4<f32>,  // collision radius (cells), deceleration of liquid touching solids (m/s^2), density below which particles fly free, density at which they follow the grid
  w0: vec4<f32>, // whitewater: particles per second at full potential, minimum speed (m/s), turbulence weight, crest weight
  w1: vec4<f32>, // whitewater: capacity (0 = off), life (s), step seed, rest density (particles)
  m: vec4<f32>,  // damping (1/s, settling during pre-roll)
};

@group(0) @binding(0) var<storage, read_write> parts: array<Particle>;
@group(0) @binding(1) var vnew: texture_3d<f32>;
@group(0) @binding(2) var vold: texture_3d<f32>;
@group(0) @binding(3) var sdf: texture_3d<f32>;
@group(0) @binding(4) var<storage, read_write> ctr: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> freelist: array<u32>;
@group(0) @binding(6) var<storage, read_write> cellcount: array<atomic<u32>>;
@group(0) @binding(7) var dens: texture_3d<f32>;
@group(0) @binding(8) var<storage, read_write> WA: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> WB: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read_write> wctr: array<atomic<u32>>;
@group(0) @binding(11) var nrm: texture_3d<f32>;
@group(0) @binding(12) var kappa: texture_3d<f32>;
//!include noise.wgsl

fn ramp(x: f32, lo: f32, hi: f32) -> f32 { return clamp((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0); }

fn emit_whitewater(i: u32, x: vec3<f32>, v: vec3<f32>, gu: vec4<f32>, gv: vec4<f32>, gw: vec4<f32>, n: vec3<i32>) {
  let sp = length(v);
  let pk = ramp(sp, U.w0.y, 3.0 * U.w0.y);
  if (pk <= 0.0) { return; }
  let h = U.g.n.w;
  // trapped air: convergence of the transferred velocity (before the pressure solve) at this cell
  let c0 = clamp(vec3<i32>(floor(x)), vec3<i32>(0), n - vec3<i32>(1));
  let a0 = textureLoad(vold, c0, 0).xyz;
  let div = (textureLoad(vold, c0 + vec3<i32>(1, 0, 0), 0).x - a0.x + textureLoad(vold, c0 + vec3<i32>(0, 1, 0), 0).y - a0.y
             + textureLoad(vold, c0 + vec3<i32>(0, 0, 1), 0).z - a0.z) / h;
  let om = vec3<f32>(gw.z - gv.w, gu.w - gw.y, gv.y - gu.z) / h;
  // both measured against the speed per cell, so the thresholds hold at any scale and resolution
  let q = h / max(sp, 1e-3);
  let ta = max(ramp(-div * q, 0.3, 1.2), 0.5 * ramp(length(om) * q, 0.8, 2.5));
  var wc = 0.0;
  let rho = interp_cell(dens, x, n).x / U.w1.w;
  if (rho < 0.9) {
    let c = clamp(vec3<i32>(floor(x)), vec3<i32>(0), n - vec3<i32>(1));
    let nm = textureLoad(nrm, c, 0);
    let k = textureLoad(kappa, c, 0).x * h;
    // a crest breaks where the surface is convex and being stretched outward, faster than the liquid
    // behind it (a rigidly moving blob has no velocity gradient and sheds nothing)
    let nn = nm.xyz;
    let stretch = dot(nn, vec3<f32>(dot(gu.yzw, nn), dot(gv.yzw, nn), dot(gw.yzw, nn))) / h;
    wc = ramp(k, 0.08, 0.5) * ramp(stretch * h / max(sp, 1e-3), 0.05, 0.4) * ramp(dot(v / max(sp, 1e-6), nn), 0.3, 0.8);
  }
  let rate = U.w0.x * (U.w0.z * ta + U.w0.w * wc) * pk;
  if (rate <= 0.0) { return; }
  let seed = i * 2246822519u + u32(U.w1.z) * 3266489917u;
  let cnt = u32(min(floor(rate * U.g.bc.w + rand1(seed)), 6.0));
  let cap = u32(U.w1.x);
  for (var j = 0u; j < cnt; j++) {
    let r = rand3(seed, j + 1u) - vec3<f32>(0.5);
    let r2 = rand3(seed, j + 17u);
    let slot = atomicAdd(&wctr[0], 1u) % cap;
    WA[slot] = vec4<f32>(x + r * 0.8, U.w1.y * (0.4 + 0.8 * r2.x));
    WB[slot] = vec4<f32>(v * (0.9 + 0.2 * r2.y) + (r2 - vec3<f32>(0.5)) * (0.15 * sp), 0.0);
  }
}
@group(1) @binding(0) var<uniform> U: Params;

fn open_side(axis: i32, lo: bool) -> bool {
  if (axis == 1) { return select(U.g.bc.y, U.g.bc.z, lo) > 0.5; }
  return U.g.bc.x > 0.5;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.k.x)) { return; }
  var P = parts[i];
  if (P.p.w < 0.5) { return; }
  let n = gdim(U.g);
  let nf = vec3<f32>(n);
  let dt = U.g.bc.w;
  let ih = 1.0 / U.g.n.w;
  let x = P.p.xyz;

  // transfer
  let gu = interp_comp(vnew, x, 0u, n);
  let gv = interp_comp(vnew, x, 1u, n);
  let gw = interp_comp(vnew, x, 2u, n);
  let pic = vec3<f32>(gu.x, gv.x, gw.x);
  let old = vec3<f32>(interp_val(vold, x, 0u, n), interp_val(vold, x, 1u, n), interp_val(vold, x, 2u, n));
  var v = mix(pic, P.v.xyz + (pic - old), U.k.y) * exp(-U.m.x * dt);
  let sp = length(v);
  if (sp > U.k.w) { v *= U.k.w / sp; }
  let ap = U.k.z;
  P.cx = vec4<f32>(gu.yzw * ap, 0.0);
  P.cy = vec4<f32>(gv.yzw * ap, 0.0);
  P.cz = vec4<f32>(gw.yzw * ap, 0.0);

  if (U.w1.x > 0.0) { emit_whitewater(i, x, v, gu, gv, gw, n); }

  // move: through the grid velocity (midpoint rule) inside the liquid, ballistically outside
  let inl = smoothstep(U.c.z, U.c.w, interp_cell(dens, x, n).x);
  var step_v = v;
  if (inl > 0.0) {
    let v1 = mac_vel(vnew, x, n);
    let xm = x + 0.5 * dt * v1 * ih;
    step_v = mix(v, mac_vel(vnew, xm, n), inl);
  }
  var p = x + dt * step_v * ih;

  // solids and the ground: push out along the distance gradient, drop the velocity into the
  // surface, and slow the liquid touching it with a constant deceleration (fast sheets slide on,
  // slow films stop and pin, as a spreading puddle does)
  let rc = U.c.x;
  var sd = interp_cell(sdf, p, n);
  if (U.g.bc.z < 0.5 && p.y < sd.x) { sd = vec4<f32>(p.y, 0.0, 1.0, 0.0); }
  let gl = length(sd.yzw);
  if (gl > 1e-6 && sd.x < 1.5) {
    let nrm = sd.yzw / gl;
    if (sd.x < rc) {
      p += (rc - sd.x) * nrm;
      let vn = dot(v, nrm);
      if (vn < 0.0) { v -= vn * nrm; }
    }
    let contact = 1.0 - smoothstep(rc, 1.5, sd.x);
    let vt = v - dot(v, nrm) * nrm;
    let st = length(vt);
    if (st > 1e-6) { v -= vt * (min(st, U.c.y * dt * contact) / st); }
  }

  // domain walls: clamp at closed sides, free the particle past open ones
  var dead = false;
  for (var a = 0; a < 3; a++) {
    if (p[a] < rc) {
      if (open_side(a, true)) {
        if (p[a] < 0.0) { dead = true; }
      } else {
        p[a] = rc;
        if (v[a] < 0.0) { v[a] = 0.0; }
      }
    }
    if (p[a] > nf[a] - rc) {
      if (open_side(a, false)) {
        if (p[a] > nf[a]) { dead = true; }
      } else {
        p[a] = nf[a] - rc;
        if (v[a] > 0.0) { v[a] = 0.0; }
      }
    }
  }
  if (!(all(abs(p) < vec3<f32>(1.0e6)) && all(abs(v) < vec3<f32>(1.0e6)))) { dead = true; }

  if (dead) {
    P.p.w = 0.0;
    parts[i] = P;
    let k = atomicAdd(&ctr[C_FREE], 1);
    freelist[u32(k)] = i;
    return;
  }
  P.p = vec4<f32>(p, 1.0);
  P.v = vec4<f32>(v, P.v.w + dt);
  parts[i] = P;
  let cc = clamp(vec3<i32>(floor(p)), vec3<i32>(0), n - vec3<i32>(1));
  atomicAdd(&cellcount[u32(cc.x + n.x * (cc.y + n.y * cc.z))], 1u);
}
