// Grid to particle, then move. Each particle takes the new grid velocity (PIC) blended with its
// own velocity plus the grid's change this step (FLIP), and the gradient of the grid velocity as
// its APIC affine field. Inside the liquid it then moves through the grid velocity (midpoint
// rule); where particles are sparse (spray, droplets, thin sheets) it flies on its own velocity,
// because a midpoint sample taken across the surface would pin it in place while its velocity
// kept growing. It is pushed out of solids and walls, and freed if it leaves through an open
// side. Survivors are counted per cell for the sources. Liquid squeezed against a moving solid
// (sand, broken pieces) with no way out, packed far denser than at rest, takes the grid's
// velocity, and past a limit the excess soaks into the sand's pores (is gone).
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
  m: vec4<f32>,  // damping (1/s, settling during pre-roll), open water level (cells, < 0 off), cooling (1/s),
                 // the open sides the liquid may leave through under the level (bits -x, +x, -z, +z)
  s: vec4<f32>,  // moving solids in svel (1/0), how packed (x rest density) liquid next to them may get, _, _
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
@group(0) @binding(13) var ocn_t: texture_2d<f32>;
@group(0) @binding(14) var svel: texture_3d<f32>;   // moving solids' velocity in the cells inside them (w = 1)
//!include liq_level.wgsl
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
  // air is trapped only where there is air: at the surface, not deep in the liquid, where it is full for
  // cells above (fast water shearing past a reef or the floor, a vortex deep down: those made bubbles
  // at the sea floor that rose and flecked a breaking wave's face with foam); and shear along a solid
  // is the wall's boundary layer, not churning (the impact term still counts there)
  let above = min(interp_cell(dens, x + vec3<f32>(0.0, 2.0, 0.0), n).x, interp_cell(dens, x + vec3<f32>(0.0, 4.0, 0.0), n).x);
  let near_air = 1.0 - smoothstep(0.75, 0.95, above / U.w1.w);
  let off_wall = smoothstep(1.0, 2.5, interp_cell(sdf, x, n).x);
  let ta = max(ramp(-div * q, 0.3, 1.2), 0.5 * ramp(length(om) * q, 0.8, 2.5) * off_wall) * near_air;
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

fn open_side(axis: i32, lo: bool, p: vec3<f32>) -> bool {
  if (axis == 1) { return select(U.g.bc.y, U.g.bc.z, lo) > 0.5; }
  // under the open water: held in, except where the sea or a current carries it out
  let bit = u32(axis) + select(1u, 0u, lo);   // -x 0, +x 1, -z 2, +z 3
  if (((u32(U.m.w + 0.5) >> (bit + 4u)) & 1u) == 1u) { return false; }   // a flume's wall, all the way up
  if (U.m.y >= 0.0 && p.y < level_at(p.xz) && ((u32(U.m.w + 0.5) >> bit) & 1u) == 0u) { return false; }
  return U.g.bc.x > 0.5;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.k.x)) { return; }
  var P = parts[i];
  if (!alive(P)) { return; }
  let n = gdim(U.g);
  let nf = vec3<f32>(n);
  let dt = U.g.bc.w;
  let ih = 1.0 / U.g.n.w;
  let x = P.p.xyz;

  // squeezed against a moving solid: how packed (x rest density; 0: it is not)
  var squeezed = 0.0;
  if (U.s.x > 0.5) {
    let cq = clamp(vec3<i32>(floor(x)), vec3<i32>(0), n - vec3<i32>(1));
    let rq = textureLoad(dens, cq, 0).x / U.w1.w;
    if (rq > 2.0) {
      var moving = textureLoad(svel, cq, 0).w > 0.5;
      for (var a = 0; a < 6; a++) {
        var e = vec3<i32>(0);
        e[a >> 1] = select(-1, 1, (a & 1) == 1);
        let q = cq + e;
        if (in_grid(q, n) && textureLoad(svel, q, 0).w > 0.5) { moving = true; }
      }
      if (moving) { squeezed = rq; }
    }
  }

  // transfer
  let gu = interp_comp(vnew, x, 0u, n);
  let gv = interp_comp(vnew, x, 1u, n);
  let gw = interp_comp(vnew, x, 2u, n);
  let pic = vec3<f32>(gu.x, gv.x, gw.x);
  let old = vec3<f32>(interp_val(vold, x, 0u, n), interp_val(vold, x, 1u, n), interp_val(vold, x, 2u, n));
  // (squeezed, the grid's velocity: its kicks there, pressing on liquid with no way out, do not add up)
  var v = mix(pic, P.v + (pic - old), select(U.k.y, 0.0, squeezed > 0.0)) * exp(-U.m.x * dt);
  let sp = length(v);
  if (sp > U.k.w) { v *= U.k.w / sp; }
  let ap = U.k.z;
  apic_set(&P, gu.yzw * ap, gv.yzw * ap, gw.yzw * ap);

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
      if (open_side(a, true, p)) {
        if (p[a] < 0.0) { dead = true; }
      } else {
        p[a] = rc;
        if (v[a] < 0.0) { v[a] = 0.0; }
      }
    }
    if (p[a] > nf[a] - rc) {
      if (open_side(a, false, p)) {
        if (p[a] > nf[a]) { dead = true; }
      } else {
        p[a] = nf[a] - rc;
        if (v[a] > 0.0) { v[a] = 0.0; }
      }
    }
  }
  if (!(all(abs(p) < vec3<f32>(1.0e6)) && all(abs(v) < vec3<f32>(1.0e6)))) { dead = true; }
  // squeezed past the limit: as much soaks away as brings its cell back to it
  if (squeezed > U.s.y && rand1(i * 2654435761u + u32(U.w1.z) * 2246822519u + 7u) < 1.0 - U.s.y / squeezed) { dead = true; }
  // open water: liquid heaped above the level at the edge of the box runs off over it (not over a
  // wave flume's walls: a crest shoaling or breaking along them, or a run-up against them, stands
  // above the sea's own surface, and skimmed off there the box drained with every wave)
  if (U.m.y >= 0.0 && U.g.bc.x > 0.5 && p.y > level_at(p.xz) + 0.5) {
    let fw = u32(U.m.w + 0.5) >> 4u;
    let big = 1.0e6;
    let side = min(min(select(p.x, big, (fw & 1u) != 0u), select(nf.x - p.x, big, (fw & 2u) != 0u)),
                   min(select(p.z, big, (fw & 4u) != 0u), select(nf.z - p.z, big, (fw & 8u) != 0u)));
    if (side < 0.75) { dead = true; }
  }

  if (dead) {
    P.p.w = -1.0;
    parts[i] = P;
    let k = atomicAdd(&ctr[C_FREE], 1);
    freelist[u32(k)] = i;
    return;
  }
  P.p = vec4<f32>(p, P.p.w + dt);
  P.v = v;
  if (U.m.z > 0.0) {
    // heat leaves through the surface (to the air and by glowing) and into the ground; the inside
    // only cools through the liquid around it
    let rho_here = interp_cell(dens, p, n).x / U.w1.w;
    var expose = 0.08 + 0.92 * (1.0 - smoothstep(0.55, 0.95, rho_here));
    if (U.g.bc.z < 0.5 && p.y < 1.5) { expose = max(expose, 0.6); }
    set_heat(&P, heat_of(P) * exp(-U.m.z * expose * dt));
  }
  parts[i] = P;
  let cc = clamp(vec3<i32>(floor(p)), vec3<i32>(0), n - vec3<i32>(1));
  atomicAdd(&cellcount[u32(cc.x + n.x * (cc.y + n.y * cc.z))], 1u);
}
