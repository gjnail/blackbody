// Grass (engine/strands.py): a solver substep's share of blade substeps for every blade, one thread each. The air
// (the gas in the box, the wind outside it) drags each point across the blade; the blade bends back toward its rest
// shape from the root up and keeps its length (rest_dir, turn_by); it stays out of the ground and the objects, which
// push it aside as they move through it. Where the gas is hot enough it heats, catches, burns down to stubble (the
// solver shortens it) and goes out, its stubble glowing a while.
//!include common.wgsl
//!include meshsdf.wgsl
//!include colliders.wgsl
//!include strand_common.wgsl

struct Params {
  sim: vec4<f32>,      // dt (one blade substep), blade substeps, blades, gas on (1/0)
  g: Grid,             // the simulation grid (fire-local metres)
  air: vec4<f32>,      // the wind outside the box (fire-local m/s), gustiness (0..1)
  air2: vec4<f32>,     // ambient K, flame K, time at the first substep (s), _
  gnd: vec4<f32>,      // ground (1/0), its height (fire-local m), _, _
  ccnt: vec4<f32>,
  col: array<Collider, MAX_COLLIDERS>,
  patches: array<Patch, MAX_PATCHES>,
};

@group(0) @binding(0) var<storage, read> BL: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> RT: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> RN: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> X: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> XP: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> ST: array<vec4<f32>>;
@group(0) @binding(6) var vel: texture_3d<f32>;
@group(0) @binding(7) var scal: texture_3d<f32>;
@group(0) @binding(8) var lin: sampler;
@group(0) @binding(9) var atlas: texture_3d<f32>;
@group(1) @binding(0) var<uniform> U: Params;

fn hash21(p: vec2<f32>) -> f32 {
  return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn vnoise(p: vec2<f32>) -> f32 {
  let i = floor(p);
  let f = p - i;
  let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash21(i), hash21(i + vec2<f32>(1.0, 0.0)), u.x), mix(hash21(i + vec2<f32>(0.0, 1.0)), hash21(i + vec2<f32>(1.0, 1.0)), u.x), u.y);
}

// The air at fire-local point p at time t: its velocity (m/s) and density (kg/m^3: hot gas is thinner). In the box
// the gas's; outside it the wind, in gusts that roll downwind across the field.
fn air_at(p: vec3<f32>, t: f32) -> vec4<f32> {
  if (U.sim.w > 0.5) {
    let n = U.g.n.xyz;
    let q = (p - U.g.org.xyz) / U.g.n.w;
    if (all(q >= vec3<f32>(0.0)) && all(q <= n)) {
      let v = vel_at(vel, lin, q, n);
      let T = max(samp_c(scal, lin, q, n).x, 0.0);
      let amb = U.air2.x;
      let tk = amb + (U.air2.y - amb) * min(T, 1.0);
      return vec4<f32>(v, 1.2 * amb / max(tk, amb));
    }
  }
  let w = U.air.xyz;
  let sp = length(w);
  if (sp < 1e-4) { return vec4<f32>(0.0, 0.0, 0.0, 1.2); }
  let d = w / sp;
  let along = dot(p.xz, d.xz) - 0.8 * sp * t;
  let across = dot(p.xz, vec2<f32>(-d.z, d.x));
  let gst = 0.65 * vnoise(vec2<f32>(along * 0.4, across * 0.25)) + 0.35 * vnoise(vec2<f32>(along * 1.3, across * 0.9) + vec2<f32>(17.0, 3.0));
  return vec4<f32>(w * max(1.0 + 1.2 * U.air.w * (2.0 * gst - 1.0), 0.0), 1.2);
}

// The gas's temperature at p (field units: 0 the air, 1 the flame); 0 outside the box.
fn temp_at(p: vec3<f32>) -> f32 {
  if (U.sim.w < 0.5) { return 0.0; }
  let n = U.g.n.xyz;
  let q = (p - U.g.org.xyz) / U.g.n.w;
  if (any(q < vec3<f32>(0.0)) || any(q > n)) { return 0.0; }
  return max(samp_c(scal, lin, q, n).x, 0.0);
}

fn col_normal(k: Collider, p: vec3<f32>, e: f32) -> vec3<f32> {
  let g = vec3<f32>(col_sdf(k, p + vec3<f32>(e, 0.0, 0.0)) - col_sdf(k, p - vec3<f32>(e, 0.0, 0.0)),
                    col_sdf(k, p + vec3<f32>(0.0, e, 0.0)) - col_sdf(k, p - vec3<f32>(0.0, e, 0.0)),
                    col_sdf(k, p + vec3<f32>(0.0, 0.0, e)) - col_sdf(k, p - vec3<f32>(0.0, 0.0, e)));
  let l = length(g);
  return select(vec3<f32>(0.0, 1.0, 0.0), g / max(l, 1e-9), l > 1e-9);
}

// How far a collider reaches from its middle (to skip the ones too far away to touch a blade).
fn col_reach(k: Collider) -> f32 {
  let shape = i32(k.a.w + 0.5);
  if (shape == 0) { return k.b.x; }
  if (shape == 1) { return length(k.b.xyz); }
  if (shape == 2) { return length(k.b.xy); }
  return length(max(abs(k.m0.xyz), abs(k.m1.xyz)) * k.b.xyz);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.sim.z)) { return; }
  let rt = RT[i];
  if (rt.w < 0.5) { return; }
  let b0 = BL[2u * i];
  let b1 = BL[2u * i + 1u];
  let pa = U.patches[u32(b0.z + 0.5)];
  let up = RN[i].xyz;
  var st = ST[i];
  let burnt = clamp(st.y, 0.0, 1.0);
  let seg = b1.x * (1.0 - (1.0 - STUBBLE) * burnt) / f32(POINTS - 1u);
  let lean = pa.phys.y * (0.3 + 1.5 * b0.w * b0.w);
  let r = 0.5 * b1.y + 0.003;       // how far it keeps off a surface (m)
  var x: array<vec3<f32>, POINTS>;
  var xp: array<vec3<f32>, POINTS>;
  for (var k = 0u; k < POINTS; k++) {
    x[k] = X[i * POINTS + k].xyz;
    xp[k] = XP[i * POINTS + k].xyz;
  }
  x[0] = rt.xyz;
  let dt = U.sim.x;
  let steps = u32(U.sim.y + 0.5);
  let keep = 1.0 - min(2.5 * dt, 0.5);              // (its own damping: 2.5 a second)
  let grav = vec3<f32>(0.0, -9.81 * pa.phys.w, 0.0);
  // the objects near enough to touch it this frame (where they are, give or take how far they go)
  var near = 0u;
  let mid = x[POINTS / 2u];
  for (var c = 0; c < i32(U.ccnt.x); c++) {
    let k = U.col[c];
    let reach = col_reach(k) + b1.x + (length(k.v.xyz) + length(k.o.xyz) * col_reach(k)) * dt * f32(steps) + 0.05;
    if (length(mid - k.a.xyz) < reach) { near |= 1u << u32(c); }
  }
  for (var s = 0u; s < steps; s++) {
    let t = U.air2.z + f32(s) * dt;
    // the air's drag across the blade, and gravity; where each point goes
    for (var k = 1u; k < POINTS; k++) {
      let p = x[k];
      let v = (p - xp[k]) / dt;
      let a4 = air_at(p, t);
      let ax = normalize(p - x[k - 1u] + vec3<f32>(0.0, 1e-7, 0.0));
      var rel = a4.xyz - v;
      rel -= ax * dot(rel, ax);
      var acc = pa.phys.z * (a4.w / 1.2) * length(rel) * rel;
      let al = length(acc);
      if (al > 400.0) { acc *= 400.0 / al; }
      acc += grav;
      let prev = xp[k];
      xp[k] = p;
      x[k] = p + (p - prev) * keep + acc * (dt * dt);
    }
    // back toward its rest shape, from the root up, every segment its length
    for (var k = 1u; k < POINTS; k++) {
      var pc = up;
      var pr = up;
      if (k > 1u) {
        pc = normalize(x[k - 1u] - x[k - 2u]);
        pr = rest_dir(up, b1.w, lean, k - 2u);
      }
      let goal = x[k - 1u] + turn_by(pr, pc, rest_dir(up, b1.w, lean, k - 1u)) * seg;
      let sk = pa.phys.x * (1.0 - 0.5 * f32(k - 1u) / f32(POINTS - 2u));
      x[k] = mix(x[k], goal, sk);
      let d = x[k] - x[k - 1u];
      x[k] = x[k - 1u] + d * (seg / max(length(d), 1e-7));
    }
    // out of the ground and the objects
    for (var k = 1u; k < POINTS; k++) {
      var p = x[k];
      if (U.gnd.x > 0.5 && p.y < U.gnd.y + r) { p.y = U.gnd.y + r; }
      if (near != 0u) {
        for (var c = 0; c < i32(U.ccnt.x); c++) {
          if (((near >> u32(c)) & 1u) == 0u) { continue; }
          let kk = U.col[c];
          let d = col_sdf(kk, p);
          if (d < r) { p += col_normal(kk, p, 0.01) * (r - d); }
        }
      }
      x[k] = p;
    }
  }
  // fire: heating in hot gas until it catches, then burning down
  if (pa.burn.x > 0.5) {
    let total = dt * f32(steps);
    if (st.z < 0.5 && st.y < 1.0) {
      let T = max(temp_at(x[1u]), temp_at(x[POINTS / 2u]));
      let ign = pa.burn.y;
      if (T > ign) {
        st.x += total * (T - ign) / max(1.0 - ign, 0.05) / max(pa.burn.z, 1e-3);
      } else {
        st.x = max(st.x - total * 0.5, 0.0);   // (out of the heat it cools again)
      }
      if (st.x >= 1.0) {
        st.x = 1.0;
        st.z = 1.0;
      }
    }
    if (st.z > 0.5) {
      st.y += total / max(pa.burn.w, 1e-3);
      if (st.y >= 1.0) {
        st.y = 1.0;
        st.z = 0.0;
      }
    } else if (st.y >= 1.0) {
      st.y = min(st.y + total / 4.0, 2.0);   // the stubble's embers dying out over four seconds
    }
  }
  for (var k = 0u; k < POINTS; k++) {
    X[i * POINTS + k] = vec4<f32>(x[k], 0.0);
    XP[i * POINTS + k] = vec4<f32>(xp[k], 0.0);
  }
  ST[i] = st;
}
