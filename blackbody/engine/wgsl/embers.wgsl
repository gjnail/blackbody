// Ember / spark particles, advected by the simulated air. They launch into a cone, fall under
// gravity and bounce off the ground and colliders.
// A: position (fire-local metres), life left (s)
// B: velocity (m/s), temperature (K)
// C: position at the previous step, age (s)
// D: visibility through smoke toward the camera, size (m), random seed, initial life
// E: colour of the emitter's colourant (rgb), how much it colours the particle (0..1)
//!include common.wgsl
//!include noise.wgsl
//!include meshsdf.wgsl
//!include emitters.wgsl

struct Params {
  g: Grid,             // simulation grid (org = grid corner in fire-local metres)
  sp: vec4<f32>,       // spawn start index, spawn count, particle count, seed
  life: vec4<f32>,     // lifetime (s), lifetime jitter (0..1), initial temperature (K), cooling (1/s)
  mot: vec4<f32>,      // drag (1/s), gravity (m/s^2), launch speed (m/s), launch spread (m/s)
  turb: vec4<f32>,     // turbulence (m/s^2), turbulence frequency (1/m), size min (m), size max (m)
  cam: vec4<f32>,      // camera position in grid cells; w = smoke extinction per unit soot per metre
  dir: vec4<f32>,      // launch direction (unit), cone half-angle (radians)
  hit: vec4<f32>,      // bounce, friction, hit colliders (1/0), _
  cnt: vec4<f32>,      // emitter count, ambient K, _, _
  em: array<Emitter, MAX_EMITTERS>,
};

@group(0) @binding(0) var vel: texture_3d<f32>;
@group(0) @binding(1) var scal: texture_3d<f32>;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var sdf: texture_3d<f32>;
@group(0) @binding(4) var atlas: texture_3d<f32>;
@group(0) @binding(5) var<storage, read_write> A: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> B: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> C: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> D: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> E: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> U: Params;

fn spawn_point(e: Emitter, r: vec3<f32>, r2: vec3<f32>, seed: u32) -> vec3<f32> {
  let shape = i32(e.a.w + 0.5);
  let ang = r.x * 6.2831853;
  if (shape == 3) {
    let rad = sqrt(r2.y) * e.b.x;
    let a2 = r2.x * 6.2831853;
    return mix(e.a.xyz, e.c.xyz, r.y) + vec3<f32>(cos(a2) * rad, (r2.z * 2.0 - 1.0) * e.b.x, sin(a2) * rad);
  }
  if (shape == 6) {
    // mesh: try random points in its box until one lands where the emitter releases fuel
    let s = max(e.b.xyz, vec3<f32>(1e-4));
    var best = e.a.xyz;
    var best_d = 1.0e9;
    for (var k = 0u; k < 16u; k++) {
      let rr = rand3(seed, 11u + k);
      let q = mix(e.m0.xyz, e.m1.xyz, rr);
      let w = e.a.xyz + yaw_to_world(q * s, e.k.z);
      let dd = emitter_sdf(e, w);
      if (dd < best_d) { best_d = dd; best = w; }
      if (dd <= 0.0) { break; }
    }
    return best;
  }
  var q = vec3<f32>(0.0);
  if (shape == 1) {
    q = (r * 2.0 - vec3<f32>(1.0)) * e.b.xyz;
  } else if (shape == 2 || shape == 5) {
    let rad = sqrt(r.y) * e.b.x;
    q = vec3<f32>(cos(ang) * rad, (r.z * 2.0 - 1.0) * e.b.y, sin(ang) * rad);
  } else if (shape == 4) {
    let rad = e.b.x + (r.y * 2.0 - 1.0) * e.b.y;
    q = vec3<f32>(cos(ang) * rad, (r.z * 2.0 - 1.0) * e.b.y, sin(ang) * rad);
  } else {
    // ellipsoid: uniform in the volume
    let z = r.y * 2.0 - 1.0;
    let sz = sqrt(max(1.0 - z * z, 0.0));
    let dir = vec3<f32>(sz * cos(ang), z, sz * sin(ang));
    q = dir * pow(r.z, 1.0 / 3.0) * e.b.xyz;
  }
  return e.a.xyz + yaw_to_world(q, e.k.z);
}

// A direction uniformly distributed over the cone of half-angle `half` around `axis`.
fn cone_dir(axis: vec3<f32>, half: f32, r: vec2<f32>) -> vec3<f32> {
  let ct = mix(1.0, cos(half), r.x);
  let st = sqrt(max(1.0 - ct * ct, 0.0));
  let phi = r.y * 6.2831853;
  let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(axis.y) > 0.99);
  let t1 = normalize(cross(up, axis));
  let t2 = cross(axis, t1);
  return axis * ct + (t1 * cos(phi) + t2 * sin(phi)) * st;
}

fn smoke_to_camera(pg: vec3<f32>, n: vec3<f32>) -> f32 {
  let to = U.cam.xyz - pg;
  let len = length(to);
  if (len < 1e-3) { return 1.0; }
  let dir = to / len;
  let stepc = 2.0;
  var od = 0.0;
  var t = 1.0;
  for (var i = 0; i < 48; i++) {
    let q = pg + dir * t;
    if (any(q < vec3<f32>(0.0)) || any(q > n) || t > len) { break; }
    od += textureSampleLevel(scal, lin, q / n, 0.0).z;
    t += stepc;
  }
  return exp(-od * stepc * U.g.n.w * U.cam.w);
}

fn sdf_at(c: vec3<i32>, d: vec3<i32>) -> f32 {
  return textureLoad(sdf, clamp(c, vec3<i32>(0), d - vec3<i32>(1)), 0).x;
}

// Collider distance (cells) at grid position pg, trilinear, with its gradient.
fn sdf_grad(pg: vec3<f32>, d: vec3<i32>) -> vec4<f32> {
  let t = pg - vec3<f32>(0.5);
  let i0 = vec3<i32>(floor(t));
  let f = t - floor(t);
  let s000 = sdf_at(i0, d);
  let s100 = sdf_at(i0 + vec3<i32>(1, 0, 0), d);
  let s010 = sdf_at(i0 + vec3<i32>(0, 1, 0), d);
  let s110 = sdf_at(i0 + vec3<i32>(1, 1, 0), d);
  let s001 = sdf_at(i0 + vec3<i32>(0, 0, 1), d);
  let s101 = sdf_at(i0 + vec3<i32>(1, 0, 1), d);
  let s011 = sdf_at(i0 + vec3<i32>(0, 1, 1), d);
  let s111 = sdf_at(i0 + vec3<i32>(1, 1, 1), d);
  let x00 = mix(s000, s100, f.x);
  let x10 = mix(s010, s110, f.x);
  let x01 = mix(s001, s101, f.x);
  let x11 = mix(s011, s111, f.x);
  let v = mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
  let gx = mix(mix(s100 - s000, s110 - s010, f.y), mix(s101 - s001, s111 - s011, f.y), f.z);
  let gy = mix(x10 - x00, x11 - x01, f.z);
  let gz = mix(x01, x11, f.y) - mix(x00, x10, f.y);
  return vec4<f32>(v, gx, gy, gz);
}

// Bounce off a surface with outward normal nrm: keep `bounce` of the speed into it (reversed) and
// lose `friction` of the sliding speed.
fn bounce_off(v: vec3<f32>, nrm: vec3<f32>) -> vec3<f32> {
  let vn = dot(v, nrm);
  if (vn >= 0.0) { return v; }
  let vt = v - nrm * vn;
  return vt * (1.0 - U.hit.y) - nrm * vn * U.hit.x;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let total = u32(U.sp.z);
  let i = id.x;
  if (i >= total) { return; }
  let n = U.g.n.xyz;
  let h = U.g.n.w;
  let dt = U.g.bc.w;
  var a = A[i];
  var b = B[i];
  var c = C[i];
  var d = D[i];
  var tint = E[i];
  let start = u32(U.sp.x);
  let rel = (i + total - start % total) % total;
  let cnt = i32(U.cnt.x);

  if (f32(rel) < U.sp.y && cnt > 0) {
    let seed = i * 7919u + u32(U.sp.w) * 104729u;
    let r0 = rand3(seed, 1u);
    let r1 = rand3(seed, 2u);
    let r2 = rand3(seed, 3u);
    let r3 = rand3(seed, 4u);
    // pick an emitter by its ember weight (its fuel, or a fair share for fuel-less spark sources)
    var total_w = 0.0;
    for (var k = 0; k < cnt; k++) { total_w += max(U.em[k].m2.w, 0.0); }
    var pick = r0.x * total_w;
    var ei = 0;
    for (var k = 0; k < cnt; k++) {
      pick -= max(U.em[k].m2.w, 0.0);
      if (pick <= 0.0) { ei = k; break; }
      ei = k;
    }
    let e = U.em[ei];
    let p = spawn_point(e, r1, r2, seed);
    let spread = (r2 * 2.0 - vec3<f32>(1.0)) * U.mot.w;
    let launch = cone_dir(U.dir.xyz, U.dir.w, r3.xy) * (U.mot.z * (0.4 + 0.6 * r0.y));
    let v = emitter_velocity(e, p) + launch + vec3<f32>(spread.x, spread.y * 0.3, spread.z);
    let lf = U.life.x * (1.0 - U.life.y * r0.z);
    a = vec4<f32>(p, lf);
    b = vec4<f32>(v, U.life.z * (0.85 + 0.3 * r1.x));
    c = vec4<f32>(p, 0.0);
    d = vec4<f32>(1.0, mix(U.turb.z, U.turb.w, r2.x * r2.x), r0.x, lf);
    let cm = max(e.col.r, max(e.col.g, e.col.b));
    tint = select(vec4<f32>(0.0), vec4<f32>(e.col.rgb / max(cm, 1e-6), smoothstep(0.0, 0.5, cm)), cm > 1e-6);
  } else if (a.w > 0.0) {
    let pg = (a.xyz - U.g.org.xyz) / h;
    var air = vec3<f32>(0.0);
    let inside = all(pg >= vec3<f32>(0.0)) && all(pg <= n);
    if (inside) { air = vel_at(vel, lin, pg, n); }
    // small, light embers follow the air closely; drag scales inversely with size
    let drag = U.mot.x * clamp(0.004 / max(d.y, 1e-4), 0.25, 4.0);
    var v = b.xyz;
    v += (air - v) * (1.0 - exp(-drag * dt));
    v.y -= U.mot.y * dt;
    if (U.turb.x > 0.0) {
      v += curl_noise(a.xyz * U.turb.y + vec3<f32>(0.0, -U.g.org.w * 0.8, d.z * 13.0)) * (U.turb.x * dt);
    }
    c = vec4<f32>(a.xyz, c.w + dt);
    // move in steps of at most half a cell when colliders are on, so fast sparks cannot pass
    // straight through a thin wall between two checks
    var p = a.xyz;
    var steps = 1;
    if (U.hit.z > 0.5) { steps = clamp(i32(ceil(length(v) * dt / (0.5 * h))), 1, 8); }
    let sdt = dt / f32(steps);
    for (var k = 0; k < steps; k++) {
      p += v * sdt;
      if (p.y < 0.0 && U.g.bc.z < 0.5) {
        p.y = 0.0;
        v = bounce_off(v, vec3<f32>(0.0, 1.0, 0.0));
      }
      if (U.hit.z > 0.5) {
        let q = (p - U.g.org.xyz) / h;
        if (all(q >= vec3<f32>(0.0)) && all(q <= n)) {
          let sg = sdf_grad(q, vec3<i32>(n));
          if (sg.x < 0.5 && dot(sg.yzw, sg.yzw) > 1e-8) {
            let nrm = normalize(sg.yzw);
            p += nrm * ((0.5 - sg.x) * h);
            v = bounce_off(v, nrm);
          }
        }
      }
    }
    a = vec4<f32>(p, a.w - dt);
    let amb = U.cnt.y;
    b = vec4<f32>(v, amb + (b.w - amb) * exp(-U.life.w * dt));
    if (inside) { d.x = smoke_to_camera(pg, n); } else { d.x = 1.0; }
  }
  A[i] = a;
  B[i] = b;
  C[i] = c;
  D[i] = d;
  E[i] = tint;
}
