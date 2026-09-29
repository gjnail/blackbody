// Whitewater: spray, foam and bubbles, as secondary particles that ride the liquid but do not
// affect it. Each step a particle is classed by the liquid density where it is: in the air it is
// spray and flies ballistically; at the surface it is foam, carried by the liquid and fading; inside
// it is a bubble, buoyant and dragged along by the flow. Spray that lands on a wet surface becomes
// foam, on a dry one it is gone; everything is gone when it leaves the box.
//
// A: position (cells), life left (s)
// B: velocity (m/s), class (0 spray, 1 foam, 2 bubble)
//!include common.wgsl
//!include liq_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,  // capacity, rest density (particles), gravity (m/s^2), bubble buoyancy (m/s^2)
  d: vec4<f32>,  // spray air drag (1/s), bubble drag (1/s), foam density band low, high (x rest density)
  w: vec4<f32>,  // wind (m/s, grid axes), foam drift (fraction of the wind speed)
};

@group(0) @binding(0) var<storage, read_write> A: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> B: array<vec4<f32>>;
@group(0) @binding(2) var vnew: texture_3d<f32>;
@group(0) @binding(3) var dens: texture_3d<f32>;
@group(0) @binding(4) var sdf: texture_3d<f32>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.k.x)) { return; }
  var a = A[i];
  if (a.w <= 0.0) { return; }
  var b = B[i];
  let n = gdim(U.g);
  let nf = vec3<f32>(n);
  let dt = U.g.bc.w;
  let ih = 1.0 / U.g.n.w;
  let x = a.xyz;
  let rho = interp_cell(dens, x, n).x / U.k.y;
  let fl = mac_vel(vnew, x, n);
  var v = b.xyz;
  var cls = 0.0;
  if (rho < U.d.z) {
    // spray
    v.y -= U.k.z * dt;
    v = U.w.xyz + (v - U.w.xyz) * exp(-U.d.x * dt);
    a.w -= 0.1 * dt;
  } else if (rho < U.d.w) {
    // foam: carried by the surface, fading
    cls = 1.0;
    v = fl + vec3<f32>(U.w.x, 0.0, U.w.z) * U.w.w;
    a.w -= dt;
  } else {
    // bubble: rises, dragged along by the liquid
    cls = 2.0;
    v.y += U.k.w * dt;
    v += (fl - v) * (1.0 - exp(-U.d.y * dt));
  }
  var p = x + v * dt * ih;
  // solids and the ground
  let s = interp_cell(sdf, p, n);
  var hit = s.x < 0.1;
  if (U.g.bc.z < 0.5 && p.y < 0.1) { hit = true; }
  if (hit) {
    if (cls < 0.5) {
      // spray landing on a wet surface turns to foam there; on dry ground it is gone
      if (rho > 0.03) {
        p = x;
        v = fl;
        cls = 1.0;
        a.w = min(a.w, 1.0);
      } else {
        a.w = 0.0;
      }
    } else {
      p = x;
      v = vec3<f32>(0.0);
    }
  }
  if (any(p < vec3<f32>(0.0)) || any(p > nf)) { a.w = 0.0; }
  A[i] = vec4<f32>(p, a.w);
  B[i] = vec4<f32>(v, cls);
}
