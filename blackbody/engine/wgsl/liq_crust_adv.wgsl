// Molten liquids: the crust coordinate, a field on the simulation grid carried along by the flow. The
// crust drawn on the surface (liq_lava_shade.wgsl) is a pattern of plates and cracks laid out in this
// coordinate, so it rides the flow instead of sliding over it, and where the flow squeezes the crust it
// wrinkles into ropes.
//
// It lives on the grid, not on the particles: a liquid's particles keep changing places at its surface
// (some come up, some go under), so neighbours on the surface can have come from anywhere and their own
// coordinates would scatter the pattern. The grid's velocity is smooth, and so is what it carries.
//
// A crust only forms once the melt is out in the air, and it is only what happens to it after that which
// shapes it: melt pouring out of a vent spreads, slows and churns many times over before it skins, and the
// crust knows nothing of that. So where no crust has formed yet the coordinate is simply the place itself,
// and at each of a series of moments (irregular, a fraction of a second apart) all the melt that has been
// out in the air since the last one takes the coordinate it has then, the place it is, for good. Each batch
// of new crust starts as an exact, undistorted copy of where the skin was and from then on rides the flow;
// batches meet along seams (the coordinate jumps across them), drawn as cracks: crust formed at different
// times. Melt a crack or a breakout brings up later skins over with a pattern of its own.
//
// Alongside it, the skin's age: how long the melt at the surface has been out in the air (it ages where
// the liquid is thin, at its surface, and is carried along; the fresh melt, as hot as it was poured, is
// new).
//
// cx: xyz = the coordinate less the place (cells), w = 1 once crust has formed. ca: the skin's age (s).
// cv: for the renderer, the coordinate less the place (cells) and the skin's age.
//!include common.wgsl
//!include liq_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,  // a shared moment this step (1/0), seconds out in the air the melt needs by then, rest density
};

@group(0) @binding(0) var vel: texture_3d<f32>;
@group(0) @binding(1) var dens: texture_3d<f32>;
@group(0) @binding(2) var heat: texture_3d<f32>;
@group(0) @binding(3) var cx_in: texture_3d<f32>;
@group(0) @binding(4) var ca_in: texture_3d<f32>;
@group(0) @binding(5) var cx_out: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var ca_out: texture_storage_3d<r32float, write>;
@group(0) @binding(7) var cv_out: texture_storage_3d<rgba32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

// Trilinear sample of a cell-centred field at x (cells), clamped to the grid.
fn samp4(t: texture_3d<f32>, x: vec3<f32>, n: vec3<i32>) -> vec4<f32> {
  let q = x - vec3<f32>(0.5);
  let b = vec3<i32>(floor(q));
  let f = q - floor(q);
  var s = vec4<f32>(0.0);
  for (var o = 0; o < 8; o++) {
    let oo = vec3<i32>(o & 1, (o >> 1) & 1, o >> 2);
    let c = clamp(b + oo, vec3<i32>(0), n - vec3<i32>(1));
    let wv = mix(vec3<f32>(1.0) - f, f, vec3<f32>(oo));
    s += wv.x * wv.y * wv.z * textureLoad(t, c, 0);
  }
  return s;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let h = U.g.n.w;
  let dt = U.g.bc.w;
  // back along the flow (midpoint rule) to where this place's material was a step ago
  let x = vec3<f32>(c) + vec3<f32>(0.5);
  let v1 = mac_vel(vel, x, n);
  let v2 = mac_vel(vel, x - v1 * (0.5 * dt / h), n);
  let xb = x - v2 * (dt / h);
  let s = samp4(cx_in, xb, n);
  var age = samp4(ca_in, xb, n).x;
  var d = s.xyz - v2 * (dt / h);
  var formed = s.w;
  // the skin ages out in the air (where the liquid is thin: its surface); fresh melt, as hot as poured, is new
  let rho = textureLoad(dens, c, 0).x / max(U.k.z, 1e-6);
  if (rho > 0.05) { age += dt * (1.0 - smoothstep(0.55, 0.95, rho)); }
  if (textureLoad(heat, c, 0).x > 0.97 && rho > 0.05) { age = 0.0; formed = 0.0; }
  if (U.k.x > 0.5 && formed < 0.5 && age >= U.k.y) { formed = 1.0; }
  if (formed < 0.5) { d = vec3<f32>(0.0); }
  textureStore(cx_out, c, vec4<f32>(d, formed));
  textureStore(ca_out, c, vec4<f32>(age, 0.0, 0.0, 0.0));
  textureStore(cv_out, c, vec4<f32>(d, age));
}
