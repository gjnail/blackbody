// Waves on the open-water layer (Tessendorf's eWave): the surface height h and velocity potential
// phi of small waves spreading from the box (wakes, the rings of a splash, a wave thrown off a hull),
// in one complex field h + i phi.
//
//   pre:  in space. Inside the box the simulation's own surface is the truth (its offset from the sea
//         is laid in, fading in over a few cells); islands and piers hold the water still; a sponge
//         round the layer's edge takes waves out before they could wrap round; nothing on dry land.
//   prop: in wavenumber space, exact for linear waves on water of the sea's depth: every component
//         turns at its own frequency (omega^2 = g k tanh(k d)), carried by the current, and the
//         shortest ones fade a little.

struct Params {
  L: Layer,
  k: vec4<f32>,   // sea depth (m), gravity, current x, z (m/s)
  m: vec4<f32>,   // sponge rate (1/s), damping of short waves (m^2/s), gain on the box's offset, the box's
                  // steady offset (m: how far its surface sits off the sea's on the whole; not a wave)
};

@group(0) @binding(0) var src: texture_3d<f32>;        // h + i phi (xy)
@group(0) @binding(1) var cols: texture_2d<f32>;       // the box's columns (ocl_cols.wgsl)
@group(0) @binding(2) var dep: texture_2d<f32>;        // the sea bed (ocl_depth.wgsl)
@group(0) @binding(3) var dst: texture_storage_3d<rgba32float, write>;
@group(1) @binding(0) var<uniform> U: Params;
//!include ocl_common.wgsl

// The box's surface offset from the sea at x (m, bilinear over its columns), and how much of it is known.
fn box_offset(x: vec2<f32>) -> vec2<f32> {
  let n = vec2<i32>(i32(U.L.bdim.x), i32(U.L.bdim.y));
  let q = box_coord(x) - vec2<f32>(0.5);
  let i0 = vec2<i32>(floor(q));
  let f = q - floor(q);
  var s = 0.0;
  var w = 0.0;
  for (var j = 0; j < 4; j++) {
    let o = vec2<i32>(j & 1, j >> 1);
    let cc = clamp(i0 + o, vec2<i32>(0), n - vec2<i32>(1));
    let v = textureLoad(cols, cc, 0).x;
    let ww = mix(1.0 - f.x, f.x, f32(o.x)) * mix(1.0 - f.y, f.y, f32(o.y));
    if (v > -1.0e3) { s += ww * v; w += ww; }
  }
  return vec2<f32>(s / max(w, 1e-6), w);
}

@compute @workgroup_size(8, 8, 1)
fn pre(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec2<i32>(id.xy);
  if (c.x >= lay_n() || c.y >= lay_n()) { return; }
  var v = textureLoad(src, vec3<i32>(c, 0), 0).xy;
  let x = lay_pos(c);
  let bs = box_share(x);
  if (bs > 0.0) {
    let o = box_offset(x);
    v.x = mix(v.x, (o.x - U.m.w) * U.m.z, bs * o.y);
  }
  let d = textureLoad(dep, c, 0);
  if (d.y <= 0.0 || d.x <= 0.02) { v = vec2<f32>(0.0); }
  v *= exp(-U.m.x * edge_ramp(c) * U.L.bdim.w);
  textureStore(dst, vec3<i32>(c, 0), vec4<f32>(v, 0.0, 0.0));
}

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

@compute @workgroup_size(8, 8, 1)
fn prop(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = lay_n();
  let c = vec2<i32>(id.xy);
  if (c.x >= n || c.y >= n) { return; }
  let F = textureLoad(src, vec3<i32>(c, 0), 0).xy;
  let Fm = textureLoad(src, vec3<i32>(wrapi(-c, n), 0), 0).xy;
  // the two real fields' spectra: H(k) = (F + conj F(-k)) / 2, P(k) = (F - conj F(-k)) / 2i
  let H = 0.5 * (F + vec2<f32>(Fm.x, -Fm.y));
  let dP = 0.5 * (F - vec2<f32>(Fm.x, -Fm.y));
  let P = vec2<f32>(dP.y, -dP.x);
  let ki = vec2<f32>(select(c, c - vec2<i32>(n), c >= vec2<i32>(n / 2)));
  let kv = ki * (6.28318530718 / (f32(n) * U.L.map.z));
  let k = length(kv);
  var Hn = vec2<f32>(0.0);
  var Pn = P;
  if (k > 0.0) {
    let K = k * tanh(min(k * U.k.x, 20.0));
    let w = sqrt(U.k.y * K);
    let dt = U.L.bdim.w;
    let cs = cos(w * dt);
    let sn = sin(w * dt);
    Hn = H * cs + P * (K / w * sn);
    Pn = P * cs - H * (U.k.y / w * sn);
    // carried by the current, and the shortest waves fade
    let ph = -dot(kv, U.k.zw) * dt;
    let e = vec2<f32>(cos(ph), sin(ph)) * exp(-U.m.y * k * k * dt);
    Hn = cmul(Hn, e);
    Pn = cmul(Pn, e);
  }
  // back into one field (the forward and inverse transforms together scale by n^2)
  let G = (Hn + vec2<f32>(-Pn.y, Pn.x)) / f32(n * n);
  textureStore(dst, vec3<i32>(c, 0), vec4<f32>(G, 0.0, 0.0));
}
