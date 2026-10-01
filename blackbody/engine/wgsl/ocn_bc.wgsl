// The sea at the simulation's edges: per column of cells, the open water's surface (cells above
// the grid floor: the level plus the waves the grid can carry) and the orbital velocity at the
// surface (m/s), from the 'sim' spectrum after the inverse FFT (ocn_spectrum.wgsl): per layer
// (h + i ux, uz + i uy) on that layer's tile. Without waves (no layers) it is the flat level. A surge
// (a solitary long wave or a bore front) rides on top: a shallow-water wave, its water moving along
// with it at u = c h / (d + h).
//!include common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,                // level (cells), layers, FFT size, _
  tile: array<vec4<f32>, 2>,   // 1 / tile size (1/m) of each layer
  sg: vec4<f32>,               // surge: height (m, 0 none), 2 / length (1/m), direction (x, z)
  sg2: vec4<f32>,              // surge: crest position along its direction (m), kind (1 bore, 2 tsunami), speed (m/s), depth (m)
};

@group(0) @binding(0) var sea_t: texture_3d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn wrap(i: vec2<i32>, n: i32) -> vec2<i32> { return ((i % vec2<i32>(n)) + vec2<i32>(n)) % vec2<i32>(n); }

// Catmull-Rom weights for the four texels around fraction f.
fn cr(f: f32) -> vec4<f32> {
  let f2 = f * f;
  let f3 = f2 * f;
  return vec4<f32>(-0.5 * f3 + f2 - 0.5 * f, 1.5 * f3 - 2.5 * f2 + 1.0, -1.5 * f3 + 2.0 * f2 + 0.5 * f, 0.5 * f3 - 0.5 * f2);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let d = gdim(U.g);
  if (i32(id.x) >= d.x || i32(id.y) >= d.z) { return; }
  let x = U.g.org.xz + (vec2<f32>(id.xy) + vec2<f32>(0.5)) * U.g.n.w;
  let n = i32(U.k.z);
  var s = vec4<f32>(0.0);
  for (var l = 0; l < i32(U.k.y); l++) {
    let inv_l = U.tile[l / 4][l % 4];
    // bicubic: the shortest waves the grid carries are only a few texels long
    let p = fract(x * inv_l) * f32(n) - vec2<f32>(0.5);
    let i0 = vec2<i32>(floor(p));
    let f = p - floor(p);
    let wx = cr(f.x);
    let wz = cr(f.y);
    for (var j = 0; j < 4; j++) {
      var row = vec4<f32>(0.0);
      for (var i = 0; i < 4; i++) {
        row += wx[i] * textureLoad(sea_t, vec3<i32>(wrap(i0 + vec2<i32>(i - 1, j - 1), n), l), 0);
      }
      s += wz[j] * row;
    }
  }
  if (U.sg.x > 0.0) {
    let q = (dot(x, U.sg.zw) - U.sg2.x) * U.sg.y;
    let th = tanh(clamp(q, -20.0, 20.0));
    let sech2 = 1.0 - th * th;
    var eta = U.sg.x * sech2;
    var deta = -2.0 * U.sg.x * U.sg.y * sech2 * th;   // d eta / d(distance along its direction)
    if (U.sg2.y > 0.5) {
      eta = 0.5 * U.sg.x * (1.0 - th);
      deta = -0.5 * U.sg.x * U.sg.y * sech2;
    }
    if (U.sg2.y > 1.5) {
      // a tsunami: the trough the sea draws back into runs three lengths ahead of the front
      let tp = tanh(clamp((q - 6.0) / 3.0, -20.0, 20.0));
      let s2p = 1.0 - tp * tp;
      eta -= 0.6 * U.sg.x * s2p;
      deta += 0.4 * U.sg.x * U.sg.y * s2p * tp;
    }
    let c = U.sg2.z;
    let uh = c * eta / max(U.sg2.w + eta, 0.05);
    s += vec4<f32>(eta, uh * U.sg.z, uh * U.sg.w, -c * deta);
  }
  // s = (h, ux, uz, uy)
  textureStore(dst, vec2<i32>(id.xy), vec4<f32>(U.k.x + s.x / U.g.n.w, s.y, s.w, s.z));
}
