// One step of the sea's foam. The foam lives on the largest cascade's tile, in a frame that drifts
// with the water (the current, and the wind's pull on the surface), so it never has to be resampled
// to move: each step reads how far the surface under each texel has folded over, from the two larger
// cascades together (whitecaps form where the crests of those scales pile up; ripples folding over
// make no foam).
//
//   x  fresh whitecap foam: born where the crest folds over (the Jacobian of the choppy
//      displacement under the threshold), fades fast
//   y  aged foam: what fresh foam leaves as it thins out; lasts much longer, spreads, and is drawn
//      out along the wind into streaks
//   z  bubbles churned under the breaking crest (a pale glow under the surface), fade in a second
//   w  the height of the surface there (m)
//
// The fold comes from the 'jac' spectrum (ocn_spectrum.wgsl), per cascade (ddx/dx, ddz/dz, ddx/dz, h).
// Near the tile's own copy around the origin the foam matches the finer cascades exactly; further
// copies repeat it (where it is only seen small).

struct Params {
  k: vec4<f32>,       // foam texels across, step (s), fresh foam life (s), aged foam life (s)
  m: vec4<f32>,       // fold threshold (Jacobian), bubble life (s), spread along the wind (texels per step), start (1: from nothing)
  wind: vec4<f32>,    // wind direction (unit x, z), cross-wind spread share, spectrum texels across
  off: vec4<f32>,     // where the foam frame has drifted to (m, x z); tile of the largest cascade (m); spread of the Jacobian
  tile: vec4<f32>,    // 1 / tile size of each cascade (1/m)
};

@group(0) @binding(0) var jac_t: texture_3d<f32>;
@group(0) @binding(1) var prev_t: texture_3d<f32>;
@group(0) @binding(2) var dst: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn wrap(i: vec2<i32>, n: i32) -> vec2<i32> { return ((i % vec2<i32>(n)) + vec2<i32>(n)) % vec2<i32>(n); }

// Cascade l's fold terms at x (m), bilinear on its tile.
fn jac_at(x: vec2<f32>, l: i32) -> vec4<f32> {
  let n = i32(U.wind.w);
  let p = fract(x * U.tile[l]) * f32(n) - vec2<f32>(0.5);
  let i0 = vec2<i32>(floor(p));
  let f = p - floor(p);
  let a = textureLoad(jac_t, vec3<i32>(wrap(i0, n), l), 0);
  let b = textureLoad(jac_t, vec3<i32>(wrap(i0 + vec2<i32>(1, 0), n), l), 0);
  let c = textureLoad(jac_t, vec3<i32>(wrap(i0 + vec2<i32>(0, 1), n), l), 0);
  let d = textureLoad(jac_t, vec3<i32>(wrap(i0 + vec2<i32>(1, 1), n), l), 0);
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn prev_at(q: vec2<f32>, n: i32) -> vec4<f32> {
  let p = q - vec2<f32>(0.5);
  let i0 = vec2<i32>(floor(p));
  let f = p - floor(p);
  let a = textureLoad(prev_t, vec3<i32>(wrap(i0, n), 0), 0);
  let b = textureLoad(prev_t, vec3<i32>(wrap(i0 + vec2<i32>(1, 0), n), 0), 0);
  let c = textureLoad(prev_t, vec3<i32>(wrap(i0 + vec2<i32>(0, 1), n), 0), 0);
  let d = textureLoad(prev_t, vec3<i32>(wrap(i0 + vec2<i32>(1, 1), n), 0), 0);
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = i32(U.k.x);
  if (i32(id.x) >= n || i32(id.y) >= n) { return; }
  let q = vec2<f32>(id.xy) + vec2<f32>(0.5);
  // the undisplaced point of the sea under this texel now
  let x0 = q / f32(n) * U.off.z + U.off.xy;
  // (ripples folding over make no foam: the finest cascade is left out)
  let s = jac_at(x0, 0) + jac_at(x0, 1);
  let J = (1.0 + s.x) * (1.0 + s.y) - s.z * s.z;
  let thr = U.m.x;
  let sig = U.off.w;
  let inj = smoothstep(thr + 0.3 * sig, thr - 0.6 * sig, J) * step(-10.0, thr);
  let dt = U.k.y;
  let df = exp(-dt / max(U.k.z, 1e-3));
  let da = exp(-dt / max(U.k.w, 1e-3));
  let db = exp(-dt / max(U.m.y, 1e-3));
  var prev = vec4<f32>(0.0);
  var aged_nb = 0.0;
  if (U.m.w < 0.5) {
    prev = textureLoad(prev_t, vec3<i32>(vec2<i32>(id.xy), 0), 0);
    // aged foam spreads, mostly along the wind
    let along = U.wind.xy * U.m.z;
    let across = vec2<f32>(-U.wind.y, U.wind.x) * (U.m.z * U.wind.z);
    aged_nb = 0.25 * (prev_at(q + along, n).y + prev_at(q - along, n).y + prev_at(q + across, n).y + prev_at(q - across, n).y);
  }
  let fresh = max(prev.x * df, inj);
  // fresh foam thinning out leaves some aged foam: on the sea its patches cover two or three times
  // what the breaking crests do
  let left = prev.x * (1.0 - df) * 0.12;
  let aged = min(mix(prev.y, aged_nb, 0.5) * da + left, 1.2);
  let bubbles = max(prev.z * db, inj * 0.9);
  textureStore(dst, vec3<i32>(vec2<i32>(id.xy), 0), vec4<f32>(fresh, aged, bubbles, s.w));
}
