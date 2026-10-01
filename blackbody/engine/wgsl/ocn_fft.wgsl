// One radix-2 Stockham pass of a 2D FFT along one axis, over every layer of a 3D texture at once.
// Each texel holds two complex numbers (xy, zw) transformed together. A thread does one butterfly:
// it reads two inputs half the length apart and writes two outputs p apart, so after log2(n)
// passes (p = 1, 2, 4 .. n/2) the result is in natural order. No normalisation: the inverse
// (sign +1) is the plain sum over the spectrum, which is what synthesising a sea needs.

struct Params {
  k: vec4<f32>,   // n, p (half the span of this pass), axis (0 x, 1 y), sign (+1 inverse, -1 forward)
};

@group(0) @binding(0) var src: texture_3d<f32>;
@group(0) @binding(1) var dst: texture_storage_3d<rgba32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = u32(U.k.x);
  let half = n / 2u;
  if (id.x >= half || id.y >= n) { return; }
  let p = u32(U.k.y);
  let i = id.x;
  let k = i & (p - 1u);
  let j = ((i - k) << 1u) + k;
  let ang = U.k.w * 3.14159265358979 * f32(k) / f32(p);
  let w = vec2<f32>(cos(ang), sin(ang));
  let l = i32(id.z);
  let r = i32(id.y);
  var a0 = vec2<i32>(i32(i), r);
  var a1 = vec2<i32>(i32(i + half), r);
  var o0 = vec2<i32>(i32(j), r);
  var o1 = vec2<i32>(i32(j + p), r);
  if (U.k.z > 0.5) {
    a0 = a0.yx;
    a1 = a1.yx;
    o0 = o0.yx;
    o1 = o1.yx;
  }
  let u0 = textureLoad(src, vec3<i32>(a0, l), 0);
  let u1 = textureLoad(src, vec3<i32>(a1, l), 0);
  let t = vec4<f32>(cmul(u1.xy, w), cmul(u1.zw, w));
  textureStore(dst, vec3<i32>(o0, l), u0 + t);
  textureStore(dst, vec3<i32>(o1, l), u0 - t);
}
