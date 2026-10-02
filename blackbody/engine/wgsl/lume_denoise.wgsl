// Lume's denoiser: the grain of the passes added up so far (stage.wgsl ACC) smoothed away, keeping edges, textures and
// shadow edges. An edge-avoiding à-trous wavelet filter (Dammertz et al. 2010) on the light alone:
// - demod: the light reaching each surface (the pixel over its albedo), so textures are not blurred, only lighting;
// - atrous: five passes of a 5x5 B3-spline kernel, its taps 1, 2, 4, 8, 16 pixels apart, each tap weighed by how alike
//   the two surfaces are (facing, distance) and how alike their light is for how noisy it still is: each pixel's own
//   spread over the passes, smoothed over its neighbours (a few passes' spread is itself noisy), and carried through
//   the levels as SVGF does (each level's spread is what its average of the taps leaves), so as the grain goes the
//   filter keeps closer to the picture: clean pixels and a thin converged highlight keep their detail;
// - remod: times the albedo again, back into the stage's picture. Pixels the camera saw no surface at are left alone.
//!include common.wgsl

struct Params {
  dims: vec4<f32>,   // width, height, passes so far, the tap spacing (px)
};

@group(0) @binding(0) var<storage, read> ACC: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> AOV: array<vec4<f32>>;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<storage, read> VIN: array<f32>;          // the light's variance at the level before
@group(0) @binding(5) var<storage, read_write> VOUT: array<f32>;   // and after this one
@group(1) @binding(0) var<uniform> U: Params;

fn idx(p: vec2<i32>) -> u32 { return u32(p.y) * u32(U.dims.x) + u32(p.x); }

fn albedo(k: u32) -> vec3<f32> { return max(AOV[2u * k].rgb / U.dims.z, vec3<f32>(0.02)); }

fn normal(k: u32) -> vec3<f32> { return AOV[2u * k + 1u].xyz / U.dims.z; }

fn dist(k: u32) -> f32 { return AOV[2u * k].w / U.dims.z; }

// How noisy the average of the passes still is at pixel k: its brightness's spread over the passes, over their number.
fn sigma(k: u32) -> f32 {
  let n = U.dims.z;
  let mean = luma(ACC[k].rgb / n);
  let m2 = AOV[2u * k + 1u].w / n;
  return sqrt(max(m2 - mean * mean, 0.0) / max(n, 1.0));
}

// The light's variance at pixel k (its spread over the passes, in the light's own units: over its albedo).
fn light_var(k: u32) -> f32 {
  let s = sigma(k) / max(luma(albedo(k)), 0.02);
  return s * s;
}

@compute @workgroup_size(8, 8, 1)
fn demod(@builtin(global_invocation_id) id: vec3<u32>) {
  let p = vec2<i32>(id.xy);
  if (f32(p.x) >= U.dims.x || f32(p.y) >= U.dims.y) { return; }
  let k = idx(p);
  // the variance, smoothed 3x3 (1 2 1)
  let kv = array<f32, 3>(0.25, 0.5, 0.25);
  var v = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let q = clamp(p + vec2<i32>(dx, dy), vec2<i32>(0), vec2<i32>(i32(U.dims.x) - 1, i32(U.dims.y) - 1));
      v += kv[dx + 1] * kv[dy + 1] * light_var(idx(q));
    }
  }
  VOUT[k] = v;
  let c = ACC[k] / U.dims.z;
  if (dot(normal(k), normal(k)) < 0.25) {
    textureStore(dst, p, c);
    return;
  }
  textureStore(dst, p, vec4<f32>(c.rgb / albedo(k), c.a));
}

@compute @workgroup_size(8, 8, 1)
fn atrous(@builtin(global_invocation_id) id: vec3<u32>) {
  let p = vec2<i32>(id.xy);
  if (f32(p.x) >= U.dims.x || f32(p.y) >= U.dims.y) { return; }
  let k = idx(p);
  let c0 = textureLoad(src, p, 0);
  let n0 = normal(k);
  if (dot(n0, n0) < 0.25) {
    textureStore(dst, p, c0);
    VOUT[k] = VIN[k];
    return;
  }
  let nn0 = normalize(n0);
  let z0 = dist(k);
  let l0 = luma(c0.rgb);
  // the light's spread here at this level, in the light's own units (its picture value over its albedo)
  let s0 = sqrt(max(VIN[k], 0.0));
  let step = i32(U.dims.w);
  let kern = array<f32, 3>(0.375, 0.25, 0.0625);
  var sum = vec3<f32>(0.0);
  var wsum = 0.0;
  var vsum = 0.0;
  for (var dy = -2; dy <= 2; dy++) {
    for (var dx = -2; dx <= 2; dx++) {
      let q = p + vec2<i32>(dx, dy) * step;
      if (q.x < 0 || q.y < 0 || f32(q.x) >= U.dims.x || f32(q.y) >= U.dims.y) { continue; }
      let kq = idx(q);
      let nq = normal(kq);
      if (dot(nq, nq) < 0.25) { continue; }
      let cq = textureLoad(src, q, 0);
      let wn = pow(max(dot(nn0, normalize(nq)), 0.0), 64.0);
      let wz = exp(-abs(dist(kq) - z0) / max(0.02 * z0 * f32(step), 1e-4));
      let wl = exp(-abs(luma(cq.rgb) - l0) / (3.0 * s0 + 1e-4));
      let w = kern[abs(dx)] * kern[abs(dy)] * wn * wz * wl;
      sum += cq.rgb * w;
      wsum += w;
      vsum += w * w * VIN[kq];
    }
  }
  textureStore(dst, p, vec4<f32>(sum / max(wsum, 1e-8), c0.a));
  VOUT[k] = vsum / max(wsum * wsum, 1e-16);
}

@compute @workgroup_size(8, 8, 1)
fn remod(@builtin(global_invocation_id) id: vec3<u32>) {
  let p = vec2<i32>(id.xy);
  if (f32(p.x) >= U.dims.x || f32(p.y) >= U.dims.y) { return; }
  let k = idx(p);
  let c = textureLoad(src, p, 0);
  if (dot(normal(k), normal(k)) < 0.25) {
    textureStore(dst, p, c);
    return;
  }
  textureStore(dst, p, vec4<f32>(c.rgb * albedo(k), c.a));
}
