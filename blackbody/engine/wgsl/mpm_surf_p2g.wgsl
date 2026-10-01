// Matter's surface, for drawing: each particle (in its 8-byte form: mpm_compact.wgsl) adds itself to the nodes round
// it within R, weighted by k = (1 - (d / R)^2)^3: the weight, the weighted offset to it (for the surface: Zhu and
// Bridson 2005) and its look.
//!include mpm_common.wgsl

struct Params {
  n: vec4<f32>,      // grid nodes (x, y, z), particles (count)
  r: vec4<f32>,      // R (grid units), _, _, _
  mats: array<MMat, MAX_MATS>,
};

@group(0) @binding(0) var<storage, read> C: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> S: array<atomic<i32>>;
@group(1) @binding(0) var<uniform> U: Params;

const SURF: u32 = 11u;          // slots per node: w, w offset (xyz), w albedo (rgb), w roughness, clear, sparkle, wrap
const FX_S: f32 = 65536.0;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.n.w)) { return; }
  let c = C[i];
  let a = unpack2x16unorm(c.x);
  let b = unpack2x16unorm(c.y);
  let tag = u32(round(b.y * 65535.0));
  let mat = tag >> 12u;
  if (mat >= 15u) { return; }
  let rnd = f32(tag & 4095u) / 4095.0;
  let n = vec3<i32>(U.n.xyz);
  let x = vec3<f32>(a.x, a.y, b.x) * U.n.xyz;
  let m = U.mats[mat];
  let R = U.r.x;
  // its look: the material's, its colour a little different from its neighbours'
  let shade = 1.0 + m.e.w * (rnd - 0.5);
  let alb = clamp(m.d.rgb * shade, vec3<f32>(0.0), vec3<f32>(1.0));
  let lo = vec3<i32>(ceil(x - vec3<f32>(R)));
  let hi = vec3<i32>(floor(x + vec3<f32>(R)));
  for (var z = lo.z; z <= hi.z; z++) {
    for (var y = lo.y; y <= hi.y; y++) {
      for (var xx = lo.x; xx <= hi.x; xx++) {
        let node = vec3<i32>(xx, y, z);
        if (any(node < vec3<i32>(0)) || any(node >= n)) { continue; }
        let off = x - vec3<f32>(node);
        let q = dot(off, off) / (R * R);
        if (q >= 1.0) { continue; }
        let wt = (1.0 - q) * (1.0 - q) * (1.0 - q);
        let k = nidx(node, n) * SURF;
        atomicAdd(&S[k], i32(round(wt * FX_S)));
        atomicAdd(&S[k + 1u], i32(round(wt * off.x * FX_S)));
        atomicAdd(&S[k + 2u], i32(round(wt * off.y * FX_S)));
        atomicAdd(&S[k + 3u], i32(round(wt * off.z * FX_S)));
        atomicAdd(&S[k + 4u], i32(round(wt * alb.r * FX_S)));
        atomicAdd(&S[k + 5u], i32(round(wt * alb.g * FX_S)));
        atomicAdd(&S[k + 6u], i32(round(wt * alb.b * FX_S)));
        atomicAdd(&S[k + 7u], i32(round(wt * m.d.w * FX_S)));
        atomicAdd(&S[k + 8u], i32(round(wt * m.e.x * FX_S)));
        atomicAdd(&S[k + 9u], i32(round(wt * m.e.y * FX_S)));
        atomicAdd(&S[k + 10u], i32(round(wt * m.e.z * FX_S)));
      }
    }
  }
}
