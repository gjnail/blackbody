// Dye for the renderer, pass 1: every packed particle adds its dye (absorption per metre rgb,
// scattering per metre) to the eight nearest cell centres of the simulation grid, with its weight.
// accumulator slots per cell: sum w, sum w dye (rgba).

const FX: f32 = 65536.0;

fn lin_id(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
  return gid.x + gid.y * nwg.x * 64u;
}

struct Params {
  n: vec4<f32>,   // simulation grid dims; w = particle count
};

@group(0) @binding(0) var<storage, read> packed: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> dye: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> acc: array<atomic<i32>>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.n.w)) { return; }
  let pk = packed[i];
  let n = vec3<i32>(U.n.xyz);
  let x = vec3<f32>(unpack2x16unorm(pk.x), unpack2x16unorm(pk.y).x) * U.n.xyz;
  let dp = dye[i];
  let d = vec4<f32>(unpack2x16float(dp.x), unpack2x16float(dp.y));
  let q = x - vec3<f32>(0.5);
  let fq = floor(q);
  let f = q - fq;
  let b = vec3<i32>(fq);
  for (var o = 0; o < 8; o++) {
    let oo = vec3<i32>(o & 1, (o >> 1) & 1, o >> 2);
    let node = b + oo;
    if (any(node < vec3<i32>(0)) || any(node >= n)) { continue; }
    let wv = mix(vec3<f32>(1.0) - f, f, vec3<f32>(oo));
    let w = wv.x * wv.y * wv.z;
    if (w <= 1e-6) { continue; }
    let s = u32(node.x + n.x * (node.y + n.y * node.z)) * 5u;
    atomicAdd(&acc[s], i32(round(w * FX)));
    if (any(d > vec4<f32>(0.0))) {
      atomicAdd(&acc[s + 1u], i32(round(w * d.x * FX)));
      atomicAdd(&acc[s + 2u], i32(round(w * d.y * FX)));
      atomicAdd(&acc[s + 3u], i32(round(w * d.z * FX)));
      atomicAdd(&acc[s + 4u], i32(round(w * d.w * FX)));
    }
  }
}
