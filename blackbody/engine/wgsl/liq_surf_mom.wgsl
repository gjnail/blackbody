// Anisotropic surfacing, pass 1: the particles' moments per block of 2x2x2 simulation cells (count,
// sum of offsets from the block centre, sum of their outer products), for the local shape of the
// liquid around every particle. Fixed point, deterministic.

const FX_M: f32 = 65536.0;

fn lin_id(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
  return gid.x + gid.y * nwg.x * 64u;
}

struct Params {
  n: vec4<f32>,   // simulation grid dims; w = particle count
};

@group(0) @binding(0) var<storage, read> packed: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> mom: array<atomic<i32>>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.n.w)) { return; }
  let pk = packed[i];
  let p = vec3<f32>(unpack2x16unorm(pk.x), unpack2x16unorm(pk.y).x) * U.n.xyz;
  let nb = (vec3<i32>(U.n.xyz) + vec3<i32>(1)) / 2;
  let b = clamp(vec3<i32>(floor(p * 0.5)), vec3<i32>(0), nb - vec3<i32>(1));
  let o = p - (vec3<f32>(b) * 2.0 + vec3<f32>(1.0));
  let base = u32(b.x + nb.x * (b.y + nb.y * b.z)) * 10u;
  atomicAdd(&mom[base], 1);
  atomicAdd(&mom[base + 1u], i32(round(o.x * FX_M)));
  atomicAdd(&mom[base + 2u], i32(round(o.y * FX_M)));
  atomicAdd(&mom[base + 3u], i32(round(o.z * FX_M)));
  atomicAdd(&mom[base + 4u], i32(round(o.x * o.x * FX_M)));
  atomicAdd(&mom[base + 5u], i32(round(o.x * o.y * FX_M)));
  atomicAdd(&mom[base + 6u], i32(round(o.x * o.z * FX_M)));
  atomicAdd(&mom[base + 7u], i32(round(o.y * o.y * FX_M)));
  atomicAdd(&mom[base + 8u], i32(round(o.y * o.z * FX_M)));
  atomicAdd(&mom[base + 9u], i32(round(o.z * o.z * FX_M)));
}
