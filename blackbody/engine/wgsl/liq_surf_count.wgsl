// Surface builder, pass 0: particles per simulation cell, so the builder can find the deep inside
// of the liquid (which cannot affect the surface) and skip it.

fn lin_id(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
  return gid.x + gid.y * nwg.x * 64u;
}

struct Params {
  n: vec4<f32>,    // simulation grid dims; w = particle count
};

@group(0) @binding(0) var<storage, read> packed: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> occ: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.n.w)) { return; }
  let pk = packed[i];
  let q = vec3<f32>(unpack2x16unorm(pk.x), unpack2x16unorm(pk.y).x);
  let n = vec3<i32>(U.n.xyz);
  let c = clamp(vec3<i32>(q * U.n.xyz), vec3<i32>(0), n - vec3<i32>(1));
  atomicAdd(&occ[u32(c.x + n.x * (c.y + n.y * c.z))], 1u);
}
