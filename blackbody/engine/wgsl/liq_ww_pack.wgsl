// Pack live whitewater particles for the frame cache and the surface builder: position as 16-bit
// fractions of the grid, then the class and the fraction of life left in one 16-bit value
// ((class + life fraction) / 3), then the velocity as half floats. 16 bytes per particle.
//!include common.wgsl

fn lin_id(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
  return gid.x + gid.y * nwg.x * 64u;
}

struct Params {
  g: Grid,
  k: vec4<f32>,  // capacity, full life (s)
};

@group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> B: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> packed: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> wctr: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.k.x)) { return; }
  let a = A[i];
  if (a.w <= 0.0) { return; }
  let q = clamp(a.xyz / vec3<f32>(gdim(U.g)), vec3<f32>(0.0), vec3<f32>(1.0));
  let code = (B[i].w + clamp(a.w / max(U.k.y, 1e-3), 0.0, 0.999)) / 3.0;
  let slot = atomicAdd(&wctr[1], 1u);
  let v = clamp(B[i].xyz, vec3<f32>(-6.0e4), vec3<f32>(6.0e4));
  packed[slot] = vec4<u32>(pack2x16unorm(q.xy), pack2x16unorm(vec2<f32>(q.z, code)),
                           pack2x16float(v.xy), pack2x16float(vec2<f32>(v.z, 0.0)));
}
