// Pack live particles for the frame cache and the surface builder: position as 16-bit fractions
// of the grid, velocity (m/s) and age (s) as half floats. 16 bytes per particle.
//!include common.wgsl
//!include liq_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,  // slot capacity
};

@group(0) @binding(0) var<storage, read> parts: array<Particle>;
@group(0) @binding(1) var<storage, read_write> packed: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> ctr: array<atomic<i32>>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.k.x)) { return; }
  let P = parts[i];
  if (P.p.w < 0.5) { return; }
  let q = clamp(P.p.xyz / vec3<f32>(gdim(U.g)), vec3<f32>(0.0), vec3<f32>(1.0));
  let slot = u32(atomicAdd(&ctr[2], 1));
  packed[slot] = vec4<u32>(pack2x16unorm(q.xy), pack2x16unorm(vec2<f32>(q.z, 0.0)),
                           pack2x16float(P.v.xy), pack2x16float(vec2<f32>(P.v.z, min(P.v.w, 60000.0))));
}
