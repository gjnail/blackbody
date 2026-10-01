// Matter: each particle as 8 bytes for drawing (and the frame cache): its position as 16-bit fractions of the grid,
// its material (4 bits; 15: gone) and its own random number (12 bits).
//!include mpm_common.wgsl

struct Params {
  n: vec4<f32>,      // grid nodes (x, y, z), particles (count)
};

@group(0) @binding(0) var<storage, read> P: array<MParticle>;
@group(0) @binding(1) var<storage, read_write> C: array<vec2<u32>>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.n.w)) { return; }
  let p = P[i];
  let f = clamp(p.x.xyz / U.n.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
  let mat = select(u32(p.x.w + 0.5), 15u, p.x.w < 0.0);
  let tag = f32(mat * 4096u + u32(clamp(p.c0.w, 0.0, 1.0) * 4095.0)) / 65535.0;
  C[i] = vec2<u32>(pack2x16unorm(f.xy), pack2x16unorm(vec2<f32>(f.z, tag)));
}
