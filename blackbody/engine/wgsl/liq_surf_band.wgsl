// Surface builder with a narrow-band liquid: the deep liquid (grid only, no particles) counts as
// full (mode 0, on the particle counts) and as deep inside (mode 1, on the deep flags), so the
// surface closes over it instead of showing the inner edge of the band.

struct Params {
  n: vec4<f32>,   // simulation grid dims; w = mode
  k: vec4<f32>,   // particles that count as full
};

@group(0) @binding(0) var<storage, read> band: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(U.n.xyz);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let i = u32(c.x + n.x * (c.y + n.y * c.z));
  if (band[i] == 0u) { return; }
  if (U.n.w < 0.5) { dst[i] = max(dst[i], u32(U.k.x)); } else { dst[i] = 1u; }
}
