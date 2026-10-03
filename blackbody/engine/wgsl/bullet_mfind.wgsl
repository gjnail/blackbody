// Bullets in sand, snow, mud, jelly and clay: where the bullets' paths through this frame cross the matter
// (engine/bullet_media.py). For each path (a segment A + s u, 0 <= s <= L), every particle within r of it marks the
// bin of the path it lies in (NB bins along it), with a bit for its material (its slot): the bullet meets the matter
// where it really is, and as what it is.
//!include mpm_common.wgsl

const NB: u32 = 256u;     // bins along a path
const MAX_PATHS: u32 = 8u;

struct FindU {
  n: vec4<f32>,                         // grid nodes (x, y, z), particles (count)
  o: vec4<f32>,                         // the grid's corner (fire-local m), node spacing (m)
  k: vec4<f32>,                         // paths (count), _, _, _
  pa: array<vec4<f32>, MAX_PATHS>,      // A (fire-local m), L (m)
  pu: array<vec4<f32>, MAX_PATHS>,      // u (unit), r (m)
};

@group(0) @binding(0) var<storage, read> P: array<MParticle>;
@group(0) @binding(1) var<storage, read_write> OCC: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> F: FindU;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(F.n.w)) { return; }
  let p = P[i];
  if (p.x.w < 0.0) { return; }
  let w = F.o.xyz + p.x.xyz * F.o.w;
  let slot = u32(p.x.w + 0.5);
  for (var k = 0u; k < u32(F.k.x); k++) {
    let A = F.pa[k];
    let U = F.pu[k];
    let s = dot(w - A.xyz, U.xyz);
    if (s < 0.0 || s > A.w) { continue; }
    let d = length(w - A.xyz - U.xyz * s);
    if (d > U.w) { continue; }
    let bin = min(u32(s / max(A.w, 1e-9) * f32(NB)), NB - 1u);
    atomicOr(&OCC[k * NB + bin], 1u << min(slot, 15u));
  }
}
