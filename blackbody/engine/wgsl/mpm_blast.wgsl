// Matter: a blast (solids.py blast_impulse) throws it away from where it goes off. The impulse per area on its surface
// goes into the matter within a depth L of it: particles near the surface are thrown at i / (rho L), deeper ones less,
// as exp(-depth / L) (their depth from the matter's surface distance, mpm_surf_pack.wgsl).
//!include mpm_common.wgsl

struct Params {
  n: vec4<f32>,      // grid nodes (x, y, z), particles (count)
  o: vec4<f32>,      // the grid's corner (fire-local m), node spacing (m)
  b: vec4<f32>,      // where it goes off (fire-local m), kg of TNT
  l: vec4<f32>,      // L (m), the closest it counts as (m), _, _
  mats: array<MMat, MAX_MATS>,
};

@group(0) @binding(0) var<storage, read_write> P: array<MParticle>;
@group(0) @binding(1) var surf: texture_3d<f32>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.n.w)) { return; }
  var p = P[i];
  if (p.x.w < 0.0 || p.c1.w > 1.0e8) { return; }
  let m = U.mats[u32(p.x.w)];
  let w = U.o.xyz + p.x.xyz * U.o.w;
  let r = w - U.b.xyz;
  let dist = max(length(r), U.l.y);
  let impulse = 2.0 * 200.0 * pow(U.b.w, 2.0 / 3.0) / dist;
  let node = clamp(vec3<i32>(round(p.x.xyz)), vec3<i32>(0), vec3<i32>(U.n.xyz) - vec3<i32>(1));
  let depth = max(-textureLoad(surf, node, 0).x, 0.0);
  let dv = impulse / (m.a.w * U.l.x) * exp(-depth / U.l.x);
  p.v = vec4<f32>(p.v.xyz + normalize(r + vec3<f32>(0.0, 1.0e-6, 0.0)) * dv, p.v.w);
  P[i] = p;
}
