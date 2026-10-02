// Matter that burns (matter.py, mpm_heat.wgsl): every particle burning where the air reaches it gives the gas the fuel
// of what of it burns away a second (as burning grass does: strand_splat.wgsl), at its burning temperature, with its
// smoke, onto the coupling grid in cloth_splat.wgsl's layout, which cloth_feed.wgsl puts into the gas every substep
// (matter_fire.py). Fixed-point atomics, as points_splat.wgsl's.
//!include mpm_common.wgsl

const FUEL_K: f32 = 1.0e9;
const HEAT_K: f32 = 1.0e4;
const CH: u32 = 13u;

struct Params {
  m: vec4<f32>,      // the matter's grid: node 0 (fire-local m), node spacing (m)
  n: vec4<f32>,      // particles (count), coupling cells in all, _, _
  cg: vec4<f32>,     // coupling grid dims, its cell size (m)
  org: vec4<f32>,    // its corner (fire-local m)
  slots: array<vec4<f32>, 16>,  // per material slot: a burning particle's fuel (F m^3/s; 0: none), its heat (field
                                // temperature), its smoke (/s), _
};

@group(0) @binding(0) var<storage, read> P: array<MParticle>;
@group(0) @binding(1) var<storage, read_write> G: array<atomic<i32>>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn clear(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.n.y) * CH) { return; }
  atomicStore(&G[i], 0);
}

@compute @workgroup_size(64, 1, 1)
fn splat(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.n.x)) { return; }
  let p = P[i];
  if (p.x.w < 0.0 || p.f1.w <= 0.0) { return; }
  let slot = u32(p.x.w + 0.5);
  if (slot >= 15u) { return; }
  let t = U.slots[slot];
  let fuel = t.x * p.f1.w;
  if (fuel <= 0.0) { return; }
  let smoke = t.z * p.f1.w;
  let dims = vec3<i32>(U.cg.xyz);
  let q = (U.m.xyz + p.x.xyz * U.m.w - U.org.xyz) / U.cg.w - vec3<f32>(0.5);
  let b = vec3<i32>(floor(q));
  let f = q - floor(q);
  for (var o = 0; o < 8; o++) {
    let oo = vec3<i32>(o & 1, (o >> 1) & 1, o >> 2);
    let c = b + oo;
    if (any(c < vec3<i32>(0)) || any(c >= dims)) { continue; }
    let w3 = mix(vec3<f32>(1.0) - f, f, vec3<f32>(oo));
    let w = w3.x * w3.y * w3.z;
    if (w <= 0.0) { continue; }
    let base = u32((c.z * dims.y + c.y) * dims.x + c.x) * CH;
    atomicAdd(&G[base + 6u], i32(fuel * w * FUEL_K));
    atomicMax(&G[base + 7u], i32(t.y * HEAT_K));
    if (smoke > 0.0) { atomicAdd(&G[base + 8u], i32(smoke * w * FUEL_K)); }
  }
}
