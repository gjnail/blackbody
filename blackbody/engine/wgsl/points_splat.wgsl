// Fuel from points -> gas: the burning pieces of things that break and burn (solids.py fuel_points) give off fuel, heat and
// smoke where they are, onto the coupling grid in cloth_splat.wgsl's layout, which cloth_feed.wgsl puts into the gas.
// Fixed-point atomics.

const FUEL_K: f32 = 1.0e9;
const HEAT_K: f32 = 1.0e4;
const CH: u32 = 13u;

struct Params {
  sim: vec4<f32>,    // points, coupling cells in all, _, _
  cg: vec4<f32>,     // coupling grid dims, its cell size (m)
  org: vec4<f32>,    // its corner (fire-local m)
};

@group(0) @binding(0) var<storage, read> P: array<vec4<f32>>;   // per point: where (fire-local m), fuel (F m^3/s); heat, smoke (/s), _, _
@group(0) @binding(1) var<storage, read_write> G: array<atomic<i32>>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn clear(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = id.x + id.y * nwg.x * 64u;
  if (i >= u32(U.sim.y) * CH) { return; }
  atomicStore(&G[i], 0);
}

@compute @workgroup_size(64, 1, 1)
fn splat(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.sim.x)) { return; }
  let a = P[2u * i];
  let e = P[2u * i + 1u];
  let dims = vec3<i32>(U.cg.xyz);
  let q = (a.xyz - U.org.xyz) / U.cg.w - vec3<f32>(0.5);
  let b = vec3<i32>(floor(q));
  let f = q - floor(q);
  for (var k = 0; k < 8; k++) {
    let o = vec3<i32>(k & 1, (k >> 1) & 1, k >> 2);
    let c = b + o;
    if (any(c < vec3<i32>(0)) || any(c >= dims)) { continue; }
    let w3 = mix(vec3<f32>(1.0) - f, f, vec3<f32>(o));
    let w = w3.x * w3.y * w3.z;
    if (w <= 0.0) { continue; }
    let base = u32((c.z * dims.y + c.y) * dims.x + c.x) * CH;
    if (a.w > 0.0) { atomicAdd(&G[base + 6u], i32(a.w * w * FUEL_K)); }
    atomicMax(&G[base + 7u], i32(e.x * HEAT_K));
    if (e.y > 0.0) { atomicAdd(&G[base + 8u], i32(e.y * w * FUEL_K)); }
  }
}
