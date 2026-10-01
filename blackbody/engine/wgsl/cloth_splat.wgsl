// Fabric -> gas: every vertex spreads onto a coupling grid (twice the simulation's cell size) how much
// cloth there is and how fast it moves, the momentum its drag took from the air since the last spread
// (cloth_air.wgsl gives it to the gas), and, where it burns, the fuel, heat and smoke it gives off
// (cloth_feed.wgsl). Fixed-point atomics.
//
// Per coupling cell, 13 values: area * n_a^2 (a = x, y, z; m^2 * AREA_K), area * n_a^2 * v_a (m^3/s *
// AREA_K), fuel (F m^3/s * FUEL_K), heat (field temperature * HEAT_K, the largest), smoke (* FUEL_K),
// momentum (kg m/s * MOM_K), steam boiling off wet cloth (kg/s * FUEL_K; cloth_steam.wgsl).
//!include cloth_common.wgsl

const AREA_K: f32 = 1.0e6;
const FUEL_K: f32 = 1.0e9;
const HEAT_K: f32 = 1.0e4;
const MOM_K: f32 = 1.0e9;
const CH: u32 = 13u;

struct Params {
  sim: vec4<f32>,    // vertex count, coupling cells in all, fuel per kg of cloth (F m^3), 1 / time the steam was gathered over (1/s)
  cg: vec4<f32>,     // coupling grid dims, its cell size (m)
  org: vec4<f32>,    // its corner (fire-local m); w = ambient K
  air: vec4<f32>,    // flame K, _, _, _
};

@group(0) @binding(0) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> N: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> S: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> M: array<Mat>;
@group(0) @binding(5) var<storage, read_write> G: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> IMP: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn clear(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = id.x + id.y * nwg.x * 64u;   // (gpu.groups_1d: a big coupling grid spills into y)
  if (i >= u32(U.sim.y) * CH) { return; }
  atomicStore(&G[i], 0);
}

@compute @workgroup_size(64, 1, 1)
fn splat(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.sim.x)) { return; }
  let st = S[i];
  let imp = IMP[i].xyz;
  let steam = IMP[i].w * U.sim.w;   // kg/s
  IMP[i] = vec4<f32>(0.0);
  if (gone(st)) { return; }
  let nn = N[i];
  let area = nn.w;
  if (area <= 0.0) { return; }
  let fi = u32(V[i].w + 0.5);
  let mat = M[fi];
  let v = V[i].xyz;
  let n2 = nn.xyz * nn.xyz;
  var fuel = 0.0;
  var heat = 0.0;
  var smoke = 0.0;
  if (st.z > 0.5) {
    // burning: the cloth gives off its fuel over its burn time, at its burning temperature
    let kg = mat.a.x * mat.c.y * area / max(mat.b.w, 1e-3);
    fuel = kg * U.sim.z * mat.e.z;
    smoke = fuel * mat.c.z;
    heat = (mat.c.x - U.org.w) / max(U.air.x - U.org.w, 1.0);
  }
  let dims = vec3<i32>(U.cg.xyz);
  let q = (X[i].xyz - U.org.xyz) / U.cg.w - vec3<f32>(0.5);
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
    let aw = area * w * AREA_K;
    atomicAdd(&G[base + 0u], i32(aw * n2.x));
    atomicAdd(&G[base + 1u], i32(aw * n2.y));
    atomicAdd(&G[base + 2u], i32(aw * n2.z));
    atomicAdd(&G[base + 3u], i32(aw * n2.x * v.x));
    atomicAdd(&G[base + 4u], i32(aw * n2.y * v.y));
    atomicAdd(&G[base + 5u], i32(aw * n2.z * v.z));
    atomicAdd(&G[base + 9u], i32(imp.x * w * MOM_K));
    atomicAdd(&G[base + 10u], i32(imp.y * w * MOM_K));
    atomicAdd(&G[base + 11u], i32(imp.z * w * MOM_K));
    if (steam > 0.0) { atomicAdd(&G[base + 12u], i32(steam * w * FUEL_K)); }
    if (fuel > 0.0) {
      atomicAdd(&G[base + 6u], i32(fuel * w * FUEL_K));
      atomicMax(&G[base + 7u], i32(heat * HEAT_K));
      atomicAdd(&G[base + 8u], i32(smoke * w * FUEL_K));
    }
  }
}
