// Burning grass -> gas: each burning blade gives off its fuel over its burn time, with its heat and its smoke, along
// what is left of it (most at its top, where it burns), onto the coupling grid in cloth_splat.wgsl's layout, which
// cloth_feed.wgsl puts into the gas. Fixed-point atomics.
//!include strand_common.wgsl

const FUEL_K: f32 = 1.0e9;
const HEAT_K: f32 = 1.0e4;
const CH: u32 = 13u;

struct Params {
  sim: vec4<f32>,    // blades, coupling cells in all, fuel per kg of grass (F m^3), a blade's density (kg/m^3)
  cg: vec4<f32>,     // coupling grid dims, its cell size (m)
  org: vec4<f32>,    // its corner (fire-local m); w = ambient K
  air: vec4<f32>,    // flame K, the grass's flame K, _, _
  patches: array<Patch, MAX_PATCHES>,
};

@group(0) @binding(0) var<storage, read> BL: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> RT: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> ST: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> G: array<atomic<i32>>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn clear(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = id.x + id.y * nwg.x * 64u;   // (gpu.groups_1d: a big coupling grid spills into y)
  if (i >= u32(U.sim.y) * CH) { return; }
  atomicStore(&G[i], 0);
}

fn put(p: vec3<f32>, fuel: f32, heat: f32, smoke: f32) {
  let dims = vec3<i32>(U.cg.xyz);
  let q = (p - U.org.xyz) / U.cg.w - vec3<f32>(0.5);
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
    atomicAdd(&G[base + 6u], i32(fuel * w * FUEL_K));
    atomicMax(&G[base + 7u], i32(heat * HEAT_K));
    atomicAdd(&G[base + 8u], i32(smoke * w * FUEL_K));
  }
}

@compute @workgroup_size(64, 1, 1)
fn splat(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.sim.x)) { return; }
  let st = ST[i];
  if (RT[i].w < 0.5 || st.z < 0.5) { return; }
  let b0 = BL[2u * i];
  let b1 = BL[2u * i + 1u];
  let pa = U.patches[u32(b0.z + 0.5)];
  let kg = U.sim.w * b1.y * pa.form.x * b1.x;               // the blade (density x width x thickness x height)
  let fuel = kg / max(pa.burn.w, 1e-3) * U.sim.z;           // F m^3 a second
  let heat = (U.air.y - U.org.w) / max(U.air.x - U.org.w, 1.0);
  let smoke = fuel * pa.base.z;
  put(X[i * POINTS + POINTS - 1u].xyz, 0.6 * fuel, heat, 0.6 * smoke);
  put(X[i * POINTS + POINTS / 2u].xyz, 0.4 * fuel, heat, 0.4 * smoke);
}
