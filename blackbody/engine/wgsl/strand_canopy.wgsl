// Grass -> gas, the canopy: the area each blade shows the wind (its width times each segment's length, half of it
// across the flow on average, as it stands and bends now, what is left of it once burnt), onto a coupling grid (the
// gas's cells, two to a side), in fixed point. strand_drag.wgsl takes the wind's drag on it out of the gas.
//!include strand_common.wgsl

const AREA_K: f32 = 65536.0;   // m^2 fixed point

struct Params {
  sim: vec4<f32>,    // blades, coupling cells, _, _
  cg: vec4<f32>,     // coupling grid dims, its cell size (m)
  org: vec4<f32>,    // its corner (fire-local m), _
};

@group(0) @binding(0) var<storage, read> BL: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> RT: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> CA: array<atomic<i32>>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn clear(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = id.x + id.y * nwg.x * 64u;
  if (i >= u32(U.sim.y)) { return; }
  atomicStore(&CA[i], 0);
}

fn put(p: vec3<f32>, area: f32) {
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
    atomicAdd(&CA[u32((c.z * dims.y + c.y) * dims.x + c.x)], i32(round(area * w * AREA_K)));
  }
}

@compute @workgroup_size(64, 1, 1)
fn splat(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = id.x + id.y * nwg.x * 64u;
  if (i >= u32(U.sim.x)) { return; }
  if (RT[i].w < 0.5) { return; }
  let width = BL[2u * i + 1u].y;
  for (var k = 0u; k + 1u < POINTS; k++) {
    let a = X[i * POINTS + k].xyz;
    let b = X[i * POINTS + k + 1u].xyz;
    put(0.5 * (a + b), 0.5 * width * length(b - a));
  }
}
