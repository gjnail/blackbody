// Fabric in the light volume (shadows): every vertex of cloth spreads onto the light grid how much light
// it stops, its area times -ln(1 - opacity), opacity from its fabric (thin cloth lets some through).
// cloth_occlude.wgsl turns that into an extinction (1/m). Fixed-point atomics.
//!include cloth_common.wgsl

const OCC_K: f32 = 1.0e6;

struct Params {
  g: vec4<f32>,                     // light-grid dims, vertex count
  o: vec4<f32>,                     // light-grid corner (fire-local m), cells in all
  c: vec4<f32>,                     // light cell size (m, per axis), _
  k: array<vec4<f32>, 4>,           // per fabric: -ln(1 - opacity)
};

@group(0) @binding(0) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> N: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> S: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> UV: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> OCC: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> U: Params;

fn kappa(f: u32) -> f32 {
  let q = U.k[f / 4u];
  let k = f % 4u;
  return select(select(q.w, q.z, k == 2u), select(q.y, q.x, k == 0u), k < 2u);
}

@compute @workgroup_size(64, 1, 1)
fn clear(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = id.x + id.y * nwg.x * 64u;   // (gpu.groups_1d: past 65535 groups it spills into y)
  if (i >= u32(U.o.w)) { return; }
  atomicStore(&OCC[i], 0u);
}

@compute @workgroup_size(64, 1, 1)
fn splat(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = id.x + id.y * nwg.x * 64u;
  if (i >= u32(U.g.w)) { return; }
  if (gone(S[i])) { return; }
  let area = N[i].w;
  if (area <= 0.0) { return; }
  let a = area * kappa(u32(UV[i].z + 0.5));
  let dims = vec3<i32>(U.g.xyz);
  let q = (X[i].xyz - U.o.xyz) / U.c.xyz - vec3<f32>(0.5);
  let b = vec3<i32>(floor(q));
  let f = q - floor(q);
  for (var k = 0; k < 8; k++) {
    let o = vec3<i32>(k & 1, (k >> 1) & 1, k >> 2);
    let c = b + o;
    if (any(c < vec3<i32>(0)) || any(c >= dims)) { continue; }
    let w3 = mix(vec3<f32>(1.0) - f, f, vec3<f32>(o));
    let w = w3.x * w3.y * w3.z;
    atomicAdd(&OCC[u32((c.z * dims.y + c.y) * dims.x + c.x)], u32(a * w * OCC_K));
  }
}
