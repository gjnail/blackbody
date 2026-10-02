// Sand, snow, mud, jelly and clay (matter.py) as solid to a grid solver (the gas, the liquid). Once a frame its
// particles are counted into the solver's cells (how much of each cell they fill, and their momentum: fixed-point
// atomics); then each substep the cells more than half full, looked at with their neighbours, are folded into the
// solver's distance to solids (sdf, cells) and given the matter's velocity there (svel; w = 1), as broken pieces are
// (bodies_sdf.wgsl), so smoke goes round a sand pile and water round a heap of snow, and a pour pushes the air aside.
//!include common.wgsl

const VOL_K: f32 = 65536.0;   // fixed point: shares of a cell
const MOM_K: f32 = 4096.0;    // fixed point: share of a cell x velocity (m/s)

// A matter particle (mpm_common.wgsl MParticle): only its position and velocity are read here.
struct MParticle {
  x: vec4<f32>,   // position (the matter's grid units); w = material, negative or 15+: none
  v: vec4<f32>,   // velocity (m/s)
  c0: vec4<f32>,
  c1: vec4<f32>,
  c2: vec4<f32>,
  f0: vec4<f32>,
  f1: vec4<f32>,
  f2: vec4<f32>,
};

struct Params {
  g: Grid,            // the solver's grid
  m: vec4<f32>,       // the matter's grid: node 0 (fire-local m), node spacing (m)
  k: vec4<f32>,       // particles (count), a particle's share of a solver cell, pieces baked first this substep (1/0), _
};

@group(0) @binding(0) var<storage, read> P: array<MParticle>;
@group(0) @binding(1) var<storage, read_write> A: array<atomic<i32>>;   // per cell: share filled, momentum x, y, z
@group(0) @binding(2) var sdf: texture_storage_3d<r32float, read_write>;
@group(0) @binding(3) var svel: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn clear(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = id.x + id.y * nwg.x * 64u;
  let d = gdim(U.g);
  if (i >= 4u * u32(d.x * d.y * d.z)) { return; }
  atomicStore(&A[i], 0);
}

@compute @workgroup_size(64, 1, 1)
fn splat(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = id.x + id.y * nwg.x * 64u;
  if (i >= u32(U.k.x)) { return; }
  let p = P[i];
  if (p.x.w < 0.0 || u32(p.x.w + 0.5) >= 15u) { return; }
  let w = U.m.xyz + p.x.xyz * U.m.w;
  let d = gdim(U.g);
  let c = vec3<i32>(floor((w - U.g.org.xyz) / U.g.n.w));
  if (any(c < vec3<i32>(0)) || any(c >= d)) { return; }
  let k = 4u * u32((c.z * d.y + c.y) * d.x + c.x);
  let s = U.k.y;
  atomicAdd(&A[k], i32(s * VOL_K));
  atomicAdd(&A[k + 1u], i32(s * p.v.x * MOM_K));
  atomicAdd(&A[k + 2u], i32(s * p.v.y * MOM_K));
  atomicAdd(&A[k + 3u], i32(s * p.v.z * MOM_K));
}

@compute @workgroup_size(8, 8, 4)
fn bake(@builtin(global_invocation_id) id: vec3<u32>) {
  let d = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= d)) { return; }
  // how full it is, with its neighbours (a cell's own count is a few particles: noisy)
  var fill = 0.0;
  var wsum = 0.0;
  var mom = vec3<f32>(0.0);
  var vol = 0.0;
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let q = c + vec3<i32>(x, y, z);
        if (any(q < vec3<i32>(0)) || any(q >= d)) { continue; }
        let w = 1.0 / f32(1 + abs(x) + abs(y) + abs(z));
        let k = 4u * u32((q.z * d.y + q.y) * d.x + q.x);
        let f = f32(atomicLoad(&A[k])) / VOL_K;
        fill += w * f;
        wsum += w;
        vol += f;
        mom += vec3<f32>(f32(atomicLoad(&A[k + 1u])), f32(atomicLoad(&A[k + 2u])), f32(atomicLoad(&A[k + 3u]))) / MOM_K;
      }
    }
  }
  fill /= wsum;
  let solid = (0.5 - fill) * 2.0;          // (cells: -1 full, 0 half full, 1 empty)
  if (fill < 0.02) {
    if (U.k.z < 0.5) { textureStore(svel, c, vec4<f32>(0.0)); }   // (nothing here: no pieces wrote it either)
    return;
  }
  let base = textureLoad(sdf, c).x;
  textureStore(sdf, c, vec4<f32>(min(base, solid), 0.0, 0.0, 0.0));
  if (solid < 0.5) {
    textureStore(svel, c, vec4<f32>(mom / max(vol, 1e-6), 1.0));
  } else if (U.k.z < 0.5) {
    textureStore(svel, c, vec4<f32>(0.0));
  }
}
