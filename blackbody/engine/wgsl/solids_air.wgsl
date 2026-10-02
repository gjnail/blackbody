// The gas velocity and temperature at points: round the rigid bodies, for the air's drag on them, and round the pieces
// of things that burn, for their fire (solids.py).
//!include common.wgsl

struct Params {
  g: vec4<f32>,   // grid corner (fire-local m), cell size (m)
  n: vec4<f32>,   // grid size (cells), point count
};

@group(0) @binding(0) var vel: texture_3d<f32>;
@group(0) @binding(1) var scal: texture_3d<f32>;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var<storage, read> P: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> V: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let i = gid.x + gid.y * nw.x * 64u;
  if (i >= u32(U.n.w)) { return; }
  let n = U.n.xyz;
  let pg = (P[i].xyz - U.g.xyz) / U.g.w;
  var v = vec3<f32>(0.0);
  var T = 0.0;
  if (all(pg >= vec3<f32>(0.0)) && all(pg <= n)) {
    v = vel_at(vel, lin, pg, n);
    T = max(samp_c(scal, lin, pg, n).x, 0.0);
  }
  V[i] = vec4<f32>(v, T);
}
