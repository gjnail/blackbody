// Gas <- matter's dust (matter.py dust_hook): the fine dust the wind lifts off sand, snow, ash and sawdust as it moves
// them, and the grains flying through the air shed (mpm_g2p.wgsl adds it up over the matter's frame, in kg, on a grid
// of the gas's cells two to a side), into the gas as smoke over the frame's substeps: a haze of blowing sand off a
// dune's crest, spindrift off a snow ridge, a cloud of dust round a blast's spray of sand. Fine dust hangs in the air,
// so the gas carries it as it carries smoke.
//!include common.wgsl

const DUST_K: f32 = 1.0e11;     // kg fixed point (mpm_g2p.wgsl)

struct Params {
  t: vec4<f32>,      // the gas's grid dims, its cell size (m)
  dg: vec4<f32>,     // the dust grid dims, its cell size (m)
  k: vec4<f32>,      // the substep's share of the frame, smoke a kg of dust a cubic metre makes, _, _
};

@group(0) @binding(0) var src: texture_3d<f32>;
@group(0) @binding(1) var<storage, read> DU: array<i32>;
@group(0) @binding(2) var dst: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(U.t.xyz);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  var s = textureLoad(src, c, 0);
  let d = vec3<i32>(U.dg.xyz);
  let q = min(vec3<i32>((vec3<f32>(c) + vec3<f32>(0.5)) * (U.t.w / U.dg.w)), d - vec3<i32>(1));
  let kg = f32(DU[u32((q.z * d.y + q.y) * d.x + q.x)]) / DUST_K;
  if (kg > 0.0) {
    s.z += kg / (U.dg.w * U.dg.w * U.dg.w) * U.k.y * U.k.x;
  }
  textureStore(dst, c, s);
}
