// Matter, the air round it (matter.py air()): once a frame, the air's velocity at each node of the matter's grid, for
// the wind's push on the grains (mpm_g2p.wgsl). In a fire box the gas's own velocity (it flows round the matter, which
// is solid to it, so the wind over a heap speeds up across its top and stalls in its lee); outside the box, or with no
// gas, the scene's wind with its gusts. w: 1 where the liquid is (no wind under water).
//!include common.wgsl

struct Params {
  m: vec4<f32>,      // node 0 (fire-local m), node spacing (m)
  mn: vec4<f32>,     // nodes (x, y, z), _
  g: vec4<f32>,      // the gas's grid corner, its cell size (m)
  gn: vec4<f32>,     // its cells, gas on (1/0)
  l: vec4<f32>,      // the liquid's grid corner, its cell size (m)
  ln: vec4<f32>,     // its cells, liquid on (1/0)
  w: vec4<f32>,      // the wind (m/s), gusts (share of it)
  t: vec4<f32>,      // time (s), _
};

@group(0) @binding(0) var gvel: texture_3d<f32>;
@group(0) @binding(1) var lin: sampler;
@group(0) @binding(2) var LTYPE: texture_3d<f32>;
@group(0) @binding(3) var air: texture_storage_3d<rgba32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn gust(p: vec3<f32>, t: f32) -> f32 {
  // rolling gusts drifting downwind: a few slow waves (as the grass's wind, strand_step.wgsl)
  let a = sin(0.9 * p.x - 1.7 * t) * sin(0.7 * p.z + 1.1 * t);
  let b = sin(2.3 * p.x + 0.4 * p.z - 3.1 * t);
  return 0.6 * a + 0.4 * b;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(U.mn.xyz);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let p = U.m.xyz + vec3<f32>(c) * U.m.w;
  var v = U.w.xyz * (1.0 + U.w.w * gust(p, U.t.x));
  if (U.gn.w > 0.5) {
    let q = (p - U.g.xyz) / U.g.w;
    if (all(q >= vec3<f32>(0.0)) && all(q <= U.gn.xyz)) {
      v = vel_at(gvel, lin, q, U.gn.xyz);
    }
  }
  var wet = 0.0;
  if (U.ln.w > 0.5) {
    let lc = vec3<i32>(floor((p - U.l.xyz) / U.l.w));
    if (all(lc >= vec3<i32>(0)) && all(lc < vec3<i32>(U.ln.xyz))) {
      let t = textureLoad(LTYPE, lc, 0).x;
      if (t > 0.5 && t < 1.5) { wet = 1.0; }
    }
  }
  textureStore(air, c, vec4<f32>(v, wet));
}
