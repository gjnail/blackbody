// The sea's height over the simulation box's footprint, once per frame: inside the box the ray
// tracer blends the simulated surface into the open water, and there it reads the sea's height
// from this map (one texture read) instead of evaluating the waves at every step.

struct Params {
  ocn: vec4<f32>,
  ocn2: vec4<f32>,
  ocx: array<vec4<f32>, 14>,
  box: vec4<f32>,    // map texels x, z; the box's corner x, z (fire-local m)
  box2: vec4<f32>,   // the box's size x, z (m)
};

@group(0) @binding(0) var oc_w: texture_3d<f32>;
@group(0) @binding(1) var oc_f: texture_3d<f32>;
@group(0) @binding(2) var rep: sampler;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var oc_l1: texture_2d<f32>;   // the open-water layer
@group(1) @binding(0) var<uniform> U: Params;
//!include ocn_sample.wgsl

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (f32(id.x) >= U.box.x || f32(id.y) >= U.box.y) { return; }
  let x = U.box.zw + (vec2<f32>(id.xy) + vec2<f32>(0.5)) / U.box.xy * U.box2.xy;
  var eta = 0.0;
  var dry = 0.0;
  if (ocean_on()) {
    // dry land, and the swash zone the waves wash up and drain back from: the simulation's own surface
    // there, not the sea's level (1 dry .. 0 sea, over depths of 0.1 to 0.4 of the wave height)
    let hs = max(U.ocx[9].z, 0.05);
    dry = 1.0 - smoothstep(0.1 * hs, 0.4 * hs, lay_at(x).z);
    let w = ocean_w(x, 0.0);
    var x0 = x;
    for (var i = 0; i < 4; i++) { x0 = x - ocean_disp_w(x0, w).yz; }
    eta = ocean_disp_w(x0, w).x + ocean_surge(x).x + lay_at(x).x;
  }
  textureStore(dst, vec2<i32>(id.xy), vec4<f32>(eta, dry, 0.0, 0.0));
}
