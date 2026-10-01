// The open-water layer's current, one step: carried along by itself; inside the box the simulation's
// own surface flow; still against islands, piers and the shore; the open water's current round the
// layer's edge. (ocl_div, ocl_jac and ocl_proj then keep it from piling up or draining.)

struct Params {
  L: Layer,
  k: vec4<f32>,   // the open water's current x, z (m/s), _, _
};

@group(0) @binding(0) var vel: texture_2d<f32>;
@group(0) @binding(1) var cols: texture_2d<f32>;
@group(0) @binding(2) var dep: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rg32float, write>;
@group(1) @binding(0) var<uniform> U: Params;
//!include ocl_common.wgsl

// The box's surface flow at x (bilinear over its columns) and how much of it is known.
fn box_flow(x: vec2<f32>) -> vec3<f32> {
  let n = vec2<i32>(i32(U.L.bdim.x), i32(U.L.bdim.y));
  let q = box_coord(x) - vec2<f32>(0.5);
  let i0 = vec2<i32>(floor(q));
  let f = q - floor(q);
  var s = vec2<f32>(0.0);
  var w = 0.0;
  for (var j = 0; j < 4; j++) {
    let o = vec2<i32>(j & 1, j >> 1);
    let v = textureLoad(cols, clamp(i0 + o, vec2<i32>(0), n - vec2<i32>(1)), 0);
    let ww = mix(1.0 - f.x, f.x, f32(o.x)) * mix(1.0 - f.y, f.y, f32(o.y));
    if (v.x > -1.0e3) { s += ww * v.yz; w += ww; }
  }
  return vec3<f32>(s / max(w, 1e-6), w);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec2<i32>(id.xy);
  if (c.x >= lay_n() || c.y >= lay_n()) { return; }
  if (lay_blocked(textureLoad(dep, c, 0))) {
    textureStore(dst, c, vec4<f32>(0.0));
    return;
  }
  let x = lay_pos(c);
  let v0 = textureLoad(vel, c, 0).xy;
  var v = lay_samp(vel, vec2<f32>(c) + vec2<f32>(0.5) - v0 * U.L.bdim.w / U.L.map.z).xy;
  let bs = box_share(x);
  if (bs > 0.0) {
    let b = box_flow(x);
    v = mix(v, b.xy, bs * b.z);
  }
  v = mix(v, U.k.xy, edge_ramp(c));
  textureStore(dst, c, vec4<f32>(v, 0.0, 0.0));
}
