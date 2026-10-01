// The open-water layer's foam, one step: carried by the current and the wind's pull on the surface;
// fading; inside the box whatever foam the simulation has there; surf where the waves break over
// the shallows; and foam churned up where the current runs past islands, piers and posts.

struct Params {
  L: Layer,
  k: vec4<f32>,   // foam life (s), the wind's drift of floating foam x, z (m/s), the sea's significant height (m)
  m: vec4<f32>,   // the sea's peak wavenumber (1/m), surf foam rate (1/s), foam from the current round obstacles (1/m), _
};

@group(0) @binding(0) var foam: texture_2d<f32>;
@group(0) @binding(1) var vel: texture_2d<f32>;
@group(0) @binding(2) var cols: texture_2d<f32>;
@group(0) @binding(3) var dep: texture_2d<f32>;
@group(0) @binding(4) var dst: texture_storage_2d<r32float, write>;
@group(1) @binding(0) var<uniform> U: Params;
//!include ocl_common.wgsl

fn cr(f: f32) -> vec4<f32> {
  let f2 = f * f;
  let f3 = f2 * f;
  return vec4<f32>(-0.5 * f3 + f2 - 0.5 * f, 1.5 * f3 - 2.5 * f2 + 1.0, -1.5 * f3 + 2.0 * f2 + 0.5 * f, 0.5 * f3 - 0.5 * f2);
}

// Bicubic (Catmull-Rom) foam at layer coordinates q, held within its four nearest texels (so foam
// keeps its edges as it drifts, without ringing).
fn foam_at(q: vec2<f32>) -> f32 {
  let n = lay_n();
  let p = q - vec2<f32>(0.5);
  let i0 = vec2<i32>(floor(p));
  let f = p - floor(p);
  let wx = cr(f.x);
  let wz = cr(f.y);
  var s = 0.0;
  var lo = 1.0e9;
  var hi = -1.0e9;
  for (var j = 0; j < 4; j++) {
    var row = 0.0;
    for (var i = 0; i < 4; i++) {
      let v = textureLoad(foam, clamp(i0 + vec2<i32>(i - 1, j - 1), vec2<i32>(0), vec2<i32>(n - 1)), 0).x;
      row += wx[i] * v;
      if (i >= 1 && i <= 2 && j >= 1 && j <= 2) { lo = min(lo, v); hi = max(hi, v); }
    }
    s += wz[j] * row;
  }
  return clamp(s, lo, hi);
}

fn box_foam(x: vec2<f32>) -> f32 {
  let n = vec2<i32>(i32(U.L.bdim.x), i32(U.L.bdim.y));
  let q = box_coord(x) - vec2<f32>(0.5);
  let i0 = vec2<i32>(floor(q));
  let f = q - floor(q);
  var s = 0.0;
  for (var j = 0; j < 4; j++) {
    let o = vec2<i32>(j & 1, j >> 1);
    let v = textureLoad(cols, clamp(i0 + o, vec2<i32>(0), n - vec2<i32>(1)), 0).w;
    s += mix(1.0 - f.x, f.x, f32(o.x)) * mix(1.0 - f.y, f.y, f32(o.y)) * v;
  }
  return s;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec2<i32>(id.xy);
  if (c.x >= lay_n() || c.y >= lay_n()) { return; }
  let d = textureLoad(dep, c, 0);
  if (lay_blocked(d)) {
    textureStore(dst, c, vec4<f32>(0.0));
    return;
  }
  let dt = U.L.bdim.w;
  let x = lay_pos(c);
  let v = textureLoad(vel, c, 0).xy + U.k.yz;
  var f = foam_at(vec2<f32>(c) + vec2<f32>(0.5) - v * dt / U.L.map.z) * exp(-dt / max(U.k.x, 0.1));
  // the simulation's foam, inside the box
  let bs = box_share(x);
  if (bs > 0.0) { f = mix(f, box_foam(x), bs); }
  // surf: waves taller than about three quarters of the depth break
  if (U.k.w > 0.0 && U.m.y > 0.0) {
    let hs = U.k.w * shoal(U.m.x, d.x);
    let surf = smoothstep(0.35, 0.8, hs / max(d.x, 0.02));
    f += U.m.y * surf * dt * max(0.6 - f, 0.0);   // the surf zone's foam, in patches, not a sheet
  }
  // churned where the current runs past a solid
  if (U.m.z > 0.0) {
    let near = 1.0 - smoothstep(0.0, 1.5 * U.L.map.z + 0.3, d.y);
    f += U.m.z * length(textureLoad(vel, c, 0).xy) * near * dt;
  }
  textureStore(dst, c, vec4<f32>(clamp(f, 0.0, 1.5), 0.0, 0.0, 0.0));
}
