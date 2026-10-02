// Matter in the liquid (matter.py): what the water does to the sand, worked out once a frame on the matter's grid from
// the liquid's last pressure solve, for mpm_grid.wgsl to apply to each node by its own density.
//  - Where grains are in the liquid, minus its pressure gradient: they are lighter by the water they put aside, and
//    pushed where the pressure falls away (a wave's front).
//  - Where the matter borders the liquid (its cells are solid to the liquid): matter the water gets into (sand, snow,
//    mud) feels the same at its skin as grains in the water beside it would; the water's pressure is between its
//    grains too, so it does not squeeze them together. Matter it does not get into (jelly, clay) is pressed on each face
//    between them, as a floating object is (liq_float.wgsl), spread through the cell so it sums to the pressure times
//    the area: it floats by the whole of the water it puts aside, and a wave shoves it.
//  - In both, the water running past drags it along: the stress of a flow over a bed, rho C |du| du, so fast water
//    carries the sand at the top off and lays it down where it slows.
//!include common.wgsl
//!include liq_common.wgsl

struct Params {
  g: Grid,           // the liquid's grid
  m: vec4<f32>,      // the matter's grid: node 0 (fire-local m), node spacing (m)
  mn: vec4<f32>,     // its nodes (x, y, z), _
  k: vec4<f32>,      // x to pressure (the liquid's density over its step: Pa per x), x's hydrostatic change over half a
                     // cell, the liquid's density times the bed's drag coefficient (kg/m^3), the matter porous (1/0)
};

@group(0) @binding(0) var X: texture_3d<f32>;   // the liquid's pressure (x = p dt / rho)
@group(0) @binding(1) var T: texture_3d<f32>;   // its cells: 0 air, 1 liquid, 2 solid
@group(0) @binding(2) var V: texture_3d<f32>;   // its velocity (on the cells' faces)
@group(0) @binding(3) var push: texture_storage_3d<rgba32float, write>;   // force on the matter (N/m^3); w: 1 where the liquid is
@group(0) @binding(4) var flow: texture_storage_3d<rgba32float, write>;   // the liquid's velocity there (m/s); w: drag (kg/m^4)
@group(1) @binding(0) var<uniform> U: Params;

fn is_liquid(c: vec3<i32>, n: vec3<i32>) -> bool {
  if (!in_grid(c, n)) { return false; }
  let t = textureLoad(T, c, 0).x;
  return t > 0.5 && t < 1.5;
}

// x at a neighbour of liquid cell c for the gradient: its own in the liquid, 0 in the air (the free surface), or none
// (w = 0) in a solid or past the grid
fn side_x(c: vec3<i32>, n: vec3<i32>) -> vec2<f32> {
  if (!in_grid(c, n)) { return vec2<f32>(0.0); }
  let t = textureLoad(T, c, 0).x;
  if (t > 1.5) { return vec2<f32>(0.0); }
  if (t < 0.5) { return vec2<f32>(0.0, 1.0); }
  return vec2<f32>(textureLoad(X, c, 0).x, 1.0);
}

// Minus the pressure gradient in liquid cell c (N/m^3), one-sided beside a solid.
fn grad_force(c: vec3<i32>, n: vec3<i32>, h: f32) -> vec3<f32> {
  let x0 = textureLoad(X, c, 0).x;
  var gx = vec3<f32>(0.0);
  for (var a = 0; a < 3; a++) {
    var e = vec3<i32>(0);
    e[a] = 1;
    let p = side_x(c + e, n);
    let m = side_x(c - e, n);
    if (p.y > 0.5 && m.y > 0.5) {
      gx[a] = 0.5 * (p.x - m.x);
    } else if (p.y > 0.5) {
      gx[a] = p.x - x0;
    } else if (m.y > 0.5) {
      gx[a] = x0 - m.x;
    }
  }
  return -gx * U.k.x / h;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let mn = vec3<i32>(U.mn.xyz);
  let c = vec3<i32>(id);
  if (any(c >= mn)) { return; }
  let n = gdim(U.g);
  let h = U.g.n.w;
  let q = (U.m.xyz + vec3<f32>(c) * U.m.w - U.g.org.xyz) / h;   // where the node is in the liquid's cells
  let lc = vec3<i32>(floor(q));
  var f = vec3<f32>(0.0);
  var u = vec3<f32>(0.0);
  var touch = 0.0;
  if (in_grid(lc, n)) {
    let t = textureLoad(T, lc, 0).x;
    if (t > 0.5 && t < 1.5) {
      // in the liquid
      f = grad_force(lc, n, h);
      u = mac_vel(V, q, n);
      touch = 1.0;
    } else if (t > 1.5) {
      // solid to the liquid: as in the liquid beside it (porous), or pressed on the faces between them (at the face,
      // half a cell up or down from the liquid cell's centre)
      let porous = U.k.w > 0.5;
      var nl = 0.0;
      for (var i = 0; i < 6; i++) {
        var d = vec3<i32>(0);
        d[i >> 1] = select(-1, 1, (i & 1) == 1);
        let b = lc + d;
        if (!is_liquid(b, n)) { continue; }
        if (porous) {
          f += grad_force(b, n, h);
        } else {
          f -= vec3<f32>(d) * (textureLoad(X, b, 0).x + U.k.y * f32(d.y)) * U.k.x / h;
        }
        u += mac_vel(V, vec3<f32>(b) + vec3<f32>(0.5), n);
        nl += 1.0;
      }
      if (nl > 0.0) {
        if (porous) { f /= nl; }
        u /= nl;
        touch = 1.0;
      }
    }
  }
  textureStore(push, c, vec4<f32>(f, touch));
  textureStore(flow, c, vec4<f32>(u, touch * U.k.z / h));
}
