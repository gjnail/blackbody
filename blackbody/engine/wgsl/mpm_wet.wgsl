// Sand and water (matter.py). A grain of dry sand next to liquid (a liquid cell where it is or beside it) soaks over
// half a second and is then damp sand: darker, its grains held together by the water between them, so it stands
// steeper and holds a cut edge. The water seeps on into damp sand, the further from it the slower (mpm_wetdist.wgsl:
// how far through the sand each cell is from the water), and sand it has soaked through lets go: its grains part (a
// sand castle the sea reaches is undermined and slumps, and the water carries it off). Away from the water, soaked
// sand drains back to damp in a few seconds.

struct MParticle {
  x: vec4<f32>,   // position (the matter's grid units); w = material slot, negative: an empty slot
  v: vec4<f32>,
  c0: vec4<f32>,
  c1: vec4<f32>,
  c2: vec4<f32>,  // w = how far it has soaked (0..1)
  f0: vec4<f32>,
  f1: vec4<f32>,
  f2: vec4<f32>,
};

struct Params {
  m: vec4<f32>,      // the matter's grid: node 0 (fire-local m), node spacing (m)
  lq: vec4<f32>,     // the liquid's grid corner (fire-local m), its cell size (m)
  ln: vec4<f32>,     // its cells (x, y, z), _
  k: vec4<f32>,      // particles (count), dt (s), time to get damp, time to soak through a cell of damp sand (s)
  k2: vec4<f32>,     // time to drain (s), the furthest from the water it soaks (cells), how far it drains from (cells), _
  to: array<vec4<f32>, 16>,   // per material slot: (its damp slot, its soaked slot, what it is: 0 dry, 1 damp, 2 soaked;
                              // -1 none of these), -1: it does not change
};

@group(0) @binding(0) var<storage, read_write> P: array<MParticle>;
@group(0) @binding(1) var LTYPE: texture_3d<f32>;   // the liquid's cells: 0 air, 1 liquid, 2 solid
@group(0) @binding(2) var WD: texture_3d<f32>;      // how far each is from the water through the sand (cells)
@group(1) @binding(0) var<uniform> U: Params;

fn liquid_at(q: vec3<i32>, n: vec3<i32>) -> bool {
  if (any(q < vec3<i32>(0)) || any(q >= n)) { return false; }
  let t = textureLoad(LTYPE, q, 0).x;
  return t > 0.5 && t < 1.5;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = id.x + id.y * nwg.x * 64u;
  if (i >= u32(U.k.x)) { return; }
  let p = P[i];
  if (p.x.w < 0.0) { return; }
  let slot = u32(p.x.w + 0.5);
  if (slot >= 15u) { return; }
  let to = U.to[slot];
  if (to.z < 0.0) { return; }
  let pos = U.m.xyz + p.x.xyz * U.m.w;
  let n = vec3<i32>(U.ln.xyz);
  let c = vec3<i32>(floor((pos - U.lq.xyz) / U.lq.w));
  let dt = U.k.y;
  var c2 = p.c2;
  var next = -1.0;
  if (to.z < 0.5) {
    // dry: damp once it has been next to the liquid long enough
    var near = liquid_at(c, n);
    for (var a = 0; a < 6; a++) {
      var e = vec3<i32>(0);
      e[a >> 1] = select(-1, 1, (a & 1) == 1);
      near = near || liquid_at(c + e, n);
    }
    if (!near) { return; }
    c2.w += dt / max(U.k.z, 1e-3);
    if (c2.w >= 1.0) { next = to.x; }
  } else {
    var d = 1.0e4;
    if (all(c >= vec3<i32>(0)) && all(c < n)) { d = textureLoad(WD, c, 0).x; }
    if (to.z < 1.5) {
      // damp: the water seeping in soaks it, the further in the slower
      if (d > U.k2.y) { return; }
      c2.w += dt / (max(U.k.w, 1e-3) * (1.0 + d));
      if (c2.w >= 1.0) { next = to.y; }
    } else {
      // soaked: stays so near the water; away from it, drains back to damp
      if (d <= U.k2.z) {
        c2.w = 0.0;
      } else {
        c2.w += dt / max(U.k2.x, 1e-3);
        if (c2.w >= 1.0) { next = to.x; }
      }
    }
  }
  if (next >= 0.0) {
    c2.w = 0.0;
    P[i].x = vec4<f32>(p.x.xyz, next);
  }
  P[i].c2 = c2;
}
