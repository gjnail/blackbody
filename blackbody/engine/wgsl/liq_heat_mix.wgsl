// Molten liquids: the particles' heat evened out a little with their cell's each step: heat moving between
// neighbouring particles, which the particles alone do not carry. Without it neighbouring streams of
// particles from a source, a little more or less exposed to the air, cooled into streaks.
//!include common.wgsl
//!include liq_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,  // slot capacity, share of the way to the cell's mean this step
};

@group(0) @binding(0) var<storage, read_write> parts: array<Particle>;
@group(0) @binding(1) var heat: texture_3d<f32>;   // the cells' mean heat (liq_norm.wgsl)
@group(0) @binding(2) var dens: texture_3d<f32>;   // their particle density
@group(1) @binding(0) var<uniform> U: Params;

// The cells' mean heat at x (cells), each cell weighted by how full of particles it is (the empty ones, in
// the air, read as fully hot).
fn cell_heat(x: vec3<f32>, n: vec3<i32>) -> f32 {
  let q = x - vec3<f32>(0.5);
  let b = vec3<i32>(floor(q));
  let f = q - floor(q);
  var s = 0.0;
  var ws = 0.0;
  for (var o = 0; o < 8; o++) {
    let oo = vec3<i32>(o & 1, (o >> 1) & 1, o >> 2);
    let c = clamp(b + oo, vec3<i32>(0), n - vec3<i32>(1));
    let wv = mix(vec3<f32>(1.0) - f, f, vec3<f32>(oo));
    let w = wv.x * wv.y * wv.z * textureLoad(dens, c, 0).x;
    s += w * textureLoad(heat, c, 0).x;
    ws += w;
  }
  return s / max(ws, 1e-6);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.k.x)) { return; }
  var P = parts[i];
  if (!alive(P) || P.p.w <= 1.5 * U.g.bc.w) { return; }
  let hc = clamp(cell_heat(P.p.xyz, gdim(U.g)), 0.0, 1.0);
  set_heat(&P, mix(heat_of(P), hc, U.k.y));
  parts[i].c.w = P.c.w;
}
