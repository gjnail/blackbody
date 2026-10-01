// Gas <- fabric: the momentum the cloth's drag took from the air goes back into the gas (both feel the
// same force, opposite ways), spread over the cells around the cloth, never pushing the air past the
// cloth's own speed. So a curtain slows the draught through a doorway, and a flapping flag stirs the
// air behind it.
//!include common.wgsl

const AREA_K: f32 = 1.0e6;
const MOM_K: f32 = 1.0e9;
const CH: u32 = 13u;
const RHO_AIR: f32 = 1.2;

struct Params {
  g: vec4<f32>,      // simulation grid dims, cell size (m)
  cg: vec4<f32>,     // coupling grid dims, its cell size (m)
};

@group(0) @binding(0) var vel: texture_3d<f32>;
@group(0) @binding(1) var<storage, read> G: array<i32>;
@group(0) @binding(2) var dst: texture_storage_3d<${VELFMT}, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn gat(c: vec3<i32>, ch: u32) -> f32 {
  let d = vec3<i32>(U.cg.xyz);
  if (any(c < vec3<i32>(0)) || any(c >= d)) { return 0.0; }
  return f32(G[u32((c.z * d.y + c.y) * d.x + c.x) * CH + ch]);
}

// (momentum given to the air per m^3, the cloth's velocity, whether there is cloth) along axis a at grid
// point p (cells)
fn cloth_at(p: vec3<f32>, a: u32) -> vec3<f32> {
  let q = p * (U.g.w / U.cg.w) - vec3<f32>(0.5);
  let b = vec3<i32>(floor(q));
  let f = q - floor(q);
  var W = 0.0;
  var Q = 0.0;
  var Mo = 0.0;
  for (var k = 0; k < 8; k++) {
    let o = vec3<i32>(k & 1, (k >> 1) & 1, k >> 2);
    let w3 = mix(vec3<f32>(1.0) - f, f, vec3<f32>(o));
    let w = w3.x * w3.y * w3.z;
    W += w * gat(b + o, a);
    Q += w * gat(b + o, a + 3u);
    Mo += w * gat(b + o, a + 9u);
  }
  if (W <= 0.0) { return vec3<f32>(0.0); }
  return vec3<f32>(Mo / MOM_K / (U.cg.w * U.cg.w * U.cg.w), Q / W, 1.0);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(U.g.xyz);
  let c = vec3<i32>(id);
  if (any(c > n)) { return; }
  var v = textureLoad(vel, c, 0);
  let pc = vec3<f32>(c) + vec3<f32>(0.5);
  for (var a = 0u; a < 3u; a++) {
    var p = pc;
    if (a == 0u) { p.x -= 0.5; } else if (a == 1u) { p.y -= 0.5; } else { p.z -= 0.5; }
    let cl = cloth_at(p, a);
    if (cl.z <= 0.0) { continue; }
    var va = v.x;
    if (a == 1u) { va = v.y; } else if (a == 2u) { va = v.z; }
    // the push, but no further than the cloth's own speed
    let gap = cl.y - va;
    var dv = cl.x / RHO_AIR;
    if (dv * gap > 0.0) { dv = sign(dv) * min(abs(dv), abs(gap)); }
    if (a == 0u) { v.x += dv; } else if (a == 1u) { v.y += dv; } else { v.z += dv; }
  }
  textureStore(dst, c, v);
}
