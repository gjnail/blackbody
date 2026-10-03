// Bullets into water: where the bullets' paths through this frame cross the liquid (engine/bullet_media.py). For each
// path (a segment A + s u, 0 <= s <= L), each of its NB bins is marked where the liquid's cell types (liq_setup.wgsl:
// 1 liquid, its deep inside included) say the cell it falls in is liquid.
//!include common.wgsl

const NB: u32 = 256u;     // bins along a path
const MAX_PATHS: u32 = 8u;

struct FindU {
  g: Grid,
  k: vec4<f32>,                         // paths (count), _, _, _
  pa: array<vec4<f32>, MAX_PATHS>,      // A (fire-local m), L (m)
  pu: array<vec4<f32>, MAX_PATHS>,      // u (unit), _
};

@group(0) @binding(0) var T: texture_3d<f32>;
@group(0) @binding(1) var<storage, read_write> OCC: array<u32>;
@group(1) @binding(0) var<uniform> F: FindU;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let j = id.x % NB;
  let k = id.x / NB;
  if (k >= u32(F.k.x)) { return; }
  let A = F.pa[k];
  let s = (f32(j) + 0.5) / f32(NB) * A.w;
  let w = A.xyz + F.pu[k].xyz * s;
  let c = vec3<i32>(floor((w - F.g.org.xyz) / F.g.n.w));
  let n = vec3<i32>(F.g.n.xyz);
  if (any(c < vec3<i32>(0)) || any(c >= n)) { return; }
  if (abs(textureLoad(T, c, 0).x - 1.0) < 0.5) { OCC[k * NB + j] = 1u; }
}
