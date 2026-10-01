// Fabric and objects in the light volume: the cloth's extinction (cloth_occ.wgsl) goes into the light
// volume's (E.a, into T1 to be copied back), so the key light, the sky, the lights in the set and the
// fire light on the smoke and the cloth are all shadowed by it. Separately (CO), the cloth's own
// extinction and that of the colliders, for shadows on the footage (raymarch.wgsl) and on the cloth
// (cloth_draw.wgsl); the colliders are left out of E so the smoke's own lighting is unchanged.
//!include common.wgsl
//!include meshsdf.wgsl
//!include colliders.wgsl

const OCC_K: f32 = 1.0e6;
const SOLID: f32 = 40.0;   // optical depth across one light cell of solid object

struct Params {
  ld: vec4<f32>,     // light-grid dims, _
  org: vec4<f32>,    // light-grid corner (fire-local m), _
  lc: vec4<f32>,     // light cell size (m, per axis), _
  ccnt: vec4<f32>,
  col: array<Collider, MAX_COLLIDERS>,
};

@group(0) @binding(0) var E: texture_3d<f32>;
@group(0) @binding(1) var<storage, read> OCC: array<u32>;
@group(0) @binding(2) var atlas: texture_3d<f32>;
@group(0) @binding(3) var T1: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var CO: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(U.ld.xyz);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let e = textureLoad(E, c, 0);
  let lc = U.lc.xyz;
  let sc = f32(OCC[u32((c.z * n.y + c.y) * n.x + c.x)]) / OCC_K / (lc.x * lc.y * lc.z);
  var so = 0.0;
  let p = U.org.xyz + (vec3<f32>(c) + 0.5) * lc;
  let l = max(lc.x, max(lc.y, lc.z));
  for (var i = 0; i < i32(U.ccnt.x); i++) {
    let d = col_sdf(U.col[i], p);
    so = max(so, clamp(0.5 - d / l, 0.0, 1.0));
  }
  textureStore(T1, c, vec4<f32>(e.rgb, e.a + sc));
  textureStore(CO, c, vec4<f32>(sc, so * SOLID / min(lc.x, min(lc.y, lc.z)), 0.0, 0.0));
}
