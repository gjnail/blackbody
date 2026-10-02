// Matter's surface, for drawing: at each node, its distance to the surface (Zhu and Bridson 2005: from the weighted
// mean of the particles round it, less a particle's radius; outside the band round them, the band's width) and the
// look of the matter there (look2: its temperature, K, for its glow, and how metallic it is).
//!include mpm_common.wgsl

struct Params {
  n: vec4<f32>,      // grid nodes (x, y, z), _
  r: vec4<f32>,      // R (grid units), a particle's radius (grid units), node spacing (m), _
};

@group(0) @binding(0) var<storage, read> S: array<i32>;
@group(0) @binding(1) var phi: texture_storage_3d<r32float, write>;
@group(0) @binding(2) var look0: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var look1: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var look2: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

const SURF: u32 = 13u;
const FX_S: f32 = 65536.0;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(U.n.xyz);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let k = nidx(c, n) * SURF;
  let w = f32(S[k]) / FX_S;
  let dx = U.r.z;
  let band = (U.r.x - U.r.y) * dx;          // (no particle within R: the surface is at least this far)
  if (w < 1.0e-4) {
    textureStore(phi, c, vec4<f32>(band));
    textureStore(look0, c, vec4<f32>(0.0));
    textureStore(look1, c, vec4<f32>(0.0));
    textureStore(look2, c, vec4<f32>(0.0));
    return;
  }
  let iw = 1.0 / (w * FX_S);
  let mean = vec3<f32>(f32(S[k + 1u]), f32(S[k + 2u]), f32(S[k + 3u])) * iw;
  // (where there is little matter round the node, the surface comes in toward the particles)
  let d = (length(mean) - U.r.y * min(w / 0.5, 1.0)) * dx;
  textureStore(phi, c, vec4<f32>(min(d, band)));
  textureStore(look0, c, vec4<f32>(vec3<f32>(f32(S[k + 4u]), f32(S[k + 5u]), f32(S[k + 6u])) * iw, f32(S[k + 7u]) * iw));
  textureStore(look1, c, vec4<f32>(f32(S[k + 8u]) * iw, f32(S[k + 9u]) * iw, f32(S[k + 10u]) * iw, 1.0));
  textureStore(look2, c, vec4<f32>(f32(S[k + 11u]) * iw * 16.0, f32(S[k + 12u]) * iw, 0.0, 0.0));
}
