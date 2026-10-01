// Two liquids and dye, pass 2: the cell means of what the particles carry. ATTR holds the dye
// (absorption per metre rgb, scattering per metre); RHO the density relative to the liquid, as
// (density x liquid fraction, liquid fraction, density) so it can be averaged over a neighbourhood
// weighted by how much liquid is there.
//!include common.wgsl
//!include liq_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,  // particles per cell
};

@group(0) @binding(0) var<storage, read> acc: array<i32>;
@group(0) @binding(1) var dens: texture_3d<f32>;
@group(0) @binding(2) var attr_t: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var rho_t: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let dw = textureLoad(dens, c, 0).x;
  let s = nidx(c, n) * ATTR_SLOTS;
  var dye = vec4<f32>(0.0);
  var rho = 1.0;
  if (dw > 1e-4) {
    dye = vec4<f32>(f32(acc[s + 1u]), f32(acc[s + 2u]), f32(acc[s + 3u]), f32(acc[s + 4u])) / FX_A / dw;
    rho = max(1.0 + f32(acc[s]) / FX_A / dw, 1e-3);
  }
  let f = clamp(dw / max(0.5 * U.k.x, 1e-3), 0.0, 1.0);
  textureStore(attr_t, c, max(dye, vec4<f32>(0.0)));
  textureStore(rho_t, c, vec4<f32>(rho * f, f, rho, 0.0));
}
