// Light volumes at half resolution.
// out0: rgb = fire light arriving here (blurred emission), a = key-light transmittance
// out1: x = sky transmittance (looking straight up), y = occupancy (for empty-space skipping)
//!include common.wgsl

struct Params {
  ln: vec4<f32>,    // light dims; w = metres per light cell
  sun: vec4<f32>,   // direction toward the key light in light-grid space; w = march steps
  k: vec4<f32>,     // shadow density multiplier, sky shadow multiplier, fire-light scale, _
};

@group(0) @binding(0) var E: texture_3d<f32>;
@group(0) @binding(1) var EB: texture_3d<f32>;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var out0: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var out1: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = U.ln.xyz;
  let c = vec3<i32>(id);
  if (any(c >= vec3<i32>(n))) { return; }
  let p = vec3<f32>(c) + 0.5;
  let steps = i32(U.sun.w);

  var od = 0.0;
  var q = p;
  for (var i = 0; i < steps; i++) {
    q += U.sun.xyz;
    if (any(q < vec3<f32>(0.0)) || any(q > n)) { break; }
    od += samp_c(E, lin, q, n).a;
  }
  let sun_tr = exp(-od * U.ln.w * U.k.x);

  var od_up = 0.0;
  q = p;
  for (var i = 0; i < steps; i++) {
    q.y += 1.0;
    if (q.y > n.y) { break; }
    od_up += samp_c(E, lin, q, n).a;
  }
  let sky_tr = exp(-od_up * U.ln.w * U.k.y);

  let eb = textureLoad(EB, c, 0);
  let here = textureLoad(E, c, 0);
  let occ = eb.a + luma(eb.rgb) + here.a + luma(here.rgb);
  textureStore(out0, c, vec4<f32>(eb.rgb * U.k.z, sun_tr));
  textureStore(out1, c, vec4<f32>(sky_tr, occ, 0.0, 0.0));
}
