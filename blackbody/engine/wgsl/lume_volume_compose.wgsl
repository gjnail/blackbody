// The light volume the ray march reads when Lume lights the smoke (lume_volume.wgsl): Lume's light (LV) in its
// fire-light channel (the march scales it by Fire light scatter, set to 1 then), the key light's transmittance kept,
// and the sky's term taken out of L1 (LV holds the sky's light already, blocked by the set as it is). The key light's
// transmittance taken over the length of the light volume's steps toward it (lume_volume.wgsl, sun_tr).
//!include common.wgsl

@group(0) @binding(0) var lv: texture_3d<f32>;
@group(0) @binding(1) var L0: texture_3d<f32>;
@group(0) @binding(2) var L1: texture_3d<f32>;
@group(0) @binding(3) var lin: sampler;
@group(0) @binding(4) var out0: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var out1: texture_storage_3d<rgba16float, write>;

struct Params {
  ld: vec4<f32>,   // the light volume's dims
  lvd: vec4<f32>,  // LV's dims, the length of the light volume's steps toward the key light (cells)
};
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (any(c >= vec3<i32>(U.ld.xyz))) { return; }
  let q = (vec3<f32>(c) + 0.5) * (U.lvd.xyz / U.ld.xyz);
  let light = samp_c(lv, lin, q, U.lvd.xyz).rgb;
  let l0 = textureLoad(L0, c, 0);
  let l1 = textureLoad(L1, c, 0);
  textureStore(out0, c, vec4<f32>(light, pow(max(l0.a, 1e-30), U.lvd.w)));
  textureStore(out1, c, vec4<f32>(0.0, l1.yzw));
}
