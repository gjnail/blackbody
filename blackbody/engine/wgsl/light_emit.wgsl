// Half-resolution emission and extinction (smoke and steam), the input to the fire-light and shadow volumes.
//!include common.wgsl

@group(0) @binding(0) var scal: texture_3d<f32>;
@group(0) @binding(1) var aux: texture_3d<f32>;
@group(0) @binding(2) var chem: texture_3d<f32>;
@group(0) @binding(3) var bb: texture_2d<f32>;
@group(0) @binding(4) var lin: sampler;
@group(0) @binding(5) var dst: texture_storage_3d<rgba16float, write>;

//!include shade.wgsl

struct Params {
  n: vec4<f32>,    // simulation dims
  ln: vec4<f32>,   // light-volume dims
  look: Look,
};
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (any(c >= vec3<i32>(U.ln.xyz))) { return; }
  let p = (vec3<f32>(c) + 0.5) * (U.n.xyz / U.ln.xyz);
  let s = samp_c(scal, lin, p, U.n.xyz);
  var a = vec4<f32>(0.0);
  if (U.look.air.z > 0.5) { a = samp_c(aux, lin, p, U.n.xyz); }
  var ch = vec4<f32>(0.0);
  if (U.look.air.w > 0.5) { ch = samp_c(chem, lin, p, U.n.xyz); }
  textureStore(dst, c, vec4<f32>(emission(s, ch, U.look), smoke_extinction(s, U.look) + steam_extinction(s, a, U.look)));
}
