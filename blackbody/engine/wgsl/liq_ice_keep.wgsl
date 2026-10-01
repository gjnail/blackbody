// Ice keeps its edges: after the surface is smoothed (liq_surf_blur, liq_surf_flatten), where it is frozen
// it goes back to the unsmoothed surface, so ice cubes stay square and frozen splashes keep their
// spikes and crowns.

struct Params {
  nf: vec4<f32>,   // surface grid dims; w = how much of its own shape ice keeps (melting ice is wet and smoother)
};

@group(0) @binding(0) var raw: texture_3d<f32>;
@group(0) @binding(1) var blurred: texture_3d<f32>;
@group(0) @binding(2) var heat: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (any(c >= vec3<i32>(U.nf.xyz))) { return; }
  let s = textureLoad(blurred, c, 0);
  let ice = smoothstep(0.3, 0.7, textureLoad(heat, c, 0).w) * U.nf.w;
  textureStore(dst, c, mix(s, textureLoad(raw, c, 0), ice));
}
