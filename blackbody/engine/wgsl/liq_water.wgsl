// Fire and liquid in one box: the share of each fire cell filled with liquid (from the liquid's
// particle density), which the fire solver reads to put the fire out where the water is.
struct Params {
  n: vec4<f32>,   // fire grid dims
  nl: vec4<f32>,  // liquid grid dims; w = rest density (particles)
};

@group(0) @binding(0) var dens: texture_3d<f32>;
@group(0) @binding(1) var dst: texture_storage_3d<r32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (any(c >= vec3<i32>(U.n.xyz))) { return; }
  let q = min(vec3<i32>(floor((vec3<f32>(c) + vec3<f32>(0.5)) * U.nl.xyz / U.n.xyz)), vec3<i32>(U.nl.xyz) - vec3<i32>(1));
  let f = clamp(textureLoad(dens, q, 0).x / max(U.nl.w, 1e-3), 0.0, 1.0);
  textureStore(dst, c, vec4<f32>(f, 0.0, 0.0, 0.0));
}
