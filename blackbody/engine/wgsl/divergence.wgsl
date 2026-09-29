// Right-hand side of the pressure equation: h^2 * (div u - expansion), zero inside solids.
//!include common.wgsl

struct Params { g: Grid };

@group(0) @binding(0) var vel: texture_3d<f32>;
@group(0) @binding(1) var expo: texture_3d<f32>;
@group(0) @binding(2) var sdf: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<r32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let d = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= d)) { return; }
  let h = U.g.n.w;
  let a = textureLoad(vel, c, 0).xyz;
  let ux = textureLoad(vel, c + vec3<i32>(1, 0, 0), 0).x;
  let vy = textureLoad(vel, c + vec3<i32>(0, 1, 0), 0).y;
  let wz = textureLoad(vel, c + vec3<i32>(0, 0, 1), 0).z;
  var div = (ux - a.x + vy - a.y + wz - a.z) / h - textureLoad(expo, c, 0).x;
  if (textureLoad(sdf, c, 0).x < 0.0) { div = 0.0; }
  textureStore(dst, c, vec4<f32>(div * h * h, 0.0, 0.0, 0.0));
}
