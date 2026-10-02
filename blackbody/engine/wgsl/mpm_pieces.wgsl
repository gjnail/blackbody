// Broken objects' pieces on the matter's grid (matter.py): before the pieces are folded in (bodies_sdf.wgsl, the grid
// taken as cells centred on the matter's nodes), nothing is near: the distance far.

struct Params {
  n: vec4<f32>,      // nodes (x, y, z), _
};

@group(0) @binding(0) var sdf: texture_storage_3d<r32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn clear(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (any(c >= vec3<i32>(U.n.xyz))) { return; }
  textureStore(sdf, c, vec4<f32>(1.0e9, 0.0, 0.0, 0.0));
}
