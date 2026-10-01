// The sea's layers after the inverse FFT, copied into the half-float texture the renderer samples
// (filtered, repeating across the tile).

@group(0) @binding(0) var src: texture_3d<f32>;
@group(0) @binding(1) var dst: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let d = textureDimensions(src);
  if (any(id >= d)) { return; }
  let v = textureLoad(src, vec3<i32>(id), 0);
  textureStore(dst, vec3<i32>(id), clamp(v, vec4<f32>(-6.0e4), vec4<f32>(6.0e4)));
}
