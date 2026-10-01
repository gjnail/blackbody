// The box moves by whole cells: the deep-liquid flags (narrow band) and the ground's wetness move
// with it. Cells the box moves over start empty (not deep, dry).
//!include common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,  // shift of the box (cells, x y z), narrow band on (1/0)
};

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var wet_src: texture_2d<f32>;
@group(0) @binding(3) var wet_dst: texture_storage_2d<r32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let s = c + vec3<i32>(U.k.xyz);
  let inside = all(s >= vec3<i32>(0)) && all(s < n);
  if (U.k.w > 0.5) {
    var v = 0u;
    if (inside) { v = src[u32(s.x + n.x * (s.y + n.y * s.z))]; }
    dst[u32(c.x + n.x * (c.y + n.y * c.z))] = v;
  }
  if (c.y == 0) {
    var w = 0.0;
    if (inside) { w = textureLoad(wet_src, s.xz, 0).x; }
    textureStore(wet_dst, c.xz, vec4<f32>(w, 0.0, 0.0, 0.0));
  }
}
