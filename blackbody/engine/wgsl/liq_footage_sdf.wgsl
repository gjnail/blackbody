// The footage's own surfaces as solids for the liquid: its depth pass says how far the camera sees
// along each pixel, so a slab just behind that surface (a real wall, a real step, a kerb) becomes
// solid and the liquid splashes against it and runs over it. Only what the camera sees is known: the
// slab is a fixed thickness deep. Min-ed into the collider distance (cells, negative inside).
//!include common.wgsl

struct Params {
  g: Grid,
  vp: mat4x4<f32>,     // world -> clip
  l2w: mat4x4<f32>,    // fire-local -> world
  eye: vec4<f32>,      // camera position (world), slab thickness (m)
  fwd: vec4<f32>,      // camera forward (world)
  hold: vec4<f32>,     // depth kind (0 Z, 1 distance, 2 inverse Z), metres per unit, footage fit scale (x, y)
};

@group(0) @binding(0) var hold: texture_2d<f32>;
@group(0) @binding(1) var sdf: texture_storage_3d<r32float, read_write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let w = world_of(U.g, vec3<f32>(c) + 0.5);
  let pw = (U.l2w * vec4<f32>(w, 1.0)).xyz;
  let clip = U.vp * vec4<f32>(pw, 1.0);
  if (clip.w <= 1e-4) { return; }
  let ndc = clip.xy / clip.w;
  let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let huv = (uv - vec2<f32>(0.5)) * U.hold.zw + vec2<f32>(0.5);
  if (any(huv < vec2<f32>(0.0)) || any(huv >= vec2<f32>(1.0))) { return; }
  let dims = vec2<f32>(textureDimensions(hold));
  let v = textureLoad(hold, vec2<i32>(huv * dims), 0).y;
  if (!(v > 0.0) || v > 1.0e20) { return; }
  let kind = i32(U.hold.x + 0.5);
  var d = v * U.hold.y;
  if (kind == 2) { d = U.hold.y / v; }
  let rd = normalize(pw - U.eye.xyz);
  if (kind != 1) { d = d / max(dot(rd, U.fwd.xyz), 1e-3); }
  let s = d - length(pw - U.eye.xyz);          // in front of the footage's surface (m)
  let phi = max(s, -(s + U.eye.w)) / U.g.n.w;  // the slab behind it, in cells
  let old = textureLoad(sdf, c).x;
  textureStore(sdf, c, vec4<f32>(min(old, phi), 0.0, 0.0, 0.0));
}
