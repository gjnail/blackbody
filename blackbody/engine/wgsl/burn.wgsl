// Burnable floor: one step of catching, flaming, smouldering and burning out for the bottom layer of
// cells (see burn_common.wgsl). Burnable colliders are stepped in burn_obj.wgsl.
//!include common.wgsl
//!include noise.wgsl
//!include meshsdf.wgsl
//!include emitters.wgsl
//!include burn_common.wgsl

struct Params {
  g: Grid,
  sp: vec4<f32>,   // catch temperature, 1 / catch time (1/s), creep speed (m/s), 1 / burn time (1/s)
  sp2: vec4<f32>,  // 1 / smoulder time (1/s), water on (1/0), how fast water puts a surface out (1/s), _
  cnt: vec4<f32>,  // emitter count
  em: array<Emitter, MAX_EMITTERS>,
};

@group(0) @binding(0) var scal: texture_3d<f32>;
@group(0) @binding(1) var src: texture_3d<f32>;
@group(0) @binding(2) var atlas: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var water: texture_3d<f32>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let d = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= d)) { return; }
  let b = textureLoad(src, c, 0);
  if (b.z < 0.5) {
    textureStore(dst, c, b);
    return;
  }
  let h = U.g.n.w;
  let T = textureLoad(scal, c, 0).x;
  var wet = 0.0;
  let wp = world_of(U.g, vec3<f32>(c) + 0.5);
  for (var i = 0; i < i32(U.cnt.x); i++) {
    let e = U.em[i];
    if (e.k.w > 0.0) { wet += e.k.w * emitter_mask(e, wp, h); }
  }
  if (U.sp2.y > 0.5) { wet += U.sp2.z * clamp(textureLoad(water, c, 0).x, 0.0, 1.0); }
  var nb = 0.0;
  if (b.y < 1.0 && b.x > 0.0) {
    for (var k = 0; k < 27; k++) {
      let o = vec3<i32>(k % 3, (k / 3) % 3, k / 9) - vec3<i32>(1);
      if (k == 13) { continue; }
      let q = c + o;
      if (!in_grid(q, d)) { continue; }
      if (burn_alight(textureLoad(src, q, 0))) { nb = max(nb, 1.0 / length(vec3<f32>(o))); }
    }
  }
  textureStore(dst, c, burn_step(b, T, nb, wet, U.g.bc.w, h, U.sp, U.sp2.x));
}
