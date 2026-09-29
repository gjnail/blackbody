// Lays out the burnable floor at the start of a simulation: the bottom layer of cells, optionally
// only a patch of it, skipping cells inside colliders. Fuel comes in patches so the fire front
// breaks up and has to find its way. Burnable colliders are laid out in burn_obj.wgsl.
//!include common.wgsl
//!include noise.wgsl
//!include meshsdf.wgsl
//!include colliders.wgsl

struct Params {
  g: Grid,
  a: vec4<f32>,    // floor burns (1/0), half width of the burnable patch (m, 0 = all), half depth, coverage (0..1)
  b: vec4<f32>,    // patch frequency (1/m), seed, _, _
  ccnt: vec4<f32>,
  col: array<Collider, MAX_COLLIDERS>,
};

@group(0) @binding(0) var atlas: texture_3d<f32>;
@group(0) @binding(1) var dst: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let d = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= d)) { return; }
  let h = U.g.n.w;
  let wp = world_of(U.g, vec3<f32>(c) + 0.5);
  var on = false;
  if (U.a.x > 0.5 && c.y == 0) {
    on = (U.a.y <= 0.0 || abs(wp.x) <= U.a.y) && (U.a.z <= 0.0 || abs(wp.z) <= U.a.z);
  }
  var inside = false;
  let cnt = i32(U.ccnt.x);
  for (var i = 0; i < cnt; i++) {
    let k = U.col[i];
    let s = col_sdf(k, wp) / h;
    if (s < 0.0) { inside = true; }
  }
  if (!on || inside) {
    textureStore(dst, c, vec4<f32>(0.0));
    return;
  }
  // patchy fuel: bare where the noise is above the coverage, a varying load elsewhere
  let q = wp * U.b.x + vec3<f32>(U.b.y * 3.17, 0.0, U.b.y * 1.93);
  let v = clamp(0.5 + fbm3(q, 3) * 2.2, 0.0, 1.0);
  var fuel = 1.0 - smoothstep(U.a.w - 0.06, U.a.w + 0.06, v);
  if (U.a.w >= 0.999) { fuel = 1.0; }
  fuel *= 0.75 + 0.5 * clamp(0.5 + fbm3(q * 3.1 + vec3<f32>(7.3), 2), 0.0, 1.0);
  textureStore(dst, c, vec4<f32>(fuel, 0.0, 1.0, 0.0));
}
