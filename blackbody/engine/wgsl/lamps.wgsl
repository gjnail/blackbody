// Lights in the set (Scene.lights): how much of each lamp's light the smoke and steam let through to
// every cell of the light volume (a march through E.a toward the lamp). The ray march works out each
// lamp's light itself at every step (its falloff, a spot's cone, an area light's facing), exactly, so
// a narrow beam stays as sharp as the march; only this smooth shadow comes from the light volume.
//
// Lamp buffer (4 vec4 per lamp, fire-local metres):
//   p: position, radius
//   c: power (rgb, in key-light units at one metre), kind (0 point, 1 spot, 2 area)
//   d: aim (unit), cos of the cone's outer angle
//   e: cos of the cone's inner angle, smoke shadows (1/0), in the footage (1/0), _
//
// Output: the light volume `blocks` times deep along z; block b holds lamps 4b..4b+3 in rgba.
//!include common.wgsl

struct Lamp { p: vec4<f32>, c: vec4<f32>, d: vec4<f32>, e: vec4<f32> };

struct Params {
  ln: vec4<f32>,   // light-volume dims (one block), blocks
  lc: vec4<f32>,   // light cell size (m, per axis), shadow density multiplier
  org: vec4<f32>,  // grid corner (fire-local m), lamp count
  k: vec4<f32>,    // most march steps, _, _, _
};

@group(0) @binding(0) var E: texture_3d<f32>;
@group(0) @binding(1) var lin: sampler;
@group(0) @binding(2) var<storage, read> lamps: array<Lamp>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

// Transmittance from light cell c toward lamp L (1 where the lamp does not shine: nothing reads it).
fn lamp_tr(L: Lamp, c: vec3<i32>) -> f32 {
  if (L.e.y < 0.5) { return 1.0; }
  let n = U.ln.xyz;
  let lc = U.lc.xyz;
  let pm = U.org.xyz + (vec3<f32>(c) + 0.5) * lc;
  let v = L.p.xyz - pm;
  let dist = length(v);
  if (dist < 1e-5) { return 1.0; }
  let dir = v / dist;
  // outside a spot's cone or behind an area light: skip, with a margin of two cells so that the
  // trilinear lookup at the beam's edge still blends only marched cells
  let kind = i32(L.c.w + 0.5);
  let facing = dot(-dir, L.d.xyz);
  let margin = atan2(2.0 * max(lc.x, max(lc.y, lc.z)), dist);
  if (kind == 1 && acos(clamp(facing, -1.0, 1.0)) > acos(clamp(L.d.w, -1.0, 1.0)) + margin) { return 1.0; }
  if (kind == 2 && facing < -sin(margin)) { return 1.0; }
  // march to the lamp or out of the box, whichever comes first
  let steps = min(i32(ceil(max(abs(v.x) / lc.x, max(abs(v.y) / lc.y, abs(v.z) / lc.z)))), i32(U.k.x));
  let dt = dist / f32(max(steps, 1));
  var od = 0.0;
  for (var s = 0; s < steps; s++) {
    let q = (pm + dir * ((f32(s) + 0.5) * dt) - U.org.xyz) / lc;
    if (any(q < vec3<f32>(0.0)) || any(q > n)) { break; }
    od += samp_c(E, lin, q, n).a * dt;
  }
  return exp(-od * U.lc.w);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(U.ln.xyz);
  let blocks = i32(U.ln.w);
  let g = vec3<i32>(id);
  if (g.x >= n.x || g.y >= n.y || g.z >= n.z * blocks) { return; }
  let b = g.z / n.z;
  let c = vec3<i32>(g.x, g.y, g.z - b * n.z);
  let count = i32(U.org.w);
  var t = vec4<f32>(1.0);
  for (var j = 0; j < 4; j++) {
    let i = b * 4 + j;
    if (i >= count) { break; }
    let tr = lamp_tr(lamps[i], c);
    if (j == 0) { t.x = tr; } else if (j == 1) { t.y = tr; } else if (j == 2) { t.z = tr; } else { t.w = tr; }
  }
  textureStore(dst, g, t);
}
