// The footage behind what stands in it, for the liquid's march (liq_march.wgsl backdrop_behind).
//
// Rays through the liquid that reach the ground or the backdrop look the footage up where the point
// they reach is on screen. Where something nearer covers that place in the footage (an object standing
// in the water, drawn in it; a real wall or step in its depth pass; its matte) the footage there shows
// that, not what lies behind it, and the water showed a ghost of the object beside it. `cover` finds
// per pixel how far from the camera the nearest such thing seen there is; `fill` paints the footage over
// those places from the nearest pixels beside them that show what lies behind (along the row and the
// column, mirrored across the edge so the ground's texture carries on, the nearer sides weighing more),
// and keeps the distance, so the march can tell a point behind the thing from one in front of it.
//
// clean: rgb = the footage with the near things filled in (as liq_march.wgsl backdrop() returns it),
//        a = distance (m) from the camera of the near thing the footage shows there (NONE: none), negative
//        where it is only a surface of the depth pass (no collider and not the matte)
//!include colour.wgsl
//!include meshsdf.wgsl
//!include colliders.wgsl

const NONE: f32 = 60000.0;   // (fits in half floats)
const REACH: i32 = 192;      // pixels looked through each way for the footage beside a near thing

struct Params {
  inv_vp: mat4x4<f32>,  // clip -> world
  w2g: mat4x4<f32>,     // world -> grid cells
  n: vec4<f32>,         // grid dims, metres per cell
  org: vec4<f32>,       // grid corner (fire-local m); colliders are drawn in the footage (1/0)
  res: vec4<f32>,       // width, height (pixels)
  scene: vec4<f32>,     // camera position (world), backdrop distance (m)
  fwd: vec4<f32>,       // camera forward (world), ground on (1/0)
  plate: vec4<f32>,     // input transform, footage gain, fit scale x, fit scale y
  hold: vec4<f32>,      // footage holdouts: matte on, depth pass on, depth kind (0 Z, 1 distance, 2 inverse Z), metres per unit
  hold2: vec4<f32>,     // their fit scale (x, y)
  ccnt: vec4<f32>,      // collider count
  col: array<Collider, MAX_COLLIDERS>,
};

@group(0) @binding(0) var plate: texture_2d<f32>;
@group(0) @binding(1) var hold_t: texture_2d<f32>;     // x = matte, y = depth pass (Renderer.holdouts)
@group(0) @binding(2) var atlas: texture_3d<f32>;      // (mesh colliders: meshsdf.wgsl)
@group(0) @binding(3) var lin: sampler;
@group(0) @binding(4) var near_t: texture_2d<f32>;     // fill: what cover found
@group(0) @binding(5) var near_out: texture_storage_2d<r32float, write>;      // cover
@group(0) @binding(6) var clean_out: texture_storage_2d<rgba16float, write>;  // fill
@group(1) @binding(0) var<uniform> U: Params;

// The footage at screen position uv (as liq_march.wgsl backdrop() reads it).
fn foot(uv: vec2<f32>) -> vec3<f32> {
  let q = (uv - vec2<f32>(0.5)) * U.plate.zw + vec2<f32>(0.5);
  let puv = clamp(vec2<f32>(1.0) - abs(vec2<f32>(1.0) - abs(q)), vec2<f32>(0.0), vec2<f32>(1.0));
  return input_transform(textureSampleLevel(plate, lin, puv, 0.0).rgb, i32(U.plate.x)) * U.plate.y;
}

// Distance (fire-local m) to the nearest collider in the shot.
fn solid_m(p: vec3<f32>) -> f32 {
  var d = 1.0e9;
  for (var i = 0; i < i32(U.ccnt.x); i++) {
    if (U.col[i].y.w < 0.5) { continue; }   // a helper, not in the shot
    d = min(d, col_sdf(U.col[i], p));
  }
  return d;
}

@compute @workgroup_size(8, 8, 1)
fn cover(@builtin(global_invocation_id) id: vec3<u32>) {
  let px = vec2<i32>(id.xy);
  if (f32(px.x) >= U.res.x || f32(px.y) >= U.res.y) { return; }
  let uv = (vec2<f32>(px) + vec2<f32>(0.5)) / U.res.xy;
  let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let pn = U.inv_vp * vec4<f32>(ndc, 0.0, 1.0);
  let pf = U.inv_vp * vec4<f32>(ndc, 1.0, 1.0);
  let ro_w = pn.xyz / pn.w;
  let rd_w = normalize(pf.xyz / pf.w - ro_w);
  let base = length(ro_w - U.scene.xyz);
  let ro = (U.w2g * vec4<f32>(ro_w, 1.0)).xyz;
  let rd = (U.w2g * vec4<f32>(rd_w, 0.0)).xyz;   // (grid cells per metre along the ray)
  // where the ray meets the ground (m along it): what the footage shows there unless something stands nearer
  var t_ground = 1.0e9;
  if (U.fwd.w > 0.5 && rd.y < -1e-6 && ro.y > 0.0) { t_ground = -ro.y / rd.y; }
  var near = NONE;
  var object = false;   // (a collider or the matte)
  if (U.org.w > 0.5 && U.ccnt.x > 0.5) {
    // an object drawn in the footage, in front of the ground (taken a little fat, so its soft edge counts)
    let k = 1.0 / (length(rd) * U.n.w);   // metres along the ray per fire-local metre
    let tmax = min(t_ground, U.scene.w * 4.0);
    var t = 0.0;
    for (var i = 0; i < 160; i++) {
      let d = solid_m(U.org.xyz + (ro + rd * t) * U.n.w) * k;
      if (d < 0.0015 * (base + t)) {
        near = base + t;
        object = true;
        break;
      }
      t += max(d, 0.002 * (base + t));
      if (t > tmax) { break; }
    }
  }
  if (U.hold.x > 0.5 || U.hold.y > 0.5) {
    let hv = textureSampleLevel(hold_t, lin, (uv - vec2<f32>(0.5)) * U.hold2.xy + vec2<f32>(0.5), 0.0);
    if (U.hold.x > 0.5 && hv.x > 0.5) {
      near = base;   // the matte: in front of everything
      object = true;
    }
    if (U.hold.y > 0.5 && hv.y > 0.0 && hv.y < 1.0e20) {
      let kind = i32(U.hold.z + 0.5);
      var d = hv.y * U.hold.w;
      if (kind == 2) { d = U.hold.w / hv.y; }
      if (kind != 1) { d = d / max(dot(rd_w, U.fwd.xyz), 1e-3); }   // Z along the view axis -> along the ray
      // (a surface of the footage's own standing above the ground: the ground itself is no cover)
      if (d < (base + t_ground) * 0.97 - 0.05) { near = min(near, d); }
    }
  }
  textureStore(near_out, px, vec4<f32>(select(-near, near, object || near >= NONE), 0.0, 0.0, 0.0));
}

// How far the near thing cover found at pixel q is (m; NONE: none), and whether it is a collider or the matte.
fn near_px(q: vec2<i32>) -> f32 {
  return abs(textureLoad(near_t, clamp(q, vec2<i32>(0), vec2<i32>(U.res.xy) - vec2<i32>(1)), 0).x);
}
fn object_px(q: vec2<i32>) -> bool {
  let v = textureLoad(near_t, clamp(q, vec2<i32>(0), vec2<i32>(U.res.xy) - vec2<i32>(1)), 0).x;
  return v > 0.0 && v < NONE;
}

@compute @workgroup_size(8, 8, 1)
fn fill(@builtin(global_invocation_id) id: vec3<u32>) {
  let px = vec2<i32>(id.xy);
  let size = vec2<i32>(U.res.xy);
  if (any(px >= size)) { return; }
  let uv = (vec2<f32>(px) + vec2<f32>(0.5)) / U.res.xy;
  // (a pixel next to a near thing counts as one: its edge is soft in the footage)
  var near = NONE;
  var object = false;
  for (var j = 0; j < 9; j++) {
    let q = px + vec2<i32>(j % 3 - 1, j / 3 - 1);
    near = min(near, near_px(q));
    object = object || object_px(q);
  }
  if (near >= NONE) {
    textureStore(clean_out, px, vec4<f32>(foot(uv), NONE));
    return;
  }
  var acc = vec3<f32>(0.0);
  var wsum = 0.0;
  for (var a = 0; a < 4; a++) {
    let dir = select(vec2<i32>(select(1, -1, a == 1), 0), vec2<i32>(0, select(1, -1, a == 3)), a >= 2);
    for (var k = 1; k <= REACH; k++) {
      let q = px + dir * k;
      if (any(q < vec2<i32>(0)) || any(q >= size)) { break; }
      if (near_px(q) < NONE || near_px(q + dir) < NONE) { continue; }
      // q + dir is the first pixel clear of the soft edge: the footage mirrored across the edge from here,
      // or that pixel where the mirror lands on something else
      var m = px + dir * max(2 * k - 1, k + 1);
      if (any(m < vec2<i32>(0)) || any(m >= size) || near_px(m) < NONE) { m = q + dir; }
      let w = select(1.0, 0.5, a >= 2) / f32(k * k);   // (along the row first: the ground at the same distance)
      acc += w * foot((vec2<f32>(m) + vec2<f32>(0.5)) / U.res.xy);
      wsum += w;
      break;
    }
  }
  var c = foot(uv);
  if (wsum > 0.0) { c = acc / wsum; }
  textureStore(clean_out, px, vec4<f32>(c, select(-near, near, object)));
}
