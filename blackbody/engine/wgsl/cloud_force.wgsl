// Clouds: forces on the air (engine/cloud.py), per MAC face, each substep.
//  - buoyancy: warm air rises, cool air sinks, by its virtual temperature against the background's (moist
//    air is lighter: vapour weighs 18 to dry air's 29), and the weight of the cloud water, rain, ice, snow
//    and hail it carries drags it down (precipitation loading: the downdraft under a storm);
//  - vorticity confinement: the small eddies the grid would smear are given back their spin (the
//    cauliflower edges of a growing cumulus);
//  - near the top, a sponge calms the air toward the background wind (as the stratosphere damps the
//    waves a storm sends up, and the top is not a lid that echoes them back);
//  - at the open sides the air is drawn toward the background wind of its height (wind shear);
//  - the ground and terrain (colliders) hold it: no flow through them.
//!include cloud_common.wgsl

@group(0) @binding(0) var vel: texture_3d<f32>;
@group(0) @binding(1) var A: texture_3d<f32>;
@group(0) @binding(2) var B: texture_3d<f32>;
@group(0) @binding(3) var curl: texture_3d<f32>;
@group(0) @binding(4) var sdf: texture_3d<f32>;
@group(0) @binding(5) var dst: texture_storage_3d<${VELFMT}, write>;
@group(0) @binding(6) var<storage, read> base: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> U: Cloud;

fn buoy(c: vec3<i32>, n: vec3<i32>) -> f32 {
  let q = clamp(c, vec3<i32>(0), n - vec3<i32>(1));
  let a = textureLoad(A, q, 0);
  let b = textureLoad(B, q, 0);
  let b0 = base[u32(q.y) * 2u];
  return CG * (a.x / b0.x + 0.608 * (a.y - b0.w) - (a.z + a.w + b.x + b.y + b.z));
}

fn solid(c: vec3<i32>, n: vec3<i32>) -> bool {
  if (!in_grid(c, n)) { return false; }
  return textureLoad(sdf, c, 0).x < 0.0;
}

// vorticity confinement force at a cell centre: eps h (N x omega), N toward stronger spin
fn confine(c: vec3<i32>, n: vec3<i32>) -> vec3<f32> {
  let q = clamp(c, vec3<i32>(1), n - vec3<i32>(2));
  let w = textureLoad(curl, q, 0);
  let gx = textureLoad(curl, q + vec3<i32>(1, 0, 0), 0).w - textureLoad(curl, q - vec3<i32>(1, 0, 0), 0).w;
  let gy = textureLoad(curl, q + vec3<i32>(0, 1, 0), 0).w - textureLoad(curl, q - vec3<i32>(0, 1, 0), 0).w;
  let gz = textureLoad(curl, q + vec3<i32>(0, 0, 1), 0).w - textureLoad(curl, q - vec3<i32>(0, 0, 1), 0).w;
  let g = vec3<f32>(gx, gy, gz);
  let gl = length(g);
  if (gl < 1e-9) { return vec3<f32>(0.0); }
  return U.vort.x * U.g.n.w * cross(g / gl, w.xyz);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c > n)) { return; }
  let dt = U.g.bc.w;
  var v = textureLoad(vel, c, 0);
  let ny = n.y;
  let jj = clamp(c.y, 0, ny - 1);
  let w0 = base[u32(jj) * 2u + 1u];
  let bw = vec3<f32>(w0.x, 0.0, w0.y);
  for (var k = 0u; k < 3u; k++) {
    var e = vec3<i32>(0);
    e[k] = 1;
    var lim = n - vec3<i32>(1);
    lim[k] = n[k];
    if (any(c > lim)) { v[k] = 0.0; continue; }
    let lo = c - e;
    // ground, lid and terrain: no flow through
    if (k == 1u && (c.y == 0 || c.y == n.y)) { v[k] = 0.0; continue; }
    if (solid(lo, n) || solid(c, n)) { v[k] = 0.0; continue; }
    var f = 0.0;
    if (k == 1u) { f += 0.5 * (buoy(lo, n) + buoy(c, n)); }
    f += 0.5 * (confine(lo, n)[k] + confine(c, n)[k]);
    var val = v[k] + f * dt;
    // the sponge under the lid, and the open sides drawn toward the background wind
    let top = f32(ny) - f32(c.y);
    let sp = clamp(1.0 - top / max(U.k.z, 1.0), 0.0, 1.0);
    var rate = sp * sp / 60.0;
    if (U.g.bc.x > 0.5) {
      let edge = f32(min(min(c.x, n.x - c.x), min(c.z, n.z - c.z)));
      rate = max(rate, U.k.w * clamp(1.0 - edge / 4.0, 0.0, 1.0));
    }
    val = mix(val, bw[k], 1.0 - exp(-rate * dt));
    v[k] = clamp(val, -90.0, 90.0);   // (safety: faster than any storm's winds)
  }
  textureStore(dst, c, v);
}
