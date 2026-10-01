// Clouds: what each cell holds for the eye (engine/cloud_render.py), one thread per cell, before the light
// is traced through it (cloud_light.wgsl):
//   out_s = (the extinction (1/m) of its cloud: droplets and ice crystals, and snow in air near saturation
//            over ice, as in an anvil, the cloud whose edges the renderer sharpens; of its precipitation:
//            rain, graupel and hail, and snow falling out into dry air, soft shafts and fallstreaks; the
//            cloud that shades the rest, with the precipitation; _)
//   out_m = (how much of the cloud is ice, how much of the precipitation (ice crystals scatter less sharply
//            forward than droplets); how much of the space round the cell is cloud, for the renderer to
//            draw the cloud's edge through; how much the cell is at a cloud's base, cloud above it and
//            clear air below, where the renderer keeps the edge flat: the condensation level)
//!include cloud_common.wgsl

@group(0) @binding(0) var A: texture_3d<f32>;
@group(0) @binding(1) var B: texture_3d<f32>;
@group(0) @binding(2) var out_s: texture_storage_3d<rgba32float, write>;
@group(0) @binding(3) var out_m: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var<storage, read> base: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> U: SigParams;

struct SigParams {
  g: Grid,
  sun: vec4<f32>,   // (unused here)
  ext: vec4<f32>,   // extinction per kg of cloud water, cloud ice, rain, snow (m^2/kg)
  ext2: vec4<f32>,  // ... graupel/hail, density scale, _, _
};

// the cell's humidity (over ice below freezing, over water above)
fn humidity(c: vec3<i32>) -> f32 {
  let a = textureLoad(A, c, 0);
  let b0 = base[u32(c.y) * 2u];
  let tc = (b0.x + a.x) * b0.y - CK;
  let p = base[u32(c.y) * 2u + 1u].w;
  return a.y / select(c_qvs_w(tc, p), c_qvs_i(tc, p), tc < 0.0);
}

// how much of a cell's snow counts as cloud: in air near saturation (over ice) it is the cloud's, in dry air
// it is falling out; and high in the cold upper cloud, where the anvil is, its snow is the anvil's ice
// whatever the humidity
fn snow_share(c: vec3<i32>, rh: f32) -> f32 {
  let b0 = base[u32(c.y) * 2u];
  let tc = (b0.x + textureLoad(A, c, 0).x) * b0.y - CK;
  return max(smoothstep(0.5, 0.8, rh), smoothstep(-15.0, -30.0, tc));
}

// a cell's cloud as a share of a full cloud's (droplet cloud 30 per km, ice cloud 2), up to 1
fn full_share(c: vec3<i32>, sat: f32) -> f32 {
  let a = textureLoad(A, c, 0);
  let b = textureLoad(B, c, 0);
  let ice = U.ext.y * b.x + sat * U.ext.w * b.y;
  let cloud = ice + U.ext.x * (a.z + b.w);
  if (cloud <= 0.0) { return 0.0; }
  let full = mix(0.03, 0.002, ice / cloud);
  return clamp(base[u32(c.y) * 2u].z * U.ext2.y * cloud / full, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let a = textureLoad(A, c, 0);
  let b = textureLoad(B, c, 0);
  let rs = base[u32(c.y) * 2u].z * U.ext2.y;
  // (the humidity averaged with the six neighbours': the snow's share in the cloud changes smoothly)
  var rh = humidity(c);
  for (var ax = 0; ax < 3; ax++) {
    var e = vec3<i32>(0);
    e[ax] = 1;
    rh += humidity(clamp(c + e, vec3<i32>(0), n - vec3<i32>(1))) + humidity(clamp(c - e, vec3<i32>(0), n - vec3<i32>(1)));
  }
  let sat = snow_share(c, rh / 7.0);
  let snow = U.ext.w * b.y;
  let cice = U.ext.y * b.x + sat * snow;
  let cloud = cice + U.ext.x * (a.z + b.w);
  let pice = (1.0 - sat) * snow + U.ext2.x * b.z;
  let fall = pice + U.ext.z * a.w;
  // how much of the space round the cell is cloud: the share of a full cloud, blurred over the cell and its
  // 26 neighbours (weights 1 2 1 along each axis; so it falls from 1 inside a cloud to 0 outside over about
  // a cell either side of its edge, however dense the cloud, and rounds the grid's corners), but at least
  // most of the cell's own (a cloud one cell across stays): the renderer puts the edge where it crosses
  // the noise
  var blur = 0.0;
  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let q = c + vec3<i32>(dx, dy, dz);
        if (any(q < vec3<i32>(0)) || any(q >= n)) { continue; }
        let wt = f32((2 - abs(dx)) * (2 - abs(dy)) * (2 - abs(dz)));
        blur += wt * full_share(q, snow_share(q, humidity(q)));
      }
    }
  }
  let mine = full_share(c, sat);
  let cov = max(0.6 * mine, blur / 64.0);
  // the cloud that shades the rest: its solid part, where the space round it is mostly cloud. (Its ragged
  // edge, which the noise leaves only part of, is left out: the renderer traces the light through that
  // itself near each point, and a point on a cloud's surface has, by its being there, clear air sunward
  // of it where the edge is, not the edge's average)
  let ic = select(0.0, cice / cloud, cloud > 0.0);
  let drawn = rs * cloud * smoothstep(0.55, 0.9, cov);
  textureStore(out_s, c, vec4<f32>(rs * cloud, rs * fall, drawn + rs * fall, 0.0));
  // a cloud's base: cloud above (the two cells over this one: a cloud more than a couple of cells deep, not
  // a thin shred), clear air below; the air condenses as it rises through one level
  var below = 0.0;
  var above = 1.0;
  if (c.y > 0) { let q = c - vec3<i32>(0, 1, 0); below = full_share(q, snow_share(q, humidity(q))); }
  for (var k = 1; k <= 2; k++) {
    if (c.y + k >= n.y) { above = 0.0; break; }
    let q = c + vec3<i32>(0, k, 0);
    above = min(above, full_share(q, snow_share(q, humidity(q))));
  }
  let base_of = clamp(above - below, 0.0, 1.0) * (1.0 - ic);
  textureStore(out_m, c, vec4<f32>(ic, select(0.0, pice / fall, fall > 0.0), cov, base_of));
}
