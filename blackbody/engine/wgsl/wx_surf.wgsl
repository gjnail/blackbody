// Weather: the surface the precipitation lands on, seen from above, one thread per cell of the cover map:
// the height of the topmost solid (a collider's top, or the ground), its temperature (the collider's own,
// or the ground's) and its slope (normal x, z). Found from the colliders' own distance functions (meshes
// through their baked fields), so it reaches past the liquid's box over the whole weather area. Snow, hail
// and glaze lie on it (wx_cover.wgsl), and resting hail melts by its temperature (wx_step.wgsl).
//!include common.wgsl
//!include liq_common.wgsl
//!include meshsdf.wgsl
//!include colliders.wgsl

struct Params {
  g: Grid,
  area: vec4<f32>,  // x0, z0, x1, z1 (m)
  map: vec4<f32>,   // nx, nz, ground on (1/0), ground temperature (C)
  box: vec4<f32>,   // top of the search (m), _, _, _
  ccnt: vec4<f32>,
  col: array<Collider, MAX_COLLIDERS>,
  ctemp: array<vec4<f32>, 4>,
};

@group(0) @binding(0) var sdf: texture_3d<f32>;
@group(0) @binding(1) var atlas: texture_3d<f32>;
@group(0) @binding(2) var out_surf: texture_storage_2d<rgba32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn collider_temp(k: u32) -> f32 { return U.ctemp[k / 4u][k % 4u]; }

// Distance to the nearest collider at p (m), and which.
fn solid_d(p: vec3<f32>, which: ptr<function, u32>) -> f32 {
  var best = 1.0e9;
  let cnt = u32(U.ccnt.x);
  for (var k = 0u; k < cnt; k++) {
    let d = col_sdf(U.col[k], p);
    if (d < best) {
      best = d;
      *which = k;
    }
  }
  return best;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let nx = u32(U.map.x);
  let nz = u32(U.map.y);
  if (id.x >= nx || id.y >= nz) { return; }
  let x = mix(U.area.x, U.area.z, (f32(id.x) + 0.5) / f32(nx));
  let z = mix(U.area.y, U.area.w, (f32(id.y) + 0.5) / f32(nz));
  let ground = U.map.z > 0.5;
  var height = select(-1.0e3, 0.0, ground);
  var temp = U.map.w;
  var nrm = vec2<f32>(0.0);
  if (u32(U.ccnt.x) > 0u) {
    // sphere-trace down the column from the top of the search to the first solid (or the ground)
    let cell = (U.area.z - U.area.x) / f32(nx);
    let eps = max(0.25 * cell, 0.002);
    var y = U.box.x;
    var which = 0u;
    for (var it = 0; it < 256; it++) {
      if (y < select(-50.0, 0.0, ground)) { break; }
      let d = solid_d(vec3<f32>(x, y, z), &which);
      if (d < eps) {
        var w2 = 0u;
        let e = max(cell, 0.005);
        let gx = solid_d(vec3<f32>(x + e, y, z), &w2) - solid_d(vec3<f32>(x - e, y, z), &w2);
        let gy = solid_d(vec3<f32>(x, y + e, z), &w2) - solid_d(vec3<f32>(x, y - e, z), &w2);
        let gz = solid_d(vec3<f32>(x, y, z + e), &w2) - solid_d(vec3<f32>(x, y, z - e), &w2);
        let g3 = vec3<f32>(gx, gy, gz);
        let n3 = select(vec3<f32>(0.0, 1.0, 0.0), normalize(g3), length(g3) > 1e-6);
        // onto the surface itself: Newton steps down the column (a stop within eps of it left the map
        // stepped by up to eps, terraces under a low sun)
        var ys = y;
        for (var k = 0; k < 2; k++) {
          let dk = solid_d(vec3<f32>(x, ys, z), &w2);
          ys -= dk / max(n3.y, 0.25);
        }
        if (ys > height) {
          height = ys;
          temp = collider_temp(which);
          nrm = n3.xz;
        }
        break;
      }
      y -= max(d, eps);
    }
  }
  textureStore(out_surf, vec2<i32>(id.xy), vec4<f32>(height, temp, nrm));
}
