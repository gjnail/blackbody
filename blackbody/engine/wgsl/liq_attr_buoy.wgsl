// Two liquids, pass 3: buoyancy between liquids of different densities. The pressure solve treats
// the liquid as one density, which is right while a parcel is surrounded by liquid like itself;
// where it is not, the surrounding liquid's pressure pushes it up (lighter: oil in water) or lets it
// sink (heavier: brine, molten metal) by g (rho_around - rho) / rho. rho_around is the mean density
// of the liquid within a few cells, weighted by how much liquid each cell holds, so a pool of one
// liquid on its own feels nothing and only the liquids' meeting places do. Stored per cell as the
// upward share of gravity.
//!include common.wgsl
//!include liq_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,  // neighbourhood spacing (cells)
};

@group(0) @binding(0) var rho_t: texture_3d<f32>;
@group(0) @binding(1) var buoy: texture_storage_3d<r32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let r = textureLoad(rho_t, c, 0);
  if (r.y < 0.05) {
    textureStore(buoy, c, vec4<f32>(0.0));
    return;
  }
  let sp = i32(U.k.x);
  var s = vec2<f32>(0.0);
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let o = vec3<i32>(x, y, z);
        let q = c + o * sp;
        if (any(q < vec3<i32>(0)) || any(q >= n)) { continue; }
        let wgt = 1.0 / (1.0 + f32(dot(o, o)));
        s += textureLoad(rho_t, q, 0).xy * wgt;
      }
    }
  }
  let around = s.x / max(s.y, 1e-4);
  let b = clamp((around - r.z) / r.z, -1.0, 10.0);
  textureStore(buoy, c, vec4<f32>(b, 0.0, 0.0, 0.0));
}
