// What the simulation box hands the open-water layer, per column of cells: how far its free surface
// stands from the sea's own there (a wake, the rings of a splash), how its surface water moves, and
// how much foam floats on it. A column with no free surface (under a hull, a dry column) is marked
// with a height of -1e4.
//
// Per column (x, z): (surface - sea (m), surface velocity x, z (m/s), foam cover 0..1); the foam
// particles per column come from ocl_count.wgsl.
//!include common.wgsl
//!include liq_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,   // rest density (particles), foam particles that cover a column, whitewater capacity, _
};

@group(0) @binding(0) var dens: texture_3d<f32>;
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var ocn_t: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> fcount: array<u32>;
@group(0) @binding(4) var dst: texture_storage_2d<rgba32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec2<i32>(id.xy);
  if (c.x >= n.x || c.y >= n.z) { return; }
  let rho0 = U.k.x;
  // the free surface: the topmost cell at least half full with a nearly empty cell over it
  var top = -1.0;
  var y = n.y - 1;
  loop {
    if (y < 0) { break; }
    let d = textureLoad(dens, vec3<i32>(c.x, y, c.y), 0).x / rho0;
    if (d >= 0.5) {
      let above = select(0.0, textureLoad(dens, vec3<i32>(c.x, min(y + 1, n.y - 1), c.y), 0).x / rho0, y + 1 < n.y);
      top = f32(y) + clamp(d, 0.0, 1.0) + clamp(above, 0.0, 0.5);
      break;
    }
    y -= 1;
  }
  let h = U.g.n.w;
  let sea = textureLoad(ocn_t, c, 0).x;
  var out = vec4<f32>(-1.0e4, 0.0, 0.0, 0.0);
  if (top >= 0.0) {
    let v = mac_vel(vel, vec3<f32>(f32(c.x) + 0.5, max(top - 0.7, 0.5), f32(c.y) + 0.5), n);
    out = vec4<f32>((top - sea) * h, v.x, v.z, 0.0);
  }
  let fc = f32(fcount[u32(c.x + n.x * c.y)]);
  out.w = 1.0 - exp(-fc / max(U.k.y, 1e-3));
  textureStore(dst, c, out);
}
