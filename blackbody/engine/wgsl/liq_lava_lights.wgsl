// Light a molten liquid casts on its surroundings, pass 2: the columns (liq_lava_cols.wgsl) gathered
// into tiles, each a point light at its glow-weighted centre, as wide as the tile, with the tile's
// power (key-light units: a surface facing it at distance r gets power / r^2).
//
// lights[0] = (count, _, _, _); then per light (position fire-local m, radius m), (power rgb, _).

struct Params {
  nf: vec4<f32>,     // surface grid dims
  tile: vec4<f32>,   // columns per tile side, tiles along x, tiles along z, metres per surface cell
};

@group(0) @binding(0) var<storage, read> cols: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> lights: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let nt = vec2<i32>(i32(U.tile.y), i32(U.tile.z));
  let t = vec2<i32>(id.xy);
  if (t.x >= nt.x || t.y >= nt.y) { return; }
  if (t.x == 0 && t.y == 0) { lights[0] = vec4<f32>(f32(nt.x * nt.y), 0.0, 0.0, 0.0); }
  let k = i32(U.tile.x);
  let nx = i32(U.nf.x);
  let nz = i32(U.nf.z);
  var power = vec3<f32>(0.0);
  var c = vec4<f32>(0.0);
  for (var z = t.y * k; z < min((t.y + 1) * k, nz); z++) {
    for (var x = t.x * k; x < min((t.x + 1) * k, nx); x++) {
      let i = u32(x + nx * z) * 2u;
      power += cols[i].rgb;
      c += cols[i + 1u];
    }
  }
  let j = 1u + u32(t.x + nt.x * t.y) * 2u;
  var pos = vec3<f32>(0.0, -1.0e4, 0.0);   // far away: nothing
  if (c.w > 1e-12) { pos = c.xyz / c.w; }
  lights[j] = vec4<f32>(pos, 0.6 * f32(k) * U.tile.w);
  lights[j + 1u] = vec4<f32>(power, 0.0);
}
