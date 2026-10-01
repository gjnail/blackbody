// Fire and liquid in one box: steam made under the water comes out on top of it. A fuel bed boiling
// under a puddle, or the liquid's own boiling (liquid_thermal.py), makes its steam in cells full of
// water, where the gas is carried along with the water (both_drag.wgsl) and would stay pinned to it,
// spreading over the ground in a white sheet. Real steam bubbles up through the water and leaves from
// its surface, so here each open cell (under a sixth water) gathers the steam made in the watery cells
// under it, up to the first open cell below; cells with any real share of water (a film on the ground
// counts) keep none. The steam off the lava (both_lava.wgsl, field z) is already placed where it comes
// out and is added as is. Boiling is not steady: steam comes off in bursts, so the release is broken
// up by a drifting noise (its mean stays the same) instead of rising in even columns cell by cell.
//!include common.wgsl
//!include noise.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,  // lava on (1/0)
};

@group(0) @binding(0) var raw: texture_3d<f32>;     // both_wet.wgsl's: x = water share, y = soaked, z = steam made here, w = char smoke
@group(0) @binding(1) var lava_f: texture_3d<f32>;  // the lava field, or a 1-cell zero
@group(0) @binding(2) var dst: texture_storage_3d<rgba16float, write>;   // the fire's water field
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let w = textureLoad(raw, c, 0);
  var steam = 0.0;
  if (w.x < 0.15) {
    steam = w.z;
    for (var j = 1; j <= 8; j++) {
      let q = c - vec3<i32>(0, j, 0);
      if (q.y < 0) { break; }
      let wq = textureLoad(raw, q, 0);
      if (wq.x < 0.15) { break; }   // an open cell under this one takes its own and what is under it
      steam += wq.z;
    }
  }
  if (U.k.x > 0.5) { steam += max(textureLoad(lava_f, c, 0).z, 0.0); }
  if (steam > 0.0) {
    let wp = world_of(U.g, vec3<f32>(c) + vec3<f32>(0.5));
    let t = U.g.org.w;
    let nz = fbm3(wp * 7.0 + vec3<f32>(0.37 * t, -2.2 * t, 0.29 * t), 3);
    steam *= clamp(1.0 + 1.8 * nz, 0.05, 2.6);
  }
  textureStore(dst, c, vec4<f32>(w.x, w.y, steam, w.w));
}
