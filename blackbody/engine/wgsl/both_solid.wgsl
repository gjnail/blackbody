// Fire, water and lava in one box: the lava is solid ground to the water. Lava is a hundred
// thousand times thicker than water and more than twice as dense, so the water flows over and
// around it as it would round a rock, and is shoved aside where the lava advances; the lava
// barely feels the water at all (only its heat, both_lava.wgsl and both_quench.wgsl). Each substep
// the lava's level set, from its particle density, is laid into the water's solid distance field
// (in cells, negative inside), over the colliders the water solver has just written there.
//!include common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,  // rest density (particles per cell)
};

@group(0) @binding(0) var lava_dens: texture_3d<f32>;
@group(0) @binding(1) var sdf: texture_storage_3d<r32float, read_write>;
@group(1) @binding(0) var<uniform> U: Params;

fn share(c: vec3<i32>, n: vec3<i32>) -> f32 {
  if (!in_grid(c, n)) { return 0.0; }
  return clamp(textureLoad(lava_dens, c, 0).x / max(U.k.x, 1e-3), 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  // a smoothed share (the cell and its six neighbours), so the surface is not as grainy as the
  // particles it comes from
  var sum = 2.0 * share(c, n);
  var near = share(c, n);
  for (var i = 0; i < 6; i++) {
    var d = vec3<i32>(0);
    d[i >> 1] = select(-1, 1, (i & 1) == 1);
    let f = share(c + d, n);
    sum += f;
    near = max(near, f);
  }
  if (near <= 0.0) { return; }   // no lava here or next door: the colliders' distance stands
  let f = sum / 8.0;
  let phi = (0.45 - f) * 2.5;    // cells: about -1 deep in the lava, +1 a cell out of it
  let cur = textureLoad(sdf, c).x;
  if (phi < cur) { textureStore(sdf, c, vec4<f32>(phi, 0.0, 0.0, 0.0)); }
}
