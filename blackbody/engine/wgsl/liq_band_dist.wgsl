// Narrow band, pass 2: distance in cells (city block, up to the band width) from each cell to the
// nearest air, by repeated relaxation. Mode 0 starts it: 0 in air, far elsewhere (the deep liquid,
// solids, and the sides past the box, which are not a surface); mode 1 takes one step.
//!include common.wgsl
//!include liq_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,   // surface density (particles), mode
};

@group(0) @binding(0) var dens: texture_3d<f32>;
@group(0) @binding(1) var<storage, read> band: array<u32>;
@group(0) @binding(2) var sdf: texture_3d<f32>;
@group(0) @binding(3) var src: texture_3d<f32>;
@group(0) @binding(4) var dst: texture_storage_3d<r32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  if (U.k.y < 0.5) {
    let air = textureLoad(dens, c, 0).x < U.k.x && band[nidx(c, n)] == 0u && textureLoad(sdf, c, 0).x >= 0.0;
    textureStore(dst, c, vec4<f32>(select(1.0e4, 0.0, air), 0.0, 0.0, 0.0));
    return;
  }
  var d = textureLoad(src, c, 0).x;
  for (var i = 0; i < 6; i++) {
    var o = vec3<i32>(0);
    o[i >> 1] = select(-1, 1, (i & 1) == 1);
    let q = c + o;
    if (in_grid(q, n)) {
      d = min(d, textureLoad(src, q, 0).x + 1.0);
    } else if (q.y >= n.y && U.g.bc.y > 0.5) {
      d = min(d, 1.0);   // open air above the box
    }
  }
  textureStore(dst, c, vec4<f32>(d, 0.0, 0.0, 0.0));
}
