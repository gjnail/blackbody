// Sand and water (matter.py, mpm_wet.wgsl): how far each of the liquid's cells is from the water through the solid
// (the sand, which the water seeps into): 0 in the liquid, a cell more for each solid cell between, out to a few cells
// (as many as there are passes of step); the air does not carry it.

struct Params {
  n: vec4<f32>,     // the liquid's cells (x, y, z), _
};

@group(0) @binding(0) var T: texture_3d<f32>;     // the liquid's cells: 0 air, 1 liquid, 2 solid
@group(0) @binding(1) var D: texture_3d<f32>;     // (step) the distance so far
@group(0) @binding(2) var out: texture_storage_3d<r32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

const FAR: f32 = 1.0e4;

@compute @workgroup_size(8, 8, 4)
fn init(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (any(c >= vec3<i32>(U.n.xyz))) { return; }
  let t = textureLoad(T, c, 0).x;
  textureStore(out, c, vec4<f32>(select(FAR, 0.0, t > 0.5 && t < 1.5), 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8, 4)
fn step(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(U.n.xyz);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  var d = textureLoad(D, c, 0).x;
  if (textureLoad(T, c, 0).x > 1.5) {
    for (var i = 0; i < 6; i++) {
      var e = vec3<i32>(0);
      e[i >> 1] = select(-1, 1, (i & 1) == 1);
      let q = c + e;
      if (any(q < vec3<i32>(0)) || any(q >= n)) { continue; }
      d = min(d, textureLoad(D, q, 0).x + 1.0);
    }
  }
  textureStore(out, c, vec4<f32>(d, 0.0, 0.0, 0.0));
}
