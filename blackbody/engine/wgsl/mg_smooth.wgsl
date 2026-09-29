// One red/black Gauss-Seidel half-sweep, in place. Threads cover only cells of the active colour.
//!include mg_common.wgsl

@group(0) @binding(0) var P: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var R: texture_3d<f32>;
@group(0) @binding(2) var S: texture_3d<f32>;
@group(1) @binding(0) var<uniform> U: MG;

fn is_solid(c: vec3<i32>) -> bool {
  return U.bc.w > 0.5 && textureLoad(S, c, 0).x < 0.0;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let d = vec3<i32>(U.n.xyz);
  let par = i32(U.n.w);
  let c = vec3<i32>(2 * i32(id.x) + ((i32(id.y) + i32(id.z) + par) & 1), i32(id.y), i32(id.z));
  if (any(c >= d)) { return; }
  if (is_solid(c)) {
    textureStore(P, c, vec4<f32>(0.0));
    return;
  }
  var sum = 0.0;
  var k = 0.0;
  for (var i = 0; i < 6; i++) {
    let q = c + mg_dir(i);
    let kind = nb_kind(q, d, U.bc);
    if (kind == 2) {
      if (!is_solid(q)) {
        sum += textureLoad(P, q).x;
        k += 1.0;
      }
    } else if (kind == 1) {
      k += 2.0;  // open face: ghost = -p, so p = 0 exactly on the boundary face
    }
  }
  if (k < 0.5) { return; }
  let pn = (sum - textureLoad(R, c, 0).x) / k;
  let po = textureLoad(P, c).x;
  textureStore(P, c, vec4<f32>(mix(po, pn, U.k.x), 0.0, 0.0, 0.0));
}
