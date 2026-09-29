// Coarse right-hand side = transpose of trilinear prolongation applied to the fine residual
// (weights 1/4, 3/4, 3/4, 1/4 per axis over fine cells 2C-1 .. 2C+2), times 1/2 for the h^2
// scaling of the system. Also clears the coarse correction.
//!include liq_mg_common.wgsl

@group(0) @binding(0) var RES: texture_3d<f32>;
@group(0) @binding(1) var COc: texture_3d<f32>;
@group(0) @binding(2) var RHSc: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var Zc: texture_storage_3d<r32float, write>;
@group(1) @binding(0) var<uniform> U: LMG;

fn w1(a: i32) -> f32 { return select(0.75, 0.25, a == -1 || a == 2); }

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let df = vec3<i32>(U.n.xyz);
  let dc = vec3<i32>(U.nc.xyz);
  let C = vec3<i32>(id);
  if (any(C >= dc)) { return; }
  var rhs = 0.0;
  if (textureLoad(COc, C, 0).x > 0.0) {
    for (var az = -1; az <= 2; az++) {
      for (var ay = -1; ay <= 2; ay++) {
        for (var ax = -1; ax <= 2; ax++) {
          let f = 2 * C + vec3<i32>(ax, ay, az);
          if (any(f < vec3<i32>(0)) || any(f >= df)) { continue; }
          rhs += w1(ax) * w1(ay) * w1(az) * textureLoad(RES, f, 0).x;
        }
      }
    }
    rhs *= 0.5;
  }
  textureStore(RHSc, C, vec4<f32>(rhs, 0.0, 0.0, 0.0));
  textureStore(Zc, C, vec4<f32>(0.0));
}
