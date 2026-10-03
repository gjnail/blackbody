// Heat -> fabric, radiation (2 of 3): the radiation falling on each vertex of the cloth (W/m^2), from the frame's
// shared radiant sources (radiant.py: the fire's hot gas, glowing matter, lava, hot objects), on either face of the
// cloth (whichever it reaches). Nothing is shadowed: cloth behind cloth is lit as if the front one were not there.
//!include cloth_common.wgsl

struct Params {
  a: vec4<f32>,   // vertex count, _, _, _
};

@group(0) @binding(0) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> N: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> S: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> RL: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> RLC: array<u32>;
@group(0) @binding(5) var<storage, read_write> QR: array<vec4<f32>>;   // x: irradiance (W/m^2)
@group(1) @binding(0) var<uniform> U: Params;

//!include rad_common.wgsl

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.a.x)) { return; }
  if (gone(S[i])) {
    QR[i] = vec4<f32>(0.0);
    return;
  }
  // (a cloth with no normal yet: as if it faced every source; -3: the cloth owns none of them)
  QR[i] = vec4<f32>(rad_irradiance(X[i].xyz, N[i].xyz, -3.0, true), 0.0, 0.0, 0.0);
}
