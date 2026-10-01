// Fire -> fabric, radiation (2 of 3): the radiation falling on each vertex of the cloth (W/m^2), from the
// coarse cells of the hot gas (cloth_rad_src.wgsl), each a point source of its power, on either face of
// the cloth (whichever it reaches). Distances are softened by the coarse cell's size (the power is
// spread through it). Nothing is shadowed: cloth behind cloth is lit as if the front one were not there.
//!include cloth_common.wgsl

const PI4: f32 = 12.566371;

struct Params {
  a: vec4<f32>,   // vertex count, coarse cells, softening (m^2), _
};

@group(0) @binding(0) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> N: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> S: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> SRC: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> QR: array<vec4<f32>>;   // x: irradiance (W/m^2)
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.a.x)) { return; }
  if (gone(S[i])) {
    QR[i] = vec4<f32>(0.0);
    return;
  }
  let x = X[i].xyz;
  let nn = N[i].xyz;
  let ln = length(nn);
  let soft = U.a.z;
  var q = 0.0;
  for (var c = 0u; c < u32(U.a.y); c++) {
    let s = SRC[c];
    if (s.w <= 0.0) { continue; }
    let d = s.xyz - x;
    let r2 = dot(d, d) + soft;
    // on whichever face it reaches (a cloth with no normal yet: as if it faced the source)
    let cosn = select(1.0, abs(dot(nn, d)) / (ln * sqrt(r2)), ln > 1e-6);
    q += s.w * cosn / (PI4 * r2);
  }
  QR[i] = vec4<f32>(q, 0.0, 0.0, 0.0);
}
