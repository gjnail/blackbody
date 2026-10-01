// Fabric: stretch and shear (XPBD distance constraints), one colour at a time so no two constraints
// in a dispatch share a vertex (Gauss-Seidel on the GPU). Each constraint keeps its Lagrange
// multiplier over the passes of a substep (cleared at its start), so a few passes converge to the
// cloth's true stiffness instead of a stiffer or softer one. Woven cloth hardly stretches along its
// threads but gives way easily when pushed together, where it buckles into wrinkles instead: a
// compressed constraint is far softer than a stretched one. Where the cloth softens (a melting
// synthetic) or chars, its threads shrink and the cloth draws up. Wet cloth carries its water: per
// vertex, its mass is (1 + the water it holds per kg of fibre) times its own (cloth_predict.wgsl's carry),
// so soaked cloth hangs heavier on its threads and moves less for the same pull.
//!include cloth_common.wgsl

struct Params {
  rng: vec4<f32>,   // first constraint of this colour, how many, dt, compression softness (x compliance)
  k: vec4<f32>,     // over-relaxation, _, _, _
};

@group(0) @binding(0) var<storage, read_write> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> C: array<vec4<u32>>;   // i, j, rest length (f32 bits), compliance (m/N, f32 bits)
@group(0) @binding(2) var<storage, read> S: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> M: array<Mat>;
@group(0) @binding(5) var<storage, read_write> LS: array<f32>;   // multiplier per constraint
@group(0) @binding(6) var<storage, read> P: array<vec4<f32>>;    // w: wetness
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= u32(U.rng.y)) { return; }
  let ci = u32(U.rng.x) + id.x;
  let c = C[ci];
  let i = c.x;
  let j = c.y;
  let si = S[i];
  let sj = S[j];
  if (gone(si) || gone(sj)) { return; }
  let mat = M[u32(V[i].w + 0.5)];
  let xi = X[i];
  let xj = X[j];
  let wi = xi.w / (1.0 + clamp(P[i].w, 0.0, 1.0) * mat.f.y);
  let wj = xj.w / (1.0 + clamp(P[j].w, 0.0, 1.0) * mat.f.y);
  let W = wi + wj;
  if (W <= 0.0) { return; }
  // threads shrink as the fabric softens (synthetics) or chars
  var shrink = mat.e.y * smoothstep(0.3, 0.95, 0.5 * (si.y + sj.y));
  if (mat.d.y > 0.0) { shrink += mat.d.z * 0.5 * (si.w + sj.w); }
  let L0 = bitcast<f32>(c.z) * (1.0 - clamp(shrink, 0.0, 0.8));
  let d = xi.xyz - xj.xyz;
  let len = length(d);
  if (len < 1e-9) { return; }
  let Cv = len - L0;
  var alpha = bitcast<f32>(c.w) / (U.rng.z * U.rng.z);
  if (Cv < 0.0) { alpha *= U.rng.w; }
  let dl = (-Cv - alpha * LS[ci]) / (W + alpha) * U.k.x;
  LS[ci] += dl;
  let n = d / len;
  X[i] = vec4<f32>(xi.xyz + n * (wi * dl), xi.w);
  X[j] = vec4<f32>(xj.xyz - n * (wj * dl), xj.w);
}
