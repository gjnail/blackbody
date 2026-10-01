// Fabric: bending, as an XPBD constraint on the isometric bending energy of Bergou et al. 2006: for
// each inner edge, its four vertices weighted by cotangents (K) sum to the discrete mean curvature
// normal, which is zero wherever the cloth is flat. The compliance comes from the fabric's measured
// bending rigidity (N m), so it drapes and folds as the real cloth does. One colour at a time.
// A fabric modelled curved (a mesh) keeps the size of its curvature instead. Multipliers are kept
// over the passes of a substep, as for stretch. Wet cloth is heavier (its water): its vertices move
// less for the same bending (cloth_mg_wet.wgsl does the same for the multigrid's).
//!include cloth_common.wgsl

struct Hinge {
  v: vec4<u32>,    // the edge's two vertices, then the two opposite them
  k: vec4<f32>,    // their cotangent weights
  c: vec4<f32>,    // compliance (m/N), rest curvature length (m), _, _
};

struct Params {
  rng: vec4<f32>,   // first hinge of this colour, how many, dt, over-relaxation
  k: vec4<f32>,     // fabrics whose whole panels' bending multigrid solves (bitmask, cloth_mg.wgsl), _, _, _
};

@group(0) @binding(0) var<storage, read_write> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> H: array<Hinge>;
@group(0) @binding(2) var<storage, read> S: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> LB: array<vec4<f32>>;   // multiplier per hinge (vector)
@group(0) @binding(4) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> P: array<vec4<f32>>;   // w: wetness
@group(0) @binding(6) var<storage, read> M: array<Mat>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= u32(U.rng.y)) { return; }
  let hi = u32(U.rng.x) + id.x;
  let hg = H[hi];
  let iv = hg.v;
  if (gone(S[iv.x]) || gone(S[iv.y]) || gone(S[iv.z]) || gone(S[iv.w])) { return; }
  // an unburnt panel hinge is solved by multigrid (cloth_mg.wgsl: the same rule there)
  let fb = u32(V[iv.x].w + 0.5);
  let m0 = S[iv.x].y + S[iv.y].y + S[iv.z].y + S[iv.w].y;
  if (hg.c.z > 0.5 && (u32(U.k.x + 0.5) & (1u << fb)) != 0u && 0.25 * m0 < 0.25) { return; }
  let x0 = X[iv.x];
  let x1 = X[iv.y];
  let x2 = X[iv.z];
  let x3 = X[iv.w];
  // inverse masses, with the water the cloth carries
  let ab = M[fb].f.y;
  let im = vec4<f32>(x0.w / (1.0 + clamp(P[iv.x].w, 0.0, 1.0) * ab), x1.w / (1.0 + clamp(P[iv.y].w, 0.0, 1.0) * ab),
                     x2.w / (1.0 + clamp(P[iv.z].w, 0.0, 1.0) * ab), x3.w / (1.0 + clamp(P[iv.w].w, 0.0, 1.0) * ab));
  let k = hg.k;
  let v = k.x * x0.xyz + k.y * x1.xyz + k.z * x2.xyz + k.w * x3.xyz;
  let wsum = im.x * k.x * k.x + im.y * k.y * k.y + im.z * k.z * k.z + im.w * k.w * k.w;
  let at = hg.c.x / (U.rng.z * U.rng.z);
  if (wsum + at <= 0.0) { return; }
  let lam = LB[hi];
  var dl = vec3<f32>(0.0);
  if (hg.c.y > 1e-7) {
    // curved at rest: keep the curvature's size (a scalar constraint along v)
    let l = length(v);
    if (l < 1e-9) { return; }
    let d = (-(l - hg.c.y) - at * lam.x) / (wsum + at) * U.rng.w;
    LB[hi] = vec4<f32>(lam.x + d, 0.0, 0.0, 0.0);
    dl = v / l * d;
  } else {
    // flat at rest: the curvature vector itself goes to zero, except where the cloth has charred: char
    // shrinks more on the side the flame heats than on the other, so it curls and crinkles (toward
    // the side the cloth's normal faces, as the faces it was built with)
    let chr = smoothstep(0.25, 0.85, 0.25 * (S[iv.x].y + S[iv.y].y + S[iv.z].y + S[iv.w].y));
    var goal = vec3<f32>(0.0);
    if (chr > 0.0) {
      let e = x1.xyz - x0.xyz;
      let nh = cross(e, x2.xyz - x0.xyz) + cross(x3.xyz - x0.xyz, e);
      let ln = length(nh);
      if (ln > 1e-12) { goal = nh / ln * (0.6 * chr * length(e)); }
    }
    dl = (-(v - goal) - at * lam.xyz) / (wsum + at) * U.rng.w;
    LB[hi] = vec4<f32>(lam.xyz + dl, 0.0);
  }
  X[iv.x] = vec4<f32>(x0.xyz + im.x * k.x * dl, x0.w);
  X[iv.y] = vec4<f32>(x1.xyz + im.y * k.y * dl, x1.w);
  X[iv.z] = vec4<f32>(x2.xyz + im.z * k.z * dl, x2.w);
  X[iv.w] = vec4<f32>(x3.xyz + im.w * k.w * dl, x3.w);
}
