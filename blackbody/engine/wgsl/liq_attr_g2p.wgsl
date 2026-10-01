// Two liquids and dye, pass 4: dye mixes. Each particle's dye relaxes toward the mean of the liquid
// around it, so a dye stirred into the liquid spreads and clouds into it instead of staying in
// particle-sized flecks. The density does not mix (oil and water stay apart).
//!include common.wgsl
//!include liq_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,  // slot capacity, share mixed this step (0..1)
};

@group(0) @binding(0) var<storage, read> parts: array<Particle>;
@group(0) @binding(1) var<storage, read_write> attr: array<vec4<u32>>;
@group(0) @binding(2) var attr_t: texture_3d<f32>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.k.x)) { return; }
  let P = parts[i];
  if (!alive(P)) { return; }
  let n = gdim(U.g);
  let q = P.p.xyz - vec3<f32>(0.5);
  let fq = floor(q);
  let f = q - fq;
  let b = vec3<i32>(fq);
  var mean = vec4<f32>(0.0);
  for (var o = 0; o < 8; o++) {
    let oo = vec3<i32>(o & 1, (o >> 1) & 1, o >> 2);
    let node = clamp(b + oo, vec3<i32>(0), n - vec3<i32>(1));
    let wv = mix(vec3<f32>(1.0) - f, f, vec3<f32>(oo));
    mean += wv.x * wv.y * wv.z * textureLoad(attr_t, node, 0);
  }
  let a = attr[i];
  let dye = attr_dye(a);
  attr[i] = make_attr(dye + (mean - dye) * U.k.y, attr_rho(a));
}
