// Two liquids and dye, pass 1: every particle adds what it carries (its density relative to the
// liquid and its dye) to the eight nearest cell centres, with the same weights as its density
// (liq_p2g.wgsl), so a cell's mean is its sum over that density weight.
//!include common.wgsl
//!include liq_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,  // slot capacity
};

@group(0) @binding(0) var<storage, read> parts: array<Particle>;
@group(0) @binding(1) var<storage, read> attr: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> acc: array<atomic<i32>>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.k.x)) { return; }
  let P = parts[i];
  if (!alive(P)) { return; }
  let a = attr[i];
  let dye = attr_dye(a);
  let dr = attr_rho(a) - 1.0;
  let n = gdim(U.g);
  let q = P.p.xyz - vec3<f32>(0.5);
  let fq = floor(q);
  let f = q - fq;
  let b = vec3<i32>(fq);
  for (var o = 0; o < 8; o++) {
    let oo = vec3<i32>(o & 1, (o >> 1) & 1, o >> 2);
    let node = b + oo;
    if (any(node < vec3<i32>(0)) || any(node >= n)) { continue; }
    let wv = mix(vec3<f32>(1.0) - f, f, vec3<f32>(oo));
    let w = wv.x * wv.y * wv.z;
    let s = nidx(node, n) * ATTR_SLOTS;
    if (dr != 0.0) { atomicAdd(&acc[s], i32(round(w * dr * FX_A))); }
    if (any(dye > vec4<f32>(0.0))) {
      atomicAdd(&acc[s + 1u], i32(round(w * dye.x * FX_A)));
      atomicAdd(&acc[s + 2u], i32(round(w * dye.y * FX_A)));
      atomicAdd(&acc[s + 3u], i32(round(w * dye.z * FX_A)));
      atomicAdd(&acc[s + 4u], i32(round(w * dye.w * FX_A)));
    }
  }
}
