// Particle to grid: each particle adds its (APIC affine) momentum and weight to the eight nearest
// faces of each velocity component, and its weight to the eight nearest cell centres (density).
//!include common.wgsl
//!include liq_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,  // slot capacity, APIC on (1/0), speed limit (m/s), _
};

@group(0) @binding(0) var<storage, read> parts: array<Particle>;
@group(0) @binding(1) var<storage, read_write> acc: array<atomic<i32>>;
@group(1) @binding(0) var<uniform> U: Params;

fn splat(x: vec3<f32>, k: u32, vel: f32, c: vec3<f32>, n: vec3<i32>) {
  let m = n + vec3<i32>(1);
  let q = x - face_off(k);
  let fq = floor(q);
  let f = q - fq;
  let b = vec3<i32>(fq);
  let lim = face_lim(k, n);
  for (var o = 0; o < 8; o++) {
    let oo = vec3<i32>(o & 1, (o >> 1) & 1, o >> 2);
    let node = b + oo;
    if (any(node < vec3<i32>(0)) || any(node > lim)) { continue; }
    let fo = vec3<f32>(oo);
    let wv = mix(vec3<f32>(1.0) - f, f, fo);
    let w = wv.x * wv.y * wv.z;
    if (w <= 1e-7) { continue; }
    let mom = w * (vel + dot(c, fo - f));
    let i = nidx(node, m) * ACC + 2u * k;
    atomicAdd(&acc[i], i32(round(mom * FX_M)));
    atomicAdd(&acc[i + 1u], i32(round(w * FX_W)));
  }
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.k.x)) { return; }
  let P = parts[i];
  if (P.p.w < 0.5) { return; }
  let n = gdim(U.g);
  let x = P.p.xyz;
  var v = P.v.xyz;
  let sp = length(v);
  if (sp > U.k.z) { v *= U.k.z / sp; }
  let ap = U.k.y;
  splat(x, 0u, v.x, P.cx.xyz * ap, n);
  splat(x, 1u, v.y, P.cy.xyz * ap, n);
  splat(x, 2u, v.z, P.cz.xyz * ap, n);

  // density at cell centres (particles per cell)
  let m = n + vec3<i32>(1);
  let q = x - vec3<f32>(0.5);
  let fq = floor(q);
  let f = q - fq;
  let b = vec3<i32>(fq);
  for (var o = 0; o < 8; o++) {
    let oo = vec3<i32>(o & 1, (o >> 1) & 1, o >> 2);
    let node = b + oo;
    if (any(node < vec3<i32>(0)) || any(node >= n)) { continue; }
    let wv = mix(vec3<f32>(1.0) - f, f, vec3<f32>(oo));
    let w = wv.x * wv.y * wv.z;
    atomicAdd(&acc[nidx(node, m) * ACC + 6u], i32(round(w * FX_W)));
  }
}
