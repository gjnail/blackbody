// Fire and liquid in one box: liquid in gas hotter than boiling slowly boils away. Each particle in
// a hot cell is freed with a probability that grows with the temperature above boiling.
//!include common.wgsl
//!include liq_common.wgsl
//!include noise.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,   // slot capacity, boiling point (fire temperature scale), rate at full heat (1/s), step seed
  nf: vec4<f32>,  // fire grid dims
};

@group(0) @binding(0) var<storage, read_write> parts: array<Particle>;
@group(0) @binding(1) var<storage, read_write> ctr: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> freelist: array<u32>;
@group(0) @binding(3) var scal: texture_3d<f32>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.k.x)) { return; }
  var P = parts[i];
  if (P.p.w < 0.5) { return; }
  let nl = vec3<f32>(gdim(U.g));
  let c = min(vec3<i32>(floor(P.p.xyz * U.nf.xyz / nl)), vec3<i32>(U.nf.xyz) - vec3<i32>(1));
  let T = textureLoad(scal, max(c, vec3<i32>(0)), 0).x;
  let boil = U.k.y;
  if (T <= boil) { return; }
  let heat = clamp((T - boil) / max(1.0 - boil, 1e-3) * 4.0, 0.0, 1.0);
  if (rand1(i * 2654435761u + u32(U.k.w) * 97u) >= U.k.z * heat * U.g.bc.w) { return; }
  P.p.w = 0.0;
  parts[i] = P;
  let k = atomicAdd(&ctr[C_FREE], 1);
  freelist[u32(k)] = i;
}
