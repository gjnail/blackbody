// The box moves by whole cells (it follows the liquid, or a boat): every particle keeps its place in
// the world, so its grid position moves the other way; what the box has left behind is dropped. The
// particles per cell are counted afresh, so the open water can fill the strip the box moved onto
// right away (an empty strip for even one step lets the liquid slump into it, and that loss stays).
//!include common.wgsl
//!include liq_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,  // shift of the box (cells, x y z), slot capacity
};

@group(0) @binding(0) var<storage, read_write> parts: array<Particle>;
@group(0) @binding(1) var<storage, read_write> ctr: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> freelist: array<u32>;
@group(0) @binding(3) var<storage, read_write> cellcount: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.k.w)) { return; }
  var P = parts[i];
  if (!alive(P)) { return; }
  let nf = vec3<f32>(gdim(U.g));
  P.p = vec4<f32>(P.p.xyz - U.k.xyz, P.p.w);
  if (any(P.p.xyz < vec3<f32>(0.0)) || any(P.p.xyz >= nf)) {
    P.p.w = -1.0;
    parts[i] = P;
    let k = atomicAdd(&ctr[C_FREE], 1);
    freelist[u32(k)] = i;
    return;
  }
  parts[i] = P;
  let n = gdim(U.g);
  let cc = min(vec3<i32>(floor(P.p.xyz)), n - vec3<i32>(1));
  atomicAdd(&cellcount[u32(cc.x + n.x * (cc.y + n.y * cc.z))], 1u);
}
