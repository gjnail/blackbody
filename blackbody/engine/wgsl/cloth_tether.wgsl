// Fabric: long-range attachments (tethers, Kim et al. 2012). A point of hanging cloth can be no farther
// from the pin nearest it than the cloth between them is long, so a curtain or a flag never stretches
// under its own weight, however fine it is (the constraint passes alone converge slowly over many
// cells). Only the free vertex moves. They let go when the fabric is released or burns through
// anywhere (a burnt-off piece must fall): the first vertex to burn away flags its fabric.
//!include cloth_common.wgsl

struct Params {
  sim: vec4<f32>,      // vertex count, slack (share of the rest distance), _, _
  fab: array<Fab, MAX_FABRICS>,
};

@group(0) @binding(0) var<storage, read_write> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> TE: array<vec4<f32>>;     // pin vertex (u32 bits, 0xffffffff: none), rest distance (m), _, _
@group(0) @binding(2) var<storage, read> S: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> HOLED: array<atomic<u32>>;   // per fabric: 1 once anything burnt through
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.sim.x)) { return; }
  let fi = u32(V[i].w + 0.5);
  if (gone(S[i])) {
    atomicMax(&HOLED[fi], 1u);
    return;
  }
  let t = TE[i];
  let pin = bitcast<u32>(t.x);
  if (pin == 0xffffffffu || U.fab[fi].scl.w > 0.5 || atomicLoad(&HOLED[fi]) != 0u) { return; }
  let x = X[i];
  if (x.w <= 0.0) { return; }
  let p = X[pin].xyz;
  let d = x.xyz - p;
  let len = length(d);
  let sc = U.fab[fi].scl;
  let lim = t.y * max(sc.x, max(sc.y, sc.z)) * (1.0 + U.sim.y);
  if (len > lim && len > 1e-9) {
    X[i] = vec4<f32>(p + d * (lim / len), x.w);
  }
}
