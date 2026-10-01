// Fabric self-collision: every vertex into a hashed grid of cells (at least twice the largest collision
// radius), as linked lists in one buffer: the bucket heads first, then one link per vertex. 'clear'
// empties the buckets, 'insert' fills them.
//!include cloth_common.wgsl

struct Params {
  h: vec4<f32>,   // vertex count, cell size (m), bucket count, _
};

@group(0) @binding(0) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> S: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> HN: array<atomic<u32>>;   // heads (bucket count), then next (vertex count)
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn clear(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= u32(U.h.z)) { return; }
  atomicStore(&HN[id.x], 0xffffffffu);
}

@compute @workgroup_size(64, 1, 1)
fn insert(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.h.x)) { return; }
  let nb = u32(U.h.z);
  if (gone(S[i])) {
    atomicStore(&HN[nb + i], 0xffffffffu);
    return;
  }
  let c = vec3<i32>(floor(X[i].xyz / U.h.y));
  atomicStore(&HN[nb + i], atomicExchange(&HN[cell_hash(c, nb)], i));
}
