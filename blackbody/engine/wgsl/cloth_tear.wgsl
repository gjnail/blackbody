// Fabric tearing (cloth.py, a fabric with Tears on): a thread pulled past its breaking stretch breaks, and of its two
// ends the one pulled harder all round tears away, so a rip runs on from its tip one vertex at a time, ragged, the way
// cloth caught on a moving object or overloaded tears. (The breaking point is a stretch, the fabric's Tears at, not a
// force: the cloth's own stretch is kept soft and its pins' tethers keep it from stretching far.) A torn vertex is
// gone as a burnt-away one is, its triangles opening at once (cloth_draw.wgsl) and leaving no ash: S = (its
// temperature, 1, -1 (torn), an age past a flake's life).
//  - load: each vertex's pull: how far past their rest lengths its threads are stretched, summed; and how many of
//    its constraints (threads or diagonals) it had and still has
//  - mark: each thread past its fabric's breaking stretch marks the end that is pulled harder
//  - tear: each marked vertex tears away, and so does one the tears round it have left hanging by two constraints or
//    fewer (a scrap of a few vertices, cut off, would fly off on its own)
// Only the threads (warp and weft) count: the shear diagonals are soft and stretch by design.
//!include cloth_common.wgsl

struct Params {
  k: vec4<f32>,   // stretch constraints (count), vertices (count), _, _
};

@group(0) @binding(0) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> C: array<vec4<u32>>;   // i, j, rest length (f32 bits), compliance
@group(0) @binding(2) var<storage, read_write> S: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> V: array<vec4<f32>>;    // w: its fabric
@group(0) @binding(4) var<storage, read> M: array<Mat>;
@group(0) @binding(5) var<storage, read> UV: array<vec4<f32>>;   // weave coordinates (m)
@group(0) @binding(6) var<storage, read_write> TR: array<atomic<u32>>;   // per vertex: its pull (fixed point), marked,
                                                                          // constraints left, constraints it had
@group(1) @binding(0) var<uniform> U: Params;

const FX: f32 = 65536.0;

// Constraint ci: how far past its rest length it is stretched (a share of it), its fabric's breaking stretch (0: it
// does not tear), 1 while both its ends are there, and 1 if it is a thread (along the warp or the weft; a shear
// diagonal: 0).
fn thread(ci: u32) -> vec4<f32> {
  let c = C[ci];
  let tear = M[u32(V[c.x].w + 0.5)].e.w;
  if (gone(S[c.x]) || gone(S[c.y])) { return vec4<f32>(-1.0, tear, 0.0, 0.0); }
  let d = abs(UV[c.x].xy - UV[c.y].xy);
  let L0 = bitcast<f32>(c.z);
  if (L0 <= 1.0e-9) { return vec4<f32>(-1.0, tear, 1.0, 0.0); }
  let along = select(0.0, 1.0, min(d.x, d.y) <= 0.25 * max(d.x, d.y));
  return vec4<f32>(length(X[c.x].xyz - X[c.y].xyz) / L0 - 1.0, tear, 1.0, along);
}

@compute @workgroup_size(64, 1, 1)
fn load(@builtin(global_invocation_id) id: vec3<u32>) {
  let ci = id.x;
  if (ci >= u32(U.k.x)) { return; }
  let t = thread(ci);
  if (t.y <= 0.0) { return; }
  let c = C[ci];
  atomicAdd(&TR[4u * c.x + 3u], 1u);
  atomicAdd(&TR[4u * c.y + 3u], 1u);
  if (t.z < 0.5) { return; }
  atomicAdd(&TR[4u * c.x + 2u], 1u);
  atomicAdd(&TR[4u * c.y + 2u], 1u);
  if (t.w < 0.5 || t.x <= 0.0) { return; }
  let q = u32(min(t.x, 1000.0) * FX);
  atomicAdd(&TR[4u * c.x], q);
  atomicAdd(&TR[4u * c.y], q);
}

@compute @workgroup_size(64, 1, 1)
fn mark(@builtin(global_invocation_id) id: vec3<u32>) {
  let ci = id.x;
  if (ci >= u32(U.k.x)) { return; }
  let t = thread(ci);
  if (t.w < 0.5 || t.y <= 0.0 || t.x <= t.y) { return; }
  let c = C[ci];
  // (per constraint it still has: the vertex at a rip's tip has lost its threads to the torn one, and the pull of
  // the rest makes it the one to go next, so the rip runs on as a slit, not a line of pinholes a vertex apart)
  let a = f32(atomicLoad(&TR[4u * c.x])) / f32(max(atomicLoad(&TR[4u * c.x + 2u]), 1u));
  let b = f32(atomicLoad(&TR[4u * c.y])) / f32(max(atomicLoad(&TR[4u * c.y + 2u]), 1u));
  atomicMax(&TR[4u * select(c.y, c.x, a >= b) + 1u], 1u);
}

@compute @workgroup_size(64, 1, 1)
fn tear(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.k.y)) { return; }
  let st = S[i];
  if (gone(st) || M[u32(V[i].w + 0.5)].e.w <= 0.0) { return; }
  let left = atomicLoad(&TR[4u * i + 2u]);
  let cut = left < atomicLoad(&TR[4u * i + 3u]) && left <= 2u;
  if (atomicLoad(&TR[4u * i + 1u]) == 0u && !cut && left > 0u) { return; }
  S[i] = vec4<f32>(st.x, 1.0, -1.0, 1.0e9);
}
