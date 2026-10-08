// Fabric bending by multigrid (engine/cloth_mg.py): each substep solves
//     (M / h^2 + Q) x = M x_predicted / h^2
// for the panels, Q the isometric bending of their unburnt hinges. Chebyshev-smoothed V-cycle: the fine
// level is the cloth itself (hinge by hinge), coarser levels are Galerkin operators in CSR form, their
// mass and bending parts kept apart so the substep length can change.
//
// A hinge is solved here when it is a panel's (Hinge.c.z = 1), its fabric uses multigrid (uniform
// bitmask), none of its vertices has burnt away and it has not charred (mean burn progress < 0.25):
// cloth_bend.wgsl does the rest (meshes, charring cloth). A vertex is held (not solved for) when it is
// pinned (X.w = 0), has burnt away or is not a multigrid fabric's.
//!include cloth_common.wgsl

struct Hinge { v: vec4<u32>, k: vec4<f32>, c: vec4<f32> };

struct Params {
  a: vec4<f32>,   // vertex count (this level), substep (s), Chebyshev: weight of the last step, weight of the residual
  b: vec4<f32>,   // fabrics using multigrid (bitmask), coarse vertex count, _, _
};

// -- level 0: the cloth ---------------------------------------------------------------------------------

@group(0) @binding(0) var<storage, read> XI: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> XO: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> XT: array<vec4<f32>>;     // positions predicted for this substep
@group(0) @binding(3) var<storage, read_write> DX: array<vec4<f32>>;     // Chebyshev direction (level 0) / residual
@group(0) @binding(4) var<storage, read> S: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> H: array<Hinge>;
@group(0) @binding(7) var<storage, read> VH: array<u32>;   // per vertex: first, count; then (hinge << 2 | slot)
@group(1) @binding(0) var<uniform> U: Params;

fn mg_fabric(fi: u32) -> bool {
  return (u32(U.b.x + 0.5) & (1u << fi)) != 0u;
}

fn handled(hg: Hinge) -> bool {
  if (hg.c.z < 0.5 || !mg_fabric(u32(V[hg.v.x].w + 0.5))) { return false; }
  let a = S[hg.v.x];
  let b = S[hg.v.y];
  let c = S[hg.v.z];
  let d = S[hg.v.w];
  if (gone(a) || gone(b) || gone(c) || gone(d)) { return false; }
  return 0.25 * (a.y + b.y + c.y + d.y) < 0.25;
}

fn held0(i: u32) -> bool {
  return XI[i].w <= 0.0 || gone(S[i]) || !mg_fabric(u32(V[i].w + 0.5));
}

// (M (x~ - x) / h^2 - Q x, diagonal of A) at vertex i
fn residual0(i: u32) -> vec4<f32> {
  let x = XI[i];
  let m = 1.0 / max(x.w, 1e-12);
  let h2 = U.a.y * U.a.y;
  var r = (XT[i].xyz - x.xyz) * (m / h2);
  var d = m / h2;
  let first = VH[2u * i];
  let cnt = VH[2u * i + 1u];
  for (var k = 0u; k < cnt; k++) {
    let e = VH[first + k];
    let hg = H[e >> 2u];
    if (!handled(hg)) { continue; }
    let s = e & 3u;
    let kh = 1.0 / max(hg.c.x, 1e-20);
    let v = hg.k.x * XI[hg.v.x].xyz + hg.k.y * XI[hg.v.y].xyz + hg.k.z * XI[hg.v.z].xyz + hg.k.w * XI[hg.v.w].xyz;
    let ks = select(select(hg.k.w, hg.k.z, s == 2u), select(hg.k.y, hg.k.x, s == 0u), s < 2u);
    r -= v * (kh * ks);
    d += kh * ks * ks;
  }
  return vec4<f32>(r, d);
}

// x~ = the predicted positions; Chebyshev starts from them
@compute @workgroup_size(64, 1, 1)
fn l0_start(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.a.x)) { return; }
  XT[i] = XI[i];
  DX[i] = vec4<f32>(0.0);
}

// one Chebyshev step: dx = c1 dx + c2 D^-1 r; x += dx (into XO)
@compute @workgroup_size(64, 1, 1)
fn l0_cheb(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.a.x)) { return; }
  let x = XI[i];
  if (held0(i)) {
    XO[i] = x;
    return;
  }
  let rd = residual0(i);
  // (x * (c / d), not x / d * c: the GPU's driver evaluates that one two ways, a bit apart, from one dispatch to
  // the next, and a run then depends on timing)
  let dx = DX[i].xyz * U.a.z + rd.xyz * (U.a.w / rd.w);
  DX[i] = vec4<f32>(dx, 0.0);
  XO[i] = vec4<f32>(x.xyz + dx, x.w);
}

// the residual (zero where held), into DX's slot of the level's residual buffer (XO here)
@compute @workgroup_size(64, 1, 1)
fn l0_residual(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.a.x)) { return; }
  if (held0(i)) {
    XO[i] = vec4<f32>(0.0);
    return;
  }
  XO[i] = vec4<f32>(residual0(i).xyz, 0.0);
}
