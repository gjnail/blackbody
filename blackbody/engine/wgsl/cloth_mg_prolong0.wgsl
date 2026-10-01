// Fabric bending multigrid: the level-1 correction added onto the cloth's positions (level 0), but not
// onto held vertices, nor onto a fabric that has burnt through somewhere (its coarse levels no longer
// describe it: its fine level alone carries on).
//!include cloth_common.wgsl

struct Params {
  a: vec4<f32>,   // vertex count (level 0), _, _, _
  b: vec4<f32>,   // fabrics using multigrid (bitmask), _, _, _
};

@group(0) @binding(0) var<storage, read_write> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> XC: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> PC: array<vec4<u32>>;   // up to four coarse vertices (0xffffffff: none)
@group(0) @binding(3) var<storage, read> PW: array<vec4<f32>>;   // and their weights
@group(0) @binding(4) var<storage, read> S: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> HOLED: array<u32>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.a.x)) { return; }
  let x = X[i];
  let fi = u32(V[i].w + 0.5);
  if (x.w <= 0.0 || gone(S[i]) || (u32(U.b.x + 0.5) & (1u << fi)) == 0u || HOLED[fi] != 0u) { return; }
  let c = PC[i];
  let w = PW[i];
  var e = vec3<f32>(0.0);
  if (c.x != 0xffffffffu) { e += XC[c.x].xyz * w.x; }
  if (c.y != 0xffffffffu) { e += XC[c.y].xyz * w.y; }
  if (c.z != 0xffffffffu) { e += XC[c.z].xyz * w.z; }
  if (c.w != 0xffffffffu) { e += XC[c.w].xyz * w.w; }
  X[i] = vec4<f32>(x.xyz + e, x.w);
}
