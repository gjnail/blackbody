// Fabric bending multigrid: a coarser level's correction added onto the next finer coarse level.

struct Params {
  a: vec4<f32>,   // vertex count (the finer level), _, _, _
  b: vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> XC: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> PC: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> PW: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> DG: array<vec4<f32>>;   // the finer level's: held in z
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.a.x)) { return; }
  if (DG[i].z > 0.5) { return; }
  let c = PC[i];
  let w = PW[i];
  var e = vec3<f32>(0.0);
  if (c.x != 0xffffffffu) { e += XC[c.x].xyz * w.x; }
  if (c.y != 0xffffffffu) { e += XC[c.y].xyz * w.y; }
  if (c.z != 0xffffffffu) { e += XC[c.z].xyz * w.z; }
  if (c.w != 0xffffffffu) { e += XC[c.w].xyz * w.w; }
  X[i] = vec4<f32>(X[i].xyz + e, 0.0);
}
