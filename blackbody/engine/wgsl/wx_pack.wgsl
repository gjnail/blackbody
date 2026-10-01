// Weather: the live precipitation particles packed for the renderer and the frame cache (engine/
// weather.py), three vec4 each: position (sim-local m) and size (mm); velocity (m/s) and phase (0..1);
// kind, share melted, resting (1/0), temperature (C).
//!include liq_common.wgsl

struct WxP {
  p: vec4<f32>,
  v: vec4<f32>,
  m: vec4<f32>,
  k: vec4<f32>,
};

struct Params {
  k: vec4<f32>,   // capacity of the packed buffer
};

@group(0) @binding(0) var<storage, read> parts: array<WxP>;
@group(0) @binding(1) var<storage, read_write> packed: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> ctr: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> U: Params;

//!include wx_common.wgsl

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= arrayLength(&parts)) { return; }
  let P = parts[i];
  if (P.p.w < 0.0) { return; }
  let j = atomicAdd(&ctr[15], 1u);
  if (j >= u32(U.k.x)) { return; }
  let s = WxState(u32(P.k.x + 0.5), P.m.x, P.m.y, P.m.z, P.m.w, P.k.y > 0.5, P.k.z);
  var kind = s.kind;
  if (s.ice <= 0.0) { kind = WX_RAIN; }
  packed[j * 3u] = vec4<f32>(P.p.xyz, wx_diameter(s));
  packed[j * 3u + 1u] = vec4<f32>(P.v.xyz, P.k.w);
  packed[j * 3u + 2u] = vec4<f32>(f32(kind), wx_melted(s), P.v.w, s.t);
}
