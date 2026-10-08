// Weather: the live precipitation particles packed for the renderer and the frame cache (engine/
// weather.py), three vec4 each: position (sim-local m) and size (mm); velocity (m/s) and phase (0..1);
// kind, share melted, resting (1/0), temperature (C).
//
// They are packed in the order of their slots, so the renderer draws them in the same order every run (it
// blends them over one another): wx_step.wgsl counts the free slots in each block of 64 as it ends,
// block_scan.wgsl turns the rest of each block (its live ones) into where each block starts, and `main` puts
// each particle where its block starts plus the live ones before it in the block. (`count` counts the free
// slots afresh, when the particles were set from outside: a reset.)
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
@group(0) @binding(3) var<storage, read_write> blk: array<u32>;   // main: live ones before each block; count: free ones in it
@group(1) @binding(0) var<uniform> U: Params;

//!include wx_common.wgsl

var<workgroup> wn: atomic<u32>;
var<workgroup> live: array<u32, 64>;

@compute @workgroup_size(64, 1, 1)
fn count(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
         @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  // (workgroup memory is not always cleared for each workgroup: the backend may skip it)
  if (li == 0u) { atomicStore(&wn, 0u); }
  workgroupBarrier();
  let i = lin_id(id, nwg);
  let n = arrayLength(&parts);
  if (i < n && parts[i].p.w < 0.0) { atomicAdd(&wn, 1u); }
  workgroupBarrier();
  let b = wid.x + wid.y * nwg.x;
  if (li == 0u && b * 64u < n) { blk[b] = atomicLoad(&wn); }
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = lin_id(id, nwg);
  var P: WxP;
  var on = false;
  if (i < arrayLength(&parts)) {
    P = parts[i];
    on = P.p.w >= 0.0;
  }
  live[li] = select(0u, 1u, on);
  workgroupBarrier();
  if (!on) { return; }
  var j = blk[wid.x + wid.y * nwg.x];
  for (var l = 0u; l < li; l++) { j += live[l]; }
  atomicAdd(&ctr[15], 1u);
  if (j >= u32(U.k.x)) { return; }
  let s = WxState(u32(P.k.x + 0.5), P.m.x, P.m.y, P.m.z, P.m.w, P.k.y > 0.5, P.k.z);
  var kind = s.kind;
  if (s.ice <= 0.0) { kind = WX_RAIN; }
  packed[j * 3u] = vec4<f32>(P.p.xyz, wx_diameter(s));
  packed[j * 3u + 1u] = vec4<f32>(P.v.xyz, P.k.w);
  packed[j * 3u + 2u] = vec4<f32>(f32(kind), wx_melted(s), P.v.w, s.t);
}
