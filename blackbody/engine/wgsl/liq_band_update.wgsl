// Narrow band, pass 3: the new deep liquid, every liquid cell further than the band width from air, but for the cells
// round a bullet's way through the water (bullet_media.py): kept in the band while its cavity opens and closes, so
// particles are there for it to push aside.
// A joining cell counts as full from now on: the bank books the cell (the cull books the particles
// it hands over), so one that joins short of a full cell is paid back by the emission.
//!include common.wgsl
//!include liq_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,   // surface density (particles), band width (cells), particles per cell, bullets' ways kept (count)
  keep: array<vec4<f32>, 16>,   // each: a (fire-local m), its reach (m); b, _
};

@group(0) @binding(0) var dens: texture_3d<f32>;
@group(0) @binding(1) var<storage, read> band_old: array<u32>;
@group(0) @binding(2) var dist: texture_3d<f32>;
@group(0) @binding(3) var sdf: texture_3d<f32>;
@group(0) @binding(4) var<storage, read_write> band_new: array<u32>;
@group(0) @binding(5) var<storage, read_write> bank: array<atomic<i32>, 8>;   // see liq_band_bank.wgsl
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let i = nidx(c, n);
  let liquid = textureLoad(dens, c, 0).x >= U.k.x || band_old[i] != 0u;
  let solid = textureLoad(sdf, c, 0).x < 0.0;
  var kept = false;
  let w = U.g.org.xyz + (vec3<f32>(c) + vec3<f32>(0.5)) * U.g.n.w;
  for (var j = 0u; j < u32(U.k.w); j++) {
    let a = U.keep[2u * j];
    let ab = U.keep[2u * j + 1u].xyz - a.xyz;
    let t = clamp(dot(w - a.xyz, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
    if (length(w - a.xyz - ab * t) < a.w) { kept = true; }
  }
  let deep = liquid && !solid && textureLoad(dist, c, 0).x > U.k.y && !kept;
  band_new[i] = select(0u, 1u, deep);
  if (deep && band_old[i] == 0u) { atomicAdd(&bank[0], i32(U.k.z)); }   // a full cell more of deep liquid
}
