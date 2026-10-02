// Grass (engine/strands.py) on the ground: a coarse map over the patches of how burnt the grass is at each spot and how
// thickly it stands there, for the stage's floor (stage.wgsl): black with char under burnt grass, in shade under thick.
// Each blade adds itself to the cell its root is in (fixed-point atomics); then each cell takes in its neighbours, so
// the gaps between blades fill.
//!include strand_common.wgsl

struct Params {
  sim: vec4<f32>,    // blades, map cells across x, across z, cell size (m)
  lo: vec4<f32>,     // the map's corner (fire-local x, z), _, _
};

@group(0) @binding(0) var<storage, read> BL: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> RT: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> ST: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> M: array<atomic<u32>>;   // per cell: burnt (the most, x 65535), blade area (x 65536 m^2 per m^2)
@group(0) @binding(4) var out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn clear(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = id.x + id.y * nwg.x * 64u;
  if (i >= 2u * u32(U.sim.y) * u32(U.sim.z)) { return; }
  atomicStore(&M[i], 0u);
}

@compute @workgroup_size(64, 1, 1)
fn splat(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.sim.x)) { return; }
  let rt = RT[i];
  if (rt.w < 0.5) { return; }
  let c = vec2<i32>(floor((rt.xz - U.lo.xy) / U.sim.w));
  let nx = i32(U.sim.y);
  if (any(c < vec2<i32>(0)) || c.x >= nx || c.y >= i32(U.sim.z)) { return; }
  let k = u32(c.y * nx + c.x);
  let burnt = clamp(ST[i].y, 0.0, 1.0);
  let b1 = BL[2u * i + 1u];
  let area = b1.y * b1.x * (1.0 - (1.0 - STUBBLE) * burnt);    // the blade (width x height, what is left of it)
  atomicMax(&M[2u * k], u32(burnt * 65535.0));
  atomicAdd(&M[2u * k + 1u], u32(area / (U.sim.w * U.sim.w) * 65536.0));
}

@compute @workgroup_size(8, 8, 1)
fn resolve(@builtin(global_invocation_id) id: vec3<u32>) {
  let nx = i32(U.sim.y);
  let nz = i32(U.sim.z);
  let c = vec2<i32>(id.xy);
  if (c.x >= nx || c.y >= nz) { return; }
  // a cell with blades in it is as burnt as the most burnt of them; one without, as the most burnt round it
  let k0 = u32(c.y * nx + c.x);
  let own = atomicLoad(&M[2u * k0 + 1u]) > 0u;
  var burnt = f32(atomicLoad(&M[2u * k0])) / 65535.0;
  var round = 0.0;
  var area = 0.0;
  var wsum = 0.0;
  for (var dz = -1; dz <= 1; dz++) {
    for (var dx = -1; dx <= 1; dx++) {
      let q = c + vec2<i32>(dx, dz);
      if (q.x < 0 || q.y < 0 || q.x >= nx || q.y >= nz) { continue; }
      let k = u32(q.y * nx + q.x);
      let w = select(0.6, 1.0, dx == 0 && dz == 0);
      round = max(round, f32(atomicLoad(&M[2u * k])) / 65535.0);
      area += w * f32(atomicLoad(&M[2u * k + 1u])) / 65536.0;
      wsum += w;
    }
  }
  if (!own) { burnt = round; }
  // blade area per ground area -> how much of the sky the grass hides from the ground under it
  let shade = 1.0 - exp(-0.5 * area / max(wsum, 1e-6));
  textureStore(out, c, vec4<f32>(burnt, shade, 0.0, 1.0));
}
