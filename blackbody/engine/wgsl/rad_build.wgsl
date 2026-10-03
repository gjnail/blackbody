// The frame's shared radiant sources (radiant.py), built:
//  - build: per coarse cell, the power its hot gas radiates (4 kappa sigma (T^4 - Ta^4) a cubic metre, from the gas's
//    real temperature) and what glowing matter and lava added to it (RA), as one source at the power-weighted centre
//    of both, if anything radiates there
//  - objects: the objects' faces (objheat.py), after the cells
//!include common.wgsl

const SIGMA: f32 = 5.670e-8;
const RAD_CELLS: u32 = 4096u;     // radiant.CELLS
const RAD_SOURCES: u32 = 4224u;   // radiant.SOURCES

struct Params {
  o: vec4<f32>,    // corner (fire-local m), coarse cell size (m)
  cd: vec4<f32>,   // coarse cells (x, y, z), how many
  t: vec4<f32>,    // ambient (K), the flames' real temperature (K), the hottest gas's (K), gas cells to a coarse cell
                   // (0: no gas)
  k: vec4<f32>,    // flame absorption (1/m), the gas's cell size (m), objects' faces, 1 / the accumulator's fixed point
  gn: vec4<f32>,   // the gas's cells (x, y, z)
};

@group(0) @binding(0) var scal: texture_3d<f32>;   // the gas: x its heat (0 ambient .. 1 flame)
@group(0) @binding(1) var<storage, read_write> RA: array<i32>;   // build: the accumulator; objects: their faces (floats)
@group(0) @binding(2) var<storage, read_write> RL: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> RLC: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> U: Params;

fn kelvin(T: f32) -> f32 {
  let amb = U.t.x;
  return amb + (U.t.y - amb) * min(T, 1.0) + (U.t.z - U.t.y) * (1.0 - exp(-max(T - 1.0, 0.0)));
}

@compute @workgroup_size(64, 1, 1)
fn build(@builtin(global_invocation_id) id: vec3<u32>) {
  let ci = id.x;
  if (ci >= u32(U.cd.w)) { return; }
  let cd = vec3<u32>(U.cd.xyz);
  let cc = vec3<i32>(vec3<u32>(ci % cd.x, (ci / cd.x) % cd.y, ci / (cd.x * cd.y)));
  // the gas in it
  var pv = 0.0;
  var cv = vec3<f32>(0.0);
  let bs = i32(U.t.w);
  if (bs > 0) {
    let n = vec3<i32>(U.gn.xyz);
    let h = U.k.y;
    let ta4 = pow(U.t.x, 4.0);
    let kk = 4.0 * U.k.x * SIGMA * h * h * h;
    for (var z = 0; z < bs; z++) {
      for (var y = 0; y < bs; y++) {
        for (var x = 0; x < bs; x++) {
          let q = cc * bs + vec3<i32>(x, y, z);
          if (any(q >= n)) { continue; }
          let T = textureLoad(scal, q, 0).x;
          if (T <= 0.05) { continue; }
          let tk = kelvin(T);
          let e = kk * max(tk * tk * tk * tk - ta4, 0.0);
          pv += e;
          cv += e * (vec3<f32>(q) + vec3<f32>(0.5)) * h;
        }
      }
    }
  }
  // the glowing surfaces added to it
  let k = 7u * ci;
  let inv = U.k.w;
  let ps = max(f32(RA[k]) * inv, 0.0);
  let corner = U.o.xyz + vec3<f32>(cc) * U.o.w;
  var cs = corner + 0.5 * U.o.w;
  var nsum = vec3<f32>(0.0);
  if (ps > 0.0) {
    cs = corner + vec3<f32>(f32(RA[k + 1u]), f32(RA[k + 2u]), f32(RA[k + 3u])) * inv / ps * U.o.w;
    nsum = vec3<f32>(f32(RA[k + 4u]), f32(RA[k + 5u]), f32(RA[k + 6u])) * inv;
  }
  let total = pv + ps;
  if (total < 1.0e-3) { return; }
  var cen = cs;
  if (pv > 0.0) { cen = (U.o.xyz + cv / pv) * (pv / total) + cs * (ps / total); }
  let i = atomicAdd(&RLC[0], 1u);
  if (i >= RAD_CELLS) { return; }
  let soft = 0.5 * U.o.w;
  RL[3u * i] = vec4<f32>(cen, soft * soft);
  RL[3u * i + 1u] = vec4<f32>(pv, ps, -1.0, 0.0);
  RL[3u * i + 2u] = vec4<f32>(nsum, 0.0);
}

@compute @workgroup_size(64, 1, 1)
fn objects(@builtin(global_invocation_id) id: vec3<u32>) {
  let j = id.x;
  if (j >= u32(U.k.z)) { return; }
  let i = atomicAdd(&RLC[0], 1u);
  if (i >= RAD_SOURCES) { return; }
  for (var w = 0u; w < 3u; w++) {
    let b = 12u * j + 4u * w;
    RL[3u * i + w] = vec4<f32>(bitcast<f32>(RA[b]), bitcast<f32>(RA[b + 1u]), bitcast<f32>(RA[b + 2u]), bitcast<f32>(RA[b + 3u]));
  }
}
