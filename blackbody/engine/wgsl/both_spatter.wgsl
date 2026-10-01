// Fire, water and lava in one box: where water meets lava it does not simmer, it spits. Water that
// gets under the lava's skin or trapped against it flashes to steam faster than it can get away and
// bursts, throwing spray (and flecks of the water) up out of the boiling line; at a lava shore that is
// the crackle of small steam explosions along the waterline. Each open cell the lava's steam comes out
// of (both_lava.wgsl, field z) throws whitewater spray upward at a rate that follows the steam.
//!include common.wgsl
//!include liq_common.wgsl
//!include noise.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,  // spray drops per g/m^3 of steam (per cell), launch speed (m/s), whitewater capacity, spray life (s)
  s: vec4<f32>,  // step seed
};

@group(0) @binding(0) var field: texture_3d<f32>;   // the lava field: z = steam into the gas (g/m^3/s)
@group(0) @binding(1) var<storage, read_write> WA: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> WB: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> wctr: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let steam = textureLoad(field, c, 0).z * U.g.bc.w;   // g/m^3 this step
  if (steam <= 0.0) { return; }
  let seed = (id.x * 73856093u) ^ (id.y * 19349663u) ^ (id.z * 83492791u) ^ (u32(U.s.x) * 2654435761u);
  // bursts come in fits: most steps nothing, now and then a handful
  let want = U.k.x * steam;
  let cnt = u32(floor(want + rand1(seed)));
  if (cnt == 0u) { return; }
  let cap = u32(U.k.z);
  for (var j = 0u; j < min(cnt, 6u); j++) {
    let r = rand3(seed, j * 13u + 3u);
    let r2 = rand3(seed, j * 29u + 7u);
    let ang = 6.2831853 * r.x;
    let up = 0.35 + 0.65 * r.y;
    let side = sqrt(max(1.0 - up * up, 0.0));
    let v = vec3<f32>(cos(ang) * side, up, sin(ang) * side) * (U.k.y * (0.4 + 1.2 * r2.x));
    let slot = atomicAdd(&wctr[0], 1u) % cap;
    WA[slot] = vec4<f32>(vec3<f32>(c) + vec3<f32>(r2.y, 0.2 * r.z, r2.z), U.k.w * (0.5 + r.z));
    WB[slot] = vec4<f32>(v, 0.0);
  }
}
