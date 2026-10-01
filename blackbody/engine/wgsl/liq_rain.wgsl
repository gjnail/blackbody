// Rain on the liquid: drops land on its surface at random, each throwing up a small crown of spray
// (whitewater spray particles, which fly, fall back and fade as foam). One thread per column finds
// the liquid's top surface there and lets as many drops land on it this step as the rain rate brings
// down on a cell's area. The ring ripples they leave are drawn by the renderer (liq_march.wgsl).
//!include common.wgsl
//!include liq_common.wgsl
//!include noise.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,  // drops per cell top and step, impact speed (m/s), whitewater capacity, surface density (particles)
  s: vec4<f32>,  // step seed, spray life (s)
};

@group(0) @binding(0) var dens: texture_3d<f32>;
@group(0) @binding(1) var sdf: texture_3d<f32>;
@group(0) @binding(2) var<storage, read_write> WA: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> WB: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> wctr: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  if (i32(id.x) >= n.x || i32(id.y) >= n.z) { return; }
  let seed = (id.x * 73856093u) ^ (id.y * 19349663u) ^ (u32(U.s.x) * 83492791u);
  let drops = u32(floor(U.k.x + rand1(seed)));
  if (drops == 0u) { return; }
  // the top of the liquid in this column (a solid above it keeps the rain off)
  var top = -1;
  for (var y = n.y - 1; y >= 0; y--) {
    let c = vec3<i32>(i32(id.x), y, i32(id.y));
    if (textureLoad(sdf, c, 0).x < 0.0) { return; }
    if (textureLoad(dens, c, 0).x > U.k.w) {
      top = y;
      break;
    }
  }
  if (top < 0) { return; }
  let cap = u32(U.k.z);
  let vt = U.k.y;
  for (var d = 0u; d < min(drops, 4u); d++) {
    let r0 = rand3(seed, d * 31u + 5u);
    let at = vec3<f32>(f32(id.x) + r0.x, f32(top) + 1.0, f32(id.y) + r0.z);
    let cnt = 4u + u32(r0.y * 5.0);
    for (var j = 0u; j < cnt; j++) {
      let r = rand3(seed, d * 31u + j * 7u + 11u);
      let ang = 6.2831853 * (f32(j) + r.x) / f32(cnt);
      let out = vec3<f32>(cos(ang), 0.0, sin(ang));
      let v = out * (0.06 * vt * (0.5 + r.y)) + vec3<f32>(0.0, 0.1 * vt * (0.5 + r.z), 0.0);
      let slot = atomicAdd(&wctr[0], 1u) % cap;
      WA[slot] = vec4<f32>(at + out * 0.3, U.s.y * (0.6 + 0.8 * r.z));
      WB[slot] = vec4<f32>(v, 0.0);
    }
  }
}
