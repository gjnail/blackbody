// Lava -> the shared radiant sources (radiant.py): once a frame, every cell of lava with air beside it radiates from
// each open face, emissivity sigma (T^4 - Ta^4) a square metre (fresh lava at 1400 K about 200 kW/m^2, a cooling flow
// less), out toward the air: what scorches cloth hung near a flow, warms and melts what lies beside it, and heats the
// objects in the way. The lava is a liquid solver's (the second one in a fire-and-liquid box, or the scene's own): its
// particles per cell, and its heat (0 the air's temperature .. 1 fresh lava's, liq heat).
//!include common.wgsl

const SIGMA: f32 = 5.670e-8;
const EMISSIVITY: f32 = 0.95;    // basalt melt and its glassy skin

struct Params {
  g: Grid,           // the lava's grid
  k: vec4<f32>,      // ambient (K), fresh lava's temperature (K), its particles a full cell holds, _
  rg: RadGrid,
};

@group(0) @binding(0) var dens: texture_3d<f32>;
@group(0) @binding(1) var heat: texture_3d<f32>;
@group(0) @binding(2) var<storage, read_write> RA: array<atomic<i32>>;
@group(1) @binding(0) var<uniform> U: Params;

//!include rad_add.wgsl

fn share(c: vec3<i32>, n: vec3<i32>) -> f32 {
  if (any(c < vec3<i32>(0)) || any(c >= n)) { return 0.0; }
  return textureLoad(dens, c, 0).x / max(U.k.z, 1.0e-3);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(U.g.n.xyz);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  if (share(c, n) < 0.5) { return; }
  let tk = U.k.x + (U.k.y - U.k.x) * clamp(textureLoad(heat, c, 0).x, 0.0, 1.0);
  if (tk < U.k.x + 50.0) { return; }
  var out = vec3<f32>(0.0);
  for (var f = 0; f < 6; f++) {
    var e = vec3<i32>(0);
    e[f / 2] = select(-1, 1, (f & 1) == 1);
    let q = c + e;
    var open = share(q, n) < 0.5;
    if (any(q < vec3<i32>(0)) || any(q >= n)) { open = e.y > 0; }
    if (open) { out += vec3<f32>(e); }
  }
  let lo = length(out);
  if (lo < 1.0e-6) { return; }
  let h = U.g.n.w;
  let p = U.g.org.xyz + (vec3<f32>(c) + vec3<f32>(0.5)) * h;
  rad_add(U.rg, p, EMISSIVITY * SIGMA * (pow(tk, 4.0) - pow(U.k.x, 4.0)) * h * h * lo, out / lo);
}
