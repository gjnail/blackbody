// Fire -> fabric, radiation (1 of 3): the power the hot gas radiates, gathered on a coarse grid. Flames
// are optically thin at these sizes: each cell gives off 4 kappa sigma (T^4 - Ta^4) per m^3 (kappa, the
// absorption coefficient of the flames, a few tenths per metre: a campfire with 1.2 m flames radiates
// about 40 kW, a fifth to a third of the heat it releases, as real fires do). T is the gas's real
// temperature (flames 1100-1300 K), not the look's flame colour temperature, which is often far hotter.
// Per coarse cell: the power-weighted centre of what it radiates, and the power (W). cloth_rad.wgsl adds
// it up at each vertex of the cloth.
//!include common.wgsl

const SIGMA: f32 = 5.670e-8;

struct Params {
  g: vec4<f32>,      // gas grid dims, its cell size (m)
  org: vec4<f32>,    // its corner (fire-local m), absorption coefficient of the flames (1/m)
  air: vec4<f32>,    // ambient K, the flames' real temperature (K), the hottest gas's (K), gas cells per coarse cell
  cd: vec4<f32>,     // coarse dims, coarse cells
};

@group(0) @binding(0) var scal: texture_3d<f32>;
@group(0) @binding(1) var<storage, read_write> SRC: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> U: Params;

fn kelvin(T: f32) -> f32 {
  let amb = U.air.x;
  return amb + (U.air.y - amb) * min(T, 1.0) + (U.air.z - U.air.y) * (1.0 - exp(-max(T - 1.0, 0.0)));
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let ci = id.x;
  if (ci >= u32(U.cd.w)) { return; }
  let cd = vec3<u32>(U.cd.xyz);
  let cc = vec3<i32>(vec3<u32>(ci % cd.x, (ci / cd.x) % cd.y, ci / (cd.x * cd.y)));
  let n = vec3<i32>(U.g.xyz);
  let bs = i32(U.air.w);
  let h = U.g.w;
  let ta4 = pow(U.air.x, 4.0);
  let k = 4.0 * U.org.w * SIGMA * h * h * h;
  var sum = 0.0;
  var cen = vec3<f32>(0.0);
  for (var z = 0; z < bs; z++) {
    for (var y = 0; y < bs; y++) {
      for (var x = 0; x < bs; x++) {
        let q = cc * bs + vec3<i32>(x, y, z);
        if (any(q >= n)) { continue; }
        let T = textureLoad(scal, q, 0).x;
        if (T <= 0.05) { continue; }
        let tk = kelvin(T);
        let e = k * max(tk * tk * tk * tk - ta4, 0.0);
        sum += e;
        cen += e * (vec3<f32>(q) + vec3<f32>(0.5));
      }
    }
  }
  var p = vec3<f32>(0.0);
  if (sum > 0.0) { p = U.org.xyz + cen / sum * h; }
  SRC[ci] = vec4<f32>(p, sum);
}
