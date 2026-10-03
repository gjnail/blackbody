// Gas <- hot objects (objheat.py): the air in the cells beside a hot object's surface is heated by convection, as the
// object is cooled by it (natural convection, 1.52 dT^(1/3) W/m^2/K, as liq_therm_heat.wgsl and objheat.py take it),
// so a red-hot bar sends up a plume of shimmering air and lights what burns above it. Each cell takes the heat through
// the face toward the surface, never past the object's own temperature. The gas's heat is its real temperature mapped
// back onto its 0 (ambient) .. 1 (flame) scale (radiant.py gas_kelvin).
//!include common.wgsl
//!include meshsdf.wgsl
//!include colliders.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,      // dt (s), ambient (K), the flames' real temperature (K), the hottest gas's (K)
  ccnt: vec4<f32>,
  col: array<Collider, MAX_COLLIDERS>,
  temp: array<vec4<f32>, 4>,   // each object's surface temperature (K)
};

@group(0) @binding(0) var src: texture_3d<f32>;
@group(0) @binding(1) var atlas: texture_3d<f32>;
@group(0) @binding(2) var dst: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn kelvin(t: f32) -> f32 {
  let amb = U.k.y;
  return amb + (U.k.z - amb) * min(t, 1.0) + (U.k.w - U.k.z) * (1.0 - exp(-max(t - 1.0, 0.0)));
}

fn heat_of(tk: f32) -> f32 {
  let amb = U.k.y;
  if (tk <= U.k.z) { return (tk - amb) / max(U.k.z - amb, 1.0); }
  return 1.0 - log(max(1.0 - (tk - U.k.z) / max(U.k.w - U.k.z, 1.0), 1.0e-4));
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(U.g.n.xyz);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  var s = textureLoad(src, c, 0);
  let h = U.g.n.w;
  let wp = U.g.org.xyz + (vec3<f32>(c) + vec3<f32>(0.5)) * h;
  var best = 1.5 * h;
  var ts = 0.0;
  for (var k = 0u; k < u32(U.ccnt.x); k++) {
    let d = col_sdf(U.col[k], wp);
    if (d > 0.0 && d < best) {
      best = d;
      ts = U.temp[k / 4u][k % 4u];
    }
  }
  if (ts > 0.0) {
    let tg = kelvin(max(s.x, 0.0));
    let diff = ts - tg;
    if (diff > 2.0) {
      let hc = max(1.52 * pow(diff, 1.0 / 3.0), 5.7);
      let rc = 3.55e5 / tg;                       // air's heat a cubic metre and kelvin (rho c_p, ideal gas)
      let t2 = tg + diff * (1.0 - exp(-hc * U.k.x / (rc * h)));
      s.x = max(s.x, heat_of(t2));
    }
  }
  textureStore(dst, c, s);
}
