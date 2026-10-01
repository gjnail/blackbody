// Gas <- wet cloth in heat: the water the heat boils off the cloth (cloth_wick.wgsl measures it,
// cloth_splat.wgsl gathers it on the coupling grid) leaves it as steam at 100 C. The heat that boiled it
// came from the hot gas round the cloth, which it cools (2.26 MJ a kg); the steam then mixes in, as
// react.wgsl mixes the steam off a wet fuel bed: a volume of it pushes as much gas aside, and the cell
// takes the mixture's temperature and water. So a soaked cloth hung in the flames sits in a cool,
// steamy pocket of its own until it has dried, and steam billows off it, condensing white as it cools
// (shade.wgsl). On the finer upres grid only the temperature is changed (the water is on the main grid).
//!include common.wgsl

const FUEL_K: f32 = 1.0e9;
const CH: u32 = 13u;
const STEAM_RHO: f32 = 590.0;   // g/m^3 of pure steam at 100 C

struct Params {
  t: vec4<f32>,      // this grid's dims, cell size (m)
  cg: vec4<f32>,     // coupling grid dims, its cell size (m)
  k: vec4<f32>,      // dt, boiling point of water (field temperature), the air's own vapour (g/m^3),
                     // latent heat (field temperature per g/m^3)
  m: vec4<f32>,      // the main grid (1: water too) or the upres grid (0), _, _, _
};

@group(0) @binding(0) var src: texture_3d<f32>;
@group(0) @binding(1) var aux_src: texture_3d<f32>;
@group(0) @binding(2) var<storage, read> G: array<i32>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var aux_dst: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn gat(c: vec3<i32>) -> f32 {
  let d = vec3<i32>(U.cg.xyz);
  if (any(c < vec3<i32>(0)) || any(c >= d)) { return 0.0; }
  return f32(G[u32((c.z * d.y + c.y) * d.x + c.x) * CH + 12u]);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(U.t.xyz);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let main_grid = U.m.x > 0.5;
  var s = textureLoad(src, c, 0);
  var a = vec4<f32>(0.0);
  if (main_grid) { a = textureLoad(aux_src, c, 0); }
  let q = (vec3<f32>(c) + vec3<f32>(0.5)) * (U.t.w / U.cg.w) - vec3<f32>(0.5);
  let b = vec3<i32>(floor(q));
  let f = q - floor(q);
  var rate = 0.0;   // kg/s * FUEL_K
  for (var k = 0; k < 8; k++) {
    let o = vec3<i32>(k & 1, (k >> 1) & 1, k >> 2);
    let w3 = mix(vec3<f32>(1.0) - f, f, vec3<f32>(o));
    rate += w3.x * w3.y * w3.z * gat(b + o);
  }
  if (rate > 0.0) {
    let vol = U.cg.w * U.cg.w * U.cg.w;
    let st = rate / FUEL_K * 1000.0 / vol * U.k.x;   // g/m^3 of steam made here this step
    let boil = U.k.y;
    var T = s.x;
    // the heat that boiled it, out of the gas (never below the boil: that is as cool as the cloth gets it)
    if (T > boil) { T = max(T - U.k.w * st, boil); }
    // the steam mixes in at the boil
    let mix_a = st / STEAM_RHO;
    T = (T + boil * mix_a) / (1.0 + mix_a);
    s.x = T;
    if (main_grid) {
      let q_air = U.k.z;
      a.y = (q_air + max(a.y, 0.0) + st) / (1.0 + mix_a) - q_air;
    }
  }
  textureStore(dst, c, s);
  if (main_grid) { textureStore(aux_dst, c, a); }
}
