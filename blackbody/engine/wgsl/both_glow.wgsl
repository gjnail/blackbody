// Fire, water and lava in one box: the fire's light volume (light_emit.wgsl) with the lava in it.
// Molten lava glows like the flames do, so it goes into the same emission the light and shadow
// volumes and the point lights for the surfaces are built from: the steam over a lava shore lights
// up orange from underneath, the smoke and the water catch its light, and the water reflects it.
// A glowing surface in a light cell gives that cell B(T) * (its area / the cell's volume) of
// emission per metre; the surface is found from the lava's rendered surface (its signed distance
// and heat, liq_surf_resolve.wgsl), so cached frames light the same as live ones. Lava under the
// water shines into it, not into the air, so it is left out there.
//!include common.wgsl

@group(0) @binding(0) var scal: texture_3d<f32>;
@group(0) @binding(1) var aux: texture_3d<f32>;
@group(0) @binding(2) var chem: texture_3d<f32>;
@group(0) @binding(3) var bb: texture_2d<f32>;
@group(0) @binding(4) var lin: sampler;
@group(0) @binding(5) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var lava_surf: texture_3d<f32>;    // x = signed distance (surface cells), negative inside
@group(0) @binding(7) var lava_heat: texture_3d<f32>;    // x = heat, y = heat * cover, z = cover
@group(0) @binding(8) var water_surf: texture_3d<f32>;

//!include shade.wgsl

struct Params {
  n: vec4<f32>,    // simulation dims
  ln: vec4<f32>,   // light-volume dims
  look: Look,
  lv: vec4<f32>,   // lava's surface cells per simulation cell, temperature of fresh lava (K), glow (x), water surface on (1/0)
  lv2: vec4<f32>,  // simulation cell (m), the water's surface cells per simulation cell
};
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (any(c >= vec3<i32>(U.ln.xyz))) { return; }
  let p = (vec3<f32>(c) + 0.5) * (U.n.xyz / U.ln.xyz);
  let s = samp_c(scal, lin, p, U.n.xyz);
  var a = vec4<f32>(0.0);
  if (U.look.air.z > 0.5) { a = samp_c(aux, lin, p, U.n.xyz); }
  var ch = vec4<f32>(0.0);
  if (U.look.air.w > 0.5) { ch = samp_c(chem, lin, p, U.n.xyz); }
  var e = emission(s, ch, U.look);

  // the lava's glowing surface in this light cell
  let uvw = p / U.n.xyz;
  let d_cells = textureSampleLevel(lava_surf, lin, uvw, 0.0).x / max(U.lv.x, 1e-3);
  let cell_l = U.n.x / U.ln.x;   // simulation cells per light cell
  let w = clamp(1.0 - abs(d_cells) / cell_l, 0.0, 1.0);
  if (w > 0.0) {
    let hv = textureSampleLevel(lava_heat, lin, uvw, 0.0);
    let heat = select(0.0, hv.y / hv.z, hv.z > 1e-3);
    var in_water = false;
    if (U.lv.w > 0.5) {
      in_water = textureSampleLevel(water_surf, lin, uvw, 0.0).x / max(U.lv2.y, 1e-3) < -0.5;
    }
    if (heat > 0.0 && !in_water) {
      let tk = U.look.fire.x + (U.lv.y - U.look.fire.x) * clamp(heat, 0.0, 1.2);
      // the temperature on the field's scale, for the blackbody radiance the flames use
      let t_field = (tk - U.look.fire.x) / max(U.look.fire.y - U.look.fire.x, 1.0);
      let area_per_volume = w / (cell_l * U.lv2.x);
      e += bb_radiance(t_field, U.look) * U.look.fire2.x * area_per_volume * U.lv.z;
    }
  }
  textureStore(dst, c, vec4<f32>(e, smoke_extinction(s, U.look) + steam_extinction(s, a, U.look)));
}
