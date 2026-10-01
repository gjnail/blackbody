// Fire and liquid in one box: the water, as the fire sees it, and the fuel bed it lands on.
//
// Each substep, per fire cell, this writes the fire solver's water field (react.wgsl):
//   x = share of the cell filled with liquid now (it cools the gas and smothers fuel and flame there)
//   y = how soaked the fuel here is (0..1): wet fuel stops giving off fuel and heat
//   z = steam boiled off a hot, wet fuel bed (g/m^3 per second)
//   w = smoke from a doused bed still hot enough to char (1/s)
// z also carries, with the liquid's heat model on (liquid_thermal.py), the vapour the liquid gives the
// gas (evaporating, boiling); the heat it takes from the gas or gives it goes straight onto the
// fire's temperature here. Steam made under the water is lifted to the water's surface afterwards,
// and the lava's steam added there (both_rise.wgsl), so this writes a raw copy of the water field.
//
// A burning fuel bed (a fire emitter's shape) keeps state between steps (bed, x = soaked, y = heat):
// - Water reaching the bed (in the cell, or pooled in the few cells under it: it runs down into the
//   fuel and wicks up it) soaks it within a fraction of a second.
// - While it burns dry, the bed stores heat, like the embers of a real fire: it heats toward the
//   emitter's temperature in about a second. Soaked, it no longer burns, so it stops heating.
// - Water on a bed hotter than boiling boils away, carrying the bed's heat off as steam: the embers
//   hiss out in a thick, white plume over a second or two, however little of the gas was burning.
//   The steam comes out of the water soaked into the bed, so a bed boils itself dry unless water
//   keeps arriving.
// - Wet fuel dries out: slowly on its own, fast in the heat of flames around it (the rest of the fire
//   drying it back out, until it catches again and the fire creeps back over it).
//!include common.wgsl
//!include noise.wgsl
//!include meshsdf.wgsl
//!include emitters.wgsl

struct Params {
  g: Grid,
  nl: vec4<f32>,   // liquid grid dims; w = rest density (particles per cell)
  k: vec4<f32>,    // soaking rate (1/s), bed heating rate (1/s), bed quench rate when soaked (1/s), bed heat capacity (x the air's)
  k2: vec4<f32>,   // steam per unit of temperature removed (g/m^3), boiling point (field), drying in flames (1/s), drying in still air (1/s)
  k3: vec4<f32>,   // smoke from a doused hot bed (1/s at full heat), heat in the water a fully soaked bed holds (field temperature x capacity), lava on (1/0), liquid heat model on (1/0)
  cnt: vec4<f32>,  // emitter count
  em: array<Emitter, MAX_EMITTERS>,
};

@group(0) @binding(0) var dens: texture_3d<f32>;        // liquid particle density (liquid grid)
@group(0) @binding(1) var scal: texture_3d<f32>;        // fire scalars: x = temperature
@group(0) @binding(2) var bed_in: texture_3d<f32>;
@group(0) @binding(3) var atlas: texture_3d<f32>;
@group(0) @binding(4) var water: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var bed_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var lava_f: texture_3d<f32>;   // the lava field (both_lava.wgsl), or a 1-cell zero
@group(0) @binding(7) var gas_flux: texture_3d<f32>; // the liquid's heat model's: x = vapour (g/m^3/s), y = gas temperature change (field/s)
@group(0) @binding(8) var scal_out: texture_storage_3d<rgba16float, write>;   // the fire's scalars, with that heat
@group(1) @binding(0) var<uniform> U: Params;

// Share of fire cell c filled with liquid.
fn liquid_share(c: vec3<i32>, n: vec3<i32>) -> f32 {
  if (!in_grid(c, n)) { return 0.0; }
  let q = min(vec3<i32>(floor((vec3<f32>(c) + vec3<f32>(0.5)) * U.nl.xyz / vec3<f32>(n))), vec3<i32>(U.nl.xyz) - vec3<i32>(1));
  return clamp(textureLoad(dens, q, 0).x / max(U.nl.w, 1e-3), 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let h = U.g.n.w;
  let dt = U.g.bc.w;
  let wp = world_of(U.g, vec3<f32>(c) + vec3<f32>(0.5));
  let wf = liquid_share(c, n);

  // the fuel bed here: the fire emitters' shapes (the fuel they release burns off a bed of it), and
  // the heat they burn at
  var bed_mask = 0.0;
  var bed_temp = 0.0;
  let cnt = i32(U.cnt.x);
  for (var i = 0; i < cnt; i++) {
    let e = U.em[i];
    if (e.d.x <= 0.0 || e.k.w > 0.0) { continue; }   // no fuel, or an extinguisher
    if (i32(e.a.w + 0.5) == 7 && e.s.w > 1.5) { continue; }   // a volume that fills the box: smoke, not a bed
    let m = emitter_mask(e, wp, h);
    if (m <= 0.0) { continue; }
    bed_mask = max(bed_mask, m);
    bed_temp = max(bed_temp, e.d.y * m);
  }

  var s = textureLoad(bed_in, c, 0);
  var soak = clamp(s.x, 0.0, 1.0);
  var heat = max(s.y, 0.0);
  var steam = 0.0;
  var smoke = 0.0;
  if (bed_mask > 0.0) {
    // water reaching the bed: in the cell, or pooled in the cells under it (it wicks up the fuel)
    var reach = wf;
    for (var j = 1; j <= 4; j++) {
      reach = max(reach, liquid_share(c - vec3<i32>(0, j, 0), n) * (1.0 - 0.15 * f32(j)));
    }
    reach = max(reach, 0.5 * liquid_share(c + vec3<i32>(0, 1, 0), n));
    soak += (1.0 - soak) * (1.0 - exp(-U.k.x * reach * dt));

    // burning dry, the bed stores heat (it heats toward the fire's temperature); soaked, it cannot
    let goal = bed_temp * (1.0 - soak);
    if (heat < goal) { heat += (goal - heat) * (1.0 - exp(-U.k.y * dt)); }

    // water on a bed hotter than boiling boils away and takes the bed's heat with it, as steam
    let boil = U.k2.y;
    let cap = U.k.w;
    if (soak > 1e-3 && heat > boil) {
      let q = (heat - boil) * (1.0 - exp(-U.k.z * soak * dt));
      heat -= q;
      steam += q * cap * U.k2.x / max(dt, 1e-6);
      // the water that boiled came out of the bed
      soak = max(soak - q * cap / max(U.k3.y, 1e-4), 0.0);
      // still glowing under the water, the wood chars: the white-grey smoke of a doused fire
      smoke = U.k3.x * clamp(heat / max(bed_temp, 1e-3), 0.0, 1.0) * soak;
    }
    // below boiling the water still cools the embers, without boiling
    heat *= exp(-0.2 * U.k.z * soak * select(0.0, 1.0, heat <= boil) * dt);

    // drying: slowly on its own, fast in the heat of flames around it; that water leaves as steam too
    let t_gas = max(textureLoad(scal, c, 0).x, 0.0);
    let hot = clamp((t_gas - boil) / max(0.5 - boil, 1e-3), 0.0, 1.0);
    let dry = soak * (1.0 - exp(-(U.k2.w + U.k2.z * hot) * dt)) * (1.0 - clamp(reach * 4.0, 0.0, 1.0));
    soak -= dry;
    // (only part of the heat drying it comes from the bed: the rest is the flames', gone with their gas)
    steam += 0.25 * dry * U.k3.y * U.k2.x / max(dt, 1e-6);
  } else {
    soak = 0.0;
    heat = 0.0;
  }
  var sc = textureLoad(scal, c, 0);
  if (U.k3.w > 0.5) {
    let gf = textureLoad(gas_flux, c, 0);
    steam += max(gf.x, 0.0);
    sc.x = max(sc.x + gf.y * dt, 0.0);
  }
  textureStore(scal_out, c, sc);
  textureStore(bed_out, c, vec4<f32>(soak, heat, 0.0, 0.0));
  textureStore(water, c, vec4<f32>(wf, soak * bed_mask, steam, smoke));
}
