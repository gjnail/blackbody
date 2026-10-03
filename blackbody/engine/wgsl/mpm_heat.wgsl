// Matter's heat (matter.py): wax, chocolate and metal warm in the fire and cool in the air and the water, their heat
// evening out through them, and past their melting point they melt into a liquid of the same stuff (a runny one, or
// chocolate's thick one), which sets again where it cools below it: a candle's top melts into a pool that runs down
// its side and sets in drips, chocolate beside a fire slumps into a puddle, an ingot in a furnace glows and runs.
//  - the heat radiated onto it is the frame's shared radiant sources' (radiant.py: the fire's hot gas, lava, hot
//    objects), and what glows radiates into them in turn (rad_add.wgsl), for the cloth, the objects and the next frame
//  - splat: each particle's heat to the eight nodes round it (fixed point), for the heat to even out through the matter
//  - main: each particle takes on its neighbourhood's temperature (the nodes'); where it meets the air, the air's (the
//    gas's next to it in a fire box; the ambient air's otherwise), the fire's radiant heat from the side it faces out
//    to, and the heat it radiates itself; where it meets the liquid, the water's; where it touches an object, the
//    object's (a hot pan, a cold mould: as fast as the two conduct, by their effusivities), adding up the heat each
//    object gives or takes (objheat.py warms and cools it by that); where it lies on the ground, the ground's; and melts
//    or sets. What burns (dry
//    leaves, sawdust, coal) catches past its ignition point and burns down to ash, held at its burning temperature
//    where the air reaches it and smouldering slowly inside a heap; mpm_fire.wgsl gives the gas its flames.
// A particle's temperature (K) is its f0.w.
//!include common.wgsl
//!include mpm_common.wgsl
//!include meshsdf.wgsl
//!include colliders.wgsl

struct Params {
  m: vec4<f32>,      // the matter's grid: node 0 (fire-local m), node spacing (m)
  mn: vec4<f32>,     // its nodes (x, y, z), particles (count)
  g: Grid,           // the gas's grid
  lq: vec4<f32>,     // the liquid's grid corner (fire-local m), its cell size (m)
  ln: vec4<f32>,     // its cells (x, y, z), liquid on (1/0)
  k: vec4<f32>,      // dt (s), ambient (K), flame (K), gas on (1/0)
  lt: vec4<f32>,     // heat sources: block size (cells), the gas's hottest it counts from (0..1), flame absorption
                     // (1/m), most sources
  heat: array<vec4<f32>, 16>,   // per material slot: melts at (K; 0: never), what it melts into, what it sets into (slots,
                                // -1: none), how fast its surface takes on the air's temperature (1/s)
  cond: array<vec4<f32>, 16>,   // per material slot: how fast its heat evens out (1/s), how much faster the water cools
                                // it than the air, how much a W/m^2 of radiant heat warms its surface (K/s), its thermal
                                // effusivity (W s^0.5/m^2/K)
  burn: array<vec4<f32>, 16>,   // per material slot: catches at (K; 0: it does not burn), the share of it that burns away a
                                // second, how hot it burns (K), what it burns down to (slot, -1: nothing)
  ash: array<vec4<f32>, 4>,     // per material slot (16): the share of it left as that once burnt (the rest is gone)
  ccnt: vec4<f32>,              // the objects (solver.pack_colliders)
  col: array<Collider, MAX_COLLIDERS>,
  touch: array<vec4<f32>, MAX_COLLIDERS>,   // per object: its temperature (K), its effusivity (W s^0.5/m^2/K)
  rg: RadGrid,                  // the shared radiant sources' coarse grid, for what glows (rad_add.wgsl)
  rad: vec4<f32>,               // radiant sources on (1/0), the air grid's cell size (m: a glowing particle's surface is its
                                // volume over it), a particle's volume (m^3), objects' heat added up (1/0)
  gr: vec4<f32>,                // the ground: its height (fire-local m), temperature (K), effusivity, on (1/0)
  jk: array<vec4<f32>, 4>,      // per material slot (16): the heat a particle of it holds a kelvin (J/K)
  ab: array<vec4<f32>, 4>,      // per material slot (16): the share of radiant heat it absorbs, and so how well it radiates
  lv: vec4<f32>,                // lava on (1/0, its field on the gas's grid), lava's effusivity, the heat speed, the
                                // simulation's time (s)
};

@group(0) @binding(0) var<storage, read_write> P: array<MParticle>;
@group(0) @binding(1) var<storage, read_write> HG: array<atomic<i32>>;   // per node: weight, weight x temperature
@group(0) @binding(2) var scal: texture_3d<f32>;   // the gas: x = its temperature (0 ambient .. 1 flame)
@group(0) @binding(3) var lin: sampler;
@group(0) @binding(4) var gsdf: texture_3d<f32>;   // the gas's distance to solids (cells; < 0 inside)
@group(0) @binding(5) var LTYPE: texture_3d<f32>;  // the liquid's cells: 0 air, 1 liquid, 2 solid
@group(0) @binding(6) var<storage, read> RL: array<vec4<f32>>;   // the shared radiant sources (rad_common.wgsl)
@group(0) @binding(7) var<storage, read> RLC: array<u32>;
@group(0) @binding(8) var atlas: texture_3d<f32>;   // the meshes' distance fields (meshsdf.wgsl)
@group(0) @binding(9) var<storage, read_write> RA: array<atomic<i32>>;   // what glows, added to them (rad_add.wgsl)
@group(0) @binding(10) var<storage, read_write> OJ: array<atomic<i32>>;  // per object: the heat it gave (J x FX_OJ)
@group(0) @binding(11) var LAVA: texture_3d<f32>;   // lava on the gas's grid: x its temperature (K), y the cell's share
@group(1) @binding(0) var<uniform> U: Params;

//!include rad_common.wgsl
//!include rad_add.wgsl

const FX_HM: f32 = 65536.0;   // weight
const FX_HT: f32 = 256.0;     // weight x temperature (K)
const SIGMA: f32 = 5.670e-8;  // Stefan-Boltzmann (W/m^2/K^4)
const FX_OJ: f32 = 256.0;     // objects' heat (matter.py OBJECT_J)

fn kelvin_of(t: f32) -> f32 { return U.k.y + (U.k.z - U.k.y) * clamp(t, 0.0, 1.0); }

fn thermal(slot: u32) -> bool {
  let h = U.heat[slot];
  let c = U.cond[slot];
  return h.x > 0.0 || h.w > 0.0 || c.x > 0.0 || c.z > 0.0 || U.burn[slot].x > 0.0;
}

@compute @workgroup_size(64, 1, 1)
fn splat(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.mn.w)) { return; }
  let p = P[i];
  if (p.x.w < 0.0) { return; }
  let slot = u32(p.x.w + 0.5);
  if (slot >= 15u || U.cond[slot].x <= 0.0) { return; }
  let n = vec3<i32>(U.mn.xyz);
  let b = vec3<i32>(floor(p.x.xyz));
  let f = p.x.xyz - vec3<f32>(b);
  for (var o = 0; o < 8; o++) {
    let oo = vec3<i32>(o & 1, (o >> 1) & 1, o >> 2);
    let node = b + oo;
    if (any(node < vec3<i32>(0)) || any(node >= n)) { continue; }
    let w3 = select(vec3<f32>(1.0) - f, f, oo == vec3<i32>(1));
    let w = w3.x * w3.y * w3.z;          // (every particle counts alike)
    if (w <= 0.0) { continue; }
    let k = 2u * nidx(node, n);
    atomicAdd(&HG[k], i32(round(w * FX_HM)));
    atomicAdd(&HG[k + 1u], i32(round(w * p.f0.w * FX_HT)));
  }
}

// Where the matter at pos meets the air: (the air's temperature, K; the way out to it, summed; 1 where it meets it).
// In a fire box: the gas cells round it that are not solid (the hottest of them); otherwise the ambient air, where the
// liquid's grid has air next to it.
struct Air { t: f32, out: vec3<f32>, on: f32 };

fn air_at(pos: vec3<f32>) -> Air {
  var a = Air(U.k.y, vec3<f32>(0.0), 0.0);
  if (U.k.w > 0.5) {
    let n = U.g.n.xyz;
    let q = (pos - U.g.org.xyz) / U.g.n.w;
    var best = -1.0;
    for (var s = 1; s < 7; s++) {
      var o = vec3<f32>(0.0);
      o[(s - 1) / 2] = select(-1.0, 1.0, (s & 1) == 0);
      let c = q + o;
      if (any(c < vec3<f32>(0.0)) || any(c >= n)) {
        best = max(best, 0.0);       // (out of the box: the ambient air)
        a.out += o;
        continue;
      }
      if (textureLoad(gsdf, vec3<i32>(floor(c)), 0).x < 0.0) { continue; }
      best = max(best, samp_c(scal, lin, c, n).x);
      a.out += o;
    }
    if (best >= 0.0) {
      a.t = kelvin_of(best);
      a.on = 1.0;
    }
    return a;
  }
  if (U.ln.w > 0.5) {
    let c = vec3<i32>(floor((pos - U.lq.xyz) / U.lq.w));
    let n = vec3<i32>(U.ln.xyz);
    for (var s = 1; s < 7; s++) {
      var o = vec3<i32>(0);
      o[(s - 1) / 2] = select(-1, 1, (s & 1) == 0);
      let q = c + o;
      if (any(q < vec3<i32>(0)) || any(q >= n) || textureLoad(LTYPE, q, 0).x < 0.5) {
        a.out += vec3<f32>(o);
        a.on = 1.0;
      }
    }
    return a;
  }
  a.on = 1.0;
  a.out = vec3<f32>(0.0, 1.0, 0.0);
  return a;
}

// The object pos touches, if any (within a node spacing of its surface: matter resting on it settles some three
// quarters of one off it; the nearest), or else the ground: its temperature (K), its effusivity, 1, which (0..15 an
// object, -1 the ground); or zeros.
fn touching(pos: vec3<f32>) -> vec4<f32> {
  var best = vec4<f32>(0.0);
  var nearest = U.m.w;
  for (var i = 0; i < i32(U.ccnt.x); i++) {
    let d = col_sdf(U.col[i], pos);
    if (d < nearest) {
      nearest = d;
      best = vec4<f32>(U.touch[i].x, U.touch[i].y, 1.0, f32(i));
    }
  }
  if (best.z < 0.5 && U.gr.w > 0.5 && pos.y - U.gr.x < U.m.w) {
    best = vec4<f32>(U.gr.y, U.gr.z, 1.0, -1.0);
  }
  return best;
}

// The temperature (K) of lava touching pos (in its cell or the six beside it), or 0.
fn lava_at(pos: vec3<f32>) -> f32 {
  if (U.lv.x < 0.5) { return 0.0; }
  let n = vec3<i32>(U.g.n.xyz);
  let c = vec3<i32>(floor((pos - U.g.org.xyz) / U.g.n.w));
  var t = 0.0;
  for (var a = 0; a < 7; a++) {
    var o = vec3<i32>(0);
    if (a > 0) { o[(a - 1) / 2] = select(-1, 1, (a & 1) == 0); }
    let q = c + o;
    if (any(q < vec3<i32>(0)) || any(q >= n)) { continue; }
    let l = textureLoad(LAVA, q, 0);
    if (l.y > 0.5) { t = max(t, l.x); }
  }
  return t;
}

// Whether the liquid is next to pos.
fn water_at(pos: vec3<f32>) -> bool {
  if (U.ln.w < 0.5) { return false; }
  let c = vec3<i32>(floor((pos - U.lq.xyz) / U.lq.w));
  let n = vec3<i32>(U.ln.xyz);
  for (var a = 0; a < 7; a++) {
    var o = vec3<i32>(0);
    if (a > 0) { o[(a - 1) / 2] = select(-1, 1, (a & 1) == 0); }
    let q = c + o;
    if (any(q < vec3<i32>(0)) || any(q >= n)) { continue; }
    let t = textureLoad(LTYPE, q, 0).x;
    if (t > 0.5 && t < 1.5) { return true; }
  }
  return false;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.mn.w)) { return; }
  let p = P[i];
  if (p.x.w < 0.0) { return; }
  let slot = u32(p.x.w + 0.5);
  if (slot >= 15u || !thermal(slot)) { return; }
  let dt = U.k.x;
  let h = U.heat[slot];
  let cd = U.cond[slot];
  var T = p.f0.w;
  // its heat evens out with the matter round it
  if (cd.x > 0.0) {
    let n = vec3<i32>(U.mn.xyz);
    let b = vec3<i32>(floor(p.x.xyz));
    let f = p.x.xyz - vec3<f32>(b);
    var ws = 0.0;
    var wt = 0.0;
    for (var o = 0; o < 8; o++) {
      let oo = vec3<i32>(o & 1, (o >> 1) & 1, o >> 2);
      let node = b + oo;
      if (any(node < vec3<i32>(0)) || any(node >= n)) { continue; }
      let w3 = select(vec3<f32>(1.0) - f, f, oo == vec3<i32>(1));
      let w = w3.x * w3.y * w3.z;
      let k = 2u * nidx(node, n);
      ws += w * f32(atomicLoad(&HG[k])) / FX_HM;
      wt += w * f32(atomicLoad(&HG[k + 1u])) / FX_HT;
    }
    if (ws > 1.0e-6) { T += (wt / ws - T) * (1.0 - exp(-cd.x * dt)); }
  }
  // the air (and the fire's radiant heat) and the water where it meets them
  let pos = U.m.xyz + p.x.xyz * U.m.w;
  let air = air_at(pos);
  if (air.on > 0.5) {
    T += (air.t - T) * (1.0 - exp(-h.w * dt));
    if (cd.z > 0.0) {
      // the radiant heat in (the fire's, lava's, hot objects'), its own out (a hot surface radiates as it absorbs:
      // Kirchhoff), into the shared sources for the rest
      var q = SIGMA * (pow(U.k.y, 4.0) - pow(T, 4.0));
      if (U.rad.x > 0.5) { q += rad_irradiance(pos, air.out, -2.0, false); }
      let lo = length(air.out);
      if (T > U.k.y + 30.0 && lo > 1.0e-6) {
        let absorbs = U.ab[slot / 4u][slot % 4u];
        let area = U.rad.z / max(U.rad.y, 1.0e-6);
        rad_add(U.rg, pos, absorbs * SIGMA * (pow(T, 4.0) - pow(U.k.y, 4.0)) * area, air.out / lo);
      }
      T = max(T + cd.z * q * dt, min(T, U.k.y));
    }
  }
  if (water_at(pos)) { T += (U.k.y - T) * (1.0 - exp(-h.w * max(cd.y, 1.0) * dt)); }
  // lava against it: their face at the temperature their effusivities weigh them to, taken on through half a
  // particle's depth
  let tl = lava_at(pos);
  if (tl > 0.0 && cd.x > 0.0) {
    let tc = (cd.w * T + U.lv.y * tl) / max(cd.w + U.lv.y, 1.0e-6);
    T += (tc - T) * (1.0 - exp(-2.0 * cd.x * dt));
  }
  // an object it touches, at its own temperature, or the ground: two bodies in contact, the heat between them flowing
  // as e1 e2 / (e1 + e2) / sqrt(pi t) a square metre and kelvin for the time t they have touched (since it was let go:
  // the skins either side warm and cool, so it falls off), into its surface layer. A steel pan gives chocolate nearly
  // its own heat, a wooden board little; molten iron poured on concrete chills some 80 K in its first second
  if (cd.x > 0.0 && (U.ccnt.x > 0.5 || U.gr.w > 0.5)) {
    let t = touching(pos);
    if (t.z > 0.5) {
      let jk = max(U.jk[slot / 4u][slot % 4u], 1.0e-12);
      let skin = pow(U.rad.z, 1.0 / 3.0);
      let per_k = U.lv.z * skin * skin / jk;                 // K/s for a W/m^2 into its surface (at the heat speed)
      let since = max((U.lv.w - p.c1.w) * U.lv.z, 0.5);     // s of heat it has touched for
      let g = cd.w * t.y / max(cd.w + t.y, 1.0e-6) / sqrt(3.14159265 * since);
      let dT = (t.x - T) * (1.0 - exp(-g * per_k * dt));
      T += dT;
      if (t.w >= 0.0 && U.rad.w > 0.5) {
        atomicAdd(&OJ[u32(t.w)], i32(round(clamp(dT * jk * FX_OJ, -2.0e9, 2.0e9))));
      }
    }
  }
  // past its melting point it melts; below it, it sets (a degree either side, so it does not flicker)
  var next = -1.0;
  if (h.x > 0.0) {
    if (h.y >= 0.0 && T > h.x + 1.0) { next = h.y; }
    if (h.z >= 0.0 && T < h.x - 1.0) { next = h.z; }
  }
  // burning: once lit it burns on while it is hot enough, at its burning temperature where the air reaches it and
  // smouldering slowly inside (a seventh as fast), until it has burnt away (c2.w: how much has)
  let bn = U.burn[slot];
  var burnt = p.c2.w;
  var lit = 0.0;
  if (bn.x > 0.0 && (T > bn.x || (burnt > 0.0 && T > 0.8 * bn.x))) {
    let open = select(0.15, 1.0, air.on > 0.5);
    T += (bn.z - T) * (1.0 - exp(-4.0 * open * dt));
    burnt += bn.y * open * dt;
    lit = open;
    if (burnt >= 1.0) {
      burnt = 0.0;
      next = bn.w;
      lit = 0.0;
      // (most of it is gone: a heap burns down to a little ash; which, by the particle's own random number)
      if (bn.w < 0.0 || p.c0.w >= U.ash[slot / 4u][slot % 4u]) { next = -2.0; }
    }
  }
  P[i].f0 = vec4<f32>(p.f0.xyz, T);
  if (bn.x > 0.0) {
    P[i].c2 = vec4<f32>(p.c2.xyz, burnt);
    P[i].f1 = vec4<f32>(p.f1.xyz, lit);    // (burning where the air reaches it: its flames, mpm_fire.wgsl)
  }
  if (next >= 0.0) { P[i].x = vec4<f32>(p.x.xyz, next); }
  if (next < -1.5) { P[i].x = vec4<f32>(p.x.xyz, -1.0); }
}
