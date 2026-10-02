// Matter's heat (matter.py): wax, chocolate and metal warm in the fire and cool in the air and the water, their heat
// evening out through them, and past their melting point they melt into a liquid of the same stuff (a runny one, or
// chocolate's thick one), which sets again where it cools below it: a candle's top melts into a pool that runs down
// its side and sets in drips, chocolate beside a fire slumps into a puddle, an ingot in a furnace glows and runs.
//  - lights: the fire as a short list of heat sources, for the heat it radiates: the gas's grid cut into blocks, every
//    block of hot gas a source at its centre with the power its gas radiates (4 kappa sigma T^4 a cubic metre, as an
//    optically thin flame does)
//  - splat: each particle's heat to the eight nodes round it (fixed point), for the heat to even out through the matter
//  - main: each particle takes on its neighbourhood's temperature (the nodes'); where it meets the air, the air's (the
//    gas's next to it in a fire box; the ambient air's otherwise), the fire's radiant heat from the side it faces out
//    to, and the heat it radiates itself; where it meets the liquid, the water's; and melts or sets. What burns (dry
//    leaves, sawdust, coal) catches past its ignition point and burns down to ash, held at its burning temperature
//    where the air reaches it and smouldering slowly inside a heap; mpm_fire.wgsl gives the gas its flames.
// A particle's temperature (K) is its f0.w.
//!include common.wgsl
//!include mpm_common.wgsl

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
                                // it than the air, how much a W/m^2 of radiant heat warms its surface (K/s), _
  burn: array<vec4<f32>, 16>,   // per material slot: catches at (K; 0: it does not burn), the share of it that burns away a
                                // second, how hot it burns (K), what it burns down to (slot, -1: nothing)
  ash: array<vec4<f32>, 4>,     // per material slot (16): the share of it left as that once burnt (the rest is gone)
};

@group(0) @binding(0) var<storage, read_write> P: array<MParticle>;
@group(0) @binding(1) var<storage, read_write> HG: array<atomic<i32>>;   // per node: weight, weight x temperature
@group(0) @binding(2) var scal: texture_3d<f32>;   // the gas: x = its temperature (0 ambient .. 1 flame)
@group(0) @binding(3) var lin: sampler;
@group(0) @binding(4) var gsdf: texture_3d<f32>;   // the gas's distance to solids (cells; < 0 inside)
@group(0) @binding(5) var LTYPE: texture_3d<f32>;  // the liquid's cells: 0 air, 1 liquid, 2 solid
@group(0) @binding(6) var<storage, read_write> HL: array<vec4<f32>>;      // heat sources: (position, _), (power W, ...)
@group(0) @binding(7) var<storage, read_write> HLC: array<atomic<u32>>;   // how many
@group(1) @binding(0) var<uniform> U: Params;

const FX_HM: f32 = 65536.0;   // weight
const FX_HT: f32 = 256.0;     // weight x temperature (K)
const SIGMA: f32 = 5.670e-8;  // Stefan-Boltzmann (W/m^2/K^4)
const PI4: f32 = 12.566371;

fn kelvin_of(t: f32) -> f32 { return U.k.y + (U.k.z - U.k.y) * clamp(t, 0.0, 1.0); }

fn thermal(slot: u32) -> bool {
  let h = U.heat[slot];
  let c = U.cond[slot];
  return h.x > 0.0 || h.w > 0.0 || c.x > 0.0 || c.z > 0.0 || U.burn[slot].x > 0.0;
}

@compute @workgroup_size(4, 4, 4)
fn lights(@builtin(global_invocation_id) id: vec3<u32>) {
  let b = i32(U.lt.x);
  let dims = vec3<i32>(U.g.n.xyz);
  let lo = vec3<i32>(id) * b;
  if (any(lo >= dims)) { return; }
  let hi = min(lo + vec3<i32>(b), dims);
  let h = U.g.n.w;
  let amb4 = pow(U.k.y, 4.0);
  var power = 0.0;
  var centre = vec3<f32>(0.0);
  for (var z = lo.z; z < hi.z; z++) {
    for (var y = lo.y; y < hi.y; y++) {
      for (var x = lo.x; x < hi.x; x++) {
        let t = textureLoad(scal, vec3<i32>(x, y, z), 0).x;
        if (t <= U.lt.y) { continue; }
        let e = max(pow(kelvin_of(t), 4.0) - amb4, 0.0);
        power += e;
        centre += (vec3<f32>(f32(x), f32(y), f32(z)) + 0.5) * e;
      }
    }
  }
  if (power <= 0.0) { return; }
  let i = atomicAdd(&HLC[0], 1u);
  if (i >= u32(U.lt.w)) { return; }
  HL[2u * i] = vec4<f32>(U.g.org.xyz + centre / power * h, 0.5 * f32(b) * h);
  HL[2u * i + 1u] = vec4<f32>(4.0 * U.lt.z * SIGMA * power * h * h * h, 0.0, 0.0, 0.0);
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

// The fire's radiant heat on a surface at pos facing out along `out` (W/m^2): every heat source on that side, by the
// inverse square of its distance and the cosine of its angle.
fn radiant(pos: vec3<f32>, out: vec3<f32>) -> f32 {
  let lo = length(out);
  if (lo < 1.0e-6) { return 0.0; }
  let nrm = out / lo;
  let cnt = min(atomicLoad(&HLC[0]), u32(U.lt.w));
  var q = 0.0;
  for (var i = 0u; i < cnt; i++) {
    let a = HL[2u * i];
    let d = a.xyz - pos;
    let r2 = max(dot(d, d), a.w * a.w);
    let c = dot(d, nrm) * inverseSqrt(r2);
    if (c <= 0.0) { continue; }
    q += HL[2u * i + 1u].x * c / (PI4 * r2);
  }
  return q;
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
      // the fire's radiant heat in, its own out (a hot surface radiates as it absorbs: Kirchhoff)
      var q = SIGMA * (pow(U.k.y, 4.0) - pow(T, 4.0));
      if (U.k.w > 0.5) { q += radiant(pos, air.out); }
      T = max(T + cd.z * q * dt, min(T, U.k.y));
    }
  }
  if (water_at(pos)) { T += (U.k.y - T) * (1.0 - exp(-h.w * max(cd.y, 1.0) * dt)); }
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
