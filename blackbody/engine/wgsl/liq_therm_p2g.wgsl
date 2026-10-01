// Heat, pass 1: every particle adds its enthalpy and frozen share to the eight nearest cell centres, with the same weights as its density (liq_p2g.wgsl): per cell sum w,
// sum w h, sum w (frozen share).
//
// born (entry point, at the end of a step and after the box moves): particles born since the particles
// last moved (age 0) are given their heat: the temperature of the source that poured them, or of the
// liquid around them (narrow-band reseeds, the open water's strips, a source without its own
// temperature). A source colder than freezing pours ice (snow, crushed ice, an ice cube placed as a
// volume).
//!include common.wgsl
//!include liq_common.wgsl
//!include noise.wgsl
//!include meshsdf.wgsl
//!include emitters.wgsl
//!include liq_therm_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,    // slot capacity
  th: Therm,
  ecnt: vec4<f32>, // source count
  em: array<Emitter, MAX_EMITTERS>,
};

@group(0) @binding(0) var<storage, read> parts: array<Particle>;
@group(0) @binding(1) var<storage, read_write> therm: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> acc: array<atomic<i32>>;
@group(0) @binding(3) var prev: texture_3d<f32>;     // last step's cell heat (liq_therm_norm.wgsl)
@group(0) @binding(4) var atlas: texture_3d<f32>;
@group(1) @binding(0) var<uniform> U: Params;

fn born_heat(x: vec3<f32>, n: vec3<i32>) -> vec2<f32> {
  let h = U.g.n.w;
  let wp = world_of(U.g, x);
  var T = U.th.b.w;
  var found = false;
  var core = 0.5;       // how deep inside its source it was poured (0 at the surface, 1 at the middle)
  let cnt = i32(U.ecnt.x);
  for (var s = 0; s < cnt; s++) {
    let e = U.em[s];
    if (e.g.x < 500.0) { continue; }       // no temperature of its own (liquid.source: 1000 + C)
    if (emitter_mask(e, wp, h) <= 0.0) { continue; }
    T = e.g.x - 1000.0;
    found = true;
    let r = max(min(e.b.x, min(e.b.y, e.b.z)), 1e-4);
    core = clamp(-emitter_sdf(e, wp) / r, 0.0, 1.0);
    break;
  }
  // below freezing it pours ice, unless it is water that can hold that much supercooling
  var frozen = T < t_freeze(U.th) - max(U.th.a.z, 0.0) || T <= -38.0;
  if (!found) {
    let c = clamp(vec3<i32>(floor(x)), vec3<i32>(0), n - vec3<i32>(1));
    let q = textureLoad(prev, c, 0);
    if (q.w > 0.05) {
      // the liquid it was reseeded into, or the water it rose out of
      return vec2<f32>(q.x, select(-1.0, 0.0, q.z > 0.5));
    }
  }
  // poured below freezing it is ice already. Ice frozen in a mould (a freezer's cubes) froze from the
  // outside in, pushing the air ahead of it: clear at the rim, milky in the core where it froze last
  // (the milky core is about a third of the way across; the rest is clear, with a few bubbles)
  let cloud = 0.06 + 0.84 * smoothstep(0.4, 0.8, core);
  return vec2<f32>(h_of_temp(U.th, T, frozen), select(-1.0, cloud, frozen));
}

@compute @workgroup_size(64, 1, 1)
fn born(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.k.x)) { return; }
  let P = parts[i];
  // age 0: born since the particles last moved (every particle that has moved is older)
  if (!alive(P) || P.p.w != 0.0) { return; }
  therm[i] = born_heat(P.p.xyz, gdim(U.g));
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.k.x)) { return; }
  let P = parts[i];
  if (!alive(P)) { return; }
  let n = gdim(U.g);
  let s = therm[i];
  let phi = ice_of(s.x, s.y);
  let q = P.p.xyz - vec3<f32>(0.5);
  let fq = floor(q);
  let f = q - fq;
  let b = vec3<i32>(fq);
  for (var o = 0; o < 8; o++) {
    let oo = vec3<i32>(o & 1, (o >> 1) & 1, o >> 2);
    let node = b + oo;
    if (any(node < vec3<i32>(0)) || any(node >= n)) { continue; }
    let wv = mix(vec3<f32>(1.0) - f, f, vec3<f32>(oo));
    let w = wv.x * wv.y * wv.z;
    if (w <= 1e-7) { continue; }
    let k = nidx(node, n) * 3u;
    atomicAdd(&acc[k], i32(round(w * FX_F)));
    atomicAdd(&acc[k + 1u], i32(round(w * s.x * FX_H)));
    if (phi > 0.0) { atomicAdd(&acc[k + 2u], i32(round(w * phi * FX_F))); }
  }
}
