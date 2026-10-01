// Steam bubbles in boiling liquid, carried in the whitewater buffers as their own class (B.w = -radius
// in mm; liq_ww.wgsl leaves them to this kernel).
//
// A bubble rises through the liquid at its terminal speed (about 0.25 m/s for the millimetre to
// centimetre bubbles of boiling water; smaller ones slower), carried by the flow; bubbles over about
// 1 mm wobble on a zigzag. In liquid below boiling it condenses: it shrinks at a rate that grows with
// the subcooling (millimetres in a few hundredths of a second in cold water, the crackle of a kettle
// before it boils) and gives its latent heat back to the liquid (liq_therm_heat.wgsl). At the surface
// it bursts: its steam goes into the air, and the burst throws up a jet drop or two (the spatter over a
// rolling boil). Its volume, per cell, lifts the liquid (liq_therm_buoy.wgsl).
//!include common.wgsl
//!include liq_common.wgsl
//!include noise.wgsl
//!include liq_therm_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,    // capacity, rest density (particles per cell), step seed, gravity (m/s^2)
  th: Therm,
  r: vec4<f32>,    // accumulator regions: gas cells, liquid cells
  e: vec4<f32>,    // gas grid dims
};

@group(0) @binding(0) var<storage, read_write> A: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> B: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> wctr: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> gacc: array<atomic<i32>>;
@group(0) @binding(4) var vnew: texture_3d<f32>;
@group(0) @binding(5) var dens: texture_3d<f32>;
@group(0) @binding(6) var sdf: texture_3d<f32>;
@group(0) @binding(7) var therm_t: texture_3d<f32>;
@group(1) @binding(0) var<uniform> U: Params;

fn gas_index(q: vec3<i32>, n: vec3<i32>) -> u32 {
  let ng = vec3<i32>(U.e.xyz);
  let g = clamp(vec3<i32>(floor((vec3<f32>(q) + vec3<f32>(0.5)) * U.e.xyz / vec3<f32>(n))), vec3<i32>(0), ng - vec3<i32>(1));
  return u32(g.x + ng.x * (g.y + ng.y * g.z));
}

fn steam_mass(r_mm: f32) -> f32 {
  let r = max(r_mm, 0.0) * 1e-3;
  return RHO_VAP * 4.18879 * r * r * r;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.k.x)) { return; }
  var a = A[i];
  if (a.w <= 0.0) { return; }
  var b = B[i];
  if (b.w >= 0.0) { return; }
  let n = gdim(U.g);
  let h = U.g.n.w;
  let dt = U.g.bc.w;
  let x = a.xyz;
  let cc = clamp(vec3<i32>(floor(x)), vec3<i32>(0), n - vec3<i32>(1));
  var r = -b.w;
  let mv0 = steam_mass(r);
  let rho = interp_cell(dens, x, n).x / max(U.k.y, 1e-3);
  let seed = i * 2654435761u + u32(U.k.z) * 97u;

  // at the surface (or in the air): it bursts
  if (rho < 0.35 || a.w - dt <= 0.0) {
    atomicAdd(&gacc[gas_index(cc + vec3<i32>(0, 1, 0), n)], i32(round(min(mv0 * 1000.0 * FX_G, 2.0e9))));
    atomicAdd(&gacc[3u * u32(U.r.x) + 2u * u32(U.r.y) + 7u], i32(round(min(mv0 * 1000.0 * FX_S, 2.0e9))));
    let cap = u32(U.k.x);
    if (r > 0.7 && rand1(seed ^ 0x165667B1u) < min(r / 3.0, 1.0)) {
      // the jet drop a burst throws up: centimetres high
      let slot = atomicAdd(&wctr[0], 1u) % cap;
      let jr = rand3(seed, 5u) - vec3<f32>(0.5);
      let up = sqrt(2.0 * U.k.w * (0.01 + 0.04 * rand1(seed ^ 0xD3A2646Cu)));
      A[slot] = vec4<f32>(x + vec3<f32>(0.0, 0.5, 0.0), 1.2);
      B[slot] = vec4<f32>(mac_vel(vnew, x, n) + vec3<f32>(jr.x * 0.4, up, jr.z * 0.4), 0.0);
    }
    A[i] = vec4<f32>(x, 0.0);
    return;
  }

  // condensing in liquid below boiling, giving its heat back
  let t = textureLoad(therm_t, cc, 0);
  let sub = select(0.0, t_boil(U.th) - t.y, t.w > 1e-3);
  if (sub > 0.3) { r -= 2.5 * sub * dt; }
  let mv1 = steam_mass(r);
  let ri = 3u * u32(U.r.x) + nidx(cc, n);
  let back = (mv0 - mv1) * lv_at(t_boil(U.th)) * 1000.0;
  if (back > 0.0) { atomicAdd(&gacc[ri], i32(round(min(back * FX_R, 2.0e9)))); }
  if (r <= 0.05) {
    A[i] = vec4<f32>(x, 0.0);
    return;
  }

  // rising, carried by the flow, zigzagging
  let fl = mac_vel(vnew, x, n);
  let ut = 0.05 + 0.25 * smoothstep(0.1, 1.0, r) + 0.5 * sqrt(U.k.w * 2e-3 * r) * smoothstep(5.0, 15.0, r);
  var v = fl + vec3<f32>(0.0, ut, 0.0);
  if (r > 0.9) {
    // a spiral a few millimetres across, six turns a second, in its own phase
    let ph = rand1(i * 2654435761u ^ 0x5BD1E995u) * 6.2831853 + U.g.org.w * 6.2831853 * 6.0;
    let amp = 0.3 * ut;
    v += vec3<f32>(cos(ph), 0.0, sin(ph)) * amp;
  }
  var p = x + v * (dt / h);
  let s = interp_cell(sdf, p, n);
  if (s.x < 0.3) {
    let g = s.yzw / max(length(s.yzw), 1e-6);
    p += g * (0.3 - s.x);
  }
  if (U.g.bc.z < 0.5) { p.y = max(p.y, 0.3); }
  p = clamp(p, vec3<f32>(0.01), vec3<f32>(n) - vec3<f32>(0.01));

  // its volume lifts the liquid (next step)
  let pc = clamp(vec3<i32>(floor(p)), vec3<i32>(0), n - vec3<i32>(1));
  let vol = 4.18879 * pow(r * 1e-3, 3.0) / (h * h * h);
  atomicAdd(&gacc[3u * u32(U.r.x) + u32(U.r.y) + nidx(pc, n)], i32(round(min(vol * FX_V, 2.0e9))));
  A[i] = vec4<f32>(p, a.w - dt);
  B[i] = vec4<f32>(v, -r);
}
