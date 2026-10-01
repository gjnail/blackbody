// Weather: precipitation particles (engine/weather.py), one thread per slot, each substep.
//
// A free slot may become a new particle at the top of the weather area (as many as the rate brings in
// this step), its state drawn from what the air column above delivers there (atmos.arrivals, uploaded as
// a table). A falling particle exchanges heat and water with the air round it (it melts, refreezes,
// evaporates or sublimates: wx_common.wgsl), is carried by the wind and its gusts and eddies toward its
// terminal fall (snow flutters as it falls), and moves in steps short enough to find what it hits: the
// ground, a collider, or the liquid's surface. The area wraps round at its sides, so the wind carries the
// fall through it without thinning it upwind.
//
// What it does when it lands depends on what it is:
//  - in the liquid: its mass and heat go to the liquid cell it entered (wx_liquid.wgsl turns them into
//    liquid particles carrying that heat: snow melts into the water, cooling it, or floats as slush);
//  - on a surface: snow settles into the cover (its mass to the cover map, wx_cover.wgsl); graupel and
//    ice pellets bounce a little, then settle into it too; hail bounces, then rests where it stops, a
//    stone on the ground that melts there; a drop wets the surface, or freezes onto it as glaze if it is
//    supercooled or the surface is below freezing (freezing rain).
//!include common.wgsl
//!include liq_common.wgsl
//!include noise.wgsl
//!include wx_common.wgsl

struct WxP {
  p: vec4<f32>,   // position (sim-local m), age (s); age < 0 marks a free slot
  v: vec4<f32>,   // velocity (m/s); w: 1 while it rests on a surface
  m: vec4<f32>,   // ice (kg), water (kg), temperature (C), dry size (mm)
  k: vec4<f32>,   // kind, nucleated (0/1), most ever melted, phase (0..1, random: flutter and looks)
};

struct Params {
  g: Grid,          // the liquid's grid (its distance field and density are read where they cover)
  area: vec4<f32>,  // x0, z0, x1, z1 of the weather area (m)
  area2: vec4<f32>, // top (m), ground on (1/0), cover map nx, nz
  air: vec4<f32>,   // air temperature at the ground (C), lapse rate (K/m), relative humidity, pressure (Pa)
  wind: vec4<f32>,  // wind (m/s), gustiness (0..1)
  turb: vec4<f32>,  // eddies (m/s), their size (m), time (s), step seed
  spawn: vec4<f32>, // particles to start this step, table entries, liquid surface density, liquid on (1/0)
  phys: vec4<f32>,  // bounce of hail, of graupel and pellets, surface contact (W/(m^2 K)), fill (1: start anywhere)
  liq: vec4<f32>,   // freezing point (C), mass of a liquid particle (kg), slowest fall in the table (m/s), spray capacity
  ww: vec4<f32>,    // spray life (s), _, _, _
};

@group(0) @binding(0) var<storage, read_write> parts: array<WxP>;
@group(0) @binding(1) var<storage, read> lut: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> ctr: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> cover_dep: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> liq_dep: array<atomic<i32>>;
@group(0) @binding(5) var sdf: texture_3d<f32>;
@group(0) @binding(6) var dens: texture_3d<f32>;
@group(0) @binding(7) var surf: texture_2d<f32>;
@group(0) @binding(8) var<storage, read_write> WA: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> WB: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read_write> wctr: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> U: Params;

// counters
const C_SPAWNED: u32 = 0u;
const C_ALIVE: u32 = 1u;
const C_LANDED: u32 = 2u;     // + kind: landed this step, by what it was when it landed
const C_WATER: u32 = 8u;      // fell into the liquid
// cover deposits per map cell (fixed point, micrograms): snow, water, glaze, packed ice (graupel, pellets)
const DEP_SNOW: u32 = 0u;
const DEP_WATER: u32 = 1u;
const DEP_GLAZE: u32 = 2u;
const DEP_GRAIN: u32 = 3u;
const FX_UG: f32 = 1.0e9;     // kg -> micrograms
// liquid deposits per liquid cell: mass (micrograms), mass times enthalpy (ug * kJ/kg / 16), momentum (y)
const FX_HM: f32 = 0.0625;

fn state_of(P: WxP) -> WxState {
  return WxState(u32(P.k.x + 0.5), P.m.x, P.m.y, P.m.z, P.m.w, P.k.y > 0.5, P.k.z);
}

fn put_state(P: ptr<function, WxP>, s: WxState) {
  (*P).m = vec4<f32>(s.ice, s.water, s.t, s.d0);
  (*P).k = vec4<f32>(f32(s.kind), select(0.0, 1.0, s.nucleated), s.fmax, (*P).k.w);
}

fn air_t(y: f32) -> f32 { return U.air.x - U.air.y * max(y, 0.0); }
fn air_p(y: f32) -> f32 { return U.air.w * exp(-y / 8400.0); }

// The wind where p is: the mean wind, slow gusts, and eddies (curl noise, so they stir without piling up).
fn wind_at(p: vec3<f32>, seed: f32) -> vec3<f32> {
  let t = U.turb.z;
  var w = U.wind.xyz;
  let gust = 1.0 + U.wind.w * (0.6 * sin(t * 0.37 + p.x * 0.05) + 0.4 * sin(t * 0.91 + p.z * 0.07 + 1.3));
  w *= gust;
  if (U.turb.x > 0.0) {
    let sc = max(U.turb.y, 0.05);
    let q = p / sc - U.wind.xyz * t / sc + vec3<f32>(0.0, t * 0.1, 0.0);
    w += U.turb.x * curl_noise(q);
  }
  return w;
}

// The liquid's grid at sim-local p (m): its cell coordinates.
fn cell_pos(p: vec3<f32>) -> vec3<f32> { return (p - U.g.org.xyz) / U.g.n.w; }
fn in_box(x: vec3<f32>) -> bool {
  let n = vec3<f32>(gdim(U.g));
  return all(x > vec3<f32>(0.5)) && all(x < n - vec3<f32>(0.5));
}

// The surface map (wx_surf.wgsl) at (x, z), bilinear: height, temperature, normal x, z.
fn surface_lin(p: vec3<f32>) -> vec4<f32> {
  let nx = i32(U.area2.z);
  let nz = i32(U.area2.w);
  let u = clamp((p.x - U.area.x) / max(U.area.z - U.area.x, 1e-6), 0.0, 1.0) * f32(nx) - 0.5;
  let v = clamp((p.z - U.area.y) / max(U.area.w - U.area.y, 1e-6), 0.0, 1.0) * f32(nz) - 0.5;
  let i = vec2<i32>(floor(vec2<f32>(u, v)));
  let f = vec2<f32>(u, v) - floor(vec2<f32>(u, v));
  let hi = vec2<i32>(nx - 1, nz - 1);
  let a = textureLoad(surf, clamp(i, vec2<i32>(0), hi), 0);
  let b = textureLoad(surf, clamp(i + vec2<i32>(1, 0), vec2<i32>(0), hi), 0);
  let c = textureLoad(surf, clamp(i + vec2<i32>(0, 1), vec2<i32>(0), hi), 0);
  let d = textureLoad(surf, clamp(i + vec2<i32>(1, 1), vec2<i32>(0), hi), 0);
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// What is at p: 0 air, 1 solid (a collider or the ground), 2 liquid. Fills the surface normal. Inside the
// liquid's box from its own distance field and density; past it, from the surface map (the colliders' tops
// and the ground, as wx_surf.wgsl found them from above).
fn probe(p: vec3<f32>, r: f32, nrm: ptr<function, vec3<f32>>) -> i32 {
  if (U.area2.y > 0.5 && p.y <= r) {
    *nrm = vec3<f32>(0.0, 1.0, 0.0);
    return 1;
  }
  if (U.spawn.w > 0.5) {
    let x = cell_pos(p);
    if (in_box(x)) {
      let n = gdim(U.g);
      let sd = interp_cell(sdf, x, n);
      if (sd.x * U.g.n.w < r) {
        let gl = length(sd.yzw);
        *nrm = select(vec3<f32>(0.0, 1.0, 0.0), sd.yzw / max(gl, 1e-6), gl > 1e-6);
        return 1;
      }
      if (interp_cell(dens, x, n).x > U.spawn.z) {
        *nrm = vec3<f32>(0.0, 1.0, 0.0);
        return 2;
      }
      return 0;
    }
  }
  let sf = surface_lin(p);
  if (sf.x > -100.0 && p.y - r <= sf.x) {
    *nrm = normalize(vec3<f32>(sf.z, sqrt(max(1.0 - dot(sf.zw, sf.zw), 0.0)), sf.w));
    return 1;
  }
  return 0;
}

fn map_index(p: vec3<f32>) -> i32 {
  let nx = i32(U.area2.z);
  let nz = i32(U.area2.w);
  let u = (p.x - U.area.x) / max(U.area.z - U.area.x, 1e-6);
  let v = (p.z - U.area.y) / max(U.area.w - U.area.y, 1e-6);
  if (u < 0.0 || u >= 1.0 || v < 0.0 || v >= 1.0) { return -1; }
  return i32(v * f32(nz)) * nx + i32(u * f32(nx));
}

fn surface_at(p: vec3<f32>) -> vec4<f32> {
  let nx = i32(U.area2.z);
  let nz = i32(U.area2.w);
  let u = clamp((p.x - U.area.x) / max(U.area.z - U.area.x, 1e-6), 0.0, 0.9999);
  let v = clamp((p.z - U.area.y) / max(U.area.w - U.area.y, 1e-6), 0.0, 0.9999);
  return textureLoad(surf, vec2<i32>(i32(u * f32(nx)), i32(v * f32(nz))), 0);
}

fn deposit(p: vec3<f32>, slot: u32, kg: f32) {
  let i = map_index(p);
  if (i < 0 || kg <= 0.0) { return; }
  atomicAdd(&cover_dep[u32(i) * 4u + slot], u32(kg * FX_UG + 0.5));
}

// The particle's enthalpy as the liquid's heat model counts it (kJ/kg from ice at the freezing point).
fn enthalpy(s: WxState) -> f32 {
  let m = max(wx_mass(s), 1e-30);
  let tf = U.liq.x;
  if (s.ice > 0.0 && s.water > 0.0) { return s.water / m * WX_LF * 1e-3; }
  if (s.ice > 0.0) { return WX_CI * 1e-3 * (s.t - tf); }
  return WX_LF * 1e-3 + WX_CW * 1e-3 * (s.t - tf);
}

fn into_liquid(p: vec3<f32>, s: WxState, vy: f32) {
  let n = gdim(U.g);
  let c = clamp(vec3<i32>(floor(cell_pos(p))), vec3<i32>(0), n - vec3<i32>(1));
  let i = nidx(c, n) * 4u;
  let ug = wx_mass(s) * FX_UG;
  atomicAdd(&liq_dep[i], i32(ug + 0.5));
  atomicAdd(&liq_dep[i + 1u], i32(round(ug * enthalpy(s) * FX_HM)));
  atomicAdd(&liq_dep[i + 2u], i32(round(ug * vy * 0.01)));
  atomicAdd(&liq_dep[i + 3u], select(0, 1, s.ice > 0.0));
}

// A crown of spray where it struck the water, as big as what struck it and as fast as it came (a hailstone
// throws up a splash; a snowflake none).
fn splash(p: vec3<f32>, s: WxState, v: vec3<f32>, seed: u32) {
  let cap = u32(U.liq.w);
  if (cap == 0u) { return; }
  let d = wx_diameter(s);
  let speed = length(v);
  let cnt = min(u32(max(d - 1.5, 0.0) * 1.5 * min(speed / 5.0, 2.0)), 48u);
  if (cnt == 0u) { return; }
  let at = cell_pos(p) + vec3<f32>(0.0, 0.6, 0.0);
  for (var j = 0u; j < cnt; j++) {
    let r = rand3(seed, j * 7u + 101u);
    let ang = 6.2831853 * (f32(j) + r.x) / f32(cnt);
    let out = vec3<f32>(cos(ang), 0.0, sin(ang));
    let vel = out * (0.08 * speed * (0.5 + r.y)) + vec3<f32>(0.0, 0.18 * speed * (0.4 + r.z), 0.0);
    let slot = atomicAdd(&wctr[0], 1u) % cap;
    WA[slot] = vec4<f32>(at + out * 0.2, U.ww.x * (0.6 + 0.8 * r.z));
    WB[slot] = vec4<f32>(vel, 0.0);
  }
}

fn free_slot(i: u32) {
  var P = parts[i];
  P.p.w = -1.0;
  parts[i] = P;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= arrayLength(&parts)) { return; }
  var P = parts[i];
  let dt = U.g.bc.w;
  let seed = i * 2654435761u + u32(U.turb.w) * 2246822519u;

  // a free slot: maybe a new particle, at the top of the area (anywhere in it when filling at the start)
  if (P.p.w < 0.0) {
    let want = u32(U.spawn.x);
    if (want == 0u || atomicLoad(&ctr[C_SPAWNED]) >= want) { return; }
    if (atomicAdd(&ctr[C_SPAWNED], 1u) >= want) { return; }
    let nlut = max(u32(U.spawn.y), 1u);
    var j = min(u32(rand1(seed ^ 0x9E3779B9u) * f32(nlut)), nlut - 1u);
    if (U.phys.w > 0.5) {
      // filling the air: as many of each as hang in it, not as cross a level (slow ones linger): pick by
      // rejection against the slowest fall
      for (var t = 0u; t < 16u; t++) {
        let vj = lut[j * 2u + 1u].w;
        if (rand1(seed ^ (0x68E31DA4u + t)) * vj <= U.liq.z) { break; }
        j = min(u32(rand1(seed ^ (0x1B56C4E9u + t * 7u)) * f32(nlut)), nlut - 1u);
      }
    }
    let a = lut[j * 2u];
    let b = lut[j * 2u + 1u];
    let r = rand3(seed, 17u);
    var y = U.area2.x - r.y * b.w * dt;
    if (U.phys.w > 0.5) { y = U.area2.x * r.y; }
    let x = mix(U.area.x, U.area.z, r.x);
    let z = mix(U.area.y, U.area.w, r.z);
    var Q: WxP;
    Q.p = vec4<f32>(x, y, z, 0.0);
    Q.v = vec4<f32>(wind_at(vec3<f32>(x, y, z), 0.0) + vec3<f32>(0.0, -b.w, 0.0), 0.0);
    Q.m = vec4<f32>(a.y, a.z, a.w, b.x);
    Q.k = vec4<f32>(a.x, b.y, b.z, rand1(seed ^ 0x51ED270Bu));
    var nrm = vec3<f32>(0.0);
    if (probe(Q.p.xyz, 0.0, &nrm) != 0) { return; }   // (started inside something: skip it)
    parts[i] = Q;
    return;
  }
  atomicAdd(&ctr[C_ALIVE], 1u);
  P.p.w += dt;
  var s = state_of(P);
  var p = P.p.xyz;
  let ta = air_t(p.y);
  let pa = air_p(p.y);
  let rv = wx_vapour(ta, U.air.z);
  let resting = P.v.w > 0.5;

  // heat and water with the air (and, resting, with the surface under it)
  var extra = 0.0;
  var su = vec4<f32>(0.0);
  if (resting) {
    su = surface_at(p);
    let dd = wx_diameter(s) * 1e-3;
    extra = U.phys.z * 0.785 * dd * dd * (su.y - s.t);
  }
  wx_exchange(&s, ta, rv, pa, dt, extra, rand1(seed ^ 0x2C1B3C6Du));
  if (wx_mass(s) < 1e-12) {
    free_slot(i);
    return;
  }
  if (resting) {
    // a stone resting on the ground melts there; when it has melted it is a wet patch
    if (s.ice <= 0.0) {
      deposit(p, DEP_WATER, wx_mass(s));
      free_slot(i);
      return;
    }
    // still held up by what it rests on?
    var nrm = vec3<f32>(0.0);
    let r = wx_diameter(s) * 0.5e-3;
    if (probe(p - vec3<f32>(0.0, 0.01 + r, 0.0), 0.0, &nrm) == 1) {
      put_state(&P, s);
      parts[i] = P;
      return;
    }
    P.v.w = 0.0;   // what held it melted or moved: it falls again
  }

  // motion: toward the terminal fall in the moving air
  let rho = wx_air_density(ta, pa);
  let vt = max(wx_speed(s, rho), 0.05);
  var air = wind_at(p, P.k.w);
  // snowflakes and graupel flutter and spiral as they fall (aggregates tumble about once a second)
  if (s.kind == WX_SNOW && s.ice > 0.0) {
    let ph = P.k.w * 6.2831853;
    let w = 3.0 + 4.0 * fract(P.k.w * 7.31);
    let amp = vt * (0.25 + 0.35 * fract(P.k.w * 3.7)) * (1.0 - wx_melted(s));
    air += amp * vec3<f32>(cos(w * P.p.w + ph), 0.0, sin(w * 0.83 * P.p.w + ph * 1.7));
  }
  let goal = air + vec3<f32>(0.0, -vt, 0.0);
  let tau = vt / WX_G;
  var v = goal + (P.v.xyz - goal) * exp(-dt / max(tau, 1e-4));

  // move in steps no longer than the particle's own size or half a cell, finding what it meets
  let travel = v * dt;
  let dist = length(travel);
  let r = wx_diameter(s) * 0.5e-3;
  var stepl = max(0.5 * U.g.n.w, 0.02);
  if (U.spawn.w < 0.5) { stepl = 0.25; }
  let nsteps = clamp(u32(ceil(dist / stepl)), 1u, 64u);
  var hit = 0;
  var nrm = vec3<f32>(0.0, 1.0, 0.0);
  var q = p;
  for (var k = 1u; k <= nsteps; k++) {
    let q1 = p + travel * (f32(k) / f32(nsteps));
    hit = probe(q1, r, &nrm);
    if (hit != 0) { break; }
    q = q1;
  }
  if (hit == 0) {
    q = p + travel;
    // wrap round at the sides of the area
    let wx = U.area.z - U.area.x;
    let wz = U.area.w - U.area.y;
    if (q.x < U.area.x) { q.x += wx; }
    if (q.x >= U.area.z) { q.x -= wx; }
    if (q.z < U.area.y) { q.z += wz; }
    if (q.z >= U.area.w) { q.z -= wz; }
    if (q.y > U.area2.x + 1.0) { q.y = U.area2.x; }
    if (q.y < -50.0) {
      free_slot(i);
      return;
    }
    P.p = vec4<f32>(q, P.p.w);
    P.v = vec4<f32>(v, 0.0);
    put_state(&P, s);
    parts[i] = P;
    return;
  }

  // it has hit something at q
  let kind_now = select(s.kind, WX_RAIN, s.ice <= 0.0);
  atomicAdd(&ctr[C_LANDED + min(kind_now, 4u)], 1u);
  if (hit == 2) {
    atomicAdd(&ctr[C_WATER], 1u);
    into_liquid(q, s, v.y);
    splash(q, s, v, seed);
    free_slot(i);
    return;
  }
  let sfc = surface_at(q);
  let on_top = q.y >= sfc.x - 0.05;
  if (s.ice <= 0.0) {
    // a drop: freezes on contact where it is supercooled or the surface is below freezing
    if (on_top) {
      if (s.t < 0.0 || sfc.y <= U.liq.x) { deposit(q, DEP_GLAZE, wx_mass(s)); }
      else { deposit(q, DEP_WATER, wx_mass(s)); }
    }
    free_slot(i);
    return;
  }
  if (s.kind == WX_SNOW) {
    if (on_top) { deposit(q, DEP_SNOW, s.ice); deposit(q, DEP_WATER, s.water); }
    free_slot(i);
    return;
  }
  // graupel, pellets and hail bounce
  let e = select(U.phys.y, U.phys.x, s.kind == WX_HAIL);
  let vn = dot(v, nrm);
  var vb = v;
  if (vn < 0.0) { vb = v - (1.0 + e) * vn * nrm; }
  vb -= 0.3 * (vb - dot(vb, nrm) * nrm);
  let settle = length(vb) < 0.6 || e <= 0.0;
  if (settle) {
    if (s.kind == WX_HAIL) {
      P.p = vec4<f32>(q + nrm * r, P.p.w);
      P.v = vec4<f32>(0.0, 0.0, 0.0, 1.0);
      put_state(&P, s);
      parts[i] = P;
      return;
    }
    if (on_top) { deposit(q, DEP_GRAIN, s.ice); deposit(q, DEP_WATER, s.water); }
    free_slot(i);
    return;
  }
  P.p = vec4<f32>(q + nrm * (r + 0.002), P.p.w);
  P.v = vec4<f32>(vb, 0.0);
  put_state(&P, s);
  parts[i] = P;
}
