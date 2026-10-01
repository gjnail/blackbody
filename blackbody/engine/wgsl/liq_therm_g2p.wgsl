// Heat, pass 5 (after the particles moved): every particle takes its cell's change of enthalpy (as
// FLIP takes the grid's change of velocity, so a hot parcel in cold water stays hot until it mixes),
// mixes a little toward its cell's mean, and changes phase:
//
// - Freezing needs a seed. Water that has none (y < 0) cools below its freezing point as liquid
//   (supercooled) until ice next to it seeds it, it touches a solid, or it is supercooled past what it
//   can hold (Supercooling, and -38 C at most). Ice races into supercooled water as dendrites at about
//   0.16 mm/s x (supercooling in K)^2.5; seeded, the same enthalpy is part ice at once (slush). Ice grown
//   fast, from drops and spray or through supercooled water, is cloudy with trapped air; ice grown slowly
//   in bulk water is clear.
// - Melted through, ice water loses its seed.
// - Above the boiling point the extra heat boils the particle: that share of its mass becomes steam and
//   it is freed with the same probability. Under the surface the steam is born as bubbles (at the cell's
//   nucleation site when a hot wall boils it, so bubbles rise in streams from fixed spots), which carry
//   it up (liq_steam.wgsl); at the surface or in a thin film it goes straight into the air.
// - Evaporation: a particle is freed with the probability of the share its cell evaporated this step.
// - Film boiling (over a surface past the Leidenfrost point) holds the liquid off the surface on its
//   vapour and lets it skitter: drops dance on a very hot pan.
//!include common.wgsl
//!include liq_common.wgsl
//!include noise.wgsl
//!include liq_therm_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,    // slot capacity, whitewater capacity (0: no bubbles), step seed, particle mass (kg)
  th: Therm,
  r: vec4<f32>,    // accumulator regions: gas cells, liquid cells; largest steam bubble (mm), grip on surfaces (m/s^2)
  e: vec4<f32>,    // gas grid dims
  bnd: vec4<f32>,  // the enthalpy of the coldest and the hottest thing the liquid can meet (kJ/kg)
};

@group(0) @binding(0) var<storage, read_write> parts: array<Particle>;
@group(0) @binding(1) var<storage, read_write> therm: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> ctr: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> freelist: array<u32>;
@group(0) @binding(4) var<storage, read_write> WA: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> WB: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> wctr: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> gacc: array<atomic<i32>>;
@group(0) @binding(8) var dh_t: texture_3d<f32>;
@group(0) @binding(9) var therm_t: texture_3d<f32>;
@group(0) @binding(10) var dens: texture_3d<f32>;
@group(0) @binding(11) var sdf: texture_3d<f32>;
@group(0) @binding(12) var<storage, read_write> cellcount: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> U: Params;

// The change of enthalpy at x, with the particle's share of each of the eight nearest cells' change
// taken by the same weights it gave them its mass with (liq_p2g.wgsl), so what the cells gained is
// exactly what their particles gain; and the new temperature of the cells round it (y, -1e9 if none).
fn heat_at(x: vec3<f32>, n: vec3<i32>) -> vec2<f32> {
  let q = x - vec3<f32>(0.5);
  let fq = floor(q);
  let f = q - fq;
  let b = vec3<i32>(fq);
  var dh = 0.0;
  var tm = 0.0;
  var wsum = 0.0;
  for (var o = 0; o < 8; o++) {
    let oo = vec3<i32>(o & 1, (o >> 1) & 1, o >> 2);
    let node = b + oo;
    if (any(node < vec3<i32>(0)) || any(node >= n)) { continue; }
    let t = textureLoad(therm_t, node, 0);
    if (t.w <= 1e-3) { continue; }
    let wv = mix(vec3<f32>(1.0) - f, f, vec3<f32>(oo));
    let w = wv.x * wv.y * wv.z;
    let d = textureLoad(dh_t, node, 0).x;
    dh += w * d;
    tm += w * cell_temp(U.th, t.x + d, t.z);
    wsum += w;
  }
  if (wsum < 1e-6) { return vec2<f32>(0.0, -1.0e9); }
  return vec2<f32>(dh, tm / wsum);
}

// The frozen share of the cells at x, interpolated, with its gradient (per cell).
fn frozen_at(x: vec3<f32>, n: vec3<i32>) -> vec4<f32> {
  let q = x - vec3<f32>(0.5);
  let fq = floor(q);
  let f = q - fq;
  let b = vec3<i32>(fq);
  var val = 0.0;
  var grad = vec3<f32>(0.0);
  for (var o = 0; o < 8; o++) {
    let oo = vec3<i32>(o & 1, (o >> 1) & 1, o >> 2);
    let node = clamp(b + oo, vec3<i32>(0), n - vec3<i32>(1));
    let fo = vec3<f32>(oo);
    let wv = mix(vec3<f32>(1.0) - f, f, fo);
    let sg = fo * 2.0 - vec3<f32>(1.0);
    let a = textureLoad(therm_t, node, 0).z;
    val += wv.x * wv.y * wv.z * a;
    grad += vec3<f32>(sg.x * wv.y * wv.z, wv.x * sg.y * wv.z, wv.x * wv.y * sg.z) * a;
  }
  return vec4<f32>(val, grad);
}

fn ice_near(c: vec3<i32>, n: vec3<i32>) -> f32 {
  var s = textureLoad(therm_t, c, 0).z;
  for (var f = 0; f < 6; f++) {
    var e = vec3<i32>(0);
    e[f / 2] = select(-1, 1, (f & 1) == 1);
    let q = c + e;
    if (in_grid(q, n)) { s = max(s, textureLoad(therm_t, q, 0).z); }
  }
  return s;
}

fn gas_index(q: vec3<i32>, n: vec3<i32>) -> u32 {
  let ng = vec3<i32>(U.e.xyz);
  let g = clamp(vec3<i32>(floor((vec3<f32>(q) + vec3<f32>(0.5)) * U.e.xyz / vec3<f32>(n))), vec3<i32>(0), ng - vec3<i32>(1));
  return u32(g.x + ng.x * (g.y + ng.y * g.z));
}

// steam given off into the air, for the statistics (g, FX_S)
fn count_vapour(kg: f32) {
  let st = 3u * u32(U.r.x) + 2u * u32(U.r.y) + 7u;
  atomicAdd(&gacc[st], i32(round(min(kg * 1000.0 * FX_S, 2.0e9))));
}

fn free_slot(i: u32, P_in: Particle) {
  var P = P_in;
  P.p.w = -1.0;
  parts[i] = P;
  let k = atomicAdd(&ctr[C_FREE], 1);
  freelist[u32(k)] = i;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.k.x)) { return; }
  var P = parts[i];
  if (!alive(P)) { return; }
  let n = gdim(U.g);
  let h = U.g.n.w;
  let dt = U.g.bc.w;
  let speed = max(U.th.a.w, 1e-3);
  let x = P.p.xyz;
  let cc = clamp(vec3<i32>(floor(x)), vec3<i32>(0), n - vec3<i32>(1));
  let seed = i * 2654435761u + u32(U.k.z) * 2246822519u;
  var s = therm[i];
  let phi0 = ice_of(s.x, s.y);
  let tf = t_freeze(U.th);

  // the cell's change, and a little exchange with what is round it: eddies smaller than a cell in water
  // (about 1e-5 m^2/s), but only conduction where there is ice (a solid does not stir, and melts only as
  // fast as heat can reach it). Conduction runs at the heat speed-up; the eddies are the flow's, at its
  // real speed (sped up, they mixed a freezing pond's cold to the bottom in seconds). Heat passes in
  // proportion to the difference in temperature, not in enthalpy: ice and water side by side at the
  // freezing point pass none, so ice does not seep into the water under it as slush
  let hc = heat_at(x, n);
  var hh = s.x;
  if (hc.y > -1.0e8) {
    // the cell's change (a particle colder than the rest of a cooling cell can take it past anything around
    // it: never colder than the coldest thing it can meet, nor hotter than the hottest)
    hh = clamp(hh + hc.x, min(hh, U.bnd.x), max(hh, U.bnd.y));
    let near = ice_near(cc, n);
    let icy = phi0 > 0.0 || near > 0.01;
    // a loose chip of ice (no ice around it) is millimetres across and melts as its own small piece
    let loose = phi0 > 0.0 && near < 0.3;
    let cond = select(select(1.4e-7, 1.4e-7, icy), 2.0e-6, loose) * speed;
    let eddy = select(1.0e-5, 0.0, icy);
    let relax = 1.0 - exp(-dt * 4.0 * (cond + eddy) / (h * h));
    let cp = select(CW, CI, seeded(s.y) && hh <= 0.0);
    hh += relax * cp * (hc.y - temp_of(U.th, hh, s.y));
  }
  let d = textureLoad(dh_t, cc, 0);
  let rho_here = interp_cell(dens, x, n).x / max(U.th.c.x, 1e-3);
  let sparse = 1.0 - smoothstep(0.2, 0.6, rho_here);

  // seeding (freezing needs it), and losing it
  if (!seeded(s.y)) {
    let T = tf + (hh - LF) / CW;
    let sc = tf - T;
    if (sc > 0.0) {
      var rate = 0.0;
      if (ice_near(cc, n) > 0.3) {
        // ice grows into it: dendrites race through supercooled water; barely supercooled water next to
        // ice freezes as its heat is taken
        rate = 1.6e-4 * pow(sc, 2.5) / h + 0.3 * (1.0 - smoothstep(0.5, 3.0, sc));
      }
      // a surface seeds it too, once it is past half what still water holds (scratches, dust, the wall's
      // own roughness give the crystals somewhere to start)
      let sd = interp_cell(sdf, x, n).x;
      let hold = max(U.th.a.z, 0.05);
      if (sd < 1.5 || (U.g.bc.z < 0.5 && x.y < 1.5)) { rate += 2.0 * smoothstep(0.5 * hold, hold, sc); }
      let own = sc >= hold || T <= -38.0;
      if (own || rand1(seed ^ 0x9E3779B9u) < 1.0 - exp(-rate * dt)) {
        // slush from supercooled water is a mass of fine crystals: cloudy
        s.y = clamp(0.15 + 0.75 * smoothstep(0.5, 8.0, sc) + 0.6 * sparse, 0.0, 1.0);
      }
    }
  } else if (hh > LF + 0.5 * CW) {
    s.y = -1.0;
  } else if (phi0 < 0.5 && ice_of(hh, s.y) >= 0.5) {
    // freezing through now: drops and spray freeze to white rime, bulk water to clearer ice
    s.y = max(s.y, clamp(0.12 + 0.85 * sparse, 0.0, 1.0));
  }

  // boiling: under the surface the steam leaves as bubbles of the size water boils off in (a particle
  // holds its extra heat, a few hundredths of a degree, until it has a bubble's worth); at the surface,
  // in a thin film or a drop it goes straight into the air
  let hb = h_boil(U.th);
  let lv = lv_at(t_boil(U.th));
  var gone = false;
  if (hh > hb) {
    let mp = U.k.w;
    var mv = 0.0;                            // kg of steam made this step
    let wcap = u32(U.k.y);
    if (wcap > 0u && rho_here > 0.55) {
      let rd = 0.5 * U.th.d.w;               // departure radius (mm)
      let mb = RHO_VAP * 4.18879 * pow(rd * 1e-3, 3.0);
      let nreal = (hh - hb) * mp / (lv * mb);
      var nb = 0u;
      var r = rd;
      var per = mb;
      if (nreal > 4.0) {
        // boiling hard: bubbles merge as they leave, into bigger ones (and columns of steam)
        nb = 4u;
        r = min(rd * pow(nreal / 4.0, 1.0 / 3.0), U.r.z);
        per = RHO_VAP * 4.18879 * pow(r * 1e-3, 3.0);
      } else {
        nb = u32(floor(nreal + rand1(seed ^ 0x27D4EB2Fu)));
      }
      if (nb > 0u) {
        var site = x;
        if (d.z > 0.0) {
          // the wall's nucleation site in this cell: bubbles rise in streams from the same spots
          let hs = rand3(u32(cc.x) * 73856093u ^ u32(cc.y) * 19349663u ^ u32(cc.z) * 83492791u, 7u);
          site = vec3<f32>(cc) + hs;
        }
        for (var j = 0u; j < nb; j++) {
          let jr = rand3(seed, j + 3u) - vec3<f32>(0.5);
          let slot = atomicAdd(&wctr[0], 1u) % wcap;
          WA[slot] = vec4<f32>(site + jr * 0.15, 8.0);
          WB[slot] = vec4<f32>(P.v * 0.5, -r);
        }
        mv = per * f32(nb);
        hh -= mv * lv / mp;
      }
      if (hh > hb + 8.0 * lv * mb / mp) {
        // more than bubbles this size can carry: the rest straight to the surface
        let extra = (hh - hb) * mp / lv;
        let gi = gas_index(cc + vec3<i32>(0, 1, 0), n);
        atomicAdd(&gacc[gi], i32(round(min(extra * 1000.0 * FX_G, 2.0e9))));
        count_vapour(extra);
        mv += extra;
        hh = hb;
      }
    } else {
      mv = (hh - hb) * mp / lv;
      hh = hb;
      let gi = gas_index(cc + vec3<i32>(0, 1, 0), n);
      atomicAdd(&gacc[gi], i32(round(min(mv * 1000.0 * FX_G, 2.0e9))));
      count_vapour(mv);
    }
    gone = rand1(seed ^ 0x85EBCA6Bu) < mv / mp;
  }
  let st = 3u * u32(U.r.x) + 2u * u32(U.r.y);
  if (gone) { atomicAdd(&gacc[st + 8u], 1); }
  // evaporation
  if (!gone && d.y > 0.0) {
    gone = rand1(seed ^ 0xC2B2AE35u) < d.y;
    if (gone) { atomicAdd(&gacc[st + 9u], 1); }
  }
  if (gone) {
    free_slot(i, P);
    return;
  }

  // ice is solid to the water: water that has run into ice is moved back out of it, into the cell next
  // to it (packed into frozen cells, the pressure solve's volume correction could only move the water,
  // and shot it out in jets). Only into a cell with room (the particles' count, kept up to date as they
  // move): along the foot of a piece of ice on a surface, every particle pushed out lands on the same
  // line, and packed there they were shot out instead. Water with nowhere to go stays, and freezes or
  // melts its way out.
  if (ice_of(hh, s.y) < 0.5) {
    let fz = frozen_at(x, n);
    let gl = length(fz.yzw);
    if (fz.x > 0.55 && gl > 1e-4) {
      let out_n = -fz.yzw / gl;
      let p = clamp(x + out_n * min(fz.x - 0.5, 0.5), vec3<f32>(0.01), vec3<f32>(n) - vec3<f32>(0.01));
      let c0 = nidx(clamp(vec3<i32>(floor(x)), vec3<i32>(0), n - vec3<i32>(1)), n);
      let c1 = nidx(clamp(vec3<i32>(floor(p)), vec3<i32>(0), n - vec3<i32>(1)), n);
      var room = true;
      if (c1 != c0) {
        room = atomicAdd(&cellcount[c1], 1u) < u32(ceil(1.5 * U.th.c.x));
        if (room) { atomicSub(&cellcount[c0], 1u); } else { atomicSub(&cellcount[c1], 1u); }
      }
      if (room) {
        var v = P.v;
        let vn = dot(v, out_n);
        if (vn < 0.0) { v -= vn * out_n; }
        P.p = vec4<f32>(p, P.p.w);
        P.v = v;
        parts[i] = P;
      }
    }
  }

  // film boiling: the drop rides on its vapour and skitters. The vapour layer takes away the grip of the
  // surface (liq_g2p.wgsl slowed the particle by the surface's grip where it touches: that is given back),
  // holds it off the surface, and its uneven jets push it about
  if (d.w > 0.5) {
    let sd = interp_cell(sdf, x, n);
    var sdist = sd.x;
    var nrm = sd.yzw;
    if (U.g.bc.z < 0.5 && x.y < sdist) {
      sdist = x.y;
      nrm = vec3<f32>(0.0, 1.0, 0.0);
    }
    let gl = length(nrm);
    if (gl > 1e-6 && sdist < 2.0) {
      nrm /= gl;
      var v = P.v;
      let vn = dot(v, nrm);
      let vt = v - nrm * vn;
      let st = length(vt);
      let contact = 1.0 - smoothstep(0.25, 1.5, sdist);
      if (st > 1e-5) { v += vt / st * (U.r.w * dt * contact); }
      v += nrm * max(0.12 - vn, 0.0);
      let jr = rand3(seed, 11u) - vec3<f32>(0.5);
      v += (jr - nrm * dot(jr, nrm)) * (6.0 * dt);
      P.v = v;
      parts[i] = P;
    }
  }
  therm[i] = vec2<f32>(hh, s.y);
}
