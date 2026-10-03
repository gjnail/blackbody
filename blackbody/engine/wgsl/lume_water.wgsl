// Lume traces the water (included by stage.wgsl; F_WATER, lume.wgsl lu_path): the liquid's surface (liquid_render.py
// build: its distance field on the surface grid, cut flat at the walls of its box and blended into open water past the
// open sides, as liq_march.wgsl draws it) is a boundary between air and water, refracting and reflecting by the exact
// Fresnel terms, rough by the water's roughness and rippled as the march ripples it. Inside, the water absorbs (its
// colour over its clarity) and its murk scatters, as a medium the paths cross and scatter in. Foam on it is a white
// surface. Light from the lights reaches what is under it through its surface straight, by the Fresnel terms and the
// absorption on the way (lu_shadow), the key light on the floor by the caustic map (liq_caustics.wgsl: the light the
// surface focuses and the shadow the liquid casts, traced through the real surface), in the water's own colour.
//
// Units: fire-local metres outside; the liquid's grid cells inside (w_*_g), as the march has them.

const WATER: i32 = 300;   // a hit on the water's surface (trace)
const MEDIUM: i32 = 301;  // a point in the water's murk where a path scatters (lume.wgsl lu_path)

var<private> g_wcol: vec3<f32> = vec3<f32>(1.0);   // the colour of the water a shadow ray went through, over its grey
                                                   // (lu_shadow returns grey: lu_direct puts the colour back)

fn water_on() -> bool { return F_WATER && U.wat[4].x > 0.5; }
fn w_open() -> bool { return U.wat[4].y > 0.5; }
fn w_h() -> f32 { return U.wat[0].w; }
fn w_n() -> vec3<f32> { return U.wat[1].xyz; }
fn w_grid(p: vec3<f32>) -> vec3<f32> { return (p - U.wat[0].xyz) / U.wat[0].w; }
fn w_sigma() -> vec3<f32> { return U.wat[2].xyz; }   // absorption (1/m)
// scattering (1/m): the look's murk, and at least what clear water scatters (clear lakes 0.05 to 0.3 per metre, a pool's
// a little less: what makes the depths under the surface hazy, not black)
fn w_murk() -> f32 { return max(U.wat[2].w, W_MURK_MIN); }
const W_MURK_MIN: f32 = 0.03;
fn w_ior() -> f32 { return U.wat[3].w; }

// The box with its mirror images (cells), and a point folded back into it across the mirrored sides (liq_march.wgsl).
fn w_mir_lo() -> vec3<f32> {
  let s = U.wat[6];
  return vec3<f32>(select(0.0, -w_n().x, abs(s.x + 1.0) < 0.5), 0.0, select(0.0, -w_n().z, abs(s.z + 1.0) < 0.5));
}
fn w_mir_hi() -> vec3<f32> {
  let s = U.wat[6];
  let n = w_n();
  return vec3<f32>(select(n.x, 2.0 * n.x, abs(s.y + 1.0) < 0.5), n.y, select(n.z, 2.0 * n.z, abs(s.w + 1.0) < 0.5));
}
fn w_mir(p: vec3<f32>) -> vec3<f32> {
  var q = p;
  let lo = w_mir_lo();
  let hi = w_mir_hi();
  let n = w_n();
  if (q.x < 0.0 && lo.x < 0.0) { q.x = -q.x; }
  if (q.x > n.x && hi.x > n.x) { q.x = 2.0 * n.x - q.x; }
  if (q.z < 0.0 && lo.z < 0.0) { q.z = -q.z; }
  if (q.z > n.z && hi.z > n.z) { q.z = 2.0 * n.z - q.z; }
  return q;
}

fn w_box_d(q: vec3<f32>) -> f32 {
  let d = max(-q, q - w_n());
  if (w_open()) { return d.y; }
  return max(d.x, max(d.y, d.z));
}

fn w_side_in(q: vec3<f32>) -> f32 {
  let lo = w_mir_lo();
  let hi = w_mir_hi();
  return min(min(q.x - lo.x, hi.x - q.x), min(q.z - lo.z, hi.z - q.z));
}

fn w_with_level(q: vec3<f32>, s: f32) -> f32 {
  if (!w_open()) { return s; }
  let w = smoothstep(1.5, U.wat[4].w + 1.5, w_side_in(q));
  if (w >= 1.0) { return s; }
  return mix(q.y - U.wat[4].z, s, w);
}

// Signed distance to the water (grid cells; negative in it).
fn w_phi_g(q: vec3<f32>) -> f32 {
  let s = textureSampleLevel(w_surf, lin, w_mir(q) / w_n(), 0.0).x / U.wat[1].w;
  return w_with_level(q, max(s, w_box_d(q)));
}

// The same, smooth (a cubic B-spline of the surface grid, as the march takes the normal from).
fn w_phi_cubic(q_in: vec3<f32>) -> f32 {
  let p = w_mir(q_in);
  let nf = w_n() * U.wat[1].w;
  let coord = p * U.wat[1].w - vec3<f32>(0.5);
  let idx = floor(coord);
  let f = coord - idx;
  let f2 = f * f;
  let f3 = f2 * f;
  let w0 = (vec3<f32>(1.0) - 3.0 * f + 3.0 * f2 - f3) / 6.0;
  let w1 = (vec3<f32>(4.0) - 6.0 * f2 + 3.0 * f3) / 6.0;
  let w2 = (vec3<f32>(1.0) + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
  let w3 = f3 / 6.0;
  let g0 = w0 + w1;
  let g1 = w2 + w3;
  let h0 = (idx - vec3<f32>(0.5) + w1 / g0) / nf;
  let h1 = (idx + vec3<f32>(1.5) + w3 / g1) / nf;
  var s = 0.0;
  for (var j = 0; j < 8; j++) {
    let o = vec3<i32>(j & 1, (j >> 1) & 1, j >> 2);
    let uvw = select(h0, h1, o == vec3<i32>(1));
    let g = select(g0, g1, o == vec3<i32>(1));
    s += g.x * g.y * g.z * textureSampleLevel(w_surf, lin, uvw, 0.0).x;
  }
  return w_with_level(q_in, max(s / U.wat[1].w, w_box_d(q_in)));
}

// Signed distance to the water at fire-local p (m).
fn w_phi(p: vec3<f32>) -> f32 { return w_phi_g(w_grid(p)) * w_h(); }

fn w_inside(p: vec3<f32>) -> bool { return water_on() && w_phi(p) < 0.0; }

// The liquid's velocity at p (m/s).
fn w_vel(p: vec3<f32>) -> vec3<f32> {
  let q = w_grid(p);
  return textureSampleLevel(w_surf, lin, w_mir(q) / w_n(), 0.0).yzw;
}

// The surface's outward normal at fire-local p (into the air), with the ripples the grid cannot hold.
fn w_normal(p: vec3<f32>) -> vec3<f32> {
  let q = w_grid(p);
  let e = 0.6 / U.wat[1].w;
  let g = vec3<f32>(w_phi_cubic(q + vec3<f32>(e, 0.0, 0.0)) - w_phi_cubic(q - vec3<f32>(e, 0.0, 0.0)),
                    w_phi_cubic(q + vec3<f32>(0.0, e, 0.0)) - w_phi_cubic(q - vec3<f32>(0.0, e, 0.0)),
                    w_phi_cubic(q + vec3<f32>(0.0, 0.0, e)) - w_phi_cubic(q - vec3<f32>(0.0, 0.0, e)));
  let l = length(g);
  var nrm = select(vec3<f32>(0.0, 1.0, 0.0), g / l, l > 1e-8);
  // ripples, stronger where the liquid moves fast (liq_march.wgsl ripple), riding the current
  let amt0 = U.wat[5].x;
  if (amt0 > 0.0) {
    let cur = vec3<f32>(U.wat[9].z, 0.0, U.wat[9].w);
    let sp = max(length(w_vel(p)), length(cur));
    let amt = amt0 * clamp(sp / max(U.wat[5].z, 1e-3), 0.0, 1.0);
    if (amt > 0.0) {
      let t = U.wat[5].w;
      let r = (q * w_h() - cur * t) * U.wat[5].y + vec3<f32>(0.37 * t, -0.9 * t, 0.21 * t);
      let gd = gnoise_d(r).yzw + 0.5 * gnoise_d(r * 2.13 + vec3<f32>(13.1, 7.7, 3.3)).yzw;
      let tg = gd - nrm * dot(nrm, gd);
      nrm = normalize(nrm - 0.35 * amt * tg);
    }
  }
  return nrm;
}

// Where a ray crosses the water's surface past t0 and before tmax (m), from the side it starts on to the other: -1 if
// it does not. In the box (and its mirror images) it steps along the distance field and closes in on the crossing; past
// it, over open water, it meets the flat sheet.
fn w_hit(ro: vec3<f32>, rd: vec3<f32>, t0: f32, tmax: f32) -> f32 {
  let h = w_h();
  let lo = w_mir_lo() - vec3<f32>(1.0);
  var hi = w_mir_hi() + vec3<f32>(1.0);
  if (U.wat[11].x >= 0.0) { hi.y = min(hi.y, U.wat[11].x + 1.0); }
  let qo = w_grid(ro);
  let qd = rd / h;
  // the ray's span in the box (m)
  var ta = t0;
  var tb = tmax;
  for (var a = 0; a < 3; a++) {
    if (abs(qd[a]) < 1e-12) {
      if (qo[a] < lo[a] || qo[a] > hi[a]) { tb = -1.0; }
      continue;
    }
    let t1 = (lo[a] - qo[a]) / qd[a];
    let t2 = (hi[a] - qo[a]) / qd[a];
    ta = max(ta, min(t1, t2));
    tb = min(tb, max(t1, t2));
  }
  let start_in = w_phi(ro + rd * t0) < 0.0;
  if (ta < tb) {
    let minstep = 0.35 * h / U.wat[1].w;
    var t = ta;
    var tp = t;
    // (the side the ray is on where it enters the box: inside the water, it looks for where it leaves)
    let s0 = w_phi(ro + rd * t) < 0.0;
    for (var i = 0; i < 256; i++) {
      if (t > tb) { break; }
      let d = w_phi(ro + rd * t);
      if ((d < 0.0) != s0) {
        // closing in on the crossing between tp and t
        var a = tp;
        var b = t;
        for (var k = 0; k < 8; k++) {
          let m = 0.5 * (a + b);
          if ((w_phi(ro + rd * m) < 0.0) == s0) { a = m; } else { b = m; }
        }
        let tc = 0.5 * (a + b);
        if (tc > t0) { return tc; }
      }
      tp = t;
      t += max(abs(d) * 0.8, minstep);
    }
  }
  // open water past the box: the flat sheet at its level
  if (w_open() && abs(rd.y) > 1e-6) {
    let y = U.wat[0].y + U.wat[4].z * h;
    let tl = (y - ro.y) / rd.y;
    if (tl > t0 && tl < tmax && (tl < ta || tl > tb || ta >= tb)) {
      let q = w_grid(ro + rd * tl);
      if (w_side_in(q) < 1.5 && (rd.y < 0.0) != start_in) { return tl; }
    }
  }
  return -1.0;
}

// What a ray meets first: the set (trace), or the water's surface. Only Lume's camera and bounce rays look for the water
// (its shadow rays go through it: lu_shadow, w_shadow), so the water's march is not copied into every trace in the kernel.
fn trace_w(ro: vec3<f32>, rd: vec3<f32>, t0: f32, tmax: f32, want: f32, floor_on: bool) -> Hit {
  var h = trace(ro, rd, t0, tmax, want, floor_on);
  if (water_on()) {
    let tw = w_hit(ro, rd, t0, min(tmax, h.t));
    if (tw > 0.0) { h = Hit(tw, WATER); }
  }
  return h;
}

// The span of a ray (0..tmax, m) that lies in the liquid's box (the whitewater is only there): (in, out), out < in where
// it misses it.
fn w_box_span(ro: vec3<f32>, rd: vec3<f32>, tmax: f32) -> vec2<f32> {
  let qo = w_grid(ro);
  let qd = rd / w_h();
  var ta = 0.0;
  var tb = tmax;
  let n = w_n();
  for (var a = 0; a < 3; a++) {
    if (abs(qd[a]) < 1e-12) {
      if (qo[a] < 0.0 || qo[a] > n[a]) { return vec2<f32>(1.0, -1.0); }
      continue;
    }
    let t1 = (0.0 - qo[a]) / qd[a];
    let t2 = (n[a] - qo[a]) / qd[a];
    ta = max(ta, min(t1, t2));
    tb = min(tb, max(t1, t2));
  }
  return vec2<f32>(ta, tb);
}

// The whitewater a ray goes through (spray in the air, bubbles in the water): its optical depth per metre at p.
fn w_ww_sigma(p: vec3<f32>, in_water: bool) -> f32 {
  let q = w_grid(p);
  if (any(q < vec3<f32>(0.0)) || any(q > w_n())) { return 0.0; }
  let ww = textureSampleLevel(w_ww, lin, q / w_n(), 0.0);
  return select(ww.x * U.wat[7].x, ww.z * U.wat[7].z, in_water) / w_h();
}

// Where along a ray (0..tmax) the whitewater scatters it first, picked by its optical depth (u: a random number, jit: the
// steps' jitter): -1 if it goes through. (Stepped a grid cell at a time, the field's own spacing.)
fn w_ww_event(ro: vec3<f32>, rd: vec3<f32>, tmax: f32, in_water: bool, u: f32, jit: f32) -> f32 {
  if (U.wat[7].w < 0.5 || select(U.wat[7].x, U.wat[7].z, in_water) <= 0.0) { return -1.0; }
  let sp = w_box_span(ro, rd, tmax);
  if (sp.x >= sp.y) { return -1.0; }
  let st = w_h();
  let want = -log(max(1.0 - u, 1e-12));
  var tau = 0.0;
  var t = sp.x + st * jit;
  for (var i = 0; i < 256; i++) {
    if (t > sp.y) { break; }
    let d = w_ww_sigma(ro + rd * t, in_water) * st;
    if (tau + d >= want) { return clamp(t - 0.5 * st + st * (want - tau) / max(d, 1e-9), sp.x, sp.y); }
    tau += d;
    t += st;
  }
  return -1.0;
}

// How much of a shadow ray's light the whitewater on its way lets through (spray in the air, bubbles in the water).
fn w_ww_tr(p: vec3<f32>, d: vec3<f32>, tmax: f32) -> f32 {
  if (U.wat[7].w < 0.5) { return 1.0; }
  let sp = w_box_span(p, d, min(tmax, 1.0e3));
  if (sp.x >= sp.y) { return 1.0; }
  let st = 2.0 * w_h();
  var tau = 0.0;
  var t = sp.x + 0.5 * st;
  for (var i = 0; i < 96; i++) {
    if (t > sp.y || tau > 6.0) { break; }
    let q = p + d * t;
    tau += w_ww_sigma(q, w_phi(q) < 0.0) * st;
    t += st;
  }
  return exp(-tau);
}

// Distances to the nearest and next-nearest of the cell points round p (a Worley pattern).
fn w_worley(p: vec3<f32>) -> vec2<f32> {
  let i = floor(p);
  let f = p - i;
  var d1 = 8.0;
  var d2 = 8.0;
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let c = vec3<f32>(f32(x), f32(y), f32(z));
        let h = vec3<f32>(pcg3d(vec3<u32>(vec3<i32>(i + c) + vec3<i32>(1048576)))) * (1.0 / 4294967296.0);
        let r = c + h - f;
        let d = dot(r, r);
        if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) { d2 = d; }
      }
    }
  }
  return sqrt(vec2<f32>(d1, d2));
}

// Foam on the surface at p: how much of it is covered (0..1). Thick foam is solid white; thin foam breaks into a lace of
// bubble walls stretched along the flow (liq_march.wgsl foam_cover).
fn w_foam(p: vec3<f32>) -> f32 {
  if (U.wat[7].w < 0.5 || U.wat[7].y <= 0.0) { return 0.0; }
  let q = w_mir(w_grid(p));
  if (any(q < vec3<f32>(0.0)) || any(q > w_n())) { return 0.0; }
  let fd = textureSampleLevel(w_ww, lin, q / w_n(), 0.0).y;
  let cov = 1.0 - exp(-fd * U.wat[7].y);
  if (cov <= 0.001 || U.wat[11].y <= 0.0) { return cov; }
  var c = p / max(U.wat[8].w, 1e-4);
  let vel = w_vel(p);
  let sp = length(vel);
  if (sp > 0.05) {
    let a = vel / sp;
    c -= a * dot(c, a) * 0.55;   // (cells stretched along the flow)
  }
  c += vec3<f32>(0.0, U.wat[5].w * 0.15, 0.0);
  let w = w_worley(c);
  let wall = 1.0 - smoothstep(0.03, 0.22, w.y - w.x);   // (1 on the bubble walls)
  let lace = mix(1.0, wall * (0.75 + 0.5 * w.x), U.wat[11].y * (1.0 - smoothstep(0.55, 0.95, cov)));
  return clamp(cov * lace * 1.15, 0.0, 1.0);
}

// What may scatter a path first along a stretch of it (0..tseg, m) that is in the water (in_water) or in the air: the
// murk (in the water), the bubbles in it or the spray over it. (t, what that scattering keeps, rgb); t -1: none (the path
// goes through, its absorption aside). u: two random numbers, jit: a third.
fn w_event(ro: vec3<f32>, rd: vec3<f32>, tseg: f32, in_water: bool, u: vec2<f32>, jit: f32) -> vec4<f32> {
  var tev = 1.0e9;
  var keep = vec3<f32>(1.0);
  if (in_water && w_murk() > 0.0) {
    tev = -log(max(1.0 - u.x, 1e-12)) / w_murk();
    // (the murk's colour: the hue of what it scatters; murk, silt and plankton scatter most of what they stop, so its
    // brightest part keeps 0.95, as spray's and bubbles' white does)
    keep = U.wat[3].xyz * (0.95 / max(max(U.wat[3].x, U.wat[3].y), max(U.wat[3].z, 1e-3)));
  }
  let tw = w_ww_event(ro, rd, min(tseg, tev), in_water, u.y, jit);
  if (tw >= 0.0 && tw < tev) {
    tev = tw;
    keep = vec3<f32>(0.95);   // (spray and bubbles: white)
  }
  if (tev >= tseg) { return vec4<f32>(-1.0, keep); }
  return vec4<f32>(tev, keep);
}

// The surface where a ray hit the water (trace: WATER): its outward normal (rippled), clear, a water's reflectance.
fn w_surface(h: Hit, ro: vec3<f32>, rd: vec3<f32>) -> Surf {
  var s: Surf;
  s.p = ro + rd * h.t;
  s.n = w_normal(s.p);
  s.alb = vec3<f32>(0.0);
  s.f0 = vec3<f32>(0.02);
  s.rough = clamp(U.wat[10].x, 0.0, 1.0);
  let fw = max(U.fit.w * h.t, 1e-5);
  s.eps = max(0.5 * fw, 2.0e-4);
  s.want = 1.0;
  s.wrap = 0.0;
  s.glint = 0.0;
  s.gn = s.n;
  s.em = vec3<f32>(0.0);
  s.self_glow = 0.0;
  return s;
}

// How much light gets from p to p + d * tmax through the water's surfaces (straight: their Fresnel terms, and the
// water's absorption, in grey; g_wcol takes its colour). sun_floor: the key light on the floor, from the caustic map.
fn w_shadow(p: vec3<f32>, d: vec3<f32>, tmax: f32, eps: f32, sun_floor: bool) -> f32 {
  g_wcol = vec3<f32>(1.0);
  if (!water_on()) { return 1.0; }
  var inside = w_phi(p) < 0.0;
  var t = 0.0;
  var trf = 1.0;
  var lw = 0.0;
  for (var k = 0; k < 4; k++) {
    let tc = w_hit(p, d, t + 2.0 * eps, tmax);
    if (tc < 0.0) {
      if (inside) { lw += clamp(tmax - t, 0.0, 50.0); }   // (the rest of the way in it: at most 50 m, past which nothing gets through)
      break;
    }
    if (inside) { lw += tc - t; }
    // (the light comes from the air along d: each crossing let through as the light crossing into the water at d's
    // angle in the air; by reciprocity the same leaving it. At d's angle the water's side would wrongly reflect it all.)
    let n = w_normal(p + d * tc);
    let ci = max(abs(dot(n, d)), 1e-4);
    trf *= 1.0 - lu_fresnel(ci, w_ior());
    if (trf <= 1e-5) { return 0.0; }
    inside = !inside;
    t = tc;
  }
  let col = exp(-w_sigma() * lw);
  let grey = luma(col);
  if (grey <= 1e-8) { return 0.0; }
  g_wcol = col / grey;
  let ww = w_ww_tr(p, d, tmax);   // (spray and bubbles on the way)
  if (sun_floor && U.wat[10].z > 0.5) {
    // the caustic map in place of the straight way: the light the surface focuses here, and the shadow the liquid casts
    // (its Fresnel terms and absorption, in grey); the water's colour over the way the light came
    let q = w_mir(w_grid(p));
    let uv = q.xz / w_n().xz;
    if (all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0))) {
      let c = textureSampleLevel(w_caus, lin, uv, 0.0).x;
      return max(1.0 + (c - 1.0) * U.wat[10].y, 0.0) * ww;
    }
  }
  return trf * grey * ww;
}

// How wet the ground is at fire-local p (0..1), where the liquid ran over it (liq_march.wgsl wet_at), and 0 under it.
fn w_wet(p: vec3<f32>) -> f32 {
  if (!water_on() || U.wat[9].x <= 0.0) { return 0.0; }
  let q = w_grid(p);
  if (w_phi_g(q + vec3<f32>(0.0, 0.6 / U.wat[1].w, 0.0)) <= 0.0) { return 0.0; }
  let n = vec2<i32>(i32(w_n().x), i32(w_n().z));
  let c0 = q.xz - vec2<f32>(0.5);
  let i0 = vec2<i32>(floor(c0));
  let f = c0 - floor(c0);
  var s = 0.0;
  for (var j = 0; j < 4; j++) {
    let o = vec2<i32>(j & 1, j >> 1);
    let c = i0 + o;
    if (any(c < vec2<i32>(0)) || any(c >= n)) { continue; }
    let w = mix(1.0 - f, f, vec2<f32>(o));
    s += w.x * w.y * textureLoad(w_wet_t, c, 0).x;
  }
  let edge = min(min(q.x, w_n().x - q.x), min(q.z, w_n().z - q.z));
  let fw = max(6.0, 0.25 * min(w_n().x, w_n().z));
  let wob = gnoise_d(vec3<f32>(q.x, 3.7, q.z) * (3.0 / fw)).x;
  return clamp(s, 0.0, 1.0) * smoothstep(0.0, fw, edge + wob * 0.35 * fw);
}
