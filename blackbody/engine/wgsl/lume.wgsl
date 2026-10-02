// Lume: the light on the stage traced as it really travels (stage.wgsl takes this in place of its direct shading when
// Lighting engine is Lume).
//
// From each surface the camera sees, a path of light is followed back toward where it came from, bounce by bounce:
// - at every bounce, the light that comes straight from each kind of light (next-event estimation): the key light,
//   picked within the sun's disc (so shadows are as soft as the sun is wide); one of the fire's point lights and one of
//   the hot matter's, picked by how much light each brings here, toward a point within it (so each flame casts its own
//   soft shadow, through the smoke); every lamp in the set; and the HDRI, picked by its brightness (a small bright sun
//   in it is found at once), weighed against finding it by bouncing (multiple importance sampling);
// - then the path bounces on: off the surface as its material scatters light (diffuse, or a GGX highlight sampled by
//   its visible normals), or through glass, ice and jelly (reflected or refracted by its Fresnel term, tinted over its
//   path through it), so light reaches the camera after bouncing from wall to floor to object;
// - a path that leaves the set sees the sky (or the HDRI), through the smoke overhead.
// Paths end by Russian roulette once they carry little light, and indirect light is capped (Clamp bright paths) so a
// stray glint does not leave a firefly.
//
// Every pass traces one path per pixel at a new place in the pixel and the shutter; the stage adds the passes up
// (ACC) with the first surface's albedo, facing and distance, and each pass's brightness squared (AOV), for the denoiser
// (lume_denoise.wgsl).

const LU_INV_PI: f32 = 0.31830988;

var<private> lu_rng: u32;
var<private> lu_alb: vec3<f32>;    // the first surface's albedo (diffuse + highlight): the denoiser's guide
var<private> lu_nrm: vec3<f32>;    // its normal (0: the camera saw no surface)
var<private> lu_dist: f32;         // its distance from the camera (m)

fn lu_seed(px: vec2<i32>, i: u32) {
  lu_rng = pcg1(u32(px.x) * 1973u + u32(px.y) * 9277u + u32(U.lume.y + 0.5) * 26699u + i * 104729u + 7u);
}

// A PCG stream (O'Neill): the state steps as a linear congruential generator and each number is its permuted output, so the
// numbers are evenly spread in every dimension (hashing the last output instead clusters them); 24 bits, in [0, 1).
fn lu_rand() -> f32 {
  lu_rng = lu_rng * 747796405u + 2891336453u;
  var w = ((lu_rng >> ((lu_rng >> 28u) + 4u)) ^ lu_rng) * 277803737u;
  w = (w >> 22u) ^ w;
  return f32(w >> 8u) * (1.0 / 16777216.0);
}

fn lu_rand2() -> vec2<f32> {
  let a = lu_rand();
  return vec2<f32>(a, lu_rand());
}

// An orthonormal frame round n (Duff et al. 2017): (t, b, n).
fn lu_frame(n: vec3<f32>) -> mat3x3<f32> {
  let s = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (s + n.z);
  let b = n.x * n.y * a;
  return mat3x3<f32>(vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x), vec3<f32>(b, s + n.y * n.y * a, -n.y), n);
}

fn lu_to_local(f: mat3x3<f32>, v: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(dot(v, f[0]), dot(v, f[1]), dot(v, f[2]));
}

fn lu_cosine(u: vec2<f32>) -> vec3<f32> {
  let r = sqrt(u.x);
  let ph = 2.0 * PI * u.y;
  return vec3<f32>(r * cos(ph), r * sin(ph), sqrt(max(1.0 - u.x, 0.0)));
}

// A direction in the cone of half-angle acos(cmax) round the local z axis, uniformly.
fn lu_cone(u: vec2<f32>, cmax: f32) -> vec3<f32> {
  let c = 1.0 - u.x * (1.0 - cmax);
  let s = sqrt(max(1.0 - c * c, 0.0));
  let ph = 2.0 * PI * u.y;
  return vec3<f32>(s * cos(ph), s * sin(ph), c);
}

// A point in the unit ball, uniformly.
fn lu_ball() -> vec3<f32> {
  let d = lu_cone(lu_rand2(), -1.0);
  return d * pow(lu_rand(), 1.0 / 3.0);
}

// ---- the material: diffuse (Lambert) and a GGX highlight -----------------------------------------------------------

fn lu_d(nh: f32, a2: f32) -> f32 {
  let k = nh * nh * (a2 - 1.0) + 1.0;
  return a2 / max(PI * k * k, 1e-12);
}

fn lu_lambda(c: f32, a2: f32) -> f32 { return sqrt(a2 + (1.0 - a2) * c * c); }

fn lu_g1(c: f32, a2: f32) -> f32 { return 2.0 * c / max(c + lu_lambda(c, a2), 1e-9); }

fn lu_g2(nl: f32, nv: f32, a2: f32) -> f32 {
  return 2.0 * nl * nv / max(nl * lu_lambda(nv, a2) + nv * lu_lambda(nl, a2), 1e-9);
}

// A visible normal of a GGX surface of roughness a, seen from vl (local, z up): Dupuy and Benyoub 2023.
fn lu_vndf(vl: vec3<f32>, a: f32, u: vec2<f32>) -> vec3<f32> {
  let vh = normalize(vec3<f32>(a * vl.x, a * vl.y, vl.z));
  let ph = 2.0 * PI * u.x;
  let z = (1.0 - u.y) * (1.0 + vh.z) - vh.z;
  let st = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
  let h = vec3<f32>(st * cos(ph), st * sin(ph), z) + vh;
  return normalize(vec3<f32>(a * h.x, a * h.y, max(h.z, 1e-6)));
}

struct LuMat { alb: vec3<f32>, f0: vec3<f32>, a: f32, ps: f32, wrap: f32 };

fn lu_mat(s: Surf, nv: f32) -> LuMat {
  var m: LuMat;
  m.alb = s.alb;
  m.f0 = s.f0;
  let r = clamp(s.rough, 0.03, 1.0);
  m.a = r * r;
  m.wrap = s.wrap;
  // how often a bounce samples the highlight: as much as it reflects against what it scatters
  let fe = luma(fresnel_rough(s.f0, nv, r));
  let de = luma(s.alb) * (1.0 - fe);
  m.ps = clamp(fe / max(fe + de, 1e-6), 0.1, 0.95);
  if (de <= 1e-6) { m.ps = 1.0; }
  return m;
}

// Light from l reflected toward v (both local, z the normal), times the cosine: (BSDF value, its pdf as sampled).
fn lu_eval(m: LuMat, vl: vec3<f32>, ll: vec3<f32>) -> vec4<f32> {
  let nv = max(vl.z, 1e-4);
  let nl = ll.z;
  // snow and grains let light wrap round a little past where they face away
  let nlw = (nl + m.wrap) / (1.0 + m.wrap);
  if (nlw <= 0.0) { return vec4<f32>(0.0); }
  let fv = fresnel_rough(m.f0, nv, sqrt(m.a));
  var f = m.alb * (LU_INV_PI * nlw) * (vec3<f32>(1.0) - fv);
  var pdf = (1.0 - m.ps) * max(nl, 0.0) * LU_INV_PI;
  if (nl > 0.0) {
    let h = normalize(vl + ll);
    let vh = max(dot(vl, h), 1e-4);
    let a2 = m.a * m.a;
    let D = lu_d(h.z, a2);
    let F = fresnel(m.f0, vh);
    f += F * (D * lu_g2(nl, nv, a2) / (4.0 * nv));
    pdf += m.ps * D * lu_g1(nv, a2) / (4.0 * nv);
  }
  return vec4<f32>(f, pdf);
}

// ---- lights -----------------------------------------------------------------------------------------------------------

// Whether nothing that stops light lies between p and p + d * tmax (1 open, 0 hidden).
fn lu_open(p: vec3<f32>, d: vec3<f32>, tmax: f32, eps: f32, want: f32) -> f32 {
  g_opaque = true;
  let h = trace(p, d, 2.0 * eps, tmax, want, true);
  g_opaque = false;
  return select(1.0, 0.0, h.id >= 0);
}

// Whether light from a patch of hot matter dist away along d gets to p: the ray may end on the hot matter itself (the
// patch, within reach of it: that is where the light comes from), but nothing else may stand in the way.
fn lu_open_glow(p: vec3<f32>, d: vec3<f32>, dist: f32, r: f32, eps: f32, want: f32) -> f32 {
  g_opaque = true;
  let h = trace(p, d, 2.0 * eps, dist, want, true);
  g_opaque = false;
  if (h.id < 0) { return 1.0; }
  return select(0.0, 1.0, h.id == MATTER && h.t > dist - 2.0 * r - 4.0 * eps);
}

// The HDRI as a picture to pick directions from: (its pdf over the picture, which pixel), and back.
fn lu_env_dir(uv: vec2<f32>) -> vec3<f32> {
  let ph = (uv.x - 0.5) * 2.0 * PI;
  let th = uv.y * PI;
  let r = vec3<f32>(sin(th) * sin(ph), cos(th), -sin(th) * cos(ph));
  let c = cos(U.envp.y);
  let s = sin(U.envp.y);
  let dw = vec3<f32>(c * r.x + s * r.z, r.y, -s * r.x + c * r.z);
  let c2 = cos(U.floor_d.z);
  let s2 = sin(U.floor_d.z);
  return vec3<f32>(c2 * dw.x - s2 * dw.z, dw.y, s2 * dw.x + c2 * dw.z);
}

fn lu_env_uv(d: vec3<f32>) -> vec2<f32> {
  let dw = to_world_dir(d);
  let c = cos(U.envp.y);
  let s = sin(U.envp.y);
  let r = vec3<f32>(c * dw.x - s * dw.z, dw.y, s * dw.x + c * dw.z);
  return vec2<f32>(atan2(r.x, -r.z) * (0.5 / PI) + 0.5, acos(clamp(r.y, -1.0, 1.0)) / PI);
}

fn lu_env_on() -> bool { return U.envp.x > 0.5 && U.lume2.x > 0.5; }

// The pdf (per steradian) of picking direction d from the HDRI.
fn lu_env_pdf(d: vec3<f32>) -> f32 {
  let w = u32(U.lume2.x);
  let h = u32(U.lume2.y);
  let uv = lu_env_uv(d);
  let x = min(u32(uv.x * f32(w)), w - 1u);
  let y = min(u32(uv.y * f32(h)), h - 1u);
  let st = sin(uv.y * PI);
  return ENV[h + w * h + y * w + x] / max(2.0 * PI * PI * st, 1e-6);
}

// A direction picked from the HDRI by its brightness, and its pdf (per steradian).
fn lu_env_pick() -> vec4<f32> {
  let w = u32(U.lume2.x);
  let h = u32(U.lume2.y);
  let u = lu_rand2();
  // the row (marginal), then the pixel in it (conditional): binary searches in their cumulative sums
  var lo = 0u;
  var hi = h - 1u;
  while (lo < hi) {
    let m = (lo + hi) / 2u;
    if (ENV[m] < u.y) { lo = m + 1u; } else { hi = m; }
  }
  let y = lo;
  let row = h + y * w;
  lo = 0u;
  hi = w - 1u;
  while (lo < hi) {
    let m = (lo + hi) / 2u;
    if (ENV[row + m] < u.x) { lo = m + 1u; } else { hi = m; }
  }
  let x = lo;
  let uv = vec2<f32>((f32(x) + lu_rand()) / f32(w), (f32(y) + lu_rand()) / f32(h));
  let d = lu_env_dir(uv);
  let st = sin(uv.y * PI);
  let pdf = ENV[h + w * h + y * w + x] / max(2.0 * PI * PI * st, 1e-6);
  return vec4<f32>(d, pdf);
}

fn lu_power(pdf_a: f32, pdf_b: f32) -> f32 {
  let a = pdf_a * pdf_a;
  return a / max(a + pdf_b * pdf_b, 1e-20);
}

// Light arriving straight from the lights at surface s (seen from v), times its BSDF.
fn lu_direct(s: Surf, m: LuMat, fr: mat3x3<f32>, vl: vec3<f32>) -> vec3<f32> {
  var c = vec3<f32>(0.0);
  let po = s.p + s.n * s.eps;
  let pl = light_cell(s.p);
  // the key light: a direction within the sun's disc
  if (max(U.sun.r, max(U.sun.g, U.sun.b)) > 0.0) {
    let cmax = cos(atan(1.0 / max(U.sun.w, 1.0)));
    let sf = lu_frame(U.sund.xyz);
    let lc = lu_cone(lu_rand2(), cmax);
    let l = sf[0] * lc.x + sf[1] * lc.y + sf[2] * lc.z;
    let e = lu_eval(m, vl, lu_to_local(fr, l));
    if (e.x + e.y + e.z > 0.0) {
      var vis = lu_open(po, l, 1.0e4, s.eps, s.want);
      let ps = pl + s.n * 1.5 + l * 0.75;
      if (vis > 0.0 && in_light(ps)) { vis *= samp_c(L0, lin, ps, U.ln.xyz).a; }
      c += U.sun.rgb * e.xyz * vis;
    }
  }
  // the fire: one of its point lights, picked by how much light it brings here, toward a point within it
  let nf = light_count[0];
  if (nf > 0u && max(U.fire.r, max(U.fire.g, U.fire.b)) > 0.0) {
    var tot = 0.0;
    for (var k = 0u; k < nf; k++) {
      let a = lights[2u * k];
      let d = a.xyz - s.p;
      tot += luma(lights[2u * k + 1u].rgb) / (dot(d, d) + a.w * a.w);
    }
    if (tot > 0.0) {
      let pick = lu_rand() * tot;
      var run = 0.0;
      var k = 0u;
      for (; k < nf - 1u; k++) {
        let a = lights[2u * k];
        let d = a.xyz - s.p;
        run += luma(lights[2u * k + 1u].rgb) / (dot(d, d) + a.w * a.w);
        if (run >= pick) { break; }
      }
      let a = lights[2u * k];
      let d0 = a.xyz - s.p;
      let wk = luma(lights[2u * k + 1u].rgb) / (dot(d0, d0) + a.w * a.w);
      let q = a.xyz + lu_ball() * a.w;
      let dv = q - po;
      let dist = length(dv);
      if (dist > 1e-4 && wk > 0.0) {
        let l = dv / dist;
        let e = lu_eval(m, vl, lu_to_local(fr, l));
        if (e.x + e.y + e.z > 0.0) {
          var vis = 1.0;
          if (U.fire.w > 0.0) {
            let reach = max(dist - a.w, 0.0);
            let sh = lu_open(po, l, reach, s.eps, s.want) * smoke_tr(po, l, reach);
            vis = mix(1.0, sh, U.fire.w);
          }
          let rad = lights[2u * k + 1u].rgb / (dot(d0, d0) + a.w * a.w);
          c += U.fire.rgb * rad * e.xyz * (vis * tot / wk);
        }
      }
    }
  }
  // hot matter: one of its glowing patches (it does not light itself)
  let nm = u32(ML[0].x);
  if (nm > 0u && s.self_glow < 0.5) {
    var tot = 0.0;
    for (var k = 0u; k < nm; k++) {
      let a = ML[1u + 2u * k];
      let d = a.xyz - s.p;
      tot += luma(ML[2u + 2u * k].rgb) / (dot(d, d) + a.w * a.w);
    }
    if (tot > 0.0) {
      let pick = lu_rand() * tot;
      var run = 0.0;
      var k = 0u;
      for (; k < nm - 1u; k++) {
        let a = ML[1u + 2u * k];
        let d = a.xyz - s.p;
        run += luma(ML[2u + 2u * k].rgb) / (dot(d, d) + a.w * a.w);
        if (run >= pick) { break; }
      }
      let a = ML[1u + 2u * k];
      let d0 = a.xyz - s.p;
      let wk = luma(ML[2u + 2u * k].rgb) / (dot(d0, d0) + a.w * a.w);
      let q = a.xyz + lu_ball() * a.w;
      let dv = q - po;
      let dist = length(dv);
      if (dist > 1e-4 && wk > 0.0) {
        let l = dv / dist;
        let e = lu_eval(m, vl, lu_to_local(fr, l));
        if (e.x + e.y + e.z > 0.0) {
          let vis = lu_open_glow(po, l, dist, a.w, s.eps, s.want);
          c += ML[2u + 2u * k].rgb / (dot(d0, d0) + a.w * a.w) * e.xyz * (vis * tot / wk);
        }
      }
    }
  }
  // the lights in the set: each, toward a point within it
  for (var k = 0; k < i32(U.ln.w); k++) {
    let lm = lamps[k];
    let q = lm.p.xyz + lu_ball() * lm.p.w;
    let dv = q - po;
    let d2 = dot(dv, dv);
    let dist = sqrt(d2);
    if (dist < 1e-4) { continue; }
    let l = dv / dist;
    let dc = lm.p.xyz - s.p;
    var f = 1.0 / (dot(dc, dc) + lm.p.w * lm.p.w);
    let kind = i32(lm.c.w + 0.5);
    let facing = dot(-l, lm.d.xyz);
    if (kind == 1) { f *= smoothstep(lm.d.w, lm.e.x, facing); }
    if (kind == 2) { f *= max(facing, 0.0); }
    if (f <= 0.0) { continue; }
    let e = lu_eval(m, vl, lu_to_local(fr, l));
    if (e.x + e.y + e.z <= 0.0) { continue; }
    var vis = lu_open(po, l, max(dist - lm.p.w, 0.0), s.eps, s.want);
    if (vis > 0.0 && lm.e.y > 0.5) {
      let pq = pl + s.n * 1.5 + l * 0.75;
      if (in_light(pq)) { vis *= lamp_tr(pq, k); }
    }
    c += lm.c.rgb * (f * vis * U.depth.w) * e.xyz;
  }
  // the HDRI: a direction picked by its brightness, weighed against finding it by bouncing
  if (lu_env_on()) {
    let pk = lu_env_pick();
    let l = pk.xyz;
    let e = lu_eval(m, vl, lu_to_local(fr, l));
    if (pk.w > 0.0 && e.x + e.y + e.z > 0.0) {
      var vis = lu_open(po, l, 1.0e4, s.eps, s.want);
      let pk2 = pl + s.n * 1.5;
      if (vis > 0.0 && in_light(pk2)) { vis *= samp_c(L1, lin, pk2, U.ln.xyz).x; }
      c += sky_dir(l) * e.xyz * (vis * lu_power(pk.w, e.w) / pk.w);
    }
  }
  return c;
}

// The sky a path sees when it leaves the set at p going d, after a bounce whose pdf was pdf (0: the camera's own ray).
fn lu_sky(p: vec3<f32>, d: vec3<f32>, pdf: f32) -> vec3<f32> {
  var c = sky_dir(d);
  if (lu_env_on() && pdf > 0.0) {
    c *= lu_power(pdf, lu_env_pdf(d));
  }
  // through the smoke overhead
  let pk = light_cell(p);
  if (in_light(pk)) { c *= samp_c(L1, lin, pk, U.ln.xyz).x; }
  return c;
}

// ---- glass, ice and jelly ------------------------------------------------------------------------------------------------

struct LuClear { clear: f32, ior: f32, tint: vec3<f32> };

fn lu_clear(h: Hit, s: Surf) -> LuClear {
  var c = LuClear(0.0, 1.0, vec3<f32>(1.0));
  var row = h.id;
  if (h.id >= PIECE) { row = i32(PC[u32(h.id - PIECE)].o.w + 0.5); }
  if (h.id == MATTER) {
    let uvw = matter_uvw(s.p);
    c.clear = clamp(textureSampleLevel(m_phi, lin, uvw, 0.0).y, 0.0, 1.0);
    c.tint = textureSampleLevel(m_look, lin, uvw, 0.0).rgb;
    c.ior = 1.35;
  } else if (h.id != FLOOR && (h.id < FLOOR || h.id >= PIECE) && row != LIGHTNING_ROW) {
    c.clear = clamp(U.mat[row].d.y, 0.0, 1.0);
    c.tint = U.mat[row].c.rgb;
    c.ior = max(U.mat[row].e.w, 1.0);
  }
  return c;
}

fn lu_fresnel(ci: f32, eta: f32) -> f32 {
  // unpolarised Fresnel reflectance of a dielectric, entering a medium eta times as dense
  let st2 = (1.0 - ci * ci) / (eta * eta);
  if (st2 >= 1.0) { return 1.0; }
  let ct = sqrt(1.0 - st2);
  let rs = (ci - eta * ct) / (ci + eta * ct);
  let rp = (eta * ci - ct) / (eta * ci + ct);
  return 0.5 * (rs * rs + rp * rp);
}

// Where a ray inside what was hit (h) leaves it, and the normal there (pointing out).
fn lu_exit(h: Hit, pin: vec3<f32>, rin: vec3<f32>, eps: f32) -> vec4<f32> {
  var tx = 0.0;
  var nout = vec3<f32>(0.0, 1.0, 0.0);
  if (h.id == MATTER) {
    tx = matter_exit_t(pin, rin, eps);
    nout = matter_normal(pin + rin * tx);
  } else if (h.id >= PIECE) {
    let kp = u32(h.id - PIECE);
    let ph = piece_hit(kp, pin, rin);
    tx = max(ph.t1, eps);
    let pose = piece_pose(kp);
    nout = quat_rotate(pose[1], normalize(PL[u32(PC[kp].v.w) + u32(max(ph.k1, 0))].xyz));
  } else {
    tx = exit_t(h.id, pin, rin, eps);
    nout = obj_normal(h.id, pin + rin * tx, eps);
  }
  return vec4<f32>(nout, tx);
}

// ---- one path ----------------------------------------------------------------------------------------------------------

// Whether a hit is on something drawn in CG (else the footage or the sky is seen there).
fn lu_drawn(h: Hit) -> bool {
  if (h.id == FLOOR || h.id >= PIECE || h.id == MATTER) { return true; }
  if (h.id >= 0 && h.id < FLOOR) { return U.mat[h.id].d.w > 1.5; }
  return false;
}

// The light coming back along a camera ray that first hits h (drawn), traced on from there.
fn lu_path(h0: Hit, ro0: vec3<f32>, rd0: vec3<f32>) -> vec3<f32> {
  var h = h0;
  var ro = ro0;
  var rd = rd0;
  var thr = vec3<f32>(1.0);
  var L = vec3<f32>(0.0);
  var pdf_last = 0.0;      // the pdf of the bounce that led here (0: the camera, or glass)
  let bounces = max(i32(U.lume.z + 0.5), 1);
  let cap = U.lume.w;
  var depth = 0;
  var inner = 0;           // reflections inside glass so far
  for (var guard = 0; guard < 64; guard++) {
    let s = surface_at(h, ro, rd, 1.0);
    let v = -rd;
    if (depth == 0) {
      lu_nrm = s.n;
      lu_alb = s.alb + s.f0;
      lu_dist = length(s.p - ro0);
    }
    // light given off: lightning always; hot matter only as seen (its glow already lights the rest as its patches)
    if (depth == 0 || s.self_glow < 0.5) { L += thr * s.em; }
    // glass, ice, jelly: reflected or refracted at random by Fresnel, tinted over its path through it
    let cl = lu_clear(h, s);
    if (cl.clear > 0.0 && lu_rand() < cl.clear) {
      let n = s.n;
      let ci = max(dot(n, v), 1e-4);
      let fr = lu_fresnel(ci, cl.ior);
      var go_on = true;
      if (lu_rand() < fr) {
        rd = reflect(rd, n);
        ro = s.p + n * s.eps;
      } else {
        // through it: in, across (bouncing inside while the light cannot get out), and out
        var rin = refract(rd, n, 1.0 / cl.ior);
        if (dot(rin, rin) < 0.5) { rin = rd; }
        var pin = s.p - n * s.eps;
        var out = false;
        for (var j = 0; j < 4; j++) {
          let ex = lu_exit(h, pin, rin, s.eps);
          let nout = ex.xyz;
          let pout = pin + rin * ex.w;
          thr *= pow(max(cl.tint, vec3<f32>(1e-3)), vec3<f32>(ex.w / 0.1));
          let co = max(dot(-rin, -nout), 1e-4);
          let fo = lu_fresnel(co, 1.0 / cl.ior);
          let rout = refract(rin, -nout, cl.ior);
          if (dot(rout, rout) > 0.5 && lu_rand() >= fo) {
            rd = rout;
            ro = pout + nout * (2.0 * s.eps);
            out = true;
            break;
          }
          rin = reflect(rin, -nout);
          pin = pout - nout * (2.0 * s.eps);
          inner += 1;
        }
        if (!out) { go_on = false; }
      }
      if (!go_on) { break; }
      pdf_last = 0.0;
    } else {
      // an opaque surface (or the part of a clear one that is not): its light from the lights, then a bounce
      var sv = s;
      if (cl.clear > 0.0) { sv.alb = s.alb / max(1.0 - cl.clear, 1e-3); }
      let fr = lu_frame(sv.n);
      let vl = lu_to_local(fr, v);
      if (vl.z <= 0.0) { break; }
      let m = lu_mat(sv, vl.z);
      var dl = lu_direct(sv, m, fr, vl);
      if (s.glint > 0.0 && max(U.sun.r, max(U.sun.g, U.sun.b)) > 0.0) {
        // a grain turned just so flashes the sun at the camera
        let gl = s.glint * pow(max(dot(s.gn, normalize(U.sund.xyz + v)), 0.0), 600.0) * 40.0;
        dl += U.sun.rgb * gl * max(dot(s.n, U.sund.xyz), 0.0);
      }
      var add = thr * dl;
      if (depth > 0 && cap > 0.0) {
        let y = luma(add);
        if (y > cap) { add *= cap / y; }
      }
      L += add;
      if (depth + 1 >= bounces) { break; }
      // the bounce: the highlight (by its visible normals) or diffuse (by the cosine)
      var ll: vec3<f32>;
      if (lu_rand() < m.ps) {
        let hl = lu_vndf(vl, m.a, lu_rand2());
        ll = reflect(-vl, hl);
      } else {
        ll = lu_cosine(lu_rand2());
      }
      if (ll.z <= 0.0) { break; }
      let e = lu_eval(m, vl, ll);
      if (e.w <= 0.0) { break; }
      thr *= e.xyz / e.w;
      pdf_last = e.w;
      rd = fr[0] * ll.x + fr[1] * ll.y + fr[2] * ll.z;
      ro = s.p + s.n * s.eps;
      depth += 1;
      // Russian roulette: paths that carry little light end, the rest carry more
      if (depth >= 2) {
        let q = clamp(max(thr.r, max(thr.g, thr.b)), 0.05, 0.95);
        if (lu_rand() >= q) { break; }
        thr /= q;
      }
    }
    if (inner > 8) { break; }
    // on to the next surface, or out to the sky
    let hn = trace(ro, rd, 0.0, 1.0e5, 1.0, true);
    if (!lu_drawn(hn)) {
      var sky = lu_sky(ro, rd, pdf_last);
      var add = thr * sky;
      if (depth > 0 && cap > 0.0) {
        let y = luma(add);
        if (y > cap) { add *= cap / y; }
      }
      L += add;
      break;
    }
    h = hn;
  }
  return L;
}

// What the camera sees along one ray with Lume (see() for the rest: the footage, its matte and holdouts, the sky).
fn lume_see(ro0: vec3<f32>, rd0: vec3<f32>, px: vec2<f32>, puv: vec2<f32>, rd_w: vec3<f32>) -> Seen {
  let footage = U.stage.w > 0.5;
  t_piece = -1.0;
  t_footage = 1.0e9;
  lu_nrm = vec3<f32>(0.0);
  lu_alb = vec3<f32>(1.0);
  lu_dist = 1.0e4;
  var t_foot = 1.0e9;
  var matte = 0.0;
  if (footage) {
    if (U.foot.w > 0.5) {
      let fd = footage_distance(puv, rd_w);
      if (fd > 0.0) { t_foot = max(fd - U.fwd.w / max(dot(rd_w, U.fwd.xyz), 1e-3), 0.0); }
    }
    t_footage = t_foot;
    if (U.foot.z > 0.5) { matte = clamp(textureSampleLevel(hold, lin, puv, 0.0).x, 0.0, 1.0); }
  }
  let h = trace(ro0, rd0, 0.0, select(1.0e5, t_foot, footage), 1.0, !footage);
  if ((h.id >= PIECE || h.id == MATTER)) { t_piece = h.t; }
  if (!lu_drawn(h)) {
    let c = past_cg(ro0, rd0, h, ro0, px, puv, t_foot, 0.0, true);
    lu_alb = max(c, vec3<f32>(1e-3));
    return Seen(c, select(1.0, 0.0, footage), 0.0);
  }
  var col = lu_path(h, ro0, rd0);
  let dist = h.t;
  if (h.id == FLOOR) {
    // far off, the floor fades into the sky at the horizon
    let fade = 1.0 - exp(-dist / max(U.floor_d.w, 1.0));
    col = mix(col, sky_dir(vec3<f32>(rd0.x, 0.0, rd0.z)) * 0.9, fade);
  }
  col = haze(col, dist);
  var cg = 1.0;
  if (footage) {
    cg = 1.0 - matte;
    if (matte > 0.0) { col = mix(col, footage_at(puv), matte); }
  }
  return Seen(col, cg, h.t);
}
