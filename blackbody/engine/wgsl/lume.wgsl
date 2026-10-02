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
// Paths end by Russian roulette (from the fourth bounce) once they carry little light, and indirect light is capped (Clamp bright paths) so a
// stray glint does not leave a firefly.
//
// Caustics: the light a curved clear thing (a glass ball, an ice cube, jelly) focuses is all but never found by a path
// from the camera. So each pass also traces light paths from the lights (caustics): from a lamp, the key light or the
// HDRI's bright parts, aimed at the curved clear things; through them (refracted and reflected by Fresnel, tinted); and
// where one lands on a surface, its light is sent to the camera (a straight line, nothing in the way: light tracing) and
// added to that pixel (CAU). The camera's paths then leave those lights through curved glass to it: their shadow rays stop
// at curved glass, and a bounce that reaches a light through glass does not count. Flat glass (a pane, a box, a shard)
// keeps its exact straight-through shadow. The fire's lights keep the straight-through shadow through curved glass too.
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

// How much light gets from p to p + d * tmax: 0 past anything that stops light; through glass, ice and jelly what their
// surfaces let in and out (Fresnel, straight through: a slab's shadow exactly, a ball's without its focused caustic) and
// their tint over the way through. (The light a curved one focuses is not found this way: paths through glass to a
// light are not counted again either, see lu_path.)
// Whether caustics are traced this pass (lume.wgsl caustics): then curved clear things stop the shadow rays of the lights
// traced through them, and their light through them comes in as caustics instead.
fn lu_caustics_on() -> bool { return U.lume3.w > 0.5; }

// Whether what h hit bends light into a caustic: a ball, a cylinder, a mesh, the matter (jelly). Not the floor, a box or a
// piece of something broken, whose flat faces only shift the light (their straight-through shadow is exact).
fn lu_curved(h: Hit) -> bool {
  if (h.id == MATTER) { return true; }
  if (h.id >= 0 && h.id < FLOOR) {
    let sh = i32(U.col[h.id].a.w + 0.5);
    return sh == 0 || sh == 2 || sh == 3;
  }
  return false;
}

fn lu_shadow(p: vec3<f32>, d: vec3<f32>, tmax: f32, eps: f32, want: f32, caustic: bool) -> f32 {
  if (lu_open(p, d, tmax, eps, want) <= 0.0) { return 0.0; }
  // a lamp is a solid bulb: it shades what is behind it (but not the light picked on its own surface, at tmax)
  for (var k = 0; k < i32(U.ln.w); k++) {
    let t = lu_sphere_t(p, d, lamps[k].p.xyz, max(lamps[k].p.w, 1e-3));
    if (t > 2.0 * eps && t < tmax * (1.0 - 1e-3) - eps) { return 0.0; }
  }
  if (U.lume2.z < 0.5) { return 1.0; }   // (nothing clear in the set)
  var tr = 1.0;
  var q = p;
  var left = tmax;
  for (var j = 0; j < 4; j++) {
    let h = trace(q, d, 2.0 * eps, left, want, true);
    if (h.id < 0 || h.id == FLOOR) { break; }
    let ph = q + d * h.t;
    let cl = lu_clear(h, ph);
    if (cl.clear <= 0.0) { return 0.0; }
    if (caustic && lu_curved(h)) { return 0.0; }   // (its light through it is traced from the light: caustics)
    let n = lu_hit_normal(h, ph, eps);
    let fin = lu_fresnel(max(abs(dot(n, d)), 1e-4), cl.ior);
    let ex = lu_exit(h, ph + d * eps, d, eps);
    let co = max(abs(dot(ex.xyz, d)), 1e-4);
    let fout = lu_fresnel(co, 1.0 / cl.ior);
    tr *= cl.clear * (1.0 - fin) * (1.0 - fout) * luma(pow(max(cl.tint, vec3<f32>(1e-3)), vec3<f32>(ex.w / 0.1)));
    if (tr < 1e-4) { return 0.0; }
    let adv = h.t + ex.w + 2.0 * eps;
    q = q + d * adv;
    left -= adv;
    if (left <= 0.0) { break; }
  }
  return tr;
}

// The outward normal of what h hit at p.
fn lu_hit_normal(h: Hit, p: vec3<f32>, eps: f32) -> vec3<f32> {
  if (h.id == MATTER) { return matter_normal(p); }
  if (h.id >= PIECE) {
    let kp = u32(h.id - PIECE);
    let pose = piece_pose(kp);
    return quat_rotate(pose[1], normalize(PL[u32(PC[kp].v.w) + u32(max(g_plane, 0))].xyz));
  }
  return obj_normal(h.id, p, eps);
}

// A sphere light of radius r at c, seen from p: (the cosine of its half-angle, the solid angle it covers), or
// (-2, 0) from inside it.
fn lu_cone_of(p: vec3<f32>, c: vec3<f32>, r: f32) -> vec2<f32> {
  let dc = c - p;
  let s2 = r * r / max(dot(dc, dc), 1e-12);
  if (s2 >= 1.0) { return vec2<f32>(-2.0, 0.0); }
  let cmax = sqrt(1.0 - s2);
  return vec2<f32>(cmax, 2.0 * PI * s2 / (1.0 + cmax));   // (2 pi (1 - cmax), without the cancellation)
}

// How far along l from p the sphere (c, r) is, or -1.
fn lu_sphere_t(p: vec3<f32>, l: vec3<f32>, c: vec3<f32>, r: f32) -> f32 {
  let oc = p - c;
  let b = dot(oc, l);
  let h = b * b - (dot(oc, oc) - r * r);
  if (h < 0.0) { return -1.0; }
  let sq = sqrt(h);
  if (-b - sq > 0.0) { return -b - sq; }
  if (-b + sq > 0.0) { return -b + sq; }
  return -1.0;
}

// The radiance of a lamp seen along l (toward it): its intensity spread over its disc, through its spot's cone or as
// an area light faces.
fn lu_lamp_radiance(lm: Lamp, l: vec3<f32>) -> vec3<f32> {
  let r = max(lm.p.w, 1e-3);
  var L = lm.c.rgb * (U.depth.w / (PI * r * r));
  let kind = i32(lm.c.w + 0.5);
  let facing = dot(-l, lm.d.xyz);
  if (kind == 1) { L *= smoothstep(lm.d.w, lm.e.x, facing); }
  if (kind == 2) { L *= max(facing, 0.0); }
  return L;
}

// Whether a light from a patch of hot matter dist away along d gets to p: the ray may end on the hot matter itself (the
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
      var vis = lu_shadow(po, l, 1.0e4, s.eps, s.want, lu_caustics_on());
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
            let sh = lu_shadow(po, l, reach, s.eps, s.want, false) * smoke_tr(po, l, reach);
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
  // the lights in the set: each a sphere, a direction picked within the solid angle it covers (its soft shadow exactly),
  // weighed against a bounce finding it (lu_path)
  for (var k = 0; k < i32(U.ln.w); k++) {
    let lm = lamps[k];
    let r = max(lm.p.w, 1e-3);
    let cone = lu_cone_of(s.p, lm.p.xyz, r);   // (the solid angle as the surface itself sees it)
    if (cone.x < -1.5) {
      // inside the lamp: its light all round, as if from its middle
      let dc = lm.p.xyz - s.p;
      let l = normalize(dc + vec3<f32>(0.0, 1e-6, 0.0));
      let e = lu_eval(m, vl, lu_to_local(fr, l));
      c += lu_lamp_radiance(lm, l) * (PI * r * r / (dot(dc, dc) + r * r)) * e.xyz;
      continue;
    }
    let lf = lu_frame(normalize(lm.p.xyz - s.p));
    let lc = lu_cone(lu_rand2(), cone.x);
    let l = lf[0] * lc.x + lf[1] * lc.y + lf[2] * lc.z;
    let rad = lu_lamp_radiance(lm, l);
    if (max(rad.r, max(rad.g, rad.b)) <= 0.0) { continue; }
    let e = lu_eval(m, vl, lu_to_local(fr, l));
    if (e.x + e.y + e.z <= 0.0) { continue; }
    let tl = max(lu_sphere_t(po, l, lm.p.xyz, r), 0.0);
    var vis = lu_shadow(po, l, tl, s.eps, s.want, lu_caustics_on());
    if (vis > 0.0 && lm.e.y > 0.5) {
      let pq = pl + s.n * 1.5 + l * 0.75;
      if (in_light(pq)) { vis *= lamp_tr(pq, k); }
    }
    let pdf = 1.0 / max(cone.y, 1e-12);
    c += rad * e.xyz * (vis * lu_power(pdf, e.w) / pdf);
  }
  // the HDRI: a direction picked by its brightness, weighed against finding it by bouncing
  if (lu_env_on()) {
    let pk = lu_env_pick();
    let l = pk.xyz;
    let e = lu_eval(m, vl, lu_to_local(fr, l));
    if (pk.w > 0.0 && e.x + e.y + e.z > 0.0) {
      var vis = lu_shadow(po, l, 1.0e4, s.eps, s.want, lu_caustics_on());
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

fn lu_clear(h: Hit, p: vec3<f32>) -> LuClear {
  var c = LuClear(0.0, 1.0, vec3<f32>(1.0));
  var row = h.id;
  if (h.id >= PIECE) { row = i32(PC[u32(h.id - PIECE)].o.w + 0.5); }
  if (h.id == MATTER) {
    let uvw = matter_uvw(p);
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
  var glassed = false;     // the path has gone through glass since its last bounce off a surface (what it finds then,
                           // the lights reached through glass, came in already as their shadows through it)
  for (var guard = 0; guard < 64; guard++) {
    var s = surface_at(h, ro, rd, 1.0);
    // a surface found exactly (the floor, a plain shape, a piece) needs only a hair's offset for the rays off it: a
    // marched one keeps the stage's (a pixel or two), which would put the light's start that much nearer the lights
    if (h.id == FLOOR || h.id >= PIECE || (h.id >= 0 && h.id < FLOOR && plain(h.id))) { s.eps = min(s.eps, 1.0e-4); }
    let v = -rd;
    if (depth == 0) {
      lu_nrm = s.n;
      lu_alb = s.alb + s.f0;
      lu_dist = length(s.p - ro0);
    }
    // light given off: lightning always; hot matter only as seen (its glow already lights the rest as its patches)
    if (depth == 0 || s.self_glow < 0.5) { L += thr * s.em; }
    // glass, ice, jelly: reflected or refracted at random by Fresnel, tinted over its path through it
    let cl = lu_clear(h, s.p);
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
      if (depth > 0) { glassed = true; }
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
      // the bounce: the highlight (by its visible normals) or diffuse (by the cosine). It is traced even after the
      // last bounce, for the sky it may reach: that is this surface's light too, the half of the sky that the light
      // picked from the HDRI above leaves to it (multiple importance sampling)
      let last = depth + 1 >= bounces;
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
      glassed = false;
      rd = fr[0] * ll.x + fr[1] * ll.y + fr[2] * ll.z;
      ro = s.p + s.n * s.eps;
      depth += 1;
      // Russian roulette, from the fourth bounce: paths that carry little light end, the rest carry more (ending them
      // sooner saves little time and adds noise: measured against Mitsuba, from the second it doubled a glossy set's)
      if (depth >= 4 && !last) {
        let q = clamp(max(thr.r, max(thr.g, thr.b)), 0.05, 0.95);
        if (lu_rand() >= q) { break; }
        thr /= q;
      }
    }
    if (inner > 8) { break; }
    // on to the next surface, or out to the sky
    let hn = trace(ro, rd, 0.0, 1.0e5, 1.0, true);
    // a lamp in the way (after a bounce: the camera does not see the lamps themselves), weighed against having picked it
    if (depth > 0) {
      var tl = 1.0e9;
      var kl = -1;
      for (var k = 0; k < i32(U.ln.w); k++) {
        let t = lu_sphere_t(ro, rd, lamps[k].p.xyz, max(lamps[k].p.w, 1e-3));
        if (t > 0.0 && t < tl) {
          tl = t;
          kl = k;
        }
      }
      if (kl >= 0 && (hn.id < 0 || tl < hn.t)) {
        if (!glassed) {
          var w = 1.0;
          if (pdf_last > 0.0) {
            let cone = lu_cone_of(ro, lamps[kl].p.xyz, max(lamps[kl].p.w, 1e-3));
            if (cone.x > -1.5) { w = lu_power(pdf_last, 1.0 / max(cone.y, 1e-12)); }
          }
          var add = thr * lu_lamp_radiance(lamps[kl], rd) * w;
          if (cap > 0.0) {
            let y = luma(add);
            if (y > cap) { add *= cap / y; }
          }
          L += add;
        }
        break;
      }
    }
    if (!lu_drawn(hn)) {
      var sky = lu_sky(ro, rd, pdf_last);
      if (glassed && lu_env_on()) { sky = vec3<f32>(0.0); }   // (the HDRI through glass: in its shadow ray already)
      var add = thr * sky;
      if (depth > 0 && cap > 0.0) {
        let y = luma(add);
        if (y > cap) { add *= cap / y; }
      }
      L += add;
      break;
    }
    if (depth >= bounces) { break; }   // (past the last bounce: only the sky counted)
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


// ---- caustics: light traced from the lights through curved glass, to the camera ---------------------------------------

const LU_CAU_SCALE: f32 = 1024.0;   // CAU holds radiance times this, rounded (atomics are integers)

// The caustics this pass put in pixel k (radiance, rgb).
fn lu_caustic_at(k: u32) -> vec3<f32> {
  return vec3<f32>(f32(atomicLoad(&CAU[3u * k])), f32(atomicLoad(&CAU[3u * k + 1u])),
                   f32(atomicLoad(&CAU[3u * k + 2u]))) / LU_CAU_SCALE;
}

struct LuThru { ro: vec3<f32>, rd: vec3<f32>, tint: vec3<f32>, ok: bool };

// Light travelling along rd that meets clear thing h at p from outside: reflected off it or refracted in, across
// (reflected inside while it cannot get out) and out, at random by Fresnel; where it goes on from, which way, and the tint
// of its way through. (ok false: it was lost inside.)
fn lu_through(h: Hit, p: vec3<f32>, rd: vec3<f32>, cl: LuClear, eps: f32) -> LuThru {
  var n = lu_hit_normal(h, p, eps);
  if (dot(n, rd) > 0.0) { n = -n; }
  let ci = max(dot(n, -rd), 1e-4);
  if (lu_rand() < lu_fresnel(ci, cl.ior)) {
    return LuThru(p + n * eps, reflect(rd, n), vec3<f32>(1.0), true);
  }
  var rin = refract(rd, n, 1.0 / cl.ior);
  if (dot(rin, rin) < 0.5) { rin = rd; }
  var pin = p - n * eps;
  var tint = vec3<f32>(1.0);
  for (var j = 0; j < 6; j++) {
    let ex = lu_exit(h, pin, rin, eps);
    let nout = ex.xyz;
    let pout = pin + rin * ex.w;
    tint *= pow(max(cl.tint, vec3<f32>(1e-3)), vec3<f32>(ex.w / 0.1));
    let fo = lu_fresnel(max(dot(rin, nout), 1e-4), 1.0 / cl.ior);
    let rout = refract(rin, -nout, cl.ior);
    if (dot(rout, rout) > 0.5 && lu_rand() >= fo) {
      return LuThru(pout + nout * (2.0 * eps), rout, tint, true);
    }
    rin = reflect(rin, -nout);
    pin = pout - nout * (2.0 * eps);
  }
  return LuThru(pin, rin, tint, false);
}

const LU_CAU_BOUNCES: i32 = 2;   // a caustic's light is followed this many bounces on from where it lands

// Light that came along rd to surface s, carrying beta, as the camera sees it: sent along a straight line to the camera
// (nothing in the way, glass too), into the pixel it is seen in (light tracing; U.lume3.z: 1 / (paths x a pixel's area)).
fn lu_splat(s: Surf, rd: vec3<f32>, beta: vec3<f32>) {
  let wc = U.ceye.xyz - s.p;
  let dist = length(wc);
  let wn = wc / max(dist, 1e-6);
  let nx = dot(s.n, wn);
  if (nx <= 0.0 || dot(s.n, -rd) <= 0.0) { return; }
  let hv = trace(s.p + s.n * max(s.eps, 1.0e-4), wn, 0.0, dist, 1.0, true);
  if (hv.id >= 0) { return; }
  let clip = U.cvp * (U.cl2w * vec4<f32>(s.p, 1.0));
  if (clip.w <= 1e-6) { return; }
  let ndc = clip.xy / clip.w;
  let px = vec2<i32>(floor(vec2<f32>(0.5 * (ndc.x + 1.0), 0.5 * (1.0 - ndc.y)) * U.res.xy));
  if (px.x < 0 || px.y < 0 || f32(px.x) >= U.res.x || f32(px.y) >= U.res.y) { return; }
  let fr = lu_frame(s.n);
  let vl = lu_to_local(fr, wn);
  let ll = lu_to_local(fr, -rd);
  let m = lu_mat(s, vl.z);
  let f = lu_eval(m, vl, ll).xyz / max(ll.z, 1e-4);
  // the camera's axis (fire-local) and how far off it the surface is seen
  let fwd = normalize((U.w2l * vec4<f32>(U.fwd.xyz, 0.0)).xyz);
  let cc = max(dot(-wn, fwd), 1e-4);
  let val = beta * f * (nx / (dist * dist * cc * cc * cc) * U.lume3.z);
  let k = u32(px.y) * u32(U.res.x) + u32(px.x);
  let q = min(val * LU_CAU_SCALE, vec3<f32>(3.0e7));
  atomicAdd(&CAU[3u * k], u32(q.r + 0.5));
  atomicAdd(&CAU[3u * k + 1u], u32(q.g + 0.5));
  atomicAdd(&CAU[3u * k + 2u], u32(q.b + 0.5));
}

// A light path from one of the lights, aimed at one of the curved clear things: (where from, which way, the light it
// carries: its share of the light's power, for one path among U.lume3.x), or a zero light when there is nothing to send.
struct LuEmit { ro: vec3<f32>, rd: vec3<f32>, beta: vec3<f32> };

fn lu_emit() -> LuEmit {
  var e = LuEmit(vec3<f32>(0.0), vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0));
  let nl = i32(U.ln.w);
  let sun_on = max(U.sun.r, max(U.sun.g, U.sun.b)) > 0.0;
  let env_on = lu_env_on();
  let kinds = nl + select(0, 1, sun_on) + select(0, 1, env_on);
  let nt = i32(U.lume3.y);
  if (kinds == 0 || nt == 0) { return e; }
  let pick = min(i32(lu_rand() * f32(kinds)), kinds - 1);
  let tg = U.ctg[min(i32(lu_rand() * f32(nt)), nt - 1)];
  let sel = f32(kinds * nt);   // (1 / the chance of this light and this target)
  if (pick < nl) {
    // a lamp: from its middle, a direction within the cone the target fills seen from it
    let lm = lamps[pick];
    let c = lm.p.xyz;
    let cone = lu_cone_of(c, tg.xyz, tg.w);
    var l = vec3<f32>(0.0, -1.0, 0.0);
    var pdf = 1.0 / (4.0 * PI);
    if (cone.x < -1.5) {
      l = lu_cone(lu_rand2(), -1.0);
    } else {
      let fr = lu_frame(normalize(tg.xyz - c));
      let lc = lu_cone(lu_rand2(), cone.x);
      l = fr[0] * lc.x + fr[1] * lc.y + fr[2] * lc.z;
      pdf = 1.0 / max(cone.y, 1e-12);
    }
    let r = max(lm.p.w, 1e-3);
    let intensity = lu_lamp_radiance(lm, -l) * (PI * r * r);
    e.ro = c;
    e.rd = l;
    e.beta = intensity * (sel / pdf);
    return e;
  }
  // the key light or the HDRI: parallel light over a disc as wide as the target, from far off
  var dir = vec3<f32>(0.0, 1.0, 0.0);   // toward the light
  var power = vec3<f32>(0.0);           // what crosses a unit area facing it
  if (sun_on && pick == nl) {
    let cmax = cos(atan(1.0 / max(U.sun.w, 1.0)));
    let sf = lu_frame(U.sund.xyz);
    let lc = lu_cone(lu_rand2(), cmax);
    dir = sf[0] * lc.x + sf[1] * lc.y + sf[2] * lc.z;
    power = U.sun.rgb;
  } else {
    let pk = lu_env_pick();
    if (pk.w <= 0.0) { return e; }
    dir = pk.xyz;
    power = sky_dir(dir) / pk.w;
  }
  if (U.stage.x > 0.5 && dir.y < 0.0) { return e; }   // (light from below the horizon: the ground stops it)
  let travel = -dir;
  let fr = lu_frame(travel);
  let u = lu_rand2();
  let rr = sqrt(u.x) * tg.w;
  let ph = 2.0 * PI * u.y;
  // (from just outside the target, if nothing between there and the light stops it: a roof over the glass shades it)
  e.ro = tg.xyz + (fr[0] * cos(ph) + fr[1] * sin(ph)) * rr - travel * (2.0 * tg.w);
  e.rd = travel;
  if (lu_open(e.ro, dir, 1.0e5, 1.0e-4, 1.0) <= 0.0) { return e; }
  e.beta = power * (PI * tg.w * tg.w * sel);
  return e;
}

@compute @workgroup_size(64, 1, 1)
fn caustics(@builtin(global_invocation_id) id: vec3<u32>) {
  if (f32(id.x) >= U.lume3.x) { return; }
  lu_rng = pcg1(id.x * 9781u + u32(U.lume.y + 0.5) * 6271u + 17u);
  g_tau = U.res.w * (lu_rand() - 0.5);
  g_jit = lu_rand();
  let em = lu_emit();
  if (max(em.beta.r, max(em.beta.g, em.beta.b)) <= 0.0) { return; }
  var ro = em.ro;
  var rd = em.rd;
  var beta = em.beta;
  var curved = 0;
  var landed = 0;
  let eps = 1.0e-4;
  for (var j = 0; j < 16; j++) {
    let h = trace(ro, rd, 0.0, 2.0e5, 1.0, true);
    if (h.id < 0 || !lu_drawn(h)) { return; }
    let p = ro + rd * h.t;
    let cl = lu_clear(h, p);
    if (cl.clear > 0.0) {
      if (lu_rand() >= cl.clear) { return; }   // (its cloudy part)
      if (!lu_curved(h)) {
        // flat glass: straight through by its Fresnel and tint, as its shadow is
        let n = lu_hit_normal(h, p, eps);
        let ex = lu_exit(h, p + rd * eps, rd, eps);
        let t = (1.0 - lu_fresnel(max(abs(dot(n, rd)), 1e-4), cl.ior)) * (1.0 - lu_fresnel(max(abs(dot(ex.xyz, rd)), 1e-4), 1.0 / cl.ior));
        beta *= t * pow(max(cl.tint, vec3<f32>(1e-3)), vec3<f32>(ex.w / 0.1));
        ro = p + rd * (ex.w + 2.0 * eps);
        continue;
      }
      let th = lu_through(h, p, rd, cl, eps);
      if (!th.ok) { return; }
      beta *= th.tint;
      ro = th.ro;
      rd = th.rd;
      curved += 1;
      continue;
    }
    if (curved == 0) { return; }   // (straight from the light: the camera's own paths have that light)
    // it has landed: the light this surface sends the camera (light tracing), then on, bounced off it (the light the
    // caustic throws round it, onto the floor beside it and the glass's underside: the camera's paths leave all light
    // that came through curved glass to these)
    let s = surface_at(h, ro, rd, 1.0);
    lu_splat(s, rd, beta);
    landed += 1;
    if (landed > LU_CAU_BOUNCES) { return; }
    let fr = lu_frame(s.n);
    let li = lu_to_local(fr, -rd);
    if (li.z <= 0.0) { return; }
    let lo = lu_cosine(lu_rand2());   // (diffuse: f cos / pdf = f pi)
    let m = lu_mat(s, lo.z);
    let ev = lu_eval(m, lo, li);
    beta *= ev.xyz / max(li.z, 1e-4) * PI;
    let q = clamp(max(beta.r, max(beta.g, beta.b)) / max(max(em.beta.r, max(em.beta.g, em.beta.b)), 1e-12) * 4.0, 0.05, 1.0);
    if (lu_rand() >= q) { return; }   // (Russian roulette on what is left of its light)
    beta /= q;
    ro = s.p + s.n * max(s.eps, 1.0e-4);
    rd = fr[0] * lo.x + fr[1] * lo.y + fr[2] * lo.z;
  }
}
