// Grass and plants (engine/strands.py), drawn into the raster layer before the fire is marched (raster.py: the march
// stops at it, and cloth_merge.wgsl puts it under the fire), in the same light as the smoke and the cloth: the key
// light (shadowed by the smoke and the objects, from the light volume), the sky, the fire (its point lights, shadowed
// toward their centre) and the lights in the set.
//
// Each blade is a ribbon along its points, its flat face turned as it grew, full width most of the way and tapering
// to a point (wheat's top segment is its ear). A blade:
// - lets light through: lit from behind it glows, tinted (translucency);
// - is waxy: a soft highlight;
// - is shaded by the blades round it: the deeper into the canopy, the less of the sky and the sun reaches it;
// - browns as it heats, chars as it burns, its burning tip glowing, and burns down to stubble whose embers die out.
// Blades narrower than a pixel are drawn a pixel wide and keep only their share of the pixels (a random share each
// anti-aliasing pass), so a field far off averages out instead of shimmering.
//
// Outputs as cloth_draw.wgsl's: col = colour (premultiplied, alpha 1); aux = view depth (clip w), distance along the
// ray from where the march starts (m), _, coverage; glow = its own light.
//!include common.wgsl
//!include shade.wgsl
//!include strand_common.wgsl

struct Lamp { p: vec4<f32>, c: vec4<f32>, d: vec4<f32>, e: vec4<f32> };

struct Params {
  vp: mat4x4<f32>,     // world -> clip
  l2w: mat4x4<f32>,    // fire-local -> world
  res: vec4<f32>,      // width, height, sub-pixel jitter (px)
  eye: vec4<f32>,      // camera (fire-local), distance from the camera to where the march starts, along the view axis
  fwd: vec4<f32>,      // camera forward (fire-local), _
  sund: vec4<f32>,     // toward the key light (fire-local), strength of the fire's light (Lighting, as on the smoke)
  lg: vec4<f32>,       // simulation grid corner (fire-local m), its cell size (m)
  ln: vec4<f32>,       // light-volume dims, lights in the set (count)
  lsc: vec4<f32>,      // light-volume cells per simulation cell (xyz), light volume present (1/0; 2: with the objects'
                       // extinction in L1.w)
  look: Look,
  xp: vec4<f32>,       // its reflected light times this (the liquid render's exposure; 1 with the fire), time (s), _, _
  patches: array<Patch, MAX_PATCHES>,
};

@group(0) @binding(0) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> BL: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> RT: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> ST: array<vec4<f32>>;
@group(0) @binding(4) var L0: texture_3d<f32>;
@group(0) @binding(5) var L1: texture_3d<f32>;
@group(0) @binding(6) var bb: texture_2d<f32>;
@group(0) @binding(7) var lin: sampler;
@group(0) @binding(8) var<storage, read> lights: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read> light_count: array<u32>;
@group(0) @binding(10) var<storage, read> lamps: array<Lamp>;
@group(0) @binding(11) var LT: texture_3d<f32>;
@group(0) @binding(12) var E: texture_3d<f32>;   // the light volume's emission and extinction (smoke and cloth)
@group(1) @binding(0) var<uniform> U: Params;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) p: vec3<f32>,
  @location(1) n: vec3<f32>,
  @location(2) w: vec3<f32>,                         // across the blade
  @location(3) u: f32,                               // along it: 0 at its root, 1 at its tip
  @location(4) side: f32,                            // across it: -1 .. 1
  @location(5) @interpolate(flat) st: vec4<f32>,
  @location(6) @interpolate(flat) blade: vec4<f32>,  // patch, its random number, its height (m), its index
  @location(7) cov: f32,                             // its share of the pixels it is drawn over
  @location(8) cw: f32,
};

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VOut {
  var o: VOut;
  let per = (POINTS - 1u) * 6u;
  let i = vid / per;
  let r = vid % per;
  let k = r / 6u;
  let c = r % 6u;
  let rt = RT[i];
  if (rt.w < 0.5) {
    o.pos = vec4<f32>(0.0, 0.0, -2.0, 1.0);   // (it does not grow: outside the view)
    return o;
  }
  let b0 = BL[2u * i];
  let b1 = BL[2u * i + 1u];
  let pa = U.patches[u32(b0.z + 0.5)];
  // the segment's two triangles: (k left, k right, k+1 left), (k+1 left, k right, k+1 right)
  let j = k + select(0u, 1u, c == 2u || c == 3u || c == 5u);
  let sd = select(-1.0, 1.0, c == 1u || c == 4u || c == 5u);
  let p = X[i * POINTS + j].xyz;
  let ja = select(j - 1u, 0u, j == 0u);
  let jb = min(j + 1u, POINTS - 1u);
  let t = normalize(X[i * POINTS + jb].xyz - X[i * POINTS + ja].xyz + vec3<f32>(0.0, 1e-7, 0.0));
  // its flat face, turned with it (an ear of grain is round: it faces the camera)
  let ear = pa.form.y > 0.5 && j >= POINTS - 2u;
  var f0 = vec3<f32>(cos(b1.z), 0.0, sin(b1.z));
  if (ear) { f0 = cross(t, U.eye.xyz - p); }
  var wv = f0 - t * dot(f0, t);
  if (length(wv) < 1e-6) { wv = cross(t, vec3<f32>(0.0, 0.0, 1.0)); }
  wv = normalize(wv);
  let n = normalize(cross(t, wv));
  // full width most of the way, to a point at the tip; wheat's ear on its last segment
  let uj = f32(j) / f32(POINTS - 1u);
  var hw = 0.5 * b1.y * min(1.0, 2.6 * (1.0 - uj));
  if (ear) { hw = 0.5 * b1.y * select(3.0, 1.6, j == POINTS - 1u); }
  // at least a pixel across: a thinner blade is drawn a pixel wide, keeping its share of the pixels
  let cp = U.vp * (U.l2w * vec4<f32>(p, 1.0));
  let ce = U.vp * (U.l2w * vec4<f32>(p + wv * max(hw, 1e-5), 1.0));
  var cov = 1.0;
  if (cp.w > 1e-4 && ce.w > 1e-4) {
    let pxw = 2.0 * length((ce.xy / ce.w - cp.xy / cp.w) * U.res.xy * 0.5);
    if (pxw < 1.0) {
      cov = max(pxw, 0.08);
      hw = max(hw, 1e-5) / max(pxw, 0.08);
    }
  }
  let q = p + wv * (hw * sd);
  var clip = U.vp * (U.l2w * vec4<f32>(q, 1.0));
  clip.x -= 2.0 * U.res.z / U.res.x * clip.w;
  clip.y += 2.0 * U.res.w / U.res.y * clip.w;
  o.pos = clip;
  o.p = q;
  o.n = n;
  o.w = wv;
  o.u = uj;
  o.side = sd;
  o.st = ST[i];
  o.blade = vec4<f32>(b0.z, b0.w, b1.x, f32(i));
  o.cov = cov;
  o.cw = clip.w;
  return o;
}

struct FOut {
  @location(0) col: vec4<f32>,
  @location(1) aux: vec4<f32>,
  @location(2) glow: vec4<f32>,
};

fn hash2(p: vec2<f32>) -> f32 {
  return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

// Light from one direction l (unit) with irradiance e (at normal incidence) on a blade: its colour seen from v, lit
// from the front (diffuse, and a waxy highlight) or coming through from behind.
fn light_blade(e: vec3<f32>, l: vec3<f32>, n: vec3<f32>, v: vec3<f32>, alb: vec3<f32>, tr: f32, gloss: f32) -> vec3<f32> {
  let nl = dot(n, l);
  // (light through a leaf is greener and brighter than its reflection: it is filtered once, not scattered back)
  if (nl <= 0.0) { return e * pow(alb, vec3<f32>(0.7)) * (tr * (-nl) / 3.14159265); }
  let h = normalize(l + v);
  let nh = max(dot(n, h), 0.0);
  let a = 0.12;   // (roughness 0.35, squared)
  let dd = nh * nh * (a - 1.0) + 1.0;
  let d = a / (3.14159265 * dd * dd);
  let f = 0.04 + 0.96 * pow(1.0 - max(dot(h, v), 0.0), 5.0);
  let spec = gloss * d * f / (4.0 * max(dot(n, v), 0.05));
  return e * (alb * ((1.0 - tr) / 3.14159265) + vec3<f32>(spec)) * nl;
}

fn lamp_tr(pl: vec3<f32>, i: i32) -> f32 {
  let ln = U.ln.xyz;
  let blocks = select(1.0, 2.0, U.ln.w > 4.5);
  let q = clamp(pl, vec3<f32>(0.5), ln - vec3<f32>(0.5));
  let t = textureSampleLevel(LT, lin, vec3<f32>(q.x / ln.x, q.y / ln.y, (q.z + f32(i / 4) * ln.z) / (ln.z * blocks)), 0.0);
  let j = i & 3;
  return select(select(t.w, t.z, j == 2), select(t.y, t.x, j == 0), j < 2);
}

fn in_light(pl: vec3<f32>) -> bool {
  return U.lsc.w > 0.5 && all(pl >= vec3<f32>(0.0)) && all(pl <= U.ln.xyz);
}

// The objects' shadow (L1.w) on light from direction l reaching light-volume point pl, within `reach` light cells.
fn obj_shadow(pl: vec3<f32>, l: vec3<f32>, reach: f32) -> f32 {
  if (U.lsc.w < 1.5) { return 1.0; }
  let st = l / max(abs(l.x), max(abs(l.y), max(abs(l.z), 1e-6)));
  let len = length(st);
  var od = 0.0;
  var q = pl;
  var t = 0.0;
  for (var k = 0; k < 128; k++) {
    q += st;
    t += len;
    if (t > reach || any(q < vec3<f32>(0.0)) || any(q > U.ln.xyz)) { break; }
    od += samp_c(L1, lin, q, U.ln.xyz).w;
  }
  return exp(-od * length(st * U.lg.w / U.lsc.xyz));
}

// How much of the fire's light gets to fire-local p from the fire's centre pc (size spread): the smoke, the cloth and
// the objects in between.
fn fire_shadow(p: vec3<f32>, pc: vec3<f32>, spread: f32) -> f32 {
  if (U.lsc.w < 0.5) { return 1.0; }
  let d = pc - p;
  let dist = length(d);
  if (dist < 1e-4) { return 1.0; }
  let dir = d / dist;
  let lc = U.lg.w / U.lsc.xyz;
  let cell = max(lc.x, max(lc.y, lc.z));
  let p0 = p + dir * (0.75 * cell);
  let reach = max(dist - 0.5 * spread - 0.75 * cell, 0.0);
  let steps = clamp(i32(ceil(reach / (0.75 * cell))), 1, 48);
  let dt = reach / f32(steps);
  let ln = U.ln.xyz;
  var od = 0.0;
  for (var k = 0; k < steps; k++) {
    let q = (p0 + dir * ((f32(k) + 0.5) * dt) - U.lg.xyz) / lc;
    if (any(q < vec3<f32>(0.0)) || any(q > ln)) { continue; }
    od += samp_c(E, lin, q, ln).a + samp_c(L1, lin, q, ln).w;
  }
  return exp(-od * dt);
}

// A glowing solid at temperature k (K), in the fire's exposure.
fn glow_at(k: f32, L: Look) -> vec3<f32> {
  let bbv = bb_lookup(k);
  return bbv.rgb * min(pow(10.0, (bbv.a - L.misc.x) * L.fire.w), 1.0e4) * L.fire2.x;
}

@fragment
fn fs(i: VOut) -> FOut {
  // thinner than a pixel: its share of the pixels, a different random share each anti-aliasing pass
  if (i.cov < 0.999) {
    let hsh = hash2(i.pos.xy + U.res.zw * 61.7 + vec2<f32>(i.blade.w * 0.0137, i.blade.w * 0.0071));
    if (hsh > i.cov) { discard; }
  }
  let pa = U.patches[u32(i.blade.x + 0.5)];
  let st = i.st;
  let rnd = i.blade.y;
  let L = U.look;
  let v = normalize(U.eye.xyz - i.p);
  var n = normalize(i.n);
  if (dot(n, v) < 0.0) { n = -n; }   // both faces
  // its midrib: a blade is folded a little, its halves turned toward its edges
  let nn = normalize(n + i.w * (0.3 * i.side));
  // its colour: its kind's, fresh to dry by its dryness, each blade a little different, drier toward the tip
  let dryness = clamp(pa.dry.w + 0.2 * (rnd - 0.5) + 0.3 * smoothstep(0.6, 1.0, i.u), 0.0, 1.0);
  var alb = mix(pa.fresh.rgb, pa.dry.rgb, dryness) * (0.8 + 0.4 * fract(rnd * 7.31));
  if (pa.form.y > 0.5 && i.u > 0.74) { alb = pa.dry.rgb * vec3<f32>(1.08, 1.0, 0.82) * (0.9 + 0.2 * fract(rnd * 3.7)); }   // the ear
  // heat: browned ahead of the flame; charred from the foot up as it burns; black stubble after
  let burning = st.z > 0.5;
  let done = st.y >= 1.0;
  alb = mix(alb, alb * vec3<f32>(0.5, 0.33, 0.15), 0.85 * clamp(st.x, 0.0, 1.0) * select(1.0, 0.0, burning || done));
  var ch = select(0.0, 1.0, done);
  if (burning) { ch = smoothstep(i.u - 0.15, i.u + 0.05, 3.0 * st.y); }
  alb = mix(alb, vec3<f32>(0.025, 0.021, 0.018) * (0.8 + 0.4 * rnd), ch);
  let tr = pa.fresh.w * (1.0 - ch);
  let gloss = 0.5 * (1.0 - ch);
  // the canopy round it: the deeper in, the less of the sky and the sun gets to it
  let depth = (1.0 - i.u) * min(i.blade.z, pa.form.w);
  let sig = pa.form.z;

  var c = vec3<f32>(0.0);
  let lg = (i.p - U.lg.xyz) / U.lg.w;
  let pl = lg * U.lsc.xyz;
  var sun_tr = exp(-sig * depth / max(U.sund.y, 0.2));
  var sky = mix(0.25, 1.0, exp(-0.7 * sig * depth));
  let ps = pl + U.sund.xyz * 0.75;
  if (in_light(ps)) { sun_tr *= samp_c(L0, lin, ps, U.ln.xyz).a * obj_shadow(ps, U.sund.xyz, 1.0e9); }
  let pk = pl + vec3<f32>(0.0, 1.5, 0.0);
  if (in_light(pk)) { sky *= samp_c(L1, lin, pk, U.ln.xyz).x; }
  c += light_blade(L.sun.rgb * sun_tr, U.sund.xyz, nn, v, alb, tr, gloss);
  // the sky, from above (a blade facing up sees more of it), a little through it
  c += L.amb.rgb * sky * alb * ((0.55 + 0.45 * abs(nn.y)) * (1.0 - 0.5 * tr) + 0.2 * tr);
  // the fire
  let nf = light_count[0];
  var cf = vec3<f32>(0.0);
  var fc = vec3<f32>(0.0);
  var fc2 = 0.0;
  var fr = 0.0;
  var fw = 0.0;
  for (var k = 0u; k < nf; k++) {
    let a = lights[2u * k];
    let d = a.xyz - i.p;
    let r2 = dot(d, d);
    let e = lights[2u * k + 1u].rgb * (U.sund.w / (r2 + a.w * a.w));
    cf += light_blade(e, d * inverseSqrt(max(r2, 1e-8)), nn, v, alb, tr, gloss);
    let w = luma(e);
    fc += a.xyz * w;
    fc2 += dot(a.xyz, a.xyz) * w;
    fr += a.w * w;
    fw += w;
  }
  if (fw > 1e-12) {
    let pc = fc / fw;
    cf *= fire_shadow(i.p, pc, sqrt(max(fc2 / fw - dot(pc, pc), 0.0)) + fr / fw);
  }
  c += cf * mix(0.35, 1.0, exp(-0.5 * sig * depth));
  // the lights in the set
  for (var k = 0; k < i32(U.ln.w); k++) {
    let lm = lamps[k];
    let d = lm.p.xyz - i.p;
    let d2 = dot(d, d);
    let dir = d * inverseSqrt(max(d2, 1e-12));
    var f = 1.0 / (d2 + lm.p.w * lm.p.w);
    let kind = i32(lm.c.w + 0.5);
    let facing = dot(-dir, lm.d.xyz);
    if (kind == 1) { f *= smoothstep(lm.d.w, lm.e.x, facing); }
    if (kind == 2) { f *= max(facing, 0.0); }
    if (f <= 0.0) { continue; }
    var tl = 1.0;
    let pq = pl + dir * 0.75;
    if (in_light(pq)) {
      let lc = U.lg.w / U.lsc.xyz;
      tl = lamp_tr(pq, k) * obj_shadow(pq, dir, length(d / lc));
    }
    c += light_blade(lm.c.rgb * (f * tl), dir, nn, v, alb, tr, gloss) * mix(0.35, 1.0, exp(-0.5 * sig * depth));
  }
  // its own light: the burning tip, then the stubble's dying embers
  var glow = vec3<f32>(0.0);
  let t = U.xp.y;
  let flick = 0.88 + 0.24 * hash2(vec2<f32>(i.blade.w * 0.31, floor(t * 24.0)));
  if (burning) {
    glow = glow_at(1150.0 * flick, L) * (2.5 * smoothstep(0.7, 0.95, i.u));
  } else if (done && st.y < 1.6) {
    let fade = (1.6 - st.y) / 0.6;
    let speck = smoothstep(0.55, 0.85, hash2(vec2<f32>(i.blade.w * 0.173, 4.0)));
    glow = glow_at(mix(750.0, 1000.0, fade) * flick, L) * (1.5 * fade * speck * smoothstep(0.6, 1.0, i.u));
  }
  var o: FOut;
  o.col = vec4<f32>(c * U.xp.x + glow, 1.0);
  let dist = max(length(i.p - U.eye.xyz) - U.eye.w / max(dot(-v, U.fwd.xyz), 1e-3), 0.0);
  o.aux = vec4<f32>(i.cw, dist, 0.0, 1.0);
  o.glow = vec4<f32>(glow, 1.0);
  return o;
}
