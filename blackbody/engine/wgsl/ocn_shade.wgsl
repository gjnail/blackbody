// Shading the sea (liq_march.wgsl only; ocn_sample.wgsl samples it). Uses the march's own helpers:
// fresnel, land, lamps_on, lamps_glint, ww_light, foam_cover, worley, sea_xz,
// sea_fade (a pixel's footprint on the water, m), lvl_at, side_in, box_d, deep_colour.
//
//   the surface: exact normals of the choppy sea, with the waves too small for the pixel moved into
//     the roughness, so sun glitter spreads into a long glitter path toward the horizon instead of
//     shimmering, and gusts darken and brighten patches of ripples
//   crest glow: sunlight through the thin top of a wave, scattered toward the eye (the green-blue
//     glow of a backlit crest)
//   foam: fresh whitecaps where crests break, white and dense; the thinner foam they leave, drawn
//     out along the wind into streaks and broken into lace; and under a breaking crest the bubbles
//     churned into the water, a pale turquoise glow

// How far p (grid cells) is inside the box from its nearest side the open water flows through (a
// flume's walls do not count: the simulation holds the water right up to them).
fn sea_side_in(p: vec3<f32>) -> f32 {
  let big = 1.0e6;
  let s = U.ocx[13];
  // (a side the sea flows through counts; past a mirrored flume wall (-1), the far edge of its mirror
  // image; a wall the simulation holds the water right up to, not at all)
  let lx = select(select(big, p.x + U.n.x, abs(s.x + 1.0) < 0.5), p.x, s.x > 0.5);
  let hx = select(select(big, 2.0 * U.n.x - p.x, abs(s.y + 1.0) < 0.5), U.n.x - p.x, s.y > 0.5);
  let lz = select(select(big, p.z + U.n.z, abs(s.z + 1.0) < 0.5), p.z, s.z > 0.5);
  let hz = select(select(big, 2.0 * U.n.z - p.z, abs(s.w + 1.0) < 0.5), U.n.z - p.z, s.w > 0.5);
  return min(min(lx, hx), min(lz, hz));
}

// How big a pixel is on the water at p (m): its size at that distance, stretched where the view
// grazes the water (short waves there only alias; they become roughness instead).
fn sea_footprint(p: vec3<f32>) -> f32 {
  let v = to_world(p) - U.scene.xyz;
  let d = length(v);
  let c = max(abs(v.y) / max(d, 1e-4), 0.02);
  return d * U.ocx[3].w / pow(c, 0.75);
}

// How much of the surface at p is the sea's own: all of the open water, and in the box where its
// surface lies near the sea's (not a splash or a sheet thrown above it).
fn sea_share(p: vec3<f32>) -> f32 {
  if (!ocean_on()) { return 0.0; }
  if (any(p.xz < vec2<f32>(0.0)) || any(p.xz > U.n.xz) || side_in(p) < 1.0) { return 1.0; }
  return 1.0 - smoothstep(0.5, 2.0, abs(p.y - lvl_box(p)) * U.n.w / max(U.ocn2.z, 0.05));
}

fn sea_normal(p: vec3<f32>) -> vec3<f32> { return ocean_normal(sea_xz(p), sea_fade(p)); }

// The sea's short waves on the simulated surface in the box, where it is the sea's own surface.
fn sea_box_normal(p: vec3<f32>, nrm: vec3<f32>) -> vec3<f32> {
  if (!ocean_on()) { return nrm; }
  let near = sea_share(p);
  if (near <= 0.0) { return nrm; }
  let sl = ocean_chop_slope(sea_xz(p), sea_fade(p)) * near * max(nrm.y, 0.0);
  return normalize(nrm - vec3<f32>(sl.x, 0.0, sl.y));
}

// A glint with an explicit microfacet width (a2: GGX alpha squared).
fn sea_glint(nrm: vec3<f32>, v: vec3<f32>, l: vec3<f32>, a2: f32) -> f32 {
  let nl = dot(nrm, l);
  if (nl <= 0.0) { return 0.0; }
  let nv = max(dot(nrm, v), 1e-4);
  let hv = normalize(l + v);
  let nh = max(dot(nrm, hv), 0.0);
  let dd = nh * nh * (a2 - 1.0) + 1.0;
  let D = a2 / (PI * dd * dd);
  let vis = 0.5 / (nl * sqrt(nv * nv * (1.0 - a2) + a2) + nv * sqrt(nl * nl * (1.0 - a2) + a2));
  return D * vis * fresnel(dot(v, hv), 1.0, U.optic.x) * nl;
}

// Microfacet width of the sea at p: the surface's own roughness, the waves too small to draw (their
// slopes, the stronger in a gust), and the sun's size.
fn sea_a2(p: vec3<f32>) -> f32 {
  let r = max(U.optic.y, 0.02);
  let x = sea_xz(p);
  let fp = sea_fade(p);
  return min(r * r * r * r + ocean_rough2(fp) * ocean_gust(x) + U.sun.w * U.sun.w, 1.0);
}

// What a reflection off the sea sees: a collider, or the environment (the sky, the set's HDRI, the
// footage as far as it shows in reflections). Not the footage behind the point, as a ray through
// the water would: that is what lies beyond it, not above it.
fn sea_sky(p: vec3<f32>, rr: vec3<f32>, lg: vec3<f32>) -> vec3<f32> {
  let ts = trace_solid(p, rr, U.scene.w / U.n.w);
  if (ts >= 0.0) { return stand_in(p + rr * ts, rr, lg); }
  return env(to_world_dir(rr));
}

// What a reflection off the box's surface sees: reflected() traces it on through the box, but where
// the surface is the sea's own it sees the sky over the next wave, as on the open water (share: how
// much of the surface is the sea's, sea_share).
fn sea_reflected(p: vec3<f32>, rr: vec3<f32>, lg: vec3<f32>, share: f32) -> vec3<f32> {
  // (each looked up in one place: the driver inlines every call, and the march is large)
  var r = vec3<f32>(0.0);
  var above = vec3<f32>(0.0);
  if (share < 0.999) { r = reflected(p, rr, lg); }
  if (share > 0.0) { above = sea_sky(p, rr, lg); }
  if (share >= 0.999) { return above; }
  if (share <= 0.0) { return r; }
  return mix(r, above, share);
}

// The key light's glint on the box's surface, as wide as the open water's where it is the sea's own.
fn sea_glint_box(p: vec3<f32>, nrm: vec3<f32>, v: vec3<f32>, lg: vec3<f32>, share: f32) -> f32 {
  if (share <= 0.0) { return glint(nrm, v, lg); }
  return mix(glint(nrm, v, lg), sea_glint(nrm, v, lg, sea_a2(p)), share);
}

// Sunlight through the thin top of a wave toward the eye.
fn sea_crest_glow(p: vec3<f32>, nrm: vec3<f32>, rd: vec3<f32>, lg: vec3<f32>) -> vec3<f32> {
  let k = U.ocx[6].w;
  if (k <= 0.0 || !ocean_on()) { return vec3<f32>(0.0); }
  let crest = max(U.ocn2.z * 0.45, 0.01);
  let up = smoothstep(-0.2, 1.0, (p.y - U.lvl.x) * U.n.w / crest);
  let towards = pow(max(dot(lg, rd), 0.0), 4.0);       // looking toward the sun, through the wave
  let facing = 0.5 + 0.5 * max(-dot(lg, nrm), 0.0);    // the face turned from the sun lets it through
  let graze = 1.0 - rd.y * rd.y;
  let lit = U.sun.rgb * (0.03 + 0.35 * towards * facing) * graze + U.sky.rgb * 0.03;
  return U.ocx[6].rgb * lit * (up * k);
}

// The open water past the box at p, seen along rd, with its normal nrm (sea_normal with any rain
// rings and ripples on it).
// A shading normal turned away from the eye (the back of a sharp crest seen at a graze) is really
// the surface's silhouette: turn it just enough to face the eye.
fn sea_facing(nrm: vec3<f32>, rd: vec3<f32>) -> vec3<f32> {
  let ndv = dot(nrm, -rd);
  if (ndv >= 0.03) { return nrm; }
  return normalize(nrm + rd * (ndv - 0.03));
}

fn sea_open_shade(p: vec3<f32>, rd: vec3<f32>, nrm_in: vec3<f32>, lg: vec3<f32>) -> Shade {
  let nrm = sea_facing(nrm_in, rd);
  let F = fresnel(clamp(-dot(rd, nrm), 0.0, 1.0), 1.0, U.optic.x);
  var spec = U.sun.rgb * (sea_glint(nrm, -rd, lg, sea_a2(p)) * PI);
  if (lamps_on()) { spec += lamps_glint(U.org.xyz + p * U.n.w, nrm, -rd); }
  let below = sea_below(p, refract(rd, nrm, 1.0 / U.optic.x), lg);
  var rr = reflect(rd, nrm);
  rr = normalize(vec3<f32>(rr.x, max(rr.y, 0.02), rr.z));   // it would meet the next wave: the sky just above it
  var sh: Shade;
  sh.col = F * U.optic.z * sea_sky(p + nrm * 0.01, rr, lg) + spec
         + (1.0 - F) * (below + sea_crest_glow(p, nrm, rd, lg));
  sh.spec = spec;
  sh.bub = 0.0;
  return sh;
}

// What a ray going down into the open water sees: over the shallows the sea bed (sand, or the footage
// where the colliders are in it), through the water between; elsewhere what the open water shows (land).
fn sea_below(p: vec3<f32>, d: vec3<f32>, lg: vec3<f32>) -> vec3<f32> {
  let depth = lay_at(sea_xz(p)).z;
  // (only where something raises the bed: over the flat ground the footage shows it)
  let level_m = U.lvl.x * U.n.w;
  // (a bed the layer's map puts above the level, under water here all the same, is only just under: the
  // edge of the water on a beach)
  let in_open = d.y >= -1e-4 || depth > 1.0e4 || depth > 60.0 || abs(depth - level_m) < 0.03 * level_m + 0.02;
  var below_m = 0.0;   // from this point of the surface down to the bed
  var L = 0.0;         // metres along the ray
  var pb = p;
  if (!in_open) {
    below_m = max(depth + (p.y - U.lvl.x) * U.n.w, 0.02);
    L = below_m / max(-d.y, 0.05);
    pb = p + d * (L / U.n.w);
  }
  // the bed as the box shows it (a collider stand-in, or the footage), lit through the water; else what
  // the open water shows (one call of land for both: it is large, and inlined wherever it is called)
  var bed = land(select(LAND_SOLID, LAND_OPEN, in_open), pb, d, lg);
  if (in_open) { return bed; }
  if (U.org.w > 0.5 || U.plate.x < 0.5) {
    bed *= exp(-dot(U.absorb.rgb, vec3<f32>(0.333)) * below_m * 0.5);
  }
  let mp = murk_path(L);
  return bed * mp[1] + mp[0];
}

// Foam lines along the wind (windrows): 0 between them .. about 2 on them, 1 on average.
fn sea_streaks(x: vec2<f32>) -> f32 {
  let amt = U.ocx[4].y;
  if (amt <= 0.0) { return 1.0; }
  let wd = U.ocx[5].xy;
  let across = vec2<f32>(-wd.y, wd.x);
  let q = vec2<f32>(dot(x, wd) / 45.0, dot(x, across) / 3.0);
  let r = 1.0 - abs(oc_noise(vec3<f32>(q, 0.3)) + 0.5 * oc_noise(vec3<f32>(q * vec2<f32>(2.1, 2.7) + vec2<f32>(3.1, 8.2), 1.7)));
  return mix(1.0, 0.15 + 2.2 * r * r * r, clamp(amt, 0.0, 1.0));
}

// Thin foam of density d at the undisplaced point x0: how much of the surface it covers (0..1).
// Foam gathers first on the walls between big bubbles and along the edges of drifting patches, so
// thin foam is a lace of filaments and thickens into sheets with holes. The pattern is warped and
// stretched along the wind, and its fine lace fades out where a pixel covers it.
fn sea_lace(x0: vec2<f32>, d: f32, fp: f32) -> f32 {
  if (d <= 0.01) { return 0.0; }
  let wd = U.ocx[5].xy;
  let across = vec2<f32>(-wd.y, wd.x);
  // stretched along the wind: thin foam is drawn out into filaments and streaks
  let q = vec2<f32>(dot(x0, wd) * 0.3, dot(x0, across)) / 0.6;
  let warp = vec2<f32>(oc_noise(vec3<f32>(q * 0.6, 1.3)), oc_noise(vec3<f32>(q * 0.6 + vec2<f32>(4.7, 2.2), 7.9)));
  let w = worley(vec3<f32>(q + 0.8 * warp, 0.37));
  let walls = 1.0 - smoothstep(0.0, 0.3, w.y - w.x);
  let patches = 0.5 + 0.5 * oc_noise(vec3<f32>(q * vec2<f32>(0.15, 0.4) + vec2<f32>(9.1, 3.7), 2.9));
  var f = 0.25 * walls + 0.75 * patches;
  let fine_w = 1.0 - smoothstep(0.03, 0.12, fp);
  if (fine_w > 0.0) {
    let w2 = worley(vec3<f32>((q + warp) * 3.7 + vec2<f32>(5.3, 1.9), 2.1));
    f += 0.18 * fine_w * ((1.0 - smoothstep(0.0, 0.3, w2.y - w2.x)) - 0.5);
  }
  // farther away the lace blurs into an even, fainter cover
  let far = smoothstep(0.08, 0.6, fp);
  let dd = clamp(0.9 * d, 0.0, 0.85);
  let t = 1.0 - dd;
  let lace = smoothstep(t - 0.06, t + 0.1, f);
  return mix(lace, dd * 0.5, far);
}

// Fresh foam of strength t (1 where the crest is breaking now, fading behind it) at x0: churned
// and bubbly, breaking up at its edges and into clumps as it fades.
fn sea_fresh(x0: vec2<f32>, t: f32, fp: f32) -> f32 {
  if (t <= 0.02) { return 0.0; }
  let q = x0 / 0.45;
  let warp = vec2<f32>(oc_noise(vec3<f32>(q * 0.8, 3.1)), oc_noise(vec3<f32>(q * 0.8 + vec2<f32>(2.3, 7.7), 5.3)));
  let near = 1.0 - smoothstep(0.02, 0.12, fp);
  var f = 0.55 + 0.35 * oc_noise(vec3<f32>(q + warp, 1.1)) + 0.2 * near * oc_noise(vec3<f32>(q * 2.7 + warp * 1.5, 4.4));
  let w = worley(vec3<f32>((q + warp) * 1.9, 6.5));
  f -= 0.3 * near * (1.0 - smoothstep(0.05, 0.35, w.x));   // the bigger bubbles are holes
  let thr = 1.0 - clamp(t, 0.0, 1.0);
  return smoothstep(thr - 0.1, thr + 0.2, f);
}

// The surface's colour with foam over it at p: the simulation's own foam (cover_in, from its
// whitewater) and the sea's. Returns the colour and the foam cover (0..1).
fn sea_surface(p: vec3<f32>, nrm: vec3<f32>, rd: vec3<f32>, lg: vec3<f32>, col: vec3<f32>, cover_in: f32) -> vec4<f32> {
  var c = col;
  var cover = cover_in;
  // (in the box the sea's foam lies only where the box's surface is still the sea's own: not on the
  // steep face of a wave the simulation is breaking, where the sea's foam pattern, laid on from above,
  // came out as strokes and flecks down the face)
  let open = 1.0 - smoothstep(1.5, U.lvl.w + 1.5, side_in(p));
  let share = sea_share(p) * mix(smoothstep(0.7, 0.92, nrm.y), 1.0, open);
  if (share > 0.0 && U.ocn2.y > 0.0) {
    let x = sea_xz(p);
    let fp = sea_fade(p);
    let x0 = ocean_x0(x, fp);
    let f = ocean_foam(x0, fp) * U.ocx[4].x;
    let fold = ocean_fold(x, fp) * U.ocx[4].x;
    let F = fresnel(clamp(-dot(rd, nrm), 0.0, 1.0), 1.0, U.optic.x);
    // bubbles churned under a breaking crest: a pale turquoise glow through the surface
    let bub = clamp(max(f.z, 0.6 * fold) * U.ocx[4].w, 0.0, 1.0) * share;
    c += (1.0 - F) * bub * 0.45 * (U.sky.rgb + U.sun.rgb * 0.35) * vec3<f32>(0.55, 0.9, 0.85);
    // fresh foam: dense and white where the crest breaks, breaking up behind it
    var fc = sea_fresh(x0, max(fold, f.x), fp) * share;
    // surf: where waves break over the shallows, white water on their crests
    let la = lay_at(x);
    // (in the box the simulation breaks the waves itself: only on the open water past it)
    let surf = ocean_surf(la.z) * U.ocx[4].x * open;
    if (surf > 0.0) {
      // white water rides the top of each breaking crest, in torn patches (the soup the broken waves
      // leave behind is the layer's foam, drifting)
      let hloc = min(U.ocx[9].z * oc_shoal(U.ocx[9].w, la.z), 0.6 * la.z);
      let rel = (p.y - U.lvl.x) * U.n.w / max(0.5 * hloc, 0.02);
      fc = max(fc, sea_fresh(x0, 0.4 * surf * smoothstep(0.4, 0.9, rel), fp) * share);
    }
    // aged foam: thin, in streaks along the wind, broken into lace; the water shows through it (with
    // the foam the layer has carried out of the box and off the surf: past the box only, in it the
    // simulation's own foam is drawn)
    let aged = f.y * sea_streaks(x) + 0.6 * la.y * U.ocx[4].x * open;
    let ac = sea_lace(x0, aged, fp) * 0.7 * share;
    // thick foam is brighter than thin (more bubble walls scatter the light back)
    let tone = 0.9 + 0.1 * oc_noise(vec3<f32>(x0 / 0.3, 8.1));
    let fl = U.foam.rgb * ww_light(nrm, lg);
    c = mix(c, fl * 0.86 * tone, ac);
    c = mix(c, fl * tone, fc);
    let sea_cover = 1.0 - (1.0 - fc) * (1.0 - ac);
    // the simulation's own foam over it
    if (cover > 0.0) { c = mix(c, fl, cover); }
    return vec4<f32>(c, 1.0 - (1.0 - cover) * (1.0 - sea_cover));
  }
  if (cover > 0.0) {
    c = mix(c, U.foam.rgb * ww_light(nrm, lg), cover);
  }
  return vec4<f32>(c, cover);
}
