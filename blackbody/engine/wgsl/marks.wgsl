// What bullets leave behind (engine/ballistics.py): the holes and craters themselves, carved out of whatever they are in
// (bores), and the marks round them drawn into the surface's material (marks), on the stage's objects, broken pieces
// and floor alike. The includer declares MK: array<vec4<f32>> (shot_draw.marks_buffer packs it):
//   MK[0]: how many marks, how many bores, where the bores start (in vec4), _
//   a mark, four vec4 from MK[1]: [0] its centre on the surface (fire-local m), its radius (m); [1] the surface's normal
//   there (toward where the bullet came from), its style; [2] the bullet's direction (its way off a glance), a random
//   number; [3] the grain (wood; 0 elsewhere), 1 on the far side (where the bullet came out) + 2 x what it is in (CLASS)
//   a bore, four vec4: [0] a (m), its radius there; [1] b, its radius there; [2] the grain (or 0), a random number;
//   [3] how much longer it is along the grain than across, how ragged its edge is, what it is in, how many ragged
//   lobes round it
// What it is in (ballistics.CLASS): 0 anything, 1 concrete, 2 stone, 3 brick, 4 plaster, 5 wood, 6 bare metal,
// 7 painted metal, 8 glass and ice, 9 pottery, 10 earth, 11 plastic, rubber, cardboard, foam, cloth.
//
// The bores make the geometry: each a cone from a to b, its edge ragged (noise round it and down it), and longer along
// wood's grain, a torn split: a hole through a board, a car's door or a pane is a hole (what is behind it shows through,
// light comes through it), a crater has walls and a floor that take the light and shade themselves. Their walls are drawn
// as the thing is inside, freshly broken: concrete's cement and its stones, granite's crystals, brick's fired clay,
// plaster's chalk, wood's torn pale fibres, bare bright metal, glass crushed white.
// The marks are what is round them: concrete's powder and hairline cracks, the grey ring a bullet wipes off on its way
// in, paint chipped off round a hole in a car, the polished dish of a dent, glass's white cone and its web of cracks,
// wood's short splits along the grain and its splinters lifted round the hole out of the back, a glancing bullet's
// furrow, the star of lead a bullet splashes over a steel plate.

fn mk_hash(x: f32) -> f32 { return fract(sin(x * 127.1 + 311.7) * 43758.5453); }

// A thin line of half-width w (m) at distance d (m), softened over the pixel: 1 on it.
fn mk_line(d: f32, w: f32, fw: f32) -> f32 {
  return 1.0 - smoothstep(w, w + fw, abs(d));
}

// Cellular noise round p: (the distance to the nearest cell's point, a random number of that cell's).
fn mk_cells(p: vec3<f32>) -> vec2<f32> {
  let i = floor(p);
  var best = vec2<f32>(9.0, 0.0);
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let c = i + vec3<f32>(f32(x), f32(y), f32(z));
        let h = vec3<f32>(pcg3d(vec3<u32>(vec3<i32>(c) + vec3<i32>(1048576)))) * (1.0 / 4294967296.0);
        let d = length(c + h - p);
        if (d < best.x) { best = vec2<f32>(d, h.x); }
      }
    }
  }
  return best;
}

// What a thing is like freshly broken (a bore's wall, a crater's floor): its albedo (rgb), roughness (w), where `inner`
// is its inside colour, at p (m), along its grain g (wood), at a pixel's footprint fw.
fn fresh(cls: i32, p: vec3<f32>, g: vec3<f32>, inner: vec3<f32>, fw: f32) -> vec4<f32> {
  let fine = 1.0 - smoothstep(0.0003, 0.0012, fw);
  if (cls == 1) {
    // concrete: grey cement paste, its stones (5-20 mm) broken through, sand in it
    let c = mk_cells(p * 90.0);
    let stone = 1.0 - smoothstep(0.32, 0.45, c.x);
    let tone = mix(vec3<f32>(0.62, 0.6, 0.56), vec3<f32>(0.42, 0.38, 0.33), c.y) * mix(1.0, 0.8, step(0.7, c.y));
    let paste = inner * 1.12 * (0.92 + 0.15 * gnoise(p * 600.0) * fine);
    return vec4<f32>(mix(paste, tone, stone * 0.85), 0.95);
  }
  if (cls == 2) {
    // granite: its crystals, white, grey, black and pink
    let c = mk_cells(p * 700.0);
    let tone = mix(mix(vec3<f32>(0.75, 0.72, 0.7), vec3<f32>(0.12), step(0.6, c.y)), vec3<f32>(0.62, 0.45, 0.4), step(0.88, c.y));
    return vec4<f32>(mix(inner * 1.2, tone, 0.8 * fine + 0.2), 0.8);
  }
  if (cls == 3) {
    // brick: fresh fired clay, a brighter orange than its weathered face, grainy
    return vec4<f32>(inner * (1.15 + 0.2 * gnoise(p * 900.0) * fine), 0.95);
  }
  if (cls == 4) {
    // plaster: white chalk, crumbly
    return vec4<f32>(min(inner * 1.15 + vec3<f32>(0.05), vec3<f32>(0.9)) * (0.95 + 0.08 * gnoise(p * 500.0)), 1.0);
  }
  if (cls == 5) {
    // wood: torn pale fibres along the grain, light and dark strands
    let gg = select(vec3<f32>(0.0, 1.0, 0.0), normalize(g), dot(g, g) > 1e-6);
    let along = dot(p, gg);
    let a1 = normalize(cross(gg, select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(gg.y) > 0.9)));
    let across = vec2<f32>(dot(p, a1), dot(p, cross(gg, a1)));
    let f = gnoise(vec3<f32>(across * 1800.0, along * 40.0)) * fine + 0.5 * gnoise(vec3<f32>(across * 400.0, along * 10.0));
    return vec4<f32>(min(inner * (1.08 + 0.35 * f), vec3<f32>(0.9)), 0.9);
  }
  if (cls == 8) {
    // glass and ice: crushed white
    return vec4<f32>(vec3<f32>(0.72, 0.75, 0.75) * (0.9 + 0.2 * gnoise(p * 3000.0) * fine), 0.5);
  }
  if (cls == 9) {
    // pottery: its fired body, paler than its glaze
    return vec4<f32>(inner * (1.0 + 0.1 * gnoise(p * 800.0) * fine), 0.95);
  }
  if (cls == 10) {
    // earth: darker and damper below its dry face, crumbs
    let c = mk_cells(p * 400.0);
    return vec4<f32>(inner * 0.7 * (0.85 + 0.3 * c.y), 1.0);
  }
  return vec4<f32>(inner * (0.95 + 0.1 * gnoise(p * 900.0) * fine), 0.95);
}

// ---- the marks round them ---------------------------------------------------------------------------------------

fn bullet_marks(s_in: Surf, inner: vec3<f32>, fw_in: f32) -> Surf {
  var s = s_in;
  let count = u32(MK[0].x + 0.5);
  if (count == 0u) { return s; }
  let fw = max(fw_in, 1.0e-5);
  let metal = max(s.f0.r, max(s.f0.g, s.f0.b)) > 0.3;
  for (var i = 0u; i < count; i++) {
    let A = MK[1u + 4u * i];
    let B = MK[2u + 4u * i];
    let C = MK[3u + 4u * i];
    let D = MK[4u + 4u * i];
    let R = max(A.w, 1.0e-4);
    let style = i32(B.w + 0.5);
    let exit_side = fract(D.w * 0.5) > 0.25;
    let cls = i32(floor(D.w * 0.5 + 0.01));
    var reach = R * 3.0;
    if (style == 4) { reach = R * 9.0; }
    if (style == 5) { reach = R * select(4.0, 6.0, exit_side); }
    if (style == 6) { reach = R * 8.0; }
    let dp = s.p - A.xyz;
    if (dot(dp, dp) > reach * reach) { continue; }
    let off = dot(dp, B.xyz);
    if (abs(off) > max(0.3 * R, 0.004) + 2.0 * fw || dot(s.n, B.xyz) < 0.25) { continue; }
    let t = dp - B.xyz * off;
    let r = length(t);
    // the mark's own axes on the surface: along the grain (wood), the bullet's way (a skid), or anywhere
    var ua = D.xyz - B.xyz * dot(D.xyz, B.xyz);
    if (dot(ua, ua) < 1e-6) { ua = C.xyz - B.xyz * dot(C.xyz, B.xyz); }
    if (dot(ua, ua) < 1e-6) { ua = cross(B.xyz, vec3<f32>(0.0, 1.0, 0.0)); }
    if (dot(ua, ua) < 1e-6) { ua = cross(B.xyz, vec3<f32>(1.0, 0.0, 0.0)); }
    ua = normalize(ua);
    let va = cross(B.xyz, ua);
    let x = dot(t, ua);
    let y = dot(t, va);
    let ang = atan2(y, x);
    let seed = C.w * 97.0;
    let cs = vec2<f32>(cos(ang), sin(ang));

    if (style == 1) {
      // round a crater (its bowl is the bore's): the powder it threw out, paler, thinning outward in tongues; a few
      // chips knocked off the edge; hairline cracks running out
      let tongue = 0.75 + 0.5 * gnoise(vec3<f32>(cs * 2.5, seed));
      let dust = (1.0 - smoothstep(R * 0.9, R * (1.8 + 0.9 * tongue), r)) * smoothstep(R * 0.75, R * 0.95, r);
      let speck = 0.6 + 0.4 * gnoise(vec3<f32>(vec2<f32>(x, y) / R * 14.0, seed));
      s.alb = mix(s.alb, min(s.alb * 1.45 + vec3<f32>(0.06), vec3<f32>(0.9)), 0.6 * dust * speck);
      s.rough = mix(s.rough, 1.0, dust);
      // chips: small shallow pits round the rim
      let ch = mk_cells(vec3<f32>(vec2<f32>(x, y) / R * 3.5, seed));
      let pit = (1.0 - smoothstep(0.12, 0.2, ch.x)) * step(0.55, ch.y) * (1.0 - smoothstep(R * 1.0, R * 1.6, r))
                * smoothstep(R * 0.85, R * 1.0, r);
      s.alb = mix(s.alb, fresh(cls, s.p, D.xyz, inner, fw).rgb, pit);
      var crack = 0.0;
      let K = 3 + i32(mk_hash(seed) * 4.0);
      for (var k = 0; k < K; k++) {
        let a0 = mk_hash(seed + f32(k) * 7.31) * 6.2831853;
        let len = R * (1.3 + 1.8 * mk_hash(seed + f32(k) * 3.7));
        let dir = vec2<f32>(cos(a0), sin(a0));
        let along = dot(vec2<f32>(x, y), dir);
        let wob = 0.05 * R * (gnoise(vec3<f32>(along / R * 5.0, f32(k), seed)) + 0.4 * gnoise(vec3<f32>(along / R * 19.0, f32(k), seed)));
        let across = dot(vec2<f32>(x, y), vec2<f32>(-dir.y, dir.x)) + wob;
        if (along > R * 0.9 && along < len) {
          let taper = 1.0 - (along - 0.9 * R) / max(len - 0.9 * R, 1e-6);
          crack = max(crack, mk_line(across, 0.00012 * taper + 0.00003, fw) * taper);
        }
      }
      s.alb = mix(s.alb, s.alb * 0.4, 0.8 * crack);
    } else if (style == 2) {
      // round a hole: the grey wipe of the bullet's lead and grease round the way in; paint chipped off round it, bare
      // metal under it; the lip pushed in (out, on the far side)
      let rim = R * (1.0 + 0.12 * gnoise(vec3<f32>(cs * 3.0, seed)));
      let wipe = (1.0 - smoothstep(rim * 1.05, rim * 1.5, r)) * smoothstep(rim * 0.9, rim * 1.0, r) * select(1.0, 0.25, exit_side);
      if (cls == 7) {
        let chip = (1.0 - smoothstep(rim * 1.4, rim * 1.7, r * (1.0 + 0.3 * gnoise(vec3<f32>(cs * 5.0, seed + 5.0)))))
                   * smoothstep(rim * 0.95, rim * 1.0, r);
        s.alb = mix(s.alb, vec3<f32>(0.0), chip);
        s.f0 = mix(s.f0, vec3<f32>(0.55, 0.56, 0.57), chip);
        s.rough = mix(s.rough, 0.3, chip);
        // (primer showing at the chip's edge: a thin grey ring)
        let primer = mk_line(r - rim * 1.55 * (1.0 + 0.3 * gnoise(vec3<f32>(cs * 5.0, seed + 5.0))), 0.0004, fw);
        s.alb = mix(s.alb, vec3<f32>(0.45, 0.45, 0.43), 0.6 * primer);
      }
      s.alb = mix(s.alb, s.alb * 0.3 + vec3<f32>(0.02), 0.85 * wipe);
      s.f0 = mix(s.f0, vec3<f32>(0.04), 0.5 * wipe);
      let lip = smoothstep(rim, rim * 1.05, r) * (1.0 - smoothstep(rim * 1.05, rim * 1.5, r));
      s.n = normalize(s.n + (t / max(r, 1e-6)) * select(0.35, -0.6, exit_side) * lip);
    } else if (style == 3) {
      // round a dent (its dish is the bore's): the paint blown off round it shows the bare metal, the metal round it
      // pushed up a little
      let rr = r / R;
      let bare = (1.0 - smoothstep(1.2, 1.7, rr * (1.0 + 0.25 * gnoise(vec3<f32>(cs * 4.0, seed))))) * smoothstep(0.9, 1.0, rr);
      if (cls == 7) {
        s.alb = mix(s.alb, vec3<f32>(0.0), bare);
        s.f0 = mix(s.f0, vec3<f32>(0.55), bare);
        s.rough = mix(s.rough, 0.3, bare);
      }
      s.n = normalize(s.n + (t / max(r, 1e-6)) * 0.25 * smoothstep(0.95, 1.05, rr) * (1.0 - smoothstep(1.05, 1.4, rr)));
    } else if (style == 4) {
      // glass, pottery, ice: a white cone of crushed stuff round the hole, cracks radiating from it, crossed by rings
      let crush = (1.0 - smoothstep(R * 0.6, R * 1.15 * (1.0 + 0.25 * gnoise(vec3<f32>(cs * 5.0, seed + 2.0))), r));
      var crack = 0.0;
      let K = 9 + i32(mk_hash(seed) * 6.0);
      for (var k = 0; k < K; k++) {
        let a0 = (f32(k) + 0.6 * mk_hash(seed + f32(k))) * 6.2831853 / f32(K);
        let len = R * (3.0 + 6.0 * mk_hash(seed * 1.7 + f32(k)));
        let dir = vec2<f32>(cos(a0), sin(a0));
        let along = dot(vec2<f32>(x, y), dir);
        let kink = 0.05 * along * gnoise(vec3<f32>(along / R * 1.5, f32(k) * 3.1, seed));
        let across = dot(vec2<f32>(x, y), vec2<f32>(-dir.y, dir.x)) + kink;
        if (along > R * 0.4 && along < len) {
          crack = max(crack, mk_line(across, 0.00015, fw) * (1.0 - 0.6 * along / len));
        }
      }
      for (var j = 1; j < 4; j++) {
        let rj = R * (1.2 + 0.9 * f32(j)) * (1.0 + 0.1 * gnoise(vec3<f32>(cs * 2.0, seed + f32(j))));
        let keep = step(0.15, gnoise(vec3<f32>(cs * 3.0, seed * 2.0 + f32(j))) + 0.3);
        crack = max(crack, mk_line(r - rj, 0.00012, fw) * keep * 0.8);
      }
      let white = vec3<f32>(0.75, 0.78, 0.78);
      s.alb = mix(s.alb, white, 0.65 * crush);
      s.alb = mix(s.alb, max(s.alb, vec3<f32>(0.55)), crack);
      s.rough = mix(s.rough, 0.6, max(crush, 0.6 * crack));
      s.n = normalize(s.n + (va * gnoise(vec3<f32>(vec2<f32>(x, y) / R * 20.0, seed)) + ua * gnoise(vec3<f32>(vec2<f32>(x, y) / R * 20.0, seed + 9.0))) * 0.4 * crush);
    } else if (style == 5) {
      // wood: going in, the fibres crushed in round the hole, the bullet's grey wipe, short splits along the grain from
      // it; coming out, torn along the grain, splinters lifted round it (pale, fresh, each with its shadowed edge)
      let along_k = select(1.15, 2.2, exit_side);
      let e = vec2<f32>(x / along_k, y);
      let re = length(e);
      let fib = gnoise(vec3<f32>(x / R * 0.9, y / R * 16.0, seed));
      let wipe = (1.0 - smoothstep(R * 1.0, R * 1.35, re)) * smoothstep(R * 0.85, R * 1.0, re) * select(1.0, 0.0, exit_side);
      s.alb = mix(s.alb, s.alb * 0.35, 0.75 * wipe);
      // splits along the grain, up and down from the hole
      var split = 0.0;
      for (var k = 0; k < 2; k++) {
        let side = select(-1.0, 1.0, k == 0);
        let len = R * (2.0 + 4.0 * mk_hash(seed + f32(k) * 5.1)) * select(1.0, 1.6, exit_side);
        let off_y = 0.25 * R * (mk_hash(seed + f32(k) * 2.3) - 0.5);
        let ax = side * x;
        if (ax > 0.7 * R && ax < len) {
          let w = 0.00012 * (1.0 - (ax - 0.7 * R) / max(len - 0.7 * R, 1e-6)) + 0.00002;
          split = max(split, mk_line(y - off_y - 0.05 * R * gnoise(vec3<f32>(ax / R * 3.0, f32(k), seed)), w, fw));
        }
      }
      s.alb = mix(s.alb, s.alb * 0.25, 0.85 * split);
      // lifted splinters out of the back: strips along the grain, pale on top, a dark gap at their edges
      let strip = smoothstep(0.2, 0.55, gnoise(vec3<f32>(x / R * 0.6, y / R * 7.0, seed + 4.0)));
      let flap = (1.0 - smoothstep(R * 1.0, R * select(1.8, 3.4, exit_side), re)) * smoothstep(R * 0.9, R * 1.05, re) * strip;
      let edge = mk_line(fract(y / R * 7.0 / 6.2831853 * 6.0 + fib) - 0.5, 0.06, 0.05) * flap;
      s.alb = mix(s.alb, fresh(5, s.p, D.xyz, inner, fw).rgb, 0.85 * flap);
      s.alb = mix(s.alb, s.alb * 0.3, 0.6 * edge);
      s.rough = mix(s.rough, 0.95, flap);
      s.n = normalize(s.n + va * (0.5 * flap * fib) + (t / max(r, 1e-6)) * select(0.0, -0.5, exit_side) * flap);
    } else if (style == 6) {
      // a skid (its furrow is the bore's): scraped bright along it on metal, pale on stone, lead-grey toward its end
      let L = R * 5.0;
      let along = clamp(x, 0.0, L);
      let w = R * 0.6 * (1.0 - 0.4 * along / L) * (1.0 + 0.2 * gnoise(vec3<f32>(x / R * 3.0, 0.0, seed)));
      let band = (1.0 - smoothstep(w - fw, w + fw, length(vec2<f32>(x - along, y)))) * step(-R * 0.3, x);
      if (metal) {
        s.rough = mix(s.rough, 0.1, band);
        s.f0 = mix(s.f0, max(s.f0, vec3<f32>(0.7)), 0.6 * band);
      } else {
        s.alb = mix(s.alb, fresh(cls, s.p, D.xyz, inner, fw).rgb, 0.8 * band);
        s.rough = mix(s.rough, 1.0, band);
      }
      let lead = band * smoothstep(L * 0.4, L, x);
      s.alb = mix(s.alb, vec3<f32>(0.28, 0.28, 0.29), 0.6 * lead);
    } else if (style == 7) {
      // the lead's star: a bullet splashed flat on steel throws its lead out in streaks over the plate
      let spokes = pow(clamp(0.5 + gnoise(vec3<f32>(cs * 4.0, seed)) + 0.3 * gnoise(vec3<f32>(cs * 11.0, seed + 3.0)), 0.0, 1.0), 3.0);
      let reach_a = R * (0.3 + 0.9 * spokes);
      let star = (1.0 - smoothstep(reach_a - fw, reach_a + fw, r)) * (0.75 + 0.25 * gnoise(vec3<f32>(vec2<f32>(x, y) / R * 30.0, seed)));
      let core = 1.0 - smoothstep(R * 0.22, R * 0.3, r);
      let grey = vec3<f32>(0.32, 0.32, 0.33) * (0.8 + 0.3 * gnoise(vec3<f32>(vec2<f32>(x, y) / R * 10.0, seed)));
      s.alb = mix(s.alb, grey, 0.9 * star);
      s.f0 = mix(s.f0, vec3<f32>(0.25), 0.8 * star);
      s.rough = mix(s.rough, 0.55, star);
      s.alb = mix(s.alb, vec3<f32>(0.18), core);
    }
  }
  return s;
}

// ---- bores: the holes and craters themselves -----------------------------------------------------------------------

fn bores_on() -> bool { return F_SHOTS && MK[0].y > 0.5; }
fn marks_on() -> bool { return F_SHOTS && MK[0].x > 0.5; }   // (F_SHOTS: stage.wgsl; false, all of it compiles out)

// The distance from p to a capped cone from a (radius ra) to b (radius rb), negative inside (Quilez's, exact).
fn sd_cone(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, ra: f32, rb: f32) -> f32 {
  let rba = rb - ra;
  let baba = dot(b - a, b - a);
  let papa = dot(p - a, p - a);
  let paba = dot(p - a, b - a) / max(baba, 1e-12);
  let x = sqrt(max(papa - paba * paba * baba, 0.0));
  let cax = max(0.0, x - select(rb, ra, paba < 0.5));
  let cay = abs(paba - 0.5) - 0.5;
  let k = rba * rba + baba;
  let f = clamp((rba * (x - ra) + paba * baba) / max(k, 1e-12), 0.0, 1.0);
  let cbx = x - ra - f * rba;
  let cby = paba - f;
  let s = select(1.0, -1.0, cbx < 0.0 && cay < 0.0);
  return s * sqrt(min(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba));
}

// Bore j's distance at p: its cone, its radius at each angle round it longer along the grain (by elong) and ragged (by
// rough, with lobes round it and a little down it). (Not an exact distance: the steps by it are cut to 0.7 of it.)
fn bore_one(p: vec3<f32>, j: u32, first: u32) -> f32 {
  let A = MK[first + 4u * j];
  let B = MK[first + 4u * j + 1u];
  let G = MK[first + 4u * j + 2u];
  let H = MK[first + 4u * j + 3u];
  let u = normalize(B.xyz - A.xyz + vec3<f32>(1e-9, 0.0, 0.0));
  var g = G.xyz - u * dot(G.xyz, u);
  if (dot(g, g) < 1e-8) { g = cross(u, select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(u.y) > 0.9)); }
  g = normalize(g);
  let h = cross(u, g);
  let w = p - A.xyz;
  let along = dot(w, u) / max(length(B.xyz - A.xyz), 1e-6);
  let rv = w - u * dot(w, u);
  let ang = atan2(dot(rv, h), dot(rv, g));
  let c = cos(ang);
  let lobes = max(H.w, 1.0);
  let q = vec3<f32>(c * lobes / 3.0, sin(ang) * lobes / 3.0, along * 1.5 + G.w * 13.0);
  let rag = gnoise(q) + 0.45 * gnoise(q * 2.7 + vec3<f32>(5.2));
  // (torn more, and longer along the grain, toward its far end: a hole's way out is rougher than its way in)
  let k = clamp(along, 0.0, 1.0);
  let f = (1.0 + H.x * k * c * c) * max(1.0 + H.y * (0.35 + 0.65 * k) * rag, 0.3);
  // its walls broken, not smooth: chunks a few millimetres across (concrete, stone), fibres along the grain (wood)
  let scale = max(A.w, B.w);
  var chunk = 0.0;
  if (H.y > 0.2) {
    let wood = dot(G.xyz, G.xyz) > 1e-6;
    let along_g = dot(p, g);
    if (wood) {
      let qq = vec3<f32>(dot(p, h) / (0.06 * scale), dot(p, u) / (0.06 * scale), along_g / (0.8 * scale));
      chunk = H.y * 0.18 * scale * (gnoise(qq + vec3<f32>(G.w * 7.0)) + 0.5 * gnoise(qq * 2.3 + vec3<f32>(3.1)));
    } else {
      // (brittle: fractured into facets with sharp creases between them, not lumps: ridged noise)
      let qq = p / (0.4 * scale) + vec3<f32>(G.w * 7.0);
      let r1 = 1.0 - 2.0 * abs(gnoise(qq));
      let r2 = 1.0 - 2.0 * abs(gnoise(qq * 2.4 + vec3<f32>(3.1)));
      chunk = H.y * 0.16 * scale * (r1 + 0.45 * r2 - 0.6);
    }
  }
  return 0.7 * (sd_cone(p, A.xyz, B.xyz, A.w * f, B.w * f) + chunk);
}

// The distance from p to the nearest bore (negative inside one), or a large number with none near.
fn bore_d(p: vec3<f32>) -> f32 {
  var best = 1.0e9;
  let n = u32(MK[0].y + 0.5);
  let first = u32(MK[0].z + 0.5);
  for (var j = 0u; j < n; j++) {
    let A = MK[first + 4u * j];
    let B = MK[first + 4u * j + 1u];
    let H = MK[first + 4u * j + 3u];
    // (a quick bound first: the sphere round it, as big as its raggedest)
    let c = 0.5 * (A.xyz + B.xyz);
    let reach = 0.5 * length(B.xyz - A.xyz) + max(A.w, B.w) * (1.0 + H.x) * (1.0 + 1.5 * H.y);
    if (length(p - c) - reach > best) { continue; }
    best = min(best, bore_one(p, j, first));
  }
  return best;
}

// The nearest bore at p: where its record starts in MK.
fn bore_near(p: vec3<f32>) -> u32 {
  var best = 1.0e9;
  let n = u32(MK[0].y + 0.5);
  let first = u32(MK[0].z + 0.5);
  var out = first;
  for (var j = 0u; j < n; j++) {
    let d = bore_one(p, j, first);
    if (d < best) {
      best = d;
      out = first + 4u * j;
    }
  }
  return out;
}

// The outward normal of the solid round a bore at p (into the bore: down the slope of its distance).
fn bore_normal(p: vec3<f32>, e: f32) -> vec3<f32> {
  let a = vec3<f32>(1.0, -1.0, -1.0);
  let b = vec3<f32>(-1.0, -1.0, 1.0);
  let c = vec3<f32>(-1.0, 1.0, -1.0);
  let d = vec3<f32>(1.0, 1.0, 1.0);
  let g = a * bore_d(p + a * e) + b * bore_d(p + b * e) + c * bore_d(p + c * e) + d * bore_d(p + d * e);
  let l = length(g);
  return select(vec3<f32>(0.0, 1.0, 0.0), -g / l, l > 1e-12);
}

// A ray through a solid it enters at tn and leaves at tf (a convex object, a piece; the floor: tf far below): the
// first point on that stretch that is solid, past the bores (where it goes into a hole, the hole's wall it meets, or,
// through a hole, nothing): its distance along the ray, or -1.
fn bore_pass(ro: vec3<f32>, rd: vec3<f32>, tn: f32, tf: f32) -> f32 {
  if (!bores_on()) { return tn; }
  var t = tn;
  var d = bore_d(ro + rd * t);
  if (d >= 0.0) { return t; }
  for (var i = 0; i < 64; i++) {
    t += max(-d, max(1.0e-5, 0.15 * U.fit.w * t));
    if (t >= tf) { return -1.0; }
    d = bore_d(ro + rd * t);
    if (d >= 0.0) { return t; }
  }
  return t;
}

// A point of a surface the trace found: whether it is on a bore's wall (where a hole was carved), and that wall's normal
// (w: 1 on a wall, 0 not).
fn bore_wall(p: vec3<f32>, eps: f32) -> vec4<f32> {
  if (!bores_on()) { return vec4<f32>(0.0); }
  let d = bore_d(p);
  if (abs(d) > 2.0 * eps) { return vec4<f32>(0.0); }
  return vec4<f32>(bore_normal(p, max(0.5 * eps, 1.0e-5)), 1.0);
}

// A bore's wall, shaded as the thing is freshly broken inside (fresh()); bare metal bright; deep in a hole, the lead
// the bullet smeared there.
fn bore_surface(s_in: Surf, wall: vec4<f32>, inner: vec3<f32>, fw: f32) -> Surf {
  var s = s_in;
  s.n = wall.xyz;
  let at = bore_near(s.p);
  let cls = i32(MK[at + 3u].z + 0.5);
  let grain = MK[at + 2u].xyz;
  // (its own occlusion over about its own size: a hole a centimetre across is shaded inside, not black)
  g_ao_reach = 1.2 * max(MK[at].w, MK[at + 1u].w);
  let metal = max(s.f0.r, max(s.f0.g, s.f0.b)) > 0.3 || cls == 6 || cls == 7;
  if (metal) {
    s.alb = vec3<f32>(0.0);
    s.f0 = vec3<f32>(0.56, 0.57, 0.58) * (0.9 + 0.2 * gnoise(s.p * 2000.0));
    s.rough = 0.35;
    return s;
  }
  let fr = fresh(cls, s.p, grain, inner, fw);
  s.alb = min(fr.rgb, vec3<f32>(0.9));
  s.rough = fr.w;
  s.f0 = vec3<f32>(0.03);
  return s;
}
