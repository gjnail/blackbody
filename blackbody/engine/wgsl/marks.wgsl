// What bullets leave behind (engine/ballistics.py): the holes and craters themselves, carved out of whatever they are in
// (bores), and the marks round them drawn into the surface's material (marks), on the stage's objects, broken pieces
// and floor alike. The includer declares MK: array<vec4<f32>> (shot_draw.marks_buffer packs it):
//   MK[0]: how many marks, how many bores, where the bores start (in vec4), _
//   a mark, four vec4 from MK[1]: [0] its centre on the surface (fire-local m), its radius (m); [1] the surface's normal
//   there (toward where the bullet came from), its style; [2] the bullet's direction (its way off a glance), a random
//   number; [3] the grain (wood; 0 elsewhere), 1 on the far side (where the bullet came out) + 2 x what it is in (CLASS)
//   a bore, four vec4: [0] a (m), its radius there; [1] b, its radius there; [2] the grain (or 0), a random number;
//   [3] how much longer it is along the grain than across, how ragged its edge is, what it is in + 16 x (1 + the stage's
//   material row of the object it is in, or 0), how many ragged lobes round it
// What it is in (ballistics.CLASS): 0 anything, 1 concrete, 2 stone, 3 brick, 4 plaster, 5 wood, 6 bare metal,
// 7 painted metal, 8 glass and ice, 9 pottery, 10 earth, 11 plastic, rubber, cardboard, foam, cloth.
//
// The bores make the geometry: a hole through a board, a car's door or a pane is a hole (what is behind it shows through,
// light comes through it); a crater has walls and a floor that take the light and shade themselves. Each is a cone from
// a to b, shaped by what it is in: brittle things (concrete, stone, brick, plaster, pottery) break into a crater of flat
// facets with sharp creases between them; glass spalls out of the back in smooth shell-shaped scallops; wood tears
// along its grain, a split whose ends are a brush of fibres torn to different lengths; metal and plastic hole about
// round, ragged. Their walls are drawn as the thing is inside, freshly broken: concrete's cement, its stones and its air
// holes, granite's crystals, brick's fired clay (and its mortar), plaster's chalk, wood's torn pale fibres, bare bright
// metal, glass crushed to a white frost; at the bottom of a crater, the bullet's lead smeared grey.
// The marks are what is round them: concrete's powder and hairline cracks, the grey ring a bullet wipes off on its way
// in, paint chipped off round a hole in a car, the polished dish of a dent, glass's frosted ring and its web of cracks
// (radial cracks that fork, joined by arcs ring by ring), wood's fibres crushed in round the way in and its strips torn
// up along the grain round the way out, a glancing bullet's furrow, and the lead a bullet splashes over a steel plate:
// a smeared disc, fine streaks sprayed out from it and droplets beyond.

// How frosted (crushed, cracked) the clear thing being shaded is here: 0 clear glass .. 1 white frost (the light it lets
// through is cut by as much: stage.wgsl see, lume.wgsl). Set by the marks and bores of glass; surface_at clears it.
var<private> g_frost: f32;
// Where the light the scene's light grids give (the sky's, the sun's through smoke) is looked up for the surface being
// shaded, when not at it (w 1): a bore's wall is inside its object, where the grids have it dark; its light is the
// light at the hole's mouth. surface_at clears it.
var<private> g_light_p: vec4<f32>;

fn mk_hash(x: f32) -> f32 { return fract(sin(x * 127.1 + 311.7) * 43758.5453); }

// A thin line of half-width w (m) at distance d (m), softened over the pixel (and fading as it gets thinner than one):
// 1 on it.
fn mk_line(d: f32, w: f32, fw: f32) -> f32 {
  let ww = max(w, 0.5 * fw);
  return (1.0 - smoothstep(ww, ww + fw, abs(d))) * min(1.0, w / ww + 0.15);
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

// Broken stone's facets, cells a unit across: a random plane through each cell's point, the cells blended so steeply
// that each is all but flat with a crease between it and the next: the height (about -0.6..0.6) at p. (Every cell within
// reach of p is in the blend, its weight falling to nothing at 1.2 cells: no seam where the cells blended change.)
fn mk_facets(p: vec3<f32>) -> f32 {
  let b = floor(p);
  var num = 0.0;
  var den = 0.0;
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let c = b + vec3<f32>(f32(x), f32(y), f32(z));
        let ci = vec3<i32>(c) + vec3<i32>(1048576);
        let h1 = vec3<f32>(pcg3d(vec3<u32>(ci))) * (1.0 / 4294967296.0);
        let h2 = vec3<f32>(pcg3d(vec3<u32>(ci + vec3<i32>(7919)))) * (2.0 / 4294967296.0) - vec3<f32>(1.0);
        let q = p - (c + vec3<f32>(0.5) + 0.3 * (h1 - vec3<f32>(0.5)));
        let t = max(1.0 - dot(q, q) * (1.0 / 1.44), 0.0);
        let t2 = t * t;
        let t4 = t2 * t2;
        let w = t4 * t4;
        num += w * dot(q, normalize(h2 + vec3<f32>(1e-3)));
        den += w;
      }
    }
  }
  return num / max(den, 1e-20);
}

// Fibres side by side, a unit across each: a random length (0..1) for each, blended over a little of its edge.
fn mk_fibres(y: f32, seed: f32) -> f32 {
  let i = floor(y);
  let a = mk_hash(i * 0.731 + seed * 17.0);
  let b = mk_hash((i + 1.0) * 0.731 + seed * 17.0);
  return mix(a, b, smoothstep(0.35, 0.65, y - i));
}

// What a thing is like freshly broken (a bore's wall, a crater's floor): its albedo (rgb), roughness (w), where `inner`
// is its inside colour, at p (m), along its grain g (wood), at a pixel's footprint fw.
fn fresh(cls: i32, p: vec3<f32>, g: vec3<f32>, inner: vec3<f32>, fw: f32) -> vec4<f32> {
  let fine = 1.0 - smoothstep(0.0003, 0.0012, fw);
  if (cls == 1) {
    // concrete: grey cement paste and its sand, its stones (5-20 mm) broken through, each its own colour, and the
    // round air holes it set with
    let c = mk_cells(p * 90.0);
    let stone = 1.0 - smoothstep(0.3, 0.42, c.x);
    let tone = mix(vec3<f32>(0.7, 0.66, 0.6), vec3<f32>(0.33, 0.3, 0.27), fract(c.y * 3.7)) * mix(1.0, 0.7, step(0.75, c.y))
               * mix(vec3<f32>(1.0), vec3<f32>(1.08, 0.97, 0.88), step(0.5, fract(c.y * 11.0)));
    let sand = 0.9 + 0.2 * gnoise(p * 1500.0) * fine + 0.08 * gnoise(p * 300.0);
    var col = mix(inner * 1.08 * sand, tone * (0.92 + 0.12 * gnoise(p * 700.0) * fine), stone * 0.9);
    let air = mk_cells(p * 450.0 + vec3<f32>(3.1));
    col *= 1.0 - 0.65 * (1.0 - smoothstep(0.1, 0.16, air.x)) * step(0.82, air.y) * (1.0 - stone) * fine;
    return vec4<f32>(col, 0.95);
  }
  if (cls == 2) {
    // granite: its crystals, white, grey, black and pink
    let c = mk_cells(p * 700.0);
    let tone = mix(mix(vec3<f32>(0.75, 0.72, 0.7), vec3<f32>(0.12), step(0.6, c.y)), vec3<f32>(0.62, 0.45, 0.4), step(0.88, c.y));
    return vec4<f32>(mix(inner * 1.2, tone, 0.8 * fine + 0.2), 0.8);
  }
  if (cls == 3) {
    // brick: fresh fired clay, brighter and redder than its weathered face, sandy, with a few dark pores
    let pore = mk_cells(p * 500.0);
    let col = inner * (1.08 + 0.15 * gnoise(p * 900.0) * fine + 0.06 * gnoise(p * 150.0));
    return vec4<f32>(col * (1.0 - 0.5 * (1.0 - smoothstep(0.08, 0.14, pore.x)) * step(0.7, pore.y) * fine), 0.95);
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
    return vec4<f32>(min(inner * (1.1 + 0.3 * f), vec3<f32>(0.9)), 0.9);
  }
  if (cls == 8) {
    // glass and ice: crushed to a white frost
    return vec4<f32>(vec3<f32>(0.76, 0.79, 0.79) * (0.88 + 0.24 * gnoise(p * 3000.0) * fine), 0.45);
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
    if (style == 4) { reach = R * 15.0; }
    if (style == 5) { reach = R * select(4.0, 3.2, exit_side); }
    if (style == 6) { reach = R * 8.0; }
    if (style == 7) { reach = R * 1.2; }
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
    let xy = vec2<f32>(x, y);
    let ang = atan2(y, x);
    let seed = C.w * 97.0;
    let cs = vec2<f32>(cos(ang), sin(ang));
    let out_dir = t / max(r, 1e-6);    // (on the surface, away from the mark's centre)

    if (style == 1) {
      // round a crater (its bowl is the bore's, narrowing a little before the face: its edge there about 0.8 R): the
      // powder it threw out, paler, thinning outward in tongues and streaks and patches; the broken edge of the face
      // round it; one or two hairline cracks running out
      let Re = 0.8 * R;
      let tongue = 0.75 + 0.5 * gnoise(vec3<f32>(cs * 2.5, seed)) + 0.2 * gnoise(vec3<f32>(cs * 7.0, seed + 1.0));
      let streaks = smoothstep(-0.15, 0.45, gnoise(vec3<f32>(cs * 16.0, seed + 3.0)) + 0.5 * gnoise(vec3<f32>(xy / R * 3.0, seed + 4.0)));
      let patches = smoothstep(-0.3, 0.3, gnoise(vec3<f32>(xy / R * 2.2, seed + 11.0)));
      let dust = (1.0 - smoothstep(Re, Re * (1.5 + 0.8 * tongue), r)) * smoothstep(Re * 0.8, Re, r)
                 * mix(streaks * patches, 1.0, 1.0 - smoothstep(Re, Re * 1.15, r));
      let speck = 0.45 + 0.55 * gnoise(vec3<f32>(xy / R * 16.0, seed));
      let pale = vec3<f32>(dot(s.alb, vec3<f32>(0.3, 0.5, 0.2)));
      s.alb = mix(s.alb, min(mix(s.alb, pale, 0.4) * 1.3 + vec3<f32>(0.06), vec3<f32>(0.9)), 0.35 * dust * speck);
      s.rough = mix(s.rough, 1.0, dust);
      // the face's broken edge: a narrow band of fresh stuff where it flaked off round the crater
      let flake = (1.0 - smoothstep(Re * (1.0 + 0.1 * tongue), Re * (1.06 + 0.16 * tongue), r)) * smoothstep(Re * 0.9, Re, r);
      s.alb = mix(s.alb, fresh(cls, s.p, D.xyz, inner, fw).rgb, 0.7 * flake);
      var crack = 0.0;
      let K = 1 + i32(mk_hash(seed) * 3.0);
      for (var k = 0; k < K; k++) {
        let a0 = mk_hash(seed + f32(k) * 7.31) * 6.2831853;
        let len = R * (1.4 + 1.6 * mk_hash(seed + f32(k) * 3.7));
        let dir = vec2<f32>(cos(a0), sin(a0));
        let along = dot(xy, dir);
        let wob = 0.06 * R * (gnoise(vec3<f32>(along / R * 4.0, f32(k), seed)) + 0.5 * gnoise(vec3<f32>(along / R * 17.0, f32(k), seed)));
        let across = dot(xy, vec2<f32>(-dir.y, dir.x)) + wob;
        if (along > Re * 0.95 && along < len) {
          let taper = 1.0 - (along - 0.95 * Re) / max(len - 0.95 * Re, 1e-6);
          crack = max(crack, mk_line(across, 0.00006 * taper + 0.00002, fw) * taper);
        }
      }
      s.alb = mix(s.alb, s.alb * 0.45, 0.6 * crack);
    } else if (style == 2) {
      // round a hole: the grey wipe of the bullet's lead and grease round the way in; paint chipped off round it, bare
      // metal under it, and the paint round that crazed; the lip pushed in (out, on the far side)
      let rim = R * (1.0 + 0.12 * gnoise(vec3<f32>(cs * 3.0, seed)));
      let wipe = (1.0 - smoothstep(rim * 1.05, rim * 1.5, r)) * smoothstep(rim * 0.9, rim * 1.0, r) * select(1.0, 0.25, exit_side);
      if (cls == 7) {
        // (the paint flakes off round the hole in chips of its own, a band a few millimetres wide)
        let edge = rim * (1.25 + 0.15 * gnoise(vec3<f32>(cs * 2.5, seed + 5.0)) + 0.07 * gnoise(vec3<f32>(cs * 9.0, seed + 6.0))
                          + 0.06 * gnoise(vec3<f32>(xy / R * 7.0, seed + 7.0)));
        let chip = 1.0 - smoothstep(edge * 0.96, edge, r);
        let sc = 0.5 + 0.5 * gnoise(vec3<f32>(xy / R * 40.0, seed));
        s.alb = mix(s.alb, vec3<f32>(0.0), chip);
        s.f0 = mix(s.f0, vec3<f32>(0.34, 0.345, 0.35) * (0.85 + 0.3 * sc), chip);
        s.rough = mix(s.rough, 0.5 + 0.15 * sc, chip);
        // (primer showing at the chip's edge: a thin grey ring)
        let primer = mk_line(r - edge * 0.97, 0.00025, fw);
        s.alb = mix(s.alb, vec3<f32>(0.45, 0.45, 0.43), 0.6 * primer * (1.0 - chip));
        // the paint beyond crazed in fine cracks running out from the hole
        var craze = 0.0;
        for (var k = 0; k < 7; k++) {
          let a0 = (f32(k) + mk_hash(seed + f32(k) * 1.9)) * 6.2831853 / 7.0;
          let dir = vec2<f32>(cos(a0), sin(a0));
          let along = dot(xy, dir);
          let len = edge * (1.3 + 1.2 * mk_hash(seed + f32(k) * 4.1));
          if (along > edge * 0.9 && along < len) {
            let across = dot(xy, vec2<f32>(-dir.y, dir.x)) + 0.05 * R * gnoise(vec3<f32>(along / R * 6.0, f32(k), seed));
            craze = max(craze, mk_line(across, 0.00004, fw) * (1.0 - (along - edge * 0.9) / (len - edge * 0.9)));
          }
        }
        s.alb = mix(s.alb, s.alb * 0.4, 0.8 * craze);
      }
      s.alb = mix(s.alb, s.alb * 0.3 + vec3<f32>(0.02), 0.85 * wipe);
      s.f0 = mix(s.f0, vec3<f32>(0.04), 0.5 * wipe);
      let lip = smoothstep(rim, rim * 1.05, r) * (1.0 - smoothstep(rim * 1.05, rim * 1.6, r));
      s.n = normalize(s.n + out_dir * select(0.15, -0.35, exit_side) * lip);
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
      s.n = normalize(s.n + out_dir * 0.25 * smoothstep(0.95, 1.05, rr) * (1.0 - smoothstep(1.05, 1.4, rr)));
    } else if (style == 4) {
      // glass, pottery, ice: round the hole a ring crushed to a white frost; radial cracks out from it, some forking,
      // joined ring by ring by arcs from one to the next (a cobweb); each crack a plane through the glass that catches
      // the light along its length
      let rr = r / R;
      let ec = 1.0 + 0.22 * gnoise(vec3<f32>(cs * 3.0, seed + 2.0)) + 0.1 * gnoise(vec3<f32>(cs * 11.0, seed + 4.0));
      let crush = 1.0 - smoothstep(0.95 * ec, 1.25 * ec, rr);
      var crack = 0.0;
      let K = 12 + i32(mk_hash(seed) * 10.0);
      let slot = 6.2831853 / f32(K);
      let a_pos = select(ang, ang + 6.2831853, ang < 0.0);
      let si = i32(floor(a_pos / slot));
      for (var dk = -1; dk <= 1; dk++) {
        let kk = (si + dk + K) % K;
        let hk = mk_hash(seed + f32(kk) * 7.31);
        let a0 = (f32(si + dk) + 0.5 + 0.7 * (hk - 0.5)) * slot;
        let len = R * (2.5 + 12.0 * pow(mk_hash(seed * 1.7 + f32(kk)), 2.0));
        let dir = vec2<f32>(cos(a0), sin(a0));
        let nrm = vec2<f32>(-dir.y, dir.x);
        let along = dot(xy, dir);
        if (along > 0.7 * R && along < len) {
          let bend = 0.05 * (hk - 0.5) * along * along / R;
          let wob = 0.03 * R * gnoise(vec3<f32>(along / R * 1.2, f32(kk) * 3.1, seed)) + 0.008 * R * gnoise(vec3<f32>(along / R * 7.0, f32(kk), seed));
          let fade = 1.0 - smoothstep(0.55 * len, len, along);
          crack = max(crack, mk_line(dot(xy, nrm) - bend - wob, 0.00006, fw) * fade);
        }
        // a fork part-way out
        let fh = mk_hash(seed + f32(kk) * 2.9);
        if (fh > 0.5) {
          let s0 = R * (1.5 + 3.5 * mk_hash(seed + f32(kk) * 4.3));
          let a1 = a0 + (fh - 0.75) * 1.4;
          let o = dir * s0 + nrm * (0.05 * (hk - 0.5) * s0 * s0 / R);
          let d1 = vec2<f32>(cos(a1), sin(a1));
          let al1 = dot(xy - o, d1);
          let len1 = max(len - s0, 0.0) * (0.4 + 0.5 * fh);
          if (al1 > 0.0 && al1 < len1) {
            let ac1 = dot(xy - o, vec2<f32>(-d1.y, d1.x)) - 0.03 * R * gnoise(vec3<f32>(al1 / R * 1.5, f32(kk) + 0.5, seed));
            crack = max(crack, mk_line(ac1, 0.00005, fw) * (1.0 - al1 / len1));
          }
        }
      }
      // the arcs: in the wedge between two radials, a few rings out, each bowed a little outward
      let ha = mk_hash(seed + f32(si) * 7.31);
      let a_s = (f32(si) + 0.5 + 0.7 * (ha - 0.5)) * slot;
      let lo_i = select(si, si - 1, a_pos < a_s);
      let h_lo = mk_hash(seed + f32((lo_i + K) % K) * 7.31);
      let h_hi = mk_hash(seed + f32((lo_i + 1 + K) % K) * 7.31);
      let a_lo = (f32(lo_i) + 0.5 + 0.7 * (h_lo - 0.5)) * slot;
      let a_hi = (f32(lo_i + 1) + 0.5 + 0.7 * (h_hi - 0.5)) * slot;
      let tw = clamp((a_pos - a_lo) / max(a_hi - a_lo, 1e-4), 0.0, 1.0);
      let len_lo = R * (2.5 + 12.0 * pow(mk_hash(seed * 1.7 + f32((lo_i + K) % K)), 2.0));
      let len_hi = R * (2.5 + 12.0 * pow(mk_hash(seed * 1.7 + f32((lo_i + 1 + K) % K)), 2.0));
      for (var j = 0; j < 3; j++) {
        let hj = mk_hash(seed * 3.3 + f32((lo_i + K) % K) * 1.7 + f32(j) * 11.0);
        let rj = R * (1.5 + 1.0 * f32(j) * (1.0 + 0.25 * f32(j)) + 0.5 * hj);
        if (hj < 0.3 || rj > min(len_lo, len_hi) * 0.9) { continue; }
        let rad = rj * (1.0 + 0.05 * sin(3.14159 * tw) * (0.5 + hj));
        crack = max(crack, mk_line(r - rad, 0.00005, fw) * 0.85);
      }
      crack *= smoothstep(0.85, 1.1, rr);
      // the frost: white, scattering most of the light, sparkling with its tiny facets
      let cell = floor(xy / 0.00025);
      let sp = vec2<f32>(mk_hash(cell.x * 1.31 + cell.y * 7.7 + seed), mk_hash(cell.x * 5.3 + cell.y * 2.1 + seed)) - vec2<f32>(0.5);
      s.alb = mix(s.alb, vec3<f32>(0.8, 0.82, 0.82), 0.85 * crush);
      g_frost = max(g_frost, 0.85 * crush);
      s.rough = mix(s.rough, 0.3, crush);
      s.n = normalize(s.n + (ua * sp.x + va * sp.y) * 1.2 * crush);
      // the cracks: bright where their faces catch the light, fainter between
      let glint = 0.4 + 0.6 * smoothstep(-0.2, 0.5, gnoise(vec3<f32>(xy / R * 1.5, seed + 6.0)));
      s.alb = mix(s.alb, vec3<f32>(0.62, 0.66, 0.66), crack * glint);
      g_frost = max(g_frost, 0.5 * crack);
      s.rough = mix(s.rough, 0.2, crack);
      let tang = va * cs.x - ua * cs.y;
      s.n = normalize(s.n + tang * 0.7 * crack * sign(gnoise(vec3<f32>(xy / R * 3.0, seed + 8.0)) + 0.01));
    } else if (style == 5) {
      // wood. Going in: the fibres crushed in round the hole, darker, the bullet's grey wipe on them, short splits
      // along the grain from it. Coming out: strips of the face torn up along the grain round the split, pale and
      // fresh, lifted most toward the hole, a dark gap down each side of each; splits running on from its ends
      let fib = gnoise(vec3<f32>(x / R * 0.9, y / R * 16.0, seed));
      let grained = dot(D.xyz, D.xyz) > 1e-6;   // (MDF has none)
      if (!exit_side) {
        let re = length(vec2<f32>(x / select(1.0, 1.15, grained), y)) * (1.0 + 0.06 * gnoise(vec3<f32>(cs * 6.0, seed)));
        let crushed = (1.0 - smoothstep(R * 1.05, R * 1.3, re)) * smoothstep(R * 0.9, R * 1.0, re);
        s.alb *= 1.0 - 0.2 * crushed;
        s.n = normalize(s.n - out_dir * 0.3 * crushed);
        let wipe = (1.0 - smoothstep(R * 1.0, R * 1.14, re)) * smoothstep(R * 0.9, R * 0.98, re);
        s.alb = mix(s.alb, s.alb * 0.4 + vec3<f32>(0.03), 0.6 * wipe);
        // torn fibre ends standing pale at the hole's ends along the grain
        let brush = (1.0 - smoothstep(R * 0.95, R * (1.2 + 0.3 * mk_fibres(y / 0.0005, seed)), abs(x)))
                    * (1.0 - smoothstep(R * 0.3, R * 0.7, abs(y))) * smoothstep(R * 0.85, R * 0.95, re);
        s.alb = mix(s.alb, fresh(5, s.p, D.xyz, inner, fw).rgb, 0.6 * brush);
      } else if (!grained) {
        // a felt of fibres (MDF): broken out round about, fuzzy, crumbs of it round the edge
        let rr = r / R * (1.0 + 0.15 * gnoise(vec3<f32>(cs * 4.0, seed)) + 0.1 * gnoise(vec3<f32>(xy / R * 9.0, seed + 1.0)));
        let ring = 1.0 - smoothstep(0.75, 1.0, rr);
        let fuzz = 0.75 + 0.35 * gnoise(vec3<f32>(xy / 0.0004, seed + 3.0));
        s.alb = mix(s.alb, min(inner * 1.05 * fuzz, vec3<f32>(0.9)), 0.8 * ring);
        s.rough = mix(s.rough, 1.0, ring);
        s.n = normalize(s.n + (ua * gnoise(vec3<f32>(xy / 0.0007, seed)) + va * gnoise(vec3<f32>(xy / 0.0007, seed + 5.0))) * 0.5 * ring);
      } else {
        // round the split, the face torn up: fresh fibres where strips peeled off it (the strips themselves stand out
        // of it: fray_*), streaked along the grain, ragged at its edge, reaching furthest along the grain
        let W = R * (0.8 + 0.2 * gnoise(vec3<f32>(x / R * 0.8, 0.0, seed + 2.0)));
        let cy = clamp(abs(y) / W, 0.0, 1.0);
        let rx = R * (1.1 + 0.8 * mk_fibres(y / 0.0009, seed)) * (1.0 - cy * cy);
        let scar = (1.0 - smoothstep(0.8 * rx, rx, abs(x))) * (1.0 - smoothstep(0.8, 1.0, cy));
        let streak = 0.6 * mk_fibres(y / 0.00035, seed + 9.0) + 0.4 * mk_fibres(y / 0.0012, seed + 4.0);
        let fr = fresh(5, s.p, D.xyz, inner, fw).rgb;
        s.alb = mix(s.alb, fr * (0.82 + 0.3 * streak), 0.8 * scar);
        s.rough = mix(s.rough, 0.95, scar);
        s.n = normalize(s.n + va * 0.35 * (streak - 0.5) * scar);
      }
      // splits along the grain, up and down from the hole
      var split = 0.0;
      for (var k = 0; k < 2; k++) {
        let side = select(-1.0, 1.0, k == 0);
        let start = select(0.7 * R, 2.2 * R, exit_side);
        let len = start + R * (1.5 + 3.5 * mk_hash(seed + f32(k) * 5.1)) * select(1.0, 1.5, exit_side);
        let off_y = 0.25 * R * (mk_hash(seed + f32(k) * 2.3) - 0.5);
        let ax = side * x;
        if (grained && ax > start && ax < len && mk_hash(seed + f32(k) * 9.7) > 0.25) {
          let w = 0.00012 * (1.0 - (ax - start) / max(len - start, 1e-6)) + 0.00002;
          split = max(split, mk_line(y - off_y - 0.05 * R * gnoise(vec3<f32>(ax / R * 3.0, f32(k), seed)), w, fw));
        }
      }
      s.alb = mix(s.alb, s.alb * 0.25, 0.85 * split);
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
      // the lead a bullet splashes flat on steel: a smeared disc, the paint (a painted plate's) blasted off round it,
      // fine streaks sprayed out from it, droplets beyond them
      let rr = r / R;
      let lead = vec3<f32>(0.55, 0.55, 0.57);
      if (cls == 7) {
        let bare = 1.0 - smoothstep(0.72, 0.8, rr / (1.0 + 0.12 * gnoise(vec3<f32>(cs * 6.0, seed + 1.0))));
        s.alb = mix(s.alb, vec3<f32>(0.0), bare);
        s.f0 = mix(s.f0, vec3<f32>(0.55, 0.56, 0.57), bare);
        s.rough = mix(s.rough, 0.35, bare);
      }
      let disc_r = 0.36 * (1.0 + 0.2 * gnoise(vec3<f32>(cs * 5.0, seed)) + 0.1 * gnoise(vec3<f32>(cs * 14.0, seed + 2.0)));
      let mottle = smoothstep(-0.35, 0.1, gnoise(vec3<f32>(xy / R * 9.0, seed + 3.0)) + 0.6 * (1.0 - rr / disc_r));
      var cover = (1.0 - smoothstep(disc_r * 0.9, disc_r, rr)) * mottle;
      // streaks, one in each of many narrow sectors, each its own length, thin and fading out
      let N = 90.0;
      let sec = 6.2831853 / N;
      let a_pos = select(ang, ang + 6.2831853, ang < 0.0);
      let si = floor(a_pos / sec);
      for (var dk = 0; dk <= 1; dk++) {
        let k = si + f32(dk) - select(0.0, 1.0, fract(a_pos / sec) < 0.5);
        let kw = k - N * floor(k / N);
        let hk = mk_hash(seed + kw * 3.17);
        let ak = (k + 0.5 + 0.6 * (hk - 0.5)) * sec;
        let lk = disc_r + (0.15 + 0.75 * hk * hk) * mk_hash(seed + kw * 5.3);
        let da = a_pos - ak;
        let across = r * sin(da);
        let wk = R * 0.006 * (1.0 - smoothstep(disc_r, lk, rr)) + R * 0.001;
        let on = mk_line(across, wk, fw) * step(disc_r * 0.85, rr) * (1.0 - smoothstep(lk * 0.8, lk, rr)) * step(0.0, cos(da));
        let broken = smoothstep(-0.2, 0.2, gnoise(vec3<f32>(rr * 14.0, kw, seed)));
        cover = max(cover, on * broken * (0.85 - 0.6 * (rr - disc_r) / max(lk - disc_r, 1e-3)) * step(0.3, hk));
      }
      // droplets
      let dc = mk_cells(vec3<f32>(xy / (0.03 * R), seed));
      let drop = (1.0 - smoothstep(0.1, 0.17, dc.x)) * step(0.6 + 0.35 * smoothstep(0.5, 1.1, rr), dc.y)
                 * smoothstep(disc_r, disc_r + 0.15, rr) * (1.0 - smoothstep(0.8, 1.15, rr));
      cover = max(cover, 0.7 * drop);
      let core = 1.0 - smoothstep(0.1, 0.16, rr);
      // (lead smeared on steel is dull and pale, a matte silver-grey on the darker, smoother plate)
      let lv = 0.85 + 0.25 * gnoise(vec3<f32>(xy / R * 25.0, seed));
      s.alb = mix(s.alb, lead * 0.75 * lv, cover);
      s.f0 = mix(s.f0, vec3<f32>(0.2) * lv, cover);
      s.rough = mix(s.rough, 0.7, cover);
      s.alb = mix(s.alb, vec3<f32>(0.25), 0.6 * core);
      s.rough = mix(s.rough, 0.5, core);
      s.n = normalize(s.n + out_dir * 0.3 * smoothstep(0.1, 0.33, rr) * (1.0 - smoothstep(disc_r * 0.85, disc_r, rr)));
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

// Bore j's distance at p (not an exact distance: the steps by it are cut to 0.7 of it).
fn bore_one(p: vec3<f32>, j: u32, first: u32) -> f32 {
  let A = MK[first + 4u * j];
  let B = MK[first + 4u * j + 1u];
  let G = MK[first + 4u * j + 2u];
  let H = MK[first + 4u * j + 3u];
  let ab = B.xyz - A.xyz;
  let len = max(length(ab), 1e-6);
  let u = ab / len;
  var g = G.xyz - u * dot(G.xyz, u);
  if (dot(g, g) < 1e-8) { g = cross(u, select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(u.y) > 0.9)); }
  g = normalize(g);
  let hh = cross(u, g);
  let w = p - A.xyz;
  let wu = dot(w, u);
  let along = wu / len;
  let k = clamp(along, 0.0, 1.0);
  let rv = w - u * wu;
  let x = dot(rv, g);
  let y = dot(rv, hh);
  let cls = bore_cls(H);
  let seed = G.w;
  let scale = max(A.w, B.w);
  if (cls == 5 && H.x > 0.0) {
    // wood, along its grain: a split, its sides nearly straight, its ends a brush of fibres torn to different
    // lengths, longer (and one end more than the other) toward the far side
    let r = mix(A.w, B.w, k);
    let asym = 0.65 + 0.7 * fract(seed * 7.7);
    let side = select(asym, 2.0 - asym, x < 0.0);
    let fib = 0.65 * mk_fibres(y / 0.0007, seed) + 0.35 * mk_fibres(y / 0.0024, seed + 3.0);
    let L = r * (1.0 + H.x * k * side) * max(1.0 + H.y * (0.3 + 0.7 * k) * (fib - 0.35) * 1.6, 0.4);
    let W = r * (1.0 + 0.1 * gnoise(vec3<f32>(x / r * 1.2, k * 2.0, seed * 13.0)));
    var d = (length(vec2<f32>(x / L, y / W)) - 1.0) * min(L, W);
    // its walls fibrous, torn along the grain
    let qq = vec3<f32>(y / (0.07 * scale), wu / (0.07 * scale), x / (0.9 * scale)) + vec3<f32>(seed * 7.0);
    d += H.y * 0.12 * scale * gnoise(qq);
    return 0.7 * max(d, max(-wu, wu - len));
  }
  // round about, ragged by its lobes (and longer along the grain by elong)
  let ang = atan2(y, x);
  let c = cos(ang);
  let lobes = max(H.w, 1.0);
  let q = vec3<f32>(c * lobes / 3.0, sin(ang) * lobes / 3.0, along * 1.5 + seed * 13.0);
  let rag = gnoise(q) + 0.45 * gnoise(q * 2.7 + vec3<f32>(5.2));
  let f = (1.0 + H.x * k * c * c) * max(1.0 + H.y * (0.35 + 0.65 * k) * rag, 0.3);
  var rough_d = 0.0;
  if (H.y > 0.15 && (cls <= 4 || cls == 9 || cls == 10)) {
    // brittle: broken into flat facets with sharp creases between them, a few millimetres across, gritty on them
    rough_d = H.y * scale * (0.2 * mk_facets(p / (0.24 * scale) + vec3<f32>(seed * 7.0)) + 0.22 * mk_facets(p / (0.6 * scale) + vec3<f32>(seed * 3.0))
                          + 0.04 * gnoise(p / (0.05 * scale)));
  } else if (cls == 8) {
    // glass: smooth shell-shaped scallops
    rough_d = H.y * scale * 0.3 * mk_facets(p / (0.55 * scale) + vec3<f32>(seed * 7.0));
  } else if (H.y > 0.2) {
    rough_d = H.y * 0.1 * scale * gnoise(p / (0.4 * scale) + vec3<f32>(seed * 7.0));
  }
  return 0.7 * (sd_cone(p, A.xyz, B.xyz, A.w * f, B.w * f) + rough_d);
}

// The distance from p to the nearest bore (negative inside one), or a large number with none near.
const BORE_FAR: f32 = 0.005;

fn bore_d(p: vec3<f32>) -> f32 {
  var best = 1.0e9;
  let n = u32(MK[0].y + 0.5);
  let first = u32(MK[0].z + 0.5);
  for (var j = 0u; j < n; j++) {
    let A = MK[first + 4u * j];
    let B = MK[first + 4u * j + 1u];
    let H = MK[first + 4u * j + 3u];
    // (a quick bound first: the sphere round it, as big as its raggedest; well outside it, the distance to the sphere
    // will do, a step that cannot reach into the bore; near it, its own, or the sphere would show as a seam in the
    // normals of another bore's wall that it crosses)
    let c = 0.5 * (A.xyz + B.xyz);
    let reach = 0.5 * length(B.xyz - A.xyz) + max(A.w, B.w) * (1.0 + 1.4 * H.x) * (1.0 + 1.5 * H.y);
    let lb = length(p - c) - reach;
    if (lb > best) { continue; }
    if (lb > BORE_FAR) {
      best = lb;
      continue;
    }
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

// A bore's wall, shaded as the thing is freshly broken inside (fresh()): `inner` its inside colour, the surface s_in
// as the object's pattern has it there (brick and its mortar); bare metal bright; glass frosted; at the bottom of a
// crater, the lead the bullet smeared there.
fn bore_surface(s_in: Surf, wall: vec4<f32>, inner: vec3<f32>, fw: f32) -> Surf {
  var s = s_in;
  s.n = wall.xyz;
  let at = bore_near(s.p);
  let A = MK[at];
  let B = MK[at + 1u];
  let G = MK[at + 2u];
  let cls = bore_cls(MK[at + 3u]);
  let ab = B.xyz - A.xyz;
  let k = clamp(dot(s.p - A.xyz, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
  // (its own occlusion over about its own size: a hole a centimetre across is shaded inside, not black; its light
  // from the light grids, as at its mouth nearer this wall)
  g_ao_reach = 0.6 * max(min(A.w, B.w), 0.25 * max(A.w, B.w));
  g_light_p = vec4<f32>(select(B.xyz, A.xyz, k < 0.5), 1.0);
  let metal = max(s.f0.r, max(s.f0.g, s.f0.b)) > 0.3 || cls == 6 || cls == 7;
  if (metal) {
    // torn bright metal, scored along the bullet's way
    let u = ab / max(length(ab), 1e-9);
    let sc = gnoise(vec3<f32>((s.p - u * dot(s.p, u)) * 9000.0)) * 0.5 + gnoise(s.p * 2500.0) * 0.5;
    s.alb = vec3<f32>(0.0);
    s.f0 = vec3<f32>(0.56, 0.57, 0.58) * (0.88 + 0.22 * sc);
    s.rough = 0.3 + 0.1 * sc;
    return s;
  }
  var base = inner;
  if (cls == 3) { base = min(s_in.alb * 1.18 + vec3<f32>(0.02), vec3<f32>(0.9)); }   // (fired clay, or the mortar's grey)
  let fr = fresh(cls, s.p, G.xyz, base, fw);
  s.alb = min(fr.rgb, vec3<f32>(0.9));
  s.rough = fr.w;
  s.f0 = vec3<f32>(0.03);
  if (cls == 8) {
    // crushed glass: white frost, its tiny facets glinting
    g_frost = 0.8;
    let hc = floor(s.p / 0.0003);
    let jit = vec3<f32>(mk_hash(dot(hc, vec3<f32>(1.3, 7.1, 3.7))), mk_hash(dot(hc, vec3<f32>(5.1, 1.7, 9.3))),
                        mk_hash(dot(hc, vec3<f32>(2.9, 4.3, 1.1)))) - vec3<f32>(0.5);
    s.n = normalize(s.n + jit * 0.9);
    s.f0 = vec3<f32>(0.04);
    s.rough = 0.3;
  }
  if (cls <= 4 || cls == 9 || cls == 10) {
    // in the bullet's own pocket at the bottom of a crater (the narrower, less ragged bore: ballistics.py _carve), its
    // lead smeared over the broken stuff, bits of its jacket glinting; at the very bottom of a crater without one
    let pocket = MK[at + 3u].y < 0.3;
    let lead = select(smoothstep(0.9, 1.0, k), smoothstep(0.0, 0.25, k), pocket) * (0.55 + 0.45 * gnoise(s.p * 1500.0));
    s.alb = mix(s.alb, vec3<f32>(0.0), 0.85 * lead);
    s.f0 = mix(s.f0, vec3<f32>(0.42, 0.42, 0.44), 0.85 * lead);
    s.rough = mix(s.rough, 0.45, lead);
  }
  return s;
}

// ---- splinters: what stands out of a bullet's way out of wood -------------------------------------------------------

// A bore's record: what it is in (CLASS), and the stage's material row of the object it is in (-1 none).
fn bore_cls(H: vec4<f32>) -> i32 { return i32(H.z + 0.5) & 15; }
fn bore_row(H: vec4<f32>) -> i32 { return (i32(H.z + 0.5) >> 4) - 1; }

// A hit on a splinter (trace(), surface_at: stage.wgsl; lume.wgsl).
const FRAY: i32 = 320;   // (300, 301: water, lume_water.wgsl)

// The splinters standing out of the back of boards where bullets came out (shot_draw.splinters works them out once a
// frame): MK[0].w exits with splinters; from fray_first(), one vec4 each: (its bore's record, its first splinter's,
// how many, the radius round its bore's far end they reach); then three vec4 a splinter: (hinge, width), (tip,
// thickness), (the way across it, how much it curls: its tip off its straight line, as a share of its length).
fn fray_count() -> u32 { return u32(MK[0].w + 0.5); }
fn fray_first() -> u32 { return u32(MK[0].z + 0.5) + 4u * u32(MK[0].y + 0.5); }

// The distance from p to splinter s (a strip from a little into the board at its hinge to its tip, tapering to a point).
fn sd_splinter(p: vec3<f32>, s: u32) -> f32 {
  let S0 = MK[s];
  let S1 = MK[s + 1u];
  let S2 = MK[s + 2u];
  let ax = S1.xyz - S0.xyz;
  let L = max(length(ax), 1e-6);
  let ad = ax / L;
  let nr = normalize(cross(ad, S2.xyz) + vec3<f32>(1e-9));
  let q = p - S0.xyz;
  let a = dot(q, ad);
  let s01 = clamp(a / L, 0.0, 1.0);
  let hw = 0.5 * S0.w * (1.0 - 0.7 * s01 * s01 * s01);
  let ht = 0.5 * S1.w * (1.0 - 0.5 * s01);
  let bend = S2.w * L * s01 * s01;
  let dd = vec3<f32>(abs(a - 0.5 * L + 0.5 * S1.w) - 0.5 * (L + S1.w), abs(dot(q, S2.xyz)) - hw, abs(dot(q, nr) - bend) - ht);
  return 0.8 * (length(max(dd, vec3<f32>(0.0))) + min(max(dd.x, max(dd.y, dd.z)), 0.0));
}

// The splinters of exit k: their distance at p.
fn fray_one(p: vec3<f32>, k: u32) -> f32 {
  let E = MK[fray_first() + k];
  var best = 1.0e9;
  let s0 = u32(E.y + 0.5);
  for (var i = 0u; i < u32(E.z + 0.5); i++) {
    best = min(best, sd_splinter(p, s0 + 3u * i));
  }
  return best;
}

// The sphere round exit k's splinters: (centre, radius).
fn fray_bound(k: u32) -> vec4<f32> {
  let E = MK[fray_first() + k];
  return vec4<f32>(MK[u32(E.x + 0.5) + 1u].xyz, E.w);
}

// The distance from p to the nearest splinter, or a large number with none near. (Far from an exit, the distance to
// the sphere round its splinters will do; nearer than FRAY_FAR, where shadows and occlusion read the distance as how
// near something is, not just how far a step may go, the splinters' own.)
const FRAY_FAR: f32 = 0.06;

fn fray_d(p: vec3<f32>) -> f32 {
  var best = 1.0e9;
  for (var k = 0u; k < fray_count(); k++) {
    let b = fray_bound(k);
    let lb = length(p - b.xyz) - b.w;
    if (lb > best) { continue; }
    if (lb > FRAY_FAR) {
      best = lb;
      continue;
    }
    best = min(best, fray_one(p, k));
  }
  return best;
}

// The exit whose splinters are nearest p: its bore's record in MK.
fn fray_near(p: vec3<f32>) -> u32 {
  var best = 1.0e9;
  var out = 0u;
  for (var k = 0u; k < fray_count(); k++) {
    let b = fray_bound(k);
    if (length(p - b.xyz) - b.w > best) { continue; }
    let d = fray_one(p, k);
    if (d < best) {
      best = d;
      out = u32(MK[fray_first() + k].x + 0.5);
    }
  }
  return out;
}

// The first splinter along a ray from t0 to tmax: its distance along the ray, or -1.
fn fray_hit(ro: vec3<f32>, rd: vec3<f32>, t0: f32, tmax: f32) -> f32 {
  if (!bores_on() || fray_count() == 0u) { return -1.0; }
  var best = tmax;
  var found = false;
  for (var k = 0u; k < fray_count(); k++) {
    let b = fray_bound(k);
    let oc = ro - b.xyz;
    let bb = dot(oc, rd);
    let disc = bb * bb - (dot(oc, oc) - b.w * b.w);
    if (disc <= 0.0) { continue; }
    let sq = sqrt(disc);
    var t = max(-bb - sq, t0);
    let t_out = min(-bb + sq, best);
    for (var i = 0; i < 64; i++) {
      if (t >= t_out) { break; }
      let d = fray_one(ro + rd * t, k);
      let eps = max(1.0e-5, 0.4 * U.fit.w * t);
      if (d < eps) {
        best = t;
        found = true;
        break;
      }
      t += max(d, 0.5 * eps);
    }
  }
  return select(-1.0, best, found);
}

// The outward normal of the splinters at p.
fn fray_normal(p: vec3<f32>, e: f32) -> vec3<f32> {
  let a = vec3<f32>(1.0, -1.0, -1.0);
  let b = vec3<f32>(-1.0, -1.0, 1.0);
  let c = vec3<f32>(-1.0, 1.0, -1.0);
  let d = vec3<f32>(1.0, 1.0, 1.0);
  let g = a * fray_d(p + a * e) + b * fray_d(p + b * e) + c * fray_d(p + c * e) + d * fray_d(p + d * e);
  let l = length(g);
  return select(vec3<f32>(0.0, 1.0, 0.0), g / l, l > 1e-12);
}

// A splinter's surface: torn pale fibres along it; its back, the board's face it was part of, in the board's colour.
fn fray_surface(s_in: Surf, fw: f32) -> Surf {
  var s = s_in;
  s.n = fray_normal(s.p, max(0.5 * fw, 1.0e-5));
  let at = fray_near(s.p);
  let A = MK[at];
  let B = MK[at + 1u];
  let G = MK[at + 2u];
  let row = clamp(bore_row(MK[at + 3u]), 0, i32(MAT_ROWS) - 1);
  let m = U.mat[row];
  let u = normalize(B.xyz - A.xyz + vec3<f32>(1e-9));
  let fr = fresh(5, s.p, G.xyz, m.e.rgb, fw).rgb;
  let back = smoothstep(0.3, 0.7, dot(s.n, u));
  s.alb = min(mix(fr, m.c.rgb * 0.95, 0.6 * back), vec3<f32>(0.9));
  s.rough = 0.85;
  s.f0 = vec3<f32>(0.04);
  g_ao_reach = 0.003;
  g_light_p = vec4<f32>(B.xyz + u * 0.01, 1.0);
  return s;
}
