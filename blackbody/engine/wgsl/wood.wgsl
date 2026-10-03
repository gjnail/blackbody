// Wood as it is (engine/wood.py SPECIES): the stage's wood pattern, solid through the object (so a broken piece, a cut,
// a bullet's hole show the same wood inside).
//
// A trunk grows a ring a year round its pith: pale, open earlywood in spring, dense dark latewood in summer, then the
// next ring starts abruptly. A board is sawn from the trunk off its middle, a little off its axis: so its wide face cuts
// the rings at a slant and shows them as long nested arches ("cathedrals"), its edge as straight lines, its end as arcs
// of rings round a pith off to one side. Rings wander (the trunk is not round), widen and narrow year by year, and bend
// round knots, where branches grew out of the pith, which show as dark ovals with their own rings. Ring-porous woods (oak,
// ash) have rows of large vessels in their earlywood, seen as dark dashes along the grain; oak's broad rays show as
// silvery flecks where the face cuts them square. Fibres streak it all along the grain. Heartwood, near the pith, is
// darker than sapwood. End grain soaks up light: darker, rougher. Plywood shows its plies on its edges; MDF is a felt.
//
// wood_look(kind, q, s, n, fw, seed, shape): kind the material's pattern (materials.PATTERNS: 1 pine, 11-24 the other
// species); q the point in the object's frame (m), s its half size, n the normal there (its frame), fw the pixel's
// footprint (m), seed the object's own number, shape 0 sphere / 1 box / 2 cylinder. It returns the albedo as a multiple
// of the material's colour (rgb) and a roughness offset (w).

struct WoodSp {
  early: vec3<f32>,
  late: vec3<f32>,
  base: vec3<f32>,     // the material's colour (materials.py): what the result is a multiple of
  heart: vec3<f32>,    // heartwood's tint
  ring: f32,           // m
  late_share: f32,
  sharp: f32,
  rays: f32,
  pores: f32,
  knots: f32,          // per metre
  figure: f32,
};

// (wood.py SPECIES, in the same order: tests/test_wood.py keeps them the same)
fn wood_sp(kind: i32) -> WoodSp {
  switch kind {
    case 11: { return WoodSp(vec3<f32>(0.74, 0.6, 0.42), vec3<f32>(0.58, 0.42, 0.25), vec3<f32>(0.66, 0.51, 0.335), vec3<f32>(1.0, 1.0, 1.0), 0.0035, 0.22, 0.85, 0.05, 0.0, 1.5, 0.03); }
    case 12: { return WoodSp(vec3<f32>(0.62, 0.38, 0.2), vec3<f32>(0.38, 0.17, 0.07), vec3<f32>(0.5, 0.275, 0.135), vec3<f32>(0.9, 0.78, 0.68), 0.004, 0.4, 0.95, 0.05, 0.0, 1.0, 0.04); }
    case 13: { return WoodSp(vec3<f32>(0.5, 0.36, 0.2), vec3<f32>(0.38, 0.25, 0.12), vec3<f32>(0.44, 0.305, 0.16), vec3<f32>(0.92, 0.85, 0.78), 0.003, 0.55, 0.3, 0.9, 1.0, 0.6, 0.05); }
    case 14: { return WoodSp(vec3<f32>(0.66, 0.55, 0.38), vec3<f32>(0.5, 0.39, 0.24), vec3<f32>(0.58, 0.47, 0.31), vec3<f32>(0.95, 0.92, 0.86), 0.004, 0.45, 0.3, 0.3, 0.9, 0.4, 0.05); }
    case 15: { return WoodSp(vec3<f32>(0.72, 0.58, 0.39), vec3<f32>(0.64, 0.49, 0.31), vec3<f32>(0.68, 0.535, 0.35), vec3<f32>(0.95, 0.9, 0.85), 0.0035, 0.15, 0.1, 0.2, 0.1, 0.3, 0.25); }
    case 16: { return WoodSp(vec3<f32>(0.73, 0.57, 0.38), vec3<f32>(0.66, 0.5, 0.31), vec3<f32>(0.695, 0.535, 0.345), vec3<f32>(0.92, 0.88, 0.82), 0.003, 0.15, 0.1, 0.1, 0.15, 0.4, 0.15); }
    case 17: { return WoodSp(vec3<f32>(0.26, 0.15, 0.08), vec3<f32>(0.17, 0.09, 0.045), vec3<f32>(0.215, 0.12, 0.062), vec3<f32>(0.85, 0.82, 0.8), 0.0035, 0.3, 0.2, 0.1, 0.4, 0.5, 0.3); }
    case 18: { return WoodSp(vec3<f32>(0.55, 0.28, 0.13), vec3<f32>(0.45, 0.2, 0.08), vec3<f32>(0.5, 0.24, 0.105), vec3<f32>(0.88, 0.82, 0.78), 0.003, 0.2, 0.15, 0.1, 0.1, 0.4, 0.15); }
    case 19: { return WoodSp(vec3<f32>(0.38, 0.16, 0.08), vec3<f32>(0.3, 0.12, 0.055), vec3<f32>(0.34, 0.14, 0.068), vec3<f32>(0.9, 0.88, 0.86), 0.005, 0.15, 0.1, 0.15, 0.35, 0.1, 0.35); }
    case 20: { return WoodSp(vec3<f32>(0.5, 0.3, 0.12), vec3<f32>(0.4, 0.22, 0.08), vec3<f32>(0.45, 0.26, 0.1), vec3<f32>(0.9, 0.85, 0.8), 0.004, 0.3, 0.3, 0.1, 0.4, 0.3, 0.1); }
    case 21: { return WoodSp(vec3<f32>(0.55, 0.27, 0.13), vec3<f32>(0.42, 0.17, 0.07), vec3<f32>(0.485, 0.22, 0.1), vec3<f32>(0.85, 0.75, 0.7), 0.003, 0.3, 0.8, 0.05, 0.0, 1.0, 0.05); }
    case 22: { return WoodSp(vec3<f32>(0.8, 0.7, 0.53), vec3<f32>(0.75, 0.64, 0.47), vec3<f32>(0.775, 0.67, 0.5), vec3<f32>(1.0, 1.0, 1.0), 0.008, 0.1, 0.1, 0.05, 0.3, 0.0, 0.05); }
    case 23: { return WoodSp(vec3<f32>(0.68, 0.52, 0.32), vec3<f32>(0.55, 0.38, 0.2), vec3<f32>(0.615, 0.45, 0.26), vec3<f32>(1.0, 1.0, 1.0), 0.004, 0.25, 0.7, 0.05, 0.1, 0.3, 0.08); }
    case 24: { return WoodSp(vec3<f32>(0.48, 0.33, 0.19), vec3<f32>(0.48, 0.33, 0.19), vec3<f32>(0.48, 0.33, 0.19), vec3<f32>(1.0, 1.0, 1.0), 0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0); }
    default: { return WoodSp(vec3<f32>(0.62, 0.43, 0.24), vec3<f32>(0.43, 0.24, 0.1), vec3<f32>(0.45, 0.29, 0.15), vec3<f32>(0.95, 0.85, 0.72), 0.0045, 0.3, 0.9, 0.05, 0.0, 2.5, 0.05); }
  }
}

fn wood_kind(kind: i32) -> bool { return kind == 1 || (kind >= 11 && kind <= 24); }

fn wood_h(x: f32) -> f32 { return fract(sin(x * 91.3458 + 47.853) * 23421.631); }

// The axis the grain runs along (wood.py grain_axis): a cylinder's axis, a box's longest side (of equal ones y, then z).
fn wood_axis(s: vec3<f32>, shape: i32) -> i32 {
  if (shape == 2) { return 1; }
  if (s.y >= s.x && s.y >= s.z) { return 1; }
  if (s.z >= s.x && s.z >= s.y) { return 2; }
  return 0;
}

fn wood_look(kind: i32, q: vec3<f32>, s: vec3<f32>, n: vec3<f32>, fw: f32, seed: f32, shape: i32) -> vec4<f32> {
  let sp = wood_sp(kind);
  let ax = wood_axis(s, shape);
  // the grain's frame: L along it, a and b across (a the wider of a box's two other sides)
  var L = vec3<f32>(0.0, 1.0, 0.0);
  var A = vec3<f32>(1.0, 0.0, 0.0);
  var B = vec3<f32>(0.0, 0.0, 1.0);
  if (ax == 0) { L = vec3<f32>(1.0, 0.0, 0.0); A = vec3<f32>(0.0, 0.0, 1.0); B = vec3<f32>(0.0, 1.0, 0.0); }
  if (ax == 2) { L = vec3<f32>(0.0, 0.0, 1.0); A = vec3<f32>(1.0, 0.0, 0.0); B = vec3<f32>(0.0, 1.0, 0.0); }
  if (dot(s, B) > dot(s, A)) { let t = A; A = B; B = t; }
  let half_a = dot(s, A);
  let half_b = dot(s, B);
  let z = dot(q, L);
  let h0 = wood_h(seed + 1.3);
  let h1 = wood_h(seed + 7.9);
  let h2 = wood_h(seed + 13.1);
  // the pith: a cylinder's middle; a board's somewhere below its face, off to one side (flat-sawn: arches on its face;
  // some quarter-sawn: straight lines), and the trunk a degree or two off the board's length
  var pith = vec2<f32>(0.0, 0.0);
  var tilt = vec2<f32>(0.0, 0.0);
  if (shape != 2) {
    let depth = mix(0.06, 0.3, h0) + half_b;
    let across = (h1 - 0.5) * 2.0 * half_a;
    let quarter = step(0.75, h2);           // (a quarter of boards are quarter-sawn: the pith off its edge)
    pith = mix(vec2<f32>(across, -depth), vec2<f32>(-(half_a + depth), (h1 - 0.5) * half_b), quarter);
    tilt = vec2<f32>((h2 - 0.5) * 0.04, 0.012 + 0.03 * h1);
  } else {
    pith = vec2<f32>((h1 - 0.5) * 0.15, (h2 - 0.5) * 0.15) * s.x;
    tilt = vec2<f32>((h0 - 0.5) * 0.01, 0.0);
  }
  let c = vec2<f32>(dot(q, A), dot(q, B)) - pith - tilt * z;
  var r = length(c);
  let theta = atan2(c.y, c.x);
  // the rings wander (an out-of-round trunk) and swell and thin along it
  let wob = vec3<f32>(cos(theta) * 1.5, sin(theta) * 1.5, z * 1.2 + seed);
  r += sp.ring * (3.0 * gnoise(wob) + 1.0 * gnoise(wob * 2.7 + vec3<f32>(4.1)));
  // knots: branches out of the pith at a few heights, each pushing the rings round it
  var knot = 0.0;
  var knot_ring = 0.0;
  let n_k = i32(clamp(sp.knots * 2.0 * dot(s, L), 0.0, 6.0) + 0.5);
  for (var k = 0; k < n_k; k++) {
    let hk = wood_h(seed * 3.1 + f32(k) * 17.7);
    let zk = (hk - 0.5) * 2.0 * dot(s, L);
    let tk = wood_h(seed * 5.7 + f32(k) * 9.1) * 6.2831853;
    let dir = vec2<f32>(cos(tk), sin(tk));
    // the branch: up from the pith at an angle, as it grew; its radius growing as it goes out
    let along_b = dot(c, dir);
    if (along_b < 0.0) { continue; }
    let zc = zk + along_b * 0.35;
    let off = vec2<f32>(dot(c, vec2<f32>(-dir.y, dir.x)), (z - zc));
    let rk = 0.004 + 0.012 * wood_h(seed + f32(k) * 3.3) * clamp(along_b / 0.15, 0.2, 1.0);
    let dk = length(off);
    knot = max(knot, 1.0 - smoothstep(rk * 0.85, rk * 1.05, dk));
    knot_ring = max(knot_ring, (1.0 - smoothstep(rk, rk * 3.5, dk)));
    r += sp.ring * 3.0 * exp(-dk * dk / (rk * rk * 4.0));
    if (dk < rk) { r = mix(r, dk * 3.0, 0.9); }      // (inside: its own rings, round the branch)
  }
  // each year's ring its own width
  var t = r / sp.ring;
  t += 1.1 * gnoise(vec3<f32>(t * 0.15, seed, 0.5)) + 0.35 * gnoise(vec3<f32>(t * 0.6, seed, 2.5));
  let f = fract(t);
  // earlywood to latewood: abruptly in conifers, gradually in hardwoods; then the next ring starts at once
  let lw = max(sp.late_share, 0.02);
  let soft = mix(0.35, 0.04, sp.sharp);
  var late = smoothstep(1.0 - lw - soft, 1.0 - lw + soft * 0.5, f) * (1.0 - smoothstep(0.985, 1.0, f));
  // (rings narrower than a couple of pixels fade to their mean: as wide as they show on this face, where the face cuts
  // them at a slant they are wide bands)
  let radial = normalize(A * c.x + B * c.y + vec3<f32>(1e-6, 0.0, 0.0));
  let slant = length(radial - n * dot(n, radial));
  let band = sp.ring / max(slant, 0.03);
  let rings_seen = 1.0 - smoothstep(0.25 * band, 0.6 * band, fw);
  late = mix(lw * 0.8, late, rings_seen);
  var col = mix(sp.early, sp.late, late);
  // heartwood: darker within the inner part of the trunk
  let heart = 1.0 - smoothstep(0.08, 0.14, length(c));
  col *= mix(vec3<f32>(1.0), sp.heart, heart);
  // fibres along the grain; figure (curl) as a ripple in their sheen
  let fine = 1.0 - smoothstep(0.0002, 0.0008, fw);
  let fib = gnoise(vec3<f32>(c * 2500.0, z * 25.0)) * fine;
  let streak = gnoise(vec3<f32>(c * 300.0, z * 4.0)) * (1.0 - smoothstep(0.0015, 0.005, fw));
  col *= 1.0 + 0.06 * fib + 0.06 * streak;
  var rough = 0.0;
  if (sp.figure > 0.0) {
    let curl = sin(z * 600.0 + 3.0 * gnoise(vec3<f32>(c * 40.0, z * 5.0))) * sp.figure;
    col *= 1.0 + 0.08 * curl;
    rough -= 0.1 * curl;
  }
  // ring-porous: rows of open vessels in the earlywood, dark dashes along the grain
  if (sp.pores > 0.0) {
    let pv = gnoise(vec3<f32>(c * 2200.0, z * 90.0));
    let pore = smoothstep(0.35, 0.6, pv) * (1.0 - smoothstep(0.0, 0.3, f)) * sp.pores * fine;
    col *= 1.0 - 0.55 * pore;
  }
  // rays: thin radial plates, silvery flecks where a face cuts them square (oak's quarter-sawn figure)
  if (sp.rays > 0.1) {
    let rq = vec3<f32>(theta * length(c) * 350.0, r * 25.0, z * 40.0);
    let ray = smoothstep(0.45, 0.65, gnoise(rq)) * sp.rays;
    let square = abs(dot(normalize(vec3<f32>(c.x, c.y, 0.001)), vec3<f32>(dot(n, A), dot(n, B), 0.0)));
    col = mix(col, sp.early * 1.25, ray * (1.0 - square) * 0.6 * fine);
  }
  // knots: dark and resinous, with a darker ring round them
  col = mix(col, sp.late * 0.45, knot * 0.9);
  col *= 1.0 - 0.12 * knot_ring * (1.0 - knot);
  rough -= 0.15 * knot;
  // end grain: open fibres end on, darker and rougher
  let endg = smoothstep(0.6, 0.9, abs(dot(n, L)));
  col *= mix(1.0, 0.72, endg);
  rough += 0.2 * endg;
  // plywood: its edges show its plies, light and dark, glue lines between
  if (kind == 23 && shape == 1 && abs(dot(n, B)) < 0.5) {
    let ply = dot(q, B) / max(2.0 * half_b, 1e-4) * 5.0;
    let layer = floor(ply);
    let glue = 1.0 - smoothstep(0.0, 0.06, abs(fract(ply) - 0.5) - 0.44);
    col = mix(col * select(1.0, 0.82, i32(layer) % 2 == 0), vec3<f32>(0.25, 0.18, 0.1), glue * 0.7);
  }
  // MDF: a felt of fibres: a fine speckle, no rings
  if (kind == 24) {
    col = sp.base * (1.0 + 0.12 * gnoise(q * 1500.0) * fine + 0.05 * gnoise(q * 80.0));
    rough = 0.05 + 0.15 * endg;
  }
  // (rings narrower than a pixel fade to their mean: the shader's own AA, as the stage's patterns do)
  return vec4<f32>(col / max(sp.base, vec3<f32>(1e-3)), rough);
}
