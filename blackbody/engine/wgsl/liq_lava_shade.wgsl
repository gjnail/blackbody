// Lava and other molten liquids: how their surface looks. The march (liq_march.wgsl) only finds the
// surface: where a ray meets a molten liquid it leaves the point, its normal and the ray in a buffer
// (liq_lava.wgsl), and this pass shades those pixels over what it wrote. (Inside the march, this much
// shading made the march's compile several times longer.)
//
// A molten surface is opaque. Its light is its own glow, from its temperature, plus what it reflects:
// the black glassy or rough crust that forms on it as it cools.
//
// Temperatures. The simulation gives every particle a heat (1 as poured, falling as it loses heat to the
// air and the ground). Just under the surface that is the melt's own temperature, T_core. The skin on top
// cools far faster than the simulation's cells can show, so its temperature comes from how long it has been
// out in the air (the skin's age, carried along with the crust coordinate: liq_crust_adv.wgsl) through the
// look's Crust forms in: fresh melt glows yellow-orange, a few seconds later its skin is a dull red and then
// black. Light from every temperature is a blackbody's (a table of its colour and luminance relative to
// 1300 K), so the glow dims and reddens as fast as real lava's does: at 1100 K a thirtieth of the brightness
// at 1300 K, at 900 K invisible beside it.
//
// The crust. A pattern of plates and cracks laid out in the crust coordinate, a field on the simulation grid
// carried along by the flow, so plates ride it; each batch of crust starts undistorted where it formed, and
// how the flow has stretched or squeezed it since decides what its cracks do. Each crack is the border
// between two plates (Voronoi cells), measured on the surface (the border's distance in the coordinate times
// the stretch across it). Plates stay about Crust plate size on the surface: where the crust has been pulled
// out, two finer levels of fracture come in. A crack is an old one, a way down to the melt through crust only
// centimetres thick and so nearly as hot (brighter in places along it than others), plus, where the surface
// is being pulled apart across it now, a strip of bared melt at the core temperature, as wide as the surface
// stretches in Crust forms in. Squeezed hard a crack closes to a hairline and the crust wrinkles into ropes
// (pahoehoe): a bumpy noise laid out in the same coordinate, which the squeeze folds into ridges across the
// flow. Next to a crack the crust is thinner and glows a dull red, and batches of crust formed at different
// times meet along seams, cracks too. Each plate's crust is a little lighter or darker than the next.
//
// The relief of plates, cracks and ropes bends the surface normal (analytically, from the gradient of the
// pattern), so the crust catches the key light and the sky as a rough black surface would; while it is still
// molten the surface is glossy.

//!include common.wgsl
//!include noise.wgsl
//!include colour.wgsl

struct Params {
  g2w: mat4x4<f32>,     // grid cells -> world
  vp: mat4x4<f32>,      // world -> clip
  n: vec4<f32>,         // grid dims, metres per cell
  nf: vec4<f32>,        // surface grid dims, surface cells per grid cell
  res: vec4<f32>,       // width, height, footage present (1/0), input transform
  org: vec4<f32>,       // grid corner (fire-local m)
  scene: vec4<f32>,     // camera position (world), footage in reflections (0..1)
  fwd: vec4<f32>,       // camera forward (world), footage gain
  sky: vec4<f32>,       // sky / ambient light (rgb)
  sun: vec4<f32>,       // key light (rgb), apparent sun radius (rad)
  lg: vec4<f32>,        // toward the key light (grid)
  envp: vec4<f32>,      // HDRI on (1/0), rotation (rad), strength
  fit: vec4<f32>,       // footage fit scale (x, y)
  back: vec4<f32>,      // background colour without footage (linear rgb)
  lava: array<vec4<f32>, 8>,
  lava_bb: array<vec4<f32>, 32>,
};

@group(0) @binding(0) var surf_t: texture_3d<f32>;
@group(0) @binding(1) var heat_t: texture_3d<f32>;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var crust_t: texture_3d<f32>;   // on the grid: crust coordinate less place (cells), skin age (s)
@group(0) @binding(4) var<storage, read> gbuf: array<vec4<f32>>;      // the march's molten hits (liq_lava.wgsl)
@group(0) @binding(5) var out_beauty: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var out_emit: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var plate: texture_2d<f32>;
@group(0) @binding(8) var env_t: texture_2d<f32>;
@group(0) @binding(9) var rep: sampler;
@group(1) @binding(0) var<uniform> U: Params;

//!include liq_lava_common.wgsl

const PI: f32 = 3.14159265;

struct LavaShade {
  col: vec3<f32>,
  spec: vec3<f32>,
};

fn surf_v(p: vec3<f32>) -> vec3<f32> { return textureSampleLevel(surf_t, lin, p / U.n.xyz, 0.0).yzw; }

fn to_world_dir(d: vec3<f32>) -> vec3<f32> { return normalize((U.g2w * vec4<f32>(d, 0.0)).xyz); }

// The footage at a screen position (or the background without footage), as the march shows it.
fn backdrop(uv: vec2<f32>) -> vec3<f32> {
  if (U.res.z > 0.5) {
    let q = (uv - vec2<f32>(0.5)) * U.fit.xy + vec2<f32>(0.5);
    let puv = clamp(vec2<f32>(1.0) - abs(vec2<f32>(1.0) - abs(q)), vec2<f32>(0.0), vec2<f32>(1.0));
    return input_transform(textureSampleLevel(plate, lin, puv, 0.0).rgb, i32(U.res.w)) * U.fwd.w;
  }
  return U.back.rgb;
}

fn sky(d: vec3<f32>) -> vec3<f32> {
  if (U.envp.x > 0.5) {
    let c = cos(U.envp.y);
    let s = sin(U.envp.y);
    let r = vec3<f32>(c * d.x - s * d.z, d.y, s * d.x + c * d.z);
    let u = atan2(r.x, -r.z) * (0.5 / PI) + 0.5;
    let v = acos(clamp(r.y, -1.0, 1.0)) / PI;
    return textureSampleLevel(env_t, rep, vec2<f32>(u, v), 0.0).rgb * U.envp.z;
  }
  let s = U.sky.rgb;
  let above = mix(s * 1.35, s * 0.9, clamp(d.y, 0.0, 1.0));
  return mix(s * 0.3, above, smoothstep(-0.06, 0.04, d.y));
}

// Environment seen in a reflection (world direction d): the sky, and the footage in front of the camera.
fn env(d: vec3<f32>) -> vec3<f32> {
  var col = sky(d);
  let a = U.scene.w;
  if (a > 0.0 && U.res.z > 0.5 && dot(d, U.fwd.xyz) > 0.15) {
    let c = U.vp * vec4<f32>(U.scene.xyz + d * 1000.0, 1.0);
    if (c.w > 1e-4) {
      let uv = vec2<f32>(c.x / c.w * 0.5 + 0.5, 0.5 - c.y / c.w * 0.5);
      let e = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
      col = mix(col, backdrop(uv), a * smoothstep(-0.03, 0.02, e));
    }
  }
  return col;
}

// The crust field at grid point p (liq_crust_adv.wgsl): the crust coordinate less the place (cells), the skin's
// age (s).
fn crust_samp(p: vec3<f32>) -> vec4<f32> {
  let n = vec3<i32>(textureDimensions(crust_t));
  let q = p - vec3<f32>(0.5);
  let b = vec3<i32>(floor(q));
  let f = q - floor(q);
  var s = vec4<f32>(0.0);
  for (var o = 0; o < 8; o++) {
    let oo = vec3<i32>(o & 1, (o >> 1) & 1, o >> 2);
    let c = clamp(b + oo, vec3<i32>(0), n - vec3<i32>(1));
    let wv = mix(vec3<f32>(1.0) - f, f, vec3<f32>(oo));
    s += wv.x * wv.y * wv.z * textureLoad(crust_t, c, 0);
  }
  return s;
}

fn crust_d(p: vec3<f32>) -> vec4<f32> { return vec4<f32>(crust_samp(p).xyz, 1.0); }

fn crust_age(p: vec3<f32>) -> vec2<f32> { return vec2<f32>(crust_samp(p).w, 1.0); }

// The melt's heat at grid point p, leaving out the surface grid's empty nodes.
fn lava_heat(p: vec3<f32>) -> f32 {
  let h = textureSampleLevel(heat_t, lin, p / U.n.xyz, 0.0);
  if (h.z > 0.02) { return clamp(h.y / h.z, 0.0, 1.0); }
  return clamp(h.x, 0.0, 1.0);
}

// Voronoi cells at q: distance to the nearest border (cell units, positive inside), the border's
// normal (unit, from the nearest cell toward its neighbour) and a hash of the nearest cell.
struct Cells {
  e: f32,
  m: vec3<f32>,
  id: u32,
};

fn cell_point(c: vec3<f32>) -> vec3<f32> {
  let h = vec3<f32>(pcg3d(vec3<u32>(vec3<i32>(c) + vec3<i32>(1048576)))) * (1.0 / 4294967296.0);
  return c + vec3<f32>(0.5) + (h - vec3<f32>(0.5)) * 0.9;
}

fn lava_cells(q: vec3<f32>) -> Cells {
  let b = floor(q);
  var d1 = 1.0e9;
  var p1 = vec3<f32>(0.0);
  var c1 = vec3<f32>(0.0);
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let c = b + vec3<f32>(f32(x), f32(y), f32(z));
        let fp = cell_point(c);
        let d = dot(q - fp, q - fp);
        if (d < d1) { d1 = d; p1 = fp; c1 = c; }
      }
    }
  }
  // the nearest border: the closest bisector plane between the nearest point and its neighbours'
  var e = 1.0e9;
  var m = vec3<f32>(1.0, 0.0, 0.0);
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let c = c1 + vec3<f32>(f32(x), f32(y), f32(z));
        if (x == 0 && y == 0 && z == 0) { continue; }
        let fp = cell_point(c);
        let dd = fp - p1;
        let l = length(dd);
        if (l < 1e-5) { continue; }
        let mm = dd / l;
        let dist = dot(0.5 * (p1 + fp) - q, mm);
        if (dist < e) { e = dist; m = mm; }
      }
    }
  }
  let idh = pcg3d(vec3<u32>(vec3<i32>(c1) + vec3<i32>(7919)));
  return Cells(e, m, idh.x);
}

struct LavaFrame {
  t1: vec3<f32>,      // tangents of the surface (grid, unit)
  t2: vec3<f32>,
  a: vec3<f32>,       // the crust coordinate's change per unit step along t1, t2
  b: vec3<f32>,
  gi: mat2x2<f32>,    // inverse of the Gram matrix of a, b
};

// Across a border with rest-space normal m: the world stretch (world length per rest length) and the
// unit world direction across it (along t1, t2).
fn lava_across(F: LavaFrame, m: vec3<f32>) -> vec3<f32> {
  let r = vec2<f32>(dot(F.a, m), dot(F.b, m));
  let ab = F.gi * r;                       // world step (along t1, t2) whose image is closest to m
  let proj2 = max(dot(r, ab), 1e-6);       // squared length of m projected onto the rest tangent plane
  let l = max(length(ab), 1e-6);
  return vec3<f32>(l / sqrt(proj2), ab / l);
}

fn ggx_spec(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, rough: f32, size: f32, f0: f32) -> f32 {
  let nl = dot(n, l);
  if (nl <= 0.0) { return 0.0; }
  let nv = max(dot(n, v), 1e-4);
  let hv = normalize(l + v);
  let nh = max(dot(n, hv), 0.0);
  let a2 = rough * rough * rough * rough + size * size;
  let dd = nh * nh * (a2 - 1.0) + 1.0;
  let D = a2 / (PI * dd * dd);
  let vis = 0.5 / (nl * sqrt(nv * nv * (1.0 - a2) + a2) + nv * sqrt(nl * nl * (1.0 - a2) + a2));
  let fh = f0 + (1.0 - f0) * pow(1.0 - max(dot(v, hv), 0.0), 5.0);
  return D * vis * fh * nl;
}

// Pits and bumps of a vesicular skin at q (cell units): 1 at the bottom of a pit .. 0 between them, and its
// gradient in q. The pits are the nearest feature points of a jittered lattice, each of its own size.
fn lava_pits(q: vec3<f32>) -> vec4<f32> {
  let b = floor(q);
  var best = vec4<f32>(0.0);
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let c = b + vec3<f32>(f32(x), f32(y), f32(z));
        let h = vec3<f32>(pcg3d(vec3<u32>(vec3<i32>(c) + vec3<i32>(52361)))) * (1.0 / 4294967296.0);
        let fp = c + h;
        let r = 0.18 + 0.32 * fract(h.x * 7.31 + h.y);   // this pit's radius
        let dv = q - fp;
        let d = length(dv);
        if (d < r) {
          let t = 1.0 - d / r;
          let v = t * t;
          if (v > best.x) { best = vec4<f32>(v, -2.0 * t / r * dv / max(d, 1e-5)); }
        }
      }
    }
  }
  return best;
}

// Gradient noise smeared along dX (in the noise's cells; its length the streak's): the skin's streaks, drawn
// out along the way the melt moves under it (a line integral convolution, taps no more than about half a cell
// apart so the streak stays one streak). Value (about unit spread) and gradient.
fn lava_streak(q: vec3<f32>, dX: vec3<f32>, taps: i32) -> vec4<f32> {
  var s = vec4<f32>(0.0);
  var ws = 0.0;
  for (var i = 0; i < taps; i++) {
    let t = (f32(i) + 0.5) / f32(taps) - 0.5;
    let w = 1.0 - 3.0 * t * t;
    s += gnoise_d(q + dX * t) * w;
    ws += w;
  }
  // (averaging along a line a few cells long takes the contrast down by about the root of its length)
  return s * (sqrt(1.0 + 0.7 * length(dX)) / ws);
}

// Shade a molten surface at grid point p hit along rd, normal nrm; fp: a pixel's footprint there (m).
fn lava_shade(p: vec3<f32>, rd: vec3<f32>, nrm: vec3<f32>, lg: vec3<f32>, fp: f32) -> LavaShade {
  let h = U.n.w;
  let s = max(U.lava[1].x, 1e-3);                      // crust plate size / spacing of tears (m)
  let Ta = U.lava[0].w;
  let Te = U.lava[0].z;
  let cr = clamp(U.lava[1].y, 0.0, 1.0);               // crust
  let tskin = max(U.lava[3].x, 1e-3);                  // crust forms in (s)
  let plates = U.lava[1].w > 0.5;                      // crust kind: 0 a skin (pahoehoe), 1 plates (a lake, a channel)

  // temperatures: the melt under the skin, and the skin, by how long it has been out in the air
  let hc = max(lava_heat(p - nrm * 1.5), lava_heat(p - nrm * 0.4));
  let d0 = crust_d(p);
  let have = U.lava[3].z > 0.5 && d0.w > 0.05;
  var tau = lava_tau_from_heat(lava_heat(p - nrm * 0.25));
  if (have) {
    // (a little wider than a cell: streams of particles a little more or less out in the air streak it)
    var t1s = cross(nrm, vec3<f32>(0.0, 0.0, 1.0));
    if (dot(t1s, t1s) < 1e-4) { t1s = cross(nrm, vec3<f32>(1.0, 0.0, 0.0)); }
    t1s = normalize(t1s) * 2.5;
    let t2s = cross(nrm, t1s);
    tau = 0.4 * crust_age(p).x + 0.15 * (crust_age(p + t1s).x + crust_age(p - t1s).x + crust_age(p + t2s).x
                                         + crust_age(p - t2s).x);
  }
  let LT = lava_temps(hc, tau);
  let Tc = LT.Tc;
  let Ts = LT.Ts;
  let young = LT.young;

  // the tangent frame, and how the crust coordinate is stretched on it (the flow's squeeze since)
  var t1 = cross(nrm, vec3<f32>(0.0, 0.0, 1.0));
  if (dot(t1, t1) < 1e-4) { t1 = cross(nrm, vec3<f32>(1.0, 0.0, 0.0)); }
  t1 = normalize(t1);
  let t2 = cross(nrm, t1);
  var X = U.org.xyz + p * h;
  var a = t1;
  var b = t2;
  if (have) {
    // the coordinate a little evened out over its neighbourhood (what is left of the melt's churning)
    let ds = 1.5;
    var dsm = 0.2 * d0.xyz;
    for (var j = 0; j < 8; j++) {
      let an = f32(j) * 0.7853982;
      let o = (t1 * cos(an) + t2 * sin(an)) * select(ds * 2.0, ds, (j & 1) == 0);
      dsm += 0.1 * crust_d(p + o).xyz;
    }
    X = U.org.xyz + (p + dsm) * h;
    let dl = 1.5;   // cells: wide enough to smooth out what is left of the particles' jostling
    let ga = crust_d(p + t1 * dl).xyz - crust_d(p - t1 * dl).xyz;
    let gb = crust_d(p + t2 * dl).xyz - crust_d(p - t2 * dl).xyz;
    a = t1 + ga / (2.0 * dl);
    b = t2 + gb / (2.0 * dl);
  }
  let G = mat2x2<f32>(vec2<f32>(dot(a, a), dot(a, b)), vec2<f32>(dot(a, b), dot(b, b)));
  let det = max(G[0][0] * G[1][1] - G[0][1] * G[0][1], 1e-8);
  let gi = mat2x2<f32>(vec2<f32>(G[1][1], -G[0][1]), vec2<f32>(-G[0][1], G[0][0])) * (1.0 / det);
  let F = LavaFrame(t1, t2, a, b, gi);
  // principal squeeze: the smallest world length per rest length (1 unsqueezed, below 1 squeezed)
  let tr = G[0][0] + G[1][1];
  let disc = sqrt(max(0.25 * tr * tr - det, 0.0));
  let lam_min = inverseSqrt(max(0.5 * tr + disc, 1e-8));
  let lam_mean = pow(max(det, 1e-8), -0.25);            // surface length per crust length, on average

  // how fast the surface is being pulled apart now (1/s), along any tangent direction: a crack opens as fast
  // as the surface stretches across it, and the melt it bares stays bright until it skins over, so the
  // glowing strip down a crack is that rate times Crust forms in
  let dv = 3.0;
  let v0 = surf_v(p);
  let g1 = (surf_v(p + t1 * dv) - surf_v(p - t1 * dv)) / (2.0 * dv * h);
  let g2 = (surf_v(p + t2 * dv) - surf_v(p - t2 * dv)) / (2.0 * dv * h);
  let e11 = dot(g1, t1);
  let e22 = dot(g2, t2);
  let e12 = 0.5 * (dot(g1, t2) + dot(g2, t1));
  let div = e11 + e22;

  let q0 = X / s;
  let warp = gnoise_d(q0 * 0.55 + vec3<f32>(3.1, 7.7, 1.3)).yzw * select(0.4, 0.22, plates)
           + gnoise_d(q0 * 2.3 + vec3<f32>(9.4, 1.2, 5.5)).yzw * select(0.09, 0.0, plates);   // jagged tears
  let aa = fp / s;                                       // a pixel, in plates
  var T = Ts;
  var cmax = 0.0;                                        // how much of the pixel is crack (bared or skinned melt)
  var near = 1.0e3;                                      // distance to the nearest crack's edge (plates)
  var gX = vec3<f32>(0.0);                               // gradient of the height in the crust coordinate (m / m)
  var tone = 1.0;
  var wlight = 0.0;                                      // the cracks' share of the surface (their light on it)
  var foot = 0.0;                                        // a glowing seam along an advancing lobe's foot
  let along = gnoise(X / (0.6 * s) + vec3<f32>(4.2, 0.7, 2.9));
  let Told = mix(Ts, Tc, clamp(0.9 + 0.15 * along, 0.7, 0.98));
  let seam = 1.0 - smoothstep(0.06, 0.16, lam_min);      // crust formed at different times meets here

  if (!plates) {
    // ---- a skin (pahoehoe) --------------------------------------------------------------------------------
    // A thin, glassy skin over the melt, which glows through it where it is thin: thickening with its age,
    // thinner where the flow has pulled it out since it formed or is pulling it now, drawn into streaks along
    // the way the melt moves under it, pocked with little thin pores, thicker in the folds where it is squeezed.
    let vt = v0 - nrm * dot(v0, nrm);
    let sp = length(vt);
    var dirw = select(t1, vt / max(sp, 1e-6), sp > 1e-4);
    let dX = a * dot(dirw, t1) + b * dot(dirw, t2);      // the flow's direction in the crust coordinate
    // stretch marks: long streaks along the flow (25 cm by 5, and 9 cm by 1.6), in the crust so they ride it
    let st = lava_streak(X / (0.45 * s) + vec3<f32>(2.2, 6.1, 0.4), dX * (0.25 / (0.45 * s)), 14);
    let st2 = lava_streak(X / (0.14 * s) + vec3<f32>(5.9, 0.8, 3.6), dX * (0.09 / (0.14 * s)), 12);
    let po = lava_pits(X / 0.035 + vec3<f32>(0.3, 9.1, 4.7));
    let fold = gnoise_d(X / (0.5 * s) + vec3<f32>(1.7, 9.2, 3.3));
    let age = 1.0 - young;                                // 0 fresh .. 1 an old, grown skin
    // the skin's thickness grows as the square root of its age (a skin forms on fresh melt within moments,
    // so even that has one), times how its thickness varies over it: streaks, pores, folds, stretching. Its
    // surface is cooler the thicker it is; a blackbody's light does the rest: young, it glows with dark streaks,
    // then dark with glowing streaks and pores, then dark
    let thick = sqrt((tau + 0.12 * tskin) / tskin);
    // (by how fast the surface is being pulled out now: how far a batch of crust has been stretched since it
    // formed jumps from one batch to the next, which would draw its edges as rings round a vent)
    let pulled = 1.0 / (1.0 + max(div, 0.0) * tskin * 3.0);
    let s01 = smoothstep(-0.6, 0.6, st.x);
    let f01 = smoothstep(-0.6, 0.6, st2.x);
    // the relief, wanted here for the skin's thickness too: wrinkles and ropes (their troughs thinner, glowing
    // longest), the inflating skin's lumps
    let ropes = U.lava[1].z * (0.35 + 0.65 * smoothstep(0.95, 0.45, lam_min)) * age;
    let rq = X / 0.05 + gnoise_d(X / 0.2 + vec3<f32>(9.0, 0.5, 4.2)).yzw * 0.8;
    let rope = lava_streak(rq + vec3<f32>(3.9, 1.1, 6.2), dX * 0.8, 3);
    let wq = X / 0.022 + gnoise_d(X / 0.06 + vec3<f32>(1.3, 4.4, 7.7)).yzw * 0.6;
    let wr = gnoise_d(wq + vec3<f32>(7.7, 2.3, 1.1));
    var m = pulled * (0.45 + 1.1 * s01) * (0.75 + 0.5 * f01) * (1.0 + 0.3 * fold.x * smoothstep(0.95, 0.5, lam_min));
    m *= (1.0 + 0.3 * clamp(wr.x, -1.0, 1.0)) * (1.0 + 0.25 * clamp(rope.x, -1.0, 1.0)) * (1.0 - 0.6 * po.x);
    let veil = 1.0 - exp(-1.5 * thick * m * mix(0.6, 1.3, cr));
    T = Tc - (Tc - Ta) * 0.92 * veil;
    // tears: where the surface is pulled apart across them now, the skin splits along jagged lines and bares
    // the melt (as bright as the core); the skin's torn edges are thin and glow red
    let C = lava_cells(q0 / 2.2 + warp);   // (tears: fewer and bigger than plates)
    let x = lava_across(F, C.m);
    let r = x.y * x.y * e11 + 2.0 * x.y * x.z * e12 + x.z * x.z * e22;
    let open = clamp(r * tskin * 0.5 * x.x / 2.2, 0.0, 0.12) * smoothstep(0.15, 0.4, age);
    let e = C.e * x.x * 2.2;
    let wt = 0.5 * open - 0.022;                          // (a skin stretches a while before it tears)
    let torn = smoothstep(0.0, 0.015, wt);
    // the gap's edge is ragged (the skin tears unevenly), the skin each side thinned and dull red a little way
    let rag = 0.006 * gnoise(X / 0.015 + vec3<f32>(2.7, 5.1, 8.3));
    let c = (1.0 - smoothstep(wt - aa, wt + aa + 0.004, e + rag)) * torn;
    near = e + rag - wt;
    let rim = exp(-max(near, 0.0) / 0.06) * torn;
    T = mix(T, mix(T, Tc, 0.7), rim);
    T = mix(T, Tc - (Tc - Ts) * 0.08, c);
    cmax = c;
    wlight = wt;
    // the foot of an advancing lobe: the front rolls its crust under and splits it along the ground as it
    // goes, so a broken seam of melt glows along the base (where the margin is moving out; a stalled one is
    // dark)
    let hy = p.y * h;                                     // height above the floor of the box (m)
    let nh = vec2<f32>(nrm.x, nrm.z);
    let adv = dot(v0.xz, nh / max(length(nh), 1e-4));
    let brk = smoothstep(-0.15, 0.35, gnoise(X / 0.09 + vec3<f32>(4.1, 2.2, 7.9)))
            * (0.55 + 0.45 * clamp(0.5 + gnoise(X / 0.025 + vec3<f32>(1.9, 6.3, 3.4)), 0.0, 1.0));
    foot = (1.0 - smoothstep(0.025, 0.07, hy + 0.02 * gnoise(X / 0.05 + vec3<f32>(3.3, 0.6, 9.1))))
         * smoothstep(0.9, 0.5, nrm.y) * smoothstep(0.004, 0.035, adv) * brk;
    // (under its crust a lobe's inside stays nearly as hot as it came out: the seam bares that)
    let Tin = Ta + (Te - Ta) * max(hc, 0.9);
    T = mix(T, Tin - (Tin - Ta) * 0.1, foot);
    // relief (m of height per m of crust coordinate): folds and ropes where squeezed, fine wrinkles drawn
    // along the flow, the pits of a vesicular skin, a torn edge standing up a little
    // lumps of a skin that inflates and sags unevenly: the bigger ones are in the surface itself
    // (liq_lava_disp.wgsl), the smallest here; and folds and ropes where squeezed
    let lump = gnoise_d(X / 0.06 + vec3<f32>(8.8, 1.6, 4.0)) * (1.0 - smoothstep(0.4, 1.2, fp / 0.06));
    gX += lump.yzw * (0.005 / 0.06) * (0.4 + 0.6 * age);
    gX += rope.yzw * (ropes * 0.012 / 0.05) * (1.0 - smoothstep(0.4, 1.2, fp / 0.05));
    gX += fold.yzw * (ropes * 0.07 * s / (0.5 * s)) * (1.0 - smoothstep(0.3, 1.0, fp / (0.5 * s)));
    // fine wrinkles across the flow (the skin's surface bunches as the melt under it drags it): short and wavy
    gX += wr.yzw * (0.0045 / 0.022) * (0.4 + 0.6 * age) * (1.0 - smoothstep(0.4, 1.2, fp / 0.022));
    // grit: the rind is rough at every scale down to millimetres, which breaks its glassy sheen into sparkle
    let g1n = gnoise_d(X / 0.009 + vec3<f32>(3.3, 8.8, 1.4));
    let g2n = gnoise_d(X / 0.0035 + vec3<f32>(6.1, 2.7, 9.9));
    gX += g1n.yzw * (0.0012 / 0.009) * (1.0 - smoothstep(0.4, 1.2, fp / 0.009));
    gX += g2n.yzw * (0.0004 / 0.0035) * (1.0 - smoothstep(0.4, 1.2, fp / 0.0035));
    gX -= po.yzw * (0.004 / 0.035) * (1.0 - smoothstep(0.4, 1.2, fp / 0.035));
    let gr = lava_pits(X / 0.011 + vec3<f32>(4.4, 1.9, 8.2));    // the finer grit of a vesicular rind
    gX -= gr.yzw * (0.0011 / 0.011) * (1.0 - smoothstep(0.4, 1.2, fp / 0.011)) * (0.3 + 0.7 * age);
    let edge = select(0.0, 1.0 / 0.05, e > wt && e < wt + 0.05) * torn;
    gX += -C.m * (edge * 0.004 * x.x / s);
    tone = 0.9 + 0.2 * clamp(0.5 + 0.6 * st2.x + 0.3 * st.x, 0.0, 1.0);
  } else {
    // ---- plates (a lava lake, a channel's rafts) ----------------------------------------------------------
    // Plates (level 0) and two finer levels of fracture that come in as the crust is pulled out, so that on the
    // surface plates stay about Crust plate size however far the flow has stretched the crust since it formed.
    // All widths are on the surface, in plates: a border's distance in the coordinate times the stretch across
    // it. A crack is an old one (a way down to the melt through crust only centimetres thick, still nearly as
    // hot) plus, where the surface is being pulled apart across it now, a strip of bared melt. Squeezed hard, a
    // crack closes to a hairline (the crust folds instead).
    let wbase = mix(0.08, 0.03, cr) * mix(1.0, 0.75, 1.0 - young);
    var Cs: array<Cells, 3>;
    var ek: array<f32, 3>;
    var wk: array<f32, 3>;
    var hk: array<f32, 3>;
    var lk: array<f32, 3>;
    let fk = array<f32, 3>(1.0, 2.4, 5.8);
    let vk = array<f32, 3>(1.0, smoothstep(1.3, 2.0, lam_mean), smoothstep(2.8, 4.2, lam_mean));
    for (var k = 0; k < 3; k++) {
      let f = fk[k];
      let C = lava_cells(q0 * f + vec3<f32>(11.3, 5.2, 8.8) * f32(k) + warp * sqrt(f));
      let x = lava_across(F, C.m);
      let r = x.y * x.y * e11 + 2.0 * x.y * x.z * e12 + x.z * x.z * e22;
      let open = clamp(r * tskin * 0.5 * x.x / f, 0.0, 0.25);
      let shut = mix(0.35, 1.0, smoothstep(0.25, 0.5, x.x));   // squeezed shut: a hairline
      Cs[k] = C;
      ek[k] = C.e * x.x / f;
      wk[k] = (wbase * select(0.7, 1.0, k == 0) + 0.5 * open) * vk[k] * shut;
      hk[k] = 0.5 * open * vk[k] * shut;
      lk[k] = x.x;
    }
    for (var k = 2; k >= 0; k--) {
      if (wk[k] <= 1e-4) { continue; }
      let c = 1.0 - smoothstep(wk[k] - aa, wk[k] + aa + 0.004, ek[k]);
      let hot = 1.0 - smoothstep(hk[k] - aa, hk[k] + aa + 0.002, ek[k]);
      T = mix(T, Told, c);
      T = mix(T, Tc, hot);
      cmax = max(cmax, c);
      near = min(near, ek[k] - wk[k]);
    }
    // the crust by a crack is thinner and hotter: a dull red rim (under the cracks drawn above)
    let rim = exp(-max(near, 0.0) / 0.06) * (0.3 + 0.7 * young) * cr * (1.0 - cmax);
    T = mix(T, mix(Ts, Told, 0.75), rim * 0.6);
    T = mix(T, Told, seam);
    wlight = wk[0];
    // relief: plates stand proud of the cracks and are rounded at their edges; squeezed crust ropes up
    let relief = s * (0.02 + 0.05 * cr * (1.0 - young));
    let round_w = 0.07;
    for (var k = 0; k < 3; k++) {
      if (wk[k] <= 1e-4 && k > 0) { continue; }
      let edge = select(0.0, 1.0 / round_w, ek[k] > wk[k] && ek[k] < wk[k] + round_w) * (1.0 - smoothstep(0.3, 0.8, aa * fk[k]));
      gX += -Cs[k].m * (edge * relief * select(0.4, 1.0, k == 0) * vk[k] * lk[k] / s);
    }
    let ropes = U.lava[1].z * smoothstep(0.95, 0.45, lam_min) * (0.3 + 0.7 * cr);
    let lr = 0.45 * s;
    let rn = gnoise_d(X / lr + vec3<f32>(1.7, 9.2, 3.3));
    gX += rn.yzw * (ropes * 0.18 * s / lr) * (1.0 - smoothstep(0.3, 1.0, fp / lr));
    let lm = 0.12 * s;
    let mn = gnoise_d(X / lm + vec3<f32>(5.5, 2.2, 7.1));
    gX += mn.yzw * (0.005 * s / lm) * (1.0 - smoothstep(0.3, 1.0, fp / lm)) * (1.0 - 0.7 * young);
    tone = 0.8 + 0.4 * f32(pcg1(Cs[0].id) & 1023u) / 1023.0;
  }

  // seen at a grazing angle a crack hides behind its edge
  let cv = clamp(-dot(rd, nrm), 0.0, 1.0);
  let hide = mix(1.0, smoothstep(0.0, 0.5, cv), cmax * cr);
  let E = lava_bb(T) * hide;
  // to the world tangent plane: the height's slope along t1 and t2
  let sl = vec2<f32>(dot(gX, a), dot(gX, b));
  let n = normalize(nrm - (t1 * sl.x + t2 * sl.y) * (1.0 - young * 0.7));

  // reflection: a glassy, rough grey rind (mottled: vesicles, glassy and dull patches); still molten, glossier
  let v = -rd;
  let mott = 0.75 + 0.5 * clamp(0.5 + gnoise(X / (0.18 * s) + vec3<f32>(8.1, 3.3, 6.6)), 0.0, 1.0);
  // (a fresh skin is black glass, so what glows through it keeps its colour; it greys to the crust's colour as
  // it cools, cracks finely and dulls)
  let grey = select(1.0, mix(0.3, 1.0, smoothstep(0.15, 0.85, 1.0 - young)), !plates);
  let alb = U.lava[2].rgb * grey * (1.0 - 0.5 * cmax) * tone * mix(1.0, mott, 1.0 - smoothstep(0.3, 1.0, fp / (0.18 * s)));
  // (a skin's rind: glassy in patches, dull in others)
  let patchy = smoothstep(-0.3, 0.3, gnoise(X / 0.25 + vec3<f32>(5.2, 8.4, 1.7)));
  let rough = select(mix(U.lava[2].w, 0.15, smoothstep(0.5, 0.95, young)), mix(mix(0.4, 0.65, patchy), 0.2, young), !plates);
  let f0 = 0.045;
  let nw = to_world_dir(n);
  var col = alb * sky(nw) * (0.55 + 0.45 * nw.y);
  var gl = 0.0;
  let rr = reflect(rd, n);
  let rw = to_world_dir(rr);
  // bared melt (nearly as hot as the melt under it: no skin to speak of) is one smooth glossy surface
  let bare = 1.0 - smoothstep(0.03, 0.15, (Tc - T) / max(Tc - Ta, 1.0));
  if (plates || bare > 0.99) {
    // one rough or glossy surface: molten, or a crust of plates
    gl = ggx_spec(n, v, lg, rough, U.sun.w, f0) * PI * 0.55;
    let blur = smoothstep(0.15, 0.6, rough);
    let fr_r = f0 + (max(1.0 - rough, f0) - f0) * pow(1.0 - clamp(dot(v, n), 0.0, 1.0), 5.0);
    col += mix(env(rw), sky(normalize(mix(rw, nw, 0.5))), blur) * fr_r * mix(1.0, 0.55, blur);
  } else {
    // A cooled skin's rind is glass full of bubbles: millimetre facets (bubble walls, shards) tipped every way
    // among dull grit. Together the facets are a broad, faint sheen (they face every way, and the grit hides
    // most of them at a grazing view); a pixel holds a few dozen, so how much of that sheen it catches varies
    // from one to the next, and now and then one facet sends the sun's glint full at the camera: it glitters.
    // Its broad sheen is weak, so a fresh rind stays near black in the light instead of shining silver.
    let gsz = max(0.003, fp * 0.8);                       // a cell: a pixel, or 3 mm close up
    let hsh = vec3<f32>(pcg3d(vec3<u32>(vec3<i32>(floor(X / gsz)) + vec3<i32>(7777, 3131, 919)))) * (1.0 / 4294967296.0);
    let share = mix(0.08, 0.3, patchy);                  // how much of the rind is glassy facets here
    let nfac = max(gsz / 0.001, 1.0);
    let ex = -log(max(hsh.x, 1e-6));                     // exponential: mean 1
    let heavy = ex * ex * ex / 6.0;                       // mean 1, heavy-tailed (the odd full glint)
    let spark = mix(1.0, heavy, clamp(3.0 / sqrt(nfac * nfac * share), 0.0, 1.0));
    let mask = mix(0.35, 1.0, clamp(dot(n, v) * 2.0, 0.0, 1.0));   // the grit hides the facets at a grazing view
    let sheen = ggx_spec(n, v, lg, 0.78, U.sun.w, f0) * PI;      // facets every way: a broad lobe
    let dull = ggx_spec(n, v, lg, 0.75, U.sun.w, f0) * PI * 0.04;   // (grit: hardly any sheen of its own)
    gl = 0.5 * dull + share * sheen * spark * mask * 0.25;
    let fr_d = f0 + (0.25 - f0) * pow(1.0 - clamp(dot(v, n), 0.0, 1.0), 5.0);
    let sk = sky(normalize(mix(rw, nw, 0.5)));
    col += sk * fr_d * (0.15 + share * spark * mask * 0.8);
    // (toward bared melt, the one smooth surface)
    let yb = bare;
    if (yb > 0.0) {
      let gs = ggx_spec(n, v, lg, rough, U.sun.w, f0) * PI * 0.55;
      gl = mix(gl, gs, yb);
      let blur = smoothstep(0.15, 0.6, rough);
      let fr_r = f0 + (max(1.0 - rough, f0) - f0) * pow(1.0 - clamp(dot(v, n), 0.0, 1.0), 5.0);
      col = mix(col, alb * sky(nw) * (0.55 + 0.45 * nw.y) + mix(env(rw), sky(normalize(mix(rw, nw, 0.5))), blur) * fr_r * mix(1.0, 0.55, blur), yb);
    }
  }
  col += U.sun.rgb * (alb * max(dot(n, lg), 0.0) + vec3<f32>(gl));
  // light from the cracks around falls on the crust between them
  col += alb * lava_bb(mix(Told, Tc, 0.5)) * clamp(2.0 * wlight, 0.0, 1.0) * 0.6;
  let spec = U.sun.rgb * gl;
  let dbg = i32(U.lava[5].y + 0.5);
  if (dbg > 0) {
    // debug views (BLACKBODY_LAVA_DEBUG): 1 young, 2 squeeze, 3 spreading, 4 crack / seam, 5 temperature,
    // 6 core heat and skin age, 7 lobe foot (seam, advance x10 m/s, height / 10 cm), 8 crust coordinate, 11 its change along the tangents
    var g = vec3<f32>(0.0);
    if (dbg == 1) { g = vec3<f32>(young); }
    else if (dbg == 2) { g = vec3<f32>(clamp(lam_min, 0.0, 1.0), clamp(lam_mean / 4.0, 0.0, 1.0), 0.0); }
    else if (dbg == 3) { g = vec3<f32>(clamp(div * tskin, 0.0, 1.0), clamp(-div * tskin, 0.0, 1.0), 0.0); }
    else if (dbg == 4) { g = vec3<f32>(cmax, seam, 0.0); }
    else if (dbg == 5) { g = vec3<f32>(clamp((T - 500.0) / (Te - 500.0), 0.0, 1.0)); }
    else if (dbg == 6) { g = vec3<f32>(hc, clamp(tau / (4.0 * tskin), 0.0, 1.0), 0.0); }
    else if (dbg == 7) {
      let nh = vec2<f32>(nrm.x, nrm.z);
      g = vec3<f32>(foot, clamp(dot(v0.xz, nh / max(length(nh), 1e-4)) * 10.0, 0.0, 1.0), clamp(p.y * h / 0.1, 0.0, 1.0));
    }
    else if (dbg == 8) { g = fract(X / s); }
    else if (dbg == 11) { g = vec3<f32>(length(a) / 4.0, length(b) / 4.0, select(0.0, 1.0, have)); }
    return LavaShade(g, vec3<f32>(0.0));
  }
  return LavaShade(col + E, spec + E * 0.12);
}


@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let w = u32(U.res.x);
  if (id.x >= w || f32(id.y) >= U.res.y) { return; }
  let k = (id.x + id.y * w) * 3u;
  let a = gbuf[k];
  let b = gbuf[k + 1u];
  let c = gbuf[k + 2u];
  if (c.w < 0.5 || c.w > 1.5) { return; }   // molten hits only (kind 1)
  let sh = lava_shade(a.xyz, c.xyz, b.xyz, U.lg.xyz, a.w);
  let keep = b.w;
  let px = vec2<i32>(id.xy);
  textureStore(out_beauty, px, vec4<f32>(sh.col, 1.0) * keep);
  // the glow feeds the bloom, compressed as the march compresses glints
  let sl = max(max(sh.spec.r, sh.spec.g), sh.spec.b);
  let bloom_in = sh.spec * (8.0 * (1.0 - exp(-sl / 8.0)) / max(sl, 1e-6));
  textureStore(out_emit, px, vec4<f32>(bloom_in, 0.0) * keep);
}
